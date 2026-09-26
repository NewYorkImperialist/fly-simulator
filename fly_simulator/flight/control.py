"""Hover / slow forward flight controller (PID on attitude with rate damping +
position loop), running at the control rate (default 5 kHz = every 4 physics
steps) and writing per-wing wingbeat parameters.

Frames: the *stroke frame* S = thorax frame rotated by the stroke-plane angle
beta about y (``R_S = R_thorax @ R_y(beta)``). In the hover posture (thorax
pitched nose-up by beta) S is level: x forward (horizontal), y left, z up
(normal of the stroke plane). Attitude errors, rates and torques live in S.

Control effectiveness (measured tethered, ``scripts/demo_flight.py effect``,
218 Hz, 160 deg stroke, aoa 45 deg, rotation phase -0.5 rad; world frame =
S frame at the tether pose; forces uN, torques uN*mm about the COM, per rad):

    input (per wing change)           Fz      Fx      Tx(roll) Ty(pitch) Tz(yaw)
    amplitude, both        (+A)      6.14    0.09     0.0      0.41     0.0
    mean stroke, both      (+phi0)   0.0     2.84     0.0    -10.32     0.0
    amplitude L+ / R-      (+dA)     0.10    0.01     5.18     0.02    -0.17
    stroke-plane tilt L+/R- (+dT)   -2.12    0.03     0.59    -0.29   -12.14
    stroke-plane tilt, both (+T)    -1.73   10.48     0.0      4.02     0.0

so: altitude <- stroke amplitude, pitch <- mean stroke angle (moves the centre of
lift fore/aft), roll <- left/right amplitude difference, yaw <- left/right
stroke-plane tilt (one wing's force tilted forward, the other's backward), forward
force <- symmetric stroke-plane tilt. The 5x5 matrix is inverted once: the
controller asks for a wrench (Fz, Fx, Tx, Ty, Tz) and gets the 5 inputs.

Lateral position goes through the roll attitude (tilting the lift vector),
forward position/velocity through the symmetric stroke-plane tilt (the body keeps
its hover pitch; real flies also pitch nose-down, not modelled).

Biology: the rate (D) terms use the thorax angular velocity, which a real fly
senses with its halteres (Coriolis gyroscopes beating in anti-phase with the
wings, ~1 wingbeat latency, mainly rate feedback to the wing steering muscles);
the slower attitude (P) and position terms correspond to visual feedback
(optomotor / horizon / optic flow). The mapping of torques onto amplitude,
mean-stroke and stroke-plane changes mirrors the steering kinematics of real
flies (Dickinson & Muijres 2016; Muijres et al. 2014).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace

import mujoco as mj
import numpy as np

from .wingbeat import DEG, WingbeatParams, WingParams

# Rows: Fz, Fx, Tx, Ty, Tz. Columns: A, phi0, dA (L+/R-), dT (L+/R-), T.
# Measured per rad (see module doc).
EFFECTIVENESS = np.array([
    [6.14, 0.00, 0.10, -2.12, -1.72],
    [0.09, 2.84, 0.01, 0.03, 10.48],
    [0.00, 0.00, 5.18, 0.59, 0.00],
    [0.41, -10.32, 0.02, -0.29, 4.02],
    [0.00, 0.00, -0.17, -12.14, 0.00],
])
# Wrench of the base wingbeat (per the same measurement): Fz, Fx, Tx, Ty, Tz.
BASE_WRENCH = np.array([10.03, -0.09, 0.0, 0.521, 0.0])


def base_wingbeat() -> WingbeatParams:
    w = WingParams(amplitude=160.0 * DEG, aoa_down=45.0 * DEG, aoa_up=45.0 * DEG,
                   rot_phase=-0.5)
    return WingbeatParams(freq=218.0, left=w, right=replace(w))


@dataclass
class HoverGains:
    att_wn: float = 60.0  # rad/s, attitude loop natural frequency
    att_zeta: float = 0.8
    yaw_wn: float = 30.0
    yaw_zeta: float = 0.9
    z_wn: float = 12.0
    z_zeta: float = 1.0
    z_ki: float = 300.0  # 1/s^3 integral on altitude (lift trim)
    xy_wn: float = 5.0
    xy_zeta: float = 1.0
    xy_ki: float = 10.0  # 1/s^3 integral on horizontal position
    att_ki_ratio: float = 0.2  # attitude integral gain = ratio * wn^3 (torque trims)
    max_roll: float = 25.0 * DEG
    max_accel_xy: float = 3000.0  # mm/s^2 (0.3 g)
    rate_tau: float = 1.0 / 436.0  # s: low-pass on rates/attitude (half a wingbeat)
    # The attitude reference starts at the measured attitude (e.g. a tumbling
    # fly at the jump apex) and turns toward the desired one at most this fast.
    att_slew: float = 10.0  # rad/s
    max_freq: float = 260.0  # Hz, upper wingbeat frequency (lift reserve)


@dataclass
class HoverTarget:
    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 10.0]))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # feed-forward, world
    yaw: float = 0.0
    pitch: float = 0.0  # rad, body pitch relative to the hover posture, + = nose-down


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _att_err(R_ref: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Attitude error vector (rad, ~ rotation vector for small errors) of R
    relative to R_ref, in the frame of R."""
    Re = R_ref.T @ R
    return 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])


def _slew(R_from: np.ndarray, R_to: np.ndarray, max_angle: float) -> np.ndarray:
    """Rotate R_from toward R_to by at most ``max_angle`` (rad)."""
    Rr = R_from.T @ R_to
    q = np.empty(4)
    mj.mju_mat2Quat(q, Rr.reshape(-1))
    v = np.empty(3)
    mj.mju_quat2Vel(v, q, 1.0)  # rotation vector
    ang = float(np.linalg.norm(v))
    if ang <= max_angle:
        return R_to.copy()
    v *= max_angle / ang
    q2 = np.empty(4)
    mj.mju_axisAngle2Quat(q2, v / max_angle, max_angle)
    M = np.empty(9)
    mj.mju_quat2Mat(M, q2)
    return R_from @ M.reshape(3, 3)


class HoverController:
    """``sim.flight_controller = HoverController(sim)``; ``FlightSimulation.step``
    calls :meth:`maybe_update` before every physics step."""

    # input limits (rad): A, phi0, dA, dT, T (offsets from the base wingbeat)
    U_LIMITS = np.array([25.0, 25.0, 25.0, 20.0, 25.0]) * DEG

    def __init__(self, sim, gains: HoverGains | None = None, target: HoverTarget | None = None,
                 base: WingbeatParams | None = None, every_steps: int = 4) -> None:
        self.g = gains or HoverGains()
        self.target = target or HoverTarget()
        self.base = base or base_wingbeat()
        self.every = every_steps
        self.B4inv = np.linalg.inv(EFFECTIVENESS[[0, 2, 3, 4]][:, :4])
        self.R_beta = _rot_y(math.radians(sim.stroke_plane_deg))
        self.I_S = self._inertia_stroke_frame(sim)
        self.mass = sim.fly_mass
        self.reset()

    def reset(self) -> None:
        self.z_int = 0.0
        self.xy_int = np.zeros(2)
        self.att_int = np.zeros(3)
        self.w_f = None  # filtered rates (S frame)
        self.R_ref = None  # slewed attitude reference (S frame, world)
        self.u = np.zeros(5)
        self.freq = self.base.freq
        # 5 kHz rows: t, com(3), vel(3), e(3), w(3), u(5), e_des(3), freq (~40 s kept)
        self.log: deque[np.ndarray] = deque(maxlen=200_000)

    def _inertia_stroke_frame(self, sim) -> np.ndarray:
        """Composite rotational inertia of the fly about its COM, in the S frame."""
        m, d = sim.model, sim.data
        root = sim.thorax_body_id
        com = d.subtree_com[root]
        I = np.zeros((3, 3))
        for b in range(m.nbody):
            if m.body_rootid[b] != root or m.body_mass[b] <= 0:
                continue
            R = d.ximat[b].reshape(3, 3)
            Ib = R @ np.diag(m.body_inertia[b]) @ R.T
            r = d.xipos[b] - com
            I += Ib + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
        R_S = d.xmat[root].reshape(3, 3) @ self.R_beta
        return R_S.T @ I @ R_S

    def maybe_update(self, sim) -> None:
        if sim.step_count % self.every:
            return
        self.update(sim)

    def update(self, sim) -> None:
        g, tgt = self.g, self.target
        dt = self.every * sim.timestep
        R_S = sim.thorax_rotmat() @ self.R_beta
        w_S = self.R_beta.T @ sim.thorax_angvel_local()
        a = min(1.0, dt / g.rate_tau) if g.rate_tau > 0 else 1.0
        self.w_f = w_S if self.w_f is None else self.w_f + a * (w_S - self.w_f)
        com = sim.com()
        mj.mj_subtreeVel(sim.model, sim.data)
        vel = sim.data.subtree_linvel[sim.thorax_body_id].copy()  # COM velocity

        # --- position loop (world) -> desired horizontal accel, lift. A velocity
        # target moves the position set point (slow forward flight).
        tgt.pos = tgt.pos + tgt.vel * dt
        e_p = tgt.pos - com
        e_v = tgt.vel - vel
        kz, dz = g.z_wn ** 2, 2 * g.z_zeta * g.z_wn
        if abs(e_p[2]) < 1.0:  # anti-windup: integrate only near the set point
            self.z_int = float(np.clip(self.z_int + e_p[2] * dt, -50.0, 50.0))
        az = kz * e_p[2] + dz * e_v[2] + g.z_ki * self.z_int
        kxy, dxy = g.xy_wn ** 2, 2 * g.xy_zeta * g.xy_wn
        self.xy_int = np.clip(self.xy_int + e_p[:2] * dt, -20.0, 20.0)
        a_xy = kxy * e_p[:2] + dxy * e_v[:2] + g.xy_ki * self.xy_int
        n = float(np.linalg.norm(a_xy))
        if n > g.max_accel_xy:
            a_xy *= g.max_accel_xy / n
        # into the heading frame
        c, s = math.cos(tgt.yaw), math.sin(tgt.yaw)
        a_fwd = c * a_xy[0] + s * a_xy[1]
        a_lat = -s * a_xy[0] + c * a_xy[1]
        roll_des = float(np.clip(-a_lat / 9810.0, -g.max_roll, g.max_roll))

        # --- attitude loop (S frame)
        R_des = _rot_z(tgt.yaw) @ _rot_x(roll_des) @ _rot_y(tgt.pitch)
        if self.R_ref is None:
            self.R_ref = R_S.copy()
        self.R_ref = _slew(self.R_ref, R_des, g.att_slew * dt)
        e = _att_err(self.R_ref, R_S)
        self.att_int = np.clip(self.att_int + e * dt, -0.1, 0.1)
        I = self.I_S
        kp = np.array([g.att_wn ** 2, g.att_wn ** 2, g.yaw_wn ** 2])
        kd = np.array([2 * g.att_zeta * g.att_wn, 2 * g.att_zeta * g.att_wn, 2 * g.yaw_zeta * g.yaw_wn])
        alpha = -kp * e - kd * self.w_f
        alpha -= g.att_ki_ratio * kp * np.sqrt(kp) * self.att_int
        tau = I @ alpha + np.cross(self.w_f, I @ self.w_f)

        # --- wrench -> wingbeat inputs. Lift must hold weight along the (tilted)
        # stroke-plane normal: Fz_S = m (g + az) / cos(tilt).
        up = R_S[2, 2]
        Fz = self.mass * (9810.0 + az) / max(up, 0.3)
        # Forward force: what the tilted lift vector already gives along the heading
        # is subtracted, so attitude errors do not push the fly around.
        h = np.array([c, s, 0.0])
        Fx = (self.mass * a_fwd - Fz * float(R_S[:, 2] @ h)) / max(float(R_S[:, 0] @ h), 0.3)
        wrench = np.array([Fz, Fx, tau[0], tau[1], tau[2]])
        # Wingbeat frequency as the second lift channel: all aerodynamic forces
        # scale ~f^2, so when the amplitude alone cannot reach the lift, raise f
        # and ask the kinematics for wrench / k.
        fz_max_amp = BASE_WRENCH[0] + EFFECTIVENESS[0, 0] * 0.8 * self.U_LIMITS[0]
        k = float(np.clip(Fz / fz_max_amp, 1.0, (g.max_freq / self.base.freq) ** 2))
        dw = wrench / k - BASE_WRENCH
        # Priorities: the forward force (symmetric stroke-plane tilt) gets what is
        # left after lift and torques; it is solved first and clipped, then the
        # other four inputs are solved with its side effects included.
        B = EFFECTIVENESS
        u4 = np.zeros(4)
        for _ in range(2):  # (second pass: include the other inputs' Fx)
            T = float(np.clip((dw[1] - B[1, :4] @ u4) / B[1, 4],
                              -self.U_LIMITS[4], self.U_LIMITS[4]))
            u4 = self.B4inv @ (dw[[0, 2, 3, 4]] - B[[0, 2, 3, 4], 4] * T)
        u = np.clip(np.r_[u4, T], -self.U_LIMITS, self.U_LIMITS)
        self.u = u
        self.freq = self.base.freq * math.sqrt(k)
        self._write(sim, u)
        e_des = _att_err(R_des, R_S)
        self.log.append(np.r_[sim.time, com, vel, e, self.w_f, u, e_des, self.freq])

    def _write(self, sim, u) -> None:
        A, phi0, dA, dT, T = u
        b = self.base
        L = replace(b.left, amplitude=b.left.amplitude + A + dA, mean_stroke=b.left.mean_stroke + phi0,
                    tilt=b.left.tilt + T + dT)
        R = replace(b.right, amplitude=b.right.amplitude + A - dA, mean_stroke=b.right.mean_stroke + phi0,
                    tilt=b.right.tilt + T - dT)
        sim.wingbeat.params = replace(b, left=L, right=R, freq=self.freq)
