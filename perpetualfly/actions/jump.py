"""Escape jump: crouch -> fast mid(+hind)-leg extension -> ballistic flight -> landing.

Biology: the giant fiber (DNp01) excites the tergotrochanteral motor neuron (TTMn)
of the *middle* legs; the TTM depresses the trochanter, i.e. snaps the mid
coxa-trochanter joint (and with it femur-tibia) into extension within a few ms,
throwing the fly up (von Reyn et al. 2014; Card & Dickinson 2008). A slower,
"non-giant-fiber" take-off is preceded by a postural adjustment (~50-200 ms in
which the mid legs are placed under the centre of mass) -- that is our
``prep`` phase.

Model: position targets only (``ActionCommand.targets``), adhesion released on all
legs for the stroke and the flight, re-engaged at touchdown. Joint names:
``ctr`` = coxa-trochanterfemur pitch (the TTM joint), ``fti`` = femur-tibia pitch,
``thc`` = thorax-coxa pitch. See docs/ACTIONS.md for the measured results and the
tuning.

Two escape modes (von Reyn et al. 2014, Nat Neurosci 17:962): a **long-mode**
take-off (default, ``mode="long"``) with the ``prep_s`` postural crouch, and a
**short-mode** take-off (``mode="short"``) in which a strong, early giant-fiber
spike drives the TTM before any preparation: the crouch shrinks to ``short_prep_s``
(a few ms of leg set-up) and the stroke follows at once. Real short-mode take-offs
are ~5-10 ms from the GF spike (long mode ~25-40+ ms including wing raising) and are
less stable (flies often tumble); ``BrainActionTriggers`` chooses short mode when
the GF rate is very high (``TriggerParams.jump_short_hz``).

Optional *escape flight* (default off, ``flight_assist=True``; documented physics
emulation, docs/SWATTER.md): real escapes continue in flight -- the wings open
during / right after the jump and the fly flies away from the threat at ~0.3-1 m/s.
Wings are not simulated aerodynamically here, so after take-off an external force on
the thorax (``data.xfrc_applied``, our contribution tracked and removed exactly)
servoes the thorax velocity toward ``flight_speed`` along ``escape_dir`` (world,
horizontal; default the heading) plus ``flight_climb`` for ``flight_s``, then a slow
descent until touchdown, capped at ``flight_max_bw`` body weights, with a PD
attitude stabiliser (the halteres / wing steering). Without it the jump is a
vertical hop that lands where it started (under a flyswatter, that is no dodge).

Optional *boost* (default off): multiply ``kp`` and the force range of the
actuators involved in the stroke by ``boost`` for the stroke only (restored right
after the stroke, on cancel and on reset). The default +-65 uN*mm limit already
makes a ~2 mm jump; the boost is there to emulate the TTM's much higher power
(documented as a physics change in docs/ACTIONS.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco as mj
import numpy as np

from .base import Action, ActionCommand, smoothstep

DEG = np.pi / 180.0


@dataclass
class JumpParams:
    prep_s: float = 0.10  # postural adjustment + crouch
    stroke_s: float = 0.015  # TTM stroke (target step, actuators do the rest)
    land_s: float = 0.15  # stance with adhesion after touchdown
    # crouch: offsets added to the standing pose (deg)
    crouch_ctr: float = -20.0
    crouch_fti: float = 20.0
    mid_thc_crouch: float = 0.0  # extra mid coxa protraction in the crouch (deg)
    # stroke: absolute targets (deg) for the mid legs
    mid_ctr: float = 45.0
    mid_fti: float = 25.0
    mid_thc: float = 12.0  # mid coxa pitch offset during the stroke (sets the take-off pitch)
    hind_frac: float = 0.0  # 0..1: fraction of the mid extension applied to hind legs
    front_frac: float = 0.0  # 0..1: same for front legs
    # flight / landing pose: standing pose + these offsets (deg), all legs
    flight_ctr: float = 10.0
    flight_fti: float = -10.0
    boost: float = 1.0  # kp and force-range multiplier during the stroke (1 = off)
    touchdown_legs: int = 3  # legs in contact that count as landed
    # Postural feedback (the fly's pre-take-off adjustment, done at stroke onset):
    # both mid legs' stroke thc targets are shifted by posture_gain * (x - foot_x_ref),
    # x = mean mid tarsus5 position ahead of the COM in the thorax frame (mm). A foot
    # ahead of the reference makes the push pitch the fly nose-up; more coxa
    # protraction cancels it (measured: ~ -375 rad/s pitch per mm, ~ +6 rad/s per deg).
    posture_gain: float = 43.0  # deg per mm (0 = off)
    foot_x_ref: float = 0.23
    # escape mode (von Reyn et al. 2014): "long" = prep_s crouch, "short" = GF-driven
    # take-off with only short_prep_s of leg set-up before the stroke
    mode: str = "long"
    short_prep_s: float = 0.006
    # escape flight emulation (off by default; see the module docstring)
    flight_assist: bool = False
    flight_s: float = 0.10  # powered flight after take-off
    flight_speed: float = 180.0  # mm/s, horizontal, along escape_dir
    flight_climb: float = 20.0  # mm/s, vertical during powered flight
    flight_descent: float = 50.0  # mm/s, sink rate after flight_s until touchdown
    flight_tau_s: float = 0.02  # velocity servo time constant
    flight_max_bw: float = 4.0  # force cap (body weights), gravity support included
    flight_att_hz: float = 15.0  # attitude stabiliser bandwidth (critically damped)
    flight_max_s: float = 0.40  # give up waiting for touchdown after this (from stroke end)
    escape_dir: tuple | None = None  # world (x, y) escape direction; None = heading


class Jump(Action):
    name = "jump"
    blend_in = 0.0  # the prep phase starts from the controller's current targets
    blend_out = 0.2

    def __init__(self, params: JumpParams | None = None, **overrides) -> None:
        p = params or JumpParams()
        if overrides:
            p = JumpParams(**{**p.__dict__, **overrides})
        if p.mode not in ("long", "short"):
            raise ValueError(f"Jump mode must be 'long' or 'short', not {p.mode!r}")
        self.p = p
        self.prep_s = p.short_prep_s if p.mode == "short" else p.prep_s
        flight_max = p.flight_max_s + (p.flight_s if p.flight_assist else 0.0)
        self._flight_max = flight_max
        super().__init__(self.prep_s + p.stroke_s + flight_max + p.land_s)
        self._t_touch: float | None = None
        self._t_takeoff: float | None = None
        self._xf = None  # our xfrc_applied contribution (flight assist)

    def phase(self, t: float) -> str:
        p = self.p
        if t < self.prep_s:
            return "crouch"
        if t < self.prep_s + p.stroke_s:
            return "stroke"
        if self._t_touch is None:
            return "flight"
        return "landing"

    def begin(self, mgr) -> None:
        sim, b, p = mgr.sim, mgr.body, self.p
        d = sim.data
        self._start = d.ctrl[b.pos_ids].copy()  # the controller's targets right now
        stand = b.stand
        self._crouch = stand.copy()
        self._ext = stand.copy()
        self._flight = stand.copy()
        stroke_joints = []
        for leg in ("lf", "lm", "lh", "rf", "rm", "rh"):
            ctr, fti, thc = b.idx(leg, "ctr_pitch"), b.idx(leg, "fti"), b.idx(leg, "thc_pitch")
            self._crouch[ctr] += p.crouch_ctr * DEG
            self._crouch[fti] += p.crouch_fti * DEG
            self._flight[ctr] += p.flight_ctr * DEG
            self._flight[fti] += p.flight_fti * DEG
            frac = {"m": 1.0, "h": p.hind_frac, "f": p.front_frac}[leg[1]]
            if leg[1] == "m":
                self._crouch[thc] += p.mid_thc_crouch * DEG
                self._ext[thc] = stand[thc] + p.mid_thc * DEG
            else:
                self._ext[thc] = self._crouch[thc]
            if frac > 0:
                self._ext[ctr] = self._crouch[ctr] + frac * (p.mid_ctr * DEG - self._crouch[ctr])
                self._ext[fti] = self._crouch[fti] + frac * (p.mid_fti * DEG - self._crouch[fti])
                stroke_joints += [ctr, fti]
            else:
                self._ext[ctr] = self._crouch[ctr]
                self._ext[fti] = self._crouch[fti]
        self._stroke_act = b.pos_ids[np.array(stroke_joints, dtype=np.int64)]
        self._boosted = False
        self._z0 = float(sim.thorax_position()[2])
        self._p0 = sim.thorax_position()
        self._h0 = sim.heading()
        self._zmax = self._z0
        self._tilt_max = 0.0
        self._p_takeoff = None
        self._t_touch = None
        self._t_takeoff = None
        self._w_stroke_end = None
        self._feet_at_stroke = None
        m = sim.model
        self._tarsus5 = np.array([mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"{sim.fly_name}/{leg}_tarsus5")
                                  for leg in ("lf", "lm", "lh", "rf", "rm", "rh")])
        self._v_stroke_end = None
        self._on = np.ones(6)
        self._off = np.zeros(6)
        self._xf = None
        self._assist_done = False
        self._flight_dist = 0.0
        if p.flight_assist:
            self._setup_flight(sim)

    def command(self, mgr, t: float) -> ActionCommand:
        sim, p = mgr.sim, self.p
        z = float(sim.data.xpos[sim.thorax_body_id, 2])
        self._zmax = max(self._zmax, z)
        self._tilt_max = max(self._tilt_max, sim.tilt_deg())
        t1 = self.prep_s
        t2 = t1 + p.stroke_s
        if t < t1:
            a = smoothstep(t / t1)
            return ActionCommand(targets=(1 - a) * self._start + a * self._crouch, adhesion=self._on)
        if t < t2:
            if self._feet_at_stroke is None:
                self._feet_at_stroke = self._feet_rel_com(mgr)
                self._compensate_posture(mgr)
            if not self._boosted and p.boost != 1.0:
                mgr.boost_actuators(self._stroke_act, p.boost, p.boost)
                self._boosted = True
            self._check_takeoff(mgr, t)
            return ActionCommand(targets=self._ext, adhesion=self._off)
        if self._boosted:
            mgr.restore_actuators()
            self._boosted = False
        if self._w_stroke_end is None:
            self._w_stroke_end = sim.thorax_angvel_local()
            self._v_stroke_end = sim.thorax_linvel()
        self._check_takeoff(mgr, t)
        if self._t_touch is None:
            legs, body = mgr.body.ground_contacts(sim)
            airborne_long_enough = self._t_takeoff is not None and t - self._t_takeoff > 0.015
            landed = airborne_long_enough and (legs.sum() >= p.touchdown_legs or body)
            if landed or t > t2 + self._flight_max:
                self._t_touch = t
                self._p_touch = sim.thorax_position()
                self._clear_flight_force(sim)
            else:
                if p.flight_assist and self._t_takeoff is not None and not self._assist_done:
                    t_air = t - self._t_takeoff
                    if t_air > p.flight_s and legs.any():
                        # first foot down after the powered flight: wings stop, the
                        # normal touchdown / landing logic takes over
                        self._clear_flight_force(sim)
                        self._assist_done = True
                    else:
                        self._flight_force(sim, t_air)
                # flight: legs swing to the landing pose
                a = smoothstep((t - t2) / 0.03)
                return ActionCommand(targets=(1 - a) * self._ext + a * self._flight, adhesion=self._off)
        if t - self._t_touch >= p.land_s:
            self.done = True
        a = smoothstep((t - self._t_touch) / 0.05)
        return ActionCommand(targets=(1 - a) * self._flight + a * mgr.body.stand, adhesion=self._on)

    def _compensate_posture(self, mgr) -> None:
        p, b = self.p, mgr.body
        if p.posture_gain == 0.0:
            return
        self._ext = self._ext.copy()
        # same shift on both mid legs (a left/right difference would roll the fly)
        dx = 0.5 * float(self._feet_at_stroke[1, 0] + self._feet_at_stroke[4, 0]) - p.foot_x_ref
        shift = float(np.clip(p.posture_gain * dx, -30.0, 30.0)) * DEG
        for leg in ("lm", "rm"):
            self._ext[b.idx(leg, "thc_pitch")] += shift

    def _feet_rel_com(self, mgr) -> np.ndarray:
        """(6, 3) tarsus5 positions relative to the fly's COM, thorax frame (mm)."""
        sim = mgr.sim
        com = sim.data.subtree_com[sim.thorax_body_id]
        return (sim.data.xpos[self._tarsus5] - com) @ sim.thorax_rotmat()

    def _check_takeoff(self, mgr, t: float) -> None:
        """Take-off = first moment (from the stroke on) with no leg on the ground.
        Legs can still tap the floor once more at the end of the stroke, so the
        take-off *velocity* is read at the end of the stroke instead."""
        if self._t_takeoff is None and not mgr.body.leg_contacts(mgr.sim).any():
            self._t_takeoff = t
            self._p_takeoff = mgr.sim.thorax_position()

    # ------------------------------------------------------------ escape flight
    def _setup_flight(self, sim) -> None:
        m, d = sim.model, sim.data
        tid = sim.thorax_body_id
        bodies = np.flatnonzero(m.body_rootid == tid)
        com = d.subtree_com[tid]
        # composite inertia about the COM (isotropic estimate: mean principal moment)
        inertia = 0.0
        for b in bodies:
            r = d.xipos[b] - com
            inertia += float(m.body_inertia[b].mean()) + float(m.body_mass[b]) * float(r @ r) * 2 / 3
        self._inertia = inertia
        self._mass = float(sim.fly_mass)
        self._g = float(np.linalg.norm(m.opt.gravity))
        if self.p.escape_dir is not None:
            e = np.array([self.p.escape_dir[0], self.p.escape_dir[1], 0.0], dtype=float)
        else:
            h = sim.heading()
            e = np.array([np.cos(h), np.sin(h), 0.0])
        n = float(np.linalg.norm(e))
        self._escape = e / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    def _flight_force(self, sim, t_air: float) -> None:
        p = self.p
        d = sim.data
        tid = sim.thorax_body_id
        v = sim.thorax_linvel()
        if t_air < p.flight_s:
            v_des = self._escape * p.flight_speed + np.array([0.0, 0.0, p.flight_climb])
        else:
            v_des = self._escape * p.flight_speed * np.exp(-(t_air - p.flight_s) / 0.05)
            v_des[2] = -p.flight_descent
        f = self._mass * (v_des - v) / max(p.flight_tau_s, 1e-4)
        f[2] += self._mass * self._g
        cap = p.flight_max_bw * self._mass * self._g
        n = float(np.linalg.norm(f))
        if n > cap:
            f *= cap / n
        # attitude: rotate the thorax z axis back to world up, damp the body rates
        R = sim.thorax_rotmat()
        err = np.cross(R[:, 2], np.array([0.0, 0.0, 1.0]))
        w = 2 * np.pi * p.flight_att_hz
        wv = sim.thorax_angvel_world()
        tq = self._inertia * (w * w * err - 2.0 * w * wv)
        new = np.concatenate([f, tq])
        if self._xf is not None:
            d.xfrc_applied[tid] -= self._xf
        d.xfrc_applied[tid] += new
        self._xf = new
        self._flight_dist = float(np.hypot(*(sim.thorax_position()[:2] - self._p0[:2])))

    def _clear_flight_force(self, sim) -> None:
        if self._xf is not None:
            sim.data.xfrc_applied[sim.thorax_body_id] -= self._xf
            self._xf = None

    def end(self, mgr, cancelled: bool) -> None:
        sim = mgr.sim
        self._clear_flight_force(sim)
        info = {
            "apex_dz_mm": self._zmax - self._z0,
            "max_tilt_deg": self._tilt_max,
            "final_tilt_deg": sim.tilt_deg(),
            "boost": self.p.boost,
            "cancelled": cancelled,
            "mode": self.p.mode,
            "prep_s": self.prep_s,
            "flight_assist": self.p.flight_assist,
        }
        if self._feet_at_stroke is not None:
            info["feet_x_at_stroke_mm"] = [float(v) for v in self._feet_at_stroke[:, 0]]
        if self._w_stroke_end is not None:
            # body rates right after the stroke: x roll, y pitch (+ = nose down), z yaw
            info["spin_rad_s"] = [float(v) for v in self._w_stroke_end]
            info["v_after_stroke_mm_s"] = [float(v) for v in self._v_stroke_end]
        if self._t_takeoff is not None:
            info["takeoff_after_s"] = self._t_takeoff
        if self._v_stroke_end is not None:
            info["takeoff_speed_mm_s"] = float(np.linalg.norm(self._v_stroke_end))
            info["takeoff_vz_mm_s"] = float(self._v_stroke_end[2])
        if self._t_takeoff is not None and self._t_touch is not None:
            info["airtime_s"] = self._t_touch - self._t_takeoff
            dp = self._p_touch - self._p0
            fwd0 = np.array([np.cos(self._h0), np.sin(self._h0)])
            info["distance_mm"] = float(np.hypot(dp[0], dp[1]))
            info["forward_mm"] = float(dp[:2] @ fwd0)
        info["landed_upright"] = bool(sim.tilt_deg() < 45.0)
        self.info = info
