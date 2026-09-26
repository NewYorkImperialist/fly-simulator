"""Flight-mode simulation: ``perpetualfly.simulation.Simulation`` with the
flight fly (``perpetualfly.flight.body``), dt = 5e-5 s, air on, a wingbeat
generator driving the wing servos at every physics step, and optionally a flight
controller (``perpetualfly.flight.control``) updating the wingbeat parameters at
the control rate.

The default walking ``Simulation`` / model / timestep are untouched: everything
here is opt-in (a copied ``AppConfig`` with ``sim.timestep = 5e-5``).

Leg modes:

* ``"walk"``: the locomotion controller runs as in ``Simulation`` (and any
  pre-step hook, e.g. the ``ActionManager`` of a jump, may override it);
* ``"flight"``: legs held at the neutral pose, adhesion off (the controller is
  not stepped);
* ``"stance"``: neutral pose with adhesion on (touchdown).

Tether: ``tethered=True`` pins the free joint (thorax) at ``tether_pose`` after
every step (qvel = 0). Aerodynamic forces are read from ``data.qfrc_fluid``: its
free-joint translational components are the total fluid force (world frame) on
the whole fly, its rotational components the torque about the free-joint origin
(thorax frame). MuJoCo force sensors do *not* see fluid forces.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import mujoco as mj
import numpy as np

from perpetualfly.config import AppConfig
from perpetualfly.simulation import Simulation

from .body import (
    FLIGHT_TIMESTEP, STROKE_PLANE_DEG, WING_JOINTS, apply_air, make_flight_fly_factory,
    wing_actuator_name, wing_joint_name,
)
from .wingbeat import WingbeatGenerator, WingbeatParams

GRAVITY = 9810.0


def quat_pitch(pitch_up_rad: float, yaw_rad: float = 0.0) -> np.ndarray:
    """(w, x, y, z) for heading ``yaw`` and nose-up pitch (rotation about -y)."""
    qy = np.array([math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)])
    qp = np.array([math.cos(-pitch_up_rad / 2), 0.0, math.sin(-pitch_up_rad / 2), 0.0])
    out = np.empty(4)
    mj.mju_mulQuat(out, qy, qp)
    return out


def flight_config(cfg: AppConfig | None = None) -> AppConfig:
    cfg = copy.deepcopy(cfg or AppConfig())
    cfg.sim.timestep = FLIGHT_TIMESTEP
    cfg.sim.control_every_steps = 1
    return cfg


class FlightSimulation(Simulation):
    def __init__(
        self,
        cfg: AppConfig | None = None,
        *,
        tethered: bool = False,
        stroke_plane_deg: float = STROKE_PLANE_DEG,
        wingbeat: WingbeatParams | None = None,
        world_factory=None,
        world_extensions=(),
        fly_kwargs: dict | None = None,
        fly_factory_wrapper=None,
    ) -> None:
        """``fly_factory_wrapper(factory) -> factory`` decorates the flight fly
        factory (e.g. ``make_eyes_fly_factory(base=...)`` adds compound eyes)."""
        self._flight_ready = False
        self.tethered = tethered
        self.leg_mode = "flight"
        self.flapping = False
        self.flight_controller = None
        self.wingbeat = WingbeatGenerator(wingbeat)
        kw = dict(fly_kwargs or {})
        kw.setdefault("stroke_plane_deg", stroke_plane_deg)
        self.stroke_plane_deg = kw["stroke_plane_deg"]
        factory = make_flight_fly_factory(**kw)
        if fly_factory_wrapper is not None:
            factory = fly_factory_wrapper(factory)
        super().__init__(flight_config(cfg), world_factory=world_factory,
                         world_extensions=world_extensions, fly_factory=factory)
        m = self.model
        apply_air(m)
        fn = self.fly_name
        self.wing_act = np.array([
            mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, f"{fn}/{wing_actuator_name(s, j)}")
            for s in "lr" for j in WING_JOINTS])
        jids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"{fn}/{wing_joint_name(s, j)}")
                for s in "lr" for j in WING_JOINTS]
        assert min(jids) >= 0 and self.wing_act.min() >= 0
        self.wing_qpos = m.jnt_qposadr[jids].astype(int)
        self.wing_qvel = m.jnt_dofadr[jids].astype(int)
        # kv / kp per wing servo (kv = joint damping): ctrl = q_ref + kv/kp * qd_ref
        # gives kp (q_ref - q) + kv (qd_ref - qd).
        self._kv_over_kp = m.dof_damping[self.wing_qvel] / m.actuator_gainprm[self.wing_act, 0]
        self.wing_bodies = np.array([mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"{fn}/{s}_wing")
                                     for s in "lr"])
        self.leg_act = np.asarray(self.controller._pos_ids)
        self.adh_act = self.controller._adh_ids
        key = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, "neutral")
        self.leg_neutral_ctrl = m.key_ctrl[key, self.leg_act].copy()
        self.weight = self.fly_mass * GRAVITY  # uN
        self.tether_pose: np.ndarray | None = None
        self._flight_ready = True
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        super().reset()
        if not getattr(self, "_flight_ready", False):
            return
        self.wingbeat.reset()
        self.data.ctrl[self.wing_act] = 0.0
        self.tether_pose = self.data.qpos[self._free_qpos:self._free_qpos + 7].copy()

    def place(self, pos, pitch_up_deg: float = STROKE_PLANE_DEG, yaw_deg: float = 0.0,
              linvel=(0.0, 0.0, 0.0)) -> None:
        """Put the fly (free joint) at ``pos`` with the given attitude, at rest,
        legs in the flight pose, wings at the start of a wingbeat."""
        d, a = self.data, self._free_qpos
        d.qpos[a:a + 3] = pos
        d.qpos[a + 3:a + 7] = quat_pitch(math.radians(pitch_up_deg), math.radians(yaw_deg))
        v = self._free_qvel
        d.qvel[v:v + 6] = 0.0
        d.qvel[v:v + 3] = linvel
        self.leg_mode = "flight"
        self.wingbeat.reset()
        q, _ = self.wingbeat.targets()
        d.qpos[self.wing_qpos] = q
        d.qvel[self.wing_qvel] = 0.0
        self._legs_flight_pose()
        mj.mj_forward(self.model, d)
        self.tether_pose = d.qpos[a:a + 7].copy()

    def _legs_flight_pose(self, adhesion: bool = False) -> None:
        self.data.ctrl[self.leg_act] = self.leg_neutral_ctrl
        if self.adh_act is not None:
            self.data.ctrl[self.adh_act] = 1.0 if adhesion else 0.0

    # ------------------------------------------------------------------ stepping
    def _apply_wings(self) -> None:
        if not self.flapping:
            return
        q, dq = self.wingbeat.targets()
        self.data.ctrl[self.wing_act] = q + self._kv_over_kp * dq

    def step(self, n: int = 1) -> None:
        m, d = self.model, self.data
        dt = m.opt.timestep
        check_every = self.cfg.sim.check_every_steps
        a, v = self._free_qpos, self._free_qvel
        for _ in range(n):
            if self.flight_controller is not None:
                self.flight_controller.maybe_update(self)
            if self.leg_mode == "walk":
                self.controller.step_and_apply()
            else:
                self._legs_flight_pose(adhesion=self.leg_mode == "stance")
            self._apply_wings()
            for hook in self.pre_step_hooks:
                hook(self)
            mj.mj_step(m, d)
            if self.flapping:
                self.wingbeat.advance(dt)
            if self.tethered:
                d.qpos[a:a + 7] = self.tether_pose
                d.qvel[v:v + 6] = 0.0
            self.step_count += 1
            for hook in self.post_step_hooks:
                hook(self)
            if self.step_count % check_every == 0:
                self.check_stability()

    # ------------------------------------------------------------------ readouts
    def fluid_force(self) -> np.ndarray:
        """Total fluid force on the fly (world frame, uN) from the last step."""
        v = self._free_qvel
        return self.data.qfrc_fluid[v:v + 3].copy()

    def fluid_torque_com(self) -> np.ndarray:
        """Fluid torque about the fly's centre of mass (world frame, uN*mm)."""
        d, v = self.data, self._free_qvel
        R = d.xmat[self.thorax_body_id].reshape(3, 3)
        tau_o = R @ d.qfrc_fluid[v + 3:v + 6]
        o = d.xpos[self.thorax_body_id]
        c = d.subtree_com[self.thorax_body_id]
        return tau_o + np.cross(o - c, d.qfrc_fluid[v:v + 3])

    def body_frame_force(self, f_world: np.ndarray) -> np.ndarray:
        return self.thorax_rotmat().T @ f_world

    def wing_servo_torque(self) -> np.ndarray:
        """Net servo ("muscle") torque on the 6 wing hinges: actuator output minus
        the servo damping kv * dq (uN*mm)."""
        d, m = self.data, self.model
        return (d.actuator_force[self.wing_act]
                - (m.dof_damping[self.wing_qvel]) * d.qvel[self.wing_qvel])

    def wing_angles(self) -> np.ndarray:
        return self.data.qpos[self.wing_qpos].copy()

    def com(self) -> np.ndarray:
        return self.data.subtree_com[self.thorax_body_id].copy()


@dataclass
class TetherResult:
    freq: float
    amplitude_cmd_deg: float
    amplitude_meas_deg: float
    force_world: np.ndarray  # cycle-averaged, uN
    torque_com: np.ndarray  # cycle-averaged, world frame, uN*mm
    weight: float
    max_actuator_force: float
    extra: dict = field(default_factory=dict)

    @property
    def lift_over_weight(self) -> float:
        return float(self.force_world[2] / self.weight)


def measure_tethered(sim: FlightSimulation, params: WingbeatParams, *, cycles: int = 6,
                     settle_cycles: int = 3, pitch_up_deg: float | None = None,
                     pos=(0.0, 0.0, 10.0)) -> TetherResult:
    """Pin the fly at ``pos`` with the stroke plane horizontal (default pitch =
    stroke-plane angle), flap with ``params`` and average the fluid force over
    ``cycles`` whole wingbeats after ``settle_cycles``."""
    sim.tethered = True
    sim.wingbeat.params = params
    sim.place(pos, sim.stroke_plane_deg if pitch_up_deg is None else pitch_up_deg)
    sim.flapping = True
    dt = sim.timestep
    n_settle = int(round(settle_cycles / params.freq / dt))
    n_meas = int(round(cycles / params.freq / dt))
    sim.step(n_settle)
    F = np.zeros(3)
    T = np.zeros(3)
    stroke = []
    fmax = 0.0
    for _ in range(n_meas):
        sim.step(1)
        F += sim.fluid_force()
        T += sim.fluid_torque_com()
        stroke.append(sim.data.qpos[sim.wing_qpos[0]])
        fmax = max(fmax, float(np.abs(sim.wing_servo_torque()).max()))
    sim.flapping = False
    stroke = np.asarray(stroke)
    return TetherResult(params.freq, math.degrees(params.left.amplitude),
                        math.degrees(stroke.max() - stroke.min()), F / n_meas, T / n_meas,
                        sim.weight, fmax)
