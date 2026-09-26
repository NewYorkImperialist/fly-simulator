"""A physical whip: a visible chain of capsules that strikes the fly through MuJoCo contact.

Model (added to the world ``MjSpec`` by ``Whip.extension`` before ``add_fly``):

* ``whip/handle``: a **mocap** body (kinematic, like a hand). It has no geoms.
* ``whip/grip``: a **free** body (the wooden handle stock, visual only) welded to the
  mocap body by a stiff ``weld`` equality. A chain parented *directly* to a mocap body
  would be carried along kinematically (mocap bodies have no velocity in MuJoCo, so
  the chain would neither lag nor carry momentum). Driving a real body through a weld
  gives the chain proper dynamics: the grip is accelerated by constraint forces and
  the flexible chain lags behind it and lashes.
* ``whip/seg0..N-1``: capsules chained from the grip along its local +x axis, each
  joined to its parent by two hinges (local y = out of the swing plane, local z = in
  the swing plane) with stiffness, damping and armature (a springy, tapered lash).
  The capsules collide **only with the fly** (``contype 0, conaffinity FLY_BIT``;
  ``collide_terrain`` adds ``TERRAIN_BIT``) with FlyGym's stiff ground-contact
  parameters and ``priority 1``. There is no whip self-collision.

A *crack* is scripted motion of the handle only: the fly moves exclusively through the
contact forces of the chain. Phases (sim time, see ``WhipConfig``):
``to_up`` (whip raised, pointing up, handle moved above the strike pivot) -> ``down``
(cocked: whip pointing away from the strike direction, beside the fly) -> ``hold`` ->
``swing`` (the handle rotates about the axis ``u x s`` so the chain sweeps across the
thorax in direction ``s``) -> ``follow`` -> ``lift`` -> ``back`` (to the idle pose).
All transitions go through the raised pose, so the whip never sweeps across the fly
except in the swing.

Strike geometry (fly heading frame, frozen at crack start). ``s`` = push direction
(e.g. a crack *from the left* pushes the fly to its right), ``u`` = whip direction at
impact (from the pivot to the aim point, perpendicular to ``s``, tilted down by
``descent_deg``). The pivot sits ``contact_radius`` mm from the aim point:
left/right: pivot behind the fly, whip along the body axis; front/rear: pivot beside
the fly; overhead: whip horizontal along the body axis, chop downward.

Measurement: a post-step hook sums the world-frame contact forces that the whip
exerts on fly geoms (``mj_contactForce``, rotated by ``contact.frame``) times ``dt``
into the strike's impulse vector, tracks the peak force, contact duration and bodies
hit, and emits a ``WhipHitEvent`` at the end of the strike window (``hit=False`` for
a miss).

Idle: the handle follows a smoothed copy of the fly pose, parked behind / above / to
the right of the fly with the whip pointing forward-right-down, clear of the fly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Callable

import mujoco as mj
import numpy as np

from fly_simulator.terrain import FLY_BIT, TERRAIN_BIT

if TYPE_CHECKING:
    from flygym.compose import BaseWorld

    from fly_simulator.simulation import Simulation

STRIKE_BUSY = ("to_up", "down", "hold", "swing", "follow", "lift")
STRIKE_WINDOW = ("swing", "follow", "lift")  # contacts in these phases count for the strike
WHIP_SIDES = ("left", "right", "front", "rear", "overhead", "random")
PREFIX = "whip/"


@dataclass
class WhipLevel:
    name: str
    omega: float  # peak handle angular speed during the swing (rad/s)
    # The handle also moves this far toward the fly during the swing (a lunge; adds
    # momentum to the whole chain, which the rotation alone can't at high speed).
    lunge: float = 0.0  # mm


def _default_levels() -> list[WhipLevel]:
    # Tuned with scripts/demo_whip.py --calibrate; see docs/WHIP.md.
    return [
        WhipLevel("gentle", 60.0),
        WhipLevel("medium", 120.0),
        WhipLevel("hard", 200.0),
        WhipLevel("absurd", 1200.0, lunge=6.0),
    ]


@dataclass
class WhipConfig:
    enabled: bool = True  # build the whip in the app (H toggles whip / shove mode)
    # ---- geometry (mm) / mass (g) -------------------------------------------
    n_segments: int = 12
    length: float = 6.0  # chain length (mm)
    radius_base: float = 0.09
    radius_tip: float = 0.04
    chain_mass: float = 2e-3  # g (2 mg; the fly is 1.02 mg); per segment ~ radius^2
    grip_mass: float = 5e-3  # g, welded to the mocap handle
    grip_length: float = 1.6  # visual handle stock behind the chain base (mm)
    grip_radius: float = 0.11
    # ---- joints (per hinge; interpolated base -> tip) -------------------------
    stiffness_base: float = 6000.0  # uN*mm/rad
    stiffness_tip: float = 120.0
    damping_base: float = 8.0  # uN*mm*s/rad
    damping_tip: float = 0.06
    # Hinges about the local y axis bend the chain out of the swing plane (e.g.
    # vertically in a sideways swing): stiffer, so the whip doesn't ride over the fly.
    out_of_plane_factor: float = 3.0
    armature: float = 8e-6  # g*mm^2, numerical stability of the light tip at dt=1e-4
    # ---- weld (mocap handle -> grip) -----------------------------------------
    weld_solref: tuple[float, float] = (4e-4, 1.0)
    # ---- contacts -------------------------------------------------------------
    collide_terrain: bool = False  # also collide with ground/terrain (default: fly only)
    friction: float = 0.2  # sliding friction of whip-fly contacts
    # ---- look -----------------------------------------------------------------
    rgba_chain: tuple[float, float, float, float] = (0.36, 0.20, 0.09, 1.0)  # leather
    rgba_tip: tuple[float, float, float, float] = (0.85, 0.15, 0.05, 1.0)  # red cracker
    rgba_grip: tuple[float, float, float, float] = (0.55, 0.36, 0.16, 1.0)  # wood
    # ---- idle pose (fly heading frame: x forward, y left, z up; mm) -----------
    idle_offset: tuple[float, float, float] = (-3.0, -3.0, 2.3)
    idle_direction: tuple[float, float, float] = (0.6, -0.6, -0.2)
    idle_tau_s: float = 0.12  # smoothing of the idle follow (position)
    heading_tau_s: float = 0.4
    track_vmax: float = 300.0  # mm/s, max speed of the handle's follow motion
    track_amax: float = 2e4  # mm/s^2
    idle_update_steps: int = 10  # idle: move the handle / check contacts every N steps
    # ---- strike geometry -------------------------------------------------------
    aim_offset: tuple[float, float, float] = (-0.45, 0.0, 0.12)  # thorax frame (mm)
    contact_radius: float = 4.6  # pivot -> aim distance (mm)
    descent_deg: float = 8.0  # whip tilted down toward the aim point
    back_angle_deg: float = 160.0  # cocked angle behind the impact line
    # per-side aim height shift (mm): the abdomen (hit from the rear) and the head
    # (front) are lower than the thorax top; overhead aims at the top of the thorax.
    # per-side swing speed factor: a crack from the front pushes the fly backward,
    # which tips it far more easily (as with the shove, docs/PERTURBATION_CALIBRATION.md)
    side_omega_scale: dict = field(default_factory=lambda: {"front": 0.9})
    side_aim_dz: dict = field(default_factory=lambda: {"front": -0.2, "rear": -0.35,
                                                       "overhead": 0.4})
    # Fly surface distance from the aim point (heading frame, mm; aim point is
    # (-0.45, 0, +0.22) from the thorax frame: head front +0.97, abdomen tip ~-1.8,
    # thorax side 0.44) and the gap of the whip's rest line at the handle stop.
    extent_front: float = 0.95
    extent_rear: float = 1.7
    extent_side: float = 0.45
    stop_gap: float = 0.15
    lift: float = 3.5  # raised pose: handle this far above the pivot (mm)
    up_tilt: float = 0.35  # raised pose: whip tilted away from the fly
    # ---- timing (sim s) --------------------------------------------------------
    t_to_up: float = 0.06
    t_down: float = 0.05
    t_hold: float = 0.06
    swing_accel_s: float = 0.006  # time to reach the peak angular speed
    swing_decel_deg: float = 6.0  # the handle brakes over this final angle
    t_follow: float = 0.02  # handle held at the stop while the chain lashes
    t_lift: float = 0.04
    t_back: float = 0.09
    levels: list[WhipLevel] = field(default_factory=_default_levels)
    random_elevation_deg: tuple[float, float] = (0.0, 20.0)  # extra descent for "random"
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WhipConfig":
        d = dict(d)
        if "levels" in d:
            d["levels"] = [lv if isinstance(lv, WhipLevel) else WhipLevel(**lv)
                           for lv in d["levels"]]
        for k in ("weld_solref", "rgba_chain", "rgba_tip", "rgba_grip", "idle_offset",
                  "idle_direction", "aim_offset", "random_elevation_deg"):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        return cls(**d)


@dataclass
class WhipHitEvent:
    """Result of one crack. Field names follow ``HitEvent`` where they overlap."""

    sim_time: float  # s, first whip-fly contact (swing start for a miss)
    step: int
    source: str  # "key" | "auto" | "api" | "demo" ...
    body: str  # fly body that received most of the impulse ("" for a miss)
    direction_name: str  # commanded side, e.g. "from_left"
    direction: tuple[float, float, float]  # unit vector of the measured impulse (world)
    magnitude_uN: float  # peak total contact force on the fly
    magnitude_bw: float  # peak force / body weight
    duration_s: float  # first to last contact
    impulse_uNs: float  # |measured impulse| (uN*s)
    level: int | None
    hit: bool = True
    side: str = ""
    impulse_vec: tuple[float, float, float] = (0.0, 0.0, 0.0)  # uN*s, world
    mean_force_uN: float = 0.0  # impulse / duration
    commanded_direction: tuple[float, float, float] = (0.0, 0.0, 0.0)  # s (world)
    omega: float = 0.0  # handle angular speed (rad/s)
    crack_time: float = 0.0  # sim time the crack was requested
    n_contact_steps: int = 0
    bodies: dict = field(default_factory=dict)  # fly body -> |impulse| (uN*s)
    kind: str = "whip"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# small rotation helpers (quaternions w, x, y, z)
# ---------------------------------------------------------------------------


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def _frame_quat(x_axis: np.ndarray, y_hint: np.ndarray) -> np.ndarray:
    """Quaternion of the frame whose x axis is ``x_axis`` and whose y axis is as close
    as possible to ``y_hint``."""
    x = _unit(x_axis)
    y = y_hint - np.dot(y_hint, x) * x
    if np.linalg.norm(y) < 1e-6:
        y = np.cross([0.0, 0.0, 1.0], x)
        if np.linalg.norm(y) < 1e-6:
            y = np.array([0.0, 1.0, 0.0])
    y = _unit(y)
    z = np.cross(x, y)
    q = np.empty(4)
    mj.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    return q


def _rot(axis: np.ndarray, angle: float, v: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of ``v`` about unit ``axis``."""
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(min(1.0, d))
    s = math.sin(th)
    return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / s


def _smooth(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * t * (t * (6 * t - 15) + 10)  # smootherstep


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


# ---------------------------------------------------------------------------
# the crack plan
# ---------------------------------------------------------------------------


@dataclass
class _Crack:
    side: str
    level: int
    omega: float
    source: str
    t_request: float
    s: np.ndarray  # push direction (world unit)
    u: np.ndarray  # whip direction at impact (world unit)
    k: np.ndarray  # swing axis u x s
    pivot: np.ndarray  # pivot offset from the anchor (world axes)
    start_pos: np.ndarray  # handle pose at crack start, relative to the anchor
    start_quat: np.ndarray
    up_pos: np.ndarray
    up_quat: np.ndarray
    theta: np.ndarray  # swing angle per physics step (rad)
    lunge: float = 0.0
    phase: str = "to_up"
    t_phase: float = 0.0  # sim time the phase started
    step_in_phase: int = 0
    # measurement
    impulse: np.ndarray = field(default_factory=lambda: np.zeros(3))
    peak: float = 0.0
    first_t: float | None = None
    first_step: int = 0
    last_t: float | None = None
    n_contact_steps: int = 0
    bodies: dict = field(default_factory=dict)
    swing_t: float = 0.0


class Whip:
    """Physical whip. Use ``Simulation(cfg, world_extensions=[whip.extension])`` then
    ``whip.attach(sim)``; ``whip.crack("left", level=2)`` starts a strike."""

    def __init__(self, cfg: WhipConfig | None = None) -> None:
        self.cfg = cfg or WhipConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.sim: "Simulation | None" = None
        self.listeners: list[Callable[[WhipHitEvent], None]] = []
        self.events: list[WhipHitEvent] = []
        self.n_cracks = 0
        self.stray_contact_steps = 0  # whip-fly contact outside any strike window
        self.stray_impulse = 0.0
        self._crack: _Crack | None = None
        self._pending: tuple | None = None
        self._attached = False

    # ------------------------------------------------------------------ model
    def _segment_params(self):
        c = self.cfg
        n = c.n_segments
        f = np.linspace(0.0, 1.0, n)  # 0 at the base, 1 at the tip
        radius = c.radius_base + (c.radius_tip - c.radius_base) * f
        w = radius ** 2
        mass = c.chain_mass * w / w.sum()
        # stiffness / damping: geometric interpolation (bending stiffness ~ r^4)
        stiff = c.stiffness_base * (c.stiffness_tip / c.stiffness_base) ** f
        damp = c.damping_base * (c.damping_tip / c.damping_base) ** f
        return radius, mass, stiff, damp

    def _idle_pose_world(self, anchor: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        qy = _yaw_quat(yaw)
        R = np.empty(9)
        mj.mju_quat2Mat(R, qy)
        R = R.reshape(3, 3)
        pos = anchor + R @ np.asarray(c.idle_offset, float)
        d = R @ _unit(c.idle_direction)
        quat = _frame_quat(d, R @ np.array([0.0, 0.0, 1.0]))
        return pos, quat

    def extension(self, world: "BaseWorld") -> None:
        """World extension: adds the whip bodies to ``world.mjcf_root`` (before add_fly)."""
        from flygym.compose import ContactParams

        c = self.cfg
        spec: mj.MjSpec = world.mjcf_root
        cp = ContactParams()
        # Spawn pose: the idle pose relative to the fly's spawn pose (thorax ~2.1 mm up
        # at t=0, facing +x). The idle offset keeps everything clear of the fly.
        spawn_anchor = np.array([0.0, 0.0, 1.2])
        pos, quat = self._idle_pose_world(spawn_anchor, 0.0)
        spec.worldbody.add_body(name=PREFIX + "handle", pos=pos, quat=quat, mocap=True)
        grip = spec.worldbody.add_body(name=PREFIX + "grip", pos=pos, quat=quat)
        grip.add_freejoint(name=PREFIX + "grip_free")
        grip.add_geom(name=PREFIX + "grip_geom", type=mj.mjtGeom.mjGEOM_CAPSULE,
                      fromto=(-c.grip_length, 0, 0, 0, 0, 0), size=(c.grip_radius, 0, 0),
                      mass=c.grip_mass, contype=0, conaffinity=0, rgba=c.rgba_grip, group=1)
        radius, mass, stiff, damp = self._segment_params()
        seg_len = c.length / c.n_segments
        affinity = FLY_BIT | (TERRAIN_BIT if c.collide_terrain else 0)
        parent = grip
        for i in range(c.n_segments):
            body = parent.add_body(name=f"{PREFIX}seg{i}", pos=(0.0 if i == 0 else seg_len, 0, 0))
            for ax_name, axis in (("y", (0, 1, 0)), ("z", (0, 0, 1))):
                body.add_joint(name=f"{PREFIX}j{i}{ax_name}", type=mj.mjtJoint.mjJNT_HINGE,
                               axis=axis, stiffness=float(stiff[i]) * (c.out_of_plane_factor if ax_name == "y" else 1.0),
                               damping=float(damp[i]) * (c.out_of_plane_factor if ax_name == "y" else 1.0),
                               armature=c.armature)
            is_tip = i == c.n_segments - 1
            body.add_geom(
                name=f"{PREFIX}seg{i}_geom", type=mj.mjtGeom.mjGEOM_CAPSULE,
                fromto=(0, 0, 0, seg_len, 0, 0), size=(float(radius[i]), 0, 0),
                mass=float(mass[i]), contype=0, conaffinity=affinity, priority=1, condim=3,
                friction=(c.friction, cp.torsional_friction, cp.rolling_friction),
                solref=cp.get_solref_tuple(), solimp=cp.get_solimp_tuple(), margin=cp.margin,
                rgba=c.rgba_tip if is_tip else c.rgba_chain, group=1,
            )
            parent = body
        eq = spec.add_equality(name=PREFIX + "weld", type=mj.mjtEq.mjEQ_WELD,
                               objtype=mj.mjtObj.mjOBJ_BODY, name1=PREFIX + "handle",
                               name2=PREFIX + "grip")
        # anchor 0, relpose identity (grip and handle coincide), torquescale 1
        eq.data = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
        eq.solref = c.weld_solref

    # ------------------------------------------------------------------ attach
    def attach(self, sim: "Simulation") -> "Whip":
        if self._attached:
            return self
        m = sim.model
        self.sim = sim
        self.handle_body = m.body(PREFIX + "handle").id
        self.mocap_id = int(m.body_mocapid[self.handle_body])
        self.grip_body = m.body(PREFIX + "grip").id
        self.seg_bodies = np.array([m.body(f"{PREFIX}seg{i}").id
                                    for i in range(self.cfg.n_segments)])
        self.whip_geoms = np.array([m.geom(f"{PREFIX}seg{i}_geom").id
                                    for i in range(self.cfg.n_segments)])
        self.is_whip_geom = np.zeros(m.ngeom, bool)
        self.is_whip_geom[self.whip_geoms] = True
        self.is_fly_geom = np.asarray(m.body_rootid[m.geom_bodyid] == sim.thorax_body_id)
        self.body_weight_uN = sim.fly_mass * float(np.linalg.norm(m.opt.gravity))
        self._force6 = np.zeros(6)
        sim.pre_step_hooks.append(self._pre_step)
        sim.post_step_hooks.append(self._post_step)
        sim.reset_hooks.append(self._on_reset)
        self._attached = True
        self._on_reset(sim)
        return self

    def detach(self) -> None:
        if self._attached and self.sim is not None:
            self.sim.pre_step_hooks.remove(self._pre_step)
            self.sim.post_step_hooks.remove(self._post_step)
            self.sim.reset_hooks.remove(self._on_reset)
            self._attached = False

    # ------------------------------------------------------------------ fly state
    def _aim_point(self) -> np.ndarray:
        sim = self.sim
        d, tid = sim.data, sim.thorax_body_id
        yaw = sim.heading()
        ox, oy, oz = self.cfg.aim_offset
        c, s = math.cos(yaw), math.sin(yaw)
        p = d.xpos[tid]
        return np.array([p[0] + c * ox - s * oy, p[1] + s * ox + c * oy, p[2] + oz])

    def _on_reset(self, sim: "Simulation") -> None:
        # Abort any crack (no event: the sim state it measured is gone), re-seed the
        # trackers and snap the handle (and the welded grip) to the idle pose.
        self._crack = None
        self._pending = None
        self._anchor = self._aim_point()
        self._anchor_v = np.zeros(3)
        self._fly_v = np.zeros(3)
        self._prev_target = None
        self._yaw = sim.heading()
        pos, quat = self._idle_pose_world(self._anchor, self._yaw)
        self._set_handle(pos, quat)
        self._snap_grip(pos, quat)

    def _snap_grip(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """Teleport the grip (free joint) onto the handle pose, zero whip velocities."""
        sim = self.sim
        m, d = sim.model, sim.data
        j = m.joint(PREFIX + "grip_free").id
        a, v = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        d.qpos[a:a + 3] = pos
        d.qpos[a + 3:a + 7] = quat
        for i in range(self.cfg.n_segments):
            for ax in "yz":
                jj = m.joint(f"{PREFIX}j{i}{ax}").id
                d.qpos[m.jnt_qposadr[jj]] = 0.0
        dofs = np.flatnonzero(m.body_rootid[m.dof_bodyid] == self.grip_body)
        d.qvel[dofs] = 0.0
        mj.mj_kinematics(m, d)

    def _set_handle(self, pos: np.ndarray, quat: np.ndarray) -> None:
        d = self.sim.data
        d.mocap_pos[self.mocap_id] = pos
        d.mocap_quat[self.mocap_id] = quat

    # ------------------------------------------------------------------ API
    @property
    def busy(self) -> bool:
        """True while a crack is between request and the end of its strike window."""
        return self._crack is not None and self._crack.phase in STRIKE_BUSY

    @property
    def phase(self) -> str:
        return "idle" if self._crack is None else self._crack.phase

    def level_name(self, level: int) -> str:
        return self.cfg.levels[level - 1].name

    def crack(self, side: str = "random", level: int = 2, source: str = "api",
              rng: np.random.Generator | None = None) -> str:
        """Request a crack from ``side`` (relative to the fly heading) at ``level``.
        Starts now, or right after the current strike window. Returns "started" or
        "queued" (at most one crack is queued; a newer request replaces it)."""
        if side not in WHIP_SIDES:
            raise ValueError(f"unknown side {side!r}; expected one of {WHIP_SIDES}")
        if not 1 <= level <= len(self.cfg.levels):
            raise ValueError(f"level must be 1..{len(self.cfg.levels)}")
        if self.sim is None:
            raise RuntimeError("Whip.attach(sim) first")
        if self.busy:
            self._pending = (side, level, source, rng)
            return "queued"
        self._start(side, level, source, rng)
        return "started"

    # ------------------------------------------------------------------ planning
    def _start(self, side: str, level: int, source: str, rng) -> None:
        sim, c = self.sim, self.cfg
        rng = rng or self.rng
        yaw = sim.heading()
        cy, sy = math.cos(yaw), math.sin(yaw)
        X = np.array([cy, sy, 0.0])
        Y = np.array([-sy, cy, 0.0])
        Z = np.array([0.0, 0.0, 1.0])
        descent = math.radians(c.descent_deg)
        name = side
        if side == "random":
            az = rng.uniform(-math.pi, math.pi)
            s = math.cos(az) * X + math.sin(az) * Y  # push direction
            # pivot behind the fly (u . X >= 0) for a good camera view
            u_h = np.cross(Z, s)
            if np.dot(u_h, X) < 0:
                u_h = -u_h
            descent += math.radians(rng.uniform(*c.random_elevation_deg))
        elif side == "overhead":
            s = -Z
            u_h = X
            descent = 0.0
        else:
            s = {"left": -Y, "right": Y, "front": -X, "rear": X}[side]
            u_h = {"left": X, "right": X, "front": -Y, "rear": -Y}[side]
        if side == "overhead":
            u = u_h
        else:
            u = math.cos(descent) * u_h - math.sin(descent) * Z
        k = _unit(np.cross(u, s))
        pivot = -c.contact_radius * u  # relative to the anchor (aim point)
        pivot = pivot + np.array([0.0, 0.0, c.side_aim_dz.get(side, 0.0)])
        # swing angle profile (per physics step)
        dt = sim.timestep
        omega = c.levels[level - 1].omega * c.side_omega_scale.get(side, 1.0)
        # The handle stops at theta_stop < 0: the whip's rest line then passes
        # stop_mm from the aim point, just outside the fly's near surface, so only
        # the chain's dynamic overshoot (the lash) hits the fly. A stiff whip that
        # kept sweeping would bulldoze the fly instead of striking it.
        stop_mm = 0.0 if side == "overhead" else self._fly_extent(-s, X, Y) + c.stop_gap
        theta_stop = -math.asin(min(0.9, stop_mm / c.contact_radius))
        total = math.radians(c.back_angle_deg) + theta_stop
        dec = min(math.radians(c.swing_decel_deg), 0.5 * total)
        th, thetas, t = 0.0, [], 0.0
        while th < total and len(thetas) < 200000:
            ramp = min(1.0, t / c.swing_accel_s) if c.swing_accel_s > 0 else 1.0
            w = omega * (3 * ramp ** 2 - 2 * ramp ** 3)
            if total - th < dec:  # brake hard over the last swing_decel_deg
                w = min(w, omega * max(0.02, (total - th) / dec))
            th += w * dt
            t += dt
            thetas.append(min(th, total))
        theta = np.asarray(thetas) - math.radians(c.back_angle_deg)
        # raised pose above the pivot, whip pointing up and away from the fly
        up_dir = _unit(Z - c.up_tilt * u)
        up_pos = pivot + np.array([0.0, 0.0, c.lift])
        up_quat = _frame_quat(up_dir, s)
        start_pos = sim.data.mocap_pos[self.mocap_id] - self._anchor
        start_quat = sim.data.mocap_quat[self.mocap_id].copy()
        self._crack = _Crack(
            side=name, level=level, omega=omega, source=source, t_request=sim.time,
            s=s, u=u, k=k, pivot=pivot, start_pos=start_pos.copy(), start_quat=start_quat,
            up_pos=up_pos, up_quat=up_quat, theta=theta, t_phase=sim.time,
            lunge=c.levels[level - 1].lunge,
        )
        self.n_cracks += 1

    def _fly_extent(self, d: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
        """Horizontal distance (mm) from the aim point to the fly's surface in world
        direction ``d`` (elliptical blend of the front / rear / side extents)."""
        c = self.cfg
        dx, dy = float(np.dot(d, X)), float(np.dot(d, Y))
        n = math.hypot(dx, dy)
        if n < 1e-9:
            return 0.0
        dx, dy = dx / n, dy / n
        ax = c.extent_front if dx >= 0 else c.extent_rear
        return 1.0 / math.sqrt((dx / ax) ** 2 + (dy / c.extent_side) ** 2)

    def _swing_quat(self, cr: _Crack, theta: float) -> np.ndarray:
        w = _rot(cr.k, theta, cr.u)
        y = _rot(cr.k, theta, cr.s)
        return _frame_quat(w, y)

    # ------------------------------------------------------------------ hooks
    def _track(self, sim: "Simulation", extrapolate: bool, nsteps: int = 1) -> None:
        """Update the anchor (smoothed aim point) and the smoothed heading."""
        dt = sim.timestep * nsteps
        if extrapolate:
            self._anchor = self._anchor + self._anchor_v * dt
            return
        # Rate-limited follower: desired velocity = estimated fly velocity (no lag
        # while walking) + a proportional correction; |v| <= track_vmax and
        # |dv/dt| <= track_amax, so the mocap target never jumps (a jump would make
        # the weld yank the chain hard enough to blow it up), even when the fly has
        # just been knocked several mm away.
        c = self.cfg
        target = self._aim_point()
        if self._prev_target is not None:
            v_meas = (target - self._prev_target) / dt
            self._fly_v += (v_meas - self._fly_v) * min(1.0, dt / c.idle_tau_s)
        self._prev_target = target
        err = target - self._anchor
        v_des = self._fly_v + err * (1.0 / c.idle_tau_s)
        n = float(np.linalg.norm(v_des))
        if n > c.track_vmax:
            v_des *= c.track_vmax / n
        dv = v_des - self._anchor_v
        n = float(np.linalg.norm(dv))
        if n > c.track_amax * dt:
            dv *= c.track_amax * dt / n
        self._anchor_v = self._anchor_v + dv
        self._anchor = self._anchor + self._anchor_v * dt
        dyaw = (sim.heading() - self._yaw + math.pi) % (2 * math.pi) - math.pi
        self._yaw += dyaw * (dt / self.cfg.heading_tau_s)

    def _pre_step(self, sim: "Simulation") -> None:
        cr = self._crack
        if cr is None:
            # Idle: the handle drifts slowly, so update it every idle_update_steps
            # (saves ~20 us per physics step; the weld smooths the small jumps).
            n = self.cfg.idle_update_steps
            if sim.step_count % n == 0:
                self._track(sim, extrapolate=False, nsteps=n)
                self._set_handle(*self._idle_pose_world(self._anchor, self._yaw))
            return
        c = self.cfg
        in_swing = cr.phase in STRIKE_WINDOW
        self._track(sim, extrapolate=in_swing)
        A = self._anchor
        t = sim.time - cr.t_phase
        if cr.phase == "to_up":
            f = _smooth(t / c.t_to_up)
            pos = cr.start_pos + (cr.up_pos - cr.start_pos) * f
            quat = _slerp(cr.start_quat, cr.up_quat, f)
            if t >= c.t_to_up:
                self._next(cr, "down")
        elif cr.phase == "down":
            f = _smooth(t / c.t_down)
            cq = self._swing_quat(cr, cr.theta[0])
            cocked = cr.pivot - cr.lunge * cr.s
            pos = cr.up_pos + (cocked - cr.up_pos) * f
            quat = _slerp(cr.up_quat, cq, f)
            if t >= c.t_down:
                self._next(cr, "hold")
        elif cr.phase == "hold":
            pos, quat = cr.pivot - cr.lunge * cr.s, self._swing_quat(cr, cr.theta[0])
            if t >= c.t_hold:
                self._next(cr, "swing")
                cr.swing_t = sim.time
        elif cr.phase == "swing":
            i = min(cr.step_in_phase, len(cr.theta) - 1)
            th0, th1 = cr.theta[0], cr.theta[-1]
            prog = (cr.theta[i] - th0) / (th1 - th0) if th1 > th0 else 1.0
            pos = cr.pivot - cr.lunge * (1.0 - prog) * cr.s
            quat = self._swing_quat(cr, cr.theta[i])
            cr.step_in_phase += 1
            if cr.step_in_phase >= len(cr.theta):
                self._next(cr, "follow")
        elif cr.phase == "follow":
            pos, quat = cr.pivot, self._swing_quat(cr, cr.theta[-1])
            if t >= c.t_follow:
                self._next(cr, "lift")
        elif cr.phase == "lift":
            # Raise the whip right after the lash so it neither rests on the fly nor
            # springs back into it; contacts until the end of the lift still count
            # for this strike.
            f = _smooth(t / c.t_lift)
            eq = self._swing_quat(cr, cr.theta[-1])
            pos = cr.pivot + (cr.up_pos - cr.pivot) * f
            quat = _slerp(eq, cr.up_quat, f)
            if t >= c.t_lift:
                self._finish_strike(sim, cr)
                self._next(cr, "back")
                if self._pending is not None:
                    side, level, source, rng = self._pending
                    self._pending = None
                    self._start(side, level, source, rng)
                    self._pre_step_pose_only(sim)
                    return
        else:  # back
            f = _smooth(t / c.t_back)
            ipos, iquat = self._idle_pose_world(A, self._yaw)
            pos = cr.up_pos + (ipos - A - cr.up_pos) * f
            quat = _slerp(cr.up_quat, iquat, f)
            if t >= c.t_back:
                self._crack = None
        self._set_handle(A + pos, quat)

    def _pre_step_pose_only(self, sim: "Simulation") -> None:
        cr = self._crack
        self._set_handle(self._anchor + cr.start_pos, cr.start_quat)

    def _next(self, cr: _Crack, phase: str) -> None:
        cr.phase = phase
        cr.t_phase = self.sim.time
        cr.step_in_phase = 0

    def _post_step(self, sim: "Simulation") -> None:
        # Outside cracks, only sample for stray contacts every idle_update_steps.
        if self._crack is None and sim.step_count % self.cfg.idle_update_steps:
            return
        d = sim.data
        n = int(d.ncon)
        if n == 0:
            return
        g1 = d.contact.geom1[:n]
        g2 = d.contact.geom2[:n]
        w1, w2 = self.is_whip_geom[g1], self.is_whip_geom[g2]
        f1, f2 = self.is_fly_geom[g1], self.is_fly_geom[g2]
        sel = np.flatnonzero((w1 & f2) | (w2 & f1))
        if sel.size == 0:
            return
        m = sim.model
        total = np.zeros(3)
        per_body: dict[str, float] = {}
        f6 = self._force6
        for i in sel:
            mj.mj_contactForce(m, d, int(i), f6)
            frame = d.contact.frame[i].reshape(3, 3)
            fw = frame.T @ f6[:3]  # force on geom2 from geom1, world frame
            fly_geom = int(g2[i]) if f2[i] else int(g1[i])
            if not f2[i]:
                fw = -fw  # fly is geom1: the force on it is the reaction
            total += fw
            bname = m.body(int(m.geom_bodyid[fly_geom])).name
            per_body[bname] = per_body.get(bname, 0.0) + float(np.linalg.norm(fw))
        dt = sim.timestep
        cr = self._crack
        if cr is None or cr.phase not in STRIKE_WINDOW:
            self.stray_contact_steps += 1
            self.stray_impulse += float(np.linalg.norm(total)) * dt
            return
        cr.impulse += total * dt
        mag = float(np.linalg.norm(total))
        cr.peak = max(cr.peak, mag)
        if cr.first_t is None:
            cr.first_t = sim.time - dt
            cr.first_step = sim.step_count - 1
        cr.last_t = sim.time
        cr.n_contact_steps += 1
        for b, v in per_body.items():
            cr.bodies[b] = cr.bodies.get(b, 0.0) + v * dt

    def _finish_strike(self, sim: "Simulation", cr: _Crack) -> None:
        hit = cr.first_t is not None and float(np.linalg.norm(cr.impulse)) > 0
        imp = float(np.linalg.norm(cr.impulse))
        direction = tuple(float(v) for v in (cr.impulse / imp)) if imp > 0 else (0.0, 0.0, 0.0)
        dur = (cr.last_t - cr.first_t) if hit else 0.0
        body = max(cr.bodies, key=cr.bodies.get) if cr.bodies else ""
        ev = WhipHitEvent(
            sim_time=cr.first_t if hit else cr.swing_t,
            step=cr.first_step if hit else sim.step_count,
            source=cr.source, body=body, direction_name=f"from_{cr.side}",
            direction=direction, magnitude_uN=cr.peak,
            magnitude_bw=cr.peak / self.body_weight_uN,
            duration_s=dur, impulse_uNs=imp, level=cr.level, hit=hit, side=cr.side,
            impulse_vec=tuple(float(v) for v in cr.impulse),
            mean_force_uN=imp / dur if dur > 0 else 0.0,
            commanded_direction=tuple(float(v) for v in cr.s), omega=cr.omega,
            crack_time=cr.t_request, n_contact_steps=cr.n_contact_steps,
            bodies={k: float(v) for k, v in sorted(cr.bodies.items(), key=lambda kv: -kv[1])},
        )
        self.events.append(ev)
        for fn in list(self.listeners):
            fn(ev)

    # ------------------------------------------------------------------ queries
    def tip_position(self) -> np.ndarray:
        """World position of the tip end (mm)."""
        d = self.sim.data
        b = int(self.seg_bodies[-1])
        R = d.xmat[b].reshape(3, 3)
        return d.xpos[b] + R[:, 0] * (self.cfg.length / self.cfg.n_segments)

    def min_distance_to_fly(self, max_dist: float = 10.0) -> float:
        """Smallest distance between a whip capsule and a fly contact geom (mm)."""
        m, d = self.sim.model, self.sim.data
        fly = np.flatnonzero(self.is_fly_geom & (m.geom_contype != 0))
        best = max_dist
        ft = np.zeros(6)
        for wg in self.whip_geoms:
            for fg in fly:
                dist = mj.mj_geomDistance(m, d, int(wg), int(fg), best, ft)
                best = min(best, dist)
        return best
