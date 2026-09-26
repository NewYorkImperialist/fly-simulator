"""A physical flyswatter the fly can see coming and try to dodge (docs/SWATTER.md).

Model (added to the world ``MjSpec`` by ``Swatter.extension`` before ``add_fly``):

* ``swatter/hand``: a **mocap** body at the wrist (the swing pivot). No geoms.
* ``swatter/handle``: a **free** body (the handle, 30 mg) welded to the hand by a
  stiff ``weld`` equality, exactly like the whip's grip (a body parented to a mocap
  body would have no velocity and hit nothing with momentum; see docs/API_NOTES.md
  §14). Local frame: x along the handle (grip -> paddle), y = the swing axis, z =
  the paddle's upper face normal.
* ``swatter/paddle``: the paddle (flat plate + grid ribs + rim, 6 mg) on a stiff but
  flexible **neck hinge** (local y, stiffness ``neck_stiffness``) at the end of the
  handle, pre-bent by ``impact_tilt_deg`` so the plate lands flat. The flex limits
  the crushing force when the plate lands on the fly and makes fast swats whip a
  little, like a real plastic swatter. Only the plate collides: with the fly
  (``FLY_BIT``) and with the ground / terrain (``TERRAIN_BIT``), stiff FlyGym contact
  parameters, ``priority 1``. The grid ribs, rim and handle are visual only.

A *swat* is scripted motion of the hand only (the fly moves exclusively through
contact forces). Phases (sim time, see ``SwatterConfig`` / ``SwatLevel``):
``raise`` (from the parked pose to the raised pose: handle ``raise_deg`` above
horizontal, the paddle high above and behind the fly) -> ``pause`` (the telegraph:
the raised paddle hovers, tracking the fly's *predicted* position) -> ``slam`` (the
hand rotates the handle about the swing axis, accelerating all the way, until the
plate is flat on the ground at the aim point; the aim is frozen at slam start) ->
``press`` (held ``overshoot_deg`` past flat, the neck flexes) -> ``lift`` -> ``back``
(to the parked pose). The strike window (contacts that count) is slam + press + lift.

Aim: the ground point under the thorax + fly velocity x (remaining pause + slam
time), i.e. where the fly will be if it keeps walking (``lead``). A fly that stops,
turns or jumps after the aim is frozen can therefore escape.

Measurement (the whip's pattern): a post-step hook sums the world-frame contact
forces the plate exerts on fly geoms x dt into the swat's impulse vector (peak force,
contact time, bodies hit) and separately the plate-ground contact (the slap). At the
end of the lift a ``SwatEvent`` goes to ``listeners``: ``outcome`` = "hit", "grazed"
(a hit below ``graze_max_uNs`` while the fly was escaping: the plate caught a leg or
the abdomen of a fly already on its way), "dodged" (no hit, and the fly jumped /
moved out of the paddle's footprint after the aim was frozen) or "miss" (no hit,
the fly was never under the plate).

Idle: the hand follows a rate-limited copy of the fly pose (never teleported: the
weld would yank the handle), parked behind-right of the fly with the swatter held
up high, clear of the fly.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Callable

import mujoco as mj
import numpy as np

from perpetualfly.interaction.whip import _frame_quat, _slerp, _smooth
from perpetualfly.terrain import FLY_BIT, TERRAIN_BIT

if TYPE_CHECKING:
    from flygym.compose import BaseWorld

    from perpetualfly.simulation import Simulation

PREFIX = "swatter/"
SWAT_SIDES = ("rear", "left", "right", "front", "random")
SWAT_BUSY = ("raise", "pause", "slam", "press", "lift")
SWAT_WINDOW = ("slam", "press", "lift")
Z = np.array([0.0, 0.0, 1.0])

# suggested app keys (integration later; app.py is not touched here)
SWATTER_KEYS = {
    "v": "swatter: swat at the fly (from behind)",
    "shift+v": "swatter: swat from a random side",
    "1-4": "swatter strength (with swatter mode): lazy / normal / quick / lightning",
}


@dataclass
class SwatLevel:
    name: str
    slam_s: float  # duration of the downswing (raised -> plate flat on the ground)
    pause_s: float  # telegraph: raised paddle hovers this long before the slam


def _default_levels() -> list[SwatLevel]:
    # Tuned with scripts/demo_swatter.py (docs/SWATTER.md): slow, telegraphed swats
    # are usually dodged by a brain-driven escape, the fastest one usually lands.
    return [
        SwatLevel("lazy", 0.50, 0.40),
        SwatLevel("normal", 0.36, 0.25),
        SwatLevel("quick", 0.22, 0.15),
        SwatLevel("lightning", 0.07, 0.08),
    ]


@dataclass
class SwatterConfig:
    enabled: bool = True
    # ---- geometry (mm) / mass (g) -------------------------------------------
    handle_length: float = 14.0
    handle_radius: float = 0.22
    handle_mass: float = 0.03
    neck_length: float = 1.0  # handle end -> near edge of the plate
    paddle_length: float = 8.0  # along the handle
    paddle_width: float = 7.0
    paddle_thickness: float = 0.15
    paddle_mass: float = 0.006
    grid: tuple[int, int] = (7, 6)  # visual grid ribs along / across
    rib_width: float = 0.12
    # ---- neck joint (paddle flex) ---------------------------------------------
    neck_stiffness: float = 3.0e4  # uN*mm/rad
    neck_damping: float = 40.0  # uN*mm*s/rad
    neck_armature: float = 1e-3
    neck_range_deg: float = 45.0
    weld_solref: tuple[float, float] = (4e-4, 1.0)
    # ---- contacts --------------------------------------------------------------
    friction: float = 0.6
    min_hit_impulse_uNs: float = 0.02  # less than this on the fly = a brush, not a hit
    # a hit below this while the fly was escaping (jumped / left the footprint) is a
    # "grazed" outcome (a full swat on a standing fly is ~100-800 uN*s)
    graze_max_uNs: float = 30.0
    # ---- look -----------------------------------------------------------------
    rgba_plate: tuple[float, float, float, float] = (0.92, 0.12, 0.10, 0.55)  # red mesh
    rgba_rib: tuple[float, float, float, float] = (0.80, 0.06, 0.05, 1.0)
    rgba_rim: tuple[float, float, float, float] = (0.95, 0.80, 0.05, 1.0)  # yellow rim
    rgba_handle: tuple[float, float, float, float] = (0.98, 0.84, 0.10, 1.0)  # yellow
    # ---- swing geometry --------------------------------------------------------
    impact_tilt_deg: float = 15.0  # handle below horizontal when the plate is flat
    overshoot_deg: float = 3.0  # hand keeps rotating this far past flat (press)
    raise_deg: float = 75.0  # raised (telegraph) handle elevation
    lift_deg: float = 55.0  # after the slam
    aim_offset: tuple[float, float] = (-0.3, 0.0)  # plate centre vs thorax (heading frame)
    lead: bool = True  # aim at the predicted position (fly velocity x time to impact)
    lead_max_mm: float = 6.0
    # ---- parked pose (fly heading frame: x forward, y left, z up; mm) ----------
    park_offset: tuple[float, float, float] = (-17.0, -9.0, 9.0)  # hand position
    park_yaw_deg: float = 30.0  # handle direction, relative to the heading
    park_deg: float = 70.0  # handle elevation
    track_tau_s: float = 0.12
    heading_tau_s: float = 0.4
    vel_tau_s: float = 0.15  # fly velocity estimate (for the lead)
    track_vmax: float = 300.0  # mm/s
    track_amax: float = 2e4  # mm/s^2
    idle_update_steps: int = 10
    # ---- timing (sim s) --------------------------------------------------------
    t_raise: float = 0.30
    t_press: float = 0.03
    t_lift: float = 0.12
    t_back: float = 0.30
    slam_power: float = 2.0  # angle ~ (t / slam_s)^power: accelerating downswing
    levels: list[SwatLevel] = field(default_factory=_default_levels)
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SwatterConfig":
        d = dict(d)
        if "levels" in d:
            d["levels"] = [lv if isinstance(lv, SwatLevel) else SwatLevel(**lv)
                           for lv in d["levels"]]
        for k in ("grid", "weld_solref", "rgba_plate", "rgba_rib", "rgba_rim",
                  "rgba_handle", "aim_offset", "park_offset"):
            if k in d and isinstance(d[k], list):
                d[k] = tuple(d[k])
        return cls(**d)

    @property
    def reach(self) -> float:
        """Pivot -> plate centre along the plate direction beyond the handle (mm)."""
        return self.neck_length + 0.5 * self.paddle_length


@dataclass
class SwatEvent:
    """Result of one swat (field names follow ``WhipHitEvent`` where they overlap)."""

    sim_time: float  # s: first plate-fly contact (hit) or plate-ground contact / slam end
    step: int
    source: str
    level: int
    level_name: str
    side: str
    hit: bool
    outcome: str  # "hit" | "grazed" | "dodged" | "miss"
    body: str = ""
    impulse_uNs: float = 0.0
    impulse_vec: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    magnitude_uN: float = 0.0  # peak force on the fly
    magnitude_bw: float = 0.0
    duration_s: float = 0.0
    n_contact_steps: int = 0
    bodies: dict = field(default_factory=dict)
    t_request: float = 0.0
    t_slam: float = 0.0  # slam start (aim frozen)
    t_ground: float | None = None  # first plate-ground contact (the slap)
    t_fly_contact: float | None = None
    ground_impulse_uNs: float = 0.0
    impact_speed_mm_s: float = 0.0  # plate centre speed at the first contact
    aim: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fly_at_slam: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fly_at_impact: tuple[float, float, float] = (0.0, 0.0, 0.0)
    under_at_slam: bool = False  # thorax inside the plate footprint (predicted) at slam start
    under_at_impact: bool = False  # thorax inside the footprint when the plate landed
    jumped: bool = False  # an escape jump started during the swat
    kind: str = "swatter"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Swat:
    side: str
    level: int
    source: str
    t_request: float
    a: np.ndarray  # approach direction (horizontal, pivot -> aim)
    start_pos: np.ndarray  # hand pose at swat start relative to the anchor
    start_quat: np.ndarray
    slam_s: float
    pause_s: float
    phase: str = "raise"
    t_phase: float = 0.0
    aim: np.ndarray | None = None  # frozen at slam start (world)
    pivot: np.ndarray | None = None  # world, frozen at slam start
    t_slam: float = 0.0
    fly_at_slam: np.ndarray | None = None
    under_at_slam: bool = False
    jumped: bool = False
    # measurement
    impulse: np.ndarray = field(default_factory=lambda: np.zeros(3))
    peak: float = 0.0
    first_t: float | None = None
    first_step: int = 0
    last_t: float | None = None
    n_contact_steps: int = 0
    bodies: dict = field(default_factory=dict)
    t_ground: float | None = None
    ground_impulse: float = 0.0
    impact_speed: float = 0.0
    fly_at_impact: np.ndarray | None = None
    under_at_impact: bool = False
    lift_rel: np.ndarray | None = None


class Swatter:
    """Physical flyswatter. ``Simulation(cfg, world_extensions=[sw.extension])`` (or
    ``Session(cfg, world_extensions=[sw.extension])``), then ``sw.attach(sim)``;
    ``sw.swat(level=2)`` starts a swat."""

    def __init__(self, cfg: SwatterConfig | None = None) -> None:
        self.cfg = cfg or SwatterConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.sim: "Simulation | None" = None
        self.listeners: list[Callable[[SwatEvent], None]] = []
        self.events: list[SwatEvent] = []
        self.n_swats = 0
        self.stray_contact_steps = 0
        # optional: () -> bool, "the fly has started an escape jump" (install_swatter
        # wires it to the ActionManager); polled during the swat
        self.jump_probe: Callable[[], bool] | None = None
        # optional: (x, y) -> ground height (procedural terrain); default 0
        self.ground_height_fn: Callable[[float, float], float] | None = None
        self._swat: _Swat | None = None
        self._pending: tuple | None = None
        self._attached = False

    # ------------------------------------------------------------------ model
    def extension(self, world: "BaseWorld") -> None:
        """World extension: adds the swatter bodies to ``world.mjcf_root``."""
        from flygym.compose import ContactParams

        c = self.cfg
        spec: mj.MjSpec = world.mjcf_root
        cp = ContactParams()
        pos, quat = self._park_pose(np.zeros(3), 0.0)
        spec.worldbody.add_body(name=PREFIX + "hand", pos=pos, quat=quat, mocap=True)
        handle = spec.worldbody.add_body(name=PREFIX + "handle", pos=pos, quat=quat)
        handle.add_freejoint(name=PREFIX + "handle_free")
        handle.add_geom(name=PREFIX + "handle_geom", type=mj.mjtGeom.mjGEOM_CAPSULE,
                        fromto=(0, 0, 0, c.handle_length, 0, 0), size=(c.handle_radius, 0, 0),
                        mass=c.handle_mass, contype=0, conaffinity=0, rgba=c.rgba_handle,
                        group=1)
        # paddle frame: pre-bent up by impact_tilt about the local y axis
        tilt = math.radians(c.impact_tilt_deg)
        q = np.array([math.cos(-tilt / 2), 0.0, math.sin(-tilt / 2), 0.0])
        paddle = handle.add_body(name=PREFIX + "paddle", pos=(c.handle_length, 0, 0), quat=q)
        rng = math.radians(c.neck_range_deg)
        paddle.add_joint(name=PREFIX + "neck", type=mj.mjtJoint.mjJNT_HINGE, axis=(0, 1, 0),
                         stiffness=c.neck_stiffness, damping=c.neck_damping,
                         armature=c.neck_armature, range=(-rng, rng), limited=True)
        L, W, T = c.paddle_length, c.paddle_width, c.paddle_thickness
        x0 = c.neck_length
        xc = x0 + 0.5 * L
        paddle.add_geom(name=PREFIX + "neck_geom", type=mj.mjtGeom.mjGEOM_CAPSULE,
                        fromto=(0, 0, 0, x0 + 0.3, 0, 0), size=(0.8 * c.handle_radius, 0, 0),
                        mass=1e-4, contype=0, conaffinity=0, rgba=c.rgba_handle, group=1)
        paddle.add_geom(
            name=PREFIX + "plate", type=mj.mjtGeom.mjGEOM_BOX, pos=(xc, 0, 0),
            size=(0.5 * L, 0.5 * W, 0.5 * T), mass=c.paddle_mass, contype=0,
            conaffinity=FLY_BIT | TERRAIN_BIT, priority=1, condim=3,
            friction=(c.friction, cp.torsional_friction, cp.rolling_friction),
            solref=cp.get_solref_tuple(), solimp=cp.get_solimp_tuple(), margin=cp.margin,
            rgba=c.rgba_plate, group=1)
        # visual grid ribs (slightly thicker than the plate: visible from both sides)
        rt = 0.5 * T * 1.6
        nx, ny = c.grid
        rw = 0.5 * c.rib_width
        for i in range(1, nx):
            x = x0 + L * i / nx
            paddle.add_geom(name=f"{PREFIX}rib_x{i}", type=mj.mjtGeom.mjGEOM_BOX, pos=(x, 0, 0),
                            size=(rw, 0.5 * W, rt), mass=0, contype=0, conaffinity=0,
                            rgba=c.rgba_rib, group=1)
        for j in range(1, ny):
            y = -0.5 * W + W * j / ny
            paddle.add_geom(name=f"{PREFIX}rib_y{j}", type=mj.mjtGeom.mjGEOM_BOX, pos=(xc, y, 0),
                            size=(0.5 * L, rw, rt), mass=0, contype=0, conaffinity=0,
                            rgba=c.rgba_rib, group=1)
        rim = 2.0 * rw
        for k, (p, s) in enumerate((((x0, 0, 0), (rim, 0.5 * W + rim, rt * 1.2)),
                                    ((x0 + L, 0, 0), (rim, 0.5 * W + rim, rt * 1.2)),
                                    ((xc, 0.5 * W, 0), (0.5 * L, rim, rt * 1.2)),
                                    ((xc, -0.5 * W, 0), (0.5 * L, rim, rt * 1.2)))):
            paddle.add_geom(name=f"{PREFIX}rim{k}", type=mj.mjtGeom.mjGEOM_BOX, pos=p, size=s,
                            mass=0, contype=0, conaffinity=0, rgba=c.rgba_rim, group=1)
        eq = spec.add_equality(name=PREFIX + "weld", type=mj.mjtEq.mjEQ_WELD,
                               objtype=mj.mjtObj.mjOBJ_BODY, name1=PREFIX + "hand",
                               name2=PREFIX + "handle")
        eq.data = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
        eq.solref = c.weld_solref

    # ------------------------------------------------------------------ attach
    def attach(self, sim: "Simulation") -> "Swatter":
        if self._attached:
            return self
        m = sim.model
        self.sim = sim
        self.hand_body = m.body(PREFIX + "hand").id
        self.mocap_id = int(m.body_mocapid[self.hand_body])
        self.handle_body = m.body(PREFIX + "handle").id
        self.paddle_body = m.body(PREFIX + "paddle").id
        self.plate_geom = m.geom(PREFIX + "plate").id
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

    # ------------------------------------------------------------------ geometry
    def _frame(self, a: np.ndarray, phi: float) -> np.ndarray:
        """Hand quaternion: handle direction at elevation ``phi`` along approach ``a``,
        y = swing axis (Z x a)."""
        x = math.cos(phi) * a + math.sin(phi) * Z
        return _frame_quat(x, np.cross(Z, a))

    def _park_pose(self, anchor: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        cy, sy = math.cos(yaw), math.sin(yaw)
        ox, oy, oz = c.park_offset
        pos = anchor + np.array([cy * ox - sy * oy, sy * ox + cy * oy, oz])
        ya = yaw + math.radians(c.park_yaw_deg)
        a = np.array([math.cos(ya), math.sin(ya), 0.0])
        return pos, self._frame(a, math.radians(c.park_deg))

    def _pivot_for(self, aim: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Hand position that puts the plate flat on the ground centred on ``aim``
        (aim z = ground height) at the impact elevation."""
        c = self.cfg
        t = math.radians(c.impact_tilt_deg)
        horiz = c.handle_length * math.cos(t) + c.reach
        return aim - horiz * a + Z * (0.5 * c.paddle_thickness + c.handle_length * math.sin(t))

    def _ground(self, x: float, y: float) -> float:
        return float(self.ground_height_fn(x, y)) if self.ground_height_fn else 0.0

    def _aim_point(self, lead_s: float = 0.0) -> np.ndarray:
        """Ground point under the thorax (+ aim_offset), + fly velocity x ``lead_s``."""
        sim = self.sim
        yaw = sim.heading()
        c, s = math.cos(yaw), math.sin(yaw)
        ox, oy = self.cfg.aim_offset
        p = sim.data.xpos[sim.thorax_body_id]
        q = np.array([p[0] + c * ox - s * oy, p[1] + s * ox + c * oy, 0.0])
        if lead_s > 0 and self.cfg.lead:
            d = self._fly_v.copy()
            d[2] = 0.0
            d *= lead_s
            n = float(np.linalg.norm(d))
            if n > self.cfg.lead_max_mm:
                d *= self.cfg.lead_max_mm / n
            q = q + d
        q[2] = self._ground(q[0], q[1])
        return q

    def paddle_quads(self) -> np.ndarray:
        """(1, 4, 3) world corners of the plate's mid-plane (for the looming vision)."""
        c = self.cfg
        d = self.sim.data
        g = self.plate_geom
        R = d.geom_xmat[g].reshape(3, 3)
        ctr = d.geom_xpos[g]
        u = R[:, 0] * (0.5 * c.paddle_length)
        v = R[:, 1] * (0.5 * c.paddle_width)
        return np.array([[ctr - u - v, ctr + u - v, ctr + u + v, ctr - u + v]])

    def plate_center(self) -> np.ndarray:
        return self.sim.data.geom_xpos[self.plate_geom].copy()

    def footprint_contains(self, p: np.ndarray, aim: np.ndarray, a: np.ndarray,
                           margin: float = 0.0) -> bool:
        """Horizontal point ``p`` inside the plate's landing rectangle at ``aim``."""
        c = self.cfg
        d = np.asarray(p, float)[:2] - aim[:2]
        along = float(d @ a[:2])
        across = float(d @ np.array([-a[1], a[0]]))
        return (abs(along) <= 0.5 * c.paddle_length + margin
                and abs(across) <= 0.5 * c.paddle_width + margin)

    # ------------------------------------------------------------------ state
    def _on_reset(self, sim: "Simulation") -> None:
        self._swat = None
        self._pending = None
        self._fly_v = np.zeros(3)
        self._prev_p = None
        self._anchor = self._aim_point()
        self._anchor_v = np.zeros(3)
        self._yaw = sim.heading()
        pos, quat = self._park_pose(self._anchor, self._yaw)
        self._set_hand(pos, quat)
        self._snap_handle(pos, quat)

    def _snap_handle(self, pos: np.ndarray, quat: np.ndarray) -> None:
        sim = self.sim
        m, d = sim.model, sim.data
        j = m.joint(PREFIX + "handle_free").id
        a = int(m.jnt_qposadr[j])
        d.qpos[a:a + 3] = pos
        d.qpos[a + 3:a + 7] = quat
        d.qpos[m.jnt_qposadr[m.joint(PREFIX + "neck").id]] = 0.0
        dofs = np.flatnonzero(m.body_rootid[m.dof_bodyid] == self.handle_body)
        d.qvel[dofs] = 0.0
        mj.mj_kinematics(m, d)

    def _set_hand(self, pos: np.ndarray, quat: np.ndarray) -> None:
        d = self.sim.data
        d.mocap_pos[self.mocap_id] = pos
        d.mocap_quat[self.mocap_id] = quat

    # ------------------------------------------------------------------ API
    @property
    def busy(self) -> bool:
        return self._swat is not None and self._swat.phase in SWAT_BUSY

    @property
    def phase(self) -> str:
        return "idle" if self._swat is None else self._swat.phase

    def level_name(self, level: int) -> str:
        return self.cfg.levels[level - 1].name

    def swat(self, level: int = 2, side: str = "rear", source: str = "api",
             rng: np.random.Generator | None = None) -> str:
        """Request a swat at ``level`` (1..4) with the handle coming from ``side`` of
        the fly (relative to its heading). Returns "started" or "queued"."""
        if side not in SWAT_SIDES:
            raise ValueError(f"unknown side {side!r}; expected one of {SWAT_SIDES}")
        if not 1 <= level <= len(self.cfg.levels):
            raise ValueError(f"level must be 1..{len(self.cfg.levels)}")
        if self.sim is None:
            raise RuntimeError("Swatter.attach(sim) first")
        if self.busy:
            self._pending = (level, side, source, rng)
            return "queued"
        self._start(level, side, source, rng)
        return "started"

    def _start(self, level: int, side: str, source: str, rng) -> None:
        sim = self.sim
        rng = rng or self.rng
        yaw = sim.heading()
        off = {"rear": 0.0, "left": -0.5 * math.pi, "right": 0.5 * math.pi,
               "front": math.pi}.get(side)
        if off is None:
            off = float(rng.uniform(-math.pi, math.pi))
        # approach direction = pivot -> aim; "rear" = the handle comes from behind
        a = np.array([math.cos(yaw + off), math.sin(yaw + off), 0.0])
        lv = self.cfg.levels[level - 1]
        self._swat = _Swat(side=side, level=level, source=source, t_request=sim.time, a=a,
                           start_pos=sim.data.mocap_pos[self.mocap_id] - self._anchor,
                           start_quat=sim.data.mocap_quat[self.mocap_id].copy(),
                           slam_s=lv.slam_s, pause_s=lv.pause_s, t_phase=sim.time)
        self.n_swats += 1

    # ------------------------------------------------------------------ hooks
    def _track(self, sim: "Simulation", nsteps: int, lead_s: float) -> None:
        """Rate-limited follower of the (predicted) aim point (the whip's tracker)."""
        c = self.cfg
        dt = sim.timestep * nsteps
        p = sim.data.xpos[sim.thorax_body_id].copy()
        if self._prev_p is not None:
            v = (p - self._prev_p) / dt
            self._fly_v += (v - self._fly_v) * min(1.0, dt / c.vel_tau_s)
        self._prev_p = p
        target = self._aim_point(lead_s)
        err = target - self._anchor
        v_des = self._fly_v * np.array([1.0, 1.0, 0.0]) + err / c.track_tau_s
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
        self._yaw += dyaw * min(1.0, dt / c.heading_tau_s)

    def _raised_pose(self, sw: _Swat) -> tuple[np.ndarray, np.ndarray]:
        """Raised hand pose relative to the anchor (the tracked predicted aim)."""
        piv = self._pivot_for(np.zeros(3), sw.a)
        return piv, self._frame(sw.a, math.radians(self.cfg.raise_deg))

    def _pre_step(self, sim: "Simulation") -> None:
        sw = self._swat
        c = self.cfg
        if sw is None:
            n = c.idle_update_steps
            if sim.step_count % n == 0:
                self._track(sim, n, 0.0)
                self._set_hand(*self._park_pose(self._anchor, self._yaw))
            return
        t = sim.time - sw.t_phase
        if sw.phase in ("raise", "pause"):
            # hover over where the fly will be when the plate lands
            left = (c.t_raise - t if sw.phase == "raise" else 0.0) + max(
                0.0, sw.pause_s - (t if sw.phase == "pause" else 0.0)) + sw.slam_s
            self._track(sim, 1, left)
        elif sw.phase == "back":
            self._track(sim, 1, 0.0)
        if self.jump_probe is not None and sw.phase in SWAT_BUSY and not sw.jumped:
            sw.jumped = bool(self.jump_probe())
        A = self._anchor
        if sw.phase == "raise":
            f = _smooth(t / c.t_raise)
            rpos, rquat = self._raised_pose(sw)
            pos = A + sw.start_pos + (rpos - sw.start_pos) * f
            quat = _slerp(sw.start_quat, rquat, f)
            if t >= c.t_raise:
                self._next(sw, "pause")
        elif sw.phase == "pause":
            rpos, quat = self._raised_pose(sw)
            pos = A + rpos
            if t >= sw.pause_s:
                self._next(sw, "slam")
                sw.aim = A.copy()
                sw.aim[2] = self._ground(A[0], A[1])
                sw.pivot = self._pivot_for(sw.aim, sw.a)
                pos = sw.pivot  # == A + rpos up to the ground height
                sw.t_slam = sim.time
                sw.fly_at_slam = sim.thorax_position()
                sw.under_at_slam = self.footprint_contains(
                    sw.fly_at_slam + self._fly_v * sw.slam_s, sw.aim, sw.a)
        elif sw.phase == "slam":
            u = min(1.0, t / sw.slam_s)
            f = u ** c.slam_power
            p0 = math.radians(c.raise_deg)
            p1 = -math.radians(c.impact_tilt_deg + c.overshoot_deg)
            pos, quat = sw.pivot, self._frame(sw.a, p0 + (p1 - p0) * f)
            if t >= sw.slam_s:
                self._next(sw, "press")
        elif sw.phase == "press":
            p1 = -math.radians(c.impact_tilt_deg + c.overshoot_deg)
            pos, quat = sw.pivot, self._frame(sw.a, p1)
            if t >= c.t_press:
                self._next(sw, "lift")
        elif sw.phase == "lift":
            f = _smooth(t / c.t_lift)
            p1 = -math.radians(c.impact_tilt_deg + c.overshoot_deg)
            p2 = math.radians(c.lift_deg)
            pos, quat = sw.pivot, self._frame(sw.a, p1 + (p2 - p1) * f)
            if t >= c.t_lift:
                self._finish(sim, sw)
                self._next(sw, "back")
                sw.lift_rel = sw.pivot - self._anchor
                if self._pending is not None:
                    level, side, source, rng = self._pending
                    self._pending = None
                    self._start(level, side, source, rng)
                    self._swat.start_pos = sw.pivot - self._anchor
                    self._swat.start_quat = quat
                    self._set_hand(pos, quat)
                    return
        else:  # back
            f = _smooth(t / c.t_back)
            ppos, pquat = self._park_pose(A, self._yaw)
            lq = self._frame(sw.a, math.radians(c.lift_deg))
            pos = A + sw.lift_rel + (ppos - A - sw.lift_rel) * f
            quat = _slerp(lq, pquat, f)
            if t >= c.t_back:
                self._swat = None
        self._set_hand(pos, quat)

    def _next(self, sw: _Swat, phase: str) -> None:
        sw.phase = phase
        sw.t_phase = self.sim.time

    def _post_step(self, sim: "Simulation") -> None:
        sw = self._swat
        if sw is None and sim.step_count % self.cfg.idle_update_steps:
            return
        d = sim.data
        n = int(d.ncon)
        if n == 0:
            return
        g1 = d.contact.geom1[:n]
        g2 = d.contact.geom2[:n]
        pg = self.plate_geom
        sel = np.flatnonzero((g1 == pg) | (g2 == pg))
        if sel.size == 0:
            return
        m = sim.model
        dt = sim.timestep
        f6 = self._force6
        fly_total = np.zeros(3)
        ground_total = 0.0
        per_body: dict[str, float] = {}
        for i in sel:
            other = int(g2[i]) if g1[i] == pg else int(g1[i])
            mj.mj_contactForce(m, d, int(i), f6)
            fw = d.contact.frame[i].reshape(3, 3).T @ f6[:3]  # on geom2 from geom1
            if g1[i] != pg:
                fw = -fw  # force on `other` from the plate
            if self.is_fly_geom[other]:
                fly_total += fw
                bname = m.body(int(m.geom_bodyid[other])).name
                per_body[bname] = per_body.get(bname, 0.0) + float(np.linalg.norm(fw))
            else:
                ground_total += float(np.linalg.norm(fw))
        in_window = sw is not None and sw.phase in SWAT_WINDOW
        if not in_window:
            if per_body:
                self.stray_contact_steps += 1
            return
        if ground_total > 0:
            if sw.t_ground is None:
                sw.t_ground = sim.time - dt
                self._note_impact(sim, sw)
            sw.ground_impulse += ground_total * dt
        if per_body:
            sw.impulse += fly_total * dt
            sw.peak = max(sw.peak, float(np.linalg.norm(fly_total)))
            if sw.first_t is None:
                sw.first_t = sim.time - dt
                sw.first_step = sim.step_count - 1
                if sw.t_ground is None:
                    self._note_impact(sim, sw)
            sw.last_t = sim.time
            sw.n_contact_steps += 1
            for b, v in per_body.items():
                sw.bodies[b] = sw.bodies.get(b, 0.0) + v * dt

    def _note_impact(self, sim: "Simulation", sw: _Swat) -> None:
        """First plate contact (ground or fly): plate speed, fly position."""
        if sw.fly_at_impact is not None:
            return
        v = np.zeros(6)
        mj.mj_objectVelocity(sim.model, sim.data, mj.mjtObj.mjOBJ_GEOM, self.plate_geom, v, 0)
        sw.impact_speed = float(np.linalg.norm(v[3:]))
        sw.fly_at_impact = sim.thorax_position()
        sw.under_at_impact = self.footprint_contains(sw.fly_at_impact, sw.aim, sw.a)

    def _finish(self, sim: "Simulation", sw: _Swat) -> None:
        c = self.cfg
        imp = float(np.linalg.norm(sw.impulse))
        hit = sw.first_t is not None and imp >= c.min_hit_impulse_uNs
        if sw.fly_at_impact is None:  # never touched anything (e.g. over a pit)
            sw.fly_at_impact = sim.thorax_position()
            sw.under_at_impact = self.footprint_contains(sw.fly_at_impact, sw.aim, sw.a)
        escaped = sw.jumped or (sw.under_at_slam and not sw.under_at_impact)
        if hit:
            outcome = "grazed" if escaped and imp < c.graze_max_uNs else "hit"
        elif escaped:
            outcome = "dodged"
        else:
            outcome = "miss"
        dur = (sw.last_t - sw.first_t) if sw.first_t is not None else 0.0
        t_ev = sw.first_t if hit else (sw.t_ground if sw.t_ground is not None
                                       else sw.t_slam + sw.slam_s)
        ev = SwatEvent(
            sim_time=float(t_ev), step=sw.first_step if hit else sim.step_count,
            source=sw.source, level=sw.level, level_name=self.level_name(sw.level),
            side=sw.side, hit=hit, outcome=outcome,
            body=max(sw.bodies, key=sw.bodies.get) if sw.bodies else "",
            impulse_uNs=imp, impulse_vec=tuple(float(v) for v in sw.impulse),
            direction=tuple(float(v) for v in sw.impulse / imp) if imp > 0 else (0.0, 0.0, 0.0),
            magnitude_uN=sw.peak, magnitude_bw=sw.peak / self.body_weight_uN,
            duration_s=dur, n_contact_steps=sw.n_contact_steps,
            bodies={k: float(v) for k, v in sorted(sw.bodies.items(), key=lambda kv: -kv[1])},
            t_request=sw.t_request, t_slam=sw.t_slam, t_ground=sw.t_ground,
            t_fly_contact=sw.first_t, ground_impulse_uNs=sw.ground_impulse,
            impact_speed_mm_s=sw.impact_speed, aim=tuple(float(v) for v in sw.aim),
            fly_at_slam=tuple(float(v) for v in sw.fly_at_slam),
            fly_at_impact=tuple(float(v) for v in sw.fly_at_impact),
            under_at_slam=sw.under_at_slam, under_at_impact=sw.under_at_impact,
            jumped=sw.jumped)
        self.events.append(ev)
        for fn in list(self.listeners):
            fn(ev)


# ---------------------------------------------------------------------------
# vision + brain + app integration
# ---------------------------------------------------------------------------


def swatter_response():
    """LC4 / LPLC2 response for the paddle (``LoomResponse``; docs/SWATTER.md).

    The whip's response (``LoomResponse()``) only fires above 12000 deg/s: tuned for
    a millisecond lash. A swatter paddle is big and comparatively slow (its angular
    expansion reaches ~500-5000 deg/s in the last 50-150 ms), which is the regime of
    the looming stimuli used to characterise LC4 / LPLC2 and the giant fiber (von
    Reyn et al. 2017; Ache et al. 2019: l/v 10-80 ms, GF-driven take-offs at angular
    sizes of ~20-60 deg). LC4 is gated to theta >= 12 deg so the jitter of a thin,
    far paddle at the edge of the dorsal / rear field of view does not read as
    looming."""
    from perpetualfly.vision.looming import LoomResponse

    return LoomResponse(lc4_v0=300.0, lc4_vscale=1500.0, lc4_theta0=12.0,
                        lplc2_theta0=18.0, lplc2_theta1=55.0,
                        lplc2_gate_v0=150.0, lplc2_gate_v1=600.0)


def swatter_source(swatter: Swatter, response=None):
    """The paddle as a flat visual source (``VisualSource.quads``). Cadence: 0.5 ms
    during the slam / press, 2 ms while raised or returning, idle otherwise."""
    from perpetualfly.vision.looming import VisualSource

    def period():
        ph = swatter.phase
        if ph in ("slam", "press"):
            return 5e-4
        return None if ph == "idle" else 2e-3

    return VisualSource("swatter", quads=swatter.paddle_quads, period=period,
                        response=response or swatter_response())


def format_swat_event(ev: SwatEvent) -> str:
    head = f"[swatter] L{ev.level} {ev.level_name} from {ev.side}: {ev.outcome.upper()}"
    if ev.hit:
        return (f"{head} {ev.body.split('/')[-1]} impulse {ev.impulse_uNs:.2f} uN*s, "
                f"peak {ev.magnitude_bw:.0f} BW, plate {ev.impact_speed_mm_s:.0f} mm/s")
    if ev.outcome == "grazed":
        return f"{head} {ev.body.split('/')[-1]} impulse {ev.impulse_uNs:.2f} uN*s (escaping)"
    if ev.outcome == "dodged":
        d = float(np.hypot(ev.fly_at_impact[0] - ev.aim[0], ev.fly_at_impact[1] - ev.aim[1]))
        return f"{head} (fly {d:.1f} mm from the aim point when the plate landed)"
    return f"{head} (aim missed)"


@dataclass
class SwatterHandle:
    """What ``install_swatter`` returns."""

    swatter: Swatter
    vision: object = None  # LoomingVision (or None)
    brain_link: object = None
    level: int = 2
    say: Callable[[str], None] | None = None
    _listeners: list = field(default_factory=list)
    _own_vision: bool = False

    def swat(self, level: int | None = None, side: str = "rear", source: str = "api") -> str:
        return self.swatter.swat(level or self.level, side=side, source=source)

    def handle_key(self, key: str) -> str | None:
        """Suggested keys: 'v' swat from behind, 'V' random side. None if not ours."""
        if key not in ("v", "V"):
            return None
        side = "rear" if key == "v" else "random"
        r = self.swat(side=side, source="key")
        return (f"[swatter] {r}: L{self.level} {self.swatter.level_name(self.level)} "
                f"from {side}")

    def detach(self) -> None:
        for fn in self._listeners:
            if fn in self.swatter.listeners:
                self.swatter.listeners.remove(fn)
        if self.vision is not None and getattr(self, "_own_vision", False):
            self.vision.detach()
        self.swatter.detach()


def install_swatter(session_or_parts, cfg: SwatterConfig | dict | None = None, *,
                    swatter: Swatter | None = None, brain_link=None, vision: bool = True,
                    looming=None, looming_cfg=None, escape: bool = True,
                    short_hz: float = 60.0, flight: bool = False,
                    hit_ref_impulse_uNs: float = 100.0,
                    say: Callable[[str], None] | None = None) -> SwatterHandle:
    """Attach a flyswatter to a running app / simulation; returns a ``SwatterHandle``.

    The swatter's bodies must already be in the model: pass the extension when the
    session is built, then install::

        sw = Swatter(SwatterConfig())
        session = Session(app_cfg, world_extensions=[sw.extension], brain=link)
        h = install_swatter(session, swatter=sw)
        h.swat(level=1)            # or h.handle_key("v")

    ``session_or_parts``: an app ``Session`` (uses ``.sim``, ``.actions``, ``.brain``,
    ``.ground_height``, ``.metrics``), a ``Simulation``, or any object with ``sim``
    (+ optional ``actions`` / ``brain`` / ``ground_height`` / ``metrics``).

    * ``vision``: the paddle becomes a flat looming source (``swatter_source``) on
      ``looming`` (an existing ``LoomingVision``, e.g. the whip's) or on a new one
      sending ``loom`` events to the brain link.
    * ``escape`` (needs a brain link with ``--brain-actions`` triggers): GF >=
      ``short_hz`` -> short-mode jump. ``flight`` (default off) adds the escape-flight
      *emulation*: an external thorax force pushing the fly away from the paddle after
      take-off. It is not wing physics, so it stays off unless explicitly requested;
      dodges then come from the real jump alone.
    * Hits go to the brain as ``StimulusEvent("hit", side="top")`` (intensity =
      impulse / ``hit_ref_impulse_uNs``) and to ``say``.
    """
    from perpetualfly.brain.schema import StimulusEvent

    parts = session_or_parts
    sim = getattr(parts, "sim", parts)
    if isinstance(cfg, dict):
        cfg = SwatterConfig.from_dict(cfg)
    if mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, PREFIX + "hand") < 0:
        raise RuntimeError("the swatter is not in the model: build the Session / Simulation "
                           "with world_extensions=[swatter.extension] first")
    sw = swatter or Swatter(cfg)
    sw.attach(sim)
    gh = getattr(parts, "ground_height", None)
    if callable(gh):
        sw.ground_height_fn = gh
    actions = getattr(parts, "actions", None)
    if actions is not None:
        sw.jump_probe = lambda: actions.active_name == "jump"
    link = brain_link if brain_link is not None else getattr(parts, "brain", None)
    h = SwatterHandle(sw, brain_link=link, say=say or getattr(parts, "say", None))
    if vision:
        if looming is None:
            from perpetualfly.vision.looming import LoomingConfig, LoomingVision

            if isinstance(looming_cfg, dict):
                looming_cfg = LoomingConfig.from_dict(looming_cfg)
            time_fn = None
            if link is not None and hasattr(link, "_run_time"):
                time_fn = link._run_time
            elif hasattr(parts, "metrics"):
                time_fn = parts.metrics.run_time_at
            looming = LoomingVision(sim, looming_cfg, sink=link, time_fn=time_fn).attach()
            h._own_vision = True
        looming.add_source(swatter_source(sw))
        h.vision = looming
    trig = getattr(link, "triggers", None) if link is not None else None
    lcfg = getattr(link, "cfg", None)
    if (escape and trig is not None and h.say is not None and lcfg is not None
            and (getattr(lcfg, "window_s", 0.0) > 0.03 or not getattr(lcfg, "sync_wait_s", 0.0))):
        h.say("[swatter] note: brain window_s > 0.03 s or sync_wait_s = 0 -> the escape "
              "decision arrives late; use BrainLinkConfig(window_s=0.02, sync_wait_s=0.05) "
              "(docs/SWATTER.md)")
    if escape and trig is not None:
        trig.p.jump_short_hz = short_hz
        trig.p.jump_flight = flight
        trig.threat_fn = lambda: sw.plate_center() if sw.busy else None

    def on_swat(ev: SwatEvent) -> None:
        if h.say is not None:
            h.say(format_swat_event(ev))
        if ev.hit and link is not None and hasattr(link, "send"):
            rt = link._run_time(ev.sim_time) if hasattr(link, "_run_time") else ev.sim_time
            link.send(StimulusEvent(
                "hit", side="top", intensity=float(min(ev.impulse_uNs / hit_ref_impulse_uNs, 1.0)),
                duration_s=max(0.05, ev.duration_s), sim_time=float(rt),
                details={"body": ev.body, "impulse_uNs": ev.impulse_uNs, "level": ev.level,
                         "source": "swatter", "label": "SWAT HIT"}), source=ev.source)

    sw.listeners.append(on_swat)
    h._listeners.append(on_swat)
    return h
