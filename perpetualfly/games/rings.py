"""FLY THROUGH RINGS: the connectome brain pilots a *flying* fly through hoops.

Embodied design (docs/GAMES.md, game 3): the fly flies with the real flapping-wing
flight model (``perpetualfly.flight``: MuJoCo fluid forces on the beating wings,
dt 5e-5 s, the ``HoverController`` hover / forward-flight controller driven through
``FlightMode``'s inputs: forward speed, heading goal, altitude clearance). Rings
(visual hoops: mocap bodies built from capsules; no collision) stand across the
flight path at varying lateral offsets. The next ring is a target for the fly's two
compound eyes: its bearing on each eye becomes Poisson drive of that eye's **LC10a**
neurons (the pursuit channel of FOLLOW THE LEADER, same ``PursuitVision`` /
``pursuit_response`` interface). In the FlyWire connectome one side's LC10a drives
the ipsilateral DNa01/DNa02 steering pair; here the DNa01/02 left-right difference
sets the **heading rate** of the flight controller (turn toward the ring).

What the brain controls and what it does not (our interface design, honest list):

* heading: yes, DNa01/02 (turn_L - turn_R) -> heading rate, with the same gain per
  unit of DN "turn" as the walking games' measured turn rate (~240 deg/s);
* forward speed: no, fixed by the level (walk DNs BDN2/oDN1 and MDN are walking
  commands; we found no principled flight-speed mapping);
* altitude: no, held by the controller; the rings are at the flight altitude (no
  vertical offsets: LC10a carries no elevation signal we could use, and we have
  no documented DN for flight altitude in this model's descending groups).

This module has no rendering and no brain: ``RingCourse`` (hoops, course layout),
``FlightPilot`` (air start / take-off, heading + speed commands on ``FlightMode``),
``RingVision`` (eyes -> LC10a events) and ``RingsGame`` (rules, score, lives).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Callable

import mujoco as mj
import numpy as np

from perpetualfly.games.asteroids import GameEvent
from perpetualfly.games.chase import PursuitResponse, PursuitVision

PREFIX = "ring"


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class RingsDifficulty:
    name: str
    speed: float  # forward flight speed (mm/s)
    max_shift_mm: float  # lateral step between consecutive rings (max)
    radius_mm: float  # ring (hoop centre-line) radius


RINGS_DIFFICULTIES: dict[str, RingsDifficulty] = {
    "easy": RingsDifficulty("easy", 40.0, 4.0, 3.5),
    "normal": RingsDifficulty("normal", 50.0, 6.0, 3.0),
    "hard": RingsDifficulty("hard", 60.0, 8.0, 2.6),
}


@dataclass
class RingsConfig:
    altitude_mm: float = 6.0  # COM height above the ground (flight + ring centres)
    spacing_mm: float = 32.0  # along the course between rings
    first_dist_mm: float = 30.0  # first ring ahead of the fly
    min_shift_mm: float = 2.0  # lateral step between rings (min)
    max_lateral_mm: float = 10.0  # course centre line +- this
    n_rings: int = 3  # rings on screen (the next one is the target)
    tube_mm: float = 0.22  # hoop tube radius
    segments: int = 20  # capsules per hoop
    rgba_next: tuple = (1.0, 0.62, 0.08, 1.0)
    rgba_later: tuple = (0.55, 0.75, 0.95, 0.85)
    rgba_passed: tuple = (0.2, 0.9, 0.3, 1.0)
    rgba_missed: tuple = (0.9, 0.2, 0.2, 1.0)
    rgba_post: tuple = (0.35, 0.33, 0.30, 1.0)
    # outcome: through = thorax COM within radius - pass_margin of the centre when it
    # crosses the ring plane
    pass_margin_mm: float = 0.3
    rim_band_mm: float = 0.8  # "clipped the rim" label for |r - radius| < this
    ring_timeout_x: float = 2.5  # miss if not crossed within x * (distance / speed)
    behind_deg: float = 110.0  # miss when the ring is this far off the heading ...
    behind_hold_s: float = 0.3  # ... for this long
    # steering (our interface): heading rate = max_turn_dps * turn, where turn =
    # tanh(turn_L / r_ref) - tanh(turn_R / r_ref) (DNa01/02 group means, low-passed)
    max_turn_dps: float = 120.0
    r_ref_hz: float = 25.0
    turn_tau_s: float = 0.06
    # flight controller for the game: heading loop stiffened (yaw_wn 80 rad/s, as fast
    # as the attitude loop) so the heading goal is tracked within ~50 ms; velocity
    # control only (xy_zeta 3, no pull back to a position set point)
    yaw_wn: float = 80.0
    yaw_zeta: float = 1.0
    xy_zeta: float = 3.0
    # game
    lives: int = 3
    launch_hold_s: float = 0.04  # air start: held while the wingbeat fades in
    ready_s: float = 0.5  # after the air start / take-off: settle before the rings
    takeoff: bool = True  # new game: real jump take-off from standing (else air start)
    respawn_s: float = 0.4  # after a crash: fall this long, then air start
    ring_points: int = 100  # x min(streak, 5) x level
    rings_per_level: int = 5
    speed_per_level: float = 5.0
    shift_per_level: float = 0.5
    difficulty: str = "normal"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RingsConfig":
        d = dict(d)
        for k, v in d.items():
            if isinstance(v, list):
                d[k] = tuple(v)
        return cls(**d)


def rings_level_params(cfg: RingsConfig, level: int) -> dict:
    """Flight speed (mm/s), max lateral step (mm) and ring radius at ``level``."""
    diff = RINGS_DIFFICULTIES[cfg.difficulty]
    k = max(level - 1, 0)
    return {"speed": diff.speed + cfg.speed_per_level * k,
            "max_shift_mm": diff.max_shift_mm + cfg.shift_per_level * k,
            "radius_mm": diff.radius_mm}


def turn_command(rates: dict, r_ref_hz: float = 25.0) -> float:
    """DNa01/02 left-right difference in [-1, 1] (+ = steer left), as in
    ``brain.mapping.descending_to_drive``."""
    def act(k):
        return float(np.tanh(max(float(rates.get(k, 0.0)), 0.0) / r_ref_hz))

    return act("turn_L") - act("turn_R")


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# the rings
# ---------------------------------------------------------------------------


@dataclass
class Ring:
    center: np.ndarray  # (3,) world
    yaw: float  # the ring's normal (course direction), rad
    radius: float
    index: int = 0  # ring number in the game
    status: str = "ahead"  # ahead | passed | missed

    @property
    def normal(self) -> np.ndarray:
        return np.array([math.cos(self.yaw), math.sin(self.yaw), 0.0])


class _Target:
    """Duck type of ``LeaderFly`` for ``PursuitVision``: the next ring's centre."""

    def __init__(self, course: "RingCourse") -> None:
        self.course = course
        self.cfg = self

    @property
    def active(self) -> bool:
        return self.course.target_ring() is not None

    @property
    def visual_radius_mm(self) -> float:
        r = self.course.target_ring()
        return r.radius if r is not None else 1.0

    def position3(self) -> np.ndarray:
        return self.course.target_ring().center.copy()


class RingCourse:
    """A pool of ``n_rings`` hoop mocap bodies laid out along a course.

    ``extension(world)`` before ``add_fly``; ``attach(sim)``. ``start(pos, yaw)`` lays
    out a new course from the fly's position / heading; ``advance()`` drops the
    first ring (passed / missed) and appends one at the far end. Rings are visual
    only (contype 0). A hoop = ``segments`` capsules on a vertical circle, plus a
    thin post down to the ground so its height reads in the image.
    """

    def __init__(self, cfg: RingsConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or RingsConfig()
        self.rng = np.random.default_rng(seed)
        self.rings: list[Ring] = []
        self.done: list[Ring] = []  # recently passed / missed (still drawn)
        self.radius = RINGS_DIFFICULTIES[self.cfg.difficulty].radius_mm
        self.max_shift = RINGS_DIFFICULTIES[self.cfg.difficulty].max_shift_mm
        self.sim = None
        self.target = _Target(self)
        self._origin = np.zeros(3)
        self._u = np.array([1.0, 0.0, 0.0])
        self._v = np.array([0.0, 1.0, 0.0])
        self._along = 0.0
        self._lat = 0.0
        self._count = 0
        self._slots: list[Ring | None] = []

    # ------------------------------------------------------------------ build
    @property
    def n_slots(self) -> int:
        return self.cfg.n_rings + 1  # + the ring just passed

    def extension(self, world) -> None:
        c = self.cfg
        spec: mj.MjSpec = world.mjcf_root
        R = self.radius  # compiled into the model: fixed for the session
        n = c.segments
        for k in range(self.n_slots):
            body = spec.worldbody.add_body(name=f"{PREFIX}{k}/body", pos=(0.0, 0.0, -50.0),
                                           mocap=True)
            for j in range(n):
                a0, a1 = 2 * math.pi * j / n, 2 * math.pi * (j + 1) / n
                p0 = (0.0, R * math.cos(a0), R * math.sin(a0))
                p1 = (0.0, R * math.cos(a1), R * math.sin(a1))
                body.add_geom(name=f"{PREFIX}{k}/seg{j}", type=mj.mjtGeom.mjGEOM_CAPSULE,
                              size=(c.tube_mm, 0, 0), fromto=p0 + p1, rgba=c.rgba_later,
                              contype=0, conaffinity=0, group=1)
            body.add_geom(name=f"{PREFIX}{k}/post", type=mj.mjtGeom.mjGEOM_CAPSULE,
                          size=(0.5 * c.tube_mm, 0, 0),
                          fromto=(0.0, 0.0, -R - c.tube_mm, 0.0, 0.0, -c.altitude_mm),
                          rgba=c.rgba_post, contype=0, conaffinity=0, group=1)

    def attach(self, sim) -> "RingCourse":
        self.sim = sim
        m = sim.model
        self.mocap_ids = []
        self.geom_ids = []
        for k in range(self.n_slots):
            b = m.body(f"{PREFIX}{k}/body").id
            self.mocap_ids.append(int(m.body_mocapid[b]))
            self.geom_ids.append([m.geom(f"{PREFIX}{k}/seg{j}").id
                                  for j in range(self.cfg.segments)])
        self._slots = [None] * self.n_slots
        self.hide_all()
        return self

    # ------------------------------------------------------------------ layout
    def set_level(self, max_shift_mm: float) -> None:
        self.max_shift = float(max_shift_mm)

    def start(self, pos, yaw: float, first_offset_mm: float | None = None,
              n: int | None = None, seed: int | None = None) -> None:
        """New course from the fly at ``pos`` (3,) heading ``yaw``: the first ring
        ``first_dist_mm`` ahead at ``first_offset_mm`` lateral (+ = left; None =
        random step), then ``n`` (default n_rings) rings in all."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        c = self.cfg
        self._origin = np.array([float(pos[0]), float(pos[1]), c.altitude_mm])
        self._u = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        self._v = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        self._along = c.first_dist_mm - c.spacing_mm
        self._lat = 0.0
        self.rings = []
        self.done = []
        n = c.n_rings if n is None else n
        for k in range(n):
            self._append(first_offset_mm if k == 0 else None)
        self._write()

    def _append(self, lateral: float | None = None) -> Ring:
        c = self.cfg
        prev = self._origin + self._along * self._u + self._lat * self._v
        self._along += c.spacing_mm
        if lateral is None:
            step = float(self.rng.uniform(c.min_shift_mm, max(self.max_shift, c.min_shift_mm)))
            sign = 1.0 if self.rng.random() < 0.5 else -1.0
            if abs(self._lat + sign * step) > c.max_lateral_mm:
                sign = -sign
            lateral = self._lat + sign * step
        self._lat = float(lateral)
        center = self._origin + self._along * self._u + self._lat * self._v
        d = center - prev
        ring = Ring(center, math.atan2(d[1], d[0]), self.radius, self._count)
        self._count += 1
        self.rings.append(ring)
        return ring

    def target_ring(self) -> Ring | None:
        return self.rings[0] if self.rings else None

    def advance(self, status: str) -> Ring | None:
        """The target ring was ``status`` ("passed" / "missed"): keep it drawn (in its
        outcome colour) until the next one, append a new ring. Returns it."""
        if not self.rings:
            return None
        r = self.rings.pop(0)
        r.status = status
        self.done = [r]
        self._append()
        self._write()
        return r

    def hide_all(self) -> None:
        self.rings = []
        self.done = []
        if self.sim is not None:
            self._write()

    # ------------------------------------------------------------------ mujoco
    def _write(self) -> None:
        if self.sim is None:
            return
        d, m, c = self.sim.data, self.sim.model, self.cfg
        shown = self.done + self.rings
        for k in range(self.n_slots):
            mid = self.mocap_ids[k]
            if k >= len(shown):
                d.mocap_pos[mid] = (0.0, 0.0, -50.0)
                continue
            r = shown[k]
            d.mocap_pos[mid] = r.center
            d.mocap_quat[mid] = (math.cos(0.5 * r.yaw), 0.0, 0.0, math.sin(0.5 * r.yaw))
            if r.status == "passed":
                col = c.rgba_passed
            elif r.status == "missed":
                col = c.rgba_missed
            elif r is self.target_ring():
                col = c.rgba_next
            else:
                col = c.rgba_later
            m.geom_rgba[self.geom_ids[k]] = col

    # ------------------------------------------------------------------ geometry
    @staticmethod
    def plane_coords(ring: Ring, p: np.ndarray) -> tuple[float, float, float]:
        """(signed distance along the ring normal, lateral + = left, vertical) of p."""
        rel = np.asarray(p, float) - ring.center
        n = ring.normal
        lat = np.array([-n[1], n[0], 0.0])
        return float(rel @ n), float(rel @ lat), float(rel[2])


# ---------------------------------------------------------------------------
# the pilot: FlightMode inputs (speed, heading goal, altitude)
# ---------------------------------------------------------------------------


# (speed mm/s, HoverController.z_int after 3 s of level flight at 6 mm)
Z_TRIM = ((0.0, -0.247), (40.0, -0.461), (50.0, -0.505), (60.0, -0.546), (70.0, -0.583),
          (80.0, -0.617))


class FlightPilot:
    """Drives ``FlightMode`` for the game: air start / take-off, then forward flight
    at ``speed`` with the heading goal integrating the brain's heading rate.

    ``update(turn, dt)`` per physics chunk (turn in [-1, 1], + = left). The body
    heading used for the eyes and the game is the yaw of the level *stroke frame*
    (``true_yaw``): ``Simulation.heading()`` projects the thorax x axis, which is
    pitched up ~48 deg in the hover posture, so any bank angle leaks into it.
    """

    def __init__(self, sim, fm, cfg: RingsConfig) -> None:
        self.sim = sim
        self.fm = fm
        self.cfg = cfg
        self.speed = RINGS_DIFFICULTIES[cfg.difficulty].speed
        self.turn = 0.0  # low-passed command
        self.yaw_rate = 0.0  # rad/s commanded
        self.flying = False
        self._R_beta = None
        self._release_at: float | None = None

    def _tune(self) -> None:
        fm, c = self.fm, self.cfg
        ctrl = fm.ctrl
        g = replace(ctrl.g, yaw_wn=c.yaw_wn, yaw_zeta=c.yaw_zeta, xy_zeta=c.xy_zeta)
        ctrl.g = g
        fm._gains = g  # (landing keeps the same gains)
        fm._clearance = c.altitude_mm
        self._R_beta = ctrl.R_beta

    def launch(self, pos_xy, yaw: float) -> None:
        """Air start: the fly placed at the flight altitude in the hover posture,
        moving forward at ``speed``, wings fading in over 10 ms, the controller on."""
        sim, fm, c = self.sim, self.fm, self.cfg
        sim.flight_controller = None
        fm._stop_wings(now=True)
        fm._reset_state()
        # held in place (tether) while the wingbeat fades in, then released: a free
        # start would drop ~2 mm before the wings reach full stroke
        sim.place((float(pos_xy[0]), float(pos_xy[1]), c.altitude_mm),
                  yaw_deg=math.degrees(yaw))
        sim.tethered = True
        self._release_at = sim.time + c.launch_hold_s
        fm._escape_dir = None
        fm._yaw_goal = float(yaw)
        fm._start_wings()
        fm.ctrl.target.pos = np.r_[sim.com()[:2], c.altitude_mm]
        fm.ctrl.target.yaw = float(yaw)
        self._tune()
        # altitude-integral trim for level flight at this speed (measured with this
        # controller, 3 s of straight flight each; without it the integral needs ~1.5 s
        # to wind up and the fly flies ~0.8 mm high meanwhile)
        fm.ctrl.z_int = float(np.interp(self.speed, *zip(*Z_TRIM)))
        fm._speed = self.speed
        fm._set("forward")
        self.turn = 0.0
        self.yaw_rate = 0.0
        self.flying = True

    def takeoff(self) -> str:
        """Real take-off: FlightMode's jump -> wings -> hover (``poll_takeoff`` then
        switches to forward flight)."""
        self.flying = False
        self._release_at = None
        self.sim.tethered = False
        return self.fm.takeoff(source="game")

    def poll_takeoff(self) -> bool:
        """True once airborne after a take-off (then set up for forward flight)."""
        fm = self.fm
        if self.flying:
            return True
        if fm.state in ("hovering", "forward") and fm.ctrl is not None:
            self._tune()
            fm.ctrl.z_int = float(np.interp(self.speed, *zip(*Z_TRIM)))
            fm._speed = self.speed
            fm._set("forward")
            self.flying = True
        return self.flying

    def set_speed(self, speed: float) -> None:
        self.speed = float(speed)
        if self.flying and self.fm.state == "forward":
            self.fm._speed = self.speed

    def true_yaw(self) -> float:
        if self._R_beta is None:
            return self.sim.heading()
        R = self.sim.thorax_rotmat() @ self._R_beta
        return math.atan2(R[1, 0], R[0, 0])

    def update(self, turn: float, dt: float) -> None:
        c, fm = self.cfg, self.fm
        if self._release_at is not None and self.sim.time >= self._release_at:
            self._release_at = None
            self.sim.tethered = False
        a = 1.0 - math.exp(-dt / max(c.turn_tau_s, 1e-6)) if dt > 0 else 1.0
        self.turn += a * (float(np.clip(turn, -1.0, 1.0)) - self.turn)
        self.yaw_rate = math.radians(c.max_turn_dps) * self.turn
        if not self.flying or fm.state != "forward" or fm.ctrl is None:
            return
        fm._yaw_goal += self.yaw_rate * dt
        # velocity control only: the position set point follows the fly
        ctrl = fm.ctrl
        com = self.sim.com()
        ctrl.target.pos = np.r_[com[:2], ctrl.target.pos[2]]
        ctrl.xy_int[:] = 0.0

    @property
    def airborne(self) -> bool:
        return self.flying and self.fm.state in ("hovering", "forward")


# ---------------------------------------------------------------------------
# the eyes: next ring -> LC10a per eye
# ---------------------------------------------------------------------------


class RingVision(PursuitVision):
    """``PursuitVision`` on the next ring's centre, with the gaze frame following the
    stroke-frame yaw (``heading_fn``; level, low-passed like ``GameVision``)."""

    def __init__(self, sim, course: RingCourse, heading_fn: Callable[[], float], **kw) -> None:
        kw.setdefault("response", RING_RESPONSE)
        super().__init__(sim, course.target, **kw)
        self.heading_fn = heading_fn

    def gaze_yaw(self) -> float:
        t, h = self.sim.time, self.heading_fn()
        if self._yaw is None or self._yaw_t is None or t < self._yaw_t:
            self._yaw = h
        else:
            a = 1.0 - math.exp(-(t - self._yaw_t) / max(self.gaze_tau_s, 1e-6))
            self._yaw += a * ((h - self._yaw + math.pi) % (2 * math.pi) - math.pi)
        self._yaw_t = t
        return self._yaw


# The pursuit interface of FOLLOW THE LEADER, unchanged except that the size tuning
# does not silence a near ring (a hoop 3 mm in radius fills > 60 deg of the eye in
# the last ~5 mm; with big_drop 0.5 it keeps half its drive, as the leader does).
RING_RESPONSE = PursuitResponse()


# ---------------------------------------------------------------------------
# the game
# ---------------------------------------------------------------------------


class RingsGame:
    """Rules, score and lives of FLY THROUGH RINGS.

    ``after_physics(crashed)`` after every physics chunk. States: ``ready`` (take-off
    / settling, no rings), ``playing``, ``respawn`` (after a crash), ``gameover``;
    ``trial`` (experiment: outcomes measured, lives unchanged, no new rings).
    """

    name = "rings"

    def __init__(self, sim, course: RingCourse, pilot: FlightPilot,
                 cfg: RingsConfig | None = None, seed: int = 0,
                 on_respawn: Callable[[], None] | None = None) -> None:
        self.sim = sim
        self.course = course
        self.pilot = pilot
        self.cfg = cfg or course.cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.on_respawn = on_respawn
        self.events: list[GameEvent] = []
        self.listeners: list[Callable[[GameEvent], None]] = []
        self._t_offset = 0.0
        self.last_outcome: dict | None = None
        self._new_game()

    # ------------------------------------------------------------------ time
    def time(self, sim_time: float | None = None) -> float:
        return self._t_offset + (self.sim.time if sim_time is None else float(sim_time))

    def _emit(self, kind: str, text: str = "", **info) -> GameEvent:
        ev = GameEvent(self.time(), kind, text, None, info)
        self.events.append(ev)
        if len(self.events) > 2000:
            del self.events[:1000]
        for fn in self.listeners:
            fn(ev)
        return ev

    # ------------------------------------------------------------------ state
    def _new_game(self, takeoff: bool | None = None) -> None:
        c = self.cfg
        self.score = 0.0
        self.lives = c.lives
        self.level = 1
        self.passed = 0
        self.missed = 0
        self.crashes = 0
        self.streak = 0
        self.best_streak = 0
        self.t_start = self.time()
        self.t_end: float | None = None
        self._t_respawn: float | None = None
        self._behind_since: float | None = None
        self._prev_s: float | None = None
        self._t_ring = self.time()
        self._ring_budget = math.inf
        self.ring_rel = (math.inf, 0.0, 0.0)  # along, lateral, vertical of the fly
        self.bearing_deg = 0.0
        self.misses_lat: list[float] = []
        self._apply_level()
        self.course.hide_all()
        use_takeoff = c.takeoff if takeoff is None else takeoff
        self.state = "ready"
        if use_takeoff:
            msg = self.pilot.takeoff()
            if not msg.startswith("[flight] take-off"):
                self._air_start()
        else:
            self._air_start()
        self.t_ready_end = None

    def _air_start(self, pos=None, yaw: float | None = None) -> None:
        sim = self.sim
        p = sim.thorax_position() if pos is None else np.asarray(pos, float)
        y = self.pilot.true_yaw() if yaw is None else float(yaw)
        self.pilot.launch(p[:2], y)

    def _apply_level(self) -> None:
        lp = rings_level_params(self.cfg, self.level)
        self.pilot.set_speed(lp["speed"])
        self.course.set_level(lp["max_shift_mm"])

    def restart(self, difficulty: str | None = None, reset_fly: bool = True) -> None:
        if difficulty is not None:
            if difficulty not in RINGS_DIFFICULTIES:
                raise ValueError(f"difficulty must be one of {sorted(RINGS_DIFFICULTIES)}")
            # (speed and lateral steps change; the hoop radius is compiled into the
            # model and stays the session's)
            self.cfg.difficulty = difficulty
        if reset_fly:
            self._reset_fly()
        self._new_game()
        self._emit("restart", f"new game ({self.cfg.difficulty})")

    def _reset_fly(self) -> None:
        self._t_offset += self.sim.time
        self.sim.reset()
        self._t_offset -= self.sim.time
        if self.on_respawn is not None:
            self.on_respawn()

    @property
    def survival_s(self) -> float:
        end = self.t_end if self.t_end is not None else self.time()
        return max(0.0, end - self.t_start)

    @property
    def pass_rate(self) -> float:
        n = self.passed + self.missed
        return self.passed / n if n else 0.0

    # ------------------------------------------------------------------ rings
    def start_course(self, first_offset_mm: float | None = None, n: int | None = None,
                     seed: int | None = None) -> None:
        p = self.sim.com()
        self.course.start(p, self.pilot.true_yaw(), first_offset_mm, n=n, seed=seed)
        self._begin_ring()

    def _begin_ring(self) -> None:
        r = self.course.target_ring()
        self._prev_s = None
        self._behind_since = None
        self._t_ring = self.time()
        if r is None:
            self._ring_budget = math.inf
            return
        d = float(np.linalg.norm(r.center[:2] - self.sim.com()[:2]))
        self._ring_budget = self.cfg.ring_timeout_x * d / max(self.pilot.speed, 1.0) + 0.3

    def measure(self) -> None:
        r = self.course.target_ring()
        if r is None:
            self.ring_rel = (math.inf, 0.0, 0.0)
            self.bearing_deg = 0.0
            return
        p = self.sim.com()
        self.ring_rel = self.course.plane_coords(r, p)
        rel = r.center[:2] - p[:2]
        self.bearing_deg = math.degrees(_wrap(math.atan2(rel[1], rel[0]) - self.pilot.true_yaw()))

    # ------------------------------------------------------------------ rules
    def after_physics(self, crashed: bool = False) -> None:
        c = self.cfg
        now = self.time()
        self.measure()
        if self.state == "gameover":
            return
        if crashed and self.state in ("playing", "ready", "trial"):
            self._crash()
            return
        if self.state == "respawn":
            if now >= self._t_respawn:
                self._air_start(yaw=self._respawn_yaw)
                self.state = "ready"
                self.t_ready_end = None
                self._emit("respawn", "RELAUNCH")
            return
        if self.state == "ready":
            if not self.pilot.poll_takeoff():
                if self.pilot.fm.state == "walking" and not self.pilot.fm.busy \
                        and now - self.t_start > 1.5:
                    self._air_start()  # take-off failed / aborted: air start
                return
            if self.t_ready_end is None:
                self.t_ready_end = now + c.ready_s
            if now >= self.t_ready_end:
                self.state = "playing"
                self.start_course()
                self._emit("go", "FLY THROUGH THE RINGS!")
            return
        if self.state not in ("playing", "trial"):
            return
        r = self.course.target_ring()
        if r is None:
            return
        s, lat, vert = self.ring_rel
        if self._prev_s is not None and self._prev_s < 0.0 <= s:
            rad = math.hypot(lat, vert)
            self._outcome(r, rad < r.radius - c.pass_margin_mm, rad, lat, vert)
            return
        self._prev_s = s
        if abs(self.bearing_deg) > c.behind_deg and s < 0:
            if self._behind_since is None:
                self._behind_since = now
            elif now - self._behind_since >= c.behind_hold_s:
                self._outcome(r, False, math.hypot(lat, vert), lat, vert, why="turned away")
                return
        else:
            self._behind_since = None
        if now - self._t_ring > self._ring_budget:
            self._outcome(r, False, math.hypot(lat, vert), lat, vert, why="timeout")

    def _outcome(self, r: Ring, through: bool, rad: float, lat: float, vert: float,
                 why: str = "crossed") -> None:
        c = self.cfg
        rim = abs(rad - r.radius) < c.rim_band_mm
        self.last_outcome = {"ring": r.index, "through": through, "radial_mm": round(rad, 3),
                             "lateral_mm": round(lat, 3), "vertical_mm": round(vert, 3),
                             "why": why, "t": round(self.time(), 3)}
        if through:
            self.passed += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            pts = c.ring_points * min(self.streak, 5) * self.level
            if self.state == "playing":
                self.score += pts
                self._emit("pass", f"THROUGH! +{pts}" + (f"  x{min(self.streak, 5)}"
                                                         if self.streak > 1 else ""),
                           radial_mm=round(rad, 2), streak=self.streak)
                if self.passed % c.rings_per_level == 0:
                    self.level += 1
                    self._apply_level()
                    self._emit("level", f"LEVEL {self.level}", level=self.level)
            else:
                self._emit("pass", "THROUGH", radial_mm=round(rad, 2))
            self.course.advance("passed")
        else:
            self.missed += 1
            self.streak = 0
            self.misses_lat.append(lat)
            label = "CLIPPED THE RIM" if (rim and why == "crossed") else "MISSED"
            self.course.advance("missed")
            if self.state == "playing":
                self.lives -= 1
                self._emit("miss", label, radial_mm=round(rad, 2), why=why)
                if self.lives <= 0:
                    self._gameover()
                    return
                if why != "crossed":  # turned away / lost: new course ahead of the fly
                    self.start_course()
            else:
                self._emit("miss", label, radial_mm=round(rad, 2), why=why)
        if self.state == "trial":
            self.course.hide_all()
        self._begin_ring()

    def _crash(self) -> None:
        self.crashes += 1
        self.streak = 0
        self.course.hide_all()
        self.last_outcome = {"ring": None, "through": False, "why": "crash",
                             "t": round(self.time(), 3)}
        if self.state == "trial":
            self._emit("crash", "CRASH")
            self.state = "crashed"
            return
        if self.state == "playing":
            self.lives -= 1
        self._emit("crash", "CRASH!")
        if self.lives <= 0:
            self._gameover()
            return
        self.state = "respawn"
        self._respawn_yaw = self.pilot.true_yaw() if self.sim.tilt_deg() < 90 else \
            self.sim.heading()
        self._t_respawn = self.time() + self.cfg.respawn_s

    def _gameover(self) -> None:
        self.state = "gameover"
        self.t_end = self.time()
        self._emit("gameover", "GAME OVER", score=int(self.score), rings=self.passed,
                   survival_s=round(self.survival_s, 2))

    def summary(self) -> dict:
        return {"score": int(self.score), "level": self.level, "rings": self.passed,
                "missed": self.missed, "crashes": self.crashes, "lives": self.lives,
                "pass_rate": round(self.pass_rate, 3), "best_streak": self.best_streak,
                "survival_s": round(self.survival_s, 2), "difficulty": self.cfg.difficulty,
                "state": self.state}

    def score_entry(self) -> dict:
        return {"score": int(self.score), "survival_s": round(self.survival_s, 2),
                "level": self.level, "rings": self.passed, "best_streak": self.best_streak,
                "pass_rate": round(self.pass_rate, 3)}
