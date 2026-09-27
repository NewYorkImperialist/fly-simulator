"""CANYON RUN: the connectome brain flies a *flying* fly through a field of rock pillars.

Embodied design (docs/GAMES.md, game 5): the fly flies with the real flapping-wing
flight model (``fly_simulator.flight``, the same ``FlightSimulation`` / ``FlightMode``
/ ``FlightPilot`` as FLY THROUGH RINGS: air start, forward speed, heading goal,
altitude clearance, level stroke-frame heading). Sandstone pillars (hoodoos: visual
mocap bodies, no collision) stand in rows across the flight path. Every pillar is a
**looming source** for the two compound eyes (``fly_simulator.vision.looming`` via the
ASTEROID DODGE eyes, ``GameVision``: the pillar's true angular width per eye, the
rocks' ``asteroid_response`` LC4 / LPLC2 tuning, field-of-view gating, level gaze).
This is the *avoidance* channel: in the FlyWire connectome one eye's LC4 / LPLC2
drive the **contralateral** DNa01/DNa02 pair (turn away); here the DNa01/02
left-right difference sets the **heading rate** of the flight controller, exactly
as in FLY THROUGH RINGS (120 deg/s per unit turn).

What the brain controls and what it does not (our interface design, honest list):

* heading: yes, DNa01/02 (turn_L - turn_R) -> heading rate (``rings.turn_command``);
* forward speed: no, set by the game (it ramps up with the level);
* altitude: no, held by the controller (pillars are taller than the flight altitude,
  so only a sideways dodge helps);
* giant fibre (DNp01): not mapped (shown in the panel).

A hit is geometric: the thorax COM comes within ``pillar radius + reach_mm`` (2.5 mm:
the wings' reach) of the pillar's axis (the pillars have no collision geoms: a hard contact can blow
up the dt 5e-5 fluid model). A hit crashes the flight (``FlightMode`` crash: the
wings stop, the fly falls), costs a life and relaunches the fly after
``respawn_s``.

This module has no rendering and no brain: ``CanyonField`` (pillar pool, layout,
looming sources), ``CanyonVision`` (eyes -> LC4 / LPLC2 events) and ``CanyonGame``
(rules, score, lives).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable

import mujoco as mj
import numpy as np

from fly_simulator.games.asteroids import GameEvent, asteroid_response
from fly_simulator.games.vision import GameVision
from fly_simulator.vision.looming import LoomResponse, VisualSource

PREFIX = "canyon/"
PARK_Z = -60.0


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class CanyonDifficulty:
    name: str
    speed: float  # forward flight speed at level 1 (mm/s)
    max_offset_mm: float  # path pillar: lateral offset from the fly's path, |y| <= this
    spacing_mm: float  # between rows (along the flight)


CANYON_DIFFICULTIES: dict[str, CanyonDifficulty] = {
    "easy": CanyonDifficulty("easy", 32.0, 7.5, 28.0),
    "normal": CanyonDifficulty("normal", 40.0, 6.5, 24.0),
    "hard": CanyonDifficulty("hard", 50.0, 5.5, 21.0),
}


@dataclass
class CanyonConfig:
    altitude_mm: float = 6.0  # COM height above the ground
    # pillar pool: one mocap body per slot, radius per slot (cycled)
    n_pillars: int = 12
    radii_mm: tuple = (1.6, 2.0, 1.8, 2.3)
    height_mm: float = 10.0
    rgba_rock: tuple = (0.74, 0.44, 0.26, 1.0)
    rgba_band: tuple = (0.58, 0.31, 0.18, 1.0)
    rgba_cap: tuple = (0.45, 0.26, 0.17, 1.0)
    rgba_hit: tuple = (0.90, 0.18, 0.14, 1.0)
    rgba_near: tuple = (0.95, 0.80, 0.25, 1.0)
    # layout: rows appear spawn_dist_mm ahead along the fly's heading, one row every
    # spacing (difficulty) mm of flight. A row = one "path" pillar aimed at the fly's
    # projected straight path + U(-max_offset, max_offset) lateral, and flank pillars
    # (canyon sides) at +-U(flank_min, flank_max) with probability flank_p per side.
    spawn_dist_mm: float = 48.0
    first_dist_mm: float = 26.0
    flank_min_mm: float = 10.0
    flank_max_mm: float = 13.5
    flank_p: float = 0.65
    min_gap_mm: float = 1.5  # between pillar surfaces in one row
    # outcome geometry
    # hit: the pillar's surface within reach_mm of the thorax COM (horizontally): the
    # beating wings would strike it (the model's wing geoms reach <= 2.6 mm from the COM)
    reach_mm: float = 2.5
    near_miss_mm: float = 0.8  # passed with a surface clearance below this
    pass_margin_mm: float = 0.5  # passed: this far behind the fly
    park_behind_mm: float = 12.0
    obstacle_timeout_x: float = 3.0  # an obstacle not reached in 3x its flight time: passed
    # steering (our interface; the FLY THROUGH RINGS values): heading rate =
    # max_turn_dps * turn, turn = tanh(turn_L / r_ref) - tanh(turn_R / r_ref)
    max_turn_dps: float = 240.0
    r_ref_hz: float = 25.0
    turn_tau_s: float = 0.06
    # flight controller (as FLY THROUGH RINGS)
    yaw_wn: float = 80.0
    yaw_zeta: float = 1.0
    xy_zeta: float = 3.0
    launch_hold_s: float = 0.04
    # game
    lives: int = 3
    ready_s: float = 0.5
    respawn_s: float = 0.5
    points_per_mm: float = 1.0
    pass_points: int = 25  # x level
    near_points: int = 200  # x level
    rows_per_level: int = 6
    speed_per_level: float = 4.0
    max_speed: float = 70.0
    difficulty: str = "normal"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CanyonConfig":
        d = dict(d)
        for k, v in d.items():
            if isinstance(v, list):
                d[k] = tuple(v)
        return cls(**d)


def canyon_level_params(cfg: CanyonConfig, level: int) -> dict:
    """Flight speed (mm/s), path-pillar offset range and row spacing at ``level``."""
    diff = CANYON_DIFFICULTIES[cfg.difficulty]
    k = max(level - 1, 0)
    return {"speed": min(diff.speed + cfg.speed_per_level * k, cfg.max_speed),
            "max_offset_mm": diff.max_offset_mm,
            "spacing_mm": max(diff.spacing_mm - 0.5 * k, 14.0)}


def canyon_response() -> LoomResponse:
    """The ASTEROID DODGE LC4 / LPLC2 tuning, unchanged (``asteroid_response``)."""
    return asteroid_response()


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# pillars
# ---------------------------------------------------------------------------


@dataclass
class Pillar:
    slot: int
    radius: float
    active: bool = False
    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, PARK_Z]))
    u: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))  # flight dir at spawn
    kind: str = "path"  # path | flank
    offset_mm: float = 0.0  # lateral offset from the fly's projected path (+ = left)
    outcome: str | None = None  # None | passed | hit
    min_clear: float = math.inf  # min surface clearance of the COM (mm; < 0 = hit)
    near: bool = False
    row: int = 0
    pid: int = 0
    t_spawn: float = 0.0
    budget_s: float = math.inf


class CanyonField:
    """The pillar bodies (world extension), layout and looming sources.

    ``extension(world)`` before ``add_fly``; ``attach(sim)``. Pillars are visual only
    (contype 0): hits are judged geometrically by ``CanyonGame``.
    """

    def __init__(self, cfg: CanyonConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or CanyonConfig()
        c = self.cfg
        self.rng = np.random.default_rng(seed)
        self.pillars = [Pillar(i, float(c.radii_mm[i % len(c.radii_mm)]))
                        for i in range(c.n_pillars)]
        self.sim = None
        self.eye_z_fn: Callable[[], float] | None = None
        self._n_ids = 0
        self._rows = 0

    # ------------------------------------------------------------------ build
    def extension(self, world) -> None:
        c = self.cfg
        spec: mj.MjSpec = world.mjcf_root
        H = c.height_mm
        rng = np.random.default_rng(4321)
        for p in self.pillars:
            r = p.radius
            body = spec.worldbody.add_body(name=f"{PREFIX}p{p.slot}",
                                           pos=(0.0, 4.0 * p.slot, PARK_Z), mocap=True)
            kw = dict(contype=0, conaffinity=0, group=1)
            body.add_geom(name=f"{PREFIX}p{p.slot}_core", type=mj.mjtGeom.mjGEOM_CYLINDER,
                          size=(r, 0.5 * H, 0), pos=(0.0, 0.0, 0.5 * H), rgba=c.rgba_rock, **kw)
            # flared base, strata bands and a cap rock (a hoodoo), visual only
            body.add_geom(name=f"{PREFIX}p{p.slot}_base", type=mj.mjtGeom.mjGEOM_CYLINDER,
                          size=(1.35 * r, 0.5, 0), pos=(0.0, 0.0, 0.5), rgba=c.rgba_band, **kw)
            for k, z in enumerate((0.22 * H + rng.uniform(-0.4, 0.4),
                                   0.55 * H + rng.uniform(-0.6, 0.6),
                                   0.8 * H + rng.uniform(-0.4, 0.4))):
                body.add_geom(name=f"{PREFIX}p{p.slot}_band{k}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                              size=(1.05 * r, 0.22 + 0.1 * k, 0), pos=(0.0, 0.0, z),
                              rgba=c.rgba_band, **kw)
            body.add_geom(name=f"{PREFIX}p{p.slot}_cap", type=mj.mjtGeom.mjGEOM_ELLIPSOID,
                          size=(1.3 * r, 1.3 * r, 0.55 * r), pos=(0.0, 0.0, H), rgba=c.rgba_cap,
                          **kw)

    def attach(self, sim) -> "CanyonField":
        self.sim = sim
        m = sim.model
        self.mocap_ids = [int(m.body_mocapid[m.body(f"{PREFIX}p{p.slot}").id])
                          for p in self.pillars]
        self.core_ids = [m.geom(f"{PREFIX}p{p.slot}_core").id for p in self.pillars]
        sim.reset_hooks.append(self._on_reset)
        self.park_all()
        return self

    def detach(self) -> None:
        if self.sim is not None and self._on_reset in self.sim.reset_hooks:
            self.sim.reset_hooks.remove(self._on_reset)

    def _on_reset(self, sim) -> None:
        self.park_all()

    # ------------------------------------------------------------------ pool
    def active(self) -> list[Pillar]:
        return [p for p in self.pillars if p.active]

    def free(self) -> list[Pillar]:
        return [p for p in self.pillars if not p.active]

    def _write(self, p: Pillar) -> None:
        if self.sim is None:
            return
        self.sim.data.mocap_pos[self.mocap_ids[p.slot]] = p.pos

    def tint(self, p: Pillar, rgba) -> None:
        if self.sim is not None:
            self.sim.model.geom_rgba[self.core_ids[p.slot]] = rgba

    def park(self, p: Pillar) -> None:
        p.active = False
        p.outcome = None
        p.pos = np.array([0.0, 4.0 * p.slot, PARK_Z])
        self.tint(p, self.cfg.rgba_rock)
        self._write(p)

    def park_all(self) -> None:
        for p in self.pillars:
            self.park(p)

    def place(self, xy, u, *, kind: str = "path", offset_mm: float = 0.0,
              slot: int | None = None, radius_pick: str = "random", t: float = 0.0,
              row: int = 0) -> Pillar | None:
        """Activate a pillar at ``xy`` (world); ``u`` = the flight direction it faces."""
        free = self.free()
        if not free:
            return None
        if slot is not None and any(p.slot == slot for p in free):
            p = self.pillars[slot]
        else:
            p = free[int(self.rng.integers(len(free)))] if radius_pick == "random" else free[0]
        self._n_ids += 1
        p.active = True
        p.pos = np.array([float(xy[0]), float(xy[1]), 0.0])
        p.u = np.asarray(u, float)[:2] / max(float(np.linalg.norm(np.asarray(u, float)[:2])), 1e-9)
        p.kind = kind
        p.offset_mm = float(offset_mm)
        p.outcome = None
        p.min_clear = math.inf
        p.near = False
        p.row = row
        p.pid = self._n_ids
        p.t_spawn = t
        p.budget_s = math.inf
        self.tint(p, self.cfg.rgba_rock)
        self._write(p)
        return p

    def spawn_row(self, pos_xy, yaw: float, dist_mm: float, max_offset_mm: float,
                  t: float = 0.0, flanks: bool = True) -> list[Pillar]:
        """One row ``dist_mm`` ahead of ``pos_xy`` along ``yaw``: a path pillar at
        U(-max_offset, max_offset) and flank pillars (canyon sides)."""
        c, rng = self.cfg, self.rng
        self._rows += 1
        u = np.array([math.cos(yaw), math.sin(yaw)])
        v = np.array([-u[1], u[0]])  # left
        centre = np.asarray(pos_xy, float)[:2] + dist_mm * u
        out = []
        y = float(rng.uniform(-max_offset_mm, max_offset_mm))
        p = self.place(centre + y * v, u, kind="path", offset_mm=y, t=t, row=self._rows)
        if p is not None:
            out.append(p)
        if flanks:
            for side in (1.0, -1.0):
                if rng.random() > c.flank_p:
                    continue
                q = self.place(centre, u, kind="flank", t=t, row=self._rows)  # slot first
                if q is None:
                    break
                lo = c.flank_min_mm
                if p is not None and side * y > 0:  # keep a gap to the path pillar
                    lo = max(lo, abs(y) + p.radius + q.radius + c.min_gap_mm)
                yy = side * float(rng.uniform(lo, max(lo, c.flank_max_mm)))
                q.pos = np.array([*(centre + yy * v + rng.uniform(-3, 3) * u), 0.0])
                q.offset_mm = yy
                self._write(q)
                out.append(q)
        return out

    def spawn_obstacle(self, pos_xy, yaw: float, dist_mm: float, offset_mm: float,
                       slot: int | None = None, t: float = 0.0) -> Pillar | None:
        """One path pillar ``dist_mm`` ahead of ``pos_xy`` along ``yaw`` at lateral
        ``offset_mm`` (+ = left) from the straight path (the experiment)."""
        self._rows += 1
        u = np.array([math.cos(yaw), math.sin(yaw)])
        v = np.array([-u[1], u[0]])
        xy = np.asarray(pos_xy, float)[:2] + dist_mm * u + offset_mm * v
        return self.place(xy, u, kind="path", offset_mm=offset_mm, slot=slot, t=t,
                          row=self._rows)

    # ------------------------------------------------------------------ vision
    def visual_sources(self, response: LoomResponse | None = None,
                       period_s: float = 2e-3) -> list[VisualSource]:
        """One looming source per pillar: its axis point at the eyes' height with the
        pillar radius (``GameVision`` then uses theta = 2 asin(r / d), the pillar's
        true angular width; inactive pillars are invisible)."""
        resp = response or canyon_response()
        out = []
        for p in self.pillars:
            def shapes(p=p):
                if not p.active:
                    z = np.zeros((0, 3))
                    return z, z, np.zeros(0), np.zeros(0)
                zc = self.eye_z_fn() if self.eye_z_fn is not None else self.cfg.altitude_mm
                c = np.array([[p.pos[0], p.pos[1], zc]])
                return c, np.array([[0.0, 0.0, 1.0]]), np.zeros(1), np.array([p.radius])

            def period(p=p):
                return period_s if p.active else None

            out.append(VisualSource(f"pillar{p.slot}", shapes=shapes, period=period,
                                    response=resp))
        return out


# ---------------------------------------------------------------------------
# the eyes: every pillar -> LC4 / LPLC2 per eye
# ---------------------------------------------------------------------------


class CanyonVision(GameVision):
    """``GameVision`` (ASTEROID DODGE's eyes) with the gaze frame following the level
    stroke-frame yaw of the flying fly (``heading_fn``), low-passed like the
    walking games' gaze."""

    def __init__(self, sim, cfg, heading_fn: Callable[[], float], **kw) -> None:
        super().__init__(sim, cfg, **kw)
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

    def n_sent(self) -> int:
        return len(self.sent)


# ---------------------------------------------------------------------------
# the game
# ---------------------------------------------------------------------------


class CanyonGame:
    """Rules, score and lives of CANYON RUN.

    ``after_physics(crashed)`` after every physics chunk. States: ``ready`` (air
    start, settling), ``playing``, ``respawn`` (after a crash), ``gameover``;
    ``trial`` / ``crashed`` / ``settle`` are used by the experiment (outcomes
    measured, lives unchanged, no automatic rows).
    """

    name = "canyon"

    def __init__(self, sim, field_: CanyonField, pilot, cfg: CanyonConfig | None = None,
                 seed: int = 0, on_respawn: Callable[[], None] | None = None) -> None:
        self.sim = sim
        self.field = field_
        self.pilot = pilot
        self.cfg = cfg or field_.cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.field.rng = np.random.default_rng(seed + 101)
        self.on_respawn = on_respawn
        self.events: list[GameEvent] = []
        self.listeners: list[Callable[[GameEvent], None]] = []
        self._t_offset = 0.0
        self.last_hit: dict | None = None
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
    def _new_game(self) -> None:
        self.score = 0.0
        self.lives = self.cfg.lives
        self.level = 1
        self.distance_mm = 0.0
        self.passed = 0
        self.near_misses = 0
        self.crashes = 0
        self.hits = 0
        self.rows_cleared = 0
        self.t_start = self.time()
        self.t_end: float | None = None
        self._t_respawn: float | None = None
        self._dist_row = 0.0
        self._prev_xy: np.ndarray | None = None
        self._prev_t: float | None = None
        self.nearest: dict | None = None
        self.speed_mm_s = 0.0  # measured ground speed (low-passed, 0.1 s)
        self._apply_level()
        self.field.park_all()
        self.state = "ready"
        self._air_start()
        self.t_ready_end = self.time() + self.cfg.ready_s

    def _air_start(self, pos=None, yaw: float | None = None) -> None:
        p = self.sim.thorax_position() if pos is None else np.asarray(pos, float)
        y = self.pilot.true_yaw() if yaw is None else float(yaw)
        self.pilot.launch(p[:2], y)
        self._prev_xy = None

    def _apply_level(self) -> None:
        self.lp = canyon_level_params(self.cfg, self.level)
        self.pilot.set_speed(self.lp["speed"])

    def restart(self, difficulty: str | None = None, reset_fly: bool = True) -> None:
        if difficulty is not None:
            if difficulty not in CANYON_DIFFICULTIES:
                raise ValueError(f"difficulty must be one of {sorted(CANYON_DIFFICULTIES)}")
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

    # ------------------------------------------------------------------ layout
    def start_rows(self) -> None:
        """Rows from ``first_dist_mm`` out to ``spawn_dist_mm`` ahead of the fly."""
        c = self.cfg
        p = self.sim.com()[:2]
        yaw = self.pilot.true_yaw()
        sp = self.lp["spacing_mm"]
        d = c.first_dist_mm
        while True:
            self.field.spawn_row(p, yaw, d, self.lp["max_offset_mm"], t=self.time())
            if d + sp > c.spawn_dist_mm:
                break
            d += sp
        # the next row (spawn_dist_mm ahead) then lands one spacing behind the last one
        self._dist_row = max(0.0, c.spawn_dist_mm - d)

    # ------------------------------------------------------------------ rules
    def measure(self) -> None:
        """Clearances and outcomes of every active pillar; the nearest one ahead."""
        c = self.cfg
        p = self.sim.com()[:2]
        yaw = self.pilot.true_yaw()
        h = np.array([math.cos(yaw), math.sin(yaw)])
        near = None
        now = self.time()
        for q in self.field.active():
            rel = p - q.pos[:2]
            d = float(np.hypot(*rel))
            clear = d - q.radius - c.reach_mm
            q.min_clear = min(q.min_clear, clear)
            along = float(rel @ q.u)  # > 0: the fly is past it (spawn direction)
            ahead = float(-rel @ h)  # > 0: the pillar is ahead of the fly (now)
            if q.outcome is None:
                if clear < 0.0 and self.state in ("playing", "trial", "ready"):
                    self._hit(q)
                    return
                gone = (along > q.radius + c.pass_margin_mm
                        or ahead < -(q.radius + c.pass_margin_mm)
                        or now - q.t_spawn > q.budget_s)
                if gone:
                    self._passed(q)
            lat = float(-rel @ np.array([-h[1], h[0]]))
            if q.outcome is None and ahead > 0 and abs(lat) < ahead + q.radius + 1.0:
                if near is None or ahead < near["ahead_mm"]:
                    near = {"ahead_mm": ahead, "lateral_mm": lat, "clear_mm": clear,
                            "bearing_deg": math.degrees(math.atan2(lat, ahead)),
                            "radius": q.radius, "kind": q.kind}
            if q.active and (along > c.park_behind_mm or ahead < -c.park_behind_mm
                             or d > 3.0 * c.spawn_dist_mm):
                self.field.park(q)
        self.nearest = near

    def _passed(self, q: Pillar) -> None:
        c = self.cfg
        q.outcome = "passed"
        self.passed += 1
        near = 0.0 <= q.min_clear < c.near_miss_mm
        q.near = near
        if near:
            self.near_misses += 1
            self.field.tint(q, c.rgba_near)
        if self.state == "playing":
            pts = c.pass_points * self.level + (c.near_points * self.level if near else 0)
            self.score += pts
            if near:
                self._emit("near", f"NEAR MISS! +{c.near_points * self.level}",
                           clear_mm=round(q.min_clear, 2))
            if q.kind == "path":
                self.rows_cleared += 1
                if self.rows_cleared % c.rows_per_level == 0:
                    self.level += 1
                    self._apply_level()
                    self._emit("level", f"LEVEL {self.level}: {self.lp['speed']:.0f} mm/s",
                               level=self.level)
        else:
            self._emit("pass", "PASSED" + (" (near miss)" if near else ""),
                       clear_mm=round(q.min_clear, 2), offset_mm=round(q.offset_mm, 2))

    def _hit(self, q: Pillar) -> None:
        q.outcome = "hit"
        self.hits += 1
        self.field.tint(q, self.cfg.rgba_hit)
        self.last_hit = {"pillar": q.pid, "kind": q.kind, "offset_mm": round(q.offset_mm, 2),
                         "radius": q.radius, "t": round(self.time(), 3)}
        fm = self.pilot.fm
        if fm.state in ("hovering", "forward", "landing"):
            fm._to_walking("crash", reason="pillar")  # the wings stop: the fly falls
        self._crash(why="pillar")

    def _crash(self, why: str = "flight") -> None:
        self.crashes += 1
        self.pilot.flying = False
        if self.state in ("trial", "settle"):
            self._emit("crash", "CRASH", why=why)
            self.state = "crashed"
            return
        if self.state == "playing":
            self.lives -= 1
        self._emit("crash", "CRASH!" if why == "pillar" else "CRASH (flight)", why=why)
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
        self._emit("gameover", "GAME OVER", score=int(self.score),
                   distance_mm=round(self.distance_mm, 1), passed=self.passed)

    def after_physics(self, crashed: bool = False) -> None:
        c = self.cfg
        now = self.time()
        if self.state == "gameover":
            return
        if crashed and self.state in ("playing", "ready", "trial", "settle"):
            self._crash(why="flight")
            return
        if self.state == "respawn":
            if now >= self._t_respawn:
                self.field.park_all()
                self._air_start(yaw=self._respawn_yaw)
                self.state = "ready"
                self.t_ready_end = now + c.ready_s
                self._emit("respawn", "RELAUNCH")
            return
        # distance flown (airborne, playing / trial)
        p = self.sim.com()[:2].copy()
        if self._prev_xy is not None and self.state in ("playing", "trial") \
                and self.pilot.airborne:
            dd = float(np.hypot(*(p - self._prev_xy)))
            dt = now - self._prev_t if self._prev_t is not None else 0.0
            if dt > 0 and dd < 5.0:
                a = 1.0 - math.exp(-dt / 0.1)
                self.speed_mm_s += a * (dd / dt - self.speed_mm_s)
            if dd < 5.0:  # (not across a relaunch)
                self.distance_mm += dd
                self._dist_row += dd
                if self.state == "playing":
                    self.score += c.points_per_mm * dd
        self._prev_xy = p
        self._prev_t = now
        self.measure()
        if self.state == "ready":
            if now >= self.t_ready_end and self.pilot.airborne:
                self.state = "playing"
                self.start_rows()
                self._emit("go", "FLY THE CANYON!")
            return
        if self.state == "playing" and self._dist_row >= self.lp["spacing_mm"]:
            self._dist_row -= self.lp["spacing_mm"]
            self.field.spawn_row(self.sim.com()[:2], self.pilot.true_yaw(), c.spawn_dist_mm,
                                 self.lp["max_offset_mm"], t=now)

    def summary(self) -> dict:
        return {"score": int(self.score), "level": self.level,
                "distance_mm": round(self.distance_mm, 1), "passed": self.passed,
                "near_misses": self.near_misses, "crashes": self.crashes, "hits": self.hits,
                "lives": self.lives, "survival_s": round(self.survival_s, 2),
                "difficulty": self.cfg.difficulty, "state": self.state}

    def score_entry(self) -> dict:
        return {"score": int(self.score), "survival_s": round(self.survival_s, 2),
                "level": self.level, "distance_mm": round(self.distance_mm, 1),
                "passed": self.passed, "near_misses": self.near_misses}
