"""ASTEROID DODGE: fly-scale boulders roll at the real fly; the connectome brain steers.

Embodied design (docs/GAMES.md): the NeuroMechFly walks on flat ground in MuJoCo.
Rocks (a pool of mocap spheres with lumpy visual crust) spawn ahead of the fly and
roll toward where it will be. Each rock is a visual looming source for the fly's
two compound eyes (``perpetualfly.vision.looming``): its angular size and expansion
on each eye become LC4 / LPLC2 Poisson drive of that eye's side of the FlyWire
brain (``StimulusEvent("loom")``). The brain's descending neurons steer the body
(``perpetualfly.games.brain_io``). A hit is a physical contact between a rock and any
fly geom.

This module has no rendering and no brain: ``AsteroidField`` (bodies, kinematics,
contacts) and ``AsteroidGame`` (waves, score, lives, game over). Everything runs in
the physics thread (hooks) or right after a physics chunk (``after_physics``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable

import mujoco as mj
import numpy as np

from perpetualfly.terrain.chunks import FLY_BIT
from perpetualfly.vision.looming import LoomResponse, VisualSource

PREFIX = "asteroid/"
PARK_Z = -40.0  # mm: inactive rocks wait under the (opaque) ground


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class Difficulty:
    name: str
    speed_scale: float  # rock speed multiplier
    interval_scale: float  # spawn interval multiplier
    offset_mm: float  # max lateral offset of the aim point (smaller = more on target)


DIFFICULTIES: dict[str, Difficulty] = {
    "easy": Difficulty("easy", 0.8, 1.25, 4.0),
    "normal": Difficulty("normal", 1.0, 1.0, 3.5),
    "hard": Difficulty("hard", 1.3, 0.75, 3.0),
}


@dataclass
class AsteroidConfig:
    # rock pool: one mocap body per slot, radius per slot (mm), cycled
    n_rocks: int = 8
    radii_mm: tuple = (1.3, 1.6, 1.9, 1.5)
    n_lumps: int = 7  # visual crust bumps per rock
    rgba_rock: tuple = (0.47, 0.40, 0.34, 1.0)
    rgba_lump: tuple = (0.33, 0.28, 0.24, 1.0)
    rgba_hit: tuple = (0.85, 0.20, 0.15, 1.0)
    # spawning (relative to the fly's position and heading at spawn time)
    spawn_dist_mm: float = 26.0  # distance ahead
    spawn_bearing_deg: float = 12.0  # +- random bearing of the spawn point
    # rocks are aimed at the fly's predicted position (lead = fly velocity x time to
    # arrival) plus a random lateral offset in [-offset, +offset] mm (difficulty)
    lead: bool = True
    base_speed: float = 10.0  # mm/s, wave 1 (the fly walks ~14 mm/s toward it)
    speed_per_wave: float = 1.5
    max_speed: float = 22.0
    base_interval_s: float = 2.2  # between spawns, wave 1
    interval_per_wave: float = 0.15
    min_interval_s: float = 0.9
    rocks_wave1: int = 4
    rocks_per_wave: int = 2  # extra rocks each wave
    wave_break_s: float = 1.5
    lives: int = 3
    ready_s: float = 1.0  # walk-in before the first rock
    # scoring
    dodge_points: int = 100  # x wave number
    points_per_s: float = 10.0
    # outcome geometry
    pass_margin_mm: float = 2.5  # rock centre this far behind the fly (along its path): passed
    contact_hold_s: float = 0.03  # a hitting rock keeps colliding this long, then ghosts
    max_life_s: float = 8.0  # a rock is retired after this long
    contact_every_steps: int = 5
    # respawn a fly that lies on its back / side this long (tilt > flip_tilt_deg)
    flip_tilt_deg: float = 100.0
    flip_respawn_s: float = 1.2
    difficulty: str = "normal"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AsteroidConfig":
        d = dict(d)
        if isinstance(d.get("radii_mm"), list):
            d["radii_mm"] = tuple(d["radii_mm"])
        return cls(**d)


def asteroid_response() -> LoomResponse:
    """Game interface choice: LC4 / LPLC2 response to a rolling boulder.

    The whip's defaults fire above 12 000 deg/s (a millisecond lash) and the
    swatter's above 300 deg/s. A 1.5 mm rock closing at ~25 mm/s expands at only
    ~30 deg/s when it is 12 mm away (about 0.4 s before impact, the last moment a
    walking fly can still turn out of its path), so the thresholds are lowered:
    LC4 = 200 tanh(max(0, dtheta - 12) / 150) Hz for theta >= 7 deg;
    LPLC2 = 200 ramp(theta; 10, 45 deg) ramp(dtheta; 8, 60 deg/s). These are
    plausible looming-detector tunings (size-and-speed dependent, silent for small
    or receding objects), not fits to recordings (docs/GAMES.md).
    """
    return LoomResponse(max_hz=200.0, lc4_v0=12.0, lc4_vscale=150.0, lc4_theta0=7.0,
                        lplc2_theta0=10.0, lplc2_theta1=45.0, lplc2_gate_v0=8.0,
                        lplc2_gate_v1=60.0, min_hz=10.0)


# ---------------------------------------------------------------------------
# rocks
# ---------------------------------------------------------------------------


def _quat_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    s = math.sin(0.5 * angle)
    return np.array([math.cos(0.5 * angle), axis[0] * s, axis[1] * s, axis[2] * s])


@dataclass
class Rock:
    slot: int
    radius: float
    active: bool = False
    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, PARK_Z]))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    q0: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    roll: float = 0.0
    t_spawn: float = 0.0
    outcome: str | None = None  # None (live) | "hit" | "dodged" | "expired"
    t_hit: float | None = None
    ghost: bool = False
    min_dist: float = math.inf  # closest centre distance to the thorax (mm)
    offset_mm: float = 0.0  # lateral offset of its aim (signed, + = fly's left)
    side: str = "none"  # side of the fly it is aimed at ("left" / "right")
    wave: int = 0
    rid: int = 0  # running id


class AsteroidField:
    """The rock bodies (world extension), their kinematics and contact detection.

    ``extension(world)`` must run before ``add_fly`` (pass it in ``world_extensions``);
    ``attach(sim)`` installs one pre-step hook (moves the mocap rocks) and one
    post-step hook (rock-fly contacts). ``listeners`` get ``(kind, rock)`` with kind
    "hit" when a live rock first touches the fly.
    """

    def __init__(self, cfg: AsteroidConfig | None = None) -> None:
        self.cfg = cfg or AsteroidConfig()
        c = self.cfg
        self.rocks = [Rock(i, float(c.radii_mm[i % len(c.radii_mm)])) for i in range(c.n_rocks)]
        self.listeners: list[Callable[[str, Rock], None]] = []
        self.sim = None
        self._attached = False
        self._n_ids = 0

    # ------------------------------------------------------------------ build
    def extension(self, world) -> None:
        from flygym.compose import ContactParams

        c = self.cfg
        spec: mj.MjSpec = world.mjcf_root
        cp = ContactParams()
        rng = np.random.default_rng(1234)
        for rk in self.rocks:
            r = rk.radius
            body = spec.worldbody.add_body(name=f"{PREFIX}rock{rk.slot}",
                                           pos=(0.0, 3.0 * rk.slot, PARK_Z), mocap=True)
            body.add_geom(name=f"{PREFIX}rock{rk.slot}_geom", type=mj.mjtGeom.mjGEOM_SPHERE,
                          size=(r, 0, 0), contype=0, conaffinity=FLY_BIT, condim=3,
                          friction=(0.6, cp.torsional_friction, cp.rolling_friction),
                          solref=cp.get_solref_tuple(), solimp=cp.get_solimp_tuple(),
                          margin=cp.margin, priority=1, rgba=c.rgba_rock, group=1)
            # crust: bumps half sunk into the surface (visual only)
            for k in range(c.n_lumps):
                d = rng.normal(size=3)
                d /= np.linalg.norm(d)
                lr = r * rng.uniform(0.25, 0.45)
                body.add_geom(name=f"{PREFIX}rock{rk.slot}_lump{k}",
                              type=mj.mjtGeom.mjGEOM_SPHERE, size=(lr, 0, 0),
                              pos=tuple(d * (r - 0.55 * lr)), contype=0, conaffinity=0,
                              rgba=c.rgba_lump if k % 2 else c.rgba_rock, group=1)
            # a few craters (darker flattened discs)
            for k in range(3):
                d = rng.normal(size=3)
                d /= np.linalg.norm(d)
                zax = d
                q = np.zeros(4)
                mj.mju_quatZ2Vec(q, zax)
                body.add_geom(name=f"{PREFIX}rock{rk.slot}_crater{k}",
                              type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.32 * r, 0.02 * r, 0),
                              pos=tuple(d * 0.985 * r), quat=tuple(q), contype=0, conaffinity=0,
                              rgba=(0.22, 0.19, 0.17, 1.0), group=1)

    def attach(self, sim) -> "AsteroidField":
        if self._attached:
            return self
        m = sim.model
        self.sim = sim
        self.body_ids = [m.body(f"{PREFIX}rock{rk.slot}").id for rk in self.rocks]
        self.mocap_ids = [int(m.body_mocapid[b]) for b in self.body_ids]
        self.geom_ids = np.array([m.geom(f"{PREFIX}rock{rk.slot}_geom").id for rk in self.rocks])
        self._geom_slot = {int(g): i for i, g in enumerate(self.geom_ids)}
        self._conaff = m.geom_conaffinity[self.geom_ids].copy()
        self._rgba = m.geom_rgba[self.geom_ids].copy()
        root = m.body_rootid[sim.thorax_body_id]
        self.is_fly_geom = m.body_rootid[m.geom_bodyid] == root
        sim.pre_step_hooks.append(self._pre_step)
        sim.post_step_hooks.append(self._post_step)
        sim.reset_hooks.append(self._on_reset)
        self._attached = True
        self.park_all()
        return self

    def detach(self) -> None:
        if not self._attached:
            return
        s = self.sim
        s.pre_step_hooks.remove(self._pre_step)
        s.post_step_hooks.remove(self._post_step)
        s.reset_hooks.remove(self._on_reset)
        self._attached = False

    # ------------------------------------------------------------------ rocks
    def free_slots(self) -> list[int]:
        return [rk.slot for rk in self.rocks if not rk.active]

    def active(self) -> list[Rock]:
        return [rk for rk in self.rocks if rk.active]

    def live(self) -> list[Rock]:
        return [rk for rk in self.rocks if rk.active and rk.outcome is None]

    def _write(self, rk: Rock) -> None:
        d = self.sim.data
        i = self.mocap_ids[rk.slot]
        d.mocap_pos[i] = rk.pos
        q = np.empty(4)
        mj.mju_mulQuat(q, _quat_axis_angle(rk.axis, rk.roll), rk.q0)
        d.mocap_quat[i] = q

    def _set_ghost(self, rk: Rock, ghost: bool) -> None:
        m = self.sim.model
        g = int(self.geom_ids[rk.slot])
        rk.ghost = ghost
        m.geom_conaffinity[g] = 0 if ghost else self._conaff[rk.slot]

    def _tint(self, rk: Rock, rgba) -> None:
        self.sim.model.geom_rgba[int(self.geom_ids[rk.slot])] = rgba

    def park(self, rk: Rock) -> None:
        rk.active = False
        rk.vel = np.zeros(3)
        rk.pos = np.array([0.0, 3.0 * rk.slot, PARK_Z])
        self._set_ghost(rk, False)
        self._tint(rk, self._rgba[rk.slot])
        self._write(rk)

    def park_all(self) -> None:
        for rk in self.rocks:
            self.park(rk)

    def spawn(self, pos, vel, *, slot: int | None = None, rng=None, t: float | None = None,
              offset_mm: float = 0.0, wave: int = 0) -> Rock | None:
        """Activate a rock at ``pos`` (x, y; it rests on the ground) rolling with
        horizontal velocity ``vel`` (mm/s). Returns None if the pool is empty."""
        free = self.free_slots()
        if not free:
            return None
        if slot is None or slot not in free:
            slot = free[0] if rng is None else int(rng.choice(free))
        rk = self.rocks[slot]
        rng = rng or np.random.default_rng()
        rk.active = True
        rk.pos = np.array([float(pos[0]), float(pos[1]), rk.radius])
        rk.vel = np.array([float(vel[0]), float(vel[1]), 0.0])
        sp = float(np.hypot(*rk.vel[:2]))
        u = rk.vel / sp if sp > 1e-9 else np.array([1.0, 0.0, 0.0])
        rk.axis = np.cross([0.0, 0.0, 1.0], u)
        n = float(np.linalg.norm(rk.axis))
        rk.axis = rk.axis / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
        q0 = rng.normal(size=4)
        rk.q0 = q0 / np.linalg.norm(q0)
        rk.roll = 0.0
        rk.t_spawn = self.sim.time if t is None else t
        rk.outcome = None
        rk.t_hit = None
        rk.min_dist = math.inf
        rk.offset_mm = float(offset_mm)
        rk.side = "left" if offset_mm > 0 else "right"
        rk.wave = wave
        self._n_ids += 1
        rk.rid = self._n_ids
        self._set_ghost(rk, False)
        self._tint(rk, self._rgba[rk.slot])
        self._write(rk)
        return rk

    # ------------------------------------------------------------------ hooks
    def _pre_step(self, sim) -> None:
        dt = sim.timestep
        for rk in self.rocks:
            if not rk.active:
                continue
            rk.pos = rk.pos + rk.vel * dt
            rk.roll += float(np.hypot(rk.vel[0], rk.vel[1])) * dt / rk.radius
            self._write(rk)

    def _post_step(self, sim) -> None:
        if sim.step_count % self.cfg.contact_every_steps:
            return
        d = sim.data
        t = sim.time
        for rk in self.rocks:  # a hitting rock ghosts after contact_hold_s
            if rk.active and rk.outcome == "hit" and not rk.ghost and t - rk.t_hit > self.cfg.contact_hold_s:
                self._set_ghost(rk, True)
        n = d.ncon
        if n == 0:
            return
        g = d.contact.geom[:n]
        for a, b in g:
            s = self._geom_slot.get(int(a))
            other = b
            if s is None:
                s = self._geom_slot.get(int(b))
                other = a
            if s is None or not self.is_fly_geom[int(other)]:
                continue
            rk = self.rocks[s]
            if rk.active and rk.outcome is None:
                rk.outcome = "hit"
                rk.t_hit = t
                self._tint(rk, self.cfg.rgba_hit)
                for fn in self.listeners:
                    fn("hit", rk)

    def _on_reset(self, sim) -> None:
        self.park_all()

    # ------------------------------------------------------------------ vision
    def visual_sources(self, response: LoomResponse | None = None,
                       period_s: float = 2e-3) -> list[VisualSource]:
        """One looming source per rock (spheres; inactive rocks are invisible)."""
        resp = response or asteroid_response()
        out = []
        for rk in self.rocks:
            def shapes(rk=rk):
                if not rk.active:
                    z = np.zeros((0, 3))
                    return z, z, np.zeros(0), np.zeros(0)
                return (rk.pos[None, :].copy(), np.array([[0.0, 0.0, 1.0]]), np.zeros(1),
                        np.array([rk.radius]))

            def period(rk=rk):
                return period_s if rk.active else None

            out.append(VisualSource(f"rock{rk.slot}", shapes=shapes, period=period,
                                    response=resp))
        return out


# ---------------------------------------------------------------------------
# the game
# ---------------------------------------------------------------------------


@dataclass
class GameEvent:
    t: float  # game time (s)
    kind: str  # spawn | hit | dodge | wave | gameover | respawn | jump | restart
    text: str = ""
    rock: int | None = None
    info: dict = field(default_factory=dict)


def wave_params(cfg: AsteroidConfig, wave: int) -> dict:
    """Speed (mm/s), spawn interval (s) and rock count of ``wave`` (1-based)."""
    diff = DIFFICULTIES[cfg.difficulty]
    k = max(wave - 1, 0)
    speed = min(cfg.base_speed + cfg.speed_per_wave * k, cfg.max_speed) * diff.speed_scale
    interval = max(cfg.base_interval_s - cfg.interval_per_wave * k, cfg.min_interval_s)
    return {"speed": speed, "interval": interval * diff.interval_scale,
            "count": cfg.rocks_wave1 + cfg.rocks_per_wave * k, "offset": diff.offset_mm}


class AsteroidGame:
    """Waves, score, lives and outcomes on top of an ``AsteroidField``.

    Call ``after_physics()`` after every physics chunk. States: ``ready`` (walk-in),
    ``playing``, ``wave_break``, ``gameover``; ``paused`` is handled by the runner
    (it simply stops stepping). ``restart()`` resets the fly and the game.
    """

    def __init__(self, sim, field_: AsteroidField, cfg: AsteroidConfig | None = None,
                 seed: int = 0, on_respawn: Callable[[], None] | None = None) -> None:
        self.sim = sim
        self.field = field_
        self.cfg = cfg or field_.cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.on_respawn = on_respawn
        self.events: list[GameEvent] = []
        self.listeners: list[Callable[[GameEvent], None]] = []
        self._t_offset = 0.0  # game time = offset + sim.time (sim.reset zeroes sim.time)
        field_.listeners.append(self._on_field)
        self._new_game()

    # ------------------------------------------------------------------ time
    def time(self, sim_time: float | None = None) -> float:
        return self._t_offset + (self.sim.time if sim_time is None else float(sim_time))

    def _emit(self, kind: str, text: str = "", rock: Rock | None = None, **info) -> GameEvent:
        ev = GameEvent(self.time(), kind, text, None if rock is None else rock.rid, info)
        self.events.append(ev)
        for fn in self.listeners:
            fn(ev)
        return ev

    # ------------------------------------------------------------------ state
    def _new_game(self) -> None:
        self.state = "ready"
        self.score = 0.0
        self.lives = self.cfg.lives
        self.wave = 1
        self.dodges = 0
        self.hits = 0
        self.spawned = 0
        self.wave_spawned = 0
        self.t_state = self.time()
        self.t_start = self.time()
        self.t_next_spawn = self.time() + self.cfg.ready_s
        self.t_end: float | None = None
        self._flip_since: float | None = None
        self._last_t = self.time()

    def restart(self, difficulty: str | None = None, reset_fly: bool = True) -> None:
        if difficulty is not None:
            if difficulty not in DIFFICULTIES:
                raise ValueError(f"difficulty must be one of {sorted(DIFFICULTIES)}")
            self.cfg.difficulty = difficulty
        if reset_fly:
            self._reset_fly()
        else:
            self.field.park_all()
        self._new_game()
        self._emit("restart", f"new game ({self.cfg.difficulty})")

    def _reset_fly(self) -> None:
        self._t_offset += self.sim.time
        self.sim.reset()  # parks the rocks (reset hook) and zeroes sim.time
        self._t_offset -= self.sim.time
        if self.on_respawn is not None:
            self.on_respawn()

    @property
    def survival_s(self) -> float:
        end = self.t_end if self.t_end is not None else self.time()
        return max(0.0, end - self.t_start)

    # ------------------------------------------------------------------ spawning
    def spawn_rock(self, offset_mm: float | None = None, speed: float | None = None,
                   bearing_deg: float | None = None, slot: int | None = None) -> Rock | None:
        """One rock ahead of the fly, aimed at its predicted position + offset."""
        c, sim = self.cfg, self.sim
        wp = wave_params(c, self.wave)
        rng = self.rng
        speed = wp["speed"] if speed is None else speed
        if offset_mm is None:
            offset_mm = float(rng.uniform(-wp["offset"], wp["offset"]))
        if bearing_deg is None:
            bearing_deg = float(rng.uniform(-c.spawn_bearing_deg, c.spawn_bearing_deg))
        p = sim.thorax_position()[:2].copy()
        h = sim.heading()
        b = h + math.radians(bearing_deg)
        spawn = p + c.spawn_dist_mm * np.array([math.cos(b), math.sin(b)])
        target = p.copy()
        if c.lead:
            v = sim.thorax_linvel()[:2]
            fwd = np.array([math.cos(h), math.sin(h)])
            vf = max(float(v @ fwd), 0.0)  # only the forward walk is led
            t_arr = c.spawn_dist_mm / max(speed + vf, 1e-6)
            target = p + fwd * vf * t_arr
        d = target - spawn
        u = d / max(float(np.linalg.norm(d)), 1e-9)
        perp = np.array([-u[1], u[0]])  # left of the rock's direction of travel
        # + offset = the rock passes on the fly's left: the rock travels roughly
        # opposite to the fly's heading, so the fly's left is the rock's right (-perp)
        target = target - perp * offset_mm
        d = target - spawn
        u = d / max(float(np.linalg.norm(d)), 1e-9)
        rk = self.field.spawn(spawn, u * speed, slot=slot, rng=rng, t=sim.time,
                              offset_mm=offset_mm, wave=self.wave)
        if rk is not None:
            self.spawned += 1
            self.wave_spawned += 1
            self._emit("spawn", f"rock {rk.rid}", rk, offset_mm=round(offset_mm, 2),
                       speed=round(speed, 1), radius=rk.radius)
        return rk

    # ------------------------------------------------------------------ outcomes
    def _on_field(self, kind: str, rk: Rock) -> None:
        if kind != "hit":
            return
        if self.state in ("gameover",):
            return
        self.hits += 1
        self.lives -= 1
        self._emit("hit", "HIT!", rk, offset_mm=round(rk.offset_mm, 2))
        if self.lives <= 0:
            self.state = "gameover"
            self.t_end = self.time()
            self._emit("gameover", "GAME OVER", score=int(self.score),
                       survival_s=round(self.survival_s, 2))

    def _check_rocks(self) -> None:
        sim = self.sim
        p = sim.thorax_position()
        t = sim.time
        for rk in self.field.active():
            rel = p[:2] - rk.pos[:2]
            dist = float(np.hypot(*rel))
            rk.min_dist = min(rk.min_dist, dist)
            sp = float(np.hypot(*rk.vel[:2]))
            along = float(rel @ rk.vel[:2]) / max(sp, 1e-9)  # <0: rock is past the fly
            if rk.outcome is None and along < -(rk.radius + self.cfg.pass_margin_mm):
                rk.outcome = "dodged"
                if self.state != "gameover":
                    self.dodges += 1
                    pts = self.cfg.dodge_points * rk.wave
                    self.score += pts
                    self._emit("dodge", f"DODGE +{pts}", rk, min_dist=round(rk.min_dist, 2),
                               offset_mm=round(rk.offset_mm, 2))
            gone = along < -(rk.radius + 12.0) or t - rk.t_spawn > self.cfg.max_life_s
            if gone:
                if rk.outcome is None:
                    rk.outcome = "expired"
                self.field.park(rk)

    def _check_flip(self) -> None:
        tilt = self.sim.tilt_deg()
        now = self.time()
        if tilt > self.cfg.flip_tilt_deg:
            if self._flip_since is None:
                self._flip_since = now
            elif now - self._flip_since > self.cfg.flip_respawn_s:
                self._flip_since = None
                self._reset_fly()
                self._emit("respawn", "fly flipped: respawn")
        else:
            self._flip_since = None

    def after_physics(self) -> None:
        now = self.time()
        dt = max(0.0, now - self._last_t)
        self._last_t = now
        self._check_rocks()
        self._check_flip()
        if self.state == "gameover":
            return
        self.score += self.cfg.points_per_s * dt
        wp = wave_params(self.cfg, self.wave)
        if self.state == "ready":
            if now >= self.t_next_spawn:
                self.state = "playing"
                self._emit("wave", f"WAVE {self.wave}", wave=self.wave)
        if self.state == "wave_break" and now >= self.t_next_spawn:
            self.state = "playing"
            self._emit("wave", f"WAVE {self.wave}", wave=self.wave)
        if self.state == "playing":
            if self.wave_spawned < wp["count"]:
                if now >= self.t_next_spawn and self.field.free_slots():
                    self.spawn_rock()
                    self.t_next_spawn = now + wp["interval"] * float(self.rng.uniform(0.8, 1.2))
            elif not self.field.live():
                self.wave += 1
                self.wave_spawned = 0
                self.state = "wave_break"
                self.t_next_spawn = now + self.cfg.wave_break_s

    def summary(self) -> dict:
        return {"score": int(self.score), "wave": self.wave, "dodges": self.dodges,
                "hits": self.hits, "lives": self.lives, "spawned": self.spawned,
                "survival_s": round(self.survival_s, 2), "difficulty": self.cfg.difficulty,
                "state": self.state}
