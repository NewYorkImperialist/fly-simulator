"""FOLLOW THE LEADER (CHASE): a leader fly weaves ahead; the connectome brain pursues it.

Embodied design (docs/GAMES.md, game 2): the NeuroMechFly walks on flat ground in
MuJoCo. A second, dark "leader fly" (a mocap body built from ellipsoids and
capsules; purely visual, it has no collision) walks ahead of it along a randomly
weaving path. The leader is a small moving object for the follower's two compound
eyes: its position on each eye becomes Poisson drive of that eye's **LC10a**
neurons (the small-object / courtship-pursuit visual projection neurons;
``StimulusEvent("manual", side=eye, details={"set": "LC10a", "rate_hz": ...})``).
In the FlyWire connectome one side's LC10a drives the *ipsilateral* DNa01/DNa02
steering pair, i.e. a turn *toward* the object (docs/SENSORY_SCREEN.md), the
opposite of the looming channel used by ASTEROID DODGE.

Game: follow the leader. Points for every second within ``follow_dist_mm``; a
CATCH when the follower gets within ``catch_dist_mm`` (the leader then dashes off);
the leader is LOST (a life) when it gets farther than ``lose_dist_mm``.

This module has no rendering and no brain: ``LeaderFly`` (body, path, kinematics),
``pursuit_response`` (the LC10a interface), ``PursuitVision`` (eyes -> events) and
``ChaseGame`` (rules, score, lives).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable

import mujoco as mj
import numpy as np

from fly_simulator.brain.schema import StimulusEvent
from fly_simulator.games.asteroids import GameEvent
from fly_simulator.games.vision import GameVision
from fly_simulator.vision.looming import EYES, fov_mask

PREFIX = "leader/"
PURSUIT_SET = "LC10a"


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class ChaseDifficulty:
    name: str
    speed: float  # leader walking speed (mm/s); the follower walks ~13 mm/s
    weave_dps: float  # SD of the leader's turn rate (deg/s)


CHASE_DIFFICULTIES: dict[str, ChaseDifficulty] = {
    "easy": ChaseDifficulty("easy", 8.0, 45.0),
    "normal": ChaseDifficulty("normal", 9.5, 65.0),
    "hard": ChaseDifficulty("hard", 11.0, 90.0),
}


@dataclass
class ChaseConfig:
    # leader look (mm, fly-sized: the NeuroMechFly is ~2.5 mm long)
    rgba_body: tuple = (0.10, 0.08, 0.07, 1.0)
    rgba_band: tuple = (0.22, 0.17, 0.12, 1.0)
    rgba_eye: tuple = (0.50, 0.10, 0.07, 1.0)
    rgba_wing: tuple = (0.75, 0.80, 0.88, 0.40)
    body_z_mm: float = 1.1  # thorax height (the NeuroMechFly stands at ~1.1 mm)
    visual_radius_mm: float = 0.8  # equivalent sphere radius for the eyes
    # path: turn rate is an Ornstein-Uhlenbeck process (SD = difficulty weave_dps)
    weave_tau_s: float = 0.7
    max_turn_dps: float = 160.0
    path_update_s: float = 0.002
    speed_per_level: float = 0.7  # mm/s faster per level
    weave_per_level: float = 0.15  # x more weave per level
    catches_per_level: int = 3
    # the leader slows down (to wait_scale x speed) when the follower is far behind
    wait_dist_mm: float = 12.0
    wait_scale: float = 0.6
    # placement (relative to the follower's position and heading)
    start_dist_mm: float = 7.0
    start_bearing_deg: float = 25.0  # +- random
    # outcomes (thorax-to-thorax distances)
    catch_dist_mm: float = 3.0
    follow_dist_mm: float = 9.0
    lose_dist_mm: float = 20.0
    lose_hold_s: float = 0.4
    dash_s: float = 0.7  # after a catch the leader dashes off ...
    dash_scale: float = 2.2  # ... this much faster ...
    dash_turn_deg: float = 60.0  # ... after a random turn of up to this
    lives: int = 3
    ready_s: float = 1.0  # the leader walks straight (no weave, no scoring) first
    follow_points_per_s: float = 10.0
    catch_points: int = 100  # x level
    flip_tilt_deg: float = 100.0
    flip_respawn_s: float = 1.2
    difficulty: str = "normal"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChaseConfig":
        d = dict(d)
        for k, v in d.items():
            if isinstance(v, list):
                d[k] = tuple(v)
        return cls(**d)


def level_params(cfg: ChaseConfig, level: int) -> dict:
    """Leader speed (mm/s) and weave SD (deg/s) at ``level`` (1-based)."""
    diff = CHASE_DIFFICULTIES[cfg.difficulty]
    k = max(level - 1, 0)
    return {"speed": diff.speed + cfg.speed_per_level * k,
            "weave_dps": diff.weave_dps * (1.0 + cfg.weave_per_level * k)}


# ---------------------------------------------------------------------------
# the pursuit interface (ours)
# ---------------------------------------------------------------------------


@dataclass
class PursuitResponse:
    """LC10a Poisson rate of one eye for a small object (our interface design).

    rate = max_hz * size(theta) * ecc(azimuth) * fov, where

    * size = ramp(theta; theta0, theta1) * (1 - big_drop * ramp(theta; big0, big1)):
      LC10a are small-object detectors (silent for sub-degree specks, weaker for
      objects that fill a large part of the eye);
    * ecc = ramp(az; az0, az1), az = the object's azimuth in that eye's (gaze
      stabilised) frame, positive toward the eye's own side. The target error angle
      is the steering signal (chasing flies turn at a rate proportional to it;
      Land & Collett 1974). An object straight ahead drives neither eye, an object
      on the left drives only the left LC10a;
    * fov = the eye's field-of-view weight (``looming.fov_mask``): blind behind.

    Not a fit to recordings; the direction of the resulting turn is the
    connectome's (docs/GAMES.md).
    """

    max_hz: float = 150.0
    theta0: float = 1.0
    theta1: float = 4.0
    big0: float = 30.0
    big1: float = 60.0
    big_drop: float = 0.5
    az0: float = 0.0
    az1: float = 30.0
    min_hz: float = 5.0


def _ramp(x: float, a: float, b: float) -> float:
    if b <= a:
        return 1.0 if x >= b else 0.0
    return min(max((x - a) / (b - a), 0.0), 1.0)


def pursuit_response(theta_deg: float, az_deg: float, fov_w: float = 1.0,
                     p: PursuitResponse | None = None) -> float:
    p = p or PursuitResponse()
    size = _ramp(theta_deg, p.theta0, p.theta1) * (1.0 - p.big_drop * _ramp(theta_deg, p.big0, p.big1))
    r = p.max_hz * size * _ramp(az_deg, p.az0, p.az1) * max(0.0, min(1.0, fov_w))
    return r if r >= p.min_hz else 0.0


# ---------------------------------------------------------------------------
# the leader fly
# ---------------------------------------------------------------------------


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])


class LeaderFly:
    """A dark fly-shaped mocap body walking along a weaving path.

    ``extension(world)`` before ``add_fly``; ``attach(sim)`` installs a pre-step hook
    (path integration). Path: heading integrates an OU turn rate (SD ``weave_dps``,
    time constant ``weave_tau_s``) at ``speed`` mm/s; ``weave = False`` walks straight.
    ``dash()`` (after a catch): a random turn and ``dash_scale`` x speed for
    ``dash_s``. ``waiting`` (set by the game): walk at ``wait_scale`` x speed.
    """

    def __init__(self, cfg: ChaseConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or ChaseConfig()
        self.rng = np.random.default_rng(seed)
        self.pos = np.array([5.0, 0.0])
        self.yaw = 0.0
        self.omega = 0.0  # rad/s
        self.speed = CHASE_DIFFICULTIES[self.cfg.difficulty].speed
        self.weave_dps = CHASE_DIFFICULTIES[self.cfg.difficulty].weave_dps
        self.weave = True
        self.active = True
        self.waiting = False
        self.dash_until = -1e9
        self._t_path = None
        self.sim = None
        self._attached = False
        self.distance_walked = 0.0

    # ------------------------------------------------------------------ build
    def extension(self, world) -> None:
        c = self.cfg
        spec: mj.MjSpec = world.mjcf_root
        body = spec.worldbody.add_body(name=f"{PREFIX}body", pos=(5.0, 0.0, c.body_z_mm),
                                       mocap=True)
        E = mj.mjtGeom.mjGEOM_ELLIPSOID

        def geom(name, type_, size, pos, rgba, quat=(1.0, 0.0, 0.0, 0.0), fromto=None):
            kw = dict(name=f"{PREFIX}{name}", type=type_, size=size, rgba=rgba, contype=0,
                      conaffinity=0, group=1)
            if fromto is not None:
                kw["fromto"] = fromto
            else:
                kw["pos"] = pos
                kw["quat"] = quat
            body.add_geom(**kw)

        geom("thorax", E, (0.45, 0.36, 0.36), (0.05, 0.0, 0.0), c.rgba_body)
        geom("abdomen", E, (0.72, 0.38, 0.33), (-0.88, 0.0, -0.06), c.rgba_body)
        for k, x in enumerate((-0.62, -0.95, -1.25)):  # tergite bands
            geom(f"band{k}", E, (0.08, 0.36 - 0.05 * k, 0.32 - 0.05 * k), (x, 0.0, -0.03),
                 c.rgba_band)
        geom("head", E, (0.22, 0.34, 0.28), (0.62, 0.0, 0.06), c.rgba_body)
        for s in (1.0, -1.0):
            geom(f"eye{'L' if s > 0 else 'R'}", mj.mjtGeom.mjGEOM_SPHERE, (0.19, 0, 0),
                 (0.66, 0.22 * s, 0.10), c.rgba_eye)
            q = np.zeros(4)
            mj.mju_axisAngle2Quat(q, np.array([0.0, 0.0, 1.0]), math.radians(-14.0 * s))
            geom(f"wing{'L' if s > 0 else 'R'}", E, (0.95, 0.30, 0.02),
                 (-0.72, 0.30 * s, 0.34), c.rgba_wing, quat=tuple(q))
            for k, (x0, dx) in enumerate(((0.32, 0.55), (0.08, 0.05), (-0.16, -0.55))):
                knee = (x0 + 0.5 * dx, 0.75 * s, 0.05)
                foot = (x0 + dx, 1.05 * s, -c.body_z_mm + 0.04)
                geom(f"leg{k}{'L' if s > 0 else 'R'}a", mj.mjtGeom.mjGEOM_CAPSULE,
                     (0.045, 0, 0), None, c.rgba_body, fromto=(x0, 0.22 * s, -0.18) + knee)
                geom(f"leg{k}{'L' if s > 0 else 'R'}b", mj.mjtGeom.mjGEOM_CAPSULE,
                     (0.035, 0, 0), None, c.rgba_body, fromto=knee + foot)

    def attach(self, sim) -> "LeaderFly":
        if self._attached:
            return self
        self.sim = sim
        b = sim.model.body(f"{PREFIX}body").id
        self.mocap_id = int(sim.model.body_mocapid[b])
        sim.pre_step_hooks.append(self._pre_step)
        self._attached = True
        self._write()
        return self

    def detach(self) -> None:
        if self._attached:
            self.sim.pre_step_hooks.remove(self._pre_step)
            self._attached = False

    # ------------------------------------------------------------------ path
    def place(self, pos, yaw: float, seed: int | None = None) -> None:
        self.pos = np.array([float(pos[0]), float(pos[1])])
        self.yaw = float(yaw)
        self.active = True
        self.omega = 0.0
        self.dash_until = -1e9
        self._t_path = None
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.sim is not None:
            self._write()

    def hide(self) -> None:
        """Park the leader under the ground; the eyes see nothing (experiment walk-in)."""
        self.active = False
        if self.sim is not None:
            d = self.sim.data
            d.mocap_pos[self.mocap_id] = (self.pos[0], self.pos[1], -40.0)

    def set_level(self, speed: float, weave_dps: float) -> None:
        self.speed = float(speed)
        self.weave_dps = float(weave_dps)

    def dash(self, t: float) -> None:
        c = self.cfg
        self.yaw += math.radians(float(self.rng.uniform(-c.dash_turn_deg, c.dash_turn_deg)))
        self.dash_until = t + c.dash_s

    def dashing(self, t: float) -> bool:
        return t < self.dash_until

    def current_speed(self, t: float) -> float:
        c = self.cfg
        if self.dashing(t):
            return self.speed * c.dash_scale
        return self.speed * (c.wait_scale if self.waiting else 1.0)

    def _pre_step(self, sim) -> None:
        if not self.active:
            return
        c = self.cfg
        t = sim.time
        dt = sim.timestep
        if self._t_path is None or t < self._t_path:
            self._t_path = t
        if t - self._t_path >= c.path_update_s:
            h = t - self._t_path
            self._t_path = t
            if self.weave:
                sd = math.radians(self.weave_dps)
                a = math.exp(-h / c.weave_tau_s)
                self.omega = a * self.omega + sd * math.sqrt(1 - a * a) * float(self.rng.normal())
                lim = math.radians(c.max_turn_dps)
                self.omega = min(max(self.omega, -lim), lim)
            else:
                self.omega = 0.0
        self.yaw += self.omega * dt
        v = self.current_speed(t)
        self.pos = self.pos + v * dt * np.array([math.cos(self.yaw), math.sin(self.yaw)])
        self.distance_walked += v * dt
        self._write()

    def _write(self) -> None:
        d = self.sim.data
        d.mocap_pos[self.mocap_id] = (self.pos[0], self.pos[1], self.cfg.body_z_mm)
        d.mocap_quat[self.mocap_id] = _yaw_quat(self.yaw)

    def position3(self) -> np.ndarray:
        return np.array([self.pos[0], self.pos[1], self.cfg.body_z_mm])


# ---------------------------------------------------------------------------
# the follower's eyes: leader -> LC10a events per eye
# ---------------------------------------------------------------------------


@dataclass
class EyePursuit:
    theta: float = 0.0  # angular size (deg)
    az: float = 0.0  # azimuth in the eye's frame, + toward the eye's side (deg)
    fov: float = 0.0
    dist: float = math.inf
    rate_hz: float = 0.0


@dataclass
class _Emit:
    rate: float = 0.0
    t_end: float = -1e9


class PursuitVision(GameVision):
    """Post-step hook: the leader on each (gaze-stabilised) eye -> LC10a events.

    Events: ``StimulusEvent("manual", side=eye, duration_s=persist_s,
    details={"set": "LC10a", "rate_hz": r, ...})``, re-sent every ``refresh_s``
    while the rate is above ``response.min_hz`` and at once when it rises by more
    than 15 % + 5 Hz. The brain keeps the max rate per neuron over overlapping
    events, so a falling rate takes effect within ``persist_s``.
    """

    def __init__(self, sim, leader: LeaderFly, *, sink=None, time_fn=None,
                 response: PursuitResponse | None = None, update_every_s: float = 0.005,
                 persist_s: float = 0.04, refresh_s: float = 0.02, looming=None, **kw) -> None:
        from fly_simulator.vision.looming import LoomingConfig

        super().__init__(sim, looming or LoomingConfig(), sink=sink, time_fn=time_fn, **kw)
        self.leader = leader
        self.response = response or PursuitResponse()
        self.update_every_s = float(update_every_s)
        self.persist_s = float(persist_s)
        self.refresh_s = float(refresh_s)
        self.eye_state = [EyePursuit(), EyePursuit()]
        self._pemit = [_Emit(), _Emit()]

    def _on_reset(self, sim) -> None:
        super()._on_reset(sim)
        self._pemit = [_Emit(), _Emit()]
        self.eye_state = [EyePursuit(), EyePursuit()]

    def __call__(self, sim) -> None:
        if not self.cfg.enabled:
            return
        t = sim.time
        if self._last_t is not None and 0 <= t - self._last_t < self.update_every_s - 1e-9:
            return
        self.update()

    def update(self):
        t = self.sim.time
        self._last_t = t
        E, R = self.eyes()
        lead = self.leader
        for i in range(2):
            if not lead.active:
                self.eye_state[i] = EyePursuit()
                continue
            D = lead.position3() - E[i]
            rho = max(float(np.linalg.norm(D)), 1e-9)
            u = D / rho
            h = R[i].T @ u  # head-frame direction
            w = float(fov_mask(h[None, None, :], self.side_sign[i:i + 1], self.cfg)[0, 0])
            az = math.degrees(math.atan2(h[1] * self.side_sign[i], h[0]))
            r = self.leader.cfg.visual_radius_mm
            th = math.degrees(2.0 * math.asin(min(r / rho, 1.0)))
            rate = pursuit_response(th, az, w, self.response)
            self.eye_state[i] = EyePursuit(th, az, w, rho - r, rate)
            self._maybe_send_pursuit(i, t, self.eye_state[i])
        self.n_updates += 1
        return self.eye_state

    def _maybe_send_pursuit(self, i: int, t: float, e: EyePursuit) -> None:
        if e.rate_hz < self.response.min_hz:
            return
        em = self._pemit[i]
        active = t < em.t_end
        rise = e.rate_hz > em.rate * 1.15 + 5.0
        if active and not rise and t < em.t_end - (self.persist_s - self.refresh_s):
            return
        side = EYES[i]
        ev = StimulusEvent(
            "manual", side=side, intensity=e.rate_hz / self.response.max_hz,
            duration_s=self.persist_s, sim_time=float(self.time_fn(t)),
            details={"set": PURSUIT_SET, "rate_hz": round(e.rate_hz, 2),
                     "lc10a_hz": round(e.rate_hz, 2), "azimuth_deg": round(e.az, 1),
                     "theta_deg": round(e.theta, 2),
                     "dist_mm": round(e.dist, 3) if math.isfinite(e.dist) else None,
                     "label": f"PURSUIT {side[0].upper()} LC10a"})
        em.rate = max(em.rate, e.rate_hz) if active else e.rate_hz
        em.t_end = t + self.persist_s
        self.sent.append(ev)
        if len(self.sent) > 5000:
            del self.sent[:2500]
            self.n_trimmed = getattr(self, "n_trimmed", 0) + 2500
        sink = self.sink
        if sink is None:
            return
        if hasattr(sink, "send"):
            sink.send(ev, source="vision")
        else:
            sink(ev)

    def n_sent(self) -> int:
        return len(self.sent) + getattr(self, "n_trimmed", 0)


# ---------------------------------------------------------------------------
# the game
# ---------------------------------------------------------------------------


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class ChaseGame:
    """Rules, score and lives of FOLLOW THE LEADER on top of a ``LeaderFly``.

    Call ``after_physics()`` after every physics chunk. States: ``ready`` (the
    leader walks straight, no scoring), ``playing``, ``gameover`` (``trial`` is used
    by the experiment: outcomes are measured but lives and levels do not change).
    """

    name = "chase"

    def __init__(self, sim, leader: LeaderFly, cfg: ChaseConfig | None = None, seed: int = 0,
                 on_respawn: Callable[[], None] | None = None) -> None:
        self.sim = sim
        self.leader = leader
        self.cfg = cfg or leader.cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.on_respawn = on_respawn
        self.events: list[GameEvent] = []
        self.listeners: list[Callable[[GameEvent], None]] = []
        self._t_offset = 0.0
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
        self.state = "ready"
        self.score = 0.0
        self.lives = self.cfg.lives
        self.level = 1
        self.catches = 0
        self.losses = 0
        self.follow_s = 0.0
        self.play_s = 0.0
        self.t_start = self.time()
        self.t_ready_end = self.time() + self.cfg.ready_s
        self.t_end: float | None = None
        self._far_since: float | None = None
        self._flip_since: float | None = None
        self._last_t = self.time()
        self.dist = math.inf
        self.error_deg = 0.0
        self.following = False
        self._apply_level()
        self.place_leader(bearing_deg=0.0)
        self.leader.weave = False

    def _apply_level(self) -> None:
        lp = level_params(self.cfg, self.level)
        self.leader.set_level(lp["speed"], lp["weave_dps"])

    def restart(self, difficulty: str | None = None, reset_fly: bool = True) -> None:
        if difficulty is not None:
            if difficulty not in CHASE_DIFFICULTIES:
                raise ValueError(f"difficulty must be one of {sorted(CHASE_DIFFICULTIES)}")
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
    def follow_frac(self) -> float:
        return self.follow_s / self.play_s if self.play_s > 0 else 0.0

    def place_leader(self, dist_mm: float | None = None, bearing_deg: float | None = None,
                     yaw_offset_deg: float = 0.0, seed: int | None = None) -> None:
        """Put the leader ``dist_mm`` ahead of the follower at ``bearing_deg`` (+ = the
        follower's left), facing the follower's heading + ``yaw_offset_deg``."""
        c = self.cfg
        dist = c.start_dist_mm if dist_mm is None else dist_mm
        if bearing_deg is None:
            bearing_deg = float(self.rng.uniform(-c.start_bearing_deg, c.start_bearing_deg))
        p = self.sim.thorax_position()[:2]
        h = self.sim.heading()
        b = h + math.radians(bearing_deg)
        self.leader.place(p + dist * np.array([math.cos(b), math.sin(b)]),
                          h + math.radians(yaw_offset_deg), seed=seed)
        self._far_since = None

    # ------------------------------------------------------------------ geometry
    def measure(self) -> tuple[float, float]:
        """(distance mm, target error deg: + = leader on the follower's left)."""
        p = self.sim.thorax_position()[:2]
        rel = self.leader.pos - p
        d = float(np.hypot(*rel))
        err = math.degrees(_wrap(math.atan2(rel[1], rel[0]) - self.sim.heading()))
        return d, err

    # ------------------------------------------------------------------ rules
    def _check_flip(self) -> None:
        tilt = self.sim.tilt_deg()
        now = self.time()
        if tilt > self.cfg.flip_tilt_deg:
            if self._flip_since is None:
                self._flip_since = now
            elif now - self._flip_since > self.cfg.flip_respawn_s:
                self._flip_since = None
                self._reset_fly()
                self.place_leader()
                self._emit("respawn", "fly flipped: respawn")
        else:
            self._flip_since = None

    def after_physics(self) -> None:
        c = self.cfg
        now = self.time()
        dt = max(0.0, now - self._last_t)
        self._last_t = now
        self.dist, self.error_deg = self.measure()
        d = self.dist
        self.leader.waiting = d > c.wait_dist_mm
        self._check_flip()
        if self.state == "gameover":
            self.following = False
            return
        if self.state == "ready":
            if now >= self.t_ready_end:
                self.state = "playing"
                self.leader.weave = True
                self._emit("go", "FOLLOW HIM!")
            return
        self.following = d < c.follow_dist_mm
        self.play_s += dt
        if self.following:
            self.follow_s += dt
            if self.state == "playing":
                self.score += c.follow_points_per_s * dt
        t_sim = self.sim.time
        if d < c.catch_dist_mm and not self.leader.dashing(t_sim):
            self.catches += 1
            self.leader.dash(t_sim)
            if self.state == "playing":
                pts = c.catch_points * self.level
                self.score += pts
                self._emit("catch", f"CATCH +{pts}", dist_mm=round(d, 2))
                if self.catches % c.catches_per_level == 0:
                    self.level += 1
                    self._apply_level()
                    self._emit("level", f"LEVEL {self.level}", level=self.level)
            else:
                self._emit("catch", "CATCH", dist_mm=round(d, 2))
        if d > c.lose_dist_mm:
            if self._far_since is None:
                self._far_since = now
            elif now - self._far_since >= c.lose_hold_s:
                self._far_since = None
                self.losses += 1
                if self.state == "playing":
                    self.lives -= 1
                    self._emit("lost", "LOST HIM!", dist_mm=round(d, 1))
                    if self.lives <= 0:
                        self.state = "gameover"
                        self.t_end = now
                        self.following = False
                        self._emit("gameover", "GAME OVER", score=int(self.score),
                                   survival_s=round(self.survival_s, 2))
                        return
                    self.place_leader()
                else:
                    self._emit("lost", "LOST", dist_mm=round(d, 1))
        else:
            self._far_since = None

    def summary(self) -> dict:
        return {"score": int(self.score), "level": self.level, "catches": self.catches,
                "losses": self.losses, "lives": self.lives,
                "follow_s": round(self.follow_s, 2), "follow_frac": round(self.follow_frac, 3),
                "survival_s": round(self.survival_s, 2), "difficulty": self.cfg.difficulty,
                "state": self.state}

    def score_entry(self) -> dict:
        return {"score": int(self.score), "survival_s": round(self.survival_s, 2),
                "level": self.level, "catches": self.catches,
                "follow_frac": round(self.follow_frac, 3)}
