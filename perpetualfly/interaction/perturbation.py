"""External-force "whip": physical shoves applied through ``data.xfrc_applied``.

``Perturbation`` attaches to a ``perpetualfly.simulation.Simulation`` as a pre-step,
post-step and reset hook. ``apply_impulse`` / ``hit`` queue a constant world-frame
force on a body (default: the thorax) for a fixed number of physics steps.

MuJoCo facts this relies on (see docs/API_NOTES.md, section 6):

* ``data.xfrc_applied`` has shape ``(nbody, 6)`` = ``[fx, fy, fz, tx, ty, tz]`` in the
  **world frame**, applied at the body's **centre of mass** (``data.xipos``), in model
  units (uN and uN*mm here, since lengths are mm and masses g).
* It is *not* cleared by ``mj_step``: whatever is written stays applied on every
  subsequent step until overwritten. We therefore clear our contribution ourselves in
  the post-step hook of the last step of a hit, so that ``xfrc_applied`` is zero again
  as soon as the window has elapsed (also visible to anyone inspecting it).
* ``mj_resetDataKeyframe`` (called by ``Simulation.reset``) zeroes ``xfrc_applied``;
  our reset hook drops all active hits so nothing is re-applied afterwards.
* With ``fusestatic`` the head is fused into the thorax body, so a thorax force acts
  on thorax+head; the welded abdomen/wings are separate (jointless) bodies that are
  dragged along through the weld, i.e. the whole rigid body core gets shoved.

Timing is step-count based (``round(duration / timestep)`` physics steps), so pausing
the app (not stepping) simply freezes a hit, and it is independent of rendering.

Magnitudes are expressed internally in multiples of the fly's body weight
(``fly_mass * |g|`` ~= 10.05 uN) and reported in uN.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

import mujoco as mj
import numpy as np

if TYPE_CHECKING:
    from perpetualfly.simulation import Simulation

DIRECTIONS = ("left", "right", "forward", "backward", "up", "random")


@dataclass
class StrengthLevel:
    name: str
    magnitude_bw: float  # force in multiples of body weight
    duration_s: float  # how long the constant force is applied (sim seconds)


def _default_levels() -> list[StrengthLevel]:
    # Empirically calibrated with scripts/demo_perturbation.py --calibrate on flat
    # ground with the default hybrid controller; see docs/PERTURBATION_CALIBRATION.md.
    return [
        StrengthLevel("gentle", 0.75, 0.020),
        StrengthLevel("medium", 1.5, 0.020),
        StrengthLevel("hard", 3.0, 0.020),
        StrengthLevel("absurd", 10.0, 0.025),
    ]


@dataclass
class PerturbationConfig:
    body: str = "thorax"  # "thorax" or a full body name like "nmf/c_thorax"
    # Strength levels 1..4 (index 0 = level 1).
    levels: list[StrengthLevel] = field(default_factory=_default_levels)
    default_level: int = 2
    # "random" direction: uniform horizontal azimuth plus an upward component whose
    # elevation angle (deg above the ground plane) is drawn uniformly from this range.
    random_elevation_deg: tuple[float, float] = (10.0, 35.0)
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PerturbationConfig":
        d = dict(d)
        if "levels" in d:
            d["levels"] = [lv if isinstance(lv, StrengthLevel) else StrengthLevel(**lv)
                           for lv in d["levels"]]
        if "random_elevation_deg" in d:
            d["random_elevation_deg"] = tuple(d["random_elevation_deg"])
        return cls(**d)


@dataclass
class HitEvent:
    sim_time: float  # s, time at which the force starts acting
    step: int  # sim.step_count at which the force starts
    source: str  # "api" | "key" | "auto" ...
    body: str
    direction_name: str
    direction: tuple[float, float, float]  # unit vector, world frame
    magnitude_bw: float
    magnitude_uN: float
    duration_s: float  # actual duration (whole physics steps)
    impulse_uNs: float  # magnitude_uN * duration_s
    level: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _ActiveHit:
    body_id: int
    force: np.ndarray  # (3,) uN world frame
    steps_left: int


def heading_relative_direction(name: str, heading_rad: float) -> np.ndarray:
    """Unit vector (world frame) for a preset direction relative to the fly heading.

    ``heading_rad`` is the yaw of the thorax forward axis projected on the ground
    (``Simulation.heading()``); "left" is +90 deg from it (thorax y axis is left).
    """
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    table = {
        "forward": (c, s, 0.0),
        "backward": (-c, -s, 0.0),
        "left": (-s, c, 0.0),
        "right": (s, -c, 0.0),
        "up": (0.0, 0.0, 1.0),
    }
    if name not in table:
        raise ValueError(f"unknown direction {name!r}; expected one of {DIRECTIONS}")
    return np.array(table[name])


class Perturbation:
    """Applies timed external forces to fly bodies. Attaches itself to ``sim``."""

    def __init__(self, sim: "Simulation", cfg: PerturbationConfig | None = None,
                 attach: bool = True) -> None:
        self.sim = sim
        self.cfg = cfg or PerturbationConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.level = self.cfg.default_level
        self.listeners: list[Callable[[HitEvent], None]] = []
        self.events: list[HitEvent] = []
        self._active: list[_ActiveHit] = []
        # What we last added to xfrc_applied per body, so our contribution can be
        # removed without clobbering forces written by anyone else.
        self._written: dict[int, np.ndarray] = {}
        self._attached = False
        if attach:
            self.attach()

    # ------------------------------------------------------------------ hooks
    def attach(self) -> None:
        if not self._attached:
            self.sim.pre_step_hooks.append(self._pre_step)
            self.sim.post_step_hooks.append(self._post_step)
            self.sim.reset_hooks.append(self._on_reset)
            self._attached = True

    def detach(self) -> None:
        if self._attached:
            self.clear()
            self.sim.pre_step_hooks.remove(self._pre_step)
            self.sim.post_step_hooks.remove(self._post_step)
            self.sim.reset_hooks.remove(self._on_reset)
            self._attached = False

    def _write(self) -> None:
        """Set our contribution to xfrc_applied = sum of active hits per body."""
        xfrc = self.sim.data.xfrc_applied
        totals: dict[int, np.ndarray] = {}
        for h in self._active:
            totals.setdefault(h.body_id, np.zeros(3))
            totals[h.body_id] += h.force
        for bid in set(self._written) | set(totals):
            new = totals.get(bid, np.zeros(3))
            old = self._written.get(bid, np.zeros(3))
            xfrc[bid, :3] += new - old
            if np.any(new):
                self._written[bid] = new
            else:
                self._written.pop(bid, None)

    def _pre_step(self, sim: "Simulation") -> None:
        # Runs right before mj_step: the forces written here act during that step.
        if self._active or self._written:
            self._write()

    def _post_step(self, sim: "Simulation") -> None:
        if not self._active:
            return
        for h in self._active:
            h.steps_left -= 1
        n = len(self._active)
        self._active = [h for h in self._active if h.steps_left > 0]
        if len(self._active) != n:
            # A hit just ended: remove its force now (xfrc_applied would otherwise
            # persist into the next mj_step and be visible to observers).
            self._write()

    def _on_reset(self, sim: "Simulation") -> None:
        # mj_resetDataKeyframe already zeroed xfrc_applied; forget our bookkeeping.
        self._active.clear()
        self._written.clear()

    def clear(self) -> None:
        """Cancel all active hits and remove their forces immediately."""
        self._active.clear()
        self._write()

    # -------------------------------------------------------------- queries
    @property
    def body_weight_uN(self) -> float:
        g = float(np.linalg.norm(self.sim.model.opt.gravity))
        return self.sim.fly_mass * g

    @property
    def is_active(self) -> bool:
        return bool(self._active)

    @property
    def last_event(self) -> HitEvent | None:
        return self.events[-1] if self.events else None

    @property
    def level_name(self) -> str:
        return self.cfg.levels[self.level - 1].name

    def active_force(self, body: str = "thorax") -> np.ndarray:
        """Sum of our currently applied force on ``body`` (uN, world frame)."""
        return self._written.get(self._body_id(body), np.zeros(3)).copy()

    def set_level(self, level: int) -> None:
        if not 1 <= level <= len(self.cfg.levels):
            raise ValueError(f"level must be 1..{len(self.cfg.levels)}")
        self.level = level

    # ------------------------------------------------------------- actions
    def _body_id(self, body: str) -> int:
        # Accept "thorax" -> "nmf/c_thorax", "lf_tibia" -> "nmf/lf_tibia", or full names.
        fly = self.sim.fly_name
        candidates = [body] if "/" in body else [f"{fly}/c_{body}", f"{fly}/{body}"]
        for name in candidates:
            bid = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                return bid
        raise ValueError(f"no body named {body!r} (tried {candidates})")

    def resolve_direction(self, direction: str | Sequence[float],
                          rng: np.random.Generator | None = None) -> tuple[str, np.ndarray]:
        """Return (name, world unit vector). Presets are relative to the fly heading;
        an explicit 3-vector is taken as world-frame and normalized."""
        if isinstance(direction, str):
            if direction == "random":
                rng = rng or self.rng
                az = rng.uniform(-math.pi, math.pi)
                lo, hi = self.cfg.random_elevation_deg
                el = math.radians(rng.uniform(lo, hi))
                yaw = self.sim.heading() + az
                vec = np.array([math.cos(el) * math.cos(yaw),
                                math.cos(el) * math.sin(yaw), math.sin(el)])
                return "random", vec
            return direction, heading_relative_direction(direction, self.sim.heading())
        vec = np.asarray(direction, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vec))
        if not norm > 0:
            raise ValueError("direction vector must be non-zero")
        return "vector", vec / norm

    def apply_impulse(
        self,
        body: str = "thorax",
        direction: str | Sequence[float] = "random",
        magnitude: float | None = None,
        duration: float | None = None,
        *,
        level: int | None = None,
        source: str = "api",
        rng: np.random.Generator | None = None,
    ) -> HitEvent:
        """Apply a constant force for ``duration`` sim seconds, starting next step.

        ``magnitude`` is in multiples of body weight. Missing magnitude/duration are
        taken from ``level`` (or the current strength level). Overlapping hits add up.
        """
        lvl = level if level is not None else self.level
        preset = self.cfg.levels[lvl - 1]
        mag_bw = preset.magnitude_bw if magnitude is None else float(magnitude)
        dur = preset.duration_s if duration is None else float(duration)
        if mag_bw < 0 or dur <= 0:
            raise ValueError("magnitude must be >= 0 and duration > 0")
        # Recorded level is only meaningful if the preset values were used.
        rec_level = lvl if (magnitude is None and duration is None) else level
        name, vec = self.resolve_direction(direction, rng)
        bid = self._body_id(body)
        steps = max(1, int(round(dur / self.sim.timestep)))
        mag_uN = mag_bw * self.body_weight_uN
        self._active.append(_ActiveHit(bid, vec * mag_uN, steps))
        actual_dur = steps * self.sim.timestep
        ev = HitEvent(
            sim_time=self.sim.time,
            step=self.sim.step_count,
            source=source,
            body=self.sim.model.body(bid).name,
            direction_name=name,
            direction=tuple(float(v) for v in vec),
            magnitude_bw=mag_bw,
            magnitude_uN=mag_uN,
            duration_s=actual_dur,
            impulse_uNs=mag_uN * actual_dur,
            level=rec_level,
        )
        self.events.append(ev)
        for listener in list(self.listeners):
            listener(ev)
        return ev

    def hit(self, direction: str | Sequence[float] = "random", level: int | None = None,
            source: str = "api", rng: np.random.Generator | None = None) -> HitEvent:
        """Thorax hit using a strength level (default: current level)."""
        return self.apply_impulse(self.cfg.body, direction, level=level, source=source,
                                  rng=rng)


# ---------------------------------------------------------------------------
# Automatic random perturbations
# ---------------------------------------------------------------------------


@dataclass
class AutoPerturbConfig:
    enabled: bool = True
    min_interval_s: float = 2.0  # sim seconds between hits (uniform in [min, max])
    max_interval_s: float = 5.0
    first_hit_after_s: float = 1.0  # delay after start / reset before the first hit
    # Strength levels to draw from and their relative weights.
    levels: tuple[int, ...] = (1, 2)
    level_weights: tuple[float, ...] = (0.6, 0.4)
    # Optional override: if set, magnitude (body weights) is drawn uniformly from
    # this range instead of using the level preset's magnitude (duration still from
    # the drawn level).
    magnitude_bw_range: tuple[float, float] | None = None
    direction_weights: dict[str, float] = field(default_factory=lambda: {
        "left": 0.3, "right": 0.3, "forward": 0.1, "backward": 0.1, "up": 0.05,
        "random": 0.15,
    })
    seed: int = 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AutoPerturbConfig":
        d = dict(d)
        for k in ("levels", "level_weights", "magnitude_bw_range"):
            if d.get(k) is not None:
                d[k] = tuple(d[k])
        return cls(**d)


class AutoPerturber:
    """Fires random hits at random intervals (post-step hook, step-count based)."""

    def __init__(self, sim: "Simulation", perturbation: Perturbation,
                 cfg: AutoPerturbConfig | None = None, attach: bool = True) -> None:
        self.sim = sim
        self.perturbation = perturbation
        self.cfg = cfg or AutoPerturbConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.enabled = self.cfg.enabled
        names = list(self.cfg.direction_weights)
        w = np.array([self.cfg.direction_weights[n] for n in names], dtype=float)
        if len(names) == 0 or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("direction_weights must be non-negative with positive sum")
        for n in names:
            if n not in DIRECTIONS:
                raise ValueError(f"unknown direction {n!r}")
        self._dir_names, self._dir_p = names, w / w.sum()
        lw = np.array(self.cfg.level_weights, dtype=float)
        if len(lw) != len(self.cfg.levels) or lw.sum() <= 0:
            raise ValueError("levels and level_weights must match")
        self._level_p = lw / lw.sum()
        self._next_step = self._steps(self.cfg.first_hit_after_s) + sim.step_count
        self._attached = False
        if attach:
            self.attach()

    def _steps(self, seconds: float) -> int:
        return max(1, int(round(seconds / self.sim.timestep)))

    def attach(self) -> None:
        if not self._attached:
            self.sim.post_step_hooks.append(self._post_step)
            self.sim.reset_hooks.append(self._on_reset)
            self._attached = True

    def detach(self) -> None:
        if self._attached:
            self.sim.post_step_hooks.remove(self._post_step)
            self.sim.reset_hooks.remove(self._on_reset)
            self._attached = False

    def toggle(self) -> bool:
        self.enabled = not self.enabled
        if self.enabled:  # don't fire immediately after re-enabling
            self._schedule_next()
        return self.enabled

    @property
    def next_hit_time(self) -> float:
        """Sim time of the next scheduled hit (valid while enabled)."""
        return self.sim.time + (self._next_step - self.sim.step_count) * self.sim.timestep

    def _schedule_next(self) -> None:
        dt = self.rng.uniform(self.cfg.min_interval_s, self.cfg.max_interval_s)
        self._next_step = self.sim.step_count + self._steps(dt)

    def _on_reset(self, sim: "Simulation") -> None:
        self._next_step = self._steps(self.cfg.first_hit_after_s)

    def _post_step(self, sim: "Simulation") -> None:
        if not self.enabled or sim.step_count < self._next_step:
            return
        self.fire()

    def fire(self) -> HitEvent:
        """Draw and apply one random hit now, then schedule the next one."""
        rng = self.rng
        direction = str(rng.choice(self._dir_names, p=self._dir_p))
        level = int(rng.choice(self.cfg.levels, p=self._level_p))
        magnitude = None
        if self.cfg.magnitude_bw_range is not None:
            magnitude = float(rng.uniform(*self.cfg.magnitude_bw_range))
        ev = self.perturbation.apply_impulse(
            self.perturbation.cfg.body, direction, magnitude=magnitude,
            level=level, source="auto", rng=rng,
        )
        self._schedule_next()
        return ev
