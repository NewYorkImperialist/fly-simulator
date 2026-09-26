"""Run-level metrics: distance/speed + falls, recoveries, hits, jogging intervals.

``RunMetrics`` composes ``LocomotionStats`` and listens to ``FallDetector`` events.
Perturbation code reports hits with ``metrics.record_hit(event)``.

Distance: ``distance`` is the total xy path of the thorax (flights after hits
included, kept for compatibility); ``walked_distance`` only counts path segments
during which the fly was walking: detector state UPRIGHT / DESTABILIZED *and* at
least one leg touching the terrain at every metric update (every 10 ms), so a hit
that launches the fly (or a fall) adds nothing. ``average_speed`` = walked distance
/ run time; the old total-path average is ``average_speed_total``.

Counting survives explicit sim resets: on a reset the current locomotion stats are
banked into totals, the open jogging interval is closed and a fresh one starts
(a reset while FALLEN leaves that fall unrecovered).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from fly_simulator.metrics.falls import FallDetector, FallEvent, FallState
from fly_simulator.metrics.locomotion import LocomotionStats

if TYPE_CHECKING:
    from fly_simulator.simulation import Simulation


@dataclass
class HitRecord:
    """A normalized external hit. ``record_hit`` accepts anything with (a subset of)
    these attributes / dict keys: time, direction, magnitude, duration, body."""

    time: float  # sim time (s)
    run_time: float  # monotonic run time (s, continues across resets)
    direction: Any = None  # e.g. "left" or a 3-vector
    magnitude: float | None = None  # force (uN)
    duration: float | None = None  # s
    body: str | None = None
    caused_fall: bool = False  # a fall followed within hit_fall_window_s
    extra: dict = field(default_factory=dict)

    @property
    def impulse(self) -> float | None:  # uN*s
        if self.magnitude is None or self.duration is None:
            return None
        return float(self.magnitude) * float(self.duration)


def _get(ev: Any, key: str, default=None):
    if ev is None:
        return default
    if isinstance(ev, dict):
        return ev.get(key, default)
    return getattr(ev, key, default)


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return [float(x) for x in v.ravel()]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def walking_now(detector: FallDetector) -> bool:
    """Walked-distance criterion: UPRIGHT / DESTABILIZED and a leg on the terrain."""
    if detector.state not in (FallState.UPRIGHT, FallState.DESTABILIZED):
        return False
    c = detector.last_contacts
    return c is None or c.legs_in_contact > 0


class RunMetrics:
    SURVIVAL_HORIZONS_S = (5.0, 10.0, 30.0, 60.0)

    def __init__(
        self,
        speed_window_s: float = 0.5,
        update_every_steps: int = 100,
        hit_fall_window_s: float = 2.0,
    ) -> None:
        self.update_every_steps = update_every_steps
        self.hit_fall_window_s = hit_fall_window_s
        self.stats = LocomotionStats(speed_window_s)
        self.hit_listeners: list[Callable[[HitRecord], None]] = []
        self.state = FallState.UPRIGHT
        # Banked totals from before the last reset.
        self._bank_time = 0.0
        self._bank_path = 0.0
        self._bank_fwd = 0.0
        self.n_resets = 0
        self.falls: list[float] = []  # run times of falls
        self.recovery_times: list[float] = []
        self.n_destabilized = 0
        self.n_relapses = 0
        self.hits: list[HitRecord] = []
        self.jog_intervals: list[float] = []  # completed uninterrupted jogging intervals
        self._jog_start: float | None = 0.0  # run time; None while fallen
        self._sim_time = 0.0
        self._sim_t0: float | None = None  # sim time at (re)start
        # walked distance (see module doc); walking_fn() -> "walking right now"
        self.walking_fn: Callable[[], bool] | None = None
        self.path_sample_s = self.stats.path_sample_s
        self._bank_walk = 0.0
        self._walk_accum = 0.0
        self._walk_time = 0.0
        self._walk_anchor: tuple[float, np.ndarray] | None = None
        self._walk_clean = True
        self._walk_last: tuple[float, np.ndarray] | None = None

    # ------------------------------------------------------------------ wiring
    def attach(self, sim: "Simulation", detector: FallDetector | None = None) -> "RunMetrics":
        """Post-step hook (every ``update_every_steps``) + reset hook + detector listener."""
        sim.post_step_hooks.append(self._hook)
        sim.reset_hooks.append(self._on_sim_reset)
        if detector is not None:
            detector.add_listener(self.on_fall_event)
            self.state = detector.state
            self.walking_fn = lambda: walking_now(detector)
        self.update(sim.time, sim.thorax_position(), walking=True)
        return self

    def _hook(self, sim: "Simulation") -> None:
        if sim.step_count % self.update_every_steps == 0:
            self.update(sim.time, sim.thorax_position())

    def _on_sim_reset(self, sim: "Simulation") -> None:
        self.notify_reset(sim.time, sim.thorax_position())

    # ------------------------------------------------------------------ time
    def run_time_at(self, sim_time: float) -> float:
        """Monotonic time since the run started (continues across resets)."""
        t0 = self._sim_t0 if self._sim_t0 is not None else sim_time
        return self._bank_time + (sim_time - t0)

    @property
    def run_time(self) -> float:
        return self.run_time_at(self._sim_time)

    # ------------------------------------------------------------------ inputs
    def update(self, sim_time: float, pos: np.ndarray, walking: bool | None = None) -> None:
        """``walking``: None = ask ``walking_fn`` (True without one)."""
        if self._sim_t0 is None:
            self._sim_t0 = sim_time
        self._sim_time = sim_time
        self.stats.update(sim_time, pos)
        if walking is None:
            walking = self.walking_fn() if self.walking_fn is not None else True
        self._update_walk(sim_time, np.asarray(pos, dtype=float), bool(walking))

    def _update_walk(self, t: float, pos: np.ndarray, walking: bool) -> None:
        # Same sampling as LocomotionStats.path_length (segments >= path_sample_s,
        # so the gait's lateral sway is not counted), but a segment only counts if
        # the fly was walking at every update inside it.
        if self._walk_anchor is None or self._walk_last is None:
            self._walk_anchor, self._walk_last, self._walk_clean = (t, pos.copy()), (t, pos.copy()), walking
            return
        if walking and t > self._walk_last[0]:
            self._walk_time += t - self._walk_last[0]
        self._walk_clean = self._walk_clean and walking
        if t - self._walk_anchor[0] >= self.path_sample_s:
            if self._walk_clean:
                self._walk_accum += float(np.linalg.norm(pos[:2] - self._walk_anchor[1][:2]))
            self._walk_anchor, self._walk_clean = (t, pos.copy()), walking
        self._walk_last = (t, pos.copy())

    def _walked_here(self) -> float:
        """Walked distance since the last reset (open segment included if clean)."""
        if self._walk_anchor is None or self._walk_last is None or not self._walk_clean:
            return self._walk_accum
        return self._walk_accum + float(np.linalg.norm(self._walk_last[1][:2]
                                                       - self._walk_anchor[1][:2]))

    def notify_reset(self, sim_time: float, pos: np.ndarray) -> None:
        rt = self.run_time
        self._bank_time = rt
        self._bank_path += self.stats.path_length
        self._bank_fwd += self.stats.forward_displacement
        self._bank_walk += self._walked_here()
        self._walk_accum = 0.0
        self._walk_anchor = self._walk_last = None
        self.n_resets += 1
        self._close_jog(rt)
        self._jog_start = rt
        self.state = FallState.UPRIGHT
        self.stats.reset()
        self._sim_t0 = None
        self.update(sim_time, pos, walking=True)

    def on_fall_event(self, ev: FallEvent) -> None:
        if ev.kind == "reset":
            return  # handled by notify_reset (sim reset hook)
        rt = self.run_time_at(ev.time)
        self._sim_time = max(self._sim_time, ev.time)
        self.state = ev.to_state
        if ev.kind == "destabilized":
            self.n_destabilized += 1
        elif ev.kind == "fall":
            self.falls.append(rt)
            self._close_jog(rt)
            for h in self.hits:
                if 0.0 <= rt - h.run_time <= self.hit_fall_window_s:
                    h.caused_fall = True
        elif ev.kind == "relapse":
            self.n_relapses += 1
        elif ev.kind == "recovered":
            if ev.recovery_time is not None:
                self.recovery_times.append(float(ev.recovery_time))
            self._jog_start = rt

    def record_hit(self, event: Any = None, **kw) -> HitRecord:
        """Count an external hit. ``event`` may be a dataclass/object/dict with
        ``time, direction, magnitude, duration, body``; keywords override. If no time
        is given, the latest known sim time is used."""
        def g(k):
            return kw[k] if k in kw else _get(event, k)

        t = g("time")
        t = self._sim_time if t is None else float(t)
        extra = {}
        if is_dataclass(event) and not isinstance(event, type):
            extra = {k: _jsonable(v) for k, v in asdict(event).items()}
        elif isinstance(event, dict):
            extra = {k: _jsonable(v) for k, v in event.items()}
        mag = g("magnitude")
        dur = g("duration")
        rec = HitRecord(
            time=t, run_time=self.run_time_at(t), direction=_jsonable(g("direction")),
            magnitude=None if mag is None else float(mag),
            duration=None if dur is None else float(dur),
            body=g("body"), extra=extra,
        )
        self.hits.append(rec)
        for fn in list(self.hit_listeners):
            fn(rec)
        return rec

    # ------------------------------------------------------------------ derived
    def _close_jog(self, rt: float) -> None:
        if self._jog_start is not None:
            self.jog_intervals.append(max(0.0, rt - self._jog_start))
        self._jog_start = None

    @property
    def n_falls(self) -> int:
        return len(self.falls)

    @property
    def n_recoveries(self) -> int:
        return len(self.recovery_times)

    @property
    def n_hits(self) -> int:
        return len(self.hits)

    @property
    def distance(self) -> float:
        return self._bank_path + self.stats.path_length  # mm

    @property
    def forward_displacement(self) -> float:
        return self._bank_fwd + self.stats.forward_displacement

    @property
    def walked_distance(self) -> float:
        """mm walked (no flights / falls; see module doc), summed over resets."""
        return self._bank_walk + self._walked_here()

    @property
    def walking_time(self) -> float:
        """Sim seconds spent walking (same criterion as walked_distance)."""
        return self._walk_time

    @property
    def walking_speed(self) -> float:
        """Walked distance / walking time (mm/s): speed while actually walking."""
        return self.walked_distance / self._walk_time if self._walk_time > 0 else 0.0

    @property
    def average_speed(self) -> float:
        """Walked distance / run time (mm/s); flights after hits don't count."""
        return self.walked_distance / self.run_time if self.run_time > 0 else 0.0

    @property
    def average_speed_total(self) -> float:
        """Total path (flights included) / run time: the pre-A9 ``average_speed``."""
        return self.distance / self.run_time if self.run_time > 0 else 0.0

    @property
    def current_speed(self) -> float:
        return self.stats.current_speed

    @property
    def current_jog_interval(self) -> float:
        return 0.0 if self._jog_start is None else max(0.0, self.run_time - self._jog_start)

    @property
    def longest_jog_interval(self) -> float:
        return max([self.current_jog_interval, *self.jog_intervals])

    @property
    def mean_time_between_failures(self) -> float | None:
        """Total run time / falls (None with no falls)."""
        return self.run_time / self.n_falls if self.n_falls else None

    @property
    def recovery_percentage(self) -> float | None:
        return 100.0 * self.n_recoveries / self.n_falls if self.n_falls else None

    @property
    def falls_per_km(self) -> float | None:
        km = self.distance / 1e6
        return self.n_falls / km if km > 0 else None

    @property
    def mean_recovery_time(self) -> float | None:
        return float(np.mean(self.recovery_times)) if self.recovery_times else None

    def survival_probability(self, horizon_s: float) -> float | None:
        """Fraction of jogging intervals (completed + current, if long enough to decide)
        that lasted >= horizon_s. Intervals still open and shorter are undecided."""
        decided = list(self.jog_intervals)
        if self.current_jog_interval >= horizon_s:
            decided.append(self.current_jog_interval)
        if not decided:
            return None
        return float(np.mean([d >= horizon_s for d in decided]))

    def summary(self) -> dict[str, Any]:
        survived = [h for h in self.hits if not h.caused_fall]
        impulses = [h.impulse for h in survived if h.impulse is not None]
        mags = [h.magnitude for h in survived if h.magnitude is not None]
        return {
            "run_time_s": self.run_time,
            "distance_mm": self.distance,  # total path, flights included
            "walked_distance_mm": self.walked_distance,
            "walking_time_s": self.walking_time,
            "walking_speed_mm_s": self.walking_speed,
            "forward_displacement_mm": self.forward_displacement,
            "average_speed_mm_s": self.average_speed,  # walked distance / run time
            "average_speed_total_mm_s": self.average_speed_total,
            "current_speed_mm_s": self.current_speed,
            "state": self.state.value,
            "n_falls": self.n_falls,
            "n_recoveries": self.n_recoveries,
            "n_relapses": self.n_relapses,
            "n_destabilized": self.n_destabilized,
            "n_hits": self.n_hits,
            "n_hits_survived": len(survived),
            "n_resets": self.n_resets,
            "longest_jog_interval_s": self.longest_jog_interval,
            "current_jog_interval_s": self.current_jog_interval,
            "jog_intervals_s": list(self.jog_intervals),
            "fall_times_s": list(self.falls),
            "recovery_times_s": list(self.recovery_times),
            "mean_recovery_time_s": self.mean_recovery_time,
            "mean_time_between_failures_s": self.mean_time_between_failures,
            "recovery_percentage": self.recovery_percentage,
            "falls_per_km": self.falls_per_km,
            "max_force_survived_uN": max(mags) if mags else None,
            "max_impulse_survived_uN_s": max(impulses) if impulses else None,
            "p_survive": {f"{h:g}s": self.survival_probability(h)
                          for h in self.SURVIVAL_HORIZONS_S},
        }

    def summary_line(self) -> str:
        return (
            f"t={self.run_time:8.2f}s  dist={self.distance:8.1f}mm "
            f"(walked {self.walked_distance:8.1f})  "
            f"speed now={self.current_speed:5.1f} avg={self.average_speed:5.1f} mm/s  "
            f"falls={self.n_falls} rec={self.n_recoveries} hits={self.n_hits}  "
            f"jog={self.current_jog_interval:6.1f}s (max {self.longest_jog_interval:6.1f})  "
            f"[{self.state.value}]"
        )
