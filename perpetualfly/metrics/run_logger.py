"""Run logging: runs/<YYYY-MM-DD_HH-MM-SS>/{config.json, events.csv, metrics.csv, summary.json}.

* ``config.json``: whatever config is passed (dataclasses / dicts / nested), written
  immediately.
* ``events.csv``: one row per fall-detector transition and per external hit.
* ``metrics.csv``: state sampled at ``LoggingConfig.sample_hz`` of *simulated* time
  (post-step hook gated by step count, so the per-step cost is one modulo).
* ``summary.json``: ``RunMetrics.summary()`` + run info, written on ``close()``
  (and refreshed every ``summary_every_s`` wall seconds so a crash leaves one).

Files are flushed every ``flush_every_s`` wall seconds. ``close()`` is idempotent and
also registered with ``atexit``; use the logger as a context manager or call
``close()`` in a ``finally`` (covers Ctrl-C / KeyboardInterrupt).
"""

from __future__ import annotations

import atexit
import csv
import datetime as _dt
import enum
import json
import math
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import mujoco as mj
import numpy as np

from perpetualfly.metrics.contacts import ContactClassifier
from perpetualfly.metrics.falls import FallDetector, FallEvent, FallState
from perpetualfly.metrics.run_metrics import HitRecord, RunMetrics

if TYPE_CHECKING:
    from perpetualfly.simulation import Simulation

EVENT_COLUMNS = [
    "timestamp",  # monotonic run time (s, continues across resets)
    "sim_time",  # MuJoCo data.time (restarts at reset)
    "wall_time",  # ISO local time
    "event_type",  # hit | destabilized | stabilized | fall | recovering | recovered | relapse | reset | ...
    "force_direction",  # hits only
    "force_magnitude",  # hits only (uN)
    "terrain_type",  # terrain_type_fn(x) at the thorax, "" if unknown
    "fall_detected",  # 1 for a fall event, or if the fly is FALLEN/RECOVERING at the time
    "recovered",  # 1 for a recovered event
    "state",  # fall-detector state after the event
    "x", "y", "z",
    "details",  # free-form (JSON)
]

METRIC_COLUMNS = [
    "timestamp", "sim_time",
    "x", "y", "z", "height",  # thorax position (mm); height above local ground
    "vx", "vy", "vz", "speed_xy",  # instantaneous world-frame velocity (mm/s)
    "window_speed",  # horizontal speed over RunMetrics' window (mm/s)
    "distance",  # accumulated path length (mm)
    "qw", "qx", "qy", "qz",  # thorax orientation (world)
    "roll_deg", "pitch_deg", "yaw_deg",  # intrinsic ZYX (yaw-pitch-roll) Euler angles
    "tilt_deg",  # angle between body up and world up
    "wx", "wy", "wz",  # angular velocity, thorax frame (rad/s)
    "joint_qvel_mean_abs", "joint_qvel_max_abs",  # leg joints (rad/s)
    "joint_qpos_rms_err",  # RMS(actuator target - joint angle) over position actuators (rad)
    "legs_in_contact", "leg_contact_mask",  # 0..6; bitmask lf,lm,lh,rf,rm,rh = bits 0..5
    "body_contact", "n_contacts",
    "state", "terrain_type",
    "falls", "recoveries", "hits",
]

# Column indices written with fixed decimals (see LoggingConfig.abs_decimals).
_ABS_COLUMNS = {"timestamp", "sim_time", "x", "y", "z", "distance"}
_EVENT_ABS = {i for i, c in enumerate(EVENT_COLUMNS) if c in _ABS_COLUMNS}
_METRIC_ABS = {i for i, c in enumerate(METRIC_COLUMNS) if c in _ABS_COLUMNS}


@dataclass
class LoggingConfig:
    enabled: bool = True
    runs_dir: str = "runs"
    sample_hz: float = 50.0  # metrics.csv rows per simulated second
    flush_every_s: float = 2.0  # wall seconds between flushes
    summary_every_s: float = 30.0  # wall seconds between interim summary.json writes
    float_precision: int = 5  # significant digits for most float columns
    # Absolute columns (times, x/y/z, distance) grow without bound in a perpetual run:
    # with 5 significant digits a run time of 1800 s would only resolve 0.1 s (50 Hz
    # rows would share timestamps) and x = 25 m only 1 mm. They use fixed decimals.
    abs_decimals: int = 4


def to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of configs / numpy / enums / paths to JSON types."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None  # JSON has no NaN/inf
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "to_dict"):
        return to_jsonable(obj.to_dict())
    return repr(obj)


def quat_to_rpy_deg(R: np.ndarray) -> tuple[float, float, float]:
    """Roll/pitch/yaw (deg) from a world rotation matrix, R = Rz(yaw) Ry(pitch) Rx(roll)."""
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class RunLogger:
    def __init__(
        self,
        cfg: LoggingConfig | None = None,
        config: Any = None,
        terrain_type_fn: Callable[[float], str] | None = None,
        ground_height_fn: Callable[[float, float], float] | None = None,
        run_name: str | None = None,
    ) -> None:
        self.cfg = cfg or LoggingConfig()
        self.terrain_type_fn = terrain_type_fn
        self.ground_height_fn = ground_height_fn
        self.metrics: RunMetrics | None = None
        self.detector: FallDetector | None = None
        self._sim: "Simulation | None" = None
        self._classifier: ContactClassifier | None = None
        self._closed = False
        self.n_events = 0
        self.n_samples = 0
        self.event_counts: dict[str, int] = {}
        # Extra top-level entries for summary.json (e.g. the app's quit reason).
        self.summary_extra: dict[str, Any] = {}
        self._wall_start = time.time()

        stamp = run_name or _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base = Path(self.cfg.runs_dir)
        run_dir = base / stamp
        i = 1
        while run_dir.exists():  # two runs in the same second
            run_dir = base / f"{stamp}_{i}"
            i += 1
        run_dir.mkdir(parents=True)
        self.run_dir = run_dir

        (run_dir / "config.json").write_text(json.dumps(to_jsonable(config), indent=2))
        # newline="" per csv module docs; line buffering is not used, we flush on a timer.
        self._ev_f = open(run_dir / "events.csv", "w", newline="")
        self._ev = csv.writer(self._ev_f)
        self._ev.writerow(EVENT_COLUMNS)
        self._m_f = open(run_dir / "metrics.csv", "w", newline="")
        self._m = csv.writer(self._m_f)
        self._m.writerow(METRIC_COLUMNS)
        self._flush(force=True)
        self._next_flush = time.monotonic() + self.cfg.flush_every_s
        self._next_summary = time.monotonic() + self.cfg.summary_every_s
        atexit.register(self.close)

    # ------------------------------------------------------------------ wiring
    def attach(
        self,
        sim: "Simulation",
        detector: FallDetector | None = None,
        metrics: RunMetrics | None = None,
    ) -> "RunLogger":
        """Sample metrics every 1/sample_hz sim seconds, log detector events and hits."""
        self._sim = sim
        self._classifier = ContactClassifier(sim.model, sim.fly_name)
        self._every = max(1, int(round(1.0 / (self.cfg.sample_hz * sim.timestep))))
        m = sim.model
        # The fly's free joint is named after the fly; its 6 dofs are qvel[a:a+6]
        # (linear world-frame, angular body-frame). Leg joint dofs follow it.
        free = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, sim.fly_name)
        self._free_qvel = int(m.jnt_dofadr[free])
        self._joint_dofs = slice(self._free_qvel + 6, m.nv)
        # Position actuators (joint transmission + affine bias) -> joint qpos addresses,
        # for a tracking-error summary. Adhesion actuators have a body transmission.
        pos_act = [a for a in range(m.nu) if m.actuator_trntype[a] == mj.mjtTrn.mjTRN_JOINT and m.actuator_biastype[a] != 0]
        self._act_ids = np.array(pos_act, dtype=int)
        self._act_qpos = np.array([m.jnt_qposadr[m.actuator_trnid[a, 0]] for a in pos_act], dtype=int)
        self.detector = detector
        self.metrics = metrics
        if detector is not None:
            detector.add_listener(self.log_fall_event)
        if metrics is not None:
            metrics.hit_listeners.append(self.log_hit)
        sim.post_step_hooks.append(self._hook)
        self._sample(sim)
        return self

    def _hook(self, sim: "Simulation") -> None:
        if sim.step_count % self._every == 0:
            self._sample(sim)

    # ------------------------------------------------------------------ helpers
    def _fmt(self, v: Any, absolute: bool = False) -> Any:
        if isinstance(v, (float, np.floating)):
            if absolute and math.isfinite(v):
                return f"{float(v):.{self.cfg.abs_decimals}f}"
            return f"{float(v):.{self.cfg.float_precision}g}"
        return v

    def _run_time(self, sim_time: float) -> float:
        return self.metrics.run_time_at(sim_time) if self.metrics else sim_time

    def _terrain(self, x: float) -> str:
        if self.terrain_type_fn is None:
            return ""
        try:
            return str(self.terrain_type_fn(x))
        except Exception as e:  # logging must never kill the sim
            return f"error:{type(e).__name__}"

    def _state(self) -> str:
        return self.detector.state.value if self.detector else ""

    def _maybe_flush(self) -> None:
        now = time.monotonic()
        if now >= self._next_flush:
            self._flush()
            self._next_flush = now + self.cfg.flush_every_s
        if now >= self._next_summary:
            self._write_summary(final=False)
            self._next_summary = now + self.cfg.summary_every_s

    def _flush(self, force: bool = False) -> None:
        for f in (self._ev_f, self._m_f):
            if not f.closed:
                f.flush()

    # ------------------------------------------------------------------ events
    def log_event(
        self,
        event_type: str,
        sim_time: float | None = None,
        *,
        force_direction: Any = "",
        force_magnitude: float | None = None,
        fall_detected: bool | None = None,
        recovered: bool = False,
        position: tuple[float, float, float] | None = None,
        details: dict | str | None = None,
    ) -> None:
        if self._closed:
            return
        sim = self._sim
        if sim_time is None:
            sim_time = sim.time if sim is not None else 0.0
        if position is None:
            position = tuple(sim.data.xpos[sim.thorax_body_id]) if sim is not None else (np.nan,) * 3
        state = self._state()
        if fall_detected is None:
            fall_detected = state in (FallState.FALLEN.value, FallState.RECOVERING.value)
        if isinstance(force_direction, (list, tuple, np.ndarray)):
            force_direction = " ".join(f"{float(v):.3g}" for v in force_direction)
        det = details if isinstance(details, str) else json.dumps(to_jsonable(details or {}))
        self._ev.writerow([self._fmt(v, i in _EVENT_ABS) for i, v in enumerate((
            float(self._run_time(sim_time)), float(sim_time),
            _dt.datetime.now().isoformat(timespec="milliseconds"),
            event_type, force_direction if force_direction is not None else "",
            "" if force_magnitude is None else float(force_magnitude),
            self._terrain(float(position[0])),
            int(bool(fall_detected)), int(bool(recovered)), state,
            float(position[0]), float(position[1]), float(position[2]), det,
        ))])
        self.n_events += 1
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
        # Events are rare and important: flush right away.
        self._ev_f.flush()

    def log_fall_event(self, ev: FallEvent) -> None:
        details = {"reason": ev.reason, "from": ev.from_state.value, **ev.details}
        if ev.recovery_time is not None:
            details["recovery_time_s"] = ev.recovery_time
        self.log_event(
            ev.kind, ev.time,
            fall_detected=True if ev.kind in ("fall", "relapse") else None,
            recovered=ev.kind == "recovered",
            position=ev.position, details=details,
        )

    def log_hit(self, hit: HitRecord | Any) -> None:
        """Log a hit. Usually wired via ``RunMetrics.hit_listeners`` (``attach`` does
        that), so perturbation code only has to call ``metrics.record_hit(ev)``."""
        g = (lambda k: hit.get(k)) if isinstance(hit, dict) else (lambda k: getattr(hit, k, None))
        details = {k: g(k) for k in ("duration", "body") if g(k) is not None}
        extra = g("extra")
        if extra:
            skip = {"time", "direction", "magnitude", *details}  # already in columns
            details.update({k: v for k, v in extra.items() if k not in skip})
        self.log_event("hit", g("time"), force_direction=g("direction"),
                       force_magnitude=g("magnitude"), details=details)

    # ------------------------------------------------------------------ metrics
    def _sample(self, sim: "Simulation") -> None:
        if self._closed:
            return
        d, tid = sim.data, sim.thorax_body_id
        x, y, z = (float(v) for v in d.xpos[tid])
        ground = float(self.ground_height_fn(x, y)) if self.ground_height_fn else 0.0
        R = d.xmat[tid].reshape(3, 3)
        roll, pitch, yaw = quat_to_rpy_deg(R)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(R[2, 2])))))
        a = self._free_qvel
        v = d.qvel[a:a + 3]
        w = d.qvel[a + 3:a + 6]
        jq = np.abs(d.qvel[self._joint_dofs])
        err = d.ctrl[self._act_ids] - d.qpos[self._act_qpos] if self._act_ids.size else np.zeros(1)
        cs = self._classifier.summarize(d)
        mask = int(np.dot(cs.leg_contact.astype(int), 1 << np.arange(6)))
        met = self.metrics
        q = d.xquat[tid]
        row = (
            self._run_time(sim.time), sim.time, x, y, z, z - ground,
            float(v[0]), float(v[1]), float(v[2]), float(math.hypot(v[0], v[1])),
            met.current_speed if met else float("nan"),
            met.distance if met else float("nan"),
            float(q[0]), float(q[1]), float(q[2]), float(q[3]),
            roll, pitch, yaw, tilt,
            float(w[0]), float(w[1]), float(w[2]),
            float(jq.mean()) if jq.size else 0.0, float(jq.max()) if jq.size else 0.0,
            float(np.sqrt(np.mean(err ** 2))),
            cs.legs_in_contact, mask, int(cs.body_contact), cs.n_contacts,
            self._state(), self._terrain(x),
            met.n_falls if met else "", met.n_recoveries if met else "",
            met.n_hits if met else "",
        )
        self._m.writerow([self._fmt(v, i in _METRIC_ABS) for i, v in enumerate(row)])
        self.n_samples += 1
        self._maybe_flush()

    # ------------------------------------------------------------------ close
    def _write_summary(self, final: bool) -> None:
        summary: dict[str, Any] = {
            "run_dir": str(self.run_dir),
            "complete": final,
            "wall_time_s": time.time() - self._wall_start,
            "n_metric_samples": self.n_samples,
            "sample_hz": self.cfg.sample_hz,
            "n_events": self.n_events,
            "event_counts": dict(self.event_counts),
        }
        if self.metrics is not None:
            summary["metrics"] = self.metrics.summary()
            summary["hits"] = [
                {"time_s": h.run_time, "direction": h.direction, "magnitude_uN": h.magnitude,
                 "duration_s": h.duration, "impulse_uN_s": h.impulse,
                 "caused_fall": h.caused_fall}
                for h in self.metrics.hits
            ]
        if self.detector is not None:
            summary["final_state"] = self.detector.state.value
        if self._sim is not None:
            summary["sim_time_s"] = self._sim.time
            summary["final_thorax_pos"] = [float(v) for v in self._sim.thorax_position()]
        summary.update(self.summary_extra)
        tmp = self.run_dir / "summary.json.tmp"
        tmp.write_text(json.dumps(to_jsonable(summary), indent=2))
        tmp.replace(self.run_dir / "summary.json")  # atomic: never a half-written file

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._flush()
            self._write_summary(final=True)
        finally:
            self._closed = True
            self._ev_f.close()
            self._m_f.close()
            atexit.unregister(self.close)
            if self._sim is not None and self._hook in self._sim.post_step_hooks:
                self._sim.post_step_hooks.remove(self._hook)

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
