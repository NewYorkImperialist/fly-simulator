"""``EternalJob``: base class of the "fly doing an absurd job forever" scenes.

Lifecycle (see docs/JOBS.md for a worked example)::

    job = get_job("sisyphus")()                 # or make_job("sisyphus", cfg)
    job.configure_app(app_cfg)                  # spawn height, terrain, fall thresholds
    session = Session(app_cfg, world_extensions=[job.extension])  # props compiled in
    job.attach(session)                         # hooks, ids, steering (install_job does this)
    ... session.sim.step(n); session.after_physics() ...   # job.after_physics is chained

Phase 1 (before compile): ``configure_app`` and ``extension(world)`` (adds the props
to the MjSpec; use fly_simulator.jobs.geometry for the contact bits).
Phase 2 (compiled): ``attach`` registers

* a post-step hook that calls ``update()`` every ``cfg.update_every_steps`` physics
  steps (job logic: steering, counters, prop checks; runs inside ``sim.step``, so
  it must not reset the sim);
* a reset hook -> ``on_reset()`` + ``reset_props()`` (FlyGym's keyframe reset already
  put every prop body back at its spec pose; jobs may re-randomise here);
* fall-detector listeners (falls / self-rightings counted);
* ``Steering`` as the locomotion controller's ``signal_filter`` (heading target +
  speed, turn-in-place for large errors);
* the job's ``ground_height`` in the fall detector / logger (a fly on a hill is not
  "airborne"), and optionally ``progress_xy`` for treadmill-like jobs.

``after_physics()`` runs once per physics chunk *outside* ``sim.step`` (the runner and
the app's loop call ``session.after_physics``, which ``install_job`` chains): it does
the eternal-operation part, an explicit, counted, logged reset
(``session.reset("auto")``) when the fly has been down for ``recover_after_s`` or the
job made no progress for ``stuck_timeout_s``. Never a hidden teleport of the fly.

Constant memory: jobs keep counters, not per-event lists; props are pooled (a fixed
set of bodies compiled once and recycled); the action history is trimmed.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from fly_simulator.jobs.geometry import wrap_angle

if TYPE_CHECKING:
    from fly_simulator.app import Session
    from fly_simulator.config import AppConfig
    from fly_simulator.simulation import Simulation


# ---------------------------------------------------------------------------
# config / camera
# ---------------------------------------------------------------------------


@dataclass
class JobConfig:
    """Settings every job has. Subclass (as a dataclass) for job parameters."""

    update_every_steps: int = 10  # job logic every N physics steps (10 = 1 ms sim)
    recover_after_s: float = 2.5  # fly down (FALLEN / RECOVERING) this long -> explicit reset
    stuck_timeout_s: float = 120.0  # no work progress this long -> explicit reset (0 = off)
    # steering (Steering): P gain on the heading error (per rad), clipped to
    # +-max_turn; above spin_above_deg of error the fly turns on the spot
    steer_gain: float = 2.5
    steer_max_turn: float = 0.8
    spin_above_deg: float = 50.0
    spin_amp: float = 0.9
    history_keep: int = 200  # ActionManager.history entries kept (constant memory)
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CameraPreset:
    """Framing of the job view (``JobCamera`` "job" mode).

    ``azimuth`` / ``elevation`` in degrees (MuJoCo free camera: looks along
    (cos el cos az, cos el sin az, sin el)); ``azimuth=None`` follows the fly's
    heading + ``azimuth_offset``. The look-at point is ``EternalJob.camera_target()``,
    smoothed with ``tau_s``."""

    azimuth: float | None = 90.0
    elevation: float = -20.0
    distance: float = 14.0
    azimuth_offset: float = -45.0
    tau_s: float = 0.4


def enable_flight(app_cfg: "AppConfig") -> None:
    """The flight fly for a ``needs_flight`` job: ``cfg.flight.enabled``, dt 5e-5 s and
    the doubled ``render_every_steps`` (same frames per simulated second), exactly as
    the app does for ``--flight`` (``apply_feature_defaults``; idempotent)."""
    from fly_simulator.config import RenderConfig
    from fly_simulator.flight import FLIGHT_TIMESTEP

    app_cfg.flight.enabled = True
    if app_cfg.sim.timestep != FLIGHT_TIMESTEP:
        if app_cfg.render.render_every_steps == RenderConfig().render_every_steps:
            app_cfg.render.render_every_steps *= 2
        app_cfg.sim.timestep = FLIGHT_TIMESTEP
    app_cfg.sim.control_every_steps = 1
    app_cfg.fly.extra_joints = False  # the flight body has its own wings


def format_uptime(seconds: float) -> str:
    """'Day N HH:MM:SS' (Day 1 = the first 24 h)."""
    s = max(0.0, float(seconds)) if math.isfinite(seconds) else 0.0
    day, rem = divmod(int(s), 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    return f"Day {day + 1} {h:02d}:{m:02d}:{sec:02d}"


# ---------------------------------------------------------------------------
# steering
# ---------------------------------------------------------------------------


class Steering:
    """Heading target + speed for the hybrid walking controller.

    Installed as ``controller.signal_filter`` (chained in front of an existing one,
    e.g. the brain's). ``target`` = world yaw (rad) or None (leave the controller's
    own heading hold alone); ``speed`` scales both CPG amplitudes (0 = stand).
    Heading error -> [1 + d, 1 - d] with d = clip(gain * err, +-max_turn); above
    ``spin_above`` rad of error the two sides step in opposite directions (turn on the
    spot, like ``TurnInPlace``). Actions (``ActionManager``) still override it.
    """

    def __init__(self, sim: "Simulation", gain: float = 2.0, max_turn: float = 0.6,
                 spin_above: float = math.radians(75), spin_amp: float = 0.9) -> None:
        self.sim = sim
        self.gain, self.max_turn = float(gain), float(max_turn)
        self.spin_above, self.spin_amp = float(spin_above), float(spin_amp)
        self.target: float | None = None
        self.speed = 1.0
        self.last_error = 0.0
        ctrl = sim.controller
        self._prev = ctrl.signal_filter
        ctrl.signal_filter = self
        self._installed = True

    def set(self, heading: float | None, speed: float = 1.0) -> None:
        self.target = None if heading is None or not math.isfinite(heading) else float(heading)
        self.speed = float(speed) if math.isfinite(speed) else 0.0

    def aim_at(self, point_xy, speed: float = 1.0) -> float:
        """Head for a world (x, y) point; returns the distance to it (mm)."""
        p = self.sim.data.xpos[self.sim.thorax_body_id]
        dx, dy = float(point_xy[0] - p[0]), float(point_xy[1] - p[1])
        self.set(math.atan2(dy, dx), speed)
        return math.hypot(dx, dy)

    def __call__(self, hold: np.ndarray) -> np.ndarray:
        if self.target is None:
            out = hold * self.speed
        else:
            err = wrap_angle(self.sim.heading() - self.target)  # > 0: fly points left
            self.last_error = err
            if abs(err) > self.spin_above:
                a = self.spin_amp
                out = np.array([a, -a]) if err > 0 else np.array([-a, a])
                out = out * max(self.speed, 0.6)  # turning on the spot needs amplitude
            else:
                d = min(max(self.gain * err, -self.max_turn), self.max_turn)
                # slow down in sharp turns (tighter turning circle)
                slow = 1.0 - 0.4 * abs(err) / max(self.spin_above, 1e-6)
                out = np.array([1.0 + d, 1.0 - d]) * (self.speed * slow)
        return out if self._prev is None else self._prev(out)

    def uninstall(self) -> None:
        if self._installed and self.sim.controller.signal_filter is self:
            self.sim.controller.signal_filter = self._prev
        self._installed = False


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------


def _make_action(name: str, **params):
    from fly_simulator.actions import make_action

    return make_action(name, **params)


class EternalJob:
    """Base class. Subclasses set ``name`` / ``title`` / ``work_label`` /
    ``config_cls`` and implement ``extension``, ``on_attach``, ``update``; usually
    also ``ground_height``, ``reset_props``, ``camera_target``, ``camera_preset``,
    ``job_hud_lines`` and ``job_stats``. Register with ``@register_job``."""

    name = "job"
    title = "ETERNAL JOB"
    tagline = ""
    work_label = "work done"
    work_format = "{:.0f}"
    config_cls: type[JobConfig] = JobConfig
    #: body / geom names that must exist in the compiled model (install_job checks)
    required_names: tuple[str, ...] = ()
    #: True: the job runs on the real flight fly (``--flight``: flapping wings, air,
    #: dt 5e-5 s; docs/FLIGHT.md). ``configure_app`` then turns ``cfg.flight`` on and the
    #: app allows ``--job NAME`` with (or without) ``--flight`` for this job only.
    needs_flight: bool = False

    def __init__(self, cfg: JobConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else self.config_cls()
        self.session: "Session | None" = None
        self.sim: "Simulation | None" = None
        self.steering: Steering | None = None
        self.state = "starting"
        self.work: float = 0.0
        self.n_falls = 0
        self.n_self_righted = 0
        self.n_auto_recoveries = 0
        self.n_resets = 0  # every sim reset seen (auto + manual)
        self.recovery_reasons: dict[str, int] = {}
        self.last_work_rt = 0.0  # run time of the last add_work
        self.wall_start = time.perf_counter()
        self._step_div = 0
        self._attached = False
        self.fly_name = "nmf"  # set from the app config in configure_app
        self.n_unstuck = 0  # back-away manoeuvres triggered by unstick()
        self._track: deque = deque(maxlen=64)  # (sim t, x, y) every 0.1 s (fixed memory)

    # ------------------------------------------------------ phase 1: build
    def configure_app(self, app_cfg: "AppConfig") -> None:
        """Adjust the app config before the Session is built (called once). Base:
        flat terrain (no procedural obstacles), no automatic hits."""
        self.fly_name = app_cfg.fly.name
        app_cfg.terrain.difficulty = "flat"
        app_cfg.auto_perturb.enabled = False
        app_cfg.session.auto_reset_after_s = None  # the job does its own recovery
        # floor reflections draw the checker plane over thin "paint" boxes (and cost
        # ~5 ms per frame)
        app_cfg.render.reflections = False
        if self.needs_flight:
            enable_flight(app_cfg)

    def extension(self, world) -> None:
        """World extension: add the props to ``world.mjcf_root`` (before add_fly)."""
        raise NotImplementedError

    def ground_height(self, x: float, y: float) -> float:
        """Height of the walkable surface under (x, y) (mm); fall detector / camera."""
        return 0.0

    def progress_xy(self, x: float, y: float) -> tuple[float, float] | None:
        """Override for treadmill-like jobs: the thorax (x, y) in the frame the fly
        walks in (e.g. + wheel surface travel), used by the fall detector's
        stall / no-progress rules. None = world frame."""
        return None

    # ------------------------------------------------------ phase 2: run
    def attach(self, session: "Session") -> "EternalJob":
        if self._attached:
            return self
        self.session, self.sim = session, session.sim
        sim = self.sim
        c = self.cfg
        self.steering = Steering(sim, c.steer_gain, c.steer_max_turn,
                                 math.radians(c.spin_above_deg), c.spin_amp)
        det = session.detector
        det.ground_height_fn = self.ground_height
        if session.logger is not None:
            session.logger.ground_height_fn = self.ground_height
        if type(self).progress_xy is not EternalJob.progress_xy:
            orig = det._progress_speed

            def progress(t, x, y, _orig=orig):
                xy = self.progress_xy(x, y)
                return _orig(t, x, y) if xy is None else _orig(t, *xy)

            det._progress_speed = progress
        det.add_listener(self._on_fall_event)
        sim.post_step_hooks.append(self._post_step)
        sim.reset_hooks.append(self._on_sim_reset)
        self.wall_start = time.perf_counter()
        self.last_work_rt = session.run_time()
        self.on_attach()
        self._attached = True
        return self

    def detach(self) -> None:
        if not self._attached:
            return
        sim = self.sim
        if self._post_step in sim.post_step_hooks:
            sim.post_step_hooks.remove(self._post_step)
        if self._on_sim_reset in sim.reset_hooks:
            sim.reset_hooks.remove(self._on_sim_reset)
        if self.steering is not None:
            self.steering.uninstall()
        self._attached = False

    def on_attach(self) -> None:
        """Look up body / geom / joint ids by name here (the model is compiled)."""

    def update(self) -> None:
        """Job logic, every ``cfg.update_every_steps`` physics steps (inside
        sim.step: set steering, trigger actions, count work; don't reset)."""

    def on_reset(self) -> None:
        """After a sim reset (fly respawned, props back at their spec pose)."""

    def reset_props(self) -> None:
        """Regenerate the props (also called by the job when a prop is lost)."""

    def camera_target(self) -> np.ndarray:
        """World point the job camera looks at (default: the thorax)."""
        return self.sim.thorax_position()

    def camera_preset(self) -> CameraPreset:
        return CameraPreset()

    def job_hud_lines(self) -> list[str]:
        return []

    def job_stats(self) -> dict[str, Any]:
        return {}

    # ------------------------------------------------------ edit effects (optional)
    def post_process(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Screen-space post-process of every rendered job frame (RGB uint8, H x W x 3),
        applied before the HUD is drawn: the window, M recordings, rolling
        recordings, timelapses and screenshots all get it (``FrameRenderer.draw``
        calls it when ``install_post_process`` wired it). ``t`` = run time of the
        rendered state. For *edit effects* (flash, bloom, shake, ...): label them
        as such. Default: identity (the runner does not even call it)."""
        return frame

    def time_scale(self, present_dt: float) -> float:
        """Presentation-time edit (hit-stop / slow motion; an *edit effect*): the
        runner calls this once per chunk with the chunk's nominal duration (s) and
        simulates only this fraction of it (1 = normal, 0 = freeze: the displayed
        frame is held; recordings are paced by ``Session.present_time``). The
        physics is untouched: the same step sequence, only fewer steps per
        displayed frame. Default: 1."""
        return 1.0

    def has_post_process(self) -> bool:
        return type(self).post_process is not EternalJob.post_process

    def has_time_scale(self) -> bool:
        return type(self).time_scale is not EternalJob.time_scale

    # ------------------------------------------------------ helpers for jobs
    def add_work(self, amount: float = 1.0) -> None:
        self.work += amount
        self.last_work_rt = self.run_time()

    def run_time(self) -> float:
        return self.session.run_time() if self.session is not None else 0.0

    def fly_xy(self) -> np.ndarray:
        return self.sim.data.xpos[self.sim.thorax_body_id, :2].copy()

    def fly_moved(self, window_s: float) -> float | None:
        """Thorax displacement (mm, xy) over the last ``window_s`` sim seconds
        (<= 6 s), None until that much history exists (cleared on reset)."""
        tr = self._track
        if not tr:
            return None
        t, x, y = tr[-1]
        for t0, x0, y0 in reversed(tr):
            if t - t0 >= window_s:
                return math.hypot(x - x0, y - y0)
        return None

    def unstick(self, window_s: float = 1.5, min_move: float = 0.5,
                duration: float = 0.6) -> bool:
        """If the fly hasn't moved ``min_move`` mm in ``window_s`` (wedged against a
        wall / prop), trigger the back-away action. Returns True if triggered."""
        acts = self.session.actions
        if acts.busy or self.fly_down():
            return False
        moved = self.fly_moved(window_s)
        if moved is None or moved >= min_move:
            return False
        acts.trigger(_make_action("back_away", duration=duration), source="job")
        self.n_unstuck += 1
        self._track.clear()
        return True

    def fly_down(self) -> bool:
        from fly_simulator.metrics import FallState

        return self.session.detector.state in (FallState.FALLEN, FallState.RECOVERING)

    def say(self, msg: str) -> None:
        if self.session is not None:
            self.session.say(f"[{self.name}] {msg}")

    # ------------------------------------------------------ eternal operation
    def recover(self, reason: str) -> None:
        """Explicit, counted, logged reset of the fly (and the props). Call only
        outside sim.step (after_physics / the runner)."""
        self.n_auto_recoveries += 1
        self.recovery_reasons[reason] = self.recovery_reasons.get(reason, 0) + 1
        self.say(f"auto-recovery #{self.n_auto_recoveries} ({reason}) at "
                 f"{format_uptime(self.run_time())}: explicit reset")
        self.session.log_event("job_recovery", job=self.name, reason=reason,
                               n=self.n_auto_recoveries)
        self.session.reset("auto")
        self.last_work_rt = self.run_time()

    def after_physics(self) -> None:
        """Once per physics chunk, outside sim.step: auto-recovery + housekeeping."""
        s = self.session
        down = s.down_for()
        if down is not None and self.fly_down() and down >= self.cfg.recover_after_s:
            self.recover("fell")
        elif (self.cfg.stuck_timeout_s > 0
              and self.run_time() - self.last_work_rt > self.cfg.stuck_timeout_s):
            self.recover("stuck")
        hist = s.actions.history
        if len(hist) > 2 * self.cfg.history_keep:
            del hist[:-self.cfg.history_keep]

    # ------------------------------------------------------ HUD / stats
    def hud_lines(self) -> list[str]:
        """The 'eternal stream' overlay."""
        rt = self.run_time()
        wall = time.perf_counter() - self.wall_start
        lines = [
            f"{self.title}" + (f"  -  {self.tagline}" if self.tagline else ""),
            f"{format_uptime(rt)} sim   (wall {format_uptime(wall)[4:]})",
            f"{self.work_label}: {self.work_format.format(self.work)}",
        ]
        lines += self.job_hud_lines()
        lines.append(f"falls {self.n_falls}   self-righted {self.n_self_righted}   "
                     f"auto-recoveries {self.n_auto_recoveries}   [{self.state}]")
        return lines

    def stats(self) -> dict[str, Any]:
        out = {
            "job": self.name, "work": self.work, "work_label": self.work_label,
            "run_time_s": self.run_time(), "falls": self.n_falls,
            "self_righted": self.n_self_righted, "auto_recoveries": self.n_auto_recoveries,
            "recovery_reasons": dict(self.recovery_reasons), "resets": self.n_resets,
            "state": self.state,
        }
        out.update(self.job_stats())
        return out

    # ------------------------------------------------------ internals
    def _post_step(self, sim: "Simulation") -> None:
        self._step_div += 1
        if self._step_div >= self.cfg.update_every_steps:
            self._step_div = 0
            t = sim.time
            if not self._track or t - self._track[-1][0] >= 0.1:
                p = sim.data.xpos[sim.thorax_body_id]
                self._track.append((t, float(p[0]), float(p[1])))
            self.update()

    def _on_sim_reset(self, sim: "Simulation") -> None:
        self.n_resets += 1
        self._track.clear()
        self._step_div = 0
        self.on_reset()
        self.reset_props()

    def _on_fall_event(self, ev) -> None:
        if ev.kind == "fall":
            self.n_falls += 1
        elif ev.kind == "recovered":
            self.n_self_righted += 1
