"""Wiring jobs into a ``perpetualfly.app.Session`` + the eternal run loop.

* ``create_job_session(name, app_cfg, job_cfg)``: configure, build the Session with the
  job's props compiled in, install the job. Returns ``(session, job)``.
* ``install_job(session, job_or_name, job_cfg)``: attach a job to a Session whose
  model already contains its props (built with ``world_extensions=[job.extension]``
  after ``job.configure_app(cfg)``); chains ``job.after_physics`` onto
  ``session.after_physics`` and sets ``session.job``. For the app (``--job``)::

      job = make_job(name, job_cfg); job.configure_app(cfg)
      session = Session(cfg, world_extensions=[job.extension, ...])
      install_job(session, job)
      # HUD: session.job.hud_lines(); camera: JobCamera(cfg.camera, job)

* ``JobCamera``: ``SmoothFollowCamera`` with an extra "job" mode (the job's preset).
* ``JobRunner``: headless / window loop with auto-recovery from instabilities,
  rolling MP4 segments, timelapse, status lines.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import mujoco as mj
import numpy as np

from perpetualfly.jobs.base import EternalJob, JobConfig, format_uptime
from perpetualfly.jobs.registry import make_job
from perpetualfly.rendering import SmoothFollowCamera

if TYPE_CHECKING:
    from perpetualfly.app import Session
    from perpetualfly.config import AppConfig, CameraConfig


# ---------------------------------------------------------------------------
# session wiring
# ---------------------------------------------------------------------------


def _as_job(job_or_name, job_cfg=None) -> EternalJob:
    if isinstance(job_or_name, EternalJob):
        return job_or_name
    return make_job(str(job_or_name), job_cfg)


def job_props_present(model: mj.MjModel, job: EternalJob) -> bool:
    for n in job.required_names:
        if (mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, n) < 0
                and mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, n) < 0):
            return False
    return True


def install_job(session: "Session", job_or_name, job_cfg: JobConfig | dict | None = None
                ) -> EternalJob:
    """Attach a job to ``session`` (its props must be compiled into the model)."""
    job = _as_job(job_or_name, job_cfg)
    if not job_props_present(session.sim.model, job):
        raise RuntimeError(
            f"job {job.name!r}: props not found in the model. Build the Session with "
            f"world_extensions=[job.extension] after job.configure_app(cfg) (or use "
            f"create_job_session).")
    job.attach(session)
    orig = session.after_physics

    def after_physics() -> None:
        orig()
        job.after_physics()

    session.after_physics = after_physics
    session.job = job
    return job


def create_job_session(job_or_name, app_cfg: "AppConfig | None" = None,
                       job_cfg: JobConfig | dict | None = None, *, log: bool = False,
                       world_extensions=(), say: Callable[[str], None] | None = None,
                       brain=None) -> tuple["Session", EternalJob]:
    """Build a Session with the job's props and install the job."""
    from perpetualfly.app import Session
    from perpetualfly.config import AppConfig

    job = _as_job(job_or_name, job_cfg)
    cfg = app_cfg if app_cfg is not None else AppConfig()
    job.configure_app(cfg)
    session = Session(cfg, log=log, world_extensions=[job.extension, *world_extensions],
                      say=say, brain=brain)
    install_job(session, job)
    return session, job


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------


class JobCamera(SmoothFollowCamera):
    """"job" mode frames ``job.camera_target()`` with ``job.camera_preset()``; C cycles
    job / follow / side / top."""

    MODES = ("job", "follow", "side", "top")

    def __init__(self, cfg: "CameraConfig", job: EternalJob) -> None:
        super().__init__(cfg)
        self.job = job
        self.mode = "job"
        self._jl: np.ndarray | None = None
        self._jt: float | None = None

    def reset(self) -> None:
        super().reset()
        self._jl = self._jt = None

    def cycle_mode(self) -> str:
        self.mode = self.MODES[(self.MODES.index(self.mode) + 1) % len(self.MODES)]
        return self.mode

    def update(self, t, target, heading_rad, **kw) -> mj.MjvCamera:
        if self.mode != "job":
            return super().update(t, target, heading_rad, **kw)
        pre = self.job.camera_preset()
        tgt = np.asarray(self.job.camera_target(), dtype=float)
        if not np.all(np.isfinite(tgt)):
            tgt = np.asarray(target, dtype=float)
        if self._jl is None or self._jt is None or t < self._jt:
            self._jl = tgt.copy()
        else:
            a = 1.0 - math.exp(-(t - self._jt) / max(pre.tau_s, 1e-6))
            self._jl += a * (tgt - self._jl)
        self._jt = t
        self.cam.lookat[:] = self._jl
        self.cam.distance = pre.distance
        self.cam.elevation = pre.elevation
        self.cam.azimuth = (pre.azimuth if pre.azimuth is not None
                            else math.degrees(heading_rad) + pre.azimuth_offset)
        return self.cam


# ---------------------------------------------------------------------------
# run loop
# ---------------------------------------------------------------------------


class JobRunner:
    """Drives one job session. ``run(max_seconds)`` returns the job stats.

    ``headless``: no window. ``record_dir``: rolling MP4 segments there
    (``segment_s`` sim seconds each, newest ``keep_segments`` kept).
    ``timelapse_every_s``: also a timelapse (one frame per that many sim seconds).
    ``stop``: optional callable checked once per chunk (True = quit).
    """

    def __init__(self, session: "Session", job: EternalJob, *, headless: bool = True,
                 record_dir: Path | str | None = None, segment_s: float = 60.0,
                 keep_segments: int = 3, record_fps: float = 15.0,
                 timelapse_every_s: float | None = None, timelapse_dir: Path | str | None = None,
                 print_every_s: float = 10.0, chunk_steps: int | None = None,
                 say: Callable[[str], None] | None = None) -> None:
        self.session, self.job = session, job
        self.sim = session.sim
        self.cfg = session.cfg
        self.headless = headless
        self.say = say or (lambda m: print(m, flush=True))
        self.chunk = int(chunk_steps or self.cfg.render.render_every_steps)
        self.print_every_s = print_every_s
        self.recorder = None
        self.timelapse = None
        if record_dir is not None:
            from perpetualfly.jobs.recording import RollingRecorder

            self.recorder = RollingRecorder(record_dir, segment_s, keep_segments, record_fps)
        if timelapse_every_s:
            from perpetualfly.jobs.recording import Timelapse

            self.timelapse = Timelapse(timelapse_dir or record_dir or "runs/timelapse",
                                       timelapse_every_s, fps=record_fps)
        self.renderer = None
        self.viewer = None
        self.n_instabilities = 0
        self.quit_reason = "max-seconds"
        self.paused = False
        self.n_shots = 0
        self.shot_dir: Path | None = Path(record_dir) if record_dir is not None else None

    # ------------------------------------------------------------ frames
    def ensure_renderer(self):
        if self.renderer is None:
            from perpetualfly.rendering import FrameRenderer

            self.renderer = FrameRenderer(self.sim.model, self.cfg.render, self.cfg.camera)
            self.renderer.camera = JobCamera(self.cfg.camera, self.job)
        return self.renderer

    def render(self, hud: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        """(clean RGB frame, RGB frame with the HUD or None)."""
        r = self.ensure_renderer()
        sim = self.sim
        x, y, _ = sim.thorax_position()
        frame = r.render(sim.data, sim.time, sim.thorax_position(), sim.heading(),
                         ground_z=self.job.ground_height(x, y), tilt_deg=sim.tilt_deg())
        if not hud:
            return frame, None
        import cv2

        from perpetualfly.interaction.viewer import compose_frame

        bgr = compose_frame(frame, self.job.hud_lines())
        return frame, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def screenshot(self, path: Path | str) -> Path:
        import cv2

        _, hud = self.render()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), cv2.cvtColor(hud, cv2.COLOR_RGB2BGR))
        return path

    # ------------------------------------------------------------ loop
    def step_chunk(self) -> None:
        from perpetualfly.simulation import SimulationInstabilityError

        try:
            self.sim.step(self.chunk)
        except SimulationInstabilityError as e:
            self.n_instabilities += 1
            self.say(f"[{self.job.name}] physics instability #{self.n_instabilities}: "
                     f"{str(e).splitlines()[0]}")
            self.job.recover("instability")
            if self.renderer is not None:
                self.renderer.camera.reset()
        self.session.after_physics()

    def status_line(self, rtf: float) -> str:
        j = self.job
        return (f"[{j.name}] {format_uptime(j.run_time())}  {j.work_label}="
                f"{j.work_format.format(j.work)}  falls={j.n_falls} "
                f"self-righted={j.n_self_righted} auto-rec={j.n_auto_recoveries} "
                f"state={j.state} RTF={rtf:.2f}  "
                + " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in j.job_stats().items()))

    def _handle_key(self, k: str) -> None:
        if k in ("q", "esc"):
            self.quit_reason = "quit key"
        elif k == "c" and self.renderer is not None:
            self.say(f"[camera] {self.renderer.camera.cycle_mode()}")
        elif k == "p":
            self.paused = not self.paused
        elif k == "x":
            self.session.reset("manual")
        elif k == "i":
            self.n_shots += 1
            d = self.shot_dir or Path(self.cfg.logging.runs_dir) / "screenshots"
            p = self.screenshot(d / f"{self.job.name}_shot{self.n_shots:03d}.png")
            self.say(f"[screenshot] {p}")

    def run(self, max_seconds: float | None = None,
            stop: Callable[[], bool] | None = None) -> dict:
        s = self.session
        rt0 = s.run_time()
        wall0 = time.perf_counter()
        next_print = rt0 + self.print_every_s
        last_wall, last_rt = wall0, rt0
        if not self.headless:
            from perpetualfly.interaction import LiveViewer

            self.viewer = LiveViewer(f"PerpetualFly - {self.job.title}",
                                     display_scale=self.cfg.render.display_scale,
                                     frame_size=(self.cfg.render.width, self.cfg.render.height))
        period = 1.0 / max(self.cfg.render.target_fps, 1.0)
        last_shown = -1e9
        try:
            while True:
                if not self.paused:
                    self.step_chunk()
                rt = s.run_time()
                need_rec = self.recorder is not None and self.recorder.due(rt)
                need_tl = self.timelapse is not None and self.timelapse.due(rt)
                show = self.viewer is not None and time.perf_counter() - last_shown >= period
                if need_rec or need_tl or show:
                    _, hud = self.render()
                    if need_rec:
                        self.recorder.add(hud, rt)
                    if need_tl:
                        self.timelapse.add(hud, rt)
                    if show:
                        last_shown = time.perf_counter()
                        self.viewer.show(hud)
                        for k in self.viewer.poll_keys(30 if self.paused else 1):
                            self._handle_key(k)
                        if not self.viewer.is_open():
                            self.quit_reason = "window closed"
                if rt >= next_print:
                    w = time.perf_counter()
                    rtf = (rt - last_rt) / max(w - last_wall, 1e-9)
                    last_wall, last_rt = w, rt
                    while next_print <= rt:
                        next_print += self.print_every_s
                    self.say(self.status_line(rtf))
                if self.quit_reason != "max-seconds":
                    break
                if max_seconds is not None and rt - rt0 >= max_seconds:
                    break
                if stop is not None and stop():
                    self.quit_reason = "stop"
                    break
        except KeyboardInterrupt:
            self.quit_reason = "Ctrl-C"
        finally:
            self.close()
        wall = time.perf_counter() - wall0
        out = self.job.stats()
        out.update(quit_reason=self.quit_reason, wall_s=wall, instabilities=self.n_instabilities,
                   rtf=(s.run_time() - rt0) / max(wall, 1e-9))
        return out

    def close(self) -> None:
        for obj in (self.recorder, self.timelapse):
            if obj is not None:
                try:
                    obj.close()
                except Exception as e:  # never mask the real error
                    self.say(f"warning: closing the recording failed: {e}")
        self.recorder = self.timelapse = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
