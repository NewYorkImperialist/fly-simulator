"""Screenshots (I key) and live MP4 recording (M key) of the fly view.

Only frames produced by the app's own renderers are saved (the MuJoCo offscreen
fly frame and, with --brain, a brain-window frame rendered in-process from the
latest BrainStates), never a capture of the screen.

Files go into the run directory (``runs/<ts>/``), or ``runs/screenshots/`` and
``runs/recordings/`` when logging is off. Everything here runs on the main thread
outside the physics lock (the physics thread is never blocked by PNG / MP4 I/O).

Recording timing: the video has ``fps`` frames per *simulated* second, i.e. it plays
at real time like ``--record``. ``due(run_time)`` says how many frames are owed
since the last write; the live window passes its latest displayed frame (dropping
frames when the display is faster than ``fps`` per sim second, duplicating when
slower), the headless loop renders a frame only when one is due.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import numpy as np


class MediaCapture:
    def __init__(self, run_dir: Path | str | None, runs_dir: Path | str = "runs",
                 fps: float = 30.0) -> None:
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.runs_dir = Path(runs_dir)
        self.fps = float(fps)
        self.n_shots = 0
        self.n_recordings = 0
        self.files: list[str] = []
        self._writer = None
        self.rec_path: Path | None = None
        self.rec_frames = 0
        self._rec_t0 = 0.0
        self._rec_wall0 = 0.0
        self.write_ms: list[float] = []  # append_data cost per written frame

    def out_dir(self, kind: str) -> Path:
        d = self.run_dir if self.run_dir is not None else self.runs_dir / kind
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------ screenshots
    def save_screenshot(self, frame_rgb: np.ndarray, run_time: float,
                        hud_bgr: np.ndarray | None = None,
                        brain_bgr: np.ndarray | None = None) -> list[Path]:
        """``frame_rgb``: clean renderer frame; ``hud_bgr``: same with the HUD drawn
        (optional); ``brain_bgr``: brain frame (optional). Returns written paths."""
        import cv2

        self.n_shots += 1
        d = self.out_dir("screenshots")
        stem = f"shot{self.n_shots:03d}_t{run_time:08.2f}s"
        out = [(d / f"{stem}_fly.png", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))]
        if hud_bgr is not None:
            out.append((d / f"{stem}_fly_hud.png", hud_bgr))
        if brain_bgr is not None:
            out.append((d / f"{stem}_brain.png", brain_bgr))
        paths = []
        for path, img in out:
            if not cv2.imwrite(str(path), img):
                raise OSError(f"could not write {path}")
            paths.append(path)
            self.files.append(str(path))
        return paths

    # ------------------------------------------------------------ recording
    @property
    def recording(self) -> bool:
        return self._writer is not None

    def start_recording(self, run_time: float) -> Path:
        import imageio.v2 as iio

        self.n_recordings += 1
        d = self.out_dir("recordings")
        self.rec_path = d / f"recording{self.n_recordings:02d}_t{run_time:08.2f}s.mp4"
        self._writer = iio.get_writer(self.rec_path, fps=self.fps, codec="libx264",
                                      quality=8, macro_block_size=8)
        self.rec_frames = 0
        self._rec_t0 = run_time
        self._rec_wall0 = time.perf_counter()
        self.write_ms = []
        self.files.append(str(self.rec_path))
        return self.rec_path

    def due(self, run_time: float) -> int:
        """Frames owed at ``run_time`` (0 if not recording)."""
        if self._writer is None:
            return 0
        return max(0, math.floor((run_time - self._rec_t0) * self.fps + 1e-6) + 1 - self.rec_frames)

    def add(self, frame_rgb: np.ndarray, run_time: float) -> int:
        """Write ``frame_rgb`` as many times as frames are due. Returns that count."""
        n = self.due(run_time)
        for _ in range(n):
            t0 = time.perf_counter()
            self._writer.append_data(frame_rgb)
            self.write_ms.append((time.perf_counter() - t0) * 1e3)
            self.rec_frames += 1
        return n

    def rec_label(self) -> str | None:
        if self._writer is None:
            return None
        return f"REC {self.rec_frames / self.fps:4.1f}s"

    def stop_recording(self) -> str:
        """Close the MP4; returns a one-line summary."""
        if self._writer is None:
            return "[record] not recording"
        self._writer.close()
        self._writer = None
        wall = time.perf_counter() - self._rec_wall0
        ms = float(np.median(self.write_ms)) if self.write_ms else 0.0
        return (f"[record] stopped: {self.rec_path} ({self.rec_frames} frames = "
                f"{self.rec_frames / self.fps:.1f} s of sim time at {self.fps:g} fps, "
                f"{wall:.1f} s wall; append {ms:.1f} ms/frame median)")

    def close(self, say: Callable[[str], None] | None = None) -> None:
        if self._writer is not None:
            msg = self.stop_recording()
            if say is not None:
                say(msg)
