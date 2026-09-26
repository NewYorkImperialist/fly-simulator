"""Long-run recording for eternal jobs, built on ``perpetualfly.media.MediaCapture``.

* ``RollingRecorder``: MP4 segments of ``segment_s`` *sim* seconds each (real-time
  playback at ``fps``), only the newest ``keep`` files are kept (older ones are
  deleted), so disk use is bounded however long the job runs.
* ``Timelapse``: one frame every ``every_s`` sim seconds, played at ``fps``; a new file
  every ``frames_per_file`` frames, newest ``keep`` files kept.

Both take frames the runner already rendered (with the HUD drawn in), never screen
captures. Call ``due(run_time)`` first and only render when it is True.
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import numpy as np

from perpetualfly.media import MediaCapture


class RollingRecorder:
    def __init__(self, out_dir: Path | str, segment_s: float = 60.0, keep: int = 3,
                 fps: float = 15.0) -> None:
        self.out_dir = Path(out_dir)
        self.segment_s = float(segment_s)
        self.keep = max(1, int(keep))
        self.media = MediaCapture(self.out_dir, fps=fps)
        self.files: deque[Path] = deque()
        self.n_segments = 0
        self.n_deleted = 0
        self._seg_t0 = 0.0

    def due(self, run_time: float) -> bool:
        if not self.media.recording:
            return True
        return self.media.due(run_time) > 0

    def add(self, frame_rgb: np.ndarray, run_time: float) -> None:
        m = self.media
        if m.recording and run_time - self._seg_t0 >= self.segment_s:
            self._close_segment()
        if not m.recording:
            path = m.start_recording(run_time)
            self._seg_t0 = run_time
            self.files.append(path)
            self.n_segments += 1
            m.files.clear()  # MediaCapture's own list would grow forever
        m.add(frame_rgb, run_time)

    def _close_segment(self) -> None:
        self.media.stop_recording()
        while len(self.files) > self.keep:
            old = self.files.popleft()
            try:
                os.remove(old)
                self.n_deleted += 1
            except OSError:
                pass

    def close(self) -> None:
        if self.media.recording:
            self._close_segment()

    def label(self) -> str:
        return f"REC seg {self.n_segments} ({self.media.rec_frames / self.media.fps:4.1f}s)"


class Timelapse:
    def __init__(self, out_dir: Path | str, every_s: float = 5.0, fps: float = 15.0,
                 frames_per_file: int = 900, keep: int = 2) -> None:
        self.out_dir = Path(out_dir)
        self.every_s = float(every_s)
        self.frames_per_file = max(1, int(frames_per_file))
        self.keep = max(1, int(keep))
        self.media = MediaCapture(self.out_dir, fps=fps)
        self.files: deque[Path] = deque()
        self.n_frames = 0
        self._t0: float | None = None

    def _vt(self, run_time: float) -> float:
        # virtual clock: MediaCapture writes fps frames per virtual second, i.e.
        # one frame per every_s of run time
        return run_time / (self.every_s * self.media.fps)

    def due(self, run_time: float) -> bool:
        if not self.media.recording:
            return True
        return self.media.due(self._vt(run_time)) > 0

    def add(self, frame_rgb: np.ndarray, run_time: float) -> None:
        m = self.media
        if m.recording and m.rec_frames >= self.frames_per_file:
            self._close_file()
        if not m.recording:
            m.start_recording(self._vt(run_time))
            self._t0 = run_time
            m.files.clear()
        # at most one frame per call (a long gap is not padded with duplicates)
        if m.due(self._vt(run_time)) > 0:
            m._writer.append_data(frame_rgb)
            m.rec_frames += 1
            # re-anchor so the next frame is due every_s after this one
            m._rec_t0 = self._vt(run_time) - (m.rec_frames - 1) / m.fps
            self.n_frames += 1

    def _close_file(self) -> None:
        m = self.media
        m.stop_recording()
        src = m.rec_path
        dst = self.out_dir / f"timelapse{m.n_recordings:03d}_from_t{self._t0 or 0:09.1f}s.mp4"
        try:
            os.replace(src, dst)
        except OSError:
            dst = src
        self.files.append(dst)
        while len(self.files) > self.keep:
            old = self.files.popleft()
            try:
                os.remove(old)
            except OSError:
                pass

    def close(self) -> None:
        if self.media.recording:
            self._close_file()
