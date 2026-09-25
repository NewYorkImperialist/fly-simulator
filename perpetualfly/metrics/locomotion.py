from __future__ import annotations

from collections import deque

import numpy as np


class LocomotionStats:
    """Tracks sim time, distance and speed from thorax positions.

    Call ``update(t, pos)`` at any rate (e.g. every render or every N physics steps).
    * ``path_length``: accumulated xy distance (mm), robust to turning; completed
      segments are sampled every ``path_sample_s`` plus the open segment to "now".
    * ``forward_displacement``: net displacement along the initial heading (+x).
    * ``current_speed``: xy speed over the last ``speed_window_s`` sim seconds.
    * ``average_speed``: path_length / elapsed sim time.
    """

    def __init__(self, speed_window_s: float = 0.5, path_sample_s: float = 0.25) -> None:
        self.speed_window_s = speed_window_s
        # Path length is accumulated between samples >= path_sample_s apart (longer
        # than one ~83 ms stride) so the tripod gait's lateral sway is not counted.
        self.path_sample_s = path_sample_s
        self.reset()

    def reset(self) -> None:
        self.start_time: float | None = None
        self.start_pos: np.ndarray | None = None
        self.last_pos: np.ndarray | None = None
        self._path_anchor: tuple[float, np.ndarray] | None = None
        self.time = 0.0
        self._path_accum = 0.0
        self._window: deque[tuple[float, np.ndarray]] = deque()

    def update(self, t: float, pos: np.ndarray) -> None:
        pos = np.asarray(pos, dtype=float).copy()
        if self.start_time is None:
            self.start_time, self.start_pos, self.last_pos = t, pos, pos
            self._path_anchor = (t, pos)
        if t - self._path_anchor[0] >= self.path_sample_s:
            self._path_accum += float(np.linalg.norm(pos[:2] - self._path_anchor[1][:2]))
            self._path_anchor = (t, pos)
        self.last_pos = pos
        self.time = t
        self._window.append((t, pos))
        while len(self._window) > 2 and t - self._window[1][0] >= self.speed_window_s:
            self._window.popleft()

    @property
    def path_length(self) -> float:
        if self._path_anchor is None:
            return 0.0
        partial = float(np.linalg.norm(self.last_pos[:2] - self._path_anchor[1][:2]))
        return self._path_accum + partial

    @property
    def elapsed(self) -> float:
        return 0.0 if self.start_time is None else self.time - self.start_time

    @property
    def forward_displacement(self) -> float:
        if self.start_pos is None:
            return 0.0
        return float(self.last_pos[0] - self.start_pos[0])

    @property
    def current_speed(self) -> float:
        if len(self._window) < 2:
            return 0.0
        (t0, p0), (t1, p1) = self._window[0], self._window[-1]
        return float(np.linalg.norm(p1[:2] - p0[:2]) / (t1 - t0)) if t1 > t0 else 0.0

    @property
    def average_speed(self) -> float:
        return self.path_length / self.elapsed if self.elapsed > 0 else 0.0

    def summary_line(self) -> str:
        return (
            f"t={self.time:8.2f}s  dist={self.path_length:8.1f}mm "
            f"(fwd {self.forward_displacement:+8.1f})  "
            f"speed now={self.current_speed:5.1f} avg={self.average_speed:5.1f} mm/s"
        )
