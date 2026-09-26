"""Run the physics loop in a background thread while the main thread displays.

Why: on macOS every displayed frame costs ~14 ms of offscreen rendering plus
~12-15 ms in ``cv2.waitKeyEx`` (Cocoa event pump; ``cv2.pollKey`` is no faster).
Single-threaded, that time is taken away from physics. ``mj_step``,
``mujoco.Renderer.render`` and ``cv2.waitKeyEx`` all release the GIL, so with the
physics in a worker thread the display runs at ~30 fps while physics keeps
(almost) its headless speed.

Rules:
* Everything that touches ``sim`` (``model``/``data``, hooks, reset, controller)
  outside the worker must be done inside ``with runner.locked():``. The worker
  holds the lock while it runs one chunk of ``sim.step``; hooks therefore still run
  every physics step, in the worker thread.
* ``FrameRenderer.update_scene`` (reads ``data``) goes under the lock;
  ``FrameRenderer.draw`` does not need it.
* Results are identical to the single-threaded loop: only *when* the main thread
  gets to look at the state differs, never the sequence of physics steps.

Two details make this work in practice (measured, see docs/API_NOTES.md):
* ``threading.Lock`` is not fair: the worker re-acquired it immediately and the
  display starved (0.7 fps). A second "gate" lock acts as a turnstile.
* With CPython's default 5 ms GIL switch interval the main thread waits for the
  GIL (the controller is Python code), which cut physics to ~0.5x. ``run()``
  lowers the switch interval to 1 ms while the worker is alive.
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator

from perpetualfly.simulation import Simulation


class PhysicsThread:
    def __init__(
        self,
        sim: Simulation,
        chunk_steps: int,
        after_chunk: Callable[[], bool] | None = None,
        max_realtime_factor: float | None = None,
        switch_interval_s: float = 0.001,
        step_fn: Callable[[int], None] | None = None,
    ) -> None:
        """``after_chunk()`` runs in the worker, under the lock, after every chunk
        (stats, printing). Returning True stops the worker (e.g. max sim time).
        ``step_fn(n)`` replaces ``sim.step(n)`` (e.g. the app's job-aware step that
        recovers from instabilities)."""
        self.sim = sim
        self.step_fn = step_fn or sim.step
        self.chunk_steps = max(1, int(chunk_steps))
        self.after_chunk = after_chunk
        self.max_realtime_factor = max_realtime_factor
        self.switch_interval_s = switch_interval_s
        self.paused = False
        self.error: BaseException | None = None
        self._lock = threading.Lock()
        self._gate = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._old_switch: float | None = None
        self._clock_wall = 0.0
        self._clock_sim = 0.0

    # ----------------------------------------------------------------- control
    def start(self) -> None:
        self._old_switch = sys.getswitchinterval()
        sys.setswitchinterval(self.switch_interval_s)
        self.reset_clock()
        self._thread = threading.Thread(target=self._run, name="physics", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._old_switch is not None:
            sys.setswitchinterval(self._old_switch)
            self._old_switch = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def raise_if_failed(self) -> None:
        """Re-raise an exception from the worker (e.g. SimulationInstabilityError)."""
        if self.error is not None:
            err, self.error = self.error, None
            raise err

    def reset_clock(self) -> None:
        """Restart the real-time cap reference (call after reset / unpause)."""
        self._clock_wall, self._clock_sim = time.perf_counter(), self.sim.time

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Exclusive access to the simulation from the calling thread."""
        with self._gate, self._lock:
            yield

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if self.paused:
                    time.sleep(0.01)
                    self.reset_clock()
                    continue
                with self._gate:  # turnstile: lets a waiting locked() go first
                    pass
                with self._lock:
                    self.step_fn(self.chunk_steps)
                    done = self.after_chunk() if self.after_chunk is not None else False
                if done:
                    return
                self._throttle()
        except BaseException as e:  # surfaced in the main thread by raise_if_failed()
            self.error = e

    def _throttle(self) -> None:
        cap = self.max_realtime_factor
        if not cap:
            return
        ahead = (self.sim.time - self._clock_sim) / cap - (time.perf_counter() - self._clock_wall)
        if ahead > 0:
            time.sleep(ahead)
