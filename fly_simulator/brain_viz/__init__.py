"""Separate brain-activity window driven by fly_simulator.brain.schema messages.

See docs/BRAIN_WINDOW.md. Public API (lazy imports keep ``import fly_simulator`` light):

* ``render_frame(layout, state) -> np.ndarray`` (BGR) -- headless single frame
* ``BrainRenderer`` / ``BrainWindow`` -- drawing / OpenCV window
* ``run_window(queue, layout, ...)`` -- multiprocessing target
* ``BrainWindowProcess`` -- parent-side handle (spawn, bounded queue, non-blocking)
* ``MockBrain`` -- fake BrainLayout/BrainState stream
"""

_EXPORTS = {
    "render_frame": "window", "render_timeline": "window", "BrainRenderer": "window",
    "BrainWindow": "window", "run_window": "window", "BrainWindowProcess": "window",
    "WINDOW_TITLE": "window", "MockBrain": "mock", "mock_scenario": "mock",
    "load_atlas": "atlas", "atlas_region_xy": "atlas",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib

        return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(name)
