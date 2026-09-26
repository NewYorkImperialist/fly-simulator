"""Shared test setup: skip tests that need an OpenGL context when there is none.

Rendering (the fly camera, FlyGym's compound eyes, screenshots, videos) needs an
offscreen MuJoCo GL context: CGL on macOS, EGL or OSMesa on headless Linux
(``MUJOCO_GL=egl``). Hosted CI machines without one (e.g. GitHub's macOS runners)
set ``FLY_SIMULATOR_SKIP_RENDERING=1``; then every test that creates a
``mujoco.Renderer`` (directly or through FlyGym) is skipped at that point instead
of failing. Tests can also be marked ``@pytest.mark.rendering`` explicitly.
"""

from __future__ import annotations

import os

import pytest

SKIP_RENDERING = os.environ.get("FLY_SIMULATOR_SKIP_RENDERING", "") not in ("", "0")
REASON = "rendering disabled (FLY_SIMULATOR_SKIP_RENDERING=1, no OpenGL context)"


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "rendering: needs an offscreen OpenGL context (skipped when "
        "FLY_SIMULATOR_SKIP_RENDERING=1)")


def pytest_collection_modifyitems(config, items):
    if not SKIP_RENDERING:
        return
    for item in items:
        if "rendering" in item.keywords:
            item.add_marker(pytest.mark.skip(reason=REASON))


@pytest.fixture(autouse=True)
def _no_renderer_without_gl(monkeypatch):
    """With rendering disabled, creating a ``mujoco.Renderer`` skips the test."""
    if SKIP_RENDERING:
        import mujoco

        def _skip(self, *a, **k):
            self._gl_context = self._mjr_context = None  # so close() / __del__ work
            pytest.skip(REASON)

        monkeypatch.setattr(mujoco.Renderer, "__init__", _skip)
    yield
