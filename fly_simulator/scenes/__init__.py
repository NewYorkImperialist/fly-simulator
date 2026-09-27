"""Scripted cinematic scenes (not eternal jobs): a fixed timeline of shots rendered
offscreen to an MP4 with ``scripts/render_scene.py``. See docs/SCENES.md."""

from __future__ import annotations

import importlib

#: scene id -> module (each module exposes ``NAME``, ``SHOTS``, ``RenderOptions``, ``render``)
SCENES = {
    "temple_standoff": "fly_simulator.scenes.temple_standoff",
}


def load_scene(name: str):
    if name not in SCENES:
        raise KeyError(f"unknown scene {name!r} (known: {', '.join(sorted(SCENES))})")
    return importlib.import_module(SCENES[name])
