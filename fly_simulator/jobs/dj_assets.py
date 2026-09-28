"""Procedural meshes and textures for the DJ job (visual only).

Same approach as ``kebab_assets``: everything is generated in code (numpy + OpenCV)
and handed to MjSpec directly; nothing is written to disk. Nothing here collides or
has mass. A club booth: a stage carpet, vinyl records with a label, an LED wall with
the club's own name ("FLY FM", ours; no real brand, label or logo), a neon booth
front, a mirror-tile disco ball.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.broccoli_toss_assets import merge, panel_mesh, panel_quat, sphere_mesh, torus_mesh  # noqa: F401
from fly_simulator.jobs.kebab_assets import (  # noqa: F401  (re-exported helpers)
    TAU, MeshData, _mix, _to_u8, add_mesh, add_texture, add_textured_material, fbm, lathe,
    tile_noise, transform, tube,
)
from fly_simulator.jobs.sisyphus_assets import box_mesh  # noqa: F401
from fly_simulator.jobs.trampoline_assets import _text, disc_mesh, polyline_tube  # noqa: F401
from fly_simulator.jobs.jump_rope_assets import slab_mesh  # noqa: F401


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def cone_mesh(r: float, depth: float, n: int = 32) -> MeshData:
    """A speaker cone about z, opening toward +z (dust cap at the bottom)."""
    rr = np.array([0.0, 0.22 * r, 0.25 * r, r, 1.08 * r, 1.08 * r])
    zz = np.array([0.02 * depth, 0.05 * depth, 0.0, depth, depth, 0.9 * depth])
    return lathe(rr, zz, n, v_coord=np.linspace(0, 1, len(rr)))


def tonearm_mesh(pivot, tip, h: float) -> MeshData:
    """A tonearm polyline from the pivot post to the head over the record."""
    p0 = np.array([pivot[0], pivot[1], h])
    p2 = np.array([tip[0], tip[1], h * 0.7])
    mid = 0.5 * (p0 + p2) + np.array([0.15, 0.0, 0.05])
    return polyline_tube(np.array([p0, mid, p2]), 0.035, 8)


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def stage_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """Black stage carpet with a faint speckle and gaffer-tape marks."""
    rng = np.random.default_rng(seed + 3)
    f = fbm(n, n, 16, 16, rng, octaves=4)
    img = _to_u8(_mix(np.array([0.04, 0.04, 0.05], np.float32), np.array([0.10, 0.09, 0.12], np.float32), f))
    for _ in range(12):
        x, y = rng.integers(0, n, 2)
        a = rng.uniform(0, math.pi)
        L = int(rng.integers(20, 50))
        cv2.line(img, (int(x), int(y)), (int(x + L * math.cos(a)), int(y + L * math.sin(a))),
                 (60, 60, 64), 5)
    return img


def record_texture(label_rgb=(0.95, 0.3, 0.2), n: int = 512, name: str = "FLY FM") -> np.ndarray:
    """Vinyl seen from above (planar uv over the disc): grooves, sheen, a coloured label."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    c = (n - 1) / 2
    r = np.hypot(x - c, y - c) / c
    ang = np.arctan2(y - c, x - c)
    groove = 0.5 + 0.5 * np.sin(r * 260.0)
    sheen = 0.5 + 0.5 * np.cos(2 * ang)
    v = 0.05 + 0.04 * groove + 0.07 * sheen * (r > 0.36)
    img = _to_u8(np.stack([v, v, v * 1.05], -1))
    lab = tuple(int(255 * q) for q in label_rgb)
    cv2.circle(img, (int(c), int(c)), int(0.34 * c), lab, -1, cv2.LINE_AA)
    cv2.circle(img, (int(c), int(c)), int(0.34 * c), (240, 240, 235), 3, cv2.LINE_AA)
    _text(img, name, (int(c), int(c - 0.14 * c)), 1.1, (0.98, 0.98, 0.95), 2, center=True)
    _text(img, "33 1/3", (int(c), int(c + 0.18 * c)), 0.7, (0.1, 0.1, 0.1), 2, center=True)
    cv2.circle(img, (int(c), int(c)), max(3, n // 90), (20, 20, 20), -1, cv2.LINE_AA)
    return img


def led_wall_texture(h: int = 256, w: int = 768) -> np.ndarray:
    """The LED wall behind the booth: a dot matrix with the club's name."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (6, 4, 14)
    _text(img, "FLY FM", (w // 2, int(h * 0.30)), 3.0, (0.95, 0.25, 0.85), 8, center=True)
    _text(img, "NIGHT SHIFT FOREVER", (w // 2, int(h * 0.55)), 1.0, (0.3, 0.85, 1.0), 2, center=True)
    # the dot matrix
    step = 6
    mask = np.zeros((h, w), bool)
    mask[::step, ::step] = True
    mask[1::step, ::step] = True
    mask[::step, 1::step] = True
    mask[1::step, 1::step] = True
    dim = (img.astype(np.float32) * 0.18).astype(np.uint8)
    img = np.where(mask[..., None], img, dim)
    return img


def booth_front_texture(h: int = 128, w: int = 768) -> np.ndarray:
    """The neon booth front facing the dance floor."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (14, 10, 22)
    for k in range(0, w, 24):
        cv2.line(img, (k, 0), (k + 60, h), (26, 20, 38), 10)
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), (255, 60, 200), 4, cv2.LINE_AA)
    _text(img, "FLY FM  *  LIVE ON THE DECKS  *  ALL NIGHT, EVERY NIGHT", (w // 2, h // 2), 1.0,
          (0.4, 0.95, 1.0), 2, center=True)
    return img


@lru_cache(maxsize=1)
def mirror_ball_texture(n: int = 256, seed: int = 0) -> np.ndarray:
    """Mirror tiles (lathe uv: u around, v pole to pole)."""
    rng = np.random.default_rng(seed + 9)
    img = np.zeros((n, n, 3), np.uint8)
    k = 24
    s = n // k
    for i in range(k):
        for j in range(k):
            t = rng.uniform(0.45, 1.0)
            tint = rng.uniform(0.9, 1.05, 3)
            col = np.clip(np.array([210, 215, 230]) * t * tint, 0, 255)
            img[i * s:(i + 1) * s - 1, j * s:(j + 1) * s - 1] = col
    return img


def grille_texture(n: int = 128) -> np.ndarray:
    """A speaker grille: dark perforated metal."""
    img = np.full((n, n, 3), 22, np.uint8)
    for y in range(3, n, 6):
        for x in range(3 + (y // 6 % 2) * 3, n, 6):
            cv2.circle(img, (x, y), 1, (6, 6, 8), -1)
    return img
