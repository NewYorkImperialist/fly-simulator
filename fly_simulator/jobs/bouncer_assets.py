"""Procedural meshes / textures for the bouncer job (visual only).

Generated in code (numpy + OpenCV) and handed to MjSpec directly; nothing on disk.
The club name ("CLUB HALTERE") and every poster are our own; no real brand or logo.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.broccoli_toss_assets import merge, panel_mesh, panel_quat, sphere_mesh  # noqa: F401
from fly_simulator.jobs.kebab_assets import (  # noqa: F401  (re-exported helpers)
    TAU, MeshData, _mix, _to_u8, add_mesh, add_texture, add_textured_material, fbm, lathe, transform, tube,
)
from fly_simulator.jobs.trampoline_assets import disc_mesh, polyline_tube  # noqa: F401

CLUB_NAME = "CLUB HALTERE"


def catenary(a, b, sag: float, n: int = 24) -> np.ndarray:
    """Points of a rope hanging from ``a`` to ``b`` (3-vectors), ``sag`` mm at the middle."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = np.linspace(0.0, 1.0, n)[:, None]
    p = a + (b - a) * u
    p[:, 2] -= sag * 4.0 * u[:, 0] * (1.0 - u[:, 0])
    return p


def stanchion_mesh(h: float = 1.9, r: float = 0.07) -> MeshData:
    """A brass stanchion: a weighted base, the post, a ball finial (z = 0 the ground)."""
    prof_r = np.array([0.0, 0.42, 0.44, 0.40, 0.16, r, r, 0.11, 0.12, 0.11, 0.0])
    prof_z = np.array([0.0, 0.0, 0.05, 0.10, 0.16, 0.24, h - 0.08, h - 0.03, h + 0.05, h + 0.14, h + 0.2])
    return lathe(prof_r, prof_z, 24)


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def brick_texture(seed: int = 0, h: int = 512, w: int = 1024) -> np.ndarray:
    """A dark red brick wall at night, with grime and a few old flyers' paste marks."""
    rng = np.random.default_rng(seed + 811)
    img = np.zeros((h, w, 3), np.float32)
    mortar = np.array([0.20, 0.19, 0.18], np.float32)
    img[:] = mortar
    bh, bw = 24, 64
    for r in range(h // bh + 1):
        off = (bw // 2) * (r % 2)
        for c in range(-1, w // bw + 2):
            x0, y0 = c * bw + off + 2, r * bh + 2
            base = np.array([0.42, 0.14, 0.10]) * rng.uniform(0.7, 1.15)
            base[1] *= rng.uniform(0.8, 1.3)
            cv2.rectangle(img, (x0, y0), (x0 + bw - 4, y0 + bh - 4), tuple(float(v) for v in base), -1)
    grime = fbm(h, w, 4, 8, rng, octaves=4)
    img *= (0.55 + 0.6 * grime)[..., None]
    streak = np.clip(np.linspace(0.3, -0.2, h, dtype=np.float32), 0, 1)[:, None, None]
    img *= 1.0 - 0.35 * streak
    return _to_u8(img)


@lru_cache(maxsize=2)
def neon_texture(text: str = CLUB_NAME, h: int = 192, w: int = 896) -> np.ndarray:
    """Neon tubes on a black backing board: pink letters with a cyan outline + a
    small fly silhouette (our own mark)."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (10, 6, 14)
    font = cv2.FONT_HERSHEY_TRIPLEX
    scale, th = 3.0, 6
    (tw, tht), _ = cv2.getTextSize(text, font, scale, th)
    x, y = (w - tw) // 2 + 40, (h + tht) // 2
    glow = np.zeros_like(img)
    cv2.putText(glow, text, (x, y), font, scale, (255, 60, 200), th + 14, cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 9)
    img = cv2.add(img, (glow * 0.8).astype(np.uint8))
    cv2.putText(img, text, (x, y), font, scale, (255, 120, 230), th, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (255, 235, 250), 2, cv2.LINE_AA)
    # the fly mark (cyan): body, head, two wings
    cx, cy = x - 70, h // 2
    col = (80, 240, 255)
    cv2.ellipse(img, (cx, cy + 8), (12, 30), 0, 0, 360, col, 4, cv2.LINE_AA)
    cv2.circle(img, (cx, cy - 30), 11, col, 4, cv2.LINE_AA)
    cv2.ellipse(img, (cx - 26, cy - 4), (26, 11), -30, 0, 360, col, 3, cv2.LINE_AA)
    cv2.ellipse(img, (cx + 26, cy - 4), (26, 11), 30, 0, 360, col, 3, cv2.LINE_AA)
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), (60, 220, 255), 3, cv2.LINE_AA)
    return img


@lru_cache(maxsize=4)
def poster_texture(kind: int = 0, h: int = 320, w: int = 224) -> np.ndarray:
    """Gig posters (our own copy)."""
    img = np.zeros((h, w, 3), np.uint8)
    bg = [(40, 20, 90), (20, 70, 40), (90, 30, 20), (30, 40, 90)][kind % 4]
    img[:] = bg
    font = cv2.FONT_HERSHEY_DUPLEX
    lines = [["TONIGHT", "DJ BUZZ", "KILL", "10 PM"], ["WING", "NIGHT", "LADIES", "FREE"],
             ["NO", "SWATTERS", "ALLOWED", "!"], ["SAT", "HALTERE", "HOUSE", "ALL NIGHT"]][kind % 4]
    for i, t in enumerate(lines):
        sc = 1.2 if i < 2 else 0.9
        (tw, _), _ = cv2.getTextSize(t, font, sc, 2)
        cv2.putText(img, t, ((w - tw) // 2, 70 + 70 * i), font, sc, (240, 230, 120) if i % 2 == 0 else
                    (250, 250, 250), 2, cv2.LINE_AA)
    cv2.rectangle(img, (4, 4), (w - 5, h - 5), (230, 230, 230), 2)
    return img


@lru_cache(maxsize=2)
def pavement_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """Wet night sidewalk: concrete slabs with dark joints and damp patches."""
    rng = np.random.default_rng(seed + 97)
    base = 0.16 + 0.07 * fbm(n, n, 8, 8, rng, octaves=4)
    img = np.stack([base, base * 0.98, base * 1.05], -1).astype(np.float32)
    wet = fbm(n, n, 3, 3, rng, octaves=3)
    img *= (0.75 + 0.4 * np.clip(wet - 0.4, 0, 1))[..., None]
    img = _to_u8(img)
    step = n // 4
    for k in range(0, n, step):
        cv2.line(img, (k, 0), (k, n), (18, 18, 20), 3)
        cv2.line(img, (0, k), (n, k), (18, 18, 20), 3)
    return img


@lru_cache(maxsize=2)
def carpet_texture(n: int = 256) -> np.ndarray:
    """The red carpet runner at the door."""
    rng = np.random.default_rng(5)
    base = 0.45 + 0.08 * rng.random((n, n)).astype(np.float32)
    img = np.stack([base, base * 0.08, base * 0.12], -1)
    img[:, :10] *= 0.4
    img[:, -10:] *= 0.4
    return _to_u8(img)


def door_texture(h: int = 256, w: int = 160) -> np.ndarray:
    """The club door: black padded leather with brass studs and a round window."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (22, 18, 20)
    for y in range(16, h, 28):
        for x in range(14 + (y // 28 % 2) * 14, w, 28):
            cv2.circle(img, (x, y), 3, (150, 120, 60), -1, cv2.LINE_AA)
    cv2.circle(img, (w // 2, h // 4), 22, (200, 60, 170), -1, cv2.LINE_AA)
    cv2.circle(img, (w // 2, h // 4), 22, (150, 120, 60), 3, cv2.LINE_AA)
    return img


def clamp01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def ease_in_out(u: float) -> float:
    u = clamp01(u)
    return 0.5 - 0.5 * math.cos(math.pi * u)
