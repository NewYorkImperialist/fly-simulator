"""Procedural meshes and textures for the trampoline job (visual only).

Same approach as ``kebab_assets``: everything is generated in code (numpy + OpenCV)
and handed to MjSpec directly; nothing is written to disk. Nothing here collides or
has mass (callers add the meshes with ``contact_kwargs("visual")`` and ``mass=0``).

* ``disc_mesh``: a thin disc with planar uv (the jumping mat, the ruler base).
* ``leg_arc_mesh``: one bent "W" leg of the frame (a tube along a polyline).
* textures: backyard lawn, woven black mat with a stitched coloured ring, the height
  ruler (1 mm ticks, labels every 2 mm, 2.5 mm "body length" bands), the scoreboard
  face with its labels, a wooden fence with a sky above it.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.broccoli_toss_assets import torus_mesh  # noqa: F401
from fly_simulator.jobs.kebab_assets import (  # noqa: F401  (re-exported helpers)
    MeshData, _mix, _smooth, _to_u8, add_mesh, add_texture, add_textured_material, fbm,
    tile_noise, tube,
)

TAU = 2.0 * math.pi


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def disc_mesh(radius: float, thick: float, n: int = 64) -> MeshData:
    """Closed disc about z (top at +thick/2); uv = planar x / y mapped to [0, 1]."""
    th = np.linspace(0, TAU, n, endpoint=False)
    ring = np.stack([radius * np.cos(th), radius * np.sin(th)], -1)
    rings = np.stack([np.concatenate([ring * 1e-4, np.full((n, 1), thick / 2)], 1),
                      np.concatenate([ring, np.full((n, 1), thick / 2)], 1),
                      np.concatenate([ring, np.full((n, 1), -thick / 2)], 1),
                      np.concatenate([ring * 1e-4, np.full((n, 1), -thick / 2)], 1)])
    uv = 0.5 + 0.5 * rings[..., :2] / radius
    md = tube(rings, uv, cap_start=False, cap_end=False)
    if md.signed_volume() < 0:
        md.faces = md.faces[:, ::-1].copy()
    return md


def polyline_tube(points: np.ndarray, r: float, n: int = 10) -> MeshData:
    """A round tube of radius ``r`` along ``points`` (K, 3), capped at both ends."""
    P = np.asarray(points, float)
    rings, uvs = [], []
    ref = np.array([0.0, 0.0, 1.0])
    th = np.linspace(0, TAU, n, endpoint=False)
    s = 0.0
    for k in range(len(P)):
        t = P[min(k + 1, len(P) - 1)] - P[max(k - 1, 0)]
        t = t / max(np.linalg.norm(t), 1e-9)
        a = ref if abs(t @ ref) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(t, a)
        u /= np.linalg.norm(u)
        v = np.cross(t, u)
        rings.append(P[k] + r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v))
        if k:
            s += float(np.linalg.norm(P[k] - P[k - 1]))
        uvs.append(np.stack([th / TAU, np.full(n, s)], -1))
    md = tube(np.asarray(rings), np.asarray(uvs))
    if md.signed_volume() < 0:
        md.faces = md.faces[:, ::-1].copy()
    return md


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------


def _text(img, text, org, scale, rgb, thick=2, font=cv2.FONT_HERSHEY_DUPLEX, center=False):
    col = tuple(int(255 * c) for c in rgb)  # the images are RGB
    if center:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        org = (int(org[0] - tw / 2), int(org[1] + th / 2))
    cv2.putText(img, text, org, font, scale, col, thick, cv2.LINE_AA)


@lru_cache(maxsize=4)
def lawn_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """Backyard grass: mottled greens, blade streaks, a few clover dots."""
    rng = np.random.default_rng(seed + 11)
    f = fbm(n, n, 6, 6, rng, octaves=5)
    base = _mix(np.array([0.20, 0.42, 0.12], np.float32), np.array([0.36, 0.58, 0.18], np.float32),
                f)
    img = _to_u8(base)
    for _ in range(2600):  # blades
        x, y = rng.integers(0, n, 2)
        L = int(rng.integers(4, 11))
        ang = rng.normal(-math.pi / 2, 0.35)
        c = rng.uniform(0.6, 1.25)
        col = tuple(int(min(255, v * c)) for v in (70, 140, 45))
        cv2.line(img, (int(x), int(y)), (int(x + L * math.cos(ang)), int(y + L * math.sin(ang))),
                 col, 1, cv2.LINE_AA)
    for _ in range(40):  # clover / daisies
        x, y = rng.integers(0, n, 2)
        cv2.circle(img, (int(x), int(y)), 2, (235, 235, 225), -1, cv2.LINE_AA)
    return img


@lru_cache(maxsize=4)
def mat_texture(seed: int = 0, n: int = 512, ring_frac: float = 0.93) -> np.ndarray:
    """Woven black polypropylene mat (planar uv over the disc), a stitched coloured
    ring near the edge and a faint target cross in the middle."""
    rng = np.random.default_rng(seed + 23)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    weave = 0.5 + 0.25 * np.sin(x * 1.6) * np.sin(y * 1.6) + 0.1 * tile_noise(n, n, 32, 32, rng)
    base = (0.06 + 0.05 * weave)[..., None] * np.ones(3, np.float32)
    img = _to_u8(base)
    c = n / 2
    r = ring_frac * n / 2
    cv2.circle(img, (int(c), int(c)), int(r), (40, 90, 210), max(2, n // 90), cv2.LINE_AA)
    for k in range(90):  # the stitching (V-rings where the springs hook in)
        a = TAU * k / 90
        p = (int(c + (r + n * 0.02) * math.cos(a)), int(c + (r + n * 0.02) * math.sin(a)))
        cv2.circle(img, p, max(1, n // 200), (170, 170, 175), -1, cv2.LINE_AA)
    cv2.circle(img, (int(c), int(c)), int(n * 0.06), (200, 200, 60), max(1, n // 160), cv2.LINE_AA)
    return img


def ruler_texture(height_mm: float, body_mm: float, h: int = 1024, w: int = 200) -> np.ndarray:
    """The height ruler, bottom (row h-1) = 0 mm, top = ``height_mm``: 1 mm ticks,
    labels every 2 mm, alternating bands every body length (``body_mm``)."""
    img = np.full((h, w, 3), 242, np.uint8)
    px = h / height_mm
    k = 0
    while k * body_mm < height_mm:
        y0, y1 = int(h - (k + 1) * body_mm * px), int(h - k * body_mm * px)
        if k % 2 == 0:
            img[max(y0, 0):y1, : w // 5] = (250, 200, 40)
        else:
            img[max(y0, 0):y1, : w // 5] = (40, 110, 220)
        _text(img, f"{k + 1}", (w // 10, int((y0 + y1) / 2)), 1.0, (0.1, 0.1, 0.1), 2,
              center=True)
        k += 1
    for mm in range(int(height_mm) + 1):
        y = int(h - mm * px)
        L = w // 2 if mm % 2 == 0 else w // 3
        cv2.line(img, (w - L, y), (w, y), (20, 20, 20), 3 if mm % 2 == 0 else 2)
        if mm % 2 == 0 and 0 < mm < height_mm:
            _text(img, f"{mm}", (int(w * 0.5), y), 1.9, (0.1, 0.1, 0.1), 4, center=True)
        # half-mm ticks
        yh = int(h - (mm + 0.5) * px)
        if mm + 0.5 < height_mm:
            cv2.line(img, (w - w // 6, yh), (w, yh), (60, 60, 60), 1)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (30, 30, 30), 3)
    return img


def scoreboard_texture(h: int = 256, w: int = 512) -> np.ndarray:
    """Scoreboard face: dark panel, title, three labelled digit windows."""
    img = np.full((h, w, 3), 18, np.uint8)
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), (230, 180, 30), 4)
    _text(img, "FLY TRAMPOLINE CLUB", (w // 2, 30), 0.85, (0.95, 0.85, 0.3), 2, center=True)
    for i, lab in enumerate(("BOUNCES", "BEST (mm)", "STREAK")):
        y = int(h * (0.32 + 0.24 * i))
        _text(img, lab, (int(w * 0.24), y), 0.72, (0.9, 0.9, 0.9), 2, center=True)
        cv2.rectangle(img, (int(w * 0.47), y - 29), (int(w * 0.96), y + 29), (5, 5, 5), -1)
    return img


@lru_cache(maxsize=2)
def backdrop_texture(seed: int = 0, h: int = 512, w: int = 1024) -> np.ndarray:
    """A sky gradient over a wooden board fence with a few shrubs."""
    rng = np.random.default_rng(seed + 5)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    sky = _mix(np.array([0.42, 0.66, 0.93], np.float32), np.array([0.80, 0.90, 0.98], np.float32), t)
    img = _to_u8(np.broadcast_to(sky, (h, w, 3)).copy())
    for _ in range(6):  # clouds
        cx, cy = int(rng.integers(0, w)), int(rng.integers(h // 12, h // 3))
        for _k in range(6):
            cv2.circle(img, (cx + int(rng.integers(-40, 40)), cy + int(rng.integers(-10, 10))),
                       int(rng.integers(14, 30)), (248, 250, 252), -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 2.0)
    top = int(h * 0.45)
    nb = 26
    bw = w / nb
    for b in range(nb):
        x0, x1 = int(b * bw), int((b + 1) * bw) - 2
        tone = rng.uniform(0.85, 1.1)
        col = np.array([150, 104, 66]) * tone
        img[top:, x0:x1] = np.clip(col, 0, 255).astype(np.uint8)
        grain = (tile_noise(h - top, max(x1 - x0, 1), 24, 2, rng) * 26).astype(np.int16)
        img[top:, x0:x1] = np.clip(img[top:, x0:x1].astype(np.int16) - grain[..., None], 0, 255)
        cv2.fillPoly(img, [np.array([[x0, top], [x1, top], [(x0 + x1) // 2, top - int(bw * 0.4)]])],
                     tuple(int(v) for v in np.clip(col, 0, 255)))
    for yy in (int(h * 0.58), int(h * 0.86)):  # rails
        img[yy:yy + 10] = (120, 82, 50)
    for _ in range(5):  # shrubs in front of the fence
        cx = int(rng.integers(0, w))
        for _k in range(9):
            cv2.circle(img, (cx + int(rng.integers(-50, 50)), h - int(rng.integers(0, 70))),
                       int(rng.integers(20, 40)), (int(rng.integers(30, 60)), int(rng.integers(90, 130)),
                                                    int(rng.integers(30, 50))), -1, cv2.LINE_AA)
    return img
