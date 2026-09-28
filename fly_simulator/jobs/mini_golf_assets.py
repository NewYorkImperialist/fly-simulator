"""Procedural meshes / textures for the mini golf job (visual only).

Everything is generated in code (numpy + OpenCV) and handed to MjSpec directly; no
files on disk. Bright arcade colours; the course name ("FLY GOLF") and all signs are
our own, no real brand or logo.
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
from fly_simulator.jobs.mowing_assets import gable_roof_mesh, shrub_mesh, flower_clump_mesh  # noqa: F401
from fly_simulator.jobs.sisyphus_assets import box_mesh  # noqa: F401
from fly_simulator.jobs.trampoline_assets import _text, disc_mesh, polyline_tube  # noqa: F401


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def cup_liner_mesh(r: float, depth: float, wall: float = 0.04, n: int = 32) -> MeshData:
    """An open cylinder (the cup liner), rim at z = 0, bottom at -depth."""
    rr = np.array([0.0, r + wall, r + wall, r, r, 0.0])
    zz = np.array([-depth - wall, -depth - wall, 0.0, 0.0, -depth, -depth])
    return lathe(rr, zz, n, v_coord=np.linspace(0, 1, len(rr)))


def ball_mesh(r: float) -> MeshData:
    return sphere_mesh(r, (0, 0, 0), 32, 18)


def blade_mesh(L: float, w: float, t: float = 0.05, hub_gap: float = 0.25) -> MeshData:
    """One windmill sail along +z in the y-z plane: a lattice frame (two stiles and
    rungs) from hub_gap to L."""
    parts = []
    for s in (-1, 1):
        parts.append(transform(box_mesh((t / 2, 0.035, (L - hub_gap) / 2)), np.eye(3),
                               np.array([0.0, s * w / 2, (L + hub_gap) / 2])))
    parts.append(transform(box_mesh((t / 2, 0.03, (L - hub_gap) / 2 + 0.05)), np.eye(3),
                           np.array([0.0, 0.0, (L + hub_gap) / 2])))
    n = 5
    for k in range(n):
        z = hub_gap + (k + 0.5) * (L - hub_gap) / n
        parts.append(transform(box_mesh((t / 2, w / 2, 0.025)), np.eye(3), np.array([0.0, 0.0, z])))
    return merge(*parts)


def post_mesh(r: float, h: float) -> MeshData:
    """A rounded bumper post (base at z = 0)."""
    zz = np.array([0.0, 0.0, 0.85 * h, 0.96 * h, h, h])
    rr = np.array([0.0, r, r, 0.93 * r, 0.7 * r, 0.0])
    return lathe(rr, zz, 28)


def pipe_mesh(r_in: float, r_out: float, L: float, n: int = 32) -> MeshData:
    """A half-buried pipe along x (axis at z = r_out - ... handled by the caller): an
    open tube about x from -L/2 to L/2 (outer, lip and inner surfaces)."""
    th = np.linspace(0, TAU, n, endpoint=False)
    ring = lambda r, x: np.column_stack([np.full(n, x), r * np.cos(th), r * np.sin(th)])  # noqa: E731
    rings = np.stack([ring(r_in, -L / 2), ring(r_out, -L / 2), ring(r_out, L / 2), ring(r_in, L / 2)])
    uv = np.zeros(rings.shape[:2] + (2,))
    uv[..., 0] = (th / TAU)[None]
    uv[..., 1] = np.array([0.0, 0.1, 1.0, 1.1])[:, None] * L
    md = tube(rings, uv, cap_start=False, cap_end=False)
    # the inner surface closes the loop back to ring 0
    idx = np.arange(4 * n).reshape(4, n)
    a, b = idx[3], idx[0]
    a1, b1 = np.roll(a, -1), np.roll(b, -1)
    f = np.concatenate([md.faces, np.stack([a, a1, b1], 1), np.stack([a, b1, b], 1)])
    return MeshData(md.verts, f, md.uv)


def palm_meshes(h: float, seed: int = 0) -> dict[str, MeshData]:
    """A cartoon palm tree: a curved segmented ``trunk`` and a crown of ``fronds``."""
    rng = np.random.default_rng(seed + 71)
    bend = rng.uniform(-0.25, 0.25) * h
    zs = np.linspace(0, h, 10)
    pts = np.column_stack([bend * (zs / h) ** 2, np.zeros(10), zs])
    trunk = polyline_tube(pts, 0.035 * h, 10)
    top = pts[-1]
    fronds = []
    for k in range(7):
        a = TAU * k / 7 + rng.uniform(-0.2, 0.2)
        d = np.array([math.cos(a), math.sin(a), 0.0])
        s = np.linspace(0, 1, 7)
        fp = top + np.outer(s, d) * 0.45 * h + np.outer(0.12 * h * s - 0.3 * h * s ** 2, [0, 0, 1])
        fronds.append(polyline_tube(fp, 0.07 * h, 6))
    return {"trunk": trunk, "fronds": merge(*fronds)}


# ---------------------------------------------------------------------------
# textures (uint8 RGB)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=6)
def felt_texture(seed: int = 0, n: int = 256, rgb=(0.10, 0.62, 0.22)) -> np.ndarray:
    """Putting-green felt: fine fibre noise over a bright green."""
    rng = np.random.default_rng(seed + 3)
    f = fbm(n, n, 8, 8, rng, octaves=4)
    fine = rng.random((n, n)).astype(np.float32)
    c = np.asarray(rgb, np.float32)
    img = c * (0.86 + 0.18 * f + 0.10 * (fine - 0.5))[..., None]
    return _to_u8(img)


@lru_cache(maxsize=2)
def paver_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """Colourful walkway pavers (sand / terracotta / slate) with grout."""
    rng = np.random.default_rng(seed + 9)
    img = np.full((n, n, 3), 205, np.uint8)
    cols = [(0.93, 0.80, 0.58), (0.86, 0.52, 0.38), (0.78, 0.74, 0.70), (0.95, 0.70, 0.45)]
    s = n // 8
    for i in range(8):
        for j in range(8):
            off = (s // 2) * (i % 2)
            x0 = (j * s + off) % n
            c = np.array(cols[int(rng.integers(0, 4))]) * rng.uniform(0.9, 1.05)
            cv2.rectangle(img, (x0 + 2, i * s + 2), (min(x0 + s - 3, n - 1), i * s + s - 3),
                          tuple(int(255 * v) for v in np.clip(c, 0, 1)), -1)
            if x0 + s > n:
                cv2.rectangle(img, (0, i * s + 2), (x0 + s - n - 3, i * s + s - 3),
                              tuple(int(255 * v) for v in np.clip(c, 0, 1)), -1)
    noise = fbm(n, n, 16, 16, rng, octaves=3)
    img = np.clip(img.astype(np.float32) * (0.9 + 0.15 * noise[..., None]), 0, 255).astype(np.uint8)
    return img


@lru_cache(maxsize=2)
def golf_ball_texture(n: int = 128) -> np.ndarray:
    """White with a dimple pattern."""
    img = np.full((n, 2 * n, 3), 248, np.uint8)
    for i in range(10):
        for j in range(22):
            x = int((j + 0.5 * (i % 2)) * 2 * n / 22)
            y = int((i + 0.5) * n / 10)
            cv2.circle(img, (x, y), max(1, n // 40), (215, 215, 222), -1, cv2.LINE_AA)
    return img


def sign_texture(hole: int, name: str, par: int, colour, h: int = 192, w: int = 320) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = tuple(int(255 * c) for c in colour)
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), (255, 255, 255), 6)
    _text(img, f"HOLE {hole}", (w // 2, int(h * 0.26)), 1.3, (1, 1, 1), 3, center=True)
    _text(img, name, (w // 2, int(h * 0.55)), 1.0, (1, 0.95, 0.3), 2, center=True)
    _text(img, f"PAR {par}", (w // 2, int(h * 0.80)), 1.1, (1, 1, 1), 2, center=True)
    return img


def flag_texture(hole: int, colour, h: int = 64, w: int = 96) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = tuple(int(255 * c) for c in colour)
    _text(img, str(hole), (w // 3, h // 2), 1.2, (1, 1, 1), 3, center=True)
    return img


def neon_sign_texture(h: int = 160, w: int = 640) -> np.ndarray:
    img = np.full((h, w, 3), 20, np.uint8)
    cv2.rectangle(img, (5, 5), (w - 6, h - 6), (255, 60, 200), 5)
    _text(img, "FLY GOLF", (w // 2, int(h * 0.40)), 2.0, (1.0, 0.9, 0.2), 5, center=True)
    _text(img, "4 HOLES OF ETERNITY", (w // 2, int(h * 0.78)), 0.9, (0.3, 1.0, 1.0), 2, center=True)
    return img


def barn_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Red board siding with white trim lines."""
    rng = np.random.default_rng(seed + 31)
    img = np.zeros((n, n, 3), np.float32)
    img[:] = (0.80, 0.12, 0.12)
    for k in range(0, n, n // 12):
        img[:, k:k + 2] *= 0.7
    img *= (0.9 + 0.15 * fbm(n, n, 8, 8, rng, octaves=3))[..., None]
    img = _to_u8(img)
    cv2.rectangle(img, (0, 0), (n - 1, n - 1), (250, 250, 250), n // 20)
    cv2.line(img, (0, 0), (n, n), (250, 250, 250), n // 28)
    cv2.line(img, (n, 0), (0, n), (250, 250, 250), n // 28)
    return img


def stripe_texture(c1, c2, n: int = 128, k: int = 8) -> np.ndarray:
    """Diagonal candy stripes (the ramp / the pipe)."""
    y, x = np.mgrid[0:n, 0:n]
    band = ((x + y) // (n // k)) % 2
    a = np.asarray(c1, np.float32)
    b = np.asarray(c2, np.float32)
    return _to_u8(np.where(band[..., None] == 0, a, b))


@lru_cache(maxsize=2)
def sky_texture(seed: int = 0, h: int = 512, w: int = 1024) -> np.ndarray:
    """A bright arcade backdrop: blue sky, clouds, palm silhouettes, a castle."""
    rng = np.random.default_rng(seed + 5)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    sky = _mix(np.array([0.25, 0.55, 0.98], np.float32), np.array([0.75, 0.90, 1.0], np.float32), t)
    img = _to_u8(np.broadcast_to(sky, (h, w, 3)).copy())
    for _ in range(8):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(h // 12, h // 2.5))
        for _k in range(7):
            cv2.circle(img, (cx + int(rng.integers(-45, 45)), cy + int(rng.integers(-10, 10))),
                       int(rng.integers(14, 32)), (250, 252, 255), -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    base = int(h * 0.78)
    # a toy castle
    cx = int(w * 0.62)
    cv2.rectangle(img, (cx - 90, base - 110), (cx + 90, base), (240, 140, 190), -1)
    for dx in (-90, 90):
        cv2.rectangle(img, (cx + dx - 25, base - 170), (cx + dx + 25, base), (200, 90, 200), -1)
        cv2.fillPoly(img, [np.array([[cx + dx - 32, base - 170], [cx + dx + 32, base - 170],
                                     [cx + dx, base - 230]])], (60, 60, 200))
    cv2.ellipse(img, (cx, base), (28, 40), 0, 180, 360, (70, 40, 60), -1)
    for _ in range(6):  # palms
        px = int(rng.integers(0, w))
        cv2.line(img, (px, base), (px + int(rng.integers(-20, 20)), base - 150), (110, 70, 40), 8)
        for k in range(6):
            a = TAU * k / 6
            cv2.ellipse(img, (px + int(45 * math.cos(a)), base - 150 + int(15 * math.sin(a))),
                        (45, 10), math.degrees(a), 0, 360, (40, 150, 60), -1, cv2.LINE_AA)
    img[base:] = (60, 170, 80)
    return img
