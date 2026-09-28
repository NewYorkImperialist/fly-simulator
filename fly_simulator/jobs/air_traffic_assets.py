"""Procedural meshes / textures for the air traffic job (visual only).

Generated in code (numpy + OpenCV) and handed to MjSpec directly; nothing on disk.
The airport code ("FLY") and every label are our own; no real airline, brand or logo.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.broccoli_toss_assets import merge, panel_mesh, panel_quat, sphere_mesh  # noqa: F401
from fly_simulator.jobs.kebab_assets import (  # noqa: F401  (re-exported helpers)
    TAU, MeshData, _mix, _to_u8, add_mesh, add_texture, add_textured_material, fbm, lathe, transform,
)
from fly_simulator.jobs.sisyphus_assets import box_mesh  # noqa: F401
from fly_simulator.jobs.trampoline_assets import disc_mesh, polyline_tube  # noqa: F401


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def plane_meshes(L: float = 2.4, span: float = 2.6) -> dict[str, MeshData]:
    """A small airliner (plane frame: +x the nose, +y the left wing, +z up):
    ``body`` (the fuselage), ``wings`` (main wings + tailplane), ``fin`` (the tail fin)."""
    r = 0.14 * L / 2.4
    z = np.array([0.0, 0.08, 0.3, 0.6, 0.85, 0.95, 1.0]) * L  # tail (z = 0) -> nose
    prof = np.array([0.05, 0.45, 0.95, 1.0, 0.95, 0.65, 0.05]) * r
    body = lathe(prof, z, 18)
    # lathe is about z: turn it so the axis is x (nose at +x), centred
    body = transform(body, _rot_y(math.pi / 2), np.array([-0.5 * L, 0.0, 0.0]))

    def slab(x0, x1, y0, y1, t, sweep=0.0, zc=0.0):
        """A flat tapered wing panel from the root chord (x0..x1 at y0) to the tip."""
        v = []
        for y in (y0, y1):
            k = abs(y - y0) / max(abs(y1 - y0), 1e-9)
            xa = x0 - sweep * k
            xb = x1 - sweep * k - 0.45 * (x1 - x0) * k
            for x in (xa, xb):
                for zz in (zc - t / 2, zc + t / 2):
                    v.append((x, y, zz))
        v = np.array(v, float)
        # 8 corners of a skewed box: index = iy * 4 + ix * 2 + iz
        f = [(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4),
             (2, 6, 7), (2, 7, 3), (1, 3, 7), (1, 7, 5), (0, 4, 6), (0, 6, 2)]
        faces = np.array(f, int)
        # orient outward (the signed volume must be positive)
        md = MeshData(v, faces, np.zeros((len(v), 2)))
        if md.signed_volume() < 0:
            md = MeshData(v, faces[:, ::-1], md.uv)
        return md

    hs = span / 2
    wings = [slab(0.25 * L, -0.12 * L, 0.0, hs, 0.035, sweep=0.18 * L, zc=-0.03),
             slab(0.25 * L, -0.12 * L, 0.0, -hs, 0.035, sweep=0.18 * L, zc=-0.03),
             slab(-0.34 * L, -0.5 * L, 0.0, 0.36 * L, 0.025, sweep=0.1 * L, zc=0.05),
             slab(-0.34 * L, -0.5 * L, 0.0, -0.36 * L, 0.025, sweep=0.1 * L, zc=0.05)]
    fin_v = np.array([[-0.28 * L, -0.015, 0.08], [-0.5 * L, -0.015, 0.08], [-0.5 * L, -0.015, 0.45 * L],
                      [-0.42 * L, -0.015, 0.45 * L], [-0.28 * L, 0.015, 0.08], [-0.5 * L, 0.015, 0.08],
                      [-0.5 * L, 0.015, 0.45 * L], [-0.42 * L, 0.015, 0.45 * L]])
    fin_f = np.array([(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
                      (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)])
    fin = MeshData(fin_v, fin_f, np.zeros((8, 2)))
    if fin.signed_volume() < 0:
        fin = MeshData(fin_v, fin_f[:, ::-1], fin.uv)
    return {"body": body, "wings": merge(*wings), "fin": fin}


@lru_cache(maxsize=2)
def radar_texture(n: int = 256) -> np.ndarray:
    """A radar scope: dark green disc, range rings, bearing ticks, a runway mark."""
    img = np.zeros((n, n, 3), np.uint8)
    img[:] = (6, 20, 8)
    c = n // 2
    cv2.circle(img, (c, c), c - 4, (10, 50, 18), -1, cv2.LINE_AA)
    for k in range(1, 5):
        cv2.circle(img, (c, c), int((c - 6) * k / 4), (40, 150, 60), 1, cv2.LINE_AA)
    for a in range(0, 360, 30):
        x, y = math.cos(math.radians(a)), math.sin(math.radians(a))
        cv2.line(img, (int(c + x * (c - 20)), int(c + y * (c - 20))), (int(c + x * (c - 6)), int(c + y * (c - 6))),
                 (60, 190, 80), 2, cv2.LINE_AA)
    cv2.line(img, (c, c), (c, c - int(0.55 * c)), (120, 230, 140), 3)  # the runway (up = +x)
    cv2.putText(img, "FLY APP", (8, n - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 100), 1, cv2.LINE_AA)
    return img


@lru_cache(maxsize=2)
def strips_texture(h: int = 128, w: int = 256) -> np.ndarray:
    """A flight-strip board (our own flight numbers)."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (30, 30, 34)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(5):
        y0 = 6 + i * 24
        col = [(230, 230, 210), (200, 230, 250), (230, 210, 180)][i % 3]
        cv2.rectangle(img, (6, y0), (w - 6, y0 + 20), col, -1)
        cv2.putText(img, f"FW{101 + 7 * i:03d}  A{3 + i}  F{40 + 10 * i:03d}", (12, y0 + 15), font, 0.45,
                    (20, 20, 20), 1, cv2.LINE_AA)
    return img


@lru_cache(maxsize=2)
def ground_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """The airfield at night: dark grass with a faint mottling."""
    rng = np.random.default_rng(seed + 29)
    g = 0.05 + 0.04 * fbm(n, n, 6, 6, rng, octaves=4)
    img = np.stack([g * 0.7, g * 1.1, g * 0.8], -1)
    return _to_u8(img)


@lru_cache(maxsize=2)
def runway_texture(h: int = 64, w: int = 1024) -> np.ndarray:
    """The runway (u along its length): asphalt, a dashed centre line, the threshold
    bars and the designator numbers (ours: 09)."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (34, 34, 36)
    for x in range(90, w - 40, 36):
        cv2.rectangle(img, (x, h // 2 - 1), (x + 18, h // 2 + 1), (220, 220, 220), -1)
    for k in range(6):
        y = 8 + k * 9
        cv2.rectangle(img, (6, y), (40, y + 4), (230, 230, 230), -1)
    cv2.putText(img, "09", (46, h // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 1), (w - 1, 3), (200, 200, 200), -1)
    cv2.rectangle(img, (0, h - 4), (w - 1, h - 2), (200, 200, 200), -1)
    return img


@lru_cache(maxsize=2)
def floor_texture(n: int = 256) -> np.ndarray:
    """The tower cab's carpet tiles."""
    rng = np.random.default_rng(3)
    base = 0.13 + 0.03 * rng.random((n, n)).astype(np.float32)
    img = np.stack([base * 0.8, base * 0.9, base * 1.2], -1)
    img = _to_u8(img)
    for k in range(0, n, n // 4):
        cv2.line(img, (k, 0), (k, n), (20, 22, 30), 2)
        cv2.line(img, (0, k), (n, k), (20, 22, 30), 2)
    return img


def bezier(p0, p1, p2, p3, u: float) -> np.ndarray:
    u = min(max(u, 0.0), 1.0)
    a = 1.0 - u
    return (a ** 3) * np.asarray(p0) + 3 * a * a * u * np.asarray(p1) + 3 * a * u * u * np.asarray(p2) \
        + (u ** 3) * np.asarray(p3)
