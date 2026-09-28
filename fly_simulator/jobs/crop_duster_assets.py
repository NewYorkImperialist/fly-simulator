"""Procedural meshes / textures for the crop duster job (visual only).

Generated in code (numpy + OpenCV), handed to MjSpec directly; nothing on disk. Every
sign is our own (no real brand).
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
from fly_simulator.jobs.mini_golf_assets import barn_texture  # noqa: F401
from fly_simulator.jobs.mowing_assets import shrub_mesh  # noqa: F401
from fly_simulator.jobs.sisyphus_assets import box_mesh  # noqa: F401
from fly_simulator.jobs.trampoline_assets import disc_mesh, polyline_tube  # noqa: F401

N_STAGES = 5  # crop dust stages: 0 = untreated green .. 4 = fully dusted


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def crop_segment_mesh(L: float, w: float, h: float, seed: int = 0) -> MeshData:
    """A short piece of a crop row (along x): three leafy mounds on a ridge."""
    rng = np.random.default_rng(seed + 300)
    parts = []
    for k in range(3):
        x = (k - 1) * L / 3
        r = w / 2 * rng.uniform(0.85, 1.0)
        parts.append(sphere_mesh(r, (x + rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), h * 0.45), 10, 7,
                                 squash=h / (2 * r) * 1.1))
    return merge(*parts)


def fence_mesh(p0, p1, h: float = 1.2, pitch: float = 2.5) -> MeshData:
    """A post-and-rail farm fence from p0 to p1 (x, y) on the ground."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    n = max(2, int(round(L / pitch)) + 1)
    parts = []
    for u in np.linspace(0, 1, n):
        p = p0 + u * (p1 - p0)
        parts.append(transform(box_mesh((0.07, 0.07, h / 2), 1.0), np.eye(3), np.array([p[0], p[1], h / 2])))
    for z in (0.45 * h, 0.85 * h):
        parts.append(polyline_tube(np.array([[p0[0], p0[1], z], [p1[0], p1[1], z]]), 0.035, 6))
    return merge(*parts)


def barn_meshes(w: float = 7.0, d: float = 9.0, h: float = 4.0, roof: float = 3.2) -> dict[str, MeshData]:
    """A red barn (ridge along y): ``walls`` and a gambrel ``roof`` (grey), white
    ``trim`` and a hay loft door."""
    walls = transform(box_mesh((w / 2, d / 2, h / 2), 0.25), np.eye(3), np.array([0, 0, h / 2]))
    # gambrel profile in (x, z)
    prof = np.array([[-w / 2 - 0.3, h], [-w / 2 * 0.62, h + roof * 0.62], [0.0, h + roof],
                     [w / 2 * 0.62, h + roof * 0.62], [w / 2 + 0.3, h]])
    rings = []
    for y in (-d / 2 - 0.3, d / 2 + 0.3):
        loop = np.column_stack([prof[:, 0], np.full(len(prof), y), prof[:, 1]])
        rings.append(loop)
    roof_m = tube(np.array(rings), np.zeros((2, len(prof), 2)))
    trim = []
    for s in (-1, 1):
        y = s * (d / 2 + 0.02)
        for a, b in ((prof[0], prof[1]), (prof[1], prof[2]), (prof[2], prof[3]), (prof[3], prof[4])):
            trim.append(polyline_tube(np.array([[a[0], y, a[1]], [b[0], y, b[1]]]), 0.08, 6))
        # the big door: an X brace in a frame
        dw, dh = w * 0.22, h * 0.7
        for a, b in (((-dw, 0), (dw, dh)), ((dw, 0), (-dw, dh)), ((-dw, dh), (dw, dh)), ((-dw, 0), (-dw, dh)),
                     ((dw, 0), (dw, dh))):
            trim.append(polyline_tube(np.array([[a[0], y * 1.002, a[1]], [b[0], y * 1.002, b[1]]]), 0.07, 6))
    return {"walls": walls, "roof": roof_m, "trim": merge(*trim)}


def silo_meshes(r: float = 1.6, h: float = 9.0) -> dict[str, MeshData]:
    body = lathe(np.array([0.0, r, r, 0.0]), np.array([0.0, 0.0, h, h]), 24, v_coord=np.array([0, 0, 1, 1.0]))
    a = np.linspace(0, math.pi / 2, 8)
    dome = lathe(np.concatenate([r * 1.04 * np.cos(a), [0.0]]),
                 np.concatenate([h + r * 0.8 * np.sin(a), [h + r * 0.8]]), 24)
    rings = merge(*[transform(lathe(np.array([0.0, r * 1.03, r * 1.03, 0.0]), np.array([0, 0, 0.08, 0.08]), 24),
                              np.eye(3), np.array([0, 0, z])) for z in np.linspace(1.0, h - 0.5, 6)])
    return {"body": body, "dome": dome, "bands": rings}


def windpump_meshes(h: float = 11.0) -> dict[str, MeshData]:
    """A farm windpump: a lattice ``tower`` (legs + braces) with a platform. The
    rotor is separate (``windpump_rotor``)."""
    parts = []
    b0, b1 = 1.4, 0.3
    corners = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
    for sx, sy in corners:
        parts.append(polyline_tube(np.array([[sx * b0, sy * b0, 0], [sx * b1, sy * b1, h]]), 0.06, 6))
    for z0, z1 in zip(np.linspace(0, h, 6)[:-1], np.linspace(0, h, 6)[1:]):
        w0 = b0 + (b1 - b0) * z0 / h
        w1 = b0 + (b1 - b0) * z1 / h
        for k in range(4):
            (ax, ay), (bx, by) = corners[k], corners[(k + 1) % 4]
            parts.append(polyline_tube(np.array([[ax * w0, ay * w0, z0], [bx * w1, by * w1, z1]]), 0.03, 5))
            parts.append(polyline_tube(np.array([[ax * w1, ay * w1, z1], [bx * w1, by * w1, z1]]), 0.03, 5))
    parts.append(transform(box_mesh((0.5, 0.5, 0.05), 1.0), np.eye(3), np.array([0, 0, h])))
    parts.append(polyline_tube(np.array([[0, 0, h], [0, 0, h + 0.8]]), 0.08, 8))
    return {"tower": merge(*parts)}


def windpump_rotor(r: float = 1.8, n: int = 14) -> dict[str, MeshData]:
    """The rotor in the y-z plane (it turns about x): ``blades`` (a ring of tilted
    vanes), ``hub``; and the ``tail`` vane behind it (-x)."""
    blades = []
    for k in range(n):
        a = TAU * k / n
        R = _rot_x(a) @ _rot_z(0.5)
        blades.append(transform(box_mesh((0.02, 0.16, r * 0.36), 1.0), R, _rot_x(a) @ np.array([0, 0, r * 0.62])))
    ring = [polyline_tube(np.array([[0, rr * math.cos(t), rr * math.sin(t)] for t in np.linspace(0, TAU, 25)]),
                          0.03, 5) for rr in (r * 0.3, r * 0.98)]
    hub = transform(lathe(np.array([0.0, 0.22, 0.22, 0.0]), np.array([-0.2, -0.2, 0.2, 0.2]), 12), _rot_y(math.pi / 2),
                    np.zeros(3))
    tail = merge(polyline_tube(np.array([[0, 0, 0], [-2.6, 0, 0]]), 0.04, 6),
                 transform(box_mesh((0.7, 0.02, 0.5), 1.0), np.eye(3), np.array([-2.7, 0, 0.1])))
    return {"blades": merge(*blades, *ring), "hub": hub, "tail": tail}


def hangar_meshes(w: float = 7.0, d: float = 6.0, h: float = 3.0) -> dict[str, MeshData]:
    """A small Quonset hangar (arched roof along x, opening toward +y)."""
    a = np.linspace(0, math.pi, 16)
    loop = np.column_stack([np.full(16, 0.0), (d / 2) * np.cos(a), h * np.sin(a)])
    rings = np.array([loop + np.array([x, 0, 0]) for x in (-w / 2, w / 2)])
    # the arch as a shell along x (rings in y-z, the tube runs along x)
    shell = tube(rings, np.zeros((2, 16, 2)))
    back = transform(box_mesh((w / 2, 0.05, h * 0.5), 1.0), np.eye(3), np.array([0, -d / 2 + 0.3, h * 0.5]))
    return {"shell": shell, "back": back}


def hopper_meshes() -> dict[str, MeshData]:
    """The dust hopper at the airstrip (the refill station): a tank on legs with a
    chute."""
    tank = lathe(np.array([0.0, 0.3, 1.0, 1.0, 0.0]), np.array([1.4, 1.4, 2.2, 3.6, 3.6]), 18)
    legs = merge(*[polyline_tube(np.array([[0.85 * math.cos(t), 0.85 * math.sin(t), 0.0],
                                           [0.85 * math.cos(t), 0.85 * math.sin(t), 2.3]]), 0.07, 6)
                   for t in np.linspace(0, TAU, 4, endpoint=False) + 0.4])
    chute = polyline_tube(np.array([[0, 0, 1.4], [0, 0, 0.9], [0.6, 0, 0.6]]), 0.12, 8)
    return {"tank": tank, "legs": legs, "chute": chute}


def windsock_meshes() -> dict[str, MeshData]:
    return {"pole": polyline_tube(np.array([[0, 0, 0], [0, 0, 3.4]]), 0.05, 6)}


def tree_meshes(h: float, seed: int = 0) -> dict[str, MeshData]:
    rng = np.random.default_rng(seed + 90)
    trunk = lathe(np.array([0.25, 0.18, 0.12, 0.0]), np.array([0.0, h * 0.4, h * 0.6, h * 0.6]), 10)
    blobs = []
    for _ in range(9):
        off = rng.normal(0, 0.26 * h, 3) * np.array([1, 1, 0.5])
        blobs.append(sphere_mesh(h * rng.uniform(0.2, 0.3), (off[0], off[1], h * 0.7 + off[2]), 12, 8))
    return {"trunk": trunk, "crown": merge(*blobs)}


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def meadow_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed + 501)
    f = fbm(n, n, 6, 6, rng, octaves=4)
    fine = fbm(n, n, 40, 40, rng, octaves=1)
    base = _mix(np.array([0.36, 0.52, 0.22], np.float32), np.array([0.52, 0.60, 0.28], np.float32), f)
    img = base * (0.94 + 0.08 * fine[..., None])
    return _to_u8(img)


@lru_cache(maxsize=2)
def soil_texture(seed: int = 0, n: int = 256, rows: int = 8) -> np.ndarray:
    """Tilled soil with furrows along u."""
    rng = np.random.default_rng(seed + 502)
    f = fbm(n, n, 8, 8, rng, octaves=3)
    v = np.linspace(0, rows * TAU, n, dtype=np.float32)[:, None]
    fur = 0.5 + 0.5 * np.cos(v)
    g = 0.30 + 0.08 * f + 0.07 * fur
    img = np.stack([g * 1.25, g * 0.92, g * 0.62], -1)
    return _to_u8(img)


@lru_cache(maxsize=8)
def crop_texture(stage: int, seed: int = 0, n: int = 128) -> np.ndarray:
    """Leafy green crop; ``stage`` 1..4 adds a pale yellow-white dust coat (the
    treated rows)."""
    rng = np.random.default_rng(seed + 503)
    f = fbm(n, n, 8, 8, rng, octaves=3)
    img = _mix(np.array([0.16, 0.42, 0.12], np.float32), np.array([0.40, 0.66, 0.22], np.float32), f)
    for _ in range(90):  # leaf highlights
        x, y = int(rng.integers(0, n)), int(rng.integers(0, n))
        cv2.ellipse(img, (x, y), (int(rng.integers(3, 7)), 2), float(rng.uniform(0, 180)), 0, 360,
                    (0.50, 0.76, 0.30), -1, cv2.LINE_AA)
    if stage > 0:
        a = stage / (N_STAGES - 1)
        speck = fbm(n, n, 24, 24, np.random.default_rng(seed + 504), octaves=2)
        cover = np.clip(a * 1.25 - 0.35 + 0.5 * (speck - 0.5), 0, 1) * (0.25 + 0.6 * a)
        img = _mix(img, (0.93, 0.90, 0.72), cover)
    return _to_u8(img)


@lru_cache(maxsize=2)
def strip_texture(n: int = 128) -> np.ndarray:
    """The airstrip: mown grass with white edge markers and a centre line."""
    img = np.zeros((n, n * 4, 3), np.float32)
    img[:] = (0.46, 0.60, 0.30)
    img[:, ::8] *= 0.95
    img = _to_u8(img)
    for x in range(8, n * 4, 48):
        cv2.rectangle(img, (x, n // 2 - 3), (x + 24, n // 2 + 3), (240, 240, 235), -1)
    cv2.rectangle(img, (0, 0), (n * 4 - 1, 6), (240, 240, 235), -1)
    cv2.rectangle(img, (0, n - 7), (n * 4 - 1, n - 1), (240, 240, 235), -1)
    return img


@lru_cache(maxsize=2)
def sign_texture(text: str, sub: str, bg=(0.85, 0.72, 0.15), h: int = 160, w: int = 384) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = [int(255 * c) for c in bg]
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), (40, 30, 20), 5)
    font = cv2.FONT_HERSHEY_DUPLEX
    for t, sc, y in ((text, 1.3, int(h * 0.42)), (sub, 0.75, int(h * 0.76))):
        (tw, th), _ = cv2.getTextSize(t, font, sc, 3)
        cv2.putText(img, t, ((w - tw) // 2, y + th // 2), font, sc, (40, 30, 20), 3, cv2.LINE_AA)
    return img


@lru_cache(maxsize=2)
def sky_texture(seed: int = 0, h: int = 512, w: int = 1024) -> np.ndarray:
    """A summer sky with fair-weather clouds over rolling farmland and hedgerows."""
    rng = np.random.default_rng(seed + 505)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    sky = _mix(np.array([0.30, 0.55, 0.92], np.float32), np.array([0.86, 0.92, 0.98], np.float32), t)
    img = np.broadcast_to(sky, (h, w, 3)).copy()
    cloud = fbm(h, w, 3, 8, rng, octaves=4)
    img = _mix(img, (1.0, 1.0, 1.0), np.clip((cloud - 0.55) * 2.5, 0, 0.85) * (t < 0.55))
    img = _to_u8(img)
    base = int(h * 0.74)
    xs = np.arange(w)
    hills = (base - 30 - 18 * np.sin(xs / w * TAU * 2 + 1.0) - 10 * np.sin(xs / w * TAU * 5)).astype(int)
    for x in range(w):
        img[hills[x]:, x] = (110, 150, 80)
    for _ in range(60):
        x = int(rng.integers(0, w))
        cv2.circle(img, (x, hills[min(x, w - 1)] + 4), int(rng.integers(8, 20)), (60, 100, 55), -1, cv2.LINE_AA)
    img[base:] = (120, 160, 80)
    return img
