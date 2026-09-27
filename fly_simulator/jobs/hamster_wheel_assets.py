"""Procedural meshes and textures for the hamster wheel job (visual only).

Same approach as ``kebab_assets``: everything is generated in code (numpy + OpenCV)
and handed to MjSpec directly; nothing is written to disk. Nothing here collides or
has mass (callers add the meshes with ``contact_kwargs("visual")`` and ``mass=0``).

A pet cage:

* meshes: the wheel's back disc, raised spokes and hub, the outer rims, an A-frame
  stand, the cage's wire bars (one merged mesh), a water bottle with its spout and
  holder, a ceramic food bowl heaped with pellets, a wooden hideout, sunflower seeds;
* textures: wood-shaving bedding, the wheel's back disc (spokes, vent slots, a
  little paw-print badge), the hideout's planks and the room behind the cage.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.broccoli_toss_assets import merge, sphere_mesh, torus_mesh  # noqa: F401
from fly_simulator.jobs.kebab_assets import (  # noqa: F401  (re-exported helpers)
    MeshData, _mix, _smooth, _to_u8, add_mesh, add_texture, add_textured_material, fbm, lathe,
    tile_noise, transform,
)
from fly_simulator.jobs.pizza_chef_assets import bowl_mesh  # noqa: F401
from fly_simulator.jobs.trampoline_assets import disc_mesh, polyline_tube  # noqa: F401

TAU = 2.0 * math.pi


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def spokes_mesh(r_in: float, r_out: float, n: int = 6, r: float = 0.12) -> MeshData:
    """``n`` straight spokes in the x-z plane (the wheel's plane, axle along y) from
    the hub to the rim, plus a ring at the rim, as one mesh."""
    parts = []
    for k in range(n):
        a = TAU * k / n
        d = np.array([math.sin(a), 0.0, -math.cos(a)])
        parts.append(polyline_tube(np.array([d * r_in, d * r_out]), r, 8))
    ring = torus_mesh(r_out, r * 1.2, 64, 8)
    parts.append(transform(ring, _rot_x(math.pi / 2), np.zeros(3)))
    return merge(*parts)


def stand_mesh(zc: float, y: float, foot: float = 4.2, r: float = 0.22) -> MeshData:
    """An A-frame of two bent tubes from the feet (z = 0, x = +-foot) up to the axle
    at (0, y, zc), and a tube foot bar joining the feet."""
    parts = []
    for sx in (-1.0, 1.0):
        pts = np.array([[sx * foot, y, r], [sx * foot * 0.55, y, zc * 0.45],
                        [sx * 0.5, y, zc - 0.2], [0.0, y, zc]])
        parts.append(polyline_tube(pts, r, 10))
    parts.append(polyline_tube(np.array([[-foot - 0.4, y, r], [foot + 0.4, y, r]]), r, 10))
    return merge(*parts)


def bars_mesh(x0: float, x1: float, y: float, z0: float, z1: float, pitch: float = 1.1,
              r: float = 0.06, rails=(0.35, 0.72)) -> MeshData:
    """A cage wall of vertical wire bars along x (at y) with horizontal rails at the
    given height fractions and a top bar, as one mesh."""
    parts = []
    n = int((x1 - x0) / pitch) + 1
    for k in range(n):
        x = x0 + k * pitch
        parts.append(polyline_tube(np.array([[x, y, z0], [x, y, z1]]), r, 6))
    for f in (*rails, 1.0):
        z = z0 + f * (z1 - z0)
        parts.append(polyline_tube(np.array([[x0, y, z], [x1, y, z]]), r * 1.6, 8))
    return merge(*parts)


def bottle_meshes(h: float, r: float) -> dict[str, MeshData]:
    """A hanging water bottle, base up at z = h (upside down, like the real ones):
    ``bottle`` (the clear shell), ``water`` (inside, ~70 % full), ``cap`` and the steel
    ``spout`` angled down out of the cap, with a ball tip."""
    pz = np.array([0.0, 0.0, 0.03, 0.10, 0.16, 0.9, 0.97, 1.0, 1.0]) * h
    pr = np.array([0.0, 0.35, 0.38, 0.8, 1.0, 1.0, 0.96, 0.86, 0.0]) * r
    bottle = lathe(pr, pz, 32)
    wz = np.array([0.03, 0.03, 0.10, 0.16, 0.66, 0.66]) * h
    wr = np.array([0.0, 0.34, 0.76, 0.95, 0.95, 0.0]) * r
    water = lathe(wr, wz, 28)
    cz = np.array([-0.12, -0.12, 0.02, 0.02]) * h
    cr = np.array([0.0, 0.45, 0.45, 0.0]) * r
    cap = lathe(cr, cz, 20)
    tip = np.array([0.0, 0.0, -0.12 * h])
    end = tip + np.array([0.0, -0.9 * r, -0.55 * h])
    spout = merge(polyline_tube(np.array([tip + [0, 0, 0.04 * h], tip, end]), 0.12 * r, 8),
                  sphere_mesh(0.16 * r, tuple(end), 10, 7))
    return {"bottle": bottle, "water": water, "cap": cap, "spout": spout}


def pellets_mesh(r_bowl: float, z0: float, n: int = 26, size: float = 0.22,
                 seed: int = 0) -> MeshData:
    """A heap of food pellets (little squashed spheres) filling a bowl of radius
    ``r_bowl`` whose fill level is at z0."""
    rng = np.random.default_rng(seed + 401)
    parts = []
    for _ in range(n):
        rr = r_bowl * 0.8 * math.sqrt(rng.uniform())
        a = rng.uniform(0, TAU)
        z = z0 + 0.5 * size * (1 - (rr / r_bowl) ** 2) * 2 + rng.uniform(0, 0.3 * size)
        parts.append(sphere_mesh(size * rng.uniform(0.7, 1.1), (rr * math.cos(a), rr * math.sin(a), z),
                                 8, 6, squash=rng.uniform(0.5, 0.8)))
    return merge(*parts)


def seeds_mesh(pts: np.ndarray, size: float = 0.28, seed: int = 0) -> MeshData:
    """Sunflower seeds (pointed, flattened ellipsoids) lying at the (x, y) points."""
    rng = np.random.default_rng(seed + 402)
    parts = []
    for x, y in pts:
        md = sphere_mesh(size, (0, 0, 0), 10, 7, squash=0.35)
        v = md.verts.copy()
        v[:, 1] *= 0.5
        a = rng.uniform(0, TAU)
        R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        v = v @ R.T + np.array([x, y, 0.35 * size * 0.35 + 0.03])
        parts.append(MeshData(v, md.faces, md.uv))
    return merge(*parts)


def hideout_meshes(w: float, d: float, h: float) -> dict[str, MeshData]:
    """A little wooden house on z = 0 (front face toward -y): ``walls`` (a closed box,
    uv: u along the width, v up) and a pitched ``roof``."""
    from fly_simulator.jobs.bowling_assets import slab_mesh

    walls = slab_mesh(-w / 2, w / 2, -d / 2, d / 2, 0.0, h)
    # the front face (y = -d/2) should show the whole door texture: planar uv x, z
    v = walls.verts
    walls.uv = np.column_stack([(v[:, 0] + w / 2) / w, 1.0 - v[:, 2] / h])
    ov = 0.25
    rt = 0.12
    pts = []
    top = np.array([0.0, 0.0, h + (w / 2) * math.tan(math.radians(35))])
    ca, sa = math.cos(math.radians(35)), math.sin(math.radians(35))
    L = (w / 2 + ov) / ca
    for s in (-1, 1):  # two roof boards (thin slabs) sloping down from the ridge
        board = slab_mesh(0.0, L, -d / 2 - ov, d / 2 + ov, 0.0, rt)
        M = np.column_stack([[s * ca, 0.0, -sa], [0.0, 1.0, 0.0], [s * sa, 0.0, ca]])
        faces = board.faces if s > 0 else board.faces[:, ::-1].copy()  # (mirror for s < 0)
        pts.append(MeshData(board.verts @ M.T + top, faces, board.uv))
    gable = []
    for yy in (-d / 2, d / 2):  # the gables: triangles closing the roof space
        apex = h + (w / 2) * math.tan(math.radians(35))
        rings = np.array([[[-w / 2, yy - 0.02, h], [w / 2, yy - 0.02, h], [0, yy - 0.02, apex]],
                          [[-w / 2, yy + 0.02, h], [w / 2, yy + 0.02, h], [0, yy + 0.02, apex]]])
        from fly_simulator.jobs.kebab_assets import tube

        gable.append(tube(rings, np.zeros(rings.shape[:2] + (2,))))
    return {"walls": merge(walls, *gable), "roof": merge(*pts)}


# ---------------------------------------------------------------------------
# textures (uint8 RGB, row 0 = v 0)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def bedding_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """Pet-cage bedding: a deep litter of pale curled wood shavings (seamless-ish)."""
    rng = np.random.default_rng(seed + 411)
    g = fbm(n, n, 8, 8, rng, 4)
    img = _to_u8(_mix(np.array([0.55, 0.42, 0.26], np.float32), np.array([0.70, 0.56, 0.36], np.float32), g))
    for _ in range(1500):
        x, y = (int(v) for v in rng.integers(-10, n + 10, 2))
        ax, ay = int(rng.integers(6, 16)), int(rng.integers(2, 6))
        ang = float(rng.uniform(0, 180))
        v = rng.uniform(0.8, 1.15)
        col = tuple(int(min(255, c * v)) for c in (236, 206, 156))
        a0 = float(rng.uniform(0, 120))
        cv2.ellipse(img, (x, y), (ax, ay), ang, a0, a0 + rng.uniform(150, 260), col,
                    int(rng.integers(2, 4)), cv2.LINE_AA)
        # a darker edge on some curls (the shading of the curl)
        if rng.uniform() < 0.4:
            cv2.ellipse(img, (x, y + 1), (ax, ay), ang, a0, a0 + 90, (150, 118, 76), 1, cv2.LINE_AA)
    return img


@lru_cache(maxsize=2)
def wheel_disc_texture(n: int = 512, spokes: int = 6, colour=(0.10, 0.62, 0.70)) -> np.ndarray:
    """The wheel's back disc (planar uv over the disc): glossy plastic with vent
    slots between the spokes, concentric ribs and a paw-print badge on the hub."""
    c = np.asarray(colour, np.float32)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    x, y = (xx - n / 2) / (n / 2), (yy - n / 2) / (n / 2)
    r, a = np.hypot(x, y), np.arctan2(y, x)
    img = np.broadcast_to(c, (n, n, 3)).copy()
    # slots: between the spokes, between radius 0.3 and 0.88
    ph = (a * spokes / TAU) % 1.0
    slot = (np.abs(ph - 0.5) < 0.36) & (r > 0.3) & (r < 0.88)
    slot = cv2.GaussianBlur(slot.astype(np.float32), (0, 0), 1.2)
    img = _mix(img, c * 0.45, slot * 0.8)
    for rr in (0.3, 0.6, 0.88, 0.97):  # ribs
        ring = np.exp(-((r - rr) / 0.008) ** 2)
        img = _mix(img, np.minimum(c * 1.35 + 0.1, 1.0), ring * 0.8)
    hub = r < 0.2
    img[hub] = (0.95, 0.95, 0.95)
    u8 = _to_u8(img)
    cc = n // 2
    s = n / 512
    pad = (60, 60, 64)
    cv2.circle(u8, (cc, cc + int(12 * s)), int(22 * s), pad, -1, cv2.LINE_AA)
    for k, (dx, dy) in enumerate(((-26, -18), (-10, -32), (10, -32), (26, -18))):
        cv2.circle(u8, (cc + int(dx * s), cc + int(dy * s)), int(9 * s), pad, -1, cv2.LINE_AA)
    return u8


@lru_cache(maxsize=2)
def plank_texture(seed: int = 0, h: int = 256, w: int = 256) -> np.ndarray:
    """The hideout: pale birch planks (horizontal), a round dark doorway in the
    middle of the lower half (the front face maps the whole image)."""
    rng = np.random.default_rng(seed + 412)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) / h
    t = fbm(h, w, 6, 2, rng, 3)
    grain = 0.5 + 0.5 * np.sin(TAU * (40 * xx / (w / h) + 2.0 * t))
    img = _mix(np.broadcast_to(np.array((0.86, 0.70, 0.46), np.float32), (h, w, 3)).copy(),
               (0.72, 0.54, 0.32), _smooth(0.4, 1.0, grain) * 0.4)
    seam = (np.abs(yy * 5 - np.round(yy * 5)) < 0.03).astype(np.float32)
    img = _mix(img, (0.45, 0.32, 0.18), seam)
    u8 = _to_u8(img)
    cv2.ellipse(u8, (w // 2, int(h * 0.98)), (int(w * 0.22), int(h * 0.42)), 0, 180, 360,
                (30, 20, 12), -1, cv2.LINE_AA)
    return u8


@lru_cache(maxsize=2)
def seed_texture(n: int = 64) -> np.ndarray:
    """Sunflower seed shell: black with grey-white stripes (lathe uv: u around)."""
    u = np.linspace(0, 1, n, endpoint=False, dtype=np.float32)[None, :] * np.ones((n, 1), np.float32)
    stripe = _smooth(0.55, 0.7, 0.5 + 0.5 * np.cos(TAU * 6 * u))
    v = 0.12 + 0.7 * stripe
    return _to_u8(np.stack([v, v, v * 0.95], -1))


@lru_cache(maxsize=2)
def room_texture(seed: int = 0, h: int = 512, w: int = 1024) -> np.ndarray:
    """Soft-focus room behind the cage: a warm painted wall, a bright window with a
    plant on the sill, a shelf with books and a framed picture; blurred like a
    shallow depth of field."""
    rng = np.random.default_rng(seed + 413)
    g = fbm(h, w, 3, 6, rng, 3)
    img = _to_u8(_mix(np.array([0.78, 0.68, 0.55], np.float32), np.array([0.86, 0.76, 0.62], np.float32), g))
    # a window with the day outside
    wx0, wx1, wy0, wy1 = int(w * 0.12), int(w * 0.40), int(h * 0.08), int(h * 0.62)
    sky = np.linspace(0, 1, wy1 - wy0, dtype=np.float32)[:, None, None]
    img[wy0:wy1, wx0:wx1] = _to_u8(np.array([0.72, 0.86, 0.98], np.float32) * (1 - sky)
                                   + np.array([0.90, 0.96, 1.0], np.float32) * sky)
    cv2.rectangle(img, (wx0, wy0), (wx1, wy1), (250, 248, 240), 10)
    cv2.line(img, ((wx0 + wx1) // 2, wy0), ((wx0 + wx1) // 2, wy1), (250, 248, 240), 8)
    cv2.line(img, (wx0, (wy0 + wy1) // 2), (wx1, (wy0 + wy1) // 2), (250, 248, 240), 8)
    cv2.rectangle(img, (wx0 - 20, wy1), (wx1 + 20, wy1 + 14), (240, 236, 226), -1)
    px = (wx0 + wx1) // 2 + 60  # the plant pot on the sill
    cv2.rectangle(img, (px - 22, wy1 - 40), (px + 22, wy1), (190, 96, 60), -1)
    for _ in range(14):
        a = rng.uniform(-2.6, -0.5)
        L = rng.uniform(40, 90)
        cv2.line(img, (px, wy1 - 40), (int(px + L * math.cos(a)), int(wy1 - 40 + L * math.sin(a))),
                 (60, 130, 60), 8, cv2.LINE_AA)
    # a shelf with books
    sy = int(h * 0.36)
    cv2.rectangle(img, (int(w * 0.55), sy), (int(w * 0.92), sy + 12), (120, 80, 50), -1)
    x = int(w * 0.57)
    while x < int(w * 0.85):
        bw = int(rng.integers(14, 30))
        bh = int(rng.integers(50, 90))
        col = tuple(int(v) for v in rng.integers(60, 210, 3))
        cv2.rectangle(img, (x, sy - bh), (x + bw, sy), col, -1)
        x += bw + 3
    # a framed picture
    cv2.rectangle(img, (int(w * 0.62), int(h * 0.45)), (int(w * 0.80), int(h * 0.70)), (70, 50, 35), 10)
    cv2.rectangle(img, (int(w * 0.62) + 10, int(h * 0.45) + 10), (int(w * 0.80) - 10, int(h * 0.70) - 10),
                  (140, 170, 200), -1)
    # a skirting board and the floor
    img[int(h * 0.86):] = (150, 110, 75)
    img[int(h * 0.84):int(h * 0.86)] = (245, 240, 230)
    return cv2.GaussianBlur(img, (0, 0), 3.0)
