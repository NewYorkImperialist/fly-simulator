"""Procedural meshes and textures for the pizza chef job (visual only).

Everything is generated in code (numpy + OpenCV) and handed to MjSpec directly, as in
``kebab_assets`` / ``taste_tester_assets`` (whose helpers are reused); nothing is
written to disk. Nothing here collides or has mass. Text is drawn with OpenCV's
Hershey fonts; the pizzeria, the box print and the signs are generic (no brands,
no logos, no real names).

Pizza parts use a *planar* texture mapping (u, v from x, y over a fixed square of
half-size ``PIZZA_TEX_R``), so the dough stages, the eight slice wedges and the
layers all line up with one another and the texture stays continuous across the
wedge seams.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.kebab_assets import (MeshData, _mix, _smooth, _to_u8, fbm, lathe,
                                             tile_noise, tube)
from fly_simulator.jobs.taste_tester_assets import _bgr, _centered_text

TAU = 2.0 * math.pi
PIZZA_TEX_R = 0.8  # planar texture half-size (mm) of every pizza part


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def planar_uv(md: MeshData, half: float = PIZZA_TEX_R) -> MeshData:
    """Replace the uv with a top-down planar mapping over [-half, half]^2."""
    uv = np.column_stack([0.5 + md.verts[:, 0] / (2 * half), 0.5 + md.verts[:, 1] / (2 * half)])
    return MeshData(md.verts, md.faces, uv)


def dough_profile(r: float, h: float, p: float, n: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Superellipse cross-section (rho / r)^p + (z / h)^p = 1, bottom centre -> rim ->
    top centre (p = 2: a squashed ball, larger p: a flat disc with a rounded edge)."""
    ph = np.linspace(0.0, math.pi / 2, n)
    rho = r * np.cos(ph) ** (2.0 / p)
    z = h * np.sin(ph) ** (2.0 / p)
    rho[-1], z[0] = 0.0, 0.0
    return np.r_[0.0, rho], np.r_[0.0, z]


def dough_top(r: float, h: float, p: float, rho: float) -> float:
    """Top height of a dough stage at radius ``rho``."""
    x = min(max(rho / max(r, 1e-6), 0.0), 1.0)
    return float(h * max(1.0 - x ** p, 0.0) ** (1.0 / p))


def dough_mesh(r: float, h: float, p: float, lump: float = 0.0, seed: int = 0) -> MeshData:
    """A dough stage (base at z = 0): a lumpy ball (p ~ 2) ... a flat disc (p ~ 5)."""
    rr, zz = dough_profile(r, h, p)
    rng = np.random.default_rng(seed)
    k = rng.integers(2, 6, 3)
    ph = rng.uniform(0, TAU, 3)

    def noise(th, z):
        return lump * r * sum(np.sin(k[i] * th + ph[i] + 3.0 * z) for i in range(3)) / 3.0
    return planar_uv(lathe(rr, zz, 40, radial_noise=noise if lump > 0 else None))


def base_profile(r: float, h: float, rim_w: float, rim_h: float, rho: np.ndarray) -> np.ndarray:
    """Top height of the stretched pizza base: flat at ``h`` with a raised, rounded
    crust (cornicione) over the outer ``rim_w``, falling to a rounded edge at r."""
    x = np.clip((rho - (r - rim_w)) / rim_w, 0.0, 1.0)
    bump = rim_h * np.sin(math.pi * np.clip(x * 0.85, 0, 1)) ** 0.7
    edge = np.sqrt(np.clip(1.0 - ((rho - (r - 0.35 * rim_w)) / (0.35 * rim_w)).clip(0, None) ** 2,
                           0.0, 1.0))
    return (h + bump) * np.where(rho > r - 0.35 * rim_w, edge, 1.0)


def wedge_mesh(a0: float, a1: float, r: float, top_fn, z0: float = 0.0, n_rho: int = 16,
               n_th: int = 8, rho0: float = 0.004) -> MeshData:
    """A pie wedge between angles a0 .. a1 (rad), bottom at ``z0``, top height
    ``top_fn(rho)`` (absolute z). Built as a tube along theta whose rings are the
    wedge's (rho, z) cross-section; the two flat cut faces are the caps."""
    rho = np.linspace(rho0, r, n_rho)
    top = np.maximum(np.asarray(top_fn(rho), float), z0 + 1e-3)
    # cross-section loop: along the bottom outward, then back along the top inward
    loop_r = np.r_[rho, rho[::-1]]
    loop_z = np.r_[np.full(n_rho, z0), top[::-1]]
    ths = np.linspace(a0, a1, n_th)
    rings = np.stack([np.stack([loop_r * math.cos(t), loop_r * math.sin(t), loop_z], -1)
                      for t in ths])
    uv = np.zeros(rings.shape[:2] + (2,))
    return planar_uv(tube(rings, uv))


def layer_disc_mesh(r: float, t: float, wobble: float = 0.0, seed: int = 0, n: int = 48) -> MeshData:
    """A thin disc (sauce spreading, the melted cheese layer) with a wavy edge."""
    rng = np.random.default_rng(seed)
    k = rng.integers(3, 9, 3)
    ph = rng.uniform(0, TAU, 3)
    rr = np.array([0.0, r, r, 0.0])
    zz = np.array([0.0, 0.0, t, t])

    def noise(th, z):
        return wobble * r * sum(np.sin(k[i] * th + ph[i]) for i in range(3)) / 3.0
    return planar_uv(lathe(rr, zz, n, radial_noise=noise if wobble > 0 else None))


def bowl_mesh(r: float, h: float, wall: float = 0.03, n: int = 36) -> MeshData:
    """A round bowl, base at z = 0, open at the top (a closed shell with a lip)."""
    rr = np.array([0.0, 0.55 * r, 0.8 * r, 0.97 * r, r, r - wall, 0.95 * r - wall,
                   0.75 * r - wall, 0.0])
    zz = np.array([0.0, 0.0, 0.18 * h, 0.6 * h, h, h, 0.62 * h, 0.2 * h + wall, wall])
    return lathe(rr, zz, n, v_coord=np.linspace(0, 1, len(rr)))


def jar_mesh(r: float, h: float, wall: float = 0.02, n: int = 32) -> MeshData:
    """A tall open glass jar with a rolled lip."""
    rr = np.array([0.0, 0.92 * r, r, r, 1.04 * r, r - wall, r - wall, 0.9 * r - wall, 0.0])
    zz = np.array([0.0, 0.0, 0.06 * h, 0.92 * h, h, h, 0.08 * h, wall, wall])
    return lathe(rr, zz, n, v_coord=np.linspace(0, 1, len(rr)))


def extrude_mesh(outline: np.ndarray, z0: float, z1: float, half: float) -> MeshData:
    """A flat plate from a star-shaped (x, y) outline, planar uv over ``half``."""
    n = len(outline)
    rings = np.stack([np.column_stack([outline, np.full(n, z)]) for z in (z0, z1)])
    return planar_uv(tube(rings, np.zeros(rings.shape[:2] + (2,))), half)


def leaf_mesh(L: float, W: float, t: float = 0.006, n: int = 28) -> MeshData:
    """A basil leaf along +x (pointed tip, rounded base), centred on the origin."""
    s = np.linspace(0, TAU, n, endpoint=False)
    x = 0.5 * L * np.cos(s)
    y = 0.5 * W * np.sin(s) * (1.0 - 0.35 * (np.cos(s) + 1) / 2) ** 1.2
    return extrude_mesh(np.column_stack([x, y]), -t / 2, t / 2, 0.5 * L)


def pepperoni_mesh(r: float, t: float) -> MeshData:
    """A slightly cupped pepperoni slice (base at z = 0)."""
    rr = np.array([0.0, 0.9 * r, r, 0.96 * r, 0.0])
    zz = np.array([0.0, 0.0, 0.4 * t, t, 0.8 * t])
    return planar_uv(lathe(rr, zz, 24), r)


def peel_blade_mesh(L: float, W: float, t: float) -> MeshData:
    """The pizza peel's paddle: a rounded blade along +x from the neck (x = 0) to
    the rounded front, top face at z = 0."""
    s = np.linspace(0, TAU, 48, endpoint=False)
    c, sn = np.cos(s), np.sin(s)
    x = 0.5 * L + 0.5 * L * np.sign(c) * np.abs(c) ** 0.6
    y = 0.5 * W * np.sign(sn) * np.abs(sn) ** 0.6
    neck = x < 0.18 * L
    y = np.where(neck, y * (0.45 + 0.55 * x / (0.18 * L)), y)
    return extrude_mesh(np.column_stack([x - 0.5 * L, y]), -t, 0.0, 0.5 * max(L, W))


def dome_shell_mesh(R: float, H: float, t: float, mouth_dir: float, mouth_half: float,
                    mouth_h: float, top_r: float = 0.14, n_th: int = 72, n_z: int = 22
                    ) -> MeshData:
    """A brick oven dome (base at z = 0) with an arched mouth facing ``mouth_dir``
    (rad, in the xy plane) and a flue hole on top: an outer and an inner surface of
    revolution joined by rims along every open edge, so the shell is closed (the
    compiler needs a closed mesh) and the inside is visible through the mouth.
    uv: u = theta / (2 pi) * 6, v = height."""
    th = np.linspace(0.0, TAU, n_th, endpoint=False)
    phs = np.linspace(0.0, math.acos(top_r / R), n_z)  # latitude of each ring (0 = base)

    def surf(rad, hh):
        rho = rad * np.cos(phs)
        z = hh * np.sin(phs)
        P = np.stack([rho[:, None] * np.cos(th)[None], rho[:, None] * np.sin(th)[None],
                      np.broadcast_to(z[:, None], (n_z, n_th))], -1)
        return P
    outer = surf(R, H)
    inner = surf(R - t, H - t)
    zc = H * np.sin(phs)  # ring heights (outer)
    dth = np.abs((th - mouth_dir + math.pi) % TAU - math.pi)

    def in_mouth(i, j):  # quad (ring i..i+1, theta j..j+1)
        zm = 0.5 * (zc[i] + zc[i + 1])
        a = 0.5 * (dth[j] + dth[(j + 1) % n_th])
        if a > mouth_half:
            return False
        # arched opening: an ellipse in (angle, height)
        return (a / mouth_half) ** 2 + (zm / mouth_h) ** 2 < 1.0 or zm < 0.55 * mouth_h
    keep = np.array([[not in_mouth(i, j) for j in range(n_th)] for i in range(n_z - 1)])
    no = n_z * n_th  # index offset of the inner surface
    verts = np.concatenate([outer.reshape(-1, 3), inner.reshape(-1, 3)])
    uvo = np.stack([np.broadcast_to(th[None] / TAU * 6, (n_z, n_th)),
                    np.broadcast_to((zc / H)[:, None], (n_z, n_th))], -1).reshape(-1, 2)
    uv = np.concatenate([uvo, uvo])
    idx = lambda i, j: i * n_th + (j % n_th)  # noqa: E731
    faces = []
    edge_count: dict = {}
    for i in range(n_z - 1):
        for j in range(n_th):
            if not keep[i, j]:
                continue
            a, b, c, d = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            # outer: counter-clockwise seen from outside
            faces += [(a, b, c), (a, c, d)]
            faces += [(no + a, no + c, no + b), (no + a, no + d, no + c)]
            for e in ((a, b), (b, c), (c, d), (d, a)):
                edge_count[e] = edge_count.get(e, 0) + 1
    # boundary edges of the outer surface: (u, v) with no (v, u) twin -> rim quads
    for (u, v), _ in list(edge_count.items()):
        if (v, u) in edge_count:
            continue
        faces += [(v, u, no + u), (v, no + u, no + v)]
    md = MeshData(verts, np.array(faces, dtype=np.int64), uv)
    if md.signed_volume() < 0:
        md.faces = md.faces[:, ::-1].copy()
    return md


# ---------------------------------------------------------------------------
# textures (uint8 RGB, row 0 = v 0)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def marble_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """White marble counter with soft grey veins (seamless)."""
    rng = np.random.default_rng(seed + 901)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    turb = fbm(n, n, 4, 4, rng, 5)
    v1 = np.abs(np.sin(TAU * (2 * x + 1 * y + 2.2 * turb)))
    v2 = np.abs(np.sin(TAU * (1 * x - 2 * y + 3.0 * fbm(n, n, 3, 3, rng, 4))))
    vein = np.clip(1 - v1 / 0.06, 0, 1) * 0.55 + np.clip(1 - v2 / 0.03, 0, 1) * 0.35
    cloud = fbm(n, n, 6, 6, rng, 3)
    base = (0.90 + 0.05 * (cloud - 0.5))[..., None] * np.array([0.97, 0.97, 0.99], np.float32)
    base = _mix(base, (0.55, 0.57, 0.62), np.clip(vein, 0, 1) * 0.8)
    return _to_u8(base)


@lru_cache(maxsize=2)
def board_texture(seed: int = 0, n: int = 384) -> np.ndarray:
    """A floury wooden prep board: long grain, a few planks, flour dust and a
    cleaner patch in the middle (where the dough is worked), fingerprints."""
    rng = np.random.default_rng(seed + 902)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    turb = fbm(n, n, 2, 8, rng, 4)
    grain = 0.5 + 0.5 * np.sin(TAU * (26 * y + 3.0 * turb))
    plank = (np.floor(y * 4) % 2)
    wood = _mix(np.broadcast_to(np.array((0.74, 0.54, 0.32), np.float32), (n, n, 3)).copy(),
                (0.60, 0.40, 0.22), _smooth(0.4, 1.0, grain) * 0.6)
    wood = wood * (0.94 + 0.06 * plank)[..., None]
    seam = np.abs((y * 4) - np.round(y * 4)) < 0.004
    wood = _mix(wood, (0.35, 0.22, 0.12), seam.astype(np.float32))
    flour = fbm(n, n, 8, 8, rng, 4)
    r = np.hypot(x - 0.5, y - 0.5)
    dust = np.clip((flour - 0.42) * 2.2, 0, 1) * (0.35 + 0.65 * _smooth(0.12, 0.34, r))
    speck = (rng.random((n, n)) > 0.992).astype(np.float32)
    dust = np.clip(dust + cv2.GaussianBlur(speck, (5, 5), 1.2) * 1.5, 0, 1)
    img = _mix(wood, (0.97, 0.96, 0.93), dust * 0.85)
    u8 = _to_u8(img)
    for _ in range(4):  # little fly footprints in the flour
        cx, cy = rng.uniform(0.15, 0.85, 2) * n
        for k in range(3):
            cv2.ellipse(u8, (int(cx + 9 * k), int(cy + 4 * (k % 2))), (3, 2), 0, 0, 360,
                        (150, 110, 70), -1, cv2.LINE_AA)
    return u8


@lru_cache(maxsize=2)
def dough_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Raw dough: cream with fine bubbles and flour dust."""
    rng = np.random.default_rng(seed + 903)
    g = fbm(n, n, 8, 8, rng, 4)
    base = (0.93 + 0.06 * (g - 0.5))[..., None] * np.array([0.98, 0.88, 0.68], np.float32)
    bub = (rng.random((n, n)) > 0.985).astype(np.float32)
    bub = cv2.GaussianBlur(bub, (5, 5), 1.0)
    base = _mix(base, (0.80, 0.74, 0.62), np.clip(bub * 2, 0, 1) * 0.5)
    flour = np.clip((fbm(n, n, 5, 5, rng, 3) - 0.55) * 3, 0, 1)
    base = _mix(base, (1.0, 0.99, 0.95), flour * 0.45)
    return _to_u8(base)


@lru_cache(maxsize=2)
def crust_texture(seed: int = 0, n: int = 256, r_frac: float = 0.75 / PIZZA_TEX_R) -> np.ndarray:
    """Baked base (planar): golden, browning toward the rim, with leopard-spot char
    blisters on the crust."""
    rng = np.random.default_rng(seed + 904)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    r = np.hypot(x - 0.5, y - 0.5) * 2 / max(r_frac, 1e-6)  # 1 at the pizza edge
    g = fbm(n, n, 8, 8, rng, 4)
    gold = np.array((0.93, 0.74, 0.42), np.float32)
    brown = np.array((0.72, 0.45, 0.20), np.float32)
    base = _mix(np.broadcast_to(gold, (n, n, 3)).copy(), brown, np.clip(_smooth(0.78, 1.0, r) + 0.3 * (g - 0.5), 0, 1))
    spots = np.zeros((n, n), np.float32)
    for _ in range(70):
        a = rng.uniform(0, TAU)
        rr = rng.uniform(0.82, 0.99) * r_frac / 2
        cx, cy = (0.5 + rr * math.cos(a)) * n, (0.5 + rr * math.sin(a)) * n
        cv2.circle(spots, (int(cx), int(cy)), int(rng.integers(1, 4)), 1.0, -1, cv2.LINE_AA)
    spots = cv2.GaussianBlur(spots, (5, 5), 1.0)
    base = _mix(base, (0.22, 0.12, 0.06), np.clip(spots, 0, 1) * 0.85)
    bub = (rng.random((n, n)) > 0.99).astype(np.float32)
    base = _mix(base, (0.98, 0.88, 0.6), cv2.GaussianBlur(bub, (5, 5), 1.0) * 0.8)
    return _to_u8(base)


@lru_cache(maxsize=2)
def sauce_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Tomato sauce: a ladle spiral (lighter / darker bands), oregano flecks."""
    rng = np.random.default_rng(seed + 905)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n - 0.5
    r = np.hypot(x, y)
    a = np.arctan2(y, x)
    turb = fbm(n, n, 6, 6, rng, 3)
    spiral = 0.5 + 0.5 * np.sin(TAU * 9 * r - a + 2.0 * turb)
    base = _mix(np.broadcast_to(np.array((0.78, 0.12, 0.06), np.float32), (n, n, 3)).copy(),
                (0.58, 0.06, 0.03), _smooth(0.3, 0.9, spiral) * 0.8)
    fleck = (rng.random((n, n)) > 0.994).astype(np.float32)
    base = _mix(base, (0.18, 0.28, 0.08), np.clip(fleck * 3, 0, 1))
    return _to_u8(base)


@lru_cache(maxsize=2)
def cheese_melt_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Melted mozzarella: creamy white-yellow pools with golden-brown blisters and
    gaps where the sauce shows through."""
    rng = np.random.default_rng(seed + 906)
    g = fbm(n, n, 7, 7, rng, 4)
    pools = _smooth(0.38, 0.5, g)
    cheese = np.broadcast_to(np.array((0.99, 0.93, 0.66), np.float32), (n, n, 3)).copy()
    img = _mix(np.broadcast_to(np.array((0.74, 0.14, 0.06), np.float32), (n, n, 3)).copy(),
               cheese, pools)
    bl = np.zeros((n, n), np.float32)
    for _ in range(45):
        cx, cy = rng.uniform(0, n, 2)
        cv2.circle(bl, (int(cx), int(cy)), int(rng.integers(2, 6)), 1.0, -1, cv2.LINE_AA)
    bl = cv2.GaussianBlur(bl, (7, 7), 2.0) * pools
    img = _mix(img, (0.80, 0.52, 0.18), np.clip(bl, 0, 1) * 0.8)
    return _to_u8(img)


@lru_cache(maxsize=2)
def pepperoni_texture(seed: int = 0, n: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed + 907)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n - 0.5
    r = np.hypot(x, y) * 2
    img = np.broadcast_to(np.array((0.72, 0.13, 0.10), np.float32), (n, n, 3)).copy()
    fat = (rng.random((n, n)) > 0.95).astype(np.float32)
    img = _mix(img, (0.95, 0.72, 0.62), cv2.GaussianBlur(fat, (3, 3), 0.7) * 1.2)
    img = _mix(img, (0.45, 0.07, 0.05), _smooth(0.8, 1.0, r))
    return _to_u8(img)


@lru_cache(maxsize=2)
def basil_texture(n: int = 64) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n - 0.5
    img = np.broadcast_to(np.array((0.16, 0.52, 0.16), np.float32), (n, n, 3)).copy()
    rib = np.clip(1 - np.abs(y) / 0.03, 0, 1)
    vein = np.clip(1 - np.abs(np.abs(y) - 0.6 * np.abs(x + 0.5) % 0.25) / 0.02, 0, 1) * 0.3
    img = _mix(img, (0.35, 0.72, 0.30), np.clip(rib + vein, 0, 1))
    return _to_u8(img)


@lru_cache(maxsize=2)
def brick_texture(seed: int = 0, h: int = 256, w: int = 512, soot: bool = True) -> np.ndarray:
    """Oven bricks: running bond, warm reds / oranges, mortar, soot toward the top."""
    rng = np.random.default_rng(seed + 908)
    rows, cols = 8, 12
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    ry = y / (h / rows)
    rx = x / (w / cols) + 0.5 * (np.floor(ry) % 2)
    fx, fy = rx - np.floor(rx), ry - np.floor(ry)
    edge = np.minimum(np.minimum(fx, 1 - fx) * 2.2, np.minimum(fy, 1 - fy))
    mortar = _smooth(0.07, 0.03, edge)
    cell = (np.floor(rx) * 7 + np.floor(ry) * 13) % 5
    tint = 0.85 + 0.05 * cell
    g = fbm(h, w, 8, 16, rng, 4)
    brick = (tint * (0.9 + 0.2 * (g - 0.5)))[..., None] * np.array([0.72, 0.30, 0.18], np.float32)
    img = _mix(brick, (0.78, 0.74, 0.68), mortar)
    if soot:
        v = y / h
        img = _mix(img, (0.10, 0.08, 0.07), np.clip(_smooth(0.55, 1.0, v) * (0.5 + 0.5 * g), 0, 1) * 0.6)
    return _to_u8(img)


@lru_cache(maxsize=2)
def stone_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """The oven's plinth: pale rough stone blocks."""
    rng = np.random.default_rng(seed + 909)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / (n / 3)
    rx = x + 0.5 * (np.floor(y) % 2)
    fx, fy = rx - np.floor(rx), y - np.floor(y)
    edge = np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy))
    g = fbm(n, n, 8, 8, rng, 5)
    base = (0.72 + 0.12 * (g - 0.5))[..., None] * np.array([0.93, 0.90, 0.84], np.float32)
    return _to_u8(_mix(base, (0.45, 0.43, 0.40), _smooth(0.05, 0.02, edge)))


@lru_cache(maxsize=2)
def wood_texture(seed: int = 0, n: int = 128) -> np.ndarray:
    """Pale beech (the peel, the cutter handle, the shelf)."""
    rng = np.random.default_rng(seed + 910)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    t = fbm(n, n, 2, 6, rng, 3)
    grain = 0.5 + 0.5 * np.sin(TAU * (30 * y + 2.5 * t))
    img = _mix(np.broadcast_to(np.array((0.84, 0.66, 0.42), np.float32), (n, n, 3)).copy(),
               (0.74, 0.55, 0.33), _smooth(0.3, 1.0, grain) * 0.35)
    return _to_u8(img)


@lru_cache(maxsize=2)
def log_texture(seed: int = 0, n: int = 128) -> np.ndarray:
    """Split firewood bark."""
    rng = np.random.default_rng(seed + 911)
    g = fbm(n, n, 4, 16, rng, 4)
    y = np.linspace(0, 1, n, dtype=np.float32)[:, None] * np.ones((1, n), np.float32)
    ridges = 0.5 + 0.5 * np.sin(TAU * (10 * y + 2.0 * g))
    v = 0.22 + 0.12 * ridges
    return _to_u8(np.stack([v * 1.25, v * 0.95, v * 0.7], -1))


@lru_cache(maxsize=2)
def mouth_glow_texture(n: int = 128) -> np.ndarray:
    """The hearth floor inside the oven: glowing embers and ash."""
    rng = np.random.default_rng(912)
    g = fbm(n, n, 8, 8, rng, 4)
    hot = _smooth(0.45, 0.75, g)
    img = _mix(np.broadcast_to(np.array((0.20, 0.10, 0.06), np.float32), (n, n, 3)).copy(),
               (1.0, 0.45, 0.08), hot)
    img = _mix(img, (1.0, 0.85, 0.35), _smooth(0.7, 0.9, g))
    return _to_u8(img)


@lru_cache(maxsize=2)
def chalkboard_texture(h: int = 384, w: int = 512) -> np.ndarray:
    """The "pizzas served" chalkboard: a wooden frame, chalk smudges, captions above
    the three 7-segment counters (the digits are separate chalk-white geoms)."""
    rng = np.random.default_rng(913)
    g = fbm(h, w, 6, 8, rng, 3)
    img = (0.13 + 0.05 * (g - 0.5))[..., None] * np.array([0.9, 1.08, 0.96], np.float32)
    smear = _smooth(0.55, 0.8, fbm(h, w, 3, 4, rng, 3))
    img = _mix(img, (0.45, 0.48, 0.46), smear * 0.25)
    u8 = _to_u8(img)
    chalk = (0.93, 0.93, 0.88)
    cv2.rectangle(u8, (0, 0), (w - 1, h - 1), _bgr((0.45, 0.30, 0.16)), 18)
    _centered_text(u8, "FLY PIZZERIA", int(h * 0.12), 1.35, chalk, 2, font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)
    _centered_text(u8, "wood fired - open forever", int(h * 0.22), 0.62, (0.95, 0.85, 0.45), 1)
    for cx, word in ((w * 0.19, "SERVED"), (w * 0.5, "PERFECT"), (w * 0.81, "TIPS $")):
        _centered_text(u8, word, int(h * 0.36), 0.72, chalk, 2, cx=int(cx))
    _centered_text(u8, "pizzas", int(h * 0.88), 0.55, chalk, 1, cx=int(w * 0.19))
    _centered_text(u8, "tosses", int(h * 0.88), 0.55, chalk, 1, cx=int(w * 0.5))
    _centered_text(u8, "for the chef", int(h * 0.88), 0.55, chalk, 1, cx=int(w * 0.81))
    return u8


@lru_cache(maxsize=2)
def menu_texture(h: int = 320, w: int = 256) -> np.ndarray:
    img = _to_u8(np.full((h, w, 3), (0.97, 0.94, 0.86), np.float32))
    cv2.rectangle(img, (5, 5), (w - 6, h - 6), _bgr((0.7, 0.1, 0.08)), 5)
    _centered_text(img, "MENU", 36, 1.1, (0.7, 0.1, 0.08), 2)
    rows = (("MARGHERITA", "3.50"), ("PEPPERONI", "4.00"), ("BASIL BOMB", "3.75"),
            ("FLY SPECIAL", "4.20"), ("THE ETERNAL", "free?"))
    for k, (a, b) in enumerate(rows):
        yy = 80 + k * 42
        cv2.putText(img, a, (16, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 30, 20), 1, cv2.LINE_AA)
        cv2.putText(img, b, (w - 70, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 30, 20), 1, cv2.LINE_AA)
        cv2.line(img, (16, yy + 10), (w - 16, yy + 10), (200, 190, 170), 1)
    _centered_text(img, "(prices in fly cents)", h - 22, 0.45, (0.3, 0.25, 0.2), 1)
    return img


@lru_cache(maxsize=2)
def neon_texture(h: int = 128, w: int = 384) -> np.ndarray:
    img = _to_u8(np.full((h, w, 3), (0.04, 0.03, 0.05), np.float32))
    glow = np.zeros((h, w, 3), np.uint8)
    _centered_text(glow, "PIZZA", h // 2, 2.4, (1.0, 0.25, 0.2), 9)
    glow = cv2.GaussianBlur(glow, (0, 0), 7)
    img = cv2.add(img, glow)
    _centered_text(img, "PIZZA", h // 2, 2.4, (1.0, 0.85, 0.8), 3)
    return img


@lru_cache(maxsize=2)
def box_lid_texture(h: int = 256, w: int = 256) -> np.ndarray:
    """A generic takeaway pizza box print: 'HOT & FRESH', a cartoon slice, 'THANK YOU'."""
    img = _to_u8(np.full((h, w, 3), (0.83, 0.70, 0.50), np.float32))
    rng = np.random.default_rng(914)
    fib = (fbm(h, w, 16, 16, rng, 2) * 20).astype(np.uint8)
    img = cv2.subtract(img, np.stack([fib] * 3, -1))
    red = (0.78, 0.12, 0.10)
    cv2.rectangle(img, (10, 10), (w - 11, h - 11), _bgr(red), 4)
    _centered_text(img, "HOT & FRESH", 44, 0.95, red, 2)
    tri = np.array([[w // 2, 80], [w // 2 - 50, 180], [w // 2 + 50, 180]], np.int32)
    cv2.fillPoly(img, [tri], _bgr((0.98, 0.85, 0.35)), cv2.LINE_AA)
    cv2.line(img, (w // 2 - 54, 184), (w // 2 + 54, 184), _bgr((0.75, 0.48, 0.2)), 9, cv2.LINE_AA)
    for dx, dy in ((-12, 130), (14, 150), (0, 110), (-20, 165), (22, 170)):
        cv2.circle(img, (w // 2 + dx, dy), 7, _bgr(red), -1, cv2.LINE_AA)
    _centered_text(img, "THANK YOU", h - 40, 0.8, red, 2)
    return img


@lru_cache(maxsize=2)
def cardboard_texture(n: int = 64) -> np.ndarray:
    rng = np.random.default_rng(915)
    g = tile_noise(n, n, 8, 8, rng)
    v = 0.78 + 0.06 * (g - 0.5)
    return _to_u8(np.stack([v * 1.0, v * 0.84, v * 0.6], -1))


@lru_cache(maxsize=8)
def label_texture(word: str, rgb: tuple = (0.97, 0.95, 0.88), ink: tuple = (0.15, 0.1, 0.08),
                  h: int = 64, w: int = 192) -> np.ndarray:
    img = _to_u8(np.full((h, w, 3), rgb, np.float32))
    cv2.rectangle(img, (3, 3), (w - 4, h - 4), _bgr(ink), 2)
    scale = 1.0 if len(word) <= 6 else 0.75
    _centered_text(img, word, h // 2, scale, ink, 2)
    return img
