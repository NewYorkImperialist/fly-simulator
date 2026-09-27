"""Procedural meshes and textures for the leaf raking job (visual only).

Same approach as ``kebab_assets``: everything is generated in code (numpy + OpenCV)
and handed to MjSpec directly; nothing is written to disk. Nothing here collides or
has mass (callers add the meshes with ``contact_kwargs("visual")`` and ``mass=0``).

An autumn backyard:

* meshes: fallen leaves (maple, oak and elm outlines, thin extruded plates with a
  planar vein texture), the tree (a tapered, root-flared trunk, branches, and a
  canopy of leaf clusters in four autumn tones), the fan rake (a wooden handle, a
  ferrule, 15 spring-steel tines with bent tips and a cross wire), a garden shed,
  a board fence, pumpkins, autumn shrubs;
* textures: an autumn lawn with a few stray leaf bits, bare earth / mulch for the
  pile spot, a grey leaf-vein texture and a foliage texture (both tinted by the
  material colours), bark, painted shed boards with a window and door, the sky.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.broccoli_toss_assets import merge, sphere_mesh  # noqa: F401
from fly_simulator.jobs.kebab_assets import (  # noqa: F401  (re-exported helpers)
    MeshData, SurfaceNoise, _mix, _smooth, _to_u8, add_mesh, add_texture, add_textured_material,
    fbm, lathe, tile_noise, transform, tube,
)
from fly_simulator.jobs.mowing_assets import gable_roof_mesh, shingle_texture, shrub_mesh  # noqa: F401
from fly_simulator.jobs.pizza_chef_assets import extrude_mesh
from fly_simulator.jobs.sisyphus_assets import box_mesh  # noqa: F401
from fly_simulator.jobs.trampoline_assets import disc_mesh, lawn_texture, polyline_tube  # noqa: F401

TAU = 2.0 * math.pi


def _norm(v):
    return (v - v.min()) / max(float(v.max() - v.min()), 1e-6)


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def leaf_outline(kind: str, L: float, W: float, n: int = 72) -> np.ndarray:
    """(n, 2) outline of a leaf about the origin, tip toward +x, base (stem) at -x,
    ~2L long and ~2W wide. ``kind``: "maple" (5 pointed lobes), "oak" (rounded
    lobes), "elm" (ovate, serrated)."""
    th = np.linspace(0, TAU, n, endpoint=False)
    if kind == "maple":
        lobes = np.zeros_like(th)
        for a, h in ((0.0, 1.0), (1.15, 0.85), (-1.15, 0.85), (2.25, 0.5), (-2.25, 0.5)):
            d = np.angle(np.exp(1j * (th - a)))
            lobes = np.maximum(lobes, h * np.exp(-(d / 0.3) ** 2))
        serr = 0.06 * np.abs(np.sin(9 * th))
        r = 0.32 + 0.68 * lobes + serr * lobes
        r = r * (1 - 0.5 * np.exp(-((np.angle(np.exp(1j * (th - math.pi)))) / 0.25) ** 2))
        x, y = L * r * np.cos(th), W * 1.15 * r * np.sin(th)
    elif kind == "oak":
        x = L * np.cos(th)
        y = W * 0.7 * np.sin(th) * (1 + 0.28 * np.abs(np.sin(3.5 * (th + 0.2)))) * (1 - 0.15 * np.cos(th))
    else:  # elm
        x = L * np.cos(th)
        y = W * 0.8 * np.sin(th) * (1 - 0.25 * np.cos(th)) * (1 + 0.04 * np.sin(22 * th))
    return np.column_stack([x, y])


def leaf_mesh(kind: str, L: float, W: float, t: float = 0.02) -> MeshData:
    """A leaf plate (planar uv over the leaf's half length L: the vein texture)."""
    return extrude_mesh(leaf_outline(kind, L, W), -t / 2, t / 2, L)


def trunk_mesh(h: float, r: float, seed: int = 0) -> MeshData:
    """A tapered trunk on z = 0 with a root flare and bark ridges (lathe)."""
    rng = np.random.default_rng(seed + 601)
    z = np.linspace(0, h, 14)
    s = z / h
    pr = r * (1.0 - 0.45 * s + 0.9 * np.exp(-s / 0.06))
    noise = SurfaceNoise(rng, 0.06 * r, 10)
    pr = np.concatenate([[0.0], pr, [0.0]])
    pz = np.concatenate([[0.0], z, [h]])
    return lathe(pr, pz, 24, radial_noise=lambda th, zz: noise(th, 4 * zz / h),
                 v_coord=np.concatenate([[0.0], s, [1.0]]) * 3.0, u_repeat=2.0)


def branches_mesh(base, lengths, seed: int = 0, r0: float = 0.35) -> tuple[MeshData, list]:
    """Branches (polyline tubes) from ``base`` fanning up and out; returns the mesh
    and the branch tips (where the leaf clusters go)."""
    rng = np.random.default_rng(seed + 602)
    base = np.asarray(base, float)
    parts, tips = [], []
    n = len(lengths)
    for k, L in enumerate(lengths):
        a = TAU * k / n + rng.uniform(-0.3, 0.3)
        up = rng.uniform(0.5, 0.9)
        d = np.array([math.cos(a), 0.6 * math.sin(a), up])
        d /= np.linalg.norm(d)
        p0 = base + np.array([0, 0, rng.uniform(-0.8, 0.4)])
        p1 = p0 + d * L * 0.5 + np.array([0, 0, 0.2 * L])
        p2 = p0 + d * L + np.array([0, 0, 0.1 * L])
        parts.append(polyline_tube(np.array([p0, p1, p2]), r0 * rng.uniform(0.7, 1.0), 8))
        tips.append(p2)
        # a twig off each branch
        q = p1 + np.array([rng.uniform(-1, 1), rng.uniform(-0.5, 0.5), 0.6]) * 0.35 * L
        parts.append(polyline_tube(np.array([p1, q]), r0 * 0.45, 6))
        tips.append(q)
    return merge(*parts), tips


def canopy_meshes(centres: list, r: float, n_groups: int = 4, seed: int = 0) -> list[MeshData]:
    """Leaf clusters (squashed spheres) around the given centres, split into
    ``n_groups`` meshes (one per autumn tone)."""
    rng = np.random.default_rng(seed + 603)
    groups: list[list] = [[] for _ in range(n_groups)]
    k = 0
    for c in centres:
        for _ in range(3):
            p = np.asarray(c) + rng.normal(0, 0.45 * r, 3) * np.array([1.0, 0.6, 0.45])
            groups[k % n_groups].append(sphere_mesh(r * rng.uniform(0.55, 0.9), tuple(p), 14, 9,
                                                    squash=rng.uniform(0.6, 0.8)))
            k += 1
    return [merge(*g) for g in groups]


def fan_rake_meshes(f: float, W: float, n_tines: int = 15) -> dict[str, MeshData]:
    """The fan rake in the rake frame (x forward from the thorax, z up): the wooden
    ``handle`` from above the fly's head down to the ``ferrule`` at the fan's neck
    (f - 0.35, 0, 0.3), the ``tines`` fanning out to the comb's front face at x = f
    (width W) with bent tips, and a cross wire."""
    neck = np.array([f - 0.35, 0.0, 0.3])
    top = np.array([0.45, 0.0, 1.75])
    d = top - neck
    handle = polyline_tube(np.array([neck + 0.12 * d / np.linalg.norm(d), top]), 0.065, 10)
    knob = sphere_mesh(0.085, tuple(top), 10, 7)
    ferrule = polyline_tube(np.array([neck - 0.02 * d, neck + 0.14 * d / np.linalg.norm(d)]), 0.085, 10)
    tines = []
    for k in range(n_tines):
        s = -1 + 2 * k / (n_tines - 1)
        y = s * W / 2
        p0 = neck + np.array([0.02, s * 0.12, 0.0])
        p1 = np.array([f - 0.18, y * 0.93, 0.16])
        p2 = np.array([f - 0.03, y, 0.07])
        p3 = np.array([f + 0.02, y, 0.02])
        tines.append(polyline_tube(np.array([p0, p1, p2, p3]), 0.022, 5))
    wire_pts = np.array([[f - 0.2, s * W / 2 * 0.72, 0.18] for s in np.linspace(-1, 1, 9)])
    tines.append(polyline_tube(wire_pts, 0.018, 5))
    return {"handle": merge(handle, knob), "ferrule": ferrule, "tines": merge(*tines)}


def shed_meshes(w: float, d: float, h: float) -> dict[str, MeshData]:
    """A garden shed on z = 0 (front toward -y): ``walls`` (box: every face a
    whole-texture planar uv, so the boards / door / window texture shows on the
    front and the boards on the sides) and the ``roof``."""
    walls = box_mesh((w / 2, d / 2, h / 2), 1.0)
    v = walls.verts
    # front / back faces: u across x, v down; side faces: u across y
    side = np.abs(np.abs(v[:, 0]) - w / 2) < 1e-9
    top = np.abs(np.abs(v[:, 2]) - h / 2) < 1e-9
    u = np.where(side, 0.92 + v[:, 1] / d * 0.07, 0.5 + v[:, 0] / w)  # (boards only)
    u = np.where(top & ~side, 0.02, u)
    vv = 0.5 - v[:, 2] / h
    walls.uv = np.column_stack([u, vv])
    walls = transform(walls, np.eye(3), np.array([0, 0, h / 2]))
    roof = transform(gable_roof_mesh(w, d, 0.35 * w, 0.35), np.eye(3), np.array([0, 0, h]))
    return {"walls": walls, "roof": roof}


def pumpkin_mesh(r: float, seed: int = 0) -> MeshData:
    """A ribbed pumpkin on z = 0 (lathe with 10 ribs)."""
    ph = np.linspace(-math.pi / 2, math.pi / 2, 14)
    pr = r * np.cos(ph)
    pz = r * 0.8 * (np.sin(ph) + 1)
    return lathe(pr, pz, 40, radial_noise=lambda th, z: -0.1 * r * np.abs(np.sin(5 * th)) ** 0.5
                 * (np.abs(z - 0.8 * r) < 0.75 * r))


def stalk_mesh(r: float) -> MeshData:
    return polyline_tube(np.array([[0, 0, 1.5 * r], [0.05 * r, 0, 1.75 * r], [0.2 * r, 0, 1.85 * r]]),
                         0.1 * r, 6)


def board_fence_mesh(x0: float, x1: float, h: float, bw: float = 1.0, t: float = 0.12) -> MeshData:
    """A wooden board fence along x on y = 0 (flat-topped boards with a small gap,
    uv: u across each board, v up), with two back rails."""
    parts = []
    n = int((x1 - x0) / bw)
    for k in range(n):
        x = x0 + k * bw + bw / 2
        hh = h * (1.0 + 0.03 * math.sin(k * 1.7))
        b = box_mesh((bw / 2 - 0.04, t / 2, hh / 2), 1.0)
        v = b.verts
        b.uv = np.column_stack([0.5 + v[:, 0] / bw + (k % 5) * 0.2, 0.5 - v[:, 2] / hh])
        parts.append(transform(b, np.eye(3), np.array([x, 0.0, hh / 2])))
    for zr in (0.2 * h, 0.8 * h):
        parts.append(transform(box_mesh(((x1 - x0) / 2, 0.08, 0.12), 0.3), np.eye(3),
                               np.array([(x0 + x1) / 2, t / 2 + 0.08, zr])))
    return merge(*parts)


# ---------------------------------------------------------------------------
# textures (uint8 RGB, row 0 = v 0)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def autumn_lawn_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """An autumn lawn: greens going yellow in patches, blade streaks, a few stray
    leaf bits."""
    rng = np.random.default_rng(seed + 611)
    f = _norm(fbm(n, n, 6, 6, rng, 5))
    g = _norm(fbm(n, n, 3, 3, rng, 3))
    base = _mix(np.array([0.22, 0.38, 0.13], np.float32), np.array([0.36, 0.50, 0.18], np.float32), f)
    base = _mix(base, (0.56, 0.54, 0.24), _smooth(0.6, 0.85, g) * 0.6)
    img = _to_u8(base)
    for _ in range(2400):
        x, y = (int(v) for v in rng.integers(0, n, 2))
        L = int(rng.integers(4, 10))
        a = rng.normal(-math.pi / 2, 0.4)
        c = rng.uniform(0.6, 1.2)
        col = tuple(int(min(255, v * c)) for v in (86, 130, 50))
        cv2.line(img, (x, y), (int(x + L * math.cos(a)), int(y + L * math.sin(a))), col, 1, cv2.LINE_AA)
    cols = ((210, 70, 30), (230, 140, 30), (200, 160, 40), (140, 70, 30))
    for _ in range(50):
        x, y = (int(v) for v in rng.integers(0, n, 2))
        cv2.ellipse(img, (x, y), (int(rng.integers(3, 6)), int(rng.integers(2, 4))),
                    float(rng.uniform(0, 180)), 0, 360, cols[int(rng.integers(0, 4))], -1, cv2.LINE_AA)
    return img


@lru_cache(maxsize=2)
def earth_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """The pile spot (planar uv over the disc): raked bare earth with a ring of
    scratch marks and crumbs."""
    rng = np.random.default_rng(seed + 612)
    f = _norm(fbm(n, n, 8, 8, rng, 4))
    img = _mix(np.array([0.32, 0.22, 0.13], np.float32), np.array([0.46, 0.33, 0.20], np.float32), f)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(xx - n / 2, yy - n / 2) / (n / 2)
    a = np.arctan2(yy - n / 2, xx - n / 2)
    scratch = (0.5 + 0.5 * np.sin(a * 60 + 3 * f)) * _smooth(0.4, 0.9, r)
    img = _mix(img, (0.25, 0.17, 0.10), scratch * 0.35)
    edge = _smooth(0.85, 1.0, r)  # fades into grass at the rim
    img = _mix(img, (0.30, 0.40, 0.16), edge)
    return _to_u8(img)


@lru_cache(maxsize=2)
def leaf_vein_texture(n: int = 128) -> np.ndarray:
    """Grey leaf detail (planar uv, leaf centre at the image centre, tip toward +u):
    a midrib and side veins radiating from the base, a paler centre, darker rim,
    tinted by the leaf material's colour."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    x, y = (xx - n / 2) / (n / 2), (yy - n / 2) / (n / 2)
    r = np.hypot(x, y)
    base = 0.95 - 0.25 * np.clip(r, 0, 1) ** 2
    img = np.repeat(base[..., None], 3, axis=2)
    u8 = _to_u8(img)
    bx = int(n * 0.2)  # the leaf base (stem end) at -x
    c = n // 2
    vein = (175, 170, 150)
    cv2.line(u8, (bx, c), (n - 2, c), vein, 2, cv2.LINE_AA)
    for a in (0.6, 1.1, 1.7, 2.3):
        for s in (-1, 1):
            L = n * (0.55 - 0.1 * a)
            cv2.line(u8, (bx + int(0.2 * n * (a - 0.6)), c),
                     (int(bx + L * math.cos(a * 0.5)), int(c + s * L * math.sin(a * 0.55))), vein, 1,
                     cv2.LINE_AA)
    return u8


@lru_cache(maxsize=2)
def foliage_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Dense foliage for the canopy clusters (grey, tinted by the cluster colour):
    overlapping bright leaves with dark gaps between them."""
    rng = np.random.default_rng(seed + 613)
    img = np.full((n, n, 3), 70, np.uint8)
    for _ in range(1300):
        x, y = (int(v) for v in rng.integers(0, n, 2))
        v = int(rng.integers(150, 256))
        a = float(rng.uniform(0, 180))
        cv2.ellipse(img, (x, y), (int(rng.integers(5, 10)), int(rng.integers(3, 6))), a, 0, 360,
                    (v, v, v), -1, cv2.LINE_AA)
        cv2.ellipse(img, (x, y), (int(rng.integers(5, 10)), 1), a, 0, 360, (v - 40, v - 40, v - 40), 1,
                    cv2.LINE_AA)
    return img


@lru_cache(maxsize=2)
def bark_texture(seed: int = 0, n: int = 128) -> np.ndarray:
    """Brown furrowed bark (lathe uv: u around, v up)."""
    rng = np.random.default_rng(seed + 614)
    g = fbm(n, n, 4, 12, rng, 4)
    x = np.linspace(0, 1, n, dtype=np.float32)[None, :] * np.ones((n, 1), np.float32)
    ridges = 0.5 + 0.5 * np.sin(TAU * (8 * x + 2.5 * g))
    v = 0.22 + 0.16 * ridges
    return _to_u8(np.stack([v * 1.35, v * 1.0, v * 0.72], -1))


@lru_cache(maxsize=2)
def shed_texture(h: int = 256, w: int = 256, paint=(0.42, 0.56, 0.44)) -> np.ndarray:
    """Painted vertical shed boards with white trim, a door (left of centre, a
    Z brace) and a small window (right), row 0 = the top."""
    x = np.arange(w, dtype=np.float32)[None, :]
    board = (x % (w / 12)) / (w / 12)
    base = np.asarray(paint, np.float32) * (0.9 + 0.1 * board)[..., None] * np.ones((h, 1, 1), np.float32)
    gap = (board > 0.94).astype(np.float32)[..., None] * np.ones((h, 1, 1), np.float32)
    img = _to_u8(_mix(base, (0.18, 0.22, 0.18), gap[..., 0] * 0.8))
    white = (238, 236, 228)
    dx0, dx1, dy0 = int(w * 0.18), int(w * 0.50), int(h * 0.28)
    cv2.rectangle(img, (dx0, dy0), (dx1, h - 1), white, 6)
    cv2.line(img, (dx0, dy0 + 10), (dx1, h - 10), white, 6)
    cv2.line(img, (dx0, (dy0 + h) // 2), (dx1, (dy0 + h) // 2), white, 6)
    cv2.circle(img, (dx1 - 14, (dy0 + h) // 2 + 18), 5, (60, 60, 60), -1, cv2.LINE_AA)
    wx0, wx1, wy0, wy1 = int(w * 0.62), int(w * 0.86), int(h * 0.3), int(h * 0.55)
    cv2.rectangle(img, (wx0, wy0), (wx1, wy1), (60, 76, 90), -1)
    cv2.rectangle(img, (wx0, wy0), (wx1, wy1), white, 6)
    cv2.line(img, ((wx0 + wx1) // 2, wy0), ((wx0 + wx1) // 2, wy1), white, 4)
    cv2.line(img, (wx0, (wy0 + wy1) // 2), (wx1, (wy0 + wy1) // 2), white, 4)
    return img


@lru_cache(maxsize=2)
def fence_wood_texture(seed: int = 0, h: int = 256, w: int = 128) -> np.ndarray:
    """Weathered cedar fence boards (vertical grain)."""
    rng = np.random.default_rng(seed + 615)
    t = fbm(h, w, 8, 2, rng, 3)
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    grain = 0.5 + 0.5 * np.sin(TAU * (7 * xx + 2.0 * t))
    img = _mix(np.broadcast_to(np.array((0.55, 0.38, 0.24), np.float32), (h, w, 3)).copy(),
               (0.42, 0.28, 0.17), _smooth(0.3, 1.0, grain) * 0.6)
    return _to_u8(img)


@lru_cache(maxsize=2)
def autumn_sky_texture(seed: int = 0, h: int = 256, w: int = 1024) -> np.ndarray:
    """A clear autumn afternoon sky over a far treeline in orange, gold and green."""
    rng = np.random.default_rng(seed + 616)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = np.array([0.40, 0.62, 0.90], np.float32) * (1 - t) + np.array([0.86, 0.88, 0.86], np.float32) * t
    img = np.broadcast_to(img, (h, w, 3)).copy()
    u8 = _to_u8(img)
    for _ in range(5):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(h // 10, h // 3))
        for _k in range(6):
            cv2.circle(u8, (cx + int(rng.integers(-40, 40)), cy + int(rng.integers(-8, 8))),
                       int(rng.integers(10, 22)), (246, 247, 250), -1, cv2.LINE_AA)
    u8 = cv2.GaussianBlur(u8, (0, 0), 2.0)
    cols = ((200, 110, 40), (215, 160, 50), (170, 70, 35), (90, 120, 50), (185, 130, 45))
    base = int(h * 0.8)
    haze = np.array([200, 205, 205], np.float32)
    for _ in range(420):  # the far treeline (hazy)
        cx = int(rng.integers(-20, w + 20))
        r = int(rng.integers(6, 15))
        cy = base - int(rng.integers(0, 34))
        col = 0.6 * np.array(cols[int(rng.integers(0, len(cols)))], np.float32) + 0.4 * haze
        cv2.circle(u8, (cx, cy), r, tuple(int(v) for v in col), -1, cv2.LINE_AA)
    u8[base:] = (96, 112, 70)
    return cv2.GaussianBlur(u8, (0, 0), 2.5)
