"""Procedural meshes and textures for the dishwasher job (visual only).

Same approach as ``kebab_assets``: everything is generated in code (numpy + OpenCV)
and handed to MjSpec directly; nothing is written to disk and nothing here collides
or has mass. A warm home kitchen: a butcher-block counter, cream subway tiles, a
window over the sink (a garden outside, gingham curtains), a stainless sink with
soapy water, plates with food smears whose grime fades in stages, a sponge, a dish
rack. The dish soap brand ("SUDSY") is our own.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.broccoli_toss_assets import merge, panel_mesh, panel_quat, sphere_mesh, torus_mesh  # noqa: F401
from fly_simulator.jobs.jump_rope_assets import slab_mesh  # noqa: F401
from fly_simulator.jobs.kebab_assets import (  # noqa: F401  (re-exported helpers)
    TAU, MeshData, _mix, _smooth, _to_u8, add_mesh, add_texture, add_textured_material, fbm, lathe,
    steel_texture, tile_noise, transform, tube,
)
from fly_simulator.jobs.pizza_chef_assets import planar_uv
from fly_simulator.jobs.sisyphus_assets import box_mesh  # noqa: F401
from fly_simulator.jobs.trampoline_assets import _text, disc_mesh, polyline_tube  # noqa: F401

GRIME_LEVELS = 8  # material stages from dirty (0) to clean (GRIME_LEVELS - 1)
N_SMEARS = 3  # food-smear patterns (a plate keeps its pattern while it fades)


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def dinner_plate_mesh(R: float, h: float) -> MeshData:
    """A dinner plate about z (base at z = 0, total height ``h``): a foot ring, the
    flat well, a raised rim; planar top-down uv over the plate (texture = the face)."""
    r = np.array([0.0, 0.52, 0.56, 0.60, 0.64, 0.80, 0.95, 1.0, 0.99, 0.80, 0.64, 0.0]) * R
    z = np.array([0.18, 0.18, 0.0, 0.0, 0.2, 0.3, 0.8, 0.95, 1.0, 0.62, 0.45, 0.45]) * h
    return planar_uv(lathe(r, z, 40, v_coord=np.linspace(0, 1, len(r))), R)


def sponge_mesh(hx: float, hy: float, hz: float) -> MeshData:
    """A kitchen sponge: a rounded box (a superellipse loop extruded, slight bulge)."""
    n = 24
    ph = np.linspace(0, TAU, n, endpoint=False)
    c, s = np.cos(ph), np.sin(ph)
    lp = np.stack([np.sign(c) * np.abs(c) ** 0.35 * hx, np.sign(s) * np.abs(s) ** 0.35 * hy], 1)
    zs = np.array([-hz, -0.8 * hz, 0.8 * hz, hz])
    scale = np.array([0.93, 1.0, 1.0, 0.93])
    rings = np.stack([np.column_stack([lp * k, np.full(n, z)]) for z, k in zip(zs, scale)])
    uv = np.zeros(rings.shape[:2] + (2,))
    uv[..., 0] = 0.5 + rings[..., 0] / (2 * hx)
    uv[..., 1] = 0.5 + rings[..., 1] / (2 * hy)
    return tube(rings, uv)


def faucet_mesh(base, spout, h: float, r: float = 0.07) -> MeshData:
    """A gooseneck faucet: up from ``base`` (x, y) to ``h``, arcing over to the spout
    (x, y, z) pointing down."""
    b = np.array([base[0], base[1], 0.0])
    sp = np.asarray(spout, float)
    top = np.array([0.6 * b[0] + 0.4 * sp[0], 0.6 * b[1] + 0.4 * sp[1], h])
    pts = [b, b + np.array([0, 0, 0.6 * h])]
    for u in np.linspace(0.0, 1.0, 12):
        # a quadratic arc from the column top through ``top`` to above the spout
        p0 = b + np.array([0, 0, 0.6 * h])
        p2 = sp + np.array([0, 0, 0.15])
        p = (1 - u) ** 2 * p0 + 2 * u * (1 - u) * (top + np.array([0, 0, 0.3])) + u ** 2 * p2
        pts.append(p)
    pts.append(sp)
    pts = np.array(pts)
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-3:
            keep.append(i)
    return polyline_tube(pts[keep], r, 14)


def bottle_mesh(r: float, h: float) -> MeshData:
    """A squeeze bottle of dish soap (lathe), base at z = 0."""
    rr = np.array([0.0, 0.9 * r, r, r, 0.93 * r, 0.55 * r, 0.3 * r, 0.3 * r, 0.18 * r, 0.0])
    zz = np.array([0.0, 0.0, 0.05 * h, 0.72 * h, 0.8 * h, 0.9 * h, 0.93 * h, 0.98 * h, 1.0 * h, 1.0 * h])
    return lathe(rr, zz, 28, v_coord=np.array([0.0, 0.0, 0.05, 0.72, 0.8, 0.9, 0.93, 0.98, 1.0, 1.0]))


def pot_mesh(r: float, h: float) -> MeshData:
    """A terracotta plant pot (open top, closed shell)."""
    rr = np.array([0.0, 0.72 * r, 0.75 * r, r, 1.08 * r, 1.08 * r, 0.93 * r, 0.0])
    zz = np.array([0.0, 0.0, 0.02 * h, 0.8 * h, 0.82 * h, h, h, 0.85 * h])
    return lathe(rr, zz, 24)


def leaf_cluster_mesh(r: float, seed: int = 0, n: int = 9) -> MeshData:
    """A potted herb: a few squashed spheres (leaf clumps)."""
    rng = np.random.default_rng(seed + 5)
    parts = []
    for _ in range(n):
        c = rng.normal(0, 0.45 * r, 3)
        c[2] = abs(c[2]) + 0.3 * r
        parts.append(sphere_mesh(rng.uniform(0.35, 0.55) * r, c, 10, 7, squash=0.7))
    return merge(*parts)


def cloth_mesh(w: float, h: float, folds: int = 5, amp: float = 0.08, n: int = 40) -> MeshData:
    """A hanging curtain / towel: a thin sheet along x (width ``w``) hanging down z
    (height ``h``), wavy in y (folds); the front (-y) maps the texture upright."""
    xs = np.linspace(-w / 2, w / 2, n)
    zs = np.linspace(0, -h, 6)
    rings, uvs = [], []
    for k, z in enumerate(zs):
        y = amp * np.sin(xs / w * folds * TAU)
        front = np.column_stack([xs, y - 0.01, np.full(n, z)])
        back = np.column_stack([xs[::-1], (y + 0.01)[::-1], np.full(n, z)])
        rings.append(np.concatenate([front, back]))
        u = np.concatenate([(xs + w / 2) / w, ((xs + w / 2) / w)[::-1]])
        uvs.append(np.column_stack([u, np.full(2 * n, k / (len(zs) - 1))]))
    return tube(np.array(rings), np.array(uvs))


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def butcher_block_texture(seed: int = 0, h: int = 512, w: int = 1024, strips: int = 22) -> np.ndarray:
    """A warm oiled butcher-block counter: glued strips (along u) of varying tone,
    long grain, a few knife marks."""
    rng = np.random.default_rng(seed + 1501)
    img = np.zeros((h, w, 3), np.float32)
    edges = np.linspace(0, h, strips + 1).astype(int)
    grain = tile_noise(h, w, h // 2, 6, rng)
    fine = tile_noise(h, w, h, 40, rng)
    for i in range(strips):
        a, b = edges[i], edges[i + 1]
        tone = rng.uniform(-0.08, 0.08)
        base = np.array([0.72 + tone, 0.48 + 0.8 * tone, 0.27 + 0.5 * tone], np.float32)
        dark = base * 0.72
        g = 0.55 * grain[a:b] + 0.45 * fine[a:b]
        img[a:b] = _mix(np.broadcast_to(base, (b - a, w, 3)).copy(), dark, _smooth(0.35, 0.8, g))
        img[a:a + 1] *= 0.7
    u8 = _to_u8(img)
    for _ in range(40):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        ang = rng.uniform(-0.5, 0.5)
        L = rng.integers(10, 40)
        cv2.line(u8, (int(x0), int(y0)), (int(x0 + L * math.cos(ang)), int(y0 + L * math.sin(ang))),
                 (120, 80, 45), 1, cv2.LINE_AA)
    return u8


@lru_cache(maxsize=2)
def subway_tile_texture(seed: int = 0, h: int = 512, w: int = 1024, rows: int = 12, cols: int = 12) -> np.ndarray:
    """Cream glazed subway tiles (staggered) with grey grout and a soft sheen."""
    rng = np.random.default_rng(seed + 1502)
    img = np.full((h, w, 3), 0.62, np.float32)
    img[:] = (0.66, 0.62, 0.56)  # grout
    th, tw = h / rows, w / cols
    for r in range(rows):
        off = 0.5 * tw if r % 2 else 0.0
        for c in range(-1, cols + 1):
            x0, y0 = int(c * tw + off + 2), int(r * th + 2)
            x1, y1 = int((c + 1) * tw + off - 2), int((r + 1) * th - 2)
            x0c, x1c = max(x0, 0), min(x1, w)
            if x1c <= x0c:
                continue
            t = rng.uniform(-0.03, 0.03)
            img[y0:y1, x0c:x1c] = (0.95 + t, 0.91 + t, 0.80 + t)
            # glaze highlight along the top edge
            img[y0:y0 + 3, x0c:x1c] = np.minimum(img[y0:y0 + 3, x0c:x1c] + 0.04, 1.0)
    n = fbm(h, w, 8, 16, rng, 3)
    img *= (0.96 + 0.06 * n)[..., None]
    return _to_u8(img)


@lru_cache(maxsize=1)
def window_texture(h: int = 512, w: int = 640, seed: int = 0) -> np.ndarray:
    """The view out of the kitchen window: a late-afternoon sky, a green garden with
    a hedge, a tree, a fence and a shed; white frame with a cross (muntins)."""
    rng = np.random.default_rng(seed + 1503)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    sky = _mix(np.broadcast_to(np.array((0.55, 0.75, 0.95), np.float32), (h, w, 3)).copy(),
               (1.0, 0.86, 0.62), _smooth(0.1, 0.62, y) * np.ones((1, w), np.float32))
    img = sky
    cl = fbm(h, w, 3, 6, rng, 4)
    img = _mix(img, (1.0, 0.97, 0.92), _smooth(0.55, 0.8, cl) * (y < 0.45))
    u8 = _to_u8(img)
    hz = int(0.62 * h)
    # the garden: lawn, hedge, fence, a shed and a tree
    lawn = fbm(h - hz, w, 6, 12, rng, 3)
    g = _mix(np.broadcast_to(np.array((0.35, 0.58, 0.22), np.float32), (h - hz, w, 3)).copy(),
             (0.5, 0.72, 0.3), lawn)
    u8[hz:] = _to_u8(g)
    for x in range(0, w, 26):  # a picket fence
        cv2.rectangle(u8, (x, hz - 46), (x + 16, hz + 6), (236, 232, 222), -1)
        cv2.fillConvexPoly(u8, np.array([[x, hz - 46], [x + 8, hz - 56], [x + 16, hz - 46]]), (236, 232, 222))
    cv2.rectangle(u8, (0, hz - 30), (w, hz - 24), (220, 214, 202), -1)
    cv2.rectangle(u8, (int(0.62 * w), hz - 120), (int(0.86 * w), hz - 10), (150, 80, 55), -1)  # shed
    cv2.fillConvexPoly(u8, np.array([[int(0.60 * w), hz - 120], [int(0.74 * w), hz - 170],
                                     [int(0.88 * w), hz - 120]]), (90, 60, 50))
    cv2.rectangle(u8, (int(0.71 * w), hz - 80), (int(0.77 * w), hz - 10), (80, 45, 30), -1)
    cv2.rectangle(u8, (int(0.2 * w) - 8, hz - 150), (int(0.2 * w) + 8, hz), (95, 65, 40), -1)  # tree
    for _ in range(40):
        cx = int(0.2 * w + rng.normal(0, 45))
        cy = int(hz - 170 + rng.normal(0, 40))
        col = tuple(int(v) for v in rng.uniform([40, 100, 30], [80, 150, 60]))
        cv2.circle(u8, (cx, cy), int(rng.integers(18, 34)), col, -1, cv2.LINE_AA)
    for _ in range(25):  # flowers in the lawn
        cx, cy = int(rng.integers(0, w)), int(rng.integers(hz + 10, h))
        col = [(240, 90, 110), (250, 220, 90), (250, 250, 250)][int(rng.integers(0, 3))]
        cv2.circle(u8, (cx, cy), 3, col, -1)
    u8 = cv2.GaussianBlur(u8, (0, 0), 1.2)  # slightly out of focus
    # the frame and the muntins
    fw = 18
    cv2.rectangle(u8, (0, 0), (w - 1, h - 1), (245, 243, 236), fw * 2)
    cv2.rectangle(u8, (w // 2 - fw // 2, 0), (w // 2 + fw // 2, h), (245, 243, 236), -1)
    cv2.rectangle(u8, (0, h // 2 - fw // 2), (w, h // 2 + fw // 2), (245, 243, 236), -1)
    return u8


@lru_cache(maxsize=1)
def gingham_texture(n: int = 256, col=(0.85, 0.22, 0.2), k: int = 12) -> np.ndarray:
    """Red-and-white gingham (curtains, the dish towel)."""
    x = (np.arange(n) // (n // k)) % 2
    a = x[None, :].astype(np.float32)
    b = x[:, None].astype(np.float32)
    t = 0.5 * (a + b)
    img = _mix(np.ones((n, n, 3), np.float32), col, t)
    return _to_u8(img)


@lru_cache(maxsize=1)
def plate_clean_texture(n: int = 256) -> np.ndarray:
    """A clean glazed plate seen from above: white, a cobalt band on the rim, a thin
    gold line, a soft glaze gradient."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    c = (n - 1) / 2
    r = np.hypot(x - c, y - c) / c
    base = 0.97 - 0.05 * r
    img = np.stack([base * 0.99, base * 0.99, base], -1)
    band = (r > 0.80) & (r < 0.90)
    img[band] = (0.18, 0.3, 0.72)
    img[(r > 0.92) & (r < 0.94)] = (0.8, 0.65, 0.3)
    img[(r > 0.66) & (r < 0.675)] = (0.55, 0.62, 0.85)
    return _to_u8(img)


@lru_cache(maxsize=N_SMEARS)
def _smear_layer(pattern: int, n: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(colour (n, n, 3), coverage mask (n, n), fade order (n, n)) of a food-smear
    pattern: sauce swipes, gravy blobs, herb flecks and crumbs. The fade order is a
    noise field: at grime level g only the part with order < g is still dirty, so
    the smears break up and vanish unevenly as the fly scrubs."""
    rng = np.random.default_rng(1600 + pattern)
    col = np.zeros((n, n, 3), np.float32)
    mask = np.zeros((n, n), np.float32)
    c = n / 2
    pal = [((0.72, 0.18, 0.08), (0.55, 0.12, 0.05)),  # tomato sauce
           ((0.45, 0.28, 0.12), (0.32, 0.2, 0.08)),  # gravy
           ((0.85, 0.66, 0.2), (0.7, 0.5, 0.12))]  # egg yolk / curry
    main, dark = pal[pattern % 3]
    other = pal[(pattern + 1) % 3][0]
    layer = np.zeros((n, n), np.uint8)
    for _ in range(4):  # swipes (the fork dragged sauce round)
        a0 = rng.uniform(0, TAU)
        rr = rng.uniform(0.2, 0.6) * c
        pts = []
        for u in np.linspace(0, rng.uniform(1.0, 2.2), 16):
            pts.append((c + rr * math.cos(a0 + u), c + rr * math.sin(a0 + u)))
        cv2.polylines(layer, [np.array(pts, np.int32)], False, 255, int(rng.integers(8, 18)), cv2.LINE_AA)
    for _ in range(5):  # blobs
        p = (int(c + rng.normal(0, 0.3 * c)), int(c + rng.normal(0, 0.3 * c)))
        cv2.ellipse(layer, p, (int(rng.integers(8, 24)), int(rng.integers(6, 18))), float(rng.uniform(0, 180)),
                    0, 360, 255, -1, cv2.LINE_AA)
    m = cv2.GaussianBlur(layer.astype(np.float32) / 255.0, (0, 0), 2.0)
    tex = fbm(n, n, 8, 8, rng, 3)
    col[:] = _mix(np.broadcast_to(np.array(main, np.float32), (n, n, 3)).copy(), dark, tex)
    mask = np.maximum(mask, m * 0.95)
    # a second food, a smaller smear
    layer2 = np.zeros((n, n), np.uint8)
    for _ in range(3):
        p = (int(c + rng.normal(0, 0.35 * c)), int(c + rng.normal(0, 0.35 * c)))
        cv2.ellipse(layer2, p, (int(rng.integers(6, 16)), int(rng.integers(4, 10))), float(rng.uniform(0, 180)),
                    0, 360, 255, -1, cv2.LINE_AA)
    m2 = cv2.GaussianBlur(layer2.astype(np.float32) / 255.0, (0, 0), 1.5)
    col = _mix(col, other, m2)
    mask = np.maximum(mask, m2 * 0.9)
    # herb flecks and crumbs
    for _ in range(70):
        p = (int(c + rng.normal(0, 0.38 * c)), int(c + rng.normal(0, 0.38 * c)))
        k = rng.random()
        cc = (0.15, 0.4, 0.1) if k < 0.4 else ((0.62, 0.44, 0.22) if k < 0.8 else (0.2, 0.12, 0.06))
        cv2.circle(col, p, int(rng.integers(1, 4)), cc, -1)
        cv2.circle(mask, p, int(rng.integers(1, 4)), 1.0, -1)
    # a greasy film everywhere near the food
    film = cv2.GaussianBlur(mask, (0, 0), 14.0)
    mask = np.maximum(mask, 0.35 * np.clip(film * 3, 0, 1))
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(x - c, y - c) / c
    mask *= (r < 0.97)
    order = 0.75 * fbm(n, n, 6, 6, rng, 3) + 0.25 * rng.random((n, n)).astype(np.float32)
    order = (order - order.min()) / max(float(order.max() - order.min()), 1e-6)
    return col, mask, order


@lru_cache(maxsize=N_SMEARS * GRIME_LEVELS)
def plate_texture(pattern: int, level: int) -> np.ndarray:
    """Plate face at grime stage ``level`` (0 = as it came from the table, the last
    = clean). The dirty part left at grime fraction g is where the fade order < g."""
    clean = plate_clean_texture().astype(np.float32) / 255.0
    col, mask, order = _smear_layer(pattern)
    g = 1.0 - level / (GRIME_LEVELS - 1)
    keep = _smooth(g + 0.04, g - 0.04, order) if g > 0 else np.zeros_like(order)
    keep = np.where(g >= 1.0, 1.0, keep)
    a = np.clip(mask * keep, 0, 1)
    img = _mix(clean, col, a)
    return _to_u8(img)


@lru_cache(maxsize=1)
def sponge_texture(n: int = 128, seed: int = 0) -> np.ndarray:
    """Top half green scourer, bottom yellow foam with pores (planar uv from above:
    the scourer layer is its own geom, so this is the foam)."""
    rng = np.random.default_rng(seed + 1604)
    f = fbm(n, n, 8, 8, rng, 3)
    img = _mix(np.broadcast_to(np.array((0.98, 0.84, 0.25), np.float32), (n, n, 3)).copy(),
               (0.88, 0.7, 0.15), f)
    u8 = _to_u8(img)
    for _ in range(160):
        p = (int(rng.integers(0, n)), int(rng.integers(0, n)))
        cv2.circle(u8, p, int(rng.integers(1, 3)), (200, 150, 30), -1)
    return u8


@lru_cache(maxsize=1)
def scourer_texture(n: int = 128, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + 1605)
    img = np.zeros((n, n, 3), np.uint8)
    img[:] = (40, 120, 60)
    for _ in range(400):
        x0, y0 = rng.integers(0, n, 2)
        a = rng.uniform(0, TAU)
        L = rng.integers(4, 12)
        c = int(rng.integers(20, 90))
        cv2.line(img, (int(x0), int(y0)), (int(x0 + L * math.cos(a)), int(y0 + L * math.sin(a))),
                 (c, 110 + c, 50 + c // 2), 1)
    return img


@lru_cache(maxsize=1)
def water_texture(n: int = 256, seed: int = 0) -> np.ndarray:
    """Soapy dish water from above: grey-blue with a milky soap sheen."""
    rng = np.random.default_rng(seed + 1606)
    f = fbm(n, n, 4, 4, rng, 4)
    img = _mix(np.broadcast_to(np.array((0.62, 0.72, 0.76), np.float32), (n, n, 3)).copy(),
               (0.88, 0.92, 0.93), _smooth(0.4, 0.8, f))
    return _to_u8(img)


@lru_cache(maxsize=1)
def foam_texture(n: int = 128, seed: int = 0) -> np.ndarray:
    """Soap foam: white with packed bubble outlines."""
    rng = np.random.default_rng(seed + 1607)
    img = np.full((n, n, 3), 246, np.uint8)
    for _ in range(220):
        p = (int(rng.integers(0, n)), int(rng.integers(0, n)))
        r = int(rng.integers(2, 8))
        cv2.circle(img, p, r, (205, 215, 225), 1, cv2.LINE_AA)
        cv2.circle(img, (p[0] - r // 3, p[1] - r // 3), max(1, r // 4), (255, 255, 255), -1)
    return img


def label_texture(lines: tuple[str, ...], bg=(0.2, 0.62, 0.35), fg=(0.98, 0.98, 0.95),
                  h: int = 128, w: int = 256) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = tuple(int(255 * v) for v in bg)
    cv2.rectangle(img, (4, 4), (w - 5, h - 5), tuple(int(255 * v) for v in fg), 2, cv2.LINE_AA)
    n = len(lines)
    for i, t in enumerate(lines):
        sc = 1.4 if i == 0 else 0.7
        _text(img, t, (w // 2, int(h * (i + 0.8) / (n + 0.6))), sc, fg, 3 if i == 0 else 2, center=True)
    return img


@lru_cache(maxsize=1)
def belt_texture(h: int = 64, w: int = 512) -> np.ndarray:
    """A black rubber conveyor belt with ribs (along u)."""
    img = np.full((h, w, 3), 30, np.uint8)
    for x in range(0, w, 16):
        cv2.line(img, (x, 0), (x, h), (55, 55, 58), 3)
    cv2.line(img, (0, 2), (w, 2), (90, 90, 90), 2)
    cv2.line(img, (0, h - 3), (w, h - 3), (90, 90, 90), 2)
    return img


def sign_texture(text: str, sub: str, bg=(0.98, 0.93, 0.8), fg=(0.35, 0.2, 0.1), h: int = 96,
                 w: int = 320) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = tuple(int(255 * v) for v in bg)
    cv2.rectangle(img, (5, 5), (w - 6, h - 6), tuple(int(255 * v) for v in fg), 3, cv2.LINE_AA)
    _text(img, text, (w // 2, int(0.42 * h)), 1.2, fg, 3, center=True)
    _text(img, sub, (w // 2, int(0.76 * h)), 0.6, fg, 1, center=True)
    return img
