"""Procedural meshes and textures for the jump rope job (visual only).

Same approach as ``kebab_assets``: everything is generated in code (numpy + OpenCV)
and handed to MjSpec directly; nothing is written to disk. Nothing here collides or
has mass. A bright schoolyard: painted asphalt (hopscotch, a four-square court, the
skipping spot), a brick school wall with windows and a chalkboard scoreboard, a
chain-link fence, a sky with clouds and trees. The school name ("FLY SCHOOL") and all
signs are our own; no real brand or logo.

* ``slab_mesh``: a thin horizontal slab whose top face maps the whole texture once
  (image row 0 at +y), for the painted playground.
* ``wheel_mesh``: a crank wheel (disc about z with a rim).
* textures: ``playground_texture``, ``school_wall_texture``, ``sky_texture``,
  ``scoreboard_texture``, ``post_texture``, ``sign_texture``, ``fence_texture``.
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
from fly_simulator.jobs.mowing_assets import shrub_mesh  # noqa: F401
from fly_simulator.jobs.sisyphus_assets import box_mesh  # noqa: F401
from fly_simulator.jobs.trampoline_assets import _text, disc_mesh, polyline_tube  # noqa: F401


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def slab_mesh(w: float, d: float, t: float = 0.05) -> MeshData:
    """Box ``w`` (x) by ``d`` (y) by ``t`` (z), top face at z = 0: the top maps the whole
    texture once (u along +x, image row 0 at +y); the other faces get a corner of it."""
    x0, x1, y0, y1, z0, z1 = -w / 2, w / 2, -d / 2, d / 2, -t, 0.0
    v = np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
                  [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]], float)
    uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 0], [0, 0]], float)
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                  [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    md = MeshData(v, f, uv)
    if md.signed_volume() < 0:
        md.faces = md.faces[:, ::-1].copy()
    return md


def wheel_mesh(r: float, t: float, n: int = 40) -> MeshData:
    """A crank wheel about z: a disc with a raised rim (planar uv over the face)."""
    rr = np.array([0.0, 0.92 * r, r, r, 0.92 * r, 0.0])
    zz = np.array([t / 2, t / 2, 0.8 * t, -0.8 * t, -t / 2, -t / 2])
    md = lathe(rr, zz, n, v_coord=np.linspace(0, 1, len(rr)))
    md.uv = 0.5 + 0.5 * md.verts[:, :2] / r
    return md


def hoop_meshes(h: float) -> dict[str, MeshData]:
    """A basketball hoop: pole, backboard, ring (base at z = 0, facing -x)."""
    pole = polyline_tube(np.array([[0, 0, 0], [0, 0, h], [-0.8, 0, h + 0.4]]), 0.16)
    ring = transform(torus_mesh(0.75, 0.05, 32, 8), np.eye(3), np.array([-1.9, 0.0, h + 0.1]))
    return {"pole": pole, "ring": ring}


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------


def _paint(img, pts, col, thick) -> None:
    cv2.polylines(img, [np.asarray(pts, np.int32)], True, col, thick, cv2.LINE_AA)


@lru_cache(maxsize=2)
def playground_texture(w_mm: float, d_mm: float, mark_xy: tuple, seed: int = 0,
                       px_per_mm: float = 24.0) -> np.ndarray:
    """Asphalt with painted games. The slab is ``w_mm`` x ``d_mm``, centred at the
    origin; ``mark_xy`` = the skipping spot (a painted star in a circle) in mm."""
    rng = np.random.default_rng(seed + 31)
    W, H = int(w_mm * px_per_mm), int(d_mm * px_per_mm)
    f = fbm(H, W, 12, 16, rng, octaves=5)
    base = _mix(np.array([0.23, 0.24, 0.26], np.float32), np.array([0.34, 0.35, 0.37], np.float32), f)
    img = _to_u8(base)
    n = int(W * H / 30)
    xs, ys = rng.integers(0, W, n), rng.integers(0, H, n)
    tone = rng.integers(-40, 50, n)
    img[ys, xs] = np.clip(img[ys, xs].astype(int) + tone[:, None], 0, 255).astype(np.uint8)

    def px(x, y):  # mm -> pixel (row 0 = +y edge)
        return (int((x + w_mm / 2) * px_per_mm), int((d_mm / 2 - y) * px_per_mm))

    s = px_per_mm
    thick = max(2, int(0.18 * s))
    # the skipping spot: a yellow circle with a star, and a white "rope zone" band
    mx, my = mark_xy
    c = px(mx, my)
    cv2.circle(img, c, int(1.6 * s), (250, 210, 40), thick, cv2.LINE_AA)
    star = []
    for k in range(10):
        a = math.pi / 2 + k * math.pi / 5
        rr = (1.0 if k % 2 == 0 else 0.42) * s
        star.append((c[0] + rr * math.cos(a), c[1] - rr * math.sin(a)))
    cv2.fillPoly(img, [np.asarray(star, np.int32)], (250, 220, 60), cv2.LINE_AA)
    for sgn in (-1, 1):
        a0, a1 = px(mx - 7.2, sgn * 0.6), px(mx + 7.2, sgn * 0.6)
        cv2.line(img, a0, a1, (235, 235, 230), max(1, thick // 2), cv2.LINE_AA)
    _text(img, "JUMP ROPE", px(mx + 3.2, -4.5), 0.055 * s, (0.95, 0.95, 0.9), max(2, thick // 2), center=True)
    # hopscotch (on the -x side): 1 2 3 [4 5] 6 [7 8], white chalk-ish paint
    hx, hy, cs = mx - 16.0, -7.0, 2.2
    boxes = [(0, 0, "1"), (1, 0, "2"), (2, 0, "3"), (3, -0.5, "4"), (3, 0.5, "5"), (4, 0, "6"),
             (5, -0.5, "7"), (5, 0.5, "8")]
    for i, j, lab in boxes:
        x0, y0 = hx + i * cs, hy + j * cs - cs / 2
        _paint(img, [px(x0, y0), px(x0 + cs, y0), px(x0 + cs, y0 + cs), px(x0, y0 + cs)], (240, 240, 235), thick)
        _text(img, lab, px(x0 + cs / 2, y0 + cs / 2), 0.06 * s, (0.95, 0.35, 0.3), max(2, thick // 2), center=True)
    cv2.ellipse(img, px(hx + 6 * cs + 1.0, hy), (int(1.1 * cs * s), int(1.1 * cs * s)), 0, -90, 90,
                (240, 240, 235), thick, cv2.LINE_AA)
    # a four-square court (+x side, far)
    qx, qy, q = mx + 10.0, 9.0, 4.0
    cols = [(220, 60, 60), (60, 130, 230), (60, 190, 90), (240, 190, 40)]
    for k, (i, j) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        a, b = px(qx + i * q, qy + j * q), px(qx + (i + 1) * q, qy + (j + 1) * q)
        cv2.rectangle(img, a, b, cols[k], -1)
        cv2.rectangle(img, a, b, (245, 245, 240), thick)
        _text(img, "ABCD"[k], px(qx + (i + 0.5) * q, qy + (j + 0.5) * q), 0.09 * s, (0.97, 0.97, 0.95),
              max(2, thick), center=True)
    # a painted track lane arc and a few chalk doodles
    cv2.ellipse(img, px(mx - 4.0, 13.0), (int(9 * s), int(3.5 * s)), 0, 180, 360, (250, 250, 245), thick,
                cv2.LINE_AA)
    for _ in range(6):
        cx, cy = int(rng.integers(0, W)), int(rng.integers(0, H))
        col = [(240, 120, 200), (120, 200, 250), (250, 240, 120)][int(rng.integers(0, 3))]
        cv2.circle(img, (cx, cy), int(rng.uniform(0.3, 0.8) * s), col, max(1, thick // 2), cv2.LINE_AA)
        cv2.line(img, (cx - int(0.5 * s), cy + int(1.2 * s)), (cx + int(0.5 * s), cy + int(1.2 * s)), col,
                 max(1, thick // 2), cv2.LINE_AA)
    # cracks
    for _ in range(10):
        x, y = float(rng.integers(0, W)), float(rng.integers(0, H))
        a = rng.uniform(0, TAU)
        pts = []
        for _k in range(8):
            pts.append((x, y))
            a += rng.normal(0, 0.5)
            x += 0.4 * s * math.cos(a)
            y += 0.4 * s * math.sin(a)
        cv2.polylines(img, [np.asarray(pts, np.int32)], False, (28, 28, 30), 1, cv2.LINE_AA)
    return img


@lru_cache(maxsize=2)
def school_wall_texture(seed: int = 0, h: int = 512, w: int = 1536) -> np.ndarray:
    """A red-brick school facade: windows with white frames, a door, a banner sign."""
    rng = np.random.default_rng(seed + 41)
    img = np.zeros((h, w, 3), np.uint8)
    bh, bw = 12, 34
    for r in range(0, h // bh + 1):
        off = (bw // 2) * (r % 2)
        for cidx in range(-1, w // bw + 2):
            x0, y0 = cidx * bw + off, r * bh
            tone = rng.uniform(0.82, 1.1)
            col = np.clip(np.array([176, 70, 52]) * tone, 0, 255)
            cv2.rectangle(img, (x0 + 1, y0 + 1), (x0 + bw - 2, y0 + bh - 2), tuple(int(v) for v in col), -1)
    img[img.sum(-1) == 0] = (196, 188, 176)  # mortar
    # a stone band and the sign
    img[int(h * 0.12):int(h * 0.26)] = (228, 222, 206)
    _text(img, "FLY SCHOOL", (w // 2, int(h * 0.19)), 2.1, (0.12, 0.2, 0.45), 5, center=True)
    _text(img, "RECESS: FOREVER", (int(w * 0.84), int(h * 0.19)), 0.9, (0.75, 0.15, 0.12), 2, center=True)
    _text(img, "EST. ALWAYS", (int(w * 0.16), int(h * 0.19)), 0.9, (0.75, 0.15, 0.12), 2, center=True)
    # windows (two rows), leaving the middle for the door
    for row, (y0, y1) in enumerate(((0.33, 0.55), (0.62, 0.84))):
        for k in range(9):
            xc = int(w * (0.07 + k * 0.108))
            if 3 <= k <= 5 and row == 1:
                continue
            a, b = (xc - 48, int(h * y0)), (xc + 48, int(h * y1))
            cv2.rectangle(img, a, b, (250, 250, 248), -1)
            cv2.rectangle(img, (a[0] + 6, a[1] + 6), (b[0] - 6, b[1] - 6), (120, 170, 215), -1)
            cv2.line(img, (xc, a[1] + 6), (xc, b[1] - 6), (250, 250, 248), 5)
            cv2.line(img, (a[0] + 6, (a[1] + b[1]) // 2), (b[0] - 6, (a[1] + b[1]) // 2), (250, 250, 248), 5)
            # paper cut-outs in the windows (stars, a sun)
            if (k + row) % 3 == 0:
                cv2.circle(img, (xc - 20, a[1] + 26), 11, (250, 210, 60), -1, cv2.LINE_AA)
            if (k + row) % 4 == 1:
                cv2.rectangle(img, (xc + 10, b[1] - 40), (xc + 34, b[1] - 16), (240, 110, 160), -1)
    # the door
    xc = w // 2
    cv2.rectangle(img, (xc - 80, int(h * 0.60)), (xc + 80, h), (40, 90, 170), -1)
    cv2.line(img, (xc, int(h * 0.60)), (xc, h), (25, 60, 120), 4)
    cv2.rectangle(img, (xc - 90, int(h * 0.57)), (xc + 90, int(h * 0.60)), (250, 250, 248), -1)
    img[int(h * 0.965):] = (150, 150, 146)
    return img


@lru_cache(maxsize=2)
def sky_texture(seed: int = 0, h: int = 512, w: int = 1024) -> np.ndarray:
    """A bright sky with clouds over a row of round trees."""
    rng = np.random.default_rng(seed + 5)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    sky = _mix(np.array([0.30, 0.58, 0.97], np.float32), np.array([0.78, 0.90, 1.0], np.float32), t)
    img = _to_u8(np.broadcast_to(sky, (h, w, 3)).copy())
    for _ in range(9):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(h // 12, h // 2.6))
        for _k in range(7):
            cv2.circle(img, (cx + int(rng.integers(-45, 45)), cy + int(rng.integers(-10, 10))),
                       int(rng.integers(14, 32)), (250, 252, 255), -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    base = int(h * 0.80)
    for _ in range(18):  # a treeline
        cx = int(rng.integers(0, w))
        r = int(rng.integers(40, 80))
        cv2.line(img, (cx, base), (cx, base - r), (90, 60, 40), 8)
        g = (int(rng.integers(40, 70)), int(rng.integers(130, 170)), int(rng.integers(50, 80)))
        for _k in range(5):
            cv2.circle(img, (cx + int(rng.integers(-r // 2, r // 2)), base - r - int(rng.integers(0, r // 2))),
                       int(r * rng.uniform(0.45, 0.7)), g, -1, cv2.LINE_AA)
    img[base:] = (70, 160, 80)
    return img


def scoreboard_texture(h: int = 256, w: int = 512) -> np.ndarray:
    """A chalkboard in a wooden frame: title, three labelled digit windows."""
    rng = np.random.default_rng(7)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (38, 70, 52)
    n = tile_noise(h, w, 16, 32, rng)
    img = np.clip(img.astype(np.float32) * (0.9 + 0.2 * n[..., None]), 0, 255).astype(np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (120, 80, 45), 14)
    _text(img, "SKIP-O-METER", (w // 2, 34), 0.95, (0.95, 0.95, 0.88), 2, center=True)
    for i, lab in enumerate(("STREAK", "BEST", "RPM")):
        y = int(h * (0.36 + 0.22 * i))
        _text(img, lab, (int(w * 0.24), y), 0.8, (0.93, 0.93, 0.85), 2, center=True)
        cv2.rectangle(img, (int(w * 0.47), y - 26), (int(w * 0.93), y + 26), (22, 40, 30), -1)
    return img


def post_texture(c1=(0.95, 0.25, 0.2), c2=(0.98, 0.95, 0.9), n: int = 128, k: int = 10) -> np.ndarray:
    """Candy stripes along v (for the turner posts: a cylinder's v runs along its axis)."""
    y = np.arange(n)[:, None] * np.ones((1, n))
    x = np.arange(n)[None, :] * np.ones((n, 1))
    band = (((y + 0.35 * x) // (n / k)) % 2).astype(int)
    a, b = np.asarray(c1, np.float32), np.asarray(c2, np.float32)
    return _to_u8(np.where(band[..., None] == 0, a, b))


def sign_texture(lines: tuple[str, ...], bg=(0.98, 0.85, 0.15), fg=(0.1, 0.1, 0.12),
                 h: int = 128, w: int = 256) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = tuple(int(255 * v) for v in bg)
    cv2.rectangle(img, (4, 4), (w - 5, h - 5), tuple(int(255 * v) for v in fg), 4)
    n = len(lines)
    for i, s in enumerate(lines):
        _text(img, s, (w // 2, int(h * (i + 0.5) / n + 2)), 0.9 if i == 0 else 0.62, fg, 2, center=True)
    return img


def wheel_texture(n: int = 128, col=(0.2, 0.45, 0.9)) -> np.ndarray:
    """A crank wheel face: coloured with white spokes (so the turning shows)."""
    img = np.zeros((n, n, 3), np.uint8)
    img[:] = tuple(int(255 * v) for v in col)
    c = n // 2
    for k in range(6):
        a = TAU * k / 6
        cv2.line(img, (c, c), (int(c + 0.45 * n * math.cos(a)), int(c + 0.45 * n * math.sin(a))),
                 (250, 250, 250), max(2, n // 22), cv2.LINE_AA)
    cv2.circle(img, (c, c), n // 9, (250, 210, 50), -1, cv2.LINE_AA)
    return img


def fence_texture(h: int = 128, w: int = 256) -> np.ndarray:
    """Chain-link diamonds (RGB; the fence material is partly transparent)."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (70, 150, 80)
    step = 16
    for k in range(-h, w + h, step):
        cv2.line(img, (k, 0), (k + h, h), (205, 210, 215), 2, cv2.LINE_AA)
        cv2.line(img, (k, h), (k + h, 0), (205, 210, 215), 2, cv2.LINE_AA)
    return img
