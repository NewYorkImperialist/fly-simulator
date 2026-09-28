"""Procedural meshes and textures for the barista job (visual only).

Same approach as ``kebab_assets``: everything is generated in code (numpy + OpenCV)
and handed to MjSpec directly; nothing is written to disk, nothing here collides or
has mass. A small café: a walnut counter, an espresso machine (our own, unbranded,
with a "compound eye" badge), a grinder, paper cups with the customer's name, a milk
jug, a pastry case, a chalkboard menu. The café's name, "THE COMPOUND EYE", is ours.
Latte art is a texture drawn progressively (heart, tulip, rosetta; a clean and a
wobbly version) on the drink's surface disc.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.broccoli_toss_assets import merge, panel_mesh, panel_quat, sphere_mesh, torus_mesh  # noqa: F401
from fly_simulator.jobs.dishwasher_assets import butcher_block_texture, label_texture  # noqa: F401
from fly_simulator.jobs.fry_cook_assets import bell_mesh  # noqa: F401
from fly_simulator.jobs.jump_rope_assets import slab_mesh  # noqa: F401
from fly_simulator.jobs.kebab_assets import (  # noqa: F401  (re-exported helpers)
    TAU, MeshData, _mix, _smooth, _to_u8, add_mesh, add_texture, add_textured_material, fbm, lathe,
    steel_texture, tile_noise, transform, tube,
)
from fly_simulator.jobs.pizza_chef_assets import planar_uv
from fly_simulator.jobs.trampoline_assets import _text, disc_mesh, polyline_tube  # noqa: F401

CAFE = "THE COMPOUND EYE"
NAMES = ("Ava", "Leo", "Maya", "Sam", "Nora", "Eli", "Zoe", "Max", "Ivy", "Theo", "Ruby", "Omar",
         "Lena", "Jude", "Aria", "Kai")
DRINKS = ("LATTE", "FLAT WHITE", "CAPPUCCINO", "CORTADO")
PATTERNS = ("heart", "tulip", "rosetta")
ART_STAGES = 8
CREMA_STAGES = 5


def drink_of(name_i: int) -> str:
    """Every regular always orders the same drink."""
    return DRINKS[name_i % len(DRINKS)]


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def paper_cup_mesh(r0: float, r1: float, h: float, wall: float = 0.012) -> MeshData:
    """An open paper cup (base r0, top r1, height h) with a rolled rim; base at z = 0."""
    rr = np.array([0.0, r0, r0 + 0.2 * (r1 - r0), r1, r1 + 0.012, r1 + 0.012, r1 - wall,
                   r1 - wall - 0.2 * (r1 - r0), r0 - wall, 0.0])
    zz = np.array([0.0, 0.0, 0.2 * h, h - 0.01, h - 0.01, h + 0.008, h, 0.8 * h, 0.03, 0.03])
    return lathe(rr, zz, 36, v_coord=np.linspace(0, 1, len(rr)))


def band_mesh(r0: float, r1: float, z0: float, z1: float, n: int = 40) -> MeshData:
    """An open conical band (a cup sleeve) from (r0, z0) to (r1, z1): u around, v up;
    a thin shell (inner and outer skin)."""
    th = np.linspace(0, TAU, n + 1)
    rings, uvs = [], []
    for k, (r, z, v) in enumerate(((r0, z0, 1.0), (r1, z1, 0.0), (r1 - 0.004, z1, 0.0), (r0 - 0.004, z0, 1.0))):
        rings.append(np.stack([r * np.cos(th), r * np.sin(th), np.full_like(th, z)], -1))
        uvs.append(np.stack([th / TAU, np.full_like(th, v)], -1))
    md = tube(np.array(rings), np.array(uvs), cap_start=False, cap_end=False)
    return md


def jug_mesh(r: float, h: float) -> MeshData:
    """A stainless milk jug (belly, narrower neck, flared lip); base at z = 0, open."""
    rr = np.array([0.0, 0.92 * r, r, r, 0.86 * r, 0.9 * r, 0.96 * r, 0.9 * r, 0.8 * r, 0.8 * r, 0.9 * r, 0.0])
    zz = np.array([0.0, 0.0, 0.08 * h, 0.55 * h, 0.85 * h, h, h, 0.97 * h, 0.85 * h, 0.1 * h, 0.03 * h, 0.03 * h])
    return lathe(rr, zz, 32)


def spout_mesh(r: float, h: float) -> MeshData:
    """The jug's pouring spout: a small wedge at +y on the lip."""
    pts = np.array([[-0.35 * r, 0.8 * r, 0.82 * h], [0.35 * r, 0.8 * r, 0.82 * h], [0.0, 1.25 * r, 1.02 * h]])
    ring0 = np.vstack([pts, pts[::-1] + np.array([0, -0.02, 0.0])])
    rings = np.stack([ring0 + np.array([0, 0, -0.01]), ring0 + np.array([0, 0, 0.01])])
    return tube(rings, np.zeros(rings.shape[:2] + (2,)))


def basket_mesh(r: float, h: float, wall: float = 0.02) -> MeshData:
    """The portafilter basket: a shallow open steel cup, base at z = 0."""
    rr = np.array([0.0, 0.85 * r, r, r, r + 0.02, r + 0.02, r - wall, r - wall, 0.0])
    zz = np.array([0.0, 0.0, 0.15 * h, h - 0.02, h - 0.02, h, h, wall, wall])
    return lathe(rr, zz, 32)


def tamper_mesh(r: float, t: float, knob_h: float) -> MeshData:
    """A tamper: a flat steel base (thickness t) and a wooden knob handle above it.
    The base's bottom at z = 0."""
    rr = np.array([0.0, r, r, 0.35 * r, 0.25 * r, 0.25 * r, 0.45 * r, 0.42 * r, 0.0])
    zz = np.array([0.0, 0.0, t, t + 0.01, t + 0.05, t + 0.55 * knob_h, t + 0.75 * knob_h, t + knob_h,
                   t + knob_h])
    return lathe(rr, zz, 28, v_coord=np.array([0, 0, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.0]))


def hopper_mesh(r: float, h: float) -> MeshData:
    """The grinder's bean hopper: an upturned cone (narrow at the bottom), open top."""
    rr = np.array([0.0, 0.3 * r, r, r - 0.02, 0.3 * r - 0.02, 0.0])
    zz = np.array([0.0, 0.0, h, h, 0.03, 0.03])
    return lathe(rr, zz, 28)


def croissant_mesh(L: float, r: float) -> MeshData:
    """A croissant: a crescent tube, fat in the middle, pinched tips."""
    pts = []
    for u in np.linspace(-1, 1, 15):
        a = 0.9 * u
        pts.append((0.5 * L * math.sin(a), -0.35 * L * math.cos(a) + 0.3 * L, 0.0))
    pts = np.array(pts)
    th = np.linspace(0, TAU, 14, endpoint=False)
    rings, uvs = [], []
    for k in range(len(pts)):
        t = pts[min(k + 1, len(pts) - 1)] - pts[max(k - 1, 0)]
        t /= np.linalg.norm(t)
        nrm = np.cross(t, [0, 0, 1.0])
        rad = r * (0.25 + 0.75 * math.sin(math.pi * k / (len(pts) - 1))) * (1.0 + 0.12 * math.cos(6 * k))
        ring = pts[k] + rad * (np.cos(th)[:, None] * nrm + 0.75 * np.sin(th)[:, None] * np.array([0, 0, 1.0]))
        ring[:, 2] = np.maximum(ring[:, 2], -0.2 * rad)
        rings.append(ring)
        uvs.append(np.stack([th / TAU, np.full_like(th, k / (len(pts) - 1))], -1))
    return tube(np.array(rings), np.array(uvs))


def muffin_mesh(r: float, h: float) -> MeshData:
    """A muffin: a fluted cup and a domed top, base at z = 0."""
    rr = np.array([0.0, 0.78 * r, 0.9 * r, 1.12 * r, 1.05 * r, 0.8 * r, 0.45 * r, 0.0])
    zz = np.array([0.0, 0.0, 0.55 * h, 0.62 * h, 0.8 * h, 0.95 * h, 1.02 * h, h])
    return lathe(rr, zz, 24, radial_noise=lambda th, z: 0.03 * r * np.sin(14 * th) * (z < 0.56 * h),
                 v_coord=np.array([0.0, 0.1, 0.5, 0.56, 0.7, 0.85, 0.95, 1.0]))


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------


def _bgr(rgb) -> tuple[int, int, int]:
    return tuple(int(255 * v) for v in rgb)


@lru_cache(maxsize=1)
def chalkboard_texture(h: int = 384, w: int = 512) -> np.ndarray:
    """The menu chalkboard: the café's name (ours), the drinks and pastries."""
    rng = np.random.default_rng(1701)
    g = fbm(h, w, 6, 8, rng, 3)
    img = (0.12 + 0.05 * (g - 0.5))[..., None] * np.array([0.92, 1.05, 0.98], np.float32)
    img = _mix(img, (0.42, 0.45, 0.43), _smooth(0.55, 0.8, fbm(h, w, 3, 4, rng, 3)) * 0.22)
    u8 = _to_u8(img)
    cv2.rectangle(u8, (0, 0), (w - 1, h - 1), _bgr((0.42, 0.27, 0.14)), 18)
    chalk = (0.95, 0.94, 0.88)
    _text(u8, CAFE, (w // 2, 48), 1.25, (0.98, 0.85, 0.5), 2, font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, center=True)
    _text(u8, "coffee - pastries - open forever", (w // 2, 86), 0.55, chalk, 1, center=True)
    rows = (("ESPRESSO", "1.80"), ("CORTADO", "2.60"), ("FLAT WHITE", "3.10"), ("LATTE", "3.40"),
            ("CAPPUCCINO", "3.20"), ("CROISSANT", "2.20"), ("MUFFIN", "2.50"))
    for k, (a, b) in enumerate(rows):
        yy = 124 + k * 34
        cv2.putText(u8, a, (40, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.72, _bgr(chalk), 1, cv2.LINE_AA)
        cv2.putText(u8, b, (w - 120, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.72, _bgr(chalk), 1, cv2.LINE_AA)
        for x in range(260, w - 130, 12):
            cv2.circle(u8, (x, yy - 5), 1, _bgr((0.7, 0.7, 0.66)), -1)
    # a doodled cup with a heart
    cx, cy = w - 80, h - 50
    cv2.ellipse(u8, (cx, cy), (32, 10), 0, 0, 360, _bgr(chalk), 2, cv2.LINE_AA)
    cv2.line(u8, (cx - 32, cy), (cx - 22, cy + 30), _bgr(chalk), 2, cv2.LINE_AA)
    cv2.line(u8, (cx + 32, cy), (cx + 22, cy + 30), _bgr(chalk), 2, cv2.LINE_AA)
    return u8


@lru_cache(maxsize=1)
def wall_texture(h: int = 512, w: int = 1024, seed: int = 0) -> np.ndarray:
    """A warm painted brick wall (terracotta / cream), soft mortar."""
    rng = np.random.default_rng(seed + 1702)
    img = np.zeros((h, w, 3), np.float32)
    img[:] = (0.86, 0.8, 0.7)
    rows, cols = 16, 12
    bh, bw = h / rows, w / cols
    for r in range(rows):
        off = 0.5 * bw if r % 2 else 0.0
        for c in range(-1, cols + 1):
            x0, x1 = int(c * bw + off + 3), int((c + 1) * bw + off - 3)
            y0, y1 = int(r * bh + 3), int((r + 1) * bh - 3)
            x0c, x1c = max(x0, 0), min(x1, w)
            if x1c <= x0c:
                continue
            t = rng.uniform(-0.06, 0.06)
            img[y0:y1, x0c:x1c] = (0.72 + t, 0.42 + t * 0.8, 0.3 + t * 0.6)
    n = fbm(h, w, 8, 16, rng, 4)
    img *= (0.9 + 0.2 * n)[..., None]
    return _to_u8(img)


@lru_cache(maxsize=len(NAMES))
def ticket_texture(name_i: int, h: int = 160, w: int = 112) -> np.ndarray:
    """An order ticket: the drink and the customer's name, a torn edge."""
    img = np.full((h, w, 3), 248, np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, 18), (240, 200, 80), -1)
    cv2.putText(img, "ORDER", (8, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 40, 20), 1, cv2.LINE_AA)
    _text(img, drink_of(name_i), (w // 2, 58), 0.5 if len(drink_of(name_i)) > 7 else 0.62, (0.1, 0.1, 0.1), 1,
          center=True)
    _text(img, NAMES[name_i], (w // 2, 104), 0.9, (0.1, 0.2, 0.55), 2, font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
          center=True)
    for x in range(0, w, 8):
        cv2.fillConvexPoly(img, np.array([[x, h - 1], [x + 4, h - 7], [x + 8, h - 1]]), (225, 225, 225))
    return img


@lru_cache(maxsize=len(NAMES))
def sleeve_texture(name_i: int, h: int = 64, w: int = 512) -> np.ndarray:
    """A kraft cup sleeve (u around the cup) with the name written twice in marker."""
    rng = np.random.default_rng(1703 + name_i)
    f = fbm(h, w, 4, 16, rng, 3)
    img = _to_u8(_mix(np.broadcast_to(np.array((0.72, 0.55, 0.36), np.float32), (h, w, 3)).copy(),
                      (0.64, 0.48, 0.3), f))
    for k in range(2):
        cx = int(w * (0.25 + 0.5 * k))
        _text(img, NAMES[name_i], (cx, h // 2), 1.2, (0.08, 0.08, 0.12), 3, font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
              center=True)
    cv2.line(img, (0, 3), (w, 3), (90, 60, 35), 2)
    cv2.line(img, (0, h - 4), (w, h - 4), (90, 60, 35), 2)
    return img


@lru_cache(maxsize=1)
def paper_texture(n: int = 128) -> np.ndarray:
    """The white paper cup with a thin green stripe near the rim."""
    img = np.full((n, n, 3), 246, np.uint8)
    img[:, :] = (246, 244, 238)
    return img


@lru_cache(maxsize=CREMA_STAGES)
def crema_texture(stage: int, n: int = 128) -> np.ndarray:
    """The espresso's surface while the shot runs: stage 0 near-black, the last a
    hazelnut crema with a darker rim and tiger flecks."""
    rng = np.random.default_rng(1704 + stage)
    p = stage / (CREMA_STAGES - 1)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    c = (n - 1) / 2
    r = np.hypot(x - c, y - c) / c
    dark = np.array((0.1, 0.05, 0.025), np.float32)
    crema = np.array((0.72, 0.46, 0.22), np.float32)
    base = _mix(np.broadcast_to(dark, (n, n, 3)).copy(), crema, np.full((n, n), 0.2 + 0.8 * p) * (1 - 0.5 * r ** 3))
    fl = fbm(n, n, 10, 10, rng, 3)
    base = _mix(base, (0.35, 0.18, 0.08), _smooth(0.6, 0.8, fl) * 0.6 * p)
    base = _mix(base, (0.9, 0.7, 0.45), _smooth(0.72, 0.9, fbm(n, n, 6, 6, rng, 2)) * 0.35 * p)
    return _to_u8(base)


def _heart_pts(cx, cy, s, n=80):
    t = np.linspace(0, TAU, n)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    return np.stack([cx + s * x / 17, cy + s * y / 17], 1).astype(np.int32)


@lru_cache(maxsize=len(PATTERNS) * 2 * ART_STAGES)
def art_texture(pattern: int, wobbly: int, stage: int, n: int = 160) -> np.ndarray:
    """Latte art on the crema (top-down), ``stage`` 0 (the milk just going in) ..
    ART_STAGES - 1 (finished). The milk (white) is drawn as a mask that grows with the
    pour; ``wobbly`` warps it and softens the edges (an unsteady pour). The pour comes
    from the -v side of the texture (the jug's side)."""
    rng = np.random.default_rng(1705 + 31 * pattern + 7 * wobbly)
    p = stage / (ART_STAGES - 1)
    c = (n - 1) / 2
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(x - c, y - c) / c
    base = crema_texture(CREMA_STAGES - 1, n).astype(np.float32) / 255.0
    # the milk first browns the crema (a lighter surface), then the white pattern rises
    base = _mix(base, (0.78, 0.58, 0.36), np.full((n, n), min(1.0, 2.5 * p) * 0.5) * (r < 0.95))
    m = np.zeros((n, n), np.uint8)
    R = 0.72 * c
    kind = PATTERNS[pattern % len(PATTERNS)]
    if kind == "heart":
        s = R * min(1.0, 0.25 + 0.9 * p)
        if p < 0.75:
            cv2.circle(m, (int(c), int(c + 0.1 * c)), int(0.9 * s), 255, -1, cv2.LINE_AA)
        else:
            cv2.fillPoly(m, [_heart_pts(c, c + 0.05 * c, s)], 255, cv2.LINE_AA)
    elif kind == "tulip":
        k = int(min(3, 1 + math.floor(3.2 * p)))
        for j in range(k):
            rr = R * (0.5 - 0.1 * j) * min(1.0, 0.4 + 1.2 * p)
            cy = c + 0.35 * c - j * 0.36 * c
            cv2.ellipse(m, (int(c), int(cy)), (int(rr), int(0.75 * rr)), 0, 180, 360, 255, -1, cv2.LINE_AA)
            cv2.ellipse(m, (int(c), int(cy)), (int(rr), int(0.35 * rr)), 0, 0, 180, 255, -1, cv2.LINE_AA)
    else:  # rosetta: nested leaves stacked along the axis
        k = int(round(2 + 7 * p))
        for j in range(k):
            yy = c + 0.55 * c - j * 0.14 * c
            w = R * (0.95 - 0.08 * j) * min(1.0, 0.5 + p)
            leaf = np.zeros_like(m)
            cv2.ellipse(leaf, (int(c), int(yy)), (int(w), int(0.17 * c)), 0, 0, 360, 255, -1, cv2.LINE_AA)
            cv2.ellipse(leaf, (int(c), int(yy + 0.06 * c)), (int(0.93 * w), int(0.15 * c)), 0, 0, 360, 0, -1,
                        cv2.LINE_AA)
            m = np.maximum(m, leaf)
        cv2.ellipse(m, (int(c), int(c + 0.55 * c - (k - 1) * 0.14 * c)), (int(0.12 * c), int(0.1 * c)), 0, 0, 360,
                    255, -1, cv2.LINE_AA)
    if p >= 0.99 and kind != "heart" or (kind == "heart" and p >= 0.99):
        # the pull-through line
        cv2.line(m, (int(c), int(c + 0.8 * c)), (int(c), int(c - 0.6 * c)), 255, 3, cv2.LINE_AA)
    mf = m.astype(np.float32) / 255.0
    if wobbly:
        dx = (fbm(n, n, 4, 4, rng, 2) - 0.5) * 0.18 * n
        dy = (fbm(n, n, 4, 4, rng, 2) - 0.5) * 0.12 * n
        mf = cv2.remap(mf, (x + dx).astype(np.float32), (y + dy).astype(np.float32), cv2.INTER_LINEAR)
        mf = cv2.GaussianBlur(mf, (0, 0), 2.2)
    else:
        mf = cv2.GaussianBlur(mf, (0, 0), 0.8)
    mf *= (r < 0.94)
    img = _mix(base, (0.97, 0.94, 0.88), np.clip(mf, 0, 1))
    return _to_u8(img)


@lru_cache(maxsize=1)
def gauge_texture(n: int = 128) -> np.ndarray:
    """The pressure gauge face: 0-15 bar, a green brew band."""
    img = np.full((n, n, 3), 244, np.uint8)
    c = n // 2
    cv2.circle(img, (c, c), c - 3, (40, 40, 40), 3, cv2.LINE_AA)
    cv2.ellipse(img, (c, c), (c - 14, c - 14), 0, 180 + 8 * 18, 180 + 11 * 18, (60, 170, 60), 6)
    for k in range(16):
        a = math.radians(180 + k * 12)
        p0 = (int(c + (c - 10) * math.cos(a)), int(c + (c - 10) * math.sin(a)))
        p1 = (int(c + (c - 20) * math.cos(a)), int(c + (c - 20) * math.sin(a)))
        cv2.line(img, p0, p1, (30, 30, 30), 2 if k % 5 == 0 else 1, cv2.LINE_AA)
    cv2.putText(img, "BAR", (c - 16, c + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    return img


@lru_cache(maxsize=1)
def machine_front_texture(h: int = 256, w: int = 512) -> np.ndarray:
    """The espresso machine's front: polished red enamel panel with a chrome trim and
    our own badge (a compound-eye honeycomb, the café's name)."""
    rng = np.random.default_rng(1706)
    f = fbm(h, w, 3, 6, rng, 3)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    img = _mix(np.broadcast_to(np.array((0.62, 0.08, 0.07), np.float32), (h, w, 3)).copy(),
               (0.85, 0.2, 0.16), 0.5 * (1 - y) + 0.15 * f)
    u8 = _to_u8(img)
    cv2.rectangle(u8, (4, 4), (w - 5, h - 5), (210, 210, 215), 6)
    cx, cy = w // 2, int(h * 0.3)
    for j in range(-2, 3):
        for i in range(-3, 4):
            px = int(cx + i * 14 + (7 if j % 2 else 0))
            py = int(cy + j * 12)
            if (px - cx) ** 2 + ((py - cy) * 1.3) ** 2 < 42 ** 2:
                pts = np.array([[px + 6 * math.cos(a), py + 6 * math.sin(a)] for a in np.linspace(0, TAU, 7)[:-1]],
                               np.int32)
                cv2.polylines(u8, [pts], True, (240, 200, 120), 1, cv2.LINE_AA)
    _text(u8, CAFE, (cx, int(h * 0.62)), 0.8, (0.98, 0.9, 0.7), 2, center=True)
    _text(u8, "espresso  -  est. forever", (cx, int(h * 0.76)), 0.5, (0.95, 0.85, 0.7), 1, center=True)
    return u8


@lru_cache(maxsize=1)
def pastry_texture(n: int = 128, seed: int = 0) -> np.ndarray:
    """Golden laminated pastry (croissant) with a glossy egg wash."""
    rng = np.random.default_rng(seed + 1707)
    v = np.linspace(0, 1, n, dtype=np.float32)[:, None] * np.ones((1, n), np.float32)
    stripes = 0.5 + 0.5 * np.sin(v * 40)
    img = _mix(np.broadcast_to(np.array((0.85, 0.55, 0.2), np.float32), (n, n, 3)).copy(), (0.6, 0.32, 0.1),
               stripes * 0.6 + 0.2 * fbm(n, n, 8, 8, rng, 2))
    return _to_u8(img)


@lru_cache(maxsize=1)
def beans_texture(n: int = 128, seed: int = 0) -> np.ndarray:
    """Roasted coffee beans seen from above (the hopper)."""
    rng = np.random.default_rng(seed + 1708)
    img = np.full((n, n, 3), (40, 22, 12), np.uint8)
    for _ in range(160):
        p = (int(rng.integers(0, n)), int(rng.integers(0, n)))
        a = float(rng.uniform(0, 180))
        col = tuple(int(v) for v in rng.uniform([70, 38, 18], [110, 62, 32]))
        cv2.ellipse(img, p, (6, 4), a, 0, 360, col, -1, cv2.LINE_AA)
        cv2.ellipse(img, p, (5, 1), a, 0, 360, (30, 15, 8), 1, cv2.LINE_AA)
    return img


@lru_cache(maxsize=1)
def grounds_texture(n: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + 1709)
    f = fbm(n, n, 8, 8, rng, 3)
    img = _mix(np.broadcast_to(np.array((0.24, 0.14, 0.08), np.float32), (n, n, 3)).copy(), (0.36, 0.22, 0.12), f)
    img[rng.random((n, n)) > 0.9] = (0.15, 0.08, 0.04)
    return _to_u8(img)


def sign_texture(text: str, sub: str, bg=(0.15, 0.12, 0.1), fg=(0.98, 0.88, 0.6), h: int = 96,
                 w: int = 320) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = _bgr(bg)
    cv2.rectangle(img, (5, 5), (w - 6, h - 6), _bgr(fg), 2, cv2.LINE_AA)
    _text(img, text, (w // 2, int(0.42 * h)), 1.1, fg, 2, center=True)
    _text(img, sub, (w // 2, int(0.78 * h)), 0.5, fg, 1, center=True)
    return img
