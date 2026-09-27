"""Procedural meshes and textures for the bowling job (visual only).

Same approach as ``kebab_assets``: everything is generated in code (numpy + OpenCV)
and handed to MjSpec directly; nothing is written to disk. Nothing here collides or
has mass (callers add the meshes with ``contact_kwargs("visual")`` and ``mass=0``).

* ``pin_mesh``: a ten-pin as a surface of revolution through the regulation profile
  (USBC pin: 15 in tall, 4.766 in at the belly, 1.797 in at the neck), scaled to
  ``height``; its v texture coordinate is z / height, so ``pin_texture`` rows map to
  heights (white lacquer, two red neck stripes).
* ``ball_mesh``: a UV sphere (lathe) for the swirled ``ball_texture``.
* ``slab_mesh``: a thin flat slab whose top face spans the whole texture once (the
  lane: ``lane_texture`` has the boards, arrows, dots, foul line and pin spots drawn
  at their positions in mm).
* ``gutter_mesh``: a half-pipe channel along x.
* textures: lane, pin, ball swirl, cosmic bowling-alley carpet, masking-unit panel
  (cv2 text).
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.kebab_assets import (  # noqa: F401  (re-exported helpers)
    MeshData, _mix, _smooth, _to_u8, add_mesh, add_texture, add_textured_material, fbm, lathe,
    tile_noise, tube,
)

TAU = 2.0 * math.pi

# regulation ten-pin profile: (height from the base, diameter), inches (USBC spec
# table, rounded); the top is closed with a dome
PIN_PROFILE_IN = np.array([
    (0.0, 2.03), (0.75, 2.83), (2.25, 3.95), (3.375, 4.51), (4.5, 4.766), (5.875, 4.5),
    (7.25, 3.62), (8.625, 2.53), (9.375, 1.99), (10.0, 1.797), (10.875, 1.87),
    (11.75, 2.1), (12.5, 2.35), (13.25, 2.44), (14.0, 2.2), (14.5, 1.75), (14.8, 1.1),
    (15.0, 0.0)])
PIN_STRIPES = ((0.615, 0.650), (0.690, 0.725))  # red neck stripes, as z / height


def pin_profile(height: float) -> tuple[np.ndarray, np.ndarray]:
    """(radius, z) of the pin profile scaled to ``height`` (mm)."""
    s = height / 15.0
    return PIN_PROFILE_IN[:, 1] / 2 * s, PIN_PROFILE_IN[:, 0] * s


def pin_mesh(height: float, n_theta: int = 36) -> MeshData:
    """The pin, base at z = 0, axis z; v = z / height (interpolated profile)."""
    zi = np.linspace(0.0, 15.0, 61)
    zi = np.unique(np.concatenate([zi, np.array(PIN_STRIPES).ravel() * 15.0]))
    d = np.interp(zi, PIN_PROFILE_IN[:, 0], PIN_PROFILE_IN[:, 1])
    # round the top: a dome over the last 1 in
    top = zi > 14.0
    d[top] = 2.2 * np.sqrt(np.clip(1.0 - ((zi[top] - 14.0) / 1.0) ** 2, 0.0, 1.0))
    s = height / 15.0
    r, z = d / 2 * s, zi * s
    r[0] = r[0] * 0.999
    return lathe(r, z, n_theta, v_coord=z / height)


def ball_mesh(radius: float, n_theta: int = 40, n_phi: int = 22) -> MeshData:
    ph = np.linspace(-math.pi / 2, math.pi / 2, n_phi)
    return lathe(radius * np.cos(ph), radius * np.sin(ph), n_theta,
                 v_coord=np.linspace(0, 1, n_phi))


def slab_mesh(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float,
              z1_end: float | None = None, v0: float = 0.0, v1: float = 1.0) -> MeshData:
    """Slab x0..x1, y0..y1 from z0 up to its top face (height z1 at x0, ``z1_end``
    at x1: a slanted top; the bottom is lowered alike) whose top (and bottom) face
    maps the texture: u runs across y (y0 -> 0, y1 -> 1), v along x (x0 -> v0,
    x1 -> v1)."""
    ze = z1 if z1_end is None else z1_end
    dz = ze - z1
    v = np.array([[x0, y0, z0], [x1, y0, z0 + dz], [x1, y1, z0 + dz], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, ze], [x1, y1, ze], [x0, y1, z1]], float)
    uv = np.array([[0, v0], [0, v1], [1, v1], [1, v0]] * 2, float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    md = MeshData(v, f, uv)
    if md.signed_volume() < 0:
        md.faces = md.faces[:, ::-1].copy()
    return md


def gutter_mesh(x0: float, x1: float, radius: float, wall: float = 0.08,
                n: int = 16) -> MeshData:
    """Half-pipe along x: inner semicircle of ``radius`` about (y=0, z=0) below its
    lips at z = 0, ``wall`` thick. Closed (two rings + caps)."""
    a = np.linspace(math.pi, TAU, n)  # lip (-y) -> bottom -> lip (+y)
    inner = np.stack([radius * np.cos(a), radius * np.sin(a)], 1)
    outer = np.stack([(radius + wall) * np.cos(a[::-1]), (radius + wall) * np.sin(a[::-1])], 1)
    loop = np.concatenate([inner, outer])
    rings = np.stack([np.column_stack([np.full(len(loop), x), loop]) for x in (x0, x1)])
    uv = np.zeros(rings.shape[:2] + (2,))
    uv[..., 0] = np.linspace(0, 1, len(loop))[None]
    uv[1, :, 1] = 1.0
    return tube(rings, uv)


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def pin_texture(seed: int = 0, h: int = 256, w: int = 64) -> np.ndarray:
    """Rows = z / height (row 0 = the base). White lacquer with a faint sheen noise
    and two red neck stripes."""
    rng = np.random.default_rng(seed + 707)
    v = (np.arange(h, dtype=np.float32) + 0.5) / h
    base = np.ones((h, w, 3), np.float32) * np.array([0.95, 0.94, 0.92], np.float32)
    base *= (0.97 + 0.03 * tile_noise(h, w, 8, 4, rng))[..., None]
    red = np.zeros(h, np.float32)
    for z0, z1 in PIN_STRIPES:
        red = np.maximum(red, _smooth(z0 - 0.004, z0 + 0.004, v) * _smooth(z1 + 0.004, z1 - 0.004, v))
    base = _mix(base, (0.80, 0.06, 0.07), np.broadcast_to(red[:, None], (h, w)))
    # a scuffed band where pins hit each other (belly)
    scuff = _smooth(0.18, 0.28, v) * _smooth(0.42, 0.32, v)
    base = _mix(base, (0.78, 0.76, 0.72),
                np.broadcast_to(scuff[:, None], (h, w)) * _smooth(0.75, 0.95, tile_noise(h, w, 32, 16, rng)))
    return _to_u8(base)


@lru_cache(maxsize=4)
def ball_texture(seed: int = 0, colour: tuple = (0.42, 0.10, 0.62), accent: tuple = (0.10, 0.55, 0.95),
                 h: int = 256, w: int = 512) -> np.ndarray:
    """Reactive-resin swirl (domain-warped noise) in two colours plus dark veins."""
    rng = np.random.default_rng(seed + 808)
    n1 = fbm(h, w, 4, 8, rng, 4)
    n2 = fbm(h, w, 4, 8, rng, 4)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = (xx + 60.0 * (n1 - 0.5) * 4) % w
    my = np.clip(yy + 40.0 * (n2 - 0.5) * 4, 0, h - 1)
    n3 = fbm(h, w, 6, 12, rng, 4)
    warped = cv2.remap(n3, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    band = 0.5 + 0.5 * np.sin(warped * 18.0)
    base = _mix(np.ones((h, w, 3), np.float32) * np.array(colour, np.float32), accent,
                _smooth(0.55, 0.95, band) * 0.85)
    veins = _smooth(0.03, 0.0, np.abs(band - 0.2))
    base = _mix(base, (0.05, 0.02, 0.08), veins * 0.6)
    sparkle = _smooth(0.93, 0.99, tile_noise(h, w, 128, 256, rng))
    base = _mix(base, (0.95, 0.9, 1.0), sparkle * 0.5)
    return _to_u8(base)


@lru_cache(maxsize=2)
def carpet_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """'Cosmic' bowling-alley carpet: dark navy with neon squiggles, rings, stars."""
    rng = np.random.default_rng(seed + 909)
    img = np.ones((n, n, 3), np.float32) * np.array([0.05, 0.05, 0.16], np.float32)
    img *= (0.8 + 0.4 * fbm(n, n, 8, 8, rng, 3))[..., None]
    neon = [(0.0, 0.95, 0.95), (1.0, 0.2, 0.75), (1.0, 0.9, 0.1), (0.4, 1.0, 0.3), (0.55, 0.4, 1.0)]
    canvas = np.zeros((n, n, 3), np.float32)
    for k in range(60):
        col = neon[k % len(neon)][::-1]  # cv2 draws BGR; flipped back below
        cx, cy = rng.integers(0, n, 2)
        kind = k % 4
        if kind == 0:  # squiggle
            t = np.linspace(0, 1, 30)
            ang = rng.uniform(0, TAU)
            L = rng.uniform(25, 60)
            px = cx + L * t * math.cos(ang) + 5 * np.sin(t * 12) * -math.sin(ang)
            py = cy + L * t * math.sin(ang) + 5 * np.sin(t * 12) * math.cos(ang)
            pts = np.stack([px, py], 1).astype(np.int32)
            cv2.polylines(canvas, [pts], False, col, 3, cv2.LINE_AA)
        elif kind == 1:  # ring
            cv2.circle(canvas, (int(cx), int(cy)), int(rng.integers(6, 16)), col, 3, cv2.LINE_AA)
        elif kind == 2:  # star
            r = rng.uniform(6, 12)
            pts = []
            for j in range(10):
                rr = r if j % 2 == 0 else r * 0.45
                a = j * math.pi / 5
                pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
            cv2.fillPoly(canvas, [np.array(pts, np.int32)], col, cv2.LINE_AA)
        else:  # triangle outline
            r = rng.uniform(8, 16)
            a0 = rng.uniform(0, TAU)
            pts = np.array([(cx + r * math.cos(a0 + j * TAU / 3), cy + r * math.sin(a0 + j * TAU / 3))
                            for j in range(3)], np.int32)
            cv2.polylines(canvas, [pts], True, col, 3, cv2.LINE_AA)
    canvas = canvas[..., ::-1]
    m = canvas.max(-1, keepdims=True)
    img = img * (1 - m) + canvas * 0.9
    return _to_u8(img)


@lru_cache(maxsize=2)
def masking_texture(text: str = "FLY LANES", sub: str = "LEAGUE NIGHT - EVERY NIGHT",
                    h: int = 128, w: int = 1024) -> np.ndarray:
    """The masking unit's front panel: a dark-to-purple gradient, a neon title, stars."""
    rng = np.random.default_rng(1010)
    g = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    img = (np.array([0.10, 0.02, 0.18], np.float32) * (1 - g)
           + np.array([0.02, 0.08, 0.25], np.float32) * g) * np.ones((h, 1, 1), np.float32)
    bgr = np.ascontiguousarray((img[..., ::-1] * 255).astype(np.uint8))
    for _ in range(40):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.circle(bgr, (x, y), int(rng.integers(1, 3)), (255, 255, 255), -1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 2.4, 5)
    org = ((w - tw) // 2, 70)
    cv2.putText(bgr, text, org, font, 2.4, (200, 60, 255), 12, cv2.LINE_AA)  # glow
    cv2.putText(bgr, text, org, font, 2.4, (255, 235, 255), 4, cv2.LINE_AA)
    (sw, _), _ = cv2.getTextSize(sub, font, 0.9, 2)
    cv2.putText(bgr, sub, ((w - sw) // 2, 110), font, 0.9, (80, 230, 255), 2, cv2.LINE_AA)
    # the image rows run with v; flip so the text reads upright on the panel
    return np.ascontiguousarray(bgr[::-1, :, ::-1])


@lru_cache(maxsize=2)
def lane_texture(width: float, x0: float, x1: float, foul_x: float, head_x: float,
                 deck_x0: float, pin_spots: tuple, arrows_x: float, dots_x: tuple,
                 approach_dots_x: tuple, n_boards: int = 39, px_per_mm: float = 26.0,
                 seed: int = 0) -> np.ndarray:
    """Top of the lane bed from x0 to x1 (rows, v = along x) across ``width`` mm
    (columns, u = across y, column 0 at y = -width / 2). Maple / pine boards with
    grain and staggered butt joints, the black foul line, the 7 target arrows, guide
    dots, approach dots, and the pin spots on the (maple) pin deck."""
    rng = np.random.default_rng(seed + 1111)
    H = int(round((x1 - x0) * px_per_mm))
    W = int(round(width * px_per_mm))
    H, W = min(H, 2048), min(W, 512)
    sx, sy = H / (x1 - x0), W / width
    rows = (np.arange(H, dtype=np.float32) + 0.5) / sx + x0  # x of each row (mm)
    cols = (np.arange(W, dtype=np.float32) + 0.5) / sy - width / 2  # y of each column
    board = np.clip(((cols + width / 2) / width * n_boards).astype(int), 0, n_boards - 1)
    tone = rng.uniform(-1, 1, n_boards).astype(np.float32)
    # region: maple (approach, first part of the lane, pin deck), pine in between
    pine = ((rows > foul_x + 0.27 * (head_x - foul_x)) & (rows < deck_x0 - 0.3)).astype(np.float32)
    maple = np.array([0.86, 0.66, 0.40], np.float32)
    pine_c = np.array([0.80, 0.56, 0.30], np.float32)
    base = maple[None, None] * (1 - pine[:, None, None]) + pine_c[None, None] * pine[:, None, None]
    base = base * np.ones((H, W, 1), np.float32)
    base *= (1.0 + 0.06 * tone[board])[None, :, None]
    grain = tile_noise(H, W, max(H // 48, 2), W, rng)  # streaks along x
    grain2 = tile_noise(H, W, max(H // 12, 2), W // 2, rng)
    base *= (0.9 + 0.12 * grain + 0.06 * grain2)[..., None]
    # board seams
    fy = (cols + width / 2) / width * n_boards
    seam = _smooth(0.07, 0.0, np.abs(fy - np.round(fy)))
    base = _mix(base, (0.45, 0.30, 0.16), np.broadcast_to(seam[None, :], (H, W)) * 0.5)
    # staggered butt joints
    for b in range(n_boards):
        c0, c1 = np.searchsorted(cols, -width / 2 + b * width / n_boards), \
            np.searchsorted(cols, -width / 2 + (b + 1) * width / n_boards)
        for xj in rng.uniform(x0, x1, 2):
            r = int((xj - x0) * sx)
            base[max(r - 1, 0):r + 1, c0:c1] *= 0.7
    bgr = np.ascontiguousarray((base[..., ::-1] * 255).clip(0, 255).astype(np.uint8))

    def px(x, y):
        return int(round((y + width / 2) * sy)), int(round((x - x0) * sx))

    def mm(d):
        return max(1, int(round(d * sy)))

    # foul line (black)
    r0 = int((foul_x - x0) * sx)
    bgr[max(r0 - mm(0.08), 0):r0 + mm(0.08)] = (25, 22, 20)
    # 7 target arrows (boards 5, 10, ..., 35 from the right), a V pointing to the pins
    for k, b in enumerate((5, 10, 15, 20, 25, 30, 35)):
        y = width / 2 - (b - 0.5) * width / n_boards  # boards counted from the right (-y)
        y = -y
        xa = arrows_x + (3 - abs(k - 3)) * 0.35  # centre arrow furthest down the lane
        tip = px(xa + 0.55, y)
        l = px(xa, y - 0.22)
        r = px(xa, y + 0.22)
        cv2.fillPoly(bgr, [np.array([tip, l, px(xa + 0.18, y), r], np.int32)], (30, 30, 150),
                     cv2.LINE_AA)
    # guide dots on the lane and approach dots
    for xd in dots_x:
        for b in (3, 5, 8, 11, 14):
            for sgn in (-1, 1):
                y = sgn * (width / 2 - (b - 0.5) * width / n_boards)
                cv2.circle(bgr, px(xd, y), mm(0.07), (30, 25, 25), -1, cv2.LINE_AA)
    for xd in approach_dots_x:
        for b in (3, 5, 10, 15, 20, 25, 30, 35, 37):
            y = -width / 2 + (b - 0.5) * width / n_boards
            cv2.circle(bgr, px(xd, y), mm(0.09), (30, 25, 25), -1, cv2.LINE_AA)
    # pin spots
    for (xp, yp) in pin_spots:
        cv2.circle(bgr, px(xp, yp), mm(0.28), (40, 45, 70), 2, cv2.LINE_AA)
        cv2.circle(bgr, px(xp, yp), mm(0.12), (40, 45, 70), -1, cv2.LINE_AA)
    # pin deck edge line
    rd = int((deck_x0 - x0) * sx)
    bgr[max(rd - 1, 0):rd + 1] = (60, 70, 95)
    return np.ascontiguousarray(bgr[..., ::-1])
