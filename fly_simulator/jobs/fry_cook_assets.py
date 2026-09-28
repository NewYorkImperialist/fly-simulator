"""Procedural meshes and textures for the fry cook job (visual only).

Everything is generated in code (numpy + OpenCV) and handed to MjSpec directly, as in
``kebab_assets`` / ``pizza_chef_assets`` (whose helpers are reused); nothing is
written to disk. Nothing here collides or has mass. Text is drawn with OpenCV's
Hershey fonts. The fast-food look is generic: "FLY FRIES" is our own name, the logo is
our own cartoon fly on a yellow disc; no real brand, logo, mascot or trade dress.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.kebab_assets import (MeshData, _mix, _smooth, _to_u8, fbm, lathe,
                                             superellipse_loop, tile_noise, tube)
from fly_simulator.jobs.taste_tester_assets import _bgr, _centered_text

TAU = 2.0 * math.pi
RED = (0.80, 0.08, 0.07)
YELLOW = (1.0, 0.80, 0.12)


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def fry_mesh(L: float, s: float, bend: float = 0.0, seed: int = 0, n_len: int = 9) -> MeshData:
    """One French fry along z (centred): a rounded-square stick of side ``s`` and length
    ``L``, a slight bend and twist, the ends rounded off. u around, v along."""
    rng = np.random.default_rng(seed)
    loop = superellipse_loop(12, 0.35) * (s / 2)
    zs = np.linspace(-L / 2, L / 2, n_len)
    rings, uvs = [], []
    tw = rng.uniform(-0.3, 0.3)
    for k, z in enumerate(zs):
        f = abs(z) / (L / 2)
        sc = 1.0 - 0.45 * _smooth(0.8, 1.0, f)  # rounded ends
        a = tw * z / L
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        p = (loop * sc) @ R.T
        off = bend * (1.0 - (2 * z / L) ** 2)
        ring = np.column_stack([p[:, 0] + off, p[:, 1], np.full(len(p), z)])
        rings.append(ring)
        uvs.append(np.column_stack([np.linspace(0, 1, len(p), endpoint=False),
                                    np.full(len(p), k / (n_len - 1))]))
    return tube(np.stack(rings), np.stack(uvs))


def cup_mesh(hx0: float, hy0: float, hx1: float, hy1: float, h: float, wall: float,
             floor: float, *, rim_wave: float = 0.0, e: float = 0.45, n: int = 40) -> MeshData:
    """An open rectangular cup (fry carton / scoop bowl): rounded-rectangle outline,
    ``hx0 x hy0`` at the base flaring to ``hx1 x hy1`` at the rim (height ``h``; the
    rim is ``rim_wave`` higher at the +-y faces), walls ``wall`` thick, a floor
    ``floor`` thick. Base at z = 0. u runs around (0 at +x, 0.25 at +y, 0.75 at -y),
    v: outer surface 0 (base) .. 0.8 (rim), inside 0.8 .. 1."""
    lp = superellipse_loop(n, e)
    wave = rim_wave * lp[:, 1] ** 2

    def ring(hx, hy, z):
        return np.column_stack([lp[:, 0] * hx, lp[:, 1] * hy, np.broadcast_to(z, n)])
    rings = [ring(hx0, hy0, 0.0), ring(hx1, hy1, h + wave),
             ring(hx1 - wall, hy1 - wall, h + wave - 0.002),
             ring(hx0 - wall, hy0 - wall, floor)]
    u = np.linspace(0, 1, n, endpoint=False)
    vs = (0.0, 0.8, 0.84, 1.0)
    uv = np.stack([np.column_stack([u, np.full(n, v)]) for v in vs])
    return tube(np.stack(rings), uv)


def box_shell_mesh(hx: float, hy: float, h: float, wall: float) -> MeshData:
    """An open-top box shell with square corners (the fryer vat / the holding bin),
    base at z = 0."""
    lp = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]], float)

    def ring(ax, ay, z):
        return np.column_stack([lp[:, 0] * ax, lp[:, 1] * ay, np.full(8, z)])
    rings = [ring(hx, hy, 0.0), ring(hx, hy, h), ring(hx - wall, hy - wall, h),
             ring(hx - wall, hy - wall, wall)]
    u = np.linspace(0, 1, 8, endpoint=False)
    uv = np.stack([np.column_stack([u, np.full(8, v)]) for v in (0.0, 0.8, 0.84, 1.0)])
    return tube(np.stack(rings), uv)


def lamp_hood_mesh(hx: float, hy: float, h: float) -> MeshData:
    """The heat lamp's hood: a flared rectangular shade, open at the bottom (z = 0)."""
    lp = superellipse_loop(32, 0.3)

    def ring(ax, ay, z):
        return np.column_stack([lp[:, 0] * ax, lp[:, 1] * ay, np.full(len(lp), z)])
    t = 0.02
    rings = [ring(hx, hy, 0.0), ring(hx * 0.8, hy * 0.75, 0.6 * h), ring(hx * 0.55, hy * 0.45, h),
             ring(hx * 0.55 - t, hy * 0.45 - t, h - t), ring(hx * 0.8 - t, hy * 0.75 - t, 0.6 * h - t),
             ring(hx - t, hy - t, 0.0)]
    u = np.linspace(0, 1, len(lp), endpoint=False)
    uv = np.stack([np.column_stack([u, np.full(len(lp), v)]) for v in np.linspace(0, 1, 6)])
    return tube(np.stack(rings), uv, cap_start=False, cap_end=False)


def shaker_mesh(r: float, h: float) -> MeshData:
    """A salt shaker body (glass), base at z = 0, closed."""
    rr = np.array([0.0, 0.9 * r, r, 0.96 * r, 0.9 * r, 0.93 * r, 0.0])
    zz = np.array([0.0, 0.0, 0.08 * h, 0.5 * h, 0.9 * h, h, h])
    return lathe(rr, zz, 28, v_coord=np.linspace(0, 1, len(rr)))


def shaker_cap_mesh(r: float, h: float) -> MeshData:
    """The shaker's domed steel cap, base at z = 0."""
    ph = np.linspace(0, math.pi / 2, 7)
    rr = np.r_[0.0, r, r, r * np.cos(ph[1:-1]), 0.0]
    zz = np.r_[0.0, 0.0, 0.35 * h, 0.35 * h + 0.65 * h * np.sin(ph[1:-1]), h]
    return lathe(rr, zz, 28)


def bell_mesh(r: float) -> MeshData:
    """A service bell's dome on its base, base at z = 0."""
    ph = np.linspace(0, math.pi / 2, 9)
    rr = np.r_[0.0, 1.15 * r, 1.15 * r, r * np.cos(ph), 0.0]
    zz = np.r_[0.0, 0.0, 0.08 * r, 0.1 * r + 0.8 * r * np.sin(ph), 0.9 * r]
    return lathe(rr, zz, 32)


# ---------------------------------------------------------------------------
# textures (uint8 RGB, row 0 = v 0)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def counter_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """A brushed stainless counter top: streaks, faint scratches and grease smudges."""
    rng = np.random.default_rng(seed + 1101)
    streak = tile_noise(n, n, n, 6, rng) * 0.6 + tile_noise(n, n, n // 2, 3, rng) * 0.4
    blot = fbm(n, n, 5, 5, rng, 3)
    v = 0.56 + 0.07 * (streak - 0.5) + 0.07 * (blot - 0.5)
    img = np.stack([v * 0.98, v * 1.0, v * 1.03], -1)
    u8 = _to_u8(img)
    for _ in range(30):  # faint scratches
        x0, y0 = rng.integers(0, n, 2)
        a = rng.uniform(-0.4, 0.4)
        L = rng.integers(15, 60)
        x1, y1 = int(x0 + L * math.cos(a)), int(y0 + L * math.sin(a))
        c = int(rng.integers(150, 162))
        cv2.line(u8, (int(x0), int(y0)), (x1, y1), (c, c, c + 4), 1, cv2.LINE_AA)
    return u8


@lru_cache(maxsize=2)
def fry_texture(seed: int = 0, h: int = 64, w: int = 64) -> np.ndarray:
    """Potato stick: pale yellow with fine starch grain and darker specks; the ends
    (v ~ 0, 1) a little browner. The job tints it from raw (pale) to golden."""
    rng = np.random.default_rng(seed + 1102)
    g = fbm(h, w, 8, 4, rng, 3)
    v = np.linspace(0, 1, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    end = _smooth(0.12, 0.0, v) + _smooth(0.88, 1.0, v)
    img = _mix(np.broadcast_to(np.array((1.0, 0.93, 0.62), np.float32), (h, w, 3)).copy(),
               (0.98, 0.84, 0.45), g)
    img = _mix(img, (0.78, 0.52, 0.2), np.clip(end, 0, 1) * 0.8)
    spk = rng.random((h, w)) > 0.985
    img[spk] = (0.7, 0.48, 0.2)
    return _to_u8(img)


@lru_cache(maxsize=2)
def oil_texture(seed: int = 0, n: int = 128) -> np.ndarray:
    """Hot frying oil seen from above: amber with slow swirls and bright glints."""
    rng = np.random.default_rng(seed + 1103)
    g = fbm(n, n, 4, 4, rng, 4)
    img = _mix(np.broadcast_to(np.array((0.62, 0.40, 0.07), np.float32), (n, n, 3)).copy(),
               (0.95, 0.72, 0.22), _smooth(0.3, 0.75, g))
    img = _mix(img, (1.0, 0.95, 0.7), _smooth(0.72, 0.82, g) * 0.6)
    return _to_u8(img)


def _logo(img: np.ndarray, cx: int, cy: int, R: int) -> None:
    """Our own "FLY FRIES" badge: a yellow disc with a cartoon fly (round body, big
    eyes, two wings) holding three fries. Generic, made up here."""
    cv2.circle(img, (cx, cy), R, _bgr(YELLOW), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), R, _bgr((0.55, 0.05, 0.04)), max(1, R // 12), cv2.LINE_AA)
    k = R / 40.0
    # fries fanning out behind the fly
    for a in (-0.35, 0.0, 0.35):
        x0, y0 = cx, int(cy + 6 * k)
        x1, y1 = int(cx + 26 * k * math.sin(a)), int(cy - 30 * k * math.cos(a))
        cv2.line(img, (x0, y0), (x1, y1), _bgr((0.93, 0.6, 0.12)), max(2, int(6 * k)), cv2.LINE_AA)
    # wings
    for s in (-1, 1):
        cv2.ellipse(img, (int(cx + s * 14 * k), int(cy - 2 * k)), (int(12 * k), int(7 * k)),
                    s * 30, 0, 360, _bgr((0.92, 0.95, 1.0)), -1, cv2.LINE_AA)
        cv2.ellipse(img, (int(cx + s * 14 * k), int(cy - 2 * k)), (int(12 * k), int(7 * k)),
                    s * 30, 0, 360, _bgr((0.3, 0.3, 0.35)), 1, cv2.LINE_AA)
    # body and head
    cv2.ellipse(img, (cx, int(cy + 10 * k)), (int(9 * k), int(13 * k)), 0, 0, 360,
                _bgr((0.18, 0.2, 0.25)), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, int(cy - 6 * k)), int(8 * k), _bgr((0.18, 0.2, 0.25)), -1, cv2.LINE_AA)
    for s in (-1, 1):
        cv2.circle(img, (int(cx + s * 6 * k), int(cy - 8 * k)), int(5 * k), _bgr((0.85, 0.12, 0.1)),
                   -1, cv2.LINE_AA)


@lru_cache(maxsize=2)
def carton_texture(h: int = 256, w: int = 512) -> np.ndarray:
    """The fry carton (``cup_mesh`` uv): red outside with a yellow band at the rim and
    the "FLY FRIES" badge on the front (-y, u 0.75) and the back (+y, u 0.25); the
    inside (v > 0.8) white card. Drawn upright, then flipped (v 0 = the base)."""
    img = _to_u8(np.full((h, w, 3), RED, np.float32))
    rng = np.random.default_rng(1104)
    fib = (fbm(h, w, 12, 24, rng, 2) * 14).astype(np.uint8)
    img = cv2.subtract(img, np.stack([fib] * 3, -1))
    top = int(0.2 * h)  # rows [0, top) of the upright image = v 0.8 .. 1 (inside)
    img[:top] = _bgr((0.96, 0.94, 0.9))
    cv2.rectangle(img, (0, top), (w, top + 12), _bgr(YELLOW), -1)
    for uc in (0.25, 0.75):
        cx = int(uc * w)
        _logo(img, cx, int(top + 0.34 * h), int(0.19 * h))
        _centered_text(img, "FLY FRIES", int(top + 0.64 * h), 0.72, (1.0, 0.95, 0.85), 2, cx=cx)
    return np.ascontiguousarray(img[::-1])


@lru_cache(maxsize=2)
def tray_texture(n: int = 256) -> np.ndarray:
    """A serving tray with a printed paper liner (checks, the badge)."""
    img = _to_u8(np.full((n, n, 3), (0.72, 0.1, 0.08), np.float32))
    m = 22
    liner = np.full((n - 2 * m, n - 2 * m, 3), 245, np.uint8)
    y, x = np.mgrid[0:n - 2 * m, 0:n - 2 * m]
    chk = ((x // 16 + y // 16) % 2 == 0)
    liner[chk] = (250, 222, 120)
    img[m:n - m, m:n - m] = liner
    _logo(img, n // 2, n // 2, 44)
    return img


@lru_cache(maxsize=2)
def tile_wall_texture(seed: int = 0, h: int = 256, w: int = 256) -> np.ndarray:
    """Kitchen wall: square white glazed tiles with a red accent row, grey grout
    (4 x 4 tiles per repeat)."""
    rng = np.random.default_rng(seed + 1105)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    y, x = y / (h / 4), x / (w / 4)
    fx, fy = x - np.floor(x), y - np.floor(y)
    edge = np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy))
    grout = _smooth(0.045, 0.02, edge)
    bevel = _smooth(0.1, 0.03, edge) - grout
    n1 = tile_noise(h, w, 8, 8, rng)
    base = (0.86 + 0.05 * n1)[..., None] * np.array([0.97, 0.97, 0.95], np.float32)
    row = np.floor(y).astype(int)
    red = (row == 1)
    base[red] = (np.array([0.78, 0.12, 0.1], np.float32) * (0.9 + 0.1 * n1[red])[:, None])
    base = _mix(base, (0.72, 0.72, 0.72), np.clip(bevel, 0, 1) * 0.4)
    base = _mix(base, (0.5, 0.5, 0.48), grout)
    return _to_u8(base)


@lru_cache(maxsize=2)
def fryer_front_texture(h: int = 192, w: int = 256) -> np.ndarray:
    """The fryer's front panel: brushed steel, a temperature dial, two knobs, a label."""
    rng = np.random.default_rng(1106)
    s = tile_noise(h, w, h, 6, rng)
    img = _to_u8(np.stack([0.6 + 0.06 * s] * 3, -1) * np.array([0.98, 1.0, 1.03], np.float32))
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), (95, 95, 100), 3)
    cx, cy, R = w // 3, int(h * 0.45), int(h * 0.22)
    cv2.circle(img, (cx, cy), R, (245, 245, 240), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), R, (60, 60, 60), 3, cv2.LINE_AA)
    for a in np.linspace(-2.3, 2.3, 9):
        cv2.line(img, (int(cx + 0.78 * R * math.sin(a)), int(cy - 0.78 * R * math.cos(a))),
                 (int(cx + 0.95 * R * math.sin(a)), int(cy - 0.95 * R * math.cos(a))), (40, 40, 40), 2)
    cv2.line(img, (cx, cy), (int(cx + 0.7 * R * math.sin(1.4)), int(cy - 0.7 * R * math.cos(1.4))),
             _bgr(RED), 3, cv2.LINE_AA)
    _centered_text(img, "350F", int(cy + R * 0.45), 0.45, (0.2, 0.2, 0.2), 1, cx=cx)
    for kx in (int(w * 0.66), int(w * 0.84)):
        cv2.circle(img, (kx, cy), int(R * 0.45), (30, 30, 32), -1, cv2.LINE_AA)
        cv2.line(img, (kx, cy), (kx, int(cy - R * 0.4)), (220, 220, 220), 2)
    _centered_text(img, "FRYER 1", int(h * 0.85), 0.8, (0.15, 0.15, 0.16), 2)
    return img


@lru_cache(maxsize=2)
def menu_board_texture(h: int = 320, w: int = 512) -> np.ndarray:
    """A generic backlit menu board."""
    img = _to_u8(np.full((h, w, 3), (0.08, 0.07, 0.08), np.float32))
    cv2.rectangle(img, (0, 0), (w - 1, 58), _bgr(RED), -1)
    _centered_text(img, "FLY FRIES", 30, 1.3, YELLOW, 3)
    rows = (("SMALL FRY", "1.00"), ("MEDIUM FRY", "1.50"), ("LARGE FRY", "2.00"),
            ("SALT", "free"), ("ETERNITY MEAL", "?"))
    for k, (a, b) in enumerate(rows):
        yy = 100 + k * 44
        cv2.putText(img, a, (40, yy), cv2.FONT_HERSHEY_DUPLEX, 0.85, (245, 240, 225), 1, cv2.LINE_AA)
        cv2.putText(img, b, (w - 130, yy), cv2.FONT_HERSHEY_DUPLEX, 0.85, _bgr(YELLOW), 1, cv2.LINE_AA)
    _logo(img, w - 60, h - 40, 30)
    _centered_text(img, "(prices in fly cents)", h - 22, 0.5, (0.7, 0.7, 0.7), 1, cx=w // 2 - 40)
    return img


@lru_cache(maxsize=4)
def order_sign_texture(lit: bool, h: int = 96, w: int = 384) -> np.ndarray:
    """The "ORDER UP" light box above the pass window (off / lit)."""
    bg = (0.12, 0.02, 0.02) if not lit else (0.25, 0.03, 0.02)
    img = _to_u8(np.full((h, w, 3), bg, np.float32))
    col = (0.45, 0.12, 0.1) if not lit else (1.0, 0.85, 0.3)
    if lit:
        glow = np.zeros_like(img)
        _centered_text(glow, "ORDER UP", h // 2, 1.9, (1.0, 0.4, 0.1), 8)
        img = cv2.add(img, cv2.GaussianBlur(glow, (0, 0), 6))
    _centered_text(img, "ORDER UP", h // 2, 1.9, col, 3)
    return img


@lru_cache(maxsize=2)
def ticket_texture(k: int, h: int = 128, w: int = 64) -> np.ndarray:
    """An order ticket on the rail (paper, a few printed lines)."""
    img = _to_u8(np.full((h, w, 3), (0.97, 0.96, 0.9), np.float32))
    cv2.putText(img, f"#{k:02d}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    for j, ln in enumerate(("1 FRY", "SALT", "TO GO" if k % 2 else "HERE")):
        cv2.putText(img, ln, (6, 50 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.line(img, (4, h - 12), (w - 4, h - 12), (150, 150, 150), 1)
    return img


@lru_cache(maxsize=2)
def dining_texture(h: int = 256, w: int = 512) -> np.ndarray:
    """What shows through the pass window: a soft-focus dining room (warm light,
    red booths, a counter, windows)."""
    rng = np.random.default_rng(1107)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = _mix(np.broadcast_to(np.array((0.95, 0.85, 0.65), np.float32), (h, w, 3)).copy(),
               (0.55, 0.35, 0.22), np.broadcast_to(yy[..., 0], (h, w)))
    u8 = _to_u8(img)
    for k in range(4):  # windows
        x0 = 30 + k * 125
        cv2.rectangle(u8, (x0, 30), (x0 + 90, 110), _bgr((0.7, 0.85, 0.95)), -1)
    for k in range(5):  # booths
        x0 = 10 + k * 105
        cv2.rectangle(u8, (x0, 150), (x0 + 80, 215), _bgr(RED), -1)
        cv2.rectangle(u8, (x0 + 15, 170), (x0 + 65, 180), _bgr((0.9, 0.9, 0.85)), -1)
    u8 = cv2.GaussianBlur(u8, (0, 0), 5)
    n = (fbm(h, w, 6, 12, rng, 2) * 18).astype(np.uint8)
    return cv2.add(u8, np.stack([n] * 3, -1))


@lru_cache(maxsize=4)
def label_texture(word: str, bg=(0.97, 0.95, 0.88), ink=(0.15, 0.1, 0.08), h: int = 64,
                  w: int = 192) -> np.ndarray:
    img = _to_u8(np.full((h, w, 3), bg, np.float32))
    cv2.rectangle(img, (3, 3), (w - 4, h - 4), _bgr(ink), 2)
    scale = 1.0 if len(word) <= 6 else 0.7
    _centered_text(img, word, h // 2, scale, ink, 2)
    return img
