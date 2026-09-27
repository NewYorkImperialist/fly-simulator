"""Procedural meshes and textures for the taste tester job (visual only).

Everything is generated in code (numpy + OpenCV) and handed to MjSpec directly, as in
``kebab_assets`` / ``broccoli_toss_assets`` (whose mesh / texture helpers are
reused). Nothing here collides or has mass. Text on the signs, cards and bins is
drawn with OpenCV's Hershey fonts; no logos, no real names.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.kebab_assets import (MeshData, _mix, _smooth, _to_u8, fbm, lathe,
                                             tile_noise)

TAU = 2.0 * math.pi

KINDS = ("sugar", "bitter", "mixed", "water")
KIND_TEXT = {"sugar": "SUGAR", "bitter": "BITTER", "mixed": "MIXED", "water": "WATER"}
KIND_RGB = {"sugar": (0.95, 0.66, 0.12), "bitter": (0.16, 0.36, 0.12),
            "mixed": (0.62, 0.52, 0.16), "water": (0.55, 0.78, 0.95)}
STAMPS = ("blank", "approved", "rejected")
GREEN = (0.10, 0.62, 0.22)
RED = (0.82, 0.10, 0.10)


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def dish_mesh(r: float, h: float, wall: float = 0.025) -> MeshData:
    """A small shallow sample dish (surface of revolution, base at z = 0)."""
    rr = np.array([0.0, r - 0.02, r, r, r - wall, r - wall, 0.0])
    zz = np.array([0.0, 0.0, 0.015, h, h, 0.02, 0.02])
    return lathe(rr, zz, 36, v_coord=np.linspace(0, 1, len(rr)))


def drop_mesh(r: float, h: float, seed: int = 0) -> MeshData:
    """A sessile drop: a flattened dome (base at z = 0) with a slightly wobbly rim."""
    ph = np.linspace(0.0, math.pi / 2, 10)
    # profile: base centre -> rim -> up over the dome -> top
    prof_r = np.r_[0.0, r, r * np.cos(ph[1:-1]), 0.0]
    prof_z = np.r_[0.0, 0.0, h * np.sin(ph[1:-1]) ** 0.8, h]
    rng = np.random.default_rng(seed)
    a1, p1 = rng.uniform(0.02, 0.05), rng.uniform(0, TAU)

    def wobble(th, z):
        return a1 * np.sin(3 * th + p1) * (1.0 - z / max(h, 1e-6))
    return lathe(prof_r, prof_z, 32, radial_noise=lambda th, z: wobble(th, z) * r,
                 v_coord=np.linspace(0, 1, len(prof_r)))


def flat_quad_mesh(hx: float, hy: float, t: float = 0.004) -> MeshData:
    """Thin box whose *top* face (+z) carries the whole texture: u runs along +y,
    image row 0 (v = 0) at -x. With a camera on the +x side looking toward -x, text
    reads left to right along +y, upright."""
    x0, x1, y0, y1, z0, z1 = -hx, hx, -hy, hy, -t / 2, t / 2
    v = np.array([[x0, y0, z1], [x0, y1, z1], [x1, y1, z1], [x1, y0, z1],
                  [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0]], float)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [1, 1], [0, 1]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    md = MeshData(v, f, uv)
    if md.signed_volume() < 0:
        md.faces = md.faces[:, ::-1].copy()
    return md


def front_panel_mesh(hy: float, hz: float, t: float = 0.01) -> MeshData:
    """Thin box in the y-z plane whose +x face carries the texture (for a camera on
    the +x side): u along +y, image row 0 at the top (+z)."""
    x0, x1, y0, y1, z0, z1 = -t / 2, t / 2, -hy, hy, -hz, hz
    v = np.array([[x1, y0, z1], [x1, y1, z1], [x1, y1, z0], [x1, y0, z0],
                  [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [x0, y0, z0]], float)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [1, 1], [0, 1]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    md = MeshData(v, f, uv)
    if md.signed_volume() < 0:
        md.faces = md.faces[:, ::-1].copy()
    return md


def roller_mesh(r: float, half_len: float, n: int = 28) -> MeshData:
    """A cylinder along x (for the belt rollers); u wraps around, so a striped
    texture shows the rotation."""
    th = np.linspace(0, TAU, n + 1)[:-1]
    rings = np.stack([np.stack([np.full(n, x), r * np.cos(th), r * np.sin(th)], -1)
                      for x in (-half_len, half_len)])
    uv = np.zeros(rings.shape[:2] + (2,))
    uv[..., 0] = (th / TAU)[None]
    uv[..., 1] = np.array([0.0, 1.0])[:, None]
    from fly_simulator.jobs.kebab_assets import tube

    return tube(rings, uv)


# ---------------------------------------------------------------------------
# textures (uint8 RGB, row 0 = v 0)
# ---------------------------------------------------------------------------


def _bgr(c) -> tuple[int, int, int]:
    """Float RGB -> an OpenCV colour for our images, which are stored RGB (so the
    tuple stays in RGB order, despite OpenCV's usual BGR)."""
    return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))


def _centered_text(img, text, cy, scale, rgb, thick=2, font=cv2.FONT_HERSHEY_DUPLEX,
                   cx=None) -> None:
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    x = int((img.shape[1] if cx is None else 2 * cx) / 2 - w / 2)
    cv2.putText(img, text, (x, int(cy + h / 2)), font, scale, _bgr(rgb), thick, cv2.LINE_AA)


@lru_cache(maxsize=2)
def lab_floor_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Speckled light-grey lab vinyl tiles (2 x 2 per texture) with thin grout."""
    rng = np.random.default_rng(seed + 501)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / (n / 2)
    fx, fy = x - np.floor(x), y - np.floor(y)
    edge = np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy))
    grout = _smooth(0.012, 0.004, edge)
    speck = (rng.random((n, n)) > 0.985).astype(np.float32)
    speck = cv2.GaussianBlur(speck, (3, 3), 0.6)
    cloud = fbm(n, n, 4, 4, rng, 3)
    tile = (np.floor(x) + 2 * np.floor(y)) % 3
    base = (0.80 + 0.04 * (cloud - 0.5) - 0.02 * tile)[..., None] * np.array([0.93, 0.95, 0.96],
                                                                           np.float32)
    base = _mix(base, (0.35, 0.37, 0.40), np.clip(speck * 1.5, 0, 1) * 0.6)
    base = _mix(base, (0.58, 0.60, 0.62), grout)
    return _to_u8(base)


@lru_cache(maxsize=2)
def wall_texture(seed: int = 0, h: int = 256, w: int = 512) -> np.ndarray:
    """Painted lab wall (pale mint) with a darker dado band and a steel chair rail."""
    rng = np.random.default_rng(seed + 502)
    n = fbm(h, w, 6, 12, rng, 3)
    v = np.linspace(0, 1, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    img = (0.86 + 0.03 * (n - 0.5))[..., None] * np.array([0.86, 0.95, 0.92], np.float32)
    img = _mix(img, (0.36, 0.52, 0.50), (v > 0.72).astype(np.float32))
    img = _mix(img, (0.70, 0.72, 0.74), ((v > 0.69) & (v < 0.72)).astype(np.float32))
    return _to_u8(img)


@lru_cache(maxsize=2)
def belt_texture(seed: int = 0, n: int = 128) -> np.ndarray:
    """Dark rubber belt: fine grain plus a faint diamond tread."""
    rng = np.random.default_rng(seed + 503)
    g = tile_noise(n, n, n // 2, n // 2, rng)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    tread = 0.5 + 0.5 * np.sin(TAU * 8 * (x + y)) * np.sin(TAU * 8 * (x - y))
    v = 0.13 + 0.035 * (g - 0.5) + 0.02 * tread
    return _to_u8(np.stack([v, v * 1.02, v * 1.06], -1))


@lru_cache(maxsize=2)
def roller_texture(n: int = 64) -> np.ndarray:
    """Steel roller with 6 dark stripes around it (so its turning shows)."""
    u = np.linspace(0, 1, n, dtype=np.float32)[None, :] * np.ones((n, 1), np.float32)
    stripe = (np.sin(TAU * 6 * u) > 0.6).astype(np.float32)
    v = 0.72 - 0.45 * stripe
    return _to_u8(np.stack([v, v * 1.01, v * 1.04], -1))


@lru_cache(maxsize=2)
def marbled_texture(seed: int = 0, n: int = 128) -> np.ndarray:
    """Golden syrup swirled with dark-green bitter extract (the mixed samples)."""
    rng = np.random.default_rng(seed + 504)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    turb = fbm(n, n, 4, 4, rng, 4)
    s = 0.5 + 0.5 * np.sin(TAU * (3.0 * x + 2.0 * y + 2.5 * turb))
    gold = np.array(KIND_RGB["sugar"], np.float32)
    green = np.array((0.14, 0.34, 0.10), np.float32)
    img = _mix(np.broadcast_to(gold, (n, n, 3)).copy(), green, _smooth(0.35, 0.65, s))
    return _to_u8(img)


@lru_cache(maxsize=24)
def card_texture(kind: str, stamp: str = "blank", h: int = 96, w: int = 128) -> np.ndarray:
    """A sample card: a colour band with the sample type, a few ruled lines, and
    (after the stamper) a tilted APPROVED / REJECTED ink stamp."""
    img = np.full((h, w, 3), (0.97, 0.96, 0.92), np.float32)
    col = KIND_RGB[kind]
    img[: h // 4] = col
    txt = KIND_TEXT[kind]
    u8 = _to_u8(img)
    _centered_text(u8, txt, h // 8, 0.62, (1, 1, 1) if kind in ("bitter",) else (0.08, 0.08, 0.08), 2)
    for k in range(3):
        yy = int(h * (0.45 + 0.17 * k))
        cv2.line(u8, (10, yy), (w - 10, yy), _bgr((0.62, 0.66, 0.74)), 1, cv2.LINE_AA)
    if stamp != "blank":
        ink = GREEN if stamp == "approved" else RED
        word = "APPROVED" if stamp == "approved" else "REJECTED"
        layer = np.zeros_like(u8)
        cv2.rectangle(layer, (8, int(h * 0.36)), (w - 8, int(h * 0.92)), (255, 255, 255), 3)
        _centered_text(layer, word, int(h * 0.64), 0.72, (1, 1, 1), 2)
        M = cv2.getRotationMatrix2D((w / 2, h * 0.64), -8.0 if stamp == "approved" else 7.0, 1.0)
        layer = cv2.warpAffine(layer, M, (w, h))
        a = (layer[..., 0:1].astype(np.float32) / 255.0) * 0.9
        rng = np.random.default_rng(len(word) + len(kind))
        a *= (0.75 + 0.25 * rng.random((h, w, 1))).astype(np.float32)  # uneven ink
        u8 = (u8 * (1 - a) + np.array(ink, np.float32) * 255 * a).astype(np.uint8)
    return u8


@lru_cache(maxsize=4)
def bin_label_texture(word: str, rgb: tuple, h: int = 96, w: int = 256) -> np.ndarray:
    img = _to_u8(np.full((h, w, 3), rgb, np.float32))
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), (255, 255, 255), 3)
    _centered_text(img, word, h // 2, 1.25, (1, 1, 1), 3)
    return img


@lru_cache(maxsize=2)
def sign_texture(h: int = 160, w: int = 640) -> np.ndarray:
    img = _to_u8(np.full((h, w, 3), (0.12, 0.20, 0.30), np.float32))
    cv2.rectangle(img, (8, 8), (w - 9, h - 9), _bgr((0.95, 0.78, 0.15)), 4)
    _centered_text(img, "QUALITY CONTROL", int(h * 0.36), 1.9, (1, 1, 1), 3)
    _centered_text(img, "TASTE TEST STATION 1  -  EVERY SAMPLE, FOREVER", int(h * 0.72), 0.85,
                   (0.95, 0.78, 0.15), 2)
    return img


@lru_cache(maxsize=2)
def tally_texture(h: int = 192, w: int = 384) -> np.ndarray:
    """Tally board face: dark panel, 'APPROVED' / 'REJECTED' captions under the two
    3-digit displays (the digits are separate emissive segments)."""
    img = _to_u8(np.full((h, w, 3), (0.06, 0.07, 0.08), np.float32))
    cv2.rectangle(img, (4, 4), (w - 5, h - 5), _bgr((0.5, 0.52, 0.55)), 3)
    _centered_text(img, "APPROVED", int(h * 0.86), 0.9, GREEN, 2, cx=w // 4)
    _centered_text(img, "REJECTED", int(h * 0.86), 0.9, (0.95, 0.2, 0.2), 2, cx=3 * w // 4)
    return img


@lru_cache(maxsize=2)
def gauge_texture(max_hz: float = 100.0, thr_hz: float = 30.0, h: int = 512, w: int = 192) -> np.ndarray:
    """The MN9 meter's scale: 0 .. max_hz bottom to top, the approve threshold line."""
    img = _to_u8(np.full((h, w, 3), (0.94, 0.94, 0.90), np.float32))
    top, bot = int(h * 0.12), int(h * 0.95)
    for k in range(0, int(max_hz) + 1, 10):
        y = int(bot - (bot - top) * k / max_hz)
        L = 40 if k % 50 == 0 else 22
        cv2.line(img, (w - 8 - L, y), (w - 8, y), (40, 40, 40), 2)
        if k % 20 == 0:
            cv2.putText(img, f"{k}", (8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2,
                        cv2.LINE_AA)
    yt = int(bot - (bot - top) * thr_hz / max_hz)
    cv2.line(img, (4, yt), (w - 4, yt), _bgr(GREEN), 4)
    _centered_text(img, "MN9 Hz", int(h * 0.05), 0.95, (0.1, 0.1, 0.1), 2)
    cv2.putText(img, "PASS", (60, yt - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _bgr(GREEN), 2,
                cv2.LINE_AA)
    return img


@lru_cache(maxsize=2)
def clipboard_texture(h: int = 320, w: int = 240) -> np.ndarray:
    """A QC log sheet on a clipboard (generic rows, ticks and crosses)."""
    img = _to_u8(np.full((h, w, 3), (0.97, 0.97, 0.94), np.float32))
    _centered_text(img, "TASTE QC LOG", 26, 0.75, (0.1, 0.1, 0.1), 2)
    rng = np.random.default_rng(7)
    for r in range(9):
        y = 60 + r * 28
        cv2.line(img, (12, y + 10), (w - 12, y + 10), _bgr((0.6, 0.66, 0.78)), 1)
        k = KINDS[int(rng.integers(4))]
        cv2.putText(img, KIND_TEXT[k], (14, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1,
                    cv2.LINE_AA)
        ok = k == "sugar"
        c = _bgr(GREEN if ok else RED)
        x0 = w - 50
        if ok:
            cv2.line(img, (x0, y - 2), (x0 + 8, y + 6), c, 2, cv2.LINE_AA)
            cv2.line(img, (x0 + 8, y + 6), (x0 + 22, y - 10), c, 2, cv2.LINE_AA)
        else:
            cv2.line(img, (x0, y - 8), (x0 + 16, y + 6), c, 2, cv2.LINE_AA)
            cv2.line(img, (x0 + 16, y - 8), (x0, y + 6), c, 2, cv2.LINE_AA)
    return img


@lru_cache(maxsize=2)
def poster_texture(h: int = 256, w: int = 192) -> np.ndarray:
    """A generic lab safety poster: 'TASTE WITH YOUR FEET' with a cartoon leg."""
    img = _to_u8(np.full((h, w, 3), (0.98, 0.86, 0.20), np.float32))
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), (20, 20, 20), 4)
    _centered_text(img, "TASTE WITH", 36, 0.8, (0.05, 0.05, 0.05), 2)
    _centered_text(img, "YOUR FEET", 68, 0.8, (0.05, 0.05, 0.05), 2)
    pts = np.array([[60, 110], [95, 150], [88, 205], [130, 222]], np.int32)
    cv2.polylines(img, [pts], False, (20, 20, 20), 7, cv2.LINE_AA)
    cv2.circle(img, (140, 226), 16, _bgr(KIND_RGB["sugar"]), -1, cv2.LINE_AA)
    _centered_text(img, "tarsal GRNs first", int(h * 0.94) - 6, 0.5, (0.05, 0.05, 0.05), 1)
    return img
