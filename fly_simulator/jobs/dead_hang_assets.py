"""Procedural meshes and textures for the dead-hang job (visual only).

Same approach as ``kebab_assets``: everything is generated in code (numpy + OpenCV)
and handed to MjSpec directly; nothing is written to disk. Nothing here collides or
has mass (callers add the meshes with ``contact_kwargs("visual")`` and ``mass=0``).

* ``trap_lobe``: one Venus-flytrap lobe as a closed thin slab in its hinge frame
  (midrib along x, lobe rising along +z, cupped toward +y). The inner face maps to
  the top half of ``trap_texture`` (red, with a green margin near the rim), the outer
  face to the bottom half (green). ``mirror_y`` gives the other lobe.
* ``trap_cilia``: the marginal "teeth" (tapered spikes along the rim), one mesh.
* ``petiole``: the flat, winged leaf stalk under a trap.
* textures: trap (red / green), terracotta pot, soil with moss, black rubber gym
  mat, painted brick wall, motivational posters (``poster_texture``, cv2 text).
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.kebab_assets import (  # noqa: F401  (re-exported helpers)
    MeshData, _mix, _smooth, _to_u8, add_mesh, add_texture, add_textured_material, fbm, lathe,
    tile_noise, transform, tube,
)

TAU = 2.0 * math.pi


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def _lobe_surface(L: float, W: float, B: float, na: int, nb: int, flare: float = 0.12):
    """Inner surface grid (na, nb, 3) of a lobe in its hinge frame, and the (a, b)
    parameters. a in [-1, 1] along the midrib (x = L a), b in [0, 1] hinge -> rim."""
    a = np.linspace(-1.0, 1.0, na)
    b = np.linspace(0.0, 1.0, nb)
    A, Bm = np.meshgrid(a, b, indexing="ij")
    shape = np.sqrt(np.clip(1.0 - A ** 2, 0.0, 1.0))
    x = L * A * (1.0 - 0.08 * Bm)
    z = W * Bm * (0.35 + 0.65 * shape)
    # cupped toward +y, back toward the rim so two closed lobes meet there; a slight
    # outward flare of the margin
    y = B * np.sin(np.pi * np.clip(Bm, 0, 1)) ** 0.85 * (0.25 + 0.75 * shape)
    y = y + flare * B * _smooth(0.8, 1.0, Bm) * shape
    return np.stack([x, y, z], -1), A, Bm


def trap_lobe(L: float = 2.5, W: float = 2.3, B: float = 1.05, thick: float = 0.05,
              na: int = 25, nb: int = 12) -> MeshData:
    """Closed slab lobe (see module doc). Inner face uv v in [0.02, 0.48], outer face
    v in [0.52, 0.98] (u runs along the midrib)."""
    P, A, Bm = _lobe_surface(L, W, B, na, nb)
    # outer sheet: offset along +y (the cup's outside), thinning to the rim
    Q = P.copy()
    Q[..., 1] += thick * (1.0 - 0.4 * Bm)
    u = (A + 1.0) / 2.0
    uv_in = np.stack([u, 0.02 + 0.46 * Bm], -1)
    uv_out = np.stack([u, 0.98 - 0.46 * Bm], -1)
    verts = np.concatenate([P.reshape(-1, 3), Q.reshape(-1, 3)])
    uv = np.concatenate([uv_in.reshape(-1, 2), uv_out.reshape(-1, 2)])
    n = na * nb
    idx = np.arange(n).reshape(na, nb)
    faces = []
    for i in range(na - 1):
        for j in range(nb - 1):
            a0, a1, b0, b1 = idx[i, j], idx[i + 1, j], idx[i, j + 1], idx[i + 1, j + 1]
            faces += [(a0, b1, a1), (a0, b0, b1)]  # inner (orientation fixed below)
            faces += [(n + a0, n + a1, n + b1), (n + a0, n + b1, n + b0)]
    # boundary loop (j = 0 row, i = na-1 column, j = nb-1 row back, i = 0 column back)
    loop = ([idx[i, 0] for i in range(na)] + [idx[na - 1, j] for j in range(1, nb)]
            + [idx[i, nb - 1] for i in range(na - 2, -1, -1)]
            + [idx[0, j] for j in range(nb - 2, 0, -1)])
    for k in range(len(loop)):
        p0, p1 = loop[k], loop[(k + 1) % len(loop)]
        faces += [(p0, p1, n + p1), (p0, n + p1, n + p0)]
    F = np.asarray(faces, dtype=np.int64)
    # orient every triangle away from the slab's mid surface
    V = verts
    tri = V[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cen = tri.mean(1)
    mid = 0.5 * (P + Q).reshape(-1, 3)
    mid_c = mid.mean(0)
    out = np.empty_like(cen)
    is_in = np.all(F < n, 1)
    is_out = np.all(F >= n, 1)
    out[is_in] = np.array([0.0, -1.0, 0.0])
    out[is_out] = np.array([0.0, 1.0, 0.0])
    rim = ~(is_in | is_out)
    out[rim] = cen[rim] - mid_c
    flip = np.einsum("ij,ij->i", nrm, out) < 0
    F[flip] = F[flip][:, ::-1]
    return MeshData(V, F, uv)


def mirror_y(md: MeshData) -> MeshData:
    v = md.verts.copy()
    v[:, 1] *= -1.0
    return MeshData(v, md.faces[:, ::-1].copy(), md.uv)


def _spike(base: np.ndarray, direction: np.ndarray, side: np.ndarray, length: float,
           width: float):
    """Triangular pyramid: 3 base corners around ``base`` + a tip (4 verts, 4 faces)."""
    d = direction / np.linalg.norm(direction)
    s = side - (side @ d) * d
    s /= np.linalg.norm(s)
    t = np.cross(d, s)
    corners = [base + width * (math.cos(k * TAU / 3) * s + math.sin(k * TAU / 3) * t)
               for k in range(3)]
    tip = base + length * d
    return np.array(corners + [tip])


def trap_cilia(L: float = 2.5, W: float = 2.3, B: float = 1.05, n: int = 19,
               length: float = 0.85, width: float = 0.028, seed: int = 0) -> MeshData:
    """Marginal teeth along the rim of ``trap_lobe`` (same frame), one mesh of
    tapered spikes continuing the lobe surface outward / upward."""
    rng = np.random.default_rng(seed + 71)
    P, A, Bm = _lobe_surface(L, W, B, 81, 24)
    rim = P[:, -1]
    tang = P[:, -1] - P[:, -3]  # rim direction of the surface (hinge -> rim)
    ai = np.linspace(4, 76, n).round().astype(int)
    verts, faces, uv = [], [], []
    for k, i in enumerate(ai):
        a = A[i, 0]
        d = tang[i] / np.linalg.norm(tang[i]) + np.array([0.0, 0.55, 0.25])
        d += rng.normal(0, 0.08, 3)
        ln = length * (1.0 - 0.45 * a ** 2) * rng.uniform(0.85, 1.1)
        side = np.array([1.0, 0.0, 0.0])
        v = _spike(rim[i], d, side, ln, width)
        o = 4 * k
        verts.append(v)
        faces += [(o, o + 1, o + 3), (o + 1, o + 2, o + 3), (o + 2, o, o + 3), (o, o + 2, o + 1)]
        uv += [(0.1, 0.1), (0.1, 0.1), (0.1, 0.1), (0.9, 0.9)]
    V = np.concatenate(verts)
    F = np.asarray(faces, dtype=np.int64)
    # outward orientation per spike (away from its centroid)
    tri = V[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cen = V.reshape(-1, 4, 3).mean(1)[np.arange(len(F)) // 4]
    flip = np.einsum("ij,ij->i", nrm, tri.mean(1) - cen) < 0
    F[flip] = F[flip][:, ::-1]
    return MeshData(V, F, np.asarray(uv, float))


def petiole(p0, p1, bend: float = 0.6, w0: float = 0.12, w1: float = 0.55, thick: float = 0.06,
            n: int = 16) -> MeshData:
    """Flat winged leaf stalk from ``p0`` (soil) to ``p1`` (under the trap): a slab
    along a curve that bows up, widening from ``w0`` to ``w1`` (half widths) then
    narrowing just under the trap."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0.0, 1.0, n)
    horiz = p1 - p0
    horiz[2] = 0.0
    hn = np.linalg.norm(horiz)
    hdir = horiz / hn if hn > 1e-9 else np.array([1.0, 0.0, 0.0])
    side = np.cross(np.array([0.0, 0.0, 1.0]), hdir)
    if np.linalg.norm(side) < 1e-9:
        side = np.array([0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    ctr = p0[None] + t[:, None] * (p1 - p0)[None]
    ctr[:, 2] += bend * np.sin(np.pi * t) * 0.3
    w = w0 + (w1 - w0) * np.sin(np.pi * np.clip(t * 1.15, 0, 1)) ** 0.7
    w[-1] = w0 * 0.8
    tang = np.gradient(ctr, axis=0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    up = np.cross(side[None], tang)
    rings = []
    for k in range(n):
        s, u = side * w[k], up[k] * thick / 2
        rings.append([ctr[k] + s + u, ctr[k] - s + u, ctr[k] - s - u, ctr[k] + s - u])
    rings = np.asarray(rings)
    uvr = np.stack([np.tile([0.0, 1.0, 1.0, 0.0], (n, 1)),
                    np.repeat(t[:, None], 4, 1)], -1)
    return tube(rings, uvr)


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def trap_texture(seed: int = 0, h: int = 256, w: int = 256) -> np.ndarray:
    """Top half: lobe interior (crimson, darker glands, lighter veins radiating from
    the midrib, green margin at the rim). Bottom half: lobe exterior (leaf green,
    veins). Row 0 = v 0; the inner face uses v 0.02..0.48 (hinge -> rim)."""
    rng = np.random.default_rng(seed + 901)
    hh = h // 2
    v = np.linspace(0, 1, hh, dtype=np.float32)[:, None]  # hinge (0) -> rim (1)
    u = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    n1 = fbm(hh, w, 6, 6, rng, 4)
    glands = tile_noise(hh, w, 48, 48, rng)
    veins = np.abs(np.sin((u - 0.5) * 38.0 / (0.25 + v))) ** 12
    crimson = np.array([0.62, 0.07, 0.10], np.float32)
    deep = np.array([0.40, 0.03, 0.07], np.float32)
    inner = _mix(crimson * np.ones((hh, w, 1), np.float32), deep, 0.6 * n1)
    inner = _mix(inner, (0.28, 0.02, 0.05), _smooth(0.62, 0.8, glands) * 0.8)
    inner = _mix(inner, (0.85, 0.35, 0.30), veins * 0.25 * (1 - v))
    margin = _smooth(0.78, 0.92, v * np.ones_like(u))
    inner = _mix(inner, (0.45, 0.62, 0.18), margin)
    inner = _mix(inner, (0.72, 0.80, 0.30), _smooth(0.94, 1.0, v * np.ones_like(u)) * 0.7)
    n2 = fbm(hh, w, 5, 5, rng, 4)
    outer = _mix(np.array([0.22, 0.46, 0.12], np.float32) * np.ones((hh, w, 1), np.float32),
                 (0.36, 0.58, 0.16), n2)
    vein_o = np.abs(np.sin((u - 0.5) * 30.0 / (0.3 + (1 - v)))) ** 14
    outer = _mix(outer, (0.52, 0.70, 0.30), vein_o * 0.35)
    img = np.concatenate([inner, outer[::-1]], 0)
    return _to_u8(img)


@lru_cache(maxsize=2)
def cilia_texture(n: int = 64) -> np.ndarray:
    """Base (uv 0.1) yellow-green, tip (uv 0.9) reddish."""
    t = np.linspace(0, 1, n, dtype=np.float32)
    diag = _smooth(0.3, 1.7, (t[:, None] + t[None, :]))
    img = _mix(np.array([0.55, 0.70, 0.22], np.float32) * np.ones((n, n, 1), np.float32),
               (0.85, 0.28, 0.16), diag)
    return _to_u8(img)


@lru_cache(maxsize=2)
def terracotta_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed + 902)
    n1 = fbm(n, n, 4, 8, rng, 4)
    spk = tile_noise(n, n, 64, 64, rng)
    base = _mix(np.array([0.66, 0.33, 0.20], np.float32) * np.ones((n, n, 1), np.float32),
                (0.78, 0.44, 0.28), n1)
    base = _mix(base, (0.50, 0.25, 0.16), _smooth(0.7, 0.85, spk) * 0.5)
    rings = 0.5 + 0.5 * np.sin(np.linspace(0, 40 * np.pi, n, dtype=np.float32))[:, None]
    base = base * (0.96 + 0.04 * rings[..., None])
    return _to_u8(base)


@lru_cache(maxsize=2)
def soil_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Dark peat with sphagnum-moss patches and grit."""
    rng = np.random.default_rng(seed + 903)
    n1 = fbm(n, n, 6, 6, rng, 5)
    moss = fbm(n, n, 3, 3, rng, 4)
    grit = tile_noise(n, n, 96, 96, rng)
    base = _mix(np.array([0.16, 0.10, 0.06], np.float32) * np.ones((n, n, 1), np.float32),
                (0.26, 0.17, 0.10), n1)
    base = _mix(base, (0.36, 0.52, 0.18), _smooth(0.55, 0.7, moss) * 0.9)
    base = _mix(base, (0.55, 0.50, 0.42), _smooth(0.82, 0.9, grit) * 0.6)
    return _to_u8(base)


@lru_cache(maxsize=2)
def gym_floor_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Black rubber gym tiles with coloured EPDM speckles and seams (2 x 2 tiles)."""
    rng = np.random.default_rng(seed + 904)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / (n / 2)
    fx, fy = x - np.floor(x), y - np.floor(y)
    edge = np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy))
    seam = _smooth(0.012, 0.004, edge)
    n1 = fbm(n, n, 8, 8, rng, 3)
    base = (0.09 + 0.03 * n1)[..., None] * np.ones((1, 1, 3), np.float32)
    spk = rng.random((n, n)).astype(np.float32)
    cols = np.array([[0.55, 0.12, 0.10], [0.20, 0.30, 0.55], [0.55, 0.55, 0.55]], np.float32)
    which = rng.integers(0, 3, (n, n))
    mask = (spk > 0.985).astype(np.float32)
    base = base * (1 - mask[..., None]) + cols[which] * mask[..., None] * 0.8
    base = _mix(base, (0.02, 0.02, 0.02), seam)
    return _to_u8(base)


@lru_cache(maxsize=2)
def brick_texture(seed: int = 0, h: int = 256, w: int = 256) -> np.ndarray:
    """Painted brick (gym wall): 8 courses x 4 bricks, offset rows, pale mortar."""
    rng = np.random.default_rng(seed + 905)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    rows = 8
    y = y / (h / rows)
    x = x / (w / 4) + 0.5 * (np.floor(y) % 2)
    fx, fy = x - np.floor(x), y - np.floor(y)
    edge = np.minimum(np.minimum(fx, 1 - fx) * 2.0, np.minimum(fy, 1 - fy))
    mortar = _smooth(0.07, 0.035, edge)
    n1 = fbm(h, w, 8, 8, rng, 4)
    bid = (np.floor(x) * 7 + np.floor(y) * 13) % 5
    shade = np.array([0.96, 1.0, 0.93, 1.03, 0.98], np.float32)[bid.astype(int)]
    base = (np.array([0.34, 0.36, 0.40], np.float32) * (0.9 + 0.2 * n1)[..., None]
            * shade[..., None])
    base = _mix(base, (0.62, 0.62, 0.60), mortar)
    return _to_u8(base)


@lru_cache(maxsize=4)
def poster_texture(lines: tuple[str, ...], bg=(0.05, 0.05, 0.06), fg=(0.95, 0.80, 0.15),
                   accent=(0.85, 0.12, 0.10), h: int = 256, w: int = 192) -> np.ndarray:
    """A gym poster: big text lines (cv2 Hershey font), the last line in ``accent``,
    a border. Returned as RGB uint8, row 0 = top of the poster (flip for MuJoCo's
    v = 0 at the bottom is done by the caller's uv / box face)."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (np.asarray(bg) * 255).astype(np.uint8)
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), tuple(int(c * 255) for c in fg), 3)
    n = len(lines)
    for i, text in enumerate(lines):
        col = accent if i == n - 1 else fg
        scale = 1.0
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
        scale = min(1.6, (w - 30) / max(tw, 1))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
        y0 = int((i + 1) * h / (n + 1) + th / 2)
        cv2.putText(img, text, ((w - tw) // 2, y0), cv2.FONT_HERSHEY_DUPLEX, scale,
                    tuple(int(c * 255) for c in col), 2, cv2.LINE_AA)
    return img
