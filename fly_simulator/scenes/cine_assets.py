"""Procedural meshes and textures for the ``moon_landing``, ``kitty_hawk`` and
``river_crossing`` scenes (visual only; generated in code, nothing on disk).

All designs are original: a small lunar lander (gold-foil descent stage, a boxy
faceted ascent stage), a fly-sized spacesuit (helmet dome, backpack), a stylised-fly
flag, a wood-and-fabric pusher biplane, a clinker rowboat with oars, ice floes, and
the terrains (a cratered regolith plain, wind-rippled sand dunes).
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import mujoco as mj
import numpy as np

from fly_simulator.jobs.broccoli_toss_assets import merge, sphere_mesh  # noqa: F401
from fly_simulator.jobs.kebab_assets import (  # noqa: F401
    MeshData, _mix, _smooth, _to_u8, add_texture, add_textured_material, fbm, lathe, tile_noise, transform, tube,
)
from fly_simulator.scenes.temple_standoff_assets import BodyProbe, box_mesh, cylinder_mesh, shell_mesh  # noqa: F401

TAU = 2.0 * math.pi


def add_mesh(spec, name: str, md: MeshData, open_surface: bool = False):
    """A visual mesh. ``open_surface``: not closed (a terrain sheet), so its inertia
    is computed as a shell."""
    mesh = spec.add_mesh(name=name, uservert=md.verts.ravel().tolist(), userface=md.faces.ravel().tolist(),
                         usertexcoord=md.uv.ravel().tolist(), maxhullvert=16)
    if open_surface:
        mesh.inertia = mj.mjtMeshInertia.mjMESH_INERTIA_SHELL
    return mesh


def make_probe():
    """An unposed unit NeuroMechFly for shrink-wrapping attire (``BodyProbe``)."""
    from flygym.compose import NeuroMechFly

    pf = NeuroMechFly(name="probe")
    ps = pf.mjcf_root.copy()
    ps.compiler.fusestatic = False
    pm = ps.compile()
    pd = mj.MjData(pm)
    return BodyProbe(pm, pd)


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def grid_mesh(xs: np.ndarray, ys: np.ndarray, Z: np.ndarray, tile: float) -> MeshData:
    """An open height-sheet over the grid xs (C) x ys (R), Z (R, C); upward faces."""
    X, Y = np.meshgrid(xs, ys)
    verts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    uv = np.stack([X / tile, Y / tile], -1).reshape(-1, 2)
    R, C = Z.shape
    idx = np.arange(R * C).reshape(R, C)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    faces = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    return MeshData(verts, faces.astype(np.int64), uv)


def lumpy_mesh(radius, rng: np.random.Generator, n_theta: int = 14, n_phi: int = 9, amp: float = 0.25,
               squash: float = 1.0, flat_bottom: float | None = None) -> MeshData:
    """A lumpy rock / ice chunk: a sphere with low-frequency radial noise."""
    md = sphere_mesh(1.0, n_theta=n_theta, n_phi=n_phi)
    v = md.verts.copy()
    k = rng.normal(0, 1, (6, 3))
    ph = rng.uniform(0, TAU, 6)
    nz = sum(np.sin(v @ k[i] * 1.3 + ph[i]) for i in range(6)) / 6
    r = 1.0 + amp * nz
    v = v * r[:, None]
    v[:, 2] *= squash
    if flat_bottom is not None:
        v[:, 2] = np.maximum(v[:, 2], flat_bottom)
    rad = np.broadcast_to(np.asarray(radius, float), (3,))
    return MeshData(v * rad, md.faces, md.uv)


def dome_mesh(r: float, cut: float = -0.35, n: int = 32, n_phi: int = 14) -> MeshData:
    """A helmet dome: a sphere cut off below ``cut`` * r (closed shell of revolution)."""
    ph = np.linspace(math.asin(cut), math.pi / 2, n_phi)
    rs = np.concatenate([[0.0], r * np.cos(ph)])
    zs = np.concatenate([[r * math.sin(ph[0])], r * np.sin(ph)])
    return lathe(rs, zs, n)


def ring_mesh(r0: float, r1: float, h: float, n: int = 32) -> MeshData:
    """A flat annulus of thickness h (a collar ring)."""
    return lathe(np.array([r0, r1, r1, r0, r0]), np.array([0.0, 0.0, h, h, 0.0]) - h / 2, n,
                 v_coord=np.linspace(0, 1, 5))


def box_between(lo, hi, tile: float = 4.0) -> tuple[MeshData, np.ndarray]:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return box_mesh(*((hi - lo) / 2), tile=tile), (lo + hi) / 2


def facet_box_mesh(hx: float, hy: float, hz: float, bevel: float) -> MeshData:
    """A box with its four vertical edges chamfered (an octagonal prism, a faceted
    cabin)."""
    b = bevel
    loop = np.array([[hx, -hy + b], [hx, hy - b], [hx - b, hy], [-hx + b, hy], [-hx, hy - b],
                     [-hx, -hy + b], [-hx + b, -hy], [hx - b, -hy]])
    rings = np.stack([np.concatenate([loop, np.full((8, 1), z)], 1) for z in (-hz, hz)])
    uv = np.zeros(rings.shape[:2] + (2,))
    uv[..., 0] = np.linspace(0, 1, 8)[None]
    uv[1, :, 1] = 1.0
    return tube(rings, uv)


def wing_panel_mesh(span: float, chord: float, thick: float, camber: float, n_span: int = 2,
                    n_chord: int = 9) -> MeshData:
    """A thin cambered wing panel: y along the span (centred), x along the chord
    (leading edge at +x), a circular-arc camber; UV (chord, span) in [0, 1]."""
    xs = np.linspace(chord / 2, -chord / 2, n_chord)
    f = (xs + chord / 2) / chord
    zc = camber * 4 * f * (1 - f)
    tk = thick * (0.35 + 0.65 * np.sin(np.pi * np.clip(1 - f, 0, 1)) ** 0.5)
    top = np.stack([xs, zc + tk / 2], 1)
    bot = np.stack([xs[::-1], (zc - tk / 2)[::-1]], 1)
    loop = np.concatenate([top, bot])  # a closed section loop
    ys = np.linspace(-span / 2, span / 2, n_span)
    rings = np.stack([np.stack([loop[:, 0], np.full(len(loop), y), loop[:, 1]], 1) for y in ys])
    uv = np.zeros(rings.shape[:2] + (2,))
    uv[..., 0] = np.concatenate([1 - f, f[::-1]])[None] * 0.999
    uv[..., 1] = ((ys + span / 2) / span)[:, None]
    return tube(rings, uv)


def blade_mesh(L: float, w: float, t: float, twist: float = 0.5) -> MeshData:
    """One propeller blade along +x from the hub (x = 0 .. L), twisted."""
    n = 8
    xs = np.linspace(0.05 * L, L, n)
    loop = np.array([[1, 0], [0.4, 0.5], [-0.4, 0.5], [-1, 0], [-0.4, -0.5], [0.4, -0.5]])
    rings, uv = [], []
    for i, x in enumerate(xs):
        f = i / (n - 1)
        ww = w * (0.55 + 0.9 * math.sin(math.pi * min(f * 0.9 + 0.1, 1.0)) ** 0.8) * (1 - 0.6 * f ** 3)
        a = twist * (1 - f)
        ca, sa = math.cos(a), math.sin(a)
        y = loop[:, 0] * ww / 2
        z = loop[:, 1] * t / 2
        rings.append(np.stack([np.full(len(loop), x), y * ca - z * sa, y * sa + z * ca], 1))
        uv.append(np.stack([np.full(len(loop), f), np.linspace(0, 1, len(loop))], 1))
    return tube(np.array(rings), np.array(uv))


def hull_mesh(L: float, B: float, D: float, n_x: int = 25, n_s: int = 17, sheer: float = 0.25,
              thick: float = 0.06) -> tuple[MeshData, np.ndarray]:
    """A small rowboat hull shell, x along the length (bow at +x), open on top:
    the outer skin from gunwale to keel to gunwale on each station, a thin shell
    (``shell_mesh``). Returns the mesh and the (n_x, 2) gunwale half-breadth / height
    table (for placing oarlocks, the thwarts and the gunwale)."""
    xs = np.linspace(-L / 2, L / 2, n_x)
    f = xs / (L / 2)
    # half breadth: a full midship, a fine bow, a broad transom stern
    half_b = B / 2 * np.where(f > 0, (1 - f ** 2.2) ** 0.62, (1 - 0.42 * (-f) ** 3.0))
    half_b = np.maximum(half_b, 0.015 * B)
    depth = D * (1 - 0.18 * f ** 2) + sheer * D * f ** 2 * (1 + 0.5 * (f > 0))
    keel = -D * (0.08 * f ** 2) + np.where(f > 0.75, D * 0.9 * ((f - 0.75) / 0.25) ** 2.2, 0.0)
    s = np.linspace(-1, 1, n_s)  # -1 port gunwale .. 0 keel .. 1 starboard gunwale
    P = np.zeros((n_x, n_s, 3))
    for i in range(n_x):
        a = np.abs(s)
        # a round-bilge section: y = hb * sin, z from keel to gunwale
        yy = half_b[i] * np.sin(a * np.pi / 2) ** 0.8 * np.sign(s)
        zz = keel[i] + (depth[i] - keel[i]) * (1 - np.cos(a * np.pi / 2) ** 1.4)
        P[i, :, 0] = xs[i]
        P[i, :, 1] = -yy
        P[i, :, 2] = zz
    uu, vv = np.meshgrid(np.linspace(0, 1, n_s), np.linspace(0, 1, n_x))
    uv = np.stack([uu * 1.0, vv * 3.0], -1)
    md = shell_mesh(P, thick, uv)
    return md, np.stack([xs, half_b, depth], 1)


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def regolith_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """Grey lunar regolith: fine speckle, soft mottling, tiny crater pits."""
    rng = np.random.default_rng(seed + 21)
    mott = fbm(n, n, 6, 6, rng, octaves=5)
    fine = rng.random((n, n)).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (0, 0), 0.7)
    v = 0.50 + 0.22 * (mott - 0.5) + 0.22 * (fine - 0.5)
    # tiny pits: a dark bowl with a bright lip (lit from one side)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    for _ in range(90):
        cx, cy = rng.uniform(0, n, 2)
        r = rng.uniform(2.0, 9.0)
        dx = (xx - cx + n / 2) % n - n / 2
        dy = (yy - cy + n / 2) % n - n / 2
        d = np.sqrt(dx * dx + dy * dy) / r
        v -= 0.10 * np.exp(-(d / 0.7) ** 4) * (0.5 + 0.5 * dx / (r + 1e-3))
        v += 0.06 * np.exp(-((d - 1.0) / 0.25) ** 2)
    col = v[..., None] * np.array([1.0, 0.98, 0.95], np.float32)
    return _to_u8(col)


@lru_cache(maxsize=None)
def gold_foil_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Crinkled gold / amber foil blanket: facets of varying brightness, seams."""
    rng = np.random.default_rng(seed + 22)
    cells = rng.random((24, 24)).astype(np.float32)
    big = cv2.resize(cells, (n, n), interpolation=cv2.INTER_NEAREST)
    warp = fbm(n, n, 8, 8, rng, octaves=3)
    shift = (warp * 12).astype(np.int32)
    yy, xx = np.mgrid[0:n, 0:n]
    facets = big[(yy + shift) % n, (xx + shift[::-1]) % n]
    crink = fbm(n, n, 32, 32, rng, octaves=2)
    v = 0.55 + 0.45 * facets + 0.2 * (crink - 0.5)
    seam = (np.abs((yy % (n // 4)) - 2) < 1.5)
    base = np.array([0.95, 0.70, 0.22], np.float32)
    col = v[..., None] * base
    col[seam] *= 0.55
    return _to_u8(col)


@lru_cache(maxsize=None)
def panel_texture(seed: int = 0, n: int = 256, base=(0.80, 0.80, 0.78)) -> np.ndarray:
    """Pale spacecraft panels: a grid of plates with dark seams and rivet dots."""
    rng = np.random.default_rng(seed + 23)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    plate = rng.uniform(0.9, 1.05, (4, 4)).astype(np.float32)
    pv = cv2.resize(plate, (n, n), interpolation=cv2.INTER_NEAREST)
    v = pv + 0.06 * (fbm(n, n, 6, 6, rng, octaves=3) - 0.5)
    seam = (np.minimum((xx * 4) % 1, 1 - (xx * 4) % 1) < 0.012) | (np.minimum((yy * 4) % 1, 1 - (yy * 4) % 1) < 0.012)
    v[seam] *= 0.45
    col = v[..., None] * np.array(base, np.float32)
    return _to_u8(col)


@lru_cache(maxsize=None)
def suit_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """A white suit fabric: a fine ripstop grid, soft creases."""
    rng = np.random.default_rng(seed + 24)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    grid = ((np.abs(np.sin(xx * TAU * 14)) < 0.08) | (np.abs(np.sin(yy * TAU * 14)) < 0.08)).astype(np.float32)
    crease = fbm(n, n, 5, 3, rng, octaves=3)
    v = 0.90 - 0.05 * grid + 0.10 * (crease - 0.5)
    return _to_u8(v[..., None] * np.array([0.97, 0.97, 0.95], np.float32))


@lru_cache(maxsize=None)
def fly_flag_texture(h: int = 256, w: int = 384) -> np.ndarray:
    """An original flag: a deep blue field, a thin gold border, a white roundel with
    a stylised fly (an oval body, a round head, two swept wings, six legs) in dark
    blue. Row 0 at the top."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (104, 42, 22)  # BGR: a deep blue
    b = max(3, h // 40)
    cv2.rectangle(img, (b, b), (w - 1 - b, h - 1 - b), (70, 175, 225), b)
    cx, cy, R = w // 2, h // 2, int(0.34 * h)
    cv2.circle(img, (cx, cy), R, (245, 245, 245), -1, cv2.LINE_AA)
    ink = (104, 42, 22)
    s = R / 100.0
    # wings (swept back, a little translucent-looking: two tones)
    for sgn in (-1, 1):
        pts = np.array([[cx + sgn * 8 * s, cy - 6 * s], [cx + sgn * 62 * s, cy + 22 * s],
                        [cx + sgn * 58 * s, cy + 40 * s], [cx + sgn * 10 * s, cy + 14 * s]], np.int32)
        cv2.fillPoly(img, [pts], (170, 120, 90), cv2.LINE_AA)
    # legs
    for sgn in (-1, 1):
        for k, (dx, dy) in enumerate(((40, -30), (46, 2), (40, 34))):
            p0 = (int(cx + sgn * 6 * s), int(cy + (k - 1) * 10 * s))
            p1 = (int(cx + sgn * dx * s), int(cy + dy * s))
            cv2.line(img, p0, p1, ink, max(1, int(4 * s)), cv2.LINE_AA)
    cv2.ellipse(img, (cx, int(cy + 12 * s)), (int(14 * s), int(36 * s)), 0, 0, 360, ink, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, int(cy - 30 * s)), int(15 * s), ink, -1, cv2.LINE_AA)
    for sgn in (-1, 1):
        cv2.circle(img, (int(cx + sgn * 9 * s), int(cy - 32 * s)), int(6 * s), (60, 60, 200), -1, cv2.LINE_AA)
    return img[..., ::-1].copy()


@lru_cache(maxsize=None)
def earth_texture(seed: int = 0, h: int = 256, w: int = 512) -> np.ndarray:
    """A procedural blue planet (not a map of any real one): deep blue oceans,
    random green-brown land masses, white cloud bands and swirls, polar caps."""
    rng = np.random.default_rng(seed + 25)
    land = fbm(h, w, 4, 8, rng, octaves=5)
    cloud = fbm(h, w, 6, 12, rng, octaves=5)
    lat = np.linspace(-1, 1, h)[:, None] * np.ones((1, w))
    ocean = np.array([0.06, 0.20, 0.52], np.float32)
    col = np.ones((h, w, 3), np.float32) * ocean
    col = col * (0.8 + 0.4 * fbm(h, w, 8, 16, rng, octaves=2))[..., None]
    lm = _smooth(0.55, 0.60, land)
    ground = _mix(np.array([0.30, 0.42, 0.20], np.float32) * np.ones((h, w, 3), np.float32),
                  np.array([0.55, 0.45, 0.30], np.float32), _smooth(0.6, 0.8, land))
    col = col * (1 - lm[..., None]) + ground * lm[..., None]
    band = 0.5 + 0.5 * np.sin(lat * 9.0 + 3.0 * cloud)
    cl = _smooth(0.52, 0.72, cloud * 0.75 + 0.25 * band)
    polar = _smooth(0.78, 0.9, np.abs(lat))
    cl = np.maximum(cl, polar)
    col = col * (1 - cl[..., None]) + np.array([0.95, 0.96, 0.98], np.float32) * cl[..., None]
    return _to_u8(col)


@lru_cache(maxsize=None)
def sand_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """Pale beach / dune sand: wind ripples, grain speckle, darker damp mottling."""
    rng = np.random.default_rng(seed + 31)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    warp = fbm(n, n, 3, 3, rng, octaves=3)
    rip = 0.5 + 0.5 * np.sin(TAU * (xx * 18 + 0.9 * warp + 0.15 * yy))
    rip = rip ** 1.6
    grain = cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), 0.6)
    mott = fbm(n, n, 5, 5, rng, octaves=4)
    v = 0.74 + 0.10 * (rip - 0.5) + 0.16 * (grain - 0.5) + 0.12 * (mott - 0.5)
    col = v[..., None] * np.array([0.93, 0.86, 0.72], np.float32)
    return _to_u8(col)


@lru_cache(maxsize=None)
def wood_texture(seed: int = 0, n: int = 256, base=(0.55, 0.40, 0.25), rings: float = 30.0) -> np.ndarray:
    """Plain wood grain along u."""
    rng = np.random.default_rng(seed + 32)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    warp = fbm(n, n, 2, 6, rng, octaves=3)
    g = 0.5 + 0.5 * np.sin(TAU * (yy * rings + 2.0 * warp))
    streak = fbm(n, n, 1, 32, rng, octaves=2)
    v = 0.80 + 0.14 * g ** 3 + 0.12 * (streak - 0.5)
    return _to_u8(v[..., None] * np.array(base, np.float32))


@lru_cache(maxsize=None)
def fabric_wing_texture(seed: int = 0, n: int = 256, ribs: int = 10) -> np.ndarray:
    """Doped muslin over ribs: a pale cloth with darker rib lines across the chord
    (u = chord, v = span) and a slightly darker leading-edge band."""
    rng = np.random.default_rng(seed + 33)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    weave = 0.5 + 0.25 * (np.sin(xx * TAU * 60) + np.sin(yy * TAU * 60))
    rib = np.abs(((yy * ribs) % 1.0) - 0.5) > 0.47
    mott = fbm(n, n, 4, 4, rng, octaves=3)
    v = 0.86 + 0.04 * weave + 0.10 * (mott - 0.5)
    v = np.where(rib, v * 0.80, v)
    v = np.where(xx < 0.08, v * 0.88, v)
    return _to_u8(v[..., None] * np.array([0.93, 0.89, 0.80], np.float32))


@lru_cache(maxsize=None)
def ice_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed + 41)
    m = fbm(n, n, 6, 6, rng, octaves=4)
    cr = fbm(n, n, 16, 16, rng, octaves=2)
    crack = (np.abs(cr - 0.5) < 0.012).astype(np.float32)
    v = 0.78 + 0.22 * (m - 0.5) - 0.2 * crack
    col = v[..., None] * np.array([0.86, 0.93, 1.0], np.float32)
    return _to_u8(col)


@lru_cache(maxsize=None)
def water_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """A tileable dark cold water surface: long wind wavelets and chop (a float
    height map, 0..1, shaded into colour per frame by the scene)."""
    rng = np.random.default_rng(seed + 42)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    h = np.zeros((n, n), np.float32)
    for (kx, ky, a) in ((3, 1, 0.5), (5, -2, 0.35), (8, 3, 0.25), (13, -5, 0.15), (21, 8, 0.1), (2, 4, 0.3)):
        h += a * np.sin(TAU * (kx * xx + ky * yy) + rng.uniform(0, TAU))
    h += 0.5 * (fbm(n, n, 16, 16, rng, octaves=3) - 0.5)
    h = (h - h.min()) / (h.max() - h.min())
    return h


@lru_cache(maxsize=None)
def canvas_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """A float (n, n) canvas-weave relief in [-1, 1] (the painted grade's overlay)."""
    rng = np.random.default_rng(seed + 43)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) / n
    w = np.sin(xx * TAU * 64) * np.sin(yy * TAU * 64)
    w += 0.6 * (fbm(n, n, 32, 32, rng, octaves=2) - 0.5)
    return (w / np.abs(w).max()).astype(np.float32)
