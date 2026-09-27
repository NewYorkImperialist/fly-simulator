"""Procedural meshes and textures for the Sisyphus job (visual only).

Same approach as ``kebab_assets``: everything is generated in code (numpy + OpenCV)
and handed to MjSpec directly; nothing is written to disk. Nothing here collides or
has mass (callers add the meshes with ``contact_kwargs("visual")`` and ``mass=0``).

A Greek-myth hillside at sunset:

* meshes: a lumpy granite boulder shell (drawn over the colliding sphere), fluted
  Doric columns (whole and broken), fallen column drums, rocks, olive trees
  (a twisted trunk and a cloud of leaf clusters);
* textures: the trodden earth path, dry hillside grass with rocks, a dry-stone wall,
  a mossy granite boulder, olive bark and leaves, and the sunset backdrop (sky
  gradient, the sun, far mountains, the sea, cypress hills and a temple).
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.broccoli_toss_assets import merge, sphere_mesh
from fly_simulator.jobs.kebab_assets import (  # noqa: F401  (re-exported helpers)
    MeshData, SurfaceNoise, _mix, _smooth, _to_u8, add_mesh, add_texture,
    add_textured_material, fbm, lathe, tile_noise, transform,
)
from fly_simulator.jobs.pizza_chef_assets import marble_texture  # noqa: F401
from fly_simulator.jobs.trampoline_assets import polyline_tube

TAU = 2.0 * math.pi


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def box_mesh(half, scale: float = 1.0) -> MeshData:
    """A box (half sizes ``half``) about the origin with a planar uv on every face in
    world units (uv = face coordinates in mm * ``scale``): a texture tiles evenly on
    all six faces (MuJoCo's own 2D mapping on a box primitive only suits its +z face)."""
    hx, hy, hz = (float(v) for v in half)
    verts, uvs, faces = [], [], []
    # (axis of the normal, sign, the two in-face axes u, v)
    for ax, sg, ua, va in ((0, 1, 1, 2), (0, -1, 1, 2), (1, 1, 0, 2), (1, -1, 0, 2),
                           (2, 1, 0, 1), (2, -1, 0, 1)):
        h = np.array([hx, hy, hz])
        n0 = len(verts)
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            p = np.zeros(3)
            p[ax] = sg * h[ax]
            p[ua] = a * h[ua]
            p[va] = b * h[va]
            verts.append(p)
            uvs.append((p[ua] * scale, -p[va] * scale))
        faces += [(n0, n0 + 1, n0 + 2), (n0, n0 + 2, n0 + 3)]
    md = MeshData(np.array(verts), np.array(faces, np.int64), np.array(uvs))
    # orient every face outward
    c = md.verts[md.faces].mean(1)
    nrm = np.cross(md.verts[md.faces[:, 1]] - md.verts[md.faces[:, 0]],
                   md.verts[md.faces[:, 2]] - md.verts[md.faces[:, 0]])
    flip = np.einsum("ij,ij->i", nrm, c) < 0
    md.faces[flip] = md.faces[flip][:, ::-1]
    return md


def boulder_mesh(radius: float, seed: int = 0, amp: float = 0.045, n_theta: int = 40,
                 n_phi: int = 24) -> MeshData:
    """A slightly lumpy sphere (radius +- ``amp`` * radius) about the origin; uv =
    (longitude, pole to pole) so a texture shows the boulder rolling."""
    rng = np.random.default_rng(seed + 301)
    noise = SurfaceNoise(rng, amp * radius, n=12)
    ph = np.linspace(-math.pi / 2, math.pi / 2, n_phi)
    return lathe(radius * np.cos(ph), radius * np.sin(ph), n_theta,
                 radial_noise=lambda th, z: noise(th, 3.0 * z / radius),
                 v_coord=np.linspace(0, 1, n_phi))


def column_mesh(r: float, h: float, *, flutes: int = 16, capital: bool = True,
                broken: float = 0.0, seed: int = 0, n_theta: int = 64) -> MeshData:
    """A Doric column standing on z = 0, shaft radius ``r``, total height ``h``:
    a base moulding, a fluted shaft with entasis, the echinus and a flat abacus
    (``capital``). ``broken`` > 0: no capital and a jagged, sloping break at the top
    (that fraction of r in height)."""
    zb = 0.07 * h  # base moulding top
    zt = h * (0.86 if capital else 1.0)  # shaft top
    zs = np.linspace(zb, zt, 10)
    s = (zs - zb) / max(zt - zb, 1e-9)
    shaft_r = r * (1.0 - 0.14 * s + 0.05 * np.sin(math.pi * s))  # entasis
    pr = [0.0, 1.32 * r, 1.32 * r, 1.18 * r, 1.08 * r, *shaft_r]
    pz = [0.0, 0.0, 0.035 * h, 0.05 * h, zb, *zs]
    if capital:
        pr += [0.94 * r, 1.25 * r, 1.42 * r, 1.42 * r, 0.0]
        pz += [zt + 0.02 * h, zt + 0.07 * h, zt + 0.09 * h, h, h]
    else:
        pr += [0.0]
        pz += [zt]
    pr, pz = np.array(pr), np.array(pz)
    z_f0, z_f1 = zb + 0.01 * h, zt - (0.0 if broken else 0.005 * h)

    def flute(th, z):
        on = (z > z_f0) & (z < z_f1)
        return -0.075 * r * np.abs(np.sin(0.5 * flutes * th)) ** 0.6 * on
    md = lathe(pr, pz, n_theta, radial_noise=flute, v_coord=pz / h)
    if broken > 0:
        rng = np.random.default_rng(seed + 302)
        top = md.verts[:, 2] > zt - 1e-6
        th = np.arctan2(md.verts[top, 1], md.verts[top, 0])
        k = rng.integers(2, 5)
        jag = broken * r * (0.5 + 0.5 * np.cos(th - rng.uniform(0, TAU))
                            + 0.25 * np.sin(k * th + rng.uniform(0, TAU)))
        md.verts[top, 2] -= jag
    return md


def drum_mesh(r: float, L: float, flutes: int = 16) -> MeshData:
    """A fallen column drum lying along x (centre at the origin)."""
    md = column_mesh(r, L, flutes=flutes, capital=False, n_theta=48)
    # (the base moulding is at z 0..0.07 L: fine as a drum's weathered end)
    md = transform(md, np.eye(3), np.array([0.0, 0.0, -L / 2]))
    R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # z -> x
    return transform(md, R, np.zeros(3))


def rock_mesh(r: float, seed: int = 0, squash: float = 0.6) -> MeshData:
    """A lumpy flattened rock (a boulder mesh squashed in z), resting on z = 0."""
    md = boulder_mesh(r, seed + 17, amp=0.18, n_theta=16, n_phi=10)
    v = md.verts.copy()
    v[:, 2] *= squash
    v[:, 2] -= v[:, 2].min() + 0.1 * r * squash
    return MeshData(v, md.faces, md.uv)


def olive_tree_meshes(height: float, seed: int = 0) -> dict[str, MeshData]:
    """An olive tree standing at the origin: a gnarled, leaning, forking trunk
    (``trunk``) and a broad cloud of flattened leaf clusters (``canopy``)."""
    rng = np.random.default_rng(seed + 303)
    H = height
    lean = rng.uniform(0, TAU)
    lv = np.array([math.cos(lean), math.sin(lean), 0.0])
    pts = []
    for k in range(7):
        t = k / 6
        wig = 0.12 * H * np.array([math.sin(5 * t + seed), math.cos(4 * t + 2 * seed), 0.0])
        pts.append(lv * 0.18 * H * t + wig * t + np.array([0, 0, 0.55 * H * t]))
    pts = np.array(pts)
    parts = [polyline_tube(pts, 0.06 * H, 9)]
    # a thick root flare
    parts.append(polyline_tube(np.array([[0, 0, -0.02 * H], [0, 0, 0.1 * H]]), 0.095 * H, 9))
    top = pts[-1]
    tips = []
    for b in range(4):  # branches forking out of the top
        a = lean + TAU * b / 4 + rng.uniform(-0.4, 0.4)
        d = np.array([math.cos(a), math.sin(a), 0.0])
        mid = top + d * 0.18 * H + np.array([0, 0, 0.1 * H])
        tip = top + d * 0.36 * H + np.array([0, 0, 0.16 * H + rng.uniform(0, 0.08) * H])
        parts.append(polyline_tube(np.array([top - d * 0.02 * H, mid, tip]), 0.028 * H, 7))
        tips.append(tip)
    trunk = merge(*parts)
    blobs = []
    centre = top + np.array([0, 0, 0.22 * H])
    for k in range(16):
        base = tips[k % 4] if k < 12 else centre
        c = base + rng.normal(0, 0.09 * H, 3) * np.array([1.0, 1.0, 0.5])
        blobs.append(sphere_mesh(rng.uniform(0.13, 0.2) * H, c, 12, 8,
                                 squash=rng.uniform(0.55, 0.75)))
    return {"trunk": trunk, "canopy": merge(*blobs)}


def cypress_mesh(height: float, r: float) -> MeshData:
    """A slim flame-shaped cypress on z = 0."""
    s = np.linspace(0, 1, 12)
    pr = r * np.sin(math.pi * np.clip(s * 1.05, 0, 1)) ** 0.7 * (1 - 0.5 * s)
    pr[0] = 0.0
    pr[-1] = 0.0
    return lathe(pr, s * height, 14)


# ---------------------------------------------------------------------------
# textures (uint8 RGB, row 0 = v 0)
# ---------------------------------------------------------------------------


def _norm(v):
    return (v - v.min()) / max(float(v.max() - v.min()), 1e-6)


@lru_cache(maxsize=2)
def hillside_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """Dry Mediterranean hillside (seamless): sun-bleached grass over ochre earth,
    grey rock outcrops, pebbles, thyme tufts."""
    rng = np.random.default_rng(seed + 311)
    f = _norm(fbm(n, n, 5, 5, rng, 5))
    g = _norm(fbm(n, n, 12, 12, rng, 3))
    earth = _mix(np.array([0.50, 0.40, 0.27], np.float32), np.array([0.60, 0.49, 0.33], np.float32), g)
    grass = _mix(np.array([0.58, 0.55, 0.32], np.float32), np.array([0.44, 0.47, 0.25], np.float32), g)
    img = _mix(earth, grass, _smooth(0.3, 0.65, f))
    rock = _smooth(0.8, 0.88, _norm(fbm(n, n, 4, 4, rng, 4)))
    rt = _mix(np.array([0.50, 0.47, 0.42], np.float32), np.array([0.62, 0.59, 0.53], np.float32), g)
    img = _mix(img, rt, rock * 0.8)
    u8 = _to_u8(img)
    for _ in range(900):  # dry grass blades
        x, y = (int(v) for v in rng.integers(0, n, 2))
        L = int(rng.integers(3, 9))
        a = rng.normal(-math.pi / 2, 0.5)
        c = rng.uniform(0.75, 1.2)
        col = tuple(int(min(255, v * c)) for v in (164, 148, 90))
        cv2.line(u8, (x, y), (int(x + L * math.cos(a)), int(y + L * math.sin(a))), col, 1, cv2.LINE_AA)
    for _ in range(140):  # pebbles
        x, y = (int(v) for v in rng.integers(4, n - 4, 2))
        v = int(rng.integers(110, 170))
        cv2.ellipse(u8, (x, y), (int(rng.integers(1, 4)), int(rng.integers(1, 3))),
                    float(rng.uniform(0, 180)), 0, 360, (v, v - 6, v - 16), -1, cv2.LINE_AA)
    for _ in range(40):  # thyme / sage tufts
        x, y = (int(v) for v in rng.integers(8, n - 8, 2))
        for _k in range(6):
            cv2.circle(u8, (x + int(rng.integers(-5, 6)), y + int(rng.integers(-4, 5))),
                       int(rng.integers(2, 4)), (int(rng.integers(70, 90)), int(rng.integers(84, 104)),
                                                  int(rng.integers(55, 70))), -1, cv2.LINE_AA)
    return u8


@lru_cache(maxsize=2)
def path_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """The trodden path up the hill: packed pale earth with grit and ruts."""
    rng = np.random.default_rng(seed + 312)
    f = _norm(fbm(n, n, 6, 6, rng, 5))
    img = _mix(np.array([0.60, 0.47, 0.32], np.float32), np.array([0.74, 0.62, 0.45], np.float32), f)
    grit = (rng.random((n, n)) > 0.985).astype(np.float32)
    img = _mix(img, (0.40, 0.33, 0.25), cv2.GaussianBlur(grit, (3, 3), 0.7) * 0.8)
    return _to_u8(img)


@lru_cache(maxsize=2)
def drystone_texture(seed: int = 0, h: int = 512, w: int = 512) -> np.ndarray:
    """A dry-stone wall: courses of irregular pale limestone blocks, dark gaps."""
    rng = np.random.default_rng(seed + 313)
    img = np.full((h, w, 3), (58, 48, 40), np.uint8)
    y = 0
    while y < h:
        ch = int(rng.integers(h // 9, h // 6))
        x = -int(rng.integers(0, 40))
        while x < w:
            bw = int(rng.integers(ch, 3 * ch))
            v = rng.uniform(0.75, 1.1)
            col = tuple(int(min(255, c * v)) for c in (196, 182, 158))
            pts = np.array([[x + 3, y + 2 + rng.integers(0, 3)], [x + bw - 3, y + 2 + rng.integers(0, 3)],
                            [x + bw - 2 - rng.integers(0, 4), y + ch - 3],
                            [x + 2 + rng.integers(0, 4), y + ch - 2]], np.int32)
            cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
            x += bw
        y += ch
    g = _norm(fbm(h, w, 8, 16, rng, 4))
    shade = (0.82 + 0.3 * g)[..., None]
    img = np.clip(img.astype(np.float32) * shade, 0, 255).astype(np.uint8)
    lichen = _smooth(0.72, 0.8, _norm(fbm(h, w, 6, 12, rng, 3)))
    return _to_u8(_mix(img.astype(np.float32) / 255, (0.74, 0.70, 0.40), lichen * 0.5))


@lru_cache(maxsize=2)
def boulder_texture(seed: int = 0, h: int = 256, w: int = 512) -> np.ndarray:
    """Weathered granite (u around, v pole to pole): grey speckle, feldspar flecks,
    a crack or two and moss on one side, so the rolling reads."""
    rng = np.random.default_rng(seed + 314)
    f = _norm(fbm(h, w, 4, 8, rng, 5))
    img = _mix(np.array([0.46, 0.45, 0.43], np.float32), np.array([0.66, 0.64, 0.60], np.float32), f)
    sp = rng.random((h, w))
    img = _mix(img, (0.16, 0.16, 0.17), (sp > 0.93).astype(np.float32) * 0.7)
    img = _mix(img, (0.86, 0.80, 0.74), (sp < 0.04).astype(np.float32) * 0.6)
    u, v = np.meshgrid(np.linspace(0, 1, w, endpoint=False), np.linspace(0, 1, h), indexing="xy")
    moss_side = 0.5 + 0.5 * np.cos(TAU * (u - 0.2)) * np.sin(math.pi * v)
    moss = _smooth(0.72, 0.9, moss_side * 0.55 + 0.55 * _norm(fbm(h, w, 6, 12, rng, 4)))
    img = _mix(img, (0.34, 0.40, 0.16), moss * 0.75)
    u8 = _to_u8(img)
    for _ in range(3):  # cracks
        x, y = int(rng.integers(0, w)), int(rng.integers(h // 5, 4 * h // 5))
        pts = [(x, y)]
        for _k in range(8):
            x += int(rng.integers(6, 18))
            y += int(rng.integers(-8, 9))
            pts.append((x, y))
        cv2.polylines(u8, [np.array(pts, np.int32)], False, (40, 38, 36), 2, cv2.LINE_AA)
    return u8


@lru_cache(maxsize=2)
def bark_texture(seed: int = 0, n: int = 128) -> np.ndarray:
    """Grey, deeply furrowed olive bark."""
    rng = np.random.default_rng(seed + 315)
    g = fbm(n, n, 4, 12, rng, 4)
    x = np.linspace(0, 1, n, dtype=np.float32)[None, :] * np.ones((n, 1), np.float32)
    ridges = 0.5 + 0.5 * np.sin(TAU * (9 * x + 2.5 * g))
    v = 0.28 + 0.2 * ridges
    return _to_u8(np.stack([v * 1.05, v * 1.0, v * 0.9], -1))


@lru_cache(maxsize=2)
def olive_leaf_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Olive foliage: dense silvery sage-green leaflets with dark gaps."""
    rng = np.random.default_rng(seed + 316)
    img = np.full((n, n, 3), (38, 52, 30), np.uint8)
    for _ in range(2200):
        x, y = (int(v) for v in rng.integers(0, n, 2))
        a = float(rng.uniform(0, 180))
        silver = rng.uniform() < 0.35
        base = (150, 166, 128) if silver else (86, 112, 58)
        c = rng.uniform(0.75, 1.15)
        cv2.ellipse(img, (x, y), (int(rng.integers(4, 8)), 1), a, 0, 360,
                    tuple(int(min(255, v * c)) for v in base), -1, cv2.LINE_AA)
    return img


@lru_cache(maxsize=2)
def sunset_texture(seed: int = 0, h: int = 512, w: int = 1536) -> np.ndarray:
    """The backdrop: a sunset sky (violet -> rose -> orange -> gold at the horizon),
    the low sun with a glow, lit streaks of cloud, far blue mountains, a sliver of
    wine-dark sea, and near hills with cypresses and a little temple, down to a
    warm hillside that meets the ground."""
    rng = np.random.default_rng(seed + 317)
    t = np.linspace(0, 1, h, dtype=np.float32)
    stops = [(0.0, (0.20, 0.16, 0.40)), (0.22, (0.48, 0.28, 0.52)), (0.40, (0.90, 0.46, 0.44)),
             (0.52, (0.99, 0.66, 0.36)), (0.60, (1.00, 0.84, 0.52))]
    sky = np.zeros((h, 3), np.float32)
    for k in range(3):
        sky[:, k] = np.interp(t, [s for s, _ in stops], [c[k] for _, c in stops])
    img = np.broadcast_to(sky[:, None], (h, w, 3)).copy()
    hz = int(h * 0.60)  # horizon row
    sx, sy = int(w * 0.24), int(h * 0.55)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot((xx - sx) / 1.0, (yy - sy) * 1.3)
    glow = np.exp(-d / (0.09 * w))
    img = img + glow[..., None] * np.array([0.5, 0.32, 0.1], np.float32)
    img = _mix(img, (1.0, 0.95, 0.78), _smooth(0.042 * h, 0.036 * h, d))
    for _ in range(14):  # streaky clouds, lit from below
        cx, cy = int(rng.integers(0, w)), int(rng.integers(int(0.1 * h), int(0.5 * h)))
        L, th = int(rng.integers(w // 12, w // 5)), int(rng.integers(3, 8))
        m = np.zeros((h, w), np.float32)
        cv2.ellipse(m, (cx, cy), (L, th), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), 3)
        lit = (1.0, 0.62, 0.52) if cy > 0.3 * h else (0.72, 0.42, 0.58)
        img = _mix(img, lit, m * 0.65)
    x = np.arange(w, dtype=np.float32) / w

    def ridge(base, amp, freqs, seed2):
        r2 = np.random.default_rng(seed + seed2)
        y = np.full(w, base, np.float32)
        for f in freqs:
            y += amp / f * np.sin(TAU * (f * x + r2.uniform()))
        return y

    def fill_below(img, yline, col, haze=0.0):
        m = (yy >= yline[None, :] * h).astype(np.float32)
        m = cv2.GaussianBlur(m, (0, 0), 0.8)
        c = np.asarray(col, np.float32) * (1 - haze) + np.array([1.0, 0.72, 0.55], np.float32) * haze
        return _mix(img, c, m)

    img = fill_below(img, ridge(0.52, 0.10, (2, 5, 11, 23), 1), (0.44, 0.34, 0.52), 0.45)  # far mountains
    img = fill_below(img, ridge(0.56, 0.05, (3, 7, 17), 2), (0.34, 0.26, 0.42), 0.3)
    sea = (yy >= hz).astype(np.float32)
    img = _mix(img, (0.30, 0.20, 0.34), sea * 0.9)
    # the sun's glitter path on the sea
    path = sea * np.exp(-np.abs(xx - sx) / (0.02 * w)) * (0.5 + 0.5 * (rng.random((h, w)) > 0.6))
    img = _mix(img, (1.0, 0.78, 0.45), np.clip(path, 0, 1) * 0.8)
    near = ridge(0.66, 0.06, (2, 3, 9), 3)
    img = fill_below(img, near, (0.46, 0.36, 0.30), 0.15)
    u8 = _to_u8(img)
    dark = (62, 46, 48)
    for _ in range(26):  # cypresses on the near hills
        px = int(rng.integers(0, w))
        base = int(near[px] * h) + int(rng.integers(0, 10))
        H, W = int(rng.integers(30, 70)), int(rng.integers(5, 10))
        cv2.ellipse(u8, (px, base - H // 2), (W, H // 2), 0, 0, 360, dark, -1, cv2.LINE_AA)
    # the temple on a hilltop (to the right)
    px = int(w * 0.7)
    base = int(near[px] * h)
    cw = 64
    cv2.rectangle(u8, (px - cw, base - 6), (px + cw, base + 4), dark, -1)
    for k in range(7):
        cx = px - cw + 8 + k * (2 * cw - 16) // 6
        cv2.rectangle(u8, (cx - 3, base - 40), (cx + 3, base - 6), dark, -1)
    cv2.rectangle(u8, (px - cw - 2, base - 48), (px + cw + 2, base - 40), dark, -1)
    cv2.fillPoly(u8, [np.array([[px - cw - 2, base - 48], [px + cw + 2, base - 48], [px, base - 64]],
                               np.int32)], dark, cv2.LINE_AA)
    # the lower band: the hillside in the evening light, meeting the ground
    lo = int(h * 0.80)
    g = _norm(fbm(h - lo, w, 4, 24, rng, 4))
    band = _mix(np.array([0.42, 0.34, 0.22], np.float32), np.array([0.58, 0.48, 0.30], np.float32), g)
    blend = _smooth(0.0, 0.25, np.linspace(0, 1, h - lo, dtype=np.float32))[:, None]
    u8[lo:] = _to_u8(_mix(u8[lo:].astype(np.float32) / 255, band, np.broadcast_to(blend, (h - lo, w))))
    return u8
