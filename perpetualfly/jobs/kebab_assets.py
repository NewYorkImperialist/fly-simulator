"""Procedural meshes, textures and materials for the doner kebab job (visual only).

Everything is generated in code (numpy + OpenCV resizes/blurs) and handed to MjSpec
directly (``texture.data``, ``mesh.uservert`` ...): no files on disk, no downloads.
Textures are cached per process, so building several sessions stays cheap.

Meshes are closed, outward-facing triangle meshes built from rings of points
(``tube``): consecutive rings are joined by quads and the first / last ring are
capped with fans. MuJoCo re-centres every mesh on its centroid (the geom's pos /
quat absorb the shift), so vertices can be given directly in the parent body's
frame with the geom at the origin.

Nothing here collides or has mass: callers add these meshes with
``contact_kwargs("visual")`` and ``mass=0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import cv2
import mujoco as mj
import numpy as np

TAU = 2.0 * math.pi


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


@dataclass
class MeshData:
    verts: np.ndarray  # (N, 3)
    faces: np.ndarray  # (M, 3) int, counter-clockwise seen from outside
    uv: np.ndarray  # (N, 2)

    def signed_volume(self) -> float:
        a, b, c = (self.verts[self.faces[:, k]] for k in range(3))
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def tube(rings: np.ndarray, uv: np.ndarray, *, cap_start: bool = True,
         cap_end: bool = True) -> MeshData:
    """Closed mesh through ``rings`` (R, P, 3): ring r point p joins ring r+1 point p
    (each ring is a closed loop of P points). ``uv``: (R, P, 2). Ends are capped with
    a fan around the ring centroid (a ring collapsed to a point needs no cap)."""
    R, Pn, _ = rings.shape
    verts = [rings.reshape(-1, 3)]
    uvs = [uv.reshape(-1, 2)]
    faces = []
    idx = np.arange(R * Pn).reshape(R, Pn)
    for r in range(R - 1):
        a, b = idx[r], idx[r + 1]
        a1, b1 = np.roll(a, -1), np.roll(b, -1)
        faces.append(np.stack([a, a1, b1], 1))
        faces.append(np.stack([a, b1, b], 1))
    n = R * Pn
    for r, cap, flip in ((0, cap_start, True), (R - 1, cap_end, False)):
        if not cap:
            continue
        verts.append(rings[r].mean(0)[None])
        uvs.append(uv[r].mean(0)[None])
        ring = idx[r]
        ring1 = np.roll(ring, -1)
        c = np.full(Pn, n)
        faces.append(np.stack([c, ring1, ring], 1) if flip else np.stack([c, ring, ring1], 1))
        n += 1
    md = MeshData(np.concatenate(verts), np.concatenate(faces).astype(np.int64),
                  np.concatenate(uvs))
    if md.signed_volume() < 0:
        md.faces = md.faces[:, ::-1].copy()
    return md


def lathe(profile_r: np.ndarray, profile_z: np.ndarray, n_theta: int, *,
          radial_noise=None, v_coord: np.ndarray | None = None, u_repeat: float = 1.0
          ) -> MeshData:
    """Surface of revolution about z through the profile points (r_i, z_i), bottom to
    top. ``radial_noise(theta, z) -> dr`` perturbs the radius. The ends are capped."""
    th = np.linspace(0.0, TAU, n_theta, endpoint=False)
    r = np.asarray(profile_r, float)[:, None] * np.ones_like(th)[None]
    z = np.asarray(profile_z, float)[:, None] * np.ones_like(th)[None]
    if radial_noise is not None:
        r = np.maximum(r + radial_noise(th[None], z) * (r > 1e-6), 0.0)
    rings = np.stack([r * np.cos(th), r * np.sin(th), z], -1)
    v = (np.linspace(0, 1, len(profile_r)) if v_coord is None else np.asarray(v_coord))
    uv = np.stack([np.broadcast_to(th[None] / TAU * u_repeat, r.shape),
                   np.broadcast_to(v[:, None], r.shape)], -1)
    return tube(rings, uv)


def superellipse_loop(n: int, e: float = 0.4) -> np.ndarray:
    """(n, 2) points (a, b) in [-1, 1]^2 of a rounded square (exponent ``e`` < 1
    squarer), counter-clockwise from (1, 0)."""
    ph = np.linspace(0.0, TAU, n, endpoint=False)
    c, s = np.cos(ph), np.sin(ph)
    return np.stack([np.sign(c) * np.abs(c) ** e, np.sign(s) * np.abs(s) ** e], 1)


class SurfaceNoise:
    """Smooth periodic-in-theta noise on the cone surface: a few random plane waves
    in (theta, z). Continuous across chunks, so neighbouring slabs line up."""

    def __init__(self, rng: np.random.Generator, amp: float, n: int = 10) -> None:
        self.k_th = rng.integers(2, 13, n).astype(float)
        self.k_z = rng.uniform(-6.0, 6.0, n)
        self.ph = rng.uniform(0, TAU, n)
        self.a = amp * rng.uniform(0.4, 1.0, n) / math.sqrt(n)

    def __call__(self, th, z):
        th = np.asarray(th, float)[..., None]
        z = np.asarray(z, float)[..., None]
        return np.sum(self.a * np.sin(self.k_th * th + self.k_z * z + self.ph), -1)


def meat_slab(th0: float, th1: float, z_lo: float, z_hi: float, radius_at, depth: float,
              noise: SurfaceNoise, rng: np.random.Generator, *, n_th: int = 7,
              n_loop: int = 14, v_span: tuple[float, float] = (0.0, 2.5),
              uv_repeat: float = 3.0, zwave=None) -> MeshData:
    """A curved slab of meat on the cone between angles th0..th1 and heights
    z_lo..z_hi: outer face on the cone surface (``radius_at(z)`` + noise + a slight
    dome), ``depth`` thick radially, rounded top / bottom edges (the stacked-layer
    look). Vertices in the spit frame; uv = (theta / 2 pi, z along ``v_span``)."""
    loop = superellipse_loop(n_loop, 0.3)  # (a radial, b vertical) in [-1, 1]
    bulge = rng.uniform(0.004, 0.01)
    tilt = rng.uniform(-0.01, 0.01)
    th = np.linspace(th0, th1, n_th)
    rings = np.empty((n_th, n_loop, 3))
    uv = np.empty((n_th, n_loop, 2))
    for i, t in enumerate(th):
        f = (t - th0) / (th1 - th0)
        a, b = loop[:, 0], loop[:, 1]
        z = z_lo + (b + 1) / 2 * (z_hi - z_lo) + tilt * (2 * f - 1)
        if zwave is not None:
            z = z + zwave(t)
        r_out = (radius_at(z) + noise(t, z)
                 + bulge * math.sin(math.pi * f) ** 0.4 * np.cos(np.clip(b, -1, 1) * math.pi / 2) ** 0.6)
        r_in = radius_at(z) - depth
        rr = r_in + (a + 1) / 2 * (r_out - r_in)
        rings[i, :, 0] = rr * math.cos(t)
        rings[i, :, 1] = rr * math.sin(t)
        rings[i, :, 2] = z
        R = radius_at(0.5 * (z_lo + z_hi))
        uv[i, :, 0] = (t * R + (r_out - rr)) / (TAU * R) * uv_repeat
        uv[i, :, 1] = (z - v_span[0]) / (v_span[1] - v_span[0]) * uv_repeat
    return tube(rings, uv)


def curled_slice(length: float, width: float, thick: float, curl: float,
                 rng: np.random.Generator, *, n_x: int = 9, n_loop: int = 10) -> MeshData:
    """A thin shaving curled up at both ends (a lying taco), about the body origin,
    its underside at z ~ -thick. Ragged width along its length."""
    loop = superellipse_loop(n_loop, 0.6)
    xs = np.linspace(-length / 2, length / 2, n_x)
    wob = rng.uniform(0.8, 1.1, n_x)
    wob[[0, -1]] *= 0.55  # tapered ends
    twist = rng.uniform(-0.25, 0.25)
    rings = np.empty((n_x, n_loop, 3))
    uv = np.empty((n_x, n_loop, 2))
    for i, x in enumerate(xs):
        f = x / (length / 2)
        zc = -thick / 2 + curl * f * f
        ang = math.atan(2 * curl * f / (length / 2))  # follow the curl's slope
        hy = width / 2 * wob[i]
        hz = thick / 2 * (0.6 + 0.4 * wob[i])
        a, b = loop[:, 0] * hy, loop[:, 1] * hz
        tw = twist * f
        y = a * math.cos(tw) - b * math.sin(tw)
        z = a * math.sin(tw) + b * math.cos(tw)
        rings[i, :, 0] = x - z * math.sin(ang)
        rings[i, :, 1] = y
        rings[i, :, 2] = zc + z * math.cos(ang)
        uv[i, :, 0] = (x / length + 0.5) * 0.35
        uv[i, :, 1] = (loop[:, 1] * 0.5 + 0.5) * 0.12 + (loop[:, 0] * 0.5 + 0.5) * 0.25
    return tube(rings, uv)


def knife_meshes(L: float, start: float) -> dict[str, MeshData]:
    """Chef's knife in its own frame: x along the blade (tip at x = start + L), y
    across the flat, z from edge (-) to spine (+). Handle behind x = start."""
    out = {}
    # ---- blade: spine, primary grind, edge bevel; belly curving up to the tip
    n = 18
    xs = start + L * (1 - (1 - np.linspace(0, 1, n)) ** 1.15)
    rings, uv = [], []
    for x in xs:
        f = (x - start) / L
        spine = 0.042 - 0.05 * max(0.0, (f - 0.55) / 0.45) ** 1.8
        edge = -0.068 + 0.075 * max(0.0, (f - 0.45) / 0.55) ** 2.2
        heel = min(1.0, f / 0.04)  # a short choil at the heel
        edge = edge * (0.75 + 0.25 * heel)
        tip = max(0.0, 1.0 - max(0.0, (f - 0.93) / 0.07))  # collapse at the point
        h = spine - edge
        ys, yg = 0.011 * (1 - 0.55 * f), 0.0055 * (1 - 0.5 * f)
        zg = edge + 0.24 * h
        mid = (spine + edge) / 2
        loop = np.array([
            [0.0, edge], [yg, zg], [ys, spine - 0.006], [ys * 0.55, spine],
            [-ys * 0.55, spine], [-ys, spine - 0.006], [-yg, zg]])
        loop[:, 0] *= max(tip, 0.02)
        loop[:, 1] = mid + (loop[:, 1] - mid) * max(tip, 0.02)
        rings.append(np.column_stack([np.full(len(loop), x), loop]))
        vv = (loop[:, 1] - edge) / max(spine - edge, 1e-6)
        uv.append(np.column_stack([np.full(len(loop), f), vv]))
    out["blade"] = tube(np.array(rings), np.array(uv))
    # ---- bolster: steel collar between handle and blade
    loop = superellipse_loop(16, 0.5)
    xs = np.linspace(start - 0.05, start + 0.012, 5)
    rings, uv = [], []
    for i, x in enumerate(xs):
        f = i / (len(xs) - 1)
        hy = 0.030 - 0.018 * f ** 1.5
        hz = 0.047 + 0.012 * math.sin(math.pi * min(f * 1.3, 1.0))
        zc = -0.012
        rings.append(np.column_stack([np.full(len(loop), x), loop[:, 0] * hy, zc + loop[:, 1] * hz]))
        uv.append(np.column_stack([np.full(len(loop), f), loop[:, 1] * 0.5 + 0.5]))
    out["bolster"] = tube(np.array(rings), np.array(uv))
    # ---- handle: rounded, slightly swelling toward the butt, with a flared end
    loop = superellipse_loop(18, 0.55)
    xs = np.linspace(start - 0.30, start - 0.045, 9)
    rings, uv = [], []
    for i, x in enumerate(xs):
        f = i / (len(xs) - 1)  # 0 at the butt
        hy = 0.024 + 0.004 * math.sin(math.pi * f)
        hz = 0.040 + 0.006 * (1 - f) ** 2 + 0.004 * math.sin(math.pi * f)
        if i == 0:
            hy, hz = hy * 0.8, hz * 0.85
        rings.append(np.column_stack([np.full(len(loop), x), loop[:, 0] * hy, -0.014 + loop[:, 1] * hz]))
        uv.append(np.column_stack([np.full(len(loop), f), loop[:, 1] * 0.5 + 0.5]))
    out["handle"] = tube(np.array(rings), np.array(uv))
    return out


def chef_hat_meshes() -> dict[str, MeshData]:
    """Pleated toque about the origin (z up): a band and a puffed, pleated crown."""
    pleat = lambda th, z: 0.012 * np.cos(12 * th)  # noqa: E731
    band = lathe(np.array([0.0, 0.16, 0.17, 0.175, 0.0]), np.array([0.0, 0.0, 0.06, 0.15, 0.15]),
                 48, radial_noise=lambda th, z: 0.004 * np.cos(12 * th))
    zs = np.array([0.12, 0.12, 0.2, 0.3, 0.38, 0.43, 0.455, 0.46])
    rs = np.array([0.0, 0.17, 0.25, 0.27, 0.24, 0.17, 0.08, 0.0])
    puff = lathe(rs, zs, 48, radial_noise=pleat)
    return {"band": band, "puff": puff}


def add_mesh(spec: mj.MjSpec, name: str, md: MeshData):
    return spec.add_mesh(name=name, uservert=md.verts.ravel().tolist(),
                         userface=md.faces.ravel().tolist(),
                         usertexcoord=md.uv.ravel().tolist(), maxhullvert=16)


def transform(md: MeshData, R: np.ndarray, t: np.ndarray) -> MeshData:
    """Mesh with vertices mapped by x -> R x + t (R a rotation)."""
    return MeshData(md.verts @ np.asarray(R).T + np.asarray(t), md.faces, md.uv)


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------


def tile_noise(h: int, w: int, gh: int, gw: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth value noise on a gh x gw lattice, (h, w) float32 in ~[0, 1], seamless
    when tiled."""
    g = rng.random((gh, gw)).astype(np.float32)
    g = np.pad(g, 2, mode="wrap")
    cy, cx = h / gh, w / gw
    big = cv2.resize(g, (int(round(cx * (gw + 4))), int(round(cy * (gh + 4)))),
                     interpolation=cv2.INTER_CUBIC)
    oy, ox = int(round(2 * cy)), int(round(2 * cx))
    return big[oy:oy + h, ox:ox + w]


def fbm(h, w, gh, gw, rng, octaves=4, gain=0.5) -> np.ndarray:
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = 2 ** o
        out += amp * tile_noise(h, w, min(gh * s, h), min(gw * s, w), rng)
        tot += amp
        amp *= gain
    return out / tot


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _mix(a, b, t):
    t = np.asarray(t, np.float32)[..., None]
    return a * (1 - t) + np.asarray(b, np.float32) * t


def _to_u8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)


MEAT_STAGES = ("raw", "seared", "cooked")


@lru_cache(maxsize=4)
def meat_textures(seed: int = 0, h: int = 256, w: int = 512) -> dict[str, np.ndarray]:
    """Doner meat textures (u around the spit, v up it), uint8 RGB (h, w, 3):
    ``raw`` (pink, white fat), ``seared`` (grey-tan), ``cooked`` (browned, fat
    streaks, char), ``inner`` (the exposed inside), ``top`` (charred crown)."""
    rng = np.random.default_rng(seed + 101)
    # layered grain: horizontal strata (stretched along u) + fibres
    strata = tile_noise(h, w, 64, 8, rng) * 0.6 + tile_noise(h, w, 128, 16, rng) * 0.4
    fibre = tile_noise(h, w, 256, 64, rng)
    blotch = fbm(h, w, 4, 8, rng, 4)
    fatn = tile_noise(h, w, 24, 6, rng) * 0.7 + tile_noise(h, w, 48, 24, rng) * 0.3
    fat = _smooth(0.035, 0.0, np.abs(fatn - 0.5)) * _smooth(0.45, 0.6, tile_noise(h, w, 8, 16, rng))
    fat = cv2.GaussianBlur(fat, (0, 0), 0.8)
    pits = _smooth(0.78, 0.9, tile_noise(h, w, 128, 256, rng))
    layers = 0.5 + 0.5 * np.sin(np.linspace(0, TAU * 34, h, endpoint=False))[:, None]
    layers = (layers ** 6) * _smooth(0.3, 0.7, tile_noise(h, w, 16, 32, rng))
    fine = tile_noise(h, w, 128, 256, rng)
    mid = fbm(h, w, 16, 32, rng, 3)
    striae = tile_noise(h, w, 96, 32, rng)
    # shreds: thin horizontal ridges (crispy dark edges, golden edges) + tiny burnt bits
    shred_d = (_smooth(0.06, 0.0, np.abs(tile_noise(h, w, 64, 64, rng) - 0.5))
               * _smooth(0.45, 0.65, tile_noise(h, w, 16, 32, rng)))
    shred_g = (_smooth(0.05, 0.0, np.abs(tile_noise(h, w, 64, 64, rng) - 0.5))
               * _smooth(0.45, 0.65, tile_noise(h, w, 16, 32, rng)))
    crumb = fbm(h, w, 32, 64, rng, 3)
    burnt = _smooth(0.72, 0.85, tile_noise(h, w, 128, 64, rng)) * _smooth(0.4, 0.7, mid)
    gold = shred_g
    out = {}
    # cooked: brown with golden and dark crispy bits, horizontal striations
    base = _mix(np.array([0.32, 0.14, 0.06], np.float32), (0.60, 0.35, 0.16),
                mid * 1.0 + strata * 0.3 - 0.12)
    base *= ((0.88 + 0.24 * fine) * (0.9 + 0.2 * striae) * (0.8 + 0.4 * crumb))[..., None]
    base = _mix(base, (0.30, 0.14, 0.05), layers * 0.3)
    base = _mix(base, (0.22, 0.10, 0.04), shred_d * 0.55)
    base = _mix(base, (0.80, 0.53, 0.24), gold * 0.45)
    base = _mix(base, (0.84, 0.64, 0.38), fat * 0.5)
    base = _mix(base, (0.14, 0.06, 0.02), burnt * 0.6)
    base = _mix(base, (0.12, 0.06, 0.03), pits * 0.35)
    out["cooked"] = base
    # seared: pale, greyish-tan, the fat starting to gild
    base = _mix(np.array([0.62, 0.40, 0.31], np.float32), (0.76, 0.54, 0.40), mid * 0.9 + strata * 0.3)
    base *= (0.9 + 0.2 * fine)[..., None] * (0.94 + 0.12 * striae)[..., None]
    base = _mix(base, (0.52, 0.30, 0.20), layers * 0.3)
    base = _mix(base, (0.90, 0.78, 0.62), fat * 0.7)
    base = _mix(base, (0.45, 0.25, 0.14), burnt * 0.3)
    out["seared"] = base
    # raw: pink / red with white fat marbling
    base = _mix(np.array([0.76, 0.33, 0.33], np.float32), (0.92, 0.54, 0.51), mid * 0.9 + strata * 0.4)
    base *= (0.9 + 0.2 * fibre)[..., None] * (0.94 + 0.12 * striae)[..., None]
    base = _mix(base, (0.62, 0.20, 0.22), layers * 0.35)
    base = _mix(base, (0.97, 0.88, 0.84), fat * 0.9)
    out["raw"] = base
    # inner: the exposed inside of the cone (cooked through, juicy, pressed layers)
    base = _mix(np.array([0.56, 0.33, 0.23], np.float32), (0.70, 0.46, 0.33), mid * 0.9 + strata * 0.4)
    base *= (0.88 + 0.24 * fibre)[..., None] * (0.92 + 0.16 * striae)[..., None]
    base = _mix(base, (0.40, 0.21, 0.13), layers * 0.5)
    base = _mix(base, (0.86, 0.72, 0.56), fat * 0.7)
    out["inner"] = base
    # top: crown, well browned, crisp
    base = _mix(np.array([0.34, 0.16, 0.07], np.float32), (0.56, 0.31, 0.14), mid * 0.8 + fibre * 0.3)
    base *= (0.85 + 0.3 * fine)[..., None]
    base = _mix(base, (0.12, 0.06, 0.02), burnt * 0.8)
    base = _mix(base, (0.74, 0.52, 0.28), fat * 0.4)
    out["top"] = base
    return {k: _to_u8(v) for k, v in out.items()}


@lru_cache(maxsize=2)
def steel_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Brushed stainless: fine streaks along u, faint blotches."""
    rng = np.random.default_rng(seed + 202)
    streak = tile_noise(n, n, n, 8, rng) * 0.6 + tile_noise(n, n, n // 2, 4, rng) * 0.4
    blot = fbm(n, n, 4, 4, rng, 3)
    v = 0.5 + 0.08 * (streak - 0.5) + 0.06 * (blot - 0.5)
    img = np.stack([v * 0.98, v * 0.99, v * 1.02], -1)
    return _to_u8(img)


@lru_cache(maxsize=2)
def blade_texture(seed: int = 0, h: int = 128, w: int = 256) -> np.ndarray:
    """Polished blade: v runs edge (0) -> spine (1). A fake environment reflection
    (bright band, dark band, sky) and a bright honed edge, fine grind lines along u."""
    rng = np.random.default_rng(seed + 303)
    v = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    env = (0.58 + 0.34 * np.exp(-((v - 0.72) / 0.12) ** 2) - 0.22 * np.exp(-((v - 0.42) / 0.12) ** 2)
           + 0.40 * np.exp(-(v / 0.06) ** 2) + 0.12 * np.exp(-((v - 0.95) / 0.05) ** 2))
    grind = tile_noise(h, w, h, 4, rng)
    val = env + 0.05 * (grind - 0.5)
    val = np.broadcast_to(val, (h, w))
    img = np.stack([val * 0.97, val * 0.99, val * 1.03], -1)
    return _to_u8(img)


@lru_cache(maxsize=2)
def heater_texture(seed: int = 0, n: int = 256, cells: int = 4) -> np.ndarray:
    """Glowing ceramic burner tile: a grid of hot cells with dark rims."""
    rng = np.random.default_rng(seed + 404)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / (n / cells)
    x = x + 0.5 * (np.floor(y) % 2)  # offset rows (honeycomb-ish)
    fx, fy = x - np.floor(x) - 0.5, y - np.floor(y) - 0.5
    d = np.sqrt(fx * fx + fy * fy) / 0.5
    heat = 0.75 + 0.25 * tile_noise(n, n, 4, 4, rng)
    core = np.clip(1.0 - d, 0, 1) ** 0.6 * heat
    img = _mix(np.array([0.78, 0.22, 0.04], np.float32) * np.ones((n, n, 1), np.float32),
               (1.0, 0.42, 0.08), _smooth(0.0, 0.5, core))
    img = _mix(img, (1.0, 0.66, 0.25), _smooth(0.6, 1.0, core) * 0.7)
    return _to_u8(img)


@lru_cache(maxsize=2)
def floor_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """2 x 2 terracotta shop-floor tiles with grout (one repeat = 2 tiles)."""
    rng = np.random.default_rng(seed + 505)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / (n / 2)
    fx, fy = x - np.floor(x), y - np.floor(y)
    edge = np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy))
    grout = _smooth(0.028, 0.012, edge)
    bevel = _smooth(0.06, 0.02, edge) - grout
    tile_id = (np.floor(x) + 2 * np.floor(y)).astype(int)
    shade = np.array([0.97, 1.03, 1.0, 0.94])[tile_id]
    n1 = fbm(n, n, 4, 4, rng, 4)
    base = _mix(np.array([0.66, 0.36, 0.24], np.float32) * np.ones((n, n, 1), np.float32),
                (0.76, 0.46, 0.31), n1)
    base *= shade[..., None]
    base = _mix(base, (0.82, 0.58, 0.44), np.clip(bevel, 0, 1) * 0.35)
    base = _mix(base, (0.72, 0.70, 0.66), grout)
    return _to_u8(base)


@lru_cache(maxsize=2)
def wall_texture(seed: int = 0, h: int = 256, w: int = 256) -> np.ndarray:
    """White glazed subway tiles (2:1, offset rows), grey grout. 4 rows x 2 tiles."""
    rng = np.random.default_rng(seed + 606)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    y = y / (h / 4)
    x = x / (w / 2) + 0.5 * (np.floor(y) % 2)
    fx, fy = x - np.floor(x), y - np.floor(y)
    edge = np.minimum(np.minimum(fx, 1 - fx) * 2.0, np.minimum(fy, 1 - fy))
    grout = _smooth(0.05, 0.025, edge)
    bevel = _smooth(0.12, 0.04, edge) - grout
    n1 = tile_noise(h, w, 8, 8, rng)
    base = (0.80 + 0.06 * n1)[..., None] * np.array([0.86, 0.93, 0.92], np.float32)
    base = _mix(base, (0.75, 0.76, 0.76), np.clip(bevel, 0, 1) * 0.5)
    base = _mix(base, (0.55, 0.55, 0.53), grout)
    return _to_u8(base)


def add_texture(spec: mj.MjSpec, name: str, img: np.ndarray):
    """2D RGB texture from a uint8 (h, w, 3) image (row 0 = v 0)."""
    h, w, _ = img.shape
    tex = spec.add_texture(name=name, type=mj.mjtTexture.mjTEXTURE_2D, width=w, height=h,
                           nchannel=3)
    tex.data = np.ascontiguousarray(img).tobytes()
    return tex


def add_textured_material(spec: mj.MjSpec, name: str, texture: str | None = None, **kw):
    mat = spec.add_material(name=name, **kw)
    if texture is not None:
        mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = texture
    return mat
