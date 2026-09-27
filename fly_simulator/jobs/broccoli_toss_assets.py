"""Procedural meshes and textures for the broccoli toss job (visual only).

A fly-scale streamer's room: stream layouts for the monitors (a generic game view,
chat, a red LIVE badge; no real names, channels or logos), posters with generic art,
acoustic foam, a wooden floor and a rug, cardboard boxes, soda cans, the plate and
the broccoli florets. Everything is generated in code (numpy + OpenCV) and handed to
MjSpec directly (``bowling_assets`` / ``kebab_assets`` helpers); nothing is written
to disk. Textures are cached per process.

Nothing here collides or has mass: callers add these meshes with
``contact_kwargs("visual")`` and ``mass=0``.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import mujoco as mj
import numpy as np

from fly_simulator.jobs.kebab_assets import (  # noqa: F401  (re-exported helpers)
    MeshData, _mix, _smooth, _to_u8, add_mesh, add_texture, add_textured_material, fbm, lathe,
    tile_noise, tube,
)

TAU = 2.0 * math.pi


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def merge(*meshes: MeshData) -> MeshData:
    """One mesh from several (vertex / face lists concatenated)."""
    verts, faces, uvs = [], [], []
    n = 0
    for md in meshes:
        verts.append(md.verts)
        faces.append(md.faces + n)
        uvs.append(md.uv)
        n += len(md.verts)
    return MeshData(np.concatenate(verts), np.concatenate(faces), np.concatenate(uvs))


def sphere_mesh(radius: float, centre=(0.0, 0.0, 0.0), n_theta: int = 14, n_phi: int = 9,
                squash: float = 1.0) -> MeshData:
    ph = np.linspace(-math.pi / 2, math.pi / 2, n_phi)
    md = lathe(radius * np.cos(ph), radius * np.sin(ph) * squash, n_theta,
               v_coord=np.linspace(0, 1, n_phi))
    return MeshData(md.verts + np.asarray(centre, float), md.faces, md.uv)


def panel_mesh(w: float, h: float, t: float = 0.02) -> MeshData:
    """Thin box, width ``w`` along x, height ``h`` along z, thickness ``t`` along y;
    the front face (normal -y) maps the whole texture upright (image row 0 at the
    top). Place it with ``panel_quat``."""
    x0, x1, y0, y1, z0, z1 = -w / 2, w / 2, -t / 2, t / 2, -h / 2, h / 2
    # front face (y0): u = 0..1 left to right (seen from -y), v = 1 at the bottom
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1],
                  [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]], float)
    uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 0], [0, 0]], float)
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                  [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    md = MeshData(v, f, uv)
    if md.signed_volume() < 0:
        md.faces = md.faces[:, ::-1].copy()
    return md


def panel_quat(normal) -> tuple[float, float, float, float]:
    """Quaternion that turns a ``panel_mesh`` so its textured front faces ``normal``
    (horizontal), upright."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    x = np.cross(-n, z)
    x /= np.linalg.norm(x)
    y = -n
    import mujoco as mj

    q = np.empty(4)
    mj.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    return tuple(float(v) for v in q)


def torus_mesh(R: float, r: float, n_big: int = 40, n_small: int = 10) -> MeshData:
    """Torus about z (ring radius R, tube radius r)."""
    a = np.linspace(0, TAU, n_big + 1)
    b = np.linspace(0, TAU, n_small, endpoint=False)
    rings = np.stack([np.stack([(R + r * np.cos(b)) * math.cos(ai), (R + r * np.cos(b)) * math.sin(ai),
                                r * np.sin(b)], -1) for ai in a])
    uv = np.zeros(rings.shape[:2] + (2,))
    uv[..., 0] = (a / TAU)[:, None]
    uv[..., 1] = (b / TAU)[None]
    md = tube(rings, uv, cap_start=False, cap_end=False)
    return md


def plate_mesh(radius: float) -> MeshData:
    """A dinner plate (surface of revolution): a foot ring, the well and a raised
    rim, about ``radius`` across; the base is at z = 0."""
    R = radius
    r = np.array([0.0, 0.55, 0.60, 0.62, 0.64, 0.80, 0.97, 1.0, 0.99, 0.80, 0.62, 0.0]) * R
    z = np.array([0.020, 0.020, 0.0, 0.0, 0.020, 0.030, 0.085, 0.10, 0.115, 0.07, 0.055, 0.055]) * R
    # a closed profile: bottom (outward) then back over the top (inward)
    return lathe(r, z, 40, v_coord=np.linspace(0, 1, len(r)))


def floret_mesh(size: float, seed: int = 0) -> MeshData:
    """A broccoli floret: a pale stalk with a crown of bumpy green buds (merged
    spheres). ``size`` ~ the crown diameter; the stalk base is at z = 0."""
    rng = np.random.default_rng(seed + 77)
    s = size
    stalk = lathe(np.array([0.0, 0.16, 0.14, 0.12, 0.0]) * s, np.array([0.0, 0.0, 0.25, 0.42, 0.45]) * s,
                  10, v_coord=np.full(5, 0.95))
    parts = [stalk]
    for k in range(11):
        th = rng.uniform(0, TAU)
        rr = rng.uniform(0.0, 0.30) * s if k else 0.0
        zc = (0.62 + 0.1 * (1 - rr / (0.30 * s + 1e-9))) * s
        parts.append(sphere_mesh(rng.uniform(0.17, 0.22) * s, (rr * math.cos(th), rr * math.sin(th), zc),
                                 10, 7))
    md = merge(*parts)
    # uv: v ~ height (the texture darkens toward the crown top); the stalk is pale
    zn = md.verts[:, 2] / max(md.verts[:, 2].max(), 1e-9)
    md.uv = np.column_stack([md.uv[:, 0], np.where(md.uv[:, 1] >= 0.95, 0.97, 0.6 * zn)])
    return md


def can_mesh(r: float, h: float) -> MeshData:
    """A soda can: tapered top and bottom rims (the texture wraps round it)."""
    pr = np.array([0.0, 0.80, 0.97, 1.0, 1.0, 0.97, 0.80, 0.78, 0.0]) * r
    pz = np.array([0.0, 0.0, 0.03, 0.08, 0.90, 0.95, 0.99, 1.0, 1.0]) * h
    return lathe(pr, pz, 28, v_coord=np.array([0.0, 0.0, 0.02, 0.06, 0.94, 0.98, 1.0, 1.0, 1.0]))


def shade_mesh(r0: float, r1: float, h: float) -> MeshData:
    """Lamp shade: a truncated cone (open look via a thin wall)."""
    return lathe(np.array([r1, r1 * 0.99, r0 * 0.99, r0]) , np.array([0.0, 0.0, h, h]), 24)


def trophy_mesh(h: float) -> MeshData:
    """A trophy cup on a stem and base."""
    pr = np.array([0.0, 0.34, 0.34, 0.22, 0.10, 0.08, 0.12, 0.30, 0.42, 0.44, 0.0]) * h
    pz = np.array([0.0, 0.0, 0.12, 0.14, 0.20, 0.42, 0.50, 0.62, 0.85, 1.0, 1.0]) * h
    return lathe(pr, pz, 24)


def pot_mesh(r: float, h: float) -> MeshData:
    pr = np.array([0.0, 0.72, 0.9, 1.0, 1.02, 0.0]) * r
    pz = np.array([0.0, 0.0, 0.7, 0.88, 1.0, 1.0]) * h
    return lathe(pr, pz, 20)


def leaf_mesh(L: float, W: float, bend: float = 0.3) -> MeshData:
    """A flat pointed leaf along +x (thin double-sided shell)."""
    n = 12
    t = np.linspace(0, 1, n)
    half = W / 2 * np.sin(math.pi * t) ** 0.8
    x = L * t
    z = bend * L * t ** 2
    rings = []
    for k in range(n):
        rings.append([[x[k], -half[k], z[k]], [x[k], 0.0, z[k] + 0.01], [x[k], half[k], z[k]],
                      [x[k], 0.0, z[k] - 0.01]])
    rings = np.array(rings, float)
    uv = np.zeros(rings.shape[:2] + (2,))
    uv[..., 1] = t[:, None]
    return tube(rings, uv)


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------


def _bgr(c) -> tuple[int, int, int]:
    """RGB float colour -> cv2 BGR ints."""
    return (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))


def _text(img, text, org, scale, rgb, thick=2, font=cv2.FONT_HERSHEY_DUPLEX) -> None:
    cv2.putText(img, text, org, font, scale, _bgr(rgb), thick, cv2.LINE_AA)


def _synthwave(h: int, w: int, seed: int = 0) -> np.ndarray:
    """A generic retro 'synthwave' landscape (float BGR): a striped sun over a neon
    grid floor and purple mountains."""
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    sky = (np.array([0.35, 0.05, 0.20], np.float32) * (1 - y) + np.array([0.35, 0.30, 0.95], np.float32) * y)
    img = np.ones((h, w, 3), np.float32) * sky  # BGR
    hz = int(h * 0.62)
    # the sun
    cx, cy, r = w // 2, int(h * 0.50), int(h * 0.26)
    sun = np.zeros((h, w), np.float32)
    cv2.circle(sun, (cx, cy), r, 1.0, -1, cv2.LINE_AA)
    yy = np.arange(h)[:, None]
    stripes = ((yy - (cy - r * 0.1)) % max(r // 4, 3) < max(r // 12, 1)) & (yy > cy - r * 0.1)
    sun = sun * (~stripes)
    sun_col = np.array([0.15, 0.55, 1.0], np.float32) * (1 - (yy / h)[..., None]) + \
        np.array([0.55, 0.15, 1.0], np.float32) * (yy / h)[..., None]
    img = img * (1 - sun[..., None]) + sun_col * sun[..., None]
    # mountains
    rng = np.random.default_rng(seed + 5)
    xs = np.arange(w)
    ridge = hz - (0.12 * h * (0.5 + 0.5 * np.sin(xs / w * 7 + 1.3)) + 0.06 * h * rng.random(w).cumsum() % 1)
    ridge = cv2.GaussianBlur(ridge.astype(np.float32)[None], (1, 31), 0)[0]
    mask = (yy >= ridge[None]) & (yy < hz)
    img[mask] = np.array([0.28, 0.05, 0.18], np.float32)
    # the grid floor
    floor = yy[:, 0] >= hz
    img[floor] = np.array([0.12, 0.0, 0.10], np.float32)
    grid = np.zeros((h, w), np.float32)
    for k in range(-14, 15):
        cv2.line(grid, (cx, hz), (int(cx + k * w * 0.18), h), 1.0, 1, cv2.LINE_AA)
    for k in range(1, 12):
        yk = int(hz + (h - hz) * (k / 11) ** 2)
        cv2.line(grid, (0, yk), (w, yk), 1.0, 1, cv2.LINE_AA)
    grid[:hz] = 0
    img = img * (1 - grid[..., None]) + np.array([0.9, 0.2, 1.0], np.float32) * grid[..., None]
    return img


def _fly_icon(img, cx, cy, s, body=(0.15, 0.15, 0.18), eye=(0.85, 0.12, 0.10)) -> None:
    """A generic cartoon fly (webcam / avatar)."""
    cv2.ellipse(img, (cx, cy + int(0.35 * s)), (int(0.42 * s), int(0.55 * s)), 0, 0, 360, _bgr(body), -1,
                cv2.LINE_AA)
    cv2.circle(img, (cx, cy - int(0.15 * s)), int(0.38 * s), _bgr(body), -1, cv2.LINE_AA)
    for sx in (-1, 1):
        cv2.circle(img, (cx + sx * int(0.28 * s), cy - int(0.25 * s)), int(0.24 * s), _bgr(eye), -1,
                   cv2.LINE_AA)
        cv2.ellipse(img, (cx + sx * int(0.55 * s), cy + int(0.1 * s)), (int(0.45 * s), int(0.18 * s)),
                    sx * 25, 0, 360, _bgr((0.75, 0.82, 0.95)), -1, cv2.LINE_AA)


@lru_cache(maxsize=4)
def stream_texture(kind: str = "main", h: int = 288, w: int = 512, seed: int = 0) -> np.ndarray:
    """A generic live-stream layout (RGB uint8, row 0 = top):

    * ``"main"``: a game view (synthwave landscape) with a webcam box (a cartoon
      fly), a red LIVE badge, a follow-goal bar, and the chat column on the right
      (header only: the job scrolls chat blocks over it);
    * ``"dash"``: a stream dashboard: a viewer graph going up, chat, LIVE.
    """
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = _bgr((0.055, 0.05, 0.08))
    chat_x = int(w * 0.72)
    if kind == "main":
        gh = int(h * 0.84)
        game = _synthwave(gh, chat_x - 8, seed)
        img[4:4 + gh, 4:chat_x - 4] = _to_u8(game)
        # webcam box, bottom left of the game view
        bx0, by0, bx1, by1 = 12, gh - 70, 104, gh - 4
        cv2.rectangle(img, (bx0, by0), (bx1, by1), _bgr((0.10, 0.10, 0.16)), -1)
        cv2.rectangle(img, (bx0, by0), (bx1, by1), _bgr((0.65, 0.25, 1.0)), 2)
        _fly_icon(img, (bx0 + bx1) // 2, (by0 + by1) // 2 - 4, 30)
        # LIVE badge
        cv2.rectangle(img, (12, 12), (82, 38), _bgr((0.90, 0.08, 0.10)), -1)
        _text(img, "LIVE", (19, 33), 0.72, (1, 1, 1), 2)
        cv2.circle(img, (96, 25), 6, _bgr((0.95, 0.95, 0.95)), -1, cv2.LINE_AA)
        # follow goal bar under the game view
        y0 = 4 + gh + 6
        cv2.rectangle(img, (8, y0), (chat_x - 8, h - 8), _bgr((0.10, 0.09, 0.15)), -1)
        cv2.rectangle(img, (14, y0 + 8), (int((chat_x - 14) * 0.62), h - 16), _bgr((0.35, 0.85, 0.45)), -1)
        _text(img, "FOLLOW GOAL", (int(chat_x * 0.66), h - 16), 0.42, (0.9, 0.9, 0.95), 1)
    else:
        # dashboard: viewer graph
        gx0, gy0, gx1, gy1 = 10, 44, chat_x - 10, h - 12
        cv2.rectangle(img, (gx0, gy0), (gx1, gy1), _bgr((0.08, 0.08, 0.12)), -1)
        for k in range(1, 5):
            yk = gy0 + (gy1 - gy0) * k // 5
            cv2.line(img, (gx0, yk), (gx1, yk), _bgr((0.18, 0.18, 0.25)), 1)
        rng = np.random.default_rng(seed + 3)
        n = 40
        vals = np.cumsum(rng.random(n) * 0.6 + 0.1)
        vals[25:] += np.linspace(0, 6, n - 25) ** 1.4  # the explosions go viral
        vals = vals / vals.max()
        pts = np.stack([np.linspace(gx0 + 4, gx1 - 4, n), gy1 - 6 - vals * (gy1 - gy0 - 20)], 1).astype(np.int32)
        poly = np.concatenate([pts, [[gx1 - 4, gy1 - 2], [gx0 + 4, gy1 - 2]]]).astype(np.int32)
        over = img.copy()
        cv2.fillPoly(over, [poly], _bgr((0.25, 0.10, 0.45)))
        img[:] = cv2.addWeighted(over, 0.7, img, 0.3, 0)
        cv2.polylines(img, [pts], False, _bgr((0.75, 0.35, 1.0)), 3, cv2.LINE_AA)
        cv2.rectangle(img, (12, 10), (82, 36), _bgr((0.90, 0.08, 0.10)), -1)
        _text(img, "LIVE", (19, 31), 0.72, (1, 1, 1), 2)
        _text(img, "VIEWERS", (96, 31), 0.6, (0.85, 0.85, 0.9), 1)
    # chat column
    cv2.rectangle(img, (chat_x, 4), (w - 4, h - 4), _bgr((0.09, 0.085, 0.13)), -1)
    cv2.rectangle(img, (chat_x, 4), (w - 4, 30), _bgr((0.16, 0.12, 0.26)), -1)
    _text(img, "CHAT", (chat_x + 10, 24), 0.55, (0.95, 0.95, 1.0), 1)
    return img[..., ::-1].copy()


@lru_cache(maxsize=8)
def art_poster_texture(kind: str, h: int = 256, w: int = 192, seed: int = 0) -> np.ndarray:
    """Generic poster art (RGB): ``"sunset"`` (synthwave + 'GG'), ``"planet"``
    (a ringed planet + 'LEVEL UP'), ``"pad"`` (a game pad + 'PRESS START')."""
    img = np.zeros((h, w, 3), np.uint8)
    if kind == "sunset":
        img[:] = _to_u8(_synthwave(h, w, seed))
        _text(img, "GG", (w // 2 - 44, 70), 2.2, (1.0, 0.9, 0.3), 5)
    elif kind == "planet":
        rng = np.random.default_rng(seed + 9)
        img[:] = _bgr((0.03, 0.02, 0.10))
        for _ in range(90):
            x, y = rng.integers(0, w), rng.integers(0, h)
            cv2.circle(img, (int(x), int(y)), int(rng.integers(1, 3)), _bgr((0.9, 0.9, 1.0)), -1)
        cx, cy = w // 2, h // 2 - 10
        cv2.circle(img, (cx, cy), 52, _bgr((0.95, 0.45, 0.25)), -1, cv2.LINE_AA)
        cv2.circle(img, (cx - 14, cy - 16), 20, _bgr((1.0, 0.6, 0.35)), -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx, cy), (86, 20), -15, 0, 360, _bgr((0.6, 0.85, 1.0)), 5, cv2.LINE_AA)
        _text(img, "LEVEL UP", (18, h - 26), 1.05, (0.55, 1.0, 0.85), 2)
    else:  # pad
        img[:] = _bgr((0.10, 0.03, 0.16))
        cx, cy = w // 2, h // 2 - 16
        cv2.ellipse(img, (cx - 36, cy + 10), (40, 34), 0, 0, 360, _bgr((0.15, 0.95, 0.85)), -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx + 36, cy + 10), (40, 34), 0, 0, 360, _bgr((0.15, 0.95, 0.85)), -1, cv2.LINE_AA)
        cv2.rectangle(img, (cx - 40, cy - 22), (cx + 40, cy + 30), _bgr((0.15, 0.95, 0.85)), -1)
        cv2.rectangle(img, (cx - 52, cy + 2), (cx - 22, cy + 10), _bgr((0.10, 0.03, 0.16)), -1)
        cv2.rectangle(img, (cx - 41, cy - 9), (cx - 33, cy + 21), _bgr((0.10, 0.03, 0.16)), -1)
        for k, col in enumerate(((1, 0.3, 0.4), (1, 0.85, 0.2), (0.3, 0.5, 1), (0.4, 1, 0.4))):
            a = k * math.pi / 2
            cv2.circle(img, (int(cx + 36 + 12 * math.cos(a)), int(cy + 6 + 12 * math.sin(a))), 5, _bgr(col), -1,
                       cv2.LINE_AA)
        _text(img, "PRESS", (40, h - 58), 1.1, (1.0, 0.35, 0.75), 2)
        _text(img, "START", (40, h - 22), 1.1, (1.0, 0.35, 0.75), 2)
    cv2.rectangle(img, (3, 3), (w - 4, h - 4), _bgr((0.9, 0.9, 0.95)), 2)
    return img[..., ::-1].copy()


@lru_cache(maxsize=2)
def foam_texture(n: int = 256, cells: int = 4) -> np.ndarray:
    """Acoustic foam panels: wedge tiles alternating direction (charcoal)."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n * cells
    fx, fy = x % 1.0, y % 1.0
    alt = ((np.floor(x) + np.floor(y)) % 2).astype(bool)
    wave = np.where(alt, 0.5 + 0.5 * np.cos(fx * TAU * 2), 0.5 + 0.5 * np.cos(fy * TAU * 2))
    edge = np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy))
    shade = 0.55 + 0.45 * wave
    shade *= 0.6 + 0.4 * _smooth(0.0, 0.05, edge)
    img = shade[..., None] * np.array([0.16, 0.15, 0.20], np.float32)
    return _to_u8(img)


@lru_cache(maxsize=2)
def wood_floor_texture(seed: int = 0, n: int = 512, planks: int = 6) -> np.ndarray:
    """Dark wooden floor boards."""
    rng = np.random.default_rng(seed + 31)
    img = np.zeros((n, n, 3), np.float32)
    ph = n // planks
    grain = fbm(n, n, 64, 4, rng, 3)
    for k in range(planks):
        tone = rng.uniform(0.75, 1.1)
        off = int(rng.integers(0, n))
        g = np.roll(grain, off, 1)[k * ph:(k + 1) * ph]
        base = np.array([0.20, 0.13, 0.09], np.float32) * tone
        img[k * ph:(k + 1) * ph] = base * (0.75 + 0.5 * g)[..., None]
        img[k * ph:k * ph + 2] *= 0.4
        cut = int(rng.integers(0, n))
        img[k * ph:(k + 1) * ph, cut:cut + 2] *= 0.4
    return _to_u8(img)


@lru_cache(maxsize=2)
def rug_texture(n: int = 256) -> np.ndarray:
    """Round gaming rug: concentric purple / black rings with a neon edge."""
    y, x = (np.mgrid[0:n, 0:n].astype(np.float32) + 0.5) / n * 2 - 1
    r = np.hypot(x, y)
    img = np.ones((n, n, 3), np.float32) * np.array([0.06, 0.04, 0.10], np.float32)
    ring = (np.sin(r * 22) > 0.6) & (r < 0.9)
    img[ring] = (0.25, 0.08, 0.40)
    img[(r > 0.9) & (r < 0.97)] = (0.10, 0.85, 0.95)
    return _to_u8(img)


@lru_cache(maxsize=2)
def cardboard_texture(seed: int = 0, n: int = 128) -> np.ndarray:
    """A cardboard box face: brown with packing tape and a 'FRAGILE' stamp."""
    rng = np.random.default_rng(seed + 41)
    base = np.array([0.62, 0.45, 0.26], np.float32) * (0.88 + 0.22 * fbm(n, n, 8, 8, rng, 3))[..., None]
    img = _to_u8(base)[..., ::-1].copy()  # BGR for cv2
    t = n // 7
    img[n // 2 - t // 2:n // 2 + t // 2] = _bgr((0.80, 0.67, 0.45))
    _text(img, "FRAGILE", (n // 2 - 46, n // 4 + 6), 0.55, (0.75, 0.10, 0.08), 2)
    for xk in (n // 2 - 26, n // 2 + 18):
        cv2.arrowedLine(img, (xk, n - 12), (xk, n - 40), _bgr((0.16, 0.12, 0.08)), 3, tipLength=0.4)
    return img[..., ::-1].copy()


@lru_cache(maxsize=4)
def can_texture(colour=(0.85, 0.08, 0.10), h: int = 64, w: int = 128) -> np.ndarray:
    """Soda can wrap: a colour band with a white wave and generic 'FIZZ' text."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = _bgr(colour)
    xs = np.arange(w)
    wave = (h / 2 + 8 * np.sin(xs / w * TAU * 2)).astype(np.int32)
    for x, yv in zip(xs, wave):
        img[yv - 3:yv + 3, x] = 250
    _text(img, "FIZZ", (8, h // 2 - 8), 0.6, (1, 1, 1), 2)
    img[:4] = img[-4:] = 190
    return img[..., ::-1].copy()


@lru_cache(maxsize=2)
def broccoli_texture(n: int = 128, seed: int = 0) -> np.ndarray:
    """v < 0.6: the crown, bumpy dark green (darker toward the top); v > 0.9: the pale
    stalk."""
    rng = np.random.default_rng(seed + 51)
    bump = fbm(n, n, 16, 16, rng, 3)
    v = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    crown = np.array([0.10, 0.38, 0.10], np.float32) * (0.6 + 0.7 * bump)[..., None] \
        * (0.8 + 0.4 * (1 - v))[..., None]
    stalk = np.array([0.55, 0.72, 0.38], np.float32) * (0.9 + 0.2 * bump)[..., None]
    img = np.where((v > 0.85)[..., None], stalk, crown)
    return _to_u8(img)


@lru_cache(maxsize=2)
def plate_texture(n: int = 64) -> np.ndarray:
    """Plate glaze along the lathe profile (v): white with a blue rim band."""
    v = np.linspace(0, 1, n, dtype=np.float32)[:, None] * np.ones((1, 8), np.float32)
    img = np.ones((n, 8, 3), np.float32) * np.array([0.94, 0.94, 0.92], np.float32)
    band = (v > 0.58) & (v < 0.76)
    img[band] = (0.15, 0.30, 0.75)
    return _to_u8(img)


@lru_cache(maxsize=2)
def fridge_texture(h: int = 256, w: int = 160) -> np.ndarray:
    """Mini fridge door: glossy black with a neon sticker and a handle groove."""
    img = np.zeros((h, w, 3), np.uint8)
    g = np.linspace(0.10, 0.04, w, dtype=np.float32)[None, :, None]
    img[:] = _to_u8(np.ones((h, 1, 1), np.float32) * g * np.array([1.0, 1.0, 1.1], np.float32))
    cv2.rectangle(img, (w - 26, 30), (w - 16, h - 30), (60, 60, 64), -1)
    cv2.circle(img, (w // 2 - 14, h // 3), 26, _bgr((1.0, 0.25, 0.7)), 3, cv2.LINE_AA)
    _text(img, "SNACKS", (14, h // 3 + 60), 0.7, (0.3, 0.95, 1.0), 2)
    return img[..., ::-1].copy()


@lru_cache(maxsize=2)
def door_texture(h: int = 256, w: int = 128) -> np.ndarray:
    """The open kitchen doorway: a warm lit room beyond (counter, tiles)."""
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = np.ones((h, w, 3), np.float32) * (np.array([1.0, 0.85, 0.60], np.float32) * (1 - 0.5 * y))
    img[int(h * 0.62):int(h * 0.66)] = (0.45, 0.30, 0.20)
    img[int(h * 0.66):] = (0.75, 0.70, 0.62)
    for k in range(0, w, 16):
        img[int(h * 0.30):int(h * 0.62), k] = (0.85, 0.72, 0.52)
    return _to_u8(img)


# ---------------------------------------------------------------------------
# blast puff texture (a cube map: the puffs are ellipsoids)
# ---------------------------------------------------------------------------


def puff_cube_texture(seed: int = 0, n: int = 64, lo: float = 0.35) -> np.ndarray:
    """Six (n, n) faces of billowy fractal noise stacked vertically (6n, n, 3), grey
    values lo..1 (they multiply the puff colour): the fire / smoke puffs get lumps,
    dark folds and bright cores instead of flat shading."""
    rng = np.random.default_rng(seed)
    faces = []
    for _ in range(6):
        v = fbm(n, n, 3, 3, rng, octaves=3, gain=0.5)
        v = (v - v.min()) / max(float(v.max() - v.min()), 1e-6)
        v = 1.0 - np.abs(2.0 * v - 1.0)  # "billows": ridges where the noise crosses 0.5
        faces.append(lo + (1.0 - lo) * v ** 0.8)
    img = np.concatenate(faces, 0)
    return _to_u8(np.repeat(img[..., None], 3, axis=2))


def add_cube_texture(spec, name: str, img: np.ndarray):
    """Cube texture from six stacked square faces (6n, n, 3) uint8."""
    h, w, _ = img.shape
    tex = spec.add_texture(name=name, type=mj.mjtTexture.mjTEXTURE_CUBE, width=w, height=h, nchannel=3)
    tex.data = np.ascontiguousarray(img).tobytes()
    return tex
