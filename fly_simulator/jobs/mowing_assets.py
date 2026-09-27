"""Procedural meshes and textures for the lawn mowing job (visual only).

Same approach as ``kebab_assets``: everything is generated in code (numpy + OpenCV)
and handed to MjSpec directly; nothing is written to disk. Nothing here collides or
has mass (callers add the meshes with ``contact_kwargs("visual")`` and ``mass=0``).

A suburban front yard:

* meshes: the push mower (a domed red deck shell drawn over the colliding
  cylinder, a finned engine with its shroud, pull-start and air filter, tyres with
  hubcaps, a bent tube handle with the bail bar and grip), a white picket fence,
  the house (walls, a gabled roof, porch steps), flower clumps, shrubs, a mailbox and
  a garden gnome;
* textures: yard grass, a grey grass-detail texture that the job's stripe colours
  tint, thatch / soil under the blades, red brick edging, a concrete sidewalk, the
  house facade (lap siding, windows with shutters, a front door), roof shingles,
  mulch, tyre tread.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.broccoli_toss_assets import merge, sphere_mesh, torus_mesh  # noqa: F401
from fly_simulator.jobs.kebab_assets import (  # noqa: F401  (re-exported helpers)
    MeshData, _mix, _smooth, _to_u8, add_mesh, add_texture, add_textured_material, fbm, lathe,
    tile_noise, transform, tube,
)
from fly_simulator.jobs.sisyphus_assets import box_mesh  # noqa: F401
from fly_simulator.jobs.trampoline_assets import lawn_texture, polyline_tube  # noqa: F401

TAU = 2.0 * math.pi


def _norm(v):
    return (v - v.min()) / max(float(v.max() - v.min()), 1e-6)


# ---------------------------------------------------------------------------
# meshes: the mower (body frame: +x forward, deck centre on the z axis)
# ---------------------------------------------------------------------------


def deck_shell_mesh(R: float, z0: float, z1: float) -> MeshData:
    """The mower deck: a skirt from z0 up, a rolled edge, then a dome rising to a
    flat top at z1 (radius R; the colliding cylinder is R, z0..z1)."""
    h = z1 - z0
    pr = np.array([0.0, R * 0.97, R * 1.01, R * 1.03, R * 1.03, R * 1.0, R * 0.9, R * 0.72,
                   R * 0.55, 0.0])
    pz = z0 + h * np.array([0.0, 0.0, 0.03, 0.12, 0.5, 0.62, 0.78, 0.93, 1.0, 1.0])
    return lathe(pr, pz, 48, v_coord=np.linspace(0, 1, len(pr)))


def engine_meshes(r: float, z0: float, h: float) -> dict[str, MeshData]:
    """``block``: a finned cylinder (cooling fins) standing on z0; ``shroud``: a
    rounded cap over it; ``starter``: the pull-start handle on a short cord."""
    n = 9
    pz, pr = [z0, z0], [0.0, 0.8 * r]
    for k in range(n):  # fins: alternating radius
        z = z0 + 0.06 * h + k * 0.6 * h / n
        pz += [z, z + 0.3 * h / n, z + 0.35 * h / n, z + 0.6 * h / n]
        pr += [0.8 * r, 0.8 * r, r, r]
    pz += [z0 + 0.66 * h, z0 + 0.66 * h]
    pr += [0.8 * r, 0.0]
    block = lathe(np.array(pr), np.array(pz), 28)
    sz = z0 + 0.62 * h + np.array([0.0, 0.0, 0.12, 0.28, 0.38, 0.38]) * h
    sr = np.array([0.0, 1.08, 1.1, 0.95, 0.6, 0.0]) * r
    shroud = lathe(sr, sz, 32)
    top = z0 + h
    grip = transform(polyline_tube(np.array([[0, -0.28 * r, 0], [0, 0.28 * r, 0]]), 0.07 * r * 1.6, 8),
                     np.eye(3), np.array([-0.2 * r, 0, top + 0.12 * h]))
    cord = polyline_tube(np.array([[-0.2 * r, 0, top - 0.05 * h], [-0.2 * r, 0, top + 0.1 * h]]),
                         0.025 * r, 5)
    return {"block": block, "shroud": shroud, "starter": merge(grip, cord)}


def tyre_mesh(r: float, w: float, n: int = 28) -> MeshData:
    """A tyre about the y axis (rounded profile), centred at the origin; uv u
    around the tread (a tread texture shows)."""
    a = np.linspace(-math.pi / 2, math.pi / 2, 9)
    prof_r = r - 0.25 * r * (1 - np.cos(a))  # rounded shoulders
    prof_y = 0.5 * w * np.sin(a)
    pr = np.concatenate([[0.55 * r], prof_r, [0.55 * r]])
    py = np.concatenate([[-0.5 * w], prof_y, [0.5 * w]])
    md = lathe(pr, py, n, v_coord=np.linspace(0, 1, len(pr)), u_repeat=6.0)
    R = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # lathe z -> y
    return transform(md, R.T, np.zeros(3))


def handle_mesh(foot_x: float, foot_z: float, grip_x: float, grip_z: float, half_w: float,
                r: float = 0.06) -> MeshData:
    """The bent tube handle: two arms from the deck's rear (foot) back and up to the
    grip height, joined by a lower cross brace, plus the bail (safety) bar that
    hangs in front of the grip."""
    parts = []
    for s in (-1.0, 1.0):
        pts = np.array([[foot_x, s * half_w, foot_z],
                        [foot_x - 0.25, s * half_w, foot_z + 0.3],
                        [0.5 * (foot_x + grip_x), s * half_w * 1.05, 0.5 * (foot_z + grip_z) + 0.1],
                        [grip_x + 0.05, s * half_w * 1.1, grip_z - 0.05],
                        [grip_x, s * half_w * 1.1, grip_z]])
        parts.append(polyline_tube(pts, r, 8))
    mx, mz = 0.55 * foot_x + 0.45 * grip_x, 0.55 * foot_z + 0.45 * grip_z
    parts.append(polyline_tube(np.array([[mx, -half_w * 1.05, mz], [mx, half_w * 1.05, mz]]), r * 0.8, 8))
    bx, bz = grip_x + 0.18, grip_z - 0.12
    bail = np.array([[grip_x + 0.02, -half_w * 1.1, grip_z - 0.02], [bx, -half_w * 0.95, bz],
                     [bx, half_w * 0.95, bz], [grip_x + 0.02, half_w * 1.1, grip_z - 0.02]])
    parts.append(polyline_tube(bail, r * 0.6, 6))
    return merge(*parts)


# ---------------------------------------------------------------------------
# meshes: the yard
# ---------------------------------------------------------------------------


def picket_fence_mesh(x0: float, x1: float, h: float, pitch: float = 0.9,
                      pw: float = 0.32, t: float = 0.1) -> MeshData:
    """A picket fence along x on y = 0, base z = 0: pointed pickets (extruded
    pentagons), two rails and posts every 6 pickets."""
    parts = []
    n = int((x1 - x0) / pitch) + 1
    for k in range(n):
        x = x0 + k * pitch
        outline = np.array([[x - pw / 2, 0.0], [x + pw / 2, 0.0], [x + pw / 2, h * 0.88],
                            [x, h], [x - pw / 2, h * 0.88]])
        rings = np.stack([np.column_stack([outline[:, 0], np.full(5, yy), outline[:, 1]])
                          for yy in (-t / 2, t / 2)])
        parts.append(tube(rings, np.zeros(rings.shape[:2] + (2,))))
    for zr in (0.25 * h, 0.7 * h):
        parts.append(box_mesh(((x1 - x0) / 2 + 0.2, t * 0.7, 0.09), 1.0))
        parts[-1] = transform(parts[-1], np.eye(3), np.array([(x0 + x1) / 2, t * 0.9, zr]))
    for k in range(0, n, 6):
        x = x0 + k * pitch + pitch / 2
        parts.append(transform(box_mesh((0.2, 0.2, h * 0.47), 1.0), np.eye(3),
                               np.array([x, t * 1.5, h * 0.47])))
    return merge(*parts)


def gable_roof_mesh(w: float, d: float, rise: float, over: float = 0.6) -> MeshData:
    """A gabled roof prism, ridge along x, eaves at z = 0: a closed triangular
    prism (w along x, d along y). uv: u along x, v up the slope (shingle rows)."""
    hw, hd = w / 2 + over, d / 2 + over
    tri = np.array([[-hd, 0.0], [hd, 0.0], [0.0, rise]])  # (y, z)
    rings = np.stack([np.column_stack([np.full(3, x), tri]) for x in (-hw, hw)])
    uv = np.zeros(rings.shape[:2] + (2,))
    uv[..., 0] = np.array([0.0, 1.0])[:, None] * (2 * hw) / 4.0
    uv[..., 1] = np.array([0.0, 0.0, 1.0])[None] * (hd / 1.5)
    return tube(rings, uv)


def flower_clump_mesh(centre, n: int, r: float, size: float, seed: int = 0) -> MeshData:
    """A clump of little round flowers (flattened spheres) over a dome of radius r."""
    rng = np.random.default_rng(seed + 501)
    parts = []
    for _ in range(n):
        rr = r * math.sqrt(rng.uniform())
        a = rng.uniform(0, TAU)
        z = 0.5 * size + 0.6 * r * (1 - (rr / r) ** 2)
        c = (centre[0] + rr * math.cos(a), centre[1] + rr * math.sin(a), z)
        parts.append(sphere_mesh(size * rng.uniform(0.7, 1.1), c, 8, 6, squash=0.6))
    return merge(*parts)


def shrub_mesh(centre, r: float, seed: int = 0, n: int = 7) -> MeshData:
    """A round shrub: overlapping spheres on the ground."""
    rng = np.random.default_rng(seed + 502)
    parts = []
    for _ in range(n):
        off = rng.normal(0, 0.35 * r, 3) * np.array([1.0, 1.0, 0.4])
        rr = r * rng.uniform(0.55, 0.8)
        parts.append(sphere_mesh(rr, (centre[0] + off[0], centre[1] + off[1], 0.75 * rr + max(off[2], 0)),
                                 12, 8))
    return merge(*parts)


def mailbox_meshes(h: float) -> dict[str, MeshData]:
    """A rural mailbox: a post, the box (a half-round-top tunnel along x) and a
    little red flag."""
    post = transform(box_mesh((0.12, 0.12, h / 2), 1.0), np.eye(3), np.array([0, 0, h / 2]))
    L, r = 1.4, 0.4
    a = np.linspace(0, math.pi, 12)
    loop = np.concatenate([[[r, 0.0]], np.column_stack([r * np.cos(a), r * np.sin(a)]), [[-r, 0.0]]])
    loop = np.concatenate([loop, [[-r, -0.3 * r], [r, -0.3 * r]]])
    rings = np.stack([np.column_stack([np.full(len(loop), x), loop[:, 0], loop[:, 1] + h + 0.3 * r])
                      for x in (-L / 2, L / 2)])
    box = tube(rings, np.zeros(rings.shape[:2] + (2,)))
    flag = merge(transform(box_mesh((0.03, 0.02, 0.35), 1.0), np.eye(3), np.array([0.2, r + 0.03, h + 0.5])),
                 transform(box_mesh((0.2, 0.02, 0.1), 1.0), np.eye(3), np.array([0.35, r + 0.03, h + 0.78])))
    return {"post": post, "box": box, "flag": flag}


def gnome_meshes(h: float) -> dict[str, MeshData]:
    """A garden gnome on z = 0: blue ``body`` (coat), pink ``face``, white ``beard``
    and a red pointed ``hat``."""
    body = lathe(np.array([0.0, 0.32, 0.30, 0.22, 0.0]) * h, np.array([0.0, 0.0, 0.35, 0.55, 0.55]) * h, 16)
    face = sphere_mesh(0.14 * h, (0.04 * h, 0, 0.64 * h), 12, 8)
    beard = lathe(np.array([0.0, 0.17, 0.12, 0.0]) * h, np.array([0.38, 0.40, 0.6, 0.62]) * h, 14)
    beard = transform(beard, np.eye(3), np.array([0.07 * h, 0, 0]))
    hat = lathe(np.array([0.0, 0.17, 0.06, 0.0]) * h, np.array([0.70, 0.70, 0.95, 1.0]) * h, 14)
    return {"body": body, "face": face, "beard": beard, "hat": hat}


# ---------------------------------------------------------------------------
# textures (uint8 RGB, row 0 = v 0)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def grass_detail_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Light grey grass detail (tinted by the stripe colours at run time): mottling
    and fine blade streaks, 0.7..1.0."""
    rng = np.random.default_rng(seed + 511)
    f = _norm(fbm(n, n, 8, 8, rng, 4))
    img = np.repeat((0.80 + 0.14 * f)[..., None], 3, axis=2)
    u8 = _to_u8(img)
    for _ in range(1400):
        x, y = (int(v) for v in rng.integers(0, n, 2))
        L = int(rng.integers(3, 8))
        a = rng.normal(-math.pi / 2, 0.4)
        v = int(rng.integers(170, 256))
        cv2.line(u8, (x, y), (int(x + L * math.cos(a)), int(y + L * math.sin(a))), (v, v, v), 1,
                 cv2.LINE_AA)
    return u8


@lru_cache(maxsize=2)
def thatch_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """The ground between the blades: dark green thatch over soil."""
    rng = np.random.default_rng(seed + 512)
    f = _norm(fbm(n, n, 10, 10, rng, 4))
    img = _mix(np.array([0.14, 0.22, 0.08], np.float32), np.array([0.24, 0.34, 0.12], np.float32), f)
    u8 = _to_u8(img)
    for _ in range(900):
        x, y = (int(v) for v in rng.integers(0, n, 2))
        a = rng.uniform(0, TAU)
        L = int(rng.integers(3, 7))
        c = (int(rng.integers(70, 110)), int(rng.integers(80, 110)), int(rng.integers(30, 50)))
        cv2.line(u8, (x, y), (int(x + L * math.cos(a)), int(y + L * math.sin(a))), c, 1, cv2.LINE_AA)
    return u8


@lru_cache(maxsize=2)
def red_brick_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Red clay bricks (4 courses x 2, offset), grey mortar."""
    rng = np.random.default_rng(seed + 513)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    y = y / (n / 4)
    x = x / (n / 2) + 0.5 * (np.floor(y) % 2)
    fx, fy = x - np.floor(x), y - np.floor(y)
    edge = np.minimum(np.minimum(fx, 1 - fx) * 2.0, np.minimum(fy, 1 - fy))
    mortar = _smooth(0.08, 0.04, edge)
    n1 = fbm(n, n, 8, 8, rng, 4)
    bid = ((np.floor(x) * 7 + np.floor(y) * 13) % 5).astype(int)
    shade = np.array([0.92, 1.0, 0.86, 1.05, 0.96], np.float32)[bid]
    base = np.array([0.66, 0.28, 0.20], np.float32) * ((0.85 + 0.25 * n1) * shade)[..., None]
    return _to_u8(_mix(base, (0.70, 0.68, 0.64), mortar))


@lru_cache(maxsize=2)
def concrete_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Sidewalk concrete: pale grey with aggregate speckle and one expansion joint."""
    rng = np.random.default_rng(seed + 514)
    f = _norm(fbm(n, n, 6, 6, rng, 4))
    img = np.repeat((0.70 + 0.1 * f)[..., None], 3, axis=2) * np.array([1.0, 0.99, 0.96], np.float32)
    sp = rng.random((n, n))
    img = _mix(img, (0.45, 0.44, 0.42), (sp > 0.97).astype(np.float32) * 0.5)
    img[:, :3] = (0.42, 0.42, 0.40)
    return _to_u8(img)


@lru_cache(maxsize=2)
def mulch_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Bark mulch in the flower beds: dark red-brown chips."""
    rng = np.random.default_rng(seed + 515)
    img = np.full((n, n, 3), (48, 28, 18), np.uint8)
    for _ in range(1800):
        x, y = (int(v) for v in rng.integers(0, n, 2))
        v = rng.uniform(0.7, 1.4)
        col = tuple(int(min(255, c * v)) for c in (104, 60, 36))
        cv2.ellipse(img, (x, y), (int(rng.integers(2, 6)), int(rng.integers(1, 3))),
                    float(rng.uniform(0, 180)), 0, 360, col, -1, cv2.LINE_AA)
    return img


@lru_cache(maxsize=2)
def facade_texture(h: int = 512, w: int = 1024, siding=(0.62, 0.74, 0.84)) -> np.ndarray:
    """The house front: lap siding, white trim, four windows with shutters and
    flower boxes, a red front door with a small window, a porch light, the house
    number. Row 0 = the top (eaves)."""
    y = np.arange(h, dtype=np.float32)[:, None]
    lap = (y % (h / 22)) / (h / 22)
    base = np.asarray(siding, np.float32) * (0.86 + 0.14 * lap)[..., None]
    img = _to_u8(np.broadcast_to(base, (h, w, 3)).copy())
    white, dark = (245, 245, 240), (40, 44, 52)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), white, 14)
    img[h - 30:] = (150, 150, 150)  # the foundation

    def window(cx, cy, ww, wh):
        cv2.rectangle(img, (cx - ww // 2 - 10, cy - wh // 2 - 10), (cx + ww // 2 + 10, cy + wh // 2 + 10),
                      white, -1)
        ys, xs = slice(cy - wh // 2, cy + wh // 2), slice(cx - ww // 2, cx + ww // 2)
        n_y = ys.stop - ys.start
        g = np.linspace(0, 1, n_y, dtype=np.float32)[:, None, None]
        glass = _to_u8(np.array([0.30, 0.42, 0.55], np.float32) * (1 - g) + np.array([0.60, 0.72, 0.82], np.float32) * g)
        img[ys, xs] = glass
        cv2.line(img, (cx, cy - wh // 2), (cx, cy + wh // 2), white, 6)
        cv2.line(img, (cx - ww // 2, cy), (cx + ww // 2, cy), white, 6)
        for s in (-1, 1):  # shutters
            x0 = cx + s * (ww // 2 + 12) + (0 if s > 0 else -ww // 3)
            cv2.rectangle(img, (x0, cy - wh // 2 - 8), (x0 + ww // 3, cy + wh // 2 + 8), (36, 70, 60), -1)
            for k in range(8):
                yy = cy - wh // 2 + k * wh // 8
                cv2.line(img, (x0 + 3, yy), (x0 + ww // 3 - 3, yy), (28, 56, 48), 2)
        cv2.rectangle(img, (cx - ww // 2 - 6, cy + wh // 2 + 12), (cx + ww // 2 + 6, cy + wh // 2 + 30),
                      (120, 76, 44), -1)  # flower box
        for k in range(9):
            col = ((230, 60, 80), (250, 200, 60), (240, 240, 240))[k % 3]
            cv2.circle(img, (cx - ww // 2 + k * ww // 8, cy + wh // 2 + 10), 7, col, -1, cv2.LINE_AA)

    for cx in (int(w * 0.13), int(w * 0.33), int(w * 0.67), int(w * 0.87)):
        window(cx, int(h * 0.42), int(w * 0.09), int(h * 0.30))
    # the front door
    dx, dw, dh = w // 2, int(w * 0.10), int(h * 0.58)
    cv2.rectangle(img, (dx - dw // 2 - 12, h - 30 - dh - 12), (dx + dw // 2 + 12, h - 30), white, -1)
    cv2.rectangle(img, (dx - dw // 2, h - 30 - dh), (dx + dw // 2, h - 30), (150, 30, 32), -1)
    for k in range(2):
        for j in range(2):
            x0 = dx - dw // 2 + 12 + j * (dw // 2 - 6)
            y0 = h - 30 - dh + int(dh * 0.35) + k * int(dh * 0.32)
            cv2.rectangle(img, (x0, y0), (x0 + dw // 2 - 30, y0 + int(dh * 0.26)), (120, 22, 24), 3)
    cv2.rectangle(img, (dx - dw // 3, h - 30 - dh + 16), (dx + dw // 3, h - 30 - dh + int(dh * 0.25)),
                  (170, 200, 220), -1)
    cv2.circle(img, (dx + dw // 2 - 14, h - 30 - dh // 2), 6, (220, 190, 80), -1, cv2.LINE_AA)
    cv2.circle(img, (dx + dw // 2 + 40, h - 30 - dh + 30), 12, (255, 230, 150), -1, cv2.LINE_AA)
    cv2.putText(img, "42", (dx - dw // 2 - 70, h - 30 - dh + 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, dark, 2,
                cv2.LINE_AA)
    return img


@lru_cache(maxsize=2)
def shingle_texture(seed: int = 0, n: int = 256, colour=(0.30, 0.28, 0.30)) -> np.ndarray:
    """Asphalt roof shingles: staggered tab rows with dark gaps."""
    rng = np.random.default_rng(seed + 516)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    rows = 8
    yy = y / (n / rows)
    xx = x / (n / 6) + 0.5 * (np.floor(yy) % 2)
    fy = yy - np.floor(yy)
    fx = xx - np.floor(xx)
    tab = ((np.floor(xx) * 7 + np.floor(yy) * 3) % 4).astype(int)
    shade = np.array([0.9, 1.0, 1.08, 0.95], np.float32)[tab]
    g = fbm(n, n, 16, 16, rng, 3)
    base = np.asarray(colour, np.float32) * ((0.85 + 0.3 * g) * shade * (0.8 + 0.2 * fy))[..., None]
    gap = ((fx < 0.03) | (fy > 0.93)).astype(np.float32)
    return _to_u8(_mix(base, (0.08, 0.08, 0.09), gap * 0.8))


@lru_cache(maxsize=2)
def tread_texture(n: int = 64) -> np.ndarray:
    """Tyre tread (u around): black rubber with chevron grooves."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    chev = np.abs(((x * 2 + np.abs(y - 0.5)) % 0.5) - 0.25) < 0.07
    mid = (y > 0.2) & (y < 0.8)
    v = np.where(chev & mid, 0.04, 0.13)
    return _to_u8(np.stack([v, v, v * 1.05], -1))
