"""Procedural meshes / textures for the shopping carts job (visual only).

Generated in code (numpy + OpenCV), handed to MjSpec directly; nothing on disk. The
store name ("FRESH FLY MARKET") and every sign are our own; no real brand or logo.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.broccoli_toss_assets import merge, panel_mesh, panel_quat, sphere_mesh  # noqa: F401
from fly_simulator.jobs.kebab_assets import (  # noqa: F401  (re-exported helpers)
    TAU, MeshData, _mix, _to_u8, add_mesh, add_texture, add_textured_material, fbm, lathe, transform, tube,
)
from fly_simulator.jobs.mowing_assets import shrub_mesh  # noqa: F401
from fly_simulator.jobs.sisyphus_assets import box_mesh  # noqa: F401
from fly_simulator.jobs.trampoline_assets import disc_mesh, polyline_tube  # noqa: F401


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def _seg(a, b, r: float, n: int = 6) -> MeshData:
    return polyline_tube(np.array([a, b], float), r, n)


# ---------------------------------------------------------------------------
# the cart (cart frame: +x = the front of the basket, the handle at -x, z = 0 the ground)
# ---------------------------------------------------------------------------


def cart_meshes(L: float = 2.4, W: float = 1.36) -> dict[str, MeshData]:
    """A wire shopping cart: ``wire`` (the chrome basket, chassis, legs), ``plastic``
    (the red handle grip, the corner bumpers, the child-seat flap), ``wheels`` (four
    black casters). The basket tapers toward the front so carts nest."""
    r = 0.017
    wires: list[MeshData] = []
    xb0, xb1, zb = -0.80, 0.95, 0.72  # basket bottom
    xt0, xt1, zt = -1.02, 1.12, 1.52  # basket top rim
    yb, yt = 0.46 * W / 1.36, 0.66 * W / 1.36

    def pt(u, v, h):  # u 0..1 back->front, v -1..1 side, h 0..1 bottom->top
        x = (1 - h) * (xb0 + u * (xb1 - xb0)) + h * (xt0 + u * (xt1 - xt0))
        y = v * ((1 - h) * yb + h * yt)
        # the front taper: narrower at the front
        y *= 1.0 - 0.18 * u
        return np.array([x, y, (1 - h) * zb + h * zt])

    # rings round the basket at 4 heights
    for h in (0.0, 0.34, 0.68, 1.0):
        loop = [pt(0, -1, h), pt(1, -1, h), pt(1, 1, h), pt(0, 1, h), pt(0, -1, h)]
        wires.append(polyline_tube(np.array(loop), r * (1.5 if h == 1.0 else 1.0), 6))
    # vertical wires on the sides, front and back
    for u in np.linspace(0, 1, 9):
        for v in (-1, 1):
            wires.append(_seg(pt(u, v, 0), pt(u, v, 1), r))
    for v in np.linspace(-1, 1, 6)[1:-1]:
        for u in (0, 1):
            wires.append(_seg(pt(u, v, 0), pt(u, v, 1), r))
    # the floor grid
    for v in np.linspace(-1, 1, 6):
        wires.append(_seg(pt(0, v, 0), pt(1, v, 0), r))
    # chassis: the lower rack and the legs
    zc = 0.2
    for s in (-1, 1):
        rear = np.array([-0.95, s * 0.52, zc])
        front = np.array([0.95, s * 0.40, zc])
        wires.append(_seg(rear, front, r * 1.6))
        wires.append(_seg(rear, np.array([-1.08, s * 0.60, 1.72]), r * 1.8))  # rear leg up to the handle
        wires.append(_seg(front, pt(1, s, 0) + np.array([0, 0, -0.02]), r * 1.6))  # front leg
        wires.append(_seg(np.array([-0.9, s * 0.5, zc]), pt(0, s, 0), r * 1.3))
    for x in np.linspace(-0.85, 0.85, 6):  # the bottom rack
        wires.append(_seg(np.array([x, -0.45, zc + 0.01]), np.array([x, 0.45, zc + 0.01]), r))
    wire = merge(*wires)
    # plastic: the handle grip, corner bumpers, the seat flap
    grip = polyline_tube(np.array([[-1.14, -0.64, 1.74], [-1.14, 0.64, 1.74]]), 0.055, 10)
    bumpers = [transform(sphere_mesh(0.06, (0, 0, 0), 8, 6), np.eye(3), pt(1, s, 1)) for s in (-1, 1)]
    flap = transform(box_mesh((0.02, 0.5 * W * 0.8, 0.26), 1.0), _rot_y(-0.35),
                     np.array([-0.93, 0.0, 1.22]))
    plastic = merge(grip, *bumpers, flap)
    # casters: small wheels (discs turned upright) with forks
    wl = []
    for x in (-0.85, 0.85):
        for s in (-1, 1):
            y = s * (0.52 if x < 0 else 0.40)
            wl.append(transform(disc_mesh(0.12, 0.06, 16), _rot_x(math.pi / 2), np.array([x, y, 0.12])))
            wl.append(transform(box_mesh((0.03, 0.045, 0.05), 1.0), np.eye(3), np.array([x, y, zc - 0.02])))
    wheels = merge(*wl)
    return {"wire": wire, "plastic": plastic, "wheels": wheels}


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


# ---------------------------------------------------------------------------
# scenery
# ---------------------------------------------------------------------------


def _rounded_loop(hw: float, hh: float, z0: float, n: int = 16, e: float = 0.35) -> np.ndarray:
    ph = np.linspace(0.0, TAU, n, endpoint=False)
    c, s = np.cos(ph), np.sin(ph)
    return np.stack([np.sign(c) * np.abs(c) ** e * hw, np.sign(s) * np.abs(s) ** e * hh + z0 + hh], 1)


def car_meshes(L: float = 6.2, W: float = 2.6, H: float = 2.0) -> dict[str, MeshData]:
    """A stylised hatchback along x: ``body`` (painted), ``glass`` (the cabin),
    ``wheels`` (tyres), ``lights`` (head / tail lights), ``trim`` (bumpers)."""
    xs = np.linspace(-L / 2, L / 2, 11)
    rings = []
    for x in xs:
        u = abs(x) / (L / 2)
        top = 0.55 * H - 0.12 * H * u ** 2 * (1.4 if x > 0 else 1.0)
        loop = _rounded_loop(W / 2 * (1 - 0.06 * u ** 3), (top - 0.18 * H) / 2, 0.18 * H)
        rings.append(np.column_stack([np.full(len(loop), x), loop]))
    rings = np.array(rings)
    uv = np.zeros(rings.shape[:2] + (2,))
    body = tube(rings, uv)
    # the cabin (glass): a tapered rounded box
    cab = []
    for x, k in ((-L * 0.36, 0.86), (-L * 0.30, 1.0), (L * 0.12, 1.0), (L * 0.22, 0.82)):
        loop = _rounded_loop(W / 2 * 0.86 * (0.96 if k < 1 else 1.0), (H - 0.55 * H) / 2 * k, 0.52 * H, e=0.5)
        cab.append(np.column_stack([np.full(len(loop), x), loop]))
    glass = tube(np.array(cab), np.zeros((4, len(cab[0]), 2)))
    roof = transform(box_mesh((L * 0.2, W / 2 * 0.8, 0.04), 1.0), np.eye(3), np.array([-L * 0.08, 0, H * 0.99]))
    body = merge(body, roof)
    wl = []
    for x in (-L * 0.32, L * 0.32):
        for s in (-1, 1):
            wl.append(transform(disc_mesh(0.42, 0.34, 20), _rot_x(math.pi / 2), np.array([x, s * (W / 2 - 0.12), 0.42])))
    wheels = merge(*wl)
    lights = merge(*[transform(box_mesh((0.03, 0.28, 0.1), 1.0), np.eye(3),
                               np.array([sx * (L / 2 - 0.02), s * (W / 2 - 0.42), 0.36 * H]))
                     for sx in (-1, 1) for s in (-1, 1)])
    trim = merge(*[transform(box_mesh((0.08, W / 2 * 0.95, 0.09), 1.0), np.eye(3),
                             np.array([sx * (L / 2 - 0.02), 0, 0.2 * H])) for sx in (-1, 1)])
    return {"body": body, "glass": glass, "wheels": wheels, "lights": lights, "trim": trim}


def light_pole_meshes(h: float = 11.0, arm: float = 2.2) -> dict[str, MeshData]:
    """A lot light: concrete base, a steel pole with an arm, a lamp head (``lens``
    faces down)."""
    base = lathe(np.array([0.0, 0.45, 0.45, 0.32, 0.0]), np.array([0.0, 0.0, 0.5, 0.62, 0.62]), 16)
    pole = merge(polyline_tube(np.array([[0, 0, 0.5], [0, 0, h]]), 0.12, 10),
                 polyline_tube(np.array([[0, 0, h - 0.05], [arm * 0.5, 0, h + 0.2], [arm, 0, h + 0.25]]), 0.07, 8))
    head = transform(box_mesh((0.55, 0.28, 0.1), 1.0), np.eye(3), np.array([arm + 0.35, 0, h + 0.25]))
    lens = transform(box_mesh((0.48, 0.22, 0.02), 1.0), np.eye(3), np.array([arm + 0.35, 0, h + 0.14]))
    return {"base": base, "pole": pole, "head": head, "lens": lens}


def corral_mesh(x0: float, x1: float, hw: float, h: float = 1.75) -> MeshData:
    """The cart corral: a pipe frame along x (closed at x0, open at x1), rails at
    two heights on both sides, posts every ~2 mm (visual; the carts nest inside)."""
    r = 0.06
    parts = []
    xs = np.linspace(x0, x1, max(2, int(round((x1 - x0) / 2.0)) + 1))
    for s in (-1, 1):
        for z in (0.75, h):
            parts.append(polyline_tube(np.array([[x0, s * hw, z], [x1, s * hw, z]]), r, 8))
        for x in xs:
            parts.append(polyline_tube(np.array([[x, s * hw, 0.0], [x, s * hw, h]]), r * 1.1, 8))
        # the flared mouth
        parts.append(polyline_tube(np.array([[x1, s * hw, h], [x1 + 0.6, s * (hw + 0.5), h]]), r, 8))
        parts.append(polyline_tube(np.array([[x1 + 0.6, s * (hw + 0.5), 0.0], [x1 + 0.6, s * (hw + 0.5), h]]), r, 8))
    for z in (0.75, h):
        parts.append(polyline_tube(np.array([[x0, -hw, z], [x0, hw, z]]), r, 8))
    for x in xs:  # hoops over the top
        a = np.linspace(0, math.pi, 9)
        parts.append(polyline_tube(np.column_stack([np.full(9, x), hw * np.cos(a), h + 0.35 * np.sin(a)]), r * 0.9, 6))
    return merge(*parts)


def tree_meshes(h: float, seed: int = 0) -> dict[str, MeshData]:
    rng = np.random.default_rng(seed + 70)
    trunk = lathe(np.array([0.22, 0.17, 0.12, 0.0]), np.array([0.0, h * 0.4, h * 0.6, h * 0.6]), 10)
    blobs = []
    for _ in range(9):
        off = rng.normal(0, 0.28 * h, 3) * np.array([1, 1, 0.5])
        blobs.append(sphere_mesh(h * rng.uniform(0.2, 0.3), (off[0], off[1], h * 0.72 + off[2]), 12, 8))
    return {"trunk": trunk, "crown": merge(*blobs)}


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def asphalt_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Parking-lot asphalt: a mid-dark grey with soft patches, a little aggregate and
    oil stains (low fine detail: fine grit shimmers in a moving camera)."""
    rng = np.random.default_rng(seed + 401)
    f = fbm(n, n, 6, 6, rng, octaves=3)
    fine = fbm(n, n, 48, 48, rng, octaves=1)
    g = 0.27 + 0.025 * f + 0.02 * (fine - 0.5)
    img = np.stack([g, g, g * 1.04], -1)
    stain = fbm(n, n, 4, 4, rng, octaves=2) > 0.78
    img[stain] *= 0.9
    return _to_u8(img)


@lru_cache(maxsize=2)
def concrete_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed + 402)
    f = fbm(n, n, 6, 6, rng, octaves=3)
    img = np.repeat((0.68 + 0.08 * f)[..., None], 3, axis=2) * np.array([1.0, 0.99, 0.95], np.float32)
    img[:, :2] = 0.5
    img[:2, :] = 0.5
    return _to_u8(img)


@lru_cache(maxsize=2)
def storefront_texture(name: str = "FRESH FLY MARKET", h: int = 512, w: int = 1536) -> np.ndarray:
    """The store's front: a cream wall with a green band, the sign, big windows with
    posters (our own copy), two sliding doors with a canopy."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (236, 226, 204)
    band_y0, band_y1 = int(h * 0.06), int(h * 0.30)
    cv2.rectangle(img, (0, band_y0), (w, band_y1), (32, 118, 72), -1)
    cv2.rectangle(img, (0, band_y1), (w, band_y1 + 10), (250, 196, 40), -1)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 3.4
    (tw, th), _ = cv2.getTextSize(name, font, scale, 7)
    cx, cy = w // 2, (band_y0 + band_y1) // 2
    cv2.putText(img, name, (cx - tw // 2 + 4, cy + th // 2 + 4), font, scale, (10, 40, 20), 9, cv2.LINE_AA)
    cv2.putText(img, name, (cx - tw // 2, cy + th // 2), font, scale, (255, 252, 240), 7, cv2.LINE_AA)
    # a little leaf mark (our own)
    lx = cx - tw // 2 - 70
    cv2.ellipse(img, (lx, cy), (34, 18), -35, 0, 360, (140, 214, 90), -1, cv2.LINE_AA)
    cv2.line(img, (lx - 26, cy + 18), (lx + 22, cy - 14), (20, 80, 40), 3, cv2.LINE_AA)
    # windows
    wy0, wy1 = int(h * 0.42), int(h * 0.94)
    doors = [(int(w * 0.40), int(w * 0.47)), (int(w * 0.53), int(w * 0.60))]
    x = int(w * 0.03)
    posters = ["SALE", "FRESH FRUIT", "OPEN 24/7", "2 FOR 1", "BAKERY"]
    k = 0
    while x < w * 0.97:
        x1 = x + int(w * 0.11)
        if any(a - 10 < x1 and x < b + 10 for a, b in doors):
            x = max(b for a, b in doors if a - 10 < x1 and x < b + 10) + 16
            continue
        cv2.rectangle(img, (x, wy0), (x1, wy1), (58, 84, 104), -1)
        cv2.line(img, (x + 10, wy1 - 12), (x1 - 30, wy0 + 10), (120, 150, 170), 6)  # reflection
        cv2.rectangle(img, (x, wy0), (x1, wy1), (90, 90, 88), 6)
        if k % 2 == 0:
            txt = posters[(k // 2) % len(posters)]
            px0, py0 = x + 14, wy0 + 30
            cv2.rectangle(img, (px0, py0), (x1 - 14, py0 + 70), (250, 240, 60) if k % 4 else (230, 60, 50), -1)
            (pw, ph), _ = cv2.getTextSize(txt, font, 0.8, 2)
            cv2.putText(img, txt, ((px0 + x1 - 14) // 2 - pw // 2, py0 + 35 + ph // 2), font, 0.8,
                        (30, 30, 30), 2, cv2.LINE_AA)
        x = x1 + 16
        k += 1
    for a, b in doors:
        cv2.rectangle(img, (a, int(h * 0.38)), (b, h), (70, 96, 116), -1)
        cv2.line(img, ((a + b) // 2, int(h * 0.38)), ((a + b) // 2, h), (160, 160, 160), 5)
        cv2.rectangle(img, (a, int(h * 0.38)), (b, h), (150, 150, 150), 6)
    cv2.putText(img, "ENTRANCE", (doors[0][0] + 10, int(h * 0.36)), font, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.putText(img, "EXIT", (doors[1][0] + 40, int(h * 0.36)), font, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    return img


@lru_cache(maxsize=2)
def sign_texture(text: str, sub: str, bg=(0.10, 0.30, 0.65), h: int = 192, w: int = 384) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = [int(255 * c) for c in bg]
    cv2.rectangle(img, (6, 6), (w - 7, h - 7), (255, 255, 255), 5)
    font = cv2.FONT_HERSHEY_DUPLEX
    for t, sc, y in ((text, 1.5, int(h * 0.45)), (sub, 0.8, int(h * 0.78))):
        (tw, th), _ = cv2.getTextSize(t, font, sc, 3)
        cv2.putText(img, t, ((w - tw) // 2, y + th // 2), font, sc, (255, 255, 255), 3, cv2.LINE_AA)
    return img


@lru_cache(maxsize=2)
def sky_texture(seed: int = 0, h: int = 512, w: int = 1024) -> np.ndarray:
    """A late-afternoon sky with clouds over a suburban treeline."""
    rng = np.random.default_rng(seed + 403)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    sky = _mix(np.array([0.36, 0.58, 0.90], np.float32), np.array([0.93, 0.88, 0.80], np.float32), t)
    img = np.broadcast_to(sky, (h, w, 3)).copy()
    cloud = fbm(h, w, 3, 8, rng, octaves=4)
    img = _mix(img, (1.0, 0.98, 0.95), np.clip((cloud - 0.55) * 2.2, 0, 0.8) * (t < 0.6))
    img = _to_u8(img)
    base = int(h * 0.80)
    for _ in range(70):
        x = int(rng.integers(0, w))
        r = int(rng.integers(18, 42))
        cv2.circle(img, (x, base - int(rng.integers(0, 30))), r, (int(rng.integers(40, 70)), int(rng.integers(90, 120)),
                                                                   int(rng.integers(45, 70))), -1, cv2.LINE_AA)
    img[base:] = (70, 110, 70)
    return img
