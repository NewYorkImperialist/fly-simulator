"""Procedural meshes / textures for the snow shovel job (visual only).

Generated in code (numpy + OpenCV), handed to MjSpec directly; nothing on disk.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cv2
import numpy as np

from fly_simulator.jobs.broccoli_toss_assets import merge, panel_mesh, panel_quat, sphere_mesh, torus_mesh  # noqa: F401
from fly_simulator.jobs.kebab_assets import (  # noqa: F401  (re-exported helpers)
    TAU, MeshData, _mix, _to_u8, add_mesh, add_texture, add_textured_material, fbm, lathe,
    tile_noise, transform, tube,
)
from fly_simulator.jobs.mowing_assets import (  # noqa: F401
    concrete_texture, facade_texture, gable_roof_mesh, mailbox_meshes, picket_fence_mesh, shingle_texture,
)
from fly_simulator.jobs.sisyphus_assets import boulder_mesh, box_mesh  # noqa: F401
from fly_simulator.jobs.trampoline_assets import disc_mesh, polyline_tube  # noqa: F401


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------


def blade_mesh(half_w: float, h: float, curve: float = 0.25, t: float = 0.05, n: int = 10) -> MeshData:
    """A curved shovel blade: a scoop across y (half width ``half_w``), rising from
    the scraping edge at z = 0 (x = +curve) curving back to x = 0 at the top."""
    s = np.linspace(0, 1, n)
    xs = curve * (1 - s) ** 2
    zs = h * s
    rings = []
    for x, z in zip(xs, zs):
        rings.append(np.array([[x, -half_w, z], [x, half_w, z], [x - t, half_w, z], [x - t, -half_w, z]]))
    rings = np.asarray(rings)
    uv = np.zeros(rings.shape[:2] + (2,))
    uv[..., 0] = np.array([0, 1, 1, 0])[None]
    uv[..., 1] = s[:, None]
    md = tube(rings, uv)
    if md.signed_volume() < 0:
        md.faces = md.faces[:, ::-1].copy()
    return md


def shovel_handle(foot, grip, r: float = 0.06) -> dict[str, MeshData]:
    """The shaft from the blade back up to the D-grip and the grip itself."""
    foot, grip = np.asarray(foot, float), np.asarray(grip, float)
    pts = np.linspace(foot, grip, 8)
    shaft = polyline_tube(pts, r, 10)
    d = merge(transform(torus_mesh(0.28, 0.045, 24, 8), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
                        grip + np.array([-0.2, 0.0, 0.1])))
    return {"shaft": shaft, "grip": d}


def cone_mesh(r: float, h: float, z0: float = 0.0, n: int = 20) -> MeshData:
    return lathe(np.array([0.0, r, 0.0]), np.array([z0, z0, z0 + h]), n)


def pine_meshes(h: float, seed: int = 0) -> dict[str, MeshData]:
    """A snowy pine: a trunk, three stacked green cones and white snow caps."""
    rng = np.random.default_rng(seed + 81)
    trunk = lathe(np.array([0.0, 0.07 * h, 0.05 * h, 0.0]), np.array([0.0, 0.0, 0.3 * h, 0.3 * h]), 10)
    green, snow = [], []
    for k in range(3):
        z0 = h * (0.18 + 0.25 * k)
        r = h * (0.36 - 0.09 * k) * rng.uniform(0.92, 1.08)
        hh = h * 0.42
        green.append(cone_mesh(r, hh, z0, 18))
        snow.append(cone_mesh(r * 0.55, hh * 0.55, z0 + hh * 0.47, 18))
    return {"trunk": trunk, "green": merge(*green), "snow": merge(*snow)}


def snowman_meshes(h: float) -> dict[str, MeshData]:
    """A snowman (base at z = 0, facing -y): body balls, coal eyes and buttons, a
    carrot nose, stick arms and a top hat."""
    r = np.array([0.36, 0.26, 0.19]) * h
    z = np.array([r[0] * 0.9, 2 * r[0] * 0.9 + r[1] * 0.8, 2 * r[0] * 0.9 + 2 * r[1] * 0.8 + r[2] * 0.85])
    body = merge(*[sphere_mesh(r[i], (0, 0, z[i]), 20, 12) for i in range(3)])
    coal = [sphere_mesh(0.025 * h, (s * 0.07 * h, -r[2] * 0.92, z[2] + 0.05 * h), 8, 6) for s in (-1, 1)]
    coal += [sphere_mesh(0.025 * h, (0, -r[1] * 0.97, z[1] + dz * h), 8, 6) for dz in (-0.08, 0.0, 0.08)]
    nose = transform(cone_mesh(0.035 * h, 0.16 * h, 0.0, 10), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
                     np.array([0.0, -r[2] * 0.95, z[2]]))
    arms = merge(*[polyline_tube(np.array([[s * r[1] * 0.8, 0, z[1] + 0.05 * h],
                                           [s * r[1] * 1.6, 0, z[1] + 0.22 * h],
                                           [s * r[1] * 2.0, 0, z[1] + 0.35 * h]]), 0.012 * h, 6)
                   for s in (-1, 1)])
    zt = z[2] + r[2] * 0.8
    hat = merge(lathe(np.array([0.0, 0.2 * h, 0.2 * h, 0.0]), np.array([zt, zt, zt + 0.02 * h, zt + 0.02 * h]), 20),
                lathe(np.array([0.0, 0.12 * h, 0.12 * h, 0.0]), np.array([zt, zt, zt + 0.2 * h, zt + 0.2 * h]), 20))
    return {"body": body, "coal": merge(*coal, hat), "nose": nose, "arms": arms}


def lamp_meshes() -> dict[str, MeshData]:
    """A porch wall lantern (on a wall facing -y): a bracket, a glass box, a cap."""
    bracket = transform(box_mesh((0.06, 0.25, 0.05)), np.eye(3), np.array([0, 0.25, 0.3]))
    glass = transform(box_mesh((0.16, 0.16, 0.24)), np.eye(3), np.array([0, 0, 0]))
    cap = lathe(np.array([0.0, 0.24, 0.0]), np.array([0.24, 0.24, 0.42]), 4)
    return {"bracket": bracket, "glass": glass, "cap": cap}


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def snow_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """Snow cover: soft blue-white undulations, drift streaks and sparkles."""
    rng = np.random.default_rng(seed + 13)
    f = fbm(n, n, 6, 6, rng, octaves=5)
    img = _mix(np.array([0.78, 0.84, 0.93], np.float32), np.array([0.97, 0.98, 1.0], np.float32), f)
    streak = cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), sigmaX=9, sigmaY=1.5)
    img = img * (0.95 + 0.10 * (streak - streak.mean()) / (streak.std() + 1e-6) * 0.3)[..., None]
    img = _to_u8(img)
    for _ in range(900):
        x, y = rng.integers(0, n, 2)
        img[y, x] = (255, 255, 255)
    return img


@lru_cache(maxsize=2)
def plowed_snow_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Shovelled / plowed snow: packed, blue-grey shadows, grit and gravel specks."""
    rng = np.random.default_rng(seed + 29)
    f = fbm(n, n, 8, 8, rng, octaves=5)
    img = _mix(np.array([0.62, 0.68, 0.80], np.float32), np.array([0.93, 0.95, 0.99], np.float32), f)
    img = _to_u8(img)
    for _ in range(700):
        x, y = rng.integers(0, n, 2)
        g = int(rng.integers(60, 130))
        cv2.circle(img, (int(x), int(y)), int(rng.integers(0, 2)), (g, g - 8, g - 16), -1)
    return img


@lru_cache(maxsize=2)
def asphalt_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed + 17)
    f = fbm(n, n, 16, 16, rng, octaves=3)
    fine = rng.random((n, n)).astype(np.float32)
    g = 0.16 + 0.05 * f + 0.06 * (fine - 0.5)
    img = np.stack([g, g, g * 1.08], -1)
    slush = fbm(n, n, 5, 5, rng, octaves=3) > 0.62
    img[slush] = img[slush] * 0.4 + np.array([0.62, 0.64, 0.68]) * 0.6
    return _to_u8(img)


@lru_cache(maxsize=2)
def snowy_roof_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    base = shingle_texture(seed, n, (0.30, 0.22, 0.20)).astype(np.float32) / 255
    rng = np.random.default_rng(seed + 19)
    m = fbm(n, n, 5, 5, rng, octaves=4)
    w = np.clip((m - 0.35) * 3.0, 0, 1)[..., None]
    img = base * (1 - w) + np.array([0.93, 0.95, 0.99], np.float32) * w
    return _to_u8(img)


def sign_texture(h: int = 128, w: int = 384) -> np.ndarray:
    img = np.full((h, w, 3), 245, np.uint8)
    cv2.rectangle(img, (4, 4), (w - 5, h - 5), (30, 60, 140), 5)
    for txt, y, s in (("THE FLY RESIDENCE", 0.42, 1.0), ("PLEASE SHOVEL YOUR WALK", 0.78, 0.65)):
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, s, 2)
        cv2.putText(img, txt, ((w - tw) // 2, int(h * y) + th // 2), cv2.FONT_HERSHEY_DUPLEX, s, (30, 60, 140), 2,
                    cv2.LINE_AA)
    return img


@lru_cache(maxsize=2)
def sky_texture(seed: int = 0, h: int = 512, w: int = 1024) -> np.ndarray:
    """An overcast winter sky with snowy hills and a pine treeline."""
    rng = np.random.default_rng(seed + 23)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    sky = _mix(np.array([0.55, 0.62, 0.74], np.float32), np.array([0.86, 0.89, 0.94], np.float32), t)
    img = np.broadcast_to(sky, (h, w, 3)).copy()
    cloud = fbm(h, w, 4, 8, rng, octaves=4)
    img = img * (0.94 + 0.12 * cloud[..., None])
    img = _to_u8(img)
    base = int(h * 0.72)
    for _ in range(40):
        x = int(rng.integers(0, w))
        th = int(rng.integers(50, 110))
        cv2.fillPoly(img, [np.array([[x - 22, base], [x + 22, base], [x, base - th]])], (60, 86, 78))
        cv2.fillPoly(img, [np.array([[x - 8, base - th + 30], [x + 8, base - th + 30], [x, base - th]])],
                     (235, 240, 248))
    xs = np.arange(w)
    hill = (base + 12 * np.sin(xs / 90.0) + 8 * np.sin(xs / 37.0)).astype(int)
    for x in range(w):
        img[hill[x]:, x] = (232, 237, 245)
    return img
