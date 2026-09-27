"""Procedural meshes and textures for the ``temple_standoff`` scene (visual only).

An original fly-scale stone hall (slab floor, ashlar walls, columns, tall windows,
a dais), the energy blade's handle, and the flies' attire: a heavy hooded cloak for
the tall fly and small tunics / sashes for the little flies. Everything is generated
in code (numpy + OpenCV) and handed to MjSpec directly; nothing is written to disk.

The garments are **shrink-wrapped** onto the fly body: rays are cast inward at the
thorax / abdomen / wing (or head) geoms of a probe NeuroMechFly (``mj_ray``), the hit
points are pushed outward by an offset (plus folds / flare), and the resulting
surface grid is turned into a thin closed shell (``shell_mesh``) in the body frame
of the segment it rides on. Nothing here collides or has mass.
"""

from __future__ import annotations

import math
from functools import lru_cache

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


def box_mesh(hx: float, hy: float, hz: float, tile: float = 4.0) -> MeshData:
    """Axis-aligned box (half sizes) with per-face UVs in world units / ``tile``, so a
    texture tiles at the same density on every face and every box."""
    verts, uvs, faces = [], [], []
    n = 0
    for axis in range(3):
        for sgn in (-1.0, 1.0):
            a, b = [k for k in range(3) if k != axis]
            h = np.array([hx, hy, hz])
            corners = []
            for ca, cb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[axis] = sgn * h[axis]
                p[a] = ca * h[a]
                p[b] = cb * h[b]
                corners.append(p)
            corners = np.array(corners)
            # vertical faces: v from z so the texture stands upright
            if axis == 2:
                uv = np.stack([corners[:, 0], corners[:, 1]], 1) / tile
            else:
                other = 1 if axis == 0 else 0
                uv = np.stack([corners[:, other], -corners[:, 2]], 1) / tile
            f = np.array([[0, 1, 2], [0, 2, 3]]) + n
            nrm = np.zeros(3)
            nrm[axis] = sgn
            e = np.cross(corners[1] - corners[0], corners[2] - corners[0])
            if np.dot(e, nrm) < 0:
                f = f[:, ::-1]
            verts.append(corners)
            uvs.append(uv)
            faces.append(f)
            n += 4
    return MeshData(np.concatenate(verts), np.concatenate(faces), np.concatenate(uvs))


def column_mesh(r: float, h: float, n_theta: int = 40) -> MeshData:
    """A stone column (surface of revolution, base at z = 0): square-ish plinth
    ring, torus mouldings, a shaft with a slight entasis and shallow flutes, a
    capital that flares to an abacus."""
    zs = [0.0, 0.0, 0.10, 0.10, 0.22, 0.30, 0.38]
    rs = [0.0, 1.55, 1.55, 1.40, 1.36, 1.22, 1.08]
    shaft = np.linspace(0.40, h - 1.0, 24)
    f = (shaft - 0.4) / (h - 1.4)
    zs += list(shaft)
    rs += list(1.0 + 0.04 * np.sin(math.pi * f) - 0.08 * f)
    zs += [h - 0.9, h - 0.75, h - 0.55, h - 0.35, h - 0.35, h - 0.2, h - 0.2, h, h]
    rs += [0.98, 1.10, 1.22, 1.40, 1.55, 1.55, 1.60, 1.60, 0.0]
    zs = np.array(zs)
    rs = np.array(rs) * r
    shaft_lo, shaft_hi = 0.40, h - 1.0

    def flutes(th, z):
        on = ((z > shaft_lo) & (z < shaft_hi)).astype(float)
        return -0.035 * r * on * np.clip(np.cos(16 * th), 0, 1) ** 2

    return lathe(rs, zs, n_theta, radial_noise=flutes, v_coord=zs / h * 3.0, u_repeat=2.0)


def cylinder_mesh(r: float, h: float, n: int = 24, u_repeat: float = 1.0) -> MeshData:
    return lathe(np.array([0.0, r, r, 0.0]), np.array([0.0, 0.0, h, h]), n,
                 v_coord=np.array([0.0, 0.0, 1.0, 1.0]), u_repeat=u_repeat)


def shell_mesh(P: np.ndarray, thickness: float | np.ndarray, uv: np.ndarray | None = None) -> MeshData:
    """A thin closed shell from an (R, C, 3) surface grid: the grid is the outer skin,
    an inner skin sits ``thickness`` inward (along the grid normals), and the four
    edges are stitched. UV defaults to (column, row) in [0, 1]."""
    R, C, _ = P.shape
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    nrm = np.cross(du, dv)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=2, keepdims=True), 1e-12)
    th = np.broadcast_to(np.asarray(thickness, float), (R, C))[..., None]
    Q = P - nrm * th
    if uv is None:
        uu, vv = np.meshgrid(np.linspace(0, 1, C), np.linspace(0, 1, R))
        uv = np.stack([uu, vv], -1)
    verts = np.concatenate([P.reshape(-1, 3), Q.reshape(-1, 3)])
    uvs = np.concatenate([uv.reshape(-1, 2), uv.reshape(-1, 2)])
    idx = np.arange(R * C).reshape(R, C)
    off = R * C
    faces = []
    for r in range(R - 1):
        for c in range(C - 1):
            a, b, cc, d = idx[r, c], idx[r, c + 1], idx[r + 1, c + 1], idx[r + 1, c]
            faces += [[a, b, cc], [a, cc, d], [a + off, cc + off, b + off], [a + off, d + off, cc + off]]
    # the rim: walk the boundary loop and stitch outer to inner
    loop = list(idx[0, :]) + list(idx[1:, -1]) + list(idx[-1, -2::-1]) + list(idx[-2:0:-1, 0])
    for i in range(len(loop)):
        a, b = loop[i], loop[(i + 1) % len(loop)]
        faces += [[a, b + off, b], [a, a + off, b + off]]
    md = MeshData(verts, np.array(faces, dtype=np.int64), uvs)
    # consistent winding: flip everything if the volume came out negative
    if md.signed_volume() < 0:
        md.faces = md.faces[:, ::-1].copy()
    return md


def shaft_mesh(p0, p1, w0: float, w1: float, depth: float) -> MeshData:
    """A light shaft volume: a box-like frustum from ``p0`` (the window, w0 wide)
    to ``p1`` (the floor patch, w1 wide), ``depth`` thick (horizontal, across)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    ax /= np.linalg.norm(ax)
    side = np.cross(ax, (0.0, 0.0, 1.0))
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    side /= np.linalg.norm(side)
    up = np.cross(side, ax)
    rings = []
    for p, w in ((p0, w0), (p1, w1)):
        ring = [p + sx * w / 2 * up + sy * depth / 2 * side for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        rings.append(ring)
    rings = np.array(rings)
    uv = np.zeros(rings.shape[:2] + (2,))
    uv[1, :, 1] = 1.0
    return tube(rings, uv)


# ---------------------------------------------------------------------------
# garments (shrink-wrapped on a probe fly)
# ---------------------------------------------------------------------------


class BodyProbe:
    """Ray-casts at a subset of a compiled single-fly model's geoms (the probe fly
    stands at the origin, unposed) and reports hits in a segment's body frame."""

    def __init__(self, model: mj.MjModel, data: mj.MjData, prefix: str = ""):
        self.m, self.d, self.pre = model, data, prefix
        mj.mj_forward(model, data)
        self._group0 = model.geom_group.copy()

    def _only(self, bodies: tuple[str, ...]) -> None:
        m = self.m
        keep = {m.body(self.pre + b).id for b in bodies}
        for g in range(m.ngeom):
            m.geom_group[g] = 0 if int(m.geom_bodyid[g]) in keep else 5

    def restore(self) -> None:
        self.m.geom_group[:] = self._group0

    def cast(self, bodies: tuple[str, ...], origins: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Distance along (target - origin) to the first hit, NaN for a miss.
        ``origins`` / ``targets``: (..., 3) world points."""
        self._only(bodies)
        grp = np.array([1, 0, 0, 0, 0, 0], np.uint8)
        gid = np.zeros(1, np.int32)
        o = origins.reshape(-1, 3)
        t = targets.reshape(-1, 3)
        out = np.full(len(o), np.nan)
        for i in range(len(o)):
            v = t[i] - o[i]
            L = float(np.linalg.norm(v))
            dist = mj.mj_ray(self.m, self.d, o[i], v / L, grp, 1, -1, gid)
            if dist >= 0 and dist < L * 1.02:
                out[i] = dist
        self.restore()
        return out.reshape(origins.shape[:-1])

    def to_body(self, body: str, pts: np.ndarray) -> np.ndarray:
        b = self.m.body(self.pre + body).id
        R = self.d.xmat[b].reshape(3, 3)
        return (pts - self.d.xpos[b]) @ R

    def pos(self, body: str) -> np.ndarray:
        return self.d.xpos[self.m.body(self.pre + body).id].copy()

    def vertices(self, body: str) -> np.ndarray:
        """World vertices of a body's mesh geoms."""
        m, d = self.m, self.d
        b = m.body(self.pre + body).id
        out = []
        for g in np.flatnonzero(m.geom_bodyid == b):
            mid = int(m.geom_dataid[g])
            if mid < 0:
                continue
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            out.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
        return np.concatenate(out)


def _smooth_grid(a: np.ndarray, k: int = 2) -> np.ndarray:
    """Box-blur an (R, C) array a few times (edge-clamped)."""
    out = a.copy()
    for _ in range(k):
        p = np.pad(out, 1, mode="edge")
        out = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] + 2 * p[1:-1, 1:-1]) / 6.0
    return out


def _wrap_around_axis(probe: BodyProbe, bodies, xs, zc, th, far: float = 3.0,
                      fallback: float = 0.2) -> np.ndarray:
    """Radius (about the line (x, 0, zc(x))) of the body surface on the grid
    xs (R) x th (C) (th = 0 is the top, +th toward +y)."""
    X, T = np.meshgrid(xs, th, indexing="ij")
    Z = np.interp(X, xs, zc)
    ctr = np.stack([X, np.zeros_like(X), Z], -1)
    dirn = np.stack([np.zeros_like(X), np.sin(T), np.cos(T)], -1)
    dist = probe.cast(bodies, ctr + far * dirn, ctr)
    r = far - dist
    # misses (beyond the body): the nearest hit radius in the same column fading out
    r = np.where(np.isnan(r), -1.0, r)
    for c in range(r.shape[1]):
        col = r[:, c]
        good = col > 0
        if good.any():
            idx = np.flatnonzero(good)
            col[~good] = np.interp(np.flatnonzero(~good), idx, col[idx]) if len(idx) > 1 else col[idx[0]]
            col[:] = np.maximum(col, fallback)
        else:
            col[:] = fallback
    return r


def cloak_meshes(probe: BodyProbe, seed: int = 0) -> dict[str, tuple[str, MeshData]]:
    """The tall fly's cloak, in unit-fly coordinates: ``{name: (body, mesh)}``.

    * ``cape``: the long outer layer over the thorax, the wings and the abdomen,
      to just past the abdomen tip, hanging down the sides to the body's equator
      (the coxae and legs stay clear), with deep lengthwise folds that grow toward
      the back and the hem;
    * ``mantle``: a shorter second layer over the shoulders (a layered look);
    * ``collar``: a raised, thick ring around the neck;
    * ``hood``: a deep cowl on the head, open at the face (the eyes stay visible),
      set back behind the eyes, bulging toward the back.
    """
    rng = np.random.default_rng(seed)
    th_c = probe.pos("c_thorax")
    head = probe.pos("c_head")
    tail = probe.vertices("c_abdomen6")
    x_tail = float(tail[:, 0].min())
    thv = probe.vertices("c_thorax")
    x_front = float(thv[:, 0].max())
    z_th = float(thv[:, 2].mean())
    z_tail = float(tail[:, 2].mean())
    bodies = ("c_thorax", "c_abdomen12", "c_abdomen3", "c_abdomen4", "c_abdomen5", "c_abdomen6",
              "l_wing", "r_wing", "l_haltere", "r_haltere")
    out = {}

    # ---- the cape
    R, C = 40, 41
    x0, x1 = x_front - 0.12, x_tail - 0.85
    xs = np.linspace(x0, x1, R)
    zc = np.interp(xs, [x1, x_tail, th_c[0] - 0.4, x0], [z_tail, z_tail, z_th, z_th])
    s = (xs - x0) / (x1 - x0)  # 0 at the neck, 1 at the end of the train
    st = np.clip((x_tail + 0.05 - xs) / 0.9, 0.0, 1.0)  # 0 over the body, 1 at the train's end
    th_max = np.radians(96.0 + 16.0 * np.clip(s / 0.7, 0, 1))  # hangs a bit lower toward the back
    uu = np.linspace(-1.0, 1.0, C)
    T = uu[None, :] * th_max[:, None]
    r = np.zeros((R, C))
    for i in range(R):
        r[i] = _wrap_around_axis(probe, bodies, xs[i:i + 1], zc[i:i + 1], T[i], fallback=0.15)[0]
    # an envelope (a heavy fabric bridges the dips between segments and the wings) ...
    r = np.maximum(_smooth_grid(r, 3), r)
    # ... and, behind the widest part of the abdomen, it does not follow the body in:
    # it falls from there (a heavy cloth)
    x_wide = th_c[0] - 1.0
    for i in range(1, R):
        if xs[i] < x_wide:
            r[i] = np.maximum(r[i], 0.975 * r[i - 1])
    ph = rng.uniform(0, TAU, 3)
    folds = (0.05 * s[:, None] ** 0.8 + 0.015) * (0.6 + 0.4 * np.abs(uu[None])) * (
        np.sin(9.0 * uu[None] * math.pi / 2 + ph[0]) * 0.7 + 0.3 * np.sin(17.0 * uu[None] + ph[1]))
    flare = 0.10 * s[:, None] ** 2 + 0.06 * np.abs(uu[None]) ** 3
    rr = r + 0.05 + folds + flare
    X = np.repeat(xs[:, None], C, 1)
    Z = np.repeat(zc[:, None], C, 1)
    Y = rr * np.sin(T) * (1.0 - 0.28 * st[:, None] ** 1.2)  # the train narrows (clear of the hind legs)
    Zs = Z + rr * np.cos(T)
    # the train: past the abdomen tip the cloth falls toward the floor
    droop = (Zs - 0.06) * (st[:, None] ** 1.5) * 0.92
    P = np.stack([X, Y, np.maximum(Zs - droop, 0.05)], -1)
    # the side hem: turned slightly outward (the fabric's weight)
    hem = np.abs(uu) > 0.85
    P[:, hem, 1] *= 1.0 + 0.10 * ((np.abs(uu[hem]) - 0.85) / 0.15)
    out["cape"] = ("c_thorax", shell_mesh(probe.to_body("c_thorax", P), 0.035))

    # ---- the mantle (a shorter layer over the shoulders)
    R2, C2 = 14, 37
    xs2 = np.linspace(x_front - 0.02, th_c[0] - 0.62, R2)
    s2 = (xs2 - xs2[0]) / (xs2[-1] - xs2[0])
    zc2 = np.full(R2, z_th)
    th2 = np.radians(84.0 + 16.0 * np.sin(np.pi * (0.2 + 0.8 * s2)) ** 0.6)
    uu2 = np.linspace(-1, 1, C2)
    T2 = uu2[None] * th2[:, None]
    r2 = np.zeros((R2, C2))
    for i in range(R2):
        r2[i] = _wrap_around_axis(probe, bodies, xs2[i:i + 1], zc2[i:i + 1], T2[i], fallback=0.2)[0]
    r2 = np.maximum(_smooth_grid(r2, 3), r2)
    wav = 0.02 * np.sin(11.0 * uu2[None] + ph[2]) * (0.3 + s2[:, None])
    rr2 = r2 + 0.11 + wav + 0.05 * s2[:, None] ** 2 + 0.10 * np.abs(uu2[None]) ** 3
    X2 = np.repeat(xs2[:, None], C2, 1)
    P2 = np.stack([X2, rr2 * np.sin(T2), zc2[:, None] + rr2 * np.cos(T2)], -1)
    out["mantle"] = ("c_thorax", shell_mesh(probe.to_body("c_thorax", P2), 0.03))

    # ---- the collar: a raised thick ring around the neck (behind the head)
    R3, C3 = 6, 33
    xs3 = np.linspace(x_front + 0.02, x_front - 0.22, R3)
    s3 = np.linspace(0, 1, R3)
    uu3 = np.linspace(-1, 1, C3)
    T3 = uu3[None] * np.radians(118.0)
    r3 = np.zeros((R3, C3))
    for i in range(R3):
        r3[i] = _wrap_around_axis(probe, bodies, xs3[i:i + 1], np.full(1, z_th), np.broadcast_to(T3[0], (C3,)),
                                  fallback=0.3)[0]
    r3 = np.maximum(_smooth_grid(r3, 3), r3)
    rr3 = r3 + 0.16 + 0.07 * (1 - s3[:, None]) + 0.02 * np.cos(uu3[None] * 6.0)
    X3 = np.repeat(xs3[:, None], C3, 1)
    Tb = np.broadcast_to(T3, (R3, C3))
    P3 = np.stack([X3, rr3 * np.sin(Tb), z_th + rr3 * np.cos(Tb) + 0.06 * (1 - s3[:, None])], -1)
    out["collar"] = ("c_thorax", shell_mesh(probe.to_body("c_thorax", P3), 0.05))

    # ---- the hood (on the head): a deep cowl behind the eyes
    hv = np.concatenate([probe.vertices("c_head"), probe.vertices("l_eye"), probe.vertices("r_eye")])
    hc = hv.mean(0)
    hc[0] = head[0] + 0.5 * (hc[0] - head[0])
    R4, C4 = 18, 37
    al = np.radians(np.linspace(-128.0, 128.0, C4))
    # the front rim: forward over the crown (a brow that frames the face from above),
    # back behind the eyes on the sides (the eyes stay clear)
    rim_x = 0.02 + 0.50 * np.clip(np.cos(al), 0, 1) ** 2
    v = np.linspace(0.0, 1.0, R4)[:, None]  # 0 at the rim, 1 at the back
    XD = rim_x[None] + (-0.97 - rim_x[None]) * v
    AL = np.broadcast_to(al[None], XD.shape)
    rho = np.sqrt(1 - XD ** 2)
    dirn = np.stack([XD, rho * np.sin(AL), rho * np.cos(AL)], -1)
    far = 2.0
    hbodies = ("c_head", "l_eye", "r_eye", "c_rostrum", "l_pedicel", "r_pedicel")
    dist = probe.cast(hbodies, hc + far * dirn, np.broadcast_to(hc, dirn.shape).copy())
    rh = far - dist
    rh = np.where(np.isnan(rh), np.nanmean(rh), rh)
    rh = np.maximum(_smooth_grid(rh, 3), rh)
    deep = 0.07 + 0.20 * v ** 1.3 + 0.05 * np.clip(np.cos(AL), 0, 1) * v  # bulges back and up
    rim = 0.03 * np.exp(-((v - 0.0) / 0.08) ** 2)  # a rolled front edge
    rr4 = rh + deep + rim
    P4 = hc + rr4[..., None] * dirn
    # the lower edge hangs straight down onto the collar (no chin strap)
    low = np.cos(AL) < -0.35
    P4[low, 2] -= 0.08
    out["hood"] = ("c_head", shell_mesh(probe.to_body("c_head", P4), 0.04))
    return out


def tunic_meshes(probe: BodyProbe, seed: int = 0) -> dict[str, tuple[str, MeshData]]:
    """A little fly's tunic (a slightly oversized mantle over the thorax, stopping in
    front of the wing hinges so the wings stay free) and a sash (a diagonal band over
    it). Unit-fly coordinates, on ``c_thorax``."""
    rng = np.random.default_rng(seed)
    thv = probe.vertices("c_thorax")
    x_front = float(thv[:, 0].max())
    z_th = float(thv[:, 2].mean())
    wing = probe.pos("l_wing")
    bodies = ("c_thorax",)
    out = {}
    R, C = 16, 33
    xs = np.linspace(x_front - 0.06, wing[0] + 0.06, R)
    s = np.linspace(0, 1, R)
    th = np.radians(66.0 + 30.0 * np.sin(np.pi * (0.12 + 0.8 * s)) ** 0.7)  # a rounded hem
    uu = np.linspace(-1, 1, C)
    T = uu[None] * th[:, None]
    r = np.zeros((R, C))
    for i in range(R):
        r[i] = _wrap_around_axis(probe, bodies, xs[i:i + 1], np.full(1, z_th), T[i], fallback=0.25)[0]
    r = np.maximum(_smooth_grid(r, 3), r)
    ph = rng.uniform(0, TAU)
    rr = r + 0.06 + 0.04 * np.abs(uu[None]) ** 2 + 0.012 * np.sin(8.0 * uu[None] + ph)
    X = np.repeat(xs[:, None], C, 1)
    P = np.stack([X, rr * np.sin(T), z_th + rr * np.cos(T)], -1)
    out["tunic"] = ("c_thorax", shell_mesh(probe.to_body("c_thorax", P), 0.03))
    # the sash: a band crossing diagonally (left shoulder, front -> right side, back)
    R2, C2 = 16, 5
    s2 = np.linspace(0, 1, R2)
    x2 = xs[0] + (xs[-1] - xs[0]) * (0.1 + 0.8 * s2)
    tc = np.radians(-70.0 + 140.0 * s2)
    wdt = np.radians(15.0)
    T2 = tc[:, None] + wdt * np.linspace(-1, 1, C2)[None]
    r2 = np.zeros((R2, C2))
    for i in range(R2):
        r2[i] = _wrap_around_axis(probe, bodies, x2[i:i + 1], np.full(1, z_th), T2[i], fallback=0.25)[0]
    r2 = np.maximum(_smooth_grid(r2, 2), r2)
    rr2 = r2 + 0.06 + 0.04 * (np.abs(T2) / th.max()) ** 2 + 0.035
    X2 = np.repeat(x2[:, None], C2, 1)
    P2 = np.stack([X2, rr2 * np.sin(T2), z_th + rr2 * np.cos(T2)], -1)
    out["sash"] = ("c_thorax", shell_mesh(probe.to_body("c_thorax", P2), 0.02))
    return out


# ---------------------------------------------------------------------------
# textures
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def stone_floor_texture(seed: int = 0, n: int = 512, slabs: int = 4) -> np.ndarray:
    """Large polished stone slabs, cool grey, with thin dark joints, per-slab tone
    and soft veins."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n * slabs
    fx, fy = x - np.floor(x), y - np.floor(y)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    tone = rng.uniform(0.85, 1.12, (slabs, slabs)).astype(np.float32)[iy % slabs, ix % slabs]
    edge = np.minimum(np.minimum(fx, 1 - fx), np.minimum(fy, 1 - fy))
    joint = _smooth(0.012, 0.004, edge)
    n1 = fbm(n, n, 6, 6, rng, octaves=4)
    veins = np.abs(np.sin(9.0 * fbm(n, n, 4, 4, rng, octaves=3) + 3.0 * x))
    v = (0.20 + 0.05 * n1) * tone - 0.05 * _smooth(0.08, 0.0, veins)
    col = v[..., None] * np.array([0.95, 1.0, 1.08], np.float32) * 1.7
    col = _mix(col, (0.05, 0.05, 0.06), joint * 0.9)
    return _to_u8(col)


@lru_cache(maxsize=None)
def ashlar_texture(seed: int = 0, h: int = 512, w: int = 512, rows: int = 8, cols: int = 4) -> np.ndarray:
    """Ashlar masonry: staggered rectangular blocks, cool dark grey, chipped edges."""
    rng = np.random.default_rng(seed + 1)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    ry = y / h * rows
    row = np.floor(ry).astype(int)
    rx = x / w * cols + 0.5 * (row % 2)
    fx, fy = rx - np.floor(rx), ry - np.floor(ry)
    blk = (row * 31 + np.floor(rx).astype(int) * 17) % 97
    tone = (0.8 + 0.35 * ((blk * 0.6180339) % 1.0)).astype(np.float32)
    edge = np.minimum(np.minimum(fx, 1 - fx) * (w / cols) / (h / rows), np.minimum(fy, 1 - fy))
    n1 = fbm(h, w, 8, 8, rng, octaves=4)
    joint = _smooth(0.05 + 0.03 * n1, 0.015, edge)
    bevel = _smooth(0.14, 0.05, edge) - joint
    v = (0.30 + 0.10 * n1) * tone
    col = v[..., None] * np.array([0.92, 0.96, 1.05], np.float32)
    col = col * (1.0 - 0.25 * np.clip(bevel, 0, 1)[..., None])
    col = _mix(col, (0.04, 0.04, 0.05), joint)
    return _to_u8(col)


@lru_cache(maxsize=None)
def column_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Pale weathered stone for the columns (vertical streaks, mottling)."""
    rng = np.random.default_rng(seed + 2)
    n1 = fbm(n, n, 6, 6, rng, octaves=4)
    streak = tile_noise(n, n, 2, 24, rng)
    v = 0.52 + 0.12 * n1 - 0.08 * streak
    col = v[..., None] * np.array([0.93, 0.95, 1.0], np.float32)
    return _to_u8(col)


@lru_cache(maxsize=None)
def window_texture(h: int = 512, w: int = 128) -> np.ndarray:
    """A tall arched window (seen from inside): pale cool daylight through leaded
    panes, the frame and mullions dark, brighter toward the top."""
    img = np.zeros((h, w, 3), np.float32)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = w / 2
    r = w * 0.42
    arch_y = r + 6
    inside = (np.abs(x - cx) < r) & (y > arch_y)
    inside |= ((x - cx) ** 2 + (y - arch_y) ** 2 < r ** 2) & (y <= arch_y)
    grad = 0.75 + 0.25 * (1.0 - y / h)
    sky = grad[..., None] * np.array([0.72, 0.86, 1.0], np.float32)
    img[inside] = sky[inside]
    mull = (np.abs(x - cx) < 2.0) | ((y.astype(int) % (h // 7)) < 3)
    img[inside & mull] *= 0.15
    return _to_u8(img)


def _weave(n: int, threads: int, twill: bool) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n * threads * TAU
    if twill:
        return 0.5 + 0.5 * np.sin(x + y) * np.sin(0.5 * (x - y) + 0.3)
    return 0.5 + 0.25 * (np.sin(x) + np.sin(y)) * np.sign(np.sin(x / 2) * np.sin(y / 2))


@lru_cache(maxsize=None)
def cloak_texture(seed: int = 0, n: int = 512) -> np.ndarray:
    """Heavy dark brown / near-black wool: a twill weave, mottling, and worn,
    slightly lighter, frayed edges (UV borders = the hems)."""
    rng = np.random.default_rng(seed + 5)
    wv = _weave(n, 96, twill=True)
    mott = fbm(n, n, 5, 5, rng, octaves=4)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    edge = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    fray = fbm(n, n, 24, 24, rng, octaves=2)
    wear = _smooth(0.07 + 0.04 * fray, 0.0, edge)
    base = np.array([0.15, 0.105, 0.075], np.float32)
    v = (0.75 + 0.35 * wv + 0.30 * (mott - 0.5))[..., None] * base
    worn = np.array([0.22, 0.18, 0.15], np.float32) * (0.8 + 0.4 * fray)[..., None]
    col = _mix(v, worn, wear * 0.7)
    return _to_u8(col)


@lru_cache(maxsize=None)
def tunic_texture(seed: int = 0, n: int = 256) -> np.ndarray:
    """Soft cream / beige linen: plain weave, slubs, a slightly darker hem band."""
    rng = np.random.default_rng(seed + 6)
    wv = _weave(n, 48, twill=False)
    slub = tile_noise(n, n, 64, 4, rng)
    mott = fbm(n, n, 4, 4, rng, octaves=3)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    edge = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    hem = _smooth(0.06, 0.03, edge)
    base = np.array([0.74, 0.68, 0.55], np.float32)
    v = (0.86 + 0.12 * wv + 0.06 * slub + 0.06 * (mott - 0.5))[..., None] * base
    col = _mix(v, base * 0.72, hem * 0.8)
    return _to_u8(col)


@lru_cache(maxsize=None)
def sash_texture(seed: int = 0, n: int = 128, base=(0.70, 0.58, 0.40)) -> np.ndarray:
    """A darker sand-beige woven band with two thin stripes along it."""
    rng = np.random.default_rng(seed + 7)
    wv = _weave(n, 32, twill=True)
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    stripe = (np.abs(x - 0.2) < 0.04) | (np.abs(x - 0.8) < 0.04)
    base = np.array(base, np.float32)
    v = (0.85 + 0.2 * wv + 0.05 * tile_noise(n, n, 8, 8, rng))[..., None] * base
    v[stripe] *= 0.7
    return _to_u8(v)


@lru_cache(maxsize=None)
def grip_texture(n: int = 128) -> np.ndarray:
    """The handle: brushed dark metal with ridged black grip bands."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32) / n
    brushed = 0.45 + 0.08 * np.sin(x * 300.0) + 0.05 * np.sin(y * 7.0)
    band = ((y > 0.25) & (y < 0.7)) & (np.sin(y * 90.0) > -0.2)
    col = brushed[..., None] * np.array([0.78, 0.80, 0.84], np.float32)
    col[band] = np.array([0.05, 0.05, 0.055], np.float32)
    return _to_u8(col)
