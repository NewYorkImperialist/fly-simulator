"""Visual looming on each compound eye -> LC4 / LPLC2 looming detectors (docs/VISION.md).

"The fly sees the whip coming." Every ``update_every_s`` of sim time (1 ms while a
source is moving, e.g. during a crack; ``idle_every_s`` otherwise) a physics
post-step hook computes, for each eye and each registered visual source:

* the **solid angle** Omega the source subtends inside that eye's field of view,
  from its MuJoCo geoms (capsules / cylinders exactly as thin rods plus an end-on
  disc bound, spheres exactly, other geoms as their bounding sphere);
* the **equivalent angular size** theta = 2 acos(1 - Omega / 2 pi) (the angular
  diameter of a disc of the same solid angle), low-passed with the photoreceptor /
  motion-detector time constant ``tau_s`` (5 ms), and its **expansion rate**
  dtheta/dt (finite difference of the filtered theta);
* the Omega-weighted mean direction (azimuth, elevation in the head frame; azimuth
  positive toward the eye's own side) and the nearest surface distance.

A documented response function (``loom_response``) turns (theta, dtheta/dt) into
firing rates of the eye's LC4 neurons (angular velocity, von Reyn et al. 2017) and
LPLC2 neurons (angular size during expansion, von Reyn et al. 2017; Ache et al.
2019). Whenever a rate is above ``min_event_hz`` the eye sends
``StimulusEvent(kind="loom", side=<eye>, details={lc4_hz, lplc2_hz, theta_deg,
dtheta_dps, ...})`` with a short duration (``persist_s``, the visual response
decay), refreshed while the looming continues and re-sent at once when the drive
rises. ``fly_simulator.brain.mapping`` maps it onto the FlyWire LC4 / LPLC2 cells of
that side, which drive the giant fibre (DNp01) through the real wiring.

Eyes: FlyGym's ``nmf/l_eye`` / ``nmf/r_eye`` geoms (centre = geom position, frame
= the eye body's frame: x forward, y left, z up). Without them: head position (the
eye bodies' origin or the thorax) +- a lateral offset.

Sources are generic (``LoomingVision.add_source``): MuJoCo geom ids (the whip's
capsules; a future obstacle), or a callable returning sphere / capsule arrays (a
virtual dark disc that is not in the model).

Threading: the hook runs in the physics thread (like the whip's hooks); sending
goes through ``BrainLink.send`` (non-blocking queue put).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import mujoco as mj
import numpy as np

from fly_simulator.brain.schema import StimulusEvent

EYES = ("left", "right")
TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class LoomResponse:
    """(theta, dtheta/dt) -> LC4 / LPLC2 rates (Hz). Angles in degrees.

    * LC4 encodes angular velocity (von Reyn et al. 2017):
      ``lc4 = max_hz * tanh(max(0, dtheta - lc4_v0) / lc4_vscale)``.
    * LPLC2 encodes angular size, but only for expanding (looming) objects (Ache et
      al. 2019; von Reyn et al. 2017):
      ``lplc2 = max_hz * ramp(theta; lplc2_theta0, lplc2_theta1)
      * ramp(dtheta; lplc2_gate_v0, lplc2_gate_v1)``, ramp = clip linear 0..1.
    Rates below ``min_hz`` are set to 0. Tuned on the whip (docs/VISION.md).
    """

    max_hz: float = 200.0
    lc4_v0: float = 12000.0  # deg/s: below this no LC4 drive
    lc4_vscale: float = 25000.0  # deg/s above lc4_v0 for tanh(1) = 76 % of max
    # LC4 only once the object is at least this big (deg; 0 = no size gate). Keeps
    # the jitter of a small object at a field-of-view edge from reading as looming.
    lc4_theta0: float = 0.0
    lplc2_theta0: float = 20.0  # deg
    lplc2_theta1: float = 60.0  # deg (saturation)
    lplc2_gate_v0: float = 12000.0  # deg/s: expansion gate
    lplc2_gate_v1: float = 30000.0
    min_hz: float = 10.0


def _ramp(x: float, a: float, b: float) -> float:
    if b <= a:
        return 1.0 if x >= a else 0.0
    return min(1.0, max(0.0, (x - a) / (b - a)))


def loom_response(theta_deg: float, dtheta_dps: float,
                  p: LoomResponse | None = None) -> tuple[float, float]:
    """(lc4_hz, lplc2_hz) for one eye (see ``LoomResponse``)."""
    p = p or LoomResponse()
    lc4 = p.max_hz * math.tanh(max(0.0, dtheta_dps - p.lc4_v0) / max(p.lc4_vscale, 1e-9))
    if theta_deg < p.lc4_theta0:
        lc4 = 0.0
    lplc2 = (p.max_hz * _ramp(theta_deg, p.lplc2_theta0, p.lplc2_theta1)
             * _ramp(dtheta_dps, p.lplc2_gate_v0, p.lplc2_gate_v1))
    lc4 = lc4 if lc4 >= p.min_hz else 0.0
    lplc2 = lplc2 if lplc2 >= p.min_hz else 0.0
    return lc4, lplc2


@dataclass
class LoomingConfig:
    enabled: bool = True
    update_every_s: float = 1e-3  # sim s between looming updates while a source moves
    strike_every_s: float = 2e-4  # whip swing / follow / lift (sub-ms lash)
    idle_every_s: float = 5e-3  # ... and while every source says it is idle
    # photoreceptor low-pass of theta before d/dt: first order, 1 ms = 160 Hz corner
    # frequency (the fast end of light-adapted fly photoreceptors; docs/VISION.md)
    tau_s: float = 1e-3
    samples_per_capsule: int = 6  # samples along each rod
    quad_subdiv: int = 6  # flat sources (VisualSource.quads): patches per edge
    # field of view per eye (head frame; azimuth measured from straight ahead,
    # positive toward the eye's own side): frontal binocular overlap, rear limit
    # (blind sector behind), lowest elevation; soft edges. Directions above
    # ``dorsal_both_deg`` are seen by both eyes.
    fov_front_deg: float = 15.0
    fov_rear_deg: float = 165.0
    fov_elev_min_deg: float = -70.0
    dorsal_both_deg: float = 70.0
    fov_edge_deg: float = 8.0
    # eye fallback (no eye geoms): head / thorax position +- this lateral offset
    eye_lateral_mm: float = 0.31
    # ---- events -----------------------------------------------------------------
    response: LoomResponse = field(default_factory=LoomResponse)
    min_event_hz: float = 10.0  # send only when LC4 or LPLC2 is at least this
    persist_s: float = 0.08  # brain time each loom event lasts (response decay)
    refresh_s: float = 0.01  # re-send this long before the last event ends
    rise_frac: float = 0.2  # re-send at once when a rate rises by > 20 % + rise_hz
    rise_hz: float = 10.0
    history: int = 20000  # samples kept in LoomingVision.history (diagnostics)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LoomingConfig":
        d = dict(d)
        if isinstance(d.get("response"), dict):
            d["response"] = LoomResponse(**d["response"])
        return cls(**d)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


@dataclass
class VisualSource:
    """Something the fly can see approach.

    ``geoms``: MuJoCo geom ids (positions read from ``data`` each update), or
    ``shapes``: callable returning ``(centers (K,3), axes (K,3), half_lengths (K,),
    radii (K,))`` in world mm (half length 0 = sphere). ``period``: optional
    callable returning the update period (s) this source needs right now, or None
    when it is idle (then ``LoomingConfig.idle_every_s``); without it the source
    always uses ``update_every_s``. The fastest request of all sources wins.
    """

    name: str
    geoms: Sequence[int] = ()
    shapes: Callable[[], tuple] | None = None
    period: Callable[[], float | None] | None = None
    # Flat surfaces (e.g. a flyswatter paddle): callable returning corners (Q, 4, 3)
    # of Q parallelograms in world mm (c0, c1, c2, c3 in order around the edge,
    # c2 = c1 + c3 - c0). When given, it replaces ``geoms`` / ``shapes`` for this
    # source (``quad_view``: exact triangle solid angles, per-patch field of view).
    quads: Callable[[], np.ndarray] | None = None
    # Per-source response function (None = ``LoomingConfig.response``, tuned on the
    # whip). Slow, large objects need a lower expansion threshold than the lash.
    response: "LoomResponse | None" = None


def geom_shapes(model, data, ids: np.ndarray) -> tuple:
    """Capsules / cylinders as rods, spheres exactly, anything else as its bounding
    sphere: (centers, axes, half_lengths, radii)."""
    ids = np.asarray(ids, dtype=np.int64)
    c = data.geom_xpos[ids].copy()
    mat = data.geom_xmat[ids].reshape(-1, 3, 3)
    axes = mat[:, :, 2].copy()
    t = model.geom_type[ids]
    size = model.geom_size[ids]
    rod = (t == mj.mjtGeom.mjGEOM_CAPSULE) | (t == mj.mjtGeom.mjGEOM_CYLINDER)
    sph = t == mj.mjtGeom.mjGEOM_SPHERE
    half = np.where(rod, size[:, 1], 0.0)
    r = np.where(rod | sph, size[:, 0], model.geom_rbound[ids])
    return c, axes, half, r


# ---------------------------------------------------------------------------
# geometry (both eyes at once)
# ---------------------------------------------------------------------------


def _smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def fov_mask(h: np.ndarray, side_sign: np.ndarray, cfg: LoomingConfig) -> np.ndarray:
    """Soft field-of-view weight (0..1) of head-frame unit directions ``h`` (..., 3)
    for eyes with ``side_sign`` (+1 left, -1 right; broadcast over the leading axis)."""
    s = side_sign.reshape((-1,) + (1,) * (h.ndim - 2))
    az = np.degrees(np.arctan2(h[..., 1] * s, h[..., 0]))  # + = toward the eye's side
    el = np.degrees(np.arcsin(np.clip(h[..., 2], -1.0, 1.0)))
    e = max(cfg.fov_edge_deg, 1e-6)
    m_az = _smooth01((az + cfg.fov_front_deg) / e) * _smooth01((cfg.fov_rear_deg - az) / e)
    m_az = np.maximum(m_az, _smooth01((el - cfg.dorsal_both_deg) / e))
    return m_az * _smooth01((el - cfg.fov_elev_min_deg) / e)


def eye_view(E: np.ndarray, R: np.ndarray, side_sign: np.ndarray, shapes: tuple,
             cfg: LoomingConfig) -> dict:
    """Solid angle etc. of one source for n eyes.

    E: (n, 3) eye centres; R: (n, 3, 3) head frames (columns x fwd, y left, z up).
    Returns arrays of shape (n,): omega (sr), theta (rad), azimuth / elevation
    (deg, head frame, azimuth + toward that eye's side), nearest surface distance.
    """
    c, ax, half, r = shapes
    K = len(r)
    n = len(E)
    if K == 0:
        z = np.zeros(n)
        return {"omega": z, "theta": z, "az": z, "el": z, "dist": np.full(n, np.inf)}
    S = max(1, int(cfg.samples_per_capsule))
    u = (2.0 * np.arange(S) + 1.0) / S - 1.0  # sample midpoints in (-1, 1)
    P = c[:, None, :] + ax[:, None, :] * (half[:, None] * u[None, :])[..., None]  # K,S,3
    D = P[None] - E[:, None, None, :]  # n,K,S,3
    rho = np.linalg.norm(D, axis=-1)
    rho = np.maximum(rho, 1e-9)
    ehat = D / rho[..., None]
    h = np.einsum("nji,nksj->nksi", R, ehat)  # head-frame directions
    m = fov_mask(h, side_sign, cfg)
    rr = r[None, :, None]
    w = 2.0 * np.arcsin(np.clip(rr / rho, 0.0, 1.0))  # angular width of the rod
    cross = np.linalg.norm(np.cross(ax[None, :, None, :], ehat), axis=-1)
    dl = (2.0 * half / S)[None, :, None]
    d_om = w * cross * dl / rho * m  # n,K,S
    rod_om = d_om.sum(axis=2)  # n,K
    # end-on / sphere bound: disc of radius r at the nearest sample
    j = np.argmin(rho, axis=2)  # n,K
    rho_min = np.take_along_axis(rho, j[..., None], axis=2)[..., 0]
    m_min = np.take_along_axis(m, j[..., None], axis=2)[..., 0]
    q = np.clip(r[None, :] / rho_min, 0.0, 1.0)
    sph_om = TWO_PI * (1.0 - np.sqrt(1.0 - q * q)) * m_min
    use_sph = sph_om > rod_om
    om_k = np.where(use_sph, sph_om, rod_om)
    omega = np.minimum(om_k.sum(axis=1), TWO_PI)
    theta = 2.0 * np.arccos(np.clip(1.0 - omega / TWO_PI, -1.0, 1.0))
    # direction: solid-angle weighted mean (head frame)
    wdir = (d_om[..., None] * h).sum(axis=2)  # n,K,3
    e_min = np.take_along_axis(h, j[..., None, None].repeat(3, -1), axis=2)[:, :, 0, :]
    wdir = np.where(use_sph[..., None], sph_om[..., None] * e_min, wdir).sum(axis=1)
    nrm = np.linalg.norm(wdir, axis=-1)
    wdir = wdir / np.maximum(nrm, 1e-12)[:, None]
    az = np.degrees(np.arctan2(wdir[:, 1] * side_sign, wdir[:, 0]))
    el = np.degrees(np.arcsin(np.clip(wdir[:, 2], -1.0, 1.0)))
    vis = nrm > 0
    dist = np.where(vis, (rho_min - r[None, :]).min(axis=1), np.inf)
    return {"omega": omega, "theta": theta, "az": np.where(vis, az, 0.0),
            "el": np.where(vis, el, 0.0), "dist": dist}


def _tri_solid_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Solid angle of triangles seen from the origin (Van Oosterom & Strackee 1983).
    a, b, c: (..., 3) vertex vectors relative to the eye."""
    la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
    num = np.abs(np.einsum("...i,...i->...", a, np.cross(b, c)))
    den = (la * lb * lc + np.einsum("...i,...i->...", a, b) * lc
           + np.einsum("...i,...i->...", a, c) * lb + np.einsum("...i,...i->...", b, c) * la)
    om = 2.0 * np.arctan2(num, den)
    return np.where(om < 0, om + TWO_PI, om)


def quad_view(E: np.ndarray, R: np.ndarray, side_sign: np.ndarray, quads: np.ndarray,
              cfg: LoomingConfig) -> dict:
    """Like ``eye_view`` for flat parallelograms (a paddle seen from below / aside).

    quads: (Q, 4, 3) corners (world mm). Each quad is cut into ``quad_subdiv`` x
    ``quad_subdiv`` patches of two triangles; each triangle's exact solid angle is
    weighted by the field-of-view mask at its centroid direction, so a paddle that
    fills the dorsal sky or slides out of view is handled patch by patch. ``dist`` is
    the exact eye-to-parallelogram distance.
    """
    quads = np.asarray(quads, dtype=float).reshape(-1, 4, 3)
    n = len(E)
    if len(quads) == 0:
        z = np.zeros(n)
        return {"omega": z, "theta": z, "az": z, "el": z, "dist": np.full(n, np.inf)}
    S = max(1, int(getattr(cfg, "quad_subdiv", 6)))
    g = np.linspace(0.0, 1.0, S + 1)
    c0 = quads[:, 0]
    eu = quads[:, 1] - c0  # (Q, 3)
    ev = quads[:, 3] - c0
    P = (c0[:, None, None, :] + g[None, :, None, None] * eu[:, None, None, :]
         + g[None, None, :, None] * ev[:, None, None, :])  # Q, S+1, S+1, 3
    p00, p10 = P[:, :-1, :-1], P[:, 1:, :-1]
    p01, p11 = P[:, :-1, 1:], P[:, 1:, 1:]
    tris = np.stack([np.stack([p00, p10, p11], -2), np.stack([p00, p11, p01], -2)], 1)
    tris = tris.reshape(-1, 3, 3)  # T, 3 (vertices), 3
    V = tris[None] - E[:, None, None, :]  # n, T, 3, 3
    om = _tri_solid_angle(V[..., 0, :], V[..., 1, :], V[..., 2, :])  # n, T
    cen = V.mean(axis=2)
    rho = np.maximum(np.linalg.norm(cen, axis=-1), 1e-9)
    h = np.einsum("nji,ntj->nti", R, cen / rho[..., None])
    m = fov_mask(h, side_sign, cfg)
    d_om = om * m
    omega = np.minimum(d_om.sum(axis=1), TWO_PI)
    theta = 2.0 * np.arccos(np.clip(1.0 - omega / TWO_PI, -1.0, 1.0))
    wdir = (d_om[..., None] * h).sum(axis=1)
    nrm = np.linalg.norm(wdir, axis=-1)
    wdir = wdir / np.maximum(nrm, 1e-12)[:, None]
    az = np.degrees(np.arctan2(wdir[:, 1] * side_sign, wdir[:, 0]))
    el = np.degrees(np.arcsin(np.clip(wdir[:, 2], -1.0, 1.0)))
    # exact distance eye -> parallelogram (clamped plane coordinates)
    G = np.stack([eu, ev], axis=-1)  # Q, 3, 2
    GtG = np.einsum("qki,qkj->qij", G, G)
    rel = E[:, None, :] - c0[None]  # n, Q, 3
    st = np.linalg.solve(GtG[None], np.einsum("qki,nqk->nqi", G, rel)[..., None])[..., 0]
    st = np.clip(st, 0.0, 1.0)
    near = c0[None] + np.einsum("qki,nqi->nqk", G, st)
    dist = np.linalg.norm(E[:, None, :] - near, axis=-1).min(axis=1)
    vis = nrm > 0
    return {"omega": omega, "theta": theta, "az": np.where(vis, az, 0.0),
            "el": np.where(vis, el, 0.0), "dist": dist}


# ---------------------------------------------------------------------------
# the looming "retina"
# ---------------------------------------------------------------------------


@dataclass
class EyeLoom:
    """Latest looming readout of one eye for one source (angles in degrees)."""

    theta: float = 0.0  # filtered equivalent angular size
    theta_raw: float = 0.0
    dtheta: float = 0.0  # deg/s
    az: float = 0.0
    el: float = 0.0
    dist: float = math.inf  # mm, nearest surface
    lc4_hz: float = 0.0
    lplc2_hz: float = 0.0


@dataclass
class _EyeEmit:
    t_end: float = -1e9  # sim time the last sent event ends
    lc4: float = 0.0
    lplc2: float = 0.0


class LoomingVision:
    """Post-step hook computing looming per eye; sends ``loom`` StimulusEvents.

    ``sink``: object with ``send(ev, source=...)`` (``BrainLink``) or a callable
    ``fn(ev)``; None = compute only. ``time_fn``: maps sim time -> the time stamp
    for events (``BrainLink`` wants the fly's run time across resets).
    """

    def __init__(self, sim, cfg: LoomingConfig | None = None, sink=None,
                 time_fn: Callable[[float], float] | None = None) -> None:
        self.sim = sim
        self.cfg = cfg or LoomingConfig()
        self.sink = sink
        self.time_fn = time_fn or (lambda t: t)
        self.sources: list[VisualSource] = []
        self._find_eyes()
        self.side_sign = np.array([1.0, -1.0])
        self.state: dict[str, list[EyeLoom]] = {}
        self._filt: dict[str, np.ndarray] = {}
        self._last_t: float | None = None
        self._emit = [_EyeEmit(), _EyeEmit()]
        self.history: deque = deque(maxlen=self.cfg.history)  # (t, source, eye, EyeLoom)
        self.sent: list[StimulusEvent] = []
        self.n_updates = 0
        self._attached = False

    # ------------------------------------------------------------------ setup
    def _find_eyes(self) -> None:
        m = self.sim.model
        self.eye_geoms = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, f"nmf/{s[0]}_eye") for s in EYES]
        self.eye_bodies = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"nmf/{s[0]}_eye") for s in EYES]
        self.has_eye_geoms = all(g >= 0 for g in self.eye_geoms)

    def eyes(self) -> tuple[np.ndarray, np.ndarray]:
        """(E (2,3), R (2,3,3)): left / right eye centres and head frames (world)."""
        d = self.sim.data
        tid = self.sim.thorax_body_id
        R = np.empty((2, 3, 3))
        E = np.empty((2, 3))
        for i in range(2):
            b = self.eye_bodies[i]
            R[i] = d.xmat[b if b >= 0 else tid].reshape(3, 3)
            if self.has_eye_geoms:
                E[i] = d.geom_xpos[self.eye_geoms[i]]
            else:
                base = d.xpos[b] if b >= 0 else d.xpos[tid]
                E[i] = base + self.side_sign[i] * self.cfg.eye_lateral_mm * R[i][:, 1]
        return E, R

    def add_source(self, source: VisualSource) -> VisualSource:
        self.sources.append(source)
        self.state[source.name] = [EyeLoom(), EyeLoom()]
        return source

    def attach(self) -> "LoomingVision":
        if not self._attached:
            self.sim.post_step_hooks.append(self)
            self.sim.reset_hooks.append(self._on_reset)
            self._attached = True
        return self

    def detach(self) -> None:
        if self._attached:
            self.sim.post_step_hooks.remove(self)
            self.sim.reset_hooks.remove(self._on_reset)
            self._attached = False

    def _on_reset(self, sim) -> None:
        self._filt.clear()
        self._last_t = None
        self._emit = [_EyeEmit(), _EyeEmit()]
        for st in self.state.values():
            st[0], st[1] = EyeLoom(), EyeLoom()

    # ------------------------------------------------------------------ hook
    def _period(self) -> float:
        best = self.cfg.idle_every_s
        for s in self.sources:
            p = self.cfg.update_every_s if s.period is None else s.period()
            if p is not None:
                best = min(best, p)
        return best

    def __call__(self, sim) -> None:
        if not self.cfg.enabled or not self.sources:
            return
        t = sim.time
        if self._last_t is not None and t - self._last_t < self._period() - 1e-9:
            return
        self.update()

    def shapes(self, src: VisualSource) -> tuple:
        if src.shapes is not None:
            return src.shapes()
        return geom_shapes(self.sim.model, self.sim.data, np.asarray(src.geoms))

    def update(self) -> dict[str, list[EyeLoom]]:
        """One looming update (normally called by the hook)."""
        cfg = self.cfg
        t = self.sim.time
        dt = None if self._last_t is None else t - self._last_t
        if dt is not None and (dt <= 0 or dt > 0.1):
            dt = None  # first update / after a gap: re-initialise the filters
        self._last_t = t
        E, R = self.eyes()
        best = [None, None]  # per eye: (score, source name, EyeLoom)
        for src in self.sources:
            if src.quads is not None:
                v = quad_view(E, R, self.side_sign, src.quads(), cfg)
            else:
                v = eye_view(E, R, self.side_sign, self.shapes(src), cfg)
            resp = src.response or cfg.response
            th = v["theta"]
            f = self._filt.get(src.name)
            if f is None or dt is None:
                f = np.array(th, dtype=float)
                dth = np.zeros(2)
            else:
                a = 1.0 - math.exp(-dt / max(cfg.tau_s, 1e-9))
                new = f + a * (th - f)
                dth = (new - f) / dt
                f = new
            self._filt[src.name] = f
            st = self.state[src.name]
            for i in range(2):
                lc4, lplc2 = loom_response(math.degrees(f[i]), math.degrees(dth[i]), resp)
                e = EyeLoom(theta=math.degrees(f[i]), theta_raw=math.degrees(th[i]),
                            dtheta=math.degrees(dth[i]), az=float(v["az"][i]),
                            el=float(v["el"][i]), dist=float(v["dist"][i]),
                            lc4_hz=lc4, lplc2_hz=lplc2)
                st[i] = e
                self.history.append((t, src.name, EYES[i], e))
                score = max(lc4, lplc2)
                if best[i] is None or score > best[i][0]:
                    best[i] = (score, src.name, e)
        self.n_updates += 1
        for i in range(2):
            if best[i] is not None:
                self._maybe_send(i, t, best[i][1], best[i][2])
        return self.state

    def _maybe_send(self, i: int, t: float, name: str, e: EyeLoom) -> None:
        cfg = self.cfg
        if max(e.lc4_hz, e.lplc2_hz) < cfg.min_event_hz:
            return
        em = self._emit[i]
        active = t < em.t_end
        rise = (e.lc4_hz > em.lc4 * (1 + cfg.rise_frac) + cfg.rise_hz
                or e.lplc2_hz > em.lplc2 * (1 + cfg.rise_frac) + cfg.rise_hz)
        if active and not rise and t < em.t_end - cfg.refresh_s:
            return
        # The brain applies the max rate per neuron over overlapping events, so a
        # weaker refresh lets the stronger drive run out at its own end (a decay).
        lc4, lplc2 = e.lc4_hz, e.lplc2_hz
        side = EYES[i]
        ev = StimulusEvent(
            "loom", side=side, intensity=max(lc4, lplc2) / cfg.response.max_hz,
            duration_s=cfg.persist_s, sim_time=float(self.time_fn(t)),
            details={"lc4_hz": round(lc4, 2), "lplc2_hz": round(lplc2, 2),
                     "theta_deg": round(e.theta, 2), "dtheta_dps": round(e.dtheta, 1),
                     "azimuth_deg": round(e.az, 1), "elevation_deg": round(e.el, 1),
                     "dist_mm": round(e.dist, 3) if math.isfinite(e.dist) else None,
                     "source": name, "label": f"LOOM {side[0].upper()} {name}"})
        keep = active and t < em.t_end  # strongest drive the brain still holds
        em.lc4 = max(em.lc4, lc4) if keep else lc4
        em.lplc2 = max(em.lplc2, lplc2) if keep else lplc2
        em.t_end = t + cfg.persist_s
        self.sent.append(ev)
        sink = self.sink
        if sink is None:
            return
        if hasattr(sink, "send"):
            sink.send(ev, source="vision")
        else:
            sink(ev)


# ---------------------------------------------------------------------------
# integration entry point
# ---------------------------------------------------------------------------


def whip_source(whip, cfg: LoomingConfig | None = None) -> VisualSource:
    """The whip's chain capsules + grip as one visual source. Update cadence:
    ``strike_every_s`` during swing / follow / lift (the lash lasts a few ms),
    ``update_every_s`` during the rest of a crack, idle otherwise."""
    cfg = cfg or LoomingConfig()
    fast, slow = cfg.strike_every_s, cfg.update_every_s
    fast_phases = ("swing", "follow", "lift")
    m = whip.sim.model
    geoms = [int(g) for g in whip.whip_geoms]
    gid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "whip/grip_geom")
    if gid >= 0:
        geoms.append(gid)
    def period():
        ph = whip.phase
        if ph in fast_phases:
            return fast
        return None if ph == "idle" else slow

    return VisualSource("whip", geoms=geoms, period=period)


def install_whip_vision(session_or_sim, whip, brain_link=None,
                        cfg: LoomingConfig | dict | None = None) -> LoomingVision:
    """Install whip looming vision; returns the ``LoomingVision`` handle.

    ``session_or_sim``: an app ``Session`` (uses ``.sim``) or a ``Simulation``.
    ``whip``: an attached ``Whip``. ``brain_link``: a ``BrainLink`` (events go through
    ``BrainLink.send``, stamped with its run time, logged to events.csv), any object
    with ``send(ev, source=...)``, a callable, or None (compute only).
    More sources: ``handle.add_source(VisualSource(...))``. Remove: ``handle.detach()``.
    """
    sim = getattr(session_or_sim, "sim", session_or_sim)
    if isinstance(cfg, dict):
        cfg = LoomingConfig.from_dict(cfg)
    time_fn = None
    if brain_link is not None and hasattr(brain_link, "_run_time"):
        time_fn = brain_link._run_time
    elif hasattr(session_or_sim, "metrics"):
        time_fn = session_or_sim.metrics.run_time_at
    vis = LoomingVision(sim, cfg, sink=brain_link, time_fn=time_fn)
    if whip is not None:
        vis.add_source(whip_source(whip, vis.cfg))
    return vis.attach()
