"""Bridge: flyvis T4 / T5 motion -> LC4 / LPLC2 looming detectors (docs/VISION.md).

"Real vision" part 3. flyvis stops at the elementary motion detectors T4 (ON edges)
and T5 (OFF edges), four direction subtypes each. LC4 and LPLC2, the visual
projection neurons that drive the giant fibre, are downstream and not in flyvis.
This module is a *designed* (not connectome-derived, not trained) model of how they
read T4 / T5, following what is known about them:

* **LPLC2** (Klapoetke et al. 2017, Nature 551:237): four dendritic arms, one per
  lobula-plate layer, each extending from the cell's receptive-field centre in the
  direction that the layer's T4 / T5 subtype prefers, so the cell is driven by
  motion *radiating outward* in all four cardinal directions at once (expansion
  centred on its receptive field), and inhibited by inward motion (via LPi
  interneurons). Translation or wide-field flow drives at most one or two arms.
  Model: for each unit centred on ommatidium c, arm direction d (up / down / front
  / back on the eye), ``A_cd = relu(mean_arm relu(m.d) - k_in * mean_arm relu(-m.d))``
  with ``m`` the local motion vector; ``LPLC2_c = (prod_d A_cd)^(1/4)`` (all four
  arms needed: a geometric mean). Receptive field radius ``rf_deg`` (30 deg ~ the
  ~60 deg LPLC2 receptive field).
* **LC4** (von Reyn et al. 2017, Neuron 94:1190): encodes the angular velocity
  of a dark looming object; weak for translating objects. Model: from T5 only (OFF
  edges: the rim of a dark expanding object is a bright-to-dark edge moving
  outward): the outward flux of the OFF motion through the unit's receptive field,
  weighted by how radial it is: ``LC4_c = relu(flux_c) * relu(ratio_c)^2`` with
  ``flux_c = mean_j m_j . r_cj`` over the disc of radius ``rf_deg`` and
  ``ratio_c = sum_j m_j . r_cj / sum_j |m_j|`` (1 for pure expansion about c, ~0 for
  translation, < 0 for contraction). Faster edges -> larger T5 increments -> more
  flux, so LC4 grows with expansion speed.
* **Local motion vector** per ommatidium: each T4 / T5 cell's *rectified increment
  over its own slow running mean* (``adapt_tau_s`` 0.3 s: static scene structure and
  tonic activity adapt away), times its subtype's preferred direction, summed over
  the four subtypes. Preferred directions were measured on flyvis model 000 with
  moving ON / OFF edges in FlyGym eye-image (row, col) coordinates (``PD_ROWCOL``:
  a -col, b +col, c up, d down; T4d is weakly tuned in that model, T5d is fine).
  Both eyes go through the same network; their images are mirror images of each
  other (image +col is anterior for the left eye, posterior for the right), which
  does not matter for expansion. T4 and T5 are combined for LPLC2 (it responds to
  bright and dark looming), T5 weighted ``off_gain`` = 1.5 (LPLC2 prefers dark).
  The eye's coherent (mean) motion vector is removed (self-motion drives every
  column the same way; expansion averages out), a stand-in for wide-field
  inhibition.
* **Rates**: the population readout per eye is the maximum over units, turned into
  firing rates by ramps: ``lplc2_hz = max_hz * ramp(LPLC2; 0.12, 0.30)``,
  ``lc4_hz = max_hz * ramp(LC4; 0.24, 0.45)``. Thresholds were set on synthetic
  stimuli on the ommatidia lattice (``scripts/demo_real_vision.py --calibrate``:
  dark looms l/v 10-80 ms give LPLC2 0.24-0.34 / LC4 0.29-0.56; translating discs,
  receding / slowly expanding discs and gratings stay <= 0.10 / 0.17) and checked
  on rendered scenes (walking on flat / normal terrain: LPLC2 <= 0.025, LC4 <= 0.045;
  a dark sphere looming at the fly in the sim: LPLC2 0.29-0.31, LC4 0.45-0.53).

Optional steering population (``steer=True``, off by default): **LC10a** (small moving
dark objects; courtship target tracking, Ribeiro et al. 2018) driven per eye by the
strongest *small-field* OFF motion (radius ``lc10_rf_deg``) after removing the global
motion, sent as ``StimulusEvent("manual", details={"cell_type": "LC10a", "rate_hz"})``
(the sensory screen found LC10a to drive the ipsilateral DNa02 turn neuron).

Its thresholds (0.45 / 0.8) were set on rendered scenes (walking <= 0.42, a dark
sphere passing 6 mm to the side 0.71). Experimental.

Eyes-only fallback (no flyvis): ``DarkExpansionBridge`` estimates, per eye, the
fraction of ommatidia much darker than their running mean and its growth rate
(-> LC4) and the fraction while growing (-> LPLC2); clearly labelled
``bridge="dark_expansion"`` in the events. It is not a neural model and was only
unit-tested, not calibrated in the closed loop (flyvis installed fine here).

All angles in eye-image degrees (~0.307 deg per FlyGym eye pixel, ~5 deg ommatidial
spacing).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from fly_simulator.brain.schema import StimulusEvent

EYES = ("left", "right")
DEG_PER_PX = 157.0 / 512.0  # FlyGym eye camera fovy / image height (fisheye-corrected)
# Preferred directions of flyvis T4/T5 subtypes a-d in FlyGym eye-image (row, col)
# coordinates, measured with moving ON (T4) / OFF (T5) edges on model flow/0000/000:
# a -> -col, b -> +col, c -> -row (up), d -> +row (down).
PD_ROWCOL = np.array([[0.0, -1.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]])
ARM_DIRS = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])  # up, down, -col, +col


def ommatidia_centers() -> np.ndarray:
    """(721, 2) ommatidium centres in FlyGym eye-image pixels (row, col)."""
    from flygym.vision.retina import Retina

    idm = Retina().ommatidia_id_map
    rr, cc = np.nonzero(idm)
    ids = idm[rr, cc].astype(np.int64) - 1
    cen = np.zeros((int(ids.max()) + 1, 2))
    np.add.at(cen, ids, np.c_[rr, cc])
    return cen / np.bincount(ids)[:, None]


def _ramp(x, a, b):
    return np.clip((x - a) / max(b - a, 1e-12), 0.0, 1.0)


@dataclass
class BridgeConfig:
    rf_deg: float = 30.0  # LPLC2 / LC4 arm length (receptive-field radius)
    arm_half_angle_deg: float = 45.0
    k_in: float = 1.0  # inward-motion inhibition
    off_gain: float = 1.5  # T5 weight relative to T4 for LPLC2
    adapt_tau_s: float = 0.3  # slow running mean (per T4/T5 cell) = its baseline
    remove_global: bool = True  # subtract each eye's mean motion vector
    # rates (calibrated, see module docstring)
    max_hz: float = 200.0
    lplc2_on: float = 0.12
    lplc2_sat: float = 0.3
    lc4_on: float = 0.24
    lc4_sat: float = 0.45
    min_hz: float = 10.0
    # steering population (off by default)
    steer: bool = False
    lc10_cell_type: str = "LC10a"
    lc10_rf_deg: float = 10.0
    lc10_on: float = 0.45
    lc10_sat: float = 0.8
    lc10_max_hz: float = 150.0

    def to_dict(self) -> dict:
        return asdict(self)


def _arm_matrices(cen: np.ndarray, rf_px: float, half_angle_deg: float) -> tuple:
    """(4, n, n) averaging weights: row c = the ommatidia in arm d of the unit at c."""
    d = cen[None, :, :] - cen[:, None, :]  # (c, j, 2)
    r = np.linalg.norm(d, axis=-1)
    cos_lim = math.cos(math.radians(half_angle_deg))
    W = np.zeros((4, len(cen), len(cen)))
    for k, a in enumerate(ARM_DIRS):
        proj = d @ a
        m = (r > 1e-6) & (r <= rf_px) & (proj >= cos_lim * r)
        W[k] = m
    n = W.sum(axis=2, keepdims=True)
    return W / np.maximum(n, 1.0), (n[..., 0] >= 3)  # units need populated arms


class LoomBridge:
    """T4 / T5 (2, 4, 721) -> per-eye LC4 / LPLC2 model activations and rates."""

    def __init__(self, cfg: BridgeConfig | None = None, centers: np.ndarray | None = None) -> None:
        self.cfg = cfg or BridgeConfig()
        self.cen = ommatidia_centers() if centers is None else np.asarray(centers, float)
        c = self.cfg
        self.W, ok = _arm_matrices(self.cen, c.rf_deg / DEG_PER_PX, c.arm_half_angle_deg)
        self.unit_ok = ok.all(axis=0)  # units whose four arms all lie on the eye
        d = self.cen[None, :, :] - self.cen[:, None, :]
        r = np.linalg.norm(d, axis=-1)
        rhat = d / np.maximum(r, 1e-9)[..., None]  # (c, j, 2) unit vectors c -> j
        self.disk = ((r > 1e-6) & (r <= c.rf_deg / DEG_PER_PX)).astype(float)
        self.disk_n = self.disk.sum(axis=1)[None, :]
        self.Dx = self.disk * rhat[..., 0]  # radial projection weights (row / col parts)
        self.Dy = self.disk * rhat[..., 1]
        self.W_small, ok_s = _arm_matrices(self.cen, c.lc10_rf_deg / DEG_PER_PX, 60.0)
        self.W_local = (self.W_small.sum(axis=0) / 4.0).astype(np.float32)
        self.W = self.W.astype(np.float32)
        self.disk = self.disk.astype(np.float32)
        self.Dx, self.Dy = self.Dx.astype(np.float32), self.Dy.astype(np.float32)
        del self.W_small
        self._base: dict[str, np.ndarray] = {}
        self.last: dict = {}

    def reset(self) -> None:
        self._base.clear()
        self.last = {}

    def _vec(self, T: np.ndarray, key: str, dt: float) -> np.ndarray:
        """(2, 4, n) subtype activity -> (2, n, 2) motion vectors: each subtype's
        rectified increment over its own slow running mean (static scene structure
        and tonic activity adapt away), times its preferred direction, summed; the
        eye's coherent (mean) vector is removed when ``remove_global``."""
        b = self._base.get(key)
        if b is None:
            b = T.copy()
        inc = np.maximum(T - b, 0.0)
        a = 1.0 - math.exp(-dt / max(self.cfg.adapt_tau_s, 1e-9))
        self._base[key] = b + a * (T - b)
        m = np.einsum("esn,sk->enk", inc, PD_ROWCOL).astype(np.float32)
        if self.cfg.remove_global:
            m = m - m.mean(axis=1, keepdims=True)
        return m

    def _arms(self, m: np.ndarray, W: np.ndarray) -> np.ndarray:
        """(2, n, 2) vectors -> (2, 4, n_units) arm drives A_cd."""
        ne = m.shape[0]
        out = np.empty((ne, 4, W.shape[1]), dtype=np.float32)
        for k, a in enumerate(ARM_DIRS.astype(np.float32)):
            p = (m @ a).T  # (n, 2 eyes) component along the arm direction
            r = W[k] @ np.concatenate([np.maximum(p, 0), np.maximum(-p, 0)], axis=1)
            out[:, k] = np.maximum(r[:, :ne] - self.cfg.k_in * r[:, ne:], 0).T
        return out

    def update(self, t4: np.ndarray, t5: np.ndarray, dt: float) -> dict:
        c = self.cfg
        m_on = self._vec(t4, "on", dt)
        m_off = self._vec(t5, "off", dt)
        m_all = (m_on + c.off_gain * m_off) / (1.0 + c.off_gain)
        A = self._arms(m_all, self.W)  # (2, 4, n)
        lplc2_u = np.prod(A, axis=1) ** 0.25 * self.unit_ok
        # LC4: outward flux of the OFF motion through each receptive field, weighted
        # by how radial it is (ratio^2: 1 for pure expansion, ~0 for translation)
        radial = (self.Dx @ m_off[..., 0].T + self.Dy @ m_off[..., 1].T).T  # (2, c)
        mag = (self.disk @ np.linalg.norm(m_off, axis=-1).T).T
        flux = radial / self.disk_n
        ratio = radial / np.maximum(mag, 1e-9)
        lc4_u = np.maximum(flux, 0.0) * np.maximum(ratio, 0.0) ** 2 * self.unit_ok
        iu = lplc2_u.argmax(axis=1)
        il = lc4_u.argmax(axis=1)
        lplc2 = lplc2_u.max(axis=1)
        lc4 = lc4_u.max(axis=1)
        out = {
            "lplc2": lplc2, "lc4": lc4, "lplc2_unit": iu, "lc4_unit": il,
            "lplc2_hz": self._rate(lplc2, c.lplc2_on, c.lplc2_sat),
            "lc4_hz": self._rate(lc4, c.lc4_on, c.lc4_sat),
            "lplc2_map": lplc2_u, "lc4_map": lc4_u, "motion_off": m_off, "motion_on": m_on,
        }
        if c.steer:
            # small-field OFF motion of any direction (a small dark moving object)
            sp = np.linalg.norm(m_off, axis=-1)  # (2, n)
            loc = sp @ self.W_local.T  # local mean speed around c
            lc10 = np.maximum(loc - 0.5 * sp.mean(axis=1, keepdims=True), 0).max(axis=1)
            out["lc10"] = lc10
            out["lc10_hz"] = self._rate(lc10, c.lc10_on, c.lc10_sat, c.lc10_max_hz)
        self.last = out
        return out

    def _rate(self, x, a, b, max_hz=None):
        r = (self.cfg.max_hz if max_hz is None else max_hz) * _ramp(np.asarray(x), a, b)
        return np.where(r >= self.cfg.min_hz, r, 0.0)


@dataclass
class DarkExpansionConfig:
    """Eyes-only fallback (no flyvis). Not a neural model."""

    dark_frac: float = 0.6  # ommatidium is "dark" below this x its running mean
    adapt_tau_s: float = 1.0
    tau_s: float = 0.01  # low-pass of the dark solid angle before d/dt
    max_hz: float = 200.0
    rate_on: float = 1.0  # d(dark fraction of the eye)/dt [1/s] for onset
    rate_sat: float = 5.0
    frac_on: float = 0.05  # LPLC2: dark fraction while expanding
    frac_sat: float = 0.3
    min_hz: float = 10.0


class DarkExpansionBridge:
    """Per eye: fraction of ommatidia much darker than their running mean, its growth
    rate -> LC4 (growth rate) and LPLC2 (fraction while growing)."""

    def __init__(self, cfg: DarkExpansionConfig | None = None) -> None:
        self.cfg = cfg or DarkExpansionConfig()
        self.reset()

    def reset(self) -> None:
        self._mean = None
        self._f = None
        self.last = {}

    def update(self, frame: np.ndarray, dt: float) -> dict:
        c = self.cfg
        if self._mean is None:
            self._mean = frame.astype(float).copy()
        dark = (frame < c.dark_frac * self._mean).mean(axis=1)
        a = 1.0 - math.exp(-dt / max(c.adapt_tau_s, 1e-9))
        self._mean = self._mean + a * (frame - self._mean)
        if self._f is None:
            self._f = dark
            rate = np.zeros(2)
        else:
            b = 1.0 - math.exp(-dt / max(c.tau_s, 1e-9))
            new = self._f + b * (dark - self._f)
            rate = (new - self._f) / max(dt, 1e-9)
            self._f = new
        lc4 = c.max_hz * _ramp(rate, c.rate_on, c.rate_sat)
        lplc2 = c.max_hz * _ramp(self._f, c.frac_on, c.frac_sat) * (rate > 0.5 * c.rate_on)
        self.last = {"dark_frac": self._f.copy(), "dark_rate": rate,
                     "lc4_hz": np.where(lc4 >= c.min_hz, lc4, 0.0),
                     "lplc2_hz": np.where(lplc2 >= c.min_hz, lplc2, 0.0)}
        return self.last


# ---------------------------------------------------------------------------
# the pipeline: eyes -> flyvis -> bridge -> brain
# ---------------------------------------------------------------------------


@dataclass
class RealVisionConfig:
    enabled: bool = False
    rate_hz: float = 100.0  # eye samples = flyvis steps per sim second
    backend: str = "auto"  # "flyvis" | "dark_expansion" | "auto" (flyvis if available)
    flyvis_model: str = "flow/0000/000"
    threads: int = 2
    warmup_s: float = 0.05  # no events for this long after start / reset (network settling)
    bridge: dict = field(default_factory=dict)  # BridgeConfig overrides
    fallback: dict = field(default_factory=dict)  # DarkExpansionConfig overrides
    # event emission (same semantics as the geometric sense, fly_simulator/vision/looming.py)
    min_event_hz: float = 10.0
    persist_s: float = 0.08
    refresh_s: float = 0.01
    rise_frac: float = 0.2
    rise_hz: float = 10.0
    history: int = 4000

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RealVisionConfig":
        return cls(**dict(d))


@dataclass
class _Emit:
    t_end: float = -1e9
    lc4: float = 0.0
    lplc2: float = 0.0
    lc10_end: float = -1e9
    lc10: float = 0.0


class RealVision:
    """Eyes -> flyvis -> bridge -> ``loom`` StimulusEvents (and LC10a with steer).

    ``eyes``: an attached ``CompoundEyes`` (its rate is the network rate). ``sink``:
    ``BrainLink`` (``send(ev, source=...)``), a callable or None. ``time_fn``: sim time
    -> event time stamp (the brain link's run time).
    """

    def __init__(self, eyes, cfg: RealVisionConfig | None = None, sink=None,
                 time_fn: Callable[[float], float] | None = None,
                 say: Callable[[str], None] | None = None) -> None:
        self.eyes = eyes
        self.cfg = cfg or RealVisionConfig(enabled=True)
        self.sink = sink
        self.time_fn = time_fn or (lambda t: t)
        self.say = say or (lambda m: None)
        self.dt = eyes.period
        backend = self.cfg.backend
        self.net = None
        self.bridge = None
        why = None
        if backend in ("auto", "flyvis"):
            from fly_simulator.vision.flyvis_net import FlyvisConfig, StepwiseFlyvis, flyvis_available

            why = flyvis_available()
            if why is None:
                self.net = StepwiseFlyvis(self.dt, FlyvisConfig(model=self.cfg.flyvis_model,
                                                                threads=self.cfg.threads))
                self.bridge = LoomBridge(BridgeConfig(**self.cfg.bridge))
                self.backend = "flyvis"
            elif backend == "flyvis":
                raise RuntimeError(f"--real-vision: {why}")
        if self.net is None:
            self.backend = "dark_expansion"
            self.bridge = DarkExpansionBridge(DarkExpansionConfig(**self.cfg.fallback))
            self.say(f"[vision] flyvis unavailable ({why}); eyes-only fallback: dark-area "
                     "expansion estimator -> LC4 / LPLC2 (not a neural model)")
        self._emit = [_Emit(), _Emit()]
        self._t0: float | None = None
        self.history: deque = deque(maxlen=self.cfg.history)  # (t, lc4_hz(2), lplc2_hz(2), raw)
        self.sent: list[StimulusEvent] = []
        self.wall_s = 0.0
        self.n_frames = 0
        self.enabled = True
        eyes.listeners.append(self.on_frame)
        eyes.sim.reset_hooks.append(self._on_reset)

    def detach(self) -> None:
        if self.on_frame in self.eyes.listeners:
            self.eyes.listeners.remove(self.on_frame)
        if self._on_reset in self.eyes.sim.reset_hooks:
            self.eyes.sim.reset_hooks.remove(self._on_reset)

    def _on_reset(self, sim) -> None:
        self._t0 = None
        self._emit = [_Emit(), _Emit()]
        self.bridge.reset()
        if self.net is not None:
            self.net.state = None

    # ------------------------------------------------------------------ frame
    def on_frame(self, t: float, frame: np.ndarray) -> None:
        if not self.enabled:
            return
        t0 = time.perf_counter()
        if self._t0 is None:
            self._t0 = t
            if self.net is not None:
                self.net.reset(frame)
        if self.net is not None:
            self.net.step(frame)
            t4, t5 = self.net.motion()
            out = self.bridge.update(t4, t5, self.dt)
        else:
            out = self.bridge.update(frame, self.dt)
        self.n_frames += 1
        lc4, lplc2 = out["lc4_hz"], out["lplc2_hz"]
        self.history.append((t, lc4.copy(), lplc2.copy(),
                             {k: out[k].copy() for k in ("lc4", "lplc2", "lc10", "dark_rate")
                              if k in out}))
        if t - self._t0 >= self.cfg.warmup_s:
            for i in range(2):
                self._maybe_send(i, t, float(lc4[i]), float(lplc2[i]), out)
                if "lc10_hz" in out:
                    self._maybe_send_lc10(i, t, float(out["lc10_hz"][i]))
        self.wall_s += time.perf_counter() - t0

    def ms_per_frame(self) -> float:
        return 1e3 * self.wall_s / max(self.n_frames, 1)

    def _deliver(self, ev: StimulusEvent) -> None:
        self.sent.append(ev)
        s = self.sink
        if s is None:
            return
        if hasattr(s, "send"):
            s.send(ev, source="real_vision")
        else:
            s(ev)

    def _maybe_send(self, i: int, t: float, lc4: float, lplc2: float, out: dict) -> None:
        c = self.cfg
        if max(lc4, lplc2) < c.min_event_hz:
            return
        em = self._emit[i]
        active = t < em.t_end
        rise = (lc4 > em.lc4 * (1 + c.rise_frac) + c.rise_hz
                or lplc2 > em.lplc2 * (1 + c.rise_frac) + c.rise_hz)
        if active and not rise and t < em.t_end - c.refresh_s:
            return
        side = EYES[i]
        det = {"lc4_hz": round(lc4, 2), "lplc2_hz": round(lplc2, 2),
               "bridge": self.backend, "source": "real_vision",
               "label": f"LOOM {side[0].upper()} eyes"}
        if "lplc2" in out:
            det["lplc2_act"] = round(float(out["lplc2"][i]), 4)
            det["lc4_act"] = round(float(out["lc4"][i]), 4)
            u = int(out["lplc2_unit"][i])
            r, col = self.bridge.cen[u]
            det["rf_row_col"] = [round(float(r), 1), round(float(col), 1)]
        ev = StimulusEvent("loom", side=side, intensity=max(lc4, lplc2) / 200.0,
                           duration_s=c.persist_s, sim_time=float(self.time_fn(t)), details=det)
        em.lc4 = max(em.lc4, lc4) if active else lc4
        em.lplc2 = max(em.lplc2, lplc2) if active else lplc2
        em.t_end = t + c.persist_s
        self._deliver(ev)

    def _maybe_send_lc10(self, i: int, t: float, hz: float) -> None:
        c = self.cfg
        if hz < c.min_event_hz:
            return
        em = self._emit[i]
        active = t < em.lc10_end
        if active and hz <= em.lc10 * (1 + c.rise_frac) + c.rise_hz and t < em.lc10_end - c.refresh_s:
            return
        side = EYES[i]
        bc = self.bridge.cfg
        ev = StimulusEvent("manual", side=side, intensity=hz / 200.0, duration_s=c.persist_s,
                           sim_time=float(self.time_fn(t)),
                           details={"cell_type": bc.lc10_cell_type, "rate_hz": round(hz, 2),
                                    "source": "real_vision",
                                    "label": f"{bc.lc10_cell_type} {side[0].upper()} eyes"})
        em.lc10 = max(em.lc10, hz) if active else hz
        em.lc10_end = t + c.persist_s
        self._deliver(ev)


def install_real_vision(session_or_sim, brain_link=None, cfg: RealVisionConfig | dict | None = None,
                        eyes=None, say: Callable[[str], None] | None = None) -> RealVision:
    """Compound eyes + flyvis + bridge on a running app ``Session`` or ``Simulation``
    whose fly was built with ``make_eyes_fly_factory()``. Returns the ``RealVision``
    handle (``.eyes``, ``.net``, ``.bridge``, ``.sent``, ``.history``)."""
    from fly_simulator.vision.eyes import CompoundEyes, EyesConfig

    sim = getattr(session_or_sim, "sim", session_or_sim)
    if isinstance(cfg, dict):
        cfg = RealVisionConfig.from_dict(cfg)
    cfg = cfg or RealVisionConfig(enabled=True)
    if eyes is None:
        eyes = CompoundEyes(sim, EyesConfig(enabled=True, rate_hz=cfg.rate_hz)).attach()
    link = brain_link if brain_link is not None else getattr(session_or_sim, "brain", None)
    time_fn = None
    if link is not None and hasattr(link, "_run_time"):
        time_fn = link._run_time
    elif hasattr(session_or_sim, "metrics"):
        time_fn = session_or_sim.metrics.run_time_at
    return RealVision(eyes, cfg, sink=link, time_fn=time_fn, say=say)
