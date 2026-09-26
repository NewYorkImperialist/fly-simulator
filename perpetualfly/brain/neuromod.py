"""Octopamine "stress / arousal" level: a phenomenological neuromodulation layer.

Flies have no cortisol. Their closest equivalent of the vertebrate noradrenaline /
fight-or-flight system is **octopamine (OA)**: OA neurons are recruited by
threatening and startling stimuli and raise arousal and locomotor vigour
(Roeder 2005; Suver et al. 2012; Crocker & Sehgal 2008). DH44 neurons (a CRF
homologue) are the insect counterpart of the stress-hormone axis; the model reports
their firing but they are not used (no stimulus here reaches them).

What is real and what is modelled
---------------------------------
* **Real (connectome model):** the *trigger*. The level is driven only by the
  spikes of the annotated octopaminergic neurons of the Shiu et al. LIF brain
  (FlyWire ``cell_type`` ``OA-*``: OA-VUMa1..8, OA-VPM3/4, OA-AL2b/i, OA-ASM1..3;
  Busch et al. 2009 nomenclature; 43 neurons in v783). Whether a body event
  raises it depends entirely on whether the connectome routes that input to those
  neurons. The predicted-transmitter label (``nt == OA``) is *not* used on its
  own: in v783 it also tags ~110 photoreceptors (R1-6/R7/R8, really histaminergic;
  the transmitter classifier has no histamine class) and some JO afferents.
* **Modelled (not in the connectome):** everything after the trigger.
  Octopamine acts by volume release on G-protein-coupled receptors whose
  expression is not in the connectome, and the Shiu model has no neuromodulation.
  So the *level* is a leaky integrator of OA-neuron firing, and its *effect* is a
  choice made here, documented as such:

  - brain: the spike threshold of the direct synaptic partners of the OA neurons
    (>= ``target_min_syn`` synapses from any OA-* neuron) is lowered by
    ``vth_shift_mv * level`` (octopamine increases the excitability of its targets,
    e.g. via OAMB / Octbeta receptors and cAMP). Synaptic partners stand in for
    "neurons near OA release sites"; receptor expression is unknown.
  - body (``perpetualfly/stress.py``): faster stepping, lower escape threshold.

Level dynamics (``level`` in [0, 1])::

    u_OA   = r_OA / ref_rate_hz                    r_OA = mean rate of the OA-* set
    u_noc  = noci_gain * r_hit / noci_ref_hz       (only with nociceptive_input)
    u      = min(u_OA + u_noc, max_drive)
    dL/dt  = u (1 - L) / tau_rise  -  L / tau_decay

With OA neurons firing at ``ref_rate_hz`` the level rises with ``tau_rise`` (~1 s)
and saturates toward 1; repeated threats accumulate because the decay (20-60 s) is
slow. Integrated exactly per engine chunk (u constant within a chunk).

**Nociceptive / "pain" input.** In this connectome model whip hits and shoves do
**not** reach the OA neurons at all (0 Hz at every strength, measured in
docs/STRESS.md): FlyWire has no annotated nociceptors, and the real route runs
through the ventral nerve cord, which the model lacks. Two clearly labelled paths
close that gap (the body-side ``StressConfig`` turns on the first):

* ``noci_relay`` (**VNC stand-in**): each hit additionally drives two groups of
  ascending neurons, both sides, for the hit duration + ``relay_extra_s``:
  ``an_walk`` (AN_AVLP_PVLP) at ``100 + 50 * intensity`` Hz and ``an_arousal``
  (AN_IPS_GNG_7) at ``50 + 70 * intensity`` Hz (docs/SENSORY_SCREEN.md found them
  to drive the walk DNs and OA-VUMa1). *Which* ascending neurons carry the signal
  is our choice; everything downstream (OA-neuron spikes, hence the level; walk-DN
  spikes, hence the brain's own steering drive) is connectome.
* ``nociceptive_input`` (off): the hit afferents' own spike rate ``r_hit`` (most
  active side / set) feeds the level directly; the afferent -> octopamine link is
  then entirely modelled.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np

OA_CELL_TYPE_PREFIX = "OA-"
DH44_CELL_TYPE = "DH44"
MODEL_LABEL = "octopamine (model)"
NOCI_SETS = ("body_mech", "head_bristle")  # brain/mapping.named_sets: the hit afferents
HIT_KINDS = ("whip_hit", "shove", "hit")
# ascending neurons driven by the nociceptive relay (docs/SENSORY_SCREEN.md)
RELAY_WALK_SET = "an_walk"        # AN_AVLP_PVLP -> BDN2 / oDN1 / P9 (walk DNs)
RELAY_AROUSAL_SET = "an_arousal"  # AN_IPS_GNG_7 -> OA-VUMa1 (octopamine)


@dataclass
class NeuromodConfig:
    enabled: bool = False
    tau_rise_s: float = 1.0        # rise time constant at r_OA = ref_rate_hz
    tau_decay_s: float = 30.0      # decay back to baseline (config: 20-60 s)
    ref_rate_hz: float = 10.0      # OA-* population mean rate giving u = 1
    max_drive: float = 10.0        # cap on u
    target_min_syn: int = 5        # synapses from OA-* neurons to count as a target
    vth_shift_mv: float = 1.0      # threshold lowering at level 1 (gap is 7 mV)
    # Nociceptive relay (VNC stand-in): each whip hit / shove additionally drives
    # ascending neurons (both sides) for the hit duration + relay_extra_s. WHICH
    # ascending neurons carry the signal is our choice (FlyWire has no annotated
    # nociceptors; they end in the VNC, which the model lacks); everything
    # downstream (OA neurons, walk DNs) is connectome.
    noci_relay: bool = False
    relay_walk_hz: tuple = (100.0, 50.0)     # an_walk rate = a + b * intensity
    relay_arousal_hz: tuple = (50.0, 70.0)   # an_arousal rate = a + b * intensity
    relay_extra_s: float = 0.2
    # Alternative, fully modelled link (off): the hit afferents' own spike rate
    # drives the level directly (u += noci_gain * r_hit / noci_ref_hz)
    nociceptive_input: bool = False
    noci_gain: float = 1.0
    noci_ref_hz: float = 35.0      # hit-afferent rate (hit side) giving u = 1
    # The Shiu model can fall into a global self-sustaining state (~475k spikes/s,
    # docs/SENSORY_SCREEN.md), a model artefact. While the whole-brain rate is above
    # runaway_sps the level gets no input (it only decays) and the threshold shift
    # is removed, so the artefact neither pins the level at 1 nor is fed by it.
    runaway_sps: float = 100_000.0
    persist_on_reset: bool = True  # a fly/brain reset keeps the level
    initial_level: float = 0.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "NeuromodConfig":
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self) -> dict:
        return asdict(self)


def select_oa_neurons(table) -> np.ndarray:
    """Model indices of the octopaminergic neurons driving the level.

    FlyWire cell types ``OA-*`` (identified OA neurons). If none exist (synthetic
    test networks), falls back to non-sensory neurons with predicted transmitter OA.
    """
    ct = table.col("cell_type")
    m = np.array([str(x).startswith(OA_CELL_TYPE_PREFIX) for x in ct], dtype=bool)
    if not m.any():
        sc = table.col("super_class")
        m = (np.asarray(table.nt) == 5) & ~np.array(
            [str(x).startswith("sensory") for x in sc], dtype=bool)
    return np.nonzero(m)[0].astype(np.int64)


def oa_targets(indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray,
               src: np.ndarray, n: int, min_syn: float, w_syn: float = 1.0) -> np.ndarray:
    """Postsynaptic partners receiving >= ``min_syn`` synapses (summed over ``src``).

    ``weights`` are signed synapse counts times ``w_syn`` (the engine's scaled
    float32 weights with ``w_syn = LIFParams.w_syn``, or raw counts with 1).
    """
    tot = np.zeros(n, dtype=np.float64)
    for j in np.asarray(src, dtype=np.int64):
        a, b = int(indptr[j]), int(indptr[j + 1])
        np.add.at(tot, indices[a:b], np.abs(weights[a:b].astype(np.float64)) / w_syn)
    return np.nonzero(tot >= float(min_syn) - 1e-6)[0].astype(np.int64)


def step_level(level: float, u: float, dt_s: float, tau_rise_s: float,
               tau_decay_s: float) -> float:
    """Exact solution of dL/dt = u(1-L)/tau_rise - L/tau_decay over dt (u const)."""
    a = max(u, 0.0) / max(tau_rise_s, 1e-9)
    b = 1.0 / max(tau_decay_s, 1e-9)
    k = a + b
    l_inf = a / k
    return float(l_inf + (level - l_inf) * math.exp(-k * dt_s))


class OctopamineModel:
    """Lives in the brain worker next to the engine (``process._Model``).

    ``observe(spike_idx, dt_s)`` after every engine chunk updates the level from the
    OA-* spikes in the chunk and refreshes the engine's per-neuron thresholds;
    ``readout()`` is published as ``BrainState.neuromod``.
    """

    def __init__(self, table, engine, cfg: NeuromodConfig | dict | None = None):
        self.table = table
        self.engine = engine
        self.cfg = cfg if isinstance(cfg, NeuromodConfig) else NeuromodConfig.from_dict(cfg)
        n = table.n
        self.oa_idx = select_oa_neurons(table)
        self.oa_mask = np.zeros(n, dtype=bool)
        self.oa_mask[self.oa_idx] = True
        ct = table.col("cell_type")
        self.dh44_idx = np.nonzero(ct == DH44_CELL_TYPE)[0]
        self.dh44_mask = np.zeros(n, dtype=bool)
        self.dh44_mask[self.dh44_idx] = True
        self._targets_for = None
        self.targets = np.zeros(0, dtype=np.int64)
        self.level = float(np.clip(self.cfg.initial_level, 0.0, 1.0))
        self._applied_shift = None
        # hit afferents (nociceptive path): one group per (set, side); the rate used is
        # the most active group's mean rate (code 0 = not a hit afferent)
        from .mapping import named_sets

        sets = named_sets(table)
        side = table.col("side")
        self.noci_code = np.zeros(n, dtype=np.int64)
        for si, name in enumerate(NOCI_SETS):
            idx = np.asarray(sets.get(name, []), dtype=np.int64)
            idx = idx[self.noci_code[idx] == 0]
            sd = side[idx]
            self.noci_code[idx] = 1 + 3 * si + np.where(sd == "left", 0,
                                                         np.where(sd == "right", 1, 2))
        self.noci_idx = np.nonzero(self.noci_code)[0]
        self._n_codes = 1 + 3 * len(NOCI_SETS)
        self.noci_n = np.bincount(self.noci_code[self.noci_idx],
                                  minlength=self._n_codes).astype(float)
        self._win_noci = 0.0
        self.runaway = False
        self._win_runaway = False
        self._win_oa = 0
        self._win_dh = 0
        self._win_s = 0.0
        self._last_u = 0.0
        self.configure(self.cfg)

    # ................................................................ config
    def configure(self, cfg: NeuromodConfig | dict) -> None:
        if not isinstance(cfg, NeuromodConfig):
            cfg = NeuromodConfig.from_dict(cfg)
        first = self._targets_for is None
        self.cfg = cfg
        if self._targets_for != cfg.target_min_syn:
            e = self.engine
            self.targets = oa_targets(e.indptr, e.indices, e.weights, self.oa_idx, self.table.n,
                                      cfg.target_min_syn, e.p.w_syn)
            self._targets_for = cfg.target_min_syn
            if self._applied_shift is not None:  # old targets: back to v_th
                self.engine.clear_threshold()
            self._applied_shift = None
        if first:
            self.level = float(np.clip(cfg.initial_level, 0.0, 1.0))
        self._apply()

    # ................................................................ dynamics
    def observe(self, spike_idx: np.ndarray, dt_s: float) -> None:
        if dt_s <= 0:
            return
        k = int(self.oa_mask[spike_idx].sum()) if len(spike_idx) else 0
        self._win_oa += k
        if len(self.dh44_idx) and len(spike_idx):
            self._win_dh += int(self.dh44_mask[spike_idx].sum())
        self._win_s += dt_s
        r_hit = self.hit_rate(spike_idx, dt_s)
        self._win_noci += r_hit * dt_s
        if not self.cfg.enabled:
            return
        c = self.cfg
        if dt_s >= 0.005:  # short chunks (cut at stimulus edges) keep the last verdict
            self.runaway = len(spike_idx) / dt_s > c.runaway_sps
        if self.runaway:
            self._win_runaway = True
            self.level = step_level(self.level, 0.0, dt_s, c.tau_rise_s, c.tau_decay_s)
            self._apply()
            return
        rate = k / (max(len(self.oa_idx), 1) * dt_s)
        u = rate / max(c.ref_rate_hz, 1e-9)
        if c.nociceptive_input:
            u += c.noci_gain * r_hit / max(c.noci_ref_hz, 1e-9)
        u = min(u, c.max_drive)
        self._last_u = u
        self.level = step_level(self.level, u, dt_s, self.cfg.tau_rise_s, self.cfg.tau_decay_s)
        self._apply()

    def relay_events(self, ev) -> list:
        """Extra StimulusEvents for the nociceptive relay (empty when off)."""
        c = self.cfg
        if not (c.enabled and c.noci_relay) or ev.kind not in HIT_KINDS:
            return []
        from .schema import StimulusEvent

        x = float(np.clip(ev.intensity, 0.0, 1.0))
        dur = max(float(ev.duration_s), 0.0) + c.relay_extra_s
        out = []
        for name, (a, b) in ((RELAY_WALK_SET, c.relay_walk_hz),
                             (RELAY_AROUSAL_SET, c.relay_arousal_hz)):
            rate = float(a) + float(b) * x
            if rate > 0:
                out.append(StimulusEvent("manual", "none", x, dur, ev.sim_time,
                                         details={"set": f"{name}", "rate_hz": rate,
                                                  "label": f"RELAY {name}"}))
        return out

    def hit_rate(self, spike_idx: np.ndarray, dt_s: float) -> float:
        """Mean spike rate (Hz) of the most active hit-afferent group (set x side)."""
        if not len(self.noci_idx) or not len(spike_idx) or dt_s <= 0:
            return 0.0
        c = np.bincount(self.noci_code[spike_idx], minlength=self._n_codes)
        n = self.noci_n[1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(n > 0, c[1:] / (np.maximum(n, 1.0) * dt_s), 0.0)
        return float(r.max())

    def on_reset(self) -> None:
        if not self.cfg.persist_on_reset:
            self.level = 0.0
            self._apply()

    def _apply(self) -> None:
        """Push the threshold shift into the engine (only when it changed)."""
        e = self.engine
        on = self.cfg.enabled and self.cfg.vth_shift_mv != 0.0 and len(self.targets) \
            and not self.runaway
        shift = float(self.cfg.vth_shift_mv * self.level) if on else 0.0
        if shift == 0.0:
            if self._applied_shift is not None:
                e.clear_threshold()
                self._applied_shift = None
            return
        if self._applied_shift is not None and abs(shift - self._applied_shift) < 1e-4:
            return
        vth = e.threshold_array()
        vth[self.targets] = e.p.v_th - shift
        self._applied_shift = shift

    # ................................................................ readout
    def readout(self) -> dict:
        w = max(self._win_s, 1e-9)
        out = {
            "enabled": bool(self.cfg.enabled),
            "octopamine": float(self.level),
            "label": MODEL_LABEL,
            "oa_rate_hz": float(self._win_oa / (max(len(self.oa_idx), 1) * w)),
            "oa_neurons": int(len(self.oa_idx)),
            "dh44_hz": float(self._win_dh / (max(len(self.dh44_idx), 1) * w)),
            "n_targets": int(len(self.targets)),
            "vth_shift_mv": float(self._applied_shift or 0.0),
            "noci_hz": float(self._win_noci / w),
            "nociceptive_input": bool(self.cfg.nociceptive_input),
            "noci_relay": bool(self.cfg.noci_relay),
            "runaway": bool(self._win_runaway),
            "tau_decay_s": float(self.cfg.tau_decay_s),
        }
        self._win_oa = self._win_dh = 0
        self._win_s = self._win_noci = 0.0
        self._win_runaway = False
        return out
