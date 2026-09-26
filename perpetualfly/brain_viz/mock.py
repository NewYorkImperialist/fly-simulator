"""Mock brain: a plausible ``BrainLayout`` + ``BrainState`` stream without the real model.

Not a neural simulation -- a small phenomenological generator so the window can be
built, tuned and tested without the LIF engine:

* regions are the 78 real FlyWire neuropils from the atlas (``*_L``/``*_R`` + midline
  CX/GNG/...), positioned at their projected centroids (micrometres);
* ~2000 display neurons are scattered inside their neuropil silhouettes, each with a
  predicted-transmitter label drawn from a region-specific mix (mostly ACh; GABA/Glu
  everywhere; DA around the mushroom body; OA/5-HT sparse);
* background: each region has a baseline rate with slow Ornstein-Uhlenbeck drift, the
  optic lobes run hotter (visual flow while walking), the central complex bumps
  rhythmically, descending "walk" neurons fire tonically;
* stimuli (``StimulusEvent``) launch a delayed ripple along a plausible pathway, e.g. a
  left whip: AMMC_L/SAD_L (mechanosensory) -> GNG, WED_L -> IPS/SPS/VES_L (DN
  dendrites) -> LAL -> CX, with the giant fibre (escape) for strong hits, a turn away
  from the hit and an octopamine/dopamine "arousal" wave.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from perpetualfly.brain.schema import (
    DESCENDING_GROUPS,
    NEUROTRANSMITTERS,
    BrainLayout,
    BrainState,
    StimulusEvent,
)
from perpetualfly.brain_viz.atlas import load_atlas, region_base, region_group, region_side

N_NEURONS_FLYWIRE = 139_255  # FlyWire v783 proofread neurons (Shiu et al. 2024 model)

# Transmitter mix (ACh, GABA, Glu, DA, 5HT, OA) by super-region.
_NT_MIX = {
    "OL": (0.52, 0.20, 0.26, 0.005, 0.01, 0.005),
    "MB": (0.80, 0.03, 0.02, 0.13, 0.01, 0.01),
    "AL": (0.62, 0.30, 0.06, 0.00, 0.01, 0.01),
    "GNG": (0.55, 0.20, 0.15, 0.01, 0.03, 0.06),
    "CX": (0.60, 0.15, 0.20, 0.03, 0.01, 0.01),
    "SNP": (0.55, 0.14, 0.14, 0.12, 0.03, 0.02),
    "INP": (0.60, 0.20, 0.16, 0.02, 0.01, 0.01),
    "VMNP": (0.58, 0.20, 0.16, 0.01, 0.02, 0.03),
}
_NT_MIX_DEFAULT = (0.62, 0.18, 0.16, 0.02, 0.01, 0.01)

# Baseline rate (Hz, mean over neurons) by super-region while the fly walks.
_BASE_RATE = {"OL": 5.0, "MB": 0.6, "AL": 2.5, "LH": 1.5, "CX": 3.0, "LX": 2.5,
              "SNP": 1.2, "VLNP": 2.0, "INP": 1.2, "VMNP": 2.5, "PENP": 1.5,
              "GNG": 3.0, "OCG": 1.0}

_LABELS = {"OL": ["T4", "T5", "Mi1", "Tm3", "LC4", "LPLC2", "Dm8", "L1"],
           "MB": ["KCg", "KCab", "KCapbp", "MBON", "PAM", "PPL1", "APL"],
           "AL": ["ORN", "PN", "LN", "vPN"], "LH": ["LHN", "LHLN", "LHON"],
           "CX": ["EPG", "PEN", "PFL3", "hDelta", "ER", "FC"], "LX": ["LAL", "PFL"],
           "GNG": ["BM", "DNg", "GNG", "AN"], "PENP": ["JO-A", "JO-B", "JO-CE", "AMMC"],
           "VMNP": ["DNp", "DNa", "IPS", "SPS"]}

# Descending groups: (label, region) of the display neurons that report them.
_DN_NEURONS = {"walk_L": ("DNg100_L", "GNG"), "walk_R": ("DNg100_R", "GNG"),
               "turn_L": ("DNa02_L", "LAL_L"), "turn_R": ("DNa02_R", "LAL_R"),
               "backward_L": ("MDN_L", "GNG"), "backward_R": ("MDN_R", "GNG"),
               "escape": ("GF", "GNG"), "groom": ("DNg12", "GNG")}
_DN_BASE = {"walk_L": 22.0, "walk_R": 22.0, "turn_L": 5.0, "turn_R": 5.0,
            "backward_L": 0.5, "backward_R": 0.5, "escape": 0.0,
            "groom": 1.0}


def _sample_in_polygons(rng, polys: list[np.ndarray], centroid: np.ndarray, n: int,
                        spread: float = 12.0) -> np.ndarray:
    """``n`` random points inside the union of ``polys`` (um); centroid blob fallback."""
    if not polys:
        return centroid + rng.normal(0, spread, (n, 2))
    allp = np.concatenate(polys)
    lo, hi = allp.min(0), allp.max(0)
    scale = 1.0  # rasterise at 1 um / px
    w, h = int(hi[0] - lo[0]) + 2, int(hi[1] - lo[1]) + 2
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.round((p - lo) / scale).astype(np.int32) for p in polys], 255)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return centroid + rng.normal(0, spread, (n, 2))
    k = rng.integers(0, len(xs), n)
    return np.stack([xs[k], ys[k]], 1) * scale + lo + rng.uniform(0, 1, (n, 2))


class MockBrain:
    """Phenomenological brain-activity generator speaking the schema.

    ``advance(dt)`` simulates ``dt`` seconds of brain time at 1 ms resolution and
    returns the ``BrainState`` for that window; ``stimulate(ev)`` injects a stimulus.
    """

    def __init__(self, n_display: int = 2000, seed: int = 0,
                 realtime_factor: float = 0.25, with_outlines: bool = False,
                 dt: float = 1e-3) -> None:
        self.rng = np.random.default_rng(seed)
        self.dt = dt
        self.realtime_factor = realtime_factor
        self.brain_time = 0.0
        self._pulses: list[tuple[int, float, float, float]] = []   # region, t0, amp, tau
        self._dn_pulses: list[tuple[int, float, float, float]] = []
        self._arousal_pulses: list[tuple[float, float, float]] = []
        self._pending_stim_labels: list[str] = []
        self.layout = self._make_layout(n_display, with_outlines)
        L = self.layout
        R = len(L.regions)
        self._groups = [region_group(n) for n in L.regions]
        self._base = np.array([_BASE_RATE.get(g, 1.5) for g in self._groups])
        self._ou = np.zeros(R)
        self._dn_ou = np.zeros(len(DESCENDING_GROUPS))
        self._cx = np.array([g == "CX" for g in self._groups])
        self._idx = {n: i for i, n in enumerate(L.regions)}
        # per-neuron gain: lognormal, ~25% near-silent
        n = len(L.display_neuron_ids)
        g = self.rng.lognormal(0.0, 0.9, n)
        g[self.rng.random(n) < 0.25] *= 0.05
        self._gain = g / g.mean()
        nt = L.display_neuron_nt
        self._is_da = nt == NEUROTRANSMITTERS.index("DA")
        self._is_oa = nt == NEUROTRANSMITTERS.index("OA")
        self._is_5ht = nt == NEUROTRANSMITTERS.index("5HT")
        self._dn_of_neuron = np.full(n, -1)
        for gi, grp in enumerate(DESCENDING_GROUPS):
            lab = _DN_NEURONS[grp][0]
            for i, s in enumerate(L.display_neuron_label):
                if s == lab:
                    self._dn_of_neuron[i] = gi

    # ------------------------------------------------------------------ layout
    def _make_layout(self, n_display: int, with_outlines: bool) -> BrainLayout:
        atlas = load_atlas()
        if atlas is None:
            raise RuntimeError("neuropil atlas asset missing")
        rng = self.rng
        names = sorted(atlas["regions"], key=lambda n: (region_group(n), region_base(n),
                                                        region_side(n)))
        regs = [atlas["regions"][n] for n in names]
        region_xy = np.array([r["centroid"] for r in regs], np.float32)
        # neurons per region ~ sqrt(area), at least 6
        w = np.sqrt(np.array([max(r["area_um2"], 100.0) for r in regs]))
        dn_count = len(DESCENDING_GROUPS)
        counts = np.maximum(6, np.floor(w / w.sum() * (n_display - dn_count))).astype(int)
        while counts.sum() > n_display - dn_count:
            counts[np.argmax(counts)] -= 1
        while counts.sum() < n_display - dn_count:
            counts[rng.integers(len(counts))] += 1
        nr, nnt, nxy, nlab = [], [], [], []
        for ri, (name, r) in enumerate(zip(names, regs)):
            k = counts[ri]
            grp = region_group(name)
            mix = np.array(_NT_MIX.get(grp, _NT_MIX_DEFAULT))
            nt = rng.choice(len(NEUROTRANSMITTERS), k, p=mix / mix.sum())
            nt[rng.random(k) < 0.03] = -1  # a few unknown predictions
            nr.append(np.full(k, ri))
            nnt.append(nt)
            nxy.append(_sample_in_polygons(rng, r["outline"], r["centroid"], k))
            labs = _LABELS.get(grp, [region_base(name)])
            nlab += [labs[j % len(labs)] for j in rng.integers(0, 1000, k)]
        for grp in DESCENDING_GROUPS:
            lab, reg = _DN_NEURONS[grp]
            ri = names.index(reg)
            nr.append(np.array([ri]))
            nnt.append(np.array([NEUROTRANSMITTERS.index("ACh")]))
            nxy.append(regs[ri]["centroid"][None] + rng.normal(0, 6, (1, 2)))
            nlab.append(lab)
        n = sum(len(a) for a in nr)
        outlines: list[np.ndarray] = []
        if with_outlines:  # one polyline per region, same order as regions
            outlines = [r["outline"][0].astype(np.float32) if r["outline"] else
                        np.zeros((0, 2), np.float32) for r in regs]
        return BrainLayout(
            regions=list(names),
            region_xy=region_xy,
            region_outline_xy=outlines,
            display_neuron_ids=np.asarray(720575940600000000 + rng.choice(10**9, n, replace=False),
                                          np.int64),
            display_neuron_region=np.concatenate(nr).astype(np.int32),
            display_neuron_nt=np.concatenate(nnt).astype(np.int8),
            display_neuron_xy=np.concatenate(nxy).astype(np.float32),
            display_neuron_label=nlab,
            n_neurons_total=N_NEURONS_FLYWIRE,
            model_name="MOCK Shiu2024-LIF FlyWire v783",
        )

    # ------------------------------------------------------------------ stimuli
    def stimulate(self, ev: StimulusEvent) -> None:
        t0 = self.brain_time
        gain = float(np.clip(ev.intensity, 0.0, 1.0))
        side = {"left": "L", "right": "R"}.get(ev.side, "")
        opp = {"L": "R", "R": "L"}.get(side, "")
        both = [side] if side else ["L", "R"]
        add = self._pulses.append

        def reg(base: str, delay_ms: float, amp: float, tau: float = 0.06,
                sides: list[str] | None = None) -> None:
            for s in (sides if sides is not None else both):
                name = f"{base}_{s}" if s else base
                if name in self._idx:
                    add((self._idx[name], t0 + delay_ms / 1000, amp, tau))
                elif base in self._idx:
                    add((self._idx[base], t0 + delay_ms / 1000, amp, tau))

        def dn(group: str, delay_ms: float, amp: float, tau: float = 0.08) -> None:
            self._dn_pulses.append((DESCENDING_GROUPS.index(group), t0 + delay_ms / 1000,
                                    amp, tau))

        kind = ev.kind
        if kind in ("whip_hit", "shove", "manual"):
            k = 1.0 if kind == "whip_hit" else 0.7
            reg("AMMC", 0, 90 * gain * k, 0.04)
            reg("SAD", 2, 70 * gain * k, 0.05)
            reg("GNG", 6, 55 * gain * k, 0.08)
            reg("WED", 8, 45 * gain * k)
            reg("AVLP", 14, 30 * gain * k)
            reg("PVLP", 16, 22 * gain * k)
            for b in ("IPS", "SPS", "VES", "EPA", "GOR"):
                reg(b, 14, 38 * gain * k, 0.07)
            reg("LAL", 22, 36 * gain * k, 0.09)
            if opp:
                reg("LAL", 26, 18 * gain * k, 0.09, sides=[opp])
                reg("IPS", 20, 16 * gain * k, 0.08, sides=[opp])
            reg("CRE", 26, 18 * gain * k)
            for b, a in (("EB", 30), ("PB", 24), ("FB", 26), ("NO", 20)):
                reg(b, 30, a * gain * k, 0.15)
            reg("SMP", 40, 16 * gain * k, 0.2, sides=["L", "R"])
            for b in ("MB_ML", "MB_VL"):
                reg(b, 45, 12 * gain * k, 0.25, sides=["L", "R"])
            reg("LH", 35, 10 * gain * k, 0.15)
            if gain > 0.35:
                dn("escape", 5, 260 * (gain - 0.3), 0.03)
            if side:
                dn(f"turn_{opp}", 22, 70 * gain, 0.18)      # steer away from the hit
                dn(f"turn_{side}", 22, -4 * gain, 0.18)
            if ev.side == "front":
                dn("backward_L", 18, 90 * gain, 0.3)
                dn("backward_R", 18, 90 * gain, 0.3)
                dn("walk_L", 18, -18 * gain, 0.3)
                dn("walk_R", 18, -18 * gain, 0.3)
            else:
                dn("walk_L", 30, 40 * gain, 0.4)
                dn("walk_R", 30, 40 * gain, 0.4)
            self._arousal_pulses.append((t0 + 0.04, 1.2 * gain, 1.5))
        elif kind == "fall":
            for i in range(len(self.layout.regions)):
                self._pulses.append((i, t0 + self.rng.uniform(0, 0.05), 12 * gain, 0.3))
            dn("escape", 10, 80 * gain, 0.05)
            self._arousal_pulses.append((t0 + 0.05, 1.0 * gain, 2.0))
        elif kind == "ground_contact":
            reg("AMMC", 0, 15 * gain, 0.03, sides=["L", "R"])
            reg("GNG", 5, 10 * gain, 0.05)
        else:  # reset / unknown: brief GNG blip
            reg("GNG", 0, 8 * gain, 0.05)
        short = ev.side if ev.side not in ("none", "") else ""
        self._pending_stim_labels.append(f"{kind} {short} {gain:.2f}".replace("  ", " "))

    # ------------------------------------------------------------------ dynamics
    @staticmethod
    def _alpha(t: np.ndarray, t0: float, tau: float) -> np.ndarray:
        """Normalised alpha kernel (peak 1 at t0 + tau)."""
        s = (t - t0) / tau
        return np.where(s > 0, s * np.exp(1 - s), 0.0)

    def advance(self, duration: float) -> BrainState:
        """Simulate ``duration`` s of brain time; return the state for that window."""
        L = self.layout
        rng = self.rng
        steps = max(1, int(round(duration / self.dt)))
        t = self.brain_time + self.dt * np.arange(1, steps + 1)
        R, D = len(L.regions), len(DESCENDING_GROUPS)
        # OU drift (tau 0.8 s) sampled at the window resolution then held.
        a = np.exp(-duration / 0.8)
        self._ou = a * self._ou + np.sqrt(1 - a * a) * rng.normal(0, 0.35, R)
        self._dn_ou = a * self._dn_ou + np.sqrt(1 - a * a) * rng.normal(0, 0.3, D)
        rate = np.tile(self._base * np.clip(1 + self._ou, 0.2, 3.0), (steps, 1))
        rate[:, self._cx] *= (1 + 0.6 * np.sin(2 * np.pi * 1.3 * t))[:, None] ** 2
        for ri, t0, amp, tau in self._pulses:
            rate[:, ri] += amp * self._alpha(t, t0, tau)
        dn_rate = np.tile(np.array([_DN_BASE[g] for g in DESCENDING_GROUPS]) *
                          np.clip(1 + self._dn_ou, 0.1, 3.0), (steps, 1))
        for gi, t0, amp, tau in self._dn_pulses:
            dn_rate[:, gi] += amp * self._alpha(t, t0, tau)
        dn_rate = np.maximum(dn_rate, 0.0)
        arousal = np.zeros(steps)
        for t0, amp, tau in self._arousal_pulses:
            s = t - t0
            arousal += np.where(s > 0, amp * (1 - np.exp(-s / 0.1)) * np.exp(-s / tau), 0)
        rate = np.maximum(rate, 0.0)
        # per-neuron rates (steps, n)
        nrate = rate[:, L.display_neuron_region] * self._gain
        nrate[:, self._is_oa] += 25 * arousal[:, None]
        nrate[:, self._is_da] += 14 * arousal[:, None]
        nrate[:, self._is_5ht] += 6 * arousal[:, None]
        dn_mask = self._dn_of_neuron >= 0
        nrate[:, dn_mask] = dn_rate[:, self._dn_of_neuron[dn_mask]]
        spikes = rng.random(nrate.shape) < np.clip(nrate * self.dt, 0, 0.5)
        si, ni = np.nonzero(spikes)
        # summaries
        nt = L.display_neuron_nt
        counts = spikes.sum(0)
        rate_by_nt = np.zeros(len(NEUROTRANSMITTERS), np.float32)
        frac_by_nt = np.zeros(len(NEUROTRANSMITTERS), np.float32)
        for k in range(len(NEUROTRANSMITTERS)):
            m = nt == k
            if m.any():
                rate_by_nt[k] = nrate[:, m].mean()
                frac_by_nt[k] = (counts[m] > 0).mean()
        self.brain_time = float(t[-1])
        # drop finished pulses
        tb = self.brain_time
        self._pulses = [p for p in self._pulses if tb - p[1] < 12 * p[3]]
        self._dn_pulses = [p for p in self._dn_pulses if tb - p[1] < 12 * p[3]]
        self._arousal_pulses = [p for p in self._arousal_pulses if tb - p[0] < 8 * p[2]]
        stim, self._pending_stim_labels = self._pending_stim_labels, []
        total = int(nrate.mean() * duration * L.n_neurons_total * 0.35)
        return BrainState(
            brain_time=self.brain_time,
            wall_time=time.time(),
            realtime_factor=float(self.realtime_factor),
            window_s=float(duration),
            rate_by_nt=rate_by_nt,
            active_frac_by_nt=frac_by_nt,
            rate_by_region=rate.mean(0).astype(np.float32),
            descending={g: float(dn_rate[:, i].mean()) for i, g in
                        enumerate(DESCENDING_GROUPS)},
            raster_idx=ni.astype(np.int32),
            raster_t=t[si].astype(np.float32),
            total_spikes=total,
            recent_stimuli=stim,
        )


def mock_scenario(kind: str = "quiet", seed: int = 0, n_display: int = 2000,
                  window_s: float = 0.25) -> tuple[BrainLayout, list[BrainState]]:
    """Canned sequences for tests/screenshots: 'quiet', 'whip' (left whip), 'high'."""
    mb = MockBrain(n_display=n_display, seed=seed)
    states = [mb.advance(window_s) for _ in range(8)]
    if kind == "whip":
        mb.stimulate(StimulusEvent("whip_hit", "left", 0.6))
        states += [mb.advance(0.05) for _ in range(2)]
    elif kind == "high":
        for i in range(3):
            mb.stimulate(StimulusEvent("whip_hit", ("left", "right")[i % 2], 1.0))
            states.append(mb.advance(0.04))
        mb.stimulate(StimulusEvent("fall", "none", 1.0))
        states += [mb.advance(0.05) for _ in range(2)]
    elif kind != "quiet":
        raise ValueError(kind)
    return mb.layout, states
