"""Fear learning in the mushroom body: dopamine-gated depression of Kenyon cell ->
MBON synapses (phenomenological engine add-on, off by default).

Biology
-------
Aversive olfactory conditioning in Drosophila: an odour activates a sparse set of
Kenyon cells (KCs, ~5-10 %; Honegger et al. 2011; Turner et al. 2008); punishment
(electric shock) activates PPL1 dopaminergic neurons (DANs; Claridge-Chang et al.
2009; Aso et al. 2010, 2012); dopamine in a mushroom-body compartment causes
long-term depression of the synapses from the *co-active* KCs onto that
compartment's output neuron (MBON), e.g. KC -> MBON-gamma1pedc>a/b after pairing
with PPL1-gamma1pedc (Hige et al. 2015 Neuron; Cohn et al. 2015 Cell; Aso & Rubin
2016 eLife; Handler et al. 2019 Cell for timing). The depressed MBON responds less
to the trained odour; since the aversive-memory compartments' MBONs promote
approach, the balance of MBON output shifts towards avoidance (Aso et al. 2014
eLife; Owald et al. 2015). Connectome: Li et al. 2020 eLife (hemibrain), FlyWire
v783 (used here).

Model (what is modelled and what is connectome)
-----------------------------------------------
* **Compartments come from the connectome.** A DAN type is a *punishment DAN* if it
  is in ``dan_types`` (default: the PPL1 types that innervate KCs, PPL101-106) and
  makes >= ``min_kc_syn`` synapses onto KCs. Its compartment's MBONs are the MBON
  types it contacts with >= ``min_dan_mbon_syn`` synapses (DAN axons synapse onto
  both KCs and MBONs in their compartment). In FlyWire v783 this gives
  PPL101 -> MBON11 (gamma1pedc>a/b, 894 synapses), PPL103 -> MBON12 (gamma2a'1),
  MBON31/32/35, PPL105 -> MBON13 (a'2), MBON18 (a2sc), MBON23, PPL106 -> MBON14
  (a3), PPL104 -> MBON16/28 (a'3); see docs/FEAR_LEARNING.md.
* **Plastic synapses** = every KC -> MBON synapse row (CSR entry) onto a
  compartment MBON. Each gets an efficacy x in [x_min, 1]; the engine's weight of
  that entry is ``w0 * x`` (written into ``LIFEngine.weights`` in place, so the
  kernel is untouched; with plasticity off, or x == 1 everywhere, the weights are
  exactly the originals -> bit-identical, tested).
* **Rule** (evaluated after every engine run, i.e. every <= 20 ms of brain time)::

      KC eligibility   e_j <- e_j exp(-dt / tau_elig) + spikes_j
      KC activity      a_j  = min(e_j / e_sat, 1)
      compartment DA   D_c  = mean spike rate (Hz) of the compartment's DANs in the run
      LTD              x   <- x - lr * a_j * g(D_c) * (x - x_min) * dt
                              g(D) = clip((D - d_min) / (d_ref - d_min), 0, 1)
      forgetting       x   <- 1 - (1 - x) exp(-dt / tau_forget)   (tau_forget 0 = never)

  i.e. a KC's synapses in a compartment depress only when that KC fired recently
  (seconds; the eligibility window) *and* the compartment's DANs fire above a
  baseline (``d_min``). This is the Hebbian-like "coincidence of KC activity and
  dopamine -> LTD" of Hige et al. 2015; the parameters are free (not fitted).
  Not modelled: timing-dependent sign reversal (backward pairing -> potentiation,
  Handler et al. 2019), DAN-alone potentiation, reward (PAM) compartments
  (configurable via ``dan_types`` but untested), memory consolidation phases.
* **Odours.** Driving ORNs or PNs in this whole-brain LIF model ignites a
  self-sustaining runaway (docs/SENSORY_SCREEN.md; 27 uniglomerular PNs at 10 Hz
  already do, see docs/FEAR_LEARNING.md). An odour is therefore represented
  directly by its KC code: ``odor_kc_code`` ranks KCs by their synapses from the
  uniglomerular PNs of a chosen set of glomeruli (the connectome's PN -> KC claws)
  and takes the top ``frac`` (default 10 %), which the worker drives with Poisson
  input. That set is stable (no runaway, other KCs not recruited, APL feedback
  intact) and odour-specific.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

import numpy as np

MODEL_LABEL = "KC>MBON plasticity (model)"

# default odours: glomerulus sets (uniglomerular PN types "<glom>_<lineage>PN").
# Labels only; no claim that a real odorant activates exactly these glomeruli.
DEFAULT_ODORS = {
    "A": ("DM1", "DM2", "DM3", "DM4", "VA2", "DC2"),
    "B": ("DL1", "DL5", "VA5", "VM3", "DA3", "VC3"),
}
PUNISH_SET = "dan_punish"  # named stimulus set of the punishment DANs


@dataclass
class PlasticityConfig:
    enabled: bool = False
    dan_types: tuple = ("PPL101", "PPL102", "PPL103", "PPL104", "PPL105", "PPL106")
    min_kc_syn: int = 100          # a DAN type must innervate KCs this much
    min_dan_mbon_syn: int = 50     # DAN type -> MBON type synapses defining a compartment
    lr: float = 0.5                # /s at full KC activity and full dopamine
    x_min: float = 0.0             # floor of the efficacy
    tau_elig_s: float = 1.0        # KC eligibility trace
    e_sat: float = 5.0             # trace (spikes) at which a KC counts as fully active
    d_min_hz: float = 100.0        # DAN rate below which nothing is learned (above the
                                   # odour-evoked DAN rates of this model, see docs)
    d_ref_hz: float = 200.0        # DAN rate for full dopamine effect
    tau_forget_s: float = 0.0      # recovery of x to 1 (0 = no forgetting)
    odors: dict = field(default_factory=lambda: {k: list(v) for k, v in DEFAULT_ODORS.items()})
    odor_frac: float = 0.10        # fraction of KCs per odour
    odor_seed: int = 0
    odor_hz: float = 30.0          # KC Poisson rate of an odour stimulus
    punish_hz: float = 200.0       # punishment-DAN Poisson rate (whip stand-in)
    # neurons silenced while plasticity is on: AL-MBDL1 (2 neurons) carries the
    # KC-driven feedback into the antennal lobe that ignites the model's runaway
    # state; the odour input bypasses the AL, so it is cut (docs/FEAR_LEARNING.md)
    silence: tuple = ("AL-MBDL1",)
    runaway_sps: float = 100_000.0  # whole-brain spikes/s above which learning pauses
    persist_on_reset: bool = True  # a fly/brain reset keeps the memory

    @classmethod
    def from_dict(cls, d: dict | None) -> "PlasticityConfig":
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        out = {k: v for k, v in d.items() if k in names}
        for k in ("dan_types", "silence"):
            if isinstance(out.get(k), (list, str)):
                v = out[k]
                out[k] = tuple([v] if isinstance(v, str) else v)
        return cls(**out)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------- helpers
def kc_indices(table) -> np.ndarray:
    ct = table.col("cell_type").astype(str)
    return np.nonzero(np.char.startswith(ct, "KC"))[0].astype(np.int64)


def mbon_indices(table) -> np.ndarray:
    ct = table.col("cell_type").astype(str)
    return np.nonzero(np.char.startswith(ct, "MBON"))[0].astype(np.int64)


def _csr_edges(engine, rows: np.ndarray):
    """(pre, post, weight index p) of the CSR entries of presynaptic ``rows``."""
    ip = engine.indptr
    starts, ends = ip[rows], ip[rows + 1]
    lens = ends - starts
    pre = np.repeat(rows, lens)
    p = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)]) if len(rows) else \
        np.zeros(0, np.int64)
    return pre, engine.indices[p].astype(np.int64), p.astype(np.int64)


def glomerulus_pns(table, glomeruli) -> np.ndarray:
    """Uniglomerular PNs whose cell type is ``<glomerulus>_*PN``."""
    ct = table.col("cell_type").astype(str)
    cc = table.col("cell_class").astype(str)
    sub = table.col("cell_sub_class").astype(str)
    glom = np.array([c.split("_")[0] for c in ct])
    m = (cc == "ALPN") & (sub == "uniglomerular") & np.isin(glom, [str(g) for g in glomeruli])
    return np.nonzero(m)[0].astype(np.int64)


def odor_kc_code(table, engine, glomeruli, frac: float = 0.10, seed: int = 0) -> np.ndarray:
    """KC indices representing an odour: the top ``frac`` of KCs ranked by synapses
    from the uniglomerular PNs of ``glomeruli`` (ties broken at random). Falls back
    to a random KC subset when no PN matches (e.g. synthetic networks)."""
    kcs = kc_indices(table)
    k = max(1, int(round(frac * len(kcs)))) if len(kcs) else 0
    rng = np.random.default_rng(seed)
    if not k:
        return np.zeros(0, np.int64)
    pns = glomerulus_pns(table, glomeruli)
    score = np.zeros(table.n)
    if len(pns):
        _, post, p = _csr_edges(engine, pns)
        w = engine.weights[p].astype(np.float64)
        np.add.at(score, post, np.maximum(w, 0.0))
    s = score[kcs] + rng.random(len(kcs)) * 1e-6  # tie-break
    if not np.any(score[kcs] > 0):
        return np.sort(rng.choice(kcs, k, replace=False))
    return np.sort(kcs[np.argsort(-s)[:k]])


def compartments(table, engine, cfg: PlasticityConfig) -> dict[str, dict]:
    """{DAN type: {"dan": idx, "mbon_types": [...], "mbon": idx, "kc_syn": n,
    "mbon_syn": {type: n}}} from the connectome (see module doc)."""
    ct = table.col("cell_type").astype(str)
    kc_mask = np.zeros(table.n, bool)
    kc_mask[kc_indices(table)] = True
    mb = mbon_indices(table)
    mb_mask = np.zeros(table.n, bool)
    mb_mask[mb] = True
    out: dict[str, dict] = {}
    for d in cfg.dan_types:
        dan = np.nonzero(ct == str(d))[0].astype(np.int64)
        if not len(dan):
            continue
        _, post, p = _csr_edges(engine, dan)
        w = np.abs(engine.weights[p].astype(np.float64)) / engine.p.w_syn
        kc_syn = float(w[kc_mask[post]].sum())
        if kc_syn < cfg.min_kc_syn:
            continue
        syn: dict[str, float] = {}
        for q, ww in zip(post[mb_mask[post]], w[mb_mask[post]]):
            syn[ct[q]] = syn.get(ct[q], 0.0) + float(ww)
        types = sorted([t for t, v in syn.items() if v >= cfg.min_dan_mbon_syn],
                       key=lambda t: -syn[t])
        if not types:
            continue
        out[str(d)] = {"dan": dan, "mbon_types": types, "kc_syn": int(round(kc_syn)),
                       "mbon_syn": {t: int(round(syn[t])) for t in types}}
    # an MBON type contacted by several DAN types belongs to the one with the most
    # synapses onto it (e.g. MBON11: PPL101 894 vs PPL102 51)
    best: dict[str, tuple[float, str]] = {}
    for d, c in out.items():
        for t, v in c["mbon_syn"].items():
            if t not in best or v > best[t][0]:
                best[t] = (v, d)
    for d in list(out):
        c = out[d]
        c["mbon_types"] = [t for t in c["mbon_types"] if best[t][1] == d]
        if not c["mbon_types"]:
            del out[d]
            continue
        c["mbon_syn"] = {t: c["mbon_syn"][t] for t in c["mbon_types"]}
        c["mbon"] = np.nonzero(np.isin(ct, c["mbon_types"]))[0].astype(np.int64)
    return out


class KCMBONPlasticity:
    """Lives next to the engine (``process._Model``); ``observe`` after each run."""

    def __init__(self, table, engine, cfg: PlasticityConfig | dict | None = None):
        self.table = table
        self.engine = engine
        self.cfg = PlasticityConfig()
        self.kc = kc_indices(table)
        self._kc_pos = np.full(table.n, -1, np.int64)
        self._kc_pos[self.kc] = np.arange(len(self.kc))
        self.elig = np.zeros(len(self.kc))
        self.comp: dict[str, dict] = {}
        self.p = np.zeros(0, np.int64)       # CSR entries (plastic synapses)
        self.w0 = np.zeros(0, np.float32)    # their original weights
        self.x = np.zeros(0)                 # efficacies
        self.syn_kc = np.zeros(0, np.int64)  # KC position of each synapse
        self.syn_comp = np.zeros(0, np.int64)
        self.syn_post = np.zeros(0, np.int64)
        self.comp_names: list[str] = []
        self.odors: dict[str, np.ndarray] = {}
        self.dan_rate: dict[str, float] = {}
        self.n_updates = 0
        self.t_learned = 0.0  # brain seconds with dopamine above d_min
        self.runaway = False  # last observed run was above runaway_sps (learning paused)
        self.n_runaway = 0
        self.configure(cfg if cfg is not None else PlasticityConfig(enabled=True))

    # ------------------------------------------------------------------ setup
    def configure(self, cfg: PlasticityConfig | dict | None) -> None:
        if not isinstance(cfg, PlasticityConfig):
            cfg = PlasticityConfig.from_dict(cfg)
        old_sets = (self.cfg.dan_types, self.cfg.min_kc_syn, self.cfg.min_dan_mbon_syn)
        new_sets = (cfg.dan_types, cfg.min_kc_syn, cfg.min_dan_mbon_syn)
        rebuild = not len(self.p) or old_sets != new_sets
        self.cfg = cfg
        if rebuild:
            self.restore()
            self._build()
        self.odors = {str(k): odor_kc_code(self.table, self.engine, v, cfg.odor_frac,
                                           cfg.odor_seed + i)
                      for i, (k, v) in enumerate(cfg.odors.items())}
        if not cfg.enabled:
            self.restore()  # engine weights back to the originals (exact)
        else:
            self._write()

    def _build(self) -> None:
        self.comp = compartments(self.table, self.engine, self.cfg)
        self.comp_names = list(self.comp)
        post_comp = np.full(self.table.n, -1, np.int64)
        for c, name in enumerate(self.comp_names):
            post_comp[self.comp[name]["mbon"]] = c  # disjoint (see compartments())
        pre, post, p = _csr_edges(self.engine, self.kc)
        keep = post_comp[post] >= 0
        keep &= self.engine.weights[p] > 0
        self.p = p[keep]
        self.syn_kc = self._kc_pos[pre[keep]]
        self.syn_post = post[keep]
        self.syn_comp = post_comp[post[keep]]
        self.w0 = self.engine.weights[self.p].copy()
        self.x = np.ones(len(self.p))
        self.elig = np.zeros(len(self.kc))
        self.dan_rate = {n: 0.0 for n in self.comp_names}

    def restore(self) -> None:
        """Write the original weights back (does not reset the efficacies)."""
        if len(self.p):
            self.engine.weights[self.p] = self.w0

    def _write(self) -> None:
        if len(self.p):
            self.engine.weights[self.p] = (self.w0.astype(np.float64) * self.x).astype(np.float32)

    def reset_memory(self) -> None:
        self.x[:] = 1.0
        self.elig[:] = 0.0
        self.t_learned = 0.0
        if self.cfg.enabled:
            self._write()

    @property
    def silenced_idx(self) -> np.ndarray:
        """Neurons to silence while the add-on exists (``cfg.silence`` cell types;
        also when learning is disabled but the odour sets are in use)."""
        if not self.cfg.silence:
            return np.zeros(0, np.int64)
        ct = self.table.col("cell_type").astype(str)
        return np.nonzero(np.isin(ct, [str(x) for x in self.cfg.silence]))[0].astype(np.int64)

    @property
    def dan_idx(self) -> np.ndarray:
        if not self.comp:
            return np.zeros(0, np.int64)
        return np.unique(np.concatenate([c["dan"] for c in self.comp.values()]))

    # ------------------------------------------------------------------ dynamics
    def observe(self, spike_idx: np.ndarray, dt_s: float) -> bool:
        """Update traces and apply the rule for one engine run of ``dt_s``; True if
        any weight changed."""
        c = self.cfg
        if not c.enabled or not len(self.p) or dt_s <= 0:
            return False
        spike_idx = np.asarray(spike_idx, dtype=np.int64)
        self.runaway = len(spike_idx) / dt_s > c.runaway_sps
        if self.runaway:  # the model's artefactual runaway state: learn nothing
            self.n_runaway += 1
            self.elig[:] = 0.0
            return False
        self.elig *= np.exp(-dt_s / max(c.tau_elig_s, 1e-6))
        if len(spike_idx):
            pos = self._kc_pos[spike_idx]
            pos = pos[pos >= 0]
            if len(pos):
                self.elig += np.bincount(pos, minlength=len(self.kc))
        counts = np.bincount(spike_idx, minlength=self.table.n) if len(spike_idx) else None
        span = max(c.d_ref_hz - c.d_min_hz, 1e-6)
        gains = np.zeros(len(self.comp_names))
        for k, name in enumerate(self.comp_names):
            dan = self.comp[name]["dan"]
            r = float(counts[dan].sum() / (len(dan) * dt_s)) if counts is not None else 0.0
            self.dan_rate[name] = r
            gains[k] = np.clip((r - c.d_min_hz) / span, 0.0, 1.0)
        changed = False
        if np.any(gains > 0):
            a = np.minimum(self.elig / max(c.e_sat, 1e-9), 1.0)
            drive = c.lr * dt_s * a[self.syn_kc] * gains[self.syn_comp]
            sel = drive > 0
            if np.any(sel):
                self.x[sel] -= np.minimum(drive[sel], 1.0) * (self.x[sel] - c.x_min)
                changed = True
                self.n_updates += 1
            self.t_learned += dt_s
        if c.tau_forget_s > 0:
            below = self.x < 1.0
            if np.any(below):
                self.x[below] = 1.0 - (1.0 - self.x[below]) * np.exp(-dt_s / c.tau_forget_s)
                changed = True
        if changed:
            self._write()
        return changed

    def on_reset(self) -> None:
        self.elig[:] = 0.0
        if not self.cfg.persist_on_reset:
            self.reset_memory()

    # ------------------------------------------------------------------ readout
    def efficacy(self, kc_idx=None) -> dict[str, float]:
        """Mean efficacy per compartment over the synapses of KCs ``kc_idx`` (model
        indices; None = all KCs), weighted by original synapse weight."""
        out = {}
        if kc_idx is None:
            sel = np.ones(len(self.p), bool)
        else:
            m = np.zeros(len(self.kc), bool)
            pos = self._kc_pos[np.asarray(kc_idx, dtype=np.int64)]
            m[pos[pos >= 0]] = True
            sel = m[self.syn_kc]
        w = self.w0.astype(np.float64)
        for k, name in enumerate(self.comp_names):
            s = sel & (self.syn_comp == k)
            out[name] = float((w[s] * self.x[s]).sum() / w[s].sum()) if np.any(s) else 1.0
        return out

    def compartment_label(self, name: str) -> str:
        c = self.comp.get(name)
        return f"{name}>{'/'.join(c['mbon_types'][:2])}" if c else name

    def readout(self) -> dict:
        on = bool(self.cfg.enabled and len(self.p))
        by_odor = {k: self.efficacy(v) for k, v in self.odors.items()}
        main = self.comp_names[0] if self.comp_names else ""
        return {"enabled": on, "label": MODEL_LABEL,
                "compartments": {n: list(self.comp[n]["mbon_types"]) for n in self.comp_names},
                "efficacy_by_odor": by_odor,  # {odour: {DAN compartment: mean x}}
                "main": main,  # the gamma1pedc compartment (PPL101) when present
                "dan_hz": {k: float(v) for k, v in self.dan_rate.items()},
                "n_synapses": int(len(self.p)), "n_updates": int(self.n_updates),
                "t_learned_s": float(self.t_learned),
                "runaway": bool(self.runaway), "n_runaway": int(self.n_runaway),
                "odor_kcs": {k: int(len(v)) for k, v in self.odors.items()}}
