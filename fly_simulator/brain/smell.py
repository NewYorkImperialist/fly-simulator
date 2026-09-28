"""Smell without runaway: antennal-lobe model fixes (off by default) and odours as
ORN input. docs/SMELL.md has the diagnosis, the measurements and the limitations.

Diagnosis (docs/SMELL.md)
-------------------------
Driving the ORNs of a single glomerulus (even DM1 at 20 Hz) ignites the Shiu et al.
model's global self-sustaining state (~475k spikes/s) within ~20 ms. The first
recruited neurons after the glomerulus' own PNs are antennal-lobe local neurons
(ALLNs) that the model treats as **excitatory** (lLN2T, lLN2X, lLN1_bc, lLN2P_b...),
then the lateral horn and the KCs. The loop is the excitatory LN -> LN network:
~1100 excitatory LN -> LN synapses per excitatory LN, uniform weight 0.275 mV per
synapse. Silencing the excitatory LNs, or scaling their output by 0.1, removes the
runaway; scaling the GABA / Glu LN output up 8x does not; silencing AL-MBDL1 (the
hub of the KC-driven ignition, docs/FEAR_LEARNING.md) does not; silencing the KCs
or the mPNs does not. 166 of the 429 ALLNs are excitatory in the model:

* 40 are GABA/Glu in the v783 annotation's top_nt, but excitatory in the Shiu
  connectivity table (its per-connection signs predate / differ from v783's
  per-neuron predictions); 16 of them are *known* inhibitory from immunostaining
  (FlyWire ``known_nt``: lLN2P_b GABA, Sizemore et al. 2023; il3LN6 GABA, Tanaka
  et al. 2012; v2LN36 Glu, Chou et al. 2022).
* 52 are predicted dopaminergic / serotonergic (low confidence, 0.29 / 0.36). No
  AL local neuron is known to be monoaminergic (the AL's serotonergic neuron is
  CSD, not an ALLN); the lineage of these LNs (ALl1) makes GABAergic and
  cholinergic LNs. The model maps DA / 5HT to excitatory.
* the rest are cholinergic; lLN1_bc, lLN2X03, lLN2T_b and lLN2T_c are *known*
  cholinergic excitatory LNs (Shang et al. 2007).

Fixes (``SmellFixConfig.name``), each a documented model change
----------------------------------------------------------------
* ``none``: the unchanged model (bit-identical; the default).
* ``gaba``: scale the outputs of the inhibitory ALLNs (GABA / Glu in the model) by
  ``gaba_gain`` (presynaptic inhibition of ORNs and LN inhibition of PNs; Olsen &
  Wilson 2008). Does *not* stop the runaway (tested up to 8x).
* ``adapt``: spike-triggered threshold adaptation (``adapt_mv`` per spike, decay
  ``adapt_tau_s``) of every AL neuron (PNs, LNs, ALIN/ALON). PN spike-frequency
  adaptation is real (e.g. Bhandawat et al. 2007); the parameters are not fitted.
* ``eln``: scale the outputs of the model-excitatory ALLNs by ``eln_scale``
  (weakening the ignition hub; lateral excitation in the real AL is largely
  electrical, via eLN-PN gap junctions, Yaksi & Wilson 2010, which this model
  does not have).
* ``mbdl1``: silence AL-MBDL1 (the KC-driven route of docs/FEAR_LEARNING.md). Does
  not help for ORN input (reported for completeness).
* ``global``: pooled activity-dependent inhibition of the AL (**last resort, not a
  mechanism**): every AL spike raises the threshold of all AL neurons by
  ``global_mv``, decaying with ``global_tau_s``. Lowers the runaway rate but leaves
  a self-sustained state.
* ``sign``: sign correction of the AL local neurons: use the v783 annotation's
  predicted transmitter for every AL local neuron (not the Shiu table's), and map
  DA / 5HT-predicted ALLNs to inhibitory (GABA). Flips 101 LNs; removes the runaway
  on its own, but also turns 26 known cholinergic LNs inhibitory.
* ``sign_adapt`` (**recommended**): ``sign`` that respects ``known_nt`` (known
  cholinergic LNs stay excitatory, known GABA / Glu LNs become inhibitory; 71 LNs
  flipped) plus a milder AL adaptation (``sign_adapt_mv``). Neither part alone is
  enough with the known eLNs kept excitatory; together they are.

Weights are changed in place in ``LIFEngine.weights`` (the rows of the changed
presynaptic neurons, originals kept for ``restore``); adaptation uses
``LIFEngine.set_adaptation``; silencing is returned as ``silenced_idx`` for the
worker's lesion union. With ``name == "none"`` nothing is touched.

Odours
------
``ODORS`` maps an odour name to glomeruli and relative ORN drive. ``odor_drive``
turns per-antenna concentrations into per-ORN Poisson rates (see docs/SMELL.md for
the bilateral ORN projection and the sides used).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

MODEL_LABEL = "smell fix (model)"
AL_CLASSES = ("ALPN", "ALLN", "ALIN", "ALON")
# FlyWire v783 known_nt (immunostaining) for AL local neurons
KNOWN_ACH_LN_TYPES = ("lLN1_bc", "lLN2X03", "lLN2T_b", "lLN2T_c")   # Shang et al. 2007
KNOWN_INH_LN_TYPES = ("lLN2P_b", "il3LN6", "v2LN36")  # Sizemore 2023; Tanaka 2012; Chou 2022
# predicted-transmitter codes of data.NT_CODE
_GABA, _GLU, _DA, _5HT = 1, 2, 3, 4

SMELL_FIXES: dict[str, str] = {
    "none": "unchanged model",
    "gaba": "inhibitory AL LN outputs x gaba_gain",
    "adapt": "AL spike-frequency (threshold) adaptation",
    "eln": "excitatory AL LN outputs x eln_scale",
    "mbdl1": "silence AL-MBDL1",
    "global": "pooled AL activity-dependent inhibition (last resort)",
    "sign": "AL LN signs from v783 predictions, DA/5HT LNs -> GABA",
    "sign_adapt": "known-NT-respecting AL LN signs + mild AL adaptation",
}
RECOMMENDED = "sign_adapt"


@dataclass
class SmellFixConfig:
    name: str = "none"
    gaba_gain: float = 4.0
    adapt_mv: float = 10.0          # "adapt": threshold raise per spike (gap is 7 mV)
    adapt_tau_s: float = 0.2
    eln_scale: float = 0.1
    global_mv: float = 0.05
    global_tau_s: float = 0.05
    sign_adapt_mv: float = 4.0      # "sign_adapt": adaptation part
    hub_types: tuple = ("AL-MBDL1",)

    @classmethod
    def from_dict(cls, d: dict | str | None) -> "SmellFixConfig":
        if isinstance(d, str):
            d = {"name": d}
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        out = {k: v for k, v in d.items() if k in names}
        if isinstance(out.get("hub_types"), (list, str)):
            v = out["hub_types"]
            out["hub_types"] = tuple([v] if isinstance(v, str) else v)
        cfg = cls(**out)
        if cfg.name not in SMELL_FIXES:
            raise ValueError(f"unknown smell fix {cfg.name!r}; known: {sorted(SMELL_FIXES)}")
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------- neuron sets
def al_indices(table, classes=AL_CLASSES) -> np.ndarray:
    return np.nonzero(np.isin(table.col("cell_class"), list(classes)))[0].astype(np.int64)


def corrected_ln_sign(table, respect_known: bool = True, mono_inhibitory: bool = True
                      ) -> np.ndarray:
    """Model signs with the AL local neurons (ALLN) re-signed: v783 annotation
    transmitter (GABA / Glu inhibitory, others excitatory); DA / 5HT-predicted LNs
    inhibitory (``mono_inhibitory``); known_nt types override (``respect_known``)."""
    s = table.sign.astype(np.int64).copy()
    ct = table.col("cell_type")
    m = (table.col("cell_class") == "ALLN") & (table.sign != 0)
    nt = table.nt
    s[m] = np.where(np.isin(nt[m], [_GABA, _GLU]), -1, 1)
    if mono_inhibitory:
        s[m & np.isin(nt, [_DA, _5HT])] = -1
    if respect_known:
        s[m & np.isin(ct, KNOWN_ACH_LN_TYPES)] = 1
        s[m & np.isin(ct, KNOWN_INH_LN_TYPES)] = -1
    return s


def _rows(engine, pre: np.ndarray) -> np.ndarray:
    """CSR entry indices of all outgoing synapses of presynaptic neurons ``pre``."""
    ip = engine.indptr
    pre = np.asarray(pre, dtype=np.int64)
    if not len(pre):
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.arange(ip[j], ip[j + 1]) for j in pre]).astype(np.int64)


class SmellFix:
    """Applies one ``SmellFixConfig`` to an engine (lives next to it, like
    ``KCMBONPlasticity``). ``configure`` restores the previous fix first."""

    def __init__(self, table, engine, cfg: SmellFixConfig | dict | str | None = None):
        self.table = table
        self.engine = engine
        self.cfg = SmellFixConfig()
        self._p = np.zeros(0, dtype=np.int64)     # changed CSR entries
        self._w0 = np.zeros(0, dtype=np.float32)  # their original weights
        self.silenced_idx = np.zeros(0, dtype=np.int64)
        self.n_flipped = 0
        self.n_scaled = 0
        self.n_adapt = 0
        self.configure(cfg)

    # .................................................................... apply
    def restore(self) -> None:
        if len(self._p):
            self.engine.weights[self._p] = self._w0
        self._p = np.zeros(0, dtype=np.int64)
        self._w0 = np.zeros(0, dtype=np.float32)
        if self.n_adapt:
            self.engine.set_adaptation(None)
        self.silenced_idx = np.zeros(0, dtype=np.int64)
        self.n_flipped = self.n_scaled = self.n_adapt = 0

    def _scale_pre(self, pre: np.ndarray, factor: float) -> None:
        p = _rows(self.engine, pre)
        if not len(p):
            return
        self._p = np.concatenate([self._p, p])
        self._w0 = np.concatenate([self._w0, self.engine.weights[p].copy()])
        self.engine.weights[p] = (self.engine.weights[p].astype(np.float64) * factor).astype(
            np.float32)

    def configure(self, cfg: SmellFixConfig | dict | str | None) -> None:
        if not isinstance(cfg, SmellFixConfig):
            cfg = SmellFixConfig.from_dict(cfg)
        self.restore()
        self.cfg = c = cfg
        t, e = self.table, self.engine
        cc = t.col("cell_class")
        alln = (cc == "ALLN") & (t.sign != 0)
        al = al_indices(t)
        if c.name == "gaba":
            pre = np.nonzero(alln & (t.sign < 0))[0]
            self._scale_pre(pre, c.gaba_gain)
            self.n_scaled = len(pre)
        elif c.name == "eln":
            pre = np.nonzero(alln & (t.sign > 0))[0]
            self._scale_pre(pre, c.eln_scale)
            self.n_scaled = len(pre)
        elif c.name == "mbdl1":
            ct = t.col("cell_type")
            self.silenced_idx = np.nonzero(np.isin(ct, list(c.hub_types)))[0].astype(np.int64)
        elif c.name == "adapt":
            e.set_adaptation(al, c.adapt_mv, c.adapt_tau_s)
            self.n_adapt = len(al)
        elif c.name == "global":
            e.set_adaptation(al, 0.0, c.global_tau_s, c.global_mv)
            self.n_adapt = len(al)
        elif c.name in ("sign", "sign_adapt"):
            s = corrected_ln_sign(t, respect_known=(c.name == "sign_adapt"))
            flip = np.nonzero((s != t.sign) & (t.sign != 0))[0]
            self._scale_pre(flip, -1.0)
            self.n_flipped = len(flip)
            if c.name == "sign_adapt" and c.sign_adapt_mv > 0:
                e.set_adaptation(al, c.sign_adapt_mv, c.adapt_tau_s)
                self.n_adapt = len(al)

    @property
    def active(self) -> bool:
        return self.cfg.name != "none"

    def readout(self) -> dict:
        return {"name": self.cfg.name, "label": MODEL_LABEL,
                "description": SMELL_FIXES[self.cfg.name], "n_flipped": int(self.n_flipped),
                "n_scaled": int(self.n_scaled), "n_adapt": int(self.n_adapt),
                "n_silenced": int(len(self.silenced_idx))}


class SmellReadout:
    """Olfactory population rates per BrainState window (BrainState.smell)."""

    def __init__(self, table):
        cc, sub = table.col("cell_class"), table.col("cell_sub_class")
        sc, side = table.col("super_class"), table.col("side")
        reg = np.array(table.regions, dtype=object)[table.region]
        upn = (cc == "ALPN") & (sub == "uniglomerular")
        self.sets = {
            "ORN": np.nonzero((sc == "sensory") & (table.col("cell_class") == "olfactory"))[0],
            "uPN_L": np.nonzero(upn & (side == "left"))[0],
            "uPN_R": np.nonzero(upn & (side == "right"))[0],
            "ALLN": np.nonzero(cc == "ALLN")[0],
            "KC": np.nonzero(cc == "Kenyon_Cell")[0],
            "LH": np.nonzero((np.isin(cc, ["LHLN", "LHCENT"]) | np.isin(reg, ["LH_L", "LH_R"]))
                             & (cc != "ALPN"))[0],
        }

    def rates(self, counts: np.ndarray, window_s: float) -> dict:
        w = max(float(window_s), 1e-9)
        out = {k: (float(counts[i].mean() / w) if len(i) else 0.0) for k, i in self.sets.items()}
        kc = self.sets["KC"]
        frac = float((counts[kc] > 0).mean()) if len(kc) else 0.0
        return {"rates": out, "kc_active_frac": frac}


# ---------------------------------------------------------------------- odours
# odour -> {glomerulus: relative ORN drive (0..1)}. Coarse, literature-based labels:
ODORS: dict[str, dict[str, float]] = {
    # apple-cider vinegar: attraction via DM1 and VA2 (Semmelhack & Wang 2009 Nature),
    # with weaker DM2 / DM4 (Or42b/Or59b/Or22a-type food-ester glomeruli; Hallem &
    # Carlson 2006 ester responses)
    "vinegar": {"DM1": 1.0, "VA2": 1.0, "DM2": 0.5, "DM4": 0.5},
    # geosmin: DA2 (Or56a), aversive (Stensmyr et al. 2012 Cell)
    "geosmin": {"DA2": 1.0},
    # CO2: V glomerulus (Gr21a/Gr63a), aversive when walking (Suh et al. 2004)
    "co2": {"V": 1.0},
    # single glomeruli for tests / screens
    **{f"glom_{g}": {g: 1.0} for g in ("DM1", "DM2", "DM4", "VA2", "DL5", "DC1", "DA2", "V")},
}
VALENCE = {"vinegar": "attractive", "geosmin": "aversive", "co2": "aversive"}


def glomerulus_orns(table, glom: str, side: str | None = None) -> np.ndarray:
    ct, sc = table.col("cell_type"), table.col("super_class")
    m = (sc == "sensory") & (ct == f"ORN_{glom}")
    if side in ("left", "right"):
        m &= table.col("side") == side
    return np.nonzero(m)[0].astype(np.int64)


def orn_rate(conc: float, r_max: float = 100.0, k_half: float = 0.5, hill: float = 1.5
             ) -> float:
    """ORN rate (Hz) for a normalised concentration (0..1+), Hill-type saturation."""
    c = max(float(conc), 0.0)
    if c <= 0:
        return 0.0
    return float(r_max * c ** hill / (c ** hill + k_half ** hill))


def odor_drive_groups(table, odor: str | dict, conc_left: float, conc_right: float,
                      r_max: float = 100.0, k_half: float = 0.5, min_hz: float = 1.0
                      ) -> list[tuple[str, np.ndarray, float]]:
    """[(label, ORN indices, rate Hz)] per glomerulus and antenna for per-antenna
    concentrations. The ORNs of each antenna (FlyWire ``side`` = nerve-entry side)
    get that antenna's concentration; each ORN projects to both ALs as in the
    connectome, so any left / right PN asymmetry comes from the connectome's ipsi-
    vs contralateral ORN synapses (docs/SMELL.md)."""
    gl = ODORS[odor] if isinstance(odor, str) else dict(odor)
    name = odor if isinstance(odor, str) else "mix"
    out = []
    for g, wgt in gl.items():
        for side, c in (("left", conc_left), ("right", conc_right)):
            r = orn_rate(c, r_max, k_half) * float(wgt)
            if r < min_hz:
                continue
            i = glomerulus_orns(table, g, side)
            if len(i):
                out.append((f"odor:{name}:{g}:{side}", i, r))
    return out


def odor_drive(table, odor: str | dict, conc_left: float, conc_right: float,
               r_max: float = 100.0, k_half: float = 0.5, min_hz: float = 1.0
               ) -> tuple[np.ndarray, np.ndarray]:
    """(ORN indices, rates) of ``odor_drive_groups`` concatenated."""
    groups = odor_drive_groups(table, odor, conc_left, conc_right, r_max, k_half, min_hz)
    if not groups:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    return (np.concatenate([g[1] for g in groups]),
            np.concatenate([np.full(len(g[1]), g[2]) for g in groups]))
