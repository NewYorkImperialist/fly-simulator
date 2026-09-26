"""Body events -> sensory Poisson input, and descending neurons -> walking drive.

Everything here is chosen from FlyWire v783 annotations (Schlegel et al. 2024,
flywire_annotations v3.1.0); see docs/BRAIN.md for the reasoning and citations.
Sides are always the *fly's* side as annotated in FlyWire (``side`` column: soma
side, or nerve-entry side for sensory neurons). Note that Shiu et al. 2024 call the
sugar GRNs below "right hemisphere"; FlyWire's current annotation puts them on the
fly's left (older FlyWire views were mirrored).

Stimulus mapping (``StimulusMapper``)
------------------------------------
* ``whip_hit`` / ``shove`` on the body (thorax, abdomen, legs, wings; default):
  sensory-ascending mechanosensory afferents on the hit side
  (super_class ``sensory_ascending``, sub-classes ``SA_DMT_*``, ``SA_DLV``,
  ``SA_MDA``, ``SA_VTV_DProN``, ``SA_VTV_PDMN``; ~550 neurons, both sides). These are
  the only body mechanosensors whose axons reach the brain directly; most body
  mechanosensation enters the VNC, which is *not* in this model, so this is an
  approximation. A ``whip_hit`` additionally drives Johnston's-organ wind/gravity
  neurons (cell_sub_class ``wind_gravity``, JO-C/E; Kamikouchi et al. 2009; Yorozu
  et al. 2009) on that side at ``whip_air_scale`` (0.5x) rate, for the air moved by
  the lash. In the model the body set alone drives little downstream activity
  (~80 neurons at 200 Hz; they have low out-degree), the JO input much more.
* hits on the head (``details["body"]`` contains head/eye/antenna/proboscis):
  head bristle mechanosensory neurons (cell_sub_class ``head bristle``) on the side,
  plus the body set at half rate (the neck/thorax is jolted too).
* ``front`` / ``rear`` / ``top`` / ``none`` sides: both sides, 0.7x rate.
* ``fall``: both sides of the body set + Johnston's organ wind/gravity neurons
  (cell_sub_class ``wind_gravity``, JO-C/E; Kamikouchi et al. 2009).
* ``ground_contact``: off by default (no identified leg proprioceptors in the brain
  dataset). With ``enable_ground_contact=True`` it drives the leg sensory
  afferents that ascend via the VTV tract (``SA_VTV_pro_meso_meta``) at a low rate.
* ``manual``: ``details`` = {"set": name} (see ``NAMED_SETS``), or
  {"cell_type": "LC4"}, or {"root_ids": [...]}; optional "side", "rate_hz".
* ``reset``: clears all active stimuli.

Rate: ``rate = max(min_rate, intensity * max_rate)`` Hz (Shiu et al. drive sensory
neurons at 10-200 Hz), for ``duration_s`` of brain time.

Descending readout
------------------
``DESCENDING_TYPES`` maps schema.DESCENDING_GROUPS to FlyWire cell types (split by
soma side); ``descending_to_drive()`` turns the group rates into the 2-vector drive
of the hybrid CPG controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import NeuronTable
from .schema import DESCENDING_GROUPS, BrainState, StimulusEvent

# --- neuron sets ------------------------------------------------------------------

# Shiu et al. 2024 (figures.ipynb / example.ipynb) labellar sugar GRNs, FlyWire
# v630 root ids; 20 of 21 are unchanged in v783 (720575940620900446 is not).
SUGAR_GRN_IDS = (
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
)
# Shiu et al. 2024 bitter GRNs (v630 ids; those still present in v783 are used).
BITTER_GRN_IDS = (
    720575940621778381, 720575940602353632, 720575940617094208, 720575940619197093,
    720575940626287336, 720575940618600651, 720575940627692048, 720575940630195909,
    720575940646212996, 720575940610483162, 720575940645743412, 720575940627578156,
    720575940622298631, 720575940621008895, 720575940629146711, 720575940610259370,
    720575940610481370, 720575940619028208, 720575940614281266, 720575940613061118,
    720575940604027168,
)
# MN9 (proboscis motor neuron; Shiu et al. id_mn9) and its contralateral homologue
# (same FlyWire cell type, CB0701).
MN9_IDS = (720575940660219265, 720575940618238523)

BODY_MECH_SUBCLASSES = ("SA_DMT_DMetaN", "SA_DMT_ADMN", "SA_DLV", "SA_MDA",
                        "SA_VTV_DProN", "SA_VTV_PDMN")
HEAD_WORDS = ("head", "eye", "antenna", "arista", "proboscis", "rostrum", "haustellum")

# descending group -> FlyWire cell types (split by soma side where lateralised)
DESCENDING_TYPES: dict[str, tuple[str, ...]] = {
    # forward walking: BDN2 = DNg100 (Sapkal et al. 2024), oDN1 = DNg97 (Sapkal et
    # al. 2024; same ids as eonsystemspbc/fly-brain's "P9_oDN1"), P9 = DNp09
    # (Bidaye et al. 2020)
    "walk": ("DNg100", "DNg97", "DNp09"),
    # steering: DNa02 and DNa01 (Rayshubskiy et al. 2020; Yang et al. 2023):
    # activity predicts ipsilateral turning
    "turn": ("DNa01", "DNa02"),
    # backward walking: moonwalker descending neuron (Bidaye et al. 2014)
    "backward": ("MDN",),
    # escape take-off: giant fiber DNp01 (von Reyn et al. 2014); not lateralised
    "escape": ("DNp01",),
    # anterior grooming (front-leg rubbing + head sweeps): DNg12 (Guo, Zhang &
    # Simpson 2022, Curr Biol); FlyWire types DNg12_a..e, 21 per side; not lateralised
    "groom": ("DNg12_a", "DNg12_b", "DNg12_c", "DNg12_d", "DNg12_e"),
}


def descending_indices(table: NeuronTable) -> dict[str, np.ndarray]:
    """Model indices for every group in schema.DESCENDING_GROUPS."""
    ct = table.col("cell_type")
    side = table.col("side")
    out: dict[str, np.ndarray] = {}
    for g in DESCENDING_GROUPS:
        base, _, s = g.partition("_")
        types = DESCENDING_TYPES[base]
        m = np.isin(ct, types)
        if s:
            m &= side == {"L": "left", "R": "right"}[s]
        out[g] = np.nonzero(m)[0]
    return out


def named_sets(table: NeuronTable) -> dict[str, np.ndarray]:
    """Named sensory sets usable in ``manual`` stimuli (model indices)."""
    sc, sub, ct = table.col("super_class"), table.col("cell_sub_class"), table.col("cell_type")
    sets = {
        "sugar": table.index_of(SUGAR_GRN_IDS),
        "bitter": table.index_of(BITTER_GRN_IDS),
        "body_mech": np.nonzero((sc == "sensory_ascending") & np.isin(sub, BODY_MECH_SUBCLASSES))[0],
        "head_bristle": np.nonzero(sub == "head bristle")[0],
        "jo_wind_gravity": np.nonzero(sub == "wind_gravity")[0],
        "jo_auditory": np.nonzero(sub == "auditory")[0],
        "jo_grooming": np.nonzero(sub == "grooming")[0],
        "leg_sa": np.nonzero(sub == "SA_VTV_pro_meso_meta")[0],
        "LC4": np.nonzero(ct == "LC4")[0],      # looming detectors -> giant fiber
        "LPLC2": np.nonzero(ct == "LPLC2")[0],  # looming detectors -> giant fiber
    }
    return sets


# --- stimulus mapping -------------------------------------------------------------

@dataclass
class ActiveStimulus:
    label: str
    idx: np.ndarray
    rate_hz: float
    t_end: float


@dataclass
class StimulusMapper:
    """Keeps the currently active stimuli and the resulting Poisson drive."""

    table: NeuronTable
    max_rate_hz: float = 200.0
    min_rate_hz: float = 30.0
    whip_air_scale: float = 0.5  # JO wind neurons for the lash's air movement (0: off)
    enable_ground_contact: bool = False
    ground_contact_rate_hz: float = 20.0
    active: list[ActiveStimulus] = field(default_factory=list)

    def __post_init__(self):
        self.sets = named_sets(self.table)
        self._side = self.table.col("side")
        self._dirty = True

    # ...................................................................... helpers
    def _sided(self, idx: np.ndarray, side: str) -> np.ndarray:
        if side in ("left", "right"):
            return idx[self._side[idx] == side]
        return idx

    def _rate(self, ev: StimulusEvent, scale: float = 1.0) -> float:
        r = float(ev.details.get("rate_hz", 0.0)) if ev.details else 0.0
        if r <= 0:
            r = max(self.min_rate_hz, float(np.clip(ev.intensity, 0.0, 1.0)) * self.max_rate_hz)
        return r * scale

    def resolve(self, ev: StimulusEvent) -> list[tuple[str, np.ndarray, float]]:
        """(label, indices, rate) triples for one event (empty if unmapped)."""
        kind, side = ev.kind, ev.side
        out: list[tuple[str, np.ndarray, float]] = []
        lateral = side in ("left", "right")
        lat_scale = 1.0 if lateral else 0.7
        if kind in ("whip_hit", "shove", "hit"):
            body = str((ev.details or {}).get("body", "")).lower()
            body_set = self._sided(self.sets["body_mech"], side)
            if any(w in body for w in HEAD_WORDS):
                out.append((f"{kind}:head_bristle:{side}",
                            self._sided(self.sets["head_bristle"], side),
                            self._rate(ev, lat_scale)))
                out.append((f"{kind}:body_mech:{side}", body_set, self._rate(ev, 0.5 * lat_scale)))
            else:
                out.append((f"{kind}:body_mech:{side}", body_set, self._rate(ev, lat_scale)))
            if kind == "whip_hit" and self.whip_air_scale > 0:
                out.append((f"{kind}:jo_wind:{side}",
                            self._sided(self.sets["jo_wind_gravity"], side),
                            self._rate(ev, self.whip_air_scale * lat_scale)))
        elif kind == "fall":
            out.append(("fall:body_mech", self.sets["body_mech"], self._rate(ev)))
            out.append(("fall:jo_wind_gravity", self.sets["jo_wind_gravity"], self._rate(ev)))
        elif kind == "ground_contact":
            if self.enable_ground_contact:
                out.append((f"ground_contact:{side}", self._sided(self.sets["leg_sa"], side),
                            self.ground_contact_rate_hz * float(np.clip(ev.intensity, 0, 1))))
        elif kind == "manual":
            d = ev.details or {}
            if "root_ids" in d:
                idx, name = self.table.index_of(d["root_ids"]), "root_ids"
            elif "cell_type" in d:
                name = str(d["cell_type"])
                idx = np.nonzero(self.table.col("cell_type") == name)[0]
            else:
                name = str(d.get("set", "sugar"))
                if name not in self.sets:
                    raise KeyError(f"unknown stimulus set {name!r}; known: {sorted(self.sets)}")
                idx = self.sets[name]
            out.append((f"manual:{name}:{side}", self._sided(idx, side), self._rate(ev)))
        return [(lab, np.asarray(i, dtype=np.int64), r) for lab, i, r in out if len(i) and r > 0]

    # ....................................................................... public
    def add(self, ev: StimulusEvent, now: float) -> list[str]:
        """Register an event at brain time ``now``; returns labels applied."""
        if ev.kind == "reset":
            self.active.clear()
            self._dirty = True
            return ["reset"]
        labels = []
        for lab, idx, rate in self.resolve(ev):
            self.active.append(ActiveStimulus(lab, idx, rate, now + max(ev.duration_s, 0.0)))
            labels.append(lab)
        self._dirty = True
        return labels

    def expire(self, now: float) -> bool:
        n = len(self.active)
        self.active = [s for s in self.active if s.t_end > now]
        if len(self.active) != n:
            self._dirty = True
        return self._dirty

    def drive(self) -> tuple[np.ndarray, np.ndarray]:
        """(indices, rates) of the union of active stimuli (max rate per neuron)."""
        self._dirty = False
        if not self.active:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        idx = np.concatenate([s.idx for s in self.active])
        rate = np.concatenate([np.full(len(s.idx), s.rate_hz) for s in self.active])
        order = np.lexsort((-rate, idx))
        idx, rate = idx[order], rate[order]
        first = np.r_[True, idx[1:] != idx[:-1]]
        return idx[first], rate[first]

    def next_change(self) -> float:
        return min((s.t_end for s in self.active), default=float("inf"))


# --- descending readout -> hybrid controller drive -------------------------------

@dataclass
class DriveGains:
    """Conservative mapping gains (rates in Hz). Baseline is plain walking [1, 1]."""

    r_ref: float = 30.0          # rate at which a group is "clearly on" (tanh scale)
    walk_gain: float = 0.2       # max +/- change of amplitude from walk groups
    turn_gain: float = 0.4       # max left/right amplitude difference / 2 from turn groups
    backward_ref: float = 40.0   # MDN rate that fully reverses stepping
    min_amp: float = 0.3
    max_amp: float = 1.5


def _rates(state) -> dict[str, float]:
    if isinstance(state, BrainState):
        return state.descending
    return dict(state)


def descending_to_drive(state, gains: DriveGains | None = None) -> np.ndarray:
    """Map descending-group rates onto the hybrid CPG drive ``[left, right]``.

    * quiet brain -> ``[1, 1]`` (walking continues);
    * walk_L/R (BDN2/oDN1/P9) raise both amplitudes by up to ``walk_gain`` (they
      promote forward walking; their lateralised steering roles are not used, to
      stay conservative);
    * turn_L - turn_R (DNa01/DNa02, ipsilateral steering): the hybrid controller
      turns *toward* the side with the smaller amplitude, so turn_L lowers left and
      raises right by up to ``turn_gain``;
    * backward_L/R (MDN) blend each side toward -1 (backward stepping); fully
      reversed when MDN fires at ``backward_ref`` Hz;
    * escape (giant fiber) triggers take-off in flies, which this body model cannot
      do; it is intentionally not mapped (the app may use it for a flinch).
    """
    g = gains or DriveGains()
    r = _rates(state)

    def act(k):
        return float(np.tanh(max(r.get(k, 0.0), 0.0) / g.r_ref))

    walk = 0.5 * (act("walk_L") + act("walk_R"))
    amp = np.full(2, 1.0 + g.walk_gain * walk)
    turn = act("turn_L") - act("turn_R")  # >0: steer left
    amp = amp + g.turn_gain * np.array([-turn, turn])
    amp = np.clip(amp, g.min_amp, g.max_amp)
    back = np.array([r.get("backward_L", 0.0), r.get("backward_R", 0.0)])
    m = np.clip(back.mean() / g.backward_ref, 0.0, 1.0)  # MDN acts bilaterally
    drive = (1.0 - m) * amp + m * (-1.0) * np.clip(amp, 0.5, 1.0)
    return drive.astype(np.float64)
