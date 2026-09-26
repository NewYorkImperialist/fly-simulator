"""Sensory screen of the whole-brain LIF model: which inputs reach which DNs.

For every sensory class in the FlyWire annotations (grouped by cell class / cell
type / sub-class, split by the fly's side where that is meaningful), drive the group
with Poisson input and read out the descending-neuron groups of
``mapping.DESCENDING_TYPES`` (walk / turn / backward / escape / groom), the proboscis
motor neuron MN9 and the most active descending cell types. The engine is used
in-process (``LIFEngine``), without the ``BrainProcess`` wrapper.

Also computes a cheap *structural* measure: the minimum number of excitatory
synaptic hops from any neuron of the group to each behaviour DN group.

``scripts/screen_sensory.py`` runs the full screen and writes
``docs/sensory_screen.csv``; see docs/SENSORY_SCREEN.md for the results.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .data import NeuronTable
from .mapping import MN9_IDS, descending_indices, named_sets
from .schema import DESCENDING_GROUPS

SIDES = ("left", "right")
BEHAVIOUR_KEYS = ("walk", "turn", "backward", "escape", "groom")
# individual behaviour DN types read out per side (the groups average over types)
KEY_DN_TYPES = ("DNa01", "DNa02", "DNg100", "DNg97", "DNp09", "MDN", "DNp01")
RUNAWAY_SPS = 50_000.0  # whole-brain spikes/s after input off: "global ignition" (~475k)
KEY_DN_COLS = tuple(f"{t}_{s}" for t in KEY_DN_TYPES for s in "LR")

# Touch-like classes a whip lash could plausibly excite (mechanosensory afferents,
# sensory-ascending body afferents, JO for air movement, ascending neurons which
# carry VNC mechanosensory information).
WHIP_CATEGORIES = ("mech_JO", "mech_bristle", "mech_other", "sensory_ascending",
                   "ascending", "combo_touch")


@dataclass
class SensoryGroup:
    name: str        # unique, e.g. "JO-CM:left"
    category: str    # e.g. "mech_JO", "olfactory", "vpn"
    side: str        # "left" / "right" / "both"
    idx: np.ndarray  # model indices

    @property
    def whip_plausible(self) -> bool:
        return self.category in WHIP_CATEGORIES


def _jo_fine(ct: str) -> str | None:
    """JO-A1 -> JO-A, JO-B1_a -> JO-B, JO-CM -> JO-CM, JO-EV3 -> JO-EV."""
    m = re.match(r"JO-([A-F])([A-Z]?)", ct)
    if not m:
        return "JO-other" if ct.startswith("JO") else None
    return f"JO-{m.group(1)}{m.group(2)}"


def sensory_groups(table: NeuronTable, min_size: int = 2, vpn_min_total: int = 20,
                   an_type_min_total: int = 8, include_photoreceptors: bool = True
                   ) -> list[SensoryGroup]:
    """Enumerate the sensory (and sensory-like input) groups of the screen."""
    sc, cc = table.col("super_class"), table.col("cell_class")
    sub, ct, side = table.col("cell_sub_class"), table.col("cell_type"), table.col("side")
    groups: list[SensoryGroup] = []
    seen: set[str] = set()

    def add(name, cat, mask, per_side=True):
        idx_all = np.nonzero(mask)[0]
        if per_side:
            for s in SIDES:
                idx = idx_all[side[idx_all] == s]
                key = f"{name}:{s}"
                if len(idx) >= min_size and key not in seen:
                    seen.add(key)
                    groups.append(SensoryGroup(key, cat, s, idx))
        else:
            key = f"{name}:both"
            if len(idx_all) >= min_size and key not in seen:
                seen.add(key)
                groups.append(SensoryGroup(key, cat, "both", idx_all))

    sens = sc == "sensory"
    mech = sens & (cc == "mechanosensory")
    # --- Johnston's organ: fine types, coarse letters, functional sub-classes
    jo_fine = np.array([_jo_fine(str(x)) or "" for x in ct], dtype=object)
    for f in sorted(set(jo_fine[mech]) - {""}):
        add(f, "mech_JO", mech & (jo_fine == f))
    jo_letter = np.array([f[:4] if f.startswith("JO-") and f != "JO-other" else "" for f in jo_fine],
                         dtype=object)
    for f in sorted(set(jo_letter[mech]) - {""}):
        if not np.array_equal(mech & (jo_letter == f), mech & (jo_fine == f)):
            add(f + "(all)", "mech_JO", mech & (jo_letter == f))
    for s in ("wind_gravity", "auditory", "grooming"):
        add(f"JO_{s}", "mech_JO", mech & (sub == s))
    add("JO_all", "mech_JO", mech & (jo_fine != ""))
    # --- bristles and other mechanosensory neurons
    for s in ("head bristle", "eye bristle"):
        add(s.replace(" ", "_"), "mech_bristle", mech & (sub == s))
    for t in sorted({str(x) for x in ct[mech] if str(x).startswith("BM_")}):
        add(t, "mech_bristle", mech & (ct == t))
    add("BM_all", "mech_bristle", mech & np.array([str(x).startswith("BM_") for x in ct]))
    for t in ("TPMN1", "TPMN2"):
        add(t, "mech_other", mech & (ct == t))
    add("aPhM", "mech_other", mech & np.array([str(x).startswith("aPhM") for x in ct]))
    add("mech_unassigned", "mech_other", mech & (jo_fine == "") & ~np.isin(sub, ["head bristle", "eye bristle"])
        & ~np.array([str(x).startswith(("BM_", "TPMN", "aPhM")) for x in ct]))
    # --- taste
    gus = sens & (cc == "gustatory")
    for s in ("sugar/water", "bitter", "low-salt", "taste peg"):
        add(f"GRN_{s.replace('/', '_').replace(' ', '_')}", "gustatory", gus & (sub == s))
    add("GRN_pharyngeal", "gustatory", gus & np.array(["pharyngeal" in str(x) for x in sub]))
    for t in sorted(set(ct[gus]) - {""}):
        if not str(t).startswith("PhG"):
            add(f"GRN_{t}", "gustatory", gus & (ct == t))
    # --- olfaction (ORN axons project bilaterally; sides pooled)
    olf = sens & (cc == "olfactory")
    for t in sorted(set(ct[olf]) - {""}):
        add(t, "olfactory", olf & (ct == t), per_side=False)
    add("ORN_pheromone", "olfactory", olf & (sub == "pheromone"))
    # --- thermo / hygro (small; sides pooled)
    for c in ("thermosensory", "hygrosensory"):
        m = sens & (cc == c)
        for t in sorted(set(ct[m]) - {""}):
            add(t, "thermo_hygro", m & (ct == t), per_side=False)
    unk = sens & (cc == "unknown_sensory")
    for t in sorted(set(ct[unk]) - {""}):
        add(t, "unknown_sensory", unk & (ct == t), per_side=False)
    # --- vision: photoreceptors, ocelli, DRA
    vis = sens & (cc == "visual")
    add("ocellar_PR", "visual_PR", vis & (sub == "ocellar"), per_side=False)
    add("DRA_PR", "visual_PR", vis & (sub == "DRA"))
    if include_photoreceptors:
        for t in ("R1-6", "R7", "R8"):
            add(t, "visual_PR", vis & (ct == t) & (sub == ""))
    # --- sensory ascending (VNC afferents reaching the brain)
    sa = sc == "sensory_ascending"
    for s in sorted(set(sub[sa]) - {""}):
        add(s, "sensory_ascending", sa & (sub == s))
    for t in sorted(set(ct[sa]) - {""}):
        add(t, "sensory_ascending", sa & (ct == t))
    add("body_mech", "sensory_ascending", np.isin(np.arange(table.n), named_sets(table)["body_mech"]))
    # --- ascending neurons (VNC -> brain; carry leg/body mechanosensory signals)
    an = sc == "ascending"
    for s in sorted(set(sub[an]) - {""}):
        add(s, "ascending", an & (sub == s))
    cnt = defaultdict(int)
    for t in ct[an]:
        cnt[str(t)] += 1
    for t in sorted(t for t, k in cnt.items() if t and k >= an_type_min_total):
        add(t, "ascending", an & (ct == t))
    # --- visual projection neurons (optic lobe -> central brain)
    vp = sc == "visual_projection"
    cnt = defaultdict(int)
    for t in ct[vp]:
        cnt[str(t)] += 1
    for t in sorted(t for t, k in cnt.items() if t and k >= vpn_min_total):
        add(t, "vpn", vp & (ct == t))
    add("ocellar_VPN", "vpn", vp & (cc == "ocellar"), per_side=False)
    # --- combinations for the whip question
    body = np.isin(np.arange(table.n), named_sets(table)["body_mech"])
    touch = (mech | sa | body)
    add("all_touch", "combo_touch", touch)
    add("all_touch", "combo_touch", touch, per_side=False)
    add("body+headbristle+JOwind", "combo_touch",
        body | (mech & np.isin(sub, ["head bristle", "wind_gravity"])))
    add("all_ascending", "combo_touch", an)
    return groups


def dn_readout_index(table: NeuronTable):
    """(behaviour group -> idx, MN9 idx, DN idx, DN labels "type_L", key-type idx,
    (OA idx, OA labels))."""
    groups = descending_indices(table)
    mn9 = table.index_of(MN9_IDS)
    dn = np.nonzero(table.col("super_class") == "descending")[0]
    ct, side = table.col("cell_type"), table.col("side")
    labels = np.array([f"{ct[i] or table.root_id[i]}_{str(side[i])[:1].upper()}" for i in dn],
                      dtype=object)
    key = {f"{t}_{s}": np.nonzero((ct == t) & (side == {"L": "left", "R": "right"}[s]))[0]
           for t in KEY_DN_TYPES for s in "LR"}
    # octopaminergic neurons: the named OA cell types (OA-VUM*, OA-AL2*, OA-VPM*,
    # OA-ASM*, ...; Busch et al. 2009), which drive arousal / locomotor speed. The
    # predicted-NT octopamine set is not used: it contains many obvious mispredictions
    # (photoreceptors, JO neurons).
    oa = np.nonzero(np.array([str(x).startswith("OA-") for x in ct]))[0]
    oa_labels = np.array([f"{ct[i] or 'OA'}_{str(side[i])[:1].upper()}" for i in oa], dtype=object)
    return groups, mn9, dn, labels, key, (oa, oa_labels)


def _mean(rates: np.ndarray, idx: np.ndarray) -> float:
    return float(rates[idx].mean()) if len(idx) else 0.0


def stimulate(engine, idx: np.ndarray, rate_hz: float, seconds: float,
              readout, chunk_s: float = 0.05, n_top: int = 20) -> dict:
    """Reset the engine, drive ``idx`` at ``rate_hz`` for ``seconds`` and read out."""
    groups, mn9, dn, labels, key, (oa, oa_labels) = readout
    engine.reset_state()
    engine.set_poisson(np.asarray(idx, dtype=np.int64), rate_hz)
    c0 = engine.counts.copy()
    n_chunks = max(1, int(round(seconds / chunk_s)))
    for _ in range(n_chunks):
        engine.run_seconds(chunk_s)
    engine.set_poisson(np.zeros(0, dtype=np.int64), 0.0)
    T = n_chunks * chunk_s
    rates = (engine.counts - c0) / T
    # self-sustained activity: total spike rate (spikes/s, whole brain) 50-100 ms
    # after the input stops. Normally ~0 (activity decays within ~20 ms); some
    # inputs (e.g. ORNs) ignite a persistent AL / mushroom-body state instead.
    engine.run_seconds(0.05)
    _, after = engine.run_seconds(0.05)
    out = {g: _mean(rates, groups[g]) for g in DESCENDING_GROUPS}
    out["MN9"] = _mean(rates, mn9)
    out.update({k: _mean(rates, i) for k, i in key.items()})
    stim = np.zeros(len(rates), dtype=bool)
    stim[idx] = True
    out["n_active"] = int(np.count_nonzero((rates > 0) & ~stim))
    dr = rates[dn]
    out["n_dn_10hz"] = int(np.count_nonzero(dr >= 10))
    out["persist_sps"] = len(after) / 0.05
    ora = np.where(stim[oa], 0.0, rates[oa])
    out["OA_mean"] = float(ora.mean()) if len(oa) else 0.0
    out["n_OA_10hz"] = int(np.count_nonzero(ora >= 10))
    order = np.argsort(-ora)[:5]
    out["top_oa"] = [(str(oa_labels[i]), float(ora[i])) for i in order if ora[i] >= 1.0]
    # top DN cell types (mean over same type+side)
    agg: dict[str, list] = defaultdict(list)
    for lab, r in zip(labels, dr):
        agg[lab].append(r)
    means = sorted(((float(np.mean(v)), k) for k, v in agg.items()), reverse=True)
    out["top_dn"] = [(k, r) for r, k in means[:n_top] if r >= 1.0]
    return out


def min_hops_to(indptr, indices, weights, targets: np.ndarray, n: int,
                min_syn: int = 1, max_hops: int = 8) -> np.ndarray:
    """Hops along excitatory edges (>= min_syn synapses) from every neuron to the
    target set (0 for targets, -1 unreachable within max_hops)."""
    from scipy.sparse import csr_matrix

    keep = weights >= min_syn
    pre = np.repeat(np.arange(n), np.diff(indptr))[keep]
    post = np.asarray(indices)[keep]
    # reverse graph: row = post, col = pre
    A = csr_matrix((np.ones(len(pre), dtype=np.int8), (post, pre)), shape=(n, n))
    dist = np.full(n, -1, dtype=np.int16)
    frontier = np.zeros(n, dtype=bool)
    frontier[targets] = True
    dist[targets] = 0
    for h in range(1, max_hops + 1):
        nxt = _expand(A, frontier)
        nxt &= dist < 0
        if not nxt.any():
            break
        dist[nxt] = h
        frontier = nxt
    return dist


def _expand(A, frontier: np.ndarray) -> np.ndarray:
    """Presynaptic partners of the frontier (A rows are postsynaptic neurons)."""
    rows = np.nonzero(frontier)[0]
    sub = A[rows]
    out = np.zeros(A.shape[0], dtype=bool)
    out[sub.indices] = True
    return out


def group_hops(indptr, indices, weights, table: NeuronTable, groups: list[SensoryGroup],
               min_syn: int = 1) -> dict[str, dict[str, int]]:
    """{group name: {behaviour: min excitatory hops}} (-1: unreachable)."""
    dn = descending_indices(table)
    n = table.n
    targets = {k: np.concatenate([dn[g] for g in DESCENDING_GROUPS if g.split("_")[0] == k])
               for k in BEHAVIOUR_KEYS}
    targets["MN9"] = table.index_of(MN9_IDS)
    dists = {k: min_hops_to(indptr, indices, weights, t, n, min_syn) for k, t in targets.items()
             if len(t)}
    out = {}
    for g in groups:
        row = {}
        for k, d in dists.items():
            v = d[g.idx]
            v = v[v >= 0]
            row[k] = int(v.min()) if len(v) else -1
        out[g.name] = row
    return out


def run_screen(engine, table: NeuronTable, groups: list[SensoryGroup],
               rates=(100.0, 200.0), seconds: float = 0.5, trials: int = 1,
               progress=None) -> list[dict]:
    """Serial screen with one engine. Returns one row per (group, rate)."""
    readout = dn_readout_index(table)
    rows = []
    for gi, g in enumerate(groups):
        for r in rates:
            rows.append(screen_one(engine, g, r, seconds, trials, readout))
            if progress:
                progress(gi, g, rows[-1])
    return rows


def screen_one(engine, g: SensoryGroup, rate: float, seconds: float, trials: int,
               readout) -> dict:
    import time

    t0 = time.perf_counter()
    res = [stimulate(engine, g.idx, rate, seconds, readout) for _ in range(trials)]
    row = {"group": g.name, "category": g.category, "side": g.side, "n": len(g.idx),
           "rate_hz": rate, "whip_plausible": g.whip_plausible}
    for k in (*DESCENDING_GROUPS, "MN9", *KEY_DN_COLS, "n_active", "n_dn_10hz", "persist_sps",
              "OA_mean", "n_OA_10hz"):
        row[k] = float(np.mean([x[k] for x in res]))
    agg: dict[str, list] = defaultdict(list)
    for x in res:
        for lab, r in x["top_dn"]:
            agg[lab].append(r)
    top = sorted(((sum(v) / trials, k) for k, v in agg.items()), reverse=True)[:20]
    row["top_dn"] = ";".join(f"{k}:{r:.0f}" for r, k in top)
    oa_agg: dict[str, float] = defaultdict(float)
    for x in res:
        for lab, r in x["top_oa"]:
            oa_agg[lab] = max(oa_agg[lab], r)
    row["top_oa"] = ";".join(f"{k}:{r:.0f}" for k, r in sorted(oa_agg.items(), key=lambda kv: -kv[1])[:5])
    row["wall_s"] = time.perf_counter() - t0
    row["runaway"] = row["persist_sps"] > RUNAWAY_SPS
    return row


def behaviour_max(row: dict) -> dict[str, float]:
    """Per behaviour: max over sides of the group rate (Hz)."""
    out = {}
    for k in BEHAVIOUR_KEYS:
        vals = [row[g] for g in DESCENDING_GROUPS if g.split("_")[0] == k]
        out[k] = max(vals) if vals else 0.0
    out["MN9"] = row["MN9"]
    return out
