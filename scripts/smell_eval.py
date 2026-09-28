"""Compare the antennal-lobe smell fixes (fly_simulator/brain/smell.py, docs/SMELL.md).

For each fix: drive single glomeruli and mixtures at ORN rates 20 / 50 / 100 Hz for
``--on`` s, then record ``--off`` s without input. Per trial: runaway (whole-brain
> 50k spikes/s 50-100 ms after offset, as in docs/SENSORY_SCREEN.md), residual
activity 0.4-0.5 s after offset, the odour's own uniglomerular PNs, other PNs,
KCs, lateral horn, and odour specificity (pattern correlations). Then the other
validated behaviours (sugar -> MN9, looming -> GF, LC10a -> DNa02) per fix.

    .venv/bin/python scripts/smell_eval.py                    # all fixes
    .venv/bin/python scripts/smell_eval.py --fixes none sign_adapt --trials 3
    .venv/bin/python scripts/smell_eval.py --csv docs/smell_eval.csv

One engine, one process, serial.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STIMULI = ("glom_DM1", "glom_DM2", "glom_VA2", "glom_DL5", "glom_DC1", "glom_DA2",
           "vinegar", "mix_DL5_DC1_DA2")
MIXES = {"mix_DL5_DC1_DA2": {"DL5": 1.0, "DC1": 1.0, "DA2": 1.0}}
RATES = (20.0, 50.0, 100.0)


def build(table):
    ct = table.col("cell_type").astype(str)
    cc, sub = table.col("cell_class"), table.col("cell_sub_class")
    upn = np.nonzero((cc == "ALPN") & (sub == "uniglomerular"))[0]
    glom_of = np.array([c.split("_")[0] for c in ct[upn]], dtype=object)
    kc = np.nonzero(cc == "Kenyon_Cell")[0]
    lh = np.nonzero(np.isin(cc, ["LHLN", "LHCENT"]) |
                    (np.array(table.regions, dtype=object)[table.region] == "LH_L") |
                    (np.array(table.regions, dtype=object)[table.region] == "LH_R"))[0]
    lh = np.setdiff1d(lh, np.nonzero(cc == "ALPN")[0])
    alln = np.nonzero(cc == "ALLN")[0]
    return dict(upn=upn, glom_of=glom_of, kc=kc, lh=lh, alln=alln)


def trial(eng, table, sets, glomeruli: dict, rate: float, on: float, off: float):
    from fly_simulator.brain.smell import glomerulus_orns

    idx = np.concatenate([glomerulus_orns(table, g) for g in glomeruli])
    rates = np.concatenate([np.full(len(glomerulus_orns(table, g)), rate * w)
                            for g, w in glomeruli.items()])
    eng.reset_state()
    eng.set_poisson(idx, rates)
    c0 = eng.counts.copy()
    eng.run_seconds(on)
    r_on = (eng.counts - c0) / on
    eng.set_poisson(np.zeros(0, np.int64), 0.0)
    eng.run_seconds(0.05)
    _, a = eng.run_seconds(0.05)
    runaway_sps = len(a) / 0.05
    n_mid = int(round((off - 0.2) / 0.1))
    for _ in range(max(n_mid, 0)):
        eng.run_seconds(0.1)
    _, b = eng.run_seconds(0.1)
    residual = len(b) / 0.1
    upn, g_of = sets["upn"], sets["glom_of"]
    own = np.isin(g_of, list(glomeruli))
    return dict(runaway=runaway_sps > 50_000, persist_sps=runaway_sps, residual_sps=residual,
                own_pn=float(r_on[upn[own]].mean()), other_pn=float(r_on[upn[~own]].mean()),
                kc=float(r_on[sets["kc"]].mean()),
                kc_frac=float((r_on[sets["kc"]] >= 2.0).mean()),
                lh=float(r_on[sets["lh"]].mean()), alln=float(r_on[sets["alln"]].mean()),
                total_sps=float(r_on.sum()),
                vec_kc=(r_on[sets["kc"]] >= 2.0).astype(np.float32),
                vec_lh=r_on[sets["lh"]].astype(np.float32),
                vec_pn=r_on[upn].astype(np.float32))


def specificity(vecs: dict[str, list]) -> tuple[float, float]:
    """Mean pattern correlation within a stimulus (across trials) and between stimuli."""
    def corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])
    keys = [k for k, v in vecs.items() if v]
    within = [corr(vecs[k][i], vecs[k][j]) for k in keys for i in range(len(vecs[k]))
              for j in range(i + 1, len(vecs[k]))]
    between = [corr(vecs[a][0], vecs[b][0]) for i, a in enumerate(keys) for b in keys[i + 1:]]
    return float(np.nanmean(within)) if within else np.nan, \
        float(np.nanmean(between)) if between else np.nan


def behaviours(eng, table) -> dict:
    from fly_simulator.brain.mapping import MN9_IDS, descending_indices, named_sets

    s = named_sets(table)
    dn = descending_indices(table)
    ct, side = table.col("cell_type"), table.col("side")
    out = {}

    def run(idx, rate, sec=0.5):
        eng.reset_state()
        eng.set_poisson(idx, rate)
        c0 = eng.counts.copy()
        eng.run_seconds(sec)
        eng.set_poisson(np.zeros(0, np.int64), 0.0)
        r = (eng.counts - c0) / sec
        eng.run_seconds(0.05)
        _, a = eng.run_seconds(0.05)
        return r, len(a) / 0.05

    r, p = run(s["sugar"], 200.0)
    out["sugar_MN9"] = float(r[table.index_of(MN9_IDS[:1])].mean())
    out["sugar_persist"] = p
    loom = np.concatenate([s["LC4"], s["LPLC2"]])
    loom = loom[side[loom] == "left"]
    r, p = run(loom, 150.0)
    out["loom_GF"] = float(r[dn["escape"]].mean())
    out["loom_persist"] = p
    lc = s["LC10a"][side[s["LC10a"]] == "left"]
    r, p = run(lc, 90.0)
    a02 = lambda sd: float(r[np.nonzero((ct == "DNa02") & (side == sd))[0]].mean())
    out["LC10aL_DNa02_L"] = a02("left")
    out["LC10aL_DNa02_R"] = a02("right")
    out["lc10_persist"] = p
    return out


def main(argv=None) -> int:
    from fly_simulator.brain.data import load_connectome, load_neuron_table
    from fly_simulator.brain.engine import LIFEngine
    from fly_simulator.brain.smell import ODORS, SMELL_FIXES, SmellFix

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixes", nargs="*", default=list(SMELL_FIXES))
    ap.add_argument("--trials", type=int, default=3, help="per stimulus and rate")
    ap.add_argument("--none-trials", type=int, default=1)
    ap.add_argument("--on", type=float, default=0.5)
    ap.add_argument("--off", type=float, default=0.5)
    ap.add_argument("--csv", default="")
    ap.add_argument("--no-behaviour", action="store_true")
    a = ap.parse_args(argv)

    table = load_neuron_table()
    eng = LIFEngine(*load_connectome()[:3], seed=0)
    eng.run(1)
    sets = build(table)
    fix = SmellFix(table, eng, "none")
    rows, summary = [], []
    for name in a.fixes:
        fix.configure(name)
        t0 = time.time()
        n_tr = a.none_trials if name == "none" else a.trials
        res = []
        vecs = {"kc": {}, "lh": {}, "pn": {}}
        for stim in STIMULI:
            gl = MIXES.get(stim) or ODORS[stim]
            for rate in RATES:
                for k in range(n_tr):
                    r = trial(eng, table, sets, gl, rate, a.on, a.off)
                    if rate == 50.0:
                        for v in vecs:
                            vecs[v].setdefault(stim, []).append(r.pop(f"vec_{v}"))
                    for v in ("kc", "lh", "pn"):
                        r.pop(f"vec_{v}", None)
                    r.update(fix=name, stim=stim, rate_hz=rate, trial=k)
                    res.append(r)
        rows += res
        wall = time.time() - t0
        spec = {v: specificity(vecs[v]) for v in vecs}
        n = len(res)
        agg = dict(fix=name, trials=n, runaway=sum(r["runaway"] for r in res),
                   returned=sum(r["residual_sps"] < 1000 for r in res),
                   wall_s=round(wall, 1))
        for rate in RATES:
            sub = [r for r in res if r["rate_hz"] == rate and not r["runaway"]]
            for k in ("own_pn", "other_pn", "kc_frac", "lh"):
                agg[f"{k}@{int(rate)}"] = float(np.mean([r[k] for r in sub])) if sub else np.nan
        for v, (w, b) in spec.items():
            agg[f"spec_{v}"] = f"{w:.2f}/{b:.2f}"
        if not a.no_behaviour:
            agg.update(behaviours(eng, table))
        summary.append(agg)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in agg.items()},
              flush=True)
    if a.csv:
        keys = [k for k in rows[0]]
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
        with open(Path(a.csv).with_name(Path(a.csv).stem + "_summary.csv"), "w", newline="") as f:
            keys = list(dict.fromkeys(k for s in summary for k in s))
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for s in summary:
                w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
