#!/usr/bin/env python
"""Sensory screen of the FlyWire whole-brain LIF model (see docs/SENSORY_SCREEN.md).

    .venv/bin/python scripts/screen_sensory.py                  # full screen (1 worker; --workers 2 for speed)
    .venv/bin/python scripts/screen_sensory.py --only JO,BM_    # groups whose name matches
    .venv/bin/python scripts/screen_sensory.py --list           # just list the groups

Every sensory group (fly_simulator.brain.screen.sensory_groups) is driven with
Poisson input at each rate for ``--seconds`` x ``--trials``; the descending groups,
MN9, key DN types and the top-20 DN cell types are written to ``--out`` (CSV).
The engine runs in-process in ``--workers`` spawned processes (each loads the
connectome, ~0.5 GB RSS).
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_simulator.brain.schema import DESCENDING_GROUPS  # noqa: E402
from fly_simulator.brain.screen import (BEHAVIOUR_KEYS, KEY_DN_COLS,  # noqa: E402
                                       dn_readout_index, group_hops, screen_one,
                                       sensory_groups)

_W: dict = {}


def _init(seed_base: int):
    from fly_simulator.brain.data import load_connectome, load_neuron_table
    from fly_simulator.brain.engine import LIFEngine

    indptr, indices, w, _ = load_connectome()
    table = load_neuron_table()
    seed = seed_base + mp.current_process()._identity[0] if mp.current_process()._identity else seed_base
    _W["engine"] = LIFEngine(indptr, indices, w, seed=seed)
    _W["table"] = table
    _W["groups"] = {g.name: g for g in sensory_groups(table)}
    _W["readout"] = dn_readout_index(table)


def _task(args):
    name, rate, seconds, trials = args
    return screen_one(_W["engine"], _W["groups"][name], rate, seconds, trials, _W["readout"])


def summarize(path: str, rate: float = 200.0, top: int = 12, thr: float = 5.0) -> None:
    """Print markdown tables from a screen CSV (stdlib only)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    num = [c for c in rows[0] if c not in ("group", "category", "side", "top_dn", "top_oa")]
    for r in rows:
        for c in num:
            r[c] = float(r[c])
    d_all = [r for r in rows if r["rate_hz"] == rate]
    ign = [r for r in d_all if r.get("runaway")]
    d = [r for r in d_all if not r.get("runaway")]  # self-sustained runs reported separately
    beh = {"walk": ["walk_L", "walk_R"], "turn": ["turn_L", "turn_R"],
           "backward": ["backward_L", "backward_R"], "escape": ["escape"],
           "groom": ["groom"], "MN9": ["MN9"]}
    for r in d:
        for k, cols in beh.items():
            r[k] = max(r[c] for c in cols)
        r["best"] = max(r[k] for k in ("walk", "turn", "backward", "escape", "groom"))
    show = ["walk_L", "walk_R", "turn_L", "turn_R", "backward_L", "backward_R",
            "escape", "groom", "MN9"]

    def table(sub):
        print("| group | cat | n | " + " | ".join(show) + " | top DNs |")
        print("|---|---|---|" + "---|" * len(show) + "---|")
        for r in sub:
            tops = ", ".join(r["top_dn"].split(";")[:4])
            print(f"| {r['group']} | {r['category']} | {r['n']:.0f} | "
                  + " | ".join(f"{r[c]:.0f}" for c in show) + f" | {tops} |")

    for k in beh:
        sub = sorted((r for r in d if r[k] >= thr), key=lambda r: -r[k])
        print(f"\n### {k} (>= {thr:g} Hz at {rate:g} Hz input): {len(sub)} groups\n")
        table(sub[:top])
    print(f"\n### whip-plausible groups with any behaviour DN >= {thr:g} Hz\n")
    table(sorted((r for r in d if r["whip_plausible"] and r["best"] >= thr),
                 key=lambda r: -r["best"]))
    print("\n### lateralisation (lateral groups, turn or walk >= thr)\n")
    print("| group | turn_L | turn_R | walk_L | walk_R | DNa01 L/R | DNa02 L/R | DNg100 L/R | DNg97 L/R | DNp09 L/R |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted((r for r in d if r["side"] != "both" and (r["turn"] >= thr or r["walk"] >= thr)),
                    key=lambda r: -max(r["turn"], r["walk"])):
        pair = " | ".join(f"{r[t + '_L']:.0f}/{r[t + '_R']:.0f}"
                          for t in ("DNa01", "DNa02", "DNg100", "DNg97", "DNp09"))
        print(f"| {r['group']} | {r['turn_L']:.0f} | {r['turn_R']:.0f} | {r['walk_L']:.0f} | "
              f"{r['walk_R']:.0f} | {pair} |")
    print(f"\n### self-sustained (runaway) runs at {rate:g} Hz, excluded above: {len(ign)}\n")
    print(", ".join(f"{r['group']} ({r['persist_sps'] / 1000:.0f}k sp/s)" for r in ign))
    print("\n### counts by category (groups with DN group >= thr)\n")
    for c in sorted({r["category"] for r in d}):
        sub = [r for r in d if r["category"] == c]
        print(f"- {c}: {len(sub)} groups; " + ", ".join(
            f"{k} {sum(r[k] >= thr for r in sub)}" for k in beh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="100,200")
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", default="", help="comma-separated substrings of group names")
    ap.add_argument("--categories", default="", help="comma-separated categories to keep")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "docs" / "sensory_screen.csv"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--summarize", metavar="CSV", help="print markdown tables and exit")
    a = ap.parse_args()
    if a.summarize:
        summarize(a.summarize)
        return

    from fly_simulator.brain.data import load_connectome, load_neuron_table

    table = load_neuron_table()
    groups = sensory_groups(table)
    if a.only:
        keys = a.only.split(",")
        groups = [g for g in groups if any(k in g.name for k in keys)]
    if a.categories:
        cats = set(a.categories.split(","))
        groups = [g for g in groups if g.category in cats]
    if a.list:
        for g in groups:
            print(f"{g.name:40s} {g.category:18s} {len(g.idx):6d}")
        print(len(groups), "groups")
        return
    t_start = time.time()
    indptr, indices, w, _ = load_connectome()
    hops = group_hops(indptr, indices, w, table, groups, min_syn=1)
    hops5 = group_hops(indptr, indices, w, table, groups, min_syn=5)
    print(f"{len(groups)} groups; hop analysis {time.time() - t_start:.1f} s", flush=True)

    rates = [float(r) for r in a.rates.split(",")]
    # biggest groups first so the tail of the pool is short
    tasks = [(g.name, r, a.seconds, a.trials)
             for g in sorted(groups, key=lambda g: -len(g.idx)) for r in rates]
    rows = []
    t0 = time.time()

    def log(k, row):
        rows.append(row)
        if k % 25 == 0 or k == len(tasks) - 1:
            print(f"  {k + 1}/{len(tasks)}  {time.time() - t0:6.0f} s  last {row['group']} "
                  f"@{row['rate_hz']:.0f}", flush=True)

    if a.workers <= 1:  # in-process: a single brain in memory
        from fly_simulator.brain.engine import LIFEngine

        engine = LIFEngine(indptr, indices, w, seed=a.seed + 1)
        del indptr, indices, w
        by_name = {g.name: g for g in groups}
        readout = dn_readout_index(table)
        for k, (name, r, sec, tr) in enumerate(tasks):
            log(k, screen_one(engine, by_name[name], r, sec, tr, readout))
    else:
        del indptr, indices, w
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.workers, initializer=_init, initargs=(a.seed,)) as pool:
            for k, row in enumerate(pool.imap_unordered(_task, tasks, chunksize=1)):
                log(k, row)
    wall = time.time() - t0

    order = {g.name: i for i, g in enumerate(groups)}
    rows.sort(key=lambda r: (order[r["group"]], r["rate_hz"]))
    cols = ["group", "category", "side", "n", "rate_hz", "whip_plausible",
            *DESCENDING_GROUPS, "MN9", *KEY_DN_COLS, "n_active", "n_dn_10hz", "persist_sps", "runaway",
            *[f"hops_{k}" for k in (*BEHAVIOUR_KEYS, "MN9")],
            *[f"hops5_{k}" for k in (*BEHAVIOUR_KEYS, "MN9")], "top_dn",
            "OA_mean", "n_OA_10hz", "top_oa"]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for r in rows:
            h, h5 = hops[r["group"]], hops5[r["group"]]
            vals = []
            for c in cols:
                if c.startswith("hops5_"):
                    vals.append(h5.get(c[6:], -1))
                elif c.startswith("hops_"):
                    vals.append(h.get(c[5:], -1))
                elif c == "OA_mean":
                    vals.append(f"{r[c]:.2f}")
                elif c == "top_dn":  # full top-20 only at the highest rate (file size)
                    vals.append(r[c] if r["rate_hz"] == max(rates) else "")
                elif c == "rate_hz":
                    vals.append(f"{r[c]:.0f}")
                elif isinstance(r[c], float):
                    vals.append(f"{r[c]:.0f}" if r[c] >= 10 or r[c] == 0 else f"{r[c]:.1f}")
                elif isinstance(r[c], bool):
                    vals.append(int(r[c]))
                else:
                    vals.append(r[c])
            wr.writerow(vals)
    sim_s = len(tasks) * a.seconds * a.trials
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB); {len(tasks)} runs, "
          f"{sim_s:.0f} s simulated in {wall:.0f} s wall with {a.workers} workers")


if __name__ == "__main__":
    main()
