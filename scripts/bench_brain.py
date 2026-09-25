#!/usr/bin/env python
"""Benchmark + validate the whole-brain LIF model (Shiu et al. 2024, FlyWire v783).

    .venv/bin/python scripts/bench_brain.py                 # engine only (fast)
    .venv/bin/python scripts/bench_brain.py --brian2        # + Brian2 runtime (cython)
    .venv/bin/python scripts/bench_brain.py --standalone    # + Brian2 cpp_standalone
    .venv/bin/python scripts/bench_brain.py --validate 10   # engine vs Brian2, 10 x 1 s

"RTF" = simulated brain seconds per wall second (>1 is faster than real time).
Each Brian2 measurement runs in a subprocess (Brian2 devices are process-global).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perpetualfly.brain.data import data_available, load_connectome, load_neuron_table  # noqa: E402
from perpetualfly.brain.mapping import MN9_IDS, named_sets  # noqa: E402

RATE = 200.0  # Hz, sugar GRN drive (Shiu et al. example result)


def _stim(tab, name):
    return named_sets(tab)[name]


# ----------------------------------------------------------------------------- engine
def bench_engine(seconds: float = 2.0) -> dict:
    from perpetualfly.brain.engine import LIFEngine

    t0 = time.time()
    indptr, indices, w, n = load_connectome()
    tab = load_neuron_table()
    t_load = time.time() - t0
    t0 = time.time()
    eng = LIFEngine(indptr, indices, w, seed=0)
    eng.run(10)
    t_init = time.time() - t0
    out = {"n_neurons": n, "n_connections": int(len(indices)), "load_s": t_load,
           "init_jit_s": t_init, "cases": {}}
    print(f"engine: {n} neurons, {len(indices)} connections; load {t_load:.2f} s, "
          f"init+jit {t_init:.2f} s")
    body_left = _stim(tab, "body_mech")
    body_left = body_left[tab.col("side")[body_left] == "left"]
    cases = [("quiet", np.zeros(0, int)), ("sugar", _stim(tab, "sugar")),
             ("body_mech_left", body_left),
             ("sugar+bitter+body+LC4", np.concatenate([
                 _stim(tab, "sugar"), _stim(tab, "bitter"), _stim(tab, "body_mech"),
                 _stim(tab, "LC4")]))]
    for name, idx in cases:
        eng.reset_state()
        eng.set_poisson(idx, RATE)
        eng.run_seconds(0.2)  # warm up into steady state
        for chunk_ms in (10, 20, 50):
            n_chunks = int(round(seconds * 1000 / chunk_ms))
            c0 = eng.counts.sum()
            t0 = time.perf_counter()
            for _ in range(n_chunks):
                eng.run_seconds(chunk_ms / 1000)
            wall = time.perf_counter() - t0
            rtf = n_chunks * chunk_ms / 1000 / wall
            spk = int(eng.counts.sum() - c0)
            out["cases"][f"{name}@{chunk_ms}ms"] = {"rtf": rtf, "spikes_per_s": spk / seconds,
                                                   "n_active_state": eng.n_active}
            print(f"  {name:24s} chunk {chunk_ms:3d} ms  RTF {rtf:7.2f}  "
                  f"spikes/s {spk / seconds:9.0f}  active-state neurons {eng.n_active}")
    return out


# ----------------------------------------------------------------------------- brian2
def _brian2_child(mode: str, seconds: float, trials: int, workdir: str) -> dict:
    import brian2 as b2
    from brian2 import ms

    from perpetualfly.brain.brian2_ref import build_network

    b2.prefs.codegen.target = "cython"
    b2.BrianLogger.suppress_name("resolution_conflict")
    indptr, indices, w, n = load_connectome()
    tab = load_neuron_table()
    sugar = _stim(tab, "sugar")
    res: dict = {"mode": mode}
    if mode == "standalone":
        b2.set_device("cpp_standalone", directory=workdir, build_on_run=False)
        t0 = time.time()
        net, neu, mon, _ = build_network(indptr, indices, w, sugar, RATE, seed=0)
        net.run(seconds * 1000 * ms)
        res["setup_s"] = time.time() - t0
        t0 = time.time()
        b2.device.build(directory=workdir, compile=True, run=False)
        res["compile_s"] = time.time() - t0
        runs = []
        for k in range(2):
            t0 = time.time()
            b2.device.run(directory=workdir, with_output=False, run_args={neu.v: -52 * b2.mV})
            runs.append(time.time() - t0)
        res["run_s"] = runs
        res["rtf_run_only"] = seconds / min(runs)
        res["spikes"] = int(mon.num_spikes)
        return res
    t0 = time.time()
    net, neu, mon, _ = build_network(indptr, indices, w, sugar, RATE, seed=0)
    res["build_s"] = time.time() - t0
    t0 = time.time()
    net.run(1 * ms)
    res["first_run_compile_s"] = time.time() - t0
    if mode == "runtime":
        for chunk_ms in (20, 50, 1000):
            n_chunks = max(1, int(round(seconds * 1000 / chunk_ms)))
            t0 = time.time()
            for _ in range(n_chunks):
                net.run(chunk_ms * ms)
            wall = time.time() - t0
            res[f"rtf@{chunk_ms}ms"] = n_chunks * chunk_ms / 1000 / wall
        return res
    # mode == "validate": trials x 1 s from rest, rates per neuron
    net.store("init")
    rates = np.zeros((trials, n), dtype=np.float32)
    for k in range(trials):
        net.restore("init")
        b2.seed(1000 + k)
        c0 = np.asarray(mon.count[:]).copy()
        net.run(1000 * ms)
        rates[k] = np.asarray(mon.count[:]) - c0
    np.save(Path(workdir) / "brian2_rates.npy", rates)
    res["rates_file"] = str(Path(workdir) / "brian2_rates.npy")
    return res


def run_brian2(mode: str, seconds: float = 1.0, trials: int = 0, workdir: str | None = None) -> dict:
    workdir = workdir or tempfile.mkdtemp(prefix="brian2_")
    cmd = [sys.executable, __file__, "--child", mode, "--seconds", str(seconds),
           "--trials", str(trials), "--workdir", workdir]
    print(f"brian2 {mode}: running {' '.join(cmd[2:])}", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-4000:])
        raise RuntimeError(f"brian2 {mode} failed")
    res = json.loads(p.stdout.strip().splitlines()[-1])
    res["total_wall_s"] = time.time() - t0
    print("  " + json.dumps(res))
    return res


# ----------------------------------------------------------------------------- validation
def validate(trials: int) -> dict:
    from perpetualfly.brain.engine import LIFEngine

    work = tempfile.mkdtemp(prefix="brian2_val_")
    b = run_brian2("validate", trials=trials, workdir=work)
    rb = np.load(b["rates_file"])
    indptr, indices, w, n = load_connectome()
    tab = load_neuron_table()
    sugar = _stim(tab, "sugar")
    re_ = np.zeros_like(rb)
    t0 = time.time()
    for k in range(trials):
        eng = LIFEngine(indptr, indices, w, seed=2000 + k)
        eng.set_poisson(sugar, RATE)
        eng.run_seconds(1.0)
        re_[k] = eng.counts
    t_eng = time.time() - t0
    mb, me = rb.mean(0), re_.mean(0)
    sem = np.sqrt(rb.var(0, ddof=1) / trials + re_.var(0, ddof=1) / trials) + 1e-9
    act = (mb > 0) | (me > 0)
    z = np.abs(mb - me)[act] / np.maximum(sem[act], 0.5)
    mn9 = tab.index_of(MN9_IDS[:1])[0]
    top = np.argsort(-mb)[:30]
    out = {
        "trials": trials,
        "active_brian2": int((mb > 0).sum()), "active_engine": int((me > 0).sum()),
        "pearson_r_active": float(np.corrcoef(mb[act], me[act])[0, 1]),
        "frac_within_3sem": float(np.mean(z < 3)),
        "mn9_brian2": [float(mb[mn9]), float(rb[:, mn9].std())],
        "mn9_engine": [float(me[mn9]), float(re_[:, mn9].std())],
        "total_spikes_per_trial": [float(rb.sum(1).mean()), float(re_.sum(1).mean())],
        "engine_wall_s": t_eng,
    }
    print(f"validation ({trials} x 1 s, sugar GRNs @ {RATE:.0f} Hz):")
    print(f"  active neurons  brian2 {out['active_brian2']}  engine {out['active_engine']}")
    print(f"  MN9 rate        brian2 {mb[mn9]:.1f} +/- {rb[:, mn9].std():.1f} Hz   "
          f"engine {me[mn9]:.1f} +/- {re_[:, mn9].std():.1f} Hz")
    print(f"  Pearson r over active neurons {out['pearson_r_active']:.4f}; "
          f"{100 * out['frac_within_3sem']:.1f}% of active neurons within 3 SEM")
    print("  top 30 (brian2 rate vs engine rate, Hz):")
    ct = tab.col("cell_type")
    for i in top:
        print(f"    {tab.root_id[i]} {ct[i]:14s} {mb[i]:7.1f} {me[i]:7.1f}")
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brian2", action="store_true", help="benchmark Brian2 runtime mode")
    ap.add_argument("--standalone", action="store_true", help="benchmark Brian2 cpp_standalone")
    ap.add_argument("--validate", type=int, default=0, metavar="TRIALS")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--trials", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--workdir", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not data_available():
        print("data/brain missing: run scripts/fetch_brain_data.py first")
        return 1
    if args.child:
        res = _brian2_child(args.child, args.seconds, args.trials, args.workdir)
        print(json.dumps(res))
        return 0
    results = {"engine": bench_engine(args.seconds)}
    if args.brian2:
        results["brian2_runtime"] = run_brian2("runtime", seconds=args.seconds)
    if args.standalone:
        import shutil
        wd = tempfile.mkdtemp(prefix="brian2_sa_")
        try:
            results["brian2_standalone"] = run_brian2("standalone", seconds=1.0, workdir=wd)
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    if args.validate:
        results["validation"] = validate(args.validate)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
