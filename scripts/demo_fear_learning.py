"""Offline fear-conditioning experiment on the whole-brain model (docs/FEAR_LEARNING.md).

Runs the LIF engine in-process with the KC -> MBON plasticity add-on
(fly_simulator/brain/plasticity.py):

  1. pre-test: odour A and odour B alone (KC code, Poisson) -> MBON / DN / DAN rates
  2. training: ``--trials`` x [odour A + punishment] interleaved with odour B alone.
     Punishment = Poisson drive of the PPL1 punishment DANs (a stand-in for the
     whip's nociceptive signal; see ``--whip-check`` for whether the modelled whip
     afferents reach PPL1 by themselves)
  3. post-test: odour A and B again (plasticity frozen during tests)

Prints effective KC -> MBON efficacies per compartment for A's and B's KCs,
MBON responses before / after and descending-neuron rates. ``--seeds`` repeats
with different engine seeds (Poisson streams). Small JSON with ``--out``.

    .venv/bin/python scripts/demo_fear_learning.py --seeds 0 1 2
    .venv/bin/python scripts/demo_fear_learning.py --whip-check --seeds 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_simulator.brain.data import load_connectome, load_neuron_table  # noqa: E402
from fly_simulator.brain.engine import LIFEngine  # noqa: E402
from fly_simulator.brain.mapping import descending_indices, named_sets  # noqa: E402
from fly_simulator.brain.plasticity import KCMBONPlasticity, PlasticityConfig  # noqa: E402

CHUNK_S = 0.02


def run(engine, plast, drive: dict, seconds: float, learn: bool) -> tuple[np.ndarray, float]:
    """Drive {name: (idx, rate)} for ``seconds``; returns (rates per neuron, persist sps)."""
    idx = np.concatenate([v[0] for v in drive.values()]) if drive else np.zeros(0, np.int64)
    rates = np.concatenate([np.full(len(v[0]), v[1]) for v in drive.values()]) if drive else \
        np.zeros(0)
    engine.set_poisson(idx, rates)
    c0 = engine.counts.copy()
    n = int(round(seconds / CHUNK_S))
    for _ in range(n):
        _, i = engine.run_seconds(CHUNK_S)
        if learn:
            plast.observe(i, CHUNK_S)
    engine.set_poisson(np.zeros(0, np.int64), 0.0)
    r = (engine.counts - c0) / (n * CHUNK_S)
    # runaway check: spikes 50-100 ms after the input stops
    engine.run_seconds(0.05)
    _, after = engine.run_seconds(0.05)
    engine.reset_state()
    return r, len(after) / 0.05


def summarize(r, groups: dict) -> dict:
    return {k: float(r[v].mean()) if len(v) else 0.0 for k, v in groups.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--odor-hz", type=float, default=None, help="default: PlasticityConfig.odor_hz")
    ap.add_argument("--dan-hz", type=float, default=None, help="default: PlasticityConfig.punish_hz")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--train-s", type=float, default=1.0)
    ap.add_argument("--test-s", type=float, default=1.0)
    ap.add_argument("--iti-s", type=float, default=5.0,
                    help="quiet inter-trial interval (learning on; lets the KC traces decay)")
    ap.add_argument("--whip-check", action="store_true",
                    help="also measure PPL1 rates for the whip's modelled afferent sets")
    ap.add_argument("--lesion-check", action="store_true",
                    help="also measure DN rates to odour A with all punished-compartment MBONs silenced")
    ap.add_argument("--out", type=str, default=None, help="write a small JSON summary")
    a = ap.parse_args()

    t0 = time.time()
    table = load_neuron_table()
    ip, ix, w, _ = load_connectome()
    engine = LIFEngine(ip, ix, w, seed=a.seeds[0])
    del ip, ix, w
    plast = KCMBONPlasticity(table, engine, PlasticityConfig(enabled=True))
    a.odor_hz = plast.cfg.odor_hz if a.odor_hz is None else a.odor_hz
    a.dan_hz = plast.cfg.punish_hz if a.dan_hz is None else a.dan_hz
    base_silenced = plast.silenced_idx
    engine.set_silenced(base_silenced)
    print(f"silenced (runaway guard): {list(plast.cfg.silence)} = {len(base_silenced)} neurons")
    ct = table.col("cell_type").astype(str)
    print(f"loaded in {time.time() - t0:.1f} s; {len(plast.p)} plastic KC>MBON entries")
    for name, c in plast.comp.items():
        print(f"  compartment {name} ({len(c['dan'])} DANs, {c['kc_syn']} syn onto KCs) -> "
              + ", ".join(f"{t} ({c['mbon_syn'][t]})" for t in c["mbon_types"]))
    A, B = plast.odors["A"], plast.odors["B"]
    print(f"odour A {len(A)} KCs, B {len(B)} KCs, overlap {len(np.intersect1d(A, B))}")
    dan = plast.dan_idx
    mbon_groups = {t: np.nonzero(ct == t)[0] for c in plast.comp.values() for t in c["mbon_types"]}
    dn = descending_indices(table)
    dan_groups = {n: c["dan"] for n, c in plast.comp.items()}
    other_mbon = {t: np.nonzero(ct == t)[0] for t in sorted(set(ct[np.char.startswith(ct, "MBON")]))
                  if t not in mbon_groups}
    results = []
    for seed in a.seeds:
        engine._rng = np.random.default_rng(seed)
        plast.reset_memory()
        engine.reset_state()

        def test(odor):
            r, pers = run(engine, plast, {"odor": (plast.odors[odor], a.odor_hz)}, a.test_s, False)
            return {"mbon": summarize(r, mbon_groups), "mbon_other": summarize(r, other_mbon),
                    "dn": summarize(r, dn), "dan": summarize(r, dan_groups),
                    "persist_sps": pers,
                    "kc_other_active": float(np.mean(r[np.setdiff1d(plast.kc, plast.odors[odor])] > 2))}

        pre = {o: test(o) for o in ("A", "B")}
        eff0 = {o: plast.efficacy(plast.odors[o]) for o in ("A", "B")}
        train_dan, kc_spill = [], []
        notA = np.setdiff1d(plast.kc, A)
        for k in range(a.trials):
            r, _ = run(engine, plast, {"odor": (A, a.odor_hz), "dan": (dan, a.dan_hz)}, a.train_s, True)
            train_dan.append(summarize(r, dan_groups))
            kc_spill.append(float(np.mean(r[notA] > 2)))
            run(engine, plast, {}, a.iti_s, True)
            run(engine, plast, {"odor": (B, a.odor_hz)}, a.train_s, True)
            run(engine, plast, {}, a.iti_s, True)
        post = {o: test(o) for o in ("A", "B")}
        eff1 = {o: plast.efficacy(plast.odors[o]) for o in ("A", "B")}
        res = {"seed": seed, "pre": pre, "post": post, "eff_pre": eff0, "eff_post": eff1,
               "train_dan": train_dan[-1], "kc_spill": kc_spill}
        results.append(res)
        print(f"\n=== seed {seed} ===")
        print("efficacy (mean x of the odour's KC>MBON synapses) per compartment:")
        for comp in plast.comp_names:
            print(f"  {plast.compartment_label(comp):24s} A {eff0['A'][comp]:.2f} -> {eff1['A'][comp]:.2f}"
                  f"   B {eff0['B'][comp]:.2f} -> {eff1['B'][comp]:.2f}")
        print("MBON rate (Hz)            A pre -> post     B pre -> post")
        for t in mbon_groups:
            print(f"  {t:22s} {pre['A']['mbon'][t]:6.1f} -> {post['A']['mbon'][t]:6.1f}"
                  f"   {pre['B']['mbon'][t]:6.1f} -> {post['B']['mbon'][t]:6.1f}")
        moved = [t for t in other_mbon if max(pre['A']['mbon_other'][t], post['A']['mbon_other'][t],
                                               pre['B']['mbon_other'][t], post['B']['mbon_other'][t]) > 2]
        for t in moved:
            print(f"  (non-plastic) {t:12s} {pre['A']['mbon_other'][t]:6.1f} -> {post['A']['mbon_other'][t]:6.1f}"
                  f"   {pre['B']['mbon_other'][t]:6.1f} -> {post['B']['mbon_other'][t]:6.1f}")
        print("DN rate (Hz)              A pre -> post     B pre -> post")
        for g in dn:
            print(f"  {g:22s} {pre['A']['dn'][g]:6.1f} -> {post['A']['dn'][g]:6.1f}"
                  f"   {pre['B']['dn'][g]:6.1f} -> {post['B']['dn'][g]:6.1f}")
        print("DAN rate to odour alone (Hz): A " + ", ".join(f"{k} {v:.1f}" for k, v in pre['A']['dan'].items())
              + " | B " + ", ".join(f"{k} {v:.1f}" for k, v in pre['B']['dan'].items()))
        print("DAN rate during pairing (Hz): " + ", ".join(f"{k} {v:.0f}" for k, v in train_dan[-1].items())
              + f"; non-A KCs >2 Hz during pairing: {max(kc_spill):.3f}; runaway chunks {plast.n_runaway}")
        print(f"runaway check (persist sps): pre A {pre['A']['persist_sps']:.0f} B {pre['B']['persist_sps']:.0f}"
              f" post A {post['A']['persist_sps']:.0f} B {post['B']['persist_sps']:.0f};"
              f" other KCs >2 Hz: {pre['A']['kc_other_active']:.3f}")

    if a.whip_check:
        sets = named_sets(table)
        print("\nwhip afferents -> punishment DANs (0.5 s, no odour):")
        for name, rate in (("body_mech", 200.0), ("jo_wind_gravity", 100.0), ("an_arousal", 120.0),
                           ("an_walk", 150.0), ("head_bristle", 200.0)):
            if name not in sets:
                continue
            r, pers = run(engine, plast, {name: (sets[name], rate)}, 0.5, False)
            print(f"  {name:16s} @{rate:.0f} Hz: " + ", ".join(
                f"{k} {v:.1f}" for k, v in summarize(r, dan_groups).items())
                + f"  (persist {pers:.0f})")
        r, pers = run(engine, plast, {"relay": (np.concatenate([sets["an_arousal"], sets["an_walk"],
                                                                sets["body_mech"]]), 150.0)}, 0.5, False)
        print("  body+relay @150: " + ", ".join(f"{k} {v:.1f}" for k, v in summarize(r, dan_groups).items()))

    if a.lesion_check:
        plast.reset_memory()
        m11 = np.concatenate([c["mbon"] for c in plast.comp.values()])
        r0, _ = run(engine, plast, {"odor": (A, a.odor_hz)}, a.test_s, False)
        engine.set_silenced(np.concatenate([base_silenced, m11]))
        r1, _ = run(engine, plast, {"odor": (A, a.odor_hz)}, a.test_s, False)
        engine.set_silenced(base_silenced)
        print("\nodour A, DN rates intact -> punished-compartment MBONs silenced: " + ", ".join(
            f"{g} {r0[v].mean():.1f}->{r1[v].mean():.1f}" for g, v in dn.items()))
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
