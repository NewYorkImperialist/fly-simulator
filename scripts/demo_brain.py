#!/usr/bin/env python
"""Headless brain demo: start the brain process, poke it, print BrainState summaries.

    .venv/bin/python scripts/demo_brain.py
    .venv/bin/python scripts/demo_brain.py --synthetic     # tiny random network, no data

Sequence: 0.5 s quiet -> left whip hit (0.1 s) -> 1 s -> sugar GRNs (0.5 s)
-> 1 s -> LC4 looming neurons (0.3 s, drives the giant fiber) -> 1 s.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_simulator.brain.mapping import descending_to_drive  # noqa: E402
from fly_simulator.brain.process import BrainConfig, BrainProcess  # noqa: E402
from fly_simulator.brain.schema import NEUROTRANSMITTERS, StimulusEvent  # noqa: E402


def summary(st, regions) -> str:
    nt = " ".join(f"{n}={r:5.2f}" for n, r in zip(NEUROTRANSMITTERS, st.rate_by_nt))
    top = np.argsort(-st.rate_by_region)[:3]
    reg = ", ".join(f"{regions[i]}={st.rate_by_region[i]:.2f}" for i in top if st.rate_by_region[i] > 0)
    dn = " ".join(f"{k}={v:.0f}" for k, v in st.descending.items() if v > 0) or "-"
    drv = descending_to_drive(st)
    stim = ",".join(st.recent_stimuli)
    return (f"t={st.brain_time:6.2f}s rtf={st.realtime_factor:5.2f} spikes={st.total_spikes:6d} "
            f"raster={len(st.raster_idx):5d} | Hz {nt} | top {reg or '-'} | DN {dn} | "
            f"drive [{drv[0]:+.2f},{drv[1]:+.2f}] {('<- ' + stim) if stim else ''}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--window", type=float, default=0.1, help="brain s per BrainState")
    args = ap.parse_args()
    cfg = BrainConfig(window_s=args.window, subscribers=("app",), queue_size=64,
                      synthetic={"n": 200, "p_conn": 0.05, "seed": 0} if args.synthetic else None)
    t0 = time.time()
    with BrainProcess(cfg) as bp:
        info = bp.wait_ready(120)
        lay = bp.layout(10)
        print(f"brain ready in {time.time() - t0:.1f} s: {info}")
        print(f"layout: {lay.model_name}, {lay.n_neurons_total} neurons, {len(lay.regions)} regions, "
              f"{len(lay.display_neuron_ids)} display neurons")

        def run_for(seconds):
            end = time.time() + seconds
            while time.time() < end:
                for st in bp.poll():
                    print(summary(st, lay.regions))
                time.sleep(0.05)

        run_for(0.5)
        print("--> whip hit, left side, intensity 1.0")
        bp.send(StimulusEvent("whip_hit", "left", 1.0, 0.1, details={"body": "Thorax"}))
        run_for(1.0)
        print("--> manual: sugar GRNs (Shiu et al. set) 0.5 s")
        bp.send(StimulusEvent("manual", "none", 1.0, 0.5, details={"set": "sugar"}))
        run_for(1.0)
        print("--> manual: LC4 looming detectors 0.3 s")
        bp.send(StimulusEvent("manual", "none", 1.0, 0.3, details={"set": "LC4"}))
        run_for(1.0)
    print("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
