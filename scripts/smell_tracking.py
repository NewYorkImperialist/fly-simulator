"""Does a lateralised odour steer the brain model? (docs/SMELL.md, "Odour tracking")

1. ``--open-loop`` (default): with a smell fix, present an odour to the left
   antenna only, the right only, both, and small left/right differences; record
   the turn DNs (DNa01 / DNa02 per side, mapping.DESCENDING_TYPES "turn"), the
   walk DNs and the descending neurons with the largest left/right difference.
2. ``--closed-loop``: a kinematic point fly (constant speed; turn rate from the
   brain's turn_L / turn_R via mapping.descending_to_drive, the app's
   --brain-steer mapping) in the plume of fly_simulator/senses/plume.py, started
   at several headings; compares the approach to an attractive odour with a
   control odour and with no odour. In-process engine, one process.

    .venv/bin/python scripts/smell_tracking.py --fix sign_adapt --trials 20
    .venv/bin/python scripts/smell_tracking.py --closed-loop --odors vinegar glom_DL5 none
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load(fix: str):
    from fly_simulator.brain.data import load_connectome, load_neuron_table
    from fly_simulator.brain.engine import LIFEngine
    from fly_simulator.brain.smell import SmellFix

    t = load_neuron_table()
    e = LIFEngine(*load_connectome()[:3], seed=0)
    e.run(1)
    SmellFix(t, e, fix)
    return t, e


def open_loop(t, e, odors, trials: int, seconds: float, conds) -> None:
    from fly_simulator.brain.mapping import descending_indices
    from fly_simulator.brain.smell import odor_drive

    dn = descending_indices(t)
    sc, ct, side = t.col("super_class"), t.col("cell_type"), t.col("side")
    dns = np.nonzero(sc == "descending")[0]
    for odor in odors:
        print(f"\n== {odor}  ({trials} trials x {seconds} s)")
        per = {}
        for cl, cr in conds:
            rows = []
            for _ in range(trials):
                e.reset_state()
                idx, rates = odor_drive(t, odor, cl, cr)
                e.set_poisson(idx, rates)
                c0 = e.counts.copy()
                e.run_seconds(seconds)
                e.set_poisson(np.zeros(0, np.int64), 0.0)
                rows.append((e.counts - c0) / seconds)
            R = np.array(rows)
            per[(cl, cr)] = R
            tl, tr = R[:, dn["turn_L"]].mean(1), R[:, dn["turn_R"]].mean(1)
            wl, wr = R[:, dn["walk_L"]].mean(1), R[:, dn["walk_R"]].mean(1)
            d = tl - tr
            se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else 0.0
            print(f"  L {cl:4.2f} R {cr:4.2f}: turn_L {tl.mean():6.2f} turn_R {tr.mean():6.2f} "
                  f"(L-R {d.mean():+6.2f} +/- {se:4.2f})  walk {wl.mean():5.1f}/{wr.mean():5.1f}  "
                  f"total {R.sum(1).mean():8.0f} sp/s")
        # descending neurons that differ most between left-only and right-only
        keys = list(per)
        if len(keys) >= 2:
            a, b = per[keys[0]].mean(0)[dns], per[keys[1]].mean(0)[dns]
            order = np.argsort(-np.abs(a - b))[:8]
            print("  DNs with the largest difference", keys[0], "vs", keys[1], ":",
                  ", ".join(f"{ct[dns[i]]}_{str(side[dns[i]])[:1].upper()} {a[i]:.0f}/{b[i]:.0f}"
                            for i in order if abs(a[i] - b[i]) >= 1))


def closed_loop(t, e, odors, headings, seconds: float, speed_mm_s: float,
                turn_gain_rad_s: float, contrast_gain: float, seed: int) -> None:
    from fly_simulator.brain.mapping import DriveGains, descending_indices
    from fly_simulator.brain.smell import odor_drive
    from fly_simulator.senses.plume import OdorPlume, PlumeConfig

    dn = descending_indices(t)
    g = DriveGains()
    dt = 0.05  # brain chunk = control step
    rng = np.random.default_rng(seed)
    print(f"\nclosed loop: {seconds} s, speed {speed_mm_s} mm/s, turn gain {turn_gain_rad_s} "
          f"rad/s, contrast_gain {contrast_gain}; source at (20, 0), start at the origin")
    for odor in odors:
        pl = OdorPlume(PlumeConfig(odor=odor if odor != "none" else "vinegar", source_x_mm=20.0,
                                   source_y_mm=0.0, contrast_gain=contrast_gain,
                                   meander_mm=1.0))
        finals, turns = [], []
        for h0 in headings:
            x, y, hd = 0.0, 0.0, math.radians(h0)
            e.reset_state()
            tsum = []
            for k in range(int(round(seconds / dt))):
                tt = k * dt
                cl, cr = pl.bilateral(x, y, hd, tt)[pl.cfg.odor]
                if odor == "none":
                    cl = cr = 0.0
                idx, rates = odor_drive(t, pl.cfg.odor, cl, cr)
                e.set_poisson(idx, rates)
                c0 = e.counts.copy()
                e.run_seconds(dt)
                r = (e.counts - c0) / dt
                act = lambda i: math.tanh(max(float(r[i].mean()), 0.0) / g.r_ref)
                turn = act(dn["turn_L"]) - act(dn["turn_R"])  # >0: steer left
                tsum.append(turn)
                hd += (turn_gain_rad_s * turn + rng.normal(0, 0.3)) * dt  # + heading noise
                x += speed_mm_s * math.cos(hd) * dt
                y += speed_mm_s * math.sin(hd) * dt
            _, sx, sy = pl.sources(seconds)[0]
            finals.append(math.hypot(x - sx, y - sy))
            turns.append(float(np.mean(np.abs(tsum))))
        print(f"  {odor:<10s} final distance to source (mm): mean {np.mean(finals):5.2f} "
              f"[{', '.join(f'{d:.1f}' for d in finals)}]  mean |turn| {np.mean(turns):.3f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fix", default="sign_adapt")
    ap.add_argument("--odors", nargs="*", default=["vinegar", "glom_DL5"])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seconds", type=float, default=0.3)
    ap.add_argument("--closed-loop", action="store_true")
    ap.add_argument("--loop-seconds", type=float, default=8.0)
    ap.add_argument("--headings", type=float, nargs="*", default=[-60, -30, 30, 60])
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--turn-gain", type=float, default=3.0)
    ap.add_argument("--contrast-gain", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    t, e = load(a.fix)
    if a.closed_loop:
        closed_loop(t, e, a.odors, a.headings, a.loop_seconds, a.speed, a.turn_gain,
                    a.contrast_gain, a.seed)
    else:
        conds = [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.8), (0.8, 1.0)]
        open_loop(t, e, a.odors, a.trials, a.seconds, conds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
