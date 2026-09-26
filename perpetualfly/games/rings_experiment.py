"""Does the connectome brain fly through rings better than chance? (docs/GAMES.md, game 3)

Paired single-ring trials: for trial i the ring's lateral offset (alternating
sides, magnitude drawn from ``offset_range``) and a settling-time jitter (a
fraction of the wingbeat / brain window alignment) are drawn once from ``seed`` and
replayed under every control condition:

* ``brain``: left eye -> left LC10a, right eye -> right LC10a;
* ``mirror``: left eye -> right LC10a and vice versa (same brain, same motor map);
* ``none``: brain disconnected, heading rate 0 (straight flight: the geometry
  baseline).

Each trial: reset the model and the brain (all neurons at rest), air start at the
origin heading +x, fly straight ``settle_s`` (+ jitter) with no ring, then one ring
appears ``first_dist_mm`` ahead at the trial's lateral offset; the trial ends when
the fly crosses the ring plane (through / missed), turns away, times out or
crashes. One brain process serves all conditions (``GameBrain.set_control``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from perpetualfly.games.chase_experiment import _se, _wilcoxon_p
from perpetualfly.games.experiment import _binom_two_sided


@dataclass
class RingTrialSpec:
    i: int
    offset_mm: float  # + = ring to the fly's left
    jitter_s: float


def make_ring_specs(n: int, seed: int = 0, offset_range=(3.5, 7.0)) -> list[RingTrialSpec]:
    out = []
    for i in range(n):
        r = np.random.default_rng(seed * 100003 + 7919 * i + 5)
        sign = 1.0 if i % 2 == 0 else -1.0  # balanced sides
        out.append(RingTrialSpec(i, sign * float(r.uniform(*offset_range)),
                                 float(r.uniform(0.0, 0.02))))
    return out


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def run_ring_trial(s, spec: RingTrialSpec, control: str, settle_s: float = 0.5,
                   initial_s: float = 0.3) -> dict:
    """One ring under ``control`` on a ``RingsSession`` ``s``."""
    g, sim, brain, pilot = s.game, s.sim, s.brain, s.pilot
    brain.set_control(control)
    g._reset_fly()
    s._crash_count = s.flight.counts.get("crash", 0)
    g.course.hide_all()
    g.state = "settle"
    g.last_outcome = None
    g.passed = g.missed = g.crashes = 0
    pilot.launch((0.0, 0.0), 0.0)
    brain.reset(g.time())
    s._last_rt = g.time()
    t_end = g.time() + settle_s + spec.jitter_s
    crashed_early = False
    while g.time() < t_end:
        s.step()
        if g.crashes or s.flight.state == "walking":
            crashed_early = True
            break
    side = 1.0 if spec.offset_mm > 0 else -1.0
    row = {"trial": spec.i, "control": control, "offset_mm": round(spec.offset_mm, 2),
           "ring_side": "left" if side > 0 else "right"}
    if crashed_early:
        row.update(through=False, why="crash before ring", radial_mm=None, lateral_mm=None)
        return row
    g.state = "trial"
    g.start_course(first_offset_mm=spec.offset_mm, n=1)
    y0 = pilot.true_yaw()
    t0 = g.time()
    yaw_initial = None
    toward_int = 0.0
    peak = {"turn_L": 0.0, "turn_R": 0.0, "walk": 0.0, "backward": 0.0, "escape": 0.0}
    lc10 = {"left": 0.0, "right": 0.0}
    max_yaw = 0.0
    seen = brain.n_states
    while g.last_outcome is None and g.time() - t0 < 3.0:
        s.step()
        t = g.time() - t0
        if yaw_initial is None and t >= initial_s:
            yaw_initial = pilot.true_yaw()
        max_yaw = max(max_yaw, abs(math.degrees(_wrap(pilot.true_yaw() - y0))))
        if brain.n_states != seen:
            seen = brain.n_states
            r = brain.rates
            for k in ("turn_L", "turn_R", "escape"):
                peak[k] = max(peak[k], float(r.get(k, 0.0)))
            peak["walk"] = max(peak["walk"], 0.5 * (r.get("walk_L", 0.0) + r.get("walk_R", 0.0)))
            peak["backward"] = max(peak["backward"],
                                   0.5 * (r.get("backward_L", 0.0) + r.get("backward_R", 0.0)))
            if t <= initial_s:
                toward_int += side * (r.get("turn_L", 0.0) - r.get("turn_R", 0.0)) * brain.window_s
        for e, v in brain.pursuit_drive(g.time()).items():
            lc10[e] = max(lc10[e], v)
    o = g.last_outcome or {"through": False, "why": "no outcome"}
    dy = _wrap((yaw_initial if yaw_initial is not None else pilot.true_yaw()) - y0)
    row.update({
        "through": bool(o.get("through")), "why": o.get("why"),
        "radial_mm": o.get("radial_mm"), "lateral_mm": o.get("lateral_mm"),
        "vertical_mm": o.get("vertical_mm"),
        "t_cross_s": round(g.time() - t0, 3),
        # + = turned toward the ring's side over the first initial_s
        "initial_turn_toward_deg": round(side * math.degrees(dy), 1),
        "initial_toward_dn_hz_s": round(toward_int, 2),
        "max_abs_yaw_deg": round(max_yaw, 1),
        "peak_turn_L": round(peak["turn_L"], 1), "peak_turn_R": round(peak["turn_R"], 1),
        "peak_walk": round(peak["walk"], 1), "peak_mdn": round(peak["backward"], 1),
        "peak_gf": round(peak["escape"], 1),
        "lc10a_peak_left": round(lc10["left"], 1), "lc10a_peak_right": round(lc10["right"], 1),
    })
    g.course.hide_all()
    return row


def run_rings_experiment(s, n: int, controls=("brain", "mirror", "none"), seed: int = 0,
                         say=print, settle_s: float = 0.5) -> list[dict]:
    specs = make_ring_specs(n, seed)
    rows = []
    t0 = time.time()
    for spec in specs:
        for c in controls:
            t1 = time.time()
            r = run_ring_trial(s, spec, c, settle_s=settle_s)
            r["wall_s"] = round(time.time() - t1, 2)
            rows.append(r)
            rad = r.get("radial_mm")
            say(f"trial {spec.i:3d} {c:6s} offset {spec.offset_mm:+5.2f} "
                f"{'THROUGH' if r['through'] else 'miss   '} ({r.get('why')}) radial "
                f"{rad if rad is not None else '-':>5} turn0 "
                f"{r.get('initial_turn_toward_deg', 0.0):+6.1f} DN L/R "
                f"{r.get('peak_turn_L', 0):4.0f}/{r.get('peak_turn_R', 0):4.0f} "
                f"({r['wall_s']:.1f}s, total {time.time() - t0:.0f}s)")
    return rows


def summarize_rings(rows: list[dict]) -> dict:
    order = ["brain", "mirror", "none"]
    controls = sorted({r["control"] for r in rows}, key=order.index)
    by = {c: {r["trial"]: r for r in rows if r["control"] == c} for c in controls}
    out = {"conditions": {}, "paired": {}}
    for c in controls:
        rs = list(by[c].values())
        k_thr = sum(bool(r["through"]) for r in rs)
        rad = [r["radial_mm"] for r in rs if r.get("radial_mm") is not None]
        turn = [r.get("initial_turn_toward_deg", 0.0) for r in rs]
        k = sum(x > 0 for x in turn)
        nz = sum(x != 0 for x in turn)
        out["conditions"][c] = {
            "n": len(rs), "through": k_thr,
            "pass_rate": round(k_thr / len(rs), 3) if rs else 0.0,
            "mean_radial_mm": round(float(np.mean(rad)), 2) if rad else None,
            "radial_se": round(_se(rad), 2) if len(rad) > 1 else None,
            "crashes": sum(1 for r in rs if str(r.get("why", "")).startswith("crash")),
            "turned_away": sum(1 for r in rs if r.get("why") == "turned away"),
            "initial_turn_toward": f"{k}/{nz}",
            "initial_turn_toward_p": float(f"{_binom_two_sided(k, nz):.3g}") if nz else 1.0,
            "mean_initial_turn_toward_deg": round(float(np.mean(turn)), 1) if turn else 0.0,
        }
    for a, b in (("brain", "none"), ("brain", "mirror"), ("mirror", "none")):
        if a not in by or b not in by:
            continue
        common = sorted(set(by[a]) & set(by[b]))
        only_a = sum(by[a][i]["through"] and not by[b][i]["through"] for i in common)
        only_b = sum(by[b][i]["through"] and not by[a][i]["through"] for i in common)
        diffs = [by[b][i]["radial_mm"] - by[a][i]["radial_mm"] for i in common
                 if by[a][i].get("radial_mm") is not None
                 and by[b][i].get("radial_mm") is not None]
        closer = sum(x > 0 for x in diffs)
        out["paired"][f"{a}_vs_{b}"] = {
            "n": len(common), f"through_only_{a}": only_a, f"through_only_{b}": only_b,
            "mcnemar_p": float(f"{_binom_two_sided(only_a, only_a + only_b):.3g}")
            if only_a + only_b else 1.0,
            "first_closer": f"{closer}/{sum(x != 0 for x in diffs)}",
            "mean_radial_diff_mm": round(float(np.mean(diffs)), 2) if diffs else None,
            "wilcoxon_p": _wilcoxon_p(diffs),
        }
    return out


def format_rings_summary(summary: dict) -> str:
    lines = ["| condition | trials | through the ring | mean distance from the ring centre at "
             "the ring plane (mm) | turned away / crashed | initial turn toward the ring (p) | "
             "mean initial turn toward (deg) |",
             "|---|---|---|---|---|---|---|"]
    for c, v in summary["conditions"].items():
        rad = "-" if v["mean_radial_mm"] is None else (
            f"{v['mean_radial_mm']:.2f}" + (f" +- {v['radial_se']:.2f}" if v["radial_se"] else ""))
        lines.append(f"| {c} | {v['n']} | {v['through']}/{v['n']} ({100 * v['pass_rate']:.0f}%) | "
                     f"{rad} | {v['turned_away']} / {v['crashes']} | "
                     f"{v['initial_turn_toward']} (p={v['initial_turn_toward_p']}) | "
                     f"{v['mean_initial_turn_toward_deg']:+.1f} |")
    lines += ["", "| comparison | pairs | through only in first | through only in second | "
                  "McNemar p | first closer to the centre | mean distance diff (mm) | Wilcoxon p |",
              "|---|---|---|---|---|---|---|---|"]
    for k, v in summary["paired"].items():
        a, b = k.split("_vs_")
        lines.append(f"| {a} vs {b} | {v['n']} | {v[f'through_only_{a}']} | "
                     f"{v[f'through_only_{b}']} | {v['mcnemar_p']} | {v['first_closer']} | "
                     f"{v['mean_radial_diff_mm']} | {v['wilcoxon_p']} |")
    return "\n".join(lines)
