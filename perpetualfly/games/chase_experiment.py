"""Does the connectome brain follow the leader better than chance? (docs/GAMES.md, game 2)

Paired trials: for trial i the leader's start bearing (alternating sides), its
heading offset, its weaving path (the OU noise seed) and the follower's gait phase
are drawn once from ``seed`` and replayed under every control condition:

* ``brain``: left eye -> left LC10a, right eye -> right LC10a;
* ``mirror``: left eye -> right LC10a and vice versa (same brain, same motor map);
* ``none``: brain disconnected, constant drive [1, 1] (the chance / geometry baseline).

Each trial: reset the fly and the brain (all neurons at rest), walk ``walk_s`` (+ a
random fraction of a gait cycle) with the leader hidden, place the leader
``start_dist_mm`` ahead at the trial's bearing, then run ``trial_s`` of game time
(no lives: outcomes are only measured). Per trial: the fraction of time within the
following distance, mean distance and target error, catches, whether (and when) the
leader was lost, the initial turn toward the leader's side, and DN / LC10a peaks.
One brain process serves all conditions (``GameBrain.set_control``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from perpetualfly.games.experiment import _binom_two_sided


@dataclass
class ChaseTrialSpec:
    i: int
    bearing_deg: float  # + = leader starts on the follower's left
    yaw_offset_deg: float
    path_seed: int
    phase_s: float


def make_chase_specs(n: int, seed: int = 0, bearing_range=(10.0, 35.0)) -> list[ChaseTrialSpec]:
    out = []
    for i in range(n):
        r = np.random.default_rng(seed * 100003 + 7919 * i + 1)
        sign = 1.0 if i % 2 == 0 else -1.0  # balanced sides
        out.append(ChaseTrialSpec(i, sign * float(r.uniform(*bearing_range)),
                                  float(r.uniform(-20.0, 20.0)), int(r.integers(1 << 30)),
                                  float(r.uniform(0.0, 0.0833))))
    return out


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def run_chase_trial(s, spec: ChaseTrialSpec, control: str, walk_s: float = 0.5,
                    trial_s: float = 6.0, initial_s: float = 0.5) -> dict:
    """One leader run under ``control`` on a ``ChaseSession`` ``s``."""
    g, sim, brain, lead = s.game, s.sim, s.brain, s.leader
    c = g.cfg
    brain.set_control(control)
    g._reset_fly()
    brain.reset(g.time())
    g._new_game()
    g.state = "trial"
    lead.hide()
    t_end = g.time() + walk_s + spec.phase_s
    while g.time() < t_end:
        s.step()
    g.catches = 0
    g.losses = 0
    g.follow_s = g.play_s = 0.0
    g.place_leader(c.start_dist_mm, spec.bearing_deg, spec.yaw_offset_deg, seed=spec.path_seed)
    lead.weave = True
    g._last_t = g.time()
    h0 = sim.heading()
    t0 = g.time()
    side = 1.0 if spec.bearing_deg > 0 else -1.0
    dists, errs = [], []
    h_initial = None
    t_lost = None
    toward_int = 0.0
    peak = {"turn_L": 0.0, "turn_R": 0.0, "walk": 0.0, "backward": 0.0, "escape": 0.0}
    lc10 = {"left": 0.0, "right": 0.0}
    seen_states = brain.n_states
    while g.time() - t0 < trial_s:
        s.step()
        t = g.time() - t0
        dists.append(g.dist)
        errs.append(abs(g.error_deg))
        if h_initial is None and t >= initial_s:
            h_initial = sim.heading()
        if t_lost is None and g.losses > 0:
            t_lost = t
        if brain.n_states != seen_states:
            seen_states = brain.n_states
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
    d = np.asarray(dists)
    dh0 = _wrap((h_initial if h_initial is not None else sim.heading()) - h0)
    row = {
        "trial": spec.i, "control": control, "bearing_deg": round(spec.bearing_deg, 2),
        "leader_side": "left" if side > 0 else "right",
        "follow_frac": round(float(np.mean(d < c.follow_dist_mm)), 4),
        "mean_dist_mm": round(float(np.mean(np.minimum(d, 30.0))), 3),
        "final_dist_mm": round(float(d[-1]), 3),
        "mean_abs_err_deg": round(float(np.mean(errs)), 2),
        "catches": int(g.catches), "lost": g.losses > 0,
        "t_lost_s": None if t_lost is None else round(t_lost, 3),
        # + = turned toward the side the leader started on, over the first initial_s
        "initial_turn_toward_deg": round(side * math.degrees(dh0), 1),
        "initial_toward_dn_hz_s": round(toward_int, 2),
        "peak_turn_L": round(peak["turn_L"], 1), "peak_turn_R": round(peak["turn_R"], 1),
        "peak_walk": round(peak["walk"], 1), "peak_mdn": round(peak["backward"], 1),
        "peak_gf": round(peak["escape"], 1),
        "lc10a_peak_left": round(lc10["left"], 1), "lc10a_peak_right": round(lc10["right"], 1),
    }
    lead.hide()
    return row


def run_chase_experiment(s, n: int, controls=("brain", "mirror", "none"), seed: int = 0,
                         say=print, trial_s: float = 6.0) -> list[dict]:
    specs = make_chase_specs(n, seed)
    rows = []
    t0 = time.time()
    for spec in specs:
        for c in controls:
            t1 = time.time()
            r = run_chase_trial(s, spec, c, trial_s=trial_s)
            r["wall_s"] = round(time.time() - t1, 2)
            rows.append(r)
            say(f"trial {spec.i:3d} {c:6s} bearing {spec.bearing_deg:+5.1f} follow "
                f"{r['follow_frac']:.2f} dist {r['mean_dist_mm']:5.1f} err "
                f"{r['mean_abs_err_deg']:5.1f} catches {r['catches']} lost {int(r['lost'])} "
                f"turn0 {r['initial_turn_toward_deg']:+6.1f} DN L/R {r['peak_turn_L']:4.0f}/"
                f"{r['peak_turn_R']:4.0f} ({r['wall_s']:.1f}s, total {time.time() - t0:.0f}s)")
    return rows


def _wilcoxon_p(x: list[float]) -> float | None:
    try:
        from scipy.stats import wilcoxon
    except ImportError:  # pragma: no cover
        return None
    x = [v for v in x if v != 0]
    if len(x) < 2:
        return None
    return float(f"{wilcoxon(x).pvalue:.3g}")


def _se(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")


def summarize_chase(rows: list[dict]) -> dict:
    order = ["brain", "mirror", "none"]
    controls = sorted({r["control"] for r in rows}, key=order.index)
    by = {c: {r["trial"]: r for r in rows if r["control"] == c} for c in controls}
    out = {"conditions": {}, "paired": {}}
    for c in controls:
        rs = list(by[c].values())
        ff = [r["follow_frac"] for r in rs]
        turn = [r["initial_turn_toward_deg"] for r in rs]
        k = sum(x > 0 for x in turn)
        nz = sum(x != 0 for x in turn)
        out["conditions"][c] = {
            "n": len(rs),
            "follow_frac": round(float(np.mean(ff)), 3), "follow_frac_se": round(_se(ff), 3),
            "mean_dist_mm": round(float(np.mean([r["mean_dist_mm"] for r in rs])), 2),
            "mean_abs_err_deg": round(float(np.mean([r["mean_abs_err_deg"] for r in rs])), 1),
            "catches": int(sum(r["catches"] for r in rs)),
            "catches_per_trial": round(float(np.mean([r["catches"] for r in rs])), 2),
            "lost": int(sum(r["lost"] for r in rs)),
            "initial_turn_toward": f"{k}/{nz}",
            "initial_turn_toward_p": float(f"{_binom_two_sided(k, nz):.3g}") if nz else 1.0,
            "mean_initial_turn_toward_deg": round(float(np.mean(turn)), 1),
        }
    for a, b in (("brain", "none"), ("brain", "mirror"), ("mirror", "none")):
        if a not in by or b not in by:
            continue
        common = sorted(set(by[a]) & set(by[b]))
        diffs = [by[a][i]["follow_frac"] - by[b][i]["follow_frac"] for i in common]
        pos = sum(x > 0 for x in diffs)
        nz = sum(x != 0 for x in diffs)
        lost_only_a = sum(by[a][i]["lost"] and not by[b][i]["lost"] for i in common)
        lost_only_b = sum(by[b][i]["lost"] and not by[a][i]["lost"] for i in common)
        out["paired"][f"{a}_vs_{b}"] = {
            "n": len(common),
            "mean_follow_diff": round(float(np.mean(diffs)), 3) if diffs else None,
            "first_follows_more": f"{pos}/{nz}",
            "sign_test_p": float(f"{_binom_two_sided(pos, nz):.3g}") if nz else 1.0,
            "wilcoxon_p": _wilcoxon_p(diffs),
            f"lost_only_{a}": lost_only_a, f"lost_only_{b}": lost_only_b,
            "mcnemar_lost_p": float(f"{_binom_two_sided(lost_only_a, lost_only_a + lost_only_b):.3g}")
            if lost_only_a + lost_only_b else 1.0,
        }
    return out


def format_chase_summary(summary: dict) -> str:
    lines = ["| condition | trials | time following (< follow dist) | mean distance (mm) | "
             "mean target error (deg) | catches (per trial) | lost | initial turn toward the "
             "leader (p) | mean initial turn toward (deg) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c, v in summary["conditions"].items():
        lines.append(f"| {c} | {v['n']} | {v['follow_frac']:.2f} +- {v['follow_frac_se']:.2f} | "
                     f"{v['mean_dist_mm']:.1f} | {v['mean_abs_err_deg']:.1f} | {v['catches']} "
                     f"({v['catches_per_trial']:.2f}) | {v['lost']}/{v['n']} | "
                     f"{v['initial_turn_toward']} (p={v['initial_turn_toward_p']}) | "
                     f"{v['mean_initial_turn_toward_deg']:+.1f} |")
    lines += ["", "| comparison | pairs | follow-time diff | first follows longer | sign test p | "
                  "Wilcoxon p | lost only in first | lost only in second | McNemar p |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k, v in summary["paired"].items():
        a, b = k.split("_vs_")
        lines.append(f"| {a} vs {b} | {v['n']} | {v['mean_follow_diff']:+.3f} | "
                     f"{v['first_follows_more']} | {v['sign_test_p']} | {v['wilcoxon_p']} | "
                     f"{v[f'lost_only_{a}']} | {v[f'lost_only_{b}']} | {v['mcnemar_lost_p']} |")
    return "\n".join(lines)
