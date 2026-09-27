"""Does the connectome brain fly around pillars better than chance? (docs/GAMES.md, game 5)

Paired pillar runs: trial i is a sequence of ``n_obstacles`` pillars whose lateral
offsets from the fly's straight path (|offset| drawn from ``offset_range``, the side
alternating along the run and between trials), radii (pool slot) and a settling
jitter are drawn once from ``seed`` and replayed under every control condition:

* ``brain``: left eye -> left LC4 / LPLC2, right eye -> right;
* ``mirror``: left eye -> right LC4 / LPLC2 and vice versa (same brain, same motor map);
* ``none``: brain disconnected, heading rate 0 (straight flight: the geometry
  baseline; every offset is inside the hit radius, so it always crashes into the
  first pillar).

Each trial: reset the model and the brain (all neurons at rest), air start at the
origin heading +x, fly straight ``settle_s`` (+ jitter) with no pillar; then the
first pillar appears ``spawn_dist_mm`` ahead along the fly's heading at its offset
from the projected straight path; each next pillar appears the same way (relative
to the fly's *current* position and heading) once the previous one is behind. The
trial ends at the first crash (a pillar hit or a flight crash) or after all
pillars. One brain process serves all conditions (``GameBrain.set_control``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from fly_simulator.games.chase_experiment import _se, _wilcoxon_p
from fly_simulator.games.experiment import _binom_two_sided


@dataclass
class CanyonTrialSpec:
    i: int
    offsets_mm: tuple  # per pillar, + = the pillar is left of the straight path
    slots: tuple  # pillar pool slot (radius) per pillar
    jitter_s: float


def make_canyon_specs(n: int, seed: int = 0, n_obstacles: int = 5,
                      offset_range=(1.5, 3.9), n_slots: int = 4) -> list[CanyonTrialSpec]:
    out = []
    for i in range(n):
        r = np.random.default_rng(seed * 100003 + 6151 * i + 3)
        offs, slots = [], []
        for k in range(n_obstacles):
            sign = 1.0 if (i + k) % 2 == 0 else -1.0  # balanced sides
            offs.append(round(sign * float(r.uniform(*offset_range)), 3))
            slots.append(int(r.integers(n_slots)))
        out.append(CanyonTrialSpec(i, tuple(offs), tuple(slots), float(r.uniform(0.0, 0.02))))
    return out


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def run_canyon_trial(s, spec: CanyonTrialSpec, control: str, settle_s: float = 0.4,
                     spawn_dist_mm: float = 30.0, obstacle_timeout_s: float = 3.0) -> dict:
    """One pillar run under ``control`` on a ``CanyonSession`` ``s``."""
    g, sim, brain, pilot = s.game, s.sim, s.brain, s.pilot
    brain.set_control(control)
    g._reset_fly()
    s._crash_count = s.flight.counts.get("crash", 0)
    g.field.park_all()
    g.state = "settle"
    g.passed = g.hits = g.crashes = g.near_misses = 0
    g.distance_mm = 0.0
    g.last_hit = None
    pilot.launch((0.0, 0.0), 0.0)
    brain.reset(g.time())
    s._last_rt = g.time()
    t_end = g.time() + settle_s + spec.jitter_s
    while g.time() < t_end and g.state == "settle":
        s.step()
    row = {"trial": spec.i, "control": control, "offsets_mm": list(spec.offsets_mm),
           "n_obstacles": len(spec.offsets_mm)}
    if g.state != "settle":
        row.update(passed=0, crashed=True, first_hit=True, why="crash before the pillars",
                   distance_mm=0.0, near_misses=0)
        return row
    g.state = "trial"
    g._prev_xy = None
    g.distance_mm = 0.0
    passed = 0
    why = "cleared"
    first: dict = {}
    for k, (off, slot) in enumerate(zip(spec.offsets_mm, spec.slots)):
        side = 1.0 if off > 0 else -1.0
        y0 = pilot.true_yaw()
        q = g.field.spawn_obstacle(sim.com()[:2], y0, spawn_dist_mm, off, slot=slot,
                                   t=g.time())
        q.budget_s = obstacle_timeout_s
        t0 = g.time()
        peak = {"turn_L": 0.0, "turn_R": 0.0, "escape": 0.0, "walk": 0.0, "backward": 0.0}
        loom = {"left": 0.0, "right": 0.0}
        away_int = 0.0
        seen = brain.n_states
        while q.outcome is None and g.state == "trial" and g.time() - t0 < obstacle_timeout_s + 0.5:
            s.step()
            if k == 0:
                if brain.n_states != seen:
                    seen = brain.n_states
                    r = brain.rates
                    for key in ("turn_L", "turn_R", "escape"):
                        peak[key] = max(peak[key], float(r.get(key, 0.0)))
                    peak["walk"] = max(peak["walk"],
                                       0.5 * (r.get("walk_L", 0.0) + r.get("walk_R", 0.0)))
                    peak["backward"] = max(peak["backward"], 0.5 * (
                        r.get("backward_L", 0.0) + r.get("backward_R", 0.0)))
                    # away = the DN group contralateral to the pillar's side
                    away_int += side * (r.get("turn_R", 0.0) - r.get("turn_L", 0.0)) \
                        * brain.window_s
                for e, (lc4, lp) in brain.eye_drive(g.time()).items():
                    loom[e] = max(loom[e], lc4, lp)
        if k == 0:
            dy = _wrap(pilot.true_yaw() - y0)
            first = {
                "first_offset_mm": off, "first_radius": q.radius,
                "first_hit": q.outcome == "hit" or g.state != "trial",
                "first_min_clear_mm": round(q.min_clear, 3),
                "first_heading_change_deg": round(math.degrees(dy), 1),
                # + = turned away from the pillar's side (pillar left -> turned right)
                "first_turn_away_deg": round(-side * math.degrees(dy), 1),
                "first_t_s": round(g.time() - t0, 3),
                "first_away_dn_hz_s": round(away_int, 2),
                "peak_turn_L": round(peak["turn_L"], 1), "peak_turn_R": round(peak["turn_R"], 1),
                "peak_gf": round(peak["escape"], 1), "peak_mdn": round(peak["backward"], 1),
                "peak_walk": round(peak["walk"], 1),
                "loom_peak_left": round(loom["left"], 1),
                "loom_peak_right": round(loom["right"], 1),
            }
        if q.outcome == "passed" and g.state == "trial":
            passed += 1
            continue
        why = "pillar" if q.outcome == "hit" else ("flight crash" if g.state != "trial"
                                                    else "timeout")
        break
    row.update(first)
    row.update(passed=passed, crashed=why in ("pillar", "flight crash"), why=why,
               distance_mm=round(g.distance_mm, 1), near_misses=g.near_misses,
               max_abs_yaw_deg=round(abs(math.degrees(_wrap(pilot.true_yaw()))), 1))
    g.field.park_all()
    g.state = "settle"
    return row


def run_canyon_experiment(s, n: int, controls=("brain", "mirror", "none"), seed: int = 0,
                          say=print, n_obstacles: int = 5, settle_s: float = 0.4) -> list[dict]:
    specs = make_canyon_specs(n, seed, n_obstacles=n_obstacles)
    rows = []
    t0 = time.time()
    for spec in specs:
        for c in controls:
            t1 = time.time()
            r = run_canyon_trial(s, spec, c, settle_s=settle_s)
            r["wall_s"] = round(time.time() - t1, 2)
            rows.append(r)
            say(f"trial {spec.i:3d} {c:6s} first {spec.offsets_mm[0]:+5.2f} "
                f"{'HIT ' if r.get('first_hit') else 'pass'} passed {r['passed']}/"
                f"{r['n_obstacles']} ({r['why']}) dist {r['distance_mm']:6.1f} mm near "
                f"{r['near_misses']} turn_away {r.get('first_turn_away_deg', 0.0):+6.1f} DN L/R "
                f"{r.get('peak_turn_L', 0):4.0f}/{r.get('peak_turn_R', 0):4.0f} loom L/R "
                f"{r.get('loom_peak_left', 0):4.0f}/{r.get('loom_peak_right', 0):4.0f} "
                f"({r['wall_s']:.1f}s, total {time.time() - t0:.0f}s)")
    return rows


def _away_from_seen(rs: list[dict], min_ratio: float = 1.25) -> dict:
    """First pillar: heading change away from the eye with the stronger peak loom."""
    sel = []
    for r in rs:
        L, R = r.get("loom_peak_left", 0.0), r.get("loom_peak_right", 0.0)
        if max(L, R) <= 0 or max(L, R) < min_ratio * max(min(L, R), 1e-9):
            continue
        stronger = 1.0 if L > R else -1.0
        dh = r.get("first_heading_change_deg", 0.0)
        if abs(dh) >= 0.5:  # (no turn: not counted either way)
            sel.append(-stronger * dh)
    if not sel:
        return {"n_lateralised": 0, "away_from_seen": "-", "away_from_seen_p": None}
    k = sum(x > 0 for x in sel)
    return {"n_lateralised": len(sel), "away_from_seen": f"{k}/{len(sel)}",
            "away_from_seen_p": float(f"{_binom_two_sided(k, len(sel)):.3g}")}


def summarize_canyon(rows: list[dict]) -> dict:
    order = ["brain", "mirror", "none"]
    controls = sorted({r["control"] for r in rows}, key=order.index)
    by = {c: {r["trial"]: r for r in rows if r["control"] == c} for c in controls}
    out = {"conditions": {}, "paired": {}}
    for c in controls:
        rs = list(by[c].values())
        n = len(rs)
        n_obs = rs[0]["n_obstacles"] if rs else 0
        first_ok = sum(not r.get("first_hit", True) for r in rs)
        passed = [r["passed"] for r in rs]
        dist = [r["distance_mm"] for r in rs]
        total_pass = sum(passed)
        total_crash = sum(bool(r["crashed"]) for r in rs)
        turn = [r.get("first_turn_away_deg", 0.0) for r in rs]
        k = sum(x > 0 for x in turn)
        nz = sum(x != 0 for x in turn)
        out["conditions"][c] = {
            "n": n, "n_obstacles": n_obs,
            "first_passed": first_ok,
            "first_pass_rate": round(first_ok / n, 3) if n else 0.0,
            "cleared_all": sum(r["passed"] == n_obs for r in rs),
            "mean_passed": round(float(np.mean(passed)), 2) if passed else 0.0,
            "passed_se": round(_se(passed), 2) if n > 1 else None,
            # crashes per pillar reached (each crash ends a trial)
            "crash_rate_per_pillar": round(total_crash / max(total_pass + total_crash, 1), 3),
            "mean_distance_mm": round(float(np.mean(dist)), 1) if dist else 0.0,
            "distance_se": round(_se(dist), 1) if n > 1 else None,
            "near_misses": sum(r["near_misses"] for r in rs),
            "first_turned_away": f"{k}/{nz}",
            "first_turned_away_p": float(f"{_binom_two_sided(k, nz):.3g}") if nz else 1.0,
            "mean_first_turn_away_deg": round(float(np.mean(turn)), 1) if turn else 0.0,
            **_away_from_seen(rs),
        }
    for a, b in (("brain", "none"), ("brain", "mirror"), ("mirror", "none")):
        if a not in by or b not in by:
            continue
        common = sorted(set(by[a]) & set(by[b]))
        only_a = sum((not by[a][i].get("first_hit", True)) and by[b][i].get("first_hit", True)
                     for i in common)
        only_b = sum((not by[b][i].get("first_hit", True)) and by[a][i].get("first_hit", True)
                     for i in common)
        dp = [by[a][i]["passed"] - by[b][i]["passed"] for i in common]
        dd = [by[a][i]["distance_mm"] - by[b][i]["distance_mm"] for i in common]
        out["paired"][f"{a}_vs_{b}"] = {
            "n": len(common), f"first_only_{a}": only_a, f"first_only_{b}": only_b,
            "mcnemar_p": float(f"{_binom_two_sided(only_a, only_a + only_b):.3g}")
            if only_a + only_b else 1.0,
            "further": f"{sum(x > 0 for x in dd)}/{sum(x != 0 for x in dd)}",
            "mean_passed_diff": round(float(np.mean(dp)), 2) if dp else None,
            "mean_distance_diff_mm": round(float(np.mean(dd)), 1) if dd else None,
            "wilcoxon_p": _wilcoxon_p(dd),
        }
    return out


def format_canyon_summary(summary: dict) -> str:
    lines = ["| condition | trials | first pillar passed | all pillars cleared | mean pillars "
             "passed | crashes per pillar reached | mean distance (mm) | near misses | turned "
             "away from the first pillar (p) | mean turn away (deg) | away from the eye that "
             "saw more (p) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in summary["conditions"].items():
        pse = f" +- {v['passed_se']:.2f}" if v.get("passed_se") else ""
        dse = f" +- {v['distance_se']:.1f}" if v.get("distance_se") else ""
        seen = (f"{v['away_from_seen']} (p={v['away_from_seen_p']})"
                if v.get("n_lateralised") else "-")
        lines.append(
            f"| {c} | {v['n']} | {v['first_passed']}/{v['n']} ({100 * v['first_pass_rate']:.0f}%) | "
            f"{v['cleared_all']}/{v['n']} | {v['mean_passed']:.2f}{pse} of {v['n_obstacles']} | "
            f"{v['crash_rate_per_pillar']:.2f} | {v['mean_distance_mm']:.1f}{dse} | "
            f"{v['near_misses']} | {v['first_turned_away']} (p={v['first_turned_away_p']}) | "
            f"{v['mean_first_turn_away_deg']:+.1f} | {seen} |")
    lines += ["", "| comparison | pairs | first pillar passed only in first | only in second | "
                  "McNemar p | first flew further | mean pillars passed diff | mean distance "
                  "diff (mm) | Wilcoxon p (distance) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k, v in summary["paired"].items():
        a, b = k.split("_vs_")
        lines.append(f"| {a} vs {b} | {v['n']} | {v[f'first_only_{a}']} | {v[f'first_only_{b}']} | "
                     f"{v['mcnemar_p']} | {v['further']} | {v['mean_passed_diff']} | "
                     f"{v['mean_distance_diff_mm']} | {v['wilcoxon_p']} |")
    return "\n".join(lines)
