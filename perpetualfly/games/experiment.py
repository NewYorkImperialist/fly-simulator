"""Does the connectome brain steer away from asteroids more than chance? (docs/GAMES.md)

Paired single-rock trials: for trial i the rock's lateral offset, speed, radius slot,
spawn bearing and the fly's gait phase are drawn once from ``seed + i`` and replayed
under every control condition:

* ``brain``: real sensory mapping (left eye -> left LC4 / LPLC2);
* ``mirror``: left eye -> right optic lobe and vice versa (same brain and motor map);
* ``none``: brain disconnected, constant drive [1, 1] (the chance / geometry baseline).

Each trial: reset the fly (and the brain: all neurons at rest), walk ``walk_s`` (+ a
random fraction of a gait cycle), spawn one rock ahead aimed at the fly's predicted
position + offset, run until the rock is past (dodged) or has touched the fly (hit).
Per trial we record the outcome, the closest approach, the signed heading change
("away" > 0 = turned away from the side the rock passes on), the lateral escape
distance, jumps and the peak DN / looming rates. One brain process serves all
conditions (``GameBrain.set_control``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class TrialSpec:
    i: int
    offset_mm: float
    speed: float
    slot: int
    bearing_deg: float
    phase_s: float


def make_specs(n: int, seed: int = 0, offset_range=(0.5, 3.0), speed: float = 10.0,
               n_slots: int = 4) -> list[TrialSpec]:
    out = []
    for i in range(n):
        r = np.random.default_rng(seed * 100003 + i)
        mag = float(r.uniform(*offset_range))
        sign = 1.0 if i % 2 == 0 else -1.0  # balanced sides
        out.append(TrialSpec(i, sign * mag, speed, int(r.integers(n_slots)),
                             float(r.uniform(-6.0, 6.0)), float(r.uniform(0.0, 0.0833))))
    return out


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def run_trial(s, spec: TrialSpec, control: str, walk_s: float = 0.8,
              timeout_s: float = 4.0) -> dict:
    """One rock under ``control`` on an ``AsteroidSession`` ``s``."""
    g, sim, brain = s.game, s.sim, s.brain
    brain.set_control(control)
    g.state = "trial"  # no automatic spawning
    g.lives = 99
    g._reset_fly()
    brain.reset(g.time())
    n_jump0 = len(s.jump_times)
    t_end = g.time() + walk_s + spec.phase_s
    while g.time() < t_end:
        s.step()
    h0 = sim.heading()
    p0 = sim.thorax_position()[:2].copy()
    rk = g.spawn_rock(offset_mm=spec.offset_mm, speed=spec.speed, bearing_deg=spec.bearing_deg,
                      slot=spec.slot)
    t_spawn = g.time()
    u = rk.vel[:2] / max(float(np.hypot(*rk.vel[:2])), 1e-9)
    perp_left = np.array([-np.sin(h0), np.cos(h0)])  # the fly's left at spawn
    side = 1.0 if spec.offset_mm > 0 else -1.0  # +1: rock passes on the fly's left
    peak = {"turn_L": 0.0, "turn_R": 0.0, "escape": 0.0, "backward": 0.0, "walk": 0.0}
    loom_peak = {"left": 0.0, "right": 0.0}
    turn_int = 0.0  # integral of (turn_away) Hz*s from spawn to outcome
    rid = rk.rid
    seen_states = brain.n_states
    while (rk.rid == rid and rk.active and rk.outcome is None
           and g.time() - t_spawn < timeout_s):
        s.step()
        if brain.n_states != seen_states:
            seen_states = brain.n_states
            r = brain.rates
            for k in ("turn_L", "turn_R", "escape"):
                peak[k] = max(peak[k], float(r.get(k, 0.0)))
            peak["backward"] = max(peak["backward"],
                                   0.5 * (r.get("backward_L", 0.0) + r.get("backward_R", 0.0)))
            peak["walk"] = max(peak["walk"], 0.5 * (r.get("walk_L", 0.0) + r.get("walk_R", 0.0)))
            # away = the DN group contralateral to the rock's side
            away = r.get("turn_R", 0.0) - r.get("turn_L", 0.0)
            turn_int += side * away * brain.window_s
        for e, (lc4, lp) in brain.eye_drive(g.time()).items():
            loom_peak[e] = max(loom_peak[e], lc4, lp)
    outcome = rk.outcome or ("lost" if not rk.active else "timeout")
    p1 = sim.thorax_position()[:2]
    dh = _wrap(sim.heading() - h0)
    lateral = float((p1 - p0) @ perp_left)  # + = moved to the fly's (initial) left
    row = {
        "trial": spec.i, "control": control, "offset_mm": round(spec.offset_mm, 3),
        "rock_side": "left" if side > 0 else "right", "radius": rk.radius,
        "speed": spec.speed, "outcome": outcome, "hit": outcome == "hit",
        "min_dist_mm": round(rk.min_dist, 3),
        "heading_change_deg": round(math.degrees(dh), 1),
        # + = turned away from the rock (rock on the left -> turned right)
        "turn_away_deg": round(-side * math.degrees(dh), 1),
        "lateral_away_mm": round(-side * lateral, 3),
        "jumped": len(s.jump_times) > n_jump0,
        "t_outcome_s": round(g.time() - t_spawn, 3),
        "peak_turn_L": round(peak["turn_L"], 1), "peak_turn_R": round(peak["turn_R"], 1),
        "peak_gf": round(peak["escape"], 1), "peak_mdn": round(peak["backward"], 1),
        "peak_walk": round(peak["walk"], 1),
        "turn_away_dn_hz_s": round(turn_int, 2),
        "loom_peak_left": round(loom_peak["left"], 1),
        "loom_peak_right": round(loom_peak["right"], 1),
        "_u": u,
    }
    row.pop("_u")
    # let the rock clear before the next trial
    if rk.active:
        s.field.park(rk)
    return row


def run_experiment(s, n: int, controls=("brain", "mirror", "none"), seed: int = 0,
                   say=print, **spec_kw) -> list[dict]:
    specs = make_specs(n, seed, **spec_kw)
    rows = []
    t0 = time.time()
    for spec in specs:
        for c in controls:
            t1 = time.time()
            r = run_trial(s, spec, c)
            r["wall_s"] = round(time.time() - t1, 2)
            rows.append(r)
            say(f"trial {spec.i:3d} {c:6s} off {spec.offset_mm:+5.2f} {r['outcome']:7s} "
                f"min {r['min_dist_mm']:5.2f} turn_away {r['turn_away_deg']:+6.1f} deg "
                f"lat_away {r['lateral_away_mm']:+5.2f} mm DN L/R {r['peak_turn_L']:4.0f}/"
                f"{r['peak_turn_R']:4.0f} GF {r['peak_gf']:4.0f} loom L/R "
                f"{r['loom_peak_left']:4.0f}/{r['loom_peak_right']:4.0f} "
                f"jump {int(r['jumped'])} ({r['wall_s']:.1f}s, total {time.time() - t0:.0f}s)")
    return rows


def _binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    from math import comb

    pk = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = pk[k]
    return float(min(1.0, sum(x for x in pk if x <= obs * (1 + 1e-9))))


def _away_from_seen(rs: list[dict], min_ratio: float = 1.25) -> dict:
    sel = []
    for r in rs:
        L, R = r["loom_peak_left"], r["loom_peak_right"]
        if max(L, R) <= 0 or max(L, R) < min_ratio * max(min(L, R), 1e-9):
            continue
        stronger = 1.0 if L > R else -1.0  # +1: left eye saw more
        # heading change > 0 = turned left; away from a left stimulus = negative
        sel.append(-stronger * r["heading_change_deg"])
    if not sel:
        return {"n_lateralised": 0, "frac_away_from_seen": None, "mean_away_from_seen_deg": None}
    k = sum(x > 0 for x in sel)
    return {"n_lateralised": len(sel), "frac_away_from_seen": round(k / len(sel), 3),
            "mean_away_from_seen_deg": round(float(np.mean(sel)), 1),
            "away_from_seen_sign_p": float(f"{_binom_two_sided(k, len(sel)):.3g}")}


def summarize(rows: list[dict]) -> dict:
    """Per-condition dodge rate etc. plus paired comparisons (exact McNemar /
    sign tests)."""
    controls = sorted({r["control"] for r in rows}, key=["brain", "mirror", "none"].index)
    out = {"conditions": {}, "paired": {}}
    by = {c: {r["trial"]: r for r in rows if r["control"] == c} for c in controls}
    for c in controls:
        rs = list(by[c].values())
        n = len(rs)
        d = sum(not r["hit"] for r in rs)
        p = d / n if n else float("nan")
        se = math.sqrt(p * (1 - p) / n) if n else float("nan")
        out["conditions"][c] = {
            "n": n, "dodged": d, "dodge_rate": round(p, 3), "dodge_rate_se": round(se, 3),
            "mean_turn_away_deg": round(float(np.mean([r["turn_away_deg"] for r in rs])), 1),
            "mean_lateral_away_mm": round(float(np.mean([r["lateral_away_mm"] for r in rs])), 3),
            "mean_min_dist_mm": round(float(np.mean([r["min_dist_mm"] for r in rs])), 3),
            "jumps": sum(r["jumped"] for r in rs),
            "mean_turn_away_dn_hz_s": round(float(np.mean([r["turn_away_dn_hz_s"] for r in rs])), 2),
            "frac_turned_away": round(float(np.mean([r["turn_away_deg"] > 0 for r in rs])), 3),
            # relative to what the eyes actually saw: heading change away from the eye
            # with the stronger peak loom drive (trials where one eye clearly dominated)
            **_away_from_seen(rs),
        }
    for a, b in (("brain", "none"), ("brain", "mirror"), ("mirror", "none")):
        if a not in by or b not in by:
            continue
        common = sorted(set(by[a]) & set(by[b]))
        only_a = sum((not by[a][i]["hit"]) and by[b][i]["hit"] for i in common)
        only_b = sum(by[a][i]["hit"] and (not by[b][i]["hit"]) for i in common)
        diffs = [by[a][i]["lateral_away_mm"] - by[b][i]["lateral_away_mm"] for i in common]
        pos = sum(x > 0 for x in diffs)
        nz = sum(x != 0 for x in diffs)
        out["paired"][f"{a}_vs_{b}"] = {
            "n": len(common), f"dodged_only_{a}": only_a, f"dodged_only_{b}": only_b,
            "mcnemar_p": round(_binom_two_sided(only_a, only_a + only_b), 4)
            if only_a + only_b else 1.0,
            "mean_lateral_away_diff_mm": round(float(np.mean(diffs)), 3) if diffs else None,
            "lateral_away_larger_in": f"{pos}/{nz}",
            "sign_test_p": round(_binom_two_sided(pos, nz), 4) if nz else 1.0,
        }
    return out


def _seen_cell(v: dict) -> str:
    if not v.get("n_lateralised"):
        return "-"
    k = round(v["frac_away_from_seen"] * v["n_lateralised"])
    return (f"{k}/{v['n_lateralised']}, {v['mean_away_from_seen_deg']:+.1f} deg "
            f"(p={v['away_from_seen_sign_p']})")


def format_summary(summary: dict) -> str:
    lines = ["| condition | trials | dodged | dodge rate | turned away | mean turn away (deg) | "
             "mean lateral escape (mm) | mean closest approach (mm) | turned away from the eye "
             "that saw more (p) | jumps |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c, v in summary["conditions"].items():
        lines.append(f"| {c} | {v['n']} | {v['dodged']} | {v['dodge_rate']:.2f} +- "
                     f"{v['dodge_rate_se']:.2f} | {v['frac_turned_away']:.2f} | "
                     f"{v['mean_turn_away_deg']:+.1f} | {v['mean_lateral_away_mm']:+.2f} | "
                     f"{v['mean_min_dist_mm']:.2f} | {_seen_cell(v)} | {v['jumps']} |")
    lines += ["", "| comparison | pairs | dodged only in first | only in second | McNemar p | "
                  "lateral escape diff (mm) | first escapes further | sign test p |",
              "|---|---|---|---|---|---|---|---|"]
    for k, v in summary["paired"].items():
        a, b = k.split("_vs_")
        lines.append(f"| {a} vs {b} | {v['n']} | {v[f'dodged_only_{a}']} | {v[f'dodged_only_{b}']} | "
                     f"{v['mcnemar_p']} | {v['mean_lateral_away_diff_mm']:+.3f} | "
                     f"{v['lateral_away_larger_in']} | {v['sign_test_p']} |")
    return "\n".join(lines)
