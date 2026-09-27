"""Does the connectome brain return the ball better than chance? (docs/GAMES.md, game 4)

Paired served rallies: for trial i the serve (start point on the opponent's face,
arrival point on the fly's baseline, alternating sides) and the arrival points of
every later ball (the opponent is a perfect "wall" that aims its k-th return at a
pre-drawn point, straight or off a side wall) are drawn once from ``seed`` and
replayed under every control condition:

* ``brain``: left eye -> left LC10a, right eye -> right LC10a;
* ``mirror``: left eye -> right LC10a and vice versa (same brain, same paddle map);
* ``none``: brain disconnected, paddle velocity 0 (it stays at the centre): the
  geometry / chance baseline (it returns the balls that happen to arrive at the
  centre).

Each trial: reset the fly and the brain (all neurons at rest), 0.3 s with no ball,
serve at the difficulty's ball speed (+6 % per hit), and play until the fly misses
or has returned ``max_returns`` balls. One brain process serves all conditions.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from fly_simulator.games.experiment import _binom_two_sided


@dataclass
class PongTrialSpec:
    i: int
    y0: float  # serve start (opponent's face)
    targets: list = field(default_factory=list)  # [(arrival y, bounce)] per ball


def make_pong_specs(n: int, seed: int = 0, n_balls: int = 4, first=(1.0, 7.5),
                    later=(0.0, 7.5), p_bounce: float = 0.3) -> list[PongTrialSpec]:
    out = []
    for i in range(n):
        r = np.random.default_rng(seed * 100003 + 7919 * i + 5)
        sign = 1.0 if i % 2 == 0 else -1.0  # balanced sides for the serve
        tg = [(sign * float(r.uniform(*first)), 0)]
        for _ in range(n_balls - 1):
            y = float(r.choice([-1.0, 1.0])) * float(r.uniform(*later))
            b = int(r.choice([-1, 1])) if r.random() < p_bounce else 0
            tg.append((y, b))
        out.append(PongTrialSpec(i, float(r.uniform(-6.0, 6.0)), tg))
    return out


def run_pong_trial(s, spec: PongTrialSpec, control: str, max_returns: int = 3,
                   settle_s: float = 0.3, timeout_s: float = 30.0) -> dict:
    """One served rally under ``control`` on a ``PongSession`` ``s``."""
    g, brain, ph = s.game, s.brain, s.court.phys
    c = g.cfg
    brain.set_control(control)
    g._reset_fly()
    brain.reset(g.time())
    g._new_game()
    g.state = "trial"
    t_end = g.time() + settle_s
    while g.time() < t_end:
        s.step()
    g.crossings.clear()
    g.returns = g.balls_faced = 0
    ph.events.clear()
    ph.ai_mode = "wall"
    ph.aim_fn = lambda k: spec.targets[min(k, len(spec.targets) - 1)]
    y1, b1 = spec.targets[0]
    ph.serve_aimed(c.ai_x - c.ball_radius_mm, spec.y0, y1, b1)
    side = 1.0 if y1 > 0 else -1.0
    t0 = g.time()
    carried0 = s.court.carried_mm
    peak = {"turn_L": 0.0, "turn_R": 0.0}
    lc10 = {"left": 0.0, "right": 0.0}
    seen = brain.n_states
    missed = False
    while g.time() - t0 < timeout_s:
        s.step()
        if brain.n_states != seen:
            seen = brain.n_states
            for k in peak:
                peak[k] = max(peak[k], float(brain.rates.get(k, 0.0)))
        for e, v in brain.pursuit_drive(g.time()).items():
            lc10[e] = max(lc10[e], v)
        if g.balls_faced > g.returns:  # the fly missed (the ball is out)
            missed = True
            break
        if g.returns >= max_returns:
            break
    cr = g.crossings
    offs = [abs(x["offset"]) for x in cr if "offset" in x]
    first = cr[0] if cr else {}
    row = {
        "trial": spec.i, "control": control, "serve_y": round(spec.y0, 2),
        "first_target_y": round(y1, 2), "first_side": "left" if side > 0 else "right",
        "first_return": bool(first.get("hit", False)),
        "returns": int(g.returns), "balls_faced": int(g.balls_faced),
        "missed": missed, "rally_s": round(g.time() - t0, 3),
        "first_offset_mm": round(float(first["offset"]), 3) if "offset" in first else None,
        # + = the paddle had moved toward the side the first ball arrived on
        "first_paddle_toward_mm": round(side * float(first["paddle_y"]), 3)
        if "paddle_y" in first else None,
        "mean_abs_offset_mm": round(float(np.mean(offs)), 3) if offs else None,
        "peak_turn_L": round(peak["turn_L"], 1), "peak_turn_R": round(peak["turn_R"], 1),
        "lc10a_peak_left": round(lc10["left"], 1), "lc10a_peak_right": round(lc10["right"], 1),
        "paddle_travel_mm": round(s.court.carried_mm - carried0, 1),
    }
    ph.hide()
    ph.ai_mode, ph.aim_fn = "track", None
    return row


def run_pong_experiment(s, n: int, controls=("brain", "mirror", "none"), seed: int = 0,
                        say=print, max_returns: int = 5) -> list[dict]:
    specs = make_pong_specs(n, seed, n_balls=max_returns + 1)
    rows = []
    t0 = time.time()
    for spec in specs:
        for c in controls:
            t1 = time.time()
            r = run_pong_trial(s, spec, c, max_returns=max_returns)
            r["wall_s"] = round(time.time() - t1, 2)
            rows.append(r)
            say(f"trial {spec.i:3d} {c:6s} first y {r['first_target_y']:+5.1f} returned "
                f"{int(r['first_return'])} returns {r['returns']}/{r['balls_faced']} paddle "
                f"toward {r['first_paddle_toward_mm'] if r['first_paddle_toward_mm'] is not None else float('nan'):+5.1f} "
                f"DN L/R {r['peak_turn_L']:4.0f}/{r['peak_turn_R']:4.0f} "
                f"({r['wall_s']:.1f}s, total {time.time() - t0:.0f}s)")
    return rows


def _se(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")


def _wilcoxon_p(x: list[float]) -> float | None:
    try:
        from scipy.stats import wilcoxon
    except ImportError:  # pragma: no cover
        return None
    x = [v for v in x if v != 0]
    if len(x) < 2:
        return None
    return float(f"{wilcoxon(x).pvalue:.3g}")


def summarize_pong(rows: list[dict]) -> dict:
    order = ["brain", "mirror", "none"]
    controls = sorted({r["control"] for r in rows}, key=order.index)
    by = {c: {r["trial"]: r for r in rows if r["control"] == c} for c in controls}
    out = {"conditions": {}, "paired": {}}
    for c in controls:
        rs = list(by[c].values())
        fr = [bool(r["first_return"]) for r in rs]
        ret = [r["returns"] for r in rs]
        faced = sum(r["balls_faced"] for r in rs)
        tow = [r["first_paddle_toward_mm"] for r in rs if r.get("first_paddle_toward_mm") is not None]
        k = sum(x > 0.1 for x in tow)
        nz = sum(abs(x) > 0.1 for x in tow)
        offs = [r["mean_abs_offset_mm"] for r in rs if r.get("mean_abs_offset_mm") is not None]
        out["conditions"][c] = {
            "n": len(rs),
            "first_return_rate": round(float(np.mean(fr)), 3),
            "first_returns": int(sum(fr)),
            "returns": int(sum(ret)), "balls_faced": int(faced),
            "return_rate": round(sum(ret) / faced, 3) if faced else 0.0,
            "mean_returns": round(float(np.mean(ret)), 2), "mean_returns_se": round(_se(ret), 2),
            "full_rallies": int(sum(not r["missed"] for r in rs)),
            "mean_paddle_travel_mm": round(float(np.mean([r.get("paddle_travel_mm", 0.0)
                                                          for r in rs])), 1),
            "paddle_toward": f"{k}/{nz}",
            "paddle_toward_p": float(f"{_binom_two_sided(k, nz):.3g}") if nz else 1.0,
            "mean_paddle_toward_mm": round(float(np.mean(tow)), 2) if tow else None,
            "mean_abs_offset_mm": round(float(np.mean(offs)), 2) if offs else None,
        }
    for a, b in (("brain", "none"), ("brain", "mirror"), ("mirror", "none")):
        if a not in by or b not in by:
            continue
        common = sorted(set(by[a]) & set(by[b]))
        only_a = sum(by[a][i]["first_return"] and not by[b][i]["first_return"] for i in common)
        only_b = sum(by[b][i]["first_return"] and not by[a][i]["first_return"] for i in common)
        diffs = [by[a][i]["returns"] - by[b][i]["returns"] for i in common]
        pos = sum(x > 0 for x in diffs)
        nz = sum(x != 0 for x in diffs)
        out["paired"][f"{a}_vs_{b}"] = {
            "n": len(common),
            f"first_only_{a}": only_a, f"first_only_{b}": only_b,
            "mcnemar_p": float(f"{_binom_two_sided(only_a, only_a + only_b):.3g}")
            if only_a + only_b else 1.0,
            "mean_returns_diff": round(float(np.mean(diffs)), 2) if diffs else None,
            "first_more_returns": f"{pos}/{nz}",
            "sign_test_p": float(f"{_binom_two_sided(pos, nz):.3g}") if nz else 1.0,
            "wilcoxon_p": _wilcoxon_p(diffs),
        }
    return out


def format_pong_summary(summary: dict) -> str:
    lines = ["| condition | trials | first serve returned | returns / balls faced (rate) | "
             "mean returns per rally (cap) | paddle moved toward the first ball (p) | "
             "mean paddle shift toward (mm) | mean miss distance (mm) |",
             "|---|---|---|---|---|---|---|---|"]
    for c, v in summary["conditions"].items():
        lines.append(f"| {c} | {v['n']} | {v['first_returns']}/{v['n']} "
                     f"({v['first_return_rate']:.2f}) | {v['returns']}/{v['balls_faced']} "
                     f"({v['return_rate']:.2f}) | {v['mean_returns']:.2f} +- "
                     f"{v['mean_returns_se']:.2f} | {v['paddle_toward']} (p={v['paddle_toward_p']}) | "
                     f"{v['mean_paddle_toward_mm']} | {v['mean_abs_offset_mm']} |")
    lines += ["", "| comparison | pairs | first returned only in first | only in second | "
                  "McNemar p | returns diff | first more returns | sign test p | Wilcoxon p |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k, v in summary["paired"].items():
        a, b = k.split("_vs_")
        lines.append(f"| {a} vs {b} | {v['n']} | {v[f'first_only_{a}']} | {v[f'first_only_{b}']} | "
                     f"{v['mcnemar_p']} | {v['mean_returns_diff']:+.2f} | {v['first_more_returns']} | "
                     f"{v['sign_test_p']} | {v['wilcoxon_p']} |")
    return "\n".join(lines)
