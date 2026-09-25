"""Long-horizon evaluation of a policy in ``PerpetualFlyEnv`` (spec "Long-term eval").

``evaluate(env, policy, sim_seconds)`` runs back-to-back episodes until the sim-time
budget is used up. Episodes end only when the fly stays FALLEN for
``fall_terminate_after_s`` (then an explicit reset, counted as a failure) or when the
budget is reached (censored). ``RunMetrics`` (the app's own metrics) is attached to
the env's Simulation, so the numbers are the same definitions as the app's
``summary.json``.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from perpetualfly.metrics.run_metrics import RunMetrics
from perpetualfly.rl.env import PerpetualFlyEnv

Policy = Callable[[np.ndarray], np.ndarray]


def zero_policy(n_actions: int) -> Policy:
    z = np.zeros(n_actions, dtype=np.float32)
    return lambda obs: z


def evaluate(env: PerpetualFlyEnv, policy: Policy, sim_seconds: float, seed: int = 0,
             stage: int | str | None = None, verbose: bool = True) -> dict[str, Any]:
    """Run ``policy`` for ``sim_seconds`` of simulated time; return the metrics dict."""
    if stage is not None:
        env.set_stage(stage)
    # One long episode per failure: truncation only by the overall budget.
    env.cfg.max_episode_s = float(sim_seconds) + 1.0
    env._max_policy_steps = int(round(env.cfg.max_episode_s / env.dt))
    obs, _ = env.reset(seed=seed)
    # Attached after the first reset (a reset closes RunMetrics' open jogging interval,
    # which would add a spurious 0 s interval to P(survive T)).
    metrics = RunMetrics(update_every_steps=100).attach(env.sim, env.detector)
    env.perturbation.listeners.append(
        lambda ev: metrics.record_hit(ev, time=ev.sim_time, magnitude=ev.magnitude_uN,
                                      duration=ev.duration_s, direction=ev.direction_name,
                                      body=ev.body))
    t_wall = time.perf_counter()
    elapsed, n_steps, episodes = 0.0, 0, []
    ep = 0
    ep_t0 = 0.0
    fall_time_in_ep: float | None = None
    while elapsed < sim_seconds:
        obs, _, term, trunc, info = env.step(policy(obs))
        n_steps += 1
        elapsed += env.dt
        if "fall" in info.get("fall_events", ()) and fall_time_in_ep is None:
            fall_time_in_ep = info["sim_time"]
        if term or trunc or elapsed >= sim_seconds:
            st = info.get("episode_stats") or env.episode_stats()
            failed = bool(term)
            episodes.append(dict(
                st, failed=failed, instability=bool(info.get("instability")),
                time_to_failure_s=(fall_time_in_ep if fall_time_in_ep is not None
                                   else st["sim_time_s"]) if failed else None,
            ))
            if verbose:
                print(f"  episode {ep}: {st['sim_time_s']:.1f}s  fwd {st['forward_mm']:.0f}mm  "
                      f"hits {st['n_hits']}  falls {st['n_falls']}  rec {st['n_recoveries']}  "
                      f"{'FAILED' if failed else 'censored'}", flush=True)
            ep += 1
            if elapsed < sim_seconds:
                obs, _ = env.reset()
            fall_time_in_ep = None
    wall = time.perf_counter() - t_wall

    s = metrics.summary()
    total_t = float(sum(e["sim_time_s"] for e in episodes))
    fwd = float(sum(e["forward_mm"] for e in episodes))
    n_fail = sum(e["failed"] for e in episodes)
    hits = s["n_hits"]
    out = {
        "stage": env.stage.name,
        "sim_seconds": total_t,
        "wall_seconds": wall,
        "env_steps_per_s": n_steps / wall,
        "episodes": len(episodes),
        "failures": n_fail,  # stayed down > fall_terminate_after_s (explicit reset)
        # failures per unit time with right-censored episodes: total time / failures
        "mean_time_to_failure_s": total_t / n_fail if n_fail else None,
        "longest_uninterrupted_run_s": s["longest_jog_interval_s"],
        "n_falls": s["n_falls"],
        "n_recoveries": s["n_recoveries"],
        "recovery_percentage": s["recovery_percentage"],
        "mean_recovery_time_s": s["mean_recovery_time_s"],
        "falls_per_km_path": s["falls_per_km"],
        "falls_per_km_forward": s["n_falls"] / (fwd / 1e6) if fwd > 0 else None,
        "n_hits": hits,
        "n_hits_survived": s["n_hits_survived"],
        "hits_survived_fraction": s["n_hits_survived"] / hits if hits else None,
        "max_impulse_survived_uN_s": s["max_impulse_survived_uN_s"],
        "avg_forward_speed_mm_s": fwd / total_t if total_t > 0 else None,
        "avg_path_speed_mm_s": s["average_speed_mm_s"],
        "p_survive": s["p_survive"],
        "instabilities": sum(e["instability"] for e in episodes),
        "episode_details": episodes,
    }
    return out


def format_report(r: dict[str, Any]) -> str:
    def f(v, fmt="{:.2f}"):
        return "n/a" if v is None else fmt.format(v)

    ps = ", ".join(f"{k}: {f(v)}" for k, v in r["p_survive"].items())
    return "\n".join([
        f"stage {r['stage']}: {r['sim_seconds']:.1f} sim s in {r['wall_seconds']:.0f} wall s "
        f"({r['env_steps_per_s']:.0f} env steps/s), {r['episodes']} episodes, "
        f"{r['failures']} failures",
        f"  mean time to failure   {f(r['mean_time_to_failure_s'], '{:.1f}')} s",
        f"  longest run            {r['longest_uninterrupted_run_s']:.1f} s",
        f"  falls / recoveries     {r['n_falls']} / {r['n_recoveries']}  "
        f"(recovery {f(r['recovery_percentage'], '{:.0f}')} %, "
        f"mean {f(r['mean_recovery_time_s'])} s)",
        f"  falls per km           {f(r['falls_per_km_forward'], '{:.0f}')} (forward), "
        f"{f(r['falls_per_km_path'], '{:.0f}')} (path)",
        f"  hits survived          {r['n_hits_survived']} / {r['n_hits']}  "
        f"({f(r['hits_survived_fraction'])})",
        f"  avg forward speed      {f(r['avg_forward_speed_mm_s'])} mm/s "
        f"(path {f(r['avg_path_speed_mm_s'])})",
        f"  P(survive T)           {ps}",
        f"  instabilities          {r['instabilities']}",
    ])
