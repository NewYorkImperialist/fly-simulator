"""air_traffic experiment: does the connectome orient the controller toward the
calling plane better than chance? brain vs mirror vs none (docs/JOBS.md,
"air_traffic").

    python -m fly_simulator.jobs.air_traffic_experiment --planes 6 --json runs/atc_exp.json

Paired single-plane trials (the chase experiment's design, docs/GAMES.md): trial i
draws a relative bearing +-U(40, 90) deg (alternating sides), a loiter phase and a
height once; every condition sees the same plane. Each trial resets the fly and the
brain, settles 0.5 s, then the plane calls; up to ``trial_s`` of sim time. The job's
scripted housekeeping (walk back to the spot, face the runway) is off.

* ``brain``: left eye -> left LC10a (the job's normal wiring);
* ``mirror``: left eye -> right LC10a and vice versa, same brain, same turn map;
* ``none``: no turn drive (the geometry baseline: it never turns).
"""

from __future__ import annotations

import math

import numpy as np

from fly_simulator.jobs.geometry import wrap_angle


def run_orientation_experiment(n_planes: int = 8, conditions=("brain", "mirror", "none"), seed: int = 0,
                               trial_s: float = 6.0, bearing_deg=(40.0, 90.0), brain=None,
                               say=print) -> dict:
    """Paired single-plane trials: for trial i a calling plane appears at relative
    bearing +-U(bearing_deg) (alternating sides), the fly reset facing +x; each
    condition sees the same plane. Per trial: cleared (faced it within the
    tolerance), response time, the first-turn direction (heading change over
    ``first_turn_s`` toward the plane) and the final bearing error.
    ``brain``: a started ``BrainLink`` (None = build one; needs the brain data)."""
    from fly_simulator.config import AppConfig
    from fly_simulator.jobs import JobRunner, create_job_session, make_job

    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.logging.enabled = False
    own = brain is None
    if own:
        from fly_simulator.brain_link import BrainLink

        cfg.brain.enabled = True
        cfg.brain.window = False
        brain = BrainLink(cfg.brain, headless=True)
    job = make_job("air_traffic", {"spawn_every_s": 1e9, "divert_s": 1e9, "seed": seed})
    session, job = create_job_session(job, cfg, brain=brain)
    job.housekeeping = False  # no scripted turning during the trials
    if own:
        brain.wait_ready()
    runner = JobRunner(session, job, headless=True, print_every_s=1e9)
    rng = np.random.default_rng(seed)
    trials = [((1 if i % 2 == 0 else -1) * rng.uniform(*bearing_deg), rng.uniform(0, 2 * math.pi),
               rng.uniform(*job.cfg.call_z)) for i in range(n_planes)]
    out: dict = {c: [] for c in conditions}
    try:
        for i, (b_deg, ph, z) in enumerate(trials):
            for cond in conditions:
                job.set_control(cond)
                session.reset("manual")
                job._next_spawn = 1e9
                for p in job.planes:
                    p.state = "hidden"
                    p.path = None
                job.holding.clear()
                job.calling = None
                t_end = session.run_time() + 0.5
                while session.run_time() < t_end:  # settle, nothing to see
                    runner.step_chunk()
                p = job.planes[0]
                t = job.sim.time
                h = job.sim.heading()
                th = job.sim.thorax_position()
                a = h + math.radians(b_deg)
                p.side = 1 if b_deg > 0 else -1
                p.flight = f"TRIAL{i}"
                p.call_pt = np.array([th[0] + job.cfg.call_dist * math.cos(a),
                                      th[1] + job.cfg.call_dist * math.sin(a), z])
                p.state, p.path, p.t_called, p.phase0 = "calling", None, t, ph
                p.pos = p.call_pt + job._loiter_offset(p, t)
                job.calling = p.k
                job._aligned_since = None
                job._first = {"t": t, "h0": h, "e0": job.bearing_error(p.k), "k": p.k}
                n_ok0, n_tr0, n_cl0 = job.n_first_ok, job.n_first_trials, job.n_cleared
                t_end = session.run_time() + trial_s
                first = None
                while session.run_time() < t_end and job.n_cleared == n_cl0:
                    runner.step_chunk()
                    if first is None and job.sim.time - t >= job.cfg.first_turn_s:
                        first = math.degrees(wrap_angle(job.sim.heading() - h)) * (1 if b_deg > 0 else -1)
                cleared = job.n_cleared > n_cl0
                resp = job.last_resp if cleared else None
                err = job.bearing_error(p.k) if not cleared else 0.0
                if first is None:
                    first = math.degrees(wrap_angle(job.sim.heading() - h)) * (1 if b_deg > 0 else -1)
                row = {"trial": i, "bearing_deg": round(b_deg, 1), "cleared": cleared,
                       "response_s": round(resp, 2) if resp is not None else None,
                       "first_turn_toward_deg": round(first, 1),
                       "final_error_deg": round(abs(math.degrees(err)), 1) if err is not None else None}
                out[cond].append(row)
                say(f"[atc-exp] {cond:6s} trial {i} bearing {b_deg:+5.1f}: "
                    f"{'CLEARED %.2f s' % resp if cleared else 'not cleared'}  first turn {first:+.0f} deg")
                job.calling = None
                p.state = "hidden"
    finally:
        session.close("experiment end")
        if own:
            brain.close()
        session.sim.close()
    summ = {}
    for cond, rows in out.items():
        n = len(rows)
        cl = [r for r in rows if r["cleared"]]
        summ[cond] = {"n": n, "cleared": len(cl),
                      "mean_response_s": round(float(np.mean([r["response_s"] for r in cl])), 2) if cl else None,
                      "first_turn_toward": sum(1 for r in rows if r["first_turn_toward_deg"] > 5.0),
                      "first_turn_away": sum(1 for r in rows if r["first_turn_toward_deg"] < -5.0),
                      "mean_final_error_deg": round(float(np.mean([r["final_error_deg"] for r in rows])), 1)
                      if rows else None,
                      "mean_first_turn_deg": round(float(np.mean([r["first_turn_toward_deg"] for r in rows])), 1)
                      if rows else None}
    return {"trials": out, "summary": summ}


def main(argv=None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    ap = argparse.ArgumentParser(description="air_traffic: brain vs mirror vs none orientation experiment")
    ap.add_argument("--planes", type=int, default=8, help="planes per condition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args(argv)
    res = run_orientation_experiment(a.planes, seed=a.seed)
    print(json.dumps(res["summary"], indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
