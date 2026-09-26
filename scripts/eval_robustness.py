"""Robustness evaluation: N headless app sessions -> long-horizon metrics report.

Runs ``scripts/run_sim.py --headless`` as separate processes (optionally in
parallel), then aggregates the spec's long-term metrics from each run folder's
``summary.json`` / ``events.csv`` / ``metrics.csv`` (the same definitions as the
app's own RunMetrics):

  mean time to failure (fall) and to unrecoverable failure (auto reset),
  longest uninterrupted run, P(survive T), mean recovery time, max impulse
  survived, hits survived (overall and per strength level), falls per km,
  recovery %, falls / time per terrain type, instabilities / crashes,
  per-process peak RSS.

Examples::

    # 4 sessions (seeds 0..3), 120 sim s each, 2 at a time
    .venv/bin/python scripts/eval_robustness.py --seeds 0,1,2,3 --duration 120 --jobs 2 \\
        --terrain normal --hit-mode whip --auto-levels 1,2 --out runs/robust/normal_whip

    # only aggregate existing run folders (e.g. soak runs)
    .venv/bin/python scripts/eval_robustness.py --aggregate runs/soak/*/2026-* --out runs/soak/report

Writes ``<out>/report.json`` and ``<out>/report.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HORIZONS = (5.0, 10.0, 30.0, 60.0, 120.0, 300.0)


# --------------------------------------------------------------------------- running
def _session_cmd(args: argparse.Namespace, seed: int, runs_dir: Path) -> list[str]:
    cmd = [sys.executable, str(ROOT / "scripts" / "run_sim.py"), "--headless",
           "--runs-dir", str(runs_dir), "--max-seconds", str(args.duration),
           "--terrain", args.terrain, "--hit-mode", args.hit_mode,
           "--terrain-seed", str(args.terrain_seed + seed), "--auto-seed", str(seed),
           "--seed", str(seed), "--log-hz", str(args.log_hz),
           "--print-interval", str(args.print_interval)]
    if not args.no_perturb:
        cmd += ["--auto-perturb", "--auto-levels", args.auto_levels]
        if args.auto_min is not None:
            cmd += ["--auto-min", str(args.auto_min)]
        if args.auto_max is not None:
            cmd += ["--auto-max", str(args.auto_max)]
    if args.auto_reset_after is not None:
        cmd += ["--auto-reset-after", str(args.auto_reset_after)]
    return cmd + list(args.extra)


def run_session(cmd: list[str], log_path: Path) -> dict[str, Any]:
    """Run one app process; return exit code, run dir, wall time, peak RSS."""
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        # os.wait4 gives this child's own rusage (peak RSS), unlike getrusage(CHILDREN).
        _, status, ru = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)
    wall = time.perf_counter() - t0
    run_dir = None
    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("logging to "):
            run_dir = line[len("logging to "):].rstrip("/")
            break
    # ru_maxrss: bytes on macOS, KiB on Linux
    rss_mb = ru.ru_maxrss / (1 << 20) if sys.platform == "darwin" else ru.ru_maxrss / 1024
    if run_dir is not None and not Path(run_dir).is_absolute():
        run_dir = str(ROOT / run_dir)
    return {"exit_code": proc.returncode, "run_dir": run_dir, "wall_s": wall,
            "peak_rss_mb": rss_mb, "log": str(log_path), "cmd": cmd}


# --------------------------------------------------------------------------- loading
def _float(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_run(run_dir: Path) -> dict[str, Any]:
    """summary.json + per-terrain fall / time counts + hit levels from the CSVs."""
    run_dir = Path(run_dir)
    out: dict[str, Any] = {"run_dir": str(run_dir)}
    sp = run_dir / "summary.json"
    out["summary"] = json.loads(sp.read_text()) if sp.exists() else None
    # events.csv: falls per terrain type, hit levels (in the same order as summary hits)
    falls_by_terrain: Counter = Counter()
    hit_levels: list[int | None] = []
    events: Counter = Counter()
    ep = run_dir / "events.csv"
    if ep.exists():
        with open(ep, newline="") as f:
            for row in csv.DictReader(f):
                et = row["event_type"]
                events[et] += 1
                if et == "fall":
                    falls_by_terrain[row["terrain_type"] or "?"] += 1
                elif et in ("hit", "whip"):
                    try:
                        lvl = json.loads(row["details"] or "{}").get("level")
                    except json.JSONDecodeError:
                        lvl = None
                    hit_levels.append(lvl)
    out["falls_by_terrain"] = dict(falls_by_terrain)
    out["hit_levels"] = hit_levels
    out["event_counts"] = dict(events)
    # metrics.csv: time on each terrain type, NaN rows, max |x|
    time_by_terrain: Counter = Counter()
    n_rows = n_nan = 0
    max_abs_x = 0.0
    mp = run_dir / "metrics.csv"
    hz = (out["summary"] or {}).get("sample_hz")
    if mp.exists():
        with open(mp, newline="") as f:
            for row in csv.DictReader(f):
                n_rows += 1
                time_by_terrain[row["terrain_type"] or "?"] += 1
                x = _float(row["x"])
                if x is None or not math.isfinite(x) or not math.isfinite(_float(row["z"]) or 0):
                    n_nan += 1
                else:
                    max_abs_x = max(max_abs_x, abs(x))
    dt = 1.0 / hz if hz else None
    out["time_by_terrain_s"] = {k: v * dt for k, v in time_by_terrain.items()} if dt else {}
    out["n_metric_rows"] = n_rows
    out["n_nan_rows"] = n_nan
    out["max_abs_x_mm"] = max_abs_x
    return out


# --------------------------------------------------------------------------- aggregate
def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def aggregate(runs: list[dict[str, Any]], horizons=DEFAULT_HORIZONS) -> dict[str, Any]:
    """Pool the runs. A 'failure' is a fall; an 'unrecoverable failure' is a fall that
    ended in an auto reset (still down after --auto-reset-after)."""
    total_t = dist = walked = fwd = 0.0
    n_falls = n_rec = n_hits = n_surv = n_auto_resets = n_manual_resets = 0
    n_destab = n_relapse = 0
    intervals: list[float] = []  # decided jog intervals (completed)
    open_intervals: list[float] = []  # still running at the end (right-censored)
    rec_times: list[float] = []
    impulses_surv: list[float] = []
    by_level: dict[Any, list[int]] = defaultdict(lambda: [0, 0])  # level -> [hits, survived]
    falls_terr: Counter = Counter()
    time_terr: Counter = Counter()
    incomplete, crashes, instab = [], [], []
    nan_rows = 0
    max_x = 0.0
    per_run = []
    for r in runs:
        s = r.get("summary")
        proc = r.get("process") or {}
        if s is None:
            crashes.append({"run_dir": r["run_dir"], "reason": "no summary.json", **proc})
            continue
        m = s["metrics"]
        qr = s.get("quit_reason", "")
        if not s.get("complete"):
            incomplete.append(r["run_dir"])
        if str(qr).startswith("error"):
            (instab if "Instability" in qr else crashes).append(
                {"run_dir": r["run_dir"], "reason": qr})
        elif proc.get("exit_code") not in (None, 0):
            crashes.append({"run_dir": r["run_dir"], "reason": f"exit {proc['exit_code']}"})
        total_t += m["run_time_s"]
        dist += m["distance_mm"]
        walked += m.get("walked_distance_mm", m["distance_mm"])  # older runs: no split
        fwd += m["forward_displacement_mm"]
        n_falls += m["n_falls"]
        n_rec += m["n_recoveries"]
        n_hits += m["n_hits"]
        n_surv += m["n_hits_survived"]
        n_destab += m.get("n_destabilized", 0)
        n_relapse += m.get("n_relapses", 0)
        n_auto_resets += s.get("n_auto_resets", 0)
        n_manual_resets += s.get("n_manual_resets", 0)
        intervals += m["jog_intervals_s"]
        if m.get("current_jog_interval_s", 0) > 0:
            open_intervals.append(m["current_jog_interval_s"])
        rec_times += m["recovery_times_s"]
        hits = s.get("hits", [])
        levels = r.get("hit_levels", [])
        for i, h in enumerate(hits):
            lvl = levels[i] if i < len(levels) else None
            by_level[lvl][0] += 1
            if not h.get("caused_fall"):
                by_level[lvl][1] += 1
                if h.get("impulse_uN_s") is not None:
                    impulses_surv.append(h["impulse_uN_s"])
        falls_terr.update(r.get("falls_by_terrain", {}))
        time_terr.update(r.get("time_by_terrain_s", {}))
        nan_rows += r.get("n_nan_rows", 0)
        max_x = max(max_x, r.get("max_abs_x_mm", 0.0))
        per_run.append({
            "run_dir": r["run_dir"], "run_time_s": m["run_time_s"],
            "wall_s": s.get("wall_time_s"), "quit_reason": qr, "complete": s.get("complete"),
            "falls": m["n_falls"], "recoveries": m["n_recoveries"],
            "auto_resets": s.get("n_auto_resets", 0), "hits": m["n_hits"],
            "hits_survived": m["n_hits_survived"],
            "longest_jog_s": m["longest_jog_interval_s"], "distance_mm": m["distance_mm"],
            "walked_mm": m.get("walked_distance_mm"),
            "forward_mm": m["forward_displacement_mm"], "max_abs_x_mm": r.get("max_abs_x_mm"),
            "peak_rss_mb": proc.get("peak_rss_mb"), "exit_code": proc.get("exit_code"),
        })

    def p_survive(T: float) -> float | None:
        # completed intervals are decided; open ones only if already >= T
        decided = intervals + [o for o in open_intervals if o >= T]
        return sum(d >= T for d in decided) / len(decided) if decided else None

    terrains = sorted(set(time_terr) | set(falls_terr))
    return {
        "n_runs": len(runs),
        "total_sim_time_s": total_t,
        "total_distance_mm": dist,  # thorax path incl. flights after hits
        "total_walked_mm": walked,  # walking only (see fly_simulator/metrics/run_metrics.py)
        "total_forward_mm": fwd,
        "avg_forward_speed_mm_s": fwd / total_t if total_t else None,
        "n_falls": n_falls,
        "n_recoveries": n_rec,
        "n_relapses": n_relapse,
        "n_destabilized": n_destab,
        "n_auto_resets": n_auto_resets,
        "n_manual_resets": n_manual_resets,
        # right-censored estimate: total exposure time / events
        "mean_time_to_failure_s": total_t / n_falls if n_falls else None,
        "mean_time_to_unrecoverable_failure_s": total_t / n_auto_resets if n_auto_resets else None,
        "longest_uninterrupted_run_s": max(intervals + open_intervals, default=0.0),
        "p_survive": {f"{T:g}s": p_survive(T) for T in horizons},
        "mean_recovery_time_s": _mean(rec_times),
        "recovery_percentage": 100.0 * n_rec / n_falls if n_falls else None,
        "falls_per_km_path": n_falls / (dist / 1e6) if dist > 0 else None,
        "falls_per_km_forward": n_falls / (fwd / 1e6) if fwd > 0 else None,
        "n_hits": n_hits,
        "n_hits_survived": n_surv,
        "hits_survived_fraction": n_surv / n_hits if n_hits else None,
        "hits_by_level": {str(k): {"hits": v[0], "survived": v[1],
                                   "survived_fraction": v[1] / v[0] if v[0] else None}
                          for k, v in sorted(by_level.items(), key=lambda kv: str(kv[0]))},
        "max_impulse_survived_uN_s": max(impulses_surv, default=None),
        # "obstacle success": falls per sim minute on each terrain type
        "terrain": {t: {"time_s": time_terr.get(t, 0.0), "falls": falls_terr.get(t, 0),
                        "falls_per_min": (60.0 * falls_terr.get(t, 0) / time_terr[t]
                                          if time_terr.get(t) else None)}
                    for t in terrains},
        "crashes": crashes,
        "instabilities": instab,
        "incomplete_summaries": incomplete,
        "nan_metric_rows": nan_rows,
        "max_abs_x_mm": max_x,
        "runs": per_run,
    }


# --------------------------------------------------------------------------- report
def _f(v, fmt="{:.2f}") -> str:
    return "n/a" if v is None else fmt.format(v)


def markdown_report(agg: dict[str, Any], title: str = "Robustness report",
                    setup: str | None = None) -> str:
    a = agg
    L = [f"# {title}", ""]
    if setup:
        L += [setup, ""]
    L += [
        f"{a['n_runs']} runs, {a['total_sim_time_s']:.0f} sim s total "
        f"({a['total_sim_time_s'] / 60:.1f} min), forward {a['total_forward_mm'] / 1000:.2f} m, "
        f"walked {a.get('total_walked_mm', a['total_distance_mm']) / 1000:.2f} m "
        f"(path incl. flights {a['total_distance_mm'] / 1000:.2f} m).",
        "",
        "| metric | value |", "|---|---|",
        f"| crashes / instabilities / incomplete summaries | {len(a['crashes'])} / "
        f"{len(a['instabilities'])} / {len(a['incomplete_summaries'])} |",
        f"| NaN rows in metrics.csv | {a['nan_metric_rows']} |",
        f"| mean time to failure (fall) | {_f(a['mean_time_to_failure_s'], '{:.1f}')} s |",
        f"| mean time to unrecoverable failure (auto reset) | "
        f"{_f(a['mean_time_to_unrecoverable_failure_s'], '{:.1f}')} s |",
        f"| longest uninterrupted run | {a['longest_uninterrupted_run_s']:.1f} s |",
        f"| falls / recoveries / relapses | {a['n_falls']} / {a['n_recoveries']} / "
        f"{a['n_relapses']} |",
        f"| recovery % | {_f(a['recovery_percentage'], '{:.0f}')} |",
        f"| mean recovery time | {_f(a['mean_recovery_time_s'])} s |",
        f"| auto resets | {a['n_auto_resets']} |",
        f"| falls per km (forward / path) | {_f(a['falls_per_km_forward'], '{:.0f}')} / "
        f"{_f(a['falls_per_km_path'], '{:.0f}')} |",
        f"| hits survived | {a['n_hits_survived']} / {a['n_hits']} "
        f"({_f(a['hits_survived_fraction'])}) |",
        f"| max impulse survived | {_f(a['max_impulse_survived_uN_s'], '{:.3f}')} uN*s |",
        f"| avg forward speed | {_f(a['avg_forward_speed_mm_s'])} mm/s |",
        f"| max abs x reached | {a['max_abs_x_mm']:.0f} mm |",
        "",
        "P(survive T) (pooled jogging intervals; open intervals count only once >= T): "
        + ", ".join(f"{k}: {_f(v)}" for k, v in a["p_survive"].items()),
        "",
        "Hits by strength level: " + ", ".join(
            f"L{k}: {v['survived']}/{v['hits']}" for k, v in a["hits_by_level"].items()),
        "",
        "| terrain | time (s) | falls | falls / min |", "|---|---|---|---|",
    ]
    for t, v in a["terrain"].items():
        L.append(f"| {t} | {v['time_s']:.0f} | {v['falls']} | {_f(v['falls_per_min'])} |")
    L += ["", "| run | sim s | wall s | quit | falls | rec | auto resets | hits (surv) | "
          "longest jog s | peak RSS MB |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in a["runs"]:
        L.append(f"| {Path(r['run_dir']).parent.name}/{Path(r['run_dir']).name} | "
                 f"{r['run_time_s']:.0f} | {_f(r['wall_s'], '{:.0f}')} | {r['quit_reason']} | "
                 f"{r['falls']} | {r['recoveries']} | {r['auto_resets']} | "
                 f"{r['hits']} ({r['hits_survived']}) | {r['longest_jog_s']:.1f} | "
                 f"{_f(r['peak_rss_mb'], '{:.0f}')} |")
    for c in a["crashes"] + a["instabilities"]:
        L.append(f"\n**FAILURE** {c}")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, required=True, help="report dir (run folders go in <out>/runs)")
    p.add_argument("--aggregate", nargs="+", type=Path, default=None,
                   help="don't run anything; aggregate these existing run folders")
    p.add_argument("--seeds", type=str, default="0", help="comma list; one session per seed")
    p.add_argument("--jobs", type=int, default=1, help="parallel processes (<= 4 on an M1)")
    p.add_argument("--duration", type=float, default=60.0, help="sim seconds per session")
    p.add_argument("--terrain", default="normal")
    p.add_argument("--terrain-seed", type=int, default=42, help="base; + session seed")
    p.add_argument("--hit-mode", choices=["whip", "shove"], default="whip")
    p.add_argument("--auto-levels", default="1,2")
    p.add_argument("--auto-min", type=float, default=None)
    p.add_argument("--auto-max", type=float, default=None)
    p.add_argument("--no-perturb", action="store_true", help="no automatic hits")
    p.add_argument("--auto-reset-after", type=float, default=5.0)
    p.add_argument("--log-hz", type=float, default=10.0)
    p.add_argument("--print-interval", type=float, default=30.0)
    p.add_argument("--horizons", type=str, default=",".join(f"{h:g}" for h in DEFAULT_HORIZONS))
    p.add_argument("--title", default="Robustness report")
    p.add_argument("extra", nargs=argparse.REMAINDER,
                   help="after --: extra run_sim.py flags passed to every session")
    args = p.parse_args(argv)
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    horizons = tuple(float(h) for h in args.horizons.split(","))
    args.out.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    setup = None
    if args.aggregate:
        for d in args.aggregate:
            if (d / "summary.json").exists() or (d / "events.csv").exists():
                runs.append(load_run(d))
            else:
                print(f"skipping {d}: not a run folder", file=sys.stderr)
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        runs_dir = args.out / "runs"
        logs_dir = args.out / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        setup = (f"Setup: terrain `{args.terrain}`, hit mode `{args.hit_mode}`, "
                 f"{'no hits' if args.no_perturb else f'auto levels {args.auto_levels}'}, "
                 f"{args.duration:g} sim s x seeds {seeds}, auto reset after "
                 f"{args.auto_reset_after} s. Extra: {' '.join(args.extra) or '-'}")
        jobs = [(_session_cmd(args, s, runs_dir), logs_dir / f"seed{s}.log") for s in seeds]
        print(f"running {len(jobs)} sessions, {args.jobs} at a time", flush=True)
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            for res in ex.map(lambda j: run_session(*j), jobs):
                print(f"  exit {res['exit_code']}  {res['wall_s']:.0f}s wall  "
                      f"rss {res['peak_rss_mb']:.0f} MB  {res['run_dir']}", flush=True)
                r = (load_run(Path(res["run_dir"])) if res["run_dir"]
                     else {"run_dir": res["log"], "summary": None})
                r["process"] = res
                runs.append(r)
    agg = aggregate(runs, horizons)
    agg["setup"] = setup
    (args.out / "report.json").write_text(json.dumps(agg, indent=2, default=str))
    md = markdown_report(agg, args.title, setup)
    (args.out / "report.md").write_text(md)
    print(md)
    print(f"wrote {args.out / 'report.json'} and {args.out / 'report.md'}")
    bad = agg["crashes"] or agg["instabilities"] or agg["nan_metric_rows"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
