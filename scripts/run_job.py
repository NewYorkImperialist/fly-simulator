#!/usr/bin/env python
"""Run an eternal job (docs/JOBS.md): a fly doing an absurd job forever.

    python scripts/run_job.py --job sisyphus                 # live window (Q quits)
    python scripts/run_job.py --job hamster_wheel --headless --max-seconds 300
    python scripts/run_job.py --job sisyphus --headless --record runs/jobs/sisyphus \
        --segment-s 60 --keep 3 --timelapse 5
    python scripts/run_job.py --rotate --rotate-minutes 10   # cycle through all jobs forever

Keys (window): Q / ESC quit, C camera (job / follow / side / top), P pause,
X explicit reset (counted), I screenshot (PNG with HUD).

Mirrors perpetualfly.app.Session construction (the job's props are compiled in via
world_extensions); the physical whip and the connectome brain are off unless
--whip / --brain. --rotate rebuilds the Session between jobs (props are compiled in).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perpetualfly.config import AppConfig  # noqa: E402
from perpetualfly.jobs import JobRunner, available_jobs, create_job_session, make_job  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--job", default="sisyphus", help=f"one of {available_jobs()}")
    p.add_argument("--list", action="store_true", help="list the jobs and exit")
    p.add_argument("--job-config", type=str, default=None,
                   help='JSON overrides of the job config, e.g. \'{"slope_deg": 8}\'')
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="stop after this many simulated seconds (per job with --rotate: total)")
    p.add_argument("--record", type=Path, default=None,
                   help="directory for rolling MP4 segments (with the HUD)")
    p.add_argument("--segment-s", type=float, default=60.0, help="sim seconds per segment")
    p.add_argument("--keep", type=int, default=3, help="segments kept (older are deleted)")
    p.add_argument("--fps", type=float, default=15.0, help="recording / timelapse frame rate")
    p.add_argument("--timelapse", type=float, default=None, metavar="K",
                   help="also a timelapse: one frame every K sim seconds (into --record "
                        "or runs/jobs/<job>)")
    p.add_argument("--rotate", action="store_true", help="cycle through all jobs forever")
    p.add_argument("--rotate-minutes", type=float, default=10.0,
                   help="sim minutes per job with --rotate")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--print-every", type=float, default=10.0, help="sim s between status lines")
    p.add_argument("--log", action="store_true", help="write runs/<ts>/ logs (off by default)")
    p.add_argument("--whip", action="store_true", help="also build the physical whip")
    p.add_argument("--brain", action="store_true",
                   help="run the connectome brain alongside (needs .[brain] + data/brain); "
                        "opens the brain window unless --headless / --no-brain-window")
    p.add_argument("--no-brain-window", action="store_true",
                   help="with --brain: no brain window")
    p.add_argument("--seed", type=int, default=None, help="controller (CPG) seed")
    return p


def app_config(args) -> AppConfig:
    cfg = AppConfig()
    cfg.whip.enabled = bool(args.whip)
    cfg.logging.enabled = bool(args.log)
    if args.width:
        cfg.render.width = args.width
    if args.height:
        cfg.render.height = args.height
    if args.seed is not None:
        cfg.controller.seed = args.seed
    return cfg


def run_one(name: str, args, max_seconds: float | None, stop_flag: dict) -> dict:
    job_cfg = json.loads(args.job_config) if args.job_config and not args.rotate else None
    job = make_job(name, job_cfg)
    cfg = app_config(args)
    brain = None
    if args.brain:
        from perpetualfly.brain_link import BrainLink, missing_requirements

        cfg.brain.enabled = True
        problem = missing_requirements(cfg.brain)
        if problem:
            raise SystemExit(f"ERROR: {problem}")
        if args.no_brain_window:
            cfg.brain.window = False
        # the brain window opens unless --headless / --no-brain-window (like run_sim.py)
        brain = BrainLink(cfg.brain, headless=args.headless)
    session, job = create_job_session(job, cfg, log=args.log, brain=brain)
    try:
        if brain is not None:
            brain.wait_ready()
            brain.start_window()
            print(f"brain: {brain.info.get('n_neurons', 0):,} neurons ready | window "
                  f"{'on' if brain.window is not None else 'off'}", flush=True)
        rec_dir = args.record / name if (args.record and args.rotate) else args.record
        tl_dir = rec_dir or Path(cfg.logging.runs_dir) / "jobs" / name
        runner = JobRunner(session, job, headless=args.headless, record_dir=rec_dir,
                           segment_s=args.segment_s, keep_segments=args.keep,
                           record_fps=args.fps, timelapse_every_s=args.timelapse,
                           timelapse_dir=tl_dir, print_every_s=args.print_every)
        print(f"== {job.title}: {job.tagline} | {'headless' if args.headless else 'window'}"
              f"{f' | recording to {rec_dir}' if rec_dir else ''}", flush=True)
        out = runner.run(max_seconds=max_seconds)
        if out["quit_reason"] in ("quit key", "window closed", "Ctrl-C"):
            stop_flag["stop"] = True
        return out
    finally:
        session.close("job end")
        if brain is not None:
            brain.close()
        session.sim.close()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for n in available_jobs():
            print(n)
        return 0
    stop = {"stop": False}
    if not args.rotate:
        out = run_one(args.job, args, args.max_seconds, stop)
        print("[result] " + json.dumps(out, default=str), flush=True)
        return 0
    names = available_jobs()
    total, i = 0.0, 0
    while not stop["stop"]:
        name = names[i % len(names)]
        chunk = args.rotate_minutes * 60.0
        if args.max_seconds is not None:
            chunk = min(chunk, args.max_seconds - total)
            if chunk <= 0:
                break
        out = run_one(name, args, chunk, stop)
        total += out["run_time_s"]
        print(f"[rotate] {name} done: " + json.dumps(out, default=str), flush=True)
        i += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
