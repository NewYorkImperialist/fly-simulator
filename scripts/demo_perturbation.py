#!/usr/bin/env python
"""Headless demo / calibration of the external-force "whip".

Examples::

    # every direction at every strength level, response metrics table
    python scripts/demo_perturbation.py
    # a subset
    python scripts/demo_perturbation.py --levels 2 3 --directions left up
    # custom force (body weights) / duration (s) instead of the level presets
    python scripts/demo_perturbation.py --mag 8 --dur 0.02 --directions left
    # calibration table over several gait phases (parallel), markdown output
    python scripts/demo_perturbation.py --calibrate --phases 4 --markdown table.md
    # continuous video with a medium left hit and a hard right hit (+ PNGs around impacts)
    python scripts/demo_perturbation.py --record out.mp4 --sequence 2:left 3:right \
        --frames-dir /tmp/frames

Each trial: fresh sim, walk ``--pre`` s (+ a phase offset), hit, keep walking ``--post`` s.
Metrics (relative to the pre-hit position and heading):
  lat_peak  peak |displacement| perpendicular to the pre-hit travel direction (mm)
  v_peak    peak horizontal thorax speed in the first 0.3 s (mm/s)
  dz_peak   peak rise of the thorax above its pre-hit height (mm)
  tilt_peak peak angle between thorax up-axis and world up (deg)
  fell      tilt ever > 60 deg (on its side / back)
  resumed   upright (tilt < 30) and walking (> 7 mm/s) during the last 0.5 s
  t_resume  time after the hit until it is upright + walking for good (s)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perpetualfly.config import AppConfig  # noqa: E402
from perpetualfly.interaction.perturbation import (  # noqa: E402
    DIRECTIONS,
    Perturbation,
    PerturbationConfig,
)
from perpetualfly.simulation import Simulation, SimulationInstabilityError  # noqa: E402

SAMPLE_EVERY = 10  # physics steps (1 ms)
FALL_TILT = 60.0
UPRIGHT_TILT = 30.0
WALK_SPEED = 7.0  # mm/s; normal jogging is ~14 mm/s


def _response_metrics(ts, pos, tilt, t_hit, p0, heading0, z0) -> dict:
    ts, pos, tilt = np.asarray(ts), np.asarray(pos), np.asarray(tilt)
    rel = pos - p0
    left = np.array([-math.sin(heading0), math.cos(heading0)])
    lat = rel[:, :2] @ left
    dt = ts[1] - ts[0]
    vel = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) / dt
    early = ts[1:] - t_hit < 0.3
    # windowed speed (net displacement over 0.25 s) -> ignores gait sway
    w = max(1, int(round(0.25 / dt)))
    wspeed = np.full(len(ts), np.nan)
    wspeed[:-w] = np.linalg.norm(pos[w:, :2] - pos[:-w, :2], axis=1) / (w * dt)
    ok = (tilt < UPRIGHT_TILT) & (wspeed > WALK_SPEED)
    upright_after = np.flip(np.cumprod(np.flip(tilt < UPRIGHT_TILT))).astype(bool)
    good = ok & upright_after
    t_resume = float(ts[np.argmax(good)] - t_hit) if good.any() else float("nan")
    last = ts > ts[-1] - 0.5
    end_speed = np.linalg.norm(pos[-1, :2] - pos[np.argmax(last), :2]) / (
        ts[-1] - ts[np.argmax(last)])
    return dict(
        lat_peak=float(np.max(np.abs(lat))),
        v_peak=float(vel[early].max()),
        dz_peak=float(np.max(pos[:, 2] - z0)),
        tilt_peak=float(tilt.max()),
        fell=bool(tilt.max() > FALL_TILT),
        down_at_end=bool(tilt[-1] > FALL_TILT),
        resumed=bool(np.all(tilt[last] < UPRIGHT_TILT) and end_speed > WALK_SPEED),
        t_resume=t_resume,
        end_speed=float(end_speed),
    )


def run_trial(direction: str, level: int | None = None, mag: float | None = None,
              dur: float | None = None, pre_s: float = 0.6, post_s: float = 2.0,
              seed: int = 0, pcfg: PerturbationConfig | None = None) -> dict:
    """One hit on a freshly walking fly; returns response metrics."""
    sim = Simulation(AppConfig())
    pert = Perturbation(sim, pcfg or PerturbationConfig(seed=seed))
    try:
        # The fly crab-walks slightly (body heading ~ a few deg off its travel
        # direction), so "lateral" is measured relative to the travel direction over
        # the last 0.25 s before the hit, not the body heading.
        sim.step(int(round((pre_s - 0.25) / sim.timestep)))
        pa = sim.thorax_position()
        sim.step(int(round(0.25 / sim.timestep)))
        p0 = sim.thorax_position()
        heading0 = math.atan2(p0[1] - pa[1], p0[0] - pa[0])
        # pre-hit height averaged over one gait cycle would be nicer; the bob is
        # only ~0.05 mm so the instantaneous value is fine.
        z0 = p0[2]
        t_hit = sim.time
        ev = None
        if direction != "none":
            ev = pert.apply_impulse("thorax", direction, magnitude=mag, duration=dur,
                                    level=level)
        ts, pos, tilt = [sim.time], [p0], [sim.tilt_deg()]
        status = "ok"
        for _ in range(int(round(post_s / sim.timestep / SAMPLE_EVERY))):
            try:
                sim.step(SAMPLE_EVERY)
            except SimulationInstabilityError as e:
                status = "UNSTABLE: " + str(e).splitlines()[0]
                break
            ts.append(sim.time)
            pos.append(sim.thorax_position())
            tilt.append(sim.tilt_deg())
        out = dict(direction=direction, level=level, pre_s=pre_s, status=status,
                   mag_bw=ev.magnitude_bw if ev else 0.0,
                   force_uN=ev.magnitude_uN if ev else 0.0,
                   dur_ms=ev.duration_s * 1e3 if ev else 0.0,
                   impulse_uNs=ev.impulse_uNs if ev else 0.0,
                   dir_used=ev.direction_name if ev else "none")
        out.update(_response_metrics(ts, pos, tilt, t_hit, p0, heading0, z0))
        out["_traj"] = (np.asarray(pos), np.asarray(tilt))
        return out
    finally:
        sim.close()


def add_baseline_deviation(rows: list[dict]) -> None:
    """Compare each hit with the unperturbed run of the same gait phase (same pre_s):
    the controller is deterministic, so any difference is caused by the hit.

    dev_peak   peak horizontal distance between perturbed and unperturbed thorax (mm)
    dtilt_peak peak extra tilt compared with the unperturbed run (deg)
    """
    base = {r["pre_s"]: r["_traj"] for r in rows if r["direction"] == "none"}
    for r in rows:
        b = base.get(r["pre_s"])
        if b is None:
            r["dev_peak"] = r["dtilt_peak"] = float("nan")
            continue
        (pos, tilt), (bpos, btilt) = r["_traj"], b
        n = min(len(pos), len(bpos))
        r["dev_peak"] = float(np.max(np.linalg.norm(pos[:n, :2] - bpos[:n, :2], axis=1)))
        r["dtilt_peak"] = float(np.max(tilt[:n] - btilt[:n]))


def _trial_star(kw):
    return run_trial(**kw)


def fmt_row(r: dict) -> str:
    return (f"{r['direction']:>8} L{r['level'] if r['level'] else '-'} "
            f"{r['mag_bw']:5.2f}BW {r['force_uN']:6.1f}uN {r['dur_ms']:4.0f}ms "
            f"dev {r.get('dev_peak', float('nan')):6.2f}mm dtilt {r.get('dtilt_peak', float('nan')):5.1f} "
            f"lat {r['lat_peak']:5.2f}mm v {r['v_peak']:6.1f}mm/s dz {r['dz_peak']:5.2f}mm "
            f"tilt {r['tilt_peak']:5.1f}deg fell {'Y' if r['fell'] else 'n'} "
            f"down {'Y' if r['down_at_end'] else 'n'} resumed {'Y' if r['resumed'] else 'n'} t_res {r['t_resume']:4.2f}s "
            f"end {r['end_speed']:4.1f}mm/s {'' if r['status'] == 'ok' else r['status']}")


def run_batch(jobs: list[dict], workers: int) -> list[dict]:
    if workers <= 1:
        return [run_trial(**kw) for kw in jobs]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_trial_star, jobs))


def aggregate_markdown(rows: list[dict], pcfg: PerturbationConfig, post_s: float) -> str:
    """One row per (strength, direction): mean (max) over gait phases."""
    lines = [
        "| level | force | dur. | impulse | direction | n | deviation from unperturbed "
        "path (mm) | peak lateral disp. (mm) | extra tilt (deg) | peak rise (mm) "
        f"| tipped >60 deg | still down at end | walking again within {post_s:g} s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["level"], r["mag_bw"], r["dur_ms"], r["direction"]), []).append(r)

    for (lvl, mag, dur, d), rs in groups.items():
        n, r0 = len(rs), rs[0]
        name = f"{lvl} {pcfg.levels[lvl - 1].name}" if lvl else "-"

        def mm(key: str, fmt: str = ".2f") -> str:
            v = np.array([r.get(key, np.nan) for r in rs], dtype=float)
            return f"{np.nanmean(v):{fmt}} ({np.nanmax(v):{fmt}})"

        def count(key: str) -> str:
            return f"{sum(bool(r[key]) for r in rs)}/{n}"

        lines.append(
            f"| {name} | {mag:g} BW = {r0['force_uN']:.1f} uN | {dur:.0f} ms "
            f"| {r0['impulse_uNs'] * 1e3:.0f} nN*s | {d} | {n} | {mm('dev_peak')} "
            f"| {mm('lat_peak')} | {mm('dtilt_peak', '.0f')} | {mm('dz_peak')} "
            f"| {count('fell')} | {count('down_at_end')} | {count('resumed')} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------- video
def record_sequence(out: Path, sequence: list[tuple[int, str]], gap_s: float, lead_s: float,
                    frames_dir: Path | None, frame_offsets: list[int], seed: int) -> None:
    import cv2
    import imageio.v2 as iio

    from perpetualfly.rendering import FrameRenderer

    cfg = AppConfig()
    sim = Simulation(cfg)
    pert = Perturbation(sim, PerturbationConfig(seed=seed))
    # Render every 50 steps (5 ms) so the ~10-20 ms shove spans a few frames; the
    # video is written at 30 fps, i.e. ~6.7x slow motion.
    every = 50
    fr = FrameRenderer(sim.model, cfg.render, cfg.camera)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(out, fps=30, codec="libx264", quality=8, macro_block_size=8)
    schedule = [(lead_s + i * gap_s, lvl, d) for i, (lvl, d) in enumerate(sequence)]
    end_t = lead_s + len(sequence) * gap_s
    # PNGs are written on the fly (keeping every frame in RAM would take GBs); a short
    # history covers negative offsets.
    history: deque[np.ndarray] = deque(maxlen=max(1, -min(frame_offsets, default=0)) + 1)
    wanted: dict[int, str] = {}
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
    label, label_until = "", -1.0
    fi = 0
    try:
        t0 = sim.time
        while sim.time - t0 < end_t:
            if schedule and sim.time - t0 >= schedule[0][0]:
                _, lvl, d = schedule.pop(0)
                ev = pert.hit(d, level=lvl, source="demo")
                tag = f"L{lvl}-{d}"
                label = (f"{pert.cfg.levels[lvl - 1].name} {d}: {ev.magnitude_uN:.0f} uN "
                         f"x {ev.duration_s * 1e3:.0f} ms")
                label_until = sim.time + 0.4
                print(f"[hit] frame {fi} t={ev.sim_time:.3f}s {label}", flush=True)
                for off in frame_offsets:
                    if off < 0 and len(history) >= -off and frames_dir is not None:
                        iio.imwrite(frames_dir / f"{tag}_f{off:+04d}.png", history[off])
                    elif off >= 0:
                        wanted[fi + off] = f"{tag}_f{off:+04d}.png"
            sim.step(every)
            frame = fr.render(sim.data, sim.time, sim.thorax_position(), sim.heading()).copy()
            if pert.is_active:  # red square while the force is acting
                frame[10:40, 10:40] = (255, 0, 0)
            texts = [(f"t {sim.time:6.3f} s  (slow motion)", (10, frame.shape[0] - 12))]
            if sim.time < label_until:
                texts.append((label, (50, 34)))
            for text, org in texts:  # dark outline + white text: readable on sky/floor
                cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 1, cv2.LINE_AA)
            writer.append_data(frame)
            if frames_dir is not None and fi in wanted:
                iio.imwrite(frames_dir / wanted.pop(fi), frame)
            history.append(frame)
            fi += 1
    finally:
        writer.close()
        fr.close()
        sim.close()
    print(f"wrote {out} ({fi} frames, {every * 1e-4 * 1e3:.0f} ms sim per frame, 30 fps)")
    if frames_dir is not None:
        print(f"PNGs around impacts in {frames_dir}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--directions", nargs="+", default=list(DIRECTIONS),
                    choices=list(DIRECTIONS) + ["none"])
    ap.add_argument("--mag", type=float, default=None, help="override force (body weights)")
    ap.add_argument("--dur", type=float, default=None, help="override duration (s)")
    ap.add_argument("--grid", nargs="+", default=None, metavar="MAG:DUR",
                    help="exploration: several magnitude(BW):duration(s) pairs")
    ap.add_argument("--pre", type=float, default=1.0, help="walk this long before the hit (s)")
    ap.add_argument("--post", type=float, default=2.0, help="observe this long after (s)")
    ap.add_argument("--phases", type=int, default=1,
                    help="repeat each trial with N different gait phases")
    ap.add_argument("--phase-step", type=float, default=0.0105,
                    help="gait-phase offset between repeats (s); the tripod cycle is "
                         "~83 ms and mirror-symmetric after ~42 ms")
    ap.add_argument("--calibrate", action="store_true",
                    help="aggregate over phases into a markdown table")
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", type=Path, default=None, help="write an mp4 instead of the table")
    ap.add_argument("--sequence", nargs="+", default=["2:left", "3:right"],
                    help="hits for --record as level:direction")
    ap.add_argument("--gap", type=float, default=1.5, help="seconds between recorded hits")
    ap.add_argument("--frames-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.record is not None:
        seq = [(int(s.split(":")[0]), s.split(":")[1]) for s in args.sequence]
        record_sequence(args.record, seq, args.gap, 0.6, args.frames_dir,
                        [-1, 0, 1, 2, 3, 4, 6, 10, 20], args.seed)
        return 0

    # Phase offsets of 21 ms ~ 1/4 of the ~83 ms (12 Hz) tripod cycle.
    if args.grid:
        combos = [(None, float(g.split(":")[0]), float(g.split(":")[1])) for g in args.grid]
    else:
        combos = [(lvl, args.mag, args.dur) for lvl in args.levels]
    jobs = [
        dict(direction=d, level=lvl, mag=mag, dur=dur,
             pre_s=args.pre + k * args.phase_step, post_s=args.post, seed=args.seed + k)
        for lvl, mag, dur in combos for d in args.directions for k in range(args.phases)
    ]
    t = time.perf_counter()
    if "none" not in args.directions:  # unperturbed reference runs, one per phase
        jobs = [dict(direction="none", pre_s=args.pre + k * args.phase_step, post_s=args.post)
                for k in range(args.phases)] + jobs
    rows = run_batch(jobs, args.workers)
    add_baseline_deviation(rows)
    rows = [r for r in rows if r["direction"] != "none" or "none" in args.directions]
    for r in rows:
        print(fmt_row(r))
    print(f"[{len(rows)} trials in {time.perf_counter() - t:.0f} s wall]")
    if args.calibrate or args.markdown:
        md = aggregate_markdown(rows, PerturbationConfig(), args.post)
        print(md)
        if args.markdown:
            args.markdown.write_text(md + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
