#!/usr/bin/env python
"""Headless demo / calibration of the physical whip (fly_simulator.interaction.whip).

Examples::

    # one crack per level x side, response table
    python scripts/demo_whip.py
    # calibration over 8 gait phases (parallel), markdown table
    python scripts/demo_whip.py --calibrate --phases 8 --markdown whip_table.md
    # same on procedural terrain
    python scripts/demo_whip.py --calibrate --phases 4 --terrain normal
    # slow-motion video of a crack sequence + PNGs around impacts
    python scripts/demo_whip.py --record out.mp4 --sequence 1:left 2:right 3:front 4:left \
        --frames-dir /tmp/whip_frames

Each trial: fresh sim with the whip, walk ``--pre`` s (+ a gait-phase offset), crack,
keep walking ``--post`` s. Response metrics (see scripts/demo_perturbation.py) plus:
  hit       the whip touched the fly during the strike window
  impulse   measured contact impulse on the fly (uN*s) and peak force (uN)
  push      thorax displacement along the commanded push direction 0.6 s after the
            crack, minus the unperturbed run (mm)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_perturbation import _response_metrics  # noqa: E402

from fly_simulator.config import AppConfig  # noqa: E402
from fly_simulator.interaction.whip import WHIP_SIDES, Whip, WhipConfig  # noqa: E402
from fly_simulator.simulation import Simulation, SimulationInstabilityError  # noqa: E402

SAMPLE_EVERY = 10  # physics steps (1 ms)


def make_sim(terrain: str, whip: Whip, terrain_seed: int = 42):
    cfg = AppConfig()
    if terrain == "plain":
        sim = Simulation(cfg, world_extensions=[whip.extension])
    else:
        from fly_simulator.terrain import ProceduralTerrain, ProceduralTerrainConfig

        tr = ProceduralTerrain(ProceduralTerrainConfig(difficulty=terrain, seed=terrain_seed))
        sim = Simulation(cfg, world_factory=tr.build_world, world_extensions=[whip.extension])
        tr.attach(sim)
    whip.attach(sim)
    return sim


def run_trial(side: str, level: int = 2, pre_s: float = 1.0, post_s: float = 2.0,
              terrain: str = "plain", seed: int = 0, whip_kw: dict | None = None) -> dict:
    whip = Whip(WhipConfig(seed=seed, **(whip_kw or {})))
    sim = make_sim(terrain, whip)
    try:
        sim.step(int(round((pre_s - 0.25) / sim.timestep)))
        pa = sim.thorax_position()
        sim.step(int(round(0.25 / sim.timestep)))
        p0 = sim.thorax_position()
        heading0 = math.atan2(p0[1] - pa[1], p0[0] - pa[0])
        z0 = p0[2]
        t_crack = sim.time
        yaw = sim.heading()
        push = {"left": (math.sin(yaw), -math.cos(yaw)), "right": (-math.sin(yaw), math.cos(yaw)),
                "front": (-math.cos(yaw), -math.sin(yaw)), "rear": (math.cos(yaw), math.sin(yaw))}
        evs = []
        whip.listeners.append(evs.append)
        if side != "none":
            whip.crack(side, level, source="demo")
        ts, pos, tilt = [sim.time], [p0], [sim.tilt_deg()]
        status = "ok"
        for _ in range(int(round(post_s / sim.timestep / SAMPLE_EVERY))):
            try:
                sim.step(SAMPLE_EVERY)
            except (SimulationInstabilityError, mujoco.FatalError) as e:
                status = "UNSTABLE: " + str(e).splitlines()[0]
                break
            ts.append(sim.time)
            pos.append(sim.thorax_position())
            tilt.append(sim.tilt_deg())
        ev = evs[0] if evs else None
        out = dict(side=side, level=level, pre_s=pre_s, status=status, terrain=terrain,
                   hit=bool(ev and ev.hit), impulse=ev.impulse_uNs if ev else 0.0,
                   peak=ev.magnitude_uN if ev else 0.0, dur_ms=ev.duration_s * 1e3 if ev else 0.0,
                   body=(ev.body.split("/")[-1] if ev and ev.body else "-"),
                   imp_dir=ev.direction if ev else (0, 0, 0),
                   cmd_dir=ev.commanded_direction if ev else (0, 0, 0),
                   t_contact=(ev.sim_time - t_crack) if ev and ev.hit else float("nan"),
                   stray=whip.stray_contact_steps,
                   push_vec=push.get(side))
        # hit direction quality: cosine between measured impulse and commanded push
        out["cos"] = float(np.dot(out["imp_dir"], out["cmd_dir"])) if out["hit"] else float("nan")
        out.update(_response_metrics(ts, pos, tilt, t_crack, p0, heading0, z0))
        out["_traj"] = (np.asarray(pos), np.asarray(tilt), np.asarray(ts) - t_crack)
        return out
    finally:
        sim.close()


def add_baseline(rows: list[dict]) -> None:
    base = {(r["pre_s"], r["terrain"]): r["_traj"] for r in rows if r["side"] == "none"}
    for r in rows:
        b = base.get((r["pre_s"], r["terrain"]))
        r["dev_peak"] = r["dtilt_peak"] = r["push"] = float("nan")
        if b is None:
            continue
        (pos, tilt, ts), (bpos, btilt, _) = r["_traj"], b
        n = min(len(pos), len(bpos))
        r["dev_peak"] = float(np.max(np.linalg.norm(pos[:n, :2] - bpos[:n, :2], axis=1)))
        r["dtilt_peak"] = float(np.max(tilt[:n] - btilt[:n]))
        if r["push_vec"] is not None:
            i = min(n - 1, int(np.searchsorted(ts, 0.6)))  # 0.6 s after the crack request
            r["push"] = float((pos[i, :2] - bpos[i, :2]) @ np.asarray(r["push_vec"]))


def _star(kw):
    return run_trial(**kw)


def run_batch(jobs, workers):
    if workers <= 1:
        return [run_trial(**kw) for kw in jobs]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_star, jobs))


def fmt_row(r: dict) -> str:
    return (f"{r['side']:>8} L{r['level']} pre {r['pre_s']:.4f} "
            f"{'HIT ' if r['hit'] else 'miss'} imp {r['impulse']:6.3f}uNs peak {r['peak']:6.1f}uN "
            f"dur {r['dur_ms']:4.1f}ms cos {r['cos']:5.2f} {r['body']:>12} "
            f"push {r.get('push', float('nan')):6.2f}mm dev {r.get('dev_peak', float('nan')):6.2f} "
            f"dz {r['dz_peak']:5.2f} tilt {r['tilt_peak']:5.1f} fell {'Y' if r['fell'] else 'n'} "
            f"down {'Y' if r['down_at_end'] else 'n'} resumed {'Y' if r['resumed'] else 'n'} "
            f"stray {r['stray']} {'' if r['status'] == 'ok' else r['status']}")


def aggregate_markdown(rows: list[dict], cfg: WhipConfig, post_s: float) -> str:
    lines = [
        "| level | omega (rad/s) | side | n | hit rate | impulse uN*s mean (max) | peak force uN "
        "mean (max) | cos(impulse, push) | push along dir. (mm) | deviation (mm) | "
        f"extra tilt (deg) | tipped >60 deg | still down at end | walking again within {post_s:g} s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["side"] != "none":
            groups.setdefault((r["level"], r["side"]), []).append(r)
    for (lvl, side), rs in sorted(groups.items()):
        n = len(rs)

        def mm(key, fmt=".2f", only_hits=False):
            v = np.array([r.get(key, np.nan) for r in rs if r["hit"] or not only_hits], float)
            if v.size == 0 or np.all(np.isnan(v)):
                return "-"
            return f"{np.nanmean(v):{fmt}} ({np.nanmax(v):{fmt}})"

        def count(key):
            return f"{sum(bool(r[key]) for r in rs)}/{n}"

        cos = np.array([r["cos"] for r in rs if r["hit"]])
        lines.append(
            f"| {lvl} {cfg.levels[lvl - 1].name} | {cfg.levels[lvl - 1].omega:g} | {side} | {n} "
            f"| {count('hit')} | {mm('impulse', '.3f', True)} | {mm('peak', '.0f', True)} "
            f"| {np.mean(cos):.2f} | {mm('push')} | {mm('dev_peak')} "
            f"| {mm('dtilt_peak', '.0f')} | {count('fell')} | {count('down_at_end')} "
            f"| {count('resumed')} |" if cos.size else
            f"| {lvl} | | {side} | {n} | 0/{n} | - | - | - | - | - | - | - | - | - |")
    return "\n".join(lines)


# --------------------------------------------------------------------- video
def record_sequence(out: Path, sequence: list[tuple[int, str]], gap_s: float, lead_s: float,
                    frames_dir: Path | None, terrain: str, seed: int, every: int = 25) -> None:
    """Slow motion: one frame every ``every`` steps (2.5 ms) around cracks, every 150
    steps (real time) otherwise; PNGs of every frame during the strike windows."""
    import cv2
    import imageio.v2 as iio

    from fly_simulator.rendering import FrameRenderer

    cfg = AppConfig()
    whip = Whip(WhipConfig(seed=seed))
    sim = make_sim(terrain, whip)
    fr = FrameRenderer(sim.model, cfg.render, cfg.camera)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(out, fps=30, codec="libx264", quality=8, macro_block_size=8)
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)
    schedule = [(lead_s + i * gap_s, lvl, s) for i, (lvl, s) in enumerate(sequence)]
    end_t = lead_s + len(sequence) * gap_s
    label, tag, fi = "", "", 0
    results = []
    whip.listeners.append(results.append)
    try:
        t0 = sim.time
        while sim.time - t0 < end_t:
            if schedule and sim.time - t0 >= schedule[0][0]:
                _, lvl, side = schedule.pop(0)
                whip.crack(side, lvl, source="demo")
                tag = f"L{lvl}-{side}"
                label = f"whip L{lvl} {whip.level_name(lvl)} from {side}"
                print(f"[crack] frame {fi} t={sim.time:.3f}s {label}", flush=True)
            active = whip.phase in ("hold", "swing", "follow")
            slow = whip.phase != "idle"
            sim.step(every if slow else 150)
            frame = fr.render(sim.data, sim.time, sim.thorax_position(), sim.heading()).copy()
            texts = [(f"t {sim.time:6.3f} s  {'SLOW MOTION x' + str(round(1 / (every * 1e-4 * 30))) if slow else 'real time x2.2'}",
                      (10, frame.shape[0] - 12))]
            if slow:
                texts.append((label, (10, 30)))
            if results and sim.time - results[-1].sim_time < 0.3:
                e = results[-1]
                txt = (f"HIT {e.body.split('/')[-1]}: {e.impulse_uNs:.2f} uN*s, peak "
                       f"{e.magnitude_uN:.0f} uN" if e.hit else "MISS")
                texts.append((txt, (10, 60)))
            for text, org in texts:
                cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4,
                            cv2.LINE_AA)
                cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                            1, cv2.LINE_AA)
            writer.append_data(frame)
            if frames_dir is not None and active:
                iio.imwrite(frames_dir / f"{tag}_{fi:05d}_{whip.phase}.png", frame)
            fi += 1
    finally:
        writer.close()
        fr.close()
        sim.close()
    for e in results:
        print(f"  {e.direction_name:>14} L{e.level}: {'HIT' if e.hit else 'miss'} "
              f"impulse {e.impulse_uNs:.3f} uN*s peak {e.magnitude_uN:.0f} uN body {e.body}")
    print(f"wrote {out} ({fi} frames)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--sides", nargs="+", default=["left", "right", "front", "rear", "overhead"],
                    choices=list(WHIP_SIDES) + ["none"])
    ap.add_argument("--pre", type=float, default=1.0)
    ap.add_argument("--post", type=float, default=2.0)
    ap.add_argument("--phases", type=int, default=1)
    ap.add_argument("--phase-step", type=float, default=0.0105)
    ap.add_argument("--terrain", default="plain",
                    help="plain (flat Simulation world) or a ProceduralTerrain difficulty")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--markdown", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--omegas", type=float, nargs=4, default=None,
                    help="override the level angular speeds (rad/s)")
    ap.add_argument("--whip-json", default=None,
                    help='WhipConfig overrides as JSON, e.g. \'{"chain_mass": 1e-3}\'')
    ap.add_argument("--lunges", type=float, nargs=4, default=None, help="override level lunges (mm)")
    ap.add_argument("--record", type=Path, default=None)
    ap.add_argument("--sequence", nargs="+", default=["1:left", "2:right", "3:front", "4:left"])
    ap.add_argument("--gap", type=float, default=1.2)
    ap.add_argument("--frames-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.record is not None:
        seq = [(int(s.split(":")[0]), s.split(":")[1]) for s in args.sequence]
        record_sequence(args.record, seq, args.gap, 0.8, args.frames_dir, args.terrain,
                        args.seed)
        return 0

    whip_kw = json.loads(args.whip_json) if args.whip_json else None
    cfg = WhipConfig(**(whip_kw or {}))
    if args.omegas:
        for lv, w in zip(cfg.levels, args.omegas):
            lv.omega = w
        whip_kw = {**(whip_kw or {}), "levels": cfg.levels}
    if args.lunges:
        for lv, x in zip(cfg.levels, args.lunges):
            lv.lunge = x
        whip_kw = {**(whip_kw or {}), "levels": cfg.levels}
    jobs = [dict(side="none", pre_s=args.pre + k * args.phase_step, post_s=args.post,
                 terrain=args.terrain) for k in range(args.phases)]
    jobs += [dict(side=s, level=lvl, pre_s=args.pre + k * args.phase_step, post_s=args.post,
                  terrain=args.terrain, seed=args.seed + k, whip_kw=whip_kw)
             for lvl in args.levels for s in args.sides for k in range(args.phases)]
    t = time.perf_counter()
    rows = run_batch(jobs, args.workers)
    add_baseline(rows)
    for r in rows:
        if r["side"] != "none":
            print(fmt_row(r))
    hits = [r for r in rows if r["side"] != "none"]
    for lvl in args.levels:
        rs = [r for r in hits if r["level"] == lvl]
        print(f"L{lvl}: hit rate {sum(r['hit'] for r in rs)}/{len(rs)}, tipped "
              f"{sum(r['fell'] for r in rs)}/{len(rs)}, mean impulse "
              f"{np.mean([r['impulse'] for r in rs if r['hit']] or [0]):.3f} uN*s")
    print(f"[{len(rows)} trials in {time.perf_counter() - t:.0f} s wall]")
    if args.calibrate or args.markdown:
        md = aggregate_markdown(rows, cfg, args.post)
        print(md)
        if args.markdown:
            args.markdown.write_text(md + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
