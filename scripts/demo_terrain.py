"""Headless endless-terrain demo: walk the fly across many recycled chunks.

    .venv/bin/python scripts/demo_terrain.py --difficulty normal --seconds 60
    .venv/bin/python scripts/demo_terrain.py --difficulty easy --seconds 20 --record out.mp4
    .venv/bin/python scripts/demo_terrain.py --frames-dir /tmp/frames --frame-every 2

Prints distance, chunk recycles, terrain under the fly, thorax z and tilt, and a
summary of stuck / fallen episodes (heuristics: <1 mm forward progress over 3 s =
stuck; tilt > 60 deg = fallen). Nothing is reset or teleported during the run.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_simulator.config import AppConfig  # noqa: E402
from fly_simulator.simulation import Simulation  # noqa: E402
from fly_simulator.terrain import ProceduralTerrain, ProceduralTerrainConfig  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--difficulty", default="normal", choices=["easy", "normal", "hard", "chaos", "flat"])
    p.add_argument("--seconds", type=float, default=60.0, help="simulated seconds")
    p.add_argument("--seed", type=int, default=42, help="terrain seed")
    p.add_argument("--controller-seed", type=int, default=0)
    p.add_argument("--print-every", type=float, default=2.0, help="simulated seconds")
    p.add_argument("--record", type=Path, default=None, help="write an mp4")
    p.add_argument("--frames-dir", type=Path, default=None, help="save PNG snapshots here")
    p.add_argument("--frame-every", type=float, default=2.0, help="seconds between PNGs")
    p.add_argument("--weights", default=None,
                   help='override the probability table, e.g. "flat=1,gap=1"')
    p.add_argument("--camera", default="follow", choices=["follow", "side", "top"])
    args = p.parse_args(argv)

    cfg = AppConfig()
    cfg.controller.seed = args.controller_seed
    cfg.camera.mode = args.camera
    terrain = None
    if args.difficulty == "flat":
        sim = Simulation(cfg)
    else:
        weights = None
        if args.weights:
            weights = {k: float(v) for k, v in (kv.split("=") for kv in args.weights.split(","))}
        terrain = ProceduralTerrain(ProceduralTerrainConfig(
            difficulty=args.difficulty, seed=args.seed, weights=weights))
        sim = Simulation(cfg, world_factory=terrain.build_world)
        terrain.attach(sim)

    chunk = cfg.render.render_every_steps  # 150 steps = 15 ms
    renderer = writer = None
    if args.record or args.frames_dir:
        from fly_simulator.rendering import FrameRenderer

        renderer = FrameRenderer(sim.model, cfg.render, cfg.camera)
    if args.record:
        import imageio.v2 as iio

        args.record.parent.mkdir(parents=True, exist_ok=True)
        writer = iio.get_writer(args.record, fps=1.0 / (chunk * sim.timestep), codec="libx264",
                                quality=8, macro_block_size=8)
    if args.frames_dir:
        args.frames_dir.mkdir(parents=True, exist_ok=True)

    type_time: Counter[str] = Counter()
    progress: list[tuple[float, float]] = []
    stuck_events: list[tuple[float, float, str]] = []
    fall_events: list[tuple[float, float, str]] = []
    stuck = fallen = False
    next_print, next_frame = args.print_every, 0.0
    t_start, wall0 = sim.time, time.perf_counter()
    max_tilt = 0.0
    try:
        while sim.time - t_start < args.seconds:
            sim.step(chunk)
            t = sim.time - t_start
            pos = sim.thorax_position()
            tilt = sim.tilt_deg()
            max_tilt = max(max_tilt, tilt)
            kind = terrain.terrain_type_at(pos[0]) if terrain else "flat"
            type_time[kind] += chunk * sim.timestep
            progress.append((t, pos[0]))
            # stuck: < 1 mm of +x progress over the last 3 s
            while progress and progress[0][0] < t - 3.0:
                progress.pop(0)
            now_stuck = t > 3.5 and pos[0] - progress[0][1] < 1.0
            if now_stuck and not stuck:
                stuck_events.append((t, float(pos[0]), kind))
                print(f"  [stuck] t={t:.1f}s x={pos[0]:.1f} on {kind}", flush=True)
            stuck = now_stuck
            now_fallen = tilt > 60.0
            if now_fallen and not fallen:
                fall_events.append((t, float(pos[0]), kind))
                print(f"  [fallen] t={t:.1f}s x={pos[0]:.1f} tilt={tilt:.0f} on {kind}", flush=True)
            fallen = now_fallen
            if renderer is not None:
                need_png = args.frames_dir is not None and t >= next_frame
                if writer is not None or need_png:
                    frame = renderer.render(sim.data, sim.time, pos, sim.heading())
                    if writer is not None:
                        writer.append_data(frame)
                    if need_png:
                        import imageio.v2 as iio

                        iio.imwrite(args.frames_dir / f"{args.difficulty}_t{t:05.1f}.png", frame)
                        next_frame += args.frame_every
            if t >= next_print:
                next_print += args.print_every
                rec = terrain.recycle_count if terrain else 0
                print(f"t={t:6.1f}s x={pos[0]:7.1f}mm y={pos[1]:5.1f} z={pos[2]:4.2f} "
                      f"tilt={tilt:4.1f}deg recycles={rec:3d} terrain={kind}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    wall = time.perf_counter() - wall0
    pos = sim.thorax_position()
    sim_t = sim.time - t_start
    print("\n=== summary ===")
    print(f"difficulty={args.difficulty} seed={args.seed} sim={sim_t:.1f}s wall={wall:.1f}s "
          f"({wall / (sim_t / sim.timestep) * 1e6:.0f} us/step incl. controller)")
    print(f"forward distance {pos[0]:.1f} mm (avg {pos[0] / sim_t:.1f} mm/s), final y {pos[1]:.2f}")
    if terrain:
        print(f"chunk recycles {terrain.recycle_count}, pool geoms {terrain.n_pool_geoms} "
              f"(active now {terrain.active_geom_count()})")
    print("time per terrain type: " + ", ".join(f"{k} {v:.1f}s" for k, v in type_time.most_common()))
    print(f"max tilt {max_tilt:.1f} deg; stuck episodes {len(stuck_events)} {stuck_events}; "
          f"falls {len(fall_events)} {fall_events}")
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
