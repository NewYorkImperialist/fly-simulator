"""Standalone obstacle-course runner (headless by default).

    .venv/bin/python scripts/run_course.py tutorial
    .venv/bin/python scripts/run_course.py gauntlet --controller cpg
    .venv/bin/python scripts/run_course.py slalom --window
    .venv/bin/python scripts/run_course.py gauntlet --record gauntlet.mp4
    .venv/bin/python scripts/run_course.py courses/my_course.json --loop --laps 3
    .venv/bin/python scripts/run_course.py --list

Builds the app's ``Session`` (terrain pool, whip, falls, metrics, logger) exactly
like ``fly_simulator.app.run`` and installs the course with
``fly_simulator.course.install_course``. Results go to <run dir>/course_results.json
and finished runs to <runs dir>/leaderboard.json (keyed by course + controller).
Window keys: Q/ESC quit, X explicit reset (to the last checkpoint), C camera.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_simulator.app import Session  # noqa: E402
from fly_simulator.config import AppConfig  # noqa: E402
from fly_simulator.course import CourseOptions, builtin_courses, install_course  # noqa: E402
from fly_simulator.course import leaderboard as lb  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("course", nargs="?", default="tutorial", help="built-in name or file path")
    p.add_argument("--list", action="store_true", help="list built-in courses + leaderboard")
    p.add_argument("--controller", choices=["hybrid", "cpg"], default="hybrid")
    p.add_argument("--controller-seed", type=int, default=0)
    p.add_argument("--window", action="store_true", help="live OpenCV window")
    p.add_argument("--record", type=Path, default=None, help="write an MP4 of the run")
    p.add_argument("--frames-dir", type=Path, default=None,
                   help="save PNG frames (with HUD) when the thorax crosses --frame-x")
    p.add_argument("--frame-x", default="", help="comma list of x positions (mm) for PNGs")
    p.add_argument("--camera", choices=["follow", "side", "top"], default="follow")
    p.add_argument("--size", default="960x640", help="frame size WxH")
    p.add_argument("--loop", action="store_true", help="start a new lap after the finish")
    p.add_argument("--laps", type=int, default=1, help="with --loop: stop after N laps")
    p.add_argument("--no-steer", action="store_true", help="plain heading hold (no route)")
    p.add_argument("--timeout", type=float, default=None, help="override the DNF timeout (s)")
    p.add_argument("--respawn-after", type=float, default=None, help="FALLEN s before respawn")
    p.add_argument("--hit-mode", choices=["whip", "shove"], default="whip",
                   help="gauntlet cracks with the physical whip or external-force shoves")
    p.add_argument("--no-log", action="store_true", help="no run dir (results not saved)")
    p.add_argument("--runs-dir", default="runs", help="run dirs + leaderboard.json")
    p.add_argument("--no-leaderboard", action="store_true")
    p.add_argument("--brain", action="store_true", help="connectome brain (needs data)")
    p.add_argument("--brain-steer", action="store_true", help="brain drives walking")
    p.add_argument("--quiet", action="store_true", help="no per-second status lines")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        print("built-in courses:", ", ".join(builtin_courses()))
        print(lb.format_board(Path(args.runs_dir) / "leaderboard.json"))
        return 0
    cfg = AppConfig()
    cfg.terrain.difficulty = "flat"
    cfg.controller.kind = args.controller
    cfg.controller.seed = args.controller_seed
    cfg.session.hit_mode = args.hit_mode
    cfg.session.auto_reset_after_s = None  # the course respawns instead
    cfg.session.fall_hint_after_s = 0.0
    cfg.logging.runs_dir = args.runs_dir
    cfg.camera.mode = args.camera
    w, h = (int(v) for v in args.size.lower().split("x"))
    cfg.render.width, cfg.render.height = w, h
    brain = None
    if args.brain or args.brain_steer:
        from fly_simulator.brain_link import BrainLink, missing_requirements

        cfg.brain.enabled = True
        cfg.brain.steer = args.brain_steer
        cfg.brain.window = False
        problem = missing_requirements(cfg.brain)
        if problem:
            print(f"ERROR: {problem}", file=sys.stderr)
            return 2
        brain = BrainLink(cfg.brain, headless=True)
    session = Session(cfg, log=not args.no_log, brain=brain)
    opts = CourseOptions(after_finish="loop" if args.loop else "stop", timeout_s=args.timeout,
                         respawn_after_s=args.respawn_after,
                         steer=False if args.no_steer else None,
                         leaderboard="" if args.no_leaderboard else None)
    course = install_course(session, args.course, cfg, options=opts)
    sim = session.sim
    if brain is not None:
        brain.wait_ready()
    lay = course.layout
    print(f"course {course.spec.name}: {lay.length:.0f} mm, {len(lay.sections)} sections, "
          f"{len(lay.checkpoints)} checkpoints, {len(lay.gates)} gates, "
          f"{len(lay.triggers)} triggers | controller {course.mode} | steering "
          f"{'route' if course.steer else 'off' if course.can_steer else 'n/a (cpg)'} | "
          f"timeout {course.timeout_s:g}s", flush=True)

    renderer = viewer = writer = None
    frame_x = sorted(float(v) for v in args.frame_x.split(",") if v.strip())
    if args.window or args.record or args.frames_dir:
        from fly_simulator.rendering import FrameRenderer

        renderer = FrameRenderer(sim.model, cfg.render, cfg.camera)
        course.respawn_listeners.append(lambda c: renderer.camera.reset())
    if args.window:
        from fly_simulator.interaction import LiveViewer

        viewer = LiveViewer(f"Fly Simulator course: {course.spec.name}",
                            frame_size=(cfg.render.width, cfg.render.height))
    chunk = cfg.render.render_every_steps
    if args.record:
        import imageio.v2 as iio

        args.record.parent.mkdir(parents=True, exist_ok=True)
        writer = iio.get_writer(args.record, fps=1.0 / (chunk * sim.timestep), codec="libx264",
                                quality=7, macro_block_size=8)
    if args.frames_dir:
        args.frames_dir.mkdir(parents=True, exist_ok=True)

    def frame():
        x, y, _ = sim.thorax_position()
        return renderer.render(sim.data, sim.time, sim.thorax_position(), sim.heading(),
                               ground_z=session.ground_height(x, y), tilt_deg=sim.tilt_deg())

    def hud():
        m = session.metrics
        return course.hud_lines() + [
            f"{session.detector.state.value:<12} speed {m.current_speed:5.1f} mm/s  "
            f"terrain {session.terrain_here()}"]

    quit_reason = "course over"
    next_print = 1.0
    wall0 = time.perf_counter()
    max_rt = course.timeout_s * max(args.laps, 1) + 30.0 * max(args.laps, 1)
    try:
        while True:
            sim.step(chunk)
            session.after_physics()
            if renderer is not None and (writer or viewer or
                                         (frame_x and sim.thorax_position()[0] >= frame_x[0])):
                import cv2

                from fly_simulator.interaction.viewer import compose_frame

                img = frame()
                if writer is not None:
                    writer.append_data(cv2.cvtColor(compose_frame(img, hud()), cv2.COLOR_BGR2RGB))
                if frame_x and sim.thorax_position()[0] >= frame_x[0]:
                    x = frame_x.pop(0)
                    out = args.frames_dir / f"{course.spec.name}_x{x:05.1f}.png"
                    cv2.imwrite(str(out), compose_frame(img, hud()))
                    print(f"[frame] {out}", flush=True)
                if viewer is not None:
                    viewer.show(img, hud())
                    for k in viewer.poll_keys(1):
                        if k in ("q", "escape"):
                            quit_reason = "quit"
                        elif k == "x":
                            session.reset("manual")
                            renderer.camera.reset()
                        elif k == "c":
                            renderer.camera.cycle_mode()
                    if not viewer.is_open():
                        quit_reason = "window closed"
                    if quit_reason != "course over":
                        break
            rt = session.run_time()
            if not args.quiet and rt >= next_print:
                next_print += 1.0
                print("  " + " | ".join(course.hud_lines()), flush=True)
            if course.done or (args.loop and len(course.laps) >= args.laps):
                break
            if rt > max_rt:
                quit_reason = "max time"
                break
    except KeyboardInterrupt:
        quit_reason = "Ctrl-C"
    finally:
        for c in (writer, viewer, renderer):
            if c is not None:
                c.close()
        session.close(quit_reason)
        if brain is not None:
            brain.close()
    wall = time.perf_counter() - wall0
    for r in course.laps:
        print(f"[{r['status'].upper()}] {r['course']} lap {r['lap']} ({r['controller_mode']}): "
              f"total {r['total_time_s']:.2f}s = race {r['time_s']:.2f}s + penalties "
              f"{r['penalty_s']:.1f}s | falls {r['n_falls']} respawns {r['n_respawns']} | "
              f"gates {r['gates_passed']}/{r['gates_total']} | progress "
              f"{r['progress_mm']:.0f}/{r['course_length_mm']:.0f} mm"
              + (f" | rank #{r['rank']}" if r.get("rank") else ""), flush=True)
        if r["splits"]:
            print("   splits: " + ", ".join(f"{s['name']} {s['time_s']:.2f}s" for s in r["splits"]))
    print(f"wall {wall:.0f}s | results: {course.results_path() or '(not logged)'}", flush=True)
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
