"""Headless demo of the action library (docs/ACTIONS.md).

Walks the fly, triggers each action during walking, lets it hand control back and
walk again, and prints metrics per action. Optionally records an MP4 (slow motion
while an action runs) and saves PNG frames at key moments (jump apex, freeze,
groom sweep, ...), rendered with perpetualfly.rendering.FrameRenderer.

    .venv/bin/python scripts/demo_actions.py
    .venv/bin/python scripts/demo_actions.py --actions jump --jump-boost 2 --trials 5
    .venv/bin/python scripts/demo_actions.py --record out.mp4 --frames /tmp/frames
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from perpetualfly import AppConfig, Simulation
from perpetualfly.actions import ActionManager, make_action, make_action_fly_factory
from perpetualfly.actions.registry import NEEDS_EXTRA_JOINTS

DEFAULT_ACTIONS = "jump,freeze,groom,back_away,turn_left,wings,proboscis"
PARAMS = {
    "freeze": {"duration": 1.0},
    "groom": {"duration": 3.0},
    "back_away": {"duration": 1.0},
    "turn_left": {"duration": 1.0},
    "turn_right": {"duration": 1.0},
    "wings": {"duration": 1.0},
    "proboscis": {"duration": 1.0},
}
# sim seconds per rendered frame while an action runs (normal: 1/30 s per frame)
SLOWMO_DT = {"jump": 1 / 600}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--actions", default=DEFAULT_ACTIONS, help=f"comma list (default {DEFAULT_ACTIONS})")
    p.add_argument("--trials", type=int, default=1, help="repetitions per action (different gait phases)")
    p.add_argument("--walk-before", type=float, default=0.5, help="s of walking before each trigger")
    p.add_argument("--walk-after", type=float, default=0.6, help="s of walking after hand-back")
    p.add_argument("--jump-boost", type=float, default=1.0, help="kp/force multiplier during the jump stroke")
    p.add_argument("--no-extra-joints", action="store_true",
                   help="default FlyGym model (skips wings / proboscis)")
    p.add_argument("--record", type=Path, default=None, help="MP4 path (slow motion around actions)")
    p.add_argument("--frames", type=Path, default=None, help="directory for PNG key frames")
    p.add_argument("--size", default="480x320", help="render WxH")
    p.add_argument("--json", type=Path, default=None, help="write per-trial metrics as JSON")
    return p.parse_args(argv)


class Recorder:
    def __init__(self, sim, args):
        self.sim = sim
        self.writer = None
        self.renderer = None
        self.frames_dir = args.frames
        if args.record is None and args.frames is None:
            return
        from perpetualfly.rendering import FrameRenderer

        w, h = (int(v) for v in args.size.split("x"))
        cfg = sim.cfg
        rc = replace(cfg.render, width=w, height=h)
        cc = replace(cfg.camera, distance=7.0)
        self.renderer = FrameRenderer(sim.model, rc, cc)
        if args.record is not None:
            import imageio.v2 as iio

            args.record.parent.mkdir(parents=True, exist_ok=True)
            self.writer = iio.get_writer(args.record, fps=30, codec="libx264", quality=6,
                                         macro_block_size=1)
        if self.frames_dir is not None:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._next_t = 0.0

    def frame(self) -> np.ndarray | None:
        if self.renderer is None:
            return None
        s = self.sim
        return self.renderer.render(s.data, s.time, s.thorax_position(), s.heading(),
                                    ground_z=0.0, tilt_deg=s.tilt_deg())

    def maybe_record(self, dt_frame: float) -> None:
        if self.writer is None or self.sim.time < self._next_t:
            return
        self._next_t = self.sim.time + dt_frame
        self.writer.append_data(self.frame())

    def closeup(self, az_offset_deg: float = 125.0, distance: float = 3.2) -> np.ndarray:
        """Front three-quarter close-up of the head / front legs (free camera)."""
        import mujoco as mj

        s = self.sim
        if not hasattr(self, "_cam"):
            self._cam = mj.MjvCamera()
            self._cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam = self._cam
        fwd = s.thorax_rotmat()[:, 0]
        cam.lookat[:] = s.thorax_position() + 0.4 * fwd
        cam.azimuth = np.degrees(s.heading()) + az_offset_deg
        cam.elevation = -12.0
        cam.distance = distance
        r = self.renderer.renderer
        r.update_scene(s.data, camera=cam, scene_option=self.renderer.scene_option)
        return r.render()

    def save(self, name: str, closeup: bool = False) -> None:
        if self.frames_dir is None or self.renderer is None:
            return
        import imageio.v2 as iio

        img = self.frame()
        if closeup:
            img = np.concatenate([img, self.closeup()], axis=1)
        iio.imwrite(self.frames_dir / f"{name}.png", img)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.renderer is not None:
            self.renderer.close()


def run_trial(sim, mgr, rec, name, params, args, trial):
    chunk = 10  # physics steps between checks (1 ms)
    normal_dt = 1 / 30

    def walk(sec):
        for _ in range(int(round(sec * 1e4 / chunk))):
            sim.step(chunk)
            rec.maybe_record(normal_dt)

    walk(args.walk_before + 0.037 * trial)  # vary the gait phase between trials
    p0, t0 = sim.thorax_position(), sim.time
    walk(0.3)
    v_before = (sim.thorax_position() - p0)[0] / (sim.time - t0)

    action = make_action(name, **params)
    mgr.trigger(action)
    t_start = sim.time
    z_max, tilt_max, snaps = -1.0, 0.0, set()
    apex_saved = False
    dt_frame = SLOWMO_DT.get(name, 1 / 60)
    while mgr.busy:
        sim.step(chunk)
        rec.maybe_record(dt_frame)
        z = sim.thorax_position()[2]
        tilt_max = max(tilt_max, sim.tilt_deg())
        ph = mgr.phase()
        tag = f"{name}_t{trial}"
        if trial == 0 and rec.frames_dir is not None:
            if name == "jump":
                if ph == "stroke" and "crouch" not in snaps:
                    rec.save(f"{tag}_1_crouch"); snaps.add("crouch")
                if ph == "flight" and z > z_max:
                    z_max = z
                elif ph == "flight" and z < z_max - 0.05 and not apex_saved:
                    rec.save(f"{tag}_2_apex"); apex_saved = True
                if ph == "landing" and "land" not in snaps:
                    rec.save(f"{tag}_3_touchdown"); snaps.add("land")
            else:
                t = sim.time - t_start
                marks = {"groom": (0.6, 1.0, 1.4, 2.2), "freeze": (0.5,),
                         "wings": (0.5,), "proboscis": (0.5,), "back_away": (0.5,),
                         "turn_left": (0.5,)}.get(name, ())
                for mk in marks:
                    if t >= mk and mk not in snaps:
                        rec.save(f"{tag}_{mk:.1f}s", closeup=True); snaps.add(mk)
        if sim.time - t_start > 30:
            raise RuntimeError(f"{name} did not finish")
    ev = [e for e in mgr.history if e.kind in ("end", "cancel")][-1]
    p1, t1 = sim.thorax_position(), sim.time
    walk(args.walk_after)
    v_after = (sim.thorax_position() - p1)[0] / (sim.time - t1)
    info = {k: v for k, v in ev.info.items() if k != "feet_x_at_stroke_mm"}
    return {"action": name, "trial": trial, "duration_s": ev.time - t_start,
            "v_before_mm_s": v_before, "v_after_mm_s": v_after, "max_tilt_deg": tilt_max,
            "tilt_after_deg": sim.tilt_deg(), **info}


def main(argv=None) -> int:
    args = parse_args(argv)
    names = [n.strip() for n in args.actions.split(",") if n.strip()]
    extra = not args.no_extra_joints
    if not extra:
        skipped = [n for n in names if n in NEEDS_EXTRA_JOINTS]
        names = [n for n in names if n not in NEEDS_EXTRA_JOINTS]
        if skipped:
            print(f"skipping {skipped} (need extra joints)")
    sim = Simulation(AppConfig(), fly_factory=make_action_fly_factory() if extra else None)
    mgr = ActionManager(sim)
    rec = Recorder(sim, args)
    results = []
    wall = time.time()
    try:
        for name in names:
            params = dict(PARAMS.get(name, {}))
            if name == "jump":
                params["boost"] = args.jump_boost
            for trial in range(args.trials):
                sim.reset()
                r = run_trial(sim, mgr, rec, name, params, args, trial)
                results.append(r)
                keys = ("apex_dz_mm", "airtime_s", "distance_mm", "forward_mm", "landed_upright",
                        "drift_mm", "turn_deg", "max_tilt_deg", "v_before_mm_s", "v_after_mm_s")
                shown = []
                for k in keys:
                    if k in r:
                        v = r[k]
                        shown.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
                print(f"{name:10s} #{trial}: " + "  ".join(shown), flush=True)
    finally:
        rec.close()
    print(f"done in {time.time() - wall:.1f} s wall")
    if args.json:
        args.json.write_text(json.dumps(results, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
