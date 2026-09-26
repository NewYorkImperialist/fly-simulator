"""Flapping-wing flight prototype demos (see docs/FLIGHT.md).

    .venv/bin/python scripts/demo_flight.py tether            # lift tables
    .venv/bin/python scripts/demo_flight.py effect            # control-effectiveness matrix
    .venv/bin/python scripts/demo_flight.py hover --duration 2 [--forward 50]
    .venv/bin/python scripts/demo_flight.py takeoff [--at apex] [--land]

Rendering (offscreen only): ``--frames DIR`` saves a few PNGs, ``--record out.mp4``
a slow-motion clip (``--slowmo``, default 20x).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import mujoco as mj
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perpetualfly.flight import FlightSimulation, WingbeatParams, measure_tethered  # noqa: E402
from perpetualfly.flight.control import HoverController, base_wingbeat  # noqa: E402
from perpetualfly.flight.wingbeat import DEG  # noqa: E402


# ---------------------------------------------------------------- rendering
class Viz:
    def __init__(self, sim: FlightSimulation, args) -> None:
        self.sim = sim
        self.renderer = self.writer = None
        self.frames = args.frames
        self.slowmo = args.slowmo
        self._next = 0.0
        if args.frames is None and args.record is None:
            return
        w, h = (int(v) for v in args.size.split("x"))
        self.renderer = mj.Renderer(sim.model, h, w)
        self.cam = mj.MjvCamera()
        self.cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.opt = mj.MjvOption()
        if args.frames is not None:
            args.frames.mkdir(parents=True, exist_ok=True)
        if args.record is not None:
            import imageio.v2 as iio

            args.record.parent.mkdir(parents=True, exist_ok=True)
            self.writer = iio.get_writer(args.record, fps=30, codec="libx264", quality=5,
                                         macro_block_size=1)

    def render(self, distance: float = 8.0, az_offset: float = -60.0, elevation: float = -15.0,
               lookat=None) -> np.ndarray:
        s = self.sim
        self.cam.lookat[:] = s.com() if lookat is None else lookat
        self.cam.distance = distance
        self.cam.azimuth = math.degrees(s.heading()) + az_offset
        self.cam.elevation = elevation
        self.renderer.update_scene(s.data, camera=self.cam, scene_option=self.opt)
        return self.renderer.render()

    def save(self, name: str, **kw) -> None:
        if self.renderer is None or self.frames is None:
            return
        import imageio.v2 as iio

        iio.imwrite(self.frames / f"{name}.png", self.render(**kw))

    def maybe_record(self) -> None:
        if self.writer is None or self.sim.time < self._next:
            return
        self._next = self.sim.time + self.slowmo_dt
        self.writer.append_data(self.render())

    @property
    def slowmo_dt(self) -> float:
        return 1.0 / (30.0 * self.slowmo)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self.renderer is not None:
            self.renderer.close()


def run(sim: FlightSimulation, seconds: float, viz: Viz | None = None, chunk: int = 20) -> None:
    n = int(round(seconds / sim.timestep))
    if viz is None or viz.writer is None:
        sim.step(n)
        return
    for _ in range(0, n, chunk):
        sim.step(chunk)
        viz.maybe_record()


# ---------------------------------------------------------------- tethered
def cmd_tether(args) -> None:
    sim = FlightSimulation()
    base = base_wingbeat()
    cyc = 3 if args.quick else 6
    freqs = (200.0, 218.0, 250.0) if args.quick else (180.0, 200.0, 218.0, 235.0, 250.0)
    amps = (140.0, 160.0) if args.quick else (120.0, 140.0, 151.0, 160.0, 170.0)
    print(f"weight {sim.weight:.2f} uN; base: aoa 45 deg, rotation phase -0.5 rad")
    print("\nLift / weight (tethered, stroke plane horizontal, cycle average):\n")
    print("| stroke (deg) | " + " | ".join(f"{f:.0f} Hz" for f in freqs) + " |")
    print("|---" * (len(freqs) + 1) + "|")
    fmax = 0.0
    for a in amps:
        row = []
        for f in freqs:
            P = replace(base, freq=f).symmetric(amplitude=a * DEG)
            r = measure_tethered(sim, P, cycles=cyc, settle_cycles=2)
            row.append(f"{r.lift_over_weight:.2f}")
            fmax = max(fmax, r.max_actuator_force)
        print(f"| {a:.0f} | " + " | ".join(row) + " |", flush=True)
    print(f"\nmax net wing servo torque {fmax:.1f} uN*mm")
    print("\nStroke-plane tilt (both wings, via deviation), 218 Hz / 160 deg:\n")
    print("| tilt (deg) | Fx / W (forward) | Fz / W | pitch torque (uN*mm, + nose-down) |")
    print("|---|---|---|---|")
    for tilt in (-20.0, -10.0, 0.0, 10.0, 20.0):
        r = measure_tethered(sim, base.symmetric(tilt=tilt * DEG), cycles=cyc, settle_cycles=2)
        print(f"| {tilt:+.0f} | {r.force_world[0] / r.weight:+.3f} | {r.force_world[2] / r.weight:.3f} "
              f"| {r.torque_com[1]:+.3f} |", flush=True)
    print("\nAngle of attack / rotation timing, 218 Hz / 160 deg (lift / weight):\n")
    print("| rotation phase (rad) | aoa 35 | aoa 45 | aoa 55 |")
    print("|---|---|---|---|")
    for rp in (0.3, 0.0, -0.5, -0.9):
        row = []
        for aoa in (35.0, 45.0, 55.0):
            P = base.symmetric(aoa_down=aoa * DEG, aoa_up=aoa * DEG, rot_phase=rp)
            row.append(f"{measure_tethered(sim, P, cycles=cyc, settle_cycles=2).lift_over_weight:.2f}")
        print(f"| {rp:+.1f} | " + " | ".join(row) + " |", flush=True)


def cmd_effect(args) -> None:
    sim = FlightSimulation()
    base = base_wingbeat()
    W = sim.weight

    def meas(P):
        r = measure_tethered(sim, P, cycles=4, settle_cycles=2)
        return np.r_[r.force_world[2], r.force_world[0], r.torque_com]

    b = meas(base)
    print(f"base wrench Fz {b[0]:.2f} uN ({b[0] / W:.3f} W), Fx {b[1]:.2f}, "
          f"T {np.round(b[2:], 3)} uN*mm")
    d = 10 * DEG

    def side(P, s, **kw):
        w = getattr(P, s)
        return replace(P, **{s: replace(w, **{k: getattr(w, k) + v for k, v in kw.items()})})

    tests = {
        "amplitude both": side(side(base, "left", amplitude=d), "right", amplitude=d),
        "mean stroke both": side(side(base, "left", mean_stroke=d), "right", mean_stroke=d),
        "amplitude L+/R-": side(side(base, "left", amplitude=d), "right", amplitude=-d),
        "tilt L+/R-": side(side(base, "left", tilt=d), "right", tilt=-d),
        "tilt both": side(side(base, "left", tilt=d), "right", tilt=d),
    }
    print("\nper rad:        Fz      Fx      Tx      Ty      Tz")
    for name, P in tests.items():
        x = (meas(P) - b) / d
        print(f"{name:16s}" + " ".join(f"{v:7.2f}" for v in (x[0], x[1], x[2], x[3], x[4])),
              flush=True)


# ---------------------------------------------------------------- free flight
def hover_stats(log: np.ndarray, t0: float, t1: float, target: np.ndarray) -> dict:
    L = log[(log[:, 0] >= t0) & (log[:, 0] <= t1)]
    com = L[:, 1:4]
    e = np.degrees(L[:, 18:21])
    return {
        "window_s": (t0, t1),
        "com_start": com[0], "com_end": com[-1],
        "drift_xy_mm": float(np.linalg.norm(com[-1, :2] - com[0, :2])),
        "max_dev_from_target_mm": float(np.linalg.norm(com - target, axis=1).max()),
        "z_rms_mm": float(np.sqrt(np.mean((com[:, 2] - target[2]) ** 2))),
        "att_rms_deg": np.sqrt(np.mean(e ** 2, axis=0)),
        "att_max_deg": np.abs(e).max(axis=0),
        "mean_speed_mm_s": np.linalg.norm(L[:, 4:7].mean(axis=0)),
    }


def print_stats(name: str, st: dict) -> None:
    print(f"[{name}] t = {st['window_s'][0]:.2f}..{st['window_s'][1]:.2f} s")
    print(f"  COM start {np.round(st['com_start'], 2)} end {np.round(st['com_end'], 2)} mm; "
          f"xy drift {st['drift_xy_mm']:.2f} mm; max distance from target "
          f"{st['max_dev_from_target_mm']:.2f} mm; z rms {st['z_rms_mm']:.2f} mm")
    print(f"  attitude error vs hover posture (roll, pitch, yaw) rms "
          f"{np.round(st['att_rms_deg'], 2)} deg, max {np.round(st['att_max_deg'], 1)} deg")


def cmd_hover(args) -> None:
    sim = FlightSimulation()
    viz = Viz(sim, args)
    sim.wingbeat.params = base_wingbeat()
    z0 = args.height
    sim.place((0.0, 0.0, z0))
    ctrl = HoverController(sim)
    ctrl.target.pos = sim.com()
    sim.flight_controller = ctrl
    sim.flapping = True
    t_start = sim.time
    wall = time.perf_counter()
    kick_done = False
    try:
        # hover
        t_hover = args.duration
        if args.kick > 0:
            run(sim, 0.5 * t_hover, viz)
            # torque kick (a gust): args.kick uN*mm for 5 ms on the thorax, about the
            # stroke-frame (level) axis: roll = x, pitch = -y (nose-up), yaw = z
            axis = {"roll": (1, 0, 0), "pitch": (0, -1, 0), "yaw": (0, 0, 1)}[args.kick_axis]
            sim.data.xfrc_applied[sim.thorax_body_id, 3:] = args.kick * np.array(axis, float)
            run(sim, 0.005, viz)
            sim.data.xfrc_applied[sim.thorax_body_id] = 0.0
            kick_done = True
            run(sim, 0.5 * t_hover - 0.005, viz)
        else:
            if viz.frames is not None:
                run(sim, t_hover - 0.01, viz)
                # two frames half a wingbeat apart: wings at opposite strokes
                viz.save("hover", distance=7.0)
                viz.save("wings_closeup_a", distance=4.0, az_offset=-100.0, elevation=5.0)
                run(sim, 0.5 / sim.wingbeat.params.freq, viz)
                viz.save("wings_closeup_b", distance=4.0, az_offset=-100.0, elevation=5.0)
                viz.save("wings_top", distance=5.0, az_offset=0.0, elevation=-80.0)
                run(sim, 0.01 - 0.5 / sim.wingbeat.params.freq, viz)
            else:
                run(sim, t_hover, viz)
        log = np.array(ctrl.log)
        st = hover_stats(log, t_start + 0.3, sim.time, ctrl.target.pos)
        print_stats("hover" + (f" + {args.kick_axis} kick" if kick_done else ""), st)
        if args.forward:
            yaw = ctrl.target.yaw
            h = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            t_f = sim.time
            for k in range(10):  # ramp the velocity over 0.2 s
                ctrl.target.vel = h * args.forward * (k + 1) / 10
                run(sim, 0.02, viz)
            run(sim, args.forward_s, viz)
            log = np.array(ctrl.log)
            L = log[log[:, 0] >= sim.time - 0.5]
            v = L[:, 4:7].mean(axis=0)
            e = np.degrees(L[:, 18:21])
            print(f"[forward {args.forward:.0f} mm/s] last 0.5 s: mean velocity {np.round(v, 1)} mm/s, "
                  f"z {L[-1, 3]:.2f} mm, attitude error rms {np.round(np.sqrt((e ** 2).mean(0)), 2)} deg, "
                  f"stroke-plane tilt T {np.degrees(L[:, 17].mean()):.1f} deg, "
                  f"distance {np.linalg.norm(L[-1, 1:3] - log[log[:, 0] >= t_f][0, 1:3]):.1f} mm")
            viz.save("forward", distance=8.0)
    finally:
        viz.close()
    el = time.perf_counter() - wall
    print(f"sim {sim.time - t_start:.2f} s in {el:.1f} s wall: {(sim.time - t_start) / el:.2f}x real time "
          f"(dt {sim.timestep:g} s)")


def cmd_takeoff(args) -> None:
    from perpetualfly.actions.base import ActionManager
    from perpetualfly.actions.jump import Jump

    sim = FlightSimulation()
    viz = Viz(sim, args)
    sim.leg_mode = "walk"
    mgr = ActionManager(sim)
    run(sim, 0.3)
    mgr.trigger(Jump())
    ctrl = None
    t_start = None
    min_z = np.inf
    try:
        for _ in range(int(0.3 / sim.timestep)):
            sim.step(1)
            viz.maybe_record()
            if mgr.phase() == "flight":
                mj.mj_subtreeVel(sim.model, sim.data)
                vz = sim.data.subtree_linvel[sim.thorax_body_id][2]
                if args.at == "takeoff" or vz <= 0.0:
                    break
        else:
            print("jump never reached the flight phase")
            return
        com0 = sim.com()
        print(f"wings start at the {args.at}: t {sim.time:.3f} s, COM {np.round(com0, 2)} mm, "
              f"tilt {sim.tilt_deg():.1f} deg, vz {vz:.0f} mm/s")
        viz.save("takeoff_start", distance=7.0)
        mgr.detach()
        sim.leg_mode = "flight"
        sim.wingbeat.params = base_wingbeat()
        sim.wingbeat.start(0.01)
        sim.flapping = True
        ctrl = HoverController(sim)
        ctrl.target.pos = com0 + np.array([0.0, 0.0, args.climb])
        ctrl.target.yaw = sim.heading()
        sim.flight_controller = ctrl
        t_start = sim.time
        n = int(round(args.duration / sim.timestep))
        for i in range(0, n, 20):
            sim.step(20)
            viz.maybe_record()
            min_z = min(min_z, sim.com()[2])
            if abs(sim.time - t_start - 0.05) < 0.5e-3:
                viz.save("takeoff_50ms", distance=7.0)
        viz.save("takeoff_hover", distance=7.0)
        log = np.array(ctrl.log)
        print(f"lowest COM after wing start {min_z:.2f} mm (legs hang ~1 mm below the COM)")
        st = hover_stats(log, t_start + min(0.5, 0.5 * args.duration), sim.time, ctrl.target.pos)
        print_stats("after take-off", st)
        if args.land:
            ctrl.target.vel = np.array([0.0, 0.0, -args.land_speed])
            touched = False
            for _ in range(int(3.0 / sim.timestep / 20)):
                sim.step(20)
                viz.maybe_record()
                if mgr.body.leg_contacts(sim).sum() >= 1:
                    touched = True
                    break
            if not touched:
                print("no touchdown within 3 s")
                return
            print(f"touchdown t {sim.time - t_start:.2f} s after wing start, COM {np.round(sim.com(), 2)}, "
                  f"tilt {sim.tilt_deg():.1f} deg")
            # Touchdown: feet stick (adhesion), the wings keep beating while the body
            # pitches down to the walking posture and the lift is reduced, then stop.
            sim.leg_mode = "stance"
            ctrl.target.vel = np.zeros(3)
            ctrl.target.pos = sim.com() - np.array([0.0, 0.0, 0.3])
            ctrl.target.pitch = math.radians(sim.stroke_plane_deg)
            run(sim, args.land_pitch_s, viz)
            sim.flight_controller = None
            sim.wingbeat.stop(0.01)
            run(sim, 0.012, viz)
            sim.flapping = False
            print(f"wings stopped: COM {np.round(sim.com(), 2)}, tilt {sim.tilt_deg():.1f} deg")
            sim.leg_mode = "walk"  # the gait controller takes over (adhesion on)
            sim.controller.reset()
            run(sim, 0.5, viz)
            print(f"after 0.5 s of walking: COM {np.round(sim.com(), 2)}, tilt {sim.tilt_deg():.1f} deg")
            viz.save("landed", distance=7.0)
    finally:
        viz.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tether")
    t.add_argument("--quick", action="store_true")
    sub.add_parser("effect")
    h = sub.add_parser("hover")
    h.add_argument("--duration", type=float, default=2.0)
    h.add_argument("--height", type=float, default=10.0)
    h.add_argument("--forward", type=float, default=0.0, help="then fly forward at this speed (mm/s)")
    h.add_argument("--forward-s", type=float, default=1.0)
    h.add_argument("--kick", type=float, default=0.0, help="torque gust (uN*mm, 5 ms) mid-hover")
    h.add_argument("--kick-axis", choices=("roll", "pitch", "yaw"), default="pitch")
    k = sub.add_parser("takeoff")
    k.add_argument("--at", choices=("takeoff", "apex"), default="takeoff")
    k.add_argument("--duration", type=float, default=1.5)
    k.add_argument("--climb", type=float, default=3.0)
    k.add_argument("--land", action="store_true")
    k.add_argument("--land-speed", type=float, default=15.0)
    k.add_argument("--land-pitch-s", type=float, default=0.2)
    for p in (h, k):
        p.add_argument("--frames", type=Path, default=None)
        p.add_argument("--record", type=Path, default=None)
        p.add_argument("--slowmo", type=float, default=20.0)
        p.add_argument("--size", default="480x360")
    args = ap.parse_args()
    {"tether": cmd_tether, "effect": cmd_effect, "hover": cmd_hover, "takeoff": cmd_takeoff}[args.cmd](args)


if __name__ == "__main__":
    main()
