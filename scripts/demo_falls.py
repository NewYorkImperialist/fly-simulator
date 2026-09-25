"""Headless fall-detection / metrics / logging demo.

Walk 5 s -> mild lateral shove -> walk -> strong lateral shove that rolls the fly
over -> keep simulating. Prints fall-detector transitions and run metrics, and
writes a run directory (runs/<timestamp>/ by default).

    .venv/bin/python scripts/demo_falls.py [--runs-dir runs] [--mild 3.5] [--strong 10]

Shoves are written straight into data.xfrc_applied (world-frame force at the
thorax COM, uN) from a pre-step hook; forces are given in multiples of body weight.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perpetualfly import AppConfig, Simulation  # noqa: E402
from perpetualfly.metrics import (  # noqa: E402
    FallDetector,
    FallDetectorConfig,
    LoggingConfig,
    RunLogger,
    RunMetrics,
)


class Shover:
    """Minimal stand-in for the perturbation module: constant force for a duration."""

    def __init__(self, sim: Simulation) -> None:
        self.sim = sim
        self.until = -1.0
        self.force = np.zeros(3)
        sim.pre_step_hooks.append(self)

    def shove(self, direction, multiple_of_weight: float, duration: float) -> dict:
        weight = self.sim.fly_mass * 9810.0  # uN (mass g * gravity mm/s^2)
        d = np.asarray(direction, float)
        self.force = d / np.linalg.norm(d) * multiple_of_weight * weight
        self.until = self.sim.time + duration
        return {"time": self.sim.time, "direction": d.tolist(),
                "magnitude": float(np.linalg.norm(self.force)), "duration": duration,
                "body": "thorax", "multiple_of_weight": multiple_of_weight}

    def __call__(self, sim: Simulation) -> None:
        tid = sim.thorax_body_id
        sim.data.xfrc_applied[tid, :3] = self.force if sim.time < self.until else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--mild", type=float, default=3.5, help="x body weight, 10 ms")
    p.add_argument("--strong", type=float, default=10.0, help="x body weight, 10 ms")
    p.add_argument("--after", type=float, default=5.0, help="seconds after strong shove")
    args = p.parse_args()

    cfg = AppConfig()
    det_cfg = FallDetectorConfig()
    log_cfg = LoggingConfig(runs_dir=args.runs_dir, sample_hz=50.0)
    sim = Simulation(cfg)
    detector = FallDetector(sim, det_cfg)
    metrics = RunMetrics().attach(sim, detector)
    logger = RunLogger(log_cfg, config={"app": cfg, "fall_detector": det_cfg,
                                        "logging": log_cfg, "demo": vars(args)},
                       terrain_type_fn=lambda x: "flat").attach(sim, detector, metrics)
    detector.add_listener(lambda ev: print(
        f"  [{ev.time:7.3f}s] {ev.from_state.value:>12} -> {ev.to_state.value:<12} "
        f"{ev.kind:<12} ({ev.reason}; tilt {ev.tilt_deg:.0f}deg, h {ev.height:.2f}mm)"
        + (f" recovery {ev.recovery_time:.2f}s" if ev.recovery_time is not None else ""),
        flush=True))
    shover = Shover(sim)
    print(f"run dir: {logger.run_dir}", flush=True)

    def walk(seconds: float) -> None:
        end = sim.time + seconds
        while sim.time < end:
            sim.step(1000)  # 0.1 s
            if round(sim.time * 10) % 10 == 0:
                print(metrics.summary_line(), flush=True)

    wall = time.perf_counter()
    try:
        walk(5.0)
        print(f"--- mild shove: {args.mild} x weight, +y (left), 10 ms", flush=True)
        metrics.record_hit(shover.shove([0, 1, 0], args.mild, 0.010))
        walk(5.0)
        print(f"--- strong shove: {args.strong} x weight, -y (right), 10 ms", flush=True)
        metrics.record_hit(shover.shove([0, -1, 0], args.strong, 0.010))
        walk(args.after)
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        logger.close()
    print(f"wall {time.perf_counter() - wall:.1f}s")
    s = metrics.summary()
    for k in ("run_time_s", "distance_mm", "average_speed_mm_s", "n_falls", "n_recoveries",
              "n_hits", "n_hits_survived", "longest_jog_interval_s", "state",
              "falls_per_km", "recovery_percentage", "max_force_survived_uN"):
        print(f"  {k:28s} {s[k]}")
    print(f"logs in {logger.run_dir}")
    sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
