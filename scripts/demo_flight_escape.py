"""Swatter vs the REAL brain with real wing flight (docs/FLIGHT.md section 7).

Runs scripts/demo_swatter.py's Trial on a Session with --flight (giant fibre ->
short-mode jump -> wings at take-off -> escape flight away from the paddle -> landing)
or without it (the short-mode jump alone, canonical dt 1e-4). Brain updates every
10 ms of fly time; start times --phase-step apart (distinct gait phases and
brain-window alignments). One brain worker.

    .venv/bin/python scripts/demo_flight_escape.py --flight --phases 3
    .venv/bin/python scripts/demo_flight_escape.py --phases 3 --post 0.6      # baseline
    .venv/bin/python scripts/demo_flight_escape.py --flight --real-vision --phases 2
"""
import argparse, importlib.util, json, sys, time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "demo_swatter", Path(__file__).resolve().parent / "demo_swatter.py")
ds = importlib.util.module_from_spec(spec); spec.loader.exec_module(ds)
from perpetualfly import AppConfig
from perpetualfly.app import Session
from perpetualfly.brain_link import BrainLink, BrainLinkConfig
from perpetualfly.interaction.swatter import Swatter, SwatterConfig, install_swatter

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flight", action="store_true")
    p.add_argument("--real-vision", action="store_true")
    p.add_argument("--no-fast", action="store_true",
                   help="old window-only readout (BrainLinkConfig.fast_path=False)")
    p.add_argument("--levels", default="1,2")
    p.add_argument("--sides", default="rear,left")
    p.add_argument("--phases", type=int, default=2)
    p.add_argument("--post", type=float, default=1.3)
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--frames", type=Path, default=None)
    p.add_argument("--phase-step", type=float, default=0.03)
    a = p.parse_args()

    cfg = AppConfig()
    cfg.terrain.difficulty = "flat"
    cfg.whip.enabled = False
    cfg.auto_perturb.enabled = False
    cfg.flight.enabled = a.flight
    if a.real_vision:
        cfg.real_vision.enabled = True
    cfg.brain = BrainLinkConfig(enabled=True, window=False, steer=True, actions=True,
                                window_s=0.02, sync_wait_s=0.05, sync_loom_only=True,
                                fast_path=not a.no_fast)
    quiet = lambda m: None
    link = BrainLink(cfg.brain, headless=True, say=quiet, start=False)
    if a.no_fast:  # observe the GF crossings (fast detector) without acting on them
        link.brain_cfg.fast_triggers = {"escape": cfg.brain.jump_escape_hz}
    link.start()
    args = argparse.Namespace(walk=1.0, post=a.post, chunk=0)
    try:
        sw = Swatter(SwatterConfig())
        s = Session(cfg, log=False, brain=link, say=quiet, world_extensions=[sw.extension])
        link.wait_ready()
        args.chunk = int(round(0.01 / s.sim.timestep))  # brain update every 10 ms of fly time
        h = install_swatter(s, swatter=sw, short_hz=60.0, flight=False, say=quiet,
                            vision=not a.real_vision)
        if s.flight is not None:  # (Session._install_features does this for cfg.swatter)
            sw.jump_probe = lambda: s.actions.active_name == "jump" or s.flight.busy
        if h.vision is None:  # real vision: Trial toggles vis.cfg.enabled
            h.vision = argparse.Namespace(cfg=argparse.Namespace(enabled=True), sent=[])
        trial = ds.Trial(s, link, h, args)
        rows = []
        for lv in [int(x) for x in a.levels.split(",")]:
            for side in a.sides.split(","):
                for k in range(a.phases):
                    fl0 = len(s.flight.events) if s.flight else 0
                    n_falls0 = s.metrics.n_falls
                    t1 = time.time()
                    want = a.frames is not None and not rows
                    args.walk = 1.0 + a.phase_step * k  # distinct trigger times vs the 20 ms brain windows
                    r = trial.run(lv, side, 0, True, frames=want)
                    r["phase"] = k
                    shots = r.pop("_shots", None)
                    if shots:
                        import imageio.v2 as iio
                        a.frames.mkdir(parents=True, exist_ok=True)
                        for off in (-0.1, -0.03, 0.0, 0.05, 0.3):
                            t, img = min(shots, key=lambda x: abs(x[0] - off))
                            iio.imwrite(a.frames / f"L{lv}{side}_{int(t*1000):+04d}ms.png", img)
                    fev = [(e.kind, e.source, {k2: v for k2, v in e.info.items()}) for e in
                           (s.flight.events[fl0:] if s.flight else [])]
                    r["flight_events"] = fev
                    r["falls"] = s.metrics.n_falls - n_falls0
                    r["tilt_end"] = round(s.sim.tilt_deg(), 1)
                    r["flight_state_end"] = s.flight.state if s.flight else None
                    r["wall_s"] = round(time.time() - t1, 1)
                    rows.append(r)
                    print(f"L{lv} {side} ph{k} {r['outcome']:7s} imp={r['impulse_uNs']:.1f} "
                          f"jump={ds.fmt(r['jump_trigger_rel_land'])} {r['jump_mode']} aim={r['fly_from_aim_mm']} "
                          f"falls={r['falls']} end={r['end_state']} tilt={r['tilt_end']} "
                          f"flight={[(k2, i.get('airtime_s'), i.get('distance_mm'), i.get('tilt_deg')) for k2, _, i in fev]} "
                          f"state={r['flight_state_end']} ({r['wall_s']}s)", flush=True)
        print(ds.summarize([dict(r, vision=True) for r in rows]))
        if a.json:
            a.json.write_text(json.dumps(rows, indent=1, default=str))
    finally:
        link.close()


if __name__ == '__main__':
    main()
