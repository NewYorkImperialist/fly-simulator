"""Flyswatter vs. the REAL connectome brain, headless (docs/SWATTER.md).

For each swat (level x gait phase x side): fresh reset, walk ``--walk`` s (+ k x
10.5 ms for gait phase k), swat, observe until the swatter is back + ``--post`` s,
with ``--brain-actions`` logic (ActionManager + brain triggers: giant fiber >
60 Hz -> Jump, short mode above ``--short-hz``; ``--flight`` adds the escape-flight
emulation away from the paddle, off by default because it is not wing physics). Reports per swat: loom onset, GF onset (first BrainState window with GF >
0 / above the jump threshold, and when it reached the fly), jump trigger, take-off,
paddle contact, outcome (hit / dodged / miss) and the measured impulse. With
``--compare`` every swat is repeated with vision off (nothing can warn the fly).

    .venv/bin/python scripts/demo_swatter.py --levels 1,2,3,4 --phases 3 --compare
    .venv/bin/python scripts/demo_swatter.py --levels 1,4 --phases 1 --frames /tmp/frames

One brain worker process; the fly and the brain are paced together (BrainLink).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from fly_simulator import AppConfig
from fly_simulator.app import Session
from fly_simulator.brain_link import BrainLink, BrainLinkConfig
from fly_simulator.interaction.swatter import Swatter, SwatterConfig, install_swatter


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", default="1,2,3,4")
    p.add_argument("--sides", default="rear")
    p.add_argument("--phases", type=int, default=2, help="gait phases per level")
    p.add_argument("--walk", type=float, default=1.0)
    p.add_argument("--post", type=float, default=0.4)
    p.add_argument("--compare", action="store_true", help="also run each swat with vision off")
    p.add_argument("--window", type=float, default=0.02, help="BrainState window (s)")
    p.add_argument("--clock", type=float, default=None, help="BrainLinkConfig.clock_every_s")
    p.add_argument("--sync", type=float, default=0.05,
                   help="BrainLinkConfig.sync_wait_s (wall s; 0 = off)")
    p.add_argument("--sync-always", action="store_true", help="sync even without a loom")
    p.add_argument("--chunk", type=int, default=100, help="physics steps between brain updates")
    p.add_argument("--short-hz", type=float, default=60.0)
    p.add_argument("--no-short", action="store_true", help="always long-mode jumps")
    p.add_argument("--flight", action="store_true",
                   help="escape-flight emulation (external force; not wing physics)")
    p.add_argument("--swatter", default="{}", help="JSON overrides of SwatterConfig")
    p.add_argument("--frames", type=Path, default=None,
                   help="save PNGs around the first dodge and the first hit")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--no-fast", action="store_true",
                   help="old window-only readout (BrainLinkConfig.fast_path=False)")
    return p.parse_args(argv)


FRAME_OFFSETS = (-0.12, -0.06, -0.03, -0.015, 0.0, 0.02, 0.06, 0.15)  # s rel. plate landing


class Trial:
    def __init__(self, session, link, handle, args):
        self.s, self.link, self.h, self.args = session, link, handle, args
        self.renderer = None

    def rt(self, t=None) -> float:
        return float(self.s.metrics.run_time_at(self.s.sim.time if t is None else t))

    def run(self, level: int, side: str, phase: int, vision: bool, frames: bool) -> dict:
        s, link, h = self.s, self.link, self.h
        sim, sw, vis = s.sim, h.swatter, h.vision
        vis.cfg.enabled = vision
        s.reset("demo")
        chunk = self.args.chunk
        states: dict[int, tuple] = {}  # seq -> (window end rt, GF Hz, arrival rt)
        acts: list = []
        s.actions.listeners.append(acts.append)
        trig = link.triggers
        n_fired0, n_modes0 = len(trig.fired), len(trig.jump_modes)
        cross: list = []  # GF trailing-window threshold crossings (FastEvents)
        nfe0 = link.n_fast
        if not link.fast:
            link.brain.poll_fast()
        shots: list[tuple[float, np.ndarray]] = []

        def advance(sec, until_idle=False):
            n = int(round(sec / sim.timestep / chunk))
            k = 0
            while k < n or (until_idle and sw.phase != "idle"):
                sim.step(chunk)
                s.after_physics()
                k += 1
                if frames and sw.phase in ("slam", "press", "lift", "back"):
                    shots.append((sim.time, self._render()))
                if not link.fast:
                    cross.extend(e for e in link.brain.poll_fast() if e.kind == "trigger")
                for st in list(link.recent):
                    if st.seq not in states:
                        states[st.seq] = (st.sim_time, st.descending.get("escape", 0.0), self.rt())

        advance(self.args.walk + 0.0105 * phase)
        t_req = self.rt()
        n_ev0, n_sent0 = len(sw.events), len(vis.sent)
        sw.swat(level, side=side, source="demo")
        advance(0.0, until_idle=True)
        advance(self.args.post)
        s.actions.listeners.remove(acts.append)
        if link.fast and link.n_fast > nfe0:
            cross = list(link.fast_events)[-min(link.n_fast - nfe0, len(link.fast_events)):]
        cross = [e for e in cross if e.group == "escape" and e.sim_time is not None
                 and e.sim_time > t_req]
        ev = sw.events[n_ev0]
        t_land = ev.t_fly_contact if ev.hit else (ev.t_ground if ev.t_ground is not None
                                                  else ev.sim_time)
        rt_land = self.rt(t_land)
        sent = [e for e in vis.sent[n_sent0:]] if vision else []
        win = sorted((v for v in states.values() if v[0] is not None and v[0] > t_req),
                     key=lambda v: v[0])
        thr = trig.p.jump_escape_hz
        gf_first = next((v for v in win if v[1] > 0), None)
        gf_thr = next((v for v in win if v[1] > thr), None)
        gf_peak = max((v[1] for v in win), default=0.0)
        fired = [(t, n, r) for (t, n, r) in trig.fired[n_fired0:] if n == "jump"]
        modes = trig.jump_modes[n_modes0:]
        jstart = [a for a in acts if a.name == "jump" and a.kind == "start"]
        jend = [a for a in acts if a.name == "jump" and a.kind in ("end", "cancel")]
        info = jend[0].info if jend else {}
        takeoff = None
        if jstart and "takeoff_after_s" in info:
            takeoff = self.rt(jstart[0].time + info["takeoff_after_s"])
        rel = lambda t: None if t is None else round(t - rt_land, 4)  # noqa: E731
        d_aim = float(np.hypot(ev.fly_at_impact[0] - ev.aim[0], ev.fly_at_impact[1] - ev.aim[1]))
        row = {
            "level": level, "side": side, "phase": phase, "vision": vision,
            "outcome": ev.outcome, "hit": ev.hit, "impulse_uNs": round(ev.impulse_uNs, 2),
            "peak_bw": round(ev.magnitude_bw, 0), "body": ev.body.split("/")[-1],
            "slam_s": sw.cfg.levels[level - 1].slam_s,
            "request_rel_land": rel(t_req), "slam_rel_land": rel(self.rt(ev.t_slam)),
            "loom_rel_land": rel(sent[0].sim_time) if sent else None,
            "gf_first_window_end_rel_land": rel(gf_first[0]) if gf_first else None,
            "gf_thr_window_end_rel_land": rel(gf_thr[0]) if gf_thr else None,
            "gf_thr_arrival_rel_land": rel(gf_thr[2]) if gf_thr else None,
            "gf_cross_rel_land": rel(cross[0].sim_time) if cross else None,
            "gf_peak_hz": round(gf_peak, 0),
            "jump_trigger_rel_land": rel(fired[0][0]) if fired else None,
            "jump_mode": modes[0] if modes else None,
            "jump_gf_hz": round(fired[0][2], 0) if fired else None,
            "takeoff_rel_land": rel(takeoff),
            "takeoff_latency_s": (round(takeoff - fired[0][0], 4)
                                  if takeoff is not None and fired else None),
            "contact": "fly" if ev.hit else ("ground" if ev.t_ground is not None else "none"),
            "fly_from_aim_mm": round(d_aim, 2),
            "impact_speed_mm_s": round(ev.impact_speed_mm_s, 0),
            "jump_distance_mm": round(info["distance_mm"], 1) if "distance_mm" in info else None,
            "landed_upright": info.get("landed_upright"),
            "end_state": s.detector.state.value,
            "max_lag_s": round(max((v[2] - v[0] for v in win), default=0.0), 3),
            "shots": [],
        }
        if frames and shots:
            row["_shots"] = [(round(t - t_land, 4), img) for t, img in shots]
        return row

    def _render(self):
        from fly_simulator.rendering import FrameRenderer

        sim = self.s.sim
        if self.renderer is None:
            cfg = sim.cfg
            self.renderer = FrameRenderer(sim.model, replace(cfg.render, width=480, height=320),
                                          replace(cfg.camera, distance=18.0, elevation=-18.0,
                                                  follow_azimuth_offset=-70.0))
        return self.renderer.render(sim.data, sim.time, sim.thorax_position(), sim.heading(),
                                    ground_z=0.0, tilt_deg=sim.tilt_deg())


def save_shots(row: dict, out: Path, tag: str) -> list[str]:
    import imageio.v2 as iio

    out.mkdir(parents=True, exist_ok=True)
    shots = row.pop("_shots", [])
    paths = []
    for off in FRAME_OFFSETS:
        if not shots:
            break
        t, img = min(shots, key=lambda x: abs(x[0] - off))
        p = out / f"{tag}_L{row['level']}_{int(round(t * 1000)):+04d}ms.png"
        if str(p) not in paths:
            iio.imwrite(p, img)
            paths.append(str(p))
    return paths


def fmt(v):
    return "-" if v is None else (f"{v:+.3f}" if isinstance(v, float) else str(v))


def summarize(rows: list[dict]) -> str:
    hdr = ("| L | slam s | phase | vision | outcome | impulse uN*s | loom | GF>0 window end | "
           "GF>thr window end / arrival | GF peak | jump trigger | mode | take-off | "
           "fly-aim mm | jump mm | upright |")
    out = ["times in s relative to the plate landing (fly or ground contact)", "",
           hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for r in rows:
        out.append(
            f"| {r['level']} | {r['slam_s']} | {r['phase']} | {'on' if r['vision'] else 'off'} | "
            f"{r['outcome']} | {r['impulse_uNs']:.1f} | {fmt(r['loom_rel_land'])} | "
            f"{fmt(r['gf_first_window_end_rel_land'])} | {fmt(r['gf_thr_window_end_rel_land'])} / "
            f"{fmt(r['gf_thr_arrival_rel_land'])} | {r['gf_peak_hz']:.0f} | "
            f"{fmt(r['jump_trigger_rel_land'])} | {r['jump_mode'] or '-'} | "
            f"{fmt(r['takeoff_rel_land'])} | {r['fly_from_aim_mm']} | {r['jump_distance_mm']} | "
            f"{r['landed_upright']} |")
    out.append("")
    out.append("| L | vision | swats | hit | grazed | dodged | miss | median loom lead | "
               "median trigger | median take-off latency | median impulse |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for lv in sorted({r["level"] for r in rows}):
        for vision in (True, False):
            rs = [r for r in rows if r["level"] == lv and r["vision"] == vision]
            if not rs:
                continue
            med = lambda k: (fmt(float(np.median([r[k] for r in rs if r[k] is not None])))  # noqa: E731
                             if any(r[k] is not None for r in rs) else "-")
            out.append(f"| {lv} | {'on' if vision else 'off'} | {len(rs)} | "
                       f"{sum(r['outcome'] == 'hit' for r in rs)} | "
                       f"{sum(r['outcome'] == 'grazed' for r in rs)} | "
                       f"{sum(r['outcome'] == 'dodged' for r in rs)} | "
                       f"{sum(r['outcome'] == 'miss' for r in rs)} | {med('loom_rel_land')} | "
                       f"{med('jump_trigger_rel_land')} | {med('takeoff_latency_s')} | "
                       f"{float(np.median([r['impulse_uNs'] for r in rs])):.1f} |")
    return "\n".join(out)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = AppConfig()
    cfg.terrain.difficulty = "flat"
    cfg.whip.enabled = False
    cfg.auto_perturb.enabled = False
    extra = {} if args.clock is None else {"clock_every_s": args.clock}
    extra.update(sync_wait_s=args.sync, sync_loom_only=not args.sync_always,
                 fast_path=not args.no_fast)
    cfg.brain = BrainLinkConfig(enabled=True, window=False, steer=True, actions=True,
                                window_s=args.window, **extra)
    quiet = lambda msg: None  # noqa: E731
    link = BrainLink(cfg.brain, headless=True, say=quiet, start=False)
    if args.no_fast:  # observe the GF crossings (fast detector) without acting on them
        link.brain_cfg.fast_triggers = {"escape": cfg.brain.jump_escape_hz}
    link.start()
    try:
        sw = Swatter(SwatterConfig.from_dict(json.loads(args.swatter)))
        session = Session(cfg, log=False, brain=link, say=quiet, world_extensions=[sw.extension])
        t0 = time.time()
        link.wait_ready()
        print(f"brain ready in {time.time() - t0:.1f}s (window {args.window} s)", flush=True)
        h = install_swatter(session, swatter=sw, short_hz=(1e9 if args.no_short else args.short_hz),
                            flight=args.flight, say=quiet)
        trial = Trial(session, link, h, args)
        rows = []
        want = {k: args.frames is not None for k in ("dodged", "hit", "grazed")}
        for lv in [int(x) for x in args.levels.split(",")]:
            for side in args.sides.split(","):
                for k in range(args.phases):
                    for vision in ((True, False) if args.compare else (True,)):
                        t1 = time.time()
                        r = trial.run(lv, side, k, vision, frames=bool(vision and any(want.values())))
                        if "_shots" in r:
                            if want.get(r["outcome"]):
                                want[r["outcome"]] = False
                                r["shots"] = save_shots(r, args.frames, r["outcome"])
                            r.pop("_shots", None)
                        rows.append(r)
                        print(f"L{lv} {side} ph{k} vision={'on ' if vision else 'off'} "
                              f"{r['outcome']:7s}imp={r['impulse_uNs']:.1f} "
                              f"loom={fmt(r['loom_rel_land'])} "
                              f"GF>thr={fmt(r['gf_thr_window_end_rel_land'])}/"
                              f"{fmt(r['gf_thr_arrival_rel_land'])} cross={fmt(r['gf_cross_rel_land'])} "
                              f"GFpk={r['gf_peak_hz']:.0f} "
                              f"jump={fmt(r['jump_trigger_rel_land'])} {r['jump_mode'] or ''} {r['jump_gf_hz']} "
                              f"takeoff={fmt(r['takeoff_rel_land'])} aim={r['fly_from_aim_mm']} "
                              f"lag={r['max_lag_s']} ({time.time() - t1:.1f}s)", flush=True)
        print()
        print(summarize(rows))
        if args.json:
            args.json.write_text(json.dumps(rows, indent=1))
    finally:
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
