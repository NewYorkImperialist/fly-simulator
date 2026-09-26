"""Headless check of whip looming vision with the REAL connectome brain (docs/VISION.md).

For each crack (level x side) on flat ground: fresh reset, walk ``--walk`` s, crack,
observe ``--post`` s with ``--brain-actions`` logic (giant fibre > 60 Hz -> Jump) and
brain steering (MDN). Reports per crack: theta max / dtheta/dt max per eye, peak
LC4 / LPLC2 drive sent, GF (DNp01) peak rate, the jump (brain trigger time and
take-off, relative to the whip's first contact), jump outcome, and, with
``--compare``, the measured whip impulse with vs without vision (was the hit
dodged?). ``--slow`` adds control sweeps with a 10 rad/s whip that should not
trigger anything.

    .venv/bin/python scripts/demo_whip_vision.py --levels 1,2,3,4 --sides left,right,front,rear,overhead --compare --slow
    .venv/bin/python scripts/demo_whip_vision.py --levels 3 --sides left --frames /tmp/frames

One brain worker process (the fly and the brain are paced together).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from perpetualfly import AppConfig
from perpetualfly.app import Session
from perpetualfly.brain_link import BrainLink, BrainLinkConfig
from perpetualfly.interaction.whip import WhipLevel
from perpetualfly.vision import LoomingConfig, install_whip_vision

SLOW_OMEGA = 10.0  # rad/s: the control sweep (level 1 omega replaced)
# frames (with --frames): 10 ms chunks after the crack request (swing starts at 170 ms)
FRAME_CHUNKS = (15, 17, 19, 21, 25, 30, 38)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", default="1,2,3,4")
    p.add_argument("--sides", default="left,right,front,rear,overhead")
    p.add_argument("--walk", type=float, default=1.0, help="s of walking before the crack")
    p.add_argument("--phases", type=int, default=1,
                   help="gait phases per crack (walk + k x 10.5 ms, as in docs/WHIP.md)")
    p.add_argument("--post", type=float, default=0.8, help="s observed after the crack")
    p.add_argument("--compare", action="store_true", help="also run each crack without vision")
    p.add_argument("--slow", action="store_true", help=f"add {SLOW_OMEGA:g} rad/s control sweeps")
    p.add_argument("--no-actions", action="store_true", help="brain triggers off")
    p.add_argument("--persist", type=float, default=None, help="LoomingConfig.persist_s")
    p.add_argument("--vision", default="{}", help="JSON overrides of LoomingConfig")
    p.add_argument("--window", type=float, default=0.1, help="BrainState window (s)")
    p.add_argument("--frames", type=Path, default=None,
                   help="save a few PNGs around contact of the first trial that jumps")
    p.add_argument("--json", type=Path, default=None)
    return p.parse_args(argv)


class Trial:
    def __init__(self, session: Session, link: BrainLink, vis, args):
        self.s, self.link, self.vis, self.args = session, link, vis, args
        self.renderer = None

    def rt(self, t=None) -> float:
        return float(self.s.metrics.run_time_at(self.s.sim.time if t is None else t))

    def run(self, side: str, level: int, vision: bool, slow: bool = False,
            frames: Path | None = None, phase: int = 0) -> dict:
        s, link, vis = self.s, self.link, self.vis
        sim, whip = s.sim, s.whip
        whip.cfg.levels[0] = WhipLevel("gentle", SLOW_OMEGA if slow else 60.0)
        vis.cfg.enabled = vision
        s.reset("demo")
        chunk = 100  # 10 ms physics chunks, then brain clock / states / triggers
        states: dict[int, tuple] = {}
        acts: list = []
        s.actions.listeners.append(acts.append)
        n_fired0 = len(link.triggers.fired) if link.triggers else 0

        shots: list[str] = []
        shot_at: dict[int, str] = {}

        def advance(sec):
            n = int(round(sec / sim.timestep / chunk))
            for k in range(n):
                sim.step(chunk)
                s.after_physics()
                if k in shot_at:
                    shots.append(self._shot(frames, f"{side}_L{level}_{shot_at[k]}"))
                for st in list(link.recent):
                    if st.seq not in states:
                        states[st.seq] = (st.sim_time, st.descending.get("escape", 0.0),
                                          0.5 * (st.descending.get("backward_L", 0.0)
                                                 + st.descending.get("backward_R", 0.0)))

        advance(self.args.walk + 0.0105 * phase)
        t_crack = self.rt()
        n_ev0, n_sent0 = len(whip.events), len(vis.sent)
        vis.history.clear()
        whip.crack(side, level, source="demo")
        if frames is not None:  # chunk index after the crack -> label (ms)
            shot_at = {k: f"{(k + 1) * 10:03d}ms" for k in FRAME_CHUNKS}
        advance(self.args.post)
        s.actions.listeners.remove(acts.append)
        # --- collect
        ev = whip.events[n_ev0] if len(whip.events) > n_ev0 else None
        t_contact = self.rt(ev.sim_time) if ev is not None and ev.hit else None
        hist = list(vis.history) if vision else []
        eyes = {}
        for eye in ("left", "right"):
            hs = [e for (_, _, ey, e) in hist if ey == eye]
            eyes[eye] = {
                "theta_max": max((e.theta for e in hs), default=0.0),
                "dtheta_max": max((e.dtheta for e in hs), default=0.0),
                "lc4_max": max((e.lc4_hz for e in hs), default=0.0),
                "lplc2_max": max((e.lplc2_hz for e in hs), default=0.0),
            }
        sent = vis.sent[n_sent0:] if vision else []
        pre = [e for e in sent if t_contact is not None and e.sim_time < t_contact]
        win = [v for v in states.values() if v[0] is not None and v[0] > t_crack]
        gf = max((v[1] for v in win), default=0.0)
        mdn = max((v[2] for v in win), default=0.0)
        fired = [(t, n, r) for (t, n, r) in (link.triggers.fired[n_fired0:] if link.triggers else [])
                 if n == "jump"]
        jump_end = [a for a in acts if a.name == "jump" and a.kind in ("end", "cancel")]
        jump_start = [a for a in acts if a.name == "jump" and a.kind == "start"]
        info = jump_end[0].info if jump_end else {}
        takeoff = None
        if jump_start and "takeoff_after_s" in info:
            takeoff = self.rt(jump_start[0].time + info["takeoff_after_s"])
        rel = (lambda t: None if t is None or t_contact is None else round(t - t_contact, 4))
        return {
            "side": side, "level": "slow" if slow else level, "phase": phase, "vision": vision,
            "hit": bool(ev.hit) if ev else False,
            "impulse_uNs": round(ev.impulse_uNs, 3) if ev else 0.0,
            "contact_after_crack_s": None if t_contact is None else round(t_contact - t_crack, 4),
            "eyes": {k: {kk: round(vv, 1) for kk, vv in v.items()} for k, v in eyes.items()},
            "n_loom_events": len(sent), "n_loom_events_pre_contact": len(pre),
            "first_loom_rel_contact_s": rel(sent[0].sim_time) if sent else None,
            "gf_peak_hz": round(gf, 1), "mdn_peak_hz": round(mdn, 1),
            "jump_trigger_rel_contact_s": rel(fired[0][0]) if fired else None,
            "jump_takeoff_rel_contact_s": rel(takeoff),
            "jump_apex_mm": round(info["apex_dz_mm"], 2) if "apex_dz_mm" in info else None,
            "jump_landed_upright": info.get("landed_upright"),
            "fell": s.detector.state.value,
            "tilt_end_deg": round(sim.tilt_deg(), 1),
            "shots": shots,
        }

    def _shot(self, out: Path, name: str) -> str:
        """One small PNG from the app's own renderer (follow camera)."""
        import imageio.v2 as iio

        from perpetualfly.rendering import FrameRenderer

        sim = self.s.sim
        if self.renderer is None:
            cfg = sim.cfg
            self.renderer = FrameRenderer(sim.model, replace(cfg.render, width=400, height=270),
                                          replace(cfg.camera, distance=8.0))
        img = self.renderer.render(sim.data, sim.time, sim.thorax_position(), sim.heading(),
                                   ground_z=0.0, tilt_deg=sim.tilt_deg())
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{name}.png"
        iio.imwrite(path, img)
        return str(path)


def summarize(rows: list[dict]) -> str:
    hdr = ("| side | L | phase | vision | hit | impulse uN*s | contact after crack (s) | theta max L/R (deg) "
           "| dtheta/dt max L/R (deg/s) | LC4 / LPLC2 max L (Hz) | LC4 / LPLC2 max R (Hz) "
           "| loom events (pre-contact) | GF peak (Hz) | MDN peak | jump trigger / take-off rel. contact (s) "
           "| apex mm | upright | end state |")
    out = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for r in rows:
        L, R = r["eyes"]["left"], r["eyes"]["right"]
        out.append(
            f"| {r['side']} | {r['level']} | {r['phase']} | {'on' if r['vision'] else 'off'} | "
            f"{'y' if r['hit'] else 'n'} | {r['impulse_uNs']:.2f} | {r['contact_after_crack_s']} | "
            f"{L['theta_max']:.0f} / {R['theta_max']:.0f} | {L['dtheta_max']:.0f} / {R['dtheta_max']:.0f} | "
            f"{L['lc4_max']:.0f} / {L['lplc2_max']:.0f} | {R['lc4_max']:.0f} / {R['lplc2_max']:.0f} | "
            f"{r['n_loom_events']} ({r['n_loom_events_pre_contact']}) | {r['gf_peak_hz']:.0f} | "
            f"{r['mdn_peak_hz']:.0f} | {r['jump_trigger_rel_contact_s']} / {r['jump_takeoff_rel_contact_s']} | "
            f"{r['jump_apex_mm']} | {r['jump_landed_upright']} | {r['fell']} |")
    return "\n".join(out)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = AppConfig()
    cfg.terrain.difficulty = "flat"
    cfg.brain = BrainLinkConfig(enabled=True, window=False, steer=True,
                                actions=not args.no_actions, window_s=args.window)
    quiet = lambda msg: None  # noqa: E731
    link = BrainLink(cfg.brain, headless=True, say=quiet)
    try:
        session = Session(cfg, log=False, brain=link, say=quiet)
        t0 = time.time()
        link.wait_ready()
        print(f"brain ready in {time.time() - t0:.1f}s", flush=True)
        vcfg = LoomingConfig.from_dict({**json.loads(args.vision)})
        if args.persist is not None:
            vcfg.persist_s = args.persist
        vis = install_whip_vision(session, session.whip, link, vcfg)
        trial = Trial(session, link, vis, args)
        plan = [(side, int(lv), False, k) for lv in args.levels.split(",")
                for side in args.sides.split(",") for k in range(args.phases)]
        if args.slow:
            plan += [(side, 1, True, k) for side in ("left", "right", "front") for k in range(args.phases)]
        rows = []
        frames_done = args.frames is None
        for side, lv, slow, k in plan:
            for vision in ((True, False) if args.compare else (True,)):
                t1 = time.time()
                fr = args.frames if (vision and not frames_done) else None
                r = trial.run(side, lv, vision, slow, frames=fr, phase=k)
                if fr is not None and r["jump_trigger_rel_contact_s"] is not None:
                    frames_done = True
                rows.append(r)
                print(f"{side:8s} L{r['level']} ph{k} vision={'on ' if vision else 'off'} hit={r['hit']} "
                      f"imp={r['impulse_uNs']:.2f} GF={r['gf_peak_hz']:.0f} MDN={r['mdn_peak_hz']:.0f} "
                      f"loomL={r['eyes']['left']['lc4_max']:.0f}/{r['eyes']['left']['lplc2_max']:.0f} "
                      f"loomR={r['eyes']['right']['lc4_max']:.0f}/{r['eyes']['right']['lplc2_max']:.0f} "
                      f"jump={r['jump_trigger_rel_contact_s']} takeoff={r['jump_takeoff_rel_contact_s']} "
                      f"({time.time() - t1:.1f}s)", flush=True)
        print()
        print(summarize(rows))
        if args.json:
            args.json.write_text(json.dumps(rows, indent=1))
    finally:
        link.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
