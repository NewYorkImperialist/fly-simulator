"""Headless checks of the octopamine stress / arousal layer (docs/STRESS.md).

Part A (``--part brain``): the connectome brain in this process (no fly).
  1. Do the real OA neurons (FlyWire OA-* cell types) respond to our stimuli?
  2. Level time course: repeated whip hits, repeated looms, decay.
  3. Brain-side effect: giant fiber / MDN / walk-DN rates for a weak loom, calm vs
     stressed (threshold shift on the OA neurons' synaptic partners).
  2b. the nociceptive relay (VNC stand-in) per hit strength, repeated hits, decay.
Part B (``--part body``): fly + real brain worker (``--brain-steer --brain-actions``
  wiring, stress installed with ``install_stress``), stress off vs on:
  ``--scenario whip`` (default): whip cracks -> level, speed-up, decay over 40 s,
  brain-window frame with the PAIN/AROUSAL gauge (PNG, ``--out``);
  ``--scenario jump``: calm speed / weak looms, 3 strong looms, speed / weak looms.

    .venv/bin/python scripts/demo_stress.py --part brain [--skip-effect]
    .venv/bin/python scripts/demo_stress.py --part body --whip-level 2 --hits 8 --out DIR
    .venv/bin/python scripts/demo_stress.py --part body --scenario jump --jump-rate 42
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_simulator.brain.schema import StimulusEvent as E  # noqa: E402


# ============================================================================ part A
class BrainBench:
    """_Model driven in-process, chunked like the worker (20 ms chunks)."""

    def __init__(self, neuromod: dict | None, seed: int = 0):
        from fly_simulator.brain.process import _Model

        self.m = _Model({"seed": seed, "neuromod": neuromod})
        self.eng = self.m.engine
        self.nm = self.m.neuromod

    def reset(self, level: float | None = None) -> None:
        self.eng.reset_state()
        self.m.mapper.add(E("reset"), self.eng.t)
        if level is not None and self.nm is not None:
            self.nm.level = level
            self.nm._apply()

    def run(self, dur: float, evs=(), win: float = 0.1):
        """Returns per-window (t, counts) list; windows of ``win`` s."""
        m, eng = self.m, self.eng
        for ev in evs:
            m.mapper.add(ev, eng.t)
            if self.nm is not None:
                for rev in self.nm.relay_events(ev):
                    m.mapper.add(rev, eng.t)
        out = []
        t_end = eng.t + dur
        while eng.t < t_end - 1e-9:
            c0 = eng.counts.copy()
            w_end = min(eng.t + win, t_end)
            while eng.t < w_end - 1e-9:
                if m.mapper.expire(eng.t):
                    idx, r = m.mapper.drive()
                    eng.set_poisson(idx, r)
                h = min(0.02, w_end - eng.t, max(1e-4, m.mapper.next_change() - eng.t))
                n = max(1, int(round(h / 1e-4)))
                _, i = eng.run(n)
                if self.nm is not None:
                    self.nm.observe(i, n * 1e-4)
            out.append((eng.t, (eng.counts - c0) / win,
                        self.nm.level if self.nm is not None else 0.0))
        return out

    def grp(self, hz, g):
        return float(hz[self.m.dn[g]].mean())


def part_brain(args) -> None:
    from fly_simulator.brain.neuromod import select_oa_neurons

    b = BrainBench({"enabled": True, "tau_decay_s": args.tau_decay})
    t = b.m.table
    oa = select_oa_neurons(t)
    ct = t.col("cell_type")
    dh = np.nonzero(ct == "DH44")[0]
    print(f"OA-* neurons: {len(oa)}  ({', '.join(sorted(set(ct[oa])))})")
    print(f"OA targets (>= {b.nm.cfg.target_min_syn} synapses): {len(b.nm.targets)}; "
          f"GF in targets: {bool(np.isin(b.m.dn['escape'], b.nm.targets).any())}, "
          f"MDN: {bool(np.isin(b.m.dn['backward_L'], b.nm.targets).any())}, "
          f"walk DNs: {bool(np.isin(b.m.dn['walk_L'], b.nm.targets).any())}")

    # 1) OA response, stress layer effect off (level frozen at 0)
    print("\n1) OA-neuron response (level held at 0; window = stimulus + 0.4 s)")
    b.nm.configure({"enabled": False})
    thorax = {"body": "Thorax"}
    tests = [
        ("whip L1 left thorax (0.15)", [E("whip_hit", "left", 0.15, 0.05, details=thorax)], 0.5),
        ("whip L4 left thorax (1.0)", [E("whip_hit", "left", 1.0, 0.05, details=thorax)], 0.5),
        ("whip L4 head", [E("whip_hit", "left", 1.0, 0.05, details={"body": "Head"})], 0.5),
        ("shove L4 left", [E("shove", "left", 1.0, 0.05, details=thorax)], 0.5),
        ("fall", [E("fall", "none", 1.0, 0.1)], 0.5),
        ("loom LC4 200 Hz 1 s (key O)", [E("manual", "none", 1, 1.0, details={"set": "LC4"})], 1.4),
        ("loom LPLC2 200 Hz 1 s", [E("manual", "none", 1, 1.0, details={"set": "LPLC2"})], 1.4),
        ("head bristles 200 Hz 1 s", [E("manual", "none", 1, 1.0,
                                        details={"set": "head_bristle"})], 1.4),
        ("sugar 200 Hz 1 s", [E("manual", "none", 1, 1.0, details={"set": "sugar"})], 1.4),
    ]
    for name, evs, dur in tests:
        b.reset()
        w = b.run(dur, evs)
        hz = np.mean([x[1] for x in w], axis=0)
        pk = max(float(x[1][oa].mean()) for x in w)
        act = oa[hz[oa] > 0]
        top = ", ".join(f"{ct[i]} {hz[i]:.0f}" for i in act[np.argsort(-hz[act])][:3])
        print(f"   {name:30s} OA-* mean {hz[oa].mean():5.2f} Hz (peak 0.1 s {pk:5.1f})  "
              f"active {len(act):2d}/{len(oa)}  DH44 {hz[dh].mean():.1f}  {top}")

    # 2) level time course (connectome-only first)
    print(f"\n2) level time course (tau_rise 1 s, tau_decay {args.tau_decay:g} s, "
          f"ref {b.nm.cfg.ref_rate_hz:g} Hz; connectome only = nociceptive paths OFF)")
    b.nm.configure({"enabled": True, "tau_decay_s": args.tau_decay})
    b.reset(level=0.0)
    for k in range(10):
        b.run(1.0, [E("whip_hit", ("left", "right")[k % 2], 1.0, 0.05, details=thorax)])
    print(f"   10 L4 whip hits, 1 s apart          -> level {b.nm.level:.3f}")
    b.reset(level=0.0)
    for k in range(3):
        b.run(1.0, [E("fall", "none", 1.0, 0.1)])
    print(f"   3 falls, 1 s apart                  -> level {b.nm.level:.3f}")
    b.reset(level=0.0)
    trace = []
    for k in range(3):
        for tt, _, lv in b.run(3.0, [E("manual", "none", 1, 1.0, details={"set": "LC4"})],
                               win=0.5):
            trace.append((tt, lv))
        print(f"   loom #{k + 1} (LC4 200 Hz 1 s) + 2 s  -> level {b.nm.level:.3f}")
    t_q = b.eng.t
    l0 = b.nm.level
    for dt in (10, 20, 30, 60):
        b.run(dt - (b.eng.t - t_q), win=1.0)
        print(f"   quiet {dt:3d} s after                  -> level {b.nm.level:.3f} "
              f"({b.nm.level / max(l0, 1e-9):.2f} of peak)")
    # "pain" paths: per hit strength, then repeated hits and decay
    def hit_series(title, extra):
        print(f"\n   {title}")
        b.nm.configure({"enabled": True, "tau_decay_s": args.tau_decay, **extra})
        for name, inten, body in (("L1", 0.15, "Thorax"), ("L2", 0.25, "Thorax"),
                                  ("L3", 0.6, "Thorax"), ("L4", 1.0, "Thorax"),
                                  ("L4 head", 1.0, "Head")):
            b.reset(level=0.0)
            b.nm.readout()  # clear the window accumulators
            w = b.run(0.6, [E("whip_hit", "left", inten, 0.05, details={"body": body})])
            oa_pk = max(float(x[1][oa].mean()) for x in w)
            walk = max(0.5 * (b.grp(x[1], "walk_L") + b.grp(x[1], "walk_R")) for x in w)
            gf = max(b.grp(x[1], "escape") for x in w)
            mdn = max(0.5 * (b.grp(x[1], "backward_L") + b.grp(x[1], "backward_R")) for x in w)
            tail = float(w[-1][1].sum())
            print(f"   one whip hit {name:8s} ({inten:.2f}): OA-* peak {oa_pk:5.1f} Hz, "
                  f"walk DN peak {walk:5.1f} Hz, GF {gf:3.0f}, MDN {mdn:3.0f}, "
                  f"spikes/s at +0.5 s {tail:7.0f} -> level +{b.nm.level:.3f}")
        b.reset(level=0.0)
        for k in range(8):
            b.run(1.5, [E("whip_hit", ("left", "right")[k % 2], 0.25, 0.05, details=thorax)])
            if k in (0, 1, 3, 7):
                print(f"   {k + 1} L2 whip hits, 1.5 s apart         -> level {b.nm.level:.3f}")
        l0, t_q = b.nm.level, b.eng.t
        for dt in (10, 30, 60):
            b.run(dt - (b.eng.t - t_q), win=1.0)
            print(f"   quiet {dt:3d} s after                  -> level {b.nm.level:.3f} "
                  f"({b.nm.level / max(l0, 1e-9):.2f} of peak)")

    hit_series("nociceptive RELAY (VNC stand-in: an_walk + an_arousal ascending neurons; "
               "level from real OA spikes)", {"noci_relay": True})
    if args.alt_link:
        hit_series("ALTERNATIVE modelled link (hit-afferent rate -> level directly)",
                   {"nociceptive_input": True})

    if args.skip_effect:
        return
    # 3) brain-side effect of the level
    print("\n3) weak loom (LC4 at R Hz for 0.5 s): GF / MDN / walk / turn DN rates, "
          "max over 0.1 s windows")
    for rate in args.weak_rates:
        for lv in (0.0, 0.5, 1.0):
            b.nm.configure({"enabled": True, "tau_rise_s": 1e9, "tau_decay_s": 1e9})
            b.reset(level=lv)
            w = b.run(0.5, [E("manual", "none", 1, 0.5, details={"set": "LC4", "rate_hz": rate})])
            gf = [b.grp(x[1], "escape") for x in w]
            mdn = [0.5 * (b.grp(x[1], "backward_L") + b.grp(x[1], "backward_R")) for x in w]
            wk = [0.5 * (b.grp(x[1], "walk_L") + b.grp(x[1], "walk_R")) for x in w]
            tn = [0.5 * (b.grp(x[1], "turn_L") + b.grp(x[1], "turn_R")) for x in w]
            tot = np.mean([x[1].sum() for x in w])
            print(f"   LC4 {rate:3.0f} Hz level {lv:.1f}: GF max {max(gf):5.1f} mean "
                  f"{np.mean(gf):5.1f} | MDN max {max(mdn):4.1f} | walk {max(wk):4.1f} | "
                  f"turn {max(tn):4.1f} | total {tot / 1e3:5.1f} k spikes/s")

    # 4) jump probability = P(some 0.1 s state has GF > threshold(level)), brain effect
    #    (threshold shift) + body effect (lower trigger threshold) combined
    from fly_simulator.stress import StressConfig, jump_threshold

    sc = StressConfig(enabled=True)
    print(f"\n4) weak-loom jump probability over {args.trials} trials (LC4 R Hz, 0.5 s; "
          f"jump if a 0.1 s state has GF > 60 Hz x (1 - 0.5 level))")
    for rate in args.weak_rates:
        row = []
        for lv in (0.0, 0.3, 0.6):
            thr = jump_threshold(lv, 60.0, sc)
            b.nm.configure({"enabled": True, "tau_rise_s": 1e9, "tau_decay_s": 1e9})
            n_jump, gfs = 0, []
            for _ in range(args.trials):
                b.reset(level=lv)
                w = b.run(0.5, [E("manual", "none", 1, 0.5,
                                  details={"set": "LC4", "rate_hz": rate})])
                g = max(b.grp(x[1], "escape") for x in w)
                gfs.append(g)
                n_jump += g > thr
            row.append(f"level {lv:.1f} (thr {thr:2.0f} Hz): {n_jump}/{args.trials} "
                       f"(GF max {np.mean(gfs):4.1f} Hz)")
        print(f"   LC4 {rate:3.0f} Hz  " + "  |  ".join(row))


# ============================================================================ part B
def part_body(args) -> None:
    from fly_simulator import AppConfig
    from fly_simulator.app import Session
    from fly_simulator.brain_link import BrainLink, BrainLinkConfig
    from fly_simulator.stress import StressConfig, install_stress

    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    def make(stress_on: bool):
        cfg = AppConfig()
        cfg.logging.enabled = False
        cfg.brain = BrainLinkConfig(enabled=True, window=False, steer=True, actions=True)
        link = BrainLink(cfg.brain, headless=True, say=lambda m: None)
        link.wait_ready(120)
        s = Session(cfg, log=False, say=lambda m: None, brain=link)
        st = install_stress(s, StressConfig(enabled=stress_on,
                                            neuromod={"tau_decay_s": args.tau_decay}))
        return s, link, st

    def advance(s, link, sec, chunk=100):
        """Physics + brain in step: wait for the brain to catch up with the fly."""
        n = int(round(sec / s.sim.timestep / chunk))
        for _ in range(n):
            s.sim.step(chunk)
            s.after_physics()
            deadline = time.time() + 5
            while (link.latest is None or link.latest.sim_time is None
                   or link.latest.sim_time < s.run_time() - 0.15) and time.time() < deadline:
                time.sleep(0.002)
                s.after_physics()

    def speed(s, link, sec):
        p0 = s.sim.thorax_position().copy()
        t0 = s.sim.time
        advance(s, link, sec)
        return float(np.linalg.norm((s.sim.thorax_position() - p0)[:2]) / (s.sim.time - t0))

    def weak_loom(s, link, rate):
        link.send(E("manual", "none", 1.0, 0.5, s.run_time(),
                    details={"set": "LC4", "rate_hz": float(rate),
                             "label": f"LOOM weak {rate:g}Hz"}), source="demo")

    def upright(s, link):
        if s.down_for() is not None:
            s.reset("demo")  # the brain's neurons reset too; the octopamine level persists
            advance(s, link, 1.0)

    def weak_trials(s, link, st, n):
        """n weak looms, 2 s apart (> jump refractory); returns (jumps, GF max list,
        levels, thresholds)."""
        jumps, gfs, lvls, thrs = 0, [], [], []
        for _ in range(n):
            upright(s, link)
            n0 = link.triggers.counts["jump"]
            lvls.append(st.level)
            thrs.append(link.triggers.p.jump_escape_hz)
            weak_loom(s, link, args.jump_rate)
            g = 0.0
            t_end = s.run_time() + 0.8
            while s.run_time() < t_end:
                advance(s, link, 0.1)
                g = max(g, link.latest.descending.get("escape", 0.0))
            gfs.append(g)
            jumps += link.triggers.counts["jump"] - n0 > 0
            advance(s, link, 1.2)
        return jumps, gfs, lvls, thrs

    if args.scenario == "whip":
        return whip_scenario(args, make, advance, speed, upright, out)

    for stress_on in (False, True):
        tag = "STRESS ON " if stress_on else "stress off"
        s, link, st = make(stress_on)
        try:
            advance(s, link, 1.0)
            v_calm = speed(s, link, 1.5)
            j, g, lv, th = weak_trials(s, link, st, args.trials)
            print(f"[{tag}] calm: speed {v_calm:5.2f} mm/s, level {np.mean(lv):.2f}; "
                  f"{args.trials} weak looms (LC4 {args.jump_rate:g} Hz): jumps {j}/{args.trials}, "
                  f"GF max {np.mean(g):.0f} Hz (range {min(g):.0f}-{max(g):.0f}), "
                  f"threshold {np.mean(th):.0f} Hz")
            # stress it with strong looms (key O), through the brain only
            for k in range(args.looms):
                upright(s, link)
                link.handle_key("o")
                advance(s, link, 2.0)
            upright(s, link)
            advance(s, link, 1.0)
            print(f"[{tag}] after {args.looms} looms (O): level {st.level:.2f}, "
                  f"freq x{st.freq_mult:.2f}, jump threshold {link.triggers.p.jump_escape_hz:.0f} Hz")
            v_st = speed(s, link, 1.5)
            if stress_on and out is not None:
                import cv2

                img = link.render_snapshot()
                if img is not None:
                    p = out / "stress_brain_window.png"
                    cv2.imwrite(str(p), img)
                    print(f"   brain window frame -> {p}")
            j, g, lv, th = weak_trials(s, link, st, args.trials)
            print(f"[{tag}] stressed: speed {v_st:5.2f} mm/s (calm {v_calm:.2f}); "
                  f"weak looms: jumps {j}/{args.trials}, GF max {np.mean(g):.0f} Hz "
                  f"(range {min(g):.0f}-{max(g):.0f}), level {min(lv):.2f}-{max(lv):.2f}, "
                  f"threshold {min(th):.0f}-{max(th):.0f} Hz")
            if stress_on:
                l1 = st.level
                upright(s, link)
                advance(s, link, args.decay_check)
                print(f"[{tag}] {args.decay_check:g} s later: level {st.level:.2f} "
                      f"(was {l1:.2f}), freq x{st.freq_mult:.2f}, speed "
                      f"{speed(s, link, 1.0):.2f} mm/s")
        finally:
            st.close()
            s.close("demo")
            link.close()


def whip_scenario(args, make, advance, speed, upright, out) -> None:
    """Whip hits -> "pain" -> the fly runs faster for a while, then calms down."""
    for stress_on in (False, True):
        tag = "STRESS ON " if stress_on else "stress off"
        s, link, st = make(stress_on)
        falls = []
        s.detector.add_listener(lambda e: falls.append(e.kind) if e.kind == "fall" else None)
        try:
            advance(s, link, 1.0)
            v0 = speed(s, link, 2.0)
            print(f"[{tag}] calm: {v0:5.2f} mm/s, level {st.level:.2f}")
            for k in range(args.hits):
                upright(s, link)
                s.whip.crack(("left", "right")[k % 2], level=args.whip_level, source="demo")
                advance(s, link, args.hit_every)
                if stress_on:
                    rec = [x.neuromod for x in list(link.recent)[-15:] if x.neuromod]
                    pk = max((r.get("noci_hz", 0.0) for r in rec), default=0.0)
                    oa = max((r.get("oa_rate_hz", 0.0) for r in rec), default=0.0)
                    print(f"   hit {k + 1} (L{args.whip_level}): level {st.level:.2f}  "
                          f"hit afferents peak {pk:.0f} Hz  OA peak {oa:.1f} Hz  "
                          f"step x{st.freq_mult:.2f} stride x{st.amp_mult:.2f}")
            upright(s, link)
            if stress_on and out is not None:
                import cv2

                img = link.render_snapshot()
                if img is not None:
                    p = out / "stress_whip_brain_window.png"
                    cv2.imwrite(str(p), img)
                    print(f"   brain window frame -> {p}")
                print(f"   HUD: {st.hud_line()}")
            t_last = s.run_time()
            for t_win in args.windows:
                wait = t_win - (s.run_time() - t_last)
                if wait > 0:
                    advance(s, link, wait)
                upright(s, link)
                v = speed(s, link, 2.0)
                print(f"[{tag}] {t_win:4.0f}-{t_win + 2:.0f} s after the last hit: {v:5.2f} mm/s "
                      f"(calm {v0:.2f}), level {st.level:.2f}, step x{st.freq_mult:.2f} "
                      f"stride x{st.amp_mult:.2f}")
                if not stress_on:
                    break
            print(f"[{tag}] falls during the run: {len(falls)}")
        finally:
            st.close()
            s.close("demo")
            link.close()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--part", choices=("brain", "body", "all"), default="brain")
    p.add_argument("--tau-decay", type=float, default=30.0)
    p.add_argument("--weak-rates", type=float, nargs="+", default=[50.0, 60.0])
    p.add_argument("--jump-rate", type=float, default=60.0,
                   help="LC4 rate (Hz) of the weak loom in part B")
    p.add_argument("--looms", type=int, default=3)
    p.add_argument("--trials", type=int, default=8, help="weak looms per condition")
    p.add_argument("--skip-effect", action="store_true", help="part A: skip sections 3-4")
    p.add_argument("--alt-link", action="store_true",
                   help="part A: also show the modelled afferent -> level link")
    p.add_argument("--scenario", choices=("whip", "jump"), default="whip",
                   help="part B: whip hits -> speed-up (default) or weak looms -> jump")
    p.add_argument("--hits", type=int, default=6)
    p.add_argument("--whip-level", type=int, default=3)
    p.add_argument("--hit-every", type=float, default=1.5)
    p.add_argument("--windows", type=float, nargs="+", default=[0.0, 8.0, 20.0, 40.0],
                   help="part B whip: speed windows (s after the last hit, 2 s each)")
    p.add_argument("--decay-check", type=float, default=10.0)
    p.add_argument("--out", default=None, help="directory for the brain-window PNG")
    args = p.parse_args(argv)
    if args.part in ("brain", "all"):
        part_brain(args)
    if args.part in ("body", "all"):
        part_body(args)


if __name__ == "__main__":
    main()
