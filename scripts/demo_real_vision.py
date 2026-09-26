"""Real vision (compound eyes -> flyvis -> LC4 / LPLC2 bridge -> brain), headless.

Modes (docs/VISION.md, "Real vision"):

* ``--calibrate``: synthetic stimuli drawn directly on the 721-ommatidium lattice
  (no physics): dark / bright looming discs (l/v 10-80 ms), slow expansion,
  receding disc, translating discs, a drifting grating. Prints the bridge's peak
  LPLC2 / LC4 activations and rates per stimulus (used to set BridgeConfig).
* ``--sim``: the physics + the REAL connectome brain (one worker process). Trials:
  walking only, a dark sphere looming from the side (fast / slow / receding /
  lateral pass) and the swatter (levels), each with ``--sense real`` (eyes) or
  ``--sense geometric`` (the calculated looming sense). Reports object onset ->
  first loom event -> GF above threshold -> jump, and turn-DN asymmetry.
* ``--figure PATH``: what the eyes see at a looming moment + T4/T5 + LPLC2 maps.

    .venv/bin/python scripts/demo_real_vision.py --calibrate
    .venv/bin/python scripts/demo_real_vision.py --sim --trials walk,loom_fast,swat2 --sense real,geometric
    .venv/bin/python scripts/demo_real_vision.py --figure /tmp/real_vision.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from perpetualfly.vision.bridge import DEG_PER_PX, BridgeConfig, LoomBridge, ommatidia_centers

CENTER = np.array([256.0, 225.0])  # eye-image centre (row, col)


# ---------------------------------------------------------------------------
# synthetic stimuli on the ommatidia lattice
# ---------------------------------------------------------------------------


def disc_frames(cen, radius_deg_fn, center_fn, T, dt, fg=0.05, bg=0.8):
    """Frames (n, 721) of a disc of angular radius radius_deg_fn(t) at center_fn(t)."""
    out = []
    for k in range(int(round(T / dt))):
        t = k * dt
        r_px = radius_deg_fn(t) / DEG_PER_PX
        c = center_fn(t)
        inside = np.linalg.norm(cen - c, axis=1) <= r_px
        out.append(np.where(inside, fg, bg).astype(np.float32))
    return np.array(out)


def loom_radius(l_over_v, t_coll, max_deg=80.0):
    def f(t):
        tau = t_coll - t
        if tau <= 0:
            return max_deg
        return min(max_deg, math.degrees(math.atan(l_over_v / tau)))
    return f


def stimuli(cen, dt):
    fix = lambda t: CENTER  # noqa: E731
    T = 1.0
    S = {}
    for lv in (0.01, 0.04, 0.08):
        S[f"dark loom l/v={int(lv * 1e3)}ms"] = disc_frames(cen, loom_radius(lv, 0.8), fix, T, dt)
    S["bright loom l/v=40ms"] = disc_frames(cen, loom_radius(0.04, 0.8), fix, T, dt, fg=1.0, bg=0.3)
    S["slow expansion 20deg/s"] = disc_frames(cen, lambda t: 2 + 20 * t, fix, T, dt)
    S["receding 60->2deg in 0.3s"] = disc_frames(
        cen, lambda t: 60.0 if t < 0.3 else max(2.0, 60 - (t - 0.3) * 193), fix, T, dt)
    for sp in (100, 400):
        S[f"dark disc 20deg translating {sp}deg/s"] = disc_frames(
            cen, lambda t: 10.0, lambda t, sp=sp: CENTER + np.array([0, (-60 + sp * (t - 0.3)) / DEG_PER_PX]) if t > 0.3 else CENTER + np.array([0, -60 / DEG_PER_PX]), T, dt)
    u = np.array([0.0, 1.0])
    S["grating 20deg, 5Hz (self-motion like)"] = np.array([
        (0.5 + 0.35 * np.sign(np.sin(2 * np.pi * ((cen @ u) * DEG_PER_PX / 20.0 - 5.0 * max(0, k * dt - 0.3)))))
        .astype(np.float32) for k in range(int(T / dt))])
    return S


def calibrate(args) -> int:
    from perpetualfly.vision.flyvis_net import StepwiseFlyvis

    dt = 1.0 / args.rate
    net = StepwiseFlyvis(dt)
    cen = ommatidia_centers()
    print(f"flyvis ready ({net.init_s:.1f}s); dt={dt * 1e3:.1f} ms")
    rows = []
    for name, frames in stimuli(cen, dt).items():
        br = LoomBridge(BridgeConfig(**json.loads(args.bridge)), centers=cen)
        net.state = None
        net.reset(np.stack([frames[0]] * 2))
        pk = {"lplc2": 0.0, "lc4": 0.0, "lplc2_hz": 0.0, "lc4_hz": 0.0}
        t_first = None
        for k, f in enumerate(frames):
            net.step(np.stack([f, f]))
            o = br.update(*net.motion(), dt)
            if k * dt < 0.2:  # adaptation settling
                continue
            for key in pk:
                pk[key] = max(pk[key], float(o[key][0]))
            if t_first is None and max(o["lc4_hz"][0], o["lplc2_hz"][0]) > 0:
                t_first = k * dt
        rows.append((name, pk, t_first))
        print(f"{name:40s} LPLC2 {pk['lplc2']:.4f} ({pk['lplc2_hz']:5.0f} Hz)  "
              f"LC4 {pk['lc4']:.4f} ({pk['lc4_hz']:5.0f} Hz)  first event "
              f"{'-' if t_first is None else f'{t_first:.2f}s'}", flush=True)
    print(f"flyvis {net.ms_per_step():.1f} ms/step (both eyes)")
    return 0


# ---------------------------------------------------------------------------
# closed loop: physics + eyes + flyvis + bridge + the real brain
# ---------------------------------------------------------------------------

# trial name -> (kind, kwargs). Object trials use LoomingObject; swatN = swatter level N.
TRIALS = {
    "walk": ("walk", {"seconds": 2.0}),
    "loom_fast": ("approach", {"azimuth_deg": 60.0, "elevation_deg": 10.0, "l_over_v": 0.04}),
    "loom_lv10": ("approach", {"azimuth_deg": 60.0, "elevation_deg": 10.0, "l_over_v": 0.01}),
    "loom_lv80": ("approach", {"azimuth_deg": 60.0, "elevation_deg": 10.0, "l_over_v": 0.08}),
    "loom_side90": ("approach", {"azimuth_deg": 90.0, "elevation_deg": 10.0, "l_over_v": 0.04}),
    "loom_right": ("approach", {"azimuth_deg": -60.0, "elevation_deg": 10.0, "l_over_v": 0.04}),
    "loom_slow": ("approach", {"azimuth_deg": 60.0, "elevation_deg": 10.0, "dist_mm": 12.0,
                               "speed_mm_s": 4.0, "stop_mm": 5.0}),
    "recede": ("recede", {"azimuth_deg": 60.0, "elevation_deg": 10.0, "dist_mm": 3.0}),
    "pass_left": ("pass", {"azimuth_deg": 90.0, "elevation_deg": 0.0, "dist_mm": 6.0}),
    "swat1": ("swat", {"level": 1}), "swat2": ("swat", {"level": 2}),
    "swat3": ("swat", {"level": 3}), "swat4": ("swat", {"level": 4}),
    "swat1L": ("swat", {"level": 1, "side": "left"}), "swat2L": ("swat", {"level": 2, "side": "left"}),
    "swat3L": ("swat", {"level": 3, "side": "left"}), "swat4L": ("swat", {"level": 4, "side": "left"}),
}


class Rig:
    """One Session with eyes + real vision + the geometric sense on the same objects;
    ``sense`` picks which one talks to the brain for a trial."""

    def __init__(self, args):
        from perpetualfly import AppConfig
        from perpetualfly.app import Session
        from perpetualfly.brain_link import BrainLink, BrainLinkConfig
        from perpetualfly.interaction.swatter import Swatter, SwatterConfig, install_swatter
        from perpetualfly.vision.looming import LoomingVision
        from perpetualfly.vision.objects import LoomingObject

        self.args = args
        cfg = AppConfig()
        cfg.terrain.difficulty = args.terrain
        cfg.whip.enabled = False
        cfg.auto_perturb.enabled = False
        cfg.real_vision.enabled = True
        cfg.real_vision.rate_hz = args.rate
        cfg.real_vision.steer = args.steer
        cfg.real_vision.vision = {"bridge": json.loads(args.bridge)}
        quiet = lambda msg: None  # noqa: E731
        self.link = None
        if not args.no_brain:
            cfg.brain = BrainLinkConfig(enabled=True, window=False, steer=True, actions=True,
                                        window_s=0.02, sync_wait_s=0.05,
                                        fast_path=not args.no_fast)
            self.link = BrainLink(cfg.brain, headless=True, say=quiet, start=False)
            if args.no_fast:  # observe the GF crossings (fast detector) without acting
                self.link.brain_cfg.fast_triggers = {"escape": cfg.brain.jump_escape_hz}
            self.link.start()
        self.obj = LoomingObject()
        self.sw = Swatter(SwatterConfig())
        t0 = time.time()
        self.s = Session(cfg, log=False, brain=self.link, say=print,
                         world_extensions=[self.sw.extension, self.obj.extension])
        self.build_s = time.time() - t0
        if self.link is not None:
            self.link.wait_ready()
        self.obj.attach(self.s.sim)
        self.rv = self.s.real_vision
        self.h = install_swatter(self.s, swatter=self.sw, vision=False, say=quiet)
        # the geometric sense, on the same two objects (for comparisons)
        tf = self.link._run_time if self.link is not None else None
        self.geo = LoomingVision(self.s.sim, sink=self.link, time_fn=tf).attach()
        self.geo.add_source(self.obj.geometric_source())
        # the geometric sense sees the sphere from motion onset on (no pop-in transient)
        self.obj.onset_listeners.append(lambda t: self.geo._filt.pop("loom_object", None))
        from perpetualfly.interaction.swatter import swatter_source

        self.geo.add_source(swatter_source(self.sw))
        trig = self.link.triggers if self.link is not None else None
        if trig is not None:  # escape away from whichever threat is active
            trig.threat_fn = lambda: (self.sw.plate_center() if self.sw.busy else
                                      (self.obj.position() if self.obj.busy else None))

    def close(self):
        if self.link is not None:
            self.link.close()

    def rt(self, t=None):
        s = self.s
        return float(s.metrics.run_time_at(s.sim.time if t is None else t))

    def run(self, name: str, sense: str) -> dict:
        s, sim, link, rv, geo = self.s, self.s.sim, self.link, self.rv, self.geo
        kind, kw = TRIALS[name]
        real = sense == "real"
        rv.enabled = real
        rv.eyes.enabled = real
        geo.cfg.enabled = not real
        s.reset("demo")
        chunk = self.args.chunk
        states = {}
        trig = link.triggers if link is not None else None
        nf0 = len(trig.fired) if trig else 0
        acts = []
        s.actions.listeners.append(acts.append)
        wall0 = time.time()
        cross = []  # GF trailing-window threshold crossings (FastEvents)
        nfe0 = link.n_fast if link is not None else 0
        if link is not None and not link.fast:
            link.brain.poll_fast()

        def advance(sec, until=None):
            n = int(round(sec / sim.timestep / chunk))
            k = 0
            while k < n or (until is not None and until()):
                sim.step(chunk)
                s.after_physics()
                k += 1
                if link is not None:
                    if not link.fast:
                        cross.extend(e for e in link.brain.poll_fast() if e.kind == "trigger")
                    for st in list(link.recent):
                        if st.seq not in states:
                            states[st.seq] = (st.sim_time, dict(st.descending))

        advance(self.args.walk)
        t_req = self.rt()
        ns_rv, ns_geo = len(rv.sent), len(geo.sent)
        nh = len(rv.history)
        onset = end = None
        if kind == "walk":
            advance(kw["seconds"])
            onset = t_req
        elif kind == "swat":
            ne = len(self.sw.events)
            self.sw.swat(kw["level"], side=kw.get("side", "rear"), source="demo")
            advance(0.0, until=lambda: self.sw.phase != "idle")
            advance(0.3)
            ev = self.sw.events[ne]
            onset = self.rt(ev.t_slam)
            t_land = ev.t_fly_contact if ev.hit else (ev.t_ground if ev.t_ground is not None else ev.sim_time)
            end = self.rt(t_land)
            outcome = ev.outcome
        else:
            kw2 = dict(kw)
            stop = kw2.pop("stop_mm", None)
            if stop is not None:
                self.obj.cfg.stop_mm = stop
            self.obj.start(kind, **kw2)
            onset = self.rt(self.obj.onset_t)
            end = self.rt(self.obj.collision_t if self.obj.collision_t else self.obj.end_t)
            advance(0.0, until=lambda: self.obj.busy)
            advance(0.2)
            self.obj.cfg.stop_mm = 2.0
        s.actions.listeners.remove(acts.append)
        if link is not None and link.fast and link.n_fast > nfe0:
            cross = list(link.fast_events)[-min(link.n_fast - nfe0, len(link.fast_events)):]
        cross = [e for e in cross if e.group == "escape" and e.sim_time is not None
                 and e.sim_time >= t_req]
        jst = [a for a in acts if a.name == "jump" and a.kind == "start"]
        jen = [a for a in acts if a.name == "jump" and a.kind in ("end", "cancel")]
        takeoff = (self.rt(jst[0].time + jen[0].info["takeoff_after_s"])
                   if jst and jen and "takeoff_after_s" in (jen[0].info or {}) else None)
        sent = (rv.sent[ns_rv:] if real else geo.sent[ns_geo:])
        sent = [e for e in sent if e.kind == "loom" and e.sim_time >= t_req]
        win = sorted((v for v in states.values() if v[0] is not None and v[0] > t_req), key=lambda v: v[0])
        thr = link.cfg.jump_escape_hz if link is not None else 60.0
        gf = [(t, d.get("escape", 0.0)) for t, d in win]
        gf_thr = next((t for t, g in gf if g > thr), None)
        # jumps after the stimulus request only (a GF burst left over from the previous
        # trial can fire a jump right after the reset)
        fired = [f for f in (trig.fired[nf0:] if trig else []) if f[1] == "jump" and f[0] >= t_req]
        rel = lambda t: None if t is None else round(t - onset, 4)  # noqa: E731
        # turn-DN asymmetry after onset (DNa01/02: + = left turn neurons stronger)
        post = [d for t, d in win if t >= onset]
        pre = [d for t, d in win if t < onset]
        asym = lambda L: float(np.mean([d.get("turn_L", 0) - d.get("turn_R", 0) for d in L])) if L else None  # noqa: E731
        peak_l = max((float(h[1][0]) for h in list(rv.history)[nh:]), default=0.0) if real else None
        row = {
            "trial": name, "sense": sense, "onset_rt": round(onset, 4),
            "end_rel_onset": rel(end),
            "n_loom_events": len(sent),
            "loom_events_LR": [sum(e.side == "left" for e in sent), sum(e.side == "right" for e in sent)],
            "first_loom_rel": rel(sent[0].sim_time) if sent else None,
            "gf_peak_hz": round(max((g for _, g in gf if _ >= onset), default=0.0), 0),
            "gf_thr_rel": rel(gf_thr),
            "jump_rel": rel(fired[0][0]) if fired else None,
            "jumped": bool(fired),
            "gf_cross_rel": rel(cross[0].sim_time) if cross else None,
            "takeoff_rel": rel(takeoff),
            "turn_asym_pre": None if asym(pre) is None else round(asym(pre), 1),
            "turn_asym_post": None if asym(post) is None else round(asym(post), 1),
            "max_turnL_post": round(max((d.get("turn_L", 0) for d in post), default=0), 0),
            "max_turnR_post": round(max((d.get("turn_R", 0) for d in post), default=0), 0),
            "peak_lc4L_hz": peak_l,
            "wall_s": round(time.time() - wall0, 1),
        }
        if kind == "swat":
            row["outcome"] = outcome
        return row


def run_sim(args) -> int:
    rig = Rig(args)
    rows = []
    try:
        print(f"session built in {rig.build_s:.1f}s; backend {rig.rv.backend}; "
              f"flyvis init {getattr(rig.rv.net, 'init_s', 0):.1f}s", flush=True)
        for name in [t for t in args.trials.split(",") for _ in range(args.repeat)]:
            for sense in args.sense.split(","):
                r = rig.run(name, sense)
                rows.append(r)
                print(json.dumps(r), flush=True)
        eyes = rig.rv.eyes
        cost = {"eye_ms_per_sample": round(eyes.ms_per_sample(), 2),
                "flyvis_ms_per_step": round(rig.rv.net.ms_per_step(), 2) if rig.rv.net else None,
                "pipeline_ms_per_frame": round(rig.rv.ms_per_frame(), 2)}
        print(json.dumps(cost))
        if args.json:
            args.json.write_text(json.dumps({"rows": rows, "cost": cost}, indent=1))
    finally:
        rig.close()
    return 0


def make_figure(args) -> int:
    """Live no-brain trial (default loom_fast): what the eyes see at the looming
    moment (hex ommatidia images), T4 / T5 activity of the looming eye, the LPLC2 /
    LC4 model unit maps, and the rates over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from flygym.vision.retina import Retina

    args.no_brain = True
    rig = Rig(args)
    rv = rig.rv
    snaps = []  # (t, frame, t4, t5, lplc2_map, lc4_map)

    def grab(t, f):
        if rv.net is not None and rv.net.activity is not None and rig.obj.busy:
            t4, t5 = rv.net.motion()
            b = rv.bridge.last
            snaps.append((t, f.copy(), t4[:, :, :].copy(), t5.copy(),
                          b["lplc2_map"].copy(), b["lc4_map"].copy()))

    try:
        rv.eyes.listeners.append(grab)  # after RealVision.on_frame: sees this frame's state
        name = args.trials.split(",")[0] if args.trials else "loom_fast"
        rig.run(name, "real")
        on = rig.obj.onset_t
        hist = [h for h in rv.history if h[0] >= on - 0.3]
    finally:
        rig.close()
    k = int(np.argmax([s_[4].max() for s_ in snaps]))  # strongest LPLC2 moment
    t, f, t4, t5, lp, lc = snaps[k]
    eye = int(np.argmax(lp.max(axis=1)))
    ret = Retina()
    hexim = lambda v: ret.hex_pxls_to_human_readable(np.asarray(v, dtype=np.float32)[:, None])[..., 0]  # noqa: E731
    fig = plt.figure(figsize=(15, 10.5), dpi=60)
    gs = fig.add_gridspec(3, 6, hspace=0.35, wspace=0.15)
    for i, lab in enumerate(("left eye", "right eye")):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        ax.imshow(hexim(f[i]), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{lab}: 721 ommatidia, t = {t - on:+.3f} s from onset", fontsize=10)
        ax.axis("off")
    for j, (arr, nm) in enumerate(((lp, "LPLC2 model units"), (lc, "LC4 model units"))):
        ax = fig.add_subplot(gs[0, 4 + j])
        ax.imshow(hexim(arr[eye]), cmap="magma", vmin=0)
        ax.set_title(f"{nm} ({('left', 'right')[eye]})\nmax {arr[eye].max():.3f}", fontsize=9)
        ax.axis("off")
    base = snaps[0]
    for row, (T, B, fam) in enumerate(((t4, base[2], "T4 (ON)"), (t5, base[3], "T5 (OFF)"))):
        d = T[eye] - B[eye]
        vmax = max(float(np.abs(d).max()), 1e-6)
        for sidx, sub in enumerate("abcd"):
            ax = fig.add_subplot(gs[1, row * 3 + sidx] if sidx < 3 else gs[2, row * 3])
            ax.imshow(hexim(d[sidx]), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"{fam[:2]}{sub} - baseline", fontsize=9)
            ax.axis("off")
    ax = fig.add_subplot(gs[2, 1:3])
    ax2 = fig.add_subplot(gs[2, 4:6])
    th = np.array([h[0] for h in hist]) - on
    for i, (c, lab) in enumerate((("tab:blue", "left"), ("tab:orange", "right"))):
        ax.plot(th, [h[1][i] for h in hist], color=c, label=f"LC4 {lab}")
        ax2.plot(th, [h[2][i] for h in hist], color=c, label=f"LPLC2 {lab}")
    for a, nm in ((ax, "LC4 rate (Hz)"), (ax2, "LPLC2 rate (Hz)")):
        a.axvline(0, color="k", lw=0.8, ls="--")
        a.axvline(t - on, color="0.5", lw=0.8)
        if rig.obj.collision_t is not None:
            a.axvline(rig.obj.end_t - on, color="r", lw=0.8, ls=":")
        a.set_xlabel("time from motion onset (s); red: object stops 2 mm from the head")
        a.set_ylabel(nm)
        a.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Real vision: dark sphere looming ({name}), FlyGym eyes -> flyvis T4/T5 -> "
                 f"LC4 / LPLC2 bridge", fontsize=12)
    fig.savefig(args.figure, bbox_inches="tight")
    print(f"figure -> {args.figure}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--rate", type=float, default=100.0, help="eye / flyvis rate (Hz)")
    p.add_argument("--bridge", default="{}", help="JSON BridgeConfig overrides")
    p.add_argument("--sim", action="store_true")
    p.add_argument("--trials", default="walk,loom_fast,loom_slow,recede,pass_left,swat2")
    p.add_argument("--sense", default="real,geometric")
    p.add_argument("--terrain", default="flat")
    p.add_argument("--walk", type=float, default=0.6, help="walking before each stimulus (s)")
    p.add_argument("--chunk", type=int, default=50, help="physics steps per brain update")
    p.add_argument("--steer", action="store_true", help="LC10a steering population on")
    p.add_argument("--no-brain", action="store_true", help="vision only (no brain process)")
    p.add_argument("--no-fast", action="store_true",
                   help="old window-only readout (BrainLinkConfig.fast_path=False)")
    p.add_argument("--repeat", type=int, default=1, help="run each trial N times")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--figure", type=Path, default=None)
    args = p.parse_args(argv)
    if args.figure is not None:
        if args.trials == p.get_default("trials"):
            args.trials = "loom_fast"
        return make_figure(args)
    if args.calibrate:
        return calibrate(args)
    if args.sim:
        return run_sim(args)
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
