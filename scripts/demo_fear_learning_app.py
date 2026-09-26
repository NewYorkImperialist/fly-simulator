"""Behaviour test of fear learning in the full app (headless, real brain, steering on).
docs/FEAR_LEARNING.md.

Protocol (sim time): the fly walks on flat ground; test trials put an odour zone
around the fly for ``--test-s`` (then remove it and rest ``--rest-s``). Pre-tests
A, B -> training ``--trials`` x [zone A + one whip crack (-> punishment DANs,
stand-in) + rest; zone B, no crack + rest] -> post-tests A, B. Measures the
brain's turn drive (L - R amplitude) and the actual heading change while in the
zone, and the gamma1pedc / alpha3 KC>MBON efficacies. ``--no-learning`` runs the
same protocol with plasticity off (control).

    .venv/bin/python scripts/demo_fear_learning_app.py [--no-learning] [--frames DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perpetualfly.app import Session  # noqa: E402
from perpetualfly.brain_link import BrainLink  # noqa: E402
from perpetualfly.config import AppConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-learning", action="store_true")
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--test-s", type=float, default=2.0)
    ap.add_argument("--rest-s", type=float, default=5.0)
    ap.add_argument("--radius", type=float, default=40.0, help="zone radius (mm); the fly walks ~14 mm/s")
    ap.add_argument("--frames", type=str, default=None, help="save a few PNG frames here")
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    learn = not a.no_learning

    cfg = AppConfig()
    cfg.logging.enabled = False
    cfg.terrain.difficulty = "flat"
    cfg.auto_perturb.enabled = False
    cfg.session.auto_reset_after_s = 3.0
    cfg.odor.enabled = True
    cfg.odor.learning = learn
    cfg.odor.layout = "none"
    cfg.odor.radius_mm = a.radius
    cfg.brain.enabled = True
    cfg.brain.window = False
    cfg.brain.steer = True
    cfg.brain.odors = True
    cfg.brain.learning = learn
    t0 = time.time()
    link = BrainLink(cfg.brain, headless=True)
    s = Session(cfg, log=False, brain=link, say=lambda m: None)
    link.wait_ready(180)
    print(f"ready in {time.time() - t0:.0f} s", flush=True)
    chunk = max(1, cfg.render.render_every_steps)
    h = s.odor
    renderer = None
    if a.frames:
        from perpetualfly.app import _make_renderer

        Path(a.frames).mkdir(parents=True, exist_ok=True)
        renderer = _make_renderer(s)

    def advance(sec, rec=None):
        t_end = s.run_time() + sec
        while s.run_time() < t_end:
            s.step(chunk)
            s.after_physics()
            if rec is not None:
                d = link.latest.descending if link.latest is not None else {}
                rec.append((s.run_time(), s.sim.heading(), float(link.drive[0] - link.drive[1]),
                            d.get("turn_L", 0.0), d.get("turn_R", 0.0), h.current or "-"))

    def frame(name):
        if renderer is None:
            return
        import cv2

        sim = s.sim
        img = renderer.render(sim.data, sim.time, sim.thorax_position(), sim.heading())
        cv2.imwrite(str(Path(a.frames) / f"{name}.png"), img[..., ::-1])

    def test(odor, tag):
        h.zones.clear()
        h.spawn(odor)
        rec = []
        advance(0.3)  # let the stimulus reach the brain
        frame(f"{tag}_{odor}")
        advance(a.test_s - 0.3, rec)
        h.zones.clear()
        advance(a.rest_s)
        r = np.array([x[:5] for x in rec], dtype=float)
        inside = np.array([x[5] == odor for x in rec])
        hd = np.unwrap(r[:, 1])
        dt = np.diff(r[:, 0])
        yaw = np.diff(hd) / np.maximum(dt, 1e-9)
        res = {"odor": odor, "tag": tag, "frac_inside": float(inside.mean()),
               "turn_drive": float(r[inside, 2].mean()) if inside.any() else 0.0,
               "yaw_deg_s": float(np.degrees(yaw[inside[1:]].mean())) if inside[1:].any() else 0.0,
               "turn_L_hz": float(r[inside, 3].mean()) if inside.any() else 0.0,
               "turn_R_hz": float(r[inside, 4].mean()) if inside.any() else 0.0}
        lr = (getattr(link.latest, "learning", None) or {})
        eff = lr.get("efficacy_by_odor") or {}
        res["eff"] = {o: {k: round(v, 3) for k, v in e.items()} for o, e in eff.items()}
        print(f"  {tag:5s} {odor}: turn drive {res['turn_drive']:+.3f}  yaw {res['yaw_deg_s']:+6.1f} deg/s"
              f"  DNa L/R {res['turn_L_hz']:.1f}/{res['turn_R_hz']:.1f} Hz  (inside {res['frac_inside']:.2f})"
              f"  eff[PPL101] A {eff.get('A', {}).get('PPL101', 1):.2f} B {eff.get('B', {}).get('PPL101', 1):.2f}",
              flush=True)
        return res

    out = {"learning": learn, "trials": []}
    advance(2.0)  # settle
    base = []
    advance(a.test_s, base)
    b = np.array([x[:3] for x in base], float)
    print(f"  baseline: turn drive {b[:, 2].mean():+.3f} yaw "
          f"{np.degrees(np.diff(np.unwrap(b[:, 1])).sum() / (b[-1, 0] - b[0, 0])):+.1f} deg/s", flush=True)
    for tag in ("pre",):
        for o in ("A", "B"):
            out["trials"].append(test(o, tag))
    n_hits = 0
    for k in range(a.trials):
        h.zones.clear()
        h.spawn("A")
        advance(0.4)
        if k == 0:
            frame("train_A_before_crack")
        before = h.n_punish.get("A", 0)
        if s.whip is not None:
            s.whip.crack("random", 1, source="script")
            advance(1.2)
        if h.n_punish.get("A", 0) == before and learn:  # whip missed: stand-in punishment
            h.punish(source="script")
        n_hits += h.n_punish.get("A", 0) - before
        advance(max(a.test_s - 1.6, 0.4))
        h.zones.clear()
        advance(a.rest_s)
        h.spawn("B")
        advance(a.test_s)
        h.zones.clear()
        advance(a.rest_s)
        lr = (getattr(link.latest, "learning", None) or {})
        eff = lr.get("efficacy_by_odor") or {}
        print(f"  train {k + 1}: punishments A {h.n_punish.get('A', 0)} (whip {h.n_punish.get('A', 0)}); "
              f"eff[PPL101] A {eff.get('A', {}).get('PPL101', 1):.2f} B {eff.get('B', {}).get('PPL101', 1):.2f}"
              f" [PPL106] A {eff.get('A', {}).get('PPL106', 1):.2f} B {eff.get('B', {}).get('PPL106', 1):.2f}",
              flush=True)
    for o in ("A", "B"):
        out["trials"].append(test(o, "post"))
    out["summary"] = h.summary()
    out["learning_state"] = {k: v for k, v in (getattr(link.latest, "learning", None) or {}).items()
                             if k in ("efficacy_by_odor", "n_updates", "n_runaway", "t_learned_s")}
    print(f"punishments {h.n_punish}, whip hits outside zones {h.n_hits_outside}, runaway chunks "
          f"{out['learning_state'].get('n_runaway')}; wall {time.time() - t0:.0f} s", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    s.close("done")
    link.close()


if __name__ == "__main__":
    main()
