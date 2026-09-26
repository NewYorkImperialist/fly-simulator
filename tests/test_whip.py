"""The physical whip: model, idle clearance, strikes through contact, metrics wiring."""

import csv
import json
import math

import numpy as np
import pytest

from fly_simulator import AppConfig, Simulation
from fly_simulator.app import Session
from fly_simulator.interaction.whip import Whip, WhipConfig
from fly_simulator.metrics import ContactClassifier, FallDetector


def _sim(whip_cfg: WhipConfig | None = None):
    whip = Whip(whip_cfg or WhipConfig())
    sim = Simulation(AppConfig(), world_extensions=[whip.extension])
    whip.attach(sim)
    return sim, whip


def _steps(sim, seconds):
    return int(round(seconds / sim.timestep))


def test_model_builds_and_resets_to_idle():
    sim, whip = _sim()
    try:
        m, d = sim.model, sim.data
        n = whip.cfg.n_segments
        assert m.nmocap == 1 and m.neq == 1
        assert sim.other_dofs.size == 6 + 2 * n  # grip free joint + 2 hinges / segment
        # segment masses sum to the configured chain mass; whip geoms hit only the fly
        seg_mass = sum(m.body_mass[b] for b in whip.seg_bodies)
        assert seg_mass == pytest.approx(whip.cfg.chain_mass, rel=1e-6)
        for g in whip.whip_geoms:
            assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 8
        assert whip.min_distance_to_fly() > 2.0
        sim.step(_steps(sim, 0.3))
        whip.crack("left", 1)
        sim.step(_steps(sim, 0.1))  # mid-crack
        assert whip.phase != "idle"
        sim.reset()
        assert whip.phase == "idle" and not whip.busy
        # handle and welded grip back together at the idle pose, clear of the fly
        grip = d.xpos[whip.grip_body]
        assert np.linalg.norm(grip - d.mocap_pos[whip.mocap_id]) < 0.05
        assert whip.min_distance_to_fly() > 2.0
        sim.step(_steps(sim, 0.2))
        assert np.all(np.isfinite(d.qpos))
    finally:
        sim.close()


def test_idle_whip_never_touches_the_fly_flat_and_terrain():
    sim, whip = _sim()
    try:
        dmin = np.inf
        for _ in range(30):  # 3 s of walking
            sim.step(1000)
            dmin = min(dmin, whip.min_distance_to_fly())
        assert whip.stray_contact_steps == 0
        assert dmin > 1.0, dmin
    finally:
        sim.close()
    cfg = AppConfig()
    cfg.terrain.difficulty = "normal"
    s = Session(cfg, log=False, say=lambda msg: None)
    try:
        for _ in range(20):  # 2 s over procedural terrain
            s.sim.step(1000)
        assert s.whip.stray_contact_steps == 0
        assert s.whip.min_distance_to_fly() > 1.0
    finally:
        s.close("test")


def test_crack_from_left_connects_and_pushes_the_fly_right():
    sim, whip = _sim()
    try:
        sim.step(_steps(sim, 0.8))
        events = []
        whip.listeners.append(events.append)
        yaw = sim.heading()
        right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
        p0 = sim.thorax_position()
        assert whip.crack("left", 3, source="test") == "started"
        assert whip.crack("left", 1) == "queued"  # while the first one is busy
        sim.step(_steps(sim, 0.45))
        ev = events[0]
        assert ev.hit and ev.side == "left" and ev.direction_name == "from_left"
        assert ev.impulse_uNs > 0.2 and ev.magnitude_uN > 0 and ev.duration_s > 0
        assert np.dot(ev.impulse_vec, right) > 0.6 * ev.impulse_uNs  # mostly rightward
        assert ev.body.startswith("nmf/")
        disp = sim.thorax_position() - p0
        assert np.dot(disp, right) > 2.0  # mm, knocked to its right
        sim.step(_steps(sim, 0.5))  # the queued crack runs too
        assert len(events) == 2 and whip.stray_contact_steps == 0
    finally:
        sim.close()


def test_whip_contacts_are_not_body_ground_contacts():
    sim, whip = _sim()
    det = FallDetector(sim)
    try:
        sim.step(_steps(sim, 0.6))
        cls = ContactClassifier(sim.model, sim.fly_name)
        thorax_geoms = {sim.model.geom("nmf/c_thorax").id, sim.model.geom("nmf/c_head").id}
        whip_geoms = set(int(g) for g in whip.whip_geoms)
        whip.crack("overhead", 1)  # chops onto the thorax of an upright fly
        seen = 0
        for _ in range(_steps(sim, 0.3)):
            sim.step(1)
            d = sim.data
            pairs = {(int(c.geom1), int(c.geom2)) for c in d.contact[: d.ncon]}
            if any((a in whip_geoms and b in thorax_geoms) or (b in whip_geoms and a in thorax_geoms)
                   for a, b in pairs):
                seen += 1
                assert not cls.summarize(d).body_contact
        assert seen > 0
        assert whip.events and whip.events[0].hit
        assert det.state.value == "UPRIGHT"
    finally:
        sim.close()


def test_repeated_cracks_at_every_level_stay_stable():
    sim, whip = _sim()
    try:
        sim.step(_steps(sim, 0.4))
        for level, side in [(1, "right"), (2, "front"), (3, "rear"), (4, "left"),
                            (2, "overhead"), (3, "random")]:
            whip.crack(side, level)
            sim.step(_steps(sim, 0.36))  # step() runs check_stability every 1 ms
        sim.step(_steps(sim, 0.2))
        assert len(whip.events) == 6
        assert np.all(np.isfinite(sim.data.qvel))
        sim.reset()
        whip.crack("left", 4)
        sim.step(_steps(sim, 0.4))
        assert whip.events[-1].hit
    finally:
        sim.close()


def test_whip_is_deterministic():
    out = []
    for _ in range(2):
        sim, whip = _sim(WhipConfig(seed=3))
        try:
            sim.step(_steps(sim, 0.5))
            whip.crack("random", 2)
            sim.step(_steps(sim, 0.35))
            out.append((whip.events[0].impulse_vec, sim.thorax_position()))
        finally:
            sim.close()
    assert out[0][0] == out[1][0]
    np.testing.assert_array_equal(out[0][1], out[1][1])


def test_session_counts_whip_hits_and_logs_misses(tmp_path):
    cfg = AppConfig()
    cfg.logging.runs_dir = str(tmp_path)
    s = Session(cfg, log=True, say=lambda msg: None)
    try:
        assert s.hit_mode == "whip"
        s.sim.step(5000)
        assert s.handle_whip_key("1").startswith("[strength]")
        assert s.handle_whip_key("left").startswith("[whip/key] crack L1")
        s.sim.step(4000)
        assert s.metrics.n_hits == 1
        hud = s.controls.hud_line()
        assert hud.startswith("WHIP L1") and "hit 1" in hud
    finally:
        s.close("test")
    rows = list(csv.DictReader(open(s.logger.run_dir / "events.csv")))
    whip_rows = [r for r in rows if r["event_type"] == "whip"]
    assert len(whip_rows) == 1 and whip_rows[0]["force_direction"] == "from_left"
    det = json.loads(whip_rows[0]["details"])
    assert det["impulse_uNs"] > 0 and det["hit"] and det["magnitude_uN"] > 0
    # the summary's impulse (mean force x duration) is the measured impulse
    summary = json.loads((s.logger.run_dir / "summary.json").read_text())
    assert summary["hits"][0]["impulse_uN_s"] == pytest.approx(det["impulse_uNs"], rel=1e-6)
    assert summary["n_whip_cracks"] == 1 and summary["hit_mode"] == "whip"
    config = json.loads((s.logger.run_dir / "config.json").read_text())
    assert config["app"]["whip"]["levels"][0]["name"] == "gentle"

    # a whip that can't reach the fly: the crack is logged as a miss, not a hit
    cfg = AppConfig()
    cfg.logging.runs_dir = str(tmp_path)
    cfg.whip.contact_radius = 14.0
    s = Session(cfg, log=True, say=lambda msg: None)
    try:
        s.sim.step(3000)
        s.handle_whip_key("right")
        s.sim.step(4000)
        assert s.metrics.n_hits == 0 and not s.whip.events[0].hit
    finally:
        s.close("test")
    kinds = [r["event_type"] for r in csv.DictReader(open(s.logger.run_dir / "events.csv"))]
    assert "whip_miss" in kinds and "whip" not in kinds


def test_auto_perturber_uses_the_hit_mode():
    cfg = AppConfig()
    cfg.auto_perturb.enabled = True
    cfg.auto_perturb.first_hit_after_s = 0.3
    cfg.auto_perturb.min_interval_s = cfg.auto_perturb.max_interval_s = 0.5
    cfg.auto_perturb.levels, cfg.auto_perturb.level_weights = (1,), (1.0,)
    s = Session(cfg, log=False, say=lambda msg: None)
    try:
        s.sim.step(_steps(s.sim, 0.7))  # first auto hit at 0.3 s -> a whip crack
        assert s.whip.n_cracks == 1 and len(s.perturbation.events) == 0
        assert s.handle_whip_key("h") == "[hit-mode] shove"
        s.sim.step(_steps(s.sim, 0.5))  # next auto hit -> a shove
        assert s.whip.n_cracks == 1 and len(s.perturbation.events) == 1
    finally:
        s.close("test")


def test_mid_crack_reset_mode_toggle_and_spawn_while_fallen():
    """Interaction edge cases from the soak review: reset during a crack (and with a
    queued crack), H during a crack, spawning / auto hits while the fly is down."""
    cfg = AppConfig()
    s = Session(cfg, log=False, say=lambda msg: None)
    sim, whip = s.sim, s.whip
    try:
        sim.step(_steps(sim, 0.3))
        assert s.handle_whip_key("3").startswith("[strength]")
        s.handle_whip_key("left")
        assert "queued" in s.handle_whip_key("right")  # second crack queued
        sim.step(_steps(sim, 0.12))  # mid wind-up
        assert whip.phase != "idle"
        s.reset("manual")  # aborts the crack and drops the queued one
        assert whip.phase == "idle" and whip._pending is None and not whip.events
        sim.step(_steps(sim, 0.6))
        assert whip.phase == "idle" and whip.stray_contact_steps == 0
        assert s.metrics.n_hits == 0 and s.detector.state.value == "UPRIGHT"

        # H while a crack is in flight: the crack finishes and is still counted.
        s.handle_whip_key("left")
        sim.step(_steps(sim, 0.05))
        assert s.handle_whip_key("h") == "[hit-mode] shove"
        sim.step(_steps(sim, 0.5))
        assert len(whip.events) == 1 and s.metrics.n_hits == whip.events[0].hit
        assert whip.phase == "idle"

        # Knock it over with an absurd shove, then spawn + auto hits while down.
        s.handle_whip_key("4")
        s.handle_whip_key("u")  # shove straight up (level 4 launches)
        s.handle_whip_key("left")
        for _ in range(30):
            sim.step(_steps(sim, 0.1))
            if s.down_for() is not None:
                break
        assert s.down_for() is not None  # it is down
        s.auto.toggle()
        assert s.spawn("rock").startswith("[spawn] rock")
        sim.step(_steps(sim, 2.0))
        assert np.all(np.isfinite(sim.data.qpos))
        s.reset("manual")
        assert s.down_for() is None and s.terrain.spawn_count == 1
    finally:
        s.close("test")
