"""Integration: Simulation extension points + the wired-up app Session."""

import csv
import json

import mujoco as mj
import numpy as np
import pytest
from flygym.compose import ContactParams

from perpetualfly import AppConfig, Simulation, SimulationInstabilityError
from perpetualfly.app import Session, build_arg_parser, config_from_args
from perpetualfly.config import CameraConfig
from perpetualfly.metrics import FallState
from perpetualfly.rendering import SmoothFollowCamera
from perpetualfly.terrain import FLY_BIT, TERRAIN_BIT

_CP = ContactParams()


def _add_ball(world):
    """World extension: a free 0.1 mg ball above the fly that collides with the fly
    and the ground (FlyGym ground-contact parameters, priority so they're used)."""
    body = world.mjcf_root.worldbody.add_body(name="ext_ball", pos=(0.3, 0.0, 3.5))
    body.add_freejoint(name="ext_ball_free")
    body.add_geom(name="ext_ball_geom", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.3, 0, 0),
                  mass=1e-4, contype=0, conaffinity=FLY_BIT | TERRAIN_BIT, priority=1,
                  condim=3, friction=(_CP.sliding_friction, _CP.torsional_friction,
                                      _CP.rolling_friction),
                  solref=_CP.get_solref_tuple(), solimp=_CP.get_solimp_tuple(),
                  margin=_CP.margin, rgba=(1, 0, 0, 1))


def test_world_extension_body_collides_with_fly_and_ground():
    sim = Simulation(AppConfig(), world_extensions=[_add_ball])
    try:
        m, d = sim.model, sim.data
        ball = m.body("ext_ball").id
        g = m.geom("ext_ball_geom").id
        # extra free joint exists and the "neutral" keyframe puts it at its spec pose
        # (FlyGym alone would give it qpos = 0; Simulation patches the keyframe)
        assert m.nq == 73 + 7 and m.nv == 72 + 6
        assert sim.other_dofs.size == 6 and sim.fly_dofs.size == 72
        a = int(m.jnt_qposadr[m.joint("ext_ball_free").id])
        np.testing.assert_allclose(m.key_qpos[0, a:a + 7], [0.3, 0, 3.5, 1, 0, 0, 0])
        sim.reset()  # ball falls during the 50 ms warmup and lands on the fly
        partners = set()
        for _ in range(100):  # 0.1 s: rolls off the fly's back onto the ground
            sim.step(10)
            for c in d.contact[: d.ncon]:
                if g in (c.geom1, c.geom2):
                    partners.add(m.geom(c.geom2 if c.geom1 == g else c.geom1).name)
        assert any(p.startswith("nmf/") for p in partners), partners
        assert "ground_plane" in partners, partners
        assert d.xpos[ball, 2] == pytest.approx(0.3, abs=0.05)  # resting on the ground
    finally:
        sim.close()


def test_stability_limits_split_fly_and_other_dofs():
    sim = Simulation(AppConfig(), world_extensions=[_add_ball])
    try:
        a = int(sim.other_dofs[0])
        sim.data.qvel[a] = 1e7  # > max_abs_qvel (fly), < max_abs_qvel_other
        sim.check_stability()
        sim.data.qvel[a] = 1e10
        with pytest.raises(SimulationInstabilityError, match="non-fly"):
            sim.check_stability()
        sim.data.qvel[a] = np.nan
        with pytest.raises(SimulationInstabilityError, match="non-finite qvel"):
            sim.check_stability()
        sim.reset()
        sim.data.qvel[int(sim.fly_dofs[10])] = 1e7
        with pytest.raises(SimulationInstabilityError, match="fly"):
            sim.check_stability()
    finally:
        sim.close()


def test_pre_reset_hooks_run_before_keyframe_reset():
    sim = Simulation(AppConfig())
    try:
        sim.step(100)
        seen = {}
        sim.pre_reset_hooks.append(lambda s: seen.setdefault("pre", (s.time, s.step_count)))
        sim.reset_hooks.append(lambda s: seen.setdefault("post", (s.time, s.step_count)))
        sim.reset()
        assert seen["pre"][1] == 100 and seen["pre"][0] > seen["post"][0]
        assert seen["post"][1] == 0
    finally:
        sim.close()


def test_camera_catches_up_after_a_launch():
    cam = SmoothFollowCamera(CameraConfig())
    cam.update(0.0, np.array([0.0, 0.0, 1.1]), 0.0)
    cam.update(0.015, np.array([8.0, 3.0, 6.0]), 0.0, ground_z=0.0)  # one frame later
    lag = np.linalg.norm(np.array([8.0, 3.0, 6.0]) - cam.cam.lookat)
    assert lag <= CameraConfig().max_lag + 1e-9
    assert cam.cam.distance > CameraConfig().distance  # zoomed out: fly is high up
    # heading frozen while upside down
    h = cam._heading
    cam.update(0.03, np.array([8.0, 3.0, 1.0]), 2.0, tilt_deg=170.0)
    assert cam._heading == h


def test_cli_defaults_and_flags():
    cfg = config_from_args(build_arg_parser().parse_args([]))
    assert cfg.terrain.difficulty == "normal" and not cfg.auto_perturb.enabled
    assert cfg.session.hit_mode == "whip"
    assert config_from_args(build_arg_parser().parse_args(
        ["--hit-mode", "shove"])).session.hit_mode == "shove"
    assert cfg.logging.enabled and cfg.session.auto_reset_after_s is None
    cfg = config_from_args(build_arg_parser().parse_args(
        ["--terrain", "hard", "--terrain-seed", "3", "--auto-perturb", "--auto-min", "1",
         "--auto-max", "2", "--auto-levels", "2:1,3:3", "--no-log", "--log-hz", "10",
         "--auto-reset-after", "5", "--strength", "3"]))
    assert cfg.terrain.difficulty == "hard" and cfg.terrain.seed == 3
    ap = cfg.auto_perturb
    assert ap.enabled and (ap.min_interval_s, ap.max_interval_s) == (1, 2)
    assert ap.levels == (2, 3) and ap.level_weights == (1.0, 3.0)
    assert not cfg.logging.enabled and cfg.logging.sample_hz == 10
    assert cfg.session.auto_reset_after_s == 5 and cfg.perturbation.default_level == 3


def test_session_logs_hits_spawns_and_summary(tmp_path):
    cfg = AppConfig()
    cfg.terrain.difficulty = "normal"
    cfg.auto_perturb.enabled = True
    cfg.auto_perturb.first_hit_after_s = 0.3
    cfg.auto_perturb.min_interval_s = cfg.auto_perturb.max_interval_s = 0.4
    cfg.logging.runs_dir = str(tmp_path)
    cfg.session.hit_mode = "shove"  # this test covers the external-force hits
    # hits every 0.4 s may knock the fly down; keep them coming (pausing while down
    # is covered in tests/test_quick_wins.py)
    cfg.session.auto_perturb_pause_when_down = False
    s = Session(cfg, log=True, say=lambda msg: None)
    try:
        s.sim.step(5000)
        assert s.handle_whip_key("space").startswith("[hit/key]")
        assert s.handle_whip_key("3").startswith("[strength]")
        assert s.handle_whip_key("h") == "[hit-mode] whip"  # H toggles the hit mode
        assert s.handle_whip_key("h") == "[hit-mode] shove"
        assert s.spawn("rock").startswith("[spawn] rock")
        assert s.flatten().startswith("[flatten]")
        s.sim.step(5000)
        s.reset("manual")
        s.sim.step(2000)
        n_hits = s.metrics.n_hits
        assert n_hits >= 3
    finally:
        s.close("test")
    run_dir = s.logger.run_dir
    rows = list(csv.DictReader(open(run_dir / "events.csv")))
    hits = [r for r in rows if r["event_type"] == "hit"]
    assert len(hits) == n_hits
    assert {"spawn", "flatten", "strength", "manual_reset", "reset", "hit_mode"} <= {
        r["event_type"] for r in rows}
    assert all(r["terrain_type"] for r in rows)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["complete"] and summary["quit_reason"] == "test"
    assert summary["metrics"]["n_hits"] == n_hits and summary["metrics"]["n_resets"] == 1
    config = json.loads((run_dir / "config.json").read_text())
    assert config["app"]["terrain"]["difficulty"] == "normal"
    assert config["procedural_terrain"]["difficulty"] == "normal"
    assert config["app"]["auto_perturb"]["enabled"] is True


def test_auto_reset_after_fall(tmp_path):
    cfg = AppConfig()
    cfg.session.auto_reset_after_s = 0.6
    cfg.logging.runs_dir = str(tmp_path)
    s = Session(cfg, log=True, say=lambda msg: None)
    try:
        s.sim.step(5000)
        # Flip the fly onto its back (test setup only; the app never does this).
        a = s.sim._free_qpos
        s.sim.data.qpos[a + 2] += 1.0
        s.sim.data.qpos[a + 3 : a + 7] = [0.0, 1.0, 0.0, 0.0]
        for _ in range(30):  # 3 s, chunked like the app loop
            s.sim.step(1000)
            s.after_physics()
            if s.n_auto_resets:
                break
        assert s.n_auto_resets == 1 and s.metrics.n_falls >= 1
        assert s.metrics.n_resets == 1 and s.detector.state == FallState.UPRIGHT
    finally:
        s.close("test")
    rows = list(csv.DictReader(open(s.logger.run_dir / "events.csv")))
    kinds = [r["event_type"] for r in rows]
    assert "fall" in kinds and "auto_reset" in kinds and kinds.index("fall") < kinds.index("auto_reset")
    ts = [float(r["timestamp"]) for r in rows]  # run time keeps counting across resets
    assert all(a <= b + 1e-9 for a, b in zip(ts, ts[1:])), ts
