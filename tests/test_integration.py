"""Integration: --swatter / --stress / --whip-vision / --course / --job wired into
the app (fly_simulator/app.py Session + CLI + keys + HUD + logging)."""

import csv
import json
from pathlib import Path

import pytest

from fly_simulator import AppConfig
from fly_simulator.app import (
    ConfigError,
    Session,
    _make_renderer,
    build_arg_parser,
    config_from_args,
    main,
    run,
)
from fly_simulator.interaction.keyboard import decode_key

SYN = {"n": 60, "p_conn": 0.1, "seed": 0}


def _cfg(*argv: str) -> AppConfig:
    return config_from_args(build_arg_parser().parse_args(list(argv)))


# ------------------------------------------------------------------ CLI / config
def test_feature_flags_and_implications():
    c = _cfg()
    assert not (c.swatter.enabled or c.stress.enabled or c.whip_vision.enabled)
    assert c.course.name is None and c.job.name is None and not c.brain.enabled
    assert c.brain.window_s == 0.1 and c.brain.sync_wait_s == 0.0
    # the swatter alone does not start the brain (and leaves its pacing alone)
    c = _cfg("--swatter")
    assert c.swatter.enabled and not c.brain.enabled and c.brain.window_s == 0.1
    assert not c.swatter.flight  # escape-flight emulation stays off
    # stress / whip vision imply the brain; swatter / vision set the fast pacing
    c = _cfg("--stress")
    assert c.stress.enabled and c.brain.enabled and c.brain.sync_wait_s == 0.0
    c = _cfg("--whip-vision")
    assert c.whip_vision.enabled and c.brain.enabled
    assert (c.brain.window_s, c.brain.sync_wait_s) == (0.02, 0.05)
    c = _cfg("--brain-actions", "--stress", "--swatter", "--whip-vision")
    assert c.brain.actions and c.brain.steer and c.stress.enabled and c.swatter.enabled
    assert (c.brain.window_s, c.brain.sync_wait_s) == (0.02, 0.05)
    # course: flat base terrain, no app auto reset; --course-loop
    c = _cfg("--course", "gauntlet", "--course-loop", "--auto-reset-after", "5")
    assert c.course.name == "gauntlet" and c.course.loop
    assert c.session.auto_reset_after_s is None and c.terrain.difficulty == "flat"
    # job: config overrides; no idle whip in the scene, hit keys shove
    c = _cfg("--job", "kebab", "--job-config", '{"gap": 1.6}')
    assert c.job.name == "kebab" and c.job.config == {"gap": 1.6}
    assert not c.whip.enabled and c.session.hit_mode == "shove"
    assert _cfg("--job", "sisyphus", "--whip-vision").whip.enabled  # vision keeps the whip


@pytest.mark.parametrize("argv, match", [
    (["--course", "gauntlet", "--job", "kebab"], "can't be combined"),
    (["--course-loop"], "needs --course"),
    (["--job-config", "{}"], "needs --job"),
    (["--job", "kebab", "--job-config", "[1]"], "JSON object"),
    (["--job", "kebab", "--job-config", "{bad"], "not valid JSON"),
    (["--job", "nope"], "unknown job"),
    (["--job", "kebab", "--job-config", '{"bogus": 1}'], "bogus"),
    (["--course", "nope"], "no built-in course"),
])
def test_incompatible_or_bad_feature_args_are_refused(argv, match, capsys):
    with pytest.raises(ConfigError, match=match):
        _cfg(*argv)
    assert main(argv + ["--headless", "--no-log"]) == 2
    assert "ERROR:" in capsys.readouterr().err


def test_session_refuses_course_with_job():
    cfg = AppConfig()
    cfg.course.name, cfg.job.name = "tutorial", "kebab"
    with pytest.raises(ConfigError):
        Session(cfg, log=False)


def test_shift_letters_decode_only_when_asked():
    assert decode_key(ord("V")) == "v"  # the default stays lower-case
    assert decode_key(ord("V"), shift_names=True) == "shift+v"
    assert decode_key(ord("v"), shift_names=True) == "v"
    assert decode_key(ord("?"), shift_names=True) == "?"


# ------------------------------------------------------------------ everything at once
def test_all_brain_features_compose_in_one_run(tmp_path, capsys):
    pytest.importorskip("numba")
    cfg = _cfg("--brain-actions", "--stress", "--swatter", "--whip-vision", "--terrain", "flat")
    cfg.brain.window, cfg.brain.synthetic = False, dict(SYN)
    cfg.logging.runs_dir = str(tmp_path)
    cfg.logging.sample_hz = 10
    res = run(cfg, headless=True, max_seconds=1.5,
              script_keys=[(0.1, "4"), (0.15, "v"), (0.2, "space")])
    out = capsys.readouterr().out
    assert "features: swatter L2" in out and "whip vision" in out and "stress" in out
    assert "swatter L4 lightning" in out  # 1-4 set the swat level too
    assert "[swatter] started: L4 lightning from rear" in out
    run_dir = Path(res.run_dir)
    rows = list(csv.DictReader(open(run_dir / "events.csv")))
    kinds = {r["event_type"] for r in rows}
    assert "swat" in kinds and ("whip" in kinds or "whip_miss" in kinds), kinds
    swat = next(r for r in rows if r["event_type"] == "swat")
    assert json.loads(swat["details"])["outcome"] in ("hit", "grazed", "dodged", "miss")
    header = next(csv.reader(open(run_dir / "metrics.csv")))
    assert {"octopamine", "stress_jump_hz", "brain_time", "brain_lag"} <= set(header)
    s = json.loads((run_dir / "summary.json").read_text())
    assert s["swatter"]["n_swats"] == 1 and s["swatter"]["level"] == 4
    assert s["stress"]["enabled"] and s["whip_vision"]["sources"] == ["whip", "swatter"]
    assert json.loads((run_dir / "config.json").read_text())["app"]["brain"]["window_s"] == 0.02


def test_swatter_session_hud_keys_and_shared_vision(tmp_path):
    cfg = AppConfig()
    cfg.whip_vision.enabled = True
    cfg.swatter.enabled = True
    s = Session(cfg, log=False, say=lambda m: None)
    try:
        assert s.vision is s.swatter_handle.vision  # one LoomingVision, two sources
        assert [src.name for src in s.vision.sources] == ["whip", "swatter"]
        assert s.handle_whip_key("1").endswith("swatter L1 lazy")
        assert s.swat_key("shift+v").endswith("from random")
        assert s.swatter.busy and not s.auto_hits_allowed()  # auto hits wait like a crack
    finally:
        s.close("test")
        s.sim.close()
    s = Session(AppConfig(), log=False, say=lambda m: None)
    try:
        assert "run with --swatter" in s.swat_key("v")
    finally:
        s.sim.close()


# ------------------------------------------------------------------ jobs / courses
def test_job_in_the_app_camera_hud_and_summary(tmp_path, capsys):
    cfg = _cfg("--job", "sisyphus")
    cfg.logging.runs_dir = str(tmp_path)
    cfg.logging.sample_hz = 5
    cfg.render.width, cfg.render.height = 240, 160
    res = run(cfg, headless=True, max_seconds=0.6, script_keys=[(0.5, "i")])
    out = capsys.readouterr().out
    assert "features: job sisyphus" in out and "job sisyphus: summits" in out
    s = json.loads((Path(res.run_dir) / "summary.json").read_text())
    assert s["job"]["job"] == "sisyphus"
    assert list(Path(res.run_dir).glob("shot*_fly_hud.png"))
    # C cycles job / follow / side / top
    cfg = _cfg("--job", "sisyphus")
    cfg.render.width, cfg.render.height = 120, 80
    sess = Session(cfg, log=False, say=lambda m: None)
    try:
        r = _make_renderer(sess)
        assert r.camera.mode == "job" and r.camera.cycle_mode() == "follow"
        r.close()
        assert sess.job is not None and sess.ground_height(0, 0) == sess.job.ground_height(0, 0)
    finally:
        sess.close("test")
        sess.sim.close()


def test_course_quits_at_the_finish(tmp_path, capsys):
    course = tmp_path / "mini.json"
    course.write_text(json.dumps({"name": "mini", "lead_in": 2.0, "start_x": 1.0,
                                  "sections": [{"type": "flat", "length": 3},
                                               {"type": "finish"}]}))
    cfg = _cfg("--course", str(course))
    cfg.logging.runs_dir = str(tmp_path)
    cfg.logging.sample_hz = 5
    res = run(cfg, headless=True, max_seconds=6.0, script_keys=[(0.2, "]"), (0.3, "f")])
    out = capsys.readouterr().out
    assert res.quit_reason == "course finished", out
    assert "[course] terrain difficulty is fixed" in out and "flatten is disabled" in out
    assert "[FINISHED] course mini lap 1" in out
    assert (Path(res.run_dir) / "course_results.json").exists()
