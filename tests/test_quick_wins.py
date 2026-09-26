"""Quick wins (docs/ROADMAP.md section A): brain backup flag, Retina scaling, floor
reflections, live terrain difficulty, screenshots, live recording, help overlay,
auto hits paused while down, walked distance, package data."""

import csv
import json
import tomllib
from pathlib import Path

import cv2
import mujoco as mj
import numpy as np
import pytest

from perpetualfly import AppConfig
from perpetualfly.app import KEY_GROUPS, KEY_TABLE, Session, build_arg_parser, config_from_args, run
from perpetualfly.brain_link import BrainLink, BrainLinkConfig
from perpetualfly.display import auto_display_scale
from perpetualfly.interaction.perturbation import AutoPerturbConfig
from perpetualfly.interaction.viewer import compose_frame
from perpetualfly.media import MediaCapture
from perpetualfly.metrics import FallState, RunMetrics
from perpetualfly.terrain import TerrainGenerator

ROOT = Path(__file__).resolve().parents[1]
ACTION_KEYS = {"j", "y", "z", "w", "n", "e", ",", "."}  # perpetualfly/actions (tests/test_actions.py)


def _session(tmp_path=None, auto: AutoPerturbConfig | None = None, **over):
    cfg = AppConfig()
    cfg.terrain.difficulty = over.pop("difficulty", "flat")
    if auto is not None:
        cfg.auto_perturb = auto
    for k, v in over.items():
        setattr(cfg.session, k, v)
    if tmp_path is not None:
        cfg.logging.runs_dir = str(tmp_path)
    msgs: list[str] = []
    return Session(cfg, log=tmp_path is not None, say=msgs.append), msgs


# ------------------------------------------------------------------ A1 / A3 CLI
def test_cli_backup_and_reflection_flags():
    cfg = config_from_args(build_arg_parser().parse_args([]))
    assert cfg.render.reflections and not cfg.brain.backup and not cfg.brain.enabled
    cfg = config_from_args(build_arg_parser().parse_args(["--brain-backup", "--no-reflections"]))
    assert cfg.brain.enabled and cfg.brain.steer and cfg.brain.backup
    assert not cfg.render.reflections
    # the flag lowers the MDN reference of the drive mapping (default 40 Hz)
    assert BrainLink(BrainLinkConfig(), start=False).gains.backward_ref == 40.0
    link = BrainLink(BrainLinkConfig(backup=True, gains={"backward_ref": 55.0}), start=False)
    assert link.gains.backward_ref == 20.0


# ------------------------------------------------------------------ A2 / A7 display
def test_display_scale_and_env_override(monkeypatch):
    monkeypatch.delenv("PERPETUALFLY_FLY_SCALE", raising=False)
    retina = (2.0, 1440, 900)  # this Mac: backing 2, 1440x900 pt
    assert auto_display_scale((960, 640), 0.67, 0.8, env_var="PERPETUALFLY_FLY_SCALE",
                              info=retina) == pytest.approx(2.0)
    # capped to fit the screen; never below 0.5
    assert auto_display_scale((1920, 1280), 0.67, 0.8, info=retina) == pytest.approx(1.005, abs=0.01)
    assert auto_display_scale((960, 640), info=(1.0, 1920, 1080)) == 1.0
    monkeypatch.setenv("PERPETUALFLY_FLY_SCALE", "1.5")
    assert auto_display_scale((960, 640), env_var="PERPETUALFLY_FLY_SCALE", info=retina) == 1.5


def test_compose_frame_scales_hud_rec_and_help():
    rgb = np.full((64 * 5, 96 * 5, 3), 128, np.uint8)
    plain = compose_frame(rgb, None, scale=2.0)
    assert plain.shape == (640, 960, 3)
    hud = compose_frame(rgb, ["t 1.00s  dist 13.0 mm"], scale=2.0)
    assert (hud[:60, :200] != plain[:60, :200]).any()  # HUD drawn top left
    rec = compose_frame(rgb, None, scale=2.0, rec="REC 1.0s")
    red = (rec[..., 2] > 200) & (rec[..., 0] < 80)
    assert red[:60, 480:].any() and not red[:, :480].any()  # red dot top right
    helped = compose_frame(rgb, None, scale=1.0, help_groups=KEY_GROUPS)
    assert helped.shape == (320, 480, 3)
    assert (helped[150:170, 200:280] != 128).any()  # panel fits (font shrinks)


def test_key_table_groups_and_action_keys():
    from perpetualfly.app import ACTION_KEY_MAP

    assert [g for g, _ in KEY_GROUPS] == ["hits", "obstacles / terrain", "brain (--brain)",
                                          "actions", "swatter (--swatter)", "view / run"]
    keys = [part.lower() for k, _ in KEY_TABLE for part in k.replace(" / ", " ").split()]
    assert {"[", "]", "i", "m", "?", "v", "shift+v"} <= set(keys)
    assert ACTION_KEYS <= set(keys) and set(ACTION_KEY_MAP) == ACTION_KEYS
    # every key is bound once (the action keys used to be reserved = free)
    dup = {k for k in keys if keys.count(k) > 1}
    assert not dup, dup


# ------------------------------------------------------------------ A3 reflections
def test_reflection_flag_on_renderer():
    from perpetualfly.rendering import FrameRenderer

    cfg = AppConfig()
    s, _ = _session()
    try:
        for on in (True, False):
            cfg.render.reflections = on
            fr = FrameRenderer(s.sim.model, cfg.render, cfg.camera)
            try:
                fr.update_scene(s.sim.data, s.sim.time, s.sim.thorax_position(), s.sim.heading())
                assert fr.renderer.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] == int(on)
            finally:
                fr.close()
    finally:
        s.close("test")
        s.sim.close()


# ------------------------------------------------------------------ A4 terrain
def test_step_difficulty_regenerates_chunks_ahead_only(tmp_path):
    s, _ = _session(tmp_path)
    t = s.terrain
    try:
        s.sim.step(2000)
        k = t.generator.chunk_index_at(t._fly_x())
        before = {spec.index: spec for spec in t.loaded_chunks()}
        assert s.step_difficulty(-1).startswith("[terrain] difficulty already flat")
        for _ in range(4):
            msg = s.step_difficulty(+1)
        assert "hard -> chaos" in msg and t.cfg.difficulty == "chaos"
        assert "already chaos" in s.step_difficulty(+1)
        ref = TerrainGenerator(t.cfg.seed, "chaos", cfg=t.cfg)
        after = {spec.index: spec for spec in t.loaded_chunks()}
        for idx, spec in after.items():
            if idx >= k + 2:
                assert spec.kind == ref.chunk(idx).kind and spec is not before[idx]
            else:
                assert spec is before[idx]  # nothing changes under / right before the fly
        assert t.chunk_kind_at(t.generator.chunk_bounds(k + 30)[0]) == ref.chunk(k + 30).kind
        with pytest.raises(ValueError):
            t.set_difficulty("impossible")
        s.sim.step(500)  # physics fine after the swap
    finally:
        s.close("test")
    rows = list(csv.DictReader(open(s.logger.run_dir / "events.csv")))
    ev = [json.loads(r["details"]) for r in rows if r["event_type"] == "terrain_difficulty"]
    assert [e["difficulty"] for e in ev] == ["easy", "normal", "hard", "chaos"]
    summary = json.loads((s.logger.run_dir / "summary.json").read_text())
    assert summary["terrain_difficulty"] == "chaos"
    assert summary["terrain_difficulty_start"] == "flat"


# ------------------------------------------------------------------ A8 auto hits
def test_auto_hits_wait_while_down_and_resume(tmp_path):
    # zero-force shoves: counted like hits but never knock the fly over
    auto = AutoPerturbConfig(enabled=True, first_hit_after_s=0.05, min_interval_s=0.1,
                             max_interval_s=0.1, magnitude_bw_range=(0.0, 0.0))
    s, msgs = _session(tmp_path, auto=auto, hit_mode="shove", auto_perturb_resume_after_s=0.3)
    try:
        s.sim.step(1500)  # start: no resume delay
        assert s.metrics.n_hits >= 1 and s.auto.n_skipped == 0
        n_hits = s.metrics.n_hits
        s.up_since = None  # as after a fall (the detector state itself is not faked)
        assert not s.auto_hits_allowed()
        s.sim.step(3000)
        assert s.metrics.n_hits == n_hits and s.auto.n_skipped >= 2
        assert any("hit skipped" in m for m in msgs)
        s.up_since = s.sim.time  # "recovered" now: waits resume_after_s
        s.sim.step(2000)
        assert s.metrics.n_hits == n_hits
        s.sim.step(3000)
        assert s.metrics.n_hits > n_hits
        s.detector.state = FallState.FALLEN
        assert not s.auto_hits_allowed()
        n_skipped = s.auto.n_skipped
    finally:
        s.close("test")
    summary = json.loads((s.logger.run_dir / "summary.json").read_text())
    assert summary["n_auto_hits_skipped"] == n_skipped
    rows = list(csv.DictReader(open(s.logger.run_dir / "events.csv")))
    assert sum(r["event_type"] == "auto_hit_skipped" for r in rows) == n_skipped


# ------------------------------------------------------------------ A9 distance
def test_walked_distance_excludes_flights():
    m = RunMetrics()
    t = 0.0
    for _ in range(101):  # walk 1 s at 10 mm/s
        m.update(t, np.array([10.0 * t, 0, 1.1]), walking=True)
        t += 0.01
    x0 = 10.0 * (t - 0.01)
    for i in range(1, 31):  # launched: 30 mm in 0.3 s, feet off the ground
        m.update(t, np.array([x0 + i, 0, 3.0]), walking=False)
        t += 0.01
    x1, t1 = x0 + 30, t
    for i in range(101):  # walk 1 s at 10 mm/s
        m.update(t, np.array([x1 + 10.0 * (t - t1), 0, 1.1]), walking=True)
        t += 0.01
    assert m.distance == pytest.approx(50.0, abs=0.5)
    assert 16.0 < m.walked_distance < 20.5  # flight + the segments it touched dropped
    assert m.walking_time == pytest.approx(2.0, abs=0.05)
    assert m.average_speed == pytest.approx(m.walked_distance / m.run_time)
    assert m.average_speed_total == pytest.approx(m.distance / m.run_time)
    s = m.summary()
    assert s["walked_distance_mm"] == m.walked_distance and s["distance_mm"] == m.distance
    assert s["average_speed_total_mm_s"] > s["average_speed_mm_s"]
    assert "walked" in m.summary_line()
    # a reset banks the walked distance
    w = m.walked_distance
    m.notify_reset(t, np.zeros(3))
    m.update(t + 0.5, np.array([5.0, 0, 1.1]), walking=True)
    assert m.walked_distance == pytest.approx(w + 5.0, abs=1e-6)


# ------------------------------------------------------------------ A5 / A6 media
def test_media_capture_screenshot_and_recording(tmp_path):
    mc = MediaCapture(None, tmp_path, fps=30.0)
    rgb = np.zeros((64, 96, 3), np.uint8)
    rgb[..., 0] = 200
    paths = mc.save_screenshot(rgb, 1.25, hud_bgr=np.zeros((64, 96, 3), np.uint8))
    assert [p.name for p in paths] == ["shot001_t00001.25s_fly.png",
                                       "shot001_t00001.25s_fly_hud.png"]
    assert all(p.parent == tmp_path / "screenshots" for p in paths)
    assert cv2.imread(str(paths[0]))[0, 0, 2] == 200  # RGB -> BGR on disk
    assert mc.due(0.0) == 0 and mc.rec_label() is None
    mc.start_recording(2.0)
    assert mc.add(rgb, 2.0) == 1 and mc.add(rgb, 2.01) == 0  # sim-time paced
    assert mc.add(rgb, 2.1) == 3  # frames at 1/30, 2/30, 3/30 s are due now
    for i in range(4, 31):
        mc.add(rgb, 2.0 + i / 30.0)
    assert mc.rec_frames == 31 and mc.rec_label().startswith("REC")
    msg = mc.stop_recording()
    assert "31 frames" in msg and mc.rec_path.parent == tmp_path / "recordings"
    import imageio.v2 as iio

    r = iio.get_reader(mc.rec_path)
    assert r.count_frames() == 31 and r.get_meta_data()["fps"] == 30.0
    r.close()


def test_headless_script_keys_screenshot_record_difficulty(tmp_path, capsys):
    cfg = AppConfig()
    cfg.terrain.difficulty = "easy"
    cfg.logging.runs_dir = str(tmp_path)
    cfg.render.width, cfg.render.height = 320, 208
    res = run(cfg, headless=True, max_seconds=1.2,
              script_keys=[(0.3, "i"), (0.4, "]"), (0.5, "m"), (0.9, "m"), (1.0, "?")])
    out = capsys.readouterr().out
    run_dir = Path(res.run_dir)
    assert "[screenshot]" in out and "[terrain] difficulty easy -> normal" in out
    assert "Keys (window focused):" in out and "[record] stopped" in out
    shots = sorted(p.name for p in run_dir.glob("shot*.png"))
    assert len(shots) == 2 and shots[0].startswith("shot001_t00000.3")
    assert shots[0].endswith("_fly.png") and shots[1].endswith("_fly_hud.png")
    img = cv2.imread(str(run_dir / shots[0]))
    assert img.shape == (208, 320, 3) and img.std() > 5  # a real rendered frame
    vids = list(run_dir.glob("recording*.mp4"))
    assert len(vids) == 1
    import imageio.v2 as iio

    r = iio.get_reader(vids[0])
    assert 11 <= r.count_frames() <= 13  # 0.4 s of sim time at 30 fps
    r.close()
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["terrain_difficulty"] == "normal" and len(summary["media"]) == 3
    assert "walked_distance_mm" in summary["metrics"]
    assert res.walked_distance > 5.0 and res.n_auto_hits_skipped == 0
    kinds = [r["event_type"] for r in csv.DictReader(open(run_dir / "events.csv"))]
    assert {"screenshot", "record_start", "record_stop", "terrain_difficulty"} <= set(kinds)


def test_screenshot_includes_brain_frame(tmp_path, capsys):
    pytest.importorskip("numba")
    cfg = AppConfig()
    cfg.terrain.difficulty = "flat"
    cfg.logging.enabled = False
    cfg.logging.runs_dir = str(tmp_path)
    cfg.render.width, cfg.render.height = 320, 208
    cfg.brain = BrainLinkConfig(enabled=True, window=False,
                                synthetic={"n": 50, "p_conn": 0.1, "seed": 0})
    run(cfg, headless=True, max_seconds=1.0, script_keys=[(0.9, "i")])
    out = capsys.readouterr().out
    brain = list((tmp_path / "screenshots").glob("*_brain.png"))
    assert len(brain) == 1, out
    img = cv2.imread(str(brain[0]))
    assert img.shape == (800, 1280, 3)


# ------------------------------------------------------------------ A10 packaging
def test_package_data_declares_brain_atlas():
    pp = tomllib.loads((ROOT / "pyproject.toml").read_text())
    pats = pp["tool"]["setuptools"]["package-data"]["perpetualfly.brain_viz"]
    assets = ROOT / "perpetualfly" / "brain_viz"
    assert any(assets.glob(p) for p in pats)
    assert list((assets / "assets").glob("*.json"))
