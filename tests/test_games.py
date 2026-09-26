"""FLY BRAIN PLAYS / ASTEROID DODGE (docs/GAMES.md). Fast: no connectome data needed
(synthetic brain or no brain)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from perpetualfly.brain.schema import StimulusEvent
from perpetualfly.games import (
    HONEST_LABEL,
    AsteroidConfig,
    GameBrain,
    GameMapping,
    asteroid_response,
    wave_params,
)
from perpetualfly.games.experiment import format_summary, make_specs, summarize
from perpetualfly.games.runner import HighScores, parse_script_keys
from perpetualfly.vision.looming import loom_response


# ----------------------------------------------------------------- pure logic
def test_honest_label_mentions_both_parts():
    assert "real connectome wiring" in HONEST_LABEL
    assert "designed by us" in HONEST_LABEL


def test_waves_get_harder_and_difficulty_scales():
    cfg = AsteroidConfig()
    w1, w4 = wave_params(cfg, 1), wave_params(cfg, 4)
    assert w4["speed"] > w1["speed"] and w4["interval"] < w1["interval"]
    assert w4["count"] > w1["count"]
    easy = wave_params(AsteroidConfig(difficulty="easy"), 1)
    hard = wave_params(AsteroidConfig(difficulty="hard"), 1)
    assert hard["speed"] > w1["speed"] > easy["speed"]
    assert hard["interval"] < easy["interval"]


def test_asteroid_response_silent_when_small_or_slow():
    p = asteroid_response()
    assert loom_response(5.0, 100.0, p) == (0.0, 0.0)  # too small for LC4, no LPLC2
    assert loom_response(20.0, 5.0, p) == (0.0, 0.0)  # not expanding
    assert loom_response(20.0, -50.0, p) == (0.0, 0.0)  # receding
    lc4, lplc2 = loom_response(30.0, 150.0, p)
    assert lc4 > 100 and lplc2 > 50


def test_game_mapping_directions():
    m = GameMapping()
    assert np.allclose(m.drive({}), [1.0, 1.0])  # quiet brain walks straight
    left = m.drive({"turn_L": 40.0})
    right = m.drive({"turn_R": 40.0})
    # the controller turns toward the smaller amplitude: turn_L -> left < right
    assert left[0] < left[1] and right[0] > right[1]
    assert np.allclose(left, right[::-1])
    assert m.drive({"walk_L": 60, "walk_R": 60}).mean() > 1.1
    assert m.drive({"backward_L": 40, "backward_R": 40}).max() < 0  # MDN backs up


def test_gamebrain_side_mapping_without_process():
    ev = StimulusEvent("loom", "left", 0.5, 0.06, 1.0, {"lc4_hz": 80.0, "lplc2_hz": 20.0})
    b = GameBrain("brain")
    b.on_loom(ev)
    assert b.sent[-1].side == "left"
    m = GameBrain("mirror")
    m.on_loom(ev)
    assert m.sent[-1].side == "right" and m.sent[-1].details["eye"] == "left"
    assert m.eye_drive(1.01)["left"] == (80.0, 20.0)  # the HUD shows the real eye
    n = GameBrain("none")
    n.on_loom(ev)
    assert n.sent == [] and n.eye_drive(1.01)["left"] == (80.0, 20.0)
    assert not n.update(1.1) and np.allclose(n.drive, 1.0)
    with pytest.raises(ValueError):
        GameBrain("shuffled")


def test_highscores_roundtrip(tmp_path):
    p = tmp_path / "hs.json"
    hs = HighScores(p)
    assert hs.best("asteroids", "brain", "normal") == 0
    assert hs.add("asteroids", "brain", "normal", {"score": 50}) == 1
    assert hs.add("asteroids", "brain", "normal", {"score": 20}) == 2
    assert hs.add("asteroids", "brain", "normal", {"score": 90}) == 1
    for s in range(10):
        hs.add("asteroids", "brain", "normal", {"score": s})
    hs2 = HighScores(p)
    lst = hs2.data["asteroids"]["brain:normal"]
    assert len(lst) == 5 and hs2.best("asteroids", "brain", "normal") == 90
    assert p.stat().st_size < 4000


def test_script_keys():
    assert parse_script_keys("4:r, 2:space") == [(2.0, "space"), (4.0, "r")]
    assert parse_script_keys(None) == []


def test_experiment_summary_counts():
    rows = []
    for i in range(6):
        rows.append(dict(trial=i, control="brain", hit=False, turn_away_deg=30.0,
                         lateral_away_mm=3.0, min_dist_mm=4.0, jumped=False,
                         turn_away_dn_hz_s=2.0, loom_peak_left=150.0 if i % 2 else 10.0,
                         loom_peak_right=10.0 if i % 2 else 150.0,
                         heading_change_deg=-30.0 if i % 2 else 30.0))
        rows.append(dict(trial=i, control="none", hit=True, turn_away_deg=0.0,
                         lateral_away_mm=0.5, min_dist_mm=2.0, jumped=False,
                         turn_away_dn_hz_s=0.0, loom_peak_left=150.0, loom_peak_right=150.0,
                         heading_change_deg=0.0))
    s = summarize(rows)
    assert s["conditions"]["brain"]["dodge_rate"] == 1.0
    assert s["conditions"]["none"]["dodge_rate"] == 0.0
    assert s["conditions"]["brain"]["frac_away_from_seen"] == 1.0
    assert s["paired"]["brain_vs_none"]["dodged_only_brain"] == 6
    assert s["paired"]["brain_vs_none"]["mcnemar_p"] == pytest.approx(0.0312, abs=1e-3)
    assert "brain vs none" in format_summary(s)
    specs = make_specs(6, seed=3)
    assert [sp.offset_mm > 0 for sp in specs] == [True, False] * 3  # balanced sides
    assert make_specs(6, seed=3)[2] == specs[2]  # reproducible


# ----------------------------------------------------------------- MuJoCo
@pytest.fixture(scope="module")
def session():
    from perpetualfly.games.session import AsteroidSession

    cfg = AsteroidConfig(spawn_dist_mm=9.0, lives=2, ready_s=0.2)
    s = AsteroidSession(GameBrain("none"), cfg, seed=0)
    yield s
    s.close()


def _run_rock(s, offset, speed=14.0, timeout=2.5):
    g = s.game
    g.state = "trial"
    rk = g.spawn_rock(offset_mm=offset, speed=speed, bearing_deg=0.0, slot=1)
    t0 = g.time()
    while rk.outcome is None and g.time() - t0 < timeout:
        s.step()
    return rk


def test_rock_hits_and_misses_physically(session):
    s = session
    s.restart()
    for _ in range(40):
        s.step()
    lives = s.game.lives
    rk = _run_rock(s, 0.0)
    assert rk.outcome == "hit" and rk.min_dist < rk.radius + 1.5
    assert s.game.lives == lives - 1 and s.game.hits == 1
    # the rock then ghosts (no more contact) and rolls on
    for _ in range(20):
        s.step()
    assert rk.ghost or not rk.active
    rk2 = _run_rock(s, 6.0)
    assert rk2.outcome == "dodged" and s.game.dodges == 1 and s.game.score > 100


def test_vision_sees_the_rock_on_its_side(session):
    s = session
    s.restart()
    for _ in range(40):
        s.step()
    n0 = len(s.vision.sent)
    rk = _run_rock(s, 2.5)  # passes on the fly's left
    evs = s.vision.sent[n0:]
    assert evs, "no looming events"
    peak = {e: max([ev.details["lc4_hz"] for ev in evs if ev.side == e] or [0.0])
            for e in ("left", "right")}
    assert peak["left"] > 50 and peak["left"] >= peak["right"]
    assert rk.outcome in ("hit", "dodged")


def test_gameover_and_restart(session):
    s = session
    s.restart()
    s.game.lives = 1
    for _ in range(40):
        s.step()
    _run_rock(s, 0.0)
    assert s.game.state == "gameover" and s.game.t_end is not None
    kinds = [e.kind for e in s.game.events]
    assert "gameover" in kinds
    s.restart("hard")
    assert s.game.state == "ready" and s.game.lives == s.cfg.lives
    assert s.game.cfg.difficulty == "hard" and s.game.score == 0
    assert all(not rk.active for rk in s.field.rocks)
    s.game.cfg.difficulty = "normal"


def test_waves_spawn_rocks(session):
    s = session
    s.restart()
    t0 = s.game.time()
    while s.game.time() - t0 < 1.0:
        s.step()
    assert s.game.state in ("playing", "wave_break", "gameover")
    assert s.game.spawned >= 1


def test_hud_frame(session, tmp_path):
    from perpetualfly.games.hud import compose
    from perpetualfly.games.session import make_renderer, render_frame

    r = make_renderer(session.sim, 320, 240)
    try:
        img = compose(render_frame(r, session.sim), session)
    finally:
        r.close()
    assert img.shape == (240, 320 + 150, 3) and img.dtype == np.uint8
    assert img[:, 320:].mean() > 5  # the brain panel is drawn


# ----------------------------------------------------------------- synthetic brain
def test_play_script_synthetic_headless(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "play.py"
    spec = importlib.util.spec_from_file_location("play_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hs = tmp_path / "hs.json"
    frames = tmp_path / "frames"
    rc = mod.main(["--game", "asteroids", "--synthetic-brain", "--max-seconds", "1.2",
                   "--highscores", str(hs), "--frames", str(frames), "--frame-times", "1.0",
                   "--width", "320", "--height", "240", "--lives", "1"])
    assert rc == 0
    pngs = list(frames.glob("*.png"))
    assert len(pngs) == 1


def test_synthetic_brain_paces_and_stops():
    from perpetualfly.games.session import AsteroidSession

    b = GameBrain("brain", synthetic={"n": 60, "p_conn": 0.1, "seed": 0}).start()
    try:
        s = AsteroidSession(b, AsteroidConfig(ready_s=0.1), seed=0)
        t0 = s.game.time()
        while s.game.time() - t0 < 0.3:
            s.step()
        assert b.n_states >= 5  # 20 ms windows
        assert b.lag(s.game.time()) is not None and b.lag(s.game.time()) < 0.3
        proc = b.brain._proc
    finally:
        b.close()
    assert b.brain is None and not proc.is_alive()
