"""FLY BRAIN PLAYS / ASTEROID DODGE (docs/GAMES.md). Fast: no connectome data needed
(synthetic brain or no brain)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from fly_simulator.brain.schema import StimulusEvent
from fly_simulator.games import (
    HONEST_LABEL,
    AsteroidConfig,
    GameBrain,
    GameMapping,
    asteroid_response,
    wave_params,
)
from fly_simulator.games.experiment import format_summary, make_specs, summarize
from fly_simulator.games.runner import HighScores, parse_script_keys
from fly_simulator.vision.looming import loom_response


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
    from fly_simulator.games.session import AsteroidSession

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
    from fly_simulator.games.hud import compose
    from fly_simulator.games.session import make_renderer, render_frame

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
    from fly_simulator.games.session import AsteroidSession

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


# =================================================================== game 2: CHASE
from fly_simulator.games import (  # noqa: E402
    ChaseConfig,
    PursuitResponse,
    level_params,
    pursuit_response,
)
from fly_simulator.games.chase_experiment import (  # noqa: E402
    format_chase_summary,
    make_chase_specs,
    summarize_chase,
)


def test_pursuit_response_is_a_lateral_error_signal():
    p = PursuitResponse()
    assert pursuit_response(5.0, 0.0, 1.0, p) == 0.0  # straight ahead: no drive
    assert pursuit_response(5.0, -10.0, 1.0, p) == 0.0  # on the other eye's side
    assert pursuit_response(0.5, 20.0, 1.0, p) == 0.0  # sub-degree speck
    assert pursuit_response(5.0, 20.0, 0.0, p) == 0.0  # outside the field of view
    r10, r20, r40 = (pursuit_response(5.0, a, 1.0, p) for a in (10.0, 20.0, 40.0))
    assert 0 < r10 < r20 < r40 == pytest.approx(p.max_hz)
    assert pursuit_response(70.0, 40.0, 1.0, p) == pytest.approx(0.5 * p.max_hz)  # big: weaker


def test_chase_levels_and_difficulty():
    cfg = ChaseConfig()
    l1, l3 = level_params(cfg, 1), level_params(cfg, 3)
    assert l3["speed"] > l1["speed"] and l3["weave_dps"] > l1["weave_dps"]
    easy = level_params(ChaseConfig(difficulty="easy"), 1)
    hard = level_params(ChaseConfig(difficulty="hard"), 1)
    assert hard["speed"] > l1["speed"] > easy["speed"]
    assert hard["weave_dps"] > easy["weave_dps"]


def test_gamebrain_pursuit_side_mapping():
    ev = StimulusEvent("manual", "left", 0.5, 0.04, 2.0, {"set": "LC10a", "rate_hz": 90.0})
    b = GameBrain("brain")
    b.on_pursuit(ev)
    assert b.sent[-1].side == "left" and b.sent[-1].details["set"] == "LC10a"
    assert b.pursuit_drive(2.01) == {"left": 90.0, "right": 0.0}
    assert b.pursuit_drive(2.5)["left"] == 0.0  # the event ran out
    m = GameBrain("mirror")
    m.on_pursuit(ev)
    assert m.sent[-1].side == "right" and m.sent[-1].details["eye"] == "left"
    n = GameBrain("none")
    n.on_pursuit(ev)
    assert n.sent == [] and n.pursuit_drive(2.01)["left"] == 90.0
    b.reset(3.0)
    assert b.pursuit_drive(3.0) == {"left": 0.0, "right": 0.0}


def test_chase_experiment_summary():
    rows = []
    for i in range(6):
        rows.append(dict(trial=i, control="brain", follow_frac=0.9, mean_dist_mm=5.0,
                         mean_abs_err_deg=20.0, catches=2, lost=False,
                         initial_turn_toward_deg=30.0))
        rows.append(dict(trial=i, control="none", follow_frac=0.4, mean_dist_mm=12.0,
                         mean_abs_err_deg=80.0, catches=0, lost=True,
                         initial_turn_toward_deg=-1.0 if i % 2 else 1.0))
    s = summarize_chase(rows)
    assert s["conditions"]["brain"]["follow_frac"] == 0.9
    assert s["conditions"]["brain"]["initial_turn_toward"] == "6/6"
    assert s["conditions"]["none"]["lost"] == 6
    p = s["paired"]["brain_vs_none"]
    assert p["first_follows_more"] == "6/6" and p["sign_test_p"] == pytest.approx(0.0312, abs=1e-3)
    assert p["lost_only_none"] == 6 and p["mcnemar_lost_p"] == pytest.approx(0.0312, abs=1e-3)
    assert "brain vs none" in format_chase_summary(s)
    specs = make_chase_specs(6, seed=2)
    assert [sp.bearing_deg > 0 for sp in specs] == [True, False] * 3
    assert make_chase_specs(6, seed=2)[3] == specs[3]


def test_lc10a_named_set_real_data():
    from fly_simulator.brain.data import data_available

    if not data_available():
        pytest.skip("data/brain not downloaded")
    from fly_simulator.brain.data import load_neuron_table
    from fly_simulator.brain.mapping import named_sets

    t = load_neuron_table()
    idx = named_sets(t)["LC10a"]
    side = t.col("side")[idx]
    assert (side == "left").sum() == 115 and (side == "right").sum() == 119


@pytest.fixture(scope="module")
def chase():
    from fly_simulator.games.session import ChaseSession

    s = ChaseSession(GameBrain("none"), ChaseConfig(lives=2, ready_s=0.1), seed=0)
    yield s
    s.close()


def _steps(s, seconds):
    t0 = s.game.time()
    while s.game.time() - t0 < seconds:
        s.step()


def test_leader_on_the_left_drives_left_lc10a(chase):
    s = chase
    s.restart()
    _steps(s, 0.2)
    s.game.state = "trial"
    s.game.place_leader(8.0, 40.0)  # 40 deg to the fly's left
    s.leader.weave = False
    n0 = len(s.vision.sent)
    _steps(s, 0.1)
    evs = s.vision.sent[n0:]
    assert evs and all(e.kind == "manual" and e.details["set"] == "LC10a" for e in evs)
    assert {e.side for e in evs} == {"left"}
    assert s.vision.eye_state[0].rate_hz > 50 and s.vision.eye_state[1].rate_hz == 0
    assert s.game.error_deg > 20
    s.game.place_leader(8.0, -40.0)
    n0 = len(s.vision.sent)
    _steps(s, 0.1)
    assert {e.side for e in s.vision.sent[n0:]} == {"right"}


def test_catch_dash_lost_and_gameover(chase):
    s = chase
    s.restart()
    _steps(s, 0.2)
    g = s.game
    assert g.state == "playing"
    g.place_leader(2.0, 0.0)  # right in front: a catch
    _steps(s, 0.05)
    assert g.catches == 1 and g.score >= 100 and s.leader.dashing(s.sim.time)
    assert any(e.kind == "catch" for e in g.events)
    lives = g.lives
    g.place_leader(25.0, 180.0)  # far behind: lost after lose_hold_s
    _steps(s, g.cfg.lose_hold_s + 0.1)
    assert g.lives == lives - 1 and g.losses == 1
    assert g.dist < g.cfg.lose_dist_mm  # the leader was put back ahead
    g.place_leader(25.0, 180.0)
    _steps(s, g.cfg.lose_hold_s + 0.1)
    assert g.state == "gameover" and g.t_end is not None
    s.restart("hard")
    assert g.state == "ready" and g.lives == g.cfg.lives and g.score == 0
    assert g.cfg.difficulty == "hard" and s.leader.speed == level_params(g.cfg, 1)["speed"]
    g.cfg.difficulty = "normal"


def test_chase_hud_frame(chase, tmp_path):
    from fly_simulator.games.hud import compose
    from fly_simulator.games.session import make_renderer, render_frame

    chase.restart()
    _steps(chase, 0.3)
    r = make_renderer(chase.sim, 320, 240, **chase.camera)
    try:
        img = compose(render_frame(r, chase.sim), chase)
    finally:
        r.close()
    assert img.shape == (240, 320 + 150, 3) and img[:, 320:].mean() > 5


def test_play_script_chase_synthetic(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "play.py"
    spec = importlib.util.spec_from_file_location("play_script_chase", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hs = tmp_path / "hs.json"
    frames = tmp_path / "frames"
    rc = mod.main(["--game", "chase", "--synthetic-brain", "--max-seconds", "1.2",
                   "--highscores", str(hs), "--frames", str(frames), "--frame-times", "1.0",
                   "--width", "320", "--height", "240"])
    assert rc == 0
    assert len(list(frames.glob("chase_*.png"))) == 1


# ---------------------------------------------------------------------------
# game 3: FLY THROUGH RINGS (real flapping-wing flight)
# ---------------------------------------------------------------------------

from fly_simulator.games import (  # noqa: E402
    RINGS_DIFFICULTIES,
    RingCourse,
    RingsConfig,
    rings_level_params,
    turn_command,
)
from fly_simulator.games.rings_experiment import (  # noqa: E402
    format_rings_summary,
    make_ring_specs,
    summarize_rings,
)


def test_turn_command_directions():
    assert turn_command({}) == 0.0
    assert turn_command({"turn_L": 50.0}) > 0.9  # left DNa01/02 -> steer left
    assert turn_command({"turn_R": 50.0}) < -0.9
    assert abs(turn_command({"turn_L": 30.0, "turn_R": 30.0})) < 1e-12
    assert turn_command({"turn_L": -5.0}) == 0.0


def test_rings_levels_and_difficulty():
    c = RingsConfig()
    p1, p3 = rings_level_params(c, 1), rings_level_params(c, 3)
    assert p1["speed"] == RINGS_DIFFICULTIES["normal"].speed and p3["speed"] > p1["speed"]
    assert p3["max_shift_mm"] > p1["max_shift_mm"]
    e = rings_level_params(RingsConfig(difficulty="easy"), 1)
    h = rings_level_params(RingsConfig(difficulty="hard"), 1)
    assert e["speed"] < h["speed"] and e["radius_mm"] > h["radius_mm"]


def test_ring_course_layout_without_sim():
    course = RingCourse(RingsConfig(), seed=3)
    course.start(np.array([0.0, 0.0, 6.0]), 0.0, first_offset_mm=4.0)
    rs = course.rings
    assert len(rs) == course.cfg.n_rings
    assert np.allclose(rs[0].center, [course.cfg.first_dist_mm, 4.0, course.cfg.altitude_mm])
    xs = [r.center[0] for r in rs]
    assert np.allclose(np.diff(xs), course.cfg.spacing_mm)
    for a, b in zip(rs, rs[1:]):
        step = abs(b.center[1] - a.center[1])
        assert course.cfg.min_shift_mm - 1e-9 <= step <= course.max_shift + 1e-9
    s, lat, vert = RingCourse.plane_coords(rs[0], np.array([20.0, 1.0, 6.5]))
    assert s < 0 and lat < 0 and vert == pytest.approx(0.5)
    first = rs[0]
    done = course.advance("passed")
    assert done is first and done.status == "passed" and course.done == [first]
    assert len(course.rings) == course.cfg.n_rings and course.target_ring() is not first


def test_rings_experiment_summary():
    specs = make_ring_specs(4, seed=1)
    assert [s.offset_mm > 0 for s in specs] == [True, False, True, False]
    assert all(3.5 <= abs(s.offset_mm) <= 7.0 for s in specs)
    rows = []
    for i in range(4):
        rows.append({"trial": i, "control": "brain", "through": i < 3, "why": "crossed",
                     "radial_mm": 1.0 + i, "initial_turn_toward_deg": 10.0})
        rows.append({"trial": i, "control": "none", "through": False, "why": "crossed",
                     "radial_mm": 5.0, "initial_turn_toward_deg": 0.0})
    summ = summarize_rings(rows)
    assert summ["conditions"]["brain"]["through"] == 3
    assert summ["conditions"]["brain"]["initial_turn_toward"] == "4/4"
    p = summ["paired"]["brain_vs_none"]
    assert p["through_only_brain"] == 3 and p["through_only_none"] == 0
    assert p["first_closer"] == "4/4"
    assert "brain vs none" in format_rings_summary(summ)


@pytest.fixture(scope="module")
def rings():
    from fly_simulator.games.session import RingsSession

    s = RingsSession(GameBrain("none"), RingsConfig(lives=2, takeoff=False, ready_s=0.3),
                     seed=0)
    yield s
    s.close()


def _fly_to_ring(s, offset, timeout=2.0):
    """Straight flight (no brain) at one ring ``offset`` mm to the left."""
    g = s.game
    g.state = "trial"
    g.last_outcome = None
    g.start_course(first_offset_mm=offset, n=1)
    t0 = g.time()
    while g.last_outcome is None and g.time() - t0 < timeout:
        s.step()
    return g.last_outcome


def test_rings_flight_through_and_miss(rings):
    s = rings
    s.restart()
    _steps(s, 0.35)
    assert s.pilot.airborne and s.flight.state == "forward"
    assert abs(s.flight.altitude() - s.cfg.altitude_mm) < 0.6
    o = _fly_to_ring(s, 0.0)
    assert o is not None and o["through"] and o["radial_mm"] < 1.0
    o = _fly_to_ring(s, 6.0)
    assert o is not None and not o["through"] and o["lateral_mm"] < -4.0
    assert s.game.passed == 1 and s.game.missed == 1 and s.game.lives == s.cfg.lives  # trial


def test_ring_on_the_left_drives_left_lc10a(rings):
    s = rings
    s.restart()
    _steps(s, 0.35)
    g = s.game
    g.state = "trial"
    g.start_course(first_offset_mm=8.0, n=1)
    n0 = len(s.vision.sent)
    _steps(s, 0.05)
    evs = s.vision.sent[n0:]
    assert evs and all(e.kind == "manual" and e.details["set"] == "LC10a" for e in evs)
    assert {e.side for e in evs} == {"left"} and g.bearing_deg > 10
    g.start_course(first_offset_mm=-8.0, n=1)
    n0 = len(s.vision.sent)
    _steps(s, 0.05)
    assert {e.side for e in s.vision.sent[n0:]} == {"right"}


def test_heading_command_turns_the_flying_fly(rings):
    s = rings
    s.restart()
    _steps(s, 0.35)
    y0 = s.pilot.true_yaw()
    for _ in range(40):  # 0.2 s of full left command
        s.sim.step(s.chunk_steps)
        s.pilot.update(1.0, s.chunk_steps * s.sim.timestep)
    dy = math.degrees((s.pilot.true_yaw() - y0 + math.pi) % (2 * math.pi) - math.pi)
    assert 8.0 < dy < 35.0, dy  # 120 deg/s commanded, minus the low-pass / tracking lag
    assert s.pilot.airborne


def test_rings_miss_costs_lives_gameover_restart(rings):
    s = rings
    s.restart()
    g = s.game
    _steps(s, g.cfg.ready_s + 0.1)
    assert g.state == "playing" and g.course.target_ring() is not None
    for _ in range(2):  # force misses: the next ring far to the side
        g.course.rings[0].center[1] += 15.0
        g._begin_ring()
        t0 = g.time()
        n = g.missed
        while g.missed == n and g.time() - t0 < 2.0:
            s.step()
    assert g.missed == 2 and g.state == "gameover" and g.lives == 0
    assert any(e.kind == "miss" for e in g.events)
    s.restart("easy")
    assert g.state == "ready" and g.lives == g.cfg.lives and g.score == 0
    assert s.pilot.speed == RINGS_DIFFICULTIES["easy"].speed
    g.cfg.difficulty = "normal"
    s.restart()


def test_rings_hud_frame(rings):
    from fly_simulator.games.hud import compose
    from fly_simulator.games.session import make_renderer

    rings.restart()
    _steps(rings, rings.cfg.ready_s + 0.1)
    r = make_renderer(rings.sim, 320, 240, **rings.camera)
    try:
        img = compose(rings.render(r), rings)
    finally:
        r.close()
    assert img.shape == (240, 320 + 150, 3) and img[:, 320:].mean() > 5


def test_play_script_rings_synthetic(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "play.py"
    spec = importlib.util.spec_from_file_location("play_script_rings", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hs = tmp_path / "hs.json"
    frames = tmp_path / "frames"
    rc = mod.main(["--game", "rings", "--synthetic-brain", "--air-start", "--max-seconds", "1.0",
                   "--highscores", str(hs), "--frames", str(frames), "--frame-times", "0.9",
                   "--width", "320", "--height", "240"])
    assert rc == 0
    assert len(list(frames.glob("rings_*.png"))) == 1


# ---------------------------------------------------------------------------
# game 4: FLY PONG (the brain moves the paddle the fly stands on)
# ---------------------------------------------------------------------------

from fly_simulator.games import (  # noqa: E402
    PONG_DIFFICULTIES,
    PongConfig,
    PongPhysics,
    paddle_command,
)
from fly_simulator.games.pong import aim_velocity, fold_y  # noqa: E402
from fly_simulator.games.pong_experiment import (  # noqa: E402
    format_pong_summary,
    make_pong_specs,
    summarize_pong,
)


def _run_phys(ph, seconds, dt=1e-3):
    for _ in range(int(round(seconds / dt))):
        ph.step(dt)


def test_pong_fold_and_aim():
    assert fold_y(3.0, 8.0) == pytest.approx(3.0)
    assert fold_y(10.0, 8.0) == pytest.approx(6.0)  # off the +wall
    assert fold_y(-19.0, 8.0) == pytest.approx(3.0)  # off both walls
    for bounce in (0, 1, -1):  # an aimed ball arrives where it was aimed
        c = PongConfig()
        ph = PongPhysics(c)
        ph.ai_mode = "off"
        ph.ay = 20.0  # out of the way
        ph.serve_aimed(c.ai_x - c.ball_radius_mm, 2.0, -4.0, bounce)
        ph.py = 30.0  # the fly's paddle far away: the ball passes its face
        ph.pcmd = 0.0
        t = 0.0
        while ph.last_cross is None and t < 5.0:
            ph.step(1e-3)
            t += 1e-3
        assert ph.last_cross is not None
        assert ph.last_cross["ball_y"] == pytest.approx(-4.0, abs=0.05), bounce
        assert (sum(k == "wall" for k, _ in ph.events) > 0) == (bounce != 0)
    vx, vy = aim_velocity(0.0, 0.0, -10.0, 0.0, 20.0, 8.0)
    assert vx == pytest.approx(-20.0) and vy == pytest.approx(0.0)


def test_pong_paddle_hit_english_speedup_and_miss():
    c = PongConfig()
    ph = PongPhysics(c)
    ph.ai_mode = "off"
    ph.serve("fly", y0=2.0, angle_deg=0.0)
    ph.py = 0.0
    s0 = ph.speed
    _run_phys(ph, 1.0)
    hits = [i for k, i in ph.events if k == "hit"]
    assert hits and hits[0]["side"] == "fly" and hits[0]["offset"] == pytest.approx(2.0)
    assert ph.vx > 0 and ph.vy > 0  # hit above the paddle centre: goes left (+y)
    assert ph.speed == pytest.approx(s0 * c.speedup)
    ph2 = PongPhysics(c)
    ph2.ai_mode = "off"
    ph2.serve("fly", y0=7.0, angle_deg=0.0)  # arrives 7 mm to the left of the paddle
    _run_phys(ph2, 1.5)
    assert ph2.events[-1][0] == "miss"
    assert ph2.events[-1][1]["side"] == "fly" and not ph2.in_play


def test_pong_paddle_velocity_and_limits():
    c = PongConfig()
    ph = PongPhysics(c)
    ph.pcmd = 20.0
    moved = sum(ph.step(1e-3) for _ in range(300))
    assert 3.0 < moved < 6.0 and ph.py == pytest.approx(moved)  # 0.3 s, 60 ms low-pass
    _run_phys(ph, 2.0)
    assert ph.py == pytest.approx(c.court_half_w - 0.5 * c.diff.paddle_len_mm)  # at the wall
    # DNa01/02 -> paddle: turn_L -> +y (the fly's left), turn_R -> -y
    assert paddle_command({"turn_L": 50.0}, c) > 0 > paddle_command({"turn_R": 50.0}, c)
    assert paddle_command({}, c) == 0.0
    assert abs(paddle_command({"turn_L": 500.0}, c)) <= c.paddle_vmax


def test_pong_ai_returns_slow_balls_and_wall_mode_aims():
    c = PongConfig()
    ph = PongPhysics(c, seed=3)
    ph.serve("ai", y0=0.0, angle_deg=5.0)
    ph._ai_err = ph._ai_aim = 0.0  # no aiming error: it reaches a slow, straight ball
    _run_phys(ph, 2.0)
    assert any(k == "hit" and i["side"] == "ai" for k, i in ph.events)
    ph = PongPhysics(c)
    ph.ai_mode = "wall"
    ph.aim_fn = lambda k: (5.0, 0)
    ph.py = -30.0
    ph.serve("ai", y0=-8.0, angle_deg=-30.0)
    _run_phys(ph, 3.0)
    assert ph.ai_hits == 1 and ph.last_cross["ball_y"] == pytest.approx(5.0, abs=0.05)
    assert PONG_DIFFICULTIES["hard"].paddle_len_mm < PONG_DIFFICULTIES["easy"].paddle_len_mm


def test_pong_experiment_summary_and_specs():
    rows = []
    for i in range(6):
        rows.append(dict(trial=i, control="brain", first_return=True, returns=5, balls_faced=5,
                         missed=False, first_paddle_toward_mm=3.0, mean_abs_offset_mm=1.0))
        rows.append(dict(trial=i, control="none", first_return=i < 2, returns=1 if i < 2 else 0,
                         balls_faced=2 if i < 2 else 1, missed=True, first_paddle_toward_mm=0.0,
                         mean_abs_offset_mm=4.0))
    s = summarize_pong(rows)
    b, n = s["conditions"]["brain"], s["conditions"]["none"]
    assert b["first_return_rate"] == 1.0 and b["return_rate"] == 1.0 and b["full_rallies"] == 6
    assert n["returns"] == 2 and n["balls_faced"] == 8 and n["paddle_toward"] == "0/0"
    p = s["paired"]["brain_vs_none"]
    assert p["first_only_brain"] == 4 and p["first_only_none"] == 0
    assert p["mcnemar_p"] == pytest.approx(0.125, abs=1e-3)
    assert p["first_more_returns"] == "6/6"
    assert "brain vs none" in format_pong_summary(s)
    specs = make_pong_specs(6, seed=2, n_balls=4)
    assert [sp.targets[0][0] > 0 for sp in specs] == [True, False] * 3
    assert all(len(sp.targets) == 4 and 1.0 <= abs(sp.targets[0][0]) <= 7.5 for sp in specs)
    assert make_pong_specs(6, seed=2, n_balls=4)[3] == specs[3]


@pytest.fixture(scope="module")
def pong():
    from fly_simulator.games.session import PongSession

    s = PongSession(GameBrain("none"), PongConfig(win_points=2, ready_s=0.2, point_pause_s=0.2),
                    seed=0)
    yield s
    s.close()


def test_pong_sled_carries_the_standing_fly(pong):
    s = pong
    s.restart()
    s.game.state = "trial"
    s.court.phys.hide()
    y0 = float(s.sim.thorax_position()[1])
    s.court.phys.pcmd = 15.0
    for _ in range(60):  # 0.3 s; the session's step would reset pcmd (no brain)
        s.sim.step(s.chunk_steps)
    dy = float(s.sim.thorax_position()[1]) - y0
    assert dy == pytest.approx(s.court.phys.py, abs=0.1) and dy > 2.5
    assert s.sim.tilt_deg() < 15 and 0.8 < s.sim.thorax_position()[2] < 1.6  # still standing
    s.restart()
    assert s.court.phys.py == 0.0 and abs(s.sim.thorax_position()[1]) < 0.2


def test_pong_ball_on_the_left_drives_left_lc10a(pong):
    s = pong
    s.restart()
    g, ph, c = s.game, s.court.phys, s.cfg
    g.state = "trial"
    ph.ai_mode = "off"
    ph.serve_aimed(c.ai_x - 8.0, 6.0, 6.0)  # a ball well to the fly's left
    n0 = len(s.vision.sent)
    _steps(s, 0.1)
    evs = s.vision.sent[n0:]
    assert evs and all(e.kind == "manual" and e.details["set"] == "LC10a" for e in evs)
    assert {e.side for e in evs} == {"left"}
    ph.serve_aimed(c.ai_x - 8.0, -6.0, -6.0)
    n0 = len(s.vision.sent)
    _steps(s, 0.1)
    assert {e.side for e in s.vision.sent[n0:]} == {"right"}
    ph.hide()
    n0 = len(s.vision.sent)
    _steps(s, 0.05)
    assert len(s.vision.sent) == n0  # no ball, no drive


def test_pong_points_gameover_restart(pong):
    s = pong
    s.restart()
    g = s.game
    _steps(s, g.cfg.ready_s + 0.05)
    assert g.state == "playing" and s.court.phys.in_play
    for _ in range(2):  # the paddle (no brain) stays put: force misses on the far side
        while g.state == "playing":
            ph = s.court.phys
            if ph.vx < 0 and ph.in_play:
                ph.by, ph.vy = 8.0, 0.0
            s.step()
        _steps(s, g.cfg.point_pause_s + 0.05)
    assert g.ai_points == 2 and g.state == "gameover" and g.winner == "ai"
    assert any(e.kind == "point_ai" for e in g.events)
    assert g.score_entry()["ai"] == 2
    s.restart("hard")
    assert g.state == "ready" and g.fly_points == g.ai_points == 0 and g.score == 0
    gid = s.court._bar_gid
    assert s.sim.model.geom_size[gid, 1] == pytest.approx(0.5 * PONG_DIFFICULTIES["hard"].paddle_len_mm)
    s.restart("normal")


def test_pong_hud_frame(pong):
    from fly_simulator.games.hud import compose
    from fly_simulator.games.session import make_renderer

    pong.restart()
    _steps(pong, pong.cfg.ready_s + 0.2)
    r = make_renderer(pong.sim, 320, 240, **pong.camera)
    try:
        img = compose(pong.render(r), pong)
        img2 = compose(pong.render(r), pong, panel=False)
    finally:
        r.close()
    assert img.shape == (240, 320 + 150, 3) and img[:, 320:].mean() > 5
    assert img2.shape == (240, 320, 3)


def test_pong_synthetic_brain_moves_the_paddle():
    from fly_simulator.games.session import PongSession

    b = GameBrain("brain", synthetic={"n": 60, "p_conn": 0.1, "seed": 0}).start()
    try:
        s = PongSession(b, PongConfig(ready_s=0.05), seed=0)
        try:
            _steps(s, 0.4)
            assert b.n_states >= 5
            assert abs(s.court.phys.pcmd) <= s.cfg.paddle_vmax
            assert abs(float(s.sim.thorax_position()[1]) - s.court.phys.py) < 0.2
        finally:
            s.close()
    finally:
        b.close()


def test_play_script_pong_synthetic(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "play.py"
    spec = importlib.util.spec_from_file_location("play_script_pong", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    hs = tmp_path / "hs.json"
    frames = tmp_path / "frames"
    rc = mod.main(["--game", "pong", "--synthetic-brain", "--max-seconds", "1.2",
                   "--highscores", str(hs), "--frames", str(frames), "--frame-times", "1.0",
                   "--width", "320", "--height", "240"])
    assert rc == 0
    assert len(list(frames.glob("pong_*.png"))) == 1
