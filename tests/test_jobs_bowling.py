"""The bowling job (fly_simulator/jobs/bowling.py): ten-pin scoring (strikes, spares,
the 10th frame), the scene (contacts, pins topple when hit), and the pinsetter
(sweep clears knocked pins, standing pins stay, a full rack is reset)."""

from __future__ import annotations

import numpy as np
import pytest

from fly_simulator.config import AppConfig
from fly_simulator.jobs import available_jobs, create_job_session, make_job
from fly_simulator.jobs.bowling import N_PINS, BowlingGame, BowlingJob, score_rolls
from fly_simulator.jobs.geometry import PROP_BIT
from fly_simulator.terrain import FLY_BIT, TERRAIN_BIT


# ---------------------------------------------------------------- scoring
@pytest.mark.parametrize("rolls, score", [
    ([0] * 20, 0),
    ([1] * 20, 20),
    ([10] * 12, 300),  # perfect game
    ([5] * 21, 150),  # all spares, 5 bonus
    ([9, 0] * 10, 90),
    ([10, 3, 4] + [0] * 16, 24),  # strike then 7: 17 + 7
    ([5, 5, 3, 0] + [0] * 16, 16),  # spare then 3: 13 + 3
    ([0] * 18 + [10, 10, 10], 30),  # 10th: X X X
    ([0] * 18 + [7, 3, 10], 20),  # 10th: spare + strike bonus
    ([0] * 18 + [10, 7, 3], 20),  # 10th: X then 7/
    ([0] * 18 + [10, 7, 2], 19),
    ([10, 7, 3, 9, 0, 10, 0, 8, 8, 2, 0, 6, 10, 10, 10, 8, 1], 167),  # classic example
])
def test_scoring(rolls, score):
    assert score_rolls(rolls) == score


def test_game_frames_marks_and_rack():
    g = BowlingGame()
    ev = g.roll(10)
    assert ev["strike"] and ev["frame_done"] and ev["reset_rack"] and g.frame == 1
    assert g.totals()[0] is None  # strike bonus pending
    ev = g.roll(7)
    assert not ev["frame_done"] and g.standing == 3 and g.ball_in_frame == 1
    ev = g.roll(3)
    assert ev["spare"] and ev["reset_rack"] and g.standing == 10
    assert g.totals()[:2] == [20, None]
    ev = g.roll(0)
    assert g.marks[2] == ["-"] and g.totals()[1] == 30
    g.roll(12)  # clipped to the pins standing
    assert g.frames[2] == [0, 10] and g.marks[2] == ["-", "/"]
    assert g.marks[0] == ["X"] and g.marks[1] == ["7", "/"]


def test_tenth_frame_bonus_and_game_over():
    g = BowlingGame()
    for _ in range(18):
        g.roll(0)
    assert g.frame == 9
    ev = g.roll(10)
    assert ev["strike"] and ev["reset_rack"] and not g.over
    ev = g.roll(4)
    assert not ev["reset_rack"] and g.standing == 6 and not g.over
    ev = g.roll(6)  # X 4 /
    assert ev["spare"] and ev["game_over"] and g.over
    assert g.marks[9] == ["X", "4", "/"] and g.final_score() == 20
    with pytest.raises(ValueError):
        g.roll(1)
    g2 = BowlingGame()
    for _ in range(18):
        g2.roll(0)
    g2.roll(3)
    ev = g2.roll(4)  # open 10th: no bonus ball
    assert ev["game_over"] and g2.final_score() == 7


# ---------------------------------------------------------------- scene
def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.logging.enabled = False
    cfg.render.width, cfg.render.height = 160, 106
    return cfg


def test_registry_and_layout():
    assert "bowling" in available_jobs()
    job = make_job("bowling", {"skill": 0.9})
    assert isinstance(job, BowlingJob) and job.cfg.skill == 0.9
    spots = job.pin_spots()
    assert spots.shape == (N_PINS, 2)
    assert spots[0, 0] == pytest.approx(job.head_x) and spots[0, 1] == 0.0
    # standard triangle: 4 rows, pin 7 on the left (+y), pin 10 on the right
    assert spots[6, 1] > 0 > spots[9, 1] and spots[6, 0] == pytest.approx(spots[9, 0])
    d = np.linalg.norm(spots[1] - spots[0])
    assert d == pytest.approx(job.cfg.pin_spacing)
    # the approach is a plateau; the ramp drops onto the lane at the foul line
    assert job.ground_height(0.0, 0.0) == pytest.approx(job.approach_z)
    assert job.ground_height(job.cfg.foul_x, 0.0) == pytest.approx(job.cfg.bed_height)
    assert job.ground_height(job.head_x, 0.0) == job.cfg.bed_height
    assert job.stop_x < job.ramp_x0 < job.cfg.foul_x


@pytest.fixture(scope="module")
def bowling():
    msgs: list[str] = []
    session, job = create_job_session("bowling", _cfg(), say=msgs.append)
    yield session, job, msgs
    session.sim.close()


def test_props_and_contacts(bowling):
    session, job, _ = bowling
    m = session.sim.model
    assert session.job is job
    pin = m.geom("bowl/pin0_belly").id
    assert m.geom_contype[pin] == PROP_BIT and not (m.geom_conaffinity[pin] & FLY_BIT)
    assert m.geom_conaffinity[pin] & TERRAIN_BIT and m.geom_conaffinity[pin] & PROP_BIT
    ball = m.geom("bowl/ball_geom").id
    assert m.geom_condim[ball] == 3 and m.geom_condim[m.geom("bowl/lane_oil").id] == 6
    # the fly touches the ball only through the low-friction head / body pairs
    pairs = {(int(m.pair_geom1[i]), int(m.pair_geom2[i])) for i in range(m.npair)}
    assert any(ball in p for p in pairs)
    # visual meshes are massless and don't collide
    for g in ("bowl/pin0_vis", "bowl/ball_vis", "bowl/lane_top", "bowl/masking"):
        gid = m.geom(g).id
        assert m.geom_contype[gid] == 0 and m.geom_conaffinity[gid] == 0
    # pins are light (a hollow pin) but real bodies
    mass = sum(m.body_mass[job.pin_body[0]] for _ in range(1))
    assert mass == pytest.approx(job.cfg.pin_mass, rel=1e-3)


def test_pins_stand_and_topple_when_hit(bowling):
    session, job, msgs = bowling
    sim = session.sim
    session.reset("manual")
    job.cfg.stuck_timeout_s = 0
    # pins stand still on their spots
    sim.step(2000)
    assert not any(job.pin_down(i) for i in range(N_PINS))
    p0, tilt = job.pin_pose(0)
    assert tilt < 2.0 and np.linalg.norm(p0[:2] - job.spots[0]) < 0.05
    # roll the ball into the head pin (pocket) by hand, the fly out of the way
    c = job.cfg
    job.phase = "rolling"
    job.fly_mode = "watch"  # the fly stands (frozen) as after a release
    job._t_release = job._t_phase = job.run_time()
    job._rack_before = job.rack.copy()
    job._place_ball((job.head_x - 6.0, -0.8, c.bed_height + c.ball_radius + 0.01))
    # ~ the speed the ramp gives (~200 mm/s), rolling without slipping
    sim.data.qvel[job.ball_v:job.ball_v + 3] = (200.0, 0.0, 0.0)
    sim.data.qvel[job.ball_v + 3:job.ball_v + 6] = (0.0, 200.0 / c.ball_radius, 0.0)
    for _ in range(100):
        sim.step(500)
        if job.phase == "sweep":
            break
    assert job.n_rolls == 1, msgs[-5:]
    assert job.last_roll["pins"] >= 3, (job.last_roll, msgs[-5:], job.fly_xy(), job.n_fouls, sim.time)  # the ball scatters pins with real contacts
    assert job.game.rolls[0] == job.last_roll["pins"]


def test_pinsetter_clears_knocked_and_resets_rack(bowling):
    session, job, _ = bowling
    sim = session.sim
    c = job.cfg
    session.reset("manual")
    job.game = BowlingGame()
    job.rack[:] = True
    job.reset_props()
    # knock pins 1 and 7 over by hand (tilt them onto the deck), then score
    for i in (0, 6):
        x, y = job.spots[i]
        job._place_pin(i, (x + 1.0, y, c.bed_height + 0.9), (0.7071, 0.0, 0.7071, 0.0))
    sim.step(1500)
    assert job.pin_down(0) and job.pin_down(6) and not job.pin_down(4)
    job._rack_before = job.rack.copy()
    job._roll_foul = job._roll_gutter = False
    cycles = job.n_pinsetter
    job._score_roll(job.run_time())
    assert job.phase == "sweep" and job.last_roll["pins"] == 2
    assert list(np.flatnonzero(~job.rack)) == [0, 6]
    for _ in range(60):
        sim.step(500)
        if job.phase != "sweep":
            break
    assert job.phase == "returning" and job.n_pinsetter == cycles + 1
    # knocked pins went into the machine (held, out of play); the rest stand again
    for i in range(N_PINS):
        if i in (0, 6):
            assert job.pin_hold[i] is not None
            assert job.pin_pose(i)[0][2] > c.bed_height + c.store_z - 0.1
        else:
            assert job.pin_hold[i] is None and not job.pin_down(i)
    # the second ball of the frame: after it, all 10 are reset
    resets = job.n_rack_resets
    job._rack_before = job.rack.copy()
    job._score_roll(job.run_time())  # nothing more knocked: 8 standing -> open frame
    assert job.game.frame == 1 and job._sweep["full"]
    for _ in range(60):
        sim.step(500)
        if job.phase != "sweep":
            break
    assert job.n_rack_resets == resets + 1 and job.rack.all()
    sim.step(1000)
    for i in range(N_PINS):
        assert job.pin_hold[i] is None and not job.pin_down(i)
        assert np.linalg.norm(job.pin_pose(i)[0][:2] - job.spots[i]) < 0.1


def test_ramp_release_and_slow_motion(bowling):
    """The ramp gives the ball ~200 mm/s; the release is decided by the ball (not
    the fly); a ball that stays on the ramp's edge is re-pushed, not scored; the
    roll is shown in (labelled) slow motion."""
    import mujoco as mj

    session, job, msgs = bowling
    sim = session.sim
    c = job.cfg
    session.reset("manual")
    job.cfg.stuck_timeout_s = 0
    R = c.ball_radius
    # a ball just short of the top edge, the fly (at the origin) far behind: no release
    job._place_ball((job.ramp_x0 - 0.2, 0.0, job.approach_z + R + 0.01))
    sim.step(20)
    job._fetch(job.run_time())
    assert job.phase == "fetch"
    assert job.time_scale(0.015) == 1.0
    frame = np.zeros((64, 96, 3), np.uint8)
    assert job.post_process(frame, 0.0) is frame  # no label at normal speed
    # over the edge: it's a roll, and gravity takes the ball down the ramp
    job._place_ball((job.ramp_x0 + c.stop_over + 0.1, 0.0, job.approach_z + R + 0.01))
    sim.step(20)
    job._fetch(job.run_time())
    assert job.phase == "rolling" and job.fly_mode == "watch"
    assert job.time_scale(0.015) == pytest.approx(c.slowmo)
    assert job.post_process(frame, 0.0).any()  # the SLOW MOTION label
    vmax = 0.0
    for _ in range(40):
        sim.step(50)
        vmax = max(vmax, float(np.hypot(*job.ball_vel()[:2])))
        if job.ball_pos()[0] > c.foul_x + 4.0:
            break
    v_ramp = np.sqrt(10.0 / 7.0 * 9810.0 * c.ramp_height)
    assert 0.7 * v_ramp < vmax < 1.2 * v_ramp, (vmax, v_ramp)
    # a "release" after which the ball sits on the approach: re-pushed, not scored
    rolls = job.n_rolls
    job._place_ball((job.ramp_x0 - 2.0, 0.0, job.approach_z + R + 0.01))
    mj.mj_forward(sim.model, sim.data)
    job._t_release = job.run_time() - 3.0
    job.phase = "rolling"
    job._watch_roll(job.run_time())
    assert job.phase == "fetch" and job.n_no_roll >= 1 and job.n_rolls == rolls


def test_shots_and_camera(bowling):
    """The job camera: the fly on the approach, a ride-along behind the ball after
    the release, a cut to the pin deck before the ball gets there."""
    session, job, _ = bowling
    c = job.cfg
    session.reset("manual")
    assert job._shot() == "fetch"
    job.camera_preset()
    job.phase = "rolling"
    job._ball_hidden = False
    job._place_ball((c.foul_x + 2.0, 0.0, c.bed_height + c.ball_radius + 0.01))
    import mujoco as mj
    mj.mj_forward(session.sim.model, session.sim.data)
    assert job._shot() == "lane"
    t = job.camera_target()
    assert t[0] > job.ball_pos()[0]  # looking ahead of the ball, down the lane
    job.camera_preset()
    job._place_ball((job.head_x - 3.0, 0.0, c.bed_height + c.ball_radius + 0.01))
    mj.mj_forward(session.sim.model, session.sim.data)
    assert job._shot() == "deck"
    pre = job.camera_preset()
    assert pre.tau_s < 1e-3  # a cut, not a pan
    session.reset("manual")


def test_ramp_guide_is_aimed_and_ball_only(bowling):
    """The engineered ramp guide: ball-only rails, set to each roll's release line."""
    session, job, _ = bowling
    m = session.sim.model
    from fly_simulator.jobs.bowling import BALL_BIT

    for name in ("bowl/guide_rail1", "bowl/guide_rail-1", "bowl/guide_funnel1"):
        g = m.geom(name).id
        assert m.geom_contype[g] == BALL_BIT and m.geom_conaffinity[g] == 0
        # (the pins and the fly don't touch it)
        assert not (m.geom_conaffinity[m.geom("bowl/pin0_belly").id] & BALL_BIT)
    job.phase = "fetch"
    job._new_aim()
    assert session.sim.data.mocap_pos[job.guide_mocap][1] == pytest.approx(job.release_y)
    assert abs(job.release_y) <= job.cfg.lane_half_width + 1.0
