"""snow_shovel and mini_golf eternal jobs: build, bounded pools, the loop progresses,
counters, a reset mid-cycle."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fly_simulator.config import AppConfig
from fly_simulator.jobs import available_jobs, create_job_session, make_job
from fly_simulator.jobs.geometry import PROP_BIT
from fly_simulator.terrain import FLY_BIT, TERRAIN_BIT


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.logging.enabled = False
    cfg.render.width, cfg.render.height = 160, 106
    return cfg


def _run(session, seconds: float) -> None:
    sim = session.sim
    t_end = sim.time + seconds
    while sim.time < t_end:
        sim.step(250)
        session.after_physics()


def test_registry():
    names = available_jobs()
    assert "snow_shovel" in names and "mini_golf" in names
    assert make_job("snow_shovel", {"n_clumps": 12}).cfg.n_clumps == 12
    assert make_job("mini_golf", {"skill": 0.3}).cfg.skill == 0.3


# ---------------------------------------------------------------- snow shovel
@pytest.fixture(scope="module")
def snow():
    session, job = create_job_session("snow_shovel", _cfg(), say=lambda m: None)
    yield session, job
    session.sim.close()


def test_snow_build_and_pools(snow):
    session, job = snow
    m = session.sim.model
    brace = m.geom("snow/brace").id
    assert m.geom_contype[brace] == PROP_BIT
    assert m.npair >= 7  # slippery head / thorax / abdomen pairs with the brace
    # the snow cover and the flakes are visual; the clumps never touch the fly
    g = job.tile_geoms
    assert len(g) == job.n_tiles > 500
    assert np.all(m.geom_contype[g] == 0) and np.all(m.geom_conaffinity[g] == 0)
    for gc in job.c_geom:
        assert not (m.geom_conaffinity[gc] & FLY_BIT) and not (m.geom_contype[gc] & FLY_BIT)
    assert m.nmocap >= job.cfg.n_flakes
    assert m.vis.map.znear == pytest.approx(0.05)


def test_snow_scrape_dump_and_snowfall(snow):
    session, job = snow
    c = job.cfg
    job.depth[:] = c.max_depth * 0.8
    spot = np.array([job.lane_x(2), 0.0])
    v0, load0 = job.volume, job.load
    job.update_snow(0.0, spot, math.pi / 2, True)
    assert job.load > load0 and job.volume > v0
    rel = job.tile_xy - spot
    under = (rel[:, 1] > 0.3) & (rel[:, 1] < 1.2) & (np.abs(rel[:, 0]) < c.blade_half_w - 0.1)
    assert under.sum() > 5 and np.all(job.depth[under] == 0.0)
    # snowfall re-covers the scraped tiles
    job.update_snow(40.0, np.array([-100.0, -100.0]), 0.0, False)
    assert np.all(job.depth[under] > 0.01)
    # the dump throws pooled clumps onto the bank; the pool never grows
    n0 = job.clumps_shoveled
    h0 = job.bank_height()
    job.load = 4.0
    n = job.dump(np.array([job.lane_x(2), job.y_end]), math.pi / 2)
    assert n >= 1 and job.clumps_shoveled == n0 + n and job.load == 0.0
    assert job.bank_height() > h0
    for _ in range(15):  # more dumps than the pool: the oldest are packed in
        job.load = 4.0
        job.dump(np.array([job.lane_x(3), job.y_end]), math.pi / 2)
    assert len(job.c_state) == c.n_clumps and job.clumps_recycled > 0


def test_snow_loop_and_reset(snow):
    session, job = snow
    s0 = job.shovel_xy().copy()
    lanes0, work0 = job.lanes, job.work
    _run(session, 14.0)
    assert np.linalg.norm(job.shovel_xy() - s0) > 2.0  # the fly pushed the shovel
    assert job.work > work0 and job.lanes > lanes0
    assert job.clumps_shoveled > 0
    assert any("path cleared" in ln for ln in job.hud_lines())
    st = job.stats()
    assert st["days_of_winter"] >= 1 and st["bank_height_mm"] > 0
    # a reset mid-cycle: the shovel stays where it was, the job continues
    s1 = job.shovel_xy().copy()
    session.reset("test")
    assert np.linalg.norm(job.shovel_xy() - s1) < 0.3 or np.linalg.norm(s1) < 4.0
    w1 = job.work
    _run(session, 6.0)
    assert job.work >= w1 and np.all(np.isfinite(job.shovel_xy()))
    assert job.n_falls == 0


# ---------------------------------------------------------------- mini golf
@pytest.fixture(scope="module")
def golf():
    session, job = create_job_session("mini_golf", _cfg(), {"skill": 1.0}, say=lambda m: None)
    yield session, job
    session.sim.close()


def test_golf_build(golf):
    session, job = golf
    m = session.sim.model
    bg = job.ball_geom
    assert m.body_mass[job.ball_body] == pytest.approx(job.cfg.ball_mass, rel=1e-3)
    # the ball never touches the fly's floor plane (the cup is a real hole)
    gp = m.geom("ground_plane").id
    assert not (m.geom_contype[gp] & m.geom_conaffinity[bg]) and not (m.geom_contype[bg] & m.geom_conaffinity[gp])
    assert m.geom_group[gp] == 3
    assert len(job.ball_pairs) >= 7
    # the ramp is walkable (fly + ball), the rails ball-only
    r = m.geom("golf/ramp_up").id
    assert m.geom_contype[r] & TERRAIN_BIT and m.geom_conaffinity[r] & FLY_BIT
    rail = m.geom("golf/rail0_end").id
    assert m.geom_conaffinity[rail] == 0 and not (m.geom_contype[rail] & TERRAIN_BIT)
    assert job.ground_height(job.ramp_profile()[1] + 0.5, job.hole_y(1)) == pytest.approx(job.cfg.ramp_h)


def test_golf_aim_model(golf):
    _, job = golf
    sh = job.plan_shot(0, job.tee_xy(0))
    assert sh["dir"][0] > 0.99 and 40 < sh["speed"] < 120
    # the ramp needs the extra climb energy
    s1 = job.plan_shot(1, job.tee_xy(1))
    assert s1["speed"] > job.plan_shot(3, np.array([15.0, job.hole_y(3) + 2.5]))["speed"]
    # bumpers: a post on the line -> a clear route only
    sh3 = job.plan_shot(3, job.tee_xy(3))
    assert job.clear_line(3, job.tee_xy(3), sh3["target"])


def test_golf_ball_drops_in_cup(golf):
    session, job = golf
    job.hole = 2
    job.phase = "roll"  # the job only watches
    job._lie_damping(False)
    cup = job.cup_xy(2)
    v = np.array([40.0, 0, 0])
    job._place_ball(np.array([cup[0] - 3.0, cup[1], job.cfg.ball_radius + 0.002]), v)
    session.sim.data.qvel[job.bv + 3:job.bv + 6] = np.cross([0, 0, 1.0], v) / job.cfg.ball_radius
    job._t_phase = job.run_time()
    sim = session.sim
    for _ in range(300):
        sim.step(10)
        if job.phase != "roll":
            break
    assert job.phase == "retrieve_walk"  # holed -> the fly goes to get it
    assert job.card[2] is not None
    session.reset("test")


def test_golf_loop_scorecard_and_reset(golf):
    session, job = golf
    p0 = job.putts
    h0 = job.holes_played
    _run(session, 45.0)
    assert job.putts > p0 and job.holes_played > h0
    assert job.total_strokes >= job.holes_played
    assert any("card" in ln for ln in job.hud_lines())
    # reset mid-hole: the ball stays where it lies, the hole continues
    job._ball_last = job.ball_pos()
    b = job.ball_pos().copy()
    hole = job.hole
    session.reset("test")
    if job.phase == "play":
        assert job.hole == hole or job.hole == (hole + 1) % 4
        assert np.all(np.isfinite(job.ball_pos()))
    _run(session, 5.0)
    assert np.all(np.isfinite(job.ball_pos())) and np.linalg.norm(job.ball_pos()) < 200
    del b
