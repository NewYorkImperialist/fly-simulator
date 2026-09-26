"""Yard-work eternal jobs (fly_simulator/jobs/mowing.py, raking.py): props and contact
bits, the grass pool (cutting, stripes, regrowth), the mower being pushed, the leaf
pool (rake carrying, heaping, gusts, recycling) and short headless runs."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fly_simulator.config import AppConfig
from fly_simulator.jobs import JobRunner, available_jobs, create_job_session, make_job
from fly_simulator.jobs.geometry import PROP_BIT
from fly_simulator.terrain import TERRAIN_BIT


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.logging.enabled = False
    cfg.render.width, cfg.render.height = 160, 106
    return cfg


def test_registry_has_yard_jobs():
    names = available_jobs()
    assert "mowing" in names and "raking" in names
    job = make_job("mowing", {"n_rows": 4})
    assert job.cfg.n_rows == 4
    # serpentine: rows 0..n-1 then back down, alternating direction
    assert [job.row_index(k) for k in range(8)] == [0, 1, 2, 3, 3, 2, 1, 0]
    assert [job.row_dir(k) for k in range(4)] == [1, -1, 1, -1]


# ---------------------------------------------------------------- mowing
@pytest.fixture(scope="module")
def mowing():
    session, job = create_job_session("mowing", _cfg(), say=lambda m: None)
    yield session, job
    session.sim.close()


def test_mowing_props(mowing):
    session, job = mowing
    m = session.sim.model
    deck = m.geom("mow/deck").id
    assert m.geom_contype[deck] == PROP_BIT and m.geom_conaffinity[deck] & TERRAIN_BIT
    assert m.body_mass[job.mower_body] == pytest.approx(job.cfg.mower_mass, rel=1e-3)
    # grass: a fixed pool of visual-only blades (can't trip the fly, no BVH refit)
    g = job.blade_geoms
    assert len(g) == job.n_blades > 1000
    assert np.all(m.geom_contype[g] == 0) and np.all(m.geom_conaffinity[g] == 0)
    assert m.npair >= 7  # slippery head / thorax / abdomen pairs with the deck
    # planar mower: 2 slides + 1 hinge, no free joint
    j = m.body_jntadr[job.mower_body]
    assert m.body_jntnum[job.mower_body] == 3 and m.jnt_type[j] == 2  # slide


def test_mowing_cut_and_regrow(mowing):
    session, job = mowing
    m = session.sim.model
    job.regrow_all()
    c = job.cfg
    spot = np.array([10.0, job.row_y(2)])
    a0 = job.area_m2
    job.row = 1  # a -x row: dark stripe
    job.update_grass(spot, 0.0)
    under = np.sum((job.blade_xy - spot) ** 2, axis=1) < c.deck_radius ** 2
    assert under.sum() > 20
    assert np.allclose(job.blade_h[under], c.cut_height)
    assert np.allclose(m.geom_size[job.blade_geoms[under], 2], c.cut_height / 2)
    assert np.all(job.blade_stripe[under] == -1)
    area = job.area_m2 - a0
    assert area == pytest.approx(under.sum() * job.cell_area_m2, rel=1e-6)
    # regrowth: half of regrow_s later the cut blades are half grown, colour fading
    far = np.array([-100.0, -100.0])
    job.update_grass(far, c.regrow_s / 2)
    mid = c.cut_height + 0.5 * (job.blade_full[under] - c.cut_height)
    assert np.allclose(job.blade_h[under], mid)
    job.update_grass(far, c.regrow_s)
    assert np.allclose(job.blade_h, job.blade_full)
    job.row = 0
    job.regrow_all()


def test_mowing_pushes_mower_and_cuts(mowing):
    session, job = mowing
    r = JobRunner(session, job, print_every_s=1e9, say=lambda m: None)
    x0 = float(job.mower_xy()[0])
    out = r.run(max_seconds=2.0)
    assert float(job.mower_xy()[0]) > x0 + 1.5  # pushed along row 0 (+x)
    assert out["area_m2"] > 0 and out["blades_cut"] > 20
    assert out["falls"] == 0 and job.pilot.n_pushes >= 1
    lines = job.hud_lines()
    assert lines[0].startswith("LAWN MOWER FLY") and any("rows mowed" in ln for ln in lines)
    # the fly respawns after an explicit reset; the mower stays where it was
    before = job.mower_xy()
    job.recover("test")
    if np.linalg.norm(before) > job.cfg.deck_radius + 3.5:
        np.testing.assert_allclose(job.mower_xy(), before, atol=1e-6)
    assert job.pilot.state == "approach"


def test_mowing_row_completion(mowing):
    session, job = mowing
    d = session.sim.data
    rows0 = job.rows_mowed
    job.row = 0
    # put the mower at the end of row 0: the next update counts the row
    x0, y0 = job.mower_start
    d.qpos[job.qx], d.qpos[job.qy] = job.x_end - x0, 0.0 - y0
    d.qvel[[job.vx, job.vy]] = 0.0
    session.sim.step(job.cfg.update_every_steps)
    assert job.rows_mowed == rows0 + 1 and job.row == 1 and job.row_dir() == -1
    # the goal is now on row 1, heading -x
    goal, x_end = job.row_goal(job.mower_xy())
    assert goal[1] == pytest.approx(job.row_y(1)) and x_end == pytest.approx(job.x_start)


# ---------------------------------------------------------------- raking
@pytest.fixture(scope="module")
def raking():
    session, job = create_job_session("raking", _cfg(), say=lambda m: None)
    yield session, job
    session.sim.close()


def test_raking_props_and_pool(raking):
    session, job = raking
    m = session.sim.model
    c = job.cfg
    assert len(job.leaf_mocap) == c.n_leaves and m.nmocap == c.n_leaves + 1
    g = m.geom("rake/leaf0_g").id
    assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0  # kinematic, visual
    k = job.counts()
    assert sum(k.values()) == c.n_leaves
    assert k["ground"] == c.leaves_on_ground_at_start


def test_rake_carries_leaf_into_pile(raking):
    session, job = raking
    c = job.cfg
    p, u, n = job.rake_frame()
    i = int(np.flatnonzero(job.leaf_state == 2)[0])  # a ground leaf
    # just behind the comb's front face, inside the comb width
    job.leaf_pos[i, :2] = p + u * (c.rake_reach - 0.3) + n * 0.4
    job.leaf_raked[i] = False
    job._rake_leaves()
    rel = job.leaf_pos[i, :2] - p
    assert float(rel @ u) >= c.rake_reach  # pushed to the front face
    assert float(rel @ n) == pytest.approx(0.4, abs=1e-6) and job.leaf_raked[i]
    # a raked leaf that reaches the pile is heaped and counted
    raked0, n0 = job.leaves_raked, job.n_pile
    job.leaf_pos[i, :2] = c.pile_xy
    job._rake_leaves()
    assert job.leaf_state[i] == 3 and job.leaves_raked == raked0 + 1
    assert job.n_pile == n0 + 1 and job.work == job.leaves_raked
    # an unraked leaf blown into the pile is heaped but not counted
    j = int(np.flatnonzero(job.leaf_state == 2)[0])
    job.leaf_raked[j] = False
    job.leaf_pos[j, :2] = c.pile_xy
    job._rake_leaves()
    assert job.leaf_state[j] == 3 and job.leaves_raked == raked0 + 1


def test_raking_gust_and_recycling(raking):
    session, job = raking
    sim = session.sim
    n_pile = job.n_pile
    assert n_pile >= 1
    g0 = job.gusts
    job.gust()
    assert job.gusts == g0 + 1 and job.n_pile == 0
    assert not np.any(job.leaf_state == 3)  # the whole pile is airborne
    assert job.counts()["gust"] >= n_pile
    sim.step(int(2.5 / sim.model.opt.timestep))  # longer than any gust flight
    k = job.counts()
    assert k["gust"] == 0 and sum(k.values()) == job.cfg.n_leaves  # pool is constant
    # the tree keeps dropping leaves
    fallen0 = job.leaves_fallen
    if k["tree"]:
        assert job.drop_leaf() and job.leaves_fallen == fallen0 + 1
    # leaves follow their mocap bodies
    d = sim.data
    np.testing.assert_allclose(d.mocap_pos[job.leaf_mocap], job.leaf_pos)


def test_raking_short_run(raking):
    session, job = raking
    r = JobRunner(session, job, print_every_s=1e9, say=lambda m: None)
    out = r.run(max_seconds=1.0)
    assert out["falls"] == 0 and job.state in ("approach", "align", "sweep", "rest")
    # the rake is kept in front of the thorax
    d = session.sim.data
    p = job.fly_xy()
    np.testing.assert_allclose(d.mocap_pos[job.rake_mocap, :2], p, atol=0.05)
    h = session.sim.heading()
    q = d.mocap_quat[job.rake_mocap]
    assert abs(math.remainder(2 * math.atan2(q[3], q[0]) - h, 2 * math.pi)) < 0.05
    lines = job.hud_lines()
    assert lines[0].startswith("LEAF RAKING FLY") and any("piles completed" in ln for ln in lines)
