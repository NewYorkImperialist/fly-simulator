"""shopping_carts and crop_duster eternal jobs: build (the duster on the flight body),
the loop progresses, counters, bounded pools, a reset mid-cycle."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fly_simulator.config import AppConfig
from fly_simulator.flight import FLIGHT_TIMESTEP, FlightSimulation
from fly_simulator.jobs import available_jobs, create_job_session, get_job, make_job
from fly_simulator.jobs.geometry import PROP_BIT
from fly_simulator.jobs.shopping_carts import LOOSE, NESTED
from fly_simulator.terrain import FLY_BIT, TERRAIN_BIT


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.logging.enabled = False
    cfg.render.width, cfg.render.height = 160, 106
    return cfg


def _run(session, seconds: float, until=None, chunk: int = 250) -> None:
    t_end = session.run_time() + seconds
    while session.run_time() < t_end:
        session.sim.step(chunk)
        session.after_physics()
        if until is not None and until():
            return


def test_registry():
    names = available_jobs()
    assert "shopping_carts" in names and "crop_duster" in names
    assert make_job("shopping_carts", {"n_carts": 7}).cfg.n_carts == 7
    assert make_job("crop_duster", {"n_rows": 4}).cfg.n_rows == 4
    assert get_job("crop_duster").needs_flight and not get_job("shopping_carts").needs_flight


# ---------------------------------------------------------------- shopping carts
@pytest.fixture(scope="module")
def carts():
    session, job = create_job_session("shopping_carts", _cfg(), say=lambda m: None)
    yield session, job
    session.sim.close()


def test_carts_build_and_pool(carts):
    session, job = carts
    m = session.sim.model
    c = job.cfg
    # the push colliders touch the fly only through the slippery pairs; the skids only the ground
    for j in range(c.n_carts):
        g = job.c_push[j]
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
    assert m.npair >= 7 * c.n_carts
    live = [j for j in range(c.n_carts) if job.state_c[j] == LOOSE]
    for j in live:
        s = job.c_skid[j]
        assert m.geom_contype[s] == PROP_BIT and m.geom_conaffinity[s] == TERRAIN_BIT
        assert not (m.geom_conaffinity[s] & FLY_BIT)
    # the ramp is a real static slope the fly walks on
    r = m.geom("cart/ramp").id
    assert m.geom_contype[r] == TERRAIN_BIT
    assert job.ground_height(12.0, c.ramp_y0 - 1.0) == 0.0
    assert job.ground_height(12.0, (c.ramp_y0 + c.lot_y1) / 2) == pytest.approx(
        (c.lot_y1 - c.ramp_y0) / 2 * math.tan(math.radians(c.ramp_deg)))
    # the start: a train in the corral, the rest loose in the lot
    assert job.n_nested() == c.n_nested0 and len(job.loose_carts()) == c.n_carts - c.n_nested0
    assert m.vis.map.znear == pytest.approx(0.05)


def test_cart_rolls_down_the_ramp_and_nests(carts):
    session, job = carts
    c = job.cfg
    j = int(job.loose_carts()[-1])
    job._place_loose(j, (12.0, c.lot_y1 - 2.0), -math.pi / 2)
    job.runaway[j] = True
    y0 = c.lot_y1 - 2.0
    n_run = job.runaways
    _run(session, 0.5)
    v = job.cart_vel(j)
    assert job.cart_xy(j)[1] < y0 - 2.0  # gravity alone: nobody pushed it
    assert 4.0 < -v[1] < 15.0 and abs(v[0]) < 1.0
    assert job.runaways == n_run + 1
    # a cart pushed into the corral mouth nests (kinematic glide into the next slot)
    k = int(job.loose_carts()[0])
    n0, r0 = job.n_nested(), job.returned
    job._place_loose(k, job.mouth + np.array([0.3, 0.0]), math.pi)
    _run(session, 1.0)
    assert job.returned == r0 + 1 and job.n_nested() == n0 + 1
    assert job.state_c[k] == NESTED
    assert np.allclose(job.cart_xy(k), job.slot_xy(int(job.slot[k])), atol=1e-3)
    assert job.longest_train >= job.n_nested()


def test_carts_loop_customer_and_reset(carts):
    session, job = carts
    c = job.cfg
    nq = session.sim.model.nq
    pushes0 = job.pilot.n_pushes
    # a customer takes the last cart of the train and leaves it in the lot
    job.target = None
    n0, cust0 = job.n_nested(), job.customers
    job._next_customer = job.run_time()
    fly = job.fly_xy()
    if float(np.linalg.norm(fly - job.mouth)) < 5.0:
        pytest.skip("fly at the corral mouth")
    _run(session, 0.1)
    assert job.customers == cust0 + 1 and job.n_nested() == n0 - 1
    _run(session, 14.0)
    assert job.pilot.n_pushes > pushes0  # the fly went after a cart and pushed it
    # bounded pool: every cart is somewhere, the model never grows
    assert session.sim.model.nq == nq
    assert len(job.state_c) == c.n_carts
    hud = "\n".join(job.hud_lines())
    assert "carts returned" in hud and "cart train" in hud and "runaways" in hud
    # an explicit reset mid-cycle: the loose carts stay where they were, the train stays
    loose = job.loose_carts()
    before = {int(j): job.cart_xy(j) for j in loose}
    train = job.n_nested()
    job.recover("test")
    for j, p in before.items():
        if job.state_c[j] == LOOSE:
            assert np.allclose(job.cart_xy(j), p, atol=0.3)
    assert job.n_nested() >= train - 1
    _run(session, 2.0)
    for j in range(c.n_carts):
        assert np.all(np.isfinite(job.cart_xy(j)))
    st = job.stats()
    assert st["auto_recoveries"] >= 1 and st["carts_lost"] == 0


# ---------------------------------------------------------------- crop duster
@pytest.fixture(scope="module")
def duster():
    session, job = create_job_session("crop_duster", _cfg(), {"gust_mean_s": 1e6}, say=lambda m: None)
    trace = {"phases": [], "max_puffs": 0, "min_pass_z": 99.0, "max_pass_z": 0.0}

    def watch():
        if not trace["phases"] or trace["phases"][-1] != job.phase:
            trace["phases"].append(job.phase)
        live = int(np.count_nonzero(session.sim.model.geom_rgba[job.puff_geom, 3] > 0))
        trace["max_puffs"] = max(trace["max_puffs"], live)
        if job.phase == "pass" and job._spraying:
            z = float(session.sim.com()[2])
            trace["min_pass_z"] = min(trace["min_pass_z"], z)
            trace["max_pass_z"] = max(trace["max_pass_z"], z)
        return job.rows >= 2

    _run(session, 12.0, until=watch, chunk=100)
    yield session, job, trace
    session.sim.close()


def test_duster_build_on_the_flight_fly(duster):
    session, job, _ = duster
    sim, m = session.sim, session.sim.model
    assert isinstance(sim, FlightSimulation) and sim.timestep == FLIGHT_TIMESTEP
    assert session.flight is not None and m.opt.density > 0
    g = job.cell_geoms
    assert len(g) == job.n_cells == len(job.fields) * job.cfg.n_rows * job.n_seg
    assert np.all(m.geom_contype[g] == 0) and np.all(m.geom_conaffinity[g] == 0)
    assert len(job.puff_mocap) == job.cfg.n_puffs and np.all(job.puff_mocap >= 0)
    assert np.all(m.geom_contype[job.puff_geom] == 0)
    assert m.vis.map.znear == pytest.approx(0.3)


def test_duster_real_flight_passes_and_dust(duster):
    session, job, trace = duster
    c = job.cfg
    fm = session.flight
    assert job.rows >= 2 and job.n_takeoffs == 1 and fm.counts["takeoff"] == 1
    ph = trace["phases"]
    for a, b in zip(("ground", "takeoff", "climb", "transit", "lineup", "pass"),
                    ("takeoff", "climb", "transit", "lineup", "pass", "turn")):
        assert ph.index(a) < ph.index(b)
    assert job.n_crash == 0 and job.n_falls == 0
    assert c.pass_alt - 0.8 < trace["min_pass_z"] and trace["max_pass_z"] < c.pass_alt + 2.0
    # the dust: bounded particles, cells under the passes tinted
    assert 0 < trace["max_puffs"] <= c.n_puffs
    f = job.field
    assert np.count_nonzero(job.dust[job.cell_f == f] > 0.5) > job.n_seg  # more than a row's worth
    assert np.count_nonzero(job.dust[job.cell_f != f]) == 0
    assert job.dust_used > 50.0 and job.hopper < c.hopper_mm
    assert job.distance > 80.0 and job.flight_time > 3.0
    assert "fields dusted" in "\n".join(job.hud_lines())


def test_duster_regrowth_and_stages(duster):
    session, job, _ = duster
    m = session.sim.model
    saved = job.dust.copy()
    job._spraying = False
    job.dust[:] = 1.0
    job._dust_step(0.02)
    assert np.all(m.geom_matid[job.cell_geoms] == job.crop_mats[-1])  # fully dusted stage
    job._dust_step(job.cfg.regrow_s * 0.8)
    assert np.all(job.dust < 0.25) and np.all(m.geom_matid[job.cell_geoms] == job.crop_mats[1])
    job.dust[:] = saved
    job._dust_step(0.0)


def test_duster_refill_gust_and_reset(duster):
    session, job, _ = duster
    c = job.cfg
    fm = session.flight
    # a gust of wind: physical medium velocity
    job._next_gust = job.run_time()
    w0 = session.sim.model.opt.wind[:2].copy()
    _run(session, c.gust_s * 0.5, chunk=100)
    assert job.gusts == 1
    assert float(np.hypot(*(session.sim.model.opt.wind[:2] - w0))) > 3.0
    # an empty hopper after this field: home to the strip, land, refill, take off again
    job.hopper = 0.0
    job.row = c.n_rows - 1
    if job.phase == "turn":
        job._braked = True
        job._turn_x = float(session.sim.com()[0])
        job._t_brake = session.sim.time - job.t_phase
    _run(session, 14.0, until=lambda: job.refills >= 1 and job.phase in ("takeoff", "climb"), chunk=100)
    assert job.refills == 1 and job.n_landings >= 1 and job.hopper == pytest.approx(c.hopper_mm)
    assert job.n_crash == 0
    # an explicit reset mid-flight: back on the strip, still air, it takes off again
    _run(session, 0.3, chunk=100)
    t0 = job.n_takeoffs
    job.recover("test")
    assert job.phase == "ground" and np.all(session.sim.model.opt.wind == 0.0)
    _run(session, 1.5, until=lambda: job.n_takeoffs > t0, chunk=100)
    assert job.n_takeoffs == t0 + 1 and fm.state != "walking"
    assert job.stats()["auto_recoveries"] >= 1
