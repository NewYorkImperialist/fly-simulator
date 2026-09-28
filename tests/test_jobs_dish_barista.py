"""The dishwasher and barista jobs (fly_simulator/jobs/dishwasher.py, barista.py): they
build (the job colliders touch only the fly, the spill drops never touch the fly, no
unshadowed spot light, fixed pools), the loops run in order (fetch -> scrub by real
contact -> rinse -> rack; a drink: cup -> grind -> tamp by real contact -> shot ->
steam -> pour -> bell -> serve), the counters move, the pools stay bounded, and a reset
mid-cycle resumes the work."""

from __future__ import annotations

import mujoco as mj
import numpy as np
import pytest

from fly_simulator.config import AppConfig
from fly_simulator.jobs import available_jobs, create_job_session, make_job
from fly_simulator.jobs.barista import DROP_BIT, BaristaJob
from fly_simulator.jobs.dishwasher import DishwasherJob
from fly_simulator.terrain import FLY_BIT


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.logging.enabled = False
    cfg.render.width, cfg.render.height = 160, 106
    return cfg


def _run(session, job, seconds: float, trace: list | None = None, attr: str = "phase") -> None:
    t_end = session.run_time() + seconds
    while session.run_time() < t_end:
        session.sim.step(50)
        session.after_physics()
        if trace is not None:
            v = getattr(job, attr)
            if not trace or trace[-1] != v:
                trace.append(v)


def _no_unshadowed_spots(m: mj.MjModel) -> None:
    for i in range(m.nlight):
        if m.light_type[i] == mj.mjtLightType.mjLIGHT_SPOT:
            assert m.light_castshadow[i], "a spot light without shadows blacks out pixels on macOS"


def _job_colliders(m: mj.MjModel, prefix: str) -> list[int]:
    out = []
    for g in range(m.ngeom):
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g) or ""
        if nm.startswith(prefix) and (m.geom_contype[g] or m.geom_conaffinity[g]):
            out.append(g)
    return out


def test_registered():
    names = available_jobs()
    assert "dishwasher" in names and "barista" in names
    assert make_job("dishwasher", {"rack_n": 4}).cfg.rack_n == 4
    assert make_job("barista", {"spill_p": 0.0}).cfg.spill_p == 0.0


# ---------------------------------------------------------------------------
# dishwasher
# ---------------------------------------------------------------------------

DISH_FAST = {"scrub_per_mm": 0.6, "tough_min": 1.0, "stack_n": 2, "rack_n": 2, "shadows": False}


@pytest.fixture(scope="module")
def dish():
    session, job = create_job_session("dishwasher", _cfg(), DISH_FAST)
    yield session, job
    session.sim.close()


def test_dish_builds(dish):
    session, job = dish
    m = session.sim.model
    assert isinstance(job, DishwasherJob)
    cols = _job_colliders(m, "dish/")
    assert cols == [job.pad_geom]  # the only job collider: the plate pad, fly-only
    assert m.geom_contype[job.pad_geom] == 0 and m.geom_conaffinity[job.pad_geom] == FLY_BIT
    assert m.body_mocapid[m.geom_bodyid[job.pad_geom]] >= 0
    _no_unshadowed_spots(m)
    assert m.vis.map.znear == pytest.approx(0.05)
    assert len(job.plates) == job.cfg.n_plates
    assert all(len(P["life"]) == n for P, n in zip(job.parts.values(),
                                                   (job.cfg.n_bubbles, job.cfg.n_drops, job.cfg.n_splash)))


def test_dish_loop(dish):
    session, job = dish
    m = session.sim.model
    nbody = m.nbody
    phases, carrier = [], []
    t_end = session.run_time() + 30.0
    grime_trace = []
    while session.run_time() < t_end and job.n_rack_loads < 1:
        session.sim.step(50)
        session.after_physics()
        for tr, v in ((phases, job.phase), (carrier, job.carrier)):
            if not tr or tr[-1] != v:
                tr.append(v)
        if job.phase == "scrub" and job.wash is not None:
            grime_trace.append((job.contact_s, job.plates[job.wash]["grime"]))
    # the order of one plate
    i = phases.index("fetch_reach")
    assert phases[i:i + 8] == ["fetch_reach", "fetch_grip", "fetch_lift", "fetch_carry", "fetch_lower",
                               "fetch_release", "scrub_hover", "scrub_down"]
    assert "scrub" in phases and "hand_over" in phases
    j = carrier.index("to_rinse")
    assert carrier[j:j + 4] == ["to_rinse", "rinse", "to_rack", "idle"]
    # the grime came off only with contact (it never drops while contact time stands still)
    for (c0, g0), (c1, g1) in zip(grime_trace, grime_trace[1:]):
        if c1 == c0:
            assert g1 >= g0 - 1e-12
    assert job.contact_s > 0.5 and job.scrub_mm > 1.0 and job.max_force > 0.0
    assert job.n_washed >= 2 and job.work == job.n_washed
    assert job.n_rack_loads == 1  # the rack of 2 was carted off
    assert 50.0 < job.grime_removed_pct() <= 100.0
    assert job.n_strokes >= 1 and job.sponge_wear > 0.0
    hud = "\n".join(job.hud_lines())
    assert "grime removed" in hud and "sponge wear" in hud and "rack loads" in hud
    st = job.stats()
    for k in ("plates_washed", "grime_removed_pct", "sponge_wear_pct", "rack_loads", "strokes"):
        assert k in st
    # the stack of 2 ran out: a new dirty stack came on the conveyor
    _run(session, job, 4.0)
    assert job.n_stacks >= 2
    # bounded pools: nothing grows
    assert m.nbody == nbody and len(job.plates) == job.cfg.n_plates
    assert sum(p["where"] != "parked" for p in job.plates) <= job.cfg.stack_n + job.cfg.rack_n + 2
    assert job.n_falls == 0


def test_dish_reset_mid_scrub(dish):
    session, job = dish
    t_end = session.run_time() + 20.0
    while job.phase != "scrub" and session.run_time() < t_end:
        session.sim.step(50)
        session.after_physics()
    assert job.phase == "scrub"
    i = job.wash
    washed = job.n_washed
    session.reset("manual")
    assert job.phase == "settle" and job.wash == i and job.plates[i]["where"] == "wash"
    _run(session, job, 1.5)
    assert job.phase.startswith("scrub") or job.phase in ("hand_over", "idle")
    _run(session, job, 8.0)
    assert job.n_washed > washed
    assert np.all(np.isfinite(session.sim.data.qpos))


# ---------------------------------------------------------------------------
# barista
# ---------------------------------------------------------------------------

CAFE_FAST = {"shot_s": 1.0, "steam_s": 0.8, "pour_s": 1.5, "grind_s": 0.5, "pf_move_s": 0.5,
             "cup_slide_s": 0.5, "serve_s": 0.8, "jug_move_s": 0.5, "order_every_s": 1e6, "queue_start": 3,
             "spill_p": 1.0, "shadows": False}


@pytest.fixture(scope="module")
def cafe():
    session, job = create_job_session("barista", _cfg(), CAFE_FAST)
    yield session, job
    session.sim.close()


def test_cafe_builds(cafe):
    session, job = cafe
    m = session.sim.model
    assert isinstance(job, BaristaJob)
    fly_only = {job.puck_geom, job.bell_geom, *job.btn_geom.values()}
    drops = {d["geom"] for d in job.drops}
    for g in _job_colliders(m, "cafe/"):
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g)
        if g in fly_only:
            assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == FLY_BIT, nm
        elif g in drops:
            assert not (m.geom_conaffinity[g] & FLY_BIT) and not (m.geom_contype[g] & FLY_BIT), nm
        else:
            assert nm == "cafe/tray_col" and m.geom_conaffinity[g] == DROP_BIT, nm
    _no_unshadowed_spots(m)
    assert m.vis.map.znear == pytest.approx(0.05)
    assert len(job.cups) == job.cfg.n_cups and len(job.queue) == 3


def test_cafe_one_drink(cafe):
    session, job = cafe
    m = session.sim.model
    nbody = m.nbody
    trace: list = []
    t_end = session.run_time() + 30.0
    while session.run_time() < t_end and job.n_served < 1:
        session.sim.step(50)
        session.after_physics()
        if not trace or trace[-1] != job.phase:
            trace.append(job.phase)
    want = ["wait_order", "cup_in", "pf_to_grinder", "grind", "pf_back", "tamp_hover", "tamp_down", "tamp_up",
            "pf_lock", "btn_hover", "btn_down", "btn_up", "shot", "btn_hover", "btn_down", "btn_up", "jug_to_wand",
            "steam", "jug_to_pour", "pour_reach", "pour_grip", "pour", "pour_back", "jug_home", "bell_hover",
            "bell_down", "bell_up", "serve"]
    seq = [p for i, p in enumerate(trace) if i == 0 or p != trace[i - 1]]
    # the tamps repeat down / up; collapse those
    col = []
    for p in seq:
        if p in ("tamp_down", "tamp_up") and "pf_lock" not in col and p in col:
            continue
        col.append(p)
    col = [p for p in col if p != "setup"]
    assert col[:len(want)] == want
    assert col[len(want)] in ("wait_order", "cup_in")  # the next order starts at once
    assert job.n_served == 1 and job.work == 1 and job.n_shots == 1
    assert job.n_tamps >= job.cfg.n_tamps and job.max_tamp_force >= job.cfg.tamp_force  # real contacts
    assert job.n_presses + job.n_auto_presses == 3 and job.n_bells == 1
    assert job.last_score is not None and 35.0 <= job.last_score <= 100.0
    assert len(job.queue) + (job.order is not None) == 2 and job.pickup  # the cup is on the pickup counter
    cp = job.cups[job.pickup[-1]]
    assert cp["where"] == "pickup" and cp["surf_mat"] in [x for p in job.art_mat for w in p for x in w]
    # the spill: a real free body dropped, landed, left a puddle and went back to the pool
    assert job.n_spills == 1
    _run(session, job, 3.0)
    assert all(d["state"] == "parked" for d in job.drops)
    assert any(pd["t"] > 0 for pd in job.puddles)
    hud = "\n".join(job.hud_lines())
    assert "shots pulled" in hud and "latte art" in hud and "queue" in hud
    st = job.stats()
    for k in ("drinks_served", "shots_pulled", "latte_art_mean", "orders_in_queue"):
        assert k in st
    assert m.nbody == nbody and len(job.cups) == job.cfg.n_cups
    assert job.n_falls == 0


def test_cafe_reset_mid_drink(cafe):
    session, job = cafe
    t_end = session.run_time() + 20.0
    while not job.phase.startswith("tamp") and session.run_time() < t_end:
        session.sim.step(50)
        session.after_physics()
    assert job.phase.startswith("tamp")
    served = job.n_served
    order = job.order
    session.reset("manual")
    assert job.phase == "setup" and job.order == order
    _run(session, job, 1.0)
    assert job.phase.startswith("tamp") or job.phase == "pf_lock"
    _run(session, job, 16.0)
    assert job.n_served > served
    assert np.all(np.isfinite(session.sim.data.qpos))
