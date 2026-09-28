"""The fry cook job (fly_simulator/jobs/fry_cook.py): the scene and its contact scheme,
the bounded fry pool, one fast-config order in the right order (basket dump -> scoop
-> carton -> tray -> ORDER UP), the counters, the sneaked-fry taste with a fake brain
(stand-in sets, proboscis), and a reset mid-cycle."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fly_simulator.config import AppConfig
from fly_simulator.jobs import available_jobs, create_job_session, make_job
from fly_simulator.jobs import fry_cook as FC
from fly_simulator.jobs.fry_cook import CookStance, FryCookJob
from fly_simulator.terrain import FLY_BIT

FAST = dict(cook_s=1.0, carton_min=2, carton_max=3, n_fries=36, batch=12, spill_p=0.0,
            sneak_p=0.0, slowmo=False)


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.logging.enabled = False
    cfg.render.width, cfg.render.height = 160, 106
    return cfg


def _step(session, n: int = 50) -> None:
    session.sim.step(n)
    session.after_physics()


class _FakeBrain:
    """``latest`` with MN9 = 60 Hz while a sugar taste pulse is on, else 0."""

    def __init__(self, session) -> None:
        self.session = session
        self.sent, self.stim_log = [], []
        self.latest = None

    def send(self, ev, source=""):
        self.sent.append(ev)
        self.stim_log.append(ev)

    def update(self):
        rt = self.session.run_time()
        on = any(ev.sim_time <= rt <= ev.sim_time + ev.duration_s + 0.05
                 and "sugar" in ev.details.get("tastes", ()) for ev in self.sent)
        self.latest = SimpleNamespace(probes={"MN9": 60.0 if on else 0.0})


@pytest.fixture(scope="module")
def cook():
    msgs: list[str] = []
    session, job = create_job_session("fry_cook", _cfg(), FAST, say=msgs.append)
    yield session, job, msgs
    session.brain = None
    session.sim.close()


def test_registry_and_config():
    assert "fry_cook" in available_jobs()
    job = make_job("fry_cook", {"cook_s": 3.0, "n_fries": 30})
    assert isinstance(job, FryCookJob)
    assert job.cfg.cook_s == 3.0 and job.cfg.n_fries == 30
    assert job.znear == 0.05
    # taste stand-in: salty enough -> the sugar set; over-salted -> + the bitter set
    d = job.taste_details(0.9)
    assert d["tastes"] == ["sugar"] and "stand-in" in d["label"]
    d = job.taste_details(1.6)
    assert d["tastes"] == ["sugar", "bitter"] and d["bitter_hz"] > 0


def test_scene_contacts_and_pool(cook):
    session, job, _ = cook
    m = session.sim.model
    names = [m.geom(g).name for g in range(m.ngeom)]
    job_geoms = [g for g, n in enumerate(names) if n.startswith("fry/")]
    assert len(job_geoms) > 300
    for g in job_geoms:  # nothing of the job ever touches the fly
        assert not (m.geom_contype[g] & FLY_BIT) and not (m.geom_conaffinity[g] & FLY_BIT), names[g]
    # the only colliding job geoms: fry capsules and the bin / carton / basket colliders
    live = [names[g] for g in job_geoms if m.geom_contype[g] or m.geom_conaffinity[g]]
    assert "fry/bin_floor" in live
    assert all("_col" in n or n.startswith("fry/bin_") for n in live)
    # a fixed pool, the fly's normal mass, stationary stance, proboscis actuators
    assert len(job.f_state) == FAST["n_fries"] and len(job.carton_mocap) == FC.N_CARTONS
    assert session.sim.fly_mass == pytest.approx(1.024e-3, rel=0.02)
    assert CookStance.name in session.STATIONARY_ACTIONS
    assert len(job.prob_ids) == 2
    assert m.vis.map.znear == pytest.approx(0.05)


def test_one_order_in_order(cook):
    """Fast cycle: the basket dumps a batch (fries fall into the bin with physics), the
    fly takes a carton, scoops fries into it (physically settled), the carton slides
    onto the tray, DING / ORDER UP, the tray goes through the window and comes back;
    counters; the pool stays bounded; no fry ever touches the fly."""
    session, job, msgs = cook
    session.reset("manual")
    m, d = session.sim.model, session.sim.data
    fly_bodies = {b for b in range(m.nbody) if m.body(b).name.startswith(session.sim.fly_name + "/")}
    fry_geoms = set(job.f_col.tolist())
    seq, fryer, tray = [], [], []
    max_bin, max_carton = 0, 0
    o0 = job.n_orders
    t_end = session.run_time() + 30.0
    k = 0
    while session.run_time() < t_end:
        _step(session)
        k += 1
        for lst, v in ((seq, job.state), (fryer, job.fryer["state"]), (tray, job.tray["state"])):
            if not lst or lst[-1] != v:
                lst.append(v)
        max_bin = max(max_bin, job.bin_count())
        if job.fill_k >= 0:
            max_carton = max(max_carton, job.carton_count())
        assert len(job.f_state) == FAST["n_fries"]  # the pool never grows
        if k % 4 == 0:
            for c in d.contact[:d.ncon]:
                g1, g2 = int(c.geom1), int(c.geom2)
                if g1 in fry_geoms or g2 in fry_geoms:
                    other = g2 if g1 in fry_geoms else g1
                    assert int(m.geom_bodyid[other]) not in fly_bodies
        if job.n_orders > o0 and job.tray["state"] == "home":
            break
    assert job.n_orders == o0 + 1 and job.work == job.n_orders
    # the fryer: cooked batch -> hold -> swing -> tip (the dump) -> return -> reload
    i = fryer.index("swing")
    assert fryer[i:i + 5] == ["swing", "tip", "untip", "return", "load"]
    assert job.n_batches >= 1 and job.n_fried >= 8 and max_bin >= 8
    # the station: take a carton, scoop, serve, take the next one
    first = seq.index("grab")
    assert seq[first + 1:first + 8] == ["scoop_wait", "scoop_reach", "scoop_dip", "scoop_drag",
                                        "scoop_lift", "scoop_carry", "scoop_tip"]
    assert "serve" in seq and seq[seq.index("serve") + 1] == "grab"
    assert tray[:5] == ["home", "slide", "ding", "out", "back"]
    # fries in the carton (settled with physics), counters
    assert job.n_cartons >= 1 and max_carton >= 2
    assert job.last_per_carton >= 2 and job.stats()["fries_per_carton"] >= 2
    assert job.n_picked >= job.last_per_carton and job.n_lost == 0
    assert any("ORDER UP" in s for s in msgs)
    # the served carton's fries were recycled into the pool
    assert not np.any(job.f_state == FC.RIDE) or job.tray["k"] >= 0
    # the ORDERS counter shows the count
    from fly_simulator.jobs.taste_tester import DIGITS, SEGS

    lit = [s for s, g in zip(SEGS, job.seg_gid[2]) if m.geom_matid[g] == job.mat["seg_on"]]
    assert set(lit) == set(DIGITS[job.n_orders % 10])
    assert job.n_falls == 0 and job.n_auto_recoveries == 0
    hud = "\n".join(job.hud_lines())
    assert "FRY COOK FLY" in hud and "salt level" in hud and "kinematic" in hud


def test_sneaked_fry_taste_fake_brain(cook):
    session, job, _ = cook
    link = _FakeBrain(session)
    session.brain = link
    try:
        session.reset("manual")
        _step(session, 200)
        # a spilled fry within reach of the left front leg
        i = int(np.flatnonzero(job.f_state == FC.PARK)[0])
        q = job.f_q[i]
        session.sim.data.qpos[q:q + 3] = (1.65, -0.05, 0.03)
        session.sim.data.qpos[q + 3:q + 7] = (0.7071, 0.0, 0.7071, 0.0)
        job._set_kinematic(i, FC.SPILL_REST, np.zeros(3), np.array([1.0, 0, 0, 0]))
        job.f_salt[i] = 1.6  # over-salted
        job.f_t[i] = session.sim.time - 1.0
        job.cfg.sneak_p = 1.0
        n0, peak = job.n_sneaked, 0.0
        t_end = session.run_time() + 12.0
        while session.run_time() < t_end and job.n_sneaked == n0:
            _step(session)
            link.update()
            peak = max(peak, job.proboscis)
        assert job.n_sneaked == n0 + 1 and job.f_state[i] == FC.PARK
        ev = [e for e in link.sent if e.kind == "taste"]
        assert len(ev) == 1
        assert ev[0].details["tastes"] == ["sugar", "bitter"] and "stand-in" in ev[0].details["label"]
        assert ev[0].duration_s == job.cfg.taste_s
        assert peak > 0.5  # the proboscis followed MN9
    finally:
        job.cfg.sneak_p = 0.0
        session.brain = None


def test_reset_mid_cycle_voids_and_continues(cook):
    session, job, _ = cook
    session.reset("manual")
    t_end = session.run_time() + 20.0
    while job.state != "scoop_carry" and session.run_time() < t_end:
        _step(session)
    assert job.state == "scoop_carry"
    v0 = job.n_voided
    session.reset("manual")
    assert job.n_voided == v0 + 1 and job.state == "settle"
    assert not np.any(np.isin(job.f_state, (FC.SCOOP, FC.DROP, FC.CARTON, FC.RIDE)))
    assert job.fill_k == -1 and job.tray["state"] == "home"
    t_end = session.run_time() + 12.0
    while job.state != "scoop_tip" and session.run_time() < t_end:
        _step(session)
    assert job.state == "scoop_tip"
