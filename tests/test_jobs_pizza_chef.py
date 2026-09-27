"""The pizza chef job (fly_simulator/jobs/pizza_chef.py): the scene and its contact
scheme, one full (fast-config) pizza cycle in the right order (kneading flattens the
dough, the toss is a real free-body flight, the toppings pool lands and is recycled,
the oven bakes, 8 slices, served, counters / chalkboard), the slow-motion edit and
its label, a fake brain's MN9 on the sauce taste, and a reset mid-cycle."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fly_simulator.config import AppConfig
from fly_simulator.jobs import available_jobs, create_job_session, make_job
from fly_simulator.jobs.pizza_chef import (N_WEDGES, PZ_BIT, ChefGroom, ChefStance,
                                           PizzaChefJob)
from fly_simulator.terrain import FLY_BIT

FAST = dict(n_presses=4, press_s=0.3, bake_s=1.0, sauce_s=0.8, pour_s=(0.2, 0.12, 0.1),
            n_cheese=12, n_pepperoni=4, n_basil=3, toss_sigma=0.0, toss_wobble=0.0,
            bonus_toss_p=0.0)

ORDER = ["dough_in", "knead", "toss_prep", "toss_air", "toss_land", "sauce", "taste", "pour",
         "peel_in", "carry_in", "set_down", "bake", "fetch", "carry_out", "peel_home", "slice",
         "serve", "dough_in"]


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.logging.enabled = False
    cfg.render.width, cfg.render.height = 160, 106
    return cfg


class _FakeBrain:
    """Publishes a BrainState every 0.1 s (``latest`` / ``recent`` with seq and
    sim_time); MN9 = 60 Hz while a sugar taste pulse is on, else 0."""

    def __init__(self, session) -> None:
        self.session = session
        self.sent, self.stim_log = [], []
        self.latest = None
        self.recent = []
        self._t = session.run_time()
        self._seq = 0

    def send(self, ev, source=""):
        self.sent.append(ev)
        self.stim_log.append(ev)

    def update(self):
        rt = self.session.run_time()
        while rt - self._t >= 0.1:
            self._t += 0.1
            self._seq += 1
            on = any(ev.sim_time <= self._t <= ev.sim_time + ev.duration_s + 0.05
                     and "sugar" in ev.details.get("tastes", ()) for ev in self.sent)
            self.latest = SimpleNamespace(seq=self._seq, sim_time=self._t,
                                          probes={"MN9": 60.0 if on else 0.0})


def _step(session, n: int = 50) -> None:
    session.sim.step(n)
    session.after_physics()


@pytest.fixture(scope="module")
def chef():
    msgs: list[str] = []
    session, job = create_job_session("pizza_chef", _cfg(), FAST, say=msgs.append)
    yield session, job, msgs
    session.brain = None
    session.sim.close()


def test_registry_and_config():
    assert "pizza_chef" in available_jobs()
    job = make_job("pizza_chef", {"bake_s": 3.0, "n_cheese": 10})
    assert isinstance(job, PizzaChefJob)
    assert job.cfg.bake_s == 3.0 and job.cfg.n_cheese == 10
    # dough stages flatten: radius grows, height shrinks, squarer profile
    shapes = [job.stage_shape(k) for k in range(job.cfg.n_stages)]
    assert all(b[0] > a[0] and b[1] < a[1] and b[2] > a[2] for a, b in zip(shapes, shapes[1:]))
    assert shapes[-1][0] <= job.cfg.pizza_r


def test_scene_contacts_and_pools(chef):
    session, job, _ = chef
    m = session.sim.model
    names = [m.geom(g).name for g in range(m.ngeom)]
    job_geoms = [g for g, n in enumerate(names) if n.startswith("pizza/")]
    assert len(job_geoms) > 250
    for g in job_geoms:
        ct, ca = int(m.geom_contype[g]), int(m.geom_conaffinity[g])
        # nothing of the job ever touches the fly
        assert not (ct & FLY_BIT) and not (ca & FLY_BIT), names[g]
    colliding = {names[g] for g in job_geoms if m.geom_contype[g] or m.geom_conaffinity[g]}
    # the colliders and free bodies (toss dough, the toppings) are the only live geoms
    assert "pizza/board_col" in colliding
    assert all(n.endswith(("_col", "_g")) for n in colliding)
    # fixed pools
    assert len(job.t_state) == FAST["n_cheese"] + FAST["n_pepperoni"] + FAST["n_basil"]
    assert len(job.coin_mocap) == job.cfg.n_coins
    assert all(len(v) == N_WEDGES for v in job.wedge_gid.values())
    # the fly: normal mass (the toque is massless), stationary actions, proboscis
    assert session.sim.fly_mass == pytest.approx(1.024e-3, rel=0.02)
    assert ChefStance.name in session.STATIONARY_ACTIONS
    assert ChefGroom.name in session.STATIONARY_ACTIONS
    assert len(job.prob_ids) == 2


def test_one_pizza_in_order(chef):
    """A full fast cycle: the states in order; the dough flattens press by press; the
    toss flies (a free body, real gravity) and lands on the board; the toppings land
    on the pizza and are recycled after serving; the pizza goes into the oven and
    bakes; 8 slices; served; counters and the chalkboard."""
    session, job, msgs = chef
    session.brain = None
    session.reset("manual")
    m, d = session.sim.model, session.sim.data
    seq, stages = [], []
    toss_zmax, max_live, slow = 0.0, 0, []
    in_oven = False
    spread = 0.0
    n0 = job.n_served
    t_end = session.run_time() + 40.0
    while session.run_time() < t_end:
        _step(session)
        if not seq or seq[-1] != job.state:
            seq.append(job.state)
        if job.state == "knead" and (not stages or stages[-1] != job.stage):
            stages.append(job.stage)
        if job.state == "toss_air":
            toss_zmax = max(toss_zmax, float(d.qpos[job.toss_q + 2]))
            slow.append(job.time_scale(1 / 30))
        max_live = max(max_live, int(np.sum(job.t_state == 1)))
        if job.state == "bake":
            dist = np.hypot(*(job.pz_pos[:2] - job.oven_c))
            in_oven = in_oven or (dist < job.cfg.oven_R and job.pz_pos[2] > job.cfg.hearth_z)
        if job.state == "serve":
            spread = max(spread, float(np.min(np.linalg.norm(job.wedge_off, axis=1))))
        if job.state == "peel_home":
            assert m.geom_matid[job.wedge_gid["base"][0]] == job.mat["crust"]  # baked
            assert (job.t_state == 2).sum() >= 0.8 * len(job.t_state)  # toppings on it
        if job.n_served > n0 and job.state == "dough_in":
            break
    assert job.n_served == n0 + 1
    first = seq.index("dough_in")
    assert seq[first:first + len(ORDER)] == ORDER
    # kneading: stage 0 .. flat, monotonic, one flour-dusted press per step
    assert stages[0] == 0 and stages[-1] == job.cfg.n_stages - 1 and stages == sorted(stages)
    # the toss: a real ballistic flight (launch set by the job), a perfect landing
    assert toss_zmax > job.cfg.board_top + 0.8 * job.cfg.toss_height
    assert job.n_tosses >= 1 and job.n_perfect == job.n_tosses and job.n_toss_lost == 0
    assert "PERFECT TOSS" in job.last_toss
    assert min(slow) == pytest.approx(job.cfg.toss_slowmo)
    # toppings: poured, landed on the pizza, recycled (parked) after serving
    assert job.n_dropped == len(job.t_state) and job.n_on_pizza >= 0.8 * len(job.t_state)
    assert 0 < max_live <= len(job.t_state)
    assert np.all(job.t_state == 0) and np.all(m.body_gravcomp[job.t_bid] == 1.0)
    assert np.all(m.geom_contype[job.t_col] == 0) and np.all(m.geom_conaffinity[job.t_col] == 0)
    # the oven cycle and the slices
    assert in_oven and job.n_bakes == 1
    assert job.n_slices == N_WEDGES and spread > 0.03  # every slice parted from the rest
    # counters, tips, chalkboard ("1" in the last digit of the served display)
    assert job.tips_cents >= 50 and job.work == job.n_served
    from fly_simulator.jobs.taste_tester import DIGITS, SEGS

    lit = [s for s, g in zip(SEGS, job.seg_gid[0][2]) if m.geom_matid[g] == job.mat["seg_on"]]
    assert set(lit) == set(DIGITS[job.n_served % 10])
    assert job.n_falls == 0 and job.n_auto_recoveries == 0
    st = job.stats()
    for k in ("served", "tosses", "perfect_tosses", "slices", "tips_cents", "kneads"):
        assert k in st
    assert any("SERVED pizza" in s for s in msgs)
    hud = "\n".join(job.hud_lines())
    assert "PIZZA CHEF FLY" in hud and "tips $" in hud and "kinematic" in hud


def test_slowmo_label_and_identity(chef):
    session, job, _ = chef
    frame = np.zeros((106, 160, 3), np.uint8)
    job._last_scale = 1.0
    assert job.post_process(frame, 0.0) is frame
    job._last_scale = 0.1
    out = job.post_process(frame, 0.0)
    assert out is not frame and out.max() > 0  # "SLOW MOTION x0.10 (edit, not physics)"
    job._last_scale = 1.0
    job._scale = 1.0


def test_fake_brain_sauce_taste(chef):
    session, job, _ = chef
    link = _FakeBrain(session)
    session.brain = link
    try:
        session.reset("manual")
        n0 = job.n_tastes
        peak = 0.0
        t_end = session.run_time() + 15.0
        while session.run_time() < t_end and not (job.n_tastes > n0 and job.state == "pour"):
            _step(session)
            peak = max(peak, job.proboscis)
        assert job.n_tastes == n0 + 1
        sent = [e for e in link.sent if e.kind == "taste"]
        assert len(sent) == 1
        ev = sent[0]
        assert ev.details["tastes"] == ["sugar"] and "stand-in" in ev.details["label"]
        assert ev.side == "left" and ev.duration_s == job.cfg.taste_s
        assert job.mn9_peak == pytest.approx(60.0)
        assert "MN9 peak 60" in job.taste_line and "real connectome" in job.taste_line
        assert peak > 0.6  # the proboscis followed MN9
    finally:
        session.brain = None


def test_reset_mid_cycle_voids_and_continues(chef):
    session, job, _ = chef
    session.reset("manual")
    t_end = session.run_time() + 30.0
    while job.state != "sauce" and session.run_time() < t_end:
        _step(session)
    assert job.state == "sauce"
    v0 = job.n_voided
    session.reset("manual")
    assert job.n_voided == v0 + 1 and job.state == "settle"
    assert np.all(job.t_state == 0) and not job._toss_live
    assert np.all(session.sim.model.geom_contype[job.t_col] == 0)
    assert session.sim.model.geom_contype[job.top_col] == 0
    t_end = session.run_time() + 5.0
    while job.state in ("settle", "dough_in") and session.run_time() < t_end:
        _step(session)
    assert job.state == "knead"
    assert PZ_BIT == 64
