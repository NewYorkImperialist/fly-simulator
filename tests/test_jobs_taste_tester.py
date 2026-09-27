"""The taste tester job (fly_simulator/jobs/taste_tester.py): a visual-only scene,
the leg tap that sends the taste stimulus, the decision (a fake brain's MN9: sugar
-> approved, bitter -> rejected; the scripted fallback without a brain), the
conveyor's recycling of a fixed sample pool, stamps / bins, counters, the HUD and a
reset mid-cycle."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from fly_simulator.config import AppConfig
from fly_simulator.jobs import available_jobs, create_job_session, make_job
from fly_simulator.jobs.taste_tester import (END_K, N_STATIONS, STAMP_K, TASTE_K,
                                             TasteTesterJob, TasterStance)


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.logging.enabled = False
    cfg.render.width, cfg.render.height = 160, 106
    return cfg


class _FakeBrain:
    """Publishes a 0.1 s BrainState per window (``recent`` / ``latest``, with seq and
    sim_time like BrainLink). MN9 = 60 Hz while a sugar-only taste pulse is on, 5 Hz
    for sugar + bitter, 0 otherwise (the connectome's pattern, docs/TASTE.md)."""

    def __init__(self, session) -> None:
        self.session = session
        self.sent, self.stim_log = [], []
        self.recent: deque = deque(maxlen=30)
        self.latest = None
        self._seq = 0
        self._t = session.run_time()

    def send(self, ev, source=""):
        self.sent.append(ev)
        self.stim_log.append(ev)

    def mn9_at(self, t: float) -> float:
        for ev in self.sent:
            if ev.sim_time <= t <= ev.sim_time + ev.duration_s + 0.05:
                tastes = set(ev.details.get("tastes", ()))
                return 60.0 if tastes == {"sugar"} else (5.0 if "sugar" in tastes else 0.0)
        return 0.0

    def update(self):  # Session.after_physics calls it
        rt = self.session.run_time()
        while rt - self._t >= 0.1:
            self._t += 0.1
            self._seq += 1
            st = SimpleNamespace(seq=self._seq, sim_time=self._t,
                                 probes={"MN9": self.mn9_at(self._t - 0.05)})
            self.recent.append(st)
            self.latest = st


def _run_until(session, job, cond, max_s: float) -> None:
    t_end = session.run_time() + max_s
    while not cond() and session.run_time() < t_end:
        session.sim.step(50)
        session.after_physics()


@pytest.fixture(scope="module")
def tester():
    msgs: list[str] = []
    session, job = create_job_session("taste_tester", _cfg(), say=msgs.append)
    yield session, job, msgs
    session.brain = None
    session.sim.close()


def test_registry_and_config():
    assert "taste_tester" in available_jobs()
    job = make_job("taste_tester", {"approve_mn9_hz": 25.0, "n_samples": 8})
    assert isinstance(job, TasteTesterJob)
    assert job.cfg.approve_mn9_hz == 25.0 and job.cfg.n_samples == 8
    assert job.x_near == pytest.approx(job.cfg.station_x + job.cfg.belt_gap)
    assert job.station_y(TASTE_K) == pytest.approx(job.cfg.station_y)
    assert job.station_y(STAMP_K) < job.station_y(TASTE_K) < job.station_y(0)


def test_scene_is_visual_and_full_body(tester):
    session, job, _ = tester
    m = session.sim.model
    # every prop geom of the job is visual (no contacts): only the floor is touched
    names = [m.geom(g).name for g in range(m.ngeom)]
    job_geoms = [g for g, n in enumerate(names) if n.startswith("taste/")]
    assert len(job_geoms) > 100
    assert all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in job_geoms)
    # fixed pool of mocap samples, proboscis actuators, the stance is stationary
    assert len(job.sample_mocap) == job.cfg.n_samples and all(job.sample_mocap >= 0)
    assert len(job.prob_ids) == 2
    assert TasterStance.name in session.STATIONARY_ACTIONS
    assert session.sim.fly_mass == pytest.approx(1.024e-3, rel=0.02)


def test_scripted_fallback_without_brain(tester):
    session, job, msgs = tester
    session.brain = None
    session.reset("manual")
    job.set_sample(TASTE_K, "sugar")
    job.set_sample(TASTE_K - 1, "bitter")
    n0 = job.n_tasted
    _run_until(session, job, lambda: job.n_tasted >= n0 + 2, 12.0)
    assert job.n_tasted == n0 + 2
    a, b = list(job.last)[-2:]
    assert "SUGAR" in a and "APPROVED" in a and "[scripted]" in a
    assert "BITTER" in b and "REJECTED" in b
    assert job.n_touch_miss == 0  # the tapping leg reached the drop both times
    assert any("SCRIPTED" in ln for ln in job.hud_lines())
    assert job.n_falls == 0


def test_fake_brain_sugar_approved_bitter_rejected(tester):
    session, job, _ = tester
    link = _FakeBrain(session)
    session.brain = link
    try:
        session.reset("manual")
        job.set_sample(TASTE_K, "sugar", sugar_hz=150.0)
        job.set_sample(TASTE_K - 1, "bitter", bitter_hz=150.0)
        job.set_sample(TASTE_K - 2, "mixed", sugar_hz=150.0, bitter_hz=150.0)
        ok0, no0, t0 = job.n_approved, job.n_rejected, job.n_tasted
        prob_max = 0.0

        def cond():
            nonlocal prob_max
            prob_max = max(prob_max, job.proboscis) if job.n_tasted == t0 else prob_max
            return job.n_tasted >= t0 + 3
        _run_until(session, job, cond, 16.0)
        assert job.n_tasted == t0 + 3
        assert job.n_approved == ok0 + 1 and job.n_rejected == no0 + 2
        lines = list(job.last)[-3:]
        assert "SUGAR" in lines[0] and "APPROVED" in lines[0] and "MN9 60" in lines[0]
        assert "BITTER" in lines[1] and "REJECTED" in lines[1] and "MN9 0" in lines[1]
        assert "MIXED" in lines[2] and "REJECTED" in lines[2]
        # the stimuli: tarsal tap -> labellar GRN stand-in sets, the sample's rates
        sent = [e for e in link.sent if e.kind == "taste"]
        assert [tuple(e.details["tastes"]) for e in sent[-3:]] == [
            ("sugar",), ("bitter",), ("sugar", "bitter")]
        assert sent[-3].details["sugar_hz"] == 150.0 and "stand-in" in sent[-3].details["label"]
        assert sent[-3].side == "left" and sent[-3].duration_s == job.cfg.taste_s
        # the proboscis followed MN9 (PER) on the sugar sample
        assert prob_max > 0.8
        assert job.decision_source == "brain"
        assert any("REAL CONNECTOME" in ln for ln in job.hud_lines())
        assert job.per_kind["sugar"]["mn9_max"] >= 59.0
    finally:
        session.brain = None


def test_water_sends_no_stimulus_and_stim_log_is_bounded(tester):
    session, job, _ = tester
    link = _FakeBrain(session)
    link.stim_log.extend([None] * 450)
    session.brain = link
    try:
        session.reset("manual")
        job.set_sample(TASTE_K, "water")
        job.set_sample(TASTE_K - 1, "sugar")
        t0 = job.n_tasted
        _run_until(session, job, lambda: job.n_tasted >= t0 + 2, 12.0)
        w, s = list(job.last)[-2:]
        assert "WATER" in w and "no taste input" in w and "REJECTED" in w
        assert "SUGAR" in s and "APPROVED" in s
        assert len([e for e in link.sent if e.kind == "taste"]) == 1  # only the sugar
        assert len(link.stim_log) <= 400
    finally:
        session.brain = None


def test_conveyor_recycles_a_fixed_pool(tester):
    """Index moves without physics: the end sample goes to its bin, a new one enters
    under the hood; once the spares are used, the oldest binned sample is recycled."""
    session, job, _ = tester
    session.reset("manual")
    d = session.sim.data
    n = job.cfg.n_samples
    rec0 = job.n_recycled
    for step in range(20):
        # decide the sample at the tasting station alternately
        i = job.station[TASTE_K]
        if i is not None:
            job.samples[i].decision = "approved" if step % 2 else "rejected"
        job._begin_move()
        for j in list(job._flying):
            job.samples[j].fly["t0"] -= 10.0
        job._animate_machines()
        job._end_move()
        where = [s.where for s in job.samples]
        on_belt = [k for k in job.station if k is not None]
        assert len(set(on_belt)) == len(on_belt)  # no sample on two stations
        assert all(job.samples[k].where == "belt" for k in on_belt)
        assert where.count("bin") == len(job._bin_fifo)
        assert len(job.samples) == n
        assert np.all(np.isfinite(d.mocap_pos))
    assert job.n_recycled > rec0 + 5
    assert job.station[0] is not None and job.station[END_K] is not None
    # bins hold at most their slots; binned samples sit inside their bin
    for which, (ctr, (hx, hy, _)) in job._bins.items():
        inside = [i for i in job._bin_fifo if job.samples[i].bin == which]
        assert len(inside) <= 6
        for i in inside:
            p = d.mocap_pos[job.sample_mocap[i]]
            assert abs(p[0] - ctr[0]) < hx and abs(p[1] - ctr[1]) < hy
    session.reset("manual")


def test_stamper_stamps_the_card_and_counters(tester):
    session, job, _ = tester
    m = session.sim.model
    session.reset("manual")
    job.set_sample(TASTE_K, "sugar")
    i = job.station[TASTE_K]
    st0 = job.n_stamped
    _run_until(session, job, lambda: job.n_stamped > st0, 10.0)
    assert job.samples[i].stamped and job.samples[i].decision == "approved"
    assert m.geom_matid[job.card_gid[i]] == job.mat["card_sugar_approved"]
    st = job.stats()
    assert st["tasted"] == job.n_tasted and st["approved"] + st["rejected"] == st["tasted"]
    assert 0.0 <= st["accuracy"] <= 1.0
    # the tally board shows the approved count's last digit
    v = job.n_approved % 10
    from fly_simulator.jobs.taste_tester import DIGITS, SEGS

    lit = [s for s, g in zip(SEGS, job.seg_gid[0][2]) if m.geom_matid[g] == job.mat["seg_on_g"]]
    assert set(lit) == set(DIGITS[v])


def test_reset_mid_cycle_voids_and_continues(tester):
    session, job, _ = tester
    session.reset("manual")
    _run_until(session, job, lambda: job.state == "taste", 4.0)
    assert job.state == "taste"
    v0, t0 = job.n_voided, job.n_tasted
    session.reset("manual")
    assert job.n_voided == v0 + 1 and job.state == "settle"
    assert sum(k is not None for k in job.station) == TASTE_K + 1 <= N_STATIONS
    _run_until(session, job, lambda: job.n_tasted > t0, 6.0)
    assert job.n_tasted == t0 + 1


def test_keys_queue_the_next_sample(tester):
    session, job, msgs = tester
    session.reset("manual")
    assert job.handle_key("6")  # settle: the sample in front of the fly
    assert job.samples[job.station[TASTE_K]].kind == "bitter"
    assert not job.handle_key("q")
