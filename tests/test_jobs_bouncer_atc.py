"""The bouncer and air_traffic jobs (fly_simulator/jobs/bouncer.py, air_traffic.py):
they build (guests / planes are kinematic and touch nothing, no unshadowed spot
light), a guest's step-up is a loom on both eyes, a GF burst makes the bouncer
flinch (never jump), a habituating GF (synthetic) makes the flinches die out, the
scripted flinch habituates without a brain, the admission rule, the night / shift
cycle; planes are cleared when the fly faces them, the brain's DNa01 / DNa02 turn
map and the mirror sink, holding / near misses / diversion, a reset mid-cycle."""

from __future__ import annotations

import math
from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest

from fly_simulator.config import AppConfig
from fly_simulator.jobs import available_jobs, create_job_session
from fly_simulator.jobs.air_traffic import AirTrafficJob, _MirrorSink
from fly_simulator.jobs.bouncer import BouncerJob, guest_response, loom_progress, lunge
from fly_simulator.jobs.geometry import wrap_angle
from fly_simulator.terrain import FLY_BIT


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.logging.enabled = False
    cfg.render.width, cfg.render.height = 160, 106
    return cfg


def _run(session, seconds: float) -> None:
    t_end = session.run_time() + seconds
    while session.run_time() < t_end:
        session.sim.step(100)
        session.after_physics()


def _no_unshadowed_spots(m: mj.MjModel) -> None:
    for i in range(m.nlight):
        if m.light_type[i] == mj.mjtLightType.mjLIGHT_SPOT:
            assert m.light_castshadow[i], "a spot light without shadows blacks out pixels on macOS"


class FakeLink:
    """Just enough of a BrainLink for the jobs: ``latest`` (a BrainState-like
    namespace), ``send`` (recorded), ``_run_time``, ``update`` (no-op)."""

    def __init__(self, session) -> None:
        self.session = session
        self.sent: list = []
        self.latest = None
        self.triggers = None
        self.stim_log: list = []
        self.cfg = SimpleNamespace(habituation=False, habituation_config={})
        self.brain = None

    def send(self, ev, source: str = "", event_type: str = "brain_stim") -> None:
        self.sent.append(ev)

    def _run_time(self, sim_time=None) -> float:
        return float(self.session.run_time())

    def update(self) -> None:
        pass

    def summary(self) -> dict:
        return {}

    def config_dict(self) -> dict:
        return {}

    def state(self, t: float, **descending) -> None:
        self.latest = SimpleNamespace(brain_time=t, descending=dict(descending), habituation=None)


# ---------------------------------------------------------------------------
# bouncer
# ---------------------------------------------------------------------------


def test_loom_profiles():
    assert lunge(0.0) == 0.0 and lunge(1.0) == pytest.approx(1.0)
    us = np.linspace(0, 1, 50)
    s = [loom_progress(u, 5.0, 1.0) for u in us]
    assert s[0] == 0.0 and s[-1] == pytest.approx(1.0) and np.all(np.diff(s) >= 0)
    # 1 / distance grows linearly: the step-up is fast while far, slow when close
    d = [5.0 - 4.0 * v for v in s]
    inv = 1.0 / np.array(d)
    assert np.allclose(np.diff(inv), np.diff(inv)[0], rtol=1e-6)
    r = guest_response()
    assert r.lc4_v0 < 10 and r.lplc2_theta0 >= 30  # LC4 for every guest, LPLC2 only for big looms


@pytest.fixture(scope="module")
def bnc():
    msgs: list[str] = []
    session, job = create_job_session("bouncer", _cfg(), {"rowdy_p": 0.0, "shadows": False},
                                      say=msgs.append)
    yield session, job, msgs
    session.sim.close()


def test_bouncer_builds(bnc):
    session, job, _ = bnc
    m = session.sim.model
    assert "bouncer" in available_jobs() and isinstance(job, BouncerJob)
    assert m.neq == 0
    # the guests are NeuroMechFly copies that touch nothing (posed, kinematic)
    for k in range(job.cfg.n_guests):
        root = m.body_rootid[m.body(f"guest{k}/c_head").id]
        gs = [g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == root]
        assert len(gs) > 20
        assert all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in gs)
        assert m.body_mocapid[m.body(f"bnc/mount{k}").id] >= 0
    assert not any(m.actuator(i).name.startswith("guest") for i in range(m.nu))
    assert m.body_mocapid[m.body("bnc/gate").id] >= 0
    _no_unshadowed_spots(m)
    assert m.vis.map.znear == pytest.approx(0.05)


def test_step_up_is_a_loom_and_guests_cycle(bnc):
    session, job, _ = bnc
    _run(session, 7.0)
    assert job.guest_no >= 2 and job.n_admitted >= 1
    assert job.work == job.n_admitted + job.n_turned
    # the step-up drove LC4 on both eyes (computed without a brain: sink None)
    hist = [h for h in job.vision.history if h[1] == "guest"]
    lc4 = {eye: max(e.lc4_hz for _, _, ey, e in hist if ey == eye) for eye in ("left", "right")}
    lplc2 = max(e.lplc2_hz for *_, e in hist)
    assert lc4["left"] > 150 and lc4["right"] > 150
    assert lplc2 < 60  # a normal guest barely reaches LPLC2's size range
    assert job.fly_down() is False and job.n_falls == 0
    # the queue: guests are at their slots, one pool of guests is recycled
    states = {g.state for g in job.guests}
    assert "queue" in states or "arriving" in states
    hud = "\n".join(job.hud_lines())
    assert "NIGHT 1" in hud and "FLINCHES" in hud and "scripted flinch" in hud


def test_gf_burst_flinches_never_jumps(bnc):
    session, job, msgs = bnc
    link = FakeLink(session)
    link.state(1.0, escape=120.0)
    n0 = job.n_flinches
    job._flinch_ready = -1e9
    job._brain_tick(link)
    assert job.n_flinches == n0 + 1 and job.n_gf_bursts >= 1
    job._brain_tick(link)  # the same brain state: not counted twice
    assert job.n_flinches == n0 + 1
    _run(session, 0.1)
    assert session.actions.active_name == "bouncer_stance"  # a flinch, not a jump
    assert job._stance.flinch > 0.5
    _run(session, 0.5)
    assert job._stance.flinch == 0.0 and not job.fly_down()


def test_habituating_gf_stops_the_flinches(bnc):
    """A synthetic habituating brain: every guest's step-up drives the GF to 150 Hz
    x an efficacy that halves per guest (the real depression is measured in the
    docs); the flinches stop once the GF stays below the 60 Hz rule."""
    session, job, _ = bnc
    link = FakeLink(session)
    eff = 1.0
    seen = set()
    bt = 10.0
    flinched = []
    no_end = job.guest_no + 5
    session.brain = link  # the brain path (no scripted flinches)
    job._flinch_ready = -1e9
    try:
        while job.guest_no < no_end:
            session.sim.step(100)
            session.after_physics()
            no = job.guest_no
            if job.phase == "stepping" and no not in seen:
                seen.add(no)
                n0 = job.n_flinches
                bt += 1.0
                link.state(bt, escape=150.0 * eff)
                job._brain_tick(link)
                flinched.append(job.n_flinches > n0)
                eff *= 0.5
                job._flinch_ready = -1e9
    finally:
        session.brain = None
    assert flinched[:2] == [True, True] and not any(flinched[2:])
    # the GF peak per guest is logged (constant memory)
    peaks = [e[3] for e in job.gf_log][-4:]
    assert peaks[0] > 60 and max(peaks[1:]) < 60


def test_scripted_flinch_habituates_and_rowdy_dishabituates():
    job = BouncerJob()
    p0 = job.script_flinch_p()
    ps = []
    for _ in range(12):
        job.script_h += 1.0
        ps.append(job.script_flinch_p())
    assert p0 > 0.9 and ps[-1] < 0.1 and all(a >= b for a, b in zip(ps, ps[1:]))
    h = job.script_h
    job.script_h *= 1.0 - job.cfg.script_dishab  # what a chest bump does
    assert job.script_h < h and job.script_flinch_p() > ps[-1] + 0.2
    assert job.script_flinch_p(rowdy=True) == job.cfg.script_p0


def test_rowdy_turned_away_bumps_and_reset_mid_cycle():
    session, job = create_job_session("bouncer", _cfg(), {"rowdy_p": 1.0, "rowdy_min_gap": -1,
                                                          "shadows": False})
    try:
        link = FakeLink(session)
        session.brain = link
        _run(session, 2.4)
        assert job.n_rowdy >= 1 and job.n_bumps >= 1
        assert any(ev.kind == "shove" for ev in link.sent)  # the chest bump -> dishabituation
        assert job.n_turned >= 1 and job.n_admitted == 0  # rule: rowdy guests are turned away
        # a reset in the middle of a cycle: the fly respawns, the door keeps working
        n0 = job.guest_no
        session.reset("manual")
        _run(session, 3.5)
        assert job.guest_no > n0 and session.actions.active_name == "bouncer_stance"
    finally:
        session.brain = None
        session.sim.close()


def test_capacity_and_new_night():
    session, job = create_job_session("bouncer", _cfg(), {"rowdy_p": 0.0, "capacity": 1, "shift_s": 6.5,
                                                          "closed_s": 0.5, "shadows": False})
    try:
        _run(session, 6.3)
        assert job.n_admitted == 1 and job.n_turned >= 1  # full after one guest
        assert sum(job.hour_guests) == job.night_guests
        _run(session, 1.0)
        assert job.night == 2 and job.phase != "closed"  # a new night: counters reset
        assert job.night_guests <= 1 and job.n_admitted + job.n_turned >= 2
    finally:
        session.sim.close()


# ---------------------------------------------------------------------------
# air_traffic
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def atc():
    session, job = create_job_session("air_traffic", _cfg(), {"spawn_every_s": 3.0})
    yield session, job
    session.sim.close()


def test_atc_builds(atc):
    session, job = atc
    m = session.sim.model
    assert "air_traffic" in available_jobs() and isinstance(job, AirTrafficJob)
    for k in range(job.cfg.n_planes):
        b = m.body(f"atc/plane{k}").id
        assert m.body_mocapid[b] >= 0
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b:
                assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
    _no_unshadowed_spots(m)
    assert m.vis.map.zfar == pytest.approx(300.0)
    # the swivel stool: a hinge with a velocity servo; its top touches only the fly
    j = m.joint("atc/swivel")
    assert m.jnt_type[j.id] == mj.mjtJoint.mjJNT_HINGE
    assert m.actuator_trnid[m.actuator("atc/swivel_motor").id][0] == j.id
    top = m.geom("atc/swivel_top").id
    assert m.geom_contype[top] == 0 and m.geom_conaffinity[top] == FLY_BIT


def test_atc_stool_turns_the_fly(atc):
    """The swivel stool turns the standing fly (contact + adhesion), in place."""
    session, job = atc
    job._next_spawn = 1e9
    job.housekeeping = False
    upd = job._update_turn
    job._update_turn = lambda t: None
    try:
        job.turn = 1.0
        _run(session, 0.3)
        h0, p0 = session.sim.heading(), session.sim.thorax_position().copy()
        _run(session, 0.5)
        dh = math.degrees(wrap_angle(session.sim.heading() - h0))
        assert 40.0 < dh < 110.0  # ~150 deg/s to the left
        assert np.linalg.norm(session.sim.thorax_position()[:2] - p0[:2]) < 0.3
        assert session.actions.active_name == "bouncer_stance" and not job.fly_down()
    finally:
        job.turn = 0.0
        job._update_turn = upd
        job.housekeeping = True
        job._next_spawn = session.sim.time + 1.0


def test_atc_scripted_clears_planes(atc):
    session, job = atc
    _run(session, 9.0)
    assert job.control_mode() == "script"
    assert job.n_cleared >= 1 and job.work == job.n_cleared
    assert job.mean_response_s() is not None and 0.0 < job.mean_response_s() < 5.0
    assert job.n_first_trials == 0 or job.n_first_ok >= 1
    assert not job.fly_down() and job.n_falls == 0
    assert any(e[0] == "cleared" for e in job.events)
    hud = "\n".join(job.hud_lines())
    assert "CLEARED" in hud and "holding" in hud and "near misses" in hud


def test_atc_cleared_when_facing(atc):
    session, job = atc
    job.cfg.control = "none"  # no turning: only the geometry decides
    job._next_spawn = 1e9
    for p in job.planes:
        p.state, p.path = "hidden", None
    job.holding.clear()
    job.calling = None
    job.housekeeping = False
    _run(session, 0.2)
    h = job.sim.heading()
    th = job.sim.thorax_position()
    p = job.planes[0]
    p.side, p.flight = 1, "TEST1"
    for rel, cleared in ((math.radians(60.0), False), (0.0, True)):
        p.call_pt = np.array([th[0] + 16 * math.cos(h + rel), th[1] + 16 * math.sin(h + rel), 4.0])
        p.state, p.path, p.t_called, p.phase0 = "calling", None, job.sim.time, 0.0
        p.pos = p.call_pt.copy()
        job.calling = 0
        job._aligned_since = None
        n0 = job.n_cleared
        _run(session, 0.6)
        assert (job.n_cleared > n0) is cleared
        if not cleared:
            assert abs(math.degrees(job.bearing_error())) > job.cfg.tolerance_deg
    assert job.planes[0].state == "landing"
    job.calling = None
    job.cfg.control = "auto"
    job.housekeeping = True


def test_atc_brain_turn_map_and_mirror(atc):
    session, job = atc
    assert job.turn_from_dn(60.0, 0.0) > 0.9 and job.turn_from_dn(0.0, 60.0) < -0.9
    assert job.turn_from_dn(0.0, 0.0) == 0.0
    job.hk = None
    job.turn = 1.0  # + = left: the stool turns counter-clockwise
    assert job.swivel_rate() == pytest.approx(math.radians(job.cfg.turn_max_dps))
    job.turn = -0.4
    assert job.swivel_rate() == pytest.approx(-math.radians(job.cfg.turn_max_dps) * 0.4 / job.cfg.turn_full)
    job.turn = 0.0
    assert job.swivel_rate() == 0.0
    # the brain path reads DNa01 / DNa02 from the brain state (fake link)
    link = FakeLink(session)
    session.brain = link
    try:
        job.set_control("brain")
        p = job.planes[1]
        th = job.sim.thorax_position()
        h = job.sim.heading()
        p.call_pt = np.array([th[0] + 16 * math.cos(h + 1.2), th[1] + 16 * math.sin(h + 1.2), 4.0])
        p.state, p.path, p.t_called, p.phase0, p.flight = "calling", None, job.sim.time, 0.0, "TEST2"
        p.pos = p.call_pt.copy()
        job.calling = 1
        link.state(1.0, turn_L=50.0, turn_R=0.0)
        _run(session, 0.1)
        assert job.turn > 0.9
        # the plane is on the left: LC10a events went out for the left eye
        ev = [e for e in link.sent if e.details.get("set") == "LC10a"]
        assert ev and all(e.side == "left" for e in ev)
        link.sent.clear()
        job.set_control("mirror")
        _run(session, 0.1)
        ev = [e for e in link.sent if e.details.get("set") == "LC10a"]
        assert ev and all(e.side == "right" for e in ev)  # the eyes swapped
        from fly_simulator.brain.schema import StimulusEvent

        _MirrorSink(link).send(StimulusEvent("manual", side="left", details={"set": "LC10a"}), source="t")
        assert link.sent[-1].side == "right"
    finally:
        job.calling = None
        job.planes[1].state = "hidden"
        session.brain = None
        job.set_control("auto")


def test_atc_holding_near_miss_divert_and_reset():
    session, job = create_job_session("air_traffic", _cfg(), {"spawn_every_s": 1e9, "control": "none",
                                                              "divert_s": 1.5, "crowded": 2})
    try:
        job._next_spawn = 1e9
        _run(session, 0.3)
        planes = [job.spawn(side=s) for s in (1, -1, 1, -1)]
        assert all(p is not None for p in planes)
        assert job.calling == planes[0].k and len(job.holding) == 3 and job.max_holding == 3
        assert job.n_near_miss == 1  # arrival 4 found 2 (= crowded) already holding
        _run(session, 3.0 + job.cfg.call_fly_s)
        assert job.n_diverted >= 1  # "none" never turns: the calling plane diverts
        assert job.calling is not None and len(job.holding) <= 2  # the next one was called
        n = job.n_diverted
        session.reset("manual")  # mid-cycle
        _run(session, 2.0)
        assert job.n_diverted >= n and not job.fly_down()
    finally:
        session.sim.close()
