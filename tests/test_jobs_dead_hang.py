"""The dead-hang job (fly_simulator/jobs/dead_hang.py): the fly hangs from a pull-up
bar by FlyGym's tarsal adhesion (no welds), re-grips, falls into the flytrap when the
grip gives out (counted CHOMP + explicit respawn), and the GF -> flinch rule."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest

from fly_simulator.actions.base import LEGS
from fly_simulator.config import AppConfig
from fly_simulator.jobs import JobRunner, available_jobs, create_job_session, make_job
from fly_simulator.jobs.dead_hang import DeadHangJob
from fly_simulator.terrain import FLY_BIT

# fatigue, slips and twitches off: the tests switch them on where needed
QUIET = dict(fatigue_per_s=0.0, cap_decay_per_s=0.0, slip_rate_per_s=0.0,
             twitch_every_s=0.0, regrip_below=-1.0, shadows=False)


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.logging.enabled = False
    cfg.render.width, cfg.render.height = 160, 106
    return cfg


def _run(session, job, seconds: float) -> None:
    r = JobRunner(session, job, headless=True, print_every_s=1e9)
    t_end = session.run_time() + seconds
    while session.run_time() < t_end:
        r.step_chunk()


@pytest.fixture(scope="module")
def hang():
    msgs: list[str] = []
    session, job = create_job_session("dead_hang", _cfg(), dict(QUIET), say=msgs.append)
    yield session, job, msgs
    session.sim.close()


def test_registry_and_config():
    assert "dead_hang" in available_jobs()
    job = make_job("dead_hang", {"bar_height": 12.0, "open_deg": 50.0})
    assert isinstance(job, DeadHangJob) and job.cfg.bar_height == 12.0
    assert job.rim_z_open() > job.cfg.trap_hinge_z


def test_props_and_no_welds(hang):
    session, job, _ = hang
    m = session.sim.model
    assert m.neq == 0  # no weld / connect constraints anywhere
    bar = m.geom("hang/bar").id
    assert m.geom_conaffinity[bar] & FLY_BIT  # the fly really touches the bar
    for nm in ("hang/lobe_far_vis", "hang/lobe_near_vis", "hang/cilia_far", "hang/pot"):
        g = m.geom(nm).id
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0  # visual only
    assert m.geom_conaffinity[m.geom("hang/pad").id] & FLY_BIT
    # the trap: hinges + a lunge slide, driven by position servos
    for a in ("hang/servo_far", "hang/servo_near", "hang/servo_lunge"):
        assert mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, a) >= 0
    assert job.ik_residual["lf"] < 0.2 and job.ik_residual["rf"] < 0.2
    assert all(r < 0.05 for r in job.brace_residual.values())  # the mid legs reach the bar
    assert m.geom_condim[bar] == job.cfg.bar_condim  # pad patch: torsional friction
    assert m.vis.map.znear == pytest.approx(job.znear)  # EternalJob.znear after compile


def test_hangs_at_full_grip(hang):
    session, job, _ = hang
    _run(session, job, 6.0)
    d, sim = session.sim.data, session.sim
    assert job.phase == "hang" and job.n_drops == 0
    assert session.actions.active_name == "dead_hang"
    z_rel = sim.thorax_position()[2] - job.cfg.bar_height
    assert -2.0 < z_rel < -0.8
    assert all(job.bar_contacts().values())
    # the adhesion actuators carry the grip: 40 uN per front leg at strength 1
    adh = d.actuator_force[session.actions.body.adh_ids]
    for leg in ("lf", "rf"):
        assert adh[LEGS.index(leg)] == pytest.approx(40.0 * job.strength[leg], rel=1e-3)
    assert not np.any(d.xfrc_applied)  # no external force on anything
    assert job.work > 5.5  # seconds on the bar


def test_fatigue_and_regrip(hang):
    session, job, _ = hang
    job.cfg.fatigue_per_s = 0.05
    s0 = dict(job.strength)
    _run(session, job, 1.0)
    for leg in ("lf", "rf"):
        assert job.strength[leg] == pytest.approx(s0[leg] - 0.05, abs=0.01)
    job.cfg.fatigue_per_s = 0.0
    job.strength["lf"] = 0.5
    cap0 = job.cap
    assert job._start_reach("lf", session.sim.time, "regrip")
    _run(session, job, 0.8)
    assert job.n_regrips == 1 and job.leg_state["lf"] == "grip"
    assert job.strength["lf"] > 0.5 and job.cap < cap0
    assert job.phase == "hang"
    # stress speeds fatigue up
    job.stress_level = 1.0
    assert job._stress_mult(job.cfg.stress_fatigue_gain) == pytest.approx(2.0)
    job.stress_level = 0.0


def test_twitch_moves_the_trap(hang):
    session, job, _ = hang
    assert job.twitch()
    _run(session, job, 0.06)
    d = session.sim.data
    assert d.qpos[job.q_lunge] > 0.5 * job.cfg.twitch_lunge
    assert job.trap_closedness() > 0.2
    _run(session, job, 1.0)
    assert job.trap_mode == "open" and job.trap_closedness() < 0.1
    assert job.n_twitches == 1


def test_flinch_instead_of_jump(hang):
    session, job, msgs = hang
    link = SimpleNamespace(latest=SimpleNamespace(brain_time=5.0, descending={"escape": 120.0}),
                           triggers=None, stim_log=[])
    n0 = job.n_flinches
    job._brain_tick(link)
    assert job.n_flinches == n0 + 1 and job.n_gf_bursts >= 1
    assert session.actions.active_name == "dead_hang"  # no jump
    job._brain_tick(link)  # same brain state: not counted twice
    assert job.n_flinches == n0 + 1
    assert any("FLINCH" in m for m in msgs)
    _run(session, job, 0.6)
    assert job.phase == "hang"
    lines = job.hud_lines()
    assert lines[0].startswith("DEAD HANG FLY") and any("GRIP [" in l for l in lines)
    assert any("CHOMPS" in l for l in lines)


def _brace_touching(session, job) -> bool:
    m, d = session.sim.model, session.sim.data
    fn = session.sim.fly_name
    ids = {m.geom(f"{fn}/{leg}_tarsus{k}").id for leg in job.cfg.brace_legs for k in (3, 4, 5)}
    bar = job.bar_gid
    return any((int(g1) == bar and int(g2) in ids) or (int(g2) == bar and int(g1) in ids)
               for g1, g2 in d.contact.geom[:d.ncon])


def test_one_arm_at_high_strength_survives_and_regrabs(hang):
    """A slip at 90 % grip: the other hand holds, the mid legs brace on the bar, the
    slipped leg re-grabs (both sides)."""
    session, job, _ = hang
    _run(session, job, 0.3)
    drops0 = job.n_drops
    for leg in ("lf", "rf"):
        assert job.phase == "hang" and all(s == "grip" for s in job.leg_state.values())
        n0, f0 = job.n_regrabs, job.n_regrab_fails
        job.strength = {"lf": 0.9, "rf": 0.9}
        job._slip(leg, session.sim.time)
        braced = False
        t0 = session.run_time()
        while job.leg_state[leg] != "grip" and session.run_time() - t0 < 4.0:
            _run(session, job, 0.05)
            braced |= _brace_touching(session, job)
        assert job.phase == "hang" and job.n_drops == drops0
        assert job.n_regrabs == n0 + 1 and job.n_regrab_fails == f0  # first try
        assert braced
        _run(session, job, 1.5)  # braces off again, both hands on
        assert job.phase == "hang" and job._brace_w < 0.05
        assert all(job.bar_contacts(tarsi_only=True).values())
    assert not np.any(session.sim.data.xfrc_applied)
    # a voluntary re-grip at high strength: weight shift, lift, put back
    job.strength = {"lf": 0.9, "rf": 0.72}
    assert job._start_reach("rf", session.sim.time, "regrip")
    _run(session, job, 0.8)
    assert job.leg_state["rf"] == "grip" and job.phase == "hang" and job.n_drops == drops0


def test_no_grip_falls_into_trap_and_respawns(hang):
    session, job, _ = hang
    streak = job.streak()
    # grip gone: adhesion 0 -> the tarsi slide off the bar (physics decides when)
    job.strength = {"lf": 0.0, "rf": 0.0}
    job.cap = 0.0
    t0 = session.run_time()
    while job.phase == "hang" and session.run_time() - t0 < 8.0:
        _run(session, job, 0.1)
    assert job.phase in ("falling", "chomped") and job.n_drops == 1
    assert job.best_streak >= streak
    while job.phase == "falling" and session.run_time() - t0 < 10.0:
        _run(session, job, 0.02)
    assert job.phase == "chomped" and job.n_chomps == 1 and job.trap_mode == "snap"
    _run(session, job, 0.3)
    assert job.trap_closedness() > 0.8  # snapped shut
    _run(session, job, job.cfg.chomp_hold_s)
    # explicit, counted respawn on the bar; the trap reopens slowly
    assert job.phase == "hang" and job.n_auto_recoveries == 1
    assert job.recovery_reasons == {"chomped": 1}
    assert job.trap_mode == "reopen" and job.strength["lf"] == 1.0
    z_rel = session.sim.thorax_position()[2] - job.cfg.bar_height
    assert -2.0 < z_rel < -0.8
    _run(session, job, 1.0)
    assert job.phase == "hang" and all(job.bar_contacts().values())
    st = job.stats()
    assert st["chomps"] == 1 and st["drops"] == 1


def test_fatigue_drains_to_a_fall(hang):
    """Fatigue on (fast), no re-grips: the grip drains until the fly falls; it was
    tired when it fell (the fall comes from fatigue, not from a strong grip)."""
    session, job, _ = hang
    while job.phase != "hang":
        _run(session, job, 0.1)
    _run(session, job, 0.5)
    c = job.cfg
    c.fatigue_per_s, c.slip_rate_per_s = 0.12, 0.045
    drops0 = job.n_drops
    t0 = session.run_time()
    while job.n_drops == drops0 and session.run_time() - t0 < 14.0:
        _run(session, job, 0.1)
    c.fatigue_per_s, c.slip_rate_per_s = 0.0, 0.0
    assert job.n_drops == drops0 + 1
    assert job.last_streak > 4.0  # it hung on for a while first
    assert job.last_fall_grip < 0.45  # tired
    assert sum(job.fall_causes.values()) == job.n_drops
    st = job.stats()
    assert st["fall_causes"] == job.fall_causes and st["mean_fall_grip"] is not None
