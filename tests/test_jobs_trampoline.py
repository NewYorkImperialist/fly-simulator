"""The trampoline job (fly_simulator/jobs/trampoline.py): the scene (a mat on a slide
joint held up by tendon springs, touched through FlyGym-ground contact pairs), the
mat is springy (deflects under load and rebounds), bouncing with the real Jump action
pumps the height above a single jump and above the same bounces on a rigid mat, the
counters / scoreboard / ruler markers, the edit-effect slow motion, a trick attempt,
and the counted recoveries after a crash landing and a fall off the trampoline."""

from __future__ import annotations

import math

import mujoco as mj
import numpy as np
import pytest

from fly_simulator.actions.jump import Jump
from fly_simulator.config import AppConfig
from fly_simulator.jobs import available_jobs, create_job_session, make_job
from fly_simulator.jobs.taste_tester import DIGITS, SEGS
from fly_simulator.jobs.trampoline import (BODY_LENGTH_MM, P, BounceJump, MatStance,
                                           TrampolineJob)

QUIET = dict(timing_jitter_ms=0.0, misstep_p=0.0, trick_p=0.0)


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


def _heights(job) -> list[float]:
    out: list[float] = []
    orig = job._touchdown

    def td(t, body, _orig=orig):
        _orig(t, body)
        out.append(job.last_height)

    job._touchdown = td
    return out


@pytest.fixture(scope="module")
def bouncing():
    msgs: list[str] = []
    session, job = create_job_session("trampoline", _cfg(), QUIET, say=msgs.append)
    hs = _heights(job)
    _run(session, 4.0)
    yield session, job, hs, msgs
    session.sim.close()


def test_registry_and_config():
    assert "trampoline" in available_jobs()
    job = make_job("trampoline", {"spring_k": 12.0, "slowmo": 1.0})
    assert isinstance(job, TrampolineJob)
    assert job.cfg.spring_k == 12.0 and job.cfg.slowmo == 1.0


def test_scene_mat_springs_and_contacts(bouncing):
    session, job, _, _ = bouncing
    m = session.sim.model
    j = m.joint(P + "mat_slide")
    assert j.type[0] == mj.mjtJoint.mjJNT_SLIDE and m.dof_damping[j.dofadr[0]] > 0
    # the springs are real tendon springs between the frame and the mat
    springs = [m.tendon(P + f"spring{i}") for i in range(job.cfg.n_springs)]
    assert all(m.tendon_stiffness[t.id] > 0 for t in springs)
    assert np.all(session.sim.data.ten_length[[t.id for t in springs]] > 0)
    # the collider touches only the fly, through explicit pairs (ground parameters)
    g = m.geom(P + "mat_plate").id
    assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
    pairs = [i for i in range(m.npair) if g in (m.pair_geom1[i], m.pair_geom2[i])]
    assert len(pairs) == 55
    assert np.allclose(m.pair_solref[pairs[0]], [2e-4, 1.0])
    # every other job geom is visual (no contacts)
    for gi in range(m.ngeom):
        if m.geom(gi).name.startswith(P) and gi != g:
            assert m.geom_contype[gi] == 0 and m.geom_conaffinity[gi] == 0, m.geom(gi).name
    assert MatStance.name in session.STATIONARY_ACTIONS


def test_mat_is_springy():
    session, job = create_job_session("trampoline", _cfg(), dict(QUIET, settle_s=1e9),
                                      say=lambda s: None)
    try:
        sim, d = session.sim, session.sim.data
        _run(session, 0.3)  # the fly stands on the mat (no jumping)
        rest = job.mat_q()
        assert -0.4 < rest < -0.1  # the standing fly's weight sinks it 0.1-0.4 mm
        d.qfrc_applied[job.v_mat] = -50.0  # press with 5 body weights
        _run(session, 0.1)
        pressed = job.mat_q()
        assert pressed < rest - 0.2
        d.qfrc_applied[job.v_mat] = 0.0
        vmax = 0.0
        for _ in range(60):
            sim.step(10)
            vmax = max(vmax, job.mat_v())
        assert vmax > 20.0  # it springs back up (mm/s)
        _run(session, 0.3)
        assert abs(job.mat_q() - rest) < 0.05
    finally:
        session.sim.close()


def test_bouncing_pumps_height_with_the_real_jump(bouncing):
    session, job, hs, _ = bouncing
    assert isinstance(job._jump_action, BounceJump) and isinstance(job._jump_action, Jump)
    assert job.n_bounces >= 25 and job.n_crashes == 0 and job.n_off == 0
    assert job.n_auto_recoveries == 0
    first = hs[0]  # the first jump from standing (long mode)
    steady = float(np.mean(hs[6:20]))
    assert steady > first + 0.6, (first, steady)
    assert job.best_height > 4.0
    # the aids stay within their caps; the steering force is horizontal only
    assert job.max_steer_bw <= job.cfg.steer_max_bw + 1e-9
    assert job.max_att_torque <= job.cfg.att_max + 1e-9
    assert job.mat_min_seen < -0.4  # the mat is pressed well below its standing level


def test_bounces_on_a_rigid_mat_are_lower(bouncing):
    _, _, hs_mat, _ = bouncing
    session, job = create_job_session("trampoline", _cfg(), dict(QUIET, rigid_mat=True),
                                      say=lambda s: None)
    try:
        hs = _heights(job)
        _run(session, 2.5)
        assert job.mat_q() == 0.0 and job.q_mat is None
        assert len(hs) >= 8
        n = min(len(hs), 16)
        assert float(np.mean(hs[4:n])) < float(np.mean(hs_mat[4:n])) - 0.6
    finally:
        session.sim.close()


def test_counters_board_markers_hud(bouncing):
    session, job, hs, _ = bouncing
    m, d = session.sim.model, session.sim.data
    assert job.work == job.n_bounces
    assert job.best_streak >= job.streak > 0
    assert job.best_height == pytest.approx(max(hs))
    # scoreboard: the last digit of BOUNCES
    lit = DIGITS[job.n_bounces % 10]
    for sname, g in zip(SEGS, job.seg_gid[0][3]):
        assert (m.geom_matid[g] == job.matid["seg_on"]) == (sname in lit)
    # ruler markers at the mat level + the height
    assert d.mocap_pos[job.mocap_best][2] == pytest.approx(job.cfg.mat_z + job.best_height)
    st = job.stats()
    assert st["best_body_lengths"] == pytest.approx(job.best_height / BODY_LENGTH_MM, abs=1e-3)
    text = "\n".join(job.hud_lines())
    assert "STREAK" in text and "body lengths" in text and "engineered" in text


def test_slow_motion_is_a_labelled_edit(bouncing):
    session, job, _, _ = bouncing
    assert job.phase in ("launch", "air", "contact")
    s = job.time_scale(0.01)
    assert s == pytest.approx(job.cfg.slowmo, abs=0.05)
    frame = np.zeros((106, 160, 3), np.uint8)
    assert job.post_process(frame, 0.0).any()  # the SLOW MOTION label is drawn


def test_trick_attempt_rotates_the_fly(bouncing):
    session, job, _, _ = bouncing
    c = job.cfg
    c.trick_p, c.trick_min_streak, c.trick_min_height, c.trick_max_height = 1.0, 0, 0.0, 99.0
    tries0 = job.n_trick_tries
    max_tilt = 0.0
    t_end = session.run_time() + 1.0
    while session.run_time() < t_end and job.n_trick_tries == tries0:
        session.sim.step(10)
        session.after_physics()
    assert job.n_trick_tries == tries0 + 1
    j = job._jump_action
    assert j.att_hz == 0.0 and j.p.boost == job.cfg.trick_boost
    for _ in range(200):
        session.sim.step(10)
        session.after_physics()
        max_tilt = max(max_tilt, session.sim.tilt_deg())
        if job.phase not in ("launch", "air"):
            break
    assert max_tilt > 90.0  # an asymmetric push really spins it
    job.cfg.trick_p = 0.0
    # whatever happened, the job carries on (next bounce / crash / off)
    _run(session, 2.5)
    assert job.phase in ("launch", "air", "contact", "settle", "crashed", "off")


def test_crash_landing_is_counted_and_recovered(bouncing):
    session, job, _, msgs = bouncing
    job.cfg.crash_deg = -1.0  # the next touchdown counts as a crash
    rec0, crash0 = job.n_auto_recoveries, job.n_crashes
    t_end = session.run_time() + 1.0
    while session.run_time() < t_end and job.n_crashes == crash0:
        _run(session, 0.01)
    assert job.n_crashes == crash0 + 1 and job.streak == 0 and job.phase == "crashed"
    job.cfg.crash_deg = 50.0
    _run(session, job.cfg.crash_hold_s + 0.05)
    assert job.n_auto_recoveries == rec0 + 1
    assert job.recovery_reasons.get("crash_landing", 0) >= 1
    assert any("auto-recovery" in s for s in msgs)
    # respawned on the mat, no leftover applied force, and it bounces again
    assert not np.any(session.sim.data.xfrc_applied)
    n0 = job.n_bounces
    _run(session, 1.2)
    assert job.n_bounces > n0


def test_fall_off_is_counted_and_recovered():
    session, job = create_job_session("trampoline", _cfg(), dict(QUIET, settle_s=1e9),
                                      say=lambda s: None)
    try:
        sim = session.sim
        _run(session, 0.2)
        # put the standing fly on the lawn beside the trampoline (test set-up), in the
        # air phase: the job must notice the lawn contact
        m = sim.model
        jid = m.body_jntadr[sim.thorax_body_id]
        adr = m.jnt_qposadr[jid]
        sim.data.qpos[adr:adr + 3] = (job.cfg.frame_radius + 3.0, 0.0, 1.0)
        mj.mj_forward(m, sim.data)
        job._go("air")
        _run(session, 0.3)
        assert job.n_off == 1 and job.phase == "off"
        _run(session, job.cfg.off_hold_s + 0.05)
        assert job.recovery_reasons.get("fell_off") == 1
        p = sim.thorax_position()
        assert math.hypot(p[0], p[1]) < 1.0 and p[2] > job.cfg.mat_z  # back on the mat
    finally:
        session.sim.close()
