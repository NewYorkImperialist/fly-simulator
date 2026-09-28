"""The jump_rope and dj jobs (fly_simulator/jobs/jump_rope.py, dj.py): they build (the
rope capsules touch only the fly, the records are motor-driven hinges the fly can
touch, no unshadowed spot light), the loops progress (skips with the real Jump, a
trip is counted and recovered; a DJ track with scratches counted from the record's
own rotation, the drop, the crossfader move, the mix), and a reset mid-cycle."""

from __future__ import annotations

import math

import mujoco as mj
import numpy as np
import pytest

from fly_simulator.config import AppConfig
from fly_simulator.jobs import available_jobs, create_job_session
from fly_simulator.jobs.dj import SIDES, DJJob
from fly_simulator.jobs.jump_rope import JumpRopeJob
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


# ---------------------------------------------------------------------------
# jump_rope
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rope():
    session, job = create_job_session("jump_rope", _cfg(), {"misstep_p": 0.0})
    _run(session, 7.0)
    yield session, job
    session.sim.close()


def test_rope_builds(rope):
    session, job = rope
    m = session.sim.model
    assert "jump_rope" in available_jobs() and isinstance(job, JumpRopeJob)
    ids = np.flatnonzero(job.rope_geom)
    assert len(ids) == job.cfg.n_seg
    for g in ids:
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == FLY_BIT  # the fly only
        assert m.body_mocapid[m.geom_bodyid[g]] >= 0  # driven (kinematic)
    _no_unshadowed_spots(m)
    assert m.vis.map.znear == pytest.approx(0.05)
    # the rope's bottom skims the ground, its top clears a standing fly
    low = job.rope_points(0.0, 0.0)[:, 2].min()
    high = job.rope_points(math.pi, 0.0)[:, 2].max()
    assert 0.05 < low < 0.25 and high > 10.0


def test_rope_loop_skips(rope):
    session, job = rope
    assert job.n_jumps >= 4 and job.n_turns >= 4
    assert job.n_skips >= 2  # the real Jump clears the rope
    assert job.work == job.n_skips
    assert job.n_skips + job.n_trips >= job.n_turns - 2
    hud = "\n".join(job.hud_lines())
    assert "STREAK" in hud and "rpm" in hud and "trips" in hud
    st = job.stats()
    for k in ("skips", "best_streak", "trips", "recoveries", "rpm"):
        assert k in st


def test_rope_trip_counted_and_reset(rope):
    session, job = rope
    # far too early: the fly lands before the rope arrives -> a real rope contact
    job.cfg.lead_s = 0.25
    n0 = job.n_trips
    _run(session, 3.0)
    assert job.n_trips > n0
    assert job.streak == 0 or job.phase == "turning"
    job.cfg.lead_s = JumpRopeJob.config_cls().lead_s
    # the turners stopped, the fly got going again (a counted recovery)
    _run(session, 4.0)
    assert job.n_recoveries >= 1
    # an explicit reset mid-cycle: the rope is parked, then turns again
    session.reset("manual")
    assert job.rope == "parked" and job.phase == "ready"
    turns = job.n_turns
    _run(session, 3.0)
    assert job.n_turns > turns
    assert np.all(np.isfinite(session.sim.data.qpos))


# ---------------------------------------------------------------------------
# dj
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dj():
    session, job = create_job_session("dj", _cfg(), {"shadows": False})
    yield session, job
    session.sim.close()


def test_dj_builds(dj):
    session, job = dj
    m = session.sim.model
    assert "dj" in available_jobs() and isinstance(job, DJJob)
    for k in range(2):
        j = m.joint(f"dj/spin{k}")
        assert j.type[0] == mj.mjtJoint.mjJNT_HINGE
        a = m.actuator(f"dj/motor{k}")
        assert a.biastype[0] == mj.mjtBias.mjBIAS_AFFINE  # velocity servo
        assert m.actuator_forcelimited[a.id]
        g = job.rec_geom[k]
        assert m.geom_conaffinity[g] == FLY_BIT and m.geom_contype[g] == 0
    _no_unshadowed_spots(m)  # shadows off: the key light is directional


def test_dj_track(dj):
    session, job = dj
    _run(session, 1.0)
    # the records turn at 33 1/3 rpm
    w = [session.sim.data.qvel[job.v_rec[k]] for k in range(2)]
    assert all(abs(abs(v) - 2 * math.pi * 33.333 / 60) < 0.4 for v in w)
    spb = 60.0 / job.bpm
    _run(session, job.cfg.beats_per_track * spb + 0.5)
    assert job.n_tracks == 1 and job.work == 1
    assert job.n_scratches >= 4  # counted from the record's rotation
    assert job.max_reverse > 1.0  # the record really went backward
    assert job.n_drops == 1
    assert job.n_fader_moves + job.n_auto_fades == 1
    assert job.deck == 1 and job.fader == pytest.approx(1.0)
    hud = "\n".join(job.hud_lines())
    assert "BPM" in hud and "scratches" in hud and "hype" in hud


def test_dj_reset_mid_track(dj):
    session, job = dj
    _run(session, 5.0)  # into the scratch section of the next track
    n = job.n_scratches
    session.reset("manual")
    assert job.phase == "setup"
    _run(session, 3.0)
    assert job.phase == "playing"
    assert job.acts.active_name == "dj_stance"
    assert job.n_scratches >= n
    w = session.sim.data.qvel[job.v_rec[0]]
    assert np.isfinite(w)
    assert SIDES == ("rf", "lf")
