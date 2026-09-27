"""Eternal jobs framework (fly_simulator/jobs): registry, steering, recording helpers,
and short runs of the sisyphus / hamster_wheel jobs (props compiled in, contacts,
counters, explicit recovery, NaN recovery)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fly_simulator.config import AppConfig
from fly_simulator.jobs import (
    EternalJob,
    JobRunner,
    Steering,
    available_jobs,
    create_job_session,
    format_uptime,
    get_job,
    install_job,
    job_props_present,
    make_job,
)
from fly_simulator.jobs.geometry import PROP_BIT, contact_kwargs, wrap_angle
from fly_simulator.terrain import FLY_BIT, TERRAIN_BIT


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.logging.enabled = False
    cfg.render.width, cfg.render.height = 160, 106
    return cfg


# ---------------------------------------------------------------- pure python
def test_registry_and_configs():
    names = available_jobs()
    assert "sisyphus" in names and "hamster_wheel" in names
    assert issubclass(get_job("sisyphus"), EternalJob)
    with pytest.raises(KeyError):
        get_job("no_such_job")
    job = make_job("sisyphus", {"slope_deg": 8.0})
    assert job.cfg.slope_deg == 8.0 and job.name == "sisyphus"


def test_format_uptime():
    assert format_uptime(0) == "Day 1 00:00:00"
    assert format_uptime(3661.9) == "Day 1 01:01:01"
    assert format_uptime(86400 * 2 + 59) == "Day 3 00:00:59"
    assert format_uptime(float("nan")) == "Day 1 00:00:00"


def test_contact_kwargs_bits():
    assert (contact_kwargs("static")["contype"], contact_kwargs("static")["conaffinity"]) == (
        TERRAIN_BIT, FLY_BIT)
    dyn = contact_kwargs("dynamic")
    assert dyn["contype"] == PROP_BIT and dyn["conaffinity"] == FLY_BIT | TERRAIN_BIT | PROP_BIT
    assert contact_kwargs("fly")["conaffinity"] == FLY_BIT and contact_kwargs("fly")["priority"] == 1
    assert contact_kwargs("visual")["contype"] == 0 == contact_kwargs("visual")["conaffinity"]
    assert wrap_angle(2.5 * math.pi) == pytest.approx(0.5 * math.pi)


class _FakeCtrl:
    signal_filter = None


class _FakeSim:
    def __init__(self):
        self.controller = _FakeCtrl()
        self.h = 0.0

    def heading(self):
        return self.h


def test_steering_signal():
    sim = _FakeSim()
    st = Steering(sim, gain=2.0, max_turn=0.6, spin_above=math.radians(50), spin_amp=0.9)
    assert sim.controller.signal_filter is st
    ones = np.ones(2)
    np.testing.assert_allclose(st(ones), ones)  # no target: pass through
    st.set(0.0, 1.0)
    sim.h = math.radians(10)  # fly points left of the target -> turn right: left > right
    out = st(ones)
    assert out[0] > out[1] and out[0] == pytest.approx(1.0 + 2.0 * math.radians(10), rel=0.3)
    sim.h = math.radians(120)  # large error: turn on the spot (sides opposite)
    out = st(ones)
    assert out[0] > 0 > out[1]
    st.set(0.0, 0.0)
    sim.h = 0.0
    np.testing.assert_allclose(st(ones), [0.0, 0.0])  # speed 0 = stand
    st.set(float("nan"))
    assert st.target is None
    st.uninstall()
    assert sim.controller.signal_filter is None


def test_rolling_recorder_and_timelapse(tmp_path):
    from fly_simulator.jobs.recording import RollingRecorder, Timelapse

    rec = RollingRecorder(tmp_path / "seg", segment_s=0.5, keep=2, fps=10)
    tl = Timelapse(tmp_path / "tl", every_s=0.3, fps=10, frames_per_file=3, keep=2)
    frame = np.zeros((32, 48, 3), np.uint8)
    t = 0.0
    while t < 2.05:
        frame[:] = int(t * 100) % 255
        if rec.due(t):
            rec.add(frame, t)
        if tl.due(t):
            tl.add(frame, t)
        t += 0.05
    rec.close()
    tl.close()
    segs = sorted((tmp_path / "seg").glob("*.mp4"))
    assert rec.n_segments >= 4 and len(segs) == 2 and rec.n_deleted >= 2
    assert 6 <= tl.n_frames <= 8  # one frame per 0.3 s of 2 s
    assert len(list((tmp_path / "tl").glob("timelapse*.mp4"))) == 2


# ---------------------------------------------------------------- sisyphus
@pytest.fixture(scope="module")
def sisyphus():
    msgs: list[str] = []
    session, job = create_job_session("sisyphus", _cfg(), say=msgs.append)
    yield session, job, msgs
    session.sim.close()


def test_sisyphus_props_and_contacts(sisyphus):
    session, job, _ = sisyphus
    m = session.sim.model
    assert session.job is job
    assert job_props_present(m, job)
    assert not job_props_present(m, get_job("hamster_wheel")())
    g = m.geom("sisyphus/boulder_geom").id
    assert m.geom_contype[g] == PROP_BIT and m.geom_conaffinity[g] & TERRAIN_BIT
    assert m.body_mass[m.body("sisyphus/boulder").id] == pytest.approx(job.cfg.ball_mass, rel=1e-3)
    ramp = m.geom("sisyphus/ramp").id
    assert m.geom_contype[ramp] == TERRAIN_BIT and m.geom_conaffinity[ramp] == FLY_BIT
    assert m.npair >= 7 and m.nexclude >= 48  # slippery head pairs + leg excludes
    # ground height follows the hill (fall detector uses it)
    assert job.ground_height(0.0, 0.0) == 0.0
    assert job.ground_height(job.x_top - 0.01, 0.0) == pytest.approx(job.hill_height, abs=0.01)
    assert job.ground_height(0.0, 5.0) > 0.5  # side bank
    assert session.detector.ground_height_fn == job.ground_height
    # the job's znear survives the fly's MuJoCo globals (merged in after extension)
    assert m.vis.map.znear == pytest.approx(job.znear)


def test_job_znear_zfar_applied_after_compile():
    import mujoco as mj

    class Deep(EternalJob):
        znear, zfar = 0.02, 123.0

    m = mj.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    z0, f0 = float(m.vis.map.znear), float(m.vis.map.zfar)
    EternalJob().apply_visual_globals(m)  # None: left as compiled
    assert m.vis.map.znear == pytest.approx(z0) and m.vis.map.zfar == pytest.approx(f0)
    Deep().apply_visual_globals(m)
    assert m.vis.map.znear == pytest.approx(0.02) and m.vis.map.zfar == pytest.approx(123.0)


def test_sisyphus_pushes_boulder_uphill(sisyphus):
    session, job, _ = sisyphus
    r = JobRunner(session, job, print_every_s=1e9, say=lambda m: None)
    x0 = float(job.ball_pos()[0])
    r.run(max_seconds=1.2)
    assert job.state == "push" or job.metres_pushed > 0
    assert float(job.ball_pos()[0]) > x0 + 2.0  # pushed toward / up the hill
    assert job.n_falls == 0
    lines = job.hud_lines()
    assert lines[0].startswith("SISYPHUS FLY") and lines[1].startswith("Day 1 00:00:01")
    assert any("summits" in ln for ln in lines)


def test_sisyphus_explicit_recovery_and_nan(sisyphus):
    session, job, msgs = sisyphus
    n0, resets0 = job.n_auto_recoveries, session.metrics.n_resets
    job.recover("test")
    assert job.n_auto_recoveries == n0 + 1 and job.recovery_reasons["test"] == 1
    assert session.metrics.n_resets == resets0 + 1  # counted, not hidden
    assert job.state == "approach"
    # a NaN in the boulder state: the runner recovers with an explicit reset
    r = JobRunner(session, job, print_every_s=1e9, chunk_steps=20, say=msgs.append)
    session.sim.data.qvel[job.ball_vadr] = float("nan")
    r.step_chunk()
    assert r.n_instabilities == 1 and job.recovery_reasons.get("instability") == 1
    assert np.all(np.isfinite(session.sim.data.qpos))
    r.step_chunk()
    lost0 = job.boulders_lost
    # lost boulder -> a new one drops from the sky, clear of the fly
    session.sim.data.qpos[job.ball_qadr + 1] = 30.0  # rolled out of the arena
    session.sim.step(20)
    assert job.boulders_lost == lost0 + 1
    assert np.linalg.norm(job.ball_xy() - job.fly_xy()) > job.cfg.ball_radius + 1.0


def test_install_job_requires_props(sisyphus):
    session, _, _ = sisyphus
    with pytest.raises(RuntimeError, match="props not found"):
        install_job(session, "hamster_wheel")


# ---------------------------------------------------------------- hamster wheel
def test_hamster_wheel_spins():
    session, job = create_job_session("hamster_wheel", _cfg(), say=lambda m: None)
    try:
        sim = session.sim
        assert sim.cfg.fly.spawn_height == pytest.approx(job.cfg.bottom_height + 0.8)
        r = JobRunner(session, job, print_every_s=1e9, say=lambda m: None)
        out = r.run(max_seconds=1.5)
        assert out["revolutions"] > 0.2  # ~15 mm/s on a 14 mm wheel
        assert job.surface_speed() > 5.0
        assert out["falls"] == 0
        p = sim.thorax_position()
        assert abs(p[0]) < 3.0 and abs(p[1]) < job.cfg.width / 2  # stays at the bottom
        # stall detection runs in the wheel frame: the fly is not "stuck"
        assert session.detector.state.value in ("UPRIGHT", "DESTABILIZED")
        frame, hud = r.render()
        assert hud.shape == frame.shape and hud.mean() != frame.mean()
        r.close()
    finally:
        sim.close()
