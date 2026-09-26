"""The doner kebab job (perpetualfly/jobs/kebab.py): props, visual-only knife, the
recorded carving stroke, cutting / shavings / regrowth, explicit reset, and the
optional brain (sugar stimulus, MN9 -> proboscis) and stress tie-ins with fakes."""

from __future__ import annotations

import math
from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest

from perpetualfly.actions.behaviours import load_grooming_clip
from perpetualfly.config import AppConfig
from perpetualfly.jobs import JobRunner, available_jobs, create_job_session, make_job
from perpetualfly.jobs.geometry import PROP_BIT
from perpetualfly.jobs.kebab import CarveStroke, KebabJob
from perpetualfly.terrain import FLY_BIT, TERRAIN_BIT


def _cfg() -> AppConfig:
    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.logging.enabled = False
    cfg.render.width, cfg.render.height = 160, 106
    return cfg


def test_registry_and_geometry():
    assert "kebab" in available_jobs()
    job = make_job("kebab", {"spin_rpm": 6.0, "r_bottom": 0.8})
    assert isinstance(job, KebabJob) and job.cfg.spin_rpm == 6.0
    assert job.radius_at(job.cfg.meat_z0) == pytest.approx(0.8)
    assert job.radius_at(job.cfg.meat_z1) == pytest.approx(job.cfg.r_top)
    assert job.radius_at(99.0) == pytest.approx(job.cfg.r_top)  # clipped
    sx, _ = job.spit_xy  # near meat surface `gap` ahead of the carving station
    assert sx - job.radius_at(1.2) == pytest.approx(job.cfg.station_x + job.cfg.gap)
    assert job.spin_rad_s == pytest.approx(6.0 * 2 * math.pi / 60)


@pytest.fixture(scope="module")
def kebab():
    msgs: list[str] = []
    session, job = create_job_session("kebab", _cfg(), say=msgs.append)
    yield session, job, msgs
    session.sim.close()


def test_props_knife_is_visual_only(kebab):
    session, job, _ = kebab
    m = session.sim.model
    assert session.job is job and job.n_chunks > 100
    for name in ("nmf/kebab_blade", "nmf/kebab_knife_handle", "nmf/kebab_hat_puff"):
        g = m.geom(name).id
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
        assert all(g not in (m.pair_geom1[i], m.pair_geom2[i]) for i in range(m.npair))
    assert m.body(m.geom_bodyid[m.geom("nmf/kebab_blade").id]).name == "nmf/rf_tarsus1"
    # the knife and hat add no mass: the fly still weighs ~1.02 mg
    assert session.sim.fly_mass == pytest.approx(1.024e-3, rel=0.01)
    # meat chunks are visual; the tray is touched only by props; shavings not by the fly
    assert np.all(m.geom_contype[job.chunk_gid] == 0)
    tray = m.geom("kebab/tray").id
    assert m.geom_contype[tray] == TERRAIN_BIT and m.geom_conaffinity[tray] == 0
    sh = m.geom("kebab/shaving0_g").id
    assert m.geom_contype[sh] == PROP_BIT and not (m.geom_conaffinity[sh] & FLY_BIT)
    assert "carve" in session.STATIONARY_ACTIONS  # standing to carve is not "stuck"


def test_visual_assets_build(kebab):
    """Procedural meshes / textures (perpetualfly/jobs/kebab_assets.py) are closed,
    outward-facing meshes and valid images, and the compiled scene uses them: meat
    slabs are massless visual meshes, the cutting reference box is hidden."""
    from perpetualfly.jobs import kebab_assets as A

    tex = A.meat_textures(0)
    assert set(A.MEAT_STAGES) <= set(tex) and "inner" in tex
    for img in (*tex.values(), A.steel_texture(0), A.blade_texture(0), A.heater_texture(0),
                A.floor_texture(0), A.wall_texture(0)):
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
        assert img.std() > 1.0  # not a flat colour
    rng = np.random.default_rng(0)
    meshes = [*A.knife_meshes(0.8, 0.08).values(), *A.chef_hat_meshes().values(),
              A.curled_slice(0.36, 0.25, 0.035, 0.04, rng),
              A.meat_slab(0.0, 0.3, 1.0, 1.25, lambda z: 1.0 + 0 * z, 0.16,
                          A.SurfaceNoise(rng, 0.05), rng)]
    for md in meshes:
        assert md.signed_volume() > 0
        assert md.faces.min() >= 0 and md.faces.max() < len(md.verts)
        assert md.uv.shape == (len(md.verts), 2)
        # closed: every edge is shared by exactly two faces
        e = np.sort(np.concatenate([md.faces[:, [0, 1]], md.faces[:, [1, 2]],
                                    md.faces[:, [2, 0]]]), axis=1)
        _, counts = np.unique(e, axis=0, return_counts=True)
        assert np.all(counts == 2)
    session, job, _ = kebab
    m = session.sim.model
    assert np.all(m.geom_type[job.chunk_gid] == int(mj.mjtGeom.mjGEOM_MESH))
    assert np.all(np.isin(m.geom_matid[job.chunk_gid], job.meat_mat))
    assert np.all(m.geom_contype[job.chunk_gid] == 0)
    # the meat meshes add no mass: the spit weighs what its rod does
    assert m.body_mass[job.spit_bid] == pytest.approx(1e-4)
    sh = m.body("kebab/shaving0").id
    assert m.body_mass[sh] == pytest.approx(job.cfg.shaving_mass)
    blade = m.geom("nmf/kebab_blade").id  # the hidden cutting reference
    assert m.geom_rgba[blade, 3] == 0.0 and m.geom_group[blade] == 3
    for name in ("nmf/kebab_blade_vis", "nmf/kebab_knife_guard", "kebab/shaving0_vis"):
        g = m.geom(name).id
        assert m.geom_type[g] == int(mj.mjtGeom.mjGEOM_MESH)
        assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
    # chunk centres (the carving test points) sit on the cone, just inside its surface
    cen = job.chunk_centres()
    sx, sy = job.spit_xy
    r = np.hypot(cen[:, 0] - sx, cen[:, 1] - sy)
    expect = np.array([job.radius_at(z) for z in cen[:, 2]])
    assert np.all(np.abs(expect - r) < job.cfg.chunk_t)


def test_carves_shavings(kebab):
    session, job, _ = kebab
    sim = session.sim
    r = JobRunner(session, job, print_every_s=1e9, say=lambda m: None)
    a0 = float(sim.data.qpos[job.spin_qadr])
    park = np.array([sim.data.qpos[qa:qa + 3].copy() for qa in job.shav_qadr])
    out = r.run(max_seconds=2.5)
    assert job.state == "carving" and session.actions.active_name == "carve"
    assert out["shavings"] >= 3 and job.work == job.shavings
    assert out["falls"] == 0 and out["auto_recoveries"] == 0
    # the spit turns at ~spin_rpm (velocity actuator)
    turned = float(sim.data.qpos[job.spin_qadr]) - a0
    assert turned == pytest.approx(job.spin_rad_s * 2.5, rel=0.25)
    # cut chunks are shrunk / hidden and regrowing; shavings left their parking spots
    assert np.sum(job.scale < 1.0) >= 1
    moved = [np.linalg.norm(sim.data.qpos[qa:qa + 3] - park[i]) > 0.05
             for i, qa in enumerate(job.shav_qadr)]
    assert sum(moved) >= min(job.shavings, len(moved)) - 1
    # fly stays at its station, upright
    p = sim.thorax_position()
    assert math.hypot(p[0] - job.cfg.station_x, p[1] - job.cfg.station_y) < job.cfg.drift_tol
    lines = job.hud_lines()
    assert lines[0].startswith("DONER KEBAB FLY") and "shavings carved" in lines[2]
    assert any("recorded fly grooming" in ln for ln in lines)


def test_carve_stroke_replays_recorded_clip(kebab):
    """ActionManager.trigger sets action.source to the trigger source ("job"); the
    carve stroke must still replay the recorded clip, not the synthetic sweep."""
    session, _, _ = kebab
    a = session.actions.action
    assert isinstance(a, CarveStroke) and a.source == "job"
    clip, names, fps = load_grooming_clip()
    cmd = a.command(session.actions, a._t_last)
    x = (a._phase_s * fps) % (len(clip) - 1)
    i = int(x)
    expect = (1 - (x - i)) * clip[i] + (x - i) * clip[i + 1]
    np.testing.assert_allclose(cmd.targets[a._cols], expect, atol=1e-9)
    # speed changes do not jump the clip phase
    ph = a._phase_s
    a.speed = 2.0
    a.command(session.actions, a._t_last + 0.01)
    assert a._phase_s == pytest.approx(ph + 0.02)
    a.speed = 1.0


def test_cut_regrow_and_reset(kebab):
    session, job, msgs = kebab
    sim = session.sim
    m = sim.model
    ripe = np.flatnonzero(job.scale >= 1.0)
    j = int(ripe[0])
    n0, i_shav = job.shavings, job._next_shaving
    job._cut(j, 50.0)
    assert job.shavings == n0 + 1 and job.scale[j] == 0.0
    assert m.geom_rgba[job.chunk_gid[j], 3] == 0.0  # gone from the spit
    qa = job.shav_qadr[i_shav]
    assert np.linalg.norm(sim.data.qpos[qa:qa + 3] - sim.data.geom_xpos[job.chunk_gid[j]]) < 0.3
    # kebab completed every shavings_per_kebab
    k0 = job.kebabs
    job.shavings = (job.shavings // job.cfg.shavings_per_kebab + 1) * job.cfg.shavings_per_kebab - 1
    job._cut(int(np.flatnonzero(job.scale >= 1.0)[0]), 50.0)
    assert job.kebabs == k0 + 1 and any("kebab #" in s for s in msgs)
    # regrowth: 1 s of updates grows the chunk by 1 / regrow_s
    sim.step(10000)
    assert job.scale[j] == pytest.approx(1.0 / job.cfg.regrow_s, abs=0.02)
    # explicit, counted reset: a fresh kebab, shavings back on the tray
    n_rec = job.n_auto_recoveries
    job.recover("test")
    assert job.n_auto_recoveries == n_rec + 1 and np.all(job.scale == 1.0)
    assert np.all(m.geom_rgba[job.chunk_gid, 3] == 1.0)
    for i, qa in enumerate(job.shav_qadr):
        np.testing.assert_allclose(sim.data.qpos[qa:qa + 3], job._park[i][:3], atol=0.05)
    # a lost shaving is put back on the tray
    sim.data.qpos[job.shav_qadr[0] + 2] = -5.0
    job.after_physics()
    assert job.n_shavings_lost == 1
    np.testing.assert_allclose(sim.data.qpos[job.shav_qadr[0]:job.shav_qadr[0] + 3],
                               job._park[0][:3])


def test_brain_and_stress_tie_ins_with_fakes(kebab):
    session, job, _ = kebab
    sent = []
    fake_link = SimpleNamespace(send=lambda ev, source="": sent.append(ev), stim_log=[],
                                latest=SimpleNamespace(probes={"MN9": 60.0}))
    session.brain = fake_link
    session.stress = SimpleNamespace(enabled=True, freq_mult=1.5)
    try:
        job._next_food = 0.0
        job.after_physics()
        assert len(sent) == 1 and sent[0].details["set"] == "sugar"
        assert sent[0].duration_s == job.cfg.food_duration_s
        assert job.speed_mult == pytest.approx(1.5)
        job.after_physics()  # not again before food_every_s
        assert len(sent) == 1
        session.sim.step(200)  # updates read MN9 (no proboscis joints in this body)
        assert job.mn9_hz == 60.0 and job.prob_ids == []
        assert any("MN9 60 Hz" in ln for ln in job.hud_lines())
    finally:
        session.brain = None
        del session.stress
        job.speed_mult = 1.0


def test_full_body_proboscis_follows_mn9():
    """With the proboscis joints (turned on when the brain is enabled), the job drives
    the proboscis from the brain's MN9 rate."""
    job = make_job("kebab")
    cfg = _cfg()
    cfg.brain.enabled = True  # configure_app -> full body; no brain process is started
    session, job = create_job_session(job, cfg, say=lambda m: None)
    try:
        assert cfg.fly.extra_joints and len(job.prob_ids) == 2
        session.brain = SimpleNamespace(send=lambda ev, source="": None, stim_log=[],
                                        latest=SimpleNamespace(probes={"MN9": 90.0}))
        session.sim.step(5000)
        assert job.proboscis > 0.95
        rostrum = session.sim.model.joint("nmf/c_head-c_rostrum-pitch").id
        assert session.sim.data.qpos[session.sim.model.jnt_qposadr[rostrum]] < -0.5
    finally:
        session.brain = None
        session.sim.close()
