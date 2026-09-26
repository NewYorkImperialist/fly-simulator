"""The doner kebab job (fly_simulator/jobs/kebab.py): props, visual-only knife, the
recorded carving stroke, cutting / shavings / regrowth, explicit reset, and the
optional brain reactions (touch per cut, event-driven taste, hunger / satiety,
startle; MN9 -> proboscis) and stress tie-ins with fakes."""

from __future__ import annotations

import math
from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest

from fly_simulator.actions.behaviours import load_grooming_clip
from fly_simulator.config import AppConfig
from fly_simulator.jobs import JobRunner, available_jobs, create_job_session, make_job
from fly_simulator.jobs.geometry import PROP_BIT
from fly_simulator.jobs.kebab import CarveStroke, KebabJob
from fly_simulator.terrain import FLY_BIT, TERRAIN_BIT


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
    """Procedural meshes / textures (fly_simulator/jobs/kebab_assets.py) are closed,
    outward-facing meshes and valid images, and the compiled scene uses them: meat
    slabs are massless visual meshes, the cutting reference box is hidden."""
    from fly_simulator.jobs import kebab_assets as A

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


class _FakeLink:
    def __init__(self, mn9=0.0):
        self.sent = []
        self.stim_log = []
        self.latest = SimpleNamespace(probes={"MN9": mn9})

    def update(self):  # Session.after_physics calls it
        pass

    def send(self, ev, source=""):
        self.sent.append(ev)
        self.stim_log.append(ev)

    def kinds(self, kind, label=""):
        return [e for e in self.sent if e.kind == kind and label in e.details.get("label", "")]


def _land_near_mouth(job, i=0):
    """Put shaving i, as if just launched by a cut, at rest on the tray below the mouth."""
    d = job.sim.data
    mouth = job.mouth_pos()
    qa, va = job.shav_qadr[i], job.shav_vadr[i]
    d.qpos[qa:qa + 3] = (mouth[0], mouth[1], job.tray_top + 0.035)
    d.qvel[va:va + 6] = 0.0
    job._shav_flying[i] = True
    job._shav_t0[i] = job.sim.time - 0.5
    return float(np.linalg.norm(d.qpos[qa:qa + 3] - mouth))


def test_brain_reactions_touch_taste_satiety_with_fakes(kebab):
    session, job, msgs = kebab
    c = job.cfg
    link = _FakeLink(mn9=60.0)
    session.brain = link
    job._shav_flying[:] = False
    job.satiety, job.sated, job._taste_ready, job._last_rt = 0.2, False, 0.0, None
    try:
        # every cut -> one brief touch pulse on the carving side (body_mech R)
        job.after_physics()
        assert link.sent == []  # no cut, no landing -> nothing (no timer any more)
        job._cut(int(np.flatnonzero(job.scale >= job.cfg.ripe_frac)[0]), 50.0)
        job._shav_flying[:] = False  # (the launched shaving is not tested here)
        job.after_physics()
        (touch,) = link.kinds("manual", "KNIFE TOUCH")
        assert touch.details["set"] == "body_mech" and touch.side == "right"
        assert touch.duration_s == c.touch_duration_s and touch.details["rate_hz"] == c.touch_hz
        assert job.n_touch == 1
        # a shaving landing near the proboscis -> a sugar taste pulse scaled by hunger
        assert _land_near_mouth(job) < c.taste_radius
        job.after_physics()
        (taste,) = link.kinds("taste")
        assert taste.details["tastes"] == ["sugar"]
        assert taste.details["sugar_hz"] == pytest.approx(c.sugar_max_hz * 0.8)
        assert "landed" in taste.details["label"] and job.taste_reasons == {"landed": 1}
        assert job.satiety == pytest.approx(0.2 + c.satiety_per_taste, abs=1e-3)
        # refractory: a second landing right away gives no pulse
        _land_near_mouth(job, 1)
        job.after_physics()
        assert len(link.kinds("taste")) == 1 and job.n_landed_near == 2
        # a landing far from the fly is not a taste
        job._taste_ready = 0.0
        d = session.sim.data
        qa = job.shav_qadr[2]
        d.qpos[qa:qa + 3] = (job._tray[1] - 0.3, job._tray[3] - 0.3, job.tray_top + 0.035)
        d.qvel[job.shav_vadr[2]:job.shav_vadr[2] + 6] = 0.0
        job._shav_flying[2], job._shav_t0[2] = True, session.sim.time - 0.5
        job.after_physics()
        assert len(link.kinds("taste")) == 1 and job.n_landed == 3
        # satiety: fills up -> sated (food ignored), carving slows down ...
        for k in range(40):
            job._taste_ready = 0.0
            _land_near_mouth(job, 3 + k % 5)
            job.after_physics()
            if job.sated:
                break
        assert job.sated and job.satiety >= c.full_at and job.n_sated_bouts == 1
        assert any("full" in m for m in msgs)
        n = len(link.kinds("taste"))
        assert link.kinds("taste")[-1].details["sugar_hz"] < taste.details["sugar_hz"]
        job._taste_ready = 0.0
        _land_near_mouth(job, 9)
        job.after_physics()
        assert len(link.kinds("taste")) == n and job.n_taste_ignored == 1
        assert job.speed_mult == pytest.approx(1 - c.satiety_carve_gain * job.satiety, abs=1e-3)
        assert job.speed_mult < 0.75
        assert any("HUNGER" in ln and "SATED" in ln for ln in job.hud_lines())
        # ... and hunger returns: satiety decays with satiety_tau_s
        s0 = job.satiety
        job._last_rt = job.run_time() - 10.0
        job.after_physics()
        assert job.satiety == pytest.approx(s0 * math.exp(-10.0 / c.satiety_tau_s), rel=0.02)
        job._last_rt = job.run_time() - 60.0
        job.after_physics()
        assert not job.sated and any("hungry again" in m for m in msgs)
        # stress: an aroused fly carves faster (combined with the satiety factor)
        session.stress = SimpleNamespace(enabled=True, freq_mult=1.5)
        job.after_physics()
        assert job.stress_mult == pytest.approx(1.5)
        assert job.speed_mult == pytest.approx(1.5 * job.satiety_mult)
        # startle (key S): a shove on the thorax (the brain adds the relay with --stress)
        assert job.handle_key("s") and not job.handle_key("z")
        (poke,) = link.kinds("shove")
        assert poke.details["body"] == "thorax" and poke.intensity == c.startle_intensity
        assert job.n_startles == 1
        # MN9 is read and shown
        session.sim.step(200)
        assert job.mn9_hz == 60.0 and job.prob_ids == []
        assert any("MN9 60 Hz" in ln for ln in job.hud_lines())
    finally:
        session.brain = None
        del session.stress
        job.speed_mult = job.stress_mult = job.satiety_mult = 1.0


def test_brain_off_no_reactions(kebab):
    """Without a brain nothing is sent and carving speed is unchanged."""
    session, job, _ = kebab
    assert getattr(session, "brain", None) is None
    job._cut(int(np.flatnonzero(job.scale >= job.cfg.ripe_frac)[0]), 50.0)
    job.after_physics()
    assert job.speed_mult == 1.0 and job._touch_pending == 0
    assert not job.startle() and job.handle_key("s")
    assert not any("HUNGER" in ln for ln in job.hud_lines())


def test_carving_sends_a_touch_per_cut_and_detects_landings(kebab):
    """Running the job with a (fake) brain: one touch pulse per cut; the launched
    shavings are seen landing."""
    session, job, _ = kebab
    link = _FakeLink()
    session.brain = link
    job._shav_flying[:] = False
    job.n_landed = 0
    try:
        n0 = job.shavings
        JobRunner(session, job, print_every_s=1e9, say=lambda m: None).run(max_seconds=1.5)
        cuts = job.shavings - n0
        assert cuts >= 3
        assert len(link.kinds("manual", "KNIFE TOUCH")) == cuts
        assert job.n_landed >= 1
    finally:
        session.brain = None
        job.speed_mult = job.stress_mult = job.satiety_mult = 1.0


def test_run_job_forwards_job_keys():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_job.py"
    spec = importlib.util.spec_from_file_location("run_job", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = mod.build_parser().parse_args(["--job", "kebab", "--stress"])
    assert args.stress and mod.app_config(args).stress.enabled
    seen = []
    r = object.__new__(mod.KeyForwardingRunner)  # (no session needed for keys)
    r.job = SimpleNamespace(handle_key=lambda k: seen.append(k) or k == "s")
    r.quit_reason, r.renderer, r.paused = "max-seconds", None, False
    r._handle_key("s")
    assert seen == ["s"] and not r.paused
    r._handle_key("p")  # not the job's: the runner's pause
    assert r.paused and seen == ["s", "p"]


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
