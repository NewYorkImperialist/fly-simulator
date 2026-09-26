"""Whip looming vision (fly_simulator/vision/looming.py, docs/VISION.md)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest

from fly_simulator.brain.mapping import StimulusMapper
from fly_simulator.brain.schema import StimulusEvent
from fly_simulator.vision import (
    LoomingConfig,
    LoomingVision,
    LoomResponse,
    VisualSource,
    install_whip_vision,
    loom_response,
)
from fly_simulator.vision.looming import eye_view, fov_mask

LEFT, RIGHT = np.array([1.0]), np.array([-1.0])
I3 = np.eye(3)[None]  # head frame = world frame (x forward, y left, z up)


def sphere(center, r):
    return (np.array([center], float), np.array([[0.0, 0.0, 1.0]]), np.zeros(1), np.array([r]))


# ---------------------------------------------------------------- response function


def test_response_thresholds_and_monotonic():
    p = LoomResponse()
    assert loom_response(80.0, 0.0, p) == (0.0, 0.0)  # large but static: nothing
    assert loom_response(5.0, p.lc4_v0 * 0.9, p) == (0.0, 0.0)  # slow expansion
    lc4 = [loom_response(10.0, v, p)[0] for v in (1.2e4, 2e4, 4e4, 8e4, 1e6)]
    assert all(b >= a for a, b in zip(lc4, lc4[1:])) and lc4[-1] == pytest.approx(p.max_hz)
    # LPLC2: angular size, only while expanding fast enough
    assert loom_response(70.0, 5e4, p)[1] == pytest.approx(p.max_hz)
    assert loom_response(10.0, 5e4, p)[1] == 0.0
    assert loom_response(70.0, 1000.0, p)[1] == 0.0


# ---------------------------------------------------------------- geometry


def test_sphere_solid_angle_exact_and_fov():
    cfg = LoomingConfig()
    E = np.zeros((1, 3))
    d, r = 2.0, 0.5
    v = eye_view(E, I3, LEFT, sphere((0.0, d, 0.0), r), cfg)  # straight out to the left
    assert v["omega"][0] == pytest.approx(2 * math.pi * (1 - math.sqrt(1 - (r / d) ** 2)), rel=1e-6)
    assert v["theta"][0] == pytest.approx(2 * math.asin(r / d), rel=1e-6)
    assert v["az"][0] == pytest.approx(90.0) and v["el"][0] == pytest.approx(0.0, abs=1e-6)
    assert v["dist"][0] == pytest.approx(d - r)
    # the right eye can't see straight left, nor can either eye see straight back
    assert eye_view(E, I3, RIGHT, sphere((0.0, d, 0.0), r), cfg)["omega"][0] == 0.0
    for sgn in (LEFT, RIGHT):
        assert eye_view(E, I3, sgn, sphere((-d, 0.0, 0.0), r), cfg)["omega"][0] == 0.0
        # straight ahead (binocular overlap) and straight up: both eyes
        assert eye_view(E, I3, sgn, sphere((d, 0.0, 0.0), r), cfg)["omega"][0] > 0
        assert eye_view(E, I3, sgn, sphere((0.0, 0.0, d), r), cfg)["omega"][0] > 0


def test_fov_mask_soft_edges():
    cfg = LoomingConfig()
    az = np.radians(np.array([-60.0, -15.0, 0.0, 90.0, 165.0, 179.0]))
    h = np.stack([np.cos(az), np.sin(az), np.zeros_like(az)], -1)[None]
    m = fov_mask(h, LEFT, cfg)[0]
    assert m[0] == 0.0 and m[-1] == 0.0 and m[3] == 1.0
    assert 0.0 <= m[1] < 0.1 and m[2] > 0.9 and m[4] < 0.1


def test_thin_rod_matches_analytic_broadside():
    cfg = LoomingConfig(samples_per_capsule=200)
    d, r, half = 1.0, 0.02, 3.0
    shapes = (np.array([[0.0, d, 0.0]]), np.array([[1.0, 0.0, 0.0]]),
              np.array([half]), np.array([r]))
    om = eye_view(np.zeros((1, 3)), I3, LEFT, shapes, cfg)["omega"][0]
    # thin rod along x at distance d: integral of (2 r / rho) * (d / rho^2) dx
    x = np.linspace(-half, half, 20001)
    rho = np.hypot(x, d)
    ref = np.trapezoid(2 * np.arcsin(r / rho) * d / rho ** 2, x)
    assert om == pytest.approx(ref, rel=0.02)


# ---------------------------------------------------------------- LoomingVision


def _stub_sim():
    m = mj.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='nmf/c_thorax'><freejoint/>"
        "<geom type='sphere' size='0.1'/></body></worldbody></mujoco>")
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    return SimpleNamespace(model=m, data=d, thorax_body_id=1, time=0.0,
                           post_step_hooks=[], reset_hooks=[])


def _approach(speed_mm_s: float, side_y: float, start=6.0, stop=0.8, r=0.3, dt=1e-4):
    """Sphere approaching the fly (at the origin) from +y (left) or -y (right)."""
    sim = _stub_sim()
    got = []
    vis = LoomingVision(sim, LoomingConfig(), sink=got.append)
    pos = {"y": start}
    vis.add_source(VisualSource("ball", shapes=lambda: sphere((0.0, side_y * pos["y"], 0.0), r),
                                period=lambda: 2e-4))
    vis.attach()
    t_contact = None
    while sim.time < 1.0:
        sim.time += dt
        pos["y"] = max(stop, pos["y"] - speed_mm_s * dt)
        if t_contact is None and pos["y"] <= stop:
            t_contact = sim.time
        for h in sim.post_step_hooks:
            h(sim)
        if t_contact is not None and sim.time > t_contact + 0.02:
            break
    return vis, got, t_contact


def test_fast_approach_drives_the_seeing_eye_before_contact():
    vis, got, t_contact = _approach(2000.0, +1.0)  # 2 m/s from the left
    assert not vis.has_eye_geoms  # fallback eyes (head +- lateral offset)
    assert got, "a fast approaching ball must loom"
    assert {e.side for e in got} == {"left"}
    first = got[0]
    assert first.kind == "loom" and first.sim_time < t_contact
    assert first.duration_s == pytest.approx(vis.cfg.persist_s)
    assert max(e.details["lc4_hz"] for e in got) > 100
    assert max(e.details["lplc2_hz"] for e in got) > 50
    assert 0 < first.details["azimuth_deg"] <= 120
    # refreshes are rate limited (not one event per update)
    assert len(got) < vis.n_updates / 3
    # mirrored: from the right -> right eye
    _, got_r, _ = _approach(2000.0, -1.0)
    assert got_r and {e.side for e in got_r} == {"right"}


def test_slow_approach_and_retreat_do_not_loom():
    _, got, _ = _approach(20.0, +1.0, start=1.5)  # 20 mm/s: a walking-speed approach
    assert got == []
    vis, got, _ = _approach(-2000.0, +1.0, start=0.8)  # receding fast
    assert got == []


def test_reset_clears_filters():
    vis, _, _ = _approach(2000.0, +1.0)
    vis._on_reset(vis.sim)
    assert vis._filt == {} and vis._last_t is None
    assert all(e.theta == 0.0 for e in vis.state["ball"])


# ---------------------------------------------------------------- brain mapping


def _loom_table():
    from fly_simulator.brain.process import _synthetic_table

    table, _ = _synthetic_table(n=40)
    ct = table.cols["cell_type"]
    for k in range(24, 28):
        ct[k] = "LC4"
    for k in range(28, 32):
        ct[k] = "LPLC2"
    return table


def test_mapping_loom_drives_lc4_lplc2_on_the_eye_side():
    table = _loom_table()
    side = table.col("side")
    mp = StimulusMapper(table)
    out = mp.resolve(StimulusEvent("loom", side="left", intensity=0.5, duration_s=0.05,
                                   details={"lc4_hz": 150.0, "lplc2_hz": 500.0}))
    labels = {lab: (idx, rate) for lab, idx, rate in out}
    idx, rate = labels["loom:LC4:left"]
    assert set(side[idx]) == {"left"} and set(table.col("cell_type")[idx]) == {"LC4"}
    assert rate == 150.0
    assert labels["loom:LPLC2:left"][1] == mp.max_rate_hz  # clipped
    # zero rate -> that set is not driven; no rates given -> intensity rate for both
    out = mp.resolve(StimulusEvent("loom", side="right", details={"lc4_hz": 80.0, "lplc2_hz": 0.0}))
    assert [lab for lab, _, _ in out] == ["loom:LC4:right"]
    out = mp.resolve(StimulusEvent("loom", side="none", intensity=0.5))
    assert len(out) == 2 and all(r == 100.0 for _, _, r in out)
    assert all(len(set(side[i])) == 2 for _, i, _ in out)


# ---------------------------------------------------------------- whip integration


@pytest.fixture(scope="module")
def whip_session():
    from fly_simulator import AppConfig
    from fly_simulator.app import Session

    cfg = AppConfig()
    cfg.terrain.difficulty = "flat"
    s = Session(cfg, log=False, say=lambda m: None)
    yield s
    s.close("test")


def _crack(s, vis, side, level, walk=0.8):
    s.reset("test")
    vis.sent.clear()
    s.sim.step(int(walk / s.sim.timestep))
    n = len(s.whip.events)
    s.whip.crack(side, level)
    while len(s.whip.events) == n:
        s.sim.step(50)
    s.sim.step(300)
    return s.whip.events[-1]


def test_whip_vision_hard_crack_looms_on_its_side(whip_session):
    s = whip_session
    got = []
    vis = install_whip_vision(s, s.whip, got.append)
    try:
        assert vis.has_eye_geoms and vis in s.sim.post_step_hooks
        E, _ = vis.eyes()
        heading_left = s.sim.thorax_rotmat()[:, 1]
        assert (E[0] - E[1]) @ heading_left > 0.4  # left eye is on the left
        ev = _crack(s, vis, "left", 3)
        assert ev.hit
        left = [e for e in got if e.side == "left"]
        assert left, "a hard crack from the left must loom on the left eye"
        assert left[0].sim_time <= s.metrics.run_time_at(ev.sim_time) + 0.005
        assert max(e.details["lc4_hz"] for e in left) > 100
        assert all(e.details["source"] == "whip" for e in got)
        # slow sweep (10 rad/s): no looming drive
        got.clear()
        s.whip.cfg.levels[0].omega = 10.0
        try:
            _crack(s, vis, "left", 1)
        finally:
            s.whip.cfg.levels[0].omega = 60.0
        assert got == []
    finally:
        vis.detach()
    assert vis not in s.sim.post_step_hooks
