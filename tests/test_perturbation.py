import math

import numpy as np
import pytest

from perpetualfly import AppConfig, Simulation
from perpetualfly.interaction import (
    AutoPerturbConfig,
    AutoPerturber,
    Perturbation,
    PerturbationConfig,
    install_perturbation,
)
from perpetualfly.interaction.perturbation import heading_relative_direction


@pytest.fixture(scope="module")
def sim():
    s = Simulation(AppConfig())
    yield s
    s.close()


@pytest.fixture
def pert(sim):
    sim.reset()
    p = Perturbation(sim, PerturbationConfig(seed=0))
    yield p
    p.detach()
    sim.reset()


def _record_thorax_force(sim):
    """Pre-step hook appended *after* the perturbation's: sees what mj_step uses."""
    seen = []
    hook = lambda s: seen.append(s.data.xfrc_applied[s.thorax_body_id].copy())  # noqa: E731
    sim.pre_step_hooks.append(hook)
    return seen, hook


def test_force_applied_exactly_during_window(sim, pert):
    bw = pert.body_weight_uN
    assert bw == pytest.approx(10.05, rel=0.02)  # uN, fly weight
    seen, hook = _record_thorax_force(sim)
    try:
        ev = pert.apply_impulse("thorax", (1.0, 0.0, 0.0), magnitude=2.0, duration=0.005)
        assert ev.duration_s == pytest.approx(0.005)
        assert ev.magnitude_uN == pytest.approx(2 * bw)
        assert ev.impulse_uNs == pytest.approx(2 * bw * 0.005)
        sim.step(49)
        assert pert.is_active
        np.testing.assert_allclose(sim.data.xfrc_applied[sim.thorax_body_id],
                                   [2 * bw, 0, 0, 0, 0, 0])
        sim.step(1)  # 50th step = end of the 5 ms window -> cleared right away
        assert not pert.is_active
        np.testing.assert_array_equal(sim.data.xfrc_applied, 0.0)
        sim.step(20)
    finally:
        sim.pre_step_hooks.remove(hook)
    active = [np.any(f != 0) for f in seen]
    assert active == [True] * 50 + [False] * 20
    np.testing.assert_allclose(seen[0], [2 * bw, 0, 0, 0, 0, 0])


def test_overlapping_hits_add_up(sim, pert):
    bw = pert.body_weight_uN
    pert.apply_impulse("thorax", (1, 0, 0), magnitude=1.0, duration=0.003)
    sim.step(10)
    pert.apply_impulse("thorax", (0, 0, 1), magnitude=2.0, duration=0.003)
    sim.step(1)
    np.testing.assert_allclose(sim.data.xfrc_applied[sim.thorax_body_id, :3], [bw, 0, 2 * bw])
    sim.step(20)  # first hit over after 30 steps, second still on
    np.testing.assert_allclose(sim.data.xfrc_applied[sim.thorax_body_id, :3], [0, 0, 2 * bw])
    sim.step(10)
    np.testing.assert_array_equal(sim.data.xfrc_applied, 0.0)


def test_directions_relative_to_heading(sim, pert):
    np.testing.assert_allclose(heading_relative_direction("forward", math.pi / 2), [0, 1, 0],
                               atol=1e-12)
    np.testing.assert_allclose(heading_relative_direction("left", math.pi / 2), [-1, 0, 0],
                               atol=1e-12)
    np.testing.assert_allclose(heading_relative_direction("right", math.pi / 2), [1, 0, 0],
                               atol=1e-12)
    sim.step(3000)  # walk a bit so the heading is not exactly 0
    R = sim.thorax_rotmat()
    fwd = R[:2, 0] / np.linalg.norm(R[:2, 0])
    _, v = pert.resolve_direction("forward")
    np.testing.assert_allclose(v[:2], fwd, atol=1e-9)
    _, v = pert.resolve_direction("left")
    # thorax y axis = left; projected on the ground it is ~ perpendicular to forward
    left = R[:2, 1] / np.linalg.norm(R[:2, 1])
    assert v[2] == 0 and np.dot(v[:2], left) > 0.99
    _, v = pert.resolve_direction("backward")
    np.testing.assert_allclose(v[:2], -fwd, atol=1e-9)
    _, v = pert.resolve_direction("up")
    np.testing.assert_allclose(v, [0, 0, 1])
    for _ in range(20):
        name, v = pert.resolve_direction("random")
        assert name == "random" and np.linalg.norm(v) == pytest.approx(1.0)
        lo, hi = pert.cfg.random_elevation_deg
        assert math.sin(math.radians(lo)) - 1e-9 <= v[2] <= math.sin(math.radians(hi)) + 1e-9
    # the event stores the world-frame vector actually applied
    ev = pert.hit("right", level=1)
    np.testing.assert_allclose(ev.direction[:2], [fwd[1], -fwd[0]], atol=1e-9)


def test_reset_clears_everything(sim, pert):
    pert.hit("left", level=4)
    sim.step(10)
    assert np.any(sim.data.xfrc_applied != 0)
    sim.reset()
    assert not pert.is_active
    np.testing.assert_array_equal(sim.data.xfrc_applied, 0.0)
    sim.step(500)
    np.testing.assert_array_equal(sim.data.xfrc_applied, 0.0)


def test_listeners_and_events(sim, pert):
    got = []
    pert.listeners.append(got.append)
    ev = pert.hit("up", level=1, source="test")
    assert got == [ev] and pert.last_event is ev
    assert ev.level == 1 and ev.source == "test" and ev.body == "nmf/c_thorax"
    assert ev.to_dict()["direction_name"] == "up"


def test_gentle_hit_is_stable_and_fly_keeps_walking(sim, pert):
    sim.step(10000)  # 1 s of walking
    x0 = sim.thorax_position()
    pert.hit("left", level=1)
    tilt_max, dev = 0.0, 0.0
    for _ in range(100):  # 0.5 s after the hit
        sim.step(50)  # check_stability() runs inside and raises on blow-ups
        tilt_max = max(tilt_max, sim.tilt_deg())
    assert tilt_max < 60.0  # not knocked over
    p_mid = sim.thorax_position()
    sim.step(5000)  # another 0.5 s
    p_end = sim.thorax_position()
    speed = np.linalg.norm(p_end[:2] - p_mid[:2]) / 0.5
    assert speed > 8.0  # normal jogging is ~14 mm/s
    assert p_end[0] - x0[0] > 7.0  # forward progress along +x
    assert sim.tilt_deg() < 30.0


def test_auto_perturber_is_deterministic(sim):
    def run(seed):
        sim.reset()
        p = Perturbation(sim, PerturbationConfig(seed=99))
        a = AutoPerturber(sim, p, AutoPerturbConfig(
            min_interval_s=0.01, max_interval_s=0.03, first_hit_after_s=0.005,
            levels=(1, 2), level_weights=(1, 1), seed=seed))
        try:
            sim.step(1000)
        finally:
            a.detach()
            p.detach()
        return [(e.step, e.direction_name, e.direction, e.level, e.source) for e in p.events]

    a1, a2, b = run(5), run(5), run(6)
    assert len(a1) >= 3
    assert a1 == a2
    assert a1 != b
    assert all(e[4] == "auto" for e in a1)
    steps = [e[0] for e in a1]
    assert steps[0] == 50 and all(100 <= d <= 300 for d in np.diff(steps))
    sim.reset()


def test_key_controls(sim):
    sim.reset()
    c = install_perturbation(sim, auto_cfg=AutoPerturbConfig(enabled=False))
    try:
        assert c.handle("q") is None and c.handle(None) is None
        assert "strength" in c.handle("3") and c.perturbation.level == 3
        assert c.handle(63234).startswith("[hit/key]")  # macOS waitKeyEx LEFT
        assert c.perturbation.last_event.direction_name == "left"
        assert c.perturbation.last_event.level == 3
        c.handle("space")
        assert c.perturbation.last_event.direction_name == "random"
        c.handle("up")
        assert c.perturbation.last_event.direction_name == "forward"
        c.handle(63233)  # macOS DOWN
        assert c.perturbation.last_event.direction_name == "backward"
        assert "on" in c.handle("a") and c.auto.enabled
        assert "L3" in c.hud_line()
    finally:
        c.detach()
        sim.reset()


def test_config_dict_roundtrip():
    import json

    c = PerturbationConfig(default_level=3)
    assert PerturbationConfig.from_dict(json.loads(json.dumps(c.to_dict()))) == c
    a = AutoPerturbConfig(levels=(2, 3), level_weights=(1.0, 2.0), magnitude_bw_range=(1, 2))
    assert AutoPerturbConfig.from_dict(json.loads(json.dumps(a.to_dict()))) == a
