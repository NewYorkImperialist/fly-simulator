"""Flapping-wing flight prototype (fly_simulator/flight, docs/FLIGHT.md)."""

import math

import mujoco as mj
import numpy as np
import pytest

from fly_simulator import AppConfig, Simulation
from fly_simulator.flight import (
    AIR_DENSITY,
    FLIGHT_TIMESTEP,
    FlightSimulation,
    WingbeatGenerator,
    WingbeatParams,
    measure_tethered,
)
from fly_simulator.flight.control import HoverController, base_wingbeat
from fly_simulator.flight.wingbeat import DEG


@pytest.fixture(scope="module")
def fsim():
    sim = FlightSimulation()
    yield sim
    sim.close()


def test_flight_model_builds_with_wings_fluid_and_42_leg_actuators(fsim):
    m = fsim.model
    assert m.opt.timestep == pytest.approx(FLIGHT_TIMESTEP)
    assert m.opt.density == pytest.approx(AIR_DENSITY)
    names = [m.joint(j).name for j in range(m.njnt)]
    for s in "lr":
        for j in ("stroke", "deviation", "rotation"):
            assert f"nmf/c_thorax-{s}_wing-{j}" in names
    fluid = [g for g in range(m.ngeom) if m.geom_fluid[g, 0] > 0]
    assert sorted(m.geom(g).name for g in fluid) == ["nmf/l_wing_fluid", "nmf/r_wing_fluid"]
    assert all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in fluid)
    # FlyGym still registers exactly the 42 leg position actuators
    assert len(fsim.fly.get_actuated_jointdofs_order("position")) == 42
    assert len(fsim.leg_act) == 42 and len(fsim.wing_act) == 6
    # wing mass unchanged by the (massless) fluid geoms
    assert m.body_mass[fsim.wing_bodies].tolist() == pytest.approx([2.5e-6, 2.5e-6])


def test_default_walking_model_is_unaffected():
    sim = Simulation(AppConfig())
    try:
        m = sim.model
        assert m.opt.density == 0.0 and m.opt.viscosity == 0.0
        assert m.opt.timestep == pytest.approx(1e-4)
        assert int(m.opt.integrator) == int(mj.mjtIntegrator.mjINT_EULER)
        assert not any("wing" in m.joint(j).name for j in range(m.njnt))
        assert not np.any(m.geom_fluid[:, 0] > 0)
        assert m.nu == 48
    finally:
        sim.close()


def test_wingbeat_generator_pattern_and_envelope():
    P = WingbeatParams(freq=200.0).symmetric(amplitude=150 * DEG, aoa_down=40 * DEG,
                                             aoa_up=50 * DEG, rot_phase=0.0)
    g = WingbeatGenerator(P)
    ph = np.linspace(0, 2 * math.pi, 400, endpoint=False)
    q = np.array([g.angles(p) for p in ph])
    assert np.degrees(q[:, 0].max() - q[:, 0].min()) == pytest.approx(150, abs=0.1)
    assert np.allclose(q[:, :3], q[:, 3:])  # symmetric wings
    # downstroke (wing moving forward, pi..2pi) at aoa_down, upstroke at pi - aoa_up
    assert np.degrees(g.angles(1.5 * math.pi)[2]) == pytest.approx(40, abs=0.5)
    assert np.degrees(g.angles(0.5 * math.pi)[2]) == pytest.approx(130, abs=0.5)
    assert np.allclose(q[:, 1], 0.0)  # no tilt -> no deviation
    # derivative matches finite differences in time
    q0, dq = g.targets(1.0)
    g2 = WingbeatGenerator(P)
    g2.reset(1.0)
    g2.advance(1e-6)
    assert np.allclose((g2.angles() - q0) / 1e-6, dq, rtol=1e-2, atol=1.0)
    # envelope fades in from the rest pose
    g.start(0.01)
    assert np.allclose(g.targets()[0], 0.0)
    for _ in range(300):
        g.advance(5e-5)
    assert g.envelope == 1.0


def test_tethered_lift_close_to_weight_and_steering_signs(fsim):
    base = base_wingbeat()
    r = measure_tethered(fsim, base, cycles=2, settle_cycles=2)
    assert 0.85 < r.lift_over_weight < 1.15
    assert abs(r.force_world[0]) < 0.05 * r.weight
    assert r.amplitude_meas_deg == pytest.approx(160, abs=8)
    assert r.max_actuator_force < 40.0
    fast = measure_tethered(fsim, WingbeatParams(freq=250.0, left=base.left, right=base.right),
                            cycles=2, settle_cycles=2)
    assert fast.lift_over_weight > r.lift_over_weight + 0.1
    tilted = measure_tethered(fsim, base.symmetric(tilt=15 * DEG), cycles=2, settle_cycles=2)
    assert tilted.force_world[0] > 0.15 * r.weight  # nose-down stroke plane -> forward force
    fsim.tethered = False


def test_short_free_hover_stays_upright_and_level(fsim):
    fsim.reset()
    fsim.wingbeat.params = base_wingbeat()
    fsim.place((0.0, 0.0, 10.0))
    ctrl = HoverController(fsim)
    ctrl.target.pos = fsim.com()
    fsim.flight_controller = ctrl
    fsim.flapping = True
    try:
        z0 = fsim.com()[2]
        fsim.step(int(0.2 / fsim.timestep))
        assert abs(fsim.tilt_deg() - fsim.stroke_plane_deg) < 5.0
        assert abs(fsim.com()[2] - z0) < 1.0
        assert np.linalg.norm(fsim.com()[:2]) < 1.0
    finally:
        fsim.flight_controller = None
        fsim.flapping = False
