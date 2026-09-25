import mujoco as mj
import numpy as np
import pytest

from flygym_demo.complex_terrain import (
    HybridController,
    HybridControllerObservation,
    PreprogrammedSteps,
    make_tripod_cpg_network,
)

from perpetualfly import AppConfig, Simulation, SimulationInstabilityError
from perpetualfly.app import run
from perpetualfly.metrics import LocomotionStats


@pytest.fixture(scope="module")
def sim():
    s = Simulation(AppConfig())
    yield s
    s.close()


def test_model_facts(sim):
    # Units / names that downstream modules rely on (see docs/API_NOTES.md).
    assert sim.timestep == pytest.approx(1e-4)
    assert sim.model.body(sim.thorax_body_id).name == "nmf/c_thorax"
    assert sim.fly_mass == pytest.approx(1.0e-3, rel=0.1)  # grams (~1 mg fly)
    assert sim.model.opt.gravity[2] == pytest.approx(-9810.0)  # mm/s^2


def test_walks_forward_headless():
    res = run(AppConfig(), headless=True, max_seconds=2.0, log=False)
    assert res.forward_displacement > 15.0  # ~14 mm/s after start-up
    assert abs(res.final_thorax_pos[1]) < 5.0  # heading hold keeps it near y=0
    assert 0.8 < res.final_thorax_pos[2] < 1.6  # upright standing height ~1.16 mm


def test_hooks_and_reset(sim):
    calls = {"pre": 0, "post": 0, "reset": 0}
    hooks = (
        (sim.pre_step_hooks, lambda s: calls.__setitem__("pre", calls["pre"] + 1)),
        (sim.post_step_hooks, lambda s: calls.__setitem__("post", calls["post"] + 1)),
        (sim.reset_hooks, lambda s: calls.__setitem__("reset", calls["reset"] + 1)),
    )
    for lst, h in hooks:
        lst.append(h)
    try:
        sim.step(50)
        assert calls["pre"] == calls["post"] == 50
        sim.reset()
        assert calls["reset"] == 1 and sim.step_count == 0
        assert np.all(np.isfinite(sim.data.qpos))
    finally:
        for lst, h in hooks:
            lst.remove(h)


def test_ground_recentering_is_physically_invisible():
    """Shifting the infinite plane in-plane must not change the trajectory."""
    a, b = Simulation(AppConfig()), Simulation(AppConfig())
    try:
        gid = a.model.geom("ground_plane").id
        a.model.geom_pos[gid, :2] = [4000.0, -4000.0]  # far away, multiple of 4 mm
        a.step(3000)
        b.step(3000)
        np.testing.assert_allclose(a.thorax_position(), b.thorax_position(), atol=1e-6)
    finally:
        a.close()
        b.close()


def test_nan_is_detected(sim):
    sim.reset()
    sim.data.qvel[10] = np.nan
    with pytest.raises(SimulationInstabilityError, match="non-finite qvel"):
        sim.check_stability()
    sim.reset()
    sim.step(20)  # recovers after explicit reset


def test_fast_controller_matches_flygym():
    """FastHybrid(Turning)Controller + FastObservationBuilder == FlyGym reference."""
    cfg = AppConfig()
    cfg.controller.heading_gain = 0.0  # descending signal [1, 1] == plain hybrid
    s = Simulation(cfg)
    try:
        ours = s.controller
        ref = HybridController(
            timestep=s.timestep,
            cpg_network=make_tripod_cpg_network(s.timestep, seed=cfg.controller.seed),
            preprogrammed_steps=PreprogrammedSteps(),
            output_dof_order=ours.dof_order,
        )
        ref.reset(seed=cfg.controller.seed)
        for _ in range(400):
            obs_ref = HybridControllerObservation.from_sim(s.fg, s.fly_name)
            obs_fast = ours._obs.build()
            np.testing.assert_allclose(obs_fast.tarsus5_z, obs_ref.tarsus5_z)
            np.testing.assert_allclose(
                obs_fast.stumbling_contact_forces, obs_ref.stumbling_contact_forces, atol=1e-12
            )
            a_ref = ref.step(obs_ref)
            a_fast = ours.impl.step(ours.descending_signal(), obs_fast)
            np.testing.assert_allclose(a_fast.joint_angles, a_ref.joint_angles, atol=1e-12)
            np.testing.assert_array_equal(a_fast.adhesion_onoff, a_ref.adhesion_onoff)
            ours.apply(a_fast)
            mj.mj_step(s.model, s.data)
    finally:
        s.close()


def test_config_json_roundtrip(tmp_path):
    cfg = AppConfig()
    cfg.controller.seed = 7
    cfg.camera.mode = "top"
    cfg.save_json(tmp_path / "c.json")
    back = AppConfig.load_json(tmp_path / "c.json")
    assert back == cfg


def test_stats_ignore_sway():
    st = LocomotionStats(speed_window_s=0.5, path_sample_s=0.25)
    for i in range(201):  # 2 s at 10 mm/s with +-0.3 mm lateral sway at 12 Hz
        t = i * 0.01
        st.update(t, np.array([10 * t, 0.3 * np.sin(2 * np.pi * 12 * t), 1.0]))
    assert st.path_length == pytest.approx(20.0, rel=0.03)
    assert st.average_speed == pytest.approx(10.0, rel=0.03)
    assert st.current_speed == pytest.approx(10.0, rel=0.1)


def test_fast_turning_controller_is_bitwise_flygym():
    """With heading hold on, our controller + obs builder reproduce FlyGym's
    HybridTurningController + from_sim *bit for bit* (not just to 1e-12)."""
    from flygym_demo.complex_terrain import HybridTurningController

    from perpetualfly.controllers.hybrid import _VectorizedSteps

    steps = PreprogrammedSteps()
    vec = _VectorizedSteps(steps, tuple(steps.legs))
    rng = np.random.default_rng(0)
    for _ in range(200):
        ph, mg = rng.uniform(-5, 500, 6), rng.uniform(0, 1.5, 6)
        ref = np.stack([steps.get_joint_angles(l, ph[i], mg[i]) for i, l in enumerate(steps.legs)])
        assert np.array_equal(vec.joint_angles(ph, mg), ref)

    cfg = AppConfig()
    s = Simulation(cfg)
    try:
        ours = s.controller
        ref = HybridTurningController(
            timestep=s.timestep,
            cpg_network=make_tripod_cpg_network(s.timestep, seed=cfg.controller.seed),
            preprogrammed_steps=PreprogrammedSteps(),
            output_dof_order=ours.dof_order,
        )
        ref.reset(seed=cfg.controller.seed)
        for _ in range(1500):  # includes stance/swing transitions and reflex corrections
            obs_ref = HybridControllerObservation.from_sim(s.fg, s.fly_name)
            obs = ours._obs.build()
            assert np.array_equal(obs.stumbling_contact_forces, obs_ref.stumbling_contact_forces)
            sig = ours.descending_signal()
            a_ref = ref.step(sig, obs_ref)
            a = ours.impl.step(sig, obs)
            assert np.array_equal(a.joint_angles, a_ref.joint_angles)
            assert np.array_equal(a.adhesion_onoff, a_ref.adhesion_onoff)
            ours.apply(a)
            mj.mj_step(s.model, s.data)
    finally:
        s.close()


def test_physics_thread_is_deterministic():
    """Stepping in the worker thread (while the main thread reads the state under
    the lock) gives exactly the same trajectory as plain sim.step()."""
    from perpetualfly.physics_thread import PhysicsThread

    a, b = Simulation(AppConfig()), Simulation(AppConfig())
    try:
        a.step(3000)
        runner = PhysicsThread(b, 50, after_chunk=lambda: b.step_count >= 3000,
                               max_realtime_factor=None)
        runner.start()
        while runner.running:
            with runner.locked():
                b.thorax_position()
        runner.stop()
        runner.raise_if_failed()
        assert b.step_count == 3000
        assert np.array_equal(a.data.qpos, b.data.qpos)
        assert np.array_equal(a.data.qvel, b.data.qvel)
    finally:
        a.close()
        b.close()


def test_control_decimation_option_walks():
    cfg = AppConfig()
    cfg.sim.control_every_steps = 5  # off by default; changes the trajectory slightly
    s = Simulation(cfg)
    try:
        s.step(10_000)  # 1 s
        assert s.thorax_position()[0] > 10.0
        assert 0.8 < s.thorax_position()[2] < 1.6
    finally:
        s.close()
