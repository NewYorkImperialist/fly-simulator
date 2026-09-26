"""Tests for the residual-RL Gymnasium env (fly_simulator.rl). Needs the ``rl`` extra."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from fly_simulator.config import AppConfig  # noqa: E402
from fly_simulator.interaction.perturbation import AutoPerturber, Perturbation  # noqa: E402
from fly_simulator.rl import CurriculumStage, EnvConfig, PerpetualFlyEnv  # noqa: E402
from fly_simulator.simulation import Simulation  # noqa: E402
from fly_simulator.terrain import ProceduralTerrain, ProceduralTerrainConfig  # noqa: E402

# Frequent hits on normal terrain, so short runs exercise the perturbation path.
BUSY = CurriculumStage("busy", "normal", True, (1, 2), (0.5, 0.5), (0.3, 0.6),
                       first_hit_after_s=0.2)


def _env(**kw) -> PerpetualFlyEnv:
    cfg = EnvConfig(**kw)
    if "curriculum" not in kw:
        cfg.curriculum = cfg.curriculum + [BUSY]
    return PerpetualFlyEnv(cfg)


def test_check_env():
    from gymnasium.utils.env_checker import check_env

    env = _env(max_episode_s=1.0)
    check_env(env, skip_render_check=True)
    env.close()


def test_obs_layout_fixed():
    env = _env(stage="busy")
    names = [n for n, _ in env.obs_layout]
    assert names == ["gravity_body", "heading_err_sincos", "angvel_body", "linvel_body",
                     "height", "joint_pos", "joint_vel", "leg_contact", "cpg_phase_sincos",
                     "prev_action"]
    assert env.observation_space.shape == (156,)
    assert env.action_space.shape == (42,)
    env.close()
    no_h = _env(obs_height=False, adhesion_residual=True)
    assert no_h.observation_space.shape == (156 - 1 + 6,)
    assert no_h.action_space.shape == (48,)
    no_h.close()


def test_zero_action_reproduces_plain_simulation_exactly():
    """Residual env with a == 0 (normal terrain + hits) vs the plain Simulation /
    ProceduralTerrain / Perturbation / AutoPerturber wiring with the same seeds."""
    env = _env(stage="busy")
    _, info = env.reset(seed=123)
    seeds = info["seeds"]

    app = AppConfig()
    app.controller.seed = seeds["controller"]
    terrain = ProceduralTerrain(ProceduralTerrainConfig(difficulty="normal",
                                                        seed=seeds["terrain"]))
    sim = Simulation(app, world_factory=terrain.build_world)
    terrain.attach(sim)
    pert = Perturbation(sim, app.perturbation)
    pert.rng = np.random.default_rng(seeds["perturbation"])
    AutoPerturber(sim, pert, BUSY.auto_perturb_config(seeds["auto_perturb"]))
    sim.reset()

    assert np.array_equal(env.sim.data.qpos, sim.data.qpos)
    n_hits = 0
    for _ in range(300):  # 1.5 s
        _, _, term, trunc, info = env.step(np.zeros(42, dtype=np.float32))
        n_hits += len(info["hits"])
        sim.step(env.cfg.control_every_steps)
        assert np.array_equal(env.sim.data.qpos, sim.data.qpos)
        assert np.array_equal(env.sim.data.qvel, sim.data.qvel)
        assert np.array_equal(env.sim.data.ctrl, sim.data.ctrl)
        assert not (term or trunc)
    assert n_hits >= 2 and len(pert.events) == n_hits
    sim.close()
    env.close()


def test_residual_changes_joint_targets():
    env = _env()
    env.reset(seed=0)
    a = np.zeros(42, dtype=np.float32)
    a[5] = 1.0
    env.step(a)
    ctrl = env.sim.data.ctrl[env._ctrl._pos_ids]
    expected = env.last_baseline_targets + env.cfg.action_scale * a
    assert np.allclose(ctrl, expected)
    assert np.isclose(ctrl[5] - env.last_baseline_targets[5], env.cfg.action_scale)
    env.close()


def test_obs_finite_in_range_and_hits_not_observed():
    env = _env(stage="busy")
    obs, _ = env.reset(seed=4)
    rng = np.random.default_rng(0)
    sl = env.obs_slices()
    hits = 0
    for i in range(300):
        a = (0.3 * rng.uniform(-1, 1, 42)).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        assert env.observation_space.contains(obs)
        assert set(info["reward_terms"]) >= {"velocity", "upright", "alive", "energy",
                                             "action_rate", "instability", "fallen",
                                             "backward", "recovery", "termination"}
        hits += len(info["hits"])
        if term or trunc:
            break
        if i == 100:  # walking, before much can go wrong: signals are O(1)
            assert np.abs(obs[sl["joint_pos"]]).max() < 6
            assert np.abs(obs[sl["gravity_body"]][2] + 1.0) < 0.1  # upright
            assert abs(obs[sl["height"]][0]) < 3
    assert hits >= 1
    # The observation never reads external forces: with a large force written into
    # xfrc_applied the observation of the same state is identical.
    before = env._observation()
    env.sim.data.xfrc_applied[env.sim.thorax_body_id, :3] = [30.0, -20.0, 10.0]
    after = env._observation()
    env.sim.data.xfrc_applied[:] = 0.0
    assert np.array_equal(before, after)
    env.close()


def test_standing_still_earns_less_than_walking():
    rates = {}
    for base in ("stand", "hybrid"):
        env = _env(baseline=base, max_episode_s=2.0)
        env.reset(seed=1)
        total, n = 0.0, 0
        while True:
            _, r, term, trunc, _ = env.step(np.zeros(42, dtype=np.float32))
            total, n = total + r, n + 1
            if term or trunc:
                break
        rates[base] = total / (n * env.dt)
        env.close()
    rc = EnvConfig().reward
    # Even an upright, never-FALLEN standing fly gets at most upright + alive.
    assert rc.upright + rc.alive < rates["hybrid"]
    assert rates["stand"] < rates["hybrid"]
    assert rates["hybrid"] > 2.0  # ~2.7 per sim second


def test_terminates_after_sustained_fall():
    env = _env(max_episode_s=5.0, fall_terminate_after_s=1.0)
    env.reset(seed=0)
    for _ in range(100):
        env.step(np.zeros(42, dtype=np.float32))
    # 5 body weights for 20 ms from the side: knocks the baseline over (seed 0).
    env.perturbation.apply_impulse("thorax", "right", magnitude=5.0, duration=0.02)
    while True:
        _, r, term, trunc, info = env.step(np.zeros(42, dtype=np.float32))
        if term or trunc:
            break
    assert term and not trunc
    assert info["fall_state"] == "FALLEN" and info["down_for_s"] > 1.0
    assert info["reward_terms"]["termination"] == -env.cfg.reward.fall_termination
    assert info["episode_stats"]["n_falls"] == 1
    with pytest.raises(RuntimeError):
        env.step(np.zeros(42, dtype=np.float32))
    env.close()


def _rollout(env, seed, n=60):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = [obs]
    for _ in range(n):
        obs, r, *_ = env.step(rng.uniform(-1, 1, 42).astype(np.float32))
        out.append(np.append(obs, r))
    return info["seeds"], np.concatenate(out)


def test_seeding_determinism_and_sim_reuse():
    env = _env(stage="busy")
    s1, a = _rollout(env, 7)
    _, other = _rollout(env, 8)
    s2, b = _rollout(env, 7)  # same env (Simulation reused), same seed
    assert s1 == s2 and np.array_equal(a, b)
    assert not np.array_equal(a, other)
    env2 = _env(stage="busy")  # fresh env
    _, c = _rollout(env2, 7)
    assert np.array_equal(a, c)
    env.close()
    env2.close()


def test_stage_switch_on_reset():
    env = _env()
    _, info = env.reset(seed=0, options={"stage": "normal"})
    assert info["stage"] == "normal" and env.terrain.generator.difficulty == "normal"
    assert env.auto.enabled
    _, info = env.reset(seed=0, options={"stage": 0})
    assert info["stage"] == "flat" and not env.auto.enabled
    with pytest.raises(ValueError):
        env.set_stage("nope")
    env.close()


def test_instability_terminates_loudly(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # MuJoCo writes its NaN warnings to ./MUJOCO_LOG.TXT
    env = _env()
    env.reset(seed=0)
    env.step(np.zeros(42, dtype=np.float32))
    env.sim.data.qvel[10] = np.nan
    obs, r, term, trunc, info = env.step(np.zeros(42, dtype=np.float32))
    assert term and info["instability"] and r == -env.cfg.reward.instability_termination
    assert np.all(np.isfinite(obs))
    assert "SimulationInstabilityError" in capsys.readouterr().err
    env.reset(seed=0)  # recovers with an explicit reset
    _, _, term, _, _ = env.step(np.zeros(42, dtype=np.float32))
    assert not term
    env.close()


def test_core_package_does_not_import_rl_deps():
    code = ("import sys, fly_simulator.app, fly_simulator.simulation; "
            "assert 'gymnasium' not in sys.modules and 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
