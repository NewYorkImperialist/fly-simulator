"""Squash damage (fly_simulator/injury.py, docs/INJURY.md)."""

from types import SimpleNamespace

import numpy as np
import pytest

from fly_simulator.app import Session, build_arg_parser, config_from_args
from fly_simulator.config import AppConfig
from fly_simulator.injury import (ABDOMEN, L_WING, LEGS, PARTS, R_WING, THORAX, DamageSensor,
                                  InjuryConfig, bar, install_injury)


def _cfg(injury: bool, terrain: str = "flat") -> AppConfig:
    cfg = AppConfig()
    cfg.logging.enabled = False
    cfg.terrain.difficulty = terrain
    cfg.injury.enabled = injury
    return cfg


@pytest.fixture(scope="module")
def session():
    s = Session(_cfg(True), log=False, say=lambda m: None)
    yield s
    s.close("test")


def _fresh(s):
    s.sim.reset()
    s.injury.cfg = InjuryConfig(enabled=True)
    s.injury.sensor.cfg = s.injury.cfg
    return s.injury


def test_off_is_inert_and_default():
    assert AppConfig().injury.enabled is False
    args = build_arg_parser().parse_args([])
    assert config_from_args(args).injury.enabled is False
    assert config_from_args(build_arg_parser().parse_args(["--injury"])).injury.enabled
    s = Session(_cfg(False), log=False, say=lambda m: None)
    try:
        assert s.injury is None
        assert not any(s.sim.model.body(i).name.startswith("injury/")
                       for i in range(s.sim.model.nbody))
        hooks = (list(s.sim.pre_step_hooks), list(s.sim.post_step_hooks),
                 list(s.sim.reset_hooks))
        h = install_injury(s, InjuryConfig(enabled=False))
        assert not h.enabled and h.hud_lines() == [] and h.sensor is None
        assert hooks == (s.sim.pre_step_hooks, s.sim.post_step_hooks, s.sim.reset_hooks)
        net = s.sim.controller.impl.cpg_network
        assert "step" not in vars(net)  # the CPG is not wrapped
        h.update()
        h.close()
    finally:
        s.close("test")


def _walk_qpos(injury: bool, steps: int = 10000) -> np.ndarray:
    s = Session(_cfg(injury, terrain="normal"), log=False, say=lambda m: None)
    try:
        for _ in range(steps // 500):
            s.step(500)
            s.after_physics()
        fly = s.sim.fly_dofs
        return np.concatenate([s.sim.data.qpos[:7], s.sim.data.qvel[fly]])
    finally:
        s.close("test")


def test_default_run_bit_identical_and_unhurt_run_identical():
    """Off: nothing is installed (bit-identical by construction, checked above). On,
    without hits: the sensor only reads contacts and the hooks are no-ops at zero
    damage, so the fly's trajectory is bit-identical too."""
    a = _walk_qpos(False)
    b = _walk_qpos(False)
    c = _walk_qpos(True)
    assert np.array_equal(a, b)
    assert np.array_equal(a, c)


def test_damage_from_synthetic_forces(session):
    inj = _fresh(session)
    sens: DamageSensor = inj.sensor
    cfg = inj.cfg
    bw = sens.bw
    fake = SimpleNamespace(timestep=1e-4, time=10.0, model=session.sim.model)
    # below the floors: nothing
    sens.force[:] = 0.0
    sens.force[0] = 0.9 * cfg.leg_floor_bw * bw
    sens._integrate(fake, None)
    assert not sens.damage.any() and sens._ep is None
    # 100 steps of 3 x floor on the thorax: D = 2 floor * dt * 100 / ref
    for k in range(100):
        fake.time = 10.0 + k * 1e-4
        sens.force[:] = 0.0
        sens.force[THORAX] = 3 * cfg.body_floor_bw * bw
        sens._integrate(fake, None)
    want = 2 * cfg.body_floor_bw * 1e-4 * 100 / cfg.thorax_ref_bws
    assert sens.damage[THORAX] == pytest.approx(want, rel=1e-9)
    assert sens.damage[ABDOMEN] == 0.0
    # the episode closes after the gap; the sharp-impact term is added then
    sens.force[:] = 0.0
    fake.time += cfg.episode_gap_s + 1e-3
    sens._integrate(fake, None)
    assert len(sens.episodes) == 1
    ev = sens.episodes.popleft()
    peak = 3 * cfg.body_floor_bw
    sharp = min(cfg.peak_cap, max(0.0, peak - cfg.peak_floor_bw) / cfg.peak_ref_bw)
    assert ev.damage["thorax"] == pytest.approx(want + sharp, rel=1e-9)
    assert ev.peak_bw["thorax"] == pytest.approx(peak)
    # damage is capped
    sens.force[:] = 0.0
    sens.force[LEGS.index("lh")] = 1e6 * bw
    sens._integrate(fake, None)
    assert sens.damage[LEGS.index("lh")] == cfg.max_damage


def test_leg_gain_reduction_and_lame(session):
    inj = _fresh(session)
    sim = session.sim
    m = sim.model
    i_rm = LEGS.index("rm")
    inj.sensor.damage[i_rm] = 0.5
    inj.update()
    assert inj.status == "limping"
    assert inj.leg_amp[i_rm] == pytest.approx(1 - inj.cfg.leg_amp_loss * 0.5)
    assert inj.leg_amp[LEGS.index("lm")] == 1.0
    dofs = inj._leg_of == i_rm
    kp = m.actuator_gainprm[inj._pos_ids, 0]
    assert np.allclose(kp[dofs], inj._kp0[dofs] * (1 - inj.cfg.leg_kp_loss * 0.5))
    assert np.array_equal(kp[~dofs], inj._kp0[~dofs])
    assert np.allclose(m.actuator_biasprm[inj._pos_ids, 1][dofs], -kp[dofs])
    # the CPG sees the per-leg amplitude only inside its step
    net = sim.controller.impl.cpg_network
    seen = {}
    orig = inj._orig_net_step
    inj._orig_net_step = lambda: seen.update(amps=net.intrinsic_amps.copy())
    try:
        base = net.intrinsic_amps.copy()
        net.step()
    finally:
        inj._orig_net_step = orig
    assert seen["amps"][i_rm] == pytest.approx(base[i_rm] * inj.leg_amp[i_rm])
    assert np.array_equal(net.intrinsic_amps, base)
    # lame: tucked targets, adhesion off (applied after the controller)
    inj.sensor.damage[i_rm] = 0.9
    inj.update()
    assert inj.lame[i_rm] and inj.status == "lame"
    sim.step(20)
    ctrl = sim.data.ctrl
    assert np.allclose(ctrl[inj._pos_ids[dofs]], inj._tuck[dofs])
    assert ctrl[inj._adh_ids[i_rm]] == 0.0
    # hysteresis: stays lame until below lame_clear
    inj.sensor.damage[i_rm] = 0.6
    inj.update()
    assert inj.lame[i_rm]
    inj.sensor.damage[i_rm] = 0.5
    inj.update()
    assert not inj.lame[i_rm]
    # thorax damage: slower stepping
    inj.sensor.damage[THORAX] = 0.4
    inj.update()
    assert inj.freq_mult == pytest.approx(1 - inj.cfg.freq_loss * 0.4)


def test_healing(session):
    inj = _fresh(session)
    inj.cfg.heal_delay_s = 0.05
    inj.cfg.heal_rate_per_s = 1.0
    inj.sensor.damage[LEGS.index("lf")] = 0.5
    inj.sensor.damage[L_WING] = 0.02
    inj.sensor.t_last_damage = session.sim.time
    inj.update()
    session.step(300)  # 0.03 s: still within the delay
    inj.update()
    assert inj.damage[LEGS.index("lf")] == 0.5
    session.step(700)  # t = 0.1 s since the damage
    inj.update()
    assert inj.damage[LEGS.index("lf")] == pytest.approx(0.5 - 0.07, abs=1e-6)
    assert inj.damage[L_WING] == 0.0  # floored at 0
    assert inj.max_damage[LEGS.index("lf")] >= 0.5


def test_stun_from_a_big_episode(session):
    inj = _fresh(session)
    ev_body = 1.0
    inj.sensor.episodes.append(SimpleNamespace(
        kind="impact", time=session.sim.time, total=1.2, damage={"thorax": ev_body},
        peak_bw={"thorax": 900.0}, impulse_bws={}, sources={"swatter/paddle": 1.0},
        info={"body": ev_body, "body_peak_bw": 900.0},
        to_dict=lambda: {"kind": "impact", "time": 0.0, "total": 1.2}))
    inj.update()
    assert inj.n_stuns == 1 and inj.status == "stunned"
    session.step(20)
    assert session.actions.active_name == "stunned"
    session.sim.reset()
    assert inj.status == "OK" and session.actions.active_name is None


def test_squash_then_counted_respawn(session):
    inj = _fresh(session)
    inj.cfg.squash_show_s = 0.02
    sim, m = session.sim, session.sim.model
    session.step(2000)
    x, y, _ = sim.thorax_position()
    resets = session.metrics.n_resets
    inj.sensor.damage[:] = 0.7
    inj._squash("test")
    assert inj.status == "squashed" and inj.n_squashes == 1
    flat = inj._flat_mocap
    assert np.allclose(sim.data.mocap_pos[flat][:2], (x, y))
    assert (m.geom_group[inj._fly_geoms] == 5).all()  # the fly itself is hidden
    session.step(300)
    inj.update()  # -> explicit, counted reset
    assert session.n_squash_resets == 1
    assert session.metrics.n_resets == resets + 1
    assert not inj.damage.any() and inj.status == "OK"
    assert np.array_equal(m.geom_group[inj._fly_geoms], inj._fly_groups)
    assert np.array_equal(m.actuator_gainprm[inj._pos_ids, 0], inj._kp0)
    assert sim.data.mocap_pos[flat][2] < -100  # replica parked, the stain stays
    assert np.allclose(sim.data.mocap_pos[inj._splat_mocap][:2], (x, y))
    s = inj.summary()
    assert s["n_squashes"] == 1 and set(s["damage"]) == set(PARTS)


def test_wing_cap_in_flight(session):
    from dataclasses import replace

    from fly_simulator.flight.control import base_wingbeat

    inj = _fresh(session)
    sim = session.sim
    base = base_wingbeat()

    class FakeHover:
        def __init__(self):
            self.base = base

        def _write(self, sim_, u):
            sim_.wingbeat.params = replace(base)

    sim.wingbeat = SimpleNamespace(params=None)
    sim.flight_controller = FakeHover()
    try:
        inj.sensor.damage[R_WING] = 1.0
        inj.update()
        sim.flight_controller._write(sim, None)
        p = sim.wingbeat.params
        assert p.left.amplitude == base.left.amplitude
        assert p.right.amplitude == pytest.approx(base.right.amplitude * (1 - inj.cfg.wing_amp_loss))
    finally:
        del sim.wingbeat
        sim.flight_controller = None


def test_hud_and_bars(session):
    inj = _fresh(session)
    assert bar(1.0) == "#####" and bar(0.0) == "....." and bar(0.5, 4) == "##.."
    inj.sensor.damage[LEGS.index("lh")] = 1.0
    inj.update()
    lines = inj.hud_lines()
    assert lines[0].startswith("INJURY LAME") and "LHx....." in lines[1]
    assert len(inj.metric_row()) == 6
