"""Action library (fly_simulator/actions, docs/ACTIONS.md)."""

import mujoco as mj
import numpy as np
import pytest

from fly_simulator import AppConfig, Simulation
from fly_simulator.actions import (
    ActionManager,
    BackAway,
    Freeze,
    Groom,
    Jump,
    ProboscisExtend,
    TurnInPlace,
    WingRaise,
    load_grooming_clip,
    make_action,
    make_action_fly_factory,
)


@pytest.fixture(scope="module")
def rig():
    sim = Simulation(AppConfig())
    mgr = ActionManager(sim)
    yield sim, mgr
    mgr.detach()
    sim.close()


def _walk(sim, seconds):
    p0 = sim.thorax_position()
    sim.step(int(round(seconds / sim.timestep)))
    return (sim.thorax_position() - p0)[0] / seconds


def _run(sim, mgr, action, walk_before=0.35):
    """Reset, walk, run ``action`` to completion (incl. hand-back blend).
    Returns (end event, samples of (phase, thorax z, tilt))."""
    sim.reset()
    sim.step(int(walk_before / sim.timestep))
    mgr.trigger(action)
    samples = []
    sim.step(1)
    while mgr.busy:
        sim.step(10)
        samples.append((mgr.phase(), sim.thorax_position()[2], sim.tilt_deg()))
        assert sim.time < 20, "action did not finish"
    ends = [e for e in mgr.history if e.kind in ("end", "cancel") and e.name == action.name]
    return ends[-1], samples


def test_idle_manager_does_not_change_walking():
    a = Simulation(AppConfig())
    b = Simulation(AppConfig())
    ActionManager(b)
    a.step(3000)
    b.step(3000)
    np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
    a.close()
    b.close()


def test_jump_takes_off_lands_and_walks_on(rig):
    sim, mgr = rig
    gain0 = sim.model.actuator_gainprm[:, 0].copy()
    z_stand = sim.thorax_position()[2]
    ev, samples = _run(sim, mgr, Jump())
    info = ev.info
    phases = [p for p, _, _ in samples]
    for ph in ("crouch", "stroke", "flight", "landing", "handback"):
        assert ph in phases
    assert ev.kind == "end"
    assert info["apex_dz_mm"] >= 1.5  # measured ~3 mm (see docs/ACTIONS.md)
    assert max(z for _, z, _ in samples) > z_stand + 1.5
    assert 0.02 < info["airtime_s"] < 0.2
    assert info["landed_upright"]
    assert info["max_tilt_deg"] < 90
    np.testing.assert_array_equal(sim.model.actuator_gainprm[:, 0], gain0)
    assert _walk(sim, 0.6) > 8.0  # back to ~14 mm/s forward walking
    assert sim.tilt_deg() < 30


def test_jump_boost_is_temporary_and_reset_restores(rig):
    sim, mgr = rig
    m = sim.model
    gain0, bias0, frc0 = (m.actuator_gainprm[:, 0].copy(), m.actuator_biasprm[:, 1].copy(),
                          m.actuator_forcerange.copy())
    sim.reset()
    sim.step(3000)
    mgr.trigger(Jump(boost=2.0))
    sim.step(1)
    while mgr.phase() != "stroke":
        sim.step(1)
    sim.step(1)
    assert m.actuator_gainprm[:, 0].max() == pytest.approx(2 * gain0.max())
    assert m.actuator_forcerange.max() == pytest.approx(2 * frc0.max())
    sim.reset()  # mid-stroke reset cancels the jump and restores the physics
    assert not mgr.busy
    assert mgr.history[-1].kind == "cancel"
    np.testing.assert_array_equal(m.actuator_gainprm[:, 0], gain0)
    np.testing.assert_array_equal(m.actuator_biasprm[:, 1], bias0)
    np.testing.assert_array_equal(m.actuator_forcerange, frc0)
    assert _walk(sim, 0.4) > 8.0


def test_freeze_holds_then_resumes(rig):
    sim, mgr = rig
    sim.reset()
    sim.step(3500)
    mgr.trigger(Freeze(duration=0.8))
    sim.step(1)
    sim.step(2000)  # settle
    assert mgr.phase() == "hold"
    v_hold = _walk(sim, 0.4)
    assert abs(v_hold) < 0.5
    while mgr.busy:
        sim.step(10)
    ev = mgr.history[-1]
    assert ev.name == "freeze" and ev.kind == "end"
    assert ev.info["drift_mm"] < 0.3
    assert sim.tilt_deg() < 20
    assert _walk(sim, 0.6) > 8.0


def test_triggered_groom_still_uses_recorded_clip(rig):
    # ActionManager.trigger sets action.source to who triggered it; that must not
    # switch Groom from the recorded clip to the synthetic sweep (regression).
    sim, mgr = rig
    sim.reset()
    sim.step(100)
    for trigger in ("key", "brain", "api"):
        g = Groom(duration=0.1)
        mgr.trigger(g, source=trigger)
        sim.step(1)
        assert g.clip_source == "recorded" and g.source == trigger
        assert hasattr(g, "_clip")  # begin() loaded the recording
        while mgr.busy:
            sim.step(50)


def test_groom_replays_recorded_front_legs(rig):
    sim, mgr = rig
    angles, names, fps = load_grooming_clip()
    assert angles.shape == (600, 14) and fps == 200
    assert all(n.split("-")[1].startswith(("lf_", "rf_")) for n in names)
    sim.reset()
    sim.step(3000)
    mgr.trigger(Groom(duration=2.0))
    sim.step(1)
    b = mgr.body
    front = np.flatnonzero(b.leg_mask(("lf", "rf")))
    t5 = [mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_BODY, f"nmf/{leg}_tarsus5") for leg in ("lf", "rf")]
    err, tarsus_z, tilt = [], [], []
    while mgr.action is not None:
        sim.step(20)
        if sim.time - mgr._t0 > 0.4:
            d = sim.data
            err.append(np.abs(d.qpos[b.qpos_adr[front]] - d.ctrl[b.pos_ids[front]]).mean())
            tarsus_z.append(d.xpos[t5, 2].min())
            tilt.append(sim.tilt_deg())
    assert np.degrees(np.mean(err)) < 5.0  # front legs follow the recording
    assert min(tarsus_z) > 0.05  # rubbing happens in the air (mid legs raise the body)
    assert max(tilt) < 30
    while mgr.busy:
        sim.step(10)
    assert _walk(sim, 0.6) > 8.0


def test_back_away_and_turn(rig):
    sim, mgr = rig
    ev, _ = _run(sim, mgr, BackAway(duration=1.0))
    assert ev.info["forward_mm"] < -4.0
    assert _walk(sim, 0.8) > 6.0
    ev, _ = _run(sim, mgr, TurnInPlace("left", duration=0.8))
    assert ev.info["turn_deg"] > 45
    ev, _ = _run(sim, mgr, TurnInPlace("right", duration=0.8))
    assert ev.info["turn_deg"] < -45


def test_cancel_and_replace(rig):
    sim, mgr = rig
    sim.reset()
    sim.step(2000)
    mgr.trigger(Freeze(duration=5.0))
    sim.step(1000)
    assert mgr.trigger(Groom(), replace=False) is False
    mgr.cancel()
    assert mgr.history[-1].kind == "cancel"
    while mgr.busy:
        sim.step(10)
    assert _walk(sim, 0.5) > 8.0


def test_jump_is_deterministic(rig):
    sim, mgr = rig
    _run(sim, mgr, Jump())
    q1 = sim.data.qpos.copy()
    _run(sim, mgr, Jump())
    np.testing.assert_array_equal(sim.data.qpos, q1)


def test_registry():
    assert make_action("jump", boost=1.5).p.boost == 1.5
    assert make_action("freeze", duration=2.0).duration == 2.0
    with pytest.raises(KeyError):
        make_action("fly_away")


def test_extra_joints_keep_walking_and_move_wings_proboscis(rig):
    sim0, mgr0 = rig
    with pytest.raises(RuntimeError, match="fly_factory"):
        mgr0.trigger(WingRaise())
    assert not mgr0.busy
    sim0.reset()
    sim0.step(10000)
    x_default = sim0.thorax_position()[0]

    sim = Simulation(AppConfig(), fly_factory=make_action_fly_factory(wings=True, proboscis=True))
    assert len(sim.controller._pos_ids) == 42  # the controller still only sees the legs
    sim.step(10000)
    assert sim.thorax_position()[0] == pytest.approx(x_default, rel=0.03)
    mgr = ActionManager(sim)
    m, d = sim.model, sim.data

    def q(name):
        return d.qpos[m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"nmf/{name}")]]

    mgr.trigger(WingRaise(duration=0.6))
    sim.step(4000)
    assert q("c_thorax-l_wing-yaw") == pytest.approx(1.2, abs=0.15)
    assert q("c_thorax-r_wing-yaw") == pytest.approx(1.2, abs=0.15)
    while mgr.busy:
        sim.step(10)
    sim.step(2000)
    assert abs(q("c_thorax-l_wing-yaw")) < 0.15
    mgr.trigger(ProboscisExtend(duration=0.6))
    sim.step(4000)
    assert q("c_head-c_rostrum-pitch") == pytest.approx(-1.0, abs=0.15)
    while mgr.busy:
        sim.step(10)
    sim.step(2000)
    assert abs(q("c_head-c_rostrum-pitch")) < 0.15
    assert _walk(sim, 0.5) > 8.0
    sim.close()


# --------------------------------------------------------------------- brain triggers
def _state(t, escape=0.0, mn9=0.0, groom=0.0):
    from types import SimpleNamespace

    return SimpleNamespace(brain_time=t, window_s=0.1, probes={"MN9": mn9},
                           descending={"escape": escape, "groom": groom})


def test_brain_triggers_jump_proboscis_groom():
    from fly_simulator.actions.brain_triggers import BrainActionTriggers, TriggerParams

    sim = Simulation(AppConfig(), fly_factory=make_action_fly_factory())
    mgr = ActionManager(sim)
    trig = BrainActionTriggers(mgr, TriggerParams())
    assert trig.can_proboscis
    sim.step(2000)
    # quiet brain: nothing
    assert trig.on_state(_state(0.1)) == [] and not mgr.busy
    # giant fiber above 60 Hz -> jump (source "brain"), refractory 1.5 s
    assert "JUMP" in trig.on_state(_state(0.2, escape=120))[0]
    sim.step(1)
    assert mgr.active_name == "jump" and mgr.history[-1].info["source"] == "brain"
    assert trig.on_state(_state(0.3, escape=150)) == []  # refractory
    while mgr.busy:
        sim.step(50)
    # MN9 -> proboscis, prolonged while MN9 stays high (no re-trigger)
    assert "PROBOSCIS" in trig.on_state(_state(1.0, mn9=90))[0]
    sim.step(2000)
    d0 = mgr.action.duration
    assert trig.on_state(_state(1.1, mn9=90)) == []
    assert mgr.action.duration > d0
    # DNg12: one state above 20 Hz (= 100 ms) -> groom, but not while busy
    assert trig.on_state(_state(1.2, groom=35)) == []  # proboscis still running
    while mgr.busy:
        sim.step(50)
    assert trig.on_state(_state(2.0, groom=10)) == []  # below threshold
    assert "GROOM" in trig.on_state(_state(2.1, groom=35))[0]
    sim.step(1)
    assert mgr.active_name == "groom"
    # a GF burst interrupts grooming (escape has priority)
    sim.step(3000)
    assert "JUMP" in trig.on_state(_state(2.5, escape=100))[0]
    sim.step(1)
    assert mgr.active_name == "jump"
    assert trig.counts == {"jump": 2, "proboscis": 1, "groom": 1}
    sim.close()


def test_brain_triggers_without_proboscis_joints(rig):
    from fly_simulator.actions.brain_triggers import BrainActionTriggers

    sim, mgr = rig
    trig = BrainActionTriggers(mgr)
    mgr.listeners.remove(trig._on_event)
    assert not trig.can_proboscis
    sim.reset()
    assert trig.on_state(_state(0.1, mn9=120)) == [] and not mgr.busy


# ------------------------------------------------------------------------ app
def test_cli_full_body_and_brain_actions_flags():
    from fly_simulator.app import build_arg_parser, config_from_args

    parse = lambda *a: config_from_args(build_arg_parser().parse_args(list(a)))  # noqa: E731
    assert AppConfig().fly.extra_joints is False  # library default: FlyGym's model
    assert parse().fly.extra_joints is True  # interactive CLI default
    assert parse("--no-full-body").fly.extra_joints is False
    c = parse("--brain-actions")
    assert c.brain.enabled and c.brain.steer and c.brain.actions
    assert not parse("--brain").brain.actions


def test_help_overlay_greys_unavailable_actions():
    from fly_simulator.app import UNAVAILABLE_MARK, help_groups, key_help_text
    from fly_simulator.interaction.viewer import compose_frame

    legs_only = {"jump", "freeze", "groom", "back_away", "turn_left", "turn_right"}
    rows = dict(dict(help_groups(legs_only))["actions"])
    assert rows["W"].startswith(UNAVAILABLE_MARK) and rows["N"].startswith(UNAVAILABLE_MARK)
    assert not rows["J"].startswith(UNAVAILABLE_MARK)
    rows = dict(dict(help_groups(legs_only | {"wings", "proboscis"}))["actions"])
    assert not rows["W"].startswith(UNAVAILABLE_MARK)
    assert "(unavailable)" in key_help_text(legs_only)
    img = compose_frame(np.zeros((480, 720, 3), np.uint8), None, help_groups=help_groups(legs_only))
    assert img.shape[:2] == (480, 720) and img.any()


def test_app_action_keys_log_and_pause_falls(tmp_path):
    import csv
    import json

    from fly_simulator.app import run

    cfg = AppConfig()
    cfg.fly.extra_joints = True
    cfg.logging.runs_dir = str(tmp_path)
    res = run(cfg, headless=True, max_seconds=4.0,
              script_keys=[(0.6, "j"), (1.6, "n"), (1.7, "w"), (2.4, "z"), (3.3, "comma")])
    assert res.n_falls == 0 and res.final_state.lower() in ("upright", "destabilized")
    run_dir = res.run_dir
    rows = list(csv.DictReader(open(f"{run_dir}/events.csv")))
    starts = [json.loads(r["details"]) for r in rows if r["event_type"] == "action_start"]
    assert [s["action"] for s in starts] == ["jump", "proboscis", "wings", "freeze", "turn_left"]
    assert all(s["source"] == "key" for s in starts)
    ends = [json.loads(r["details"]) for r in rows if r["event_type"] in ("action_end", "action_cancel")]
    jump = next(e for e in ends if e["action"] == "jump")
    assert jump["apex_dz_mm"] > 1.5 and jump["landed_upright"]
    summary = json.loads(open(f"{run_dir}/summary.json").read())
    assert summary["full_body"] and summary["actions"]["jump"] == 1


def test_session_jump_pauses_auto_hits_and_fall_detector():
    from fly_simulator.app import Session
    from fly_simulator.metrics import FallState

    cfg = AppConfig()
    s = Session(cfg, log=False, say=lambda m: None)
    try:
        s.sim.step(3000)
        s.auto.enabled = True
        s.up_since = -1e9
        assert s.auto_hits_allowed()
        assert s.trigger_action("jump").startswith("[action] jump")
        assert s.trigger_action("wings").endswith("(run with --full-body)")  # legs-only body
        s.sim.step(1)
        states = set()
        while s.actions.busy:
            if s.actions.active_name == "jump":
                assert not s.auto_hits_allowed()
            s.sim.step(20)
            states.add(s.detector.state)
        assert FallState.FALLEN not in states and s.metrics.n_falls == 0
        s.auto.enabled = False
        # a 3 s groom (standing still) is not read as "stuck / no progress"
        s.trigger_action("groom", duration=3.5)
        s.sim.step(1)
        while s.actions.busy:
            s.sim.step(50)
        assert s.metrics.n_falls == 0
    finally:
        s.close("test")
        s.sim.close()


def test_full_body_walking_stats_match_default():
    from fly_simulator.app import run

    out = []
    for extra in (False, True):
        cfg = AppConfig()
        cfg.fly.extra_joints = extra
        out.append(run(cfg, headless=True, max_seconds=3.0, log=False))
    a, b = out
    assert b.forward_displacement == pytest.approx(a.forward_displacement, rel=0.02)
    assert b.final_thorax_pos[2] == pytest.approx(a.final_thorax_pos[2], abs=0.1)
    assert abs(b.final_thorax_pos[1]) < 5.0


def test_brain_link_installs_action_triggers_synthetic(tmp_path):
    pytest.importorskip("numba")
    from fly_simulator.app import Session
    from fly_simulator.brain_link import BrainLink, BrainLinkConfig

    cfg = AppConfig()
    cfg.fly.extra_joints = True
    cfg.brain = BrainLinkConfig(enabled=True, window=False, steer=True, actions=True,
                                synthetic={"n": 50, "p_conn": 0.1, "seed": 0})
    link = BrainLink(cfg.brain, headless=True, say=lambda m: None)
    link.wait_ready(60)
    s = Session(cfg, log=False, say=lambda m: None, brain=link)
    try:
        assert link.triggers is not None and link.triggers.can_proboscis
        for _ in range(10):
            s.sim.step(150)
            s.after_physics()
        assert "groom" in (link.latest.descending if link.latest else {"groom": 0})
        link.triggers.on_state(_state(1.0, escape=200))
        s.sim.step(1)
        assert s.actions.active_name == "jump"
        assert any("brain actions ON" in line for line in link.hud_lines()) or link.latest is None
        s.sim.reset()  # resets the refractory timers too
        assert link.triggers._next_jump < 0
    finally:
        s.close("test")
        link.close()
