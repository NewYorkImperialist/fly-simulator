"""--flight in the app: config, manual take-off / hover / land (key L), arrow
steering, and the brain's giant-fibre escape handing over to real wing flight
(docs/FLIGHT.md section 7)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from fly_simulator import AppConfig
from fly_simulator.app import (
    ConfigError,
    KEY_TABLE,
    Session,
    apply_feature_defaults,
    build_arg_parser,
    config_from_args,
)
from fly_simulator.metrics import FallState


def test_flight_cli_config():
    cfg = config_from_args(build_arg_parser().parse_args(["--flight"]))
    assert cfg.flight.enabled
    assert cfg.sim.timestep == 5e-5 and cfg.sim.control_every_steps == 1
    assert cfg.render.render_every_steps == 300  # same frames per simulated second
    assert not cfg.fly.extra_joints  # the flight body has its own wings
    assert not config_from_args(build_arg_parser().parse_args([])).flight.enabled
    # idempotent (a second pass does not double the render interval again)
    assert apply_feature_defaults(cfg).render.render_every_steps == 300
    assert any(k == "L" for k, _ in KEY_TABLE)


@pytest.mark.parametrize("extra", [["--job", "sisyphus"], ["--course", "gauntlet"]])
def test_flight_refuses_jobs_and_courses(extra):
    with pytest.raises(ConfigError, match="--flight"):
        config_from_args(build_arg_parser().parse_args(["--flight", *extra]))


@pytest.fixture(scope="module")
def session():
    cfg = AppConfig()
    cfg.terrain.difficulty = "flat"
    cfg.flight.enabled = True
    msgs: list[str] = []
    s = Session(cfg, log=False, say=msgs.append)
    s.msgs = msgs
    yield s
    s.close("test")
    s.sim.close()


def _run_until(s, cond, max_s, chunk=40):
    n = int(round(max_s / s.sim.timestep / chunk))
    for _ in range(n):
        s.step(chunk)
        s.after_physics()
        if cond():
            return True
    return False


def test_flight_session_walks_takes_off_hovers_steers_lands(session):
    s = session
    sim, fl = s.sim, s.flight
    assert sim.timestep == 5e-5 and fl.state == "walking" and sim.leg_mode == "walk"
    assert "wings" not in s.available_actions  # W needs the action body's wings
    x0 = sim.thorax_position()[0]
    s.step(int(0.3 / sim.timestep))
    assert sim.thorax_position()[0] - x0 > 1.5  # the gait still walks at dt 5e-5

    assert "take-off" in s.flight_key("l")
    assert _run_until(s, lambda: fl.state == "hovering", 0.3)
    assert sim.flapping and fl.freq > 200 and s.actions.active_name is None
    assert "not while flying" in s.trigger_action("jump")
    no_push = True
    for _ in range(6):  # 0.3 s of hover: wings only, no external force
        s.step(1000)
        no_push &= not sim.data.xfrc_applied[sim.thorax_body_id].any()
    assert no_push and fl.altitude() > 2.0 and fl.state == "hovering"
    assert s.detector.state == FallState.UPRIGHT  # paused while airborne
    assert "FLIGHT HOVERING" in fl.hud_line() and "Hz" in fl.hud_line()

    # arrows steer while airborne (UP = faster)
    assert "speed +50" in s.flight_key("up")
    assert fl.state == "forward"
    p0 = sim.com()
    s.step(int(0.3 / sim.timestep))
    assert np.hypot(*(sim.com() - p0)[:2]) > 4.0
    s.flight_key("down")
    assert fl.state == "hovering"

    assert s.flight_key("l") == "[flight] landing"
    assert _run_until(s, lambda: fl.state == "walking", 2.0)
    ev = fl.events[-1]
    assert ev.kind == "land" and ev.info["tilt_deg"] < 30.0
    assert not sim.flapping and sim.leg_mode == "walk"
    x1 = sim.thorax_position()[0]
    s.step(int(0.3 / sim.timestep))
    assert sim.tilt_deg() < 30.0 and sim.thorax_position()[0] - x1 > 1.0  # walks again
    assert fl.counts["takeoff"] == 1 and fl.counts["land"] == 1 and fl.counts["crash"] == 0


def _gf(t, hz):
    return SimpleNamespace(brain_time=t, window_s=0.02, probes={},
                           descending={"escape": hz, "groom": 0.0})


def test_brain_escape_hands_over_to_real_flight(session):
    from fly_simulator.actions.brain_triggers import BrainActionTriggers, TriggerParams

    s = session
    sim, fl = s.sim, s.flight
    s.reset("test")
    assert fl.state == "walking" and not sim.flapping
    trig = BrainActionTriggers(s.actions, TriggerParams(jump_short_hz=60.0, jump_flight=True))
    trig.flight = fl
    s.step(int(0.3 / sim.timestep))
    # threat 5 mm to the fly's left -> escape to its right
    h = sim.heading()
    left = np.array([-math.sin(h), math.cos(h), 0.0])
    trig.threat_fn = lambda: sim.thorax_position() + 5.0 * left
    msg = trig.on_state(_gf(0.3, 150.0))[0]
    assert "JUMP (short mode)" in msg and "wings" in msg
    assert fl.state == "takeoff"
    assert np.dot(fl._escape_dir, left[:2]) < -0.95
    assert _run_until(s, lambda: fl.state == "forward", 0.2, chunk=10)
    jump = s.actions.history[-1]
    assert jump.name == "jump" and not jump.info.get("flight_assist")  # no emulated push
    p0 = sim.com()
    push = False
    for _ in range(5):
        s.step(1000)
        push |= bool(sim.data.xfrc_applied[sim.thorax_body_id].any())
    d = (sim.com() - p0)[:2]
    assert not push and fl.state == "forward"
    assert np.dot(d, -left[:2]) > 8.0  # flew away from the threat
    # a new GF burst while airborne re-directs the escape (no jump)
    trig._next_jump = -1e9
    trig.threat_fn = lambda: sim.thorax_position() - 5.0 * left
    n_jumps = trig.counts["jump"]
    assert "escape flight" in trig.on_state(_gf(0.6, 150.0))[0]
    assert trig.counts["jump"] == n_jumps and trig.counts["escape_flight"] == 1
    assert np.dot(fl._escape_dir, left[:2]) > 0.95
    s.reset("test")  # reset mid-flight: wings off, walking
    assert fl.state == "walking" and not sim.flapping and sim.leg_mode == "walk"
