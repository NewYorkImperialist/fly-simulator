"""Flyswatter (fly_simulator/interaction/swatter.py, docs/SWATTER.md): model, swat,
hit / dodge measurement, flat-source looming, short-mode + escape-flight jumps,
brain triggers and BrainLink low-latency pacing. No connectome data needed."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from fly_simulator import AppConfig, Simulation
from fly_simulator.actions import ActionManager
from fly_simulator.actions.jump import Jump
from fly_simulator.interaction.swatter import (
    Swatter,
    SwatterConfig,
    install_swatter,
    swatter_response,
)
from fly_simulator.vision.looming import LoomingConfig, LoomResponse, loom_response, quad_view

I3 = np.eye(3)[None]


# ---------------------------------------------------------------- flat-source geometry


def _square(d, a):
    """Square of half side a centred at distance d straight up (z) from the origin."""
    c = np.array([[-a, -a, d], [a, -a, d], [a, a, d], [-a, a, d]], float)
    return c[None]


def test_quad_solid_angle_matches_analytic_square():
    full = LoomingConfig(fov_front_deg=200.0, fov_rear_deg=200.0, fov_elev_min_deg=-100.0)
    cfg = LoomingConfig()
    E = np.zeros((1, 3))
    for d, a in ((5.0, 1.0), (1.0, 3.0), (0.2, 4.0)):
        v = quad_view(E, I3, np.array([1.0]), _square(d, a), full)
        exact = 4.0 * math.asin(a * a / (a * a + d * d))
        assert v["omega"][0] == pytest.approx(exact, rel=1e-6)
        assert v["dist"][0] == pytest.approx(d)
        assert v["el"][0] == pytest.approx(90.0, abs=1e-6)
    # edge-on (eye in the plane of the square): no solid angle
    edge = np.array([[[2, -1, 0], [4, -1, 0], [4, 1, 0], [2, 1, 0]]], float)
    assert quad_view(E, I3, np.array([1.0]), edge, cfg)["omega"][0] == pytest.approx(0.0, abs=1e-9)
    # straight behind the fly (blind sector) at eye level: invisible
    behind = np.array([[[-3, -0.3, -0.3], [-3, 0.3, -0.3], [-3, 0.3, 0.3], [-3, -0.3, 0.3]]])
    assert quad_view(E, I3, np.array([1.0]), behind, cfg)["omega"][0] == 0.0


def test_lc4_size_gate_is_opt_in():
    assert LoomResponse().lc4_theta0 == 0.0  # whip response unchanged
    p = swatter_response()
    assert loom_response(8.0, 2000.0, p)[0] == 0.0  # small: gated
    lc4, lplc2 = loom_response(40.0, 2000.0, p)
    assert lc4 > 100.0 and lplc2 > 50.0
    assert loom_response(40.0, -2000.0, p) == (0.0, 0.0)  # receding: nothing
    assert loom_response(30.0, 50.0, p) == (0.0, 0.0)  # static / slow


# ---------------------------------------------------------------- physical swatter


@pytest.fixture(scope="module")
def rig():
    sw = Swatter()
    sim = Simulation(AppConfig(), world_extensions=[sw.extension])
    mgr = ActionManager(sim)
    h = install_swatter(SimpleNamespace(sim=sim, actions=mgr), swatter=sw)
    yield sim, sw, mgr, h
    sim.close()


def _swat(sim, sw, level, walk=0.4, on_slam=None):
    sim.reset()
    sim.step(int(walk / sim.timestep))
    n_ev = len(sw.events)
    sw.swat(level)
    fired = False
    while sw.phase != "idle":
        sim.step(10)
        if on_slam is not None and not fired and sw.phase == "slam":
            on_slam()
            fired = True
        assert np.all(np.isfinite(sim.data.qpos))
    return sw.events[n_ev]


def test_model_builds_parks_clear_and_never_touches_idle(rig):
    sim, sw, _, _ = rig
    sim.reset()
    assert sw.phase == "idle"
    assert sw.plate_center()[2] > 10.0  # held up high, away from the fly
    sim.step(4000)
    assert sw.stray_contact_steps == 0
    assert sim.thorax_position()[0] > 3.0  # walking, undisturbed


def test_fast_swat_hits_and_slaps(rig):
    sim, sw, _, h = rig
    seen = []
    sw.listeners.append(seen.append)
    n_sent = len(h.vision.sent)
    ev = _swat(sim, sw, 4)
    sw.listeners.remove(seen.append)
    assert seen == [ev] and ev.hit and ev.outcome == "hit"
    assert ev.impulse_uNs > 10.0 and ev.magnitude_bw > 50.0
    assert ev.under_at_slam and ev.under_at_impact and not ev.jumped
    # contact happens at the end of the slam (plate flat at the aim point)
    lv = sw.cfg.levels[3]
    assert ev.t_slam - ev.t_request == pytest.approx(sw.cfg.t_raise + lv.pause_s, abs=1e-3)
    assert 0.7 * lv.slam_s < ev.t_fly_contact - ev.t_slam < lv.slam_s + 0.01
    assert ev.impact_speed_mm_s > 500.0
    assert sim.tilt_deg() < 60.0  # squashed, not blown up
    # the paddle looms strongly during the slam, never while raised / pausing
    sent = h.vision.sent[n_sent:]
    pre = [e for e in sent if e.sim_time < ev.t_slam]
    slam = [e for e in sent if ev.t_slam <= e.sim_time <= ev.t_fly_contact]
    assert not pre
    assert slam and max(max(e.details["lc4_hz"], e.details["lplc2_hz"]) for e in slam) > 150
    assert {e.side for e in slam} == {"left", "right"}


def test_escaping_fly_dodges_and_ground_is_slapped(rig):
    sim, sw, mgr, _ = rig
    ev = _swat(sim, sw, 1, on_slam=lambda: mgr.trigger(Jump(mode="short", flight_assist=True)))
    assert not ev.hit and ev.outcome == "dodged" and ev.jumped
    assert ev.t_ground is not None and ev.ground_impulse_uNs > 1.0  # the plate hit the floor
    assert not ev.under_at_impact
    assert math.dist(ev.fly_at_impact[:2], ev.aim[:2]) > 5.0
    while mgr.busy:
        sim.step(50)
    assert not np.any(sim.data.xfrc_applied[sim.thorax_body_id])  # flight force removed


def test_install_needs_the_extension():
    sim = Simulation(AppConfig())
    with pytest.raises(RuntimeError, match="world_extensions"):
        install_swatter(sim)
    sim.close()


def test_config_roundtrip():
    c = SwatterConfig()
    d = c.to_dict()
    assert SwatterConfig.from_dict(d) == c
    assert [lv.slam_s for lv in c.levels] == sorted([lv.slam_s for lv in c.levels], reverse=True)


# ---------------------------------------------------------------- jump modes


def _jump(sim, mgr, **kw):
    sim.reset()
    sim.step(3000)
    p0 = sim.thorax_position()
    evs = []
    mgr.listeners.append(evs.append)
    mgr.trigger(Jump(**kw))
    while mgr.busy:
        sim.step(50)
    mgr.listeners.remove(evs.append)
    info = [e for e in evs if e.kind == "end"][0].info
    return info, float(np.hypot(*(sim.thorax_position() - p0)[:2]))


def test_short_mode_takes_off_within_ms_and_flight_escapes(rig):
    sim, _, mgr, _ = rig
    info, _ = _jump(sim, mgr, mode="short", flight_assist=True)
    assert info["mode"] == "short" and info["takeoff_after_s"] < 0.02
    assert info["landed_upright"] and info["distance_mm"] > 10.0
    assert not np.any(sim.data.xfrc_applied[sim.thorax_body_id])
    long, _ = _jump(sim, mgr)
    assert long["mode"] == "long" and long["takeoff_after_s"] >= 0.1
    with pytest.raises(ValueError):
        Jump(mode="medium")


def test_flight_force_removed_on_reset(rig):
    sim, _, mgr, _ = rig
    sim.reset()
    sim.step(2000)
    mgr.trigger(Jump(mode="short", flight_assist=True, escape_dir=(0.0, 1.0)))
    sim.step(400)  # airborne, force on
    assert mgr.phase() == "flight" and np.any(sim.data.xfrc_applied[sim.thorax_body_id])
    assert sim.thorax_linvel()[1] > 20.0  # heading for +y
    sim.reset()
    assert not mgr.busy and not np.any(sim.data.xfrc_applied[sim.thorax_body_id])


# ---------------------------------------------------------------- brain triggers


def test_triggers_short_mode_and_escape_direction(rig):
    from fly_simulator.actions.brain_triggers import BrainActionTriggers, TriggerParams

    sim, _, mgr, _ = rig
    listeners = list(mgr.listeners)
    trig = BrainActionTriggers(mgr, TriggerParams(jump_short_hz=100.0, jump_flight=True))
    try:
        sim.reset()
        sim.step(500)
        p = sim.thorax_position()
        trig.threat_fn = lambda: p + np.array([-5.0, 0.0, 10.0])  # behind, above
        j = trig.make_jump(80.0)
        assert j.p.mode == "long" and j.p.flight_assist
        assert j.p.escape_dir[0] == pytest.approx(1.0)  # away from the threat
        assert trig.make_jump(150.0).p.mode == "short"
        trig.threat_fn = lambda: p + np.array([0.0, 0.0, 10.0])  # straight overhead
        assert trig.make_jump(150.0).p.escape_dir is None  # -> heading
        st = SimpleNamespace(brain_time=0.1, window_s=0.02, probes={},
                             descending={"escape": 150.0, "groom": 0.0})
        assert "short mode" in trig.on_state(st)[0]
        assert trig.jump_modes == ["short"]
        # defaults: behaviour unchanged (long mode, no flight)
        j0 = BrainActionTriggers(mgr).make_jump(500.0)
        assert j0.p.mode == "long" and not j0.p.flight_assist
    finally:
        mgr.listeners[:] = listeners
        sim.reset()


# ---------------------------------------------------------------- brain link pacing


def _brain_state(t, window):
    from fly_simulator.brain.schema import BrainState

    z = np.zeros(1, np.float32)
    return BrainState(brain_time=t, wall_time=0.0, realtime_factor=1.0, window_s=window,
                      rate_by_nt=z, active_frac_by_nt=z, rate_by_region=z, descending={},
                      raster_idx=np.zeros(0, np.int32), raster_t=np.zeros(0),
                      total_spikes=0, recent_stimuli=[], seq=int(round(t / window)),
                      sim_time=t)


class _FakeBrain:
    """Publishes a state (window end = t) a few polls after the clock reaches t."""

    def __init__(self, window, delay_polls=3):
        self.window, self.delay, self.clock_t = window, delay_polls, 0.0
        self.next_end, self.pending, self.n_poll = window, [], 0

    def clock(self, t):
        self.clock_t = t

    def send(self, ev):
        pass

    def is_alive(self):
        return True

    def poll(self, name=None):
        self.n_poll += 1
        while self.next_end <= self.clock_t + 1e-9:
            self.pending.append([self.next_end, self.delay])
            self.next_end += self.window
        out = []
        for p in self.pending:
            p[1] -= 1
        while self.pending and self.pending[0][1] <= 0:
            t = self.pending.pop(0)[0]
            out.append(_brain_state(t, self.window))
        return out


def _link(sync, loom_only=True):
    from fly_simulator.brain_link import BrainLink, BrainLinkConfig

    link = BrainLink(BrainLinkConfig(enabled=True, window_s=0.02, sync_wait_s=sync,
                                     sync_loom_only=loom_only), start=False, headless=True,
                     say=lambda m: None)
    link.brain = _FakeBrain(0.02)
    rt = [0.0]
    link._run_time = lambda sim_time=None: rt[0] if sim_time is None else sim_time
    return link, rt


def test_brain_link_sync_wait_cuts_the_chunk_latency():
    from fly_simulator.brain.schema import StimulusEvent

    for sync, loom_only, expect_fresh in ((0.0, False, False), (0.05, False, True),
                                          (0.05, True, False)):
        link, rt = _link(sync, loom_only=loom_only)
        rt[0] = 0.1
        link.update()
        if expect_fresh:
            assert link.latest is not None and link.latest.sim_time == pytest.approx(0.1)
            assert link.n_sync_waits == 1
        else:  # without waiting the state for (0.08, 0.1] is not there yet
            assert link.latest is None or link.latest.sim_time < 0.1 - 1e-9
    # loom-only: a visual loom event switches the wait on for its duration
    link, rt = _link(0.05, loom_only=True)
    link.send(StimulusEvent("loom", side="left", intensity=1.0, duration_s=0.08, sim_time=0.05))
    rt[0] = 0.1
    link.update()
    assert link.latest.sim_time == pytest.approx(0.1) and link.n_sync_waits == 1
    rt[0] = 0.2  # loom over: no waiting
    link.update()
    assert link.n_sync_waits == 1
