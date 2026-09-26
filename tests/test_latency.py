"""Brain -> body latency: the event-driven fast path (docs/BRAIN.md, "Latency").

Fast tests only (synthetic network / stubs; no connectome, no physics)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("numba")

from fly_simulator.brain.process import BrainConfig, BrainProcess, FastDetector  # noqa: E402
from fly_simulator.brain.schema import FastEvent, StimulusEvent  # noqa: E402

DT = 1e-4  # brain step (s)
W = 200  # 20 ms window in steps


def _det(thr=60.0):
    d = FastDetector({"escape": np.array([10, 11]), "MN9": np.array([3])}, W, DT)
    d.set_thresholds({"escape": thr})
    return d


def _run(d, spikes, step_now):
    """spikes: [(step, neuron)] in step order."""
    s = np.array([x for x, _ in spikes], dtype=np.int64)
    i = np.array([n for _, n in spikes], dtype=np.int32)
    return d.update(s, i, step_now)


# ------------------------------------------------------------------ FastDetector
def test_detector_crosses_on_the_third_spike_in_20ms():
    d = _det()  # 2 GF neurons, 20 ms: 60 Hz -> needs 3 spikes (75 Hz)
    assert _run(d, [(100, 10), (150, 11), (5, 0)][:2], 160) == []
    out = _run(d, [(170, 10), (180, 3)], 200)
    assert len(out) == 1
    g, step, rate, thr = out[0]
    assert (g, step, thr) == ("escape", 170, 60.0) and rate == pytest.approx(75.0)
    # still above: no second event (edge-triggered) ...
    assert _run(d, [(190, 11)], 210) == []
    # ... re-armed once the trailing rate is back below, then crosses again
    assert _run(d, [], 600) == []
    out = _run(d, [(700, 10), (710, 10), (720, 11)], 730)
    assert [o[1] for o in out] == [720]


def test_detector_catches_a_burst_split_by_tiled_windows():
    """3 spikes straddling a window boundary: no tiled 20 ms window exceeds 60 Hz,
    the trailing window does (same rate definition)."""
    d = _det()
    out = _run(d, [(190, 10), (195, 11), (205, 10)], 400)
    assert [o[1] for o in out] == [205]


def test_detector_never_later_than_the_tiled_window():
    rng = np.random.default_rng(0)
    for trial in range(30):
        d = _det()
        n = rng.integers(2, 12)
        steps = np.sort(rng.integers(0, 2000, n))
        neur = rng.choice([10, 11], n)
        # tiled windows [k W, (k + 1) W): first one above 60 Hz
        counts = np.bincount(steps // W, minlength=10)
        above = np.nonzero(counts / (2 * W * DT) > 60.0)[0]
        crossings = []
        for k in range(20):  # feed in 10 ms engine runs
            m = (steps >= k * 100) & (steps < (k + 1) * 100)
            crossings += _run(d, list(zip(steps[m], neur[m])), (k + 1) * 100)
        if len(above):
            assert crossings and crossings[0][1] < (above[0] + 1) * W


def test_detector_thresholds_unknown_groups_and_reset():
    d = FastDetector({"escape": np.array([1, 2]), "empty": np.array([], dtype=np.int64)}, W, DT)
    assert not d.enabled
    d.set_thresholds({"escape": 30.0, "empty": 1.0, "nope": 1.0})
    assert d.enabled and set(d.thr) == {"escape"} and d.unknown == ["empty", "nope"]
    # 30 Hz with 2 neurons in 20 ms: 2 spikes (50 Hz) suffice
    assert [o[1] for o in _run(d, [(10, 1), (20, 2)], 50)] == [20]
    d.set_thresholds({"escape": 20.0})  # live change keeps the trailing spikes
    assert d.rate("escape") == pytest.approx(50.0) and not d.armed["escape"]
    d.reset()
    assert d.rate("escape") == 0.0 and d.armed["escape"]
    d.set_thresholds(None)
    assert not d.enabled and _run(d, [(10, 1), (20, 2), (30, 1)], 50) == []


# ------------------------------------------------------------------ worker process
SYN = {"n": 50, "p_conn": 0.1, "seed": 0}


def _wait(bp, pred, timeout=20.0):
    got = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        got += bp.poll_fast()
        if pred(got):
            return got
        time.sleep(0.01)
    return got


def test_worker_publishes_triggers_and_progress_marks():
    cfg = BrainConfig(synthetic=SYN, window_s=0.02, pace="sim", subscribers=("app",),
                      fast_triggers={"escape": 60.0})
    with BrainProcess(cfg) as bp:
        bp.wait_ready(60)
        bp.clock(0.05)
        got = _wait(bp, lambda g: any(e.kind == "progress" and e.sim_time >= 0.05 - 1e-6
                                      for e in g))
        assert any(e.kind == "progress" for e in got)
        assert not any(e.kind == "trigger" for e in got)  # quiet GF
        # drive the synthetic giant fiber (DNp01) hard, run the brain to 0.3 s
        bp.send(StimulusEvent("opto", "none", 1.0, 0.2, 0.06,
                              details={"target": "DNp01", "rate_hz": 400.0}))
        for k in range(1, 26):
            bp.clock(0.05 + 0.01 * k)
        got = _wait(bp, lambda g: any(e.kind == "progress" and e.sim_time >= 0.3 - 1e-6
                                      for e in g))
        trig = [e for e in got if e.kind == "trigger"]
        assert trig and isinstance(trig[0], FastEvent)
        e = trig[0]
        assert e.group == "escape" and e.rate_hz > 60.0 and e.threshold_hz == 60.0
        assert 0.06 <= e.sim_time <= 0.3 and e.window_s == pytest.approx(0.02)
        # the published windows agree: the GF is above threshold after the crossing
        sts = bp.poll("app")
        assert any(st.descending["escape"] > 60.0 for st in sts)
        # thresholds can be switched off at run time
        bp.set_fast_triggers(None)
        bp.clock(0.4)
        time.sleep(0.5)
        assert not [e for e in bp.poll_fast() if e.sim_time and e.sim_time > 0.31]


def test_worker_without_fast_triggers_publishes_nothing_fast():
    cfg = BrainConfig(synthetic=SYN, window_s=0.02, pace="sim", subscribers=("app",))
    with BrainProcess(cfg) as bp:
        bp.wait_ready(60)
        bp.clock(0.1)
        t0 = time.time()
        while time.time() - t0 < 10 and not any((st.sim_time or 0) >= 0.1 - 1e-6
                                                for st in bp.poll("app")):
            time.sleep(0.01)
        assert bp.poll_fast() == []


# ------------------------------------------------------------------ body side
class _Mgr:
    """Minimal ActionManager stand-in (no physics)."""

    def __init__(self):
        import mujoco as mj

        self.sim = SimpleNamespace(model=mj.MjModel.from_xml_string("<mujoco/>"),
                                   fly_name="fly", time=1.0, heading=lambda: 0.0,
                                   thorax_position=lambda: np.zeros(3))
        self.listeners = []
        self.action = self._pending = None
        self.triggered = []

    def trigger(self, action, replace=False, source=""):
        self.triggered.append((action, source))
        return True


def test_on_fast_fires_the_jump_once_and_the_window_state_respects_refractory():
    from fly_simulator.actions.brain_triggers import BrainActionTriggers, TriggerParams

    mgr = _Mgr()
    trig = BrainActionTriggers(mgr, TriggerParams(jump_short_hz=60.0))
    below = FastEvent("trigger", 0.5, 0.5, group="escape", rate_hz=50.0, threshold_hz=40.0)
    assert trig.on_fast(below, 0.5) == []  # body threshold (60 Hz) decides
    fe = FastEvent("trigger", 0.51, 0.51, group="escape", rate_hz=75.0, threshold_hz=60.0)
    msgs = trig.on_fast(fe, 0.52)
    assert msgs and "JUMP (short mode)" in msgs[0] and "fast" in msgs[0]
    assert trig.fired == [(0.52, "jump", 75.0)] and len(mgr.triggered) == 1
    st = SimpleNamespace(brain_time=0.52, window_s=0.02, probes={},
                         descending={"escape": 150.0, "groom": 0.0})
    assert trig.on_state(st, 0.53) == []  # the late window: refractory, no 2nd jump
    other = FastEvent("trigger", 0.6, 0.6, group="walk_L", rate_hz=500.0)
    assert trig.on_fast(other, 0.6) == []


# ------------------------------------------------------------------ BrainLink wiring
def test_brainlink_fast_path_config_and_off_switch():
    from fly_simulator.brain_link import BrainLink, BrainLinkConfig

    off = BrainLink(BrainLinkConfig(enabled=True, synthetic=SYN, actions=True, fast_path=False),
                    start=False)
    assert not off.fast and off.brain_cfg.fast_triggers is None
    on = BrainLink(BrainLinkConfig(enabled=True, synthetic=SYN, actions=True, fast_path=True,
                                   jump_escape_hz=45.0), start=False)
    assert on.fast and on.brain_cfg.fast_triggers == {"escape": 45.0, "MN9": 30.0}
    assert BrainLinkConfig().fast_path  # default on (measured, docs/BRAIN.md)
    noact = BrainLink(BrainLinkConfig(enabled=True, synthetic=SYN, fast_path=True), start=False)
    assert not noact.fast  # nothing to trigger without --brain-actions


def test_brainlink_fast_path_end_to_end_synthetic():
    """BrainLink + synthetic worker: a fast trigger reaches BrainActionTriggers via
    update(), with the sync wait covering the brain's progress to the clock."""
    from fly_simulator.brain_link import BrainLink, BrainLinkConfig

    link = BrainLink(BrainLinkConfig(enabled=True, synthetic=SYN, actions=True,
                                     fast_path=True, window_s=0.02, sync_wait_s=0.5),
                     headless=True, say=lambda m: None)
    got = []
    from fly_simulator.actions.brain_triggers import TriggerParams

    link.triggers = SimpleNamespace(p=TriggerParams(),
                                    fired=[], on_fast=lambda fe, rt: got.append((fe, rt)),
                                    on_state=lambda st, rt: [])
    clock = {"t": 0.0}
    link._run_time = lambda sim_time=None: clock["t"]
    try:
        link.wait_ready(60)
        link.send(StimulusEvent("loom", "left", 1.0, 1.0, 0.0))  # enables the sync wait
        link.send(StimulusEvent("opto", "none", 1.0, 0.3, 0.02,
                                details={"target": "DNp01", "rate_hz": 400.0}))
        for k in range(1, 31):
            clock["t"] = 0.01 * k
            link.update()
            if got:
                break
        assert got, "no fast trigger reached the body"
        fe, rt = got[0]
        assert fe.group == "escape" and fe.sim_time <= rt + 1e-9
        # the sync wait saw the brain reach the clock: the trigger arrives in the
        # same update() that covers its crossing time (within one 10 ms step)
        assert rt - fe.sim_time <= 0.01 + 1e-6
        assert link.n_sync_waits >= 1
    finally:
        link.close()
