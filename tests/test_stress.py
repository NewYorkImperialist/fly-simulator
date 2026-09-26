"""Octopamine stress / arousal layer (fly_simulator/brain/neuromod.py, fly_simulator/stress.py)."""

import math

import numpy as np
import pytest

pytest.importorskip("numba")

from fly_simulator.brain.engine import LIFEngine, random_network  # noqa: E402
from fly_simulator.brain.neuromod import (NeuromodConfig, OctopamineModel, oa_targets,  # noqa: E402
                                         select_oa_neurons, step_level)
from fly_simulator.brain.process import _Model  # noqa: E402
from fly_simulator.brain.schema import BrainState, StimulusEvent  # noqa: E402
from fly_simulator.stress import (StressConfig, amp_multiplier, freq_multiplier,  # noqa: E402
                                 install_stress, jump_threshold)

SYN = {"n": 60, "p_conn": 0.15, "seed": 3}


def _drive(model, dur_s, idx, rate=300.0, chunk=200):
    eng = model.engine
    eng.set_poisson(np.asarray(idx), rate)
    steps, neurons = [], []
    for _ in range(int(round(dur_s / (chunk * 1e-4)))):
        s, i = eng.run(chunk)
        if model.neuromod is not None:
            model.neuromod.observe(i, chunk * 1e-4)
        steps.append(s)
        neurons.append(i)
    return np.concatenate(steps), np.concatenate(neurons)


def test_level_dynamics_exact():
    # rise: u = 1, tau_rise = 1 s, no decay -> 1 - e^-1 after 1 s
    lv = step_level(0.0, 1.0, 1.0, 1.0, 1e12)
    assert lv == pytest.approx(1 - math.exp(-1), rel=1e-9)
    # chunked integration == one step (u constant)
    l2 = 0.0
    for _ in range(50):
        l2 = step_level(l2, 1.0, 0.02, 1.0, 1e12)
    assert l2 == pytest.approx(lv, rel=1e-9)
    # decay: e^-1 after tau_decay
    assert step_level(0.8, 0.0, 30.0, 1.0, 30.0) == pytest.approx(0.8 * math.exp(-1))
    # bounded, accumulates with repeated drive
    l3 = 0.0
    for _ in range(100):
        l3 = step_level(l3, 5.0, 0.5, 1.0, 30.0)
        assert 0.0 <= l3 < 1.0
    assert l3 > 0.9


def test_oa_selection_and_targets():
    m = _Model({"synthetic": SYN, "seed": 0})
    oa = select_oa_neurons(m.table)  # synthetic: NT fallback, non-sensory
    assert len(oa) and np.all(m.table.nt[oa] == 5)
    e = m.engine
    tg = oa_targets(e.indptr, e.indices, e.weights, oa, m.table.n, 1, e.p.w_syn)
    # brute force
    want = set()
    for j in oa:
        want |= set(e.indices[e.indptr[j]:e.indptr[j + 1]].tolist())
    assert set(tg.tolist()) == want
    tg5 = oa_targets(e.indptr, e.indices, e.weights, oa, m.table.n, 50, e.p.w_syn)
    assert set(tg5.tolist()) <= want


def test_engine_bit_identical_without_modulation():
    """No neuromod / disabled / level 0: exactly the unmodulated spike train."""
    ref = _Model({"synthetic": SYN, "seed": 1})
    s0, i0 = _drive(ref, 0.3, [0, 1, 2, 3])
    for nm in ({"enabled": False}, {"enabled": True, "tau_rise_s": 1e12}):
        m = _Model({"synthetic": SYN, "seed": 1, "neuromod": nm})
        s1, i1 = _drive(m, 0.3, [0, 1, 2, 3])
        assert np.array_equal(s0, s1) and np.array_equal(i0, i1)
        assert np.array_equal(ref.engine.v, m.engine.v)
    # an explicit per-neuron threshold array equal to v_th is also exact
    m = _Model({"synthetic": SYN, "seed": 1})
    m.engine.threshold_array()
    s2, i2 = _drive(m, 0.3, [0, 1, 2, 3])
    assert np.array_equal(s0, s2) and np.array_equal(i0, i2)


def test_threshold_shift_raises_excitability():
    indptr, indices, w, _ = random_network(40, 0.15, seed=2)
    a = LIFEngine(indptr, indices, w, seed=0)
    b = LIFEngine(indptr, indices, w, seed=0)
    b.threshold_array()[:] = b.p.v_th - 2.0
    for e in (a, b):
        e.set_poisson(np.arange(5), 150.0)
        e.run(3000)
    assert b.counts.sum() > a.counts.sum()
    b.clear_threshold()
    assert b.v_th_arr is None


def test_level_driven_by_oa_spikes_and_shift_applied():
    m = _Model({"synthetic": SYN, "seed": 0,
                "neuromod": {"enabled": True, "target_min_syn": 1, "ref_rate_hz": 5.0}})
    nm = m.neuromod
    assert m.engine.v_th_arr is None  # level 0 -> scalar path
    _drive(m, 0.5, nm.oa_idx, rate=400.0)  # drive the OA neurons directly
    assert nm.level > 0.05
    vth = m.engine.v_th_arr
    assert vth is not None
    assert np.allclose(vth[nm.targets], m.engine.p.v_th - nm.cfg.vth_shift_mv * nm.level,
                       atol=2e-4)
    others = np.setdiff1d(np.arange(m.table.n), nm.targets)
    assert np.all(vth[others] == m.engine.p.v_th)
    r = nm.readout()
    assert r["enabled"] and r["oa_rate_hz"] > 0 and r["label"] == "octopamine (model)"
    # decays back, and disabling restores the exact scalar model
    lv = nm.level
    m.engine.set_poisson(np.zeros(0, np.int64), 0.0)
    for _ in range(20):
        m.engine.run(1000)
        nm.observe(np.zeros(0, np.int64), 0.1)
    assert nm.level < lv * math.exp(-2.0 / nm.cfg.tau_decay_s) + 1e-9
    m.configure_neuromod(None)
    assert m.engine.v_th_arr is None


def test_nociceptive_path_off_by_default_and_scales_with_hit_rate():
    base = {"enabled": True, "ref_rate_hz": 1e12}  # ignore OA spikes here
    m = _Model({"synthetic": SYN, "seed": 0, "neuromod": base})
    nm = m.neuromod
    assert len(nm.noci_idx) == 4  # synthetic body_mech afferents 0..3
    _drive(m, 0.05, nm.noci_idx, rate=200.0)
    assert nm.level == 0.0  # connectome-only: afferents alone do nothing
    assert nm.readout()["noci_hz"] > 50  # ... but they are measured
    levels = []
    for rate in (50.0, 200.0):
        m = _Model({"synthetic": SYN, "seed": 0,
                    "neuromod": {**base, "nociceptive_input": True}})
        m.engine.run(1)
        _drive(m, 0.05, m.neuromod.noci_idx, rate=rate)
        levels.append(m.neuromod.level)
    assert 0 < levels[0] < levels[1] < 0.5
    # a 50 ms hit at 200 Hz ~ u = 200/35 over 0.05 s -> ~ 1 - exp(-0.29)
    assert levels[1] == pytest.approx(1 - math.exp(-200 / 35 * 0.05), abs=0.08)
    c = StressConfig(enabled=True).neuromod_config()
    assert c.noci_relay and not c.nociceptive_input
    assert not StressConfig(enabled=True, noci_relay=False).neuromod_config().noci_relay


def test_relay_events():
    m = _Model({"synthetic": SYN, "seed": 0, "neuromod": {"enabled": True}})
    hit = StimulusEvent("whip_hit", "left", 0.5, 0.05, sim_time=1.0)
    assert m.neuromod.relay_events(hit) == []  # off by default
    m.configure_neuromod({"enabled": True, "noci_relay": True})
    evs = m.neuromod.relay_events(hit)
    got = {e.details["set"]: e for e in evs}
    assert set(got) == {"an_walk", "an_arousal"}
    assert got["an_walk"].details["rate_hz"] == pytest.approx(125.0)
    assert got["an_arousal"].details["rate_hz"] == pytest.approx(85.0)
    assert all(e.side == "none" and e.duration_s == pytest.approx(0.25)
               and e.sim_time == 1.0 for e in evs)
    assert m.neuromod.relay_events(StimulusEvent("fall")) == []
    assert m.neuromod.relay_events(StimulusEvent("manual", details={"set": "LC4"})) == []


def test_runaway_guard():
    m = _Model({"synthetic": SYN, "seed": 0,
                "neuromod": {"enabled": True, "runaway_sps": 10.0, "initial_level": 0.5,
                             "target_min_syn": 1}})
    nm = m.neuromod
    fake = np.repeat(nm.oa_idx, 50)  # lots of OA spikes, above the "runaway" rate
    nm.observe(fake, 0.02)
    assert nm.runaway and nm.level < 0.5 and m.engine.v_th_arr is None
    assert nm.readout()["runaway"]


def test_state_carries_neuromod_only_when_configured():
    m = _Model({"synthetic": SYN, "seed": 0})
    st = m.summarize(np.zeros(0, np.int64), np.zeros(0, np.int32), 0.1, 0.1, 1.0, [])
    assert st.neuromod == {}
    m = _Model({"synthetic": SYN, "seed": 0, "neuromod": {"enabled": True}})
    st = m.summarize(np.zeros(0, np.int64), np.zeros(0, np.int32), 0.1, 0.1, 1.0, [])
    assert st.neuromod["enabled"] and st.neuromod["octopamine"] == 0.0


def test_body_mapping_bounded():
    c = StressConfig(enabled=True)
    assert freq_multiplier(0.0, c) == 1.0
    assert freq_multiplier(1.0, c) == pytest.approx(1.5)
    assert freq_multiplier(5.0, c) == pytest.approx(1.5)
    assert amp_multiplier(1.0, c) == pytest.approx(1.2)
    assert amp_multiplier(-1.0, c) == 1.0
    assert jump_threshold(0.0, 60.0, c) == 60.0
    assert jump_threshold(1.0, 60.0, c) == pytest.approx(30.0)
    assert jump_threshold(1.0, 60.0, StressConfig(jump_threshold_drop=0.9)) == 20.0


class _Impl:
    def __init__(self):
        self._base_intrinsic_freqs = np.full(6, 12.0)


class _P:
    jump_escape_hz = 60.0


class _Link:
    def __init__(self):
        self.latest = None
        self.triggers = type("T", (), {"p": _P()})()
        self.brain = type("B", (), {"sent": [], "set_neuromod": lambda s, c: s.sent.append(c)})()
        self.n_updates = 0

    def update(self):
        self.n_updates += 1


def _state(level):
    return BrainState(0.1, 0.0, 1.0, 0.1, np.zeros(6), np.zeros(6), np.zeros(1), {},
                      np.zeros(0, np.int32), np.zeros(0), 0,
                      neuromod={"enabled": True, "octopamine": level, "oa_rate_hz": 3.0})


def test_install_stress_noop_when_disabled():
    ctl = type("C", (), {"impl": _Impl()})()
    link = _Link()
    h = install_stress({"controller": ctl, "link": link}, StressConfig(enabled=False))
    link.latest = _state(0.9)
    link.update()
    h.update()
    assert np.all(ctl.impl._base_intrinsic_freqs == 12.0)
    assert link.triggers.p.jump_escape_hz == 60.0
    assert link.brain.sent == [] and "update" not in vars(link)
    assert h.hud_line() == ""


def test_install_stress_applies_and_restores():
    ctl = type("C", (), {"impl": _Impl()})()
    link = _Link()
    h = install_stress({"controller": ctl, "link": link},
                       StressConfig(enabled=True, neuromod={"tau_decay_s": 40.0}))
    assert link.brain.sent[-1]["enabled"] and link.brain.sent[-1]["tau_decay_s"] == 40.0
    link.latest = _state(0.5)
    link.update()  # hooked: BrainLink.update() then stress update
    assert link.n_updates == 1
    assert np.allclose(ctl.impl._base_intrinsic_freqs, 12.0 * 1.25)
    assert link.triggers.p.jump_escape_hz == pytest.approx(45.0)
    assert "PAIN/AROUSAL 0.50 [octopamine (model)]" in h.hud_line()
    assert h.amp_mult == pytest.approx(1.1)
    h.close()
    assert np.all(ctl.impl._base_intrinsic_freqs == 12.0)
    assert link.triggers.p.jump_escape_hz == 60.0
    assert "update" not in vars(link) and link.brain.sent[-1] is None


def test_amplitude_filter_composes_and_restores():
    prev = lambda hold: np.asarray(hold) * 0.5  # noqa: E731  (e.g. the brain's steering)
    ctl = type("C", (), {"impl": _Impl()})()
    ctl.signal_filter = prev
    link = _Link()
    h = install_stress({"controller": ctl, "link": link}, StressConfig(enabled=True))
    assert np.allclose(ctl.signal_filter(np.array([1.0, 1.0])), [0.5, 0.5])  # level 0
    h.set_level(1.0)
    assert np.allclose(ctl.signal_filter(np.array([1.2, 0.8])), [0.72, 0.48])
    assert np.allclose(ctl.signal_filter(np.array([-4.0, 4.0])), [-1.5, 1.5])  # capped
    h.close()
    assert ctl.signal_filter is prev


def test_neuromod_config_roundtrip():
    c = NeuromodConfig.from_dict({"tau_decay_s": 45, "bogus": 1})
    assert c.tau_decay_s == 45 and NeuromodConfig.from_dict(c.to_dict()) == c
    assert StressConfig(enabled=True).neuromod_config().enabled


def test_window_gauge_renders():
    from fly_simulator.brain_viz.window import BrainRenderer, render_frame

    m = _Model({"synthetic": SYN, "seed": 0, "neuromod": {"enabled": True}})
    states = []
    for k in range(5):
        st = m.summarize(np.zeros(0, np.int64), np.zeros(0, np.int32), 0.1, 0.1 * (k + 1),
                         1.0, [])
        st.neuromod["octopamine"] = 0.2 * k
        states.append(st)
    img = render_frame(m.layout, states, size=(1280, 800), interval=0.1)
    plain = render_frame(m.layout, [m.summarize(np.zeros(0, np.int64), np.zeros(0, np.int32),
                                                0.1, 0.1, 1.0, [])], size=(1280, 800))
    assert img.shape == plain.shape == (800, 1280, 3)
    r = BrainRenderer(m.layout, size=(1280, 800))
    for s in states:
        r.ingest(s, 1000.0)
    assert r._show_nm and r._nm_row is not None and r._tgt_nm == pytest.approx(0.8)
    # without neuromod the row is absent
    r2 = BrainRenderer(m.layout, size=(1280, 800))
    r2.ingest(BrainState(0.1, 0.0, 1.0, 0.1, np.zeros(6), np.zeros(6), np.zeros(1), {},
                         np.zeros(0, np.int32), np.zeros(0), 0), 1000.0)
    assert not r2._show_nm and r2._nm_row is None
