"""Looming habituation: short-term synaptic depression in the brain engine
(perpetualfly/brain/habituation.py, docs/HABITUATION.md). Synthetic brain only."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("numba")

from perpetualfly.app import build_arg_parser, config_from_args  # noqa: E402
from perpetualfly.brain.engine import LIFEngine, csr_from_edges, random_network  # noqa: E402
from perpetualfly.brain.habituation import HabituationConfig, LoomHabituation  # noqa: E402
from perpetualfly.brain.process import _Model  # noqa: E402
from perpetualfly.brain.schema import StimulusEvent  # noqa: E402

SYN = {"n": 60, "p_conn": 0.1, "seed": 2}


def _run(dep=None, steps=6000, seed=3):
    ip, ix, w, _ = random_network(200, 0.08, 1)
    e = LIFEngine(ip, ix, w, seed=seed)
    e.set_poisson(np.arange(10), 150.0)
    if dep is not None:
        e.set_depression(*dep)
    s, i = e.run(steps)
    return s, i, e


def test_off_is_bit_identical():
    s0, i0, e0 = _run()
    s1, i1, e1 = _run((None,))  # explicitly off
    s2, i2, e2 = _run((np.arange(10), None, 0.0, 5.0))  # on, but U = 0: x stays 1
    for s, i, e in ((s1, i1, e1), (s2, i2, e2)):
        assert np.array_equal(s0, s) and np.array_equal(i0, i)
        assert np.array_equal(e0.v, e.v) and np.array_equal(e0.g, e.g)
    s3, i3, _ = _run((np.arange(10), None, 0.3, 5.0))
    assert len(s3) < len(s0)  # depression changes the dynamics


def test_model_off_is_bit_identical_and_readout_absent():
    def spikes(hab):
        m = _Model({"synthetic": SYN, "habituation": hab})
        m.engine.set_poisson(np.arange(4), 200.0)
        s, i = m.engine.run(3000)
        return m, s, i

    m0, s0, i0 = spikes(None)
    m1, s1, i1 = spikes({"enabled": False, "pre_types": ["DNa02"]})
    assert m0.habituation is None and m1.engine.std_pre is None
    assert np.array_equal(s0, s1) and np.array_equal(i0, i1)
    st = m0.summarize(s0, i0, 0.3, m0.engine.t, 1.0, [])
    assert st.habituation == {}


def _chain():
    """0 -> 1 and 0 -> 2 (10 synapses each); only neuron 0 is driven."""
    ip, ix, w = csr_from_edges(np.array([0, 0]), np.array([1, 2]), np.array([10.0, 10.0]), 3)
    return LIFEngine(ip, ix, w, seed=0)


def test_depression_follows_depletion_model_and_recovers():
    u, tau = 0.2, 0.5
    e = _chain()
    e.set_depression([0], None, u, tau)
    e.set_poisson([0], 400.0)
    s, i = e.run(2000)  # 0.2 s
    n0 = int((i == 0).sum())
    assert n0 > 20
    e.set_poisson([], [])
    e.run(50)  # deliver the last spikes (depletion happens at delivery, after t_dly)
    x = float(e.efficacy([0])[0])
    assert 0.0 < x < 0.5  # depressed well below 1 at 400 Hz
    e.run(int(round(tau * 1e4)))  # one recovery time constant, no spikes
    x2 = float(e.efficacy([0])[0])
    assert x2 == pytest.approx(1 - (1 - x) * math.exp(-1.0), abs=1e-9)
    e.scale_depression(1.0)
    assert float(e.efficacy([0])[0]) == pytest.approx(1.0)


def test_post_mask_only_depresses_selected_targets():
    ref, dep = _chain(), _chain()
    dep.set_depression([0], [1], 0.5, 10.0)
    for e in (ref, dep):
        e.set_poisson([0], 300.0)
    gmax = {k: [] for k in ("ref", "dep")}
    for _ in range(30):
        for k, e in (("ref", ref), ("dep", dep)):
            e.run(50)
            gmax[k].append((e.g[1], e.g[2]))
    r, d = np.array(gmax["ref"]), np.array(gmax["dep"])
    assert np.array_equal(r[:, 1], d[:, 1])  # target 2 untouched (bit-identical)
    assert d[:, 0].max() < r[:, 0].max()     # target 1 depressed


def test_loom_habituation_readout_dishabituation_and_reset():
    m = _Model({"synthetic": SYN, "habituation": {"enabled": True, "pre_types": ["DNa02"],
                                                  "post_types": [], "u": 0.3,
                                                  "tau_rec_s": 30.0,
                                                  "dishabituate_frac": 0.8}})
    h = m.habituation
    assert isinstance(h, LoomHabituation) and len(h.pre) == 2 and h.post is None
    m.engine.set_poisson(h.pre, 300.0)
    s, i = m.engine.run(2000)
    ro = m.summarize(s, i, 0.2, m.engine.t, 1.0, []).habituation
    assert ro["enabled"] and ro["label"] == "habituation (model)"
    assert ro["efficacy"] < 0.3 and set(ro["by_type"]) == {"DNa02"}
    before = ro["efficacy"]
    assert not h.on_stimulus(StimulusEvent("manual"))
    assert h.on_stimulus(StimulusEvent("whip_hit"))
    after = h.readout()["efficacy"]
    assert after == pytest.approx(before + 0.8 * (1 - before), abs=1e-6)
    h.on_reset()  # persist_on_reset: kept
    assert h.readout()["efficacy"] == pytest.approx(after, abs=1e-6)
    # reconfigure U only: efficacies kept; disable: engine back to static synapses
    m.configure_habituation({**h.cfg.to_dict(), "u": 0.1})
    assert h.readout()["efficacy"] == pytest.approx(after, abs=1e-6)
    m.configure_habituation(None)
    assert m.engine.std_pre is None and not h.readout()["enabled"]


def test_config_from_dict_and_cli_flags():
    c = HabituationConfig.from_dict({"pre_types": ["LC4"], "post_types": "DNp01", "x": 1})
    assert c.pre_types == ("LC4",) and c.post_types == ("DNp01",) and not c.enabled
    parse = lambda *a: config_from_args(build_arg_parser().parse_args(list(a))).brain  # noqa: E731
    b = parse()
    assert not b.habituation and not b.record
    b = parse("--habituation")
    assert b.enabled and b.habituation and not b.record
    b = parse("--brain-record")
    assert b.enabled and b.record
    from perpetualfly.brain_link import BrainLink

    link = BrainLink(b, headless=True, start=False)
    assert link.brain_cfg.habituation is None
    b.habituation, b.habituation_config = True, {"u": 0.01}
    link = BrainLink(b, headless=True, start=False)
    assert link.brain_cfg.habituation == {"u": 0.01, "enabled": True}


def test_config_dict_without_enabled_means_on():
    m = _Model({"synthetic": SYN, "habituation": {"pre_types": ["DNa02"]}})
    assert m.engine.std_pre is not None and m.habituation.readout()["enabled"]
