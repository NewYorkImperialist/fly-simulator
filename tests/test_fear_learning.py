"""Fear learning: dopamine-gated KC -> MBON plasticity (fly_simulator/brain/plasticity.py)
and odour zones (fly_simulator/senses/odor.py). docs/FEAR_LEARNING.md. Synthetic only."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from fly_simulator.brain.data import NeuronTable  # noqa: E402
from fly_simulator.brain.engine import LIFEngine, csr_from_edges  # noqa: E402
from fly_simulator.brain.plasticity import (KCMBONPlasticity, PlasticityConfig,  # noqa: E402
                                           compartments, odor_kc_code)

N_KC = 20
MB11 = (20, 21)
MB14 = (22,)
D101 = (23, 24)
D106 = (25,)
N = 30
CFG = dict(min_kc_syn=5, min_dan_mbon_syn=5, silence=(), odor_frac=0.25,
           d_min_hz=20.0, d_ref_hz=100.0, lr=2.0)


def _net():
    """20 KCs -> MBON11 x2 (12 syn) and MBON14 (2 syn); PPL101 -> KCs + MBON11;
    PPL106 -> KCs + MBON14; a few extra neurons."""
    pre, post, w = [], [], []
    for k in range(N_KC):
        for m in MB11:
            pre.append(k); post.append(m); w.append(12)
        for m in MB14:
            pre.append(k); post.append(m); w.append(2)
    for d in D101:
        for k in range(N_KC):
            pre.append(d); post.append(k); w.append(1)
        for m in MB11:
            pre.append(d); post.append(m); w.append(10)
    for d in D106:
        for k in range(N_KC):
            pre.append(d); post.append(k); w.append(1)
        pre.append(d); post.append(MB14[0]); w.append(10)
    pre.append(22); post.append(26); w.append(30)  # MBON14 -> a downstream neuron
    ct = np.array(["KCab"] * N_KC + ["MBON11"] * 2 + ["MBON14"] + ["PPL101"] * 2
                  + ["PPL106"] + [""] * (N - 26), dtype=object)
    n = N
    cols = {"super_class": np.full(n, "central", dtype=object),
            "cell_class": np.full(n, "", dtype=object),
            "cell_sub_class": np.full(n, "", dtype=object), "cell_type": ct,
            "hemibrain_type": np.full(n, "", dtype=object),
            "side": np.array(["left", "right"] * (n // 2), dtype=object),
            "flow": np.full(n, "intrinsic", dtype=object)}
    pos = np.zeros((n, 3), np.float32)
    table = NeuronTable(root_id=np.arange(n, dtype=np.int64) + 10**17,
                        nt=np.zeros(n, np.int8), sign=np.ones(n, np.int8),
                        region=np.zeros(n, np.int16), regions=["X"], pos_um=pos,
                        soma_um=pos.copy(), cols=cols)
    return table, csr_from_edges(np.array(pre), np.array(post), np.array(w), n)


def _engine(seed=0):
    table, (ip, ix, w) = _net()
    return table, LIFEngine(ip, ix, w, seed=seed)


def _drive(e, pl, idx, rate, seconds, chunk=0.02, learn=True):
    e.set_poisson(np.asarray(idx, np.int64), rate)
    out = []
    for _ in range(int(round(seconds / chunk))):
        s, i = e.run_seconds(chunk)
        out.append((s, i))
        if learn and pl is not None:
            pl.observe(i, chunk)
    e.set_poisson(np.zeros(0, np.int64), 0.0)
    return out


def test_compartments_from_connectome():
    table, e = _engine()
    comp = compartments(table, e, PlasticityConfig(**CFG))
    assert list(comp) == ["PPL101", "PPL106"]
    assert comp["PPL101"]["mbon_types"] == ["MBON11"]
    assert comp["PPL106"]["mbon_types"] == ["MBON14"]
    pl = KCMBONPlasticity(table, e, PlasticityConfig(enabled=True, **CFG))
    assert len(pl.p) == N_KC * 3  # every KC -> MBON entry is plastic
    assert np.all(pl.x == 1.0)


def test_pairing_depresses_only_coactive_kcs_in_the_dopamine_compartment():
    table, e = _engine()
    pl = KCMBONPlasticity(table, e, PlasticityConfig(enabled=True, **CFG))
    A, B = np.arange(0, 5), np.arange(10, 15)
    # KC alone: no dopamine -> nothing learned
    _drive(e, pl, A, 50.0, 0.5)
    assert np.all(pl.x == 1.0)
    # DAN alone (the KCs it excites aside): KCs of B were quiet for seconds
    # pairing A with PPL101 only
    _drive(e, pl, np.r_[A, list(D101)], 50.0 * np.r_[np.ones(5), 4 * np.ones(2)], 1.0)
    effA, effB = pl.efficacy(A), pl.efficacy(B)
    assert effA["PPL101"] < 0.5  # A's KC -> MBON11 synapses depressed
    assert effB["PPL101"] > 0.95  # B's untouched (their KCs weren't active)
    assert effA["PPL106"] == 1.0  # other compartment untouched (no dopamine there)
    # engine weights follow w0 * x
    np.testing.assert_allclose(e.weights[pl.p], (pl.w0 * pl.x).astype(np.float32))
    r = pl.readout()
    assert r["enabled"] and r["n_updates"] > 0 and set(r["efficacy_by_odor"]) == {"A", "B"}


def test_mbon_response_drops_for_paired_kcs():
    def mbon_rate(e, idx):
        c0 = e.counts.copy()
        _drive(e, None, idx, 60.0, 1.0, learn=False)
        e.reset_state()
        return float((e.counts - c0)[list(MB11)].mean())

    table, e = _engine(seed=1)
    pl = KCMBONPlasticity(table, e, PlasticityConfig(enabled=True, **CFG))
    A = np.arange(0, 10)
    before = mbon_rate(e, A)
    _drive(e, pl, np.r_[A, list(D101)], np.r_[60.0 * np.ones(10), 200.0, 200.0], 2.0)
    e.reset_state()
    after = mbon_rate(e, A)
    assert before > 20 and after < 0.5 * before


def test_off_and_no_dopamine_are_bit_identical():
    def spikes(mode):
        table, e = _engine(seed=3)
        pl = None
        if mode == "off":
            pl = KCMBONPlasticity(table, e, PlasticityConfig(enabled=False, **CFG))
        elif mode == "on_quiet":  # on, but the DANs never exceed d_min
            pl = KCMBONPlasticity(table, e, PlasticityConfig(enabled=True, **CFG))
        out = _drive(e, pl, np.arange(8), 80.0, 0.5)
        return np.concatenate([s for s, _ in out]), np.concatenate([i for _, i in out]), e

    s0, i0, e0 = spikes("none")
    for mode in ("off", "on_quiet"):
        s, i, e = spikes(mode)
        assert np.array_equal(s0, s) and np.array_equal(i0, i)
        assert np.array_equal(e0.weights, e.weights)
        assert np.array_equal(e0.v, e.v)


def test_disable_restores_exact_weights_and_reenable_keeps_memory():
    table, e = _engine()
    w_orig = e.weights.copy()
    pl = KCMBONPlasticity(table, e, PlasticityConfig(enabled=True, **CFG))
    _drive(e, pl, np.r_[np.arange(5), list(D101)], np.r_[50.0 * np.ones(5), 200.0, 200.0], 1.0)
    assert not np.array_equal(e.weights, w_orig)
    x = pl.x.copy()
    pl.configure(PlasticityConfig(enabled=False, **CFG))
    assert np.array_equal(e.weights, w_orig)
    pl.configure(PlasticityConfig(enabled=True, **CFG))
    assert np.array_equal(pl.x, x) and not np.array_equal(e.weights, w_orig)
    pl.reset_memory()
    assert np.array_equal(e.weights, w_orig)


def test_runaway_guard_and_forgetting():
    table, e = _engine()
    pl = KCMBONPlasticity(table, e, PlasticityConfig(enabled=True, runaway_sps=1000.0,
                                                    **{**CFG, "lr": 5.0}))
    spikes = np.repeat(np.r_[np.arange(5), list(D101)], 50)  # 350 spikes in 20 ms
    assert not pl.observe(spikes, 0.02)
    assert pl.runaway and np.all(pl.x == 1.0)
    pl.configure(PlasticityConfig(enabled=True, tau_forget_s=1.0, **{**CFG, "lr": 5.0}))
    pl.observe(np.repeat(np.r_[np.arange(5), list(D101)], 3), 0.02)
    low = pl.x.min()
    assert low < 1.0
    for _ in range(100):
        pl.observe(np.zeros(0, np.int64), 0.02)
    assert pl.x.min() > low + 0.5 * (1 - low)


def test_odor_code_is_sparse_and_deterministic():
    table, e = _engine()
    a = odor_kc_code(table, e, ("DM1",), 0.25, 0)  # no PNs -> random fallback
    assert len(a) == 5 and np.array_equal(a, odor_kc_code(table, e, ("DM1",), 0.25, 0))
    assert np.all(a < N_KC)


def test_model_integration_sets_readout_and_lesions():
    from fly_simulator.brain.process import _Model
    from fly_simulator.brain.schema import StimulusEvent

    syn = {"n": 60, "p_conn": 0.1, "seed": 2}
    m0 = _Model({"synthetic": syn})
    m = _Model({"synthetic": syn, "plasticity": {"enabled": False}})
    assert np.array_equal(m0.engine.weights, m.engine.weights)
    assert "odor_A" in m.mapper.sets and "dan_punish" in m.mapper.sets
    m.mapper.add(StimulusEvent("manual", details={"set": "odor_A", "rate_hz": 30}), 0.0)
    st = m.summarize(np.zeros(0, np.int64), np.zeros(0, np.int32), 0.1, 0.1, 1.0, [])
    assert st.learning["enabled"] is False and "efficacy_by_odor" in st.learning
    assert m0.summarize(np.zeros(0, np.int64), np.zeros(0, np.int32), 0.1, 0.1, 1.0,
                        []).learning == {}
    m.set_lesions(["DNa02"])  # playground lesions still work with the add-on
    assert m.engine.silenced is not None


def test_cli_flags():
    from fly_simulator.app import build_arg_parser, config_from_args
    from fly_simulator.brain_link import BrainLink

    parse = lambda *a: config_from_args(build_arg_parser().parse_args(list(a)))  # noqa: E731
    c = parse("--odor-zones")
    assert c.odor.enabled and not c.odor.learning and c.brain.enabled and c.brain.odors
    c = parse("--learning")
    assert c.odor.enabled and c.odor.learning and c.brain.learning
    link = BrainLink(c.brain, headless=True, start=False)
    assert link.plasticity_dict()["enabled"] is True
    assert link.brain_cfg.plasticity["enabled"] is True
    c = parse()
    assert not c.odor.enabled
    assert BrainLink(c.brain, headless=True, start=False).brain_cfg.plasticity is None


class _FakeLink:
    def __init__(self):
        self.sent = []
        self.brain = None
        self.latest = None

    def send(self, ev, source=""):
        self.sent.append(ev)


def test_odor_zones_sense_and_punish():
    from fly_simulator.app import Session
    from fly_simulator.config import AppConfig

    cfg = AppConfig()
    cfg.logging.enabled = False
    cfg.odor.enabled = True
    cfg.odor.learning = True
    cfg.odor.layout = "none"
    s = Session(cfg, log=False, say=lambda m: None)
    try:
        h = s.odor
        assert h is not None and h.zones.zones == []
        link = _FakeLink()
        s.brain = link
        h.update()
        assert link.sent == []  # no zone
        assert "outside" in h.punish()
        h.spawn("A")  # zone around the fly
        h.update()
        assert link.sent and link.sent[-1].details["set"] == "odor_A"
        assert "punishment in odour A" in h.punish()
        assert link.sent[-1].details["set"] == "dan_punish"
        assert h.summary()["punishments"]["A"] == 1
        # visual only: the zone geoms never collide
        g = h.zones.zones[0].gid
        assert s.sim.model.geom_contype[g] == 0 and s.sim.model.geom_conaffinity[g] == 0
        assert "ODOUR in A" in h.hud_line()
    finally:
        s.brain = None
