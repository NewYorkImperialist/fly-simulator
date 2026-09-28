"""Smell without runaway (fly_simulator/brain/smell.py, docs/SMELL.md): engine
threshold adaptation, the antennal-lobe fixes on a synthetic network, odour input,
bit-identity when off; a short real-data check when the FlyWire data are present."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from fly_simulator.brain.data import NeuronTable, data_available  # noqa: E402
from fly_simulator.brain.engine import LIFEngine, csr_from_edges, random_network  # noqa: E402
from fly_simulator.brain.smell import (ODORS, SMELL_FIXES, SmellFix, SmellFixConfig,  # noqa: E402
                                       corrected_ln_sign, odor_drive, odor_drive_groups,
                                       orn_rate)

needs_data = pytest.mark.skipif(not data_available(), reason="data/brain not downloaded")


# ------------------------------------------------------------------ engine adaptation
def _pair(count=200):
    """Neuron 0 (Poisson-driven) -> neuron 1 with a synapse strong enough that every
    presynaptic spike makes neuron 1 fire."""
    return csr_from_edges(np.array([0]), np.array([1]), np.array([count]), 2)


def test_adaptation_off_is_bit_identical():
    ip, ix, w, _ = random_network(60, 0.15, seed=4)
    runs = []
    for mode in ("never", "set_then_off"):
        e = LIFEngine(ip, ix, w, seed=1)
        if mode == "set_then_off":
            e.set_adaptation(np.arange(10, 40), 3.0, 0.2, 0.1)
            e.set_adaptation(None)
        e.set_poisson(np.arange(8), 150.0)
        s, i = e.run(3000)
        runs.append((s, i, e.v.copy(), e.g.copy()))
    for a, b in zip(runs[0], runs[1]):
        assert np.array_equal(a, b)


def test_threshold_adaptation_lowers_the_follower_rate():
    rates = {}
    for inc in (0.0, 10.0):
        e = LIFEngine(*_pair(), seed=0)
        if inc:
            e.set_adaptation([1], inc, 0.2)
        e.set_poisson(np.array([0]), 100.0)
        e.run_seconds(1.0)
        rates[inc] = e.counts[1]
        if inc:
            assert e.adaptation_mv([1])[0] > 5.0
            e.reset_state()  # reset clears the adaptation state
            assert e.adaptation_mv([1])[0] == 0.0
    assert rates[0.0] > 40 and rates[10.0] < 0.6 * rates[0.0]


def test_pooled_adaptation_raises_every_masked_threshold():
    e = LIFEngine(*csr_from_edges(np.array([0, 0]), np.array([1, 2]), np.array([200, 0]), 3),
                  seed=0)
    e.set_adaptation([1, 2], 0.0, 0.1, global_inc_mv=0.5)
    e.set_poisson(np.array([0]), 100.0)
    e.run_seconds(0.3)
    a = e.adaptation_mv([1, 2])
    assert a[0] > 0.5 and a[0] == pytest.approx(a[1])  # neuron 2 never fired


# ------------------------------------------------------------------ synthetic AL
def _toy_al(seed=0):
    """10 ORNs -> 20 recurrently coupled 'excitatory' LNs (predicted dopamine, the
    model's excitatory default) + 10 PNs. The LN loop sustains itself after the
    ORN input stops, like the model's antennal-lobe runaway."""
    rng = np.random.default_rng(seed)
    n = 40
    orn, ln, pn = np.arange(10), np.arange(10, 30), np.arange(30, 40)
    pre, post, cnt = [], [], []

    def link(a, b, c, p=1.0):
        for i in a:
            for j in b:
                if i != j and rng.random() < p:
                    pre.append(i), post.append(j), cnt.append(c)
    link(orn, pn, 60)
    link(orn, ln, 20, 0.5)
    link(ln, ln, 15, 0.8)
    link(pn, ln, 10, 0.5)
    ip, ix, w = csr_from_edges(np.array(pre), np.array(post), np.array(cnt, dtype=float), n)
    sc = np.full(n, "central", dtype=object)
    cc = np.full(n, "", dtype=object)
    sub = np.full(n, "", dtype=object)
    ct = np.full(n, "", dtype=object)
    sc[orn], cc[orn] = "sensory", "olfactory"
    ct[orn[:5]], ct[orn[5:]] = "ORN_DM1", "ORN_VA2"
    cc[ln], cc[pn], sub[pn] = "ALLN", "ALPN", "uniglomerular"
    ct[pn[:5]], ct[pn[5:]] = "DM1_lPN", "VA2_lPN"
    ct[ln[:4]] = "lLN1_bc"  # a known cholinergic type: stays excitatory with sign_adapt
    nt = np.zeros(n, dtype=np.int8)
    nt[ln] = 3  # predicted dopamine -> excitatory in the model
    side = np.array(["left", "right"] * (n // 2), dtype=object)
    sign = np.ones(n, dtype=np.int8)
    table = NeuronTable(root_id=np.arange(n, dtype=np.int64) + 10**17, nt=nt, sign=sign,
                        region=np.zeros(n, dtype=np.int16), regions=["AL_L"],
                        pos_um=np.zeros((n, 3), np.float32), soma_um=np.zeros((n, 3), np.float32),
                        cols={"super_class": sc, "cell_class": cc, "cell_sub_class": sub,
                              "cell_type": ct, "hemibrain_type": np.full(n, "", dtype=object),
                              "side": side, "flow": np.full(n, "", dtype=object)})
    return table, (ip, ix, w)


def _persist(fix: str, **kw) -> tuple[float, float]:
    table, csr = _toy_al()
    e = LIFEngine(*csr, seed=2)
    SmellFix(table, e, {"name": fix, **kw})
    idx, rates = odor_drive(table, "glom_DM1", 1.0, 1.0, r_max=80.0)
    e.set_poisson(idx, rates)
    c0 = e.counts.copy()
    e.run_seconds(0.3)
    pn_rate = float((e.counts - c0)[30:35].mean() / 0.3)
    e.set_poisson(np.zeros(0, np.int64), 0.0)
    e.run_seconds(0.1)
    c1 = e.counts.copy()
    e.run_seconds(0.2)
    return float((e.counts - c1).sum() / 0.2), pn_rate


def test_toy_al_runs_away_without_a_fix():
    after, pn = _persist("none")
    assert after > 1000.0 and pn > 20.0


@pytest.mark.parametrize("fix", ["sign", "adapt", "eln"])
def test_toy_al_fixes_stop_the_runaway_and_keep_the_odour_response(fix):
    after, pn = _persist(fix)
    assert after == 0.0
    assert pn > 10.0  # the odour's own PNs still respond


def test_sign_correction_respects_known_types_and_restores_exactly():
    table, csr = _toy_al()
    s = corrected_ln_sign(table, respect_known=True)
    assert (s[10:14] == 1).all() and (s[14:30] == -1).all()
    assert (corrected_ln_sign(table, respect_known=False)[10:30] == -1).all()
    e = LIFEngine(*csr, seed=0)
    w0 = e.weights.copy()
    f = SmellFix(table, e, "sign_adapt")
    assert f.n_flipped == 16 and f.n_adapt == 30 and e.ad_mask is not None
    assert not np.array_equal(e.weights, w0)
    f.configure("none")
    assert np.array_equal(e.weights, w0) and e.ad_mask is None
    f.configure({"name": "gaba", "gaba_gain": 3.0})  # no inhibitory LNs in the toy: no-op
    assert np.array_equal(e.weights, w0)
    f.configure({"name": "mbdl1", "hub_types": ["lLN1_bc"]})
    assert list(f.silenced_idx) == [10, 11, 12, 13]
    assert set(f.readout()) >= {"name", "label", "n_flipped", "n_adapt", "n_silenced"}


def test_config_parsing():
    assert SmellFixConfig.from_dict("sign").name == "sign"
    assert SmellFixConfig.from_dict({"name": "mbdl1", "hub_types": "X"}).hub_types == ("X",)
    with pytest.raises(ValueError):
        SmellFixConfig.from_dict("nope")
    assert "sign_adapt" in SMELL_FIXES and "none" in SMELL_FIXES


# ------------------------------------------------------------------ odours
def test_odor_drive_per_antenna():
    table, _ = _toy_al()
    assert orn_rate(0.0) == 0.0 and orn_rate(0.5) == pytest.approx(50.0)
    assert orn_rate(10.0) < 100.0 and orn_rate(1.0) > orn_rate(0.5)
    g = odor_drive_groups(table, "glom_DM1", 1.0, 0.0)
    assert len(g) == 1 and g[0][0] == "odor:glom_DM1:DM1:left"
    assert set(table.col("side")[g[0][1]]) == {"left"}
    idx, rates = odor_drive(table, {"DM1": 1.0, "VA2": 0.5}, 1.0, 1.0)
    assert len(idx) == 10 and rates.max() == pytest.approx(2 * rates.min())
    assert "vinegar" in ODORS


def test_mapper_odor_event():
    from fly_simulator.brain.mapping import StimulusMapper
    from fly_simulator.brain.schema import StimulusEvent

    table, _ = _toy_al()
    m = StimulusMapper(table)
    labels = m.add(StimulusEvent("odor", "none", 1.0, 0.1, 0.0,
                                 details={"odor": "glom_VA2", "left": 0.0, "right": 2.0}), 0.0)
    assert labels == ["odor:glom_VA2:VA2:right"]
    idx, rates = m.drive()
    assert set(table.col("side")[idx]) == {"right"} and rates.min() > 50


def test_model_smell_fix_off_is_bit_identical_and_readout():
    from fly_simulator.brain.process import _Model

    syn = {"n": 60, "p_conn": 0.15, "seed": 3}
    runs = []
    for smell in (None, {"name": "sign_adapt"}):
        m = _Model({"synthetic": syn, "seed": 0, "smell_fix": smell})
        m.engine.set_poisson(np.arange(4), 150.0)
        s, i = m.engine.run(2000)
        runs.append((s, i))
        st = m.summarize(s, i, 0.2, m.engine.t, 1.0, [])
        if smell is None:
            assert st.smell == {}
        else:  # the synthetic table has no AL: nothing flipped, readout present
            assert st.smell["name"] == "sign_adapt" and st.smell["n_flipped"] == 0
            assert "rates" in st.smell and st.smell["runaway"] in (True, False)
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])


# ------------------------------------------------------------------ real data (short)
@needs_data
def test_real_single_glomerulus_no_runaway_with_the_recommended_fix():
    from fly_simulator.brain.data import load_connectome, load_neuron_table

    t = load_neuron_table()
    e = LIFEngine(*load_connectome()[:3], seed=0)
    SmellFix(t, e, "sign_adapt")
    idx, rates = odor_drive(t, "glom_DM1", 1.0, 1.0)  # ~67 Hz ORNs, both antennae
    e.set_poisson(idx, rates)
    c0 = e.counts.copy()
    e.run_seconds(0.3)
    r = (e.counts - c0) / 0.3
    cc, ct = t.col("cell_class"), t.col("cell_type").astype(str)
    own = np.nonzero((cc == "ALPN") & np.char.startswith(ct, "DM1_"))[0]
    assert r[own].mean() > 20.0  # the odour's own PNs respond
    e.set_poisson(np.zeros(0, np.int64), 0.0)
    e.run_seconds(0.1)
    _, after = e.run_seconds(0.1)
    assert len(after) / 0.1 < 1000.0  # back to (near) rest, no runaway


# ------------------------------------------------------------------ app / window
def test_cli_flags_and_config():
    from fly_simulator.app import (ConfigError, build_arg_parser, check_feature_config,
                                   config_from_args)

    p = build_arg_parser()
    cfg = config_from_args(p.parse_args(["--odor-plume", "--plume-control", "glom_DL5"]))
    assert cfg.plume.enabled and cfg.plume.odor == "vinegar"
    assert cfg.plume.control_odor == "glom_DL5"
    assert cfg.brain.enabled and cfg.brain.smell_fix == "sign_adapt"
    check_feature_config(cfg)
    cfg = config_from_args(p.parse_args(["--smell-fix", "eln"]))
    assert cfg.brain.smell_fix == "eln" and not cfg.plume.enabled
    cfg.brain.smell_fix = "bogus"
    with pytest.raises(ConfigError):
        check_feature_config(cfg)
    from fly_simulator.brain_link import BrainLink, BrainLinkConfig

    link = BrainLink(BrainLinkConfig(enabled=True, smell_fix="sign"), start=False)
    assert link.brain_cfg.smell_fix == {"name": "sign"}
    assert BrainLink(BrainLinkConfig(enabled=True), start=False).brain_cfg.smell_fix is None


def test_plume_field_and_antennae():
    import math

    from fly_simulator.senses.plume import OdorPlume, PlumeConfig

    pl = OdorPlume(PlumeConfig(source_x_mm=10.0, source_y_mm=0.0, meander_mm=0.0,
                               control_odor="glom_DL5", control_x_mm=-10.0, control_y_mm=0.0))
    c = pl.conc(10.0, 0.0, 0.0)
    assert c["vinegar"] == pytest.approx(2.0 + 0.0, rel=0.01) and c["glom_DL5"] < 0.5
    (xl, yl), (xr, yr) = pl.antennae(0.0, 0.0, 0.0)  # facing +x: left antenna at +y
    assert yl > 0 > yr and xl == pytest.approx(1.0)
    cl, cr = pl.bilateral(0.0, 0.0, math.pi / 2, 0.0)["vinegar"]  # facing +y: source right
    assert cr > cl
    g = OdorPlume(PlumeConfig(source_x_mm=10.0, contrast_gain=10.0, meander_mm=0.0))
    gl, gr = g.bilateral(0.0, 0.0, math.pi / 2, 0.0)["vinegar"]
    assert (gr - gl) / (gr + gl) > 5 * (cr - cl) / (cr + cl)


def test_window_renders_the_smell_readout():
    from fly_simulator.brain.schema import NEUROTRANSMITTERS, BrainState
    from fly_simulator.brain_viz.mock import mock_scenario
    from fly_simulator.brain_viz.window import render_frame

    L, states = mock_scenario("whip", n_display=200)
    s = states[-1]
    kw = {k: getattr(s, k) for k in s.__dataclass_fields__}
    kw["smell"] = {"name": "sign_adapt", "rates": {"uPN_L": 40.0, "uPN_R": 30.0, "LH": 1.5},
                   "kc_active_frac": 0.01, "runaway": False}
    img = render_frame(L, [*states[:-1], BrainState(**kw)])
    assert img.ndim == 3 and len(NEUROTRANSMITTERS) == 6
