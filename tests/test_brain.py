"""Brain engine tests.

Synthetic-network tests always run (need numba). Tests marked ``needs_data`` use the
FlyWire files from scripts/fetch_brain_data.py and are skipped without them; the
Brian2 cross-check is skipped when brian2 is not installed.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

pytest.importorskip("numba")

from fly_simulator.brain.data import data_available  # noqa: E402
from fly_simulator.brain.engine import LIFEngine, LIFParams, csr_from_edges, random_network  # noqa: E402
from fly_simulator.brain.mapping import (  # noqa: E402
    StimulusMapper, descending_indices, descending_to_drive)
from fly_simulator.brain.process import BrainConfig, BrainProcess, _Model, _synthetic_table  # noqa: E402
from fly_simulator.brain.schema import (  # noqa: E402
    DESCENDING_GROUPS, NEUROTRANSMITTERS, BrainLayout, BrainState, StimulusEvent)

needs_data = pytest.mark.skipif(not data_available(), reason="data/brain not downloaded")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # a zombie still answers kill(0); ask ps for the state
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(out.stdout.strip()) and not out.stdout.strip().startswith("Z")


def _dense_reference(indptr, indices, weights, n_steps, v_init, p: LIFParams):
    """Straightforward O(N) per-step reimplementation (Brian2 semantics) in numpy."""
    n = len(indptr) - 1
    W = np.zeros((n, n))
    for j in range(n):
        for k in range(indptr[j], indptr[j + 1]):
            W[j, indices[k]] += weights[k] * p.w_syn
    e11, e12, e22 = p.propagator()
    v = v_init.astype(float).copy()
    g = np.zeros(n)
    blocked = np.zeros(n, dtype=int)
    hist: list[np.ndarray] = []
    spikes = []
    for t in range(n_steps):
        free = t >= blocked
        dv = v - p.v_0
        v = np.where(free, p.v_0 + e11 * dv + e12 * g, v)
        g = np.where(free, e22 * g, g)
        spk = free & (v > p.v_th)
        blocked[spk] = t + max(p.refractory_steps, 1)
        hist.append(spk)
        if t >= p.delay_steps:
            inp = hist[t - p.delay_steps].astype(float) @ W
            g = g + np.where(t < blocked, 0.0, inp)
        v[spk] = p.v_rst
        g[spk] = 0.0
        spikes += [(t, i) for i in np.nonzero(spk)[0]]
    return spikes


# ------------------------------------------------------------------------ engine
def test_propagator_is_exact_solution():
    from scipy.linalg import expm

    p = LIFParams()
    A = np.array([[-1 / p.t_mbr, 1 / p.t_mbr], [0.0, -1 / p.tau]])
    M = expm(A * p.dt)
    e11, e12, e22 = p.propagator()
    np.testing.assert_allclose([M[0, 0], M[0, 1], M[1, 1]], [e11, e12, e22], rtol=1e-12)


def test_engine_matches_dense_reference_spike_for_spike():
    rng = np.random.default_rng(3)
    n = 40
    pre, post, w = [0, 0, 0, 0], [0, 1, 2, 3], [1000, 60, 45, 30]  # 0 re-excites itself
    for i in range(1, n):
        for j in range(n):
            if i != j and rng.random() < 0.2:
                pre.append(i)
                post.append(j)
                w.append(int(rng.integers(-40, 90)))
    ip, ind, ww = csr_from_edges(np.array(pre), np.array(post), np.array(w), n)
    p = LIFParams(t_rfc=1.0)
    v0 = np.full(n, p.v_0)
    v0[[0, 5, 9]] = 0.0  # kick three neurons over threshold at t=0
    ref = _dense_reference(ip, ind, ww, 1500, v0, p)
    eng = LIFEngine(ip, ind, ww, params=p)
    eng.v[:] = v0
    for i in (0, 5, 9):  # mark as active
        eng._active[eng.n_active] = i
        eng._is_active[i] = 1
        eng.n_active += 1
    steps, idx = [], []
    for _ in range(15):  # chunked
        s, i = eng.run(100)
        steps.append(s)
        idx.append(i)
    got = sorted(zip(np.concatenate(steps).tolist(), np.concatenate(idx).tolist()))
    assert len(ref) > 50
    assert got == sorted((int(t), int(i)) for t, i in ref)


def test_refractory_neuron_ignores_synaptic_input():
    ip, ind, w = csr_from_edges(np.array([0]), np.array([1]), np.array([400]), 2)
    eng = LIFEngine(ip, ind, w)
    eng.v[:] = 0.0  # both spike at step 0; 1 is refractory when the input arrives (step 18)
    eng._active[:2] = [0, 1]
    eng._is_active[:] = 1
    eng.n_active = 2
    s, i = eng.run(100)
    assert sorted(i.tolist()) == [0, 1]  # input during refractoriness is discarded
    assert eng.g[1] == 0.0


def test_poisson_drive_rates_and_chunking_invariance():
    indptr, indices, w, sign = random_network(50, 0.1, seed=1)
    a = LIFEngine(indptr, indices, w, seed=7)
    b = LIFEngine(indptr, indices, w, seed=7)
    a.set_poisson(np.arange(5), 150.0)
    b.set_poisson(np.arange(5), 150.0)
    sa, ia = a.run(10000)
    parts = [b.run(n) for n in (1, 999, 3000, 6000)]
    sb = np.concatenate([p[0] for p in parts])
    ib = np.concatenate([p[1] for p in parts])
    np.testing.assert_array_equal(sa, sb)
    np.testing.assert_array_equal(ia, ib)
    rates = a.counts[:5]  # 1 s
    assert np.all(np.abs(rates - 150) < 60), rates
    assert a.counts[5:].sum() > 0  # activity propagates into the network
    # turning the drive off returns the network to rest (and the active set empties)
    a.set_poisson(np.zeros(0, int), 0.0)
    a.run(5000)
    s, _ = a.run(2000)
    assert len(s) == 0 and a.n_active == 0


# ------------------------------------------------------------------------ mapping
def _synthetic_model(**kw):
    return _Model({"synthetic": {"n": 50, "p_conn": 0.1, "seed": 0}, **kw})


def test_stimulus_mapping_sides_expiry_and_reset():
    table, _ = _synthetic_table(50)
    m = StimulusMapper(table)
    labels = m.add(StimulusEvent("whip_hit", "left", 0.5, 0.1), now=0.0)
    assert labels and all("left" in lab for lab in labels)
    idx, rates = m.drive()
    assert len(idx) and np.all(table.col("side")[idx] == "left")
    assert np.allclose(rates, 100.0)  # 0.5 * 200 Hz
    assert m.next_change() == pytest.approx(0.1)
    m.expire(0.2)
    assert len(m.drive()[0]) == 0
    m.add(StimulusEvent("whip_hit", "front", 1.0, 1.0), now=0.0)
    idx, _ = m.drive()
    assert set(table.col("side")[idx]) == {"left", "right"}
    m.add(StimulusEvent("reset"), now=0.1)
    assert len(m.drive()[0]) == 0
    with pytest.raises(KeyError):
        m.add(StimulusEvent("manual", details={"set": "nope"}), now=0.0)
    assert m.add(StimulusEvent("ground_contact", "left"), now=0.0) == []  # off by default


def test_descending_groups_and_drive_mapping():
    table, _ = _synthetic_table(50)
    dn = descending_indices(table)
    assert set(dn) == set(DESCENDING_GROUPS)
    assert all(len(v) > 0 for v in dn.values())
    quiet = {g: 0.0 for g in DESCENDING_GROUPS}
    np.testing.assert_allclose(descending_to_drive(quiet), [1.0, 1.0])
    left = descending_to_drive({**quiet, "turn_L": 60.0})
    assert left[0] < 1.0 < left[1]  # smaller left amplitude -> turns left
    right = descending_to_drive({**quiet, "turn_R": 60.0})
    assert right[0] > 1.0 > right[1]
    back = descending_to_drive({**quiet, "backward_L": 80.0, "backward_R": 80.0})
    assert np.all(back < 0)
    fwd = descending_to_drive({**quiet, "walk_L": 50.0, "walk_R": 50.0})
    assert np.all(fwd > 1.0) and np.all(fwd <= 1.5)
    assert np.allclose(descending_to_drive({**quiet, "escape": 300.0}), [1, 1])


def test_summary_rates_by_nt_region_and_raster():
    m = _synthetic_model()
    n = m.table.n
    idx = np.array([0, 0, 1, 7, 7, 7], dtype=np.int32)
    steps = np.arange(len(idx), dtype=np.int64) + 1000
    st = m.summarize(steps, idx, 0.5, 1.0, 1.0, ["x"])
    assert isinstance(st, BrainState) and st.total_spikes == 6
    counts = np.bincount(idx, minlength=n)
    for k in range(len(NEUROTRANSMITTERS)):
        sel = m.table.nt == k
        if sel.any():
            assert st.rate_by_nt[k] == pytest.approx(counts[sel].sum() / (sel.sum() * 0.5))
            assert st.active_frac_by_nt[k] == pytest.approx((counts[sel] > 0).mean())
    for r in range(len(m.table.regions)):
        sel = m.table.region == r
        assert st.rate_by_region[r] == pytest.approx(counts[sel].sum() / (sel.sum() * 0.5))
    assert len(st.raster_idx) == 6  # synthetic: every neuron is a display neuron
    assert np.allclose(st.raster_t, steps * 1e-4)
    assert st.recent_stimuli == ["x"]
    lay = m.layout
    assert isinstance(lay, BrainLayout)
    assert len(lay.display_neuron_ids) == len(lay.display_neuron_xy) == len(lay.display_neuron_label)
    assert lay.region_xy.shape == (len(lay.regions), 2)


# ------------------------------------------------------------------------ process
def test_brain_process_start_publish_stop_no_orphans():
    cfg = BrainConfig(synthetic={"n": 50, "p_conn": 0.1, "seed": 0}, window_s=0.05)
    bp = BrainProcess(cfg)
    try:
        info = bp.wait_ready(60)
        pid = bp.pid
        assert info["n_neurons"] == 50
        lay = bp.layout(10)
        assert isinstance(lay, BrainLayout)
        assert isinstance(bp.subscriber("window").layout(10), BrainLayout)
        bp.send(StimulusEvent("whip_hit", "left", 1.0, 0.2))
        deadline = time.time() + 10
        got = []
        while time.time() < deadline and sum(s.total_spikes for s in got) == 0:
            got += bp.poll("app")
            time.sleep(0.02)
        assert got and sum(s.total_spikes for s in got) > 0
        assert all(len(s.rate_by_nt) == len(NEUROTRANSMITTERS) for s in got)
        assert set(got[-1].descending) == set(DESCENDING_GROUPS)
        assert got[-1].realtime_factor > 0.5
        assert bp.latest("window") is not None  # the second subscriber got states too
    finally:
        bp.stop()
    assert not bp.is_alive()
    assert not _pid_alive(pid)


def test_brain_worker_exits_when_parent_is_killed(tmp_path):
    script = tmp_path / "parent.py"
    script.write_text(textwrap.dedent("""
        import os, sys
        from fly_simulator.brain.process import BrainConfig, BrainProcess
        if __name__ == "__main__":
            bp = BrainProcess(BrainConfig(synthetic={"n": 30, "p_conn": 0.1, "seed": 0}))
            bp.wait_ready(60)
            print(bp.pid, flush=True)
            os._exit(0)   # no stop(), no atexit: simulate a crashed app
    """))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                         timeout=120, env={**os.environ, "PYTHONPATH": root})
    assert out.stdout.strip(), f"parent script printed no pid; stderr:\n{out.stderr[-2000:]}"
    pid = int(out.stdout.strip().splitlines()[-1])
    deadline = time.time() + 10
    while time.time() < deadline and _pid_alive(pid):
        time.sleep(0.1)
    assert not _pid_alive(pid)


# ------------------------------------------------------------------------ real data
@needs_data
def test_real_descending_groups_and_sets():
    from fly_simulator.brain.data import load_neuron_table
    from fly_simulator.brain.mapping import named_sets

    t = load_neuron_table()
    dn = descending_indices(t)
    assert len(dn["backward_L"]) == 2 and len(dn["backward_R"]) == 2  # MDN
    assert len(dn["escape"]) == 2  # giant fibers
    assert len(dn["groom"]) == 42  # DNg12_a..e, 21 per side
    for g in ("walk_L", "walk_R"):
        assert len(dn[g]) == 3  # DNg100 (BDN2), DNg97 (oDN1), DNp09 (P9)
    for g in ("turn_L", "turn_R"):
        assert len(dn[g]) == 2  # DNa01, DNa02
    s = named_sets(t)
    assert len(s["sugar"]) == 20
    assert len(s["body_mech"]) > 300


@needs_data
def test_sugar_grns_drive_mn9():
    """Shiu et al. 2024 Fig. 1: sugar GRN activation recruits MN9 (proboscis)."""
    from fly_simulator.brain.data import load_connectome, load_neuron_table
    from fly_simulator.brain.mapping import MN9_IDS, named_sets

    t = load_neuron_table()
    eng = LIFEngine(*load_connectome()[:3], seed=0)
    eng.set_poisson(named_sets(t)["sugar"], 200.0)
    eng.run_seconds(0.5)
    mn9 = t.index_of(MN9_IDS[:1])[0]
    assert eng.counts[mn9] / 0.5 > 40.0
    assert (eng.counts > 0).sum() > 200


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.skipif(importlib.util.find_spec("brian2") is None, reason="brian2 not installed")
def test_engine_matches_brian2_spike_for_spike():
    import brian2 as b2

    from fly_simulator.brain.brian2_ref import build_network

    b2.prefs.codegen.target = "numpy"
    b2.BrianLogger.suppress_name("resolution_conflict")
    rng = np.random.default_rng(0)
    n = 40
    pre, post, w = [0, 0, 0, 0], [0, 1, 2, 3], [1000, 40, 30, 60]
    for i in range(1, n):
        for j in range(1, n):
            if i != j and rng.random() < 0.25:
                pre.append(i)
                post.append(j)
                w.append(int(rng.integers(-40, 80)))
    ip, ind, ww = csr_from_edges(np.array(pre), np.array(post), np.array(w), n)
    p = LIFParams(t_rfc=1.0)  # neuron 0 re-excites itself (delay 1.8 ms > t_rfc)
    net, neu, mon, _ = build_network(ip, ind, ww, [], 0.0, params=p)
    neu.v[0] = 0 * b2.mV
    net.run(200 * b2.ms)
    bt = np.round(np.asarray(mon.t[:] / b2.ms) * 10).astype(int)
    brian = sorted(zip(bt.tolist(), np.asarray(mon.i[:]).tolist()))
    eng = LIFEngine(ip, ind, ww, params=p)
    eng.v[0] = 0.0
    eng._active[0], eng._is_active[0], eng.n_active = 0, 1, 1
    s, i = eng.run(2000)
    assert len(brian) > 100
    assert sorted(zip(s.tolist(), i.tolist())) == brian
