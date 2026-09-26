"""Brain playground (docs/PLAYGROUND.md): virtual optogenetics, virtual lesions,
decision meters, the clickable palette and the app wiring. Synthetic brain only."""

import csv
import json
import time

import numpy as np
import pytest

from fly_simulator.brain.schema import StimulusEvent
from fly_simulator.brain_viz.playground import (PRESETS, canonical_label, decision_meters,
                                               describe_command, parse_lesion_specs,
                                               parse_stim_specs)

numba = pytest.importorskip("numba")

SYN = {"n": 60, "seed": 1}


def _model(**kw):
    from fly_simulator.brain.process import _Model

    return _Model({"synthetic": dict(SYN, **kw)})


def _drive(m, ev):
    labs = m.mapper.add(ev, m.engine.t)
    m.note_stim(ev, labs)
    m.mapper.expire(m.engine.t)
    m.engine.set_poisson(*m.mapper.drive())
    return labs


# ------------------------------------------------------------------ specs
def test_parse_specs():
    s = parse_stim_specs("DNa02_L:120:1.0@3, MDN@6 ; GF:200 ,region:GNG:50:0.5")
    got = [(x.target, x.rate_hz, x.duration_s, x.at_s) for x in s]
    assert got == [("GF", 200.0, 1.0, None), ("region:GNG", 50.0, 0.5, None),
                   ("DNa02_L", 120.0, 1.0, 3.0), ("MDN", 120.0, 1.0, 6.0)]
    assert parse_stim_specs(None) == []
    for bad in ("DNa02:0", ":120", "MDN:120:-1", "MDN@x"):
        with pytest.raises(ValueError):
            parse_stim_specs(bad)
    assert parse_lesion_specs("DNp01, MDN@4") == [("DNp01", None), ("MDN", 4.0)]
    assert canonical_label("DNa02_L") == "DNa02 L" and canonical_label("GF") == "DNp01"
    assert describe_command({"op": "lesion", "target": "MDN", "on": True}) == "LESION MDN"


def test_resolve_target_synthetic():
    from fly_simulator.brain.mapping import resolve_target

    m = _model()
    t = m.table
    ct = t.col("cell_type")
    r = resolve_target(t, "DNa02_L")
    assert r.kind == "cell_type" and r.label == "DNa02 L" and len(r.idx) == 1
    assert ct[r.idx[0]] == "DNa02" and t.col("side")[r.idx[0]] == "left"
    assert set(resolve_target(t, "GF").idx) == set(np.nonzero(ct == "DNp01")[0])
    assert set(resolve_target(t, "BDN2").idx) == set(np.nonzero(ct == "DNg100")[0])
    assert resolve_target(t, "walk_R").kind == "group"
    assert resolve_target(t, "AL_L").kind == "region"  # region name wins over side suffix
    assert len(resolve_target(t, "region:GNG").idx) == int((t.region == 2).sum())
    assert resolve_target(t, "body_mech").kind == "set"
    assert resolve_target(t, "DNg1*").kind == "cell_type"
    assert resolve_target(t, "nonsense").kind == "unknown"
    assert len(resolve_target(t, "nonsense").idx) == 0


# ------------------------------------------------------------------ engine
def _spikes(eng, drive_idx, steps=3000, chunks=3):
    eng.set_poisson(np.asarray(drive_idx), 150.0)
    out = []
    for _ in range(chunks):
        s, i = eng.run(steps)
        out.append((s, i))
    return out


def test_engine_bit_identical_when_unused():
    from fly_simulator.brain.engine import LIFEngine, random_network

    indptr, indices, w, _ = random_network(80, 0.1, 3)
    a = LIFEngine(indptr, indices, w, seed=5)
    b = LIFEngine(indptr, indices, w, seed=5)
    b.set_silenced([3, 4])
    b.set_silenced(None)  # lesion added then removed: back to the exact path
    c = LIFEngine(indptr, indices, w, seed=5)
    c.set_silenced([])
    ra, rb, rc = (_spikes(e, [0, 1, 2]) for e in (a, b, c))
    for (sa, ia), (sb, ib), (sc, ic) in zip(ra, rb, rc):
        assert np.array_equal(sa, sb) and np.array_equal(ia, ib)
        assert np.array_equal(sa, sc) and np.array_equal(ia, ic)
    assert np.array_equal(a.v, b.v) and np.array_equal(a.g, b.g)
    assert sum(len(i) for _, i in ra) > 100


def test_lesion_silences_and_blocks_output():
    from fly_simulator.brain.engine import LIFEngine, random_network

    indptr, indices, w, _ = random_network(80, 0.1, 3)
    eng = LIFEngine(indptr, indices, w, seed=5)
    ref = LIFEngine(indptr, indices, w, seed=5)
    driven = [0, 1, 2]
    eng.set_silenced(driven)  # silence the driven neurons: nothing can spike
    out = _spikes(eng, driven)
    assert sum(len(i) for _, i in out) == 0
    base = _spikes(ref, driven)
    assert sum(len(i) for _, i in base) > 100
    # partial lesion: the silenced neuron never appears, the others still fire
    eng2 = LIFEngine(indptr, indices, w, seed=5)
    eng2.set_silenced([1])
    ids = np.concatenate([i for _, i in _spikes(eng2, driven)])
    assert 1 not in set(ids.tolist()) and {0, 2} <= set(ids.tolist())


def test_lesion_with_neuromod_thresholds():
    """Lesions and the octopamine threshold array compose; clearing either keeps
    the other."""
    from fly_simulator.brain.engine import LIFEngine, random_network

    indptr, indices, w, _ = random_network(60, 0.1, 2)
    eng = LIFEngine(indptr, indices, w, seed=1)
    vth = eng.threshold_array()
    vth[:] = eng.p.v_th - 1.0
    eng.set_silenced([0])
    ids = np.concatenate([i for _, i in _spikes(eng, [0, 1])])
    assert 0 not in ids and 1 in ids
    assert np.all(eng.v_th_arr == eng.p.v_th - 1.0)  # the modulation array untouched
    eng.clear_threshold()
    ids = np.concatenate([i for _, i in _spikes(eng, [0, 1])])
    assert 0 not in ids and 1 in ids


# ------------------------------------------------------------------ worker model
def test_model_opto_and_lesion_readout():
    m = _model()
    # unused: no playground field (and the engine is on the exact path)
    s, i = m.engine.run(100)
    assert m.summarize(s, i, 0.01, m.engine.t, 1.0, []).playground == {}
    assert m.engine.silenced is None
    labs = _drive(m, StimulusEvent("opto", "none", 1.0, 1.0, 0.0,
                                   details={"target": "DNa02_L", "rate_hz": 120.0}))
    assert labs == ["opto:DNa02 L"]
    s, i = m.engine.run(2000)
    st = m.summarize(s, i, 0.2, m.engine.t, 1.0, labs)
    assert st.descending["turn_L"] > 40 and st.descending["turn_R"] < 1  # 120 Hz / 2 DNs
    pg = st.playground
    assert pg["opto"][0]["label"] == "DNa02 L" and pg["opto"][0]["rate_hz"] == 120.0
    # GF: stimulate + lesion -> silent
    m.set_lesions(["GF"])
    _drive(m, StimulusEvent("opto", "none", 1.0, 1.0, m.engine.t,
                            details={"target": "DNp01", "rate_hz": 200.0}))
    s, i = m.engine.run(2000)
    st = m.summarize(s, i, 0.2, m.engine.t, 1.0, [])
    assert st.descending["escape"] == 0.0
    assert st.playground["lesions"] == [{"target": "GF", "label": "DNp01", "n": 2}]
    assert st.playground["n_silenced"] == 2 and len(st.playground["lesion_disp"]) == 2
    # lesions survive a brain reset; unlesioning restores the stimulated GF
    m.engine.reset_state()
    m.set_lesions([])
    s, i = m.engine.run(2000)
    st = m.summarize(s, i, 0.2, m.engine.t, 1.0, [])
    assert st.descending["escape"] > 100
    # unknown targets never crash the worker; they are reported
    labs = _drive(m, StimulusEvent("opto", "none", 1.0, 1.0, m.engine.t,
                                   details={"target": "nope"}))
    m.set_lesions(["nope"])
    assert labs == [] and len(m.pg_errors) == 2
    # stop all stimulation
    m.mapper.add(StimulusEvent("opto_stop"), m.engine.t)
    assert not [a for a in m.mapper.active if a.label.startswith("opto:")]


# ------------------------------------------------------------------ meters
def test_decision_meters_thresholds():
    thr = {"jump_hz": 60.0, "groom_hz": 20.0, "mn9_hz": 30.0, "backward_ref_hz": 20.0,
           "r_ref_hz": 30.0, "actions": True, "steer": True}
    d = {"escape": 130.0, "backward_L": 25.0, "backward_R": 15.0, "walk_L": 10.0,
         "walk_R": 0.0, "turn_L": 60.0, "turn_R": 0.0, "groom": 5.0}
    ms = {m.key: m for m in decision_meters(d, {"MN9": 55.0}, None, thr)}
    assert list(ms) == ["escape", "backup", "walk", "turn", "groom", "feed"]
    assert ms["escape"].lean == "JUMP" and ms["escape"].frac > 2
    assert ms["backup"].lean == "BACK UP" and ms["backup"].value == 20.0  # ref 20 (--brain-backup)
    assert ms["walk"].lean == "" and ms["turn"].lean == "LEFT" and ms["turn"].bipolar
    assert ms["groom"].lean == "" and ms["feed"].lean == "EXTEND"
    assert (60.0, "jump") in ms["escape"].marks
    nm = {"enabled": True, "octopamine": 0.7}
    assert decision_meters({}, {}, nm, thr)[-1].lean == "AROUSED"
    quiet = decision_meters({}, {}, None, None)
    assert all(m.lean == "" for m in quiet) and not quiet[0].active


# ------------------------------------------------------------------ window / clicks
def _states(m, n=4, lesion=True):
    if lesion:
        m.set_lesions(["DNp01"])
    labs = _drive(m, StimulusEvent("opto", "none", 1.0, 5.0, m.engine.t,
                                   details={"target": "MDN", "rate_hz": 120.0}))
    out = []
    for k in range(n):
        s, i = m.engine.run(1000)
        st = m.summarize(s, i, 0.1, m.engine.t, 1.0, labs if k == 0 else [])
        st.playground.update({"log": ["t=  0.00  LESION DNp01 (cli)"],
                              "thresholds": {"actions": True, "steer": True}})
        out.append(st)
    return out


def test_renderer_palette_clicks_and_overlays():
    from fly_simulator.brain_viz.window import BrainRenderer, render_frame

    m = _model()
    states = _states(m)
    plain = render_frame(m.layout, states)
    img = render_frame(m.layout, states, playground=True)
    assert img.shape == plain.shape == (800, 1280, 3)
    assert not np.array_equal(img, plain)
    r = BrainRenderer(m.layout, playground=True)
    for st in states:
        r.ingest(st, 1000.0)
    r.render(1000.1)  # lays out the clickable areas
    # palette cell 0 = GF DNp01: left = stimulate at the chosen rate / duration
    (x0, y0, x1, y1), p = r.pg_ui._grid()[0]
    assert p.target == "DNp01"
    cmd = r.click(x0 + 20, (y0 + y1) / 2)
    assert cmd == {"op": "stim", "target": "DNp01", "rate_hz": 120.0, "duration_s": 1.0}
    # right click: toggle lesion (DNp01 is lesioned in the state -> un-lesion)
    assert r.click(x0 + 20, (y0 + y1) / 2, "right") == {"op": "lesion", "target": "DNp01",
                                                         "on": False}
    (x0, y0, x1, y1), p = r.pg_ui._grid()[1]
    assert r.click(x0 + 20, (y0 + y1) / 2, "right")["on"] is True  # MDN not lesioned
    # the rate / duration chips change the next stimulation
    hits = {tuple(sorted(a.items())): box for box, a in r.pg_ui._hits if a["kind"] in ("rate", "dur")}
    bx = hits[(("kind", "rate"), ("value", 200.0))]
    assert r.click((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2) is None
    bx = hits[(("kind", "dur"), ("value", 0.5))]
    r.click((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)
    (x0, y0, x1, y1), _ = r.pg_ui._grid()[4]
    assert r.click(x0 + 20, (y0 + y1) / 2) == {"op": "stim", "target": "DNa02_L",
                                                "rate_hz": 200.0, "duration_s": 0.5}
    # buttons
    kinds = {a["kind"]: box for box, a in r.pg_ui._hits}
    for kind, op in (("clear", "clear_lesions"), ("stop", "stop_stim")):
        bx = kinds[kind]
        assert r.click((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2) == {"op": op}
    # the brain map: a region under the cursor
    ri = next(i for i, mk in enumerate(r._masks) if mk is not None)
    cx, cy = r._reg_px[ri]
    got = r.region_at(cx, cy)
    assert got is not None
    cmd = r.click(cx, cy)
    assert cmd["op"] == "stim" and cmd["target"] == f"region:{m.layout.regions[got]}"
    assert r.click(5, 5) is None  # header: nothing
    r.toggle_playground()
    assert not r.show_playground and r.render(1000.2).shape == (800, 1280, 3)
    (x0, y0, x1, y1), _ = r.pg_ui._grid()[0]
    assert r.click(x0 + 20, (y0 + y1) / 2) is None  # classic panels: no palette
    assert r.click(cx, cy)["op"] == "stim"  # ... but the map still works


def test_presets_resolve_on_real_names():
    """Every palette target resolves on the synthetic table or is a known name."""
    from fly_simulator.brain.mapping import TARGET_ALIASES, resolve_target

    m = _model()
    known = {"MN9", "sugar", "bitter", "LC4_L", "LPLC2", "an_walk", "OA-VUMa1",
             "jo_wind_gravity", "DNg12"} | set(TARGET_ALIASES)
    for p in PRESETS:
        r = resolve_target(m.table, p.target, m.mapper.sets)
        assert len(r.idx) or p.target in known, p.target
    assert len(PRESETS) == 16


# ------------------------------------------------------------------ app wiring
def test_cli_flags():
    from fly_simulator.app import ConfigError, build_arg_parser, config_from_args

    parse = lambda *a: config_from_args(build_arg_parser().parse_args(list(a))).brain  # noqa: E731
    b = parse("--stim", "DNa02_L:120:1@3")
    assert b.enabled and b.stim == "DNa02_L:120:1@3" and b.playground
    b = parse("--lesion", "DNp01,MDN", "--no-playground")
    assert b.enabled and b.lesion == "DNp01,MDN" and not b.playground
    with pytest.raises(ConfigError):
        parse("--stim", "MDN:-3")


def test_brain_link_playground(tmp_path):
    from fly_simulator import AppConfig
    from fly_simulator.app import Session
    from fly_simulator.brain_link import BrainLink, BrainLinkConfig

    cfg = AppConfig()
    cfg.logging.runs_dir = str(tmp_path)
    cfg.terrain.difficulty = "flat"
    cfg.brain = BrainLinkConfig(enabled=True, window=False, steer=True, synthetic=SYN,
                                stim="DNa02_L:120:0.3@0.05", lesion="DNp01")
    msgs = []
    link = BrainLink(cfg.brain, headless=True, say=msgs.append)
    link.wait_ready(60)
    s = Session(cfg, log=True, say=lambda m: None, brain=link)
    try:
        seen_turn = 0.0
        deadline = time.time() + 30
        while time.time() < deadline and s.run_time() < 0.5:
            s.sim.step(150)
            s.after_physics()
            if link.latest is not None:
                seen_turn = max(seen_turn, link.latest.descending["turn_L"])
        assert link.lesions == ["DNp01"]
        assert seen_turn > 30  # DNa02_L stimulated -> turn_L group rate
        pg = link.latest.playground
        assert pg["lesions"][0]["target"] == "DNp01" and pg["thresholds"]["steer"]
        assert any("LESION DNp01" in ln for ln in pg["log"])
        # keys: select / stimulate / lesion
        assert "selected" in link.handle_playground_key("9")
        assert "STIM MDN" in link.handle_playground_key("0")
        assert "LESION MDN" in link.handle_playground_key("-")
        assert link.lesions == ["DNp01", "MDN"]
        # window commands
        assert "UNLESION DNp01" in link.execute({"op": "lesion", "target": "DNp01", "on": False})
        assert "CLEAR" in link.execute({"op": "clear_lesions"}) and link.lesions == []
        assert "STOP" in link.execute({"op": "stop_stim"})
        assert link.execute({"op": "bogus"}) is None
        assert any(ln.startswith("PLAYGROUND") for ln in link.hud_lines())
        img = link.render_snapshot()
        assert img is not None and img.shape == (800, 1280, 3)
    finally:
        s.close("test")
        link.close()
    rows = list(csv.DictReader(open(s.logger.run_dir / "events.csv")))
    kinds = [r["event_type"] for r in rows]
    assert kinds.count("brain_opto") == 3 and kinds.count("brain_lesion") == 4
    summary = json.loads((s.logger.run_dir / "summary.json").read_text())
    assert summary["brain"]["playground"]["lesions"] == []
    assert len(summary["brain"]["playground"]["log"]) == 7
