"""Brain recording + replay (fly_simulator/brain_viz/replay.py, docs/BRAIN_REPLAY.md)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fly_simulator.brain.schema import StimulusEvent
from fly_simulator.brain_viz.mock import MockBrain
from fly_simulator.brain_viz.replay import (TIMELINE_H, BrainRecorder, BrainRecording,
                                           ReplayPlayer, event_class, render_png, render_range)

SIZE = (640, 400)


def _record(tmp_path, seconds=3.0, win=0.02, **kw):
    mb = MockBrain(n_display=300, seed=1)
    rec = BrainRecorder(tmp_path, mb.layout, say=lambda m: None, **kw)
    states = []
    for k in range(int(round(seconds / win))):
        if k == int(1.0 / win):
            ev = StimulusEvent("manual", "none", 1.0, 0.5, mb.brain_time,
                               details={"set": "LC4", "label": "LOOM LC4"})
            mb.stimulate(ev)
            rec.add_stimulus(ev, "key")
        st = mb.advance(win)
        st.sim_time = st.brain_time + 0.5  # fly time = brain time + a clock offset
        st.habituation = {"enabled": True, "efficacy": 0.5}
        states.append(st)
        rec.add_state(st)
    rec.add_event("action", 1.62, "jump", rate_hz=75.0)
    return rec, states


def test_round_trip_merges_and_preserves(tmp_path):
    rec, states = _record(tmp_path, seconds=3.0, win=0.02, chunk_states=7)
    s = rec.close()
    assert s["states"] == 30 and rec.n_chunks == 5 and s["events"] == 2
    r = BrainRecording(tmp_path)  # run dir or brain_rec/ both work
    assert len(r.states) == 30 and len(r.events) == 2
    lay = MockBrain(n_display=300, seed=1).layout
    assert list(r.layout.regions) == list(lay.regions)
    assert np.array_equal(r.layout.display_neuron_ids, lay.display_neuron_ids)
    # 5 x 0.02 s states merged into one 0.1 s state
    m, src = r.states[3], states[15:20]
    assert m.window_s == pytest.approx(0.1)
    assert m.brain_time == pytest.approx(src[-1].brain_time)
    assert m.sim_time == pytest.approx(src[-1].sim_time)
    assert m.total_spikes == sum(x.total_spikes for x in src)
    np.testing.assert_allclose(m.rate_by_nt, np.mean([x.rate_by_nt for x in src], 0), rtol=1e-5)
    gf = [x.descending["escape"] for x in src]
    assert m.descending["escape"] == pytest.approx(np.mean(gf), rel=1e-5)
    assert r.desc_peak_arr[3][6] == pytest.approx(max(gf), rel=1e-5)
    ridx = np.concatenate([x.raster_idx for x in src])
    rt = np.concatenate([x.raster_t for x in src])
    assert np.array_equal(np.sort(m.raster_idx), np.sort(ridx))
    np.testing.assert_allclose(np.sort(m.raster_t), np.sort(rt), atol=6e-5)
    assert m.habituation == {"enabled": True, "efficacy": 0.5}
    assert r.t_start == pytest.approx(0.5) and r.t_end == pytest.approx(3.5)
    assert [event_class(e) for e in r.events] == ["loom", "action"]
    meta = json.loads((tmp_path / "brain_rec" / "meta.json").read_text())
    assert meta["states"] == 30 and meta["bytes"] > 0


def test_size_limit_stops_recording(tmp_path):
    rec, _ = _record(tmp_path, seconds=3.0, win=0.1, chunk_states=2, max_mb=0.001)
    assert rec.stopped
    n = rec.n_states
    rec.add_state(MockBrain(n_display=300).advance(0.1))
    assert rec.n_states == n
    rec.close()
    assert json.loads((tmp_path / "brain_rec" / "meta.json").read_text())["stopped_at_limit"]


def test_player_controls_and_headless_render(tmp_path):
    rec, _ = _record(tmp_path, seconds=3.0, win=0.1)
    rec.close()
    r = BrainRecording(tmp_path)
    pl = ReplayPlayer(r, size=SIZE)
    assert pl.t == pytest.approx(r.t_start)
    pl.advance(1.0)
    assert pl.t == pytest.approx(r.t_start + 1.0) and pl._next == r.index_at(pl.t) + 1
    assert pl.handle_key(" ") and pl.paused
    pl.advance(1.0)
    assert pl.t == pytest.approx(r.t_start + 1.0)  # paused
    pl.handle_key("right")
    assert pl.t == pytest.approx(r.t_start + 3.0)
    pl.handle_key("left")
    assert pl.t == pytest.approx(r.t_start + 1.0)
    pl.handle_key("]")
    assert pl.speed == 2.0
    pl.handle_key("[")
    pl.handle_key("[")
    assert pl.speed == 0.5
    assert not pl.handle_key("q")
    assert pl.time_at_x(14) == pytest.approx(r.t_start)
    img = pl.frame()
    assert img.shape == (SIZE[1] + TIMELINE_H, SIZE[0], 3)
    pl.paused = False
    pl.speed = 8.0
    pl.advance(10.0)
    assert pl.done and pl.paused
    out = render_png(r, 2.0, tmp_path / "f.png", size=SIZE)
    assert out.is_file() and out.stat().st_size > 1000
    n = render_range(r, 1.4, 1.5, tmp_path / "frames", fps=30, size=SIZE)
    assert n == 3 and len(list((tmp_path / "frames").glob("*.png"))) == 3


def test_brain_link_records_states_stimuli_and_habituation(tmp_path):
    """--brain-record + --habituation through the app's BrainLink (synthetic brain)."""
    import time

    pytest.importorskip("numba")
    from fly_simulator import AppConfig
    from fly_simulator.app import Session
    from fly_simulator.brain_link import BrainLink, BrainLinkConfig

    cfg = AppConfig()
    cfg.logging.runs_dir = str(tmp_path)
    cfg.brain = BrainLinkConfig(enabled=True, window=False, synthetic={"n": 50, "p_conn": 0.1,
                                                                       "seed": 0},
                                record=True, habituation=True,
                                habituation_config={"pre_types": ["DNa02"], "post_types": []})
    link = BrainLink(cfg.brain, headless=True, say=lambda m: None)
    link.wait_ready(60)
    s = Session(cfg, log=True, say=lambda m: None, brain=link)
    try:
        assert link.recorder is not None
        for k in range(40):
            s.sim.step(150)
            s.after_physics()
            if k == 10:
                link.handle_key("o")
        deadline = time.time() + 20
        while time.time() < deadline and link.n_states < 5:
            s.sim.step(150)
            s.after_physics()
        assert any(line.startswith("HABITUATION") for line in link.hud_lines())
    finally:
        s.close("test")
        link.close()
    r = BrainRecording(s.logger.run_dir)
    assert len(r.states) >= 3 and r.states[-1].sim_time is not None
    assert any(e["label"].startswith("LOOM") for e in r.events)
    assert r.states[-1].habituation.get("enabled")
