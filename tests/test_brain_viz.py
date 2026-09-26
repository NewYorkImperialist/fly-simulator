"""Brain window: headless rendering on mock data, robustness, process lifecycle."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np
import pytest

from fly_simulator.brain.schema import (
    DESCENDING_GROUPS,
    NEUROTRANSMITTERS,
    BrainLayout,
    BrainState,
    StimulusEvent,
)
from fly_simulator.brain_viz.atlas import load_atlas, normalize_region_name
from fly_simulator.brain_viz.mock import MockBrain, mock_scenario
from fly_simulator.brain_viz.window import (
    DEFAULT_SIZE,
    BrainRenderer,
    BrainWindowProcess,
    render_frame,
    render_timeline,
    short_stimulus,
)

H, W = DEFAULT_SIZE[1], DEFAULT_SIZE[0]


@pytest.fixture(scope="module")
def whip():
    return mock_scenario("whip", n_display=600)


def _state(n_regions: int, **over) -> BrainState:
    kw = dict(brain_time=1.0, wall_time=time.time(), realtime_factor=0.1, window_s=0.1,
              rate_by_nt=np.zeros(len(NEUROTRANSMITTERS)),
              active_frac_by_nt=np.zeros(len(NEUROTRANSMITTERS)),
              rate_by_region=np.zeros(n_regions), descending={},
              raster_idx=np.zeros(0, np.int32), raster_t=np.zeros(0, np.float32),
              total_spikes=0)
    kw.update(over)
    return BrainState(**kw)


def test_atlas_asset():
    a = load_atlas()
    assert a is not None and len(a["regions"]) >= 70
    assert {"AL_L", "AL_R", "GNG", "FB", "EB", "PB", "MB_CA_L", "ME_R"} <= set(a["regions"])
    assert a["brain_outline"] and all(len(o) >= 3 for o in a["brain_outline"])
    # left neuropils on the image left
    assert a["regions"]["AL_L"]["centroid"][0] < a["regions"]["AL_R"]["centroid"][0]
    assert normalize_region_name("mb ca (left)") == "MB_CA_L"
    assert normalize_region_name("AL") == "AL" and normalize_region_name("SEZ") == "GNG"


def test_mock_stream_consistent():
    mb = MockBrain(n_display=500, seed=3)
    L = mb.layout
    n = len(L.display_neuron_ids)
    assert n == 500 and len(L.display_neuron_label) == n
    assert L.display_neuron_xy.shape == (n, 2) and L.region_xy.shape == (len(L.regions), 2)
    assert L.display_neuron_region.max() < len(L.regions)
    quiet = mb.advance(0.2)
    mb.stimulate(StimulusEvent("whip_hit", "left", 0.9))
    hit = mb.advance(0.1)
    assert set(hit.descending) == set(DESCENDING_GROUPS)
    assert hit.descending["escape"] > quiet.descending["escape"] + 10
    assert hit.descending["turn_R"] > hit.descending["turn_L"]  # turn away from the hit
    i = L.regions.index("AMMC_L")
    assert hit.rate_by_region[i] > 3 * quiet.rate_by_region[i]
    assert hit.recent_stimuli and "whip_hit" in hit.recent_stimuli[0]
    assert len(hit.raster_idx) == len(hit.raster_t) and hit.raster_idx.max() < n


@pytest.mark.parametrize("kind", ["quiet", "whip", "high"])
def test_render_frame_mock(kind):
    L, states = mock_scenario(kind, n_display=600)
    img = render_frame(L, states)
    assert img.shape == (H, W, 3) and img.dtype == np.uint8
    assert img.mean() > 10 and img.std() > 10


def test_whip_lights_up_left_side(whip):
    L, states = whip
    r = BrainRenderer(L)
    img = render_frame(L, states)
    before = render_frame(L, states[:-2])  # same run just before the whip
    # the whip brightens the fly's left half of the brain map far more than the right
    m = r.r_map
    mid = int(r._reg_px[L.regions.index("GNG")][0])
    gain = img[m.y:m.y2, m.x:m.x2].astype(float) - before[m.y:m.y2, m.x:m.x2]
    left, right = gain[:, :mid - m.x].mean(), gain[:, mid - m.x:].mean()
    assert left > 3 and left > 1.3 * right


def test_empty_and_degenerate_states(whip):
    L, _ = whip
    R = len(L.regions)
    assert render_frame(L, None).shape == (H, W, 3)          # nothing received yet
    render_frame(L, _state(R))                                 # all zeros
    render_frame(L, _state(R, rate_by_region=None, descending=None, recent_stimuli=None,
                           raster_idx=None, raster_t=None, rate_by_nt=None,
                           active_frac_by_nt=None, realtime_factor=None, window_s=None))
    render_frame(L, _state(R, rate_by_region=np.full(3, np.nan),        # wrong length
                           raster_idx=np.array([0, 10**6, -5]),          # out of range
                           raster_t=np.array([0.95, 0.99]),              # mismatched
                           descending={"walk_L": float("nan"), "bogus": 3.0},
                           recent_stimuli=["whip_hit left 0.6", 42]))
    render_frame(L, _state(R, realtime_factor=3.0), size=(900, 600))   # faster than RT


def _layout(regions, region_xy, n=50, xy=None, outlines=None, nt=None) -> BrainLayout:
    rng = np.random.default_rng(0)
    return BrainLayout(
        regions=list(regions), region_xy=np.asarray(region_xy, np.float32),
        region_outline_xy=outlines if outlines is not None else [],
        display_neuron_ids=np.arange(n, dtype=np.int64),
        display_neuron_region=rng.integers(0, max(len(regions), 1), n),
        display_neuron_nt=nt if nt is not None else rng.integers(-1, 6, n),
        display_neuron_xy=xy if xy is not None else np.zeros((n, 2), np.float32),
        display_neuron_label=[""] * n, n_neurons_total=1000, model_name="test")


def test_layout_variants():
    atlas = load_atlas()
    names = ["AL_L", "AL_R", "GNG", "MB_CA_L", "MB_CA_R", "FB", "LO_L", "LO_R"]
    um = np.array([atlas["regions"][n]["centroid"] for n in names])
    # engine uses nm with y flipped: renderer should register it onto the atlas
    nm = np.stack([um[:, 0] * 1000, -um[:, 1] * 1000], 1)
    r = BrainRenderer(_layout(names, nm, xy=nm[np.arange(50) % len(names)]))
    assert r.mode == "atlas"
    ref = BrainRenderer(_layout(names, um))
    assert np.allclose(r._reg_px, ref._reg_px, atol=1.0)
    # names the atlas doesn't know + explicit outlines -> own coordinates
    sq = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32) + i * 2 for i in range(3)]
    r2 = BrainRenderer(_layout(["foo", "bar", "baz"], [[0.5, 0.5], [2.5, 2.5], [4.5, 4.5]],
                               outlines=sq))
    assert r2.mode == "layout"
    r2.ingest(_state(3, rate_by_region=np.array([1.0, 50.0, 150.0])), 0.0)
    assert r2.render(0.1).shape == (H, W, 3)
    # no usable positions at all, and a zero-neuron layout
    BrainRenderer(_layout(["x", "y"], np.full((2, 2), np.nan))).render(0.0)
    BrainRenderer(_layout(names, um, n=0, xy=np.zeros((0, 2)), nt=np.zeros(0, int))) \
        .render(0.0)
    # neuron grouping toggle
    r.toggle_grouping()
    assert r.group_by == "nt" and r.render(0.0).shape == (H, W, 3)


def test_interpolation_is_smooth():
    """A slow brain (1 state/s) still animates: playhead advances between states."""
    mb = MockBrain(n_display=400, realtime_factor=0.05)
    r = BrainRenderer(mb.layout)
    r.ingest(mb.advance(0.05), 0.0)
    r.ingest(mb.advance(0.05), 1.0)
    heads = []
    for k in range(1, 30):
        r.step(1.0 + k / 30)
        heads.append(r._playhead(1.0 + k / 30))
    d = np.diff(heads)
    assert (d > 0).sum() > 20 and heads[-1] <= mb.brain_time + 1e-9


def test_render_timeline_and_stimulus_labels():
    L, events = mock_scenario("whip", n_display=300)
    evs = [(0.25 * i, s) for i, s in enumerate(events)]
    evs.append((0.3, StimulusEvent("shove", "front", 0.7)))
    frames = list(render_timeline(L, sorted(evs, key=lambda e: e[0]), 1.0, fps=10))
    assert len(frames) == 10 and frames[-1].shape == (H, W, 3)
    assert short_stimulus("whip_hit left 0.6") == "WHIP L 0.60"
    assert short_stimulus(StimulusEvent("fall", "none", 1)) == "FALL 1.00"


@pytest.mark.skipif(sys.platform != "darwin" and not os.environ.get("DISPLAY"),
                    reason="needs a GUI display")
def test_window_process_starts_and_stops(whip):
    L, states = whip
    proc = BrainWindowProcess(L, max_seconds=20, display_scale=0.5).start()
    try:
        time.sleep(0.5)
        for s in states:
            assert proc.send(s)
        time.sleep(0.8)
        assert proc.is_alive()
    finally:
        proc.close(timeout=5)
    assert not proc.is_alive() and proc.process.exitcode is not None
    assert not proc.send(states[-1])   # sending after close is a no-op


@pytest.mark.skipif(sys.platform != "darwin" and not os.environ.get("DISPLAY"),
                    reason="needs a GUI display")
def test_window_process_exits_when_parent_dies():
    code = (
        "import time, os\n"
        "from fly_simulator.brain_viz.mock import MockBrain\n"
        "from fly_simulator.brain_viz.window import BrainWindowProcess\n"
        "if __name__ == '__main__':\n"
        "    p = BrainWindowProcess(MockBrain(n_display=200).layout, max_seconds=30,\n"
        "                           display_scale=0.5).start()\n"
        "    time.sleep(1.5)\n"
        "    print(p.process.pid, flush=True)\n"
        "    os._exit(0)\n"   # abrupt: no close(), no atexit
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=60, cwd=os.path.dirname(os.path.dirname(__file__)))
    pid = int(out.stdout.split()[-1])
    deadline = time.time() + 10
    alive = True
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            alive = False
            break
        time.sleep(0.2)
    if alive:
        os.kill(pid, 9)
    assert not alive, "brain window outlived its parent"
