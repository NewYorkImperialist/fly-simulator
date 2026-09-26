"""Real vision: compound eyes -> flyvis -> LC4 / LPLC2 bridge (docs/VISION.md).

Fast tests; the flyvis ones skip when the optional 'vision' extra (or its pretrained
model) is missing.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from fly_simulator import AppConfig
from fly_simulator.vision.bridge import (
    DEG_PER_PX,
    BridgeConfig,
    DarkExpansionBridge,
    LoomBridge,
    RealVision,
    RealVisionConfig,
    ommatidia_centers,
)
from fly_simulator.vision.eyes import CompoundEyes, EyesConfig, has_eyes, make_eyes_fly_factory
from fly_simulator.vision.flyvis_net import flyvis_available

CEN = ommatidia_centers()
C0 = CEN.mean(axis=0)
needs_flyvis = pytest.mark.skipif(flyvis_available() is not None,
                                  reason=f"flyvis unavailable: {flyvis_available()}")


# ---------------------------------------------------------------- eyes


@pytest.fixture(scope="module")
def eyes_sim():
    from fly_simulator.simulation import Simulation

    cfg = AppConfig()
    cfg.whip.enabled = False
    return Simulation(cfg, fly_factory=make_eyes_fly_factory())


def test_default_fly_has_no_eyes():
    from fly_simulator.simulation import Simulation

    cfg = AppConfig()
    cfg.whip.enabled = False
    assert not AppConfig().real_vision.enabled
    assert not has_eyes(Simulation(cfg))


def test_compound_eyes_sample_at_rate(eyes_sim):
    sim = eyes_sim
    assert has_eyes(sim)
    eyes = CompoundEyes(sim, EyesConfig(enabled=True, rate_hz=200.0)).attach()
    got = []
    eyes.listeners.append(lambda t, f: got.append((t, f.shape)))
    try:
        sim.step(int(round(0.05 / sim.timestep)))  # 50 ms at 200 Hz -> ~10 samples
    finally:
        eyes.detach()
    assert 9 <= len(got) <= 12
    assert all(s == (2, 721) for _, s in got)
    f = eyes.frame
    assert np.all((f >= 0) & (f <= 1)) and f.std() > 0.05  # sky + ground, not blank
    # the sky (top rows of the eye image) is brighter than the ground
    top = CEN[:, 0] < 150
    bottom = CEN[:, 0] > 380
    assert f[:, top].mean() > f[:, bottom].mean() + 0.2
    img = eyes.human_readable()
    assert img.shape == (2, 512, 450)


def test_swatter_props_visible_to_eyes(eyes_sim):
    """FlyGym hides geom group 1 from the eyes; our props live there, so the eye
    renderer is switched to show it."""
    eyes = CompoundEyes(eyes_sim, EyesConfig(enabled=True))
    eyes.sample()
    assert eyes_sim.fg.eye_renderer_scene_option.geomgroup[1] == 1


# ---------------------------------------------------------------- bridge (no flyvis)


def radial_field(center, speed, radius_deg=25.0, off=True):
    """Synthetic T4 / T5 (2, 4, 721): increments of the subtypes whose preferred
    direction points away from ``center`` (expansion) on a ring."""
    from fly_simulator.vision.bridge import PD_ROWCOL

    d = CEN - center
    r = np.linalg.norm(d, axis=1)
    u = d / np.maximum(r, 1e-9)[:, None]
    ring = (np.abs(r * DEG_PER_PX - radius_deg) < 6.0).astype(float)
    act = np.maximum(u @ PD_ROWCOL.T, 0).T * ring * speed  # (4, 721)
    T = np.zeros((2, 4, 721))
    T[0] = act
    zero = np.zeros_like(T)
    return (zero, T) if off else (T, zero)


def translation_field(speed, direction=(0.0, 1.0)):
    from fly_simulator.vision.bridge import PD_ROWCOL

    d = np.array(direction)
    blob = (np.linalg.norm(CEN - C0, axis=1) * DEG_PER_PX < 15).astype(float)
    act = np.maximum(PD_ROWCOL @ d, 0)[:, None] * blob * speed
    T = np.zeros((2, 4, 721))
    T[0] = act
    return np.zeros_like(T), T


def run_bridge(fields, cfg=None):
    br = LoomBridge(cfg or BridgeConfig(), centers=CEN)
    z = np.zeros((2, 4, 721))
    br.update(z, z, 0.01)  # baseline
    return [br.update(t4, t5, 0.01) for t4, t5 in fields]


def test_bridge_expansion_drives_lplc2_and_lc4_on_that_eye():
    out = run_bridge([radial_field(C0, 1.0)])[-1]
    assert out["lplc2_hz"][0] > 100 and out["lc4_hz"][0] > 100
    assert out["lplc2_hz"][1] == 0 and out["lc4_hz"][1] == 0
    # the best LPLC2 unit sits at the expansion centre
    assert np.linalg.norm(CEN[out["lplc2_unit"][0]] - C0) < 40


def test_bridge_translation_and_contraction_do_not():
    tr = run_bridge([translation_field(1.0)])[-1]
    assert tr["lplc2_hz"][0] == 0 and tr["lc4_hz"][0] == 0
    # contraction: the inward subtypes (swap a<->b, c<->d)
    t4, t5 = radial_field(C0, 1.0)
    inward = t5[:, [1, 0, 3, 2]]
    co = run_bridge([(t4, inward)])[-1]
    assert co["lplc2_hz"][0] == 0 and co["lc4_hz"][0] == 0


def test_bridge_whole_eye_flow_removed():
    """Uniform motion on every column (self-rotation) is removed as global motion."""
    from fly_simulator.vision.bridge import PD_ROWCOL

    T = np.zeros((2, 4, 721))
    T[:, 1] = 1.0  # every column: +col motion
    out = run_bridge([(np.zeros_like(T), T)])[-1]
    assert np.all(out["lplc2_hz"] == 0) and np.all(out["lc4_hz"] == 0)
    assert PD_ROWCOL.shape == (4, 2)


def test_dark_expansion_fallback():
    br = DarkExpansionBridge()
    bg = np.full((2, 721), 0.8, np.float32)
    br.update(bg, 0.01)
    fired = []
    for k in range(30):
        f = bg.copy()
        f[0, np.linalg.norm(CEN - C0, axis=1) * DEG_PER_PX < 2 + 3 * k] = 0.05
        fired.append(br.update(f, 0.01)["lc4_hz"].copy())
    fired = np.array(fired)
    assert fired[:, 0].max() > 50 and fired[:, 1].max() == 0
    br.reset()
    br.update(bg, 0.01)
    assert all(br.update(bg, 0.01)["lc4_hz"].max() == 0 for _ in range(5))


# ---------------------------------------------------------------- pipeline / events


class FakeEyes:
    def __init__(self):
        self.listeners = []
        self.period = 0.01
        self.sim = SimpleNamespace(reset_hooks=[])


def test_real_vision_sends_loom_events_fallback_backend():
    eyes = FakeEyes()
    sent = []
    rv = RealVision(eyes, RealVisionConfig(enabled=True, backend="dark_expansion", warmup_s=0.0),
                    sink=sent.append)
    assert rv.backend == "dark_expansion"
    bg = np.full((2, 721), 0.8, np.float32)
    for k in range(40):
        f = bg.copy()
        if k >= 10:
            f[1, np.linalg.norm(CEN - C0, axis=1) * DEG_PER_PX < 2 + 3 * (k - 10)] = 0.05
        for fn in eyes.listeners:
            fn(0.01 * k, f)
    assert sent and all(e.kind == "loom" and e.side == "right" for e in sent)
    assert sent[0].sim_time >= 0.1 and sent[0].details["bridge"] == "dark_expansion"
    assert {"lc4_hz", "lplc2_hz"} <= set(sent[0].details)
    # refreshes, not one event per frame
    assert len(sent) < 25
    rv.detach()
    assert not eyes.listeners


# ---------------------------------------------------------------- looming object


def test_looming_object_approach_fade_and_geometric_source():
    from fly_simulator.simulation import Simulation
    from fly_simulator.vision.objects import LoomingObject, LoomingObjectConfig

    cfg = AppConfig()
    cfg.whip.enabled = False
    obj = LoomingObject(LoomingObjectConfig(appear_s=0.02, fade_s=0.01, hold_s=0.01))
    sim = Simulation(cfg, world_extensions=[obj.extension])
    obj.attach(sim)
    src = obj.geometric_source()
    assert len(src.shapes()[3]) == 0  # parked: invisible to the geometric sense
    onsets = []
    obj.onset_listeners.append(onsets.append)
    obj.start("approach", azimuth_deg=60.0, l_over_v=0.1, dist_mm=5.0)  # 10 mm/s
    assert sim.model.geom_rgba[obj.geom_id][3] == 0.0  # fades in
    sim.step(int(round(0.021 / sim.timestep)))
    assert obj.phase == "move" and onsets
    d0 = np.linalg.norm(obj.position() - obj.head_position())
    sim.step(int(round(0.1 / sim.timestep)))
    d1 = np.linalg.norm(obj.position() - obj.head_position())
    assert d0 - d1 == pytest.approx(1.0, abs=0.1)  # 10 mm/s for 0.1 s
    assert len(src.shapes()[3]) == 1
    # fixed bearing: ~60 deg to the left of the heading
    rel = obj.position() - obj.head_position()
    yaw = sim.heading()
    az = math.degrees(math.atan2(rel[1], rel[0]) - yaw)
    assert az == pytest.approx(60.0, abs=8.0)
    sim.step(int(round(0.4 / sim.timestep)))
    assert obj.phase == "idle"  # stopped, held, faded out, parked


# ---------------------------------------------------------------- app wiring


def test_cli_real_vision_implies_brain_and_low_latency_pacing():
    from fly_simulator.app import build_arg_parser, config_from_args

    cfg = config_from_args(build_arg_parser().parse_args(["--real-vision", "--headless"]))
    assert cfg.real_vision.enabled and cfg.brain.enabled
    assert cfg.brain.window_s == 0.02 and cfg.brain.sync_wait_s == 0.05
    back = AppConfig.from_dict(cfg.to_dict())
    assert back.real_vision.enabled


def test_session_eyes_only_summary():
    from fly_simulator.app import Session

    cfg = AppConfig()
    cfg.whip.enabled = False
    cfg.real_vision.eyes_only = True
    cfg.real_vision.rate_hz = 100.0
    s = Session(cfg, log=False, say=lambda m: None)
    s.sim.step(int(round(0.03 / s.sim.timestep)))
    summ = s.summary_extra("test")["real_vision"]
    assert summ["eye_samples"] >= 3 and "n_events" not in summ


# ---------------------------------------------------------------- flyvis


@needs_flyvis
def test_flyvis_stepwise_and_retina_mapper():
    from fly_simulator.vision.flyvis_net import StepwiseFlyvis

    net = StepwiseFlyvis(dt=0.01)
    m = net.mapper
    x = np.arange(721)
    assert np.array_equal(m.flyvis_to_flygym(m.flygym_to_flyvis(x)), x)
    frame = np.full((2, 721), 0.5, np.float32)
    frame[:, CEN[:, 0] < 250] = 0.9
    for _ in range(3):
        a = net.step(frame)
    assert a.shape[0] == 2 and np.isfinite(a.cpu().numpy()).all()
    t4, t5 = net.motion()
    assert t4.shape == (2, 4, 721) and t5.shape == (2, 4, 721)
    assert net.ms_per_step() > 0
