"""Taste patches (perpetualfly/senses/taste.py, docs/TASTE.md): patches, leg sensing,
the brain mapping of ``taste`` events and the feeding rule (synthetic / fake brain)."""

import json

import numpy as np
import pytest

from perpetualfly.actions import ActionManager
from perpetualfly.actions.jump import Jump
from perpetualfly.app import (KEY_TABLE, ConfigError, Session, build_arg_parser,
                              config_from_args)
from perpetualfly.brain.schema import StimulusEvent
from perpetualfly.config import AppConfig
from perpetualfly.senses.taste import (PATCH_KINDS, TASTE_KEYS, FeedingRule, TasteConfig,
                                       TastePatches)
from perpetualfly.simulation import Simulation


def _session(**taste):
    cfg = AppConfig()
    cfg.taste.enabled = True
    cfg.taste.density_per_cm = 0.0
    for k, v in taste.items():
        setattr(cfg.taste, k, v)
    return Session(cfg, log=False, say=lambda m: None)


def _run(s, n_chunks, steps=150):
    for _ in range(n_chunks):
        s.sim.step(steps)
        s.after_physics()


# ------------------------------------------------------------------ config / CLI
def test_cli_flags_and_keys():
    parse = lambda *a: config_from_args(build_arg_parser().parse_args(list(a)))  # noqa: E731
    assert not parse().taste.enabled and AppConfig().taste.enabled is False
    c = parse("--taste-patches", "--taste-density", "0.5")
    assert c.taste.enabled and c.taste.density_per_cm == 0.5
    with pytest.raises(ConfigError):
        parse("--taste-density", "0.5")  # needs --taste-patches
    with pytest.raises(ConfigError):
        parse("--taste-patches", "--job", "sisyphus")
    keys = [part.lower() for k, _ in KEY_TABLE for part in k.replace(" / ", " ").split()]
    assert set(TASTE_KEYS) <= set(keys) and set(TASTE_KEYS.values()) == set(PATCH_KINDS)
    # config round trip (tuple field)
    d = AppConfig().to_dict()
    assert AppConfig.from_dict(d).taste.radius_mm == TasteConfig().radius_mm


# ------------------------------------------------------------------ patches
def test_procedural_patches_are_deterministic_and_in_the_band():
    p = TastePatches(TasteConfig(density_per_cm=1.0, seed=3))
    cells = [p.cell_patches(i, 1.0) for i in range(0, 30)]
    again = [p.cell_patches(i, 1.0) for i in range(0, 30)]
    assert [[(q.x, q.y, q.r, q.kind) for q in c] for c in cells] == \
           [[(q.x, q.y, q.r, q.kind) for q in c] for c in again]
    flat = [q for c in cells for q in c]
    assert 30 < len(flat) < 90  # ~2 per 20 mm cell
    assert {q.kind for q in flat} == set(PATCH_KINDS)
    c = p.cfg
    for q in flat:
        assert abs(q.y - 1.0) <= c.band_mm + 1e-9 and c.radius_mm[0] <= q.r <= c.radius_mm[1]
        assert q.x - q.r >= c.first_x_mm
    assert TastePatches(TasteConfig(density_per_cm=0.0)).cell_patches(3) == []


def test_patches_are_visual_only_and_physics_is_unchanged():
    def run(enabled):
        cfg = AppConfig()
        cfg.terrain.difficulty = "flat"
        cfg.taste.enabled = enabled
        s = Session(cfg, log=False, say=lambda m: None)
        if enabled:
            s.taste.patches.place("sugar", 3.0, 0.0, 2.2)
        _run(s, 15)
        return s

    a, b = run(False), run(True)
    m = b.sim.model
    ids = b.taste.patches._pool + b.taste.patches._spawn_pool
    assert all(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 for g in ids)
    assert np.array_equal(a.sim.data.qpos, b.sim.data.qpos)  # bit-identical walking
    assert b.taste.sense.contact_s["sugar"] > 0.05  # ... and it did walk onto the patch


def test_procedural_window_recycles_and_spawn_ahead():
    s = _session(density_per_cm=1.0, pool_size=12)
    tp = s.taste.patches
    shown = tp.patches()
    assert shown and all(p.gid >= 0 for p in shown)
    x = s.sim.thorax_position()[0]
    assert all(x - tp.cfg.behind_mm - tp.cfg.cell_mm <= p.x <= x + tp.cfg.ahead_mm + tp.cfg.cell_mm
               for p in shown)
    msg = s.spawn_taste("bitter")
    assert "bitter" in msg and tp.spawned[-1].kind == "bitter"
    p = tp.spawned[-1]
    assert p.x - p.r == pytest.approx(x + tp.cfg.spawn_distance_mm, abs=0.3)
    for _ in range(tp.cfg.spawn_pool + 2):  # the spawn pool recycles the oldest
        tp.spawn_ahead("sugar")
    assert len(tp.spawned) == tp.cfg.spawn_pool
    s.sim.reset()  # a reset drops the spawned patches
    assert tp.spawned == []


# ------------------------------------------------------------------ sensing
def test_walking_onto_patches_sends_sustained_taste_events():
    s = _session()
    sent = []
    s.taste.sense.send = lambda ev, refresh: sent.append((ev, refresh))
    s.taste.patches.place("mixed", 4.0, 0.0, 2.2)
    _run(s, 25)  # 0.375 s: the front legs reach the patch at ~0.1 s
    assert sent, "no taste event"
    ev, refresh = sent[0]
    assert ev.kind == "taste" and not refresh and ev.duration_s == s.cfg.taste.stim_duration_s
    assert set(ev.details["tastes"]) == {"sugar", "bitter"}
    assert 0 < ev.details["sugar_hz"] <= 200 and ev.details["sugar_legs"]
    assert any(r for _, r in sent), "unchanged contact must be refreshed"
    gaps = np.diff([e.sim_time for e, _ in sent])
    assert gaps.max() <= s.cfg.taste.refresh_s + 0.02  # sustained while in contact
    n = len(sent)
    _run(s, 70)  # walks off the far edge
    assert not s.taste.sense.reading.any
    t_last = sent[-1][0].sim_time
    _run(s, 10)
    assert len(sent) >= n and sent[-1][0].sim_time == t_last  # stopped after leaving


def test_rate_scales_with_legs():
    s = _session()
    sense = s.taste.sense
    assert sense.rate(1) == pytest.approx(200 / 3) and sense.rate(3) == 200 == sense.rate(6)


def test_mapper_resolves_taste_events():
    from perpetualfly.brain.mapping import StimulusMapper
    from perpetualfly.brain.process import _synthetic_table

    table, _ = _synthetic_table(n=40)
    mp = StimulusMapper(table)
    mp.sets["sugar"] = np.array([0, 1])
    mp.sets["bitter"] = np.array([2])
    mp.sets["leg_gustatory"] = np.array([0, 1, 2, 3])  # sides alternate L R L R
    ev = StimulusEvent("taste", "left", 1.0, 0.15, 0.0,
                       details={"tastes": ["sugar", "bitter"], "sugar_hz": 133.0,
                                "bitter_hz": 67.0, "leg_hz": 40.0})
    out = {lab: (list(i), r) for lab, i, r in mp.resolve(ev)}
    assert out["taste:sugar:left"] == ([0, 1], 133.0)
    assert out["taste:bitter:left"] == ([2], 67.0)
    assert out["taste:leg_gustatory:left"] == ([0, 2], 40.0)
    assert mp.resolve(StimulusEvent("taste", details={"tastes": ["umami"]})) == []


# ------------------------------------------------------------------ feeding rule
@pytest.fixture(scope="module")
def full_body_mgr():
    cfg = AppConfig()
    cfg.fly.extra_joints = True
    sim = Simulation(cfg)
    return sim, ActionManager(sim)


def test_feeding_rule_stops_extends_and_resumes(full_body_mgr):
    sim, mgr = full_body_mgr
    sim.reset()
    c = TasteConfig(feed_max_s=0.6, feed_release_s=0.1, feed_refractory_s=0.5)
    rule = FeedingRule(mgr, c, can_proboscis=True)
    try:
        sim.step(500)
        assert rule.update(sim.time, 80.0, sugar_recent=False) is None  # no sugar: no feed
        assert rule.update(sim.time, 10.0, sugar_recent=True) is None  # MN9 low: no feed
        assert rule.update(sim.time, 80.0, sugar_recent=True) is not None
        x0 = sim.thorax_position()[:2].copy()
        t0 = sim.time
        while not rule.bouts and sim.time - t0 < 1.0:  # MN9 high -> feeds until feed_max_s
            sim.step(100)
            rule.update(sim.time, 80.0, True)
        assert rule.bouts and rule.bouts[0]["reason"] == "max_bout"
        assert rule.bouts[0]["duration_s"] == pytest.approx(0.6, abs=0.02)
        assert np.hypot(*(sim.thorax_position()[:2] - x0)) < 0.3  # stood still
        sim.step(2000)
        rule.update(sim.time, 80.0, True)
        # refractory: no new bout right away, then feeding again; MN9 drop ends it
        assert rule.update(sim.time, 80.0, True) is None
        sim.step(int(0.6 / sim.timestep))
        assert rule.update(sim.time, 80.0, True) is not None
        sim.step(200)
        rule.update(sim.time, 80.0, True)
        sim.step(100)
        rule.update(sim.time, 0.0, True)
        sim.step(1200)
        rule.update(sim.time, 0.0, True)
        sim.step(4000)
        assert rule.bouts[-1]["reason"] == "mn9_low" and not mgr.busy
    finally:
        mgr.listeners.remove(rule._on_event)


def test_feed_extends_the_proboscis_and_never_interrupts_a_jump(full_body_mgr):
    sim, mgr = full_body_mgr
    sim.reset()
    rule = FeedingRule(mgr, TasteConfig(), can_proboscis=True)
    try:
        sim.step(300)
        rule.update(sim.time, 80.0, True)
        sim.step(3000)
        m = sim.model
        j = m.joint(f"{sim.fly_name}/c_head-c_rostrum-pitch")
        assert sim.data.qpos[j.qposadr[0]] < -0.5  # rostrum swung down
        assert mgr.active_name == "feed"
        mgr.cancel()
        sim.step(3000)
        mgr.trigger(Jump(), source="brain")
        sim.step(10)
        assert rule.update(sim.time, 80.0, True) is None and mgr.active_name == "jump"
    finally:
        mgr.listeners.remove(rule._on_event)
        mgr.cancel()


# ------------------------------------------------------------------ app + synthetic brain
def test_headless_app_with_synthetic_brain(tmp_path):
    pytest.importorskip("numba")
    from perpetualfly.app import run
    from perpetualfly.brain_link import BrainLinkConfig

    cfg = AppConfig()
    cfg.logging.runs_dir = str(tmp_path)
    cfg.fly.extra_joints = True
    cfg.taste.enabled = True
    cfg.taste.density_per_cm = 0.0
    cfg.brain = BrainLinkConfig(enabled=True, window=False, steer=True, actions=True,
                                synthetic={"n": 50, "p_conn": 0.1, "seed": 0})
    run(cfg, headless=True, max_seconds=0.8, script_keys=[(0.05, "5")])
    run_dir = next(tmp_path.iterdir())
    s = json.loads((run_dir / "summary.json").read_text())
    extra = s.get("taste")
    assert extra is not None and extra["patches_spawned"] == 1
    assert extra["taste_events"] >= 1 and extra["feeding_bouts"] == []  # no MN9 in the toy net
    events = (run_dir / "events.csv").read_text()
    assert "taste_spawn" in events and "brain_stim" in events and "taste" in events
