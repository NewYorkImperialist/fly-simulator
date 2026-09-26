"""Obstacle-course mode: format, layout, pool fit, race logic, respawn, leaderboard."""

import json

import pytest

from fly_simulator import AppConfig
from fly_simulator.app import Session
from fly_simulator.course import (CourseOptions, CourseSpec, build_layout, builtin_courses,
                                 install_course)
from fly_simulator.course import leaderboard as lb
from fly_simulator.metrics import FallState
from fly_simulator.terrain import ProceduralTerrainConfig, SectionedGenerator


def _spec(sections, **kw):
    return CourseSpec.from_dict({"name": "t", "sections": sections, **kw})


# ---------------------------------------------------------------- format / layout
def test_builtin_courses_load_and_fit_the_pool():
    assert {"tutorial", "gauntlet", "slalom", "brain_test"} <= set(builtin_courses())
    for name in builtin_courses():
        spec = CourseSpec.load(name)
        lay = build_layout(spec)
        assert lay.finish_x > lay.start_x and spec.sections[-1].type == "finish"
        cfg = ProceduralTerrainConfig(chunk_length=spec.chunk_length, chunks_behind=1,
                                      chunks_ahead=5)
        assert SectionedGenerator(lay.geoms, lay.segments, cfg).validate() == [], name
        # checkpoints are on flat floor: no geom other than the gate covers them
        for cp in lay.checkpoints:
            for g in lay.geoms:
                x0, x1, y0, y1 = g.xy_extent()
                if x0 <= cp.x <= x1 and y0 <= 0.0 <= y1 and g.top_z() > 0.05:
                    assert g.kind in ("gate_bar",), (name, cp, g.kind)


def test_format_validation_errors():
    with pytest.raises(ValueError, match="unknown type"):
        _spec([{"type": "lava"}])
    with pytest.raises(ValueError, match="unknown params"):
        _spec([{"type": "stairs", "params": {"step_hight": 0.1}}])
    with pytest.raises(ValueError, match="unknown keys"):
        CourseSpec.from_dict({"name": "x", "sections": [{"type": "flat"}], "speed": 3})
    with pytest.raises(ValueError, match="no sections"):
        CourseSpec.from_dict({"name": "x", "sections": []})
    with pytest.raises(ValueError, match="last section"):
        _spec([{"type": "finish"}, {"type": "flat"}])
    s = _spec([{"type": "flat", "length": 5}])
    assert s.sections[-1].type == "finish"  # appended


def test_layout_sections_route_gates_and_triggers():
    spec = _spec([
        {"type": "stairs", "params": {"steps_up": 3, "steps_down": 3, "step_height": 0.1}},
        {"type": "checkpoint"},
        {"type": "pillars", "params": {"count": 3, "spacing": 8, "offset": 2.5}},
        {"type": "tunnel", "length": 10},
        {"type": "whip_gauntlet", "length": 12, "params": {"cracks": 2}},
        {"type": "loom_zone", "length": 6},
    ])
    lay = build_layout(spec)
    types = [s.type for s in lay.sections]
    assert types == ["stairs", "checkpoint", "pillars", "tunnel", "whip_gauntlet",
                     "loom_zone", "finish"]
    assert all(a.x1 == pytest.approx(b.x0) for a, b in zip(lay.sections, lay.sections[1:]))
    stairs = [g for g in lay.geoms if g.kind == "stairs"]
    assert max(g.top_z() for g in stairs) == pytest.approx(0.3)
    assert len(lay.gates) == 3 and [g.side for g in lay.gates] == ["left", "right", "left"]
    for g in lay.gates:  # the racing line passes each pillar on its side
        y = lay.route_y(g.x)
        assert (y > 1.5) if g.side == "left" else (y < -1.5)
    assert lay.route_y(-5.0) == 0.0 and lay.route_y(lay.finish_x + 1) == 0.0
    assert [t.kind for t in lay.triggers] == ["whip", "whip", "loom"]
    assert any(g.kind == "ceiling" and g.pos[2] > 1.8 for g in lay.geoms)
    assert lay.section_at(lay.sections[3].x0 + 1).type == "tunnel"


def test_leaderboard_ranks_and_keys(tmp_path):
    p = tmp_path / "lb.json"
    e = lambda t: {"total_time_s": t, "time_s": t, "penalty_s": 0.0, "n_falls": 0,  # noqa: E731
                   "n_respawns": 0}
    assert lb.add_entry(p, "c", "hybrid", e(10.0)) == 1
    assert lb.add_entry(p, "c", "hybrid", e(8.0)) == 1
    assert lb.add_entry(p, "c", "hybrid", e(12.0)) == 3
    assert lb.add_entry(p, "c", "cpg", e(20.0)) == 1
    assert lb.best(p, "c", "hybrid")["total_time_s"] == 8.0
    assert lb.add_entry(p, "c", "hybrid", e(30.0), keep=3) is None
    data = json.loads(p.read_text())
    assert len(data["courses"]["c"]["hybrid"]) == 3 and "cpg" in data["courses"]["c"]
    cfg = AppConfig()
    assert lb.controller_mode(cfg) == "hybrid"
    cfg.brain.enabled = cfg.brain.steer = True
    assert lb.controller_mode(cfg) == "brain-steer"


# --------------------------------------------------------------------- live race
@pytest.fixture(scope="module")
def session(tmp_path_factory):
    cfg = AppConfig()
    cfg.logging.runs_dir = str(tmp_path_factory.mktemp("runs"))
    cfg.session.fall_hint_after_s = 0.0
    s = Session(cfg, log=True, say=lambda msg: None)
    yield s


def test_course_race_checkpoint_respawn_finish(session, tmp_path):
    s = session
    spec = _spec([
        {"type": "flat", "length": 1},
        {"type": "checkpoint", "params": {"pad": 2}},
        {"type": "tunnel", "length": 3},
        {"type": "whip_gauntlet", "length": 3, "params": {"cracks": 1, "level": 1}},
        {"type": "loom_zone", "length": 2},
    ], start_x=0.5, lead_in=1.0, respawn_after_s=0.3)
    stims = []
    course = install_course(s, spec, s.cfg, stimulus_fn=stims.append,
                            options=CourseOptions(leaderboard=str(tmp_path / "lb.json")))
    try:
        # endless pool reused: nothing was rebuilt, pool geoms now show the course
        kinds = {course.layout.geoms[0].kind}
        assert kinds and s.terrain.n_pool_geoms == len(s.terrain.geom_ids)
        cp = course.layout.checkpoints[0]
        # ground under the tunnel roof is the floor, not the roof
        tun = next(sec for sec in course.layout.sections if sec.type == "tunnel")
        assert s.ground_height((tun.x0 + tun.x1) / 2, 0.0) == pytest.approx(0.0, abs=1e-6)
        assert s.step_difficulty(1).startswith("[course]")
        while course.next_cp == 0 and s.run_time() < 3.0:
            s.sim.step(200)
            s.after_physics()
        assert course.status == "running" and course.next_cp == 1
        assert course.splits[0]["time_s"] > 0
        # Flip the fly onto its back (test setup only) -> FALLEN -> explicit respawn
        a = s.sim._free_qpos
        s.sim.data.qpos[a + 2] += 1.0
        s.sim.data.qpos[a + 3:a + 7] = [0.0, 1.0, 0.0, 0.0]
        n_resets = s.metrics.n_resets
        for _ in range(40):
            s.sim.step(500)
            s.after_physics()
            if course.respawns:
                break
        assert len(course.respawns) == 1 and len(course.falls) == 1
        assert s.metrics.n_resets == n_resets + 1  # counted as a reset
        x, y, _ = s.sim.thorax_position()
        assert 0.0 < x - cp.x < 1.5 and s.detector.state == FallState.UPRIGHT
        assert course.penalty_s == pytest.approx(spec.fall_penalty_s)
        while course.status == "running" and s.run_time() < 12.0:
            s.sim.step(300)
            s.after_physics()
        assert course.status == "finished", course.hud_lines()
        assert course.done and course.n_whip == 1 and course.n_loom == 1 and stims
        assert stims[0].kind == "manual" and stims[0].details["set"] == "LC4"
        r = course.laps[-1]
        assert r["total_time_s"] == pytest.approx(r["time_s"] + r["penalty_s"], abs=1e-3)
        assert r["rank"] == 1 and lb.best(tmp_path / "lb.json", "t", "hybrid")
        assert any(line.startswith("FINISH") for line in course.hud_lines())
        res = json.loads(course.results_path().read_text())
        assert res["laps"][0]["status"] == "finished" and res["laps"][0]["n_respawns"] == 1
    finally:
        course.uninstall()
    # back to endless: generator restored, respawn keyframe restored
    assert not isinstance(s.terrain.generator, SectionedGenerator)
    assert s.terrain.cfg.chunk_length == 12.0
    s.reset("manual")
    assert abs(s.sim.thorax_position()[0]) < 1.0  # back at the spawn point


def test_slalom_steering_sets_heading_target(session, tmp_path):
    s = session
    spec = _spec([{"type": "pillars", "params": {"count": 2, "spacing": 8, "lead": 3}}],
                 start_x=0.5, lead_in=1.0, steer_lookahead=4.0)
    course = install_course(s, spec, options=CourseOptions(leaderboard=""), reset=True)
    try:
        g = course.layout.gates[0]
        heads = []
        while s.sim.thorax_position()[0] < g.x - 1.0 and s.run_time() < 30:
            s.sim.step(200)
            heads.append(s.sim.controller._target_heading)
        # route to a left-side gate -> target heading turns left (positive yaw)
        assert max(heads) > 0.2
        assert s.sim.thorax_position()[1] > 0.5  # and the fly actually moved left
    finally:
        course.uninstall()
    assert s.sim.controller._target_heading == 0.0
