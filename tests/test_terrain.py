import math

import mujoco as mj
import numpy as np
import pytest

from fly_simulator import AppConfig, Simulation
from fly_simulator.terrain import (
    DIFFICULTY_PRESETS,
    SPAWN_KINDS,
    ProceduralTerrain,
    ProceduralTerrainConfig,
    TerrainGenerator,
)
from fly_simulator.terrain.chunks import FLY_BIT, TERRAIN_BIT
from fly_simulator.terrain.generator import slab_box


# ------------------------------------------------------------ generator only
def _layout(gen, indices):
    return [(c.kind, [(g.shape, g.pos, g.size, g.quat) for g in c.geoms])
            for c in (gen.chunk(i) for i in indices)]


def test_generator_deterministic_under_seed():
    idx = range(-3, 60)
    a = _layout(TerrainGenerator(seed=42, difficulty="normal"), idx)
    b = _layout(TerrainGenerator(seed=42, difficulty="normal"), idx)
    c = _layout(TerrainGenerator(seed=43, difficulty="normal"), idx)
    assert a == b
    assert a != c
    # Order independence: generating chunk 30 alone gives the same result.
    gen = TerrainGenerator(seed=42, difficulty="normal")
    assert _layout(gen, [30]) == [a[list(idx).index(30)]]


def test_start_chunks_flat():
    gen = TerrainGenerator(seed=1, difficulty="chaos")
    for i in range(-10, gen.cfg.flat_start_chunks):
        spec = gen.chunk(i)
        assert spec.kind == "flat" and not spec.geoms


@pytest.mark.parametrize("difficulty", list(DIFFICULTY_PRESETS))
def test_probability_table_respected(difficulty):
    gen = TerrainGenerator(seed=7, difficulty=difficulty)
    n = 4000
    kinds = [gen.sample_kind(i) for i in range(10, 10 + n)]
    for kind, p in zip(gen.kinds, gen.probabilities):
        freq = kinds.count(kind) / n
        sigma = math.sqrt(p * (1 - p) / n)
        assert abs(freq - p) < 4 * sigma + 1e-9, (kind, freq, p)
    assert set(kinds) <= set(gen.kinds)


def test_custom_weights():
    gen = TerrainGenerator(seed=0, weights={"flat": 1, "gap": 1})
    kinds = {gen.sample_kind(i) for i in range(10, 200)}
    assert kinds == {"flat", "gap"}
    with pytest.raises(ValueError):
        TerrainGenerator(weights={"lava": 1})


@pytest.mark.parametrize("difficulty", list(DIFFICULTY_PRESETS))
def test_chunks_fit_bounds_and_pool(difficulty):
    cfg = ProceduralTerrainConfig()
    gen = TerrainGenerator(seed=3, difficulty=difficulty, cfg=cfg)
    for i in range(2, 300):
        spec = gen.chunk(i)
        assert spec.n_boxes <= cfg.boxes_per_chunk
        assert spec.n_ellipsoids <= cfg.ellipsoids_per_chunk
        for g in spec.geoms:
            x0, x1, y0, y1 = g.xy_extent()
            # ramps' buried underside may poke out a little; the walkable top may not
            assert spec.x0 - 0.3 <= x0 and x1 <= spec.x1 + 0.3, (spec.kind, g)
            assert g.top_z() < 1.5  # nothing taller than the fly's body
        for k in {"slope", "gap", "dip"} & {spec.kind}:
            assert any(lbl == "slope_up" for *_, lbl in spec.segments), k


# ------------------------------------------------------------ with physics
@pytest.fixture(scope="module")
def flat_terrain():
    """All-flat procedural terrain (pool present, no random features)."""
    t = ProceduralTerrain(ProceduralTerrainConfig(weights={"flat": 1.0}))
    sim = Simulation(AppConfig(), world_factory=t.build_world)
    t.attach(sim)
    yield t, sim
    sim.close()


def test_collision_filter_setup(flat_terrain):
    t, sim = flat_terrain
    m = sim.model
    pool = t.geom_ids
    assert len(pool) == t.n_pool_geoms == len(set(pool))
    # pool: contype TERRAIN_BIT (only for opt-in extension geoms, e.g. a whip; it
    # never matches the fly/pool/plane conaffinity), conaffinity FLY_BIT
    assert np.all(m.geom_contype[pool] == TERRAIN_BIT)
    assert np.all(m.geom_conaffinity[pool] == FLY_BIT)
    plane = m.geom("ground_plane").id
    assert m.geom_contype[plane] == TERRAIN_BIT and m.geom_conaffinity[plane] == 0
    fly_contact = np.flatnonzero(m.geom_contype == FLY_BIT)
    assert len(fly_contact) == m.npair  # every fly geom paired with the plane collides
    assert np.all(m.geom_conaffinity[fly_contact] == 0)  # no fly self-collision
    # contact params of the pool == FlyGym's plane pairs
    np.testing.assert_allclose(m.geom_solref[pool[0]], m.pair_solref[0])
    np.testing.assert_allclose(m.geom_solimp[pool[0]], m.pair_solimp[0])
    assert m.geom_margin[pool[0]] == pytest.approx(m.pair_margin[0])


def _walk(sim, steps, every=100):
    xs, zs = [], []
    for _ in range(steps // every):
        sim.step(every)
        p = sim.thorax_position()
        xs.append(p[0])
        zs.append(p[2])
    return np.array(xs), np.array(zs)


def test_fly_stands_on_raised_slab(flat_terrain):
    t, sim = flat_terrain
    sim.reset()
    sim.step(2000)
    x = sim.thorax_position()[0]
    base_z = sim.thorax_position()[2]
    # Place a 0.25 mm high slab ahead via the spawn pool (same code path as spawns).
    gid = t._spawn_boxes[0]
    t._place(gid, slab_box(x + 4.0, x + 30.0, 0.25, 6.0, "blocks"))
    t._refit_bvh()
    xs, zs = _walk(sim, 12000)
    on = xs > x + 8.0
    assert on.sum() > 10, "fly did not walk onto the slab"
    assert np.median(zs[on]) == pytest.approx(base_z + 0.25, abs=0.12)
    t._park(gid)
    t._refit_bvh()


def test_fly_walks_over_spawned_slope_without_falling_through(flat_terrain):
    t, sim = flat_terrain
    sim.reset()
    sim.step(1500)
    base_z = np.mean(_walk(sim, 1000)[1])
    res = t.spawn_ahead("slope", distance=4.0)
    assert res is not None
    top = t.generator.params.slope_height[1]  # spawned hills use the max height
    xs, zs = _walk(sim, 14000)
    lbl = np.array([t.terrain_type_at(x) for x in xs])
    on_top = lbl == "slope_top"
    assert on_top.any(), f"never reached the top (max x {xs.max():.1f}, start {res.x_start:.1f})"
    # thorax rides ~one hill height above its flat-ground level while on top
    assert np.median(zs[on_top]) == pytest.approx(base_z + top, abs=0.15)
    assert zs.min() > 0.8  # never sank through the ramps


def test_spawn_never_intersects_fly(flat_terrain):
    t, sim = flat_terrain
    sim.reset()
    sim.step(500)
    cfg = t.cfg
    fly_bodies = t._fly_body_ids
    for i in range(40):
        kind = SPAWN_KINDS[i % len(SPAWN_KINDS)]
        dist = [-5.0, 0.0, 1.0, 4.0, 7.0][i % 5]
        res = t.spawn_ahead(kind, distance=dist)
        assert res is not None
        thorax = sim.thorax_position()
        assert res.distance >= cfg.min_spawn_distance - 1e-9
        feat = t._spawned[-1]
        for gid in feat.geoms:
            spec_ext = _world_xy_extent(sim.model, gid)
            assert spec_ext[0] >= thorax[0] + cfg.min_spawn_distance - 1e-6
            for b in fly_bodies:
                p = sim.data.xpos[b]
                dx = max(spec_ext[0] - p[0], 0, p[0] - spec_ext[1])
                dy = max(spec_ext[2] - p[1], 0, p[1] - spec_ext[3])
                assert math.hypot(dx, dy) >= 0.5
        sim.step(50)
    assert t.active_geom_count() <= t.n_pool_geoms
    with pytest.raises(ValueError):
        t.spawn_ahead("volcano")


def _world_xy_extent(m, gid):
    R = np.zeros(9)
    mj.mju_quat2Mat(R, m.geom_quat[gid])
    half = np.abs(R.reshape(3, 3)) @ m.geom_size[gid]
    p = m.geom_pos[gid]
    return p[0] - half[0], p[0] + half[0], p[1] - half[1], p[1] + half[1]


def test_flatten_next_chunk_and_labels():
    t = ProceduralTerrain(ProceduralTerrainConfig(weights={"gap": 1.0}, flat_start_chunks=1))
    sim = Simulation(AppConfig(), world_factory=t.build_world)
    t.attach(sim)
    try:
        spec1 = next(c for c in t.loaded_chunks() if c.index == 1)
        assert spec1.kind == "gap" and spec1.geoms
        assert {t.terrain_type_at(x) for x in np.linspace(spec1.x0, spec1.x1, 400)} >= {
            "slope_up", "gap", "slope_down"}
        idx = t.flatten_next_chunk()
        assert idx == 1
        slot = [c.index for c in t._slot_spec].index(1)
        assert t._slot_active[slot] == []
        assert all(t.terrain_type_at(x) == "flat" for x in np.linspace(spec1.x0, spec1.x1, 50))
    finally:
        sim.close()


def test_recycling_constant_pool_window_and_reset():
    cfg = ProceduralTerrainConfig(difficulty="normal", seed=5)
    t = ProceduralTerrain(cfg)
    sim = Simulation(AppConfig(), world_factory=t.build_world)
    t.attach(sim)
    try:
        m = sim.model
        ngeom = m.ngeom
        initial_pos = m.geom_pos[t.geom_ids].copy()
        # Drive the window synthetically far along +x (and back): pure bookkeeping.
        for x in list(np.arange(0.0, 2000.0, 3.0)) + list(np.arange(2000.0, 1900.0, -3.0)):
            t._update_window(float(x))
            k = t.generator.chunk_index_at(float(x))
            loaded = sorted(c.index for c in t.loaded_chunks())
            assert loaded == list(range(k - cfg.chunks_behind, k + cfg.chunks_ahead + 1))
        assert t.recycle_count > 150
        assert m.ngeom == ngeom and t.n_pool_geoms == len(t.geom_ids)
        # Every active geom lies inside a loaded chunk.
        lo = min(c.x0 for c in t.loaded_chunks()) - 0.3
        hi = max(c.x1 for c in t.loaded_chunks()) + 0.3
        active = [g for a in t._slot_active for g in a]
        assert np.all((m.geom_pos[active, 0] > lo) & (m.geom_pos[active, 0] < hi))
        # Reset rebuilds the start layout around the spawn point.
        sim.reset()
        np.testing.assert_allclose(m.geom_pos[t.geom_ids], initial_pos)
    finally:
        sim.close()


def test_multi_recycle_run_no_nans():
    """Real walk over short chunks so several recycles happen quickly."""
    cfg = ProceduralTerrainConfig(difficulty="easy", seed=11, chunk_length=5.0,
                                  flat_start_chunks=1)
    t = ProceduralTerrain(cfg)
    sim = Simulation(AppConfig(), world_factory=t.build_world)
    t.attach(sim)
    try:
        ngeom = sim.model.ngeom
        for _ in range(20):
            sim.step(1000)  # check_stability runs inside step()
            x = sim.thorax_position()[0]
            k = t.generator.chunk_index_at(x)
            assert max(c.index for c in t.loaded_chunks()) == k + cfg.chunks_ahead
        assert np.all(np.isfinite(sim.data.qpos)) and np.all(np.isfinite(sim.data.qvel))
        assert t.recycle_count >= 3
        assert sim.model.ngeom == ngeom
        assert sim.thorax_position()[0] > 15.0
        assert sim.tilt_deg() < 45.0
    finally:
        sim.close()


def test_ground_recentering_moves_the_drawn_plane():
    # Regression: the plane sits at the origin, so MuJoCo compiles it with
    # geom_sameframe = 1 and ignored geom_pos: the checkerboard never moved.
    from types import SimpleNamespace

    from fly_simulator.terrain import GroundRecentering

    m = mj.MjModel.from_xml_string(
        '<mujoco><worldbody><geom name="ground_plane" type="plane" size="10 10 1"/>'
        '<body name="b" pos="0 0 1"><freejoint/><geom size=".1"/></body>'
        '</worldbody></mujoco>')
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    d.qpos[0] = 7.3  # the "fly" is far from the plane centre
    mj.mj_forward(m, d)
    sim = SimpleNamespace(model=m, data=d, thorax_body_id=1, step_count=0)
    GroundRecentering("ground_plane", half_size=10.0, checker_size_mm=1.0)(sim)
    assert m.geom_pos[0, 0] == pytest.approx(8.0)  # whole 2 mm periods
    for _ in range(3):
        mj.mj_step(m, d)
    assert d.geom_xpos[0, 0] == pytest.approx(8.0)  # drawn (and stays) there
