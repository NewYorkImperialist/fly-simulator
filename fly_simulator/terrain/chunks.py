"""Endless chunked terrain on a pre-allocated MuJoCo geom pool.

How it works (see docs/dev/API_NOTES.md sections 7-8 for the verified MuJoCo facts):

* **Contacts (why not explicit pairs).** FlyGym gives every geom ``contype =
  conaffinity = 0`` and creates an explicit ``<pair>`` per (fly contact geom x
  ground geom) in ``add_fly``. Geoms cannot be added after compilation, so the whole
  pool must exist up front, but ~260 pooled geoms x 55 fly geoms = 14k pairs, and
  ``MjSpec.add_pair`` is quadratic: building took ~30 s (measured). Instead
  ``TerrainWorld`` keeps FlyGym's explicit pairs only for the base plane and uses
  MuJoCo's *dynamic* collision filter for the pool: fly contact geoms get
  ``contype = FLY_BIT, conaffinity = 0``, pooled geoms ``contype = TERRAIN_BIT,
  conaffinity = FLY_BIT``, the plane ``contype = TERRAIN_BIT, conaffinity = 0`` ->
  fly<->pool collide, fly<->fly and pool<->pool/plane never do (see FLY_BIT below;
  TERRAIN_BIT only matters for extra geoms that opt in, e.g. a whip). Pooled
  geoms get ``priority = 1`` so their condim/friction/solref/solimp (copied from
  FlyGym's ``ContactParams``) are used verbatim for those contacts; margin is the
  max of both geoms = ContactParams.margin. So contacts are identical in parameters
  to FlyGym's ground pairs. Pooled geoms are still listed in ``world.ground_geoms``
  so FlyGym's ``ground_only`` contact queries (and the hybrid controller's
  stumbling detection) see them.
* **Recycling.** At runtime a geom is re-used by writing ``model.geom_pos/quat/size/
  rgba``. Collision uses ``model.geom_rbound`` (bounding sphere) and
  ``model.geom_aabb`` (local bounding box); both are compile-time caches of the size
  and are *not* refreshed automatically, so they are updated on every resize (a
  stale rbound silently drops contacts).
* **World-body BVH refit.** Dynamic collisions go through the midphase, which uses a
  bounding-volume hierarchy over the world body's geoms (``model.bvh_aabb``) built
  at compile time from the *compile-time* geom positions. Moving a world geom
  without refitting it makes contacts silently disappear (verified: a box moved
  under a falling body was ignored). After every layout change the leaf AABBs are
  recomputed from geom pose/aabb and internal nodes are refit bottom-up (children
  always have larger indices than their parent). Tree topology is kept; only
  efficiency, not correctness, depends on it.
* **Unused geoms** are parked tiny, transparent and 200 mm below the ground.
* **Continuous floor.** The infinite base plane (z = 0) stays under everything; all
  features sit on top of it. Gaps and dips are slots in a raised plateau (reached
  by ramps), so their bottom is the base plane: depth = plateau height and nothing
  is ever bottomless. The plane is re-centred under the fly (visual only).

Chunk ``i`` spans ``[(i - 0.5) L, (i + 0.5) L)``. The manager keeps chunks
``k - chunks_behind .. k + chunks_ahead`` loaded (``k`` = chunk under the thorax);
whenever ``k`` changes, slots that fell out of that window are re-filled with the
missing chunk indices (content is a pure function of seed + index). The geom count
never changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco as mj
import numpy as np
from flygym.compose import FlatGroundWorld

from fly_simulator.terrain.config import ProceduralTerrainConfig
from fly_simulator.terrain.flat import GroundRecentering
from fly_simulator.terrain.generator import ChunkSpec, GeomSpec, TerrainGenerator, _rotmat

if TYPE_CHECKING:
    from flygym.compose import BaseWorld

    from fly_simulator.simulation import Simulation

PARK_POS = (0.0, 0.0, -200.0)
PARK_SIZE = 0.01

# Colours per terrain kind (rgba). The base plane is a grey checkerboard.
KIND_COLORS: dict[str, tuple[float, float, float, float]] = {
    "bumps": (0.62, 0.52, 0.36, 1.0),   # sandy mounds
    "rough": (0.45, 0.36, 0.26, 1.0),   # dark soil lumps
    "rocks": (0.55, 0.56, 0.60, 1.0),   # grey stones
    "blocks": (0.30, 0.42, 0.62, 1.0),  # blue blocks
    "slope": (0.36, 0.55, 0.32, 1.0),   # green hills
    "gap": (0.72, 0.44, 0.22, 1.0),     # orange plateau around a gap
    "dip": (0.60, 0.40, 0.55, 1.0),     # mauve plateau around a dip
    # Obstacle-course pieces (fly_simulator/course); never produced by endless mode.
    "stairs": (0.42, 0.46, 0.66, 1.0),  # slate-blue steps
    "pillar": (0.80, 0.52, 0.16, 1.0),  # orange slalom pillars
    "ceiling": (0.55, 0.75, 0.90, 0.35),  # translucent tunnel roof (camera sees through)
    "tunnel_wall": (0.50, 0.68, 0.82, 0.55),
    "start": (0.20, 0.75, 0.30, 1.0),  # green start gate / line
    "checkpoint": (0.95, 0.80, 0.15, 1.0),  # yellow checkpoint gates
    "finish": (0.90, 0.18, 0.18, 1.0),  # red finish gate
    "gate_bar": (0.95, 0.95, 0.95, 1.0),  # crossbar over a gate
    "whip_zone": (0.78, 0.30, 0.30, 1.0),  # carpet marking a whip gauntlet
    "loom_zone": (0.48, 0.30, 0.72, 1.0),  # carpet marking a looming zone
}
# Geoms above the walking surface (a tunnel roof, gate crossbars): ignored by
# ground_height_at(), so the fall detector never takes a roof for the floor.
OVERHEAD_KINDS = frozenset({"ceiling", "gate_bar"})
SPAWN_TINT = 1.2  # interactively spawned features are drawn slightly brighter

SPAWN_KINDS = ("rock", "bump", "slope", "gap", "dip")


@dataclass
class SpawnedFeature:
    kind: str
    x0: float
    x1: float
    geoms: list[int]
    segments: list[tuple[float, float, str]] = field(default_factory=list)


@dataclass
class SpawnResult:
    kind: str
    x_start: float  # near edge of the feature (world x, mm)
    x_end: float
    distance: float  # x_start - thorax x at spawn time


_DOWN = np.array([0.0, 0.0, -1.0])

# Dynamic-collision bits (contype/conaffinity). Two geoms collide if
# (contype1 & conaffinity2) or (contype2 & conaffinity1).
#   fly contact geoms:  contype = FLY_BIT,     conaffinity = 0
#   pooled terrain:     contype = TERRAIN_BIT, conaffinity = FLY_BIT
#   base plane:         contype = TERRAIN_BIT, conaffinity = 0  (+ FlyGym's explicit
#                       pairs with the fly geoms)
# -> fly<->pool collide; fly<->fly, pool<->pool, pool<->plane, fly<->plane (dynamic)
# never do. A world extension geom with contype = 0 and conaffinity = FLY_BIT hits the
# fly only; conaffinity = FLY_BIT | TERRAIN_BIT also hits the plane and the terrain.
FLY_BIT = 1 << 3
TERRAIN_BIT = 1 << 4


class TerrainWorld(FlatGroundWorld):
    """FlyGym flat ground + a pool of terrain geoms colliding via contype/conaffinity.

    Overrides FlyGym's (private) ``_set_ground_contact`` so explicit pairs are only
    made with the base plane; see the module docstring for why.
    """

    def __init__(self, half_size: float = 1000.0, checker_size_mm: float = 2.0) -> None:
        super().__init__(half_size=half_size)
        repeat = (2 * half_size) / (2 * checker_size_mm)  # same as build_flat_world
        self.mjcf_root.material("grid").texrepeat = [repeat, repeat]
        self.pool_geoms: list = []

    def add_pool_geom(self, name: str, gtype) -> None:
        g = self.mjcf_root.worldbody.add_geom(
            type=gtype, name=name, size=(PARK_SIZE,) * 3, pos=PARK_POS,
            rgba=(0.5, 0.5, 0.5, 0.0), contype=0, conaffinity=0,
        )
        self.pool_geoms.append(g)
        self.ground_geoms.append(g)  # for FlyGym's ground-contact queries only

    def _set_ground_contact(self, fly, bodysegs_with_ground_contact, ground_contact_params):
        pool = self.pool_geoms
        self.ground_geoms = [self.ground_geom]  # explicit pairs with the plane only
        try:
            super()._set_ground_contact(fly, bodysegs_with_ground_contact, ground_contact_params)
        finally:
            self.ground_geoms = [self.ground_geom, *pool]
        cp = ground_contact_params
        for seg in bodysegs_with_ground_contact:
            for body_geom in fly.bodyseg_to_mjcfgeom[seg]:
                g = self.mjcf_root.geom(body_geom.name) or body_geom
                g.contype, g.conaffinity = FLY_BIT, 0
        plane = self.mjcf_root.geom(self.ground_geom.name) or self.ground_geom
        plane.contype, plane.conaffinity = TERRAIN_BIT, 0
        for g in pool:
            g.contype, g.conaffinity = TERRAIN_BIT, FLY_BIT
            g.priority = 1  # pool geom's contact params win over the fly geom's defaults
            g.condim = 3  # FlyGym's pairs are condim 3
            g.friction = (cp.sliding_friction, cp.torsional_friction, cp.rolling_friction)
            g.solref = cp.get_solref_tuple()
            g.solimp = cp.get_solimp_tuple()
            g.margin = cp.margin
            g.gap = 0.0


class ProceduralTerrain:
    """Endless procedural terrain for ``fly_simulator.simulation.Simulation``.

    Usage::

        terrain = ProceduralTerrain(ProceduralTerrainConfig(difficulty="normal", seed=42))
        sim = Simulation(cfg, world_factory=terrain.build_world)
        terrain.attach(sim)          # registers hooks + lays out chunks
        sim.step(...)
        terrain.spawn_ahead("rock"); terrain.flatten_next_chunk()
        terrain.terrain_type_at(sim.thorax_position()[0])
    """

    def __init__(self, cfg: ProceduralTerrainConfig | None = None, *,
                 seed: int | None = None, difficulty: str | None = None) -> None:
        self.cfg = cfg or ProceduralTerrainConfig()
        if seed is not None:
            self.cfg.seed = seed
        if difficulty is not None:
            self.cfg.difficulty = difficulty
        self.generator = TerrainGenerator(self.cfg.seed, self.cfg.difficulty, cfg=self.cfg,
                                          weights=self.cfg.weights)
        self.n_slots = self.cfg.chunks_behind + self.cfg.chunks_ahead + 1
        self.sim: "Simulation | None" = None
        self.recycle_count = 0
        self.spawn_count = 0
        self._world = None
        self._slot_boxes: list[list[int]] = []
        self._slot_ells: list[list[int]] = []
        self._slot_index: list[int | None] = [None] * self.n_slots
        self._slot_spec: list[ChunkSpec | None] = [None] * self.n_slots
        self._slot_active: list[list[int]] = [[] for _ in range(self.n_slots)]
        self._spawn_boxes: list[int] = []
        self._spawn_ells: list[int] = []
        self._spawned: list[SpawnedFeature] = []
        self._overrides: dict[int, str] = {}
        self._center: int | None = None
        self._fly_body_ids: np.ndarray | None = None
        self._geom_specs: dict[int, GeomSpec] = {}  # current layout of chunk geoms
        self._spawn_free: dict[str, list[int]] = {"box": [], "ellipsoid": []}
        self._dirty = False
        # Pool geoms currently placed with an OVERHEAD_KINDS kind (course mode only;
        # always empty in endless mode, so ground_height_at is unchanged there).
        self._overhead: set[int] = set()

    # ------------------------------------------------------------ construction
    def pool_names(self) -> tuple[list[str], list[str]]:
        boxes = [f"terrain_c{s}_box{j}" for s in range(self.n_slots)
                 for j in range(self.cfg.boxes_per_chunk)]
        boxes += [f"terrain_spawn_box{j}" for j in range(self.cfg.spawn_boxes)]
        ells = [f"terrain_c{s}_ell{j}" for s in range(self.n_slots)
                for j in range(self.cfg.ellipsoids_per_chunk)]
        ells += [f"terrain_spawn_ell{j}" for j in range(self.cfg.spawn_ellipsoids)]
        return boxes, ells

    def build_world(self) -> "BaseWorld":
        """World factory for ``Simulation(cfg, world_factory=terrain.build_world)``."""
        boxes, ells = self.pool_names()
        world = TerrainWorld(self.cfg.ground_half_size, self.cfg.checker_size_mm)
        for names, gtype in ((boxes, mj.mjtGeom.mjGEOM_BOX),
                             (ells, mj.mjtGeom.mjGEOM_ELLIPSOID)):
            for name in names:
                world.add_pool_geom(name, gtype)
        self._world = world
        return world

    def attach(self, sim: "Simulation") -> None:
        """Bind to a compiled Simulation built with ``world_factory=self.build_world``."""
        if self._world is None or sim.world is not self._world:
            raise RuntimeError("Simulation must be built with world_factory=terrain.build_world")
        self.sim = sim
        m = sim.model
        gid = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        B, E = self.cfg.boxes_per_chunk, self.cfg.ellipsoids_per_chunk
        self._slot_boxes = [[gid(f"terrain_c{s}_box{j}") for j in range(B)]
                            for s in range(self.n_slots)]
        self._slot_ells = [[gid(f"terrain_c{s}_ell{j}") for j in range(E)]
                           for s in range(self.n_slots)]
        self._spawn_boxes = [gid(f"terrain_spawn_box{j}") for j in range(self.cfg.spawn_boxes)]
        self._spawn_ells = [gid(f"terrain_spawn_ell{j}") for j in range(self.cfg.spawn_ellipsoids)]
        self._spawn_free = {"box": list(self._spawn_boxes), "ellipsoid": list(self._spawn_ells)}
        self._ground_geom = gid("ground_plane")
        # All bodies of the fly (thorax subtree) for the spawn-safety check.
        root = sim.thorax_body_id
        self._fly_body_ids = np.array(
            [b for b in range(m.nbody) if b == root or m.body_rootid[b] == root]
        )
        self._pool_ids = np.array(self.geom_ids, dtype=int)
        self._init_bvh()

        # Simulation only installs plane re-centring for its default world.
        sim.post_step_hooks.append(
            GroundRecentering("ground_plane", self.cfg.ground_half_size, self.cfg.checker_size_mm)
        )
        sim.post_step_hooks.append(self._post_step)
        sim.reset_hooks.append(self._post_reset)
        # The layout must be rebuilt *before* the reset warmup so the fly never
        # settles onto stale features: Simulation runs pre_reset_hooks before
        # FlyGym's keyframe reset.
        sim.pre_reset_hooks.append(lambda s: self._pre_reset())
        self.relayout(self._fly_x(), self._fly_y())

    # --------------------------------------------------------------- queries
    @property
    def geom_ids(self) -> list[int]:
        ids = [g for s in self._slot_boxes for g in s] + [g for s in self._slot_ells for g in s]
        return ids + self._spawn_boxes + self._spawn_ells

    @property
    def n_pool_geoms(self) -> int:
        return len(self.geom_ids)

    def loaded_chunks(self) -> list[ChunkSpec]:
        return sorted((s for s in self._slot_spec if s is not None), key=lambda c: c.index)

    def active_geom_count(self) -> int:
        return sum(len(a) for a in self._slot_active) + sum(len(f.geoms) for f in self._spawned)

    def terrain_type_at(self, x: float) -> str:
        """Terrain label at world x: flat, bumps, rough, rocks, blocks, slope_up,
        slope_top, slope_down, gap, dip (interactive spawns take precedence)."""
        for f in reversed(self._spawned):
            for a, b, label in f.segments:
                if a <= x < b:
                    return label
        idx = self.generator.chunk_index_at(x)
        for spec in self._slot_spec:
            if spec is not None and spec.index == idx:
                return spec.label_at(x)
        return self._make_chunk(idx).label_at(x)

    def ground_height_at(self, x: float, y: float) -> float:
        """Height (mm) of the walkable surface at (x, y): the highest pooled geom
        under that point, or the base plane (0). Exact (ray cast against the live
        geom poses with ``mju_rayGeom``); cost ~10 us + a few us per geom whose
        bounding sphere covers the point. Used by the fall detector and logger."""
        if self.sim is None:
            return 0.0
        m = self.sim.model
        ids = self._pool_ids
        pos = m.geom_pos[ids]
        d2 = (pos[:, 0] - x) ** 2 + (pos[:, 1] - y) ** 2
        cand = ids[(d2 <= m.geom_rbound[ids] ** 2) & (pos[:, 2] > -50.0)]
        if self._overhead:
            cand = np.array([g for g in cand if g not in self._overhead], dtype=int)
        if cand.size == 0:
            return 0.0
        top = 10.0  # mm; well above any feature (features are < 1.5 mm tall)
        pnt = np.array([x, y, top])
        best = 0.0
        mat = np.empty(9)
        for g in cand:
            mj.mju_quat2Mat(mat, m.geom_quat[g])
            dist = mj.mju_rayGeom(m.geom_pos[g], mat, m.geom_size[g], pnt, _DOWN,
                                  int(m.geom_type[g]))
            if dist >= 0.0:
                best = max(best, top - dist)
        return float(best)

    def chunk_kind_at(self, x: float) -> str:
        idx = self.generator.chunk_index_at(x)
        for spec in self._slot_spec:
            if spec is not None and spec.index == idx:
                return spec.kind
        return self._make_chunk(idx).kind

    # ---------------------------------------------------------------- layout
    def _fly_x(self) -> float:
        return float(self.sim.data.xpos[self.sim.thorax_body_id, 0])

    def _make_chunk(self, index: int) -> ChunkSpec:
        return self.generator.chunk(index, kind=self._overrides.get(index))

    def _fly_y(self) -> float:
        return float(self.sim.data.xpos[self.sim.thorax_body_id, 1])

    def relayout(self, fly_x: float, fly_y: float = 0.0) -> None:
        """Park everything and load the chunk window around ``fly_x``."""
        for s in range(self.n_slots):
            self._load_slot(s, None)
        for f in self._spawned:
            self._free_feature(f)
        self._spawned.clear()
        self._spawn_free = {"box": list(self._spawn_boxes), "ellipsoid": list(self._spawn_ells)}
        self._center = None
        self._update_window(fly_x, fly_y, count=False)

    def _pre_reset(self) -> None:
        # Fresh episode: same seeded world, spawns and "flatten" overrides dropped.
        self._overrides.clear()
        self.relayout(0.0)  # the fly respawns at x = 0

    def _post_reset(self, sim) -> None:
        self._update_window(self._fly_x(), self._fly_y(), count=False)

    def _post_step(self, sim) -> None:
        if sim.step_count % self.cfg.update_every_steps:
            return
        self._update_window(self._fly_x(), self._fly_y())

    def _update_window(self, fly_x: float, fly_y: float = 0.0, count: bool = True) -> None:
        k = self.generator.chunk_index_at(fly_x)
        if k == self._center:
            return
        self._center = k
        want = set(range(k - self.cfg.chunks_behind, k + self.cfg.chunks_ahead + 1))
        have = {idx for idx in self._slot_index if idx is not None}
        free = [s for s, idx in enumerate(self._slot_index) if idx is None or idx not in want]
        # Nearest missing chunks first (only matters on the very first layout).
        for idx in sorted(want - have, key=lambda i: abs(i - k)):
            s = free.pop()
            recycled = self._slot_index[s] is not None
            self._load_slot(s, idx, fly_y)
            if count and recycled:
                self.recycle_count += 1
        # Interactive spawns far behind the fly are released.
        behind = fly_x - self.cfg.chunks_behind * self.cfg.chunk_length
        for f in [f for f in self._spawned if f.x1 < behind]:
            self._free_feature(f)
            self._spawned.remove(f)
        self._refit_bvh()

    def _load_slot(self, s: int, index: int | None, fly_y: float = 0.0) -> None:
        for g in self._slot_active[s]:
            self._park(g)
        self._slot_active[s] = []
        self._slot_index[s] = index
        self._slot_spec[s] = None
        if index is None:
            return
        spec = self._make_chunk(index)
        # Lateral re-centring: the heading hold keeps the fly's *heading* along +x,
        # but stumbles can shift it sideways by 10+ mm over time. New chunks (loaded
        # several chunks ahead) are centred on the fly's current y, snapped to
        # lateral_snap mm, so features stay in its path. Content is still a pure
        # function of (seed, index, offset); after reset the offset is 0 again.
        snap = self.cfg.lateral_snap
        dy = round(fly_y / snap) * snap if snap > 0 else 0.0
        if dy:
            spec.shift_y(dy)
        boxes, ells = iter(self._slot_boxes[s]), iter(self._slot_ells[s])
        for g in spec.geoms:
            gid = next(boxes) if g.shape == "box" else next(ells)
            self._place(gid, g)
            self._slot_active[s].append(gid)
        self._slot_spec[s] = spec
        for gid, g in zip(self._slot_active[s], spec.geoms):
            self._geom_specs[gid] = g

    # ------------------------------------------------------------- BVH refit
    def _init_bvh(self) -> None:
        m = self.sim.model
        a, n = int(m.body_bvhadr[0]), int(m.body_bvhnum[0])
        nodes = np.arange(a, a + n)
        pool = set(self.geom_ids)
        node_geom = m.bvh_nodeid[nodes]
        self._bvh_leaves = np.array([i for i, g in zip(nodes, node_geom) if g in pool], dtype=int)
        self._bvh_leaf_geoms = m.bvh_nodeid[self._bvh_leaves].astype(int)
        # Internal nodes children-first (child index > parent index in MuJoCo's BVH).
        self._bvh_internal = [int(i) for i in nodes[::-1] if m.bvh_nodeid[i] < 0]
        self._dirty = True

    def _refit_bvh(self) -> None:
        """Refit the world body's BVH after moving pooled geoms (see module doc)."""
        if not self._dirty:
            return
        m = self.sim.model
        g = self._bvh_leaf_geoms
        q = m.geom_quat[g]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = np.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ], axis=1).reshape(-1, 3, 3)
        aabb = m.geom_aabb[g]
        center = m.geom_pos[g] + np.einsum("nij,nj->ni", R, aabb[:, :3])
        half = np.einsum("nij,nj->ni", np.abs(R), aabb[:, 3:])
        bvh = m.bvh_aabb
        bvh[self._bvh_leaves, :3] = center
        bvh[self._bvh_leaves, 3:] = half
        child = m.bvh_child
        for i in self._bvh_internal:
            c1, c2 = child[i]
            b1, b2 = bvh[c1], bvh[c2]
            lo = np.minimum(b1[:3] - b1[3:], b2[:3] - b2[3:])
            hi = np.maximum(b1[:3] + b1[3:], b2[:3] + b2[3:])
            bvh[i, :3] = (lo + hi) / 2
            bvh[i, 3:] = (hi - lo) / 2
        self._dirty = False

    # ---------------------------------------------------------- geom writes
    def _place(self, gid: int, g: GeomSpec, tint: float = 1.0) -> None:
        m, d = self.sim.model, self.sim.data
        self._dirty = True
        if g.kind in OVERHEAD_KINDS:
            self._overhead.add(gid)
        elif self._overhead:
            self._overhead.discard(gid)
        size = np.asarray(g.size, dtype=float)
        m.geom_pos[gid] = g.pos
        m.geom_quat[gid] = g.quat
        m.geom_size[gid] = size
        # rbound / aabb are compile-time caches of the geom's extent that collision
        # detection relies on (bounding-sphere test before narrow phase); they must
        # follow the new size or contacts get missed.
        m.geom_rbound[gid] = float(np.linalg.norm(size)) if g.shape == "box" else float(size.max())
        m.geom_aabb[gid] = (0.0, 0.0, 0.0, *size)
        r, gg, b, a = KIND_COLORS.get(g.kind, (0.5, 0.5, 0.5, 1.0))
        # Small deterministic brightness jitter so neighbouring pieces are distinguishable.
        jitter = 0.92 + 0.16 * ((gid * 2654435761) % 1000) / 1000.0
        c = np.clip(np.array([r, gg, b]) * tint * jitter, 0, 1)
        m.geom_rgba[gid] = (*c, a)
        # World-body geoms get geom_xpos/xmat from mj_kinematics at the next step;
        # set them now too so a render before the next step already shows the change.
        d.geom_xpos[gid] = g.pos
        d.geom_xmat[gid] = _rotmat(g.quat).ravel()

    def _park(self, gid: int) -> None:
        m, d = self.sim.model, self.sim.data
        self._dirty = True
        if self._overhead:
            self._overhead.discard(gid)
        m.geom_pos[gid] = PARK_POS
        m.geom_quat[gid] = (1.0, 0.0, 0.0, 0.0)
        m.geom_size[gid] = PARK_SIZE
        m.geom_rbound[gid] = PARK_SIZE * math.sqrt(3)
        m.geom_aabb[gid] = (0.0, 0.0, 0.0, PARK_SIZE, PARK_SIZE, PARK_SIZE)
        m.geom_rgba[gid, 3] = 0.0
        d.geom_xpos[gid] = PARK_POS
        d.geom_xmat[gid] = (1, 0, 0, 0, 1, 0, 0, 0, 1)

    # ----------------------------------------------------------- spawn API
    def _fly_points_xy(self) -> np.ndarray:
        return self.sim.data.xpos[self._fly_body_ids, :2]

    def _is_safe(self, xmin: float, xmax: float, ymin: float, ymax: float) -> bool:
        """True if the xy box is clear of the thorax (fly_clearance) and of every fly
        body (0.5 mm), i.e. no part of the fly's leg span can intersect it."""
        def dist(p):
            dx = max(xmin - p[0], 0.0, p[0] - xmax)
            dy = max(ymin - p[1], 0.0, p[1] - ymax)
            return math.hypot(dx, dy)

        thorax = self.sim.data.xpos[self.sim.thorax_body_id, :2]
        if dist(thorax) < self.cfg.fly_clearance:
            return False
        return all(dist(p) >= 0.5 for p in self._fly_points_xy())

    def spawn_ahead(self, kind: str, distance: float | None = None) -> SpawnResult | None:
        """Place a feature (rock, bump, slope, gap, dip) ``distance`` mm ahead (+x)
        of the thorax. ``distance`` is clamped to ``min_spawn_distance`` and pushed
        further out if any part would come near the fly. Full-width features
        (slope/gap/dip) replace chunk geometry in their footprint (except pieces
        near the fly). Returns None if the spawn pool is exhausted."""
        if kind not in SPAWN_KINDS:
            raise ValueError(f"unknown spawn kind {kind!r}; choose from {SPAWN_KINDS}")
        if self.sim is None:
            raise RuntimeError("attach(sim) first")
        fly_x, fly_y = self._fly_x(), self._fly_y()
        d = max(self.cfg.default_spawn_distance if distance is None else float(distance),
                self.cfg.min_spawn_distance)
        # Reproducible per spawn; every retry redraws the same shape, only shifted.
        seed_seq = np.random.SeedSequence([self.cfg.seed, 7919, self.spawn_count])
        for _ in range(100):
            geoms, segments = self.generator.feature(
                kind, fly_x + d, np.random.default_rng(seed_seq),
                len(self._spawn_boxes), len(self._spawn_ells))
            for g in geoms:  # centre on the fly's lateral position
                g.pos = (g.pos[0], g.pos[1] + fly_y, g.pos[2])
            ext = np.array([g.xy_extent() for g in geoms])
            box = (ext[:, 0].min(), ext[:, 1].max(), ext[:, 2].min(), ext[:, 3].max())
            if box[0] >= fly_x + self.cfg.min_spawn_distance - 1e-9 and self._is_safe(*box):
                break
            d += 0.5
        else:
            return None

        need = {"box": sum(g.shape == "box" for g in geoms),
                "ellipsoid": sum(g.shape == "ellipsoid" for g in geoms)}
        # Evict the oldest spawned features that are clear of the fly if needed.
        for f in list(self._spawned):
            if all(len(self._spawn_free[s]) >= n for s, n in need.items()):
                break
            if self._is_safe(f.x0, f.x1, -1e3, 1e3) or f.x1 < fly_x - self.cfg.fly_clearance:
                self._free_feature(f)
                self._spawned.remove(f)
        if not all(len(self._spawn_free[s]) >= n for s, n in need.items()):
            return None

        if kind in ("slope", "gap", "dip"):
            self._clear_region(box[0] - 0.1, box[1] + 0.1)
        ids = []
        for g in geoms:
            gid = self._spawn_free[g.shape].pop(0)
            self._place(gid, g, tint=SPAWN_TINT)
            ids.append(gid)
        self._spawned.append(SpawnedFeature(kind, box[0], box[1], ids, segments))
        self.spawn_count += 1
        self._refit_bvh()
        return SpawnResult(kind, float(box[0]), float(box[1]), float(box[0] - fly_x))

    def _free_feature(self, f: SpawnedFeature) -> None:
        for gid in f.geoms:
            self._park(gid)
            self._spawn_free["box" if gid in self._spawn_boxes else "ellipsoid"].append(gid)
        f.geoms = []

    def _clear_region(self, x0: float, x1: float) -> int:
        """Park chunk/spawn geoms overlapping [x0, x1] that are safely away from the fly."""
        n = 0
        specs = self._geom_specs
        for s in range(self.n_slots):
            keep = []
            for gid in self._slot_active[s]:
                ext = specs[gid].xy_extent()
                if ext[1] >= x0 and ext[0] <= x1 and self._is_safe(*ext):
                    self._park(gid)
                    n += 1
                else:
                    keep.append(gid)
            self._slot_active[s] = keep
            spec = self._slot_spec[s]
            if spec is not None and not keep:
                spec.kind, spec.segments = "flat", []
        for f in [f for f in self._spawned if f.x1 >= x0 and f.x0 <= x1]:
            if self._is_safe(f.x0, f.x1, -1e3, 1e3):
                self._free_feature(f)
                self._spawned.remove(f)
                n += 1
        return n

    def set_difficulty(self, difficulty: str, reload_from_ahead: int | None = 2) -> list[int]:
        """Switch the difficulty preset at runtime (no model rebuild).

        Chunks generated from now on use the new preset (and its probability table:
        a ``weights`` override is dropped). Loaded chunks at least
        ``reload_from_ahead`` chunks ahead of the fly's chunk are regenerated right
        away (None = only chunks loaded later change); the fly's own chunk and the
        next one are never touched, so nothing changes under its feet. "flatten"
        overrides and spawned obstacles stay. A reset keeps the new difficulty.
        Returns the regenerated chunk indices.
        """
        from dataclasses import replace

        replace(self.cfg, difficulty=difficulty).preset()  # validates the name
        self.cfg.difficulty = difficulty.lower()
        self.cfg.weights = None
        self.generator = TerrainGenerator(self.cfg.seed, self.cfg.difficulty, cfg=self.cfg)
        reloaded: list[int] = []
        if self.sim is None or reload_from_ahead is None:
            return reloaded
        k = self.generator.chunk_index_at(self._fly_x())
        fly_y = self._fly_y()
        for s, idx in enumerate(self._slot_index):
            if idx is not None and idx >= k + reload_from_ahead:
                self._load_slot(s, idx, fly_y)
                reloaded.append(idx)
        self._refit_bvh()
        return sorted(reloaded)

    def flatten_next_chunk(self) -> int:
        """Make the chunk after the fly's current one flat (pieces within the fly's
        clearance are kept). Persists until reset. Returns the chunk index."""
        if self.sim is None:
            raise RuntimeError("attach(sim) first")
        idx = self.generator.chunk_index_at(self._fly_x()) + 1
        self._overrides[idx] = "flat"
        x0, x1 = self.generator.chunk_bounds(idx)
        self._clear_region(x0, x1 - 1e-6)
        for spec in self._slot_spec:
            if spec is not None and spec.index == idx:
                spec.kind, spec.segments = "flat", []
        self._refit_bvh()
        return idx
