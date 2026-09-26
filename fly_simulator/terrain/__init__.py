"""Terrain / world construction.

* ``flat``: FlyGym's flat ground plane + visual re-centring (default world).
* ``ProceduralTerrain``: endless chunk-based terrain on a pre-allocated geom pool
  (``Simulation(cfg, world_factory=terrain.build_world); terrain.attach(sim)``).
  The pool collides with the fly via contype/conaffinity (explicit pairs would be
  14k and take ~30 s to build); geom_rbound/geom_aabb and the world-body BVH are
  updated whenever geoms move. See the docstring of ``chunks.py``.
"""

from fly_simulator.terrain.chunks import (
    FLY_BIT,
    KIND_COLORS,
    OVERHEAD_KINDS,
    SPAWN_KINDS,
    ProceduralTerrain,
    SpawnResult,
    TERRAIN_BIT,
    TerrainWorld,
)
from fly_simulator.terrain.config import (
    DIFFICULTY_PRESETS,
    TERRAIN_KINDS,
    FeatureParams,
    ProceduralTerrainConfig,
)
from fly_simulator.terrain.flat import GroundRecentering, build_flat_world
from fly_simulator.terrain.generator import ChunkSpec, GeomSpec, TerrainGenerator
from fly_simulator.terrain.sectioned import SectionedGenerator

__all__ = [
    "ChunkSpec",
    "FLY_BIT",
    "DIFFICULTY_PRESETS",
    "FeatureParams",
    "GeomSpec",
    "GroundRecentering",
    "KIND_COLORS",
    "OVERHEAD_KINDS",
    "ProceduralTerrain",
    "ProceduralTerrainConfig",
    "SPAWN_KINDS",
    "SectionedGenerator",
    "SpawnResult",
    "TERRAIN_BIT",
    "TERRAIN_KINDS",
    "TerrainGenerator",
    "TerrainWorld",
    "build_flat_world",
]
