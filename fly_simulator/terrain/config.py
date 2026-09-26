"""Configuration for the endless procedural terrain (dataclasses, JSON-friendly).

Units are model units: mm, degrees for angles. Sizes are chosen for a fruit fly:
thorax ~1.1 mm above the ground while walking, legs span ~3-4 mm (tarsi reach
~2 mm in front/behind and ~1.5 mm to the side of the thorax), walking speed
~14 mm/s. Reference sizes that FlyGym's hybrid controller is known to cope with
(``flygym.compose.world.complex_terrain``): blocks 1.3 mm wide with 0.35 mm height
steps, gaps 0.3 mm wide between 1 mm blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

Range = tuple[float, float]

TERRAIN_KINDS = ("flat", "bumps", "rough", "rocks", "blocks", "dip", "slope", "gap")


@dataclass
class FeatureParams:
    """Size ranges (uniformly sampled) for each terrain kind at one difficulty.

    Heights are measured above the base plane (z = 0). Every raised feature sits on
    that plane, so the floor is continuous everywhere (see ``chunks.py``).
    """

    # Broad, gentle ellipsoidal mounds.
    bump_height: Range = (0.08, 0.2)
    bump_radius: Range = (1.0, 2.5)  # horizontal semi-axes
    bump_count: tuple[int, int] = (2, 4)
    # Dense small lumps ("rough ground").
    rough_height: Range = (0.03, 0.07)
    rough_radius: Range = (0.3, 0.7)
    rough_count: tuple[int, int] = (14, 22)
    # Isolated rocks: compact ellipsoids, randomly oriented.
    rock_height: Range = (0.1, 0.22)
    rock_radius: Range = (0.2, 0.45)
    rock_count: tuple[int, int] = (2, 5)
    # Blocks: yaw-rotated boxes.
    block_height: Range = (0.06, 0.16)
    block_size: Range = (0.8, 1.5)  # full edge length in x/y
    block_count: tuple[int, int] = (3, 7)
    # Mild slopes: a trapezoidal hill (ramp up, flat top, ramp down).
    slope_angle_deg: Range = (4.0, 8.0)
    slope_height: Range = (0.2, 0.4)
    slope_top_length: Range = (1.0, 3.0)
    # Gaps and dips are slots cut into a raised plateau reached via ramps; the base
    # plane is the bottom, so depth == plateau height (never a bottomless hole).
    plateau_ramp_angle_deg: float = 6.0
    plateau_length: Range = (2.0, 3.0)  # flat plateau part before/after the slot
    gap_width: Range = (0.15, 0.3)
    gap_depth: Range = (0.2, 0.3)
    dip_width: Range = (1.5, 3.0)
    dip_depth: Range = (0.12, 0.2)


@dataclass
class DifficultyPreset:
    weights: dict[str, float]
    params: FeatureParams


def _scaled(p: FeatureParams, height: float, count: float = 1.0, angle: float = 1.0,
            width: float = 1.0) -> FeatureParams:
    def s(r: Range, k: float) -> Range:
        return (r[0] * k, r[1] * k)

    def c(r: tuple[int, int], k: float) -> tuple[int, int]:
        return (max(1, round(r[0] * k)), max(1, round(r[1] * k)))

    return replace(
        p,
        bump_height=s(p.bump_height, height), bump_count=c(p.bump_count, count),
        rough_height=s(p.rough_height, height), rough_count=c(p.rough_count, count),
        rock_height=s(p.rock_height, height), rock_radius=s(p.rock_radius, width),
        rock_count=c(p.rock_count, count),
        block_height=s(p.block_height, height), block_count=c(p.block_count, count),
        slope_angle_deg=s(p.slope_angle_deg, angle), slope_height=s(p.slope_height, height),
        gap_width=s(p.gap_width, width), gap_depth=s(p.gap_depth, height),
        dip_width=s(p.dip_width, width), dip_depth=s(p.dip_depth, height),
    )


_NORMAL = FeatureParams()

DIFFICULTY_PRESETS: dict[str, DifficultyPreset] = {
    # Everything flat (the geom pool still exists, so interactive spawns work).
    "flat": DifficultyPreset(weights={"flat": 1}, params=_NORMAL),
    # Mostly flat with low, gentle features.
    "easy": DifficultyPreset(
        weights={"flat": 70, "bumps": 15, "rough": 10, "rocks": 5},
        params=_scaled(_NORMAL, height=0.6, count=0.7, width=0.8),
    ),
    "normal": DifficultyPreset(
        weights={"flat": 57, "bumps": 13, "rough": 5, "rocks": 10, "blocks": 5,
                 "dip": 2, "slope": 6, "gap": 2},
        params=_NORMAL,
    ),
    "hard": DifficultyPreset(
        weights={"flat": 35, "bumps": 15, "rough": 10, "rocks": 12, "blocks": 10,
                 "dip": 5, "slope": 8, "gap": 5},
        params=_scaled(_NORMAL, height=1.5, count=1.3, angle=1.4, width=1.2),
    ),
    "chaos": DifficultyPreset(
        weights={"flat": 10, "bumps": 15, "rough": 15, "rocks": 15, "blocks": 15,
                 "dip": 8, "slope": 12, "gap": 10},
        params=_scaled(_NORMAL, height=2.2, count=1.6, angle=1.8, width=1.5),
    ),
}


@dataclass
class ProceduralTerrainConfig:
    difficulty: str = "normal"  # flat | easy | normal | hard | chaos
    seed: int = 42
    # Optional override of the preset's probability table, e.g. {"flat": 50, "gap": 10}.
    # Keys must be in TERRAIN_KINDS; weights need not sum to 100.
    weights: dict[str, float] | None = None
    chunk_length: float = 12.0  # mm along +x (~0.85 s of walking)
    chunks_behind: int = 2
    chunks_ahead: int = 4
    # Chunk i spans [(i - 0.5) * L, (i + 0.5) * L): the fly spawns in the middle of
    # chunk 0. Chunks with index < flat_start_chunks are always flat.
    flat_start_chunks: int = 2
    # Features are laid out over |y| <= lateral_half_width (the fly may drift or be
    # shoved sideways); outside is the flat, infinite base plane.
    lateral_half_width: float = 25.0  # full-width features (hills, plateaus)
    scatter_sigma: float = 3.0  # lateral spread (std, mm) of point-like features
    # New chunks are centred on the fly's current y, snapped to this (0 = off).
    lateral_snap: float = 1.0
    # Keep features this far from chunk ends so neighbouring chunks never overlap
    # and every chunk starts/ends at base level.
    chunk_margin: float = 0.5
    # Pre-allocated geoms per chunk slot (constant forever).
    boxes_per_chunk: int = 10
    ellipsoids_per_chunk: int = 24
    # Extra pool for interactive spawns.
    spawn_boxes: int = 12
    spawn_ellipsoids: int = 8
    # Spawn safety: the near edge of a spawned feature is at least this far ahead
    # (+x) of the thorax, and no part of it may come within fly_clearance of the
    # thorax or of any fly body (legs included).
    min_spawn_distance: float = 4.0
    default_spawn_distance: float = 6.0
    fly_clearance: float = 3.0
    # Chunk bookkeeping runs every N physics steps (1e-4 s): 50 steps ~ 0.07 mm.
    update_every_steps: int = 50
    # Base plane re-centring (visual only; same numbers as fly_simulator TerrainConfig).
    ground_half_size: float = 1000.0
    checker_size_mm: float = 2.0
    params_override: FeatureParams | None = field(default=None)

    def preset(self) -> DifficultyPreset:
        try:
            p = DIFFICULTY_PRESETS[self.difficulty.lower()]
        except KeyError:
            raise ValueError(
                f"unknown difficulty {self.difficulty!r}; choose from {list(DIFFICULTY_PRESETS)}"
            ) from None
        return DifficultyPreset(
            weights=dict(self.weights) if self.weights is not None else dict(p.weights),
            params=self.params_override or p.params,
        )
