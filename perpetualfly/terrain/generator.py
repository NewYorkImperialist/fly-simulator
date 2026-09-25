"""Reproducible chunk layouts for the endless terrain.

A chunk's content is a pure function of ``(seed, difficulty/weights, chunk index)``:
each chunk draws from its own RNG seeded with ``SeedSequence([seed, index])``, so the
world is identical no matter in which order chunks are (re)generated, and walking
back over recycled ground reproduces it exactly.

Geometry is described in world coordinates as boxes and ellipsoids (the two shape
pools of the chunk manager). All features rest on the base plane z = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from perpetualfly.terrain.config import (
    TERRAIN_KINDS,
    FeatureParams,
    ProceduralTerrainConfig,
)

# Ramps and slabs extend this far below z = 0 (hidden by the base plane) so their
# sides never show a sliver of air and there is no seam to catch a tarsus.
BURY = 0.1


@dataclass
class GeomSpec:
    shape: str  # "box" | "ellipsoid"
    pos: tuple[float, float, float]
    size: tuple[float, float, float]  # box half-sizes / ellipsoid semi-axes
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w, x, y, z
    kind: str = "flat"  # terrain kind, used for colouring

    def xy_extent(self) -> tuple[float, float, float, float]:
        """Conservative world AABB in x/y: (xmin, xmax, ymin, ymax)."""
        r = _rotmat(self.quat)
        half = np.abs(r) @ np.asarray(self.size)
        return (self.pos[0] - half[0], self.pos[0] + half[0],
                self.pos[1] - half[1], self.pos[1] + half[1])

    def top_z(self) -> float:
        r = _rotmat(self.quat)
        return float(self.pos[2] + (np.abs(r) @ np.asarray(self.size))[2])


@dataclass
class ChunkSpec:
    index: int
    kind: str
    x0: float
    x1: float
    geoms: list[GeomSpec] = field(default_factory=list)
    # Sub-regions along x for terrain_type_at(): (x_start, x_end, label). Anything
    # inside the chunk but not covered is "flat".
    segments: list[tuple[float, float, str]] = field(default_factory=list)

    def label_at(self, x: float) -> str:
        for a, b, label in self.segments:
            if a <= x < b:
                return label
        return "flat"

    def shift_y(self, dy: float) -> None:
        for g in self.geoms:
            g.pos = (g.pos[0], g.pos[1] + dy, g.pos[2])

    @property
    def n_boxes(self) -> int:
        return sum(g.shape == "box" for g in self.geoms)

    @property
    def n_ellipsoids(self) -> int:
        return sum(g.shape == "ellipsoid" for g in self.geoms)


def _rotmat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def random_quat(rng: np.random.Generator) -> tuple[float, float, float, float]:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


# ------------------------------------------------------------------ primitives
def ramp_box(xa: float, za: float, xb: float, zb: float, half_width: float,
             kind: str, y: float = 0.0) -> GeomSpec:
    """Tilted slab whose top face runs exactly from (xa, za) to (xb, zb).

    The box is rotated about +y; its top face is the plane through both points, the
    slab thickness is chosen so its underside reaches below z = 0 everywhere, hence
    it looks and acts like a solid wedge together with the plateau it leads to.
    """
    dx, dz = xb - xa, zb - za
    length = math.hypot(dx, dz)
    theta = math.atan2(dz, dx)  # rise angle
    # Rotation about y by phi maps local x -> (cos phi, 0, -sin phi); we need +theta
    # rise, so phi = -theta. Local z (top-face normal) -> (-sin theta, 0, cos theta).
    phi = -theta
    quat = (math.cos(phi / 2), 0.0, math.sin(phi / 2), 0.0)
    normal = np.array([-math.sin(theta), 0.0, math.cos(theta)])
    half_t = (max(za, zb) + BURY) / (2 * math.cos(theta)) + 0.02
    mid = np.array([(xa + xb) / 2, y, (za + zb) / 2])
    center = mid - half_t * normal
    return GeomSpec("box", tuple(float(v) for v in center),
                    (length / 2, half_width, half_t), quat, kind)


def slab_box(xa: float, xb: float, top: float, half_width: float, kind: str,
             y: float = 0.0) -> GeomSpec:
    """Axis-aligned slab from x=xa..xb with its top at z=top (buried below 0)."""
    hz = (top + BURY) / 2
    return GeomSpec("box", ((xa + xb) / 2, y, top - hz), ((xb - xa) / 2, half_width, hz),
                    kind=kind)


def mound(x: float, y: float, height: float, rx: float, ry: float, rz: float,
          quat=(1.0, 0.0, 0.0, 0.0), kind: str = "bumps") -> GeomSpec:
    """Ellipsoid sunk into the ground so that only a cap of ``height`` protrudes."""
    rz = max(rz, height)
    return GeomSpec("ellipsoid", (x, y, height - rz), (rx, ry, rz), quat, kind)


# ------------------------------------------------------------------- generator
class TerrainGenerator:
    """Chunk layouts from a difficulty preset / probability table.

    ``TerrainGenerator(seed=42, difficulty="normal")``; ``chunk(i)`` -> ``ChunkSpec``.
    """

    def __init__(self, seed: int = 42, difficulty: str = "normal",
                 cfg: ProceduralTerrainConfig | None = None,
                 weights: dict[str, float] | None = None) -> None:
        self.cfg = cfg or ProceduralTerrainConfig()
        self.seed = int(seed)
        self.difficulty = difficulty
        preset = replace(self.cfg, difficulty=difficulty).preset()
        w = dict(weights) if weights is not None else preset.weights
        unknown = set(w) - set(TERRAIN_KINDS)
        if unknown:
            raise ValueError(f"unknown terrain kinds {sorted(unknown)}; valid: {TERRAIN_KINDS}")
        if any(v < 0 for v in w.values()) or sum(w.values()) <= 0:
            raise ValueError(f"invalid weights {w}")
        self.params: FeatureParams = preset.params
        self.kinds = [k for k in TERRAIN_KINDS if w.get(k, 0) > 0]
        p = np.array([w[k] for k in self.kinds], dtype=float)
        self.probabilities = p / p.sum()

    # geometry of the chunk grid
    def chunk_bounds(self, index: int) -> tuple[float, float]:
        L = self.cfg.chunk_length
        return ((index - 0.5) * L, (index + 0.5) * L)

    def chunk_index_at(self, x: float) -> int:
        return int(math.floor(x / self.cfg.chunk_length + 0.5))

    def rng_for(self, index: int) -> np.random.Generator:
        # SeedSequence needs non-negative entries; map the index bijectively.
        idx = 2 * index if index >= 0 else -2 * index - 1
        return np.random.default_rng(np.random.SeedSequence([self.seed, idx]))

    def sample_kind(self, index: int) -> str:
        if index < self.cfg.flat_start_chunks:
            return "flat"
        rng = self.rng_for(index)
        return str(self.kinds[int(rng.choice(len(self.kinds), p=self.probabilities))])

    def chunk(self, index: int, kind: str | None = None) -> ChunkSpec:
        """Layout of chunk ``index`` (optionally forcing a kind)."""
        x0, x1 = self.chunk_bounds(index)
        rng = self.rng_for(index)
        sampled = "flat"
        if index >= self.cfg.flat_start_chunks:  # first draw of the chunk's stream
            sampled = str(self.kinds[int(rng.choice(len(self.kinds), p=self.probabilities))])
        kind = kind or sampled
        spec = ChunkSpec(index, kind, x0, x1)
        m = self.cfg.chunk_margin
        builder = getattr(self, f"_build_{kind}")
        builder(spec, rng, x0 + m, x1 - m)
        self._fit_pool(spec)
        return spec

    def feature(self, kind: str, x_start: float, rng: np.random.Generator,
                max_boxes: int, max_ellipsoids: int) -> tuple[list[GeomSpec], list]:
        """A single interactive feature starting at ``x_start`` (for spawn_ahead)."""
        spec = ChunkSpec(-1, kind, x_start, x_start + 100.0)
        p = self.params
        W = self.cfg.lateral_half_width
        if kind == "rock":
            h = rng.uniform(*p.rock_height) * 1.3
            r = rng.uniform(*p.rock_radius) * 1.3
            spec.geoms.append(mound(x_start + r, rng.uniform(-0.3, 0.3), h, r, r * 0.8,
                                    h * 1.2, random_quat(rng), "rocks"))
            spec.segments.append((x_start, x_start + 2 * r, "rocks"))
        elif kind == "bump":
            h = p.bump_height[1]
            r = p.bump_radius[1]
            spec.geoms.append(mound(x_start + r, 0.0, h, r, r, 3 * h, kind="bumps"))
            spec.segments.append((x_start, x_start + 2 * r, "bumps"))
        elif kind == "slope":
            self._hill(spec, rng, x_start, W, angle=p.slope_angle_deg[1],
                       height=p.slope_height[1], top=p.slope_top_length[1])
        elif kind == "gap":
            self._plateau(spec, x_start, W, depth=p.gap_depth[1],
                          plate=p.plateau_length[0], slot=p.gap_width[1], label="gap")
        elif kind == "dip":
            self._plateau(spec, x_start, W, depth=p.dip_depth[1],
                          plate=p.plateau_length[0], slot=p.dip_width[1], label="dip")
        else:
            raise ValueError(f"cannot spawn {kind!r}; choose rock, bump, slope, gap, dip")
        if spec.n_boxes > max_boxes or spec.n_ellipsoids > max_ellipsoids:
            raise ValueError(f"spawn pool too small for {kind}")
        return spec.geoms, spec.segments

    # --------------------------------------------------------------- builders
    def _fit_pool(self, spec: ChunkSpec) -> None:
        """Drop surplus geoms so a chunk always fits its fixed pool slots."""
        boxes = [g for g in spec.geoms if g.shape == "box"][: self.cfg.boxes_per_chunk]
        ells = [g for g in spec.geoms if g.shape == "ellipsoid"][: self.cfg.ellipsoids_per_chunk]
        spec.geoms = boxes + ells

    def _scatter_y(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Lateral positions concentrated near the walking line (|y| small)."""
        W = self.cfg.lateral_half_width
        return np.clip(rng.normal(0.0, self.cfg.scatter_sigma, size=n), -W, W)

    def _build_flat(self, spec, rng, a, b) -> None:
        pass

    def _build_bumps(self, spec, rng, a, b) -> None:
        p = self.params
        n = int(rng.integers(p.bump_count[0], p.bump_count[1] + 1))
        for y in self._scatter_y(rng, n):
            h = rng.uniform(*p.bump_height)
            rx, ry = np.minimum(rng.uniform(*p.bump_radius, size=2), (b - a) / 2)
            r = max(rx, ry)  # yaw-rotated below: keep the whole footprint in the chunk
            x = rng.uniform(a + r, b - r)
            # rz = 3*h keeps the cap's edge slope gentle.
            spec.geoms.append(mound(x, float(y), h, rx, ry, 3 * h,
                                    yaw_quat(rng.uniform(0, math.pi)), "bumps"))
        spec.segments.append((spec.x0, spec.x1, "bumps"))

    def _build_rough(self, spec, rng, a, b) -> None:
        p = self.params
        n = int(rng.integers(p.rough_count[0], p.rough_count[1] + 1))
        W = min(self.cfg.lateral_half_width, 1.5 * self.cfg.scatter_sigma)
        for _ in range(n):
            h = rng.uniform(*p.rough_height)
            rx, ry = rng.uniform(*p.rough_radius, size=2)
            r = max(rx, ry)
            x = rng.uniform(a + r, b - r)
            y = rng.uniform(-W, W)
            spec.geoms.append(mound(x, y, h, rx, ry, 2 * h,
                                    yaw_quat(rng.uniform(0, math.pi)), "rough"))
        spec.segments.append((spec.x0, spec.x1, "rough"))

    def _build_rocks(self, spec, rng, a, b) -> None:
        p = self.params
        n = int(rng.integers(p.rock_count[0], p.rock_count[1] + 1))
        for y in self._scatter_y(rng, n):
            h = rng.uniform(*p.rock_height)
            r = rng.uniform(*p.rock_radius)
            x = rng.uniform(a + r, b - r)
            # Compact, randomly tilted ellipsoid, mostly above ground.
            rx, ry = r, r * rng.uniform(0.6, 1.0)
            spec.geoms.append(GeomSpec("ellipsoid", (x, float(y), h - 1.2 * h),
                                       (rx, ry, 1.2 * h), yaw_quat(rng.uniform(0, 2 * math.pi)),
                                       "rocks"))
        spec.segments.append((spec.x0, spec.x1, "rocks"))

    def _build_blocks(self, spec, rng, a, b) -> None:
        p = self.params
        n = int(rng.integers(p.block_count[0], p.block_count[1] + 1))
        for y in self._scatter_y(rng, n):
            h = rng.uniform(*p.block_height)
            sx, sy = rng.uniform(*p.block_size, size=2)
            half_diag = math.hypot(sx, sy) / 2
            x = rng.uniform(a + half_diag, b - half_diag)
            hz = (h + BURY) / 2
            spec.geoms.append(GeomSpec("box", (x, float(y), h - hz), (sx / 2, sy / 2, hz),
                                       yaw_quat(rng.uniform(0, math.pi / 2)), "blocks"))
        spec.segments.append((spec.x0, spec.x1, "blocks"))

    def _build_slope(self, spec, rng, a, b) -> None:
        p = self.params
        angle = rng.uniform(*p.slope_angle_deg)
        h = rng.uniform(*p.slope_height)
        top = rng.uniform(*p.slope_top_length)
        run = h / math.tan(math.radians(angle))
        total = 2 * run + top
        if total > b - a:  # shrink to fit the chunk
            h *= (b - a) / total
            top *= (b - a) / total
            total = b - a
        x_start = rng.uniform(a, b - total)
        self._hill(spec, rng, x_start, self.cfg.lateral_half_width, angle, h, top)

    def _build_gap(self, spec, rng, a, b) -> None:
        p = self.params
        self._plateau_in(spec, rng, a, b, rng.uniform(*p.gap_depth),
                         rng.uniform(*p.gap_width), "gap")

    def _build_dip(self, spec, rng, a, b) -> None:
        p = self.params
        self._plateau_in(spec, rng, a, b, rng.uniform(*p.dip_depth),
                         rng.uniform(*p.dip_width), "dip")

    def _plateau_in(self, spec, rng, a, b, depth, slot, label) -> None:
        p = self.params
        plate = rng.uniform(*p.plateau_length)
        run = depth / math.tan(math.radians(p.plateau_ramp_angle_deg))
        total = 2 * run + 2 * plate + slot
        if total > b - a:  # shorten plateaus, then steepen ramps, to fit the chunk
            plate = max(0.8, (b - a - 2 * run - slot) / 2)
            run = min(run, (b - a - 2 * plate - slot) / 2)
            total = 2 * run + 2 * plate + slot
        x_start = rng.uniform(a, max(a, b - total))
        self._plateau(spec, x_start, self.cfg.lateral_half_width, depth, plate, slot, label,
                      run=run)

    # ----------------------------------------------------- composite shapes
    def _hill(self, spec: ChunkSpec, rng, x_start: float, W: float, angle: float,
              height: float, top: float) -> None:
        run = height / math.tan(math.radians(angle))
        x1, x2, x3 = x_start + run, x_start + run + top, x_start + 2 * run + top
        spec.geoms += [
            ramp_box(x_start, 0.0, x1, height, W, "slope"),
            slab_box(x1, x2, height, W, "slope"),
            ramp_box(x2, height, x3, 0.0, W, "slope"),
        ]
        spec.segments += [(x_start, x1, "slope_up"), (x1, x2, "slope_top"),
                          (x2, x3, "slope_down")]

    def _plateau(self, spec: ChunkSpec, x_start: float, W: float, depth: float,
                 plate: float, slot: float, label: str, run: float | None = None) -> None:
        """Ramp up -> plateau -> slot (to the base plane) -> plateau -> ramp down."""
        if run is None:
            run = depth / math.tan(math.radians(self.params.plateau_ramp_angle_deg))
        x1 = x_start + run
        x2 = x1 + plate
        x3 = x2 + slot
        x4 = x3 + plate
        x5 = x4 + run
        spec.geoms += [
            ramp_box(x_start, 0.0, x1, depth, W, label),
            slab_box(x1, x2, depth, W, label),
            slab_box(x3, x4, depth, W, label),
            ramp_box(x4, depth, x5, 0.0, W, label),
        ]
        spec.segments += [(x_start, x1, "slope_up"), (x1, x5, label), (x4, x5, "slope_down")]
        # label_at returns the first match, so put the ramp-down before the plateau.
        spec.segments[-2], spec.segments[-1] = spec.segments[-1], spec.segments[-2]
