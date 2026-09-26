"""A fixed (hand-designed) layout served chunk by chunk to ``ProceduralTerrain``.

The endless ``TerrainGenerator`` draws each chunk from a seeded RNG. A
``SectionedGenerator`` instead holds a pre-built list of world-space geoms (e.g. an
obstacle course from ``perpetualfly.course``) and returns, for chunk ``i``, the
geoms whose *centre* lies in that chunk. Everything outside the layout is flat
base plane. It is a drop-in replacement for ``ProceduralTerrain.generator`` (no
model rebuild, same pooled geoms), so the pool size never changes.

Constraints checked by ``validate()``:

* a chunk may not hold more boxes / ellipsoids than the pool slot has;
* a geom may not stick out of its chunk by more than one chunk length behind /
  in front, otherwise part of it could be missing while the fly is next to it
  (the manager loads ``chunks_behind`` chunks behind and ``chunks_ahead`` ahead).
"""

from __future__ import annotations

from dataclasses import replace

from perpetualfly.terrain.config import ProceduralTerrainConfig
from perpetualfly.terrain.generator import ChunkSpec, GeomSpec, TerrainGenerator


class SectionedGenerator(TerrainGenerator):
    """``chunk(i)`` -> the geoms of a fixed layout whose centre x is in chunk ``i``."""

    kind_label = "course"

    def __init__(self, geoms: list[GeomSpec], segments: list[tuple[float, float, str]],
                 cfg: ProceduralTerrainConfig, seed: int = 0) -> None:
        # The base class only supplies the chunk grid (+ spawn_ahead features).
        super().__init__(seed, "normal", cfg=cfg, weights={"flat": 1.0})
        self.layout_geoms = list(geoms)
        self.layout_segments = sorted(segments, key=lambda s: s[0])
        self._by_chunk: dict[int, list[GeomSpec]] = {}
        for g in self.layout_geoms:
            self._by_chunk.setdefault(self.chunk_index_at(g.pos[0]), []).append(g)

    def sample_kind(self, index: int) -> str:
        return self.kind_label if index in self._by_chunk else "flat"

    def chunk(self, index: int, kind: str | None = None) -> ChunkSpec:
        x0, x1 = self.chunk_bounds(index)
        if kind == "flat":  # ProceduralTerrain.flatten_next_chunk override
            return ChunkSpec(index, "flat", x0, x1)
        geoms = [replace(g) for g in self._by_chunk.get(index, [])]
        segs = [(max(a, x0), min(b, x1), lab) for a, b, lab in self.layout_segments
                if a < x1 and b > x0]
        kind = self.kind_label if geoms or segs else "flat"
        return ChunkSpec(index, kind, x0, x1, geoms=geoms, segments=segs)

    def validate(self) -> list[str]:
        """Problems that would make the layout render / collide incompletely."""
        c = self.cfg
        problems = []
        for idx, gs in sorted(self._by_chunk.items()):
            nb = sum(g.shape == "box" for g in gs)
            ne = len(gs) - nb
            if nb > c.boxes_per_chunk or ne > c.ellipsoids_per_chunk:
                x0, x1 = self.chunk_bounds(idx)
                problems.append(
                    f"chunk {idx} (x {x0:.1f}..{x1:.1f} mm) needs {nb} boxes / {ne} ellipsoids,"
                    f" pool has {c.boxes_per_chunk} / {c.ellipsoids_per_chunk}")
            x0, x1 = self.chunk_bounds(idx)
            L = c.chunk_length
            for g in gs:
                xmin, xmax, _, _ = g.xy_extent()
                if xmin < x0 - L * c.chunks_behind or xmax > x1 + L * c.chunks_ahead:
                    problems.append(f"{g.kind} geom at x={g.pos[0]:.1f} spans {xmin:.1f}.."
                                    f"{xmax:.1f} mm: too long for chunk length {L:g} mm")
        return problems
