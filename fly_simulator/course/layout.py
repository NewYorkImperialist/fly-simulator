"""Course spec -> world geometry (boxes / ellipsoids) + race markers.

Everything is laid out along +x from the spawn point; every section starts and
ends at base level (z = 0), and checkpoints sit on flat pads, so a respawn there
always lands on the floor. Geoms use the terrain pool's shapes only (box,
ellipsoid) and are served chunk by chunk by ``SectionedGenerator``.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np

from fly_simulator.course.spec import CourseSpec, SectionSpec
from fly_simulator.terrain.generator import BURY, GeomSpec, mound, ramp_box, slab_box, yaw_quat

CARPET_TOP = 0.03  # mm; coloured carpets / lines just above the floor (0.01 z-fights)
GATE_POST = 0.3  # mm edge
GATE_HEIGHT = 2.6
GATE_Y = 4.5  # posts at |y| = GATE_Y (well clear of the legs, ~1.5 mm span)


@dataclass
class SectionInfo:
    index: int
    type: str
    name: str
    x0: float
    x1: float


@dataclass
class Checkpoint:
    index: int  # 1-based (0 = the spawn point)
    x: float
    name: str


@dataclass
class Gate:
    """Slalom pillar at (x, y): pass it on ``side`` ("left" = +y side)."""
    index: int
    x: float
    y: float
    side: str


@dataclass
class Trigger:
    """Fires when the thorax crosses ``x``: kind "whip" (params level, side) or
    "loom" (params set, duration_s, repeats, interval_s)."""
    index: int
    x: float
    kind: str
    params: dict
    section: int


@dataclass
class CourseLayout:
    spec: CourseSpec
    geoms: list[GeomSpec] = field(default_factory=list)
    segments: list[tuple[float, float, str]] = field(default_factory=list)
    sections: list[SectionInfo] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)
    triggers: list[Trigger] = field(default_factory=list)
    route: list[tuple[float, float]] = field(default_factory=list)  # (x, y), x increasing
    start_x: float = 0.0
    finish_x: float = 0.0

    @property
    def length(self) -> float:
        return self.finish_x - self.start_x

    def section_at(self, x: float) -> SectionInfo | None:
        for s in self.sections:
            if s.x0 <= x < s.x1:
                return s
        return None

    def route_y(self, x: float) -> float:
        """Racing line: piecewise linear through the route points, 0 elsewhere."""
        r = self.route
        if not r or x <= r[0][0] or x >= r[-1][0]:
            return 0.0
        i = bisect_right([p[0] for p in r], x)
        (xa, ya), (xb, yb) = r[i - 1], r[i]
        return ya + (yb - ya) * (x - xa) / (xb - xa) if xb > xa else yb

    def counts(self) -> dict[str, int]:
        return {"boxes": sum(g.shape == "box" for g in self.geoms),
                "ellipsoids": sum(g.shape == "ellipsoid" for g in self.geoms)}


def _box(x, y, z, hx, hy, hz, kind, quat=(1.0, 0.0, 0.0, 0.0)) -> GeomSpec:
    return GeomSpec("box", (float(x), float(y), float(z)), (hx, hy, hz), quat, kind)


class _Builder:
    def __init__(self, spec: CourseSpec) -> None:
        self.spec = spec
        self.W = spec.track_half_width
        self.out = CourseLayout(spec)
        self.x = spec.lead_in

    # -------------------------------------------------------------- helpers
    def rng(self, i: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([self.spec.seed, 104729, i]))

    def carpet(self, xa: float, xb: float, kind: str, half_width: float | None = None) -> None:
        """Coloured strip on the floor (split so no piece outgrows a chunk)."""
        hw = self.W if half_width is None else half_width
        n = max(1, math.ceil((xb - xa) / (0.8 * self.spec.chunk_length)))
        edges = np.linspace(xa, xb, n + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            self.out.geoms.append(slab_box(float(a), float(b), CARPET_TOP, hw, kind))

    def gate(self, x: float, kind: str) -> None:
        h = GATE_POST / 2
        for y in (-GATE_Y, GATE_Y):
            self.out.geoms.append(_box(x, y, GATE_HEIGHT / 2 - BURY / 2, h, h,
                                       GATE_HEIGHT / 2 + BURY / 2, kind))
        self.out.geoms.append(_box(x, 0.0, GATE_HEIGHT - h, h, GATE_Y + h, h, "gate_bar"))
        self.out.geoms.append(slab_box(x - 0.15, x + 0.15, CARPET_TOP, GATE_Y, kind))

    def seg(self, a: float, b: float, label: str) -> None:
        self.out.segments.append((a, b, label))

    # ------------------------------------------------------------- building
    def build(self) -> CourseLayout:
        spec, out = self.spec, self.out
        out.start_x = spec.start_x
        self.gate(spec.start_x, "start")
        out.route.append((0.0, 0.0))
        for i, sec in enumerate(spec.sections):
            x0 = self.x
            used = getattr(self, f"_sec_{sec.type}")(i, sec, x0)
            x1 = x0 + max(sec.length, used)
            out.sections.append(SectionInfo(i, sec.type, sec.name or sec.type, x0, x1))
            self.x = x1
        out.route.append((self.x + 50.0, 0.0))
        out.route.sort(key=lambda p: p[0])
        return out

    def _sec_flat(self, i, sec, x0) -> float:
        return 0.0

    def _scatter_y(self, rng, n, spread) -> np.ndarray:
        return np.clip(rng.normal(0.0, spread, size=n), -self.W + 1, self.W - 1)

    def _sec_bumps(self, i, sec, x0) -> float:
        rng, L = self.rng(i), sec.length
        n, h, r = int(sec.param("count")), sec.param("height"), sec.param("radius")
        r = min(r, L / 2 - 0.1)
        xs = np.linspace(x0 + r, x0 + L - r, n) if n > 1 else [x0 + L / 2]
        for x, y in zip(xs, self._scatter_y(rng, n, sec.param("spread"))):
            self.out.geoms.append(mound(float(x), float(y), h, r, r * rng.uniform(0.7, 1.0),
                                        3 * h, yaw_quat(rng.uniform(0, math.pi)), "bumps"))
        self.seg(x0, x0 + L, "bumps")
        return L

    def _sec_rubble(self, i, sec, x0) -> float:
        rng, L = self.rng(i), sec.length
        spread = sec.param("spread")
        for _ in range(int(sec.param("lumps"))):
            h = sec.param("lump_height") * rng.uniform(0.6, 1.2)
            rx, ry = sec.param("lump_radius") * rng.uniform(0.6, 1.2, size=2)
            x = rng.uniform(x0 + rx, x0 + L - rx)
            y = float(np.clip(rng.uniform(-1.5, 1.5) * spread, -self.W + 1, self.W - 1))
            self.out.geoms.append(mound(x, y, h, rx, ry, 2 * h,
                                        yaw_quat(rng.uniform(0, math.pi)), "rough"))
        n = int(sec.param("rocks"))
        for y in self._scatter_y(rng, n, spread):
            h = sec.param("rock_height") * rng.uniform(0.7, 1.2)
            r = sec.param("rock_radius") * rng.uniform(0.8, 1.2)
            x = rng.uniform(x0 + r, x0 + L - r)
            self.out.geoms.append(GeomSpec("ellipsoid", (x, float(y), -0.2 * h),
                                           (r, r * rng.uniform(0.6, 1.0), 1.2 * h),
                                           yaw_quat(rng.uniform(0, 2 * math.pi)), "rocks"))
        self.seg(x0, x0 + L, "rubble")
        return L

    def _sec_blocks(self, i, sec, x0) -> float:
        rng, L = self.rng(i), sec.length
        n = int(sec.param("count"))
        for y in self._scatter_y(rng, n, sec.param("spread")):
            h = sec.param("height") * rng.uniform(0.7, 1.2)
            sx, sy = sec.param("size") * rng.uniform(0.7, 1.2, size=2)
            hd = math.hypot(sx, sy) / 2
            x = rng.uniform(x0 + hd, x0 + L - hd)
            hz = (h + BURY) / 2
            self.out.geoms.append(_box(x, y, h - hz, sx / 2, sy / 2, hz, "blocks",
                                       yaw_quat(rng.uniform(0, math.pi / 2))))
        self.seg(x0, x0 + L, "blocks")
        return L

    def _sec_stairs(self, i, sec, x0) -> float:
        nu, nd = int(sec.param("steps_up")), int(sec.param("steps_down"))
        h, d, land = sec.param("step_height"), sec.param("step_depth"), sec.param("landing")
        x = x0 + 0.5
        top = 0.0
        for k in range(1, nu + 1):
            top = k * h
            self.out.geoms.append(slab_box(x, x + d, top, self.W, "stairs"))
            x += d
        up_end = x
        if land > 0 and top > 0:
            self.out.geoms.append(slab_box(x, x + land, top, self.W, "stairs"))
            x += land
        land_end = x
        nd = max(nd, 1)
        for k in range(1, nd):  # equal steps back down to the floor
            self.out.geoms.append(slab_box(x, x + d, top * (1 - k / nd), self.W, "stairs"))
            x += d
        self.seg(x0, up_end, "stairs_up")
        self.seg(up_end, land_end, "stairs_top")
        self.seg(land_end, x, "stairs_down")
        return x - x0 + 0.5

    def _plateau(self, sec, x0, label) -> float:
        depth, slot, plate = sec.param("depth"), sec.param("width"), sec.param("plateau")
        run = depth / math.tan(math.radians(sec.param("ramp_angle_deg")))
        x1 = x0 + 0.5
        pts = np.cumsum([x1, run, plate, slot, plate, run])
        xa, xb, xc, xd, xe, xf = (float(v) for v in pts)
        self.out.geoms += [
            ramp_box(xa, 0.0, xb, depth, self.W, label),
            slab_box(xb, xc, depth, self.W, label),
            slab_box(xd, xe, depth, self.W, label),
            ramp_box(xe, depth, xf, 0.0, self.W, label),
        ]
        self.seg(xa, xb, "slope_up")
        self.seg(xb, xe, label)
        self.seg(xe, xf, "slope_down")
        return xf - x0 + 0.5

    def _sec_gap(self, i, sec, x0) -> float:
        return self._plateau(sec, x0, "gap")

    def _sec_dip(self, i, sec, x0) -> float:
        return self._plateau(sec, x0, "dip")

    def _sec_ramp(self, i, sec, x0) -> float:
        h, top = sec.param("height"), sec.param("top")
        run = h / math.tan(math.radians(sec.param("angle_deg")))
        a = x0 + 0.5
        b, c, e = a + run, a + run + top, a + 2 * run + top
        self.out.geoms += [ramp_box(a, 0.0, b, h, self.W, "slope"),
                           slab_box(b, c, h, self.W, "slope"),
                           ramp_box(c, h, e, 0.0, self.W, "slope")]
        self.seg(a, b, "slope_up")
        self.seg(b, c, "slope_top")
        self.seg(c, e, "slope_down")
        return e - x0 + 0.5

    def _sec_pillars(self, i, sec, x0) -> float:
        n, sp, off = int(sec.param("count")), sec.param("spacing"), sec.param("offset")
        s, hgt, lead = sec.param("size") / 2, sec.param("height"), sec.param("lead")
        side = sec.param("first_side")
        if side not in ("left", "right"):
            raise ValueError("pillars.first_side must be left or right")
        out = self.out
        out.route.append((x0, 0.0))
        for k in range(n):
            x = x0 + lead + k * sp
            out.geoms.append(_box(x, 0.0, hgt / 2 - BURY / 2, s, s, hgt / 2 + BURY / 2, "pillar"))
            y = off if side == "left" else -off
            out.gates.append(Gate(len(out.gates), x, 0.0, side))
            out.route.append((x, y))
            side = "right" if side == "left" else "left"
        end = x0 + 2 * lead + (n - 1) * sp
        out.route.append((end, 0.0))
        self.seg(x0, end, "slalom")
        return end - x0

    def _sec_tunnel(self, i, sec, x0) -> float:
        L, cz, t = sec.length, sec.param("ceiling_z"), sec.param("thickness")
        wy, piece = sec.param("wall_offset"), min(sec.param("piece"), 0.8 * self.spec.chunk_length)
        n = max(1, math.ceil(L / piece))
        edges = np.linspace(x0, x0 + L, n + 1)
        for a, b in zip(edges[:-1], edges[1:]):
            xm, hx = float(a + b) / 2, float(b - a) / 2
            self.out.geoms.append(_box(xm, 0.0, cz + t / 2, hx, wy + 0.1, t / 2, "ceiling"))
            for y in (-wy, wy):
                self.out.geoms.append(_box(xm, y, (cz - BURY) / 2, hx, 0.08, (cz + BURY) / 2,
                                           "tunnel_wall"))
        self.seg(x0, x0 + L, "tunnel")
        return L

    def _sec_whip_gauntlet(self, i, sec, x0) -> float:
        L, n = sec.length, int(sec.param("cracks"))
        sides = list(sec.param("sides")) or ["random"]
        self.carpet(x0, x0 + L, "whip_zone")
        for k in range(n):
            x = x0 + L * (k + 1) / (n + 1)
            self.out.triggers.append(Trigger(len(self.out.triggers), x, "whip",
                                             {"level": int(sec.param("level")),
                                              "side": sides[k % len(sides)]}, i))
        self.seg(x0, x0 + L, "whip_zone")
        return L

    def _sec_loom_zone(self, i, sec, x0) -> float:
        L = sec.length
        self.carpet(x0, x0 + L, "loom_zone")
        self.out.triggers.append(Trigger(len(self.out.triggers), x0 + 0.5, "loom", {
            k: sec.param(k) for k in ("set", "duration_s", "repeats", "interval_s")}, i))
        self.seg(x0, x0 + L, "loom_zone")
        return L

    def _sec_checkpoint(self, i, sec, x0) -> float:
        pad = max(sec.param("pad"), 1.0)
        x = x0 + pad / 2
        self.gate(x, "checkpoint")
        cps = self.out.checkpoints
        cps.append(Checkpoint(len(cps) + 1, x, sec.name or f"CP{len(cps) + 1}"))
        return pad

    def _sec_finish(self, i, sec, x0) -> float:
        pad = max(sec.param("pad"), 1.0)
        self.out.finish_x = x0 + pad / 2
        self.gate(self.out.finish_x, "finish")
        return pad


def build_layout(spec: CourseSpec) -> CourseLayout:
    return _Builder(spec).build()
