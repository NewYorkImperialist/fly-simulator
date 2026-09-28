"""Odour plume: a concentration field in the world that the fly smells with its two
antennae (``--odor-plume``; docs/SMELL.md). Needs a smell fix in the brain
(``--smell-fix``, fly_simulator/brain/smell.py): without one, ORN input ignites
the model's runaway state.

* ``OdorPlume``: a source at (x, y) whose position meanders sideways
  (``meander_mm * sin(2 pi t / meander_period_s)``, perpendicular to the
  source's x axis); concentration ``c = c_source * exp(-r / length_mm)`` with r
  the distance to the (meandering) source. Normalised units: ``smell.orn_rate``
  gives half-maximal ORN rates at c = 0.5. Optionally a second source with a
  control odour (``control_odor``) at (``control_x_mm``, ``control_y_mm``).
  A translucent disc marks each source (visual only, no physics).
* ``PlumeSense``: every ``refresh_s`` it samples the field at the two antennae
  (``antenna_ahead_mm`` ahead of the thorax, ``antenna_sep_mm`` apart; about
  1 mm and 0.4 mm on a real fly) and sends ``StimulusEvent("odor", ...)`` with
  the left / right concentrations. The brain drives the odour glomeruli's ORNs of
  each antenna (``smell.odor_drive_groups``); the connectome decides whether that
  reaches the turning DNs (it largely does not: docs/SMELL.md).
* ``contrast_gain`` (1 = off): exaggerates the left / right difference
  ``c_L,R = mean * (1 +/- gain * (c_L - c_R) / (c_L + c_R))``. **A modelling aid,
  not biology**: with the real 0.4 mm antennal separation the bilateral
  difference is a few percent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import mujoco as mj
import numpy as np

from fly_simulator.brain.schema import StimulusEvent

if TYPE_CHECKING:
    from fly_simulator.app import Session

PREFIX = "odor_plume_src"
PARK_POS = (0.0, 0.0, -200.0)
SOURCE_COLOR = (0.95, 0.65, 0.15, 0.35)   # vinegar: amber haze
CONTROL_COLOR = (0.6, 0.6, 0.6, 0.30)


@dataclass
class PlumeConfig:
    """``--odor-plume`` (AppConfig.plume)."""

    enabled: bool = False
    odor: str = "vinegar"            # fly_simulator.brain.smell.ODORS
    source_x_mm: float = 25.0
    source_y_mm: float = 6.0
    c_source: float = 2.0            # normalised concentration at the source
    length_mm: float = 12.0          # e-fold distance of the concentration
    meander_mm: float = 1.5
    meander_period_s: float = 5.0
    control_odor: str = ""           # "" = no control source
    control_x_mm: float = 25.0
    control_y_mm: float = -6.0
    antenna_ahead_mm: float = 1.0
    antenna_sep_mm: float = 0.4
    contrast_gain: float = 1.0       # 1 = real geometry (see module doc)
    r_max_hz: float = 100.0          # ORN rate ceiling
    refresh_s: float = 0.05
    stim_duration_s: float = 0.08
    min_conc: float = 0.01           # below this no event is sent
    marker_radius_mm: float = 1.0

    @classmethod
    def from_dict(cls, d: dict) -> "PlumeConfig":
        d = dict(d)
        bad = set(d) - set(cls.__dataclass_fields__)
        if bad:
            raise ValueError(f"Unknown config keys for PlumeConfig: {sorted(bad)}")
        return cls(**d)


class OdorPlume:
    """The concentration field (pure function of position and time)."""

    def __init__(self, cfg: PlumeConfig | None = None) -> None:
        self.cfg = cfg or PlumeConfig()
        self.sim = None
        self._gids: list[int] = []

    def sources(self, t: float) -> list[tuple[str, float, float]]:
        c = self.cfg
        dy = c.meander_mm * math.sin(2 * math.pi * t / max(c.meander_period_s, 1e-6))
        out = [(c.odor, c.source_x_mm, c.source_y_mm + dy)]
        if c.control_odor:
            out.append((c.control_odor, c.control_x_mm, c.control_y_mm - dy))
        return out

    def conc(self, x: float, y: float, t: float) -> dict[str, float]:
        c = self.cfg
        out: dict[str, float] = {}
        for odor, sx, sy in self.sources(t):
            r = math.hypot(x - sx, y - sy)
            out[odor] = out.get(odor, 0.0) + c.c_source * math.exp(-r / max(c.length_mm, 1e-6))
        return out

    def antennae(self, x: float, y: float, heading: float) -> tuple[tuple[float, float],
                                                                    tuple[float, float]]:
        """World (x, y) of the left and right antenna."""
        c = self.cfg
        fx, fy = math.cos(heading), math.sin(heading)
        lx, ly = -fy, fx  # left of the heading
        ax, ay = x + c.antenna_ahead_mm * fx, y + c.antenna_ahead_mm * fy
        h = 0.5 * c.antenna_sep_mm
        return (ax + h * lx, ay + h * ly), (ax - h * lx, ay - h * ly)

    def bilateral(self, x: float, y: float, heading: float, t: float
                  ) -> dict[str, tuple[float, float]]:
        """{odour: (c_left, c_right)} at the antennae (contrast_gain applied)."""
        (xl, yl), (xr, yr) = self.antennae(x, y, heading)
        cl, cr = self.conc(xl, yl, t), self.conc(xr, yr, t)
        g = self.cfg.contrast_gain
        out = {}
        for o in cl:
            a, b = cl[o], cr.get(o, 0.0)
            if g != 1.0 and a + b > 0:
                m, d = 0.5 * (a + b), (a - b) / (a + b)
                a, b = m * max(1 + g * d, 0.0), m * max(1 - g * d, 0.0)
            out[o] = (a, b)
        return out

    # ------------------------------------------------------------ visual markers
    def extension(self, world) -> None:
        spec: mj.MjSpec = world.mjcf_root
        for j in range(2):
            spec.worldbody.add_geom(
                name=f"{PREFIX}{j}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.01, 0.01, 0),
                pos=PARK_POS, rgba=(1, 1, 1, 0), contype=0, conaffinity=0, group=0)

    def attach(self, sim) -> "OdorPlume":
        self.sim = sim
        self._gids = [mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, f"{PREFIX}{j}")
                      for j in range(2)]
        if any(g < 0 for g in self._gids):
            raise RuntimeError("odour plume geoms missing: add OdorPlume.extension")
        self.update_markers(0.0)
        return self

    def update_markers(self, t: float) -> None:
        if self.sim is None:
            return
        m, d = self.sim.model, self.sim.data
        r, h = self.cfg.marker_radius_mm, 0.6
        srcs = self.sources(t)
        for j, g in enumerate(self._gids):
            if j >= len(srcs):
                m.geom_rgba[g, 3] = 0.0
                continue
            _, x, y = srcs[j]
            m.geom_size[g] = (r, h / 2, 0.0)
            m.geom_rbound[g] = math.hypot(r, h / 2)
            m.geom_aabb[g] = (0.0, 0.0, 0.0, r, r, h / 2)
            m.geom_rgba[g] = SOURCE_COLOR if j == 0 else CONTROL_COLOR
            m.geom_pos[g] = (x, y, h / 2)
            d.geom_xpos[g] = (x, y, h / 2)
            d.geom_xmat[g] = (1, 0, 0, 0, 1, 0, 0, 0, 1)


class PlumeHandle:
    """Session glue: antennae -> brain (odor events), HUD, summary."""

    def __init__(self, session: "Session", plume: OdorPlume, cfg: PlumeConfig,
                 say: Callable[[str], None] | None = None) -> None:
        self.session, self.plume, self.cfg = session, plume, cfg
        self.say = say or (lambda m: None)
        self._last_send = -1e9
        self.last: dict[str, tuple[float, float]] = {}
        self.n_sent = 0
        self.track: list[tuple[float, float, float, float]] = []  # (t, x, y, dist)
        self.min_dist = {o: math.inf for o, *_ in plume.sources(0.0)}

    def _pose(self) -> tuple[float, float, float]:
        sim = self.session.sim
        p = sim.data.xpos[sim.thorax_body_id]
        return float(p[0]), float(p[1]), sim.heading()

    def update(self) -> None:
        """Once per physics chunk."""
        rt = self.session.run_time()
        self.plume.update_markers(rt)
        x, y, hd = self._pose()
        for o, sx, sy in self.plume.sources(rt):
            self.min_dist[o] = min(self.min_dist.get(o, math.inf), math.hypot(x - sx, y - sy))
        if rt - self._last_send < self.cfg.refresh_s:
            return
        self._last_send = rt
        self.last = self.plume.bilateral(x, y, hd, rt)
        _, sx, sy = self.plume.sources(rt)[0]
        self.track.append((rt, x, y, math.hypot(x - sx, y - sy)))
        link = self.session.brain
        if link is None or link.brain is None or not link.brain.is_alive():
            return
        for o, (cl, cr) in self.last.items():
            if max(cl, cr) < self.cfg.min_conc:
                continue
            ev = StimulusEvent("odor", "none", float(max(cl, cr)), self.cfg.stim_duration_s, rt,
                               details={"odor": o, "left": float(cl), "right": float(cr),
                                        "r_max_hz": float(self.cfg.r_max_hz)})
            try:
                link.brain.send(ev)
                self.n_sent += 1
            except Exception:
                pass

    def hud_line(self) -> str:
        parts = [f"{o} L{cl:.2f} R{cr:.2f}" for o, (cl, cr) in self.last.items()]
        st = self.session.brain.latest if self.session.brain is not None else None
        sm = (getattr(st, "smell", None) or {}) if st is not None else {}
        r = sm.get("rates") or {}
        tail = (f"  PN L{r.get('uPN_L', 0):.0f} R{r.get('uPN_R', 0):.0f} "
                f"KC {100 * sm.get('kc_active_frac', 0):.1f}% LH {r.get('LH', 0):.1f} Hz"
                + ("  RUNAWAY" if sm.get("runaway") else "")) if sm else ""
        d = self.track[-1][3] if self.track else float("nan")
        return f"PLUME {'  '.join(parts) or 'no odour'}  source {d:.1f} mm{tail}"

    def summary(self) -> dict:
        d = [row[3] for row in self.track]
        return {"odor": self.cfg.odor, "control": self.cfg.control_odor or None,
                "events_sent": self.n_sent,
                "dist_start_mm": round(d[0], 2) if d else None,
                "dist_end_mm": round(d[-1], 2) if d else None,
                "min_dist_mm": {k: round(v, 2) for k, v in self.min_dist.items()},
                "contrast_gain": self.cfg.contrast_gain}


def install_plume(session: "Session", plume: OdorPlume, cfg: PlumeConfig | None = None,
                  say: Callable[[str], None] | None = None) -> PlumeHandle:
    cfg = cfg or plume.cfg
    plume.attach(session.sim)
    return PlumeHandle(session, plume, cfg, say=say)
