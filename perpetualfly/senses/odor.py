"""Odour zones: coloured haze regions in the world that the fly smells
(``--odor-zones``; with ``--learning`` whip hits inside a zone punish that odour).
docs/FEAR_LEARNING.md.

* ``OdorZones``: translucent coloured columns (odour A = magenta haze, B = teal
  haze) on a small pool of MuJoCo cylinders with ``contype = conaffinity = 0``
  (visual only, no physics change). Zones come from the config (``zones``: list of
  ``{"odor", "x", "y", "r"}``), from ``layout="row"`` (alternating A / B zones on
  the fly's path from ``first_x_mm`` every ``spacing_mm``) or are spawned around
  the fly with keys 8 (A) / = (B).
* ``OdorSense``: while the thorax is inside a zone the brain receives the odour's
  **Kenyon-cell code** (``StimulusEvent("manual", details={"set": "odor_A"})``,
  refreshed every ``refresh_s``). Odours are delivered to KCs, not ORNs / PNs,
  because olfactory input ignites the whole-brain model's runaway state
  (docs/SENSORY_SCREEN.md); the KC sets are built in the brain worker from the
  connectome (perpetualfly/brain/plasticity.py, ``odor_kc_code``).
* Punishment (``--learning``): a whip hit while the fly is in a zone (or left it
  less than ``punish_window_s`` ago) additionally drives the PPL1 punishment
  dopaminergic neurons (named set ``dan_punish``) at ``punish_hz`` for
  ``punish_duration_s``. **This is a stand-in**: in this model the whip's
  modelled afferents (body mechanosensory, JO, the nociceptive AN relay) do not
  reach PPL1 at all (measured, docs/FEAR_LEARNING.md), so the hit -> DAN link is
  ours; the dopamine-gated KC -> MBON depression that follows runs in the brain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import mujoco as mj
import numpy as np

from perpetualfly.brain.schema import StimulusEvent

if TYPE_CHECKING:
    from perpetualfly.app import Session

ODORS = ("A", "B")
ODOR_COLORS: dict[str, tuple[float, float, float, float]] = {
    "A": (0.95, 0.30, 0.75, 0.22),  # magenta haze
    "B": (0.20, 0.80, 0.70, 0.22),  # teal haze
}
PREFIX = "odor_zone"
PARK_POS = (0.0, 0.0, -200.0)
PARK_SIZE = 0.01
ODOR_KEYS: dict[str, str] = {"8": "A", "=": "B"}  # spawn a zone around the fly
PUNISH_SET = "dan_punish"  # = perpetualfly.brain.plasticity.PUNISH_SET


@dataclass
class OdorConfig:
    """``--odor-zones`` / ``--learning`` (AppConfig.odor)."""

    enabled: bool = False
    learning: bool = False             # whip hits in a zone -> punishment DANs
    layout: str = "row"                # "row" | "none" (only config zones / keys)
    first_x_mm: float = 12.0
    spacing_mm: float = 14.0
    n_row: int = 6                     # zones in the row (A, B, A, B, ...)
    radius_mm: float = 4.0
    height_mm: float = 2.5             # haze column height
    zones: list = field(default_factory=list)  # [{"odor", "x", "y", "r"}]
    pool_size: int = 12
    rate_hz: float = 0.0               # KC rate; 0 = the brain's PlasticityConfig.odor_hz
    refresh_s: float = 0.1
    stim_duration_s: float = 0.15
    punish_hz: float = 0.0             # 0 = PlasticityConfig.punish_hz
    punish_duration_s: float = 1.0
    punish_window_s: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "OdorConfig":
        d = dict(d)
        bad = set(d) - set(cls.__dataclass_fields__)
        if bad:
            raise ValueError(f"Unknown config keys for OdorConfig: {sorted(bad)}")
        return cls(**d)


@dataclass
class Zone:
    odor: str
    x: float
    y: float
    r: float
    gid: int = -1


class OdorZones:
    """Visual odour zones on a geom pool."""

    def __init__(self, cfg: OdorConfig | None = None) -> None:
        self.cfg = cfg or OdorConfig()
        self.sim = None
        self.zones: list[Zone] = []
        self._free: list[int] = []

    def extension(self, world) -> None:
        spec: mj.MjSpec = world.mjcf_root
        for j in range(self.cfg.pool_size):
            spec.worldbody.add_geom(
                name=f"{PREFIX}{j}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                size=(PARK_SIZE, PARK_SIZE, 0), pos=PARK_POS, rgba=(1, 1, 1, 0),
                contype=0, conaffinity=0, group=0)

    def attach(self, sim) -> "OdorZones":
        self.sim = sim
        ids = [mj.mj_name2id(sim.model, mj.mjtObj.mjOBJ_GEOM, f"{PREFIX}{j}")
               for j in range(self.cfg.pool_size)]
        if any(i < 0 for i in ids):
            raise RuntimeError("odour zone geoms missing: add OdorZones.extension")
        self._pool = ids
        self.relayout()
        return self

    def initial_zones(self) -> list[Zone]:
        c = self.cfg
        out = [Zone(str(z["odor"]), float(z["x"]), float(z.get("y", 0.0)),
                    float(z.get("r", c.radius_mm))) for z in c.zones]
        if c.layout == "row":
            out += [Zone(ODORS[k % 2], c.first_x_mm + k * c.spacing_mm, 0.0, c.radius_mm)
                    for k in range(c.n_row)]
        return out

    def relayout(self) -> None:
        for g in self._pool:
            self._park(g)
        self._free = list(self._pool)
        self.zones = []
        for z in self.initial_zones():
            self.place(z.odor, z.x, z.y, z.r)

    def place(self, odor: str, x: float, y: float, r: float | None = None) -> Zone:
        if not self._free:  # recycle the oldest zone
            old = self.zones.pop(0)
            self._park(old.gid)
            self._free.append(old.gid)
        z = Zone(odor, float(x), float(y), float(r or self.cfg.radius_mm), self._free.pop(0))
        self.zones.append(z)
        m, d = self.sim.model, self.sim.data
        h = self.cfg.height_mm / 2
        g = z.gid
        m.geom_size[g] = (z.r, h, 0.0)
        m.geom_rbound[g] = math.hypot(z.r, h)
        m.geom_aabb[g] = (0.0, 0.0, 0.0, z.r, z.r, h)
        m.geom_rgba[g] = ODOR_COLORS.get(odor, (0.8, 0.8, 0.8, 0.2))
        pos = (z.x, z.y, h)
        m.geom_pos[g] = pos
        m.geom_quat[g] = (1.0, 0.0, 0.0, 0.0)
        d.geom_xpos[g] = pos
        d.geom_xmat[g] = (1, 0, 0, 0, 1, 0, 0, 0, 1)
        return z

    def clear(self) -> None:
        """Remove every zone."""
        for z in self.zones:
            self._park(z.gid)
            self._free.append(z.gid)
        self.zones = []

    def _park(self, g: int) -> None:
        m, d = self.sim.model, self.sim.data
        m.geom_pos[g] = PARK_POS
        m.geom_size[g] = (PARK_SIZE, PARK_SIZE, 0.0)
        m.geom_rgba[g, 3] = 0.0
        d.geom_xpos[g] = PARK_POS

    def odor_at(self, x: float, y: float) -> str | None:
        """The odour at (x, y) (the most recently placed zone wins)."""
        for z in reversed(self.zones):
            if (x - z.x) ** 2 + (y - z.y) ** 2 <= z.r ** 2:
                return z.odor
        return None


class OdorHandle:
    """Session glue: sense -> brain, whip -> punishment, HUD / summary."""

    def __init__(self, session: "Session", zones: OdorZones, cfg: OdorConfig,
                 say: Callable[[str], None] | None = None) -> None:
        self.session, self.zones, self.cfg = session, zones, cfg
        self.say = say or (lambda m: None)
        self.current: str | None = None
        self.last_odor: str | None = None
        self.last_in_t = -1e9
        self._last_send = -1e9
        self.time_in = {o: 0.0 for o in ODORS}
        self.n_entries = {o: 0 for o in ODORS}
        self.n_punish = {o: 0 for o in ODORS}
        self.n_hits_outside = 0
        self.sent: list[StimulusEvent] = []  # onset + punishment events (tests)
        self._last_t: float | None = None
        if session.whip is not None:
            session.whip.listeners.append(self.on_whip)
        session.sim.reset_hooks.append(lambda s: self._on_reset())

    def _on_reset(self) -> None:
        self.current = None
        self._last_t = None

    def _thorax(self) -> tuple[float, float]:
        sim = self.session.sim
        p = sim.data.xpos[sim.thorax_body_id]
        return float(p[0]), float(p[1])

    def _send(self, ev: StimulusEvent, log: bool) -> None:
        link = self.session.brain
        if link is None:
            return
        if log:
            self.sent.append(ev)
            link.send(ev, source="odor")
        elif link.brain is not None and link.brain.is_alive():
            try:
                link.brain.send(ev)
            except Exception:
                pass

    def update(self) -> None:
        """Once per physics chunk."""
        rt = self.session.run_time()
        dt = 0.0 if self._last_t is None else max(rt - self._last_t, 0.0)
        self._last_t = rt
        odor = self.zones.odor_at(*self._thorax())
        if odor is not None:
            self.time_in[odor] = self.time_in.get(odor, 0.0) + dt
            self.last_in_t, self.last_odor = rt, odor
        onset = odor != self.current
        if onset and odor is not None:
            self.n_entries[odor] = self.n_entries.get(odor, 0) + 1
        self.current = odor
        if odor is None:
            return
        if onset or rt - self._last_send >= self.cfg.refresh_s:
            self._last_send = rt
            det = {"set": f"odor_{odor}", "odor": odor, "proxy": "KC code (plasticity.py)"}
            if self.cfg.rate_hz > 0:
                det["rate_hz"] = float(self.cfg.rate_hz)
            else:
                det["rate_hz"] = self._brain_cfg("odor_hz", 30.0)
            self._send(StimulusEvent("manual", "none", 1.0, self.cfg.stim_duration_s, rt,
                                     details=det), log=onset)

    def _brain_cfg(self, key: str, default: float) -> float:
        link = self.session.brain
        pc = getattr(link, "plasticity_dict", lambda: None)() if link is not None else None
        from perpetualfly.brain.plasticity import PlasticityConfig

        return float((pc or {}).get(key, getattr(PlasticityConfig(), key, default)))

    def on_whip(self, ev) -> None:
        if not getattr(ev, "hit", False):
            return
        self.punish(source="whip")

    def punish(self, source: str = "api") -> str:
        """Punishment DANs if the fly is (or just was) in a zone and learning is on."""
        if not self.cfg.learning:
            return "[odor] learning off (--learning)"
        rt = self.session.run_time()
        odor = self.current or (self.last_odor if rt - self.last_in_t <= self.cfg.punish_window_s
                                else None)
        if odor is None:
            self.n_hits_outside += 1
            return "[odor] hit outside any odour zone: no punishment"
        rate = self.cfg.punish_hz or self._brain_cfg("punish_hz", 200.0)
        self._send(StimulusEvent("manual", "none", 1.0, self.cfg.punish_duration_s, rt,
                                 details={"set": PUNISH_SET, "rate_hz": float(rate),
                                          "odor": odor, "source": source,
                                          "proxy": "PPL1 DANs (whip stand-in)"}), log=True)
        self.n_punish[odor] = self.n_punish.get(odor, 0) + 1
        return f"[odor] punishment in odour {odor} (PPL1 DANs {rate:.0f} Hz, stand-in)"

    def spawn(self, odor: str) -> str:
        x, y = self._thorax()
        self.zones.place(odor, x, y)
        return f"[odor] odour {odor} zone around the fly"

    def hud_line(self) -> str:
        st = self.session.brain.latest if self.session.brain is not None else None
        lr = (getattr(st, "learning", None) or {}) if st is not None else {}
        eff = lr.get("efficacy_by_odor") or {}
        main = lr.get("main") or ""
        mem = "  ".join(f"{o} {float((eff.get(o) or {}).get(main, 1.0)):.2f}" for o in ODORS
                        if o in eff)
        here = f"in {self.current}" if self.current else "no odour"
        tail = (f"  KC>MBON({main}) {mem}" if mem else "") + (
            "  learning ON" if self.cfg.learning else "")
        return (f"ODOUR {here}  punish A{self.n_punish.get('A', 0)} "
                f"B{self.n_punish.get('B', 0)}{tail}")

    def summary(self) -> dict:
        return {"zones": [(z.odor, round(z.x, 2), round(z.y, 2), z.r) for z in self.zones.zones],
                "time_in_s": {k: round(v, 3) for k, v in self.time_in.items()},
                "entries": dict(self.n_entries), "punishments": dict(self.n_punish),
                "hits_outside": self.n_hits_outside, "learning": self.cfg.learning}


def install_odor(session: "Session", zones: OdorZones, cfg: OdorConfig | None = None,
                 say: Callable[[str], None] | None = None) -> OdorHandle:
    cfg = cfg or zones.cfg
    zones.attach(session.sim)
    return OdorHandle(session, zones, cfg, say=say)
