"""Taste patches: sugar and bitter spots on the ground, tasted with the legs
(``--taste-patches``; docs/TASTE.md).

Three parts, all optional and off by default:

* ``TastePatches``: coloured discs on the ground (sugar = pale yellow with a
  sheen, bitter = dark olive-brown and matte, mixed = amber). They are a small
  pool of MuJoCo cylinders with ``contype = conaffinity = 0``, added by a world
  extension, so they are **visual only**: they never collide and change no
  physics. Patches are placed procedurally in cells along the fly's path (content
  is a pure function of the seed and the cell index, with a configurable
  density), recycled as the fly moves, and can be spawned ahead with keys 5 / 6 /
  7 (sugar / bitter / mixed).
* ``TasteSense``: every ``sense_every_s`` of sim time, the legs that touch
  something (tibia / tarsus contacts, the same test the action library uses) and
  whose tarsus tip lies inside a patch disc taste it. While any leg is on a patch
  the sensor sends ``StimulusEvent("taste", ...)`` to the brain, refreshed every
  ``refresh_s`` so the drive is sustained while the contact lasts and stops about
  ``stim_duration_s`` after it ends. Rate = ``max_rate_hz * min(n_legs /
  legs_full, 1)`` per taste.

  **Which neurons.** Leg (tarsal) gustatory neurons project to the VNC, which the
  brain model does not contain. The model's own leg gustatory afferents that
  ascend straight to the brain (``SA_VTV_pro_meso_meta``, cell class gustatory,
  37 per side, taste modality not annotated) reach no MN9 and nothing useful (and
  the left set ignites the model's runaway state at 100 Hz; docs/TASTE.md), so by
  default taste drives the **labellar GRNs of Shiu et al. 2024 as a stand-in**:
  the 20 sugar GRNs (LB3, ``mapping.SUGAR_GRN_IDS``) and the 20 bitter GRNs
  (LB1a-d, ``mapping.BITTER_GRN_IDS``). Both sets are on the fly's left labellar
  nerve (the right LB3 population ignites the runaway state at >= 100 Hz), so the
  touching side is recorded in the event but does not change which GRNs are
  driven. ``leg_afferent_hz > 0`` additionally drives the ascending leg
  gustatory afferents on the touching side.
* ``FeedingRule`` + ``Feed``: with ``--brain-actions`` the real brain's MN9
  (proboscis motor neuron) decides. Our rule (the MN9 activity is the model's;
  the rule is ours): while the fly tastes sugar (a sugar contact within
  ``feed_taste_window_s``) and MN9 exceeds ``feed_mn9_hz``, the fly stops walking
  (freeze stance, all tarsi adhering) and extends the proboscis (full body).
  Feeding ends when MN9 has been below the threshold for ``feed_release_s`` (e.g.
  bitter mixed in suppresses MN9, or the sugar contact ended) or after
  ``feed_max_s`` (a satiation stand-in); the fly then walks on and cannot start
  another bout for ``feed_refractory_s``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Callable

import mujoco as mj
import numpy as np

from perpetualfly.actions.base import LEGS, ActionCommand
from perpetualfly.actions.behaviours import PROBOSCIS_DOFS, Freeze, _extra_actuators
from perpetualfly.brain.schema import StimulusEvent

if TYPE_CHECKING:
    from perpetualfly.app import Session

PATCH_KINDS = ("sugar", "bitter", "mixed")
PATCH_TASTES: dict[str, tuple[str, ...]] = {
    "sugar": ("sugar",), "bitter": ("bitter",), "mixed": ("sugar", "bitter")}
PATCH_COLORS: dict[str, tuple[float, float, float, float]] = {
    "sugar": (1.00, 0.94, 0.62, 1.0),   # pale yellow / white sugar with a sheen
    "bitter": (0.25, 0.30, 0.10, 1.0),  # dark olive-brown, matte
    "mixed": (0.78, 0.56, 0.20, 1.0),   # amber: sugar laced with bitter
}
TASTES = ("sugar", "bitter")
PREFIX = "taste_patch"
PARK_POS = (0.0, 0.0, -200.0)
PARK_SIZE = 0.01
HALF_HEIGHT = 0.015  # mm; disc half-thickness (drawn just above the ground)
LIFT = 0.02  # mm; disc centre above the ground (thinner / lower discs z-fight with the floor)
# key -> patch kind (keys 5 / 6 / 7 are free in the app; docs/TASTE.md)
TASTE_KEYS: dict[str, str] = {"5": "sugar", "6": "bitter", "7": "mixed"}


@dataclass
class TasteConfig:
    """``--taste-patches`` (AppConfig.taste)."""

    enabled: bool = False
    seed: int = 7
    # ---- procedural patches ------------------------------------------------------
    density_per_cm: float = 0.3  # expected patches per 10 mm of path (0 = only spawned)
    cell_mm: float = 20.0  # placement cell length along x
    band_mm: float = 3.0  # lateral half-width of the band around the fly's path
    radius_mm: tuple[float, float] = (1.8, 2.6)  # patch radius range
    kind_weights: dict = field(default_factory=lambda: {"sugar": 0.5, "bitter": 0.3,
                                                        "mixed": 0.2})
    first_x_mm: float = 8.0  # no procedural patch starts before this x (start area)
    ahead_mm: float = 40.0  # cells loaded ahead of / behind the thorax
    behind_mm: float = 15.0
    pool_size: int = 24  # procedural patch geoms
    spawn_pool: int = 6  # key-spawned patch geoms
    spawn_distance_mm: float = 3.0  # near edge of a spawned patch ahead of the thorax
    lateral_snap_mm: float = 1.0
    # ---- sensing -------------------------------------------------------------------
    sense_every_s: float = 0.01  # sim s between contact checks
    refresh_s: float = 0.1  # re-send the unchanged taste stimulus this often
    stim_duration_s: float = 0.15  # each event's drive (> refresh_s: sustained)
    contact_height_mm: float = 0.3  # tarsus tip within this height above the patch
    leg_hold_s: float = 0.1  # a leg keeps tasting this long after lifting off (swing)
    legs_full: int = 3  # legs on a patch for the full rate
    max_rate_hz: float = 200.0
    leg_afferent_hz: float = 0.0  # also drive SA_VTV leg gustatory afferents (0 = off)
    # ---- feeding rule (--brain-actions) --------------------------------------------
    feed: bool = True
    feed_mn9_hz: float = 30.0
    feed_release_s: float = 0.3
    feed_max_s: float = 4.0
    feed_refractory_s: float = 2.5
    feed_taste_window_s: float = 0.3

    @classmethod
    def from_dict(cls, d: dict) -> "TasteConfig":
        d = dict(d)
        known = set(cls.__dataclass_fields__)
        bad = set(d) - known
        if bad:
            raise ValueError(f"Unknown config keys for TasteConfig: {sorted(bad)}")
        if isinstance(d.get("radius_mm"), list):
            d["radius_mm"] = tuple(d["radius_mm"])
        return cls(**d)


@dataclass
class Patch:
    x: float
    y: float
    r: float
    kind: str
    z: float = 0.0
    gid: int = -1
    spawned: bool = False

    @property
    def tastes(self) -> tuple[str, ...]:
        return PATCH_TASTES[self.kind]


# ============================================================================ patches
class TastePatches:
    """Visual taste patches on a geom pool (see the module docstring)."""

    def __init__(self, cfg: TasteConfig | None = None) -> None:
        self.cfg = cfg or TasteConfig()
        self.sim = None
        self.ground_fn: Callable[[float, float], float] = lambda x, y: 0.0
        self._pool: list[int] = []
        self._spawn_pool: list[int] = []
        self._free: list[int] = []
        self._spawn_free: list[int] = []
        self._cells: dict[int, list[Patch]] = {}
        self._cell_y: dict[int, float] = {}
        self.spawned: list[Patch] = []
        self.spawn_count = 0
        self.n_dropped = 0  # procedural patches not shown (pool exhausted)
        self._mat = {"sheen": -1, "matte": -1}
        self._arr = np.zeros((0, 5))  # x, y, r, z, kind index (cache for sensing)
        self._patches: list[Patch] = []
        self._dirty = True

    # ------------------------------------------------------------- construction
    def extension(self, world) -> None:
        """World extension: the patch pool (before add_fly). contype = conaffinity =
        0 -> never collides; group 0 -> rendered."""
        spec: mj.MjSpec = world.mjcf_root
        spec.add_material(name=f"{PREFIX}_sheen", specular=0.9, shininess=0.9, reflectance=0.0,
                          emission=0.35)
        spec.add_material(name=f"{PREFIX}_matte", specular=0.05, shininess=0.05, reflectance=0.0)
        n = self.cfg.pool_size + self.cfg.spawn_pool
        for j in range(n):
            spec.worldbody.add_geom(
                name=f"{PREFIX}{j}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                size=(PARK_SIZE, PARK_SIZE, 0), pos=PARK_POS, rgba=(1, 1, 1, 0),
                contype=0, conaffinity=0, group=0, material=f"{PREFIX}_matte")

    def attach(self, sim, ground_fn: Callable[[float, float], float] | None = None) -> "TastePatches":
        self.sim = sim
        if ground_fn is not None:
            self.ground_fn = ground_fn
        m = sim.model
        ids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, f"{PREFIX}{j}")
               for j in range(self.cfg.pool_size + self.cfg.spawn_pool)]
        if any(i < 0 for i in ids):
            raise RuntimeError("taste patch geoms missing: add TastePatches.extension to the "
                               "Simulation's world_extensions")
        self._pool = ids[: self.cfg.pool_size]
        self._spawn_pool = ids[self.cfg.pool_size:]
        for k in self._mat:
            self._mat[k] = mj.mj_name2id(m, mj.mjtObj.mjOBJ_MATERIAL, f"{PREFIX}_{k}")
        sim.reset_hooks.append(lambda s: self.relayout())
        self.relayout()
        return self

    # ------------------------------------------------------------- procedural
    def cell_patches(self, index: int, y_center: float = 0.0) -> list[Patch]:
        """Patches of cell ``index`` (pure function of seed, index, y_center)."""
        c = self.cfg
        rng = np.random.default_rng([c.seed, 4099, index & 0xFFFFFFFF])
        lam = max(c.density_per_cm, 0.0) * c.cell_mm / 10.0
        n = int(rng.poisson(lam)) if lam > 0 else 0
        kinds = [k for k in PATCH_KINDS if c.kind_weights.get(k, 0) > 0]
        w = np.array([c.kind_weights[k] for k in kinds], dtype=float)
        out: list[Patch] = []
        x0 = index * c.cell_mm
        for _ in range(n):
            for _try in range(8):
                r = float(rng.uniform(*c.radius_mm))
                x = float(rng.uniform(x0 + r, x0 + c.cell_mm - r))
                y = float(y_center + rng.uniform(-c.band_mm, c.band_mm))
                kind = kinds[int(rng.choice(len(kinds), p=w / w.sum()))] if kinds else "sugar"
                if x - r < c.first_x_mm:
                    continue
                if all(math.hypot(x - p.x, y - p.y) > r + p.r + 0.5 for p in out):
                    out.append(Patch(x, y, r, kind))
                    break
        return out

    def _fly_xy(self) -> tuple[float, float]:
        p = self.sim.data.xpos[self.sim.thorax_body_id]
        return float(p[0]), float(p[1])

    def relayout(self) -> None:
        """Park everything (spawned patches dropped) and load around the fly."""
        for gid in self._pool + self._spawn_pool:
            self._park(gid)
        self._free = list(self._pool)
        self._spawn_free = list(self._spawn_pool)
        self._cells.clear()
        self._cell_y.clear()
        self.spawned.clear()
        self._dirty = True
        self.update()

    def update(self) -> None:
        """Load / release cells around the thorax (cheap when nothing changes)."""
        if self.sim is None:
            return
        c = self.cfg
        fx, fy = self._fly_xy()
        lo = math.floor((fx - c.behind_mm) / c.cell_mm)
        hi = math.floor((fx + c.ahead_mm) / c.cell_mm)
        want = set(range(lo, hi + 1))
        for idx in [i for i in self._cells if i not in want]:
            for p in self._cells.pop(idx):
                if p.gid >= 0:
                    self._park(p.gid)
                    self._free.append(p.gid)
            self._cell_y.pop(idx, None)
            self._dirty = True
        snap = c.lateral_snap_mm
        yc = round(fy / snap) * snap if snap > 0 else 0.0
        for idx in sorted(want - set(self._cells), key=lambda i: abs(i * c.cell_mm - fx)):
            y0 = self._cell_y.setdefault(idx, yc)
            ps = self.cell_patches(idx, y0)
            for p in ps:
                if not self._free:
                    self.n_dropped += 1
                    continue
                p.gid = self._free.pop()
                self._place(p)
            self._cells[idx] = ps
            self._dirty = True
        behind = fx - c.behind_mm - c.cell_mm
        for p in [p for p in self.spawned if p.x + p.r < behind]:
            self._release_spawn(p)

    def refresh_heights(self) -> None:
        """Re-seat every patch on the current ground (terrain chunks recycle)."""
        for p in self.patches():
            z = float(self.ground_fn(p.x, p.y))
            if abs(z - p.z) > 1e-6:
                p.z = z
                self._write_pose(p)
                self._dirty = True

    # ------------------------------------------------------------- spawning
    def spawn_ahead(self, kind: str, distance: float | None = None,
                    radius: float | None = None, lateral: float = 0.0) -> Patch:
        """A patch whose near edge is ``distance`` mm ahead of the thorax along the
        fly's heading (``lateral`` mm to its left). The oldest spawned patch is
        recycled when the spawn pool is full."""
        if kind not in PATCH_KINDS:
            raise ValueError(f"unknown patch kind {kind!r}; choose from {PATCH_KINDS}")
        if self.sim is None:
            raise RuntimeError("attach(sim) first")
        c = self.cfg
        d = c.spawn_distance_mm if distance is None else float(distance)
        r = float(radius) if radius is not None else float(np.mean(c.radius_mm))
        fx, fy = self._fly_xy()
        h = float(self.sim.heading())
        fwd = np.array([math.cos(h), math.sin(h)])
        left = np.array([-fwd[1], fwd[0]])
        xy = np.array([fx, fy]) + (d + r) * fwd + lateral * left
        if not self._spawn_free:
            self._release_spawn(self.spawned[0])
        p = Patch(float(xy[0]), float(xy[1]), r, kind, spawned=True)
        p.gid = self._spawn_free.pop(0)
        self._place(p)
        self.spawned.append(p)
        self.spawn_count += 1
        self._dirty = True
        return p

    def place(self, kind: str, x: float, y: float, r: float) -> Patch:
        """A spawned patch at an absolute position (scripts / tests)."""
        if not self._spawn_free:
            self._release_spawn(self.spawned[0])
        p = Patch(float(x), float(y), float(r), kind, spawned=True)
        p.gid = self._spawn_free.pop(0)
        self._place(p)
        self.spawned.append(p)
        self.spawn_count += 1
        self._dirty = True
        return p

    def _release_spawn(self, p: Patch) -> None:
        self.spawned.remove(p)
        self._park(p.gid)
        self._spawn_free.append(p.gid)
        self._dirty = True

    # ------------------------------------------------------------- queries
    def patches(self) -> list[Patch]:
        return [p for ps in self._cells.values() for p in ps if p.gid >= 0] + list(self.spawned)

    def arrays(self) -> tuple[np.ndarray, list[Patch]]:
        """(n, 4) [x, y, r, z] of the shown patches + the Patch list (cached)."""
        if self._dirty:
            self._patches = self.patches()
            self._arr = np.array([[p.x, p.y, p.r, p.z] for p in self._patches],
                                 dtype=float).reshape(-1, 4)
            self._dirty = False
        return self._arr, self._patches

    def patch_at(self, x: float, y: float) -> list[Patch]:
        arr, ps = self.arrays()
        if not len(ps):
            return []
        inside = (arr[:, 0] - x) ** 2 + (arr[:, 1] - y) ** 2 <= arr[:, 2] ** 2
        return [ps[i] for i in np.flatnonzero(inside)]

    # ------------------------------------------------------------- geom writes
    def _place(self, p: Patch) -> None:
        p.z = float(self.ground_fn(p.x, p.y))
        m = self.sim.model
        g = p.gid
        m.geom_size[g] = (p.r, HALF_HEIGHT, 0.0)
        m.geom_rbound[g] = math.hypot(p.r, HALF_HEIGHT)
        m.geom_aabb[g] = (0.0, 0.0, 0.0, p.r, p.r, HALF_HEIGHT)
        m.geom_rgba[g] = PATCH_COLORS[p.kind]
        mat = self._mat["matte" if p.kind == "bitter" else "sheen"]
        if mat >= 0:
            m.geom_matid[g] = mat
        self._write_pose(p)

    def _write_pose(self, p: Patch) -> None:
        m, d = self.sim.model, self.sim.data
        pos = (p.x, p.y, p.z + LIFT)
        m.geom_pos[p.gid] = pos
        m.geom_quat[p.gid] = (1.0, 0.0, 0.0, 0.0)
        d.geom_xpos[p.gid] = pos  # render before the next step already shows it
        d.geom_xmat[p.gid] = (1, 0, 0, 0, 1, 0, 0, 0, 1)

    def _park(self, gid: int) -> None:
        m, d = self.sim.model, self.sim.data
        m.geom_pos[gid] = PARK_POS
        m.geom_size[gid] = (PARK_SIZE, PARK_SIZE, 0.0)
        m.geom_rbound[gid] = PARK_SIZE * math.sqrt(2)
        m.geom_aabb[gid] = (0.0, 0.0, 0.0, PARK_SIZE, PARK_SIZE, PARK_SIZE)
        m.geom_rgba[gid, 3] = 0.0
        d.geom_xpos[gid] = PARK_POS


# ============================================================================ sensing
@dataclass
class TasteReading:
    """Legs (LEGS indices) on each taste at one sense tick."""

    sugar: tuple[int, ...] = ()
    bitter: tuple[int, ...] = ()

    def legs(self, taste: str) -> tuple[int, ...]:
        return getattr(self, taste)

    @property
    def any(self) -> bool:
        return bool(self.sugar or self.bitter)

    def side(self) -> str:
        legs = set(self.sugar) | set(self.bitter)
        left = any(LEGS[i][0] == "l" for i in legs)
        right = any(LEGS[i][0] == "r" for i in legs)
        return "left" if left and not right else "right" if right and not left else "none"

    def names(self, taste: str) -> list[str]:
        return [LEGS[i] for i in self.legs(taste)]


class TasteSense:
    """Legs on patches -> ``taste`` StimulusEvents (see the module docstring).

    ``send(ev, refresh)`` delivers an event: ``refresh`` False on onset / change
    (logged, shown in the brain window), True for the periodic re-send of an
    unchanged contact (brain only)."""

    def __init__(self, sim, patches: TastePatches, cfg: TasteConfig,
                 send: Callable[[StimulusEvent, bool], None] | None = None,
                 time_fn: Callable[[], float] | None = None,
                 leg_contacts: Callable[[], np.ndarray] | None = None) -> None:
        self.sim, self.patches, self.cfg = sim, patches, cfg
        self.send = send
        self.time_fn = time_fn or (lambda: float(sim.time))
        m = sim.model
        self._tarsus = np.array([mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY,
                                               f"{sim.fly_name}/{leg}_tarsus5") for leg in LEGS])
        if leg_contacts is None:
            from perpetualfly.actions.base import BodyIndex

            body = BodyIndex(sim)
            leg_contacts = lambda: body.ground_contacts(sim)[0]  # noqa: E731
        self.leg_contacts = leg_contacts
        self.reading = TasteReading()
        self.last_sugar_t = -1e9  # run time of the last sugar contact
        self.last_bitter_t = -1e9
        self._last_send = -1e9
        self._last_sig: tuple | None = None
        self._last_tastes: tuple = ()
        self.n_events = 0  # onset / change events
        self.n_refresh = 0
        self.contact_s = {"sugar": 0.0, "bitter": 0.0}  # run seconds with >= 1 leg on it
        self._last_t: float | None = None
        self.sent: list[StimulusEvent] = []  # onset / change events (tests / summary)
        # run time each leg last touched each taste (leg_hold_s bridges swing phases)
        self._leg_t = {t: np.full(6, -1e9) for t in TASTES}

    def read(self) -> TasteReading:
        """Which legs taste what right now."""
        arr, ps = self.patches.arrays()
        if not len(ps):
            return TasteReading()
        touch = np.asarray(self.leg_contacts(), dtype=bool)
        if not touch.any():
            return TasteReading()
        tips = self.sim.data.xpos[self._tarsus]
        out: dict[str, list[int]] = {"sugar": [], "bitter": []}
        for leg in np.flatnonzero(touch):
            x, y, z = tips[leg]
            d2 = (arr[:, 0] - x) ** 2 + (arr[:, 1] - y) ** 2
            hit = np.flatnonzero((d2 <= arr[:, 2] ** 2)
                                 & (z - arr[:, 3] <= self.cfg.contact_height_mm))
            tastes = {t for i in hit for t in ps[i].tastes}
            for t in tastes:
                out[t].append(int(leg))
        return TasteReading(tuple(out["sugar"]), tuple(out["bitter"]))

    def rate(self, n_legs: int) -> float:
        c = self.cfg
        return float(c.max_rate_hz * min(n_legs / max(c.legs_full, 1), 1.0))

    def event(self, rd: TasteReading, now: float) -> StimulusEvent:
        c = self.cfg
        tastes = [t for t in TASTES if rd.legs(t)]
        det: dict = {"tastes": tastes, "proxy": "labellar GRNs (Shiu et al. 2024)"}
        parts = []
        for t in tastes:
            det[f"{t}_hz"] = round(self.rate(len(rd.legs(t))), 1)
            det[f"{t}_legs"] = rd.names(t)
            parts.append(f"{t.upper()} {len(rd.legs(t))}")
        if c.leg_afferent_hz > 0:
            det["leg_hz"] = float(c.leg_afferent_hz)
        det["label"] = "TASTE " + " + ".join(parts)
        n = max(len(rd.sugar), len(rd.bitter))
        return StimulusEvent("taste", side=rd.side(), intensity=min(n / max(c.legs_full, 1), 1.0),
                             duration_s=c.stim_duration_s, sim_time=now, details=det)

    def tick(self) -> TasteReading:
        now = self.time_fn()
        raw = self.read()
        hold = self.cfg.leg_hold_s
        legs = {}
        for t in TASTES:
            lt = self._leg_t[t]
            lt[list(raw.legs(t))] = now
            legs[t] = tuple(int(i) for i in np.flatnonzero(now - lt <= hold + 1e-9))
        rd = TasteReading(legs["sugar"], legs["bitter"])
        if self._last_t is not None and now > self._last_t:
            dt = now - self._last_t
            for t in TASTES:
                if self.reading.legs(t):
                    self.contact_s[t] += dt
        self._last_t = now
        self.reading = rd
        if rd.sugar:
            self.last_sugar_t = now
        if rd.bitter:
            self.last_bitter_t = now
        # onset / a taste added or lost -> logged event; a new leg count or side ->
        # immediate brain-only update; otherwise re-sent every refresh_s
        tastes = tuple(t for t in TASTES if rd.legs(t))
        sig = (len(rd.sugar), len(rd.bitter), rd.side())
        if tastes:
            changed = tastes != self._last_tastes
            if (changed or sig != self._last_sig
                    or now - self._last_send >= self.cfg.refresh_s - 1e-9):
                ev = self.event(rd, now)
                self._last_send = now
                if changed:
                    self.n_events += 1
                    self.sent.append(ev)
                else:
                    self.n_refresh += 1
                if self.send is not None:
                    self.send(ev, not changed)
        self._last_sig = sig
        self._last_tastes = tastes
        return rd


# ============================================================================ feeding
class Feed(Freeze):
    """Feeding stop: the freeze stance (all tarsi adhering) plus proboscis extension
    when the body has the proboscis joints. Ended by ``FeedingRule`` (``done``)."""

    name = "feed"
    blend_out = 0.25

    def __init__(self, duration: float = 4.0, proboscis: bool = True) -> None:
        super().__init__(duration)
        self.proboscis = proboscis
        self._extra: dict[int, float] = {}
        self.reason = ""

    def begin(self, mgr) -> None:
        super().begin(mgr)
        self._extra = {}
        if self.proboscis:
            try:
                ids = _extra_actuators(mgr, PROBOSCIS_DOFS, "proboscispos")
            except RuntimeError:
                ids = {}
            pose = {"c_head-c_rostrum-pitch": -1.0, "c_rostrum-c_haustellum-pitch": 1.0}
            self._extra = {ids[k]: v for k, v in pose.items() if k in ids}

    def command(self, mgr, t: float) -> ActionCommand:
        cmd = super().command(mgr, t)
        cmd.extra = self._extra
        return cmd

    def end(self, mgr, cancelled: bool) -> None:
        super().end(mgr, cancelled)
        self.info["reason"] = self.reason or ("cancelled" if cancelled else "max_bout")
        self.info["proboscis"] = bool(self._extra)


class FeedingRule:
    """MN9 (real brain) + sugar contact -> ``Feed`` (see the module docstring).

    ``update(now, mn9_hz, sugar_recent)`` once per sense tick (sim time ``now``)."""

    def __init__(self, mgr, cfg: TasteConfig, can_proboscis: bool = True,
                 say: Callable[[str], None] | None = None) -> None:
        self.mgr, self.cfg = mgr, cfg
        self.can_proboscis = can_proboscis
        self.say = say or (lambda msg: None)
        self.action: Feed | None = None
        self._t_start = 0.0
        self._last_high = -1e9
        self.next_allowed = -1e9
        self.bouts: list[dict] = []  # {"start", "end", "duration_s", "reason", "mn9_max"}
        self._mn9_max = 0.0
        mgr.listeners.append(self._on_event)

    @property
    def feeding(self) -> bool:
        a = self.action
        return a is not None and (self.mgr.action is a or self.mgr._pending is a)

    def reset(self) -> None:
        self.action = None
        self.next_allowed = -1e9
        self._last_high = -1e9

    def _on_event(self, ev) -> None:
        if ev.name == "feed" and ev.kind in ("end", "cancel") and self.action is not None:
            now = self.mgr.sim.time
            b = {"start": round(self._t_start, 3), "end": round(now, 3),
                 "duration_s": round(now - self._t_start, 3),
                 "reason": ev.info.get("reason", ev.kind), "mn9_max": round(self._mn9_max, 1)}
            self.bouts.append(b)
            self.action = None
            wait = self.cfg.feed_refractory_s if b["reason"] == "max_bout" else 0.5
            self.next_allowed = now + wait
            self.say(f"[taste] feeding ended t={now:.2f}s after {b['duration_s']:.2f}s "
                     f"({b['reason']}); walking on")

    def update(self, now: float, mn9_hz: float, sugar_recent: bool,
               blocked: bool = False) -> str | None:
        c = self.cfg
        high = mn9_hz > c.feed_mn9_hz
        if self.feeding:
            a = self.action
            self._mn9_max = max(self._mn9_max, mn9_hz)
            if high:
                self._last_high = now
            if now - self._last_high >= c.feed_release_s and not a.done:
                a.reason = "mn9_low"
                a.done = True
            return None
        if self.action is not None:  # replaced / cancelled without our event
            self.action = None
        if not (high and sugar_recent) or blocked or now < self.next_allowed:
            return None
        name = self.mgr.active_name
        if name not in (None, "proboscis"):
            return None  # never interrupts another action (jump, groom, ...)
        feed = Feed(duration=c.feed_max_s, proboscis=self.can_proboscis)
        if not self.mgr.trigger(feed, replace=True, source="brain"):
            return None
        self.action = feed
        self._t_start = now
        self._last_high = now
        self._mn9_max = mn9_hz
        msg = (f"[taste] t={now:.2f}s sugar + MN9 {mn9_hz:.0f} Hz > {c.feed_mn9_hz:g} -> "
               f"FEEDING (stop{' + proboscis' if self.can_proboscis else ''})")
        self.say(msg)
        return msg


# ============================================================================ app glue
class TasteHandle:
    """What ``install_taste`` returns: patches + sense (+ feeding rule)."""

    def __init__(self, session: "Session", cfg: TasteConfig, patches: TastePatches,
                 say: Callable[[str], None] | None = None) -> None:
        self.session, self.cfg, self.patches = session, cfg, patches
        self.say = say or (lambda msg: None)
        sim = session.sim
        link = session.brain
        self.link = link
        self.sense = TasteSense(sim, patches, cfg, send=self._send, time_fn=session.run_time,
                                leg_contacts=lambda: session.actions.body.ground_contacts(sim)[0])
        self.rule: FeedingRule | None = None
        if cfg.feed and link is not None and getattr(link, "triggers", None) is not None:
            self.rule = FeedingRule(session.actions, cfg,
                                    can_proboscis="proboscis" in session.available_actions,
                                    say=self.say)
        self._next_sense = -1e9
        self._next_heights = -1e9
        self.mn9 = 0.0
        self._prev_any = False
        sim.reset_hooks.append(self._on_reset)

    def _on_reset(self, sim) -> None:
        self._next_sense = self._next_heights = -1e9
        if self.rule is not None:
            self.rule.reset()

    def _send(self, ev: StimulusEvent, refresh: bool) -> None:
        link = self.link
        if link is None:
            return
        if refresh:  # unchanged contact: brain only (no window chip / events.csv row)
            b = link.brain
            if b is not None and b.is_alive():
                try:
                    b.send(ev)
                except Exception:
                    pass
        else:
            link.send(ev, source="taste")

    def _mn9_now(self, rt: float) -> float:
        """MN9 rate of the newest BrainState (window_s, 0.1 s by default). The
        fast-path MN9 triggers (20 ms window: one spike = 50 Hz) are deliberately
        not used: in the sugar + bitter mixture single MN9 spikes would start
        0.3 s "feeding" twitches although the rate stays below threshold."""
        link = self.link
        if link is None:
            return 0.0
        st = link.latest
        if st is None or st.sim_time is None or rt - float(st.sim_time) > 0.3:
            return 0.0
        return float((st.probes or {}).get("MN9", 0.0) or 0.0)

    def update(self) -> None:
        """Once per physics chunk (Session.after_physics, after the brain update)."""
        s = self.session
        rt = s.run_time()
        if rt < self._next_sense:
            return
        self._next_sense = rt + self.cfg.sense_every_s
        self.patches.update()
        if rt >= self._next_heights:
            self._next_heights = rt + 0.2
            self.patches.refresh_heights()
        prev = self.sense.reading
        rd = self.sense.tick()
        if rd.any != self._prev_any or (rd.any and (rd.sugar, rd.bitter) != (prev.sugar,
                                                                           prev.bitter)):
            s.log_event("taste" if rd.any else "taste_off",
                        sugar_legs=rd.names("sugar"), bitter_legs=rd.names("bitter"))
        self._prev_any = rd.any
        self.mn9 = self._mn9_now(rt)
        if self.rule is not None:
            flying = s.flight is not None and s.flight.busy
            recent = rt - self.sense.last_sugar_t <= self.cfg.feed_taste_window_s
            self.rule.update(s.sim.time, self.mn9, recent, blocked=flying)

    def spawn(self, kind: str) -> str:
        p = self.patches.spawn_ahead(kind)
        self.session.log_event("taste_spawn", kind=kind, x=round(p.x, 2), y=round(p.y, 2),
                               r=round(p.r, 2))
        return (f"[taste] {kind} patch spawned {self.cfg.spawn_distance_mm:g} mm ahead "
                f"(r {p.r:.1f} mm)")

    def hud_line(self) -> str:
        rd = self.sense.reading
        parts = []
        for t in TASTES:
            legs = rd.legs(t)
            if legs:
                parts.append(f"{t} {len(legs)} leg{'s' if len(legs) > 1 else ''} "
                             f"{self.sense.rate(len(legs)):.0f}Hz")
        txt = "TASTE " + (", ".join(parts) if parts else "-")
        txt += "  (labellar GRN stand-in)"
        if self.link is not None:
            txt += f"  MN9 {self.mn9:.0f}Hz"
        r = self.rule
        if r is not None and r.feeding:
            txt += f"  FEEDING {self.session.sim.time - r._t_start:.1f}s"
        elif r is not None and self.session.sim.time < r.next_allowed:
            txt += "  (fed)"
        return txt

    def summary(self) -> dict:
        r = self.rule
        return {"patches_spawned": self.patches.spawn_count,
                "patches_dropped": self.patches.n_dropped,
                "taste_events": self.sense.n_events, "taste_refreshes": self.sense.n_refresh,
                "contact_s": {k: round(v, 3) for k, v in self.sense.contact_s.items()},
                "feeding_bouts": list(r.bouts) if r is not None else None,
                "config": asdict(self.cfg)}


def install_taste(session: "Session", patches: TastePatches, cfg: TasteConfig | None = None,
                  say: Callable[[str], None] | None = None) -> TasteHandle:
    """Attach the patches (built into the world by ``patches.extension``) and the
    sense / feeding rule to a Session. The session calls ``handle.update()`` after
    every physics chunk."""
    cfg = cfg or patches.cfg
    patches.attach(session.sim, session.ground_height)
    return TasteHandle(session, cfg, patches, say=say)
