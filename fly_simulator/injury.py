"""Squash damage: a phenomenological injury model (``--injury``; docs/INJURY.md).

A swatted, whipped or crushed fly should not walk on unharmed. This module is a
**phenomenological model, not a biomechanical damage model**: there is no tissue,
cuticle or haemolymph mechanics. Measured contact forces are turned into a
"damage" number per body part with a force floor and a reference impulse
(calibrated against the swatter / whip / fall numbers, docs/INJURY.md), and the
damage is mapped onto a few bounded, reversible control and model changes.

Parts: the six legs (coxa .. tarsus5), thorax (+ head), abdomen, left / right wing.

* **Damage** (post-step hook, every physics step): for every contact between a fly
  geom and a non-fly geom (tarsus-on-static-ground contacts are normal walking and
  are skipped) the world-frame contact force (``mj_contactForce``) is summed per
  part. ``D_part += max(0, F_part - floor_part) * dt / ref_part``. Wings have no
  collision geoms in NeuroMechFly, so a wing is damaged by *dorsal crush*: the
  component of a thorax / abdomen contact force that pushes the body from its
  dorsal side (body -z), assigned to the wing on the side of the contact point.
* **Effects** (bounded, reversible; ``update()`` once per physics chunk):
  leg: CPG stride amplitude x (1 - leg_amp_loss D) for that leg and its position
  actuators' kp x (1 - leg_kp_loss D) (a limp; the heading hold compensates the
  asymmetry); D >= lame_at: the leg is **lame** (held tucked, adhesion off) until it
  heals below lame_clear. Thorax / abdomen: CPG frequency x (1 - freq_loss D_body),
  and above rest_at the fly takes rest bouts (a freeze). One impact episode with a
  large total damage: **stunned** (legs slack, lies still for a few s, then gets
  up). Very large: **squashed** (a flattened replica + splat decal, visual only,
  then a counted, explicit respawn). Wings: with ``--flight`` the stroke amplitude
  of a damaged side is capped (less lift on that side).
* **Healing**: ``heal_delay_s`` after the last damage, every part heals linearly at
  ``heal_rate_per_s`` (default 0.025/s: damage 1 heals in 40 s of sim time). A
  reset / respawn heals completely (a fresh fly).

Off (``enabled=False``): ``install_injury`` returns an inert handle and touches
nothing, and the app does not add the decal bodies, so the default run is
bit-identical (tests/test_injury.py).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import mujoco as mj
import numpy as np

MODEL_LABEL = "phenomenological injury model (not biomechanical)"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
PARTS = LEGS + ("thorax", "abdomen", "l_wing", "r_wing")
N_LEGS = 6
THORAX, ABDOMEN, L_WING, R_WING = 6, 7, 8, 9
PREFIX = "injury/"
STATUSES = ("OK", "limping", "lame", "resting", "stunned", "squashed")
METRIC_COLUMNS = ("injury_status", "injury_max_leg", "injury_thorax", "injury_abdomen",
                  "injury_wings", "injury_freq_mult")
PARK = np.array([0.0, 0.0, -200.0])  # decal bodies wait here (below the ground)
HIDDEN_GROUP = 5  # mjvOption.geomgroup[5] is off by default -> not rendered


@dataclass
class InjuryConfig:
    enabled: bool = False  # --injury
    # --- damage (force in body weights, BW; calibrated in docs/INJURY.md) ---
    # 1) sustained crushing: D += max(0, F - floor) * dt / ref per step; floors sit
    #    above everything walking, jumps and falls produce (<= 7 BW measured)
    leg_floor_bw: float = 15.0
    body_floor_bw: float = 25.0       # thorax (+ head) and abdomen
    wing_floor_bw: float = 25.0       # dorsal crush component
    leg_ref_bws: float = 5.0          # BW*s of excess leg impulse -> damage 1
    thorax_ref_bws: float = 25.0
    abdomen_ref_bws: float = 40.0
    wing_ref_bws: float = 20.0
    # 2) sharp impact, once per episode and part: min(peak_cap, (peak - peak_floor)
    #    / peak_ref). Whip lashes are brief (1-2 ms, 0.01-0.2 BW*s) but hard (50-860 BW)
    peak_floor_bw: float = 150.0
    peak_ref_bw: float = 2000.0
    peak_cap: float = 0.25
    max_damage: float = 1.5           # per part
    # --- effects ---
    leg_amp_loss: float = 0.6         # stride amplitude x (1 - loss * D), D < lame_at
    leg_kp_loss: float = 0.5          # leg actuator kp x (1 - loss * D)
    min_leg_factor: float = 0.35
    lame_at: float = 0.8              # D >= this -> the leg is held tucked, no adhesion
    lame_clear: float = 0.55          # ... until it heals below this
    freq_loss: float = 0.5            # CPG frequency x (1 - loss * D_body)
    min_freq_mult: float = 0.5
    rest_at: float = 0.35             # D_body >= this -> rest bouts
    rest_every_s: float = 3.0         # walking time between rests (at rest_at; shorter when worse)
    rest_s: float = 1.2               # rest bout length (longer when worse)
    # stunned / squashed use the *body* (thorax + abdomen) damage of one episode
    stun_episode: float = 0.6         # body damage of one impact episode -> stunned
    stun_s: float = 3.0               # stunned for stun_s * min(2, episode / stun_episode)
    stun_kp_scale: float = 0.3        # leg kp while stunned (slack legs, lies still)
    squash_peak_bw: float = 4500.0    # thorax / abdomen peak force of one episode -> squashed
    squash_episode: float = 2.5       # ... or body damage of one episode
    squash_thorax: float = 1.3        # ... or cumulative thorax damage
    squash_show_s: float = 2.0        # flattened replica shown this long, then respawn
    wing_amp_loss: float = 0.5        # --flight: stroke amplitude cap x (1 - loss * D_wing)
    episode_gap_s: float = 0.05       # an impact episode ends after this long below the floors
    min_report: float = 0.01          # episodes adding less damage are counted as minor only
    # --- healing ---
    heal_rate_per_s: float = 0.025    # linear; damage 1 -> 0 in 40 s of sim time
    heal_delay_s: float = 2.0         # no healing this long after the last damage
    # --- visuals (visual only; no collision, no mass) ---
    splat_visual: bool = True
    hide_fly_when_squashed: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> "InjuryConfig":
        d = dict(d or {})
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class InjuryEvent:
    """One impact episode (contiguous force above the floors) or a state change."""

    kind: str                 # "impact" | "lame" | "stunned" | "squashed" | "respawn" | "recovered"
    time: float
    damage: dict = field(default_factory=dict)       # part -> damage added (impact)
    peak_bw: dict = field(default_factory=dict)      # part -> peak force (BW)
    impulse_bws: dict = field(default_factory=dict)  # part -> raw impulse (BW*s)
    sources: dict = field(default_factory=dict)      # other body -> impulse (BW*s)
    total: float = 0.0
    info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        r = lambda x: {k: round(float(v), 4) for k, v in x.items()}  # noqa: E731
        return {"kind": self.kind, "time": round(self.time, 4), "total": round(self.total, 4),
                "damage": r(self.damage), "peak_bw": r(self.peak_bw),
                "impulse_bws": r(self.impulse_bws), "sources": r(self.sources), **self.info}


def bar(h: float, n: int = 5) -> str:
    """Tiny text health bar: 1.0 -> '#####', 0 -> '.....'."""
    k = int(round(float(np.clip(h, 0.0, 1.0)) * n))
    return "#" * k + "." * (n - k)


# --------------------------------------------------------------------- visuals
def splat_extension(world) -> None:
    """World extension: two mocap bodies with visual-only geoms (contype 0,
    conaffinity 0, no mass): ``injury/flat``, a flattened fly replica, and
    ``injury/splat``, a goo stain that stays where the fly was squashed."""
    spec: mj.MjSpec = world.mjcf_root
    E, C, S = mj.mjtGeom.mjGEOM_ELLIPSOID, mj.mjtGeom.mjGEOM_CAPSULE, mj.mjtGeom.mjGEOM_CYLINDER
    kw = dict(contype=0, conaffinity=0, mass=0, group=1)
    flat = spec.worldbody.add_body(name=PREFIX + "flat", pos=PARK, mocap=True)
    brown, dark = (0.55, 0.36, 0.2, 1.0), (0.28, 0.17, 0.1, 1.0)
    flat.add_geom(name=PREFIX + "flat_thorax", type=E, pos=(0.0, 0, 0.05),
                  size=(0.55, 0.5, 0.05), rgba=brown, **kw)
    flat.add_geom(name=PREFIX + "flat_head", type=E, pos=(0.62, 0, 0.045),
                  size=(0.25, 0.42, 0.045), rgba=dark, **kw)
    for s in (-1, 1):
        flat.add_geom(name=f"{PREFIX}flat_eye{s:+d}", type=E, pos=(0.66, 0.3 * s, 0.07),
                      size=(0.16, 0.14, 0.03), rgba=(0.6, 0.08, 0.05, 1.0), **kw)
    flat.add_geom(name=PREFIX + "flat_abdomen", type=E, pos=(-0.95, 0, 0.04),
                  size=(0.75, 0.55, 0.04), rgba=(0.45, 0.3, 0.17, 1.0), **kw)
    for s in (-1, 1):  # wings splayed flat, translucent
        q = (math.cos(0.25 * s), 0.0, 0.0, math.sin(0.25 * s))
        flat.add_geom(name=f"{PREFIX}flat_wing{s:+d}", type=E, pos=(-0.9, 0.9 * s, 0.1),
                      quat=q, size=(1.1, 0.45, 0.012), rgba=(0.85, 0.85, 0.9, 0.45), **kw)
    for i, ang in enumerate((40, 90, 140)):  # legs splayed out, bent
        for s in (-1, 1):
            a = math.radians(ang) * s
            p0 = np.array([0.35 - 0.35 * i, 0.3 * s, 0.03])
            p1 = p0 + 1.1 * np.array([math.cos(a), math.sin(a), 0.0])
            b = a + 0.6 * s * (1 if i < 2 else -1)
            p2 = p1 + 0.9 * np.array([math.cos(b), math.sin(b), 0.0])
            for k, (u, v) in enumerate(((p0, p1), (p1, p2))):
                flat.add_geom(name=f"{PREFIX}flat_leg{i}{s:+d}{k}", type=C,
                              fromto=(*u, *v), size=(0.045, 0, 0), rgba=dark, **kw)
    goo = spec.worldbody.add_body(name=PREFIX + "splat", pos=PARK, mocap=True)
    goo.add_geom(name=PREFIX + "splat_main", type=S, pos=(-0.2, 0, 0.004),
                 size=(1.6, 0.004, 0), rgba=(0.62, 0.72, 0.12, 0.8), **kw)
    rng = np.random.default_rng(7)
    for k in range(9):  # droplets around the stain
        a = rng.uniform(0, 2 * math.pi)
        r = rng.uniform(1.6, 2.8)
        goo.add_geom(name=f"{PREFIX}splat_drop{k}", type=S,
                     pos=(r * math.cos(a) - 0.2, r * math.sin(a), 0.005),
                     size=(rng.uniform(0.12, 0.35), 0.005, 0),
                     rgba=(0.62, 0.72, 0.12, 0.8), **kw)
    goo.add_geom(name=PREFIX + "splat_dark", type=S, pos=(0.3, 0.25, 0.006),
                 size=(0.7, 0.006, 0), rgba=(0.55, 0.1, 0.06, 0.55), **kw)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


# ---------------------------------------------------------------------- sensor
class DamageSensor:
    """Per-step contact-force accumulator (post-step hook). Pure measurement plus the
    damage integration; the effects live in ``InjuryHandle``."""

    def __init__(self, sim, cfg: InjuryConfig) -> None:
        self.sim = sim
        self.cfg = cfg
        m = sim.model
        self.bw = float(sim.fly_mass * np.linalg.norm(m.opt.gravity))  # uN
        prefix = f"{sim.fly_name}/"
        root = sim.thorax_body_id
        self.part_of_geom = np.full(m.ngeom, -1, dtype=np.int64)
        self.is_tarsus = np.zeros(m.ngeom, dtype=bool)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if m.body_rootid[b] != root:
                continue
            name = m.body(b).name.removeprefix(prefix)
            leg, _, link = name.partition("_")
            if leg in LEGS:
                self.part_of_geom[g] = LEGS.index(leg)
                self.is_tarsus[g] = link.startswith("tarsus")
            elif name.startswith("c_abdomen"):
                self.part_of_geom[g] = ABDOMEN
            elif name in ("l_wing", "r_wing"):
                self.part_of_geom[g] = L_WING if name[0] == "l" else R_WING
            else:  # thorax, head parts, halteres
                self.part_of_geom[g] = THORAX
        body_of = m.geom_bodyid
        self.is_static = np.asarray(m.body_weldid[body_of] == 0)
        self.is_fly = self.part_of_geom >= 0
        n = len(PARTS)
        self.floor = np.array([cfg.leg_floor_bw] * N_LEGS
                              + [cfg.body_floor_bw, cfg.body_floor_bw,
                                 cfg.wing_floor_bw, cfg.wing_floor_bw]) * self.bw
        ref = np.array([cfg.leg_ref_bws] * N_LEGS
                       + [cfg.thorax_ref_bws, cfg.abdomen_ref_bws,
                          cfg.wing_ref_bws, cfg.wing_ref_bws]) * self.bw
        self.inv_ref = 1.0 / ref
        self.damage = np.zeros(n)
        self.force = np.zeros(n)   # this step's force per part (uN)
        self.peak_force = np.zeros(n)  # since construction / last clear (uN), diagnostics
        self.t_last_damage = -math.inf
        self._f6 = np.zeros(6)
        self._ep: dict | None = None
        self.episodes: deque[InjuryEvent] = deque()  # finished, not yet processed
        self.frozen = False  # squashed: no more damage until the respawn

    def clear(self) -> None:
        self.damage[:] = 0.0
        self.force[:] = 0.0
        self._ep = None
        self.t_last_damage = -math.inf

    def __call__(self, sim) -> None:
        d = sim.data
        ncon = d.ncon
        f = self.force
        if ncon == 0:
            if f.any():
                f[:] = 0.0
            self._close_episode(sim)
            return
        con = d.contact
        geom = con.geom[:ncon]
        g1, g2 = geom[:, 0], geom[:, 1]
        fly1, fly2 = self.is_fly[g1], self.is_fly[g2]
        sel = fly1 ^ fly2
        if not sel.any():
            if f.any():
                f[:] = 0.0
            self._close_episode(sim)
            return
        fg = np.where(fly1, g1, g2)
        og = np.where(fly1, g2, g1)
        sel &= ~(self.is_tarsus[fg] & self.is_static[og])
        sel &= con.exclude[:ncon] == 0
        f[:] = 0.0
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            self._close_episode(sim)
            return
        m = sim.model
        f6 = self._f6
        frames = con.frame
        R = None
        sources = None
        for cid in idx:
            cid = int(cid)
            mj.mj_contactForce(m, d, cid, f6)
            fw = frames[cid].reshape(3, 3).T @ f6[:3]  # on geom2 from geom1
            if fly1[cid]:
                fw = -fw
            mag = float(math.sqrt(fw[0] * fw[0] + fw[1] * fw[1] + fw[2] * fw[2]))
            p = int(self.part_of_geom[fg[cid]])
            f[p] += mag
            if p in (THORAX, ABDOMEN) and mag > 0.0:
                if R is None:
                    R = d.xmat[sim.thorax_body_id].reshape(3, 3)
                fb = R.T @ fw
                if fb[2] < 0.0:  # pushed from the dorsal side: crush onto the wings
                    rel = R.T @ (con.pos[cid] - d.xpos[sim.thorax_body_id])
                    f[L_WING if rel[1] >= 0.0 else R_WING] += -fb[2]
            if (mag > 0.0 and self._ep is not None) or mag > self.floor[p]:
                if sources is None:
                    sources = {}
                ob = int(m.geom_bodyid[og[cid]])
                sources[ob] = sources.get(ob, 0.0) + mag
        self._integrate(sim, sources)

    def _integrate(self, sim, sources) -> None:
        f = self.force
        excess = f - self.floor
        hot = excess > 0.0
        dt = sim.timestep
        if hot.any() and not self.frozen:
            dd = np.where(hot, excess, 0.0) * dt * self.inv_ref
            np.minimum(self.damage + dd, self.cfg.max_damage, out=self.damage)
            self.t_last_damage = sim.time
            ep = self._ep
            if ep is None:
                ep = self._ep = {"t0": sim.time, "damage": np.zeros(len(PARTS)),
                                 "peak": np.zeros(len(PARTS)), "imp": np.zeros(len(PARTS)),
                                 "src": {}, "t_hot": sim.time}
            ep["damage"] += dd
            ep["t_hot"] = sim.time
        ep = self._ep
        if ep is not None:
            np.maximum(ep["peak"], f, out=ep["peak"])
            ep["imp"] += f * dt
            if sources:
                src = ep["src"]
                for b, v in sources.items():
                    src[b] = src.get(b, 0.0) + v * dt
            self._close_episode(sim)

    def _close_episode(self, sim) -> None:
        ep = self._ep
        if ep is None or sim.time - ep["t_hot"] < self.cfg.episode_gap_s:
            return
        self._ep = None
        bw = self.bw
        m = sim.model
        c = self.cfg
        if not self.frozen:  # sharp-impact term, once per episode
            sharp = np.minimum(np.maximum(ep["peak"] / bw - c.peak_floor_bw, 0.0) / c.peak_ref_bw,
                               c.peak_cap)
            np.minimum(self.damage + sharp, c.max_damage, out=self.damage)
            ep["damage"] += sharp
        names = {}
        # moving bodies (swatter, whip, props) first, then the static world / terrain
        for b, v in sorted(ep["src"].items(), key=lambda kv: (m.body_weldid[kv[0]] == 0, -kv[1]))[:4]:
            names[m.body(b).name or "world"] = v / bw
        dmg = {PARTS[i]: float(v) for i, v in enumerate(ep["damage"]) if v > 1e-4}
        self.episodes.append(InjuryEvent(
            kind="impact", time=float(ep["t0"]), damage=dmg,
            peak_bw={PARTS[i]: float(v / bw) for i, v in enumerate(ep["peak"]) if v > 0},
            impulse_bws={PARTS[i]: float(v / bw) for i, v in enumerate(ep["imp"]) if v > 0},
            sources=names, total=float(ep["damage"].sum()),
            info={"duration_s": round(float(ep["t_hot"] - ep["t0"]), 4),
                  "body": round(float(ep["damage"][THORAX] + ep["damage"][ABDOMEN]), 4),
                  "body_peak_bw": round(float(max(ep["peak"][THORAX], ep["peak"][ABDOMEN]) / bw), 1)}))


# --------------------------------------------------------------------- actions
def _make_stunned(duration: float, kp_scale: float):
    """A slack-legged stillness: leg kp scaled down (the body sinks), all tarsi
    adhering, targets held where they were. The manager blends back to walking."""
    from fly_simulator.actions.base import Action, ActionCommand

    class Stunned(Action):
        name = "stunned"
        blend_in = 0.0
        blend_out = 0.4

        def __init__(self) -> None:
            super().__init__(duration)

        def begin(self, mgr) -> None:
            b = mgr.body
            self._targets = mgr.sim.data.ctrl[b.pos_ids].copy()
            self._on = np.ones(6)
            mgr.boost_actuators(b.pos_ids, kp_scale, 1.0)

        def command(self, mgr, t: float) -> ActionCommand:
            return ActionCommand(targets=self._targets, adhesion=self._on, drive=np.zeros(2))

    return Stunned()


# ---------------------------------------------------------------------- handle
class InjuryHandle:
    """Applies the damage to the body. Disabled -> inert (touches nothing)."""

    def __init__(self, cfg: InjuryConfig, sim=None, actions=None, *,
                 session=None, say: Callable[[str], None] | None = None,
                 log: Callable[..., None] | None = None) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.enabled) and sim is not None
        self.sim = sim
        self.actions = actions
        self.session = session
        self.say = say or (lambda msg: None)
        self.log = log or (lambda event_type, **kw: None)
        self.listeners: list[Callable[[InjuryEvent], None]] = []
        self.events: list[InjuryEvent] = []
        self.status = "OK"
        self.freq_mult = 1.0
        self.leg_amp = np.ones(N_LEGS)
        self.leg_kp = np.ones(N_LEGS)
        self.lame = np.zeros(N_LEGS, dtype=bool)
        self.n_impacts = 0
        self.n_minor = 0
        self.n_stuns = 0
        self.n_squashes = 0
        self.n_rests = 0
        self.n_lame = 0
        self.max_damage = np.zeros(len(PARTS))
        self.stunned_until = -math.inf
        self.squashed_at: float | None = None
        self.squash_pos: np.ndarray | None = None
        self._t_prev: float | None = None
        self._walk_since = None
        self._resting_until = -math.inf
        self.sensor: DamageSensor | None = None
        if not self.enabled:
            return
        self.sensor = DamageSensor(sim, cfg)
        sim.post_step_hooks.append(self.sensor)
        sim.pre_step_hooks.append(self._pre_step)  # after the ActionManager's hook
        sim.reset_hooks.append(self._on_reset)
        m = sim.model
        ctrl = sim.controller
        self._pos_ids = np.asarray(ctrl._pos_ids, dtype=np.int64)
        self._adh_ids = None if ctrl._adh_ids is None else np.asarray(ctrl._adh_ids, dtype=np.int64)
        from fly_simulator.actions.base import BodyIndex

        self.body = BodyIndex(sim) if actions is None else actions.body
        self._leg_of = self.body.leg_of  # (42,) leg index per position actuator
        self._kp0 = m.actuator_gainprm[self._pos_ids, 0].copy()
        self._bias0 = m.actuator_biasprm[self._pos_ids, 1].copy()
        self._kp_written = np.ones(len(self._pos_ids))
        self._tuck = self._tuck_pose()
        # CPG hooks: per-leg stride amplitude and the frequency, around each CPG step
        # (composes with --stress, which scales _base_intrinsic_freqs)
        impl = getattr(ctrl, "impl", None)
        net = getattr(impl, "cpg_network", None)
        self._net = net
        self._orig_net_step = None
        if net is not None:
            self._orig_net_step = net.step
            net.step = self._net_step  # instance attribute; undone in close()
        # decal bodies (present only if the session added splat_extension)
        self._flat_mocap = self._splat_mocap = None
        try:
            self._flat_mocap = int(m.body_mocapid[m.body(PREFIX + "flat").id])
            self._splat_mocap = int(m.body_mocapid[m.body(PREFIX + "splat").id])
        except KeyError:
            pass
        fly = np.flatnonzero(m.body_rootid[m.geom_bodyid] == sim.thorax_body_id)
        self._fly_geoms = fly
        self._fly_groups = m.geom_group[fly].copy()
        self._hover_wrapped = None

    # ------------------------------------------------------------------ helpers
    @property
    def damage(self) -> np.ndarray:
        return self.sensor.damage if self.sensor is not None else np.zeros(len(PARTS))

    @property
    def health(self) -> np.ndarray:
        return np.clip(1.0 - self.damage, 0.0, 1.0)

    def body_damage(self) -> float:
        d = self.damage
        return float(max(d[THORAX], 0.7 * d[ABDOMEN]))

    # (coxa pitch, femur pitch, tibia pitch) offsets from the standing pose that lift
    # the leg: searched kinematically for the highest tibia / tarsus (rad)
    TUCK = {"f": (-0.6, -0.8, 0.0), "m": (0.3, -0.8, -0.6), "h": (0.6, -0.4, -0.6)}

    def _tuck_pose(self) -> np.ndarray:
        """Standing pose with the leg swung up and folded (held off the ground)."""
        b = self.body
        pose = b.stand.copy()
        for leg in LEGS:
            for j, dv in zip(("thc_pitch", "ctr_pitch", "fti"), self.TUCK[leg[1]]):
                i = b.idx(leg, j)
                pose[i] = b.stand[i] + dv
        return pose

    def _net_step(self) -> None:
        net = self._net
        f0, a0 = net.intrinsic_freqs, net.intrinsic_amps
        m, amp = self.freq_mult, self.leg_amp
        if m == 1.0 and not (amp != 1.0).any():
            return self._orig_net_step()
        net.intrinsic_freqs = f0 * m
        net.intrinsic_amps = a0 * amp
        try:
            self._orig_net_step()
        finally:
            net.intrinsic_freqs, net.intrinsic_amps = f0, a0

    def _pre_step(self, sim) -> None:
        """After the controller / actions: lame legs held tucked, adhesion off."""
        if not self.lame.any() or self.status in ("stunned", "squashed"):
            return
        if getattr(sim, "leg_mode", "walk") != "walk":
            return
        d = sim.data
        dofs = self.lame[self._leg_of]
        d.ctrl[self._pos_ids[dofs]] = self._tuck[dofs]
        if self._adh_ids is not None:
            d.ctrl[self._adh_ids[self.lame]] = 0.0

    # ------------------------------------------------------------------ update
    def update(self) -> None:
        """Once per physics chunk (outside sim.step): episodes -> events / stun /
        squash, healing, effects."""
        if not self.enabled:
            return
        sim, cfg, s = self.sim, self.cfg, self.sensor
        t = sim.time
        dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
        self._t_prev = t
        while s.episodes:
            self._on_episode(s.episodes.popleft())
        if self.squashed_at is not None:
            if t - self.squashed_at >= cfg.squash_show_s:
                self._respawn()
            return
        # healing
        if dt > 0 and t - s.t_last_damage >= cfg.heal_delay_s and s.damage.any():
            np.maximum(s.damage - cfg.heal_rate_per_s * dt, 0.0, out=s.damage)
        np.maximum(self.max_damage, s.damage, out=self.max_damage)
        if s.damage[THORAX] >= cfg.squash_thorax:
            self._squash("thorax damage")
            return
        self._apply_effects(t)

    def _apply_effects(self, t: float) -> None:
        cfg = self.cfg
        D = self.damage
        legs = D[:N_LEGS]
        was = self.lame.copy()
        self.lame = np.where(was, legs >= cfg.lame_clear, legs >= cfg.lame_at)
        for i in np.flatnonzero(self.lame & ~was):
            self.n_lame += 1
            self._event("lame", info={"leg": LEGS[i], "damage": round(float(legs[i]), 3)})
        for i in np.flatnonzero(was & ~self.lame):
            self._event("recovered", info={"leg": LEGS[i], "damage": round(float(legs[i]), 3)})
        lo = cfg.min_leg_factor
        self.leg_amp = np.where(self.lame, 1.0, np.clip(1.0 - cfg.leg_amp_loss * legs, lo, 1.0))
        self.leg_kp = np.clip(1.0 - cfg.leg_kp_loss * legs, lo, 1.0)
        db = self.body_damage()
        self.freq_mult = float(np.clip(1.0 - cfg.freq_loss * db, cfg.min_freq_mult, 1.0))
        self._write_kp()
        self._wings()
        act = self.actions
        name = act.active_name if act is not None else None
        if t < self.stunned_until or name == "stunned":
            self.status = "stunned"
            return
        # rest bouts when the body is hurt
        if name == "freeze" and t < self._resting_until:
            self.status = "resting"
            return
        if db >= cfg.rest_at and act is not None and not act.busy and not self._airborne():
            if self._walk_since is None:
                self._walk_since = t
            every = cfg.rest_every_s * max(0.3, 1.0 - (db - cfg.rest_at))
            if t - self._walk_since >= every:
                dur = cfg.rest_s * (1.0 + 2.0 * db)
                from fly_simulator.actions.registry import make_action

                act.trigger(make_action("freeze", duration=dur), source="injury")
                self._resting_until = t + dur
                self._walk_since = None
                self.n_rests += 1
                self.log("injury_rest", duration_s=round(dur, 2), body_damage=round(db, 3))
                self.status = "resting"
                return
        elif db < cfg.rest_at:
            self._walk_since = None
        if self.lame.any():
            self.status = "lame"
        elif (legs > 0.05).any() or db > 0.05:
            self.status = "limping"
        else:
            self.status = "OK"

    def _rt(self, t: float) -> float:
        """Sim time -> the session's run time (what the terminal / HUD show)."""
        m = getattr(self.session, "metrics", None)
        if m is not None and hasattr(m, "run_time_at"):
            return float(m.run_time_at(t))
        return t

    def _airborne(self) -> bool:
        fl = getattr(self.session, "flight", None)
        return bool(fl is not None and fl.busy)

    def _write_kp(self) -> None:
        """Leg actuator kp x leg_kp (not while an action holds boosted params: the
        ActionManager restores to whatever it saved; rewritten next update)."""
        act = self.actions
        if act is not None and act._saved_params:
            return
        g = self.leg_kp[self._leg_of]
        if np.array_equal(g, self._kp_written):
            return
        m = self.sim.model
        m.actuator_gainprm[self._pos_ids, 0] = self._kp0 * g
        m.actuator_biasprm[self._pos_ids, 1] = self._bias0 * g
        self._kp_written = g.copy()

    def _wings(self) -> None:
        """--flight: cap each side's stroke amplitude (less lift on a damaged side)."""
        ctrl = getattr(self.sim, "flight_controller", None)
        if ctrl is None or ctrl is self._hover_wrapped:
            return
        orig = ctrl._write
        sim = self.sim

        def capped_write(sim_, u, _orig=orig):
            _orig(sim_, u)
            p = sim.wingbeat.params
            base = ctrl.base
            caps = [base.left.amplitude * (1 - self.cfg.wing_amp_loss * min(1.0, self.damage[L_WING])),
                    base.right.amplitude * (1 - self.cfg.wing_amp_loss * min(1.0, self.damage[R_WING]))]
            if p.left.amplitude > caps[0] or p.right.amplitude > caps[1]:
                from dataclasses import replace

                sim.wingbeat.params = replace(
                    p, left=replace(p.left, amplitude=min(p.left.amplitude, caps[0])),
                    right=replace(p.right, amplitude=min(p.right.amplitude, caps[1])))

        ctrl._write = capped_write
        self._hover_wrapped = ctrl

    # ------------------------------------------------------------------ events
    def _event(self, kind: str, ev: InjuryEvent | None = None, **kw) -> InjuryEvent:
        ev = ev or InjuryEvent(kind=kind, time=self.sim.time, **kw)
        self.events.append(ev)
        d = ev.to_dict()
        d.pop("kind")
        d.pop("time")
        self.log(f"injury_{kind}", **{k: v for k, v in d.items() if v not in ({}, 0.0)})
        for fn in list(self.listeners):
            fn(ev)
        return ev

    def _on_episode(self, ev: InjuryEvent) -> None:
        cfg = self.cfg
        if ev.total < cfg.min_report and ev.info["body_peak_bw"] < cfg.squash_peak_bw:
            self.n_minor += 1  # a knock above the floor that did (almost) nothing
            return
        self.n_impacts += 1
        self._event("impact", ev)
        worst = max(ev.damage.items(), key=lambda kv: kv[1])[0] if ev.damage else "-"
        src = next(iter(ev.sources), "?").removeprefix(f"{self.sim.fly_name}/")
        self.say(f"[injury] t={self._rt(ev.time):.2f}s impact from {src}: damage +{ev.total:.2f} "
                 f"(worst {worst}), peak {max(ev.peak_bw.values(), default=0):.0f} BW")
        if self.squashed_at is not None:
            return
        body, peak = ev.info["body"], ev.info["body_peak_bw"]
        if peak >= cfg.squash_peak_bw:
            self._squash(f"crushed at {peak:.0f} BW")
        elif body >= cfg.squash_episode:
            self._squash(f"body damage +{body:.2f}")
        elif body >= cfg.stun_episode:
            self._stun(body)

    def _stun(self, body: float) -> None:
        cfg, act = self.cfg, self.actions
        dur = cfg.stun_s * min(2.0, body / cfg.stun_episode)
        self.stunned_until = self.sim.time + dur
        self.n_stuns += 1
        self.status = "stunned"
        self._event("stunned", info={"duration_s": round(dur, 2), "episode_body": round(body, 3)})
        self.say(f"[injury] stunned for {dur:.1f}s")
        if act is not None and not self._airborne():
            act.trigger(_make_stunned(dur, cfg.stun_kp_scale), source="injury")

    def _squash(self, why: str) -> None:
        cfg, sim = self.cfg, self.sim
        self.squashed_at = sim.time
        self.n_squashes += 1
        self.status = "squashed"
        self.sensor.frozen = True
        x, y, _ = sim.thorax_position()
        gz = 0.0
        if self.session is not None and hasattr(self.session, "ground_height"):
            gz = float(self.session.ground_height(x, y))
        self.squash_pos = np.array([x, y, gz])
        self._event("squashed", info={"why": why, "x": round(float(x), 3),
                                      "y": round(float(y), 3), "n": self.n_squashes})
        self.say(f"[injury] SQUASHED ({why}) at ({x:.1f}, {y:.1f}) mm: respawn in "
                 f"{cfg.squash_show_s:g}s (#{self.n_squashes})")
        if cfg.splat_visual and self._flat_mocap is not None:
            d = sim.data
            q = _yaw_quat(sim.heading())
            d.mocap_pos[self._flat_mocap] = self.squash_pos
            d.mocap_quat[self._flat_mocap] = q
            d.mocap_pos[self._splat_mocap] = self.squash_pos
            d.mocap_quat[self._splat_mocap] = q
            if cfg.hide_fly_when_squashed:
                sim.model.geom_group[self._fly_geoms] = HIDDEN_GROUP
            mj.mj_kinematics(sim.model, d)
        if self.actions is not None and not self._airborne():
            self.actions.trigger(_make_stunned(cfg.squash_show_s + 1.0, 0.05), source="injury")

    def _respawn(self) -> None:
        """Counted, explicit reset (the job's recover() inside a job)."""
        s = self.session
        self._event("respawn", info={"n": self.n_squashes})
        self.say(f"[injury] respawn after squash #{self.n_squashes} (explicit reset)")
        job = getattr(s, "job", None)
        if job is not None:
            job.recover("squashed")
        elif s is not None and hasattr(s, "reset"):
            s.reset("squash")
        else:
            self.sim.reset()

    def _on_reset(self, sim) -> None:
        """Any reset = a fresh fly: full health, visuals restored; the goo stays."""
        s = self.sensor
        s.clear()
        s.frozen = False
        s.episodes.clear()
        self.squashed_at = None
        self.stunned_until = -math.inf
        self._resting_until = -math.inf
        self._walk_since = None
        self.lame[:] = False
        self.leg_amp = np.ones(N_LEGS)
        self.leg_kp = np.ones(N_LEGS)
        self.freq_mult = 1.0
        self.status = "OK"
        self._t_prev = sim.time
        m = sim.model
        m.geom_group[self._fly_geoms] = self._fly_groups
        m.actuator_gainprm[self._pos_ids, 0] = self._kp0
        m.actuator_biasprm[self._pos_ids, 1] = self._bias0
        self._kp_written = np.ones(len(self._pos_ids))
        if self._flat_mocap is not None:
            sim.data.mocap_pos[self._flat_mocap] = PARK
            if self.squash_pos is not None:  # the stain stays on the floor
                sim.data.mocap_pos[self._splat_mocap] = self.squash_pos
                sim.data.mocap_quat[self._splat_mocap] = _yaw_quat(0.3 * self.n_squashes)

    # ------------------------------------------------------------------ readouts
    def hud_lines(self) -> list[str]:
        """Two HUD lines: status + body, then per-leg health bars (x = lame)."""
        if not self.enabled:
            return []
        h = self.health
        legs = "  ".join(f"{LEGS[i].upper()}{'x' if self.lame[i] else ' '}{bar(h[i])}"
                         for i in range(N_LEGS))
        return [f"INJURY {self.status.upper()} (model)  thorax {bar(h[THORAX])} abdomen "
                f"{bar(h[ABDOMEN])} wings {bar(h[L_WING], 3)}/{bar(h[R_WING], 3)}  step "
                f"x{self.freq_mult:.2f}  hits {self.n_impacts} stuns {self.n_stuns} "
                f"squashed {self.n_squashes}",
                f"  legs {legs}"]

    def hud_line(self) -> str:
        return "  |  ".join(self.hud_lines())

    def metric_row(self) -> tuple:
        D = self.damage
        return (self.status, float(D[:N_LEGS].max()), float(D[THORAX]), float(D[ABDOMEN]),
                float(D[L_WING:].max()), self.freq_mult)

    def summary(self) -> dict:
        D = self.damage
        return {"enabled": self.enabled, "model": MODEL_LABEL, "status": self.status,
                "damage": {p: round(float(D[i]), 4) for i, p in enumerate(PARTS)},
                "max_damage": {p: round(float(self.max_damage[i]), 4) for i, p in enumerate(PARTS)},
                "n_impacts": self.n_impacts, "n_minor_impacts": self.n_minor, "n_lame": self.n_lame, "n_rests": self.n_rests,
                "n_stuns": self.n_stuns, "n_squashes": self.n_squashes,
                "config": asdict(self.cfg)}

    def close(self) -> None:
        if not self.enabled:
            return
        sim = self.sim
        for lst, fn in ((sim.post_step_hooks, self.sensor), (sim.pre_step_hooks, self._pre_step),
                        (sim.reset_hooks, self._on_reset)):
            if fn in lst:
                lst.remove(fn)
        if self._orig_net_step is not None:
            try:
                del self._net.step
            except AttributeError:
                pass
        m = sim.model
        m.actuator_gainprm[self._pos_ids, 0] = self._kp0
        m.actuator_biasprm[self._pos_ids, 1] = self._bias0
        m.geom_group[self._fly_geoms] = self._fly_groups
        self.enabled = False


def install_injury(session_or_parts: Any, cfg: InjuryConfig | dict | None = None,
                   say: Callable[[str], None] | None = None) -> InjuryHandle:
    """Wire the injury model to a ``Session`` (or {"sim", "actions"}). With
    ``cfg.enabled`` False this returns an inert handle and touches nothing."""
    if not isinstance(cfg, InjuryConfig):
        cfg = InjuryConfig.from_dict(cfg)
    s = session_or_parts
    if isinstance(s, dict):
        return InjuryHandle(cfg, s.get("sim"), s.get("actions"), session=s.get("session"),
                            say=say)
    log = None
    if getattr(s, "log_event", None) is not None:
        log = s.log_event
    return InjuryHandle(cfg, getattr(s, "sim", None), getattr(s, "actions", None), session=s,
                        say=say or getattr(s, "say", None), log=log)
