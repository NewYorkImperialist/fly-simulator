"""Doner kebab: a fly carving a doner kebab for eternity.

A vertical rotisserie spit turns slowly in front of the fly (hinge joint driven by a
velocity actuator). The meat is an inverted cone made of stacked chunks, next to a
glowing heater, with a drip tray below. The fly stands at its station and carves
with a tiny knife held in its right front "hand" (a blade on the right front
tarsus). The stroke is the **real recorded front-leg grooming bout** from
NeuroMechFly (DeepFly3D, 2 kHz; the same clip ``Groom`` replays, docs/ACTIONS.md
section 3), looped as a slicing motion, with the mid and hind legs planted and
adhering. The fly does not know what a kebab is: a fly cleaning its legs, plus a
knife, gives a kebab carver.

* **Cut**: every job update (1 ms) the blade (three points from mid-blade to tip) is
  tested against the ripe chunks. If a blade point is inside a chunk and the tip
  moves faster than ``cut_speed``, the chunk comes off. Blade and meat do not
  collide physically (they are visual geoms): contacts between a 1 mg fly's leg and
  a turning spit would knock the fly over, and the cut is a geometric test.
* **Shaving**: the chunk vanishes from the spit and a shaving from a fixed pool of
  free bodies (recycled oldest first) is launched from the chunk's place with the
  spit's surface speed. It falls onto the drip tray (real physics; shavings touch only
  the tray, the floor and each other, not the fly).
* **Regrowth**: cut chunks regrow at the spit (raw pink to browned) over ``regrow_s``,
  so the meat never runs out. Constant memory: pools, counters and fixed deques only.
* **Counters**: shavings carved (the work counter), kebabs completed (every
  ``shavings_per_kebab``), micrograms served (and the human-scale equivalent), and
  the carving rate.
* **Brain (only with --brain)**: the kebab counts as food. Every ``food_every_s`` the
  job sends a sugar-taste stimulus (sugar GRNs, the "T" key's set) to the connectome
  brain. With the full body (the job turns on the proboscis joints when the brain is
  enabled) the proboscis follows the brain's MN9 feeding motor neuron rate. The
  food response is the model's real wiring; calling sugar input "kebab" is a label.
* **Stress (optional)**: if the session has an enabled stress handle
  (``session.stress``, docs/STRESS.md), the carving playback speed scales with its
  CPG frequency multiplier, so an aroused fly carves faster.

Knife and chef hat: visual-only geoms (no mass, no contacts) added to the fly's
right front tarsus1 / thorax just before ``add_fly``. They never change the dynamics.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from perpetualfly.actions.base import ActionCommand
from perpetualfly.actions.behaviours import Groom
from perpetualfly.jobs.base import CameraPreset, EternalJob, JobConfig
from perpetualfly.jobs.geometry import PROP_BIT, add_box, contact_kwargs, quat_axis_angle
from perpetualfly.jobs.registry import register_job
from perpetualfly.terrain import TERRAIN_BIT

P = "kebab/"

STEEL = (0.72, 0.73, 0.76, 1.0)
DARK_STEEL = (0.25, 0.25, 0.28, 1.0)
GLOW = (1.0, 0.35, 0.08, 1.0)
CORE = (0.66, 0.36, 0.26, 1.0)  # inner meat (exposed where chunks were cut)
CRUST = (0.36, 0.18, 0.08, 1.0)
RAW = np.array([0.86, 0.47, 0.45])  # regrowing chunk colour at the start
BROWNS = np.array([[0.55, 0.29, 0.12], [0.62, 0.34, 0.15], [0.48, 0.24, 0.10],
                   [0.68, 0.40, 0.19], [0.58, 0.31, 0.16]])
TILE = (0.93, 0.92, 0.88, 1.0)
WOOD = (0.45, 0.26, 0.12, 1.0)
BLADE = (0.86, 0.88, 0.92, 1.0)
HAT = (0.98, 0.98, 0.98, 1.0)

# human / fly length ratio (1.75 m / 2.5 mm): masses scale with its cube
HUMAN_SCALE = (1750.0 / 2.5) ** 3


# ---------------------------------------------------------------------------
# the carving stroke: recorded grooming, looped
# ---------------------------------------------------------------------------


class CarveStroke(Groom):
    """The recorded NeuroMechFly front-leg grooming clip, looped as a carving stroke.

    Same targets as ``Groom`` (recorded front legs, mid legs extended, mid / hind
    legs adhering), but always the recorded clip, and playback speed may change
    while it runs (a phase accumulator, so stress can speed it up smoothly)."""

    name = "carve"
    blend_in = 0.25
    blend_out = 0.3

    def __init__(self, duration: float = 60.0, speed: float = 1.0) -> None:
        super().__init__(duration, speed=speed, source="recorded")
        self._phase_s = 0.0
        self._t_last = 0.0

    def begin(self, mgr) -> None:
        super().begin(mgr)
        self._phase_s = 0.0
        self._t_last = 0.0

    def clip_time(self) -> float:
        """Current position in the recorded clip (s)."""
        n = len(self._clip) - 1
        return ((self._phase_s * self._fps) % n) / self._fps

    def phase(self, t: float) -> str:
        return "carve" if t >= self.blend_in else "raise knife"

    def command(self, mgr, t: float) -> ActionCommand:
        self._phase_s += max(t - self._t_last, 0.0) * self.speed
        self._t_last = t
        tg = self._targets
        x = (self._phase_s * self._fps) % (len(self._clip) - 1)
        i = int(x)
        f = x - i
        tg[self._cols] = (1 - f) * self._clip[i] + f * self._clip[i + 1]
        return ActionCommand(targets=tg.copy(), adhesion=self._adh)

    def end(self, mgr, cancelled: bool) -> None:
        self.info = {"source": "recorded", "tilt_deg": mgr.sim.tilt_deg(), "cancelled": cancelled}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class KebabConfig(JobConfig):
    # ---- where the fly stands (thorax while carving; it spawns at the origin) -----
    station_x: float = 0.44
    station_y: float = 0.08
    # ---- the spit / meat (inverted cone, like a real doner) ------------------------
    gap: float = 1.7  # thorax -> near meat surface at z = 1.2 mm (the blade reaches ~1.9)
    spit_dy: float = -0.2  # spit axis offset in y (the right leg sweeps mostly at y < 0)
    meat_z0: float = 0.35
    meat_z1: float = 2.85
    r_bottom: float = 0.9
    r_top: float = 1.25
    n_rings: int = 10
    chunk_w: float = 0.34  # tangential chunk width (mm)
    chunk_t: float = 0.16  # radial chunk thickness (mm)
    spin_rpm: float = 12.0  # a real doner turns ~1-3 rpm; this one is in a hurry
    spin_kv: float = 2e-5  # velocity actuator gain (uN*mm*s/rad)
    # ---- carving -----------------------------------------------------------------
    carve_speed: float = 1.0  # recorded clip playback rate
    blade_len: float = 0.8  # mm
    cut_radius: float = 0.2  # blade point within this of a chunk centre = inside it
    cut_speed: float = 12.0  # mm/s: minimum blade-tip speed for a cut
    cut_cooldown_s: float = 0.12  # at most one shaving per this long
    ripe_frac: float = 0.85  # a regrowing chunk is cuttable from this size on
    regrow_s: float = 18.0  # a cut chunk regrows over this long
    shavings_per_kebab: int = 60
    slice_ug: float = 3.5  # micrograms per shaving (a 0.34 x 0.24 x 0.04 mm slice)
    # ---- shavings pool (free bodies) ----------------------------------------------
    n_shavings: int = 18
    shaving_mass: float = 3e-6  # g (3 ug)
    shaving_kick: float = 6.0  # mm/s outward when it comes off
    # ---- station keeping -----------------------------------------------------------
    drift_tol: float = 0.45  # mm from the station -> stop carving and walk back
    heading_tol_deg: float = 25.0
    # ---- brain tie-in (only if the session has a brain) ----------------------------
    food_every_s: float = 4.0
    food_duration_s: float = 1.0
    food_intensity: float = 0.8
    proboscis: bool = True  # full body when the brain is on (MN9 -> proboscis)
    mn9_ref_hz: float = 60.0  # MN9 rate that extends the proboscis fully
    # ---- stress tie-in -------------------------------------------------------------
    stress_speed_gain: float = 1.0  # carve speed *= 1 + gain * (freq_mult - 1)
    chef_hat: bool = True
    stuck_timeout_s: float = 90.0


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------


def _quat_from_x_axis(x_axis, up_hint=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """Quaternion whose frame has its x axis along ``x_axis``."""
    x = np.asarray(x_axis, float)
    x = x / np.linalg.norm(x)
    h = np.asarray(up_hint, float)
    y = np.cross(h, x)
    if np.linalg.norm(y) < 1e-6:
        y = np.cross((0.0, 1.0, 0.0), x)
    y = y / np.linalg.norm(y)
    z = np.cross(x, y)
    q = np.empty(4)
    mj.mju_mat2Quat(q, np.column_stack([x, y, z]).ravel())
    return tuple(float(v) for v in q)


@register_job
class KebabJob(EternalJob):
    name = "kebab"
    title = "DONER KEBAB FLY"
    tagline = "carving for eternity"
    work_label = "shavings carved"
    config_cls = KebabConfig
    required_names = (P + "spit", P + "tray")

    cfg: KebabConfig

    # knife in the right front tarsus1 frame (found by a search over the recorded
    # stroke: the tip is > 1.6 mm ahead of the thorax 58 % of the time and never
    # below the floor): handle at the tarsal claws, blade pointing forward-down
    KNIFE_ANCHOR = np.array([0.08, 0.0, -0.45])
    KNIFE_DIR = np.array([0.7, 0.3, -0.65]) / np.linalg.norm([0.7, 0.3, -0.65])

    # ------------------------------------------------------------ geometry
    def radius_at(self, z: float) -> float:
        c = self.cfg
        f = min(max((z - c.meat_z0) / (c.meat_z1 - c.meat_z0), 0.0), 1.0)
        return c.r_bottom + f * (c.r_top - c.r_bottom)

    @property
    def spit_xy(self) -> tuple[float, float]:
        c = self.cfg
        return (c.station_x + c.gap + self.radius_at(1.2), c.station_y + c.spit_dy)

    @property
    def tray_top(self) -> float:
        return 0.06

    def configure_app(self, app_cfg) -> None:
        super().configure_app(app_cfg)
        app_cfg.controller.target_heading_deg = 0.0
        if getattr(app_cfg.brain, "enabled", False) and self.cfg.proboscis:
            app_cfg.fly.extra_joints = True  # proboscis joints for the MN9 read-out

    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        sx, sy = self.spit_xy
        rng = np.random.default_rng(c.seed + 11)
        # --- the fly's knife and chef hat: added to the fly spec right before add_fly
        orig_add_fly = world.add_fly

        def add_fly(fly, *a, **kw):
            world.add_fly = orig_add_fly
            self._dress_fly(fly)
            return orig_add_fly(fly, *a, **kw)

        world.add_fly = add_fly
        # --- floor: kebab shop tiles
        mat = spec.material("grid")
        if mat is not None:
            mat.rgba = TILE
        # --- the spit: hinge body turning about z
        spit = wb.add_body(name=P + "spit", pos=(sx, sy, 0.0))
        spit.add_joint(name=P + "spin", type=mj.mjtJoint.mjJNT_HINGE, axis=(0, 0, 1),
                       damping=1e-7, armature=1e-6)
        rod_top = c.meat_z1 + 0.55
        spit.add_geom(name=P + "rod", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.05, rod_top / 2, 0),
                      pos=(0, 0, rod_top / 2 + 0.06), rgba=STEEL, mass=1e-4,
                      **contact_kwargs("visual"))
        spit.add_geom(name=P + "rod_tip", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.09, 0, 0),
                      pos=(0, 0, rod_top + 0.06), rgba=STEEL, mass=0.0, **contact_kwargs("visual"))
        spit.add_geom(name=P + "cap", type=mj.mjtGeom.mjGEOM_CYLINDER,
                      size=(c.r_top * 0.93, 0.05, 0), pos=(0, 0, c.meat_z1 + 0.03), rgba=CRUST,
                      mass=0.0, **contact_kwargs("visual"))
        spit.add_geom(name=P + "collar", type=mj.mjtGeom.mjGEOM_CYLINDER,
                      size=(0.45, 0.06, 0), pos=(0, 0, c.meat_z0 - 0.06), rgba=STEEL,
                      mass=0.0, **contact_kwargs("visual"))
        # meat: rings of chunks on the cone surface over an inner core
        n_r = c.n_rings
        dz = (c.meat_z1 - c.meat_z0) / n_r
        self._chunk_names: list[str] = []
        self._chunk_colors: list[np.ndarray] = []
        self._chunk_ring: list[int] = []
        for k in range(n_r):
            zc = c.meat_z0 + (k + 0.5) * dz
            r = self.radius_at(zc)
            spit.add_geom(name=f"{P}core{k}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                          size=(r - c.chunk_t * 0.9, dz / 2, 0), pos=(0, 0, zc), rgba=CORE,
                          mass=0.0, **contact_kwargs("visual"))
            n = max(8, int(round(2 * math.pi * r / c.chunk_w)))
            off = rng.uniform(0, 2 * math.pi)
            for i in range(n):
                th = off + 2 * math.pi * i / n
                t = c.chunk_t * rng.uniform(0.85, 1.15)
                rc = r - t / 2
                half = (t / 2, math.pi * r / n * 1.08, dz / 2 * rng.uniform(0.95, 1.08))
                col = BROWNS[rng.integers(len(BROWNS))] * rng.uniform(0.9, 1.08)
                if k == n_r - 1:
                    col = col * 0.8  # the top ring is the crispiest
                col = np.clip(col, 0, 1)
                name = f"{P}chunk{k}_{i}"
                spit.add_geom(name=name, type=mj.mjtGeom.mjGEOM_BOX, size=half,
                              pos=(rc * math.cos(th), rc * math.sin(th),
                                   zc + dz * rng.uniform(-0.06, 0.06)),
                              quat=quat_axis_angle((0, 0, 1), th), rgba=(*col, 1.0),
                              mass=0.0, **contact_kwargs("visual"))
                self._chunk_names.append(name)
                self._chunk_colors.append(col)
                self._chunk_ring.append(k)
        act = spec.add_actuator(name=P + "motor", target=P + "spin",
                                trntype=mj.mjtTrn.mjTRN_JOINT)
        act.set_to_velocity(kv=c.spin_kv)
        # --- base, drip tray (only shavings touch it), heater
        rmax = max(c.r_bottom, c.r_top)
        add_box(wb, P + "base", (0.5, 0.5, 0.1), (sx, sy, 0.1), rgba=DARK_STEEL,
                collide="visual")
        tx0, tx1 = sx - rmax - 0.75, sx + rmax + 0.35
        ty0, ty1 = sy - rmax - 0.9, sy + rmax + 0.9
        tray_kw = dict(contype=TERRAIN_BIT, conaffinity=0)
        tray_c = contact_kwargs("static", friction=1.0)
        tray_c.update(tray_kw)
        wb.add_geom(name=P + "tray", type=mj.mjtGeom.mjGEOM_BOX,
                    size=((tx1 - tx0) / 2, (ty1 - ty0) / 2, self.tray_top / 2),
                    pos=((tx0 + tx1) / 2, (ty0 + ty1) / 2, self.tray_top / 2), rgba=STEEL,
                    **tray_c)
        lip_h = 0.16
        for nm, half, pos in (
                ("lip_near", (0.03, (ty1 - ty0) / 2, lip_h / 2), (tx0, (ty0 + ty1) / 2, lip_h / 2)),
                ("lip_far", (0.03, (ty1 - ty0) / 2, lip_h / 2), (tx1, (ty0 + ty1) / 2, lip_h / 2)),
                ("lip_l", ((tx1 - tx0) / 2, 0.03, lip_h / 2), ((tx0 + tx1) / 2, ty1, lip_h / 2)),
                ("lip_r", ((tx1 - tx0) / 2, 0.03, lip_h / 2), ((tx0 + tx1) / 2, ty0, lip_h / 2))):
            wb.add_geom(name=P + nm, type=mj.mjtGeom.mjGEOM_BOX, size=half, pos=pos,
                        rgba=STEEL, **tray_c)
        self._tray = (tx0, tx1, ty0, ty1)
        hx = sx + rmax + 0.55
        hz = (c.meat_z0 + c.meat_z1) / 2 + 0.1
        add_box(wb, P + "heater", (0.07, rmax + 0.5, (c.meat_z1 - c.meat_z0) / 2 + 0.35),
                (hx, sy, hz), rgba=DARK_STEEL, collide="visual")
        spec.add_material(name=P + "glow", rgba=GLOW, emission=1.0)
        for j in range(4):
            z = c.meat_z0 + 0.25 + j * (c.meat_z1 - c.meat_z0 - 0.4) / 3
            wb.add_geom(name=f"{P}glow{j}", type=mj.mjtGeom.mjGEOM_BOX,
                        size=(0.02, rmax + 0.3, 0.06), pos=(hx - 0.08, sy, z), rgba=GLOW,
                        material=P + "glow", **contact_kwargs("visual"))
        # --- shaving pool: free bodies resting on the tray (a pile from earlier)
        kw = contact_kwargs("dynamic", friction=1.0)
        kw["contype"] = PROP_BIT
        kw["conaffinity"] = TERRAIN_BIT | PROP_BIT  # tray, floor, each other; not the fly
        self._park: list[tuple[float, ...]] = []
        for i in range(c.n_shavings):
            px = tx0 + 0.3 + (i % 3) * 0.3 + rng.uniform(-0.08, 0.08)
            py = sy - rmax - 0.45 + (i // 3) * 0.4 + rng.uniform(-0.1, 0.1)
            pz = self.tray_top + 0.035
            yaw = rng.uniform(0, math.pi)
            q = quat_axis_angle((0, 0, 1), yaw)
            b = wb.add_body(name=f"{P}shaving{i}", pos=(px, py, pz), quat=q)
            b.add_freejoint(name=f"{P}shaving{i}_free")
            col = np.clip(BROWNS[i % len(BROWNS)] * 1.05, 0, 1)
            b.add_geom(name=f"{P}shaving{i}_g", type=mj.mjtGeom.mjGEOM_ELLIPSOID,
                       size=(0.17, 0.12, 0.03), rgba=(*col, 1.0), mass=c.shaving_mass, **kw)
            self._park.append((px, py, pz, *q))

    def _dress_fly(self, fly) -> None:
        """Knife on the right front tarsus1, chef hat on the head (visual only)."""
        root = fly.mjcf_root
        vis = dict(contype=0, conaffinity=0, group=1, mass=0.0)
        tarsus = root.body("rf_tarsus1")
        a, d = self.KNIFE_ANCHOR, self.KNIFE_DIR
        q = _quat_from_x_axis(d, up_hint=(0.0, 1.0, 0.0))
        L = self.cfg.blade_len
        tarsus.add_geom(name="kebab_knife_handle", type=mj.mjtGeom.mjGEOM_BOX,
                        size=(0.12, 0.035, 0.035), pos=tuple(a - d * 0.06), quat=q, rgba=WOOD, **vis)
        tarsus.add_geom(name="kebab_knife_guard", type=mj.mjtGeom.mjGEOM_BOX,
                        size=(0.012, 0.06, 0.05), pos=tuple(a + d * 0.07), quat=q, rgba=DARK_STEEL,
                        **vis)
        tarsus.add_geom(name="kebab_blade", type=mj.mjtGeom.mjGEOM_BOX,
                        size=(L / 2, 0.012, 0.055), pos=tuple(a + d * (0.08 + L / 2)), quat=q,
                        rgba=BLADE, **vis)
        if self.cfg.chef_hat:
            th = root.body("c_thorax")
            th.add_geom(name="kebab_hat_band", type=mj.mjtGeom.mjGEOM_CYLINDER,
                        size=(0.17, 0.07, 0), pos=(0.3, 0.0, 0.42), rgba=HAT, **vis)
            th.add_geom(name="kebab_hat_puff", type=mj.mjtGeom.mjGEOM_ELLIPSOID,
                        size=(0.24, 0.24, 0.17), pos=(0.3, 0.0, 0.6), rgba=HAT, **vis)

    # ------------------------------------------------------------ attach / reset
    def on_attach(self) -> None:
        sim = self.sim
        m = sim.model
        c = self.cfg
        fly = sim.fly_name
        self.spin_qadr = int(m.jnt_qposadr[m.joint(P + "spin").id])
        self.spin_vadr = int(m.jnt_dofadr[m.joint(P + "spin").id])
        self.motor_id = m.actuator(P + "motor").id
        self.blade_gid = m.geom(f"{fly}/kebab_blade").id
        self.chunk_gid = np.array([m.geom(n).id for n in self._chunk_names])
        self.chunk_base_size = m.geom_size[self.chunk_gid].copy()
        self.chunk_col = np.array(self._chunk_colors)
        self.n_chunks = len(self.chunk_gid)
        self.scale = np.ones(self.n_chunks)  # 1 = full chunk on the spit, 0 = just cut
        self._dirty = False
        self.shav_qadr = []
        self.shav_vadr = []
        for i in range(c.n_shavings):
            j = m.joint(f"{P}shaving{i}_free").id
            self.shav_qadr.append(int(m.jnt_qposadr[j]))
            self.shav_vadr.append(int(m.jnt_dofadr[j]))
        self._next_shaving = 0
        # carving is standing still on purpose: keep the fall detector's "no progress"
        # window empty during it, as the session does for freeze / groom
        st = getattr(self.session, "STATIONARY_ACTIONS", None)
        if st is not None and "carve" not in st:
            self.session.STATIONARY_ACTIONS = (*st, "carve")
        # proboscis actuators (full body only)
        self.prob_ids = []
        for dof, sign in (("c_head-c_rostrum-pitch", -1.0), ("c_rostrum-c_haustellum-pitch", 1.0)):
            a = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, f"{fly}/{dof}-proboscispos")
            if a >= 0:
                self.prob_ids.append((a, sign))
        # counters (O(1) memory)
        self.shavings = 0
        self.kebabs = 0
        self.cuts_attempted_fast = 0
        self.n_carve_bouts = 0
        self.n_repositions = 0
        self.n_shavings_lost = 0
        self.max_rate = 0.0
        self.mn9_hz = 0.0
        self.proboscis = 0.0
        self.n_food = 0
        self.speed_mult = 1.0
        self._cut_times: deque = deque(maxlen=512)  # run times of recent cuts (rate)
        self._rate_t0 = self.run_time()
        self._next_food = 0.0
        self._update_div = 0
        self._last_msg = ""
        self.on_reset()

    def on_reset(self) -> None:
        self._tip_last: np.ndarray | None = None
        self._t_last_cut = -1e9
        self._since_start = 0.0
        self._reposition_until = -1.0
        self.state = "starting"
        self._carve_ok_since = None

    def reset_props(self) -> None:
        """A fresh kebab (all chunks back), shavings back on the tray (keyframe)."""
        self.scale[:] = 1.0
        self._apply_chunk_visuals(np.arange(self.n_chunks))
        self.sim.data.ctrl[self.motor_id] = self.spin_rad_s

    # ------------------------------------------------------------ helpers
    @property
    def spin_rad_s(self) -> float:
        return self.cfg.spin_rpm * 2 * math.pi / 60.0

    def blade_points(self) -> np.ndarray:
        """(3, 3) world points on the blade: middle, 3/4 and the tip."""
        d = self.sim.data
        g = self.blade_gid
        p = d.geom_xpos[g]
        ax = d.geom_xmat[g].reshape(3, 3)[:, 0]
        h = self.cfg.blade_len / 2
        return np.stack([p, p + ax * (h / 2), p + ax * h])

    def _apply_chunk_visuals(self, idx) -> None:
        m = self.sim.model
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            return
        s = self.scale[idx]
        g = self.chunk_gid[idx]
        m.geom_size[g] = self.chunk_base_size[idx] * np.maximum(s, 0.05)[:, None]
        # regrowing meat starts raw pink and browns in front of the heater
        w = np.clip(s, 0.0, 1.0)[:, None] ** 2
        col = (1 - w) * RAW + w * self.chunk_col[idx]
        m.geom_rgba[g, :3] = col
        m.geom_rgba[g, 3] = np.where(s < 0.08, 0.0, 1.0)

    def carving(self) -> bool:
        a = self.session.actions.action
        return a is not None and a.name == "carve"

    def rate_per_min(self) -> float:
        """Shavings per minute over the last 60 s of run time."""
        rt = self.run_time()
        n = 0
        for t in reversed(self._cut_times):
            if rt - t > 60.0:
                break
            n += 1
        window = min(60.0, rt - self._rate_t0)
        return n * 60.0 / max(window, 5.0)

    # ------------------------------------------------------------ behaviour
    def update(self) -> None:
        c = self.cfg
        sim = self.sim
        d = sim.data
        d.ctrl[self.motor_id] = self.spin_rad_s
        self._update_div += 1
        dt = c.update_every_steps * sim.timestep
        # regrow (every 10 ms)
        if self._update_div % 10 == 0:
            grow = self.scale < 1.0
            if grow.any():
                idx = np.flatnonzero(grow)
                self.scale[idx] = np.minimum(self.scale[idx] + 10 * dt / c.regrow_s, 1.0)
                self._apply_chunk_visuals(idx)
        link = getattr(self.session, "brain", None)
        if link is not None:
            self._read_mn9(link)
        if self.prob_ids:
            self._drive_proboscis(dt)
        if self.fly_down():
            self.state = "down"
            self._tip_last = None
            return
        acts = self.session.actions
        p = d.xpos[sim.thorax_body_id]
        dx, dy = p[0] - c.station_x, p[1] - c.station_y
        off = math.hypot(dx, dy)
        herr = abs(math.degrees(sim.heading()))
        t = sim.time
        if self.state == "reposition":
            self._reposition(off, dx, dy, herr)
            self._tip_last = None
            return
        if off > c.drift_tol or herr > c.heading_tol_deg:
            if t > 1.0:  # (the spawn settle may look off for a moment)
                self.state = "reposition"
                self.n_repositions += 1
                acts.cancel()
                self.say(f"drifted {off:.2f} mm / {herr:.0f} deg from the spit: walking back")
                return
        self.steering.set(0.0, 0.0)  # stand (the carve action drives the legs)
        if not acts.busy:
            acts.trigger(CarveStroke(duration=60.0, speed=c.carve_speed * self.speed_mult),
                         source="job")
            self.n_carve_bouts += 1
        if not self.carving():
            self.state = "raising knife"
            self._tip_last = None
            return
        a = acts.action
        a.speed = c.carve_speed * self.speed_mult
        self.state = "carving"
        pts = self.blade_points()
        tip = pts[-1]
        speed = 0.0 if self._tip_last is None else float(np.linalg.norm(tip - self._tip_last)) / dt
        self._tip_last = tip.copy()
        if speed < c.cut_speed or t - self._t_last_cut < c.cut_cooldown_s:
            return
        ripe = self.scale >= c.ripe_frac
        if not ripe.any():
            return
        centres = d.geom_xpos[self.chunk_gid]
        dist = np.min(np.linalg.norm(centres[None, :, :] - pts[:, None, :], axis=2), axis=0)
        dist[~ripe] = np.inf
        j = int(np.argmin(dist))
        if dist[j] < c.cut_radius:
            self._cut(j, speed)

    def _cut(self, j: int, speed: float) -> None:
        c = self.cfg
        sim = self.sim
        d = sim.data
        g = self.chunk_gid[j]
        pos = d.geom_xpos[g].copy()
        q = np.empty(4)
        mj.mju_mat2Quat(q, d.geom_xmat[g])
        sx, sy = self.spit_xy
        radial = np.array([pos[0] - sx, pos[1] - sy, 0.0])
        rn = np.linalg.norm(radial)
        radial = radial / rn if rn > 1e-9 else np.array([-1.0, 0.0, 0.0])
        omega = float(d.qvel[self.spin_vadr])
        v_surf = omega * np.array([-radial[1], radial[0], 0.0]) * rn
        i = self._next_shaving
        self._next_shaving = (i + 1) % c.n_shavings
        qa, va = self.shav_qadr[i], self.shav_vadr[i]
        d.qpos[qa:qa + 3] = pos + radial * 0.05
        d.qpos[qa + 3:qa + 7] = q
        rng = np.random.default_rng(self.shavings + 7)
        d.qvel[va:va + 3] = v_surf + radial * c.shaving_kick + np.array([0.0, 0.0, -2.0])
        d.qvel[va + 3:va + 6] = rng.normal(0.0, 8.0, 3)
        self.scale[j] = 0.0
        self._apply_chunk_visuals([j])
        self._t_last_cut = sim.time
        self.shavings += 1
        self.add_work(1.0)
        self._cut_times.append(self.run_time())
        if self.shavings % c.shavings_per_kebab == 0:
            self.kebabs += 1
            self.say(f"kebab #{self.kebabs} served ({self.shavings} shavings)")

    def _reposition(self, off: float, dx: float, dy: float, herr: float) -> None:
        """Walk back to the station (from behind it) facing the spit."""
        c = self.cfg
        acts = self.session.actions
        if acts.busy:
            return
        if dx > 0.25:  # too close to the spit: back off first
            from perpetualfly.actions import make_action

            acts.trigger(make_action("back_away", duration=0.4), source="job")
            return
        target = (c.station_x - 0.15, c.station_y)
        dist = self.steering.aim_at(target, 0.5)
        if dist < 0.25:
            if herr < 8.0:
                self.state = "starting"
                self._carve_ok_since = None
            else:
                self.steering.set(0.0, 0.4)
        elif dist < 0.6:
            # close: face the spit while creeping in
            self.steering.set(math.atan2(target[1] - self.fly_xy()[1],
                                         target[0] - self.fly_xy()[0]), 0.35)

    def _read_mn9(self, link) -> None:
        st = getattr(link, "latest", None)
        if st is not None:
            self.mn9_hz = float((getattr(st, "probes", None) or {}).get("MN9", 0.0))

    def _drive_proboscis(self, dt: float) -> None:
        target = min(max(self.mn9_hz / max(self.cfg.mn9_ref_hz, 1e-6), 0.0), 1.0)
        a = min(dt / 0.15, 1.0)
        self.proboscis += a * (target - self.proboscis)
        d = self.sim.data
        for aid, sign in self.prob_ids:
            d.ctrl[aid] = sign * 1.0 * self.proboscis

    # ------------------------------------------------------------ outside sim.step
    def after_physics(self) -> None:
        super().after_physics()
        c = self.cfg
        s = self.session
        rt = self.run_time()
        # stress: an aroused fly carves faster
        st = getattr(s, "stress", None)
        if st is not None and getattr(st, "enabled", False):
            fm = float(getattr(st, "freq_mult", 1.0))
            self.speed_mult = max(0.2, 1.0 + c.stress_speed_gain * (fm - 1.0))
        else:
            self.speed_mult = 1.0
        # brain: the kebab counts as food (sugar-taste neurons) every food_every_s
        link = getattr(s, "brain", None)
        if link is not None and rt >= self._next_food:
            from perpetualfly.brain.schema import StimulusEvent

            self._next_food = rt + c.food_every_s
            link.send(StimulusEvent("manual", "none", c.food_intensity, c.food_duration_s, rt,
                                    details={"set": "sugar", "label": "KEBAB (sugar GRNs)"}),
                      source="job")
            self.n_food += 1
            log = getattr(link, "stim_log", None)
            if isinstance(log, list) and len(log) > 400:
                del log[:-200]  # constant memory over an eternal run
        # shavings that left the arena (or went NaN): back onto the tray
        d = self.sim.data
        for i, qa in enumerate(self.shav_qadr):
            p = d.qpos[qa:qa + 3]
            if (not np.all(np.isfinite(p))) or p[2] < -0.5 or abs(p[0]) > 60 or abs(p[1]) > 60:
                self._park_shaving(i)
                self.n_shavings_lost += 1
        self.max_rate = max(self.max_rate, self.rate_per_min())

    def _park_shaving(self, i: int) -> None:
        d = self.sim.data
        qa, va = self.shav_qadr[i], self.shav_vadr[i]
        d.qpos[qa:qa + 7] = self._park[i]
        d.qvel[va:va + 6] = 0.0

    # ------------------------------------------------------------ view / HUD
    def camera_target(self) -> np.ndarray:
        sx, sy = self.spit_xy
        p = self.sim.thorax_position()
        return np.array([0.55 * p[0] + 0.45 * (sx - 0.6), 0.5 * (p[1] + sy), 2.2])

    def camera_preset(self) -> CameraPreset:
        # from the fly's right (-y), a little from the front and above
        return CameraPreset(azimuth=58.0, elevation=-10.0, distance=9.5, tau_s=0.5)

    def job_hud_lines(self) -> list[str]:
        c = self.cfg
        ug = self.shavings * c.slice_ug
        human_kg = ug * 1e-9 * HUMAN_SCALE
        lines = [
            f"kebabs {self.kebabs} (every {c.shavings_per_kebab})   served {ug / 1000:.3f} mg"
            f" = {human_kg:,.0f} kg human-scale   {self.rate_per_min():.0f}/min",
            f"spit {c.spin_rpm:.0f} rpm   regrowing {int(np.sum(self.scale < 1.0))}/{self.n_chunks}"
            + (f"   stress x{self.speed_mult:.2f}" if self.speed_mult != 1.0 else "")
            + (f"   MN9 {self.mn9_hz:.0f} Hz  proboscis {self.proboscis:.0%}"
               if getattr(self.session, "brain", None) is not None else ""),
            "knife stroke = a real recorded fly grooming bout",
        ]
        return lines

    def job_stats(self) -> dict:
        return {"shavings": self.shavings, "kebabs": self.kebabs,
                "served_ug": self.shavings * self.cfg.slice_ug,
                "rate_per_min": self.rate_per_min(), "max_rate_per_min": self.max_rate,
                "regrowing": int(np.sum(self.scale < 1.0)), "carve_bouts": self.n_carve_bouts,
                "repositions": self.n_repositions, "shavings_lost": self.n_shavings_lost,
                "food_stimuli": self.n_food, "mn9_hz": self.mn9_hz}
