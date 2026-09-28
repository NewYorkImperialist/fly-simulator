"""Fry cook: the fly works a fast-food fry station forever, putting fries in the bag.

A fly-scale fry station on a stainless counter ("FLY FRIES", a generic fast-food look:
our own name and badge, no real brand). The fly stands at the station facing +x:

* **Fryer (a machine, kinematic, labelled).** A stainless fryer with hot oil (animated
  shimmer, rising bubbles, steam). A wire basket on an overhead AUTO-LIFT gantry
  cooks a batch (the fries turn from pale to golden), lifts out and drips, and, when
  the holding bin runs low, swings over the bin and tips: the fries (pooled free
  bodies with capsule colliders) are **physically** in the basket from the drip on
  (the basket's walls are colliders on a mocap body), slide out over its lip and fall
  into the bin with real contacts. Raw fries are loaded into the basket from an
  unseen freezer (a hidden recycle of the pool, labelled).
* **Salt shaker (kinematic, labelled).** After each dump it flies over the bin and
  shakes; its salt level drops and it is refilled when empty (counted).
* **The fly** holds a stance (all tarsi planted and adhering); the job drives both
  front legs by damped least-squares IK (the dead_hang ``LegIK``), position-controlled
  like any action. The **left front leg** pulls a carton off the stack to the fill
  spot; the **right front leg holds the fry scoop** (a kinematic prop that follows the
  tarsus; its tilt is set by the job). A scoop: over the bin, dip, drag; the fries
  under the scoop are **picked up by a labelled kinematic transfer** (they ride in the
  scoop), carried over the carton, the scoop tips and they drop into the carton with
  **real physics** (the carton's walls are colliders) and settle there. Now and then a
  fry slips off the scoop and lands / rolls on the counter (a spill); the fly may
  sneak it with its left front leg to its mouth (a taste: with ``--brain`` the
  connectome's MN9 drives the proboscis, **stand-in taste sets, labelled**).
* **Order up.** A full carton slides onto the tray at the pass window (kinematic), the
  bell dings, the ORDER UP sign lights, the tray goes through the window and comes
  back empty; a new carton. When the bin runs low the basket dumps the next batch.

Constant memory: fixed pools (fries, cartons, particles), counters only. Fries never
touch the fly; they touch the bin, carton and basket colliders, the counter and each
other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from fly_simulator.actions.base import smoothstep
from fly_simulator.jobs import fry_cook_assets as A
from fly_simulator.jobs import kebab_assets as KA
from fly_simulator.jobs import taste_tester_assets as TA
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import add_box, contact_kwargs, quat_axis_angle
from fly_simulator.jobs.pizza_chef import ChefStance, _Puffs, _qmul, _yaw_quat
from fly_simulator.jobs.geometry import spot_or_directional
from fly_simulator.jobs.registry import register_job
from fly_simulator.jobs.taste_tester import DIGITS, SEGS

F = "fry/"
COL_BIT = 64  # the bin / carton / basket colliders (props only)
FRY_BIT = 128  # fries touch each other
TERRAIN_BIT = 16

# fry states
PARK, BASKET, BASKET_LIVE, BIN, SCOOP, DROP, CARTON, RIDE, SPILL, SPILL_REST, SNEAK = range(11)
LIVE_STATES = (BASKET_LIVE, BIN, DROP, CARTON, SPILL)
N_CARTONS = 3


class CookStance(ChefStance):
    """The fry cook's stance: all tarsi planted and adhering (like ``freeze``) while the
    job moves the front legs (``pose`` / ``w`` per leg, see ``ChefStance``)."""

    name = "fry_stance"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class FryCookConfig(JobConfig):
    # ---- layout (the fly spawns at the origin facing +x; thorax settles at ~(0.61, 0, 1.0))
    bin_x: float = 2.12  # holding bin (inner centre, half sizes)
    bin_y: float = -0.62
    bin_hx: float = 0.42
    bin_hy: float = 0.36
    bin_floor: float = 0.05
    bin_wall: float = 0.2
    fill_x: float = 1.72  # carton fill spot
    fill_y: float = 0.30
    stack_x: float = 1.22  # carton stack
    stack_y: float = 1.08
    tray_x: float = 1.72  # tray at the pass window
    tray_y: float = 1.95
    window_y: float = 2.6  # the back wall with the pass window
    fryer_x: float = 3.55
    fryer_y: float = -0.6
    oil_z: float = 0.8
    # ---- fries (pooled free bodies) ------------------------------------------------
    n_fries: int = 48
    batch: int = 14
    fry_r: float = 0.024  # capsule radius (collider)
    fry_half: float = 0.19  # capsule half length
    fry_mass: float = 1e-6  # g (1 ug)
    low_mark: int = 8  # bin count below this -> dump the next batch
    cook_s: float = 6.0
    # ---- scooping / cartons --------------------------------------------------------
    carton_min: int = 6  # fries per carton (random in [min, max])
    carton_max: int = 8
    scoop_cap: int = 5
    max_scoops: int = 5
    spill_p: float = 0.15  # a fry slips off the scoop on the way
    sneak_p: float = 0.6  # the fly sneaks a spilled fry in reach
    # ---- salt ---------------------------------------------------------------------------
    salt_per_shake: float = 0.07
    # ---- taste (--brain; stand-in sets, labelled) ------------------------------------
    taste_s: float = 0.5
    sugar_hz: float = 150.0  # low salt / starch: the appetitive (sweet GRN) stand-in
    bitter_hz: float = 160.0  # over-salted: high salt recruits bitter GRNs (stand-in)
    oversalt: float = 1.3  # batch saltiness above this also drives the bitter set
    mn9_ref_hz: float = 60.0
    # ---- presentation (edit effects, labelled) ----------------------------------------
    slowmo: bool = True
    dump_slowmo: float = 0.3
    drop_slowmo: float = 0.5
    close_ups: bool = True
    shadows: bool = True
    captions: bool = True
    stuck_timeout_s: float = 90.0


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _quat_mat(q) -> np.ndarray:
    R = np.empty(9)
    mj.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def _yaw_pitch_quat(yaw: float, pitch: float) -> np.ndarray:
    """Yaw about z, then pitch about the body's y (positive = nose down)."""
    return _qmul(_yaw_quat(yaw), quat_axis_angle((0, 1, 0), pitch))


def _slerp(q0, q1, u: float) -> np.ndarray:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


class _Particles:
    """A fixed pool of small spheres on mocap bodies with ballistic motion (salt
    grains, oil drips, rising bubbles): p = p0 + v t + g t^2 / 2 until ``life`` or
    below ``floor``; alpha fades out over the last 30 %."""

    def __init__(self, n: int, rgb, life: float, r: float, alpha: float, g: float) -> None:
        self.n, self.rgb, self.life, self.r, self.alpha, self.g = n, rgb, life, r, alpha, g
        self.t0 = np.full(n, -1e9)
        self.p0 = np.zeros((n, 3))
        self.v = np.zeros((n, 3))
        self.floor = np.full(n, -1e9)
        self.grow = np.zeros(n)
        self.k = 0

    def spawn(self, t, p, v, floor: float = -1e9, grow: float = 0.0) -> None:
        i = self.k % self.n
        self.k += 1
        self.t0[i], self.p0[i], self.v[i], self.floor[i], self.grow[i] = t, p, v, floor, grow

    def clear(self) -> None:
        self.t0[:] = -1e9

    def draw(self, t, m, d, mids, gids) -> None:
        for i in range(self.n):
            tt = t - self.t0[i]
            g = gids[i]
            if not 0.0 <= tt < self.life:
                if m.geom_rgba[g, 3] != 0.0:
                    m.geom_rgba[g, 3] = 0.0
                    d.mocap_pos[mids[i]] = (0.0, 0.0, -5.0)
                continue
            p = self.p0[i] + self.v[i] * tt
            p[2] += 0.5 * self.g * tt * tt
            if p[2] < self.floor[i]:
                self.t0[i] = -1e9
                continue
            d.mocap_pos[mids[i]] = p
            u = tt / self.life
            m.geom_size[g, 0] = self.r * (1.0 + self.grow[i] * u)
            m.geom_rgba[g, :3] = self.rgb
            m.geom_rgba[g, 3] = self.alpha * min((1.0 - u) / 0.3, 1.0)


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------


@register_job
class FryCookJob(EternalJob):
    name = "fry_cook"
    znear = 0.05  # depth precision for the shadow map (applied after compile)
    title = "FRY COOK FLY"
    tagline = "the fly works the fry station forever"
    work_label = "orders served"
    config_cls = FryCookConfig
    required_names = (F + "basket", F + "scoop", F + "carton0", F + "bin_floor", F + "fry0")

    cfg: FryCookConfig

    # geometry of the props (mm)
    CARTON = dict(hx0=0.12, hy0=0.14, hx1=0.15, hy1=0.18, h=0.34, wave=0.06)
    BASKET_H = (0.38, 0.28, 0.14)  # inner half sizes of the basket (z: floor at -0.13)
    BOWL_OFF = np.array([0.3, 0.0, -0.04])  # scoop: grip (the tarsus) -> bowl centre

    def configure_app(self, app_cfg) -> None:
        super().configure_app(app_cfg)
        app_cfg.controller.target_heading_deg = 0.0
        app_cfg.fly.extra_joints = True  # proboscis joints (the sneaked-fry taste)

    # ------------------------------------------------------------ build
    def extension(self, world) -> None:
        spec = world.mjcf_root
        self._materials(spec)
        self._add_fries(spec)
        self._add_bin(spec)
        self._add_fryer(spec)
        self._add_cartons(spec)
        self._add_tools(spec)
        self._add_room(spec)

    def _materials(self, spec) -> None:
        c = self.cfg
        mat = spec.material("grid")
        if mat is not None:  # the floor plane is the stainless counter top
            KA.add_texture(spec, F + "tex_counter", A.counter_texture(c.seed))
            mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = F + "tex_counter"
            mat.rgba = (1.0, 1.0, 1.0, 1.0)
            mat.reflectance = 0.05
            mat.specular = 0.6
            mat.shininess = 0.7
        tex = (("fry", A.fry_texture(c.seed), dict(specular=0.35, shininess=0.4)),
               ("oil", A.oil_texture(c.seed), dict(specular=0.9, shininess=0.9, emission=0.25)),
               ("carton", A.carton_texture(), dict(specular=0.25, shininess=0.4)),
               ("tray", A.tray_texture(), dict(specular=0.3, shininess=0.4)),
               ("tiles", A.tile_wall_texture(c.seed), dict(specular=0.5, shininess=0.6)),
               ("fryer_front", A.fryer_front_texture(), dict(specular=0.6, shininess=0.6)),
               ("menu", A.menu_board_texture(), dict(emission=0.55)),
               ("sign_off", A.order_sign_texture(False), dict(emission=0.2)),
               ("sign_on", A.order_sign_texture(True), dict(emission=1.0)),
               ("dining", A.dining_texture(), dict(emission=0.35)),
               ("lbl_lift", A.label_texture("AUTO-LIFT", (1.0, 0.8, 0.12), (0.1, 0.1, 0.1)), dict(emission=0.1)),
               ("lbl_salt", A.label_texture("SALT", (0.96, 0.96, 0.96), (0.1, 0.1, 0.12)), dict(emission=0.1)),
               ("lbl_hold", A.label_texture("HOT & READY", A.RED, (1.0, 0.95, 0.85)), dict(emission=0.2)),
               ("lbl_orders", A.label_texture("ORDERS", (0.1, 0.1, 0.1), (1.0, 0.3, 0.2)), dict(emission=0.3)))
        for nm, img, kw in tex:
            KA.add_texture(spec, f"{F}tex_{nm}", img)
            KA.add_textured_material(spec, F + nm, f"{F}tex_{nm}", rgba=(1, 1, 1, 1), **kw)
        self._ticket_mats = []
        for k in range(4):
            KA.add_texture(spec, f"{F}tex_ticket{k}", A.ticket_texture(41 + k))
            KA.add_textured_material(spec, f"{F}ticket{k}", f"{F}tex_ticket{k}", rgba=(1, 1, 1, 1),
                                     emission=0.15)
        KA.add_texture(spec, F + "tex_steel", KA.steel_texture(c.seed))
        KA.add_textured_material(spec, F + "steel", F + "tex_steel", rgba=(1, 1, 1, 1), specular=0.9,
                                 shininess=0.9, texuniform=True, texrepeat=(2.0, 2.0))
        spec.add_material(name=F + "oil_mat", rgba=(1, 1, 1, 0.82), specular=1.0, shininess=1.0,
                          emission=0.25)
        spec.materials[-1].textures[mj.mjtTextureRole.mjTEXROLE_RGB] = F + "tex_oil"
        spec.add_material(name=F + "wire", rgba=(0.75, 0.76, 0.78, 1), specular=1.0, shininess=0.9)
        spec.add_material(name=F + "dark", rgba=(0.08, 0.08, 0.09, 1), specular=0.4, shininess=0.5)
        spec.add_material(name=F + "black", rgba=(0.03, 0.03, 0.035, 1), specular=0.5, shininess=0.6)
        spec.add_material(name=F + "red", rgba=(0.78, 0.08, 0.07, 1), specular=0.5, shininess=0.6)
        spec.add_material(name=F + "yellow", rgba=(1.0, 0.78, 0.12, 1), specular=0.5, shininess=0.6)
        spec.add_material(name=F + "glass", rgba=(0.85, 0.93, 0.97, 0.35), specular=1.0, shininess=1.0)
        spec.add_material(name=F + "salt", rgba=(0.98, 0.98, 0.98, 1), specular=0.3)
        spec.add_material(name=F + "bulb", rgba=(1.0, 0.55, 0.15, 1), emission=1.0)
        spec.add_material(name=F + "glow", rgba=(1.0, 0.45, 0.1, 0.25), emission=1.0)
        spec.add_material(name=F + "seg_on", rgba=(1.0, 0.18, 0.1, 1), emission=1.0)
        spec.add_material(name=F + "seg_off", rgba=(0.09, 0.03, 0.03, 1), emission=0.0, specular=0.1)
        spec.add_material(name=F + "brass", rgba=(0.9, 0.72, 0.3, 1), specular=1.0, shininess=0.9)
        spec.add_material(name=F + "paper", rgba=(0.97, 0.96, 0.92, 1), specular=0.05)

    def _add_fries(self, spec) -> None:
        """The pool: capsule colliders (hidden, render group 3) under fry meshes."""
        c = self.cfg
        wb = spec.worldbody
        vm = dict(contact_kwargs("visual"), mass=0.0)
        for k in range(4):
            KA.add_mesh(spec, f"{F}fry_mesh{k}",
                        A.fry_mesh(2 * c.fry_half + 2 * c.fry_r * (0.6 + 0.25 * k), 2.1 * c.fry_r,
                                   bend=0.008 * (k - 1.5), seed=k))
        self._fry_names = []
        for j in range(c.n_fries):
            nm = f"{F}fry{j}"
            b = wb.add_body(name=nm, pos=(-6.0 - 0.12 * j, 6.0, -3.0))
            b.add_freejoint(name=nm + "_free")
            b.add_geom(name=nm + "_col", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(c.fry_r, c.fry_half, 0),
                       mass=c.fry_mass, contype=FRY_BIT, conaffinity=FRY_BIT | COL_BIT | TERRAIN_BIT, condim=3,
                       friction=(0.7, 0.005, 0.0015), solref=(0.0005, 1.0),
                       rgba=(1, 1, 1, 0), group=3)
            b.add_geom(name=nm + "_vis", type=mj.mjtGeom.mjGEOM_MESH, meshname=f"{F}fry_mesh{j % 4}",
                       material=F + "fry", rgba=(1, 1, 1, 1), **vm)
            b.gravcomp = 1.0
            self._fry_names.append(nm)

    def _col(self, **kw) -> dict:
        d = dict(contype=COL_BIT, conaffinity=0, condim=3, friction=(0.7, 0.005, 0.0001),
                 solref=(0.0005, 1.0), rgba=(1, 1, 1, 0), group=3)
        d.update(kw)
        return d

    def _add_bin(self, spec) -> None:
        """The heated holding bin (static): a stainless tray with thick hidden
        colliders, the heat lamp above it (a warm point light)."""
        c = self.cfg
        wb = spec.worldbody
        vis = contact_kwargs("visual")
        bx, by, hx, hy = c.bin_x, c.bin_y, c.bin_hx, c.bin_hy
        t = 0.03
        KA.add_mesh(spec, F + "bin_mesh", A.box_shell_mesh(hx + t, hy + t, c.bin_wall, t))
        wb.add_geom(name=F + "bin", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "bin_mesh",
                    pos=(bx, by, 0.0), material=F + "steel", **vis)
        # perforated false floor (visual) and the colliders (thick: v * tau sink)
        add_box(wb, F + "bin_plate", (hx, hy, 0.004), (bx, by, c.bin_floor - 0.004),
                material=F + "wire", collide="visual")
        col = self._col()
        wb.add_geom(name=F + "bin_floor", type=mj.mjtGeom.mjGEOM_BOX, size=(hx + 0.2, hy + 0.2, 0.3),
                    pos=(bx, by, c.bin_floor - 0.3), **col)
        H = c.bin_wall + 0.06
        for s in (-1, 1):
            wb.add_geom(name=f"{F}bin_wx{s}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.15, hy + 0.3, H / 2),
                        pos=(bx + s * (hx + 0.15), by, H / 2), **col)
            wb.add_geom(name=f"{F}bin_wy{s}", type=mj.mjtGeom.mjGEOM_BOX, size=(hx + 0.3, 0.15, H / 2),
                        pos=(bx, by + s * (hy + 0.15), H / 2), **col)
        # "HOT & READY" strip on the bin's front (camera side, -y)
        KA.add_mesh(spec, F + "hold_lbl_mesh", TA.front_panel_mesh(0.3, 0.06, 0.006))
        wb.add_geom(name=F + "bin_label", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "hold_lbl_mesh",
                    pos=(bx, by - hy - t - 0.005, 0.11), quat=quat_axis_angle((0, 0, 1), -math.pi / 2),
                    material=F + "lbl_hold", **vis)
        # heat lamp: hood, bulbs, a rod up out of the frame, the warm light
        lx, lz = bx - 0.1, 1.9
        KA.add_mesh(spec, F + "hood_mesh", A.lamp_hood_mesh(0.26, 0.36, 0.18))
        wb.add_geom(name=F + "hood", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "hood_mesh",
                    pos=(lx, by, lz), material=F + "red", **vis)
        self._bulb_names = []
        for j, dy in enumerate((-0.15, 0.0, 0.15)):
            nm = f"{F}bulb{j}"
            wb.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(0.06, 0.06, 0.04),
                        pos=(lx, by + dy, lz + 0.05), material=F + "bulb", **vis)
            self._bulb_names.append(nm)
        wb.add_geom(name=F + "lamp_glow", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.28, 0.002, 0),
                    pos=(lx, by, lz - 0.002), material=F + "glow", **vis)
        wb.add_geom(name=F + "lamp_rod", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.02, 1.5, 0),
                    pos=(lx, by, lz + 0.22 + 1.5), material=F + "steel", **vis)
        wb.add_light(name=F + "lamp_light", type=mj.mjtLightType.mjLIGHT_POINT,
                     pos=(lx, by, lz - 0.15), diffuse=(0.75, 0.38, 0.12), specular=(0.2, 0.1, 0.04),
                     attenuation=(0.4, 0.6, 0.35), castshadow=False)

    def _add_fryer(self, spec) -> None:
        """The fryer (visual), its oil, the basket (mocap with hidden colliders) on the
        AUTO-LIFT gantry, bubbles / drips / steam pools."""
        c = self.cfg
        wb = spec.worldbody
        vis = contact_kwargs("visual")
        vm = dict(vis, mass=0.0)
        fx, fy = c.fryer_x, c.fryer_y
        hx, hy, H = 0.62, 0.62, 0.95
        t = 0.05
        KA.add_mesh(spec, F + "vat_mesh", A.box_shell_mesh(hx, hy, H, t))
        wb.add_geom(name=F + "fryer", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "vat_mesh",
                    pos=(fx, fy, 0.0), material=F + "steel", **vis)
        KA.add_mesh(spec, F + "fryer_front_mesh", TA.front_panel_mesh(0.5, 0.32, 0.006))
        wb.add_geom(name=F + "fryer_front", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "fryer_front_mesh",
                    pos=(fx, fy - hy - 0.004, 0.42), quat=quat_axis_angle((0, 0, 1), -math.pi / 2),
                    material=F + "fryer_front", **vis)
        add_box(wb, F + "vat_dark", (hx - t, hy - t, 0.01), (fx, fy, c.oil_z - 0.45), material=F + "dark",
                collide="visual")
        add_box(wb, F + "oil", (hx - t, hy - t, 0.004), (fx, fy, c.oil_z), material=F + "oil_mat",
                collide="visual")
        # the basket: wire frame + a handle to the lift yoke; hidden colliders
        bhx, bhy, bhz = self.BASKET_H
        bk = wb.add_body(name=F + "basket", pos=(fx, fy, 1.45), mocap=True)
        w = 0.008
        for s in (-1, 1):  # rims (top) and bottom frame
            for z in (bhz, -bhz):
                bk.add_geom(name=f"{F}bk_rx{s}{z > 0}", type=mj.mjtGeom.mjGEOM_BOX, size=(bhx, w, w),
                            pos=(0, s * bhy, z), material=F + "wire", **vm)
                bk.add_geom(name=f"{F}bk_ry{s}{z > 0}", type=mj.mjtGeom.mjGEOM_BOX, size=(w, bhy, w),
                            pos=(s * bhx, 0, z), material=F + "wire", **vm)
            for sx in (-1, 1):
                bk.add_geom(name=f"{F}bk_post{s}{sx}", type=mj.mjtGeom.mjGEOM_BOX, size=(w, w, bhz),
                            pos=(sx * bhx, s * bhy, 0), material=F + "wire", **vm)
        for k, x in enumerate(np.linspace(-bhx, bhx, 9)[1:-1]):  # wires across the floor and sides
            bk.add_geom(name=f"{F}bk_fx{k}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.004, bhy, 0.004),
                        pos=(x, 0, -bhz), material=F + "wire", **vm)
            for s in (-1, 1):
                bk.add_geom(name=f"{F}bk_sx{k}{s}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.004, 0.004, bhz),
                            pos=(x, s * bhy, 0), material=F + "wire", **vm)
        for k, y in enumerate(np.linspace(-bhy, bhy, 7)[1:-1]):
            bk.add_geom(name=f"{F}bk_fy{k}", type=mj.mjtGeom.mjGEOM_BOX, size=(bhx, 0.004, 0.004),
                        pos=(0, y, -bhz), material=F + "wire", **vm)
            for s in (-1, 1):
                bk.add_geom(name=f"{F}bk_sy{k}{s}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.004, 0.004, bhz),
                            pos=(s * bhx, y, 0), material=F + "wire", **vm)
        for z in (-bhz / 3, bhz / 3):
            for s in (-1, 1):
                bk.add_geom(name=f"{F}bk_hx{s}{z > 0}", type=mj.mjtGeom.mjGEOM_BOX, size=(bhx, 0.004, 0.004),
                            pos=(0, s * bhy, z), material=F + "wire", **vm)
                bk.add_geom(name=f"{F}bk_hy{s}{z > 0}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.004, bhy, 0.004),
                            pos=(s * bhx, 0, z), material=F + "wire", **vm)
        bk.add_geom(name=F + "bk_handle", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.018, 0.2, 0),
                    pos=(bhx + 0.2, 0, bhz), quat=quat_axis_angle((0, 1, 0), math.pi / 2),
                    material=F + "black", **vm)
        bk.add_geom(name=F + "bk_yoke", type=mj.mjtGeom.mjGEOM_BOX, size=(0.03, 0.05, 0.03),
                    pos=(bhx + 0.42, 0, bhz), material=F + "steel", **vm)
        kc = self._col(friction=(0.6, 0.005, 0.0001))
        bk.add_geom(name=F + "bk_col_floor", type=mj.mjtGeom.mjGEOM_BOX, size=(bhx + 0.08, bhy + 0.08, 0.06),
                    pos=(0, 0, -bhz - 0.06), **kc)
        for s in (-1, 1):
            bk.add_geom(name=f"{F}bk_col_x{s}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.05, bhy + 0.08, bhz + 0.06),
                        pos=(s * (bhx + 0.05), 0, 0), **kc)
            bk.add_geom(name=f"{F}bk_col_y{s}", type=mj.mjtGeom.mjGEOM_BOX, size=(bhx + 0.08, 0.05, bhz + 0.06),
                        pos=(0, s * (bhy + 0.05), 0), **kc)
        # gantry: a post behind the fryer, a rail along x, a trolley and a hanger rod
        rail_z = 2.55
        px = fx + hx + 0.3
        wb.add_geom(name=F + "lift_post", type=mj.mjtGeom.mjGEOM_BOX, size=(0.05, 0.05, rail_z / 2),
                    pos=(px, fy, rail_z / 2), material=F + "steel", **vis)
        x0 = c.bin_x + 0.1
        wb.add_geom(name=F + "lift_rail", type=mj.mjtGeom.mjGEOM_BOX, size=((px - x0) / 2 + 0.05, 0.04, 0.035),
                    pos=((px + x0) / 2, fy, rail_z), material=F + "steel", **vis)
        KA.add_mesh(spec, F + "lift_lbl_mesh", TA.front_panel_mesh(0.22, 0.055, 0.006))
        wb.add_geom(name=F + "lift_label", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "lift_lbl_mesh",
                    pos=(px, fy - 0.056, 1.6), quat=quat_axis_angle((0, 0, 1), -math.pi / 2),
                    material=F + "lbl_lift", **vis)
        tr = wb.add_body(name=F + "trolley", pos=(fx + 0.8, fy, rail_z - 0.08), mocap=True)
        tr.add_geom(name=F + "trolley_g", type=mj.mjtGeom.mjGEOM_BOX, size=(0.09, 0.07, 0.05),
                    material=F + "yellow", **vm)
        hg = wb.add_body(name=F + "hanger", pos=(fx + 0.8, fy, 2.0), mocap=True)
        hg.add_geom(name=F + "hanger_g", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.014, 0.4, 0),
                    material=F + "steel", **vm)
        self.rail_z = rail_z
        # pools: bubbles in the oil, drips, steam, salt grains
        self._pnames = {"bubble": [], "drip": [], "salt": [], "steam": []}
        for kind, n, mat, r in (("bubble", 36, F + "glass", 0.02), ("drip", 14, F + "oil_mat", 0.012),
                                ("salt", 30, F + "salt", 0.008), ("steam", 14, None, 0.05)):
            for j in range(n):
                b = wb.add_body(name=f"{F}{kind}{j}", pos=(0, 0, -5.0), mocap=True)
                kw = dict(material=mat) if mat else {}
                b.add_geom(name=f"{F}{kind}{j}_g", type=mj.mjtGeom.mjGEOM_SPHERE, size=(r, 0, 0),
                           rgba=(1, 1, 1, 0), **kw, **vm)
                self._pnames[kind].append(f"{F}{kind}{j}")

    def _add_cartons(self, spec) -> None:
        """Three pooled cartons (mocap; hidden colliders, switched on while filling),
        a static nested stack of empty cartons, and the tray at the window."""
        c = self.cfg
        wb = spec.worldbody
        vis = contact_kwargs("visual")
        vm = dict(vis, mass=0.0)
        C = self.CARTON
        KA.add_mesh(spec, F + "carton_mesh", A.cup_mesh(C["hx0"], C["hy0"], C["hx1"], C["hy1"], C["h"],
                                                        0.012, 0.02, rim_wave=C["wave"]))
        ihx, ihy = 0.5 * (C["hx0"] + C["hx1"]) - 0.012, 0.5 * (C["hy0"] + C["hy1"]) - 0.012
        self.carton_inner = (ihx, ihy)
        Hc = C["h"] + 0.02  # no higher than the rim: a fry must not bridge the walls
        cc = self._col(friction=(0.8, 0.005, 0.0001))
        for k in range(N_CARTONS):
            b = wb.add_body(name=f"{F}carton{k}", pos=(-5.0 - k, 5.0, -3.0), mocap=True)
            b.add_geom(name=f"{F}carton{k}_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "carton_mesh",
                       material=F + "carton", **vm)
            b.add_geom(name=f"{F}carton{k}_col_floor", type=mj.mjtGeom.mjGEOM_BOX,
                       size=(ihx + 0.1, ihy + 0.1, 0.2), pos=(0, 0, 0.02 - 0.2), **cc)
            for s in (-1, 1):
                b.add_geom(name=f"{F}carton{k}_col_x{s}", type=mj.mjtGeom.mjGEOM_BOX,
                           size=(0.08, ihy + 0.1, Hc / 2), pos=(s * (ihx + 0.08), 0, Hc / 2), **cc)
                b.add_geom(name=f"{F}carton{k}_col_y{s}", type=mj.mjtGeom.mjGEOM_BOX,
                           size=(ihx + 0.1, 0.08, Hc / 2), pos=(0, s * (ihy + 0.08), Hc / 2), **cc)
        for k in range(4):  # the nested stack (static decoration)
            wb.add_geom(name=f"{F}stack{k}", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "carton_mesh",
                        pos=(c.stack_x, c.stack_y, 0.022 * k), material=F + "carton", **vis)
        # the tray (mocap: slides through the window and back)
        tr = wb.add_body(name=F + "tray", pos=(c.tray_x, c.tray_y, 0.0), mocap=True)
        KA.add_mesh(spec, F + "tray_top_mesh", TA.flat_quad_mesh(0.32, 0.4, 0.004))
        tr.add_geom(name=F + "tray_base", type=mj.mjtGeom.mjGEOM_BOX, size=(0.34, 0.42, 0.012),
                    pos=(0, 0, 0.012), material=F + "red", **vm)
        tr.add_geom(name=F + "tray_top", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "tray_top_mesh",
                    pos=(0, 0, 0.026), material=F + "tray", **vm)
        for s in (-1, 1):
            tr.add_geom(name=f"{F}tray_lip_x{s}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.012, 0.42, 0.02),
                        pos=(s * 0.34, 0, 0.03), material=F + "red", **vm)
            tr.add_geom(name=f"{F}tray_lip_y{s}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.34, 0.012, 0.02),
                        pos=(0, s * 0.42, 0.03), material=F + "red", **vm)

    def _add_tools(self, spec) -> None:
        """The fry scoop (mocap, follows the right front tarsus), the salt shaker, the
        service bell."""
        c = self.cfg
        wb = spec.worldbody
        vm = dict(contact_kwargs("visual"), mass=0.0)
        sc = wb.add_body(name=F + "scoop", pos=(1.3, -0.9, 0.2), mocap=True)
        bo = self.BOWL_OFF
        KA.add_mesh(spec, F + "scoop_mesh", A.cup_mesh(0.1, 0.085, 0.14, 0.12, 0.09, 0.01, 0.01, e=0.5))
        sc.add_geom(name=F + "scoop_bowl", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "scoop_mesh",
                    pos=(bo[0], bo[1], bo[2] - 0.035), material=F + "steel", **vm)
        sc.add_geom(name=F + "scoop_handle", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.018, 0.09, 0),
                    pos=(0.07, 0, 0.0), quat=quat_axis_angle((0, 1, 0), math.pi / 2 - 0.3),
                    material=F + "red", **vm)
        sc.add_geom(name=F + "scoop_neck", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.01, 0.045, 0),
                    pos=(0.17, 0, -0.02), quat=quat_axis_angle((0, 1, 0), math.pi / 2 + 0.4),
                    material=F + "steel", **vm)
        # salt shaker: body (glass), the salt inside (height = level), cap, label
        sh = wb.add_body(name=F + "shaker", pos=(0, 0, 0), mocap=True)
        self.shaker_r, self.shaker_h = 0.11, 0.3
        KA.add_mesh(spec, F + "shaker_mesh", A.shaker_mesh(self.shaker_r, self.shaker_h))
        KA.add_mesh(spec, F + "shaker_cap_mesh", A.shaker_cap_mesh(self.shaker_r * 0.95, 0.1))
        sh.add_geom(name=F + "shaker_glass", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "shaker_mesh",
                    material=F + "glass", **vm)
        sh.add_geom(name=F + "shaker_salt", type=mj.mjtGeom.mjGEOM_CYLINDER,
                    size=(self.shaker_r * 0.82, 0.12, 0), pos=(0, 0, 0.13), material=F + "salt", **vm)
        sh.add_geom(name=F + "shaker_cap", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "shaker_cap_mesh",
                    pos=(0, 0, self.shaker_h), material=F + "steel", **vm)
        KA.add_mesh(spec, F + "salt_lbl_mesh", TA.front_panel_mesh(0.07, 0.03, 0.004))
        sh.add_geom(name=F + "shaker_label", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "salt_lbl_mesh",
                    pos=(0, -self.shaker_r - 0.004, 0.15), quat=quat_axis_angle((0, 0, 1), -math.pi / 2),
                    material=F + "lbl_salt", **vm)
        # service bell by the window (the plunger moves on a ding)
        bx, by = c.tray_x + 0.7, c.tray_y + 0.1
        KA.add_mesh(spec, F + "bell_mesh", A.bell_mesh(0.1))
        wb.add_geom(name=F + "bell", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "bell_mesh",
                    pos=(bx, by, 0.0), material=F + "brass", **contact_kwargs("visual"))
        pl = wb.add_body(name=F + "plunger", pos=(bx, by, 0.1), mocap=True)
        pl.add_geom(name=F + "plunger_g", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.012, 0.03, 0),
                    pos=(0, 0, 0.02), material=F + "steel", **vm)
        rg = wb.add_body(name=F + "ring", pos=(bx, by, -5), mocap=True)
        rg.add_geom(name=F + "ring_g", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.1, 0.002, 0),
                    material=F + "brass", rgba=(1, 1, 1, 0), **vm)
        self.bell_xy = (bx, by)

    def _add_room(self, spec) -> None:
        """The tiled back wall with the pass window ("ORDER UP" light box, ticket rail,
        an ORDERS counter), the dining room behind it, the menu board; lights."""
        c = self.cfg
        wb = spec.worldbody
        vis = contact_kwargs("visual")
        vm = dict(vis, mass=0.0)
        yw = c.window_y
        q_back = quat_axis_angle((0, 0, 1), -math.pi / 2)  # a front panel facing -y
        # window opening: x in [wx0, wx1], z in [0, wz]
        wx0, wx1, wz = c.tray_x - 0.75, c.tray_x + 0.75, 1.45
        x_lo, x_hi, z_hi = -4.0, 8.0, 5.0
        KA.add_texture(spec, F + "tex_tiles2", A.tile_wall_texture(c.seed + 1))
        panels = (((x_lo + wx0) / 2, (wx0 - x_lo) / 2, z_hi / 2, z_hi / 2),
                  ((wx1 + x_hi) / 2, (x_hi - wx1) / 2, z_hi / 2, z_hi / 2),
                  ((wx0 + wx1) / 2, (wx1 - wx0) / 2, (wz + z_hi) / 2, (z_hi - wz) / 2))
        tile4 = 0.8  # one texture repeat = 4 x 4 tiles of 0.2 mm
        for k, (x, hxp, z, hzp) in enumerate(panels):
            # textured panel meshes (a box face maps a 2D texture badly); the texture
            # is anchored at the top edge (z_hi) and the left edge so the tiles line up
            spec.add_material(name=f"{F}wall{k}", rgba=(1, 1, 1, 1), specular=0.45, shininess=0.6,
                              texrepeat=(2 * hxp / tile4, 2 * hzp / tile4))
            spec.materials[-1].textures[mj.mjtTextureRole.mjTEXROLE_RGB] = F + "tex_tiles2"
            KA.add_mesh(spec, f"{F}wall{k}_mesh", TA.front_panel_mesh(hxp, hzp, 0.05))
            wb.add_geom(name=f"{F}wall{k}", type=mj.mjtGeom.mjGEOM_MESH, meshname=f"{F}wall{k}_mesh",
                        pos=(x, yw + 0.03, z), quat=q_back, material=f"{F}wall{k}", **vis)
        # the side wall behind the fly (the kitchen corner; seen in the sneak close-up)
        xs, y0 = -2.2, -4.0
        hys, hzs = (yw - y0) / 2, z_hi / 2
        spec.add_material(name=F + "wall_side", rgba=(1, 1, 1, 1), specular=0.45, shininess=0.6,
                          texrepeat=(2 * hys / tile4, 2 * hzs / tile4))
        spec.materials[-1].textures[mj.mjtTextureRole.mjTEXROLE_RGB] = F + "tex_tiles2"
        KA.add_mesh(spec, F + "wall_side_mesh", TA.front_panel_mesh(hys, hzs, 0.05))
        wb.add_geom(name=F + "wall_side", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "wall_side_mesh",
                    pos=(xs, (yw + y0) / 2, hzs), material=F + "wall_side", **vis)
        # window frame, pass shelf edge
        for s in (-1, 1):
            add_box(wb, f"{F}win_side{s}", (0.04, 0.08, wz / 2), ((wx0 + wx1) / 2 + s * ((wx1 - wx0) / 2 + 0.04),
                    yw + 0.02, wz / 2), material=F + "steel", collide="visual")
        add_box(wb, F + "win_top", ((wx1 - wx0) / 2 + 0.08, 0.08, 0.04), ((wx0 + wx1) / 2, yw + 0.02, wz + 0.04),
                material=F + "steel", collide="visual")
        # ORDER UP light box, ticket rail with tickets
        KA.add_mesh(spec, F + "sign_mesh", TA.front_panel_mesh(0.62, 0.155, 0.04))
        wb.add_geom(name=F + "sign", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "sign_mesh",
                    pos=((wx0 + wx1) / 2, yw - 0.03, wz + 0.42), quat=q_back, material=F + "sign_off", **vis)
        add_box(wb, F + "ticket_rail", ((wx1 - wx0) / 2, 0.03, 0.02), ((wx0 + wx1) / 2, yw - 0.04, wz + 0.14),
                material=F + "steel", collide="visual")
        KA.add_mesh(spec, F + "ticket_mesh", TA.front_panel_mesh(0.07, 0.14, 0.004))
        for k in range(4):
            wb.add_geom(name=f"{F}ticket{k}", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "ticket_mesh",
                        pos=(wx0 + 0.3 + 0.3 * k, yw - 0.075, wz + 0.02), quat=q_back,
                        material=f"{F}ticket{k}", **vis)
        # ORDERS counter (3 seven-segment digits) left of the window
        ox, oz = wx0 - 0.75, 1.15
        add_box(wb, F + "orders_box", (0.42, 0.03, 0.22), (ox, yw - 0.02, oz), material=F + "black",
                collide="visual")
        KA.add_mesh(spec, F + "orders_lbl_mesh", TA.front_panel_mesh(0.3, 0.06, 0.004))
        wb.add_geom(name=F + "orders_label", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "orders_lbl_mesh",
                    pos=(ox, yw - 0.055, oz + 0.3), quat=q_back, material=F + "lbl_orders", **vis)
        self._seg_names = []
        dw, dh, st = 0.16, 0.14, 0.018
        for k in range(3):
            xd = ox + (k - 1) * 0.24
            segs = []
            for s, (dx, dz, horiz) in SEGS.items():
                nm = f"{F}seg{k}_{s}"
                size = (dw / 2 - 0.015, 0.008, st) if horiz else (st, 0.008, dh / 2 - 0.015)
                wb.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_BOX, size=size,
                            pos=(xd + dx * dw, yw - 0.055, oz + dz * dh), material=F + "seg_off", **vm)
                segs.append(nm)
            self._seg_names.append(segs)
        # the dining room seen through the window
        KA.add_mesh(spec, F + "dining_mesh", TA.front_panel_mesh(3.0, 1.6, 0.02))
        wb.add_geom(name=F + "dining", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "dining_mesh",
                    pos=(c.tray_x, yw + 2.4, 1.3), quat=q_back, material=F + "dining", **vis)
        # menu board over the fryer
        KA.add_mesh(spec, F + "menu_mesh", TA.front_panel_mesh(1.0, 0.62, 0.04))
        wb.add_geom(name=F + "menu", type=mj.mjtGeom.mjGEOM_MESH, meshname=F + "menu_mesh",
                    pos=(c.fryer_x + 0.4, yw - 0.04, 2.35), quat=q_back, material=F + "menu", **vis)
        # lights: key spot with shadows, a cool directional fill, dim headlight
        spec.visual.headlight.ambient = (0.3, 0.29, 0.28)
        spec.visual.headlight.diffuse = (0.28, 0.28, 0.28)
        spec.visual.headlight.specular = (0.1, 0.1, 0.1)
        tgt = np.array([2.2, 0.0, 0.3])
        key = np.array([-1.5, -6.0, 11.0])
        wb.add_light(name=F + "key", type=spot_or_directional(c.shadows), pos=tuple(key),
                     dir=tuple(tgt - key), diffuse=(0.62, 0.6, 0.56), specular=(0.4, 0.4, 0.4),
                     cutoff=40.0, exponent=0.5, castshadow=bool(c.shadows))
        wb.add_light(name=F + "fill", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(6.0, -6.0, 8.0),
                     dir=(-0.5, 0.6, -0.6), diffuse=(0.2, 0.22, 0.27), specular=(0.05, 0.05, 0.05),
                     castshadow=False)

    # ------------------------------------------------------------ attach / reset
    def on_attach(self) -> None:
        sim = self.sim
        m = sim.model
        c = self.cfg
        fly = sim.fly_name
        mid = lambda n: int(m.body_mocapid[m.body(n).id])  # noqa: E731
        gid = lambda n: int(m.geom(n).id)  # noqa: E731
        matid = lambda n: int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_MATERIAL, F + n))  # noqa: E731
        n = c.n_fries
        self.f_bid = np.array([m.body(nm).id for nm in self._fry_names])
        self.f_q = np.array([int(m.jnt_qposadr[m.joint(nm + "_free").id]) for nm in self._fry_names])
        self.f_v = np.array([int(m.jnt_dofadr[m.joint(nm + "_free").id]) for nm in self._fry_names])
        self.f_col = np.array([gid(nm + "_col") for nm in self._fry_names])
        self.f_vis = np.array([gid(nm + "_vis") for nm in self._fry_names])
        self.f_state = np.zeros(n, int)
        self.f_t = np.zeros(n)  # time of the last state change
        self.f_rel = np.zeros((n, 3))  # pose relative to the carrier (basket / scoop / carton / leg)
        self.f_relq = np.tile([1.0, 0, 0, 0], (n, 1))
        self.f_from = np.zeros((n, 3))  # pick blend start pose
        self.f_fromq = np.tile([1.0, 0, 0, 0], (n, 1))
        self.f_carton = np.full(n, -1)
        self.f_cook = np.zeros(n)  # 0 raw .. 1 golden
        self.f_salt = np.zeros(n)  # saltiness of its batch
        self.f_spilled = np.zeros(n, bool)
        self.f_sleep = np.zeros(n, bool)
        self.f_slow = np.full(n, math.inf)
        for i in range(n):  # a little rolling resistance (a spilled fry rolls, then stops)
            m.dof_damping[self.f_v[i] + 3:self.f_v[i] + 6] = 4e-9
        self.f_park = np.array([[-6.0 - 0.12 * j, 6.0, -3.0] for j in range(n)])
        # machines, props
        self.basket_mocap = mid(F + "basket")
        self.basket_cols = np.array([gid(nm) for nm in (F + "bk_col_floor",) +
                                     tuple(f"{F}bk_col_{a}{s}" for a in "xy" for s in (-1, 1))])
        self.trolley_mocap = mid(F + "trolley")
        self.hanger_mocap = mid(F + "hanger")
        self.hanger_gid = gid(F + "hanger_g")
        m.geom_rbound[self.hanger_gid] = 3.0
        self.scoop_mocap = mid(F + "scoop")
        self.shaker_mocap = mid(F + "shaker")
        self.shaker_salt = gid(F + "shaker_salt")
        self.carton_mocap = np.array([mid(f"{F}carton{k}") for k in range(N_CARTONS)])
        self.carton_cols = [np.array([gid(f"{F}carton{k}_col_floor")] +
                                     [gid(f"{F}carton{k}_col_{a}{s}") for a in "xy" for s in (-1, 1)])
                            for k in range(N_CARTONS)]
        self.tray_mocap = mid(F + "tray")
        self.plunger_mocap = mid(F + "plunger")
        self.ring_mocap = mid(F + "ring")
        self.ring_gid = gid(F + "ring_g")
        self.sign_gid = gid(F + "sign")
        self.oil_gid = gid(F + "oil")
        self.bulb_gid = np.array([gid(nm) for nm in self._bulb_names])
        self.mat = {k: matid(k) for k in ("sign_on", "sign_off", "seg_on", "seg_off", "oil_mat", "bulb")}
        self.lamp_light = int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_LIGHT, F + "lamp_light"))
        self.lamp0 = m.light_diffuse[self.lamp_light].copy()
        self.seg_gid = [[gid(nm) for nm in dig] for dig in self._seg_names]
        self.p_mocap = {k: np.array([mid(nm) for nm in v]) for k, v in self._pnames.items()}
        self.p_gid = {k: np.array([gid(nm + "_g") for nm in v]) for k, v in self._pnames.items()}
        for v in self.p_gid.values():
            m.geom_rbound[v] = 0.5
        self.parts = {"bubble": _Particles(36, (1.0, 0.92, 0.6), 0.5, 0.018, 0.55, 0.0),
                      "drip": _Particles(14, (0.95, 0.75, 0.25), 0.35, 0.012, 0.9, -9810.0),
                      "salt": _Particles(30, (0.98, 0.98, 0.98), 0.3, 0.008, 1.0, -9810.0)}
        self.steam = _Puffs(14, (0.95, 0.95, 0.97), 1.1, 0.03, 0.09, 0.12)
        # the fly: IK, proboscis, stationary action
        self.prob_ids = []
        for dof, sign in (("c_head-c_rostrum-pitch", -1.0), ("c_rostrum-c_haustellum-pitch", 1.0)):
            a = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, f"{fly}/{dof}-proboscispos")
            if a >= 0:
                self.prob_ids.append((a, sign))
        self.tarsus_bid = {leg: m.body(f"{fly}/{leg}_tarsus5").id for leg in ("lf", "rf")}
        self.haus_bid = m.body(f"{fly}/c_haustellum").id
        from fly_simulator.jobs.dead_hang import LegIK

        self.ik = LegIK(sim, self.session.actions.body)
        st = getattr(self.session, "STATIONARY_ACTIONS", None)
        if st is not None and CookStance.name not in st:
            self.session.STATIONARY_ACTIONS = (*st, CookStance.name)
        self.rng = np.random.default_rng(c.seed + 91)
        # counters (O(1) memory)
        self.n_orders = 0
        self.n_cartons = 0  # cartons filled
        self.n_in_cartons = 0  # fries that ended up in served cartons
        self.last_per_carton = 0
        self.n_batches = 0
        self.n_fried = 0  # fries dumped into the bin
        self.n_scoops = 0
        self.n_picked = 0
        self.n_spilled = 0
        self.n_sneaked = 0
        self.n_lost = 0
        self.n_recycled_spills = 0
        self.n_shakes = 0
        self.n_refills = 0
        self.n_voided = 0
        self.n_tastes = 0
        self.n_stim = 0
        self.salt_level = 1.0
        self.last_msg = ""
        self.taste_line = ""
        self.mn9_hz = 0.0
        self.mn9_peak = 0.0
        self.proboscis = 0.0
        self._pending_stim = None
        self._cam = None
        self._shot_name = "station"
        self._shot_t = -1e9
        self._scale = 1.0
        self._last_scale = 1.0
        self._caption = ("", -1e9)
        self.on_reset()
        self.reset_props()

    def on_reset(self) -> None:
        if getattr(self, "state", "") not in ("", "starting", "settle", "grab"):
            self.n_voided += 1
        self.state = "settle"
        self._t_state = self.sim.time
        self._fresh = True
        self._stance = None
        self._seg = {"lf": None, "rf": None}
        self._scoop_seg = None
        self._pending_stim = None
        self._per_until = -1.0
        self._sneak = None
        self._scoop_k = 0
        self._want = 7
        self._fx_t = -1.0

    def reset_props(self) -> None:
        """A clean station: pools parked, a cooked batch waiting in the raised basket,
        a carton on the stack, the tray home, the shaker parked."""
        c = self.cfg
        d = self.sim.data
        for i in range(c.n_fries):
            self._park(i)
        self.fill_k = -1  # carton at the fill spot
        self.stack_k = 0  # carton on top of the stack
        self.c_pos = np.array([[-5.0 - k, 5.0, -3.0] for k in range(N_CARTONS)])
        self.c_role = ["park"] * N_CARTONS
        self.c_role[0] = "stack"
        self.c_pos[0] = self.stack_top
        for k in range(N_CARTONS):
            self._set_carton_cols(k, False)
            self._place_carton(k)
        self.tray = {"state": "home", "t": 0.0, "k": -1, "y": c.tray_y}
        self._place_tray()
        self.sign_until = -1.0
        self._ding_t = -1e9
        self.fryer = {"state": "load_cooked", "t": self.sim.time}
        self.b_pos = self.basket_up.copy()
        self.b_tilt = 0.0
        self._place_basket()
        self._set_basket_cols(False)
        self.shaker = {"state": "park", "t": self.sim.time}
        self.sh_pos = self.shaker_park.copy()
        self.sh_q = np.array([1.0, 0, 0, 0])
        self._place_shaker()
        self.scoop_yaw, self.scoop_pitch = -0.5, 0.0
        self._scoop_q = _yaw_pitch_quat(self.scoop_yaw, self.scoop_pitch)
        for p in self.parts.values():
            p.clear()
        self.steam.t0[:] = -1e9
        self._batch_salt = 1.0
        self._update_counter()
        self.sim.model.geom_matid[self.sign_gid] = self.mat["sign_off"]

    # ------------------------------------------------------------ layout helpers
    @property
    def fill_spot(self) -> np.ndarray:
        return np.array([self.cfg.fill_x, self.cfg.fill_y, 0.0])

    @property
    def stack_top(self) -> np.ndarray:
        return np.array([self.cfg.stack_x, self.cfg.stack_y, 0.088])

    @property
    def basket_home(self) -> np.ndarray:  # cooking: in the oil
        c = self.cfg
        return np.array([c.fryer_x - 0.05, c.fryer_y, c.oil_z - 0.02])

    @property
    def basket_up(self) -> np.ndarray:
        return self.basket_home + np.array([0.0, 0.0, 0.62])

    @property
    def basket_dump(self) -> np.ndarray:
        c = self.cfg
        return np.array([c.bin_x + 0.42, c.bin_y + 0.02, 1.0])

    @property
    def shaker_park(self) -> np.ndarray:
        c = self.cfg
        return np.array([c.bin_x + 0.7, c.bin_y - 0.72, 0.0])

    def in_bin(self, p, margin: float = 0.0) -> bool:
        c = self.cfg
        return (abs(p[0] - c.bin_x) < c.bin_hx + margin and abs(p[1] - c.bin_y) < c.bin_hy + margin
                and p[2] < c.bin_floor + c.bin_wall + 0.35)

    def in_carton(self, p, k: int, margin: float = 0.0) -> bool:
        if k < 0:
            return False
        cp = self.c_pos[k]
        ihx, ihy = self.carton_inner
        return (abs(p[0] - cp[0]) < ihx + margin and abs(p[1] - cp[1]) < ihy + margin
                and cp[2] < p[2] < cp[2] + self.CARTON["h"] + 0.4)

    # ------------------------------------------------------------ fries (pool)
    def _park(self, i: int) -> None:
        m, d = self.sim.model, self.sim.data
        q, v = self.f_q[i], self.f_v[i]
        d.qpos[q:q + 3] = self.f_park[i]
        d.qpos[q + 3:q + 7] = (1, 0, 0, 0)
        d.qvel[v:v + 6] = 0
        self._contacts(i, False)
        self.f_sleep[i] = False
        self.f_state[i] = PARK
        self.f_carton[i] = -1
        self.f_spilled[i] = False
        self.f_salt[i] = 0.0
        self.f_cook[i] = 0.0
        m.geom_rgba[self.f_vis[i]] = (1, 1, 1, 1)

    def _contacts(self, i: int, on: bool) -> None:
        m = self.sim.model
        g = self.f_col[i]
        m.geom_contype[g] = FRY_BIT if on else 0
        m.geom_conaffinity[g] = (FRY_BIT | COL_BIT | TERRAIN_BIT) if on else 0
        m.body_gravcomp[self.f_bid[i]] = 0.0 if on else 1.0

    def _set_live(self, i: int, state: int, vel=None) -> None:
        d = self.sim.data
        self._contacts(i, True)
        self.f_sleep[i] = False
        self.f_slow[i] = math.inf
        v = self.f_v[i]
        d.qvel[v:v + 6] = 0
        if vel is not None:
            d.qvel[v:v + 3] = vel
        self.f_state[i] = state
        self.f_t[i] = self.sim.time

    def _set_kinematic(self, i: int, state: int, rel, relq) -> None:
        d = self.sim.data
        self._contacts(i, False)
        self.f_sleep[i] = False
        d.qvel[self.f_v[i]:self.f_v[i] + 6] = 0
        self.f_state[i] = state
        self.f_t[i] = self.sim.time
        self.f_rel[i] = rel
        self.f_relq[i] = relq

    def _fry_pose(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        q = self.f_q[i]
        d = self.sim.data
        return d.qpos[q:q + 3].copy(), d.qpos[q + 3:q + 7].copy()

    def _attach_to(self, i: int, state: int, frame_p, frame_q) -> None:
        """Make fry i ride kinematically on a carrier frame, keeping its world pose."""
        p, q = self._fry_pose(i)
        R = _quat_mat(frame_q)
        rel = R.T @ (p - frame_p)
        qi = np.array([frame_q[0], -frame_q[1], -frame_q[2], -frame_q[3]])
        self._set_kinematic(i, state, rel, _qmul(qi, q))

    def _ride(self, idx, frame_p, frame_q) -> None:
        d = self.sim.data
        R = _quat_mat(frame_q)
        for i in idx:
            q = self.f_q[i]
            d.qpos[q:q + 3] = frame_p + R @ self.f_rel[i]
            d.qpos[q + 3:q + 7] = _qmul(frame_q, self.f_relq[i])
            d.qvel[self.f_v[i]:self.f_v[i] + 6] = 0

    def _tint(self, i: int) -> None:
        """Raw (pale) -> golden -> a little brown at the edges."""
        u = float(np.clip(self.f_cook[i], 0.0, 1.0))
        rgb = (1.0 - 0.04 * u, 1.0 - 0.2 * u, 1.0 - 0.55 * u)
        self.sim.model.geom_rgba[self.f_vis[i], :3] = rgb

    def _free_fries(self, n: int) -> np.ndarray:
        """Up to n parked fries; recycles resting spills (oldest first) if short."""
        free = np.flatnonzero(self.f_state == PARK)
        if len(free) < n:
            sp = np.flatnonzero(self.f_state == SPILL_REST)
            sp = sp[np.argsort(self.f_t[sp])][:n - len(free)]
            for i in sp:
                self._park(i)
                self.n_recycled_spills += 1
            free = np.flatnonzero(self.f_state == PARK)
        return free[:n]

    def bin_count(self) -> int:
        return int(np.sum(self.f_state == BIN))

    def carton_count(self, k: int | None = None) -> int:
        k = self.fill_k if k is None else k
        return int(np.sum((self.f_carton == k) & np.isin(self.f_state, (CARTON, RIDE))))

    def _fries_step(self) -> None:
        """Physics fries: classify where they are (bin / carton / spilled / lost)."""
        d = self.sim.data
        t = self.sim.time
        live = np.flatnonzero(np.isin(self.f_state, (BASKET_LIVE, BIN, DROP, CARTON, SPILL)) & ~self.f_sleep)
        for i in live:
            q, v = self.f_q[i], self.f_v[i]
            p = d.qpos[q:q + 3]
            if not np.all(np.isfinite(d.qpos[q:q + 7])) or p[2] < -0.5 or abs(p[0]) > 12 or abs(p[1]) > 12:
                self._park(i)
                self.n_lost += 1
                continue
            s = self.f_state[i]
            speed = float(np.linalg.norm(d.qvel[v:v + 3]))
            age = t - self.f_t[i]
            if s == BASKET_LIVE:
                if not self._in_basket(p):
                    self.f_state[i] = DROP
                    self.f_t[i] = t
                continue
            if s in (DROP, SPILL) and (speed < 4.0 and age > 0.04 or age > 0.8):
                if self.in_bin(p):
                    self.f_state[i] = BIN
                    if not self.f_spilled[i] and self.f_carton[i] == -2:
                        self.n_fried += 1
                    self.f_carton[i] = -1
                elif self.in_carton(p, self.fill_k, 0.02):
                    self.f_state[i] = CARTON
                    self.f_carton[i] = self.fill_k
                else:  # on the counter (or a rim): a spill
                    if s == DROP:
                        self.f_state[i] = SPILL
                        self.f_t[i] = t
                        if not self.f_spilled[i]:
                            self.f_spilled[i] = True
                            self.n_spilled += 1
                            self.last_msg = "oops, a fry hit the counter"
                    elif speed < 1.5 or age > 3.0:
                        self._set_kinematic(i, SPILL_REST, np.zeros(3), np.array([1.0, 0, 0, 0]))
                        self.f_t[i] = t
            elif s == BIN and not self.in_bin(p, 0.05):
                self.f_state[i] = DROP  # bounced out
                self.f_t[i] = t
            elif s == CARTON and not self.in_carton(p, self.f_carton[i], 0.05):
                self.f_state[i] = DROP
                self.f_t[i] = t
            elif s in (BIN, CARTON):
                # at rest for 0.3 s: asleep (contacts off, held in place) until a dump /
                # a drop wakes the pile up again (saves the contact solver)
                if speed < 1.5 and float(np.linalg.norm(d.qvel[v + 3:v + 6])) < 20.0:
                    if not math.isfinite(self.f_slow[i]):
                        self.f_slow[i] = t
                    elif t - self.f_slow[i] > 0.3:
                        self._sleep(i)
                else:
                    self.f_slow[i] = math.inf

    def _sleep(self, i: int) -> None:
        self._contacts(i, False)
        self.sim.data.qvel[self.f_v[i]:self.f_v[i] + 6] = 0
        self.f_sleep[i] = True

    def _wake(self, idx) -> None:
        for i in idx:
            if self.f_sleep[i]:
                self._contacts(i, True)
                self.f_sleep[i] = False
                self.f_slow[i] = math.inf

    # ------------------------------------------------------------ cartons / tray
    def _place_carton(self, k: int) -> None:
        d = self.sim.data
        d.mocap_pos[self.carton_mocap[k]] = self.c_pos[k]
        d.mocap_quat[self.carton_mocap[k]] = (1, 0, 0, 0)

    def _set_carton_cols(self, k: int, on: bool) -> None:
        self.sim.model.geom_contype[self.carton_cols[k]] = COL_BIT if on else 0

    def _place_tray(self) -> None:
        d = self.sim.data
        d.mocap_pos[self.tray_mocap] = (self.cfg.tray_x, self.tray["y"], 0.0)

    def _carton_ride(self, k: int) -> None:
        idx = np.flatnonzero((self.f_carton == k) & (self.f_state == RIDE))
        if len(idx):
            self._ride(idx, self.c_pos[k], np.array([1.0, 0, 0, 0]))

    def _tray_step(self, t: float) -> None:
        """The pass: carton slides onto the tray (kinematic), ding + ORDER UP, the tray
        goes through the window with it, comes back empty."""
        c = self.cfg
        tr = self.tray
        s = tr["state"]
        dt = t - tr["t"]
        if s == "slide":  # carton from the fill spot onto the tray
            k = tr["k"]
            u = smoothstep(min(dt / 1.0, 1.0))
            a, b = tr["from"], np.array([c.tray_x, c.tray_y, 0.026])
            p = a + u * (b - a)
            p[2] += 0.08 * math.sin(math.pi * u)
            self.c_pos[k] = p
            self._place_carton(k)
            self._carton_ride(k)
            if dt >= 1.0:
                tr.update(state="ding", t=t)
                self._ding(t)
        elif s == "ding":
            if dt >= 0.7:
                tr.update(state="out", t=t)
        elif s == "out":  # through the window
            u = smoothstep(min(dt / 1.0, 1.0))
            tr["y"] = c.tray_y + 1.9 * u
            self._place_tray()
            k = tr["k"]
            self.c_pos[k] = np.array([c.tray_x, tr["y"], 0.026])
            self._place_carton(k)
            self._carton_ride(k)
            if dt >= 1.0:
                for i in np.flatnonzero(self.f_carton == k):
                    self._park(i)
                self.c_role[k] = "park"
                self.c_pos[k] = np.array([-5.0 - k, 5.0, -3.0])
                self._place_carton(k)
                tr.update(state="back", t=t, k=-1)
        elif s == "back":
            u = smoothstep(min(dt / 0.9, 1.0))
            tr["y"] = c.tray_y + 1.9 * (1 - u)
            self._place_tray()
            if dt >= 0.9:
                tr.update(state="home", t=t)
        # the next carton appears on the stack when the old one is taken
        if self.stack_k < 0:
            for k in range(N_CARTONS):
                if self.c_role[k] == "park":
                    self.c_role[k] = "stack"
                    self.stack_k = k
                    self.c_pos[k] = self.stack_top
                    self._place_carton(k)
                    break

    def _ding(self, t: float) -> None:
        self._ding_t = t
        self.sign_until = t + 2.2
        self.sim.model.geom_matid[self.sign_gid] = self.mat["sign_on"]
        self.n_orders += 1
        self.add_work(1)
        self._update_counter()
        n = self.last_per_carton
        self.last_msg = f"*DING* ORDER UP! #{self.n_orders} ({n} fries)"
        self._caption = (f"DING!  ORDER UP #{self.n_orders}", t)
        self.say(self.last_msg)

    def _update_counter(self) -> None:
        m = self.sim.model
        v = int(self.n_orders) % 1000
        digits = (v // 100, (v // 10) % 10, v % 10)
        for k, dg in enumerate(digits):
            show = k == 2 or v >= 10 ** (2 - k)
            lit = DIGITS[dg] if show else ""
            for s, g in zip(SEGS, self.seg_gid[k]):
                m.geom_matid[g] = self.mat["seg_on" if s in lit else "seg_off"]

    # ------------------------------------------------------------ fryer machine
    def _in_basket(self, p) -> bool:
        R = _quat_mat(self._basket_q())
        loc = R.T @ (p - self.b_pos)
        bhx, bhy, bhz = self.BASKET_H
        return abs(loc[0]) < bhx + 0.03 and abs(loc[1]) < bhy + 0.03 and -bhz - 0.05 < loc[2] < bhz + 0.08

    def _basket_q(self) -> np.ndarray:
        return np.array(quat_axis_angle((0, 1, 0), -self.b_tilt))

    def _place_basket(self) -> None:
        d, m = self.sim.data, self.sim.model
        q = self._basket_q()
        d.mocap_pos[self.basket_mocap] = self.b_pos
        d.mocap_quat[self.basket_mocap] = q
        bhx, _, bhz = self.BASKET_H
        yoke = self.b_pos + _quat_mat(q) @ np.array([bhx + 0.42, 0.0, bhz])
        top = self.rail_z - 0.08
        d.mocap_pos[self.trolley_mocap] = (yoke[0], yoke[1], top)
        L = max(top - yoke[2], 0.05)
        d.mocap_pos[self.hanger_mocap] = (yoke[0], yoke[1], yoke[2] + L / 2)
        m.geom_size[self.hanger_gid, 1] = L / 2
        idx = np.flatnonzero(self.f_state == BASKET)
        if len(idx):
            self._ride(idx, self.b_pos, q)

    def _set_basket_cols(self, on: bool) -> None:
        self.sim.model.geom_contype[self.basket_cols] = COL_BIT if on else 0

    def _load_basket(self, cooked: bool) -> int:
        """Raw fries into the basket (from the unseen freezer: a hidden recycle of the
        pool): two layers of rows, jittered, not overlapping."""
        c = self.cfg
        idx = self._free_fries(c.batch)
        bhx, bhy, bhz = self.BASKET_H
        q0 = self._basket_q()
        rows = 7
        for n, i in enumerate(idx):
            layer, row = divmod(n, rows)
            y = -bhy + 0.05 + row * (2 * bhy - 0.1) / (rows - 1)
            x = self.rng.uniform(-0.05, 0.05)
            z = -bhz + c.fry_r + 0.004 + layer * (2 * c.fry_r + 0.006)
            yaw = self.rng.normal(0, 0.1)
            rel = np.array([x, y, z])
            relq = _qmul(_yaw_quat(yaw), quat_axis_angle((0, 1, 0), math.pi / 2))
            self._set_kinematic(i, BASKET, rel, relq)
            self.f_cook[i] = 1.0 if cooked else 0.0
            self.f_carton[i] = -2  # "from this batch" (counted when it lands in the bin)
            self.f_spilled[i] = False
            self._tint(i)
        self._ride(idx, self.b_pos, q0)
        return len(idx)

    def _fryer_step(self, t: float) -> None:
        c = self.cfg
        fr = self.fryer
        s = fr["state"]
        dt = t - fr["t"]

        def go(state):
            fr.clear()
            fr.update(state=state, t=t)
        if s == "load_cooked":  # the first batch of a run: already cooked, raised
            self.b_pos = self.basket_up.copy()
            self._load_basket(cooked=True)
            self._place_basket()
            go("drip")
        elif s == "load":
            n = self._load_basket(cooked=False)
            self._place_basket()
            self.last_msg = f"a fresh basket of {n} fries goes in"
            go("lower")
        elif s == "lower":
            u = smoothstep(min(dt / 0.6, 1.0))
            self.b_pos = self.basket_up + u * (self.basket_home - self.basket_up)
            self._place_basket()
            if dt >= 0.6:
                for _ in range(10):  # splash
                    self._bubble(t, strong=True)
                go("cook")
        elif s == "cook":
            idx = np.flatnonzero(self.f_state == BASKET)
            for i in idx:
                self.f_cook[i] = min(dt / c.cook_s, 1.0)
            if int(dt * 10) != int((dt - 0.001) * 10):
                for i in idx:
                    self._tint(i)
            wob = 0.01 * math.sin(9 * t)
            self.b_pos = self.basket_home + np.array([0, 0, wob])
            self._place_basket()
            if dt >= c.cook_s:
                go("lift")
        elif s == "lift":
            u = smoothstep(min(dt / 0.7, 1.0))
            self.b_pos = self.basket_home + u * (self.basket_up - self.basket_home)
            self._place_basket()
            if dt >= 0.7:
                go("drip")
        elif s == "drip":
            if dt >= 1.0:
                go("hold")
        elif s == "hold":  # wait for the bin to run low
            if self.bin_count() < c.low_mark and self.shaker["state"] == "park" and \
                    not np.any(np.isin(self.f_state, (DROP,))):
                self.last_msg = "bin running low: dumping a fresh batch!"
                go("swing")
        elif s == "swing":
            u = smoothstep(min(dt / 1.0, 1.0))
            a, b = self.basket_up, self.basket_dump
            self.b_pos = a + u * (b - a)
            self.b_pos[2] += 0.12 * math.sin(math.pi * u)
            self._place_basket()
            if dt >= 1.0:
                self.n_batches += 1
                go("tip")
        elif s == "tip":
            if self._first_in("tip"):  # the fries become live bodies in the basket
                self._set_basket_cols(True)
                for i in np.flatnonzero(self.f_state == BASKET):
                    self._set_live(i, BASKET_LIVE)
                self._wake(np.flatnonzero(self.f_state == BIN))
            u = smoothstep(min(dt / 1.1, 1.0))
            self.b_tilt = math.radians(118.0) * u + math.radians(4.0) * math.sin(40 * dt) * (u > 0.8)
            self._place_basket()
            if dt >= 1.7:
                go("untip")
        elif s == "untip":
            u = smoothstep(min(dt / 0.5, 1.0))
            self.b_tilt = math.radians(118.0) * (1 - u)
            self._place_basket()
            if dt >= 0.5:
                # a fry stuck in the basket goes back to the freezer (hidden recycle)
                for i in np.flatnonzero(self.f_state == BASKET_LIVE):
                    self._park(i)
                self._set_basket_cols(False)
                self._start_shaker(t)
                go("return")
        elif s == "return":
            u = smoothstep(min(dt / 1.0, 1.0))
            a, b = self.basket_dump, self.basket_up
            self.b_pos = a + u * (b - a)
            self.b_pos[2] += 0.12 * math.sin(math.pi * u)
            self._place_basket()
            if dt >= 1.0:
                go("load")

    def _first_in(self, state: str) -> bool:
        k = "_first_" + state
        if self.fryer.get(k):
            return False
        self.fryer[k] = True
        return True

    # ------------------------------------------------------------ salt shaker
    def _start_shaker(self, t: float) -> None:
        self.shaker = {"state": "up", "t": t}

    def _place_shaker(self) -> None:
        d, m = self.sim.data, self.sim.model
        d.mocap_pos[self.shaker_mocap] = self.sh_pos
        d.mocap_quat[self.shaker_mocap] = self.sh_q
        lv = max(self.salt_level, 0.02)
        m.geom_size[self.shaker_salt, 1] = 0.12 * lv
        m.geom_pos[self.shaker_salt, 2] = 0.02 + 0.12 * lv

    def _shaker_step(self, t: float) -> None:
        c = self.cfg
        sh = self.shaker
        s = sh["state"]
        dt = t - sh["t"]
        over = np.array([c.bin_x + 0.05, c.bin_y, 1.05])
        inv = np.array(quat_axis_angle((1, 0, 0), math.pi * 0.92))

        def go(state):
            sh["state"], sh["t"] = state, t
        if s == "up":
            u = smoothstep(min(dt / 0.8, 1.0))
            self.sh_pos = self.shaker_park + u * (over - self.shaker_park)
            self.sh_pos[2] += 0.3 * math.sin(math.pi * u)
            self.sh_q = _slerp([1.0, 0, 0, 0], inv, u)
            self._place_shaker()
            if dt >= 0.8:
                go("shake")
        elif s == "shake":
            T = 1.1
            ph = math.sin(2 * math.pi * 6.0 * dt)
            self.sh_pos = over + np.array([0.05 * ph, 0.0, 0.07 * abs(ph)])
            self.sh_q = inv
            self._place_shaker()
            if self.salt_level > 0.01 and self.rng.random() < 0.45:
                cap = self.sh_pos + _quat_mat(self.sh_q) @ np.array([0, 0, self.shaker_h + 0.06])
                self.parts["salt"].spawn(t, cap + np.r_[self.rng.normal(0, 0.03, 2), 0],
                                         np.r_[self.rng.normal(0, 3.0, 2), -5.0], floor=c.bin_floor + 0.02)
            if dt >= T:
                dose = min(self.salt_level, c.salt_per_shake) / c.salt_per_shake
                self.salt_level = max(self.salt_level - c.salt_per_shake, 0.0)
                self.n_shakes += 1
                # this batch's saltiness (the next sneaked fry tastes it)
                self._batch_salt = float(max(0.0, dose * self.rng.normal(1.0, 0.3)))
                for i in np.flatnonzero(self.f_state == BIN):
                    self.f_salt[i] = self.f_salt[i] + self._batch_salt if self.f_salt[i] > 0 else self._batch_salt
                go("back")
        elif s == "back":
            u = smoothstep(min(dt / 0.8, 1.0))
            self.sh_pos = over + u * (self.shaker_park - over)
            self.sh_pos[2] += 0.3 * math.sin(math.pi * u)
            self.sh_q = _slerp(inv, [1.0, 0, 0, 0], u)
            self._place_shaker()
            if dt >= 0.8:
                if self.salt_level < 0.08:
                    self.salt_level = 1.0
                    self.n_refills += 1
                    self.last_msg = "salt shaker refilled"
                self._place_shaker()
                go("park")

    # ------------------------------------------------------------ the fly's legs
    def _ensure_stance(self) -> CookStance:
        acts = self.session.actions
        a = acts.action if isinstance(acts.action, CookStance) else None
        if a is None and not isinstance(acts._pending, CookStance):
            a = CookStance()
            acts.trigger(a, source="job")
        elif a is None:
            a = acts._pending
        self._stance = a
        return a

    def _stand_pose(self, leg: str) -> np.ndarray:
        b = self.session.actions.body
        return b.stand[np.flatnonzero(b.leg_mask([leg]))].copy()

    def _leg_now(self, leg: str) -> np.ndarray:
        st = self._stance
        if st is not None and st.pose.get(leg) is not None and st.w[leg] > 0:
            return st.pose[leg].copy()
        return self._stand_pose(leg)

    def _leg_to(self, leg: str, target, dur: float) -> None:
        """Move ``leg`` (joint space, eased) to the IK pose for world point ``target``
        (tarsus5); None = back to the stand pose."""
        if target is None:
            q1 = self._stand_pose(leg)
        else:
            b = self.session.actions.body
            cols = np.flatnonzero(b.leg_mask([leg]))
            q1, _ = self.ik.solve(self.sim.data.qpos, leg, [("tarsus5", np.asarray(target, float))],
                                  ref=b.stand[cols], ref_gain=0.1)
        self._seg[leg] = (self._leg_now(leg), q1, self.sim.time, max(dur, 1e-3), target is None)

    def _legs_step(self) -> float:
        st = self._stance
        u_min = 1.0
        for leg, seg in self._seg.items():
            if st is None or seg is None:
                continue
            q0, q1, t0, dur, home = seg
            u = min(max((self.sim.time - t0) / dur, 0.0), 1.0)
            st.pose[leg] = q0 + smoothstep(u) * (q1 - q0)
            st.w[leg] = 1.0
            if u >= 1.0 and home:
                st.w[leg] = 0.0
                st.pose[leg] = None
                self._seg[leg] = None
            u_min = min(u_min, u)
        return u_min

    def tarsus(self, leg: str) -> np.ndarray:
        return self.sim.data.xpos[self.tarsus_bid[leg]].copy()

    def _scoop_to(self, bowl, yaw: float, pitch: float, dur: float) -> None:
        """Right front leg + scoop: put the scoop's bowl centre at ``bowl`` with the
        given yaw / pitch (the tarsus target is the grip)."""
        q = _yaw_pitch_quat(yaw, pitch)
        grip = np.asarray(bowl, float) - _quat_mat(q) @ self.BOWL_OFF
        self._leg_to("rf", grip, dur)
        self._scoop_seg = (self.scoop_yaw, self.scoop_pitch, yaw, pitch, self.sim.time, dur)

    def _scoop_home(self, dur: float) -> None:
        self._leg_to("rf", None, dur)
        self._scoop_seg = (self.scoop_yaw, self.scoop_pitch, -0.5, 0.0, self.sim.time, dur)

    def scoop_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(grip = the right front tarsus, orientation) of the scoop now."""
        return self.tarsus("rf"), self._scoop_q

    def bowl_pos(self) -> np.ndarray:
        p, q = self.scoop_pose()
        return p + _quat_mat(q) @ self.BOWL_OFF

    def _scoop_step(self) -> None:
        sg = self._scoop_seg
        if sg is not None:
            y0, p0, y1, p1, t0, dur = sg
            u = smoothstep(min(max((self.sim.time - t0) / dur, 0.0), 1.0))
            self.scoop_yaw = y0 + u * (y1 - y0)
            self.scoop_pitch = p0 + u * (p1 - p0)
            if u >= 1.0:
                self._scoop_seg = None
        self._scoop_q = _yaw_pitch_quat(self.scoop_yaw, self.scoop_pitch)
        p, q = self.scoop_pose()
        d = self.sim.data
        d.mocap_pos[self.scoop_mocap] = p
        d.mocap_quat[self.scoop_mocap] = q
        # fries riding in the scoop (blending in from where they were picked)
        idx = np.flatnonzero(self.f_state == SCOOP)
        if len(idx):
            R = _quat_mat(q)
            for i in idx:
                pt = p + R @ self.f_rel[i]
                qt = _qmul(q, self.f_relq[i])
                u = smoothstep(min((self.sim.time - self.f_t[i]) / 0.12, 1.0))
                if u < 1.0:
                    pt = self.f_from[i] + u * (pt - self.f_from[i])
                    qt = _slerp(self.f_fromq[i], qt, u)
                qa = self.f_q[i]
                d.qpos[qa:qa + 3] = pt
                d.qpos[qa + 3:qa + 7] = qt
                d.qvel[self.f_v[i]:self.f_v[i] + 6] = 0
        idx = np.flatnonzero(self.f_state == SNEAK)
        if len(idx):
            self._ride(idx, self.tarsus("lf"), np.array([1.0, 0, 0, 0]))

    # ------------------------------------------------------------ main loop (1 ms)
    def _go(self, state: str) -> None:
        self.state = state
        self._t_state = self.sim.time
        self._fresh = True

    def _first(self) -> bool:
        f = self._fresh
        self._fresh = False
        return f

    def update(self) -> None:
        t = self.sim.time
        dt = t - self._t_state
        self.steering.set(None, 0.0)
        if self.state != "settle" or dt > 0.3:
            self._ensure_stance()
        s = self.state
        if s == "settle":
            if dt > 0.6:
                self._go("grab")
        elif s == "grab":
            self._grab(dt)
        elif s.startswith("scoop"):
            self._scoop(s, dt)
        elif s == "serve":
            self._serve(dt)
        elif s.startswith("sneak"):
            self._sneak_step(s, dt)
        self._legs_step()
        self._scoop_step()
        self._fryer_step(t)
        self._shaker_step(t)
        self._tray_step(t)
        self._fries_step()
        self._background(t)
        self._drive_proboscis()

    # ---- take a carton ------------------------------------------------------------
    def _grab(self, dt: float) -> None:
        k = self.fill_k if getattr(self, "_ph", 0) == 3 and self.state == "grab" and self.fill_k >= 0 \
            else self.stack_k
        if self._first():
            self._ph = 0
            if k < 0:
                self.last_msg = "waiting for a carton..."
        if k < 0:
            if dt > 0.2:
                self._go("grab")
            return
        C = self.CARTON
        rim = lambda p: p + np.array([-C["hx1"] - 0.01, 0.0, C["h"] - 0.02])  # noqa: E731
        if self._ph == 0:
            self._leg_to("lf", rim(self.c_pos[k]) + np.array([0, 0, 0.12]), 0.3)
            self._ph = 1
        elif self._ph == 1 and dt >= 0.3:
            self._leg_to("lf", rim(self.c_pos[k]), 0.12)
            self._ph = 2
        elif self._ph == 2 and dt >= 0.45:
            self.c_role[k] = "fill"
            self.fill_k = k
            self.stack_k = -1
            self._from = self.c_pos[k].copy()
            self._leg_to("lf", rim(self.fill_spot), 0.7)
            self._ph = 3
        elif self._ph == 3:
            u = smoothstep(min((dt - 0.45) / 0.7, 1.0))
            p = self._from + u * (self.fill_spot - self._from)
            p[2] += 0.05 * math.sin(math.pi * u)
            self.c_pos[k] = p
            self._place_carton(k)
            if u >= 1.0:
                self.c_pos[k] = self.fill_spot.copy()
                self._place_carton(k)
                self._set_carton_cols(k, True)
                self._leg_to("lf", None, 0.3)
                self._want = int(self.rng.integers(self.cfg.carton_min, self.cfg.carton_max + 1))
                self._scoop_k = 0
                self.last_msg = f"new carton: {self._want} fries, please"
                self._go("scoop_wait")

    # ---- scoop ------------------------------------------------------------------------
    def _resting_bin(self) -> np.ndarray:
        d = self.sim.data
        idx = np.flatnonzero(self.f_state == BIN)
        if not len(idx):
            return idx
        sp = np.linalg.norm(d.qvel[self.f_v[idx][:, None] + np.arange(3)], axis=1)
        return idx[sp < 5.0]

    def _dip_point(self) -> np.ndarray | None:
        c = self.cfg
        idx = self._resting_bin()
        if not len(idx):
            return None
        P = self.sim.data.qpos[self.f_q[idx][:, None] + np.arange(3)]
        D = np.linalg.norm(P[:, None, :2] - P[None, :, :2], axis=2)
        score = (D < 0.2).sum(1) - 0.3 * np.abs(P[:, 0] - (c.bin_x - 0.1))
        p = P[int(np.argmax(score))].copy()
        p[0] = np.clip(p[0], c.bin_x - c.bin_hx + 0.12, c.bin_x + c.bin_hx - 0.1)
        p[1] = np.clip(p[1], c.bin_y - c.bin_hy + 0.12, c.bin_y + c.bin_hy - 0.12)
        p[2] = c.bin_floor + 0.05
        return p

    def _scoop(self, s: str, dt: float) -> None:
        c = self.cfg
        yaw_bin, yaw_carton = -0.75, 0.25
        k = self.fill_k
        top = self.fill_spot + np.array([0.0, 0.0, C_TOP])
        if s == "scoop_wait":  # fries in the bin?
            if self._first():
                self._scoop_home(0.3)
            if self.carton_count() >= self._want or self._scoop_k >= c.max_scoops:
                self._go("serve")
                return
            if dt > 0.3 and self._maybe_sneak():
                return
            p = self._dip_point()
            if p is not None and dt > 0.3:
                self._dip = p
                self._go("scoop_reach")
            elif p is None and int(dt * 2) != int((dt - 0.001) * 2) and dt > 0.5:
                self.last_msg = "waiting for fries..."
        elif s == "scoop_reach":  # over the bin
            if self._first():
                self._scoop_to(self._dip + np.array([-0.05, 0.0, 0.3]), yaw_bin, 0.15, 0.35)
            if dt >= 0.38:
                self._go("scoop_dip")
        elif s == "scoop_dip":
            if self._first():
                self._scoop_to(self._dip + np.array([0.06, 0.0, 0.0]), yaw_bin, 0.35, 0.22)
            if dt >= 0.24:
                self._go("scoop_drag")
        elif s == "scoop_drag":  # drag toward the fly, bowl tipping up: pick up
            if self._first():
                self._scoop_to(self._dip + np.array([-0.1, 0.0, 0.02]), yaw_bin, -0.1, 0.22)
            if dt >= 0.22:
                self._pick()
                self._go("scoop_lift")
        elif s == "scoop_lift":
            if self._first():
                self._scoop_to(self._dip + np.array([-0.15, 0.1, 0.35]), yaw_bin, -0.1, 0.25)
                self._spill_at = (self.sim.time + self.rng.uniform(0.3, 0.5)
                                  if self.rng.random() < c.spill_p else None)
            if dt >= 0.26:
                self._go("scoop_carry")
        elif s == "scoop_carry":
            if self._first():
                self._scoop_to(top + np.array([-0.02, 0.0, 0.12]), yaw_carton, -0.1, 0.42)
            if self._spill_at is not None and self.sim.time >= self._spill_at:
                self._spill_at = None
                self._spill_one()
            if dt >= 0.44:
                self._go("scoop_tip")
        elif s == "scoop_tip":  # tip into the carton; the fries drop (physics)
            if self._first():
                self._scoop_to(top + np.array([0.02, 0.0, 0.0]), yaw_carton, 1.75, 0.35)
                self._rel_t = self.sim.time
                self._wake(np.flatnonzero((self.f_state == CARTON) & (self.f_carton == k)))
            if self.scoop_pitch > 1.3:
                idx = np.flatnonzero(self.f_state == SCOOP)
                if len(idx) and self.sim.time - self._rel_t > 0.03:
                    self._rel_t = self.sim.time
                    self._release(idx[0])
            if dt >= 0.55:
                for i in np.flatnonzero(self.f_state == SCOOP):
                    self._release(i)
                self._go("scoop_back")
        elif s == "scoop_back":
            if self._first():
                self._scoop_home(0.3)
                self._scoop_k += 1
            if dt >= 0.4 and not np.any(self.f_state == DROP):
                self._go("scoop_wait")

    def _pick(self) -> None:
        """The labelled kinematic transfer: resting bin fries under the bowl ride in
        the scoop from now on (blended into their slots over 0.12 s)."""
        c = self.cfg
        b = self.bowl_pos()
        idx = self._resting_bin()
        if not len(idx):
            return
        P = self.sim.data.qpos[self.f_q[idx][:, None] + np.arange(3)]
        dist = np.linalg.norm(P[:, :2] - b[:2], axis=1)
        order = np.argsort(dist)
        cap = int(self.rng.integers(max(c.scoop_cap - 1, 1), c.scoop_cap + 1))
        cap = min(cap, max(self._want - self.carton_count(), 1))
        take = [idx[j] for j in order if dist[j] < 0.22][:cap]
        if not take:
            take = [idx[j] for j in order if dist[j] < 0.45][:min(2, cap)]
        self.n_scoops += 1
        for n, i in enumerate(take):
            p, q = self._fry_pose(i)
            self.f_from[i], self.f_fromq[i] = p, q
            # side by side across the bowl (0.052 apart > a fry's 0.048 thickness), pointing
            # along the scoop and a little down: at release (pitch > 1.3) they hang ~vertical
            rel = self.BOWL_OFF + np.array([0.01 * (n % 2), 0.052 * (n - (len(take) - 1) / 2), 0.035])
            relq = quat_axis_angle((0, 1, 0), math.pi / 2 + 0.22 + 0.04 * (n % 2))
            self._set_kinematic(i, SCOOP, rel, relq)
            self.n_picked += 1
        self.last_msg = f"scoop #{self._scoop_k + 1}: {len(take)} fries"

    def _release(self, i: int) -> None:
        self._set_live(i, DROP, vel=self.rng.normal(0, 2.0, 3) * np.array([1, 1, 0]))

    def _spill_one(self) -> None:
        idx = np.flatnonzero(self.f_state == SCOOP)
        if not len(idx):
            return
        i = idx[-1]
        v = np.array([self.rng.uniform(-15, -5), self.rng.uniform(-12, 12), 0.0])
        self._set_live(i, DROP, vel=v)
        self.say("a fry slips off the scoop")

    def _serve(self, dt: float) -> None:
        k = self.fill_k
        if self._first():
            self._ph = 0
        if self._ph == 0:
            if self.tray["state"] != "home":
                if int(dt * 2) != int((dt - 0.001) * 2):
                    self.last_msg = "waiting for the tray..."
                return
            if np.any(self.f_state == DROP) and dt < 1.0:
                return
            n = self.carton_count(k)
            self.n_cartons += 1
            self.last_per_carton = n
            self.n_in_cartons += n
            for i in np.flatnonzero((self.f_carton == k) & (self.f_state == CARTON)):
                self._attach_to(i, RIDE, self.c_pos[k], np.array([1.0, 0, 0, 0]))
            self._set_carton_cols(k, False)
            self.c_role[k] = "serve"
            self.tray.update(state="slide", t=self.sim.time, k=k, **{"from": self.c_pos[k].copy()})
            self.fill_k = -1
            self.last_msg = f"carton #{self.n_cartons} full ({n} fries): to the window"
            self._ph = 1
        elif dt >= 0.6:
            self._go("grab")

    # ---- sneak a spilled fry (the taste) ---------------------------------------------
    def _maybe_sneak(self) -> bool:
        c = self.cfg
        idx = np.flatnonzero((self.f_state == SPILL_REST) & (self.f_t < self.sim.time - 0.3))
        if not len(idx):
            return False
        P = self.sim.data.qpos[self.f_q[idx][:, None] + np.arange(3)]
        home = np.array([1.45, 0.45])
        dist = np.linalg.norm(P[:, :2] - home, axis=1)
        j = int(np.argmin(dist))
        i = idx[j]
        if dist[j] > 0.9 or P[j, 0] < 1.0 or self.in_carton(P[j], self.fill_k, 0.1):
            return False
        self.f_t[i] = self.sim.time + 1e6  # one decision per spill
        if self.rng.random() > c.sneak_p:
            return False
        self._sneak = {"i": i, "p": P[j].copy()}
        self._go("sneak_reach")
        return True

    @property
    def mouth_pt(self) -> np.ndarray:
        h = self.sim.data.xpos[self.haus_bid]
        return h + np.array([0.3, 0.1, -0.02])

    def _sneak_step(self, s: str, dt: float) -> None:
        sn = self._sneak
        i = sn["i"]
        if s == "sneak_reach":
            if self._first():
                self._leg_to("lf", sn["p"] + np.array([-0.05, 0.05, 0.25]), 0.3)
            if dt >= 0.32:
                self._go("sneak_grab")
        elif s == "sneak_grab":
            if self._first():
                self._leg_to("lf", sn["p"] + np.array([0.0, 0.0, 0.03]), 0.15)
            if dt >= 0.17:
                p, q = self._fry_pose(i)
                self._attach_to(i, SNEAK, self.tarsus("lf"), np.array([1.0, 0, 0, 0]))
                self.f_rel[i] = np.array([0.0, 0.0, -0.02])
                self._go("sneak_lift")
        elif s == "sneak_lift":  # to the mouth
            if self._first():
                self._leg_to("lf", self.mouth_pt, 0.45)
            if dt >= 0.47:
                self._taste(i)
                self._go("sneak_eat")
        elif s == "sneak_eat":
            u = min(dt / 0.8, 1.0)
            self.sim.model.geom_rgba[self.f_vis[i], 3] = 1.0 - 0.9 * u  # nibbled away
            if dt >= 0.8:
                self._park(i)
                self.n_sneaked += 1
                self._leg_to("lf", None, 0.35)
                self._go("sneak_done")
        elif s == "sneak_done":
            if self._first() and self._brain() is not None:
                v = "mmm, salty" if self.mn9_peak >= 30 else "meh"
                self.taste_line = (f"sneaked fry #{self.n_tastes} (salt x{self._taste_salt:.1f}): MN9 peak "
                                   f"{self.mn9_peak:.0f} Hz -> '{v}' (real connectome; stand-in taste sets)")
                self.say(self.taste_line)
            if dt >= 0.4:
                self._go("scoop_wait")

    def _taste(self, i: int) -> None:
        c = self.cfg
        self.n_tastes += 1
        self.mn9_peak = 0.0
        salt = float(self.f_salt[i])
        self._taste_salt = salt
        self._pending_stim = (self.run_time(), salt)
        if self._brain() is None:
            self._per_until = self.sim.time + 0.9  # scripted dab (no brain), labelled
            self.taste_line = (f"sneaked fry #{self.n_tastes} (salt x{salt:.1f}): 'mmm' "
                               "[scripted: no brain]")
        else:
            self._taste_t0 = self.sim.time
        self.last_msg = "the fly sneaks a fry..."
        self.say(f"sneaks a fry (salt x{salt:.1f})")

    # ------------------------------------------------------------ background fx
    def _bubble(self, t: float, strong: bool = False) -> None:
        c = self.cfg
        hx = 0.5
        p = np.array([c.fryer_x + self.rng.uniform(-hx, hx), c.fryer_y + self.rng.uniform(-hx, hx),
                      c.oil_z - self.rng.uniform(0.05, 0.25)])
        v = np.array([0.0, 0.0, self.rng.uniform(0.4, 0.8)])
        self.parts["bubble"].spawn(t, p, v, grow=1.5 if strong else 0.8)

    def _background(self, t: float) -> None:
        m, d = self.sim.model, self.sim.data
        c = self.cfg
        # bell: plunger bob + a ring flash after a ding
        u = t - self._ding_t
        bx, by = self.bell_xy
        d.mocap_pos[self.plunger_mocap] = (bx, by, 0.1 - (0.025 if u < 0.12 else 0.0))
        if 0 <= u < 0.5:
            d.mocap_pos[self.ring_mocap] = (bx, by, 0.05)
            m.geom_size[self.ring_gid, 0] = 0.1 + 0.4 * u
            m.geom_rgba[self.ring_gid, 3] = 0.6 * (1 - u / 0.5)
        elif m.geom_rgba[self.ring_gid, 3] != 0:
            m.geom_rgba[self.ring_gid, 3] = 0
            d.mocap_pos[self.ring_mocap] = (bx, by, -5)
        if self.sign_until > 0 and t > self.sign_until:
            self.sign_until = -1.0
            m.geom_matid[self.sign_gid] = self.mat["sign_off"]
        elif self.sign_until > 0:
            blink = int((self.sign_until - t) * 4) % 2 == 0
            m.geom_matid[self.sign_gid] = self.mat["sign_on" if blink else "sign_off"]
        if t - self._fx_t < 0.02:
            return
        self._fx_t = t
        cooking = self.fryer["state"] in ("cook", "lower")
        rate = 0.9 if cooking else 0.2
        if self.rng.random() < rate:
            self._bubble(t)
        if self.fryer["state"] in ("drip", "hold") and self.rng.random() < (
                0.5 if self.fryer["state"] == "drip" else 0.08):
            bhx, bhy, bhz = self.BASKET_H
            p = self.b_pos + np.array([self.rng.uniform(-bhx, bhx), self.rng.uniform(-bhy, bhy), -bhz - 0.01])
            self.parts["drip"].spawn(t, p, np.zeros(3), floor=c.oil_z)
        if cooking and self.rng.random() < 0.3:
            p = np.array([c.fryer_x + self.rng.normal(0, 0.25), c.fryer_y + self.rng.normal(0, 0.25),
                          c.oil_z + 0.1])
            self.steam.spawn(t, p, (self.rng.normal(0, 0.1), self.rng.normal(0, 0.1), 0.8))
        # oil shimmer, heat lamp glow
        m.mat_emission[self.mat["oil_mat"]] = 0.22 + 0.08 * math.sin(7 * t) * math.sin(3.1 * t + 1) + \
            0.08 * cooking
        fl = 0.92 + 0.08 * math.sin(11 * t) * math.sin(4.3 * t)
        m.light_diffuse[self.lamp_light] = self.lamp0 * fl
        for k, pr in self.parts.items():
            pr.draw(t, m, d, self.p_mocap[k], self.p_gid[k])
        self.steam.draw(t, m, d, self.p_mocap["steam"], self.p_gid["steam"])

    # ------------------------------------------------------------ brain / proboscis
    def _brain(self):
        return getattr(self.session, "brain", None)

    def _drive_proboscis(self) -> None:
        c = self.cfg
        target = 0.0
        if self._brain() is not None:
            target = min(max(self.mn9_hz / max(c.mn9_ref_hz, 1e-6), 0.0), 1.0)
        if self.sim.time < self._per_until:
            target = max(target, 0.85)
        a = min(0.001 * c.update_every_steps / 0.1, 1.0)
        self.proboscis += a * (target - self.proboscis)
        d = self.sim.data
        for aid, sign in self.prob_ids:
            d.ctrl[aid] = sign * self.proboscis

    def after_physics(self) -> None:
        super().after_physics()
        link = self._brain()
        if link is not None:
            lt = getattr(link, "latest", None)
            if lt is not None:
                self.mn9_hz = float((getattr(lt, "probes", None) or {}).get("MN9", 0.0))
            if self.state.startswith("sneak"):
                self.mn9_peak = max(self.mn9_peak, self.mn9_hz)
            if self._pending_stim is not None:
                self._send_taste(link, *self._pending_stim)
        else:
            self.mn9_hz = 0.0
        self._pending_stim = None

    def taste_details(self, salt: float) -> dict:
        """The stimulus of a sneaked fry. **Stand-in, labelled:** the model has no
        tarsal / salt-specific input; low salt (and starch) is appetitive and in flies
        activates sweet GRNs, high salt recruits bitter GRNs (Jaeger et al. 2018), so
        the labellar LB3 sugar set carries the 'tasty' part and, above ``oversalt``,
        the LB1 bitter set the 'too salty' part (docs/TASTE.md stand-in sets)."""
        c = self.cfg
        det = {"tastes": ["sugar"], "sugar_hz": c.sugar_hz,
               "label": "FRY TASTE (stand-in: LB3 sugar GRNs = low salt / starch; "
                        "LB1 bitter GRNs when over-salted)"}
        if salt > c.oversalt:
            det["tastes"].append("bitter")
            det["bitter_hz"] = float(min(c.bitter_hz * (salt - 1.0) / 0.6, 200.0))
        return det

    def _send_taste(self, link, t0: float, salt: float) -> None:
        from fly_simulator.brain.schema import StimulusEvent

        link.send(StimulusEvent("taste", "left", 1.0, self.cfg.taste_s, t0, details=self.taste_details(salt)),
                  source="job")
        self.n_stim += 1
        log = getattr(link, "stim_log", None)
        if isinstance(log, list) and len(log) > 400:
            del log[:-200]

    # ------------------------------------------------------------ edit effects (labelled)
    def _slowmo_target(self) -> float:
        c = self.cfg
        if not c.slowmo:
            return 1.0
        if self.fryer["state"] in ("tip", "untip") and np.any(np.isin(self.f_state, (DROP, BASKET_LIVE))):
            return c.dump_slowmo
        if self.state == "scoop_tip" and self.scoop_pitch > 0.7:
            return c.drop_slowmo
        return 1.0

    def time_scale(self, present_dt: float) -> float:
        """Slow motion while the basket dumps and the fries drop into the carton (an
        *edit effect*: fewer physics steps per displayed frame; physics unchanged)."""
        tgt = self._slowmo_target()
        if tgt < self._scale:
            self._scale = tgt
        else:
            a = 1.0 - math.exp(-present_dt / 0.12)
            self._scale += a * (tgt - self._scale)
            if self._scale > 0.98:
                self._scale = 1.0
        self._last_scale = self._scale
        return self._scale

    def post_process(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Labels: the slow motion (edit, not physics) and the ORDER UP caption."""
        cap, t_cap = self._caption
        show_cap = self.cfg.captions and cap and 0 <= t - t_cap < 1.6
        if self._last_scale >= 1.0 and not show_cap:
            return frame
        import cv2

        H, W = frame.shape[:2]
        out = np.ascontiguousarray(frame).copy()
        fs = max(0.4, W / 1400.0)
        th = max(1, int(round(W / 700)))
        if self._last_scale < 1.0:
            txt = f"SLOW MOTION x{self._last_scale:.2f}  (edit, not physics)"
            (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            x, y = W - tw - int(0.015 * W), H - int(0.03 * H)
            cv2.rectangle(out, (x - 6, y - tht - 6), (x + tw + 6, y + 6), (0, 0, 0), -1)
            cv2.putText(out, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 220, 90), th, cv2.LINE_AA)
        if show_cap:
            f2 = fs * 2.2
            t2 = th * 2 + 1
            (tw, tht), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_DUPLEX, f2, t2)
            x, y = (W - tw) // 2, int(0.14 * H) + tht
            cv2.putText(out, cap, (x + 2, y + 2), cv2.FONT_HERSHEY_DUPLEX, f2, (40, 0, 0), t2 + 2, cv2.LINE_AA)
            cv2.putText(out, cap, (x, y), cv2.FONT_HERSHEY_DUPLEX, f2, (255, 214, 40), t2, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------ camera / HUD / stats
    SHOTS = {
        "station": (np.array([1.75, 0.15, 0.6]), (108.0, -22.0, 7.6)),
        "scoop": (np.array([1.85, -0.2, 0.35]), (118.0, -30.0, 4.0)),
        "dump": (np.array([2.45, -0.6, 0.65]), (102.0, -16.0, 4.6)),
        "order": (np.array([1.8, 1.4, 0.55]), (108.0, -14.0, 4.6)),
        "sneak": (np.array([1.05, 0.25, 0.65]), (220.0, -14.0, 3.0)),
    }

    def _pick_shot(self) -> str:
        c = self.cfg
        if not c.close_ups:
            return "station"
        fr = self.fryer["state"]
        if fr in ("swing", "tip", "untip"):
            return "dump"
        if self.state.startswith("sneak"):
            return "sneak"
        if self.tray["state"] in ("slide", "ding", "out"):
            return "order"
        if self.state.startswith("scoop_") and self.state != "scoop_wait":
            return "scoop"
        return "station"

    def _shot(self):
        t = self.run_time()
        want = self._pick_shot()
        if want != self._shot_name and (t - self._shot_t > 1.6 or want in ("dump", "sneak")):
            if "sneak" in (want, self._shot_name):
                self._cam = None  # a hard cut to / from the sneak close-up (the other side)
            self._shot_name, self._shot_t = want, t
        tgt, (az, el, dist) = self.SHOTS[self._shot_name]
        return tgt, CameraPreset(azimuth=az, elevation=el, distance=dist, tau_s=0.6)

    def camera_target(self) -> np.ndarray:
        return self._shot()[0]

    def camera_preset(self) -> CameraPreset:
        _, pre = self._shot()
        t = self.run_time()
        want = np.array([pre.azimuth, pre.elevation, pre.distance])
        if self._cam is None or t < self._cam[0]:
            self._cam = (t, want)
        else:
            t0, cur = self._cam
            a = 1.0 - math.exp(-(t - t0) / 0.6)
            self._cam = (t, cur + a * (want - cur))
        az, el, dist = self._cam[1]
        return CameraPreset(azimuth=float(az), elevation=float(el), distance=float(dist), tau_s=0.6)

    def job_hud_lines(self) -> list[str]:
        brain = self._brain() is not None
        mean = self.n_in_cartons / max(self.n_cartons, 1)
        lines = [
            f"cartons filled {self.n_cartons}   fries/carton {mean:.1f} (last {self.last_per_carton})   "
            f"spilled {self.n_spilled}   sneaked {self.n_sneaked}",
            f"batches fried {self.n_batches} ({self.n_fried} fries)   in the bin {self.bin_count()}   "
            f"scoops {self.n_scoops}   salt level {self.salt_level:.0%} (refills {self.n_refills})",
            f"fryer: {self.fryer['state']}   station: {self.state}",
        ]
        if self.last_msg:
            lines.append(self.last_msg)
        if self.taste_line:
            lines.append(self.taste_line)
        lines.append((f"MN9 {self.mn9_hz:.0f} Hz (live)   proboscis {self.proboscis:.0%}" if brain
                      else f"proboscis {self.proboscis:.0%}   (sneaked-fry taste: --brain for the connectome)"))
        lines.append("(fries: real physics in the basket / bin / carton; scoop pick-up, basket lift, "
                     "shaker, carton + tray slides: kinematic; slow motion: edit)")
        return lines

    def job_stats(self) -> dict:
        return {"orders": self.n_orders, "cartons": self.n_cartons,
                "fries_per_carton": round(self.n_in_cartons / max(self.n_cartons, 1), 2),
                "fries_in_cartons": self.n_in_cartons, "batches": self.n_batches, "fried": self.n_fried,
                "scoops": self.n_scoops, "picked": self.n_picked, "spilled": self.n_spilled,
                "sneaked": self.n_sneaked, "lost": self.n_lost, "recycled_spills": self.n_recycled_spills,
                "shakes": self.n_shakes, "salt_level": round(self.salt_level, 3), "refills": self.n_refills,
                "tastes": self.n_tastes, "stimuli": self.n_stim, "voided": self.n_voided}


C_TOP = FryCookJob.CARTON["h"] + 0.28  # scoop bowl height over the carton floor when tipping
