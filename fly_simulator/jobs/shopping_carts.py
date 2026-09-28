"""Shopping carts: the fly returns shopping carts forever (docs/JOBS.md, "shopping_carts").

World (the fly spawns at the origin facing +x, in the middle of the lot):

* a fly-scale supermarket parking lot: asphalt with painted bays, a raised sidewalk in
  front of the store ("FRESH FLY MARKET", our own name), a concrete **loading ramp**
  from the lot up to the sidewalk (a real static slope, 4 deg, walkable), a cart
  corral on the left, light poles (point lights), parked cars (visual) and trees;
* a pool of ``n_carts`` **carts**. Each is on planar joints like the mowing job's
  mower (slide x, slide y, yaw hinge) plus a vertical slide, so it can't tip, but it
  rests on a hidden frictionless skid and gravity really acts on it: on the ramp it
  rolls down by itself. The casters' rolling resistance is the slides' dry friction
  (``frictionloss``) plus a little viscous drag. The fly pushes a hidden round
  collider with its head / thorax / abdomen at low friction (``slippery_body_contact``,
  legs excluded); the basket, the handle and the wheels are visual. The job turns the
  yaw hinge toward the push direction (rate limited, like the mower) and, for a rolling
  cart, toward its motion.

Behaviour: the fly picks the loose cart that is cheapest to return, ``PushPilot``
(the mowing push loop) pushes it to a staging point in front of the corral mouth and
then straight in. Inside the mouth the cart **nests** into the train: a kinematic
glide into the next slot (engineered, labelled); nested carts are held (contacts off).
A **customer** (kinematic, labelled: nobody is drawn, the cart just rolls out of the
corral by itself) takes the last cart of the train now and then and leaves it at a
random spot in the lot; sometimes on the ramp, and then it rolls away down the slope
(real physics) and the fly chases it. So it never ends.

Counters: carts returned (the work counter), longest cart train, runaways caught (and
how many were caught while still rolling), customers, carts lost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from fly_simulator.jobs import shopping_carts_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import (PROP_BIT, add_box, add_plane_box, contact_kwargs, quat_axis_angle,
                                        slippery_body_contact, spot_or_directional, wrap_angle)
from fly_simulator.jobs.mowing import PushPilot
from fly_simulator.jobs.registry import register_job
from fly_simulator.terrain import TERRAIN_BIT

P = "cart/"
LOOSE, NESTING, NESTED, CUSTOMER = 0, 1, 2, 3
STATE_NAMES = ("loose", "nesting", "nested", "customer")


@dataclass
class ShoppingCartsConfig(JobConfig):
    # ---- lot (mm) -------------------------------------------------------------------
    lot_x0: float = -16.0
    lot_x1: float = 24.0
    lot_y0: float = -12.0
    lot_y1: float = 12.5  # the sidewalk edge
    sidewalk_y1: float = 17.0  # the store front
    ramp_x0: float = 7.0
    ramp_x1: float = 17.0
    ramp_y0: float = 2.5  # the ramp's foot (lot level); it rises to lot_y1
    ramp_deg: float = 4.0
    # ---- corral ------------------------------------------------------------------------
    corral_x0: float = -13.5  # closed end
    corral_x1: float = -3.5  # the mouth (open toward +x)
    corral_y: float = 7.0
    corral_hw: float = 1.05
    nest_pitch: float = 1.0  # nested carts' spacing along the corral
    first_slot: float = 1.35  # slot 0 centre from the closed end
    staging: float = 3.8  # the staging point this far in front of the mouth
    nest_speed: float = 14.0  # mm/s, the kinematic nesting glide
    # ---- carts -------------------------------------------------------------------------
    n_carts: int = 9
    n_nested0: int = 4  # carts in the corral at the start
    cart_radius: float = 1.2  # the hidden push collider (round, like the mower deck)
    cart_z: float = 0.85
    cart_half_h: float = 0.38
    cart_mass: float = 5e-4  # g (0.5 mg)
    rolling_friction: float = 0.08  # uN, slide frictionloss (casters' rolling resistance)
    drag: float = 0.03  # uN per mm/s, slide damping (bearings)
    head_friction: float = 0.05
    yaw_rate: float = 2.5
    # ---- customer (kinematic, labelled) ----------------------------------------------
    customer_mean_s: float = 16.0
    customer_idle_s: float = 2.0  # no loose carts: a customer comes this soon
    customer_speed: float = 12.0  # mm/s, the cart's glide out of the corral
    runaway_p: float = 0.3  # share of the carts the customer leaves on the ramp
    # ---- behaviour ---------------------------------------------------------------------
    behind_gap: float = 1.2
    orbit_clearance: float = 1.4
    push_speed: float = 0.75
    approach_speed: float = 0.9
    pursuit: float = 2.5
    catch_gap: float = 1.6  # a runaway is caught when the thorax is this close to its collider
    retarget_s: float = 60.0  # a cart not returned in this long: pick another
    captions: bool = True
    shadows: bool = True


@register_job
class ShoppingCartsJob(EternalJob):
    name = "shopping_carts"
    title = "SHOPPING CARTS FLY"
    tagline = "the fly returns shopping carts forever"
    work_label = "carts returned"
    znear = 0.05
    config_cls = ShoppingCartsConfig
    required_names = (P + "cart0", P + "cart0_push", P + "ramp")

    cfg: ShoppingCartsConfig

    # ------------------------------------------------------------ geometry
    @property
    def ramp_tan(self) -> float:
        return math.tan(math.radians(self.cfg.ramp_deg))

    @property
    def sidewalk_top(self) -> float:
        c = self.cfg
        return (c.lot_y1 - c.ramp_y0) * self.ramp_tan

    def ground_height(self, x: float, y: float) -> float:
        c = self.cfg
        if y >= c.lot_y1 and c.lot_x0 - 4 <= x <= c.lot_x1 + 4 and y <= c.sidewalk_y1:
            return self.sidewalk_top
        if c.ramp_x0 <= x <= c.ramp_x1 and c.ramp_y0 <= y < c.lot_y1:
            return (y - c.ramp_y0) * self.ramp_tan
        return 0.0

    def on_ramp(self, x: float, y: float) -> bool:
        c = self.cfg
        return c.ramp_x0 <= x <= c.ramp_x1 and c.ramp_y0 + 0.3 <= y < c.lot_y1

    def slot_xy(self, k: int) -> np.ndarray:
        c = self.cfg
        return np.array([c.corral_x0 + c.first_slot + k * c.nest_pitch, c.corral_y])

    @property
    def mouth(self) -> np.ndarray:
        return np.array([self.cfg.corral_x1, self.cfg.corral_y])

    @property
    def staging_xy(self) -> np.ndarray:
        return self.mouth + np.array([self.cfg.staging, 0.0])

    def park_xy(self, j: int) -> np.ndarray:
        return np.array([-30.0 - 3.0 * j, -30.0])

    # ------------------------------------------------------------ build
    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        self._materials(spec)
        vis = dict(contact_kwargs("visual"), mass=0.0)
        # the loading ramp and the sidewalk (static, walkable; the carts' skids touch them)
        a = math.radians(c.ramp_deg)
        L = c.lot_y1 - c.ramp_y0
        mid = ((c.ramp_x0 + c.ramp_x1) / 2, (c.ramp_y0 + c.lot_y1) / 2, L / 2 * math.tan(a))
        add_plane_box(wb, P + "ramp", mid, (1, 0, 0), (0, -math.sin(a), math.cos(a)),
                      (c.ramp_x1 - c.ramp_x0) / 2, L / 2 / math.cos(a), thickness=1.0, collide="static",
                      friction=1.0, material=P + "ramp")
        st = self.sidewalk_top
        xm = (c.lot_x0 + c.lot_x1) / 2
        add_box(wb, P + "sidewalk", ((c.lot_x1 - c.lot_x0) / 2 + 4, (c.sidewalk_y1 - c.lot_y1) / 2, st / 2),
                (xm, (c.lot_y1 + c.sidewalk_y1) / 2, st / 2), collide="static", material=P + "concrete")
        # carts
        cm = A.cart_meshes()
        for nm, md in cm.items():
            A.add_mesh(spec, P + "cart_" + nm, md)
        for j in range(c.n_carts):
            self._add_cart(spec, j, vis)
        self._add_scene(spec, vis)

    def _add_cart(self, spec, j: int, vis) -> None:
        c = self.cfg
        px, py = self.park_xy(j)
        body = spec.worldbody.add_body(name=f"{P}cart{j}", pos=(px, py, 0.0))
        body.gravcomp = 1.0  # (compiled in so it can be switched at run time)
        for nm, ax in (("x", (1, 0, 0)), ("y", (0, 1, 0))):
            body.add_joint(name=f"{P}cart{j}_s{nm}", type=mj.mjtJoint.mjJNT_SLIDE, axis=ax,
                           damping=c.drag, frictionloss=c.rolling_friction)
        body.add_joint(name=f"{P}cart{j}_sz", type=mj.mjtJoint.mjJNT_SLIDE, axis=(0, 0, 1), damping=0.05)
        body.add_joint(name=f"{P}cart{j}_yaw", type=mj.mjtJoint.mjJNT_HINGE, axis=(0, 0, 1), damping=1.0,
                       armature=1e-3)
        # the hidden push collider: contacts only through the slippery pairs with the fly
        body.add_geom(name=f"{P}cart{j}_push", type=mj.mjtGeom.mjGEOM_CYLINDER,
                      size=(c.cart_radius, c.cart_half_h, 0), pos=(0, 0, c.cart_z), mass=c.cart_mass * 0.9,
                      contype=0, conaffinity=0, rgba=(0.2, 0.2, 0.25, 0.0), group=3)
        # the skid it rests on (frictionless; the rolling resistance is the slides' friction)
        body.add_geom(name=f"{P}cart{j}_skid", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.2, 0, 0),
                      pos=(0, 0, 0.2), mass=c.cart_mass * 0.1, contype=PROP_BIT, conaffinity=TERRAIN_BIT,
                      priority=2, condim=3, friction=(0.0, 0.0, 0.0), solref=(0.004, 1.0),
                      rgba=(0, 0, 0, 0), group=3)
        for nm, mat in (("wire", "chrome"), ("plastic", "red_plastic"), ("wheels", "rubber")):
            body.add_geom(name=f"{P}cart{j}_{nm}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "cart_" + nm,
                          material=P + mat, **vis)
        slippery_body_contact(spec, f"{P}cart{j}", f"{P}cart{j}_push", self.fly_name, friction=c.head_friction)

    def _materials(self, spec) -> None:
        c = self.cfg
        T = A.add_textured_material
        mat = spec.material("grid")
        if mat is not None:  # the lot: asphalt, one texture tile per 12 mm
            A.add_texture(spec, P + "tex_asphalt", A.asphalt_texture(c.seed))
            mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = P + "tex_asphalt"
            mat.texrepeat = [2000.0 / 12.0, 2000.0 / 12.0]
            mat.rgba = (1.0, 1.0, 1.0, 1.0)
            mat.reflectance = 0.0
            mat.specular = 0.15
        A.add_texture(spec, P + "tex_concrete", A.concrete_texture(c.seed))
        T(spec, P + "concrete", P + "tex_concrete", rgba=(1, 1, 1, 1), specular=0.1, texrepeat=(4, 1),
          texuniform=False)
        T(spec, P + "ramp", P + "tex_concrete", rgba=(0.95, 0.93, 0.88, 1), specular=0.1, texrepeat=(3, 3),
          texuniform=False)
        T(spec, P + "curb_face", P + "tex_concrete", rgba=(0.9, 0.9, 0.88, 1), specular=0.1)
        A.add_texture(spec, P + "tex_store", A.storefront_texture())
        T(spec, P + "storefront", P + "tex_store", rgba=(1, 1, 1, 1), specular=0.2, emission=0.12)
        A.add_texture(spec, P + "tex_corral_sign", A.sign_texture("CART RETURN", "THANK YOU!"))
        T(spec, P + "corral_sign", P + "tex_corral_sign", rgba=(1, 1, 1, 1), emission=0.25)
        A.add_texture(spec, P + "tex_ramp_sign", A.sign_texture("CAUTION", "SLOPE - HOLD YOUR CART",
                                                                bg=(0.85, 0.62, 0.05)))
        T(spec, P + "ramp_sign", P + "tex_ramp_sign", rgba=(1, 1, 1, 1), emission=0.2)
        A.add_texture(spec, P + "tex_sky", A.sky_texture(c.seed))
        T(spec, P + "sky", P + "tex_sky", rgba=(1, 1, 1, 1), emission=0.85)
        M = spec.add_material
        for nm, rgba, sp in (("chrome", (0.78, 0.80, 0.84, 1), 0.9), ("red_plastic", (0.82, 0.10, 0.10, 1), 0.5),
                             ("rubber", (0.06, 0.06, 0.07, 1), 0.2), ("paint", (0.95, 0.95, 0.92, 1), 0.1),
                             ("paint_yellow", (0.98, 0.80, 0.12, 1), 0.1), ("steel", (0.55, 0.58, 0.62, 1), 0.7),
                             ("corral_blue", (0.12, 0.32, 0.72, 1), 0.6), ("glass", (0.12, 0.18, 0.26, 1), 1.0),
                             ("tyre", (0.05, 0.05, 0.05, 1), 0.2), ("trim", (0.12, 0.12, 0.13, 1), 0.3),
                             ("wall", (0.86, 0.82, 0.74, 1), 0.1), ("roof", (0.45, 0.45, 0.47, 1), 0.1),
                             ("grass", (0.30, 0.50, 0.22, 1), 0.05), ("bark", (0.38, 0.27, 0.18, 1), 0.05),
                             ("leaves", (0.22, 0.46, 0.20, 1), 0.1), ("stone", (0.62, 0.60, 0.57, 1), 0.1)):
            M(name=P + nm, rgba=rgba, specular=sp, shininess=0.6)
        for k, rgba in enumerate(((0.75, 0.12, 0.10, 1), (0.15, 0.32, 0.62, 1), (0.88, 0.88, 0.86, 1),
                                  (0.20, 0.45, 0.30, 1))):
            M(name=P + f"car{k}", rgba=rgba, specular=0.9, shininess=0.9)
        M(name=P + "lamp", rgba=(1.0, 0.95, 0.75, 1), emission=1.0)
        M(name=P + "tail", rgba=(0.9, 0.1, 0.1, 1), emission=0.4)

    def _add_scene(self, spec, vis) -> None:
        c = self.cfg
        wb = spec.worldbody

        def mesh(name, md, pos, material, quat=(1.0, 0.0, 0.0, 0.0)):
            A.add_mesh(spec, P + name + "_mesh", md)
            wb.add_geom(name=P + name, type=mj.mjtGeom.mjGEOM_MESH, meshname=P + name + "_mesh",
                        pos=tuple(pos), quat=tuple(quat), material=P + material, **vis)

        def paint(name, half, pos, mat="paint", quat=(1.0, 0.0, 0.0, 0.0)):
            wb.add_geom(name=P + name, type=mj.mjtGeom.mjGEOM_BOX, size=half, pos=pos, quat=quat,
                        material=P + mat, **vis)

        # painted bays: two rows of stalls in the flat lot, back to back
        k = 0
        for x in np.arange(-10.8, 19.0, 3.6):
            for y0, y1 in ((-11.5, -4.6), (-4.4, 1.6)):
                paint(f"bay{k}", (0.07, (y1 - y0) / 2, 0.01), (float(x), (y0 + y1) / 2, 0.02))
                k += 1
        paint("bay_mid", (15.0, 0.07, 0.01), (4.2, -4.5, 0.02))
        # the right-hand bays with the parked cars (along y)
        for y in np.arange(-10.0, 9.0, 3.6):
            paint(f"rbay{k}", (3.6, 0.07, 0.01), (23.5, float(y), 0.02))
            k += 1
        # the crosswalk to the sidewalk and the fire lane
        for i in range(7):
            paint(f"zebra{i}", (0.35, 1.0, 0.01), (-2.0 + 1.0 * i, c.lot_y1 - 1.3, 0.02))
        paint("fire_lane", (9.5, 0.1, 0.01), (-8.5, c.lot_y1 - 0.3, 0.02), "paint_yellow")
        # yellow edges up the ramp
        a = math.radians(c.ramp_deg)
        L = c.lot_y1 - c.ramp_y0
        q = quat_axis_angle((1, 0, 0), a)
        for x in (c.ramp_x0 + 0.25, c.ramp_x1 - 0.25):
            paint(f"ramp_edge{x:+.0f}", (0.12, L / 2 / math.cos(a), 0.012),
                  (x, (c.ramp_y0 + c.lot_y1) / 2, L / 2 * math.tan(a) + 0.012), "paint_yellow", q)
        # the curb face of the sidewalk (a lighter strip) and the store
        st = self.sidewalk_top
        xm = (c.lot_x0 + c.lot_x1) / 2
        bw = (c.lot_x1 - c.lot_x0) / 2 + 4
        mesh("store", A.box_mesh((bw, 5.0, 4.5), 0.2), (xm, c.sidewalk_y1 + 5.0, st + 4.5), "wall")
        mesh("store_front", A.panel_mesh(2 * bw, 9.0, 0.05), (xm, c.sidewalk_y1 - 0.03, st + 4.5), "storefront",
             A.panel_quat((0, -1, 0)))
        mesh("parapet", A.box_mesh((bw + 0.2, 0.3, 0.4), 0.2), (xm, c.sidewalk_y1 + 0.2, st + 9.3), "roof")
        mesh("canopy", A.box_mesh((5.2, 1.2, 0.12), 0.5), (4.0, c.sidewalk_y1 - 1.1, st + 3.9), "corral_blue")
        for s in (-1, 1):
            mesh(f"canopy_post{s:+d}", A.polyline_tube(np.array([[0, 0, 0], [0, 0, 3.9]]), 0.1, 8),
                 (4.0 + s * 4.8, c.sidewalk_y1 - 2.0, st), "steel")
        # planters with shrubs on the sidewalk
        for i, x in enumerate((-12.0, -5.0, 13.0, 20.0)):
            mesh(f"planter{i}", A.box_mesh((1.1, 0.7, 0.35), 0.5), (x, c.lot_y1 + 1.6, st + 0.35), "stone")
            mesh(f"shrub{i}", A.shrub_mesh((0, 0), 0.9, seed=i), (x, c.lot_y1 + 1.6, st + 0.55), "leaves")
        # the corral (visual pipe frame) with its sign
        mesh("corral", A.corral_mesh(c.corral_x0, c.corral_x1, c.corral_hw + 0.4), (0, c.corral_y, 0), "corral_blue")
        mesh("corral_sign", A.panel_mesh(3.0, 1.5, 0.05), (c.corral_x1 - 0.3, c.corral_y, 3.2), "corral_sign",
             A.panel_quat((1, 0, 0)))
        mesh("corral_sign_post", A.polyline_tube(np.array([[0, 0, 0], [0, 0, 2.5]]), 0.08, 8),
             (c.corral_x1 - 0.3, c.corral_y + c.corral_hw + 0.4, 0), "steel")
        mesh("ramp_sign", A.panel_mesh(2.6, 1.3, 0.05), (c.ramp_x1 + 1.2, c.ramp_y0 + 2.0, 2.6), "ramp_sign",
             A.panel_quat((0, -1, 0)))
        mesh("ramp_sign_post", A.polyline_tube(np.array([[0, 0, 0], [0, 0, 1.95]]), 0.07, 8),
             (c.ramp_x1 + 1.2, c.ramp_y0 + 2.05, 0), "steel")
        # parked cars (visual) in the right-hand bays, one on the left
        cars = A.car_meshes()
        for nm, md in cars.items():
            A.add_mesh(spec, P + "car_" + nm, md)
        for i, (x, y, yaw) in enumerate(((23.5, -8.2, math.pi), (23.5, -1.0, 0.0), (23.3, 6.2, math.pi),
                                         (-19.5, -6.0, math.pi / 2))):
            qz = quat_axis_angle((0, 0, 1), yaw)
            for nm, mat in (("body", f"car{i % 4}"), ("glass", "glass"), ("wheels", "tyre"), ("lights", "tail"),
                            ("trim", "trim")):
                wb.add_geom(name=f"{P}car{i}_{nm}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "car_" + nm,
                            pos=(x, y, 0.0), quat=qz, material=P + mat, **vis)
        # light poles (point lights under the heads)
        for nm, md in A.light_pole_meshes().items():
            A.add_mesh(spec, P + f"pole_{nm}", md)
        for i, (x, y, yaw) in enumerate(((-16.5, -3.0, 0.0), (20.0, -12.5, 0.75 * math.pi), (19.5, 10.0, math.pi))):
            qz = quat_axis_angle((0, 0, 1), yaw)
            for nm, mat in (("base", "stone"), ("pole", "steel"), ("head", "steel"), ("lens", "lamp")):
                wb.add_geom(name=f"{P}pole{i}_{nm}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + f"pole_{nm}",
                            pos=(x, y, 0.0), quat=qz, material=P + mat, **vis)
            head = np.array([x, y, 0.0]) + np.array([math.cos(yaw), math.sin(yaw), 0]) * 2.55
            wb.add_light(name=f"{P}lamp{i}", type=mj.mjtLightType.mjLIGHT_POINT, pos=(head[0], head[1], 10.6),
                         diffuse=(0.30, 0.27, 0.20), specular=(0.1, 0.1, 0.08), attenuation=(0.6, 0.02, 0.002),
                         castshadow=False)
        # grass verges and trees round the lot
        for i, (x, y, hx, hy) in enumerate(((-21.0, 0.0, 3.0, 16.0), (29.5, 0.0, 2.5, 16.0),
                                            (4.0, -15.5, 27.0, 1.8))):
            wb.add_geom(name=f"{P}verge{i}", type=mj.mjtGeom.mjGEOM_BOX, size=(hx, hy, 0.03), pos=(x, y, 0.03),
                        material=P + "grass", **vis)
        rng = np.random.default_rng(c.seed + 9)
        for i, (x, y) in enumerate(((-21.5, -11.0), (-22.0, 8.0), (30.0, -9.0), (30.5, 11.0))):
            tm = A.tree_meshes(rng.uniform(6.0, 8.5), seed=i)
            mesh(f"tree{i}_trunk", tm["trunk"], (x, y, 0), "bark")
            mesh(f"tree{i}_crown", tm["crown"], (x, y, 0), "leaves")
        mesh("sky", A.panel_mesh(300.0, 100.0, 0.5), (xm, c.sidewalk_y1 + 80.0, 28.0), "sky", A.panel_quat((0, -1, 0)))
        sky = spec.texture("skybox")
        if sky is not None:
            sky.rgb1 = (0.45, 0.62, 0.88)
            sky.rgb2 = (0.90, 0.88, 0.84)
        spec.visual.headlight.ambient = (0.32, 0.32, 0.32)
        spec.visual.headlight.diffuse = (0.30, 0.30, 0.30)
        spec.visual.headlight.specular = (0.06, 0.06, 0.06)
        tgt = np.array([2.0, 0.0, 0.0])
        key = np.array([-22.0, -30.0, 40.0])
        wb.add_light(name=P + "sun", type=spot_or_directional(c.shadows), pos=tuple(key), dir=tuple(tgt - key),
                     diffuse=(0.62, 0.58, 0.50), specular=(0.3, 0.3, 0.28), cutoff=45.0, exponent=0.3,
                     castshadow=bool(c.shadows))
        wb.add_light(name=P + "fill", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(20, -10, 30),
                     dir=(-0.4, 0.5, -1.0), diffuse=(0.20, 0.22, 0.26), specular=(0.05, 0.05, 0.06),
                     castshadow=False)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        m, d = self.sim.model, self.sim.data
        c = self.cfg
        n = c.n_carts
        self.c_body = np.array([m.body(f"{P}cart{j}").id for j in range(n)])
        self.c_push = np.array([m.geom(f"{P}cart{j}_push").id for j in range(n)])
        self.c_skid = np.array([m.geom(f"{P}cart{j}_skid").id for j in range(n)])
        jid = {k: [m.joint(f"{P}cart{j}_{k}").id for j in range(n)] for k in ("sx", "sy", "sz", "yaw")}
        self.q = {k: np.array([int(m.jnt_qposadr[i]) for i in v]) for k, v in jid.items()}
        self.v = {k: np.array([int(m.jnt_dofadr[i]) for i in v]) for k, v in jid.items()}
        self.base = np.array([self.park_xy(j) for j in range(n)])
        self.push_z = float(m.geom_pos[self.c_push[0], 2])
        self.state_c = np.full(n, LOOSE)
        self.slot = np.full(n, -1)
        self.yaw = np.zeros(n)
        self.pose = np.zeros((n, 3))  # last good (x, y, yaw) of each cart (for resets)
        self.anim: list = [None] * n  # kinematic glide: (waypoints, t0, speed, yaw0, yaw1)
        self.runaway = np.zeros(n, bool)  # left on the ramp by a customer, not yet caught
        self.rolling = np.zeros(n, bool)
        self.rng = np.random.default_rng(c.seed + 31)
        # counters
        self.returned = 0
        self.longest_train = 0
        self.customers = 0
        self.runaways = 0
        self.runaways_caught = 0
        self.caught_rolling = 0
        self.carts_lost = 0
        self.retargets = 0
        self.best_runaway_speed = 0.0
        self.target: int | None = None
        self.mode = "fetch"
        self._t_target = 0.0
        self._next_customer = self.run_time() + self.rng.exponential(c.customer_mean_s)
        self._caption = ("", -1e9, (255, 255, 255))
        self.message = ""
        pad = 1.0

        def clamp(p):
            return np.array([min(max(float(p[0]), c.lot_x0 + pad), c.lot_x1 - 3.0),
                             min(max(float(p[1]), c.lot_y0 + pad), c.lot_y1 - 0.6)])

        self.pilot = PushPilot(self, c.cart_radius, behind_gap=c.behind_gap, orbit_clearance=c.orbit_clearance,
                               push_speed=c.push_speed, approach_speed=c.approach_speed, clamp=clamp)
        # the start: a short train in the corral, the rest scattered round the lot
        for j in range(n):
            if j < c.n_nested0:
                self._nest_now(j, j)
            else:
                xy = self._drop_spot(on_ramp=False)
                self._place_loose(j, xy, self.rng.uniform(-math.pi, math.pi))
        self.longest_train = self.n_nested()
        mj.mj_forward(m, d)
        self.state = "fetch"

    # ------------------------------------------------------------ cart helpers
    def cart_xy(self, j: int) -> np.ndarray:
        return self.sim.data.xpos[self.c_body[j], :2].copy()

    def cart_vel(self, j: int) -> np.ndarray:
        d = self.sim.data
        return np.array([d.qvel[self.v["sx"][j]], d.qvel[self.v["sy"][j]]])

    def n_nested(self) -> int:
        return int(np.count_nonzero(np.isin(self.state_c, (NESTED, NESTING))))

    def loose_carts(self) -> np.ndarray:
        return np.flatnonzero(self.state_c == LOOSE)

    def _physics(self, j: int, on: bool) -> None:
        m = self.sim.model
        m.geom_contype[self.c_skid[j]] = PROP_BIT if on else 0
        m.geom_conaffinity[self.c_skid[j]] = TERRAIN_BIT if on else 0
        # the pairs with the fly can't be switched off: move the push collider away
        m.geom_pos[self.c_push[j], 2] = self.push_z if on else 80.0
        m.body_gravcomp[self.c_body[j]] = 0.0 if on else 1.0

    def _write_pose(self, j: int, x: float, y: float, yaw: float, z: float | None = None) -> None:
        d = self.sim.data
        b = self.base[j]
        d.qpos[self.q["sx"][j]] = x - b[0]
        d.qpos[self.q["sy"][j]] = y - b[1]
        d.qpos[self.q["sz"][j]] = self.ground_height(x, y) + 0.001 if z is None else z
        d.qpos[self.q["yaw"][j]] = yaw
        for k in ("sx", "sy", "sz", "yaw"):
            d.qvel[self.v[k][j]] = 0.0
        self.yaw[j] = yaw
        self.pose[j] = (x, y, yaw)

    def _place_loose(self, j: int, xy, yaw: float) -> None:
        self.state_c[j] = LOOSE
        self.slot[j] = -1
        self.anim[j] = None
        self._physics(j, True)
        self._write_pose(j, float(xy[0]), float(xy[1]), yaw)

    def _nest_now(self, j: int, k: int) -> None:
        self.state_c[j] = NESTED
        self.slot[j] = k
        self.anim[j] = None
        self._physics(j, False)
        s = self.slot_xy(k)
        self._write_pose(j, s[0], s[1], math.pi)

    def _drop_spot(self, on_ramp: bool) -> np.ndarray:
        """A random spot for a customer's cart: in the open lot (or on the ramp),
        away from the fly and from the other loose carts."""
        c = self.cfg
        fly = self.fly_xy() if self.sim is not None else np.zeros(2)
        others = [self.pose[j, :2] for j in range(c.n_carts) if self.state_c[j] in (LOOSE, CUSTOMER)]
        best, best_d = None, -1.0
        for _ in range(40):
            if on_ramp:
                p = np.array([self.rng.uniform(c.ramp_x0 + 2.0, c.ramp_x1 - 2.0),
                              self.rng.uniform(c.lot_y1 - 4.0, c.lot_y1 - 1.6)])
            else:
                p = np.array([self.rng.uniform(-10.0, 17.5), self.rng.uniform(c.lot_y0 + 2.2, 1.2)])
            dmin = min([float(np.linalg.norm(p - o)) for o in others] + [99.0])
            dfly = float(np.linalg.norm(p - fly))
            score = min(dmin / 3.0, dfly / 4.5)
            if dmin > 3.0 and dfly > 4.5:
                return p
            if score > best_d:
                best, best_d = p, score
        return best

    # ------------------------------------------------------------ reset
    def on_reset(self) -> None:
        self.pilot.reset()
        if self.target is not None and self.state_c[self.target] != LOOSE:
            self.target = None

    def reset_props(self) -> None:
        """The keyframe reset put every cart back at its parking spot: put them back
        where they were (loose carts at rest), the train in the corral."""
        for j in range(self.cfg.n_carts):
            x, y, yaw = self.pose[j]
            if not np.all(np.isfinite(self.pose[j])):
                x, y = self._drop_spot(False)
                yaw = 0.0
            if self.state_c[j] == LOOSE:
                self._physics(j, True)
            else:
                self._physics(j, False)
            self._write_pose(j, x, y, yaw)
        mj.mj_forward(self.sim.model, self.sim.data)

    # ------------------------------------------------------------ the loop
    def update(self) -> None:
        c = self.cfg
        sim = self.sim
        dt = c.update_every_steps * sim.timestep
        rt = self.run_time()
        self._carts_step(dt)
        self._customer_step(rt)
        acts = self.session.actions
        if acts.busy:
            self.steering.set(None, 1.0)
            return
        # a runaway takes priority
        run = [j for j in np.flatnonzero(self.runaway) if self.state_c[j] == LOOSE]
        if run and self.mode != "chase":
            self.target = int(run[0])
            self.mode = "chase"
            self.pilot.reset()
            self._t_target = rt
        if self.mode == "chase":
            self._chase(rt)
            return
        if self.target is None or self.state_c[self.target] != LOOSE or rt - self._t_target > c.retarget_s:
            if self.target is not None and rt - self._t_target > c.retarget_s:
                self.retargets += 1
            self._pick_target(rt)
        if self.target is None:
            self.state = "waiting for customers"
            self.steering.set(None, 0.0)
            return
        j = self.target
        s = self.cart_xy(j)
        g = self.goal(s)
        st = self.pilot.step(s, g)
        self.state = f"fetch cart {j}: {st}"
        self._turn_cart(j, math.atan2(g[1] - s[1], g[0] - s[0]), dt)
        # nesting: the cart is inside the corral mouth
        if self._in_mouth(s):
            self._start_nesting(j)

    def goal(self, s: np.ndarray) -> np.ndarray:
        """Pure pursuit onto the corral lane (the line through the mouth, y =
        corral_y) and along it into the mouth; a cart left of the staging point and
        off the lane is first brought round to the lane's outer end."""
        c = self.cfg
        S = self.staging_xy
        dy = abs(s[1] - c.corral_y)
        if s[0] < S[0] - 0.3 and dy > 0.8:
            return S + np.array([1.5, 0.0])
        gx = s[0] - c.pursuit if dy < 1.5 else max(s[0] - c.pursuit, S[0])
        return np.array([gx, c.corral_y])

    def _in_mouth(self, s: np.ndarray) -> bool:
        c = self.cfg
        return c.corral_x1 - 2.0 < s[0] < c.corral_x1 + 0.6 and abs(s[1] - c.corral_y) < 0.9

    def _pick_target(self, rt: float) -> None:
        loose = self.loose_carts()
        self.pilot.reset()
        if len(loose) == 0:
            self.target = None
            return
        fly = self.fly_xy()
        S = self.staging_xy
        cost = [float(np.linalg.norm(self.cart_xy(j) - fly)) + 0.5 * float(np.linalg.norm(self.cart_xy(j) - S))
                for j in loose]
        if self.target is not None and self.target in loose and rt - self._t_target > self.cfg.retarget_s:
            # this one has been a pain: try another first
            k = list(loose).index(self.target)
            cost[k] += 1e3
        self.target = int(loose[int(np.argmin(cost))])
        self._t_target = rt

    def _chase(self, rt: float) -> None:
        c = self.cfg
        j = self.target
        if j is None or self.state_c[j] != LOOSE:
            self.mode = "fetch"
            return
        s = self.cart_xy(j)
        v = self.cart_vel(j)
        sp = float(np.hypot(*v))
        lead = s + v * 0.4
        self.pilot.aim_at(lead, 1.0)
        self.state = f"CHASING runaway cart {j} ({sp:.0f} mm/s)"
        if sp > 0.5:
            self._turn_cart(j, math.atan2(v[1], v[0]), c.update_every_steps * self.sim.timestep)
        dist = float(np.linalg.norm(self.fly_xy() - s))
        if dist < c.cart_radius + c.catch_gap:
            self.runaway[j] = False
            self.runaways_caught += 1
            rolling = sp > 1.5
            self.caught_rolling += int(rolling)
            self._say_cap("CAUGHT IT!" + (" (still rolling)" if rolling else ""), (140, 255, 150))
            self.mode = "fetch"
            self.pilot.reset()
            self._t_target = rt
        elif rt - self._t_target > c.retarget_s:
            self.runaway[j] = False
            self.mode = "fetch"

    def _turn_cart(self, j: int, psi: float, dt: float) -> None:
        c = self.cfg
        dy = wrap_angle(psi - self.yaw[j])
        step = c.yaw_rate * dt
        self.yaw[j] = wrap_angle(self.yaw[j] + min(max(dy, -step), step))
        d = self.sim.data
        d.qpos[self.q["yaw"][j]] = self.yaw[j]
        d.qvel[self.v["yaw"][j]] = 0.0

    def _start_nesting(self, j: int) -> None:
        c = self.cfg
        k = self.n_nested()
        s = self.cart_xy(j)
        tgt = self.slot_xy(k)
        self.state_c[j] = NESTING
        self.slot[j] = k
        self._physics(j, False)
        self.anim[j] = ([s, tgt], self.run_time(), c.nest_speed, self.yaw[j], math.pi)
        self.returned += 1
        self.add_work(1)
        n = self.n_nested()
        record = n > self.longest_train
        self.longest_train = max(self.longest_train, n)
        self.target = None
        self.pilot.reset()
        self.message = f"cart #{self.returned} returned (train of {n})"
        self._say_cap(f"CART #{self.returned} RETURNED" + (f" - NEW RECORD TRAIN OF {n}!" if record else ""),
                      (255, 235, 140))
        self.session.log_event("cart_returned", job=self.name, n=self.returned, train=n)

    # ------------------------------------------------------------ carts (physics / kinematic)
    def _carts_step(self, dt: float) -> None:
        c = self.cfg
        rt = self.run_time()
        for j in range(c.n_carts):
            st = self.state_c[j]
            if st == LOOSE:
                s = self.cart_xy(j)
                q = self.sim.data.qpos
                ok = np.all(np.isfinite(s)) and np.isfinite(q[self.q["sz"][j]])
                if (not ok or s[0] < c.lot_x0 - 6 or s[0] > c.lot_x1 + 8 or s[1] < c.lot_y0 - 5
                        or s[1] > c.sidewalk_y1 or q[self.q["sz"][j]] < -0.5):
                    self.carts_lost += 1
                    self._place_loose(j, self._drop_spot(False), 0.0)
                    continue
                self.pose[j] = (s[0], s[1], self.yaw[j])
                v = self.cart_vel(j)
                sp = float(np.hypot(*v))
                pushed = self.target == j and self.pilot.pushing(s)
                if sp > 2.0 and not pushed and not self.rolling[j]:
                    self.rolling[j] = True
                    if self.runaway[j]:
                        self.runaways += 1
                        self._say_cap("RUNAWAY CART!", (255, 140, 110))
                        self.session.log_event("runaway_cart", job=self.name, cart=int(j))
                elif sp < 0.5:
                    self.rolling[j] = False
                if self.rolling[j]:
                    if self.runaway[j]:
                        self.best_runaway_speed = max(self.best_runaway_speed, sp)
                    if self.target != j or self.mode == "chase":
                        self._turn_cart(j, math.atan2(v[1], v[0]), dt)
                if self._in_mouth(s) and self.target != j:  # rolled / pushed in by accident
                    self._start_nesting(j)
            elif st in (NESTING, CUSTOMER):
                wps, t0, speed, y0, y1 = self.anim[j]
                segs = [float(np.linalg.norm(wps[i + 1] - wps[i])) for i in range(len(wps) - 1)]
                total = max(sum(segs), 1e-6)
                s_done = min((rt - t0) * speed, total)
                u = s_done / total
                acc, p = 0.0, wps[-1]
                for i, L in enumerate(segs):
                    if s_done <= acc + L or i == len(segs) - 1:
                        a = 0.0 if L < 1e-9 else min(max((s_done - acc) / L, 0.0), 1.0)
                        p = wps[i] + a * (wps[i + 1] - wps[i])
                        break
                    acc += L
                yaw = y0 + wrap_angle(y1 - y0) * min(1.0, u * 1.6)
                self._write_pose(j, float(p[0]), float(p[1]), yaw)
                if u >= 1.0:
                    if st == NESTING:
                        self._nest_now(j, int(self.slot[j]))
                    else:
                        self._arrive(j)
            elif st == NESTED:
                s = self.slot_xy(int(self.slot[j]))
                self._write_pose(j, s[0], s[1], math.pi)

    def _customer_step(self, rt: float) -> None:
        c = self.cfg
        if len(self.loose_carts()) == 0 and self.n_nested() > 0:
            self._next_customer = min(self._next_customer, rt + c.customer_idle_s)
        if rt < self._next_customer:
            return
        busy = np.any(self.state_c == CUSTOMER) or np.any(self.state_c == NESTING)
        nested = [j for j in range(c.n_carts) if self.state_c[j] == NESTED]
        near = float(np.linalg.norm(self.fly_xy() - self.mouth)) < 5.0
        if busy or not nested or near:
            self._next_customer = rt + 1.0
            return
        j = max(nested, key=lambda i: self.slot[i])  # the last cart of the train
        ramp = self.rng.random() < c.runaway_p
        dest = self._drop_spot(on_ramp=ramp)
        out = self.mouth + np.array([1.5, 0.0])
        self.state_c[j] = CUSTOMER
        self.slot[j] = -1
        self.runaway[j] = bool(ramp)
        self.anim[j] = ([self.cart_xy(j), out, dest], rt, c.customer_speed, self.yaw[j],
                        float(self.rng.uniform(-math.pi, math.pi)))
        self.customers += 1
        self._next_customer = rt + c.customer_idle_s + self.rng.exponential(c.customer_mean_s)
        self.message = f"a customer took a cart{' (up the ramp)' if ramp else ''}"

    def _arrive(self, j: int) -> None:
        s = self.cart_xy(j)
        self._place_loose(j, s, self.yaw[j])
        self.rolling[j] = False
        where = "ON THE RAMP" if self.runaway[j] else "IN THE LOT"
        self._say_cap(f"A CUSTOMER LEFT A CART {where}", (230, 230, 255))

    # ------------------------------------------------------------ view / HUD
    def _say_cap(self, text: str, rgb=(255, 255, 255)) -> None:
        self.message = text
        self._caption = (text, self.run_time(), rgb)
        self.say(text)

    def post_process(self, frame: np.ndarray, t: float) -> np.ndarray:
        """A caption overlay (CART #n RETURNED, RUNAWAY CART!, CAUGHT IT!)."""
        cap, t0, rgb = self._caption
        if not (self.cfg.captions and cap and 0 <= t - t0 < 2.0):
            return frame
        import cv2

        H, W = frame.shape[:2]
        out = np.ascontiguousarray(frame).copy()
        fs = max(0.4, W / 1400.0) * 1.5
        th = max(1, int(round(W / 700))) * 2 + 1
        (tw, tht), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_DUPLEX, fs, th)
        x, y = max(4, (W - tw) // 2), int(0.11 * H) + tht
        cv2.putText(out, cap, (x + 2, y + 2), cv2.FONT_HERSHEY_DUPLEX, fs, (20, 20, 30), th + 2, cv2.LINE_AA)
        cv2.putText(out, cap, (x, y), cv2.FONT_HERSHEY_DUPLEX, fs, rgb, th, cv2.LINE_AA)
        return out

    def camera_target(self) -> np.ndarray:
        f = self.sim.thorax_position()
        c = self.cfg
        j = self.target
        s = np.append(self.cart_xy(j), 0.6) if j is not None else f
        mid = np.array([2.0, -1.0, 0.0])
        out = 0.4 * f + 0.25 * s + 0.35 * mid
        return out if np.all(np.isfinite(out)) else f

    def camera_preset(self) -> CameraPreset:
        return CameraPreset(azimuth=95.0, elevation=-36.0, distance=32.0, tau_s=1.2)

    def job_hud_lines(self) -> list[str]:
        n_loose = len(self.loose_carts())
        return [
            f"cart train {self.n_nested()} (longest {self.longest_train})   loose carts {n_loose}   "
            f"customers {self.customers}",
            f"runaways {self.runaways}   caught {self.runaways_caught} ({self.caught_rolling} still rolling)   "
            f"top runaway speed {self.best_runaway_speed:.0f} mm/s   carts lost {self.carts_lost}",
            "(carts: planar joints + gravity on the ramp, real physics; nesting glide + customer: kinematic)",
        ]

    def job_stats(self) -> dict:
        return {"carts_returned": self.returned, "longest_train": self.longest_train, "train": self.n_nested(),
                "customers": self.customers, "runaways": self.runaways, "runaways_caught": self.runaways_caught,
                "caught_rolling": self.caught_rolling, "best_runaway_speed": round(self.best_runaway_speed, 1),
                "carts_lost": self.carts_lost, "retargets": self.retargets, "loose": len(self.loose_carts()),
                "pushes_started": self.pilot.n_pushes, "unstuck": self.n_unstuck}
