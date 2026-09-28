"""Snow shovel: the fly shovels snow forever. It keeps snowing.

World (the fly spawns at the origin facing +x, on the driveway):

* a fly-scale driveway (``path_x0 .. path_x1`` x ``path_y0 .. path_y1``) in front of a
  little house, the front lawn with the snowbank between them (+y), the street on
  the camera side (-y);
* the **snow cover** on the driveway: a fixed pool of ``nx x ny`` flattened snow
  domes (visual ellipsoids, no contacts: the fly walks through the snow, its feet
  sink in), each with its own depth (0 .. ``max_depth`` mm). The snowfall adds depth
  everywhere at a rate that follows the weather (storms and lulls), with a fixed
  drift pattern; cleared driveway shows the wet concrete;
* the **snowfall** is a fixed pool of kinematic flakes (mocap, visual) fluttering down
  over the scene; a flake that lands is recycled to the sky;
* the **shovel**: planar joints (slide x, slide y, yaw hinge) at a fixed height like
  the mowing job's mower, so it can't tip or be lost; the only colliding part is a
  round brace behind the blade that the fly pushes with its head / thorax / abdomen
  at low friction (``slippery_body_contact``, legs excluded). The blade, the shaft
  and the D-grip over the fly's head are visual. The job turns the yaw hinge toward
  the push direction (rate limited, like the mower).

Behaviour: lanes across the driveway toward the bank (+y), one every
``lane_width`` mm along x. ``PushPilot`` (the mowing push loop) pushes the shovel up
a lane with the blade down: every snow dome under the blade's footprint is scraped
(depth -> 0) and its volume goes into the load on the blade (a growing heap).
At the bank edge the load is **dumped**: the heap turns into pooled snow clumps
(real free bodies; the throw velocity is set by the job, **engineered**) that tumble
onto the bank with real contacts and then sleep (contacts off, held) until the pool
needs them; the bank itself (visual segments along the lawn edge) grows with the
volume shovelled and slowly settles (compaction / sublimation, τ ``bank_settle_s``).
Then the fly walks round the shovel and pushes it back to the next lane's start with
the blade up (no scraping on the return, labelled), and the next lane. The snowfall
re-covers the driveway behind it, so it never ends.

Counters: metres of path cleared (swath length of snow scraped with the 3 mm blade),
clumps shovelled, snow volume, bank height (mm and at human scale), lanes, passes,
days of winter (``day_s`` sim seconds each).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from fly_simulator.jobs import snow_shovel_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import contact_kwargs, quat_axis_angle, slippery_body_contact, wrap_angle
from fly_simulator.jobs.mowing import PushPilot
from fly_simulator.jobs.geometry import spot_or_directional
from fly_simulator.jobs.registry import register_job
from fly_simulator.terrain import TERRAIN_BIT

P = "snow/"
CLUMP_BIT = 128  # clumps touch each other and the ground, never the fly / the shovel
HUMAN = 1750.0 / 2.5  # 1.75 m human per 2.5 mm fly

PARK, FLY, REST, SLEEP = 0, 1, 2, 3


@dataclass
class SnowShovelConfig(JobConfig):
    # ---- driveway / snow ----------------------------------------------------------
    path_x0: float = -3.0
    path_x1: float = 19.0
    path_y0: float = -4.5
    path_y1: float = 4.5
    tile: float = 0.5  # snow dome spacing (mm)
    max_depth: float = 0.2  # mm of fresh snow at most
    start_depth: float = 0.14
    # snowfall: depth rate (mm/s) = snow_rate * intensity(t); intensity cycles
    # between lulls and storms (0.35 .. 1.65, period storm_period_s)
    snow_rate: float = 0.0012
    storm_period_s: float = 97.0
    n_flakes: int = 110
    day_s: float = 60.0  # sim seconds per "day of winter"
    # ---- shovel ---------------------------------------------------------------------
    lane_width: float = 3.0
    blade_half_w: float = 1.5
    brace_radius: float = 0.8
    shovel_z: float = 0.85
    shovel_half_h: float = 0.38
    shovel_mass: float = 2e-4
    shovel_damping: float = 0.3  # slide damping (uN per mm/s); x (1 + load / load_ref)
    load_ref: float = 3.0  # mm^3
    head_friction: float = 0.05
    yaw_rate: float = 2.5
    # ---- clumps / bank ------------------------------------------------------------------
    n_clumps: int = 30
    clump_radius: float = 0.26
    clump_mass: float = 3e-6
    clump_volume: float = 0.6  # mm^3 of shovelled snow per clump thrown (cartoon clumps)
    max_clumps_per_dump: int = 6
    bank_y: float = 7.2  # bank centre line
    bank_segments: int = 12
    bank_settle_s: float = 1500.0  # the bank settles / sublimates with this time constant
    bank_max_h: float = 2.6  # visual cap of a bank segment (mm)
    # ---- behaviour --------------------------------------------------------------------
    behind_gap: float = 1.2
    orbit_clearance: float = 1.3
    push_speed: float = 0.75
    approach_speed: float = 0.9
    pursuit: float = 2.5
    celebrate_s: float = 1.5  # groom after a full pass
    captions: bool = True
    shadows: bool = True


@register_job
class SnowShovelJob(EternalJob):
    name = "snow_shovel"
    title = "SNOW SHOVEL FLY"
    tagline = "the fly shovels snow forever"
    work_label = "path cleared"
    work_format = "{:.4f} m"
    znear = 0.05
    config_cls = SnowShovelConfig
    required_names = (P + "shovel", P + "brace", P + "clump0")

    cfg: SnowShovelConfig

    # ------------------------------------------------------------ geometry
    @property
    def n_lanes(self) -> int:
        c = self.cfg
        return int((c.path_x1 - c.path_x0) // c.lane_width)

    def lane_x(self, k: int) -> float:
        c = self.cfg
        return c.path_x0 + c.lane_width * (k % self.n_lanes + 0.5)

    @property
    def y_start(self) -> float:
        return self.cfg.path_y0 + self.cfg.brace_radius + 0.4

    @property
    def y_end(self) -> float:  # the blade's edge reaches past the driveway edge
        return self.cfg.path_y1 - 0.5

    @property
    def shovel_start(self) -> tuple[float, float]:
        return (self.lane_x(1), self.y_start)

    def tile_layout(self) -> np.ndarray:
        c = self.cfg
        xs = np.arange(c.path_x0 + c.tile / 2, c.path_x1, c.tile)
        ys = np.arange(c.path_y0 + c.tile / 2, c.path_y1, c.tile)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        return np.column_stack([gx.ravel(), gy.ravel()])

    def bank_x(self) -> np.ndarray:
        c = self.cfg
        return np.linspace(c.path_x0 + 0.8, c.path_x1 - 0.8, c.bank_segments)

    # ------------------------------------------------------------ build
    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        self._materials(spec)
        vis = dict(contact_kwargs("visual"), mass=0.0)
        cx, cy = (c.path_x0 + c.path_x1) / 2, (c.path_y0 + c.path_y1) / 2
        hx, hy = (c.path_x1 - c.path_x0) / 2, (c.path_y1 - c.path_y0) / 2
        A.add_mesh(spec, P + "drive_mesh", A.box_mesh((hx, hy, 0.02), 0.35))
        wb.add_geom(name=P + "driveway", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "drive_mesh",
                    pos=(cx, cy, 0.02), material=P + "wet_concrete", **vis)
        # the snow domes (a fixed pool of visual ellipsoids)
        xy = self.tile_layout()
        r = c.tile * 0.74
        for i, (x, y) in enumerate(xy):
            wb.add_geom(name=f"{P}tile{i}", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(r, r, c.start_depth),
                        pos=(float(x), float(y), 0.04), material=P + "snow_tile", rgba=(1, 1, 1, 1), **vis)
        # the shovel (planar joints, fixed height)
        x0, y0 = self.shovel_start
        body = wb.add_body(name=P + "shovel", pos=(x0, y0, 0.0))
        pad = 3.0
        body.add_joint(name=P + "slide_x", type=mj.mjtJoint.mjJNT_SLIDE, axis=(1, 0, 0),
                       damping=c.shovel_damping, limited=True,
                       range=(c.path_x0 - pad - x0, c.path_x1 + pad - x0))
        body.add_joint(name=P + "slide_y", type=mj.mjtJoint.mjJNT_SLIDE, axis=(0, 1, 0),
                       damping=c.shovel_damping, limited=True,
                       range=(c.path_y0 - pad - y0, c.path_y1 + pad - y0))
        body.add_joint(name=P + "yaw", type=mj.mjtJoint.mjJNT_HINGE, axis=(0, 0, 1), damping=1.0, armature=1e-3)
        kw = contact_kwargs("dynamic", 1.0)
        body.add_geom(name=P + "brace", type=mj.mjtGeom.mjGEOM_CYLINDER,
                      size=(c.brace_radius, c.shovel_half_h, 0), pos=(0, 0, c.shovel_z),
                      mass=c.shovel_mass, rgba=(0.2, 0.2, 0.25, 0.0), group=3, **kw)
        self._add_shovel_looks(spec, body, vis)
        slippery_body_contact(spec, P + "shovel", P + "brace", self.fly_name, friction=c.head_friction)
        # the load heap on the blade (visual, resized at run time)
        body.add_geom(name=P + "load", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(0.3, 1.2, 0.05),
                      pos=(1.35, 0, 0.05), material=P + "snow_tile", rgba=(1, 1, 1, 0), **vis)
        # snow clumps (free bodies, compiled with their live contact bits; parked at run time)
        for j in range(c.n_clumps):
            b = wb.add_body(name=f"{P}clump{j}", pos=(-8.0 - 0.7 * j, -12.0, -3.0))
            b.add_freejoint(name=f"{P}clump{j}_free")
            b.add_geom(name=f"{P}clump{j}_col", type=mj.mjtGeom.mjGEOM_SPHERE, size=(c.clump_radius, 0, 0),
                       mass=c.clump_mass, contype=CLUMP_BIT, conaffinity=CLUMP_BIT | TERRAIN_BIT, condim=6,
                       friction=(0.9, 0.01, 0.02), solref=(0.002, 1.0), rgba=(1, 1, 1, 0), group=3)
            A.add_mesh(spec, f"{P}clump{j}_mesh", A.boulder_mesh(c.clump_radius * 1.05, seed=j, amp=0.12,
                                                                    n_theta=14, n_phi=9))
            b.add_geom(name=f"{P}clump{j}_vis", type=mj.mjtGeom.mjGEOM_MESH, meshname=f"{P}clump{j}_mesh",
                       material=P + "clump", **vis)
            b.gravcomp = 1.0
        # the bank (visual segments, resized at run time)
        for j, x in enumerate(self.bank_x()):
            wb.add_geom(name=f"{P}bank{j}", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(1.4, 1.6, 0.2),
                        pos=(float(x), c.bank_y, 0.0), material=P + "bank", **vis)
        # snowflakes (mocap, visual)
        A.add_mesh(spec, P + "flake_mesh", A.disc_mesh(0.12, 0.02, 6))
        for j in range(c.n_flakes):
            f = wb.add_body(name=f"{P}flake{j}", mocap=True, pos=(0.0, 0.0, -5.0))
            f.add_geom(name=f"{P}flake{j}_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "flake_mesh",
                       material=P + "flake", **vis)
        self._add_scene(spec, vis)

    def _materials(self, spec) -> None:
        c = self.cfg
        T = A.add_textured_material
        mat = spec.material("grid")
        if mat is not None:  # the snowy yard (the floor plane)
            A.add_texture(spec, P + "tex_ground", A.snow_texture(c.seed))
            mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = P + "tex_ground"
            mat.rgba = (0.97, 0.98, 1.0, 1.0)
            mat.reflectance = 0.0
            mat.specular = 0.25
        wet = (A.concrete_texture(c.seed).astype(np.float32) * 0.55).astype(np.uint8)
        A.add_texture(spec, P + "tex_wet", wet)
        T(spec, P + "wet_concrete", P + "tex_wet", rgba=(0.52, 0.55, 0.62, 1), specular=0.6, shininess=0.8)
        spec.add_material(name=P + "snow_tile", rgba=(1, 1, 1, 1), specular=0.25, shininess=0.3)
        A.add_texture(spec, P + "tex_plowed", A.plowed_snow_texture(c.seed + 1, 256))
        T(spec, P + "bank", P + "tex_plowed", rgba=(0.92, 0.94, 1, 1), specular=0.3, shininess=0.4)
        T(spec, P + "clump", P + "tex_plowed", rgba=(0.95, 0.97, 1, 1), specular=0.3, shininess=0.3)
        spec.add_material(name=P + "flake", rgba=(1, 1, 1, 0.9), emission=0.5, specular=0.5)
        A.add_texture(spec, P + "tex_asphalt", A.asphalt_texture(c.seed))
        T(spec, P + "asphalt", P + "tex_asphalt", rgba=(1, 1, 1, 1), specular=0.3)
        A.add_texture(spec, P + "tex_facade", A.facade_texture(siding=(0.66, 0.78, 0.70)))
        T(spec, P + "facade", P + "tex_facade", rgba=(1, 1, 1, 1), specular=0.1, emission=0.08)
        A.add_texture(spec, P + "tex_roof", A.snowy_roof_texture(c.seed))
        T(spec, P + "roof", P + "tex_roof", rgba=(1, 1, 1, 1), specular=0.15)
        A.add_texture(spec, P + "tex_sign", A.sign_texture())
        T(spec, P + "sign", P + "tex_sign", rgba=(1, 1, 1, 1), specular=0.1)
        A.add_texture(spec, P + "tex_sky", A.sky_texture(c.seed))
        T(spec, P + "sky", P + "tex_sky", rgba=(1, 1, 1, 1), emission=0.8)
        M = spec.add_material
        for nm, rgba, sp in (("siding", (0.62, 0.74, 0.66, 1), 0.1), ("blade", (0.10, 0.35, 0.85, 1), 0.8),
                             ("blade_edge", (0.75, 0.77, 0.8, 1), 1.0), ("wood", (0.62, 0.44, 0.26, 1), 0.2),
                             ("grip", (0.1, 0.1, 0.1, 1), 0.3), ("white", (0.96, 0.96, 0.97, 1), 0.3),
                             ("snowman", (0.97, 0.98, 1.0, 1), 0.2), ("coal", (0.06, 0.06, 0.07, 1), 0.3),
                             ("carrot", (0.95, 0.45, 0.08, 1), 0.3), ("twig", (0.35, 0.22, 0.12, 1), 0.1),
                             ("pine", (0.10, 0.30, 0.20, 1), 0.1), ("trunk", (0.35, 0.24, 0.15, 1), 0.1),
                             ("mailbox", (0.15, 0.18, 0.22, 1), 0.5), ("post", (0.45, 0.32, 0.2, 1), 0.1),
                             ("red", (0.85, 0.1, 0.1, 1), 0.4), ("curb", (0.62, 0.62, 0.64, 1), 0.1),
                             ("brass", (0.7, 0.55, 0.25, 1), 0.9), ("door", (0.55, 0.12, 0.10, 1), 0.3),
                             ("scarf", (0.85, 0.12, 0.18, 1), 0.2)):
            M(name=P + nm, rgba=rgba, specular=sp, shininess=0.5)
        M(name=P + "lamp_glass", rgba=(1.0, 0.85, 0.5, 1), emission=1.0, specular=0.5)

    def _add_shovel_looks(self, spec, body, vis) -> None:
        """Blade (in the shovel frame, +x = push direction), shaft and D-grip."""
        c = self.cfg

        def mesh(name, md, material, pos=(0.0, 0.0, 0.0)):
            A.add_mesh(spec, P + name + "_mesh", md)
            body.add_geom(name=P + name, type=mj.mjtGeom.mjGEOM_MESH, meshname=P + name + "_mesh",
                          pos=tuple(pos), material=P + material, **vis)

        mesh("blade", A.transform(A.blade_mesh(c.blade_half_w, 0.95, 0.3), np.eye(3), np.array([0.95, 0, 0.01])),
             "blade")
        body.add_geom(name=P + "blade_edge", type=mj.mjtGeom.mjGEOM_BOX, size=(0.04, c.blade_half_w, 0.02),
                      pos=(1.23, 0, 0.02), material=P + "blade_edge", **vis)
        grip = np.array([-(c.brace_radius + c.behind_gap + 0.2), 0.0, 2.35])
        hd = A.shovel_handle((0.85, 0.0, 0.6), grip)
        mesh("shaft", hd["shaft"], "wood")
        mesh("dgrip", hd["grip"], "grip")
        body.add_geom(name=P + "blade_brace", type=mj.mjtGeom.mjGEOM_BOX, size=(0.25, 0.5, 0.05),
                      pos=(0.75, 0, 0.55), quat=quat_axis_angle((0, 1, 0), -0.6), material=P + "blade", **vis)

    def _add_scene(self, spec, vis) -> None:
        """Street, curb, lawn edge, the house with a porch light, a mailbox with a snow
        cap, a snowman, snowy pines, a picket fence, the sky; lights."""
        c = self.cfg
        wb = spec.worldbody

        def mesh(name, md, pos, material, quat=(1.0, 0.0, 0.0, 0.0)):
            A.add_mesh(spec, P + name + "_mesh", md)
            wb.add_geom(name=P + name, type=mj.mjtGeom.mjGEOM_MESH, meshname=P + name + "_mesh",
                        pos=tuple(pos), quat=tuple(quat), material=P + material, **vis)

        xm = (c.path_x0 + c.path_x1) / 2
        # the street and the curb on the camera side
        mesh("street", A.box_mesh((40.0, 4.0, 0.02), 1.0), (xm, c.path_y0 - 6.5, 0.02), "asphalt")
        mesh("curb", A.box_mesh((40.0, 0.25, 0.12), 1.0), (xm, c.path_y0 - 2.3, 0.12), "curb")
        # the house behind the bank
        hy0 = c.bank_y + 5.0
        hw_, hd, hh = 14.0, 5.0, 8.0
        mesh("house", A.box_mesh((hw_, hd, hh / 2), 0.2), (xm, hy0 + hd, hh / 2), "siding")
        from fly_simulator.jobs.taste_tester_assets import front_panel_mesh

        mesh("facade", front_panel_mesh(hw_, hh / 2, 0.02), (xm, hy0 - 0.02, hh / 2), "facade",
             quat_axis_angle((0, 0, 1), -math.pi / 2))
        mesh("roof", A.gable_roof_mesh(2 * hw_, 2 * hd, 4.0), (xm, hy0 + hd, hh - 0.3), "roof")
        # porch: steps, a front walk to the driveway, the lantern with a warm light
        for k in range(2):
            mesh(f"step{k}", A.box_mesh((1.6 - 0.3 * k, 0.5, 0.15), 0.5),
                 (xm, hy0 - 0.5 - 0.5 * (1 - k), 0.15 + 0.3 * k), "curb")
            mesh(f"step{k}_snow", A.box_mesh((1.6 - 0.3 * k, 0.5, 0.04), 0.5),
                 (xm, hy0 - 0.5 - 0.5 * (1 - k), 0.32 + 0.3 * k), "snowman")
        lamp = A.lamp_meshes()
        lp = np.array([xm - 2.6, hy0 - 0.3, 3.4])
        for nm, mat in (("bracket", "coal"), ("glass", "lamp_glass"), ("cap", "coal")):
            mesh(f"lamp_{nm}", lamp[nm], lp, mat)
        wb.add_light(name=P + "porch_light", type=mj.mjtLightType.mjLIGHT_POINT, pos=tuple(lp - np.array([0, 0.6, 0])),
                     diffuse=(0.55, 0.40, 0.18), specular=(0.2, 0.15, 0.05), attenuation=(0.3, 0.06, 0.01),
                     castshadow=False)
        mesh("sign", A.panel_mesh(3.6, 1.2, 0.05), (xm + 5.0, hy0 - 0.08, 3.4), "sign", A.panel_quat((0, -1, 0)))
        # mailbox with a snow cap beside the driveway
        mb = A.mailbox_meshes(2.2)
        mbp = (c.path_x1 + 1.8, c.path_y0 - 1.4, 0.0)
        for nm, mat in (("post", "post"), ("box", "mailbox"), ("flag", "red")):
            mesh(f"mailbox_{nm}", mb[nm], mbp, mat)
        wb.add_geom(name=P + "mailbox_snow", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(0.78, 0.46, 0.18),
                    pos=(mbp[0], mbp[1], 2.2 + 0.72), material=P + "snowman", **vis)
        # a snowman on the lawn
        sm = A.snowman_meshes(3.2)
        smp = (c.path_x0 - 3.5, c.bank_y - 0.5, 0.0)
        for nm, mat in (("body", "snowman"), ("coal", "coal"), ("nose", "carrot"), ("arms", "twig")):
            mesh(f"snowman_{nm}", sm[nm], smp, mat, quat_axis_angle((0, 0, 1), 0.35))
        wb.add_geom(name=P + "snowman_scarf", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.62, 0.08, 0),
                    pos=(smp[0], smp[1], 2.07), material=P + "scarf", **vis)
        # pines and a fence
        rng = np.random.default_rng(c.seed + 3)
        for j, (px, py) in enumerate(((c.path_x0 - 8, c.bank_y + 2), (c.path_x1 + 5, c.bank_y + 1.0),
                                      (c.path_x1 + 9, c.bank_y + 6), (c.path_x0 - 12, c.bank_y + 8))):
            pm = A.pine_meshes(rng.uniform(7.0, 10.0), seed=j)
            for nm, mat in (("trunk", "trunk"), ("green", "pine"), ("snow", "snowman")):
                mesh(f"pine{j}_{nm}", pm[nm], (px, py, 0.0), mat)
        mesh("fence", A.picket_fence_mesh(-24.0, 42.0, 1.8), (0, hy0 - 2.2 + 0.0, 0), "white")
        mesh("fence_snow", A.box_mesh((33.0, 0.08, 0.05), 1.0), (9.0, hy0 - 2.2, 1.85), "snowman")
        mesh("sky", A.panel_mesh(260.0, 90.0, 0.5), (xm, hy0 + 70.0, 25.0), "sky", A.panel_quat((0, -1, 0)))
        sky = spec.texture("skybox")
        if sky is not None:
            sky.rgb1 = (0.60, 0.66, 0.76)
            sky.rgb2 = (0.88, 0.90, 0.95)
        spec.visual.headlight.ambient = (0.30, 0.32, 0.38)
        spec.visual.headlight.diffuse = (0.28, 0.30, 0.34)
        spec.visual.headlight.specular = (0.05, 0.05, 0.06)
        tgt = np.array([xm, 1.5, 0.0])
        key = np.array([xm - 18.0, -26.0, 34.0])
        wb.add_light(name=P + "key", type=spot_or_directional(c.shadows), pos=tuple(key), dir=tuple(tgt - key),
                     diffuse=(0.55, 0.60, 0.72), specular=(0.3, 0.32, 0.4), cutoff=45.0, exponent=0.3,
                     castshadow=bool(c.shadows))
        wb.add_light(name=P + "fill", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(xm + 20, -10, 30),
                     dir=(-0.3, 0.4, -1.0), diffuse=(0.22, 0.25, 0.32), specular=(0.05, 0.05, 0.06),
                     castshadow=False)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        m = self.sim.model
        c = self.cfg
        self.shovel_body = m.body(P + "shovel").id
        jx, jy, jz = (m.joint(P + n).id for n in ("slide_x", "slide_y", "yaw"))
        self.qx, self.qy, self.qyaw = (int(m.jnt_qposadr[j]) for j in (jx, jy, jz))
        self.vx, self.vy, self.vyaw = (int(m.jnt_dofadr[j]) for j in (jx, jy, jz))
        self.load_geom = m.geom(P + "load").id
        self.tile_xy = self.tile_layout()
        self.n_tiles = len(self.tile_xy)
        g0 = m.geom(f"{P}tile0").id
        self.tile_geoms = np.arange(g0, g0 + self.n_tiles)
        assert m.geom(f"{P}tile{self.n_tiles - 1}").id == self.tile_geoms[-1]
        rng = np.random.default_rng(c.seed + 77)
        # a fixed drift pattern: some spots collect more snow
        fx = self.tile_xy[:, 0]
        fy = self.tile_xy[:, 1]
        self.drift = (1.0 + 0.18 * np.sin(fx * 0.9 + 1.3) * np.cos(fy * 0.7) + 0.08 * rng.normal(size=self.n_tiles))
        self.drift = np.clip(self.drift, 0.6, 1.4)
        self.tint = rng.uniform(0.95, 1.0, self.n_tiles)
        self.depth = np.clip(c.start_depth * self.drift, 0, c.max_depth)
        self.tile_area = c.tile ** 2
        self._write_tiles(np.ones(self.n_tiles, bool))
        # clumps
        self.c_bid = np.array([m.body(f"{P}clump{j}").id for j in range(c.n_clumps)])
        self.c_geom = np.array([m.geom(f"{P}clump{j}_col").id for j in range(c.n_clumps)])
        self.c_q = np.array([int(m.jnt_qposadr[m.joint(f"{P}clump{j}_free").id]) for j in range(c.n_clumps)])
        self.c_v = np.array([int(m.jnt_dofadr[m.joint(f"{P}clump{j}_free").id]) for j in range(c.n_clumps)])
        self.c_state = np.zeros(c.n_clumps, int)
        self.c_t = np.zeros(c.n_clumps)
        self.c_slow = np.full(c.n_clumps, math.inf)
        self.c_park = np.array([[-8.0 - 0.7 * j, -12.0, -3.0] for j in range(c.n_clumps)])
        for j in range(c.n_clumps):
            self._park_clump(j)
        # bank
        self.bank_geoms = np.array([m.geom(f"{P}bank{j}").id for j in range(c.bank_segments)])
        self.bank_vol = np.full(c.bank_segments, 1.5)  # a little snow left from last night
        self._write_bank()
        # flakes
        self.fl_mocap = np.array([int(m.body_mocapid[m.body(f"{P}flake{j}").id]) for j in range(c.n_flakes)])
        self._frng = np.random.default_rng(c.seed + 5)
        self.fl_pos = np.zeros((c.n_flakes, 3))
        self.fl_v = np.zeros(c.n_flakes)
        self.fl_ph = self._frng.uniform(0, math.tau, c.n_flakes)
        self.fl_on = np.zeros(c.n_flakes, bool)
        for j in range(c.n_flakes):
            self._respawn_flake(j, high=False)
        self._dmp = 1.0
        # counters
        self.area_cleared = 0.0  # mm^2 of snow-covered driveway scraped
        self.volume = 0.0  # mm^3 shovelled
        self.clumps_shoveled = 0
        self.lanes = 0
        self.passes = 0
        self.dumps = 0
        self.clumps_recycled = 0
        self.clumps_lost = 0
        self.load = 0.0
        self.lane = 1  # index of the lane being worked
        self.mode = "scrape"  # scrape (blade down, +y) / return (blade up, to the next start)
        self._yaw = math.pi / 2
        self._t_snow = -1.0
        self._last_shovel = np.array(self.shovel_start, float)
        self._caption = ("", -1e9)
        self._day = 0
        self.message = ""
        pad = 2.3

        def clamp(p):
            return np.array([min(max(float(p[0]), c.path_x0 - pad), c.path_x1 + pad),
                             min(max(float(p[1]), c.path_y0 - pad), c.path_y1 + 1.6)])

        self.pilot = PushPilot(self, c.brace_radius, behind_gap=c.behind_gap, orbit_clearance=c.orbit_clearance,
                               push_speed=c.push_speed, approach_speed=c.approach_speed, clamp=clamp)
        self.sim.data.qpos[self.qyaw] = self._yaw
        self.state = "approach"

    def on_reset(self) -> None:
        self.pilot.reset()
        self._t_snow = -1.0

    def reset_props(self) -> None:
        """The keyframe reset put the shovel back at its spawn pose and every clump at
        its spec pose: put the shovel back where it was, the clumps back in the pool
        or on the bank (asleep)."""
        c = self.cfg
        d = self.sim.data
        last = self._last_shovel
        x0, y0 = self.shovel_start
        if np.all(np.isfinite(last)) and np.linalg.norm(last) > c.brace_radius + 3.0:
            d.qpos[self.qx], d.qpos[self.qy] = last[0] - x0, last[1] - y0
        d.qpos[self.qyaw] = self._yaw
        d.qvel[[self.vx, self.vy, self.vyaw]] = 0.0
        for j in range(c.n_clumps):
            if self.c_state[j] in (REST, SLEEP) and hasattr(self, "_c_rest"):
                self._place_clump(j, self._c_rest[j])
                self._sleep_clump(j)
            else:
                self._park_clump(j)
        mj.mj_forward(self.sim.model, d)

    # ------------------------------------------------------------ snow
    def intensity(self, t: float) -> float:
        c = self.cfg
        return 1.0 + 0.65 * math.sin(2 * math.pi * t / c.storm_period_s - 0.8)

    def _write_tiles(self, sel: np.ndarray) -> None:
        m = self.sim.model
        c = self.cfg
        g = self.tile_geoms[sel]
        dd = self.depth[sel]
        h = np.maximum(dd, 0.004)
        m.geom_size[g, 2] = h
        m.geom_pos[g, 2] = 0.04
        m.geom_aabb[g, 5] = h
        r = c.tile * 0.74
        m.geom_rbound[g] = math.sqrt(2 * r * r) + h
        a = np.clip((dd - 0.008) / 0.03, 0.0, 1.0)
        fresh = self.tint[sel]
        rgba = np.column_stack([0.93 * fresh + 0.05 * (dd / c.max_depth), 0.95 * fresh + 0.04 * (dd / c.max_depth),
                                np.ones_like(dd), a])
        m.geom_rgba[g] = np.clip(rgba, 0, 1)

    def update_snow(self, dt: float, shovel_xy, yaw: float, scraping: bool) -> None:
        c = self.cfg
        rate = c.snow_rate * self.intensity(self.run_time())
        grow = self.depth < c.max_depth * self.drift
        if dt > 0:
            self.depth[grow] = np.minimum(self.depth[grow] + rate * dt * self.drift[grow],
                                          c.max_depth * self.drift[grow])
        cut = np.zeros(self.n_tiles, bool)
        if scraping:
            rel = self.tile_xy - shovel_xy
            ca, sa = math.cos(yaw), math.sin(yaw)
            fwd = rel[:, 0] * ca + rel[:, 1] * sa
            side = -rel[:, 0] * sa + rel[:, 1] * ca
            cut = (fwd > 0.2) & (fwd < 1.35) & (np.abs(side) < c.blade_half_w) & (self.depth > 0.004)
            if np.any(cut):
                vol = float(np.sum(self.depth[cut])) * self.tile_area * 0.6  # dome volume ~ 0.6 x area x h
                snowy = self.depth[cut] > 0.03
                area = float(np.count_nonzero(snowy)) * self.tile_area
                self.area_cleared += area
                self.add_work(area / c.blade_half_w / 2 * 1e-3)
                self.work = self.area_cleared / (2 * c.blade_half_w) * 1e-3
                self.load += vol
                self.volume += vol
                self.depth[cut] = 0.0
        self._write_tiles(grow | cut)

    def fraction_clear(self) -> float:
        return float(np.mean(self.depth < 0.03))

    # ------------------------------------------------------------ clumps / bank
    def _clump_contacts(self, j: int, on: bool) -> None:
        m = self.sim.model
        g = self.c_geom[j]
        m.geom_contype[g] = CLUMP_BIT if on else 0
        m.geom_conaffinity[g] = (CLUMP_BIT | TERRAIN_BIT) if on else 0
        m.body_gravcomp[self.c_bid[j]] = 0.0 if on else 1.0

    def _place_clump(self, j: int, p, vel=None) -> None:
        d = self.sim.data
        q, v = self.c_q[j], self.c_v[j]
        d.qpos[q:q + 3] = p
        d.qpos[q + 3:q + 7] = (1, 0, 0, 0)
        d.qvel[v:v + 6] = 0.0
        if vel is not None:
            d.qvel[v:v + 3] = vel

    def _park_clump(self, j: int) -> None:
        self._place_clump(j, self.c_park[j])
        self._clump_contacts(j, False)
        self.c_state[j] = PARK

    def _sleep_clump(self, j: int) -> None:
        self._clump_contacts(j, False)
        self.sim.data.qvel[self.c_v[j]:self.c_v[j] + 6] = 0.0
        self.c_state[j] = SLEEP

    def _free_clumps(self, n: int) -> np.ndarray:
        free = np.flatnonzero(self.c_state == PARK)
        if len(free) < n:  # pack the oldest resting clumps into the bank
            old = np.flatnonzero(np.isin(self.c_state, (FLY, REST, SLEEP)))
            old = old[np.argsort(self.c_t[old])][:n - len(free)]
            for j in old:
                self._park_clump(j)
                self.clumps_recycled += 1
            free = np.flatnonzero(self.c_state == PARK)
        return free[:n]

    def dump(self, shovel_xy, yaw: float) -> int:
        """Throw the load off the blade onto the bank: pooled clumps (engineered throw
        velocity, real physics after), the bank segment grows by the volume."""
        c = self.cfg
        load = self.load
        self.load = 0.0
        n = int(np.clip(round(load / c.clump_volume), 1 if load > 0.05 else 0, c.max_clumps_per_dump))
        if load > 0:
            j = int(np.argmin(np.abs(self.bank_x() - shovel_xy[0])))
            self.bank_vol[j] += load * 0.6
            self.bank_vol[max(j - 1, 0)] += load * 0.2
            self.bank_vol[min(j + 1, c.bank_segments - 1)] += load * 0.2
            self._write_bank()
        if n == 0:
            return 0
        fwd = np.array([math.cos(yaw), math.sin(yaw)])
        side = np.array([-fwd[1], fwd[0]])
        rng = self._frng
        # wake resting clumps near the landing zone so the new ones pile on them
        near = np.flatnonzero((self.c_state == SLEEP))
        for j in near:
            p = self.sim.data.qpos[self.c_q[j]:self.c_q[j] + 3]
            if abs(p[0] - shovel_xy[0]) < 3.0:
                self._clump_contacts(j, True)
                self.c_state[j] = REST
                self.c_slow[j] = math.inf
        for i, j in enumerate(self._free_clumps(n)):
            off = rng.uniform(-1.1, 1.1)
            p0 = np.array([*(shovel_xy + fwd * 1.3 + side * off), 0.45 + 0.1 * i])
            v = np.array([*(fwd * rng.uniform(18, 32) + side * rng.normal(0, 4)), rng.uniform(22, 34)])
            self._place_clump(j, p0, v)
            self._clump_contacts(j, True)
            self.c_state[j] = FLY
            self.c_t[j] = self.sim.time
            self.c_slow[j] = math.inf
            self.clumps_shoveled += 1
        self.dumps += 1
        return n

    def _clumps_step(self) -> None:
        d = self.sim.data
        t = self.sim.time
        live = np.flatnonzero(np.isin(self.c_state, (FLY, REST)))
        for j in live:
            q, v = self.c_q[j], self.c_v[j]
            p = d.qpos[q:q + 3]
            if not np.all(np.isfinite(d.qpos[q:q + 7])) or p[2] < -0.5 or abs(p[1]) > 40 or abs(p[0]) > 60:
                self._park_clump(j)
                self.clumps_lost += 1
                continue
            speed = float(np.linalg.norm(d.qvel[v:v + 3]))
            if self.c_state[j] == FLY and t - self.c_t[j] > 0.05 and speed < 5.0:
                self.c_state[j] = REST
            if self.c_state[j] == REST or t - self.c_t[j] > 2.0:
                if speed < 1.0:
                    if not math.isfinite(self.c_slow[j]):
                        self.c_slow[j] = t
                    elif t - self.c_slow[j] > 0.3:
                        if not hasattr(self, "_c_rest"):
                            self._c_rest = np.zeros((self.cfg.n_clumps, 3))
                        self._c_rest[j] = p.copy()
                        self._sleep_clump(j)
                else:
                    self.c_slow[j] = math.inf
                if t - self.c_t[j] > 4.0 and self.c_state[j] != SLEEP:
                    self._c_rest = getattr(self, "_c_rest", np.zeros((self.cfg.n_clumps, 3)))
                    self._c_rest[j] = p.copy()
                    self._sleep_clump(j)

    def bank_heights(self) -> np.ndarray:
        c = self.cfg
        return np.minimum(0.2 + 0.8 * np.sqrt(np.maximum(self.bank_vol, 0.0)), c.bank_max_h)

    def bank_height(self) -> float:
        return float(np.max(self.bank_heights()))

    def _write_bank(self) -> None:
        m = self.sim.model
        h = self.bank_heights()
        g = self.bank_geoms
        m.geom_size[g, 2] = h
        m.geom_size[g, 1] = 1.3 + 0.35 * h
        m.geom_aabb[g, 5] = h
        m.geom_aabb[g, 4] = 1.3 + 0.35 * h
        m.geom_rbound[g] = np.sqrt(1.4 ** 2 + (1.3 + 0.35 * h) ** 2 + h ** 2)

    # ------------------------------------------------------------ flakes
    def _respawn_flake(self, j: int, high: bool = True) -> None:
        rng = self._frng
        c = self.cfg
        x = rng.uniform(c.path_x0 - 5, c.path_x1 + 5)
        y = rng.uniform(c.path_y0 - 7, c.bank_y + 3)
        z = rng.uniform(7, 11) if high else rng.uniform(0.2, 11)
        self.fl_pos[j] = (x, y, z)
        self.fl_v[j] = rng.uniform(2.2, 3.6)

    def _flakes_step(self, dt: float) -> None:
        if dt <= 0:
            return
        d = self.sim.data
        t = self.sim.time
        # the number of flakes in the air follows the weather
        n_on = int(round(self.cfg.n_flakes * min(1.0, 0.25 + 0.55 * self.intensity(self.run_time()))))
        self.fl_on[:] = False
        self.fl_on[:n_on] = True
        P_ = self.fl_pos
        P_[:, 2] -= self.fl_v * dt
        sway = 0.6 * np.sin(1.7 * t + self.fl_ph)
        landed = np.flatnonzero(P_[:, 2] < 0.05)
        for j in landed:
            self._respawn_flake(j)
        out = P_.copy()
        out[:, 0] += sway
        out[:, 1] += 0.4 * np.cos(1.3 * t + self.fl_ph)
        out[~self.fl_on, 2] = -5.0
        d.mocap_pos[self.fl_mocap] = out
        q = np.zeros((len(out), 4))
        a = 0.5 * (t * 2.0 + self.fl_ph)
        q[:, 0] = np.cos(a)
        q[:, 1] = np.sin(a) * 0.6
        q[:, 3] = np.sin(a) * 0.8
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        d.mocap_quat[self.fl_mocap] = q

    # ------------------------------------------------------------ behaviour
    def shovel_xy(self) -> np.ndarray:
        return self.sim.data.xpos[self.shovel_body, :2].copy()

    def goal(self, s: np.ndarray) -> np.ndarray:
        c = self.cfg
        x = self.lane_x(self.lane)
        if self.mode == "scrape":
            gy = min(s[1] + c.pursuit, self.y_end + 0.6)
            return np.array([x, gy])
        return np.array([self.lane_x(self.lane + 1), self.y_start])

    def update(self) -> None:
        c = self.cfg
        sim = self.sim
        d = sim.data
        t = sim.time
        s = self.shovel_xy()
        if not np.all(np.isfinite(s)):
            return
        self._last_shovel = s
        if self._t_snow < 0 or t < self._t_snow:
            self._t_snow = t
        dt = t - self._t_snow
        if dt >= 0.02:
            self.update_snow(dt, s, self._yaw, self.mode == "scrape" and self.pilot.pushing(s))
            self._flakes_step(dt)
            self._clumps_step()
            if dt > 0:
                self.bank_vol *= math.exp(-dt / c.bank_settle_s)
                self._write_bank()
            self._write_load()
            self._t_snow = t
            day = int(self.run_time() // c.day_s)
            if day != self._day:
                self._day = day
                self._say_cap(f"DAY {day + 1} OF WINTER  (still snowing)")
        # heavier with snow on the blade
        k = 1.0 + min(self.load / c.load_ref, 1.0)
        m = sim.model
        m.dof_damping[self.vx] = c.shovel_damping * k
        m.dof_damping[self.vy] = c.shovel_damping * k
        if self.session.actions.busy:
            self.steering.set(None, 1.0)
            return
        g = self.goal(s)
        self.state = f"{self.mode}:{self.pilot.step(s, g)}"
        psi = math.atan2(g[1] - s[1], g[0] - s[0])
        dy = wrap_angle(psi - self._yaw)
        step = c.yaw_rate * c.update_every_steps * m.opt.timestep
        self._yaw = wrap_angle(self._yaw + min(max(dy, -step), step))
        d.qpos[self.qyaw] = self._yaw
        d.qvel[self.vyaw] = 0.0
        x = self.lane_x(self.lane)
        if self.mode == "scrape":
            if s[1] >= self.y_end - 0.15 and abs(s[0] - x) < 1.0:
                n = self.dump(s, self._yaw)
                self.lanes += 1
                self.mode = "return"
                self.message = f"lane {self.lane % self.n_lanes + 1} cleared, {n} clumps onto the bank"
                self.pilot.reset()
        else:
            tgt = np.array([self.lane_x(self.lane + 1), self.y_start])
            if float(np.linalg.norm(s - tgt)) < 0.6:
                self.lane += 1
                self.mode = "scrape"
                self.pilot.reset()
                if self.lane % self.n_lanes == 0:
                    self.passes += 1
                    self._say_cap(f"DRIVEWAY DONE (pass {self.passes})... and it's snowing again")
                    if c.celebrate_s > 0:
                        from fly_simulator.jobs.base import _make_action

                        self.session.actions.trigger(_make_action("groom", duration=c.celebrate_s), source="job")

    def _write_load(self) -> None:
        m = self.sim.model
        g = self.load_geom
        L = self.load
        if L <= 0.02:
            m.geom_rgba[g, 3] = 0.0
            return
        h = min(0.08 + 0.12 * math.sqrt(L), 0.75)
        m.geom_size[g] = (min(0.2 + 0.1 * math.sqrt(L), 0.55), self.cfg.blade_half_w * 0.85, h)
        m.geom_pos[g] = (1.3 + 0.5 * m.geom_size[g, 0], 0.0, 0.03)
        m.geom_rgba[g, 3] = 1.0

    def _say_cap(self, text: str) -> None:
        self.message = text
        self._caption = (text, self.run_time())
        self.say(text)

    # ------------------------------------------------------------ view / HUD
    def post_process(self, frame: np.ndarray, t: float) -> np.ndarray:
        """A caption overlay (DAY n OF WINTER, DRIVEWAY DONE)."""
        cap, t0 = self._caption
        if not (self.cfg.captions and cap and 0 <= t - t0 < 2.0):
            return frame
        import cv2

        H, W = frame.shape[:2]
        out = np.ascontiguousarray(frame).copy()
        fs = max(0.4, W / 1400.0) * 1.6
        th = max(1, int(round(W / 700))) * 2 + 1
        (tw, tht), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_DUPLEX, fs, th)
        x, y = (W - tw) // 2, int(0.12 * H) + tht
        cv2.putText(out, cap, (x + 2, y + 2), cv2.FONT_HERSHEY_DUPLEX, fs, (20, 30, 60), th + 2, cv2.LINE_AA)
        cv2.putText(out, cap, (x, y), cv2.FONT_HERSHEY_DUPLEX, fs, (235, 245, 255), th, cv2.LINE_AA)
        return out

    def camera_target(self) -> np.ndarray:
        f = self.sim.thorax_position()
        s = np.append(self.shovel_xy(), 0.6)
        c = self.cfg
        mid = np.array([(c.path_x0 + c.path_x1) / 2, 1.5, 0.0])
        out = 0.35 * f + 0.3 * s + 0.35 * mid
        return out if np.all(np.isfinite(out)) else f

    def camera_preset(self) -> CameraPreset:
        return CameraPreset(azimuth=80.0, elevation=-30.0, distance=22.0, tau_s=1.0)

    def job_hud_lines(self) -> list[str]:
        h = self.bank_height()
        return [
            f"clumps shoveled {self.clumps_shoveled}   snow moved {self.volume:.1f} mm^3   "
            f"bank height {h:.2f} mm (~{h * HUMAN / 1000:.2f} m human scale)",
            f"day {self._day + 1} of winter   lane {self.lane % self.n_lanes + 1}/{self.n_lanes} [{self.mode}]   "
            f"lanes {self.lanes}  passes {self.passes}   driveway clear {100 * self.fraction_clear():.0f}%   "
            f"snowfall x{self.intensity(self.run_time()):.2f}",
            "(dump throw set by the job, clumps real physics; blade up on the way back)",
        ]

    def job_stats(self) -> dict:
        return {"path_cleared_m": self.work, "area_cleared_mm2": self.area_cleared,
                "clumps_shoveled": self.clumps_shoveled, "volume_mm3": self.volume,
                "bank_height_mm": self.bank_height(), "days_of_winter": self._day + 1,
                "lanes": self.lanes, "passes": self.passes, "dumps": self.dumps,
                "clumps_recycled": self.clumps_recycled, "clumps_lost": self.clumps_lost,
                "driveway_clear_frac": self.fraction_clear(), "pushes_started": self.pilot.n_pushes,
                "unstuck": self.n_unstuck}
