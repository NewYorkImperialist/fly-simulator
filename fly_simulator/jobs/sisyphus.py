"""Sisyphus: the fly pushes a boulder up a hill; at the summit it lets go, the boulder
rolls back down into the valley, the fly walks down after it, gets behind it and
pushes again. Forever.

World (along +x = uphill; the fly spawns at the origin facing +x):

* valley floor (the ground plane, painted as a meadow) from ``floor_x0`` to
  ``ramp_x0``, closed at the back by a counter-slope that stops the boulder;
* the hill: a ramp rising at ``slope_deg`` from ``ramp_x0`` over ``ramp_run`` mm to a
  summit curb (a stone lip the boulder can't pass) with a flag;
* low side walls along the whole trough so neither the boulder nor the fly leaves.

The boulder is a real free body (sphere, ``ball_mass``) that rolls with friction;
its contacts with the ground / hill are condim 6 with rolling friction, so it rolls
back down the 10 deg slope but comes to rest ~10 mm into the valley.

Behaviour (``update``, every 1 ms): APPROACH (walk to the point behind the boulder,
going around it if the fly is beside / in front of it) -> PUSH (aim at the boulder
centre with an over-steer term that keeps the fly on the goal line, so the boulder
goes straight up) -> summit (boulder at the curb: counted) -> RELEASE (the fly turns
aside and steps out of the way; the boulder rolls down) -> DESCEND (walk down beside
the boulder's path until it has come to rest) -> APPROACH ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from fly_simulator.jobs import sisyphus_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import (
    add_box,
    add_plane_box,
    add_slope,
    contact_kwargs,
    quat_axis_angle,
    slippery_body_contact,
    wrap_angle,
)
from fly_simulator.jobs.registry import register_job
from fly_simulator.terrain import TERRAIN_BIT

P = "sisyphus/"

STONE = (0.56, 0.55, 0.52, 1.0)


@dataclass
class SisyphusConfig(JobConfig):
    # ---- boulder ---------------------------------------------------------------
    ball_radius: float = 2.0  # mm (4 mm boulder; the fly is ~2.5 mm long)
    ball_mass: float = 3e-4  # g (0.3 mg, a pumice boulder; the fly is 1.02 mg)
    ball_damping: float = 0.0  # free-joint damping (all 6 DoFs; strongly nonlinear, avoid)
    # linear air-drag-like force -ball_drag * v (uN per mm/s, applied by the job
    # through xfrc_applied every update): caps the roll-back speed at
    # m g (sin(slope) - rolling / R) / drag; it reaches ~75 mm/s on the way down
    ball_drag: float = 0.0
    # rolling resistance (condim-6 contacts): resisting torque = rolling * normal force
    # on the hill: just below tan(slope) * R, so the boulder starts rolling back
    # slowly, gathers speed (~70 mm/s at the foot) and doesn't squirt sideways easily
    ball_rolling: float = 0.28
    mud_rolling: float = 0.3  # valley floor centre band
    ball_friction: float = 1.0
    ball_start_x: float = 4.0
    spawn_jitter: float = 0.6  # boulder spawn jitter after a reset (mm, seeded)  # boulder centre at spawn (the fly's head is at ~+1 mm)
    # the fly touches the boulder with head / thorax / abdomen only, with this low
    # friction (a rolling boulder otherwise drags the head up and flips the fly;
    # sticky tarsi would climb onto it), see geometry.slippery_body_contact
    head_friction: float = 0.05
    # ---- hill -------------------------------------------------------------------
    slope_deg: float = 10.0
    ramp_x0: float = 5.0  # foot of the hill
    ramp_run: float = 14.0  # horizontal length of the ramp (mm)
    curb_height: float = 1.2  # summit lip above the ramp top
    floor_x0: float = -12.0  # back end of the valley floor
    back_slope_deg: float = 25.0
    back_run: float = 4.0
    half_width: float = 7.0  # trough half width (inner wall faces at +-half_width)
    flat_half_width: float = 1.0  # flat centre band; outside it the banks rise
    bank_deg: float = 12.0
    wall_height: float = 1.4
    wall_friction: float = 0.1
    keep_off_wall: float = 2.2  # waypoints stay this far from the wall faces (mm)
    # ---- behaviour ---------------------------------------------------------------
    behind_gap: float = 1.25  # thorax -> boulder surface when pushing (mm)
    lookahead: float = 2.0  # approach: pursuit point this far ahead on the goal line (mm)
    approach_speed: float = 0.9
    align_radius: float = 1.0  # within this of the pushing position: turn to face the boulder
    orbit_clearance: float = 1.7  # thorax keeps this far from the boulder surface
    push_oversteer: float = 0.5  # heading = phi + k * (phi - goal)
    push_max_dev_deg: float = 25.0
    push_speed: float = 0.7
    summit_margin: float = 0.35  # boulder within this of touching the curb = summit
    release_turn_deg: float = 110.0
    release_s: float = 0.6
    settle_speed: float = 2.0  # boulder at rest below this (mm/s) ...
    settle_hold_s: float = 0.3  # ... for this long
    descend_timeout_s: float = 6.0
    # ---- looks -------------------------------------------------------------------
    shadows: bool = True  # the low sun's shadow map (~several ms per 960x640 frame)


@register_job
class SisyphusJob(EternalJob):
    name = "sisyphus"
    title = "SISYPHUS FLY"
    tagline = "one must imagine the fly happy"
    work_label = "summits"
    config_cls = SisyphusConfig
    required_names = (P + "boulder", P + "ramp")

    cfg: SisyphusConfig

    # ------------------------------------------------------------ geometry
    @property
    def x_top(self) -> float:
        c = self.cfg
        return c.ramp_x0 + c.ramp_run

    @property
    def hill_height(self) -> float:
        c = self.cfg
        return c.ramp_run * math.tan(math.radians(c.slope_deg))

    def _base_height(self, x: float) -> float:
        c = self.cfg
        if x < c.floor_x0:
            return min(c.floor_x0 - x, c.back_run) * math.tan(math.radians(c.back_slope_deg))
        if x <= c.ramp_x0:
            return 0.0
        return min(x - c.ramp_x0, c.ramp_run) * math.tan(math.radians(c.slope_deg))

    def ground_height(self, x: float, y: float) -> float:
        c = self.cfg
        h = self._base_height(x)
        if c.floor_x0 <= x <= self.x_top:  # the trough's sloping side banks
            h += max(0.0, min(abs(y), c.half_width) - c.flat_half_width) * math.tan(
                math.radians(c.bank_deg))
        return h

    def extension(self, world) -> None:
        c = self.cfg
        wb = world.mjcf_root.worldbody
        a = math.radians(c.slope_deg)
        ab = math.radians(c.back_slope_deg)
        tb = math.tan(math.radians(c.bank_deg))
        hw, w0 = c.half_width, c.flat_half_width
        h = self.hill_height
        xt = self.x_top
        x_back = c.floor_x0 - c.back_run
        h_back = c.back_run * math.tan(ab)
        bank_top = (hw - w0) * tb
        wall_hw = 0.4
        wy = hw + wall_hw
        span = hw + 2 * wall_hw
        spec = world.mjcf_root
        self._add_materials(spec)
        # valley floor centre band "mud": a thin box 4 um above the ground plane that
        # only the boulder touches (contype TERRAIN_BIT, conaffinity 0: not the fly),
        # with rolling friction, so the boulder comes to rest in the valley
        from flygym.compose import ContactParams

        cp = ContactParams()
        wb.add_geom(name=P + "mud", type=mj.mjtGeom.mjGEOM_BOX,
                    size=((c.ramp_x0 - c.floor_x0) / 2, w0 + 0.3, 0.05),
                    pos=((c.ramp_x0 + c.floor_x0) / 2, 0.0, 0.004 - 0.05), material=P + "path",
                    contype=TERRAIN_BIT, conaffinity=0, priority=2, condim=6,
                    friction=(1.0, 0.02, c.mud_rolling), solref=cp.get_solref_tuple(),
                    solimp=cp.get_solimp_tuple(), margin=cp.margin)
        # the hill: centre band of the ramp
        add_slope(wb, P + "ramp", c.ramp_x0, xt, 0.0, a, w0 + 0.05, thickness=1.5,
                  material=P + "path", collide="static")
        # side banks (valley floor and ramp): planes rising bank_deg outward, so the
        # boulder (and the fly) drift back to the centre line
        run_v = c.ramp_x0 - c.floor_x0
        run_r = c.ramp_run
        bank_w = (hw - w0) / 2
        for side, sgn in (("l", 1.0), ("r", -1.0)):
            yc = sgn * (w0 + bank_w)
            nrm_v = (0.0, -sgn * tb, 1.0)
            add_plane_box(wb, f"{P}bank_floor_{side}",
                          (c.floor_x0 + run_v / 2, yc, bank_w * tb), (1, 0, 0), nrm_v,
                          run_v / 2, bank_w / math.cos(math.atan(tb)) + 0.02,
                          thickness=1.0, material=P + "hill", collide="static")
            nrm_r = (-math.tan(a), -sgn * tb, 1.0)
            xm = c.ramp_x0 + run_r / 2
            add_plane_box(wb, f"{P}bank_ramp_{side}",
                          (xm, yc, (xm - c.ramp_x0) * math.tan(a) + bank_w * tb),
                          (1, 0, math.tan(a)), nrm_r,
                          run_r / 2 / math.cos(a) + 0.02,
                          bank_w / math.cos(math.atan(tb)) + 0.02,
                          thickness=1.0, material=P + "hill", collide="static")
        # summit curb + the hill's back side (steep, never walked on)
        top = h + bank_top + c.curb_height
        add_box(wb, P + "curb", (0.5, span, top / 2), (xt + 0.5, 0.0, top / 2),
                material=P + "marble", collide="static")
        add_slope(wb, P + "backside", xt + 1.0, xt + 1.0 + top / math.tan(1.0), top, -1.0,
                  span, thickness=1.0, material=P + "hill", collide="static")
        # counter-slope behind the valley
        add_slope(wb, P + "counter", x_back, c.floor_x0, h_back, -ab, span, thickness=1.5,
                  material=P + "hill", collide="static")
        add_box(wb, P + "counter_top", (1.0, span, h_back / 2),
                (x_back - 1.0, 0.0, h_back / 2), material=P + "hill", collide="static")
        # side walls: valley floor, ramp, counter-slope
        wall_top = bank_top + c.wall_height
        for side, sgn in (("l", 1.0), ("r", -1.0)):
            y = sgn * wy
            # slippery walls: sticky tarsi can't climb them (the fly would flip)
            wf = c.wall_friction
            add_box(wb, f"{P}wall_floor_{side}", (run_v / 2, wall_hw, wall_top / 2),
                    ((c.ramp_x0 + c.floor_x0) / 2, y, wall_top / 2),
                    material=P + "wall", collide="static", friction=wf)
            add_slope(wb, f"{P}wall_ramp_{side}", c.ramp_x0, xt, wall_top, a, wall_hw,
                      y=y, thickness=wall_top + 1.0, material=P + "wall", collide="static", friction=wf)
            add_slope(wb, f"{P}wall_counter_{side}", x_back, c.floor_x0,
                      h_back + wall_top, -ab, wall_hw, y=y,
                      thickness=wall_top + 1.0, material=P + "wall", collide="static", friction=wf)
        # the walls and the curb: invisible colliders (group 3) under textured box
        # meshes of the same size and pose (a box primitive only maps a 2D texture
        # well onto its +z face)
        for side in ("l", "r"):
            for part in ("floor", "ramp", "counter"):
                self._dress_box(spec, f"{P}wall_{part}_{side}", "wall_mesh", 0.34)
        self._dress_box(spec, P + "curb", "marble_mesh", 0.3)
        # summit flag (decoration)
        add_box(wb, P + "flagpole", (0.05, 0.05, 1.6), (xt + 0.5, hw - 0.6, top + 1.6),
                material=P + "pole", collide="visual")
        add_box(wb, P + "flag", (0.02, 0.7, 0.4), (xt + 0.5, hw - 1.3, top + 2.8),
                material=P + "flag", collide="visual")
        # the boulder
        R = c.ball_radius
        body = wb.add_body(name=P + "boulder", pos=(c.ball_start_x, 0.0, R + 0.01))
        joint = body.add_freejoint(name=P + "boulder_free")
        joint.damping = [c.ball_damping] * 3
        kw = contact_kwargs("dynamic", c.ball_friction)
        kw["condim"] = 6
        kw["friction"] = (c.ball_friction, 0.02, c.ball_rolling)
        # (the colliding sphere is drawn by the lumpy granite shell below: invisible)
        body.add_geom(name=P + "boulder_geom", type=mj.mjtGeom.mjGEOM_SPHERE, size=(R, 0, 0),
                      mass=c.ball_mass, rgba=STONE[:3] + (0.0,), **kw)
        A.add_mesh(spec, P + "boulder_mesh", A.boulder_mesh(R, c.seed))
        body.add_geom(name=P + "boulder_vis", type=mj.mjtGeom.mjGEOM_MESH,
                      meshname=P + "boulder_mesh", material=P + "boulder",
                      **dict(contact_kwargs("visual"), mass=0.0))
        self._add_scenery(spec)

        slippery_body_contact(world.mjcf_root, P + "boulder", P + "boulder_geom",
                              self.fly_name, friction=c.head_friction)

    # ------------------------------------------------------------ looks
    def _add_materials(self, spec) -> None:
        c = self.cfg
        mat = spec.material("grid")
        if mat is not None:  # the ground plane: dry hillside (8 mm texture period)
            A.add_texture(spec, P + "tex_hill", A.hillside_texture(c.seed))
            mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = P + "tex_hill"
            mat.rgba = (1.0, 1.0, 1.0, 1.0)
            mat.reflectance = 0.0
            mat.texrepeat = [v * 0.5 for v in mat.texrepeat]
        T = A.add_textured_material
        # (texuniform: MuJoCo maps a 2D texture on a box by its local x, y, per mm)
        T(spec, P + "hill", P + "tex_hill", rgba=(1, 1, 1, 1), specular=0.05, texuniform=True,
          texrepeat=(0.125, 0.125))
        A.add_texture(spec, P + "tex_path", A.path_texture(c.seed))
        T(spec, P + "wall_mesh", P + "tex_wall", rgba=(1, 1, 1, 1), specular=0.08)
        T(spec, P + "path", P + "tex_path", rgba=(1, 1, 1, 1), specular=0.05, texuniform=True,
          texrepeat=(0.3, 0.3))
        A.add_texture(spec, P + "tex_wall", A.drystone_texture(c.seed))
        T(spec, P + "wall", P + "tex_wall", rgba=(1, 1, 1, 1), specular=0.08, texuniform=True,
          texrepeat=(0.3, 0.3))
        A.add_texture(spec, P + "tex_marble", A.marble_texture(c.seed))
        T(spec, P + "marble", P + "tex_marble", rgba=(1.0, 0.96, 0.9, 1), specular=0.35,
          shininess=0.5, texuniform=True, texrepeat=(0.35, 0.35))
        # the columns (lathe uv: u around, v up)
        T(spec, P + "marble_mesh", P + "tex_marble", rgba=(0.74, 0.70, 0.64, 1), specular=0.3,
          shininess=0.5)
        T(spec, P + "column", P + "tex_marble", rgba=(1.0, 0.95, 0.88, 1), specular=0.3,
          shininess=0.5, texrepeat=(2.0, 1.0))
        A.add_texture(spec, P + "tex_boulder", A.boulder_texture(c.seed))
        T(spec, P + "boulder", P + "tex_boulder", rgba=(1, 1, 1, 1), specular=0.18, shininess=0.25)
        A.add_texture(spec, P + "tex_bark", A.bark_texture(c.seed))
        T(spec, P + "bark", P + "tex_bark", rgba=(1, 1, 1, 1), specular=0.05, texrepeat=(2.0, 3.0))
        A.add_texture(spec, P + "tex_olive", A.olive_leaf_texture(c.seed))
        T(spec, P + "olive", P + "tex_olive", rgba=(1, 1, 1, 1), specular=0.15, shininess=0.3,
          texrepeat=(3.0, 2.0))
        T(spec, P + "rock", P + "tex_boulder", rgba=(0.95, 0.9, 0.82, 1), specular=0.1,
          texrepeat=(2.0, 1.0))
        spec.add_material(name=P + "cypress", rgba=(0.12, 0.2, 0.1, 1), specular=0.05)
        A.add_texture(spec, P + "tex_sunset", A.sunset_texture(c.seed))
        T(spec, P + "sunset", P + "tex_sunset", rgba=(1, 1, 1, 1), emission=0.8, specular=0.0)
        spec.add_material(name=P + "pole", rgba=(0.62, 0.45, 0.25, 1), specular=0.2)
        spec.add_material(name=P + "flag", rgba=(0.85, 0.10, 0.08, 1), specular=0.1, emission=0.15)

    @staticmethod
    def _dress_box(spec, name: str, material: str, scale: float) -> None:
        """Hide the box collider ``name`` (render group 3, transparent; its contacts
        are unchanged) and draw a textured visual box mesh in its place."""
        g = spec.geom(name)
        g.group = 3
        g.material = ""
        g.rgba = (0.0, 0.0, 0.0, 0.0)
        A.add_mesh(spec, name + "_vis_mesh", A.box_mesh(g.size, scale))
        spec.worldbody.add_geom(name=name + "_vis", type=mj.mjtGeom.mjGEOM_MESH,
                                meshname=name + "_vis_mesh", pos=tuple(g.pos),
                                quat=tuple(g.quat), material=P + material,
                                **dict(contact_kwargs("visual"), mass=0.0))

    def _add_scenery(self, spec) -> None:
        """Set dressing (all visual): a ruined temple behind the summit, broken
        columns and drums along the trough, olive trees, cypresses and rocks, the
        sunset backdrop, and the light rig (a low warm sun with shadows)."""
        c = self.cfg
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)

        def mesh(name, md, pos, material, quat=(1.0, 0.0, 0.0, 0.0)):
            A.add_mesh(spec, P + name + "_mesh", md)
            wb.add_geom(name=P + name, type=mj.mjtGeom.mjGEOM_MESH, meshname=P + name + "_mesh",
                        pos=tuple(pos), quat=tuple(quat), material=P + material, **vis)

        xt = self.x_top
        # the temple ruin beyond the summit: a stylobate, a colonnade (two columns
        # still carry a lintel, the others are broken off) and a fallen drum
        xc = xt + 7.0
        mesh("stylobate", A.box_mesh((2.2, 7.5, 0.3), 0.3), (xc, 0.5, 0.3), "marble_mesh")
        mesh("stylobate2", A.box_mesh((2.6, 8.0, 0.15), 0.3), (xc, 0.5, 0.15), "marble_mesh")
        col_r, col_h = 0.55, 8.5
        heights = {-5.5: 0.0, -2.0: 0.0, 1.5: 0.55, 5.0: 0.3}
        for k, (y, broken) in enumerate(heights.items()):
            h = col_h if broken == 0.0 else col_h * broken
            md = A.column_mesh(col_r, h, capital=broken == 0.0, broken=0.6 if broken else 0.0,
                               seed=k)
            mesh(f"column{k}", md, (xc, y, 0.6), "column")
        mesh("lintel", A.box_mesh((0.75, 2.45, 0.45), 0.3), (xc, -3.75, 0.6 + col_h + 0.45),
             "marble_mesh")
        mesh("drum0", A.drum_mesh(col_r * 0.95, 1.4), (xc - 3.2, 5.6, col_r * 0.95), "column",
             quat_axis_angle((0, 0, 1), 0.6))
        # broken columns and fallen drums along the far side of the trough
        for k, (x, y, h, br) in enumerate(((-6.5, 9.4, 3.2, 0.7), (9.5, 10.2, 4.6, 0.5))):
            mesh(f"ruin{k}", A.column_mesh(0.5, h, capital=False, broken=br, seed=10 + k),
                 (x, y, 0.0), "column")
        for k, (x, y, yaw) in enumerate(((-2.0, 10.0, 0.3), (15.5, 9.6, -0.8), (-12.0, -9.6, 1.2))):
            mesh(f"drum{k + 1}", A.drum_mesh(0.48, 1.2), (x, y, 0.48), "column",
                 quat_axis_angle((0, 0, 1), yaw))
        # olive trees behind the far wall
        for k, (x, y, H) in enumerate(((-10.0, 12.5, 7.0), (2.5, 14.5, 8.5), (13.5, 12.0, 6.5),
                                       (-20.0, 10.5, 6.0), (22.0, 13.5, 7.5))):
            parts = A.olive_tree_meshes(H, seed=k)
            mesh(f"olive{k}_trunk", parts["trunk"], (x, y, 0.0), "bark")
            mesh(f"olive{k}_canopy", parts["canopy"], (x, y, 0.0), "olive")
        for k, (x, y, H) in enumerate(((-15.0, 17.0, 9.0), (-13.5, 18.5, 11.0), (30.0, 16.0, 10.0))):
            mesh(f"cypress{k}", A.cypress_mesh(H, 0.9), (x, y, 0.0), "cypress")
        # rocks on both sides
        rng = np.random.default_rng(c.seed + 5)
        for k in range(12):
            side = 1.0 if k % 2 else -1.0
            x = rng.uniform(c.floor_x0 - 2, xt + 4)
            y = side * rng.uniform(c.half_width + 1.3, c.half_width + 5.0)
            r = rng.uniform(0.35, 1.0)
            mesh(f"rock{k}", A.rock_mesh(r, seed=k), (x, y, 0.0), "rock",
                 quat_axis_angle((0, 0, 1), rng.uniform(0, 6.3)))
        # the sunset backdrop far behind the hill (faces -y, toward the camera)
        from fly_simulator.jobs.taste_tester_assets import front_panel_mesh

        mesh("backdrop", front_panel_mesh(62.0, 12.0, 0.05), (20.0, 27.0, 9.5), "sunset",
             quat_axis_angle((0, 0, 1), -math.pi / 2))
        sky = spec.texture("skybox")
        if sky is not None:
            sky.rgb1 = (0.95, 0.62, 0.42)
            sky.rgb2 = (0.35, 0.25, 0.45)
        # lights: a low warm sun from the left (behind), a cool fill from the camera side
        spec.visual.headlight.ambient = (0.30, 0.27, 0.28)
        spec.visual.headlight.diffuse = (0.30, 0.28, 0.28)
        spec.visual.headlight.specular = (0.05, 0.05, 0.05)
        tgt = np.array([6.0, 0.0, 1.0])
        sun = np.array([-22.0, 14.0, 20.0])
        wb.add_light(name=P + "sun", type=mj.mjtLightType.mjLIGHT_SPOT, pos=tuple(sun),
                     dir=tuple(tgt - sun), diffuse=(0.78, 0.55, 0.36), specular=(0.35, 0.28, 0.2),
                     cutoff=42.0, exponent=0.3, castshadow=bool(c.shadows))
        fill = np.array([8.0, -22.0, 14.0])
        wb.add_light(name=P + "fill", type=mj.mjtLightType.mjLIGHT_SPOT, pos=tuple(fill),
                     dir=tuple(tgt - fill), diffuse=(0.2, 0.2, 0.3), specular=(0.05, 0.05, 0.08),
                     cutoff=55.0, exponent=0.5, castshadow=False)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        m = self.sim.model
        # the fly's MuJoCo globals (merged in after the job's spec edits) set znear to
        # 0.5 um: far too little depth precision for this scene tens of mm away
        # (z-fighting); 10 um still allows close-ups of the fly
        m.vis.map.znear = 0.01
        self.ball_body = m.body(P + "boulder").id
        j = m.joint(P + "boulder_free").id
        self.ball_qadr = int(m.jnt_qposadr[j])
        self.ball_vadr = int(m.jnt_dofadr[j])
        m.dof_damping[self.ball_vadr:self.ball_vadr + 6] = self.cfg.ball_damping
        self.summits = 0
        self.metres_pushed = 0.0
        self.boulders_lost = 0
        self.best_height = 0.0  # highest boulder climb of the current attempt (mm)
        self.record_height = 0.0
        self.n_pushes = 0
        self._rng = np.random.default_rng(self.cfg.seed)
        self._reset_behaviour()

    def _reset_behaviour(self) -> None:
        self.state = "approach"
        self._t_state = self.sim.time
        self._side = 1.0
        self._settle_since: float | None = None
        self._orbit_dir = 0.0
        self._last_ball = self.ball_xy().copy()
        self.best_height = 0.0

    def on_reset(self) -> None:
        self._reset_behaviour()

    # ------------------------------------------------------------ state
    def ball_pos(self) -> np.ndarray:
        return self.sim.data.xpos[self.ball_body]

    def ball_xy(self) -> np.ndarray:
        return self.sim.data.xpos[self.ball_body, :2]

    def ball_speed(self) -> float:
        v = self.sim.data.qvel[self.ball_vadr:self.ball_vadr + 3]
        return float(np.hypot(v[0], v[1]))

    def ball_lost(self) -> bool:
        c = self.cfg
        p = self.ball_pos()
        if not np.all(np.isfinite(p)):
            return True
        return (p[2] < -1.0 or abs(p[1]) > c.half_width + 2.0
                or p[0] < c.floor_x0 - c.back_run - 3.0 or p[0] > self.x_top + 2.0)

    def reset_props(self) -> None:
        """The keyframe reset put the boulder back at its spawn pose; jitter it a
        little (seeded) so an eternal run doesn't replay the same failure forever."""
        c = self.cfg
        if self.n_resets > 0 and c.spawn_jitter > 0:
            j = self._rng.uniform(-c.spawn_jitter, c.spawn_jitter, 2)
            d = self.sim.data
            d.qpos[self.ball_qadr:self.ball_qadr + 2] = (c.ball_start_x + abs(j[0]), j[1])
            d.qvel[self.ball_vadr:self.ball_vadr + 6] = 0.0
            mj.mj_forward(self.sim.model, d)
        self._last_ball = self.ball_xy().copy()

    def respawn_boulder(self) -> None:
        """A new boulder drops from the sky onto the valley floor, clear of the fly."""
        c = self.cfg
        fx = float(self.fly_xy()[0])
        x = min(max(fx + 4.0, c.floor_x0 + 2.5), c.ramp_x0 - 0.5)
        if abs(x - fx) < c.ball_radius + 2.5:
            x = fx - 4.0 if fx - 4.0 > c.floor_x0 + 2.0 else fx + 4.0
        d = self.sim.data
        d.qpos[self.ball_qadr:self.ball_qadr + 7] = (x, 0.0, c.ball_radius + 3.0, 1, 0, 0, 0)
        d.qvel[self.ball_vadr:self.ball_vadr + 6] = 0.0
        self.boulders_lost += 1
        self._last_ball = np.array([x, 0.0])
        self._goto("approach")

    def apply_drag(self) -> None:
        """Rolling-resistance stand-in: F = -ball_drag * v on the boulder."""
        d = self.sim.data
        v = d.qvel[self.ball_vadr:self.ball_vadr + 3]
        if np.all(np.isfinite(v)):
            d.xfrc_applied[self.ball_body, :3] = -self.cfg.ball_drag * v

    def _goto(self, state: str) -> None:
        self._orbit_dir = 0.0
        self.state = state
        self._t_state = self.sim.time
        self._settle_since = None

    # ------------------------------------------------------------ behaviour
    def update(self) -> None:
        c = self.cfg
        sim = self.sim
        if self.ball_lost():
            self.respawn_boulder()
            return
        self.apply_drag()
        R = c.ball_radius
        b = self.ball_xy().copy()
        p = self.fly_xy()
        t = sim.time
        goal = np.array([self.x_top + 1.0, 0.0])
        g = goal - b
        g /= max(np.linalg.norm(g), 1e-9)
        n = np.array([-g[1], g[0]])
        rel = p - b
        along, lateral = float(rel @ g), float(rel @ n)
        dist = float(np.linalg.norm(rel))
        step = float(np.linalg.norm(b - self._last_ball))
        self._last_ball = b
        steer = self.steering
        behind_d = R + c.behind_gap
        if self.state in ("approach", "align", "descend") and self.unstick():
            pass
        if self.state == "approach":
            cone = -along - R + 0.5  # behind the boulder: inside a 90 deg cone
            if along > -(R + 0.6) or abs(lateral) > cone:
                # beside / in front of the boulder: go round it (orbit) to its back
                far = b - g * (behind_d + 2.5)
                wp = self._orbit_waypoint(p, b, far, R + c.orbit_clearance)
                steer.aim_at(self._clamp(wp), 1.0)
            else:
                self._orbit_dir = 0.0
                # behind it: pure pursuit of a point on the goal line a lookahead
                # ahead of the fly's projection, up to the pushing position
                s_aim = min(along + c.lookahead, -behind_d)
                steer.aim_at(b + g * s_aim, c.approach_speed)
                if float(np.linalg.norm(p - (b - g * behind_d))) < c.align_radius:
                    self._goto("align")
        elif self.state == "align":
            # at the pushing position: face the boulder (on the spot), then push
            phi = math.atan2(-rel[1], -rel[0])
            steer.set(phi, 0.5)
            err = abs(wrap_angle(sim.heading() - phi))
            if err < math.radians(25):
                self.n_pushes += 1
                self._goto("push")
            elif (along > -(R + 0.3) or abs(lateral) > R * 0.8
                  or dist > behind_d + c.align_radius + 1.0):
                self._goto("approach")
        elif self.state == "push":
            phi = math.atan2(-rel[1], -rel[0])  # fly -> boulder
            psi = math.atan2(g[1], g[0])
            dev = min(max(c.push_oversteer * wrap_angle(phi - psi),
                          -math.radians(c.push_max_dev_deg)), math.radians(c.push_max_dev_deg))
            steer.set(phi + dev, c.push_speed)
            if dist < behind_d + 0.8:
                self.metres_pushed += step * 1e-3
            self.best_height = max(self.best_height, self.ground_height(b[0], b[1]))
            if b[0] >= self.x_top - R - c.summit_margin:
                self.summits += 1
                self.add_work(1)
                self.record_height = max(self.record_height, self.best_height)
                self.say(f"SUMMIT #{self.summits} (boulder at x={b[0]:.1f} mm, "
                         f"{self.metres_pushed:.3f} m pushed in total)")
                self._side = 1.0 if lateral >= 0 else -1.0
                if abs(lateral) < 0.2:
                    self._side = -1.0 if p[1] > 0 else 1.0
                self._goto("release")
            elif along > -R * 0.6 or abs(lateral) > R + 0.6 or dist > behind_d + 5.0:
                self._goto("approach")
        elif self.state == "release":
            psi = math.atan2(g[1], g[0])
            steer.set(psi + self._side * math.radians(c.release_turn_deg), 1.0)
            if t - self._t_state >= c.release_s:
                self._goto("descend")
        elif self.state == "descend":
            # walk downhill in a lane beside the boulder's path, until it rests
            lane_y = self._side * min(R + 1.8, c.half_width - 1.2)
            steer.aim_at(self._clamp((p[0] - 4.0, lane_y)), 1.0)
            if self.ball_speed() < c.settle_speed and b[0] < self.x_top - R - 1.0:
                if self._settle_since is None:
                    self._settle_since = t
                if t - self._settle_since >= c.settle_hold_s:
                    self._goto("approach")
            else:
                self._settle_since = None
            if t - self._t_state > c.descend_timeout_s:
                self._goto("approach")
            if p[0] < b[0] - 0.5:  # already below the boulder: get behind it
                self._goto("approach")
            self.best_height = 0.0

    def _clamp(self, wp) -> np.ndarray:
        lim = self.cfg.half_width - self.cfg.keep_off_wall
        return np.array([float(wp[0]), min(max(float(wp[1]), -lim), lim)])

    def _orbit_waypoint(self, p: np.ndarray, C: np.ndarray, T: np.ndarray, r: float):
        """``T`` if the straight path to it clears the circle (C, r), else a point
        50 deg further round the circle (the way that doesn't scrape a side wall)."""
        d = T - p
        L = float(np.linalg.norm(d))
        dp = float(np.linalg.norm(p - C))
        if L < 1e-6:
            return T
        u = d / L
        s = min(max(float((C - p) @ u), 0.0), L)
        if dp >= r and float(np.linalg.norm(C - (p + u * s))) >= r:
            return T
        th_p = math.atan2(p[1] - C[1], p[0] - C[0])
        dth = wrap_angle(math.atan2(T[1] - C[1], T[0] - C[0]) - th_p)
        if self._orbit_dir == 0.0:  # choose a way round once (hysteresis)
            wall = self.cfg.half_width - self.cfg.keep_off_wall
            if abs(C[1] + r * math.sin(th_p + dth / 2)) > wall:
                dth -= math.copysign(2 * math.pi, dth)  # go round the other side
            self._orbit_dir = 1.0 if dth >= 0 else -1.0
        elif dth * self._orbit_dir < 0:
            dth += self._orbit_dir * 2 * math.pi
        th = th_p + math.copysign(min(abs(dth), math.radians(50)), dth)
        rr = max(r + 0.3, dp) if dp > r else r + 0.6
        return np.array([C[0] + rr * math.cos(th), C[1] + rr * math.sin(th)])

    # ------------------------------------------------------------ view / HUD
    def camera_target(self) -> np.ndarray:
        f = self.sim.thorax_position()
        b = self.ball_pos()
        c = self.cfg
        hill = np.array([c.ramp_x0 + 0.4 * c.ramp_run, 0.0, 0.5 * self.hill_height])
        out = 0.45 * f + 0.35 * b + 0.2 * hill  # the fly, its boulder and the hill in frame
        return out if np.all(np.isfinite(out)) else f

    def camera_preset(self) -> CameraPreset:
        # from the fly's right side, slightly downhill, looking across the trough
        return CameraPreset(azimuth=68.0, elevation=-26.0, distance=19.0, tau_s=0.8)

    def job_hud_lines(self) -> list[str]:
        b = self.ball_pos()
        h = self.ground_height(float(b[0]), float(b[1]))
        return [f"boulder pushed {self.metres_pushed:.3f} m   height {h:.1f}/"
                f"{self.hill_height:.1f} mm   boulders lost {self.boulders_lost}"]

    def job_stats(self) -> dict:
        return {"summits": self.summits, "metres_pushed": self.metres_pushed,
                "boulders_lost": self.boulders_lost, "pushes_started": self.n_pushes,
                "unstuck": self.n_unstuck,
                "hill_height_mm": self.hill_height}
