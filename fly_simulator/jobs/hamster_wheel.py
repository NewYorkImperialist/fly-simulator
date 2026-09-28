"""Hamster wheel: the fly runs inside a big wheel (hinge joint) and spins it with its
own legs. Forever.

The wheel (axle along world y, centred above the origin so the fly spawns standing
on the inside bottom) is a ring of ``n_slats`` box slats in alternating bright
colours (rotation is easy to see), with slippery inner lips on both sides that keep
the fly on the running surface, a decorative back disc with spokes, an axle and a
stand. Only the fly touches the wheel (contact kind "fly"). The hinge has a little
damping; the wheel turns only because the fly's feet push the slats backward.

Behaviour: heading hold along the wheel tangent (+x) with a small correction toward
the centre line (y = 0). Counters: revolutions, distance run (surface travel), top
speed. The fall detector's stall rule measures progress relative to the wheel
surface (``progress_xy``), so running on the spot is not "stuck".
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from fly_simulator.jobs import hamster_wheel_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import add_box, contact_kwargs, quat_axis_angle
from fly_simulator.jobs.geometry import spot_or_directional
from fly_simulator.jobs.registry import register_job

P = "wheel/"
SLAT_A = (0.55, 0.92, 0.95, 1.0)  # light aqua rungs (the tan / orange fly stands out)
SLAT_B = (0.25, 0.72, 0.86, 1.0)  # deeper aqua
LIP = (0.92, 0.22, 0.55, 1.0)  # candy-pink rims
LIP_NEAR = (0.92, 0.22, 0.55, 0.3)  # camera side: translucent, the fly stays visible
DISC = (0.90, 0.25, 0.56, 1.0)
STEEL = (0.75, 0.75, 0.78, 1.0)


@dataclass
class HamsterWheelConfig(JobConfig):
    inner_radius: float = 7.0  # running surface radius (14 mm wheel)
    width: float = 5.0  # running surface width between the lips (mm)
    n_slats: int = 60
    slat_thickness: float = 0.3
    bottom_height: float = 1.0  # running surface height at the bottom (mm above the floor)
    wheel_mass: float = 4e-3  # g (4 mg, the fly is 1.02 mg)
    damping: float = 2.0  # hinge damping (uN*mm*s/rad); with ~2 rad/s the fly sits ~3 deg forward
    armature: float = 0.0
    slat_friction: float = 1.0
    lip_height: float = 0.8  # inner lips, radially inward from the running surface
    lip_friction: float = 0.1  # slippery: sticky tarsi can't climb out
    # behaviour
    centre_gain: float = 0.35  # heading correction per mm of lateral offset (rad/mm)
    centre_max_deg: float = 20.0
    speed: float = 1.0
    speed_window_s: float = 1.0  # top speed = max over windows this long
    # looks
    shadows: bool = True  # the lamp's shadow map (~several ms per 960x640 frame)


@register_job
class HamsterWheelJob(EternalJob):
    name = "hamster_wheel"
    #: depth precision for a scene tens of mm away (FlyGym's 5e-4 z-fights); 10 um
    #: still allows close-ups of the fly. Applied by EternalJob.attach.
    znear = 0.01
    title = "HAMSTER WHEEL FLY"
    tagline = "the wheel is the destination"
    work_label = "revolutions"
    work_format = "{:.1f}"
    config_cls = HamsterWheelConfig
    required_names = (P + "wheel",)

    cfg: HamsterWheelConfig

    @property
    def centre_z(self) -> float:
        return self.cfg.bottom_height + self.cfg.inner_radius

    def configure_app(self, app_cfg) -> None:
        super().configure_app(app_cfg)
        # stand on the inside bottom of the wheel (FlyGym's flat-ground spawn is 0.8)
        app_cfg.fly.spawn_height = self.cfg.bottom_height + 0.8
        app_cfg.controller.target_heading_deg = 0.0

    def ground_height(self, x: float, y: float) -> float:
        R = self.cfg.inner_radius
        if abs(x) >= R * 0.95:
            return self.cfg.bottom_height
        return self.centre_z - math.sqrt(R * R - x * x)

    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        self._add_materials(spec)
        vis = dict(contact_kwargs("visual"), mass=0.0)
        R, t, hw = c.inner_radius, c.slat_thickness, c.width / 2
        zc = self.centre_z
        wheel = wb.add_body(name=P + "wheel", pos=(0.0, 0.0, zc))
        wheel.add_joint(name=P + "hinge", type=mj.mjtJoint.mjJNT_HINGE, axis=(0, 1, 0),
                        damping=c.damping, armature=c.armature)
        n = c.n_slats
        dth = 2 * math.pi / n
        slat_len = (R + t) * dth * 1.02 / 2  # half length, slight overlap: no gaps
        m_slat = c.wheel_mass * 0.8 / n
        for i in range(n):
            th = i * dth  # angle from the bottom, about +y
            # position of the slat centre: bottom is (0, 0, -R); rotate about y by th
            r = R + t / 2
            pos = (-r * math.sin(th), 0.0, -r * math.cos(th))
            q = quat_axis_angle((0, 1, 0), th)
            # glossy plastic rungs in two tones, a white marker every 10 (rotation)
            mat = "slat_mark" if i % 10 == 0 else ("slat_a" if i % 2 == 0 else "slat_b")
            add_box(wheel, f"{P}slat{i}", (slat_len, hw + 0.3, t / 2), pos, quat=q,
                    material=P + mat, collide="fly", friction=c.slat_friction, mass=m_slat)
            # lips on both sides (radially inward)
            rl = R - c.lip_height / 2
            lpos = (-rl * math.sin(th), 0.0, -rl * math.cos(th))
            for side, sgn in (("l", 1.0), ("r", -1.0)):
                add_box(wheel, f"{P}lip{i}{side}", (slat_len, 0.12, c.lip_height / 2),
                        (lpos[0], sgn * (hw + 0.12), lpos[2]), quat=q,
                        material=P + ("lip" if sgn > 0 else "lip_near"),
                        collide="fly", friction=c.lip_friction,
                        mass=c.wheel_mass * 0.1 / n)
        # the far side (+y), no contacts: a textured back disc (vent slots, ribs, a
        # badge: it turns with the wheel), raised spokes with a ring, the outer rim
        # and a chrome hub
        axle_q = quat_axis_angle((1, 0, 0), math.pi / 2)  # z -> -y
        A.add_mesh(spec, P + "disc_mesh", A.disc_mesh(R + t + 0.05, 0.1, 96))
        wheel.add_geom(name=P + "disc", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "disc_mesh",
                       pos=(0, hw + 0.4, 0), quat=axle_q, material=P + "disc", **vis)
        A.add_mesh(spec, P + "spokes_mesh", A.spokes_mesh(0.7, R - c.lip_height - 0.08, 6, 0.1))
        wheel.add_geom(name=P + "spokes", type=mj.mjtGeom.mjGEOM_MESH,
                       meshname=P + "spokes_mesh", pos=(0, hw + 0.22, 0), material=P + "lip", **vis)
        A.add_mesh(spec, P + "rim_mesh", A.torus_mesh(R + t + 0.05, 0.16, 96, 10))
        wheel.add_geom(name=P + "rim", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "rim_mesh",
                       pos=(0, hw + 0.3, 0), quat=axle_q, material=P + "lip", **vis)
        wheel.add_geom(name=P + "hub", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.8, 0.3, 0),
                       pos=(0, hw + 0.4, 0), quat=axle_q, material=P + "chrome", **vis)
        # axle + an A-frame stand behind the wheel (static)
        wb.add_geom(name=P + "axle", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.25, 1.2, 0),
                    pos=(0, hw + 1.4, zc), quat=axle_q, material=P + "chrome", **vis)
        A.add_mesh(spec, P + "stand_mesh", A.stand_mesh(zc, hw + 2.3))
        wb.add_geom(name=P + "stand", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "stand_mesh",
                    material=P + "chrome", **vis)
        add_box(wb, P + "base", (5.5, 1.2, 0.12), (0, hw + 2.3, 0.12), material=P + "tray",
                collide="visual")
        self._add_cage(spec)

    def _add_materials(self, spec) -> None:
        c = self.cfg
        mat = spec.material("grid")
        if mat is not None:  # the ground: wood-shaving bedding
            A.add_texture(spec, P + "tex_bedding", A.bedding_texture(c.seed))
            mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = P + "tex_bedding"
            mat.rgba = (1.0, 1.0, 1.0, 1.0)
            mat.reflectance = 0.0
            mat.texrepeat = [v * 0.5 for v in mat.texrepeat]  # an 8 mm texture period
        M = spec.add_material
        M(name=P + "slat_a", rgba=SLAT_A, specular=0.7, shininess=0.8, reflectance=0.05)
        M(name=P + "slat_b", rgba=SLAT_B, specular=0.7, shininess=0.8, reflectance=0.05)
        M(name=P + "slat_mark", rgba=(0.97, 0.97, 0.95, 1.0), specular=0.7, shininess=0.8)
        M(name=P + "lip", rgba=LIP, specular=0.8, shininess=0.85, reflectance=0.08)
        M(name=P + "lip_near", rgba=LIP_NEAR, specular=0.8, shininess=0.85)
        M(name=P + "chrome", rgba=STEEL, specular=1.0, shininess=0.95, reflectance=0.2)
        M(name=P + "tray", rgba=(0.20, 0.55, 0.60, 1.0), specular=0.5, shininess=0.6)
        M(name=P + "bars", rgba=(0.82, 0.83, 0.86, 1.0), specular=0.9, shininess=0.9)
        T = A.add_textured_material
        A.add_texture(spec, P + "tex_disc", A.wheel_disc_texture(colour=DISC[:3]))
        T(spec, P + "disc", P + "tex_disc", rgba=(1, 1, 1, 1), specular=0.7, shininess=0.8)
        A.add_texture(spec, P + "tex_planks", A.plank_texture(c.seed))
        T(spec, P + "planks", P + "tex_planks", rgba=(1, 1, 1, 1), specular=0.1)
        T(spec, P + "roof", P + "tex_planks", rgba=(0.75, 0.42, 0.3, 1), specular=0.1,
          texrepeat=(1.0, 3.0))
        A.add_texture(spec, P + "tex_seed", A.seed_texture())
        T(spec, P + "seed", P + "tex_seed", rgba=(1, 1, 1, 1), specular=0.4, shininess=0.5)
        A.add_texture(spec, P + "tex_room", A.room_texture(c.seed))
        T(spec, P + "room", P + "tex_room", rgba=(1, 1, 1, 1), emission=0.7, specular=0.0)
        M(name=P + "bottle", rgba=(0.86, 0.93, 1.0, 0.28), specular=1.0, shininess=0.95)
        M(name=P + "water", rgba=(0.35, 0.62, 0.98, 0.5), specular=0.8, shininess=0.9)
        M(name=P + "cap", rgba=(0.20, 0.72, 0.30, 1.0), specular=0.5, shininess=0.6)
        M(name=P + "bowl", rgba=(0.96, 0.94, 0.86, 1.0), specular=0.8, shininess=0.85,
          reflectance=0.05)
        M(name=P + "pellets", rgba=(0.62, 0.44, 0.24, 1.0), specular=0.1)
        M(name=P + "pellets2", rgba=(0.52, 0.62, 0.22, 1.0), specular=0.1)

    def _add_cage(self, spec) -> None:
        """The pet cage round the wheel (all visual): a plastic tray, wire bars on
        three sides (the camera side is open), a water bottle hanging on the back
        bars, a food bowl, a wooden hideout, a few sunflower seeds, the room behind
        and the lights (a warm lamp with shadows)."""
        c = self.cfg
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)

        def mesh(name, md, pos, material, quat=(1.0, 0.0, 0.0, 0.0)):
            A.add_mesh(spec, P + name + "_mesh", md)
            wb.add_geom(name=P + name, type=mj.mjtGeom.mjGEOM_MESH, meshname=P + name + "_mesh",
                        pos=tuple(pos), quat=tuple(quat), material=P + material, **vis)

        x0, x1, y0, y1, th = -18.0, 18.0, -10.0, 13.0, 2.2  # tray inside, wall height
        for k, (cx, cy, ex, ey) in enumerate((((x0 + x1) / 2, y1 + 0.2, (x1 - x0) / 2 + 0.4, 0.2),
                                              ((x0 + x1) / 2, y0 - 0.2, (x1 - x0) / 2 + 0.4, 0.2),
                                              (x0 - 0.2, (y0 + y1) / 2, 0.2, (y1 - y0) / 2),
                                              (x1 + 0.2, (y0 + y1) / 2, 0.2, (y1 - y0) / 2))):
            add_box(wb, f"{P}tray{k}", (ex, ey, th / 2), (cx, cy, th / 2), material=P + "tray",
                    collide="visual")
        top = 24.0
        mesh("bars_back", A.bars_mesh(x0, x1, y1 + 0.2, th, top), (0, 0, 0), "bars")
        for k, x in enumerate((x0 - 0.2, x1 + 0.2)):
            md = A.bars_mesh(y0, y1, 0.0, th, top)  # along x, then turned to run along y
            v = md.verts[:, [1, 0, 2]].copy()
            v[:, 0] += x
            mesh(f"bars_side{k}", A.MeshData(v, md.faces[:, ::-1].copy(), md.uv), (0, 0, 0), "bars")
        # the water bottle on the back bars (upside down, the spout into the cage)
        bx, by, bz = 10.5, y1 - 1.1, 7.2
        parts = A.bottle_meshes(8.0, 1.2)
        for nm, mat in (("water", "water"), ("cap", "cap"), ("spout", "chrome"), ("bottle", "bottle")):
            mesh(f"bottle_{nm}", parts[nm], (bx, by, bz), mat)
        mesh("bottle_clip", A.torus_mesh(1.27, 0.07, 40, 6), (bx, by, bz + 5.0), "bars")
        # the food bowl with pellets
        fx, fy = 8.0, -1.8
        mesh("bowl", A.bowl_mesh(1.7, 0.9, 0.08), (fx, fy, 0.0), "bowl")
        mesh("pellets", A.pellets_mesh(1.55, 0.45, 22, 0.24, c.seed), (fx, fy, 0.0), "pellets")
        mesh("pellets2", A.pellets_mesh(1.4, 0.55, 8, 0.2, c.seed + 1), (fx, fy, 0.0), "pellets2")
        # a wooden hideout (door toward the camera)
        parts = A.hideout_meshes(5.0, 3.6, 3.0)
        q = quat_axis_angle((0, 0, 1), 0.35)
        mesh("hideout", parts["walls"], (-11.5, 3.5, 0.0), "planks", q)
        mesh("hideout_roof", parts["roof"], (-11.5, 3.5, 0.0), "roof", q)
        # sunflower seeds strewn on the bedding
        rng = np.random.default_rng(c.seed + 3)
        pts = np.column_stack([rng.uniform(-15, 15, 14), rng.uniform(-8, -3.5, 14)])
        mesh("seeds", A.seeds_mesh(pts, 0.3, c.seed), (0, 0, 0), "seed")
        # the room behind the cage (faces -y, toward the camera)
        from fly_simulator.jobs.taste_tester_assets import front_panel_mesh

        mesh("room", front_panel_mesh(42.0, 17.0, 0.05), (-4.0, 30.0, 15.0), "room",
             quat_axis_angle((0, 0, 1), -math.pi / 2))
        sky = spec.texture("skybox")
        if sky is not None:
            sky.rgb1 = (0.85, 0.78, 0.68)
            sky.rgb2 = (0.55, 0.48, 0.40)
        spec.visual.headlight.ambient = (0.30, 0.30, 0.30)
        spec.visual.headlight.diffuse = (0.32, 0.32, 0.32)
        spec.visual.headlight.specular = (0.12, 0.12, 0.12)
        tgt = np.array([0.0, 2.0, 3.0])
        lamp = np.array([-9.0, -12.0, 26.0])
        wb.add_light(name=P + "lamp", type=spot_or_directional(c.shadows), pos=tuple(lamp),
                     dir=tuple(tgt - lamp), diffuse=(0.62, 0.57, 0.5), specular=(0.45, 0.42, 0.4),
                     cutoff=45.0, exponent=0.4, castshadow=bool(c.shadows))
        fill = np.array([14.0, -12.0, 10.0])
        wb.add_light(name=P + "fill", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=tuple(fill),
                     dir=tuple(tgt - fill), diffuse=(0.2, 0.22, 0.27), specular=(0.15, 0.15, 0.2),
                     cutoff=50.0, exponent=0.5, castshadow=False)

    # ------------------------------------------------------------ attach / state
    def on_attach(self) -> None:
        m = self.sim.model
        j = m.joint(P + "hinge").id
        self.qadr = int(m.jnt_qposadr[j])
        self.vadr = int(m.jnt_dofadr[j])
        self.revolutions = 0.0
        self.distance_mm = 0.0
        self.top_speed = 0.0
        self._speed_hist: deque = deque(maxlen=int(self.cfg.speed_window_s / 0.05) + 1)
        self._t_last_sample = -1.0
        self.on_reset()
        self.state = "running"

    def on_reset(self) -> None:
        self._theta_last = float(self.sim.data.qpos[self.qadr])
        self._theta_accum = 0.0  # surface travel since the last reset (rad)
        self._speed_hist.clear()

    def wheel_angle(self) -> float:
        return float(self.sim.data.qpos[self.qadr])

    def surface_speed(self) -> float:
        """mm/s of the running surface (positive = the fly runs forward)."""
        return float(self.sim.data.qvel[self.vadr]) * self.cfg.inner_radius

    def progress_xy(self, x: float, y: float):
        return (x + self._theta_accum * self.cfg.inner_radius, y)

    # ------------------------------------------------------------ behaviour
    def update(self) -> None:
        c = self.cfg
        th = self.wheel_angle()
        if not math.isfinite(th):
            return
        d = th - self._theta_last
        self._theta_last = th
        if abs(d) < 1.0:  # (a reset re-baselines; guard against jumps)
            self._theta_accum += d
            if d > 0:
                self.revolutions += d / (2 * math.pi)
                self.distance_mm += d * c.inner_radius
                self.add_work(d / (2 * math.pi))
                self.work = self.revolutions
        t = self.sim.time
        if t - self._t_last_sample >= 0.05 or t < self._t_last_sample:
            self._t_last_sample = t
            self._speed_hist.append((t, self._theta_accum))
            (t0, a0), (t1, a1) = self._speed_hist[0], self._speed_hist[-1]
            if t1 - t0 >= c.speed_window_s * 0.9:
                self.top_speed = max(self.top_speed, (a1 - a0) / (t1 - t0) * c.inner_radius)
        y = float(self.sim.data.xpos[self.sim.thorax_body_id, 1])
        lim = math.radians(c.centre_max_deg)
        self.steering.set(min(max(-c.centre_gain * y, -lim), lim), c.speed)
        self.state = "down" if self.fly_down() else "running"

    # ------------------------------------------------------------ view / HUD
    def camera_target(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.centre_z - 0.6 * self.cfg.inner_radius])

    def camera_preset(self) -> CameraPreset:
        # from the open side (-y), a little from the front and above
        return CameraPreset(azimuth=102.0, elevation=-18.0, distance=16.0, tau_s=0.3)

    def job_hud_lines(self) -> list[str]:
        return [f"distance {self.distance_mm / 1000:.3f} m   speed {self.surface_speed():5.1f} mm/s"
                f"   top {self.top_speed:5.1f} mm/s"]

    def job_stats(self) -> dict:
        return {"revolutions": self.revolutions, "distance_m": self.distance_mm / 1000,
                "top_speed_mm_s": self.top_speed, "unstuck": self.n_unstuck}
