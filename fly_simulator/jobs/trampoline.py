"""TRAMPOLINE FLY: the fly bounces on a trampoline forever (docs/JOBS.md, "trampoline").

Physics (the mat is a real springy body, the jumps are the real ``Jump`` action):

* **The mat** is a body on a vertical slide joint (joint damping ``mat_damping``). What
  holds it up is a ring of ``n_springs`` real MuJoCo tendon springs (spatial tendons
  with stiffness, pre-stretched) from the frame's rim to the mat's edge, like the coil
  springs of a backyard trampoline: they are also what you see stretching. Their
  vertical restoring force is geometric (tension x sin of the spring angle), so the
  mat stiffens as it sinks, as real trampolines do. The fly touches the mat's
  (hidden) box collider through explicit contact pairs with FlyGym's own ground
  parameters (``add_ground_pairs``); the visible mat is a disc on the same body.
* **The bounce** is the real ``Jump`` action (crouch -> mid-leg TTM stroke ->
  ballistic flight -> landing), as ``BounceJump``: the first jump from standing in the
  default long mode, then, ``fire_delay_ms`` after each touchdown on the mat (while
  it is being pressed down), a short-mode jump whose stroke coincides with the mat's
  rebound, so the energy stored in the springs and the leg push add up.
* **Engineered, labelled** (the wings have no aerodynamics in the walking model):
  ``BounceJump`` adds an airborne attitude stabiliser (a torque on the thorax,
  wings / halteres emulation) and a small horizontal "wing steering" force (<=
  ``steer_max_bw`` body weights, horizontal only, never lift) that keeps the fly
  over the mat. Both act only between take-off and touchdown.

Counters: bounces (the work counter), best height (mm and body lengths), current /
best streak, crash landings, wobbly landings, falls off the trampoline. A crash or a
fall off ends the streak; the fly then gets a counted, explicit reset onto the mat
(``job.recover``), never a hidden teleport. Slow motion ``slowmo`` is an *edit*
(presentation time), labelled on screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco as mj
import numpy as np

from fly_simulator.actions.base import Action, ActionCommand, smoothstep
from fly_simulator.actions.jump import Jump
from fly_simulator.jobs import trampoline_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import (FLY_BODY_GEOMS, LEG_SEGMENTS, LEGS, contact_kwargs,
                                         quat_axis_angle)
from fly_simulator.jobs.registry import register_job
from fly_simulator.jobs.taste_tester import DIGITS, SEGS

P = "tramp/"
BODY_LENGTH_MM = 2.5  # the fly's body length (for "body lengths" of height)


# ---------------------------------------------------------------------------
# contacts
# ---------------------------------------------------------------------------


def add_ground_pairs(spec, geom_name: str, fly_name: str = "nmf") -> int:
    """Contact ``geom_name`` with the fly exactly like FlyGym's ground plane does:
    explicit ``<pair>``s to the 55 fly geoms (body + every leg segment) with the
    ground's parameters (stiff 0.2 ms solref, friction 1, margin 1 um, condim 3).
    Give the geom contype = conaffinity = 0 (no other contacts). Measured: with these
    pairs a standing fly and a Jump behave on a box as on the floor; with a
    soft-contact prop the tarsal adhesion pulls the legs deep into the contact and
    drives the light spring-mounted mat into a sustained jitter."""
    from flygym.compose import ContactParams

    cp = ContactParams()
    names = list(FLY_BODY_GEOMS) + [f"{leg}_{seg}" for leg in LEGS for seg in LEG_SEGMENTS]
    for n in names:
        spec.add_pair(geomname1=geom_name, geomname2=f"{fly_name}/{n}", condim=3,
                      friction=[cp.sliding_friction, cp.sliding_friction, cp.torsional_friction,
                                cp.rolling_friction, cp.rolling_friction],
                      solref=list(cp.get_solref_tuple()), solimp=list(cp.get_solimp_tuple()),
                      margin=cp.margin)
    return len(names)


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------


class MatStance(Action):
    """Stand still on the mat: the standing pose, all tarsi adhering (like ``freeze``;
    walking on the springy mat makes it bounce the fly around). Stationary."""

    name = "mat_stance"
    blend_in = 0.0
    blend_out = 0.1

    def __init__(self, duration: float = 3600.0) -> None:
        super().__init__(duration)

    def begin(self, mgr) -> None:
        self._start = mgr.sim.data.ctrl[mgr.body.pos_ids].copy()
        self._stand = mgr.body.stand.copy()

    def command(self, mgr, t: float) -> ActionCommand:
        a = smoothstep(t / 0.05)
        return ActionCommand(targets=(1 - a) * self._start + a * self._stand, adhesion=np.ones(6))


class BounceJump(Jump):
    """The real ``Jump`` action (crouch -> mid-leg TTM stroke -> flight -> landing),
    plus two optional **engineered, labelled** airborne aids (the wings have no
    aerodynamics in the walking model), both only between take-off and touchdown:

    * an attitude stabiliser (wings / halteres emulation): a torque on the thorax
      turns its z axis back to ``up_fn()`` (default world up) and damps the body
      rates, critically damped at ``att_hz``, capped at ``att_max`` uN*mm;
    * "wing steering": a horizontal force servoing the horizontal velocity to
      ``steer_gain * (home_xy - p)`` (time constant ``steer_tau_s``), capped at
      ``steer_max_bw`` body weights. Horizontal only: it never adds lift or height.

    ``max_att_torque`` / ``max_steer_bw`` record the largest aid used."""

    def __init__(self, att_hz: float = 0.0, att_max: float = 2.0, up_fn=None,
                 home_xy=None, steer_max_bw: float = 0.0, steer_gain: float = 8.0,
                 steer_tau_s: float = 0.02, **overrides) -> None:
        super().__init__(**overrides)
        self.att_hz, self.att_max = float(att_hz), float(att_max)
        self.up_fn = up_fn
        self.home_xy = None if home_xy is None else np.asarray(home_xy, float)
        self.steer_max_bw = float(steer_max_bw)
        self.steer_gain, self.steer_tau_s = float(steer_gain), float(steer_tau_s)
        self.max_att_torque = 0.0
        self.max_steer_bw = 0.0

    def begin(self, mgr) -> None:
        super().begin(mgr)
        if not self.p.flight_assist:
            self._setup_flight(mgr.sim)  # composite inertia, mass, g

    @property
    def took_off(self) -> bool:
        """Past the stroke and no leg on anything since take-off was detected."""
        return self._t_takeoff is not None and self._w_stroke_end is not None

    def command(self, mgr, t: float) -> ActionCommand:
        cmd = super().command(mgr, t)
        airborne = (self._t_touch is None and self._t_takeoff is not None
                    and t >= self.prep_s + self.p.stroke_s)
        if (self.att_hz > 0 or self.steer_max_bw > 0) and airborne:
            sim = mgr.sim
            tq = np.zeros(3)
            if self.att_hz > 0:
                R = sim.thorax_rotmat()
                up = (np.array([0.0, 0.0, 1.0]) if self.up_fn is None
                      else np.asarray(self.up_fn(), float))
                err = np.cross(R[:, 2], up / max(float(np.linalg.norm(up)), 1e-9))
                w = 2 * np.pi * self.att_hz
                tq = self._inertia * (w * w * err - 2.0 * w * sim.thorax_angvel_world())
                n = float(np.linalg.norm(tq))
                if n > self.att_max:
                    tq *= self.att_max / n
                self.max_att_torque = max(self.max_att_torque, min(n, self.att_max))
            f = np.zeros(3)
            if self.home_xy is not None and self.steer_max_bw > 0:
                p = sim.thorax_position()
                v = sim.thorax_linvel()
                v_des = self.steer_gain * (self.home_xy - p[:2])
                f[:2] = self._mass * (v_des - v[:2]) / max(self.steer_tau_s, 1e-4)
                cap = self.steer_max_bw * self._mass * self._g
                nf = float(np.linalg.norm(f))
                if nf > cap:
                    f *= cap / nf
                self.max_steer_bw = max(self.max_steer_bw,
                                        min(nf, cap) / (self._mass * self._g))
            new = np.concatenate([f, tq])
            d = sim.data
            if self._xf is not None:
                d.xfrc_applied[sim.thorax_body_id] -= self._xf
            d.xfrc_applied[sim.thorax_body_id] += new
            self._xf = new
        elif self._xf is not None and not self.p.flight_assist:
            self._clear_flight_force(mgr.sim)
        return cmd


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class TrampolineConfig(JobConfig):
    # --- the trampoline (mm, g, uN) ------------------------------------------
    mat_z: float = 2.0  # mat top at rest, above the lawn
    mat_radius: float = 6.0
    frame_radius: float = 6.9
    collider_half: float = 5.3  # the mat's square box collider (box-capsule contacts are exact)
    mat_mass: float = 1.5e-4  # 0.15 mg
    n_springs: int = 48
    spring_k: float = 10.0  # uN/mm per spring (tendon stiffness)
    spring_pre: float = 0.04  # pre-stretch of every spring at rest (mm)
    mat_damping: float = 0.02  # slide-joint damping, uN per mm/s
    rigid_mat: bool = False  # tests: no slide joint (the mat is as hard as the frame)
    # --- the bounce ---------------------------------------------------------------
    settle_s: float = 0.3  # stance on the mat before the first jump
    # touchdown -> next jump (ms), by streak: late at first (the stroke comes after the
    # mat has started to rebound), then on the beat (the stroke starts at the bottom
    # of the compression): the height is pumped up over a few bounces
    fire_delay_ms: tuple = (4.0, 3.0, 2.0, 1.0)
    fire_delay_late_ms: float = 0.5  # after the schedule
    # the fly's timing noise (our behaviour model): N(0, jitter) per bounce, and now
    # and then a misstep (late by misstep_ms: the mat has already rebounded)
    timing_jitter_ms: float = 0.6
    misstep_p: float = 0.03
    misstep_ms: float = 4.0
    # tricks: a backflip (real dynamics: an asymmetric, boosted push with the mid coxae
    # trimmed back, the attitude stabiliser off for that flight)
    trick_p: float = 0.12  # per bounce, in the trick window below
    trick_min_streak: int = 5
    trick_min_height: float = 3.6  # the window: tricks from a medium bounce (measured:
    trick_max_height: float = 4.7  # from the highest bounces most flips crash)
    trick_mid_thc: float = -30.0  # deg (the Jump's default is +12)
    trick_boost: float = 1.8  # the Jump's documented boost option (stroke kp + force range)
    trick_delay_ms: float = 3.0  # touchdown -> the trick jump (a less energetic take-off)
    after_trick_delay_ms: float = 6.0  # the bounce after a landed flip (settles it down)
    trick_inverted_deg: float = 140.0  # a flip passes through upside down (tilt >= this)
    bounce_mode: str = "short"  # Jump mode of the rebound jumps (first jump: long)
    att_hz: float = 15.0  # airborne attitude stabiliser (engineered, 0 = off)
    att_max: float = 2.0  # uN*mm
    steer_max_bw: float = 0.15  # horizontal wing-steering force cap (engineered, 0 = off)
    steer_gain: float = 8.0  # 1/s: v_des = gain * (centre - p)
    # "foot placement" in the air: the stabiliser's target up vector tilts by
    # lean_gain rad per mm of offset from the centre (at most lean_max rad). Negative =
    # the body tilts away from the centre; measured: then the next take-off pushes
    # the fly back toward the centre (with 0 it drifts ~backward and falls off)
    lean_gain: float = -0.05
    lean_max: float = 0.25
    wobbly_deg: float = 10.0  # tilt at touchdown -> "wobbly landing"
    crash_deg: float = 50.0  # tilt at touchdown (or later on the mat) -> crash landing
    crash_hold_s: float = 1.2  # a crashed fly lies there this long, then is reset
    off_hold_s: float = 1.0  # a fly on the lawn, then reset onto the mat
    # --- looks / presentation ---------------------------------------------------------
    slowmo: float = 0.2  # slow motion while bouncing (edit effect; 1 = off)
    ruler_mm: float = 8.0
    shadows: bool = True


@register_job
class TrampolineJob(EternalJob):
    name = "trampoline"
    title = "TRAMPOLINE FLY"
    tagline = "the fly bounces on a trampoline forever"
    work_label = "bounces"
    config_cls = TrampolineConfig
    required_names = (P + "mat", P + "mat_plate", P + "ruler")

    def __init__(self, cfg: TrampolineConfig | None = None) -> None:
        super().__init__(cfg)
        self.phase = "settle"  # settle | launch | air | contact | crashed | off
        self.t_phase = 0.0
        self.n_bounces = 0
        self.streak = 0
        self.best_streak = 0
        self.best_height = 0.0
        self.last_height = 0.0
        self.n_crashes = 0
        self.n_wobbly = 0
        self.n_off = 0
        self.n_tricks = 0  # flips landed
        self.n_trick_tries = 0
        self.n_missteps = 0
        self._trick = False
        self._misstep = False
        self._next_trick = False
        self._pitch_int = 0.0
        self._trick_tilt = 0.0
        self._delay_s = 0.004
        self.last_flip_deg = 0.0
        self.rng = np.random.default_rng(self.cfg.seed + 99)
        self.n_jumps = 0
        self.flash = ""
        self._flash_until = -1.0
        self.z_stand = self.cfg.mat_z + 1.0
        self._apex = 0.0
        self._t_touch = 0.0
        self._tilt_td = 0.0
        self._jump_action: BounceJump | None = None
        self._need_recover: str | None = None
        self._scale = 1.0
        self._last_scale = 1.0
        self.sum_height = 0.0
        self.max_steer_bw = 0.0
        self.max_att_torque = 0.0
        self.mat_min_seen = 0.0

    # ------------------------------------------------------------ build
    def configure_app(self, app_cfg) -> None:
        super().configure_app(app_cfg)
        app_cfg.fly.spawn_height = self.cfg.mat_z + 0.8

    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        self._add_materials(spec)
        self._add_yard(spec)
        z = c.mat_z
        # --- the mat: slide joint + a ring of tendon springs to the frame ----------
        # rigid_mat: a fixed (mocap) body, as hard as the frame (for comparisons)
        mat = wb.add_body(name=P + "mat", pos=(0, 0, z), mocap=bool(c.rigid_mat))
        if not c.rigid_mat:
            mat.add_joint(name=P + "mat_slide", type=mj.mjtJoint.mjJNT_SLIDE, axis=(0, 0, 1),
                          damping=c.mat_damping, range=(-z + 0.3, 1.0), limited=True)
        mat.gravcomp = 1.0  # the springs carry the fly; the mat's own weight is trimmed out
        mat.add_geom(name=P + "mat_plate", type=mj.mjtGeom.mjGEOM_BOX,
                     size=(c.collider_half, c.collider_half, 0.4), pos=(0, 0, -0.4),
                     mass=c.mat_mass, rgba=(0.1, 0.1, 0.1, 0.0), group=3,
                     contype=0, conaffinity=0)
        add_ground_pairs(spec, P + "mat_plate", self.fly_name)
        A.add_mesh(spec, P + "mat_mesh", A.disc_mesh(c.mat_radius, 0.03))
        mat.add_geom(name=P + "mat_vis", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "mat_mesh",
                     pos=(0, 0, -0.015), material=P + "mat", **vis)
        g = c.frame_radius - c.mat_radius
        L0 = g - c.spring_pre
        for i in range(c.n_springs):
            th = 2 * math.pi * (i + 0.5) / c.n_springs
            u = np.array([math.cos(th), math.sin(th)])
            mat.add_site(name=P + f"hook_m{i}", pos=(*(c.mat_radius * u), -0.01),
                         size=(0.02, 0, 0), group=3)
            wb.add_site(name=P + f"hook_f{i}", pos=(*(c.frame_radius * u), z - 0.01),
                        size=(0.02, 0, 0), group=3)
            t = spec.add_tendon(name=P + f"spring{i}", stiffness=c.spring_k,
                                springlength=(L0, L0), width=0.035, material=P + "spring")
            t.wrap_site(P + f"hook_f{i}")
            t.wrap_site(P + f"hook_m{i}")
        # --- the frame: a steel ring on six bent legs (visual) -------------------
        A.add_mesh(spec, P + "frame_mesh", A.torus_mesh(c.frame_radius + 0.05, 0.14, 96, 12))
        wb.add_geom(name=P + "frame", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "frame_mesh",
                    pos=(0, 0, z - 0.02), material=P + "steel", **vis)
        for k in range(6):
            a = 2 * math.pi * (k + 0.25) / 6
            u = np.array([math.cos(a), math.sin(a), 0.0])
            v = np.array([-math.sin(a), math.cos(a), 0.0])
            R = c.frame_radius + 0.05
            top, knee = R * u + np.array([0, 0, z - 0.1]), R * u + np.array([0, 0, z * 0.45])
            f1 = (R + 0.25) * u + 0.5 * v + np.array([0, 0, 0.1])
            f2 = (R + 0.25) * u - 0.5 * v + np.array([0, 0, 0.1])
            for nm, pts in (("a", [top, knee, f1]), ("b", [knee, f2])):
                A.add_mesh(spec, P + f"leg{k}{nm}_mesh", A.polyline_tube(np.array(pts), 0.09))
                wb.add_geom(name=P + f"leg{k}{nm}", type=mj.mjtGeom.mjGEOM_MESH,
                            meshname=P + f"leg{k}{nm}_mesh", material=P + "steel", **vis)
        self._add_ruler(spec)
        self._add_scoreboard(spec)

    def _add_materials(self, spec) -> None:
        c = self.cfg
        A.add_texture(spec, P + "tex_mat", A.mat_texture(c.seed))
        A.add_textured_material(spec, P + "mat", P + "tex_mat", rgba=(1, 1, 1, 1),
                                specular=0.25, shininess=0.3)
        spec.add_material(name=P + "spring", rgba=(0.78, 0.8, 0.84, 1), specular=0.9,
                          shininess=0.9, reflectance=0.1)
        spec.add_material(name=P + "steel", rgba=(0.55, 0.58, 0.62, 1), specular=0.8,
                          shininess=0.8)
        A.add_texture(spec, P + "tex_ruler", A.ruler_texture(c.ruler_mm, BODY_LENGTH_MM))
        A.add_textured_material(spec, P + "ruler", P + "tex_ruler", rgba=(1, 1, 1, 1),
                                specular=0.1, emission=0.2)
        spec.add_material(name=P + "pole", rgba=(0.85, 0.85, 0.82, 1), specular=0.4)
        A.add_texture(spec, P + "tex_board", A.scoreboard_texture())
        A.add_textured_material(spec, P + "board", P + "tex_board", rgba=(1, 1, 1, 1),
                                emission=0.25)
        A.add_texture(spec, P + "tex_backdrop", A.backdrop_texture(c.seed))
        A.add_textured_material(spec, P + "backdrop", P + "tex_backdrop", rgba=(1, 1, 1, 1),
                                emission=0.35, specular=0.0)
        spec.add_material(name=P + "seg_on", rgba=(1.0, 0.42, 0.08, 1), emission=0.8)
        spec.add_material(name=P + "seg_off", rgba=(0.09, 0.07, 0.06, 1), emission=0.0)
        spec.add_material(name=P + "mark_last", rgba=(0.98, 0.25, 0.15, 1), emission=0.5)
        spec.add_material(name=P + "mark_best", rgba=(1.0, 0.82, 0.1, 1), emission=0.6)

    def _add_yard(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        mat = spec.material("grid")
        if mat is not None:
            A.add_texture(spec, P + "tex_lawn", A.lawn_texture(c.seed))
            mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = P + "tex_lawn"
            mat.rgba = (1.0, 1.0, 1.0, 1.0)
            mat.reflectance = 0.0
        from fly_simulator.jobs.taste_tester_assets import front_panel_mesh

        # the fence + sky backdrop behind the trampoline (faces -y, toward the camera)
        A.add_mesh(spec, P + "backdrop_mesh", front_panel_mesh(40.0, 20.0, 0.05))
        wb.add_geom(name=P + "backdrop", type=mj.mjtGeom.mjGEOM_MESH,
                    meshname=P + "backdrop_mesh", pos=(0.0, 30.0, 17.0),
                    quat=quat_axis_angle((0, 0, 1), -math.pi / 2), material=P + "backdrop",
                    **dict(contact_kwargs("visual"), mass=0.0))
        spec.visual.map.znear = 0.05
        spec.visual.headlight.ambient = (0.32, 0.32, 0.32)
        spec.visual.headlight.diffuse = (0.35, 0.35, 0.35)
        spec.visual.headlight.specular = (0.1, 0.1, 0.1)
        tgt = np.array([0.0, 0.0, c.mat_z + 2.0])
        sun = np.array([-10.0, -14.0, 30.0])
        wb.add_light(name=P + "sun", type=mj.mjtLightType.mjLIGHT_SPOT, pos=tuple(sun),
                     dir=tuple(tgt - sun), diffuse=(0.62, 0.6, 0.55), specular=(0.4, 0.4, 0.4),
                     cutoff=35.0, exponent=0.5, castshadow=bool(c.shadows))
        fill = np.array([12.0, -10.0, 12.0])
        wb.add_light(name=P + "fill", type=mj.mjtLightType.mjLIGHT_SPOT, pos=tuple(fill),
                     dir=tuple(tgt - fill), diffuse=(0.25, 0.27, 0.32), specular=(0.1, 0.1, 0.1),
                     cutoff=45.0, exponent=1.0, castshadow=False)

    def _add_ruler(self, spec) -> None:
        """The height ruler behind the trampoline (0 = the mat at rest), with a red
        marker for the last bounce and a gold one for the best (mocap)."""
        c = self.cfg
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        from fly_simulator.jobs.taste_tester_assets import front_panel_mesh

        self.ruler_xy = np.array([-(c.frame_radius + 1.1), 0.0])  # beside the fly's plane
        x, y = self.ruler_xy
        hz = c.ruler_mm / 2
        face = quat_axis_angle((0, 0, 1), -math.pi / 2)  # +x face -> -y (the camera)
        A.add_mesh(spec, P + "ruler_mesh", front_panel_mesh(0.8, hz, 0.04))
        wb.add_geom(name=P + "ruler", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "ruler_mesh",
                    pos=(x, y, c.mat_z + hz), quat=face, material=P + "ruler", **vis)
        wb.add_geom(name=P + "pole", type=mj.mjtGeom.mjGEOM_CYLINDER,
                    size=(0.12, (c.mat_z + c.ruler_mm + 0.3) / 2, 0),
                    pos=(x - 0.95, y + 0.05, (c.mat_z + c.ruler_mm + 0.3) / 2),
                    material=P + "pole", **vis)
        for nm in ("mark_last", "mark_best"):
            b = wb.add_body(name=P + nm, mocap=True, pos=(x, y - 0.08, c.mat_z))
            off = 0.95 if nm == "mark_best" else -0.95
            b.add_geom(name=P + nm + "_bar", type=mj.mjtGeom.mjGEOM_BOX, size=(0.85, 0.02, 0.025),
                       material=P + nm, **vis)
            b.add_geom(name=P + nm + "_tip", type=mj.mjtGeom.mjGEOM_BOX, size=(0.14, 0.02, 0.1),
                       pos=(off, 0, 0), material=P + nm, **vis)

    def _add_scoreboard(self, spec) -> None:
        """Scoreboard to the right: BOUNCES (4 digits), BEST mm (xx.x), STREAK (3)."""
        c = self.cfg
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        from fly_simulator.jobs.taste_tester_assets import front_panel_mesh

        bx, by, bz, hw, hh = 6.2, c.frame_radius + 2.5, c.mat_z + 7.4, 3.2, 1.6
        face = quat_axis_angle((0, 0, 1), -math.pi / 2)
        A.add_mesh(spec, P + "board_mesh", front_panel_mesh(hw, hh, 0.06))
        wb.add_geom(name=P + "board", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "board_mesh",
                    pos=(bx, by, bz), quat=face, material=P + "board", **vis)
        for s in (-1, 1):
            wb.add_geom(name=P + f"board_post{s:+d}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                        size=(0.1, (bz - hh) / 2, 0),
                        pos=(bx + s * (hw - 0.4), by + 0.1, (bz - hh) / 2),
                        material=P + "steel", **vis)
        self._seg_names: list[list[list[str]]] = []
        dw, dh, st, gap = 0.38, 0.27, 0.055, 0.58
        n_dig = (4, 3, 3)
        # label rows at 0.32 / 0.56 / 0.80 of the texture height (row 0 = the top)
        for di, frac in enumerate((0.32, 0.56, 0.80)):
            zc = bz + hh - 2 * hh * frac
            x_right = bx - hw + 2 * hw * 0.89
            disp = []
            for k in range(n_dig[di]):
                xd = x_right - (n_dig[di] - 1 - k) * gap
                segs = []
                for sname, (dx, dz, horiz) in SEGS.items():
                    nm = f"{P}seg{di}_{k}_{sname}"
                    size = (dw / 2 - 0.03, 0.01, st) if horiz else (st, 0.01, dh / 2 - 0.03)
                    wb.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_BOX, size=size,
                                pos=(xd + dx * dw, by - 0.05, zc + dz * dh),
                                material=P + "seg_off", **vis)
                    segs.append(nm)
                disp.append(segs)
            self._seg_names.append(disp)
            if di == 1:  # decimal point of BEST (xx.x)
                xd = x_right - gap
                wb.add_geom(name=P + "seg_dot", type=mj.mjtGeom.mjGEOM_BOX,
                            size=(st, 0.01, st), pos=(xd + gap / 2, by - 0.05, zc - dh),
                            material=P + "seg_on", **vis)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        sim, s = self.sim, self.session
        m = sim.model
        c = self.cfg
        if c.rigid_mat:
            self.q_mat = self.v_mat = None
        else:
            j = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, P + "mat_slide")
            self.q_mat = int(m.jnt_qposadr[j])
            self.v_mat = int(m.jnt_dofadr[j])
        self.body = s.actions.body
        self.mat_gid = m.geom(P + "mat_plate").id
        self.mocap_last = int(m.body_mocapid[m.body(P + "mark_last").id])
        self.mocap_best = int(m.body_mocapid[m.body(P + "mark_best").id])
        self.seg_gid = [[[m.geom(n).id for n in dig] for dig in disp] for disp in self._seg_names]
        self.matid = {n: m.material(P + n).id for n in ("seg_on", "seg_off")}
        self.spring_ids = [m.tendon(P + f"spring{i}").id for i in range(c.n_springs)]
        st = getattr(s, "STATIONARY_ACTIONS", None)
        if st is not None and MatStance.name not in st:
            s.STATIONARY_ACTIONS = (*st, MatStance.name)
        self._start_settle()
        self._update_board()

    # ------------------------------------------------------------ helpers
    def mat_q(self) -> float:
        return 0.0 if self.q_mat is None else float(self.sim.data.qpos[self.q_mat])

    def mat_v(self) -> float:
        return 0.0 if self.v_mat is None else float(self.sim.data.qvel[self.v_mat])

    def mat_top(self) -> float:
        return self.cfg.mat_z + self.mat_q()

    def ground_height(self, x: float, y: float) -> float:
        if self.sim is None:
            return self.cfg.mat_z
        if math.hypot(x, y) < self.cfg.frame_radius:
            return self.mat_top()
        return 0.0

    def on_mat(self) -> bool:
        p = self.sim.thorax_position()
        return math.hypot(p[0], p[1]) < self.cfg.mat_radius and p[2] > self.mat_top() - 0.3

    def mat_contacts(self) -> tuple[int, bool, bool]:
        """(fly legs touching the mat, fly body touching the mat, fly touching the lawn)."""
        d, m = self.sim.data, self.sim.model
        legs = np.zeros(6, bool)
        body = lawn = False
        fly = self.body._fly_body
        for i in range(d.ncon):
            g1, g2 = d.contact.geom[i]
            if g1 == self.mat_gid or g2 == self.mat_gid:
                g = g2 if g1 == self.mat_gid else g1
                leg = self.body.geom_leg[g]
                if leg >= 0:
                    legs[leg] = True
                else:
                    body = True
            elif fly[m.geom_bodyid[g1]] != fly[m.geom_bodyid[g2]]:
                lawn = True  # the fly touches something that is not the mat
        return int(legs.sum()), body, lawn

    def height_now(self) -> float:
        """Bounce height: the thorax's rise above its standing height on the resting
        mat (about the feet's clearance above the mat)."""
        return float(self.sim.thorax_position()[2]) - self.z_stand

    def _say_flash(self, text: str, secs: float = 1.5) -> None:
        self.flash = text
        self._flash_until = self.run_time() + secs

    def _go(self, phase: str) -> None:
        self.phase = self.state = phase
        self.t_phase = self.sim.time

    def _start_settle(self) -> None:
        self._go("settle")
        self.steering.set(None, 0.0)
        self.session.actions.trigger(MatStance(), source="job")

    def fire_delay_s(self) -> float:
        """Touchdown -> next jump for this bounce (drawn at touchdown)."""
        return self._delay_s

    def _draw_delay(self) -> None:
        c = self.cfg
        sched = c.fire_delay_ms
        k = self.streak
        ms = float(sched[k] if k < len(sched) else c.fire_delay_late_ms)
        ms += float(self.rng.normal(0.0, c.timing_jitter_ms)) if c.timing_jitter_ms > 0 else 0.0
        self._next_trick = self._want_trick()
        self._misstep = bool(not self._next_trick and c.misstep_p > 0
                             and self.rng.random() < c.misstep_p)
        if self._next_trick:
            ms = c.trick_delay_ms
        if self._misstep:
            ms += c.misstep_ms
            self.n_missteps += 1
        self._delay_s = max(ms, 0.0) / 1000.0

    def _lean_up(self) -> np.ndarray:
        """Target up vector of the attitude stabiliser: tilted toward the centre."""
        c = self.cfg
        p = self.sim.thorax_position()
        h = -c.lean_gain * p[:2]
        n = float(np.linalg.norm(h))
        if n > c.lean_max:
            h *= c.lean_max / n
        return np.array([h[0], h[1], 1.0])

    def _want_trick(self) -> bool:
        c = self.cfg
        return (c.trick_p > 0 and self.streak >= c.trick_min_streak
                and c.trick_min_height <= self.last_height <= c.trick_max_height
                and self.rng.random() < c.trick_p)

    def _jump(self, first: bool, trick: bool = False) -> None:
        c = self.cfg
        kw = dict(att_hz=c.att_hz, att_max=c.att_max, home_xy=(0.0, 0.0),
                  steer_max_bw=c.steer_max_bw, steer_gain=c.steer_gain,
                  up_fn=self._lean_up if c.lean_gain != 0 else None)
        if not first:
            kw["mode"] = c.bounce_mode
        if trick:  # backflip: stabiliser off, mid coxae trimmed back, boosted stroke
            kw.update(att_hz=0.0, mid_thc=c.trick_mid_thc, boost=c.trick_boost)
            self.n_trick_tries += 1
            self._say_flash("BACKFLIP ATTEMPT!", 1.0)
        self._trick = trick
        self._pitch_int = 0.0
        self._trick_tilt = 0.0
        j = BounceJump(**kw)
        self._jump_action = j
        self.session.actions.trigger(j, source="job")
        self.n_jumps += 1
        self._go("launch")

    # ------------------------------------------------------------ job logic
    def update(self) -> None:
        c = self.cfg
        sim = self.sim
        t = sim.time
        age = t - self.t_phase
        acts = self.session.actions
        self.mat_min_seen = min(self.mat_min_seen, self.mat_q())
        if self._trick and self.phase in ("launch", "air"):
            # pitch rotation (nose-up < 0) integrated at 1 ms, and the largest tilt
            self._pitch_int += (float(sim.thorax_angvel_local()[1])
                                * c.update_every_steps * sim.timestep)
            self._trick_tilt = max(self._trick_tilt, sim.tilt_deg())
        if self.phase == "settle":
            if age >= c.settle_s:
                self.z_stand = float(sim.thorax_position()[2]) - self.mat_q()
                self._jump(first=True)
        elif self.phase == "launch":
            j = self._jump_action
            if acts.action is not j and acts._pending is not j:
                self._landed_elsewhere("the jump ended before take-off")
            elif j.took_off and not self.body.leg_contacts(sim).any():
                self._apex = float(sim.thorax_position()[2])
                self._go("air")
            elif age > 0.3:
                self._landed_elsewhere("no take-off")
        elif self.phase == "air":
            self._apex = max(self._apex, float(sim.thorax_position()[2]))
            n_legs, body, lawn = self.mat_contacts()
            if lawn:
                self._fell_off("landed off the trampoline")
            elif n_legs > 0 or body:
                self._touchdown(t, body)
            elif age > 1.0:
                self._landed_elsewhere("lost in the air")
        elif self.phase == "contact":
            n_legs, body, lawn = self.mat_contacts()
            if lawn:
                self._fell_off("slid off the trampoline")
            elif sim.tilt_deg() > c.crash_deg or body:
                self._crash(sim.tilt_deg())
            elif t - self._t_touch >= self.fire_delay_s() - 1e-9:
                self._count_bounce()
                self._jump(first=False, trick=self._next_trick)
        elif self.phase == "crashed":
            if age >= c.crash_hold_s:
                self._need_recover = "crash_landing"
        elif self.phase == "off":
            if age >= c.off_hold_s:
                self._need_recover = "fell_off"
        if self.flash and self.run_time() > self._flash_until:
            self.flash = ""

    def _touchdown(self, t: float, body: bool) -> None:
        c = self.cfg
        self._t_touch = t
        self._tilt_td = self.sim.tilt_deg()
        h = self._apex - self.z_stand
        self.last_height = h
        j = self._jump_action
        self.max_steer_bw = max(self.max_steer_bw, j.max_steer_bw)
        self.max_att_torque = max(self.max_att_torque, j.max_att_torque)
        self._place_marker(self.mocap_last, h)
        if h > self.best_height:
            self.best_height = h
            self._place_marker(self.mocap_best, h)
            self._update_board()
            if self.n_bounces > 2:
                self._say_flash(f"NEW BEST {h:.2f} mm ({h / BODY_LENGTH_MM:.2f} body lengths)!")
        if self._trick:
            flip = abs(math.degrees(self._pitch_int))
            self.last_flip_deg = flip
            # a flip = it went through upside down and landed on its feet
            ok = (not body and self._tilt_td <= c.crash_deg
                  and self._trick_tilt >= c.trick_inverted_deg)
            self.session.log_event("trampoline_trick", flip_deg=round(flip, 1),
                                   max_tilt_deg=round(self._trick_tilt, 1),
                                   tilt_deg=round(self._tilt_td, 1), landed=bool(ok))
            if ok:
                self._trick = False
                self.n_tricks += 1
                self._count_bounce()
                self._say_flash(f"BACKFLIP LANDED! ({flip:.0f} deg)", 2.0)
                self._draw_delay()  # and on into the next bounce, a gentle one
                self._delay_s = c.after_trick_delay_ms / 1000.0
                self._next_trick = self._misstep = False
                self._go("contact")
                return
            if body or self._tilt_td > c.crash_deg:
                self._crash(self._tilt_td)
                self._trick = False
                return
            self._trick = False
            self._say_flash(f"half a flip ({flip:.0f} deg)", 1.2)
        if body or self._tilt_td > c.crash_deg:
            self._crash(self._tilt_td)
            return
        self._draw_delay()
        if self._tilt_td > c.wobbly_deg:
            self.n_wobbly += 1
            self._say_flash(f"wobbly landing ({self._tilt_td:.0f} deg)", 0.8)
        self._go("contact")

    def _count_bounce(self) -> None:
        self.n_bounces += 1
        self.streak += 1
        self.best_streak = max(self.best_streak, self.streak)
        self.sum_height += self.last_height
        self.add_work(1)
        self._update_board()

    def _end_streak(self) -> None:
        self.streak = 0
        self._update_board()

    def _crash(self, tilt: float) -> None:
        self.n_crashes += 1
        self._end_streak()
        self.session.actions.trigger(MatStance(), source="job")
        self._say_flash(f"CRASH LANDING! ({tilt:.0f} deg)", 2.0)
        self.session.log_event("trampoline_crash", tilt_deg=round(float(tilt), 1),
                               after=self._last_kind())
        self._go("crashed")

    def _fell_off(self, why: str) -> None:
        self.n_off += 1
        self._end_streak()
        self._say_flash(f"OFF THE TRAMPOLINE! ({why})", 2.0)
        p = self.sim.thorax_position()
        self.session.log_event("trampoline_off", why=why, after=self._last_kind(),
                               r_mm=round(math.hypot(p[0], p[1]), 2))
        self._go("off")

    def _last_kind(self) -> str:
        return "trick" if self._trick else ("misstep" if self._misstep else "bounce")

    def _landed_elsewhere(self, why: str) -> None:
        if self.on_mat():
            self._crash(self.sim.tilt_deg())
        else:
            self._fell_off(why)

    def _place_marker(self, mocap: int, h: float) -> None:
        x, y = self.ruler_xy
        z = self.cfg.mat_z + float(np.clip(h, 0.0, self.cfg.ruler_mm))
        self.sim.data.mocap_pos[mocap] = (x, y - 0.08, z)

    def _update_board(self) -> None:
        if not hasattr(self, "seg_gid"):
            return
        m = self.sim.model
        vals = [(int(self.n_bounces) % 10000, 4), (int(round(self.best_height * 10)) % 1000, 3),
                (int(self.streak) % 1000, 3)]
        for di, (v, n) in enumerate(vals):
            for k in range(n):
                dg = (v // 10 ** (n - 1 - k)) % 10
                show = v >= 10 ** (n - 1 - k) or k == n - 1 or (di == 1 and k == n - 2)
                lit = DIGITS[dg] if show else ""
                for sname, g in zip(SEGS, self.seg_gid[di][k]):
                    m.geom_matid[g] = self.matid["seg_on" if sname in lit else "seg_off"]

    def on_reset(self) -> None:
        self._need_recover = None
        self._trick = False
        self._scale = 1.0
        self._start_settle()

    def reset_props(self) -> None:
        # the keyframe put the mat back at rest and the markers at their spec pose
        if self.last_height:
            self._place_marker(self.mocap_last, self.last_height)
        if self.best_height:
            self._place_marker(self.mocap_best, self.best_height)
        self._update_board()

    # ------------------------------------------------------------ outside sim.step
    def after_physics(self) -> None:
        super().after_physics()
        why = self._need_recover
        if why:
            self._need_recover = None
            self.recover(why)

    # ------------------------------------------------------------ edit effects
    def time_scale(self, present_dt: float) -> float:
        """Slow motion while bouncing (an *edit effect*: fewer physics steps per
        displayed frame; the physics is unchanged). At fly scale a bounce lasts
        ~0.1 s: 1-3 frames at real time."""
        tgt = self.cfg.slowmo if self.phase in ("launch", "air", "contact") else 1.0
        tgt = min(max(float(tgt), 0.02), 1.0)
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
        """Label the slow motion on screen (edit effect, not physics)."""
        if self._last_scale >= 1.0:
            return frame
        import cv2

        H, W = frame.shape[:2]
        out = np.ascontiguousarray(frame).copy()
        txt = f"SLOW MOTION x{self._last_scale:.2f}  (edit, not physics)"
        fs = max(0.4, W / 1400.0)
        th = max(1, int(round(W / 700)))
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        x, y = W - tw - int(0.015 * W), H - int(0.03 * H)
        cv2.rectangle(out, (x - 6, y - tht - 6), (x + tw + 6, y + 6), (0, 0, 0), -1)
        cv2.putText(out, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 220, 90), th, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------ camera / HUD / stats
    def camera_target(self) -> np.ndarray:
        return np.array([-1.0, 0.0, self.cfg.mat_z + 2.3])

    def camera_preset(self) -> CameraPreset:
        return CameraPreset(azimuth=90.0, elevation=-1.0, distance=18.0, tau_s=0.3)

    def job_hud_lines(self) -> list[str]:
        avg = self.sum_height / self.n_bounces if self.n_bounces else 0.0
        lines = [
            f"BEST {self.best_height:.2f} mm = {self.best_height / BODY_LENGTH_MM:.2f} body lengths"
            f"   last {self.last_height:.2f} mm   avg {avg:.2f} mm",
            f"STREAK {self.streak} (best {self.best_streak})   crash landings {self.n_crashes}   "
            f"wobbly {self.n_wobbly}   off the trampoline {self.n_off}   "
            f"BACKFLIPS {self.n_tricks}/{self.n_trick_tries}   missteps {self.n_missteps}",
            f"mat: {self.cfg.n_springs} tendon springs, deflection {-self.mat_q():.2f} mm "
            f"(max {-self.mat_min_seen:.2f})   next jump {1000 * self.fire_delay_s():.0f} ms "
            f"after touchdown",
            "(real Jump action on a spring mat; airborne attitude + steering aid: "
            "engineered wing emulation)",
        ]
        if self.flash:
            lines.append(self.flash)
        return lines

    def job_stats(self) -> dict[str, Any]:
        n = self.n_bounces
        return {
            "bounces": n, "best_height_mm": round(self.best_height, 3),
            "best_body_lengths": round(self.best_height / BODY_LENGTH_MM, 3),
            "avg_height_mm": round(self.sum_height / n, 3) if n else 0.0,
            "streak": self.streak, "best_streak": self.best_streak, "crashes": self.n_crashes,
            "wobbly": self.n_wobbly, "off": self.n_off, "tricks": self.n_tricks,
            "trick_tries": self.n_trick_tries, "missteps": self.n_missteps,
            "jumps": self.n_jumps, "max_mat_deflection_mm": round(-self.mat_min_seen, 3),
            "max_steer_bw": round(self.max_steer_bw, 3),
            "max_att_torque": round(self.max_att_torque, 3), "phase": self.phase,
        }
