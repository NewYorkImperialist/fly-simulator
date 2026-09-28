"""JUMP ROPE FLY: the fly skips rope forever (docs/JOBS.md, "jump_rope").

A schoolyard. Two turner posts with **driven cranks** (kinematic, labelled) turn a long
rope over the fly; the fly jumps it every turn with the **real ``Jump`` action**
(short-mode escape jump, docs/ACTIONS.md), fired from the rope's phase.

* **The rope (kinematic, engineered, labelled).** A chain of ``n_seg`` capsules on
  mocap bodies laid along a driven curve: a loop of radius ``rope_R`` turning about
  the axis between the two cranks (y), the middle trailing the cranks a little, faster
  through the bottom than over the top (``speed_bump``: a swinging rope gains speed as
  it falls). At fly scale a real rope turned at 1-2 turns/s would just hang: the
  centrifugal acceleration w^2 R is ~0.1-0.2 g, so the loop shape is driven, not
  simulated. The capsules **collide with the fly** (contact kind "fly"): a rope that
  catches a leg is a real contact, the fly stumbles or is knocked over.
* **The skip** is the real ``Jump`` (crouch -> mid-leg TTM stroke -> ballistic flight
  -> landing) in short mode, fired ``lead_s`` before the rope reaches the fly (from the
  rope's phase and the fly's own position) plus the fly's timing error (our behaviour
  model: ``jitter_ms``, growing with the rope speed, and occasional missteps). The
  airborne attitude stabiliser and the small horizontal "wing steering" force of the
  trampoline's ``BounceJump`` are used (engineered wing emulation, labelled; horizontal
  only, never lift). The short-mode stroke pushes the fly back ~0.5 mm per jump, so
  between turns it shuffles back onto its spot (real walking, ``Steering``).
* **Trips.** Any rope-fly contact during a turn is a trip: the streak ends, the turners
  stop the rope at the top, the fly stumbles (or falls: the base class's explicit,
  counted reset after 2.5 s down), walks back to its spot, and the rope starts again
  (counted recovery). The rope speed ramps up with the streak (``period_step`` per
  skip, down to ``period_min``) and falls back to ``period0`` after a trip.

Counters: skips (the work counter), streak, best streak, trips (stumbles / knocked
down), recoveries, pauses (the fly drifted off its spot), rope speed (rpm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco as mj
import numpy as np

from fly_simulator.jobs import jump_rope_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import contact_kwargs, quat_axis_angle, spot_or_directional
from fly_simulator.jobs.registry import register_job
from fly_simulator.jobs.taste_tester import DIGITS, SEGS
from fly_simulator.jobs.trampoline import BounceJump, MatStance

P = "rope/"


def _quat_z_to(v: np.ndarray) -> np.ndarray:
    """(w, x, y, z) of the shortest rotation taking +z onto ``v`` (unit)."""
    z = np.array([0.0, 0.0, 1.0])
    c = float(z @ v)
    if c < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    ax = np.cross(z, v)
    q = np.array([1.0 + c, ax[0], ax[1], ax[2]])
    return q / np.linalg.norm(q)


def _wrap(a: float) -> float:
    """Angle to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi if a != math.pi else math.pi


@dataclass
class JumpRopeConfig(JobConfig):
    # --- the rope (mm) -----------------------------------------------------------
    axis_x: float = -0.9  # the rope's turning axis (x): under the middle of the footprint
    mark_dx: float = 1.45  # the fly's spot (thorax) is this far ahead of the axis (the
    # tarsi span ~3 mm, centred ~1.4 mm behind the thorax)
    rope_len: float = 16.0  # crank to crank (along y)
    rope_R: float = 7.0  # loop radius in the middle
    crank_r: float = 0.7  # crank radius at the ends
    rope_r: float = 0.09  # rope (capsule) radius
    clearance: float = 0.04  # the rope's bottom above the ground
    n_seg: int = 40
    lag: float = 0.28  # rad: the middle trails the cranks (at period0)
    speed_bump: float = 0.65  # dphi/dt = w (1 + a cos phi): faster through the bottom
    rope_solref: float = 0.004  # rope-fly contact time constant (s): softer than FlyGym's
    # --- the turners (driven) ---------------------------------------------------------
    period0: float = 0.8  # s per turn at streak 0
    period_step: float = 0.012  # shorter per skip in the streak
    period_min: float = 0.45
    spin_up_s: float = 0.3  # turners start from rest over this long
    ready_s: float = 0.8  # the fly stands on its spot this long before the rope starts
    # --- the skip (our behaviour model) ----------------------------------------------
    lead_s: float = 0.036  # jump trigger -> the rope under the middle of the footprint (the
    # feet are above 0.25 mm from ~21 to ~58 ms after the trigger, measured on the floor)
    jitter_ms: float = 3.0  # N(0, jitter) per jump at period0, x (period0 / period)
    misstep_p: float = 0.03
    misstep_ms: float = 22.0  # early or late by this much
    att_hz: float = 15.0  # airborne attitude stabiliser (engineered, 0 = off)
    att_max: float = 2.0  # uN*mm
    steer_max_bw: float = 0.3  # horizontal wing-steering force cap (engineered, 0 = off)
    lean_gain: float = 0.05  # rad/mm: stabiliser's up vector tilts away from the spot
    shuffle_speed: float = 0.6  # walking back onto the spot between turns
    max_off: float = 2.2  # mm off the spot (along x) -> pause the rope, walk back
    # --- looks / presentation --------------------------------------------------------
    slowmo: float = 0.3  # slow motion while the rope passes under (edit; 1 = off)
    shadows: bool = True
    captions: bool = True


@register_job
class JumpRopeJob(EternalJob):
    name = "jump_rope"
    znear = 0.05
    title = "JUMP ROPE FLY"
    tagline = "the fly skips rope forever"
    work_label = "skips"
    config_cls = JumpRopeConfig
    required_names = (P + "seg0", P + "wheel0", P + "wheel1", P + "ground")

    def __init__(self, cfg: JumpRopeConfig | None = None) -> None:
        super().__init__(cfg)
        c = self.cfg
        self.rng = np.random.default_rng(c.seed + 17)
        self.mark = np.array([c.axis_x + c.mark_dx, 0.0])
        self.zc = c.rope_R + c.rope_r + c.clearance
        # counters
        self.n_skips = 0
        self.streak = 0
        self.best_streak = 0
        self.n_trips = 0
        self.n_stumbles = 0  # trips the fly stayed up after
        self.n_knocked = 0  # trips that ended in a fall (explicit reset)
        self.n_recoveries = 0
        self.n_pauses = 0
        self.n_jumps = 0
        self.n_missteps = 0
        self.n_turns = 0
        self.last_error_ms = 0.0
        # rope state
        self.phi = math.pi  # the middle's phase (0 = bottom, decreasing = forward skip)
        self.spin = 0.0  # 0..1 turners' effort
        self.rope = "parked"  # parked | starting | turning | stopping
        self.period = c.period0
        self._w = self._omega(self.period)
        # fly state
        self.mode = "stance"  # stance | jump | walk | down
        self.phase = "ready"  # ready | turning | tripped | paused
        self.t_phase = 0.0
        self._t_mode = 0.0
        self._fired = False
        self._err_s = 0.0
        self._contact_turn = False
        self._counted_turn = False
        self._after_trip = False
        self._reset_since_trip = False
        self._psi_prev = math.pi
        self._jump_action: BounceJump | None = None
        self._caption = ("", -1e9)
        self._scale = 1.0
        self._last_scale = 1.0
        self.max_steer_bw = 0.0
        self.message = ""

    # ------------------------------------------------------------ geometry
    def _omega(self, period: float) -> float:
        a = self.cfg.speed_bump
        return 2 * math.pi / (period * math.sqrt(1 - a * a))

    def rpm(self) -> float:
        return 60.0 / self.period

    def rope_points(self, phi: float, lag: float) -> np.ndarray:
        """(n_seg + 1, 3) points along the rope for the middle's phase ``phi``."""
        c = self.cfg
        s = np.linspace(-1.0, 1.0, c.n_seg + 1)
        r = c.crank_r + (c.rope_R - c.crank_r) * (1.0 - s * s)
        ph = phi - lag * s * s  # the cranks lead (phi decreases), the middle trails
        return np.stack([c.axis_x + r * np.sin(ph), s * c.rope_len / 2, self.zc - r * np.cos(ph)], -1)

    def _lag(self) -> float:
        return self.cfg.lag * self.spin * (self.cfg.period0 / self.period)

    # ------------------------------------------------------------ build
    def configure_app(self, app_cfg) -> None:
        super().configure_app(app_cfg)

    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        self._materials(spec)
        # --- the rope: capsules on mocap bodies (collide with the fly only) ---------
        pts = self.rope_points(0.0, 0.0)
        seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        half = 0.5 * float(seg_len.max()) * 1.04
        rk = contact_kwargs("fly", friction=0.6)
        rk["solref"] = (c.rope_solref, 1.0)
        park = self.rope_points(math.pi, 0.0)
        for i in range(c.n_seg):
            mid = 0.5 * (park[i] + park[i + 1])
            b = wb.add_body(name=P + f"seg{i}", mocap=True, pos=tuple(mid))
            mat = P + ("rope_a" if (i // 2) % 2 == 0 else "rope_b")
            b.add_geom(name=P + f"seg{i}_g", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(c.rope_r, half, 0),
                       material=mat, mass=0.0, **rk)
        # --- the turner posts with their driven crank wheels -----------------------
        for k, sgn in enumerate((-1.0, 1.0)):
            y_end = sgn * c.rope_len / 2
            py = y_end + sgn * 0.75
            hpost = self.zc + 1.6
            wb.add_geom(name=P + f"post{k}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.32, hpost / 2, 0),
                        pos=(c.axis_x, py, hpost / 2), material=P + f"post{k}", **vis)
            wb.add_geom(name=P + f"post{k}_cap", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.48, 0, 0),
                        pos=(c.axis_x, py, hpost + 0.2), material=P + "yellow", **vis)
            wb.add_geom(name=P + f"post{k}_base", type=mj.mjtGeom.mjGEOM_BOX, size=(1.0, 0.8, 0.45),
                        pos=(c.axis_x, py, 0.45), material=P + "motor", **vis)
            # the axle from the post to the wheel
            wb.add_geom(name=P + f"axle{k}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.09, 0.35, 0),
                        pos=(c.axis_x, y_end + sgn * 0.4, self.zc),
                        quat=quat_axis_angle((1, 0, 0), math.pi / 2), material=P + "steel", **vis)
            # "DRIVEN" sign on the post (faces the camera side, -y / +x)
            A.add_mesh(spec, P + f"sign{k}_mesh", A.panel_mesh(2.0, 1.0, 0.05))
            wb.add_geom(name=P + f"sign{k}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + f"sign{k}_mesh",
                        pos=(c.axis_x + 0.9, py, 1.55), quat=A.panel_quat((1, -0.25, 0)),
                        material=P + f"sign{k}", **vis)
            w = wb.add_body(name=P + f"wheel{k}", mocap=True, pos=(c.axis_x, y_end + sgn * 0.12, self.zc))
            A.add_mesh(spec, P + f"wheel{k}_mesh", A.wheel_mesh(c.crank_r + 0.35, 0.12))
            w.add_geom(name=P + f"wheel{k}_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + f"wheel{k}_mesh",
                       quat=quat_axis_angle((1, 0, 0), math.pi / 2), material=P + "wheel", **vis)
            # the crank handle at the rope's end (at -z of the wheel body)
            w.add_geom(name=P + f"handle{k}", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.14, 0.32, 0),
                       pos=(0.0, -sgn * 0.3, -c.crank_r), quat=quat_axis_angle((1, 0, 0), math.pi / 2),
                       material=P + "handle", **vis)
        self._add_schoolyard(spec, vis)
        self._add_scoreboard(spec, vis)
        self._add_lights(spec)

    def _materials(self, spec) -> None:
        c = self.cfg
        T = A.add_textured_material
        spec.add_material(name=P + "rope_a", rgba=(0.95, 0.2, 0.35, 1), specular=0.3, shininess=0.3)
        spec.add_material(name=P + "rope_b", rgba=(1.0, 0.92, 0.3, 1), specular=0.3, shininess=0.3)
        spec.add_material(name=P + "steel", rgba=(0.62, 0.64, 0.68, 1), specular=0.8, shininess=0.8)
        spec.add_material(name=P + "yellow", rgba=(1.0, 0.82, 0.1, 1), specular=0.4)
        spec.add_material(name=P + "motor", rgba=(0.2, 0.55, 0.9, 1), specular=0.5, shininess=0.5)
        spec.add_material(name=P + "handle", rgba=(0.15, 0.75, 0.35, 1), specular=0.4)
        spec.add_material(name=P + "white", rgba=(0.95, 0.95, 0.93, 1), specular=0.3)
        spec.add_material(name=P + "wood", rgba=(0.62, 0.40, 0.22, 1), specular=0.2)
        spec.add_material(name=P + "orange", rgba=(0.95, 0.48, 0.12, 1), specular=0.3)
        spec.add_material(name=P + "leaf", rgba=(0.25, 0.62, 0.25, 1), specular=0.1)
        spec.add_material(name=P + "seg_on", rgba=(0.98, 0.98, 0.9, 1), emission=0.7)
        spec.add_material(name=P + "seg_off", rgba=(0.12, 0.22, 0.16, 1), emission=0.0)
        A.add_texture(spec, P + "tex_ground",
                      A.playground_texture(90.0, 80.0, (-5.0, -10.0), c.seed, 16.0))
        T(spec, P + "ground", P + "tex_ground", rgba=(1, 1, 1, 1), specular=0.05, shininess=0.05)
        A.add_texture(spec, P + "tex_wall", A.school_wall_texture(c.seed))
        T(spec, P + "wall", P + "tex_wall", rgba=(1, 1, 1, 1), specular=0.05, emission=0.1)
        A.add_texture(spec, P + "tex_sky", A.sky_texture(c.seed))
        T(spec, P + "sky", P + "tex_sky", rgba=(1, 1, 1, 1), emission=0.55, specular=0.0)
        A.add_texture(spec, P + "tex_board", A.scoreboard_texture())
        T(spec, P + "board", P + "tex_board", rgba=(1, 1, 1, 1), emission=0.2, specular=0.05)
        A.add_texture(spec, P + "tex_fence", A.fence_texture())
        T(spec, P + "fence", P + "tex_fence", rgba=(1, 1, 1, 0.55), specular=0.3, texrepeat=(6, 2))
        A.add_texture(spec, P + "tex_wheel", A.wheel_texture())
        T(spec, P + "wheel", P + "tex_wheel", rgba=(1, 1, 1, 1), specular=0.4)
        for k, cols in enumerate((((0.95, 0.25, 0.2), (0.98, 0.95, 0.9)),
                                  ((0.2, 0.45, 0.95), (0.98, 0.95, 0.9)))):
            A.add_texture(spec, P + f"tex_post{k}", A.post_texture(*cols))
            T(spec, P + f"post{k}", P + f"tex_post{k}", rgba=(1, 1, 1, 1), specular=0.4)
            A.add_texture(spec, P + f"tex_sign{k}",
                          A.sign_texture((f"TURNER {'AB'[k]}", "DRIVEN CRANK")))
            T(spec, P + f"sign{k}", P + f"tex_sign{k}", rgba=(1, 1, 1, 1), emission=0.15)

    def _add_schoolyard(self, spec, vis) -> None:
        c = self.cfg
        wb = spec.worldbody
        # the painted playground (the fly stands on FlyGym's own plane, hidden in
        # on_attach; this slab's top is flush with it)
        A.add_mesh(spec, P + "ground_mesh", A.slab_mesh(90.0, 80.0, 0.1))
        wb.add_geom(name=P + "ground", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "ground_mesh",
                    pos=(c.axis_x + 5.0, 10.0, 0.0), material=P + "ground", **vis)
        # the school wall (faces -y) and the sky behind it
        A.add_mesh(spec, P + "wall_mesh", A.panel_mesh(66.0, 22.0, 0.4))
        wb.add_geom(name=P + "wall", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "wall_mesh",
                    pos=(-4.0, 24.0, 11.0), quat=A.panel_quat((0, -1, 0)), material=P + "wall", **vis)
        A.add_mesh(spec, P + "sky_mesh", A.panel_mesh(260.0, 90.0, 0.5))
        wb.add_geom(name=P + "sky", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "sky_mesh",
                    pos=(-10.0, 75.0, 30.0), quat=A.panel_quat((0, -1, 0)), material=P + "sky", **vis)
        wb.add_geom(name=P + "sky_w", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "sky_mesh",
                    pos=(-75.0, 0.0, 30.0), quat=A.panel_quat((1, 0, 0)), material=P + "sky", **vis)
        # a chain-link fence on the -x side (partly transparent)
        A.add_mesh(spec, P + "fence_mesh", A.panel_mesh(46.0, 7.0, 0.05))
        wb.add_geom(name=P + "fence", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "fence_mesh",
                    pos=(-26.0, 1.0, 3.5), quat=A.panel_quat((1, 0, 0)), material=P + "fence", **vis)
        for j, y in enumerate(np.linspace(-22.0, 24.0, 7)):
            wb.add_geom(name=P + f"fpost{j}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.12, 3.6, 0),
                        pos=(-26.0, float(y), 3.6), material=P + "steel", **vis)
        # a basketball hoop by the fence, benches by the wall, a ball, shrubs
        hm = A.hoop_meshes(9.0)
        for part, mat in (("pole", "steel"), ("ring", "orange")):
            A.add_mesh(spec, P + f"hoop_{part}_mesh", hm[part])
            wb.add_geom(name=P + f"hoop_{part}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + f"hoop_{part}_mesh",
                        pos=(-22.0, 14.0, 0.0), quat=quat_axis_angle((0, 0, 1), math.pi), material=P + mat, **vis)
        wb.add_geom(name=P + "hoop_board", type=mj.mjtGeom.mjGEOM_BOX, size=(0.08, 1.6, 1.1),
                    pos=(-21.2, 14.0, 10.0), material=P + "white", **vis)
        for j, x in enumerate((-14.0, 4.0, 14.0)):
            wb.add_geom(name=P + f"bench{j}", type=mj.mjtGeom.mjGEOM_BOX, size=(2.6, 0.5, 0.08),
                        pos=(x, 20.5, 1.1), material=P + "wood", **vis)
            for dx in (-2.2, 2.2):
                wb.add_geom(name=P + f"bench{j}_leg{int(dx > 0)}", type=mj.mjtGeom.mjGEOM_BOX,
                            size=(0.1, 0.45, 0.5), pos=(x + dx, 20.5, 0.5), material=P + "steel", **vis)
        wb.add_geom(name=P + "ball", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.9, 0, 0),
                    pos=(-9.0, 9.0, 0.9), material=P + "orange", **vis)
        shr = [A.shrub_mesh((x, 22.6), 1.3, seed=int(x + 40)) for x in np.arange(-30.0, 26.0, 5.0)]
        A.add_mesh(spec, P + "shrubs_mesh", A.merge(*shr))
        wb.add_geom(name=P + "shrubs", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "shrubs_mesh",
                    material=P + "leaf", **vis)
        sky = spec.texture("skybox")
        if sky is not None:
            sky.rgb1 = (0.40, 0.66, 0.98)
            sky.rgb2 = (0.82, 0.92, 1.0)

    def _add_scoreboard(self, spec, vis) -> None:
        """The chalkboard on the wall: STREAK (3 digits), BEST (3), RPM (3)."""
        wb = spec.worldbody
        bx, by, bz, hw, hh = -9.0, 23.7, 6.0, 4.0, 2.0
        A.add_mesh(spec, P + "board_mesh", A.panel_mesh(2 * hw, 2 * hh, 0.1))
        wb.add_geom(name=P + "board", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "board_mesh",
                    pos=(bx, by, bz), quat=A.panel_quat((0, -1, 0)), material=P + "board", **vis)
        self._seg_names: list[list[list[str]]] = []
        dw, dh, st, gap = 0.46, 0.34, 0.06, 0.66
        for di, frac in enumerate((0.36, 0.58, 0.80)):
            zc = bz + hh - 2 * hh * frac
            x_right = bx - hw + 2 * hw * 0.86
            disp = []
            for k in range(3):
                xd = x_right - (2 - k) * gap
                segs = []
                for sname, (dx, dz, horiz) in SEGS.items():
                    nm = f"{P}seg{di}_{k}_{sname}"
                    size = (dw / 2 - 0.03, 0.01, st) if horiz else (st, 0.01, dh / 2 - 0.03)
                    wb.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_BOX, size=size,
                                pos=(xd + dx * dw, by - 0.07, zc + dz * dh), material=P + "seg_off", **vis)
                    segs.append(nm)
                disp.append(segs)
            self._seg_names.append(disp)

    def _add_lights(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        spec.visual.headlight.ambient = (0.34, 0.34, 0.34)
        spec.visual.headlight.diffuse = (0.30, 0.30, 0.30)
        spec.visual.headlight.specular = (0.1, 0.1, 0.1)
        tgt = np.array([c.axis_x, 0.0, 2.0])
        sun = np.array([c.axis_x + 16.0, -18.0, 34.0])
        wb.add_light(name=P + "sun", type=spot_or_directional(c.shadows), pos=tuple(sun), dir=tuple(tgt - sun),
                     diffuse=(0.72, 0.70, 0.64), specular=(0.35, 0.35, 0.35), cutoff=40.0, exponent=0.3,
                     castshadow=bool(c.shadows))
        wb.add_light(name=P + "fill", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(0, 0, 30),
                     dir=(-0.3, 0.6, -1.0), diffuse=(0.22, 0.24, 0.30), specular=(0.05, 0.05, 0.05),
                     castshadow=False)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        m, s = self.sim.model, self.session
        c = self.cfg
        gp = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "ground_plane")
        if gp >= 0:  # the fly's floor stays (physics); the playground slab draws the ground
            m.geom_group[gp] = 3
            m.geom_rgba[gp, 3] = 0.0
        self.acts = s.actions
        self.body = s.actions.body
        self.seg_mocap = np.array([int(m.body_mocapid[m.body(P + f"seg{i}").id]) for i in range(c.n_seg)])
        self.rope_geom = np.zeros(m.ngeom, bool)
        for i in range(c.n_seg):
            self.rope_geom[m.geom(P + f"seg{i}_g").id] = True
        self.tarsi = np.array([m.body(f"{self.fly_name}/{leg}_tarsus5").id
                               for leg in ("lf", "lm", "lh", "rf", "rm", "rh")])
        self.wheel_mocap = [int(m.body_mocapid[m.body(P + f"wheel{k}").id]) for k in range(2)]
        root = self.sim.thorax_body_id
        self.fly_body = np.zeros(m.nbody, bool)
        self.fly_body[np.flatnonzero(m.body_rootid == root)] = True
        self.seg_gid = [[[m.geom(n).id for n in dig] for dig in disp] for disp in self._seg_names]
        self.matid = {n: m.material(P + n).id for n in ("seg_on", "seg_off")}
        st = getattr(s, "STATIONARY_ACTIONS", None)
        if st is not None and MatStance.name not in st:
            s.STATIONARY_ACTIONS = (*st, MatStance.name)
        self._start_ready()
        self._place_rope()
        self._update_board()

    # ------------------------------------------------------------ rope kinematics
    def _place_rope(self) -> None:
        d = self.sim.data
        pts = self.rope_points(self.phi, self._lag())
        mids = 0.5 * (pts[1:] + pts[:-1])
        dirs = pts[1:] - pts[:-1]
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-9)
        d.mocap_pos[self.seg_mocap] = mids
        for i, k in enumerate(self.seg_mocap):
            d.mocap_quat[k] = _quat_z_to(dirs[i])
        # the crank wheels turn with the rope's ends (the cranks lead by the lag)
        phi_e = self.phi - self._lag()
        for k in range(2):
            d.mocap_quat[self.wheel_mocap[k]] = quat_axis_angle((0, 1, 0), -phi_e)

    def _advance_rope(self, dt: float) -> None:
        c = self.cfg
        if self.rope == "parked":
            return
        if self.rope == "starting":
            self.spin = min(1.0, self.spin + dt / max(c.spin_up_s, 1e-3))
            if self.spin >= 1.0:
                self.rope = "turning"
        prev = _wrap(self.phi)
        self.phi -= self.spin * self._w * (1.0 + c.speed_bump * math.cos(self.phi)) * dt
        now = _wrap(self.phi)
        # the top (psi = +-pi) is crossed when the wrapped angle jumps up
        if now > prev + math.pi:
            self._at_top()

    def _at_top(self) -> None:
        """The rope goes over the top: a new turn (the speed is set here)."""
        self.n_turns += 1
        self._fired = False
        self._contact_turn = False
        self._counted_turn = False
        if self.rope == "stopping":
            self.rope = "parked"
            self.phi = math.pi
            self.spin = 0.0
            return
        self.period = max(self.cfg.period_min, self.cfg.period0 - self.cfg.period_step * self.streak)
        self._w = self._omega(self.period)

    def _start_rope(self) -> None:
        self.phi = math.pi
        self.spin = 0.0
        self.rope = "starting"
        self.period = max(self.cfg.period_min, self.cfg.period0 - self.cfg.period_step * self.streak)
        self._w = self._omega(self.period)
        self._fired = False
        self._contact_turn = False
        self._counted_turn = False

    def feet_centre_x(self) -> float:
        """x of the middle of the fly's footprint (the tarsi span ~3 mm, centred ~1.2 mm
        behind the thorax): the rope must pass under all of it while the fly is up."""
        x = self.sim.data.xpos[self.tarsi, 0]
        return 0.5 * float(x.max() + x.min())

    def pass_angle(self) -> float:
        """The middle's phase at which the rope is under the middle of the footprint."""
        d = self.feet_centre_x() - self.cfg.axis_x
        return math.asin(max(-0.95, min(0.95, d / self.cfg.rope_R)))

    def time_to_pass(self) -> float:
        """Seconds until the rope (turning at full speed) is under the fly."""
        a = self.cfg.speed_bump
        psi = _wrap(self.phi)
        psp = self.pass_angle()
        k = math.sqrt((1 - a) / (1 + a))
        g = 2.0 / math.sqrt(1 - a * a)

        def F(x):
            x = max(-math.pi + 1e-6, min(math.pi - 1e-6, x))
            return g * math.atan(k * math.tan(x / 2))

        if psi >= psp:
            return (F(psi) - F(psp)) / self._w
        return (F(math.pi) - F(psp) + F(psi) - F(-math.pi)) / self._w

    def rope_fly_contact(self) -> bool:
        d, m = self.sim.data, self.sim.model
        n = d.ncon
        if n == 0:
            return False
        g = d.contact.geom[:n]
        r1, r2 = self.rope_geom[g[:, 0]], self.rope_geom[g[:, 1]]
        f1, f2 = self.fly_body[m.geom_bodyid[g[:, 0]]], self.fly_body[m.geom_bodyid[g[:, 1]]]
        return bool(np.any((r1 & f2) | (r2 & f1)))

    # ------------------------------------------------------------ fly helpers
    def _lean_up(self) -> np.ndarray:
        c = self.cfg
        p = self.sim.thorax_position()
        h = c.lean_gain * (p[:2] - self.mark)
        n = float(np.linalg.norm(h))
        if n > 0.25:
            h *= 0.25 / n
        return np.array([h[0], h[1], 1.0])

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self._t_mode = self.sim.time
        if mode == "stance":
            self.steering.set(None, 0.0)
            self.acts.trigger(MatStance(), source="job")
        elif mode == "walk":
            self.acts.cancel()

    def _go(self, phase: str) -> None:
        self.phase = self.state = phase
        self.t_phase = self.sim.time

    def _start_ready(self) -> None:
        self._go("ready")
        self.rope = "parked"
        self.phi = math.pi
        self.spin = 0.0
        self._set_mode("stance")

    def _fire(self, err_s: float) -> None:
        c = self.cfg
        j = BounceJump(mode="short", att_hz=c.att_hz, att_max=c.att_max, home_xy=tuple(self.mark),
                       steer_max_bw=c.steer_max_bw, up_fn=self._lean_up if c.lean_gain else None)
        self._jump_action = j
        self.acts.trigger(j, source="job")
        self.n_jumps += 1
        self.mode = "jump"
        self._t_mode = self.sim.time
        self._fired = True
        self.last_error_ms = 1000.0 * err_s

    def _draw_error(self) -> float:
        c = self.cfg
        sig = c.jitter_ms * (c.period0 / self.period) / 1000.0
        e = float(self.rng.normal(0.0, sig)) if sig > 0 else 0.0
        if c.misstep_p > 0 and self.rng.random() < c.misstep_p:
            self.n_missteps += 1
            e += (1 if self.rng.random() < 0.5 else -1) * c.misstep_ms / 1000.0
        return e

    def off_mark(self) -> np.ndarray:
        return self.mark - self.sim.thorax_position()[:2]

    def _caption_say(self, text: str) -> None:
        self._caption = (text, self.run_time())
        self.message = text

    # ------------------------------------------------------------ job logic
    def update(self) -> None:
        c = self.cfg
        sim = self.sim
        t = sim.time
        dt = c.update_every_steps * sim.timestep
        self._advance_rope(dt)
        self._place_rope()
        contact = self.rope_fly_contact() if self.rope != "parked" else False
        self._fly_logic(t)
        if self.phase == "turning":
            psi = _wrap(self.phi)
            psp = self.pass_angle()
            if contact and not self._contact_turn:
                self._contact_turn = True
                self._trip(t)
            elif (not self._counted_turn and not self._contact_turn and self.rope == "turning"
                  and psp - 0.4 > psi > -math.pi / 2 and self._psi_prev >= psi):
                self._counted_turn = True
                self._skip()
            elif self.rope == "turning" and not self._fired and self.mode in ("stance", "walk") \
                    and psi > psp:
                if self.time_to_pass() <= c.lead_s + self._err_s:
                    self._fire(self._err_s)
            # the fly wandered off its spot: pause (the turners stop at the top)
            off = self.off_mark()
            if abs(off[0]) > c.max_off or abs(off[1]) > 2.5 * c.max_off:
                self.n_pauses += 1
                self.rope = "stopping"
                self._caption_say("WAIT UP! (back to the spot)")
                self._go("paused")
            self._psi_prev = psi
        elif self.phase in ("tripped", "paused", "ready"):
            ready = (self.rope == "parked" and self.mode == "stance" and not self.fly_down()
                     and sim.tilt_deg() < 25.0 and t - self._t_mode >= c.ready_s
                     and float(np.linalg.norm(self.off_mark())) < 0.7)
            if ready:
                if self._after_trip:
                    self.n_recoveries += 1
                    if not self._reset_since_trip:
                        self.n_stumbles += 1  # it stayed on its feet
                    self._after_trip = False
                self._start_rope()
                self._err_s = self._draw_error()
                self._go("turning")
                self._psi_prev = _wrap(self.phi)
        if self.message and self.run_time() - self._caption[1] > 2.0:
            self.message = ""

    def _fly_logic(self, t: float) -> None:
        c = self.cfg
        if self.fly_down():
            if self.mode != "down":
                self.mode = "down"
                self.steering.set(None, 0.0)
            return
        if self.mode == "down":  # righted itself
            self._set_mode("stance")
            return
        if self.mode == "jump":
            j = self._jump_action
            if self.acts.action is not j and self.acts._pending is not j:
                self.max_steer_bw = max(self.max_steer_bw, j.max_steer_bw)
                off = self.off_mark()
                h = self.sim.heading()
                fwd = np.array([math.cos(h), math.sin(h)])
                time_left = self.time_to_pass() - c.lead_s if self.rope == "turning" else 9.0
                if float(np.linalg.norm(off)) > 0.2 and (off @ fwd > 0.1 or self.phase != "turning") \
                        and time_left > 0.25:
                    self._set_mode("walk")
                else:
                    self._set_mode("stance")
        elif self.mode == "walk":
            off = self.off_mark()
            h = self.sim.heading()
            fwd = np.array([math.cos(h), math.sin(h)])
            time_left = self.time_to_pass() - c.lead_s if self.rope == "turning" else 9.0
            self.steering.aim_at(self.mark, c.shuffle_speed)
            done = float(np.linalg.norm(off)) < 0.15 or (self.phase == "turning" and off @ fwd < 0.03)
            if done or time_left < 0.15 or t - self._t_mode > 4.0:
                self._set_mode("stance")
        elif self.mode == "stance":
            if self.phase != "turning" and t - self._t_mode > 0.4 \
                    and float(np.linalg.norm(self.off_mark())) > 0.6:
                self._set_mode("walk")

    def _skip(self) -> None:
        self.n_skips += 1
        self.streak += 1
        self.best_streak = max(self.best_streak, self.streak)
        self.add_work(1)
        self._err_s = self._draw_error()
        if self.streak in (10, 25, 50, 100, 200, 500, 1000) or (self.streak % 100 == 0 and self.streak):
            self._caption_say(f"{self.streak} IN A ROW!")
        self._update_board()

    def _trip(self, t: float) -> None:
        self.n_trips += 1
        lost = self.streak
        self.streak = 0
        self.rope = "stopping"
        self._after_trip = True
        self._reset_since_trip = False
        self.session.log_event("jump_rope_trip", streak=lost, rpm=round(self.rpm(), 1),
                               error_ms=round(self.last_error_ms, 1))
        self._caption_say(f"TRIPPED!  (streak {lost})")
        self._go("tripped")
        self._update_board()

    def _update_board(self) -> None:
        if not hasattr(self, "seg_gid"):
            return
        m = self.sim.model
        vals = (int(self.streak) % 1000, int(self.best_streak) % 1000, int(round(self.rpm())) % 1000)
        for di, v in enumerate(vals):
            for k in range(3):
                dg = (v // 10 ** (2 - k)) % 10
                show = v >= 10 ** (2 - k) or k == 2
                lit = DIGITS[dg] if show else ""
                for sname, g in zip(SEGS, self.seg_gid[di][k]):
                    m.geom_matid[g] = self.matid["seg_on" if sname in lit else "seg_off"]

    def on_reset(self) -> None:
        # a trip that ended in a fall: the base class reset the fly explicitly
        if self._after_trip and not self._reset_since_trip:
            self.n_knocked += 1
            self._reset_since_trip = True
        self._jump_action = None
        self._scale = 1.0
        self._start_ready()

    def reset_props(self) -> None:
        self.period = self.cfg.period0
        self._place_rope()
        self._update_board()

    def after_physics(self) -> None:
        super().after_physics()

    # ------------------------------------------------------------ edit effects
    def time_scale(self, present_dt: float) -> float:
        """Slow motion while the rope passes under the fly (an *edit effect*: fewer
        physics steps per displayed frame; the physics is unchanged). At fly scale
        the jump is ~50 ms in the air: 1-2 frames at real time."""
        psi = _wrap(self.phi)
        near = self.rope != "parked" and -0.7 < psi < 0.9
        tgt = min(max(float(self.cfg.slowmo), 0.02), 1.0) if near else 1.0
        if tgt < self._scale:
            self._scale = tgt
        else:
            self._scale += (1.0 - math.exp(-present_dt / 0.08)) * (tgt - self._scale)
            if self._scale > 0.98:
                self._scale = 1.0
        self._last_scale = self._scale
        return self._scale

    def post_process(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Screen labels: the slow motion (edit, not physics) and captions."""
        cap, t_cap = self._caption
        show = self.cfg.captions and cap and 0 <= t - t_cap < 1.6
        if self._last_scale >= 1.0 and not show:
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
        if show:
            f2, t2 = fs * 2.0, th * 2 + 1
            (tw, tht), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_DUPLEX, f2, t2)
            x, y = (W - tw) // 2, int(0.13 * H) + tht
            cv2.putText(out, cap, (x + 2, y + 2), cv2.FONT_HERSHEY_DUPLEX, f2, (30, 0, 40), t2 + 2, cv2.LINE_AA)
            cv2.putText(out, cap, (x, y), cv2.FONT_HERSHEY_DUPLEX, f2, (255, 230, 60), t2, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------ camera / HUD / stats
    def camera_target(self) -> np.ndarray:
        return np.array([self.cfg.axis_x + 0.3, 0.0, 4.4])

    def camera_preset(self) -> CameraPreset:
        return CameraPreset(azimuth=116.0, elevation=-9.0, distance=22.0, tau_s=0.5)

    def job_hud_lines(self) -> list[str]:
        return [
            f"STREAK {self.streak} (best {self.best_streak})   rope {self.rpm():.0f} rpm   "
            f"[{self.phase} / rope {self.rope} / fly {self.mode}]",
            f"trips {self.n_trips} (stumbled {self.n_stumbles}, knocked down {self.n_knocked})   "
            f"recoveries {self.n_recoveries}   pauses {self.n_pauses}   missteps {self.n_missteps}   "
            f"last timing {self.last_error_ms:+.0f} ms",
            "(real Jump action; rope + cranks: driven kinematic curve with real contacts; "
            "airborne attitude + steering aid: engineered wing emulation)",
        ] + ([self.message] if self.message else [])

    def job_stats(self) -> dict[str, Any]:
        return {
            "skips": self.n_skips, "streak": self.streak, "best_streak": self.best_streak,
            "trips": self.n_trips, "stumbles": self.n_stumbles, "knocked_down": self.n_knocked,
            "recoveries": self.n_recoveries, "pauses": self.n_pauses, "jumps": self.n_jumps,
            "missteps": self.n_missteps, "turns": self.n_turns, "rpm": round(self.rpm(), 1),
            "max_steer_bw": round(self.max_steer_bw, 3), "phase": self.phase,
        }
