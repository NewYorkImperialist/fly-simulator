"""DEAD HANG FLY: the fly dead-hangs for its life, forever, from a pull-up bar over a
Venus flytrap (docs/JOBS.md, "dead_hang").

Physics (no welds, no hidden forces on the fly):

* The fly hangs from a fly-scale chrome pull-up bar (a static capsule, R 0.15 mm)
  by its two front legs. What holds it is FlyGym's own tarsal adhesion actuators
  (MuJoCo ``adhesion`` actuators on tarsus5, 40 uN per leg at ctrl 1 = 4 body
  weights) plus friction, and the leg posture: the front legs reach up over the bar
  and the tarsi drape over it. Measured (see the docs): with adhesion off the tarsi
  slide off the bar within ~0.3 s; at ctrl >= 0.2 on both legs it hangs
  indefinitely; at 0.15 it falls. One arm holds while the fly is fresh (the mid legs
  brace on the bar meanwhile) and fails below ~0.35: one-arm moments are survivable
  when strong, fatal when tired.
* The posture is a job action (``HangGrip``, run by the ``ActionManager``): position
  targets for all 42 leg joints plus the 6 adhesion commands. The front-leg grip
  and reach poses come from a small inverse-kinematics solve against the actual
  bar (``LegIK``: damped least squares on the 7 actuated DoFs of one leg, on a
  scratch MjData). The mid / hind legs dangle and kick; while a front leg is off the
  bar the mid legs brace against it (friction + a little adhesion).
* The spawn / respawn pose is written into FlyGym's "neutral" keyframe (free joint
  + leg angles, like the course respawn does); ``sim.warmup_s`` is 0 so the
  standing-pose warm-up does not run. Every respawn is an explicit, counted reset.

Grip fatigue (phenomenological, ours): each front leg has a grip strength s (0-1,
the adhesion command is s, so force = 40 uN x s) that drains while it carries load
(faster on one arm, faster when stressed), capped by a slowly decaying capacity.
When a leg gets tired the fly re-grips it (shifts its weight, lifts it off the bar
and puts it back: a real, physical leg lift, the other leg holds alone meanwhile),
which restores part of the strength and costs capacity; once tired it stops daring
to let go and clings. Random slips (more likely when tired) make it
hang one-armed and kick until it re-grabs the bar. Whether it falls is decided by
the physics: once the adhesion is too weak, the tarsi slide off.

The flytrap: two lobes (procedural meshes, red inside, green outside, marginal
teeth) on hinge joints with position servos, on a stem that can lunge (slide
joint). The lobes are visual; the fly lands on a colliding pad on the midrib (a
labelled simplification: snapping lobes that squeeze the fly would need a
stiff multi-geom contact model). When the falling fly enters the mouth the trap
SNAPS (counted "CHOMP"), holds, then the fly is explicitly respawned on the bar and
the trap slowly reopens. Occasional "twitches" (a small lunge + half snap) are the
looming threat for the brain tie-in (``--brain``): the looming machinery drives
LC4 / LPLC2, and a giant-fibre burst makes the fly FLINCH (clench + kick) instead of
jumping: on the bar a jump is suicide, so the job re-maps GF -> flinch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco as mj
import numpy as np

from fly_simulator.actions.base import LEGS, Action, ActionCommand, smoothstep
from fly_simulator.jobs import dead_hang_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import add_box, contact_kwargs, quat_axis_angle, quat_mul
from fly_simulator.jobs.geometry import spot_or_directional
from fly_simulator.jobs.registry import register_job

P = "hang/"
FRONT = ("lf", "rf")
REAR = ("lm", "rm", "lh", "rh")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class DeadHangConfig(JobConfig):
    # --- bar (world: bar axis along x at y = 0, z = bar_height) -------------------
    bar_height: float = 11.0
    bar_radius: float = 0.15
    bar_half_len: float = 4.0
    bar_friction: float = 1.0
    # tarsus-on-bar contacts: condim 4 (sliding + torsional friction: the pad is a
    # small patch, not a point, so it resists spinning about the contact normal);
    # condim 3 = FlyGym's point contacts (a one-arm fly pivots freely), 6 adds rolling
    bar_condim: int = 4
    bar_torsion: float = 0.02  # mm (FlyGym's default torsional friction)
    bar_rolling: float = 0.01  # mm (used only with condim 6)
    # --- hanging pose (fly faces -y, ventral side toward the bar and the camera) --
    hang_pitch_deg: float = 80.0  # body axis this far from horizontal (nose up)
    hang_drop: float = 1.2  # thorax centre below the bar axis at spawn (mm)
    hang_back: float = 0.0  # thorax centre this far behind the bar (+y) at spawn (mm)
    grip_half_width: float = 0.32  # the front tarsi this far either side of x = 0
    grip_over: float = -0.05  # mm: tarsus5 this far past the top of the bar (<0: short of it)
    grip_from: float = -1.0  # the legs come up on this side of the bar (+1 = +y, -1 = -y)
    # --- flytrap -----------------------------------------------------------------
    trap_y: float = 0.0  # trap centre (midrib) at this y (the fly swings in y)
    trap_hinge_z: float = 5.0
    lobe_half_len: float = 2.5
    lobe_width: float = 2.3
    lobe_cup: float = 1.05
    open_deg: float = 55.0
    closed_deg: float = 4.0
    snap_hz: float = 12.0  # lobe servo natural frequency (critically damped)
    reopen_s: float = 6.0
    chomp_hold_s: float = 2.0  # closed on the fly this long -> respawn
    miss_hold_s: float = 1.5  # landed outside the trap this long -> respawn
    lost_after_s: float = 3.0  # falling this long without landing anywhere -> respawn
    # --- grip fatigue (phenomenological) ------------------------------------------
    fatigue_per_s: float = 0.011  # strength lost per second per loaded leg
    one_arm_load: float = 2.5  # ... x this when one leg holds alone
    cap_decay_per_s: float = 0.002  # the capacity (max strength) decays too
    regrip_below: float = 0.9  # re-grip a leg when its strength drops below this ...
    regrip_margin: float = 0.12  # ... and this far below the capacity (a tired fly
    #                              re-grips relative to what it has left)
    regrip_jitter: float = 0.04
    regrip_other_min: float = 0.6  # ... only if the other leg can hold alone (else it clings)
    regrip_restore: float = 0.6  # a re-grip restores this fraction of (cap - s)
    regrip_cap_cost: float = 0.045  # ... and costs this much capacity
    regrip_cooldown_s: float = 1.2
    regrip_release_s: float = 0.2  # re-grip: shift the weight first (adhesion ramps off)
    regrip_lift_s: float = 0.12
    regrip_place_s: float = 0.18
    slip_rate_per_s: float = 0.045  # hazard at strength 0.5 (x ((1 - s) / 0.5)^2)
    struggle_s: tuple[float, float] = (0.8, 1.8)  # one-arm kicking before a re-grab
    grip_lost_s: float = 0.4  # tarsi off the bar this long while "gripping" = slipped
    one_arm_pull: float = 0.3  # rad: the holding leg flexes (pulls up) while the other is off
    slip_drop: float = 0.5  # mm: a slipped front leg hangs this far below the re-grab point
    slip_drop_s: float = 0.15
    slip_release_s: float = 0.15  # a slip: the tarsus loses its adhesion over this long
    flail_hz: float = 5.5
    flail_amp: float = 0.15  # rad
    regrab_retry_s: float = 0.7
    # mid-leg brace: while one front leg is off the bar, the mid legs press their
    # tarsi against the front of the bar (friction + a little adhesion) to steady the
    # body (without it one arm is a frictionless pivot: the fly swings and spins)
    brace: bool = True
    brace_legs: tuple = ("lm", "rm")
    brace_x: float = 0.62  # mm from the grip centre along the bar
    brace_z: float = -0.05  # mm: height on the bar's front face (0 = the axis)
    brace_push: float = 0.02  # mm: the target is this far inside the bar surface
    brace_in_s: float = 0.12
    brace_out_s: float = 0.4
    brace_adhesion: float = 0.25  # adhesion command of a bracing mid leg
    kick_amp: float = 0.5  # rad: mid / hind leg kicks while struggling (one arm, weak grip)
    kick_hz: float = 7.0
    kick_tau_s: float = 0.1  # the kick amplitude eases in / out (a sudden stop jolts the grip)
    ik_max_residual: float = 0.35  # mm: farther than this = the bar is out of reach
    reach_lift: float = 0.15  # mm: a re-gripping leg lifts this far above its grip
    reach_press: float = 0.0  # mm: ... and is put down this far "into" the bar top
    reach_ik_iters: int = 12  # IK iterations per job tick while reaching (warm start)
    settle_s: float = 1.0  # spawn pre-computation: let the IK hang settle this long
    regrip_place_ik: bool = False  # re-grip: IK-track the placement (else joint replay)
    reach_mode: str = "joint"  # "joint": replay grip -> lift -> grip poses; "ik": track
    regrab_mode: str = "joint"  # re-grab: "joint" (corrected spawn poses) or "ik" (track)
    reach_correct_max: float = 0.5  # rad: max IK correction of the spawn grip pose
    grip_relax_s: float = 0.3  # both hands on: the grip poses relax back to the spawn hang
    grip_settled: bool = True  # aim reaches at the settled spawn grip (relative to the bar)
    regrab_spacing: tuple = (1.0, 0.75, 0.5, 0.25)  # hand spacings tried when re-grabbing
    # --- trap twitches (looming threat; also shown without a brain) ----------------
    twitch_every_s: float = 14.0  # mean interval (0 = off)
    twitch_deg: float = 22.0  # lobes snap this much toward closed
    twitch_lunge: float = 1.6  # mm: the trap head lunges up this far
    twitch_up_s: float = 0.07
    twitch_down_s: float = 0.6
    # --- brain tie-in (--brain) -----------------------------------------------------
    gf_flinch_hz: float = 60.0  # giant fibre (DNp01) rate -> flinch (= the jump rule)
    # efference copy (phenomenological): while hanging, the trap is seen as if the body
    # were at its resting hang, so the fly's own swings / re-grips / flinches do not
    # make the (large, close) trap loom; only the trap's own motion does
    efference_copy: bool = True
    flinch_s: float = 0.35
    flinch_refractory_s: float = 1.0
    flinch_fatigue: float = 0.02  # a flinch costs this much strength per front leg
    flinch_flex: float = 0.12  # rad: femur-tibia flexion of the clenching front legs
    flinch_kick: float = 1.0  # x kick_amp: mid / hind legs kick while flinching
    # --- stress (--stress): arousal speeds up fatigue and struggling ---------------
    stress_fatigue_gain: float = 1.0  # fatigue x (1 + gain * level)
    stress_slip_gain: float = 2.0  # slip hazard x (1 + gain * level)
    # --- looks ----------------------------------------------------------------------
    shadows: bool = True


# ---------------------------------------------------------------------------
# the hang action and the leg IK
# ---------------------------------------------------------------------------


class HangGrip(Action):
    """The dead-hang posture: all leg targets and adhesion come from the job (which
    updates them every job tick); runs until cancelled or reset."""

    name = "dead_hang"
    blend_in = 0.0
    blend_out = 0.15

    def __init__(self, job: "DeadHangJob") -> None:
        super().__init__(duration=math.inf)
        self.job = job

    def command(self, mgr, t: float) -> ActionCommand:
        return ActionCommand(targets=self.job.targets, adhesion=self.job.adhesion)


class LegIK:
    """Position IK for one leg's 7 actuated DoFs (damped least squares) on a scratch
    MjData, so the live simulation is never touched."""

    def __init__(self, sim, body) -> None:
        self.sim = sim
        self.m = sim.model
        self.d = mj.MjData(self.m)
        self.body = body
        m = self.m
        self.dof = np.array([m.jnt_dofadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT,
                                                         f"{sim.fly_name}/{n}")]
                             for n in body.names])
        self._jac = np.zeros((3, m.nv))
        self._bid = {}

    def bid(self, leg: str, seg: str) -> int:
        k = (leg, seg)
        if k not in self._bid:
            self._bid[k] = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY,
                                         f"{self.sim.fly_name}/{leg}_{seg}")
        return self._bid[k]

    def solve(self, qpos: np.ndarray, leg: str, targets, iters: int = 120,
              damping: float = 1e-3, max_step: float = 0.1, ref: np.ndarray | None = None,
              ref_gain: float = 0.2, weights=None) -> tuple[np.ndarray, float]:
        """``targets``: [(segment, world point)], e.g. [("tarsus1", p1), ("tarsus5", p5)].
        ``ref``: preferred joint angles (7,): the redundant DoF is pulled toward them in
        the Jacobian's null space (keeps the leg shape, so the body hangs at the same
        height after a re-grip). Returns (the leg's 7 joint angles in controller order,
        residual mm)."""
        m, d, b = self.m, self.d, self.body
        d.qpos[:] = qpos
        cols = np.flatnonzero(b.leg_mask([leg]))
        qa = b.qpos_adr[cols]
        dof = self.dof[cols]
        err = math.inf
        for _ in range(iters):
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            J, E = [], []
            for k, (seg, p) in enumerate(targets):
                bi = self.bid(leg, seg)
                w = 1.0 if weights is None else weights[k]
                mj.mj_jacBody(m, d, self._jac, None, bi)
                J.append(w * self._jac[:, dof])
                E.append(w * (np.asarray(p, float) - d.xpos[bi]))
            J = np.vstack(J)
            E = np.concatenate(E)
            err = float(np.linalg.norm(E))
            if err < 1e-3:
                break
            Jp = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(len(E)))
            dq = 0.5 * (Jp @ E)
            if ref is not None:
                dq += (np.eye(len(qa)) - Jp @ J) @ (ref_gain * (ref - d.qpos[qa]))
            d.qpos[qa] += np.clip(dq, -max_step, max_step)
        mj.mj_kinematics(m, d)
        wts = [1.0] * len(targets) if weights is None else weights
        err = float(np.linalg.norm(np.concatenate(
            [w * (np.asarray(p, float) - d.xpos[self.bid(leg, s)])
             for w, (s, p) in zip(wts, targets)])))
        return d.qpos[qa].copy(), err


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------


@register_job
class DeadHangJob(EternalJob):
    name = "dead_hang"
    #: depth precision (shadow maps span znear..zfar; EternalJob applies it after compile)
    znear = 0.05
    title = "DEAD HANG FLY"
    tagline = "hanging on for dear life"
    work_label = "time on the bar"
    work_format = "{:.0f} s"
    config_cls = DeadHangConfig
    required_names = (P + "bar", P + "trap_head", P + "lobe_near", P + "lobe_far")

    def __init__(self, cfg: DeadHangConfig | None = None) -> None:
        super().__init__(cfg)
        c = self.cfg
        self.rng = np.random.default_rng(c.seed + 404)
        self.phase = "hang"  # hang | falling | chomped | missed
        self.leg_state = {leg: "grip" for leg in FRONT}  # grip | regrip | slipped | regrab
        self.strength = {leg: 1.0 for leg in FRONT}
        self.cap = 1.0
        # counters (constant memory)
        self.n_regrips = 0
        self.n_slips = 0
        self.n_out_of_reach = 0  # re-grab attempts refused: bar out of reach
        self.n_lost = 0  # slips where a gripping leg physically slid off the bar
        self.n_regrabs = 0
        self.n_regrab_fails = 0
        self.n_drops = 0  # falls off the bar
        self.n_chomps = 0
        self.n_escapes = 0  # falls that missed the trap
        self.n_twitches = 0
        self.n_flinches = 0
        self.n_gf_bursts = 0
        # falls by situation: "two-arm" (both hands on) / "one-arm" (a hand was off),
        # and the grip of the hand(s) still on the bar when it fell
        self.fall_causes: dict[str, int] = {}
        self.last_fall_grip = 1.0
        self._fall_grip_sum = 0.0
        self.best_streak = 0.0
        self.last_streak = 0.0
        self.streak_t0 = 0.0
        self.min_strength_seen = 1.0
        self.trap_mode = "open"
        self.stress_level = 0.0
        self.flash = ""  # HUD banner
        self._flash_until = -1.0
        self.gf_hz = 0.0
        self.grip_cx = 0.0  # x on the bar the hands are centred on
        self.grip_rel = None  # settled tarsus1 / tarsus5 positions relative to the bar
        self.hand_x: dict[str, float] = {}
        self._reopen_after_reset = False
        self._trap_t0 = 0.0
        self._twitch_t0 = 0.0

    # ------------------------------------------------------------ build
    def configure_app(self, app_cfg) -> None:
        super().configure_app(app_cfg)
        # no standing warm-up: the fly spawns hanging (keyframe) and the hang
        # action takes over at the first physics step
        app_cfg.sim.warmup_s = 0.0

    @property
    def bar_pos(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.cfg.bar_height])

    @property
    def trap_center(self) -> np.ndarray:
        c = self.cfg
        return np.array([0.0, c.trap_y, c.trap_hinge_z])

    def rim_z_open(self) -> float:
        c = self.cfg
        return c.trap_hinge_z + c.lobe_width * math.cos(math.radians(c.open_deg))

    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        self._add_materials(spec)
        self._add_gym(spec)
        # --- the pull-up bar: a static capsule along x (the only thing the fly grips)
        H, R = c.bar_height, c.bar_radius
        qx = quat_axis_angle((0, 1, 0), math.pi / 2)  # capsule z -> world x
        bar_kw = contact_kwargs("static", c.bar_friction)
        bar_kw.update(condim=int(c.bar_condim),
                      friction=(c.bar_friction, c.bar_torsion, c.bar_rolling))
        wb.add_geom(name=P + "bar", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(R, c.bar_half_len, 0),
                    pos=(0, 0, H), quat=qx, material=P + "chrome", **bar_kw)
        # chalk marks where the hands go (visual)
        for s in (-1, 1):
            wb.add_geom(name=P + f"chalk{'lr'[s > 0]}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                        size=(R * 1.02, 0.11, 0), pos=(s * c.grip_half_width, 0, H), quat=qx,
                        material=P + "chalk", **vis)
        # posts, end caps and brackets, base plates (visual: the fly never reaches them)
        L = c.bar_half_len
        for s in (-1, 1):
            x = s * L
            wb.add_geom(name=P + f"post{s:+d}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                        size=(0.22, (H + 0.35) / 2, 0), pos=(x, 0, (H + 0.35) / 2),
                        material=P + "powder", **vis)
            wb.add_geom(name=P + f"bracket{s:+d}", type=mj.mjtGeom.mjGEOM_BOX,
                        size=(0.3, 0.3, 0.3), pos=(x, 0, H), material=P + "powder", **vis)
            wb.add_geom(name=P + f"cap{s:+d}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                        size=(0.27, 0.05, 0), pos=(x, 0, H + 0.4), material=P + "powder", **vis)
            wb.add_geom(name=P + f"plate{s:+d}", type=mj.mjtGeom.mjGEOM_BOX,
                        size=(0.7, 1.3, 0.05), pos=(x, 0, 0.05), material=P + "powder", **vis)
            for k in (-1, 1):
                wb.add_geom(name=P + f"bolt{s:+d}{k:+d}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                            size=(0.08, 0.04, 0), pos=(x, k * 0.95, 0.12),
                            material=P + "chrome", **vis)
        self._add_trap(spec)

    def _add_materials(self, spec) -> None:
        s = self.cfg.seed
        spec.add_material(name=P + "chrome", rgba=(0.62, 0.64, 0.68, 1), specular=1.0,
                          shininess=0.95, reflectance=0.15)
        spec.add_material(name=P + "powder", rgba=(0.10, 0.10, 0.11, 1), specular=0.35,
                          shininess=0.4)
        spec.add_material(name=P + "chalk", rgba=(0.97, 0.97, 0.95, 0.72), specular=0.05,
                          shininess=0.05)
        A.add_texture(spec, P + "tex_trap", A.trap_texture(s))
        A.add_textured_material(spec, P + "trap", P + "tex_trap", rgba=(1, 1, 1, 1),
                                specular=0.35, shininess=0.5)
        A.add_texture(spec, P + "tex_cilia", A.cilia_texture())
        A.add_textured_material(spec, P + "cilia", P + "tex_cilia", rgba=(1, 1, 1, 1),
                                specular=0.2, shininess=0.3)
        spec.add_material(name=P + "stem", rgba=(0.30, 0.52, 0.16, 1), specular=0.25,
                          shininess=0.4)
        A.add_texture(spec, P + "tex_pot", A.terracotta_texture(s))
        A.add_textured_material(spec, P + "pot", P + "tex_pot", rgba=(1, 1, 1, 1),
                                specular=0.15, shininess=0.2)
        A.add_texture(spec, P + "tex_soil", A.soil_texture(s))
        A.add_textured_material(spec, P + "soil", P + "tex_soil", rgba=(1, 1, 1, 1),
                                specular=0.05, shininess=0.1, texuniform=True, texrepeat=(0.6, 0.6))
        A.add_texture(spec, P + "tex_brick", A.brick_texture(s))
        A.add_textured_material(spec, P + "brick", P + "tex_brick", rgba=(1, 1, 1, 1),
                                specular=0.1, shininess=0.1, texuniform=True, texrepeat=(0.25, 0.25))
        posters = {
            "poster1": (("NO PAIN", "NO GAIN"), (0.05, 0.05, 0.06)),
            "poster2": (("DON'T", "LOOK", "DOWN"), (0.08, 0.10, 0.20)),
            "poster3": (("HANG", "IN", "THERE"), (0.10, 0.06, 0.06)),
        }
        for nm, (lines, bg) in posters.items():
            img = A.poster_texture(lines, bg=bg)
            A.add_texture(spec, P + "tex_" + nm, img)
            A.add_textured_material(spec, P + nm, P + "tex_" + nm, rgba=(1, 1, 1, 1),
                                    specular=0.3, shininess=0.5)

    def _add_gym(self, spec) -> None:
        """Rubber gym floor, a painted-brick wall with posters, the light rig."""
        c = self.cfg
        wb = spec.worldbody
        mat = spec.material("grid")
        if mat is not None:
            A.add_texture(spec, P + "tex_floor", A.gym_floor_texture(c.seed))
            mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = P + "tex_floor"
            mat.rgba = (1.0, 1.0, 1.0, 1.0)
            mat.reflectance = 0.04
        # back wall (behind the fly, as seen from the camera at -y); its tiled face is
        # the box's local +z face turned toward -y
        face_my = quat_axis_angle((1, 0, 0), math.pi / 2)
        add_box(wb, P + "wall", (40.0, 16.0, 0.1), (0.0, 9.0, 15.0), quat=face_my,
                material=P + "brick", collide="visual")
        add_box(wb, P + "wall_side", (16.0, 30.0, 0.1), (17.0, -20.0, 15.0),
                quat=quat_axis_angle((0, 1, 0), -math.pi / 2), material=P + "brick",
                collide="visual")
        for nm, x, z, w, h in (("poster1", -7.2, 9.5, 1.7, 2.3), ("poster2", 6.8, 10.2, 1.5, 2.0),
                               ("poster3", 10.5, 6.5, 1.2, 1.6)):
            add_box(wb, P + nm, (w, h, 0.02), (x, 8.85, z), quat=face_my,
                    material=P + nm, collide="visual")
        spec.visual.headlight.ambient = (0.30, 0.30, 0.30)
        spec.visual.headlight.diffuse = (0.30, 0.30, 0.30)
        spec.visual.headlight.specular = (0.12, 0.12, 0.12)
        tgt = np.array([0.0, 0.3, 8.0])
        key = np.array([-6.0, -9.0, 20.0])
        wb.add_light(name=P + "key", type=spot_or_directional(c.shadows), pos=tuple(key),
                     dir=tuple(tgt - key), diffuse=(0.62, 0.60, 0.56), specular=(0.5, 0.5, 0.5),
                     cutoff=40.0, exponent=0.5, castshadow=bool(c.shadows))
        rim = np.array([4.0, 6.0, 15.0])
        wb.add_light(name=P + "rim", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=tuple(rim),
                     dir=tuple(tgt - rim), diffuse=(0.30, 0.32, 0.36), specular=(0.4, 0.4, 0.45),
                     cutoff=35.0, exponent=2.0, castshadow=False)
        # a greenish uplight from the trap (drama)
        up = np.array([0.0, -3.0, 3.0])
        wb.add_light(name=P + "trap_glow", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=tuple(up),
                     dir=(0.0, 0.5, 1.0), diffuse=(0.10, 0.16, 0.06), specular=(0, 0, 0),
                     cutoff=50.0, exponent=1.0, castshadow=False)

    def _add_trap(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        T = self.trap_center
        pot_z0, pot_h = 0.0, 3.2
        # --- pot + soil (the soil top and pot rim collide: a missed fly lands there)
        pot = A.lathe(np.array([0.0, 1.85, 2.15, 2.5, 2.62, 2.62, 2.4, 0.0]),
                      np.array([0.0, 0.0, 2.3, 2.45, 2.55, 3.2, 3.2, 3.2]), 48)
        A.add_mesh(spec, P + "pot_mesh", pot)
        wb.add_geom(name=P + "pot", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "pot_mesh",
                    pos=(T[0], T[1], pot_z0), material=P + "pot", **vis)
        soil_top = pot_z0 + pot_h - 0.2
        wb.add_geom(name=P + "soil", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(2.45, 0.1, 0),
                    pos=(T[0], T[1], soil_top - 0.1), material=P + "soil",
                    **contact_kwargs("static"))
        self.soil_top = soil_top
        # moss tufts
        vr = np.random.default_rng(c.seed + 31)
        for k in range(14):
            r, th = 2.1 * math.sqrt(vr.random()), vr.uniform(0, 2 * math.pi)
            wb.add_geom(name=P + f"moss{k}", type=mj.mjtGeom.mjGEOM_ELLIPSOID,
                        size=(vr.uniform(0.15, 0.35), vr.uniform(0.15, 0.35), 0.07),
                        pos=(T[0] + r * math.cos(th), T[1] + r * math.sin(th), soil_top),
                        rgba=(0.30 + 0.1 * vr.random(), 0.50 + 0.12 * vr.random(), 0.16, 1),
                        **vis)
        # --- decorative rosette: petioles with small traps (visual, fixed)
        lobe = A.trap_lobe(c.lobe_half_len, c.lobe_width, c.lobe_cup)
        cil = A.trap_cilia(c.lobe_half_len, c.lobe_width, c.lobe_cup, seed=c.seed)
        A.add_mesh(spec, P + "lobe_far_mesh", lobe)
        A.add_mesh(spec, P + "lobe_near_mesh", A.mirror_y(lobe))
        A.add_mesh(spec, P + "cilia_far_mesh", cil)
        A.add_mesh(spec, P + "cilia_near_mesh", A.mirror_y(cil))
        for k, (ang, dist, op, sc) in enumerate(((2.3, 2.0, 40, 0.42), (-2.6, 2.1, 8, 0.38),
                                                 (0.5, 2.2, 30, 0.35))):
            base = T[:2] + dist * np.array([math.cos(ang), math.sin(ang)])
            p0 = np.array([T[0], T[1], soil_top])
            p1 = np.array([base[0], base[1], soil_top + 0.5])
            A.add_mesh(spec, P + f"pet{k}_mesh", A.petiole(p0, p1, bend=0.8, w0=0.08, w1=0.3))
            wb.add_geom(name=P + f"pet{k}", type=mj.mjtGeom.mjGEOM_MESH,
                        meshname=P + f"pet{k}_mesh", material=P + "stem", **vis)
            yaw = quat_axis_angle((0, 0, 1), ang + math.pi / 2)
            for side, mesh in ((1, "far"), (-1, "near")):
                q = quat_mul(yaw, quat_axis_angle((1, 0, 0), -side * math.radians(op)))
                sm = A.MeshData(lobe.verts * sc, lobe.faces, lobe.uv)
                A.add_mesh(spec, P + f"mini{k}{mesh}_mesh", sm if side > 0 else A.mirror_y(sm))
                wb.add_geom(name=P + f"mini{k}{mesh}", type=mj.mjtGeom.mjGEOM_MESH,
                            meshname=P + f"mini{k}{mesh}_mesh", pos=(p1[0], p1[1], p1[2]),
                            quat=q, material=P + "trap", **vis)
        # --- the main petiole (stem) up to the trap
        A.add_mesh(spec, P + "stem_mesh", A.petiole(
            np.array([T[0] - 0.9, T[1] + 0.3, soil_top]), T - np.array([0.0, 0.0, 0.1]),
            bend=0.4, w0=0.14, w1=0.42))
        wb.add_geom(name=P + "stem", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "stem_mesh",
                    material=P + "stem", **vis)
        # --- the catch pad on the midrib (static: the falling fly lands here)
        pad_top = T[2] + 0.12
        self.pad_top = pad_top
        # (invisible, inside the trap: labelled in the docs as the engineered part)
        add_box(wb, P + "pad", (c.lobe_half_len * 0.95, 1.05, 0.1), (T[0], T[1], pad_top - 0.1),
                rgba=(0.45, 0.08, 0.10, 0.0), collide="static", friction=1.0, group=3)
        # --- trap head (slide z: the lunge) with the two lobes (hinges about x)
        head = wb.add_body(name=P + "trap_head", pos=tuple(T), gravcomp=1.0)
        head.add_joint(name=P + "lunge", type=mj.mjtJoint.mjJNT_SLIDE, axis=(0, 0, 1),
                       range=(-0.2, 3.0), limited=mj.mjtLimited.mjLIMITED_TRUE, damping=0.0, armature=0.0)
        m_head = 2e-4
        head.mass = m_head
        head.inertia = (1e-4, 1e-4, 1e-4)
        head.explicitinertial = True
        head.add_geom(name=P + "midrib", type=mj.mjtGeom.mjGEOM_CAPSULE,
                      size=(0.16, c.lobe_half_len * 0.92, 0), quat=quat_axis_angle((0, 1, 0), math.pi / 2),
                      material=P + "stem", **vis)
        self._servo = {}
        w = 2 * math.pi * c.snap_hz
        m_lobe, I_lobe = 5e-5, 5e-5 * c.lobe_width ** 2 / 3
        for side, nm in ((1, "far"), (-1, "near")):
            lb = head.add_body(name=P + f"lobe_{nm}", pos=(0, 0, 0.05), gravcomp=1.0)
            lb.mass = m_lobe
            lb.inertia = (I_lobe, I_lobe, 1e-6)
            lb.explicitinertial = True
            lb.add_joint(name=P + f"hinge_{nm}", type=mj.mjtJoint.mjJNT_HINGE, axis=(1, 0, 0),
                         damping=0.0, armature=0.0)
            lb.add_geom(name=P + f"lobe_{nm}_vis", type=mj.mjtGeom.mjGEOM_MESH,
                        meshname=P + f"lobe_{nm}_mesh", material=P + "trap", **vis)
            lb.add_geom(name=P + f"cilia_{nm}", type=mj.mjtGeom.mjGEOM_MESH,
                        meshname=P + f"cilia_{nm}_mesh", material=P + "cilia", **vis)
            act = spec.add_actuator(name=P + f"servo_{nm}", target=P + f"hinge_{nm}",
                                    trntype=mj.mjtTrn.mjTRN_JOINT)
            kp = I_lobe * w * w
            act.set_to_position(kp=kp, kv=2.0 * I_lobe * w)
        wl = 2 * math.pi * 10.0
        act = spec.add_actuator(name=P + "servo_lunge", target=P + "lunge",
                                trntype=mj.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=m_head * wl * wl, kv=2.0 * m_head * wl)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        sim, s = self.sim, self.session
        m, d = sim.model, sim.data
        c = self.cfg
        # the walking fall detector does not apply to a hanging fly (it reads a fly
        # hanging vertically as "fallen"): pause it; the job's own rule (off the bar)
        # decides falls
        hooks = sim.post_step_hooks
        det_hook = getattr(s, "_detector_hook", None)
        if det_hook is not None and det_hook in hooks:
            hooks[hooks.index(det_hook)] = self._detector_paused
        self.body = s.actions.body
        self.ik = LegIK(sim, self.body)
        g = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        j = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        a = lambda n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, n)  # noqa: E731
        self.bar_gid = g(P + "bar")
        fn = sim.fly_name
        self._any_geom = {int(gg): LEGS[int(self.body.geom_leg[gg])]
                          for gg in np.flatnonzero(self.body.geom_leg >= 0)
                          if LEGS[int(self.body.geom_leg[gg])] in FRONT}
        self._grip_geom = {g(f"{fn}/{leg}_tarsus{k}"): leg for leg in FRONT for k in (3, 4, 5)}
        self.pad_gid = g(P + "pad")
        self.act_far, self.act_near, self.act_lunge = (a(P + "servo_far"), a(P + "servo_near"),
                                                       a(P + "servo_lunge"))
        self.q_far = int(m.jnt_qposadr[j(P + "hinge_far")])
        self.q_near = int(m.jnt_qposadr[j(P + "hinge_near")])
        self.q_lunge = int(m.jnt_qposadr[j(P + "lunge")])
        self.v_far = int(m.jnt_dofadr[j(P + "hinge_far")])
        self.v_near = int(m.jnt_dofadr[j(P + "hinge_near")])
        self.v_lunge = int(m.jnt_dofadr[j(P + "lunge")])
        self.lobe_geoms = [g(P + "lobe_far_vis"), g(P + "lobe_near_vis")]
        self.head_bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, P + "trap_head")
        self.fly_weight = sim.fly_mass * 9810.0
        # the hanging keyframe: free joint + leg angles
        self._build_hang_keyframe()
        # initial spawn: no physics step has run yet (warmup 0); put the fly into the
        # keyframe pose it will also respawn in
        key = self._key
        mj.mj_resetDataKeyframe(m, d, key)
        mj.mj_forward(m, d)
        s.metrics.notify_reset(sim.time, sim.thorax_position())
        s.metrics.n_resets -= 1  # the initial spawn is not a reset
        self.steering.set(None, 0.0)  # the CPG idles underneath the hang action
        self._begin_hang(first=True)
        self._apply_trap(0.0)
        # brain tie-in
        self.vision = None
        self._gf_seen_t = -1.0
        self._flinch_ready = -1e9
        link = getattr(s, "brain", None)
        if link is not None:
            self._install_brain(link)

    def _build_hang_keyframe(self) -> None:
        sim, c = self.sim, self.cfg
        m = sim.model
        self._key = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, "neutral")
        q = m.key_qpos[self._key].copy()
        fq = sim._free_qpos
        pitch = math.radians(c.hang_pitch_deg)
        heading = -math.pi / 2  # the fly faces -y: ventral side toward the bar / camera
        quat = quat_mul(quat_axis_angle((0, 0, 1), heading), quat_axis_angle((0, 1, 0), -pitch))
        dd = self.ik.d
        dd.qpos[:] = q
        dd.qpos[fq + 3:fq + 7] = quat
        mj.mj_kinematics(m, dd)
        want = self.bar_pos + np.array([0.0, c.hang_back, -c.hang_drop])
        q[fq + 3:fq + 7] = quat
        q[fq:fq + 3] = dd.qpos[fq:fq + 3] + (want - dd.xpos[sim.thorax_body_id])
        b = self.body
        self.q_grip = {}
        self.ik_residual = {}
        for leg in FRONT:
            ang, err = self.ik.solve(q, leg, self.grip_targets(leg))
            cols = np.flatnonzero(b.leg_mask([leg]))
            q[b.qpos_adr[cols]] = ang
            self.q_grip[leg] = ang
            self.ik_residual[leg] = err
        self.q_grip0 = {leg: v.copy() for leg, v in self.q_grip.items()}
        self.dangle = self._dangle_pose()
        self._brace_w = 0.0
        self._brace_on = False
        self._kick_amp, self._kick_ph = 0.08, 0.0
        for leg in REAR:
            cols = np.flatnonzero(b.leg_mask([leg]))
            q[b.qpos_adr[cols]] = self.dangle[cols]
        # trap open in the keyframe
        o = math.radians(c.open_deg)
        q[self.q_far], q[self.q_near], q[self.q_lunge] = -o, o, 0.0
        if c.settle_s > 0:
            q = self._settle(q)
        m.key_qpos[self._key] = q
        self.key_qpos = q
        # where the tarsi really rest on the bar after settling: re-grips and re-grabs
        # aim there (relative to the bar), not at the nominal IK targets
        dd = self.ik.d
        dd.qpos[:] = q
        mj.mj_kinematics(m, dd)
        self.grip_rel = {leg: {sg: dd.xpos[self.ik.bid(leg, sg)] - self.bar_pos
                               for sg in ("tarsus1", "tarsus5")} for leg in FRONT}
        self.hand_x0 = {leg: float(self.grip_rel[leg]["tarsus5"][0]) for leg in FRONT}
        self.hand_spacing = self.hand_x0["lf"] - self.hand_x0["rf"]
        self.hand_x = dict(self.hand_x0)
        # the re-grip lift and the slipped-leg pose, from the settled hang
        self.q_lift0 = {leg: self.ik.solve(q, leg, self.lift_targets(leg, q),
                                           ref=self.q_grip0[leg])[0] for leg in FRONT}
        self.q_slip = {leg: self.ik.solve(q, leg, self.front_targets(leg, drop=c.slip_drop),
                                          ref=self.q_grip0[leg])[0] for leg in FRONT}
        # the mid-leg brace pose (from the settled hang)
        self.q_brace0, self.brace_residual = {}, {}
        for leg in c.brace_legs:
            cols = b.leg_mask([leg])
            ang, err = self.ik.solve(q, leg, [("tarsus5", self.brace_target(leg))],
                                     ref=self.dangle[cols])
            self.q_brace0[leg], self.brace_residual[leg] = ang, err
        self.q_brace = {k: v.copy() for k, v in self.q_brace0.items()}
        self.brace_aim_err = dict(self.brace_residual)

    def brace_target(self, leg: str, below_deg: float | None = None,
                     x: float | None = None) -> np.ndarray:
        """Where a bracing mid leg puts tarsus5: on the front face of the bar
        (``brace_z``), or ``below_deg`` round the bar toward its underside."""
        c = self.cfg
        s = 1.0 if leg[0] == "l" else -1.0
        cx = 0.5 * (self.hand_x["lf"] + self.hand_x["rf"]) if self.hand_x else 0.0
        n = c.grip_from
        r = c.bar_radius + 0.02 - c.brace_push
        bx = cx + s * (c.brace_x if x is None else x)
        if below_deg is None:
            return self.bar_pos + np.array([bx, n * r, c.brace_z])
        a = math.radians(below_deg)
        return self.bar_pos + np.array([bx, n * r * math.cos(a), -r * math.sin(a)])

    def _settle(self, q: np.ndarray) -> np.ndarray:
        """Pre-compute the spawn: let the IK hanging pose settle for ``settle_s`` on a
        scratch MjData (same model, the grip targets and full adhesion held), so the
        keyframe is the fly's resting hang (the tarsi slide a little on the bar before
        the adhesion holds). Returns the settled qpos; the live sim is not touched."""
        m, c, b = self.sim.model, self.cfg, self.body
        dd = mj.MjData(m)
        dd.qpos[:] = q
        mj.mj_forward(m, dd)
        tg = q[b.qpos_adr].copy()
        adh = np.array([1.0 if l in FRONT else 0.0 for l in LEGS])
        o = math.radians(c.open_deg)
        for _ in range(int(round(c.settle_s / m.opt.timestep))):
            dd.ctrl[b.pos_ids] = tg
            dd.ctrl[b.adh_ids] = adh
            dd.ctrl[self.act_far], dd.ctrl[self.act_near], dd.ctrl[self.act_lunge] = -o, o, 0.0
            mj.mj_step(m, dd)
        out = dd.qpos.copy()
        if not np.all(np.isfinite(out)):
            return q
        self.settled_thorax = dd.xpos[self.sim.thorax_body_id].copy()
        self.settled_thorax_R = dd.xmat[self.sim.thorax_body_id].reshape(3, 3).copy()
        return out

    def lift_targets(self, leg: str, q: np.ndarray):
        """Tarsus1 / tarsus5 of ``leg`` in pose ``q``, raised by ``reach_lift`` and moved
        a little toward the side the legs come up on (off the bar)."""
        c = self.cfg
        dd = self.ik.d
        dd.qpos[:] = q
        mj.mj_kinematics(self.sim.model, dd)
        up = np.array([0.0, 0.5 * c.grip_from * c.reach_lift, c.reach_lift])
        return [(sg, dd.xpos[self.ik.bid(leg, sg)] + up) for sg in ("tarsus1", "tarsus5")]

    def grip_targets(self, leg: str, lift: float = 0.0, press: float = 0.0):
        """World targets for a gripping (lift = 0) or reaching (lift > 0) front leg:
        tarsus1 on the fly's side of the bar, tarsus5 over the top."""
        c = self.cfg
        s = 1.0 if leg[0] == "l" else -1.0  # the fly faces -y: its left is +x
        bar = self.bar_pos
        R = c.bar_radius
        n = c.grip_from  # the side of the bar the legs come up on (+1 = +y)
        if self.grip_rel is not None and c.grip_settled:
            rel = self.grip_rel[leg]
            off = np.array([self.hand_x[leg] - rel["tarsus5"][0], 0.0, 0.0])
            return [("tarsus1", bar + rel["tarsus1"] + off + np.array([0.0, n * 0.1 * lift, lift])),
                    ("tarsus5", bar + rel["tarsus5"] + off
                     + np.array([0.0, n * 0.2 * lift, lift - press]))]
        x = self.grip_cx + s * c.grip_half_width  # around the fly, wherever it hangs
        return [("tarsus1", bar + np.array([x, n * (R + 0.2 + 0.1 * lift), 0.1 + lift])),
                ("tarsus5", bar + np.array([x, n * (-c.grip_over + 0.2 * lift),
                                            R + 0.035 + lift - press]))]

    def front_targets(self, leg: str, drop: float = 0.0):
        """Targets in front of the bar (the side the legs come up on): above the bar
        top (drop = 0, the re-grab waypoint) or ``drop`` mm lower (a slipped leg)."""
        c = self.cfg
        s = 1.0 if leg[0] == "l" else -1.0
        bar = self.bar_pos
        R, n = c.bar_radius, c.grip_from
        x = (self.hand_x[leg] if self.grip_rel is not None and c.grip_settled
             else self.grip_cx + s * c.grip_half_width)  # around the fly, wherever it hangs
        return [("tarsus1", bar + np.array([x, n * (R + 0.45), 0.05 - drop])),
                ("tarsus5", bar + np.array([x, n * (R + 0.2), R + 0.2 - drop]))]

    def _dangle_pose(self) -> np.ndarray:
        """Mid / hind legs hanging down along the body (the standing pose with the
        legs swung back and straightened a little)."""
        b = self.body
        q = b.stand.copy()
        for leg in ("lm", "rm"):
            q[b.idx(leg, "thc_pitch")] += 0.35
            q[b.idx(leg, "ctr_pitch")] += 0.35
            q[b.idx(leg, "fti")] -= 0.3
        for leg in ("lh", "rh"):
            q[b.idx(leg, "ctr_pitch")] += 0.3
            q[b.idx(leg, "fti")] -= 0.35
        return q

    def _detector_paused(self, sim) -> None:
        """Replaces the session's walking fall detector hook (see on_attach)."""

    # ------------------------------------------------------------ reset / spawn
    def _begin_hang(self, first: bool = False) -> None:
        c = self.cfg
        rt = self.run_time()
        self.phase = "hang"
        self.state = "hanging"
        self.leg_state = {leg: "grip" for leg in FRONT}
        self.strength = {leg: 1.0 for leg in FRONT}
        self.cap = 1.0
        self.q_grip = {leg: v.copy() for leg, v in self.q_grip0.items()}  # the spawn grip
        self._pull = {leg: 0.0 for leg in FRONT}
        self._streak = {"regrips": 0, "regrabs": 0, "slips": 0, "missed": 0}
        self.streak_t0 = rt
        self.min_strength_seen = 1.0
        self._leg_t0 = {leg: 0.0 for leg in FRONT}
        self._leg_from = {leg: self.q_grip[leg].copy() for leg in FRONT}
        self._leg_until = {leg: 0.0 for leg in FRONT}
        self._touched = {leg: False for leg in FRONT}
        self._reach_from = {leg: [] for leg in FRONT}
        self._ik_sol = {leg: self.q_grip[leg].copy() for leg in FRONT}
        self._ik_err = {leg: 0.0 for leg in FRONT}
        self._q_from = {leg: self.q_grip[leg].copy() for leg in FRONT}
        self._regrab_tries = {leg: 0 for leg in FRONT}
        self._place_from = {leg: None for leg in FRONT}
        self._released = {leg: True for leg in FRONT}
        self._slip_fresh = {leg: False for leg in FRONT}
        self._q_lift = {leg: self.q_lift0[leg].copy() for leg in FRONT}
        self._q_place = {leg: None for leg in FRONT}
        self._grip_seen = {leg: self.sim.time for leg in FRONT}
        self.grip_cx = 0.0
        if self.grip_rel is not None:
            self.hand_x = dict(self.hand_x0)
        self._last_bar_t = self.sim.time
        self._next_regrip_t = self.sim.time + c.regrip_cooldown_s
        self._regrip_thr = self._draw_regrip_threshold()
        self._flinch_until = -1.0
        self._brace_w, self._brace_on = 0.0, False
        self._t_phase = self.sim.time
        self._next_twitch = self.sim.time + self._draw_twitch_gap()
        self._twitch_t0 = None
        self.targets = self.key_qpos[self.body.qpos_adr].copy()
        self.adhesion = np.array([1.0 if l in FRONT else 0.0 for l in LEGS])
        mgr = self.session.actions
        mgr.trigger(HangGrip(self), source="job")

    def _draw_regrip_threshold(self) -> float:
        c = self.cfg
        return self.rng.uniform(-c.regrip_jitter, c.regrip_jitter)

    def regrip_threshold(self) -> float:
        c = self.cfg
        return min(c.regrip_below, self.cap - c.regrip_margin) + self._regrip_thr

    def _draw_twitch_gap(self) -> float:
        c = self.cfg
        if c.twitch_every_s <= 0:
            return math.inf
        return c.twitch_every_s * self.rng.uniform(0.6, 1.4)

    def on_reset(self) -> None:
        self._begin_hang()

    def reset_props(self) -> None:
        c = self.cfg
        d = self.sim.data
        if getattr(self, "_reopen_after_reset", False):
            # the trap starts closed (it just ate the fly) and reopens slowly
            cl = math.radians(c.closed_deg)
            d.qpos[self.q_far], d.qpos[self.q_near] = -cl, cl
            self.trap_mode = "reopen"
            self._trap_t0 = self.sim.time
        else:
            self.trap_mode = "open"
        self._reopen_after_reset = False
        self._apply_trap(0.0)

    # ------------------------------------------------------------ trap servo
    def _lobe_angle(self) -> float:
        """Current commanded opening (rad from vertical, per lobe)."""
        c = self.cfg
        o, cl = math.radians(c.open_deg), math.radians(c.closed_deg)
        t = self.sim.time
        if self.trap_mode == "snap":
            return cl
        if self.trap_mode == "reopen":
            x = (t - self._trap_t0) / max(c.reopen_s, 1e-6)
            if x >= 1.0:
                self.trap_mode = "open"
                return o
            return cl + (o - cl) * smoothstep(x)
        if self.trap_mode == "twitch":
            dt = t - self._twitch_t0
            tw = math.radians(c.twitch_deg)
            if dt < c.twitch_up_s:
                return o - tw
            x = (dt - c.twitch_up_s) / c.twitch_down_s
            if x >= 1.0:
                self.trap_mode = "open"
                return o
            return o - tw * (1.0 - smoothstep(x))
        return o

    def _lunge(self) -> float:
        c = self.cfg
        if self.trap_mode != "twitch":
            return 0.0
        dt = self.sim.time - self._twitch_t0
        if dt < c.twitch_up_s:
            return c.twitch_lunge
        x = (dt - c.twitch_up_s) / c.twitch_down_s
        return c.twitch_lunge * (1.0 - smoothstep(min(x, 1.0)))

    def _apply_trap(self, _dt: float) -> None:
        d = self.sim.data
        ang = self._lobe_angle()
        d.ctrl[self.act_far] = -ang
        d.ctrl[self.act_near] = ang
        d.ctrl[self.act_lunge] = self._lunge()

    def trap_closedness(self) -> float:
        """0 = fully open, 1 = closed (measured lobe angles)."""
        c = self.cfg
        d = self.sim.data
        o, cl = math.radians(c.open_deg), math.radians(c.closed_deg)
        ang = 0.5 * (abs(d.qpos[self.q_far]) + abs(d.qpos[self.q_near]))
        return float(np.clip((o - ang) / (o - cl), 0.0, 1.0))

    def twitch(self) -> bool:
        """A trap twitch (small lunge + half snap). Returns False if not possible."""
        if self.trap_mode != "open" or self.phase != "hang":
            return False
        self.trap_mode = "twitch"
        self._twitch_t0 = self.sim.time
        self.n_twitches += 1
        return True

    def snap(self) -> None:
        self.trap_mode = "snap"
        self._trap_t0 = self.sim.time

    # ------------------------------------------------------------ helpers
    def bar_contacts(self, tarsi_only: bool = False) -> dict[str, bool]:
        """Front legs touching the bar: any tibia / tarsus geom, or (``tarsi_only``)
        only the distal tarsi (tarsus3-5: a real grip, not a shin resting on it)."""
        d = self.sim.data
        out = {leg: False for leg in FRONT}
        lut = self._grip_geom if tarsi_only else self._any_geom
        bar = self.bar_gid
        for i in range(d.ncon):
            g1, g2 = d.contact.geom[i]
            if g1 == bar:
                other = g2
            elif g2 == bar:
                other = g1
            else:
                continue
            leg = lut.get(int(other))
            if leg is not None:
                out[leg] = True
        return out

    def in_trap_mouth(self, p) -> bool:
        c = self.cfg
        T = self.trap_center
        return (abs(p[0] - T[0]) < c.lobe_half_len * 1.05
                and abs(p[1] - T[1]) < c.lobe_width * math.sin(math.radians(c.open_deg)) + 0.3
                and p[2] < self.rim_z_open() + 0.8)

    def streak(self) -> float:
        return self.run_time() - self.streak_t0 if self.phase == "hang" else self.last_streak

    def grip_pct(self) -> float:
        held = [self.strength[l] for l in FRONT if self.leg_state[l] == "grip"]
        return float(np.mean(held)) if held else 0.0

    def _say_flash(self, text: str, secs: float = 2.0) -> None:
        self.flash = text
        self._flash_until = self.run_time() + secs

    # ------------------------------------------------------------ the job tick
    def update(self) -> None:
        c = self.cfg
        sim = self.sim
        t = sim.time
        dt = c.update_every_steps * sim.timestep
        mgr = self.session.actions
        if mgr.active_name is None and self.phase == "hang":
            # something cancelled the hang (e.g. a key action ended): take over again
            mgr.trigger(HangGrip(self), source="job")
        p = sim.thorax_position()
        if self.phase == "hang":
            self._hang_tick(t, dt, p)
        elif self.phase == "falling":
            self._falling_tick(t, p)
        self._pose_rear(t)
        self._apply_trap(dt)

    def _stress_mult(self, gain: float) -> float:
        return 1.0 + gain * self.stress_level

    def _hang_tick(self, t: float, dt: float, p: np.ndarray) -> None:
        c = self.cfg
        if any(self.bar_contacts().values()):
            self._last_bar_t = t
        contacts = self.bar_contacts(tarsi_only=True)
        # a "gripping" leg whose tarsi have been off the bar for a while has lost it
        for leg in FRONT:
            if contacts[leg] or self.leg_state[leg] != "grip":
                self._grip_seen[leg] = t
            elif t - self._grip_seen[leg] > c.grip_lost_s:
                self._lost_grip(leg, t)
        self.add_work(dt)  # seconds on the bar (also keeps the stuck timer quiet)
        # --- fatigue
        gripping = [l for l in FRONT if self.leg_state[l] == "grip"]
        load = 1.0 if len(gripping) == 2 else c.one_arm_load
        f = c.fatigue_per_s * load * self._stress_mult(c.stress_fatigue_gain) * dt
        self.cap = max(0.0, self.cap - c.cap_decay_per_s * dt)
        for leg in FRONT:
            if leg in gripping:
                self.strength[leg] -= f
            self.strength[leg] = float(np.clip(self.strength[leg], 0.0, self.cap))
        if gripping:
            self.min_strength_seen = min(self.min_strength_seen, self.grip_pct())
        # --- decisions: re-grip the tired leg, or slip
        both = len(gripping) == 2
        if both and t >= self._next_regrip_t:
            weak = min(FRONT, key=lambda l: self.strength[l])
            other = FRONT[1 - FRONT.index(weak)]
            if (self.strength[weak] < self.regrip_threshold()
                    and self.strength[other] > c.regrip_other_min):
                self._start_reach(weak, t, "regrip")
            else:
                s_min = min(self.strength.values())
                haz = (c.slip_rate_per_s * ((1.0 - s_min) / 0.5) ** 2
                       * self._stress_mult(c.stress_slip_gain))
                if self.rng.random() < haz * dt:
                    self._slip(weak, t)
        # --- per-leg motion
        for leg in FRONT:
            self._front_leg_tick(leg, t, contacts)
        # --- fell off?
        z_rel = p[2] - c.bar_height
        off_bar = t - self._last_bar_t > 0.12
        if (off_bar and z_rel < -2.4) or z_rel < -3.6:
            self._fall(t)

    def _front_leg_tick(self, leg: str, t: float, contacts) -> None:
        c = self.cfg
        b = self.body
        cols = b.leg_mask([leg])
        i = LEGS.index(leg)
        st = self.leg_state[leg]
        tt = t - self._leg_t0[leg]
        clench = t < self._flinch_until
        if st == "grip":
            other = FRONT[1 - FRONT.index(leg)]
            dt = c.update_every_steps * self.sim.timestep
            if self.leg_state[other] == "grip" and c.grip_relax_s > 0:
                # both hands on the bar: the arms ease back into the spawn hang (the
                # body re-centres under the hands after a one-arm moment)
                g = self.q_grip[leg]
                g += (self.q_grip0[leg] - g) * min(1.0, dt / c.grip_relax_s)
            q = self.q_grip[leg]
            want = c.one_arm_pull if self.leg_state[other] in ("slipped", "regrab") else 0.0
            # low-passed (tau 0.2 s): letting go of the pull all at once jolts the grip
            self._pull[leg] += (want - self._pull[leg]) * min(1.0, dt / 0.2)
            flex = (c.flinch_flex if clench else 0.0) + self._pull[leg]
            if flex > 1e-4:  # flinch / one-arm: pull up (flex the femur-tibia joint)
                q = q.copy()
                q[list(np.flatnonzero(cols)).index(b.idx(leg, "fti"))] += flex
            self.targets[cols] = q
            self.adhesion[i] = 1.0 if clench else self.strength[leg]
        elif st in ("regrip", "regrab"):
            # a reach: the tarsi go from where they were to a point above the bar, then
            # down onto it. Default (joint space): the spawn lift / grip poses, each
            # IK-corrected once for where the body hangs now. reach_mode / regrab_mode
            # "ik": a warm-started IK tracks the moving targets every tick (the old
            # behaviour; it chases a swinging body into contortions)
            T0 = c.regrip_release_s if st == "regrip" else 0.0
            if tt < T0:
                # weight shift: the tarsi stay put while their adhesion ramps off, so
                # the body settles under the other hand without a pendulum kick
                self.targets[cols] = self._q_from[leg]
                self.adhesion[i] = (1.0 - smoothstep(tt / T0)) * self.strength[leg]
                return
            if not self._released[leg]:
                self._released[leg] = True
                d = self.sim.data
                self._reach_from[leg] = [(sg, d.xpos[self.ik.bid(leg, sg)].copy())
                                         for sg in ("tarsus1", "tarsus5")]
                self._ik_sol[leg] = d.qpos[b.qpos_adr[cols]].copy()
            tt -= T0
            T1, T2 = c.regrip_lift_s, c.regrip_lift_s + c.regrip_place_s
            # a re-grab comes up in front of the bar (the leg hangs below it) and then
            # down onto the top; a re-grip lifts straight off and back
            lift = (self.front_targets(leg) if st == "regrab"
                    else self.grip_targets(leg, lift=c.reach_lift))
            if tt < T1:
                a = smoothstep(tt / T1)
                tg = [(sg, (1 - a) * p0 + a * p1)
                      for (sg, p0), (_, p1) in zip(self._reach_from[leg], lift)]
                adh = 0.0
            else:
                a = smoothstep(min((tt - T1) / c.regrip_place_s, 1.0))
                grip = self.grip_targets(leg, press=c.reach_press)
                if self._place_from[leg] is None:  # start from where the tarsi really are
                    self._place_from[leg] = [(sg, self.sim.data.xpos[self.ik.bid(leg, sg)].copy())
                                             for sg, _ in grip]
                tg = [(sg, (1 - a) * p0 + a * p1)
                      for (sg, p0), (_, p1) in zip(self._place_from[leg], grip)]
                late = tt > T1 + 0.6 * c.regrip_place_s
                adh = self.strength[leg] if late else 0.0
                self._touched[leg] |= late and contacts[leg]
            if (c.reach_mode == "ik" or (st == "regrab" and c.regrab_mode == "ik")
                    or (c.regrip_place_ik and tt >= T1)):
                q = self.sim.data.qpos.copy()
                q[b.qpos_adr[cols]] = self._ik_sol[leg]
                sol, err = self.ik.solve(q, leg, tg, iters=c.reach_ik_iters,
                                         ref=self.q_grip0[leg], weights=(0.3, 1.0))
                self._ik_err[leg] = err
            else:  # joint space: current pose -> lifted grip pose -> the grip pose (the
                # spawn poses, IK-corrected for where the body hangs now)
                if tt < T1:
                    sol = (1 - a) * self._q_from[leg] + a * self._q_lift[leg]
                else:
                    if self._q_place[leg] is None:
                        self._q_place[leg] = self._corrected(leg, self.q_grip0[leg],
                                                             self.grip_targets(leg))
                    sol = (1 - a) * self._q_lift[leg] + a * self._q_place[leg]
            self._ik_sol[leg] = sol
            self.targets[cols] = sol
            self.adhesion[i] = adh
            if tt >= T2 and (tt > T2 + 0.15 or (self._touched[leg] and tt > T2 + 0.03)):
                self._finish_reach(leg, t, self._touched[leg])
        elif st == "slipped":
            # the tarsus loses its hold (adhesion ramps off over slip_release_s), then
            # the free leg drops away from the bar (over slip_drop_s) and flails
            if tt < c.slip_release_s and self._slip_fresh[leg]:
                self.targets[cols] = self._leg_from[leg]
                self.adhesion[i] = (1.0 - smoothstep(tt / c.slip_release_s)) * self.strength[leg]
                return
            if self._slip_fresh[leg]:
                tt -= c.slip_release_s
            a = smoothstep(tt / c.slip_drop_s)
            q = (1 - a) * self._leg_from[leg] + a * self.q_slip[leg]
            ph = 2 * math.pi * c.flail_hz * t + (0.0 if leg == "lf" else 1.7)
            k = list(np.flatnonzero(cols))
            q[k.index(b.idx(leg, "ctr_pitch"))] += a * c.flail_amp * math.sin(ph)
            q[k.index(b.idx(leg, "fti"))] += a * c.flail_amp * 1.1 * math.sin(ph + 1.2)
            self.targets[cols] = q
            self.adhesion[i] = 0.0
            if t >= self._leg_until[leg]:
                self._start_reach(leg, t, "regrab")

    def _start_reach(self, leg: str, t: float, kind: str) -> bool:
        """Lift the leg and put it back on the bar (closed-loop IK, see
        ``_front_leg_tick``). A re-grab is refused while the bar is out of reach from
        the current pose (the fly keeps kicking and tries again)."""
        c = self.cfg
        b = self.body
        cols = b.leg_mask([leg])
        d = self.sim.data
        # reach for the bar around where the body hangs now (it can slide along it)
        lim = c.bar_half_len - 1.0
        self.grip_cx = float(np.clip(d.xpos[self.sim.thorax_body_id, 0], -lim, lim))
        other = FRONT[1 - FRONT.index(leg)]
        settled = self.grip_rel is not None and c.grip_settled
        if settled and self.leg_state[other] == "grip":
            # aim relative to where the other hand really is on the bar
            sgn = 1.0 if leg[0] == "l" else -1.0
            ox = float(d.xpos[self.ik.bid(other, "tarsus5"), 0])
            spacings = c.regrab_spacing if kind == "regrab" else (1.0,)
            best = None
            for f in spacings:
                self.hand_x[leg] = float(np.clip(ox + sgn * f * self.hand_spacing, -lim, lim))
                if kind != "regrab":
                    break
                _, err = self.ik.solve(d.qpos.copy(), leg, self.grip_targets(leg), iters=60,
                                       ref=self.q_grip0[leg], weights=(0.3, 1.0))
                if best is None or err < best[0]:
                    best = (err, self.hand_x[leg])
                if err <= c.ik_max_residual:
                    break
            if best is not None:
                self.hand_x[leg] = best[1]
        if kind == "regrab":
            _, err = self.ik.solve(d.qpos.copy(), leg, self.grip_targets(leg), iters=60,
                                   ref=self.q_grip0[leg], weights=(0.3, 1.0))
            if err > c.ik_max_residual:
                self.n_out_of_reach += 1
                self.n_regrab_fails += 1
                self._leg_until[leg] = t + c.regrab_retry_s
                return False
        self._reach_from[leg] = [(sg, d.xpos[self.ik.bid(leg, sg)].copy())
                                 for sg, _ in self.grip_targets(leg)]
        self._ik_sol[leg] = d.qpos[b.qpos_adr[cols]].copy()
        self._q_from[leg] = self.targets[cols].copy()
        self._place_from[leg] = None
        lift_t = (self.front_targets(leg) if kind == "regrab"
                  else self.grip_targets(leg, lift=c.reach_lift))
        self._q_lift[leg] = self._corrected(leg, self.q_lift0[leg], lift_t)
        self._q_place[leg] = None
        self._released[leg] = not (kind == "regrip" and c.regrip_release_s > 0)
        if kind == "regrab":
            self._regrab_tries[leg] += 1
        self._leg_t0[leg] = t
        self.leg_state[leg] = kind
        self._touched[leg] = False
        return True

    def _finish_reach(self, leg: str, t: float, touching: bool) -> None:
        c = self.cfg
        if not touching:
            # missed the bar: dangle and try again
            self.leg_state[leg] = "slipped"
            self._slip_fresh[leg] = False
            self._leg_from[leg] = self._ik_sol[leg].copy()
            self._leg_until[leg] = t + c.regrab_retry_s
            self._leg_t0[leg] = t
            self.n_regrab_fails += 1
            self._streak["missed"] += 1
            if self._streak["missed"] <= 3 or self._streak["missed"] % 10 == 0:
                self.say(f"{leg} missed the bar (x{self._streak['missed']} this hang), kicking")
            return
        kind = self.leg_state[leg]
        self.leg_state[leg] = "grip"
        self.q_grip[leg] = self._ik_sol[leg].copy()
        if kind == "regrip":
            self.n_regrips += 1
            s = self.strength[leg]
            self.strength[leg] = s + c.regrip_restore * max(self.cap - s, 0.0)
            self.cap = max(0.0, self.cap - c.regrip_cap_cost)
            self._streak["regrips"] += 1
            self.say(f"re-grip #{self.n_regrips}: {leg} shaken out, grip {s:.0%} -> "
                     f"{self.strength[leg]:.0%} (capacity {self.cap:.0%})")
        else:
            self.n_regrabs += 1
            self._streak["regrabs"] += 1
            self.say(f"re-grab #{self.n_regrabs}: {leg} is back on the bar")
        self._next_regrip_t = t + c.regrip_cooldown_s
        self._regrip_thr = self._draw_regrip_threshold()

    def _corrected(self, leg: str, q0: np.ndarray, targets) -> np.ndarray:
        """``q0`` (a spawn pose of ``leg``) corrected by IK for the current body pose:
        start at q0, pulled toward it in the null space, each joint at most
        ``reach_correct_max`` from it (a natural leg shape, not an IK contortion)."""
        c = self.cfg
        if c.reach_correct_max <= 0:
            return q0.copy()
        b = self.body
        q = self.sim.data.qpos.copy()
        q[b.qpos_adr[b.leg_mask([leg])]] = q0
        sol, _ = self.ik.solve(q, leg, targets, iters=40, ref=q0, ref_gain=0.5,
                               weights=(0.3, 1.0))
        return q0 + np.clip(sol - q0, -c.reach_correct_max, c.reach_correct_max)

    def _lost_grip(self, leg: str, t: float) -> None:
        """The leg's tarsi slid off the bar (physics): now it is a slip."""
        self.n_lost += 1
        self._slip(leg, t, why="slid off")

    def _slip(self, leg: str, t: float, why: str = "lost the bar") -> None:
        c = self.cfg
        self.n_slips += 1
        self._streak["slips"] += 1
        self.leg_state[leg] = "slipped"
        self._slip_fresh[leg] = True
        self._regrab_tries[leg] = 0
        b = self.body
        # the slipped leg drops away from the bar and flails (see _front_leg_tick)
        self._leg_from[leg] = self.targets[b.leg_mask([leg])].copy()
        self._leg_t0[leg] = t
        self._leg_until[leg] = t + self.rng.uniform(*c.struggle_s)
        self._say_flash("SLIP!  one-arm hang", 1.5)
        self.say(f"slip #{self.n_slips}: {leg} {why} (grip {self.grip_pct():.0%})")

    def _pose_rear(self, t: float) -> None:
        """Mid / hind legs: dangle with a lazy sway, kick when struggling / falling."""
        b = self.body
        struggling = (self.phase == "hang" and (
            any(s in ("slipped", "regrab") for s in self.leg_state.values())
            or self.grip_pct() < 0.35 or t < self._flinch_until))
        flinching = self.phase == "hang" and t < self._flinch_until
        if self.phase in ("falling", "chomped"):
            amp, hz = 0.7, 9.0
        elif struggling:
            amp, hz = self.cfg.kick_amp * self._stress_mult(1.0), self.cfg.kick_hz
            if flinching and all(s == "grip" for s in self.leg_state.values()):
                amp *= self.cfg.flinch_kick
        elif self.phase == "missed":
            amp, hz = 0.15, 2.0
        else:
            amp, hz = 0.08, 1.1
        c = self.cfg
        dt = c.update_every_steps * self.sim.timestep
        want = c.brace and self.phase == "hang" and self.bracing_wanted()
        tau = c.brace_in_s if want else c.brace_out_s
        k = min(1.0, 2.5 * dt / max(tau, 1e-6))  # ~95 % of the way in tau
        self._brace_w += ((1.0 if want else 0.0) - self._brace_w) * k
        if want and not self._brace_on:
            self._aim_brace()  # a new brace: aim it from where the body hangs now
        self._brace_on = want
        wb = smoothstep(min(max(self._brace_w, 0.0), 1.0))
        # continuous phase and a low-passed amplitude: switching from kicking to the
        # lazy sway (or back) must not step the leg targets
        self._kick_amp += (amp - self._kick_amp) * min(1.0, dt / max(c.kick_tau_s, 1e-6))
        self._kick_ph = (self._kick_ph + 2 * math.pi * hz * dt) % (2 * math.pi)
        for k, leg in enumerate(REAR):
            cols = b.leg_mask([leg])
            q = self.dangle[cols].copy()
            idx = list(np.flatnonzero(cols))
            ph = self._kick_ph + k * 1.9
            braced = leg in self.q_brace and self.phase == "hang"
            a = self._kick_amp * ((1.0 - wb) if braced else 1.0)
            q[idx.index(b.idx(leg, "ctr_pitch"))] += a * math.sin(ph)
            q[idx.index(b.idx(leg, "fti"))] += a * 1.2 * math.sin(ph + 1.3)
            adh = 0.0
            if braced and wb > 0:
                q = (1.0 - wb) * q + wb * self.q_brace[leg]
                adh = c.brace_adhesion * wb
            self.targets[cols] = q
            self.adhesion[LEGS.index(leg)] = adh
        if self.phase != "hang":
            # front legs flail too, no adhesion (nothing to hold)
            for leg in FRONT:
                cols = b.leg_mask([leg])
                idx = list(np.flatnonzero(cols))
                q = b.stand[cols].copy()
                ph = 2 * math.pi * hz * t + (0.0 if leg == "lf" else 2.1)
                q[idx.index(b.idx(leg, "ctr_pitch"))] += 0.6 * math.sin(ph)
                q[idx.index(b.idx(leg, "fti"))] += 0.6 * math.sin(ph + 1.0)
                self.targets[cols] = q
                self.adhesion[LEGS.index(leg)] = 0.0

    def bracing_wanted(self) -> bool:
        """Brace with the mid legs while a front leg is (about to be) off the bar."""
        return any(s != "grip" for s in self.leg_state.values())

    def _aim_brace(self) -> None:
        """Re-aim the brace from the current body pose (it may have swung)."""
        c = self.cfg
        q = self.sim.data.qpos.copy()
        # the body may hang lower than at spawn (a tired grip slides down the bar):
        # try spots further round the front / underside of the bar and closer in
        spots = [(None, None)] + [(a, x) for x in (c.brace_x, 0.7 * c.brace_x)
                                  for a in (30.0, 55.0, 80.0)]
        b = self.body
        for leg in self.q_brace:
            q[b.qpos_adr[b.leg_mask([leg])]] = self.q_brace0[leg]  # start from the spawn brace
            best = None
            for a, x in spots:
                ang, err = self.ik.solve(q, leg, [("tarsus5", self.brace_target(leg, a, x))],
                                         iters=40, ref=self.q_brace0[leg])
                if best is None or err < best[1]:
                    best = (ang, err)
                if err < 0.05:
                    break
            self.q_brace[leg] = best[0] if best[1] < 0.6 else self.q_brace0[leg]
            self.brace_aim_err[leg] = best[1]

    def _fall(self, t: float) -> None:
        self.phase = "falling"
        self.state = "FALLING"
        self._t_phase = t
        self.n_drops += 1
        held = [self.strength[l] for l in FRONT if self.leg_state[l] == "grip"]
        kind = "two-arm" if len(held) == 2 else "one-arm"
        self.fall_causes[kind] = self.fall_causes.get(kind, 0) + 1
        self.last_fall_grip = max(held) if held else 0.0
        self._fall_grip_sum += self.last_fall_grip
        self.last_streak = self.run_time() - self.streak_t0
        self.best_streak = max(self.best_streak, self.last_streak)
        if self.trap_mode == "twitch":
            self.trap_mode = "open"
        self._say_flash("AAAAAAAAAH!", 1.0)
        k = self._streak
        legs = ", ".join(f"{l} {self.leg_state[l]} {self.strength[l]:.0%}" for l in FRONT)
        self.say(f"FELL off the bar after {self.last_streak:.1f} s ({legs}; this hang: "
                 f"{k['regrips']} re-grips, {k['slips']} slips, {k['regrabs']} re-grabs)")
        self.session.log_event("dead_hang_fall", streak_s=round(self.last_streak, 2),
                               grip=round(self.grip_pct(), 3))

    def _falling_tick(self, t: float, p: np.ndarray) -> None:
        c = self.cfg
        if self.in_trap_mouth(p) and self.trap_mode != "snap":
            self.snap()
            self.n_chomps += 1
            self.phase = "chomped"
            self.state = "CHOMPED"
            self._t_phase = t
            self._say_flash("*** CHOMP! ***", 2.5)
            self.say(f"CHOMP #{self.n_chomps}: the flytrap snapped shut")
            self.session.log_event("dead_hang_chomp", n=self.n_chomps)
            return
        if p[2] < self.rim_z_open() - 0.8 or (p[2] < self.soil_top + 1.2):
            self.n_escapes += 1
            self.phase = "missed"
            self.state = "missed the trap"
            self._t_phase = t
            self._say_flash("missed the trap! (lucky)", 2.0)
            self.say(f"missed the trap (escape #{self.n_escapes})")

    # ------------------------------------------------------------ outside sim.step
    def after_physics(self) -> None:
        super().after_physics()
        c = self.cfg
        s = self.session
        t = self.sim.time
        st = getattr(s, "stress", None)
        self.stress_level = (float(getattr(st, "level", 0.0))
                             if st is not None and getattr(st, "enabled", False) else 0.0)
        link = getattr(s, "brain", None)
        if link is not None:
            self._brain_tick(link)
        if self.phase == "hang" and t >= self._next_twitch:
            self.twitch()
            self._next_twitch = t + self._draw_twitch_gap()
        age = t - self._t_phase
        if self.phase == "chomped" and age >= c.chomp_hold_s:
            self._reopen_after_reset = True
            self.recover("chomped")
        elif self.phase == "missed" and age >= c.miss_hold_s:
            self.recover("missed_trap")
        elif self.phase == "falling" and age >= c.lost_after_s:
            self.recover("lost")
        if self.flash and self.run_time() > self._flash_until:
            self.flash = ""

    def handle_key(self, k: str) -> bool:
        """T: make the trap twitch (a looming threat for the brain)."""
        if k == "t":
            ok = self.twitch()
            self.say("trap twitch" if ok else "trap busy")
            return True
        return False

    # ------------------------------------------------------------ brain tie-in
    def _install_brain(self, link) -> None:
        from fly_simulator.interaction.swatter import swatter_response
        from fly_simulator.vision.looming import LoomingConfig, LoomingVision, VisualSource

        vis = LoomingVision(self.sim, LoomingConfig(), sink=link,
                            time_fn=getattr(link, "_run_time", None))

        def period():
            return 1e-3 if self.trap_mode in ("twitch", "snap") else None

        vis.add_source(VisualSource("flytrap", shapes=self.trap_shapes, period=period,
                                    response=swatter_response()))
        self.vision = vis.attach()
        trig = getattr(link, "triggers", None)
        if trig is not None:
            # on the bar a jump is suicide: the giant fibre makes the fly flinch
            def check_gf(gf, now, run_time, tag, _trig=trig):
                if gf > _trig.p.jump_escape_hz:
                    return [self.request_flinch(gf, "brain-actions")]
                return []

            trig._check_gf = check_gf

    def trap_shapes(self) -> tuple:
        """The lobes as spheres (centres along each lobe at mid height) for the
        looming computation: (centres, axes, half lengths, radii)."""
        d = self.sim.data
        c = self.cfg
        cs = []
        for gid in self.lobe_geoms:
            bid = self.sim.model.geom_bodyid[gid]
            R = d.xmat[bid].reshape(3, 3)
            o = d.xpos[bid]
            for xa in (-0.6, 0.0, 0.6):
                local = np.array([xa * c.lobe_half_len, 0.5 * c.lobe_cup, 0.5 * c.lobe_width])
                if gid == self.lobe_geoms[1]:
                    local[1] *= -1
                cs.append(o + R @ local)
        C = np.asarray(cs)
        n = len(C)
        ax = np.tile([0.0, 0.0, 1.0], (n, 1))
        if c.efference_copy and self.phase == "hang" and hasattr(self, "settled_thorax_R"):
            # the trap in the resting body frame, placed around the body as it is now:
            # the eyes see it as from the resting hang (self-motion cancelled)
            tb = self.sim.thorax_body_id
            M = d.xmat[tb].reshape(3, 3) @ self.settled_thorax_R.T
            C = d.xpos[tb] + (C - self.settled_thorax) @ M.T
            ax = ax @ M.T
        return C, ax, np.zeros(n), np.full(n, 0.9)

    def request_flinch(self, gf_hz: float, source: str) -> str:
        c = self.cfg
        t = self.sim.time
        self.n_gf_bursts += 1
        if self.phase != "hang" or t < self._flinch_ready:
            return f"[{self.name}] giant fibre {gf_hz:.0f} Hz (refractory / not hanging)"
        self._flinch_until = t + c.flinch_s
        self._flinch_ready = t + c.flinch_s + c.flinch_refractory_s
        self.n_flinches += 1
        for leg in FRONT:
            self.strength[leg] = max(0.0, self.strength[leg] - c.flinch_fatigue)
        self._say_flash("FLINCH! (giant fibre: no jumping!)", 1.5)
        msg = f"giant fibre {gf_hz:.0f} Hz ({source}) -> FLINCH (clench + kick), not a jump"
        self.say(msg)
        self.session.log_event("dead_hang_flinch", gf_hz=round(gf_hz, 1), source=source)
        return msg

    def _brain_tick(self, link) -> None:
        st = getattr(link, "latest", None)
        if st is None:
            return
        bt = float(getattr(st, "brain_time", 0.0) or 0.0)
        if bt <= self._gf_seen_t:
            return
        self._gf_seen_t = bt
        self.gf_hz = float((getattr(st, "descending", None) or {}).get("escape", 0.0) or 0.0)
        thr = self.cfg.gf_flinch_hz
        if self.gf_hz > thr and getattr(link, "triggers", None) is None:
            self.request_flinch(self.gf_hz, "brain")
        log = getattr(link, "stim_log", None)
        if isinstance(log, list) and len(log) > 400:
            del log[:-200]

    # ------------------------------------------------------------ HUD / camera
    def camera_target(self) -> np.ndarray:
        c = self.cfg
        # the bar sits just below the HUD text; the trap and the pot rim at the bottom
        return np.array([0.0, 0.3, c.bar_height - 1.5])

    def camera_preset(self) -> CameraPreset:
        return CameraPreset(azimuth=68.0, elevation=-9.0, distance=15.0, tau_s=0.3)

    def job_hud_lines(self) -> list[str]:
        g = self.grip_pct()
        n = 20
        k = int(round(g * n))
        bar = "#" * k + "-" * (n - k)
        lf, rf = (self.strength[l] if self.leg_state[l] == "grip" else None for l in FRONT)
        fmt = lambda v: f"{v:.0%}" if v is not None else "--"  # noqa: E731
        rate = (self.n_escapes / self.n_drops) if self.n_drops else 1.0
        lines = [
            f"HANG STREAK {self.streak():6.1f} s   best {self.best_streak:.1f} s   "
            f"CHOMPS {self.n_chomps}",
            f"GRIP [{bar}] {g:4.0%}  (L {fmt(lf)}  R {fmt(rf)}  capacity {self.cap:.0%})",
            f"re-grips {self.n_regrips}   slips {self.n_slips}   re-grabs {self.n_regrabs}   "
            f"survival rate {rate:.0%} ({self.n_escapes}/{self.n_drops} falls missed the trap)",
        ]
        if getattr(self.session, "brain", None) is not None:
            lines.append(f"trap twitches {self.n_twitches}   GF bursts {self.n_gf_bursts}   "
                         f"FLINCHES {self.n_flinches} (GF -> flinch, never jump)"
                         + (f"   stress {self.stress_level:.2f}" if self.stress_level else ""))
        if self.flash:
            lines.append(self.flash)
        return lines

    def job_stats(self) -> dict[str, Any]:
        return {
            "chomps": self.n_chomps, "drops": self.n_drops, "escapes": self.n_escapes,
            "regrips": self.n_regrips, "slips": self.n_slips, "regrabs": self.n_regrabs,
            "regrab_fails": self.n_regrab_fails, "lost_grips": self.n_lost, "best_streak_s": round(self.best_streak, 2),
            "fall_causes": dict(self.fall_causes),
            "mean_fall_grip": (round(self._fall_grip_sum / self.n_drops, 3)
                               if self.n_drops else None),
            "streak_s": round(self.streak(), 2), "grip": round(self.grip_pct(), 3),
            "capacity": round(self.cap, 3), "twitches": self.n_twitches,
            "flinches": self.n_flinches, "gf_bursts": self.n_gf_bursts, "phase": self.phase,
        }
