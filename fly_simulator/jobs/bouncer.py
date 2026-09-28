"""BOUNCER FLY: the fly works the door of a club forever (docs/JOBS.md, "bouncer").

A club entrance at night ("CLUB HALTERE", our own name): a brick facade, a padded
door, a neon sign, a red carpet, brass stanchions with velvet ropes and a queue of
guest flies. The bouncer is the simulated fly (the real NeuroMechFly body, walking
controller underneath a standing posture). It faces the queue (+x).

* **Guests (posed, kinematic; labelled).** A pool of NeuroMechFly copies (leg + wing
  joints, no actuators, no contacts, gravity compensated) on mocap mounts, tinted in
  a few colour variations. The job writes their joint angles and mount poses: a
  shuffling tripod pattern while they walk, standing still otherwise.
* **The step-up = a loom.** The guest at the head of the queue steps up to the door
  suddenly (``step_up_s``, a lunge profile that is fastest near the end). It is a
  ``LoomingVision`` source (head sphere + body capsule), so it drives LC4 / LPLC2 on
  both eyes through the looming machinery (``guest_response``: our interface
  tuning for a fly-sized object stepping up to ~1.8 mm, docs). With ``--brain`` the
  loom events go to the connectome; the giant fibre (DNp01) window rate above
  ``gf_flinch_hz`` makes the bouncer **FLINCH** (the dead_hang rule: a crouch of all
  legs with the front legs pulled in, 0.35 s, our motor pattern; never a jump: the
  brain-actions jump trigger is replaced).
* **Habituation.** The job turns the LC4 / LPLC2 -> GF short-term depression
  (``fly_simulator/brain/habituation.py``, docs/HABITUATION.md) on by default, so
  the repeated harmless step-ups depress the GF's looming input over the shift and
  the flinches die out. A **rowdy** guest (occasionally) spreads its wings, lunges
  faster and closer (a bigger, faster loom) and chest-bumps the bouncer: the bump is
  sent to the brain as a ``shove`` touch stimulus, which triggers the model's
  dishabituation shortcut (``dishabituate_frac``; not a real physical push).
* **Decision rule (ours).** A rowdy guest is always turned away; so is everyone
  while the club is at capacity (``capacity``; guests inside leave over time).
  Everyone else is let in: the velvet rope unclips and swings open (kinematic), the
  guest walks through the door. Turned away: the bouncer waves a front leg (a small
  gesture, joint targets) and the guest walks off.
* **Without a brain** the flinch is a labelled scripted probability that decays
  with the number of recent harmless guests and resets partly after a rowdy one.
* **Shifts.** A night runs from 22:00 to 04:00 on the club clock (``shift_s`` sim
  s); then the door closes for ``closed_s``, the queue refills, the per-night
  counters reset and (with a brain) the depression is reset too (a day's rest is far
  longer than its recovery time constant): a new night. It never ends.

Counters: guests admitted / turned away (the work counter is guests handled),
flinches, flinch rate per club hour and per sim hour, rowdy guests, GF peak per guest
(brain).
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import mujoco as mj
import numpy as np

from fly_simulator.actions.base import LEGS, Action, ActionCommand, smoothstep
from fly_simulator.jobs import bouncer_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import contact_kwargs, quat_axis_angle, spot_or_directional
from fly_simulator.jobs.registry import register_job

P = "bnc/"
GUEST = "guest"
LEG_JOINTS = ("c_thorax-{l}_coxa-yaw", "c_thorax-{l}_coxa-pitch", "c_thorax-{l}_coxa-roll",
              "{l}_coxa-{l}_trochanterfemur-pitch", "{l}_coxa-{l}_trochanterfemur-roll",
              "{l}_trochanterfemur-{l}_tibia-pitch", "{l}_tibia-{l}_tarsus1-pitch")
WING_JOINTS = ("c_thorax-{w}_wing-yaw", "c_thorax-{w}_wing-pitch", "c_thorax-{w}_wing-roll")
_PASSIVE_TARSUS = re.compile(r"_tarsus\d-.*_tarsus\d")
# colour variations of the guests (rgb multipliers of the NeuroMechFly materials)
TINTS = ((1.0, 1.0, 1.0), (0.75, 0.85, 1.25), (0.8, 1.2, 0.8), (1.25, 1.05, 0.6),
         (1.15, 0.75, 1.2), (1.3, 1.3, 1.3), (0.6, 0.6, 0.65))
TRIPOD_A = ("lf", "rm", "lh")
CLUB_HOURS = 6  # 22:00 -> 04:00


def guest_response(lc4_v0: float = 4.0, lc4_vscale: float = 10.0, lplc2_theta0: float = 30.0,
                   lplc2_theta1: float = 60.0, max_hz: float = 200.0):
    """LC4 / LPLC2 response for a guest stepping up (``LoomResponse``; interface
    choice, docs/JOBS.md "bouncer"). A guest stepping up expands at ~25-40 deg/s on
    the bouncer's eyes to ~32 deg; a rowdy one (wings spread, a fast lunge) at
    ~60-150 deg/s to ~40 deg. LC4 = 200 tanh(max(0, dtheta - 4) / 10) Hz for
    theta >= 10 deg (both saturate near 200 Hz); LPLC2 = 200 ramp(theta; 30, 60
    deg) ramp(dtheta; 4, 14 deg/s): a normal guest barely reaches LPLC2's size range
    (<= ~10 Hz), a rowdy one drives it (~60 Hz). Plausible looming-detector tunings
    (LC4 ~ angular velocity, LPLC2 ~ large expanding objects; silent for small,
    still or receding objects), not fits to recordings."""
    from fly_simulator.vision.looming import LoomResponse

    return LoomResponse(max_hz=float(max_hz), lc4_v0=float(lc4_v0), lc4_vscale=float(lc4_vscale),
                        lc4_theta0=10.0, lplc2_theta0=float(lplc2_theta0), lplc2_theta1=float(lplc2_theta1),
                        lplc2_gate_v0=4.0, lplc2_gate_v1=14.0, min_hz=10.0)


def loom_progress(u: float, d0: float, d1: float) -> float:
    """Step-up progress 0 -> 1 such that 1 / distance grows linearly in time (the
    guest's angular size grows at a near-constant rate: it steps up fast while far
    and leans in slowly at the end, a steady loom). ``d0`` / ``d1``: the start /
    end distance from the bouncer's eyes."""
    u = min(max(u, 0.0), 1.0)
    d = 1.0 / (1.0 / d0 + u * (1.0 / d1 - 1.0 / d0))
    return (d0 - d) / max(d0 - d1, 1e-9)


def lunge(u: float, brake: float = 0.08) -> float:
    """Step-up progress 0 -> 1: accelerates over the first 40 %, then fast and
    straight, braking only in the last ``brake`` (a sudden step up)."""
    u = min(max(u, 0.0), 1.0)
    a = 0.4
    b = 1.0 - brake
    # piecewise velocity: ramp 0..1 on [0, a], 1 on [a, b], ramp 1..0 on [b, 1]
    tot = 0.5 * a + (b - a) + 0.5 * (1.0 - b)
    if u < a:
        s = 0.5 * u * u / a
    elif u < b:
        s = 0.5 * a + (u - a)
    else:
        w = u - b
        s = 0.5 * a + (b - a) + w - 0.5 * w * w / (1.0 - b)
    return s / tot


# ---------------------------------------------------------------------------
# the bouncer's posture
# ---------------------------------------------------------------------------


class BouncerStance(Action):
    """Stand still with all tarsi adhering (like ``freeze``). ``flinch`` (0-1): a
    crouch (femur-tibia flexion of all legs, the front legs pulled in); ``wave``
    (0-1): the right front leg lifts and waves (the "not tonight" gesture), its
    adhesion off; ``wave_phase`` drives the wave. Stationary; our motor pattern."""

    name = "bouncer_stance"
    blend_in = 0.0
    blend_out = 0.3

    def __init__(self, duration: float = 3600.0, flex: float = 0.35) -> None:
        super().__init__(duration)
        self.flinch = 0.0
        self.wave = 0.0
        self.wave_phase = 0.0
        self.flex = float(flex)

    def begin(self, mgr) -> None:
        b = mgr.body
        self._start = mgr.sim.data.ctrl[b.pos_ids].copy()
        self._stand = b.stand.copy()
        self._fti = np.array([b.idx(leg, "fti") for leg in LEGS])
        self._front_pitch = np.array([b.idx(leg, "thc_pitch") for leg in ("lf", "rf")])
        self._front_ctr = np.array([b.idx(leg, "ctr_pitch") for leg in ("lf", "rf")])
        self._rf = {j: b.idx("rf", j) for j in ("thc_yaw", "thc_pitch", "thc_roll", "ctr_pitch", "fti", "tita")}
        self._rf_i = LEGS.index("rf")

    def command(self, mgr, t: float) -> ActionCommand:
        a = smoothstep(t / 0.2)
        tg = (1 - a) * self._start + a * self._stand
        f = float(min(max(self.flinch, 0.0), 1.0))
        if f > 0:
            tg[self._fti] += self.flex * f  # crouch
            tg[self._front_pitch] -= 0.25 * f  # front legs pulled in (a guard)
            tg[self._front_ctr] -= 0.2 * f
        adh = np.ones(6)
        w = float(min(max(self.wave, 0.0), 1.0))
        if w > 0:
            r = self._rf
            tg[r["thc_pitch"]] -= 0.6 * w  # the leg swings forward and up
            tg[r["ctr_pitch"]] -= 0.4 * w
            tg[r["fti"]] -= 0.5 * w  # tibia out (a raised "stop" leg)
            tg[r["thc_roll"]] += 0.35 * w * math.sin(self.wave_phase)  # the wave
            tg[r["tita"]] -= 0.3 * w
            if w > 0.05:
                adh[self._rf_i] = 0.0
        return ActionCommand(targets=tg, adhesion=adh)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class BouncerConfig(JobConfig):
    # --- layout (mm; the bouncer stands at the origin facing +x, the queue along +x)
    door_spot_x: float = 2.55  # a guest's thorax at the door (facing the bouncer)
    queue_x0: float = 6.0  # the first queue slot
    queue_gap: float = 2.9
    n_guests: int = 6  # pooled NeuroMechFly copies (1 at the door, the queue, 1 leaving)
    wall_y: float = 3.6
    door_x: float = 1.45
    door_half_w: float = 0.95
    gate_y: float = 1.85
    gate_x0: float = 0.45  # the fixed stanchion (the rope's clip end)
    gate_x1: float = 2.45  # the hinge stanchion (the rope stays hooked here)
    gate_open_deg: float = -95.0
    rope_line_y: float = -1.45
    # --- timing (sim s) -----------------------------------------------------------
    step_up_s: float = 1.1
    loom_response: dict = field(default_factory=dict)  # guest_response() overrides
    eye_x: float = 0.75  # the bouncer's eyes (x) for the step-up profile
    check_s: float = 0.9
    gate_s: float = 0.35
    admit_walk_s: float = 1.3
    wave_s: float = 1.0
    leave_walk_s: float = 1.2
    shuffle_s: float = 1.3
    arrive_s: float = 2.0
    # --- rowdy guests (a bigger, faster loom + a chest bump) ------------------------
    rowdy_p: float = 0.08
    rowdy_min_gap: int = 6  # at least this many guests between two rowdy ones
    rowdy_step_up_s: float = 0.4
    rowdy_stop_x: float = 2.38  # comes closer than a normal guest (in the bouncer's face)
    rowdy_bump: bool = True  # the chest bump -> a "shove" touch stimulus to the brain
    rowdy_bump_intensity: float = 0.6
    # --- the rule (ours) ------------------------------------------------------------
    capacity: int = 18
    leave_tau_s: float = 50.0  # guests inside leave at inside / leave_tau per s
    # --- the night ----------------------------------------------------------------
    shift_s: float = 180.0  # 22:00 -> 04:00 on the club clock
    closed_s: float = 12.0
    # --- brain (--brain) ------------------------------------------------------------
    gf_flinch_hz: float = 60.0  # giant fibre (DNp01) window rate -> flinch (the jump rule)
    flinch_s: float = 0.35
    flinch_refractory_s: float = 1.0
    flinch_flex: float = 0.35
    gf_window_s: float = 1.6  # GF peak per guest: from its step-up start this long
    efference_copy: bool = True  # the loom is computed from the resting eyes
    habituation: bool = True  # turn the LC4 / LPLC2 -> GF depression on (brain)
    habituation_config: dict = field(default_factory=lambda: {
        "u": 0.006, "tau_rec_s": 20.0, "dishabituate_frac": 0.6})
    reset_depression_each_night: bool = True
    # --- scripted flinch (no brain; labelled) ------------------------------------------
    script_p0: float = 0.95
    script_p_floor: float = 0.03
    script_h_scale: float = 2.5  # p = floor + (p0 - floor) exp(-h / scale)
    script_tau_s: float = 40.0  # h decays with this time constant, +1 per harmless guest
    script_dishab: float = 0.7  # a rowdy guest: h *= 1 - dishab
    # --- looks --------------------------------------------------------------------------
    shadows: bool = True


# ---------------------------------------------------------------------------
# a guest (pooled kinematic copy)
# ---------------------------------------------------------------------------


@dataclass
class Guest:
    k: int
    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    yaw: float = math.pi
    state: str = "hidden"  # hidden / arriving / queue / stepping / door / entering / leaving
    slot: int = -1
    rowdy: bool = False
    tint: int = 0
    path: list = field(default_factory=list)  # [(p0, p1, yaw0, yaw1, t0, T, profile)]
    walk_phase: float = 0.0
    wings: float = 0.0  # 0 folded .. 1 spread (rowdy)
    number: int = 0  # guest number of the night


@register_job
class BouncerJob(EternalJob):
    name = "bouncer"
    znear = 0.05
    title = "BOUNCER FLY"
    tagline = "the fly works the door of a club forever"
    work_label = "guests handled"
    config_cls = BouncerConfig
    required_names = (P + "gate", P + "neon_g", GUEST + "0/c_head")

    def __init__(self, cfg: BouncerConfig | None = None) -> None:
        super().__init__(cfg)
        c = self.cfg
        self.rng = np.random.default_rng(c.seed + 41)
        self.guests = [Guest(k) for k in range(c.n_guests)]
        self.queue: list[int] = []
        self.door: int | None = None
        self.phase = "setup"
        self._t_phase = 0.0
        self.night = 1
        self.night_t0 = 0.0
        self.inside = 0.0
        self.n_admitted = self.n_turned = self.n_flinches = self.n_rowdy = 0
        self.night_admitted = self.night_turned = self.night_flinches = self.night_guests = 0
        self.night_rowdy = 0
        self.hour_guests = np.zeros(CLUB_HOURS, int)
        self.hour_flinches = np.zeros(CLUB_HOURS, int)
        self.n_gf_bursts = 0
        self.n_bumps = 0
        self.gf_hz = 0.0
        self.guest_no = 0
        self._since_rowdy = 99
        self.decision = ""
        self.gf_log: deque = deque(maxlen=120)  # (night, guest no, rowdy, GF peak, flinched)
        self._gf_cur: dict | None = None
        self._flinch_until = -1e9
        self._flinch_ready = -1e9
        self.flash = ""
        self._flash_until = -1e9
        self.script_h = 0.0
        self._script_t = 0.0
        self.vision = None
        self._gf_seen_t = -1.0
        self.rest_eyes = None
        self._pending_script_flinch = None

    # ------------------------------------------------------------ build
    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        orig_add_fly = world.add_fly

        def add_fly(fly, *a, **kw):
            world.add_fly = orig_add_fly
            self._dress_bouncer(fly)
            return orig_add_fly(fly, *a, **kw)

        world.add_fly = add_fly
        self._materials(spec)
        self._add_street(spec)
        self._add_facade(spec)
        self._add_ropes(spec)
        for k in range(c.n_guests):
            self._add_guest(spec, k)
        self._add_lights(spec)

    def _materials(self, spec) -> None:
        c = self.cfg
        T = A.add_textured_material
        A.add_texture(spec, P + "tex_brick", A.brick_texture(c.seed))
        T(spec, P + "brick", P + "tex_brick", rgba=(1, 1, 1, 1), specular=0.05)
        A.add_texture(spec, P + "tex_neon", A.neon_texture())
        T(spec, P + "neon", P + "tex_neon", rgba=(1, 1, 1, 1), emission=1.0, specular=0.0)
        A.add_texture(spec, P + "tex_pave", A.pavement_texture(c.seed))
        T(spec, P + "pave", P + "tex_pave", rgba=(1, 1, 1, 1), specular=0.35, shininess=0.6, texrepeat=(8, 8))
        A.add_texture(spec, P + "tex_carpet", A.carpet_texture())
        T(spec, P + "carpet", P + "tex_carpet", rgba=(1, 1, 1, 1), specular=0.05)
        A.add_texture(spec, P + "tex_door", A.door_texture())
        T(spec, P + "door", P + "tex_door", rgba=(1, 1, 1, 1), specular=0.4, emission=0.05)
        for k in range(4):
            A.add_texture(spec, P + f"tex_poster{k}", A.poster_texture(k))
            T(spec, P + f"poster{k}", P + f"tex_poster{k}", rgba=(1, 1, 1, 1), specular=0.1, emission=0.08)
        spec.add_material(name=P + "brass", rgba=(0.85, 0.66, 0.3, 1), specular=1.0, shininess=0.9)
        spec.add_material(name=P + "velvet", rgba=(0.55, 0.02, 0.08, 1), specular=0.25, shininess=0.3)
        spec.add_material(name=P + "black", rgba=(0.03, 0.03, 0.04, 1), specular=0.2)
        spec.add_material(name=P + "interior", rgba=(0.16, 0.04, 0.2, 1), emission=0.25)
        spec.add_material(name=P + "frame", rgba=(0.1, 0.09, 0.1, 1), specular=0.6, shininess=0.7)
        spec.add_material(name=P + "curb", rgba=(0.3, 0.3, 0.32, 1), specular=0.1)
        spec.add_material(name=P + "road", rgba=(0.06, 0.06, 0.07, 1), specular=0.5, shininess=0.7)
        spec.add_material(name=P + "lamp", rgba=(1.0, 0.85, 0.55, 1), emission=1.0)
        spec.add_material(name=P + "pole", rgba=(0.12, 0.13, 0.14, 1), specular=0.5)
        spec.add_material(name=P + "window", rgba=(0.9, 0.6, 0.3, 1), emission=0.55)
        spec.add_material(name=P + "shades", rgba=(0.02, 0.02, 0.03, 1), specular=1.0, shininess=1.0)
        spec.add_material(name=P + "earpiece", rgba=(0.1, 0.1, 0.1, 1), specular=0.6)
        spec.add_material(name=P + "wire", rgba=(0.85, 0.85, 0.8, 1), specular=0.5)

    def _dress_bouncer(self, fly) -> None:
        """Sunglasses and an earpiece on the bouncer's head (visual only)."""
        root = fly.mjcf_root
        head = root.body("c_head")
        vis = dict(contype=0, conaffinity=0, group=1, mass=0.0)
        root.add_material(name="bnc_shades", rgba=(0.02, 0.02, 0.03, 1), specular=1.0, shininess=1.0,
                          reflectance=0.2)
        root.add_material(name="bnc_wire", rgba=(0.85, 0.85, 0.8, 1), specular=0.5)
        # sunglasses: a dark shell over the front of each compound eye (eye body frame:
        # the eye's centre is 0.31 mm to the side), a bridge across the frons; an earpiece
        for side, sy in (("l", 1), ("r", -1)):
            eye = root.body(f"{side}_eye")
            eye.add_geom(name=f"bnc_lens_{side}", type=mj.mjtGeom.mjGEOM_ELLIPSOID,
                         size=(0.2, 0.07, 0.19), pos=(0.07, 0.345 * sy, 0.02),
                         quat=quat_axis_angle((0, 0, 1), -0.35 * sy), material="bnc_shades", **vis)
        head.add_geom(name="bnc_bridge", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.025, 0.16, 0),
                      pos=(0.45, 0.0, 0.08), quat=quat_axis_angle((1, 0, 0), math.pi / 2),
                      material="bnc_shades", **vis)
        head.add_geom(name="bnc_ear", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.06, 0, 0),
                      pos=(0.1, -0.5, -0.12), material="bnc_shades", **vis)
        head.add_geom(name="bnc_coil", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.014, 0.2, 0),
                      pos=(0.0, -0.5, -0.32), quat=quat_axis_angle((1, 0, 0), 0.25),
                      material="bnc_wire", **vis)

    def _add_street(self, spec) -> None:
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        wb.add_geom(name=P + "pavement", type=mj.mjtGeom.mjGEOM_BOX, size=(30.0, 7.0, 0.05),
                    pos=(4.0, 1.0, -0.052), material=P + "pave", **vis)
        wb.add_geom(name=P + "curb", type=mj.mjtGeom.mjGEOM_BOX, size=(30.0, 0.25, 0.12),
                    pos=(4.0, -6.0, -0.04), material=P + "curb", **vis)
        wb.add_geom(name=P + "road", type=mj.mjtGeom.mjGEOM_BOX, size=(30.0, 12.0, 0.05),
                    pos=(4.0, -18.2, -0.4), material=P + "road", **vis)
        # the red carpet from the door spot to the door
        c = self.cfg
        wb.add_geom(name=P + "carpet", type=mj.mjtGeom.mjGEOM_BOX,
                    size=(0.75, (c.wall_y + 0.3) / 2, 0.003),
                    pos=(c.door_x, (c.wall_y - 0.3) / 2, 0.0005), material=P + "carpet", **vis)
        # a street lamp
        wb.add_geom(name=P + "lamp_pole", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.12, 5.5, 0),
                    pos=(13.0, -5.4, 5.5), material=P + "pole", **vis)
        wb.add_geom(name=P + "lamp_arm", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.08, 0.8, 0),
                    pos=(13.0, -4.6, 11.0), quat=quat_axis_angle((1, 0, 0), math.pi / 2),
                    material=P + "pole", **vis)
        wb.add_geom(name=P + "lamp_head", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(0.5, 0.35, 0.18),
                    pos=(13.0, -3.8, 10.85), material=P + "lamp", **vis)
        sky = spec.texture("skybox")
        if sky is not None:
            sky.rgb1 = (0.03, 0.04, 0.09)
            sky.rgb2 = (0.0, 0.0, 0.01)

    def _add_facade(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        y0, t = c.wall_y, 0.3
        x0, x1 = c.door_x - c.door_half_w, c.door_x + c.door_half_w
        H, dh = 13.0, 3.5  # wall height, door height
        L0, L1 = -14.0, 26.0
        for nm, xa, xb, za, zb in (("wall_l", L0, x0, 0.0, H), ("wall_r", x1, L1, 0.0, H),
                                   ("wall_top", x0, x1, dh, H)):
            md = A.panel_mesh(xb - xa, zb - za, t)
            # bricks at a fixed size: the texture repeats every 4 x 2 mm
            md = A.MeshData(md.verts, md.faces, md.uv * np.array([(xb - xa) / 4.0, (zb - za) / 2.0]))
            A.add_mesh(spec, P + nm + "_mesh", md)
            wb.add_geom(name=P + nm, type=mj.mjtGeom.mjGEOM_MESH, meshname=P + nm + "_mesh",
                        pos=((xa + xb) / 2, y0 + t / 2, (za + zb) / 2), quat=A.panel_quat((0, -1, 0)),
                        material=P + "brick", **vis)
        # the doorway: frame, the (open) padded door swung inward, the dim interior
        for sx in (x0, x1):
            wb.add_geom(name=P + f"jamb{int(sx > c.door_x)}", type=mj.mjtGeom.mjGEOM_BOX,
                        size=(0.08, t / 2 + 0.03, dh / 2), pos=(sx, y0 + t / 2, dh / 2),
                        material=P + "frame", **vis)
        wb.add_geom(name=P + "lintel", type=mj.mjtGeom.mjGEOM_BOX, size=(c.door_half_w + 0.08, t / 2 + 0.03, 0.08),
                    pos=(c.door_x, y0 + t / 2, dh), material=P + "frame", **vis)
        A.add_mesh(spec, P + "door_mesh", A.panel_mesh(2 * c.door_half_w - 0.1, dh - 0.1, 0.08))
        hinge = np.array([x1 - 0.05, y0 + t])
        ang = math.radians(70.0)
        dvec = np.array([-math.cos(ang), math.sin(ang)])
        mid = hinge + dvec * (c.door_half_w - 0.05)
        wb.add_geom(name=P + "door", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "door_mesh",
                    pos=(mid[0], mid[1], dh / 2), quat=A.panel_quat((-dvec[1], -dvec[0], 0.0)),
                    material=P + "door", **vis)
        wb.add_geom(name=P + "interior", type=mj.mjtGeom.mjGEOM_BOX, size=(3.0, 0.05, dh / 2),
                    pos=(c.door_x, y0 + 2.6, dh / 2), material=P + "interior", **vis)
        wb.add_geom(name=P + "interior_floor", type=mj.mjtGeom.mjGEOM_BOX, size=(3.0, 1.3, 0.02),
                    pos=(c.door_x, y0 + 1.3, 0.0), material=P + "black", **vis)
        # the neon sign above the door (emissive), on two brackets
        A.add_mesh(spec, P + "neon_mesh", A.panel_mesh(8.4, 1.8, 0.08))
        nb = wb.add_body(name=P + "neon", pos=(c.door_x + 3.2, y0 - 0.25, dh + 2.0))
        nb.add_geom(name=P + "neon_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "neon_mesh",
                    quat=A.panel_quat((0, -1, 0)), material=P + "neon", **vis)
        for sx in (-3.5, 3.5):
            nb.add_geom(name=P + f"neon_bracket{int(sx > 0)}", type=mj.mjtGeom.mjGEOM_BOX,
                        size=(0.05, 0.14, 0.05), pos=(sx, 0.14, 0.95), material=P + "frame", **vis)
        # posters and lit windows upstairs
        for k, (px, pz) in enumerate(((7.4, 1.9), (10.0, 1.9), (-3.4, 1.9), (12.6, 1.9))):
            A.add_mesh(spec, P + f"poster{k}_mesh", A.panel_mesh(2.0, 2.9, 0.02))
            wb.add_geom(name=P + f"poster{k}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + f"poster{k}_mesh",
                        pos=(px, y0 - 0.02, pz), quat=A.panel_quat((0, -1, 0)), material=P + f"poster{k}", **vis)
        for k, wx in enumerate((-6.0, -1.0, 8.5, 14.0, 19.0)):
            wb.add_geom(name=P + f"window{k}", type=mj.mjtGeom.mjGEOM_BOX, size=(1.2, 0.02, 1.4),
                        pos=(wx, y0 - 0.02, 9.8), material=P + "window", **vis)
        # a tiny "GUEST LIST" lectern next to the bouncer
        wb.add_geom(name=P + "lectern", type=mj.mjtGeom.mjGEOM_BOX, size=(0.3, 0.35, 0.8),
                    pos=(-1.5, 1.4, 0.8), material=P + "frame", **vis)
        wb.add_geom(name=P + "lectern_top", type=mj.mjtGeom.mjGEOM_BOX, size=(0.38, 0.42, 0.03),
                    pos=(-1.5, 1.4, 1.62), quat=quat_axis_angle((0, 1, 0), -0.25), material=P + "brass", **vis)

    def _add_ropes(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        A.add_mesh(spec, P + "stanchion_mesh", A.stanchion_mesh())
        hz = 1.72  # rope hook height
        posts = [(c.gate_x0, c.gate_y), (c.gate_x1, c.gate_y)]
        line = [(x, c.rope_line_y) for x in np.arange(4.1, 20.0, 3.0)]
        for i, (x, y) in enumerate(posts + line):
            wb.add_geom(name=P + f"post{i}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "stanchion_mesh",
                        pos=(x, y, 0.0), material=P + "brass", **vis)
        for i in range(len(line) - 1):
            a, b = np.array([*line[i], hz]), np.array([*line[i + 1], hz])
            A.add_mesh(spec, P + f"rope{i}_mesh", A.polyline_tube(A.catenary(a, b, 0.45), 0.06, 8))
            wb.add_geom(name=P + f"rope{i}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + f"rope{i}_mesh",
                        material=P + "velvet", **vis)
        # the gate rope: hooked on the hinge stanchion (x1), clipped to x0; the body
        # turns about the hinge post to open (kinematic)
        g = wb.add_body(name=P + "gate", mocap=True, pos=(c.gate_x1, c.gate_y, hz))
        L = c.gate_x1 - c.gate_x0
        pts = A.catenary((0, 0, 0), (-L, 0, 0), 0.4)
        A.add_mesh(spec, P + "gate_rope_mesh", A.polyline_tube(pts, 0.065, 8))
        g.add_geom(name=P + "gate_rope", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "gate_rope_mesh",
                   material=P + "velvet", **vis)
        for nm, x in (("gate_hook0", 0.0), ("gate_hook1", -L)):
            g.add_geom(name=P + nm, type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.09, 0, 0), pos=(x, 0, 0),
                       material=P + "brass", **vis)

    def _add_guest(self, spec, k: int) -> None:
        """A NeuroMechFly copy on a mocap mount: leg + wing joints only, no
        actuators, no contacts, gravity compensated (posed by the job)."""
        from flygym.anatomy import AnatomicalJoint, AxesSet, AxisOrder, JointPreset, Skeleton
        from flygym.compose import KinematicPosePreset, NeuroMechFly

        name = f"{GUEST}{k}"
        fly = NeuroMechFly(name=name)
        joints = JointPreset.LEGS_ONLY.to_joint_list() + [
            AnatomicalJoint("c_thorax", f"{w}_wing", AxesSet(["yaw", "pitch", "roll"])) for w in ("l", "r")]
        sk = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, anatomical_joints=joints)
        neutral = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
        fly.add_joints(sk, neutral_pose=neutral, stiffness=0.0, damping=0.05)
        try:
            fly.colorize()
        except Exception:
            pass
        root = fly.mjcf_root
        for j in list(root.joints):
            if _PASSIVE_TARSUS.search(j.name):
                root.delete(j)
            else:
                j.limited = mj.mjtLimited.mjLIMITED_FALSE
                j.stiffness[0] = 0.0
                j.armature = 1e-6
        for b in root.bodies:
            if b.name != "world":
                b.gravcomp = 1.0
        for g in root.geoms:
            g.contype = 0
            g.conaffinity = 0
        for key in list(root.keys):
            root.delete(key)
        mount = spec.worldbody.add_body(name=P + f"mount{k}", mocap=True, pos=(0.0, 0.0, -30.0 - 3 * k))
        site = mount.add_site(name=P + f"mount{k}_site")
        spec.attach(root, prefix=name + "/", site=site)

    def _add_lights(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        spec.visual.headlight.ambient = (0.12, 0.1, 0.16)
        spec.visual.headlight.diffuse = (0.2, 0.18, 0.24)
        spec.visual.headlight.specular = (0.05, 0.05, 0.05)
        key = np.array([6.0, -9.0, 12.0])
        tgt = np.array([2.5, 0.5, 0.5])
        wb.add_light(name=P + "key", type=spot_or_directional(c.shadows), pos=tuple(key), dir=tuple(tgt - key),
                     diffuse=(0.55, 0.48, 0.42), specular=(0.3, 0.3, 0.3), cutoff=35.0, exponent=1.0,
                     castshadow=bool(c.shadows))
        wb.add_light(name=P + "neon_glow", type=mj.mjtLightType.mjLIGHT_POINT,
                     pos=(c.door_x + 3.2, c.wall_y - 2.0, 5.2), diffuse=(0.7, 0.2, 0.55),
                     specular=(0.3, 0.1, 0.3), attenuation=(0.4, 0.05, 0.01), castshadow=False)
        wb.add_light(name=P + "door_light", type=mj.mjtLightType.mjLIGHT_POINT,
                     pos=(c.door_x, c.wall_y - 0.8, 3.9), diffuse=(0.6, 0.45, 0.3),
                     specular=(0.2, 0.2, 0.2), attenuation=(0.3, 0.1, 0.03), castshadow=False)
        wb.add_light(name=P + "street", type=mj.mjtLightType.mjLIGHT_POINT, pos=(13.0, -3.8, 10.2),
                     diffuse=(0.5, 0.42, 0.28), specular=(0.2, 0.2, 0.2), attenuation=(0.5, 0.02, 0.002),
                     castshadow=False)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        m, s = self.sim.model, self.session
        c = self.cfg
        gp = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "ground_plane")
        if gp >= 0:  # the fly's floor stays (physics); the pavement draws it
            m.geom_group[gp] = 3
            m.geom_rgba[gp, 3] = 0.0
        self.acts = s.actions
        self.body = s.actions.body
        self.gate_mocap = int(m.body_mocapid[m.body(P + "gate").id])
        self.gate_home = m.body(P + "gate").pos.copy()
        self.gate_open = 0.0  # 0 clipped .. 1 open
        self.gate_target = 0.0
        # guests: joint addresses, mounts, materials, the standing pose
        body = self.body
        stand = {n: float(v) for n, v in zip(body.names, body.stand)}
        names = [jn.format(l=leg) for leg in LEGS for jn in LEG_JOINTS]
        wnames = [jn.format(w=w) for w in ("l", "r") for jn in WING_JOINTS]
        self.g_q0 = np.array([stand.get(n, 0.0) for n in names])
        self.g_qadr, self.g_wadr, self.g_dofs, self.g_mocap, self.g_mats, self.g_mat0 = [], [], [], [], [], []
        for k in range(c.n_guests):
            pre = f"{GUEST}{k}/"
            jid = [m.joint(pre + n).id for n in names]
            wid = [m.joint(pre + n).id for n in wnames]
            self.g_qadr.append(np.array([m.jnt_qposadr[i] for i in jid]))
            self.g_wadr.append(np.array([m.jnt_qposadr[i] for i in wid]))
            mb = m.body(P + f"mount{k}").id
            self.g_mocap.append(int(m.body_mocapid[mb]))
            self.g_dofs.append(np.flatnonzero(m.body_rootid[m.dof_bodyid] == m.body_rootid[mb]))
            mats = sorted({int(m.geom_matid[g]) for g in range(m.ngeom)
                           if m.geom_matid[g] >= 0 and m.body_rootid[m.geom_bodyid[g]] == m.body_rootid[mb]})
            self.g_mats.append(np.array(mats, int))
            self.g_mat0.append(m.mat_rgba[mats].copy())
        self.g_head = [m.body(f"{GUEST}{k}/c_head").id for k in range(c.n_guests)]
        self._guest_height()
        # the bouncer's stance
        st = getattr(s, "STATIONARY_ACTIONS", None)
        if st is not None and BouncerStance.name not in st:
            s.STATIONARY_ACTIONS = (*st, BouncerStance.name)
        self._stance: BouncerStance | None = None
        self._begin()
        self._new_night(first=True)
        self._install_vision(getattr(s, "brain", None))

    def _guest_height(self) -> None:
        """Mount height so a standing guest's tarsi touch the ground; the offsets of
        its head and abdomen in the mount frame (for the looming shapes)."""
        m = self.sim.model
        d = mj.MjData(m)
        k = 0
        d.mocap_pos[self.g_mocap[k]] = (0.0, 0.0, 0.0)
        d.mocap_quat[self.g_mocap[k]] = (1.0, 0.0, 0.0, 0.0)
        d.qpos[self.g_qadr[k]] = self.g_q0
        mj.mj_kinematics(m, d)
        pre = f"{GUEST}{k}/"
        zs = [d.xpos[m.body(f"{pre}{leg}_tarsus5").id][2] for leg in LEGS]
        self.g_z = -float(np.mean(zs)) + 0.02
        self.g_head_off = d.xpos[m.body(pre + "c_head").id].copy()
        self.g_abd_off = d.xpos[m.body(pre + "a_a3").id].copy() if mj.mj_name2id(
            m, mj.mjtObj.mjOBJ_BODY, pre + "a_a3") >= 0 else np.array([-0.9, 0.0, -0.1])
        self.g_head_off[2] += self.g_z
        self.g_abd_off[2] += self.g_z

    def _begin(self) -> None:
        self._stance = BouncerStance(flex=self.cfg.flinch_flex)
        self.acts.trigger(self._stance, source="job")
        self.steering.set(None, 0.0)

    # ------------------------------------------------------------ the night
    def club_hour(self, t: float | None = None) -> int:
        t = self.sim.time if t is None else t
        f = (t - self.night_t0) / max(self.cfg.shift_s, 1e-6)
        return int(min(max(f * CLUB_HOURS, 0), CLUB_HOURS - 1))

    def club_clock(self) -> str:
        f = (self.sim.time - self.night_t0) / max(self.cfg.shift_s, 1e-6)
        mins = int(min(max(f, 0.0), 1.0) * CLUB_HOURS * 60)
        h, mm = divmod(22 * 60 + mins, 60)
        return f"{h % 24:02d}:{mm:02d}"

    def _new_night(self, first: bool = False) -> None:
        c = self.cfg
        t = self.sim.time
        if not first:
            self.night += 1
        self.night_t0 = t
        self.night_admitted = self.night_turned = self.night_flinches = self.night_guests = 0
        self.night_rowdy = 0
        self.hour_guests[:] = 0
        self.hour_flinches[:] = 0
        self.inside = 0.0
        self.guest_no = 0
        self.script_h = 0.0
        self._script_t = t
        self._since_rowdy = 0  # the first guests of a night are never rowdy
        self.door = None
        self.queue = []
        # the queue: every guest but one in the slots, the last one arriving
        n_slots = c.n_guests - 1
        for k, g in enumerate(self.guests):
            self._recycle(g)
            if k < n_slots:
                g.state, g.slot = "queue", k
                g.pos = np.array([self.slot_x(k), 0.0])
                g.yaw = math.pi
                self.queue.append(k)
            else:
                g.state = "hidden"
        self.phase = "open"
        self._t_phase = t
        if not first:
            self.say(f"night {self.night}: the door opens (22:00)")
            link = getattr(self.session, "brain", None)
            if link is not None and c.habituation and c.reset_depression_each_night:
                self._set_habituation(link, reset=True)
        self._write_guests()

    def slot_x(self, k: int) -> float:
        return self.cfg.queue_x0 + self.cfg.queue_gap * k

    def _recycle(self, g: Guest) -> None:
        c = self.cfg
        g.tint = int(self.rng.integers(0, len(TINTS)))
        rowdy = (self._since_rowdy >= c.rowdy_min_gap and self.rng.random() < c.rowdy_p)
        g.rowdy = bool(rowdy)
        if rowdy:
            self._since_rowdy = 0
        else:
            self._since_rowdy += 1
        g.wings = 0.0
        g.path = []
        self._tint(g)

    def _tint(self, g: Guest) -> None:
        m = self.sim.model
        k = g.k
        t = np.array(TINTS[g.tint] if not g.rowdy else (1.35, 0.55, 0.5))
        rgba = self.g_mat0[k].copy()
        rgba[:, :3] = np.clip(rgba[:, :3] * t, 0.0, 1.0)
        m.mat_rgba[self.g_mats[k]] = rgba

    # ------------------------------------------------------------ guest motion
    def _move(self, g: Guest, pts, T: float, profile: str = "ease", face_last: float | None = None) -> None:
        """Walk ``g`` through the points ``pts`` (list of xy) over T s (split by
        length); it faces its direction of travel (then ``face_last``)."""
        t = self.sim.time
        p = [g.pos.copy()] + [np.asarray(q, float) for q in pts]
        L = [float(np.linalg.norm(p[i + 1] - p[i])) for i in range(len(p) - 1)]
        tot = max(sum(L), 1e-6)
        g.path = []
        t0 = t
        yaw = g.yaw
        for i in range(len(p) - 1):
            Ti = T * L[i] / tot
            d = p[i + 1] - p[i]
            y1 = math.atan2(d[1], d[0]) if L[i] > 1e-6 else yaw
            if profile in ("lunge", "loom"):
                y1 = yaw  # stepping up facing forward
            g.path.append((p[i], p[i + 1], yaw, y1, t0, max(Ti, 1e-3), profile))
            t0 += Ti
            yaw = y1
        if face_last is not None:
            g.path.append((p[-1], p[-1], yaw, face_last, t0, 0.25, "ease"))

    def _advance(self, g: Guest, t: float, dt: float) -> bool:
        """Advance ``g`` along its path; True while it moves."""
        if not g.path:
            return False
        p0, p1, y0, y1, t0, T, prof = g.path[0]
        u = (t - t0) / T
        if u >= 1.0:
            g.pos = p1.copy()
            g.yaw = y1
            g.path.pop(0)
            return bool(g.path)
        if prof == "lunge":
            s = lunge(u)
        elif prof == "loom":
            e = self.cfg.eye_x + 0.7  # the guest's head is ~0.7 mm ahead of its thorax
            s = loom_progress(u, float(p0[0]) - e, float(p1[0]) - e)
        else:
            s = A.ease_in_out(u)
        g.pos = p0 + (p1 - p0) * s
        dy = (y1 - y0 + math.pi) % (2 * math.pi) - math.pi
        g.yaw = y0 + dy * min(1.0, u / 0.35) if abs(dy) > 1e-6 else y0
        v = float(np.linalg.norm(p1 - p0)) / T
        g.walk_phase += dt * 2 * math.pi * (6.0 if prof == "ease" else 9.0) * min(1.0, v / 2.0 + 0.3)
        return True

    def _write_guests(self) -> None:
        d = self.sim.data
        for g in self.guests:
            k = g.k
            mid = self.g_mocap[k]
            if g.state == "hidden":
                d.mocap_pos[mid] = (0.0, 0.0, -30.0 - 3 * k)
                continue
            moving = bool(g.path)
            bob = 0.03 * abs(math.sin(g.walk_phase)) if moving else 0.0
            d.mocap_pos[mid] = (g.pos[0], g.pos[1], self.g_z + bob)
            d.mocap_quat[mid] = quat_axis_angle((0, 0, 1), g.yaw)
            q = self.g_q0.copy()
            if moving:
                for i, leg in enumerate(LEGS):
                    ph = g.walk_phase + (0.0 if leg in TRIPOD_A else math.pi)
                    q[7 * i + 1] += 0.22 * math.sin(ph)  # coxa pitch: swing
                    q[7 * i + 5] += 0.25 * max(0.0, math.cos(ph))  # tibia: lift in swing
            d.qpos[self.g_qadr[k]] = q
            w = g.wings
            # wings: folded (0) .. spread up and out (1): yaw / pitch / roll per wing
            wq = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            if w > 0:
                wq[:] = [0.9 * w, -0.5 * w, 0.6 * w, -0.9 * w, -0.5 * w, -0.6 * w]
            d.qpos[self.g_wadr[k]] = wq
            d.qvel[self.g_dofs[k]] = 0.0

    # ------------------------------------------------------------ job logic
    def update(self) -> None:
        c = self.cfg
        sim = self.sim
        t = sim.time
        dt = c.update_every_steps * sim.timestep
        if self.rest_eyes is None and t >= 0.5 and self.vision is not None:
            self.rest_eyes = self.vision.live_eyes()
        # the bouncer's posture (flinch envelope, wave)
        st = self._stance
        if st is not None and self.acts.action is st:
            fu = self._flinch_until - t
            if fu > 0:
                a = c.flinch_s - fu
                st.flinch = min(1.0, a / 0.06) if a < c.flinch_s - 0.12 else max(0.0, fu / 0.12)
            else:
                st.flinch = 0.0
            if self.phase == "wave":
                age = t - self._t_phase
                st.wave = min(1.0, age / 0.2) if age < c.wave_s - 0.25 else max(0.0, (c.wave_s - age) / 0.25)
                st.wave_phase = 2 * math.pi * 3.0 * age
            else:
                st.wave = 0.0
        elif self.acts.action is None and not self.fly_down():
            self._begin()
        # guests inside leave over time
        self.inside = max(0.0, self.inside - self.inside * dt / max(c.leave_tau_s, 1e-6))
        # scripted habituation state (no brain)
        self.script_h *= math.exp(-dt / max(c.script_tau_s, 1e-6))
        self._door_logic(t)
        moving = False
        for g in self.guests:
            if g.state == "hidden":
                continue
            moving |= self._advance(g, t, dt)
            self._guest_arrived(g)
        self._write_guests()
        self._gate_tick(t)
        v = self.vision
        if v is not None and len(v.sent) > 2000:  # constant memory
            del v.sent[:-500]
        if self.flash and self.run_time() > self._flash_until:
            self.flash = ""

    def _guest_arrived(self, g: Guest) -> None:
        if g.path:
            return
        if g.state == "arriving":
            g.state = "queue"
        elif g.state in ("entering", "leaving"):
            g.state = "hidden"
            self._recycle(g)
            # back in line: it walks up to the end of the queue from off-screen
            n = len(self.queue)
            g.pos = np.array([self.slot_x(n) + 5.0, 0.0])
            g.yaw = math.pi
            g.state, g.slot = "arriving", n
            self.queue.append(g.k)
            self._move(g, [(self.slot_x(n), 0.0)], self.cfg.arrive_s, face_last=math.pi)

    def _door_logic(self, t: float) -> None:
        c = self.cfg
        age = t - self._t_phase
        ph = self.phase
        if ph == "closed":
            if age >= c.closed_s:
                self._new_night()
            return
        if ph == "open":
            if t - self.night_t0 >= c.shift_s:
                self._close(t)
                return
            if self.queue and age >= 0.2:
                h = self.guests[self.queue[0]]
                if h.state == "queue" and not h.path:
                    self._step_up(t)
            return
        g = self.guests[self.door] if self.door is not None else None
        if ph == "stepping":
            if g is not None and g.rowdy:
                g.wings = min(1.0, age / 0.25)
            if g is None or not g.path:
                self._set_phase("check", t)
                if g is not None and g.rowdy:
                    self._bump(t)
        elif ph == "check":
            if age >= c.check_s:
                self._decide(t)
        elif ph == "gate_open":
            self.gate_target = 1.0
            if self.gate_open >= 1.0:
                self._set_phase("admit_walk", t)
                gx = c.door_x
                self._move(g, [(gx + 0.3, 0.8), (gx, c.gate_y), (gx, c.wall_y + 1.4)], c.admit_walk_s)
                g.state = "entering"
                self.door = None
        elif ph == "admit_walk":
            if age >= 0.55 * c.admit_walk_s:  # past the rope: it closes, the next steps up
                self.gate_target = 0.0
                self._set_phase("open", t)
        elif ph == "wave":
            if age >= c.wave_s:
                g.state = "leaving"
                g.wings = 0.0
                self._move(g, [(c.door_spot_x + 0.4, -1.0), (c.door_spot_x + 1.2, -7.5)], c.leave_walk_s)
                self.door = None
                self._set_phase("open", t)
        if self._gf_cur is not None and t - self._gf_cur["t0"] >= c.gf_window_s:
            self._close_gf(t)

    def _set_phase(self, ph: str, t: float) -> None:
        self.phase = ph
        self._t_phase = t
        self.state = ph

    def _close(self, t: float) -> None:
        self._set_phase("closed", t)
        self.say(f"04:00 closing time: night {self.night} done, {self.night_admitted} in, "
                 f"{self.night_turned} turned away, {self.night_flinches} flinches")
        self.session.log_event("bouncer_night", night=self.night, admitted=self.night_admitted,
                               turned=self.night_turned, flinches=self.night_flinches,
                               guests=self.night_guests, rowdy=self.night_rowdy)
        self._flash("CLOSING TIME", 3.0)

    def _step_up(self, t: float) -> None:
        c = self.cfg
        k = self.queue.pop(0)
        g = self.guests[k]
        self.door = k
        g.state = "stepping"
        self.guest_no += 1
        self.night_guests += 1
        g.number = self.guest_no
        self.hour_guests[self.club_hour(t)] += 1
        x1 = c.rowdy_stop_x if g.rowdy else c.door_spot_x
        T = c.rowdy_step_up_s if g.rowdy else c.step_up_s
        self._move(g, [(x1, 0.0)], T, profile="lunge" if g.rowdy else "loom")
        self._set_phase("stepping", t)
        self._gf_cur = {"t0": t, "peak": 0.0, "flinched": False, "no": g.number, "rowdy": g.rowdy}
        # the rest of the queue shuffles up one slot
        for i, kk in enumerate(self.queue):
            q = self.guests[kk]
            q.slot = i
            if q.state == "queue":
                self._move(q, [(self.slot_x(i), 0.0)], c.shuffle_s, face_last=math.pi)
            elif q.state == "arriving":
                self._move(q, [(self.slot_x(i), 0.0)], c.arrive_s, face_last=math.pi)
        if g.rowdy:
            self.n_rowdy += 1
            self.night_rowdy += 1
            self._flash("ROWDY GUEST!", 1.5)
        # scripted flinch (no brain): decided at the step-up
        if getattr(self.session, "brain", None) is None:
            p = self.script_flinch_p(g.rowdy)
            if self.rng.random() < p:
                self._pending_script_flinch = t + 0.7 * T
            if not g.rowdy:
                self.script_h += 1.0

    def script_flinch_p(self, rowdy: bool = False) -> float:
        """Scripted flinch probability (no brain; labelled): decays with the recent
        harmless guests (h), resets partly after a rowdy one."""
        c = self.cfg
        if rowdy:
            return c.script_p0
        return c.script_p_floor + (c.script_p0 - c.script_p_floor) * math.exp(-self.script_h / c.script_h_scale)

    def _bump(self, t: float) -> None:
        """The rowdy guest's chest bump: a touch stimulus to the brain (the
        dishabituation shortcut); without a brain the scripted habituation resets."""
        c = self.cfg
        self.n_bumps += 1
        self.script_h *= 1.0 - c.script_dishab
        self._flash("CHEST BUMP!", 1.2)
        link = getattr(self.session, "brain", None)
        if link is not None and c.rowdy_bump:
            from fly_simulator.brain.schema import StimulusEvent

            link.send(StimulusEvent("shove", side="front", intensity=float(c.rowdy_bump_intensity),
                                    duration_s=0.05, sim_time=float(link._run_time()),
                                    details={"label": "CHEST BUMP (rowdy guest)", "source": "bouncer"}),
                      source="bouncer")

    def _decide(self, t: float) -> None:
        c = self.cfg
        g = self.guests[self.door]
        full = round(self.inside) >= c.capacity
        if g.rowdy or full:
            self.decision = "TURNED AWAY (" + ("rowdy" if g.rowdy else "club full") + ")"
            self.n_turned += 1
            self.night_turned += 1
            self._set_phase("wave", t)
            self._flash("NOT TONIGHT." if g.rowdy else "FULL. NOT TONIGHT.", 1.5)
        else:
            self.decision = "LET IN"
            self.n_admitted += 1
            self.night_admitted += 1
            self.inside += 1.0
            self._set_phase("gate_open", t)
        self.add_work(1)
        self.session.log_event("bouncer_guest", night=self.night, guest=g.number, rowdy=g.rowdy,
                               decision=self.decision)

    def _gate_tick(self, t: float) -> None:
        c = self.cfg
        step = c.update_every_steps * self.sim.timestep / max(c.gate_s, 1e-6)
        self.gate_open = min(max(self.gate_open + np.sign(self.gate_target - self.gate_open) * step, 0.0), 1.0)
        if abs(self.gate_open - self.gate_target) < step:
            self.gate_open = self.gate_target
        a = math.radians(c.gate_open_deg) * A.ease_in_out(self.gate_open)
        self.sim.data.mocap_quat[self.gate_mocap] = quat_axis_angle((0, 0, 1), a)

    # ------------------------------------------------------------ flinch
    def request_flinch(self, gf_hz: float | None, source: str) -> bool:
        c = self.cfg
        t = self.sim.time
        if gf_hz is not None:
            self.n_gf_bursts += 1
        if t < self._flinch_ready or self.phase == "closed":
            return False
        self._flinch_until = t + c.flinch_s
        self._flinch_ready = t + c.flinch_s + c.flinch_refractory_s
        self.n_flinches += 1
        self.night_flinches += 1
        self.hour_flinches[self.club_hour(t)] += 1
        if self._gf_cur is not None:
            self._gf_cur["flinched"] = True
        self._flash("FLINCH!" + (" (giant fibre)" if gf_hz is not None else " (scripted)"), 1.0)
        self.session.log_event("bouncer_flinch", gf_hz=None if gf_hz is None else round(gf_hz, 1),
                               source=source, guest=self.guest_no)
        return True

    def _close_gf(self, t: float) -> None:
        cur = self._gf_cur
        self._gf_cur = None
        if getattr(self.session, "brain", None) is None:
            return  # no GF without a brain
        self.gf_log.append((self.night, cur["no"], cur["rowdy"], round(cur["peak"], 1), cur["flinched"]))

    # ------------------------------------------------------------ brain tie-in
    def _set_habituation(self, link, reset: bool = False) -> None:
        br = getattr(link, "brain", None)
        cfg = {**dict(self.cfg.habituation_config), **dict(getattr(link.cfg, "habituation_config", {}) or {}),
               "enabled": True}
        if br is None:
            return
        try:
            if reset:
                br.set_habituation(None)
            br.set_habituation(cfg)
        except Exception as e:  # pragma: no cover - a dead brain process
            self.say(f"habituation could not be set: {e}")

    def _install_vision(self, link) -> None:
        """The looming source (always computed; sent to the brain if there is one)."""
        from fly_simulator.vision.looming import LoomingConfig, VisualSource

        c = self.cfg
        tf = getattr(link, "_run_time", None) if link is not None else self.session.metrics.run_time_at
        vis = _BouncerVision(self, LoomingConfig(), sink=link, time_fn=tf)

        def period():
            return 1e-3 if self.phase in ("stepping", "check") else None

        vis.add_source(VisualSource("guest", shapes=self.guest_shapes, period=period,
                                    response=guest_response(**dict(c.loom_response))))
        self.vision = vis.attach()
        if link is None:
            return
        if c.habituation:
            link.cfg.habituation = True  # the HUD line
            self._set_habituation(link)
        trig = getattr(link, "triggers", None)
        if trig is not None:
            # at the door a jump is out of the question: the job reads the giant fibre
            # from the brain states and flinches instead (no jump trigger)
            trig._check_gf = lambda gf, now, run_time, tag: []

    def guest_shapes(self) -> tuple:
        """The guest at the door (stepping up / being checked) for the looming
        computation: its head (sphere) and body (capsule head -> abdomen, with the
        legs' spread in its radius) and, for a rowdy guest, its spread wings."""
        z = np.zeros((0, 3))
        k = self.door if self.door is not None else (self.queue[0] if self.queue else None)
        if k is None or self.phase == "closed" or self.guests[k].state == "hidden":
            return z, z, np.zeros(0), np.zeros(0)
        g = self.guests[k]
        cy, sy = math.cos(g.yaw), math.sin(g.yaw)
        Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        o = np.array([g.pos[0], g.pos[1], 0.0])
        head = o + Rz @ self.g_head_off
        abd = o + Rz @ self.g_abd_off
        mid = 0.5 * (head + abd)
        ax = (head - abd) / max(np.linalg.norm(head - abd), 1e-9)
        C = [head, mid]
        AX = [np.array([0.0, 0.0, 1.0]), ax]
        H = [0.0, 0.5 * float(np.linalg.norm(head - abd))]
        R = [0.32, 0.5]
        if g.wings > 0.05:
            for s in (-1, 1):
                C.append(o + Rz @ np.array([-0.3, 0.9 * s * g.wings, self.g_z + 0.6 * g.wings]))
                AX.append(np.array([0.0, 0.0, 1.0]))
                H.append(0.0)
                R.append(0.55 * g.wings)
        return np.array(C), np.array(AX), np.array(H), np.array(R)

    def after_physics(self) -> None:
        super().after_physics()
        link = getattr(self.session, "brain", None)
        t = self.sim.time
        if link is not None:
            self._brain_tick(link)
        else:
            pend = getattr(self, "_pending_script_flinch", None)
            if pend is not None and t >= pend:
                self._pending_script_flinch = None
                self.request_flinch(None, "script")

    def _brain_tick(self, link) -> None:
        st = getattr(link, "latest", None)
        if st is None:
            return
        bt = float(getattr(st, "brain_time", 0.0) or 0.0)
        if bt <= self._gf_seen_t:
            return
        self._gf_seen_t = bt
        self.gf_hz = float((getattr(st, "descending", None) or {}).get("escape", 0.0) or 0.0)
        if self._gf_cur is not None:
            self._gf_cur["peak"] = max(self._gf_cur["peak"], self.gf_hz)
        if self.gf_hz > self.cfg.gf_flinch_hz:
            self.request_flinch(self.gf_hz, "brain")
        log = getattr(link, "stim_log", None)
        if isinstance(log, list) and len(log) > 400:
            del log[:-200]

    def efficacy(self) -> dict | None:
        link = getattr(self.session, "brain", None)
        st = getattr(link, "latest", None) if link is not None else None
        h = getattr(st, "habituation", None) if st is not None else None
        return dict(h.get("by_type") or {}) if isinstance(h, dict) and h.get("enabled") else None

    # ------------------------------------------------------------ reset
    def on_reset(self) -> None:
        self.rest_eyes = None
        self._flinch_until = -1e9
        self._begin()

    def reset_props(self) -> None:
        self._write_guests()
        self._gate_tick(self.sim.time)

    # ------------------------------------------------------------ HUD / camera
    def _flash(self, text: str, secs: float) -> None:
        self.flash = text
        self._flash_until = self.run_time() + secs

    def flinch_rate_per_hour(self) -> float:
        """Flinches per sim hour over the whole run."""
        rt = self.run_time()
        return 3600.0 * self.n_flinches / rt if rt > 1.0 else 0.0

    def camera_target(self) -> np.ndarray:
        return np.array([4.6, 1.4, 2.7])

    def camera_preset(self) -> CameraPreset:
        return CameraPreset(azimuth=74.0, elevation=-9.0, distance=19.0, tau_s=0.3)

    def job_hud_lines(self) -> list[str]:
        h = self.club_hour()
        hg, hf = self.hour_guests[h], self.hour_flinches[h]
        rate = "  ".join(f"{(22 + i) % 24:02d}h {self.hour_flinches[i]}/{self.hour_guests[i]}"
                         for i in range(h + 1))
        lines = [
            f"NIGHT {self.night}  {self.club_clock()}  [{'CLOSED' if self.phase == 'closed' else 'OPEN'}]   "
            f"let in {self.night_admitted}  turned away {self.night_turned}  inside {self.inside:.0f}/"
            f"{self.cfg.capacity}   (all nights: {self.n_admitted} in, {self.n_turned} away)",
            f"FLINCHES tonight {self.night_flinches}  this hour {hf}/{hg} guests   "
            f"({self.flinch_rate_per_hour():.0f} per sim hour)   rowdy {self.night_rowdy}",
            f"flinches / guests per club hour: {rate}",
        ]
        link = getattr(self.session, "brain", None)
        if link is not None:
            eff = self.efficacy()
            last = self.gf_log[-1] if self.gf_log else None
            lines.append(f"GF (DNp01) {self.gf_hz:3.0f} Hz" + (f"   last guest #{last[1]} GF peak {last[3]:.0f} Hz"
                                                               f"{' FLINCH' if last[4] else ''}" if last else "")
                         + ("   GF-input efficacy " + " ".join(f"{k} {v:.2f}" for k, v in eff.items())
                            if eff else "") + "  (GF -> flinch, never jump)")
        else:
            lines.append(f"(no brain: scripted flinch p = {self.script_flinch_p():.2f}, habituating)")
        if self.flash:
            lines.append(">> " + self.flash + (f"   {self.decision}" if self.decision and self.phase == "wave" else ""))
        return lines

    def job_stats(self) -> dict[str, Any]:
        peaks = [e[3] for e in self.gf_log if not e[2]]
        return {
            "night": self.night, "admitted": self.n_admitted, "turned_away": self.n_turned,
            "flinches": self.n_flinches, "rowdy": self.n_rowdy, "bumps": self.n_bumps,
            "night_guests": self.night_guests, "night_flinches": self.night_flinches,
            "hour_guests": self.hour_guests.tolist(), "hour_flinches": self.hour_flinches.tolist(),
            "flinch_rate_per_sim_hour": round(self.flinch_rate_per_hour(), 1),
            "gf_bursts": self.n_gf_bursts, "inside": round(self.inside, 1), "phase": self.phase,
            "gf_peak_mean": round(float(np.mean(peaks)), 1) if peaks else None,
            # the last guests: [guest no., rowdy, GF peak Hz, flinched] (brain only)
            "gf_last": [[e[1], e[2], e[3], e[4]] for e in list(self.gf_log)[-6:]],
        }


class _BouncerVision:
    """``LoomingVision`` whose eyes can be frozen at the resting stance (a perfect
    efference copy: the bouncer's own flinches / waves do not make the close guest
    loom). Built lazily as a subclass so the module imports without the vision
    package being loaded first."""

    def __new__(cls, job, *a, **kw):
        from fly_simulator.vision.looming import LoomingVision

        class _V(LoomingVision):
            def live_eyes(self):
                E, R = LoomingVision.eyes(self)
                return E.copy(), R.copy()

            def eyes(self):
                if job.cfg.efference_copy and job.rest_eyes is not None:
                    return job.rest_eyes
                return LoomingVision.eyes(self)

        return _V(job.sim, *a, **kw)
