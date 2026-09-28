"""DJ FLY: the fly DJs forever (docs/JOBS.md, "dj").

A club booth on a stage over a dance floor. A **beat clock** (the track's BPM) drives
everything: the fly's moves, the lights, the speakers, the dance floor and the crowd.

* **Turntables (physics).** Each record is a disc on a hinge joint, spun at 33 1/3 rpm
  by a velocity actuator with a small torque limit (the slip mat), like the kebab
  spit. The record collides with the fly (contact kind "fly").
* **The scratch (real contact).** The fly stands in ``DJStance`` (all tarsi planted
  and adhering, like ``freeze``) while the job drives the front legs with joint
  targets from damped least-squares IK on a scratch ``MjData`` (the dead_hang
  ``LegIK``), interpolated in joint space. The scratching leg presses on the record
  with its tarsal adhesion on and drags it back and forth along the groove; the record
  is moved only by that contact and friction (the motor's torque limit is the slip
  mat). A **scratch** is counted from the record itself: every time its angular
  velocity reverses against the play direction (so a stroke that slips does not count).
* **The crossfader (kinematic, labelled).** The other front leg reaches the fader
  knob and slides it across; the knob follows the tarsus while the tarsus is on it (a
  geometric test, like the taste tester's touch). If the leg misses, the job fades the
  rest of the way (counted, ``auto_fades``).
* **The drop.** On the drop the other leg goes up (hands in the air), the lights
  strobe, the floor flashes and the crowd jumps.
* **Show props (kinematic / emissive, labelled).** Speaker cones pulse on the beat
  (mocap), the LED wall's visualizer and meters are emissive cells switched by
  material, the dance floor tiles change colour per beat, the crowd is a set of
  cartoon fly silhouettes (mocap, visual) bobbing with the hype, the disco ball turns
  and three point lights orbit it. No audio (jobs have none).

Counters: tracks mixed (the work counter), scratches, drops, crowd hype (0-100 %),
BPM, auto fades.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco as mj
import numpy as np

from fly_simulator.actions.base import Action, ActionCommand, smoothstep
from fly_simulator.jobs import dj_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import contact_kwargs, quat_axis_angle, spot_or_directional
from fly_simulator.jobs.registry import register_job
from fly_simulator.jobs.taste_tester import DIGITS, SEGS

P = "dj/"
TRACKS = (("Buzz Kill", 124), ("Compound Eyes", 126), ("Wing Beat (200 Hz Mix)", 128),
          ("Halteres", 122), ("Fruit Fly Anthem", 125), ("Metamorphosis", 120))
SIDES = ("rf", "lf")  # the right deck (-y) is scratched by rf, the left (+y) by lf
TILE_COLS = ((1.0, 0.15, 0.55), (0.2, 0.55, 1.0), (0.15, 0.95, 0.55), (1.0, 0.85, 0.15),
             (0.7, 0.25, 1.0), (1.0, 0.45, 0.1))


class DJStance(Action):
    """Stand still with all tarsi adhering (like ``freeze``) while the job moves the
    front legs: ``pose[leg]`` (7 joint targets, controller order) with weight
    ``w[leg]``; a driven leg's adhesion follows ``grip[leg]`` (on for the scratching
    tarsus on the record). ``bob`` (rad) is added to the mid / hind femur-tibia
    targets: the body bobs to the beat. Stationary."""

    name = "dj_stance"
    blend_in = 0.0
    blend_out = 0.3

    def __init__(self, duration: float = 3600.0) -> None:
        super().__init__(duration)
        self.pose: dict[str, np.ndarray | None] = {"lf": None, "rf": None}
        self.w = {"lf": 0.0, "rf": 0.0}
        self.grip = {"lf": False, "rf": False}
        self.bob = 0.0

    def begin(self, mgr) -> None:
        from fly_simulator.actions.base import LEGS

        b = mgr.body
        self._start = mgr.sim.data.ctrl[b.pos_ids].copy()
        self._stand = b.stand.copy()
        self.cols = {leg: np.flatnonzero(b.leg_mask([leg])) for leg in self.pose}
        self._li = {leg: LEGS.index(leg) for leg in self.pose}
        self._fti = np.array([b.idx(leg, "fti") for leg in ("lm", "rm", "lh", "rh")])

    def command(self, mgr, t: float) -> ActionCommand:
        a = smoothstep(t / 0.2)
        tg = (1 - a) * self._start + a * self._stand
        tg[self._fti] += self.bob
        adh = np.ones(6)
        for leg, pose in self.pose.items():
            w = float(min(max(self.w[leg], 0.0), 1.0))
            if pose is not None and w > 0:
                c = self.cols[leg]
                tg[c] = (1 - w) * tg[c] + w * pose
                if w > 0.05:
                    adh[self._li[leg]] = 1.0 if self.grip[leg] else 0.0
        return ActionCommand(targets=tg, adhesion=adh)


@dataclass
class DJConfig(JobConfig):
    # --- the decks (mm, g) ---------------------------------------------------------
    deck_x: float = 2.55  # record centres (the right deck at -deck_y, the left at +deck_y)
    deck_y: float = 1.25
    record_r: float = 0.9
    record_mass: float = 1e-4  # 0.1 mg
    record_top: float = 0.13
    rpm: float = 33.333
    motor_kv: float = 0.02  # velocity actuator gain (uN*mm*s/rad)
    slipmat_torque: float = 0.5  # the motor's torque limit (uN*mm): the slip mat
    scratch_r: float = 0.66  # the scratch point's distance from the centre
    scratch_len: float = 0.3  # drag length along the groove
    scratch_rad: float = 0.2  # a backward excursion of the record this large = a scratch
    press: float = 0.01  # the tarsus target this far above the record top (adhesion pulls it on)
    fader_x: float = 1.55
    fader_half: float = 0.2  # knob travel +-
    fader_top: float = 0.22
    # --- the set ---------------------------------------------------------------------
    beats_per_track: int = 32
    drop_beat: float = 24.0
    bob_rad: float = 0.05
    hype_tau_s: float = 25.0
    n_crowd: int = 28
    shadows: bool = True
    captions: bool = True
    close_ups: bool = True


@register_job
class DJJob(EternalJob):
    name = "dj"
    znear = 0.05
    title = "DJ FLY"
    tagline = "the fly DJs forever"
    work_label = "tracks mixed"
    config_cls = DJConfig
    required_names = (P + "record0", P + "record1", P + "knob", P + "ball")

    def __init__(self, cfg: DJConfig | None = None) -> None:
        super().__init__(cfg)
        self.rng = np.random.default_rng(self.cfg.seed + 23)
        self.n_tracks = 0
        self.n_scratches = 0
        self.n_drops = 0
        self.n_auto_fades = 0
        self.n_fader_moves = 0
        self.hype = 20.0
        self.track = 0
        self.deck = 0  # the playing deck: 0 = right (-y, rf scratches), 1 = left
        self.fader = -1.0  # knob position -1 (right deck) .. +1 (left deck)
        self.beat = 0.0  # beat within the track
        self.t_track0 = 0.0
        self.bpm = float(TRACKS[0][1])
        self.phase = "setup"
        self._rev = [False, False]
        self._a_ext = [0.0, 0.0]
        self._poses: dict | None = None
        self._stance: DJStance | None = None
        self._caption = ("", -1e9)
        self._div = 0
        self.message = ""
        self._dropped = False
        self._faded = False
        self.max_reverse = 0.0

    # ------------------------------------------------------------ geometry
    def deck_centre(self, k: int) -> np.ndarray:
        c = self.cfg
        return np.array([c.deck_x, -c.deck_y if k == 0 else c.deck_y])

    def knob_y(self, fader: float) -> float:
        return -fader * self.cfg.fader_half  # -1 (right deck) -> knob toward -y

    # ------------------------------------------------------------ build
    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        self._materials(spec)
        # --- the stage (the fly stands on FlyGym's plane, hidden in on_attach)
        A.add_mesh(spec, P + "stage_mesh", A.slab_mesh(9.45, 20.0, 0.1))
        wb.add_geom(name=P + "stage", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "stage_mesh",
                    pos=(-1.275, 0.0, 0.0), material=P + "stage", **vis)
        # --- the decks -------------------------------------------------------------
        rk = contact_kwargs("fly", friction=1.0)
        w_play = 2 * math.pi * c.rpm / 60.0
        for k in range(2):
            cx, cy = self.deck_centre(k)
            sg = -1.0 if k == 0 else 1.0
            wb.add_geom(name=P + f"plinth{k}", type=mj.mjtGeom.mjGEOM_BOX, size=(1.08, 1.02, 0.03),
                        pos=(cx, cy + 0.02 * sg, 0.03), material=P + "plinth", **vis)
            # the platter (collides with the fly only: no leg slips under the record)
            wb.add_geom(name=P + f"platter{k}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                        size=(c.record_r + 0.04, (c.record_top - 0.03) / 2, 0),
                        pos=(cx, cy, (c.record_top - 0.03) / 2), material=P + "chrome",
                        **dict(contact_kwargs("fly", friction=0.3), mass=0.0))
            rz = c.record_top - 0.015
            b = wb.add_body(name=P + f"record{k}", pos=(cx, cy, rz))
            b.add_joint(name=P + f"spin{k}", type=mj.mjtJoint.mjJNT_HINGE, axis=(0, 0, 1), damping=1e-7)
            b.add_geom(name=P + f"record{k}_col", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(c.record_r, 0.015, 0),
                       mass=c.record_mass, rgba=(0, 0, 0, 0), group=3, **rk)
            A.add_mesh(spec, P + f"record{k}_mesh", A.disc_mesh(c.record_r, 0.03))
            b.add_geom(name=P + f"record{k}_vis", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + f"record{k}_mesh",
                       material=P + f"vinyl{k}", **vis)
            act = spec.add_actuator(name=P + f"motor{k}", target=P + f"spin{k}", trntype=mj.mjtTrn.mjTRN_JOINT)
            act.set_to_velocity(kv=c.motor_kv)
            act.forcelimited = True
            act.forcerange = (-c.slipmat_torque, c.slipmat_torque)
            act.ctrlrange = (-50.0, 50.0)
            self._w_play = -w_play  # clockwise seen from above
            # tonearm (visual), a pitch slider strip
            piv = (cx + 0.75, cy + 0.8 * sg)
            wb.add_geom(name=P + f"arm_post{k}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.09, 0.08, 0),
                        pos=(piv[0], piv[1], 0.14), material=P + "chrome", **vis)
            A.add_mesh(spec, P + f"arm{k}_mesh", A.tonearm_mesh(piv, (cx + 0.45, cy + 0.2 * sg), 0.24))
            wb.add_geom(name=P + f"arm{k}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + f"arm{k}_mesh",
                        material=P + "chrome", **vis)
            wb.add_geom(name=P + f"pitch{k}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.3, 0.04, 0.012),
                        pos=(cx + 0.55, cy + 0.93 * sg, 0.07), material=P + "black", **vis)
        # --- the mixer -----------------------------------------------------------------
        wb.add_geom(name=P + "mixer", type=mj.mjtGeom.mjGEOM_BOX, size=(0.7, 0.3, 0.08),
                    pos=(2.2, 0.0, 0.08), material=P + "mixer", **vis)
        for i in range(3):
            for s in (-1, 1):
                wb.add_geom(name=P + f"eq{i}{int(s > 0)}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.05, 0.03, 0),
                            pos=(2.75 - 0.22 * i, 0.13 * s, 0.19), material=P + "chrome", **vis)
        wb.add_geom(name=P + "fader_slot", type=mj.mjtGeom.mjGEOM_BOX, size=(0.03, c.fader_half + 0.04, 0.005),
                    pos=(c.fader_x, 0.0, 0.165), material=P + "black", **vis)
        wb.add_geom(name=P + "mixer_front", type=mj.mjtGeom.mjGEOM_BOX, size=(0.12, 0.3, 0.06),
                    pos=(c.fader_x - 0.0, 0.0, 0.06), material=P + "mixer", **vis)
        kb = wb.add_body(name=P + "knob", mocap=True, pos=(c.fader_x, self.knob_y(self.fader), c.fader_top - 0.03))
        kb.add_geom(name=P + "knob_g", type=mj.mjtGeom.mjGEOM_BOX, size=(0.06, 0.035, 0.03),
                    material=P + "knob", **vis)
        self._vu_names = []
        for s_i, s in enumerate((-1, 1)):
            col = []
            for j in range(6):
                nm = P + f"vu{s_i}_{j}"
                wb.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_BOX, size=(0.025, 0.03, 0.004),
                            pos=(1.95 + 0.07 * j, 0.22 * s, 0.163), material=P + "led_off", **vis)
                col.append(nm)
            self._vu_names.append(col)
        # --- booth front (faces the dance floor) ------------------------------------
        A.add_mesh(spec, P + "front_mesh", A.panel_mesh(18.0, 4.4, 0.1))
        wb.add_geom(name=P + "front", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "front_mesh",
                    pos=(3.5, 0.0, -1.9), quat=A.panel_quat((1, 0, 0)), material=P + "front", **vis)
        self._add_room(spec, vis)
        self._add_crowd(spec, vis)
        self._add_lights(spec)

    def _materials(self, spec) -> None:
        c = self.cfg
        T = A.add_textured_material
        A.add_texture(spec, P + "tex_stage", A.stage_texture(c.seed))
        T(spec, P + "stage", P + "tex_stage", rgba=(1, 1, 1, 1), specular=0.1)
        for k, col in enumerate(((0.95, 0.3, 0.2), (0.25, 0.7, 1.0))):
            A.add_texture(spec, P + f"tex_vinyl{k}", A.record_texture(col))
            T(spec, P + f"vinyl{k}", P + f"tex_vinyl{k}", rgba=(1, 1, 1, 1), specular=0.7, shininess=0.8)
        A.add_texture(spec, P + "tex_wall", A.led_wall_texture())
        T(spec, P + "wall", P + "tex_wall", rgba=(1, 1, 1, 1), emission=0.9, specular=0.0)
        A.add_texture(spec, P + "tex_front", A.booth_front_texture())
        T(spec, P + "front", P + "tex_front", rgba=(1, 1, 1, 1), emission=0.7)
        A.add_texture(spec, P + "tex_ball", A.mirror_ball_texture())
        T(spec, P + "ball", P + "tex_ball", rgba=(1, 1, 1, 1), specular=1.0, shininess=1.0, reflectance=0.2,
          emission=0.25)
        A.add_texture(spec, P + "tex_grille", A.grille_texture())
        T(spec, P + "grille", P + "tex_grille", rgba=(1, 1, 1, 1), specular=0.2)
        spec.add_material(name=P + "plinth", rgba=(0.12, 0.12, 0.14, 1), specular=0.5, shininess=0.6)
        spec.add_material(name=P + "chrome", rgba=(0.75, 0.77, 0.8, 1), specular=1.0, shininess=0.9)
        spec.add_material(name=P + "mixer", rgba=(0.18, 0.18, 0.22, 1), specular=0.5)
        spec.add_material(name=P + "black", rgba=(0.03, 0.03, 0.04, 1), specular=0.2)
        spec.add_material(name=P + "knob", rgba=(0.95, 0.95, 0.9, 1), specular=0.5, emission=0.2)
        spec.add_material(name=P + "cab", rgba=(0.08, 0.08, 0.09, 1), specular=0.2)
        spec.add_material(name=P + "cone", rgba=(0.15, 0.15, 0.17, 1), specular=0.6, shininess=0.6)
        spec.add_material(name=P + "ring", rgba=(0.2, 0.9, 1.0, 1), emission=0.4)
        spec.add_material(name=P + "led_off", rgba=(0.06, 0.08, 0.06, 1), emission=0.0)
        spec.add_material(name=P + "led_green", rgba=(0.2, 1.0, 0.3, 1), emission=0.9)
        spec.add_material(name=P + "led_red", rgba=(1.0, 0.2, 0.15, 1), emission=0.9)
        spec.add_material(name=P + "seg_on", rgba=(1.0, 0.3, 0.8, 1), emission=0.9)
        spec.add_material(name=P + "seg_off", rgba=(0.08, 0.04, 0.1, 1), emission=0.0)
        spec.add_material(name=P + "crowd", rgba=(0.07, 0.05, 0.12, 1), specular=0.3)
        spec.add_material(name=P + "crowd_eye", rgba=(0.9, 0.12, 0.1, 1), emission=0.5)
        spec.add_material(name=P + "crowd_wing", rgba=(0.45, 0.5, 0.7, 0.25), specular=0.8)
        spec.add_material(name=P + "room", rgba=(0.07, 0.05, 0.11, 1), specular=0.1)
        spec.add_material(name=P + "floor_dark", rgba=(0.05, 0.05, 0.07, 1), emission=0.0)
        for i, col in enumerate(TILE_COLS):
            spec.add_material(name=P + f"tile{i}", rgba=col + (1.0,), emission=0.9, specular=0.4)
            spec.add_material(name=P + f"bar{i}", rgba=col + (1.0,), emission=1.0)

    def _add_room(self, spec, vis) -> None:
        c = self.cfg
        wb = spec.worldbody
        # the LED wall behind the booth (faces +x) with visualizer bars, BPM and HYPE
        A.add_mesh(spec, P + "wall_mesh", A.panel_mesh(16.0, 6.0, 0.1))
        wb.add_geom(name=P + "wall", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "wall_mesh",
                    pos=(-3.8, 0.0, 4.2), quat=A.panel_quat((1, 0, 0)), material=P + "wall", **vis)
        wb.add_geom(name=P + "wall_low", type=mj.mjtGeom.mjGEOM_BOX, size=(0.05, 8.0, 0.6),
                    pos=(-3.8, 0.0, 0.6), material=P + "black", **vis)
        self._bar_names = []
        for i in range(16):
            col = []
            for j in range(8):
                nm = P + f"bar{i}_{j}"
                wb.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_BOX, size=(0.02, 0.28, 0.1),
                            pos=(-3.72, -4.5 + 0.6 * i, 1.35 + 0.24 * j), material=P + "led_off", **vis)
                col.append(nm)
            self._bar_names.append(col)
        # BPM (3 digits) on the wall's right, HYPE (10 cells) on the left
        self._seg_names = []
        dw, dh, st, gap = 0.5, 0.42, 0.07, 0.7
        for k in range(3):
            y = 5.2 + (2 - k) * gap * -1 + 1.4
            segs = []
            for sname, (dx, dz, horiz) in SEGS.items():
                nm = f"{P}bpm{k}_{sname}"
                size = (0.01, dw / 2 - 0.03, st) if horiz else (0.01, st, dh / 2 - 0.03)
                # seen from +x: +y is to the viewer's left, so digits run toward -y
                wb.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_BOX, size=size,
                            pos=(-3.72, y - dx * dw, 2.0 + dz * dh), material=P + "seg_off", **vis)
                segs.append(nm)
            self._seg_names.append(segs)
        self._hype_names = []
        for j in range(10):
            nm = P + f"hype{j}"
            wb.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_BOX, size=(0.02, 0.14, 0.3),
                        pos=(-3.72, 7.2 - 0.34 * j, 1.9), material=P + "led_off", **vis)
            self._hype_names.append(nm)
        # speakers on the stage, left and right, facing the floor
        for k, sy in enumerate((-4.6, 4.6)):
            wb.add_geom(name=P + f"cab{k}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.9, 1.0, 1.9),
                        pos=(1.2, sy, 1.9), material=P + "cab", **vis)
            wb.add_geom(name=P + f"grille{k}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.02, 0.92, 1.8),
                        pos=(2.11, sy, 1.9), material=P + "grille", **vis)
            for j, (z, r) in enumerate(((1.1, 0.75), (2.75, 0.4))):
                sb = wb.add_body(name=P + f"cone{k}_{j}", mocap=True, pos=(2.1, sy, z))
                A.add_mesh(spec, P + f"cone{k}_{j}_mesh", A.cone_mesh(r, 0.18))
                sb.add_geom(name=P + f"cone{k}_{j}_g", type=mj.mjtGeom.mjGEOM_MESH,
                            meshname=P + f"cone{k}_{j}_mesh", quat=quat_axis_angle((0, 1, 0), -math.pi / 2),
                            material=P + "cone", **vis)
                A.add_mesh(spec, P + f"ring{k}_{j}_mesh", A.torus_mesh(r * 1.05, 0.04, 32, 8))
                sb.add_geom(name=P + f"ring{k}_{j}_g", type=mj.mjtGeom.mjGEOM_MESH,
                            meshname=P + f"ring{k}_{j}_mesh", quat=quat_axis_angle((0, 1, 0), math.pi / 2),
                            pos=(0.02, 0, 0), material=P + "ring", **vis)
        # the dance floor below the stage: light tiles
        self._tile_names = []
        tz = -4.0
        for i in range(12):
            for j in range(12):
                nm = P + f"tile{i}_{j}"
                wb.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_BOX, size=(0.72, 0.72, 0.05),
                            pos=(4.3 + 1.5 * i, -8.25 + 1.5 * j, tz - 0.05), material=P + "floor_dark", **vis)
                self._tile_names.append(nm)
        wb.add_geom(name=P + "floor", type=mj.mjtGeom.mjGEOM_BOX, size=(30.0, 30.0, 0.05),
                    pos=(12.0, 0.0, tz - 0.16), material=P + "black", **vis)
        # the disco ball
        ball = wb.add_body(name=P + "ball", mocap=True, pos=(8.0, 0.0, 8.5))
        A.add_mesh(spec, P + "ball_mesh", A.sphere_mesh(1.0, (0, 0, 0), 32, 18))
        ball.add_geom(name=P + "ball_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "ball_mesh",
                      material=P + "ball", **vis)
        # the room: dark walls and ceiling
        for nm, size, pos in (("wall_l", (16.0, 0.1, 8.0), (6.0, 10.5, 4.0)),
                              ("wall_r", (16.0, 0.1, 8.0), (6.0, -10.5, 4.0)),
                              ("ceiling", (16.0, 10.6, 0.1), (6.0, 0.0, 12.5)),
                              ("wall_far", (0.1, 10.6, 8.0), (26.0, 0.0, 4.0)),
                              ("wall_back", (0.1, 10.6, 8.0), (-4.2, 0.0, 4.0))):
            wb.add_geom(name=P + nm, type=mj.mjtGeom.mjGEOM_BOX, size=size, pos=pos, material=P + "room", **vis)
        sky = spec.texture("skybox")
        if sky is not None:
            sky.rgb1 = (0.02, 0.01, 0.04)
            sky.rgb2 = (0.0, 0.0, 0.0)
        wb.add_geom(name=P + "ball_chain", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.03, 2.0, 0),
                    pos=(8.0, 0.0, 11.5), material=P + "chrome", **vis)

    def _add_crowd(self, spec, vis) -> None:
        """Cartoon fly silhouettes on the dance floor (mocap, visual): an upright
        body, a head with two red eyes toward the booth, wings, arms up."""
        c = self.cfg
        wb = spec.worldbody
        rng = np.random.default_rng(c.seed + 77)
        self.crowd_home = []
        pts = []
        while len(pts) < c.n_crowd:
            p = np.array([rng.uniform(5.0, 18.0), rng.uniform(-8.0, 8.0)])
            if abs(p[1]) < 1.6 and p[0] < 10.0:
                continue  # an aisle in front of the booth (the close-up camera)
            if all(np.linalg.norm(p - q) > 1.5 for q in pts):
                pts.append(p)
        for i, p in enumerate(pts):
            s = rng.uniform(0.85, 1.15)
            b = wb.add_body(name=P + f"fan{i}", mocap=True, pos=(p[0], p[1], -4.0))
            b.add_geom(name=P + f"fan{i}_body", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(0.4 * s, 0.42 * s, 0.8 * s),
                       pos=(0, 0, 0.8 * s), material=P + "crowd", **vis)
            b.add_geom(name=P + f"fan{i}_head", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.38 * s, 0, 0),
                       pos=(-0.05, 0, 1.85 * s), material=P + "crowd", **vis)
            for sy in (-1, 1):
                b.add_geom(name=P + f"fan{i}_eye{int(sy > 0)}", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.16 * s, 0, 0),
                           pos=(-0.3 * s, 0.2 * sy * s, 1.92 * s), material=P + "crowd_eye", **vis)
                b.add_geom(name=P + f"fan{i}_wing{int(sy > 0)}", type=mj.mjtGeom.mjGEOM_ELLIPSOID,
                           size=(0.05, 0.35 * s, 0.75 * s), pos=(0.4 * s, 0.35 * sy * s, 1.0 * s),
                           quat=quat_axis_angle((1, 0, 0), 0.5 * sy), material=P + "crowd_wing", **vis)
                b.add_geom(name=P + f"fan{i}_arm{int(sy > 0)}", type=mj.mjtGeom.mjGEOM_CAPSULE,
                           size=(0.06 * s, 0.5 * s, 0), pos=(-0.1, 0.5 * sy * s, 1.7 * s),
                           quat=quat_axis_angle((1, 0, 0), -0.45 * sy), material=P + "crowd", **vis)
            self.crowd_home.append((p[0], p[1], s, rng.uniform(0, 2 * math.pi)))

    def _add_lights(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        spec.visual.headlight.ambient = (0.16, 0.14, 0.2)
        spec.visual.headlight.diffuse = (0.22, 0.2, 0.26)
        spec.visual.headlight.specular = (0.1, 0.1, 0.1)
        key = np.array([7.0, -3.0, 9.0])
        tgt = np.array([1.6, 0.0, 0.3])
        wb.add_light(name=P + "key", type=spot_or_directional(c.shadows), pos=tuple(key), dir=tuple(tgt - key),
                     diffuse=(0.55, 0.5, 0.6), specular=(0.4, 0.4, 0.4), cutoff=30.0, exponent=0.5,
                     castshadow=bool(c.shadows))
        for k in range(3):
            wb.add_light(name=P + f"disco{k}", type=mj.mjtLightType.mjLIGHT_POINT, pos=(8.0, 0.0, 6.0),
                         diffuse=(0.5, 0.2, 0.6), specular=(0.3, 0.3, 0.3), attenuation=(0.3, 0.05, 0.006),
                         castshadow=False)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        m, s = self.sim.model, self.session
        c = self.cfg
        gp = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "ground_plane")
        if gp >= 0:  # the fly's floor stays (physics); the stage slab draws it
            m.geom_group[gp] = 3
            m.geom_rgba[gp, 3] = 0.0
        self.acts = s.actions
        self.body = s.actions.body
        self.q_rec = [int(m.jnt_qposadr[m.joint(P + f"spin{k}").id]) for k in range(2)]
        self.v_rec = [int(m.jnt_dofadr[m.joint(P + f"spin{k}").id]) for k in range(2)]
        self.a_rec = [m.actuator(P + f"motor{k}").id for k in range(2)]
        self.rec_geom = [m.geom(P + f"record{k}_col").id for k in range(2)]
        self.knob_mocap = int(m.body_mocapid[m.body(P + "knob").id])
        self.ball_mocap = int(m.body_mocapid[m.body(P + "ball").id])
        self.fan_mocap = [int(m.body_mocapid[m.body(P + f"fan{i}").id]) for i in range(c.n_crowd)]
        self.cone_mocap = [[int(m.body_mocapid[m.body(P + f"cone{k}_{j}").id]) for j in range(2)] for k in range(2)]
        self.cone_home = [[m.body(P + f"cone{k}_{j}").pos.copy() for j in range(2)] for k in range(2)]
        self.tile_gid = np.array([m.geom(n).id for n in self._tile_names])
        self.bar_gid = [[m.geom(n).id for n in col] for col in self._bar_names]
        self.vu_gid = [[m.geom(n).id for n in col] for col in self._vu_names]
        self.hype_gid = [m.geom(n).id for n in self._hype_names]
        self.seg_gid = [[m.geom(n).id for n in d] for d in self._seg_names]
        self.mat = {n: m.material(P + n).id for n in ("led_off", "led_green", "led_red", "seg_on", "seg_off",
                                                      "floor_dark")}
        self.tile_mat = [m.material(P + f"tile{i}").id for i in range(len(TILE_COLS))]
        self.bar_mat = [m.material(P + f"bar{i}").id for i in range(len(TILE_COLS))]
        self.ring_mat = m.material(P + "ring").id
        self.lights = [m.light(P + f"disco{k}").id for k in range(3)]
        self.key_light = m.light(P + "key").id
        self.key_diffuse0 = m.light_diffuse[self.key_light].copy()
        self.tarsus5 = {leg: m.body(f"{self.fly_name}/{leg}_tarsus5").id for leg in SIDES}
        from fly_simulator.jobs.dead_hang import LegIK

        self.ik = LegIK(self.sim, self.body)
        st = getattr(s, "STATIONARY_ACTIONS", None)
        if st is not None and DJStance.name not in st:
            s.STATIONARY_ACTIONS = (*st, DJStance.name)
        for k in range(2):
            self.sim.data.ctrl[self.a_rec[k]] = self._w_play
        self._begin()

    def _begin(self) -> None:
        self._stance = DJStance()
        self.acts.trigger(self._stance, source="job")
        self.steering.set(None, 0.0)
        self.phase = self.state = "setup"
        self.t_setup = self.sim.time
        self._place_knob()

    # ------------------------------------------------------------ IK keyframes
    def _solve_poses(self) -> None:
        """Joint-space keyframes for both front legs from the settled pose."""
        c = self.cfg
        b = self.body
        th = self.sim.thorax_position()
        top = c.record_top + c.press
        poses = {}
        for side, leg in enumerate(SIDES):
            cols = np.flatnonzero(b.leg_mask([leg]))
            ref = b.stand[cols]

            def ik(p, _leg=leg, _ref=ref):
                q, err = self.ik.solve(self.sim.data.qpos, _leg, [("tarsus5", np.asarray(p, float))],
                                       ref=_ref, ref_gain=0.1)
                return q, err

            cen = self.deck_centre(side)
            sg = -1.0 if side == 0 else 1.0
            # the scratch point: toward the fly from the record centre
            u = np.array([th[0], th[1]]) - cen
            u /= np.linalg.norm(u)
            pa = cen + c.scratch_r * u
            tang = np.array([-u[1], u[0]])
            if tang[0] > 0:
                tang = -tang
            pb = pa + c.scratch_len * tang  # the drag pulls toward the fly
            kv = {}
            errs = []
            rest_p = self.sim.data.xpos[self.tarsus5[leg]]
            for nm, p in (("lift", (rest_p[0] + 0.1, rest_p[1], 0.5)), ("hoverA", (*pa, top + 0.25)), ("pressA", (*pa, top)), ("pressB", (*pb, top)),
                          ("raise", (th[0] + 0.9, th[1] + 0.55 * -sg, th[2] + 1.1))):
                q, e = ik(p)
                kv[nm] = q
                errs.append(e)
            for f in (-1.0, 1.0):
                ky = self.knob_y(f)
                q, e = ik((c.fader_x, ky, c.fader_top + 0.02))
                kv[f"knob{int(f)}"] = q
                errs.append(e)
                q, e = ik((c.fader_x, ky, c.fader_top + 0.25))
                kv[f"knobhover{int(f)}"] = q
            kv["rest"] = ref.copy()
            kv["_pa"], kv["_pb"] = pa, pb
            poses[leg] = kv
            self.ik_err = max(getattr(self, "ik_err", 0.0), max(errs))
        self._poses = poses

    # ------------------------------------------------------------ schedule
    def _leg_plan(self, role: str) -> list[tuple[float, float, str, bool]]:
        """(beat start, beat end, keyframe, grip) for the scratching ("S") or the
        fader ("F") leg within one track."""
        if role == "S":
            plan = [(7.0, 7.6, "lift", False), (7.6, 8.4, "hoverA", False), (8.4, 9.0, "pressA", True)]
            for k in range(14):
                plan.append((9.0 + 0.5 * k, 9.5 + 0.5 * k, "pressB" if k % 2 == 0 else "pressA", True))
            plan += [(16.0, 16.4, "hoverA", False), (16.4, 16.9, "lift", False), (16.9, 17.5, "rest", False)]
            return plan
        d = self.cfg.drop_beat
        f0 = -1 if self.deck == 0 else 1  # the knob starts at the playing deck's side
        f1 = -f0
        return [(d - 1.0, d - 0.5, "lift", False), (d - 0.5, d, "raise", False), (d, d + 3.0, "raise", False),
                (d + 3.0, d + 4.0, f"knobhover{f0}", False), (d + 4.0, d + 4.5, f"knob{f0}", False),
                (d + 4.5, d + 7.0, f"knob{f1}", False), (d + 7.0, d + 7.4, f"knobhover{f1}", False),
                (d + 7.4, d + 7.7, "lift", False), (d + 7.7, d + 8.0, "rest", False)]

    def _leg_pose(self, leg: str, role: str, b: float) -> tuple[np.ndarray, bool]:
        kv = self._poses[leg]
        prev = "rest"
        for b0, b1, nm, grip in self._leg_plan(role):
            if b < b0:
                return kv[prev], False
            if b < b1:
                a = smoothstep((b - b0) / (b1 - b0))
                return (1 - a) * kv[prev] + a * kv[nm], grip
            prev = nm
        return kv[prev], False

    # ------------------------------------------------------------ job logic
    def update(self) -> None:
        c = self.cfg
        sim = self.sim
        t = sim.time
        d = sim.data
        if self.phase == "setup":
            if t - self.t_setup >= 0.4:
                if self._poses is None:
                    self._solve_poses()
                self._start_track(t)
            else:
                return
        spb = 60.0 / self.bpm
        self.beat = (t - self.t_track0) / spb
        b = self.beat
        st = self._stance
        if st is not None and self.acts.action is st:
            s_leg, f_leg = SIDES[self.deck], SIDES[1 - self.deck]
            for leg, role in ((s_leg, "S"), (f_leg, "F")):
                q, grip = self._leg_pose(leg, role, b)
                st.pose[leg] = q
                st.w[leg] = 1.0
                st.grip[leg] = grip
            groove = b < 7.5 or (c.drop_beat <= b < c.drop_beat + 3)
            st.bob = (c.bob_rad * math.sin(math.pi * b) ** 2) if groove else 0.0
        elif self.acts.action is None and not self.fly_down():
            self._stance = DJStance()
            self.acts.trigger(self._stance, source="job")
        # --- scratches, counted from the record's own angle (hysteresis): the record
        # turned back against the play direction by >= scratch_rad from its furthest
        # point = one scratch; it re-arms once it has gone forward again by as much
        c_sr = c.scratch_rad
        for k in range(2):
            a = float(d.qpos[self.q_rec[k]]) * np.sign(self._w_play)  # + = forward
            if not self._rev[k]:
                self._a_ext[k] = max(self._a_ext[k], a)
                if a < self._a_ext[k] - c_sr:
                    self._rev[k] = True
                    self._a_ext[k] = a
                    self.n_scratches += 1
                    self.hype = min(100.0, self.hype + 2.0)
                    if self.n_scratches % 10 == 0:
                        self._caption_say(f"SCRATCH x{self.n_scratches}!")
            else:
                self._a_ext[k] = min(self._a_ext[k], a)
                if a > self._a_ext[k] + c_sr:
                    self._rev[k] = False
                    self._a_ext[k] = a
            w = float(d.qvel[self.v_rec[k]])
            self.max_reverse = max(self.max_reverse, -w * float(np.sign(self._w_play)))
        # --- the drop
        if not self._dropped and b >= c.drop_beat:
            self._dropped = True
            self.n_drops += 1
            self.hype = min(100.0, self.hype + 30.0)
            self._caption_say("THE DROP!")
        # --- the crossfader: the knob follows the fader leg's tarsus while on it
        if c.drop_beat + 4.0 <= b < c.drop_beat + 7.2:
            p = d.xpos[self.tarsus5[SIDES[1 - self.deck]]]
            ky = self.knob_y(self.fader)
            if abs(p[0] - c.fader_x) < 0.16 and abs(p[1] - ky) < 0.16 and p[2] < c.fader_top + 0.12:
                f = float(np.clip(-p[1] / c.fader_half, -1.0, 1.0))
                if abs(f - self.fader) > 1e-3:
                    self.fader = f
                    self._moved = True
                    self._place_knob()
        # --- end of the track: mixed into the other deck
        if b >= c.beats_per_track:
            target = 1.0 if self.deck == 0 else -1.0
            if abs(self.fader - target) > 0.3:
                self.n_auto_fades += 1
            elif getattr(self, "_moved", False):
                self.n_fader_moves += 1
            self.fader = target
            self._place_knob()
            self.n_tracks += 1
            self.add_work(1)
            self.deck = 1 - self.deck
            self.track += 1
            self._start_track(t)
        dt = c.update_every_steps * sim.timestep
        self.hype = max(0.0, self.hype * math.exp(-dt / c.hype_tau_s))
        self._div += 1
        if self._div >= 10:  # the show (lights, floor, crowd): every 10 ms
            self._div = 0
            self._show(t, b)
        if self.message and self.run_time() - self._caption[1] > 2.0:
            self.message = ""

    def _start_track(self, t: float) -> None:
        name, bpm = TRACKS[self.track % len(TRACKS)]
        self.bpm = float(bpm)
        self.t_track0 = t
        self.beat = 0.0
        self._dropped = False
        self._moved = False
        self.phase = self.state = "playing"
        self._caption_say(f"NOW PLAYING: {name}  ({bpm} BPM)")
        self._update_bpm()

    def _place_knob(self) -> None:
        c = self.cfg
        self.sim.data.mocap_pos[self.knob_mocap] = (c.fader_x, self.knob_y(self.fader), c.fader_top - 0.03)

    def section(self, b: float | None = None) -> str:
        c = self.cfg
        b = self.beat if b is None else b
        if b < 8:
            return "groove"
        if b < 16:
            return "scratch"
        if b < c.drop_beat:
            return "build-up"
        if b < c.drop_beat + 4:
            return "DROP"
        return "mix"

    # ------------------------------------------------------------ the show
    def _show(self, t: float, b: float) -> None:
        c = self.cfg
        m, d = self.sim.model, self.sim.data
        sec = self.section(b)
        drop = sec == "DROP"
        frac = b % 1.0
        pulse = math.exp(-frac / 0.18)  # a kick on every beat
        beat_i = int(b)
        # dance floor tiles
        rng = np.random.default_rng(beat_i * 7 + self.track * 131 + (int(b * 4) if drop else 0))
        n = len(self.tile_gid)
        if drop:
            mats = np.array(self.tile_mat)[rng.integers(0, len(self.tile_mat), n)]
            mats[rng.random(n) < 0.25] = self.mat["floor_dark"]
        else:
            ij = np.arange(n)
            pattern = ((ij // 12 + ij % 12 + beat_i) % 4 == 0) if sec != "build-up" else \
                (rng.random(n) < 0.15 + 0.6 * (b - 16) / max(c.drop_beat - 16, 1))
            mats = np.where(pattern, np.array(self.tile_mat)[(ij // 12 + beat_i) % len(self.tile_mat)],
                            self.mat["floor_dark"])
        m.geom_matid[self.tile_gid] = mats
        # visualizer bars, VU meters, hype meter
        level = 0.35 + 0.5 * pulse + (0.3 if drop else 0.0)
        for i, col in enumerate(self.bar_gid):
            hgt = level * (0.55 + 0.45 * math.sin(1.7 * i + 3.1 * b)) * len(col)
            for j, g in enumerate(col):
                m.geom_matid[g] = self.bar_mat[(i + beat_i) % len(self.bar_mat)] if j < hgt else self.mat["led_off"]
        for s_i, col in enumerate(self.vu_gid):
            lv = level * (1.0 if (s_i == 0) == (self.fader < 0) else 0.45) * len(col)
            for j, g in enumerate(col):
                m.geom_matid[g] = (self.mat["led_red"] if j >= 4 else self.mat["led_green"]) if j < lv \
                    else self.mat["led_off"]
        for j, g in enumerate(self.hype_gid):
            m.geom_matid[g] = self.bar_mat[j % len(self.bar_mat)] if j < self.hype / 10.0 else self.mat["led_off"]
        # speakers pulse (cones in / out, the rings flash)
        amp = 0.06 + 0.06 * (self.hype / 100.0) + (0.06 if drop else 0.0)
        for k in range(2):
            for j in range(2):
                home = self.cone_home[k][j]
                d.mocap_pos[self.cone_mocap[k][j]] = home + np.array([amp * pulse * (1.0 if j == 0 else 0.5), 0, 0])
        m.mat_emission[self.ring_mat] = 0.2 + 0.8 * pulse
        # disco ball and the orbiting lights
        rate = 0.6 if not drop else 2.0
        d.mocap_quat[self.ball_mocap] = quat_axis_angle((0, 0, 1), rate * t)
        for k, li in enumerate(self.lights):
            a = rate * 1.3 * t + k * 2 * math.pi / 3
            m.light_pos[li] = (8.0 + 3.0 * math.cos(a), 3.0 * math.sin(a), 6.0)
            col = np.array(TILE_COLS[(k + beat_i) % len(TILE_COLS)])
            inten = 0.5 + 0.7 * pulse if not drop else (1.4 if (int(b * 4) + k) % 2 == 0 else 0.1)
            m.light_diffuse[li] = col * inten
        m.light_diffuse[self.key_light] = self.key_diffuse0 * (1.0 + (0.8 * pulse if drop else 0.15 * pulse))
        # the crowd bobs with the hype, jumps on the drop
        hy = self.hype / 100.0
        for i, (x, y, s, ph) in enumerate(self.crowd_home):
            bob = (0.08 + 0.25 * hy) * abs(math.sin(math.pi * b + 0.3 * ph))
            if drop:
                bob = 0.9 * abs(math.sin(math.pi * b + 0.5 * ph))
            sway = (0.15 + 0.2 * hy) * math.sin(0.5 * math.pi * b + ph)
            d.mocap_pos[self.fan_mocap[i]] = (x, y + 0.2 * sway, -4.0 + bob)
            d.mocap_quat[self.fan_mocap[i]] = quat_axis_angle((1, 0, 0), 0.25 * sway)

    def _update_bpm(self) -> None:
        if not hasattr(self, "seg_gid"):
            return
        m = self.sim.model
        v = int(round(self.bpm)) % 1000
        for k in range(3):
            dg = (v // 10 ** (2 - k)) % 10
            lit = DIGITS[dg] if (v >= 10 ** (2 - k) or k == 2) else ""
            for sname, g in zip(SEGS, self.seg_gid[k]):
                m.geom_matid[g] = self.mat["seg_on" if sname in lit else "seg_off"]

    def _caption_say(self, text: str) -> None:
        self._caption = (text, self.run_time())
        self.message = text

    def on_reset(self) -> None:
        # an explicit reset (the fly fell): the record motors keep spinning, the set
        # restarts on the same track
        for k in range(2):
            self.sim.data.ctrl[self.a_rec[k]] = self._w_play
        self._rev = [False, False]
        self._a_ext = [0.0, 0.0]
        self._begin()

    def reset_props(self) -> None:
        self._place_knob()
        self._update_bpm()

    # ------------------------------------------------------------ presentation
    def post_process(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Screen captions (NOW PLAYING, THE DROP!, SCRATCH x10!)."""
        cap, t_cap = self._caption
        if not (self.cfg.captions and cap and 0 <= t - t_cap < 1.8):
            return frame
        import cv2

        H, W = frame.shape[:2]
        out = np.ascontiguousarray(frame).copy()
        fs = max(0.4, W / 1400.0) * (2.0 if cap == "THE DROP!" else 1.3)
        th = max(1, int(round(W / 700))) * 2 + 1
        (tw, tht), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_DUPLEX, fs, th)
        x, y = (W - tw) // 2, int(0.12 * H) + tht
        cv2.putText(out, cap, (x + 2, y + 2), cv2.FONT_HERSHEY_DUPLEX, fs, (30, 0, 40), th + 2, cv2.LINE_AA)
        cv2.putText(out, cap, (x, y), cv2.FONT_HERSHEY_DUPLEX, fs, (255, 120, 230), th, cv2.LINE_AA)
        return out

    def _shot(self) -> str:
        if not self.cfg.close_ups:
            return "wide"
        sec = self.section()
        return "drop" if sec == "DROP" else ("wide" if sec == "build-up" and self.beat >= 20 else "close")

    def camera_target(self) -> np.ndarray:
        return {"wide": np.array([4.5, 0.0, 0.0]), "drop": np.array([2.2, 0.0, 0.6]),
                "close": np.array([1.7, 0.0, 0.45])}[self._shot()]

    def camera_preset(self) -> CameraPreset:
        shot = self._shot()
        if shot == "wide":
            return CameraPreset(azimuth=200.0, elevation=-16.0, distance=21.0, tau_s=0.25)
        if shot == "drop":
            return CameraPreset(azimuth=192.0, elevation=-12.0, distance=10.0, tau_s=0.2)
        return CameraPreset(azimuth=205.0, elevation=-30.0, distance=7.0, tau_s=0.25)

    def job_hud_lines(self) -> list[str]:
        name = TRACKS[self.track % len(TRACKS)][0]
        return [
            f"track {self.track + 1}: {name}  {self.bpm:.0f} BPM  beat {self.beat:4.1f}/{self.cfg.beats_per_track}"
            f"  [{self.section()}]  deck {'RL'[self.deck]}",
            f"scratches {self.n_scratches}   drops {self.n_drops}   crowd hype {self.hype:.0f} %   "
            f"fader moves {self.n_fader_moves}   auto fades {self.n_auto_fades}",
            "(records: hinge + motor, scratched by real leg contact; crossfader knob follows the tarsus: "
            "kinematic; crowd / lights / speakers: show props)",
        ] + ([self.message] if self.message else [])

    def job_stats(self) -> dict[str, Any]:
        return {"tracks_mixed": self.n_tracks, "scratches": self.n_scratches, "drops": self.n_drops,
                "hype": round(self.hype, 1), "bpm": self.bpm, "fader_moves": self.n_fader_moves,
                "auto_fades": self.n_auto_fades, "max_reverse_rad_s": round(self.max_reverse, 2),
                "ik_err_mm": round(getattr(self, "ik_err", 0.0), 3), "section": self.section()}
