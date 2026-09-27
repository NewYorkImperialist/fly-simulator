"""Broccoli toss: absolutely not. Forever.

A reaction-meme format recreated at fly scale in a streamer's room: someone is
handed a plate of broccoli, looks at it for a few seconds, then casually flicks the
whole plate backward over their shoulder without looking, and everything behind
them explodes.

World (mm; the viewer faces +x, toward the desk):

* the room: a dark wooden floor with a round rug, acoustic-foam back wall (x < 0,
  behind the chair) and a side wall with a (decorative) doorway, posters with generic
  art, RGB LED strips (emissive, colour cycling), a ring light, a mic on an arm, a
  gaming desk with two glowing monitors (a generic stream layout: game view, webcam
  box, red LIVE badge, scrolling chat blocks; no real names or logos), a big stream
  TV on the side wall, a shelf with clutter, a kitchenette (mini fridge, counter,
  microwave, a stack of plates) and a gaming chair. Moody purple / blue lights. The
  back of the room (behind the chair) is full of breakable props: free bodies
  (boxes, a can pyramid, a floor lamp, a speaker, a PC tower, a plant, trophies on
  the shelf). While intact they are **parked kinematically** (contacts off, gravity
  compensated, at rest on their spots: no resting contacts to solve); they go live
  (real contacts and gravity) at the plate's release;
* the **viewer fly** sits in the gaming chair facing the monitors. It is a second
  NeuroMechFly body, **posed / animated kinematically** (engineered, labelled): it
  is welded to the chair (no free joint), its 42 leg joints and 3 neck joints are
  written by the job every 1 ms (joint-space keyframes solved by IK at start-up),
  gravity-compensated, with no actuators and no contacts. It is not a simulated
  walker, so it costs no contact physics;
* the **host fly** is the normal simulated fly (real walking physics). It walks in
  from the kitchenette with the plate riding on its back (**kinematic carry,
  engineered**), stops beside the chair and presents it.

Sequence (a state machine in ``update``, every 1 ms):

1. ``deliver``: the host walks the straight line from the kitchen spot to its spot
   beside the chair (pure pursuit on the line, so it arrives facing the viewer: at
   this scale "turning on the spot" is really arc walking), then freezes;
2. ``handover``: the plate moves from the host's back to the viewer's front legs
   (kinematic, labelled) while the viewer reaches for it;
3. ``ponder`` (3-4 s): the viewer holds the plate still in front of its head, tilts
   its head at it, adjusts its legs a little (with ``--brain``: a bitter taste
   pulse, labelled stand-in, see below);
4. ``flick``: one sudden casual flick of the left front leg over the shoulder;
   the head is already back on the monitors (it never looks). At release the plate
   becomes a free body with a **launch velocity set by the job** (engineered,
   labelled: aimed so the ballistic arc peaks ``apex_mm`` above the release point
   and lands among the props behind the chair; the leg does not exert it). The
   broccoli florets ride on it as pooled free bodies;
5. ``flight``: real physics (gravity, contacts). The first contact of the plate
   with anything (or a timeout) is the landing;
6. ``boom``: **cartoon blast** (engineered, labelled): a bright emissive flash and
   fireball (spheres expanding and fading), a light flash, a smoke ring, a
   shockwave that applies one radial velocity kick (an impulse) to the breakable
   props in range, pooled debris chips and broccoli launched from the blast
   centre, and a camera shake. After the kick every prop flies with real physics;
7. ``aftermath``: the viewer keeps facing the monitors, unbothered; the host stands
   there; the stream viewer count goes up; chat scrolls fast;
8. ``rebuild``: the room rebuilds (**kinematic**, counted): every prop glides back
   to its spot and is released at rest; debris and florets go back to the pool.
   Meanwhile the host walks back to the kitchen on a clockwise loop under the desk
   edge (arriving facing the chair again), gets another plate, and 1. again.

The host is the only simulated walker: if it falls it gets the framework's explicit,
counted reset (it respawns at the kitchen with a new plate and the room rebuilds).

Brain tie-in (only with ``--brain``): during the ponder the job sends the bitter
taste stimulus (``StimulusEvent("taste", tastes=["bitter"])``: the Shiu et al.
labellar bitter GRN set, LB1a-e). **Stand-in, labelled:** the scene has one
connectome brain and it belongs to the simulated host fly; the job feeds it the
viewer's broccoli so the brain window shows the bitter / rejection pathway. Bitter
drives no behaviour DN in the model (docs/TASTE.md), so nothing moves because of it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from fly_simulator.jobs import broccoli_toss_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig, _make_action
from fly_simulator.jobs.geometry import (
    PROP_BIT,
    add_box,
    contact_kwargs,
    quat_axis_angle,
    quat_mul,
    wrap_angle,
)
from fly_simulator.jobs.registry import register_job
from fly_simulator.terrain import TERRAIN_BIT

P = "broc/"
VIEWER = "viewer"
N_FLORETS = 4
N_DEBRIS = 14
N_CHAT = 9
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
LEG_JOINTS = ("c_thorax-{l}_coxa-yaw", "c_thorax-{l}_coxa-pitch", "c_thorax-{l}_coxa-roll",
              "{l}_coxa-{l}_trochanterfemur-pitch", "{l}_coxa-{l}_trochanterfemur-roll",
              "{l}_trochanterfemur-{l}_tibia-pitch", "{l}_tibia-{l}_tarsus1-pitch")
HEAD_JOINTS = ("c_thorax-c_head-yaw", "c_thorax-c_head-pitch", "c_thorax-c_head-roll")
_PASSIVE_TARSUS = re.compile(r"_tarsus\d-.*_tarsus\d")


@dataclass
class BroccoliConfig(JobConfig):
    # ---- layout (the chair is at the origin, the viewer faces +x) -----------------
    # the host spawns / fetches plates at the kitchenette (in front of the mini
    # fridge) and walks a straight line to its spot beside the chair: the stand spot
    # lies on the line from the kitchen to the viewer's hands, so the host arrives
    # facing the viewer (turning on the spot is really arc walking at this scale)
    kitchen_xy: tuple = (2.4, -6.5)
    # the way back: a clockwise loop under the desk edge (the host turns in arcs of
    # ~2 mm radius), so it reaches the kitchen facing the chair again
    return_loop: tuple = ((3.2, -1.4), (5.2, -2.1), (5.9, -4.4), (5.9, -7.0), (5.67, -7.88), (5.03, -8.52), (4.15, -8.75), (3.28, -8.52), (2.63, -7.88), (2.4, -7.0))
    stand_back: float = 2.5  # the stand spot: this far from the plate hold point
    seat_z: float = 2.1  # seat cushion top
    sit_pitch_deg: float = 50.0  # the viewer's body pitch (nose up), leaning back
    walk_speed: float = 0.8
    # ---- the sequence (s) -----------------------------------------------------------
    present_s: float = 0.5  # host at the chair, facing the viewer -> handover starts
    handover_s: float = 1.1
    ponder_min_s: float = 3.0
    ponder_max_s: float = 4.0
    look_back_s: float = 0.35  # the head swings back to the monitors before the flick
    windup_s: float = 0.10
    throw_s: float = 0.10  # the flick (release at its end)
    flight_timeout_s: float = 1.0
    boom_s: float = 1.8
    aftermath_s: float = 2.4
    rebuild_s: float = 1.4
    fetch_s: float = 0.6  # at the kitchenette: the next plate
    # ---- the launch (engineered: aimed ballistic arc) -------------------------------
    target_xy: tuple = (-7.4, -2.6)  # where the plate is aimed (among the props)
    target_jitter: float = 0.9
    apex_mm: float = 3.2  # the arc peaks this far above the release point
    plate_spin: float = 45.0  # rad/s tumble
    plate_radius: float = 0.85
    plate_mass: float = 4e-5  # g (40 ug)
    floret_mass: float = 6e-6
    # ---- the blast (cartoon, engineered) -------------------------------------------
    blast_radius: float = 9.0  # props whose centre is within this get the kick
    blast_speed: float = 230.0  # mm/s kick at blast_r0 from the centre
    blast_r0: float = 2.0
    blast_up: float = 0.7  # upward bias of the kick direction
    blast_spin: float = 25.0  # rad/s random spin
    debris_speed: tuple = (120.0, 300.0)
    shake_mm: float = 0.35
    # ---- the stream -----------------------------------------------------------------
    viewers0: int = 1337
    chat_speed: float = 0.35  # mm/s chat scroll (x6 after a blast)
    # ---- brain (only with --brain) ----------------------------------------------------
    bitter_hz: float = 150.0
    bitter_every_s: float = 0.5
    bitter_pulse_s: float = 0.4
    shadows: bool = True
    stuck_timeout_s: float = 60.0


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _hue(h: float) -> tuple[float, float, float]:
    """Saturated RGB for hue h (0..1)."""
    h = (h % 1.0) * 6
    k = int(h)
    f = h - k
    return [(1, f, 0), (1 - f, 1, 0), (0, 1, f), (0, 1 - f, 1), (f, 0, 1), (1, 0, 1 - f)][k % 6]


def _quat_yaw(yaw: float):
    return quat_axis_angle((0, 0, 1), yaw)


def _slerp(q0, q1, s: float) -> np.ndarray:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1 - s) * th) * q0 + math.sin(s * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# breakable props: (name, kind, (x, y, z of the base), params)
# ---------------------------------------------------------------------------

BOX = 0.8  # half size of a cardboard box
BREAKABLES = (
    ("box0", "box", (-8.3, -3.4, 0.0), dict(h=BOX)),
    ("box1", "box", (-8.3, -1.7, 0.0), dict(h=BOX)),
    ("box2", "box", (-8.3, -2.55, 2 * BOX + 0.004), dict(h=BOX * 0.85)),
    ("can0", "can", (-6.2, -5.0, 0.0), {}),
    ("can1", "can", (-6.2, -4.3, 0.0), {}),
    ("can2", "can", (-6.2, -4.65, 0.93), {}),
    ("lamp", "lamp", (-10.8, 1.8, 0.0), {}),
    ("speaker", "speaker", (-6.2, 2.2, 0.0), {}),
    ("pc", "pc", (-5.2, -0.9, 0.0), {}),
    ("plant", "plant", (-11.2, -6.2, 0.0), {}),
    ("trophy0", "trophy", (-12.3, -5.2, 5.62), {}),
    ("trophy1", "trophy", (-12.3, -3.2, 5.62), {}),
    ("figure", "figure", (-12.3, -4.2, 5.62), {}),
)
SHELF = dict(x=-12.3, y0=-6.4, y1=-2.0, z=5.5, depth=0.9)


@register_job
class BroccoliTossJob(EternalJob):
    name = "broccoli_toss"
    title = "BROCCOLI TOSS FLY"
    tagline = "absolutely not"
    work_label = "plates yeeted"
    config_cls = BroccoliConfig
    required_names = (P + "plate", P + "chair_seat", P + "viewer_mount", VIEWER + "/c_head")

    cfg: BroccoliConfig

    @property
    def stand_xy(self) -> tuple[float, float]:
        """On the line from the kitchen to the viewer's hands, ``stand_back`` from them."""
        h = self.hold_point()[:2]
        k = np.asarray(self.cfg.kitchen_xy, float)
        u = (k - h) / np.linalg.norm(k - h)
        p = h + self.cfg.stand_back * u
        return (float(p[0]), float(p[1]))

    # ------------------------------------------------------------ app / build
    def configure_app(self, app_cfg) -> None:
        super().configure_app(app_cfg)
        sx, sy = self.stand_xy
        dx, dy = sx - self.cfg.kitchen_xy[0], sy - self.cfg.kitchen_xy[1]
        app_cfg.controller.target_heading_deg = math.degrees(math.atan2(dy, dx))

    def extension(self, world) -> None:
        spec = world.mjcf_root
        self._add_materials(spec)
        self._add_room(spec)
        self._add_desk(spec)
        self._add_chair(spec)
        self._add_viewer(spec)
        self._add_breakables(spec)
        self._add_plate(spec)
        self._add_blast(spec)
        self._add_lights(spec)
        spec.visual.map.znear = 0.05
        sky = spec.texture("skybox")
        if sky is not None:  # a dark room, not FlyGym's white sky
            sky.rgb1 = (0.03, 0.025, 0.05)
            sky.rgb2 = (0.01, 0.01, 0.02)

    # ---- materials
    def _add_materials(self, spec) -> None:
        c = self.cfg
        mat = spec.material("grid")
        if mat is not None:
            A.add_texture(spec, P + "tex_floor", A.wood_floor_texture(c.seed))
            mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = P + "tex_floor"
            mat.rgba = (1.0, 1.0, 1.0, 1.0)
            mat.reflectance = 0.05
        tex = {
            "stream_main": A.stream_texture("main", seed=c.seed),
            "stream_dash": A.stream_texture("dash", seed=c.seed),
            "poster_sunset": A.art_poster_texture("sunset", seed=c.seed),
            "poster_planet": A.art_poster_texture("planet", seed=c.seed),
            "poster_pad": A.art_poster_texture("pad", seed=c.seed),
            "foam": A.foam_texture(),
            "rug": A.rug_texture(),
            "cardboard": A.cardboard_texture(c.seed),
            "can_red": A.can_texture((0.85, 0.08, 0.10)),
            "can_blue": A.can_texture((0.10, 0.35, 0.90)),
            "broccoli": A.broccoli_texture(seed=c.seed),
            "plate": A.plate_texture(),
            "fridge": A.fridge_texture(),
            "door": A.door_texture(),
        }
        for k, img in tex.items():
            A.add_texture(spec, P + "tex_" + k, img)
        T = A.add_textured_material
        T(spec, P + "screen_main", P + "tex_stream_main", rgba=(1, 1, 1, 1), emission=0.9, specular=0.3)
        T(spec, P + "screen_dash", P + "tex_stream_dash", rgba=(1, 1, 1, 1), emission=0.9, specular=0.3)
        for k in ("poster_sunset", "poster_planet", "poster_pad"):
            T(spec, P + k, P + "tex_" + k, rgba=(1, 1, 1, 1), emission=0.25, specular=0.2)
        T(spec, P + "foam", P + "tex_foam", rgba=(1, 1, 1, 1), specular=0.05, texuniform=True,
          texrepeat=(0.5, 0.5))
        T(spec, P + "rug", P + "tex_rug", rgba=(1, 1, 1, 1), specular=0.05)
        T(spec, P + "cardboard", P + "tex_cardboard", rgba=(1, 1, 1, 1), specular=0.1)
        T(spec, P + "can_red", P + "tex_can_red", rgba=(1, 1, 1, 1), specular=0.9, shininess=0.8)
        T(spec, P + "can_blue", P + "tex_can_blue", rgba=(1, 1, 1, 1), specular=0.9, shininess=0.8)
        T(spec, P + "broccoli", P + "tex_broccoli", rgba=(1, 1, 1, 1), specular=0.15)
        T(spec, P + "plate", P + "tex_plate", rgba=(1, 1, 1, 1), specular=0.8, shininess=0.8)
        T(spec, P + "fridge", P + "tex_fridge", rgba=(1, 1, 1, 1), specular=0.8, shininess=0.8)
        T(spec, P + "door", P + "tex_door", rgba=(1, 1, 1, 1), emission=0.6)
        M = spec.add_material
        M(name=P + "wall", rgba=(0.10, 0.09, 0.15, 1), specular=0.05)
        M(name=P + "wall_side", rgba=(0.13, 0.11, 0.19, 1), specular=0.05)
        M(name=P + "desk", rgba=(0.06, 0.06, 0.07, 1), specular=0.6, shininess=0.6)
        M(name=P + "metal", rgba=(0.35, 0.35, 0.38, 1), specular=0.9, shininess=0.8)
        M(name=P + "black", rgba=(0.03, 0.03, 0.035, 1), specular=0.5, shininess=0.5)
        M(name=P + "chair", rgba=(0.05, 0.05, 0.06, 1), specular=0.4, shininess=0.5)
        M(name=P + "chair_accent", rgba=(0.55, 0.12, 0.85, 1), specular=0.5, shininess=0.5)
        M(name=P + "white_glow", rgba=(1.0, 0.97, 0.92, 1), emission=1.0)
        M(name=P + "wood", rgba=(0.30, 0.20, 0.13, 1), specular=0.2)
        M(name=P + "speaker", rgba=(0.08, 0.08, 0.09, 1), specular=0.3)
        M(name=P + "cone", rgba=(0.20, 0.20, 0.22, 1), specular=0.6)
        M(name=P + "gold", rgba=(0.90, 0.70, 0.20, 1), specular=1.0, shininess=0.9)
        M(name=P + "shade", rgba=(0.95, 0.85, 0.65, 1), emission=0.35)
        M(name=P + "leaf", rgba=(0.15, 0.45, 0.18, 1), specular=0.2)
        M(name=P + "pot", rgba=(0.70, 0.35, 0.20, 1), specular=0.1)
        M(name=P + "figure", rgba=(0.20, 0.80, 0.90, 1), specular=0.6)
        M(name=P + "debris", rgba=(0.55, 0.42, 0.28, 1), specular=0.1)
        # animated materials (mat_rgba written every update): LED strips (hue cycle),
        # the blast layers (alpha fade), the chat blocks, the PC glow
        for k in range(3):
            M(name=f"{P}led{k}", rgba=(1, 0, 1, 1), emission=1.0)
        M(name=P + "blast_core", rgba=(1.0, 0.92, 0.55, 0.0), emission=1.0)
        M(name=P + "blast_fire", rgba=(1.0, 0.28, 0.04, 0.0), emission=0.7)
        M(name=P + "blast_fire2", rgba=(1.0, 0.62, 0.1, 0.0), emission=0.7)
        M(name=P + "blast_smoke", rgba=(0.18, 0.16, 0.17, 0.0), emission=0.0, specular=0.0)
        M(name=P + "blast_ring", rgba=(0.55, 0.47, 0.42, 0.0), emission=0.2)
        M(name=P + "live_dot", rgba=(1.0, 0.1, 0.1, 1), emission=1.0)

    # ---- the room
    def _add_room(self, spec) -> None:
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        # walls: back wall (behind the chair, x = -13, foam) and a side wall (+y = 9)
        # with a doorway (decoration); invisible walls on the open sides keep props in
        add_box(wb, P + "back_wall", (0.5, 11.0, 7.0), (-13.5, 0.0, 7.0), material=P + "wall",
                collide="static", friction=0.5)
        add_box(wb, P + "side_wall_a", (6.4, 0.5, 7.0), (-8.6, 9.5, 7.0), material=P + "wall_side",
                collide="static", friction=0.5)
        add_box(wb, P + "side_wall_b", (6.9, 0.5, 7.0), (6.1, 9.5, 7.0), material=P + "wall_side",
                collide="static", friction=0.5)
        add_box(wb, P + "side_wall_top", (0.9, 0.5, 2.4), (-1.3, 9.5, 11.6), material=P + "wall_side",
                collide="static")
        for nm, hs, pos in (("inv_front", (13.0, 0.5, 8.0), (0.0, -10.5, 8.0)),
                            ("inv_right", (0.5, 11.0, 8.0), (13.5, 0.0, 8.0))):
            add_box(wb, P + nm, hs, pos, rgba=(0, 0, 0, 0), collide="static", group=3)
        # the doorway: a warm lit hallway beyond (visual panel), a frame
        A.add_mesh(spec, P + "door_mesh", A.panel_mesh(1.8, 9.2, 0.02))
        wb.add_geom(name=P + "door", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "door_mesh",
                    pos=(-1.3, 9.95, 4.6), quat=A.panel_quat((0, -1, 0)), material=P + "door", **vis)
        for sx in (-1, 1):
            add_box(wb, f"{P}door_frame{sx}", (0.08, 0.55, 4.7), (-1.3 + sx * 0.95, 9.45, 4.7),
                    material=P + "wood", collide="visual")
        # acoustic foam on the back wall, posters
        face_px = A.panel_quat((1, 0, 0))
        face_my = A.panel_quat((0, -1, 0))
        A.add_mesh(spec, P + "foam_mesh", A.panel_mesh(6.0, 4.0, 0.12))
        for k, (y, z) in enumerate(((3.0, 7.5), (-3.8, 9.8))):
            wb.add_geom(name=f"{P}foam{k}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "foam_mesh",
                        pos=(-12.95, y, z), quat=face_px, material=P + "foam", **vis)
        A.add_mesh(spec, P + "poster_mesh", A.panel_mesh(2.4, 3.2, 0.03))
        for nm, pos, q in (("poster_sunset", (-12.97, -1.0, 3.8), face_px),
                           ("poster_planet", (-12.97, 7.3, 3.4), face_px),
                           ("poster_pad", (-6.0, 8.97, 5.4), face_my)):
            wb.add_geom(name=P + nm, type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "poster_mesh",
                        pos=pos, quat=q, material=P + nm, **vis)
        # the rug under the chair
        wb.add_geom(name=P + "rug", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(3.6, 0.01, 0), pos=(0.3, 0.0, 0.012),
                    material=P + "rug", **vis)
        # LED strips (emissive, hue-cycled): along the back wall top and bottom, the
        # side wall, under the desk
        leds = (((0.05, 10.5, 0.05), (-12.95, 0.0, 12.0), 0),
                ((0.05, 10.5, 0.05), (-12.95, 0.0, 0.25), 1),
                ((12.5, 0.05, 0.05), (0.0, 8.95, 12.0), 2),
                ((12.5, 0.05, 0.05), (0.0, 8.95, 0.25), 0),
                ((0.05, 4.1, 0.04), (2.75, 0.0, 3.32), 1))
        for k, (hs, pos, mk) in enumerate(leds):
            add_box(wb, f"{P}led_strip{k}", hs, pos, material=f"{P}led{mk}", collide="visual")
        # shelf on the back wall (static: the shelf props rest on it)
        s = SHELF
        add_box(wb, P + "shelf", (s["depth"] / 2, (s["y1"] - s["y0"]) / 2, 0.06),
                (s["x"] - 0.1, (s["y0"] + s["y1"]) / 2, s["z"] - 0.06 + 0.12), material=P + "wood",
                collide="static", friction=0.8)
        for k, y in enumerate((s["y0"] + 0.4, s["y1"] - 0.4)):
            add_box(wb, f"{P}shelf_bracket{k}", (0.3, 0.05, 0.3), (s["x"] - 0.4, y, s["z"] - 0.3),
                    material=P + "metal", collide="visual")
        # a few non-breakable books on the shelf end (decoration)
        for k in range(4):
            add_box(wb, f"{P}book{k}", (0.3, 0.09, 0.42 - 0.03 * k),
                    (s["x"], s["y1"] - 0.35 - 0.2 * k, s["z"] + 0.12 + 0.42 - 0.03 * k),
                    rgba=((0.7, 0.15, 0.2, 1), (0.15, 0.3, 0.7, 1), (0.8, 0.7, 0.2, 1), (0.2, 0.6, 0.3, 1))[k],
                    collide="visual")
        # the kitchenette (front right): mini fridge, a counter with a microwave and a
        # stack of clean plates; the host fetches each plate here
        kx, ky = self.cfg.kitchen_xy
        face_mx = A.panel_quat((-1, 0, 0))
        fx, fy = kx + 6.0, ky - 0.2  # (clear of the fly at the spawn spot and the way back)
        add_box(wb, P + "fridge", (1.0, 1.1, 1.2), (fx, fy, 1.2), material=P + "black", collide="static")
        A.add_mesh(spec, P + "fridge_door_mesh", A.panel_mesh(2.0, 2.2, 0.04))
        wb.add_geom(name=P + "fridge_door", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "fridge_door_mesh",
                    pos=(fx - 1.02, fy, 1.2), quat=face_mx, material=P + "fridge", **vis)
        cy = fy - 2.6
        add_box(wb, P + "counter", (1.0, 1.4, 1.3), (fx, cy, 1.3), material=P + "wood", collide="static")
        add_box(wb, P + "counter_top", (1.05, 1.45, 0.06), (fx, cy, 2.66), material=P + "metal",
                collide="visual")
        add_box(wb, P + "microwave", (0.6, 0.8, 0.5), (fx + 0.2, cy - 0.5, 3.22), material=P + "black",
                collide="visual")
        add_box(wb, P + "microwave_window", (0.01, 0.45, 0.35), (fx - 0.41, cy - 0.6, 3.22),
                material=P + "shade", collide="visual")
        A.add_mesh(spec, P + "plate_stack_mesh", A.plate_mesh(self.cfg.plate_radius))
        for k in range(5):
            wb.add_geom(name=f"{P}plate_stack{k}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "plate_stack_mesh",
                        pos=(fx - 0.1, cy + 0.75, 2.72 + 0.06 * k), material=P + "plate", **vis)
        # the stream TV on the side wall (for the camera: the live stream)
        A.add_mesh(spec, P + "tv_mesh", A.panel_mesh(6.0, 3.4, 0.04))
        add_box(wb, P + "tv_frame", (3.15, 0.08, 1.85), (5.2, 8.9, 7.6), material=P + "black", collide="visual")
        wb.add_geom(name=P + "tv", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "tv_mesh", pos=(5.2, 8.8, 7.6),
                    quat=face_my, material=P + "screen_main", **vis)

    # ---- desk, monitors, ring light, mic
    def _add_desk(self, spec) -> None:
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        top = 3.3
        add_box(wb, P + "desk_top", (2.0, 4.2, 0.1), (4.7, 0.0, top - 0.1), material=P + "desk",
                collide="visual")
        for sx in (-1, 1):
            for sy in (-1, 1):
                add_box(wb, f"{P}desk_leg{sx}{sy}", (0.12, 0.12, (top - 0.2) / 2),
                        (4.7 + sx * 1.8, sy * 4.0, (top - 0.2) / 2), material=P + "metal", collide="visual")
        # keyboard, mouse, mouse pad (glowing edge), snacks
        add_box(wb, P + "keyboard", (0.45, 1.3, 0.05), (3.3, -0.3, top + 0.05), material=P + "black",
                collide="visual")
        add_box(wb, P + "keyboard_glow", (0.43, 1.28, 0.01), (3.3, -0.3, top + 0.105), material=P + "led2",
                collide="visual")
        add_box(wb, P + "mousepad", (0.7, 0.9, 0.01), (3.4, -2.6, top + 0.01), material=P + "black",
                collide="visual")
        wb.add_geom(name=P + "mouse", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(0.25, 0.16, 0.1),
                    pos=(3.4, -2.6, top + 0.1), material=P + "black", **vis)
        # monitors: on mocap bodies (the scrolling chat blocks move within them)
        self._monitors = []
        for k, (y, yaw, mat) in enumerate(((-1.6, math.radians(8), "screen_main"),
                                           (2.6, math.radians(-22), "screen_dash"))):
            W, H = 4.2, 2.4
            pos = np.array([5.4, y, top + 0.9 + H / 2])
            normal = np.array([-math.cos(yaw), -math.sin(yaw), 0.0])  # faces the viewer (-x)
            body = wb.add_body(name=f"{P}monitor{k}", mocap=True, pos=tuple(pos))
            A.add_mesh(spec, f"{P}screen{k}_mesh", A.panel_mesh(W, H, 0.03))
            q = A.panel_quat(normal)
            body.add_geom(name=f"{P}screen{k}", type=mj.mjtGeom.mjGEOM_MESH, meshname=f"{P}screen{k}_mesh",
                          quat=q, material=P + mat, **vis)
            # the bezel / back shell sits just behind the screen (hides the panel's
            # mirrored back face)
            body.add_geom(name=f"{P}bezel{k}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.08, W / 2 + 0.08, H / 2 + 0.08),
                          pos=tuple(-normal * 0.1), quat=_quat_yaw(yaw), material=P + "black", **vis)
            body.add_geom(name=f"{P}stand{k}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.08, 0.2, 0.5),
                          pos=tuple(-normal * 0.3 + np.array([0, 0, -H / 2 - 0.4])), quat=_quat_yaw(yaw),
                          material=P + "metal", **vis)
            body.add_geom(name=f"{P}foot{k}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.45, 0.6, 0.03),
                          pos=tuple(-normal * 0.3 + np.array([0, 0, -H / 2 - 0.87])), quat=_quat_yaw(yaw),
                          material=P + "metal", **vis)
            # chat blocks: thin emissive bars over the chat column (right 28 %)
            right = np.cross(-normal, (0, 0, 1))
            for i in range(N_CHAT):
                body.add_geom(name=f"{P}chat{k}_{i}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.004, 0.2, 0.045),
                              pos=tuple(normal * 0.03), quat=_quat_yaw(yaw), rgba=(1, 1, 1, 1),
                              material=P + "led1", **vis)
            body.add_geom(name=f"{P}live_dot{k}", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.05, 0, 0),
                          pos=tuple(normal * 0.035 + right * (-W / 2 + 0.72) + np.array([0, 0, H / 2 - 0.2])),
                          material=P + "live_dot", **vis)
            self._monitors.append(dict(pos=pos, normal=normal, right=right, W=W, H=H))
        # ring light on a stand (left of the desk, facing the viewer)
        rl = np.array([4.6, 5.9, 6.0])
        n = -rl.copy()
        n[2] = 0.0
        n /= np.linalg.norm(n)
        ax = np.cross((0, 0, 1), n)
        q = quat_axis_angle(ax, math.pi / 2)
        A.add_mesh(spec, P + "ring_mesh", A.torus_mesh(1.1, 0.12))
        wb.add_geom(name=P + "ring_light", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "ring_mesh", pos=tuple(rl),
                    quat=q, material=P + "white_glow", **vis)
        wb.add_geom(name=P + "ring_pole", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.05, rl[2] / 2 - 0.5, 0),
                    pos=(rl[0] + 0.1, rl[1] + 0.1, rl[2] / 2 - 0.55), material=P + "metal", **vis)
        # mic on an arm clamped to the desk edge, pointing at the viewer
        clamp = np.array([3.0, 3.0, top + 0.1])
        elbow = np.array([2.7, 2.4, top + 1.7])
        mic = np.array([1.9, 0.85, top + 1.1])
        back = mic + np.array([0.25, 0.3, 0.2])
        for k, (a, b) in enumerate(((clamp, elbow), (elbow, back))):
            wb.add_geom(name=f"{P}mic_arm{k}", type=mj.mjtGeom.mjGEOM_CAPSULE, fromto=(*a, *b), size=(0.04, 0, 0),
                        material=P + "black", **vis)
        wb.add_geom(name=P + "mic", type=mj.mjtGeom.mjGEOM_CAPSULE, fromto=(*back, *mic), size=(0.14, 0, 0),
                    material=P + "metal", **vis)

    # ---- gaming chair (visual: nothing walks into it)
    def _add_chair(self, spec) -> None:
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        c = self.cfg
        z = c.seat_z
        for k in range(5):
            a = k * 2 * math.pi / 5 + 0.3
            e = np.array([1.5 * math.cos(a), 1.5 * math.sin(a), 0.2])
            wb.add_geom(name=f"{P}chair_leg{k}", type=mj.mjtGeom.mjGEOM_CAPSULE, fromto=(0, 0, 0.3, *e),
                        size=(0.08, 0, 0), material=P + "black", **vis)
            wb.add_geom(name=f"{P}chair_wheel{k}", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.13, 0, 0),
                        pos=(e[0], e[1], 0.13), material=P + "black", **vis)
        wb.add_geom(name=P + "chair_gas", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.12, (z - 0.6) / 2, 0),
                    pos=(0, 0, 0.3 + (z - 0.6) / 2), material=P + "metal", **vis)
        # the seat (a named geom: required_names)
        wb.add_geom(name=P + "chair_seat", type=mj.mjtGeom.mjGEOM_BOX, size=(1.25, 1.2, 0.18),
                    pos=(-0.1, 0, z - 0.18), material=P + "chair", **vis)
        for sy in (-1, 1):
            wb.add_geom(name=f"{P}chair_bolster{sy}", type=mj.mjtGeom.mjGEOM_CAPSULE,
                        fromto=(-1.2, sy * 1.15, z - 0.05, 1.1, sy * 1.15, z - 0.05), size=(0.14, 0, 0),
                        material=P + "chair_accent", **vis)
        # backrest: leaning back 12 deg, winged, with a headrest pillow
        lean = math.radians(12)
        base = np.array([-1.35, 0.0, z])
        up = np.array([-math.sin(lean), 0.0, math.cos(lean)])
        H = 3.9
        qb = quat_axis_angle((0, 1, 0), -lean)
        wb.add_geom(name=P + "chair_back", type=mj.mjtGeom.mjGEOM_BOX, size=(0.2, 1.15, H / 2),
                    pos=tuple(base + up * H / 2), quat=qb, material=P + "chair", **vis)
        for sy in (-1, 1):
            wb.add_geom(name=f"{P}chair_wing{sy}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.25, 0.14, H * 0.42),
                        pos=tuple(base + up * H * 0.45 + np.array([0.12, sy * 1.2, 0])), quat=qb,
                        material=P + "chair_accent", **vis)
            # armrests
            wb.add_geom(name=f"{P}chair_arm{sy}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.8, 0.14, 0.07),
                        pos=(0.1, sy * 1.5, z + 0.85), material=P + "chair", **vis)
            wb.add_geom(name=f"{P}chair_armpost{sy}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.07, 0.07, 0.45),
                        pos=(-0.1, sy * 1.5, z + 0.4), material=P + "metal", **vis)
        wb.add_geom(name=P + "chair_pillow", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(0.18, 0.7, 0.35),
                    pos=tuple(base + up * (H - 0.4) + np.array([0.25, 0, 0])), quat=qb,
                    material=P + "chair_accent", **vis)

    # ---- the viewer fly (posed, kinematic)
    def _add_viewer(self, spec) -> None:
        """A second NeuroMechFly, welded to a mocap body on the seat: leg + neck
        joints only (the passive tarsal joints are removed), no actuators, no
        contacts, gravity compensated. The job writes its joint angles."""
        from flygym.anatomy import AnatomicalJoint, AxesSet, AxisOrder, JointPreset, Skeleton
        from flygym.compose import KinematicPosePreset, NeuroMechFly

        c = self.cfg
        fly = NeuroMechFly(name=VIEWER)
        joints = JointPreset.LEGS_ONLY.to_joint_list() + [
            AnatomicalJoint("c_thorax", "c_head", AxesSet(["pitch", "roll", "yaw"]))]
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
        for k in [k for k in root.keys]:
            root.delete(k)
        self._dress_viewer(root)
        pitch = math.radians(c.sit_pitch_deg)
        mount = spec.worldbody.add_body(name=P + "viewer_mount", mocap=True, pos=self._viewer_mount_pos())
        site = mount.add_site(name=P + "viewer_site", quat=quat_axis_angle((0, 1, 0), -pitch))
        spec.attach(root, prefix=VIEWER + "/", site=site)

    def _viewer_mount_pos(self) -> tuple[float, float, float]:
        return (-0.05, 0.0, self.cfg.seat_z + 0.55)

    def _dress_viewer(self, root) -> None:
        """Gaming headset on the viewer's head (visual only)."""
        head = root.body("c_head")
        vis = dict(contype=0, conaffinity=0, group=1, mass=0.0)
        # (materials live in the viewer's spec: its names get the "viewer/" prefix)
        root.add_material(name="headset", rgba=(0.07, 0.07, 0.08, 1), specular=0.7, shininess=0.7)
        root.add_material(name="headset_glow", rgba=(0.2, 1.0, 0.6, 1), emission=1.0)
        # (head frame: x forward, z up; the head is ~0.5 mm across)
        pts = [np.array([0.02, math.sin(a) * 0.36, 0.07 + math.cos(a) * 0.34])
               for a in np.linspace(-1.35, 1.35, 7)]
        for k in range(len(pts) - 1):
            head.add_geom(name=f"headset_band{k}", type=mj.mjtGeom.mjGEOM_CAPSULE,
                          fromto=(*pts[k], *pts[k + 1]), size=(0.035, 0, 0), material="headset", **vis)
        for sy in (-1, 1):
            head.add_geom(name=f"headset_cup{sy}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.15, 0.05, 0),
                          pos=(0.0, sy * 0.36, 0.0), quat=quat_axis_angle((1, 0, 0), math.pi / 2),
                          material="headset", **vis)
            head.add_geom(name=f"headset_glow{sy}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.09, 0.012, 0),
                          pos=(0.0, sy * 0.415, 0.0), quat=quat_axis_angle((1, 0, 0), math.pi / 2),
                          material="headset_glow", **vis)

    # ---- breakable props
    def _add_breakables(self, spec) -> None:
        wb = spec.worldbody
        kw = contact_kwargs("dynamic", 0.6)
        kw["conaffinity"] = TERRAIN_BIT | PROP_BIT  # not the fly
        kw["priority"] = 2
        kw["solref"] = (1e-3, 1.0)  # softer than FlyGym's 0.2 ms (fast hits)
        hid = dict(rgba=(1, 1, 1, 0), group=3)
        vis = dict(contact_kwargs("visual"), mass=0.0)
        A.add_mesh(spec, P + "can_mesh", A.can_mesh(0.3, 0.9))
        A.add_mesh(spec, P + "shade_mesh", A.shade_mesh(0.35, 0.75, 0.9))
        A.add_mesh(spec, P + "trophy_mesh", A.trophy_mesh(0.9))
        A.add_mesh(spec, P + "pot_mesh", A.pot_mesh(0.55, 0.9))
        A.add_mesh(spec, P + "leaf_mesh", A.leaf_mesh(1.3, 0.45))
        for k, (nm, kind, base, par) in enumerate(BREAKABLES):
            b = wb.add_body(name=P + nm, pos=tuple(base))
            b.add_freejoint(name=P + nm + "_free")
            if kind == "box":
                h = par["h"]
                b.add_geom(name=P + nm + "_col", type=mj.mjtGeom.mjGEOM_BOX, size=(h, h, h), pos=(0, 0, h),
                           mass=4e-5 * (h / BOX) ** 3, material=P + "cardboard", **kw)
            elif kind == "can":
                b.add_geom(name=P + nm + "_col", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.3, 0.45, 0),
                           pos=(0, 0, 0.45), mass=1.2e-5, **hid, **kw)
                b.add_geom(name=P + nm + "_vis", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "can_mesh",
                           material=P + ("can_red" if k % 2 == 0 else "can_blue"), **vis)
            elif kind == "lamp":
                b.add_geom(name=P + nm + "_col", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.6, 0.06, 0),
                           pos=(0, 0, 0.06), mass=5e-5, material=P + "black", **kw)
                b.add_geom(name=P + nm + "_pole", type=mj.mjtGeom.mjGEOM_CAPSULE, fromto=(0, 0, 0.1, 0, 0, 5.4),
                           size=(0.06, 0, 0), mass=1e-5, material=P + "metal", **kw)
                b.add_geom(name=P + nm + "_shade", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "shade_mesh",
                           pos=(0, 0, 5.0), material=P + "shade", **vis)
            elif kind == "speaker":
                b.add_geom(name=P + nm + "_col", type=mj.mjtGeom.mjGEOM_BOX, size=(0.6, 0.6, 1.5),
                           pos=(0, 0, 1.5), mass=8e-5, material=P + "speaker", **kw)
                for j, (zz, r) in enumerate(((0.8, 0.42), (2.2, 0.3))):
                    b.add_geom(name=f"{P}{nm}_cone{j}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(r, 0.02, 0),
                               pos=(0.61, 0, zz), quat=quat_axis_angle((0, 1, 0), math.pi / 2),
                               material=P + "cone", **vis)
            elif kind == "pc":
                b.add_geom(name=P + nm + "_col", type=mj.mjtGeom.mjGEOM_BOX, size=(0.9, 0.45, 1.0),
                           pos=(0, 0, 1.0), mass=9e-5, material=P + "black", **kw)
                b.add_geom(name=P + nm + "_glow", type=mj.mjtGeom.mjGEOM_BOX, size=(0.8, 0.01, 0.9),
                           pos=(0, -0.46, 1.0), material=P + "led0", **vis)
                for j in range(3):
                    b.add_geom(name=f"{P}{nm}_fan{j}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.22, 0.01, 0),
                               pos=(0.91, 0, 0.45 + 0.55 * j), quat=quat_axis_angle((0, 1, 0), math.pi / 2),
                               material=P + "led1", **vis)
            elif kind == "plant":
                b.add_geom(name=P + nm + "_col", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.5, 0.45, 0),
                           pos=(0, 0, 0.45), mass=6e-5, **hid, **kw)
                b.add_geom(name=P + nm + "_pot", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "pot_mesh",
                           material=P + "pot", **vis)
                for j in range(7):
                    a = j * 2 * math.pi / 7
                    q = quat_mul(_quat_yaw(a), quat_axis_angle((0, 1, 0), -0.9 + 0.1 * (j % 3)))
                    b.add_geom(name=f"{P}{nm}_leaf{j}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "leaf_mesh",
                               pos=(0, 0, 0.85), quat=q, material=P + "leaf", **vis)
            elif kind == "trophy":
                b.add_geom(name=P + nm + "_col", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.3, 0.45, 0),
                           pos=(0, 0, 0.45), mass=1.5e-5, **hid, **kw)
                b.add_geom(name=P + nm + "_vis", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "trophy_mesh",
                           material=P + "gold", **vis)
            else:  # figure: a little action figure (capsule body, sphere head)
                b.add_geom(name=P + nm + "_col", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.2, 0.3, 0),
                           pos=(0, 0, 0.5), mass=1e-5, material=P + "figure", **kw)
                b.add_geom(name=P + nm + "_head", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.18, 0, 0),
                           pos=(0, 0, 1.15), mass=2e-6, material=P + "figure", **kw)

    # ---- plate, florets, debris (pooled free bodies; parked behind the back wall)
    def _park_pos(self, k: int) -> np.ndarray:
        return np.array([-16.5 - 0.9 * (k // 12), -6.5 + 1.1 * (k % 12), 0.3])

    def _add_plate(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        kw = contact_kwargs("dynamic", 0.6)
        kw["conaffinity"] = TERRAIN_BIT | PROP_BIT
        kw["solref"] = (1e-3, 1.0)
        R = c.plate_radius
        plate = wb.add_body(name=P + "plate", pos=tuple(self._park_pos(0)))
        plate.add_freejoint(name=P + "plate_free")
        plate.add_geom(name=P + "plate_col", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(R, 0.04, 0),
                       pos=(0, 0, 0.04), mass=c.plate_mass, rgba=(1, 1, 1, 0), group=3, **kw)
        A.add_mesh(spec, P + "plate_mesh", A.plate_mesh(R))
        plate.add_geom(name=P + "plate_vis", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "plate_mesh",
                       material=P + "plate", **vis)
        fk = dict(kw)
        fk["priority"] = 2
        for i in range(N_FLORETS):
            b = wb.add_body(name=f"{P}floret{i}", pos=tuple(self._park_pos(1 + i)))
            b.add_freejoint(name=f"{P}floret{i}_free")
            b.add_geom(name=f"{P}floret{i}_col", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.2, 0, 0),
                       pos=(0, 0, 0.2), mass=c.floret_mass, rgba=(1, 1, 1, 0), group=3, **fk)
            A.add_mesh(spec, f"{P}floret{i}_mesh", A.floret_mesh(0.5, seed=c.seed + i))
            b.add_geom(name=f"{P}floret{i}_vis", type=mj.mjtGeom.mjGEOM_MESH, meshname=f"{P}floret{i}_mesh",
                       material=P + "broccoli", **vis)
        dk = dict(kw)
        dk["conaffinity"] = TERRAIN_BIT  # debris only hits the floor / walls (cheap)
        rng = np.random.default_rng(c.seed + 5)
        mats = (P + "debris", P + "cardboard", P + "black", P + "can_red", P + "gold", P + "broccoli")
        for i in range(N_DEBRIS):
            b = wb.add_body(name=f"{P}debris{i}", pos=tuple(self._park_pos(1 + N_FLORETS + i)))
            b.add_freejoint(name=f"{P}debris{i}_free")
            s = rng.uniform(0.08, 0.2, 3)
            b.add_geom(name=f"{P}debris{i}_col", type=mj.mjtGeom.mjGEOM_BOX, size=tuple(s), mass=3e-6,
                       material=mats[i % len(mats)], **dk)

    # ---- the blast (mocap, visual; parked under the floor)
    # fireball puffs: fixed pseudo-random offsets (unit ball, biased up) and sizes
    _rb = np.random.default_rng(1234)
    FIRE = [(tuple(float(v) for v in o), float(r)) for o, r in zip(
        _rb.normal(0, 1, (12, 3)) * np.array([0.8, 0.8, 0.6]) + np.array([0, 0, 0.5]),
        _rb.uniform(0.7, 1.35, 12))]
    SMOKE = [(tuple(float(v) for v in o), float(r)) for o, r in zip(
        _rb.normal(0, 1, (9, 3)) * np.array([1.0, 1.0, 0.4]), _rb.uniform(0.8, 1.2, 9))]
    N_RING = 16
    del _rb

    def _add_blast(self, spec) -> None:
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        b = wb.add_body(name=P + "blast", mocap=True, pos=(0.0, 0.0, -30.0))
        S = mj.mjtGeom.mjGEOM_SPHERE
        b.add_geom(name=P + "blast_core", type=S, size=(0.1, 0, 0), material=P + "blast_core", **vis)
        for k, (off, _) in enumerate(self.FIRE):
            b.add_geom(name=f"{P}blast_fire{k}", type=S, size=(0.1, 0, 0), pos=off,
                       material=P + ("blast_fire" if k % 3 else "blast_fire2"), **vis)
        for k in range(len(self.SMOKE)):
            b.add_geom(name=f"{P}blast_smoke{k}", type=S, size=(0.1, 0, 0), material=P + "blast_smoke", **vis)
        for k in range(self.N_RING):
            b.add_geom(name=f"{P}blast_ring{k}", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(0.1, 0.1, 0.05),
                       material=P + "blast_ring", **vis)
        b.add_light(name=P + "blast_light", type=mj.mjtLightType.mjLIGHT_POINT, pos=(0, 0, 1.0),
                    diffuse=(0, 0, 0), specular=(0, 0, 0), castshadow=False, attenuation=(1.0, 0.0, 0.004))

    def _add_lights(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        spec.visual.headlight.ambient = (0.16, 0.13, 0.22)
        spec.visual.headlight.diffuse = (0.18, 0.16, 0.24)
        spec.visual.headlight.specular = (0.05, 0.05, 0.08)

        def spot(name, pos, tgt, col, cutoff=45.0, exp=0.8, shadow=False):
            pos, tgt = np.array(pos, float), np.array(tgt, float)
            wb.add_light(name=P + name, type=mj.mjtLightType.mjLIGHT_SPOT, pos=tuple(pos), dir=tuple(tgt - pos),
                         diffuse=col, specular=(0.3, 0.3, 0.35), cutoff=cutoff, exponent=exp, castshadow=shadow)

        spot("key", (8.0, -9.0, 16.0), (-2.0, 0.0, 2.0), (0.50, 0.32, 0.72), 45.0, 0.5, bool(c.shadows))
        spot("rim", (-10.0, 6.0, 13.0), (0.0, 0.0, 3.0), (0.15, 0.45, 0.75), 40.0, 1.0)
        spot("monitor_glow", (4.8, 0.0, 5.2), (0.0, 0.0, 3.4), (0.30, 0.38, 0.55), 50.0, 1.0)
        spot("ring", (4.4, 5.6, 6.0), (0.0, 0.0, 3.4), (0.35, 0.33, 0.30), 35.0, 1.0)
        spot("backroom", (-2.0, -7.0, 14.0), (-8.5, -1.0, 1.5), (0.45, 0.20, 0.50), 45.0, 0.5)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        sim = self.sim
        m, c = sim.model, self.cfg
        j = lambda n: m.joint(n).id  # noqa: E731
        self.plate_body = m.body(P + "plate").id
        self.plate_q = int(m.jnt_qposadr[j(P + "plate_free")])
        self.plate_v = int(m.jnt_dofadr[j(P + "plate_free")])
        self.flor_body = np.array([m.body(f"{P}floret{i}").id for i in range(N_FLORETS)])
        self.flor_q = np.array([int(m.jnt_qposadr[j(f"{P}floret{i}_free")]) for i in range(N_FLORETS)])
        self.flor_v = np.array([int(m.jnt_dofadr[j(f"{P}floret{i}_free")]) for i in range(N_FLORETS)])
        self.deb_q = np.array([int(m.jnt_qposadr[j(f"{P}debris{i}_free")]) for i in range(N_DEBRIS)])
        self.deb_v = np.array([int(m.jnt_dofadr[j(f"{P}debris{i}_free")]) for i in range(N_DEBRIS)])
        self.brk_body = np.array([m.body(P + b[0]).id for b in BREAKABLES])
        self.brk_q = np.array([int(m.jnt_qposadr[j(P + b[0] + "_free")]) for b in BREAKABLES])
        self.brk_v = np.array([int(m.jnt_dofadr[j(P + b[0] + "_free")]) for b in BREAKABLES])
        self.brk_home = np.array([m.qpos0[q:q + 7] for q in self.brk_q])
        # "live" switch (contacts + gravity) per body group: props at rest, the parked
        # pool and the held plate are kinematic with no contacts (much cheaper: no
        # resting contacts); they go live when the physics matters
        deb_body = np.array([m.body(f"{P}debris{i}").id for i in range(N_DEBRIS)])
        self._groups = {"props": self.brk_body, "plate": np.array([self.plate_body]),
                        "florets": self.flor_body, "debris": deb_body}
        self._geoms = {k: np.flatnonzero(np.isin(m.geom_bodyid, b)) for k, b in self._groups.items()}
        self._ct0 = m.geom_contype.copy()
        self._ca0 = m.geom_conaffinity.copy()
        self.live = {k: True for k in self._groups}
        # the pool bodies' "ignore" mask for the plate's landing test
        self._own = np.zeros(m.nbody, bool)
        self._own[self.plate_body] = True
        self._own[self.flor_body] = True
        # blast / animated materials
        self.blast_mocap = int(m.body_mocapid[m.body(P + "blast").id])
        g = lambda n: m.geom(n).id  # noqa: E731
        self.g_core = g(P + "blast_core")
        self.g_fire = [g(f"{P}blast_fire{k}") for k in range(len(self.FIRE))]
        self.g_smoke = [g(f"{P}blast_smoke{k}") for k in range(len(self.SMOKE))]
        self.g_ring = [g(f"{P}blast_ring{k}") for k in range(self.N_RING)]
        # geoms whose pos / quat the job animates: MuJoCo skips the geom offset of
        # geoms compiled at their body's frame ("sameframe"), so turn that off
        if hasattr(m, "geom_sameframe"):
            for gid in [*self.g_fire, *self.g_smoke, *self.g_ring]:
                m.geom_sameframe[gid] = 0
        mat = lambda n: m.material(n).id  # noqa: E731
        self.m_led = [mat(f"{P}led{k}") for k in range(3)]
        self.m_core, self.m_fire, self.m_fire2 = mat(P + "blast_core"), mat(P + "blast_fire"), mat(P + "blast_fire2")
        self.m_smoke, self.m_ring, self.m_live = mat(P + "blast_smoke"), mat(P + "blast_ring"), mat(P + "live_dot")
        self.base_rgba = {k: m.mat_rgba[k].copy() for k in (self.m_core, self.m_fire, self.m_fire2,
                                                              self.m_smoke, self.m_ring)}
        self.blast_light = m.light(P + "blast_light").id
        self.g_chat = [[g(f"{P}chat{k}_{i}") for i in range(N_CHAT)] for k in range(2)]
        if hasattr(m, "geom_sameframe"):
            for row in self.g_chat:
                for gid in row:
                    m.geom_sameframe[gid] = 0
        self._chat_phase = np.zeros(2)
        rng = np.random.default_rng(c.seed + 21)
        self._chat_w = rng.uniform(0.25, 1.0, (2, N_CHAT))
        self._chat_col = [[_hue(rng.random()) for _ in range(N_CHAT)] for _ in range(2)]
        self._rng = np.random.default_rng(c.seed)
        # viewer joints
        self._setup_viewer()
        # counters
        self.n_yeeted = 0
        self.n_explosions = 0
        self.viewers = int(c.viewers0)
        self.last_gain = 0
        self.best_gain = 0
        self.n_rebuilds = 0
        self.n_props_launched = 0
        self.n_resets_mid = 0
        self.n_plates_lost = 0
        self.n_flight_timeouts = 0
        self.n_bitter = 0
        self.vegetables_eaten = 0  # absolutely not
        self.last_flight = None  # dict: launch / landing of the last plate
        self.message = ""
        self._msg_t = -1e9
        self._shake_t = -1e9
        self._cam = None
        self._bitter_next = 0.0
        self._fast_chat_until = -1e9
        # the host spawns at the kitchenette (keyframe), then an explicit first reset
        self._set_spawn_keyframe()
        self._phase_init()
        self.sim.reset()
        self.session.metrics.n_resets -= 1  # the initial spawn is not a reset
        self.n_resets -= 1

    def _set_spawn_keyframe(self) -> None:
        sim, c = self.sim, self.cfg
        m = sim.model
        key = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, "neutral")
        fq = sim._free_qpos
        dx, dy = self.stand_xy[0] - c.kitchen_xy[0], self.stand_xy[1] - c.kitchen_xy[1]
        yaw = math.atan2(dy, dx)
        m.key_qpos[key, fq:fq + 2] = c.kitchen_xy
        m.key_qpos[key, fq + 3:fq + 7] = _quat_yaw(yaw)
        m.key_qpos[key, self.v_qadr] = self.q_rest
        self._key = key

    # ------------------------------------------------------------ the viewer (IK poses)
    def _setup_viewer(self) -> None:
        m = self.sim.model
        pre = VIEWER + "/"
        self.v_names = [jn.format(l=leg) for leg in LEGS for jn in LEG_JOINTS] + list(HEAD_JOINTS)
        jid = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, pre + n) for n in self.v_names]
        if min(jid) < 0:
            raise RuntimeError("viewer joints missing: " + str([n for n, i in zip(self.v_names, jid) if i < 0]))
        self.v_qadr = np.array([m.jnt_qposadr[i] for i in jid])
        self.v_dadr = np.array([m.jnt_dofadr[i] for i in jid])
        # (the viewer's thorax has no joint: MuJoCo fuses it into the mocap mount)
        vroot = m.body(P + "viewer_mount").id
        self.v_thorax = vroot
        self.v_dofs_all = np.flatnonzero(m.body_rootid[m.dof_bodyid] == m.body_rootid[vroot])
        self.v_bid = {(leg, seg): m.body(f"{pre}{leg}_{seg}").id for leg in LEGS
                      for seg in ("tarsus1", "tarsus5", "tibia")}
        self.v_head = m.body(pre + "c_head").id
        # start from the host's standing angles (same joint names)
        body = self.session.actions.body
        stand = {n: float(v) for n, v in zip(body.names, body.stand)}
        q0 = np.array([stand.get(n, 0.0) for n in self.v_names])
        self._ikd = mj.MjData(m)
        self._jac = np.zeros((3, m.nv))
        self.q_stand = q0.copy()
        self._solve_poses()

    def _leg_cols(self, leg: str) -> np.ndarray:
        i = LEGS.index(leg)
        return np.arange(7 * i, 7 * i + 7)

    def _ik(self, q: np.ndarray, leg: str, targets, iters: int = 150, ref=None) -> tuple[np.ndarray, float]:
        """Damped least squares for one viewer leg (7 DoFs) on a scratch MjData.
        ``targets``: [(segment, world point, weight)]."""
        m, d = self.sim.model, self._ikd
        cols = self._leg_cols(leg)
        da = self.v_dadr[cols]
        q = q.copy()
        err = math.inf
        ref = self.q_stand[cols] if ref is None else ref
        for _ in range(iters):
            d.qpos[self.v_qadr] = q
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            J, E = [], []
            for seg, p, w in targets:
                bi = self.v_bid[(leg, seg)]
                mj.mj_jacBody(m, d, self._jac, None, bi)
                J.append(w * self._jac[:, da])
                E.append(w * (np.asarray(p, float) - d.xpos[bi]))
            J, E = np.vstack(J), np.concatenate(E)
            err = float(np.linalg.norm(E))
            if err < 1e-3:
                break
            Jp = J.T @ np.linalg.inv(J @ J.T + 2e-3 * np.eye(len(E)))
            dq = 0.5 * (Jp @ E)
            dq += 0.05 * (np.eye(7) - Jp @ J) @ (ref - q[cols])
            n = float(np.linalg.norm(dq))
            if n > 0.15:
                dq *= 0.15 / n
            q[cols] += dq
        return q, err

    def hold_point(self) -> np.ndarray:
        """World centre of the plate while the viewer holds it (in front of its head)."""
        return np.array([1.3, 0.0, self.cfg.seat_z + 1.3])

    def _solve_poses(self) -> None:
        """Joint-space keyframes of the viewer, solved once (IK on scratch data)."""
        d = self._ikd
        m = self.sim.model
        d.qpos[:] = m.qpos0
        z = self.cfg.seat_z
        H = self.hold_point()
        R = self.cfg.plate_radius
        q = self.q_stand.copy()
        # rest: hind legs over the seat front, mid legs on the armrests, front legs
        # in the lap
        tg = {"lh": [("tarsus5", (0.95, 0.55, z - 0.35), 1.0)], "rh": [("tarsus5", (0.95, -0.55, z - 0.35), 1.0)],
              "lm": [("tarsus5", (0.35, 1.45, z + 0.93), 1.0)], "rm": [("tarsus5", (0.35, -1.45, z + 0.93), 1.0)],
              "lf": [("tarsus5", (1.05, 0.35, z + 0.55), 1.0)], "rf": [("tarsus5", (1.05, -0.35, z + 0.55), 1.0)]}
        self.ik_err = {}
        for leg, t in tg.items():
            q, e = self._ik(q, leg, t)
            self.ik_err[f"rest_{leg}"] = e
        self.q_rest = q.copy()
        # hold: front tarsi on the plate rim, left / right
        qh = q.copy()
        for leg, sy in (("lf", 1), ("rf", -1)):
            qh, e = self._ik(qh, leg, [("tarsus5", H + np.array([0.0, sy * (R - 0.05), 0.06]), 1.0)])
            self.ik_err[f"hold_{leg}"] = e
        self.q_hold = qh
        # take: reaching toward the host's side (+y)
        qt = q.copy()
        for leg, sy in (("lf", 1), ("rf", -1)):
            qt, e = self._ik(qt, leg, [("tarsus5", self.take_point() + np.array([0.0, sy * (R - 0.05), 0.06]), 1.0)])
            self.ik_err[f"take_{leg}"] = e
        self.q_take = qt
        # windup: the left front leg dips forward with the plate, the right lets go
        qw = self.q_hold.copy()
        qw[self._leg_cols("rf")] = self.q_rest[self._leg_cols("rf")]
        qw, e = self._ik(qw, "lf", [("tarsus5", H + np.array([0.25, 0.45, -0.25]), 1.0)])
        self.ik_err["wind_lf"] = e
        self.q_wind = qw
        # flick: over the left shoulder (away from the camera), up and back
        qf = qw.copy()
        qf, e = self._ik(qf, "lf", [("tarsus5", np.array(self.FLICK_TIP) + np.array([0.0, 0.0, z]), 1.0)],
                         iters=300)
        self.ik_err["flick_lf"] = e
        self.q_flick = qf

    FLICK_TIP = (-0.45, 0.75, 2.4)  # the flicking tarsus at the release (z above the seat)

    def take_point(self) -> np.ndarray:
        return np.array([1.3, -0.95, self.cfg.seat_z + 1.1])

    def _set_viewer(self, q: np.ndarray, head=(0.0, 0.0, 0.0)) -> None:
        d = self.sim.data
        qq = q.copy()
        qq[-3:] = head
        d.qpos[self.v_qadr] = qq
        d.qvel[self.v_dofs_all] = 0.0

    def viewer_tarsus(self, leg: str = "rf") -> np.ndarray:
        return self.sim.data.xpos[self.v_bid[(leg, "tarsus5")]].copy()

    # ------------------------------------------------------------ props helpers
    def plate_pos(self) -> np.ndarray:
        return self.sim.data.qpos[self.plate_q:self.plate_q + 3].copy()

    def _place(self, qadr: int, vadr: int, pos, quat=(1.0, 0.0, 0.0, 0.0), vel=None, angvel=None) -> None:
        d = self.sim.data
        d.qpos[qadr:qadr + 3] = pos
        d.qpos[qadr + 3:qadr + 7] = quat
        d.qvel[vadr:vadr + 3] = 0.0 if vel is None else vel
        d.qvel[vadr + 3:vadr + 6] = 0.0 if angvel is None else angvel

    FLORET_SPOTS = ((0.28, 0.12), (-0.22, 0.25), (-0.12, -0.28), (0.18, -0.2))

    def _place_plate(self, pos, quat, vel=None) -> None:
        """Plate (and the florets on it) at a held pose (kinematic)."""
        self._place(self.plate_q, self.plate_v, pos, quat, vel)
        Rm = np.empty(9)
        mj.mju_quat2Mat(Rm, np.asarray(quat, float))
        Rm = Rm.reshape(3, 3)
        for i in range(N_FLORETS):
            off = np.array([*self.FLORET_SPOTS[i], 0.045])
            q = quat_mul(quat, _quat_yaw(1.3 * i))
            self._place(self.flor_q[i], self.flor_v[i], np.asarray(pos) + Rm @ off, q, vel)

    def _set_live(self, group: str, live: bool) -> None:
        """Contacts + gravity on (real physics) or off (kinematic: no contacts,
        gravity compensated, held where the job puts it)."""
        if self.live.get(group) == live:
            return
        m = self.sim.model
        g = self._geoms[group]
        m.geom_contype[g] = self._ct0[g] if live else 0
        m.geom_conaffinity[g] = self._ca0[g] if live else 0
        m.body_gravcomp[self._groups[group]] = 0.0 if live else 1.0
        self.live[group] = live

    def _park_pool(self) -> None:
        self._set_live("florets", False)
        self._set_live("debris", False)
        for i in range(N_FLORETS):
            self._place(self.flor_q[i], self.flor_v[i], self._park_pos(1 + i))
        for i in range(N_DEBRIS):
            self._place(self.deb_q[i], self.deb_v[i], self._park_pos(1 + N_FLORETS + i))

    def carry_pose(self) -> tuple[np.ndarray, tuple]:
        """The plate on the host's back (kinematic carry)."""
        d = self.sim.data
        th = self.sim.thorax_body_id
        Rm = d.xmat[th].reshape(3, 3)
        pos = d.xpos[th] + Rm @ np.array([-0.15, 0.0, 0.62])
        return pos, _quat_yaw(self.sim.heading())

    def present_pose(self) -> np.ndarray:
        """The plate held up in front of the host's head (the offer)."""
        d = self.sim.data
        th = self.sim.thorax_body_id
        Rm = d.xmat[th].reshape(3, 3)
        return d.xpos[th] + Rm @ np.array([1.1, 0.0, 0.55])

    # ------------------------------------------------------------ the sequence
    def _phase_init(self) -> None:
        self.phase = "deliver"
        self.state = "deliver"
        self._t_phase = 0.0
        self._ponder_s = self.cfg.ponder_min_s
        self._blast = None
        self._rebuild = None
        self._flight = None
        self._path_s = 0.0

    def _set_phase(self, ph: str, t: float) -> None:
        self.phase = self.state = ph
        self._t_phase = t
        self._path_s = 0.0

    def on_reset(self) -> None:
        if self.phase not in ("deliver", "fetch"):
            self.n_resets_mid += 1
        self._phase_init()
        self._t_phase = self.run_time()

    def reset_props(self) -> None:
        """After a reset the host is back at the kitchenette: a fresh plate on its back,
        the room rebuilt at once, the pool parked, the blast off."""
        d = self.sim.data
        self._set_live("props", False)
        self._set_live("plate", False)
        for k in range(len(BREAKABLES)):
            self._place(self.brk_q[k], self.brk_v[k], self.brk_home[k][:3], self.brk_home[k][3:])
        self._park_pool()
        self._blast_off()
        self._set_viewer(self.q_rest)
        mj.mj_forward(self.sim.model, d)
        pos, q = self.carry_pose()
        self._place_plate(pos, q)
        mj.mj_forward(self.sim.model, d)

    def update(self) -> None:
        t = self.run_time()
        c = self.cfg
        ph = self.phase
        s = t - self._t_phase
        self._animate_room(t)
        acts = self.session.actions
        if ph == "deliver":
            self._set_viewer(self.q_rest, self._idle_head(t))
            pos, q = self.carry_pose()
            self._place_plate(pos, q, self._host_vel())
            if acts.busy and acts.action is not None and acts.action.name != "freeze":
                self.steering.set(None, 1.0)
                return
            if self._follow([self.cfg.kitchen_xy, self.stand_xy]):
                self._set_phase("face", t)
            else:
                if s > 2.0:  # (not right after standing still at the kitchen)
                    self.unstick()
        elif ph == "face":
            self._set_viewer(self.q_rest, self._idle_head(t))
            pos, q = self.carry_pose()
            self._place_plate(pos, q)
            # stop first (0.25 s), then short turns on the spot toward the viewer
            # (TurnInPlace bursts, re-aimed each time), then freeze
            want = self._face_yaw()
            err = wrap_angle(self.sim.heading() - want)
            self.steering.set(self.sim.heading(), 0.0)
            if s < 0.25:
                if not acts.busy:
                    acts.trigger(_make_action("freeze", duration=0.25), source="job")
            elif abs(err) < math.radians(45) or s > 1.5:
                if acts.busy:
                    acts.cancel()
                acts.trigger(_make_action("freeze", duration=60.0), source="job")
                self._set_phase("present", t)
                self._say_msg("the host fly: 'made you broccoli'")
            elif not acts.busy:
                # err > 0: the fly points left of the target -> turn right
                acts.trigger(_make_action("turn_right" if err > 0 else "turn_left",
                                          duration=min(abs(err) / 5.0, 0.35), amp=0.7), source="job")
        elif ph == "present":
            self._keep_frozen()
            self._set_viewer(self.q_rest, self._idle_head(t))
            a = _ease(s / c.present_s)
            p0, q0 = self.carry_pose()
            p = p0 + (self.present_pose() - p0) * a
            self._place_plate(p, q0)
            if s >= c.present_s:
                self._handover_from = (p.copy(), np.array(q0))
                self._set_phase("handover", t)
        elif ph == "handover":
            self._keep_frozen()
            self._handover(s)
            if s >= c.handover_s:
                self._ponder_s = float(self._rng.uniform(c.ponder_min_s, c.ponder_max_s))
                self._bitter_next = t
                self._set_phase("ponder", t)
                self._say_msg("hmm...")
        elif ph == "ponder":
            self._keep_frozen()
            self._ponder(s, t)
            if s >= self._ponder_s:
                self._set_phase("flick", t)
        elif ph == "flick":
            self._keep_frozen()
            self._flick(s, t)
        elif ph == "flight":
            self._keep_frozen()
            self._set_viewer(self.q_flick if s < 0.25 else self._blend(self.q_flick, self.q_rest, (s - 0.25) / 0.5),
                             self._monitor_head(t))
            self._check_landing(t, s)
        elif ph == "boom":
            self._keep_frozen()
            self._set_viewer(self._blend(self.q_flick, self.q_rest, (t - self._flight["t_release"] - 0.25) / 0.5),
                             self._monitor_head(t))
            self._run_blast(t)
            if s >= c.boom_s:
                self._blast_off()
                self._set_phase("aftermath", t)
        elif ph == "aftermath":
            self._keep_frozen()
            self._set_viewer(self.q_rest, self._monitor_head(t))
            if s >= c.aftermath_s:
                self._start_rebuild(t)
        elif ph in ("rebuild", "walk_off"):
            self._set_viewer(self.q_rest, self._idle_head(t))
            if self._rebuild is not None:
                self._run_rebuild(t)
            if acts.busy and acts.action is not None and acts.action.name == "freeze":
                acts.cancel()
            arrived = self._follow(self.return_path(), arrive=1.0)
            if s > 2.0:  # (the track still holds the long stand at the chair)
                self.unstick()
            if arrived and self._rebuild is None:
                self.steering.set(self.sim.heading(), 0.0)
                self._set_phase("fetch", t)
        elif ph == "fetch":
            self._set_viewer(self.q_rest, self._idle_head(t))
            self.steering.set(self.sim.heading(), 0.0)
            if s >= c.fetch_s:
                pos, q = self.carry_pose()
                self._place_plate(pos, q)
                self._set_phase("deliver", t)
                self._say_msg("the kitchen: another plate of broccoli")
            else:
                # the plate appears on the host's back as it turns (kitchen hand-off)
                pos, q = self.carry_pose()
                self._place_plate(pos, q)

    def _follow(self, pts, lookahead: float = 1.5, arrive: float = 0.45,
                cross: float = 1.5) -> bool:
        """Pure pursuit along the polyline ``pts``: the carrot is ``lookahead`` mm
        (of path length) ahead of the fly's closest point on it (past the end it
        runs on along the last segment, so the host never orbits the end point: it
        turns in ~2 mm arcs). Arrival: within ``arrive`` of the end, or crossing the
        end (projection past it) within ``cross``. The host converges onto the path
        and ends heading along its last segment."""
        P_ = np.asarray(pts, float)
        seg = P_[1:] - P_[:-1]
        L = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(L)])
        p = self.fly_xy()
        best, s_best = math.inf, 0.0
        n = len(seg)
        for k in range(n):
            u = float((p - P_[k]) @ seg[k] / max(L[k] ** 2, 1e-12))
            u = min(max(u, 0.0), 1.0 if k < n - 1 else 1.5)
            dd = float(np.linalg.norm(P_[k] + u * seg[k] - p))
            # (monotone: never jump back to an earlier stretch of the loop)
            sk = cum[k] + u * L[k]
            if dd < best - 1e-9 and sk >= self._path_s - 0.5:
                best, s_best = dd, sk
        self._path_s = max(self._path_s, s_best)
        sc = self._path_s + lookahead
        k = int(np.clip(np.searchsorted(cum, sc) - 1, 0, n - 1))
        carrot = P_[k] + seg[k] * (sc - cum[k]) / max(L[k], 1e-12)
        self.steering.aim_at(carrot, self.cfg.walk_speed)
        dist = float(np.linalg.norm(P_[-1] - p))
        return dist < arrive or (self._path_s >= cum[-1] - 0.05 and dist < cross)

    def return_path(self) -> list:
        """Stand spot -> a clockwise loop under the desk edge -> the kitchen spot,
        arriving heading along the delivery line (the host turns in ~2 mm arcs)."""
        return [self.stand_xy, *self.cfg.return_loop, self.cfg.kitchen_xy]

    def _host_vel(self) -> np.ndarray:
        return self.sim.data.qvel[self.sim._free_qvel:self.sim._free_qvel + 3].copy()

    def _face_yaw(self) -> float:
        p = self.stand_xy
        h = self.hold_point()
        return math.atan2(h[1] - p[1], h[0] - p[0])

    def _keep_frozen(self) -> None:
        acts = self.session.actions
        self.steering.set(self.sim.heading(), 0.0)
        if not acts.busy:
            acts.trigger(_make_action("freeze", duration=60.0), source="job")

    # ---- viewer motion
    @staticmethod
    def _blend(qa, qb, a: float) -> np.ndarray:
        a = _ease(a)
        return qa + (qb - qa) * a

    def _monitor_head(self, t: float) -> tuple[float, float, float]:
        """Looking at the monitors: the body leans back, so the head pitches down to
        level the gaze; a tiny game-watching sway."""
        return (0.04 * math.sin(t * 1.3), self._level_pitch(), 0.03 * math.sin(t * 0.7))

    def _idle_head(self, t: float):
        return self._monitor_head(t)

    def _level_pitch(self) -> float:
        return math.radians(self.cfg.sit_pitch_deg) * 0.8

    def _handover(self, s: float) -> None:
        c = self.cfg
        T = c.handover_s
        k1 = 0.55 * T
        p0, q0 = self._handover_from
        take = self.take_point()
        H = self.hold_point()
        if s < k1:
            a = _ease(s / k1)
            p = p0 + (take - p0) * a + np.array([0, 0, 0.5 * math.sin(math.pi * a)])
            qv = self._blend(self.q_rest, self.q_take, s / k1)
        else:
            a = _ease((s - k1) / (T - k1))
            p = take + (H - take) * a
            qv = self._blend(self.q_take, self.q_hold, (s - k1) / (T - k1))
        qp = _slerp(q0, (1.0, 0.0, 0.0, 0.0), _ease(s / T))
        self._set_viewer(qv, self._monitor_head(self.run_time()))
        self._place_plate(p, qp)

    def _ponder(self, s: float, t: float) -> None:
        """Plate still; the head tilts down at it and rocks a little; a leg twitch."""
        c = self.cfg
        lp = self._level_pitch()
        look = _ease(s / 0.5) * (1.0 - _ease((s - (self._ponder_s - c.look_back_s)) / c.look_back_s))
        pitch = lp + look * math.radians(18)
        roll = look * (math.radians(14) * math.sin(0.9 * s + 0.3) + math.radians(6))
        yaw = look * math.radians(6) * math.sin(0.6 * s)
        q = self.q_hold.copy()
        # small leg adjustments: a mid-leg tap on the armrest, a hind-leg swing
        q[self._leg_cols("lm")[5]] += 0.25 * max(0.0, math.sin(2.2 * s)) ** 4
        q[self._leg_cols("rh")[1]] += 0.12 * math.sin(1.4 * s)
        self._set_viewer(q, (yaw, pitch, roll))
        self._place_plate(self.hold_point(), (1.0, 0.0, 0.0, 0.0))
        self._bitter(t)

    def _flick(self, s: float, t: float) -> None:
        c = self.cfg
        head = self._monitor_head(t)
        if s < c.windup_s:
            q = self._blend(self.q_hold, self.q_wind, s / c.windup_s)
        else:
            q = self._blend(self.q_wind, self.q_flick, (s - c.windup_s) / c.throw_s)
        self._set_viewer(q, head)
        # the plate follows the left front "hand" (kinematic until the release)
        mj.mj_kinematics(self.sim.model, self.sim.data)
        tip = self.viewer_tarsus("lf")
        a = _ease(min(s / (c.windup_s + c.throw_s), 1.0))
        tilt = quat_axis_angle((0, 1, 0), 1.6 * a)  # the plate tips over the shoulder
        if s < c.windup_s:
            base = self.hold_point() + (tip + np.array([0.0, -0.5, 0.0]) - self.hold_point()) * _ease(s / c.windup_s)
        else:
            base = tip + np.array([0.0, -0.5 * (1 - a), 0.0])
        self._place_plate(base, tilt)
        if s >= c.windup_s + c.throw_s:
            self._launch(t, base, tilt)

    def _launch(self, t: float, p0: np.ndarray, quat) -> None:
        """Release: the plate becomes a free body with the job's aimed launch velocity
        (engineered; see module doc). The florets ride on it (free bodies too)."""
        c = self.cfg
        rng = self._rng
        tgt = np.array([*c.target_xy, 0.2]) + np.array([*rng.normal(0, c.target_jitter, 2), 0.0])
        g = 9810.0
        vz = math.sqrt(2 * g * c.apex_mm)
        dz = tgt[2] - p0[2]
        tau = (vz + math.sqrt(vz * vz - 2 * g * dz)) / g  # z(tau) = target z
        vxy = (tgt[:2] - p0[:2]) / tau
        v = np.array([vxy[0], vxy[1], vz])
        spin = np.array([0.0, -c.plate_spin, 0.0]) + rng.normal(0, 5.0, 3)
        self._place(self.plate_q, self.plate_v, p0, quat, v, spin)
        for grp in ("plate", "florets", "props"):  # real physics from here on
            self._set_live(grp, True)
        for i in range(N_FLORETS):
            d = self.sim.data
            pv = v + rng.normal(0, 25.0, 3)
            d.qvel[self.flor_v[i]:self.flor_v[i] + 3] = pv
            d.qvel[self.flor_v[i] + 3:self.flor_v[i] + 6] = rng.normal(0, 20.0, 3)
        self.n_yeeted += 1
        self.add_work(1.0)
        self._flight = dict(t_release=t, p0=p0.copy(), v0=v.copy(), target=tgt, tau=tau, apex=None)
        self._set_phase("flight", t)
        self._say_msg("*flick*")

    def _check_landing(self, t: float, s: float) -> None:
        d, m = self.sim.data, self.sim.model
        p = self.plate_pos()
        f = self._flight
        f["apex"] = max(f["apex"] or -1e9, float(p[2]))
        hit = False
        n = d.ncon
        if n and s > 0.01:
            geom = d.contact.geom[:n]
            gb = m.geom_bodyid[geom]
            own = self._own[gb]
            plate = (gb == self.plate_body).any(1)
            hit = bool(np.any(plate & ~own.all(1)))
        lost = not np.all(np.isfinite(p)) or p[2] < -1.0
        if hit or lost or s > self.cfg.flight_timeout_s:
            if lost:
                self.n_plates_lost += 1
                p = f["target"].copy()
            if s > self.cfg.flight_timeout_s and not hit:
                self.n_flight_timeouts += 1
            f.update(t_land=t, p_land=p.copy(), flight_s=s)
            self.last_flight = dict(f)
            self._start_blast(t, p)

    # ---- the blast (cartoon)
    def _start_blast(self, t: float, centre: np.ndarray) -> None:
        c = self.cfg
        rng = self._rng
        d = self.sim.data
        centre = np.array([centre[0], centre[1], max(float(centre[2]), 0.3)])
        self.n_explosions += 1
        self._blast = dict(t0=t, c=centre)
        d.mocap_pos[self.blast_mocap] = centre
        # the shockwave: one radial velocity kick to the props in range (an impulse)
        n = 0
        for k in range(len(BREAKABLES)):
            b = self.brk_body[k]
            p = d.xipos[b]
            r = p - centre
            dist = float(np.linalg.norm(r))
            if dist > c.blast_radius or not np.isfinite(dist):
                continue
            u = r / max(dist, 1e-6)
            u = u + np.array([0.0, 0.0, c.blast_up])
            u /= np.linalg.norm(u)
            sp = c.blast_speed * (c.blast_r0 / max(dist, c.blast_r0)) ** 0.6 * rng.uniform(0.8, 1.2)
            va = self.brk_v[k]
            d.qvel[va:va + 3] += sp * u
            d.qvel[va + 3:va + 6] += rng.normal(0, c.blast_spin, 3)
            n += 1
        self.n_props_launched += n
        # debris chips and the florets: out of the blast centre
        self._set_live("debris", True)
        for i in range(N_DEBRIS):
            u = rng.normal(0, 1, 3)
            u[2] = abs(u[2]) + 0.6
            u /= np.linalg.norm(u)
            self._place(self.deb_q[i], self.deb_v[i], centre + 0.3 * u,
                        tuple(_slerp((1, 0, 0, 0), rng.normal(0, 1, 4) / 2, 1.0)),
                        u * rng.uniform(*c.debris_speed), rng.normal(0, 40.0, 3))
        for i in range(N_FLORETS):
            u = rng.normal(0, 1, 3)
            u[2] = abs(u[2]) + 1.0
            u /= np.linalg.norm(u)
            va = self.flor_v[i]
            d.qvel[va:va + 3] = u * rng.uniform(120.0, 260.0)
        # the plate itself is shattered: park it (a fresh one comes from the kitchen)
        self._place(self.plate_q, self.plate_v, self._park_pos(0))
        self._set_live("plate", False)
        self._shake_t = t
        self._fast_chat_until = t + c.boom_s + c.aftermath_s
        gain = int(80 + 0.12 * self.viewers * rng.uniform(0.6, 1.4))
        self.viewers += gain
        self.last_gain = gain
        self.best_gain = max(self.best_gain, gain)
        self._say_msg(f"KA-BOOM! (cartoon blast)   chat goes wild: +{gain:,} viewers")
        self._set_phase("boom", t)
        self.say(f"plate #{self.n_yeeted}: KA-BOOM at ({centre[0]:.1f}, {centre[1]:.1f}), "
                 f"{n} props launched, viewers {self.viewers:,}")

    def _run_blast(self, t: float) -> None:
        m = self.sim.model
        b = self._blast
        if b is None:
            return
        s = t - b["t0"]

        def setg(gid, r, pos=None):
            m.geom_size[gid, 0] = max(r, 1e-3)
            m.geom_rbound[gid] = max(r, 1e-3)
            if pos is not None:
                m.geom_pos[gid] = pos

        def alpha(mid, a):
            m.mat_rgba[mid, 3] = min(max(a, 0.0), 1.0)

        # core flash (0-0.2 s), fireball (0-0.8 s), smoke (0.25-1.6 s), dust ring (0-0.5 s)
        setg(self.g_core, 1.2 * _ease(s / 0.05) * (1.0 - _ease((s - 0.05) / 0.2)) + 0.02)
        alpha(self.m_core, 1.0 - _ease((s - 0.08) / 0.15))
        grow = _ease(s / 0.22)
        for k, gid in enumerate(self.g_fire):
            off, r = self.FIRE[k]
            pos = np.array(off) * (0.4 + 1.5 * grow) + np.array([0.0, 0.0, 1.5 * s])
            setg(gid, r * (0.2 + 1.2 * grow) * (1.0 - 0.3 * _ease((s - 0.4) / 0.4)), pos)
        fa = 1.0 - _ease((s - 0.35) / 0.45)
        alpha(self.m_fire, 0.95 * fa)
        alpha(self.m_fire2, 0.95 * fa)
        sg = _ease((s - 0.25) / 1.3)
        for k, gid in enumerate(self.g_smoke):
            off, r = self.SMOKE[k]
            pos = np.array(off) * (1.2 + 1.5 * sg) + np.array([0.0, 0.0, 1.0 + 3.5 * sg])
            setg(gid, r * (0.5 + 0.8 * sg), pos)
        alpha(self.m_smoke, 0.55 * _ease((s - 0.25) / 0.25) * (1.0 - _ease((s - 0.8) / 0.8)))
        rr = 7.0 * _ease(s / 0.5)
        zr = 0.15 - b["c"][2]  # the dust ring runs along the floor
        for k, gid in enumerate(self.g_ring):
            a = k * 2 * math.pi / len(self.g_ring)
            m.geom_size[gid] = (0.5 + 0.5 * _ease(s / 0.5), 0.35 + 0.3 * _ease(s / 0.5), 0.12)
            m.geom_rbound[gid] = 1.0
            m.geom_pos[gid] = (rr * math.cos(a), rr * math.sin(a), zr)
            m.geom_quat[gid] = _quat_yaw(a)
        alpha(self.m_ring, 0.35 * (1.0 - _ease((s - 0.1) / 0.4)))
        li = 6.0 * math.exp(-s / 0.18)
        m.light_diffuse[self.blast_light] = (li, 0.55 * li, 0.18 * li)

    def _blast_off(self) -> None:
        m, d = self.sim.model, self.sim.data
        d.mocap_pos[self.blast_mocap] = (0.0, 0.0, -30.0)
        for mid in (self.m_core, self.m_fire, self.m_fire2, self.m_smoke, self.m_ring):
            m.mat_rgba[mid, 3] = 0.0
        m.light_diffuse[self.blast_light] = (0.0, 0.0, 0.0)
        self._blast = None

    # ---- the room rebuilds (kinematic, counted)
    def _start_rebuild(self, t: float) -> None:
        d = self.sim.data
        start = np.array([d.qpos[q:q + 7].copy() for q in self.brk_q])
        bad = ~np.all(np.isfinite(start), 1)
        start[bad] = self.brk_home[bad]
        self._rebuild = dict(t0=t, start=start)
        self._park_pool()
        self._set_live("props", False)  # the rebuild is kinematic
        self._set_phase("rebuild", t)
        self._say_msg("rebuilding the room")
        acts = self.session.actions
        if acts.busy:
            acts.cancel()

    def _run_rebuild(self, t: float) -> None:
        r = self._rebuild
        s = (t - r["t0"]) / self.cfg.rebuild_s
        for k in range(len(BREAKABLES)):
            # lift, glide, set down (each prop a little staggered)
            a = _ease((s - 0.03 * k) / 0.6)
            p0, h = r["start"][k][:3], self.brk_home[k][:3]
            p = p0 + (h - p0) * a
            p[2] += 1.2 * math.sin(math.pi * a)
            q = _slerp(r["start"][k][3:], self.brk_home[k][3:], a)
            self._place(self.brk_q[k], self.brk_v[k], p, q)
        if s >= 1.0:
            for k in range(len(BREAKABLES)):
                self._place(self.brk_q[k], self.brk_v[k], self.brk_home[k][:3], self.brk_home[k][3:])
            self._rebuild = None
            self.n_rebuilds += 1
            if self.phase == "rebuild":
                self.phase = self.state = "walk_off"  # (keeps the host's waypoint)

    # ---- room animation: LED hue cycle, chat scroll, LIVE dot blink
    def _animate_room(self, t: float) -> None:
        m = self.sim.model
        for k, mid in enumerate(self.m_led):
            m.mat_rgba[mid, :3] = _hue(0.08 * t + k / 3)
        m.mat_rgba[self.m_live, 3] = 1.0 if (t % 1.0) < 0.6 else 0.15
        fast = t < self._fast_chat_until
        dt = 1e-3 * self.cfg.update_every_steps
        for k, mon in enumerate(self._monitors):
            self._chat_phase[k] += dt * self.cfg.chat_speed * (6.0 if fast else 1.0)
            W, H = mon["W"], mon["H"]
            x0 = W / 2 - 0.28 * W  # chat column left edge (monitor frame, along "right")
            span = H - 0.45
            gap = span / N_CHAT
            for i, gid in enumerate(self.g_chat[k]):
                z = -H / 2 + 0.1 + ((self._chat_phase[k] + i * gap) % span)
                w = 0.13 * W * self._chat_w[k, i]
                pos = mon["normal"] * 0.03 + mon["right"] * (x0 + 0.06 + w) + np.array([0.0, 0.0, z])
                m.geom_pos[gid] = pos
                m.geom_size[gid, 1] = w
                m.geom_rgba[gid, :3] = (1.0, 0.3, 0.3) if (fast and i % 2) else self._chat_col[k][i]

    # ---- brain tie-in (only with --brain)
    def _bitter(self, t: float) -> None:
        link = getattr(self.session, "brain", None)
        if link is None or t < self._bitter_next:
            return
        c = self.cfg
        self._bitter_next = t + c.bitter_every_s
        from fly_simulator.brain.schema import StimulusEvent

        link.send(StimulusEvent("taste", "none", 1.0, c.bitter_pulse_s, t,
                                details={"tastes": ["bitter"], "bitter_hz": c.bitter_hz,
                                         "label": "BROCCOLI: BITTER (stand-in: LB1 bitter GRNs)"}),
                  source="job")
        self.n_bitter += 1
        log = getattr(link, "stim_log", None)
        if isinstance(log, list) and len(log) > 400:
            del log[:-200]

    def _say_msg(self, msg: str) -> None:
        self.message = msg
        self._msg_t = self.run_time()

    # ------------------------------------------------------------ view / HUD
    SHOTS = {"close": ((120.0, -15.0, 7.5), 0.6), "walk": ((112.0, -14.0, 14.0), 0.6),
             "wide": ((100.0, -13.0, 16.5), 0.35)}

    def shot(self) -> str:
        ph = self.phase
        if ph in ("ponder", "handover", "flick"):
            return "close"
        if ph in ("deliver", "face", "present", "fetch"):
            return "walk"
        return "wide"

    def camera_target(self) -> np.ndarray:
        t = self.sim.time
        sh = self.shot()
        if sh == "close":
            out = np.array([0.7, -0.5, self.cfg.seat_z + 1.2])
        elif sh == "walk":
            f = self.sim.thorax_position()
            out = 0.5 * f + 0.5 * np.array([0.0, 0.0, 3.0]) + np.array([-1.0, 0.0, 0.8])
        else:
            out = np.array([-4.6, -1.2, 3.9])
        dt = self.run_time() - self._shake_t
        if 0 <= dt < 0.7:
            a = self.cfg.shake_mm * (1 - dt / 0.7)
            out = out + a * np.array([math.sin(t * 173), math.sin(t * 211 + 1), 0.6 * math.sin(t * 157 + 2)])
        return out

    def camera_preset(self) -> CameraPreset:
        """Three shots; close -> wide (the flick) is a hard cut, the rest glide."""
        sh = self.shot()
        goal, tau = self.SHOTS[sh]
        t = self.sim.time
        cut = self._cam is not None and self._cam[2] == "close" and sh == "wide"
        if self._cam is None or t < self._cam[0] or cut:
            self._cam = (t, np.array(goal), sh)
            self._cut_t = t
        else:
            t0, cur, _ = self._cam
            a = 1.0 - math.exp(-(t - t0) / tau)
            self._cam = (t, cur + a * (np.array(goal) - cur), sh)
        if t - getattr(self, "_cut_t", -1e9) < 0.02:
            tau = 1e-3  # the look-at point snaps too
        az, el, dist = self._cam[1]
        return CameraPreset(azimuth=float(az), elevation=float(el), distance=float(dist), tau_s=tau)

    def job_hud_lines(self) -> list[str]:
        msg = self.message if self.run_time() - self._msg_t < 4.0 else ""
        lines = [f"explosions {self.n_explosions}   stream viewers {self.viewers:,}"
                 + (f" (+{self.last_gain:,})" if self.last_gain else "")
                 + f"   vegetables eaten: {self.vegetables_eaten}",
                 f"room rebuilds {self.n_rebuilds}   props launched {self.n_props_launched}"]
        if getattr(self.session, "brain", None) is not None:
            lines.append(f"bitter pulses {self.n_bitter} (stand-in: LB1 bitter GRNs, the host's brain)")
        lines.append(">> " + msg if msg else "(viewer: posed kinematic fly; plate launch + blast: cartoon, see docs)")
        return lines

    def job_stats(self) -> dict:
        lf = self.last_flight or {}
        return {"plates_yeeted": self.n_yeeted, "explosions": self.n_explosions, "viewers": self.viewers,
                "vegetables_eaten": self.vegetables_eaten, "rebuilds": self.n_rebuilds,
                "props_launched": self.n_props_launched, "resets_mid_cycle": self.n_resets_mid,
                "plates_lost": self.n_plates_lost, "flight_timeouts": self.n_flight_timeouts,
                "bitter_pulses": self.n_bitter, "unstuck": self.n_unstuck,
                "last_flight_s": lf.get("flight_s")}
