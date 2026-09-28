"""BARISTA FLY: the fly makes coffee forever (docs/JOBS.md, "barista").

A small café at fly scale ("THE COMPOUND EYE", our own name): a walnut counter, an
espresso machine (portafilter, group head, steam wand, indicator lamps, a pressure
gauge), a grinder, paper cups with the customer's name, a milk jug, a pastry case, a
chalkboard menu and a ticket rail of orders.

* **The fly** stands at the counter in ``BaristaStance`` (all tarsi planted and
  adhering, like ``freeze``; stationary); the job drives both front legs with joint
  targets from damped least-squares IK (``LegDriver`` from the dishwasher job).
* **Real contacts.** The left front leg holds the tamper (a mocap prop at the tarsus)
  and **tamps** the grounds: each press counts only when the tarsus really touches the
  puck's hidden collider (raised by the tamper's thickness) with at least
  ``tamp_force`` uN; the grounds heap flattens press by press. The same leg presses
  the machine's **SHOT** and **STEAM** buttons, and the right front leg rings the
  **bell**: each is triggered by the leg's contact with the button's collider (a press
  that misses is counted, and the job presses it: ``auto_presses``).
* **Kinematic / visual (labelled).** The portafilter's slides (to the grinder, back,
  into the group head), the cups' slides (from the stack, to the pickup counter), the
  jug's slides (to the steam wand, to the pour), the coffee and milk streams, the
  steam, the grounds and the rising liquid level are kinematic / visual. The espresso
  surface steps through crema stages; **latte art** (heart, tulip or rosetta) is a
  texture drawn progressively on the surface while the jug pours. The right front leg
  holds the jug's handle while pouring (the jug follows the tarsus; its tilt is set by
  the job); the **latte-art score** comes from how steadily the real leg tracked the
  pour path (RMS tarsus error), and an unsteady pour draws the wobbly version.
* **Spills (real physics).** Now and then a drop escapes the cup: a free body that
  falls onto the drip tray / counter and leaves a puddle.

Counters: drinks served (the work counter), shots pulled, latte-art score (last and
mean), orders in the queue (the ticket rail), tamps, spills, walk-outs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco as mj
import numpy as np

from fly_simulator.actions.base import smoothstep
from fly_simulator.jobs import barista_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.dishwasher import (DishStance, LegDriver, cam_shot, caption_overlay,
                                           contact_force, leg_geoms)
from fly_simulator.jobs.geometry import contact_kwargs, quat_axis_angle, spot_or_directional
from fly_simulator.jobs.registry import register_job
from fly_simulator.terrain import TERRAIN_BIT

P = "cafe/"
IDENT = np.array([1.0, 0.0, 0.0, 0.0])
DROP_BIT = 128


class BaristaStance(DishStance):
    """All tarsi planted and adhering while the job moves the front legs. Stationary."""

    name = "barista_stance"


def _quat_from_z(v) -> np.ndarray:
    """Quaternion turning +z onto the direction ``v``."""
    v = np.asarray(v, float)
    v = v / max(np.linalg.norm(v), 1e-9)
    z = np.array([0.0, 0.0, 1.0])
    ax = np.cross(z, v)
    s = np.linalg.norm(ax)
    if s < 1e-9:
        return IDENT.copy() if v[2] > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    ang = math.atan2(s, float(z @ v))
    return np.array(quat_axis_angle(ax / s, ang))


def _qmat(q) -> np.ndarray:
    R = np.zeros(9)
    mj.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


@dataclass
class BaristaConfig(JobConfig):
    # --- layout (mm): the fly spawns at the origin facing +x (thorax ~(0.64, 0, 1.08))
    mach_x0: float = 1.0
    mach_x1: float = 3.7
    mach_y: float = 1.35  # the machine's front face (it faces -y, the camera)
    mach_h: float = 2.3
    group_x: float = 2.1
    cup_y: float = 1.0
    tray_z: float = 0.06  # drip tray top
    cup_r0: float = 0.21
    cup_r1: float = 0.25
    cup_h: float = 0.46
    tamp_x: float = 1.92  # the tamping station (basket centre)
    tamp_y: float = 0.12
    basket_r: float = 0.2
    basket_h: float = 0.12
    tamper_t: float = 0.05  # the tamper's base thickness (the puck collider offset)
    grinder_x: float = 4.3
    pad_x: float = 1.62  # the button pad (SHOT, STEAM)
    pad_y: float = 0.55
    pad_top: float = 0.2
    bell_x: float = 1.75
    bell_y: float = -0.72
    bell_r: float = 0.15
    wand_x: float = 2.95
    jug_home: tuple = (3.05, -0.35)
    jug_r: float = 0.2
    jug_h: float = 0.5
    pickup_x0: float = 3.4
    pickup_y: float = -1.2
    pickup_top: float = 0.3
    # --- timing / the work ------------------------------------------------------------
    cup_slide_s: float = 1.0
    pf_move_s: float = 1.0
    grind_s: float = 1.4
    n_tamps: int = 3
    tamp_force: float = 0.3  # uN: a press that touches the puck with at least this counts
    press: float = 0.04  # the tarsus target this far into a collider (a firm press)
    shot_s: float = 3.0
    steam_s: float = 2.4
    jug_move_s: float = 0.9
    pour_s: float = 3.0
    serve_s: float = 1.6
    order_every_s: float = 26.0  # mean time between orders (random, exponential)
    queue_max: int = 6
    queue_start: int = 2
    spill_p: float = 0.3  # chance per drink that a drop escapes (a real free body)
    wobble_mm: float = 0.025  # pour RMS tarsus error above this -> the wobbly art
    jitter_mm: float = 0.05  # the pour's unsteadiness (our model): amplitude, random in [0, this] per drink
    n_cups: int = 6
    n_steam: int = 24
    n_part: int = 30
    n_drops: int = 3
    n_puddles: int = 4
    particle_g: float = 900.0  # mm/s^2: slowed "cartoon" gravity of the visual particles
    shadows: bool = True
    captions: bool = True
    close_ups: bool = True


@register_job
class BaristaJob(EternalJob):
    name = "barista"
    znear = 0.05
    title = "BARISTA FLY"
    tagline = "the fly makes coffee forever"
    work_label = "drinks served"
    config_cls = BaristaConfig
    required_names = (P + "portafilter", P + "puck", P + "btn_shot", P + "bell", P + "jug", P + "cup0")

    def __init__(self, cfg: BaristaConfig | None = None) -> None:
        super().__init__(cfg)
        c = self.cfg
        self.rng = np.random.default_rng(c.seed + 57)
        self.t = 0.0
        self.n_served = 0
        self.n_shots = 0
        self.n_tamps = 0
        self.n_presses = 0
        self.n_auto_presses = 0
        self.n_spills = 0
        self.n_walkouts = 0
        self.n_orders = 0
        self.n_bells = 0
        self.scores: list[float] = []  # last few only (constant memory)
        self.score_sum = 0.0
        self.last_score = None
        self.queue: list[int] = []
        self.order: int | None = None  # the name index being made
        self.pattern = 0
        self.phase = "setup"
        self.message = ""
        self._caption = ("", -1e9)
        self._stance: BaristaStance | None = None
        self.max_tamp_force = 0.0
        self.pour_rms = 0.0

    # ------------------------------------------------------------ positions
    def cup_spot(self) -> np.ndarray:
        c = self.cfg
        return np.array([c.group_x, c.cup_y, c.tray_z])

    def stack_spot(self) -> np.ndarray:
        c = self.cfg
        return np.array([3.45, 0.95, c.tray_z + 0.1])

    def pickup_spot(self, k: int) -> np.ndarray:
        c = self.cfg
        return np.array([c.pickup_x0 + 0.25 + 0.5 * k, c.pickup_y, c.pickup_top])

    def pf_tamp(self) -> np.ndarray:
        c = self.cfg
        return np.array([c.tamp_x, c.tamp_y, 0.0])

    def pf_grinder(self) -> np.ndarray:
        c = self.cfg
        return np.array([c.grinder_x, 0.72, 0.16])

    def pf_group(self) -> np.ndarray:
        c = self.cfg
        return np.array([c.group_x, c.cup_y + 0.02, 0.84])

    def jug_wand(self) -> np.ndarray:
        c = self.cfg
        return np.array([c.wand_x, c.cup_y - 0.02, c.tray_z])

    def jug_pour(self) -> np.ndarray:
        c = self.cfg
        return np.array([c.group_x, c.cup_y - 0.42, c.tray_z + c.cup_h + 0.02])

    def btn_pos(self, which: str) -> np.ndarray:
        c = self.cfg
        dy = 0.1 if which == "shot" else -0.1
        return np.array([c.pad_x, c.pad_y + dy, c.pad_top])

    # ------------------------------------------------------------ build
    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        self._materials(spec)
        self._add_room(spec, vis)
        self._add_machine(spec, vis)
        self._add_tools(spec, vis)
        self._add_cups(spec, vis)
        self._add_particles(spec, vis)
        self._add_lights(spec)
        # the spill drops: real free bodies (they touch the counter, the drip tray and
        # each other, never the fly); parked (contacts off, gravity compensated) when idle
        for i in range(c.n_drops):
            b = wb.add_body(name=P + f"spill{i}", pos=(-20.0 - i, -20.0, 0.1), gravcomp=1.0)
            b.add_freejoint(name=P + f"spill{i}_j")  # a drop: condim-6 rolling friction, so it stops
            b.add_geom(name=P + f"spill{i}_g", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.03, 0, 0), mass=1e-7,
                       contype=DROP_BIT, conaffinity=DROP_BIT | TERRAIN_BIT, condim=6, priority=1,
                       friction=(1.0, 0.01, 0.02), solref=(0.0005, 1.0), material=P + "coffee_drop")

    def _materials(self, spec) -> None:
        c = self.cfg
        T = A.add_textured_material
        A.add_texture(spec, P + "tex_counter", A.butcher_block_texture(c.seed + 3))
        T(spec, P + "counter", P + "tex_counter", rgba=(0.62, 0.45, 0.36, 1), specular=0.35, shininess=0.6)
        A.add_texture(spec, P + "tex_wall", A.wall_texture())
        T(spec, P + "wall", P + "tex_wall", rgba=(1, 1, 1, 1), specular=0.05)
        A.add_texture(spec, P + "tex_board", A.chalkboard_texture())
        T(spec, P + "board", P + "tex_board", rgba=(1, 1, 1, 1), specular=0.05, emission=0.15)
        A.add_texture(spec, P + "tex_front", A.machine_front_texture())
        T(spec, P + "front", P + "tex_front", rgba=(1, 1, 1, 1), specular=0.8, shininess=0.9)
        A.add_texture(spec, P + "tex_steel", A.steel_texture(c.seed))
        T(spec, P + "steel", P + "tex_steel", rgba=(1, 1, 1, 1), specular=0.9, shininess=0.85, reflectance=0.05)
        A.add_texture(spec, P + "tex_gauge", A.gauge_texture())
        T(spec, P + "gauge", P + "tex_gauge", rgba=(1, 1, 1, 1), specular=0.6, emission=0.2)
        A.add_texture(spec, P + "tex_paper", A.paper_texture())
        T(spec, P + "paper", P + "tex_paper", rgba=(1, 1, 1, 1), specular=0.2)
        A.add_texture(spec, P + "tex_pastry", A.pastry_texture())
        T(spec, P + "pastry", P + "tex_pastry", rgba=(1, 1, 1, 1), specular=0.5, shininess=0.6)
        A.add_texture(spec, P + "tex_beans", A.beans_texture())
        T(spec, P + "beans", P + "tex_beans", rgba=(1, 1, 1, 1), specular=0.4)
        A.add_texture(spec, P + "tex_grounds", A.grounds_texture())
        T(spec, P + "grounds", P + "tex_grounds", rgba=(1, 1, 1, 1), specular=0.05)
        A.add_texture(spec, P + "tex_pickup", A.sign_texture("PICK UP", "orders here - kinematic slide"))
        T(spec, P + "pickup_sign", P + "tex_pickup", rgba=(1, 1, 1, 1), emission=0.25)
        for j in range(len(A.NAMES)):
            A.add_texture(spec, P + f"tex_ticket{j}", A.ticket_texture(j))
            T(spec, P + f"ticket{j}", P + f"tex_ticket{j}", rgba=(1, 1, 1, 1), emission=0.1)
            A.add_texture(spec, P + f"tex_sleeve{j}", A.sleeve_texture(j))
            T(spec, P + f"sleeve{j}", P + f"tex_sleeve{j}", rgba=(1, 1, 1, 1), specular=0.1)
        for k in range(A.CREMA_STAGES):
            A.add_texture(spec, P + f"tex_crema{k}", A.crema_texture(k))
            T(spec, P + f"crema{k}", P + f"tex_crema{k}", rgba=(1, 1, 1, 1), specular=0.7, shininess=0.8)
        for pat in range(len(A.PATTERNS)):
            for wob in range(2):
                for s in range(A.ART_STAGES):
                    A.add_texture(spec, P + f"tex_art{pat}{wob}{s}", A.art_texture(pat, wob, s))
                    T(spec, P + f"art{pat}{wob}{s}", P + f"tex_art{pat}{wob}{s}", rgba=(1, 1, 1, 1), specular=0.6,
                      shininess=0.8)
        M = spec.add_material
        M(name=P + "chrome", rgba=(0.82, 0.84, 0.88, 1), specular=1.0, shininess=0.95, reflectance=0.12)
        M(name=P + "black", rgba=(0.06, 0.06, 0.07, 1), specular=0.4)
        M(name=P + "wood", rgba=(0.45, 0.28, 0.15, 1), specular=0.3)
        M(name=P + "plank", rgba=(0.3, 0.19, 0.12, 1), specular=0.3)
        M(name=P + "cup_in", rgba=(0.96, 0.95, 0.92, 1), specular=0.2)
        M(name=P + "hidden", rgba=(1, 1, 1, 0))
        M(name=P + "milk", rgba=(0.98, 0.97, 0.94, 1), specular=0.4)
        M(name=P + "foam", rgba=(1.0, 0.99, 0.97, 1), specular=0.2)
        M(name=P + "coffee", rgba=(0.28, 0.15, 0.07, 0.95), specular=0.8, shininess=0.8)
        M(name=P + "coffee_drop", rgba=(0.3, 0.16, 0.07, 1), specular=0.9, shininess=0.9)
        M(name=P + "milk_stream", rgba=(0.98, 0.96, 0.9, 0.95), specular=0.5)
        M(name=P + "steam", rgba=(1, 1, 1, 0.35), specular=0.0)
        M(name=P + "glass", rgba=(0.85, 0.92, 0.95, 0.18), specular=1.0, shininess=1.0)
        M(name=P + "shelf", rgba=(0.9, 0.9, 0.88, 1), specular=0.4)
        M(name=P + "muffin_top", rgba=(0.45, 0.25, 0.12, 1), specular=0.2)
        M(name=P + "muffin_cup", rgba=(0.85, 0.3, 0.35, 1), specular=0.3)
        M(name=P + "cookie", rgba=(0.8, 0.6, 0.35, 1), specular=0.2)
        M(name=P + "choc", rgba=(0.25, 0.13, 0.07, 1), specular=0.3)
        M(name=P + "brass", rgba=(0.85, 0.65, 0.28, 1), specular=1.0, shininess=0.9, reflectance=0.1)
        M(name=P + "lamp_off", rgba=(0.2, 0.2, 0.2, 1), emission=0.0, specular=0.6)
        M(name=P + "lamp_green", rgba=(0.3, 1.0, 0.4, 1), emission=1.0)
        M(name=P + "lamp_amber", rgba=(1.0, 0.65, 0.15, 1), emission=1.0)
        M(name=P + "lamp_blue", rgba=(0.35, 0.65, 1.0, 1), emission=1.0)
        M(name=P + "btn_shot", rgba=(0.45, 0.25, 0.12, 1), specular=0.6)
        M(name=P + "btn_steam", rgba=(0.92, 0.92, 0.95, 1), specular=0.6)
        M(name=P + "grinder", rgba=(0.12, 0.12, 0.13, 1), specular=0.6, shininess=0.7)
        M(name=P + "bulb", rgba=(1.0, 0.85, 0.55, 1), emission=1.0)
        M(name=P + "shade", rgba=(0.15, 0.3, 0.28, 1), specular=0.5)
        M(name=P + "mat", rgba=(0.12, 0.12, 0.12, 1), specular=0.1)
        M(name=P + "rail", rgba=(0.7, 0.72, 0.75, 1), specular=0.8)

    def _add_room(self, spec, vis) -> None:
        c = self.cfg
        wb = spec.worldbody
        md = A.slab_mesh(18.0, 6.6, 0.45)
        md = A.MeshData(md.verts, md.faces, np.column_stack([md.verts[:, 0] / 5.0, md.verts[:, 1] / 2.5
                                                             + 0.3 * md.verts[:, 2]]))
        A.add_mesh(spec, P + "counter_mesh", md)
        wb.add_geom(name=P + "counter", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "counter_mesh",
                    pos=(1.0, -0.4, 0.0), material=P + "counter", **vis)
        wy = 2.9
        wmd = A.panel_mesh(20.0, 8.0, 0.05)
        A.add_mesh(spec, P + "wall_mesh", A.MeshData(wmd.verts, wmd.faces, wmd.uv * np.array([3.0, 1.2])))
        wb.add_geom(name=P + "wall", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "wall_mesh",
                    pos=(1.0, wy, 4.0), quat=A.panel_quat((0, -1, 0)), material=P + "wall", **vis)
        # the chalkboard menu over the pastry case
        A.add_mesh(spec, P + "board_mesh", A.panel_mesh(3.4, 2.55, 0.06))
        wb.add_geom(name=P + "board", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "board_mesh",
                    pos=(-2.6, wy - 0.06, 2.35), quat=A.panel_quat((0, -1, 0)), material=P + "board", **vis)
        # the ticket rail over the machine
        wb.add_geom(name=P + "rail", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.03, 1.5, 0),
                    pos=(2.35, wy - 0.1, 2.98), quat=quat_axis_angle((0, 1, 0), math.pi / 2), material=P + "rail",
                    **vis)
        A.add_mesh(spec, P + "ticket_mesh", A.panel_mesh(0.34, 0.48, 0.01))
        self._ticket_home = [np.array([1.15 + 0.46 * k, wy - 0.13, 2.74]) for k in range(c.queue_max)]
        for k in range(c.queue_max):
            b = wb.add_body(name=P + f"ticket{k}", mocap=True, pos=(0, 0, -30.0))
            b.add_geom(name=P + f"ticket{k}_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "ticket_mesh",
                       quat=A.panel_quat((0, -1, 0)), material=P + "ticket0", **vis)
        # the "now making" ticket clip on the machine front
        b = wb.add_body(name=P + "now", mocap=True, pos=(0, 0, -30.0))
        b.add_geom(name=P + "now_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "ticket_mesh",
                   quat=A.panel_quat((0, -1, 0)), material=P + "ticket0", **vis)
        # the pastry case (glass, two shelves, croissants, muffins, cookies)
        px0, px1, py0, py1, ph = -4.3, -1.3, 0.9, 2.3, 1.3
        cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
        wb.add_geom(name=P + "case_base", type=mj.mjtGeom.mjGEOM_BOX, size=((px1 - px0) / 2, (py1 - py0) / 2, 0.06),
                    pos=(cx, cy, 0.06), material=P + "wood", **vis)
        for nm, size, pos in (("case_front", ((px1 - px0) / 2, 0.01, ph / 2), (cx, py0, 0.12 + ph / 2)),
                              ("case_top", ((px1 - px0) / 2, (py1 - py0) / 2, 0.01), (cx, cy, 0.12 + ph)),
                              ("case_l", (0.01, (py1 - py0) / 2, ph / 2), (px0, cy, 0.12 + ph / 2)),
                              ("case_r", (0.01, (py1 - py0) / 2, ph / 2), (px1, cy, 0.12 + ph / 2))):
            wb.add_geom(name=P + nm, type=mj.mjtGeom.mjGEOM_BOX, size=size, pos=pos, material=P + "glass", **vis)
        wb.add_geom(name=P + "case_shelf", type=mj.mjtGeom.mjGEOM_BOX, size=((px1 - px0) / 2 - 0.05, (py1 - py0) / 2 - 0.05, 0.015),
                    pos=(cx, cy, 0.12 + 0.55 * ph), material=P + "glass", **vis)
        A.add_mesh(spec, P + "croissant_mesh", A.croissant_mesh(0.55, 0.13))
        A.add_mesh(spec, P + "muffin_mesh", A.muffin_mesh(0.18, 0.3))
        A.add_mesh(spec, P + "cookie_mesh", A.disc_mesh(0.16, 0.04, 24))
        rng = np.random.default_rng(c.seed + 9)
        for i in range(5):
            wb.add_geom(name=P + f"croissant{i}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "croissant_mesh",
                        pos=(px0 + 0.4 + 0.55 * i, cy - 0.2 + 0.2 * (i % 2), 0.18),
                        quat=quat_axis_angle((0, 0, 1), rng.uniform(-0.5, 0.5)), material=P + "pastry", **vis)
        for i in range(5):
            x = px0 + 0.4 + 0.58 * i
            wb.add_geom(name=P + f"muffin{i}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "muffin_mesh",
                        pos=(x, cy + 0.1, 0.12 + 0.55 * ph + 0.015), material=P + "muffin_cup", **vis)
            wb.add_geom(name=P + f"muffin{i}_top", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(0.2, 0.2, 0.1),
                        pos=(x, cy + 0.1, 0.12 + 0.55 * ph + 0.28), material=P + "muffin_top", **vis)
            wb.add_geom(name=P + f"cookie{i}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "cookie_mesh",
                        pos=(x + 0.1, cy - 0.35, 0.12 + 0.55 * ph + 0.035), material=P + "cookie", **vis)
            for j in range(3):
                a = rng.uniform(0, 2 * math.pi)
                wb.add_geom(name=P + f"chip{i}_{j}", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.018, 0, 0),
                            pos=(x + 0.1 + 0.08 * math.cos(a), cy - 0.35 + 0.08 * math.sin(a), 0.12 + 0.55 * ph + 0.06),
                            material=P + "choc", **vis)
        # the pickup counter (a raised plank) with its sign
        L = 1.7
        wb.add_geom(name=P + "pickup", type=mj.mjtGeom.mjGEOM_BOX, size=(L / 2, 0.3, c.pickup_top / 2),
                    pos=(c.pickup_x0 + L / 2 - 0.1, c.pickup_y, c.pickup_top / 2), material=P + "plank", **vis)
        A.add_mesh(spec, P + "pickup_sign_mesh", A.panel_mesh(1.2, 0.3, 0.02))
        wb.add_geom(name=P + "pickup_sign", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "pickup_sign_mesh",
                    pos=(c.pickup_x0 + L / 2 - 0.1, c.pickup_y - 0.31, c.pickup_top / 2),
                    quat=A.panel_quat((0, -1, 0)), material=P + "pickup_sign", **vis)
        # pendant lamps
        for i, x in enumerate((-1.5, 2.3, 5.5)):
            wb.add_geom(name=P + f"cord{i}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.01, 1.0, 0),
                        pos=(x, 0.6, 6.0), material=P + "black", **vis)
            A.add_mesh(spec, P + f"shade{i}_mesh", A.lathe(np.array([0.0, 0.08, 0.45, 0.43, 0.06, 0.0]),
                                                          np.array([0.35, 0.35, 0.0, 0.0, 0.33, 0.33]), 24))
            wb.add_geom(name=P + f"shade{i}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + f"shade{i}_mesh",
                        pos=(x, 0.6, 4.65), material=P + "shade", **vis)
            wb.add_geom(name=P + f"bulb{i}", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.12, 0, 0),
                        pos=(x, 0.6, 4.7), material=P + "bulb", **vis)
        sky = spec.texture("skybox")
        if sky is not None:
            sky.rgb1 = (0.3, 0.24, 0.18)
            sky.rgb2 = (0.08, 0.06, 0.05)

    def _add_machine(self, spec, vis) -> None:
        c = self.cfg
        wb = spec.worldbody
        x0, x1, fy, H = c.mach_x0, c.mach_x1, c.mach_y, c.mach_h
        cx = (x0 + x1) / 2
        depth = 1.1
        wb.add_geom(name=P + "mach_body", type=mj.mjtGeom.mjGEOM_BOX, size=((x1 - x0) / 2, depth / 2, H / 2),
                    pos=(cx, fy + depth / 2, H / 2), material=P + "steel", **vis)
        A.add_mesh(spec, P + "front_mesh", A.panel_mesh(x1 - x0 - 0.1, 0.9, 0.02))
        wb.add_geom(name=P + "mach_front", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "front_mesh",
                    pos=(cx, fy - 0.012, H - 0.6), quat=A.panel_quat((0, -1, 0)), material=P + "front", **vis)
        # cup warmer on top with a few cups
        wb.add_geom(name=P + "warmer", type=mj.mjtGeom.mjGEOM_BOX, size=((x1 - x0) / 2, depth / 2, 0.03),
                    pos=(cx, fy + depth / 2, H + 0.03), material=P + "chrome", **vis)
        A.add_mesh(spec, P + "cup_mesh", A.paper_cup_mesh(c.cup_r0, c.cup_r1, c.cup_h))
        # the group head (chrome) and its neck
        gx = c.group_x
        wb.add_geom(name=P + "group_neck", type=mj.mjtGeom.mjGEOM_BOX, size=(0.2, 0.2, 0.12),
                    pos=(gx, fy - 0.15, 1.1), material=P + "chrome", **vis)
        wb.add_geom(name=P + "group", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.27, 0.07, 0),
                    pos=(gx, c.cup_y + 0.02, 1.02), material=P + "chrome", **vis)
        # the pressure gauge and the lamps (READY / BREW / STEAM)
        A.add_mesh(spec, P + "gauge_mesh", A.disc_mesh(0.22, 0.03, 32))
        wb.add_geom(name=P + "gauge", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "gauge_mesh",
                    pos=(gx + 0.8, fy - 0.03, 1.35), quat=quat_axis_angle((1, 0, 0), math.pi / 2),
                    material=P + "gauge", **vis)
        nb = wb.add_body(name=P + "needle", mocap=True, pos=(gx + 0.8, fy - 0.06, 1.35))
        nb.add_geom(name=P + "needle_g", type=mj.mjtGeom.mjGEOM_BOX, size=(0.14, 0.005, 0.008),
                    pos=(0.1, 0, 0), material=P + "black", **vis)
        self._lamps = ("ready", "brew", "steam")
        for i, nm in enumerate(self._lamps):
            wb.add_geom(name=P + f"lamp_{nm}", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.06, 0, 0),
                        pos=(x0 + 0.35 + 0.22 * i, fy - 0.03, 1.35), material=P + "lamp_off", **vis)
        # the drip tray (a static collider for the spill drops, the grate drawn over it)
        tx0, tx1, ty0 = 1.6, 3.7, 0.7
        wb.add_geom(name=P + "tray_col", type=mj.mjtGeom.mjGEOM_BOX, size=((tx1 - tx0) / 2, (fy - ty0) / 2, c.tray_z / 2),
                    pos=((tx0 + tx1) / 2, (ty0 + fy) / 2, c.tray_z / 2), rgba=(0, 0, 0, 0), group=3, mass=0.0,
                    contype=TERRAIN_BIT, conaffinity=DROP_BIT, condim=3, priority=1)
        wb.add_geom(name=P + "tray", type=mj.mjtGeom.mjGEOM_BOX, size=((tx1 - tx0) / 2, (fy - ty0) / 2, c.tray_z / 2),
                    pos=((tx0 + tx1) / 2, (ty0 + fy) / 2, c.tray_z / 2 - 0.002), material=P + "chrome", **vis)
        for k in range(14):
            wb.add_geom(name=P + f"slot{k}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.025, (fy - ty0) / 2 - 0.05, 0.002),
                        pos=(tx0 + 0.12 + 0.14 * k, (ty0 + fy) / 2, c.tray_z), material=P + "black", **vis)
        # the steam wand (a bent chrome tube from the machine front)
        wx = c.wand_x
        pts = np.array([[wx + 0.15, fy, 1.25], [wx + 0.1, fy - 0.15, 1.1], [wx, c.cup_y + 0.05, 0.85],
                        [wx, c.cup_y - 0.02, 0.45]])
        A.add_mesh(spec, P + "wand_mesh", A.polyline_tube(pts, 0.035, 10))
        wb.add_geom(name=P + "wand", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "wand_mesh", material=P + "chrome",
                    **vis)
        wb.add_geom(name=P + "wand_knob", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.09, 0.06, 0),
                    pos=(wx + 0.4, fy - 0.04, 1.5), quat=quat_axis_angle((1, 0, 0), math.pi / 2),
                    material=P + "black", **vis)
        # the cup stack (upside down, nested) at the tray's end
        sp = self.stack_spot()
        for i in range(4):
            wb.add_geom(name=P + f"stack_cup{i}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "cup_mesh",
                        pos=(sp[0], sp[1], c.tray_z + c.cup_h + 0.07 * i), quat=(0, 1, 0, 0),
                        material=P + "paper", **vis)
        # the grinder (right of the machine)
        gx0 = c.grinder_x
        wb.add_geom(name=P + "grinder", type=mj.mjtGeom.mjGEOM_BOX, size=(0.4, 0.45, 0.75),
                    pos=(gx0, 1.45, 0.75), material=P + "grinder", **vis)
        wb.add_geom(name=P + "grinder_chute", type=mj.mjtGeom.mjGEOM_BOX, size=(0.12, 0.25, 0.08),
                    pos=(gx0, 0.95, 0.72), material=P + "chrome", **vis)
        wb.add_geom(name=P + "grinder_fork", type=mj.mjtGeom.mjGEOM_BOX, size=(0.22, 0.2, 0.02),
                    pos=(gx0, 0.86, 0.14), material=P + "chrome", **vis)
        A.add_mesh(spec, P + "hopper_mesh", A.hopper_mesh(0.42, 0.75))
        wb.add_geom(name=P + "hopper", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "hopper_mesh",
                    pos=(gx0, 1.45, 1.5), material=P + "glass", **vis)
        wb.add_geom(name=P + "beans", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.33, 0.01, 0),
                    pos=(gx0, 1.45, 2.05), material=P + "beans", **vis)
        wb.add_geom(name=P + "beans_heap", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(0.3, 0.3, 0.12),
                    pos=(gx0, 1.45, 2.05), material=P + "beans", **vis)

    def _add_tools(self, spec, vis) -> None:
        c = self.cfg
        wb = spec.worldbody
        # the tamping mat
        wb.add_geom(name=P + "tamp_mat", type=mj.mjtGeom.mjGEOM_BOX, size=(0.34, 0.34, 0.006),
                    pos=(c.tamp_x, c.tamp_y, 0.006), material=P + "mat", **vis)
        # the portafilter (mocap): basket, grounds heap, handle, two spouts
        pf = wb.add_body(name=P + "portafilter", mocap=True, pos=tuple(self.pf_tamp()))
        A.add_mesh(spec, P + "basket_mesh", A.basket_mesh(c.basket_r, c.basket_h))
        pf.add_geom(name=P + "basket", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "basket_mesh",
                    material=P + "chrome", **vis)
        pf.add_geom(name=P + "grounds", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(c.basket_r - 0.025, c.basket_r - 0.025, 0.02),
                    pos=(0, 0, 0.03), material=P + "grounds", **vis)
        pf.add_geom(name=P + "pf_handle", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.045, 0.28, 0),
                    pos=(0, -c.basket_r - 0.3, 0.07), quat=quat_axis_angle((1, 0, 0), math.pi / 2 - 0.15),
                    material=P + "black", **vis)
        for s in (-1, 1):
            pf.add_geom(name=P + f"spout{int(s > 0)}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.025, 0.04, 0),
                        pos=(0.06 * s, 0, -0.04), material=P + "chrome", **vis)
        # the puck's hidden collider (mocap; at the tamp station only while tamping)
        pk = wb.add_body(name=P + "puck", mocap=True, pos=(0, 0, -30.0))
        pk.add_geom(name=P + "puck_col", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(c.basket_r - 0.03, 0.05, 0),
                    pos=(0, 0, -0.05), rgba=(0, 0, 0, 0), group=3, mass=0.0, **contact_kwargs("fly", friction=0.6))
        # the tamper (mocap at the left front tarsus)
        tb = wb.add_body(name=P + "tamper", mocap=True, pos=(1.4, 0.9, 0.1))
        A.add_mesh(spec, P + "tamper_mesh", A.tamper_mesh(c.basket_r - 0.04, c.tamper_t, 0.2))
        tb.add_geom(name=P + "tamper_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "tamper_mesh",
                    pos=(0, 0, -c.tamper_t + 0.01), material=P + "steel", **vis)
        # the button pad: SHOT, STEAM (colliders on mocap caps: the leg presses them)
        wb.add_geom(name=P + "pad", type=mj.mjtGeom.mjGEOM_BOX, size=(0.11, 0.2, c.pad_top / 2 - 0.01),
                    pos=(c.pad_x, c.pad_y, c.pad_top / 2 - 0.01), material=P + "steel", **vis)
        for which in ("shot", "steam"):
            p = self.btn_pos(which)
            b = wb.add_body(name=P + f"btn_{which}", mocap=True, pos=tuple(p))
            b.add_geom(name=P + f"btn_{which}_col", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.065, 0.02, 0),
                       pos=(0, 0, 0.0), rgba=(0, 0, 0, 0), group=3, mass=0.0, **contact_kwargs("fly", friction=0.8))
            b.add_geom(name=P + f"btn_{which}_g", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.05, 0.018, 0),
                       material=P + f"btn_{which}", **vis)
        # the bell (brass dome; the plunger's collider on the mocap body)
        bb = wb.add_body(name=P + "bell", mocap=True, pos=(c.bell_x, c.bell_y, 0.0))
        bb.add_geom(name=P + "bell_col", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.05, 0.02, 0),
                    pos=(0, 0, 0.9 * c.bell_r + 0.03), rgba=(0, 0, 0, 0), group=3, mass=0.0,
                    **contact_kwargs("fly", friction=0.8))
        A.add_mesh(spec, P + "bell_mesh", A.bell_mesh(c.bell_r))
        bb.add_geom(name=P + "bell_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "bell_mesh", material=P + "brass",
                    **vis)
        bb.add_geom(name=P + "bell_plunger", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.02, 0.025, 0),
                    pos=(0, 0, 0.9 * c.bell_r + 0.02), material=P + "chrome", **vis)
        # the milk jug (mocap): the jug, its spout, the milk, the handle
        jb = wb.add_body(name=P + "jug", mocap=True, pos=(*c.jug_home, 0.0))
        A.add_mesh(spec, P + "jug_mesh", A.jug_mesh(c.jug_r, c.jug_h))
        A.add_mesh(spec, P + "jug_spout_mesh", A.spout_mesh(c.jug_r, c.jug_h))
        jb.add_geom(name=P + "jug_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "jug_mesh", material=P + "chrome",
                    **vis)
        jb.add_geom(name=P + "jug_spout", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "jug_spout_mesh",
                    material=P + "chrome", **vis)
        jb.add_geom(name=P + "milk", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(c.jug_r * 0.86, 0.004, 0),
                    pos=(0, 0, 0.3 * c.jug_h), material=P + "foam", **vis)
        A.add_mesh(spec, P + "handle_mesh", A.polyline_tube(np.array([[0, -0.85 * c.jug_r, 0.8 * c.jug_h],
                                                                        [0, -1.5 * c.jug_r, 0.75 * c.jug_h],
                                                                        [0, -1.55 * c.jug_r, 0.4 * c.jug_h],
                                                                        [0, -0.95 * c.jug_r, 0.25 * c.jug_h]]),
                                                              0.03, 8))
        jb.add_geom(name=P + "jug_handle", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "handle_mesh",
                    material=P + "black", **vis)

    def _add_cups(self, spec, vis) -> None:
        c = self.cfg
        wb = spec.worldbody
        A.add_mesh(spec, P + "sleeve_mesh", A.band_mesh(c.cup_r0 + 0.3 * (c.cup_r1 - c.cup_r0) + 0.012,
                                                        c.cup_r0 + 0.75 * (c.cup_r1 - c.cup_r0) + 0.012,
                                                        0.3 * c.cup_h, 0.75 * c.cup_h))
        A.add_mesh(spec, P + "surf_mesh", A.disc_mesh(c.cup_r0 + 0.012, 0.006, 40))
        for i in range(c.n_cups):
            b = wb.add_body(name=P + f"cup{i}", mocap=True, pos=(0, -40.0 - i, -30.0))
            b.add_geom(name=P + f"cup{i}_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "cup_mesh",
                       material=P + "paper", **vis)
            b.add_geom(name=P + f"cup{i}_sleeve", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "sleeve_mesh",
                       material=P + "sleeve0", **vis)
            b.add_geom(name=P + f"cup{i}_surf", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "surf_mesh",
                       pos=(0, 0, 0.04), material=P + "cup_in", **vis)

    def _add_particles(self, spec, vis) -> None:
        c = self.cfg
        wb = spec.worldbody
        for kind, n, mat, r in (("steam", c.n_steam, "steam", 0.06), ("part", c.n_part, "coffee_drop", 0.018)):
            for i in range(n):
                b = wb.add_body(name=P + f"{kind}{i}", mocap=True, pos=(0, -30.0 - i, -30.0))
                b.add_geom(name=P + f"{kind}{i}_g", type=mj.mjtGeom.mjGEOM_SPHERE, size=(r, 0, 0),
                           material=P + mat, **vis)
        for k in range(2):
            b = wb.add_body(name=P + f"stream{k}", mocap=True, pos=(0, 0, -30.0))
            b.add_geom(name=P + f"stream{k}_g", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.018, 0.2, 0),
                       material=P + "coffee", **vis)
        b = wb.add_body(name=P + "milkstream", mocap=True, pos=(0, 0, -30.0))
        b.add_geom(name=P + "milkstream_g", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.028, 0.2, 0),
                   material=P + "milk_stream", **vis)
        for i in range(c.n_puddles):
            b = wb.add_body(name=P + f"puddle{i}", mocap=True, pos=(0, -30.0, -30.0))
            b.add_geom(name=P + f"puddle{i}_g", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(0.09, 0.07, 0.004),
                       rgba=(0.3, 0.16, 0.07, 0.85), **vis)

    def _add_lights(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        spec.visual.headlight.ambient = (0.28, 0.24, 0.2)
        spec.visual.headlight.diffuse = (0.3, 0.27, 0.23)
        spec.visual.headlight.specular = (0.1, 0.1, 0.1)
        key = np.array([0.0, -5.5, 8.5])
        tgt = np.array([1.8, 0.6, 0.0])
        wb.add_light(name=P + "key", type=spot_or_directional(c.shadows), pos=tuple(key), dir=tuple(tgt - key),
                     diffuse=(0.72, 0.6, 0.45), specular=(0.35, 0.3, 0.25), cutoff=40.0, exponent=0.6,
                     castshadow=bool(c.shadows))
        wb.add_light(name=P + "fill", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(6.0, -3.0, 5.0),
                     dir=(-0.6, 0.5, -0.6), diffuse=(0.2, 0.2, 0.24), specular=(0.05, 0.05, 0.05), castshadow=False)
        for i, x in enumerate((-1.5, 2.3, 5.5)):
            wb.add_light(name=P + f"pendant{i}", type=mj.mjtLightType.mjLIGHT_POINT, pos=(x, 0.6, 4.4),
                         diffuse=(0.35, 0.26, 0.14), specular=(0.1, 0.08, 0.05), attenuation=(0.5, 0.06, 0.01),
                         castshadow=False)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        m, s = self.sim.model, self.session
        c = self.cfg
        gp = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "ground_plane")
        if gp >= 0:
            m.geom_group[gp] = 3
            m.geom_rgba[gp, 3] = 0.0
        self.acts = s.actions
        mid = lambda nm: int(m.body_mocapid[m.body(nm).id])  # noqa: E731
        gid = lambda nm: m.geom(nm).id  # noqa: E731
        mat = lambda nm: m.material(P + nm).id  # noqa: E731
        self.pf_mocap = mid(P + "portafilter")
        self.grounds_geom = gid(P + "grounds")
        self.puck_mocap = mid(P + "puck")
        self.puck_geom = gid(P + "puck_col")
        self.tamper_mocap = mid(P + "tamper")
        self.btn_mocap = {w: mid(P + f"btn_{w}") for w in ("shot", "steam")}
        self.btn_geom = {w: gid(P + f"btn_{w}_col") for w in ("shot", "steam")}
        self.bell_mocap = mid(P + "bell")
        self.bell_geom = gid(P + "bell_col")
        self.jug_mocap = mid(P + "jug")
        self.milk_geom = gid(P + "milk")
        self.needle_mocap = mid(P + "needle")
        self.ticket_mocap = [mid(P + f"ticket{k}") for k in range(c.queue_max)]
        self.ticket_geom = [gid(P + f"ticket{k}_g") for k in range(c.queue_max)]
        self.now_mocap = mid(P + "now")
        self.now_geom = gid(P + "now_g")
        self.stream_mocap = [mid(P + f"stream{k}") for k in range(2)]
        self.stream_geom = [gid(P + f"stream{k}_g") for k in range(2)]
        self.milkstream_mocap = mid(P + "milkstream")
        self.milkstream_geom = gid(P + "milkstream_g")
        self.lamp_geom = {nm: gid(P + f"lamp_{nm}") for nm in self._lamps}
        self.mat = {nm: mat(nm) for nm in ("lamp_off", "lamp_green", "lamp_amber", "lamp_blue", "cup_in", "hidden")}
        self.ticket_mat = [mat(f"ticket{j}") for j in range(len(A.NAMES))]
        self.sleeve_mat = [mat(f"sleeve{j}") for j in range(len(A.NAMES))]
        self.crema_mat = [mat(f"crema{k}") for k in range(A.CREMA_STAGES)]
        self.art_mat = [[[mat(f"art{p}{w}{s}") for s in range(A.ART_STAGES)] for w in range(2)]
                        for p in range(len(A.PATTERNS))]
        self.cups = [dict(mocap=mid(P + f"cup{i}"), sleeve=gid(P + f"cup{i}_sleeve"), surf=gid(P + f"cup{i}_surf"),
                          where="parked", pos=np.array([0.0, -40.0 - i, -30.0]), level=0.0, surf_mat=self.mat["cup_in"],
                          name=0) for i in range(c.n_cups)]
        self.pickup: list[int] = []
        self.parts = {}
        for kind, n in (("steam", c.n_steam), ("part", c.n_part)):
            g0 = gid(P + f"{kind}0_g")
            self.parts[kind] = dict(mocap=np.array([mid(P + f"{kind}{i}") for i in range(n)]),
                                    geom=np.array([gid(P + f"{kind}{i}_g") for i in range(n)]),
                                    pos=np.zeros((n, 3)), vel=np.zeros((n, 3)), life=np.zeros(n), age=np.zeros(n),
                                    r0=float(m.geom_size[g0, 0]), k=0, floor=np.zeros(n))
        self.drops = []
        for i in range(c.n_drops):
            b = m.body(P + f"spill{i}").id
            j = m.joint(P + f"spill{i}_j")
            self.drops.append(dict(body=b, geom=gid(P + f"spill{i}_g"), q=int(m.jnt_qposadr[j.id]),
                                   v=int(m.jnt_dofadr[j.id]), state="parked", t=0.0, park=np.array([-20.0 - i, -20.0, 0.1])))
        self.puddles = [dict(mocap=mid(P + f"puddle{i}"), geom=gid(P + f"puddle{i}_g"), t=-1e9) for i in range(c.n_puddles)]
        self._k_puddle = 0
        self.lf_geoms = leg_geoms(m, self.fly_name, "lf")
        self.rf_geoms = leg_geoms(m, self.fly_name, "rf")
        self.tarsus5 = {leg: m.body(f"{self.fly_name}/{leg}_tarsus5").id for leg in ("lf", "rf")}
        self.legs = LegDriver(self)
        st = getattr(s, "STATIONARY_ACTIONS", None)
        if st is not None and BaristaStance.name not in st:
            s.STATIONARY_ACTIONS = (*st, BaristaStance.name)
        # the props' state
        self.pf_pos = self.pf_tamp()
        self.pf_quat = IDENT.copy()
        self.grounds = 0.0  # 0 = empty basket, 1 = dosed heap
        self.tamp_level = 0.0  # 0 loose .. 1 fully tamped
        self.jug_pos = np.array([*self.cfg.jug_home, 0.0])
        self.jug_quat = IDENT.copy()
        self.foam = 0.0
        self.cup: int | None = None
        self.pressure = 0.0
        self.lamp = "ready"
        self._t_next_order = 0.0
        for _ in range(c.queue_start):
            self._new_order()
        self._resume = "wait_order"
        self._begin()

    def _begin(self) -> None:
        self._stance = BaristaStance()
        self.acts.trigger(self._stance, source="job")
        self.legs.stance = self._stance
        self.legs.clear()
        self.steering.set(None, 0.0)
        self._park_puck()
        self._go("setup")

    def _go(self, phase: str) -> None:
        self.phase = self.state = phase
        self._t_phase = self.t

    # ------------------------------------------------------------ orders
    def _new_order(self) -> None:
        c = self.cfg
        if len(self.queue) >= c.queue_max:
            self.n_walkouts += 1
        else:
            self.queue.append(int(self.rng.integers(0, len(A.NAMES))))
            self.n_orders += 1
        self._t_next_order = self.t + float(self.rng.exponential(c.order_every_s))

    def tarsus(self, leg: str) -> np.ndarray:
        return self.sim.data.xpos[self.tarsus5[leg]].copy()

    def puck_top(self) -> float:
        """The grounds' top (the tamper's face meets it) at the tamp station."""
        return 0.03 + self._grounds_hz() * 2 * 0.8

    def _grounds_hz(self) -> float:
        # heap half-height: loose dose 0.075 -> tamped flat 0.035
        return 0.02 + self.grounds * (0.055 - 0.04 * self.tamp_level)

    def _park_puck(self) -> None:
        self.sim.data.mocap_pos[self.puck_mocap] = (0.0, 0.0, -30.0)

    # ------------------------------------------------------------ the loop
    def update(self) -> None:
        c = self.cfg
        dt = c.update_every_steps * self.sim.timestep
        self.t += dt
        t = self.t
        st = self._stance
        if st is None or self.acts.action is not st:
            if self.acts.action is None and not self.fly_down():
                self._stance = BaristaStance()
                self.acts.trigger(self._stance, source="job")
                self.legs.stance = self._stance
        if t >= self._t_next_order:
            self._new_order()
        self._step(t, dt)
        self.legs.step()
        self._props_step(t, dt)
        if self.message and t - self._caption[1] > 2.4:
            self.message = ""

    def _press(self, leg: str, geom: int, p_top: np.ndarray, label: str, el: float, sub: str) -> bool | None:
        """A button press by a front leg: down to ``press`` into the collider; returns
        True on contact (the leg really touched it), False if the motion ended without
        (a miss: counted, the job presses it), None while still going."""
        c = self.cfg
        L = self.legs
        if sub == "down":
            f = contact_force(self.sim.model, self.sim.data, geom, self.lf_geoms if leg == "lf" else self.rf_geoms)
            if f > 0.05:
                self.n_presses += 1
                return True
            if L.done(leg) and el > 0.5:
                self.n_auto_presses += 1
                self.last_miss = (label, self.tarsus(leg).round(3).tolist())
                return False
        return None

    def _step(self, t: float, dt: float) -> None:
        c = self.cfg
        el = t - self._t_phase
        ph = self.phase
        L = self.legs
        if ph == "setup":
            if el > 0.45:
                r, self._resume = self._resume, "wait_order"
                if r.endswith("_redo"):
                    self._redo(r)
                else:
                    self._go(r)
            return
        if ph == "wait_order":
            if self.queue:
                self.order = self.queue.pop(0)
                self.pattern = int(self.rng.integers(0, len(A.PATTERNS)))
                self._spill_at = (self.rng.choice(["shot", "pour"]) if self.rng.random() < c.spill_p else None)
                self._spill_done = False
                self.cup = self._free_cup()
                cp = self.cups[self.cup]
                cp["name"] = self.order
                cp["where"] = "moving"
                cp["level"] = 0.0
                cp["surf_mat"] = self.mat["cup_in"]
                self._path = (self.stack_spot(), self.cup_spot(), t, c.cup_slide_s, 0.6)
                self._caption_say(f"ORDER: {A.drink_of(self.order)} for {A.NAMES[self.order].upper()}")
                self._go("cup_in")
        elif ph == "cup_in":
            cp = self.cups[self.cup]
            cp["pos"] = self._along(t)
            if el >= c.cup_slide_s:
                cp["where"] = "tray"
                cp["pos"] = self.cup_spot()
                self._path = (self.pf_tamp(), self.pf_grinder(), t, c.pf_move_s, 0.35)
                self._go("pf_to_grinder")
        elif ph == "pf_to_grinder":
            self.pf_pos = self._along(t)
            if el >= c.pf_move_s:
                self._go("grind")
        elif ph == "grind":
            self.grounds = min(1.0, el / c.grind_s)
            self.tamp_level = 0.0
            if self.rng.random() < 0.7:
                ch = np.array([c.grinder_x, 0.8, 0.62])
                self._spawn("part", ch + np.r_[self.rng.normal(0, 0.03, 2), 0], np.array([0, 0, -0.3]), 0.8,
                            floor=self.pf_grinder()[2] + 0.06)
            if el >= c.grind_s:
                self._path = (self.pf_grinder(), self.pf_tamp(), t, c.pf_move_s, 0.35)
                self._go("pf_back")
        elif ph == "pf_back":
            self.pf_pos = self._along(t)
            if el >= c.pf_move_s:
                self.pf_pos = self.pf_tamp()
                self._n_tamp = 0
                self._place_puck()
                L.move("lf", np.array([c.tamp_x, c.tamp_y, self._puck_col_top() + 0.25]), 0.45)
                self._go("tamp_hover")
        elif ph == "tamp_hover":
            if L.done("lf"):
                L.move("lf", np.array([c.tamp_x, c.tamp_y, self._puck_col_top() - c.press]), 0.25)
                self._tamp_f = 0.0
                self._go("tamp_down")
        elif ph == "tamp_down":
            f = contact_force(self.sim.model, self.sim.data, self.puck_geom, self.lf_geoms)
            self._tamp_f = max(self._tamp_f, f)
            self.max_tamp_force = max(self.max_tamp_force, f)
            if L.done("lf") and el > 0.4:
                if self._tamp_f >= c.tamp_force:
                    self.n_tamps += 1
                    self._n_tamp += 1
                    self.tamp_level = min(1.0, self._n_tamp / c.n_tamps)
                    self._place_puck()
                else:
                    self._n_tamp_miss = getattr(self, "_n_tamp_miss", 0) + 1
                    self._n_tamp += 0.34  # a light press; another try
                L.move("lf", np.array([c.tamp_x, c.tamp_y, self._puck_col_top() + 0.2]), 0.22)
                self._go("tamp_up")
        elif ph == "tamp_up":
            if L.done("lf"):
                if self.tamp_level >= 1.0 or self._n_tamp >= c.n_tamps + 3:
                    self.tamp_level = 1.0
                    self._park_puck()
                    L.move("lf", None, 0.4)
                    self._path = (self.pf_tamp(), self.pf_group(), t, c.pf_move_s, 0.5)
                    self._go("pf_lock")
                else:
                    L.move("lf", np.array([c.tamp_x, c.tamp_y, self._puck_col_top() - c.press]), 0.25)
                    self._tamp_f = 0.0
                    self._go("tamp_down")
        elif ph == "pf_lock":
            self.pf_pos = self._along(t)
            u = smoothstep(el / c.pf_move_s)
            self.pf_quat = np.array(quat_axis_angle((0, 0, 1), -0.5 * max(0.0, u - 0.7) / 0.3))
            if el >= c.pf_move_s and L.done("lf"):
                self.pf_quat = np.array(quat_axis_angle((0, 0, 1), -0.5))
                self._btn_go("shot")
        elif ph.startswith("btn_"):
            self._btn_step(ph, el)
        elif ph == "shot":
            u = min(el / c.shot_s, 1.0)
            cp = self.cups[self.cup]
            cp["level"] = 0.32 * u
            cp["surf_mat"] = self.crema_mat[min(A.CREMA_STAGES - 1, int(u * A.CREMA_STAGES))]
            self.pressure = 9.0 * smoothstep(min(el / 0.6, 1.0)) * (1.0 if u < 0.95 else 0.3)
            self.lamp = "brew"
            if self._spill_at == "shot" and not self._spill_done and u > 0.5:
                self._spill()
            if el >= c.shot_s:
                self.pressure = 0.0
                self.n_shots += 1
                self.lamp = "ready"
                self._btn_go("steam")
        elif ph == "jug_to_wand":
            self.jug_pos = self._along(t)
            if el >= c.jug_move_s:
                self.jug_pos = self.jug_wand()
                self.foam = 0.0
                self._go("steam")
        elif ph == "steam":
            self.lamp = "steam"
            self.foam = min(1.0, el / c.steam_s)
            tip = np.array([c.wand_x, c.cup_y - 0.02, c.tray_z + 0.5])
            for _ in range(2):
                if self.rng.random() < 0.6:
                    self._spawn("steam", tip + np.r_[self.rng.normal(0, 0.05, 2), 0.05],
                                np.array([*self.rng.normal(0, 0.08, 2), self.rng.uniform(0.35, 0.7)]),
                                self.rng.uniform(0.9, 1.6))
            if el >= c.steam_s:
                self.lamp = "ready"
                self._path = (self.jug_wand(), self.jug_pour(), t, c.jug_move_s, 0.4)
                self._go("jug_to_pour")
        elif ph == "jug_to_pour":
            self.jug_pos = self._along(t)
            if el >= c.jug_move_s:
                self.jug_pos = self.jug_pour()
                L.move("rf", self._handle_world(0.0) + np.array([0, 0, 0.15]), 0.45)
                self._go("pour_reach")
        elif ph == "pour_reach":
            if L.done("rf"):
                L.move("rf", self._handle_world(0.0), 0.25)
                self._go("pour_grip")
        elif ph == "pour_grip":
            if L.done("rf"):
                self._grip_off = self.jug_pos - self.tarsus("rf")
                self._pour_err = []
                self._pour_k = -1
                self._go("pour")
        elif ph == "pour":
            self._pour_step(t, el)
        elif ph == "pour_back":
            self._jug_follow(max(0.0, 1.0 - el / 0.4) * self._tilt_end)
            if L.done("rf") and el >= 0.4:
                L.move("rf", self.tarsus("rf") + np.array([0, -0.1, 0.2]), 0.3)
                self._path = (self.jug_pos.copy(), np.array([*c.jug_home, 0.0]), t, c.jug_move_s, 0.3)
                self.jug_quat = IDENT.copy()
                self._go("jug_home")
        elif ph == "jug_home":
            self.jug_pos = self._along(t)
            if el >= c.jug_move_s:
                self.jug_pos = np.array([*c.jug_home, 0.0])
                self.foam = 0.0
                L.move("rf", np.array([c.bell_x, c.bell_y, self._bell_top() + 0.25]), 0.4)
                self._go("bell_hover")
        elif ph == "bell_hover":
            if L.done("rf"):
                L.move("rf", np.array([c.bell_x, c.bell_y, self._bell_top() - c.press]), 0.2)
                self._go("bell_down")
        elif ph == "bell_down":
            r = self._press("rf", self.bell_geom, None, "bell", el, "down")
            if r is not None:
                self.n_bells += 1
                self._t_ding = t
                nm = A.NAMES[self.order].upper()
                self._caption_say(f"DING!  {nm}, YOUR {A.drink_of(self.order)}!")
                L.move("rf", np.array([c.bell_x, c.bell_y, self._bell_top() + 0.25]), 0.2)
                self._go("bell_up")
        elif ph == "bell_up":
            if L.done("rf"):
                L.move("rf", None, 0.4)
                cp = self.cups[self.cup]
                if len(self.pickup) >= 3:
                    old = self.pickup.pop(0)
                    self.cups[old]["where"] = "parked"
                k = len(self.pickup)
                cp["where"] = "moving"
                self._path = (cp["pos"].copy(), self.pickup_spot(k), t, c.serve_s, 0.45)
                self._pf_from = self.pf_pos.copy()
                self._go("serve")
        elif ph == "serve":
            cp = self.cups[self.cup]
            cp["pos"] = self._along(t)
            # the portafilter unlocks and goes back to the tamp station (knocked out)
            u = smoothstep(min(el / c.serve_s, 1.0))
            self.pf_pos = self._pf_from + u * (self.pf_tamp() - self._pf_from) + np.array([0, 0, 0.3 * math.sin(math.pi * u)])
            self.pf_quat = np.array(quat_axis_angle((0, 0, 1), -0.5 * (1 - u)))
            if u > 0.5:
                self.grounds = 0.0
                self.tamp_level = 0.0
            if el >= c.serve_s:
                cp["where"] = "pickup"
                self.pickup.append(self.cup)
                cp["pos"] = self.pickup_spot(len(self.pickup) - 1)
                self.cup = None
                self.n_served += 1
                self.add_work(1)
                self.order = None
                self.pf_pos = self.pf_tamp()
                self.pf_quat = IDENT.copy()
                self._go("wait_order")

    # ---- buttons (SHOT / STEAM) by the left front leg ----------------------------------
    TIP = np.array([0.07, 0.0, 0.0])  # the tarsus5 origin sits this far behind the claws

    def _btn_go(self, which: str) -> None:
        p = self.btn_pos(which) - self.TIP
        self._btn = which
        self.legs.move("lf", p + np.array([0, 0, 0.25]), 0.4)
        self._go("btn_hover")

    def _btn_step(self, ph: str, el: float) -> None:
        c = self.cfg
        L = self.legs
        which = self._btn
        p = self.btn_pos(which) - self.TIP
        if ph == "btn_hover":
            if L.done("lf"):
                L.move("lf", p + np.array([0, 0, 0.02 - c.press]), 0.2)
                self._go("btn_down")
        elif ph == "btn_down":
            r = self._press("lf", self.btn_geom[which], p, which, el, "down")
            if r is not None:
                self._btn_t = self.t
                L.move("lf", p + np.array([0, 0, 0.25]), 0.2)
                self._go("btn_up")
        elif ph == "btn_up":
            if L.done("lf"):
                L.move("lf", None, 0.35)
                if which == "shot":
                    self._go("shot")
                else:
                    self._path = (self.jug_pos.copy(), self.jug_wand(), self.t, c.jug_move_s, 0.35)
                    self._go("jug_to_wand")

    # ---- the pour: the right leg holds the jug, the job tilts it -------------------------
    def _handle_world(self, tilt: float) -> np.ndarray:
        c = self.cfg
        R = _qmat(quat_axis_angle((1, 0, 0), -tilt))
        return self.jug_pos + R @ np.array([0.0, -1.5 * c.jug_r, 0.62 * c.jug_h])

    def _jug_follow(self, tilt: float) -> None:
        tip = self.tarsus("rf")
        q = np.array(quat_axis_angle((1, 0, 0), -tilt))
        R = _qmat(q)
        c = self.cfg
        h_local = np.array([0.0, -1.5 * c.jug_r, 0.62 * c.jug_h])
        self.jug_pos = tip - R @ h_local
        self.jug_quat = q

    def _pour_path(self, u: float, jitter: bool = False) -> np.ndarray:
        """The ideal handle point during the pour (world): forward toward the cup, a
        little down, with the rosetta's side-to-side wiggle. ``jitter``: plus this
        pour's unsteadiness (our model: a random amplitude per drink), which is what
        the leg is actually sent."""
        base = self._h0
        wig = 0.06 * math.sin(2 * math.pi * 3.0 * u) * (1.0 if A.PATTERNS[self.pattern] == "rosetta" else 0.35)
        p = base + np.array([wig, 0.12 * smoothstep(u), -0.08 * smoothstep(u)])
        if jitter:
            a, ph = self._jit
            p = p + a * np.array([math.sin(2 * math.pi * 7.0 * u + ph), math.cos(2 * math.pi * 5.0 * u + 2 * ph), 0.0])
        return p

    def _pour_step(self, t: float, el: float) -> None:
        c = self.cfg
        L = self.legs
        u = min(el / c.pour_s, 1.0)
        if self._pour_k < 0:
            self._h0 = self.tarsus("rf")
            self._pour_k = 0
            self._jit = (float(self.rng.uniform(0.0, c.jitter_mm)), float(self.rng.uniform(0, 2 * math.pi)))
        # re-target the leg every 0.1 s along the path (IK, eased)
        k = int(u * c.pour_s / 0.1)
        if k != self._pour_k and u < 1.0:
            self._pour_k = k
            L.move("rf", self._pour_path(min(1.0, (k + 1) * 0.1 / c.pour_s), jitter=True), 0.1)
        target = self._pour_path(u)
        err = float(np.linalg.norm(self.tarsus("rf") - target))
        if el > 0.2:
            self._pour_err.append(err)
            if len(self._pour_err) > 4000:
                self._pour_err = self._pour_err[-4000:]
        tilt = 1.25 * smoothstep(min(el / 0.7, 1.0)) + 0.25 * smoothstep(max(0.0, u - 0.6) / 0.4)
        self._jug_follow(tilt)
        self._tilt_end = tilt
        cp = self.cups[self.cup]
        cp["level"] = 0.32 + 0.6 * smoothstep(u)
        rms = float(np.sqrt(np.mean(np.square(self._pour_err)))) if self._pour_err else 0.0
        wob = 1 if (u > 0.5 and rms > c.wobble_mm) else 0
        stage = min(A.ART_STAGES - 1, int(u * A.ART_STAGES))
        cp["surf_mat"] = self.art_mat[self.pattern][wob][stage]
        cp["art"] = (self.pattern, wob)
        self.foam = max(0.0, 1.0 - u)
        if self._spill_at == "pour" and not self._spill_done and u > 0.6:
            self._spill()
        if el >= c.pour_s:
            self.pour_rms = rms
            score = float(np.clip(100.0 - 1400.0 * max(0.0, rms - 0.006), 40.0, 100.0))
            score = round(score - 5.0 * wob, 0)
            self.last_score = score
            self.score_sum += score
            self._n_art = getattr(self, "_n_art", 0) + 1
            self.scores = (self.scores + [score])[-20:]
            cp["surf_mat"] = self.art_mat[self.pattern][wob][A.ART_STAGES - 1]
            self._caption_say(f"LATTE ART: {A.PATTERNS[self.pattern].upper()}  {score:.0f}/100")
            L.move("rf", self.tarsus("rf") + np.array([0, -0.1, 0.05]), 0.4)
            self._go("pour_back")

    # ---- spills: a real free-body drop -------------------------------------------------
    def _spill(self) -> None:
        self._spill_done = True
        d = self.sim.data
        m = self.sim.model
        free = [dr for dr in self.drops if dr["state"] == "parked"]
        if not free:
            return
        dr = free[0]
        c = self.cfg
        a = float(self.rng.uniform(-2.4, -0.7))  # toward the fly / the camera side
        cup = self.cup_spot()
        rim = cup + np.array([(c.cup_r1 + 0.03) * math.cos(a), (c.cup_r1 + 0.03) * math.sin(a), c.cup_h + 0.03])
        d.qpos[dr["q"]:dr["q"] + 3] = rim
        d.qpos[dr["q"] + 3:dr["q"] + 7] = IDENT
        d.qvel[dr["v"]:dr["v"] + 6] = 0.0
        d.qvel[dr["v"]:dr["v"] + 3] = (12.0 * math.cos(a), 12.0 * math.sin(a), 6.0)
        m.body_gravcomp[dr["body"]] = 0.0
        m.geom_contype[dr["geom"]] = DROP_BIT
        m.geom_conaffinity[dr["geom"]] = DROP_BIT | TERRAIN_BIT
        dr["state"] = "live"
        dr["t"] = self.t
        self.n_spills += 1
        self._caption_say("OOPS - A DRIP!")

    def _park_drop(self, dr: dict) -> None:
        d, m = self.sim.data, self.sim.model
        d.qpos[dr["q"]:dr["q"] + 3] = dr["park"]
        d.qpos[dr["q"] + 3:dr["q"] + 7] = IDENT
        d.qvel[dr["v"]:dr["v"] + 6] = 0.0
        m.body_gravcomp[dr["body"]] = 1.0
        m.geom_contype[dr["geom"]] = 0
        m.geom_conaffinity[dr["geom"]] = 0
        dr["state"] = "parked"

    def _drops_step(self) -> None:
        d = self.sim.data
        for dr in self.drops:
            if dr["state"] == "parked":
                d.qvel[dr["v"]:dr["v"] + 6] = 0.0
                d.qpos[dr["q"]:dr["q"] + 3] = dr["park"]
                continue
            p = d.qpos[dr["q"]:dr["q"] + 3]
            v = d.qvel[dr["v"]:dr["v"] + 3]
            if not (np.all(np.isfinite(p)) and np.all(np.abs(p) < 50)):
                self._park_drop(dr)
                continue
            if self.t - dr["t"] > 0.4 and np.linalg.norm(v) < 2.0 or self.t - dr["t"] > 3.0:
                # it has landed: a puddle where it lies, the drop goes back to the pool
                pd = self.puddles[self._k_puddle]
                self._k_puddle = (self._k_puddle + 1) % len(self.puddles)
                z = float(p[2]) - 0.028
                d.mocap_pos[pd["mocap"]] = (float(p[0]), float(p[1]), max(z, 0.001))
                d.mocap_quat[pd["mocap"]] = quat_axis_angle((0, 0, 1), float(self.rng.uniform(0, math.pi)))
                pd["t"] = self.t
                self._park_drop(dr)

    # ------------------------------------------------------------ props
    def _place_puck(self) -> None:
        c = self.cfg
        self.sim.data.mocap_pos[self.puck_mocap] = (c.tamp_x, c.tamp_y, self._puck_col_top())

    def _puck_col_top(self) -> float:
        return self.puck_top() + self.cfg.tamper_t

    def _bell_top(self) -> float:
        c = self.cfg
        return 0.9 * c.bell_r + 0.05

    def _free_cup(self) -> int:
        for i, cp in enumerate(self.cups):
            if cp["where"] == "parked":
                return i
        i = self.pickup.pop(0)
        return i

    def _along(self, t: float) -> np.ndarray:
        p0, p1, t0, dur, lift = self._path
        u = smoothstep(min(max((t - t0) / dur, 0.0), 1.0))
        return p0 + u * (p1 - p0) + np.array([0.0, 0.0, lift * math.sin(math.pi * u)])

    def _props_step(self, t: float, dt: float) -> None:
        c = self.cfg
        m, d = self.sim.model, self.sim.data
        # portafilter + grounds
        d.mocap_pos[self.pf_mocap] = self.pf_pos
        d.mocap_quat[self.pf_mocap] = self.pf_quat
        hz = self._grounds_hz()
        m.geom_size[self.grounds_geom, 2] = hz
        m.geom_pos[self.grounds_geom, 2] = 0.03 + (hz - 0.02) * 0.6
        m.geom_rbound[self.grounds_geom] = c.basket_r
        # tamper on the left front tarsus
        d.mocap_pos[self.tamper_mocap] = self.tarsus("lf")
        # buttons dip while pressed, the bell's plunger
        for w in ("shot", "steam"):
            down = 0.012 if (self.phase in ("btn_down", "btn_up") and self._btn_is(w)
                             and t - getattr(self, "_btn_t", -1) < 0.3) else 0.0
            d.mocap_pos[self.btn_mocap[w]] = self.btn_pos(w) - np.array([0, 0, down])
        ding = t - getattr(self, "_t_ding", -1e9)
        d.mocap_pos[self.bell_mocap] = (c.bell_x, c.bell_y, -0.012 if ding < 0.15 else 0.0)
        d.mocap_quat[self.bell_mocap] = quat_axis_angle((1, 0, 0), 0.06 * math.sin(60 * ding) * math.exp(-ding / 0.3)
                                                        if ding < 1.0 else 0.0)
        # the jug and its milk
        d.mocap_pos[self.jug_mocap] = self.jug_pos
        d.mocap_quat[self.jug_mocap] = self.jug_quat
        m.geom_pos[self.milk_geom, 2] = c.jug_h * (0.35 + 0.3 * self.foam)
        # cups
        for cp in self.cups:
            if cp["where"] == "parked":
                d.mocap_pos[cp["mocap"]] = (0.0, -40.0, -30.0)
                continue
            d.mocap_pos[cp["mocap"]] = cp["pos"]
            m.geom_matid[cp["sleeve"]] = self.sleeve_mat[cp["name"]]
            m.geom_matid[cp["surf"]] = cp["surf_mat"]
            m.geom_pos[cp["surf"], 2] = 0.035 + cp["level"] * (c.cup_h - 0.06)
        # the gauge needle, the lamps
        ang = math.radians(180 + 12 * self.pressure)
        d.mocap_quat[self.needle_mocap] = quat_axis_angle((0, 1, 0), -ang)
        for nm, g in self.lamp_geom.items():
            on = {"ready": "lamp_green", "brew": "lamp_amber", "steam": "lamp_blue"}[nm]
            m.geom_matid[g] = self.mat[on] if self.lamp == nm else self.mat["lamp_off"]
        # the streams
        pf_on = self.phase == "shot" and (t - self._t_phase) > 0.4 and (t - self._t_phase) < c.shot_s - 0.15
        cup = self.cup_spot()
        if pf_on and self.cup is not None:
            lvl = c.tray_z + 0.035 + self.cups[self.cup]["level"] * (c.cup_h - 0.06)
            for k, s in enumerate((-1, 1)):
                top = self.pf_pos + np.array([0.06 * s, 0, -0.08])
                bot = np.array([cup[0] + 0.02 * s, cup[1], lvl])
                L = max((top[2] - bot[2]) / 2, 0.01)
                m.geom_size[self.stream_geom[k], 1] = L
                m.geom_rbound[self.stream_geom[k]] = L + 0.02
                d.mocap_pos[self.stream_mocap[k]] = 0.5 * (top + bot)
                d.mocap_quat[self.stream_mocap[k]] = _quat_from_z(top - bot)
                if self.rng.random() < 0.3:
                    self._spawn("part", top, np.array([0, 0, -1.0]), 1.0, floor=lvl)
        else:
            for k in range(2):
                d.mocap_pos[self.stream_mocap[k]] = (0, 0, -30.0)
        if self.phase == "pour" and self.cup is not None:
            R = _qmat(self.jug_quat)
            sp = self.jug_pos + R @ np.array([0.0, 1.2 * c.jug_r, 1.0 * c.jug_h])
            lvl = c.tray_z + 0.035 + self.cups[self.cup]["level"] * (c.cup_h - 0.06)
            bot = np.array([cup[0], cup[1] - 0.02, lvl])
            if sp[2] > bot[2] + 0.01:
                L = float(np.linalg.norm(sp - bot)) / 2
                m.geom_size[self.milkstream_geom, 1] = L
                m.geom_rbound[self.milkstream_geom] = L + 0.03
                d.mocap_pos[self.milkstream_mocap] = 0.5 * (sp + bot)
                d.mocap_quat[self.milkstream_mocap] = _quat_from_z(sp - bot)
            else:
                d.mocap_pos[self.milkstream_mocap] = (0, 0, -30.0)
        else:
            d.mocap_pos[self.milkstream_mocap] = (0, 0, -30.0)
        # the ticket rail: the queue; the order being made clipped on the machine
        for k in range(c.queue_max):
            if k < len(self.queue):
                d.mocap_pos[self.ticket_mocap[k]] = self._ticket_home[k] + np.array(
                    [0, 0, 0.02 * math.sin(2.0 * t + k)])
                m.geom_matid[self.ticket_geom[k]] = self.ticket_mat[self.queue[k]]
            else:
                d.mocap_pos[self.ticket_mocap[k]] = (0, 0, -30.0)
        if self.order is not None:
            d.mocap_pos[self.now_mocap] = (c.mach_x0 + 0.35, c.mach_y - 0.05, 0.75)
            m.geom_matid[self.now_geom] = self.ticket_mat[self.order]
        else:
            d.mocap_pos[self.now_mocap] = (0, 0, -30.0)
        # puddles fade over 25 s
        for pd in self.puddles:
            age = t - pd["t"]
            m.geom_rgba[pd["geom"], 3] = 0.85 * max(0.0, 1.0 - age / 25.0)
            if age > 25.0:
                d.mocap_pos[pd["mocap"]] = (0, -30.0, -30.0)
        self._drops_step()
        self._particles_step(dt)

    def _btn_is(self, w: str) -> bool:
        return getattr(self, "_btn", None) == w

    def _spawn(self, kind: str, pos, vel, life: float, floor: float = -1.0) -> None:
        P_ = self.parts[kind]
        k = P_["k"]
        P_["k"] = (k + 1) % len(P_["life"])
        P_["pos"][k] = pos
        P_["vel"][k] = vel
        P_["life"][k] = life
        P_["age"][k] = 0.0
        P_["floor"][k] = floor

    def _particles_step(self, dt: float) -> None:
        c = self.cfg
        m, d = self.sim.model, self.sim.data
        for kind, P_ in self.parts.items():
            alive = P_["life"] > 0
            if not alive.any():
                d.mocap_pos[P_["mocap"]] = (0.0, -30.0, -30.0)
                continue
            P_["age"][alive] += dt
            if kind == "steam":
                P_["vel"][alive] *= 0.985
                P_["pos"][alive] += P_["vel"][alive] * dt
                a = np.clip(P_["age"] / np.maximum(P_["life"], 1e-6), 0, 1)
                size = P_["r0"] * (0.5 + 2.2 * a)
            else:
                P_["vel"][alive, 2] -= c.particle_g * dt
                P_["pos"][alive] += P_["vel"][alive] * dt
                P_["life"][alive & (P_["pos"][:, 2] < P_["floor"])] = 0.0
                size = np.full(len(P_["life"]), P_["r0"])
            P_["life"][P_["age"] >= P_["life"]] = 0.0
            alive = P_["life"] > 0
            d.mocap_pos[P_["mocap"]] = np.where(alive[:, None], P_["pos"], np.array([0.0, -30.0, -30.0]))
            m.geom_size[P_["geom"], 0] = np.maximum(size, 1e-4)
            m.geom_rbound[P_["geom"]] = np.maximum(size, 1e-4)

    # ------------------------------------------------------------ reset
    def on_reset(self) -> None:
        # an explicit reset (the fly fell): the drink in progress carries on; a leg
        # move that was interrupted (a tamp, a button, the pour, the bell) is redone,
        # the props' timed steps resume
        for dr in self.drops:
            self._park_drop(dr)
        ph = self.phase
        if ph.startswith("tamp"):
            r = "tamp_redo"
        elif ph.startswith("btn_"):
            r = "btn_redo"
        elif ph in ("pour_reach", "pour_grip", "pour"):
            r = "pour_redo"
        elif ph == "pour_back":
            r = "jughome_redo"
        elif ph.startswith("bell"):
            r = "bell_redo"
        elif ph == "setup":
            r = self._resume
        else:
            r = ph
        self._begin()
        self._resume = r

    def reset_props(self) -> None:
        self._props_step(self.t, 0.0)

    def _redo(self, what: str) -> None:
        """Entry points after a reset (see ``on_reset``)."""
        c = self.cfg
        L = self.legs
        if what == "tamp_redo":
            self._n_tamp = int(round(self.tamp_level * c.n_tamps))
            self._place_puck()
            L.move("lf", np.array([c.tamp_x, c.tamp_y, self._puck_col_top() + 0.25]), 0.45)
            self._go("tamp_hover")
        elif what == "btn_redo":
            self._btn_go(self._btn)
        elif what == "pour_redo":
            self.jug_pos = self.jug_pour()
            self.jug_quat = IDENT.copy()
            L.move("rf", self._handle_world(0.0) + np.array([0, 0, 0.15]), 0.45)
            self._go("pour_reach")
        elif what == "jughome_redo":
            self.jug_quat = IDENT.copy()
            self._path = (self.jug_pos.copy(), np.array([*c.jug_home, 0.0]), self.t, c.jug_move_s, 0.3)
            self._go("jug_home")
        elif what == "bell_redo":
            L.move("rf", np.array([c.bell_x, c.bell_y, self._bell_top() + 0.25]), 0.4)
            self._go("bell_hover")
        else:
            self._go("wait_order")

    # ------------------------------------------------------------ presentation
    def post_process(self, frame: np.ndarray, t: float) -> np.ndarray:
        cap, t_cap = self._caption
        if not (self.cfg.captions and cap and 0 <= self.t - t_cap < 2.0):
            return frame
        return caption_overlay(frame, cap, rgb=(255, 232, 190), shadow=(50, 25, 10))

    def _shot(self) -> str:
        if not self.cfg.close_ups:
            return "wide"
        ph = self.phase
        if ph.startswith("tamp") or ph == "pf_back":
            want = "tamp"
        elif ph in ("shot",) or (ph.startswith("btn") and getattr(self, "_btn", "") == "shot"):
            want = "shot"
        elif ph.startswith("pour"):
            want = "pour"
        elif ph in ("steam", "jug_to_wand"):
            want = "steam"
        else:
            want = "wide"
        return cam_shot(self, want, self.t)

    def camera_target(self) -> np.ndarray:
        c = self.cfg
        return {"wide": np.array([1.9, 0.5, 0.7]), "tamp": np.array([1.8, 0.2, 0.25]),
                "shot": np.array([2.0, 0.9, 0.5]), "pour": np.array([2.1, 0.85, 0.5]),
                "steam": np.array([2.6, 0.9, 0.6])}[self._shot()]

    def camera_preset(self) -> CameraPreset:
        shot = self._shot()
        if shot == "tamp":
            return CameraPreset(azimuth=118.0, elevation=-34.0, distance=4.0, tau_s=0.3)
        if shot == "shot":
            return CameraPreset(azimuth=100.0, elevation=-14.0, distance=4.0, tau_s=0.3)
        if shot == "pour":
            return CameraPreset(azimuth=105.0, elevation=-48.0, distance=3.4, tau_s=0.3)
        if shot == "steam":
            return CameraPreset(azimuth=95.0, elevation=-16.0, distance=4.6, tau_s=0.3)
        return CameraPreset(azimuth=100.0, elevation=-20.0, distance=7.0, tau_s=0.3)

    def job_hud_lines(self) -> list[str]:
        nm = f"{A.drink_of(self.order)} for {A.NAMES[self.order]}" if self.order is not None else "waiting for an order"
        sc = f"{self.last_score:.0f}" if self.last_score is not None else "-"
        return [
            f"now: {nm}   [{self.phase}]   queue {len(self.queue)} (walk-outs {self.n_walkouts})",
            f"shots pulled {self.n_shots}   latte art {sc}/100 (mean {self.mean_art():.0f})   tamps {self.n_tamps}"
            f"   presses {self.n_presses} (auto {self.n_auto_presses})   spills {self.n_spills}",
            "(tamps / buttons / bell: real leg contact; slides, streams, steam, latte art texture: kinematic / "
            "visual; spilled drops: real physics)",
        ] + ([self.message] if self.message else [])

    def mean_art(self) -> float:
        n = getattr(self, "_n_art", 0)
        return self.score_sum / n if n else 0.0

    def job_stats(self) -> dict[str, Any]:
        return {"drinks_served": self.n_served, "shots_pulled": self.n_shots, "tamps": self.n_tamps,
                "latte_art_last": self.last_score, "latte_art_mean": round(self.mean_art(), 1),
                "orders_in_queue": len(self.queue), "orders": self.n_orders, "walkouts": self.n_walkouts,
                "presses": self.n_presses, "auto_presses": self.n_auto_presses, "bells": self.n_bells,
                "spills": self.n_spills, "pour_rms_mm": round(self.pour_rms, 4),
                "max_tamp_force_uN": round(self.max_tamp_force, 2), "phase": self.phase,
                "ik_err_mm": round(self.legs.max_err, 3) if hasattr(self, "legs") else 0.0}

    def _caption_say(self, text: str) -> None:
        self._caption = (text, self.t)
        self.message = text
