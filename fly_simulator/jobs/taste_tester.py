"""Taste tester: the fly tastes food samples on a conveyor belt forever, and the
connectome brain decides: approve or reject.

A fly-scale quality-control station. An indexing conveyor belt brings one sample
at a time (a small dish with a drop and a sample card) to the fly, who stands at the
belt (a stance action: all six legs planted, then one front leg lifted). Flies taste
with their legs first (tarsal gustatory neurons), and sugar on the tarsi triggers the
proboscis extension response (PER). So:

1. **Tap.** The left front leg reaches over the belt edge and touches the drop
   (joint-space targets from damped least-squares IK on a scratch MjData; the leg
   is position-controlled like every other action, nothing is teleported).
2. **Taste -> brain.** On contact the job sends a taste stimulus to the brain:
   sugar / bitter at the sample's rates, the same stand-in sets as the taste patches
   (docs/TASTE.md: Shiu et al.'s labellar sugar GRNs LB3 and bitter GRNs LB1; the
   model has no tarsal taste neurons, so this is a labelled stand-in). Water drives
   no neurons (water GRNs are not in the stand-in sets).
3. **Decision.** With ``--brain`` the decision is the connectome's: the mean MN9
   (proboscis motor neuron) rate over the brain windows of the tasting pulse, against
   ``approve_mn9_hz`` (30 Hz, the taste patches' feeding threshold). Without a brain
   a clearly labelled **scripted** rule decides (sugar without bitter passes).
4. **Response.** APPROVED: the proboscis extends (with a brain it follows MN9 the
   whole time; the approval adds a held PER beat), the green lamp lights. REJECTED:
   the red lamp lights and the leg pushes the dish away (the dish follows the leg
   kinematically: engineered).
5. Downstream a kinematic **stamper** stamps the sample card APPROVED / REJECTED, and
   at the belt end an overhead **diverter paddle** sweeps approved samples into the
   green bin (far side) and rejected ones into the red bin (the fly's side). A fixed pool of samples is recycled
   (the oldest from the bins, washed and refilled with a new random sample under the
   "SAMPLES IN" hood: a hidden teleport, counted).

Everything but the floor is visual (no contacts), so the fly cannot trip over the
machines. Constant memory: pools, counters and fixed deques only; the brain's
stimulus log is trimmed.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import mujoco as mj
import numpy as np

from fly_simulator.actions.base import Action, ActionCommand, smoothstep
from fly_simulator.jobs import kebab_assets as KA
from fly_simulator.jobs import taste_tester_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import add_box, contact_kwargs, quat_axis_angle, quat_mul
from fly_simulator.jobs.registry import register_job

P = "taste/"
N_STATIONS = 6  # 0 hidden under the hood, 1-2 queue, 3 tasting, 4 stamper, 5 belt end
TASTE_K, STAMP_K, END_K = 3, 4, 5

# 7-segment digit layout: (dy, dz, horizontal?) in digit units (width 1, height 2)
SEGS = {"a": (0.0, 1.0, True), "b": (0.5, 0.5, False), "c": (0.5, -0.5, False),
        "d": (0.0, -1.0, True), "e": (-0.5, -0.5, False), "f": (-0.5, 0.5, False),
        "g": (0.0, 0.0, True)}
DIGITS = {0: "abcdef", 1: "bc", 2: "abged", 3: "abgcd", 4: "fgbc", 5: "afgcd", 6: "afgedc",
          7: "abc", 8: "abcdefg", 9: "abcdfg"}


# ---------------------------------------------------------------------------
# the stance: all legs planted, one front leg driven by the job
# ---------------------------------------------------------------------------


class TasterStance(Action):
    """Stand still with all tarsi adhering (like ``freeze``), while the job moves one
    front leg: ``leg_pose`` (7 joint targets in controller order) blended in with
    ``leg_w`` (0..1). The lifted leg's adhesion is off."""

    name = "taste_stance"
    blend_in = 0.25
    blend_out = 0.3

    def __init__(self, leg: str = "lf", duration: float = 3600.0) -> None:
        super().__init__(duration)
        self.leg = leg
        self.leg_pose: np.ndarray | None = None
        self.leg_w = 0.0

    def begin(self, mgr) -> None:
        from fly_simulator.actions.base import LEGS

        b = mgr.body
        self._start = mgr.sim.data.ctrl[b.pos_ids].copy()
        self._stand = b.stand.copy()
        self.cols = np.flatnonzero(b.leg_mask([self.leg]))
        self._li = LEGS.index(self.leg)
        self._on = np.ones(6)

    def command(self, mgr, t: float) -> ActionCommand:
        a = smoothstep(t / 0.2)
        tg = (1 - a) * self._start + a * self._stand
        adh = self._on.copy()
        w = float(min(max(self.leg_w, 0.0), 1.0))
        if self.leg_pose is not None and w > 0:
            tg[self.cols] = (1 - w) * tg[self.cols] + w * self.leg_pose
            if w > 0.05:
                adh[self._li] = 0.0
        return ActionCommand(targets=tg, adhesion=adh)

    def end(self, mgr, cancelled: bool) -> None:
        self.info = {"cancelled": cancelled, "tilt_deg": mgr.sim.tilt_deg()}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class TasteTesterConfig(JobConfig):
    # ---- station (thorax at rest; FlyGym spawns it here facing +x) -------------------
    station_x: float = 0.61
    station_y: float = 0.0
    # ---- belt (runs along -y in front of the fly) -----------------------------------
    belt_gap: float = 0.95  # thorax -> near belt edge (the front tarsi stand at +0.78)
    belt_width: float = 1.5
    belt_top: float = 0.24
    pitch: float = 2.0  # station spacing (indexing conveyor)
    move_s: float = 1.1  # one index move
    n_cleats: int = 22
    # ---- samples ----------------------------------------------------------------------
    n_samples: int = 12  # fixed pool (6 belt stations + 6 bin slots)
    cup_dx: float = 0.38  # dish centre from the near belt edge
    cup_r: float = 0.3
    cup_h: float = 0.09
    drop_r: float = 0.23
    drop_h: float = 0.12
    card_dx: float = 0.57  # card centre from the dish centre (+x)
    kind_weights: tuple = (0.4, 0.25, 0.2, 0.15)  # sugar, bitter, mixed, water
    sugar_hz: tuple = (50.0, 200.0)  # sugar GRN drive of a sugar sample (concentration)
    bitter_hz: tuple = (100.0, 200.0)
    mixed_sugar_hz: tuple = (100.0, 200.0)
    mixed_bitter_hz: tuple = (50.0, 200.0)
    # ---- tasting (leg tap) and the decision --------------------------------------------
    tap_leg: str = "lf"
    tap_dy: float = 0.1  # touch point: drop centre + this in y (the leg's side)
    lift_s: float = 0.3
    reach_s: float = 0.3
    touch_tol: float = 0.12  # tarsus tip within drop_r + this (xy) and this above the top
    touch_grace_s: float = 0.3  # no contact this long after the reach -> tasted anyway (miss)
    taste_s: float = 0.5  # stimulus duration
    approve_mn9_hz: float = 30.0  # docs/TASTE.md feeding threshold
    mn9_lag_s: float = 0.1  # brain windows ending in (t0 + lag, t0 + taste_s + lag] count
    brain_wait_s: float = 4.0  # max sim s to wait for those windows
    per_s: float = 0.9  # approve: held proboscis extension beat
    push_s: float = 0.55  # reject: push the dish away with the leg
    push_mm: float = 0.3
    retract_s: float = 0.35
    pause_s: float = 0.15
    mn9_ref_hz: float = 60.0  # MN9 rate that extends the proboscis fully
    gauge_max_hz: float = 100.0
    # ---- look ----------------------------------------------------------------------------
    shadows: bool = True
    stuck_timeout_s: float = 60.0


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------


@dataclass
class _Sample:
    kind: str = "sugar"
    sugar_hz: float = 0.0
    bitter_hz: float = 0.0
    serial: int = 0
    where: str = "spare"  # spare | belt | flying | bin
    dx: float = 0.0  # pushed away (mm, +x)
    decision: str = ""  # "" | approved | rejected
    stamped: bool = False
    bin: str = ""
    slot: int = -1
    fly: dict = field(default_factory=dict)  # bin flight (t0, p0, p1, q0, q1)


def _slerp_quat(q0, q1, u: float) -> np.ndarray:
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if float(q0 @ q1) < 0:
        q1 = -q1
    q = (1 - u) * q0 + u * q1
    return q / np.linalg.norm(q)


@register_job
class TasteTesterJob(EternalJob):
    name = "taste_tester"
    #: depth precision (shadow maps span znear..zfar; EternalJob applies it after compile)
    znear = 0.05
    title = "TASTE TESTER FLY"
    tagline = "quality control, one drop at a time, forever"
    work_label = "samples tasted"
    config_cls = TasteTesterConfig
    required_names = (P + "belt", P + "sample0")

    cfg: TasteTesterConfig

    # ------------------------------------------------------------ geometry
    @property
    def x_near(self) -> float:
        return self.cfg.station_x + self.cfg.belt_gap

    @property
    def x_far(self) -> float:
        return self.x_near + self.cfg.belt_width

    @property
    def x_cup(self) -> float:
        return self.x_near + self.cfg.cup_dx

    def station_y(self, k: int) -> float:
        return self.cfg.station_y + (TASTE_K - k) * self.cfg.pitch

    @property
    def y_top(self) -> float:  # upstream roller
        return self.station_y(0) + 0.5

    @property
    def y_end(self) -> float:  # downstream roller
        return self.station_y(END_K) - 0.6

    @property
    def drop_top(self) -> float:
        return self.cfg.belt_top + 0.02 + self.cfg.drop_h

    def configure_app(self, app_cfg) -> None:
        super().configure_app(app_cfg)
        app_cfg.controller.target_heading_deg = 0.0
        app_cfg.fly.extra_joints = True  # proboscis joints (PER)

    # ------------------------------------------------------------ build
    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        vis = contact_kwargs("visual")
        vm = dict(vis, mass=0.0)
        self._add_materials(spec)
        xn, xf, bt = self.x_near, self.x_far, c.belt_top
        xc = 0.5 * (xn + xf)
        y0, y1 = self.y_end, self.y_top
        yc, hl = 0.5 * (y0 + y1), 0.5 * (y1 - y0)
        # ---- conveyor: rubber belt, frame, legs, rollers, cleats
        add_box(wb, P + "belt", (0.5 * c.belt_width, hl, 0.02), (xc, yc, bt - 0.02),
                material=P + "belt", collide="visual")
        for side, x in (("near", xn - 0.03), ("far", xf + 0.03)):
            add_box(wb, f"{P}rail_{side}", (0.03, hl + 0.15, 0.035), (x, yc, bt + 0.01),
                    material=P + "steel", collide="visual")
            add_box(wb, f"{P}skirt_{side}", (0.02, hl + 0.1, 0.5 * (bt - 0.06)),
                    (x, yc, 0.5 * (bt - 0.06) + 0.02), material=P + "frame", collide="visual")
        for k, y in enumerate(np.linspace(y0 + 0.3, y1 - 0.3, 5)):
            for x in (xn + 0.05, xf - 0.05):
                wb.add_geom(name=f"{P}leg{k}_{x:.1f}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                            size=(0.04, 0.5 * bt, 0), pos=(x, y, 0.5 * bt), material=P + "frame",
                            **vis)
        rr = 0.12
        A_roll = A.roller_mesh(rr, 0.5 * c.belt_width + 0.02)
        KA.add_mesh(spec, P + "roller_mesh", A_roll)
        self._roller_names = []
        for nm, y in (("roller_top", y1), ("roller_end", y0)):
            b = wb.add_body(name=P + nm, pos=(xc, y, bt - rr), mocap=True)
            b.add_geom(name=P + nm + "_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "roller_mesh",
                       material=P + "roller", **vm)
            self._roller_names.append(P + nm)
        self._cleat_names = []
        for i in range(c.n_cleats):
            b = wb.add_body(name=f"{P}cleat{i}", pos=(xc, y1 - i * 0.6, bt + 0.004), mocap=True)
            b.add_geom(name=f"{P}cleat{i}_g", type=mj.mjtGeom.mjGEOM_BOX,
                       size=(0.5 * c.belt_width - 0.04, 0.018, 0.008), material=P + "cleat", **vm)
            self._cleat_names.append(f"{P}cleat{i}")
        # ---- the "SAMPLES IN" hood over the upstream end (hides the recycling)
        hy0 = self.station_y(0) - c.pitch / 2 + 0.25
        hood_len = y1 + 0.5 - hy0
        hw, hyc = 0.5 * c.belt_width + 0.12, hy0 + 0.5 * hood_len
        add_box(wb, P + "hood", (hw, 0.5 * hood_len, 0.03), (xc, hyc, bt + 0.8),
                material=P + "hood", collide="visual")
        for sgn in (-1, 1):
            add_box(wb, f"{P}hood_side{sgn}", (0.03, 0.5 * hood_len, 0.41),
                    (xc + sgn * (hw - 0.03), hyc, bt + 0.41), material=P + "hood", collide="visual")
        add_box(wb, P + "hood_back", (hw, 0.03, 0.41), (xc, hy0 + hood_len - 0.03, bt + 0.41),
                material=P + "hood", collide="visual")
        KA.add_mesh(spec, P + "hood_label_mesh", A.front_panel_mesh(0.5, 0.2, 0.01))
        wb.add_geom(name=P + "hood_label", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "hood_label_mesh",
                    pos=(xf + 0.13, hy0 + 0.5 * hood_len, bt + 0.45), material=P + "sign_in", **vm)
        # strip curtain at the hood mouth
        for j in range(6):
            yj = hy0 - 0.005
            xj = xn + (j + 0.5) * c.belt_width / 6
            add_box(wb, f"{P}curtain{j}", (0.5 * c.belt_width / 6 - 0.008, 0.004, 0.385),
                    (xj, yj, bt + 0.39), material=P + "curtain", collide="visual")
        # ---- samples: a pool of mocap bodies (dish + drop + card)
        KA.add_mesh(spec, P + "dish_mesh", A.dish_mesh(c.cup_r, c.cup_h))
        for v in range(3):
            KA.add_mesh(spec, f"{P}drop_mesh{v}", A.drop_mesh(c.drop_r, c.drop_h, seed=v))
        KA.add_mesh(spec, P + "card_mesh", A.flat_quad_mesh(0.2, 0.28, 0.01))
        self._sample_names = []
        for i in range(c.n_samples):
            b = wb.add_body(name=f"{P}sample{i}", pos=(xf + 3.0 + 0.8 * i, -9.0, -1.0), mocap=True)
            b.add_geom(name=f"{P}sample{i}_dish", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "dish_mesh",
                       material=P + "glass", **vm)
            b.add_geom(name=f"{P}sample{i}_drop", type=mj.mjtGeom.mjGEOM_MESH,
                       meshname=f"{P}drop_mesh{i % 3}", pos=(0, 0, 0.02), material=P + "drop_sugar", **vm)
            b.add_geom(name=f"{P}sample{i}_card", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "card_mesh",
                       pos=(c.card_dx, 0, 0.008), quat=quat_axis_angle((0, 0, 1), 0.04 * (i % 3 - 1)),
                       material=P + "card_sugar_blank", **vm)
            self._sample_names.append(f"{P}sample{i}")
        # ---- the stamper over station 4 (kinematic machine)
        ys = self.station_y(STAMP_K)
        xcard = self.x_cup + c.card_dx
        # a compact low stamper: housing z 0.78-1.14, stamps plunge 0.35 mm onto the card
        add_box(wb, P + "stamper_housing", (0.22, 0.86, 0.18), (xcard, ys, 0.96),
                material=P + "machine", collide="visual")
        xp = xn - 0.3  # the post stands on the fly's side of the belt
        add_box(wb, P + "stamper_arm", (0.5 * (xcard - xp), 0.08, 0.05), (0.5 * (xcard + xp), ys, 1.05),
                material=P + "machine", collide="visual")
        wb.add_geom(name=P + "stamper_post", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.05, 0.55, 0),
                    pos=(xp, ys, 0.55), material=P + "frame", **vis)
        add_box(wb, P + "stamper_lamp_ok", (0.01, 0.12, 0.04), (xcard + 0.225, ys - 0.3, 1.0),
                material=P + "ink_green", collide="visual")
        add_box(wb, P + "stamper_lamp_no", (0.01, 0.12, 0.04), (xcard + 0.225, ys + 0.3, 1.0),
                material=P + "ink_red", collide="visual")
        self._stamp_names = []
        for nm, ink, dy in (("stamp_ok", P + "ink_green", -0.3), ("stamp_no", P + "ink_red", 0.3)):
            b = wb.add_body(name=P + nm, pos=(xcard, ys + dy, 0.63), mocap=True)
            b.add_geom(name=P + nm + "_rod", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.03, 0.2, 0),
                       pos=(0, 0, 0.25), material=P + "steel", **vm)
            b.add_geom(name=P + nm + "_block", type=mj.mjtGeom.mjGEOM_BOX, size=(0.17, 0.24, 0.06),
                       pos=(0, 0, 0.08), material=ink.replace("ink", "stamp"), **vm)
            b.add_geom(name=P + nm + "_pad", type=mj.mjtGeom.mjGEOM_BOX, size=(0.16, 0.23, 0.012),
                       pos=(0, 0, 0.012), material=ink, **vm)
            self._stamp_names.append(P + nm)
        # ---- bins beside the belt end: approved (green) on the far side, rejected (red)
        # on the fly's side; an overhead diverter paddle pushes each sample into its bin
        ye = self.station_y(END_K)
        self._bins = {
            "approved": (np.array([xf + 0.75, ye]), (0.6, 0.6, 0.3)),
            "rejected": (np.array([xn - 0.75, ye]), (0.6, 0.6, 0.3)),
        }
        for nm, (ctr, (hx, hy, hz)) in self._bins.items():
            mat = P + ("bin_green" if nm == "approved" else "bin_red")
            t = 0.03
            bx, by = float(ctr[0]), float(ctr[1])
            add_box(wb, f"{P}bin_{nm}_floor", (hx, hy, 0.015), (bx, by, 0.015), material=mat,
                    collide="visual")
            for s in (-1, 1):
                add_box(wb, f"{P}bin_{nm}_wx{s}", (t, hy, hz), (bx + s * (hx - t), by, hz),
                        material=mat, collide="visual")
                add_box(wb, f"{P}bin_{nm}_wy{s}", (hx, t, hz), (bx, by + s * (hy - t), hz),
                        material=mat, collide="visual")
            # labels on the +x face (toward the camera) and the +y face (toward the station)
            KA.add_mesh(spec, f"{P}bin_{nm}_label_mesh", A.front_panel_mesh(0.95 * hy, 0.6 * hz))
            for face, pos, q in (("x", (bx + hx + 0.008, by, hz), (1.0, 0.0, 0.0, 0.0)),
                                 ("y", (bx, by + hy + 0.008, hz),
                                  quat_axis_angle((0, 0, 1), math.pi / 2))):
                wb.add_geom(name=f"{P}bin_{nm}_label_{face}", type=mj.mjtGeom.mjGEOM_MESH,
                            meshname=f"{P}bin_{nm}_label_mesh", pos=pos, quat=q,
                            material=f"{P}label_{nm}", **vm)
        # diverter gantry: a beam across the belt end, posts beyond the bins; the paddle
        # hangs from a carriage on the beam (kinematic machine)
        bx0, bx1, bz = xn - 1.55, xf + 1.55, 1.0
        add_box(wb, P + "diverter_beam", (0.5 * (bx1 - bx0), 0.06, 0.05), (0.5 * (bx0 + bx1), ye, bz),
                material=P + "machine", collide="visual")
        for j, x in enumerate((bx0, bx1)):
            wb.add_geom(name=f"{P}diverter_post{j}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                        size=(0.05, 0.5 * bz, 0), pos=(x, ye, 0.5 * bz), material=P + "frame", **vis)
        b = wb.add_body(name=P + "paddle", pos=(xc, ye, 0.8), mocap=True)
        b.add_geom(name=P + "paddle_plate", type=mj.mjtGeom.mjGEOM_BOX, size=(0.02, 0.4, 0.1),
                   material=P + "machine", **vm)
        b.add_geom(name=P + "paddle_rod", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.025, 0.25, 0),
                   pos=(0, 0, 0.3), material=P + "steel", **vm)
        b.add_geom(name=P + "paddle_carriage", type=mj.mjtGeom.mjGEOM_BOX, size=(0.1, 0.09, 0.06),
                   pos=(0, 0, bz - 0.8), material=P + "frame", **vm)
        self._add_room(spec)

    def _add_materials(self, spec) -> None:
        c = self.cfg
        mat = spec.material("grid")
        if mat is not None:
            KA.add_texture(spec, P + "tex_floor", A.lab_floor_texture(c.seed))
            mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = P + "tex_floor"
            mat.rgba = (1.0, 1.0, 1.0, 1.0)
            mat.reflectance = 0.06
            mat.texrepeat = [v * 1.5 for v in mat.texrepeat]
        KA.add_texture(spec, P + "tex_steel", KA.steel_texture(c.seed))
        for nm, rgba, kw in (("steel", (1, 1, 1, 1), dict(specular=0.8, shininess=0.8)),
                             ("frame", (0.72, 0.74, 0.78, 1), dict(specular=0.5, shininess=0.6)),
                             ("machine", (0.62, 0.72, 0.80, 1), dict(specular=0.5, shininess=0.6)),
                             ("hood", (0.82, 0.84, 0.86, 1), dict(specular=0.5, shininess=0.6))):
            KA.add_textured_material(spec, P + nm, P + "tex_steel", rgba=rgba, texuniform=True,
                                     texrepeat=(2.0, 2.0), **kw)
        KA.add_texture(spec, P + "tex_belt", A.belt_texture(c.seed))
        KA.add_textured_material(spec, P + "belt", P + "tex_belt", rgba=(1, 1, 1, 1), specular=0.15,
                                 shininess=0.3, texuniform=True, texrepeat=(4.0, 4.0))
        spec.add_material(name=P + "cleat", rgba=(0.30, 0.31, 0.33, 1), specular=0.2)
        KA.add_texture(spec, P + "tex_roller", A.roller_texture())
        KA.add_textured_material(spec, P + "roller", P + "tex_roller", rgba=(1, 1, 1, 1),
                                 specular=0.9, shininess=0.9)
        spec.add_material(name=P + "curtain", rgba=(0.62, 0.70, 0.76, 0.9), specular=0.6,
                          shininess=0.8)
        spec.add_material(name=P + "glass", rgba=(0.86, 0.93, 0.97, 0.38), specular=1.0,
                          shininess=1.0, reflectance=0.1)
        spec.add_material(name=P + "drop_sugar", rgba=(0.98, 0.64, 0.10, 0.93), specular=1.0,
                          shininess=0.95, emission=0.18)
        spec.add_material(name=P + "drop_bitter", rgba=(0.12, 0.30, 0.09, 1.0), specular=0.5,
                          shininess=0.5)
        KA.add_texture(spec, P + "tex_marble", A.marbled_texture(c.seed))
        KA.add_textured_material(spec, P + "drop_mixed", P + "tex_marble", rgba=(1, 1, 1, 1),
                                 specular=0.9, shininess=0.9, emission=0.08)
        spec.add_material(name=P + "drop_water", rgba=(0.78, 0.9, 1.0, 0.45), specular=1.0,
                          shininess=1.0, reflectance=0.2)
        for k in A.KINDS:
            for s in A.STAMPS:
                KA.add_texture(spec, f"{P}tex_card_{k}_{s}", A.card_texture(k, s))
                KA.add_textured_material(spec, f"{P}card_{k}_{s}", f"{P}tex_card_{k}_{s}",
                                         rgba=(1, 1, 1, 1), specular=0.1)
        spec.add_material(name=P + "ink_green", rgba=(*A.GREEN, 1), specular=0.3)
        spec.add_material(name=P + "ink_red", rgba=(*A.RED, 1), specular=0.3)
        spec.add_material(name=P + "stamp_green", rgba=(0.12, 0.45, 0.2, 1), specular=0.6, shininess=0.7)
        spec.add_material(name=P + "stamp_red", rgba=(0.7, 0.12, 0.12, 1), specular=0.6, shininess=0.7)
        spec.add_material(name=P + "bin_green", rgba=(0.16, 0.58, 0.26, 1), specular=0.4)
        spec.add_material(name=P + "bin_red", rgba=(0.78, 0.16, 0.14, 1), specular=0.4)
        for nm, word, rgb in (("approved", "APPROVED", A.GREEN), ("rejected", "REJECTED", A.RED)):
            KA.add_texture(spec, f"{P}tex_label_{nm}", A.bin_label_texture(word, rgb))
            KA.add_textured_material(spec, f"{P}label_{nm}", f"{P}tex_label_{nm}", rgba=(1, 1, 1, 1),
                                     emission=0.25)
        for nm, img, em in (("sign", A.sign_texture(), 0.35), ("tally", A.tally_texture(), 0.2),
                            ("gauge", A.gauge_texture(c.gauge_max_hz, c.approve_mn9_hz), 0.25),
                            ("clip", A.clipboard_texture(), 0.1), ("poster", A.poster_texture(), 0.15),
                            ("sign_in", A.bin_label_texture("SAMPLES IN", (0.12, 0.2, 0.3)), 0.3)):
            KA.add_texture(spec, f"{P}tex_{nm}", img)
            KA.add_textured_material(spec, P + nm, f"{P}tex_{nm}", rgba=(1, 1, 1, 1), emission=em,
                                     specular=0.2)
        KA.add_texture(spec, P + "tex_wall", A.wall_texture(c.seed))
        KA.add_textured_material(spec, P + "wall", P + "tex_wall", rgba=(1, 1, 1, 1), specular=0.2)
        spec.add_material(name=P + "seg_on_g", rgba=(0.12, 0.85, 0.22, 1), emission=0.7)
        spec.add_material(name=P + "seg_on_r", rgba=(0.95, 0.12, 0.08, 1), emission=0.7)
        spec.add_material(name=P + "seg_off", rgba=(0.07, 0.075, 0.08, 1), emission=0.0)
        spec.add_material(name=P + "lamp_off", rgba=(0.3, 0.3, 0.3, 1), specular=0.8, shininess=0.8)
        spec.add_material(name=P + "lamp_green", rgba=(0.3, 1.0, 0.4, 1), emission=1.0)
        spec.add_material(name=P + "lamp_red", rgba=(1.0, 0.25, 0.2, 1), emission=1.0)
        spec.add_material(name=P + "bar_amber", rgba=(0.95, 0.6, 0.1, 1), emission=0.6)
        spec.add_material(name=P + "bar_green", rgba=(0.2, 0.85, 0.3, 1), emission=0.6)
        spec.add_material(name=P + "clip_board", rgba=(0.55, 0.36, 0.2, 1), specular=0.3)

    def _add_room(self, spec) -> None:
        """Back wall with sign, tally board, poster; the MN9 meter and decision lamps
        beside the fly; a clipboard; lab lights."""
        c = self.cfg
        wb = spec.worldbody
        vis = contact_kwargs("visual")
        vm = dict(vis, mass=0.0)
        fx, fy = c.station_x, c.station_y
        xw = fx - 3.4
        KA.add_mesh(spec, P + "wall_mesh", A.front_panel_mesh(16.0, 6.0, 0.05))
        wb.add_geom(name=P + "wall", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "wall_mesh",
                    pos=(xw, fy, 6.0), material=P + "wall", **vm)
        KA.add_mesh(spec, P + "sign_mesh", A.front_panel_mesh(3.0, 0.75, 0.05))
        wb.add_geom(name=P + "sign", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "sign_mesh",
                    pos=(xw + 0.04, fy + 0.3, 4.0), material=P + "sign", **vm)
        KA.add_mesh(spec, P + "poster_mesh", A.front_panel_mesh(0.75, 1.0, 0.02))
        wb.add_geom(name=P + "poster", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "poster_mesh",
                    pos=(xw + 0.04, fy + 4.6, 3.0), material=P + "poster", **vm)
        # tally board: two 3-digit displays
        tb_y, tb_z, hy, hz = fy - 2.6, 2.3, 1.5, 0.75
        KA.add_mesh(spec, P + "tally_mesh", A.front_panel_mesh(hy, hz, 0.05))
        wb.add_geom(name=P + "tally", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "tally_mesh",
                    pos=(xw + 0.06, tb_y, tb_z), material=P + "tally", **vm)
        self._seg_names: list[list[list[str]]] = []  # [display][digit][segment]
        dw, dh, st = 0.26, 0.24, 0.04  # digit unit width, half height unit, stroke
        for di, yc in enumerate((tb_y - hy / 2, tb_y + hy / 2)):
            disp = []
            for k in range(3):
                y_d = yc + (k - 1) * 0.4
                segs = []
                for s, (dy, dz, horiz) in SEGS.items():
                    nm = f"{P}seg{di}_{k}_{s}"
                    size = (0.01, dw / 2 - 0.02, st) if horiz else (0.01, st, dh / 2 - 0.02)
                    wb.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_BOX, size=size,
                                pos=(xw + 0.1, y_d + dy * dw, tb_z + 0.14 + dz * dh),
                                material=P + "seg_off", **vm)
                    segs.append(nm)
                disp.append(segs)
            self._seg_names.append(disp)
        # MN9 meter (the brain's proboscis motor neuron, live) and the decision lamps
        gx, gy, gz, ghy, ghz = fx - 0.7, fy + 2.3, 1.45, 0.3, 1.1
        KA.add_mesh(spec, P + "gauge_mesh", A.front_panel_mesh(ghy, ghz, 0.04))
        wb.add_geom(name=P + "gauge", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "gauge_mesh",
                    pos=(gx, gy, gz), material=P + "gauge", **vm)
        wb.add_geom(name=P + "gauge_post", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.05, 0.5 * (gz - ghz), 0),
                    pos=(gx - 0.05, gy, 0.5 * (gz - ghz)), material=P + "frame", **vis)
        top_f, bot_f = 0.12, 0.95  # the scale's 100 Hz / 0 Hz rows (gauge_texture)
        self._gauge_z0 = gz + ghz - 2 * ghz * bot_f
        self._gauge_span = 2 * ghz * (bot_f - top_f)
        bar = wb.add_body(name=P + "gauge_bar", pos=(gx + 0.03, gy - ghy + 2 * ghy * 0.52, self._gauge_z0),
                          mocap=True)
        bar.add_geom(name=P + "gauge_bar_g", type=mj.mjtGeom.mjGEOM_BOX, size=(0.012, 0.11, 0.01),
                     pos=(0, 0, 0.01), material=P + "bar_amber", **vm)
        for nm, dy in (("lamp_ok", -0.14), ("lamp_no", 0.14)):
            wb.add_geom(name=P + nm, type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.11, 0, 0),
                        pos=(gx, gy + dy, gz + ghz + 0.16), material=P + "lamp_off", **vm)
        # clipboard on a little easel at the fly's right
        cx, cy = fx - 0.5, fy - 2.3
        q = quat_mul(quat_axis_angle((0, 1, 0), -0.3), (1.0, 0.0, 0.0, 0.0))
        KA.add_mesh(spec, P + "clip_board_mesh", A.front_panel_mesh(0.45, 0.62, 0.03))
        KA.add_mesh(spec, P + "clip_sheet_mesh", A.front_panel_mesh(0.4, 0.55, 0.01))
        wb.add_geom(name=P + "clipboard", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "clip_board_mesh",
                    pos=(cx, cy, 0.62), quat=q, material=P + "clip_board", **vm)
        wb.add_geom(name=P + "clip_sheet", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "clip_sheet_mesh",
                    pos=(cx + 0.025, cy, 0.6), quat=q, material=P + "clip", **vm)
        wb.add_geom(name=P + "clip_clip", type=mj.mjtGeom.mjGEOM_BOX, size=(0.03, 0.14, 0.05),
                    pos=(cx + 0.2, cy, 1.18), quat=q, material=P + "steel", **vis)
        # lights: a cool key spot from above the belt (shadows), a fill, the headlight
        spec.visual.headlight.ambient = (0.30, 0.31, 0.32)
        spec.visual.headlight.diffuse = (0.34, 0.35, 0.36)
        spec.visual.headlight.specular = (0.12, 0.12, 0.12)
        tgt = np.array([self.x_near, fy - 0.5, 0.4])
        key = np.array([fx + 7.0, fy + 5.0, 14.0])
        wb.add_light(name=P + "key", type=mj.mjtLightType.mjLIGHT_SPOT, pos=tuple(key),
                     dir=tuple(tgt - key), diffuse=(0.62, 0.64, 0.66), specular=(0.5, 0.5, 0.5),
                     cutoff=40.0, exponent=0.5, castshadow=bool(c.shadows))
        fill = np.array([fx + 4.0, fy - 7.0, 8.0])
        wb.add_light(name=P + "fill", type=mj.mjtLightType.mjLIGHT_SPOT, pos=tuple(fill),
                     dir=tuple(tgt - fill), diffuse=(0.28, 0.29, 0.32), specular=(0.2, 0.2, 0.2),
                     cutoff=50.0, exponent=1.0, castshadow=False)
        # fluorescent tube fixtures on the ceiling (emissive, out of view mostly)
        spec.add_material(name=P + "tube", rgba=(0.95, 0.98, 1.0, 1), emission=1.0)
        for j, y in enumerate((fy - 4.0, fy + 1.0, fy + 6.0)):
            wb.add_geom(name=f"{P}tube{j}", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.08, 1.6, 0),
                        pos=(fx + 1.0, y, 9.0), quat=quat_axis_angle((0, 1, 0), math.pi / 2),
                        material=P + "tube", **vis)

    # ------------------------------------------------------------ attach / reset
    def on_attach(self) -> None:
        sim = self.sim
        m = sim.model
        c = self.cfg
        fly = sim.fly_name
        mid = lambda n: int(m.body_mocapid[m.body(n).id])  # noqa: E731
        self.sample_mocap = np.array([mid(n) for n in self._sample_names])
        self.drop_gid = np.array([m.geom(f"{n}_drop").id for n in self._sample_names])
        self.card_gid = np.array([m.geom(f"{n}_card").id for n in self._sample_names])
        self.cleat_mocap = np.array([mid(n) for n in self._cleat_names])
        self.roller_mocap = np.array([mid(n) for n in self._roller_names])
        self.stamp_mocap = {"approved": mid(self._stamp_names[0]), "rejected": mid(self._stamp_names[1])}
        self.stamp_home = {k: m.body_pos[m.body(n).id].copy() for k, n in
                           zip(("approved", "rejected"), self._stamp_names)}
        self.paddle_mocap = mid(P + "paddle")
        self.paddle_home = m.body_pos[m.body(P + "paddle").id].copy()
        self.paddle_gid = np.array([m.geom(P + n).id for n in ("paddle_plate", "paddle_rod")])
        m.geom_sameframe[self.paddle_gid] = 0  # the plate is lowered by moving its geoms
        self.paddle_pos0 = m.geom_pos[self.paddle_gid].copy()
        self.bar_mocap = mid(P + "gauge_bar")
        self.bar_gid = m.geom(P + "gauge_bar_g").id
        m.geom_sameframe[self.bar_gid] = 0  # geom_pos / size edited at runtime
        m.geom_rbound[self.bar_gid] = 2.0
        self.lamp_gid = {"approved": m.geom(P + "lamp_ok").id, "rejected": m.geom(P + "lamp_no").id}
        matid = lambda n: int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_MATERIAL, n))  # noqa: E731
        self.mat = {n: matid(P + n) for n in (
            "lamp_off", "lamp_green", "lamp_red", "bar_amber", "bar_green", "seg_on_g", "seg_on_r",
            "seg_off", *(f"drop_{k}" for k in A.KINDS),
            *(f"card_{k}_{s}" for k in A.KINDS for s in A.STAMPS))}
        self.seg_gid = [[[m.geom(n).id for n in dig] for dig in disp] for disp in self._seg_names]
        # proboscis actuators and the tapping leg
        self.prob_ids = []
        for dof, sign in (("c_head-c_rostrum-pitch", -1.0), ("c_rostrum-c_haustellum-pitch", 1.0)):
            a = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, f"{fly}/{dof}-proboscispos")
            if a >= 0:
                self.prob_ids.append((a, sign))
        self.tarsus_bid = m.body(f"{fly}/{c.tap_leg}_tarsus5").id
        from fly_simulator.jobs.dead_hang import LegIK

        self.ik = LegIK(sim, self.session.actions.body)
        st = getattr(self.session, "STATIONARY_ACTIONS", None)
        if st is not None and TasterStance.name not in st:
            self.session.STATIONARY_ACTIONS = (*st, TasterStance.name)
        self.rng = np.random.default_rng(c.seed + 77)
        self.samples = [_Sample() for _ in range(c.n_samples)]
        self._serial = 0
        # counters (O(1) memory)
        self.n_tasted = 0
        self.n_approved = 0
        self.n_rejected = 0
        self.n_correct = 0  # vs the label (only pure sugar should pass)
        self.n_scored = 0  # sugar / bitter / water decisions (mixed scored apart)
        self.per_kind = {k: {"n": 0, "approved": 0, "mn9_sum": 0.0, "mn9_min": math.inf,
                             "mn9_max": 0.0} for k in A.KINDS}
        self.n_stamped = 0
        self.n_binned = {"approved": 0, "rejected": 0}
        self.n_recycled = 0
        self.n_touch_miss = 0
        self.n_brain_timeouts = 0
        self.n_voided = 0
        self.n_stim = 0
        self.last = deque(maxlen=6)  # recent decisions (text)
        self.last_line = ""
        self.mn9_hz = 0.0
        self.proboscis = 0.0
        self._states: deque = deque(maxlen=80)  # (sim_time, MN9) of recent brain windows
        self._last_seq = -1
        self._pending_stim = None
        self._last_rt: float | None = None
        self.decision_source = "scripted"
        self.on_reset()
        self.reset_props()

    def on_reset(self) -> None:
        if getattr(self, "_cur", None) is not None and self.state not in ("move", "settle"):
            self.n_voided += 1
        self.state = "settle"
        self._t_state = self.sim.time
        self._cur = None  # the sample being tasted (index)
        self._stance = None
        self._leg_seg = None
        self._touch_rt = None
        self._decision = ""
        self._per_until = -1.0
        self._lamp = ""
        self._stamp_job = None
        self._pending_stim = None

    def reset_props(self) -> None:
        """Fresh line: samples on stations 0-3, bins empty, machines home."""
        d = self.sim.data
        for s in self.samples:
            s.where, s.slot, s.bin, s.decision, s.stamped, s.dx = "spare", -1, "", "", False, 0.0
        self.station = [None] * N_STATIONS
        self._bin_fifo: deque = deque()
        self._bin_slots = {k: [None] * 6 for k in self._bins}
        for k in range(TASTE_K + 1):
            i = self._take_sample()
            self.station[k] = i
        self.belt_s = 0.0
        self._belt_s0 = 0.0
        self._flying: list[int] = []
        self._paddle_t0 = None
        for k, mid in self.stamp_mocap.items():
            d.mocap_pos[mid] = self.stamp_home[k]
        d.mocap_pos[self.paddle_mocap] = self.paddle_home
        self.sim.model.geom_pos[self.paddle_gid] = self.paddle_pos0
        self._place_all(0.0)
        self._set_lamp("")
        self._update_tally()

    # ------------------------------------------------------------ samples
    def _new_kind(self, s: _Sample) -> None:
        c = self.cfg
        w = np.asarray(c.kind_weights, float)
        s.kind = A.KINDS[int(self.rng.choice(len(A.KINDS), p=w / w.sum()))]
        u = lambda r: float(self.rng.uniform(*r))  # noqa: E731
        s.sugar_hz = {"sugar": u(c.sugar_hz), "mixed": u(c.mixed_sugar_hz)}.get(s.kind, 0.0)
        s.bitter_hz = {"bitter": u(c.bitter_hz), "mixed": u(c.mixed_bitter_hz)}.get(s.kind, 0.0)
        s.sugar_hz, s.bitter_hz = round(s.sugar_hz / 5) * 5.0, round(s.bitter_hz / 5) * 5.0
        self._serial += 1
        s.serial = self._serial
        s.decision, s.stamped, s.dx = "", False, 0.0

    def _apply_looks(self, i: int) -> None:
        s = self.samples[i]
        m = self.sim.model
        m.geom_matid[self.drop_gid[i]] = self.mat[f"drop_{s.kind}"]
        m.geom_matid[self.card_gid[i]] = self.mat[f"card_{s.kind}_{s.decision if s.stamped else 'blank'}"]

    def set_sample(self, k: int, kind: str, sugar_hz: float | None = None,
                   bitter_hz: float | None = None) -> bool:
        """Make the sample on belt station ``k`` a given type (keys 5-8 queue the next
        one; tests). Rates default to the middle of the config ranges. Returns False if
        the station is empty or its sample was already tasted."""
        c = self.cfg
        i = self.station[k] if 0 <= k < N_STATIONS else None
        if kind not in A.KINDS or i is None or self.samples[i].decision:
            return False
        s = self.samples[i]
        mid = lambda r: 0.5 * (r[0] + r[1])  # noqa: E731
        s.kind = kind
        if sugar_hz is None:
            sugar_hz = {"sugar": mid(c.sugar_hz), "mixed": mid(c.mixed_sugar_hz)}.get(kind, 0.0)
        if bitter_hz is None:
            bitter_hz = {"bitter": mid(c.bitter_hz), "mixed": mid(c.mixed_bitter_hz)}.get(kind, 0.0)
        s.sugar_hz, s.bitter_hz = float(sugar_hz), float(bitter_hz)
        self._apply_looks(i)
        return True

    def handle_key(self, k: str) -> bool:
        """run_job.py forwards keys: 5 / 6 / 7 / 8 make the next sample in the queue
        sugar / bitter / mixed / water (like the taste patches' 5 / 6 / 7)."""
        kind = {"5": "sugar", "6": "bitter", "7": "mixed", "8": "water"}.get(k)
        if kind is None:
            return False
        # (during a move the stations shift at its end: the next one is still upstream)
        k_next = TASTE_K if self.state == "settle" else TASTE_K - 1
        ok = self.set_sample(k_next, kind)
        self.say(f"next sample: {kind}" if ok else "no sample to change")
        return True

    def _take_sample(self) -> int:
        """A spare sample, or the oldest one in a bin (washed and refilled: recycled)."""
        spare = [i for i, s in enumerate(self.samples) if s.where == "spare"]
        if spare:
            i = spare[0]
        else:
            i = self._bin_fifo.popleft()
            s = self.samples[i]
            self._bin_slots[s.bin][s.slot] = None
            self.n_recycled += 1
        s = self.samples[i]
        self._new_kind(s)
        s.where, s.bin, s.slot = "belt", "", -1
        self._apply_looks(i)
        return i

    def _place_all(self, travel: float) -> None:
        """Belt samples at their stations minus ``travel`` (mm along -y), cleats and
        rollers by the belt position."""
        d = self.sim.data
        c = self.cfg
        for k, i in enumerate(self.station):
            if i is None:
                continue
            s = self.samples[i]
            d.mocap_pos[self.sample_mocap[i]] = (self.x_cup + s.dx, self.station_y(k) - travel,
                                                c.belt_top)
            d.mocap_quat[self.sample_mocap[i]] = (1.0, 0.0, 0.0, 0.0)
        L = self.y_top - self.y_end
        n = len(self.cleat_mocap)
        sp = L / n
        for j, mid in enumerate(self.cleat_mocap):
            yy = self.y_top - ((j * sp + self.belt_s) % L)
            d.mocap_pos[mid, 1] = yy
        ang = self.belt_s / 0.12
        q = np.array(quat_axis_angle((1, 0, 0), ang))
        for mid in self.roller_mocap:
            d.mocap_quat[mid] = q

    # ------------------------------------------------------------ helpers
    def _set_lamp(self, which: str) -> None:
        m = self.sim.model
        self._lamp = which
        m.geom_matid[self.lamp_gid["approved"]] = self.mat["lamp_green" if which == "approved" else "lamp_off"]
        m.geom_matid[self.lamp_gid["rejected"]] = self.mat["lamp_red" if which == "rejected" else "lamp_off"]

    def _update_tally(self) -> None:
        m = self.sim.model
        for di, (val, on) in enumerate(((self.n_approved, "seg_on_g"), (self.n_rejected, "seg_on_r"))):
            v = int(val) % 1000
            digits = (v // 100, (v // 10) % 10, v % 10)
            for k, dg in enumerate(digits):
                lit = DIGITS[dg] if (k == 2 or v >= 10 ** (2 - k)) else ""
                for s, gid in zip(SEGS, self.seg_gid[di][k]):
                    m.geom_matid[gid] = self.mat[on if s in lit else "seg_off"]

    def _set_gauge(self, hz: float) -> None:
        m, d = self.sim.model, self.sim.data
        c = self.cfg
        f = min(max(hz / c.gauge_max_hz, 0.0), 1.0)
        h = max(0.5 * f * self._gauge_span, 0.004)
        m.geom_size[self.bar_gid, 2] = h
        m.geom_pos[self.bar_gid, 2] = h
        m.geom_matid[self.bar_gid] = self.mat["bar_green" if hz >= c.approve_mn9_hz else "bar_amber"]

    def tarsus_pos(self) -> np.ndarray:
        return self.sim.data.xpos[self.tarsus_bid].copy()

    def drop_centre(self, i: int | None = None) -> np.ndarray:
        i = self._cur if i is None else i
        p = self.sim.data.mocap_pos[self.sample_mocap[i]]
        return np.array([p[0], p[1], self.drop_top])

    def _ensure_stance(self) -> TasterStance:
        acts = self.session.actions
        a = acts.action if isinstance(acts.action, TasterStance) else None
        if a is None and not isinstance(acts._pending, TasterStance):
            a = TasterStance(self.cfg.tap_leg)
            acts.trigger(a, source="job")
        elif a is None:
            a = acts._pending
        self._stance = a
        return a

    def _ik(self, target) -> np.ndarray:
        b = self.session.actions.body
        cols = np.flatnonzero(b.leg_mask([self.cfg.tap_leg]))
        ref = b.stand[cols]
        q, _err = self.ik.solve(self.sim.data.qpos, self.cfg.tap_leg, [("tarsus5", target)],
                                ref=ref, ref_gain=0.1)
        return q

    def _leg_now(self) -> np.ndarray:
        st = self._stance
        if st is not None and st.leg_pose is not None and st.leg_w > 0:
            return st.leg_pose.copy()
        b = self.session.actions.body
        return b.stand[np.flatnonzero(b.leg_mask([self.cfg.tap_leg]))].copy()

    def _leg_to(self, q_to: np.ndarray, dur: float) -> None:
        self._leg_seg = (self._leg_now(), np.asarray(q_to, float), self.sim.time, max(dur, 1e-3))

    def _leg_step(self) -> float:
        """Advance the leg segment; returns its progress 0..1."""
        st = self._stance
        if st is None or self._leg_seg is None:
            return 1.0
        q0, q1, t0, dur = self._leg_seg
        u = min(max((self.sim.time - t0) / dur, 0.0), 1.0)
        st.leg_pose = q0 + smoothstep(u) * (q1 - q0)
        st.leg_w = 1.0
        return u

    # ------------------------------------------------------------ main loop (1 ms)
    def update(self) -> None:
        c = self.cfg
        t = self.sim.time
        dt = t - self._t_state
        self.steering.set(None, 0.0)
        if self.state == "settle":
            if dt > 0.3:
                self._ensure_stance()
            if dt > 0.6:  # taste the sample already in front of the fly
                self._cur = self.station[TASTE_K]
                self._leg_seg = None
                self._decision = ""
                self._go("lift" if self._cur is not None else "retract")
        else:
            self._ensure_stance()
        if self.state == "move":
            u = min(dt / c.move_s, 1.0)
            e = smoothstep(u)
            self.belt_s = self._belt_s0 + c.pitch * e
            self._place_all(c.pitch * e)
            if u >= 1.0:
                self._end_move()
        elif self.state == "lift":
            p = self.drop_centre()
            if self._leg_seg is None:
                tgt = p + np.array([-0.15, c.tap_dy, 0.35])
                self._leg_to(self._ik(tgt), c.lift_s)
            if self._leg_step() >= 1.0:
                tgt = p + np.array([-0.04, c.tap_dy, 0.01])
                self._leg_to(self._ik(tgt), c.reach_s)
                self._go("reach")
        elif self.state == "reach":
            u = self._leg_step()
            if self._touching() or (u >= 1.0 and dt > c.reach_s + c.touch_grace_s):
                if not self._touching():
                    self.n_touch_miss += 1
                self._start_taste()
        elif self.state == "taste":
            self._leg_step()
            self._try_decide()
        elif self.state == "respond":
            self._leg_step()
            if self._decision == "rejected":
                if self._push_step():
                    self._retract()
            elif dt >= c.per_s:
                self._retract()
        elif self.state == "retract":
            u = self._leg_step()
            if u >= 1.0:
                if self._stance is not None:
                    self._stance.leg_w = 0.0
                    self._stance.leg_pose = None
                self._leg_seg = None
                if dt >= c.retract_s + c.pause_s:
                    self._begin_move()
        self._animate_machines()
        self._drive_proboscis()

    def _go(self, state: str) -> None:
        self.state = state
        self._t_state = self.sim.time

    def _touching(self) -> bool:
        c = self.cfg
        p = self.tarsus_pos()
        q = self.drop_centre()
        r = math.hypot(p[0] - q[0], p[1] - q[1])
        return r < c.drop_r + c.touch_tol and p[2] < q[2] + c.touch_tol

    # ------------------------------------------------------------ belt
    def _begin_move(self) -> None:
        self._cur = None
        self._belt_s0 = self.belt_s
        self._set_lamp("")
        end = self.station[END_K]
        if end is not None:
            self._send_to_bin(end)
            self.station[END_K] = None
        self._go("move")

    def _end_move(self) -> None:
        self.station = [None] + self.station[:END_K]
        self.station[0] = self._take_sample()
        self._place_all(0.0)
        # stamp the sample that just arrived under the stamper
        j = self.station[STAMP_K]
        if j is not None and self.samples[j].decision and not self.samples[j].stamped:
            self._stamp_job = (j, self.samples[j].decision, self.sim.time)
        i = self.station[TASTE_K]
        self._cur = i
        self._leg_seg = None
        self._decision = ""
        self._go("lift" if i is not None else "retract")

    # ------------------------------------------------------------ tasting / decision
    def _start_taste(self) -> None:
        s = self.samples[self._cur]
        self._touch_rt = self.run_time()
        tastes = [n for n, hz in (("sugar", s.sugar_hz), ("bitter", s.bitter_hz)) if hz > 0]
        self._pending_stim = (s, tastes, self._touch_rt)
        self._go("taste")

    def _brain(self):
        return getattr(self.session, "brain", None)

    def _try_decide(self) -> None:
        c = self.cfg
        rt = self.run_time()
        t0 = self._touch_rt
        s = self.samples[self._cur]
        link = self._brain()
        if link is None:  # scripted fallback (clearly labelled)
            if rt - t0 >= c.taste_s:
                ok = s.sugar_hz > 0 and s.bitter_hz == 0
                self._decide("approved" if ok else "rejected", float("nan"), float("nan"), "scripted")
            return
        lo, hi = t0 + c.mn9_lag_s, t0 + c.taste_s + c.mn9_lag_s
        win = [hz for (ts, hz) in self._states if lo < ts <= hi + 1e-6]
        covered = any(ts >= hi - 1e-6 for ts, _ in self._states)
        if covered or rt - t0 > c.brain_wait_s:
            if not covered:
                self.n_brain_timeouts += 1
            if win:
                mean, peak = float(np.mean(win)), float(np.max(win))
            else:
                mean, peak = 0.0, 0.0
            self._decide("approved" if mean >= c.approve_mn9_hz else "rejected", mean, peak,
                         "brain" if win else "brain (no response)")

    def _decide(self, decision: str, mean: float, peak: float, source: str) -> None:
        c = self.cfg
        s = self.samples[self._cur]
        s.decision = decision
        self._decision = decision
        self.decision_source = source
        self.n_tasted += 1
        self.add_work(1)
        ok = decision == "approved"
        self.n_approved += ok
        self.n_rejected += not ok
        pk = self.per_kind[s.kind]
        pk["n"] += 1
        pk["approved"] += ok
        if math.isfinite(mean):
            pk["mn9_sum"] += mean
            pk["mn9_min"] = min(pk["mn9_min"], mean)
            pk["mn9_max"] = max(pk["mn9_max"], mean)
        if s.kind != "mixed":
            self.n_scored += 1
            self.n_correct += ok == (s.kind == "sugar")
        self._update_tally()
        self._set_lamp(decision)
        mn9 = f"MN9 {mean:.0f} Hz (peak {peak:.0f})" if math.isfinite(mean) else "no brain: rule"
        self.last_line = (f"#{s.serial} {s.kind.upper()} {self._taste_text(s)} -> {mn9} -> "
                          f"{decision.upper()} [{source}]")
        self.last.append(self.last_line)
        self.say(self.last_line)
        self._go("respond")
        if ok:
            self._per_until = self.sim.time + c.per_s
        else:
            self._push_phase = 0
            p = self.drop_centre()
            tgt = p + np.array([-c.cup_r - 0.02, c.tap_dy, -c.drop_h + 0.03])
            tgt[2] = max(tgt[2], c.belt_top + 0.06)
            lift = p + np.array([-0.2, c.tap_dy, 0.2])
            self._push_pts = (lift, tgt, tgt + np.array([c.push_mm, 0, 0]))
            self._leg_to(self._ik(lift), 0.4 * c.push_s)

    def _push_step(self) -> bool:
        """Reject: lift, drop behind the dish's near rim, shove it away (the dish follows
        the leg's push phase kinematically). Returns True when the push is done."""
        c = self.cfg
        if self._leg_seg is None:
            return True
        u = min(max((self.sim.time - self._leg_seg[2]) / self._leg_seg[3], 0.0), 1.0)
        if self._push_phase == 2:
            s = self.samples[self._cur]
            s.dx = c.push_mm * smoothstep(u)
            self._place_all(0.0)
            return u >= 1.0
        if u < 1.0:
            return False
        if self._push_phase == 0:
            self._push_phase = 1
            self._leg_to(self._ik(self._push_pts[1]), 0.25 * c.push_s)
        elif self._push_phase == 1:
            self._push_phase = 2
            self._leg_to(self._ik(self._push_pts[2]), 0.45 * c.push_s)
        return False

    def _retract(self) -> None:
        b = self.session.actions.body
        self._leg_to(b.stand[np.flatnonzero(b.leg_mask([self.cfg.tap_leg]))], self.cfg.retract_s)
        self._go("retract")

    @staticmethod
    def _taste_text(s: _Sample) -> str:
        parts = []
        if s.sugar_hz > 0:
            parts.append(f"sugar {s.sugar_hz:.0f}")
        if s.bitter_hz > 0:
            parts.append(f"bitter {s.bitter_hz:.0f}")
        return ("(" + " + ".join(parts) + " Hz)") if parts else "(no taste input)"

    # ------------------------------------------------------------ machines
    def _send_to_bin(self, i: int) -> None:
        s = self.samples[i]
        which = s.decision or "rejected"
        slots = self._bin_slots[which]
        if None not in slots:  # full: recycle its oldest now (cannot happen with 12)
            j = next(k for k in self._bin_fifo if self.samples[k].bin == which)
            self._bin_fifo.remove(j)
            slots[self.samples[j].slot] = None
            self._park(j)
        slot = slots.index(None)
        slots[slot] = i
        s.where, s.bin, s.slot = "flying", which, slot
        ctr, (hx, hy, _hz) = self._bins[which]
        gx, gy = slot % 3, slot // 3
        p1 = np.array([ctr[0] - hx + (gx + 0.5) * 2 * hx / 3,
                       ctr[1] - hy + (gy + 0.5) * hy, 0.04])
        p1[:2] += self.rng.uniform(-0.05, 0.05, 2)
        p0 = self.sim.data.mocap_pos[self.sample_mocap[i]].copy()
        q1 = quat_mul(quat_axis_angle((0, 0, 1), float(self.rng.uniform(-0.8, 0.8))),
                      quat_axis_angle((1, 0, 0), float(self.rng.uniform(-0.35, 0.35))))
        s.fly = {"t0": self.sim.time, "p0": p0, "p1": p1, "q1": np.array(q1),
                 "dur": 0.9}  # (the diverter's timing: _diverter)
        self._flying.append(i)
        self._paddle_t0 = (self.sim.time, which, float(p0[0]))
        self.n_binned[which] += 1

    def _edge_x(self, which: str) -> float:
        """Dish centre x where a swept sample leaves the belt toward its bin."""
        return self.x_far + 0.12 if which == "approved" else self.x_near - 0.25

    def _diverter(self, tt: float, which: str, cup_x: float) -> tuple[float, float, float]:
        """The diverter at ``tt`` s into a push: (paddle x, plate drop 0..1, dish x).
        The carriage runs behind the dish (0-0.25 s; approved: from the fly's side,
        rejected: from beyond the card), the plate drops and sweeps the dish to the belt
        edge (0.25-0.6 s), lifts and returns (0.6-1 s)."""
        off = -0.45 if which == "approved" else 1.0  # plate x - dish x while pushing
        home = float(self.paddle_home[0])
        x_start, x_end = cup_x + off, self._edge_x(which) + off
        if tt < 0.25:
            a = smoothstep(tt / 0.25)
            return home + a * (x_start - home), a, cup_x
        if tt < 0.6:
            a = smoothstep((tt - 0.25) / 0.35)
            x = x_start + a * (x_end - x_start)
            return x, 1.0, x - off
        a = smoothstep(min((tt - 0.6) / 0.4, 1.0))
        return x_end + a * (home - x_end), 1.0 - a, self._edge_x(which)

    def _park(self, i: int) -> None:
        s = self.samples[i]
        s.where = "spare"
        self.sim.data.mocap_pos[self.sample_mocap[i]] = (self.x_far + 3.0 + 0.8 * i, -9.0, -1.0)

    def _animate_machines(self) -> None:
        d = self.sim.data
        t = self.sim.time
        c = self.cfg
        # samples leaving the belt end: swept off by the diverter, then dropped into the bin
        for i in list(self._flying):
            s = self.samples[i]
            f = s.fly
            tt = t - f["t0"]
            p0, p1 = f["p0"], f["p1"]
            _, _, xs = self._diverter(tt, s.bin, p0[0])
            e = np.array([self._edge_x(s.bin), p0[1], p0[2]])
            if tt < 0.6:
                p = np.array([xs, p0[1], p0[2]])
                q = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                v = min((tt - 0.6) / 0.3, 1.0)
                p = e + (p1 - e) * smoothstep(v)
                p[2] = e[2] + (p1[2] - e[2]) * v * v  # falls in, accelerating
                q = _slerp_quat((1, 0, 0, 0), f["q1"], smoothstep(v))
            d.mocap_pos[self.sample_mocap[i]] = p
            d.mocap_quat[self.sample_mocap[i]] = q
            if tt >= 0.9:
                self._flying.remove(i)
                s.where = "bin"
                self._bin_fifo.append(i)
        # the diverter paddle
        if self._paddle_t0 is not None:
            t0, which, x0 = self._paddle_t0
            x, drop, _ = self._diverter(t - t0, which, x0)
            if t - t0 >= 1.0:
                x, drop = self.paddle_home[0], 0.0
                self._paddle_t0 = None
            d.mocap_pos[self.paddle_mocap] = (x, self.paddle_home[1], self.paddle_home[2])
            self.sim.model.geom_pos[self.paddle_gid] = self.paddle_pos0 - np.array([0.0, 0.0, 0.42 * drop])
        # the stamper: the chosen stamp swings over the card, presses, returns
        if self._stamp_job is not None:
            j, which, t0 = self._stamp_job
            u = t - t0 - 0.15
            home = self.stamp_home[which]
            card = d.mocap_pos[self.sample_mocap[j]] + np.array([c.card_dx, 0, 0.02])
            shift, down, hold = 0.2, 0.25, 0.1
            tl = (shift, shift + down, shift + down + hold, shift + 2 * down + hold,
                  2 * shift + 2 * down + hold)
            if u < 0:
                sa, a = 0.0, 0.0
            elif u < tl[0]:  # the carriage slides the chosen stamp over the card
                sa, a = smoothstep(u / shift), 0.0
            elif u < tl[1]:
                sa, a = 1.0, smoothstep((u - tl[0]) / down)
            elif u < tl[2]:
                sa, a = 1.0, 1.0
                if not self.samples[j].stamped:
                    self.samples[j].stamped = True
                    self.n_stamped += 1
                    self.sim.model.geom_matid[self.card_gid[j]] = self.mat[
                        f"card_{self.samples[j].kind}_{which}"]
            elif u < tl[3]:
                sa, a = 1.0, 1.0 - smoothstep((u - tl[2]) / down)
            elif u < tl[4]:
                sa, a = 1.0 - smoothstep((u - tl[3]) / shift), 0.0
            else:
                sa, a = 0.0, 0.0
                self._stamp_job = None
            dy = sa * (card[1] - home[1])
            for k, mid in self.stamp_mocap.items():
                p = self.stamp_home[k] + np.array([0.0, dy, 0.0])
                if k == which:
                    p[2] = (1 - a) * home[2] + a * card[2]
                d.mocap_pos[mid] = p
        # the MN9 gauge
        self._set_gauge(self.mn9_hz)

    def _drive_proboscis(self) -> None:
        c = self.cfg
        link = self._brain()
        target = 0.0
        if link is not None:
            target = min(max(self.mn9_hz / max(c.mn9_ref_hz, 1e-6), 0.0), 1.0)
        if self.sim.time < self._per_until:
            target = max(target, 1.0)
        a = min(0.001 * self.cfg.update_every_steps / 0.1, 1.0)
        self.proboscis += a * (target - self.proboscis)
        d = self.sim.data
        for aid, sign in self.prob_ids:
            d.ctrl[aid] = sign * self.proboscis

    # ------------------------------------------------------------ outside sim.step
    def after_physics(self) -> None:
        super().after_physics()
        link = self._brain()
        rt = self.run_time()
        if link is not None:
            self._read_states(link, rt)
            if self._pending_stim is not None:
                s, tastes, t0 = self._pending_stim
                self._send_taste(link, s, tastes, t0)
        else:
            self.mn9_hz = 0.0
        self._pending_stim = None

    def _read_states(self, link, rt: float) -> None:
        if id(link) != getattr(self, "_link_id", None):  # a new brain: its own seq numbers
            self._link_id = id(link)
            self._last_seq = -1
        recent = getattr(link, "recent", None)
        states = list(recent) if recent is not None else []
        if not states and getattr(link, "latest", None) is not None:
            states = [link.latest]
        for st in states:
            seq = getattr(st, "seq", None)
            if seq is not None:
                if seq <= self._last_seq:
                    continue
                self._last_seq = seq
            ts = getattr(st, "sim_time", None)
            hz = float((getattr(st, "probes", None) or {}).get("MN9", 0.0))
            self._states.append((float(ts) if ts is not None else rt, hz))
        lt = getattr(link, "latest", None)
        if lt is not None:
            self.mn9_hz = float((getattr(lt, "probes", None) or {}).get("MN9", 0.0))

    def _send_taste(self, link, s: _Sample, tastes: list, t0: float) -> None:
        from fly_simulator.brain.schema import StimulusEvent

        c = self.cfg
        if not tastes:  # water: nothing to drive (no water GRNs in the stand-in sets)
            return
        det = {"tastes": tastes, "label": (f"SAMPLE #{s.serial} {s.kind.upper()} "
                                           f"{self._taste_text(s)} (tarsal tap; stand-in: "
                                           f"labellar GRNs)")}
        if s.sugar_hz > 0:
            det["sugar_hz"] = s.sugar_hz
        if s.bitter_hz > 0:
            det["bitter_hz"] = s.bitter_hz
        side = "left" if c.tap_leg.startswith("l") else "right"
        link.send(StimulusEvent("taste", side, 1.0, c.taste_s, t0, details=det), source="job")
        self.n_stim += 1
        log = getattr(link, "stim_log", None)
        if isinstance(log, list) and len(log) > 400:
            del log[:-200]  # constant memory over an eternal run

    # ------------------------------------------------------------ camera / HUD / stats
    def camera_target(self) -> np.ndarray:
        c = self.cfg
        return np.array([c.station_x + 1.1, c.station_y - 1.4, 1.25])

    def camera_preset(self) -> CameraPreset:
        return CameraPreset(azimuth=188.0, elevation=-21.0, distance=7.0, tau_s=0.6)

    def accuracy(self) -> float | None:
        return self.n_correct / self.n_scored if self.n_scored else None

    def job_hud_lines(self) -> list[str]:
        acc = self.accuracy()
        pk = self.per_kind
        mixed = pk["mixed"]
        brain = self._brain() is not None
        kinds = []
        for k in A.KINDS:
            v = pk[k]
            mn = f" MN9 {v['mn9_sum'] / v['n']:.0f}" if v["n"] and brain else ""
            kinds.append(f"{k} {v['approved']}/{v['n']}{mn}")
        lines = [
            f"approved {self.n_approved}   rejected {self.n_rejected}   accuracy "
            + (f"{acc:.0%}" if acc is not None else "-") + " (sugar pass, bitter / water fail)"
            + f"   mixed passed {mixed['approved']}/{mixed['n']}",
            "passed per type: " + "   ".join(kinds),
            ("decision: REAL CONNECTOME - mean MN9 over the taste pulse >= "
             f"{self.cfg.approve_mn9_hz:.0f} Hz" if brain else
             "decision: SCRIPTED (no brain): sugar without bitter passes"),
            f"MN9 {self.mn9_hz:.0f} Hz (live)   proboscis {self.proboscis:.0%}" if brain
            else f"proboscis {self.proboscis:.0%}",
        ]
        if self.last_line:
            lines.append(self.last_line)
        lines.append("(taste input: labellar GRN stand-in; belt / stamper / diverter / "
                     "recycling: kinematic machines)")
        return lines

    def job_stats(self) -> dict:
        acc = self.accuracy()
        out = {"tasted": self.n_tasted, "approved": self.n_approved, "rejected": self.n_rejected,
               "accuracy": acc if acc is not None else float("nan"),
               "stamped": self.n_stamped, "recycled": self.n_recycled,
               "binned_ok": self.n_binned["approved"], "binned_no": self.n_binned["rejected"],
               "touch_miss": self.n_touch_miss, "brain_timeouts": self.n_brain_timeouts,
               "voided": self.n_voided, "stimuli": self.n_stim}
        for k, v in self.per_kind.items():
            out[f"{k}_n"] = v["n"]
            out[f"{k}_ok"] = v["approved"]
            if v["n"] and self._brain() is not None:
                out[f"{k}_mn9"] = v["mn9_sum"] / v["n"]
        return out
