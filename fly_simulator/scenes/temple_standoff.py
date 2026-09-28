"""``temple_standoff``: a short, dead-serious scripted scene with NeuroMechFly bodies.

An original stone hall at fly scale. A group of little flies (juvenile-looking,
0.65x, pale, rounder, in cream tunics and sashes) is gathered on the floor. A tall
fly (1.5x, dark, in a heavy hooded cloak, holding a small handle on its right front
leg) walks in through the back doorway onto the dais. The little flies turn; the
lead little fly steps forward and looks up; a long, uncomfortable hold on the tall
fly; a tiny leg movement brings the handle forward; the energy blade ignites (a real
MuJoCo light relights the room); the little flies recoil; cut to black. Violence is
implied only: nothing swings, nothing hits.

What is engineered (all of it: this is a film, not a simulation):

* every fly is a **posed, kinematic NeuroMechFly** on a mocap mount (the technique
  of ``jobs/broccoli_toss.py``'s viewer fly): its leg, neck, wing and antenna joint
  angles are written every frame and only ``mj_forward`` runs (no physics steps, no
  contacts). Walking uses FlyGym's recorded single-leg steps
  (``PreprogrammedSteps``) with the body speed tied to the stepping phase, so the
  stance feet barely slide; the tall fly's handle leg is posed by damped
  least-squares IK;
* the **sizes** are scaled copies of the same body (every body offset and mesh scale
  multiplied at build time): the little flies also get a relatively bigger head and
  a rounder abdomen, the tall fly a slightly smaller head;
* the **attire** is visual-only meshes shrink-wrapped onto the body with ``mj_ray``
  (``temple_standoff_assets``), on the thorax / head bodies so it follows the pose;
* the **energy blade** is a glowing rod (white core, blue glow layers) on a mocap
  body placed at the tall fly's right front tarsus every frame, with a MuJoCo point
  light that is dark until ignition and then relights everything; bloom, an
  exposure lift, grade, grain, vignette and the 2.39:1 letterbox are a screen-space
  post-process;
* the cameras are fixed per shot with hard cuts (``SHOTS``); the audio is
  synthesized with numpy (``synth_audio``).
"""

from __future__ import annotations

import math
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np

from fly_simulator.jobs.broccoli_toss_assets import panel_mesh, panel_quat
from fly_simulator.jobs.geometry import quat_axis_angle, quat_mul
from fly_simulator.scenes import temple_standoff_assets as A

NAME = "temple_standoff"
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
TRIPOD_A = ("lf", "rm", "lh")
DOFS = (("thorax", "coxa", "pitch"), ("thorax", "coxa", "roll"), ("thorax", "coxa", "yaw"),
        ("coxa", "trochanterfemur", "pitch"), ("coxa", "trochanterfemur", "roll"),
        ("trochanterfemur", "tibia", "pitch"), ("tibia", "tarsus1", "pitch"))
AX3 = ("yaw", "pitch", "roll")

# ---- the timeline (s) ---------------------------------------------------------
T_WALK0, T_WALK1 = 1.15, 5.35   # the tall fly's walk (from the dark corridor to the dais)
T_NOTICE = 4.05                 # the little flies begin to turn
T_STEP0, T_STEP1 = 5.65, 6.70   # the lead little fly steps forward
T_LOOKUP = 6.70                 # ... and looks up
T_TILT = 7.30                   # close-up: the head tilt
T_HANDLE0, T_HANDLE1 = 10.55, 10.97  # the front leg brings the handle forward
T_IGNITE = 11.0                 # the blade ignites
IGNITE_S = 0.09                 # blade growth (2-3 frames at 24-30 fps)
T_RECOIL = 11.30                # the little flies recoil
T_LEAD_RECOIL = 11.66           # the lead one freezes a beat first
T_BLACK = 12.55                 # cut to black
T_END = 13.0

FLOOR_Z = 0.0
DAIS_Z = 0.6
DAIS_X0 = 10.5  # the dais front edge (a step in front of it)


@dataclass(frozen=True)
class Shot:
    name: str
    t0: float
    t1: float
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    fovy: float  # vertical field of view of the full frame (the letterbox crops it)
    key: tuple | None = None  # (pos, target, rgb, cutoff): the shot's key light (a film set light)


SHOTS: tuple[Shot, ...] = (
    Shot("establishing_wide", 0.0, 2.0, (-8.5, -4.5, 4.2), (7.0, 0.8, 1.2), 50.0),
    Shot("tall_fly_enters", 2.0, 4.0, (6.2, -3.8, 1.35), (19.0, 0.4, 2.9), 34.0),
    Shot("little_flies_notice", 4.0, 5.5, (9.4, -1.3, 1.55), (2.4, 0.4, 0.35), 38.0,
         key=((4.0, 9.0, 9.0), (2.6, 0.2, 0.3), (0.16, 0.20, 0.28), 16.0)),
    Shot("height_contrast", 5.5, 7.0, (8.9, -12.6, 0.85), (9.2, 0.0, 1.75), 37.0,
         key=((5.0, -9.0, 13.0), (9.0, 0.0, 1.2), (0.14, 0.17, 0.24), 22.0)),
    Shot("lead_close_up", 7.0, 8.5, (8.4, -0.75, 1.55), (5.81, 0.11, 0.74), 30.0,
         key=((9.5, 3.0, 5.5), (5.8, 0.1, 0.8), (0.26, 0.31, 0.42), 9.0)),
    Shot("tall_fly_low_angle", 8.5, 10.5, (7.6, 1.7, 0.34), (12.3, -0.05, 2.55), 36.0,
         key=((9.0, -3.8, 6.8), (12.1, 0.0, 2.5), (0.55, 0.62, 0.80), 8.0)),
    Shot("handle_ignition", 10.5, 11.3, (8.6, -1.9, 0.75), (12.2, 0.4, 2.4), 46.0,
         key=((9.0, -3.8, 6.8), (12.1, 0.0, 2.5), (0.55, 0.62, 0.80), 8.0)),
    Shot("recoil", 11.3, T_BLACK, (11.8, 2.2, 2.3), (4.5, 0.2, 0.5), 40.0),
    Shot("black", T_BLACK, T_END, (0.0, 0.0, 5.0), (1.0, 0.0, 5.0), 40.0),
)


def shot_at(t: float) -> Shot:
    for s in SHOTS:
        if s.t0 <= t < s.t1:
            return s
    return SHOTS[-1]


def _smooth(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _smoother(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def _quat_yaw(yaw: float):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def _quat_from_z(d) -> np.ndarray:
    """Quaternion rotating +z onto direction ``d``."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    ax = np.cross(z, d)
    s = float(np.linalg.norm(ax))
    if s < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0]) if d[2] > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    return np.array(quat_axis_angle(ax / s, math.atan2(s, float(np.dot(z, d)))))


# ---- the cast -----------------------------------------------------------------
@dataclass
class FlyLook:
    scale: float
    head_scale: float = 1.0
    abdomen_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0)  # multiplies the body textures
    lift: tuple[float, float, float] = (0.0, 0.0, 0.0)  # ... then added (paler)
    eye: tuple[float, float, float, float] = (0.67, 0.21, 0.12, 1.0)
    wing: tuple[float, float, float, float] = (0.8, 0.8, 0.9, 0.3)
    specular: float = 0.2
    shininess: float = 0.3
    speckle: float = 1.0  # the texture's random marks (bristle dots)


TALL_LOOK = FlyLook(scale=1.5, head_scale=0.95, tint=(0.10, 0.10, 0.12), lift=(0.012, 0.012, 0.018),
                    eye=(0.30, 0.04, 0.05, 1.0), wing=(0.25, 0.25, 0.30, 0.55), specular=0.65, shininess=0.75)


def _little_look(k: int) -> FlyLook:
    s = (0.64, 0.66, 0.62, 0.68, 0.65, 0.63)[k % 6]
    return FlyLook(scale=s, head_scale=1.16, abdomen_scale=(0.96, 1.12, 1.06),
                   tint=(0.45, 0.44, 0.42), lift=(0.50, 0.45, 0.36), eye=(0.88, 0.42, 0.30, 1.0),
                   wing=(0.62, 0.64, 0.70, 0.18), specular=0.12, shininess=0.3, speckle=0.4)


@dataclass
class Little:
    name: str
    xy: tuple[float, float]
    yaw0: float  # deg, the gathering
    delay: float  # s, the notice stagger
    recoil: float  # mm


# the lead little fly first; the others around it (mount positions, mm)
LITTLE = (
    Little("lead", (4.05, 0.25), 175.0, 0.10, 0.50),
    Little("l1", (2.35, 2.35), -40.0, 0.00, 0.85),
    Little("l2", (2.05, -1.85), 35.0, 0.22, 0.80),
    Little("l3", (3.75, -3.75), 95.0, 0.30, 0.75),
    Little("l4", (0.15, 0.55), 5.0, 0.38, 0.80),
    Little("l5", (0.85, 4.35), -75.0, 0.16, 0.80),
)
LEAD_STEP_TO = (5.55, 0.12)
TALL_X0, TALL_X1, TALL_Y = 25.8, 12.9, 0.0


def _joint_name(leg: str, dof) -> str:
    p, c, ax = dof
    parent = "c_thorax" if p == "thorax" else f"{leg}_{p}"
    return f"{parent}-{leg}_{c}-{ax}"


def _make_fly(name: str, look: FlyLook, garments: dict, garment_tex: dict):
    """A NeuroMechFly spec: leg, neck, wing and antenna joints (no actuators),
    tinted, scaled, dressed. Returns the spec root (attach it with prefix name/)."""
    from flygym.anatomy import AnatomicalJoint, AxesSet, AxisOrder, JointPreset, Skeleton
    from flygym.compose import KinematicPosePreset, NeuroMechFly

    fly = NeuroMechFly(name=name)
    ax = AxesSet(["pitch", "roll", "yaw"])
    joints = JointPreset.LEGS_ONLY.to_joint_list() + [
        AnatomicalJoint("c_thorax", "c_head", ax), AnatomicalJoint("c_thorax", "l_wing", ax),
        AnatomicalJoint("c_thorax", "r_wing", ax), AnatomicalJoint("c_head", "l_pedicel", ax),
        AnatomicalJoint("c_head", "r_pedicel", ax)]
    sk = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, anatomical_joints=joints)
    neutral = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
    fly.add_joints(sk, neutral_pose=neutral, stiffness=0.0, damping=0.05)
    fly.colorize()
    root = fly.mjcf_root
    for j in root.joints:
        j.limited = mj.mjtLimited.mjLIMITED_FALSE
        j.stiffness[0] = 0.0
    for k in list(root.keys):
        root.delete(k)
    # ---- look: tint the (flat builtin) segment textures, eyes, wings, gloss
    tn, lf = np.array(look.tint), np.array(look.lift)
    for t in root.textures:
        t.rgb1 = tuple(np.clip(np.array(t.rgb1) * tn + lf, 0, 1))
        t.rgb2 = tuple(np.clip(np.array(t.rgb2) * tn + lf, 0, 1))
        t.markrgb = tuple(np.clip(np.array(t.markrgb) * tn + 0.7 * lf, 0, 1))
        t.random = t.random * look.speckle
    for m in root.materials:
        m.specular = look.specular
        m.shininess = look.shininess
        if m.name == "eye":
            m.rgba = look.eye
            m.specular, m.shininess = 0.9, 0.9
        elif m.name == "wing":
            m.rgba = look.wing
        elif m.name == "arista":
            m.rgba = tuple(np.clip(np.array(m.rgba[:3]) * tn * 1.4 + lf, 0, 1)) + (1.0,)
        elif m.name == "haltere":
            m.rgba = tuple(np.clip(np.array(m.rgba[:3]) * tn + lf, 0, 1)) + (1.0,)
    # ---- scale (body offsets and meshes; a head / abdomen factor on their subtrees)
    k = look.scale
    head_set, abd_set = set(), set()

    def walk(b, in_head: bool, in_abd: bool):
        for c in b.bodies:
            h = in_head or c.name == "c_head"
            a = in_abd or c.name.startswith("c_abdomen")
            if h:
                head_set.add(c.name)
            if a:
                abd_set.add(c.name)
            walk(c, h, a)

    walk(root.worldbody, False, False)
    abd = np.array(look.abdomen_scale)
    for b in root.bodies:
        if b.name == "world":
            continue
        par = b.parent.name
        f = np.full(3, k)
        if par in head_set:
            f = f * look.head_scale
        if par in abd_set:
            f = f * abd
        b.pos = tuple(np.array(b.pos) * f)
    for g in root.geoms:
        if g.type != mj.mjtGeom.mjGEOM_MESH:
            continue
        mesh = root.mesh(g.meshname)
        f = np.full(3, k)
        bname = g.parent.name
        if bname in head_set:
            f = f * look.head_scale
        if bname in abd_set:
            f = f * abd
        mesh.scale = tuple(np.array(mesh.scale) * f)
    for b in root.bodies:
        if b.name != "world":
            b.gravcomp = 1.0
    # ---- attire (unit-fly meshes, scaled like the segment they ride on)
    vis = dict(contype=0, conaffinity=0, group=1, mass=0.0)
    for tex, (img, kw) in garment_tex.items():
        A.add_texture(root, "tex_" + tex, img)
        A.add_textured_material(root, "mat_" + tex, "tex_" + tex, **kw)
    for gname, (body, md, mat) in garments.items():
        f = k * (look.head_scale if body == "c_head" else 1.0)
        md = A.MeshData(md.verts * f, md.faces, md.uv)
        A.add_mesh(root, "mesh_" + gname, md)
        root.body(body).add_geom(name="garment_" + gname, type=mj.mjtGeom.mjGEOM_MESH, meshname="mesh_" + gname,
                                 material="mat_" + mat, **vis)
    return root


class FlyRig:
    """Joint / mount indices of one posed fly in the compiled scene."""

    def __init__(self, m: mj.MjModel, name: str, look: FlyLook):
        self.name, self.look = name, look
        pre = name + "/"
        jq = lambda n: int(m.jnt_qposadr[m.joint(pre + n).id])  # noqa: E731
        self.leg_q = np.array([[jq(_joint_name(leg, d)) for d in DOFS] for leg in LEGS])
        self.head_q = np.array([jq(f"c_thorax-c_head-{a}") for a in AX3])
        self.wing_q = np.array([[jq(f"c_thorax-{s}_wing-{a}") for a in AX3] for s in "lr"])
        self.ant_q = np.array([[jq(f"c_head-{s}_pedicel-{a}") for a in AX3] for s in "lr"])
        self.mocap = int(m.body_mocapid[m.body(name + "_mount").id])
        self.thorax = m.body(pre + "c_thorax").id
        self.head = m.body(pre + "c_head").id
        self.tarsus5 = {leg: m.body(f"{pre}{leg}_tarsus5").id for leg in LEGS}
        self.feet_geoms = np.array([g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in
                                    {m.body(f"{pre}{leg}_tarsus{i}").id for leg in LEGS for i in (3, 4, 5)}])
        self.z_off = 0.0
        self.stand = np.zeros((6, 7))


class TempleStandoff:
    """The scene: build once, then ``pose(t)`` / ``render_frame(t)`` are pure
    functions of the scene time (deterministic)."""

    def __init__(self, seed: int = 0, shadows: bool = True):
        from flygym_demo.complex_terrain import PreprogrammedSteps

        self.seed = seed
        self.steps = PreprogrammedSteps()
        spec = mj.MjSpec()
        spec.modelname = NAME
        self._settings(spec, shadows)
        self._garments = self._make_garments()
        self._hall(spec)
        self._cast(spec)
        self._blade(spec)
        self._lights(spec, shadows)
        self.spec = spec
        self.model = spec.compile()
        self.data = mj.MjData(self.model)
        self._attach_rigs()
        self._calibrate()
        self._solve_ik()
        self.pose(0.0)

    # ------------------------------------------------------------ build
    def _settings(self, spec, shadows: bool) -> None:
        spec.compiler.fusestatic = False  # keep the flies' segment bodies (the attire rides on them)
        spec.option.gravity = (0.0, 0.0, 0.0)
        spec.option.timestep = 1e-3
        spec.stat.extent = 40.0
        spec.stat.center = (4.0, 0.0, 5.0)
        v = spec.visual
        v.map.znear = 0.0025  # x extent: 0.1 mm (close-ups at ~2 mm)
        v.map.zfar = 3.0
        v.map.shadowclip = 0.6
        v.map.shadowscale = 0.5
        v.quality.shadowsize = 4096 if shadows else 0
        v.quality.offsamples = 8
        v.global_.offwidth = 1920
        v.global_.offheight = 1088
        v.global_.fovy = 40.0
        v.headlight.ambient = (0.018, 0.021, 0.030)
        v.headlight.diffuse = (0.030, 0.035, 0.048)
        v.headlight.specular = (0.02, 0.02, 0.03)
        v.rgba.haze = (0.02, 0.025, 0.035, 1.0)

    def _make_garments(self) -> dict:
        """Shrink-wrap the attire on a probe fly (unit scale, unposed)."""
        from flygym.compose import NeuroMechFly

        pf = NeuroMechFly(name="probe")
        ps = pf.mjcf_root.copy()
        ps.compiler.fusestatic = False  # keep every segment body (a lone fly has no free joint)
        pm = ps.compile()
        pd = mj.MjData(pm)
        probe = A.BodyProbe(pm, pd)
        cloak = A.cloak_meshes(probe, self.seed)
        tunic = A.tunic_meshes(probe, self.seed)
        return {"cloak": cloak, "tunic": tunic}

    def _hall(self, spec) -> None:
        """An original stone hall: a slab floor, ashlar walls, two rows of columns, tall
        windows on one side with cool light shafts, a raised dais at the back with a
        step, a back doorway into a lit corridor, a dark ceiling."""
        wb = spec.worldbody
        vis = dict(contype=0, conaffinity=0, group=0, mass=0.0)
        s = self.seed
        A.add_texture(spec, "tex_floor", A.stone_floor_texture(s))
        A.add_texture(spec, "tex_wall", A.ashlar_texture(s))
        A.add_texture(spec, "tex_column", A.column_texture(s))
        A.add_texture(spec, "tex_window", A.window_texture())
        A.add_textured_material(spec, "floor", "tex_floor", rgba=(1, 1, 1, 1), specular=0.22, shininess=0.6,
                                reflectance=0.08, texrepeat=(4.0, 4.0))
        A.add_textured_material(spec, "dais", "tex_floor", rgba=(0.92, 0.94, 1.0, 1), specular=0.3,
                                shininess=0.5, reflectance=0.08)
        A.add_textured_material(spec, "wall", "tex_wall", rgba=(1, 1, 1, 1), specular=0.08)
        self.WALL_TILE = 11.0
        A.add_textured_material(spec, "column", "tex_column", rgba=(1, 1, 1, 1), specular=0.15)
        A.add_textured_material(spec, "window", "tex_window", rgba=(1, 1, 1, 1), emission=1.0)
        spec.add_material(name="ceiling", rgba=(0.03, 0.03, 0.035, 1), specular=0.0)
        spec.add_material(name="corridor", rgba=(0.46, 0.50, 0.58, 1), specular=0.05)
        # floor (a plane: the whole hall, and the corridor beyond the doorway)
        wb.add_geom(name="floor", type=mj.mjtGeom.mjGEOM_PLANE, size=(24.0, 18.0, 0.1), pos=(4.0, 0.0, FLOOR_Z),
                    material="floor", **vis)

        def box(name, lo, hi, mat, tile=4.0):
            lo, hi = np.array(lo, float), np.array(hi, float)
            h = (hi - lo) / 2
            A.add_mesh(spec, "mesh_" + name, A.box_mesh(*h, tile=self.WALL_TILE if mat == "wall" else tile))
            wb.add_geom(name=name, type=mj.mjtGeom.mjGEOM_MESH, meshname="mesh_" + name, pos=tuple((lo + hi) / 2),
                        material=mat, **vis)

        XB, XF, YW, ZC = 20.0, -16.0, 14.0, 20.0  # back wall, front wall, side walls, ceiling
        DW, DH = 2.1, 8.2  # doorway half width, top
        # the dais and its step
        box("dais", (DAIS_X0, -8.0, 0.0), (XB, 8.0, DAIS_Z), "dais", tile=6.0)
        box("dais_step", (DAIS_X0 - 0.7, -8.0, 0.0), (DAIS_X0, 8.0, DAIS_Z / 2), "dais", tile=6.0)
        # back wall with the doorway (a deep reveal)
        box("back_l", (XB, -YW, 0.0), (XB + 1.2, -DW, ZC), "wall")
        box("back_r", (XB, DW, 0.0), (XB + 1.2, YW, ZC), "wall")
        box("back_top", (XB, -DW, DH), (XB + 1.2, DW, ZC), "wall")
        # the corridor beyond (floor at the dais level, walls, an end wall lit from above)
        box("cor_floor", (XB, -DW - 1.0, 0.0), (XB + 13.0, DW + 1.0, DAIS_Z), "dais", tile=6.0)
        box("cor_l", (XB + 1.2, -DW - 1.6, 0.0), (XB + 13.0, -DW - 0.6, DH + 1.0), "corridor")
        box("cor_r", (XB + 1.2, DW + 0.6, 0.0), (XB + 13.0, DW + 1.6, DH + 1.0), "corridor")
        box("cor_end", (XB + 13.0, -DW - 1.6, 0.0), (XB + 14.0, DW + 1.6, DH + 1.0), "corridor")
        box("cor_top", (XB + 1.2, -DW - 1.6, DH + 1.0), (XB + 14.0, DW + 1.6, DH + 2.0), "ceiling")
        # side walls (the +y wall has tall window openings), front wall, ceiling
        self.window_x = (-9.0, -1.0, 7.0, 15.0)
        WX, WZ0, WZ1 = 1.3, 5.0, 16.0  # window half width, sill, head
        xs = [XF] + [v for x in self.window_x for v in (x - WX, x + WX)] + [XB + 1.2]
        for i in range(0, len(xs), 2):  # the piers between the windows (full height)
            box(f"side_p{i // 2}", (xs[i], YW, 0.0), (xs[i + 1], YW + 1.0, ZC), "wall")
        for i, x in enumerate(self.window_x):  # below the sill / above the head
            box(f"side_lo{i}", (x - WX, YW, 0.0), (x + WX, YW + 1.0, WZ0), "wall")
            box(f"side_hi{i}", (x - WX, YW, WZ1), (x + WX, YW + 1.0, ZC), "wall")
        box("side_m", (XF, -YW - 1.0, 0.0), (XB + 1.2, -YW, ZC), "wall")
        box("front", (XF - 1.0, -YW - 1.0, 0.0), (XF, YW + 1.0, ZC), "wall")
        box("ceiling", (XF - 1.0, -YW - 1.0, ZC), (XB + 1.2, YW + 1.0, ZC + 1.0), "ceiling")
        # a low plinth along the walls (a skirting course)
        box("plinth_p", (XF, YW - 0.5, 0.0), (XB, YW, 0.7), "column")
        box("plinth_m", (XF, -YW, 0.0), (XB, -YW + 0.5, 0.7), "column")
        # the windows: real openings (the daylight casts the column shadows through
        # them), a glowing pane of leaded glass just outside each (behind its light)
        A.add_mesh(spec, "mesh_window", panel_mesh(2 * WX, WZ1 - WZ0, 0.02))
        self.window_geoms = []
        for i, x in enumerate(self.window_x):
            wb.add_geom(name=f"window{i}", type=mj.mjtGeom.mjGEOM_MESH, meshname="mesh_window",
                        pos=(x, YW + 1.35, (WZ0 + WZ1) / 2), quat=panel_quat((0, -1, 0)), material="window", **vis)
            self.window_geoms.append(f"window{i}")
            box(f"sill{i}", (x - WX - 0.3, YW - 0.6, WZ0 - 0.35), (x + WX + 0.3, YW, WZ0), "column")
            box(f"lintel{i}", (x - WX - 0.3, YW - 0.45, WZ1), (x + WX + 0.3, YW, WZ1 + 0.45), "column")
        self.WIN = (YW, WZ0, WZ1)
        # columns: two rows; the last pair stands on the dais
        for i, x in enumerate((-10.0, -3.5, 3.0, 14.2)):
            z0 = DAIS_Z if x > DAIS_X0 else 0.0
            A.add_mesh(spec, f"mesh_column{i}", A.column_mesh(0.95, ZC - z0))
            for sy in (-1, 1):
                wb.add_geom(name=f"column{i}_{'mp'[sy > 0]}", type=mj.mjtGeom.mjGEOM_MESH,
                            meshname=f"mesh_column{i}", pos=(x, sy * 7.2, z0), material="column", **vis)

    def _garment_textures(self):
        s = self.seed
        return {
            "cloak": (A.cloak_texture(s), dict(rgba=(1, 1, 1, 1), specular=0.12, shininess=0.15)),
            "tunic": (A.tunic_texture(s), dict(rgba=(1, 1, 1, 1), specular=0.1, shininess=0.2)),
            "sash": (A.sash_texture(s), dict(rgba=(1, 1, 1, 1), specular=0.15, shininess=0.2)),
            # the lead little fly's sash: a muted rust (the one to watch)
            "sash_lead": (A.sash_texture(s, base=(0.62, 0.28, 0.16)), dict(rgba=(1, 1, 1, 1), specular=0.15,
                                                                           shininess=0.2)),
        }

    def _cast(self, spec) -> None:
        tex = self._garment_textures()
        cloak = {k: (b, md, "cloak") for k, (b, md) in self._garments["cloak"].items()}
        tunic = {"tunic": (*self._garments["tunic"]["tunic"], "tunic"),
                 "sash": (*self._garments["tunic"]["sash"], "sash")}
        self.looks = {"tall": TALL_LOOK}
        roots = {"tall": _make_fly("tall", TALL_LOOK, cloak, {"cloak": tex["cloak"]})}
        for i, lf in enumerate(LITTLE):
            look = _little_look(i)
            self.looks[lf.name] = look
            sash = tex["sash_lead"] if lf.name == "lead" else tex["sash"]
            roots[lf.name] = _make_fly(lf.name, look, tunic, {"tunic": tex["tunic"], "sash": sash})
        for name, root in roots.items():
            mount = spec.worldbody.add_body(name=name + "_mount", mocap=True, pos=(0.0, 0.0, -20.0))
            site = mount.add_site(name=name + "_site")
            spec.attach(root, prefix=name + "/", site=site)

    def _blade(self, spec) -> None:
        """The handle (a small ridged metal cylinder) and the energy blade (a white
        core in two blue glow layers) on a mocap body, with its point light."""
        vis = dict(contype=0, conaffinity=0, group=0, mass=0.0)
        A.add_texture(spec, "tex_grip", A.grip_texture())
        A.add_textured_material(spec, "grip", "tex_grip", rgba=(1, 1, 1, 1), specular=0.9, shininess=0.85)
        spec.add_material(name="emitter", rgba=(0.55, 0.57, 0.62, 1), specular=1.0, shininess=0.95)
        spec.add_material(name="blade_core", rgba=(0.96, 0.98, 1.0, 0.0), emission=1.0, specular=0.0)
        spec.add_material(name="blade_glow", rgba=(0.30, 0.62, 1.0, 0.0), emission=1.0, specular=0.0)
        spec.add_material(name="blade_halo", rgba=(0.16, 0.42, 1.0, 0.0), emission=1.0, specular=0.0)
        b = spec.worldbody.add_body(name="blade", mocap=True, pos=(0.0, 0.0, -20.0))
        self.HANDLE_L, self.HANDLE_R = 0.62, 0.075
        self.BLADE_L = 4.1
        A.add_mesh(spec, "mesh_grip", A.cylinder_mesh(self.HANDLE_R, self.HANDLE_L, 24))
        b.add_geom(name="grip", type=mj.mjtGeom.mjGEOM_MESH, meshname="mesh_grip",
                   pos=(0.0, 0.0, -self.HANDLE_L / 2), material="grip", **vis)
        b.add_geom(name="emitter", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(self.HANDLE_R * 1.2, 0.035, 0),
                   pos=(0.0, 0.0, self.HANDLE_L / 2 + 0.02), material="emitter", **vis)
        b.add_geom(name="pommel", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(self.HANDLE_R * 1.1, 0.03, 0),
                   pos=(0.0, 0.0, -self.HANDLE_L / 2 - 0.02), material="emitter", **vis)
        for nm, r, mat in (("blade_core", 0.032, "blade_core"), ("blade_glow", 0.062, "blade_glow"),
                           ("blade_halo", 0.13, "blade_halo")):
            b.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_CAPSULE, size=(r, 0.01, 0),
                       pos=(0.0, 0.0, self.HANDLE_L / 2 + 0.05), material=mat, **vis)
        b.add_light(name="blade_light", type=mj.mjtLightType.mjLIGHT_POINT, pos=(0.0, 0.0, 1.5),
                    diffuse=(0.0, 0.0, 0.0), specular=(0.0, 0.0, 0.0), ambient=(0.0, 0.0, 0.0),
                    attenuation=(1.0, 0.0, 0.045), castshadow=False, active=True)

    def _lights(self, spec, shadows: bool) -> None:
        """Every spot light casts shadows: on macOS' OpenGL a non-shadowing spot light
        blacks out every fragment behind its plane (NaN), a shadowing one does not."""
        wb = spec.worldbody
        YW, WZ0, WZ1 = self.WIN
        self.sun_dirs = []

        def spot(name, pos, tgt, col, cutoff, exp=1.0):
            pos, tgt = np.array(pos, float), np.array(tgt, float)
            wb.add_light(name=name, type=mj.mjtLightType.mjLIGHT_SPOT, pos=tuple(pos), dir=tuple(tgt - pos),
                         diffuse=col, specular=(0.08, 0.10, 0.13), cutoff=cutoff, exponent=exp,
                         castshadow=bool(shadows))

        # cool daylight from inside each window's head, raking down onto the floor
        for i, x in enumerate(self.window_x[:3]):
            p0 = (x, YW + 0.55, WZ1 - 0.3)
            p1 = (x + 2.6, -1.8, 0.0)
            spot(f"window_sun{i}", p0, p1, (0.46, 0.58, 0.74), 15.0, 3.0)
            self.sun_dirs.append((np.array((x, YW, (WZ0 + WZ1) / 2)), np.array(p1)))
        # the corridor: a pale cool point light at its end (the doorway glows; a rim on
        # whoever comes through)
        wb.add_light(name="corridor", type=mj.mjtLightType.mjLIGHT_POINT, pos=(30.5, 0.0, 7.0),
                     diffuse=(0.62, 0.72, 0.88), specular=(0.1, 0.12, 0.16), attenuation=(1.0, 0.0, 0.012),
                     castshadow=False)
        # a dim cool fill (directional, from the front, high)
        wb.add_light(name="fill", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(0.0, 0.0, 30.0),
                     dir=(0.55, 0.25, -0.8), diffuse=(0.045, 0.055, 0.08), specular=(0.0, 0.0, 0.0),
                     castshadow=False)
        # the shot key: a film set light, re-placed per shot (see Shot.key)
        spot("key", (0.0, 0.0, 30.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 20.0, 1.0)

    # ------------------------------------------------------------ attach / calibrate
    def _attach_rigs(self) -> None:
        m = self.model
        self.rigs = {"tall": FlyRig(m, "tall", TALL_LOOK)}
        for lf in LITTLE:
            self.rigs[lf.name] = FlyRig(m, lf.name, self.looks[lf.name])
        self.blade_mocap = int(m.body_mocapid[m.body("blade").id])
        self.blade_light = m.light("blade_light").id
        self.key_light = m.light("key").id
        self.window_ids = np.array([m.geom(n).id for n in self.window_geoms])
        self.g_blade = {n: m.geom(n).id for n in ("blade_core", "blade_glow", "blade_halo")}
        self.m_blade = {n: m.material(n).id for n in ("blade_core", "blade_glow", "blade_halo")}
        self.blade_r = {n: float(m.geom_size[g, 0]) for n, g in self.g_blade.items()}
        self.mat_rgba0 = {n: m.mat_rgba[i].copy() for n, i in self.m_blade.items()}
        if hasattr(m, "geom_sameframe"):
            for g in self.g_blade.values():
                m.geom_sameframe[g] = 0

    def _stand_angles(self) -> np.ndarray:
        return np.array([self.steps.get_joint_angles(leg, math.pi) for leg in LEGS])

    def _calibrate(self) -> None:
        """Stance heights, joint signs (head up, head left, wing spread / lift) and
        the stride of the recorded step, measured on the compiled model."""
        m = self.model
        d = mj.MjData(m)
        self._scratch = d
        stand = self._stand_angles()
        for rig in self.rigs.values():
            rig.stand = stand.copy()
        # stance heights: the lowest foot vertex at z = 0 with the mount at the origin
        for rig in self.rigs.values():
            d.qpos[:] = m.qpos0
            d.mocap_pos[rig.mocap] = (0, 0, 0)
            d.mocap_quat[rig.mocap] = (1, 0, 0, 0)
            d.qpos[rig.leg_q.ravel()] = stand.ravel()
            mj.mj_kinematics(m, d)
            zmin = min(self._geom_zmin(m, d, g) for g in rig.feet_geoms)
            rig.z_off = -zmin
        # signs, on the tall fly
        rig = self.rigs["tall"]
        d.qpos[:] = m.qpos0
        d.qpos[rig.leg_q.ravel()] = stand.ravel()
        mj.mj_kinematics(m, d)
        eye = m.body("tall/l_eye").id
        e0 = d.xpos[eye].copy()
        d.qpos[rig.head_q[1]] = 0.3
        mj.mj_kinematics(m, d)
        self.head_up = -1.0 if d.xpos[eye][2] < e0[2] else 1.0
        d.qpos[rig.head_q[1]] = 0.0
        d.qpos[rig.head_q[0]] = 0.3
        mj.mj_kinematics(m, d)
        hc = d.xpos[rig.head]
        self.head_left = 1.0 if (d.xpos[eye] - hc)[0] < (e0 - hc)[0] else -1.0
        # (left eye swings back when the head yaws left, i.e. +y)
        d.qpos[rig.head_q[0]] = 0.0
        self.wing_sign = {}
        for si, s in enumerate("lr"):
            wid = m.body(f"tall/{s}_wing").id
            gid = [g for g in range(m.ngeom) if m.geom_bodyid[g] == wid][0]

            def tip():
                mj.mj_kinematics(m, d)
                return self._geom_centroid(m, d, gid)

            t0 = tip()
            res = {}
            for a in range(3):
                d.qpos[rig.wing_q[si, a]] = 0.4
                res[a] = tip() - t0
                d.qpos[rig.wing_q[si, a]] = 0.0
            ysign = 1.0 if s == "l" else -1.0
            # spread: the axis that moves the wing outward the most; lift: the most upward
            sp = max(range(3), key=lambda a: abs(res[a][1]))
            li = max((a for a in range(3) if a != sp), key=lambda a: abs(res[a][2]))
            self.wing_sign[s] = dict(spread=(sp, 1.0 if res[sp][1] * ysign > 0 else -1.0),
                                     lift=(li, 1.0 if res[li][2] > 0 else -1.0))
        # the stride of one recorded step (fly units, measured on the tall fly, / scale)
        ph = np.linspace(0, 2 * math.pi, 181)
        strides = []
        for li, leg in enumerate(LEGS):
            if leg == "rf":
                continue
            xs = []
            for p in ph:
                d.qpos[rig.leg_q[li]] = self.steps.get_joint_angles(leg, p)
                mj.mj_kinematics(m, d)
                xs.append(d.xpos[rig.tarsus5[leg]][0])
            d.qpos[rig.leg_q[li]] = stand[li]
            xs = np.array(xs)
            sw_end = self.steps.swing_period[leg][1]
            i0 = int(np.argmin(np.abs(ph - sw_end)))
            strides.append(xs[i0] - xs[-1])
        self.stride_unit = float(np.mean(strides)) / TALL_LOOK.scale

    @staticmethod
    def _geom_verts(m, d, g) -> np.ndarray:
        mid = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]

    def _geom_zmin(self, m, d, g) -> float:
        return float(self._geom_verts(m, d, g)[:, 2].min())

    def _geom_centroid(self, m, d, g) -> np.ndarray:
        return self._geom_verts(m, d, g).mean(0)

    def _ik(self, rig: FlyRig, leg: str, target_fly, q0: np.ndarray, iters: int = 300) -> tuple[np.ndarray, float]:
        """Damped least squares on one leg (7 DoFs), the mount at the origin:
        ``target_fly`` is the tarsus-5 point in the fly frame."""
        m, d = self.model, self._scratch
        li = LEGS.index(leg)
        d.qpos[:] = m.qpos0
        d.mocap_pos[rig.mocap] = (0, 0, 0)
        d.mocap_quat[rig.mocap] = (1, 0, 0, 0)
        d.qpos[rig.leg_q.ravel()] = rig.stand.ravel()
        q = q0.copy()
        dadr = np.array([m.jnt_dofadr[np.flatnonzero(m.jnt_qposadr == a)[0]] for a in rig.leg_q[li]])
        jac = np.zeros((3, m.nv))
        bi = rig.tarsus5[leg]
        err = math.inf
        tgt = np.asarray(target_fly, float)
        for _ in range(iters):
            d.qpos[rig.leg_q[li]] = q
            mj.mj_kinematics(m, d)
            mj.mj_comPos(m, d)
            e = tgt - d.xpos[bi]
            err = float(np.linalg.norm(e))
            if err < 2e-4:
                break
            mj.mj_jacBody(m, d, jac, None, bi)
            J = jac[:, dadr]
            Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(3))
            dq = 0.5 * Jp @ e + 0.03 * (np.eye(7) - Jp @ J) @ (q0 - q)
            n = float(np.linalg.norm(dq))
            if n > 0.12:
                dq *= 0.12 / n
            q += dq
        return q, err

    def _solve_ik(self) -> None:
        rig = self.rigs["tall"]
        k = TALL_LOOK.scale
        li = LEGS.index("rf")
        q0 = rig.stand[li]
        self.q_hold, e1 = self._ik(rig, "rf", np.array(self.HOLD_TIP) * k, q0)
        self.q_present, e2 = self._ik(rig, "rf", np.array(self.PRESENT_TIP) * k, self.q_hold)
        self.ik_err = {"hold": e1, "present": e2}

    # the tall fly's right front tarsus, fly frame / scale: the handle held low at its
    # side, then brought forward and up
    HOLD_TIP = (1.12, -0.78, 0.50)
    PRESENT_TIP = (1.42, -0.80, 0.98)
    HOLD_DIR = (0.62, -0.30, -0.72)  # the emitter end of the handle (fly frame)
    PRESENT_DIR = (0.30, -0.42, 0.86)

    # ------------------------------------------------------------ the script
    def _tall_walk(self, t: float) -> tuple[float, float, float]:
        """(distance walked, step phase, step magnitude) of the tall fly at t."""
        D = TALL_X0 - TALL_X1
        ta, td = 0.7, 1.1
        T = T_WALK1 - T_WALK0
        vmax = D / (T - (ta + td) / 2)

        def dist(tt: float) -> float:
            u = tt - T_WALK0
            if u <= 0:
                return 0.0
            if u < ta:
                return vmax * u * u / (2 * ta)
            if u < T - td:
                return vmax * (ta / 2 + (u - ta))
            if u < T:
                w = T - u
                return D - vmax * w * w / (2 * td)
            return D

        s = dist(t)
        u = t - T_WALK0
        v = vmax if ta <= u <= T - td else (vmax * u / ta if 0 < u < ta else (vmax * (T - u) / td if T - td < u < T else 0.0))
        mag = min(1.0, v / vmax) ** 0.7 if v > 0 else 0.0
        stride = self.stride_unit * TALL_LOOK.scale
        f0 = vmax / stride  # steps / s at full magnitude
        # the phase advances at f0 while walking (the step shrinks as the fly slows)
        tt = min(max(u, 0.0), T)
        phase = 2 * math.pi * f0 * tt
        return s, phase, mag

    def _legs_walking(self, rig: FlyRig, phase: float, mag: float, skip=()) -> np.ndarray:
        q = rig.stand.copy()
        if mag <= 0:
            return q
        for li, leg in enumerate(LEGS):
            if leg in skip:
                continue
            p = phase + (0.0 if leg in TRIPOD_A else math.pi)
            q[li] = self.steps.get_joint_angles(leg, p, mag)
        return q

    def _set_fly(self, rig: FlyRig, pos, yaw: float, legs: np.ndarray, head=(0.0, 0.0, 0.0),
                 wings=None, ant=None, pitch: float = 0.0, ground: float = FLOOR_Z) -> None:
        d = self.data
        q = _quat_yaw(yaw)
        if pitch:
            q = quat_mul(q, quat_axis_angle((0, 1, 0), -pitch))
        d.mocap_pos[rig.mocap] = (pos[0], pos[1], ground + rig.z_off + (pos[2] if len(pos) > 2 else 0.0))
        d.mocap_quat[rig.mocap] = q
        d.qpos[rig.leg_q.ravel()] = legs.ravel()
        yaw_h, up, roll = head
        d.qpos[rig.head_q] = (self.head_left * yaw_h, self.head_up * up, roll)
        wq = np.zeros((2, 3))
        if wings is not None:
            for si, s in enumerate("lr"):
                (a1, s1), (a2, s2) = self.wing_sign[s]["spread"], self.wing_sign[s]["lift"]
                wq[si, a1] += s1 * wings[0]
                wq[si, a2] += s2 * wings[1]
        d.qpos[rig.wing_q.ravel()] = wq.ravel()
        d.qpos[rig.ant_q.ravel()] = (np.zeros((2, 3)) if ant is None else ant).ravel()

    def tall_state(self, t: float) -> dict:
        s, phase, mag = self._tall_walk(t)
        x = TALL_X0 - s
        return dict(pos=(x, TALL_Y), yaw=math.pi, phase=phase, mag=mag)

    def tall_head_world(self) -> np.ndarray:
        return self.data.xpos[self.rigs["tall"].head].copy()

    def _pose_tall(self, t: float) -> None:
        rig = self.rigs["tall"]
        st = self.tall_state(t)
        legs = self._legs_walking(rig, st["phase"], st["mag"], skip=("rf",))
        # the handle leg: held low; 10.55-10.97 brought forward (a tiny, deliberate move)
        a = _smoother((t - T_HANDLE0) / (T_HANDLE1 - T_HANDLE0))
        legs[LEGS.index("rf")] = (1 - a) * self.q_hold + a * self.q_present
        # the head: level while walking; then lowered a touch toward the little flies,
        # very slowly (8.5-10.5 barely moves)
        down = 0.10 * _smooth((t - 5.0) / 1.2) + 0.03 * _smooth((t - 8.6) / 1.9)
        yaw_h = 0.05 * _smooth((t - 5.2) / 1.5)  # toward the lead
        # the wings folded tight under the cloak
        self._set_fly(rig, st["pos"], st["yaw"], legs, head=(yaw_h, -down, 0.0), wings=(-0.18, -0.06),
                      ground=DAIS_Z)
        self._handle_blend = a

    def _little_state(self, i: int, t: float) -> dict:
        lf = LITTLE[i]
        x, y = lf.xy
        if lf.name == "lead":
            u = _smoother((t - T_STEP0) / (T_STEP1 - T_STEP0))
            x = lf.xy[0] + u * (LEAD_STEP_TO[0] - lf.xy[0])
            y = lf.xy[1] + u * (LEAD_STEP_TO[1] - lf.xy[1])
        # facing the tall fly (its head) once noticed
        tx, ty = TALL_X1 - 1.6, TALL_Y
        yaw_face = math.atan2(ty - y, tx - x)
        yaw0 = math.radians(lf.yaw0)
        dyaw = (yaw_face - yaw0 + math.pi) % (2 * math.pi) - math.pi
        t_turn = T_NOTICE + lf.delay + 0.18  # the head goes first
        dur = 0.45 + 0.5 * abs(dyaw) / math.pi
        u = _smoother((t - t_turn) / dur)
        yaw = yaw0 + u * dyaw
        return dict(x=x, y=y, yaw=yaw, dyaw=dyaw, t_turn=t_turn, dur=dur, u=u, yaw_face=yaw_face)

    def _pose_little(self, i: int, t: float) -> None:
        lf = LITTLE[i]
        rig = self.rigs[lf.name]
        st = self._little_state(i, t)
        x, y, yaw = st["x"], st["y"], st["yaw"]
        rng = np.random.default_rng(100 + i)
        # the notice: the head snaps toward the tall fly first, the body follows
        th = T_NOTICE + lf.delay
        head_yaw = 0.0
        if t > th:
            want = st["yaw_face"] - yaw
            want = (want + math.pi) % (2 * math.pi) - math.pi
            head_yaw = float(np.clip(want, -0.55, 0.55)) * _smooth((t - th) / 0.16)
        # a shuffle while turning (small quick steps), a walk for the lead's step
        mag, phase = 0.0, 0.0
        if st["t_turn"] < t < st["t_turn"] + st["dur"]:
            w = (t - st["t_turn"]) / st["dur"]
            mag = 0.55 * math.sin(math.pi * w)
            phase = 2 * math.pi * 7.0 * (t - st["t_turn"]) + rng.uniform(0, 6.3)
        if lf.name == "lead" and T_STEP0 < t < T_STEP1:
            w = (t - T_STEP0) / (T_STEP1 - T_STEP0)
            mag = 0.8 * math.sin(math.pi * w) ** 0.6
            phase = 2 * math.pi * 3.2 * (t - T_STEP0)
        legs = self._legs_walking(rig, phase, mag)
        # look up at the tall fly (all of them a little; the lead properly, after its step)
        up = 0.12 * _smooth((t - th - 0.2) / 0.5)
        pitch = 0.0
        roll = 0.0
        ant = np.zeros((2, 3))
        if lf.name == "lead":
            up += 0.30 * _smooth((t - T_LOOKUP) / 0.45)
            pitch = math.radians(5.0) * _smooth((t - T_LOOKUP) / 0.45)
            # close-up: a slow head tilt, antenna micro-motion, then stillness
            roll = math.radians(13.0) * _smooth((t - T_TILT) / 0.35) - math.radians(4.0) * _smooth((t - 8.05) / 0.4)
            for tc, amp in ((7.72, 0.22), (7.95, -0.15), (8.28, 0.12)):
                env = math.exp(-((t - tc) / 0.07) ** 2)
                ant[0, 1] += amp * env
                ant[1, 1] += amp * 0.8 * env
            head_yaw += 0.04 * math.sin(2 * math.pi * 0.6 * (t - 7.0)) * _smooth((t - 7.0) / 0.3) * (1 - _smooth((t - 8.3) / 0.2))
        else:
            # idle micro-motion before the notice (tiny antenna / head flickers)
            if t < th:
                ph = rng.uniform(0, 6.3, 3)
                head_yaw += 0.05 * math.sin(1.3 * t + ph[0])
                ant[:, 1] = 0.08 * math.sin(2.1 * t + ph[1])
        # stillness before the ignition: freeze all idle motion (already none after th)
        # the recoil: a short backward jerk (away from the tall fly), nose up, wings flick
        t_r = (T_LEAD_RECOIL if lf.name == "lead" else T_RECOIL + 0.035 * rng.random())
        wings = None
        if t > t_r:
            u = t - t_r
            back = lf.recoil * (1 - math.exp(-u / 0.045)) * (1 + 0.12 * math.exp(-u / 0.12) * math.sin(u * 40))
            dirb = np.array([math.cos(st["yaw_face"]), math.sin(st["yaw_face"])])
            x, y = x - back * dirb[0], y - back * dirb[1]
            pitch += math.radians(9.0) * math.exp(-u / 0.25) * (1 - math.exp(-u / 0.03)) + math.radians(3.0) * (1 - math.exp(-u / 0.2))
            up += 0.18 * (1 - math.exp(-u / 0.05))
            flick = math.exp(-((u - 0.07) / 0.06) ** 2)
            wings = (0.95 * flick, 0.55 * flick)
            ant[:, 1] -= 0.35 * (1 - math.exp(-u / 0.04))
        elif lf.name == "lead" and T_RECOIL < t <= t_r:
            pass  # the lead: frozen (everyone else moved)
        self._set_fly(rig, (x, y), yaw, legs, head=(head_yaw, up, roll), wings=wings, ant=ant, pitch=pitch)

    def blade_progress(self, t: float) -> float:
        """0 before ignition, 0..1 while it grows, 1 after."""
        if t < T_IGNITE:
            return 0.0
        return _smoother((t - T_IGNITE + 0.35 * IGNITE_S) / IGNITE_S)

    def blade_light_on(self, t: float) -> bool:
        return t >= T_IGNITE

    def _pose_blade(self, t: float) -> None:
        m, d = self.model, self.data
        rig = self.rigs["tall"]
        a = getattr(self, "_handle_blend", 0.0)
        dfly = (1 - a) * np.array(self.HOLD_DIR) + a * np.array(self.PRESENT_DIR)
        yaw = math.pi
        c, s = math.cos(yaw), math.sin(yaw)
        dw = np.array([c * dfly[0] - s * dfly[1], s * dfly[0] + c * dfly[1], dfly[2]])
        dw /= np.linalg.norm(dw)
        tip = d.xpos[rig.tarsus5["rf"]]
        # the grip sits in the tarsus (a little below its middle)
        d.mocap_pos[self.blade_mocap] = tip - dw * 0.06
        d.mocap_quat[self.blade_mocap] = _quat_from_z(dw)
        p = self.blade_progress(t)
        L = max(p * self.BLADE_L, 1e-3)
        base = self.HANDLE_L / 2 + 0.04
        flick = 1.0 + (0.035 * math.sin(2 * math.pi * 23.0 * t) + 0.02 * math.sin(2 * math.pi * 7.3 * t) if p >= 1 else 0.0)
        for n, g in self.g_blade.items():
            r = self.blade_r[n] * (1.0 if n == "blade_core" else flick)
            m.geom_size[g, 0] = r
            m.geom_size[g, 1] = L / 2
            m.geom_pos[g] = (0.0, 0.0, base + L / 2)
            rgba = self.mat_rgba0[n].copy()
            rgba[3] = 0.0 if p <= 0 else {"blade_core": 1.0, "blade_glow": 0.55, "blade_halo": 0.20}[n]
            m.mat_rgba[self.m_blade[n]] = rgba
        li = self.blade_light
        if self.blade_light_on(t):
            k = (0.6 + 0.4 * p) * flick
            # the ignition flare: brighter for an instant
            k *= 1.0 + 0.8 * math.exp(-max(t - T_IGNITE, 0.0) / 0.12)
            m.light_diffuse[li] = np.array((0.42, 0.66, 1.0)) * 1.7 * k
            m.light_specular[li] = np.array((0.4, 0.6, 1.0)) * k
            m.light_pos[li] = (0.0, 0.0, base + 0.5 * self.BLADE_L)
        else:
            m.light_diffuse[li] = 0.0
            m.light_specular[li] = 0.0

    def _pose_key(self, t: float) -> None:
        m, li = self.model, self.key_light
        sh = shot_at(t)
        if sh.key is None:
            m.light_diffuse[li] = 0.0
            m.light_specular[li] = 0.0
            return
        pos, tgt, col, cut = sh.key
        pos, tgt = np.array(pos, float), np.array(tgt, float)
        v = tgt - pos
        m.light_pos[li] = pos
        m.light_dir[li] = v / np.linalg.norm(v)
        m.light_diffuse[li] = col
        m.light_specular[li] = np.array(col) * 0.5
        m.light_cutoff[li] = cut

    def pose(self, t: float) -> None:
        """Write every pose / prop / light for scene time t and run the kinematics."""
        m, d = self.model, self.data
        self._pose_key(t)
        self._pose_tall(t)
        for i in range(len(LITTLE)):
            self._pose_little(i, t)
        mj.mj_kinematics(m, d)
        self._pose_blade(t)
        mj.mj_forward(m, d)

    # ------------------------------------------------------------ camera / render
    def camera(self, t: float) -> tuple[mj.MjvCamera, Shot]:
        sh = shot_at(t)
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        eye, tgt = np.array(sh.eye, float), np.array(sh.target, float)
        v = tgt - eye
        cam.lookat[:] = tgt
        cam.distance = float(np.linalg.norm(v))
        cam.azimuth = math.degrees(math.atan2(v[1], v[0]))
        cam.elevation = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
        return cam, sh


# ---------------------------------------------------------------------------
# post-process (screen space): bloom, exposure lift, grade, grain, vignette, bars
# ---------------------------------------------------------------------------


class Post:
    def __init__(self, width: int, height: int, letterbox: bool = True, grain: float = 3.0):
        self.W, self.H = width, height
        self.letterbox = letterbox
        self.grain = grain
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        r2 = ((xx - width / 2) / (width / 2)) ** 2 + ((yy - height / 2) / (height / 2)) ** 2
        self.vig = (1.0 - 0.32 * np.clip(r2 - 0.15, 0, None) ** 1.2)[..., None].astype(np.float32)
        self.bar = int(round((height - width / 2.39) / 2)) if letterbox else 0
        self.caption: tuple[str, float, float] | None = None  # (text, t_start, t_end)
        self._cap_img = None  # (key, RGBA uint8 image) cache of the rendered caption
        self.rays = None  # (mask (H, W) float32, (dx, dy) px): the window light streaks

    def _god_rays(self, cv2, x: np.ndarray) -> np.ndarray:
        """Streaks of the window glow along the daylight's screen-space direction
        (a cheap stand-in for light scattering in the air of the hall)."""
        mask, (dx, dy) = self.rays
        W, H = self.W, self.H
        w4, h4 = max(W // 4, 1), max(H // 4, 1)
        src = cv2.resize(mask, (w4, h4), interpolation=cv2.INTER_AREA)
        acc = np.zeros_like(src)
        n = 28
        L = math.hypot(dx, dy)
        if L < 1e-3:
            return x
        ux, uy = dx / L, dy / L
        reach = 0.55 * H / 4
        for i in range(n):
            f = i / (n - 1)
            M = np.float32([[1, 0, ux * reach * f], [0, 1, uy * reach * f]])
            acc += math.exp(-2.6 * f) * cv2.warpAffine(src, M, (w4, h4), flags=cv2.INTER_LINEAR,
                                                        borderMode=cv2.BORDER_CONSTANT)
        acc = cv2.GaussianBlur(acc, (0, 0), 2.0) / n
        up = cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)
        return x + 0.9 * up[..., None] * np.array((0.55, 0.72, 0.98), np.float32)

    def __call__(self, frame: np.ndarray, t: float, k: int) -> np.ndarray:
        import cv2

        if t >= T_BLACK:
            return np.zeros_like(frame)
        W, H = self.W, self.H
        x = frame.astype(np.float32)
        ig = t - T_IGNITE
        lit = ig >= 0
        # exposure: subdued before, a brief lift at the ignition
        exp = 1.0 + (0.55 * math.exp(-ig / 0.16) + 0.08 * math.exp(-ig / 0.8) if lit else 0.0)
        x *= exp
        if self.rays is not None:
            x = self._god_rays(cv2, x)
        # bloom: bright pass blurred at 1/4 and 1/8 size (the blade and the windows glow)
        thr = 150.0 if not lit else 170.0
        bright = np.maximum(x - thr, 0.0)
        s4 = cv2.resize(bright, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        g4 = cv2.GaussianBlur(s4, (0, 0), 3.0 * W / 1920 * 4)
        g8 = cv2.GaussianBlur(cv2.resize(s4, (W // 8, H // 8), interpolation=cv2.INTER_AREA), (0, 0), 3.0 * W / 1920 * 4)
        glow = cv2.resize(g4, (W, H), interpolation=cv2.INTER_LINEAR) + 1.6 * cv2.resize(g8, (W, H), interpolation=cv2.INTER_LINEAR)
        tint = np.array((0.70, 0.88, 1.15), np.float32) if lit else np.array((0.85, 0.95, 1.05), np.float32)
        x = x + (1.1 if lit else 0.7) * glow * tint
        # grade: cool shadows, a gentle S-curve
        x = x / 255.0
        x = np.clip(x, 0.0, None)
        x = x * np.array((0.95, 1.0, 1.06), np.float32) + np.array((0.0, 0.004, 0.012), np.float32)
        x = x / (1.0 + 0.35 * x)  # soft shoulder
        x = np.power(np.maximum(x * 1.28, 0.0), 1.12)  # deeper shadows
        x = np.clip(x, 0.0, 1.0)
        x = x * x * (3.0 - 2.0 * x) * 0.35 + x * 0.65
        x = x * self.vig
        if self.grain > 0:
            rng = np.random.default_rng(1000 + k)
            n = rng.normal(0.0, self.grain / 255.0, (H // 2, W // 2)).astype(np.float32)
            x = x + cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
        out = np.clip(x * 255.0, 0, 255).astype(np.uint8)
        if self.bar > 0:
            out[:self.bar] = 0
            out[H - self.bar:] = 0
        if self.caption is not None:
            self._draw_caption(out, t)
        return out

    # subtitles: white text with a soft shadow, centred in the bottom letterbox bar
    # (or in the lower third without the bar), with a short fade in and out
    CAPTION_FADE_S = 0.18

    def _caption_rgba(self, text: str) -> np.ndarray:
        from PIL import Image, ImageDraw, ImageFont

        W, H = self.W, self.H
        size = max(12, int(round(0.030 * H)))
        font = None
        for f in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                  "/Library/Fonts/Arial.ttf", "DejaVuSans.ttf"):
            try:
                font = ImageFont.truetype(f, size)
                break
            except OSError:
                continue
        if font is None:
            font = ImageFont.load_default()
        # wrap to ~80 % of the width
        words, lines, cur = text.split(), [], ""
        probe = ImageDraw.Draw(Image.new("L", (1, 1)))
        for w in words:
            t = (cur + " " + w).strip()
            if probe.textlength(t, font=font) > 0.8 * W and cur:
                lines.append(cur)
                cur = w
            else:
                cur = t
        if cur:
            lines.append(cur)
        lh = int(size * 1.25)
        img = Image.new("RGBA", (W, lh * len(lines) + size // 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        for i, ln in enumerate(lines):
            x = (W - probe.textlength(ln, font=font)) / 2
            y = i * lh
            for dx, dy in ((2, 2), (1, 1)):
                d.text((x + dx, y + dy), ln, font=font, fill=(0, 0, 0, 200))
            d.text((x, y), ln, font=font, fill=(245, 242, 235, 255))
        return np.asarray(img)

    def _draw_caption(self, out: np.ndarray, t: float) -> None:
        text, t0, t1 = self.caption
        if not text or t < t0 or t > t1:
            return
        f = self.CAPTION_FADE_S
        a = min(1.0, (t - t0) / f, (t1 - t) / f)
        if a <= 0:
            return
        key = (text, self.W, self.H)
        if self._cap_img is None or self._cap_img[0] != key:
            self._cap_img = (key, self._caption_rgba(text))
        img = self._cap_img[1]
        h = img.shape[0]
        H = self.H
        if self.bar > 0:  # centred in the bottom bar
            y0 = H - self.bar + max(0, (self.bar - h) // 2)
        else:  # lower third
            y0 = int(0.84 * H) - h // 2
        y0 = max(0, min(y0, H - h))
        region = out[y0:y0 + h].astype(np.float32)
        alpha = (img[..., 3:4].astype(np.float32) / 255.0) * a
        region = region * (1 - alpha) + img[..., :3].astype(np.float32) * alpha
        out[y0:y0 + h] = np.clip(region, 0, 255).astype(np.uint8)


class SceneRenderer:
    """Renders graded frames of a ``TempleStandoff`` (the raw MuJoCo frame, a
    segmentation pass for the window glow, the post-process)."""

    def __init__(self, scene: TempleStandoff, width: int, height: int, letterbox: bool = True,
                 grain: float = 3.0, rays: bool = True):
        self.scene = scene
        self.W, self.H = width, height
        self.r = mj.Renderer(scene.model, height, width)
        self.r.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 1
        self.r.scene.flags[mj.mjtRndFlag.mjRND_SKYBOX] = 0
        self.post = Post(width, height, letterbox, grain)
        self.rays = rays
        self.opt = mj.MjvOption()

    def _project(self, cam: mj.MjvCamera, fovy: float, p) -> tuple[float, float, float]:
        az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
        f = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        eye = np.array(cam.lookat) - cam.distance * f
        right = np.cross(f, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        v = np.asarray(p, float) - eye
        z = float(v @ f)
        th = math.tan(math.radians(fovy) / 2)
        xn = float(v @ right) / max(z, 1e-6) / (th * self.W / self.H)
        yn = float(v @ up) / max(z, 1e-6) / th
        return (self.W / 2 * (1 + xn), self.H / 2 * (1 - yn), z)

    def frame(self, t: float, k: int) -> np.ndarray:
        sc = self.scene
        if t >= T_BLACK:
            return np.zeros((self.H, self.W, 3), np.uint8)
        sc.pose(t)
        cam, sh = sc.camera(t)
        sc.model.vis.global_.fovy = sh.fovy
        self.r.update_scene(sc.data, camera=cam, scene_option=self.opt)
        raw = self.r.render()
        self.post.rays = None
        if self.rays:
            self.r.enable_segmentation_rendering()
            seg = self.r.render()
            self.r.disable_segmentation_rendering()
            win = np.isin(seg[..., 0], sc.window_ids) & (seg[..., 1] == int(mj.mjtObj.mjOBJ_GEOM))
            if win.any():
                lum = raw.astype(np.float32).mean(2) * win
                a = self._project(cam, sh.fovy, sc.sun_dirs[1][0])
                b = self._project(cam, sh.fovy, sc.sun_dirs[1][1])
                if a[2] > 0 and b[2] > 0:
                    self.post.rays = (lum, (b[0] - a[0], b[1] - a[1]))
        return self.post(raw, t, k)

    def close(self) -> None:
        self.r.close()


# ---------------------------------------------------------------------------
# audio (numpy): room tone, steps, flutters, the ignition, the hum; silence at the cut
# ---------------------------------------------------------------------------


def synth_audio(scene: TempleStandoff | None = None, sr: int = 48000, seed: int = 0) -> np.ndarray:
    """Mono float32 in [-1, 1], T_END long. Everything is synthesized here."""
    import scipy.signal as sg

    rng = np.random.default_rng(seed)
    n = int(T_END * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, np.float64)

    def lp(x, fc, order=2):
        b, a = sg.butter(order, fc / (sr / 2), "low")
        return sg.lfilter(b, a, x)

    def bp(x, f0, f1, order=2):
        b, a = sg.butter(order, [f0 / (sr / 2), f1 / (sr / 2)], "band")
        return sg.lfilter(b, a, x)

    def db(v):
        return 10 ** (v / 20)

    # room tone: a low rumble of air in a big stone room + faint high air
    rum = lp(np.cumsum(rng.normal(0, 1, n)) * 0.02, 180.0)
    rum -= lp(rum, 20.0)
    rum /= np.abs(rum).max() + 1e-9
    air = bp(rng.normal(0, 1, n), 2500.0, 7000.0)
    air /= np.abs(air).max() + 1e-9
    room = db(-40) * rum + db(-58) * air
    # the room tone thins a little into the hold (tension), comes back at the ignition
    env = np.ones(n)
    env *= 1.0 - 0.35 * np.clip((t - 8.5) / 1.5, 0, 1) * (t < T_IGNITE)
    out += room * env

    def add(ev, at):
        i = int(at * sr)
        j = min(n, i + len(ev))
        if i < n:
            out[i:j] += ev[:j - i]

    def tick(level_db, dur=0.03, f0=900.0, f1=4200.0):
        k = int(dur * sr)
        e = bp(rng.normal(0, 1, k), f0, f1) * np.exp(-np.arange(k) / (0.006 * sr))
        return db(level_db) * e / (np.abs(e).max() + 1e-9)

    # the tall fly's steps (touchdowns of the tripods) once it is in the hall
    if scene is not None:
        prev = None
        for ti in np.arange(1.9, T_WALK1, 1.0 / 400):
            _, phase, mag = scene._tall_walk(ti)
            c = int(phase // math.pi)
            if prev is not None and c != prev and mag > 0.2:
                add(tick(-34 + 6 * mag, 0.04, 500.0, 3000.0), ti + 0.08)
            prev = c
    # the lead little fly's careful steps
    for ti in np.arange(T_STEP0 + 0.08, T_STEP1 - 0.05, 1 / 6.4):
        add(tick(-46, 0.02, 1500.0, 6000.0), ti)
    # the notice: a soft stir of tiny feet
    for ti in T_NOTICE + 0.2 + np.sort(rng.uniform(0, 0.9, 7)):
        add(tick(-50, 0.015, 2000.0, 7000.0), ti)

    # wing flutters (recoil): a short buzzy burst
    def flutter(level_db, dur=0.14, f=190.0):
        k = int(dur * sr)
        tt = np.arange(k) / sr
        buzz = np.sign(np.sin(2 * np.pi * f * tt)) * 0.5 + np.sin(2 * np.pi * 2 * f * tt) * 0.3
        noise = bp(rng.normal(0, 1, k), 150.0, 3000.0)
        e = (buzz * 0.5 + noise / (np.abs(noise).max() + 1e-9)) * np.sin(np.pi * tt / dur) ** 2
        return db(level_db) * lp(e, 2500.0)

    for i in range(5):
        add(flutter(-30 - 3 * rng.random(), 0.12 + 0.05 * rng.random(), 170 + 60 * rng.random()),
            T_RECOIL + 0.035 * rng.random())
    add(flutter(-31, 0.14, 205.0), T_LEAD_RECOIL)

    # the ignition: a snap, a pitched-down whoomp, a hiss, then a low hum with beating
    k = int(0.004 * sr)
    snap = rng.normal(0, 1, k) * np.exp(-np.arange(k) / (0.0008 * sr))
    add(db(-9) * snap / np.abs(snap).max(), T_IGNITE)
    k = int(0.35 * sr)
    tt = np.arange(k) / sr
    f = 520.0 * np.exp(-tt / 0.06) + 70.0
    whoomp = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt / 0.12) * (1 - np.exp(-tt / 0.004))
    add(db(-12) * whoomp, T_IGNITE)
    k = int(0.9 * sr)
    tt = np.arange(k) / sr
    hiss = bp(rng.normal(0, 1, k), 1800.0, 9000.0)
    hiss = hiss / np.abs(hiss).max() * np.exp(-tt / 0.18) * (1 - np.exp(-tt / 0.003))
    add(db(-15) * hiss, T_IGNITE)
    th = t[t >= T_IGNITE] - T_IGNITE
    beat = 1.0 + 0.18 * np.sin(2 * np.pi * 0.9 * th) + 0.08 * np.sin(2 * np.pi * 3.1 * th)
    vib = 1.0 + 0.004 * np.sin(2 * np.pi * 5.0 * th)
    ph = 2 * np.pi * 86.0 * np.cumsum(vib) / sr
    hum = np.sin(ph) + 0.55 * np.sin(2 * ph + 0.3) + 0.28 * np.sin(3 * ph + 1.1) + 0.12 * np.sin(5 * ph)
    hum += 0.25 * np.sin(2 * np.pi * 87.3 * th)  # a second slightly detuned voice (beating)
    swell = (1 - np.exp(-th / 0.05)) * (1.0 + 0.6 * np.exp(-th / 0.25))
    out[t >= T_IGNITE] += db(-17) * hum * beat * swell / 2.2
    # silence at the cut (a 6 ms fade, no click)
    i = int(T_BLACK * sr)
    fade = int(0.006 * sr)
    out[i - fade:i] *= np.linspace(1, 0, fade)
    out[i:] = 0.0
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def write_wav(path: Path, x: np.ndarray, sr: int = 48000) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# the render
# ---------------------------------------------------------------------------

DEFAULT_CAPTION = "Master Flywalker\u2026 they're everywhere. Please help us."


@dataclass
class RenderOptions:
    width: int = 1920
    height: int = 1080
    fps: float = 24.0
    letterbox: bool = True
    audio: bool = True
    shadows: bool = True
    grain: float = 3.0
    t0: float = 0.0
    t1: float = T_END
    png_dir: str | None = None  # also dump PNG frames here (every png_every-th frame)
    png_every: int = 0
    crf: int = 18
    # subtitle for the lead little fly (None = off). Default: an original line, shown
    # during the lead close-up while it looks up at the tall fly.
    caption: str | None = DEFAULT_CAPTION
    caption_t: tuple[float, float] = (7.1, 8.4)


def render(out: str | Path, opt: RenderOptions, log=print) -> dict:
    import time

    import cv2

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    scene = TempleStandoff(shadows=opt.shadows)
    log(f"[{NAME}] built in {time.perf_counter() - t_start:.1f} s "
        f"(nbody {scene.model.nbody}, ngeom {scene.model.ngeom}, IK err {scene.ik_err})")
    W, H = opt.width, opt.height
    srend = SceneRenderer(scene, W, H, opt.letterbox, opt.grain)
    if opt.caption:
        srend.post.caption = (opt.caption, float(opt.caption_t[0]), float(opt.caption_t[1]))
    wav = None
    if opt.audio:
        wav = out.with_suffix(".wav")
        a = synth_audio(scene)
        i0, i1 = int(opt.t0 * 48000), int(opt.t1 * 48000)
        write_wav(wav, a[i0:i1])
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(opt.fps), "-i", "-"]
    if wav is not None:
        cmd += ["-i", str(wav), "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(opt.crf), "-preset", "medium",
            "-movflags", "+faststart", str(out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n0 = int(round(opt.t0 * opt.fps))
    n1 = int(round(opt.t1 * opt.fps))
    png = Path(opt.png_dir) if opt.png_dir else None
    if png:
        png.mkdir(parents=True, exist_ok=True)
    t_r = time.perf_counter()
    try:
        for k in range(n0, n1):
            t = k / opt.fps
            frame = srend.frame(t, k)
            ff.stdin.write(frame.tobytes())
            if png and opt.png_every and (k - n0) % opt.png_every == 0:
                cv2.imwrite(str(png / f"f{k:04d}_{t:05.2f}.png"), frame[..., ::-1])
    finally:
        ff.stdin.close()
        ff.wait()
        srend.close()
        if wav is not None and wav.exists():
            wav.unlink()
    dt = time.perf_counter() - t_r
    log(f"[{NAME}] {n1 - n0} frames {W}x{H} @ {opt.fps:g} fps in {dt:.1f} s -> {out}")
    return dict(frames=n1 - n0, seconds=dt, out=str(out))
