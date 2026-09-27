"""Pizza chef: the fly makes pizza forever.

A fly-scale pizzeria on a marble counter: a floury prep board in front of the fly
(who wears a chef's toque), a wood-fired brick oven with glowing embers, a rail of
ingredient bowls overhead (sauce, cheese, pepperoni, basil), a pizza peel, a cutter
wheel, a takeaway box, a tip jar and a "pizzas served" chalkboard. One pizza:

1. **Dough.** A dough ball plops onto the board.
2. **Knead.** The front legs press it, left, right, left ...: the leg targets come
   from damped least-squares IK on a scratch MjData (the dead_hang ``LegIK``) and
   the legs are position-controlled like any action (nothing is teleported). Every
   press flattens the dough one stage (a set of dough meshes from a lumpy ball to a
   flat disc; the dough is visual, the flattening is triggered by the press, not
   computed from a contact force). Flour puffs up at each press.
3. **Toss.** The legs slide under the dough and fling up; the dough becomes a free
   body (a spinning disc) with a **launch velocity set by the job (engineered,
   labelled)**: straight up to ``toss_height`` plus a random lateral error, spin
   about its axis. From then on it is real rigid-body physics: it flies, lands on
   the board and settles. Landing within ``perfect_off`` of the centre and flat is
   a PERFECT toss. A toss at fly scale lasts ~60 ms, so the flight is shown in
   **slow motion (an edit effect: fewer physics steps per displayed frame, the
   physics unchanged; labelled on screen)**. Off-centre dough is slid back to the
   middle (kinematic, labelled "re-centred").
4. **Sauce + taste.** The sauce bowl tips and a stream spirals out over the base
   (the sauce spreads through a set of discs). The left front leg dips into the sauce;
   with ``--brain`` that sends a taste pulse to the connectome (**stand-in,
   labelled**: Shiu et al.'s labellar sugar GRNs LB3, as the taste patches /
   taste_tester use; the model has no tarsal taste neurons) and the proboscis
   follows the brain's MN9. Without a brain a labelled scripted proboscis dab.
5. **Toppings rain.** The cheese, pepperoni and basil bowls tip and pour: pooled
   small free bodies (shreds, slices, leaves) launched from the bowl lips at the
   pizza (**aim set by the job**) that fall and land with real contacts on the
   pizza (slow motion edit while they fall). Once landed they ride on the pizza
   (kinematic carry: contacts off, gravity compensated).
6. **Oven.** The peel (kinematic, labelled) slides under the pizza, carries it into
   the dome and out again after ``bake_s``: the crust browns, the cheese melts, the
   pepperoni darkens, the embers glow brighter, steam rises. While it bakes the chef
   grooms (the real recorded NeuroMechFly grooming clip, like kebab's stroke).
7. **Slice.** A cutter wheel (kinematic) rolls four times across: 8 slices, which
   separate a little (the base / sauce / cheese are 8 wedge meshes; the toppings
   follow their slice).
8. **Serve.** The pizza slides into a takeaway box, the lid closes, the box slides
   off to the pick-up; tips drop into the jar; the chalkboard counts. New dough.

Constant memory: fixed pools (toppings, puffs, coins), counters, no per-event lists.
Everything but the free bodies is visual; the free bodies touch only the board /
pizza colliders and the counter, never the fly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from fly_simulator.actions.base import Action, ActionCommand, smoothstep
from fly_simulator.jobs import kebab_assets as KA
from fly_simulator.jobs import pizza_chef_assets as A
from fly_simulator.jobs import taste_tester_assets as TA
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import add_box, contact_kwargs, quat_axis_angle, wrap_angle
from fly_simulator.jobs.kebab import CarveStroke
from fly_simulator.jobs.registry import register_job
from fly_simulator.jobs.taste_tester import DIGITS, SEGS

P = "pizza/"
PZ_BIT = 64  # contact bit: pizza / board colliders <-> toppings and the tossed dough
TERRAIN_BIT = 16
KINDS = ("cheese", "pepperoni", "basil")
N_WEDGES = 8


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------


class ChefStance(Action):
    """Stand still with all tarsi adhering (like ``freeze``) while the job moves the
    front legs: ``pose[leg]`` (7 joint targets, controller order) blended in with
    ``w[leg]`` (0..1). A lifted leg's adhesion is off."""

    name = "chef_stance"
    blend_in = 0.25
    blend_out = 0.3

    def __init__(self, duration: float = 3600.0) -> None:
        super().__init__(duration)
        self.pose: dict[str, np.ndarray | None] = {"lf": None, "rf": None}
        self.w = {"lf": 0.0, "rf": 0.0}

    def begin(self, mgr) -> None:
        from fly_simulator.actions.base import LEGS

        b = mgr.body
        self._start = mgr.sim.data.ctrl[b.pos_ids].copy()
        self._stand = b.stand.copy()
        self.cols = {leg: np.flatnonzero(b.leg_mask([leg])) for leg in self.pose}
        self._li = {leg: LEGS.index(leg) for leg in self.pose}

    def command(self, mgr, t: float) -> ActionCommand:
        a = smoothstep(t / 0.2)
        tg = (1 - a) * self._start + a * self._stand
        adh = np.ones(6)
        for leg, pose in self.pose.items():
            w = float(min(max(self.w[leg], 0.0), 1.0))
            if pose is not None and w > 0:
                c = self.cols[leg]
                tg[c] = (1 - w) * tg[c] + w * pose
                if w > 0.05:
                    adh[self._li[leg]] = 0.0
        return ActionCommand(targets=tg, adhesion=adh)

    def end(self, mgr, cancelled: bool) -> None:
        self.info = {"cancelled": cancelled, "tilt_deg": mgr.sim.tilt_deg()}


class ChefGroom(CarveStroke):
    """While the pizza bakes the chef grooms: the real recorded NeuroMechFly
    front-leg grooming clip (``CarveStroke`` loops it; here without a knife)."""

    name = "chef_groom"

    def phase(self, t: float) -> str:
        return "groom"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class PizzaChefConfig(JobConfig):
    # ---- layout (the fly spawns at the origin facing +x; thorax settles at ~0.61) ----
    pizza_x: float = 2.05  # prep spot (thorax + 1.44: the front legs reach it)
    pizza_y: float = 0.0
    board_top: float = 0.02
    # ---- dough and pizza ------------------------------------------------------------
    n_stages: int = 7  # dough meshes: lumpy ball -> flat disc
    ball_r: float = 0.30
    ball_h: float = 0.32
    flat_r: float = 0.62
    flat_h: float = 0.06
    pizza_r: float = 0.75  # after the toss (stretched)
    base_h: float = 0.045
    rim_w: float = 0.13
    rim_h: float = 0.035
    sauce_r: float = 0.6
    sauce_t: float = 0.008
    # ---- kneading ---------------------------------------------------------------------
    n_presses: int = 10
    press_s: float = 0.42
    # ---- toss (launch set by the job; flight = real physics) -------------------------
    toss_height: float = 3.5  # mm above the board
    toss_sigma: float = 0.13  # sd of the landing error (mm, per axis)
    toss_spin: float = 30.0  # rad/s about the disc axis
    toss_wobble: float = 3.0  # sd of the tumble rate (rad/s)
    toss_mass: float = 8e-5  # g (0.08 mg of dough)
    bonus_toss_p: float = 0.35  # chance of a second (show-off) toss
    perfect_off: float = 0.25  # mm from the centre
    perfect_tilt_deg: float = 20.0
    toss_timeout_s: float = 1.0
    # ---- sauce / taste -----------------------------------------------------------------
    sauce_s: float = 1.6
    taste_s: float = 0.5  # stimulus duration
    sugar_hz: float = 150.0  # the sauce's "sweetness" drive (stand-in)
    mn9_ref_hz: float = 60.0  # MN9 rate that extends the proboscis fully
    # ---- toppings (pooled free bodies) ------------------------------------------------
    n_cheese: int = 36
    n_pepperoni: int = 9
    n_basil: int = 5
    pour_s: tuple = (0.5, 0.35, 0.3)  # cheese, pepperoni, basil pour windows
    topping_settle_s: float = 0.15  # landed this long (or at rest) -> rides on the pizza
    # ---- oven / bake -------------------------------------------------------------------
    oven_x: float = 1.6
    oven_y: float = 4.0
    oven_R: float = 1.55
    oven_H: float = 1.3
    oven_mouth_deg: float = -60.0  # mouth direction (world yaw)
    oven_reach: float = 3.3  # oven centre -> the peel's turning point in front of it
    hearth_z: float = 0.5
    bake_s: float = 5.0
    # ---- serving / tips ----------------------------------------------------------------
    box_y: float = -2.7
    jar_x: float = 0.0
    jar_y: float = -2.35
    n_coins: int = 24
    # ---- presentation (edit effects, labelled) -----------------------------------------
    slowmo: bool = True
    toss_slowmo: float = 0.1
    pour_slowmo: float = 0.3
    close_ups: bool = True  # camera: close-up at the board, wide for the oven / serving
    shadows: bool = True
    chef_hat: bool = True
    stuck_timeout_s: float = 90.0


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _qmul(a, b) -> np.ndarray:
    out = np.empty(4)
    mj.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def _rot2(yaw: float, v) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class _Puffs:
    """A fixed pool of fading spheres (flour, steam, smoke) on mocap bodies."""

    def __init__(self, n: int, rgb, life: float, r0: float, r1: float, alpha: float) -> None:
        self.n, self.rgb, self.life, self.r0, self.r1, self.alpha = n, rgb, life, r0, r1, alpha
        self.t0 = np.full(n, -1e9)
        self.p0 = np.zeros((n, 3))
        self.v = np.zeros((n, 3))
        self.k = 0

    def spawn(self, t: float, p, v, life_scale: float = 1.0) -> None:
        i = self.k % self.n
        self.k += 1
        self.t0[i] = t
        self.p0[i] = p
        self.v[i] = v
        self._ls = getattr(self, "_ls", np.ones(self.n))
        self._ls[i] = life_scale

    def draw(self, t: float, m, d, mids, gids) -> None:
        ls = getattr(self, "_ls", np.ones(self.n))
        for i in range(self.n):
            tau = (t - self.t0[i]) / (self.life * ls[i])
            g = gids[i]
            if not 0.0 <= tau < 1.0:
                if m.geom_rgba[g, 3] != 0.0:
                    m.geom_rgba[g, 3] = 0.0
                    d.mocap_pos[mids[i]] = (0.0, 0.0, -5.0)
                continue
            e = 1.0 - (1.0 - tau) ** 2
            d.mocap_pos[mids[i]] = self.p0[i] + self.v[i] * e * self.life * ls[i]
            m.geom_size[g, 0] = self.r0 + (self.r1 - self.r0) * e
            m.geom_rgba[g, :3] = self.rgb
            m.geom_rgba[g, 3] = self.alpha * (1.0 - tau) * min(tau / 0.08, 1.0)


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------


@register_job
class PizzaChefJob(EternalJob):
    name = "pizza_chef"
    #: depth precision (shadow maps span znear..zfar; EternalJob applies it after compile)
    znear = 0.05
    title = "PIZZA CHEF FLY"
    tagline = "the fly makes pizza forever"
    work_label = "pizzas served"
    config_cls = PizzaChefConfig
    required_names = (P + "pizza", P + "toss", P + "peel", P + "board_col")

    cfg: PizzaChefConfig

    # ------------------------------------------------------------ geometry
    @property
    def p0(self) -> np.ndarray:
        return np.array([self.cfg.pizza_x, self.cfg.pizza_y])

    @property
    def mouth(self) -> np.ndarray:
        a = math.radians(self.cfg.oven_mouth_deg)
        return np.array([math.cos(a), math.sin(a)])

    @property
    def oven_c(self) -> np.ndarray:
        return np.array([self.cfg.oven_x, self.cfg.oven_y])

    @property
    def turn_pt(self) -> np.ndarray:  # where the peel turns toward the oven
        return self.oven_c + self.cfg.oven_reach * self.mouth

    @property
    def u_dir(self) -> np.ndarray:  # prep spot -> turning point
        v = self.turn_pt - self.p0
        return v / np.linalg.norm(v)

    @property
    def bake_pt(self) -> np.ndarray:
        return self.oven_c + 0.2 * self.mouth

    @property
    def peel_home(self) -> np.ndarray:
        return self.turn_pt + 0.55 * self.u_dir

    @property
    def box_c(self) -> np.ndarray:
        return np.array([self.cfg.pizza_x, self.cfg.box_y])

    @property
    def pizza_top(self) -> float:  # top of the sauce layer (the toppings land here)
        c = self.cfg
        return c.board_top + c.base_h + c.sauce_t

    def stage_shape(self, k: int) -> tuple[float, float, float]:
        """(r, h, superellipse p) of dough stage k."""
        c = self.cfg
        f = k / max(c.n_stages - 1, 1)
        e = f ** 0.8
        return (c.ball_r + (c.flat_r - c.ball_r) * e, c.ball_h + (c.flat_h - c.ball_h) * e,
                2.0 + 3.0 * f)

    def configure_app(self, app_cfg) -> None:
        super().configure_app(app_cfg)
        app_cfg.controller.target_heading_deg = 0.0
        app_cfg.fly.extra_joints = True  # proboscis joints (the sauce taste)

    # ------------------------------------------------------------ build
    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        vis = contact_kwargs("visual")
        vm = dict(vis, mass=0.0)
        self._add_materials(spec)
        if c.chef_hat:
            orig_add_fly = world.add_fly

            def add_fly(fly, *a, **kw):
                world.add_fly = orig_add_fly
                self._dress_fly(fly)
                return orig_add_fly(fly, *a, **kw)
            world.add_fly = add_fly
        zb = c.board_top
        px, py = c.pizza_x, c.pizza_y
        # ---- the prep board (visual) and its colliders (props only)
        KA.add_mesh(spec, P + "board_mesh", TA.flat_quad_mesh(1.05, 1.25, zb))
        wb.add_geom(name=P + "board", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "board_mesh",
                    pos=(px + 0.1, py, zb / 2), material=P + "board", **vm)
        # (thick boxes: a fast piece sinks ~ v * solref[0] into a soft contact)
        col = dict(contype=PZ_BIT, conaffinity=0, condim=3, friction=(0.9, 0.005, 0.0001),
                   solref=(0.0005, 1.0), rgba=(1, 1, 1, 0), group=3)
        wb.add_geom(name=P + "board_col", type=mj.mjtGeom.mjGEOM_BOX, size=(1.05, 1.25, 0.3),
                    pos=(px + 0.1, py, zb - 0.3), **col)
        wb.add_geom(name=P + "top_col", type=mj.mjtGeom.mjGEOM_BOX, size=(0.8, 0.8, 0.3),
                    pos=(px, py, self.pizza_top - 0.3), **col)
        # ---- the pizza: one mocap body carrying every stage / layer (visual)
        pz = wb.add_body(name=P + "pizza", pos=(px, py, zb), mocap=True)
        self._dough_names = []
        for k in range(c.n_stages):
            r, h, p = self.stage_shape(k)
            lump = 0.05 * (1 - k / max(c.n_stages - 1, 1)) + 0.008
            KA.add_mesh(spec, f"{P}dough{k}_mesh", A.dough_mesh(r, h, p, lump, seed=k))
            nm = f"{P}dough{k}"
            pz.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_MESH, meshname=nm + "_mesh",
                        material=P + "dough", rgba=(1, 1, 1, 0), **vm)
            self._dough_names.append(nm)
        top_base = lambda rho: A.base_profile(c.pizza_r, c.base_h, c.rim_w, c.rim_h, rho)  # noqa: E731
        top_sauce = lambda rho: np.full_like(rho, c.base_h + c.sauce_t)  # noqa: E731
        top_cheese = lambda rho: np.full_like(rho, c.base_h + c.sauce_t + 0.009)  # noqa: E731
        self._wedge_names = {"base": [], "sauce": [], "cheese": []}
        dA = math.tau / N_WEDGES
        for i in range(N_WEDGES):
            a0, a1 = i * dA, (i + 1) * dA
            for layer, fn, r, z0, mat in (
                    ("base", top_base, c.pizza_r, 0.0, P + "dough"),
                    ("sauce", top_sauce, c.sauce_r, c.base_h - 0.004, P + "sauce"),
                    ("cheese", top_cheese, c.sauce_r + 0.02, c.base_h + c.sauce_t - 0.002, P + "melt")):
                KA.add_mesh(spec, f"{P}{layer}_w{i}_mesh", A.wedge_mesh(a0, a1, r, fn, z0=z0))
                nm = f"{P}{layer}_w{i}"
                pz.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_MESH, meshname=nm + "_mesh",
                            material=mat, rgba=(1, 1, 1, 0), **vm)
                self._wedge_names[layer].append(nm)
        self._sauce_names = []
        for k, r in enumerate((0.16, 0.3, 0.44, 0.54)):
            KA.add_mesh(spec, f"{P}sauce{k}_mesh", A.layer_disc_mesh(r, c.sauce_t, 0.08, seed=k))
            nm = f"{P}sauce{k}"
            pz.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_MESH, meshname=nm + "_mesh",
                        pos=(0, 0, c.base_h - 0.004), material=P + "sauce", rgba=(1, 1, 1, 0), **vm)
            self._sauce_names.append(nm)
        # ---- the tossed dough: a free body (real physics in the air), parked otherwise
        tb = wb.add_body(name=P + "toss", pos=(px, py, -4.0))
        tb.add_freejoint(name=P + "toss_free")
        tb.add_geom(name=P + "toss_col", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.66, 0.025, 0),
                    mass=c.toss_mass, contype=0, conaffinity=PZ_BIT | TERRAIN_BIT, condim=3,
                    friction=(0.8, 0.005, 0.0001), solref=(0.0005, 1.0), rgba=(1, 1, 1, 0), group=3)
        r, h, p = self.stage_shape(c.n_stages - 1)
        tb.add_geom(name=P + "toss_dough", type=mj.mjtGeom.mjGEOM_MESH, meshname=f"{P}dough{c.n_stages - 1}_mesh",
                    pos=(0, 0, -0.025), material=P + "dough", rgba=(1, 1, 1, 0), **vm)
        rr = np.linspace(0.0, c.pizza_r, 18)
        zz = A.base_profile(c.pizza_r, c.base_h, c.rim_w, c.rim_h, rr)
        KA.add_mesh(spec, P + "base_full_mesh",
                    A.planar_uv(KA.lathe(np.r_[0.0, rr[1:], rr[::-1][:-1], 0.0],
                                         np.r_[0.0, np.zeros(17), zz[::-1][:-1], zz[0]], 48)))
        tb.add_geom(name=P + "toss_base", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "base_full_mesh",
                    pos=(0, 0, -0.025), material=P + "dough", rgba=(1, 1, 1, 0), **vm)
        # ---- topping pool: free bodies (cheese shreds, pepperoni slices, basil leaves)
        KA.add_mesh(spec, P + "pep_mesh", A.pepperoni_mesh(0.1, 0.02))
        KA.add_mesh(spec, P + "leaf_mesh", A.leaf_mesh(0.26, 0.15))
        self._top_names = []
        tcol = dict(contype=0, conaffinity=PZ_BIT | TERRAIN_BIT, condim=6,
                    friction=(1.0, 0.02, 0.02), solref=(0.0005, 1.0))
        n_per = {"cheese": c.n_cheese, "pepperoni": c.n_pepperoni, "basil": c.n_basil}
        rng = np.random.default_rng(c.seed + 31)
        for kind in KINDS:
            for j in range(n_per[kind]):
                nm = f"{P}{kind}{j}"
                b = wb.add_body(name=nm, pos=(-3.0 - 0.1 * len(self._top_names), 8.0, -3.0))
                b.add_freejoint(name=nm + "_free")
                if kind == "cheese":
                    L = float(rng.uniform(0.07, 0.11))
                    b.add_geom(name=nm + "_g", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.017, L, 0),
                               quat=quat_axis_angle((0, 1, 0), math.pi / 2), mass=1e-6,
                               material=P + "cheese_shred", rgba=(1, 1, 1, 1), **tcol)
                elif kind == "pepperoni":
                    b.add_geom(name=nm + "_g", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.1, 0.01, 0),
                               mass=2e-6, rgba=(1, 1, 1, 0), group=3, **tcol)
                    b.add_geom(name=nm + "_v", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "pep_mesh",
                               pos=(0, 0, -0.005), material=P + "pepperoni", rgba=(1, 1, 1, 1), **vm)
                else:
                    b.add_geom(name=nm + "_g", type=mj.mjtGeom.mjGEOM_BOX, size=(0.12, 0.06, 0.006),
                               mass=1e-6, rgba=(1, 1, 1, 0), group=3, **tcol)
                    b.add_geom(name=nm + "_v", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "leaf_mesh",
                               material=P + "basil", rgba=(1, 1, 1, 1), **vm)
                b.gravcomp = 1.0
                self._top_names.append((kind, nm))
        self._add_bowls(spec)
        self._add_oven(spec)
        self._add_tools(spec)
        self._add_room(spec)

    def _dress_fly(self, fly) -> None:
        """The chef's toque (visual, massless, no contacts) on the thorax."""
        root = fly.mjcf_root if hasattr(fly, "mjcf_root") else fly.spec
        vis = contact_kwargs("visual")
        th = root.body("c_thorax")
        root.add_material(name="pizza_hat_cloth", rgba=(0.98, 0.98, 0.98, 1.0), specular=0.15,
                          shininess=0.2)
        for part, md in KA.chef_hat_meshes().items():
            KA.add_mesh(root, f"pizza_hat_{part}_mesh",
                        KA.transform(md, np.eye(3), np.array([0.3, 0.0, 0.35])))
            th.add_geom(name=f"pizza_hat_{part}", type=mj.mjtGeom.mjGEOM_MESH,
                        meshname=f"pizza_hat_{part}_mesh", material="pizza_hat_cloth", mass=0.0, **vis)

    def _add_materials(self, spec) -> None:
        c = self.cfg
        mat = spec.material("grid")
        if mat is not None:
            KA.add_texture(spec, P + "tex_marble", A.marble_texture(c.seed))
            mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = P + "tex_marble"
            mat.rgba = (1.0, 1.0, 1.0, 1.0)
            mat.reflectance = 0.08
            mat.specular = 0.5
            mat.shininess = 0.6
        tex = (("board", A.board_texture(c.seed), dict(specular=0.1, shininess=0.2)),
               ("dough", A.dough_texture(c.seed), dict(specular=0.15, shininess=0.3)),
               ("crust", A.crust_texture(c.seed), dict(specular=0.2, shininess=0.35)),
               ("sauce", A.sauce_texture(c.seed), dict(specular=0.6, shininess=0.7)),
               ("melt", A.cheese_melt_texture(c.seed), dict(specular=0.5, shininess=0.6)),
               ("pepperoni", A.pepperoni_texture(c.seed), dict(specular=0.5, shininess=0.6)),
               ("basil", A.basil_texture(), dict(specular=0.4, shininess=0.5)),
               ("stone", A.stone_texture(c.seed), dict(specular=0.1)),
               ("wood", A.wood_texture(c.seed), dict(specular=0.2, shininess=0.3)),
               ("log", A.log_texture(c.seed), dict(specular=0.05)),
               ("chalk", A.chalkboard_texture(), dict(specular=0.05, emission=0.12)),
               ("menu", A.menu_texture(), dict(specular=0.1, emission=0.1)),
               ("neon", A.neon_texture(), dict(emission=0.9)),
               ("lid", A.box_lid_texture(), dict(specular=0.1)),
               ("card", A.cardboard_texture(), dict(specular=0.05)),
               ("hearth", A.mouth_glow_texture(), dict(emission=0.5)))
        for nm, img, kw in tex:
            KA.add_texture(spec, f"{P}tex_{nm}", img)
            KA.add_textured_material(spec, P + nm, f"{P}tex_{nm}", rgba=(1, 1, 1, 1), **kw)
        KA.add_texture(spec, P + "tex_brick", A.brick_texture(c.seed))
        KA.add_textured_material(spec, P + "brick", P + "tex_brick", rgba=(1, 1, 1, 1), specular=0.1)
        KA.add_texture(spec, P + "tex_brick2", A.brick_texture(c.seed + 1, soot=False))
        KA.add_textured_material(spec, P + "brickwall", P + "tex_brick2", rgba=(1, 1, 1, 1), specular=0.1,
                                 texuniform=True, texrepeat=(1.0, 1.0))
        KA.add_texture(spec, P + "tex_wall", KA.wall_texture(c.seed))
        KA.add_textured_material(spec, P + "wall", P + "tex_wall", rgba=(1, 1, 1, 1), specular=0.3)
        KA.add_texture(spec, P + "tex_steel", KA.steel_texture(c.seed))
        KA.add_textured_material(spec, P + "steel", P + "tex_steel", rgba=(1, 1, 1, 1), specular=0.9,
                                 shininess=0.9, texuniform=True, texrepeat=(2.0, 2.0))
        for word in ("SAUCE", "CHEESE", "PEPPERONI", "BASIL", "TIPS", "FLOUR"):
            KA.add_texture(spec, f"{P}tex_lbl_{word}", A.label_texture(word))
            KA.add_textured_material(spec, f"{P}lbl_{word}", f"{P}tex_lbl_{word}", rgba=(1, 1, 1, 1),
                                     emission=0.15)
        spec.add_material(name=P + "cheese_shred", rgba=(1.0, 0.95, 0.72, 1), specular=0.4, shininess=0.5)
        spec.add_material(name=P + "ceramic", rgba=(0.95, 0.94, 0.9, 1), specular=0.8, shininess=0.8)
        spec.add_material(name=P + "ceramic_blue", rgba=(0.2, 0.36, 0.62, 1), specular=0.8, shininess=0.8)
        spec.add_material(name=P + "glass", rgba=(0.85, 0.93, 0.97, 0.3), specular=1.0, shininess=1.0)
        spec.add_material(name=P + "gold", rgba=(0.95, 0.75, 0.25, 1), specular=1.0, shininess=0.9,
                          emission=0.1)
        spec.add_material(name=P + "ember", rgba=(1.0, 0.42, 0.08, 1), emission=0.9)
        spec.add_material(name=P + "flame", rgba=(1.0, 0.72, 0.2, 0.8), emission=1.0)
        spec.add_material(name=P + "soot", rgba=(0.06, 0.05, 0.05, 1), specular=0.05)
        spec.add_material(name=P + "seg_on", rgba=(0.95, 0.95, 0.9, 1), emission=0.35)
        spec.add_material(name=P + "seg_off", rgba=(0.14, 0.17, 0.15, 0.0))
        spec.add_material(name=P + "bowl_fill_sauce", rgba=(0.72, 0.1, 0.05, 1), specular=0.7, shininess=0.8)
        spec.add_material(name=P + "black", rgba=(0.05, 0.05, 0.06, 1), specular=0.5, shininess=0.6)
        spec.add_material(name=P + "tomato", rgba=(0.85, 0.12, 0.08, 1), specular=0.8, shininess=0.8)
        spec.add_material(name=P + "leafgreen", rgba=(0.2, 0.5, 0.15, 1), specular=0.3)

    def _add_bowls(self, spec) -> None:
        """Four ingredient bowls hanging from a wall-mounted rail above the pizza
        (they tip to pour: kinematic)."""
        c = self.cfg
        wb = spec.worldbody
        vis = contact_kwargs("visual")
        vm = dict(vis, mass=0.0)
        self.bowl_r, self.bowl_h = 0.22, 0.2
        self.bowl_x, self.bowl_z = c.pizza_x + 0.3, 1.62
        rail_z = 3.2
        self.bowl_ys = [c.pizza_y + v for v in (-0.78, -0.26, 0.26, 0.78)]
        KA.add_mesh(spec, P + "bowl_mesh", A.bowl_mesh(self.bowl_r, self.bowl_h))
        fill_r = self.bowl_r * 0.86
        KA.add_mesh(spec, P + "fill_mesh", A.dough_mesh(fill_r, 0.06, 2.5, 0.0))
        KA.add_mesh(spec, P + "bowl_label_mesh", TA.front_panel_mesh(0.14, 0.05, 0.006))
        fills = (P + "bowl_fill_sauce", P + "cheese_shred", P + "pepperoni", P + "basil")
        words = ("SAUCE", "CHEESE", "PEPPERONI", "BASIL")
        self._bowl_names = []
        for k, y in enumerate(self.bowl_ys):
            b = wb.add_body(name=f"{P}bowl{k}", pos=(self.bowl_x, y, self.bowl_z), mocap=True)
            b.add_geom(name=f"{P}bowl{k}_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "bowl_mesh",
                       material=P + ("ceramic_blue" if k % 2 else "ceramic"), **vm)
            b.add_geom(name=f"{P}bowl{k}_fill", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "fill_mesh",
                       pos=(0, 0, 0.12), material=fills[k], **vm)
            b.add_geom(name=f"{P}bowl{k}_label", type=mj.mjtGeom.mjGEOM_MESH,
                       meshname=P + "bowl_label_mesh", pos=(self.bowl_r * 0.93 + 0.01, 0, 0.1),
                       material=f"{P}lbl_{words[k]}", **vm)
            # hanger: a yoke over the bowl and a rod up to the rail
            b.add_geom(name=f"{P}bowl{k}_rod", type=mj.mjtGeom.mjGEOM_CYLINDER,
                       size=(0.012, 0.5 * (rail_z - self.bowl_z - 0.3), 0),
                       pos=(0, 0, 0.3 + 0.5 * (rail_z - self.bowl_z - 0.3)), material=P + "steel", **vm)
            b.add_geom(name=f"{P}bowl{k}_yoke", type=mj.mjtGeom.mjGEOM_BOX, size=(0.012, self.bowl_r + 0.03, 0.012),
                       pos=(0, 0, 0.3), material=P + "steel", **vm)
            for s in (-1, 1):
                b.add_geom(name=f"{P}bowl{k}_arm{s}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.01, 0.01, 0.1),
                           pos=(0, s * (self.bowl_r + 0.03), 0.2), material=P + "steel", **vm)
            self._bowl_names.append(f"{P}bowl{k}")
        # the rail and its arm to the back wall
        xw = -2.6
        add_box(wb, P + "rail", (0.04, 1.15, 0.04), (self.bowl_x, c.pizza_y, rail_z), material=P + "steel",
                collide="visual")
        add_box(wb, P + "rail_arm", (0.5 * (self.bowl_x - xw), 0.05, 0.05),
                (0.5 * (self.bowl_x + xw), c.pizza_y, rail_z + 0.08), material=P + "steel", collide="visual")
        # the sauce stream (a capsule on a mocap body, stretched at runtime)
        s = wb.add_body(name=P + "stream", pos=(0, 0, -5.0), mocap=True)
        s.add_geom(name=P + "stream_g", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.03, 0.2, 0),
                   material=P + "bowl_fill_sauce", rgba=(1, 1, 1, 0), **vm)

    def _add_oven(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        vis = contact_kwargs("visual")
        vm = dict(vis, mass=0.0)
        ox, oy = c.oven_x, c.oven_y
        hz = c.hearth_z
        md = math.radians(c.oven_mouth_deg)
        m = self.mouth
        # stone plinth and hearth slab
        wb.add_geom(name=P + "plinth", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(c.oven_R + 0.25, hz / 2, 0),
                    pos=(ox, oy, hz / 2), material=P + "stone", **vis)
        wb.add_geom(name=P + "hearth", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(c.oven_R - 0.14, 0.004, 0),
                    pos=(ox, oy, hz + 0.004), material=P + "hearth", **vis)
        # the dome with an arched mouth (open shell: the fire and the pizza show inside)
        KA.add_mesh(spec, P + "dome_mesh", A.dome_shell_mesh(c.oven_R, c.oven_H, 0.12, md, 0.72, 0.78))
        wb.add_geom(name=P + "dome", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "dome_mesh",
                    pos=(ox, oy, hz), material=P + "brick", **vis)
        # arch frame bricks around the mouth
        for j, a in enumerate(np.linspace(-0.72, 0.72, 11)):
            z = 0.78 * math.cos(a / 0.72 * math.pi / 2) ** 0.6
            ang = md + a
            pr = (c.oven_R + 0.03) * np.array([math.cos(ang), math.sin(ang)])
            wb.add_geom(name=f"{P}arch{j}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.07, 0.1, 0.06),
                        pos=(ox + pr[0], oy + pr[1], hz + max(z, 0.06)),
                        quat=quat_axis_angle((0, 0, 1), ang), material=P + "brickwall", **vis)
        # flue on top
        wb.add_geom(name=P + "flue", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.16, 0.35, 0),
                    pos=(ox, oy, hz + c.oven_H + 0.25), material=P + "soot", **vis)
        # fire: logs, embers, flames (flicker at runtime) at the back of the hearth
        back = self.oven_c - 0.75 * m
        fire = wb.add_body(name=P + "fire", pos=(back[0], back[1], hz + 0.01), mocap=True)
        rng = np.random.default_rng(c.seed + 5)
        perp = np.array([-m[1], m[0]])
        for j in range(3):
            q = quat_axis_angle((perp[0], perp[1], 0.0), math.pi / 2 + 0.2 * (j - 1))
            fire.add_geom(name=f"{P}log{j}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.06, 0.32, 0),
                          pos=(0.12 * (j - 1) * m[0], 0.12 * (j - 1) * m[1], 0.06 + 0.05 * (j == 1)),
                          quat=q, material=P + "log", **vm)
        self._ember_names, self._flame_names = [], []
        for j in range(14):
            off = rng.normal(0, 0.18, 2)
            nm = f"{P}ember{j}"
            fire.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_SPHERE, size=(float(rng.uniform(0.03, 0.06)), 0, 0),
                          pos=(off[0], off[1], 0.03), material=P + "ember", **vm)
            self._ember_names.append(nm)
        for j in range(6):
            off = rng.normal(0, 0.12, 2)
            nm = f"{P}flame{j}"
            fire.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(0.08, 0.08, 0.22),
                          pos=(off[0], off[1], 0.2), material=P + "flame", **vm)
            self._flame_names.append(nm)
        # firewood stack beside the oven
        for j in range(6):
            row, col = divmod(j, 3)
            p = self.oven_c + (c.oven_R + 0.7) * np.array([math.cos(md + 1.9), math.sin(md + 1.9)])
            wb.add_geom(name=f"{P}woodpile{j}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.09, 0.45, 0),
                        pos=(p[0] + 0.19 * col - 0.19 + 0.095 * row, p[1], 0.09 + 0.16 * row),
                        quat=quat_axis_angle((1, 0, 0), math.pi / 2), material=P + "log", **vis)
        # the oven's glow: a warm spot light out of the mouth (flickers at runtime)
        lp = self.oven_c + 0.1 * m
        wb.add_light(name=P + "oven_light", type=mj.mjtLightType.mjLIGHT_SPOT,
                     pos=(lp[0], lp[1], hz + 0.55), dir=(m[0], m[1], -0.35),
                     diffuse=(0.9, 0.45, 0.15), specular=(0.2, 0.1, 0.05), cutoff=60.0, exponent=2.0,
                     castshadow=False)
        wb.add_light(name=P + "fire_light", type=mj.mjtLightType.mjLIGHT_POINT,
                     pos=(back[0], back[1], hz + 0.4), diffuse=(0.6, 0.25, 0.08), specular=(0, 0, 0),
                     attenuation=(0.3, 0.9, 0.3), castshadow=False)
        # smoke puffs from the flue, steam from the pizza (pools)
        self._puff_names = {"flour": [], "steam": [], "smoke": []}
        for kind, n in (("flour", 12), ("steam", 14), ("smoke", 10)):
            for j in range(n):
                b = wb.add_body(name=f"{P}{kind}{j}", pos=(0, 0, -5.0), mocap=True)
                b.add_geom(name=f"{P}{kind}{j}_g", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.05, 0, 0),
                           rgba=(1, 1, 1, 0), **vm)
                self._puff_names[kind].append(f"{P}{kind}{j}")

    def _add_tools(self, spec) -> None:
        """Peel, cutter wheel, takeaway boxes, tip jar and coins (all kinematic)."""
        c = self.cfg
        wb = spec.worldbody
        vis = contact_kwargs("visual")
        vm = dict(vis, mass=0.0)
        # ---- peel: blade centre at the body origin, blade along +x, handle toward -x
        self.peel_L, self.peel_W = 1.75, 1.7
        KA.add_mesh(spec, P + "peel_mesh", A.peel_blade_mesh(self.peel_L, self.peel_W, 0.018))
        pe = wb.add_body(name=P + "peel", pos=(0, 0, 0), mocap=True)
        pe.add_geom(name=P + "peel_blade", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "peel_mesh",
                    material=P + "wood", **vm)
        hl = 0.9
        pe.add_geom(name=P + "peel_handle", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.05, hl, 0),
                    pos=(-0.5 * self.peel_L - hl + 0.05, 0, -0.01),
                    quat=quat_axis_angle((0, 1, 0), math.pi / 2), material=P + "wood", **vm)
        # ---- cutter: handle / guard body and a separate wheel body (it rolls)
        self.wheel_r = 0.28
        cu = wb.add_body(name=P + "cutter", pos=(0, 0, 0), mocap=True)
        cu.add_geom(name=P + "cutter_guard", type=mj.mjtGeom.mjGEOM_BOX, size=(0.14, 0.022, 0.03),
                    pos=(0, 0, self.wheel_r * 0.72), material=P + "steel", **vm)
        for sy in (-1, 1):
            cu.add_geom(name=f"{P}cutter_fork{sy}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.02, 0.006, 0.1),
                        pos=(0, sy * 0.02, self.wheel_r * 0.4), material=P + "steel", **vm)
        q = quat_axis_angle((0, 1, 0), -math.radians(50.0))  # capsule axis up and back (-x)
        dirv = np.array([-math.sin(math.radians(50.0)), 0.0, math.cos(math.radians(50.0))])
        cu.add_geom(name=P + "cutter_shaft", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.022, 0.2, 0),
                    pos=tuple(0.2 * dirv), quat=q, material=P + "steel", **vm)
        cu.add_geom(name=P + "cutter_handle", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.045, 0.26, 0),
                    pos=tuple(0.64 * dirv), quat=q, material=P + "tomato", **vm)
        wh = wb.add_body(name=P + "wheel", pos=(0, 0, 0), mocap=True)
        wq = quat_axis_angle((1, 0, 0), math.pi / 2)
        wh.add_geom(name=P + "wheel_disc", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(self.wheel_r, 0.008, 0),
                    quat=wq, material=P + "steel", **vm)
        wh.add_geom(name=P + "wheel_hub", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.06, 0.03, 0),
                    quat=wq, material=P + "black", **vm)
        for j in range(3):  # rivets so the rolling shows
            a = j * math.tau / 3
            wh.add_geom(name=f"{P}wheel_dot{j}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.025, 0.012, 0),
                        pos=(0.17 * math.cos(a), 0, 0.17 * math.sin(a)), quat=wq, material=P + "black", **vm)
        # ---- takeaway box: tray (3 walls) + lid (with the front flap), hinge at -y
        self.box_hx, self.box_hy, self.box_wh = 0.95, 0.95, 0.15
        hx, hy, wh_ = self.box_hx, self.box_hy, self.box_wh
        bx = wb.add_body(name=P + "box", pos=(c.pizza_x, c.box_y, 0), mocap=True)
        bx.add_geom(name=P + "box_floor", type=mj.mjtGeom.mjGEOM_BOX, size=(hx, hy, 0.012),
                    pos=(0, 0, 0.012), material=P + "card", **vm)
        for s in (-1, 1):
            bx.add_geom(name=f"{P}box_side{s}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.008, hy, wh_ / 2),
                        pos=(s * hx, 0, wh_ / 2), material=P + "card", **vm)
        bx.add_geom(name=P + "box_back", type=mj.mjtGeom.mjGEOM_BOX, size=(hx, 0.008, wh_ / 2),
                    pos=(0, -hy, wh_ / 2), material=P + "card", **vm)
        ld = wb.add_body(name=P + "lid", pos=(c.pizza_x, c.box_y - hy, wh_), mocap=True)
        ld.add_geom(name=P + "lid_top", type=mj.mjtGeom.mjGEOM_BOX, size=(hx + 0.01, hy + 0.01, 0.008),
                    pos=(0, hy, 0.004), material=P + "card", **vm)
        KA.add_mesh(spec, P + "lid_print_mesh", TA.flat_quad_mesh(hy * 0.95, hx * 0.95, 0.004))
        ld.add_geom(name=P + "lid_print", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "lid_print_mesh",
                    pos=(0, hy, 0.011), material=P + "lid", **vm)
        ld.add_geom(name=P + "lid_flap", type=mj.mjtGeom.mjGEOM_BOX, size=(hx, 0.008, wh_ / 2),
                    pos=(0, 2 * hy + 0.004, -wh_ / 2), material=P + "card", **vm)
        # ---- tip jar and a pool of coins
        KA.add_mesh(spec, P + "jar_mesh", A.jar_mesh(0.28, 0.62))
        wb.add_geom(name=P + "jar", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "jar_mesh",
                    pos=(c.jar_x, c.jar_y, 0.0), material=P + "glass", **vis)
        KA.add_mesh(spec, P + "jar_label_mesh", TA.front_panel_mesh(0.14, 0.07, 0.006))
        wb.add_geom(name=P + "jar_label", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "jar_label_mesh",
                    pos=(c.jar_x + 0.29, c.jar_y, 0.4), material=P + "lbl_TIPS", **vis)
        self._coin_names = []
        for j in range(c.n_coins):
            b = wb.add_body(name=f"{P}coin{j}", pos=(0, 0, -5.0), mocap=True)
            b.add_geom(name=f"{P}coin{j}_g", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.075, 0.01, 0),
                       material=P + "gold", **vm)
            self._coin_names.append(f"{P}coin{j}")

    def _add_room(self, spec) -> None:
        """Walls (tiles behind the fly, bricks behind the oven), the chalkboard with
        its 7-segment counters, the menu, a neon sign, a flour sack, tomatoes; lights."""
        c = self.cfg
        wb = spec.worldbody
        vis = contact_kwargs("visual")
        vm = dict(vis, mass=0.0)
        xw = -2.6
        KA.add_mesh(spec, P + "wall_mesh", TA.front_panel_mesh(10.0, 4.0, 0.05))
        wb.add_geom(name=P + "wall", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "wall_mesh",
                    pos=(xw, 1.0, 4.0), material=P + "wall", **vm)
        KA.add_mesh(spec, P + "wall2_mesh", TA.front_panel_mesh(7.0, 4.0, 0.05))
        wb.add_geom(name=P + "wall2", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "wall2_mesh",
                    pos=(xw + 7.0, 6.6, 4.0), quat=quat_axis_angle((0, 0, 1), -math.pi / 2),
                    material=P + "brickwall", **vm)
        # chalkboard (behind the fly, to its right) with 3 x 3 seven-segment digits
        cb_y, cb_z, hy, hz = c.pizza_y - 2.2, 2.3, 1.55, 1.15
        KA.add_mesh(spec, P + "chalk_mesh", TA.front_panel_mesh(hy, hz, 0.05))
        wb.add_geom(name=P + "chalkboard", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "chalk_mesh",
                    pos=(xw + 0.06, cb_y, cb_z), material=P + "chalk", **vm)
        self._seg_names = []
        dw, dh, st = 0.24, 0.22, 0.035
        for di, yc in enumerate((cb_y - hy * 0.62, cb_y, cb_y + hy * 0.62)):
            disp = []
            for k in range(3):
                y_d = yc + (k - 1) * 0.32
                segs = []
                for s, (dy, dz, horiz) in SEGS.items():
                    nm = f"{P}seg{di}_{k}_{s}"
                    size = (0.01, dw / 2 - 0.02, st) if horiz else (0.01, st, dh / 2 - 0.02)
                    wb.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_BOX, size=size,
                                pos=(xw + 0.1, y_d + dy * dw, cb_z - 0.2 + dz * dh),
                                material=P + "seg_off", **vm)
                    segs.append(nm)
                disp.append(segs)
            self._seg_names.append(disp)
            if di == 2:  # the decimal point of the tips (dollars.dimes)
                wb.add_geom(name=P + "seg_dot", type=mj.mjtGeom.mjGEOM_BOX, size=(0.01, st, st),
                            pos=(xw + 0.1, yc + 0.16, cb_z - 0.2 - dh), material=P + "seg_on", **vm)
        KA.add_mesh(spec, P + "menu_mesh", TA.front_panel_mesh(0.62, 0.78, 0.03))
        wb.add_geom(name=P + "menu", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "menu_mesh",
                    pos=(xw + 0.05, c.pizza_y + 1.7, 2.2), material=P + "menu", **vm)
        KA.add_mesh(spec, P + "neon_mesh", TA.front_panel_mesh(1.3, 0.43, 0.04))
        wb.add_geom(name=P + "neon", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "neon_mesh",
                    pos=(xw + 0.06, c.pizza_y + 4.3, 3.3), material=P + "neon", **vm)
        # flour sack and tomatoes (decoration)
        sack = KA.lathe(np.array([0.0, 0.42, 0.5, 0.46, 0.3, 0.14, 0.18, 0.0]),
                        np.array([0.0, 0.0, 0.25, 0.7, 0.95, 1.05, 1.15, 1.15]), 32)
        KA.add_mesh(spec, P + "sack_mesh", sack)
        spec.add_material(name=P + "sack", rgba=(0.93, 0.9, 0.82, 1), specular=0.05)
        sx, sy = -1.0, c.pizza_y - 2.9
        wb.add_geom(name=P + "sack", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "sack_mesh",
                    pos=(sx, sy, 0.0), material=P + "sack", **vis)
        KA.add_mesh(spec, P + "sack_label_mesh", TA.front_panel_mesh(0.28, 0.12, 0.006))
        wb.add_geom(name=P + "sack_label", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "sack_label_mesh",
                    pos=(sx + 0.49, sy, 0.45), material=P + "lbl_FLOUR", **vis)
        for j, (dx, dy) in enumerate(((0.0, 0.0), (0.34, 0.18), (0.14, 0.38))):
            p = (c.pizza_x + 1.55 + dx, c.pizza_y + 1.9 + dy)
            wb.add_geom(name=f"{P}tomato{j}", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(0.17, 0.17, 0.14),
                        pos=(p[0], p[1], 0.14), material=P + "tomato", **vis)
            wb.add_geom(name=f"{P}tomato{j}_stem", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.05, 0.012, 0),
                        pos=(p[0], p[1], 0.285), material=P + "leafgreen", **vis)
        # lights: a key spot (shadows), a cool fill, dim headlight (+ the oven's glow)
        spec.visual.headlight.ambient = (0.28, 0.27, 0.26)
        spec.visual.headlight.diffuse = (0.3, 0.3, 0.3)
        spec.visual.headlight.specular = (0.1, 0.1, 0.1)
        tgt = np.array([c.pizza_x, c.pizza_y + 0.5, 0.3])
        key = np.array([c.pizza_x + 6.0, c.pizza_y - 4.0, 13.0])
        wb.add_light(name=P + "key", type=mj.mjtLightType.mjLIGHT_SPOT, pos=tuple(key),
                     dir=tuple(tgt - key), diffuse=(0.66, 0.63, 0.58), specular=(0.45, 0.45, 0.45),
                     cutoff=42.0, exponent=0.5, castshadow=bool(c.shadows))
        fill = np.array([c.pizza_x + 5.0, c.pizza_y + 6.0, 7.0])
        wb.add_light(name=P + "fill", type=mj.mjtLightType.mjLIGHT_SPOT, pos=tuple(fill),
                     dir=tuple(tgt - fill), diffuse=(0.22, 0.24, 0.3), specular=(0.1, 0.1, 0.1),
                     cutoff=50.0, exponent=1.0, castshadow=False)

    # ------------------------------------------------------------ attach / reset
    def on_attach(self) -> None:
        sim = self.sim
        m = sim.model
        c = self.cfg
        fly = sim.fly_name
        mid = lambda n: int(m.body_mocapid[m.body(n).id])  # noqa: E731
        gid = lambda n: int(m.geom(n).id)  # noqa: E731
        self.pz_mocap = mid(P + "pizza")
        self.dough_gid = np.array([gid(n) for n in self._dough_names])
        self.wedge_gid = {k: np.array([gid(n) for n in v]) for k, v in self._wedge_names.items()}
        self.sauce_gid = np.array([gid(n) for n in self._sauce_names])
        allw = np.concatenate(list(self.wedge_gid.values()))
        m.geom_sameframe[allw] = 0  # slices separate: their geom_pos is edited
        self.wedge_pos0 = {k: m.geom_pos[v].copy() for k, v in self.wedge_gid.items()}
        self.top_col = gid(P + "top_col")
        self.board_col = gid(P + "board_col")
        # tossed dough
        self.toss_bid = m.body(P + "toss").id
        j = m.joint(P + "toss_free").id
        self.toss_q, self.toss_v = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        self.toss_col = gid(P + "toss_col")
        self.toss_vis = {"dough": gid(P + "toss_dough"), "base": gid(P + "toss_base")}
        # toppings
        n = len(self._top_names)
        self.t_kind = np.array([KINDS.index(k) for k, _ in self._top_names])
        self.t_bid = np.array([m.body(nm).id for _, nm in self._top_names])
        self.t_q = np.array([int(m.jnt_qposadr[m.joint(nm + "_free").id]) for _, nm in self._top_names])
        self.t_v = np.array([int(m.jnt_dofadr[m.joint(nm + "_free").id]) for _, nm in self._top_names])
        self.t_col = np.array([gid(nm + "_g") for _, nm in self._top_names])
        self.t_vis = np.array([gid(nm + ("_g" if k == "cheese" else "_v")) for k, nm in self._top_names])
        self.t_size0 = m.geom_size[self.t_col].copy()
        self.t_state = np.zeros(n, int)  # 0 parked, 1 falling, 2 on the pizza, 3 resting off it
        self.t_tland = np.full(n, np.inf)
        self.t_rel = np.zeros((n, 3))
        self.t_relq = np.tile([1.0, 0, 0, 0], (n, 1))
        self.t_wedge = np.zeros(n, int)
        self.t_park = np.array([[-3.0 - 0.12 * i, 8.0, -3.0] for i in range(n)])
        # bowls, stream, oven fx, tools
        self.bowl_mocap = np.array([mid(nm) for nm in self._bowl_names])
        self.bowl_home = np.array([m.body_pos[m.body(nm).id] for nm in self._bowl_names])
        self.stream_mocap = mid(P + "stream")
        self.stream_gid = gid(P + "stream_g")
        m.geom_rbound[self.stream_gid] = 3.0
        self.fire_mocap = mid(P + "fire")
        self.ember_gid = np.array([gid(n) for n in self._ember_names])
        self.flame_gid = np.array([gid(n) for n in self._flame_names])
        self.flame_size0 = m.geom_size[self.flame_gid].copy()
        m.geom_rbound[self.flame_gid] = 0.5
        matid = lambda n: int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_MATERIAL, P + n))  # noqa: E731
        self.mat = {n: matid(n) for n in ("dough", "crust", "ember", "flame", "seg_on", "seg_off",
                                          "pepperoni", "basil", "cheese_shred", "melt", "sauce")}
        self.oven_light = int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_LIGHT, P + "oven_light"))
        self.fire_light = int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_LIGHT, P + "fire_light"))
        self.light0 = {i: m.light_diffuse[i].copy() for i in (self.oven_light, self.fire_light)}
        self.puff_mocap = {k: np.array([mid(n) for n in v]) for k, v in self._puff_names.items()}
        self.puff_gid = {k: np.array([gid(n + "_g") for n in v]) for k, v in self._puff_names.items()}
        for v in self.puff_gid.values():
            m.geom_rbound[v] = 1.0
        self.puffs = {"flour": _Puffs(12, (0.98, 0.97, 0.94), 0.55, 0.03, 0.12, 0.8),
                      "steam": _Puffs(14, (0.95, 0.96, 0.98), 1.2, 0.04, 0.16, 0.22),
                      "smoke": _Puffs(10, (0.35, 0.34, 0.33), 2.2, 0.1, 0.35, 0.5)}
        self.peel_mocap = mid(P + "peel")
        self.cutter_mocap = mid(P + "cutter")
        self.wheel_mocap = mid(P + "wheel")
        self.box_mocap = mid(P + "box")
        self.lid_mocap = mid(P + "lid")
        self.coin_mocap = np.array([mid(n) for n in self._coin_names])
        self.seg_gid = [[[gid(n) for n in dig] for dig in disp] for disp in self._seg_names]
        # the fly: IK, proboscis, stationary actions
        self.prob_ids = []
        for dof, sign in (("c_head-c_rostrum-pitch", -1.0), ("c_rostrum-c_haustellum-pitch", 1.0)):
            a = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, f"{fly}/{dof}-proboscispos")
            if a >= 0:
                self.prob_ids.append((a, sign))
        self.tarsus_bid = {leg: m.body(f"{fly}/{leg}_tarsus5").id for leg in ("lf", "rf")}
        from fly_simulator.jobs.dead_hang import LegIK

        self.ik = LegIK(sim, self.session.actions.body)
        st = getattr(self.session, "STATIONARY_ACTIONS", None)
        if st is not None:
            add = tuple(a for a in (ChefStance.name, ChefGroom.name) if a not in st)
            self.session.STATIONARY_ACTIONS = (*st, *add)
        self.rng = np.random.default_rng(c.seed + 77)
        # counters (O(1) memory)
        self.n_served = 0
        self.n_presses = 0
        self.n_tosses = 0
        self.n_perfect = 0
        self.n_floor_tosses = 0  # the dough landed off the board
        self.n_toss_lost = 0
        self.n_slices = 0
        self.tips_cents = 0
        self.n_dropped = 0  # toppings poured
        self.n_on_pizza = 0  # ... that landed on the pizza
        self.n_missed = 0
        self.n_tastes = 0
        self.n_stim = 0
        self.n_voided = 0
        self.n_bakes = 0
        self.n_coins_banked = 0
        self.last_toss = ""
        self.last_msg = ""
        self.taste_line = ""
        self.mn9_hz = 0.0
        self.mn9_peak = 0.0
        self.proboscis = 0.0
        self._pending_stim = None
        self._cam = None
        self._scale = 1.0
        self._last_scale = 1.0
        self.coin_k = 0
        self.on_reset()
        self.reset_props()

    def on_reset(self) -> None:
        if getattr(self, "state", "") not in ("", "starting", "settle", "dough_in"):
            self.n_voided += 1
        self.state = "settle"
        self._t_state = self.sim.time
        self._stance = None
        self._seg = {"lf": None, "rf": None}
        self._press_k = 0
        self._toss_k = 0
        self._tosses_this = 1
        self._perfect_this = 0
        self._per_until = -1.0
        self._pending_stim = None
        self._mode = "stance"
        self._pour = None
        self._carry_yaw = 0.0
        self._slide = None
        self._coin_fly: list = []
        self._box_in = None
        self._fx_t = -1.0
        self._smoke_t = -1.0
        self._steam_until = -1.0
        self._served_counted = False
        self._cut = {"k": 0}

    def reset_props(self) -> None:
        """A clean counter: no pizza, pools parked, tools home, a fresh open box."""
        d = self.sim.data
        self.bake = 0.0
        self.stage = -1  # dough stage shown (-1: none)
        self.base_on = False
        self.sauce_stage = -1  # 0-3 discs, 4 = the wedges
        self.melt = 0.0
        self.sliced = False
        self.wedge_off = np.zeros((N_WEDGES, 2))
        self.pz_pos = np.array([self.cfg.pizza_x, self.cfg.pizza_y, self.cfg.board_top])
        self.pz_yaw = 0.0
        for i in range(len(self.t_state)):
            self._park_topping(i)
        self._park_toss()
        self._toss_live = False
        self._set_topcol(False)
        for k, mid in enumerate(self.bowl_mocap):
            d.mocap_pos[mid] = self.bowl_home[k]
            d.mocap_quat[mid] = (1, 0, 0, 0)
        d.mocap_pos[self.stream_mocap] = (0, 0, -5)
        self.sim.model.geom_rgba[self.stream_gid, 3] = 0.0
        self.peel = (self.peel_home.copy(), self.cfg.board_top, self._yaw_approach)
        self._place_peel()
        self._cutter_park()
        self.box_y = self.cfg.box_y
        self.lid_ang = math.radians(110.0)
        self._place_box()
        for j, mid in enumerate(self.coin_mocap):
            d.mocap_pos[mid] = (0, 0, -5)
        self.coins_in_jar = 0
        self._apply_pizza_looks()
        self._place_pizza()
        self._update_board()

    # ------------------------------------------------------------ pools
    def _park_topping(self, i: int) -> None:
        m, d = self.sim.model, self.sim.data
        q, v = self.t_q[i], self.t_v[i]
        d.qpos[q:q + 3] = self.t_park[i]
        d.qpos[q + 3:q + 7] = (1, 0, 0, 0)
        d.qvel[v:v + 6] = 0
        m.geom_contype[self.t_col[i]] = 0
        m.geom_conaffinity[self.t_col[i]] = 0
        m.body_gravcomp[self.t_bid[i]] = 1.0
        m.geom_size[self.t_col[i]] = self.t_size0[i]
        self.t_state[i] = 0
        self.t_tland[i] = np.inf
        m.geom_rgba[self.t_vis[i]] = (1, 1, 1, 1)

    def _topping_live(self, i: int, p, vel, quat) -> None:
        m, d = self.sim.model, self.sim.data
        q, v = self.t_q[i], self.t_v[i]
        d.qpos[q:q + 3] = p
        d.qpos[q + 3:q + 7] = quat
        d.qvel[v:v + 3] = vel
        d.qvel[v + 3:v + 6] = self.rng.normal(0, 6.0, 3)
        m.geom_contype[self.t_col[i]] = 0
        m.geom_conaffinity[self.t_col[i]] = PZ_BIT | TERRAIN_BIT
        m.body_gravcomp[self.t_bid[i]] = 0.0
        self.t_state[i] = 1
        self.t_tland[i] = np.inf

    def _freeze_topping(self, i: int, on_pizza: bool) -> None:
        """Landed: ride on the pizza (kinematic carry) or rest where it fell."""
        m, d = self.sim.model, self.sim.data
        q, v = self.t_q[i], self.t_v[i]
        d.qvel[v:v + 6] = 0
        m.geom_contype[self.t_col[i]] = 0
        m.geom_conaffinity[self.t_col[i]] = 0
        m.body_gravcomp[self.t_bid[i]] = 1.0
        if on_pizza:
            rel = d.qpos[q:q + 3] - self.pz_pos
            rel[:2] = _rot2(-self.pz_yaw, rel[:2])
            self.t_rel[i] = rel
            self.t_relq[i] = _qmul(_yaw_quat(-self.pz_yaw), d.qpos[q + 3:q + 7])
            ang = math.atan2(rel[1], rel[0]) % math.tau
            self.t_wedge[i] = int(ang / (math.tau / N_WEDGES)) % N_WEDGES
            self.t_state[i] = 2
            self.n_on_pizza += 1
        else:
            self.t_state[i] = 3
            self.n_missed += 1

    def _park_toss(self) -> None:
        m, d = self.sim.model, self.sim.data
        q, v = self.toss_q, self.toss_v
        d.qpos[q:q + 3] = (self.cfg.pizza_x, 7.5, -4.0)
        d.qpos[q + 3:q + 7] = (1, 0, 0, 0)
        d.qvel[v:v + 6] = 0
        m.geom_conaffinity[self.toss_col] = 0
        m.body_gravcomp[self.toss_bid] = 1.0
        for g in self.toss_vis.values():
            m.geom_rgba[g, 3] = 0.0

    def _set_topcol(self, on: bool) -> None:
        self.sim.model.geom_contype[self.top_col] = PZ_BIT if on else 0

    # ------------------------------------------------------------ pizza looks / pose
    def _apply_pizza_looks(self) -> None:
        m = self.sim.model
        c = self.cfg
        b = self.bake
        for k, g in enumerate(self.dough_gid):
            m.geom_rgba[g, 3] = 1.0 if k == self.stage and not self.base_on else 0.0
        # base: raw dough, turning into the baked crust texture half way
        if b < 0.5:
            f = b / 0.5
            m.geom_matid[self.wedge_gid["base"]] = self.mat["dough"]
            tint = (1.0 - 0.05 * f, 1.0 - 0.14 * f, 1.0 - 0.3 * f)
        else:
            f = (b - 0.5) / 0.5
            m.geom_matid[self.wedge_gid["base"]] = self.mat["crust"]
            tint = (1.0, 0.93 + 0.07 * f, 0.86 + 0.14 * f)
        for g in self.wedge_gid["base"]:
            m.geom_rgba[g] = (*tint, 1.0 if self.base_on else 0.0)
        st = (1.0 - 0.12 * b, 1.0 - 0.2 * b, 1.0 - 0.2 * b)
        for k, g in enumerate(self.sauce_gid):
            m.geom_rgba[g] = (*st, 1.0 if k == self.sauce_stage else 0.0)
        for g in self.wedge_gid["sauce"]:
            m.geom_rgba[g] = (*st, 1.0 if self.sauce_stage >= len(self.sauce_gid) else 0.0)
        a = float(np.clip((b - 0.25) / 0.5, 0.0, 1.0)) * 0.97
        for g in self.wedge_gid["cheese"]:
            m.geom_rgba[g] = (1.0, 1.0 - 0.08 * b, 1.0 - 0.12 * b, a)
        # toppings on the pizza: shreds melt away, pepperoni / basil darken
        on = self.t_state == 2
        for i in np.flatnonzero(on):
            g = self.t_vis[i]
            k = self.t_kind[i]
            if k == 0:
                m.geom_rgba[g] = (1.0, 1.0 - 0.1 * b, 1.0 - 0.3 * b,
                                  float(np.clip(1.0 - (b - 0.35) / 0.5, 0.0, 1.0)))
            elif k == 1:
                m.geom_rgba[g] = (1.0 - 0.25 * b, 1.0 - 0.4 * b, 1.0 - 0.4 * b, 1.0)
            else:
                m.geom_rgba[g] = (1.0 - 0.45 * b, 1.0 - 0.35 * b, 1.0 - 0.45 * b, 1.0)

    def _set_wedges(self) -> None:
        m = self.sim.model
        for layer, gids in self.wedge_gid.items():
            p0 = self.wedge_pos0[layer]
            m.geom_pos[gids, 0] = p0[:, 0] + self.wedge_off[:, 0]
            m.geom_pos[gids, 1] = p0[:, 1] + self.wedge_off[:, 1]

    def _place_pizza(self) -> None:
        d = self.sim.data
        d.mocap_pos[self.pz_mocap] = self.pz_pos
        qz = _yaw_quat(self.pz_yaw)
        d.mocap_quat[self.pz_mocap] = qz
        idx = np.flatnonzero(self.t_state == 2)
        for i in idx:
            rel = self.t_rel[i].copy()
            rel[:2] += self.wedge_off[self.t_wedge[i]]
            p = self.pz_pos.copy()
            p[:2] += _rot2(self.pz_yaw, rel[:2])
            p[2] += rel[2]
            q, v = self.t_q[i], self.t_v[i]
            d.qpos[q:q + 3] = p
            d.qpos[q + 3:q + 7] = _qmul(qz, self.t_relq[i])
            d.qvel[v:v + 6] = 0

    # ------------------------------------------------------------ board (chalk counters)
    def _update_board(self) -> None:
        m = self.sim.model
        vals = (self.n_served, self.n_perfect, self.tips_cents // 10)  # tips: dollars.dimes
        for di, val in enumerate(vals):
            v = int(val) % 1000
            digits = (v // 100, (v // 10) % 10, v % 10)
            for k, dg in enumerate(digits):
                show = k == 2 or v >= 10 ** (2 - k) or (di == 2 and k == 1)
                lit = DIGITS[dg] if show else ""
                for s, g in zip(SEGS, self.seg_gid[di][k]):
                    m.geom_matid[g] = self.mat["seg_on" if s in lit else "seg_off"]

    # ------------------------------------------------------------ the fly's legs
    def _ensure_stance(self) -> ChefStance:
        acts = self.session.actions
        a = acts.action if isinstance(acts.action, ChefStance) else None
        if a is None and not isinstance(acts._pending, ChefStance):
            a = ChefStance()
            acts.trigger(a, source="job")
        elif a is None:
            a = acts._pending
        self._stance = a
        return a

    def _ik(self, leg: str, target) -> np.ndarray:
        b = self.session.actions.body
        cols = np.flatnonzero(b.leg_mask([leg]))
        q, _err = self.ik.solve(self.sim.data.qpos, leg, [("tarsus5", np.asarray(target, float))],
                                ref=b.stand[cols], ref_gain=0.1)
        return q

    def _stand_pose(self, leg: str) -> np.ndarray:
        b = self.session.actions.body
        return b.stand[np.flatnonzero(b.leg_mask([leg]))].copy()

    def _leg_now(self, leg: str) -> np.ndarray:
        st = self._stance
        if st is not None and st.pose.get(leg) is not None and st.w[leg] > 0:
            return st.pose[leg].copy()
        return self._stand_pose(leg)

    def _leg_to(self, leg: str, target, dur: float) -> None:
        """Move ``leg`` to the IK pose for world point ``target`` (None: stand)."""
        q1 = self._stand_pose(leg) if target is None else self._ik(leg, target)
        self._seg[leg] = (self._leg_now(leg), q1, self.sim.time, max(dur, 1e-3), target is None)

    def _legs_step(self) -> float:
        """Advance both legs' segments; returns the smaller progress (0..1)."""
        st = self._stance
        u_min = 1.0
        for leg, seg in self._seg.items():
            if st is None or seg is None:
                continue
            q0, q1, t0, dur, home = seg
            u = min(max((self.sim.time - t0) / dur, 0.0), 1.0)
            st.pose[leg] = q0 + smoothstep(u) * (q1 - q0)
            st.w[leg] = 1.0
            if u >= 1.0 and home:
                st.w[leg] = 0.0
                st.pose[leg] = None
                self._seg[leg] = None
            u_min = min(u_min, u)
        return u_min

    def tarsus(self, leg: str) -> np.ndarray:
        return self.sim.data.xpos[self.tarsus_bid[leg]].copy()

    # ------------------------------------------------------------ main loop (1 ms)
    def _go(self, state: str) -> None:
        self.state = state
        self._t_state = self.sim.time
        self._fresh = True

    def _first(self) -> bool:
        """True on the first update in the current state."""
        f = getattr(self, "_fresh", False)
        self._fresh = False
        return f

    def update(self) -> None:
        c = self.cfg
        t = self.sim.time
        dt = t - self._t_state
        self.steering.set(None, 0.0)
        if self._mode == "stance" and (self.state != "settle" or dt > 0.3):
            self._ensure_stance()
        s = self.state
        if s == "settle":
            if dt > 0.6:
                self._go("dough_in")
        elif s == "dough_in":
            self._dough_in(dt)
        elif s == "knead":
            self._knead(dt)
        elif s.startswith("toss"):
            self._toss(s, dt)
        elif s == "sauce":
            self._sauce(dt)
        elif s == "taste":
            self._taste(dt)
        elif s == "pour":
            self._pour_step(dt)
        elif s in ("peel_in", "carry_in", "set_down", "bake", "fetch", "carry_out", "peel_home"):
            self._oven(s, dt)
        elif s == "slice":
            self._slice(dt)
        elif s == "serve":
            self._serve(dt)
        self._legs_step()
        self._toppings_step()
        self._background(t)
        self._drive_proboscis()

    # ---- 1. dough ------------------------------------------------------------------
    def _dough_in(self, dt: float) -> None:
        c = self.cfg
        if self._first():
            self.stage, self.base_on, self.sauce_stage, self.bake, self.melt = 0, False, -1, 0.0, 0.0
            self.sliced = False
            self.wedge_off[:] = 0
            self._set_wedges()
            self.pz_yaw = float(self.rng.uniform(0, math.tau))
            self._perfect_this = 0
            self._tosses_this = 1 + int(self.rng.random() < c.bonus_toss_p)
            self._toss_k = 0
            self._apply_pizza_looks()
            self.last_msg = "a fresh dough ball!"
        u = min(dt / 0.35, 1.0)
        z = c.board_top + 2.2 * (1.0 - u * u)  # plop
        self.pz_pos = np.array([c.pizza_x, c.pizza_y, z])
        self._place_pizza()
        if u >= 1.0 and dt < 0.36:
            self._flour_puff(self.p0, 4, 0.6)
        if dt >= 0.6:
            self._press_k = 0
            self._go("knead")

    # ---- 2. knead --------------------------------------------------------------------
    def _press_point(self, k: int) -> tuple[str, np.ndarray]:
        leg = "lf" if k % 2 == 0 else "rf"
        r, h, p = self.stage_shape(max(self.stage, 0))
        a = math.pi - math.radians(38.0 if k % 4 < 2 else 22.0)
        a = a if leg == "lf" else -a
        rho = 0.3 * r
        xy = self.p0 + rho * np.array([math.cos(a), math.sin(a)])
        z = self.cfg.board_top + A.dough_top(r, h, p, rho)
        return leg, np.array([xy[0], xy[1], z])

    def _knead(self, dt: float) -> None:
        c = self.cfg
        T = c.press_s
        k = self._press_k
        leg, pt = self._press_point(k)
        ph = dt / T
        if self._first():
            self._press_ph = 0
        if self._press_ph == 0:
            self._leg_to(leg, pt + np.array([-0.05, 0, 0.28]), 0.4 * T)
            self._press_ph = 1
        elif self._press_ph == 1 and ph >= 0.4:
            self._leg_to(leg, pt + np.array([0, 0, -0.01]), 0.25 * T)
            self._press_ph = 2
        elif self._press_ph == 2 and ph >= 0.65:
            # squash: the dough flattens one stage under the press
            self.n_presses += 1
            self.stage = min(c.n_stages - 1, int(round((k + 1) * (c.n_stages - 1) / c.n_presses)))
            self._apply_pizza_looks()
            self._flour_puff(pt[:2], 2, 0.35)
            self._press_ph = 3
        elif self._press_ph == 3 and ph >= 0.75:
            self._leg_to(leg, None, 0.25 * T)
            self._press_ph = 4
        elif self._press_ph == 4 and ph >= 1.0:
            self._press_ph = 0
            self._press_k += 1
            self._t_state = self.sim.time
            if self._press_k >= c.n_presses:
                self.last_msg = "kneaded. now the toss..."
                self._go("toss_prep")

    # ---- 3. toss ---------------------------------------------------------------------
    def _toss(self, s: str, dt: float) -> None:
        c = self.cfg
        zb = c.board_top
        pc = self.pz_pos
        if s == "toss_prep":  # both legs slide under the near edge
            if self._first():
                for leg, sy in (("lf", 1), ("rf", -1)):
                    self._leg_to(leg, (pc[0] - 0.5, pc[1] + 0.28 * sy, zb + 0.07), 0.3)
            if dt >= 0.34:
                for leg, sy in (("lf", 1), ("rf", -1)):  # fling up
                    self._leg_to(leg, (pc[0] - 0.35, pc[1] + 0.32 * sy, 1.35), 0.12)
                self._launch()
                self._go("toss_air")
        elif s == "toss_air":
            if self._first():
                self._catch_set = False
            if dt >= 0.3 and not self._catch_set:
                self._catch_set = True
                for leg, sy in (("lf", 1), ("rf", -1)):  # ready to "catch"
                    self._leg_to(leg, (pc[0] - 0.45, pc[1] + 0.3 * sy, 0.5), 0.25)
            done = self._toss_check()
            if done or dt > c.toss_timeout_s:
                self._toss_landed(timeout=not done)
                for leg in ("lf", "rf"):
                    self._leg_to(leg, None, 0.3)
                self._go("toss_land")
        elif s == "toss_land":  # re-centre an off-centre landing (kinematic, labelled)
            p0, yaw0 = self._land
            u = smoothstep(min(dt / 0.45, 1.0))
            self.pz_pos = np.array([p0[0] + u * (c.pizza_x - p0[0]), p0[1] + u * (c.pizza_y - p0[1]), zb])
            self.pz_yaw = yaw0
            self._place_pizza()
            if dt >= 0.6:
                self._toss_k += 1
                if self._toss_k < self._tosses_this:
                    self.last_msg = "one more time, for the crowd!"
                    self._go("toss_prep")
                else:
                    self.sauce_stage = -1
                    self._go("sauce")

    def _launch(self) -> None:
        m, d = self.sim.model, self.sim.data
        c = self.cfg
        g = 9810.0
        vz = math.sqrt(2 * g * c.toss_height)
        T = 2 * vz / g
        err = self.rng.normal(0, c.toss_sigma, 2)
        q, v = self.toss_q, self.toss_v
        d.qpos[q:q + 3] = (self.pz_pos[0], self.pz_pos[1], c.board_top + 0.03)
        d.qpos[q + 3:q + 7] = _yaw_quat(self.pz_yaw)
        d.qvel[v:v + 3] = (err[0] / T, err[1] / T, vz)
        w = self.rng.normal(0, c.toss_wobble, 2)
        spin = c.toss_spin * (1 if self.rng.random() < 0.5 else -1)
        d.qvel[v + 3:v + 6] = (w[0], w[1], spin)
        m.geom_conaffinity[self.toss_col] = PZ_BIT | TERRAIN_BIT
        m.body_gravcomp[self.toss_bid] = 0.0
        vis = "dough" if not self.base_on else "base"
        m.geom_rgba[self.toss_vis[vis], 3] = 1.0
        self.stage = -1
        self.base_on = False
        self._apply_pizza_looks()
        self.pz_pos = np.array([self.pz_pos[0], self.pz_pos[1], -4.0])  # hidden meanwhile
        self._place_pizza()
        self._toss_live = True
        self._toss_touch_t = None
        self._toss_tilt0 = None
        self.n_tosses += 1
        self._toss_t0 = self.sim.time
        self.last_msg = "TOSS!"

    def _toss_contact(self) -> bool:
        d = self.sim.data
        g = self.toss_col
        for k in range(d.ncon):
            con = d.contact[k]
            if con.geom1 == g or con.geom2 == g:
                return True
        return False

    def _toss_tilt(self) -> float:
        R = self.sim.data.xmat[self.toss_bid].reshape(3, 3)
        return math.degrees(math.acos(min(max(abs(R[2, 2]), -1.0), 1.0)))

    def _toss_check(self) -> bool:
        d = self.sim.data
        q, v = self.toss_q, self.toss_v
        if not np.all(np.isfinite(d.qpos[q:q + 7])):
            return True
        if self._toss_touch_t is None and self._toss_contact():
            self._toss_touch_t = self.sim.time
            self._toss_tilt0 = self._toss_tilt()
        if self._toss_touch_t is None:
            return False
        speed = float(np.linalg.norm(d.qvel[v:v + 3]))
        return (self.sim.time - self._toss_touch_t > 0.05 and speed < 3.0) or \
            self.sim.time - self._toss_touch_t > 0.4

    def _toss_landed(self, timeout: bool) -> None:
        c = self.cfg
        d = self.sim.data
        q = self.toss_q
        p = d.qpos[q:q + 3].copy()
        ok = np.all(np.isfinite(d.qpos[q:q + 7]))
        if ok:
            R = d.xmat[self.toss_bid].reshape(3, 3)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            yaw = self.pz_yaw
        self._park_toss()
        self._toss_live = False
        off = float(np.hypot(p[0] - c.pizza_x, p[1] - c.pizza_y)) if ok else math.inf
        tilt = self._toss_tilt0 if self._toss_tilt0 is not None else 90.0
        if not ok or off > 6.0:
            self.n_toss_lost += 1
            p = np.array([c.pizza_x, c.pizza_y, c.board_top])
            self.last_toss = "the dough got away (a new one, re-shaped)"
        elif off <= c.perfect_off and tilt <= c.perfect_tilt_deg and not timeout:
            self.n_perfect += 1
            self._perfect_this += 1
            self.last_toss = f"PERFECT TOSS! (landed {off:.2f} mm from the centre, tilt {tilt:.0f} deg)"
        elif off > 1.15:
            self.n_floor_tosses += 1
            self.last_toss = f"oops, off the board ({off:.1f} mm): scooped back"
        else:
            self.last_toss = f"caught it, a bit off ({off:.2f} mm, tilt {tilt:.0f} deg): re-centred"
        self.last_msg = self.last_toss
        self.say(self.last_toss)
        self.base_on = True  # the toss stretched it into a base
        self.stage = -1
        self._apply_pizza_looks()
        self._land = (np.array([p[0], p[1]]), yaw)
        self.pz_pos = np.array([p[0], p[1], c.board_top])
        self.pz_yaw = yaw
        self._place_pizza()

    # ---- 4. sauce + taste ------------------------------------------------------------
    def _bowl_pose(self, k: int, tilt: float, shake: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Bowl k tipped by ``tilt`` rad toward the pizza centre (about its base)."""
        home = self.bowl_home[k]
        to = np.array([self.pz_pos[0] - home[0], self.pz_pos[1] - home[1]])
        to = to / max(np.linalg.norm(to), 1e-6)
        axis = (-to[1], to[0], 0.0)  # tilting about it moves the rim toward the pizza
        q = np.array(quat_axis_angle(axis, tilt + shake))
        pos = home.copy()
        pos[:2] += 0.12 * math.sin(tilt) * to  # leans out a little
        return pos, q

    def _lip(self, k: int, tilt: float) -> np.ndarray:
        pos, q = self._bowl_pose(k, tilt)
        home = self.bowl_home[k]
        to = np.array([self.pz_pos[0] - home[0], self.pz_pos[1] - home[1], 0.0])
        to /= max(np.linalg.norm(to), 1e-6)
        R = np.empty(9)
        mj.mju_quat2Mat(R, q)
        local = to * self.bowl_r + np.array([0, 0, self.bowl_h])
        return pos + R.reshape(3, 3) @ local

    def _place_bowl(self, k: int, tilt: float, shake: float = 0.0) -> None:
        d = self.sim.data
        pos, q = self._bowl_pose(k, tilt, shake)
        d.mocap_pos[self.bowl_mocap[k]] = pos
        d.mocap_quat[self.bowl_mocap[k]] = q

    def _sauce(self, dt: float) -> None:
        c = self.cfg
        m, d = self.sim.model, self.sim.data
        T = c.sauce_s
        tilt = math.radians(70.0) * smoothstep(min(dt / 0.3, 1.0)) * smoothstep(min((T - dt) / 0.25, 1.0))
        self._place_bowl(0, tilt, 0.04 * math.sin(60 * dt))
        u = np.clip((dt - 0.3) / (T - 0.6), 0.0, 1.0)
        n = len(self.sauce_gid)
        if 0.0 < u < 1.0:
            lip = self._lip(0, tilt)
            rho = 0.55 * u
            ang = self.pz_yaw + 5.5 * math.tau * u
            end = np.array([self.pz_pos[0] + rho * math.cos(ang), self.pz_pos[1] + rho * math.sin(ang),
                            self.pz_pos[2] + c.base_h])
            mid = 0.5 * (lip + end)
            seg = end - lip
            L = float(np.linalg.norm(seg))
            z = np.array([0, 0, 1.0])
            ax = np.cross(z, seg / L)
            s = float(np.linalg.norm(ax))
            ang_z = math.atan2(s, float(z @ seg / L))
            d.mocap_pos[self.stream_mocap] = mid
            d.mocap_quat[self.stream_mocap] = quat_axis_angle(ax if s > 1e-9 else (1, 0, 0), ang_z)
            m.geom_size[self.stream_gid, 1] = 0.5 * L
            m.geom_size[self.stream_gid, 0] = 0.028
            m.geom_rgba[self.stream_gid, 3] = 1.0
            st = min(int(u * (n + 1)), n)
            if st != self.sauce_stage:
                self.sauce_stage = st
                self._apply_pizza_looks()
        else:
            m.geom_rgba[self.stream_gid, 3] = 0.0
            d.mocap_pos[self.stream_mocap] = (0, 0, -5)
        if dt >= T:
            self.sauce_stage = n
            self._apply_pizza_looks()
            self._place_bowl(0, 0.0)
            self.last_msg = "sauce! chef tastes it..."
            self._go("taste")

    def _taste(self, dt: float) -> None:
        c = self.cfg
        p = self.pz_pos
        dip = np.array([p[0] - 0.46, p[1] + 0.2, c.board_top + c.base_h + c.sauce_t + 0.005])
        if self._first():
            self._leg_to("lf", dip + np.array([0, 0, 0.25]), 0.25)
            self._taste_ph = 0
        if self._taste_ph == 0 and dt >= 0.25:
            self._leg_to("lf", dip, 0.15)
            self._taste_ph = 1
        elif self._taste_ph == 1 and dt >= 0.42:  # the tarsus is in the sauce: taste
            self.n_tastes += 1
            self._pending_stim = self.run_time()
            self.mn9_peak = 0.0
            self._taste_t0 = self.sim.time
            if self._brain() is None:
                self._per_until = self.sim.time + 0.9  # scripted dab (no brain), labelled
            self._taste_ph = 2
        elif self._taste_ph == 2 and dt >= 0.42 + c.taste_s:
            self._leg_to("lf", None, 0.3)
            self._taste_ph = 3
        elif self._taste_ph == 3 and dt >= 1.35:
            if self._brain() is not None:
                verdict = "mmm, perfetto" if self.mn9_peak >= 30 else "hmm, needs sugar?"
                self.taste_line = (f"sauce taste #{self.n_tastes}: MN9 peak {self.mn9_peak:.0f} Hz -> "
                                   f"'{verdict}' (real connectome; stand-in: LB3 sugar GRNs)")
            else:
                self.taste_line = f"sauce taste #{self.n_tastes}: 'mmm' [scripted: no brain]"
            self.say(self.taste_line)
            self._taste_ph = 9
            self._pour = {"k": 0}
            self._go("pour")

    # ---- 5. toppings -------------------------------------------------------------------
    def _pour_step(self, dt: float) -> None:
        c = self.cfg
        k = self._pour["k"]  # 0 cheese, 1 pepperoni, 2 basil (bowls 1-3)
        bowl = k + 1
        T = c.pour_s[k]
        t_tip, t_back = 0.22, 0.2
        idx = np.flatnonzero((self.t_kind == k) & (self.t_state == 0))
        if self._first() or "n" not in self._pour:
            self._pour.update(n=0, total=int(np.sum(self.t_kind == k)), t0=self.sim.time)
            self._set_topcol(True)
            self.last_msg = f"{KINDS[k]} rain!"
        tilt = math.radians(80.0) * smoothstep(min(dt / t_tip, 1.0)) * \
            smoothstep(min(max(t_tip + T + t_back - dt, 0.0) / t_back, 1.0))
        self._place_bowl(bowl, tilt, 0.06 * math.sin(70 * dt))
        # pour the pool evenly over the window
        want = int(min(max((dt - t_tip) / T, 0.0), 1.0) * self._pour["total"] + 1e-9)
        while self._pour["n"] < want and len(idx):
            i = idx[0]
            idx = idx[1:]
            self._drop(i, bowl, tilt)
            self._pour["n"] += 1
        live = np.any(self.t_state == 1)
        if dt >= t_tip + T + t_back and not live:
            self._place_bowl(bowl, 0.0)
            if k + 1 < len(KINDS):
                self._pour = {"k": k + 1}
                self._go("pour")
            else:
                self._set_topcol(False)
                self.last_msg = "into the oven!"
                self._go("peel_in")

    def _drop(self, i: int, bowl: int, tilt: float) -> None:
        """Launch topping i from the bowl lip, aimed at a random spot on the pizza
        (the aim is set by the job; the fall and landing are physics)."""
        c = self.cfg
        lip = self._lip(bowl, tilt) + self.rng.normal(0, 0.03, 3)
        rr = 0.62 * math.sqrt(self.rng.random()) if self.t_kind[i] == 0 else \
            0.5 * math.sqrt(self.rng.uniform(0.05, 1.0))
        a = self.rng.uniform(0, math.tau)
        tgt = np.array([self.pz_pos[0] + rr * math.cos(a), self.pz_pos[1] + rr * math.sin(a)])
        z_land = self.pizza_top + 0.02
        H = max(lip[2] - z_land, 0.05)
        T = math.sqrt(2 * H / 9810.0)
        vxy = (tgt - lip[:2]) / T
        quat = _qmul(_yaw_quat(self.rng.uniform(0, math.tau)),
                     quat_axis_angle((1, 0, 0), float(self.rng.normal(0, 0.4))))
        self._topping_live(i, lip, (vxy[0], vxy[1], 0.0), quat)
        self.n_dropped += 1

    def _toppings_step(self) -> None:
        """Falling toppings: settled (or on a contact long enough) -> ride / rest."""
        live = np.flatnonzero(self.t_state == 1)
        if not len(live):
            return
        d = self.sim.data
        c = self.cfg
        t = self.sim.time
        for i in live:
            q, v = self.t_q[i], self.t_v[i]
            p = d.qpos[q:q + 3]
            if not np.all(np.isfinite(d.qpos[q:q + 7])) or p[2] < -1.0 or abs(p[0]) > 30 or abs(p[1]) > 30:
                self._park_topping(i)
                self.n_missed += 1
                continue
            low = p[2] < self.pizza_top + 0.06
            if low and not math.isfinite(self.t_tland[i]):
                self.t_tland[i] = t
            slow = float(np.linalg.norm(d.qvel[v:v + 3])) < 1.5
            if math.isfinite(self.t_tland[i]) and (t - self.t_tland[i] >= c.topping_settle_s or
                                                   (slow and t - self.t_tland[i] > 0.03)):
                on = math.hypot(p[0] - self.pz_pos[0], p[1] - self.pz_pos[1]) < c.pizza_r - 0.03 and \
                    p[2] > self.pizza_top - 0.03
                self._freeze_topping(i, on)

    # ---- 6. oven -------------------------------------------------------------------------
    @property
    def _yaw_approach(self) -> float:
        u = -self.u_dir
        return math.atan2(u[1], u[0])

    @property
    def _yaw_oven(self) -> float:
        m = -self.mouth
        return math.atan2(m[1], m[0])

    def _place_peel(self) -> None:
        d = self.sim.data
        xy, z, yaw = self.peel
        d.mocap_pos[self.peel_mocap] = (xy[0], xy[1], z)
        d.mocap_quat[self.peel_mocap] = _yaw_quat(yaw)

    def _path(self, s: float) -> tuple[np.ndarray, float, float]:
        """The carry path, s = 0 (prep spot) .. 1 (baking spot): (xy, z, peel yaw)."""
        c = self.cfg
        a, b, e = self.p0, self.turn_pt, self.bake_pt
        L1, L2 = float(np.linalg.norm(b - a)), float(np.linalg.norm(e - b))
        x = s * (L1 + L2)
        xy = a + (b - a) * (x / L1) if x <= L1 else b + (e - b) * ((x - L1) / L2)
        zlo, zhi = c.board_top, c.hearth_z + 0.004
        z = zlo + (zhi - zlo) * smoothstep(min(x / (0.6 * L1), 1.0))
        f = smoothstep(np.clip((x - 0.55 * L1) / (0.45 * L1 + 0.35 * L2), 0.0, 1.0))
        dyaw = wrap_angle(self._yaw_oven - self._yaw_approach)
        return xy, z, self._yaw_approach + f * dyaw

    def _carry(self) -> None:
        xy, z, yaw = self.peel
        self.pz_pos = np.array([xy[0], xy[1], z + 0.001])
        self.pz_yaw = yaw + self._carry_yaw
        self._place_pizza()

    def _oven(self, s: str, dt: float) -> None:
        c = self.cfg
        zb = c.board_top
        Tc = 1.9
        if s == "peel_in":  # slide the peel under the pizza
            u = smoothstep(min(dt / 0.7, 1.0))
            h = self.peel_home
            self.peel = (h + u * (self.p0 - h), zb, self._yaw_approach)
            self._place_peel()
            if dt >= 0.75:
                self._carry_yaw = self.pz_yaw - self._yaw_approach
                self._go("carry_in")
        elif s == "carry_in":
            xy, z, yaw = self._path(smoothstep(min(dt / Tc, 1.0)))
            self.peel = (xy, z, yaw)
            self._place_peel()
            self._carry()
            if dt >= Tc:
                self._go("set_down")
        elif s == "set_down":  # leave it on the hearth, pull the peel back out
            e, b = self.bake_pt, self.turn_pt
            u = smoothstep(min(dt / 0.7, 1.0))
            xy = e + u * (b - e)
            self.peel = (xy, c.hearth_z + 0.004 - 0.02 * min(dt / 0.1, 1.0) - (c.hearth_z - zb) * u * u,
                         self._yaw_oven)
            self._place_peel()
            if dt >= 0.75:
                self.n_bakes += 1
                self._mode = "groom"
                self.session.actions.trigger(ChefGroom(duration=max(c.bake_s - 0.4, 0.5)), source="job")
                self._stance = None
                self._seg = {"lf": None, "rf": None}
                self.last_msg = "baking... (the chef grooms: a real recorded grooming bout)"
                self._go("bake")
        elif s == "bake":
            self.bake = min(dt / c.bake_s, 1.0)
            if int(dt * 20) != int((dt - 0.001) * 20):
                self._apply_pizza_looks()
            if dt >= c.bake_s:
                self.bake = 1.0
                self._apply_pizza_looks()
                self._mode = "stance"
                self._go("fetch")
        elif s == "fetch":
            e, b = self.bake_pt, self.turn_pt
            u = smoothstep(min(dt / 0.7, 1.0))
            xy = b + u * (e - b)
            z = zb + (c.hearth_z + 0.004 - zb) * smoothstep(min(dt / 0.35, 1.0))
            self.peel = (xy, z, self._yaw_oven)
            self._place_peel()
            if dt >= 0.75:
                self._carry_yaw = self.pz_yaw - self._yaw_oven
                self._go("carry_out")
        elif s == "carry_out":
            u = smoothstep(min(dt / Tc, 1.0))
            xy, z, yaw = self._path(1.0 - u)
            self.peel = (xy, z, yaw)
            self._place_peel()
            self._carry()
            if u > 0.02:
                self._steam_until = self.sim.time + 6.0
            if dt >= Tc:
                self.pz_pos[2] = zb
                self._place_pizza()
                self.last_msg = "out of the oven! bubbling hot"
                self._go("peel_home")
        elif s == "peel_home":
            u = smoothstep(min(dt / 0.6, 1.0))
            h = self.peel_home
            self.peel = (self.p0 + u * (h - self.p0), zb, self._yaw_approach)
            self._place_peel()
            if dt >= 0.65:
                self._cut = {"k": 0}
                self._go("slice")

    # ---- 7. slice --------------------------------------------------------------------
    def _cutter_park(self) -> None:
        d = self.sim.data
        c = self.cfg
        hub = np.array([c.pizza_x - 0.3, c.pizza_y + 1.65, self.wheel_r])
        self._place_cutter(hub, math.radians(160.0), 0.0)

    def _place_cutter(self, hub, yaw: float, roll: float) -> None:
        d = self.sim.data
        d.mocap_pos[self.cutter_mocap] = hub
        d.mocap_quat[self.cutter_mocap] = _yaw_quat(yaw)
        d.mocap_pos[self.wheel_mocap] = hub
        d.mocap_quat[self.wheel_mocap] = _qmul(_yaw_quat(yaw), quat_axis_angle((0, 1, 0), roll))

    def _slice(self, dt: float) -> None:
        c = self.cfg
        k = self._cut["k"]
        L = c.pizza_r + 0.2
        a = self.pz_yaw + k * math.pi / 4  # cuts at 0, 45, 90, 135 deg: 8 slices
        dvec = np.array([math.cos(a), math.sin(a)])
        z_cut = self.pizza_top + self.wheel_r - 0.05
        start = np.r_[self.p0 - L * dvec, z_cut]
        end = np.r_[self.p0 + L * dvec, z_cut]
        t_move, t_roll = 0.28, 0.45
        if self._first() or "from" not in self._cut:
            d = self.sim.data
            self._cut["from"] = d.mocap_pos[self.cutter_mocap].copy()
            self._cut["yaw0"] = float(2 * math.atan2(d.mocap_quat[self.cutter_mocap][3],
                                                    d.mocap_quat[self.cutter_mocap][0]))
            self._cut["roll"] = self._cut.get("roll", 0.0)
            if k == 0:
                self.last_msg = "slicing: 8 slices"
        yaw = a
        if dt < t_move:  # swing over to the start of this cut
            u = smoothstep(dt / t_move)
            p = self._cut["from"] + u * (start + np.array([0, 0, 0.25]) - self._cut["from"])
            p[2] += 0.3 * math.sin(math.pi * u)
            yw = self._cut["yaw0"] + u * wrap_angle(yaw - self._cut["yaw0"])
            self._place_cutter(p, yw, self._cut["roll"])
        elif dt < t_move + 0.08:
            u = (dt - t_move) / 0.08
            self._place_cutter(start + np.array([0, 0, 0.25 * (1 - u)]), yaw, self._cut["roll"])
        elif dt < t_move + 0.08 + t_roll:
            u = smoothstep((dt - t_move - 0.08) / t_roll)
            p = start + u * (end - start)
            roll = self._cut["roll"] + u * 2 * L / self.wheel_r
            self._place_cutter(p, yaw, roll)
            if not self.sliced:
                self.sliced = True
        else:
            self._cut["roll"] += 2 * L / self.wheel_r
            # the slices on each side of the cut part a little
            n = np.array([-math.sin(k * math.pi / 4), math.cos(k * math.pi / 4)])
            for i in range(N_WEDGES):
                ca = (i + 0.5) * math.tau / N_WEDGES
                side = 1.0 if (math.cos(ca) * n[0] + math.sin(ca) * n[1]) > 0 else -1.0
                self.wedge_off[i] += 0.012 * side * n
            self._set_wedges()
            self._place_pizza()
            self._cut = {"k": k + 1, "roll": self._cut["roll"]}
            self._t_state = self.sim.time
            if k + 1 >= 4:
                for i in range(N_WEDGES):
                    ca = (i + 0.5) * math.tau / N_WEDGES
                    self.wedge_off[i] += 0.035 * np.array([math.cos(ca), math.sin(ca)])
                self._set_wedges()
                self._place_pizza()
                self.n_slices += N_WEDGES
                self._cutter_park()
                self.last_msg = "8 slices. order up!"
                self._go("serve")

    # ---- 8. serve --------------------------------------------------------------------
    def _place_box(self) -> None:
        d = self.sim.data
        c = self.cfg
        d.mocap_pos[self.box_mocap] = (c.pizza_x, self.box_y, 0.0)
        d.mocap_pos[self.lid_mocap] = (c.pizza_x, self.box_y - self.box_hy, self.box_wh)
        d.mocap_quat[self.lid_mocap] = quat_axis_angle((1, 0, 0), self.lid_ang)

    def _serve(self, dt: float) -> None:
        c = self.cfg
        zbox = 0.026
        t_slide, t_lid, t_away = 1.0, 0.5, 1.0
        if dt < t_slide:  # slides off the board into the box
            u = smoothstep(dt / t_slide)
            self.pz_pos = np.array([c.pizza_x, c.pizza_y + u * (c.box_y - c.pizza_y),
                                    c.board_top + (zbox - c.board_top) * u])
            self._place_pizza()
        elif dt < t_slide + t_lid:
            u = smoothstep((dt - t_slide) / t_lid)
            self.lid_ang = math.radians(110.0) * (1 - u)
            if u > 0.6:
                self._steam_until = -1.0
            self._place_box()
        elif dt < t_slide + t_lid + t_away:
            if not getattr(self, "_served_counted", False):
                self._served_counted = True
                self._count_served()
            u = smoothstep((dt - t_slide - t_lid) / t_away)
            self.box_y = c.box_y - 5.0 * u * u
            self.lid_ang = 0.0
            self._place_box()
            self.pz_pos = np.array([c.pizza_x, self.box_y, zbox])
            self._place_pizza()
        else:
            self._served_counted = False
            # the box is gone: recycle the pizza's pieces, a new open box slides in
            for i in np.flatnonzero(self.t_state > 0):
                self._park_topping(i)
            self.base_on, self.sauce_stage, self.bake, self.stage = False, -1, 0.0, -1
            self._apply_pizza_looks()
            self.pz_pos = np.array([c.pizza_x, c.pizza_y, -4.0])
            self._place_pizza()
            self._box_in = self.sim.time
            self._go("dough_in")

    def _count_served(self) -> None:
        c = self.cfg
        self.n_served += 1
        self.add_work(1)
        on = int(np.sum(self.t_state == 2))
        cov = on / max(len(self.t_state), 1)
        tip = 25 + 25 * (self._perfect_this >= self._tosses_this) + int(round(25 * cov))
        self.tips_cents += tip
        n_coins = 1 + (self._perfect_this >= self._tosses_this) + (cov > 0.85)
        for j in range(n_coins):
            self._coin_fly.append((self.sim.time + 0.12 * j, self.coin_k % c.n_coins, self.coins_in_jar))
            self.coin_k += 1
            self.coins_in_jar += 1
            if self.coins_in_jar >= c.n_coins:
                self.coins_in_jar = 0
                self.n_coins_banked += c.n_coins
        self._update_board()
        self.last_msg = (f"SERVED pizza #{self.n_served}! tip {tip} fly cents "
                         f"({on} toppings on it)")
        self.say(self.last_msg)

    # ------------------------------------------------------------ background fx (every 1 ms)
    def _background(self, t: float) -> None:
        m, d = self.sim.model, self.sim.data
        c = self.cfg
        # a new box slides in from the pick-up side
        if self._box_in is not None:
            u = smoothstep(min((t - self._box_in) / 0.8, 1.0))
            self.box_y = c.box_y - 5.0 * (1 - u)
            self.lid_ang = math.radians(110.0)
            self._place_box()
            if u >= 1.0:
                self._box_in = None
        # coins flying into the jar
        for item in list(self._coin_fly):
            t0, j, slot = item
            tt = (t - t0) / 0.55
            if tt < 0:
                continue
            layer, pos_k = divmod(slot, 3)
            a = pos_k * math.tau / 3 + 0.9 * layer
            end = np.array([c.jar_x + 0.1 * math.cos(a), c.jar_y + 0.1 * math.sin(a), 0.03 + 0.022 * layer])
            start = np.array([c.pizza_x - 0.3, c.box_y - 0.9, 1.2])
            u = min(tt, 1.0)
            p = start + u * (end - start)
            p[2] += 0.9 * math.sin(math.pi * u)
            d.mocap_pos[self.coin_mocap[j]] = p
            d.mocap_quat[self.coin_mocap[j]] = quat_axis_angle((1, 0, 0), 9.0 * u * (1 - u))
            if tt >= 1.0:
                d.mocap_quat[self.coin_mocap[j]] = quat_axis_angle((0, 0, 1), a)
                self._coin_fly.remove(item)
                if slot == 0:  # the jar was emptied (tips banked): hide the old coins
                    for jj in range(c.n_coins):
                        if jj != j:
                            d.mocap_pos[self.coin_mocap[jj]] = (0, 0, -5)
        if t - self._fx_t < 0.02:
            return
        self._fx_t = t
        # oven flicker: embers, flames, lights (brighter while baking)
        hot = 1.0 + 0.5 * (self.state == "bake")
        fl = 0.75 + 0.25 * math.sin(23 * t) * math.sin(7.3 * t + 1.0) + 0.1 * self.rng.random()
        m.mat_emission[self.mat["ember"]] = min(0.55 + 0.4 * fl * hot, 1.0)
        m.mat_rgba[self.mat["ember"], 1] = 0.3 + 0.2 * fl
        for j, g in enumerate(self.flame_gid):
            s = 0.75 + 0.35 * math.sin(17 * t + 1.7 * j) + 0.15 * self.rng.random()
            m.geom_size[g] = self.flame_size0[j] * np.array([1.0, 1.0, s * (0.8 + 0.3 * hot)])
        for i in (self.oven_light, self.fire_light):
            m.light_diffuse[i] = self.light0[i] * (0.65 + 0.35 * fl) * hot
        # puffs
        if t - self._smoke_t > (0.35 if self.state == "bake" else 0.8):
            self._smoke_t = t
            top = np.array([c.oven_x, c.oven_y, c.hearth_z + c.oven_H + 0.62])
            self.puffs["smoke"].spawn(t, top, (0.1, 0.05, 0.9))
        if t < self._steam_until and self.rng.random() < 0.35:
            p = self.pz_pos.copy()
            a, r = self.rng.uniform(0, math.tau), 0.5 * math.sqrt(self.rng.random())
            p[:2] += r * np.array([math.cos(a), math.sin(a)])
            p[2] += 0.1
            self.puffs["steam"].spawn(t, p, (self.rng.normal(0, 0.1), self.rng.normal(0, 0.1), 0.9))
        for k, pf in self.puffs.items():
            pf.draw(t, m, d, self.puff_mocap[k], self.puff_gid[k])

    def _flour_puff(self, xy, n: int, spread: float) -> None:
        t = self.sim.time
        for _ in range(n):
            v = np.r_[self.rng.normal(0, spread, 2), self.rng.uniform(0.3, 0.7)]
            self.puffs["flour"].spawn(t, np.r_[xy, self.cfg.board_top + 0.08], v)

    # ------------------------------------------------------------ brain / proboscis
    def _brain(self):
        return getattr(self.session, "brain", None)

    def _drive_proboscis(self) -> None:
        c = self.cfg
        target = 0.0
        if self._brain() is not None:
            target = min(max(self.mn9_hz / max(c.mn9_ref_hz, 1e-6), 0.0), 1.0)
        if self.sim.time < self._per_until:
            target = max(target, 0.85)
        a = min(0.001 * c.update_every_steps / 0.1, 1.0)
        self.proboscis += a * (target - self.proboscis)
        d = self.sim.data
        for aid, sign in self.prob_ids:
            d.ctrl[aid] = sign * self.proboscis

    def after_physics(self) -> None:
        super().after_physics()
        link = self._brain()
        if link is not None:
            lt = getattr(link, "latest", None)
            if lt is not None:
                self.mn9_hz = float((getattr(lt, "probes", None) or {}).get("MN9", 0.0))
            if self.state in ("taste", "pour"):
                self.mn9_peak = max(self.mn9_peak, self.mn9_hz)
            if self._pending_stim is not None:
                self._send_taste(link, self._pending_stim)
        else:
            self.mn9_hz = 0.0
        self._pending_stim = None

    def _send_taste(self, link, t0: float) -> None:
        from fly_simulator.brain.schema import StimulusEvent

        c = self.cfg
        det = {"tastes": ["sugar"], "sugar_hz": c.sugar_hz,
               "label": "SAUCE TASTE (tarsal dip; stand-in: labellar LB3 sugar GRNs)"}
        link.send(StimulusEvent("taste", "left", 1.0, c.taste_s, t0, details=det), source="job")
        self.n_stim += 1
        log = getattr(link, "stim_log", None)
        if isinstance(log, list) and len(log) > 400:
            del log[:-200]

    # ------------------------------------------------------------ edit effects (labelled)
    def _slowmo_target(self) -> float:
        c = self.cfg
        if not c.slowmo:
            return 1.0
        if self.state == "toss_air" and self._toss_live and self._toss_touch_t is None:
            return c.toss_slowmo
        if self.state == "pour" and (np.any(self.t_state == 1) or
                                     (self._pour and self._pour.get("n", 0) < self._pour.get("total", 1)
                                      and self.sim.time - self._t_state > 0.2)):
            return c.pour_slowmo
        return 1.0

    def time_scale(self, present_dt: float) -> float:
        """Slow motion while the dough flies and the toppings fall (an *edit effect*:
        fewer physics steps per displayed frame; the physics is unchanged)."""
        tgt = self._slowmo_target()
        if tgt < self._scale:
            self._scale = tgt  # cut straight into slow motion
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
    def _shot(self) -> tuple[np.ndarray, CameraPreset]:
        c = self.cfg
        s = self.state
        pz = np.array([c.pizza_x, c.pizza_y])
        if not c.close_ups:
            return np.array([1.9, 0.6, 0.8]), CameraPreset(azimuth=162.0, elevation=-21.0, distance=9.0)
        late_knead = s == "knead" and self._press_k >= c.n_presses - 2
        if s.startswith("toss") or late_knead:  # low angle: the dough flies up into view
            return np.array([pz[0] - 0.3, pz[1], 1.55]), CameraPreset(azimuth=160.0, elevation=-3.0,
                                                                       distance=6.6)
        if s in ("dough_in", "knead", "settle") or s == "taste":
            return np.array([pz[0] - 0.45, pz[1], 0.5]), CameraPreset(azimuth=158.0, elevation=-20.0,
                                                                       distance=3.8)
        if s in ("sauce", "pour"):
            return np.array([pz[0] - 0.1, pz[1], 0.75]), CameraPreset(azimuth=160.0, elevation=-20.0,
                                                                       distance=5.0)
        if s in ("peel_in", "carry_in", "set_down", "bake", "fetch", "carry_out"):
            return np.array([c.pizza_x - 0.1, 1.9, 0.8]), CameraPreset(azimuth=150.0, elevation=-18.0,
                                                                        distance=8.2)
        if s in ("peel_home", "slice"):
            return np.array([pz[0] - 0.1, pz[1], 0.5]), CameraPreset(azimuth=160.0, elevation=-30.0,
                                                                      distance=5.0)
        return np.array([c.pizza_x + 0.1, -1.1, 0.7]), CameraPreset(azimuth=170.0, elevation=-22.0,
                                                                     distance=7.4)

    def camera_target(self) -> np.ndarray:
        return self._shot()[0]

    def camera_preset(self) -> CameraPreset:
        """The shot for the current step, gliding (distance / angles smoothed here;
        the look-at point is smoothed by JobCamera)."""
        _, pre = self._shot()
        t = self.run_time()
        want = np.array([pre.azimuth, pre.elevation, pre.distance])
        if self._cam is None or t < self._cam[0]:
            self._cam = (t, want)
        else:
            t0, cur = self._cam
            a = 1.0 - math.exp(-(t - t0) / 0.7)
            self._cam = (t, cur + a * (want - cur))
        az, el, dist = self._cam[1]
        return CameraPreset(azimuth=float(az), elevation=float(el), distance=float(dist), tau_s=0.7)

    def job_hud_lines(self) -> list[str]:
        brain = self._brain() is not None
        lines = [
            f"dough tosses {self.n_tosses}   perfect {self.n_perfect}   off the board "
            f"{self.n_floor_tosses}   slices {self.n_slices}   tips ${self.tips_cents / 100:.2f}",
            f"kneads {self.n_presses}   toppings dropped {self.n_dropped} (on the pizza "
            f"{self.n_on_pizza}, missed {self.n_missed})   bakes {self.n_bakes}",
        ]
        if self.last_msg:
            lines.append(self.last_msg)
        if self.taste_line:
            lines.append(self.taste_line)
        lines.append((f"MN9 {self.mn9_hz:.0f} Hz (live)   proboscis {self.proboscis:.0%}" if brain
                      else f"proboscis {self.proboscis:.0%}   (sauce taste: --brain for the connectome)"))
        lines.append("(toss launch set by the job, flight real physics; peel / cutter / box / bowls "
                     "kinematic; slow motion: edit)")
        return lines

    def job_stats(self) -> dict:
        return {"served": self.n_served, "kneads": self.n_presses, "tosses": self.n_tosses,
                "perfect_tosses": self.n_perfect, "floor_tosses": self.n_floor_tosses,
                "tosses_lost": self.n_toss_lost, "slices": self.n_slices,
                "tips_cents": self.tips_cents, "toppings_dropped": self.n_dropped,
                "toppings_on_pizza": self.n_on_pizza, "toppings_missed": self.n_missed,
                "bakes": self.n_bakes, "tastes": self.n_tastes, "stimuli": self.n_stim,
                "voided": self.n_voided}
