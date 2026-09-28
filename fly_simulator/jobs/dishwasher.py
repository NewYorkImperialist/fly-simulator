"""DISHWASHER FLY: the fly washes dishes forever (docs/JOBS.md, "dishwasher").

A warm home kitchen at fly scale: a butcher-block counter with a stainless sink of
soapy water under a window, a running faucet, a dish rack.

* **The fly** stands at the sink in ``DishStance`` (all tarsi planted and adhering,
  like ``freeze``; stationary) while the job drives both front legs with joint
  targets from damped least-squares IK (the dead_hang ``LegIK``), interpolated in
  joint space (``LegDriver``).
* **Fetch (kinematic, labelled).** The left front leg reaches the top plate of the
  dirty stack, the plate follows the tarsus (a kinematic carry) to the wash spot in
  the water.
* **Scrub (real contact).** A sponge sits on the right front tarsus (a mocap prop).
  The leg presses on the plate and scrubs it in circles; the plate's hidden collider
  (``dish/pad``, contact kind "fly") is raised by the sponge's thickness, so the
  tarsus touches it where the sponge meets the plate. **The grime fades only while
  the tarsus is in contact with the plate**, by the distance scrubbed (grime per mm,
  ``scrub_per_mm``): the plate's texture steps through 8 stages in which the smears
  break up and vanish. Foam bubbles spawn at the sponge while it scrubs.
* **Rinse, rack, cart (kinematic, labelled).** The plate carrier slides the plate
  under the faucet (a water column, drops and splashes: visual particles, bounded
  pools), then into the next slot of the dish rack. A full rack is carted off and
  comes back empty; an empty stack is replaced by a new dirty stack on the conveyor.

Counters: plates washed (the work counter), grime removed (%), strokes, sponge wear
and sponges used, rack loads, stacks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mujoco as mj
import numpy as np

from fly_simulator.actions.base import smoothstep
from fly_simulator.jobs import dishwasher_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.dj import DJStance
from fly_simulator.jobs.geometry import contact_kwargs, quat_axis_angle, spot_or_directional
from fly_simulator.jobs.registry import register_job
from fly_simulator.jobs.taste_tester import DIGITS, SEGS

P = "dish/"
FRONT = ("lf", "rf")
IDENT = np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# the stance and the leg driver (shared with the barista job)
# ---------------------------------------------------------------------------


class DishStance(DJStance):
    """All tarsi planted and adhering (like ``freeze``) while the job moves the front
    legs (``pose`` / ``w`` / ``grip`` per leg, see ``DJStance``). Stationary."""

    name = "dish_stance"


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    if q0 @ q1 < 0:
        q1 = -q1
    d = float(np.clip(q0 @ q1, -1.0, 1.0))
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


class LegDriver:
    """Front-leg motions for a stance job: ``move(leg, target, dur)`` solves the IK for
    the tarsus5 world point ``target`` (cached; ``None`` = back to the stand pose) and
    eases the leg there in joint space; ``step()`` writes the stance's targets.
    ``grip[leg]``: the driven leg's adhesion (off by default)."""

    def __init__(self, job: EternalJob) -> None:
        from fly_simulator.jobs.dead_hang import LegIK

        self.job = job
        self.sim = job.sim
        self.body = job.session.actions.body
        self.ik = LegIK(self.sim, self.body)
        self.seg: dict[str, tuple | None] = {leg: None for leg in FRONT}
        self.cache: dict[tuple, tuple[np.ndarray, float]] = {}
        self.stance: DJStance | None = None
        self.grip = {leg: False for leg in FRONT}
        self.max_err = 0.0
        self.cols = {leg: np.flatnonzero(self.body.leg_mask([leg])) for leg in FRONT}

    def stand(self, leg: str) -> np.ndarray:
        return self.body.stand[self.cols[leg]].copy()

    def solve(self, leg: str, p) -> np.ndarray:
        key = (leg,) + tuple(round(float(v), 3) for v in p)
        hit = self.cache.get(key)
        if hit is None:
            q, err = self.ik.solve(self.sim.data.qpos, leg, [("tarsus5", np.asarray(p, float))],
                                   ref=self.stand(leg), ref_gain=0.1)
            if len(self.cache) > 400:
                self.cache.clear()
            hit = self.cache[key] = (q, err)
            self.max_err = max(self.max_err, err)
        return hit[0]

    def now(self, leg: str) -> np.ndarray:
        st = self.stance
        if st is not None and st.pose.get(leg) is not None and st.w[leg] > 0:
            return np.asarray(st.pose[leg], float).copy()
        return self.stand(leg)

    def move(self, leg: str, target, dur: float) -> None:
        if target is None:
            q1, home = self.stand(leg), True
        else:
            q1, home = self.solve(leg, target), False
        self.seg[leg] = (self.now(leg), q1, self.sim.time, max(float(dur), 1e-3), home)

    def done(self, leg: str) -> bool:
        s = self.seg[leg]
        return s is None or self.sim.time - s[2] >= s[3]

    def clear(self) -> None:
        self.seg = {leg: None for leg in FRONT}
        self.grip = {leg: False for leg in FRONT}

    def step(self) -> None:
        st = self.stance
        if st is None:
            return
        for leg in FRONT:
            s = self.seg[leg]
            st.grip[leg] = self.grip[leg]
            if s is None:
                continue
            q0, q1, t0, dur, home = s
            u = min(max((self.sim.time - t0) / dur, 0.0), 1.0)
            st.pose[leg] = q0 + smoothstep(u) * (q1 - q0)
            st.w[leg] = 1.0
            if u >= 1.0 and home:
                st.pose[leg] = None
                st.w[leg] = 0.0
                self.seg[leg] = None


def leg_geoms(model: mj.MjModel, fly_name: str, leg: str) -> np.ndarray:
    """Geom ids of the fly's ``leg`` (every segment)."""
    pre = f"{fly_name}/{leg}_"
    ids = []
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) or ""
        if nm.startswith(pre):
            ids.append(g)
    return np.array(ids, dtype=int)


def contact_force(model: mj.MjModel, data: mj.MjData, geom: int, others: np.ndarray) -> float:
    """Total normal force (uN) of the contacts between ``geom`` and any of ``others``
    in the current step (0 if none)."""
    n = data.ncon
    if n == 0:
        return 0.0
    gg = data.contact.geom[:n]
    hit = np.flatnonzero(((gg[:, 0] == geom) & np.isin(gg[:, 1], others))
                         | ((gg[:, 1] == geom) & np.isin(gg[:, 0], others)))
    if len(hit) == 0:
        return 0.0
    f6 = np.zeros(6)
    tot = 0.0
    for i in hit:
        mj.mj_contactForce(model, data, int(i), f6)
        tot += abs(float(f6[0]))
    return tot


def cam_shot(job, shot: str, t: float, min_s: float = 1.6) -> str:
    """Camera cut hysteresis: keep the current shot at least ``min_s`` s."""
    cur, t0 = getattr(job, "_shot_state", (shot, -1e9))
    if shot != cur and t - t0 >= min_s:
        job._shot_state = (shot, t)
        return shot
    if shot == cur:
        return cur
    job._shot_state = (cur, t0)
    return cur


def caption_overlay(frame: np.ndarray, text: str, rgb=(255, 236, 200), shadow=(40, 20, 10),
                    big: bool = False) -> np.ndarray:
    import cv2

    H, W = frame.shape[:2]
    out = np.ascontiguousarray(frame).copy()
    fs = max(0.4, W / 1400.0) * (1.7 if big else 1.2)
    th = max(1, int(round(W / 700))) * 2 + 1
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, th)
    x, y = (W - tw) // 2, int(0.1 * H) + tht
    cv2.putText(out, text, (x + 2, y + 2), cv2.FONT_HERSHEY_DUPLEX, fs, shadow, th + 2, cv2.LINE_AA)
    cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, fs, rgb, th, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class DishwasherConfig(JobConfig):
    # --- layout (mm): the fly spawns at the origin facing +x (thorax ~(0.64, 0, 1.08))
    sink_x0: float = 1.5
    sink_x1: float = 3.45
    sink_y0: float = -1.05
    sink_y1: float = 1.1
    water_z: float = -0.07
    plate_r: float = 0.42
    plate_h: float = 0.05
    wash_x: float = 1.95  # the wash spot (plate centre, in the water)
    wash_y: float = -0.12
    wash_z: float = -0.012  # plate base
    sponge_below: float = 0.05  # the sponge's bottom this far under the tarsus (the collider offset)
    stack_x: float = 1.62  # the dirty stack on the conveyor end
    stack_y: float = 1.62
    stack_z: float = 0.05  # the belt top
    stack_n: int = 6
    stack_dz: float = 0.034  # nesting pitch
    rinse_x: float = 2.8
    rinse_y: float = 0.42
    rinse_z: float = 0.32
    faucet_z: float = 1.35  # the spout
    rack_x0: float = 3.85
    rack_dx: float = 0.15
    rack_y: float = -0.1
    rack_n: int = 8
    n_plates: int = 16  # pool (stack + rack + one in hand, plus spares)
    # --- the work --------------------------------------------------------------------
    scrub_r: float = 0.19  # stroke circle radius on the plate
    scrub_pts: int = 8  # keyframes per circle
    stroke_s: float = 0.075  # s per keyframe segment
    press: float = 0.018  # the tarsus target this far below the collider top (presses)
    scrub_per_mm: float = 0.22  # grime fraction removed per mm scrubbed in contact
    grime_min: float = 0.7  # initial grime of a plate (random in [min, 1])
    tough_min: float = 0.35  # scrub-off rate factor of a plate (random in [min, 1]: baked-on food)
    clean_below: float = 0.03  # grime left -> clean
    scrub_timeout_s: float = 4.5
    sponge_life_mm: float = 45.0  # mm of scrubbing per sponge
    rinse_s: float = 1.5
    # --- visuals ---------------------------------------------------------------------
    n_bubbles: int = 36
    n_drops: int = 24
    n_splash: int = 24
    particle_g: float = 900.0  # mm/s^2: slowed "cartoon" gravity of the water particles (visual)
    shadows: bool = True
    captions: bool = True
    close_ups: bool = True


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------


@register_job
class DishwasherJob(EternalJob):
    name = "dishwasher"
    znear = 0.05
    title = "DISHWASHER FLY"
    tagline = "the fly washes dishes forever"
    work_label = "plates washed"
    config_cls = DishwasherConfig
    required_names = (P + "pad", P + "sponge", P + "rack", P + "plate0")

    def __init__(self, cfg: DishwasherConfig | None = None) -> None:
        super().__init__(cfg)
        c = self.cfg
        self.rng = np.random.default_rng(c.seed + 41)
        self.n_washed = 0
        self.n_strokes = 0
        self.n_sponges = 1
        self.n_rack_loads = 0
        self.n_stacks = 1
        self.sponge_wear = 0.0
        self.grime_total = 0.0  # initial grime of the washed plates
        self.grime_removed = 0.0
        self.scrub_mm = 0.0
        self.contact_s = 0.0
        self.last_grime_removed = 0.0
        self.t = 0.0  # job clock (monotonic across resets)
        self.phase = "setup"
        self.carrier = "idle"
        self.cart = "home"
        self.belt = "home"
        self.message = ""
        self._caption = ("", -1e9)
        self.plates: list[dict] = []
        self.wash: int | None = None  # plate on the wash spot / in the left hand
        self.carry: int | None = None  # plate on the carrier (rinse -> rack)
        self.rack: list[int] = []
        self.stack: list[int] = []  # bottom .. top
        self._stance: DishStance | None = None
        self.max_force = 0.0

    # ------------------------------------------------------------ geometry
    def wash_top(self) -> float:
        """The plate's well (the face the sponge touches) at the wash spot."""
        c = self.cfg
        return c.wash_z + 0.18 * c.plate_h

    def pad_top(self) -> float:
        return self.wash_top() + self.cfg.sponge_below

    def stack_pos(self, k: int, belt_x: float = 0.0) -> np.ndarray:
        c = self.cfg
        return np.array([c.stack_x + belt_x, c.stack_y, c.stack_z + k * c.stack_dz])

    def slot_pos(self, k: int, cart_dx: float = 0.0) -> np.ndarray:
        c = self.cfg
        return np.array([c.rack_x0 + k * c.rack_dx + cart_dx, c.rack_y, 0.07 + c.plate_r])

    # ------------------------------------------------------------ build
    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        self._vis = vis
        self._materials(spec)
        self._add_kitchen(spec, vis)
        # --- the pad: the hidden collider under the plate being scrubbed (mocap, parked
        # away unless scrubbing); the fly touches it, nothing else
        pad = wb.add_body(name=P + "pad", mocap=True, pos=(0.0, 0.0, -30.0))
        pad.add_geom(name=P + "pad_col", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(c.plate_r * 0.85, 0.05, 0),
                     pos=(0, 0, -0.05), rgba=(0, 0, 0, 0), group=3, mass=0.0,
                     **contact_kwargs("fly", friction=0.6))
        # --- the plates (a fixed pool of mocap bodies)
        A.add_mesh(spec, P + "plate_mesh", A.dinner_plate_mesh(c.plate_r, c.plate_h))
        A.add_mesh(spec, P + "suds_mesh", A.disc_mesh(c.plate_r * 0.62, 0.006, 32))
        for i in range(c.n_plates):
            b = wb.add_body(name=P + f"plate{i}", mocap=True, pos=(0.0, 30.0 + i, -30.0))
            b.add_geom(name=P + f"plate{i}_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "plate_mesh",
                       material=P + "plate0_0", **vis)
            b.add_geom(name=P + f"plate{i}_suds", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "suds_mesh",
                       pos=(0, 0, 0.18 * c.plate_h + 0.006), rgba=(0.97, 0.98, 1.0, 0.0), **vis)
        # --- the sponge (mocap at the right front tarsus)
        sb = wb.add_body(name=P + "sponge", mocap=True, pos=(1.4, -0.9, 0.05))
        A.add_mesh(spec, P + "sponge_mesh", A.sponge_mesh(0.13, 0.085, 0.022))
        A.add_mesh(spec, P + "scourer_mesh", A.sponge_mesh(0.125, 0.08, 0.009))
        sb.add_geom(name=P + "sponge_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "sponge_mesh",
                    pos=(0, 0, -c.sponge_below + 0.024), material=P + "sponge0", **vis)
        sb.add_geom(name=P + "scourer_g", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "scourer_mesh",
                    pos=(0, 0, -c.sponge_below + 0.054), material=P + "scourer", **vis)
        # --- particles: bubbles, faucet drops, splashes (visual mocap pools)
        for kind, n, mat, r in (("bub", c.n_bubbles, "bubble", 0.03), ("drop", c.n_drops, "water_drop", 0.018),
                                ("spl", c.n_splash, "water_drop", 0.012)):
            for i in range(n):
                b = wb.add_body(name=P + f"{kind}{i}", mocap=True, pos=(0.0, -30.0 - i, -30.0))
                b.add_geom(name=P + f"{kind}{i}_g", type=mj.mjtGeom.mjGEOM_SPHERE, size=(r, 0, 0),
                           material=P + mat, **vis)
        # the faucet's water column (mocap; its length is set at run time)
        col = wb.add_body(name=P + "stream", mocap=True, pos=(c.rinse_x, c.rinse_y, -30.0))
        col.add_geom(name=P + "stream_g", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.03, 0.4, 0),
                     material=P + "stream", **vis)
        self._add_rack(spec, vis)
        self._add_belt(spec, vis)
        self._add_lights(spec)

    def _materials(self, spec) -> None:
        c = self.cfg
        T = A.add_textured_material
        A.add_texture(spec, P + "tex_counter", A.butcher_block_texture(c.seed))
        T(spec, P + "counter", P + "tex_counter", rgba=(1, 1, 1, 1), specular=0.25, shininess=0.4)
        A.add_texture(spec, P + "tex_tiles", A.subway_tile_texture(c.seed))
        T(spec, P + "tiles", P + "tex_tiles", rgba=(1, 1, 1, 1), specular=0.5, shininess=0.7)
        A.add_texture(spec, P + "tex_window", A.window_texture())
        T(spec, P + "window", P + "tex_window", rgba=(1, 1, 1, 1), emission=0.75, specular=0.3)
        A.add_texture(spec, P + "tex_gingham", A.gingham_texture())
        T(spec, P + "gingham", P + "tex_gingham", rgba=(1, 1, 1, 1), specular=0.05)
        A.add_texture(spec, P + "tex_steel", A.steel_texture(c.seed))
        T(spec, P + "steel", P + "tex_steel", rgba=(1, 1, 1, 1), specular=0.9, shininess=0.8, reflectance=0.05)
        A.add_texture(spec, P + "tex_water", A.water_texture())
        T(spec, P + "water", P + "tex_water", rgba=(1, 1, 1, 0.62), specular=0.9, shininess=0.9)
        A.add_texture(spec, P + "tex_foam", A.foam_texture())
        T(spec, P + "foam", P + "tex_foam", rgba=(1, 1, 1, 0.93), specular=0.4)
        for pat in range(A.N_SMEARS):
            for lv in range(A.GRIME_LEVELS):
                A.add_texture(spec, P + f"tex_plate{pat}_{lv}", A.plate_texture(pat, lv))
                T(spec, P + f"plate{pat}_{lv}", P + f"tex_plate{pat}_{lv}", rgba=(1, 1, 1, 1),
                  specular=0.6 if lv == A.GRIME_LEVELS - 1 else 0.35, shininess=0.8)
        A.add_texture(spec, P + "tex_sponge", A.sponge_texture())
        for k, tint in enumerate(((1, 1, 1), (0.9, 0.86, 0.78), (0.78, 0.72, 0.62), (0.66, 0.6, 0.5))):
            T(spec, P + f"sponge{k}", P + "tex_sponge", rgba=tint + (1,), specular=0.05)
        A.add_texture(spec, P + "tex_scourer", A.scourer_texture())
        T(spec, P + "scourer", P + "tex_scourer", rgba=(1, 1, 1, 1), specular=0.1)
        A.add_texture(spec, P + "tex_belt", A.belt_texture())
        T(spec, P + "belt", P + "tex_belt", rgba=(1, 1, 1, 1), specular=0.2)
        A.add_texture(spec, P + "tex_soap", A.label_texture(("SUDSY", "lemon dish soap")))
        T(spec, P + "soap_label", P + "tex_soap", rgba=(1, 1, 1, 1), specular=0.4)
        A.add_texture(spec, P + "tex_sign_in", A.sign_texture("DIRTY DISHES", "conveyor (kinematic)"))
        T(spec, P + "sign_in", P + "tex_sign_in", rgba=(1, 1, 1, 1), emission=0.2)
        A.add_texture(spec, P + "tex_sign_out", A.sign_texture("CLEAN", "rack cart (kinematic)",
                                                               bg=(0.85, 0.95, 0.98), fg=(0.1, 0.3, 0.45)))
        T(spec, P + "sign_out", P + "tex_sign_out", rgba=(1, 1, 1, 1), emission=0.2)
        spec.add_material(name=P + "bubble", rgba=(0.95, 0.97, 1.0, 0.55), specular=1.0, shininess=1.0)
        spec.add_material(name=P + "water_drop", rgba=(0.75, 0.88, 1.0, 0.7), specular=1.0, shininess=1.0)
        spec.add_material(name=P + "stream", rgba=(0.78, 0.9, 1.0, 0.45), specular=1.0, shininess=1.0)
        spec.add_material(name=P + "chrome", rgba=(0.82, 0.84, 0.88, 1), specular=1.0, shininess=0.95,
                          reflectance=0.1)
        spec.add_material(name=P + "rack_wire", rgba=(0.93, 0.94, 0.95, 1), specular=0.7, shininess=0.7)
        spec.add_material(name=P + "wall", rgba=(0.96, 0.72, 0.5, 1), specular=0.05)
        spec.add_material(name=P + "cabinet", rgba=(0.62, 0.42, 0.26, 1), specular=0.25)
        spec.add_material(name=P + "sill", rgba=(0.96, 0.95, 0.92, 1), specular=0.3)
        spec.add_material(name=P + "soap", rgba=(0.98, 0.9, 0.2, 0.85), specular=0.8, shininess=0.8)
        spec.add_material(name=P + "cap", rgba=(0.2, 0.55, 0.3, 1), specular=0.5)
        spec.add_material(name=P + "pot", rgba=(0.75, 0.38, 0.22, 1), specular=0.1)
        spec.add_material(name=P + "herb", rgba=(0.25, 0.55, 0.2, 1), specular=0.1)
        spec.add_material(name=P + "roller", rgba=(0.55, 0.56, 0.6, 1), specular=0.7)
        spec.add_material(name=P + "frame", rgba=(0.3, 0.32, 0.36, 1), specular=0.4)
        spec.add_material(name=P + "seg_on", rgba=(0.3, 1.0, 0.5, 1), emission=0.9)
        spec.add_material(name=P + "seg_off", rgba=(0.07, 0.1, 0.08, 1), emission=0.0)
        spec.add_material(name=P + "dig_bg", rgba=(0.03, 0.04, 0.035, 1), specular=0.3)
        spec.add_material(name=P + "cleat", rgba=(0.45, 0.45, 0.48, 1), specular=0.3)

    def _add_kitchen(self, spec, vis) -> None:
        c = self.cfg
        wb = spec.worldbody
        # --- the counter (world-scale uv, wraps) with the sink cut out
        pieces = (("ctr_back", -8.0, 10.0, c.sink_y1, 2.62), ("ctr_front", -8.0, 10.0, -3.2, c.sink_y0),
                  ("ctr_left", -8.0, c.sink_x0, c.sink_y0, c.sink_y1), ("ctr_right", c.sink_x1, 10.0, c.sink_y0, c.sink_y1))
        for nm, x0, x1, y0, y1 in pieces:
            md = A.slab_mesh(x1 - x0, y1 - y0, 0.45)
            md = A.MeshData(md.verts, md.faces, np.column_stack([(md.verts[:, 0] + (x0 + x1) / 2) / 5.0,
                                                                  (md.verts[:, 1] + (y0 + y1) / 2) / 2.5
                                                                  + 0.3 * md.verts[:, 2]]))
            A.add_mesh(spec, P + nm + "_mesh", md)
            wb.add_geom(name=P + nm, type=mj.mjtGeom.mjGEOM_MESH, meshname=P + nm + "_mesh",
                        pos=((x0 + x1) / 2, (y0 + y1) / 2, 0.0), material=P + "counter", **vis)
        # --- the sink: steel walls, bottom, a drain, soapy water, foam
        x0, x1, y0, y1 = c.sink_x0, c.sink_x1, c.sink_y0, c.sink_y1
        depth = 0.8
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for nm, size, pos in (("sink_bottom", ((x1 - x0) / 2, (y1 - y0) / 2, 0.02), (cx, cy, -depth)),
                              ("sink_w0", (0.02, (y1 - y0) / 2, depth / 2), (x0 + 0.02, cy, -depth / 2)),
                              ("sink_w1", (0.02, (y1 - y0) / 2, depth / 2), (x1 - 0.02, cy, -depth / 2)),
                              ("sink_w2", ((x1 - x0) / 2, 0.02, depth / 2), (cx, y0 + 0.02, -depth / 2)),
                              ("sink_w3", ((x1 - x0) / 2, 0.02, depth / 2), (cx, y1 - 0.02, -depth / 2))):
            wb.add_geom(name=P + nm, type=mj.mjtGeom.mjGEOM_BOX, size=size, pos=pos, material=P + "steel", **vis)
        A.add_mesh(spec, P + "rim_mesh", A.torus_mesh(1.0, 0.02, 4, 6))
        for nm, size, pos in (("rim0", (0.035, (y1 - y0) / 2 + 0.035, 0.012), (x0, cy, 0.005)),
                              ("rim1", (0.035, (y1 - y0) / 2 + 0.035, 0.012), (x1, cy, 0.005)),
                              ("rim2", ((x1 - x0) / 2 + 0.035, 0.035, 0.012), (cx, y0, 0.005)),
                              ("rim3", ((x1 - x0) / 2 + 0.035, 0.035, 0.012), (cx, y1, 0.005))):
            wb.add_geom(name=P + nm, type=mj.mjtGeom.mjGEOM_BOX, size=size, pos=pos, material=P + "chrome", **vis)
        wb.add_geom(name=P + "drain", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.16, 0.01, 0),
                    pos=(cx + 0.4, cy, -depth + 0.03), material=P + "chrome", **vis)
        wmd = A.slab_mesh(x1 - x0 - 0.08, y1 - y0 - 0.08, 0.02)
        A.add_mesh(spec, P + "water_mesh", wmd)
        wb.add_geom(name=P + "water", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "water_mesh",
                    pos=(cx, cy, c.water_z), material=P + "water", **vis)
        # foam islands on the water (flattened lumpy spheres), more round the wash spot
        rng = np.random.default_rng(c.seed + 5)
        self._foam = []
        for i in range(16):
            if i < 7:
                a = rng.uniform(0, 2 * math.pi)
                rr = c.plate_r + rng.uniform(0.02, 0.2)
                p = (c.wash_x + rr * math.cos(a), c.wash_y + rr * math.sin(a))
            else:
                p = (rng.uniform(x0 + 0.2, x1 - 0.2), rng.uniform(y0 + 0.2, y1 - 0.2))
            s = rng.uniform(0.08, 0.2)
            md = A.leaf_cluster_mesh(s, seed=i, n=5)
            md = A.MeshData(md.verts * np.array([1.0, 1.0, 0.35]), md.faces, md.uv)
            A.add_mesh(spec, P + f"foam{i}_mesh", md)
            wb.add_geom(name=P + f"foam{i}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + f"foam{i}_mesh",
                        pos=(p[0], p[1], c.water_z - 0.01), material=P + "foam", **vis)
        # --- the faucet (behind the sink) and its lever
        fb = (c.rinse_x, c.sink_y1 + 0.3)
        A.add_mesh(spec, P + "faucet_mesh", A.faucet_mesh(fb, (c.rinse_x, c.rinse_y, c.faucet_z), c.faucet_z + 0.6))
        wb.add_geom(name=P + "faucet", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "faucet_mesh",
                    material=P + "chrome", **vis)
        wb.add_geom(name=P + "faucet_base", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.13, 0.04, 0),
                    pos=(fb[0], fb[1], 0.04), material=P + "chrome", **vis)
        wb.add_geom(name=P + "aerator", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.085, 0.04, 0),
                    pos=(c.rinse_x, c.rinse_y, c.faucet_z + 0.02), material=P + "chrome", **vis)
        lev = wb.add_body(name=P + "lever", mocap=True, pos=(fb[0] + 0.3, fb[1] + 0.05, 0.08))
        lev.add_geom(name=P + "lever_g", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.03, 0.2, 0),
                     pos=(0, 0, 0.2), material=P + "chrome", **vis)
        # --- soap bottle, a dish towel, herbs on the sill
        A.add_mesh(spec, P + "bottle_mesh", A.bottle_mesh(0.22, 1.1))
        wb.add_geom(name=P + "bottle", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "bottle_mesh",
                    pos=(3.8, 1.6, 0.0), material=P + "soap", **vis)
        A.add_mesh(spec, P + "soap_label_mesh", A.panel_mesh(0.34, 0.3, 0.01))
        wb.add_geom(name=P + "soap_label", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "soap_label_mesh",
                    pos=(3.8, 1.37, 0.4), quat=A.panel_quat((0, -1, 0)), material=P + "soap_label", **vis)
        wb.add_geom(name=P + "soap_cap", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.07, 0.06, 0),
                    pos=(3.8, 1.6, 1.12), material=P + "cap", **vis)
        A.add_mesh(spec, P + "towel_mesh", A.cloth_mesh(0.9, 0.9, 3, 0.03))
        wb.add_geom(name=P + "towel", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "towel_mesh",
                    pos=(-0.8, -3.24, 0.0), material=P + "gingham", **vis)
        # --- the back wall: tiles, a window over the sink, a sill with herbs, curtains
        wy = 2.62
        wall_md = A.panel_mesh(18.0, 1.5, 0.05)
        A.add_mesh(spec, P + "tiles_mesh", A.MeshData(wall_md.verts, wall_md.faces, wall_md.uv * np.array([4.0, 1.0])))
        wb.add_geom(name=P + "tiles", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "tiles_mesh",
                    pos=(1.0, wy, 0.75), quat=A.panel_quat((0, -1, 0)), material=P + "tiles", **vis)
        wx0, wx1, wz0, wz1 = 0.8, 4.8, 1.75, 4.9
        for nm, size, pos in (("wall_l", ((wx0 + 8.0) / 2, 0.05, 2.6), ((wx0 - 8.0) / 2, wy + 0.05, 4.1)),
                              ("wall_r", ((10.0 - wx1) / 2, 0.05, 2.6), ((10.0 + wx1) / 2, wy + 0.05, 4.1)),
                              ("wall_t", ((wx1 - wx0) / 2, 0.05, 0.6), ((wx0 + wx1) / 2, wy + 0.05, wz1 + 0.6)),
                              ("wall_b", ((wx1 - wx0) / 2, 0.05, (wz0 - 1.5) / 2), ((wx0 + wx1) / 2, wy + 0.05, (wz0 + 1.5) / 2))):
            wb.add_geom(name=P + nm, type=mj.mjtGeom.mjGEOM_BOX, size=size, pos=pos, material=P + "wall", **vis)
        A.add_mesh(spec, P + "window_mesh", A.panel_mesh(wx1 - wx0, wz1 - wz0, 0.04))
        wb.add_geom(name=P + "window", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "window_mesh",
                    pos=((wx0 + wx1) / 2, wy + 0.06, (wz0 + wz1) / 2), quat=A.panel_quat((0, -1, 0)),
                    material=P + "window", **vis)
        wb.add_geom(name=P + "sill", type=mj.mjtGeom.mjGEOM_BOX, size=((wx1 - wx0) / 2 + 0.2, 0.25, 0.05),
                    pos=((wx0 + wx1) / 2, wy - 0.2, wz0 - 0.05), material=P + "sill", **vis)
        A.add_mesh(spec, P + "pot_mesh", A.pot_mesh(0.2, 0.35))
        for i, px in enumerate((1.5, 3.9)):
            wb.add_geom(name=P + f"pot{i}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "pot_mesh",
                        pos=(px, wy - 0.22, wz0), material=P + "pot", **vis)
            A.add_mesh(spec, P + f"herb{i}_mesh", A.leaf_cluster_mesh(0.28, seed=10 + i))
            wb.add_geom(name=P + f"herb{i}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + f"herb{i}_mesh",
                        pos=(px, wy - 0.22, wz0 + 0.35), material=P + "herb", **vis)
        A.add_mesh(spec, P + "curtain_mesh", A.cloth_mesh(1.1, 2.6, 5, 0.06))
        for i, px in enumerate((wx0 + 0.3, wx1 - 0.3)):
            wb.add_geom(name=P + f"curtain{i}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "curtain_mesh",
                        pos=(px, wy - 0.12, wz1 + 0.1), material=P + "gingham", **vis)
        wb.add_geom(name=P + "curtain_rod", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.03, (wx1 - wx0) / 2 + 0.5, 0),
                    pos=((wx0 + wx1) / 2, wy - 0.12, wz1 + 0.12), quat=quat_axis_angle((0, 1, 0), math.pi / 2),
                    material=P + "chrome", **vis)
        # upper cabinets either side of the window
        for i, (xa, xb) in enumerate(((-8.0, wx0 - 0.3), (wx1 + 0.3, 10.0))):
            wb.add_geom(name=P + f"cab{i}", type=mj.mjtGeom.mjGEOM_BOX, size=((xb - xa) / 2, 0.6, 1.3),
                        pos=((xa + xb) / 2, wy - 0.6, 3.6), material=P + "cabinet", **vis)
        # the WASHED counter (3 seven-segment digits) on the wall right of the window
        self._seg_names = []
        dw, dh, st = 0.3, 0.26, 0.04
        for k in range(3):
            segs = []
            x = 5.5 + k * 0.42
            for sname, (dx, dz, horiz) in SEGS.items():
                nm = f"{P}dig{k}_{sname}"
                size = (dw / 2 - 0.02, 0.01, st) if horiz else (st, 0.01, dh / 2 - 0.02)
                wb.add_geom(name=nm, type=mj.mjtGeom.mjGEOM_BOX, size=size, pos=(x + dx * dw, wy - 0.075, 1.0 + dz * dh),
                            material=P + "seg_off", **vis)
                segs.append(nm)
            self._seg_names.append(segs)
        wb.add_geom(name=P + "dig_bg", type=mj.mjtGeom.mjGEOM_BOX, size=(0.72, 0.01, 0.34),
                    pos=(5.92, wy - 0.055, 1.0), material=P + "dig_bg", **vis)
        # the floor plane's skybox: warm room tone
        sky = spec.texture("skybox")
        if sky is not None:
            sky.rgb1 = (0.35, 0.3, 0.25)
            sky.rgb2 = (0.1, 0.08, 0.06)

    def _add_rack(self, spec, vis) -> None:
        c = self.cfg
        wb = spec.worldbody
        L = c.rack_dx * (c.rack_n + 1)
        rk = wb.add_body(name=P + "rack", mocap=True, pos=(c.rack_x0 - c.rack_dx + L / 2, c.rack_y, 0.0))
        self._rack_home = np.array([c.rack_x0 - c.rack_dx + L / 2, c.rack_y, 0.0])
        cap = mj.mjtGeom.mjGEOM_CAPSULE
        ry = quat_axis_angle((0, 1, 0), math.pi / 2)
        rx = quat_axis_angle((1, 0, 0), math.pi / 2)
        W = 0.62
        for s in (-1, 1):  # base rails along x, cross bars
            rk.add_geom(name=P + f"rail{int(s > 0)}", type=cap, size=(0.018, L / 2, 0), pos=(0, s * W / 2, 0.04),
                        quat=ry, material=P + "rack_wire", **vis)
            rk.add_geom(name=P + f"toprail{int(s > 0)}", type=cap, size=(0.014, L / 2, 0),
                        pos=(0, s * W / 2, 0.42), quat=ry, material=P + "rack_wire", **vis)
        for k in range(c.rack_n + 2):
            x = -L / 2 + k * c.rack_dx - c.rack_dx / 2 + c.rack_dx / 2
            rk.add_geom(name=P + f"cross{k}", type=cap, size=(0.012, W / 2, 0), pos=(x, 0, 0.04), quat=rx,
                        material=P + "rack_wire", **vis)
            for s in (-1, 1):  # tines between the slots
                rk.add_geom(name=P + f"tine{k}_{int(s > 0)}", type=cap, size=(0.01, 0.2, 0),
                            pos=(x, s * 0.26, 0.24), material=P + "rack_wire", **vis)
        for s in (-1, 1):
            for e in (-1, 1):
                rk.add_geom(name=P + f"post{int(s > 0)}{int(e > 0)}", type=cap, size=(0.016, 0.2, 0),
                            pos=(e * L / 2, s * W / 2, 0.23), material=P + "rack_wire", **vis)
        A.add_mesh(spec, P + "sign_out_mesh", A.panel_mesh(0.9, 0.26, 0.02))
        rk.add_geom(name=P + "sign_out", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "sign_out_mesh",
                    pos=(0, -W / 2 - 0.03, 0.62), quat=A.panel_quat((0, -1, 0)), material=P + "sign_out", **vis)

    def _add_belt(self, spec, vis) -> None:
        """The dirty-dish conveyor along x at the back of the counter (visual)."""
        c = self.cfg
        wb = spec.worldbody
        x0, x1 = -7.0, c.stack_x + 0.55
        md = A.slab_mesh(x1 - x0, 0.95, 0.05)
        md = A.MeshData(md.verts, md.faces, np.column_stack([(md.verts[:, 0] - x0) / 1.2, 0.5 + md.verts[:, 1]]))
        A.add_mesh(spec, P + "belt_mesh", md)
        wb.add_geom(name=P + "belt", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "belt_mesh",
                    pos=((x0 + x1) / 2, c.stack_y, c.stack_z), material=P + "belt", **vis)
        for s in (-1, 1):
            wb.add_geom(name=P + f"belt_side{int(s > 0)}", type=mj.mjtGeom.mjGEOM_BOX,
                        size=((x1 - x0) / 2, 0.03, 0.05), pos=((x0 + x1) / 2, c.stack_y + s * 0.5, 0.05),
                        material=P + "frame", **vis)
        wb.add_geom(name=P + "belt_end", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.05, 0.5, 0),
                    pos=(x1, c.stack_y, 0.03), quat=quat_axis_angle((1, 0, 0), math.pi / 2),
                    material=P + "roller", **vis)
        self._cleat_x = np.linspace(x0 + 0.2, x1 - 0.2, 14)
        for i, x in enumerate(self._cleat_x):
            b = wb.add_body(name=P + f"cleat{i}", mocap=True, pos=(x, c.stack_y, c.stack_z + 0.004))
            b.add_geom(name=P + f"cleat{i}_g", type=mj.mjtGeom.mjGEOM_BOX, size=(0.03, 0.44, 0.006),
                       material=P + "cleat", **vis)
        A.add_mesh(spec, P + "sign_in_mesh", A.panel_mesh(1.1, 0.33, 0.02))
        wb.add_geom(name=P + "sign_in", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "sign_in_mesh",
                    pos=(-0.6, c.stack_y + 0.55, 0.45), quat=A.panel_quat((0, -1, 0)), material=P + "sign_in", **vis)
        for i, x in enumerate((-0.6 - 0.45, -0.6 + 0.45)):
            wb.add_geom(name=P + f"sign_in_post{i}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.02, 0.15, 0),
                        pos=(x, c.stack_y + 0.56, 0.15), material=P + "frame", **vis)

    def _add_lights(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        spec.visual.headlight.ambient = (0.3, 0.26, 0.22)
        spec.visual.headlight.diffuse = (0.3, 0.27, 0.24)
        spec.visual.headlight.specular = (0.1, 0.1, 0.1)
        key = np.array([0.5, -5.0, 8.0])
        tgt = np.array([2.6, 0.3, 0.0])
        wb.add_light(name=P + "key", type=spot_or_directional(c.shadows), pos=tuple(key), dir=tuple(tgt - key),
                     diffuse=(0.75, 0.62, 0.45), specular=(0.35, 0.3, 0.25), cutoff=38.0, exponent=0.6,
                     castshadow=bool(c.shadows))
        # daylight through the window (directional, cool)
        wb.add_light(name=P + "daylight", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(2.8, 6.0, 6.0),
                     dir=(0.1, -0.8, -0.6), diffuse=(0.28, 0.32, 0.38), specular=(0.1, 0.1, 0.12), castshadow=False)
        # a warm under-cabinet glow
        wb.add_light(name=P + "glow", type=mj.mjtLightType.mjLIGHT_POINT, pos=(-2.0, 1.8, 2.2),
                     diffuse=(0.35, 0.25, 0.12), specular=(0.05, 0.05, 0.05), attenuation=(0.6, 0.1, 0.02),
                     castshadow=False)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        m, s = self.sim.model, self.session
        c = self.cfg
        gp = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "ground_plane")
        if gp >= 0:  # the fly's floor stays (physics); the counter slabs draw it
            m.geom_group[gp] = 3
            m.geom_rgba[gp, 3] = 0.0
        self.acts = s.actions
        mid = lambda nm: int(m.body_mocapid[m.body(nm).id])  # noqa: E731
        self.pad_mocap = mid(P + "pad")
        self.pad_geom = m.geom(P + "pad_col").id
        self.sponge_mocap = mid(P + "sponge")
        self.sponge_geom = m.geom(P + "sponge_g").id
        self.rack_mocap = mid(P + "rack")
        self.lever_mocap = mid(P + "lever")
        self.stream_mocap = mid(P + "stream")
        self.stream_geom = m.geom(P + "stream_g").id
        self.cleat_mocap = [mid(P + f"cleat{i}") for i in range(len(self._cleat_x))]
        self.plate_mat = [[m.material(P + f"plate{p}_{lv}").id for lv in range(A.GRIME_LEVELS)]
                          for p in range(A.N_SMEARS)]
        self.sponge_mat = [m.material(P + f"sponge{k}").id for k in range(4)]
        self.seg_gid = [[m.geom(n).id for n in d] for d in self._seg_names]
        self.seg_mat = (m.material(P + "seg_off").id, m.material(P + "seg_on").id)
        self.plates = []
        for i in range(c.n_plates):
            self.plates.append(dict(mocap=mid(P + f"plate{i}"), geom=m.geom(P + f"plate{i}_g").id,
                                    suds=m.geom(P + f"plate{i}_suds").id, pat=0, grime=1.0, g0=1.0,
                                    where="parked", pos=np.array([0.0, 30.0 + i, -30.0]), quat=IDENT.copy(),
                                    suds_a=0.0))
        self.parts = {}
        for kind, n in (("bub", c.n_bubbles), ("drop", c.n_drops), ("spl", c.n_splash)):
            self.parts[kind] = dict(mocap=np.array([mid(P + f"{kind}{i}") for i in range(n)]),
                                    geom=np.array([m.geom(P + f"{kind}{i}_g").id for i in range(n)]),
                                    pos=np.zeros((n, 3)), vel=np.zeros((n, 3)), life=np.zeros(n),
                                    age=np.zeros(n), r0=m.geom_size[m.geom(P + f"{kind}0_g").id, 0], k=0)
        self.rf_geoms = leg_geoms(m, self.fly_name, "rf")
        self.tarsus5 = {leg: m.body(f"{self.fly_name}/{leg}_tarsus5").id for leg in FRONT}
        self.legs = LegDriver(self)
        st = getattr(s, "STATIONARY_ACTIONS", None)
        if st is not None and DishStance.name not in st:
            s.STATIONARY_ACTIONS = (*st, DishStance.name)
        self._new_stack(arrive=False)
        self._update_digits()
        self._begin()

    def _begin(self) -> None:
        self._stance = DishStance()
        self.acts.trigger(self._stance, source="job")
        self.legs.stance = self._stance
        self.legs.clear()
        self.steering.set(None, 0.0)
        self._go("settle")
        self._park_pad()
        self._prev_tip = None

    def _go(self, phase: str) -> None:
        self.phase = self.state = phase
        self._t_phase = self.t

    # ------------------------------------------------------------ plates
    def _new_stack(self, arrive: bool = True) -> None:
        c = self.cfg
        free = [i for i, p in enumerate(self.plates) if p["where"] == "parked"]
        self.stack = free[:c.stack_n]
        for k, i in enumerate(self.stack):
            p = self.plates[i]
            p["where"] = "stack"
            p["pat"] = int(self.rng.integers(0, A.N_SMEARS))
            p["g0"] = p["grime"] = float(self.rng.uniform(c.grime_min, 1.0))
            p["tough"] = float(self.rng.uniform(c.tough_min, 1.0))  # baked-on food scrubs off slower
            p["suds_a"] = 0.0
            p["quat"] = np.array(quat_axis_angle((0, 0, 1), self.rng.uniform(0, 2 * math.pi)))
        if arrive:
            self.belt = "arriving"
            self._t_belt = self.t
            self.n_stacks += 1
            self._caption_say("A NEW STACK OF DIRTY DISHES")
        self._belt_x = -8.5 if arrive else 0.0

    def _plate_level(self, p: dict) -> int:
        return int(np.clip(round((1.0 - p["grime"]) * (A.GRIME_LEVELS - 1)), 0, A.GRIME_LEVELS - 1))

    def _write_plates(self) -> None:
        m, d = self.sim.model, self.sim.data
        cart_dx = getattr(self, "_cart_dx", 0.0)
        for k, i in enumerate(self.stack):
            p = self.plates[i]
            p["pos"] = self.stack_pos(k, self._belt_x)
        for k, i in enumerate(self.rack):
            p = self.plates[i]
            p["pos"] = self.slot_pos(k, cart_dx)
            p["quat"] = np.array(quat_axis_angle((1, 0, 0), math.pi / 2))
        for p in self.plates:
            d.mocap_pos[p["mocap"]] = p["pos"] if p["where"] != "parked" else (0.0, 30.0, -30.0)
            d.mocap_quat[p["mocap"]] = p["quat"]
            m.geom_matid[p["geom"]] = self.plate_mat[p["pat"]][self._plate_level(p)]
            m.geom_rgba[p["suds"], 3] = p["suds_a"]

    # ------------------------------------------------------------ IK helpers
    def tarsus(self, leg: str) -> np.ndarray:
        return self.sim.data.xpos[self.tarsus5[leg]].copy()

    def _circle(self, k: int) -> np.ndarray:
        c = self.cfg
        a = 2 * math.pi * k / c.scrub_pts + self._scrub_phase
        r = c.scrub_r * (1.0 + 0.12 * math.sin(3 * a + self._scrub_phase))
        return np.array([c.wash_x + r * math.cos(a) - 0.03, c.wash_y + r * math.sin(a), self.pad_top() - c.press])

    def _park_pad(self) -> None:
        self.sim.data.mocap_pos[self.pad_mocap] = (0.0, 0.0, -30.0)

    def _place_pad(self) -> None:
        c = self.cfg
        self.sim.data.mocap_pos[self.pad_mocap] = (c.wash_x, c.wash_y, self.pad_top())

    # ------------------------------------------------------------ the loop
    def update(self) -> None:
        c = self.cfg
        dt = c.update_every_steps * self.sim.timestep
        self.t += dt
        t = self.t
        st = self._stance
        if st is None or self.acts.action is not st:
            if self.acts.action is None and not self.fly_down():
                self._stance = DishStance()
                self.acts.trigger(self._stance, source="job")
                self.legs.stance = self._stance
        self._fly_step(t, dt)
        self.legs.step()
        self._carrier_step(t)
        self._cart_step(t)
        self._belt_step(t)
        self._place_sponge()
        self._particles_step(dt)
        self._write_plates()
        if self.message and t - self._caption[1] > 2.2:
            self.message = ""

    def _fly_step(self, t: float, dt: float) -> None:
        c = self.cfg
        el = t - self._t_phase
        ph = self.phase
        L = self.legs
        hold_rf = np.array([1.55, -0.62, 0.32])
        if ph == "settle":
            if el > 0.45:
                L.move("rf", hold_rf, 0.4)
                self._go("idle")
        elif ph == "idle":
            if self.wash is not None:  # a plate already on the spot (after a reset)
                self._go("scrub_hover")
            elif self.stack and self.belt == "home" and L.done("rf"):
                self._fetch_i = self.stack[-1]
                grip = self._grip_point(len(self.stack) - 1)
                L.move("lf", grip + np.array([0, 0, 0.28]), 0.5)
                self._go("fetch_reach")
        elif ph == "fetch_reach":
            if L.done("lf"):
                L.move("lf", self._grip_point(len(self.stack) - 1), 0.3)
                self._go("fetch_grip")
        elif ph == "fetch_grip":
            if L.done("lf"):
                i = self.stack.pop()
                self.wash = i
                self.plates[i]["where"] = "hand"
                self._hand_off = self.plates[i]["pos"] - self.tarsus("lf")
                if not self.stack:
                    self._stack_empty_t = t
                lift = self.tarsus("lf") + np.array([0, 0, 0.4])
                L.move("lf", lift, 0.4)
                self._go("fetch_lift")
        elif ph in ("fetch_lift", "fetch_carry", "fetch_lower"):
            p = self.plates[self.wash]
            p["pos"] = self.tarsus("lf") + self._hand_off
            if L.done("lf"):
                dest = np.array([c.wash_x, c.wash_y, c.wash_z]) - self._hand_off
                if ph == "fetch_lift":
                    L.move("lf", dest + np.array([0, 0, 0.4]), 0.7)
                    self._go("fetch_carry")
                elif ph == "fetch_carry":
                    L.move("lf", dest, 0.35)
                    self._go("fetch_lower")
                else:
                    p["pos"] = np.array([c.wash_x, c.wash_y, c.wash_z])
                    p["where"] = "wash"
                    L.move("lf", self.tarsus("lf") + np.array([-0.05, 0.1, 0.3]), 0.3)
                    self._go("fetch_release")
        elif ph == "fetch_release":
            if L.done("lf"):
                L.move("lf", None, 0.4)
                self._go("scrub_hover")
        elif ph == "scrub_hover":
            if el > 0.1 and L.done("lf") and L.done("rf"):
                self._scrub_phase = float(self.rng.uniform(0, 2 * math.pi))
                self._place_pad()
                self._k = 0
                p0 = self._circle(0)
                L.move("rf", p0 + np.array([0, 0, 0.25]), 0.35)
                self._go("scrub_down")
        elif ph == "scrub_down":
            if L.done("rf"):
                L.move("rf", self._circle(0), 0.2)
                self._t_scrub = t
                self._go("scrub")
        elif ph == "scrub":
            self._scrub_contact(dt)
            p = self.plates[self.wash]
            if L.done("rf"):
                clean = p["grime"] <= c.clean_below
                if (clean or t - self._t_scrub > c.scrub_timeout_s) and self._k % c.scrub_pts == 0:
                    L.move("rf", self.tarsus("rf") + np.array([0, 0, 0.3]), 0.3)
                    self._go("scrub_lift")
                else:
                    self._k += 1
                    if self._k % c.scrub_pts == 0:
                        self.n_strokes += 1
                        self._scrub_phase += float(self.rng.uniform(-0.6, 0.6))
                    L.move("rf", self._circle(self._k % c.scrub_pts), c.stroke_s)
        elif ph == "scrub_lift":
            if L.done("rf"):
                self._park_pad()
                L.move("rf", hold_rf, 0.35)
                self._go("hand_over")
        elif ph == "hand_over":
            if self.carrier == "idle" and L.done("rf"):
                i = self.wash
                p = self.plates[i]
                removed = p["g0"] - max(p["grime"], 0.0)
                self.grime_total += p["g0"]
                self.grime_removed += removed
                self.last_grime_removed = removed / max(p["g0"], 1e-6)
                if p["grime"] <= c.clean_below:
                    p["grime"] = 0.0
                self.wash = None
                self.carry = i
                p["where"] = "carrier"
                self.carrier = "to_rinse"
                self._t_car = t
                self._car_from = (p["pos"].copy(), p["quat"].copy())
                self._go("idle")

    def _grip_point(self, k: int) -> np.ndarray:
        c = self.cfg
        pc = self.stack_pos(k, self._belt_x)
        return pc + np.array([-0.02, -(c.plate_r - 0.05), c.plate_h + 0.01])

    def _scrub_contact(self, dt: float) -> None:
        """The grime comes off only while the scrubbing tarsus touches the plate."""
        c = self.cfg
        m, d = self.sim.model, self.sim.data
        tip = self.tarsus("rf")
        prev = self._prev_tip if self._prev_tip is not None else tip
        self._prev_tip = tip
        f = contact_force(m, d, self.pad_geom, self.rf_geoms)
        self.max_force = max(self.max_force, f)
        if f <= 0.0:
            return
        dist = float(np.linalg.norm((tip - prev)[:2]))
        self.contact_s += dt
        self.scrub_mm += dist
        p = self.plates[self.wash]
        p["grime"] = max(0.0, p["grime"] - c.scrub_per_mm * p.get("tough", 1.0) * dist)
        p["suds_a"] = min(0.85, p["suds_a"] + 1.2 * dist)
        self.sponge_wear += dist / c.sponge_life_mm
        if self.sponge_wear >= 1.0:
            self.sponge_wear = 0.0
            self.n_sponges += 1
            self._caption_say(f"NEW SPONGE (#{self.n_sponges})")
        if dist > 0.002 and self.rng.random() < 0.6:
            self._spawn("bub", tip + np.array([0, 0, -c.sponge_below + 0.02]) + self.rng.normal(0, 0.05, 3),
                        np.array([*self.rng.normal(0, 0.12, 2), self.rng.uniform(0.05, 0.25)]),
                        self.rng.uniform(1.2, 2.6))

    # ------------------------------------------------------------ carrier / rack / belt
    def _carrier_step(self, t: float) -> None:
        c = self.cfg
        if self.carrier == "idle" or self.carry is None:
            return
        p = self.plates[self.carry]
        el = t - self._t_car
        rinse_q = np.array(quat_axis_angle((1, 0, 0), 0.45))
        rinse_p = np.array([c.rinse_x, c.rinse_y, c.rinse_z])
        if self.carrier == "to_rinse":
            u = smoothstep(el / 0.8)
            p0, q0 = self._car_from
            mid = 0.5 * (p0 + rinse_p) + np.array([0, 0, 0.25])
            p["pos"] = (1 - u) ** 2 * p0 + 2 * u * (1 - u) * mid + u ** 2 * rinse_p
            p["quat"] = _slerp(q0, rinse_q, u)
            if el >= 0.8:
                self.carrier = "rinse"
                self._t_car = t
        elif self.carrier == "rinse":
            p["pos"] = rinse_p
            p["quat"] = rinse_q
            p["suds_a"] = max(0.0, p["suds_a"] - 0.9 * c.update_every_steps * self.sim.timestep)
            if el >= c.rinse_s and self.cart == "home":
                self.carrier = "to_rack"
                self._t_car = t
                self._car_from = (p["pos"].copy(), p["quat"].copy())
        elif self.carrier == "to_rack":
            dur = 1.2
            u = smoothstep(el / dur)
            k = len(self.rack)
            dest = self.slot_pos(k)
            p0, q0 = self._car_from
            above = dest + np.array([0, 0, 0.7])
            mid = 0.5 * (p0 + above) + np.array([0, 0, 0.5])
            if u < 0.8:
                v = u / 0.8
                p["pos"] = (1 - v) ** 2 * p0 + 2 * v * (1 - v) * mid + v ** 2 * above
            else:
                v = (u - 0.8) / 0.2
                p["pos"] = above + v * (dest - above)
            p["quat"] = _slerp(q0, np.array(quat_axis_angle((1, 0, 0), math.pi / 2)), min(1.0, u / 0.8))
            if el >= dur:
                p["suds_a"] = 0.0
                p["where"] = "rack"
                self.rack.append(self.carry)
                self.carry = None
                self.carrier = "idle"
                self.n_washed += 1
                self.add_work(1)
                self._update_digits()
                self._caption_say(f"PLATE #{self.n_washed} CLEAN  ({100 * self.last_grime_removed:.0f}% grime off)")
                if len(self.rack) >= c.rack_n:
                    self.cart = "out"
                    self._t_cart = t
                    self.n_rack_loads += 1
                    self._caption_say(f"RACK FULL - CARTED OFF (load #{self.n_rack_loads})")

    def _cart_step(self, t: float) -> None:
        d = self.sim.data
        if self.cart == "home":
            self._cart_dx = 0.0
        else:
            el = t - self._t_cart
            if self.cart == "out":
                self._cart_dx = 9.0 * smoothstep(el / 2.0)
                if el >= 2.0:
                    for i in self.rack:
                        self.plates[i]["where"] = "parked"
                    self.rack = []
                    self.cart = "back"
                    self._t_cart = t
            elif self.cart == "back":
                self._cart_dx = 9.0 * (1.0 - smoothstep(el / 2.0))
                if el >= 2.0:
                    self.cart = "home"
                    self._cart_dx = 0.0
        d.mocap_pos[self.rack_mocap] = self._rack_home + np.array([self._cart_dx, 0, 0])

    def _belt_step(self, t: float) -> None:
        d = self.sim.data
        c = self.cfg
        if self.belt == "home" and not self.stack and t - getattr(self, "_stack_empty_t", t) > 0.3:
            if sum(p["where"] == "parked" for p in self.plates) >= c.stack_n:
                self._new_stack(arrive=True)
        if self.belt == "arriving":
            el = t - self._t_belt
            u = min(el / 2.6, 1.0)
            x_old = self._belt_x
            self._belt_x = -8.5 * (1.0 - smoothstep(u))
            self._belt_shift = getattr(self, "_belt_shift", 0.0) + (self._belt_x - x_old)
            if u >= 1.0:
                self.belt = "home"
                self._belt_x = 0.0
        shift = getattr(self, "_belt_shift", 0.0)
        x0, x1 = self._cleat_x[0], self._cleat_x[-1]
        span = x1 - x0 + (self._cleat_x[1] - self._cleat_x[0])
        for i, mi in enumerate(self.cleat_mocap):
            x = x0 + (self._cleat_x[i] - x0 + shift) % span
            d.mocap_pos[mi] = (x, c.stack_y, c.stack_z + 0.004)

    # ------------------------------------------------------------ sponge / particles
    def _place_sponge(self) -> None:
        d = self.sim.data
        m = self.sim.model
        tip = self.tarsus("rf")
        d.mocap_pos[self.sponge_mocap] = tip
        d.mocap_quat[self.sponge_mocap] = quat_axis_angle((0, 0, 1), -0.35)
        m.geom_matid[self.sponge_geom] = self.sponge_mat[min(3, int(self.sponge_wear * 4))]

    def _spawn(self, kind: str, pos, vel, life: float) -> None:
        P_ = self.parts[kind]
        k = P_["k"]
        P_["k"] = (k + 1) % len(P_["life"])
        P_["pos"][k] = pos
        P_["vel"][k] = vel
        P_["life"][k] = life
        P_["age"][k] = 0.0

    def _particles_step(self, dt: float) -> None:
        c = self.cfg
        m, d = self.sim.model, self.sim.data
        water_on = self.carrier == "rinse" or (self.carrier == "to_rinse" and self.t - self._t_car > 0.6)
        # the faucet: a water column + falling drops, splashing on the plate
        plate_top = c.rinse_z + 0.03
        top = c.faucet_z - 0.02
        if water_on:
            L = (top - plate_top) / 2
            m.geom_size[self.stream_geom, 1] = L
            m.geom_rbound[self.stream_geom] = L + 0.03
            d.mocap_pos[self.stream_mocap] = (c.rinse_x, c.rinse_y, plate_top + L)
            d.mocap_quat[self.stream_mocap] = IDENT
            if self.rng.random() < 0.5:
                self._spawn("drop", np.array([c.rinse_x, c.rinse_y, top]) + np.r_[self.rng.normal(0, 0.01, 2), 0],
                            np.array([0, 0, -self.rng.uniform(2.0, 3.0)]), 2.0)
        else:
            d.mocap_pos[self.stream_mocap] = (0, 0, -30.0)
        d.mocap_quat[self.lever_mocap] = quat_axis_angle((1, 0, 0), 0.5 if water_on else 0.0)
        g = np.array([0.0, 0.0, -c.particle_g])
        for kind, P_ in self.parts.items():
            alive = P_["life"] > 0
            if not alive.any():
                d.mocap_pos[P_["mocap"]] = (0.0, -30.0, -30.0)
                continue
            P_["age"][alive] += dt
            if kind == "bub":
                # rise to the water / plate surface and drift, then pop
                P_["vel"][alive] *= 0.97
                P_["pos"][alive] += P_["vel"][alive] * dt
                P_["pos"][alive, 2] = np.minimum(P_["pos"][alive, 2], c.water_z + 0.12)
                size = P_["r0"] * (0.5 + 0.7 * np.clip(P_["age"] / 0.3, 0, 1)) * \
                    np.clip((P_["life"] - P_["age"]) / 0.3, 0, 1)
            else:
                P_["vel"][alive] += g * dt
                P_["pos"][alive] += P_["vel"][alive] * dt
                size = np.full(len(P_["life"]), P_["r0"])
                if kind == "drop":
                    hit = alive & (P_["pos"][:, 2] < plate_top)
                    for k in np.flatnonzero(hit):
                        P_["life"][k] = 0.0
                        for _ in range(2):
                            a = self.rng.uniform(0, 2 * math.pi)
                            v = self.rng.uniform(0.6, 1.4)
                            self._spawn("spl", P_["pos"][k] + np.array([0, 0, 0.02]),
                                        np.array([v * math.cos(a), v * math.sin(a), self.rng.uniform(0.6, 1.3)]),
                                        0.6)
                else:
                    P_["life"][alive & (P_["pos"][:, 2] < c.water_z)] = 0.0
            dead = P_["age"] >= P_["life"]
            P_["life"][dead] = 0.0
            alive = P_["life"] > 0
            pos = np.where(alive[:, None], P_["pos"], np.array([0.0, -30.0, -30.0]))
            d.mocap_pos[P_["mocap"]] = pos
            m.geom_size[P_["geom"], 0] = np.maximum(size, 1e-4)

    # ------------------------------------------------------------ reset
    def on_reset(self) -> None:
        # an explicit reset (the fly fell): a plate in the left hand goes onto the wash
        # spot (scrubbing resumes); the carrier, the rack cart and the belt carry on
        c = self.cfg
        if self.wash is not None:
            p = self.plates[self.wash]
            p["pos"] = np.array([c.wash_x, c.wash_y, c.wash_z])
            p["quat"] = IDENT.copy()
            p["where"] = "wash"
        self._begin()

    def reset_props(self) -> None:
        self._write_plates()
        self._update_digits()
        self.sim.data.mocap_pos[self.rack_mocap] = self._rack_home + np.array([getattr(self, "_cart_dx", 0.0), 0, 0])

    def _update_digits(self) -> None:
        if not hasattr(self, "seg_gid"):
            return
        m = self.sim.model
        v = int(self.n_washed) % 1000
        for k in range(3):
            dg = (v // 10 ** (2 - k)) % 10
            lit = DIGITS[dg] if (v >= 10 ** (2 - k) or k == 2) else ""
            for sname, g in zip(SEGS, self.seg_gid[k]):
                m.geom_matid[g] = self.seg_mat[1 if sname in lit else 0]

    def _caption_say(self, text: str) -> None:
        self._caption = (text, self.t)
        self.message = text

    # ------------------------------------------------------------ presentation
    def post_process(self, frame: np.ndarray, t: float) -> np.ndarray:
        cap, t_cap = self._caption
        if not (self.cfg.captions and cap and 0 <= self.t - t_cap < 1.8):
            return frame
        return caption_overlay(frame, cap)

    def _shot(self) -> str:
        if not self.cfg.close_ups:
            return "wide"
        ph = self.phase
        if self.cart == "out" or self.belt == "arriving":
            want = "wide"  # the rack cart / the new stack
        elif ph.startswith("scrub"):
            want = "scrub"
        elif self.carrier in ("rinse",) or (self.carrier == "to_rinse"):
            want = "rinse"
        elif self.cart != "home" or self.belt != "home":
            want = "wide"
        else:
            want = "wide"
        return cam_shot(self, want, self.t)

    def camera_target(self) -> np.ndarray:
        c = self.cfg
        return {"wide": np.array([2.5, 0.6, 0.85]), "scrub": np.array([1.9, -0.1, 0.15]),
                "rinse": np.array([2.6, 0.4, 0.5])}[self._shot()]

    def camera_preset(self) -> CameraPreset:
        shot = self._shot()
        if shot == "scrub":
            return CameraPreset(azimuth=115.0, elevation=-40.0, distance=4.2, tau_s=0.3)
        if shot == "rinse":
            return CameraPreset(azimuth=105.0, elevation=-22.0, distance=5.0, tau_s=0.3)
        return CameraPreset(azimuth=98.0, elevation=-17.0, distance=7.2, tau_s=0.3)

    def grime_removed_pct(self) -> float:
        return 100.0 * self.grime_removed / self.grime_total if self.grime_total > 0 else 0.0

    def job_hud_lines(self) -> list[str]:
        g = self.plates[self.wash]["grime"] if self.wash is not None and self.plates else None
        return [
            f"grime removed {self.grime_removed_pct():.0f} %   strokes {self.n_strokes}   "
            f"scrubbed {self.scrub_mm:.1f} mm in contact" + (f"   this plate {100 * (g or 0):.0f} % grime" if g is not None else ""),
            f"sponge wear {100 * self.sponge_wear:.0f} %  (sponges {self.n_sponges})   rack {len(self.rack)}/{self.cfg.rack_n}"
            f"   rack loads {self.n_rack_loads}   stack {len(self.stack)}   stacks {self.n_stacks}",
            "(scrub: the sponge leg's real contact with the plate; fetch carry, plate carrier, rack cart, "
            "conveyor: kinematic; water / bubbles: visual particles)",
        ] + ([self.message] if self.message else [])

    def job_stats(self) -> dict[str, Any]:
        return {"plates_washed": self.n_washed, "grime_removed_pct": round(self.grime_removed_pct(), 1),
                "strokes": self.n_strokes, "scrub_mm": round(self.scrub_mm, 2),
                "contact_s": round(self.contact_s, 2), "sponge_wear_pct": round(100 * self.sponge_wear, 1),
                "sponges": self.n_sponges, "rack_loads": self.n_rack_loads, "stacks": self.n_stacks,
                "rack": len(self.rack), "stack": len(self.stack), "phase": self.phase,
                "max_force_uN": round(self.max_force, 2), "ik_err_mm": round(self.legs.max_err, 3)
                if hasattr(self, "legs") else 0.0}
