"""Mowing: the fly pushes a tiny push mower up and down a lawn in stripes, forever.
The grass grows back.

World (the fly spawns at the origin facing +x):

* a lawn (``lawn_x0..lawn_x1`` x ``lawn_y0..``) of ``n_rows`` rows along x, covered
  by a jittered grid of short grass blades: a fixed pool of thin *visual* box geoms
  (no contacts, so they can't trip the fly and moving / resizing them needs no BVH
  refit, docs/API_NOTES.md section 8);
* a push mower: a round red deck (the only colliding part) with an engine, four
  wheels and a handle (visual). It sits on planar joints (slide x, slide y, hinge z)
  at a fixed height, so it can't tip or climb, and the slide joints' damping plays
  the part of the wheels' rolling resistance. Only the fly's head / thorax / abdomen
  touch the deck, at low friction (``slippery_body_contact``, legs excluded). The
  job turns the hinge (rate limited) so the mower faces the way it is being pushed
  and the handle trails toward the fly; the deck is round, so turning it doesn't
  push anything.

Behaviour: a boustrophedon over the rows (row 0 along +x, row 1 back along -x, ...,
then the rows in reverse order). ``PushPilot`` (the sisyphus push loop, shared with
the raking job) walks round the mower to its back relative to the goal, then pushes
it with an over-steer that keeps the fly on the line through the mower and the
goal. The goal is a pure-pursuit point on the current row's centre line, so after a
row end the mower comes round in a U-turn onto the next row. Blades under the deck
get cut to ``cut_height`` and painted light (rows mown along +x) or dark (along -x):
the classic mowing stripes. Cut grass grows back to full height in ``regrow_s``
(its colour fades back to long grass), so the job never ends.

Counters: area mowed (m^2 of full-height grass cut; partly regrown grass counts
pro rata), rows mowed, lawns completed (a full pass over all rows).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import mujoco as mj
import numpy as np

from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import (
    add_box,
    contact_kwargs,
    quat_axis_angle,
    slippery_body_contact,
    wrap_angle,
)
from fly_simulator.jobs.registry import register_job

P = "mow/"

GRASS_LONG = np.array([0.24, 0.55, 0.16, 1.0])
STRIPE_LIGHT = np.array([0.58, 0.86, 0.38, 1.0])
STRIPE_DARK = np.array([0.12, 0.40, 0.10, 1.0])
PATCH_LONG = np.array([0.20, 0.45, 0.13, 1.0])
SOIL = (0.17, 0.33, 0.11, 1.0)
BLADE_HX, BLADE_HY = 0.06, 0.022
LAWN_EDGE = (0.62, 0.60, 0.56, 1.0)
DECK = (0.85, 0.12, 0.10, 1.0)
ENGINE = (0.20, 0.20, 0.22, 1.0)
TYRE = (0.08, 0.08, 0.08, 1.0)
STEEL = (0.78, 0.78, 0.80, 1.0)
GRIP = (0.10, 0.10, 0.10, 1.0)


# ---------------------------------------------------------------------------
# the push loop (also used by the raking job)
# ---------------------------------------------------------------------------


class PushPilot:
    """Closed-loop "push a round prop toward a goal point with the head" steering
    (the sisyphus loop, generalised).

    States: **approach** (walk round the prop, at ``orbit_clearance`` from its
    surface, to its back relative to the goal; from behind, pure pursuit onto the
    goal line) -> **align** (turn on the spot to face it) -> **push** (heading =
    phi + k (phi - psi), phi = fly->prop, psi = prop->goal, clipped) -> approach when
    the fly loses the prop. ``step(prop_xy, goal_xy)`` sets the job's steering and
    returns the state. ``clamp`` limits waypoints (keep the fly off walls / edges);
    ``unstick``: back away when wedged (``EternalJob.unstick``) while approaching.
    """

    def __init__(self, job: EternalJob, radius: float, *, behind_gap: float = 1.25,
                 orbit_clearance: float = 1.7, lookahead: float = 2.0,
                 align_radius: float = 1.0, approach_speed: float = 0.9,
                 push_speed: float = 0.7, oversteer: float = 0.5,
                 max_dev_deg: float = 25.0,
                 clamp: Callable[[np.ndarray], np.ndarray] | None = None,
                 unstick: bool = True) -> None:
        self.job = job
        self.R = float(radius)
        self.behind_gap = behind_gap
        self.orbit_clearance = orbit_clearance
        self.lookahead = lookahead
        self.align_radius = align_radius
        self.approach_speed = approach_speed
        self.push_speed = push_speed
        self.oversteer = oversteer
        self.max_dev = math.radians(max_dev_deg)
        self.clamp = clamp or (lambda p: np.asarray(p, float))
        self.use_unstick = unstick
        self.n_pushes = 0
        self.reset()

    def reset(self) -> None:
        self.state = "approach"
        self._orbit_dir = 0.0
        self._spin = 0.0

    def set_heading(self, heading: float, speed: float) -> None:
        """``steering.set`` that commits to one turning direction when the target is
        (nearly) straight behind: with |error| near 180 deg its sign flips from one
        update to the next and a fly turning on the spot freezes."""
        err = wrap_angle(self.job.sim.heading() - heading)
        lim = math.radians(150)
        if abs(err) > lim:
            if self._spin == 0.0:
                self._spin = 1.0 if err > 0 else -1.0
            heading = self.job.sim.heading() - self._spin * lim
        else:
            self._spin = 0.0
        self.job.steering.set(heading, speed)

    def aim_at(self, point, speed: float) -> None:
        p = self.job.fly_xy()
        self.set_heading(math.atan2(point[1] - p[1], point[0] - p[0]), speed)

    def _goto(self, s: str) -> None:
        self.state = s
        self._orbit_dir = 0.0

    def pushing(self, prop_xy, fly_xy=None) -> bool:
        """In the push state and the head actually near the prop."""
        if self.state != "push":
            return False
        p = self.job.fly_xy() if fly_xy is None else fly_xy
        return float(np.linalg.norm(p - prop_xy)) < self.R + self.behind_gap + 0.8

    def step(self, prop_xy, goal_xy) -> str:
        job = self.job
        R = self.R
        b = np.asarray(prop_xy, float)
        p = job.fly_xy()
        g = np.asarray(goal_xy, float) - b
        gl = float(np.linalg.norm(g))
        g = g / gl if gl > 1e-9 else np.array([1.0, 0.0])
        n = np.array([-g[1], g[0]])
        rel = p - b
        along, lateral = float(rel @ g), float(rel @ n)
        dist = float(np.linalg.norm(rel))
        behind_d = R + self.behind_gap
        if self.use_unstick and self.state in ("approach", "align"):
            job.unstick()
        if self.state == "approach":
            cone = -along - R + 0.5
            if along > -(R + 0.6) or abs(lateral) > cone:
                far = b - g * (behind_d + 2.5)
                wp = self.orbit_waypoint(p, b, far, R + self.orbit_clearance)
                self.aim_at(self.clamp(wp), 1.0)
            else:
                self._orbit_dir = 0.0
                s_aim = min(along + self.lookahead, -behind_d)
                self.aim_at(b + g * s_aim, self.approach_speed)
                if float(np.linalg.norm(p - (b - g * behind_d))) < self.align_radius:
                    self._goto("align")
        elif self.state == "align":
            phi = math.atan2(-rel[1], -rel[0])
            self.set_heading(phi, 0.5)
            err = abs(wrap_angle(job.sim.heading() - phi))
            if err < math.radians(25):
                self.n_pushes += 1
                self._goto("push")
            # (the lateral tolerance must exceed align_radius, or approach and align
            # hand the fly back and forth every update and it freezes)
            elif (along > -(R + 0.3) or abs(lateral) > max(R * 0.8, self.align_radius + 0.3)
                  or dist > behind_d + self.align_radius + 1.0):
                self._goto("approach")
        elif self.state == "push":
            phi = math.atan2(-rel[1], -rel[0])
            psi = math.atan2(g[1], g[0])
            dev = min(max(self.oversteer * wrap_angle(phi - psi), -self.max_dev), self.max_dev)
            self.set_heading(phi + dev, self.push_speed)
            if along > -R * 0.6 or abs(lateral) > R + 0.6 or dist > behind_d + 5.0:
                self._goto("approach")
        return self.state

    def orbit_waypoint(self, p: np.ndarray, C: np.ndarray, T: np.ndarray, r: float):
        """``T`` if the straight path to it clears the circle (C, r), else a point
        50 deg further round the circle (direction chosen once, with hysteresis; the
        side whose waypoint ``clamp`` moves less is preferred)."""
        d = T - p
        L = float(np.linalg.norm(d))
        dp = float(np.linalg.norm(p - C))
        if L < 1e-6:
            return T
        u = d / L
        s = min(max(float((C - p) @ u), 0.0), L)
        if dp >= r and float(np.linalg.norm(C - (p + u * s))) >= r:
            return T
        th_p = math.atan2(p[1] - C[1], p[0] - C[0])
        dth = wrap_angle(math.atan2(T[1] - C[1], T[0] - C[0]) - th_p)
        if self._orbit_dir == 0.0:
            mid = C + r * np.array([math.cos(th_p + dth / 2), math.sin(th_p + dth / 2)])
            if float(np.linalg.norm(self.clamp(mid) - mid)) > 0.5:
                dth -= math.copysign(2 * math.pi, dth)  # the other way round
            self._orbit_dir = 1.0 if dth >= 0 else -1.0
        elif dth * self._orbit_dir < 0:
            dth += self._orbit_dir * 2 * math.pi
        th = th_p + math.copysign(min(abs(dth), math.radians(50)), dth)
        rr = max(r + 0.3, dp) if dp > r else r + 0.6
        return np.array([C[0] + rr * math.cos(th), C[1] + rr * math.sin(th)])


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------


@dataclass
class MowingConfig(JobConfig):
    # ---- lawn ---------------------------------------------------------------------
    lawn_x0: float = 0.0
    lawn_x1: float = 20.0
    n_rows: int = 6
    row_width: float = 2.8  # row spacing (mm); the deck cuts 2 * deck_radius
    lawn_margin: float = 1.6  # lawn beyond the outer row centre lines (mm)
    blade_spacing: float = 0.42  # jittered grid (mm)
    patch_spacing: float = 0.8  # stripe tiles (mm)
    blade_height: float = 0.5  # full height (mm), +-blade_height_jitter
    blade_height_jitter: float = 0.15
    cut_height: float = 0.12
    regrow_s: float = 45.0  # cut -> full height (linear; ~1.5 lawn passes)
    grass_every_s: float = 0.02  # cutting / regrowth update period (sim s)
    # ---- mower ------------------------------------------------------------------
    deck_radius: float = 1.6
    deck_z: float = 0.85  # deck centre height; spans deck_z +- deck_half_height
    deck_half_height: float = 0.38
    mower_mass: float = 3e-4  # g (0.3 mg)
    # slide-joint damping = rolling resistance (uN per mm/s): pushing at 5 mm/s
    # takes 1.5 uN ~ 0.15 body weight
    mower_damping: float = 0.3
    head_friction: float = 0.05
    yaw_rate: float = 2.5  # visual turn of the mower toward the push direction (rad/s)
    # ---- behaviour ---------------------------------------------------------------
    behind_gap: float = 1.25
    orbit_clearance: float = 1.6
    push_speed: float = 0.75
    approach_speed: float = 0.9
    pursuit: float = 3.0  # goal = the row line this far ahead of the mower (mm)
    row_end_margin: float = 0.2  # row done when the mower centre is this close to its end
    row_tolerance: float = 1.0  # ... and within this of the row line (mm)
    celebrate_s: float = 2.0  # groom (wipe brow) after a lawn; 0 = off


@register_job
class MowingJob(EternalJob):
    name = "mowing"
    title = "LAWN MOWER FLY"
    tagline = "the grass is always growing"
    work_label = "lawn mowed"
    work_format = "{:.6f} m^2"
    config_cls = MowingConfig
    required_names = (P + "mower", P + "deck")

    cfg: MowingConfig

    # ------------------------------------------------------------ geometry
    def row_y(self, i: int) -> float:
        return i * self.cfg.row_width

    @property
    def lawn_y0(self) -> float:
        return -self.cfg.lawn_margin

    @property
    def lawn_y1(self) -> float:
        return self.row_y(self.cfg.n_rows - 1) + self.cfg.lawn_margin

    @property
    def x_start(self) -> float:
        return self.cfg.lawn_x0 + self.cfg.deck_radius

    @property
    def x_end(self) -> float:
        return self.cfg.lawn_x1 - self.cfg.deck_radius

    @property
    def mower_start(self) -> tuple[float, float]:
        # in front of the spawned fly (head at ~+1 mm), on row 0
        return (self.cfg.deck_radius + 1.9, 0.0)

    def blade_layout(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Seeded jittered grid: (xy (N, 2), full height (N,), yaw (N,))."""
        c = self.cfg
        rng = np.random.default_rng(1234)
        s = c.blade_spacing
        xs = np.arange(c.lawn_x0 + s / 2, c.lawn_x1, s)
        ys = np.arange(self.lawn_y0 + s / 2, self.lawn_y1, s)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        xy = np.column_stack([gx.ravel(), gy.ravel()])
        xy += rng.uniform(-0.35 * s, 0.35 * s, xy.shape)
        h = c.blade_height + rng.uniform(-c.blade_height_jitter, c.blade_height_jitter, len(xy))
        yaw = rng.uniform(0, math.pi, len(xy))
        return xy, h, yaw

    def patch_layout(self) -> np.ndarray:
        c = self.cfg
        s = c.patch_spacing
        xs = np.arange(c.lawn_x0 + s / 2, c.lawn_x1 - s / 2 + 1e-6, s)
        ys = np.arange(self.lawn_y0 + s / 2, self.lawn_y1 - s / 2 + 1e-6, s)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        return np.column_stack([gx.ravel(), gy.ravel()])

    def extension(self, world) -> None:
        c = self.cfg
        root = world.mjcf_root
        wb = root.worldbody
        mat = root.material("grid")
        if mat is not None:
            mat.rgba = (0.34, 0.52, 0.24, 1.0)  # the yard around the lawn
        lx, ly = (c.lawn_x0 + c.lawn_x1) / 2, (self.lawn_y0 + self.lawn_y1) / 2
        hx, hy = (c.lawn_x1 - c.lawn_x0) / 2, (self.lawn_y1 - self.lawn_y0) / 2
        # (top 50 um up: a box face (nearly) flush with the ground plane z-fights with
        # the checker at this camera distance)
        add_box(wb, P + "soil", (hx, hy, 0.025), (lx, ly, 0.025), rgba=SOIL, collide="visual")
        # a stone edging round the lawn (decoration)
        for k, (cx, cy, ex, ey) in enumerate((
                (lx, self.lawn_y0 - 0.25, hx + 0.5, 0.25),
                (lx, self.lawn_y1 + 0.25, hx + 0.5, 0.25),
                (c.lawn_x0 - 0.25, ly, 0.25, hy), (c.lawn_x1 + 0.25, ly, 0.25, hy))):
            add_box(wb, f"{P}edge{k}", (ex, ey, 0.04), (cx, cy, 0.04), rgba=LAWN_EDGE,
                    collide="visual")
        # stripe patches: a coarse grid of flat tiles on the soil that take the stripe
        # colour when mown (the stripes read from afar even where the blades are short)
        pxy = self.patch_layout()
        ps = c.patch_spacing / 2
        for i in range(len(pxy)):
            add_box(wb, f"{P}patch{i}", (ps, ps, 0.005), (pxy[i, 0], pxy[i, 1], 0.06),
                    rgba=tuple(PATCH_LONG), collide="visual")
        # the grass: a fixed pool of visual blades (thin vertical boxes)
        xy, h, yaw = self.blade_layout()
        for i in range(len(xy)):
            add_box(wb, f"{P}blade{i}", (BLADE_HX, BLADE_HY, h[i] / 2),
                    (xy[i, 0], xy[i, 1], h[i] / 2),
                    quat=quat_axis_angle((0, 0, 1), float(yaw[i])),
                    rgba=tuple(GRASS_LONG), collide="visual")
        # the mower: planar joints (x, y, yaw) at a fixed height
        x0, y0 = self.mower_start
        body = wb.add_body(name=P + "mower", pos=(x0, y0, 0.0))
        pad = 3.0
        body.add_joint(name=P + "slide_x", type=mj.mjtJoint.mjJNT_SLIDE, axis=(1, 0, 0),
                       damping=c.mower_damping, limited=True,
                       range=(c.lawn_x0 - pad - x0, c.lawn_x1 + pad - x0))
        body.add_joint(name=P + "slide_y", type=mj.mjtJoint.mjJNT_SLIDE, axis=(0, 1, 0),
                       damping=c.mower_damping, limited=True,
                       range=(self.lawn_y0 - pad - y0, self.lawn_y1 + pad - y0))
        body.add_joint(name=P + "yaw", type=mj.mjtJoint.mjJNT_HINGE, axis=(0, 0, 1),
                       damping=1.0, armature=1e-3)
        R = c.deck_radius
        kw = contact_kwargs("dynamic", 1.0)
        body.add_geom(name=P + "deck", type=mj.mjtGeom.mjGEOM_CYLINDER,
                      size=(R, c.deck_half_height, 0), pos=(0, 0, c.deck_z),
                      mass=c.mower_mass, rgba=DECK, **kw)
        vis = dict(contype=0, conaffinity=0, group=1, mass=0.0)
        top = c.deck_z + c.deck_half_height
        body.add_geom(name=P + "skirt", type=mj.mjtGeom.mjGEOM_CYLINDER,
                      size=(R * 1.04, 0.05, 0), pos=(0, 0, top - 0.05),
                      rgba=(0.65, 0.08, 0.07, 1.0), **vis)
        body.add_geom(name=P + "engine", type=mj.mjtGeom.mjGEOM_CYLINDER,
                      size=(0.75, 0.35, 0), pos=(0.1, 0, top + 0.35), rgba=ENGINE, **vis)
        body.add_geom(name=P + "engine_cap", type=mj.mjtGeom.mjGEOM_CYLINDER,
                      size=(0.3, 0.12, 0), pos=(0.1, 0, top + 0.82), rgba=STEEL, **vis)
        body.add_geom(name=P + "fuel_cap", type=mj.mjtGeom.mjGEOM_CYLINDER,
                      size=(0.15, 0.08, 0), pos=(0.7, 0.55, top + 0.08),
                      rgba=(0.95, 0.80, 0.10, 1.0), **vis)
        wq = quat_axis_angle((1, 0, 0), math.pi / 2)
        for k, (wx, wy) in enumerate(((1.0, 1.3), (1.0, -1.3), (-1.0, 1.3), (-1.0, -1.3))):
            body.add_geom(name=f"{P}wheel{k}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                          size=(0.45, 0.14, 0), pos=(wx, wy, 0.45), quat=wq, rgba=TYRE, **vis)
            body.add_geom(name=f"{P}hubcap{k}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                          size=(0.2, 0.15, 0), pos=(wx, wy, 0.45), quat=wq, rgba=STEEL, **vis)
        # handle: two arms from the rear of the deck back and up over the fly, and a
        # black grip bar just above its head
        grip_x, grip_z = -(R + c.behind_gap + 0.1), 2.35
        for side, sy in (("l", 0.8), ("r", -0.8)):
            a = np.array([-0.9, sy, top])
            b = np.array([grip_x, sy, grip_z])
            mid, d = (a + b) / 2, b - a
            L = float(np.linalg.norm(d))
            pitch = math.atan2(d[2], -d[0])
            body.add_geom(name=f"{P}handle_{side}", type=mj.mjtGeom.mjGEOM_CAPSULE,
                          size=(0.06, L / 2, 0), pos=tuple(mid),
                          quat=quat_axis_angle((0, 1, 0), -(math.pi / 2 - pitch)),
                          rgba=STEEL, **vis)
        body.add_geom(name=P + "grip", type=mj.mjtGeom.mjGEOM_CAPSULE,
                      size=(0.09, 0.9, 0), pos=(grip_x, 0, grip_z), quat=wq, rgba=GRIP, **vis)
        slippery_body_contact(root, P + "mower", P + "deck", self.fly_name,
                              friction=c.head_friction)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        m = self.sim.model
        c = self.cfg
        self.mower_body = m.body(P + "mower").id
        jx, jy, jz = (m.joint(P + n).id for n in ("slide_x", "slide_y", "yaw"))
        self.qx, self.qy, self.qyaw = (int(m.jnt_qposadr[j]) for j in (jx, jy, jz))
        self.vx, self.vy, self.vyaw = (int(m.jnt_dofadr[j]) for j in (jx, jy, jz))
        xy, h, _ = self.blade_layout()
        self.blade_xy = xy
        self.blade_full = h
        self.n_blades = len(xy)
        g0 = m.geom(f"{P}blade0").id
        self.blade_geoms = np.arange(g0, g0 + self.n_blades)
        assert m.geom(f"{P}blade{self.n_blades - 1}").id == self.blade_geoms[-1]
        self.patch_xy = self.patch_layout()
        p0 = m.geom(f"{P}patch0").id
        self.patch_geoms = np.arange(p0, p0 + len(self.patch_xy))
        self.patch_grown = np.ones(len(self.patch_xy))
        self.patch_stripe = np.zeros(len(self.patch_xy))
        self.blade_h = h.copy()  # current heights
        self.blade_stripe = np.zeros(self.n_blades)  # +1 light / -1 dark / 0 never cut
        self.cell_area_m2 = c.blade_spacing ** 2 * 1e-6
        self.area_m2 = 0.0
        self.rows_mowed = 0
        self.lawns = 0
        self.blades_cut = 0
        self.row = 0  # index into the serpentine sequence
        self._pass_dir = 1  # +1: rows 0..n-1, -1: back down
        self._t_grass = -1.0
        self._last_mower = np.array(self.mower_start, float)
        self._yaw = 0.0
        clamp_pad = 2.5

        def clamp(p):
            return np.array([min(max(float(p[0]), c.lawn_x0 - clamp_pad), c.lawn_x1 + clamp_pad),
                             min(max(float(p[1]), self.lawn_y0 - clamp_pad),
                                 self.lawn_y1 + clamp_pad)])

        self.pilot = PushPilot(self, c.deck_radius, behind_gap=c.behind_gap,
                               orbit_clearance=c.orbit_clearance, push_speed=c.push_speed,
                               approach_speed=c.approach_speed, clamp=clamp)
        self._write_grass(np.ones(self.n_blades, bool))
        self.state = "approach"

    def on_reset(self) -> None:
        self.pilot.reset()
        self.state = "approach"
        self._t_grass = -1.0

    def reset_props(self) -> None:
        """The keyframe reset put the mower back at its spawn pose. Put it back where
        it was (the lawn is half mown), unless that is where the fly respawns."""
        last = self._last_mower
        if np.all(np.isfinite(last)) and np.linalg.norm(last) > self.cfg.deck_radius + 3.5:
            x0, y0 = self.mower_start
            d = self.sim.data
            d.qpos[self.qx], d.qpos[self.qy] = last[0] - x0, last[1] - y0
            d.qvel[[self.vx, self.vy, self.vyaw]] = 0.0
            mj.mj_forward(self.sim.model, d)

    # ------------------------------------------------------------ state
    def mower_xy(self) -> np.ndarray:
        return self.sim.data.xpos[self.mower_body, :2].copy()

    def mower_speed(self) -> float:
        v = self.sim.data.qvel
        return float(math.hypot(v[self.vx], v[self.vy]))

    def row_index(self, k: int | None = None) -> int:
        """Lawn row of step ``k`` of the serpentine (0..n-1 then n-1..0)."""
        n = self.cfg.n_rows
        k = self.row if k is None else k
        k %= 2 * n
        return k if k < n else 2 * n - 1 - k

    def row_dir(self, k: int | None = None) -> int:
        """+1 if step ``k`` mows along +x, -1 along -x."""
        k = self.row if k is None else k
        return 1 if k % 2 == 0 else -1

    def row_goal(self, mower: np.ndarray) -> tuple[np.ndarray, float]:
        """Pure-pursuit goal on the current row line and the row's end x."""
        c = self.cfg
        y = self.row_y(self.row_index())
        d = self.row_dir()
        x_end = self.x_end if d > 0 else self.x_start
        gx = mower[0] + d * c.pursuit
        gx = min(gx, x_end + 0.6) if d > 0 else max(gx, x_end - 0.6)
        return np.array([gx, y]), x_end

    def grass_fraction_cut(self) -> float:
        """Fraction of the lawn currently below half height."""
        c = self.cfg
        half = c.cut_height + 0.5 * (self.blade_full - c.cut_height)
        return float(np.mean(self.blade_h < half))

    # ------------------------------------------------------------ grass
    def _write_grass(self, sel: np.ndarray) -> None:
        m = self.sim.model
        c = self.cfg
        g = self.blade_geoms[sel]
        h = self.blade_h[sel]
        m.geom_size[g, 2] = h / 2
        m.geom_pos[g, 2] = h / 2
        m.geom_aabb[g, 5] = h / 2
        m.geom_rbound[g] = np.sqrt(BLADE_HX ** 2 + BLADE_HY ** 2 + (h / 2) ** 2)
        # colour: stripe colour just after the cut, fading back to long grass
        grown = np.clip((h - c.cut_height) / np.maximum(self.blade_full[sel] - c.cut_height,
                                                         1e-6), 0, 1)
        st = self.blade_stripe[sel]
        base = np.where(st[:, None] > 0, STRIPE_LIGHT, STRIPE_DARK)
        base = np.where(st[:, None] == 0, GRASS_LONG, base)
        w = (grown ** 1.5)[:, None]
        m.geom_rgba[g] = (1 - w) * base + w * GRASS_LONG

    def update_grass(self, mower: np.ndarray, dt: float) -> None:
        c = self.cfg
        h = self.blade_h
        # regrowth
        growing = h < self.blade_full
        if dt > 0 and np.any(growing):
            rate = (self.blade_full - c.cut_height) / c.regrow_s
            h[growing] = np.minimum(h[growing] + rate[growing] * dt, self.blade_full[growing])
        # cutting (blades under the deck)
        d2 = np.sum((self.blade_xy - mower) ** 2, axis=1)
        under = (d2 < c.deck_radius ** 2) & (h > c.cut_height + 1e-6)
        if np.any(under):
            frac = (h[under] - c.cut_height) / (self.blade_full[under] - c.cut_height)
            area = float(np.sum(np.clip(frac, 0, 1))) * self.cell_area_m2
            self.area_m2 += area
            self.add_work(area)
            self.work = self.area_m2
            self.blades_cut += int(np.count_nonzero(frac > 0.5))
            h[under] = c.cut_height
            self.blade_stripe[under] = self.row_dir()
        self._write_grass(growing | under)
        # stripe tiles
        pg = self.patch_grown
        pgrow = pg < 1.0
        if dt > 0:
            pg[pgrow] = np.minimum(pg[pgrow] + dt / c.regrow_s, 1.0)
        pd2 = np.sum((self.patch_xy - mower) ** 2, axis=1)
        pcut = pd2 < (c.deck_radius - 0.2) ** 2
        pg[pcut] = 0.0
        self.patch_stripe[pcut] = self.row_dir()
        self._write_patches(pgrow | pcut)

    def _write_patches(self, sel: np.ndarray) -> None:
        st = self.patch_stripe[sel]
        base = np.where(st[:, None] > 0, STRIPE_LIGHT, STRIPE_DARK)
        base = np.where(st[:, None] == 0, PATCH_LONG, base)
        w = (self.patch_grown[sel] ** 1.5)[:, None]
        self.sim.model.geom_rgba[self.patch_geoms[sel]] = (1 - w) * base + w * PATCH_LONG

    def regrow_all(self) -> None:
        """Instantly regrow the whole lawn (tests / a fresh start)."""
        self.blade_h[:] = self.blade_full
        self.blade_stripe[:] = 0
        self._write_grass(np.ones(self.n_blades, bool))
        self.patch_grown[:] = 1.0
        self.patch_stripe[:] = 0
        self._write_patches(np.ones(len(self.patch_xy), bool))

    # ------------------------------------------------------------ behaviour
    def update(self) -> None:
        c = self.cfg
        sim = self.sim
        d = sim.data
        mower = self.mower_xy()
        if not np.all(np.isfinite(mower)):
            return
        self._last_mower = mower
        t = sim.time
        if self._t_grass < 0 or t < self._t_grass:
            self._t_grass = t
        if t - self._t_grass >= c.grass_every_s:
            self.update_grass(mower, t - self._t_grass)
            self._t_grass = t
        if self.session.actions.busy:  # celebrating / backing away
            self.steering.set(None, 1.0)
            return
        goal, x_end = self.row_goal(mower)
        self.state = self.pilot.step(mower, goal)
        # visual yaw of the mower toward the push direction (rate limited)
        psi = math.atan2(goal[1] - mower[1], goal[0] - mower[0])
        dy = wrap_angle(psi - self._yaw)
        step = c.yaw_rate * c.update_every_steps * sim.model.opt.timestep
        self._yaw = wrap_angle(self._yaw + min(max(dy, -step), step))
        d.qpos[self.qyaw] = self._yaw
        d.qvel[self.vyaw] = 0.0
        # row finished?
        y = self.row_y(self.row_index())
        done = (mower[0] >= x_end - c.row_end_margin if self.row_dir() > 0
                else mower[0] <= x_end + c.row_end_margin)
        if done and abs(mower[1] - y) < c.row_tolerance:
            self.rows_mowed += 1
            self.row += 1
            if self.row % c.n_rows == 0:
                self.lawns += 1
                self.say(f"LAWN #{self.lawns} done ({self.rows_mowed} rows, "
                         f"{self.area_m2 * 1e6:.0f} mm^2 in total); it's growing back already")
                if c.celebrate_s > 0:
                    from fly_simulator.actions import make_action

                    self.session.actions.trigger(make_action("groom", duration=c.celebrate_s),
                                                 source="job")
            self.pilot.reset()

    # ------------------------------------------------------------ view / HUD
    def camera_target(self) -> np.ndarray:
        f = self.sim.thorax_position()
        m = np.append(self.mower_xy(), 0.8)
        lawn = np.array([(self.cfg.lawn_x0 + self.cfg.lawn_x1) / 2,
                         (self.lawn_y0 + self.lawn_y1) / 2, 0.0])
        out = 0.3 * f + 0.3 * m + 0.4 * lawn
        return out if np.all(np.isfinite(out)) else f

    def camera_preset(self) -> CameraPreset:
        # from the lawn's -y side, high enough to see the stripes
        return CameraPreset(azimuth=70.0, elevation=-45.0, distance=26.0, tau_s=1.0)

    def job_hud_lines(self) -> list[str]:
        return [f"rows mowed {self.rows_mowed}   lawns completed {self.lawns}   "
                f"row {self.row_index() + 1}/{self.cfg.n_rows} "
                f"{'->' if self.row_dir() > 0 else '<-'}   "
                f"lawn short {100 * self.grass_fraction_cut():.0f}%"]

    def job_stats(self) -> dict:
        return {"rows_mowed": self.rows_mowed, "lawns_completed": self.lawns,
                "area_m2": self.area_m2, "area_mm2": self.area_m2 * 1e6,
                "blades_cut": self.blades_cut, "lawn_short_frac": self.grass_fraction_cut(),
                "pushes_started": self.pilot.n_pushes, "unstuck": self.n_unstuck}
