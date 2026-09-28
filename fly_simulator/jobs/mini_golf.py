"""Mini golf: the fly plays mini golf forever. Four holes, round after round.

World (the fly spawns at the origin facing +x, behind the tee of hole 1):

* four parallel lanes along +x (``lane_x0 .. lane_x1``, half width ``lane_hw``), one
  every ``lane_pitch`` mm in y: 1 WINDMILL (par 3), 2 THE RAMP (par 2), 3 THE TUNNEL
  (par 3), 4 BUMPERS (par 2). Each lane is bright green felt with coloured rails, a
  tee mat, a cup with a flag, and an opening in the back rail where the fly walks in.
* The **ball** is a free sphere (R 0.6 mm, 0.05 mg) with real rolling physics:
  condim-6 rolling friction on the felt, bouncy rails and obstacles, gravity on the
  ramp, and a real hole: the felt is made of ball-only boxes that leave an octagonal
  cup open, and a ball that runs over it slowly enough drops in.
* **The fly walks on FlyGym's own (hidden) ground plane**, which is flush with the felt
  top. The ball never touches that plane (its contact bits), so the cup can be a hole
  for the ball while the fly's feet simply stand on the plane (over the cup too:
  engineered, labelled). The rails and obstacles are ball-only colliders (the fly's
  path keeps clear of them; its legs can overlap a rail visually); the ramp is a real
  static slope that both the fly and the ball go over.
* Obstacles: a windmill whose four sails turn (kinematic mocap, ball-only colliders)
  in front of the door through the mill, a hump-shaped ramp, a striped pipe tunnel,
  and three round rubber bumper posts.

Behaviour per stroke: the **aim model** (``plan_shot``: a documented skill model, not
the fly's brain) picks a target (the cup if the line is clear, else a gate such as the
windmill door / the tunnel, else a lay-up point) and a speed from the felt's rolling
deceleration (+ the energy to climb the ramp). ``PushPilot`` (the mowing push loop)
walks the fly round the ball to its back relative to the aim line, turns it to face
the ball and walks it into the ball. At the first real contact of the fly's head /
thorax with the ball the job sets the ball's velocity to the planned putt plus the
skill model's errors (**PUTT impulse, engineered, labelled**: the fly's walking push
alone rolls the ball only a few mm and can't be sized). At the windmill the fly
waits at the ball until the aim model says the sails will be clear when the ball gets
there. The fly stops and watches; the ball rolls with real physics. Then: in the cup
(counted), or the next stroke from where it stopped. After the hole the fly walks to
the cup, the ball is lifted out onto its back (**kinematic carry**, labelled), the fly
walks to the next tee and the ball is set down in front of it (kinematic).

House rules (engineered, labelled, counted): a ball resting against a rail / an
obstacle or inside the mill / the tunnel is moved to a playable spot (no penalty,
like real mini golf's "one putter-head from the wall"); if the fly has no room
behind the ball it is moved along its line until it has; a ball that leaves the
course is out of bounds (+1 stroke, replayed from where it was hit); a hole ends
after ``max_strokes`` (6) strokes (picked up). Scorecard: strokes vs par per hole,
round totals, holes-in-one, birdies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from fly_simulator.jobs import mini_golf_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import (
    add_box,
    add_slope,
    contact_kwargs,
    quat_axis_angle,
    slippery_body_contact,
    wrap_angle,
)
from fly_simulator.jobs.mowing import PushPilot
from fly_simulator.jobs.registry import register_job
from fly_simulator.terrain import FLY_BIT, TERRAIN_BIT

P = "golf/"
BALL_BIT = 64  # ball-only colliders (felt, rails, obstacles, cup)
G = 9810.0  # mm/s^2

HOLES = (
    # name, par, rail colour
    ("WINDMILL", 3, (0.95, 0.20, 0.25)),
    ("THE RAMP", 2, (0.15, 0.45, 0.95)),
    ("THE TUNNEL", 3, (1.00, 0.62, 0.05)),
    ("BUMPERS", 2, (0.75, 0.25, 0.90)),
)


@dataclass
class MiniGolfConfig(JobConfig):
    # ---- course --------------------------------------------------------------------
    lane_x0: float = -3.0
    lane_x1: float = 23.0
    lane_hw: float = 5.5  # lane half width (mm)
    lane_pitch: float = 17.0  # lanes every this many mm in y
    entry_hw: float = 1.9  # half width of the opening in the back rail
    tee_x: float = 2.6
    cup_x: float = 20.0
    cup_r: float = 0.9  # cup inradius (octagon); the ball is R 0.6
    cup_depth: float = 1.4
    rail_h: float = 0.75
    # ---- ball -----------------------------------------------------------------------
    ball_radius: float = 0.6
    ball_mass: float = 5e-5  # g (0.05 mg)
    # felt rolling friction (condim-6, mm): deceleration (5/7) g mu / R
    felt_rolling: float = 0.012
    #: measured deceleration of a rolling ball on the felt (mm/s^2), used by the aim model
    felt_decel: float = 135.0
    rail_tc: float = 0.003  # rail contact time constant (s)
    rail_dampratio: float = 0.08  # bouncy rails (restitution ~0.6 at 40 mm/s, measured)
    head_friction: float = 0.05
    lie_damping: float = 1e-3  # free-joint damping of a ball waiting on its lie
    # ---- obstacles --------------------------------------------------------------------
    mill_x: float = 10.0
    mill_rpm: float = 15.0  # windmill sails (0.25 rev/s)
    ramp_x: float = 8.0  # ramp foot
    ramp_h: float = 0.3  # hump height (mm)
    ramp_deg: float = 7.0  # slopes
    tunnel_x0: float = 9.0
    tunnel_x1: float = 14.0
    # ---- aim / skill model (documented, not the fly's) --------------------------------
    skill: float = 0.5
    aim_sd_deg: float = 1.0  # + aim_sd_unskilled_deg * (1 - skill)
    aim_sd_unskilled_deg: float = 6.0
    speed_sd: float = 0.05  # relative, + speed_sd_unskilled * (1 - skill)
    speed_sd_unskilled: float = 0.25
    overshoot: float = 0.8  # plan the ball to stop this far past the cup centre (mm)
    max_speed: float = 260.0
    max_strokes: int = 6
    start_hole: int = 0  # 0-based (tests / demos)
    # ---- behaviour -------------------------------------------------------------------
    behind_gap: float = 1.2
    orbit_clearance: float = 1.2
    approach_speed: float = 0.8
    push_speed: float = 0.45
    follow_s: float = 0.6
    roll_timeout_s: float = 12.0
    stop_speed: float = 0.6  # ball below this (mm/s) for stop_hold_s = at rest
    stop_hold_s: float = 0.3
    # ---- presentation ------------------------------------------------------------------
    slowmo: float = 0.35  # while the ball rolls fast (edit effect, labelled); 1 = off
    captions: bool = True
    shadows: bool = True
    stuck_timeout_s: float = 120.0


class GolfPilot(PushPilot):
    """PushPilot whose walking waypoints go round the course's solid obstacles."""

    def aim_at(self, point, speed: float) -> None:
        super().aim_at(self.job.route(self.job.fly_xy(), np.asarray(point, float)), speed)


@register_job
class MiniGolfJob(EternalJob):
    name = "mini_golf"
    title = "MINI GOLF FLY"
    tagline = "the fly plays mini golf forever"
    work_label = "holes played"
    znear = 0.05
    config_cls = MiniGolfConfig
    required_names = (P + "ball", P + "ball_geom", P + "sails")

    cfg: MiniGolfConfig

    # ------------------------------------------------------------ geometry
    def hole_y(self, k: int) -> float:
        return k * self.cfg.lane_pitch

    def cup_xy(self, k: int) -> np.ndarray:
        return np.array([self.cfg.cup_x, self.hole_y(k)])

    def tee_xy(self, k: int) -> np.ndarray:
        return np.array([self.cfg.tee_x, self.hole_y(k)])

    def par(self, k: int) -> int:
        return HOLES[k][1]

    def lane_of(self, x: float, y: float, pad: float = 0.4) -> int:
        """Index of the lane containing (x, y) (-1 outside all lanes)."""
        c = self.cfg
        k = int(round(y / c.lane_pitch))
        if 0 <= k < len(HOLES) and abs(y - self.hole_y(k)) <= c.lane_hw + pad \
                and c.lane_x0 - pad <= x <= c.lane_x1 + pad:
            return k
        return -1

    def ramp_profile(self) -> tuple[float, float, float, float]:
        c = self.cfg
        run = c.ramp_h / math.tan(math.radians(c.ramp_deg))
        a = c.ramp_x
        return a, a + run, a + run + 1.0, a + 2 * run + 1.0

    def ground_height(self, x: float, y: float) -> float:
        if self.lane_of(x, y, 0.0) != 1:
            return 0.0
        a, b, cc, d = self.ramp_profile()
        h = self.cfg.ramp_h
        if a <= x < b:
            return h * (x - a) / (b - a)
        if b <= x <= cc:
            return h
        if cc < x <= d:
            return h * (d - x) / (d - cc)
        return 0.0

    def obstacles(self, k: int) -> list[tuple[float, float, float, float]]:
        """Ball-only solid obstacles of hole k as axis-aligned boxes (x0, x1, y0, y1)."""
        c = self.cfg
        y = self.hole_y(k)
        if k == 0:
            x0, x1 = c.mill_x - 0.8, c.mill_x + 0.8
            return [(x0, x1, y + 0.8, y + 1.5), (x0, x1, y - 1.5, y - 0.8)]
        if k == 2:
            return [(c.tunnel_x0, c.tunnel_x1, y + 0.82, y + 1.2),
                    (c.tunnel_x0, c.tunnel_x1, y - 1.2, y - 0.82)]
        return []

    def posts(self, k: int) -> list[tuple[float, float, float]]:
        y = self.hole_y(k)
        if k == 3:
            return [(9.0, y - 1.6, 0.55), (12.5, y + 1.7, 0.55), (16.0, y - 0.7, 0.55)]
        return []

    def fly_avoid(self, k: int) -> list[tuple[float, float, float, float, float]]:
        """Capsules (ax, ay, bx, by, r) the fly's thorax keeps out of (the solid-looking
        props); a circle is a capsule with a = b."""
        c = self.cfg
        y = self.hole_y(k)
        if k == 0:
            return [(c.mill_x + 0.1, y, c.mill_x + 0.1, y, 2.4)]
        if k == 2:
            return [(c.tunnel_x0, y, c.tunnel_x1, y, 2.15)]
        if k == 3:
            return [(px, py, px, py, r + 1.1) for px, py, r in self.posts(k)]
        return []

    @staticmethod
    def _seg_dist(p, a, b) -> tuple[float, np.ndarray]:
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-12 else min(max(float((p - a) @ ab) / L2, 0.0), 1.0)
        q = a + t * ab
        return float(np.linalg.norm(p - q)), q

    def unplayable(self, k: int) -> list[tuple[tuple[float, float, float, float], np.ndarray]]:
        """Zones where a resting ball can't be putted -> drop point."""
        c = self.cfg
        y = self.hole_y(k)
        if k == 0:
            return [((c.mill_x - 1.3, c.mill_x + 1.1, y - 0.8, y + 0.8), np.array([c.mill_x + 2.0, y]))]
        if k == 2:
            return [((c.tunnel_x0 - 0.2, c.tunnel_x1 + 0.2, y - 0.82, y + 0.82),
                     np.array([c.tunnel_x1 + 1.0, y]))]
        return []

    # ------------------------------------------------------------ build
    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        self._materials(spec)
        from flygym.compose import ContactParams

        cp = ContactParams()
        self._cp = cp
        R = c.ball_radius
        felt = dict(contype=BALL_BIT, conaffinity=0, priority=2, condim=6,
                    friction=(1.0, 0.005, c.felt_rolling), solref=cp.get_solref_tuple(),
                    solimp=cp.get_solimp_tuple(), margin=0.0)
        bouncy = dict(contype=BALL_BIT, conaffinity=0, priority=2, condim=3,
                      friction=(0.15, 0.005, 0.0001), solref=(c.rail_tc, c.rail_dampratio),
                      solimp=cp.get_solimp_tuple(), margin=0.0)
        vis = dict(contact_kwargs("visual"), mass=0.0)
        th = 0.8  # felt slab half thickness (top at z = 0)
        for k, (nm, par, col) in enumerate(HOLES):
            y = self.hole_y(k)
            x0, x1, hw = c.lane_x0, c.lane_x1, c.lane_hw + 0.3
            cx, cy = self.cup_xy(k)
            rc = c.cup_r
            mat = P + f"felt{k}"
            # the felt: 4 boxes round a square hole + 4 45-deg corner boxes -> octagon
            rects = ((x0, cx - rc, y - hw, y + hw), (cx + rc, x1, y - hw, y + hw),
                     (cx - rc, cx + rc, y - hw, cy - rc), (cx - rc, cx + rc, cy + rc, y + hw))
            for j, (a0, a1, b0, b1) in enumerate(rects):
                wb.add_geom(name=f"{P}felt{k}_{j}", type=mj.mjtGeom.mjGEOM_BOX,
                            size=((a1 - a0) / 2, (b1 - b0) / 2, th),
                            pos=((a0 + a1) / 2, (b0 + b1) / 2, -th), material=mat, **felt)
            s = 0.46 * rc
            for j in range(4):
                a = math.pi / 4 + j * math.pi / 2
                d = rc + s
                wb.add_geom(name=f"{P}felt{k}_c{j}", type=mj.mjtGeom.mjGEOM_BOX, size=(s, s, th),
                            pos=(cx + d * math.cos(a), cy + d * math.sin(a), -th),
                            quat=quat_axis_angle((0, 0, 1), a), material=mat, **felt)
            # the cup floor (ball only) and its white liner (visual)
            wb.add_geom(name=f"{P}cup{k}_floor", type=mj.mjtGeom.mjGEOM_BOX, size=(rc * 1.2, rc * 1.2, 0.3),
                        pos=(cx, cy, -c.cup_depth - 0.3), rgba=(0.1, 0.1, 0.1, 1), **felt)
            A.add_mesh(spec, f"{P}cup{k}_liner_mesh", A.cup_liner_mesh(rc * 0.99, c.cup_depth - 0.01))
            wb.add_geom(name=f"{P}cup{k}_liner", type=mj.mjtGeom.mjGEOM_MESH, meshname=f"{P}cup{k}_liner_mesh",
                        pos=(cx, cy, -c.cup_depth / 2), material=P + "liner", **vis)
            # rails (ball only, bouncy): sides, the far end, the back with the entry gap
            rm = P + f"rail{k}"
            rh = c.rail_h / 2
            for sgn in (-1, 1):
                wb.add_geom(name=f"{P}rail{k}_s{int(sgn > 0)}", type=mj.mjtGeom.mjGEOM_BOX,
                            size=((x1 - x0) / 2 + 0.3, 0.15, rh),
                            pos=((x0 + x1) / 2, y + sgn * (c.lane_hw + 0.15), rh), material=rm, **bouncy)
            wb.add_geom(name=f"{P}rail{k}_end", type=mj.mjtGeom.mjGEOM_BOX, size=(0.15, c.lane_hw, rh),
                        pos=(x1 + 0.15, y, rh), material=rm, **bouncy)
            seg = (c.lane_hw - c.entry_hw) / 2
            for sgn in (-1, 1):
                wb.add_geom(name=f"{P}rail{k}_back{int(sgn > 0)}", type=mj.mjtGeom.mjGEOM_BOX,
                            size=(0.15, seg, rh), pos=(x0 - 0.15, y + sgn * (c.entry_hw + seg), rh),
                            material=rm, **bouncy)
            # tee mat, flag, hole sign
            wb.add_geom(name=f"{P}tee{k}", type=mj.mjtGeom.mjGEOM_BOX, size=(1.2, 1.0, 0.004),
                        pos=(c.tee_x, y, 0.004), material=P + "tee_mat", **vis)
            wb.add_geom(name=f"{P}tee{k}_spot", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.25, 0.006, 0),
                        pos=(c.tee_x, y, 0.006), rgba=(1, 1, 1, 1), **vis)
            fx, fy = cx + 1.25, cy + 0.9
            wb.add_geom(name=f"{P}flag{k}_pole", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.05, 2.1, 0),
                        pos=(fx, fy, 2.1), material=P + "white", **vis)
            A.add_mesh(spec, f"{P}flag{k}_mesh", A.panel_mesh(1.3, 0.85, 0.02))
            wb.add_geom(name=f"{P}flag{k}", type=mj.mjtGeom.mjGEOM_MESH, meshname=f"{P}flag{k}_mesh",
                        pos=(fx + 0.68, fy, 3.75), quat=A.panel_quat((0, -1, 0)), material=P + f"flagm{k}", **vis)
            A.add_mesh(spec, f"{P}sign{k}_mesh", A.panel_mesh(3.4, 2.0, 0.08))
            sx, sy = c.lane_x0 - 3.2, y - 3.6
            nrm = np.array([-0.6, -0.8, 0.0])
            wb.add_geom(name=f"{P}sign{k}", type=mj.mjtGeom.mjGEOM_MESH, meshname=f"{P}sign{k}_mesh",
                        pos=(sx, sy, 2.3), quat=A.panel_quat(nrm), material=P + f"sign{k}", **vis)
            tang = np.array([0.8, -0.6])
            for dx in (-1.3, 1.3):
                wb.add_geom(name=f"{P}sign{k}_post{int(dx > 0)}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                            size=(0.07, 0.65, 0), pos=(sx + dx * tang[0], sy + dx * tang[1], 0.65),
                            material=P + "white", **vis)
        self._add_windmill(spec, felt, bouncy, vis)
        self._add_ramp(spec)
        self._add_tunnel(spec, bouncy, vis)
        self._add_posts(spec, bouncy, vis)
        # out-of-bounds catch floor for the ball (well below everything)
        wb.add_geom(name=P + "oob_floor", type=mj.mjtGeom.mjGEOM_BOX, size=(60, 60, 0.5),
                    pos=(10, 25, -4.5), rgba=(0, 0, 0, 0), group=3, **felt)
        # ---- the ball
        ball = wb.add_body(name=P + "ball", pos=(c.tee_x, 0.0, R + 0.002))
        ball.add_freejoint(name=P + "ball_free")
        ball.add_geom(name=P + "ball_geom", type=mj.mjtGeom.mjGEOM_SPHERE, size=(R, 0, 0),
                      mass=c.ball_mass, contype=BALL_BIT, conaffinity=BALL_BIT, condim=3,
                      friction=(1.0, 0.005, 0.0001), solref=cp.get_solref_tuple(),
                      solimp=cp.get_solimp_tuple(), material=P + "ball")
        slippery_body_contact(spec, P + "ball", P + "ball_geom", self.fly_name,
                              friction=c.head_friction)
        self._add_grounds(spec, vis)
        self._add_lights(spec)

    def _materials(self, spec) -> None:
        c = self.cfg
        T = A.add_textured_material
        felt_cols = ((0.10, 0.64, 0.22), (0.12, 0.60, 0.30), (0.20, 0.66, 0.18), (0.08, 0.58, 0.34))
        for k in range(len(HOLES)):
            A.add_texture(spec, f"{P}tex_felt{k}", A.felt_texture(c.seed + k, 256, felt_cols[k]))
            T(spec, f"{P}felt{k}", f"{P}tex_felt{k}", rgba=(1, 1, 1, 1), specular=0.05, shininess=0.1,
              texuniform=True, texrepeat=(0.35, 0.35))
            spec.add_material(name=f"{P}rail{k}", rgba=HOLES[k][2] + (1.0,), specular=0.8, shininess=0.8)
            A.add_texture(spec, f"{P}tex_sign{k}", A.sign_texture(k + 1, HOLES[k][0], HOLES[k][1], HOLES[k][2]))
            T(spec, f"{P}sign{k}", f"{P}tex_sign{k}", rgba=(1, 1, 1, 1), specular=0.2, emission=0.15)
            A.add_texture(spec, f"{P}tex_flag{k}", A.flag_texture(k + 1, HOLES[k][2]))
            T(spec, f"{P}flagm{k}", f"{P}tex_flag{k}", rgba=(1, 1, 1, 1), specular=0.1, emission=0.1)
        A.add_texture(spec, P + "tex_ball", A.golf_ball_texture())
        T(spec, P + "ball", P + "tex_ball", rgba=(1, 1, 1, 1), specular=0.6, shininess=0.7)
        A.add_texture(spec, P + "tex_pavers", A.paver_texture(c.seed))
        T(spec, P + "pavers", P + "tex_pavers", rgba=(1, 1, 1, 1), specular=0.05, texuniform=True,
          texrepeat=(0.18, 0.18))
        A.add_texture(spec, P + "tex_barn", A.barn_texture(c.seed))
        T(spec, P + "barn", P + "tex_barn", rgba=(1, 1, 1, 1), specular=0.2)
        A.add_texture(spec, P + "tex_ramp", A.stripe_texture((0.15, 0.45, 0.95), (0.98, 0.98, 1.0)))
        T(spec, P + "ramp", P + "tex_ramp", rgba=(1, 1, 1, 1), specular=0.5, texuniform=True,
          texrepeat=(0.6, 0.6))
        A.add_texture(spec, P + "tex_pipe", A.stripe_texture((1.0, 0.62, 0.05), (0.95, 0.15, 0.2), k=6))
        T(spec, P + "pipe", P + "tex_pipe", rgba=(1, 1, 1, 1), specular=0.7, shininess=0.7)
        A.add_texture(spec, P + "tex_neon", A.neon_sign_texture())
        T(spec, P + "neon", P + "tex_neon", rgba=(1, 1, 1, 1), emission=0.7)
        A.add_texture(spec, P + "tex_sky", A.sky_texture(c.seed))
        T(spec, P + "sky", P + "tex_sky", rgba=(1, 1, 1, 1), emission=0.85, specular=0.0)
        A.add_texture(spec, P + "tex_tee", A.felt_texture(c.seed + 9, 128, (0.05, 0.25, 0.10)))
        T(spec, P + "tee_mat", P + "tex_tee", rgba=(1, 1, 1, 1), specular=0.05)
        M = spec.add_material
        M(name=P + "liner", rgba=(0.95, 0.95, 0.95, 1), specular=0.5)
        M(name=P + "white", rgba=(0.97, 0.97, 0.97, 1), specular=0.5, shininess=0.6)
        M(name=P + "roof", rgba=(0.20, 0.25, 0.75, 1), specular=0.4)
        M(name=P + "sail", rgba=(0.99, 0.95, 0.85, 0.92), specular=0.2)
        M(name=P + "door", rgba=(0.12, 0.06, 0.05, 1), specular=0.1)
        M(name=P + "rubber", rgba=(0.1, 0.1, 0.12, 1), specular=0.3)
        M(name=P + "post_a", rgba=(1.0, 0.25, 0.75, 1), specular=0.8, shininess=0.8)
        M(name=P + "post_b", rgba=(0.1, 0.85, 0.95, 1), specular=0.8, shininess=0.8)
        M(name=P + "leaf", rgba=(0.15, 0.62, 0.22, 1), specular=0.2)
        M(name=P + "trunk", rgba=(0.55, 0.36, 0.2, 1), specular=0.1)
        for nm, rgba in (("fl_red", (0.95, 0.15, 0.25, 1)), ("fl_yel", (1.0, 0.85, 0.1, 1)),
                         ("fl_pink", (1.0, 0.5, 0.8, 1))):
            M(name=P + nm, rgba=rgba, specular=0.3)

    def _add_windmill(self, spec, felt, bouncy, vis) -> None:
        """Hole 1: a barn-red windmill across the lane; the ball can go through the
        door (the straight line to the cup) or round the sides. Four sails turn in
        front of the door (a mocap body, ball-only colliders: kinematic, labelled)."""
        c = self.cfg
        wb = spec.worldbody
        y = self.hole_y(0)
        mx = c.mill_x
        H = 3.2  # house height
        for j, (x0, x1, y0, y1) in enumerate(self.obstacles(0)):
            wb.add_geom(name=f"{P}mill_col{j}", type=mj.mjtGeom.mjGEOM_BOX,
                        size=((x1 - x0) / 2, (y1 - y0) / 2, 0.9), pos=((x0 + x1) / 2, (y0 + y1) / 2, 0.9),
                        rgba=(0, 0, 0, 0), group=3, **bouncy)
            A.add_mesh(spec, f"{P}mill_side{j}_mesh", A.box_mesh(((x1 - x0) / 2, (y1 - y0) / 2, H / 2), 1.0))
            wb.add_geom(name=f"{P}mill_side{j}", type=mj.mjtGeom.mjGEOM_MESH, meshname=f"{P}mill_side{j}_mesh",
                        pos=((x0 + x1) / 2, (y0 + y1) / 2, H / 2), material=P + "barn", **vis)
        lz0 = 1.3
        A.add_mesh(spec, P + "mill_lintel_mesh", A.box_mesh((0.8, 0.8, (H - lz0) / 2), 1.0))
        wb.add_geom(name=P + "mill_lintel", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "mill_lintel_mesh",
                    pos=(mx, y, (H + lz0) / 2), material=P + "barn", **vis)
        A.add_mesh(spec, P + "mill_roof_mesh", A.gable_roof_mesh(1.6, 3.0, 1.5, 0.2))
        wb.add_geom(name=P + "mill_roof", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "mill_roof_mesh",
                    pos=(mx, y, H), material=P + "roof", **vis)
        wb.add_geom(name=P + "mill_doorway", type=mj.mjtGeom.mjGEOM_BOX, size=(0.79, 0.79, 0.64),
                    pos=(mx, y, 0.66), material=P + "door", **vis)
        # (the dark door block is visual; the ball-only door is the gap between the colliders)
        hub = np.array([mx - 0.95, y, 3.1])
        sails = wb.add_body(name=P + "sails", mocap=True, pos=tuple(hub))
        L = 2.95
        A.add_mesh(spec, P + "sail_mesh", A.blade_mesh(L, 0.75))
        A.add_mesh(spec, P + "canvas_mesh", A.box_mesh((0.01, 0.34, (L - 0.5) / 2)))
        for j in range(4):
            a = j * math.pi / 2
            q = quat_axis_angle((1, 0, 0), a)
            R = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
            cpos = R @ np.array([0.0, 0.0, (L + 0.25) / 2])
            sails.add_geom(name=f"{P}sail{j}_col", type=mj.mjtGeom.mjGEOM_BOX,
                           size=(0.08, 0.2, (L - 0.25) / 2), pos=tuple(cpos), quat=q, rgba=(0, 0, 0, 0),
                           group=3, **bouncy)
            sails.add_geom(name=f"{P}sail{j}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "sail_mesh", quat=q,
                           material=P + "roof", **vis)
            sails.add_geom(name=f"{P}canvas{j}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "canvas_mesh",
                           pos=tuple(R @ np.array([0.03, 0.0, (L + 0.5) / 2])), quat=q, material=P + "sail",
                           **vis)
        sails.add_geom(name=P + "hub", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.3, 0.14, 0),
                       quat=quat_axis_angle((0, 1, 0), math.pi / 2), material=P + "white", **vis)

    def _add_ramp(self, spec) -> None:
        """Hole 2: a hump across the lane (10 deg up, a flat top, 10 deg down); a real
        static slope for both the fly and the ball."""
        c = self.cfg
        wb = spec.worldbody
        y = self.hole_y(1)
        a, b, cc, d = self.ramp_profile()
        h = c.ramp_h
        hw = c.lane_hw
        ang = math.radians(c.ramp_deg)
        geoms = [add_slope(wb, P + "ramp_up", a, b, 0.0, ang, hw, y=y, thickness=0.4, collide="static"),
                 add_box(wb, P + "ramp_top", ((cc - b) / 2, hw, 0.2), ((b + cc) / 2, y, h - 0.2),
                         collide="static"),
                 add_slope(wb, P + "ramp_down", cc, d, h, -ang, hw, y=y, thickness=0.4, collide="static")]
        for g in geoms:  # the fly and the ball both roll / walk on it
            g.contype = TERRAIN_BIT | BALL_BIT
            g.conaffinity = FLY_BIT
            g.priority = 2
            g.condim = 6
            g.friction = [1.0, 0.005, c.felt_rolling]
            g.material = P + "ramp"

    def _add_tunnel(self, spec, bouncy, vis) -> None:
        c = self.cfg
        wb = spec.worldbody
        y = self.hole_y(2)
        for j, (x0, x1, y0, y1) in enumerate(self.obstacles(2)):
            wb.add_geom(name=f"{P}tunnel_col{j}", type=mj.mjtGeom.mjGEOM_BOX,
                        size=((x1 - x0) / 2, (y1 - y0) / 2, 0.8), pos=((x0 + x1) / 2, (y0 + y1) / 2, 0.8),
                        rgba=(0, 0, 0, 0), group=3, **bouncy)
        L = c.tunnel_x1 - c.tunnel_x0
        A.add_mesh(spec, P + "pipe_mesh", A.pipe_mesh(0.95, 1.2, L))
        wb.add_geom(name=P + "pipe", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "pipe_mesh",
                    pos=((c.tunnel_x0 + c.tunnel_x1) / 2, y, 0.62), material=P + "pipe", **vis)
        for x in (c.tunnel_x0, c.tunnel_x1):
            A.add_mesh(spec, f"{P}pipe_ring{int(x)}_mesh", A.torus_mesh(1.08, 0.14, 40, 10))
            wb.add_geom(name=f"{P}pipe_ring{int(x)}", type=mj.mjtGeom.mjGEOM_MESH,
                        meshname=f"{P}pipe_ring{int(x)}_mesh", pos=(x, y, 0.62),
                        quat=quat_axis_angle((0, 1, 0), math.pi / 2), material=P + "white", **vis)

    def _add_posts(self, spec, bouncy, vis) -> None:
        wb = spec.worldbody
        for j, (px, py, r) in enumerate(self.posts(3)):
            wb.add_geom(name=f"{P}post{j}_col", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(r, 0.6, 0),
                        pos=(px, py, 0.6), rgba=(0, 0, 0, 0), group=3, **bouncy)
            A.add_mesh(spec, f"{P}post{j}_mesh", A.post_mesh(r, 1.4))
            wb.add_geom(name=f"{P}post{j}", type=mj.mjtGeom.mjGEOM_MESH, meshname=f"{P}post{j}_mesh",
                        pos=(px, py, 0.7), material=P + ("post_a" if j % 2 == 0 else "post_b"), **vis)
            A.add_mesh(spec, f"{P}post{j}_ring_mesh", A.torus_mesh(r + 0.03, 0.09, 32, 8))
            wb.add_geom(name=f"{P}post{j}_ring", type=mj.mjtGeom.mjGEOM_MESH, meshname=f"{P}post{j}_ring_mesh",
                        pos=(px, py, 0.55), material=P + "rubber", **vis)

    def _add_grounds(self, spec, vis) -> None:
        """Walkway pavers round the lanes (visual), palms, flower beds, a neon sign and
        a sky backdrop."""
        c = self.cfg
        wb = spec.worldbody
        n = len(HOLES)
        X0, X1 = -45.0, 60.0
        Y0, Y1 = -30.0, self.hole_y(n - 1) + 16.0
        lx0, lx1 = c.lane_x0 - 0.3, c.lane_x1 + 0.3
        rects = [(X0, lx0, Y0, Y1), (lx1, X1, Y0, Y1)]
        ys = [Y0] + [v for k in range(n) for v in (self.hole_y(k) - c.lane_hw - 0.3,
                                                   self.hole_y(k) + c.lane_hw + 0.3)] + [Y1]
        for j in range(0, len(ys), 2):
            rects.append((lx0, lx1, ys[j], ys[j + 1]))
        for j, (a0, a1, b0, b1) in enumerate(rects):
            wb.add_geom(name=f"{P}ground{j}", type=mj.mjtGeom.mjGEOM_BOX,
                        size=((a1 - a0) / 2, (b1 - b0) / 2, 0.05), pos=((a0 + a1) / 2, (b0 + b1) / 2, -0.05),
                        material=P + "pavers", **vis)
        rng = np.random.default_rng(c.seed + 4)
        for j in range(8):
            px = rng.choice([-12.0, 32.0]) if j < 6 else rng.uniform(0, 20)
            py = rng.uniform(-8.0, Y1 - 4) if j < 6 else Y1 - 3.0
            pm = A.palm_meshes(rng.uniform(9.0, 13.0), seed=j)
            for part, mat in (("trunk", "trunk"), ("fronds", "leaf")):
                A.add_mesh(spec, f"{P}palm{j}_{part}_mesh", pm[part])
                wb.add_geom(name=f"{P}palm{j}_{part}", type=mj.mjtGeom.mjGEOM_MESH,
                            meshname=f"{P}palm{j}_{part}_mesh", pos=(px, py, 0.0), material=P + mat, **vis)
        for k in range(n - 1):  # flower beds between the lanes
            yb = self.hole_y(k) + c.lane_pitch / 2
            for m_, nm in enumerate(("fl_red", "fl_yel", "fl_pink")):
                parts = [A.flower_clump_mesh((x, yb + rng.uniform(-1, 1), 0), 7, 0.7, 0.18, seed=int(x * 10) + m_)
                         for x in np.arange(3.0 + 2 * m_, 22.0, 6.0)]
                A.add_mesh(spec, f"{P}bed{k}_{nm}_mesh", A.merge(*parts))
                wb.add_geom(name=f"{P}bed{k}_{nm}", type=mj.mjtGeom.mjGEOM_MESH,
                            meshname=f"{P}bed{k}_{nm}_mesh", material=P + nm, **vis)
            shr = [A.shrub_mesh((x, yb), 0.9, seed=int(x)) for x in np.arange(0.0, 24.0, 6.0)]
            A.add_mesh(spec, f"{P}bed{k}_shrubs_mesh", A.merge(*shr))
            wb.add_geom(name=f"{P}bed{k}_shrubs", type=mj.mjtGeom.mjGEOM_MESH,
                        meshname=f"{P}bed{k}_shrubs_mesh", material=P + "leaf", **vis)
        A.add_mesh(spec, P + "neon_mesh", A.panel_mesh(16.0, 4.0, 0.2))
        wb.add_geom(name=P + "neon", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "neon_mesh",
                    pos=(10.0, Y1 - 1.0, 7.0), quat=A.panel_quat((0, -1, 0)), material=P + "neon", **vis)
        for dx in (-6.0, 6.0):
            wb.add_geom(name=f"{P}neon_post{int(dx > 0)}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.15, 2.5, 0),
                        pos=(10.0 + dx, Y1 - 0.9, 2.5), material=P + "white", **vis)
        A.add_mesh(spec, P + "sky_mesh", A.panel_mesh(240.0, 80.0, 0.5))
        wb.add_geom(name=P + "sky", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "sky_mesh",
                    pos=(10.0, Y1 + 60.0, 25.0), quat=A.panel_quat((0, -1, 0)), material=P + "sky", **vis)
        sky = spec.texture("skybox")
        if sky is not None:
            sky.rgb1 = (0.35, 0.62, 0.98)
            sky.rgb2 = (0.80, 0.92, 1.0)

    def _add_lights(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        spec.visual.headlight.ambient = (0.33, 0.33, 0.33)
        spec.visual.headlight.diffuse = (0.30, 0.30, 0.30)
        spec.visual.headlight.specular = (0.1, 0.1, 0.1)
        ymid = self.hole_y(len(HOLES) - 1) / 2
        tgt = np.array([10.0, ymid, 0.0])
        sun = np.array([-10.0, ymid - 45.0, 70.0])
        wb.add_light(name=P + "sun", type=mj.mjtLightType.mjLIGHT_SPOT, pos=tuple(sun), dir=tuple(tgt - sun),
                     diffuse=(0.75, 0.73, 0.68), specular=(0.4, 0.4, 0.4), cutoff=45.0, exponent=0.2,
                     castshadow=bool(c.shadows))
        wb.add_light(name=P + "fill", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(30, ymid, 40),
                     dir=(-0.4, 0.3, -1.0), diffuse=(0.22, 0.24, 0.30), specular=(0.05, 0.05, 0.05),
                     castshadow=False)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        m = self.sim.model
        c = self.cfg
        gp = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "ground_plane")
        if gp >= 0:  # the fly's floor stays (physics); the course draws its own ground
            m.geom_group[gp] = 3
            m.geom_rgba[gp, 3] = 0.0
        self.ball_body = m.body(P + "ball").id
        self.ball_geom = m.geom(P + "ball_geom").id
        j = m.joint(P + "ball_free").id
        self.bq, self.bv = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        self.sails_mocap = int(m.body_mocapid[m.body(P + "sails").id])
        self.ball_pairs = np.flatnonzero((m.pair_geom1 == self.ball_geom) | (m.pair_geom2 == self.ball_geom))
        root = self.sim.thorax_body_id
        self.fly_bodies = np.array([b for b in range(m.nbody) if b == root or m.body_rootid[b] == root])
        self._rng = np.random.default_rng(c.seed)
        # counters / scorecard
        self.hole = 0
        self.strokes = 0  # on the current hole (incl. penalties)
        self.card: list[int | None] = [None] * len(HOLES)
        self.rounds = 0
        self.best_round: int | None = None
        self.last_round: int | None = None
        self.total_strokes = 0
        self.total_par = 0
        self.holes_played = 0
        self.holes_in_one = 0
        self.birdies = 0
        self.eagles = 0
        self.pars = 0
        self.bogeys = 0
        self.putts = 0
        self.penalties = 0
        self.oob = 0
        self.drops = 0
        self.picked_up = 0
        self.sail_hits = 0
        self.mill_waits = 0
        self.last_shot = ""
        self.message = ""
        self._caption = ("", -1e9)
        self._scale = 1.0
        self._last_scale = 1.0
        self.pilot = GolfPilot(self, c.ball_radius, behind_gap=c.behind_gap,
                               orbit_clearance=c.orbit_clearance, push_speed=c.push_speed,
                               approach_speed=c.approach_speed, clamp=self._clamp)
        self._avoid_dir: dict[int, float] = {}
        self.hole = int(c.start_hole) % len(HOLES)
        self._ball_last = np.array([c.tee_x, self.hole_y(self.hole), c.ball_radius + 0.002])
        self._place_ball(self._ball_last)
        self._lie = self._ball_last.copy()
        mj.mj_forward(m, self.sim.data)
        self._begin_play()

    # ------------------------------------------------------------ ball helpers
    def ball_pos(self) -> np.ndarray:
        return self.sim.data.qpos[self.bq:self.bq + 3].copy()

    def ball_speed(self) -> float:
        return float(np.linalg.norm(self.sim.data.qvel[self.bv:self.bv + 3]))

    def _place_ball(self, p, vel=None) -> None:
        d = self.sim.data
        d.qpos[self.bq:self.bq + 3] = p
        d.qpos[self.bq + 3:self.bq + 7] = (1, 0, 0, 0)
        d.qvel[self.bv:self.bv + 6] = 0.0
        if vel is not None:
            d.qvel[self.bv:self.bv + 3] = vel

    def _ball_contacts(self, on: bool) -> None:
        m = self.sim.model
        m.geom_contype[self.ball_geom] = BALL_BIT if on else 0
        m.geom_conaffinity[self.ball_geom] = BALL_BIT if on else 0
        m.body_gravcomp[self.ball_body] = 0.0 if on else 1.0

    def fly_touching_ball(self) -> bool:
        d = self.sim.data
        n = d.ncon
        if n == 0:
            return False
        g = d.contact.geom[:n]
        m = self.sim.model
        for a, b in g:
            if a == self.ball_geom or b == self.ball_geom:
                o = b if a == self.ball_geom else a
                if m.geom_bodyid[o] in self.fly_bodies:
                    return True
        return False

    def sail_angle(self, t: float) -> float:
        return 2 * math.pi * self.cfg.mill_rpm / 60.0 * t

    def sails_block(self, t: float) -> bool:
        """True if a sail is near straight down (in the door's way) at time t."""
        a = self.sail_angle(t) % (math.pi / 2)
        a = min(a, math.pi / 2 - a)
        return a < 0.52

    # ------------------------------------------------------------ planning (aim model)
    def _clear_point(self, k: int, p, margin: float) -> bool:
        c = self.cfg
        y = self.hole_y(k)
        if not (c.lane_x0 + margin <= p[0] <= c.lane_x1 - margin and abs(p[1] - y) <= c.lane_hw - margin):
            return False
        for x0, x1, y0, y1 in self.obstacles(k):
            if x0 - margin < p[0] < x1 + margin and y0 - margin < p[1] < y1 + margin:
                return False
        for px, py, r in self.posts(k):
            if math.hypot(p[0] - px, p[1] - py) < r + margin:
                return False
        return True

    def clear_line(self, k: int, a, b, margin: float | None = None) -> bool:
        m = self.cfg.ball_radius + 0.05 if margin is None else margin
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        for s in np.linspace(0, 1, max(2, int(L / 0.1) + 1)):
            if not self._clear_point(k, a + s * (b - a), m):
                return False
        return True

    def gates(self, k: int) -> list[np.ndarray]:
        """Points just past the door / tunnel: aim through them toward the cup."""
        c = self.cfg
        y = self.hole_y(k)
        if k == 0:
            return [np.array([c.mill_x + 1.4, y])]
        if k == 2:
            return [np.array([c.tunnel_x1 + 0.6, y])]
        return []

    def speed_for(self, k: int, ball, target, d_stop: float) -> float:
        """Putt speed to stop d_stop mm along the line (felt deceleration; + the
        energy to get over the ramp's top)."""
        c = self.cfg
        a = c.felt_decel
        v2 = 2 * a * d_stop
        if k == 1:
            r0, _, _, r3 = self.ramp_profile()
            if ball[0] < r0 and target[0] > r0:
                top = (r0 + r3) / 2 - ball[0]
                v2 = max(v2, 2 * a * top + 1.25 * (10.0 / 7.0) * G * c.ramp_h)
        return min(math.sqrt(max(v2, 1.0)), c.max_speed)

    def plan_shot(self, k: int, ball) -> dict:
        """Aim model: target point, direction and speed for a putt from ``ball``."""
        c = self.cfg
        b = np.asarray(ball[:2], float)
        cup = self.cup_xy(k)
        if self.clear_line(k, b, cup):
            d = float(np.linalg.norm(cup - b)) + c.overshoot
            kind, tgt = "at the cup", cup
        else:
            kind = tgt = None
            for g in self.gates(k):
                if g[0] > b[0] + 0.5 and self.clear_line(k, b, g):
                    tgt, kind = g, "through the " + ("mill" if k == 0 else "tunnel")
                    d = float(np.linalg.norm(cup - b)) + c.overshoot
                    break
            if tgt is None:  # a lay-up: the via point with the shortest clear route
                best = None
                y = self.hole_y(k)
                for vx in np.arange(c.lane_x0 + 1.5, c.lane_x1 - 1.0, 1.0):
                    for vy in np.arange(y - c.lane_hw + 1.2, y + c.lane_hw - 1.1, 1.0):
                        v = np.array([vx, vy])
                        if np.linalg.norm(v - b) < 2.0:
                            continue
                        if self.clear_line(k, b, v) and self.clear_line(k, v, cup):
                            cost = float(np.linalg.norm(v - b) + np.linalg.norm(cup - v))
                            if best is None or cost < best[0]:
                                best = (cost, v)
                if best is None:
                    tgt, d, kind = cup, float(np.linalg.norm(cup - b)) + c.overshoot, "at the cup (blocked)"
                else:
                    tgt, d, kind = best[1], float(np.linalg.norm(best[1] - b)), "lay-up"
        u = tgt - b
        u = u / max(float(np.linalg.norm(u)), 1e-9)
        v = self.speed_for(k, b, tgt, d)
        return {"target": tgt, "dir": u, "speed": v, "kind": kind, "dist": d}

    # ------------------------------------------------------------ fly routing
    def _clamp(self, p) -> np.ndarray:
        c = self.cfg
        y = self.hole_y(self.hole)
        return np.array([min(max(float(p[0]), c.lane_x0 + 0.6), c.lane_x1 + 0.8),
                         min(max(float(p[1]), y - c.lane_hw + 0.5), y + c.lane_hw - 0.5)])

    def route(self, p, goal) -> np.ndarray:
        """Next waypoint from p toward goal: through the lanes' entry gaps and the
        walkway behind them, and round the solid obstacles inside a lane."""
        c = self.cfg
        kp = self.lane_of(p[0], p[1])
        kg = self.lane_of(goal[0], goal[1])
        if kg < 0 and kp >= 0 and self.lane_of(goal[0], goal[1], 2.6) == kp:
            kg = kp  # a (clamped) waypoint just outside the rails of this lane
        xin, xout = c.lane_x0 + 0.8, c.lane_x0 - 2.2
        if kp != kg:
            if kp >= 0:  # leave this lane through its entry gap
                yp = self.hole_y(kp)
                if p[0] > c.lane_x0 + 1.2 or abs(p[1] - yp) > c.entry_hw - 0.6:
                    return self._avoid(kp, p, np.array([xin, yp]))
                return np.array([xout, yp])
            if kg >= 0:  # on the walkway: along it, then in through the gap
                yg = self.hole_y(kg)
                if abs(p[1] - yg) > 1.3:
                    return np.array([xout, yg]) if p[0] < c.lane_x0 - 0.5 else np.array([xout, p[1]])
                return np.array([xin + 0.8, yg])
            return goal
        if kp >= 0:
            return self._avoid(kp, p, goal)
        return goal

    def _avoid(self, k: int, p, goal) -> np.ndarray:
        p = np.asarray(p, float)
        goal = np.asarray(goal, float).copy()
        caps = [(np.array(o[:2]), np.array(o[2:4]), o[4]) for o in self.fly_avoid(k)]
        for A_, B_, r in caps:  # a goal inside an obstacle -> just outside it
            dd, q = self._seg_dist(goal, A_, B_)
            if dd < r + 0.3:
                v = goal - q
                n = float(np.linalg.norm(v))
                goal = q + (v / n if n > 1e-6 else np.array([0.0, 1.0])) * (r + 0.35)
        d = goal - p
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            return goal
        u = d / L
        best = None
        for i, (A_, B_, r) in enumerate(caps):
            for s_ in np.linspace(0.0, L, max(2, int(L / 0.2) + 1)):
                dd, _ = self._seg_dist(p + u * s_, A_, B_)
                if dd < r:
                    if best is None or s_ < best[0]:
                        best = (s_, i)
                    break
        if best is None:
            return goal
        i = best[1]
        A_, B_, r = caps[i]
        ax = B_ - A_
        La = float(np.linalg.norm(ax))
        ax = ax / La if La > 1e-9 else u
        nrm = np.array([-ax[1], ax[0]])
        _, q = self._seg_dist(p, A_, B_)
        side = self._avoid_dir.get(k * 100 + i)
        if side is None:
            side = 1.0 if float((p - q) @ nrm) >= 0 else -1.0
            if abs(float((p - q) @ nrm)) < 0.2:  # dead centre: the side nearer the goal
                side = 1.0 if float((goal - q) @ nrm) >= 0 else -1.0
            self._avoid_dir[k * 100 + i] = side
        off = nrm * side * (r + 0.45)
        lat = float((p - q) @ nrm) * side
        if lat < r + 0.2:  # step out to the side first
            return q + off
        # then along the side to the corner toward the goal
        end = B_ + ax * 0.2 if float((goal - p) @ ax) >= 0 else A_ - ax * 0.2
        return end + off

    def _goto(self, goal, speed: float = 0.9, tol: float = 0.45) -> bool:
        p = self.fly_xy()
        if float(np.linalg.norm(goal - p)) < tol:
            return True
        wp = self.route(p, goal)
        PushPilot.aim_at(self.pilot, wp, speed)
        return False

    # ------------------------------------------------------------ phases
    def _set_phase(self, ph: str) -> None:
        self.phase = ph
        self.state = ph
        self._t_phase = self.run_time()
        if hasattr(self, "_avoid_dir"):
            self._avoid_dir.clear()

    def _begin_play(self) -> None:
        self._set_phase("play")
        self.pilot.reset()
        self._avoid_dir.clear()
        self.shot = self.plan_shot(self.hole, self.ball_pos())
        self._stop_since = None
        self._waited = False
        self._mill_wait = False

    def _caption_say(self, text: str) -> None:
        self.message = text
        self._caption = (text, self.run_time())
        self.say(text)

    def _putt(self) -> None:
        c = self.cfg
        sh = self.shot
        s = 1.0 - c.skill
        err = math.radians(c.aim_sd_deg + c.aim_sd_unskilled_deg * s) * self._rng.normal()
        sp = sh["speed"] * max(0.3, 1.0 + (c.speed_sd + c.speed_sd_unskilled * s) * self._rng.normal())
        u = sh["dir"]
        ca, sa = math.cos(err), math.sin(err)
        u = np.array([ca * u[0] - sa * u[1], sa * u[0] + ca * u[1]])
        b = self.ball_pos()
        self._lie = b.copy()
        self._lie_damping(False)
        vel = np.array([u[0] * sp, u[1] * sp, 0.0])
        self._place_ball(b, vel)
        R = c.ball_radius
        self.sim.data.qvel[self.bv + 3:self.bv + 6] = np.cross([0, 0, 1.0], vel) / R  # rolling
        self.strokes += 1
        self.putts += 1
        self.add_work(0.0)
        self.last_shot = (f"stroke {self.strokes}: {sh['kind']}, {sp:.0f} mm/s "
                          f"(plan {sh['speed']:.0f}), aim err {math.degrees(err):+.1f} deg")
        self.say(f"hole {self.hole + 1} {self.last_shot}")
        self._set_phase("follow")

    def _hole_done(self, holed: bool) -> None:
        k = self.hole
        n = self.strokes
        par = self.par(k)
        self.card[k] = n
        self.total_strokes += n
        self.total_par += par
        self.holes_played += 1
        self.add_work(1.0)
        if holed and n == 1:
            self.holes_in_one += 1
            cap = "HOLE IN ONE!"
        elif not holed:
            self.picked_up += 1
            cap = f"picked up ({n} strokes max)"
        else:
            diff = n - par
            cap = {-2: "EAGLE!", -1: "BIRDIE!", 0: "PAR", 1: "BOGEY"}.get(diff, f"+{diff}" if diff > 0 else "ALBATROSS!")
            if diff == -1:
                self.birdies += 1
            elif diff <= -2:
                self.eagles += 1
            elif diff == 0:
                self.pars += 1
            elif diff == 1:
                self.bogeys += 1
        self._caption_say(f"HOLE {k + 1}: {cap}" if cap != "HOLE IN ONE!" else f"HOLE {k + 1}: HOLE IN ONE!")
        if k == len(HOLES) - 1:
            tot = sum(v for v in self.card if v is not None)
            self.rounds += 1
            self.last_round = tot
            self.best_round = tot if self.best_round is None else min(self.best_round, tot)
            self.say(f"ROUND {self.rounds}: {tot} strokes (par {sum(h[1] for h in HOLES)})")
        self._retrieve_at = self.ball_pos()
        self._set_phase("retrieve_walk")

    def _next_hole(self) -> None:
        self.hole = (self.hole + 1) % len(HOLES)
        if self.hole == 0:
            self.card = [None] * len(HOLES)
        self.strokes = 0

    # ------------------------------------------------------------ the loop
    def update(self) -> None:
        c = self.cfg
        d = self.sim.data
        t = self.sim.time
        rt = self.run_time()
        # the windmill sails turn forever
        a = self.sail_angle(t)
        d.mocap_quat[self.sails_mocap] = quat_axis_angle((1, 0, 0), a)
        b = self.ball_pos()
        if not np.all(np.isfinite(b)):
            self._place_ball(self._ball_last)
            b = self.ball_pos()
        ph = self.phase
        if ph in ("play", "follow", "roll"):
            self._ball_last = b.copy()
        acts = self.session.actions
        stand = ph in ("follow", "roll", "retrieve", "place") or (ph == "play" and self._mill_wait)
        if stand and not acts.busy:
            from fly_simulator.jobs.base import _make_action

            acts.trigger(_make_action("freeze", duration=120.0), source="job")
        elif not stand and acts.active_name == "freeze":
            acts.cancel()
        if acts.busy and acts.active_name != "freeze":
            self.steering.set(None, 1.0)
            return
        if ph == "play":
            self._play(b, t)
        elif ph == "follow":
            self.steering.set(None, 0.0)
            if rt - self._t_phase > c.follow_s:
                self._set_phase("roll")
                self._stop_since = None
            self._roll_check(b, rt)
        elif ph == "roll":
            self.steering.set(None, 0.0)
            self._roll_check(b, rt)
        elif ph == "retrieve_walk":
            cup = self._retrieve_at[:2]
            fp = self.fly_xy()
            u = fp - cup
            u = u / max(float(np.linalg.norm(u)), 1e-9)
            if self._goto(cup + u * 1.9, 0.9, 0.5) or rt - self._t_phase > 25.0:
                self._ball_contacts(False)
                self._lift_from = self.ball_pos()
                self._set_phase("retrieve")
        elif ph == "retrieve":
            self.steering.set(None, 0.0)
            u = min(1.0, (rt - self._t_phase) / 0.8)
            e = u * u * (3 - 2 * u)
            p = (1 - e) * self._lift_from + e * self._carry_pos()
            p[2] += 1.2 * math.sin(math.pi * e)
            self._place_ball(p)
            if u >= 1.0:
                self._next_hole()
                self._set_phase("carry")
        elif ph == "carry":
            self._place_ball(self._carry_pos())
            tee = self.tee_xy(self.hole)
            stand = tee - np.array([c.ball_radius + c.behind_gap + 0.9, 0.0])
            if self._goto(stand, 0.9, 0.45) or rt - self._t_phase > 40.0:
                self._lift_from = self.ball_pos()
                self._set_phase("place")
        elif ph == "place":
            self.steering.set(None, 0.0)
            u = min(1.0, (rt - self._t_phase) / 0.7)
            e = u * u * (3 - 2 * u)
            tee = np.append(self.tee_xy(self.hole), c.ball_radius + 0.002)
            p = (1 - e) * self._lift_from + e * tee
            p[2] += 1.4 * math.sin(math.pi * e)
            self._place_ball(p)
            if u >= 1.0:
                self._place_ball(tee)
                self._ball_contacts(True)
                self._caption_say(f"HOLE {self.hole + 1}: {HOLES[self.hole][0]}  (par {self.par(self.hole)})")
                self._begin_play()

    def _carry_pos(self) -> np.ndarray:
        d = self.sim.data
        p = d.xpos[self.sim.thorax_body_id].copy()
        R = d.xmat[self.sim.thorax_body_id].reshape(3, 3)
        return p + R @ np.array([-0.25, 0.0, 0.0]) + np.array([0.0, 0.0, 1.9])

    def _lie_damping(self, on: bool) -> None:
        """A resting ball on its lie: heavy damping on its free joint, so a bump of the
        walking fly only nudges it (a real ball would roll away); off for the putt."""
        m = self.sim.model
        m.dof_damping[self.bv:self.bv + 6] = self.cfg.lie_damping if on else 0.0

    def _play(self, b, t: float) -> None:
        c = self.cfg
        sh = self.shot
        # the ball lies still on its spot until it is putted (a walking fly's bump
        # would roll the light ball away); the fly touches it only when addressing it
        self._lie_damping(True)
        goal = b[:2] + sh["dir"] * 8.0
        st = self.pilot.step(b[:2], goal)
        if st != "push":
            return
        # the windmill: wait at the ball until the sails will be clear at the door
        if self.hole == 0 and self._through_door(b[:2], sh["dir"]):
            door = np.array([c.mill_x - 1.0, self.hole_y(0)])
            dist = float(np.linalg.norm(door - b[:2]))
            v = sh["speed"]
            disc = v * v - 2 * c.felt_decel * dist
            t_travel = (v - math.sqrt(disc)) / c.felt_decel if disc > 0 else 1.0
            t_arr = t + 0.25 + t_travel
            if self.sails_block(t_arr) and self.run_time() - self._t_phase < 25.0:
                if not self._waited:
                    self._waited = True
                    self.mill_waits += 1
                self._mill_wait = True
                self.pilot.set_heading(math.atan2(b[1] - self.fly_xy()[1], b[0] - self.fly_xy()[0]), 0.0)
                return
            self._mill_wait = False
        fp = self.fly_xy()
        if self.fly_touching_ball() or float(np.linalg.norm(fp - b[:2])) < c.ball_radius + 0.75:
            self._putt()

    def _through_door(self, b, u) -> bool:
        c = self.cfg
        dx = c.mill_x - 1.0 - b[0]
        if dx <= 0 or u[0] <= 0.2:
            return False
        yd = b[1] + u[1] / u[0] * dx - self.hole_y(0)
        return abs(yd) < 1.2

    def _roll_check(self, b, rt: float) -> None:
        c = self.cfg
        k = self.hole
        cup = self.cup_xy(k)
        y = self.hole_y(k)
        if float(np.linalg.norm(b[:2] - cup)) < c.cup_r + 0.1 and b[2] < -0.2:
            if self.ball_speed() < 20.0 or b[2] < -0.5:
                self._hole_done(True)
            return
        if self.hole == 0 and c.mill_x - 1.3 < b[0] < c.mill_x - 0.8 and abs(b[1] - y) < 1.0 \
                and self.ball_speed() > 5 and b[0] > 0 and self.sim.data.qvel[self.bv] < -5 \
                and not getattr(self, "_hit_sail", False):
            self._hit_sail = True
            self.sail_hits += 1
        out = (b[2] < -0.4 or b[0] < c.lane_x0 - 0.2 or b[0] > c.lane_x1 + 0.2
               or abs(b[1] - y) > c.lane_hw + 0.2)
        if out:
            self.oob += 1
            self.penalties += 1
            self.strokes += 1
            self._caption_say("OUT OF BOUNDS  (+1)")
            self._place_ball(np.append(self._lie[:2], c.ball_radius + 0.002))
            self._after_stroke()
            return
        slow = self.ball_speed() < c.stop_speed
        if slow:
            if self._stop_since is None:
                self._stop_since = rt
        else:
            self._stop_since = None
        if (self._stop_since is not None and rt - self._stop_since > c.stop_hold_s) \
                or rt - self._t_phase > c.roll_timeout_s:
            self._after_stroke()

    def _after_stroke(self) -> None:
        c = self.cfg
        self._hit_sail = False
        if self.strokes >= c.max_strokes:
            self._hole_done(False)
            return
        self._fix_lie()
        self._begin_play()

    def _fix_lie(self) -> None:
        """House rules: move a ball the fly can't putt to a playable spot (counted)."""
        c = self.cfg
        k = self.hole
        b = self.ball_pos()
        p = b[:2].copy()
        moved = False
        for (x0, x1, y0, y1), drop in self.unplayable(k):
            if x0 < p[0] < x1 and y0 < p[1] < y1:
                p, moved = drop.copy(), True
        y = self.hole_y(k)
        lim = c.lane_hw - c.ball_radius - 0.9
        if abs(p[1] - y) > lim:
            p[1] = y + math.copysign(lim, p[1] - y)
            moved = True
        xlo, xhi = c.lane_x0 + c.ball_radius + 0.9, c.lane_x1 - c.ball_radius - 0.9
        if not xlo <= p[0] <= xhi:
            p[0] = min(max(p[0], xlo), xhi)
            moved = True
        for _ in range(30):  # off obstacles / posts: one putter-head away
            if self._clear_point(k, p, c.ball_radius + 0.6):
                break
            p = p + np.array([0.3, 0.0]) if k != 3 else p + (p - self._nearest_post(k, p)) * 0.3
            moved = True
        # room for the fly behind the ball on the planned line
        sh = self.plan_shot(k, p)
        for _ in range(20):
            st = p - sh["dir"] * (c.ball_radius + c.behind_gap)
            inside = float(np.linalg.norm(self._clamp(st) - st)) < 0.05
            if inside and all(self._seg_dist(st, np.array(o[:2]), np.array(o[2:4]))[0] > o[4] - 0.2
                              for o in self.fly_avoid(k)):
                break
            p = p + sh["dir"] * 0.3
            moved = True
            sh = self.plan_shot(k, p)
        f = self.fly_xy()
        for _ in range(12):
            if float(np.linalg.norm(p - f)) > 2.0:
                break
            aw = p - f
            aw = aw / max(float(np.linalg.norm(aw)), 1e-6)
            q = p + aw * 0.5
            if self._clear_point(k, q, c.ball_radius + 0.5):
                p, moved = q, True
            else:
                break
        if moved:
            self.drops += 1
            self.say(f"ball moved to a playable spot ({p[0]:.1f}, {p[1] - y:.1f}) (house rule, no penalty)")
            self._place_ball(np.array([p[0], p[1], c.ball_radius + 0.002 + self.ground_height(*p)]))

    def _nearest_post(self, k: int, p) -> np.ndarray:
        best = min(self.posts(k), key=lambda q: math.hypot(p[0] - q[0], p[1] - q[1]))
        return np.array(best[:2])

    # ------------------------------------------------------------ resets
    def on_reset(self) -> None:
        self._reset_phase = self.phase

    def reset_props(self) -> None:
        """The keyframe reset put the ball on tee 1: put it back where the game has it."""
        c = self.cfg
        ph = getattr(self, "_reset_phase", "play")
        self._ball_contacts(True)
        if ph in ("retrieve_walk", "retrieve", "carry", "place"):
            if ph in ("retrieve_walk", "retrieve"):
                self._next_hole()
            self._place_ball(np.append(self.tee_xy(self.hole), c.ball_radius + 0.002))
        else:
            p = self._ball_last
            if np.all(np.isfinite(p)):
                self._place_ball(np.array([p[0], p[1], max(p[2], c.ball_radius + 0.002)]))
        mj.mj_forward(self.sim.model, self.sim.data)
        self._begin_play()

    # ------------------------------------------------------------ presentation
    def time_scale(self, present_dt: float) -> float:
        c = self.cfg
        fast = self.phase in ("follow", "roll") and self.ball_speed() > 15.0
        tgt = c.slowmo if (fast and c.slowmo < 1.0) else 1.0
        if tgt < self._scale:
            self._scale = tgt
        else:
            self._scale += (1.0 - math.exp(-present_dt / 0.15)) * (tgt - self._scale)
            if self._scale > 0.98:
                self._scale = 1.0
        self._last_scale = self._scale
        return self._scale

    def post_process(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Labels (a screen overlay): the slow motion (edit, not physics) and the
        hole captions (HOLE IN ONE!, BIRDIE!, ...)."""
        cap, t_cap = self._caption
        show = self.cfg.captions and cap and 0 <= t - t_cap < 1.8
        if self._last_scale >= 1.0 and not show:
            return frame
        import cv2

        H, W = frame.shape[:2]
        out = np.ascontiguousarray(frame).copy()
        fs = max(0.4, W / 1400.0)
        th = max(1, int(round(W / 700)))
        if self._last_scale < 1.0:
            txt = f"SLOW MOTION x{self._last_scale:.2f}  (edit, not physics)"
            (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            x, y = W - tw - int(0.015 * W), H - int(0.03 * H)
            cv2.rectangle(out, (x - 6, y - tht - 6), (x + tw + 6, y + 6), (0, 0, 0), -1)
            cv2.putText(out, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 220, 90), th, cv2.LINE_AA)
        if show:
            f2 = fs * 2.0
            t2 = th * 2 + 1
            (tw, tht), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_DUPLEX, f2, t2)
            x, y = (W - tw) // 2, int(0.13 * H) + tht
            cv2.putText(out, cap, (x + 2, y + 2), cv2.FONT_HERSHEY_DUPLEX, f2, (30, 0, 40), t2 + 2, cv2.LINE_AA)
            cv2.putText(out, cap, (x, y), cv2.FONT_HERSHEY_DUPLEX, f2, (255, 230, 60), t2, cv2.LINE_AA)
        return out

    def camera_target(self) -> np.ndarray:
        f = self.sim.thorax_position()
        b = self.ball_pos()
        k = self.hole
        if self.phase in ("carry", "retrieve_walk") and self.lane_of(f[0], f[1]) != k:
            out = f.copy()
            out[2] = 0.0
            return out
        lane = np.array([11.0, self.hole_y(k), 0.0])
        mid = 0.45 * b + 0.2 * f + 0.35 * lane
        mid[2] = 0.3
        return mid if np.all(np.isfinite(mid)) else f

    def camera_preset(self) -> CameraPreset:
        return CameraPreset(azimuth=50.0, elevation=-33.0, distance=23.0, tau_s=0.8)

    def scorecard_line(self) -> str:
        cells = []
        for k in range(len(HOLES)):
            v = self.card[k]
            cells.append(f"{k + 1}:{'-' if v is None else v}/{self.par(k)}")
        played = [v for v in self.card if v is not None]
        par = sum(self.par(k) for k in range(len(HOLES)) if self.card[k] is not None)
        tot = sum(played)
        rel = tot - par
        return "  ".join(cells) + f"   round {tot} ({'E' if rel == 0 else f'{rel:+d}'})"

    def job_hud_lines(self) -> list[str]:
        rel = self.total_strokes - self.total_par
        return [
            f"hole {self.hole + 1} {HOLES[self.hole][0]} (par {self.par(self.hole)})  stroke {self.strokes}  [{self.phase}]",
            "card " + self.scorecard_line(),
            f"rounds {self.rounds}  last {self.last_round}  best {self.best_round}  total {self.total_strokes} "
            f"vs par {self.total_par} ({'E' if rel == 0 else f'{rel:+d}'})",
            f"holes-in-one {self.holes_in_one}  eagles {self.eagles}  birdies {self.birdies}  pars {self.pars}  "
            f"bogeys {self.bogeys}  OOB {self.oob}  drops {self.drops}",
            self.last_shot,
            "(putt speed/aim: skill model + engineered impulse at the real head touch; "
            "sails / ball carry: kinematic)",
        ]

    def job_stats(self) -> dict:
        return {"hole": self.hole + 1, "holes_played": self.holes_played, "rounds": self.rounds,
                "best_round": self.best_round, "last_round": self.last_round,
                "total_strokes": self.total_strokes, "total_par": self.total_par,
                "holes_in_one": self.holes_in_one, "eagles": self.eagles, "birdies": self.birdies,
                "pars": self.pars, "bogeys": self.bogeys, "putts": self.putts, "penalties": self.penalties,
                "oob": self.oob, "drops": self.drops, "picked_up": self.picked_up,
                "mill_waits": self.mill_waits, "sail_hits": self.sail_hits,
                "unstuck": self.n_unstuck, "card": list(self.card)}
