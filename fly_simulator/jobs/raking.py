"""Raking: leaves fall from a big autumn tree onto a yard; the fly sweeps them into a
pile with a rake; the wind blows the pile away. Forever.

World (the fly spawns at the origin facing +x):

* a yard (``yard_x0..yard_x1`` x ``-yard_half_y..yard_half_y``) with a bare-earth pile
  spot (``pile_xy``, ``pile_radius``);
* a tree just beyond the far (+y) edge: a trunk and an autumn canopy of big
  ellipsoids overhanging the yard (decoration, no contacts; the fly never goes there);
* a fixed pool of ``n_leaves`` leaves: flat ellipsoids in autumn colours on *mocap*
  bodies with visual-only geoms. Leaves are kinematic: the job moves them (falling
  with a flutter, lying on the ground, carried by the rake, heaped on the pile, blown
  by a gust). No leaf physics, so any number of leaves is stable and cheap, and the
  model never grows (leaves are recycled: blown out of the yard -> back into the
  canopy -> fall again);
* the rake: a mocap body (visual only) that the job keeps in front of the fly's
  thorax every update, like a rake welded to the thorax front: a handle from above
  the fly's head down to a comb of tines at ``rake_reach`` in front of it.

Rake -> leaf: every update each leaf on the ground is looked at in the rake frame;
a leaf in the strip just behind the comb's front face (within the comb width) is
pushed to the front face and marked "raked". Leaves carried in front of the comb
therefore go wherever the fly walks, and slide off the comb ends when it turns.

Behaviour: pick a ground leaf (nearest to the fly, weighted by its distance to the
pile) -> walk round it to the staging point behind it relative to the pile (orbit,
like the sisyphus loop) -> sweep: walk to the pile centre; raked leaves that enter the
pile circle are heaped on the pile and counted -> next leaf. With no leaves on the
ground the fly rests on its rake next to the pile.

Eternal part: a leaf falls from the canopy every ``fall_every_s``. When the pile has
``pile_target`` leaves (a completed pile), a gust comes ``gust_delay_s`` later; gusts
also come every ``gust_every_s`` anyway. A gust blows every leaf of the pile (and
some leaves from the ground) up and across the yard; a share of them fly out of the
yard and return to the canopy. Counters: leaves raked (into the pile), piles
completed, gusts survived.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import add_box, quat_axis_angle
from fly_simulator.jobs.mowing import PushPilot
from fly_simulator.jobs.registry import register_job

P = "rake/"

LEAF_COLOURS = (
    (0.86, 0.22, 0.08, 1.0),  # red
    (0.95, 0.45, 0.06, 1.0),  # orange
    (0.98, 0.70, 0.10, 1.0),  # yellow
    (0.62, 0.30, 0.10, 1.0),  # brown
    (0.78, 0.12, 0.10, 1.0),  # crimson
    (0.90, 0.58, 0.20, 1.0),  # amber
)
CANOPY = ((0.90, 0.40, 0.08, 1.0), (0.85, 0.20, 0.08, 1.0), (0.96, 0.66, 0.12, 1.0),
          (0.80, 0.32, 0.06, 1.0))
BARK = (0.35, 0.22, 0.13, 1.0)
YARD = (0.42, 0.52, 0.24, 1.0)
EARTH = (0.45, 0.33, 0.20, 1.0)
WOOD = (0.72, 0.52, 0.30, 1.0)
TINES = (0.20, 0.55, 0.25, 1.0)

# leaf states
TREE, FALL, GROUND, PILE, GUST = 0, 1, 2, 3, 4
STATE_NAMES = ("tree", "falling", "ground", "pile", "gust")


@dataclass
class RakingConfig(JobConfig):
    # ---- yard -------------------------------------------------------------------
    yard_x0: float = -4.0
    yard_x1: float = 18.0
    yard_half_y: float = 8.0
    pile_xy: tuple[float, float] = (13.0, -3.5)
    pile_radius: float = 2.0
    pile_height: float = 2.2  # heap height when complete (mm)
    tree_xy: tuple[float, float] = (6.0, 12.5)
    canopy_z: float = 8.0
    canopy_radius: float = 5.0
    # ---- leaves -------------------------------------------------------------------
    n_leaves: int = 40
    leaf_size: tuple[float, float] = (0.55, 0.36)  # half length / half width (mm)
    leaves_on_ground_at_start: int = 16
    fall_every_s: float = 1.6  # a leaf leaves the canopy this often
    fall_time_s: tuple[float, float] = (2.0, 3.5)
    flutter_mm: float = 1.2
    # ---- wind ---------------------------------------------------------------------
    pile_target: int = 18  # leaves in the pile = a completed pile
    gust_delay_s: float = 3.0  # completed pile -> gust
    gust_every_s: float = 75.0  # a gust at least this often anyway
    gust_time_s: tuple[float, float] = (1.2, 2.2)
    gust_away_frac: float = 0.35  # share of gusted leaves blown out of the yard (-> tree)
    gust_ground_frac: float = 0.3  # share of ground leaves a gust also moves
    # ---- rake ---------------------------------------------------------------------
    rake_reach: float = 2.1  # thorax -> comb front face (mm)
    rake_width: float = 3.0
    rake_depth: float = 1.0  # strip behind the comb face that grabs leaves
    # ---- behaviour -----------------------------------------------------------------
    stage_gap: float = 0.8  # comb this far behind the leaf at the staging point
    stage_radius: float = 0.9  # within this of the staging point -> sweep
    sweep_speed: float = 0.8
    approach_speed: float = 1.0
    retarget_s: float = 2.5
    # waypoints stay within the yard grown by this much (the fly may step off the
    # lawn to get behind a leaf at the edge; there are no walls)
    walk_margin: float = 3.5
    orbit_radius: float = 3.0  # thorax keeps this far from the target leaf when going round
    celebrate_s: float = 1.5  # groom after a completed pile; 0 = off


@register_job
class RakingJob(EternalJob):
    name = "raking"
    title = "LEAF RAKING FLY"
    tagline = "autumn is forever"
    work_label = "leaves raked"
    config_cls = RakingConfig
    required_names = (P + "rake", P + "leaf0")

    cfg: RakingConfig

    # ------------------------------------------------------------ build
    def extension(self, world) -> None:
        c = self.cfg
        root = world.mjcf_root
        wb = root.worldbody
        mat = root.material("grid")
        if mat is not None:
            mat.rgba = (0.36, 0.46, 0.22, 1.0)
        cx, hx = (c.yard_x0 + c.yard_x1) / 2, (c.yard_x1 - c.yard_x0) / 2
        # (top 60 um up: thinner separations from the ground plane z-fight at this
        # camera distance)
        add_box(wb, P + "yard", (hx, c.yard_half_y, 0.03), (cx, 0.0, 0.03), rgba=YARD,
                collide="visual")
        px, py = c.pile_xy
        wb.add_geom(name=P + "pile_spot", type=mj.mjtGeom.mjGEOM_CYLINDER,
                    size=(c.pile_radius, 0.01, 0), pos=(px, py, 0.065), rgba=EARTH,
                    contype=0, conaffinity=0, group=1)
        # the tree (decoration beyond the far edge)
        tx, ty = c.tree_xy
        wb.add_geom(name=P + "trunk", type=mj.mjtGeom.mjGEOM_CYLINDER,
                    size=(0.9, c.canopy_z / 2, 0), pos=(tx, ty, c.canopy_z / 2), rgba=BARK,
                    contype=0, conaffinity=0, group=1)
        for k, (a, b) in enumerate(((-0.5, 0.7), (0.6, 0.9))):
            L = 3.5
            wb.add_geom(name=f"{P}branch{k}", type=mj.mjtGeom.mjGEOM_CAPSULE,
                        size=(0.35, L / 2, 0),
                        pos=(tx + a * L * 0.6, ty - 0.8, c.canopy_z * b),
                        quat=quat_axis_angle((1, 0.4 * a, 0), -0.9 if a < 0 else 0.9),
                        rgba=BARK, contype=0, conaffinity=0, group=1)
        rng = np.random.default_rng(7)
        R = c.canopy_radius
        for k in range(9):
            off = rng.uniform(-1, 1, 3) * np.array([R * 0.6, R * 0.35, R * 0.18])
            r = R * rng.uniform(0.42, 0.6)
            wb.add_geom(name=f"{P}canopy{k}", type=mj.mjtGeom.mjGEOM_ELLIPSOID,
                        size=(r, r * 0.85, r * 0.6),
                        pos=(tx + off[0], ty - 2.0 + off[1], c.canopy_z + off[2]),
                        rgba=CANOPY[k % len(CANOPY)], contype=0, conaffinity=0, group=1)
        # the leaves: a fixed pool of mocap bodies (kinematic, visual only)
        lx, ly = c.leaf_size
        for i in range(c.n_leaves):
            b = wb.add_body(name=f"{P}leaf{i}", mocap=True, pos=(0.0, 30.0 + i, 0.05))
            b.add_geom(name=f"{P}leaf{i}_g", type=mj.mjtGeom.mjGEOM_ELLIPSOID,
                       size=(lx, ly, 0.025), rgba=LEAF_COLOURS[i % len(LEAF_COLOURS)],
                       contype=0, conaffinity=0, group=1)
            b.add_geom(name=f"{P}leaf{i}_stem", type=mj.mjtGeom.mjGEOM_CAPSULE,
                       size=(0.02, 0.12, 0), pos=(-lx - 0.08, 0, 0),
                       quat=quat_axis_angle((0, 1, 0), math.pi / 2),
                       rgba=(0.45, 0.28, 0.12, 1.0), contype=0, conaffinity=0, group=1)
        # the rake (mocap, kept in front of the thorax by the job)
        rake = wb.add_body(name=P + "rake", mocap=True, pos=(0.0, 0.0, 0.0))
        f, W = c.rake_reach, c.rake_width
        vis = dict(contype=0, conaffinity=0, group=1)
        a = np.array([f - 0.35, 0.0, 0.3])  # handle foot at the comb head
        bpt = np.array([0.45, 0.0, 1.75])  # handle top above the fly's head
        mid, d = (a + bpt) / 2, bpt - a
        L = float(np.linalg.norm(d))
        pitch = math.atan2(d[2], -d[0])
        rake.add_geom(name=P + "handle", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.06, L / 2, 0),
                      pos=tuple(mid), quat=quat_axis_angle((0, 1, 0), -(math.pi / 2 - pitch)),
                      rgba=WOOD, **vis)
        rake.add_geom(name=P + "head", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.08, W / 2, 0),
                      pos=(f - 0.35, 0, 0.3), quat=quat_axis_angle((1, 0, 0), math.pi / 2),
                      rgba=TINES, **vis)
        n_t = 11
        for k in range(n_t):
            y = -W / 2 + W * k / (n_t - 1)
            # tines fan from the head bar forward and down to the front face
            rake.add_geom(name=f"{P}tine{k}", type=mj.mjtGeom.mjGEOM_CAPSULE,
                          size=(0.03, 0.2, 0), pos=(f - 0.17, y, 0.17),
                          quat=quat_axis_angle((0, 1, 0), math.radians(135)), rgba=TINES, **vis)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        m = self.sim.model
        c = self.cfg
        self.rake_mocap = int(m.body_mocapid[m.body(P + "rake").id])
        self.leaf_mocap = np.array([int(m.body_mocapid[m.body(f"{P}leaf{i}").id])
                                    for i in range(c.n_leaves)])
        self._rng = np.random.default_rng(c.seed + 11)
        n = c.n_leaves
        self.leaf_state = np.full(n, TREE)
        self.leaf_pos = np.zeros((n, 3))
        self.leaf_yaw = self._rng.uniform(-math.pi, math.pi, n)
        self.leaf_roll = np.zeros(n)
        self.leaf_raked = np.zeros(n, bool)
        self.leaf_jit = self._rng.uniform(0.0, 0.35, n)  # stacking offset in front of the comb
        self.leaf_z = 0.1 + 0.006 * (np.arange(n) % 8)  # no z-fighting on the ground
        # flight (FALL / GUST): start, end, t0, duration, arc height, spin, then -> state
        self.fl_p0 = np.zeros((n, 3))
        self.fl_p1 = np.zeros((n, 3))
        self.fl_t0 = np.zeros(n)
        self.fl_T = np.ones(n)
        self.fl_arc = np.zeros(n)
        self.fl_phase = self._rng.uniform(0, 2 * math.pi, n)
        self.fl_next = np.full(n, GROUND)
        for i in range(n):
            self.leaf_pos[i] = self.canopy_point()
        k0 = min(c.leaves_on_ground_at_start, n)
        for i in range(k0):
            self.leaf_state[i] = GROUND
            self.leaf_pos[i] = (*self.yard_point(), self.leaf_z[i])
        self.leaves_raked = 0
        self.piles = 0
        self.gusts = 0
        self.leaves_blown_away = 0
        self.leaves_fallen = 0
        self.n_pile = 0
        self.target: int | None = None
        self._t_target = -1e9
        self._t_last_fall = 0.0
        self._t_last_gust = 0.0
        self._t_complete: float | None = None
        self._t_gust_msg = -1e9
        self._pile_done = False
        # the target leaf is pushed like a prop of radius 1 mm: the thorax stays
        # rake_reach + stage_gap behind it, the comb's front face touches it
        R = 1.0
        self.pilot = PushPilot(self, R, behind_gap=c.rake_reach + c.stage_gap - R,
                               orbit_clearance=c.orbit_radius - R, lookahead=2.0,
                               align_radius=c.stage_radius, approach_speed=c.approach_speed,
                               push_speed=c.sweep_speed, clamp=self._clamp, unstick=False)
        self.state = "approach"
        self._sync_leaves()
        self._place_rake()

    def on_reset(self) -> None:
        # the keyframe reset put every mocap body back at its spec pose: restore the
        # yard (leaves keep their state; only the fly respawns)
        self.target = None
        self.state = "approach"
        self.pilot.reset()
        t = self.sim.time
        # flights restart from where they are, on the new clock
        fl = (self.leaf_state == FALL) | (self.leaf_state == GUST)
        self.fl_t0[fl] = t
        self.fl_p0[fl] = self.leaf_pos[fl]
        self._t_last_fall = self._t_last_gust = t
        if self._t_complete is not None:
            self._t_complete = t
        self._sync_leaves()
        self._place_rake()

    # ------------------------------------------------------------ geometry helpers
    def _clamp(self, p) -> np.ndarray:
        c = self.cfg
        k = -c.walk_margin
        return np.array([min(max(float(p[0]), c.yard_x0 + k), c.yard_x1 - k),
                         min(max(float(p[1]), -c.yard_half_y + k), c.yard_half_y - k)])

    def _on_yard(self, p, margin: float) -> np.ndarray:
        c = self.cfg
        return np.array([min(max(float(p[0]), c.yard_x0 + margin), c.yard_x1 - margin),
                         min(max(float(p[1]), -c.yard_half_y + margin),
                             c.yard_half_y - margin)])

    def canopy_point(self) -> np.ndarray:
        c = self.cfg
        r = c.canopy_radius * 0.75 * math.sqrt(self._rng.uniform())
        a = self._rng.uniform(0, 2 * math.pi)
        return np.array([c.tree_xy[0] + r * math.cos(a),
                         c.tree_xy[1] - 2.0 + 0.5 * r * math.sin(a),
                         c.canopy_z - 0.5 + self._rng.uniform(-1.0, 1.0)])

    def yard_point(self, margin: float = 1.0) -> tuple[float, float]:
        """A random point on the yard, not on the pile spot."""
        c = self.cfg
        for _ in range(20):
            x = self._rng.uniform(c.yard_x0 + margin, c.yard_x1 - margin)
            y = self._rng.uniform(-c.yard_half_y + margin, c.yard_half_y - margin)
            if math.hypot(x - c.pile_xy[0], y - c.pile_xy[1]) > c.pile_radius + 1.0:
                return x, y
        return c.yard_x0 + margin, 0.0

    def in_pile(self, xy) -> np.ndarray:
        c = self.cfg
        xy = np.atleast_2d(xy)
        return np.hypot(xy[:, 0] - c.pile_xy[0], xy[:, 1] - c.pile_xy[1]) < c.pile_radius

    def rake_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(thorax xy, heading unit vector, left unit vector)."""
        p = self.fly_xy()
        h = self.sim.heading()
        u = np.array([math.cos(h), math.sin(h)])
        return p, u, np.array([-u[1], u[0]])

    def counts(self) -> dict[str, int]:
        return {nm: int(np.count_nonzero(self.leaf_state == k)) for k, nm in
                enumerate(STATE_NAMES)}

    # ------------------------------------------------------------ mocap writes
    def _place_rake(self) -> None:
        d = self.sim.data
        p = self.fly_xy()
        h = self.sim.heading()
        if not (np.all(np.isfinite(p)) and math.isfinite(h)):
            return
        d.mocap_pos[self.rake_mocap] = (p[0], p[1], 0.0)
        d.mocap_quat[self.rake_mocap] = (math.cos(h / 2), 0.0, 0.0, math.sin(h / 2))

    def _sync_leaves(self) -> None:
        d = self.sim.data
        d.mocap_pos[self.leaf_mocap] = self.leaf_pos
        cy, sy = np.cos(self.leaf_yaw / 2), np.sin(self.leaf_yaw / 2)
        cr, sr = np.cos(self.leaf_roll / 2), np.sin(self.leaf_roll / 2)
        d.mocap_quat[self.leaf_mocap] = np.column_stack([cy * cr, cy * sr, sy * sr, sy * cr])

    # ------------------------------------------------------------ leaf events
    def _launch(self, i: int, p1, T: float, arc: float, then: int, state: int) -> None:
        self.fl_p0[i] = self.leaf_pos[i]
        self.fl_p1[i] = p1
        self.fl_t0[i] = self.sim.time
        self.fl_T[i] = T
        self.fl_arc[i] = arc
        self.fl_next[i] = then
        self.leaf_state[i] = state
        self.leaf_raked[i] = False

    def drop_leaf(self) -> bool:
        """One canopy leaf starts to fall (onto a random spot of the yard)."""
        idx = np.flatnonzero(self.leaf_state == TREE)
        if len(idx) == 0:
            return False
        i = int(self._rng.choice(idx))
        c = self.cfg
        x, y = self.yard_point()
        self._launch(i, (x, y, self.leaf_z[i]), float(self._rng.uniform(*c.fall_time_s)), 0.0,
                     GROUND, FALL)
        self.leaves_fallen += 1
        return True

    def gust(self) -> None:
        """Wind: the pile (and some ground leaves) blow up and across the yard."""
        c = self.cfg
        a = float(self._rng.uniform(-math.pi, math.pi))
        wind = np.array([math.cos(a), math.sin(a)])
        movers = np.flatnonzero(self.leaf_state == PILE)
        ground = np.flatnonzero(self.leaf_state == GROUND)
        if len(ground):
            movers = np.concatenate([movers, ground[self._rng.uniform(size=len(ground))
                                                    < c.gust_ground_frac]])
        for i in movers:
            T = float(self._rng.uniform(*c.gust_time_s))
            if self._rng.uniform() < c.gust_away_frac:
                p = self.leaf_pos[i, :2] + wind * self._rng.uniform(25, 35)
                self._launch(i, (p[0], p[1], 4.0), T, float(self._rng.uniform(3, 6)), TREE, GUST)
                self.leaves_blown_away += 1
            else:
                x, y = self.yard_point()
                p1 = np.array([x, y]) + wind * self._rng.uniform(0, 3)
                p1 = self._on_yard(p1, 0.8)
                self._launch(i, (p1[0], p1[1], self.leaf_z[i]), T,
                             float(self._rng.uniform(1.5, 4.0)), GROUND, GUST)
        self.n_pile = 0
        self.gusts += 1
        self._pile_done = False
        self._t_complete = None
        self._t_last_gust = self.sim.time
        self._t_gust_msg = self.sim.time
        self.target = None
        self.say(f"WIND GUST #{self.gusts}: {len(movers)} leaves blown about "
                 f"({self.leaves_raked} raked so far)")

    def _fly_leaves(self, t: float) -> None:
        idx = np.flatnonzero((self.leaf_state == FALL) | (self.leaf_state == GUST))
        if len(idx) == 0:
            return
        c = self.cfg
        s = np.clip((t - self.fl_t0[idx]) / self.fl_T[idx], 0.0, 1.0)
        p0, p1 = self.fl_p0[idx], self.fl_p1[idx]
        ph = self.fl_phase[idx]
        falling = self.leaf_state[idx] == FALL
        # falling: drift down with a side-to-side flutter; gust: arc up and over
        e = np.where(falling, s, 1 - (1 - s) ** 2)
        pos = p0 + (p1 - p0) * e[:, None]
        sway = np.where(falling, c.flutter_mm * np.sin(2 * math.pi * 0.8 * (t - self.fl_t0[idx])
                                                          + ph) * np.sin(math.pi * s), 0.0)
        pos[:, 0] += sway * np.cos(ph)
        pos[:, 1] += sway * np.sin(ph)
        pos[:, 2] += self.fl_arc[idx] * np.sin(math.pi * s)
        self.leaf_pos[idx] = pos
        self.leaf_roll[idx] = np.where(falling, 0.9 * np.sin(2 * math.pi * 1.6 * t + ph),
                                       6 * math.pi * s + ph) * (1 - s)
        self.leaf_yaw[idx] += 0.02
        for j in np.flatnonzero(s >= 1.0):
            i = idx[j]
            nxt = self.fl_next[i]
            self.leaf_roll[i] = 0.0
            if nxt == TREE:
                self.leaf_pos[i] = self.canopy_point()
            else:
                self.leaf_pos[i, 2] = self.leaf_z[i]
            self.leaf_state[i] = nxt

    def _rake_leaves(self) -> None:
        """Push the ground leaves the comb meets; heap raked leaves on the pile."""
        c = self.cfg
        g = np.flatnonzero(self.leaf_state == GROUND)
        if len(g) == 0:
            return
        p, u, n = self.rake_frame()
        rel = self.leaf_pos[g, :2] - p
        a = rel @ u
        lat = rel @ n
        face = c.rake_reach
        hit = (np.abs(lat) < c.rake_width / 2) & (a > face - c.rake_depth) & (a < face + 0.02)
        if np.any(hit) and not self.fly_down():
            gi = g[hit]
            new_a = face + 0.03 + self.leaf_jit[gi]
            self.leaf_pos[gi, :2] = p + np.outer(new_a, u) + np.outer(lat[hit], n)
            self.leaf_raked[gi] = True
            self.leaf_yaw[gi] += 0.01 * np.sign(lat[hit] + 1e-9)
        # keep leaves on the yard
        lim_x0, lim_x1 = c.yard_x0 + 0.3, c.yard_x1 - 0.3
        self.leaf_pos[g, 0] = np.clip(self.leaf_pos[g, 0], lim_x0, lim_x1)
        self.leaf_pos[g, 1] = np.clip(self.leaf_pos[g, 1], -c.yard_half_y + 0.3,
                                      c.yard_half_y - 0.3)
        into = g[self.in_pile(self.leaf_pos[g, :2])]
        for i in into:
            self._heap(int(i))

    def _heap(self, i: int) -> None:
        """Leaf ``i`` joins the pile (counted if it was raked in)."""
        c = self.cfg
        k = self.n_pile
        frac = min(1.0, (k + 1) / c.pile_target)
        r = c.pile_radius * 0.75 * math.sqrt(self._rng.uniform())
        a = self._rng.uniform(0, 2 * math.pi)
        z = 0.12 + c.pile_height * frac * (1 - (r / c.pile_radius) ** 2)
        self.leaf_pos[i] = (c.pile_xy[0] + r * math.cos(a), c.pile_xy[1] + r * math.sin(a), z)
        self.leaf_roll[i] = self._rng.uniform(-0.9, 0.9)
        self.leaf_yaw[i] = self._rng.uniform(-math.pi, math.pi)
        self.leaf_state[i] = PILE
        self.n_pile += 1
        if self.leaf_raked[i]:
            self.leaves_raked += 1
            self.add_work(1)
        if self.target == i:
            self.target = None
        if not self._pile_done and self.n_pile >= c.pile_target:
            self._pile_done = True
            self.piles += 1
            self._t_complete = self.sim.time
            self.say(f"PILE #{self.piles} complete ({self.n_pile} leaves). "
                     f"The wind is picking up...")
            if c.celebrate_s > 0 and not self.session.actions.busy:
                from fly_simulator.actions import make_action

                self.session.actions.trigger(make_action("groom", duration=c.celebrate_s),
                                             source="job")

    # ------------------------------------------------------------ behaviour
    def _pick_target(self) -> int | None:
        c = self.cfg
        g = np.flatnonzero(self.leaf_state == GROUND)
        if len(g) == 0:
            return None
        p = self.fly_xy()
        pile = np.asarray(c.pile_xy)
        xy = self.leaf_pos[g, :2]
        cost = np.linalg.norm(xy - p, axis=1) + 0.5 * np.linalg.norm(xy - pile, axis=1)
        k = int(np.argmin(cost))
        cur = self.target
        if cur is not None and self.leaf_state[cur] == GROUND:
            (j,) = np.flatnonzero(g == cur)
            if cost[k] > 0.7 * cost[j]:  # hysteresis: keep the current leaf
                return cur
        return int(g[k])

    def update(self) -> None:
        c = self.cfg
        t = self.sim.time
        self._place_rake()
        # the tree drops leaves; the wind blows
        if t - self._t_last_fall >= c.fall_every_s:
            self._t_last_fall = t
            self.drop_leaf()
        if ((self._t_complete is not None and t - self._t_complete >= c.gust_delay_s)
                or t - self._t_last_gust >= c.gust_every_s):
            self.gust()
        self._fly_leaves(t)
        self._rake_leaves()
        self._sync_leaves()
        if self.session.actions.busy:
            self.steering.set(None, 1.0)
            return
        self._steer(t)

    def _steer(self, t: float) -> None:
        c = self.cfg
        steer = self.steering
        pile = np.asarray(c.pile_xy, float)
        if (self.target is None or self.leaf_state[self.target] != GROUND
                or (self.pilot.state == "approach" and t - self._t_target > c.retarget_s)):
            new = self._pick_target()
            if new != self.target:
                self.pilot.reset()
            self.target = new
            self._t_target = t
        if self.target is None:
            # nothing on the ground: lean on the rake beside the pile
            rest = pile + np.array([-c.pile_radius - 3.0, 0.0])
            if steer.aim_at(rest, 0.8) < 0.8:
                steer.set(0.0, 0.0)
            self.state = "rest"
            return
        # the leaf is the "prop": get behind it (relative to the pile) and push it
        # there with the comb, exactly like the sisyphus boulder
        st = self.pilot.step(self.leaf_pos[self.target, :2], pile)
        self.state = {"push": "sweep"}.get(st, st)

    # ------------------------------------------------------------ view / HUD
    def camera_target(self) -> np.ndarray:
        c = self.cfg
        f = self.sim.thorax_position()
        yard = np.array([(c.yard_x0 + c.yard_x1) / 2, 1.5, 3.0])
        out = 0.5 * f + 0.5 * yard
        return out if np.all(np.isfinite(out)) else f

    def camera_preset(self) -> CameraPreset:
        return CameraPreset(azimuth=90.0, elevation=-24.0, distance=28.0, tau_s=1.2)

    def job_hud_lines(self) -> list[str]:
        k = self.counts()
        lines = [f"piles completed {self.piles}   gusts survived {self.gusts}   "
                 f"pile {self.n_pile}/{self.cfg.pile_target}",
                 f"leaves: {k['ground']} on the lawn, {k['falling']} falling, "
                 f"{k['tree']} still on the tree"]
        if self.sim.time - self._t_gust_msg < 3.0:
            lines.append("~~~ WIND GUST! ~~~")
        return lines

    def job_stats(self) -> dict:
        k = self.counts()
        return {"leaves_raked": self.leaves_raked, "piles_completed": self.piles,
                "gusts_survived": self.gusts, "leaves_fallen": self.leaves_fallen,
                "blown_away": self.leaves_blown_away, "in_pile": self.n_pile,
                "on_ground": k["ground"], "unstuck": self.n_unstuck}
