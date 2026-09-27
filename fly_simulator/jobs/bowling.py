"""Bowling: the fly bowls forever. League night, every night.

World (along +x = down the lane; the fly spawns at the origin facing +x, on the
approach):

* a raised lane bed (``bed_height``) the fly walks on: the approach, the black foul
  line at ``foul_x``, then the lane with its target arrows and guide dots, and the pin
  deck with the pin spots (``bowling_assets.lane_texture``: maple / pine boards).
  Semicircular gutters run along both sides of the lane (static facets, a real
  channel the ball drops into), kickbacks flank the deck, and the pit behind it has a
  "carpet" floor (rolling friction) and a back cushion that stop ball and pins;
* 10 pins in the standard triangle: free bodies (collision: a flat foot cylinder, a
  belly capsule, a neck capsule, a head sphere; visual: a lathe mesh through the
  regulation pin profile, white with two red neck stripes). They collide with the
  lane, the ball and each other, never with the fly;
* the ball: a free sphere (glossy swirled resin, three finger holes), rolling with
  condim-6 rolling friction; the fly touches it with head / thorax / abdomen only
  at low friction (``slippery_body_contact``, legs excluded).

Behaviour: the fly walks to the ball on the approach and lines it up (``PushPilot``
from the mowing job, pushing along the roll's target line), pushes it into the
funnel of the ramp guide and over the top edge of a bowling ramp (**engineered**: a
walking-speed ball can't topple pins; the ramp's 4 mm drop gives it ~236 mm/s) and
stops there, behind the foul line (**stop rule, engineered**: once the ball is over
the edge the job sets the walking speed to 0 and freezes the fly; a thorax past the
line counts as a foul and the roll scores 0, as in real bowling). The ball rolls
down the lane on its own momentum and the pins scatter with real contact physics.
When everything has settled, the job counts the pins that are down (tilted more
than ``down_tilt_deg`` or off the deck) and scores the roll (standard ten-pin
scoring, ``BowlingGame``). The roll is shown in slow motion (an **edit effect**,
labelled on screen) and the camera cuts like a TV broadcast (see ``camera_preset``).

Engineered, labelled parts (a real bowling alley has machines for these too):

* **pinsetter** (kinematic, like the real machine): after roll 1 it lifts the
  standing pins, the sweep bar clears the deck (knocked pins vanish into the machine
  as the bar passes them: a teleport into the hidden pinsetter housing), and puts the
  standing pins back down where they stood, upright. After a frame (or a strike) it
  clears everything and lowers a fresh rack of 10 onto the spots. Every cycle and
  every rack reset is counted.
* **ball return**: the ball is taken out of the pit (teleport) into the ball-return
  hood beside the approach and pops out at the approach return spot.
* **aim + ramp guide**: each roll picks a target line from a documented "skill" model
  (normal aim error at the pins, occasional wild throws) and the ramp guide (two
  ball-only rails with a funnel, like the rails of a kids' bowling ramp that a
  helper aims) is set to it. The physics decides what happens next.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from fly_simulator.jobs import bowling_assets as A
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig, _make_action
from fly_simulator.jobs.geometry import (
    PROP_BIT,
    add_box,
    add_plane_box,
    add_slope,
    contact_kwargs,
    quat_axis_angle,
    slippery_body_contact,
)
from fly_simulator.jobs.mowing import PushPilot
from fly_simulator.jobs.registry import register_job
from fly_simulator.terrain import TERRAIN_BIT

P = "bowl/"
BALL_BIT = 64  # contact bit of the ball-only rolling surfaces (lane oil, dry approach)
N_PINS = 10


# ---------------------------------------------------------------------------
# scoring (pure logic, unit tested)
# ---------------------------------------------------------------------------


class BowlingGame:
    """One game of standard ten-pin scoring: 10 frames; a frame is a strike (all 10
    with the first ball) or two balls (a spare if they total 10); strike = 10 + the
    next two balls, spare = 10 + the next ball; the 10th frame gives the bonus balls
    (a strike or spare there gets a fresh rack and a 3rd ball). Max 300."""

    def __init__(self) -> None:
        self.rolls: list[int] = []
        self.frames: list[list[int]] = [[] for _ in range(10)]
        self.marks: list[list[str]] = [[] for _ in range(10)]
        self.frame = 0  # current frame index (0..9)
        self.standing = 10  # pins standing on the current rack
        self.fresh_rack = True  # the next ball is the first on this rack
        self.over = False

    @property
    def ball_in_frame(self) -> int:
        return len(self.frames[self.frame]) if not self.over else 0

    def roll(self, pins: int) -> dict:
        """Record a ball that knocked down ``pins`` (clipped to the pins standing).
        Returns the events: strike, spare, frame_done, reset_rack, game_over."""
        if self.over:
            raise ValueError("game over")
        s = self.standing
        n = int(min(max(pins, 0), s))
        f = self.frame
        fr = self.frames[f]
        fr.append(n)
        self.rolls.append(n)
        strike = self.fresh_rack and n == 10
        spare = not self.fresh_rack and n == s
        self.fresh_rack = False
        mark = "X" if strike else "/" if spare else "-" if n == 0 else str(n)
        self.marks[f].append(mark)
        ev = dict(pins=n, strike=strike, spare=spare, frame_done=False, reset_rack=False,
                  game_over=False, frame=f + 1, ball=len(fr))
        if f < 9:
            if strike or len(fr) == 2:
                ev["frame_done"] = ev["reset_rack"] = True
                self.frame += 1
                self.standing = 10
                self.fresh_rack = True
            else:
                self.standing = s - n
        else:
            k = len(fr)
            if strike or spare:
                self.standing = 10
                self.fresh_rack = True
                ev["reset_rack"] = True
            else:
                self.standing = s - n
            if k == 3 or (k == 2 and fr[0] + fr[1] < 10):
                self.over = True
                ev["frame_done"] = ev["game_over"] = ev["reset_rack"] = True
                self.standing = 10
                self.fresh_rack = True
        return ev

    def totals(self) -> list[int | None]:
        """Cumulative score after each frame (None while a bonus is still pending)."""
        out: list[int | None] = []
        r = self.rolls
        i, cum = 0, 0
        for f in range(10):
            if i >= len(r):
                out.append(None)
                continue
            if f == 9:
                fr = self.frames[9]
                need = 3 if (fr and (fr[0] == 10 or sum(fr[:2]) == 10)) else 2
                if len(fr) < need:
                    out.append(None)
                else:
                    cum += sum(fr)
                    out.append(cum)
                continue
            if r[i] == 10:
                if i + 2 < len(r):
                    cum += 10 + r[i + 1] + r[i + 2]
                    out.append(cum)
                else:
                    out.append(None)
                    cum = None if cum is None else cum
                i += 1
            elif i + 1 < len(r):
                if r[i] + r[i + 1] == 10:
                    if i + 2 < len(r):
                        cum += 10 + r[i + 2]
                        out.append(cum)
                    else:
                        out.append(None)
                else:
                    cum += r[i] + r[i + 1]
                    out.append(cum)
                i += 2
            else:
                out.append(None)
                i += 1
        # a frame after a pending one can't have a total either
        seen_none = False
        for k, v in enumerate(out):
            if v is None:
                seen_none = True
            elif seen_none:
                out[k] = None
        return out

    def score(self) -> int:
        """Score so far (the last known cumulative total; bonus balls pending count 0)."""
        known = [t for t in self.totals() if t is not None]
        return known[-1] if known else 0

    def final_score(self) -> int:
        t = self.totals()
        return int(t[9]) if self.over and t[9] is not None else self.score()


def score_rolls(rolls) -> int:
    """Final score of a complete game given as a list of balls (tests / docs)."""
    g = BowlingGame()
    for n in rolls:
        g.roll(n)
    return g.final_score()


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class BowlingConfig(JobConfig):
    # ---- ball -------------------------------------------------------------------
    ball_radius: float = 1.5  # mm (3 mm ball; the fly is ~2.5 mm long)
    ball_mass: float = 3e-4  # g (0.3 mg, like the sisyphus boulder)
    # rolling resistance (condim-6, length units): a polished, oiled lane. At 1e-4
    # (FlyGym's default) the ball loses ~0.5 mm/s^2
    ball_rolling: float = 1e-4
    pit_rolling: float = 0.25  # the pit "carpet" stops the ball
    # the approach (dry, unpolished wood) rolls harder than the oiled lane: a
    # ball-only overlay with this rolling friction keeps the pushed ball at the
    # fly's head instead of being batted ahead by every head bump (engineered,
    # labelled: stand-in for the dry approach)
    approach_rolling: float = 0.02
    # the bowling ramp (a real device, used by kids; engineered, labelled): the
    # approach is a plateau that ends in a ramp dropping ramp_height over
    # ramp_length onto the lane at the foul line. The fly pushes the ball over its
    # top edge and the ball rolls down, reaching ~sqrt(10/7 g h) = 236 mm/s (a
    # rolling ball, h = 4 mm). A ball pushed at walking speed (~10 mm/s) only nudges
    # a pin (measured: the pin leans on the ball and the ball stops dead), and the
    # first, 0.35 mm ramp (70 mm/s) left pins tottering: 2 pins per roll, no strikes
    # in 180 s. Placed-ball sweeps (1.5 s, 6 lines y -2.4 .. 0): 70 mm/s 3.8 pins,
    # 150 mm/s 6.0, 200 mm/s 7.0 (a strike in the pocket), 320 mm/s 8.2
    ramp_height: float = 4.0
    ramp_length: float = 10.0
    head_friction: float = 0.05
    # ramp guide (engineered, labelled: the rails of a kids' / adaptive bowling ramp,
    # which a helper aims before each ball): two low chrome rails run down the ramp
    # with a funnel on the plateau, set each roll to the aim model's release line
    # (kinematic, moved only while the ball is away). They touch only the ball.
    # Without it the fly's head bumps sent the ball over the edge 1-2 mm off line and
    # drifting sideways at up to 13 mm/s: 2-3 mm (sd) off at the pins, 3.7 pins per roll
    guide: bool = True
    guide_gap: float = 0.15  # clearance each side of the ball in the channel
    guide_funnel: float = 3.5  # funnel length on the plateau (mm)
    guide_flare: float = 3.0  # the funnel's mouth is this much wider each side
    # ---- pins -------------------------------------------------------------------
    pin_height: float = 5.2  # mm (regulation 15 in, scaled like the ball: 8.5 in -> 3 mm)
    pin_mass: float = 6.7e-5  # g: the regulation ball / pin mass ratio (~4.5)
    pin_spacing: float = 4.2  # mm between spots (regulation 12 in, same scale)
    pin_friction: float = 0.25
    pin_solref: float = 1e-3  # contact time constant of the pins (s)
    # damping ratio of the pin contacts: real pins bounce off each other and the ball
    # (lacquered maple, restitution ~0.6); MuJoCo's default 1 (critical) makes them
    # clay-like. 0.3 added ~0.5 pins per roll in the placed-ball sweep
    pin_dampratio: float = 0.3
    down_tilt_deg: float = 35.0  # a pin tilted more than this (or off the deck) is down
    # ---- lane -------------------------------------------------------------------
    bed_height: float = 1.6  # lane bed above the floor (the gutters are this deep)
    lane_half_width: float = 7.25  # (regulation 41.5 in, scaled)
    gutter_radius: float = 1.6
    # the back of the approach plateau (5.6 mm above the floor: a fly that walks off
    # it falls; the first 10 mm plateau was too short for the fly's turns at the
    # waiting spot)
    approach_x0: float = -16.0
    foul_x: float = 19.0
    # foul line -> head pin (regulation 60 ft would be 250 mm at this scale; the lane
    # is compressed ~11x so a fly-pushed ball gets there)
    lane_length: float = 22.0
    pit_length: float = 6.5
    # ---- behaviour ----------------------------------------------------------------
    # aim model (the helper who aims the ramp guide): a line straight down the lane.
    # Roll 1 aims at the pocket (between pins 1 and 3), roll 2 at the standing pins
    # (weighted to the front ones). Aim error at the pins ~ N(0, aim_sd), aim_sd =
    # 0.4 + 2.0 (1 - skill) mm, and with probability wild_prob (1 - skill) a wild
    # throw (+ N(0, wild_sd)), which can go into a gutter.
    skill: float = 0.9
    # the pocket: placed-ball sweeps at 236 mm/s struck at y -1.0 and -0.7 (8 pins
    # at -1.3 and -0.4)
    pocket_y: float = -0.85
    wild_prob: float = 0.5
    wild_sd: float = 5.0
    # release point y = frac * aim y (+ N(0, 0.3 mm)): the ball leaves the ramp
    # going straight down the lane (the ramp guide keeps it on its line), so
    # where it goes over the edge is where it meets the pins
    release_board_frac: float = 1.0
    push_speed: float = 0.75
    # the last runup_mm before the edge: a gentle nudge (a faster run-up batted the
    # ball sideways: up to 13 mm/s across at the release, 2-4 mm off at the pins)
    runup_speed: float = 0.5
    runup_mm: float = 5.0
    approach_speed: float = 0.9
    behind_gap: float = 1.25
    orbit_clearance: float = 1.5
    pursuit: float = 3.0  # push goal: the target line this far ahead of the ball (mm)
    # release: the ball's centre stop_over mm past the ramp's top edge (gravity
    # takes it from there); the fly then stops (a thorax past foul_x = foul)
    stop_over: float = 0.3
    # the fly never walks closer than this to the ramp's top edge (thorax x)
    edge_margin: float = 0.8
    # ball-return spot on the approach (8 mm behind the edge: room to steer the
    # ball onto a line near the lane edge before the guide's funnel)
    return_x: float = 1.0
    return_y_jitter: float = 1.5
    wait_x: float = -4.5  # the fly waits here for the ball
    # ---- pinsetter / timing -----------------------------------------------------
    settle_speed: float = 2.0  # pins / ball below this (mm/s) ...
    settle_hold_s: float = 0.5  # ... for this long = settled
    settle_max_s: float = 5.0
    roll_timeout_s: float = 15.0
    stall_speed: float = 0.8  # ball slower than this on the lane for stall_s = dead ball
    stall_s: float = 1.2
    store_z: float = 8.4  # pins in the pinsetter: base this far above the bed (hidden)
    lift_mm: float = 2.6  # the pinsetter lifts standing pins this much for the sweep
    # ---- presentation -------------------------------------------------------------
    # slow motion (an *edit effect*, not physics: fewer physics steps per displayed
    # frame, the step sequence is unchanged; labelled "SLOW MOTION" on screen) while
    # the ball is on the ramp / lane and for the first slowmo_settle_s of the pin
    # action; it eases back to real time over the next 0.5 s. 1 = off. (At 236 mm/s
    # the ball crosses the 22 mm lane in ~0.1 s: a single frame at 10 fps)
    slowmo: float = 0.25
    slowmo_settle_s: float = 0.45
    shadows: bool = True
    stuck_timeout_s: float = 90.0


# ---------------------------------------------------------------------------
# the job
# ---------------------------------------------------------------------------


def _quat_z_to(d) -> tuple[float, float, float, float]:
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    ax = np.cross((0.0, 0.0, 1.0), d)
    s = float(np.linalg.norm(ax))
    if s < 1e-9:
        return (1.0, 0.0, 0.0, 0.0) if d[2] > 0 else (0.0, 1.0, 0.0, 0.0)
    return quat_axis_angle(ax, math.atan2(s, float(d[2])))


@register_job
class BowlingJob(EternalJob):
    name = "bowling"
    #: depth precision (FlyGym's 5e-4 gave shadow acne on the fly, ball and lane);
    #: applied by EternalJob.attach
    znear = 0.05
    title = "BOWLING FLY"
    tagline = "league night, every night"
    work_label = "pins knocked down"
    config_cls = BowlingConfig
    required_names = (P + "ball", P + "pin0", P + "lane")

    cfg: BowlingConfig

    # ------------------------------------------------------------ geometry
    @property
    def head_x(self) -> float:
        return self.cfg.foul_x + self.cfg.lane_length

    @property
    def row_dx(self) -> float:
        return self.cfg.pin_spacing * math.sqrt(3) / 2

    @property
    def deck_x0(self) -> float:
        return self.head_x - 1.8

    @property
    def deck_end(self) -> float:
        return self.head_x + 3 * self.row_dx + 1.8

    @property
    def pit_end(self) -> float:
        return self.deck_end + self.cfg.pit_length

    @property
    def gutter_y(self) -> float:
        return self.cfg.lane_half_width + self.cfg.gutter_radius

    @property
    def cap_y(self) -> float:  # outer edge of the gutter
        return self.cfg.lane_half_width + 2 * self.cfg.gutter_radius

    @property
    def lane_pitch(self) -> float:
        return 2 * (self.cap_y + 0.6)

    @property
    def ramp_x0(self) -> float:
        return self.cfg.foul_x - self.cfg.ramp_length

    @property
    def approach_z(self) -> float:
        return self.cfg.bed_height + self.cfg.ramp_height

    @property
    def stop_x(self) -> float:
        """Thorax x at which the fly stops pushing (the ball is then over the edge)."""
        c = self.cfg
        return self.ramp_x0 + c.stop_over - (c.ball_radius + c.behind_gap)

    def pin_spots(self) -> np.ndarray:
        """(10, 2) spots, standard numbering: 1 head pin; 2 (left, +y), 3; 4, 5, 6;
        7, 8, 9, 10 from left (+y) to right (-y), seen from the bowler."""
        s, dx = self.cfg.pin_spacing, self.row_dx
        out = []
        for row in range(4):
            for k in range(row + 1):
                out.append((self.head_x + row * dx, (row / 2 - k) * s))
        return np.array(out)

    def ground_height(self, x: float, y: float) -> float:
        c = self.cfg
        if c.approach_x0 <= x <= self.ramp_x0 and abs(y) <= self.lane_pitch / 2:
            return self.approach_z
        if self.ramp_x0 < x <= c.foul_x and abs(y) <= self.lane_pitch / 2:
            return c.bed_height + c.ramp_height * (c.foul_x - x) / c.ramp_length
        if c.foul_x < x <= self.deck_end and abs(y) <= c.lane_half_width:
            return c.bed_height
        return 0.0

    def configure_app(self, app_cfg) -> None:
        super().configure_app(app_cfg)
        app_cfg.fly.spawn_height += self.approach_z
        app_cfg.controller.target_heading_deg = 0.0

    # ------------------------------------------------------------ build
    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        wb = spec.worldbody
        vis = contact_kwargs("visual")
        vis_m = dict(vis, mass=0.0)
        bed, hw, Rg = c.bed_height, c.lane_half_width, c.gutter_radius
        x0, xf, xd, xe, xp = c.approach_x0, c.foul_x, self.deck_x0, self.deck_end, self.pit_end
        spots = self.pin_spots()
        self._add_materials(spec)
        # ---- floor: cosmic carpet
        mat = spec.material("grid")
        if mat is not None:
            A.add_texture(spec, P + "tex_carpet", A.carpet_texture(c.seed))
            mat.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = P + "tex_carpet"
            mat.rgba = (1.0, 1.0, 1.0, 1.0)
            mat.reflectance = 0.0
        # ---- approach (full width between the neighbouring lanes) and lane bed
        pw = self.lane_pitch / 2
        az, xr = self.approach_z, self.ramp_x0
        add_box(wb, P + "approach", ((xr - x0) / 2, pw, az / 2), ((xr + x0) / 2, 0, az / 2),
                material=P + "bed_side", collide="static")
        ramp_a = math.atan2(c.ramp_height, c.ramp_length)
        add_slope(wb, P + "ramp", xr, xf, az, -ramp_a, pw, thickness=az, material=P + "bed_side",
                  collide="static")
        add_box(wb, P + "lane", ((xe - xf) / 2, hw, bed / 2), ((xe + xf) / 2, 0, bed / 2),
                material=P + "bed_side", collide="static")
        if c.guide:
            self._add_guide(wb, ramp_a)
        # the polished top (visual slab mesh carrying the lane texture)
        # (three pieces: approach, ramp, lane; the texture runs over all three)
        L = xe - x0
        tops = (("top_a", x0, xr, az, az), ("top_r", xr, xf, az, bed), ("top_l", xf, xe, bed, bed))
        for nm, xa, xb, za, zb in tops:
            A.add_mesh(spec, P + nm + "_mesh",
                       A.slab_mesh(xa, xb, -hw, hw, za - 0.02, za + 0.03, zb + 0.03,
                                   (xa - x0) / L, (xb - x0) / L))
            wb.add_geom(name=P + ("lane_top" if nm == "top_l" else nm), type=mj.mjtGeom.mjGEOM_MESH,
                        meshname=P + nm + "_mesh", material=P + "lane_wood", **vis_m)
            # approach wings beside the lane-width strip (plain maple)
            if nm != "top_l":
                for side, sgn in (("l", 1.0), ("r", -1.0)):
                    A.add_mesh(spec, f"{P}{nm}_wing_{side}_mesh",
                               A.slab_mesh(xa, xb, sgn * hw, sgn * pw, za - 0.02, za + 0.03, zb + 0.03))
                    wb.add_geom(name=f"{P}{nm}_wing_{side}", type=mj.mjtGeom.mjGEOM_MESH,
                                meshname=f"{P}{nm}_wing_{side}_mesh", material=P + "maple", **vis_m)
        # ---- gutters: semicircular channels (static facets; visual half-pipe)
        nf = 7
        for side, sgn in (("l", 1.0), ("r", -1.0)):
            yc = sgn * self.gutter_y
            a = np.linspace(math.pi, 2 * math.pi, nf + 1)
            gz = bed - 0.08  # (lips just below the lane top: no ridge for the ball to hit)
            for k in range(nf):
                p0 = np.array([yc + Rg * math.cos(a[k]), gz + Rg * math.sin(a[k])])
                p1 = np.array([yc + Rg * math.cos(a[k + 1]), gz + Rg * math.sin(a[k + 1])])
                mid = (p0 + p1) / 2
                ch = float(np.linalg.norm(p1 - p0))
                nrm = np.array([yc, gz]) - mid
                nrm /= np.linalg.norm(nrm)
                add_plane_box(wb, f"{P}gutter_{side}{k}", ((xf + xe) / 2, mid[0], mid[1]),
                              (1, 0, 0), (0.0, nrm[0], nrm[1]), (xe - xf) / 2, ch / 2 + 0.02,
                              thickness=0.3, rgba=(0.3, 0.3, 0.32, 0.0), collide="static",
                              group=3)
            gm = A.gutter_mesh(xf, xe, Rg + 0.01)
            A.add_mesh(spec, f"{P}gutter_mesh_{side}",
                       A.MeshData(gm.verts + np.array([0.0, yc, bed]), gm.faces, gm.uv))
            wb.add_geom(name=f"{P}gutter_vis_{side}", type=mj.mjtGeom.mjGEOM_MESH,
                        meshname=f"{P}gutter_mesh_{side}", material=P + "gutter", **vis_m)
            # capping between this lane and the next
            add_box(wb, f"{P}capping_{side}", ((xe - xf) / 2, 0.3, (bed + 0.25) / 2),
                    ((xf + xe) / 2, sgn * (self.cap_y + 0.3), (bed + 0.25) / 2),
                    material=P + "capping", collide="static", friction=0.3)
            # kickbacks along the deck and pit
            kx0 = self.head_x - 3.0
            add_box(wb, f"{P}kickback_{side}", ((xp - kx0) / 2, 0.3, (bed + 7.6) / 2),
                    ((xp + kx0) / 2, sgn * (self.cap_y + 0.3), (bed + 7.6) / 2),
                    material=P + "kickback", collide="static", friction=0.3)
        # ---- pit: carpet floor (props only, rolling friction) + back cushion
        from flygym.compose import ContactParams

        cp = ContactParams()
        wb.add_geom(name=P + "pit_floor", type=mj.mjtGeom.mjGEOM_BOX,
                    size=((xp - xe) / 2, self.cap_y + 0.6, 0.05), pos=((xe + xp) / 2, 0.0, 0.004 - 0.05),
                    rgba=(0.06, 0.06, 0.07, 1.0), contype=TERRAIN_BIT, conaffinity=0, priority=2,
                    condim=6, friction=(1.0, 0.02, c.pit_rolling), solref=cp.get_solref_tuple(),
                    solimp=cp.get_solimp_tuple(), margin=cp.margin)
        # ball-only rolling surfaces (contype BALL_BIT: only the ball touches them, not
        # the fly or the pins): priority 2 gives the ball's contact condim 6 (rolling
        # friction) here while ball-pin contacts stay condim 3 (with the noslip
        # solver, rolling constraints between ball and pin weld them together)
        def ball_surface(name, x_0, x_1, half_y, rolling, z=bed):
            wb.add_geom(name=name, type=mj.mjtGeom.mjGEOM_BOX, size=((x_1 - x_0) / 2, half_y, 0.05),
                        pos=((x_1 + x_0) / 2, 0.0, z + 0.004 - 0.05), rgba=(0, 0, 0, 0), group=3,
                        contype=BALL_BIT, conaffinity=0, priority=2, condim=6,
                        friction=(1.0, 0.02, rolling), solref=cp.get_solref_tuple(),
                        solimp=cp.get_solimp_tuple(), margin=cp.margin)

        ball_surface(P + "approach_dry", x0, xr, pw, c.approach_rolling, az)
        ball_surface(P + "lane_oil", xf, xe, hw, c.ball_rolling)
        add_box(wb, P + "cushion", (0.4, self.cap_y + 0.6, (bed + 7.6) / 2),
                (xp + 0.4, 0.0, (bed + 7.6) / 2), rgba=(0.04, 0.04, 0.05, 1.0), collide="static",
                friction=0.5)
        # ---- pinsetter housing + masking unit (visual; hides the stored pins)
        hz0, hz1 = bed + c.store_z - 0.6, bed + c.store_z + c.pin_height + 1.2
        hx0 = self.head_x - 2.6
        add_box(wb, P + "housing", ((xp - hx0) / 2, self.cap_y + 0.6, (hz1 - hz0) / 2),
                ((xp + hx0) / 2, 0.0, (hz0 + hz1) / 2), material=P + "housing", collide="visual")
        mask_h = hz1 - hz0 + 1.5
        # the panel faces the bowler (-x); u runs from +y to -y so the text reads
        mz0 = hz0 - 0.4
        Y = self.lane_pitch * 1.5
        slab = A.slab_mesh(mz0, mz0 + mask_h, Y, -Y, 0.0, 0.1)
        v = slab.verts
        slab = A.MeshData(np.column_stack([hx0 - 0.1 - v[:, 2], v[:, 1], v[:, 0]]), slab.faces,
                          slab.uv)
        if slab.signed_volume() < 0:
            slab.faces = slab.faces[:, ::-1].copy()
        A.add_mesh(spec, P + "masking_mesh", slab)
        wb.add_geom(name=P + "masking", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "masking_mesh",
                    material=P + "masking", **vis_m)
        # sweep bar (mocap, visual): parked inside the housing
        bar = wb.add_body(name=P + "sweep", mocap=True, pos=self._bar_park())
        bar.add_geom(name=P + "sweep_bar", type=mj.mjtGeom.mjGEOM_BOX, size=(0.18, hw + 0.4, 0.45),
                     pos=(0, 0, 0), material=P + "sweep", **vis_m)
        for sgn in (1.0, -1.0):
            bar.add_geom(name=f"{P}sweep_arm{int(sgn)}", type=mj.mjtGeom.mjGEOM_BOX,
                         size=(0.1, 0.1, 3.5), pos=(0, sgn * (hw + 0.3), 3.5), material=P + "sweep",
                         **vis_m)
        # ---- ball return hood beside the approach (visual) + a couple of house balls
        hx, hy = c.return_x, pw
        add_box(wb, P + "return_base", (2.2, 1.6, 0.5), (hx, hy, az + 0.5), material=P + "chrome",
                collide="visual")
        wb.add_geom(name=P + "return_hood", type=mj.mjtGeom.mjGEOM_ELLIPSOID, size=(2.3, 1.8, 2.5),
                    pos=(hx, hy, az + 0.9), material=P + "hood", **vis_m)
        for k, (dx, col) in enumerate(((-3.6, P + "house_ball_g"), (3.6, P + "house_ball_o"))):
            wb.add_geom(name=f"{P}house_ball{k}", type=mj.mjtGeom.mjGEOM_SPHERE, size=(c.ball_radius, 0, 0),
                        pos=(hx + dx, hy, az + 1.0 + c.ball_radius), material=col, **vis_m)
        add_box(wb, P + "return_rail", (5.0, 0.8, 0.5), (hx, hy, az + 0.5), material=P + "chrome",
                collide="visual")
        # ---- neighbouring lanes (decoration: lane top, gutters, a full rack)
        A.add_mesh(spec, P + "pin_mesh", A.pin_mesh(c.pin_height))
        for j, off in enumerate((self.lane_pitch, -self.lane_pitch)):
            for nm in ("top_a", "top_r", "top_l"):
                wb.add_geom(name=f"{P}nb_{nm}{j}", type=mj.mjtGeom.mjGEOM_MESH,
                            meshname=P + nm + "_mesh", pos=(0.0, off, 0.0), material=P + "lane_wood",
                            **vis_m)
            add_box(wb, f"{P}nb_bed{j}", ((xe - xf) / 2, hw, bed / 2), ((xe + xf) / 2, off, bed / 2),
                    material=P + "bed_side", collide="visual")
            # their approach plateaus and ramps (visual)
            add_box(wb, f"{P}nb_approach{j}", ((xr - x0) / 2, pw, az / 2), ((xr + x0) / 2, off, az / 2),
                    material=P + "bed_side", collide="visual")
            add_slope(wb, f"{P}nb_ramp{j}", xr, xf, az, -ramp_a, pw, y=off, thickness=az,
                      material=P + "bed_side", collide="visual")
            for side, sgn in (("l", 1.0), ("r", -1.0)):
                for nm in ("top_a", "top_r"):
                    wb.add_geom(name=f"{P}nb_{nm}_wing_{side}{j}", type=mj.mjtGeom.mjGEOM_MESH,
                                meshname=f"{P}{nm}_wing_{side}_mesh", pos=(0.0, off, 0.0),
                                material=P + "maple", **vis_m)
            for side, sgn in (("l", 1.0), ("r", -1.0)):
                wb.add_geom(name=f"{P}nb_gutter{j}{side}", type=mj.mjtGeom.mjGEOM_MESH,
                            meshname=f"{P}gutter_mesh_{side}", pos=(0.0, off, 0.0),
                            material=P + "gutter", **vis_m)
            for i, (sx, sy) in enumerate(spots):
                wb.add_geom(name=f"{P}nb_pin{j}_{i}", type=mj.mjtGeom.mjGEOM_MESH,
                            meshname=P + "pin_mesh", pos=(sx, sy + off, bed), material=P + "pin",
                            **vis_m)
        # ---- the pins (free bodies)
        H = c.pin_height
        pr, pz = A.pin_profile(H)
        m = c.pin_mass
        pkw = contact_kwargs("dynamic", c.pin_friction)
        pkw["conaffinity"] = TERRAIN_BIT | PROP_BIT  # not the fly
        # priority 2: the pins' (lacquered, low) friction governs ball-pin, pin-pin and
        # pin-deck contacts (with the default max-mixing the ball's friction 1 won: its
        # spin then pinned the light pin to the deck and the ball stopped dead)
        pkw["priority"] = 2
        # softer contacts for the light pins (timeconst 1 ms instead of FlyGym's 0.2 ms,
        # which is at the 0.1 ms step's stability limit: fast pin-pin hits blew up)
        pkw["solref"] = (c.pin_solref, c.pin_dampratio)
        for i, (sx, sy) in enumerate(spots):
            b = wb.add_body(name=f"{P}pin{i}", pos=(sx, sy, bed + 0.002))
            b.add_freejoint(name=f"{P}pin{i}_free")
            hid = dict(rgba=(1, 1, 1, 0), group=3)
            r_foot = float(pr[0])
            b.add_geom(name=f"{P}pin{i}_foot", type=mj.mjtGeom.mjGEOM_CYLINDER,
                       size=(r_foot, 0.12 * H / 2, 0), pos=(0, 0, 0.06 * H), mass=0.2 * m, **hid, **pkw)
            r_belly = float(pr.max()) * 0.97
            zb0, zb1 = 0.30 * H, 0.40 * H
            b.add_geom(name=f"{P}pin{i}_belly", type=mj.mjtGeom.mjGEOM_CAPSULE,
                       size=(r_belly, (zb1 - zb0) / 2, 0), pos=(0, 0, (zb0 + zb1) / 2),
                       mass=0.6 * m, **hid, **pkw)
            b.add_geom(name=f"{P}pin{i}_neck", type=mj.mjtGeom.mjGEOM_CAPSULE,
                       size=(0.9 * H / 15, 0.10 * H, 0),
                       pos=(0, 0, 0.66 * H), mass=0.12 * m, **hid, **pkw)
            b.add_geom(name=f"{P}pin{i}_head", type=mj.mjtGeom.mjGEOM_SPHERE,
                       size=(2.44 / 2 * H / 15, 0, 0), pos=(0, 0, 0.885 * H), mass=0.08 * m,
                       **hid, **pkw)
            b.add_geom(name=f"{P}pin{i}_vis", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "pin_mesh",
                       material=P + "pin", **vis_m)
        # ---- the ball
        R = c.ball_radius
        ball = wb.add_body(name=P + "ball", pos=(3.5, 0.0, az + R + 0.01))
        ball.add_freejoint(name=P + "ball_free")
        bkw = contact_kwargs("dynamic", 1.0)
        bkw["conaffinity"] |= BALL_BIT
        ball.add_geom(name=P + "ball_geom", type=mj.mjtGeom.mjGEOM_SPHERE, size=(R, 0, 0),
                      mass=c.ball_mass, rgba=(1, 1, 1, 0), group=3, **bkw)
        A.add_mesh(spec, P + "ball_mesh", A.ball_mesh(R * 1.002))
        ball.add_geom(name=P + "ball_vis", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "ball_mesh",
                      material=P + "ball", **vis_m)
        # three finger holes (thumb lower), dark discs flush with the surface
        for k, (th, ph, r) in enumerate(((0.0, 0.62, 0.17), (0.32, 0.95, 0.15), (-0.32, 0.95, 0.15))):
            d = np.array([math.cos(ph) * math.cos(th), math.cos(ph) * math.sin(th), math.sin(ph)])
            ball.add_geom(name=f"{P}ball_hole{k}", type=mj.mjtGeom.mjGEOM_CYLINDER,
                          size=(r, 0.03, 0), pos=tuple(d * (R * 1.002 - 0.02)), quat=_quat_z_to(d),
                          rgba=(0.03, 0.02, 0.03, 1.0), **vis_m)
        slippery_body_contact(spec, P + "ball", P + "ball_geom", self.fly_name,
                              friction=c.head_friction)
        self._add_lights(spec)

    def _add_guide(self, wb, ramp_a: float) -> None:
        """The aimed ramp guide: a mocap body (moved in y) with two rails down the
        ramp and a funnel on the plateau; ball-only contacts (contype BALL_BIT)."""
        c = self.cfg
        R = c.ball_radius
        g = wb.add_body(name=P + "guide", mocap=True, pos=(0.0, 0.0, 0.0))
        rail = dict(contype=BALL_BIT, conaffinity=0, priority=2, condim=3,
                    friction=(0.05, 0.005, 0.0001), material=P + "chrome", mass=0.0)
        hh, hw_r = 0.5, 0.1  # rail half height / half width; top 1.4 mm up (the ball's
        # equator is 1.5 mm up: the rails touch it just below)
        yin = R + c.guide_gap + hw_r
        xr, xf, az = self.ramp_x0, c.foul_x, self.approach_z
        run = (xf - 0.8) - xr
        L = run / math.cos(ramp_a)
        zc = az - 0.5 * run * math.tan(ramp_a)
        for sgn in (1.0, -1.0):
            nrm = np.array([math.sin(ramp_a), 0.0, math.cos(ramp_a)])
            ctr = np.array([xr + run / 2, sgn * yin, zc]) + nrm * (0.9)
            g.add_geom(name=f"{P}guide_rail{int(sgn)}", type=mj.mjtGeom.mjGEOM_BOX,
                       size=(L / 2, hw_r, hh), pos=tuple(ctr), quat=quat_axis_angle((0, 1, 0), ramp_a),
                       **rail)
            # the funnel on the plateau, flaring back from the edge
            fl = c.guide_funnel
            y0, y1 = sgn * yin, sgn * (yin + c.guide_flare)
            ln = math.hypot(fl, y1 - y0)
            yaw = math.atan2(y1 - y0, -fl)
            q = quat_axis_angle((0, 0, 1), yaw)
            g.add_geom(name=f"{P}guide_funnel{int(sgn)}", type=mj.mjtGeom.mjGEOM_BOX,
                       size=(ln / 2 + 0.05, hw_r, hh), pos=(xr - fl / 2 + 0.05, (y0 + y1) / 2, az + 0.9),
                       quat=q, **rail)

    def _set_guide(self) -> None:
        if getattr(self, "guide_mocap", -1) >= 0:
            self.sim.data.mocap_pos[self.guide_mocap] = (0.0, self.release_y, 0.0)

    def _bar_park(self) -> tuple[float, float, float]:
        return (self.head_x - 1.6, 0.0, self.cfg.bed_height + self.cfg.store_z + 2.0)

    def _add_materials(self, spec) -> None:
        c = self.cfg
        spots = tuple(tuple(float(v) for v in p) for p in self.pin_spots())
        lane = A.lane_texture(2 * c.lane_half_width, c.approach_x0, self.deck_end, c.foul_x,
                              self.head_x, self.deck_x0, spots, c.foul_x + 0.27 * c.lane_length,
                              (c.foul_x + 0.11 * c.lane_length,),
                              (self.ramp_x0 - 2.0, self.ramp_x0 - 5.5), seed=c.seed)
        A.add_texture(spec, P + "tex_lane", lane)
        A.add_textured_material(spec, P + "lane_wood", P + "tex_lane", rgba=(1, 1, 1, 1),
                                specular=0.9, shininess=0.9, reflectance=0.12)
        spec.add_material(name=P + "maple", rgba=(0.78, 0.60, 0.38, 1), specular=0.5, shininess=0.6)
        spec.add_material(name=P + "bed_side", rgba=(0.45, 0.30, 0.18, 1), specular=0.2)
        spec.add_material(name=P + "gutter", rgba=(0.22, 0.23, 0.26, 1), specular=0.8, shininess=0.8)
        spec.add_material(name=P + "capping", rgba=(0.12, 0.12, 0.14, 1), specular=0.5)
        spec.add_material(name=P + "kickback", rgba=(0.16, 0.10, 0.22, 1), specular=0.3)
        spec.add_material(name=P + "housing", rgba=(0.08, 0.06, 0.12, 1), specular=0.1)
        spec.add_material(name=P + "sweep", rgba=(0.85, 0.85, 0.88, 1), specular=0.9, shininess=0.9)
        spec.add_material(name=P + "chrome", rgba=(0.62, 0.63, 0.68, 1), specular=1.0, shininess=0.95)
        spec.add_material(name=P + "hood", rgba=(0.75, 0.10, 0.12, 1), specular=0.8, shininess=0.9)
        spec.add_material(name=P + "house_ball_g", rgba=(0.10, 0.55, 0.25, 1), specular=1.0, shininess=1.0)
        spec.add_material(name=P + "house_ball_o", rgba=(0.95, 0.45, 0.05, 1), specular=1.0, shininess=1.0)
        A.add_texture(spec, P + "tex_pin", A.pin_texture(c.seed))
        A.add_textured_material(spec, P + "pin", P + "tex_pin", rgba=(1, 1, 1, 1), specular=0.8,
                                shininess=0.85)
        A.add_texture(spec, P + "tex_ball", A.ball_texture(c.seed))
        A.add_textured_material(spec, P + "ball", P + "tex_ball", rgba=(1, 1, 1, 1), specular=1.0,
                                shininess=1.0, reflectance=0.2)
        A.add_texture(spec, P + "tex_masking", A.masking_texture())
        A.add_textured_material(spec, P + "masking", P + "tex_masking", rgba=(1, 1, 1, 1),
                                emission=0.6, specular=0.2)

    def _add_lights(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        spec.visual.headlight.ambient = (0.30, 0.30, 0.32)
        spec.visual.headlight.diffuse = (0.35, 0.35, 0.36)
        spec.visual.headlight.specular = (0.15, 0.15, 0.15)
        bed = c.bed_height
        a_pos = np.array([self.ramp_x0 - 11.0, -6.0, self.approach_z + 20.0])
        a_tgt = np.array([self.ramp_x0 - 3.5, 0.0, self.approach_z])
        wb.add_light(name=P + "approach_light", type=mj.mjtLightType.mjLIGHT_SPOT, pos=tuple(a_pos),
                     dir=tuple(a_tgt - a_pos), diffuse=(0.85, 0.82, 0.76), specular=(0.4, 0.4, 0.4),
                     cutoff=32.0, exponent=0.5, castshadow=bool(c.shadows))
        d_tgt = np.array([self.head_x + 5.0, 0.0, bed])
        d_pos = np.array([self.head_x - 6.0, -3.0, bed + 16.0])
        wb.add_light(name=P + "deck_light", type=mj.mjtLightType.mjLIGHT_SPOT, pos=tuple(d_pos),
                     dir=tuple(d_tgt - d_pos), diffuse=(0.75, 0.72, 0.66), specular=(0.5, 0.5, 0.5),
                     cutoff=40.0, exponent=1.0, castshadow=False)
        m_pos = np.array([c.foul_x + 6.0, 0.0, 18.0])
        m_tgt = np.array([c.foul_x + 12.0, 0.0, bed])
        wb.add_light(name=P + "lane_light", type=mj.mjtLightType.mjLIGHT_SPOT, pos=tuple(m_pos),
                     dir=tuple(m_tgt - m_pos), diffuse=(0.35, 0.33, 0.40), specular=(0.3, 0.3, 0.3),
                     cutoff=50.0, exponent=0.5, castshadow=False)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        m = self.sim.model
        c = self.cfg
        self.ball_body = m.body(P + "ball").id
        j = m.joint(P + "ball_free").id
        self.ball_q = int(m.jnt_qposadr[j])
        self.ball_v = int(m.jnt_dofadr[j])
        self.pin_body = np.array([m.body(f"{P}pin{i}").id for i in range(N_PINS)])
        self.pin_q = np.array([int(m.jnt_qposadr[m.joint(f"{P}pin{i}_free").id]) for i in range(N_PINS)])
        self.pin_v = np.array([int(m.jnt_dofadr[m.joint(f"{P}pin{i}_free").id]) for i in range(N_PINS)])
        self.bar_mocap = int(m.body_mocapid[m.body(P + "sweep").id])
        self.guide_mocap = int(m.body_mocapid[m.body(P + "guide").id]) if c.guide else -1
        self.spots = self.pin_spots()
        self._rng = np.random.default_rng(c.seed)
        # game state and counters
        self.game = BowlingGame()
        self.games_completed = 0
        self.best_game = 0
        self.last_game = None
        self.total_game_pins = 0  # sum of completed game scores (average)
        self.n_rolls = 0
        self.n_strikes = 0
        self.n_spares = 0
        self.n_gutters = 0
        self.n_fouls = 0
        self.n_dead_balls = 0
        self.n_no_roll = 0  # "releases" after which the ball stayed on the ramp
        self.n_voided = 0
        self.n_pinsetter = 0  # pinsetter cycles (every roll)
        self.n_rack_resets = 0  # full racks of 10 lowered
        self.n_ball_returns = 0
        self.n_balls_lost = 0
        self.pins_total = 0
        self.streak = 0  # strikes in a row
        self.best_streak = 0
        self.max_pins_one_roll = 0
        self.message = ""
        self._msg_t = -1e9
        self.last_roll = None  # dict of the last scored roll
        # pins: standing on the rack (the game's view), physical mode per pin
        self.rack = np.ones(N_PINS, bool)
        self.pin_hold: list[np.ndarray | None] = [None] * N_PINS  # kinematic target pose
        self.pilot = PushPilot(self, c.ball_radius, behind_gap=c.behind_gap,
                               orbit_clearance=c.orbit_clearance, push_speed=c.push_speed,
                               approach_speed=c.approach_speed, clamp=self._clamp)
        self._cam = None
        self._celebrate = False
        self._last_board = None
        self._reset_behaviour(first=True)

    def _clamp(self, p) -> np.ndarray:
        c = self.cfg
        lim = self.lane_pitch / 2 - 2.0
        return np.array([min(max(float(p[0]), c.approach_x0 + 1.5), self.ramp_x0 - 1.5),
                         min(max(float(p[1]), -lim), lim)])

    def _reset_behaviour(self, first: bool = False) -> None:
        self.phase = "fetch"
        self.state = "fetch"
        self._t_phase = self.run_time()
        self.fly_mode = "pilot"
        self.pilot.reset()
        self._settle_since = None
        self._stall_since = None
        self._ball_hidden = False
        self._roll_foul = False
        self._roll_gutter = False
        self._arrive_y = None
        self._sweep = None
        self._new_aim()

    def on_reset(self) -> None:
        # a reset during a roll voids it (the rack is restored as it was before it)
        if self.phase in ("rolling", "settle"):
            self.n_voided += 1
            self._say_msg("reset mid-roll: ball voided, re-roll")
        elif self.phase == "sweep" and self._sweep is not None:
            self._finish_sweep_state()
        self._reset_behaviour()

    def reset_props(self) -> None:
        """FlyGym's keyframe reset put every pin on its spot and the ball in front of
        the fly: put the rack back as the game has it (knocked pins in the machine)."""
        d = self.sim.data
        for i in range(N_PINS):
            if self.rack[i]:
                self._place_pin(i, (*self.spots[i], self.cfg.bed_height + 0.002))
                self.pin_hold[i] = None
            else:
                self._store_pin(i)
        self._park_bar()
        mj.mj_forward(self.sim.model, d)

    # ------------------------------------------------------------ props
    def ball_pos(self) -> np.ndarray:
        return self.sim.data.xpos[self.ball_body]

    def ball_vel(self) -> np.ndarray:
        return self.sim.data.qvel[self.ball_v:self.ball_v + 3]

    def _place_pin(self, i: int, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        d = self.sim.data
        q = self.pin_q[i]
        d.qpos[q:q + 3] = pos
        d.qpos[q + 3:q + 7] = quat
        d.qvel[self.pin_v[i]:self.pin_v[i] + 6] = 0.0

    def _store_pos(self, i: int) -> np.ndarray:
        return np.array([*self.spots[i], self.cfg.bed_height + self.cfg.store_z])

    def _store_pin(self, i: int) -> None:
        p = self._store_pos(i)
        self.pin_hold[i] = p
        self._place_pin(i, p)

    def _park_bar(self) -> None:
        self.sim.data.mocap_pos[self.bar_mocap] = self._bar_park()

    def pin_pose(self, i: int) -> tuple[np.ndarray, float]:
        """(base position, tilt from vertical in degrees)."""
        d = self.sim.data
        b = self.pin_body[i]
        up_z = float(d.xmat[b, 8])
        return d.xpos[b].copy(), math.degrees(math.acos(min(max(up_z, -1.0), 1.0)))

    def pin_down(self, i: int) -> bool:
        c = self.cfg
        p, tilt = self.pin_pose(i)
        if not np.all(np.isfinite(p)):
            return True
        return (tilt > c.down_tilt_deg or p[2] < c.bed_height - 0.3
                or p[0] > self.deck_end + 0.1 or p[0] < self.deck_x0 - 3.0
                or abs(p[1]) > c.lane_half_width + 0.1)

    def pins_moving(self) -> float:
        """Largest speed (linear + 0.5 mm x angular) of a free, (nearly) upright pin on
        the deck (mm/s)."""
        d = self.sim.data
        vmax = 0.0
        hw = self.cfg.lane_half_width
        for i in range(N_PINS):
            p = d.xpos[self.pin_body[i]]
            # pins rolling about in the gutters / pit, or lying down and rolling
            # anywhere, can't change the count
            if (self.pin_hold[i] is None and abs(p[1]) <= hw + 0.5 and p[0] <= self.deck_end
                    and self.pin_pose(i)[1] < self.cfg.down_tilt_deg + 15.0):
                v = d.qvel[self.pin_v[i]:self.pin_v[i] + 3]
                w = d.qvel[self.pin_v[i] + 3:self.pin_v[i] + 6]
                s = float(np.linalg.norm(v)) + 0.5 * float(np.linalg.norm(w))
                vmax = max(vmax, s if np.isfinite(s) else 1e9)
        return vmax

    def _hold_props(self) -> None:
        for i in range(N_PINS):
            if self.pin_hold[i] is not None:
                self._place_pin(i, self.pin_hold[i])
        if self._ball_hidden:
            self._place_ball(self._ball_store())

    def _ball_store(self) -> np.ndarray:
        c = self.cfg
        return np.array([c.return_x, self.lane_pitch / 2, self.approach_z + c.ball_radius + 0.01])

    def _place_ball(self, pos) -> None:
        d = self.sim.data
        d.qpos[self.ball_q:self.ball_q + 7] = (*pos, 1.0, 0.0, 0.0, 0.0)
        d.qvel[self.ball_v:self.ball_v + 6] = 0.0

    def ball_lost(self) -> bool:
        p = self.ball_pos()
        if not np.all(np.isfinite(p)):
            return True
        c = self.cfg
        return (p[2] < -1.0 or abs(p[1]) > self.lane_pitch or p[0] < c.approach_x0 - 4.0
                or p[0] > self.pit_end + 3.0)

    # ------------------------------------------------------------ aim
    def _new_aim(self) -> None:
        c = self.cfg
        rng = self._rng
        skill = min(max(c.skill, 0.0), 1.0)
        if self.game.fresh_rack:
            target = c.pocket_y
        else:
            # spare: aim at the standing pins, weighted to the front ones (the key
            # pin; a lone pin: straight at it)
            idx = np.flatnonzero(self.rack)
            if len(idx):
                x = self.spots[idx, 0]
                w = np.exp(-(x - x.min()) / 3.0)
                target = float(np.sum(w * self.spots[idx, 1]) / np.sum(w))
            else:
                target = 0.0
        sd = 0.4 + 2.0 * (1.0 - skill)
        aim = target + rng.normal(0.0, sd)
        self.wild = bool(rng.random() < c.wild_prob * (1.0 - skill))
        if self.wild:
            aim += rng.normal(0.0, c.wild_sd)
        aim = float(np.clip(aim, -c.lane_half_width - 1.5, c.lane_half_width + 1.5))
        lim = c.lane_half_width - 1.0
        rel = float(np.clip(c.release_board_frac * aim + rng.normal(0.0, 0.3), -lim, lim))
        if c.guide:
            # the guide can be aimed past the lane edge (a wild throw: into the gutter)
            lim = c.lane_half_width + 1.0
            rel = float(np.clip(c.release_board_frac * aim + rng.normal(0.0, 0.3), -lim, lim))
        self.aim_y, self.release_y = aim, rel
        if self.phase != "rolling" and getattr(self, "sim", None) is not None:
            self._set_guide()

    def aim_line_y(self, x: float) -> float:
        c = self.cfg
        return self.release_y + (x - c.foul_x) * (self.aim_y - self.release_y) / c.lane_length

    # ------------------------------------------------------------ behaviour
    def update(self) -> None:
        c = self.cfg
        self._hold_props()
        if self.phase != "returning" and not self._ball_hidden and self.ball_lost():
            self.n_balls_lost += 1
            if self.phase in ("fetch",):
                self._ball_to_return_spot()
                return
            self._ball_hidden = True  # a lost ball mid-roll ends the roll
            self._place_ball(self._ball_store())
        t = self.run_time()
        ph = self.phase
        if ph == "fetch":
            self._fetch(t)
        elif ph in ("rolling", "settle"):
            self._watch_roll(t)
        elif ph == "sweep":
            self._run_sweep(t)
        elif ph == "returning":
            if self._fly_clear_of_return() and t - self._t_phase > 0.6:
                self._ball_to_return_spot()
        if ph != "fetch":
            self._fly_idle(t)
        self._edge_guard()

    def _edge_guard(self) -> None:
        """Keep the fly on the plateau: near its back / side edges, walk back in."""
        c = self.cfg
        if self.session.actions.busy or self.fly_mode == "watch":
            return
        f = self.fly_xy()
        if f[0] < c.approach_x0 + 3.0 or abs(f[1]) > self.lane_pitch / 2 - 2.5:
            self.pilot.aim_at(np.array([c.wait_x, 0.0]), 0.6)

    def _fetch(self, t: float) -> None:
        c = self.cfg
        if self.session.actions.busy:
            self.steering.set(None, 1.0)
            return
        b = self.ball_pos()[:2].copy()
        fly = self.fly_xy()
        # the ball went over the ramp's edge: it's a roll. (Only the ball decides:
        # the first version also released when the *fly* reached stop_x, and a ball
        # that had lagged 0.1 mm behind the edge sat on the dry approach, the fly
        # frozen behind it, for the whole 15 s roll timeout)
        if b[0] > self.ramp_x0 + c.stop_over:
            self._release(t)
            return
        gx = max(b[0] + c.pursuit, self.ramp_x0 + 1.0)
        goal = np.array([gx, self.aim_line_y(gx)])
        near = self.stop_x - fly[0] < c.runup_mm
        self.pilot.push_speed = c.runup_speed if near else c.push_speed
        self.state = self.pilot.step(b, goal)
        if fly[0] > self.ramp_x0 - c.edge_margin:
            # never onto the ramp: go round again (the approach waypoints are clamped
            # behind the edge)
            self.pilot.reset()
            self.steering.set(math.pi, 0.6)

    def _release(self, t: float) -> None:
        self.phase = "rolling"
        self.state = "rolling"
        self._t_phase = t
        self._t_release = t
        self._stall_since = None
        self._settle_since = None
        self._roll_foul = False
        self._roll_gutter = False
        self._rack_before = self.rack.copy()
        self._arrive_y = None
        self.fly_mode = "watch"
        self.steering.set(self.sim.heading(), 0.0)
        self.session.actions.trigger(_make_action("freeze", duration=30.0), source="job")

    def _watch_roll(self, t: float) -> None:
        c = self.cfg
        fly = self.fly_xy()
        if fly[0] > c.foul_x and t - self._t_release < 3.0 and not self._roll_foul:
            self._roll_foul = True
            self.n_fouls += 1
            self._say_msg("FOUL! (over the line: the ball scores 0)")
        b = self.ball_pos()
        v = self.ball_vel()
        speed = float(np.hypot(v[0], v[1])) if np.all(np.isfinite(v)) else 0.0
        if self._arrive_y is None and not self._ball_hidden and b[0] > self.head_x - 2.0:
            self._arrive_y = float(b[1])  # where the ball meets the rack (aim vs result)
        if (not self._roll_gutter and not self._ball_hidden and abs(b[1]) > c.lane_half_width
                and b[2] < c.bed_height + c.ball_radius - 0.3 and b[0] < self.head_x):
            self._roll_gutter = True
        if (self.phase == "rolling" and b[0] < c.foul_x and speed < c.stall_speed
                and t - self._t_release > 2.0 and not self._ball_hidden):
            # the ball never went down the ramp (it stopped on the edge): not a roll,
            # the fly goes and pushes it again
            self.n_no_roll += 1
            self.say("the ball stayed on the ramp: push again")
            if self.session.actions.busy:
                self.session.actions.cancel()
            self.phase = self.state = "fetch"
            self._t_phase = t
            self.fly_mode = "pilot"
            self.pilot.reset()
            return
        if self.phase == "rolling":
            # done: in the pit, or in a gutter beside the deck (a ball bouncing off
            # the kickbacks may roll back up the gutter; the roll is over anyway)
            done = (self._ball_hidden or b[0] > self.deck_end + 0.3
                    or (b[0] > self.deck_x0 - 2.0 and abs(b[1]) > c.lane_half_width + 0.5))
            if b[0] > c.foul_x + 1.0 and speed < c.stall_speed and not done:
                if self._stall_since is None:
                    self._stall_since = t
                elif t - self._stall_since > c.stall_s:
                    done = True
                    self.n_dead_balls += 1
            else:
                self._stall_since = None
            if t - self._t_release > c.roll_timeout_s:
                done = True
            if done:
                self.phase = self.state = "settle"
                self._t_phase = t
                self._settle_since = None
        else:  # settle: wait for the pins (and a ball among them) to stop moving
            moving = self.pins_moving()
            if b[0] < self.deck_end and not self._ball_hidden:
                moving = max(moving, speed)
            if moving < c.settle_speed:
                if self._settle_since is None:
                    self._settle_since = t
            else:
                self._settle_since = None
            if ((self._settle_since is not None and t - self._settle_since >= c.settle_hold_s)
                    or t - self._t_phase > c.settle_max_s):
                self._score_roll(t)

    def _score_roll(self, t: float) -> None:
        c = self.cfg
        before = self._rack_before
        down = np.array([before[i] and self.pin_down(i) for i in range(N_PINS)])
        n = int(down.sum())
        self.rack = before & ~down
        knocked = 0 if self._roll_foul else n
        if self._roll_foul:
            self.rack = before.copy()  # foul: pins knocked down don't count and are re-set
        standing_after = [i + 1 for i in range(N_PINS) if self.rack[i]]
        ev = self.game.roll(knocked)
        self.n_rolls += 1
        self.n_pinsetter += 1
        self.pins_total += ev["pins"]
        self.add_work(ev["pins"])
        self.work = self.pins_total
        self.max_pins_one_roll = max(self.max_pins_one_roll, ev["pins"])
        gutter = self._roll_gutter and ev["pins"] == 0
        if gutter:
            self.n_gutters += 1
        if ev["strike"]:
            self.n_strikes += 1
            self.streak += 1
            self.best_streak = max(self.best_streak, self.streak)
            names = {2: "  DOUBLE!", 3: "  TURKEY!", 4: "  HAMBONE! (4 in a row)"}
            self._say_msg("STRIKE!" + names.get(self.streak, f"  {self.streak} IN A ROW"
                                                if self.streak > 4 else ""))
        else:
            self.streak = 0
            if ev["spare"]:
                self.n_spares += 1
                self._say_msg("SPARE!")
            elif gutter:
                self._say_msg("GUTTER BALL")
            elif not self._roll_foul:
                split = ("  (the 7-10 split!)" if standing_after == [7, 10] and ev["ball"] == 1
                         and self.game.frames[ev["frame"] - 1][:1] == [ev["pins"]] else "")
                self._say_msg(f"{ev['pins']} pin{'s' if ev['pins'] != 1 else ''}{split}")
        self.last_roll = dict(ev, gutter=gutter, foul=self._roll_foul, standing=standing_after,
                              aim_y=self.aim_y, arrive_y=self._arrive_y)
        # the terminal gets the marks as events (the HUD is off by default)
        loud = ev["strike"] or ev["spare"] or gutter or self._roll_foul or "split" in self.message
        self.say(("*** " + self.message + " ***  " if loud else "")
                 + f"frame {ev['frame']} ball {ev['ball']}: {ev['pins']} pins"
                 f"{' STRIKE' if ev['strike'] else ' SPARE' if ev['spare'] else ''}"
                 f"{' gutter' if gutter else ''}{' FOUL' if self._roll_foul else ''}"
                 f"  score {self.game.score()}"
                 + (f"  (aim y {self.aim_y:+.1f} mm, ball at the rack {self._arrive_y:+.1f})"
                    if self._arrive_y is not None else ""))
        reset_rack = ev["reset_rack"]
        if ev["game_over"]:
            s = self.game.final_score()
            self.games_completed += 1
            self.total_game_pins += s
            self.last_game = s
            self.best_game = max(self.best_game, s)
            self._say_msg(f"GAME {self.games_completed} OVER: {s}"
                          + ("  (new best!)" if s == self.best_game else ""))
            self.say(f"GAME #{self.games_completed}: {s} (best {self.best_game})")
            self._last_board = (self.game.marks, self.game.totals())
            self.game = BowlingGame()
        if ev["strike"]:
            self._celebrate = True  # the fly grooms (rubs its hands) on the way back
        self._start_sweep(t, reset_rack)

    # ------------------------------------------------------------ pinsetter
    def _start_sweep(self, t: float, full: bool) -> None:
        """Pinsetter cycle (kinematic, like the real machine)."""
        c = self.cfg
        lift = [] if full else [i for i in range(N_PINS) if self.rack[i]]
        lift_from = {}
        for i in lift:
            p, _ = self.pin_pose(i)
            lift_from[i] = np.array([p[0], p[1], c.bed_height + 0.002])
        self._sweep = dict(t0=t, full=full, lift=lift, lift_from=lift_from)
        self.phase = self.state = "sweep"
        self._t_phase = t
        # the ball goes into the ball return (hidden in the hood beside the approach)
        self._ball_hidden = True
        self._place_ball(self._ball_store())

    # timeline (s from the start of the cycle)
    T_LIFT, T_DOWN, T_SWEEP, T_LOWER = 0.5, 0.45, 0.9, 0.75

    def _run_sweep(self, t: float) -> None:
        c = self.cfg
        sw = self._sweep
        s = t - sw["t0"]
        bed = c.bed_height
        t1 = self.T_LIFT
        t2 = t1 + self.T_DOWN
        t3 = t2 + self.T_SWEEP
        t4 = t3 + self.T_LOWER
        # standing pins: lifted by the table (roll 1 of a frame)
        for i in sw["lift"]:
            p0 = sw["lift_from"][i]
            if s < t1:
                h = c.lift_mm * self._ease(s / t1)
            elif s < t3:
                h = c.lift_mm
            else:
                h = c.lift_mm * (1.0 - self._ease((s - t3) / self.T_LOWER))
            self.pin_hold[i] = p0 + np.array([0.0, 0.0, h])
        # sweep bar: down in front of the deck, back to the pit, up again
        park = np.array(self._bar_park())
        front = np.array([self.deck_x0 - 0.5, 0.0, bed + 0.5])
        back = np.array([self.deck_end + 0.6, 0.0, bed + 0.5])
        if s < t1:
            bar = park
        elif s < t2:
            bar = park + (front - park) * self._ease((s - t1) / self.T_DOWN)
        elif s < t3:
            bar = front + (back - front) * self._ease((s - t2) / self.T_SWEEP)
        else:
            k = self._ease((s - t3) / self.T_LOWER)
            bar = back + (np.array([back[0], 0.0, park[2]]) - back) * k
        self.sim.data.mocap_pos[self.bar_mocap] = bar
        # everything not held (knocked pins; all pins on a full reset) that the bar
        # has passed goes into the machine (teleport into the hidden housing)
        if t2 <= s:
            for i in range(N_PINS):
                if self.pin_hold[i] is None:
                    p, _ = self.pin_pose(i)
                    if not np.all(np.isfinite(p)) or p[0] < bar[0] + 0.3 or s >= t3:
                        self._store_pin(i)
        # a full rack: all 10 lowered from the machine onto the spots
        if sw["full"] and s >= t3:
            k = self._ease(min((s - t3) / self.T_LOWER, 1.0))
            for i in range(N_PINS):
                self.pin_hold[i] = np.array([*self.spots[i], bed + 0.002 + (1 - k) * c.store_z])
        if s >= t4:
            self._finish_sweep_state()
            self.phase = self.state = "returning"
            self._t_phase = t

    def _finish_sweep_state(self) -> None:
        """End of a pinsetter cycle: release the pins that stand, the rest stay in the
        machine."""
        sw = self._sweep
        bed = self.cfg.bed_height
        if sw["full"]:
            self.rack[:] = True
            self.n_rack_resets += 1
            for i in range(N_PINS):
                self._place_pin(i, (*self.spots[i], bed + 0.002))
                self.pin_hold[i] = None
        else:
            for i in range(N_PINS):
                if self.rack[i]:
                    p0 = sw["lift_from"].get(i, np.array([*self.spots[i], bed + 0.002]))
                    self._place_pin(i, p0)
                    self.pin_hold[i] = None
                else:
                    self._store_pin(i)
        self._park_bar()
        self._sweep = None

    @staticmethod
    def _ease(x: float) -> float:
        x = min(max(x, 0.0), 1.0)
        return x * x * (3 - 2 * x)

    # ------------------------------------------------------------ ball return / fly
    def _fly_clear_of_return(self) -> bool:
        spot = np.array([self.cfg.return_x, 0.0])
        far = float(np.linalg.norm(self.fly_xy() - spot)) > 3.0
        return far and (self.fly_mode == "wait" or self.run_time() - self._t_phase > 8.0)

    def _ball_to_return_spot(self) -> None:
        c = self.cfg
        y = float(self._rng.uniform(-c.return_y_jitter, c.return_y_jitter))
        self._ball_hidden = False
        self._place_ball((c.return_x, y, self.approach_z + c.ball_radius + 0.01))
        self.n_ball_returns += 1
        if self.session.actions.busy:
            self.session.actions.cancel()
        self.phase = self.state = "fetch"
        self._t_phase = self.run_time()
        self.fly_mode = "pilot"
        self.pilot.reset()
        self._new_aim()

    def _fly_idle(self, t: float) -> None:
        """The fly while the ball is away: watch the roll (frozen at the line), then
        walk back to the waiting spot behind the ball-return spot and wait."""
        c = self.cfg
        acts = self.session.actions
        if self.fly_mode == "watch":
            if self.phase in ("sweep", "returning"):
                if acts.busy:
                    acts.cancel()
                if getattr(self, "_celebrate", False):
                    self._celebrate = False
                    acts.trigger(_make_action("groom", duration=1.5), source="job")
                self.fly_mode = "walk_back"
            else:
                if not acts.busy:
                    acts.trigger(_make_action("freeze", duration=30.0), source="job")
                self.steering.set(self.sim.heading(), 0.0)
        elif self.fly_mode == "walk_back":
            if acts.busy:  # celebrating
                self.steering.set(None, 1.0)
                return
            wp = np.array([c.wait_x, 0.0])
            self.pilot.aim_at(wp, 0.9)
            self.unstick()
            if float(np.linalg.norm(self.fly_xy() - wp)) < 0.7:
                self.fly_mode = "wait"
        elif self.fly_mode == "wait":
            # face down the lane, then stand (freeze) until the ball is back
            err = abs(math.remainder(self.sim.heading(), 2 * math.pi))
            if err > math.radians(20) and not acts.busy:
                self.pilot.set_heading(0.0, 0.25)
            elif not acts.busy:
                acts.trigger(_make_action("freeze", duration=30.0), source="job")

    def _say_msg(self, msg: str) -> None:
        self.message = msg
        self._msg_t = self.run_time()

    # ------------------------------------------------------------ view / HUD
    # Shots (like a TV bowling broadcast): "fetch" (the fly and the ball on the
    # approach, the rack in the background), "lane" (after the release the camera
    # rides along behind the ball), "deck" (a cut to the pins before the ball gets
    # there; the pin action and the pinsetter), "back" (a cut back to the fly).
    CUTS = {("lane", "deck"), ("deck", "back"), ("deck", "fetch")}

    def _shot(self) -> str:
        b = self.ball_pos()
        if self.phase == "fetch":
            return "fetch"
        if self.phase in ("settle", "sweep"):
            return "deck"
        if self.phase == "rolling":
            if self._ball_hidden or not np.isfinite(b[0]) or b[0] > self.head_x - 7.0:
                return "deck"
            return "lane"
        return "back"

    def _deck_view(self) -> bool:
        return self._shot() == "deck"

    def camera_target(self) -> np.ndarray:
        c = self.cfg
        f = self.sim.thorax_position()
        b = self.ball_pos()
        if not np.all(np.isfinite(b)):
            b = f
        shot = self._shot()
        if shot == "fetch":
            out = 0.5 * f + 0.5 * b + np.array([2.5, 0.8, 0.6])
        elif shot == "deck":
            # (low enough to see under the masking unit to the back row)
            out = np.array([self.head_x + 4.5, 0.6, c.bed_height + 2.6])
        elif shot == "lane":
            # ride along: the ball in the lower third, the lane ahead of it
            # (starting a little down the lane: from right behind the ball at the top
            # of the ramp the guide rails fill the picture)
            out = np.array([max(b[0] + 9.0, c.foul_x + 7.0), 0.5 * b[1], c.bed_height + 1.5])
        else:  # back: to the fly
            out = f + np.array([2.5, 0.8, 0.6])
        return out if np.all(np.isfinite(out)) else f

    def camera_preset(self) -> CameraPreset:
        shot = self._shot()
        # (azimuth, elevation, distance), smoothing tau in sim s (the ride-along is
        # tight: in slow motion 0.03 sim s is ~0.1 s on screen)
        goal, tau = {"fetch": ((16.0, -22.0, 13.0), 0.5), "lane": ((6.0, -18.0, 17.0), 0.03),
                     "deck": ((10.0, -15.0, 21.0), 0.3), "back": ((16.0, -22.0, 13.5), 0.5)}[shot]
        t = self.sim.time
        prev = getattr(self, "_shot_prev", None)
        cut = prev is not None and (prev, shot) in self.CUTS
        self._shot_prev = shot
        if self._cam is None or t < self._cam[0] or cut:
            self._cam = (t, np.array(goal))
        else:
            t0, cur = self._cam
            a = 1.0 - math.exp(-(t - t0) / tau)
            self._cam = (t, cur + a * (np.array(goal) - cur))
        az, el, dist = self._cam[1]
        # a cut: the camera's look-at snaps too (JobCamera smooths with tau_s)
        return CameraPreset(azimuth=float(az), elevation=float(el), distance=float(dist),
                            tau_s=1e-6 if cut else tau)

    # ------------------------------------------------------------ slow motion (edit)
    def _slowmo_scale(self) -> float:
        c = self.cfg
        k = min(max(float(c.slowmo), 0.05), 1.0)
        if k >= 1.0:
            return 1.0
        if self.phase == "rolling":
            return k
        if self.phase == "settle":
            s = self.run_time() - self._t_phase - c.slowmo_settle_s
            return k if s <= 0 else min(1.0, k + (1.0 - k) * s / 0.5)
        return 1.0

    def time_scale(self, present_dt: float) -> float:
        """Slow motion while the ball rolls and the pins fly (an *edit effect*: fewer
        physics steps per displayed frame; the physics itself is unchanged)."""
        self._last_scale = self._slowmo_scale()
        return self._last_scale

    def post_process(self, frame: np.ndarray, t: float) -> np.ndarray:
        """Label the slow motion on screen (edit effect, not physics)."""
        if getattr(self, "_last_scale", 1.0) >= 1.0:
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

    def scoreboard_lines(self) -> list[str]:
        g = self.game
        tot = g.totals()
        cells = []
        for f in range(10):
            mk = " ".join(g.marks[f]) or "."
            t = tot[f]
            cells.append(f"{f + 1}: {mk}" + ("" if t is None else f" ={t}"))
        return [" | ".join(cells[:5]), " | ".join(cells[5:])]

    def job_hud_lines(self) -> list[str]:
        g = self.game
        avg = self.total_game_pins / self.games_completed if self.games_completed else 0.0
        frame = f"frame {min(g.frame + 1, 10)} ball {g.ball_in_frame + 1}"
        head = (f"GAME {self.games_completed + 1}  {frame}  score {g.score()}   "
                f"last {self.last_game if self.last_game is not None else '-'}  "
                f"best {self.best_game}  avg {avg:.0f}")
        msg = self.message if self.run_time() - self._msg_t < 5.0 else ""
        return [head, *self.scoreboard_lines(),
                f"strikes {self.n_strikes}  spares {self.n_spares}  gutters {self.n_gutters}  "
                f"fouls {self.n_fouls}  rolls {self.n_rolls}  racks {self.n_rack_resets}",
                ">> " + msg if msg else "(ramp, guide, pinsetter, ball return: engineered, see docs)"]

    def job_stats(self) -> dict:
        avg = self.total_game_pins / self.games_completed if self.games_completed else 0.0
        return {"rolls": self.n_rolls, "pins": self.pins_total,
                "pins_per_roll": self.pins_total / self.n_rolls if self.n_rolls else 0.0,
                "strikes": self.n_strikes, "spares": self.n_spares, "gutters": self.n_gutters,
                "fouls": self.n_fouls, "dead_balls": self.n_dead_balls,
                "no_roll": self.n_no_roll, "voided": self.n_voided,
                "games": self.games_completed, "best_game": self.best_game,
                "last_game": self.last_game, "avg_game": avg, "score": self.game.score(),
                "frame": self.game.frame + 1, "best_streak": self.best_streak,
                "pinsetter_cycles": self.n_pinsetter, "rack_resets": self.n_rack_resets,
                "ball_returns": self.n_ball_returns, "balls_lost": self.n_balls_lost,
                "pushes_started": self.pilot.n_pushes, "unstuck": self.n_unstuck}
