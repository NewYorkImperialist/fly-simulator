"""FLY PONG: the connectome brain is the paddle controller.

Embodied design (docs/GAMES.md, game 4): the NeuroMechFly stands (standing pose,
all tarsi adhering) on a sled behind its paddle at one end of a fly-scale Pong
court in MuJoCo (26 mm long, 18 mm wide; the fly is 2.5 mm long). A ball (a
mocap sphere, arcade kinematics: straight lines, mirror bounces off the side
walls) travels between the fly's paddle and a scripted opponent paddle at the far
end. The ball is a small moving object for the fly's two compound eyes: its
position on each (gaze-stabilised) eye drives that eye's **LC10a** neurons, the
pursuit channel of FOLLOW THE LEADER (same ``PursuitVision`` / ``pursuit_response``
interface). In the FlyWire connectome one side's LC10a drives the *ipsilateral*
DNa01/DNa02 steering pair, i.e. a turn *toward* the ball.

The game interface (ours): the DNa01/02 left-right difference, the same
``turn_command`` as FLY THROUGH RINGS, sets the paddle's **lateral velocity**:
v = vmax * (tanh(turn_L / 25 Hz) - tanh(turn_R / 25 Hz)), + = the fly's left,
low-passed with 60 ms. This is the closed-loop "flight simulator" idea of fly
vision research (the fly's yaw command moves its visual world), with lateral
translation instead of rotation. The sled carries the fly kinematically: every
physics step the fly's free joint is shifted by the paddle's displacement (a
position shift with no velocity change, so the standing legs feel nothing).

Opponent: a scripted AI paddle (labelled on screen; not a brain): it predicts
where the ball will arrive, with an aiming error, and moves there at a limited
speed. Game: first to 7 points; each hit speeds the ball up.

This module has no rendering and no brain: ``PongPhysics`` (pure-Python ball /
paddle kinematics, testable without MuJoCo), ``PongCourt`` (the MuJoCo bodies,
the pre-step hook, the fly carry) and ``PongGame`` (serve, points, score).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import mujoco as mj
import numpy as np

from fly_simulator.games.asteroids import GameEvent
from fly_simulator.games.rings import turn_command

PREFIX = "pong/"


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class PongDifficulty:
    name: str
    paddle_len_mm: float  # the fly's paddle
    ball_speed: float  # serve speed (mm/s)
    ai_speed: float  # opponent paddle max speed (mm/s)
    ai_error_mm: float  # SD of the opponent's arrival prediction error


PONG_DIFFICULTIES: dict[str, PongDifficulty] = {
    "easy": PongDifficulty("easy", 7.0, 16.0, 8.0, 2.0),
    "normal": PongDifficulty("normal", 6.0, 20.0, 10.0, 1.6),
    "hard": PongDifficulty("hard", 5.0, 24.0, 13.0, 1.2),
}


@dataclass
class PongConfig:
    # court (mm): the fly's paddle face at x = paddle_x, the opponent's at
    # paddle_x + court_len; side walls at y = +- court_half_w
    paddle_x: float = 2.4
    court_len: float = 26.0
    court_half_w: float = 9.0
    ball_radius_mm: float = 0.75
    # the table surface (visual): 0.1 mm above the ground the fly stands on (a thinner
    # slab loses depth-buffer precision at the court camera's distance and flickers)
    table_top_mm: float = 0.1
    paddle_thick_mm: float = 0.4
    paddle_height_mm: float = 0.7
    ai_paddle_len_mm: float = 6.0
    # ball
    speedup: float = 1.06  # x speed per paddle hit
    max_ball_speed: float = 45.0
    max_angle_deg: float = 45.0  # bounce angle at the paddle's end (English)
    serve_angle_deg: float = 25.0  # +- random serve angle
    miss_behind_mm: float = 1.2  # the ball is out this far behind a paddle face
    # the fly's paddle: v = paddle_vmax * turn_command(DNa01/02), low-passed
    paddle_vmax: float = 25.0  # mm/s
    paddle_tau_s: float = 0.06
    r_ref_hz: float = 25.0
    # the ball as the eyes see it (PursuitVision reads cfg.visual_radius_mm)
    visual_radius_mm: float = 0.75
    # rules
    win_points: int = 7
    ready_s: float = 1.0
    point_pause_s: float = 1.0
    return_points: int = 10
    point_points: int = 100
    win_bonus: int = 500
    lives: int = 0  # unused (points instead); kept for the shared HUD helpers
    difficulty: str = "normal"
    # look
    rgba_ball: tuple = (0.97, 0.93, 0.75, 1.0)
    rgba_ball_band: tuple = (0.85, 0.35, 0.15, 1.0)
    rgba_paddle: tuple = (0.25, 0.85, 0.40, 1.0)
    rgba_ai: tuple = (0.90, 0.30, 0.25, 1.0)
    rgba_wall: tuple = (0.35, 0.55, 0.90, 1.0)
    rgba_line: tuple = (0.92, 0.92, 0.92, 1.0)
    rgba_sled: tuple = (0.20, 0.45, 0.28, 1.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def diff(self) -> PongDifficulty:
        return PONG_DIFFICULTIES[self.difficulty]

    @property
    def ai_x(self) -> float:
        return self.paddle_x + self.court_len


def fold_y(y: float, ymax: float) -> float:
    """Unfold a straight path through mirror walls at +-ymax (ball centre)."""
    p = 4.0 * ymax
    u = (y + ymax) % p
    return u - ymax if u <= 2.0 * ymax else 3.0 * ymax - u


def aim_velocity(x0: float, y0: float, x1: float, y1: float, speed: float, ymax: float,
                 bounce: int = 0) -> tuple[float, float]:
    """Velocity from (x0, y0) that arrives at (x1, y1), straight (``bounce`` 0) or
    off the +y (1) / -y (-1) wall."""
    yi = y1 if bounce == 0 else (2.0 * ymax - y1 if bounce > 0 else -2.0 * ymax - y1)
    dx, dy = x1 - x0, yi - y0
    n = math.hypot(dx, dy)
    return speed * dx / n, speed * dy / n


# ---------------------------------------------------------------------------
# kinematics (pure Python)
# ---------------------------------------------------------------------------


class PongPhysics:
    """Ball and paddles. ``step(dt)`` integrates; outcomes go to ``events`` as
    ``(kind, info)``: ``hit`` (side fly|ai, offset), ``wall``, ``miss`` (side).

    * fly paddle: ``pcmd`` (mm/s, + = +y = the fly's left) low-passed with
      ``paddle_tau_s``; centre clipped so the paddle stays between the walls.
    * AI paddle, ``ai_mode``: ``track`` (the game's opponent: predicted arrival +
      aim offset + error, speed-limited), ``wall`` (the experiment's opponent:
      always returns, aiming the k-th return with ``aim_fn(k) -> (y, bounce)``),
      ``off`` (stays put).
    """

    def __init__(self, cfg: PongConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.ai_mode = "track"
        self.aim_fn: Callable[[int], tuple[float, int]] | None = None
        self.events: list[tuple[str, dict]] = []
        self.reset()

    # geometry
    @property
    def ymax(self) -> float:
        return self.cfg.court_half_w - self.cfg.ball_radius_mm

    @property
    def half(self) -> float:
        return 0.5 * self.cfg.diff.paddle_len_mm

    @property
    def ai_half(self) -> float:
        return 0.5 * self.cfg.ai_paddle_len_mm

    def reset(self) -> None:
        self.bx, self.by = self.cfg.paddle_x + 0.5 * self.cfg.court_len, 0.0
        self.vx = self.vy = 0.0
        self.speed = self.cfg.diff.ball_speed
        self.in_play = False
        self.visible = False
        self.out_t = 0.0  # time since the ball went out (it sinks into the gutter)
        self.z_drop = 0.0
        self._passed = None  # side whose face the ball got past
        self.py = self.pv = self.pcmd = 0.0
        self.ay = 0.0
        self.ai_target = 0.0
        self.ai_hits = 0
        self.fly_hits = 0
        self.last_cross: dict | None = None  # the last crossing of the fly's face
        self.roll = np.array([1.0, 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------ serve
    def serve(self, toward: str = "fly", y0: float = 0.0, angle_deg: float | None = None,
              speed: float | None = None, x0: float | None = None) -> None:
        c = self.cfg
        a = math.radians(angle_deg if angle_deg is not None
                         else float(self.rng.uniform(-c.serve_angle_deg, c.serve_angle_deg)))
        self.speed = float(speed or c.diff.ball_speed)
        s = -1.0 if toward == "fly" else 1.0
        self.vx, self.vy = s * self.speed * math.cos(a), self.speed * math.sin(a)
        self._launch(c.paddle_x + 0.5 * c.court_len if x0 is None else x0, y0)
        self._new_ai_target()

    def serve_aimed(self, x0: float, y0: float, y1: float, bounce: int = 0,
                    speed: float | None = None) -> None:
        """Serve toward the fly from (x0, y0) so it arrives at the fly's face at y1."""
        c = self.cfg
        self.speed = float(speed or c.diff.ball_speed)
        self.vx, self.vy = aim_velocity(x0, y0, c.paddle_x + c.ball_radius_mm, y1, self.speed,
                                        self.ymax, bounce)
        self._launch(x0, y0)

    def _launch(self, x0: float, y0: float) -> None:
        self.bx, self.by = float(x0), float(np.clip(y0, -self.ymax, self.ymax))
        self.in_play = self.visible = True
        self.out_t = self.z_drop = 0.0
        self._passed = None

    def hide(self) -> None:
        self.in_play = self.visible = False

    # ------------------------------------------------------------------ AI
    def arrival_y(self, x: float) -> float | None:
        if self.vx == 0:
            return None
        t = (x - self.bx) / self.vx
        if t < 0:
            return None
        return fold_y(self.by + self.vy * t, self.ymax)

    def _new_ai_target(self) -> None:
        """Sampled when the ball starts toward the AI: its aim offset (hit off-centre to
        angle the return) and its prediction error."""
        d = self.cfg.diff
        self._ai_aim = float(self.rng.uniform(-0.7, 0.7)) * self.ai_half
        # the error grows with the ball speed (a faster ball is harder to read)
        sd = d.ai_error_mm * max(1.0, self.speed / d.ball_speed)
        self._ai_err = float(self.rng.normal(0.0, sd))

    def _step_ai(self, dt: float) -> None:
        c = self.cfg
        lim = c.court_half_w - self.ai_half
        if self.ai_mode == "off":
            return
        if self.ai_mode == "wall":
            tgt = self.by if self.in_play else 0.0
            self.ay = float(np.clip(tgt, -lim, lim))
            return
        if self.in_play and self.vx > 0:
            y = self.arrival_y(c.ai_x - c.ball_radius_mm)
            tgt = (y if y is not None else self.by) + self._ai_err - self._ai_aim
            vmax = c.diff.ai_speed
        else:
            tgt, vmax = 0.0, 0.5 * c.diff.ai_speed
        self.ai_target = float(np.clip(tgt, -lim, lim))
        step = vmax * dt
        self.ay += float(np.clip(self.ai_target - self.ay, -step, step))

    # ------------------------------------------------------------------ step
    def step(self, dt: float) -> float:
        """Advance by ``dt`` s. Returns the fly paddle's displacement (mm, +y)."""
        c = self.cfg
        a = 1.0 - math.exp(-dt / max(c.paddle_tau_s, 1e-6))
        self.pv += a * (self.pcmd - self.pv)
        lim = c.court_half_w - self.half
        y_new = min(max(self.py + self.pv * dt, -lim), lim)
        dy = y_new - self.py
        self.py = y_new
        self._step_ai(dt)
        if self.visible:
            self._step_ball(dt)
        return dy

    def _step_ball(self, dt: float) -> None:
        c = self.cfg
        r = c.ball_radius_mm
        x0 = self.bx
        self.bx += self.vx * dt
        self.by += self.vy * dt
        v = math.hypot(self.vx, self.vy)
        if v > 0:  # rolling (visual): axis z x v, angle |v| dt / r
            ax = np.array([-self.vy / v, self.vx / v, 0.0])
            q = np.zeros(4)
            mj.mju_axisAngle2Quat(q, ax, v * dt / r)
            out = np.zeros(4)
            mj.mju_mulQuat(out, q, self.roll)
            self.roll = out / np.linalg.norm(out)
        ym = self.ymax
        if self.by > ym:
            self.by, self.vy = 2 * ym - self.by, -self.vy
            self.events.append(("wall", {"y": ym}))
        elif self.by < -ym:
            self.by, self.vy = -2 * ym - self.by, -self.vy
            self.events.append(("wall", {"y": -ym}))
        if not self.in_play:
            self.out_t += dt
            self.z_drop = min(self.z_drop + 6.0 * dt, 3.0)
            if self.out_t > 0.6:
                self.visible = False
            return
        fx, ax_ = c.paddle_x + r, c.ai_x - r  # ball-centre planes of the two faces
        if self._passed is None and self.vx < 0 and x0 >= fx > self.bx:
            off = self.by - self.py
            self.last_cross = {"offset": off, "ball_y": self.by, "paddle_y": self.py,
                               "hit": abs(off) <= self.half + r}
            if abs(off) <= self.half + r:
                self.fly_hits += 1
                self.bx = 2 * fx - self.bx
                self.speed = min(self.speed * c.speedup, c.max_ball_speed)
                ang = math.radians(c.max_angle_deg) * max(-1.0, min(1.0, off / (self.half + r)))
                self.vx, self.vy = self.speed * math.cos(ang), self.speed * math.sin(ang)
                self._new_ai_target()
                self.events.append(("hit", {"side": "fly", "offset": off,
                                            "speed": self.speed}))
            else:
                self._passed = "fly"
        elif self._passed is None and self.vx > 0 and x0 <= ax_ < self.bx:
            off = self.by - self.ay
            if self.ai_mode == "wall" or abs(off) <= self.ai_half + r:
                self.ai_hits += 1
                self.bx = 2 * ax_ - self.bx
                self.speed = min(self.speed * c.speedup, c.max_ball_speed)
                if self.ai_mode == "wall" and self.aim_fn is not None:
                    y1, bounce = self.aim_fn(self.ai_hits)
                    self.vx, self.vy = aim_velocity(self.bx, self.by, fx, y1, self.speed,
                                                    self.ymax, bounce)
                else:
                    ang = math.radians(c.max_angle_deg) * max(-1.0, min(1.0, off / (self.ai_half + r)))
                    self.vx, self.vy = -self.speed * math.cos(ang), self.speed * math.sin(ang)
                self.events.append(("hit", {"side": "ai", "offset": off, "speed": self.speed}))
            else:
                self._passed = "ai"
        if self._passed == "fly" and self.bx < c.paddle_x - c.miss_behind_mm:
            self.in_play = False
            self.events.append(("miss", {"side": "fly", "offset": self.last_cross["offset"]
                                         if self.last_cross else None}))
        elif self._passed == "ai" and self.bx > c.ai_x + c.miss_behind_mm:
            self.in_play = False
            self.events.append(("miss", {"side": "ai", "offset": self.by - self.ay}))


# ---------------------------------------------------------------------------
# the court in MuJoCo
# ---------------------------------------------------------------------------


def _quat_x90() -> tuple:
    return (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)


class PongCourt:
    """Visual court (walls, lines), the paddles, the sled and the ball; all mocap /
    static bodies without collision. ``extension(world)`` before ``add_fly``;
    ``attach(sim)`` installs the pre-step hook (kinematics + fly carry) and a reset
    hook (paddle back to the centre with the fly). Also the "target" for
    ``PursuitVision``: ``active``, ``position3()``, ``cfg.visual_radius_mm``."""

    def __init__(self, cfg: PongConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or PongConfig()
        self.cfg.visual_radius_mm = self.cfg.ball_radius_mm
        self.phys = PongPhysics(self.cfg, seed=seed)
        self.sim = None
        self._attached = False
        self.carried_mm = 0.0

    # ------------------------------------------------------------------ build
    def extension(self, world) -> None:
        c = self.cfg
        spec: mj.MjSpec = world.mjcf_root
        B, S = mj.mjtGeom.mjGEOM_BOX, mj.mjtGeom.mjGEOM_SPHERE
        wb = spec.worldbody
        x0, x1 = c.paddle_x - 3.0, c.ai_x + 3.0  # walls run a little past both ends
        xm, hx = 0.5 * (x0 + x1), 0.5 * (x1 - x0)

        def static(name, size, pos, rgba):
            wb.add_geom(name=f"{PREFIX}{name}", type=B, size=size, pos=pos, rgba=rgba,
                        contype=0, conaffinity=0, group=1)

        # the playing surface: a dark green table under the court (visual; the fly
        # stands on the ground plane 0.1 mm below its top, hidden by the sled deck)
        static("table", (hx, c.court_half_w, 0.5 * c.table_top_mm),
               (xm, 0.0, 0.5 * c.table_top_mm), (0.13, 0.33, 0.29, 1.0))
        for s in (1.0, -1.0):  # side walls (low bumpers)
            static(f"wall{'L' if s > 0 else 'R'}", (hx, 0.25, 0.35),
                   (xm, s * (c.court_half_w + 0.25), 0.35), c.rgba_wall)
        # painted lines: centre dashes, the two goal lines
        for k in range(9):
            y = -c.court_half_w + (k + 0.5) * (2 * c.court_half_w / 9)
            static(f"dash{k}", (0.06, 0.45, 0.003), (c.paddle_x + 0.5 * c.court_len, y, c.table_top_mm + 0.003),
                   c.rgba_line)
        for nm, x in (("goalF", c.paddle_x - c.miss_behind_mm), ("goalA", c.ai_x + c.miss_behind_mm)):
            static(nm, (0.05, c.court_half_w, 0.003), (x, 0.0, c.table_top_mm + 0.003), (0.9, 0.5, 0.2, 1.0))

        def mocap(name, pos):
            return wb.add_body(name=f"{PREFIX}{name}", pos=pos, mocap=True)

        half = 0.5 * c.diff.paddle_len_mm
        t, h = c.paddle_thick_mm, c.paddle_height_mm
        # the fly's paddle: a green bar in front of it, on the sled deck under it
        pb = mocap("paddle", (c.paddle_x - 0.5 * t, 0.0, 0.5 * h))
        pb.add_geom(name=f"{PREFIX}paddle_bar", type=B, size=(0.5 * t, half, 0.5 * h),
                    rgba=c.rgba_paddle, contype=0, conaffinity=0, group=1)
        sled = mocap("sled", (0.0, 0.0, c.table_top_mm + 0.004))
        sled.add_geom(name=f"{PREFIX}sled_deck", type=B, size=(1.9, 1.7, 0.002),
                      pos=(-0.3, 0.0, 0.0), rgba=c.rgba_sled, contype=0, conaffinity=0, group=1)
        for s in (1.0, -1.0):  # rails from the deck to the paddle
            sled.add_geom(name=f"{PREFIX}sled_rail{'L' if s > 0 else 'R'}", type=B,
                          size=(0.5 * (c.paddle_x - t - 1.6), 0.08, 0.02),
                          pos=(0.5 * (c.paddle_x - t + 1.6), s * 1.2, 0.02), rgba=c.rgba_paddle,
                          contype=0, conaffinity=0, group=1)
        ab = mocap("ai", (c.ai_x + 0.5 * t, 0.0, 0.5 * h))
        ab.add_geom(name=f"{PREFIX}ai_bar", type=B, size=(0.5 * t, 0.5 * c.ai_paddle_len_mm, 0.5 * h),
                    rgba=c.rgba_ai, contype=0, conaffinity=0, group=1)
        r = c.ball_radius_mm
        bb = mocap("ball", (c.paddle_x + 0.5 * c.court_len, 0.0, -20.0))
        bb.add_geom(name=f"{PREFIX}ball", type=S, size=(r, 0, 0), rgba=c.rgba_ball,
                    contype=0, conaffinity=0, group=1)
        bb.add_geom(name=f"{PREFIX}ball_band", type=mj.mjtGeom.mjGEOM_CYLINDER,
                    size=(1.01 * r, 0.18 * r, 0), rgba=c.rgba_ball_band, contype=0,
                    conaffinity=0, group=1)

    def attach(self, sim) -> "PongCourt":
        if self._attached:
            return self
        self.sim = sim
        m = sim.model
        self.ids = {k: int(m.body_mocapid[m.body(f"{PREFIX}{k}").id])
                    for k in ("paddle", "sled", "ai", "ball")}
        self._free = int(sim._free_qpos)
        sim.pre_step_hooks.append(self._pre_step)
        sim.reset_hooks.append(self._on_reset)
        self._bar_gid = int(m.geom(f"{PREFIX}paddle_bar").id)
        self._attached = True
        self.sync_paddle_size()
        self._write()
        return self

    def sync_paddle_size(self) -> None:
        """The fly's paddle length follows the difficulty (resized in the model)."""
        if self.sim is not None:
            self.sim.model.geom_size[self._bar_gid, 1] = 0.5 * self.cfg.diff.paddle_len_mm

    def detach(self) -> None:
        if self._attached:
            self.sim.pre_step_hooks.remove(self._pre_step)
            self.sim.reset_hooks.remove(self._on_reset)
            self._attached = False

    def _on_reset(self, sim) -> None:
        # the fly is back at the origin: put the paddle back with it
        p = self.phys
        p.py = p.pv = p.pcmd = 0.0
        self._write()

    def _pre_step(self, sim) -> None:
        dy = self.phys.step(sim.timestep)
        if dy:
            sim.data.qpos[self._free + 1] += dy  # the sled carries the fly
            self.carried_mm += abs(dy)
        self._write()

    def _write(self) -> None:
        c, p, d = self.cfg, self.phys, self.sim.data
        t, h = c.paddle_thick_mm, c.paddle_height_mm
        d.mocap_pos[self.ids["paddle"]] = (c.paddle_x - 0.5 * t, p.py, 0.5 * h)
        d.mocap_pos[self.ids["sled"]] = (0.0, p.py, c.table_top_mm + 0.004)
        d.mocap_pos[self.ids["ai"]] = (c.ai_x + 0.5 * t, p.ay, 0.5 * h)
        z = c.table_top_mm + c.ball_radius_mm - p.z_drop if p.visible else -20.0
        d.mocap_pos[self.ids["ball"]] = (p.bx, p.by, z)
        q = np.zeros(4)
        mj.mju_mulQuat(q, p.roll, np.array(_quat_x90()))
        d.mocap_quat[self.ids["ball"]] = q

    # ------------------------------------------------------------------ eyes' target
    @property
    def active(self) -> bool:
        return bool(self.phys.in_play and self.phys.visible)

    def position3(self) -> np.ndarray:
        return np.array([self.phys.bx, self.phys.by, self.cfg.table_top_mm + self.cfg.ball_radius_mm])


# ---------------------------------------------------------------------------
# the game
# ---------------------------------------------------------------------------


class PongGame:
    """Serve, points and score of FLY PONG. ``after_physics()`` after every chunk.

    States: ``ready`` (before the first serve), ``playing``, ``point`` (pause after
    a point, then the next serve toward the side that lost it), ``gameover`` (someone
    reached ``win_points``), ``trial`` (experiment: outcomes are only recorded)."""

    name = "pong"

    def __init__(self, sim, court: PongCourt, cfg: PongConfig | None = None, seed: int = 0,
                 on_respawn: Callable[[], None] | None = None) -> None:
        self.sim = sim
        self.court = court
        self.phys = court.phys
        self.cfg = cfg or court.cfg
        self.rng = np.random.default_rng(seed)
        self.on_respawn = on_respawn
        self.events: list[GameEvent] = []
        self.listeners: list[Callable[[GameEvent], None]] = []
        self.crossings: list[dict] = []  # every ball reaching the fly's face
        self._t_offset = 0.0
        self._new_game()

    def time(self, sim_time: float | None = None) -> float:
        return self._t_offset + (self.sim.time if sim_time is None else float(sim_time))

    def _emit(self, kind: str, text: str = "", **info) -> GameEvent:
        ev = GameEvent(self.time(), kind, text, None, info)
        self.events.append(ev)
        if len(self.events) > 2000:
            del self.events[:1000]
        for fn in self.listeners:
            fn(ev)
        return ev

    def _new_game(self) -> None:
        self.state = "ready"
        self.score = 0.0
        self.fly_points = 0
        self.ai_points = 0
        self.returns = 0  # the fly's paddle hits
        self.balls_faced = 0  # balls that reached the fly's face
        self.rally = 0  # paddle hits (both sides) in the current rally
        self.best_rally = 0
        self.winner: str | None = None
        self.t_start = self.time()
        self.t_next = self.time() + self.cfg.ready_s
        self.t_end: float | None = None
        self._serve_to = "fly"
        self.phys.ai_mode = "track"
        self.phys.aim_fn = None
        self.phys.hide()
        self.phys.events.clear()

    def restart(self, difficulty: str | None = None, reset_fly: bool = True) -> None:
        if difficulty is not None:
            if difficulty not in PONG_DIFFICULTIES:
                raise ValueError(f"difficulty must be one of {sorted(PONG_DIFFICULTIES)}")
            self.cfg.difficulty = difficulty
            self.court.sync_paddle_size()
        if reset_fly:
            self._reset_fly()
        self._new_game()
        self._emit("restart", f"new game ({self.cfg.difficulty})")

    def _reset_fly(self) -> None:
        self._t_offset += self.sim.time
        self.sim.reset()
        self._t_offset -= self.sim.time
        if self.on_respawn is not None:
            self.on_respawn()

    @property
    def survival_s(self) -> float:
        end = self.t_end if self.t_end is not None else self.time()
        return max(0.0, end - self.t_start)

    @property
    def return_rate(self) -> float:
        return self.returns / self.balls_faced if self.balls_faced else 0.0

    # ------------------------------------------------------------------ rules
    def serve(self) -> None:
        self.rally = 0
        y0 = float(self.rng.uniform(-3.0, 3.0))
        self.phys.serve(self._serve_to, y0=y0)
        self.state = "playing"
        self._emit("serve", "SERVE" if self._serve_to == "fly" else "SERVE ->")

    def after_physics(self) -> None:
        c = self.cfg
        now = self.time()
        evs, self.phys.events[:] = list(self.phys.events), []
        for kind, info in evs:
            if kind == "hit":
                self.rally += 1
                self.best_rally = max(self.best_rally, self.rally)
                if info["side"] == "fly":
                    self.returns += 1
                    self.balls_faced += 1
                    self.crossings.append(dict(self.phys.last_cross or {}, t=now))
                    if self.state == "playing":
                        self.score += c.return_points
                    self._emit("return", f"RETURN  rally {self.rally}", rally=self.rally,
                               offset_mm=round(info["offset"], 2),
                               speed=round(info["speed"], 1))
                else:
                    self._emit("ai_hit", "", rally=self.rally)
            elif kind == "miss":
                if info["side"] == "fly":
                    self.balls_faced += 1
                    self.crossings.append(dict(self.phys.last_cross or {}, t=now))
                self._point(winner="ai" if info["side"] == "fly" else "fly", info=info)
        if self.state in ("ready", "point") and now >= self.t_next:
            self.serve()

    def _point(self, winner: str, info: dict) -> None:
        c = self.cfg
        if self.state == "trial":
            self._emit("miss", "MISS" if winner == "ai" else "AI MISSED", side=info["side"])
            return
        if self.state != "playing":
            return
        now = self.time()
        if winner == "fly":
            self.fly_points += 1
            self.score += c.point_points
            self._emit("point_fly", "FLY SCORES!", rally=self.rally)
            self._serve_to = "ai"
        else:
            self.ai_points += 1
            off = info.get("offset")
            self._emit("point_ai", "MISSED!  AI SCORES", rally=self.rally,
                       offset_mm=None if off is None else round(off, 2))
            self._serve_to = "fly"
        if max(self.fly_points, self.ai_points) >= c.win_points:
            self.winner = winner
            if winner == "fly":
                self.score += c.win_bonus
            self.state = "gameover"
            self.t_end = now
            self._emit("gameover", "THE FLY WINS!" if winner == "fly" else "AI WINS",
                       score=int(self.score), fly=self.fly_points, ai=self.ai_points)
            return
        self.state = "point"
        self.t_next = now + c.point_pause_s

    def summary(self) -> dict:
        return {"score": int(self.score), "fly_points": self.fly_points,
                "ai_points": self.ai_points, "returns": self.returns,
                "balls_faced": self.balls_faced, "return_rate": round(self.return_rate, 3),
                "best_rally": self.best_rally, "winner": self.winner,
                "survival_s": round(self.survival_s, 2), "difficulty": self.cfg.difficulty,
                "state": self.state}

    def score_entry(self) -> dict:
        return {"score": int(self.score), "fly": self.fly_points, "ai": self.ai_points,
                "returns": self.returns, "best_rally": self.best_rally,
                "return_rate": round(self.return_rate, 3)}


def paddle_command(rates: dict, cfg: PongConfig) -> float:
    """Paddle velocity command (mm/s, + = the fly's left) from DNa01/02."""
    return cfg.paddle_vmax * turn_command(rates, cfg.r_ref_hz)
