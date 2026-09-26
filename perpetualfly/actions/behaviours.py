"""Freeze, groom, back-away / turn-in-place, wing raise, proboscis extension."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import mujoco as mj
import numpy as np

from .base import Action, ActionCommand, smoothstep
from .body import PROBOSCIS_DOFS, WING_DOFS


# ---------------------------------------------------------------------- freeze
class Freeze(Action):
    """Stop stepping and hold a stable stance with all tarsi adhering.

    Legs that are in stance when the freeze starts (controller adhesion on) keep
    their current targets, so the fly stops "mid-stride" as real flies do; legs in
    swing are put down by blending them to the standing pose over ``settle_s``.
    The CPG keeps running underneath; on release the targets blend back to the
    live gait over ``blend_out`` s.
    """

    name = "freeze"
    blend_in = 0.0
    blend_out = 0.25

    def __init__(self, duration: float = 1.0, settle_s: float = 0.08) -> None:
        super().__init__(duration)
        self.settle_s = settle_s

    def phase(self, t: float) -> str:
        return "settle" if t < self.settle_s else "hold"

    def begin(self, mgr) -> None:
        b, d = mgr.body, mgr.sim.data
        self._start = d.ctrl[b.pos_ids].copy()
        if b.adh_ids is not None:
            stance = d.ctrl[b.adh_ids] > 0.5
        else:
            stance = np.ones(6, dtype=bool)
        self._goal = self._start.copy()
        swing_dofs = ~stance[b.leg_of]
        self._goal[swing_dofs] = b.stand[swing_dofs]
        self._p0 = mgr.sim.thorax_position()
        self._on = np.ones(6)

    def command(self, mgr, t: float) -> ActionCommand:
        a = smoothstep(t / self.settle_s)
        return ActionCommand(targets=(1 - a) * self._start + a * self._goal, adhesion=self._on)

    def end(self, mgr, cancelled: bool) -> None:
        p = mgr.sim.thorax_position()
        self.info = {"drift_mm": float(np.hypot(*(p - self._p0)[:2])), "tilt_deg": mgr.sim.tilt_deg(),
                     "cancelled": cancelled}


# ---------------------------------------------------------------------- groom
@lru_cache(maxsize=1)
def load_grooming_clip():
    """Recorded front-leg grooming (NeuroMechFly v1 / DeepFly3D, aDN>CsChrimson
    antennal grooming, converted to FlyGym 2.1 joint angles; see
    perpetualfly/actions/convert_grooming.py). Returns (angles (T, 14) rad,
    dof_names, fps)."""
    path = files("perpetualfly.actions") / "data/grooming_front_legs.npz"
    with path.open("rb") as f:
        z = np.load(f)
        return z["angles"].astype(float), [str(n) for n in z["dof_names"]], float(z["fps"])


class Groom(Action):
    """Front-leg (antennal / head) grooming by **replaying recorded kinematics**.

    The front legs follow the recorded joint angles (200 Hz clip, interpolated);
    mid and hind legs blend to the standing pose and hold it with adhesion on, the
    front tarsi release adhesion. ``speed`` scales playback rate, the clip loops
    (0-3 s of the recording = ~4 rubbing cycles) until ``duration`` elapses.
    ``source="synthetic"`` instead uses a hand-designed rhythmic sweep (front
    legs lifted to the head and rubbed up/down in antiphase), for comparison.
    """

    name = "groom"
    blend_in = 0.2
    blend_out = 0.3

    def __init__(self, duration: float = 3.0, speed: float = 1.0, source: str = "recorded",
                 sweep_hz: float = 6.0, mid_raise_deg: float = 20.0) -> None:
        super().__init__(duration)
        self.speed = speed
        # ``source`` picks recorded clip vs synthetic sweep. Stored as ``clip_source``
        # because ActionManager.trigger overwrites ``action.source`` with who
        # triggered it ("key" / "brain" / "api").
        self.clip_source = source
        self.sweep_hz = sweep_hz
        # Mid legs extended (coxa-trochanter +raise, femur-tibia -raise): lifts the
        # front of the body so the recorded front-leg rubbing (recorded tethered,
        # tarsi ~1 mm below the thorax) happens in the air, not on the floor.
        # Measured: 0 deg -> thorax 0.87 mm, tarsi on the floor 34 % of the time,
        # nose-down 8 deg; 20 deg -> thorax 1.28 mm, tarsi never touch, pitch -3 deg.
        self.mid_raise_deg = mid_raise_deg

    def begin(self, mgr) -> None:
        b = mgr.body
        self._front = b.leg_mask(("lf", "rf"))
        self._targets = b.stand.copy()
        r = np.radians(self.mid_raise_deg)
        for leg in ("lm", "rm"):
            self._targets[b.idx(leg, "ctr_pitch")] += r
            self._targets[b.idx(leg, "fti")] -= r
        self._adh = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 1.0])  # lf lm lh rf rm rh
        if self.clip_source == "recorded":
            angles, names, fps = load_grooming_clip()
            self._cols = np.array([b.names.index(n) for n in names])
            self._clip, self._fps = angles, fps
        else:
            self._cols = np.flatnonzero(self._front)
        self._n_contacts = []

    def _recorded(self, t: float) -> np.ndarray:
        x = (t * self.speed * self._fps) % (len(self._clip) - 1)
        i = int(x)
        f = x - i
        return (1 - f) * self._clip[i] + f * self._clip[i + 1]

    def _synthetic(self, mgr, t: float) -> np.ndarray:
        """Hand-designed alternative: both front legs protracted toward the head and
        rubbed up / down in antiphase at ``sweep_hz``."""
        b = mgr.body
        tgt = b.stand.copy()
        ph = 2 * np.pi * self.sweep_hz * t
        for k, leg in enumerate(("lf", "rf")):
            s = np.sin(ph + k * np.pi)
            tgt[b.idx(leg, "thc_pitch")] += 0.9 + 0.15 * s  # swing forward, toward the head
            tgt[b.idx(leg, "ctr_pitch")] += -0.4 + 0.3 * s
            tgt[b.idx(leg, "fti")] += 0.6 - 0.4 * s
            tgt[b.idx(leg, "thc_roll")] -= 0.3
        return tgt[self._cols]

    def command(self, mgr, t: float) -> ActionCommand:
        tg = self._targets
        tg[self._cols] = self._recorded(t) if self.clip_source == "recorded" else self._synthetic(mgr, t)
        return ActionCommand(targets=tg.copy(), adhesion=self._adh)

    def end(self, mgr, cancelled: bool) -> None:
        self.info = {"source": self.clip_source, "trigger": self.source, "tilt_deg": mgr.sim.tilt_deg(), "cancelled": cancelled}


# ------------------------------------------------------------ drive overrides
class DriveOverride(Action):
    """Replace the hybrid controller's descending signal [left, right] for a while
    (|v| = CPG amplitude per side, v < 0 = that side steps backward). The CPG does
    the stepping, so the legs stay coordinated; the manager blends the signal in
    and out."""

    name = "drive"
    uses_drive = True
    blend_in = 0.1
    blend_out = 0.2

    def __init__(self, left: float, right: float, duration: float = 1.0, name: str | None = None):
        super().__init__(duration)
        self.signal = np.array([left, right], dtype=float)
        if name:
            self.name = name

    def begin(self, mgr) -> None:
        self._p0 = mgr.sim.thorax_position()
        self._h0 = self._h_prev = mgr.sim.heading()
        self._turn = 0.0  # unwrapped heading change (rad)

    def command(self, mgr, t: float) -> ActionCommand:
        h = mgr.sim.heading()
        self._turn += (h - self._h_prev + np.pi) % (2 * np.pi) - np.pi
        self._h_prev = h
        return ActionCommand(drive=self.signal)

    def end(self, mgr, cancelled: bool) -> None:
        sim = mgr.sim
        dp = sim.thorax_position() - self._p0
        fwd = np.array([np.cos(self._h0), np.sin(self._h0)])
        self.info = {"forward_mm": float(dp[:2] @ fwd), "turn_deg": float(np.degrees(self._turn)),
                     "cancelled": cancelled}


def BackAway(duration: float = 1.0, speed: float = 1.0) -> DriveOverride:
    """Walk backward (moonwalker-like): both sides at -speed."""
    return DriveOverride(-speed, -speed, duration, name="back_away")


def TurnInPlace(direction: str = "left", duration: float = 1.0, amp: float = 1.0) -> DriveOverride:
    """Turn on the spot: the two sides step in opposite directions. "left" =
    counter-clockwise seen from above: left side steps backward, right side
    forward (signal [-1, +1]; measured ~270 deg/s at amp 1 after a ~0.1 s ramp).
    The heading hold turns the fly back toward ``ControllerConfig.target_heading_deg``
    after the action (the shorter way round)."""
    s = -1.0 if direction == "left" else 1.0
    return DriveOverride(s * amp, -s * amp, duration, name=f"turn_{direction}")


# ------------------------------------------------- wings / proboscis (extra joints)
def _extra_actuators(mgr, dofs, tag):
    m = mgr.sim.model
    ids = {}
    for dof in dofs:
        a = mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, f"{mgr.sim.fly_name}/{dof}-{tag}")
        if a < 0:
            raise RuntimeError(
                f"actuator for {dof} not found: build the Simulation with "
                "fly_factory=perpetualfly.actions.make_action_fly_factory(...)")
        ids[dof] = a
    return ids


class _PoseAction(Action):
    """Hold extra-joint position targets (rad) for ``duration``, legs untouched."""

    pose: dict[str, float] = {}
    dofs: tuple[str, ...] = ()
    tag = ""

    def __init__(self, duration: float, pose: dict[str, float] | None = None,
                 blend_in: float = 0.12, blend_out: float = 0.2) -> None:
        super().__init__(duration)
        if pose is not None:
            self.pose = pose
        self.blend_in, self.blend_out = blend_in, blend_out

    def validate(self, mgr) -> None:
        _extra_actuators(mgr, self.dofs, self.tag)

    def begin(self, mgr) -> None:
        ids = _extra_actuators(mgr, self.dofs, self.tag)
        self._extra = {ids[k]: v for k, v in self.pose.items()}

    def command(self, mgr, t: float) -> ActionCommand:
        return ActionCommand(extra=self._extra)


class WingRaise(_PoseAction):
    """Both wings raised up and slightly spread (a wing-raise / threat display, and
    the posture before a voluntary take-off). Needs the extra wing joints."""

    name = "wings"
    dofs = WING_DOFS
    tag = "wingpos"
    pose = {"c_thorax-l_wing-yaw": 1.2, "c_thorax-r_wing-yaw": 1.2,
            "c_thorax-l_wing-roll": 0.35, "c_thorax-r_wing-roll": 0.35}

    def __init__(self, duration: float = 1.0, **kw) -> None:
        super().__init__(duration, **kw)


class ProboscisExtend(_PoseAction):
    """Proboscis extension (rostrum swung down, haustellum unfolded). Needs the
    extra proboscis joints."""

    name = "proboscis"
    dofs = PROBOSCIS_DOFS
    tag = "proboscispos"
    pose = {"c_head-c_rostrum-pitch": -1.0, "c_rostrum-c_haustellum-pitch": 1.0}

    def __init__(self, duration: float = 1.0, **kw) -> None:
        super().__init__(duration, **kw)
