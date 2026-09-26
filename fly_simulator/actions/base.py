"""Action framework: timed body behaviours that take over (or blend with) the
locomotion controller's joint targets, then hand control back.

How it plugs into ``fly_simulator.simulation.Simulation`` (no edits there):

* ``Simulation.step`` calls ``controller.step_and_apply()`` (writes the CPG / hybrid
  controller's position targets + adhesion into ``data.ctrl``) and *then* runs the
  ``pre_step_hooks`` right before ``mj_step``. ``ActionManager`` registers one
  pre-step hook that, while an action is active, overwrites ``data.ctrl`` for the
  42 leg position actuators and the 6 adhesion actuators with
  ``w * action + (1 - w) * controller`` (``w`` ramps 0 -> 1 -> 0 at the start /
  end of the action), so the controller keeps running underneath (CPG phases keep
  advancing) and walking resumes from wherever the rhythm is when the action ends.
* Drive actions (back away, turn in place) instead replace the hybrid
  controller's descending signal ``[left, right]`` (the manager wraps
  ``controller.descending_signal``; the brain link's ``signal_filter`` still runs
  inside it), so the CPG itself walks backward / turns.
* ``pre_reset_hooks``: ``sim.reset()`` cancels the active action and restores any
  temporarily changed physics parameter (actuator gain / force range).
* Nothing ever writes ``qpos`` / ``qvel``: the fly is never teleported.

Actions only ever write actuator commands (``ctrl``) -- plus, for the documented jump
boost, a temporary per-actuator ``kp`` / force-range change that is restored at the
end of the stroke, on cancel and on reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import mujoco as mj
import numpy as np

if TYPE_CHECKING:
    from fly_simulator.simulation import Simulation

LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")  # FlyGym leg order (adhesion order too)


def smoothstep(x: float) -> float:
    """C1 ramp 0 -> 1 on x in [0, 1] (clipped)."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class BodyIndex:
    """Index helpers for the fly's actuators, joints and contact geoms.

    ``targets`` arrays are in the controller's position-actuator order
    (``controller.dof_order``, 7 DoFs per leg: coxa yaw / pitch / roll, femur
    (trochanter) pitch / roll, tibia pitch, tarsus1 pitch).
    """

    def __init__(self, sim: "Simulation") -> None:
        ctrl = sim.controller
        m = sim.model
        self.names = [d.name for d in ctrl.dof_order]
        self._ix = {n: i for i, n in enumerate(self.names)}
        self.pos_ids = np.asarray(ctrl._pos_ids, dtype=np.int64)
        self.adh_ids = None if ctrl._adh_ids is None else np.asarray(ctrl._adh_ids, dtype=np.int64)
        self.n = len(self.names)
        self.stand = np.asarray(ctrl.initial_action().joint_angles, dtype=float).copy()
        # joint qpos addresses of the actuated DoFs (for "hold current pose")
        prefix = f"{sim.fly_name}/"
        self.qpos_adr = np.array([
            m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, prefix + n)] for n in self.names
        ])
        # leg of every actuated DoF
        self.leg_of = np.array([LEGS.index(d.child.pos) for d in ctrl.dof_order])
        # geoms of each leg's tibia + tarsi (for touchdown detection)
        fly_bodies = np.flatnonzero(m.body_rootid == sim.thorax_body_id)
        self._fly_body = np.zeros(m.nbody, dtype=bool)
        self._fly_body[fly_bodies] = True
        self.geom_leg = np.full(m.ngeom, -1, dtype=np.int64)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if not self._fly_body[b]:
                continue
            bname = m.body(b).name.removeprefix(prefix)
            leg, _, link = bname.partition("_")
            if leg in LEGS and (link == "tibia" or link.startswith("tarsus")):
                self.geom_leg[g] = LEGS.index(leg)

    def idx(self, leg: str, joint: str) -> int:
        """``joint`` in {"thc_yaw", "thc_pitch", "thc_roll", "ctr_pitch", "ctr_roll",
        "fti", "tita"} (thorax-coxa, coxa-trochanter, femur-tibia, tibia-tarsus)."""
        name = {
            "thc_yaw": f"c_thorax-{leg}_coxa-yaw",
            "thc_pitch": f"c_thorax-{leg}_coxa-pitch",
            "thc_roll": f"c_thorax-{leg}_coxa-roll",
            "ctr_pitch": f"{leg}_coxa-{leg}_trochanterfemur-pitch",
            "ctr_roll": f"{leg}_coxa-{leg}_trochanterfemur-roll",
            "fti": f"{leg}_trochanterfemur-{leg}_tibia-pitch",
            "tita": f"{leg}_tibia-{leg}_tarsus1-pitch",
        }[joint]
        return self._ix[name]

    def leg_mask(self, legs) -> np.ndarray:
        legs = [LEGS.index(l) for l in legs]
        return np.isin(self.leg_of, legs)

    def current_angles(self, sim: "Simulation") -> np.ndarray:
        return sim.data.qpos[self.qpos_adr].copy()

    def leg_contacts(self, sim: "Simulation") -> np.ndarray:
        """bool[6]: leg (tibia/tarsus) touches something that is not the fly."""
        return self.ground_contacts(sim)[0]

    def ground_contacts(self, sim: "Simulation") -> tuple[np.ndarray, bool]:
        """(bool[6] legs (tibia/tarsi) touching non-fly geoms, any other fly geom --
        thorax, head, abdomen, femur... -- touching non-fly geoms)."""
        d = sim.data
        legs = np.zeros(6, dtype=bool)
        body = False
        n = d.ncon
        if n == 0:
            return legs, body
        geom = d.contact.geom[:n]
        gb = sim.model.geom_bodyid
        fly = self._fly_body
        for g1, g2 in geom:
            f1, f2 = fly[gb[g1]], fly[gb[g2]]
            if f1 == f2:
                continue
            g = g1 if f1 else g2
            leg = self.geom_leg[g]
            if leg >= 0:
                legs[leg] = True
            else:
                body = True
        return legs, body


@dataclass
class ActionCommand:
    """What an action wants this step. ``None`` fields leave the controller's value."""

    targets: np.ndarray | None = None  # (42,) leg position targets (rad)
    adhesion: np.ndarray | None = None  # (6,) in [0, 1]
    drive: np.ndarray | None = None  # (2,) descending signal override [left, right]
    extra: dict[int, float] = field(default_factory=dict)  # actuator id -> ctrl (wings, ...)


@dataclass
class ActionEvent:
    kind: str  # "start" | "end" | "cancel"
    name: str
    time: float  # sim time (s)
    info: dict = field(default_factory=dict)


class Action:
    """Base class. Subclasses implement ``begin`` / ``command`` and set ``duration``.

    Lifecycle (driven by ``ActionManager``, all in the physics thread):
    ``begin(mgr)`` once at the first step -> ``command(mgr, t)`` every physics step
    with ``t`` = seconds since start, until ``t >= duration`` or ``done`` -> ``end``.
    The manager blends the command in over ``blend_in`` s and out over the last
    ``blend_out`` s (``w`` = 1 in between). An action that computes its own
    transition from the controller pose (e.g. the jump crouch) uses
    ``blend_in = 0``.
    """

    name = "action"
    blend_in = 0.1
    blend_out = 0.2
    #: True if the action uses the hybrid controller's descending signal
    uses_drive = False

    def __init__(self, duration: float) -> None:
        self.duration = float(duration)
        self.done = False  # set to finish early (the blend-out then starts)
        self.info: dict = {}  # metrics reported in the "end" event
        self.source = "api"  # who triggered it (set by ActionManager.trigger)

    def phase(self, t: float) -> str:
        return "active"

    def validate(self, mgr: "ActionManager") -> None:
        """Raise if the action cannot run on ``mgr.sim`` (called by ``trigger``)."""

    def begin(self, mgr: "ActionManager") -> None:  # pragma: no cover - default
        pass

    def command(self, mgr: "ActionManager", t: float) -> ActionCommand:
        raise NotImplementedError

    def end(self, mgr: "ActionManager", cancelled: bool) -> None:
        pass


class ActionManager:
    """Runs one action at a time on a ``Simulation``.

    ``mgr = ActionManager(sim)`` attaches (one pre-step hook + one pre-reset hook,
    and wraps ``controller.descending_signal``). ``mgr.trigger(Jump())`` starts an
    action (replacing a running one unless ``queue=False`` and busy), ``mgr.cancel()``
    stops it (blending back over ``cancel_blend`` s), ``sim.reset()`` cancels at once.

    Thread safety: call ``trigger`` / ``cancel`` from the physics thread, or under
    the app's ``runner.locked()`` (the action itself starts at the next physics step).
    """

    def __init__(self, sim: "Simulation", cancel_blend: float = 0.15) -> None:
        self.sim = sim
        self.body = BodyIndex(sim)
        self.cancel_blend = float(cancel_blend)
        self.listeners: list[Callable[[ActionEvent], None]] = []
        self.action: Action | None = None
        self._pending: Action | None = None
        self._t0 = 0.0
        self._started = False
        self._fade: tuple[float, float, ActionCommand] | None = None  # (t_start, dur, last cmd)
        self._last_cmd: ActionCommand | None = None
        self._saved_params: dict[int, tuple[float, float, np.ndarray]] = {}
        self._drive_override: np.ndarray | None = None
        self._drive_w = 0.0
        self.history: list[ActionEvent] = []
        sim.pre_step_hooks.append(self._pre_step)
        sim.pre_reset_hooks.append(self._on_reset)
        ctrl = sim.controller
        if hasattr(ctrl, "descending_signal"):
            self._orig_signal = ctrl.descending_signal
            ctrl.descending_signal = self._descending_signal  # instance attribute wrapper
        else:  # pragma: no cover
            self._orig_signal = None

    # ------------------------------------------------------------------ public
    @property
    def busy(self) -> bool:
        return self.action is not None or self._pending is not None or self._fade is not None

    @property
    def active_name(self) -> str | None:
        a = self._pending or self.action
        return a.name if a is not None else None

    def phase(self) -> str | None:
        if self.action is None:
            return "handback" if self._fade is not None else None
        return self.action.phase(self.sim.time - self._t0)

    def trigger(self, action: Action, replace: bool = True, source: str = "api") -> bool:
        """Start ``action`` at the next physics step. Returns False if busy and
        ``replace`` is False. ``source`` (e.g. "key", "brain") is reported in the
        "start" event's info."""
        if action.uses_drive and self._orig_signal is None:
            raise RuntimeError(f"{action.name} needs the hybrid controller's descending signal")
        action.validate(self)
        if self.busy and not replace:
            return False
        action.source = source
        if self.action is not None:
            self._finish(cancelled=True, fade=True)
        self._pending = action
        return True

    def cancel(self) -> None:
        """Stop the running action and blend back to the controller."""
        self._pending = None
        if self.action is not None:
            self._finish(cancelled=True, fade=True)

    def detach(self) -> None:
        self._abort()
        if self._pre_step in self.sim.pre_step_hooks:
            self.sim.pre_step_hooks.remove(self._pre_step)
        if self._on_reset in self.sim.pre_reset_hooks:
            self.sim.pre_reset_hooks.remove(self._on_reset)
        if self._orig_signal is not None:
            self.sim.controller.descending_signal = self._orig_signal

    # ------------------------------------------------ temporary physics params
    def boost_actuators(self, act_ids, kp_scale: float, force_scale: float) -> None:
        """Temporarily scale ``kp`` (position gain + matching bias) and the force
        range of the given actuators. Restored by ``restore_actuators`` (called
        automatically at action end / cancel / reset)."""
        m = self.sim.model
        for a in np.asarray(act_ids, dtype=np.int64):
            a = int(a)
            if a not in self._saved_params:
                self._saved_params[a] = (float(m.actuator_gainprm[a, 0]),
                                         float(m.actuator_biasprm[a, 1]),
                                         m.actuator_forcerange[a].copy())
            g0, b0, f0 = self._saved_params[a]
            m.actuator_gainprm[a, 0] = g0 * kp_scale
            m.actuator_biasprm[a, 1] = b0 * kp_scale
            m.actuator_forcerange[a] = f0 * force_scale

    def restore_actuators(self) -> None:
        m = self.sim.model
        for a, (g0, b0, f0) in self._saved_params.items():
            m.actuator_gainprm[a, 0] = g0
            m.actuator_biasprm[a, 1] = b0
            m.actuator_forcerange[a] = f0
        self._saved_params.clear()

    # --------------------------------------------------------------- internals
    def _emit(self, kind: str, action: Action, info: dict | None = None) -> None:
        ev = ActionEvent(kind, action.name, self.sim.time, dict(info or {}))
        self.history.append(ev)
        for fn in list(self.listeners):
            fn(ev)

    def _finish(self, cancelled: bool, fade: bool) -> None:
        a = self.action
        if a is None:
            return
        a.end(self, cancelled)
        self.restore_actuators()
        if fade and self._last_cmd is not None:
            self._fade = (self.sim.time, self.cancel_blend, self._last_cmd)
        else:
            self._fade = None
        self.action = None
        self._drive_override = None
        self._emit("cancel" if cancelled else "end", a, a.info)

    def _abort(self) -> None:
        """Immediate stop (reset): no fade, params restored."""
        self._pending = None
        if self.action is not None:
            self._finish(cancelled=True, fade=False)
        self._fade = None
        self._last_cmd = None
        self._drive_override = None
        self.restore_actuators()

    def _on_reset(self, sim: "Simulation") -> None:
        self._abort()

    def _descending_signal(self) -> np.ndarray:
        sig = self._orig_signal()
        if self._drive_override is None:
            return sig
        w = self._drive_w
        return (1.0 - w) * np.asarray(sig, dtype=float) + w * self._drive_override

    def _weight(self, a: Action, t: float) -> float:
        w = 1.0
        if a.blend_in > 0:
            w = min(w, smoothstep(t / a.blend_in))
        return w

    def _pre_step(self, sim: "Simulation") -> None:
        if self._pending is not None:
            self.action, self._pending = self._pending, None
            self._t0 = sim.time
            self._fade = None
            self.action.begin(self)
            self._emit("start", self.action, {"source": getattr(self.action, "source", "api")})
        a = self.action
        d = sim.data
        if a is not None:
            t = sim.time - self._t0
            if a.done or t >= a.duration:
                # natural end: blend out from the last command over blend_out
                last = self._last_cmd
                a.end(self, False)
                self.restore_actuators()
                self.action = None
                self._drive_override = None
                self._emit("end", a, a.info)
                if last is not None and a.blend_out > 0:
                    self._fade = (sim.time, a.blend_out, last)
                    a = None
                else:
                    return
            else:
                cmd = a.command(self, t)
                w = self._weight(a, t)
                self._apply(cmd, w)
                self._last_cmd = cmd
                return
        if self._fade is not None:
            t_start, dur, cmd = self._fade
            x = (sim.time - t_start) / dur if dur > 0 else 1.0
            if x >= 1.0:
                for aid in cmd.extra:
                    d.ctrl[aid] = 0.0
                self._fade = None
                self._last_cmd = None
                self._drive_override = None
                return
            self._apply(cmd, 1.0 - smoothstep(x))

    def _apply(self, cmd: ActionCommand, w: float) -> None:
        d = self.sim.data
        b = self.body
        if cmd.targets is not None:
            if w >= 1.0:
                d.ctrl[b.pos_ids] = cmd.targets
            else:
                d.ctrl[b.pos_ids] = w * cmd.targets + (1.0 - w) * d.ctrl[b.pos_ids]
        if cmd.adhesion is not None and b.adh_ids is not None:
            # adhesion is switched, not blended (on when the blended command says > 0.5)
            ctrl_adh = d.ctrl[b.adh_ids]
            d.ctrl[b.adh_ids] = np.where(w >= 0.5, cmd.adhesion, ctrl_adh)
        for aid, v in cmd.extra.items():
            # extra actuators (wings, proboscis) are not driven by the controller:
            # blend toward their rest value 0
            d.ctrl[aid] = w * v
        if cmd.drive is not None:
            self._drive_override = np.asarray(cmd.drive, dtype=float)
            self._drive_w = w
        elif self.action is None:
            self._drive_override = None
