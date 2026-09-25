"""Fall detection: UPRIGHT / DESTABILIZED / FALLEN / RECOVERING state machine.

``FallDetector`` is a post-step hook (``sim.post_step_hooks.append(detector)`` or
``detector.attach(sim)``). It only observes; it never terminates or resets the sim.

Signals (evaluated every ``check_every_steps`` physics steps, default 1 ms):

* ``tilt``: angle between the thorax up axis and world up (deg).
* ``height``: thorax z minus the local ground height under the thorax
  (``ground_height_fn(x, y)``, default flat ground at z=0), so it stays meaningful
  on procedural terrain.
* ``body_contact``: thorax/head/abdomen touching terrain (never happens while walking;
  see ``ContactClassifier``). A single leg losing contact is *not* a signal.
* ``progress_speed``: horizontal displacement / time over the last
  ``progress_window_s`` (heading-agnostic; a stalled fly has ~0).

Transitions (all conditions have to *persist* for a hold time -> hysteresis):

* UPRIGHT -> DESTABILIZED  ("destabilized"): tilt > destab_tilt, height outside
  [destab_min_height, destab_max_height], body contact, or stalled.
* DESTABILIZED -> UPRIGHT  ("stabilized"): all "normal" conditions (tighter
  thresholds) hold for ``stabilize_hold_s``.
* UPRIGHT/DESTABILIZED -> FALLEN ("fall"): one of the fall rules below persists.
* FALLEN -> RECOVERING     ("recovering"): back on its feet for ``recovering_hold_s``.
* RECOVERING -> UPRIGHT    ("recovered"): stays on its feet for
  ``recovered_confirm_s`` *and* walks >= ``recovered_min_speed``.
* RECOVERING -> FALLEN     ("relapse"): loses footing again (not counted as a new fall).
* sim reset -> UPRIGHT     ("reset").

Calibration (flat ground, baseline hybrid controller, see docs in FallDetectorConfig
and scripts/demo_falls.py): 60 s of normal walking gives thorax height 1.074-1.189 mm,
tilt <= 7.4 deg, no body contact, 2-6 legs in contact, speed over any 0.5 s window
>= 14.0 mm/s -> zero false DESTABILIZED/FALLEN with the defaults.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Callable

import numpy as np

from perpetualfly.metrics.contacts import ContactClassifier, ContactSummary

if TYPE_CHECKING:
    from perpetualfly.simulation import Simulation


class FallState(str, Enum):
    UPRIGHT = "UPRIGHT"
    DESTABILIZED = "DESTABILIZED"
    FALLEN = "FALLEN"
    RECOVERING = "RECOVERING"


@dataclass
class FallDetectorConfig:
    """Thresholds (mm, deg, s). Rationale from measurements on flat ground:

    Normal walking (60 s): height 1.07-1.19 mm, tilt <= 7.4 deg, no body contact,
    speed >= 14 mm/s over every 0.5 s window. Lateral shoves on the thorax
    (multiples of body weight W ~= 10 uN, 10 ms): 1-2 W -> tilt 11-13 deg, height
    >= 0.88 (gait absorbs it); 3-4 W -> tilt 17-25 deg, height down to 0.79 (visible
    stumble, recovers within ~0.1 s); >= 5 W -> the fly rolls over and lies on its
    back (tilt 155-170 deg, thorax height ~0.55 mm, body on the ground) and the
    baseline controller never rights itself. 20 W can launch it into a full aerial
    roll that lands on its feet (tilt > 35 deg for 0.3 s while *airborne*) -> the
    ``near_ground_height`` gate keeps that from counting as a fall.
    """

    check_every_steps: int = 10  # evaluate every N physics steps (1 ms at dt=1e-4)

    # --- UPRIGHT -> DESTABILIZED (any of) ---
    destab_tilt_deg: float = 18.0  # normal max 7.4; 3 W shove -> 17-25
    destab_min_height: float = 0.90  # normal min 1.07; 3 W shove -> 0.82
    destab_max_height: float = 2.0  # thorax well above standing height -> airborne
    destab_hold_s: float = 0.005  # tiny debounce
    # After a reset the fly drops from FlyGym's neutral standing pose into its gait
    # (thorax height dips to ~0.79 mm around t=0.1 s, settles by ~0.3 s). DESTABILIZED
    # is suppressed for this long after (re)start; fall rules stay active.
    startup_grace_s: float = 0.4

    # --- DESTABILIZED -> UPRIGHT (all of, for stabilize_hold_s) ---
    stable_tilt_deg: float = 12.0
    stable_min_height: float = 0.95
    stable_max_height: float = 1.6
    stabilize_hold_s: float = 0.3

    # --- forward progress ---
    progress_window_s: float = 1.0
    progress_sample_s: float = 0.02
    stall_speed: float = 3.0  # mm/s; normal walking >= 14

    # --- -> FALLEN (any rule persisting for its hold time) ---
    # All tilt rules require the thorax to be near the ground; a fly tumbling in the
    # air after a big hit is DESTABILIZED until it lands.
    near_ground_height: float = 2.5
    fallen_tilt_deg: float = 60.0  # on its side / back
    fallen_tilt_hold_s: float = 0.15
    sustained_tilt_deg: float = 35.0  # strongly tilted for a long time
    sustained_tilt_hold_s: float = 0.5
    body_contact_tilt_deg: float = 30.0  # body on the ground and tilted ...
    body_contact_low_height: float = 0.7  # ... or body on the ground and low
    body_contact_hold_s: float = 0.25
    stuck_body_contact_hold_s: float = 1.0  # body on ground + no progress
    no_progress_hold_s: float = 3.0  # destabilized + no progress (e.g. wedged)

    # --- FALLEN -> RECOVERING -> UPRIGHT ---
    recover_tilt_deg: float = 20.0
    recover_min_height: float = 0.85
    recover_max_height: float = 2.0
    recovering_hold_s: float = 0.3  # on its feet this long -> RECOVERING
    recovered_confirm_s: float = 1.0  # ... and this long + walking -> recovered
    recovered_min_speed: float = 5.0  # mm/s over progress_window_s


@dataclass
class FallEvent:
    kind: str  # destabilized | stabilized | fall | recovering | recovered | relapse | reset
    time: float  # sim time (s)
    reason: str
    from_state: FallState
    to_state: FallState
    position: tuple[float, float, float]
    tilt_deg: float
    height: float
    # For "recovered": time since the fall event (s). None otherwise.
    recovery_time: float | None = None
    details: dict = field(default_factory=dict)


FallListener = Callable[[FallEvent], None]


class FallDetector:
    def __init__(
        self,
        sim: "Simulation | None" = None,
        cfg: FallDetectorConfig | None = None,
        ground_height_fn: Callable[[float, float], float] | None = None,
    ) -> None:
        self.cfg = cfg or FallDetectorConfig()
        self.ground_height_fn = ground_height_fn
        self.listeners: list[FallListener] = []
        self._classifier: ContactClassifier | None = None
        self.state = FallState.UPRIGHT
        self._reset_internal(0.0)
        if sim is not None:
            self.attach(sim)

    # ---------------------------------------------------------------- wiring
    def attach(self, sim: "Simulation") -> "FallDetector":
        """Register as post-step + reset hook of ``sim``."""
        self._classifier = ContactClassifier(sim.model, sim.fly_name)
        sim.post_step_hooks.append(self)
        sim.reset_hooks.append(self.on_reset)
        self._reset_internal(sim.time)
        return self

    def add_listener(self, fn: FallListener) -> None:
        self.listeners.append(fn)

    def on(self, kind: str, fn: Callable[[FallEvent], None]) -> None:
        """Convenience: call ``fn(event)`` only for events of one kind."""
        self.listeners.append(lambda ev: fn(ev) if ev.kind == kind else None)

    # ---------------------------------------------------------------- state
    def _reset_internal(self, t: float) -> None:
        self.state = FallState.UPRIGHT
        self.t_start = t
        self.fall_time: float | None = None
        self._since: dict[str, float] = {}  # condition name -> time it became true
        self._progress: deque[tuple[float, float, float]] = deque()
        self.last_signals: dict[str, float] = {}
        self.last_contacts: ContactSummary | None = None

    def on_reset(self, sim: "Simulation") -> None:
        prev = self.state
        self._reset_internal(sim.time)
        self._emit(sim, "reset", "sim reset", prev, FallState.UPRIGHT)

    def _held(self, name: str, cond: bool, t: float, hold: float) -> bool:
        """True once ``cond`` has been continuously true for ``hold`` seconds."""
        if not cond:
            self._since.pop(name, None)
            return False
        t0 = self._since.setdefault(name, t)
        return t - t0 >= hold

    def _progress_speed(self, t: float, x: float, y: float) -> float | None:
        """Horizontal speed over the progress window; None until the window is full."""
        c = self.cfg
        q = self._progress
        if not q or t - q[-1][0] >= c.progress_sample_s:
            q.append((t, x, y))
        while len(q) > 2 and t - q[1][0] >= c.progress_window_s:
            q.popleft()
        t0, x0, y0 = q[0]
        if t - t0 < c.progress_window_s * 0.95:
            return None
        return float(np.hypot(x - x0, y - y0) / (t - t0))

    # ---------------------------------------------------------------- hook
    def __call__(self, sim: "Simulation") -> None:
        if sim.step_count % self.cfg.check_every_steps:
            return
        self.update(sim)

    def update(self, sim: "Simulation") -> None:
        c = self.cfg
        if self._classifier is None:
            self._classifier = ContactClassifier(sim.model, sim.fly_name)
        t = sim.time
        tid = sim.thorax_body_id
        x, y, z = (float(v) for v in sim.data.xpos[tid])
        up_z = float(sim.data.xmat[tid, 8])  # R[2,2]: world-z component of body z
        tilt = float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))
        ground = self.ground_height_fn(x, y) if self.ground_height_fn else 0.0
        h = z - float(ground)
        contacts = self._classifier.summarize(sim.data)
        body = contacts.body_contact
        speed = self._progress_speed(t, x, y)
        stalled = speed is not None and speed < c.stall_speed
        self.last_contacts = contacts
        self.last_signals = dict(
            tilt=tilt, height=h, body_contact=float(body),
            legs_in_contact=float(contacts.legs_in_contact),
            progress_speed=float("nan") if speed is None else speed,
        )

        # ---- fall rules (evaluated every tick so hold timers stay accurate)
        near = h < c.near_ground_height
        fall_reason = None
        if self._held("tilt", near and tilt > c.fallen_tilt_deg, t, c.fallen_tilt_hold_s):
            fall_reason = "upside_down" if tilt > 120 else "on_side"
        if self._held("sustained", near and tilt > c.sustained_tilt_deg, t,
                      c.sustained_tilt_hold_s) and fall_reason is None:
            fall_reason = f"tilt>{c.sustained_tilt_deg:g}deg for {c.sustained_tilt_hold_s:g}s"
        if self._held("body", body and (tilt > c.body_contact_tilt_deg
                                        or h < c.body_contact_low_height),
                      t, c.body_contact_hold_s) and fall_reason is None:
            fall_reason = "body_ground_contact"
        if self._held("stuck", body and stalled, t, c.stuck_body_contact_hold_s) \
                and fall_reason is None:
            fall_reason = "stuck_on_body"
        if self._held("noprog", stalled and self.state == FallState.DESTABILIZED, t,
                      c.no_progress_hold_s) and fall_reason is None:
            fall_reason = "no_progress"

        destab_reasons = []
        if tilt > c.destab_tilt_deg:
            destab_reasons.append(f"tilt {tilt:.0f}deg")
        if h < c.destab_min_height:
            destab_reasons.append(f"low height {h:.2f}mm")
        if h > c.destab_max_height:
            destab_reasons.append(f"airborne {h:.2f}mm")
        if body:
            destab_reasons.append("body contact")
        if stalled:
            destab_reasons.append(f"stalled {speed:.1f}mm/s")
        stable = (tilt < c.stable_tilt_deg and c.stable_min_height < h < c.stable_max_height
                  and not body and not stalled)
        on_feet = (tilt < c.recover_tilt_deg and c.recover_min_height < h < c.recover_max_height
                   and not body)

        s = self.state
        if s in (FallState.UPRIGHT, FallState.DESTABILIZED):
            if fall_reason is not None:
                self.fall_time = t
                self._goto(sim, "fall", fall_reason, FallState.FALLEN)
            elif s == FallState.UPRIGHT:
                if t - self.t_start < c.startup_grace_s:
                    self._since.pop("destab", None)
                elif self._held("destab", bool(destab_reasons), t, c.destab_hold_s):
                    self._goto(sim, "destabilized", ", ".join(destab_reasons),
                               FallState.DESTABILIZED)
            elif self._held("stable", stable, t, c.stabilize_hold_s):
                self._goto(sim, "stabilized", "normal posture", FallState.UPRIGHT)
        elif s == FallState.FALLEN:
            if self._held("on_feet", on_feet, t, c.recovering_hold_s):
                self._goto(sim, "recovering", "back on its feet", FallState.RECOVERING)
        elif s == FallState.RECOVERING:
            if fall_reason is not None or not on_feet:
                self._goto(sim, "relapse", fall_reason or "lost footing", FallState.FALLEN)
            elif (self._held("on_feet", on_feet, t, c.recovered_confirm_s)
                  and speed is not None and speed >= c.recovered_min_speed):
                rt = t - self.fall_time if self.fall_time is not None else None
                self._goto(sim, "recovered", f"walking {speed:.1f}mm/s", FallState.UPRIGHT,
                           recovery_time=rt)
                self.fall_time = None

    def _goto(self, sim, kind: str, reason: str, new: FallState, **kw) -> None:
        prev = self.state
        self.state = new
        # Transition-scoped timers restart; fall-rule timers keep running so e.g. a
        # relapse is detected with the proper hold time.
        for k in ("destab", "stable"):
            self._since.pop(k, None)
        if kind == "relapse":
            self._since.pop("on_feet", None)
        self._emit(sim, kind, reason, prev, new, **kw)

    def _emit(self, sim, kind, reason, prev, new, recovery_time=None) -> None:
        sig = self.last_signals
        ev = FallEvent(
            kind=kind, time=sim.time, reason=reason, from_state=prev, to_state=new,
            position=tuple(float(v) for v in sim.data.xpos[sim.thorax_body_id]),
            tilt_deg=float(sig.get("tilt", sim.tilt_deg())),
            height=float(sig.get("height", float("nan"))),
            recovery_time=recovery_time,
            details={k: round(float(v), 4) for k, v in sig.items()},
        )
        for fn in list(self.listeners):
            fn(ev)
