"""Flight as an app mode (``--flight``; docs/FLIGHT.md section 7).

``FlightMode`` sits on a :class:`~perpetualfly.flight.sim.FlightSimulation` that
normally walks (leg mode ``"walk"``, wings still, spread and flat) and runs a small
state machine in a post-step hook:

    walking -> takeoff -> hovering / forward -> landing -> touchdown -> walking

* **take-off**: a ``Jump`` (through the session's ``ActionManager``). When the jump
  reaches its flight phase (end of the leg stroke) the jump hands back (``done``), the
  legs go to the flight pose (adhesion off), the wingbeat fades in over
  ``wing_ramp_s`` and a :class:`~perpetualfly.flight.control.HoverController` takes
  over with a set point ``climb_mm`` above the take-off COM. No external force at
  any point: lift, thrust and steering torques all come from the flapping wings in
  MuJoCo's fluid model.
* **hovering**: manual take-off (key L). Lands again after ``hover_s`` (None = wait
  for the key). While airborne, :meth:`steer` changes the forward speed / heading.
* **forward**: escape flight (``escape()``): the velocity target ramps to
  ``escape_speed`` along the escape direction (world, horizontal; away from the
  threat), the heading turns toward it at most ``turn_rate`` rad/s, for
  ``escape_s``, then brakes over ``brake_s`` and lands.
* **landing**: descend at ``land_speed`` until a leg touches something; **touchdown**:
  adhesion on, the wings keep beating ``land_pitch_s`` while the body pitches down to
  the walking posture, then they fade out and the gait controller takes over again
  (``controller.reset()``).
* **crash**: a body (non-leg) contact or a tilt above ``crash_tilt_deg`` while
  airborne stops the wings and hands the fly to the fall detector.

The altitude set point follows the terrain under the fly (``ground_height_fn``):
``clearance`` = COM height above the ground at take-off + ``climb_mm``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from perpetualfly.config import FlightModeConfig

from .control import HoverController, HoverGains, base_wingbeat

STATES = ("walking", "takeoff", "hovering", "forward", "landing", "touchdown")
AIRBORNE = ("hovering", "forward", "landing")


@dataclass
class FlightEvent:
    kind: str  # "takeoff" | "land" | "crash" | "abort" | "escape"
    time: float
    source: str
    info: dict = field(default_factory=dict)


class FlightMode:
    def __init__(self, sim, actions, cfg: FlightModeConfig | None = None, *,
                 ground_height_fn: Callable[[float, float], float] | None = None,
                 say: Callable[[str], None] | None = None) -> None:
        self.sim = sim
        self.actions = actions
        self.cfg = cfg or FlightModeConfig(enabled=True)
        self.ground_height_fn = ground_height_fn or (lambda x, y: 0.0)
        self.say = say or (lambda msg: None)
        self.listeners: list[Callable[[FlightEvent], None]] = []
        self.events: list[FlightEvent] = []
        self.counts = {"takeoff": 0, "land": 0, "crash": 0, "abort": 0, "escape": 0}
        self.airtime = 0.0  # s, summed over flights
        self.ctrl: HoverController | None = None
        self._reset_state()
        sim.leg_mode = "walk"
        sim.post_step_hooks.append(self._post_step)
        sim.reset_hooks.append(self._on_reset)

    # ------------------------------------------------------------------ state
    def _reset_state(self) -> None:
        self.state = "walking"
        self.source = ""
        self._jump = None
        self._t_state = self.sim.time
        self._t_takeoff = None
        self._escape_dir: np.ndarray | None = None
        self._yaw_goal = 0.0
        self._speed = 0.0
        self._clearance = 0.0
        self._stopping_at = None
        self._bad_since = None
        self._p_takeoff = None
        self._max_alt = 0.0
        self._fading = False  # wings fading out after a crash / abort (state walking)

    def _on_reset(self, sim) -> None:
        self._stop_wings(now=True)
        self._reset_state()
        sim.leg_mode = "walk"

    @property
    def airborne(self) -> bool:
        """Wings on (hovering, flying, landing, or the touchdown pitch-down)."""
        return self.state in AIRBORNE or self.state == "touchdown"

    @property
    def busy(self) -> bool:
        return self.state != "walking"

    @property
    def freq(self) -> float:
        return float(self.ctrl.freq) if self.ctrl is not None and self.sim.flapping else 0.0

    def altitude(self) -> float:
        c = self.sim.com()
        return float(c[2] - self.ground_height_fn(float(c[0]), float(c[1])))

    def speed(self) -> float:
        if self.ctrl is not None and self.ctrl.log:
            return float(np.hypot(*self.ctrl.log[-1][4:6]))
        return float(np.hypot(*self.sim.thorax_linvel()[:2]))

    def _set(self, state: str) -> None:
        self.state = state
        self._t_state = self.sim.time

    def _emit(self, kind: str, **info) -> None:
        ev = FlightEvent(kind, self.sim.time, self.source, info)
        self.events.append(ev)
        self.counts[kind] = self.counts.get(kind, 0) + 1
        for fn in list(self.listeners):
            fn(ev)

    # ------------------------------------------------------------------ commands
    def toggle(self, source: str = "key") -> str:
        """Key L: take off when walking, land when airborne."""
        if self.state == "walking":
            return self.takeoff(source=source)
        if self.state in ("hovering", "forward"):
            return self.land(source=source)
        return f"[flight] {self.state} in progress"

    def takeoff(self, source: str = "key", jump=None, escape_dir=None) -> str:
        """Start a take-off: ``jump`` (already triggered on the ActionManager, e.g. by
        the brain) or a new manual jump; ``escape_dir`` (world x, y) = escape flight
        in that direction after take-off, None = hover."""
        if self.state != "walking":
            if escape_dir is not None and self.state in ("hovering", "forward"):
                return self.escape(escape_dir, source)
            return f"[flight] can't take off: {self.state}"
        if jump is None and self.sim.tilt_deg() > self.cfg.max_takeoff_tilt_deg:
            return (f"[flight] can't take off: the fly is not upright "
                    f"(tilt {self.sim.tilt_deg():.0f} deg)")
        if jump is None:
            from perpetualfly.actions.jump import Jump

            jump = Jump(**dict(self.cfg.manual_jump))
            self.actions.trigger(jump, source=source)
        self._jump = jump
        self.source = source
        self._escape_dir = self._unit(escape_dir)
        self._yaw_goal = self.sim.heading()
        self._set("takeoff")
        where = "" if self._escape_dir is None else (
            f", escape toward {math.degrees(math.atan2(*self._escape_dir[::-1])):.0f} deg")
        return f"[flight] take-off ({source}): {jump.p.mode}-mode jump -> wings{where}"

    def escape(self, direction, source: str = "brain") -> str:
        """Airborne: fly off in ``direction`` (world x, y) for ``escape_s``."""
        d = self._unit(direction)
        if d is None:
            d = np.array([math.cos(self._yaw_goal), math.sin(self._yaw_goal)])
        if self.state not in ("hovering", "forward"):
            return f"[flight] can't escape: {self.state}"
        self._escape_dir = d
        self.source = source
        self._start_forward()
        return f"[flight] escape flight toward {math.degrees(math.atan2(d[1], d[0])):.0f} deg"

    def land(self, source: str = "key") -> str:
        if self.state not in ("hovering", "forward"):
            return f"[flight] can't land: {self.state}"
        self.source = source
        self._begin_landing()
        return "[flight] landing"

    def steer(self, key: str) -> str | None:
        """Arrow keys while airborne: UP / DOWN forward speed, LEFT / RIGHT heading.
        Returns None when not airborne (the key keeps its normal meaning)."""
        if self.state not in ("hovering", "forward") or self._escape_dir is not None:
            return None
        c = self.cfg
        if key in ("up", "down"):
            self._speed = float(np.clip(self._speed + (c.speed_step if key == "up" else -c.speed_step),
                                        -c.max_speed, c.max_speed))
            self._set("forward" if self._speed != 0 else "hovering")
        elif key in ("left", "right"):
            self._yaw_goal += math.radians(c.turn_step_deg) * (1 if key == "left" else -1)
        else:
            return None
        return (f"[flight] speed {self._speed:+.0f} mm/s, heading "
                f"{math.degrees(self._yaw_goal):.0f} deg")

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _unit(v) -> np.ndarray | None:
        if v is None:
            return None
        v = np.asarray(v, dtype=float)[:2]
        n = float(np.hypot(*v))
        return v / n if n > 1e-9 else None

    def _start_wings(self) -> None:
        sim, c = self.sim, self.cfg
        sim.leg_mode = "flight"
        sim.wingbeat.params = base_wingbeat()
        sim.wingbeat.start(c.wing_ramp_s)
        sim.flapping = True
        self._gains = HoverGains(max_accel_xy=c.max_accel_xy)
        ctrl = HoverController(sim, self._gains)
        ctrl.log = deque(maxlen=500)  # 0.1 s at 5 kHz is plenty for the HUD
        com = sim.com()
        climb = c.escape_climb_mm if self._escape_dir is not None else c.climb_mm
        self._clearance = self.altitude() + climb
        if self._escape_dir is not None and c.escape_altitude_mm is not None:
            # escape low: the paddle reaches a high fly before it reaches the ground
            climb += c.escape_altitude_mm - self._clearance
            self._clearance = c.escape_altitude_mm
        ctrl.target.pos = com + np.array([0.0, 0.0, climb])
        ctrl.target.yaw = self._yaw_goal
        sim.flight_controller = ctrl
        self.ctrl = ctrl
        self._t_takeoff = sim.time
        self._p_takeoff = com
        self._max_alt = 0.0

    def _stop_wings(self, now: bool = False) -> None:
        """Fade the wingbeat out (``wing_stop_s``); ``now``: switch off at once --
        only safe when the wings are already still (after a fade, after a reset):
        a step of the servo target at 2000 rad/s wing speed blows the model up."""
        sim = self.sim
        sim.flight_controller = None
        if now or not sim.flapping or sim.wingbeat.stopped:
            sim.flapping = False
            sim.data.ctrl[sim.wing_act] = 0.0
            sim.wingbeat.reset()
            self.ctrl = None
        else:
            sim.wingbeat.stop(self.cfg.wing_stop_s)
            self._stopping_at = sim.time

    def _start_forward(self) -> None:
        self._speed = self.cfg.escape_speed
        if self.ctrl is not None and self.cfg.escape_velocity_mode:
            self.ctrl.g = replace(self._gains, xy_zeta=self.cfg.escape_xy_zeta)
        self._set("forward")

    def _begin_landing(self) -> None:
        if self.ctrl is not None:
            self.ctrl.g = self._gains
            self.ctrl.target.vel = np.array([0.0, 0.0, -self.cfg.land_speed])
            self.ctrl.target.pos = self.sim.com()
        self._escape_dir = None
        self._speed = 0.0
        self._set("landing")

    def _to_walking(self, kind: str, **info) -> None:
        sim = self.sim
        if self._t_takeoff is not None:
            self.airtime += sim.time - self._t_takeoff
            info.setdefault("airtime_s", round(sim.time - self._t_takeoff, 3))
            if self._p_takeoff is not None:
                info.setdefault("distance_mm", round(float(np.hypot(
                    *(sim.com()[:2] - self._p_takeoff[:2]))), 1))
            info.setdefault("max_altitude_mm", round(self._max_alt, 2))
        self._stop_wings()  # (crash / abort: the wings fade out while walking)
        fading = sim.flapping
        sim.leg_mode = "walk"
        sim.controller.reset()
        self._emit(kind, tilt_deg=round(sim.tilt_deg(), 1), **info)
        self._reset_state()
        self._fading = fading

    def _post_step(self, sim) -> None:
        st = self.state
        if st == "walking":
            if self._fading and sim.wingbeat.stopped:
                self._fading = False
                self._stop_wings(now=True)
            return
        if st == "takeoff":
            self._update_takeoff()
            return
        if sim.step_count % 4:  # the rest at 5 kHz, like the controller
            return
        t = sim.time - self._t_state
        c = self.cfg
        ctrl = self.ctrl
        if self._stopping_at is not None:  # wings fading out after the touchdown
            if sim.time - self._stopping_at >= c.wing_stop_s + 2e-3:
                self._stopping_at = None
                self._to_walking("land")
            return
        if ctrl is None:
            return
        self._max_alt = max(self._max_alt, self.altitude())
        if st in AIRBORNE and self._crashed():
            self.say(f"[flight] crash at t={sim.time:.2f}s (tilt {sim.tilt_deg():.0f} deg)")
            self._to_walking("crash")
            return
        dt = 4 * sim.timestep
        if st in ("hovering", "forward"):
            # terrain following: keep the take-off clearance above the local ground
            com = sim.com()
            z_goal = self.ground_height_fn(float(com[0]), float(com[1])) + self._clearance
            dz = float(np.clip(z_goal - ctrl.target.pos[2], -50.0 * dt, 50.0 * dt))
            ctrl.target.pos = ctrl.target.pos + np.array([0.0, 0.0, dz])
            tgt = ctrl.target
            if self._escape_dir is not None:
                d = self._escape_dir
                self._yaw_goal = self._escape_yaw(d)
                ramp = min(1.0, t / c.escape_ramp_s) if c.escape_ramp_s > 0 else 1.0
                if t >= c.escape_s:
                    ramp = max(0.0, 1.0 - (t - c.escape_s) / c.brake_s)
                    if t >= c.escape_s + c.brake_s:
                        self._begin_landing()
                        return
                v = d * self._speed * ramp
                if c.escape_velocity_mode:
                    # pure velocity control: wherever the jump threw the fly is fine
                    # (a position loop would pull it back under the threat)
                    tgt.pos = np.r_[com[:2], tgt.pos[2]]
                    ctrl.xy_int[:] = 0.0
            else:
                v = self._speed * np.array([math.cos(self._yaw_goal), math.sin(self._yaw_goal)])
                if st == "hovering" and c.hover_s is not None and t >= c.hover_s:
                    self.source = "auto"
                    self._begin_landing()
                    return
            tgt.vel = np.array([v[0], v[1], 0.0])
            # heading: turn toward the goal at most turn_rate
            err = (self._yaw_goal - tgt.yaw + math.pi) % (2 * math.pi) - math.pi
            tgt.yaw += float(np.clip(err, -c.turn_rate * dt, c.turn_rate * dt))
        elif st == "landing":
            legs, _ = self.actions.body.ground_contacts(sim)
            if legs.any():
                sim.leg_mode = "stance"
                ctrl.target.vel = np.zeros(3)
                ctrl.target.pos = sim.com() - np.array([0.0, 0.0, 0.3])
                ctrl.target.pitch = math.radians(sim.stroke_plane_deg)
                self._set("touchdown")
            elif t > c.max_land_s:
                self.say("[flight] no touchdown: wings stopped")
                self._to_walking("abort", reason="no touchdown")
        elif st == "touchdown":
            if t >= c.land_pitch_s:
                self._stop_wings()

    def _update_takeoff(self) -> None:
        sim, mgr, j = self.sim, self.actions, self._jump
        running = mgr.action is j or mgr._pending is j
        if not running:
            # the jump was replaced / cancelled / ended before its flight phase
            self._emit("abort", reason="jump ended before take-off")
            self._reset_state()
            return
        if mgr.action is j and j.phase(sim.time - mgr._t0) == "flight":
            tilt = sim.tilt_deg()
            if tilt > self.cfg.max_wings_tilt_deg:
                # tumbling (e.g. knocked over by a hit while jumping): wings would
                # only crash it -- the jump carries on as a plain jump
                self._emit("abort", reason="tumbling at take-off", tilt_deg=round(tilt, 1))
                self._reset_state()
                return
            j.done = True  # natural end: the manager blends the legs to the flight pose
            self._start_wings()
            self._emit("takeoff", mode=j.p.mode, tilt_deg=round(sim.tilt_deg(), 1),
                       vz_mm_s=round(float(sim.thorax_linvel()[2]), 0),
                       escape=self._escape_dir is not None)
            if self._escape_dir is not None:
                self._start_forward()
            else:
                self._set("hovering")

    def _escape_yaw(self, d) -> float:
        """Heading for an escape along ``d``: turn toward it when that is at most
        ``max_turn_deg`` away from the current heading target, else face the other
        way and fly backward (a threat ahead: the fly backs away from it, as real
        flies take off backward from a frontal looming stimulus)."""
        esc = math.atan2(d[1], d[0])
        ref = self.ctrl.target.yaw if self.ctrl is not None else self._yaw_goal
        err = (esc - ref + math.pi) % (2 * math.pi) - math.pi
        if abs(err) <= math.radians(self.cfg.max_turn_deg):
            return esc
        return esc + math.pi if err < 0 else esc - math.pi

    def _crashed(self) -> bool:
        sim, c = self.sim, self.cfg
        if sim.time - self._t_takeoff < c.crash_grace_s:
            return False
        _, body = self.actions.body.ground_contacts(sim)
        bad = body or sim.tilt_deg() > c.crash_tilt_deg
        if not bad:
            self._bad_since = None
            return False
        if self._bad_since is None:
            self._bad_since = sim.time
        return sim.time - self._bad_since > 0.05

    # ------------------------------------------------------------------ display
    def hud_line(self) -> str:
        label = {"forward": "FORWARD" if self._escape_dir is None else "ESCAPE",
                 "touchdown": "LANDING"}.get(self.state, self.state.upper())
        if self.state == "walking":
            return "FLIGHT walking  (L: take off)"
        line = f"FLIGHT {label}"
        if self.sim.flapping:
            line += (f"  {self.freq:3.0f} Hz  alt {self.altitude():4.1f} mm  "
                     f"v {self.speed():4.0f} mm/s")
        if self.state in ("hovering", "forward") and self._escape_dir is None:
            line += "  (L: land, arrows: steer)"
        return line

    def summary(self) -> dict:
        return {"counts": dict(self.counts), "airtime_s": round(self.airtime, 3),
                "state": self.state}
