"""Connectome brain -> body actions (``--brain-actions``; docs/ACTIONS.md §4).

``BrainActionTriggers.on_state(state, run_time)`` is called by
``perpetualfly.brain_link.BrainLink`` for every new ``BrainState`` (in the physics
thread) and triggers actions on an ``ActionManager``:

* giant fiber (``descending["escape"]`` = DNp01) above ``jump_escape_hz`` -> ``Jump``,
  then ``jump_refractory_s`` of fly time without another brain-triggered jump. The
  jump replaces any running action (escape has priority). With ``jump_short_hz``
  set, a GF rate at or above it launches a **short-mode** jump (no preparatory
  crouch; von Reyn et al. 2014: a strong early GF spike gives the fast short-mode
  take-off, a weaker one the slower long mode). ``jump_flight`` adds the escape
  flight emulation (``JumpParams.flight_assist``), directed away from the threat
  when ``threat_fn`` (set e.g. by ``install_swatter``) returns the threat's world
  position: the looming detectors are retinotopic, so the fly knows roughly where
  the looming object is (Card & Dickinson 2008: take-off direction is away from it).
  With ``flight`` set (``--flight``: a ``perpetualfly.flight.FlightMode``) the jump
  instead hands over to REAL flapping-wing flight at take-off and the fly flies away
  from the threat with the flight controller (no external force; ``jump_flight`` /
  ``flight_assist`` is then ignored). A giant-fibre burst while already airborne
  re-directs the escape flight instead of jumping.
* MN9 (``probes["MN9"]``) above ``proboscis_mn9_hz`` -> ``ProboscisExtend`` for
  ``proboscis_hold_s``. While MN9 stays high the running extension is prolonged
  (no re-blend). Needs the extra proboscis joints; never interrupts another action.
* DNg12 (``descending["groom"]``) above ``groom_hz`` over consecutive states that
  cover at least ``groom_sustain_s`` of brain time -> ``Groom``; never interrupts
  another action; ``groom_refractory_s`` after a groom ends.

Freeze is deliberately not mapped (no unambiguous freezing DN in the readout).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .base import ActionManager
from .behaviours import Groom, ProboscisExtend
from .jump import Jump
from .registry import available_actions


@dataclass
class TriggerParams:
    jump_escape_hz: float = 60.0
    jump_refractory_s: float = 1.5
    proboscis_mn9_hz: float = 30.0
    proboscis_hold_s: float = 0.5
    groom_hz: float = 20.0
    groom_sustain_s: float = 0.1
    groom_duration_s: float = 2.0
    groom_refractory_s: float = 1.0
    jump_short_hz: float | None = None  # GF >= this -> short-mode jump (None = never)
    jump_flight: bool = False  # escape flight after the jump (JumpParams.flight_assist)
    jump_overrides: dict | None = None  # extra JumpParams for brain-triggered jumps

    @classmethod
    def from_config(cls, cfg) -> "TriggerParams":
        return cls(**{k: getattr(cfg, k) for k in cls.__dataclass_fields__ if hasattr(cfg, k)})


class BrainActionTriggers:
    def __init__(self, mgr: ActionManager, params: TriggerParams | None = None,
                 say: Callable[[str], None] | None = None) -> None:
        self.mgr = mgr
        self.p = params or TriggerParams()
        self.say = say or (lambda msg: None)
        self.can_proboscis = "proboscis" in available_actions(mgr.sim)
        self._next_jump = -1e9
        self._next_groom = -1e9
        self._groom_run = 0.0  # brain seconds of consecutive above-threshold states
        self._groom_end = None  # (action object) to start the refractory when it ends
        self.counts = {"jump": 0, "proboscis": 0, "groom": 0}  # (+ "escape_flight")
        self.fired: list[tuple[float, str, float]] = []  # (run time, action, rate)
        self.jump_modes: list[str] = []  # mode of each brain-triggered jump
        # () -> world position (3,) of the looming threat, or None (escape direction)
        self.threat_fn: Callable[[], object] | None = None
        # perpetualfly.flight.FlightMode (--flight): real wing-powered escape flight
        self.flight = None
        self._flight_dir: tuple[float, float] | None = None
        mgr.listeners.append(self._on_event)

    def reset(self) -> None:
        self._next_jump = self._next_groom = -1e9
        self._groom_run = 0.0

    def _on_event(self, ev) -> None:
        if ev.name == "groom" and ev.kind in ("end", "cancel"):
            self._next_groom = self._now() + self.p.groom_refractory_s

    def _now(self) -> float:
        return self.mgr.sim.time

    def on_state(self, st, run_time: float | None = None) -> list[str]:
        """Check one BrainState; returns the messages for what was triggered.
        Refractory periods use sim time (the fly's clock)."""
        p, mgr = self.p, self.mgr
        now = self._now()
        msgs = []
        gf = float(st.descending.get("escape", 0.0) or 0.0)
        mn9 = float((st.probes or {}).get("MN9", 0.0) or 0.0)
        groom = float(st.descending.get("groom", 0.0) or 0.0)
        tag = f"t_brain={st.brain_time:.2f}s" if getattr(st, "brain_time", None) is not None else ""

        msgs += self._check_gf(gf, now, run_time, tag)
        msgs += self._check_mn9(mn9, now, run_time, tag)

        win = float(getattr(st, "window_s", 0.1) or 0.1)
        self._groom_run = self._groom_run + win if groom > p.groom_hz else 0.0
        if (self._groom_run >= p.groom_sustain_s - 1e-9 and now >= self._next_groom
                and mgr.trigger(Groom(duration=p.groom_duration_s), replace=False,
                                source="brain")):
            self._next_groom = now + p.groom_duration_s + p.groom_refractory_s
            self._groom_run = 0.0
            self._record(run_time, "groom", groom)
            msgs.append(f"[brain-action] {tag} DNg12 {groom:.0f} Hz > {p.groom_hz:g} for "
                        f">= {p.groom_sustain_s * 1e3:.0f} ms -> GROOM")
        for m in msgs:
            self.say(m)
        return msgs

    def on_fast(self, fe, run_time: float | None = None) -> list[str]:
        """A fast-path trigger (``perpetualfly.brain.schema.FastEvent`` from
        ``BrainConfig.fast_triggers``): the group's trailing-window rate crossed the
        threshold at spike time ``fe.brain_time``. Applies the same checks as
        ``on_state`` for that readout (giant fiber -> jump, MN9 -> proboscis)."""
        now = self._now()
        tag = f"t_brain={fe.brain_time:.3f}s fast"
        if fe.group == "escape":
            msgs = self._check_gf(float(fe.rate_hz), now, run_time, tag)
        elif fe.group == "MN9":
            msgs = self._check_mn9(float(fe.rate_hz), now, run_time, tag)
        else:
            msgs = []
        for m in msgs:
            self.say(m)
        return msgs

    def _check_gf(self, gf: float, now: float, run_time, tag: str) -> list[str]:
        p, mgr = self.p, self.mgr
        msgs = []
        flight = self.flight
        if (gf > p.jump_escape_hz and flight is not None and flight.airborne
                and now >= self._next_jump):
            if flight.state in ("hovering", "forward"):
                msg = flight.escape(self.escape_direction(), source="brain")
                self._next_jump = now + p.jump_refractory_s
                self._record(run_time, "escape_flight", gf)
                msgs.append(f"[brain-action] {tag} giant fiber {gf:.0f} Hz (airborne) -> {msg}")
        elif gf > p.jump_escape_hz and now >= self._next_jump and (
                flight is None or flight.state == "walking"):
            jump = self.make_jump(gf)
            mgr.trigger(jump, replace=True, source="brain")
            self._next_jump = now + p.jump_refractory_s
            self._record(run_time, "jump", gf)
            self.jump_modes.append(jump.p.mode)
            extra = " (short mode)" if jump.p.mode == "short" else ""
            if flight is not None:
                flight.takeoff(source="brain", jump=jump, escape_dir=self._flight_dir)
                extra += " -> wings at take-off, escape flight"
            msgs.append(f"[brain-action] {tag} giant fiber {gf:.0f} Hz > {p.jump_escape_hz:g} "
                        f"-> JUMP{extra}")
        return msgs

    def _check_mn9(self, mn9: float, now: float, run_time, tag: str) -> list[str]:
        p, mgr = self.p, self.mgr
        msgs = []
        if self.can_proboscis and mn9 > p.proboscis_mn9_hz:
            a = mgr.action if mgr.action is not None else mgr._pending
            if a is not None and a.name == "proboscis":
                # keep it out: extend the running action instead of re-blending
                if mgr.action is a:
                    a.duration = max(a.duration, now - mgr._t0 + p.proboscis_hold_s)
            elif mgr.trigger(ProboscisExtend(duration=p.proboscis_hold_s), replace=False,
                             source="brain"):
                self._record(run_time, "proboscis", mn9)
                msgs.append(f"[brain-action] {tag} MN9 {mn9:.0f} Hz > {p.proboscis_mn9_hz:g} "
                            "-> PROBOSCIS EXTENSION")
        return msgs

    def escape_direction(self) -> tuple[float, float] | None:
        """World (x, y) unit vector away from ``threat_fn()``; None without a threat
        position or with the threat straight overhead."""
        if self.threat_fn is None:
            return None
        threat = self.threat_fn()
        if threat is None:
            return None
        away = self.mgr.sim.thorax_position()[:2] - np.asarray(threat, dtype=float)[:2]
        n = float(np.hypot(*away))
        if n <= 0.3:
            return None
        return (float(away[0] / n), float(away[1] / n))

    def make_jump(self, gf_hz: float) -> Jump:
        """The Jump for a giant-fiber rate: short mode above ``jump_short_hz``,
        escape flight away from ``threat_fn()`` with ``jump_flight`` (emulation) or,
        with ``flight`` set, real flight after take-off (``_flight_dir``)."""
        p = self.p
        kw = dict(p.jump_overrides or {})
        if p.jump_short_hz is not None and gf_hz >= p.jump_short_hz:
            kw["mode"] = "short"
        self._flight_dir = None
        if self.flight is not None:
            # real wings take over at take-off: no emulated push
            kw.pop("flight_assist", None)
            kw.pop("escape_dir", None)
            d = self.escape_direction()
            if d is None:  # no threat position: straight ahead
                h = self.mgr.sim.heading()
                d = (float(np.cos(h)), float(np.sin(h)))
            self._flight_dir = d
        elif p.jump_flight:
            kw["flight_assist"] = True
            if "escape_dir" not in kw:
                d = self.escape_direction()
                if d is not None:
                    kw["escape_dir"] = d
        return Jump(**kw)

    def _record(self, run_time, name, rate) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1
        self.fired.append((self._now() if run_time is None else run_time, name, rate))
