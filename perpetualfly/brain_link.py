"""Glue between the fly app and the connectome brain (``--brain`` / ``--brain-steer``).

``BrainLink`` owns the brain worker process (``perpetualfly.brain.BrainProcess``,
paced to the fly's simulated time) and, optionally, the brain window process
(``perpetualfly.brain_viz.BrainWindowProcess``). ``Session`` wires it up:

* body -> brain: whip hits and shoves (intensity = measured impulse / the absurd
  level's impulse, side = where the fly was hit, in the fly's frame, body = the
  body that took most of the impulse), falls, recoveries, resets and pauses become
  ``StimulusEvent``s stamped with the fly's run time; the brain applies each at that
  time. The same events go to the brain window (instant stimulus chips) and to
  events.csv (``brain_stim`` rows). Keys O / T / K inject a looming stimulus (LC4),
  sugar and bitter taste.
* brain -> body: every ``BrainState`` is mapped to a CPG drive with
  ``mapping.descending_to_drive``, held and low-pass filtered (``drive_tau_s``) in
  sim time. With ``steer`` on, the walking controller's heading-hold signal is
  combined with it (``combine_drive``); without it the drive is only displayed.
* clock: after every physics chunk ``update()`` sends the fly run time to the
  brain, which never runs ahead of it (so pausing the fly pauses the brain). A
  brain that falls behind lags (shown in the HUD, capped at ``max_lag_s``), it
  never blocks physics.

Threading: every method is called from the physics side (hooks / listeners /
``update``) or from the main thread under ``runner.locked()``, never concurrently.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from perpetualfly.brain.mapping import DriveGains, descending_to_drive
from perpetualfly.brain.schema import DESCENDING_GROUPS, BrainState, StimulusEvent

if TYPE_CHECKING:
    from perpetualfly.app import Session

FETCH_COMMAND = ".venv/bin/python scripts/fetch_brain_data.py"
INSTALL_COMMAND = '.venv/bin/python -m pip install -e ".[brain]"'

# keys (active only with --brain / --brain-headless): key -> (set, label, help)
BRAIN_KEYS: dict[str, tuple[str, str, str]] = {
    "o": ("loom", "LOOM", "brain: looming shadow (LC4 looming detectors -> giant fiber + "
                          "MDN backward-walking neurons; with --brain-steer the fly stops / backs "
                          "up, with --brain-actions the giant fiber makes it jump)"),
    "t": ("sugar", "SUGAR", "brain: sugar taste (sugar GRNs -> MN9 proboscis motor neuron; "
                            "with --brain-actions + --full-body the proboscis extends)"),
    "k": ("bitter", "BITTER", "brain: bitter taste (bitter GRNs; display only, no body effect)"),
}


@dataclass
class BrainLinkConfig:
    enabled: bool = False  # --brain / --brain-headless
    window: bool = True  # brain window (--brain; off with --brain-headless / --no-brain-window)
    steer: bool = False  # --brain-steer: brain descending drive modulates walking
    data_dir: str | None = None  # default <repo>/data/brain
    # ---- body -> brain ---------------------------------------------------------
    # Intensity 1 = the absurd level's impulse: whip 1.0 uN*s (mean measured impulse
    # of L4 cracks, docs/WHIP.md); shove None = level 4's force x duration.
    whip_ref_impulse_uNs: float = 1.0
    shove_ref_impulse_uNs: float | None = None
    hit_duration_s: float = 0.05  # sensory drive per hit (brain s; >= the contact)
    fall_duration_s: float = 0.1
    loom_sets: tuple[str, ...] = ("LC4",)  # looming detectors driven by O
    loom_duration_s: float = 1.0
    taste_duration_s: float = 1.0
    # ---- brain -> body ----------------------------------------------------------
    drive_tau_s: float = 0.1  # low-pass of the brain drive (sim s)
    stale_after_s: float = 2.0  # no state covering the last N sim s -> drive [1, 1]
    gains: dict = field(default_factory=dict)  # DriveGains overrides
    # --brain-backup: MDN (moonwalker) rate that fully reverses stepping is lowered
    # from DriveGains.backward_ref (40 Hz) to backup_ref_hz, so looming (O, with
    # steering on) makes the fly walk backward instead of just stopping. Overrides
    # gains["backward_ref"]. See docs/BRAIN.md ("Looming with the real brain").
    backup: bool = False
    backup_ref_hz: float = 20.0
    # ---- process ------------------------------------------------------------------
    clock_every_s: float = 0.01  # sim s between clock marks sent to the brain
    max_lag_s: float = 1.0  # brain gives up catching up beyond this lag
    window_s: float = 0.1  # brain time per BrainState
    # Low-latency pacing (docs/SWATTER.md). A state covering fly time T normally
    # reaches the body one physics chunk after the chunk containing T (the brain can
    # only run to T once that chunk's clock mark is sent, and it is polled after the
    # next chunk). With sync_wait_s > 0, update() waits up to that much wall time
    # after sending the clock for the brain to publish the window ending in the last
    # window_s, removing that chunk of latency at the cost of wall time. With
    # sync_loom_only it only waits while a visual loom stimulus is active (sent in
    # the last loom event's duration), i.e. when an escape decision may be pending.
    # Pair with a short window_s (0.01-0.02 s; the GF rate is read per window).
    sync_wait_s: float = 0.0
    sync_loom_only: bool = True
    # ---- brain -> actions (--brain-actions; docs/ACTIONS.md) ----------------------
    actions: bool = False  # descending readouts trigger body actions
    jump_escape_hz: float = 60.0  # giant fiber (DNp01) rate that fires a Jump
    jump_refractory_s: float = 1.5  # sim s after a brain-triggered jump
    proboscis_mn9_hz: float = 30.0  # MN9 rate -> ProboscisExtend (needs extra joints)
    proboscis_hold_s: float = 0.5  # each trigger holds the proboscis out this long
    groom_hz: float = 20.0  # DNg12 group rate -> Groom ...
    groom_sustain_s: float = 0.1  # ... sustained this long (brain states cover 0.1 s)
    groom_duration_s: float = 2.0
    groom_refractory_s: float = 1.0
    synthetic: dict | None = None  # tests: tiny random network, no data needed

    @classmethod
    def from_dict(cls, d: dict) -> "BrainLinkConfig":
        d = dict(d)
        if isinstance(d.get("loom_sets"), list):
            d["loom_sets"] = tuple(d["loom_sets"])
        return cls(**d)


def missing_requirements(cfg: BrainLinkConfig) -> str | None:
    """None if the brain can run, else a message with the exact commands to fix it."""
    if cfg.synthetic:
        return None
    missing = [m for m in ("numba", "pyarrow", "scipy") if importlib.util.find_spec(m) is None]
    if missing:
        return (f"--brain needs the 'brain' extra ({', '.join(missing)} not installed).\n"
                f"  {INSTALL_COMMAND}\n  {FETCH_COMMAND}")
    from perpetualfly.brain.data import DEFAULT_DATA_DIR, data_available

    d = Path(cfg.data_dir) if cfg.data_dir else DEFAULT_DATA_DIR
    if not data_available(d):
        return (f"--brain needs the FlyWire connectome data in {d}/ (not found).\n"
                f"Fetch it (~153 MB download, checksummed) with:\n  {FETCH_COMMAND}")
    return None


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def combine_drive(hold: np.ndarray, brain: np.ndarray, max_amp: float = 1.5) -> np.ndarray:
    """Final ``[left, right]`` CPG drive from heading hold + brain drive.

    ``hold = [1 + d, 1 - d]`` is the heading-hold P-controller (d = its turn
    differential), ``brain = [bL, bR]`` the (smoothed) brain drive (quiet brain =
    [1, 1]; sign = stepping direction). The brain sets speed/direction and its own
    turn; the heading hold is added as a corrective differential, faded out as the
    brain slows or reverses the stepping::

        w     = clip(mean(brain), 0, 1)
        final = brain + w * d * [+1, -1]          (each side clipped to +/-max_amp)

    For a forward-walking brain (mean >= 1) this is exactly ``hold + (brain - 1)``;
    a quiet brain gives back ``hold`` unchanged. When MDN drives stepping toward -1
    the heading correction (tuned for forward walking; its sign is not meaningful
    for reversed stepping) goes to zero.
    """
    bl, br = float(brain[0]), float(brain[1])
    d = 0.5 * (float(hold[0]) - float(hold[1]))
    w = min(max(0.5 * (bl + br), 0.0), 1.0)
    left = min(max(bl + w * d, -max_amp), max_amp)
    right = min(max(br - w * d, -max_amp), max_amp)
    return np.array([left, right])


def hit_side(direction_world, heading_rad: float) -> str:
    """Side of the fly that was hit ("left"/"right"/"front"/"rear"/"top"/"none") from
    the unit push direction (world frame) and the fly's heading: the contact is on
    the side opposite to the push (a push to the fly's right = hit on its left)."""
    v = np.asarray(direction_world, dtype=float)
    if not np.all(np.isfinite(v)) or float(np.linalg.norm(v)) < 1e-9:
        return "none"
    c, s = math.cos(heading_rad), math.sin(heading_rad)
    fx = v[0] * c + v[1] * s  # along the fly's forward axis
    fy = -v[0] * s + v[1] * c  # along the fly's left axis
    fz = v[2]
    if abs(fz) > max(abs(fx), abs(fy)):
        return "top" if fz < 0 else "none"
    if abs(fy) >= abs(fx):
        return "left" if fy < 0 else "right"
    return "front" if fx < 0 else "rear"


def drive_display(brain: np.ndarray, escape_hz: float, applied: bool,
                  gains: DriveGains) -> dict:
    """``BrainState.drive`` dict for the brain window's DRIVE panel (the app adds
    ``lag_s``, the brain's lag behind the fly, for the header badge)."""
    bl, br = float(brain[0]), float(brain[1])
    return {
        "left": bl, "right": br,
        "forward": float(np.clip(0.5 * (bl + br), -1.0, 1.0)),
        # + = turn left: the controller turns toward the smaller amplitude
        "turn": float(np.clip(-0.5 * (bl - br) / max(gains.turn_gain, 1e-9), -1.0, 1.0)),
        "escape": float(np.clip(escape_hz / 60.0, 0.0, 1.0)),
        "applied": bool(applied),
    }


def render_brain_frame(layout, states: list, window_s: float = 0.1,
                       size: tuple[int, int] | None = None):
    """BGR brain-window frame for ``states`` (oldest first) via the brain window's
    headless ``render_frame``; None without layout / states."""
    if layout is None or not states:
        return None
    from perpetualfly.brain_viz.window import DEFAULT_SIZE, render_frame

    return render_frame(layout, list(states), size=size or DEFAULT_SIZE, interval=window_s)


# ---------------------------------------------------------------------------
# BrainLink
# ---------------------------------------------------------------------------


# whip crack side (where it comes from) -> side of the fly that is hit
WHIP_SIDE_TO_HIT_SIDE = {"left": "left", "right": "right", "front": "front", "rear": "rear",
                         "overhead": "top"}

METRIC_COLUMNS = (["brain_time", "brain_lag", "brain_drive_L", "brain_drive_R",
                   "ctrl_drive_L", "ctrl_drive_R"]
                  + [f"dn_{g}" for g in DESCENDING_GROUPS] + ["mn9_hz"])


class BrainLink:
    def __init__(self, cfg: BrainLinkConfig, *, headless: bool = False,
                 say: Callable[[str], None] | None = None, start: bool = True) -> None:
        self.cfg = cfg
        self.say = say or (lambda msg: print(msg, flush=True))
        gains = dict(cfg.gains)
        if cfg.backup:
            gains["backward_ref"] = float(cfg.backup_ref_hz)
        self.gains = DriveGains(**gains)
        self.show_window = bool(cfg.window and not headless)
        from perpetualfly.brain.process import BrainConfig

        self.brain_cfg = BrainConfig(
            data_dir=cfg.data_dir, window_s=cfg.window_s, pace="sim",
            max_lag_s=cfg.max_lag_s, subscribers=("app",), synthetic=cfg.synthetic)
        self.brain = None
        self.window = None
        self.layout = None
        self.info: dict = {}
        self.session: "Session | None" = None
        self.triggers = None  # BrainActionTriggers (--brain-actions), set in attach()
        # state
        self.latest: BrainState | None = None
        # last few states (with .drive set): screenshots render a brain frame from them
        self.recent: deque[BrainState] = deque(maxlen=30)
        self.n_states = 0
        self.n_dropped = 0
        self.n_sent = 0
        self._last_seq = 0
        self._target = np.ones(2)
        self.drive = np.ones(2)  # smoothed brain drive
        self._w = 1.0
        self._last_rt: float | None = None
        self._last_clock = -1e9
        self._dead_reported = False
        self._window_closed_reported = False
        self._flags: dict[str, bool] = {}
        self.stim_log: list[StimulusEvent] = []  # everything sent (tests / summary)
        self._loom_until = -1e9  # run time the last visual loom stimulus ends
        self.n_sync_waits = 0
        self.sync_wait_wall_s = 0.0
        if start:
            self.start()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        from perpetualfly.brain.process import BrainProcess

        if self.brain is None:
            self.brain = BrainProcess(self.brain_cfg)

    def wait_ready(self, timeout: float = 120.0) -> dict:
        self.info = self.brain.wait_ready(timeout)
        self.layout = self.brain.layout(timeout)
        return self.info

    def start_window(self) -> None:
        if not self.show_window or self.window is not None or self.layout is None:
            return
        from perpetualfly.brain_viz import BrainWindowProcess

        self.window = BrainWindowProcess(self.layout).start()

    @property
    def pids(self) -> list[int]:
        out = []
        if self.brain is not None and self.brain.pid:
            out.append(self.brain.pid)
        if self.window is not None and self.window.process.pid:
            out.append(self.window.process.pid)
        return out

    def close(self) -> None:
        if self.session is not None and self.cfg.steer:
            self.session.sim.controller.signal_filter = None
        if self.window is not None:
            try:
                self.window.close()
            except Exception as e:  # never mask the real error
                print(f"warning: closing the brain window failed: {e}", file=sys.stderr)
        if self.brain is not None:
            try:
                self.brain.stop()
            except Exception as e:
                print(f"warning: stopping the brain failed: {e}", file=sys.stderr)

    # ------------------------------------------------------------- wiring
    def attach(self, session: "Session") -> "BrainLink":
        """Subscribe to the session's hits / falls / resets; install steering."""
        self.session = session
        session.perturbation.listeners.append(self.on_shove)
        if session.whip is not None:
            session.whip.listeners.append(self.on_whip)
        session.detector.add_listener(self.on_fall_event)
        session.sim.reset_hooks.append(self._on_sim_reset)
        self.triggers = None
        if self.cfg.actions and getattr(session, "actions", None) is not None:
            from perpetualfly.actions.brain_triggers import BrainActionTriggers, TriggerParams

            self.triggers = BrainActionTriggers(session.actions, TriggerParams.from_config(self.cfg),
                                                say=self.say)
            if not self.triggers.can_proboscis:
                self.say("[brain] --brain-actions: no proboscis joints (run with --full-body) "
                         "-> MN9 has no body effect")
        if self.cfg.steer:
            if session.cfg.controller.kind != "hybrid":
                self.say("[brain] warning: --brain-steer needs the hybrid controller "
                         "(the cpg controller has no descending input); steering is off")
            else:
                session.sim.controller.signal_filter = self._steer_filter
        return self

    def _run_time(self, sim_time: float | None = None) -> float:
        s = self.session
        if s is None:
            return 0.0 if sim_time is None else float(sim_time)
        return float(s.metrics.run_time_at(s.sim.time if sim_time is None else sim_time))

    # ------------------------------------------------------------- body -> brain
    def send(self, ev: StimulusEvent, source: str = "") -> None:
        """Brain + window + events.csv. ``ev.sim_time`` must be the fly run time."""
        self.stim_log.append(ev)
        self.n_sent += 1
        if ev.kind == "loom" and ev.sim_time is not None:
            self._loom_until = max(self._loom_until, float(ev.sim_time) + float(ev.duration_s))
        if self.brain is not None and self.brain.is_alive():
            try:
                self.brain.send(ev)
            except Exception:
                pass
        if self.window is not None:
            self.window.send(ev)
        s = self.session
        if s is not None and s.logger is not None:
            s.logger.log_event("brain_stim", force_direction=ev.side, details={
                "kind": ev.kind, "side": ev.side, "intensity": round(ev.intensity, 4),
                "duration_s": ev.duration_s, "stim_time": ev.sim_time, "source": source,
                **{k: v for k, v in (ev.details or {}).items()}})

    def _heading(self) -> float:
        return self.session.sim.heading() if self.session is not None else 0.0

    def on_whip(self, ev) -> None:
        if not ev.hit:
            return
        ref = max(self.cfg.whip_ref_impulse_uNs, 1e-12)
        # The crack's side is set in the fly's frame when it starts; the event arrives
        # after the strike, when a tumbling fly's heading may have flipped, so only
        # "random" cracks are classified from the measured impulse.
        side = WHIP_SIDE_TO_HIT_SIDE.get(ev.side) or (
            hit_side(ev.direction, self._heading()) if ev.impulse_uNs > 0 else "none")
        self.send(StimulusEvent(
            "whip_hit", side=side, intensity=float(min(ev.impulse_uNs / ref, 1.0)),
            duration_s=max(self.cfg.hit_duration_s, float(ev.duration_s)),
            sim_time=self._run_time(ev.sim_time),
            details={"body": ev.body, "impulse_uNs": float(ev.impulse_uNs),
                     "level": ev.level, "from": ev.direction_name}), source=ev.source)

    def _shove_ref(self) -> float:
        if self.cfg.shove_ref_impulse_uNs:
            return self.cfg.shove_ref_impulse_uNs
        p = self.session.perturbation
        lv = p.cfg.levels[-1]
        return lv.magnitude_bw * p.body_weight_uN * lv.duration_s

    def on_shove(self, ev) -> None:
        self.send(StimulusEvent(
            "shove", side=hit_side(ev.direction, self._heading()),
            intensity=float(min(ev.impulse_uNs / max(self._shove_ref(), 1e-12), 1.0)),
            duration_s=max(self.cfg.hit_duration_s, float(ev.duration_s)),
            sim_time=self._run_time(ev.sim_time),
            details={"body": ev.body, "impulse_uNs": float(ev.impulse_uNs),
                     "level": ev.level, "push": ev.direction_name}), source=ev.source)

    def on_fall_event(self, ev) -> None:
        if ev.kind == "fall":
            self.send(StimulusEvent("fall", "none", 1.0, self.cfg.fall_duration_s,
                                    self._run_time(ev.time),
                                    details={"reason": ev.reason}), source="detector")
        elif ev.kind == "recovered":
            # no sensory mapping (marker in the window / events.csv only)
            self.send(StimulusEvent("recovered", "none", 0.0, 0.0, self._run_time(ev.time),
                                    details={"label": "RECOVERED"}), source="detector")

    def _on_sim_reset(self, sim) -> None:
        """Fly reset -> brain reset too (all neurons to rest, stimuli cleared): the
        reset teleports the body, so leftover activity would not belong to it."""
        rt = self._run_time()
        if self.brain is not None and self.brain.is_alive():
            self.brain.reset_state(sim_time=rt)
        if self.window is not None:
            self.window.send(StimulusEvent("reset", "none", 0.0, 0.0, rt))
        self.drive[:] = 1.0
        self._target[:] = 1.0
        self._w = 1.0
        if getattr(self, "triggers", None) is not None:
            self.triggers.reset()
        s = self.session
        if s is not None and s.logger is not None:
            s.logger.log_event("brain_reset", details={"stim_time": rt})

    def on_pause(self, paused: bool) -> None:
        """Pausing needs nothing from the brain (no clock marks -> it waits); the
        window gets a chip and events.csv a row."""
        rt = self._run_time()
        if self.window is not None:
            self.window.send(StimulusEvent("pause", "none", 0.0, 0.0, rt,
                                           details={"label": "PAUSED" if paused else "RESUMED"}))
        s = self.session
        if s is not None and s.logger is not None:
            s.logger.log_event("brain_pause" if paused else "brain_resume")

    def manual(self, what: str) -> StimulusEvent:
        rt = self._run_time()
        if what == "loom":
            evs = [StimulusEvent("manual", "none", 1.0, self.cfg.loom_duration_s, rt,
                                 details={"set": name, "label": f"LOOM {name}"})
                   for name in self.cfg.loom_sets]
        elif what in ("sugar", "bitter"):
            evs = [StimulusEvent("manual", "none", 1.0, self.cfg.taste_duration_s, rt,
                                 details={"set": what, "label": what.upper()})]
        else:
            raise ValueError(f"unknown manual stimulus {what!r}")
        for ev in evs:
            self.send(ev, source="key")
        return evs[0]

    def handle_key(self, key: str) -> str | None:
        if key not in BRAIN_KEYS:
            return None
        what = BRAIN_KEYS[key][0]
        ev = self.manual(what)
        acts = self.triggers is not None
        extra = {"loom": "watch GF / MDN" + (" (GF > threshold -> jump)" if acts else
                                              "" if self.cfg.steer else
                                              " (no body effect without --brain-steer)"),
                 "sugar": "watch MN9" + (" (MN9 > threshold -> proboscis)"
                                         if acts and self.triggers.can_proboscis
                                         else " (no body effect)"),
                 "bitter": "no body effect"}[what]
        return (f"[brain] {what}: {'+'.join(self.cfg.loom_sets) if what == 'loom' else what} "
                f"neurons at 200 Hz for {ev.duration_s:g} s from t={ev.sim_time:.2f}s; {extra}")

    # ------------------------------------------------------------- brain -> body
    def _steer_filter(self, hold: np.ndarray) -> np.ndarray:
        # per physics step: keep it cheap (cached drive / weight)
        b0, b1 = self.drive[0], self.drive[1]
        d = 0.5 * (hold[0] - hold[1])
        w, m = self._w, self.gains.max_amp
        return np.array([min(max(b0 + w * d, -m), m), min(max(b1 - w * d, -m), m)])

    def update(self) -> None:
        """Once per physics chunk: clock mark, new states, drive smoothing, window."""
        if self.brain is None:
            return
        rt = self._run_time()
        if rt - self._last_clock >= self.cfg.clock_every_s:
            self._last_clock = rt
            try:
                self.brain.clock(rt)
            except Exception:
                pass
        states = self.brain.poll("app")
        if self.cfg.sync_wait_s > 0 and (not self.cfg.sync_loom_only or rt <= self._loom_until):
            states += self._sync_wait(rt, states)
        for st in states:
            if self._last_seq and st.seq > self._last_seq + 1:
                self.n_dropped += st.seq - self._last_seq - 1
            self._last_seq = st.seq
            self.n_states += 1
            self.latest = st
            self._target = descending_to_drive(st, self.gains)
            self._notice(st)
            if self.triggers is not None:
                n0 = len(self.triggers.fired)
                self.triggers.on_state(st, rt)
                s = self.session
                for t_fired, name, rate in self.triggers.fired[n0:]:
                    if s is not None and s.logger is not None:
                        s.logger.log_event("brain_action", details={
                            "action": name, "rate_hz": round(rate, 2),
                            "brain_time": st.brain_time, "stim_time": t_fired})
        alive = self.brain.is_alive()
        if not alive and not self._dead_reported:
            self._dead_reported = True
            self.say(f"[brain] the brain process stopped ({self.brain.error or 'exited'}); "
                     "the fly keeps walking with drive [1, 1]")
        st = self.latest
        stale = (not alive or st is None or st.sim_time is None
                 or rt - st.sim_time > self.cfg.stale_after_s)
        target = np.ones(2) if stale else self._target
        if self._last_rt is not None and rt > self._last_rt:
            a = 1.0 - math.exp(-(rt - self._last_rt) / max(self.cfg.drive_tau_s, 1e-6))
            self.drive += a * (target - self.drive)
        elif self._last_rt is None:
            self.drive[:] = target
        self._last_rt = rt
        self._w = min(max(0.5 * (self.drive[0] + self.drive[1]), 0.0), 1.0)
        if states:
            esc = float(states[-1].descending.get("escape", 0.0))
            dd = drive_display(self.drive, esc, self.cfg.steer, self.gains)
            for s in states:
                s.drive = dict(dd, lag_s=max(0.0, rt - s.sim_time)
                               if s.sim_time is not None else None)
                self.recent.append(s)
                if self.window is not None:
                    self.window.send(s)
        if self.window is not None:
            if not self.window.is_alive() and not self._window_closed_reported:
                self._window_closed_reported = True
                self.say("[brain] brain window closed (the fly and the brain keep running)")

    def _sync_wait(self, rt: float, states: list) -> list:
        """Wait (<= sync_wait_s wall) until the brain has published the state whose
        window ends within the last window_s of fly time; returns the extra states."""
        import time

        def newest(sts):
            for st in reversed(sts):
                if st.sim_time is not None:
                    return float(st.sim_time)
            return None

        target = rt - self.cfg.window_s + 1e-9
        t_new = newest(states)
        if t_new is None and self.latest is not None and self.latest.sim_time is not None:
            t_new = float(self.latest.sim_time)
        if t_new is not None and t_new >= target:
            return []
        extra: list = []
        t0 = time.perf_counter()
        deadline = t0 + self.cfg.sync_wait_s
        while time.perf_counter() < deadline and self.brain.is_alive():
            time.sleep(0.0003)
            more = self.brain.poll("app")
            if more:
                extra += more
                t = newest(more)
                if t is not None and t >= target:
                    break
        self.n_sync_waits += 1
        self.sync_wait_wall_s += time.perf_counter() - t0
        return extra

    def _notice(self, st: BrainState) -> None:
        """Edge-triggered terminal notes for the readouts worth knowing about."""
        d = st.descending
        checks = {
            "escape": (d.get("escape", 0.0) > 50.0, f"giant fiber (DNp01) firing "
                                                    f"{d.get('escape', 0.0):.0f} Hz (escape command)"),
            "mdn": (0.5 * (d.get("backward_L", 0) + d.get("backward_R", 0)) > 10.0,
                    f"MDN (backward walking) {d.get('backward_L', 0):.0f}/"
                    f"{d.get('backward_R', 0):.0f} Hz"),
            "mn9": (st.probes.get("MN9", 0.0) > 20.0,
                    f"MN9 (proboscis motor neuron) {st.probes.get('MN9', 0.0):.0f} Hz"),
        }
        for k, (on, text) in checks.items():
            if on and not self._flags.get(k):
                self.say(f"[brain] t_brain={st.brain_time:.2f}s {text}")
            self._flags[k] = on

    # ------------------------------------------------------------- readouts
    def lag(self) -> float | None:
        st = self.latest
        if st is None or st.sim_time is None:
            return None
        return max(0.0, self._run_time() - st.sim_time)

    def status_line(self) -> str:
        st = self.latest
        if self.brain is None or not self.brain.is_alive():
            return "BRAIN stopped"
        if st is None:
            return "BRAIN starting (waiting for the first state)"
        d = st.descending
        mdn = 0.5 * (d.get("backward_L", 0.0) + d.get("backward_R", 0.0))
        return (f"BRAIN t {st.brain_time:6.2f}s lag {self.lag() or 0.0:4.2f}s "
                f"rtf {st.realtime_factor:4.2f} (cap x{st.compute_rtf:.1f})  "
                f"drive L{self.drive[0]:+.2f} R{self.drive[1]:+.2f} "
                f"{'STEER' if self.cfg.steer else 'view-only'}  "
                f"GF {d.get('escape', 0.0):3.0f} MDN {mdn:3.0f} "
                f"walk {0.5 * (d.get('walk_L', 0) + d.get('walk_R', 0)):3.0f} "
                f"turn {d.get('turn_L', 0):.0f}/{d.get('turn_R', 0):.0f} "
                f"MN9 {st.probes.get('MN9', 0.0):3.0f} Hz")

    def hud_lines(self) -> list[str]:
        st = self.latest
        if self.brain is None or not self.brain.is_alive():
            return ["BRAIN stopped"]
        if st is None:
            return ["BRAIN starting..."]
        d = st.descending
        mdn = 0.5 * (d.get("backward_L", 0.0) + d.get("backward_R", 0.0))
        return [
            f"BRAIN t {st.brain_time:6.2f}s  lag {self.lag() or 0.0:4.2f}s  "
            f"x{st.realtime_factor:.2f} real time (can x{st.compute_rtf:.1f})",
            f"drive L{self.drive[0]:+.2f} R{self.drive[1]:+.2f} "
            f"{'(steering)' if self.cfg.steer else '(view only)'}  GF {d.get('escape', 0):.0f} "
            f"MDN {mdn:.0f} MN9 {st.probes.get('MN9', 0.0):.0f} DNg12 {d.get('groom', 0):.0f} Hz",
            "O loom  T sugar  K bitter" + ("   brain actions ON" if self.triggers is not None
                                           else ""),
        ]

    def render_snapshot(self, size: tuple[int, int] | None = None):
        """BGR brain-window frame rendered in this process from the recent states
        (None before the first state). Call with a copy of ``recent`` if the
        physics thread is running (see app.py screenshots)."""
        return render_brain_frame(self.layout, list(self.recent), self.cfg.window_s, size)

    def metric_row(self) -> tuple:
        st = self.latest
        ctrl = (np.nan, np.nan)
        s = self.session
        if s is not None and s.cfg.controller.kind == "hybrid":
            sig = s.sim.controller.descending_signal()
            ctrl = (float(sig[0]), float(sig[1]))
        if st is None:
            dn = [np.nan] * len(DESCENDING_GROUPS)
            return (np.nan, np.nan, float(self.drive[0]), float(self.drive[1]), *ctrl, *dn, np.nan)
        lag = self.lag()
        return (float(st.brain_time), np.nan if lag is None else lag,
                float(self.drive[0]), float(self.drive[1]), *ctrl,
                *[float(st.descending.get(g, 0.0)) for g in DESCENDING_GROUPS],
                float(st.probes.get("MN9", np.nan)))

    def config_dict(self) -> dict:
        return {"link": asdict(self.cfg), "process": asdict(self.brain_cfg),
                "drive_gains": asdict(self.gains)}

    def summary(self) -> dict:
        st = self.latest
        return {"brain_states": self.n_states, "brain_states_dropped": self.n_dropped,
                "brain_stimuli_sent": self.n_sent,
                "brain_time_s": None if st is None else st.brain_time,
                "brain_lag_s": self.lag(), "brain_info": self.info,
                "brain_actions": (dict(self.triggers.counts) if self.triggers is not None
                                  else None)}
