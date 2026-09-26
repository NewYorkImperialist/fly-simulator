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
from perpetualfly.brain_viz.playground import canonical_label

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
    # Event-driven fast path (docs/BRAIN.md, "Latency"; needs actions). The worker
    # watches the giant fiber (DNp01) and MN9 spike by spike and sends a small
    # trigger the moment their rate over the trailing fast_window_s exceeds the
    # body thresholds (jump_escape_hz / proboscis_mn9_hz, kept in sync live, e.g.
    # when --stress lowers the jump threshold), instead of waiting for the
    # BrainState window to end. While syncing (sync_wait_s, loom active) update()
    # also waits for the brain to reach the clock mark just sent, and sends a clock
    # mark every update. Same spikes, same threshold; only the reading is earlier.
    # Measured (docs/BRAIN.md "Latency"): GF crossing -> jump trigger 14-21 ms ->
    # 2-3 ms median at 5 ms physics chunks, no extra wall time. False = old path.
    fast_path: bool = True
    fast_window_s: float = 0.02
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
    # ---- brain playground (docs/PLAYGROUND.md) -------------------------------------
    # virtual optogenetics / lesions / decision meters. The brain window shows the
    # palette + meters (key P in the brain window toggles the classic panels).
    playground: bool = True
    # --stim "DNa02_L:120:1.0@3,MDN@6": TARGET[:RATE_HZ[:DURATION_S]][@RUN_TIME_S]
    stim: str | None = None
    # --lesion "DNp01,MDN" (silenced from the start) / "MDN@4" (from run time 4 s)
    lesion: str | None = None
    stim_rate_hz: float = 120.0  # default rate / duration (CLI, keys 9 0 -)
    stim_duration_s: float = 1.0
    # ---- looming habituation (--habituation; docs/HABITUATION.md) ------------------
    # short-term depression of the LC4 / LPLC2 -> giant fibre synapses in the brain
    # worker (perpetualfly/brain/habituation.py; HabituationConfig overrides in
    # habituation_config, e.g. {"u": 0.006, "tau_rec_s": 20, "dishabituate_frac": 0.8})
    habituation: bool = False
    habituation_config: dict = field(default_factory=dict)
    # ---- odours / fear learning (--odor-zones / --learning; docs/FEAR_LEARNING.md) --
    # odors: the worker builds the odour KC sets (odor_A, odor_B) + dan_punish;
    # learning: KC -> MBON plasticity on (perpetualfly/brain/plasticity.py;
    # PlasticityConfig overrides in plasticity_config)
    odors: bool = False
    learning: bool = False
    plasticity_config: dict = field(default_factory=dict)
    # ---- brain recording (--brain-record; docs/BRAIN_REPLAY.md) --------------------
    # BrainStates + stimuli / actions -> <run dir>/brain_rec/ (compressed chunks);
    # replay with scripts/brain_replay.py RUN_DIR
    record: bool = False
    record_window_s: float = 0.1  # states merged to at least this much brain time
    record_max_mb: float = 50.0  # recording stops beyond this size
    record_chunk_states: int = 100

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
                       size: tuple[int, int] | None = None, playground: bool = False):
    """BGR brain-window frame for ``states`` (oldest first) via the brain window's
    headless ``render_frame``; None without layout / states."""
    if layout is None or not states:
        return None
    from perpetualfly.brain_viz.window import DEFAULT_SIZE, render_frame

    return render_frame(layout, list(states), size=size or DEFAULT_SIZE, interval=window_s,
                        playground=playground)


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

        self.fast = bool(cfg.fast_path and cfg.actions)
        self._fast_thr: dict | None = self._fast_thresholds() if self.fast else None
        self.brain_cfg = BrainConfig(
            data_dir=cfg.data_dir, window_s=cfg.window_s, pace="sim",
            max_lag_s=cfg.max_lag_s, subscribers=("app",), synthetic=cfg.synthetic,
            fast_triggers=self._fast_thr, fast_window_s=cfg.fast_window_s,
            habituation=self.habituation_dict(), plasticity=self.plasticity_dict())
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
        self._progress = -1e9  # fast path: fly time the brain has simulated up to
        self.fast_events: deque = deque(maxlen=200)  # recent trigger FastEvents
        self.n_fast = 0
        # brain playground (docs/PLAYGROUND.md)
        from perpetualfly.brain_viz.playground import (PRESETS, parse_lesion_specs,
                                                       parse_stim_specs)

        self.presets = PRESETS
        self.lesions: list[str] = []  # active lesion targets (names), in order
        self.pg_log: deque[tuple[float, str]] = deque(maxlen=200)  # (run time, text)
        self.pg_selected = 0  # palette index for keys 9 / 0 / -
        self._pg_key_used = False
        self.pg_used = False
        self._stim_plan = parse_stim_specs(cfg.stim, cfg.stim_rate_hz, cfg.stim_duration_s)
        self._lesion_plan = parse_lesion_specs(cfg.lesion)
        self._pg_errors_seen: set[str] = set()
        self.recorder = None  # BrainRecorder (--brain-record), created in attach()
        if start:
            self.start()

    def habituation_dict(self) -> dict | None:
        """BrainConfig.habituation (None = off)."""
        if not self.cfg.habituation:
            return None
        return {**dict(self.cfg.habituation_config), "enabled": True}

    def plasticity_dict(self) -> dict | None:
        """BrainConfig.plasticity (None = off: no odours, no learning)."""
        if not (self.cfg.odors or self.cfg.learning):
            return None
        return {**dict(self.cfg.plasticity_config), "enabled": bool(self.cfg.learning)}

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        from perpetualfly.brain.process import BrainProcess

        if self.brain is None:
            self.brain = BrainProcess(self.brain_cfg)

    def wait_ready(self, timeout: float = 120.0) -> dict:
        self.info = self.brain.wait_ready(timeout)
        self.layout = self.brain.layout(timeout)
        if self.recorder is not None and self.layout is not None:
            self.recorder.set_layout(self.layout)
        return self.info

    def start_window(self) -> None:
        if not self.show_window or self.window is not None or self.layout is None:
            return
        from perpetualfly.brain_viz import BrainWindowProcess

        self.window = BrainWindowProcess(self.layout, playground=self.cfg.playground).start()

    @property
    def pids(self) -> list[int]:
        out = []
        if self.brain is not None and self.brain.pid:
            out.append(self.brain.pid)
        if self.window is not None and self.window.process.pid:
            out.append(self.window.process.pid)
        return out

    def close(self) -> None:
        if self.recorder is not None:
            try:
                r = self.recorder.close()
                self.say(f"[brain-record] {r['states']} states, {r['events']} events, "
                         f"{r['duration_s']:.1f} s, {r['mb']:.2f} MB"
                         + (f" ({r['mb_per_min']:.2f} MB/min)" if r['mb_per_min'] else "")
                         + f" -> {r['dir']}  (replay: .venv/bin/python scripts/brain_replay.py "
                           f"{Path(r['dir']).parent})")
            except Exception as e:
                print(f"warning: closing the brain recording failed: {e}", file=sys.stderr)
            self.recorder = None
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
        if self.cfg.record and self.recorder is None:
            self._start_recorder(session)
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

    def _start_recorder(self, session) -> None:
        from perpetualfly.brain_viz.replay import BrainRecorder

        lg = getattr(session, "logger", None)
        if lg is None or getattr(lg, "run_dir", None) is None:
            self.say("[brain-record] needs a run directory (logging is off); not recording")
            return
        self.recorder = BrainRecorder(lg.run_dir, self.layout, window_s=self.cfg.record_window_s,
                                      chunk_states=self.cfg.record_chunk_states,
                                      max_mb=self.cfg.record_max_mb, say=self.say)
        self.say(f"[brain-record] recording brain states to {self.recorder.dir}")

    def _record_actions(self, fired: list) -> None:
        if self.recorder is not None:
            for t_fired, name, rate in fired:
                self.recorder.add_event("action", t_fired, str(name), rate_hz=float(rate))

    def _run_time(self, sim_time: float | None = None) -> float:
        s = self.session
        if s is None:
            return 0.0 if sim_time is None else float(sim_time)
        return float(s.metrics.run_time_at(s.sim.time if sim_time is None else sim_time))

    # ------------------------------------------------------------- body -> brain
    def send(self, ev: StimulusEvent, source: str = "", event_type: str = "brain_stim") -> None:
        """Brain + window + events.csv. ``ev.sim_time`` must be the fly run time."""
        self.stim_log.append(ev)
        self.n_sent += 1
        if self.recorder is not None:
            self.recorder.add_stimulus(ev, source)
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
            s.logger.log_event(event_type, force_direction=ev.side, details={
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
        if self.recorder is not None:
            self.recorder.add_event("stim", rt, "RESET", stim_kind="reset")
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

    # ------------------------------------------------------------- playground
    def _pg_note(self, text: str, rt: float | None = None) -> str:
        rt = self._run_time() if rt is None else rt
        self.pg_log.append((rt, text))
        self.pg_used = True
        return f"[playground] t={rt:.2f}s {text}"

    def opto(self, target: str, rate_hz: float | None = None, duration_s: float | None = None,
             source: str = "key") -> str:
        """Virtual optogenetics: drive every neuron of ``target`` (see
        ``mapping.resolve_target``) at ``rate_hz`` for ``duration_s`` from now
        (direct stimulation, not a natural sense)."""
        rt = self._run_time()
        rate = float(rate_hz if rate_hz is not None else self.cfg.stim_rate_hz)
        dur = float(duration_s if duration_s is not None else self.cfg.stim_duration_s)
        ev = StimulusEvent("opto", "none", 1.0, dur, rt,
                           details={"target": str(target), "rate_hz": rate,
                                    "label": f"OPTO {canonical_label(target)}"})
        self.send(ev, source=source, event_type="brain_opto")
        return self._pg_note(f"STIM {target} {rate:g} Hz {dur:g} s ({source})", rt)

    def stop_stim(self, source: str = "key") -> str:
        rt = self._run_time()
        self.send(StimulusEvent("opto_stop", "none", 0.0, 0.0, rt,
                                details={"label": "OPTO STOP"}), source=source,
                  event_type="brain_opto")
        return self._pg_note(f"STOP all stimulation ({source})", rt)

    def set_lesion(self, target: str, on: bool | None = None, source: str = "key") -> str:
        """Virtual lesion on / off (``on=None`` toggles): the target's neurons can
        no longer spike. Lesions persist across fly / brain resets."""
        target = str(target).strip()
        cur = target in self.lesions
        on = (not cur) if on is None else bool(on)
        if on and not cur:
            self.lesions.append(target)
        elif not on and cur:
            self.lesions.remove(target)
        return self._apply_lesions(f"{'LESION' if on else 'UNLESION'} {target}", source,
                                   target, on)

    def clear_lesions(self, source: str = "key") -> str:
        self.lesions.clear()
        return self._apply_lesions("CLEAR LESIONS", source, "", False)

    def _apply_lesions(self, what: str, source: str, target: str, on: bool) -> str:
        rt = self._run_time()
        if self.brain is not None and self.brain.is_alive():
            try:
                self.brain.set_lesions(list(self.lesions))
            except Exception:
                pass
        if self.window is not None:
            self.window.send(StimulusEvent("lesion", "none", 0.0, 0.0, rt,
                                           details={"label": what[:28]}))
        s = self.session
        if s is not None and s.logger is not None:
            s.logger.log_event("brain_lesion", details={
                "target": target, "on": on, "active": "+".join(self.lesions),
                "stim_time": rt, "source": source})
        return self._pg_note(f"{what} ({source}); silenced: "
                             f"{', '.join(self.lesions) or 'none'}", rt)

    def execute(self, cmd: dict, source: str = "window") -> str | None:
        """A playground command dict (brain-window clicks; see brain_viz.playground)."""
        op = cmd.get("op") if isinstance(cmd, dict) else None
        try:
            if op == "stim":
                return self.opto(str(cmd["target"]), cmd.get("rate_hz"), cmd.get("duration_s"),
                                 source)
            if op == "lesion":
                return self.set_lesion(str(cmd["target"]), cmd.get("on"), source)
            if op == "clear_lesions":
                return self.clear_lesions(source)
            if op == "stop_stim":
                return self.stop_stim(source)
        except (KeyError, TypeError, ValueError) as e:
            return f"[playground] bad command {cmd!r}: {e}"
        return None

    def handle_playground_key(self, key: str) -> str | None:
        """9: select the next palette target, 0: stimulate it, -: (un)lesion it."""
        p = self.presets[self.pg_selected % len(self.presets)]
        if key in ("9", "0", "-"):
            self._pg_key_used = True
        if key == "9":
            self.pg_selected = (self.pg_selected + 1) % len(self.presets)
            p = self.presets[self.pg_selected]
            self.pg_used = True
            return (f"[playground] selected {p.label} ({p.role}); 0 = stimulate "
                    f"{self.cfg.stim_rate_hz:g} Hz {self.cfg.stim_duration_s:g} s, - = lesion")
        if key == "0":
            return self.opto(p.target, source="key")
        if key == "-":
            return self.set_lesion(p.target, source="key")
        return None

    def _playground_tick(self, rt: float) -> None:
        """Scheduled --stim / --lesion entries and window clicks (in update())."""
        while self._lesion_plan and (self._lesion_plan[0][1] is None
                                     or self._lesion_plan[0][1] <= rt):
            name, _ = self._lesion_plan.pop(0)
            if name not in self.lesions:
                self.say(self.set_lesion(name, True, source="cli"))
        while self._stim_plan and (self._stim_plan[0].at_s is None
                                   or self._stim_plan[0].at_s <= rt):
            sp = self._stim_plan.pop(0)
            self.say(self.opto(sp.target, sp.rate_hz, sp.duration_s, source="cli"))
        if self.window is not None:
            for cmd in self.window.poll_commands():
                msg = self.execute(cmd, source="window")
                if msg:
                    self.say(msg)

    def thresholds(self) -> dict:
        """Body thresholds for the decision meters (live: --stress lowers the jump one)."""
        tp = self.triggers.p if self.triggers is not None else self.cfg
        return {"jump_hz": float(tp.jump_escape_hz), "groom_hz": float(tp.groom_hz),
                "mn9_hz": float(tp.proboscis_mn9_hz),
                "backward_ref_hz": float(self.gains.backward_ref),
                "r_ref_hz": float(self.gains.r_ref),
                "actions": self.triggers is not None, "steer": bool(self.cfg.steer)}

    def playground_info(self) -> dict:
        """App-side part of ``BrainState.playground`` (log, thresholds, selection)."""
        return {"log": [f"t={t:6.2f}  {txt}" for t, txt in list(self.pg_log)[-6:]],
                "thresholds": self.thresholds(),
                "selected": (self.presets[self.pg_selected % len(self.presets)].target
                             if self._pg_key_used else None),
                "active_lesions": list(self.lesions)}

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
        syncing = self.cfg.sync_wait_s > 0 and (not self.cfg.sync_loom_only
                                                or rt <= self._loom_until)
        if (rt - self._last_clock >= self.cfg.clock_every_s
                or (self.fast and syncing and rt > self._last_clock)):
            self._last_clock = rt
            try:
                self.brain.clock(rt)
            except Exception:
                pass
        self._playground_tick(rt)
        fast_evs: list = []
        if self.fast:
            self._update_fast_thresholds()
            fast_evs = self._poll_fast()
        states = self.brain.poll("app")
        if syncing:
            states += self._sync_wait(rt, states, fast_evs)
        for fe in fast_evs:
            self._on_fast(fe, rt)
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
                self._record_actions(self.triggers.fired[n0:])
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
            pgi = self.playground_info()
            for s in states:
                s.drive = dict(dd, lag_s=max(0.0, rt - s.sim_time)
                               if s.sim_time is not None else None)
                pg = getattr(s, "playground", None)
                if not isinstance(pg, dict):
                    pg = {}
                for e in pg.get("errors", []) or []:
                    if e not in self._pg_errors_seen:
                        self._pg_errors_seen.add(e)
                        self.say(f"[playground] warning: {e}")
                s.playground = {**pg, **pgi}
                self.recent.append(s)
                if self.recorder is not None:
                    self.recorder.add_state(s)
                if self.window is not None:
                    self.window.send(s)
        if self.window is not None:
            if not self.window.is_alive() and not self._window_closed_reported:
                self._window_closed_reported = True
                self.say("[brain] brain window closed (the fly and the brain keep running)")

    # ------------------------------------------------------------- fast path
    def _fast_thresholds(self) -> dict:
        trig = getattr(self, "triggers", None)
        tp = trig.p if trig is not None else self.cfg
        return {"escape": float(tp.jump_escape_hz), "MN9": float(tp.proboscis_mn9_hz)}

    def _update_fast_thresholds(self) -> None:
        """Keep the worker's thresholds equal to the body's (e.g. --stress lowers
        the jump threshold at run time)."""
        thr = self._fast_thresholds()
        if thr != self._fast_thr:
            self._fast_thr = thr
            try:
                self.brain.set_fast_triggers(thr)
            except Exception:
                pass

    def _poll_fast(self) -> list:
        """Pending trigger FastEvents (progress marks are consumed here)."""
        out = []
        for fe in self.brain.poll_fast():
            if fe.kind == "progress":
                if fe.sim_time is not None:
                    self._progress = max(self._progress, float(fe.sim_time))
            else:
                out.append(fe)
        return out

    def _on_fast(self, fe, rt: float) -> None:
        """A fast-path trigger: the same threshold check as for a BrainState,
        applied as soon as the crossing spike is simulated."""
        self.n_fast += 1
        self.fast_events.append(fe)
        if self.triggers is None:
            return
        n0 = len(self.triggers.fired)
        self.triggers.on_fast(fe, rt)
        s = self.session
        self._record_actions(self.triggers.fired[n0:])
        for t_fired, name, rate in self.triggers.fired[n0:]:
            if s is not None and s.logger is not None:
                s.logger.log_event("brain_action", details={
                    "action": name, "rate_hz": round(rate, 2), "brain_time": fe.brain_time,
                    "stim_time": t_fired, "fast_path": True, "crossing_time": fe.sim_time})

    def _sync_wait(self, rt: float, states: list, fast_evs: list | None = None) -> list:
        """Wait (<= sync_wait_s wall) until the brain has published the state whose
        window ends within the last window_s of fly time; returns the extra states.
        With the fast path, also until the brain has simulated up to the last clock
        mark (or a trigger arrived); new triggers are appended to ``fast_evs``."""
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
        fast = self.fast and fast_evs is not None

        def fast_done() -> bool:
            return (not fast or bool(fast_evs)
                    or self._progress >= self._last_clock - 1e-9)

        state_ok = t_new is not None and t_new >= target
        if state_ok and fast_done():
            return []
        extra: list = []
        t0 = time.perf_counter()
        deadline = t0 + self.cfg.sync_wait_s
        while time.perf_counter() < deadline and self.brain.is_alive():
            time.sleep(0.0003)
            if fast:
                fast_evs += self._poll_fast()
            more = self.brain.poll("app")
            if more:
                extra += more
                t = newest(more)
                if t is not None and t >= target:
                    state_ok = True
            if state_ok and fast_done():
                break
            if fast and fast_evs:
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
        ] + ([self.playground_hud()] if self.pg_used or self.lesions else []) \
          + ([self.habituation_hud()] if self.cfg.habituation else []) \
          + ([self.learning_hud()] if self.cfg.learning else []) \
          + (["REC brain -> brain_rec/" + (" (size limit: stopped)" if self.recorder.stopped
                                            else "")] if self.recorder is not None else [])

    def habituation_hud(self) -> str:
        st = self.latest
        h = (getattr(st, "habituation", None) or {}) if st is not None else {}
        if not h.get("enabled"):
            return "HABITUATION (model) waiting for the brain"
        by = "  ".join(f"{k} {v:.2f}" for k, v in (h.get("by_type") or {}).items())
        return (f"HABITUATION (model) GF-input efficacy {by}  "
                f"(U {h.get('u', 0):g}, recovery {h.get('tau_rec_s', 0):g} s)")

    def learning_hud(self) -> str:
        """Fear learning (docs/FEAR_LEARNING.md): gamma1pedc KC>MBON efficacy of
        each odour's KCs, DAN rate."""
        st = self.latest
        lr = (getattr(st, "learning", None) or {}) if st is not None else {}
        if not lr.get("enabled"):
            return "LEARNING (model) waiting for the brain"
        main = lr.get("main", "")
        eff = lr.get("efficacy_by_odor") or {}
        mem = "  ".join(f"{o} {float(v.get(main, 1.0)):.2f}" for o, v in eff.items())
        return (f"LEARNING (model) KC>MBON efficacy [{main}] {mem}  DAN "
                f"{float((lr.get('dan_hz') or {}).get(main, 0.0)):.0f} Hz"
                + ("  RUNAWAY (paused)" if lr.get("runaway") else ""))

    def playground_hud(self) -> str:
        p = self.presets[self.pg_selected % len(self.presets)]
        st = self.latest
        opto = [f"{o.get('label')} {o.get('rate_hz', 0):.0f}Hz"
                for o in ((getattr(st, "playground", None) or {}).get("opto", []) if st else [])]
        return (f"PLAYGROUND lesioned: {', '.join(self.lesions) or 'none'}  "
                f"stim: {', '.join(opto) or 'none'}  [9] {p.label} [0] stim [-] lesion")

    def render_snapshot(self, size: tuple[int, int] | None = None):
        """BGR brain-window frame rendered in this process from the recent states
        (None before the first state). Call with a copy of ``recent`` if the
        physics thread is running (see app.py screenshots)."""
        return render_brain_frame(self.layout, list(self.recent), self.cfg.window_s, size,
                                  playground=self.cfg.playground)

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
                                  else None),
                "habituation": (dict(getattr(st, "habituation", None) or {})
                                if self.cfg.habituation and st is not None else None),
                "learning": (dict(getattr(st, "learning", None) or {})
                             if (self.cfg.learning or self.cfg.odors) and st is not None
                             else None),
                "brain_record": self.recorder.summary() if self.recorder is not None else None,
                "playground": ({"lesions": list(self.lesions),
                                "log": [[round(t, 3), txt] for t, txt in self.pg_log]}
                               if self.pg_used else None)}
