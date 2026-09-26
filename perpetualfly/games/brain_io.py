"""The connectome brain as the game controller: senses in, descending neurons out.

``GameBrain`` owns one brain worker (``perpetualfly.brain.BrainProcess``, the Shiu et
al. 2024 LIF model of FlyWire v783) paced to the fly's simulated time, exactly like
the app's ``BrainLink`` but without the app (so it runs standalone in
``scripts/play.py``). It is the sink for the game's looming events and turns the
published ``BrainState``s into a CPG drive for the walking controller and a jump
request.

Control modes (the scientific check, docs/GAMES.md):

* ``brain``: left-eye looming drives the left-side LC4 / LPLC2 neurons, right-eye
  looming the right side (FlyWire soma side = optic lobe = eye).
* ``mirror``: the sensory mapping is mirrored: left-eye looming drives the *right*
  optic lobe's LC4 / LPLC2 and vice versa. Same brain, same motor mapping.
* ``none``: no brain process at all; constant drive ``[1, 1]`` (straight walking),
  no jumps. This is the "disconnected" control.

The same side mapping applies to FOLLOW THE LEADER's LC10a (pursuit) events
(``on_pursuit``): ``mirror`` sends the left eye's leader to the right LC10a.

Motor mapping (``GameMapping``, our design; read by ``descending_to_drive`` with
game gains): DNa01/DNa02 (turn_L - turn_R) -> steering (the controller turns toward
the smaller amplitude, so left DN activity turns left), BDN2/oDN1/P9 (walk) ->
speed, MDN -> brake / walk backward, giant fibre DNp01 (escape) -> jump.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from perpetualfly.brain.mapping import DriveGains, descending_to_drive
from perpetualfly.brain.schema import BrainState, StimulusEvent

CONTROLS = ("brain", "mirror", "none")
MIRROR_SIDE = {"left": "right", "right": "left"}


@dataclass
class GameMapping:
    """Descending rates -> controls. Our interface design, stated in the HUD/docs.

    Steering uses ``mapping.descending_to_drive`` with ``turn_gain`` / ``r_ref``
    raised from the app's conservative defaults (0.4 / 30 Hz): in the connectome a
    one-eyed loom drives the contralateral DNa01/02 pair to ~20-40 Hz (group mean),
    which with the defaults gives drive [1 - d, 1 + d] with d ~ 0.27 (~95 deg/s);
    with these gains d ~ 0.46 (~180 deg/s). Measured on the CPG controller:
    d = 0.2 / 0.4 / 0.6 / 0.8 -> 61 / 154 / 235 / 323 deg/s.
    """

    r_ref_hz: float = 25.0
    turn_gain: float = 0.6
    walk_gain: float = 0.25
    backward_ref_hz: float = 40.0
    min_amp: float = 0.3
    max_amp: float = 1.6
    drive_tau_s: float = 0.06  # low-pass of the drive in sim time
    # giant fibre -> jump (off by default: the GF fires for nearly every rock and the
    # hop interrupts steering; docs/GAMES.md)
    jump: bool = False
    jump_escape_hz: float = 60.0
    jump_refractory_s: float = 1.5
    jump_mode: str = "long"  # "long" | "short" (docs/SWATTER.md)
    stale_after_s: float = 1.0

    def gains(self) -> DriveGains:
        return DriveGains(r_ref=self.r_ref_hz, walk_gain=self.walk_gain,
                          turn_gain=self.turn_gain, backward_ref=self.backward_ref_hz,
                          min_amp=self.min_amp, max_amp=self.max_amp)

    def drive(self, rates: dict) -> np.ndarray:
        return descending_to_drive(rates, self.gains())


class GameBrain:
    """Brain worker + pacing + mapping for a game (one brain process at most).

    ``on_loom(ev)``: sink for ``LoomingVision`` (applies the control mode's side
    mapping). ``update(run_time)`` once per physics chunk: clock mark, new states,
    drive smoothing; returns True when the giant fibre asks for a jump.
    """

    def __init__(self, control: str = "brain", *, synthetic: dict | None = None,
                 data_dir: str | None = None, mapping: GameMapping | None = None,
                 window_s: float = 0.02, sync_wait_s: float = 0.05, seed: int = 0,
                 max_lag_s: float = 0.5) -> None:
        if control not in CONTROLS:
            raise ValueError(f"control must be one of {CONTROLS}, not {control!r}")
        self.control = control
        self.map = mapping or GameMapping()
        self.window_s = float(window_s)
        self.sync_wait_s = float(sync_wait_s)
        self.synthetic = synthetic
        self.data_dir = data_dir
        self.seed = seed
        self.max_lag_s = max_lag_s
        self.brain = None
        self.layout = None
        self.info: dict = {}
        self.latest: BrainState | None = None
        self.rates: dict[str, float] = {}
        self.recent: deque[BrainState] = deque(maxlen=60)
        self.drive = np.ones(2)
        self._target = np.ones(2)
        self._last_rt: float | None = None
        self._next_jump = -1e9
        self._loom_until = -1e9
        self.loom_hz = {"left": (0.0, 0.0), "right": (0.0, 0.0)}  # eye -> (LC4, LPLC2) sent
        self._loom_t = {"left": -1e9, "right": -1e9}
        self.pursuit_hz = {"left": 0.0, "right": 0.0}  # eye -> LC10a Hz sent (CHASE)
        self._pursuit_t = {"left": -1e9, "right": -1e9}
        self.n_states = 0
        self.n_sent = 0
        self.n_jumps = 0
        self.sent: list[StimulusEvent] = []
        self.gf_peak = 0.0
        self.listeners: list = []  # fn(BrainState) per new state (e.g. a brain window)

    # ------------------------------------------------------------ lifecycle
    @property
    def connected(self) -> bool:
        return self.control != "none"

    def set_control(self, control: str) -> None:
        """Switch the control mode (the worker keeps running; ``none`` simply stops
        feeding it and ignores it). Call ``reset()`` afterwards."""
        if control not in CONTROLS:
            raise ValueError(f"control must be one of {CONTROLS}, not {control!r}")
        self.control = control

    def start(self, timeout: float = 120.0, force: bool = False) -> "GameBrain":
        """Start the worker (skipped for ``none`` unless ``force``: the experiment
        switches one worker between conditions)."""
        if (not self.connected and not force) or self.brain is not None:
            return self
        from perpetualfly.brain.process import BrainConfig, BrainProcess

        cfg = BrainConfig(data_dir=self.data_dir, window_s=self.window_s, pace="sim",
                          max_lag_s=self.max_lag_s, subscribers=("app",),
                          synthetic=self.synthetic, seed=self.seed)
        self.brain = BrainProcess(cfg)
        self.info = self.brain.wait_ready(timeout)
        self.layout = self.brain.layout(timeout)
        return self

    def close(self) -> None:
        if self.brain is not None:
            try:
                self.brain.stop()
            finally:
                self.brain = None

    def alive(self) -> bool:
        return self.brain is not None and self.brain.is_alive()

    def reset(self, run_time: float) -> None:
        """Body teleported / new trial: all neurons to rest, stimuli cleared."""
        if self.alive():
            self.brain.reset_state(sim_time=run_time)
        self.drive[:] = 1.0
        self._target[:] = 1.0
        self._next_jump = -1e9
        self._loom_until = -1e9
        self.rates = {}
        self.loom_hz = {"left": (0.0, 0.0), "right": (0.0, 0.0)}
        self.pursuit_hz = {"left": 0.0, "right": 0.0}

    # ------------------------------------------------------------ senses
    def on_loom(self, ev: StimulusEvent) -> None:
        eye = ev.side
        d = ev.details or {}
        self.loom_hz[eye] = (float(d.get("lc4_hz", 0.0)), float(d.get("lplc2_hz", 0.0)))
        self._loom_t[eye] = float(ev.sim_time)
        self._forward(ev)

    def on_pursuit(self, ev: StimulusEvent) -> None:
        """Sink for the CHASE game's ``PursuitVision`` (LC10a events per eye)."""
        eye = ev.side
        d = ev.details or {}
        self.pursuit_hz[eye] = float(d.get("rate_hz", 0.0))
        self._pursuit_t[eye] = float(ev.sim_time)
        self._forward(ev)

    def _forward(self, ev: StimulusEvent) -> None:
        """Apply the control mode's side mapping and send to the worker."""
        if not self.connected:
            return
        eye = ev.side
        d = ev.details or {}
        if self.control == "mirror" and eye in MIRROR_SIDE:
            ev = StimulusEvent(ev.kind, MIRROR_SIDE[eye], ev.intensity, ev.duration_s,
                               ev.sim_time, dict(d, eye=eye, mirrored=True))
        self._loom_until = max(self._loom_until, float(ev.sim_time) + float(ev.duration_s))
        self.sent.append(ev)
        if len(self.sent) > 5000:  # long games: keep memory bounded
            del self.sent[:2500]
        self.n_sent += 1
        if self.alive():
            self.brain.send(ev)

    def pursuit_drive(self, run_time: float, hold_s: float = 0.05) -> dict[str, float]:
        """LC10a Hz currently driven per *eye* (0 once the event ran out)."""
        return {e: (v if run_time - self._pursuit_t[e] <= hold_s else 0.0)
                for e, v in self.pursuit_hz.items()}

    def eye_drive(self, run_time: float, hold_s: float = 0.08) -> dict[str, tuple[float, float]]:
        """(LC4, LPLC2) Hz currently driven per *eye* (0 once the event ran out)."""
        return {e: (v if run_time - self._loom_t[e] <= hold_s else (0.0, 0.0))
                for e, v in self.loom_hz.items()}

    # ------------------------------------------------------------ outputs
    def update(self, run_time: float) -> bool:
        """Clock + states + drive. Returns True if a jump should start now."""
        jump = False
        if self.connected and self.brain is not None:
            try:
                self.brain.clock(run_time)
            except Exception:
                pass
            states = self.brain.poll("app")
            if self.sync_wait_s > 0 and run_time <= self._loom_until:
                states += self._sync_wait(run_time, states)
            for st in states:
                self.n_states += 1
                self.latest = st
                self.recent.append(st)
                self.rates = dict(st.descending)
                for fn in self.listeners:
                    fn(st)
                self._target = self.map.drive(st.descending)
                gf = float(st.descending.get("escape", 0.0))
                self.gf_peak = max(self.gf_peak, gf)
                if (self.map.jump and gf > self.map.jump_escape_hz
                        and run_time >= self._next_jump):
                    self._next_jump = run_time + self.map.jump_refractory_s
                    self.n_jumps += 1
                    jump = True
            st = self.latest
            stale = (not self.alive() or st is None or st.sim_time is None
                     or run_time - st.sim_time > self.map.stale_after_s)
            target = np.ones(2) if stale else self._target
        else:
            target = np.ones(2)
        if self._last_rt is not None and run_time > self._last_rt:
            a = 1.0 - math.exp(-(run_time - self._last_rt) / max(self.map.drive_tau_s, 1e-6))
            self.drive += a * (target - self.drive)
        elif self._last_rt is None or run_time < self._last_rt:
            self.drive[:] = target
        self._last_rt = run_time
        return jump

    def _sync_wait(self, rt: float, states: list) -> list:
        """While a loom is active, wait (<= sync_wait_s wall) for the brain window
        that ends within the last window_s (removes one physics chunk of latency;
        same scheme as BrainLink, docs/SWATTER.md)."""
        target = rt - self.window_s + 1e-9

        def newest(sts):
            for s in reversed(sts):
                if s.sim_time is not None:
                    return float(s.sim_time)
            return None

        t_new = newest(states)
        if t_new is None and self.latest is not None and self.latest.sim_time is not None:
            t_new = float(self.latest.sim_time)
        if t_new is not None and t_new >= target:
            return []
        extra: list = []
        deadline = time.perf_counter() + self.sync_wait_s
        while time.perf_counter() < deadline and self.alive():
            time.sleep(0.0003)
            more = self.brain.poll("app")
            if more:
                extra += more
                t = newest(more)
                if t is not None and t >= target:
                    break
        return extra

    def lag(self, run_time: float) -> float | None:
        st = self.latest
        if st is None or st.sim_time is None:
            return None
        return max(0.0, run_time - st.sim_time)

    def signal_filter(self, hold: np.ndarray) -> np.ndarray:
        """Controller signal filter: the brain drive replaces the heading hold."""
        return self.drive.copy()

    def config_dict(self) -> dict:
        return {"control": self.control, "mapping": asdict(self.map), "window_s": self.window_s,
                "sync_wait_s": self.sync_wait_s, "synthetic": self.synthetic}
