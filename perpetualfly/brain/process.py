"""Run the whole-brain model in its own process and publish BrainState snapshots.

Design (simplest robust one we found):

* ``BrainProcess(cfg)`` starts one worker with the ``spawn`` context (safe on macOS,
  no fork of the MuJoCo/OpenCV parent). The worker loads the connectome, builds the
  engine and the BrainLayout, then loops: apply queued StimulusEvents -> simulate one
  chunk (``chunk_ms`` of brain time) -> every ``window_s`` of brain time publish a
  BrainState. Brain time is paced to wall time (``realtime=True``): the worker never
  runs ahead of the clock; if the model is slower than real time it simply runs as
  fast as it can and ``BrainState.realtime_factor`` drops below 1.
* Commands go parent -> worker through one ``cmd`` queue (``send()`` never blocks).
* Each *subscriber* (e.g. ``"app"`` and ``"window"``) gets its own bounded state
  queue plus a one-slot layout queue. The worker publishes every state to every
  subscriber; when a subscriber's queue is full the oldest state is dropped, so a
  slow or absent reader never blocks the worker or the others. ``latest()`` drains a
  queue and returns the newest state (non-blocking); ``poll()`` returns all pending.
  Hand ``subscriber(name)`` (picklable) to another process, e.g. the brain window,
  and call ``latest()`` / ``layout()`` on it there.
* States tile brain time: each spike appears in exactly one BrainState (window =
  the ``window_s`` of brain time since the previous state).
* Shutdown: ``stop()`` (also registered with ``atexit``) asks the worker to exit,
  then terminates/kills it. The worker is a daemon and also exits by itself as soon
  as its parent process is gone, so no orphans survive a crashed or killed app.
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import queue
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .schema import DESCENDING_GROUPS, NEUROTRANSMITTERS, BrainLayout, BrainState, StimulusEvent


@dataclass
class BrainConfig:
    data_dir: str | None = None       # default: <repo>/data/brain
    chunk_ms: float = 20.0            # brain time simulated between command polls
    window_s: float = 0.1             # brain time per published BrainState
    realtime: bool = True             # pace brain time to wall time
    seed: int = 0
    subscribers: tuple[str, ...] = ("app", "window")
    queue_size: int = 8               # states kept per subscriber before dropping
    n_per_region: int = 16            # display neurons sampled per region
    max_rate_hz: float = 200.0        # stimulus mapping (see mapping.StimulusMapper)
    whip_air_scale: float = 0.5
    enable_ground_contact: bool = False
    synthetic: dict | None = None     # tests: {"n": 50, "p_conn": .1, "seed": 0}
    extra: dict = field(default_factory=dict)


class BrainSubscriber:
    """Read side of one subscriber's queues. Picklable (pass to a child process)."""

    def __init__(self, name: str, states, layout_q):
        self.name = name
        self._states = states
        self._layout_q = layout_q
        self._layout: BrainLayout | None = None

    def layout(self, timeout: float | None = 0.0) -> BrainLayout | None:
        """The BrainLayout (cached after the first successful read)."""
        if self._layout is None:
            try:
                if timeout == 0.0:
                    self._layout = self._layout_q.get_nowait()
                else:
                    self._layout = self._layout_q.get(timeout=timeout)
            except queue.Empty:
                return None
        return self._layout

    def poll(self) -> list[BrainState]:
        """All pending states, oldest first (non-blocking)."""
        out = []
        while True:
            try:
                out.append(self._states.get_nowait())
            except (queue.Empty, OSError, EOFError, ValueError):
                return out

    def latest(self) -> BrainState | None:
        """Newest pending state, dropping older ones (non-blocking)."""
        states = self.poll()
        return states[-1] if states else None

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_layout"] = None
        return d


class BrainProcess:
    """Parent-side handle of the brain worker process."""

    def __init__(self, cfg: BrainConfig | None = None, start: bool = True):
        self.cfg = cfg or BrainConfig()
        ctx = mp.get_context("spawn")
        self._ctx = ctx
        self._cmd = ctx.Queue()
        self._status = ctx.Queue()
        self._subs = {name: BrainSubscriber(name, ctx.Queue(self.cfg.queue_size), ctx.Queue(1))
                      for name in self.cfg.subscribers}
        self._proc: mp.process.BaseProcess | None = None
        self._info: dict | None = None
        self.error: str | None = None
        if start:
            self.start()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._proc is not None:
            return
        pubs = {n: (s._states, s._layout_q) for n, s in self._subs.items()}
        self._proc = self._ctx.Process(
            target=_worker_main, name="perpetualfly-brain", daemon=True,
            args=(asdict(self.cfg), self._cmd, self._status, pubs))
        self._proc.start()
        atexit.register(self.stop)

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    def is_alive(self) -> bool:
        return bool(self._proc and self._proc.is_alive())

    def wait_ready(self, timeout: float = 60.0) -> dict:
        """Block until the worker has loaded the model; returns its info dict."""
        if self._info is not None:
            return self._info
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                kind, payload = self._status.get(timeout=0.2)
            except queue.Empty:
                if self._proc is not None and not self._proc.is_alive():
                    raise RuntimeError(f"brain worker died (exit {self._proc.exitcode})")
                continue
            if kind == "ready":
                self._info = payload
                return payload
            if kind == "error":
                self.error = payload
                raise RuntimeError(f"brain worker failed:\n{payload}")
        raise TimeoutError("brain worker not ready")

    def stop(self, timeout: float = 3.0) -> None:
        p = self._proc
        if p is None:
            return
        try:
            self._cmd.put_nowait(("stop", None))
        except Exception:
            pass
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join(1.0)
        if p.is_alive():
            p.kill()
            p.join(1.0)
        for q in [self._cmd, self._status] + [x for s in self._subs.values()
                                             for x in (s._states, s._layout_q)]:
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass
        self._proc = None
        try:
            atexit.unregister(self.stop)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()

    # ------------------------------------------------------------------ commands
    def send(self, ev: StimulusEvent) -> None:
        """Queue a stimulus (non-blocking; applied at the start of the next chunk)."""
        self._cmd.put_nowait(("stim", ev))

    def reset_state(self) -> None:
        """Clear stimuli and return every neuron to rest."""
        self._cmd.put_nowait(("reset_state", None))

    # ------------------------------------------------------------------ outputs
    def subscriber(self, name: str = "app") -> BrainSubscriber:
        return self._subs[name]

    def layout(self, timeout: float | None = 60.0, name: str | None = None) -> BrainLayout | None:
        return self._subs[name or self.cfg.subscribers[0]].layout(timeout)

    def latest(self, name: str | None = None) -> BrainState | None:
        return self._subs[name or self.cfg.subscribers[0]].latest()

    def poll(self, name: str | None = None) -> list[BrainState]:
        return self._subs[name or self.cfg.subscribers[0]].poll()


# ====================================================================== worker side
def _publish(q, item) -> None:
    """put_nowait, dropping the oldest item when the queue is full."""
    for _ in range(3):
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass


class _Model:
    """Engine + mapping + readout; everything the worker needs (also used in tests)."""

    def __init__(self, cfg: dict):
        from .engine import LIFEngine

        t0 = time.time()
        if cfg.get("synthetic"):
            self.table, (indptr, indices, w) = _synthetic_table(**cfg["synthetic"])
        else:
            from .data import load_connectome, load_neuron_table

            dd = Path(cfg["data_dir"]) if cfg.get("data_dir") else None
            indptr, indices, w, _ = load_connectome(dd)
            self.table = load_neuron_table(dd)
        from .layout import build_layout
        from .mapping import StimulusMapper, descending_indices

        self.engine = LIFEngine(indptr, indices, w, seed=cfg.get("seed", 0))
        del indptr, indices, w
        self.mapper = StimulusMapper(self.table, max_rate_hz=cfg.get("max_rate_hz", 200.0),
                                     whip_air_scale=cfg.get("whip_air_scale", 0.5),
                                     enable_ground_contact=cfg.get("enable_ground_contact", False))
        self.layout, self.display_idx = build_layout(self.table, cfg.get("n_per_region", 16),
                                                     cfg.get("seed", 0))
        self.dn = descending_indices(self.table)
        n = self.table.n
        self.disp_lut = np.full(n, -1, dtype=np.int64)
        self.disp_lut[self.display_idx] = np.arange(len(self.display_idx))
        nt = self.table.nt.astype(np.int64)
        self._nt_ok = nt >= 0
        self._nt = np.where(self._nt_ok, nt, 0)
        self.n_by_nt = np.bincount(self._nt[self._nt_ok], minlength=len(NEUROTRANSMITTERS)).astype(float)
        self.n_regions = len(self.table.regions)
        self.n_by_region = np.bincount(self.table.region, minlength=self.n_regions).astype(float)
        self.load_s = time.time() - t0
        self.engine.run(1)  # JIT warm-up (numba cache makes this fast after the first time)

    def summarize(self, steps: np.ndarray, idx: np.ndarray, window_s: float,
                  brain_time: float, rtf: float, labels: list[str]) -> BrainState:
        n = self.table.n
        counts = np.bincount(idx, minlength=n).astype(float) if len(idx) else np.zeros(n)
        w = max(window_s, 1e-9)
        ok = self._nt_ok
        spk_nt = np.bincount(self._nt[ok], weights=counts[ok], minlength=len(NEUROTRANSMITTERS))
        act_nt = np.bincount(self._nt[ok], weights=(counts[ok] > 0), minlength=len(NEUROTRANSMITTERS))
        with np.errstate(invalid="ignore", divide="ignore"):
            rate_nt = np.nan_to_num(spk_nt / (self.n_by_nt * w))
            frac_nt = np.nan_to_num(act_nt / self.n_by_nt)
            spk_r = np.bincount(self.table.region, weights=counts, minlength=self.n_regions)
            rate_r = np.nan_to_num(spk_r / (self.n_by_region * w))
        desc = {g: (float(counts[i].mean() / w) if len(i) else 0.0) for g, i in self.dn.items()}
        d = self.disp_lut[idx] if len(idx) else np.zeros(0, dtype=np.int64)
        sel = d >= 0
        dt_s = self.engine.p.dt * 1e-3
        return BrainState(
            brain_time=brain_time, wall_time=time.time(), realtime_factor=float(rtf),
            window_s=float(window_s), rate_by_nt=rate_nt.astype(np.float32),
            active_frac_by_nt=frac_nt.astype(np.float32), rate_by_region=rate_r.astype(np.float32),
            descending={g: desc[g] for g in DESCENDING_GROUPS},
            raster_idx=d[sel].astype(np.int32), raster_t=(steps[sel] * dt_s).astype(np.float64),
            total_spikes=int(len(idx)), recent_stimuli=list(labels))


def _synthetic_table(n: int = 50, p_conn: float = 0.1, seed: int = 0, **_):
    """Tiny annotated random network for tests (every NT, both sides, all DN groups)."""
    from .data import NeuronTable
    from .engine import random_network

    indptr, indices, w, sign = random_network(n, p_conn, seed)
    rng = np.random.default_rng(seed)
    nt = np.where(sign < 0, rng.choice([1, 2], n), rng.choice([0, 3, 4, 5], n)).astype(np.int8)
    side = np.array(["left", "right"] * (n // 2 + 1), dtype=object)[:n]
    sc = np.full(n, "central", dtype=object)
    ct = np.full(n, "", dtype=object)
    sub = np.full(n, "", dtype=object)
    # neurons 0..3 sensory (body mech), 4..17 descending groups
    sc[:4] = "sensory_ascending"
    sub[:4] = "SA_DMT_DMetaN"
    types = ["DNg100", "DNg100", "DNa02", "DNa02", "MDN", "MDN", "DNp01", "DNp01",
             "DNa01", "DNa01", "DNg97", "DNg97", "DNp09", "DNp09"]
    for k, t in enumerate(types):
        sc[4 + k] = "descending"
        ct[4 + k] = t
    regions = ["AL_L", "AL_R", "GNG", "OTHER"]
    region = (np.arange(n) % len(regions)).astype(np.int16)
    pos = np.column_stack([np.where(side == "left", 300.0, 700.0) + rng.normal(0, 50, n),
                           rng.uniform(100, 400, n), rng.uniform(0, 200, n)]).astype(np.float32)
    cols = {"super_class": sc, "cell_class": np.full(n, "", dtype=object), "cell_sub_class": sub,
            "cell_type": ct, "hemibrain_type": np.full(n, "", dtype=object), "side": side,
            "flow": np.full(n, "intrinsic", dtype=object)}
    table = NeuronTable(root_id=np.arange(n, dtype=np.int64) + 10**17, nt=nt,
                        sign=sign.astype(np.int8), region=region, regions=regions,
                        pos_um=pos, soma_um=pos.copy(), cols=cols)
    return table, (indptr, indices, w)


def _worker_main(cfg: dict, cmd_q, status_q, pubs: dict) -> None:
    try:
        _worker_loop(cfg, cmd_q, status_q, pubs)
    except Exception:
        try:
            status_q.put(("error", traceback.format_exc()))
        except Exception:
            pass


def _worker_loop(cfg: dict, cmd_q, status_q, pubs: dict) -> None:
    parent = mp.parent_process()
    model = _Model(cfg)
    eng, mapper = model.engine, model.mapper
    for states_q, layout_q in pubs.values():
        _publish(layout_q, model.layout)
    status_q.put(("ready", {"n_neurons": model.table.n, "load_s": model.load_s,
                            "n_display": len(model.display_idx), "pid": mp.current_process().pid}))

    dt_s = eng.p.dt * 1e-3
    chunk_steps = max(1, int(round(cfg.get("chunk_ms", 20.0) * 1e-3 / dt_s)))
    win_steps = max(1, int(round(cfg.get("window_s", 0.1) / dt_s)))
    realtime = bool(cfg.get("realtime", True))
    buf_steps: list[np.ndarray] = []
    buf_idx: list[np.ndarray] = []
    labels: list[str] = []
    rtf_hist: deque = deque(maxlen=50)  # (wall, brain) samples
    step0 = eng.step_count
    wall0 = time.perf_counter()
    next_pub = eng.step_count + win_steps
    running = True

    def handle(cmd) -> bool:
        kind, payload = cmd
        if kind == "stop":
            return False
        if kind == "stim":
            labels.extend(mapper.add(payload, eng.t))
        elif kind == "reset_state":
            mapper.add(StimulusEvent(kind="reset"), eng.t)
            eng.reset_state()
            labels.append("reset_state")
        return True

    while running:
        # 1) commands (non-blocking)
        while True:
            try:
                cmd = cmd_q.get_nowait()
            except queue.Empty:
                break
            if not handle(cmd):
                running = False
                break
        if not running or (parent is not None and not parent.is_alive()):
            break
        # 2) stimulus bookkeeping
        if mapper.expire(eng.t):
            idx, rates = mapper.drive()
            eng.set_poisson(idx, rates)
        # 3) simulate up to the next chunk / stimulus end / publish boundary
        n = min(chunk_steps, next_pub - eng.step_count)
        t_change = mapper.next_change()
        if np.isfinite(t_change):
            n = min(n, max(1, int(np.ceil((t_change - eng.t) / dt_s - 1e-9))))
        s, i = eng.run(max(1, n))
        if len(s):
            buf_steps.append(s)
            buf_idx.append(i)
        now = time.perf_counter()
        rtf_hist.append((now, eng.step_count))
        # 4) publish
        if eng.step_count >= next_pub:
            steps = np.concatenate(buf_steps) if buf_steps else np.zeros(0, np.int64)
            idx = np.concatenate(buf_idx) if buf_idx else np.zeros(0, np.int32)
            buf_steps.clear()
            buf_idx.clear()
            (w0, b0), (w1, b1) = rtf_hist[0], rtf_hist[-1]
            rtf = (b1 - b0) * dt_s / (w1 - w0) if w1 > w0 else 0.0
            st = model.summarize(steps, idx, win_steps * dt_s, eng.t, rtf, labels)
            labels = []
            for states_q, _ in pubs.values():
                _publish(states_q, st)
            next_pub += win_steps
        # 5) pace to wall time (wake early on commands)
        if realtime:
            ahead = (eng.step_count - step0) * dt_s - (time.perf_counter() - wall0)
            if ahead > 0.0005:
                try:
                    cmd = cmd_q.get(timeout=ahead)
                    if not handle(cmd):
                        break
                except queue.Empty:
                    pass
            elif ahead < -0.25:  # can't keep up: don't try to catch up later
                step0, wall0 = eng.step_count, time.perf_counter()
