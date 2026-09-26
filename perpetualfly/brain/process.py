"""Run the whole-brain model in its own process and publish BrainState snapshots.

Design (simplest robust one we found):

* ``BrainProcess(cfg)`` starts one worker with the ``spawn`` context (safe on macOS,
  no fork of the MuJoCo/OpenCV parent). The worker loads the connectome, builds the
  engine and the BrainLayout, then loops: apply queued StimulusEvents -> simulate one
  chunk (``chunk_ms`` of brain time) -> every ``window_s`` of brain time publish a
  BrainState. Brain time is paced to wall time (``realtime=True``): the worker never
  runs ahead of the clock; if the model is slower than real time it simply runs as
  fast as it can and ``BrainState.realtime_factor`` drops below 1.
* ``pace="sim"`` (what the fly app uses): brain time follows a clock supplied by the
  parent (``clock(t)``, the fly's simulated run time) instead of wall time. The
  worker never runs ahead of the latest clock mark, so a paused or slow fly pauses /
  slows the brain and brain and body stay causally consistent. Stimuli are applied
  at their ``StimulusEvent.sim_time`` (or immediately if that is already past). If
  the brain falls more than ``max_lag_s`` behind the clock it stops trying to catch
  up (the clock offset ``slip`` grows); ``BrainState.sim_time`` is the fly time the
  end of each state corresponds to, so the parent can show the lag.
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
    realtime: bool = True             # pace brain time to wall time (pace="wall")
    pace: str = "wall"                # "wall" | "sim" (follow clock() marks, see above)
    max_lag_s: float = 1.0            # pace="sim": max brain lag behind the clock
    seed: int = 0
    subscribers: tuple[str, ...] = ("app", "window")
    queue_size: int = 8               # states kept per subscriber before dropping
    n_per_region: int = 16            # display neurons sampled per region
    max_rate_hz: float = 200.0        # stimulus mapping (see mapping.StimulusMapper)
    whip_air_scale: float = 0.5
    enable_ground_contact: bool = False
    synthetic: dict | None = None     # tests: {"n": 50, "p_conn": .1, "seed": 0}
    extra: dict = field(default_factory=dict)
    # octopamine stress / arousal layer (perpetualfly/brain/neuromod.py,
    # NeuromodConfig fields); None = off, engine untouched. Also settable at run
    # time with BrainProcess.set_neuromod().
    neuromod: dict | None = None


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

    def reset_state(self, sim_time: float | None = None) -> None:
        """Clear stimuli and return every neuron to rest (pace="sim": at ``sim_time``)."""
        self._cmd.put_nowait(("reset_state", sim_time))

    def set_neuromod(self, cfg: dict | None) -> None:
        """Enable / reconfigure (dict of ``NeuromodConfig`` fields) or disable
        (None) the octopamine layer in the worker (applied at the next chunk)."""
        self._cmd.put_nowait(("neuromod", None if cfg is None else dict(cfg)))

    def set_lesions(self, targets) -> None:
        """Virtual lesions (docs/PLAYGROUND.md): silence exactly these targets
        (names for ``mapping.resolve_target``; empty = no lesion). Applied at the
        next chunk, not scheduled; survives ``reset_state``."""
        self._cmd.put_nowait(("lesion", [str(t) for t in (targets or [])]))

    def clock(self, sim_time: float) -> None:
        """pace="sim": the fly has simulated up to ``sim_time``; the brain may run to it."""
        self._cmd.put_nowait(("clock", float(sim_time)))

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
        from .mapping import MN9_IDS

        # extra readouts (BrainState.probes, Hz): the proboscis motor neuron MN9
        # (Shiu et al.'s sugar -> feeding readout)
        self.probes = {"MN9": self.table.index_of(MN9_IDS)}
        n = self.table.n
        self.disp_lut = np.full(n, -1, dtype=np.int64)
        self.disp_lut[self.display_idx] = np.arange(len(self.display_idx))
        nt = self.table.nt.astype(np.int64)
        self._nt_ok = nt >= 0
        self._nt = np.where(self._nt_ok, nt, 0)
        self.n_by_nt = np.bincount(self._nt[self._nt_ok], minlength=len(NEUROTRANSMITTERS)).astype(float)
        self.n_regions = len(self.table.regions)
        self.n_by_region = np.bincount(self.table.region, minlength=self.n_regions).astype(float)
        self.neuromod = None  # OctopamineModel (neuromod.py) when configured
        # brain playground (docs/PLAYGROUND.md): active lesions [(Target)], errors
        self.lesions: list = []
        self.pg_errors: list[str] = []
        self._pg_used = False
        if cfg.get("neuromod"):
            self.configure_neuromod(cfg["neuromod"])
        self.load_s = time.time() - t0
        self.engine.run(1)  # JIT warm-up (numba cache makes this fast after the first time)

    def configure_neuromod(self, d: dict | None) -> None:
        """None / {"enabled": False, ...}: off (thresholds back to the scalar
        model, the OA readout keeps running if the layer existed)."""
        from dataclasses import replace

        from .neuromod import OctopamineModel

        if d is None:
            if self.neuromod is not None:
                self.neuromod.configure(replace(self.neuromod.cfg, enabled=False))
            return
        if self.neuromod is None:
            self.neuromod = OctopamineModel(self.table, self.engine, d)
        else:
            self.neuromod.configure(d)

    def set_lesions(self, names: list[str]) -> None:
        from .mapping import resolve_target

        self.lesions = []
        for nm in names:
            tg = resolve_target(self.table, nm, self.mapper.sets)
            if not len(tg.idx):
                self.pg_errors.append(f"lesion {nm!r}: no neurons match")
                continue
            self.lesions.append(tg)
        self._pg_used = self._pg_used or bool(names)
        idx = np.concatenate([t.idx for t in self.lesions]) if self.lesions else None
        self.engine.set_silenced(idx)

    def note_stim(self, ev: StimulusEvent, labels: list[str]) -> None:
        if ev.kind == "opto":
            self._pg_used = True
            if not labels:
                self.pg_errors.append(
                    f"stim {(ev.details or {}).get('target')!r}: no neurons match")

    def playground_readout(self) -> dict:
        if not self._pg_used:
            return {}
        now = self.engine.t
        opto = [s for s in self.mapper.active if s.label.startswith("opto:")]
        sil = self.engine.silenced
        lut = self.disp_lut

        def disp(idx):
            d = lut[idx] if idx is not None and len(idx) else np.zeros(0, np.int64)
            return d[d >= 0].astype(np.int32)

        out = {
            "lesions": [{"target": t.name, "label": t.label, "n": int(len(t.idx))}
                        for t in self.lesions],
            "n_silenced": int(0 if sil is None else len(sil)),
            "opto": [{"label": s.label[5:], "rate_hz": float(s.rate_hz),
                      "n": int(len(s.idx)), "t_left": float(max(s.t_end - now, 0.0))}
                     for s in opto],
            "lesion_disp": disp(sil),
            "opto_disp": disp(np.concatenate([s.idx for s in opto]) if opto else None),
            "errors": list(self.pg_errors),
        }
        self.pg_errors = []
        return out

    def summarize(self, steps: np.ndarray, idx: np.ndarray, window_s: float,
                  brain_time: float, rtf: float, labels: list[str],
                  **extra) -> BrainState:
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
        probes = {k: (float(counts[i].mean() / w) if len(i) else 0.0)
                  for k, i in self.probes.items()}
        d = self.disp_lut[idx] if len(idx) else np.zeros(0, dtype=np.int64)
        sel = d >= 0
        dt_s = self.engine.p.dt * 1e-3
        return BrainState(
            brain_time=brain_time, wall_time=time.time(), realtime_factor=float(rtf),
            window_s=float(window_s), rate_by_nt=rate_nt.astype(np.float32),
            active_frac_by_nt=frac_nt.astype(np.float32), rate_by_region=rate_r.astype(np.float32),
            descending={g: desc[g] for g in DESCENDING_GROUPS},
            raster_idx=d[sel].astype(np.int32), raster_t=(steps[sel] * dt_s).astype(np.float64),
            total_spikes=int(len(idx)), recent_stimuli=list(labels), probes=probes,
            **({"neuromod": self.neuromod.readout()} if self.neuromod is not None else {}),
            **({"playground": pg} if (pg := self.playground_readout()) else {}),
            **extra)


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
    # neurons 0..3 sensory (body mech), 4..19 descending groups
    sc[:4] = "sensory_ascending"
    sub[:4] = "SA_DMT_DMetaN"
    types = ["DNg100", "DNg100", "DNa02", "DNa02", "MDN", "MDN", "DNp01", "DNp01",
             "DNa01", "DNa01", "DNg97", "DNg97", "DNp09", "DNp09", "DNg12_b", "DNg12_b"]
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
    import signal

    try:  # Ctrl-C in the terminal belongs to the parent (it stops us; we also exit
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # by ourselves if it dies)
    except ValueError:
        pass
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
    sim_paced = cfg.get("pace", "wall") == "sim"
    realtime = bool(cfg.get("realtime", True)) and not sim_paced
    max_lag = float(cfg.get("max_lag_s", 1.0))
    buf_steps: list[np.ndarray] = []
    buf_idx: list[np.ndarray] = []
    labels: list[str] = []
    rtf_hist: deque = deque(maxlen=50)  # (wall, brain) samples
    busy_hist: deque = deque(maxlen=50)  # (busy wall s, brain steps) per engine call
    step0 = eng.step_count
    wall0 = time.perf_counter()
    next_pub = eng.step_count + win_steps
    running = True
    seq = 0
    # pace="sim": latest clock mark (fly run time), clock offset, scheduled commands
    clock = eng.t
    slip = 0.0  # fly time - brain time (grows only when the brain gives up catching up)
    pending: list[tuple[float, int, tuple]] = []  # (fly time, order, cmd), sorted
    order = [0]

    def apply(cmd) -> None:
        nonlocal labels
        kind, payload = cmd
        if kind == "stim":
            new = mapper.add(payload, eng.t)
            labels.extend(new)
            model.note_stim(payload, new)
            if model.neuromod is not None:  # nociceptive relay (neuromod.py), if on
                for rev in model.neuromod.relay_events(payload):
                    labels.extend(lab.replace("manual:", "hit_relay:", 1)
                                  for lab in mapper.add(rev, eng.t))
        elif kind == "reset_state":
            mapper.add(StimulusEvent(kind="reset"), eng.t)
            eng.reset_state()
            labels.append("reset_state")
            if model.neuromod is not None:
                model.neuromod.on_reset()
        elif kind == "neuromod":
            model.configure_neuromod(payload)
        elif kind == "lesion":
            model.set_lesions(list(payload))
            labels.append("lesion:" + (", ".join(t.label for t in model.lesions) or "none"))

    def handle(cmd) -> bool:
        nonlocal clock
        kind, payload = cmd
        if kind == "stop":
            return False
        if kind == "clock":
            clock = max(clock, float(payload))
            return True
        if kind in ("neuromod", "lesion"):
            apply(cmd)
            return True
        if sim_paced and kind in ("stim", "reset_state"):
            t = payload.sim_time if kind == "stim" else payload
            if t is not None and float(t) - slip > eng.t + 0.5 * dt_s:
                order[0] += 1
                pending.append((float(t), order[0], cmd))
                pending.sort(key=lambda x: (x[0], x[1]))
                return True
        apply(cmd)
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
        # 1b) scheduled commands that are due
        while pending and pending[0][0] - slip <= eng.t + 0.5 * dt_s:
            apply(pending.pop(0)[2])
        # 2) stimulus bookkeeping
        if mapper.expire(eng.t):
            idx, rates = mapper.drive()
            eng.set_poisson(idx, rates)
        # 3) simulate up to the next chunk / stimulus end / publish boundary
        n = min(chunk_steps, next_pub - eng.step_count)
        t_change = mapper.next_change()
        if np.isfinite(t_change):
            n = min(n, max(1, int(np.ceil((t_change - eng.t) / dt_s - 1e-9))))
        if sim_paced:
            lag = clock - slip - eng.t
            if lag > max_lag:  # can't keep up: give up the excess instead of racing
                slip += lag - max_lag
            budget = int(np.floor((clock - slip - eng.t) / dt_s + 1e-6))
            if budget < 1:  # caught up with the fly: wait for the next clock mark
                try:
                    cmd = cmd_q.get(timeout=0.05)
                    if not handle(cmd):
                        break
                except queue.Empty:
                    pass
                continue
            n = min(n, budget)
            if pending:
                n = min(n, max(1, int(np.ceil((pending[0][0] - slip - eng.t) / dt_s - 1e-9))))
        t_run = time.perf_counter()
        s, i = eng.run(max(1, n))
        now = time.perf_counter()
        busy_hist.append((now - t_run, max(1, n)))
        if model.neuromod is not None:
            model.neuromod.observe(i, max(1, n) * dt_s)
        if len(s):
            buf_steps.append(s)
            buf_idx.append(i)
        rtf_hist.append((now, eng.step_count))
        # 4) publish
        if eng.step_count >= next_pub:
            steps = np.concatenate(buf_steps) if buf_steps else np.zeros(0, np.int64)
            idx = np.concatenate(buf_idx) if buf_idx else np.zeros(0, np.int32)
            buf_steps.clear()
            buf_idx.clear()
            (w0, b0), (w1, b1) = rtf_hist[0], rtf_hist[-1]
            rtf = (b1 - b0) * dt_s / (w1 - w0) if w1 > w0 else 0.0
            busy = sum(b for b, _ in busy_hist)
            crtf = sum(k for _, k in busy_hist) * dt_s / busy if busy > 0 else 0.0
            seq += 1
            st = model.summarize(steps, idx, win_steps * dt_s, eng.t, rtf, labels,
                                 seq=seq, compute_rtf=float(crtf),
                                 sim_time=float(eng.t + slip) if sim_paced else None)
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
