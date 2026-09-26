"""Brain replay: record BrainStates + key events during a run, replay them later.

Recording (``--brain-record``; docs/BRAIN_REPLAY.md). ``BrainRecorder`` lives in the
fly app (``BrainLink``) and writes ``<run dir>/brain_rec/``:

* ``layout.npz``: the static ``BrainLayout`` (once);
* ``chunk_00000.npz`` ...: ``np.savez_compressed`` chunks of ``chunk_states``
  states. States are first merged to at least ``window_s`` (0.1 s) of brain time
  (the low-latency pacing publishes 0.02 s states), rates window-weighted, the
  descending peak kept separately (``desc_peak``), the display-neuron raster capped
  at ``raster_cap`` spikes per stored state and stored as int16/uint16 offsets;
* ``events.jsonl``: stimuli sent to the brain and brain-triggered actions
  (appended as they happen);
* ``meta.json``: format, window, counts, bytes (rewritten at every chunk).

Recording stops (with one message) once the files exceed ``max_mb``; a run never
fills the disk. ``BrainRecording`` reads it back as BrainStates, ``ReplayPlayer``
feeds them to the brain window's ``BrainRenderer`` on a virtual clock (play /
pause / speed / seek) and draws a timeline bar with the events below the frame.
``scripts/brain_replay.py`` is the viewer / headless PNG-MP4 exporter.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from fly_simulator.brain.schema import (DESCENDING_GROUPS, NEUROTRANSMITTERS, BrainLayout,
                                       BrainState, StimulusEvent)

FORMAT = "fly_simulator-brain-rec/1"
REC_DIR = "brain_rec"
TIMELINE_H = 74
SEEK_STEP_S = 2.0
SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
PREROLL_S = 3.0  # history fed to a fresh renderer on seek (traces, raster)


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    return str(x)


# ============================================================================ writer
class BrainRecorder:
    def __init__(self, run_dir: Path | str, layout: BrainLayout | None = None, *,
                 window_s: float = 0.1, chunk_states: int = 100, max_mb: float = 50.0,
                 raster_cap: int = 1500, say=None) -> None:
        self.dir = Path(run_dir) / REC_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.window_s = float(window_s)
        self.chunk_states = int(max(chunk_states, 1))
        self.max_bytes = int(max_mb * 1e6)
        self.raster_cap = int(raster_cap)
        self.say = say or (lambda m: print(m, flush=True))
        self._buf: list[BrainState] = []
        self._rows: list[dict] = []
        self.n_chunks = 0
        self.n_states_in = 0
        self.n_states = 0
        self.n_events = 0
        self.n_raster_dropped = 0
        self.bytes = 0
        self.stopped = False
        self.t_first: float | None = None
        self.t_last: float | None = None
        self.probe_keys: list[str] | None = None
        self.n_regions: int | None = None
        self._ev = open(self.dir / "events.jsonl", "a", encoding="utf-8")
        if layout is not None:
            self.set_layout(layout)
        self._write_meta()

    # ................................................................ inputs
    def set_layout(self, layout: BrainLayout) -> None:
        p = self.dir / "layout.npz"
        np.savez_compressed(
            p, region_xy=np.asarray(layout.region_xy, np.float32),
            display_neuron_ids=np.asarray(layout.display_neuron_ids, np.int64),
            display_neuron_region=np.asarray(layout.display_neuron_region, np.int32),
            display_neuron_nt=np.asarray(layout.display_neuron_nt, np.int32),
            display_neuron_xy=np.asarray(layout.display_neuron_xy, np.float32),
            outline_len=np.array([len(o) for o in layout.region_outline_xy], np.int64),
            outline_xy=(np.concatenate([np.asarray(o, np.float32).reshape(-1, 2)
                                        for o in layout.region_outline_xy])
                        if layout.region_outline_xy else np.zeros((0, 2), np.float32)),
            json=np.array(json.dumps({"regions": list(layout.regions),
                                      "labels": list(layout.display_neuron_label),
                                      "n_neurons_total": int(layout.n_neurons_total),
                                      "model_name": str(layout.model_name)})))
        self.n_regions = len(layout.regions)
        self.bytes += p.stat().st_size

    def add_state(self, st: BrainState) -> None:
        if self.stopped:
            return
        self.n_states_in += 1
        self._buf.append(st)
        if sum(float(s.window_s or 0.0) for s in self._buf) >= self.window_s - 1e-9:
            self._merge_flush()

    def add_event(self, kind: str, t: float | None, label: str = "", **details) -> None:
        """A timeline event (``t`` = fly run time, like BrainState.sim_time)."""
        if self.stopped or self._ev.closed:
            return
        rec = {"kind": str(kind), "t": None if t is None else float(t), "label": str(label),
               **_jsonable(details)}
        line = json.dumps(rec) + "\n"
        self._ev.write(line)
        self._ev.flush()
        self.n_events += 1
        self.bytes += len(line)

    def add_stimulus(self, ev: StimulusEvent, source: str = "") -> None:
        lab = (ev.details or {}).get("label") or ev.kind
        self.add_event("stim", ev.sim_time, str(lab), stim_kind=ev.kind, side=ev.side,
                       intensity=float(ev.intensity), duration_s=float(ev.duration_s),
                       source=source, details=ev.details or {})

    # ................................................................ merging
    def _merge_flush(self) -> None:
        sts, self._buf = self._buf, []
        if not sts:
            return
        w = np.array([max(float(s.window_s or 0.0), 1e-9) for s in sts])
        W = float(w.sum())
        last = sts[-1]

        def wavg(get):
            a = np.stack([np.asarray(get(s), np.float64) for s in sts])
            return (a * w[:, None]).sum(0) / W

        if self.probe_keys is None:
            self.probe_keys = sorted(last.probes or {})
        desc = np.stack([[float(s.descending.get(g, 0.0)) for g in DESCENDING_GROUPS]
                         for s in sts])
        ridx = np.concatenate([np.asarray(s.raster_idx, np.int64) for s in sts])
        rt = np.concatenate([np.asarray(s.raster_t, np.float64) for s in sts])
        if len(ridx) > self.raster_cap:
            self.n_raster_dropped += len(ridx) - self.raster_cap
            keep = np.sort(np.random.default_rng(len(self._rows)).choice(
                len(ridx), self.raster_cap, replace=False))
            ridx, rt = ridx[keep], rt[keep]
        t0 = float(last.brain_time) - W
        stims = [x for s in sts for x in (s.recent_stimuli or [])]
        extra = {"neuromod": last.neuromod, "habituation": getattr(last, "habituation", {}),
                 "drive": last.drive, "playground": last.playground, "stims": stims}
        row = {
            "t_brain": float(last.brain_time),
            "t_sim": np.nan if last.sim_time is None else float(last.sim_time),
            "window": W, "rtf": float(last.realtime_factor or 0.0),
            "crtf": float(last.compute_rtf or 0.0), "seq": int(last.seq or 0),
            "total": int(sum(int(s.total_spikes) for s in sts)),
            "rate_nt": wavg(lambda s: s.rate_by_nt), "frac_nt": wavg(lambda s: s.active_frac_by_nt),
            "rate_region": wavg(lambda s: s.rate_by_region),
            "desc": (desc * w[:, None]).sum(0) / W, "desc_peak": desc.max(0),
            "probes": np.array([sum(float(s.probes.get(k, 0.0)) * wi for s, wi in zip(sts, w)) / W
                                for k in self.probe_keys]),
            "r_idx": ridx, "r_dt": np.clip(np.round((rt - t0) * 1e4), 0, 65535).astype(np.uint16),
            "json": json.dumps(_jsonable(extra)),
        }
        self._rows.append(row)
        self.n_states += 1
        tt = row["t_sim"] if np.isfinite(row["t_sim"]) else row["t_brain"]
        self.t_first = tt - W if self.t_first is None else self.t_first
        self.t_last = tt
        if len(self._rows) >= self.chunk_states:
            self._write_chunk()

    def _write_chunk(self) -> None:
        rows, self._rows = self._rows, []
        if not rows:
            return
        r_off = np.zeros(len(rows) + 1, np.int64)
        np.cumsum([len(r["r_idx"]) for r in rows], out=r_off[1:])
        idx = np.concatenate([r["r_idx"] for r in rows])
        p = self.dir / f"chunk_{self.n_chunks:05d}.npz"
        np.savez_compressed(
            p,
            t_brain=np.array([r["t_brain"] for r in rows]),
            t_sim=np.array([r["t_sim"] for r in rows]),
            window=np.array([r["window"] for r in rows], np.float32),
            rtf=np.array([r["rtf"] for r in rows], np.float32),
            crtf=np.array([r["crtf"] for r in rows], np.float32),
            seq=np.array([r["seq"] for r in rows], np.int32),
            total=np.array([r["total"] for r in rows], np.int32),
            rate_nt=np.stack([r["rate_nt"] for r in rows]).astype(np.float32),
            frac_nt=np.stack([r["frac_nt"] for r in rows]).astype(np.float32),
            rate_region=np.stack([r["rate_region"] for r in rows]).astype(np.float16),
            desc=np.stack([r["desc"] for r in rows]).astype(np.float32),
            desc_peak=np.stack([r["desc_peak"] for r in rows]).astype(np.float32),
            probes=np.stack([r["probes"] for r in rows]).astype(np.float32).reshape(len(rows), -1),
            r_off=r_off, r_idx=idx.astype(np.int16 if (idx.max(initial=0) < 32767) else np.int32),
            r_dt=np.concatenate([r["r_dt"] for r in rows]),
            json=np.array([r["json"] for r in rows]))
        self.n_chunks += 1
        self.bytes += p.stat().st_size
        self._write_meta()
        if self.bytes > self.max_bytes and not self.stopped:
            self.stopped = True
            self.say(f"[brain-record] size limit {self.max_bytes / 1e6:.0f} MB reached at "
                     f"t={self.t_last:.1f}s; recording stopped (raise max_mb to keep more)")
            self._write_meta()

    def _write_meta(self) -> None:
        meta = {"format": FORMAT, "window_s": self.window_s, "chunks": self.n_chunks,
                "states": self.n_states, "states_in": self.n_states_in,
                "events": self.n_events, "bytes": self.bytes, "t_first": self.t_first,
                "t_last": self.t_last, "probe_keys": self.probe_keys,
                "descending_groups": list(DESCENDING_GROUPS),
                "neurotransmitters": list(NEUROTRANSMITTERS),
                "raster_cap": self.raster_cap, "raster_dropped": self.n_raster_dropped,
                "stopped_at_limit": self.stopped, "max_mb": self.max_bytes / 1e6,
                "updated": time.time()}
        (self.dir / "meta.json").write_text(json.dumps(meta, indent=1))

    def close(self) -> dict:
        if not self.stopped:
            self._merge_flush()
            self._write_chunk()
        self._write_meta()
        if not self._ev.closed:
            self._ev.close()
        return self.summary()

    def summary(self) -> dict:
        dur = (self.t_last - self.t_first) if self.t_first is not None else 0.0
        return {"dir": str(self.dir), "states": self.n_states, "events": self.n_events,
                "duration_s": round(dur, 2), "mb": round(self.bytes / 1e6, 3),
                "mb_per_min": round(self.bytes / 1e6 / (dur / 60.0), 3) if dur > 0 else None,
                "stopped_at_limit": self.stopped}


# ============================================================================ reader
def find_recording(path: Path | str) -> Path:
    p = Path(path)
    if (p / REC_DIR / "meta.json").is_file():
        return p / REC_DIR
    if (p / "meta.json").is_file():
        return p
    raise FileNotFoundError(f"no brain recording in {p} (run with --brain-record)")


class BrainRecording:
    def __init__(self, path: Path | str) -> None:
        self.dir = find_recording(path)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.layout = self._load_layout()
        self.events = self._load_events()
        self.states: list[BrainState] = []
        self.desc_peak: list[np.ndarray] = []
        for p in sorted(self.dir.glob("chunk_*.npz")):
            self._load_chunk(p)
        self.desc_peak_arr = (np.stack(self.desc_peak) if self.desc_peak
                              else np.zeros((0, len(DESCENDING_GROUPS))))
        self.t = np.array([self.state_time(s) for s in self.states])  # window end times
        w = np.array([s.window_s for s in self.states])
        self.t_start = float(self.t[0] - w[0]) if len(self.t) else 0.0
        self.t_end = float(self.t[-1]) if len(self.t) else 0.0

    @staticmethod
    def state_time(s: BrainState) -> float:
        return float(s.sim_time) if s.sim_time is not None else float(s.brain_time)

    def _load_layout(self) -> BrainLayout:
        z = np.load(self.dir / "layout.npz")
        j = json.loads(str(z["json"]))
        ol, oxy = z["outline_len"], z["outline_xy"]
        outlines, k = [], 0
        for n in ol:
            outlines.append(oxy[k:k + int(n)])
            k += int(n)
        return BrainLayout(regions=j["regions"], region_xy=z["region_xy"],
                           region_outline_xy=outlines,
                           display_neuron_ids=z["display_neuron_ids"],
                           display_neuron_region=z["display_neuron_region"],
                           display_neuron_nt=z["display_neuron_nt"],
                           display_neuron_xy=z["display_neuron_xy"],
                           display_neuron_label=j["labels"],
                           n_neurons_total=j["n_neurons_total"], model_name=j["model_name"])

    def _load_events(self) -> list[dict]:
        p = self.dir / "events.jsonl"
        out = []
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a line cut by a crash
        return [e for e in out if e.get("t") is not None]

    def _load_chunk(self, p: Path) -> None:
        z = np.load(p)
        keys = self.meta.get("probe_keys") or []
        groups = self.meta.get("descending_groups") or list(DESCENDING_GROUPS)
        r_off = z["r_off"]
        for k in range(len(z["t_brain"])):
            ex = json.loads(str(z["json"][k]))
            tb, W = float(z["t_brain"][k]), float(z["window"][k])
            a, b = int(r_off[k]), int(r_off[k + 1])
            ts = float(z["t_sim"][k])
            hab = ex.get("habituation") or {}
            self.states.append(BrainState(
                brain_time=tb, wall_time=0.0, realtime_factor=float(z["rtf"][k]), window_s=W,
                rate_by_nt=z["rate_nt"][k], active_frac_by_nt=z["frac_nt"][k],
                rate_by_region=z["rate_region"][k].astype(np.float32),
                descending={g: float(z["desc"][k][i]) for i, g in enumerate(groups)},
                raster_idx=z["r_idx"][a:b].astype(np.int32),
                raster_t=(tb - W) + z["r_dt"][a:b].astype(np.float64) * 1e-4,
                total_spikes=int(z["total"][k]), recent_stimuli=list(ex.get("stims") or []),
                seq=int(z["seq"][k]), sim_time=ts if math.isfinite(ts) else None,
                compute_rtf=float(z["crtf"][k]),
                probes={key: float(z["probes"][k][i]) for i, key in enumerate(keys)},
                drive=ex.get("drive"), neuromod=ex.get("neuromod") or {},
                playground=_restore_pg(ex.get("playground") or {}),
                habituation=hab))
            self.desc_peak.append(z["desc_peak"][k])

    def index_at(self, t: float) -> int:
        """Index of the last state whose window ends at or before ``t`` (-1 = none)."""
        return int(np.searchsorted(self.t, t + 1e-9, side="right")) - 1

    def summary(self) -> dict:
        d = self.t_end - self.t_start
        mb = sum(f.stat().st_size for f in self.dir.iterdir() if f.is_file()) / 1e6
        return {"states": len(self.states), "events": len(self.events),
                "t_start": self.t_start, "t_end": self.t_end, "mb": round(mb, 3),
                "mb_per_min": round(mb / (d / 60.0), 3) if d > 0 else None}


def _restore_pg(pg: dict) -> dict:
    pg = dict(pg)
    for k in ("lesion_disp", "opto_disp"):
        if k in pg and pg[k] is not None:
            pg[k] = np.asarray(pg[k], np.int32)
    return pg


# ============================================================================ player
EVENT_COLORS = {  # BGR
    "loom": (60, 150, 255), "whip": (80, 80, 240), "shove": (80, 80, 240),
    "hit": (80, 80, 240), "sugar": (120, 210, 110), "bitter": (200, 120, 200),
    "action": (230, 220, 90), "reset": (150, 150, 150), "opto": (255, 170, 90),
    "other": (170, 160, 150),
}


def event_class(e: dict) -> str:
    if e.get("kind") == "action":
        return "action"
    txt = f"{e.get('stim_kind', '')} {e.get('label', '')}".lower()
    for key in ("loom", "whip", "shove", "sugar", "bitter", "reset", "opto"):
        if key in txt:
            return key
    if "lc4" in txt or "lplc2" in txt:
        return "loom"
    return "other"


class ReplayPlayer:
    """Replays a ``BrainRecording`` through a ``BrainRenderer`` on a virtual clock."""

    def __init__(self, rec: BrainRecording, size=None, playground: bool = True,
                 speed: float = 1.0) -> None:
        from fly_simulator.brain_viz.window import DEFAULT_SIZE

        self.rec = rec
        self.size = tuple(size or DEFAULT_SIZE)
        self.playground = playground
        self.speed = float(speed)
        self.paused = False
        self.t = rec.t_start
        self._vwall = 1000.0  # renderer clock (virtual wall seconds)
        self._next = 0  # next state index to feed
        self._next_ev = 0
        self.renderer = None
        self.seek(rec.t_start)

    # ................................................................ control
    def _new_renderer(self):
        from fly_simulator.brain_viz.window import BrainRenderer

        r = BrainRenderer(self.rec.layout, size=self.size, playground=self.playground)
        return r

    def seek(self, t: float) -> None:
        """Jump to fly time ``t``: fresh renderer pre-fed with ``PREROLL_S`` of history."""
        rec = self.rec
        t = float(min(max(t, rec.t_start), rec.t_end))
        self.t = t
        r = self._new_renderer()
        self._vwall = 1000.0
        i1 = rec.index_at(t) + 1
        i0 = rec.index_at(t - PREROLL_S) + 1
        r.step(self._vwall)
        fps = 30.0
        ev_i = 0
        evs = rec.events
        for i in range(i0, i1):
            s = rec.states[i]
            st = rec.state_time(s)
            while ev_i < len(evs) and evs[ev_i]["t"] <= st:
                if evs[ev_i]["t"] > t - PREROLL_S:
                    self._feed_event(r, evs[ev_i])
                ev_i += 1
            r.ingest(s, self._vwall)
            for _ in range(max(1, int(round(s.window_s * fps)))):
                self._vwall += 1.0 / fps
                r.step(self._vwall)
        self.renderer = r
        self._next = i1
        self._next_ev = int(np.searchsorted([e["t"] for e in evs], t, side="right"))

    def _feed_event(self, r, e: dict) -> None:
        if e.get("kind") == "stim":
            r.ingest(StimulusEvent(str(e.get("stim_kind", "manual")), str(e.get("side", "none")),
                                   float(e.get("intensity", 1.0)),
                                   float(e.get("duration_s", 0.0)), float(e["t"]),
                                   details=dict(e.get("details") or {})), self._vwall)
        elif e.get("kind") == "action":
            r.ingest(StimulusEvent("manual", "none", 1.0, 0.0, float(e["t"]),
                                   details={"label": f"ACTION {e.get('label', '')}".strip()}),
                     self._vwall)

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def faster(self, k: int = 1) -> None:
        i = int(np.argmin([abs(math.log(s / self.speed)) for s in SPEEDS]))
        self.speed = SPEEDS[int(np.clip(i + k, 0, len(SPEEDS) - 1))]

    def handle_key(self, key: str) -> bool:
        """space / left / right / [ / ] / home; False for q / esc (quit)."""
        if key in ("q", "esc"):
            return False
        if key == " ":
            self.toggle_pause()
        elif key == "left":
            self.seek(self.t - SEEK_STEP_S)
        elif key == "right":
            self.seek(self.t + SEEK_STEP_S)
        elif key == "[":
            self.faster(-1)
        elif key == "]":
            self.faster(+1)
        elif key == "home":
            self.seek(self.rec.t_start)
        return True

    # ................................................................ time
    def advance(self, dt_wall: float) -> None:
        """Advance the virtual clock by ``dt_wall`` wall seconds (x speed unless paused)."""
        dt_wall = max(float(dt_wall), 0.0)
        self._vwall += dt_wall
        if self.paused:
            return
        rec = self.rec
        self.t = min(self.t + dt_wall * self.speed, rec.t_end)
        evs = rec.events
        while self._next_ev < len(evs) and evs[self._next_ev]["t"] <= self.t:
            self._feed_event(self.renderer, evs[self._next_ev])
            self._next_ev += 1
        while self._next < len(rec.states) and rec.t[self._next] <= self.t + 1e-9:
            self.renderer.ingest(rec.states[self._next], self._vwall)
            self._next += 1
        if self.t >= rec.t_end:
            self.paused = True

    @property
    def done(self) -> bool:
        return self.t >= self.rec.t_end - 1e-9

    # ................................................................ drawing
    def frame(self, settle: bool = False) -> np.ndarray:
        r = self.renderer
        r.step(self._vwall)
        if settle:
            r.settle()
        img = r.draw(self._vwall)
        return np.vstack([img, self.timeline()])

    def timeline(self) -> np.ndarray:
        import cv2

        from fly_simulator.brain_viz.window import (BG, BORDER, DIM, ESC_COL, FAINT, TEXT,
                                                   put_text)

        rec = self.rec
        W, H = self.size[0], TIMELINE_H
        img = np.full((H, W, 3), BG, np.uint8)
        cv2.line(img, (0, 0), (W, 0), BORDER, 1)
        x0, x1, y0, y1 = 14, W - 14, 28, H - 10
        span = max(rec.t_end - rec.t_start, 1e-6)

        def X(t):
            return int(round(x0 + (t - rec.t_start) / span * (x1 - x0)))

        cv2.rectangle(img, (x0, y0), (x1, y1), (32, 27, 25), -1)
        # giant-fibre (escape) peak trace
        if len(rec.t):
            gi = DESCENDING_GROUPS.index("escape")
            gf = rec.desc_peak_arr[:, gi]
            top = max(float(gf.max()), 60.0)
            pts = np.stack([np.array([X(t) for t in rec.t]),
                            (y1 - np.clip(gf / top, 0, 1) * (y1 - y0 - 2)).astype(int)], 1)
            cv2.polylines(img, [pts.astype(np.int32)], False, ESC_COL, 1, cv2.LINE_AA)
            thr_y = int(y1 - 60.0 / top * (y1 - y0 - 2))
            cv2.line(img, (x0, thr_y), (x1, thr_y), FAINT, 1)
        # events
        for e in rec.events:
            c = event_class(e)
            col = EVENT_COLORS.get(c, EVENT_COLORS["other"])
            x = X(float(e["t"]))
            if c == "action":
                cv2.fillConvexPoly(img, np.array([[x, y0 - 2], [x - 4, y0 - 9], [x + 4, y0 - 9]],
                                                 np.int32), col, cv2.LINE_AA)
            else:
                dur = float(e.get("duration_s") or 0.0)
                xe = max(X(float(e["t"]) + dur), x + 1)
                cv2.rectangle(img, (x, y1 - 5), (xe, y1), col, -1)
                cv2.line(img, (x, y0), (x, y1), col, 1)
        # playhead
        xp = X(self.t)
        cv2.line(img, (xp, y0 - 10), (xp, y1 + 3), TEXT, 2)
        status = "PAUSED" if self.paused else "PLAYING"
        i = rec.index_at(self.t)
        hab = (rec.states[i].habituation or {}) if i >= 0 else {}
        hab_txt = (f"  habituation {float(hab.get('efficacy', 1.0)):.2f}"
                   if hab.get("enabled") else "")
        put_text(img, f"REPLAY  t {self.t:7.2f} s / {rec.t_end:.2f}  x{self.speed:g}  {status}"
                      f"{hab_txt}", (x0, 18), TEXT, 13, 700)
        put_text(img, "space play/pause  ←/→ seek 2 s  [ ] speed  click bar: seek  "
                      "q quit   ▼ action  | stimulus   red: GF peak (line = 60 Hz)",
                 (W - 14, 18), DIM, 11, 400, "right")
        return img

    def time_at_x(self, x: float) -> float | None:
        W = self.size[0]
        x0, x1 = 14, W - 14
        if not x0 - 6 <= x <= x1 + 6:
            return None
        f = (x - x0) / (x1 - x0)
        return self.rec.t_start + float(np.clip(f, 0, 1)) * (self.rec.t_end - self.rec.t_start)


# ============================================================================ headless
def render_png(rec: BrainRecording, t: float, out: Path | str, size=None,
               playground: bool = True) -> Path:
    import cv2

    pl = ReplayPlayer(rec, size=size, playground=playground)
    pl.seek(t)
    pl.paused = True
    out = Path(out)
    cv2.imwrite(str(out), pl.frame(settle=True))
    return out


def render_range(rec: BrainRecording, t0: float, t1: float, out: Path | str, fps: float = 30.0,
                 speed: float = 1.0, size=None, playground: bool = True) -> int:
    """MP4 (or a directory of PNGs if ``out`` has no .mp4 suffix) of [t0, t1]."""
    import cv2

    pl = ReplayPlayer(rec, size=size, playground=playground, speed=speed)
    pl.seek(t0)
    out = Path(out)
    n = int(max(1, round((min(t1, rec.t_end) - pl.t) / speed * fps)))
    vw = None
    for k in range(n):
        img = pl.frame()
        if out.suffix.lower() == ".mp4":
            if vw is None:
                h, w = img.shape[:2]
                vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            vw.write(img)
        else:
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"frame_{k:05d}.png"), img)
        pl.advance(1.0 / fps)
    if vw is not None:
        vw.release()
    return n


# ============================================================================ GUI
def run_viewer(rec: BrainRecording, speed: float = 1.0, start: float | None = None,
               playground: bool = True, fps: float = 30.0, size=None) -> None:
    import cv2

    from fly_simulator.brain_viz.window import auto_display_scale

    pl = ReplayPlayer(rec, size=size, playground=playground, speed=speed)
    if start is not None:
        pl.seek(start)
    title = "PerpetualFly - Brain replay"
    h = pl.size[1] + TIMELINE_H
    scale = auto_display_scale((pl.size[0], h))
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags=0, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            sx, sy = x / scale, y / scale
            if sy >= pl.size[1]:
                t = pl.time_at_x(sx)
                if t is not None:
                    pl.seek(t)

    cv2.setMouseCallback(title, on_mouse)
    # arrow key codes of cv2.waitKeyEx (macOS Cocoa / GTK / Windows)
    arrows = {63234: "left", 63235: "right", 65361: "left", 65363: "right",
              2424832: "left", 2555904: "right", 63273: "home", 65360: "home"}
    last = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            pl.advance(now - last)
            last = now
            img = pl.frame()
            if abs(scale - 1.0) > 0.02:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                                 if scale < 1 else cv2.INTER_CUBIC)
            cv2.imshow(title, img)
            code = cv2.waitKeyEx(max(1, int(1000 / fps)))
            if code != -1:
                key = arrows.get(code)
                if key is None:
                    c = code & 0xFF
                    key = "esc" if c == 27 else chr(c).lower() if 32 <= c < 127 else ""
                if key and not pl.handle_key(key):
                    break
            try:
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        cv2.destroyWindow(title)
        for _ in range(3):
            cv2.waitKey(1)


__all__ = ["BrainRecorder", "BrainRecording", "ReplayPlayer", "render_png", "render_range",
           "run_viewer", "find_recording"]
