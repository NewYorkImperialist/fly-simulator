"""Race logic for an installed course: timer, checkpoints, gates, triggers,
fall penalties, explicit respawns, DNF, results + leaderboard.

Threading: ``_post_step`` runs as a sim post-step hook; ``after_physics`` must run
once per physics chunk outside ``sim.step`` (``install_course`` chains it onto
``Session.after_physics``, so the app's loop and the standalone runner both call
it under the physics lock). Respawns reset the sim, so they only happen there.

Respawn = explicit reset. The respawn point is written into FlyGym's "neutral"
keyframe (free-joint x / y), so ``sim.reset()`` spawns the fly standing at the last
checkpoint (flat floor by construction) and every reset hook (metrics, fall
detector, whip, brain) sees a normal reset. It is logged as
``course_respawn_reset`` in events.csv, counted in ``RunMetrics.n_resets`` and in
the course results. Any other reset during a race (X key, app auto reset) also
lands at the last checkpoint and is counted as ``n_external_resets``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import mujoco as mj

from fly_simulator.course import leaderboard as lb
from fly_simulator.course.layout import CourseLayout
from fly_simulator.course.spec import CourseSpec
from fly_simulator.metrics import FallState
from fly_simulator.terrain.sectioned import SectionedGenerator

if TYPE_CHECKING:
    from fly_simulator.app import Session

# whip crack "from the left" pushes the fly to its right, etc.
SHOVE_FOR_SIDE = {"left": "right", "right": "left", "front": "backward", "rear": "forward",
                  "overhead": "up", "random": "random"}
READY, RUNNING, FINISHED, DNF, ABORTED = "ready", "running", "finished", "dnf", "aborted"


@dataclass
class CourseOptions:
    """Run-time options (None = take the course file's value)."""
    after_finish: str | None = None  # stop | loop
    timeout_s: float | None = None
    respawn_after_s: float | None = None
    steer: bool | None = None
    # leaderboard file; None = <cfg.logging.runs_dir>/leaderboard.json, "" = off
    leaderboard: str | None = None
    results_dir: str | None = None  # None = the session's run dir (if logging)
    controller_mode: str | None = None  # None = derived from the AppConfig
    loop_delay_s: float = 1.0  # after the finish, before the next lap starts
    check_every_steps: int = 20  # race bookkeeping interval (2 ms at dt = 1e-4)


class CourseRun:
    def __init__(self, session: "Session", spec: CourseSpec, layout: CourseLayout,
                 options: CourseOptions | None = None,
                 stimulus_fn: Callable | None = None) -> None:
        self.session = session
        self.sim = session.sim
        self.spec = spec
        self.layout = layout
        self.opt = options or CourseOptions()
        o = self.opt
        self.after_finish = o.after_finish or spec.after_finish
        self.timeout_s = spec.timeout_s if o.timeout_s is None else o.timeout_s
        self.respawn_after_s = spec.respawn_after_s if o.respawn_after_s is None else o.respawn_after_s
        self.mode = o.controller_mode or lb.controller_mode(session.cfg)
        ctrl = self.sim.controller
        self.can_steer = ctrl.cfg.kind == "hybrid" and ctrl.cfg.heading_gain > 0
        self.steer = (spec.steer if o.steer is None else o.steer) and self.can_steer
        self.stimulus_fn = stimulus_fn
        self.say = session.say
        if self.opt.leaderboard is None:
            self.leaderboard_path = str(Path(session.cfg.logging.runs_dir) / "leaderboard.json")
        else:
            self.leaderboard_path = self.opt.leaderboard
        self.respawn_listeners: list[Callable[["CourseRun"], None]] = []
        self.laps: list[dict] = []
        self.installed = False
        self._finalized = False
        self._new_lap(lap=1)

    # ----------------------------------------------------------------- state
    def _new_lap(self, lap: int) -> None:
        self.lap = lap
        self.status = READY
        self.t_start: float | None = None
        self.t_end: float | None = None
        self.splits: list[dict] = []
        self.gate_results: list[dict] = []
        self.falls: list[float] = []
        self.respawns: list[dict] = []
        self.n_external_resets = 0
        self.n_whip = 0
        self.n_loom = 0
        self.next_cp = 0
        self.next_gate = 0
        self.next_trigger = 0
        self.max_x = 0.0
        self.fallen_since: float | None = None
        self._pending_loom: list[tuple[float, dict]] = []
        self._finish_time_wall: float | None = None
        self.respawn_x = self.respawn_y = 0.0
        self.rank: int | None = None
        self._in_reset = False

    def now(self) -> float:
        """Monotonic run time (continues across resets)."""
        return self.session.metrics.run_time_at(self.sim.time)

    @property
    def race_time(self) -> float:
        if self.t_start is None:
            return 0.0
        return (self.t_end if self.t_end is not None else self.now()) - self.t_start

    @property
    def penalty_s(self) -> float:
        s = self.spec
        return (len(self.falls) * s.fall_penalty_s + len(self.respawns) * s.respawn_penalty_s
                + sum(not g["passed"] for g in self.gate_results) * s.gate_miss_penalty_s)

    @property
    def done(self) -> bool:
        """True once the race is over and the course does not loop."""
        return self.status in (FINISHED, DNF) and self.after_finish == "stop"

    @property
    def last_checkpoint(self) -> int:
        return self.next_cp  # number of checkpoints reached (0 = start)

    # --------------------------------------------------------------- install
    def install(self) -> "CourseRun":
        s, sim, t = self.session, self.sim, self.session.terrain
        tc = t.cfg
        self._saved_terrain = (t.generator, tc.chunk_length, tc.chunks_behind, tc.chunks_ahead,
                               tc.lateral_snap)
        tc.chunk_length = self.spec.chunk_length
        tc.chunks_behind, tc.chunks_ahead = 1, t.n_slots - 2
        tc.lateral_snap = 0.0
        gen = SectionedGenerator(self.layout.geoms, self.layout.segments, tc, seed=self.spec.seed)
        problems = gen.validate()
        if problems:
            self._restore_terrain()
            raise ValueError(f"course {self.spec.name!r} does not fit the terrain pool:\n  "
                             + "\n  ".join(problems))
        t.generator = gen
        m = sim.model
        self._key = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, "neutral")
        self._adr = sim._free_qpos
        self._key_xy0 = m.key_qpos[self._key, self._adr:self._adr + 2].copy()
        self._target0 = sim.controller._target_heading
        sim.post_step_hooks.append(self._post_step)
        sim.pre_reset_hooks.append(self._pre_reset)
        s.detector.add_listener(self._on_fall_event)
        # chain onto the session so the app loop drives the course unchanged
        self._orig = {k: getattr(s, k) for k in
                      ("after_physics", "close", "summary_extra", "step_difficulty", "flatten")}
        s.after_physics = self._after_physics_chained
        s.close = self._close_chained
        s.summary_extra = self._summary_extra_chained
        s.step_difficulty = lambda delta: "[course] terrain difficulty is fixed in course mode"
        s.flatten = lambda: "[course] flatten is disabled in course mode"
        s.course = self
        self.installed = True
        x, y, _ = sim.thorax_position()
        t.relayout(float(x), float(y))
        s.log_event("course_install", course=self.spec.name, path=self.spec.path,
                    length_mm=round(self.layout.length, 2), mode=self.mode,
                    after_finish=self.after_finish, steer=self.steer)
        return self

    def _restore_terrain(self) -> None:
        t = self.session.terrain
        (t.generator, t.cfg.chunk_length, t.cfg.chunks_behind, t.cfg.chunks_ahead,
         t.cfg.lateral_snap) = self._saved_terrain

    def uninstall(self) -> None:
        """Back to endless terrain (the fly stays where it is)."""
        if not self.installed:
            return
        s, sim = self.session, self.sim
        sim.post_step_hooks.remove(self._post_step)
        sim.pre_reset_hooks.remove(self._pre_reset)
        s.detector.listeners.remove(self._on_fall_event)
        for k, v in self._orig.items():
            setattr(s, k, v)
        s.course = None
        sim.model.key_qpos[self._key, self._adr:self._adr + 2] = self._key_xy0
        sim.controller._target_heading = self._target0
        self._restore_terrain()
        x, y, _ = sim.thorax_position()
        s.terrain.relayout(float(x), float(y))
        self.installed = False

    def _set_respawn(self, x: float, y: float) -> None:
        self.respawn_x, self.respawn_y = float(x), float(y)
        self.sim.model.key_qpos[self._key, self._adr:self._adr + 2] = (
            self._key_xy0 + (self.respawn_x, self.respawn_y))

    # ---------------------------------------------------------------- chains
    def _after_physics_chained(self) -> None:
        self._orig["after_physics"]()
        self.after_physics()

    def _close_chained(self, quit_reason: str) -> None:
        self.finalize(quit_reason)
        self._orig["close"](quit_reason)

    def _summary_extra_chained(self, quit_reason: str) -> dict:
        d = self._orig["summary_extra"](quit_reason)
        d["course"] = self.summary()
        return d

    # ----------------------------------------------------------------- hooks
    def _pre_reset(self, sim) -> None:
        # after the terrain's own pre-reset (which lays out around x = 0)
        self.session.terrain.relayout(self.respawn_x, self.respawn_y)

    def _on_fall_event(self, ev) -> None:
        if ev.kind in ("fall", "relapse"):
            self.fallen_since = self.now()
            if ev.kind == "fall" and self.status == RUNNING:
                self.falls.append(round(self.race_time, 3))
                self._log("course_fall", reason=ev.reason, n_falls=len(self.falls))
        elif ev.kind in ("recovering", "recovered", "reset"):
            self.fallen_since = None
            if ev.kind == "reset" and not self._in_reset and self.status == RUNNING:
                self.n_external_resets += 1

    def _post_step(self, sim) -> None:
        if sim.step_count % self.opt.check_every_steps:
            return
        d = sim.data.xpos[sim.thorax_body_id]
        x, y = float(d[0]), float(d[1])
        lay = self.layout
        if self.status == READY and x >= lay.start_x:
            self.status = RUNNING
            self.t_start = self.now()
            self._log("course_start", lap=self.lap)
            self.say(f"[course] {self.spec.name} lap {self.lap}: GO")
        if self.status == RUNNING:
            self.max_x = max(self.max_x, x)
            t = self.race_time
            cps = lay.checkpoints
            while self.next_cp < len(cps) and x >= cps[self.next_cp].x:
                cp = cps[self.next_cp]
                self.next_cp += 1
                split = {"checkpoint": cp.index, "name": cp.name, "x": round(cp.x, 2),
                         "time_s": round(t, 3)}
                self.splits.append(split)
                self._set_respawn(cp.x, lay.route_y(cp.x))
                self._log("course_checkpoint", **split)
                self.say(f"[course] {cp.name} split {t:.2f}s")
            gates = lay.gates
            while self.next_gate < len(gates) and x >= gates[self.next_gate].x:
                g = gates[self.next_gate]
                self.next_gate += 1
                ok = (y > g.y) if g.side == "left" else (y < g.y)
                res = {"gate": g.index, "x": round(g.x, 2), "side": g.side,
                       "fly_y": round(y - g.y, 3), "passed": bool(ok)}
                self.gate_results.append(res)
                self._log("course_gate", **res)
                if not ok:
                    self.say(f"[course] gate {g.index + 1} missed (wanted {g.side}) "
                             f"+{self.spec.gate_miss_penalty_s:g}s")
            trg = lay.triggers
            while self.next_trigger < len(trg) and x >= trg[self.next_trigger].x:
                self._fire(trg[self.next_trigger])
                self.next_trigger += 1
            if self._pending_loom and self._pending_loom[0][0] <= self.now():
                self._send_loom(self._pending_loom.pop(0)[1])
            if x >= lay.finish_x:
                self._end(FINISHED)
            elif t > self.timeout_s:
                self._end(DNF)
        if self.steer:
            look = self.spec.steer_lookahead
            yt = lay.route_y(x + look) if self.status in (READY, RUNNING) else y
            sim.controller._target_heading = math.atan2(yt - y, look)

    # -------------------------------------------------------------- triggers
    def _fire(self, trg) -> None:
        s = self.session
        if trg.kind == "whip":
            side, level = trg.params["side"], trg.params["level"]
            if s.whip is not None and s.hit_mode == "whip":
                level = min(max(level, 1), len(s.whip.cfg.levels))
                r = s.whip.crack(side, level, source="course")
                how = f"whip crack from {side} L{level} ({r})"
            else:
                level = min(max(level, 1), len(s.perturbation.cfg.levels))
                s.perturbation.hit(SHOVE_FOR_SIDE[side], level=level, source="course")
                how = f"shove {SHOVE_FOR_SIDE[side]} L{level}"
            self.n_whip += 1
            self._log("course_trigger", trigger=trg.index, kind="whip", how=how)
            self.say(f"[course] gauntlet: {how}")
        elif trg.kind == "loom":
            p = trg.params
            now = self.now()
            for k in range(int(p["repeats"])):
                self._pending_loom.append((now + k * p["interval_s"], p))
            self._pending_loom.sort(key=lambda e: e[0])
            if self._pending_loom[0][0] <= now:
                self._send_loom(self._pending_loom.pop(0)[1])

    def _send_loom(self, p: dict) -> None:
        from fly_simulator.brain.schema import StimulusEvent

        ev = StimulusEvent("manual", "none", 1.0, float(p["duration_s"]), self.session.run_time(),
                           details={"set": p["set"], "label": f"LOOM {p['set']} (course)"})
        self.n_loom += 1
        sent_to = "log only"
        if self.stimulus_fn is not None:
            self.stimulus_fn(ev)
            sent_to = "hook"
        elif self.session.brain is not None:
            self.session.brain.send(ev, source="course")
            sent_to = "brain"
        self._log("course_loom", set=p["set"], duration_s=p["duration_s"], sent_to=sent_to)
        self.say(f"[course] looming stimulus {p['set']} ({sent_to})")

    # ---------------------------------------------------------- after_physics
    def after_physics(self) -> None:
        """Once per physics chunk (outside sim.step): respawns and lap restarts."""
        if not self.installed:
            return
        now = self.now()
        if (self.status == RUNNING and self.fallen_since is not None
                and self.session.detector.state == FallState.FALLEN
                and now - self.fallen_since >= self.respawn_after_s):
            self.respawn("fallen")
        elif (self.status in (FINISHED, DNF) and self.after_finish == "loop"
              and self.t_end is not None and now - self.t_end >= self.opt.loop_delay_s):
            self._restart_lap()

    def _reset(self, event: str, **details) -> None:
        s = self.session
        s.log_event(event, down_for_s=s.down_for(), state=s.detector.state.value,
                    x=round(self.respawn_x, 2), **details)
        self._in_reset = True
        try:
            self.sim.reset()
        finally:
            self._in_reset = False
        s.down_since = None
        self.fallen_since = None
        for fn in self.respawn_listeners:
            fn(self)

    def respawn(self, reason: str = "fallen") -> None:
        """Explicit reset to the last checkpoint (never hidden; see module doc)."""
        cp = self.last_checkpoint
        down = self.session.down_for()
        rec = {"time_s": round(self.race_time, 3), "checkpoint": cp, "x": round(self.respawn_x, 2),
               "reason": reason, "down_for_s": None if down is None else round(down, 3)}
        self.respawns.append(rec)
        self.say(f"[course] respawn #{len(self.respawns)} at "
                 f"{'start' if cp == 0 else self.layout.checkpoints[cp - 1].name} "
                 f"({reason}, down {rec['down_for_s']}s) - explicit reset")
        self._reset("course_respawn_reset", checkpoint=cp, reason=reason,
                    n_respawns=len(self.respawns))
        # Triggers are *not* re-armed: the sim is deterministic, so re-firing the
        # crack that knocked the fly over would replay the same fall forever.
        self._pending_loom.clear()
        if len(self.respawns) >= self.spec.max_respawns:
            self._end(DNF, reason=f"{len(self.respawns)} respawns (max_respawns)")

    def _restart_lap(self) -> None:
        self._set_respawn(0.0, 0.0)
        self._reset("course_lap_reset", lap=self.lap + 1)
        self._new_lap(self.lap + 1)
        self._set_respawn(0.0, 0.0)

    # ---------------------------------------------------------------- results
    def _end(self, status: str, reason: str = "") -> None:
        self.t_end = self.now()
        self.status = status
        res = self.result()
        if status == DNF:
            res["dnf_reason"] = reason or f"timeout {self.timeout_s:g}s"
        if status == FINISHED and self.leaderboard_path:
            entry = {k: res[k] for k in ("total_time_s", "time_s", "penalty_s", "n_falls",
                                         "n_respawns", "gates_missed", "date", "run_dir",
                                         "lap", "seed")}
            self.rank = lb.add_entry(self.leaderboard_path, self.spec.name, self.mode, entry)
            res["rank"] = self.rank
        self.laps.append(res)
        self._log(f"course_{status}", time_s=res["time_s"], total_time_s=res["total_time_s"],
                  penalty_s=res["penalty_s"], rank=self.rank)
        if status == FINISHED:
            rank = f" - leaderboard #{self.rank} ({self.mode})" if self.rank else ""
            self.say(f"[course] FINISH {self.spec.name}: {res['total_time_s']:.2f}s "
                     f"(race {res['time_s']:.2f}s + penalties {res['penalty_s']:.1f}s, "
                     f"falls {res['n_falls']}, respawns {res['n_respawns']}){rank}")
        else:
            self.say(f"[course] DNF {self.spec.name}: {res['dnf_reason']}, reached "
                     f"{res['progress_mm']:.0f}/{self.layout.length:.0f} mm")
        self.write_results()

    def result(self) -> dict:
        lay = self.layout
        run_dir = self.session.logger.run_dir if self.session.logger is not None else None
        return {
            "course": self.spec.name,
            "lap": self.lap,
            "status": self.status,
            "controller_mode": self.mode,
            "steering": self.steer,
            "time_s": round(self.race_time, 3),
            "penalty_s": round(self.penalty_s, 3),
            "total_time_s": round(self.race_time + self.penalty_s, 3),
            "splits": list(self.splits),
            "n_falls": len(self.falls),
            "fall_times_s": list(self.falls),
            "n_respawns": len(self.respawns),
            "respawns": list(self.respawns),
            "n_external_resets": self.n_external_resets,
            "gates_total": len(lay.gates),
            "gates_passed": sum(g["passed"] for g in self.gate_results),
            "gates_missed": sum(not g["passed"] for g in self.gate_results),
            "gates": list(self.gate_results),
            "n_whip_triggers": self.n_whip,
            "n_loom_stimuli": self.n_loom,
            "progress_mm": round(max(self.max_x - lay.start_x, 0.0), 2),
            "course_length_mm": round(lay.length, 2),
            "checkpoints_reached": self.next_cp,
            "checkpoints_total": len(lay.checkpoints),
            "seed": self.spec.seed,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": str(run_dir) if run_dir else None,
        }

    def results_path(self) -> Path | None:
        d = self.opt.results_dir
        if d is None and self.session.logger is not None:
            d = self.session.logger.run_dir
        return None if d is None else Path(d) / "course_results.json"

    def summary(self) -> dict:
        return {"course": self.spec.name, "path": self.spec.path, "mode": self.mode,
                "after_finish": self.after_finish, "laps": self.laps,
                "current": None if self._finalized else self.result(),
                "layout": {"length_mm": round(self.layout.length, 2),
                           "sections": [asdict(s) for s in self.layout.sections],
                           **self.layout.counts()}}

    def write_results(self) -> Path | None:
        p = self.results_path()
        if p is None:
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.summary(), indent=2, default=str))
        tmp.replace(p)
        return p

    def finalize(self, quit_reason: str = "") -> dict:
        """End of session: an unfinished lap is recorded as aborted (not ranked)."""
        if self._finalized:
            return self.summary()
        if self.status in (READY, RUNNING) and (self.status == RUNNING or not self.laps):
            self.t_end = self.now() if self.t_start is not None else None
            self.status = ABORTED
            res = self.result()
            res["quit_reason"] = quit_reason
            self.laps.append(res)
        self._finalized = True
        self.write_results()
        return self.summary()

    def _log(self, event: str, **details) -> None:
        self.session.log_event(event, course=self.spec.name, **details)

    # -------------------------------------------------------------------- HUD
    def hud_lines(self) -> list[str]:
        lay = self.layout
        x = float(self.sim.data.xpos[self.sim.thorax_body_id, 0])
        status = {READY: "READY - walk to the start line", RUNNING: "RUNNING",
                  FINISHED: "FINISHED", DNF: "DNF", ABORTED: "ABORTED"}[self.status]
        lines = [f"COURSE {self.spec.name}  lap {self.lap}  {status}  "
                 f"time {self.race_time:6.2f}s  pen +{self.penalty_s:.1f}s  [{self.mode}]"]
        sec = lay.section_at(x)
        where = f"{sec.name} ({sec.index + 1}/{len(lay.sections)})" if sec else "-"
        prog = min(max(x - lay.start_x, 0.0), lay.length)
        cp = f"CP {self.next_cp}/{len(lay.checkpoints)}"
        if self.splits:
            cp += f" (last {self.splits[-1]['time_s']:.2f}s)"
        lines.append(f"{where:<22} {prog:5.1f}/{lay.length:.0f} mm  {cp}")
        tail = f"falls {len(self.falls)}  respawns {len(self.respawns)}"
        if lay.gates:
            tail += (f"  gates {sum(g['passed'] for g in self.gate_results)}/{len(lay.gates)}"
                     f" (missed {sum(not g['passed'] for g in self.gate_results)})")
        if not self.can_steer and lay.gates:
            tail += "  (no steering: cpg)"
        lines.append(tail)
        if self.status == RUNNING and self.fallen_since is not None:
            down = self.now() - self.fallen_since
            lines.append(f"DOWN {down:4.1f}s -> respawn at checkpoint {self.next_cp} in "
                         f"{max(self.respawn_after_s - down, 0.0):.1f}s")
        if self.status == FINISHED:
            tot = self.race_time + self.penalty_s
            lines.append(f"FINISH {tot:.2f}s" + (f"  leaderboard #{self.rank}" if self.rank else "")
                         + ("  (next lap soon)" if self.after_finish == "loop" else ""))
        return lines
