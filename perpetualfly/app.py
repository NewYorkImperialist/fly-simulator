"""Main loop / CLI: physics in chunks of N steps, optional rendering / window / video.

``Session`` wires everything around one ``Simulation``: procedural terrain, the hits
(the physical whip and the external-force shove; keys + auto perturber, H toggles
the mode), fall detection, run metrics and the run logger (runs/<timestamp>/...). ``run()`` drives a session headless, recording
or with a live OpenCV window (physics in a worker thread by default).

Threading rule (see perpetualfly/physics_thread.py): in the threaded window mode
every hook, ``after_physics`` and the auto-reset run in the physics thread under
the runner's lock; every main-thread access to the sim (key handlers, spawning,
hits, reset, reading state for the HUD / camera) goes through ``runner.locked()``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from perpetualfly.actions import ActionEvent, ActionManager, make_action
from perpetualfly.actions.registry import available_actions
from perpetualfly.brain_link import BRAIN_KEYS, BrainLink, missing_requirements
from perpetualfly.config import AppConfig
from perpetualfly.interaction import HitEvent, install_perturbation
from perpetualfly.interaction.perturb_controls import HIT_MODES, format_event, format_whip_event
from perpetualfly.interaction.whip import Whip, WhipHitEvent
from perpetualfly.metrics import FallDetector, FallEvent, FallState, RunLogger, RunMetrics
from perpetualfly.simulation import Simulation, SimulationInstabilityError, WorldExtension
from perpetualfly.terrain import DIFFICULTY_PRESETS, ProceduralTerrain, ProceduralTerrainConfig

# (group, [(keys, description), ...]): terminal help (?) and the on-screen overlay
KEY_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("hits", [
        ("SPACE", "whip: crack from a random side | shove: random direction (+ upward)"),
        ("LEFT / RIGHT", "whip: crack from the fly's left / right | shove: to its left / right"),
        ("UP / DOWN", "whip: crack from the front / rear | shove: forward / backward"),
        ("U", "whip: overhead crack | shove: straight up"),
        ("1 2 3 4", "hit strength: gentle / medium / hard / absurd"),
        ("H", "hit mode: whip <-> shove"),
        ("A", "toggle automatic random hits (current hit mode; wait while the fly is down)"),
    ]),
    ("obstacles / terrain", [
        ("R B S G D", "spawn a rock / bump / slope / gap / dip ahead"),
        ("F", "flatten the next terrain chunk"),
        ("[ / ]", "terrain difficulty down / up (flat easy normal hard chaos), new chunks"),
    ]),
    ("brain (--brain)", [
        ("O", "looming (LC4 -> giant fiber + MDN; --brain-steer: stop; --brain-actions: jump)"),
        ("T", "sugar taste (sugar GRNs -> MN9; --brain-actions: proboscis extension)"),
        ("K", "bitter taste (bitter GRNs; display only)"),
    ]),
    ("actions", [
        ("J", "jump (escape: crouch, mid-leg push, flight, landing)"),
        ("Z", "freeze: stop and hold a stance for 1.5 s"),
        ("Y", "groom: front legs replay a recorded grooming bout (3 s)"),
        ("E", "back away: walk backward for 1 s"),
        (", / .", "turn in place left / right (1 s)"),
        ("W", "wing raise (1.5 s; needs --full-body)"),
        ("N", "proboscis extension (1.5 s; needs --full-body)"),
    ]),
    ("view / run", [
        ("C", "camera: follow / side / top"),
        ("P", "pause / resume"),
        ("X", "reset the fly (explicit reset, counted in the metrics)"),
        ("I", "screenshot: fly frame (+ brain frame) as PNG into the run dir"),
        ("M", "start / stop MP4 recording of the fly view (run dir)"),
        ("?", "show / hide this help"),
        ("Q / ESC", "quit (closing the window or Ctrl-C also quits)"),
    ]),
]
KEY_TABLE: list[tuple[str, str]] = [row for _, rows in KEY_GROUPS for row in rows]
KEY_HELP = ("SPACE/arrows/U hit | 1-4 strength | H whip/shove | A auto | R B S G D spawn | "
            "F flatten | [ ] terrain | J Z Y E , . W N actions | P X C | I shot | M rec | "
            "O T K brain | ? help | Q quit")
# app key -> (action name, parameters); see perpetualfly/actions (docs/ACTIONS.md)
ACTION_KEY_MAP: dict[str, tuple[str, dict]] = {
    "j": ("jump", {}),
    "z": ("freeze", {"duration": 1.5}),
    "y": ("groom", {"duration": 3.0}),
    "e": ("back_away", {"duration": 1.0}),
    ",": ("turn_left", {"duration": 1.0}),
    ".": ("turn_right", {"duration": 1.0}),
    "w": ("wings", {"duration": 1.5}),
    "n": ("proboscis", {"duration": 1.5}),
}
# --script-keys names for keys that clash with its syntax
SCRIPT_KEY_ALIASES = {"comma": ",", "period": ".", "dot": ".", "colon": ":"}
UNAVAILABLE_MARK = "~"  # help rows starting with this are drawn greyed out


def help_groups(available: set[str] | None = None) -> list[tuple[str, list[tuple[str, str]]]]:
    """KEY_GROUPS with the action rows the current body can't do marked greyed out
    (description prefixed with UNAVAILABLE_MARK; the overlay draws them grey)."""
    if available is None:
        return KEY_GROUPS
    need = {"W": "wings", "N": "proboscis"}
    out = []
    for group, rows in KEY_GROUPS:
        if group == "actions":
            rows = [(k, (UNAVAILABLE_MARK + d) if need.get(k) and need[k] not in available else d)
                    for k, d in rows]
        out.append((group, rows))
    return out
SPAWN_KEYS = {"r": "rock", "b": "bump", "s": "slope", "g": "gap", "d": "dip"}
TERRAIN_CHOICES = tuple(DIFFICULTY_PRESETS)  # flat, easy, normal, hard, chaos


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def key_help_text(available: set[str] | None = None) -> str:
    w = max(len(k) for k, _ in KEY_TABLE)
    out = ["Keys (window focused):"]
    for group, rows in help_groups(available):
        out.append(f" {group}:")
        out.extend(f"  {k:<{w}}  " + (f"(unavailable) {d[1:]}" if d.startswith(UNAVAILABLE_MARK)
                                        else d) for k, d in rows)
    return "\n".join(out)


@dataclass
class RunResult:
    sim_time: float  # monotonic run time (s), continues across resets
    path_length: float  # mm, summed over resets
    forward_displacement: float
    average_speed: float
    final_thorax_pos: tuple[float, float, float]
    quit_reason: str
    n_falls: int = 0
    n_recoveries: int = 0
    n_hits: int = 0
    n_resets: int = 0
    final_state: str = "upright"
    run_dir: str | None = None
    walked_distance: float = 0.0  # mm, walking only (no flights / falls), summed over resets
    n_auto_hits_skipped: int = 0  # auto hits skipped while the fly was down


# ---------------------------------------------------------------------------
# Session: sim + terrain + whip + falls + metrics + logging
# ---------------------------------------------------------------------------


class Session:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        log: bool | None = None,
        world_extensions: list[WorldExtension] | tuple[WorldExtension, ...] = (),
        say: Callable[[str], None] | None = None,
        brain: BrainLink | None = None,
    ) -> None:
        """``brain``: an already started ``BrainLink`` (``run()`` creates it for
        --brain); the session wires hits / falls / resets / steering to it."""
        self.cfg = cfg
        self.say = say or (lambda msg: print(msg, flush=True))
        self.brain = brain
        tc = cfg.terrain
        self.terrain = ProceduralTerrain(ProceduralTerrainConfig(
            difficulty=tc.difficulty, seed=tc.seed, weights=tc.weights,
            ground_half_size=tc.ground_half_size, checker_size_mm=tc.checker_size_mm,
        ))
        # The physical whip is a world extension (bodies must exist before add_fly).
        self.whip: Whip | None = Whip(cfg.whip) if cfg.whip.enabled else None
        exts = ([self.whip.extension] if self.whip else []) + list(world_extensions)
        self.sim = sim = Simulation(cfg, world_factory=self.terrain.build_world,
                                    world_extensions=exts)
        self.terrain.attach(sim)
        if self.whip is not None:
            self.whip.attach(sim)
        # Action library (J Z Y E , . W N; brain triggers with --brain-actions). Must
        # exist before brain.attach() below.
        self.actions = ActionManager(sim)
        self.available_actions = set(available_actions(sim))
        if cfg.session.hit_mode not in HIT_MODES:
            raise ValueError(f"session.hit_mode must be one of {HIT_MODES}")
        # Always install the auto perturber (so A can switch it on); it starts in
        # the state given by cfg.auto_perturb.enabled and uses the current hit mode.
        self.controls = install_perturbation(sim, cfg.perturbation, cfg.auto_perturb,
                                             whip=self.whip, mode=cfg.session.hit_mode)
        self.perturbation = self.controls.perturbation
        self.auto = self.controls.auto
        self.detector = FallDetector(sim, cfg.falls, ground_height_fn=self.ground_height)
        # A jump is a deliberate flight: the fall detector is paused while it runs
        # (hold timers cleared when it resumes, so a failed landing is still caught).
        hooks = sim.post_step_hooks
        hooks[hooks.index(self.detector)] = self._detector_hook
        self._falls_paused = False
        self.metrics = RunMetrics(cfg.stats.speed_window_s).attach(sim, self.detector)
        # RunMetrics must bank the run time *before* the detector emits its "reset"
        # event (whose events.csv timestamp is metrics.run_time_at(sim.time)), so
        # move its reset hook in front of the detector's.
        sim.reset_hooks.remove(self.metrics._on_sim_reset)
        sim.reset_hooks.insert(sim.reset_hooks.index(self.detector.on_reset),
                               self.metrics._on_sim_reset)
        log = cfg.logging.enabled if log is None else log
        self.logger: RunLogger | None = None
        if log:
            from perpetualfly.brain_link import METRIC_COLUMNS as BRAIN_COLUMNS

            self.logger = RunLogger(
                cfg.logging, config=self.full_config(),
                terrain_type_fn=self.terrain.terrain_type_at,
                ground_height_fn=self.ground_height,
                extra_metric_columns=BRAIN_COLUMNS if brain is not None else (),
                extra_metric_fn=brain.metric_row if brain is not None else None,
            ).attach(sim, self.detector, self.metrics)
            # Whip hits go to events.csv as "whip" rows (shove hits stay "hit").
            hl = self.metrics.hit_listeners
            hl[hl.index(self.logger.log_hit)] = self._log_hit_record
        self.perturbation.listeners.append(self._on_hit)
        if self.whip is not None:
            self.whip.listeners.append(self._on_whip)
        self.detector.add_listener(self._on_fall_event)
        self.actions.listeners.append(self._on_action_event)
        if brain is not None:
            brain.attach(self)
        self.down_since: float | None = None  # sim time of the last fall (until recovered/reset)
        # sim time since which the fly is back on its feet after a fall (None while
        # down; -inf after start / reset, where no resume delay applies)
        self.up_since: float | None = float("-inf")
        self.difficulty_start = self.terrain.cfg.difficulty
        self.media: list[str] = []  # screenshots / recordings written (summary.json)
        if cfg.session.auto_perturb_pause_when_down:
            self.auto.gate = self.auto_hits_allowed
            self.auto.skip_listeners.append(self._on_auto_skip)
        self._hint_shown = False
        self.n_auto_resets = 0
        self.n_manual_resets = 0

    # ------------------------------------------------------------- queries
    def ground_height(self, x: float, y: float) -> float:
        return self.terrain.ground_height_at(x, y)

    def terrain_here(self) -> str:
        return self.terrain.terrain_type_at(float(self.sim.data.xpos[self.sim.thorax_body_id, 0]))

    def height_above_ground(self) -> float:
        x, y, z = self.sim.thorax_position()
        return float(z - self.ground_height(x, y))

    def run_time(self) -> float:
        return self.metrics.run_time_at(self.sim.time)

    def auto_hits_allowed(self) -> bool:
        """Auto-perturber gate: not FALLEN / RECOVERING and on its feet for
        ``session.auto_perturb_resume_after_s``."""
        if self.detector.state in (FallState.FALLEN, FallState.RECOVERING) or self.up_since is None:
            return False
        if self.actions.active_name == "jump":
            return False
        return self.sim.time - self.up_since >= self.cfg.session.auto_perturb_resume_after_s

    # actions during which the fly deliberately stands still: the detector's
    # "no progress" window is kept empty so standing is not read as being stuck
    STATIONARY_ACTIONS = ("freeze", "groom")

    def _detector_hook(self, sim: Simulation) -> None:
        name = self.actions.active_name
        if name == "jump":
            self._falls_paused = True
            return
        if self._falls_paused:
            self._falls_paused = False
            self.detector._since.clear()
            self.detector._progress.clear()
        if name in self.STATIONARY_ACTIONS:
            self.detector._progress.clear()
        self.detector(sim)

    def trigger_action(self, name: str, source: str = "key", **params) -> str:
        """Start an action by registry name (key handler / scripts); returns the
        terminal message. Threaded window mode: call inside runner.locked()."""
        if name not in self.available_actions:
            return f"[action] {name}: not available with this body (run with --full-body)"
        defaults = dict(next((p for n, p in ACTION_KEY_MAP.values() if n == name), {}))
        defaults.update(params)
        self.actions.trigger(make_action(name, **defaults), source=source)
        return f"[action] {name} ({source})"

    def _on_action_event(self, ev: ActionEvent) -> None:
        info = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ev.info.items()
                if k != "feet_x_at_stroke_mm"}
        self.log_event(f"action_{ev.kind}", action=ev.name, **info)
        if ev.kind != "start":
            keys = ("apex_dz_mm", "airtime_s", "distance_mm", "landed_upright", "drift_mm",
                    "forward_mm", "turn_deg")
            shown = " ".join(f"{k}={info[k]}" for k in keys if k in info)
            self.say(f"[action] {ev.name} {ev.kind} t={ev.time:.2f}s {shown}".rstrip())

    def _on_auto_skip(self, t: float) -> None:
        n = self.auto.n_skipped
        state = self.detector.state.value
        why = (state if state in ("FALLEN", "RECOVERING") else
               "jumping" if self.actions.active_name == "jump" else "just recovered")
        self.say(f"[auto-perturb] t={t:.2f}s hit skipped: fly {why} ({n} skipped so far)")
        self.log_event("auto_hit_skipped", state=state, n_skipped=n)

    def down_for(self) -> float | None:
        return None if self.down_since is None else self.sim.time - self.down_since

    def full_config(self) -> dict:
        return {
            "app": self.cfg.to_dict(),
            "procedural_terrain": asdict(self.terrain.cfg),
            "runtime": {
                "argv": sys.argv,
                "timestep": self.sim.timestep,
                "fly_mass_g": self.sim.fly_mass,
                "body_weight_uN": self.perturbation.body_weight_uN,
                "n_pool_geoms": self.terrain.n_pool_geoms,
                "world_extensions": [getattr(e, "__name__", repr(e))
                                     for e in self.sim.world_extensions],
            },
            "brain": self.brain.config_dict() if self.brain is not None else None,
        }

    # ------------------------------------------------------------- events
    def _on_hit(self, ev: HitEvent) -> None:
        # HitEvent uses sim_time/magnitude_uN/duration_s; RunMetrics reads
        # time/magnitude/duration. The full event goes into "extra" (-> events.csv).
        self.metrics.record_hit(ev, time=ev.sim_time, magnitude=ev.magnitude_uN,
                                duration=ev.duration_s, direction=ev.direction_name,
                                body=ev.body)
        if ev.source != "key":  # key hits are printed by the key handler
            names = [lv.name for lv in self.perturbation.cfg.levels]
            self.say(format_event(ev, names) + f"  terrain={self.terrain_here()}")

    def _on_whip(self, ev: WhipHitEvent) -> None:
        """A crack finished (runs in the physics thread). Hits are counted like
        shoves; the record's magnitude is the *mean* contact force (impulse /
        contact duration) so that RunMetrics' impulse = magnitude x duration is the
        measured impulse; the peak force is in the details (magnitude_uN)."""
        names = [lv.name for lv in self.perturbation.cfg.levels]
        if ev.hit:
            self.metrics.record_hit(ev, time=ev.sim_time, magnitude=ev.mean_force_uN,
                                    duration=ev.duration_s, direction=ev.direction_name,
                                    body=ev.body)
        elif self.logger is not None:
            self.logger.log_event("whip_miss", ev.sim_time, force_direction=ev.direction_name,
                                  details=ev.to_dict())
        self.say(format_whip_event(ev, names) + f"  terrain={self.terrain_here()}")

    def _log_hit_record(self, rec) -> None:
        """metrics.hit_listeners -> events.csv: event_type "whip" for whip hits (with
        the measured impulse etc. in details), "hit" for shoves."""
        if rec.extra.get("kind") != "whip":
            self.logger.log_hit(rec)
            return
        details = {"duration": rec.duration, "body": rec.body}
        details.update({k: v for k, v in rec.extra.items()
                        if k not in ("sim_time", "direction_name", "body", "duration_s")})
        self.logger.log_event("whip", rec.time, force_direction=rec.direction,
                              force_magnitude=rec.magnitude, details=details)

    @property
    def hit_mode(self) -> str:
        return self.controls.mode

    def _on_fall_event(self, ev: FallEvent) -> None:
        if ev.kind == "fall":
            self.down_since = ev.time
            self.up_since = None
            self._hint_shown = False
        elif ev.kind in ("recovered", "reset"):
            self.down_since = None
            self.up_since = ev.time if ev.kind == "recovered" else float("-inf")
        if ev.kind in ("fall", "recovered", "relapse"):
            extra = f" after {ev.recovery_time:.2f}s" if ev.recovery_time is not None else ""
            self.say(f"[{ev.kind}] t={ev.time:.2f}s {ev.reason}{extra}  "
                     f"terrain={self.terrain.terrain_type_at(ev.position[0])}")

    def log_event(self, event_type: str, **details) -> None:
        if self.logger is not None:
            self.logger.log_event(event_type, details=details)

    # ------------------------------------------------------------- actions
    # Threaded window mode: call these only inside runner.locked().
    def reset(self, source: str = "manual") -> None:
        """Explicit reset (never hidden): logged as '<source>_reset', counted by
        RunMetrics.n_resets (via the sim reset hook)."""
        down = self.down_for()
        self.log_event(f"{source}_reset", down_for_s=down, state=self.detector.state.value)
        if source == "auto":
            self.n_auto_resets += 1
        else:
            self.n_manual_resets += 1
        self.sim.reset()
        self.down_since = None

    def spawn(self, kind: str) -> str:
        r = self.terrain.spawn_ahead(kind, self.cfg.session.spawn_distance)
        if r is None:
            return f"[spawn] {kind}: no free geoms / no safe spot, nothing spawned"
        self.log_event("spawn", kind=kind, x_start=r.x_start, x_end=r.x_end, distance=r.distance)
        return (f"[spawn] {kind} {r.distance:.1f} mm ahead "
                f"(x {r.x_start:.1f}..{r.x_end:.1f} mm)")

    def flatten(self) -> str:
        idx = self.terrain.flatten_next_chunk()
        x0, x1 = self.terrain.generator.chunk_bounds(idx)
        self.log_event("flatten", chunk=idx, x0=x0, x1=x1)
        return f"[flatten] chunk {idx} (x {x0:.0f}..{x1:.0f} mm) is now flat"

    def step_difficulty(self, delta: int) -> str:
        """[ / ]: terrain difficulty one preset down / up (flat..chaos). Chunks from
        two ahead of the fly on are regenerated; nothing changes under its feet."""
        old = self.terrain.cfg.difficulty
        i = TERRAIN_CHOICES.index(old) if old in TERRAIN_CHOICES else 2
        j = min(max(i + delta, 0), len(TERRAIN_CHOICES) - 1)
        if j == i:
            return f"[terrain] difficulty already {old} ({'hardest' if delta > 0 else 'easiest'})"
        new = TERRAIN_CHOICES[j]
        reloaded = self.terrain.set_difficulty(new)
        self.log_event("terrain_difficulty", difficulty=new, previous=old,
                       regenerated_chunks=reloaded)
        ahead = 2 * self.terrain.cfg.chunk_length
        return (f"[terrain] difficulty {old} -> {new} (new terrain from ~{ahead * 0.5:.0f}-"
                f"{ahead:.0f} mm ahead; chunks {reloaded} regenerated)")

    def handle_whip_key(self, key: str) -> str | None:
        """Hits / strength / auto toggle; None if the key isn't a whip key."""
        msg = self.controls.handle(key)
        if msg is None:
            return None
        if msg.startswith("[strength]"):
            self.log_event("strength", level=self.perturbation.level,
                           name=self.perturbation.level_name)
        elif msg.startswith("[auto-perturb]"):
            self.log_event("auto_perturb", enabled=self.auto.enabled)
        elif msg.startswith("[hit-mode]"):
            self.log_event("hit_mode", mode=self.controls.mode)
        elif msg.startswith("[hit"):
            msg += f"  terrain={self.terrain_here()}"
        return msg

    def after_physics(self) -> None:
        """Once per physics chunk: brain clock / drive, fall-reset hint and optional
        auto reset."""
        if self.brain is not None:
            self.brain.update()
        down = self.down_for()
        if down is None:
            return
        sc = self.cfg.session
        auto_s = sc.auto_reset_after_s
        if sc.fall_hint_after_s and not self._hint_shown and down >= sc.fall_hint_after_s:
            self._hint_shown = True
            tail = f" (auto reset after {auto_s:g}s)" if auto_s is not None else ""
            self.say(f"[hint] the fly has been down for {down:.1f}s - press X to reset{tail}")
        if auto_s is not None and down >= auto_s:
            self.say(f"[auto-reset] down for {down:.1f}s -> explicit reset "
                     f"(#{self.n_auto_resets + 1})")
            self.reset("auto")

    def _action_counts(self) -> dict:
        out: dict[str, int] = {}
        for ev in self.actions.history:
            if ev.kind == "start":
                out[ev.name] = out.get(ev.name, 0) + 1
        return out

    def summary_extra(self, quit_reason: str) -> dict:
        return {
            "quit_reason": quit_reason,
            "terrain_difficulty": self.terrain.cfg.difficulty,  # at the end ([ / ] change it)
            "terrain_difficulty_start": self.difficulty_start,
            "terrain_seed": self.terrain.cfg.seed,
            "n_manual_resets": self.n_manual_resets,
            "n_auto_resets": self.n_auto_resets,
            "n_spawns": self.terrain.spawn_count,
            "n_chunks_recycled": self.terrain.recycle_count,
            "strength_level": self.perturbation.level,
            "hit_mode": self.controls.mode,
            "n_whip_cracks": self.whip.n_cracks if self.whip else 0,
            "n_whip_hits": sum(e.hit for e in self.whip.events) if self.whip else 0,
            "n_whip_misses": sum(not e.hit for e in self.whip.events) if self.whip else 0,
            "whip_stray_contact_steps": self.whip.stray_contact_steps if self.whip else 0,
            "auto_perturb_enabled": bool(self.auto and self.auto.enabled),
            "n_auto_hits_skipped": self.auto.n_skipped if self.auto else 0,
            "media": list(self.media),
            "full_body": "wings" in self.available_actions,
            "actions": self._action_counts(),
            "brain": self.brain.summary() if self.brain is not None else None,
        }

    def close(self, quit_reason: str) -> None:
        if self.logger is not None:
            self.logger.summary_extra.update(self.summary_extra(quit_reason))
            self.logger.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PerpetualFly: NeuroMechFly jogging forever.")
    p.add_argument("--headless", action="store_true", help="no window, terminal stats only")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="stop after this many *simulated* seconds (run time, across resets)")
    p.add_argument("--record", type=Path, default=None,
                   help="also write an mp4 of the offscreen-rendered camera (works headless)")
    p.add_argument("--config", type=Path, default=None, help="JSON config (see AppConfig)")
    g = p.add_argument_group("terrain")
    g.add_argument("--terrain", choices=TERRAIN_CHOICES, default=None,
                   help="terrain difficulty (default: normal, or the --config value)")
    g.add_argument("--terrain-seed", type=int, default=None)
    g.add_argument("--spawn-distance", type=float, default=None,
                   help="mm ahead of the thorax for key-spawned obstacles")
    g = p.add_argument_group("perturbation")
    g.add_argument("--strength", type=int, choices=[1, 2, 3, 4], default=None,
                   help="initial hit strength level")
    g.add_argument("--hit-mode", choices=list(HIT_MODES), default=None,
                   help="whip (physical whip, default) or shove (thorax force); H toggles")
    g.add_argument("--auto-perturb", action="store_true", help="automatic random hits on")
    g.add_argument("--auto-min", type=float, default=None, help="min sim seconds between hits")
    g.add_argument("--auto-max", type=float, default=None, help="max sim seconds between hits")
    g.add_argument("--auto-levels", type=str, default=None,
                   help="strength levels to draw from, e.g. '1,2,3' (equal weights) or "
                        "'1:0.5,2:0.3,3:0.2'")
    g.add_argument("--auto-seed", type=int, default=None)
    g = p.add_argument_group("falls / resets")
    g.add_argument("--fall-hint-after", type=float, default=None,
                   help="print 'press X to reset' after the fly is down this long (s; 0=off)")
    g.add_argument("--auto-reset-after", type=float, default=None,
                   help="explicitly reset after the fly is down this long (s; default off)")
    g = p.add_argument_group("logging")
    g.add_argument("--no-log", action="store_true", help="don't write runs/<timestamp>/")
    g.add_argument("--runs-dir", type=Path, default=None, help="parent dir of run folders")
    g.add_argument("--log-hz", type=float, default=None, help="metrics.csv rows per sim second")
    g = p.add_argument_group("sim / display")
    g.add_argument("--full-body", action=argparse.BooleanOptionalAction, default=None,
                   help="extra wing + proboscis joints so W / N work (default ON in the CLI; "
                        "walking is unchanged; --no-full-body = FlyGym's legs-only model)")
    g.add_argument("--controller", choices=["hybrid", "cpg"], default=None)
    g.add_argument("--seed", type=int, default=None, help="controller (CPG) seed")
    g.add_argument("--camera", choices=["follow", "side", "top"], default=None)
    g.add_argument("--no-reflections", action="store_true",
                   help="no floor reflections (faster rendering: ~11 -> ~6 ms per frame)")
    g.add_argument("--render-every", type=int, default=None,
                   help="physics steps between rendered frames")
    g.add_argument("--no-thread", action="store_true",
                   help="live window: step physics on the main thread (slower; debugging)")
    g.add_argument("--print-interval", type=float, default=None,
                   help="simulated seconds between terminal stat lines")
    g.add_argument("--script-keys", type=str, default=None,
                   help="press keys at given run times (sim s), e.g. '2:left,5:o,8:3,8.1:space'"
                        " (works headless too; key names as in the key help, lowercase)")
    g = p.add_argument_group("connectome brain (docs/BRAIN.md; needs .[brain] + data/brain)")
    g.add_argument("--brain", action="store_true",
                   help="run the FlyWire whole-brain model next to the fly + the brain window")
    g.add_argument("--brain-headless", action="store_true",
                   help="run the brain model without its window (HUD / terminal / logs only)")
    g.add_argument("--no-brain-window", action="store_true",
                   help="with --brain: don't open the brain window")
    g.add_argument("--brain-steer", action="store_true",
                   help="the brain's descending neurons modulate the walking controller "
                        "(implies --brain; off by default: the brain only watches)")
    g.add_argument("--brain-backup", action="store_true",
                   help="lower the MDN backward-walking reference (40 -> 20 Hz) so looming "
                        "(O) makes the fly walk backward (implies --brain-steer)")
    g.add_argument("--brain-actions", action="store_true",
                   help="descending neurons trigger body actions: giant fiber > 60 Hz -> jump, "
                        "MN9 > 30 Hz -> proboscis (full body), DNg12 > 20 Hz -> groom "
                        "(implies --brain-steer)")
    return p


def _parse_levels(text: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
    levels, weights = [], []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        lv, _, w = part.partition(":")
        levels.append(int(lv))
        weights.append(float(w) if w else 1.0)
    if not levels or any(not 1 <= lv <= 4 for lv in levels):
        raise argparse.ArgumentTypeError(f"bad --auto-levels {text!r}")
    return tuple(levels), tuple(weights)


def parse_script_keys(text: str | None) -> list[tuple[float, str]]:
    """'2:left,5:o' -> [(2.0, 'left'), (5.0, 'o')] sorted by time."""
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        t, sep, key = part.partition(":")
        if not sep or not key.strip():
            raise argparse.ArgumentTypeError(f"bad --script-keys entry {part!r} (want T:KEY)")
        key = key.strip().lower()
        key = SCRIPT_KEY_ALIASES.get(key, key)
        out.append((float(t), key))
    return sorted(out, key=lambda x: x[0])


def config_from_args(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig.load_json(args.config) if args.config else AppConfig()
    if args.controller:
        cfg.controller.kind = args.controller
    # full body: CLI default ON (library default off), unless a --config decides
    fb = getattr(args, "full_body", None)
    if fb is not None:
        cfg.fly.extra_joints = fb
    elif not args.config:
        cfg.fly.extra_joints = True
    if args.seed is not None:
        cfg.controller.seed = args.seed
    if args.camera:
        cfg.camera.mode = args.camera
    if args.render_every:
        cfg.render.render_every_steps = args.render_every
    if getattr(args, "no_thread", False):
        cfg.render.threaded_physics = False
    if getattr(args, "no_reflections", False):
        cfg.render.reflections = False
    if args.print_interval:
        cfg.stats.print_interval_s = args.print_interval
    # terrain: the CLI default is "normal" (the library default is "flat")
    if args.terrain:
        cfg.terrain.difficulty = args.terrain
    elif not args.config:
        cfg.terrain.difficulty = "normal"
    if args.terrain_seed is not None:
        cfg.terrain.seed = args.terrain_seed
    if args.spawn_distance is not None:
        cfg.session.spawn_distance = args.spawn_distance
    # perturbation
    if args.strength is not None:
        cfg.perturbation.default_level = args.strength
    if getattr(args, "hit_mode", None):
        cfg.session.hit_mode = args.hit_mode
    ap = cfg.auto_perturb
    if args.auto_perturb:
        ap.enabled = True
    if args.auto_min is not None:
        ap.min_interval_s = args.auto_min
    if args.auto_max is not None:
        ap.max_interval_s = args.auto_max
    if ap.max_interval_s < ap.min_interval_s:
        ap.max_interval_s = ap.min_interval_s
    if args.auto_levels:
        ap.levels, ap.level_weights = _parse_levels(args.auto_levels)
    if args.auto_seed is not None:
        ap.seed = args.auto_seed
    # falls
    if args.fall_hint_after is not None:
        cfg.session.fall_hint_after_s = args.fall_hint_after
    if args.auto_reset_after is not None:
        cfg.session.auto_reset_after_s = args.auto_reset_after if args.auto_reset_after > 0 else None
    # logging
    if args.no_log:
        cfg.logging.enabled = False
    if args.runs_dir is not None:
        cfg.logging.runs_dir = str(args.runs_dir)
    if args.log_hz is not None:
        cfg.logging.sample_hz = args.log_hz
    # brain
    b = cfg.brain
    brain_actions = getattr(args, "brain_actions", False)
    if args.brain or args.brain_headless or args.brain_steer or args.brain_backup or brain_actions:
        b.enabled = True
    if args.brain_headless or args.no_brain_window:
        b.window = False
    if args.brain_steer or args.brain_backup or brain_actions:
        b.steer = True
    if brain_actions:
        b.actions = True
    if args.brain_backup:
        b.backup = True
    return cfg


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


@dataclass
class _LoopState:
    next_print: float
    last_print_wall: float
    last_print_rt: float
    paused: bool = False
    done: bool = False
    threaded: bool = False
    quit_reason: str = "max-seconds"
    hud: list[str] | None = None
    show_help: bool = False  # ? toggles the on-screen key help
    # I pressed: (run time, copy of the brain's recent states) taken under the lock;
    # the main loop saves the PNGs outside it
    shot: tuple | None = None
    rec_toggle: bool = False  # M pressed (handled by the main loop, outside the lock)
    frame_rt: float = 0.0  # threaded: run time of the frame being shown (read under the lock)


def run(
    cfg: AppConfig,
    *,
    headless: bool = False,
    max_seconds: float | None = None,
    record: Path | None = None,
    log: bool | None = None,
    world_extensions: list[WorldExtension] | tuple[WorldExtension, ...] = (),
    script_keys: list[tuple[float, str]] | None = None,
) -> RunResult:
    """Run until quit (or ``max_seconds`` of simulated run time). ``log=None``
    follows ``cfg.logging.enabled``. ``cfg.brain.enabled`` starts the connectome
    brain (+ its window unless headless / cfg.brain.window is off)."""
    brain: BrainLink | None = None
    if cfg.brain.enabled:
        # Spawned now so the connectome loads while MuJoCo builds the world.
        brain = BrainLink(cfg.brain, headless=headless)
    try:
        session = Session(cfg, log=log, world_extensions=world_extensions, brain=brain)
    except BaseException:
        if brain is not None:
            brain.close()
        raise
    sim, metrics, detector, terrain = session.sim, session.metrics, session.detector, session.terrain
    chunk = max(1, cfg.render.render_every_steps)
    need_frames = (not headless) or record is not None

    frame_renderer = viewer = writer = None
    from perpetualfly.media import MediaCapture

    media = MediaCapture(session.logger.run_dir if session.logger is not None else None,
                         cfg.logging.runs_dir, fps=cfg.render.record_fps)
    st = _LoopState(next_print=cfg.stats.print_interval_s, last_print_wall=time.perf_counter(),
                    last_print_rt=0.0)
    try:
        if brain is not None:
            t0 = time.perf_counter()
            info = brain.wait_ready()
            brain.start_window()
            print(f"brain: {brain.layout.model_name}, {info['n_neurons']:,} neurons, ready in "
                  f"{time.perf_counter() - t0:.1f}s (pid {info.get('pid')}), paced to fly "
                  f"time | steering {'ON' if cfg.brain.steer else 'off (view only)'} | "
                  f"window {'on' if brain.window is not None else 'off'}", flush=True)
        if need_frames:
            from perpetualfly.rendering import FrameRenderer

            frame_renderer = FrameRenderer(sim.model, cfg.render, cfg.camera)
        if not headless:
            from perpetualfly.interaction import LiveViewer

            viewer = LiveViewer(cfg.render.window_title, display_scale=cfg.render.display_scale,
                                frame_size=(cfg.render.width, cfg.render.height))
        if record is not None:
            import imageio.v2 as iio

            record.parent.mkdir(parents=True, exist_ok=True)
            fps = 1.0 / (chunk * sim.timestep)  # real-time playback
            writer = iio.get_writer(record, fps=fps, codec="libx264", quality=8,
                                    macro_block_size=8)
    except BaseException:
        session.close("startup error")
        if brain is not None:
            brain.close()
        raise

    ap = cfg.auto_perturb
    print(f"PerpetualFly | dt={sim.timestep:g}s | fly mass={sim.fly_mass * 1e3:.3f} mg | "
          f"controller={cfg.controller.kind} | terrain={terrain.cfg.difficulty} "
          f"(seed {terrain.cfg.seed}) | hit mode {session.hit_mode} | "
          f"strength L{session.perturbation.level} "
          f"{session.perturbation.level_name} | auto-perturb "
          f"{'on' if session.auto.enabled else 'off'} "
          f"({ap.min_interval_s:g}-{ap.max_interval_s:g}s, levels {list(ap.levels)}) | "
          f"body {'full (wings + proboscis)' if cfg.fly.extra_joints else 'legs-only'} | "
          f"{'headless' if headless else 'window'}"
          f"{f' (x{viewer.display_scale:.2f} display scale)' if viewer is not None else ''}",
          flush=True)
    if session.logger is not None:
        print(f"logging to {session.logger.run_dir}/", flush=True)
    if not headless:
        print(key_help_text(session.available_actions), flush=True)

    def status_line() -> str:
        wall = time.perf_counter()
        rt = session.run_time()
        dw = wall - st.last_print_wall
        rtf = (rt - st.last_print_rt) / dw if dw > 0 else 0.0
        st.last_print_wall, st.last_print_rt = wall, rt
        return (f"{metrics.summary_line()}  terrain={session.terrain_here():<10} "
                f"h={session.height_above_ground():4.2f}mm tilt={sim.tilt_deg():5.1f}deg "
                f"RTF={rtf:4.2f}")

    pending_keys = list(script_keys or [])

    def after_physics() -> bool:
        """Physics side, after each chunk of sim.step: hints / auto reset, terminal
        line, scripted keys. Returns True once max_seconds is reached or a scripted
        key quit. Threaded mode: runs in the worker thread, under the lock."""
        session.after_physics()
        rt = session.run_time()
        if pending_keys and pending_keys[0][0] <= rt:
            keys = []
            while pending_keys and pending_keys[0][0] <= rt:
                k = pending_keys.pop(0)[1]
                if k == "p" and headless:  # nothing would ever unpause a headless run
                    print("[script] 'p' ignored in headless mode", flush=True)
                    continue
                keys.append(k)
            print(f"[script] t={rt:.2f}s keys {keys}", flush=True)
            handle_keys(keys)
            if st.quit_reason != "max-seconds":
                st.done = True
                return True
        if rt >= st.next_print:
            while st.next_print <= rt:
                st.next_print += cfg.stats.print_interval_s
            print(status_line(), flush=True)
            if brain is not None:
                print("  " + brain.status_line(), flush=True)
        st.done = max_seconds is not None and rt >= max_seconds
        return st.done

    def camera_kw() -> dict:
        x, y, _ = sim.thorax_position()
        return {"ground_z": session.ground_height(x, y), "tilt_deg": sim.tilt_deg()}

    def hud_lines() -> list[str] | None:
        if not cfg.render.show_hud:
            return None
        m = metrics
        lines = [
            f"t {session.run_time():7.2f}s   dist {m.distance:7.1f} mm   "
            f"speed {m.current_speed:5.1f} mm/s (avg {m.average_speed:4.1f})",
            f"{detector.state.value.upper():<12} falls {m.n_falls}  rec {m.n_recoveries}  "
            f"resets {m.n_resets}  jog {m.current_jog_interval:5.1f}s",
            f"terrain {terrain.cfg.difficulty} ([ ] change): {session.terrain_here()}",
            session.controls.hud_line(),
            f"cam {frame_renderer.camera.mode}   {'PAUSED' if st.paused else ''}",
        ]
        if session.auto.enabled and not session.auto_hits_allowed():
            lines[3] += "  (waiting: fly down)"
        if m.distance - m.walked_distance > 1.0:
            lines[0] += f"  walked {m.walked_distance:.0f}"
        down = session.down_for()
        if down is not None and detector.state != FallState.UPRIGHT:
            lines.append(f"DOWN {down:4.1f}s - press X to reset")
        act = session.actions
        if act.busy:
            lines.append(f"ACTION {(act.active_name or 'back to walking').upper()}"
                         f"  [{act.phase() or 'starting'}]")
        if brain is not None:
            lines.extend(brain.hud_lines())
        lines.append("? = key help   I = screenshot   M = record")
        return lines

    def ensure_renderer():
        """Headless runs have no renderer until I / M need one (main thread)."""
        nonlocal frame_renderer
        if frame_renderer is None:
            from perpetualfly.rendering import FrameRenderer

            frame_renderer = FrameRenderer(sim.model, cfg.render, cfg.camera)
        return frame_renderer

    def process_media(frame=None) -> None:
        """Main thread, *outside* the physics lock: screenshots and the recording
        toggle requested by I / M, then the recording itself. ``frame`` = the
        current (clean) fly frame if the loop already has one."""
        if st.shot is not None:
            rt_shot, states = st.shot
            st.shot = None
            if frame is None:
                frame = render_now()
            hud = st.hud if (st.threaded or viewer is not None) else hud_lines()
            hud_img = None
            if hud:
                from perpetualfly.interaction.viewer import compose_frame

                hud_img = compose_frame(frame, hud)
            brain_img = None
            if brain is not None and states:
                from perpetualfly.brain_link import render_brain_frame

                t0 = time.perf_counter()
                brain_img = render_brain_frame(brain.layout, states, cfg.brain.window_s)
                brain_ms = (time.perf_counter() - t0) * 1e3
            paths = media.save_screenshot(frame, rt_shot, hud_img, brain_img)
            print(f"[screenshot] {', '.join(str(p) for p in paths)}"
                  + (f" (brain frame rendered in {brain_ms:.0f} ms)" if brain_img is not None
                     else " (no brain frame yet)" if brain is not None else ""), flush=True)
            session.media.extend(str(p) for p in paths)
            log_locked("screenshot", files=[str(p) for p in paths])
        if st.rec_toggle:
            st.rec_toggle = False
            if media.recording:
                print(media.stop_recording(), flush=True)
                log_locked("record_stop", path=str(media.rec_path), frames=media.rec_frames)
            else:
                path = media.start_recording(st.frame_rt if st.threaded else session.run_time())
                session.media.append(str(path))
                log_locked("record_start", path=str(path), fps=media.fps)
                print(f"[record] recording the fly view to {path} ({media.fps:g} fps of sim "
                      f"time; M stops)", flush=True)
        if media.recording:
            rt = st.frame_rt if st.threaded else session.run_time()
            if media.due(rt):
                media.add(frame if frame is not None else render_now(), rt)

    def log_locked(event_type: str, **details) -> None:
        """events.csv row from the main thread outside handle_keys (the logger reads
        the sim state, so threaded mode takes the physics lock)."""
        if st.threaded:
            with runner.locked():
                session.log_event(event_type, **details)
        else:
            session.log_event(event_type, **details)

    def render_now():
        if st.threaded:
            with runner.locked():
                ensure_renderer().update_scene(sim.data, sim.time, sim.thorax_position(),
                                               sim.heading(), **camera_kw())
            return frame_renderer.draw()
        return ensure_renderer().render(sim.data, sim.time, sim.thorax_position(),
                                        sim.heading(), **camera_kw())

    def handle_keys(keys: list[str]) -> None:
        """Display side; touches the sim, so threaded mode calls it under the lock."""
        for k in keys:
            msg = None
            k = SCRIPT_KEY_ALIASES.get(k, k)
            if k in ("q", "esc"):
                st.quit_reason = "quit key"
            elif k == "p":
                st.paused = not st.paused
                msg = "[paused]" if st.paused else "[resumed]"
                if brain is not None:
                    brain.on_pause(st.paused)
                    msg += " (the brain follows fly time and waits too)"
            elif k == "x":
                session.reset("manual")
                if frame_renderer is not None:
                    frame_renderer.camera.reset()
                msg = f"[reset] explicit reset #{metrics.n_resets}"
            elif k == "c":
                msg = (f"[camera] {frame_renderer.camera.cycle_mode()}"
                       if frame_renderer is not None else "[camera] no camera (headless)")
            elif k in SPAWN_KEYS:
                msg = session.spawn(SPAWN_KEYS[k])
            elif k == "f":
                msg = session.flatten()
            elif k in ("?", "/"):
                st.show_help = not st.show_help and viewer is not None
                msg = key_help_text(session.available_actions) + (
                    "\n(on-screen help shown; ? hides it)"
                                         if st.show_help else "")
            elif k in ("[", "]"):
                msg = session.step_difficulty(-1 if k == "[" else 1)
            elif k == "i":
                st.shot = (session.run_time(),
                           list(brain.recent) if brain is not None else [])
                msg = None  # the main loop prints the file names
            elif k == "m":
                st.rec_toggle = True
            elif k in ACTION_KEY_MAP:
                msg = session.trigger_action(ACTION_KEY_MAP[k][0], source="key")
            elif k in BRAIN_KEYS:
                msg = (brain.handle_key(k) if brain is not None else
                       f"[brain] {k.upper()} needs the brain: run with --brain")
            else:
                msg = session.handle_whip_key(k)
            if msg:
                print(msg, flush=True)

    def display_frame(frame) -> list[str]:
        viewer.show(frame, st.hud, help_groups=overlay_groups if st.show_help else None,
                    rec=media.rec_label())
        keys = viewer.poll_keys(30 if st.paused else 1)
        if not viewer.is_open():
            st.quit_reason = "window closed"
        return keys

    overlay_groups = help_groups(session.available_actions)
    frame_period = 1.0 / cfg.render.target_fps if cfg.render.target_fps > 0 else 0.0
    st.threaded = viewer is not None and writer is None and cfg.render.threaded_physics
    wall_start, sim_start = time.perf_counter(), session.run_time()
    try:
        if st.threaded:
            # ---- live window, physics in a worker thread (default) ----------
            from perpetualfly.physics_thread import PhysicsThread

            runner = PhysicsThread(sim, cfg.render.thread_chunk_steps, after_chunk=after_physics,
                                   max_realtime_factor=cfg.render.max_realtime_factor)
            runner.start()
            try:
                next_frame = time.perf_counter()
                while True:
                    runner.raise_if_failed()
                    with runner.locked():
                        frame_renderer.update_scene(sim.data, sim.time, sim.thorax_position(),
                                                    sim.heading(), **camera_kw())
                        st.hud = hud_lines()
                        st.frame_rt = session.run_time()
                    frame = frame_renderer.draw()
                    keys = display_frame(frame)
                    if keys:
                        with runner.locked():
                            handle_keys(keys)
                        runner.paused = st.paused
                        runner.reset_clock()
                    elif runner.paused != st.paused:  # paused by a scripted key
                        runner.paused = st.paused
                    process_media(frame)
                    if st.quit_reason != "max-seconds" or st.done:
                        break
                    if not runner.running:  # worker ended (max_seconds or error)
                        runner.raise_if_failed()
                        break
                    next_frame += frame_period
                    delay = next_frame - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_frame = time.perf_counter()
            finally:
                runner.stop()
        else:
            # ---- headless / recording / single-threaded window --------------
            last_shown = -1e9
            while True:
                if not st.paused:
                    sim.step(chunk)
                    after_physics()
                show = viewer is not None and time.perf_counter() - last_shown >= frame_period
                frame = None
                if writer is not None or show:
                    frame = frame_renderer.render(sim.data, sim.time, sim.thorax_position(),
                                                  sim.heading(), **camera_kw())
                    if writer is not None and not st.paused:
                        writer.append_data(frame)
                    if show:
                        last_shown = time.perf_counter()
                        st.hud = hud_lines()
                        handle_keys(display_frame(frame))
                        if st.quit_reason != "max-seconds":
                            break
                elif st.paused:
                    time.sleep(0.005)  # nothing to do until the next frame
                if st.shot is not None or st.rec_toggle or media.recording:
                    process_media(frame)
                # Never run faster than max_realtime_factor x real time (rarely
                # binding: this model + controller is usually slower than real time).
                wall = time.perf_counter() - wall_start
                ahead = (session.run_time() - sim_start) / cfg.render.max_realtime_factor - wall
                if ahead > 0 and not st.paused and not headless:
                    time.sleep(ahead)
                if st.done:
                    break
    except KeyboardInterrupt:
        st.quit_reason = "Ctrl-C"
    except BaseException as e:
        st.quit_reason = f"error: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
        raise
    finally:
        try:
            media.close(lambda msg: print(msg, flush=True))
        except Exception as e:
            print(f"warning: closing the recording failed: {e}", file=sys.stderr)
        for closer in (writer, viewer, frame_renderer):
            if closer is not None:
                try:
                    closer.close()
                except Exception as e:  # never mask the real error / skip the summary
                    print(f"warning: closing {type(closer).__name__} failed: {e}", file=sys.stderr)
        session.close(st.quit_reason)
        if brain is not None:
            brain.close()
    quit_reason = st.quit_reason

    pos = sim.thorax_position()
    m = metrics
    result = RunResult(
        sim_time=m.run_time_at(sim.time),
        path_length=m.distance,
        forward_displacement=m.forward_displacement,
        average_speed=m.average_speed,
        final_thorax_pos=tuple(float(v) for v in pos),
        quit_reason=quit_reason,
        n_falls=m.n_falls,
        n_recoveries=m.n_recoveries,
        n_hits=m.n_hits,
        n_resets=m.n_resets,
        final_state=detector.state.value,
        run_dir=str(session.logger.run_dir) if session.logger is not None else None,
        walked_distance=m.walked_distance,
        n_auto_hits_skipped=session.auto.n_skipped,
    )
    print(f"[done: {quit_reason}] {m.summary_line()}  "
          f"thorax=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) mm", flush=True)
    print(f"  resets={m.n_resets} (auto {session.n_auto_resets})  spawns={terrain.spawn_count}  "
          f"chunks recycled={terrain.recycle_count}  falls/km={_fmt(m.falls_per_km)}  "
          f"recovery%={_fmt(m.recovery_percentage)}", flush=True)
    print(f"  walked {m.walked_distance:.1f} mm in {m.walking_time:.1f} s "
          f"(walking speed {m.walking_speed:.1f} mm/s; avg over run {m.average_speed:.1f}, "
          f"total path incl. flights {m.distance:.1f} mm)  auto hits skipped while down: "
          f"{session.auto.n_skipped}", flush=True)
    if brain is not None:
        b = brain.summary()
        print(f"  brain: {b['brain_states']} states ({b['brain_states_dropped']} dropped), "
              f"{b['brain_stimuli_sent']} stimuli, brain time {b['brain_time_s'] or 0:.2f}s, "
              f"final lag {b['brain_lag_s'] or 0:.2f}s", flush=True)
    if session.logger is not None:
        print(f"  run dir: {session.logger.run_dir}", flush=True)
    sim.close()
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = config_from_args(args)
    if cfg.brain.enabled:
        problem = missing_requirements(cfg.brain)
        if problem:
            print(f"ERROR: {problem}", file=sys.stderr)
            return 2
    try:
        run(cfg, headless=args.headless, max_seconds=args.max_seconds, record=args.record,
            script_keys=parse_script_keys(args.script_keys))
    except SimulationInstabilityError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    return 0
