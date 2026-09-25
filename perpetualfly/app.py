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

from perpetualfly.config import AppConfig
from perpetualfly.interaction import HitEvent, install_perturbation
from perpetualfly.interaction.perturb_controls import HIT_MODES, format_event, format_whip_event
from perpetualfly.interaction.whip import Whip, WhipHitEvent
from perpetualfly.metrics import FallDetector, FallEvent, FallState, RunLogger, RunMetrics
from perpetualfly.simulation import Simulation, SimulationInstabilityError, WorldExtension
from perpetualfly.terrain import DIFFICULTY_PRESETS, ProceduralTerrain, ProceduralTerrainConfig

# (keys, description)
KEY_TABLE: list[tuple[str, str]] = [
    ("SPACE", "whip: crack from a random side | shove: random direction (+ upward)"),
    ("LEFT / RIGHT", "whip: crack from the fly's left / right | shove: to its left / right"),
    ("UP / DOWN", "whip: crack from the front / rear | shove: forward / backward"),
    ("U", "whip: overhead crack | shove: straight up"),
    ("1 2 3 4", "hit strength: gentle / medium / hard / absurd"),
    ("H", "hit mode: whip <-> shove"),
    ("A", "toggle automatic random hits (current hit mode)"),
    ("R", "spawn a rock ahead"),
    ("B", "spawn a bump ahead"),
    ("S", "spawn a slope ahead"),
    ("G", "spawn a gap ahead"),
    ("D", "spawn a dip ahead"),
    ("F", "flatten the next terrain chunk"),
    ("P", "pause / resume"),
    ("X", "reset the fly (explicit reset, counted in the metrics)"),
    ("C", "camera: follow / side / top"),
    ("?", "print this key help"),
    ("Q / ESC", "quit (closing the window or Ctrl-C also quits)"),
]
KEY_HELP = "SPACE/arrows/U hit | 1-4 strength | H whip/shove | A auto | R B S G D spawn | F flatten | P X C | ? help | Q quit"
SPAWN_KEYS = {"r": "rock", "b": "bump", "s": "slope", "g": "gap", "d": "dip"}
TERRAIN_CHOICES = tuple(DIFFICULTY_PRESETS)  # flat, easy, normal, hard, chaos


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def key_help_text() -> str:
    w = max(len(k) for k, _ in KEY_TABLE)
    return "Keys (window focused):\n" + "\n".join(f"  {k:<{w}}  {d}" for k, d in KEY_TABLE)


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
    ) -> None:
        self.cfg = cfg
        self.say = say or (lambda msg: print(msg, flush=True))
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
        if cfg.session.hit_mode not in HIT_MODES:
            raise ValueError(f"session.hit_mode must be one of {HIT_MODES}")
        # Always install the auto perturber (so A can switch it on); it starts in
        # the state given by cfg.auto_perturb.enabled and uses the current hit mode.
        self.controls = install_perturbation(sim, cfg.perturbation, cfg.auto_perturb,
                                             whip=self.whip, mode=cfg.session.hit_mode)
        self.perturbation = self.controls.perturbation
        self.auto = self.controls.auto
        self.detector = FallDetector(sim, cfg.falls, ground_height_fn=self.ground_height)
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
            self.logger = RunLogger(
                cfg.logging, config=self.full_config(),
                terrain_type_fn=self.terrain.terrain_type_at,
                ground_height_fn=self.ground_height,
            ).attach(sim, self.detector, self.metrics)
            # Whip hits go to events.csv as "whip" rows (shove hits stay "hit").
            hl = self.metrics.hit_listeners
            hl[hl.index(self.logger.log_hit)] = self._log_hit_record
        self.perturbation.listeners.append(self._on_hit)
        if self.whip is not None:
            self.whip.listeners.append(self._on_whip)
        self.detector.add_listener(self._on_fall_event)
        self.down_since: float | None = None  # sim time of the last fall (until recovered/reset)
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
            self._hint_shown = False
        elif ev.kind in ("recovered", "reset"):
            self.down_since = None
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
        """Once per physics chunk: fall-reset hint and optional auto reset."""
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

    def summary_extra(self, quit_reason: str) -> dict:
        return {
            "quit_reason": quit_reason,
            "terrain_difficulty": self.terrain.cfg.difficulty,
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
    g.add_argument("--controller", choices=["hybrid", "cpg"], default=None)
    g.add_argument("--seed", type=int, default=None, help="controller (CPG) seed")
    g.add_argument("--camera", choices=["follow", "side", "top"], default=None)
    g.add_argument("--render-every", type=int, default=None,
                   help="physics steps between rendered frames")
    g.add_argument("--no-thread", action="store_true",
                   help="live window: step physics on the main thread (slower; debugging)")
    g.add_argument("--print-interval", type=float, default=None,
                   help="simulated seconds between terminal stat lines")
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


def config_from_args(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig.load_json(args.config) if args.config else AppConfig()
    if args.controller:
        cfg.controller.kind = args.controller
    if args.seed is not None:
        cfg.controller.seed = args.seed
    if args.camera:
        cfg.camera.mode = args.camera
    if args.render_every:
        cfg.render.render_every_steps = args.render_every
    if getattr(args, "no_thread", False):
        cfg.render.threaded_physics = False
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


def run(
    cfg: AppConfig,
    *,
    headless: bool = False,
    max_seconds: float | None = None,
    record: Path | None = None,
    log: bool | None = None,
    world_extensions: list[WorldExtension] | tuple[WorldExtension, ...] = (),
) -> RunResult:
    """Run until quit (or ``max_seconds`` of simulated run time). ``log=None``
    follows ``cfg.logging.enabled``."""
    session = Session(cfg, log=log, world_extensions=world_extensions)
    sim, metrics, detector, terrain = session.sim, session.metrics, session.detector, session.terrain
    chunk = max(1, cfg.render.render_every_steps)
    need_frames = (not headless) or record is not None

    frame_renderer = viewer = writer = None
    st = _LoopState(next_print=cfg.stats.print_interval_s, last_print_wall=time.perf_counter(),
                    last_print_rt=0.0)
    try:
        if need_frames:
            from perpetualfly.rendering import FrameRenderer

            frame_renderer = FrameRenderer(sim.model, cfg.render, cfg.camera)
        if not headless:
            from perpetualfly.interaction import LiveViewer

            viewer = LiveViewer(cfg.render.window_title)
        if record is not None:
            import imageio.v2 as iio

            record.parent.mkdir(parents=True, exist_ok=True)
            fps = 1.0 / (chunk * sim.timestep)  # real-time playback
            writer = iio.get_writer(record, fps=fps, codec="libx264", quality=8,
                                    macro_block_size=8)
    except BaseException:
        session.close("startup error")
        raise

    ap = cfg.auto_perturb
    print(f"PerpetualFly | dt={sim.timestep:g}s | fly mass={sim.fly_mass * 1e3:.3f} mg | "
          f"controller={cfg.controller.kind} | terrain={terrain.cfg.difficulty} "
          f"(seed {terrain.cfg.seed}) | hit mode {session.hit_mode} | "
          f"strength L{session.perturbation.level} "
          f"{session.perturbation.level_name} | auto-perturb "
          f"{'on' if session.auto.enabled else 'off'} "
          f"({ap.min_interval_s:g}-{ap.max_interval_s:g}s, levels {list(ap.levels)}) | "
          f"{'headless' if headless else 'window'}", flush=True)
    if session.logger is not None:
        print(f"logging to {session.logger.run_dir}/", flush=True)
    if not headless:
        print(key_help_text(), flush=True)

    def status_line() -> str:
        wall = time.perf_counter()
        rt = session.run_time()
        dw = wall - st.last_print_wall
        rtf = (rt - st.last_print_rt) / dw if dw > 0 else 0.0
        st.last_print_wall, st.last_print_rt = wall, rt
        return (f"{metrics.summary_line()}  terrain={session.terrain_here():<10} "
                f"h={session.height_above_ground():4.2f}mm tilt={sim.tilt_deg():5.1f}deg "
                f"RTF={rtf:4.2f}")

    def after_physics() -> bool:
        """Physics side, after each chunk of sim.step: hints / auto reset, terminal
        line. Returns True once max_seconds is reached. Threaded mode: runs in the
        worker thread, under the lock."""
        session.after_physics()
        rt = session.run_time()
        if rt >= st.next_print:
            while st.next_print <= rt:
                st.next_print += cfg.stats.print_interval_s
            print(status_line(), flush=True)
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
            f"terrain {terrain.cfg.difficulty}: {session.terrain_here()}",
            session.controls.hud_line(),
            f"cam {frame_renderer.camera.mode}   {'PAUSED' if st.paused else ''}",
        ]
        down = session.down_for()
        if down is not None and detector.state != FallState.UPRIGHT:
            lines.append(f"DOWN {down:4.1f}s - press X to reset")
        lines.append("? = key help (terminal)")
        return lines

    def handle_keys(keys: list[str]) -> None:
        """Display side; touches the sim, so threaded mode calls it under the lock."""
        for k in keys:
            msg = None
            if k in ("q", "esc"):
                st.quit_reason = "quit key"
            elif k == "p":
                st.paused = not st.paused
                msg = "[paused]" if st.paused else "[resumed]"
            elif k == "x":
                session.reset("manual")
                frame_renderer.camera.reset()
                msg = f"[reset] explicit reset #{metrics.n_resets}"
            elif k == "c":
                msg = f"[camera] {frame_renderer.camera.cycle_mode()}"
            elif k in SPAWN_KEYS:
                msg = session.spawn(SPAWN_KEYS[k])
            elif k == "f":
                msg = session.flatten()
            elif k in ("?", "/"):
                msg = key_help_text()
            else:
                msg = session.handle_whip_key(k)
            if msg:
                print(msg, flush=True)

    def display_frame(frame) -> list[str]:
        viewer.show(frame, st.hud)
        keys = viewer.poll_keys(30 if st.paused else 1)
        if not viewer.is_open():
            st.quit_reason = "window closed"
        return keys

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
                    keys = display_frame(frame_renderer.draw())
                    if keys:
                        with runner.locked():
                            handle_keys(keys)
                        runner.paused = st.paused
                        runner.reset_clock()
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
        for closer in (writer, viewer, frame_renderer):
            if closer is not None:
                try:
                    closer.close()
                except Exception as e:  # never mask the real error / skip the summary
                    print(f"warning: closing {type(closer).__name__} failed: {e}", file=sys.stderr)
        session.close(st.quit_reason)
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
    )
    print(f"[done: {quit_reason}] {m.summary_line()}  "
          f"thorax=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) mm", flush=True)
    print(f"  resets={m.n_resets} (auto {session.n_auto_resets})  spawns={terrain.spawn_count}  "
          f"chunks recycled={terrain.recycle_count}  falls/km={_fmt(m.falls_per_km)}  "
          f"recovery%={_fmt(m.recovery_percentage)}", flush=True)
    if session.logger is not None:
        print(f"  run dir: {session.logger.run_dir}", flush=True)
    sim.close()
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = config_from_args(args)
    try:
        run(cfg, headless=args.headless, max_seconds=args.max_seconds, record=args.record)
    except SimulationInstabilityError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    return 0
