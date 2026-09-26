"""Run a FLY BRAIN PLAYS game: headless, in a window, recorded, or as the experiment.

Used by ``scripts/play.py`` (``play(args)``); the pieces are usable on their own
(``HighScores``, ``GameRunner``).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HIGHSCORE_FILE = Path("runs") / "games_highscores.json"
KEEP_SCORES = 5
FPS = 30.0
KEYS_HELP = "SPACE pause  R restart  1/2/3 easy/normal/hard  B brain window  Q quit"


class HighScores:
    """Tiny JSON file: {game: {"<control>:<difficulty>": [top KEEP_SCORES entries]}}."""

    def __init__(self, path: Path | str = HIGHSCORE_FILE) -> None:
        self.path = Path(path)
        try:
            self.data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {}

    @staticmethod
    def key(control: str, difficulty: str) -> str:
        return f"{control}:{difficulty}"

    def best(self, game: str, control: str, difficulty: str) -> int:
        lst = self.data.get(game, {}).get(self.key(control, difficulty), [])
        return int(lst[0]["score"]) if lst else 0

    def add(self, game: str, control: str, difficulty: str, entry: dict) -> int:
        """Insert; returns the rank (1 = new best, 0 = not in the top list). Saves."""
        lst = self.data.setdefault(game, {}).setdefault(self.key(control, difficulty), [])
        entry = dict(entry, date=time.strftime("%Y-%m-%d %H:%M"))
        lst.append(entry)
        lst.sort(key=lambda e: -e["score"])
        del lst[KEEP_SCORES:]
        rank = lst.index(entry) + 1 if entry in lst else 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))
        return rank


def parse_script_keys(text: str | None) -> list[tuple[float, str]]:
    """"2:space,4:space,9:r" -> [(2.0, "space"), ...] (game time in s)."""
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        t, _, k = part.partition(":")
        out.append((float(t), k.strip().lower()))
    return sorted(out)


@dataclass
class GameRunner:
    session: object
    game_name: str = "asteroids"
    highscores: HighScores | None = None
    render: bool = False
    width: int = 960
    height: int = 640
    panel: bool = True
    say: object = print
    paused: bool = False
    frames_written: list = field(default_factory=list)

    def __post_init__(self):
        self.renderer = None
        if self.render:
            from perpetualfly.games.session import make_renderer

            self.renderer = make_renderer(self.session.sim, self.width, self.height,
                                          **getattr(self.session, "camera", {}))
        self._recorded = False
        self._events_seen = 0
        self.brain_window = None

    # ---------------------------------------------------------------- helpers
    @property
    def game(self):
        return self.session.game

    def best(self) -> int:
        if self.highscores is None:
            return 0
        g = self.game
        return self.highscores.best(self.game_name, self.session.brain.control, g.cfg.difficulty)

    def frame(self) -> np.ndarray:
        from perpetualfly.games.hud import compose
        from perpetualfly.games.session import render_frame

        if hasattr(self.session, "render"):
            rgb = self.session.render(self.renderer)
        else:
            rgb = render_frame(self.renderer, self.session.sim)
        return compose(rgb, self.session, panel=self.panel, high_score=self.best(),
                       paused=self.paused, hint=KEYS_HELP)

    def print_events(self) -> None:
        evs = self.game.events[self._events_seen:]
        self._events_seen = len(self.game.events)
        for e in evs:
            if e.kind == "spawn":
                continue
            extra = " ".join(f"{k}={v}" for k, v in e.info.items())
            self.say(f"[{e.t:7.2f}s] {e.kind:8s} {e.text} {extra}")

    def record_score(self) -> int:
        if self._recorded or self.highscores is None or self.game.state != "gameover":
            return 0
        self._recorded = True
        g = self.game
        if hasattr(g, "score_entry"):
            entry = g.score_entry()
        else:
            entry = {"score": int(g.score), "survival_s": round(g.survival_s, 2),
                     "wave": g.wave, "dodges": g.dodges}
        rank = self.highscores.add(self.game_name, self.session.brain.control,
                                   g.cfg.difficulty, entry)
        if rank == 1:
            self.say(f"NEW HIGH SCORE {int(g.score)} ({self.session.brain.control}, "
                     f"{g.cfg.difficulty}) -> {self.highscores.path}")
        return rank

    def handle_key(self, key: str) -> bool:
        """Returns False to quit."""
        s = self.session
        if key in ("q", "esc"):
            return False
        if key == "space":
            self.paused = not self.paused
        elif key == "r":
            s.restart()
            self._recorded = False
            self.paused = False
        elif key in ("1", "2", "3"):
            s.restart({"1": "easy", "2": "normal", "3": "hard"}[key])
            self._recorded = False
            self.paused = False
        elif key == "b":
            self.toggle_brain_window()
        return True

    def toggle_brain_window(self) -> None:
        b = self.session.brain
        if b.layout is None:
            self.say("[brain window] no brain running (control none)")
            return
        if self.brain_window is not None and self.brain_window.is_alive():
            self.brain_window.close()
            self.brain_window = None
            return
        from perpetualfly.brain_viz import BrainWindowProcess

        self.brain_window = BrainWindowProcess(b.layout).start()
        bw = self.brain_window
        if self._send_state not in b.listeners:
            b.listeners.append(self._send_state)
        self.say(f"[brain window] started (pid {bw.process.pid})")

    def _send_state(self, st) -> None:
        if self.brain_window is not None:
            self.brain_window.send(st)

    def close(self) -> None:
        if self.brain_window is not None:
            try:
                self.brain_window.close()
            except Exception:
                pass
        if self.renderer is not None:
            self.renderer.close()

    # ---------------------------------------------------------------- loops
    def run_headless(self, max_seconds: float, record: Path | None = None,
                     frame_times: list[float] | None = None, frames_dir: Path | None = None,
                     script_keys=None, stop_at_gameover: bool = True) -> dict:
        """Run until game over (or ``max_seconds`` of game time). With ``record`` /
        ``frame_times`` frames are rendered at ``FPS`` per game second."""
        s = self.session
        writer = None
        if record is not None:
            import imageio.v2 as iio

            record.parent.mkdir(parents=True, exist_ok=True)
            writer = iio.get_writer(record, fps=FPS, codec="libx264", quality=6,
                                    macro_block_size=8)
        frame_times = sorted(frame_times or [])
        keys = list(script_keys or [])
        t0 = s.game.time()
        next_frame = t0
        wall0 = time.time()
        try:
            while s.game.time() - t0 < max_seconds:
                while keys and s.game.time() - t0 >= keys[0][0]:
                    if not self.handle_key(keys.pop(0)[1]):
                        return self.result(wall0)
                if self.paused:
                    self.paused = False  # no pausing headless
                s.step()
                self.print_events()
                t = s.game.time()
                if writer is not None and t >= next_frame:
                    writer.append_data(self.frame()[..., ::-1])
                    next_frame += 1.0 / FPS
                while frame_times and t - t0 >= frame_times[0]:
                    ft = frame_times.pop(0)
                    self.save_frame(frames_dir, f"t{ft:05.1f}s")
                if s.game.state == "gameover":
                    self.record_score()
                    if stop_at_gameover:
                        # a short tail so a recording shows the game-over screen
                        tail = s.game.time() + (1.0 if writer is not None else 0.0)
                        while s.game.time() < tail:
                            s.step()
                            if writer is not None and s.game.time() >= next_frame:
                                writer.append_data(self.frame()[..., ::-1])
                                next_frame += 1.0 / FPS
                        if frame_times and frames_dir is not None:
                            self.save_frame(frames_dir, "gameover")
                        break
        finally:
            if writer is not None:
                writer.close()
        return self.result(wall0)

    def save_frame(self, frames_dir: Path | None, tag: str) -> Path | None:
        if frames_dir is None or self.renderer is None:
            return None
        import cv2

        frames_dir.mkdir(parents=True, exist_ok=True)
        p = frames_dir / f"{self.game_name}_{self.session.brain.control}_{tag}.png"
        cv2.imwrite(str(p), self.frame())
        self.frames_written.append(str(p))
        return p

    def run_window(self, max_seconds: float | None = None, script_keys=None,
                   record: Path | None = None, scale: float = 1.0) -> dict:
        import cv2

        s = self.session
        title = "FLY BRAIN PLAYS"
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        keys = list(script_keys or [])
        writer = None
        if record is not None:
            import imageio.v2 as iio

            record.parent.mkdir(parents=True, exist_ok=True)
            writer = iio.get_writer(record, fps=FPS, codec="libx264", quality=6,
                                    macro_block_size=8)
        wall0 = time.time()
        steps_per_frame = max(1, int(round((1.0 / FPS) / (s.sim.timestep * s.chunk_steps))))
        keymap = {27: "esc", 32: "space"}
        try:
            while True:
                wall = time.time() - wall0
                if max_seconds is not None and wall > max_seconds:
                    break
                quit_ = False
                while keys and wall >= keys[0][0]:  # scripted keys: wall-clock seconds
                    if not self.handle_key(keys.pop(0)[1]):
                        quit_ = True
                        break
                if quit_:
                    break
                if not self.paused:
                    for _ in range(steps_per_frame):
                        s.step()
                    self.print_events()
                    if s.game.state == "gameover":
                        self.record_score()
                img = self.frame()
                if writer is not None and not self.paused:
                    writer.append_data(img[..., ::-1])
                if scale != 1.0:
                    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                cv2.imshow(title, img)
                k = cv2.waitKey(1 if not self.paused else 30) & 0xFF
                if k != 255 and not self.handle_key(keymap.get(k, chr(k).lower())):
                    break
                try:
                    if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
        finally:
            if writer is not None:
                writer.close()
            cv2.destroyWindow(title)
            for _ in range(3):
                cv2.waitKey(1)
        return self.result(wall0)

    def result(self, wall0: float) -> dict:
        g = self.game
        out = g.summary()
        out.update(control=self.session.brain.control, wall_s=round(time.time() - wall0, 1),
                   game_time_s=round(g.time(), 2), jumps=len(self.session.jump_times),
                   brain_states=self.session.brain.n_states,
                   stim_events=(v.n_sent() if hasattr(v := self.session.vision, "n_sent")
                                else len(v.sent)),
                   frames=self.frames_written)
        return out


# ---------------------------------------------------------------------------
# entry point used by scripts/play.py
# ---------------------------------------------------------------------------


def play(args) -> int:
    from perpetualfly.games import GAMES, GameBrain
    from perpetualfly.games.asteroids import AsteroidConfig
    from perpetualfly.games.brain_io import GameMapping

    if args.game not in GAMES:
        print(f"unknown game {args.game!r}", file=sys.stderr)
        return 2
    synthetic = {"n": 60, "p_conn": 0.1, "seed": args.seed} if args.synthetic_brain else None
    if not synthetic:
        from perpetualfly.brain_link import BrainLinkConfig, missing_requirements

        msg = missing_requirements(BrainLinkConfig(enabled=True, data_dir=args.data_dir))
        if msg:
            print(msg, file=sys.stderr)
            return 2
    mapping = GameMapping(jump=args.jump, jump_mode=args.jump_mode)
    control = args.control if args.experiment is None else "brain"
    brain = GameBrain(control, synthetic=synthetic, data_dir=args.data_dir, mapping=mapping,
                      seed=args.seed)
    t0 = time.time()
    if control != "none" or args.experiment is not None:
        print(f"[play] starting the brain ({'synthetic' if synthetic else 'FlyWire v783'}) ...",
              flush=True)
    brain.start(force=args.experiment is not None)
    if brain.brain is not None:
        print(f"[play] brain ready in {time.time() - t0:.1f}s "
              f"({brain.info.get('n_neurons', '?')} neurons)", flush=True)
    runner = None
    try:
        if args.game == "rings":
            from perpetualfly.games.rings import RINGS_DIFFICULTIES, RingsConfig
            from perpetualfly.games.session import RingsSession

            if args.difficulty not in RINGS_DIFFICULTIES:
                print(f"difficulty must be one of {sorted(RINGS_DIFFICULTIES)}", file=sys.stderr)
                return 2
            session = RingsSession(brain, RingsConfig(difficulty=args.difficulty,
                                                      lives=args.lives,
                                                      takeoff=not args.air_start),
                                   seed=args.seed)
            title = "FLY THROUGH RINGS"
        elif args.game == "chase":
            from perpetualfly.games.chase import CHASE_DIFFICULTIES, ChaseConfig
            from perpetualfly.games.session import ChaseSession

            if args.difficulty not in CHASE_DIFFICULTIES:
                print(f"difficulty must be one of {sorted(CHASE_DIFFICULTIES)}", file=sys.stderr)
                return 2
            session = ChaseSession(brain, ChaseConfig(difficulty=args.difficulty,
                                                      lives=args.lives), seed=args.seed)
            title = "FOLLOW THE LEADER"
        else:
            from perpetualfly.games.session import AsteroidSession

            cfg = AsteroidConfig(difficulty=args.difficulty, lives=args.lives)
            session = AsteroidSession(brain, cfg, seed=args.seed)
            title = "ASTEROID DODGE"
        if args.experiment is not None:
            return _experiment(session, args)
        from perpetualfly.games import HONEST_LABEL

        print(f"[play] {title}  control={control}  difficulty={args.difficulty}  "
              f"jump={'on' if args.jump and args.game == 'asteroids' else 'off'}\n"
              f"[play] {HONEST_LABEL}", flush=True)
        need_render = args.window or args.record is not None or args.frames is not None
        hs = HighScores(args.highscores) if not args.no_highscore else None
        runner = GameRunner(session, game_name=args.game, highscores=hs, render=need_render,
                            panel=not args.no_panel, width=args.width, height=args.height)
        if args.brain_window:
            runner.toggle_brain_window()
        keys = parse_script_keys(args.script_keys)
        frame_times = ([float(x) for x in args.frame_times.split(",")] if args.frame_times
                       else None)
        if args.window:
            res = runner.run_window(max_seconds=args.max_wall_seconds, script_keys=keys,
                                    record=args.record, scale=args.scale)
        else:
            res = runner.run_headless(args.max_seconds, record=args.record,
                                      frame_times=frame_times, frames_dir=args.frames,
                                      script_keys=keys)
        print("[play] " + json.dumps(res))
        return 0
    finally:
        if runner is not None:
            runner.close()
        brain.close()


def _experiment(session, args) -> int:
    controls = tuple(args.controls.split(","))
    say = lambda m: print(m, flush=True)  # noqa: E731
    if getattr(session, "game_name", "asteroids") == "rings":
        from perpetualfly.games.rings_experiment import (
            format_rings_summary, run_rings_experiment, summarize_rings)

        rows = run_rings_experiment(session, args.experiment, controls=controls, seed=args.seed,
                                    say=say)
        summ = summarize_rings(rows)
        text = format_rings_summary(summ)
    elif getattr(session, "game_name", "asteroids") == "chase":
        from perpetualfly.games.chase_experiment import (
            format_chase_summary, run_chase_experiment, summarize_chase)

        rows = run_chase_experiment(session, args.experiment, controls=controls, seed=args.seed,
                                    say=say, trial_s=args.trial_seconds)
        summ = summarize_chase(rows)
        text = format_chase_summary(summ)
    else:
        from perpetualfly.games.experiment import format_summary, run_experiment, summarize

        rows = run_experiment(session, args.experiment, controls=controls, seed=args.seed,
                              speed=args.rock_speed, say=say)
        summ = summarize(rows)
        text = format_summary(summ)
    print()
    print(text)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"rows": rows, "summary": summ}, indent=1))
        print(f"[play] wrote {args.json}")
    return 0
