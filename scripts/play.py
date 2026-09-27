"""FLY BRAIN PLAYS: games controlled by the real FlyWire connectome brain.

Brain responses are real connectome wiring (FlyWire v783, Shiu et al. 2024 LIF model);
the game interface (what the brain sees, how its outputs map to controls) is designed
by us. See docs/GAMES.md.

    # play in a window (brain panel on the right; SPACE pause, R restart, 1/2/3 difficulty,
    # B brain window, Q quit)
    .venv/bin/python scripts/play.py --game asteroids --brain --window
    # headless game, record a clip / save frames
    .venv/bin/python scripts/play.py --game asteroids --brain --record runs/asteroids.mp4
    .venv/bin/python scripts/play.py --game asteroids --brain --frames runs/frames --frame-times 2,5,8
    # controls: brain disconnected (constant walk) or eyes mirrored
    .venv/bin/python scripts/play.py --game asteroids --brain --control none
    .venv/bin/python scripts/play.py --game asteroids --brain --control mirror
    # the scientific check: paired single-rock trials, brain vs mirror vs none
    .venv/bin/python scripts/play.py --game asteroids --brain --experiment 40 --json runs/exp.json
    # quick tests without the connectome data
    .venv/bin/python scripts/play.py --game asteroids --synthetic-brain --max-seconds 5

    # game 2, FOLLOW THE LEADER: a leader fly weaves ahead; LC10a (pursuit) -> DNa01/02
    .venv/bin/python scripts/play.py --game chase --brain --window
    .venv/bin/python scripts/play.py --game chase --brain --experiment 24 --json runs/chase_exp.json

    # game 3, FLY THROUGH RINGS: real flapping-wing flight; the next ring -> LC10a ->
    # DNa01/02 -> heading rate of the flight controller (speed / altitude held)
    .venv/bin/python scripts/play.py --game rings --brain --window
    .venv/bin/python scripts/play.py --game rings --brain --experiment 16 --json runs/rings_exp.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fly_simulator.games import CONTROLS, DIFFICULTIES, GAMES


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", choices=GAMES, default="asteroids")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--brain", action="store_true",
                     help="the real FlyWire brain (data in data/brain)")
    src.add_argument("--synthetic-brain", action="store_true",
                     help="tiny random network (tests; no connectome, no real behaviour)")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--control", choices=CONTROLS, default="brain",
                   help="brain (default) | mirror (left eye -> right optic lobe) | "
                        "none (brain disconnected, constant walk)")
    p.add_argument("--difficulty", choices=sorted(DIFFICULTIES), default="normal")
    p.add_argument("--lives", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jump", action="store_true",
                   help="giant fibre (DNp01) > 60 Hz triggers a jump (off by default)")
    p.add_argument("--jump-mode", choices=("long", "short"), default="long")
    p.add_argument("--air-start", action="store_true",
                   help="rings: start in the air instead of the jump take-off")
    # presentation
    p.add_argument("--window", action="store_true", help="OpenCV game window")
    p.add_argument("--brain-window", action="store_true",
                   help="also open the brain activity window (separate process)")
    p.add_argument("--panel", action="store_true", help="start with the brain panel beside the game (off by default; TAB toggles it)")
    p.add_argument("--no-panel", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--scale", type=float, default=1.0, help="window display scale")
    p.add_argument("--record", type=Path, default=None, help="write an MP4 (30 fps game time)")
    p.add_argument("--frames", type=Path, default=None, help="directory for PNG frames")
    p.add_argument("--frame-times", default=None,
                   help="game seconds at which to save frames (with --frames), e.g. 2,5,8")
    p.add_argument("--max-seconds", type=float, default=600.0,
                   help="headless: stop after this much game time (default: at game over)")
    p.add_argument("--max-wall-seconds", type=float, default=None,
                   help="window: close after this many wall seconds")
    p.add_argument("--script-keys", default=None,
                   help='inject keys, e.g. "3:space,4:space,20:r" (headless: game s; window: wall s)')
    p.add_argument("--highscores", type=Path, default=Path("runs/games_highscores.json"))
    p.add_argument("--no-highscore", action="store_true")
    # experiment
    p.add_argument("--experiment", type=int, default=None, metavar="N",
                   help="paired trials per condition (brain, mirror, none): single rocks "
                        "(asteroids), leader runs (chase) or single rings (rings)")
    p.add_argument("--controls", default="brain,mirror,none")
    p.add_argument("--rock-speed", type=float, default=10.0, help="experiment rock speed (mm/s)")
    p.add_argument("--trial-seconds", type=float, default=6.0,
                   help="chase experiment: game seconds per leader run")
    p.add_argument("--json", type=Path, default=None, help="experiment rows + summary")
    return p.parse_args(argv)


def main(argv=None) -> int:
    from fly_simulator.games.runner import play

    return play(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
