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

    # game 4, FLY PONG: the ball -> LC10a -> DNa01/02 -> the paddle's sideways speed
    # (the fly stands on the paddle's sled) against a scripted AI; first to 7
    .venv/bin/python scripts/play.py --game pong --brain --window
    .venv/bin/python scripts/play.py --game pong --brain --experiment 30 --json runs/pong_exp.json

    # game 5, CANYON RUN: real flapping-wing flight through rock pillars; each pillar
    # looms on the eye that sees it -> LC4 / LPLC2 -> opposite DNa01/02 (turn away)
    .venv/bin/python scripts/play.py --game canyon --brain --window
    .venv/bin/python scripts/play.py --game canyon --brain --experiment 20 --json runs/canyon_exp.json
"""

from __future__ import annotations

import sys


def parse_args(argv=None):
    from fly_simulator.games.runner import build_parser

    return build_parser(__doc__).parse_args(argv)


def main(argv=None) -> int:
    from fly_simulator.games.runner import play

    return play(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
