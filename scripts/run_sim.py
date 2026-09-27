#!/usr/bin/env python
"""Entry point: ``python scripts/run_sim.py [--headless] [--max-seconds S] [--record out.mp4]``.

``--game NAME --brain`` (or ``--synthetic-brain``) plays a FLY BRAIN PLAYS game through
the same runner as ``scripts/play.py`` (docs/GAMES.md, docs/USAGE.md).
"""

import sys
from pathlib import Path

# Allow running from a checkout without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_simulator.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
