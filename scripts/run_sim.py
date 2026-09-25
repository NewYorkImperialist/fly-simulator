#!/usr/bin/env python
"""Entry point: ``python scripts/run_sim.py [--headless] [--max-seconds S] [--record out.mp4]``."""

import sys
from pathlib import Path

# Allow running from a checkout without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perpetualfly.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
