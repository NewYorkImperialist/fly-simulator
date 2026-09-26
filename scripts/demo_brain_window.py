"""Open the PerpetualFly brain window on the mock brain.

    .venv/bin/python scripts/demo_brain_window.py --mock            # live window
    .venv/bin/python scripts/demo_brain_window.py --mock --process  # child process
    .venv/bin/python scripts/demo_brain_window.py --record brain.mp4 --seconds 12

See fly_simulator/brain_viz/demo.py for all options.
"""

from fly_simulator.brain_viz.demo import main

if __name__ == "__main__":
    main()
