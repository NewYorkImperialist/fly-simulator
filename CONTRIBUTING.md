# Contributing

Thanks for your interest. Bug reports, fixes, new scenes and better biology are all
welcome. Please open an issue first for larger changes.

## Setup

Python 3.12 (the tested version; FlyGym 2.1 allows 3.12-3.14).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev,brain]"
# optional: the FlyWire connectome (~153 MB, CC BY-NC 4.0) for the brain model
.venv/bin/python scripts/fetch_brain_data.py
```

Other extras: `rl` (PPO), `vision` (flyvis real vision), `brain-ref` (Brian2
cross-check). See the [README](README.md), [docs/USAGE.md](docs/USAGE.md) and
[docs/README.md](docs/README.md).

## Tests

```bash
.venv/bin/python -m pytest -q
```

Tests that need the FlyWire data, flyvis, Brian2 or gymnasium skip themselves when
those are missing. `FLY_SIMULATOR_BRAIN_DATA=/nonexistent` runs the suite as if the
data were not downloaded (this is what CI does). On a headless Linux machine set
`MUJOCO_GL=egl` (or `osmesa`) for rendering; where no OpenGL context is
available at all, `FLY_SIMULATOR_SKIP_RENDERING=1` skips the tests that render. New behaviour needs a test; keep tests
headless (no windows) and reasonably fast.

## Style

* Follow the existing code: type hints, `from __future__ import annotations`, short
  docstrings that say what a module models and where the numbers come from.
* Lines up to about 100 characters. No new dependencies in the core install without
  a good reason; put optional ones in an extra.
* Put measured numbers in the docs together with how they were measured.

## Honesty rule: don't fake physics or brain

This project is about a fly whose behaviour comes from real physics and, where
claimed, real connectome wiring. So:

* **Physics**: the fly moves only through MuJoCo (actuators, contacts, documented
  external forces). Don't teleport it or add hidden forces to make a behaviour
  work. Scripted parts (whip handle, props, resets) must be explicit and documented.
* **Brain**: if a behaviour is said to come from the FlyWire brain, the decision has
  to come from the model's spikes. Hand-built mappings, stand-in neurons, thresholds
  and phenomenological layers (for example habituation or octopamine) are fine,
  but they must be labelled as such in the docs and, where the user sees it, in the
  HUD (see the labels in docs/GAMES.md and the "brain-real" column in
  docs/dev/ROADMAP.md).
* Report negative results as they are (for example docs/FEAR_LEARNING.md: learning
  in the brain, but no avoidance behaviour).
* Cite the papers that numbers and circuits come from, and add new references to
  [NOTICE.md](NOTICE.md).

## Licences

Code contributions are accepted under the MIT licence. Anything derived from FlyWire
data is CC BY-NC 4.0 and must be listed in NOTICE.md. Don't commit the downloaded
data (`data/` is gitignored).
