# Documentation

One line per document. Files derived from FlyWire data are CC BY-NC 4.0; see
[NOTICE.md](../NOTICE.md).

## Getting started

* [README](../README.md): what the project is, feature tour, quickstart, main keys.
* [USAGE.md](USAGE.md): full reference for the app: install, run examples, every key and flag, outputs, robustness evaluation.
* [ARCHITECTURE.md](ARCHITECTURE.md): code layout, how the physical whip works, performance and known limitations.
* [CONTRIBUTING](../CONTRIBUTING.md): development setup, tests, style and the honesty rule.
* [NOTICE](../NOTICE.md): third-party credits, data licences and the reference list.

## Features

* [ACTIONS.md](ACTIONS.md): the action library (jump, freeze, groom, back away, turn, wing raise, proboscis).
* [WHIP.md](WHIP.md): the physical whip that hits the fly only through MuJoCo contacts.
* [SWATTER.md](SWATTER.md): a flyswatter the fly sees coming and dodges with its brain.
* [COURSE.md](COURSE.md): hand-designed, timed obstacle courses.
* [JOBS.md](JOBS.md): "eternal jobs" scenes (Sisyphus, hamster wheel, mowing, raking, doner kebab).
* [FLIGHT.md](FLIGHT.md): flapping-wing flight with MuJoCo's fluid model.
* [SCENES.md](SCENES.md): scripted cinematic scenes rendered offscreen (`temple_standoff`).
* [GAMES.md](GAMES.md): games played by the connectome brain (asteroids, chase, fly through rings).
* [TASTE.md](TASTE.md): sugar and bitter patches tasted with the legs.
* [RL.md](RL.md): residual-RL environment, PPO training and evaluation scripts.

## Brain

* [BRAIN.md](BRAIN.md): the Shiu et al. 2024 whole-brain LIF model on FlyWire v783, data download, sensory and motor mapping.
* [BRAIN_WINDOW.md](BRAIN_WINDOW.md): the live brain activity window.
* [BRAIN_REPLAY.md](BRAIN_REPLAY.md): record a run's brain activity and replay or render it later.
* [PLAYGROUND.md](PLAYGROUND.md): virtual optogenetics, virtual lesions and decision meters.
* [VISION.md](VISION.md): geometric looming sense and real vision (compound eyes, flyvis, LC4 / LPLC2).
* [STRESS.md](STRESS.md): octopamine arousal layer (whip hits speed the fly up).
* [HABITUATION.md](HABITUATION.md): short-term depression makes the giant-fibre escape habituate to harmless looms.
* [FEAR_LEARNING.md](FEAR_LEARNING.md): dopamine-gated mushroom-body learning of an odour paired with whipping.
* [SMELL.md](SMELL.md): why olfactory input ignites the model's runaway, antennal-lobe fixes, and an odour plume (tracking: negative result).

## Science notes

* [SENSORY_SCREEN.md](SENSORY_SCREEN.md): which sensory inputs of the brain model reach which descending neurons.
* [sensory_screen.csv](sensory_screen.csv), [sensory_screen_touch_oa.csv](sensory_screen_touch_oa.csv): raw screen results (CC BY-NC 4.0).

## Development

* [dev/SPEC.md](dev/SPEC.md): the original project spec and milestones.
* [dev/STATUS.md](dev/STATUS.md): what works, soak and robustness results, known issues.
* [dev/ROADMAP.md](dev/ROADMAP.md): planned features compared by effort and how much comes from the real wiring.
* [dev/API_NOTES.md](dev/API_NOTES.md): FlyGym 2.x / MuJoCo API facts verified against the installed code.
* [dev/PERTURBATION_CALIBRATION.md](dev/PERTURBATION_CALIBRATION.md): calibration of the thorax-force shove strength levels.
