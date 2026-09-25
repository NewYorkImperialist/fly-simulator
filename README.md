# PerpetualFly

A physically simulated fruit fly (FlyGym 2.x / NeuroMechFly on MuJoCo) that jogs
forward forever through endless procedural terrain while you whack it with physical
forces. Full plan: [docs/SPEC.md](docs/SPEC.md). FlyGym API reference for
contributors: [docs/API_NOTES.md](docs/API_NOTES.md). Hit strengths:
[docs/PERTURBATION_CALIBRATION.md](docs/PERTURBATION_CALIBRATION.md) (shove) and
[docs/WHIP.md](docs/WHIP.md) (physical whip).

**Status: Milestones 1–4** (plus a residual-RL env, not trained yet). Morning summary,
soak results and next steps: [docs/STATUS.md](docs/STATUS.md). NeuroMechFly walks with FlyGym's hybrid CPG controller
(plus a heading hold along +x) over chunked, recycled procedural terrain (rocks,
bumps, rough ground, blocks, slopes, gaps, dips). You can spawn obstacles and hit
the fly from the keyboard, or let an auto-perturber do it. By default hits are
cracks of a **physical whip**: a visible chain of capsules on a scripted handle whose
lash strikes the fly through real MuJoCo contact (the fly moves only through the
contact forces). H switches to the older **shove** mode (a constant external force
on the thorax). A fall detector tracks falls and recoveries. Every run is logged to
`runs/<timestamp>/`. There is no episode timeout.

## Install (macOS / Linux, Python 3.12–3.14)

FlyGym 2.1.0 needs Python >= 3.12, < 3.15 and pins `mujoco>=3.9,<3.10`.

```bash
cd fly-run
uv venv --python 3.12 .venv                    # or: python3.12 -m venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
# without uv:  .venv/bin/pip install -e ".[dev]"
```

This installs `flygym==2.1.0` (which includes the `flygym_demo` locomotion
controllers and `imageio`, used for `--record`), `mujoco`, `opencv-python` and `pytest`.

## Run

```bash
# interactive window, normal terrain (plain python works on macOS, no mjpython needed)
.venv/bin/python scripts/run_sim.py

# harder terrain + automatic random hits
.venv/bin/python scripts/run_sim.py --terrain hard --auto-perturb

# no window, terminal stats only, stop after 60 *simulated* seconds
.venv/bin/python scripts/run_sim.py --headless --terrain normal --auto-perturb --max-seconds 60

# unattended long run: explicit reset whenever the fly has been down for 5 s
.venv/bin/python scripts/run_sim.py --headless --auto-perturb --auto-reset-after 5

# render an mp4 offscreen (with or without a window)
.venv/bin/python scripts/run_sim.py --headless --max-seconds 10 --auto-perturb --record runs/walk.mp4

# flat ground, no run folder
.venv/bin/python scripts/run_sim.py --terrain flat --no-log
```

With the venv activated (`source .venv/bin/activate.fish` in fish, or
`source .venv/bin/activate` in bash/zsh), `python scripts/run_sim.py` or the
`perpetualfly` console script does the same thing.

## Keys (window focused)

| key | action |
|---|---|
| SPACE | whip: crack from a random side · shove: random direction (random azimuth, 10–35° upward) |
| LEFT / RIGHT | whip: crack *from* the fly's left / right (pushes it right / left) · shove: push to the fly's left / right |
| UP / DOWN | whip: crack from the front / rear (pushes it back / forward) · shove: push forward / backward |
| U | whip: overhead crack (chops down on the thorax) · shove: straight up |
| 1 2 3 4 | hit strength: gentle / medium / hard / absurd (default 2), both modes |
| H | hit mode: whip ↔ shove (default whip; the HUD shows `WHIP ...` or `SHOVE ...`) |
| A | toggle automatic random hits (they use the current hit mode) |
| R | spawn a rock ahead |
| B | spawn a bump ahead |
| S | spawn a slope ahead |
| G | spawn a gap ahead |
| D | spawn a dip ahead |
| F | flatten the next terrain chunk |
| P | pause / resume |
| X | reset the fly (an explicit reset, counted in the metrics) |
| C | camera: follow / side / top |
| ? | print the key table in the terminal |
| Q / ESC | quit (closing the window or Ctrl-C in the terminal also quits) |

Hit directions are relative to the fly's heading. Whip keys name the side the whip
comes from; a crack takes ~0.2 s from the key press to the strike (wind-up), and a key
pressed during a crack is queued (one at a time). Whip results are printed when the
strike window closes: `HIT <body>, impulse ... nN*s, peak ... uN` or `MISS`. Obstacles spawn `--spawn-distance` mm (default 6) ahead of
the thorax, and never inside the fly: they're pushed further out if needed. If the fly
has been down for 3 s, the terminal prints `press X to reset`, and the HUD shows how
long it has been down.

## Command-line flags

| flag | meaning |
|---|---|
| `--headless` | no window |
| `--max-seconds S` | stop after S simulated seconds of run time (counted across resets) |
| `--record out.mp4` | also write a video (real-time playback, one frame per `--render-every` steps) |
| `--config file.json` | load any fields of `perpetualfly.config.AppConfig` (same layout as `config.json["app"]` in a run folder) |
| `--terrain {flat,easy,normal,hard,chaos}` | terrain difficulty (default `normal`). `flat` still lets you spawn obstacles |
| `--terrain-seed N` | terrain layout seed (default 42) |
| `--spawn-distance MM` | how far ahead key-spawned obstacles appear |
| `--strength {1,2,3,4}` | initial hit strength |
| `--hit-mode {whip,shove}` | what keys and automatic hits do (default `whip`; H toggles) |
| `--auto-perturb` | start with automatic hits on (otherwise A turns them on); in whip mode each automatic hit is a crack |
| `--auto-min S`, `--auto-max S` | sim seconds between automatic hits, uniform (default 2–5) |
| `--auto-levels 1,2` or `1:0.6,2:0.4` | strength levels the automatic hits draw from (optional weights) |
| `--auto-seed N` | automatic-hit RNG seed |
| `--fall-hint-after S` | print "press X to reset" after the fly has been down S s (default 3, 0 = off) |
| `--auto-reset-after S` | reset automatically after the fly has been down S s (default off) |
| `--no-log` | don't write a run folder |
| `--runs-dir DIR` | parent folder for run folders (default `runs`) |
| `--log-hz HZ` | `metrics.csv` rows per simulated second (default 50) |
| `--controller {hybrid,cpg}`, `--seed N` | locomotion controller, CPG seed |
| `--camera {follow,side,top}` | initial camera |
| `--render-every N` | physics steps per recorded frame (default 150 = 66.7 fps) |
| `--no-thread` | window mode: step physics on the main thread (slower; debugging) |
| `--print-interval S` | simulated seconds between terminal lines (default 1) |

## Output

Terminal line, every simulated second:

```
t=   20.01s  dist=   280.4mm  speed now= 14.0 avg= 14.0 mm/s  falls=0 rec=0 hits=6  jog=  20.0s (max   20.0)  [UPRIGHT]  terrain=rocks      h=1.13mm tilt=  4.0deg RTF=0.61
```

`t` is run time (it keeps counting across resets), `dist` is xy path length, `jog` is
the current and longest run without a fall, `[...]` is the fall-detector state
(UPRIGHT / DESTABILIZED / FALLEN / RECOVERING), `terrain` is the terrain under the
thorax, `h` is thorax height above the local ground, `tilt` is the angle of the
thorax up-axis from vertical, and `RTF` is simulated seconds per wall second.
Hits, falls, recoveries, spawns and resets are also printed as they happen.

Run folder `runs/<YYYY-MM-DD_HH-MM-SS>/`:

| file | contents |
|---|---|
| `config.json` | `app` (the full `AppConfig`: fly, terrain, controller, sim, camera, render, stats, perturbation, auto_perturb, falls, logging, session), `procedural_terrain` (the full `ProceduralTerrainConfig`), `runtime` (argv, timestep, fly mass, body weight, pool size) |
| `events.csv` | one row per event: `hit` (shove), `whip` (whip hit: force_direction = commanded side e.g. `from_left`, force_magnitude = mean contact force, details = measured impulse vector / `impulse_uNs`, peak force `magnitude_uN`, contact duration, bodies hit), `whip_miss`, `hit_mode`, fall-detector transitions (`destabilized`, `stabilized`, `fall`, `recovering`, `recovered`, `relapse`, `reset`), `spawn`, `flatten`, `strength`, `auto_perturb`, `manual_reset`, `auto_reset`. Columns: timestamp (run time), sim_time, wall_time, event_type, force_direction, force_magnitude (µN), terrain_type, fall_detected, recovered, state, x, y, z, details (JSON) |
| `metrics.csv` | sampled at `--log-hz`: position, height above ground, velocity, orientation (quaternion, roll/pitch/yaw, tilt), angular velocity, joint velocity and tracking-error summary, leg and body contacts, state, terrain type, falls/recoveries/hits so far |
| `summary.json` | `RunMetrics.summary()` (distance, falls, recoveries, hits survived, resets, jog intervals, MTBF, recovery %, falls/km, max force/impulse survived, P(survive T)), every hit with `caused_fall`, the quit reason, terrain, number of manual and auto resets, spawns, and recycled chunks. It's rewritten every 30 s and written again on any exit (Q, window close, Ctrl-C, error) with `complete: true` |

## Robustness evaluation

```bash
# N headless sessions (one per seed), 2 at a time -> <out>/report.md + report.json
.venv/bin/python scripts/eval_robustness.py --seeds 0,1,2,3 --jobs 2 --duration 300 \
    --terrain normal --hit-mode whip --auto-levels 1,2 --auto-reset-after 5 --out runs/robust/normal
# aggregate run folders that already exist
.venv/bin/python scripts/eval_robustness.py --aggregate runs/soak/*/20* --out runs/soak/report
```

Reports mean time to failure, longest run, P(survive T), recovery time/%, hits
survived per level, max impulse survived, falls per km, falls per terrain type,
crashes/instabilities and peak RSS per process.

## Test

```bash
.venv/bin/python -m pytest -q
```

## Layout

```
perpetualfly/
  config.py        dataclass configs (JSON load/save); AppConfig holds all sub-configs
  simulation.py    Simulation: FlyGym sim + controller, hooks, world extensions, NaN checks
  app.py           Session (wires terrain/whip/falls/metrics/logging) + main loop / CLI
  physics_thread.py  physics worker thread for the live window
  rendering.py     offscreen mujoco.Renderer + adaptive smoothed follow camera
  controllers/     fast bit-identical versions of FlyGym's HybridController
  terrain/         flat ground, endless chunked procedural terrain (geom pool)
  interaction/     OpenCV live window, keyboard, external-force shoves, physical whip (whip.py)
  metrics/         locomotion stats, fall detector, run metrics, run logger
  rl/              residual RL env (Gymnasium), wrappers, long-horizon evaluation
scripts/run_sim.py        the app
scripts/demo_*.py         stand-alone demos of terrain / perturbation / falls / whip
scripts/eval_robustness.py  N headless sessions -> long-horizon robustness report
scripts/train_ppo.py, eval_policy.py  residual PPO training / evaluation (docs/RL.md)
tests/
```

## The whip in one paragraph

`perpetualfly/interaction/whip.py`: a mocap "hand" drives a free wooden grip through
a stiff weld; 12 tapered capsules (6 mm, 2 mg, leather brown with a red tip) hang
from it on pairs of stiff, damped hinges. The capsules collide only with the fly. A
crack raises the whip, cocks it beside the fly, swings the handle and **stops** it
so the whip's rest line lies just outside the fly: only the lash (the chain's
overshoot) hits. A post-step hook sums the contact forces on fly geoms into the
measured impulse. On flat ground 192/192 calibration cracks connected; levels 1–4
give ≈ 0.17 / 0.29 / 0.64 / 1.0 uN*s and tip the fly 0 % / 21 % / 75 % / 90 % of
the time. Details, the calibration table and limitations: [docs/WHIP.md](docs/WHIP.md).
Demo: `.venv/bin/python scripts/demo_whip.py` (table), `--calibrate --phases 8`,
`--record whip.mp4 --frames-dir frames/` (slow motion around impacts).

## Notes and known limitations

- **Whip cost:** the whip adds 30 DoFs and a weld. The app always builds it (so H
  works), which costs ~22 % of step time while idle (+26 µs/step in `mj_step`, +4 µs
  of hooks, which are throttled to every 10th step) and more during the ~0.3 s of a
  crack. `AppConfig.whip.enabled = false`
  (in a `--config` file) removes it; shove mode still works.
- **Speed:** the simulation runs slower than real time: the FlyGym timestep is 1e-4 s
  and the controller runs every step. On an Apple-silicon Mac it's about 0.70x real
  time on flat ground and 0.60–0.65x on normal/hard terrain (more contacts), headless
  or windowed. That breaks down to `mj_step` ~85–95 µs, controller ~45 µs, terrain
  features ~15 µs, and hooks (detector, metrics, logger, whip) ~5–8 µs per 0.1 ms
  step. See docs/API_NOTES.md §13.
- **Live display:** the window is an OpenCV window showing frames from MuJoCo's
  offscreen renderer. MuJoCo's own passive viewer isn't used: on macOS it needs `mjpython`,
  and its built-in shortcuts clash with this project's keys (see docs/API_NOTES.md §11).
  There is no mouse camera control; press C to switch views. OpenCV only reports key
  presses, not held keys.
- **What was checked:** the windowed, threaded mode was driven by a script that
  injected every key listed above into the real app loop. It didn't deadlock, and each
  key had the expected effect. Window close, Ctrl-C and an injected NaN each shut
  down cleanly and wrote `summary.json`. All of this ran on one Mac, not by hand.
  The whip was driven the same way: H toggled whip ↔ shove both ways, and cracks from
  every side and level, a queued crack, strength changes, A, X and Q all ran in the
  threaded window loop without deadlock. The run folder of a 60 s headless run
  (`--terrain normal --auto-perturb --hit-mode whip`) had a `whip` row with a measured
  impulse for every automatic crack.
- **Hits are strong on purpose:** level 3 knocks the fly over about 2/3 of the time.
  Level 4 launches it 0.3–0.6 m, far outside the band of terrain features (|y| ≤ 25 mm
  around where the fly was when the chunk loaded). New chunks follow the fly's
  lateral position, but chunks that are already loaded don't move. The camera speeds
  up and zooms out to keep the fly in frame. A fly on its back often can't get up
  with the baseline controller, so press X (or use `--auto-reset-after`).
- Automatic hits keep coming while the fly is down; those hits are logged like any other.
- Distance and average speed include flight after a hit: a level-4 hit adds 0.3–0.6 m.
  The fall/recovery counts, jog intervals and `hits_survived` are the robust metrics.
- The terrain type used for logs and the terminal depends only on x (chunk layout),
  not on y. So a fly that was shoved sideways off the feature band still reports its
  chunk's type. `h` (height above ground) is exact: it's a ray cast against the live
  geoms.
- The flat base plane is moved along under the fly in whole checker periods. This
  doesn't change the physics, because MuJoCo planes are infinite. It just keeps the
  checkerboard in view forever.
- Heading hold (`ControllerConfig.heading_gain`, default 1.5) steers through FlyGym's
  `HybridTurningController`. Set it to 0 for the plain FlyGym `HybridController`, which
  walks straight but settles about 10 degrees to the left of +x.
