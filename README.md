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
| 1 2 3 4 | hit strength: gentle / medium / hard / absurd (default 2), both modes; with `--swatter` also the swat level |
| H | hit mode: whip ↔ shove (default whip; the HUD shows `WHIP ...` or `SHOVE ...`) |
| A | toggle automatic random hits (they use the current hit mode; they wait while the fly is FALLEN / RECOVERING and for 1 s after it recovers, and skipped hits are counted) |
| R | spawn a rock ahead |
| B | spawn a bump ahead |
| S | spawn a slope ahead |
| G | spawn a gap ahead |
| D | spawn a dip ahead |
| F | flatten the next terrain chunk |
| [ / ] | terrain difficulty one step down / up (flat → easy → normal → hard → chaos) at runtime. Chunks from 12–24 mm ahead of the fly onward are regenerated; nothing changes under its feet. The HUD shows the current difficulty, and each change is logged (`terrain_difficulty` event) |
| P | pause / resume |
| X | reset the fly (an explicit reset, counted in the metrics) |
| C | camera: follow / side / top (with `--job`: job view / follow / side / top) |
| I | screenshot: saves the fly frame from the renderer as PNGs (`shotNNN_t<run time>s_fly.png` clean, `..._fly_hud.png` with the HUD), plus a brain-window frame (`..._brain.png`) rendered from the latest brain states with `--brain`. Files go into the run folder, or `runs/screenshots/` with `--no-log`. The screen itself is never captured |
| M | start / stop an MP4 recording of the fly view (`recordingNN_t<run time>s.mp4` in the run folder, or `runs/recordings/`). The video has 30 frames per *simulated* second and plays at real time, like `--record`. While recording, a red `REC` badge shows in the top-right corner of the window |
| J | action: **jump** (escape: crouch, 15 ms mid-leg push, ~3 mm up, ~55 ms airtime, lands upright). Auto hits and fall detection pause during the jump |
| Z | action: **freeze** (stop, hold a stance for 1.5 s, then walk on) |
| Y | action: **groom** (front legs replay a recorded NeuroMechFly grooming bout for 3 s while the mid / hind legs stand) |
| E | action: **back away** (walk backward for 1 s) |
| , / . | action: **turn in place** left / right (1 s). In `--script-keys` write `comma` / `period` |
| W | action: **wing raise** (1.5 s; needs the full body, the default in the CLI, greyed out in the `?` overlay otherwise) |
| N | action: **proboscis extension** (1.5 s; full body) |
| O | brain (`--brain`): looming shadow: LC4 looming detectors → giant fiber (escape) + MDN (backward walking). With `--brain-steer` the fly stops (see below); with `--brain-actions` the giant fiber makes it **jump** |
| T | brain: sugar taste: sugar GRNs → MN9 (proboscis motor neuron). With `--brain-actions` (and the full body) MN9 extends the proboscis |
| K | brain: bitter taste (bitter GRNs). Display only, no body effect |
| V | swatter (`--swatter`): swat at the fly **from behind** at the current strength (1–4 = lazy / normal / quick / lightning). A second V during a swat is queued. The outcome (`HIT` / `GRAZED` / `DODGED` / `MISS`) is printed, shown in the HUD and written to `events.csv` (`swat` rows) |
| Shift+V | swatter: swat from a random direction around the fly. In `--script-keys` write `shift+v`. Any other Shift / Caps-Lock letter acts as the plain key |
| ? | show / hide the on-screen key help (grouped: hits, obstacles / terrain, brain, actions, swatter, view / run; actions the body can't do are greyed out); it is also printed in the terminal |
| Q / ESC | quit (closing the window or Ctrl-C in the terminal also quits) |

Actions (docs/ACTIONS.md) take over the leg targets for a moment and then blend back
into walking; the HUD shows `ACTION <name> [<phase>]`, and every action writes
`action_start` / `action_end` / `action_cancel` rows (with metrics such as jump
height) to `events.csv`. Pressing another action key replaces the running action;
X (reset) cancels it.

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
| `--full-body` / `--no-full-body` | extra wing + proboscis joints so W / N work. **On by default in the CLI** (walking unchanged: 3 s forward displacement within 2 %); the library default (`FlyConfig.extra_joints`) and a `--config` file without it keep FlyGym's legs-only model |
| `--controller {hybrid,cpg}`, `--seed N` | locomotion controller, CPG seed |
| `--camera {follow,side,top}` | initial camera |
| `--render-every N` | physics steps per recorded frame (default 150 = 66.7 fps) |
| `--no-thread` | window mode: step physics on the main thread (slower; debugging) |
| env `PERPETUALFLY_FLY_SCALE=1.5` | fly window display scale. The default is automatic: on a Retina Mac the frame is upscaled ×2, because OpenCV shows one image pixel per physical pixel, and the HUD is drawn after the upscale so it stays sharp. `PERPETUALFLY_BRAIN_SCALE` does the same for the brain window |
| `--print-interval S` | simulated seconds between terminal lines (default 1) |
| `--script-keys '2:left,5:o'` | press keys at the given run times (sim s); works headless too (P is ignored headless) |
| `--brain` | run the connectome brain model next to the fly, plus the brain window (see below) |
| `--brain-headless` | brain without its window (terminal / HUD / logs) |
| `--no-brain-window` | with `--brain`: no brain window |
| `--brain-steer` | the brain's descending neurons modulate the walking controller (implies `--brain`; off by default) |
| `--brain-backup` | lowers the MDN (backward-walking) reference from 40 to 20 Hz, so looming (O) makes the fly walk backward instead of just stopping (implies `--brain-steer`; docs/BRAIN.md) |
| `--brain-actions` | descending neurons trigger body actions: giant fiber (DNp01) > 60 Hz → jump (1.5 s refractory), MN9 > 30 Hz → proboscis extension (full body), DNg12 > 20 Hz for ≥ 100 ms → groom. Implies `--brain-steer` |
| `--no-reflections` | no floor reflections: faster rendering (the draw goes from ~11 to ~6 ms per 960×640 frame on an M1) |
| `--swatter` | a physical flyswatter ([docs/SWATTER.md](docs/SWATTER.md)): V / Shift+V swat, 1–4 strength. The paddle is a looming source on the compound eyes (LC4 / LPLC2); with `--brain-actions` the giant fiber triggers a short-mode escape jump, so slow swats can be dodged. The escape-flight emulation stays off. Hits count as hits in the metrics; every swat is a `swat` row in `events.csv`; `summary.json` gets a `swatter` block. Does not start the brain by itself |
| `--stress` | octopamine pain / arousal layer ([docs/STRESS.md](docs/STRESS.md); a phenomenological model on top of the connectome). Implies `--brain`. Use with `--brain-steer` (walk-DN bursts per hit) and/or `--brain-actions` (lower jump threshold when stressed). HUD `PAIN/AROUSAL` line; `metrics.csv` gains `octopamine`, `oa_rate_hz`, `noci_hz`, `stress_freq_mult`, `stress_amp_mult`, `stress_jump_hz` |
| `--whip-vision` | the fly sees the whip coming: compound-eye looming → LC4 / LPLC2 `loom` stimuli (implies `--brain`; with `--brain-actions` the giant fiber can jump). Shares one looming model with `--swatter`. HUD `EYES` line |
| `--course NAME` | obstacle course instead of endless terrain ([docs/COURSE.md](docs/COURSE.md)): `tutorial`, `gauntlet`, `slalom`, `brain_test` or a `.json` / `.toml` path. Flat base terrain, no app auto reset (the course respawns the fly), `[ ]` and F disabled. Course HUD; the app quits at the finish and prints the lap; results in `<run dir>/course_results.json`, leaderboard in `<runs dir>/leaderboard.json` |
| `--course-loop` | with `--course`: start a new lap after the finish instead of quitting |
| `--job NAME` | eternal job ([docs/JOBS.md](docs/JOBS.md)): `sisyphus`, `hamster_wheel`, `mowing`, `raking`, `kebab`. The job's props are compiled into the world, flat terrain, no auto hits / auto reset (the job recovers the fly itself, also from physics instabilities), no whip (hit keys shove; `--whip-vision` keeps the whip). Job HUD on top, C adds the job camera. Same as `scripts/run_job.py`, but with every app key and flag |
| `--job-config JSON` | with `--job`: job config overrides, e.g. `'{"gap": 1.6}'` |

The features compose: e.g. `--brain-actions --stress --swatter --whip-vision`,
`--job kebab --brain`, `--course gauntlet --stress`. `--course` and `--job` can't be
combined (both replace the world); bad combinations, unknown jobs / courses and bad
`--job-config` exit with a clear `ERROR:` (code 2). With `--swatter` or
`--whip-vision` and a brain, the low-latency brain pacing from docs/SWATTER.md is set
automatically (`brain.window_s = 0.02`, `brain.sync_wait_s = 0.05`; the wait only
happens while a loom is active and can lower the window's frame rate then).

```bash
.venv/bin/python scripts/run_sim.py --brain-actions --swatter --whip-vision --stress   # dodge + pain
.venv/bin/python scripts/run_sim.py --course gauntlet --stress --brain-steer
.venv/bin/python scripts/run_sim.py --job kebab --brain                              # + brain window
```

## Connectome brain (optional)

`--brain` runs the Shiu et al. (2024) leaky integrate-and-fire model of the whole
FlyWire v783 central brain (138,639 neurons, 15 M connections) in its own process,
and opens a second window with its activity (brain map, transmitter activity,
descending neurons, spike raster). Details: [docs/BRAIN.md](docs/BRAIN.md) (model,
checks, integration, steering) and [docs/BRAIN_WINDOW.md](docs/BRAIN_WINDOW.md).

```bash
.venv/bin/python -m pip install -e ".[brain]"      # numba, pyarrow, scipy
.venv/bin/python scripts/fetch_brain_data.py        # ~153 MB into data/brain/ (FlyWire, CC BY-NC 4.0)

.venv/bin/python scripts/run_sim.py --brain                     # fly + brain windows; brain only watches
.venv/bin/python scripts/run_sim.py --brain-steer               # ... and its descending neurons steer the walk
.venv/bin/python scripts/run_sim.py --headless --brain-headless --brain-steer --max-seconds 20 \
    --terrain flat --script-keys "2:left,6:o,12:t"               # scripted, no windows
```

Without the data, `--brain` prints the fetch command and exits (code 2).

* **Body → brain.** Every whip hit and shove becomes a stimulus: intensity = measured
  impulse / the absurd level's impulse (whip 1.0 uN*s, shove 2.5 uN*s), on the side of
  the fly that was hit, with the body that took the impulse (head hits drive head
  bristles). Falls drive body mechanosensors plus Johnston's-organ wind/gravity
  neurons. A fly reset also resets the brain. O / T / K inject looming, sugar and
  bitter. Everything sent is in `events.csv` (`brain_stim`).
* **Time.** The brain follows the fly's *simulated* time: it never runs ahead of it,
  pauses when the fly pauses, and applies each stimulus at the fly time it happened.
  Its latest state is usually 0.02–0.12 s behind the fly (states cover 0.1 s). If it falls further behind
  it lags (HUD `lag`, capped at 1 s); physics is never blocked.
* **Brain → body (`--brain-steer`).** Descending-neuron rates → CPG drive
  (`descending_to_drive`, smoothed with τ = 0.1 s) combined with the heading hold:
  `final = brain + clip(mean(brain), 0, 1) · d_hold · [+1, −1]`. A quiet brain gives
  exactly the normal walk. Measured (flat, headless): pressing O fires the giant
  fiber at 120–150 Hz and MDN at 15–45 Hz; the fly's forward speed falls from 14.6 to
  ~1 mm/s for about a second (it stops, with brief backward steps), then it walks
  on. With `"brain": {"gains": {"backward_ref": 20}}` in a `--config` file it clearly
  walks backward (mean −4 mm/s, peaks −22 mm/s).
* **Brain → actions (`--brain-actions`).** Measured with the real brain (flat,
  headless): O → giant fiber 115–120 Hz in the first 0.1 s state → **jump** ~0.12 s
  after the key (apex +2.8–3.2 mm, lands upright); T → MN9 55 Hz → **proboscis out**
  from 0.1 s after the key until 0.5 s after the sugar stops. DNg12 (grooming) is read
  out, but no stimulus the app can give reaches 20 Hz (head bristles at 200 Hz:
  ~10–13 Hz; JO grooming neurons: 0 Hz), so brain grooming does not fire by itself.
  Details in docs/ACTIONS.md §4 and docs/BRAIN.md.
* **HUD / terminal / logs.** `BRAIN t … lag … x… real time (can x…)`, the drive and
  GF / MDN / MN9 rates. `metrics.csv` gains `brain_*`, `ctrl_drive_L/R`, `dn_*` and
  `mn9_hz` columns; `config.json` has the brain config.
* **Real vs approximated.** Real: the connectome, Shiu et al.'s LIF equations and
  parameters (engine checked spike for spike against Brian2), the FlyWire cell types
  used as inputs and outputs, and the sugar → MN9 pathway (validated by Shiu et al.).
  Approximated: there is no VNC, so body touch reaches the brain only through the few
  ascending mechanosensory afferents; whip hits light up mechanosensory/auditory
  areas but **never reach the walking or turning descending neurons** in this model.
  Looming is injected directly into LC4 (there is no visual input from the rendered
  scene). The DN-to-CPG mapping (gains, the heading-hold blend) is ours, not from a
  paper. The giant fiber's escape (take-off) is not mapped: the body can't fly.
  Taste has no body effect.
* Cost (M1, flat): the brain worker uses up to one CPU core, but it hardly slows the
  fly (headless 0.56x → 0.55x real time). The brain window's 30 fps rendering does:
  windowed 0.54x without the brain, 0.50x with the brain but no brain window, 0.45x
  with both windows. Closing the brain window leaves the fly (and the brain) running.
  Q / ESC / Ctrl-C / closing the fly window stop both child processes; they also exit
  by themselves if the app is killed.

## Output

Terminal line, every simulated second:

```
t=   20.01s  dist=   280.4mm (walked    280.4)  speed now= 14.0 avg= 14.0 mm/s  falls=0 rec=0 hits=6  jog=  20.0s (max   20.0)  [UPRIGHT]  terrain=rocks      h=1.13mm tilt=  4.0deg RTF=0.61
```

`t` is run time (it keeps counting across resets). `dist` is the thorax's total xy
path length, including flights after hits. `walked` counts only the path covered
while the fly was walking: state UPRIGHT or DESTABILIZED, and at least one leg touching
the terrain at every 10 ms check. `avg` is walked distance / run time. `jog` is
the current and longest run without a fall, `[...]` is the fall-detector state
(UPRIGHT / DESTABILIZED / FALLEN / RECOVERING), `terrain` is the terrain under the
thorax, `h` is thorax height above the local ground, `tilt` is the angle of the
thorax up-axis from vertical, and `RTF` is simulated seconds per wall second.
Hits, falls, recoveries, spawns and resets are also printed as they happen.

Run folder `runs/<YYYY-MM-DD_HH-MM-SS>/`:

| file | contents |
|---|---|
| `config.json` | `app` (the full `AppConfig`: fly, terrain, controller, sim, camera, render, stats, perturbation, auto_perturb, falls, logging, session), `procedural_terrain` (the full `ProceduralTerrainConfig`), `runtime` (argv, timestep, fly mass, body weight, pool size) |
| `events.csv` | one row per event: `hit` (shove), `whip` (whip hit: force_direction = commanded side e.g. `from_left`, force_magnitude = mean contact force, details = measured impulse vector / `impulse_uNs`, peak force `magnitude_uN`, contact duration, bodies hit), `whip_miss`, `hit_mode`, fall-detector transitions (`destabilized`, `stabilized`, `fall`, `recovering`, `recovered`, `relapse`, `reset`), `spawn`, `flatten`, `strength`, `auto_perturb`, `auto_hit_skipped`, `terrain_difficulty`, `screenshot`, `record_start` / `record_stop`, `manual_reset`, `auto_reset`. Columns: timestamp (run time), sim_time, wall_time, event_type, force_direction, force_magnitude (µN), terrain_type, fall_detected, recovered, state, x, y, z, details (JSON) |
| `metrics.csv` | sampled at `--log-hz`: position, height above ground, velocity, orientation (quaternion, roll/pitch/yaw, tilt), angular velocity, joint velocity and tracking-error summary, leg and body contacts, state, terrain type, falls/recoveries/hits so far |
| `summary.json` | `RunMetrics.summary()` (`distance_mm` = total path incl. flights, `walked_distance_mm` / `walking_time_s` / `walking_speed_mm_s`, `average_speed_mm_s` = walked / run time, `average_speed_total_mm_s` = the old total-path average, falls, recoveries, hits survived, resets, jog intervals, MTBF, recovery %, falls/km, max force/impulse survived, P(survive T)), every hit with `caused_fall`, the quit reason, terrain, number of manual and auto resets, spawns, and recycled chunks, `n_auto_hits_skipped`, the start and end terrain difficulty, and the screenshots / recordings written (`media`). It's rewritten every 30 s and written again on any exit (Q, window close, Ctrl-C, error) with `complete: true` |

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
  actions/         action library: jump, freeze, groom, back away, turn, wings, proboscis
                   (ActionManager, brain triggers, full-body fly; docs/ACTIONS.md)
  stress.py        octopamine pain / arousal body effects (--stress; docs/STRESS.md)
  vision/          compound-eye looming -> LC4 / LPLC2 (--whip-vision, the swatter)
  course/          obstacle courses (--course; docs/COURSE.md)
  jobs/            eternal jobs (--job, scripts/run_job.py; docs/JOBS.md)
scripts/run_sim.py        the app
scripts/demo_*.py         stand-alone demos of terrain / perturbation / falls / whip / actions
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
  presses, not held keys. On a Retina display the ×2 upscale costs the main thread
  about 10 ms per frame (resize ~3 ms, then `imshow` of 4× the pixels). The window then
  shows ~24 fps instead of ~30. The physics thread isn't affected: RTF was 0.48–0.49
  vs 0.50–0.52 at ×1. `PERPETUALFLY_FLY_SCALE=1` restores the old size.
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
- Automatic hits wait while the fly is down (FALLEN / RECOVERING) and for
  `session.auto_perturb_resume_after_s` (1 s) after it recovers. A hit that comes due
  in that time is skipped: it is printed, logged as `auto_hit_skipped`, and counted in
  `n_auto_hits_skipped`. Set `session.auto_perturb_pause_when_down = false` for the old
  behaviour.
- `dist` / `distance_mm` still include flight after a hit (a level-4 hit adds 0.3–0.6 m).
  `walked_distance_mm` and the average speed don't. The fall/recovery counts, jog
  intervals and `hits_survived` are still the most robust metrics.
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
