# Usage: running the app, keys, flags and output

The full reference for `scripts/run_sim.py` (the app; also installed as the
`fly-simulator` console script). The [README](../README.md) has the short version.
Internals, the code layout and known limitations are in
[ARCHITECTURE.md](ARCHITECTURE.md).

NeuroMechFly walks with FlyGym's hybrid CPG controller (plus a heading hold along +x)
over chunked, recycled procedural terrain (rocks, bumps, rough ground, blocks, slopes,
gaps, dips). You can spawn obstacles and hit the fly from the keyboard, or let an
auto-perturber do it. By default hits are cracks of a **physical whip**: a visible
chain of capsules on a scripted handle whose lash strikes the fly through real MuJoCo
contact (the fly moves only through the contact forces). H switches to the older
**shove** mode (a constant external force on the thorax; strengths in
[dev/PERTURBATION_CALIBRATION.md](dev/PERTURBATION_CALIBRATION.md)). A fall detector
tracks falls and recoveries. Every run is logged to `runs/<timestamp>/`. There is no
episode timeout.

## Install

Tested on macOS (Apple Silicon) with Python 3.12. FlyGym 2.1.0 needs Python >= 3.12
and pins `mujoco>=3.9,<3.10`.

```bash
cd fly-simulator
uv venv --python 3.12 .venv                    # or: python3.12 -m venv .venv
uv pip install --python .venv/bin/python -e ".[dev,brain]"
# without uv:  .venv/bin/pip install -e ".[dev,brain]"
.venv/bin/python scripts/fetch_brain_data.py   # ~153 MB FlyWire data into data/brain/ (CC BY-NC 4.0)
```

This installs `flygym==2.1.0` (which includes the `flygym_demo` locomotion
controllers and `imageio`, used for `--record`), `mujoco`, `opencv-python`, `pytest`
and the brain engine's `numba`, `pyarrow` and `scipy`. Other extras: `vision`
(`--real-vision`, flyvis; then `.venv/bin/flyvis download-pretrained --skip_large_files`),
`rl` (PPO, [RL.md](RL.md)) and `brain-ref` (Brian2 cross-check).

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
`fly-simulator` console script does the same thing.

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
| 5 / 6 / 7 | taste patches (`--taste-patches`): spawn a sugar / bitter / mixed spot 3 mm ahead ([TASTE.md](TASTE.md)) |
| 8 / = | odour zones (`--odor-zones` / `--learning`): an odour A (magenta) / B (teal) haze zone around the fly ([FEAR_LEARNING.md](FEAR_LEARNING.md)) |
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
| 9 / 0 / - | brain playground ([PLAYGROUND.md](PLAYGROUND.md)): **9** selects the next palette target (GF, MDN, BDN2, DNa02 L/R, …; shown in the HUD), **0** stimulates it (optogenetic, 120 Hz for 1 s), **-** lesions / un-lesions it |
| V | swatter (`--swatter`): swat at the fly **from behind** at the current strength (1–4 = lazy / normal / quick / lightning). A second V during a swat is queued. The outcome (`HIT` / `GRAZED` / `DODGED` / `MISS`) is printed, shown in the HUD and written to `events.csv` (`swat` rows) |
| L | flight (`--flight`): **take off** (jump → flapping wings → hover ~2 mm up; lands by itself after 4 s of hovering) / **land**. While flying, the arrows **steer** instead of cracking the whip: UP / DOWN forward speed ±50 mm/s, LEFT / RIGHT turn 30°. HUD `FLIGHT` line (state, wingbeat Hz, altitude, speed) |
| Shift+V | swatter: swat from a random direction around the fly. In `--script-keys` write `shift+v`. Any other Shift / Caps-Lock letter acts as the plain key |
| ? | show / hide the on-screen key help (grouped: hits, obstacles / terrain, brain, actions, flight, swatter, view / run; actions the body can't do are greyed out); it is also printed in the terminal |
| TAB | show the stats / controls box: off by default, then full, then compact (jobs and courses: on / off; games: brain side panel on / off, start with it via `--panel`) |
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
| `--config file.json` | load any fields of `fly_simulator.config.AppConfig` (same layout as `config.json["app"]` in a run folder) |
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
| env `FLY_SIMULATOR_FLY_SCALE=1.5` | fly window display scale. The default is automatic: on a Retina Mac the frame is upscaled ×2, because OpenCV shows one image pixel per physical pixel, and the HUD is drawn after the upscale so it stays sharp. `FLY_SIMULATOR_BRAIN_SCALE` does the same for the brain window |
| `--print-interval S` | simulated seconds between terminal lines (default 1) |
| `--script-keys '2:left,5:o'` | press keys at the given run times (sim s); works headless too (P is ignored headless) |
| `--brain` | run the connectome brain model next to the fly, plus the brain window (see below) |
| `--brain-headless` | brain without its window (terminal / HUD / logs) |
| `--stim SPECS` | brain playground ([PLAYGROUND.md](PLAYGROUND.md); implies `--brain`): scripted optogenetic stimulation, `TARGET[:RATE_HZ[:DURATION_S]][@RUN_TIME_S]`, comma separated, e.g. `"DNa02_L:120:1.0@3,MDN@6"` |
| `--lesion TARGETS` | brain playground (implies `--brain`): silence targets from the start (`"DNp01,MDN"`) or from a run time (`"MDN@4"`) |
| `--no-playground` | classic brain window panels (spike raster + transmitters) instead of the playground palette + decision meters |
| `--no-brain-window` | with `--brain`: no brain window |
| `--brain-steer` | the brain's descending neurons modulate the walking controller (implies `--brain`; off by default) |
| `--brain-backup` | lowers the MDN (backward-walking) reference from 40 to 20 Hz, so looming (O) makes the fly walk backward instead of just stopping (implies `--brain-steer`; docs/BRAIN.md) |
| `--brain-actions` | descending neurons trigger body actions: giant fiber (DNp01) > 60 Hz → jump (1.5 s refractory), MN9 > 30 Hz → proboscis extension (full body), DNg12 > 20 Hz for ≥ 100 ms → groom. Implies `--brain-steer` |
| `--no-reflections` | no floor reflections: faster rendering (the draw goes from ~11 to ~6 ms per 960×640 frame on an M1) |
| `--swatter` | a physical flyswatter ([SWATTER.md](SWATTER.md)): V / Shift+V swat, 1–4 strength. The paddle is a looming source on the compound eyes (LC4 / LPLC2); with `--brain-actions` the giant fiber triggers a short-mode escape jump, so slow swats can be dodged. The escape-flight emulation stays off. Hits count as hits in the metrics; every swat is a `swat` row in `events.csv`; `summary.json` gets a `swatter` block. Does not start the brain by itself |
| `--stress` | octopamine pain / arousal layer ([STRESS.md](STRESS.md); a phenomenological model on top of the connectome). Implies `--brain`. Use with `--brain-steer` (walk-DN bursts per hit) and/or `--brain-actions` (lower jump threshold when stressed). HUD `PAIN/AROUSAL` line; `metrics.csv` gains `octopamine`, `oa_rate_hz`, `noci_hz`, `stress_freq_mult`, `stress_amp_mult`, `stress_jump_hz` |
| `--habituation` | looming habituation ([HABITUATION.md](HABITUATION.md); phenomenological short-term depression of the LC4 / LPLC2 → giant fibre synapses in the brain engine). Implies `--brain`. With `--brain-actions`, repeated harmless looms (O every 2 s) stop triggering jumps after ~3 trials and recover after 10–30 s of rest. HUD `HABITUATION` line, `habituation` in `BrainState` / `summary.json` |
| `--brain-record` | record brain states + stimuli / brain actions to `<run dir>/brain_rec/` (compressed chunks, ~0.5–1 MB per minute, capped at 50 MB; implies `--brain`). Replay / export: `scripts/brain_replay.py RUN_DIR` ([BRAIN_REPLAY.md](BRAIN_REPLAY.md)) |
| `--whip-vision` | the fly sees the whip coming: compound-eye looming → LC4 / LPLC2 `loom` stimuli (implies `--brain`; with `--brain-actions` the giant fiber can jump). Shares one looming model with `--swatter`. HUD `EYES` line |
| `--real-vision` | the fly actually sees ([VISION.md](VISION.md)): compound eyes → flyvis connectome visual system → LC4 / LPLC2 (implies `--brain`; replaces `--whip-vision` and the swatter's geometric looming sense). Needs the `vision` extra; slows the simulation (~25 ms wall per 10 ms of sim time) |
| `--flight` | real flapping-wing flight ([FLIGHT.md](FLIGHT.md) §7): the flight fly (stroke-plane wing hinges + MuJoCo fluid model, same leg actuators) at dt 5e-5 s. It walks as usual; L takes off / lands, the arrows steer while flying. With `--brain-actions` the giant-fibre escape jump starts the wings at take-off and the fly flies away from the threat (the swatter paddle) and lands ~1.3 s later: no external force, only wing aerodynamics. About half the RTF of the normal app. Not with `--course` / `--job`; `--full-body` is ignored (W / N greyed out) |
| `--taste-patches` | sugar / bitter / mixed spots on the ground tasted with the legs ([TASTE.md](TASTE.md)): taste → sugar / bitter GRNs (labellar stand-in); with `--brain-actions` the real MN9 makes the fly stop and feed (proboscis). Keys 5 / 6 / 7 spawn a spot ahead; `--taste-density PER_CM` sets the procedural density (default 0.3 per 10 mm, 0 = only spawned). HUD `TASTE` line, `taste` block in `summary.json`. Not with `--course` / `--job` |
| `--odor-zones` | odour A / B haze zones on the path ([FEAR_LEARNING.md](FEAR_LEARNING.md)); implies `--brain`. Inside a zone the brain gets that odour's Kenyon-cell code (10 % of KCs picked from the connectome's PN → KC synapses; ORN / PN input would ignite the model's runaway state). Keys 8 / = spawn a zone. HUD `ODOUR` line, `odor` block in `summary.json`. Not with `--course` / `--job` / `--flight` |
| `--learning` | fear learning (implies `--odor-zones`): a whip hit inside a zone drives the PPL1 punishment DANs (stand-in: the modelled whip afferents don't reach PPL1), and dopamine-gated depression of KC → MBON synapses makes the brain respond less to that odour (model rule, compartments from the connectome). With `--brain-steer` the odour's walk / left-turn DN drive disappears after conditioning; it is not active avoidance. HUD `LEARNING` line, `learning` in `BrainState` / `summary.json` |
| `--course NAME` | obstacle course instead of endless terrain ([COURSE.md](COURSE.md)): `tutorial`, `gauntlet`, `slalom`, `brain_test` or a `.json` / `.toml` path. Flat base terrain, no app auto reset (the course respawns the fly), `[ ]` and F disabled. Course HUD; the app quits at the finish and prints the lap; results in `<run dir>/course_results.json`, leaderboard in `<runs dir>/leaderboard.json` |
| `--course-loop` | with `--course`: start a new lap after the finish instead of quitting |
| `--job NAME` | eternal job ([JOBS.md](JOBS.md)): `sisyphus`, `hamster_wheel`, `mowing`, `raking`, `kebab`. The job's props are compiled into the world, flat terrain, no auto hits / auto reset (the job recovers the fly itself, also from physics instabilities), no whip (hit keys shove; `--whip-vision` keeps the whip). Job HUD on top, C adds the job camera. Same as `scripts/run_job.py`, but with every app key and flag |
| `--job-config JSON` | with `--job`: job config overrides, e.g. `'{"gap": 1.6}'` |
| `--game NAME` | play a brain game ([GAMES.md](GAMES.md)) instead of the simulator: `asteroids`, `chase`, `rings`, `pong`, `canyon`. The same runner as `scripts/play.py`: needs `--brain` (FlyWire) or `--synthetic-brain` (tests); opens the game window unless `--headless` (then `--max-seconds` is game time, in the window it is wall time); `--record`, `--seed`, `--script-keys` pass through. Game options: `--control brain\|mirror\|none`, `--difficulty easy\|normal\|hard`, `--lives N`, `--panel`, `--experiment N` (headless paired trials) with `--controls` / `--json`. Game keys: SPACE pause, R restart, 1/2/3 difficulty, B brain window, TAB panel, M record, Q quit. Any simulator option (`--job`, `--course`, `--flight`, `--terrain`, `--swatter`, `--config`, ...) is refused with `ERROR:` (code 2); the game options without `--game` are refused too. For the game-specific options (`--win-points`, `--air-start`, `--frames`, ...) use `scripts/play.py` |

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
.venv/bin/python scripts/run_sim.py --flight --brain-actions --swatter                # escape by real flight
.venv/bin/python scripts/run_sim.py --game canyon --brain                             # a brain game (window)
.venv/bin/python scripts/run_sim.py --game pong --synthetic-brain --headless --max-seconds 3
```

### Flight (`--flight`)

The fly can fly for real: its wings flap (218 Hz, 160° stroke) and all lift, thrust
and steering come from MuJoCo's fluid model acting on the wings; a PID "haltere +
visual" controller sets the wingbeat. Press **L** to take off (jump → wings → hover),
steer with the arrows while airborne, L again to land (it lands by itself after 4 s of
hovering). With `--brain-actions --swatter`, a swat seen coming makes the giant fibre
fire: short-mode jump → wings at take-off → 200 mm/s escape flight away from the
paddle (low, 2 mm above the ground) → landing and walking ~1.3 s later. Real brain,
L1–L2 swats from behind / the left: 6/12 dodged with flight vs 5/12 with the short
jump alone, and **2/12 falls vs 9/12** (the short jump tumbles; the flights land
upright). Walking speed is unchanged at dt 5e-5, but the app runs at about half its
normal RTF. Details and limits: [FLIGHT.md](FLIGHT.md) §7.

## Connectome brain (optional)

`--brain` runs the Shiu et al. (2024) leaky integrate-and-fire model of the whole
FlyWire v783 central brain (138,639 neurons, 15 M connections) in its own process,
and opens a second window with its activity (brain map, transmitter activity,
descending neurons, spike raster). Details: [BRAIN.md](BRAIN.md) (model,
checks, integration, steering) and [BRAIN_WINDOW.md](BRAIN_WINDOW.md).

```bash
uv pip install --python .venv/bin/python -e ".[brain]"   # numba, pyarrow, scipy
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
  (`--stress` adds a labelled "VNC stand-in" relay for that, docs/STRESS.md.) The O
  key injects looming directly into LC4; the swatter and `--whip-vision` use a
  geometric looming sense, and only `--real-vision` computes it from the rendered
  scene (docs/VISION.md). The DN-to-CPG mapping (gains, the heading-hold blend) and
  the action thresholds are ours, not from a paper. Escape flight needs `--flight`
  (docs/FLIGHT.md); taste feeding uses labellar GRNs as a stand-in (docs/TASTE.md).
* Cost (M1, flat): the brain worker uses up to one CPU core, but it hardly slows the
  fly (headless 0.56x → 0.55x real time). The brain window's 30 fps rendering does:
  windowed 0.54x without the brain, 0.50x with the brain but no brain window, 0.45x
  with both windows. Closing the brain window leaves the fly (and the brain) running.
  Q / ESC / Ctrl-C / closing the fly window stop both child processes; they also exit
  by themselves if the app is killed.

### Brain playground: optogenetics, lesions, decision meters

Details and measured effects: [PLAYGROUND.md](PLAYGROUND.md). The brain
window's lower-left panel is a clickable palette of 16 targets (GF DNp01, MDN, BDN2,
P9, DNa02 L/R, DNg12, MN9, sugar, LC4, OA-VUMa1, …):
* left-click stimulates the target at the chosen rate / duration (optogenetic
  stimulation: direct, not a natural sense);
* right-click (or its LES box) lesions it, i.e. silences it: the neurons can no
  longer fire;
* clicking a region on the brain map stimulates that neuropil (right-click lesions it).
The top-right panel shows **decision meters**: escape, back up, walk faster, turn,
groom, feed (and arousal with `--stress`) against the thresholds the body uses. It
shows what the brain is leaning toward before the fly acts. Key P in the brain
window switches to the classic panels.

```bash
.venv/bin/python scripts/run_sim.py --brain-actions                         # click away
.venv/bin/python scripts/run_sim.py --brain-steer --stim "DNa02_L:120:1.0@3,MDN@6"
.venv/bin/python scripts/run_sim.py --brain-actions --lesion DNp01 --script-keys "3:o"   # no jump
```

Measured with the real brain:
* DNa02 L at 120 Hz turns the fly left (+19.5° in 1 s);
* MDN backs it up (−5.0 mm/s mean);
* BDN2 speeds it up by 10 %;
* GF makes it jump;
* lesioning GF abolishes the looming jump;
* lesioning MDN abolishes backing up under `--brain-backup`;
* lesioning DNa02 roughly halves the turn away from a one-eyed loom (DNa01 + DNa02
  lesioned: no turn).

`events.csv` gets `brain_opto` / `brain_lesion` rows. Lesions persist across resets.

### Taste patches (`--taste-patches`)

Details and measurements: [TASTE.md](TASTE.md).

* **Patches.** Sugar (pale yellow, shiny), bitter (dark olive) and mixed (amber)
  spots are placed procedurally along the fly's path. Set the density with
  `--taste-density`; keys 5 / 6 / 7 spawn one ahead. They are visual only: walking
  is bit-identical with and without them.
* **Tasting.** Legs standing on a spot taste it. While the contact lasts, the brain
  gets sugar / bitter GRN input: 67 / 133 / 200 Hz for 1 / 2 / 3+ legs.
* **Which neurons.** Leg taste neurons mostly end in the VNC, which the model lacks,
  and the model's own ascending leg gustatory afferents reach no MN9. So the
  labellar sugar / bitter GRNs of Shiu et al. are used as a **stand-in**. The HUD
  says so.
* **Feeding.** With `--brain-actions`, sugar drives the real MN9 to 60–90 Hz. Our
  rule stops the fly and extends the proboscis while MN9 stays above 30 Hz, for up
  to 4 s. Then it walks on.
* **Bitter.** Measured: bitter drives no avoidance DN (no MDN, no escape; at most
  5–7 Hz on the walk DNs), so bitter has no body effect. Mixed into sugar it
  suppresses MN9 (to about 4–10 Hz), so a mixed spot gives no feeding.

```bash
.venv/bin/python scripts/run_sim.py --taste-patches --brain-actions
```

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