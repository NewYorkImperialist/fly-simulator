# Fly Simulator: status (2026-09-26, after the integration pass)

## What works

| milestone | status |
|---|---|
| M1 walking fly | NeuroMechFly (FlyGym 2.1 / MuJoCo 3.9) walks forever with FlyGym's hybrid CPG controller plus a heading hold along +x. There is a smooth follow camera (C: follow / side / top), a live OpenCV window (physics runs in a worker thread) and a `--headless` mode. |
| M2 hits | Default: a **physical whip** whose chain of capsules hits the fly only through MuJoCo contact, with the impulse measured. H switches to **shove** (an `xfrc_applied` force on the thorax). Arrow keys, SPACE and U choose the direction; 1-4 set strength (calibrated, see docs/WHIP.md and docs/PERTURBATION_CALIBRATION.md). |
| M3 terrain | Endless chunked procedural terrain from a fixed geom pool (constant model size; recycled, and 2173 chunks were recycled in the 31 min soak). Presets flat / easy / normal / hard / chaos. R B S G D spawn obstacles, F flattens the next chunk. |
| M4 falls, logging, auto hits | The FallDetector tracks UPRIGHT / DESTABILIZED / FALLEN / RECOVERING. X resets, `--auto-reset-after` resets automatically, `--auto-perturb` gives seeded random hits. Every run writes `runs/<ts>/{config.json, events.csv, metrics.csv, summary.json}`. |
| RL (phase 10) | A residual Gymnasium env, a curriculum, `scripts/train_ppo.py` and `scripts/eval_policy.py` exist and have been smoke-tested. **No real training run has been done yet** (docs/RL.md). |
| connectome brain | `--brain` runs the Shiu et al. 2024 LIF model of the whole FlyWire v783 brain (138,639 neurons) in its own process, paced to the fly's *simulated* time (lag ~0.1 s), plus a brain window. Whip hits, shoves, falls and resets become sensory input; O (looming), T (sugar), K (bitter) inject stimuli. `--brain-steer` feeds the descending neurons back into the walking controller: looming (giant fiber 120-150 Hz, MDN 15-45 Hz) stops the fly for ~1 s. Whip hits never reach the walking DNs in this model. See docs/BRAIN.md, *Integration*. |
| quick wins (ROADMAP A) | `--brain-backup` (looming makes the fly walk backward), automatic Retina ×2 fly window, `--no-reflections`, `[` / `]` live terrain difficulty, `I` screenshot (fly + brain PNG, renderer frames only), `M` live MP4 with a REC badge, `?` on-screen help, auto hits paused while the fly is down, walked distance separate from total path, brain atlas in the wheel. See "Quick wins" below. |
| actions | `fly_simulator/actions/`: jump (J), freeze (Z), groom (Y, a recorded NeuroMechFly grooming bout), back away (E), turn in place (, .), wing raise (W), proboscis extension (N). They blend back into walking, are logged to events.csv and shown in the HUD. Auto hits and fall counting pause during a jump. `--full-body` (default in the CLI) adds wing and proboscis joints with walking unchanged. `--brain-actions`: giant fiber → jump (0.12 s after O), MN9 → proboscis (T), DNg12 → groom (not reached by natural input in this model). The unboosted jump is +3.0 ± 0.2 mm and lands upright in 16/16 gait phases. See docs/ACTIONS.md. |
| integration pass (2026-09-26) | The standalone features are now app flags that compose: `--swatter` (V / Shift+V), `--stress`, `--whip-vision`, `--course NAME` (+ `--course-loop`), `--job NAME` (+ `--job-config`). Session wiring, HUD lines, `?` overlay (new "swatter" group), events / metrics / summary logging, refused combos. See "Integration pass" below. |
| real flight in the app (2026-09-26) | `--flight`: the flight fly (flapping wings, MuJoCo fluid model, dt 5e-5) walks as usual; **L** take off / hover / land, arrows steer while airborne, HUD `FLIGHT` line. `--flight --brain-actions`: giant fibre → short-mode jump → wings at take-off → escape flight away from the swatter → landing → walking, with no external force. Real brain, L1–L2 swats (rear + left, 12): 6 dodged / 4 grazed (≤1.6 µN·s) / 2 hits, 2 falls, 10/12 upright landings, vs the short jump alone: 5 / 4 / 3, 9 falls. See docs/FLIGHT.md §7. |
| robustness eval | `scripts/eval_robustness.py` runs N headless sessions in parallel, or aggregates existing run folders, and writes `report.md` and `report.json` with the spec's long-horizon metrics. |

Tests: `.venv/bin/python -m pytest -q`: 262 passed, ~5.5 min (15 in `tests/test_integration.py`).

## How to run

```bash
.venv/bin/python scripts/run_sim.py                                   # window, normal terrain
.venv/bin/python scripts/run_sim.py --terrain hard --auto-perturb     # + random whip cracks
.venv/bin/python scripts/run_sim.py --headless --auto-perturb --auto-reset-after 5 \
    --max-seconds 600 --log-hz 10                                     # unattended
.venv/bin/python scripts/eval_robustness.py --seeds 0,1,2 --jobs 3 --duration 300 \
    --terrain normal --hit-mode whip --auto-levels 1,2 --out runs/robust/normal
.venv/bin/python scripts/eval_policy.py --baseline --stage normal --sim-seconds 180
.venv/bin/python scripts/run_sim.py --brain-steer                     # + connectome brain and its window
.venv/bin/python scripts/run_sim.py --brain-actions                   # + GF -> jump, MN9 -> proboscis, DNg12 -> groom
.venv/bin/python scripts/demo_actions.py --frames /tmp/frames          # every action headless, metrics + key frames
```

## Flight in the app (2026-09-26)

`--flight` (fly_simulator/flight/mode.py `FlightMode`, `FlightModeConfig` = `AppConfig.flight`),
key L, arrows steer while flying, brain escape flight through
`BrainActionTriggers.flight`. Evidence and numbers: docs/FLIGHT.md §7. Summary:

* walking at dt 5e-5: 14.14 vs 14.08 mm/s, straight; RTF halves (bare sim 0.33 vs 0.70,
  app ~0.25–0.4, brain 0.2–0.37, real vision ~0.12);
* manual L cycle: take-off (long jump, tilt 3°) → hover → auto landing (tilt 1.8°) →
  walking; combined real-brain run (`--flight --brain-actions --stress --swatter`,
  normal terrain): 10 s steered flight, swat while flying → GF → escape re-directed →
  dodged → upright landing;
* swatter, real brain, geometric looming (12 swats): flight 6/12 dodged, 2 falls vs
  short jump 5/12, 9 falls; `--real-vision`: 1/8 dodged (the eyes fire after the plate
  lands);
* refused: `--flight` + `--course` / `--job`. `--full-body` ignored in flight mode.
* found and fixed on the way: a crash switched the wing servos off at full wing speed
  → NaN (now always a fade); a 180° turn into a backward escape crashed (now flown
  backward); take-off from a tumbling / upside-down fly (now refused / no wings).

## Integration pass (2026-09-26)

Flags (all in `fly_simulator/app.py`; config blocks `swatter`, `stress`, `whip_vision`,
`course`, `job` in `AppConfig`, so `--config` files work too):

| flag | wiring | evidence (headless unless noted) |
|---|---|---|
| `--swatter` | `Swatter.extension` compiled in, `install_swatter` (vision source on the shared `LoomingVision`, short-mode escape with `--brain-actions`, flight emulation **off**). V rear / Shift+V random, 1-4 (and `--strength`) set the level. `swat` rows in events.csv, hits in RunMetrics, auto hits wait during a swat, HUD `SWATTER` line, `summary.swatter` | L2 swats without a brain: 2/2 hits (154 / 99 uN·s). Renderer frame shows the red paddle flat on the fly |
| `--stress` | implies `--brain`; `install_stress` after the brain link; restored in `Session.close` before `brain.close`; HUD `PAIN/AROUSAL`, 6 metrics.csv columns, `summary.stress` | see the combined run |
| `--whip-vision` | implies `--brain`; `install_whip_vision` with the brain link as sink; the swatter adds its paddle to the same instance | L4 cracks: GF 100-150 Hz → brain jump at contact (L2 cracks stay below the whip's 12 000 deg/s response threshold: 0 loom events) |
| brain pacing | `window_s 0.02`, `sync_wait_s 0.05` when the swatter or whip vision run with a brain (only if still at the defaults) | config.json of the combined run |
| `--course NAME` / `--course-loop` | `install_course` in the Session; flat base terrain, auto reset off; the app quits at the finish (`quit_reason "course finished"`) and prints the lap | gauntlet finished in 10.32 s (0 falls, 3 whip hits); `--course gauntlet --stress --brain-steer`: 8.80 s, octopamine 0.35 |
| `--job NAME` / `--job-config` | `make_job` + `configure_app` before the world is built, props via `world_extensions`, `install_job`; `Session.ground_height` follows the job; `JobCamera` in the C cycle; job HUD on top; no whip (hits shove); instabilities recovered by `job.recover` (also in the physics thread via `PhysicsThread(step_fn=...)`) | kebab 8 s: 48 shavings; sisyphus 10 s: 2 summits, 0 falls; `kebab --brain`: 61 shavings, MN9 40-60 Hz on each food pulse |

Combined run, real brain: `run_sim.py --headless --brain-actions --stress --swatter
--whip-vision --max-seconds 20 --script-keys "3:v,6:space,9:o"` (normal terrain):
the swat at 3 s was **DODGED** (GF 100 Hz at 3.76 s → short-mode jump; the fly 7.7 mm
from the aim point when the plate landed), but the jump tumbled and the fly fell at
4.18 s (short-mode jumps without flight tumble, docs/SWATTER.md). SPACE at 6 s: whip
**HIT** (338 nN·s). The octopamine level rose 0 → 0.20 (9 s) → **0.64** after O and
decayed to 0.48 at 20 s; the jump threshold followed (60 → 41 Hz). 5 brain-triggered
jumps. 999 brain states, 0 dropped, final lag 0.03 s.

Window (threaded, keys injected into `LiveViewer.poll_keys`, `--brain-actions
--no-brain-window --swatter --stress`): V, Shift+V (queued), SPACE, O, ?, ?, C, Q all
handled without deadlock; frames checked (HUD lines, `?` overlay with the swatter group
fits at ×1). `--job kebab` window: C job → follow, Q. `scripts/run_job.py --brain` now
opens the brain window when not `--headless` (it always passed `headless=True`
before); checked with a 1.5 s windowed kebab run ("window on", both processes exit).

Small changes outside app/config: `decode_key(code, shift_names=False)` and
`LiveViewer.shift_names` (Shift+V; the app turns it on), `PhysicsThread(step_fn=...)`,
`scripts/run_job.py` (brain window, `--no-brain-window`).

## Quick wins (ROADMAP section A, 2026-09-25)

| item | what changed | evidence |
|---|---|---|
| A1 `--brain-backup` | `BrainLinkConfig.backup` / `backup_ref_hz` (20 Hz) override `DriveGains.backward_ref`; the flag implies `--brain-steer` | Headless, flat, O at 3 s (real brain): 0.5-1.2 s after O, vx mean **-4.2 mm/s, min -21.0**, net -1.8 mm, control drive -0.43. Without the flag: +2.3 mm/s, drive +0.13 (table in docs/BRAIN.md) |
| A2 Retina | `fly_simulator/display.py` (CoreGraphics backing scale, shared with the brain window). `LiveViewer` upscales ×2 and draws the HUD after the upscale. Env `FLY_SIMULATOR_FLY_SCALE` / `RenderConfig.display_scale` override it | Window frames saved from `LiveViewer.last_shown` are 1920×1280 with sharp HUD text. The window runs ~24 fps (was ~30) and physics RTF is unchanged (0.48-0.49 vs 0.50-0.52) |
| A3 reflections | `RenderConfig.reflections`, `--no-reflections` (the `mjRND_REFLECTION` scene flag) | Draw at 960×640: 10.6-11.2 ms → 5.5-6.4 ms |
| A4 `[` / `]` | `ProceduralTerrain.set_difficulty(name)` regenerates loaded chunks ≥ 2 ahead (12-24 mm), with no model rebuild; the RL env can use it too. Printed, shown in the HUD, logged as `terrain_difficulty` | Test: regenerated chunks match a fresh generator of the new preset, and nearer chunks are untouched |
| A5 `I` | `fly_simulator/media.py`: `shotNNN_t<rt>s_fly.png` (clean), `_fly_hud.png`, and with a brain `_brain.png` rendered in-process from the last 30 BrainStates (~40-60 ms). Saved to the run dir or `runs/screenshots/` | Real-brain run: GF 135 Hz / MDN / MANUAL LC4 visible in the brain PNG. Window and headless runs checked |
| A6 `M` | MP4 at 30 frames per *sim* second (real-time playback, like `--record`). The live window writes its latest displayed frame when one is due, on the main thread outside the lock. Red `REC <s>` badge | `append_data` 0.5-0.6 ms/frame. 1.5 s of sim → 45 frames. Physics RTF not affected |
| A7 `?` | Grouped overlay (hits, obstacles / terrain, brain, view / run); still printed in the terminal | Overlay checked at ×1 and ×2 |
| A8 auto hits | `AutoPerturber.gate` + `skip_listeners`. Hits wait while FALLEN / RECOVERING and for `session.auto_perturb_resume_after_s` (1 s) after "recovered". `auto_hit_skipped` events, `n_auto_hits_skipped` in summary.json | 20 s shove L3 run: 2 hits skipped while fallen |
| A9 distance | `RunMetrics.walked_distance` counts only 0.25 s path segments with UPRIGHT / DESTABILIZED + ≥ 1 leg in contact. `average_speed` = walked / run time; the old one is `average_speed_total`. New summary fields; eval_robustness, rl/evaluation and demo_falls updated | Same 20 s run: path 208 mm, walked 140.8 mm, walking speed 12.9 mm/s |
| A10 package data | `[tool.setuptools.package-data]` for `brain_viz/assets/*.json` | `uv build --wheel`: the wheel contains `flywire_neuropils_frontal.json` (wheel then deleted) |

## Soak / robustness results (Apple M1, 3-4 processes at once, `--log-hz 5`)

All of these runs used `--auto-reset-after 5`. Auto hits came every 2-5 sim s. Reports
are in `runs/soak/report_*/report.md`, `runs/soak/chaos/report.md` and
`runs/soak/hard_noperturb/report.md`. RSS was sampled with `ps` every 60 s
(`runs/soak/*.rss`).

| run | sim time (wall) | crashes / NaN / instab. | RSS start -> end | falls (auto resets) | hits survived | mean time to fall | longest run | P(survive 60 s) | falls / km (fwd) |
|---|---|---|---|---|---|---|---|---|---|
| normal, whip L1-2 | **1860 s** (64 min) | 0 / 0 / 0 | 350.0 -> 352.1 MB | 15 (15) | 523 / 538 (97 %) | 124 s | 354 s | 0.50 | 608 |
| normal, shove L1-2 | 1260 s (42 min) | 0 / 0 / 0 | 348.4 -> 349.1 MB | 15 (15) | 347 / 362 (96 %) | 84 s | 190 s | 0.38 | 914 |
| hard, whip L1-3 | 960 s (34 min) | 0 / 0 / 0 | 344.0 -> 345.2 MB | 64 (59) | 218 / 287 (76 %) | 15 s | 56 s | 0.00 | 7311 |
| chaos, whip L1-4 | 300 s (11 min) | 0 / 0 / 0 | peak 354 MB | 27 (27) | 61 / 90 (68 %) | 11 s | 20 s | 0.00 | 14226 |
| hard, no hits | 300 s (11 min) | 0 / 0 / 0 | - | 0 | - | - | 300 s (whole run) | 1.00 | 0 |

- **Stability:** 4680 simulated seconds in total. There were no crashes, no NaNs and no
  instability errors, and every `summary.json` ended with `complete: true`.
- **Memory:** RSS stayed flat (+0.2 to +2 MB over each run; the per-hit event lists
  grow by a few KB). The model is compiled once and the terrain uses a fixed geom
  pool (258 pool geoms), so the geom count is constant.
- **Speed:** RTF was 0.45-0.50 with 3-4 processes at once (about 0.6 alone).
- **Hits per level:** L1 was survived 99-100 % of the time. L2 was survived 92-95 %
  on normal and hard terrain (79 % on chaos). L3 was survived 39 % on hard and 70 %
  on chaos (small sample). L4 was survived 43 % on chaos.
- **Terrain alone:** hard terrain without hits caused 0 falls in 5 min. Falls come
  from hits, not obstacles. Per-terrain falls/min is in each report ("obstacle
  success").
- **Recovery:** only 4 of the 121 falls recovered by themselves (recovery 0-6 %,
  mean 4.3 s). There is effectively no righting reflex. The mean time to
  unrecoverable failure equals the mean time to fall.

## Bugs found and fixed in this pass

1. **The checkerboard ground never followed the fly (visible after ~1 m, i.e. about
   70 s of walking).** MuJoCo compiles a geom that sits at its body's origin with
   `geom_sameframe = 1` and then ignores `geom_pos`, so `GroundRecentering` had no
   effect. It now clears the flag (`fly_simulator/terrain/flat.py`). Physics wasn't
   affected, because the collision plane is infinite. Regression test:
   `tests/test_terrain.py::test_ground_recentering_moves_the_drawn_plane`.
2. **CSV precision in long runs.** Every float column was written with 5 significant
   digits. After 1000 s the timestamps only resolved 0.1 s, so 50 Hz rows shared
   timestamps, and x = 25 m only resolved 1 mm. Times, x/y/z and distance are now
   written with 4 fixed decimals (`LoggingConfig.abs_decimals`). Regression test:
   `tests/test_logging.py::test_absolute_columns_keep_precision_in_long_runs`.
3. **Checked with no bug found:** reset during a crack and with a queued crack, H
   during a crack, spawning and auto hits while the fly is down, auto reset in the
   window (by reading the code: the camera snaps back when sim time goes backwards), and physics, fall detection,
   chunk math and rendering at x = 100 m and 1 km (the fly teleported there). All
   were fine. The edge-case sequence is now a test:
   `tests/test_whip.py::test_mid_crack_reset_mode_toggle_and_spawn_while_fallen`.

## Known limitations

- Flight (`--flight`): half the RTF (dt 5e-5 while walking too); L2 whip cracks topple
  the fly more often at dt 5e-5 (2/5 vs 0/5, same with the normal body at that dt);
  the escape direction comes from the swatter's position, not from the eyes; wings
  rest spread while walking; PID flight controller, scripted landing
  (docs/FLIGHT.md, Limitations).

- Integration: with `--brain-actions`, brain-triggered jumps also fire while the fly
  is lying down (pre-existing). Short-mode escape jumps often tumble (no flight
  emulation by default). `sync_wait_s` costs wall time while a loom is active. A job
  runs on top of the (flat) procedural terrain, which keeps recycling chunks.

- **No righting reflex.** A fly on its back stays down, so long unattended runs need
  `--auto-reset-after`. This is the main gap and the reason for the RL work.
- The sim runs slower than real time (0.6-0.7x alone, whip included, which adds ~22 %).
- Rendering uses float32. Far from the origin (x > ~1 km, about 20 h of walking with no
  reset) the whip shows slight jitter. Physics is float64 and unaffected.
- Brain: body touch reaches the model only through the few ascending mechanosensory
  afferents (there is no VNC), so whips and shoves don't change the walk even with
  `--brain-steer`. Looming is injected into LC4 directly, not seen. The DN -> CPG
  mapping gains are ours: with the default `backward_ref` of 40 Hz, looming stops the
  fly; at 20 Hz it walks backward. The brain costs one CPU core.
- Level-4 hits launch the fly 5-60 cm, off the terrain feature band. `distance_mm`
  includes that flight; `walked_distance_mm` and the average speed don't.
- The terrain label depends only on x. See the README's notes for the rest.
- Actions: jumps go mostly straight up (~1 mm backward drift), and boosted jumps tumble
  (no wing aerodynamics). Groom is one 3 s recorded bout of the front legs only.
  Brain grooming doesn't fire from natural input (DNg12 ≤ 13 Hz vs the 20 Hz
  threshold). Head bristle / head whip input drives MN9 (~120 Hz), so with
  `--brain-actions` strong head input would extend the proboscis. Walking can be
  irregular for ~0.5 s after an action. docs/ACTIONS.md §6.

## Disk space warning

The disk had **~820 MB free** at the end of this pass. `metrics.csv` is ~225 bytes per
row: at the default `--log-hz 50` that is ~40 MB per simulated hour. Use `--log-hz 5-10`
for long runs, and `--no-log` for casual sessions. PPO checkpoints and TensorBoard logs
also add up. Free some space before the first real training run. `runs/` is
gitignored. The soak outputs (`runs/soak/`, ~7 MB) can be deleted.

## Recommended next steps

1. **First PPO run** (docs/RL.md, "Recommendations"):
   `.venv/bin/python scripts/train_ppo.py --timesteps 10000000 --n-envs 8 --tensorboard`,
   overnight. Check `episode/forward_speed` (~13-14) early. If the gait degrades,
   lower `--action-scale` or `--log-std-init`.
2. **Target the righting reflex.** Recovery is 0-6 % today. Consider a larger action
   scale (0.3), `--adhesion-residual`, a longer `--fall-terminate-after` (3-5 s) and a
   righting bonus.
3. **Compare policy and baseline** with `eval_policy.py` and `eval_robustness.py` on
   the same seeds, >= 10 sim min per condition. The table above is the baseline to
   beat: normal terrain with whip L1-2 gives a mean time to fall of 124 s and 608
   falls/km.
4. Optional: a speed pass (the controller is ~45 us/step in Python), and whip-terrain
   collision calibration.
