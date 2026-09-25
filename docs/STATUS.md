# PerpetualFly: status (2026-09-25, after the overnight soak pass)

## What works

| milestone | status |
|---|---|
| M1 walking fly | NeuroMechFly (FlyGym 2.1 / MuJoCo 3.9) walks forever with FlyGym's hybrid CPG controller plus a heading hold along +x. There is a smooth follow camera (C: follow / side / top), a live OpenCV window (physics runs in a worker thread) and a `--headless` mode. |
| M2 hits | Default: a **physical whip** whose chain of capsules hits the fly only through MuJoCo contact, with the impulse measured. H switches to **shove** (an `xfrc_applied` force on the thorax). Arrow keys, SPACE and U choose the direction; 1-4 set strength (calibrated, see docs/WHIP.md and docs/PERTURBATION_CALIBRATION.md). |
| M3 terrain | Endless chunked procedural terrain from a fixed geom pool (constant model size; recycled, and 2173 chunks were recycled in the 31 min soak). Presets flat / easy / normal / hard / chaos. R B S G D spawn obstacles, F flattens the next chunk. |
| M4 falls, logging, auto hits | The FallDetector tracks UPRIGHT / DESTABILIZED / FALLEN / RECOVERING. X resets, `--auto-reset-after` resets automatically, `--auto-perturb` gives seeded random hits. Every run writes `runs/<ts>/{config.json, events.csv, metrics.csv, summary.json}`. |
| RL (phase 10) | A residual Gymnasium env, a curriculum, `scripts/train_ppo.py` and `scripts/eval_policy.py` exist and have been smoke-tested. **No real training run has been done yet** (docs/RL.md). |
| robustness eval | `scripts/eval_robustness.py` runs N headless sessions in parallel, or aggregates existing run folders, and writes `report.md` and `report.json` with the spec's long-horizon metrics. |

Tests: `.venv/bin/python -m pytest -q`: 78 passed, ~2 min.

## How to run

```bash
.venv/bin/python scripts/run_sim.py                                   # window, normal terrain
.venv/bin/python scripts/run_sim.py --terrain hard --auto-perturb     # + random whip cracks
.venv/bin/python scripts/run_sim.py --headless --auto-perturb --auto-reset-after 5 \
    --max-seconds 600 --log-hz 10                                     # unattended
.venv/bin/python scripts/eval_robustness.py --seeds 0,1,2 --jobs 3 --duration 300 \
    --terrain normal --hit-mode whip --auto-levels 1,2 --out runs/robust/normal
.venv/bin/python scripts/eval_policy.py --baseline --stage normal --sim-seconds 180
```

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
   effect. It now clears the flag (`perpetualfly/terrain/flat.py`). Physics wasn't
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

- **No righting reflex.** A fly on its back stays down, so long unattended runs need
  `--auto-reset-after`. This is the main gap and the reason for the RL work.
- The sim runs slower than real time (0.6-0.7x alone, whip included, which adds ~22 %).
- Rendering uses float32. Far from the origin (x > ~1 km, about 20 h of walking with no
  reset) the whip shows slight jitter. Physics is float64 and unaffected.
- Level-4 hits launch the fly 5-60 cm, off the terrain feature band. Distance and
  speed include that flight.
- The terrain label depends only on x, and automatic hits keep coming while the fly
  is down. See the README's notes for the rest.

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
