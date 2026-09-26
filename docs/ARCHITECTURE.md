# Architecture, internals and known limitations

How the code is laid out, how the physical whip works, and the known limits of the
simulation. How to run things is in [USAGE.md](USAGE.md); the original plan is
[dev/SPEC.md](dev/SPEC.md) and the FlyGym / MuJoCo API facts the code relies on are
in [dev/API_NOTES.md](dev/API_NOTES.md).

## Layout

```
fly_simulator/
  config.py        dataclass configs (JSON load/save); AppConfig holds all sub-configs
  simulation.py    Simulation: FlyGym sim + controller, hooks, world extensions, NaN checks
  app.py           Session (wires terrain/whip/falls/metrics/logging) + main loop / CLI
  physics_thread.py  physics worker thread for the live window
  rendering.py     offscreen mujoco.Renderer + adaptive smoothed follow camera
  display.py       display-size helpers (Retina upscale) for the fly and brain windows
  media.py         screenshots (I) and live MP4 recording (M)
  controllers/     fast bit-identical versions of FlyGym's HybridController
  terrain/         flat ground, endless chunked procedural terrain (geom pool)
  interaction/     OpenCV live window, keyboard, external-force shoves, physical whip (whip.py)
  metrics/         locomotion stats, fall detector, run metrics, run logger
  rl/              residual RL env (Gymnasium), wrappers, long-horizon evaluation
  actions/         action library: jump, freeze, groom, back away, turn, wings, proboscis
                   (ActionManager, brain triggers, full-body fly; docs/ACTIONS.md)
  stress.py        octopamine pain / arousal body effects (--stress; docs/STRESS.md)
  vision/          compound-eye looming -> LC4 / LPLC2 (--whip-vision, the swatter) and
                   real vision via flyvis (--real-vision; docs/VISION.md)
  brain/           Shiu et al. 2024 LIF engine on FlyWire v783 in a worker process, sensory /
                   motor mapping, habituation, octopamine, plasticity (docs/BRAIN.md)
  brain_link.py    glue between the app and the brain worker (--brain*, keys, HUD, logging)
  brain_viz/       brain window, playground, brain replay, neuropil atlas (docs/BRAIN_WINDOW.md)
  flight/          flapping-wing flight: flight fly, wing kinematics, flight controller
                   (--flight; docs/FLIGHT.md)
  games/           connectome brain games (scripts/play.py; docs/GAMES.md)
  course/          obstacle courses (--course; docs/COURSE.md)
  jobs/            eternal jobs (--job, scripts/run_job.py; docs/JOBS.md)
  senses/          taste patches: sugar / bitter spots tasted by the legs (--taste-patches;
                   docs/TASTE.md); odour zones (--odor-zones / --learning; docs/FEAR_LEARNING.md)
scripts/run_sim.py        the app
scripts/play.py           brain games (docs/GAMES.md)
scripts/run_job.py, run_course.py  jobs / courses without the full app
scripts/fetch_brain_data.py  download the FlyWire data (docs/BRAIN.md)
scripts/demo_*.py         stand-alone demos of terrain / perturbation / falls / whip / actions
scripts/brain_replay.py   replay / export a --brain-record recording (docs/BRAIN_REPLAY.md)
scripts/eval_robustness.py  N headless sessions -> long-horizon robustness report
scripts/train_ppo.py, eval_policy.py  residual PPO training / evaluation (docs/RL.md)
tests/
```

## The whip in one paragraph

`fly_simulator/interaction/whip.py`: a mocap "hand" drives a free wooden grip through
a stiff weld; 12 tapered capsules (6 mm, 2 mg, leather brown with a red tip) hang
from it on pairs of stiff, damped hinges. The capsules collide only with the fly. A
crack raises the whip, cocks it beside the fly, swings the handle and **stops** it
so the whip's rest line lies just outside the fly: only the lash (the chain's
overshoot) hits. A post-step hook sums the contact forces on fly geoms into the
measured impulse. On flat ground 192/192 calibration cracks connected; levels 1–4
give ≈ 0.17 / 0.29 / 0.64 / 1.0 uN*s and tip the fly 0 % / 21 % / 75 % / 90 % of
the time. Details, the calibration table and limitations: [WHIP.md](WHIP.md).
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
  step. See docs/dev/API_NOTES.md §13.
- **Live display:** the window is an OpenCV window showing frames from MuJoCo's
  offscreen renderer. MuJoCo's own passive viewer isn't used: on macOS it needs `mjpython`,
  and its built-in shortcuts clash with this project's keys (see docs/dev/API_NOTES.md §11).
  There is no mouse camera control; press C to switch views. OpenCV only reports key
  presses, not held keys. On a Retina display the ×2 upscale costs the main thread
  about 10 ms per frame (resize ~3 ms, then `imshow` of 4× the pixels). The window then
  shows ~24 fps instead of ~30. The physics thread isn't affected: RTF was 0.48–0.49
  vs 0.50–0.52 at ×1. `FLY_SIMULATOR_FLY_SCALE=1` restores the old size.
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