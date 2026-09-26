# Eternal jobs (`perpetualfly/jobs/`)

This is the internet genre of "a fly doing an absurd job for eternity". Each job is a
scene with props (compiled into the MuJoCo model), task logic that steers the
walking fly, a work counter, and an "eternal stream" HUD. It runs forever: when the
fly falls it gets an explicit, counted reset, the props regenerate, and memory stays
constant.

| job | what happens | counter |
|---|---|---|
| `sisyphus` | pushes a 4 mm boulder up a 10° hill to a summit curb; lets go; the boulder rolls back into the valley; walks down, gets behind it, pushes again | summits, metres pushed |
| `hamster_wheel` | runs inside a 14 mm wheel (hinge joint) that turns only because its feet push the slats | revolutions, distance, top speed |
| `kebab` | stands at a turning doner spit in a chef hat and carves it with a knife on its right front leg, using the real recorded grooming stroke; shavings fall onto the drip tray, the meat regrows | shavings carved, kebabs completed, µg served, rate |

```bash
python scripts/run_job.py --job sisyphus                   # window; Q quit, C camera, P pause, X reset, I screenshot
python scripts/run_job.py --job hamster_wheel --headless --max-seconds 300
python scripts/run_job.py --job sisyphus --headless --record runs/jobs/sis --segment-s 60 --keep 3 --timelapse 5
python scripts/run_job.py --rotate --rotate-minutes 10     # all jobs in turn, forever
python scripts/run_job.py --list
```

Options: `--job-config '{"slope_deg": 8}'` (job config overrides), `--width/--height`,
`--fps` (recording, default 15), `--log` (write `runs/<ts>/`, off by default), and
`--whip` / `--brain` to add the physical whip or the connectome brain (off by default).
`--rotate` builds a new Session for every job, because props are compiled into the model.

## Framework API (for new jobs: lawn mowing, leaf raking, doner kebab ...)

```python
from perpetualfly.jobs import (EternalJob, JobConfig, CameraPreset, register_job,
                               create_job_session, install_job, make_job, JobRunner)
from perpetualfly.jobs.geometry import (add_box, add_slope, add_plane_box, contact_kwargs,
                                        slippery_body_contact, exclude_fly_legs, PROP_BIT)
```

### Lifecycle

1. `job = make_job(name, cfg_or_dict)` (or `get_job(name)(cfg)`).
2. `job.configure_app(app_cfg)`: called once before the Session is built. The base
   version sets flat terrain, turns automatic hits off, turns off the app's own auto
   reset (the job does its own) and turns floor reflections off. Override it to
   change e.g. `app_cfg.fly.spawn_height` (the hamster wheel spawns the fly on the
   inside bottom of the wheel), and call `super()`.
3. `job.extension(world)`: a world extension that adds props to `world.mjcf_root`
   before `add_fly`.
4. `Session(cfg, world_extensions=[job.extension, ...])`, then `install_job(session, job)`.
   `install_job` checks that `job.required_names` exist in the model and calls
   `job.attach(session)`. It also chains `job.after_physics` onto
   `session.after_physics` and sets `session.job`. `create_job_session(name, app_cfg,
   job_cfg)` does steps 1–4 and returns `(session, job)`.
5. Run it: `JobRunner(session, job, headless=True, record_dir=..., timelapse_every_s=...)
   .run(max_seconds)`. You can also use the app's own loop: call `sim.step(n)` then
   `session.after_physics()`.

### What `attach` wires up

* **`update()`**: a post-step hook that runs every `cfg.update_every_steps` physics
  steps (default 10 = 1 ms). This is where the job logic lives: steering, counters,
  prop checks, triggering actions. It runs inside `sim.step`, so it must not reset the sim.
* **`on_reset()` + `reset_props()`**: run after every sim reset. FlyGym's keyframe
  reset has already put every prop body back at its spec pose (the fly respawns at the
  origin facing +x). Use `reset_props()` to re-randomise (seeded) so an eternal run
  doesn't repeat the same failure forever.
* **`ground_height(x, y)`**: the walkable surface height. It goes into the fall
  detector and the logger, so a fly on a hill or inside a wheel isn't read as
  "airborne". Implement it.
* **`progress_xy(x, y)`** (optional): for treadmill-like jobs. It returns the thorax
  position in the frame the fly walks in (the wheel adds the surface travel), so the
  detector's stall / no-progress rules don't flag running on the spot as stuck.
* **`self.steering`** (`Steering`, installed as the controller's `signal_filter` and
  chained in front of the brain's if one is set):
  `steering.set(heading_rad, speed)`, `steering.aim_at((x, y), speed)`. Heading error
  drives `[1+d, 1-d]` with d = clip(2.5·err, ±0.8), the fly slows down in sharp turns,
  and above 50° of error it turns on the spot. `speed` scales the CPG amplitude
  (0 = stand still). `ActionManager` actions still override it.
* **Fall listeners** count `n_falls` and `n_self_righted`.

### Eternal operation (base class, `after_physics`, outside `sim.step`)

* The fly has been down (FALLEN / RECOVERING) for `cfg.recover_after_s` (2.5 s) →
  `job.recover("fell")`: `session.reset("auto")`. This is logged as `job_recovery` and
  `auto_reset`, and counted in `RunMetrics.n_resets`, `job.n_auto_recoveries` and
  `job.recovery_reasons`. It is never a hidden teleport of the fly.
* No `add_work()` for `cfg.stuck_timeout_s` (120 s) → `recover("stuck")`.
* `JobRunner` catches `SimulationInstabilityError` (NaN / blow-up) →
  `recover("instability")`.
* `job.unstick()`: if the fly hasn't moved 0.5 mm in 1.5 s (wedged against a wall or a
  prop), it triggers the `back_away` action (counted in `n_unstuck`).
* Constant memory: jobs keep counters, not per-event lists. The fly track is a
  fixed-size deque and `ActionManager.history` is trimmed to `cfg.history_keep`. Props
  that come and go (grass blades, leaves, kebab slices) must be a fixed pool of bodies
  compiled once and recycled (move them with `qpos` / mocap; never grow the model).

### Helpers for jobs

`add_work(n)` (the main counter; also resets the stuck timer), `run_time()` (monotonic
across resets), `fly_xy()`, `fly_down()`, `fly_moved(window_s)`, `say(msg)`.

### HUD / camera / stats

* `hud_lines()` returns: title + tagline, `Day N HH:MM:SS sim (wall …)`,
  `<work_label>: <work>`, then `job_hud_lines()` (your per-job stats), then falls /
  self-righted / auto-recoveries / state. Set `title`, `tagline`, `work_label` and
  `work_format`.
* `camera_target()` (a world point) and `camera_preset()` (`CameraPreset(azimuth,
  elevation, distance, tau_s)`; `azimuth=None` follows the fly's heading plus
  `azimuth_offset`). `JobCamera` adds a "job" mode to the app's `SmoothFollowCamera`,
  and C cycles job / follow / side / top.
* `stats()` merges the base counters with `job_stats()` (printed by the runner,
  returned by `JobRunner.run`).

### Props and contacts (`jobs/geometry.py`)

Contact bits: `FLY_BIT` = 8 and `TERRAIN_BIT` = 16 (docs/API_NOTES.md §14), plus
`PROP_BIT` = 32 for jobs.

| `collide=` | contype | conaffinity | touches |
|---|---|---|---|
| `"static"` | TERRAIN | FLY | fly, dynamic props (terrain-like) |
| `"dynamic"` | PROP | FLY, TERRAIN, PROP | fly, ground plane, static props, other dynamic props |
| `"fly"` | 0 | FLY | only the fly |
| `"visual"` | 0 | 0 | nothing |

Every colliding geom gets `priority=1` plus FlyGym's stiff contact parameters. Always
give bodies an explicit mass. Static world-body geoms must never move at runtime (BVH
gotcha, API_NOTES §8); geoms on jointed or mocap bodies can move freely.

Lessons from tuning (they apply to any prop the fly pushes):

* **Sticky tarsi climb light props and walls.** Adhesion lets the legs walk up a free
  body or a vertical wall, and the fly flips over backward.
  `slippery_body_contact(spec, body, geom, fly_name, friction=0.05)` lets only head,
  thorax and abdomen touch a prop, through explicit low-friction `<pair>`s, and
  excludes the legs. Give walls `friction≈0.1`, and keep waypoints away from walls.
* **A rolling prop drags the head up.** At friction 1 the rolling boulder's surface
  lifted the fly's head by ~1 body weight and flipped it. Low head friction fixes this.
* **Pushing a heavy prop at head height pitches the fly nose-up.** Keep push forces
  ≲ 0.3 body weight (10 µN ≈ 1 BW). The boulder is 0.3 mg.
* **Rolling resistance:** free-joint `damping` behaves very nonlinearly on a rolling
  sphere, so avoid it. Use condim-6 rolling friction instead, per surface: the
  contact takes the higher-priority geom's value. `contype=TERRAIN, conaffinity=0`
  makes a surface that only props touch (sisyphus's valley "mud").
* **Troughs beat walls.** Banks sloping 12° toward the centre line (`add_plane_box`)
  bring the boulder back to the centre, which made the push loop robust.
* Keep props out of the fly's spawn volume (thorax ~2.1 mm up at t=0).

### Skeleton of a new job

```python
@dataclass
class MowConfig(JobConfig):
    n_blades: int = 200

@register_job
class LawnMowerJob(EternalJob):
    name, title, tagline, work_label = "lawn_mower", "LAWN MOWER FLY", "…", "m² mowed"
    config_cls = MowConfig
    required_names = ("mow/mower",)

    def extension(self, world): ...          # mower body + a pool of grass blades
    def on_attach(self): ...                 # ids by name
    def update(self):                        # steer a boustrophedon path, cut blades
        self.steering.aim_at(next_wp); ...; self.add_work(area)
    def reset_props(self): ...               # regrow the lawn (recycle the pool)
    def job_hud_lines(self): return [...]
```

Then import the module in `registry._load_builtin` so `available_jobs()` finds it.
The app can integrate jobs without further framework changes:
`job = make_job(name); job.configure_app(cfg); Session(cfg,
world_extensions=[job.extension]); install_job(session, job)`, then draw
`session.job.hud_lines()` and use `JobCamera(cfg.camera, job)` as the renderer's camera.

## The jobs

### sisyphus

The world runs along +x (uphill). The valley floor has a 12° banked trough (flat
centre band ±1 mm, walls at ±7 mm, slippery), the ramp is 14 mm long at 10° (2.5 mm
high), and a stone summit curb with a red flag is at x = 19 mm, with a counter-slope
behind the valley. The boulder is a free sphere, R = 2 mm, 0.3 mg, friction 1 with the
ground. Its rolling friction is 0.28 on the hill (just under tan 10° · R, so it
starts rolling back slowly and reaches ~70 mm/s) and 0.3 on the valley "mud", where
it stops. The fly touches it only with head, thorax and abdomen, at friction 0.05.

State machine: **approach** (orbit round the boulder at a clearance, remembering
the direction it chose; from behind, pure pursuit onto the goal line) → **align**
(turn on the spot to face it) → **push** (heading = φ + 0.5·(φ − ψ), clipped ±25°, at
speed 0.7) → the boulder reaches the curb = **summit** → **release** (turn 110° aside
for 0.6 s) → **descend** (walk down in a lane beside its path until it rests) →
approach … If it loses the boulder halfway, the boulder rolls back down and the fly
goes after it (Sisyphean by design). A boulder that leaves the arena (or goes NaN)
is replaced by a new one dropped from the sky in front of the fly (`boulders_lost`).

### hamster_wheel

The wheel has an inner radius of 7 mm and a 5 mm running width. It is 60 box slats
in yellow / orange blocks, so the rotation is visible, with slippery blue lips (the
near one translucent so the camera sees the fly), a back disc with spokes, and an
axle and stand. It weighs 4 mg, has hinge damping 2 µN·mm·s/rad, and only the fly
touches it. The fly spawns on the inside bottom (`spawn_height = 1.8`). Steering is a
heading hold along +x (the tangent) with a correction of −0.35 rad per mm of lateral
offset. The job counts revolutions, distance (surface travel) and top speed (max over
1 s windows).

## Verification

The test file is `tests/test_jobs.py`: synthetic tests of the registry, steering and
recording helpers, plus short sisyphus and wheel runs covering contacts, pushing,
explicit recovery, NaN recovery, a lost boulder and the wheel spinning.
Results of the long headless runs are below.

Measured on 2026-09-26 on this machine, with other agents' simulations running at
the same time:

| run | sim time | result | falls / auto-recoveries | RTF |
|---|---|---|---|---|
| `sisyphus` headless + 640×426 rolling MP4 (60 s segments, keep 1) + timelapse every 5 s | 300 s | **45 summits** (~6.7 s per cycle), 0.80 m pushed, 48 pushes started, 6 back-away unsticks, 0 boulders lost, 0 instabilities | 5 / 5 (all falls reset explicitly after 2.5 s) | 0.43 |
| `hamster_wheel` headless + 480×320 MP4 + timelapse every 10 s | 300 s | **99.4 revolutions**, 4.37 m, top speed 14.7 mm/s (~0.33 rev/s) | 0 / 0 | 0.41 |
| `--rotate` smoke test (both jobs, rebuilding the Session) | 2.5 s | the job switched as expected | 0 | – |
| `kebab` headless + 640×426 MP4 (20 s segments, keep 1) + timelapse every 3 s | 200 s | **1006 shavings**, 16 kebabs, 3.5 mg served, ~300 shavings/min steady (peak 382), ~88 of 199 chunks regrowing at any time, 0 repositions, 0 shavings lost, 0 instabilities | 0 / 0 | 0.26 |
| `kebab --brain` (full body, real FlyWire brain) | 12 s | 72 shavings; 3 food pulses, MN9 40–60 Hz on each, proboscis driven; 0 falls | 0 / 0 | 0.27 |

I checked the timelapse frames by eye (rendered by the job renderer only). The
sisyphus shot shows the trough, the boulder, the flag and the summit counter
climbing. The wheel shot shows the fly at the bottom of a spinning yellow / orange
wheel, visible through the translucent near lip.

Disk use: a 60 s segment at 640×426 and 15 fps is ~10 MB (quality 8), and a 60-frame
timelapse is ~0.9 MB. Keep `--keep` small.

The kebab frames (job renderer and timelapse) show the fly in its chef hat at the
spit, knife raised or in the meat, a carved band of pink regrowing chunks, the
shavings piling up on the tray, and the counters climbing.

## Limitations

* kebab: the knife passes through the meat (visual geoms; the cut is a geometric
  test, not a contact force). Only the band the recorded stroke reaches (about
  z 0.5–2 mm, on the fly's side of the spit) ever gets carved. The top stays whole,
  and the rotation brings fresh meat round. The stroke is the grooming clip replayed
  open loop. It does not aim at the meat.
* The window mode (`LiveViewer`, single-threaded loop) is the app's viewer code and
  was not opened in this session. Headless mode, recording and `--rotate` are verified.
* The sisyphus loop is closed-loop steering, not learned. About one fall per minute
  (usually during orbiting / turning on the spot next to the boulder or wall) is
  recovered by an explicit reset. Sometimes the fly loses the boulder halfway and it
  rolls back (Sisyphean, and counted as nothing). The boulder is light (0.3 mg,
  pumice) because heavier props make the fly rear up and flip.
* The fly respawns at the origin after a reset (FlyGym keyframe). Jobs that need
  another respawn point can write it into the keyframe, as `perpetualfly/course` does.
* `RunMetrics` keeps one float per fall (`falls`, `recovery_times`). This grows very
  slowly and is the only per-event list; the job's own counters are O(1).
* The app (`perpetualfly/app.py`) is not wired yet. `install_job` + `JobCamera` +
  `session.job.hud_lines()` are the integration points (see above).
