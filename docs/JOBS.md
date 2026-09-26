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
| `mowing` | pushes a red push mower up and down a lawn in rows, leaving light / dark mowing stripes; the grass grows back | m² mowed, rows mowed, lawns completed |
| `raking` | leaves fall from an autumn tree; the fly sweeps them into a pile with a rake; when the pile is done, the wind blows it away | leaves raked, piles completed, gusts survived |

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

### kebab

`perpetualfly/jobs/kebab.py`, tests in `tests/test_jobs_kebab.py`.

**Scene.** A vertical spit (hinge about z, driven by a velocity actuator at 12 rpm)
stands 1.7 mm in front of the carving station. Its meat is an inverted cone 2.5 mm
tall (r 0.9 mm at the bottom, 1.25 mm at the top): 10 rings, 199 browned chunks
(visual box geoms on the spit body) over a pinkish core. Behind it is a heater with
glowing bars, and below it a steel drip tray and base. The floor is shop tiles. The
fly wears a chef hat. The props are fly-scale: the doner is about as tall as the fly
is long.

**Knife.** A handle, guard and 0.8 mm blade sit on the right front `rf_tarsus1`.
They are visual-only geoms with no mass and no contacts, added to the fly spec just
before `add_fly` (the extension wraps `world.add_fly` once). They cannot change the
dynamics: the fly still weighs 1.02 mg and no contact pair involves them. I chose
the knife's direction by searching over the recorded stroke. With it, the tip is
more than 1.6 mm ahead of the thorax 58 % of the time and never goes below the
floor.

**Stroke.** `CarveStroke` is a `Groom` subclass. It loops the recorded NeuroMechFly
front-leg grooming clip, with the mid legs extended and the mid / hind legs planted
and adhering. It advances the clip with a phase accumulator, so its speed can change
while it runs. It registers itself as a stationary action
(`session.STATIONARY_ACTIONS`), so standing still to carve is not flagged as
`no_progress`. The steering speed is 0. Found while building it: `Groom` picks the
clip with `self.source`, but `ActionManager.trigger(..., source=...)` overwrites
that attribute with the trigger source ("key" / "api" / "brain"). So a triggered
`Groom()` has actually been running the **synthetic** sweep, not the recording.
`CarveStroke` works around this, and a test checks that its targets match the
recorded clip. `Groom` itself is not changed here.

**Cut, shaving and regrowth.** Every 1 ms the job checks three blade points
(mid-blade to tip) against the ripe chunks. A point within 0.2 mm of a chunk centre,
with the tip moving faster than 12 mm/s, cuts that chunk. There is at most one cut
per 0.12 s. The blade and meat do not collide physically: the cut is a geometric
test. When a chunk is cut, it is hidden (size and alpha), and a shaving from a pool
of 18 free bodies (3 µg ellipsoids, recycled oldest first) is launched from the
chunk's pose with the spit's surface speed plus an outward kick. It falls onto the
tray with real physics. Shavings touch only the tray, the floor and each other, not
the fly. A shaving that goes NaN or leaves the arena is put back on the tray
(`shavings_lost`). A cut chunk regrows over 18 s, from raw pink to browned, and can
be cut again from 85 % of its size. A reset brings back a whole kebab.

**Counters.** Shavings (the work counter), kebabs completed (every 60 shavings), and
µg served (3.5 µg per slice), also shown as a human-scale mass: × (1.75 m / 2.5 mm)³,
so a 3.5 µg slice is ~1.2 kg. There is also a rate per minute (last 60 s). The HUD
states that the knife stroke is a real recorded fly grooming bout.

**Station keeping.** If the thorax drifts more than 0.45 mm or 25° from its station,
the fly stops carving, backs off or walks back, and resumes. This never triggered in
the runs below.

**Brain (`--brain`, off by default).** The job turns on the full body (proboscis
joints) and treats the kebab as food: every 4 s it sends a 1 s sugar-GRN stimulus
(the T key's set, labelled `KEBAB (sugar GRNs)`). The proboscis follows the brain's
MN9 rate: fully extended at 60 Hz, smoothed with τ 0.15 s. The job drives the
proboscis itself, because the brain-trigger path never interrupts a running action,
and the carve action always runs. The food response is the connectome's wiring.
"Kebab" is only a label on sugar input. The job trims `BrainLink.stim_log` to keep
memory constant.

**Stress (optional).** If `session.stress` is an enabled handle (docs/STRESS.md),
carve speed is multiplied by `1 + gain · (freq_mult − 1)`. `run_job.py` does not
install stress, so this path is tested with a fake handle only.

### mowing

`perpetualfly/jobs/mowing.py`, tests in `tests/test_jobs_lawn.py`.

**Lawn.** 20 × 17.2 mm, 6 rows along x, 2.8 mm apart, starting at the fly's spawn
row (y = 0). The grass is a fixed pool of 1,968 thin blade boxes (0.5 ± 0.15 mm
tall, jittered 0.42 mm grid) plus 525 flat 0.8 mm "stripe tiles" on the soil. All
are `collide="visual"`, so they can't trip the fly, and resizing / recolouring them
needs no BVH refit (API_NOTES §8). They only need `geom_size`, `geom_pos`,
`geom_aabb`, `geom_rbound` and `geom_rgba`. The soil box's top is 50 µm above the
ground plane: a box face flush with the plane z-fights with the checker at a camera
distance of 25–30 mm (this showed up as big dark squares on the lawn).

**Mower.** It sits on planar joints: slide x, slide y and a yaw hinge, at a fixed
height. So it can't tip, climb or be lost, and the slide ranges keep it within 3 mm
of the lawn. The slide damping (0.3 µN per mm/s) stands in for the wheels' rolling
resistance: pushing at 5 mm/s takes ~0.15 body weight. The only colliding part is a
round deck (R 1.6 mm, z 0.47–1.23 mm, 0.3 mg). The fly touches it with head, thorax
and abdomen only, at friction 0.05 (`slippery_body_contact`, legs excluded). The
engine, wheels and a handle whose grip sits just above the fly's head are visual.
The job turns the yaw hinge (rate-limited to 2.5 rad/s) so the mower faces the way
it is pushed and the handle trails toward the fly. The deck is round, so turning it
pushes nothing.

**Behaviour.** `PushPilot` is the sisyphus push loop, generalised and shared with
raking: approach (orbit round the prop to its back relative to the goal), then align,
then push with over-steer. The goal is a pure-pursuit point 3 mm ahead on the
current row's centre line. At a row end, the next row's goal is behind the mower, so
the fly walks round it and the mower comes about in a U-turn onto the next row. Rows
run as a serpentine, 0→5 and then back 5→0. A row counts when the mower centre gets
within 0.2 mm of the row end and within 1 mm of the row line. Blades under the deck
are cut to 0.12 mm and painted light (rows mown along +x) or dark (along −x), the
classic stripes. They grow back linearly in 45 s, about 1.5 lawn passes, and their
colour fades back to long grass. After each lawn the fly grooms for 2 s (wipes its
brow).

**Counters.** Area mowed: m² of full-height grass cut. Partly regrown grass counts pro
rata, and the area is the work counter, so the stuck timer only runs when nothing is
cut. Also rows mowed and lawns completed (every 6 rows). The HUD shows the current
row and direction and the share of the lawn that is short. Two `PushPilot` details
matter for any job:

* The align state's lateral tolerance must be larger than `align_radius`. Otherwise
  approach and align hand the fly back and forth every update and it freezes (found
  with a small prop).
* When the target heading is almost straight behind (|error| > 150°), the pilot
  commits to one turning direction. The error's sign otherwise flips between updates,
  and a fly turning on the spot freezes.

### raking

`perpetualfly/jobs/raking.py`, tests in `tests/test_jobs_lawn.py`.

**Yard.** 22 × 16 mm, with a bare-earth pile spot (R 2 mm) at (13, −3.5). A tree
stands beyond the far edge: a trunk plus 9 orange / red / yellow canopy ellipsoids,
overhanging the yard at z ≈ 8 mm. All of it is visual.

**Leaves are kinematic.** A fixed pool of 40 mocap bodies, each a flat ellipsoid
(1.1 × 0.7 mm) plus a stem, in 6 autumn colours, all visual-only. The job moves them
through these states:

* **tree**: inside the canopy.
* **falling**: one leaf every 1.6 s, 2–3.5 s down with a side-to-side flutter and a
  rocking roll.
* **ground**
* **pile**: heaped on a dome whose height grows with the count.
* **gust**: an arc up and across the yard.

No leaf physics means no instabilities, and 40 leaves cost nothing. The model never
grows: leaves blown out of the yard go back into the canopy and fall again.

**Rake.** A mocap body (visual) that the job puts at the thorax pose every update,
like a rake welded to the thorax front. A wooden handle runs from above the fly's
head down to a green comb (3 mm wide, 11 tines) 2.1 mm in front of the thorax. The
rake moves leaves like this: every update, a ground leaf in the 1 mm strip behind the
comb's front face (within the comb width) is moved onto the face and marked
*raked*. Leaves in front of the comb go wherever the fly walks, and slide off the
comb ends when it turns. A ground leaf inside the pile circle joins the heap. It is
counted only if it was raked (leaves that fall straight onto the pile don't count).

**Behaviour.** The target is the ground leaf that minimises distance to fly + 0.5 ×
distance to pile, with 30 % hysteresis and a re-check every 2.5 s while
approaching. That leaf is the "prop" of a `PushPilot` (radius 1 mm): the fly goes
round it to the staging point 2.9 mm behind it (relative to the pile), faces it, and
then "pushes" it to the pile with the comb, exactly like the boulder. The raking
pilot has `unstick=False`, because nothing blocks the fly here. With no leaves on
the ground, it leans on its rake beside the pile.

**Wind.** A pile of 18 leaves is complete (the fly grooms for 1.5 s). A gust comes
3 s later, and also every 75 s regardless, sometimes flattening a pile that is
nearly done. It blows the whole pile plus 30 % of the ground leaves up and across the
yard. 35 % of them fly out of the yard and return to the canopy. The HUD flashes
`~~~ WIND GUST! ~~~`.

## Verification

The test file is `tests/test_jobs.py`: synthetic tests of the registry, steering and
recording helpers, plus short sisyphus and wheel runs covering contacts, pushing,
explicit recovery, NaN recovery, a lost boulder and the wheel spinning. `tests/test_jobs_lawn.py`
(9 tests, ~13 s) covers mowing and raking: contact bits, the grass pool (cut, stripe
colour, regrowth), row completion and the serpentine, the mower being pushed, the
mower surviving an explicit reset, rake carrying, pile counting, gusts, the constant
leaf pool, and the rake following the thorax.
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
| `mowing` headless + 640×432 timelapse every 3 s (job renderer) | 185 s | **30 rows, 5 lawns**, 1251 mm² (0.00125 m²) mowed, 6627 blades cut, 30 pushes started, 1 back-away unstick, 0 instabilities | 0 / 0 | 0.47 |
| `raking` headless + 640×432 timelapse every 3 s | 185 s | **46 leaves raked**, 2 piles completed, 2 gusts survived (47 leaves blown about, 14 out of the yard and back to the tree), 38 leaves fallen from the tree, 0 instabilities | 0 / 0 | 0.55 |

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

* mowing: the mower is on planar joints at a fixed height and turned kinematically
  toward its push direction (the fly pushes only its translation). The stripes wobble
  a little, because the pure-pursuit push is not perfectly straight. The serpentine
  mows the last row twice (once each way) at the turnaround.
* raking: the rake and leaves are kinematic (mocap, visual only). The rake moves
  leaves by a geometric test, not by contact forces, so the fly can't trip over
  either. Leaves pushed against the yard edge line up there until the fly fetches
  them.

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
* The main app runs jobs too: `run_sim.py --job NAME [--job-config JSON]` (all app keys
  and flags available; the whip is turned off because the idle whip lies across the
  job scenes, so hit keys shove). `scripts/run_job.py` remains the standalone runner.
