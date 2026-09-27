# Eternal jobs (`fly_simulator/jobs/`)

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
| `dead_hang` | dead-hangs by its front legs from a pull-up bar over a Venus flytrap; its grip tires, it re-grips and slips; when it falls, the trap snaps shut, and the fly respawns on the bar | time on the bar, hang streak (best), re-grips, slips, chomps, survival rate |
| `bowling` | pushes a 3 mm ball over the ramp at the head of the lane; it rolls down and scatters 10 free-body pins; a kinematic pinsetter clears / resets them; standard ten-pin scoring | pins knocked down, games, best / average game, strikes, spares, gutters, fouls |
| `broccoli_toss` | in a streamer's room the host fly brings a plate of broccoli to the viewer fly in the gaming chair; the viewer ponders it for 0.75–1 s, flicks the whole plate over its shoulder without looking, and everything behind it explodes (cartoon blast, props fly with real physics); the room rebuilds | plates yeeted, explosions, stream viewers, vegetables eaten: 0 |

```bash
python scripts/run_job.py --job sisyphus                   # window; Q quit, C camera, P pause, X reset, I screenshot, TAB HUD
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
from fly_simulator.jobs import (EternalJob, JobConfig, CameraPreset, register_job,
                               create_job_session, install_job, make_job, JobRunner)
from fly_simulator.jobs.geometry import (add_box, add_slope, add_plane_box, contact_kwargs,
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
* Optional **edit effects** (label them as such in the job's docs / HUD):
  `post_process(frame_rgb, t) -> frame_rgb` is a screen-space post-process of every
  rendered job frame, before the HUD (`install_post_process` wires it into the
  `FrameRenderer`, so the window, M recordings, rolling recordings, timelapses and
  screenshots of both `run_job.py` and `run_sim.py --job` get it; the default is
  identity and is not even called). `time_scale(present_dt) -> float` is a
  hit-stop / slow-motion request in presentation time: `install_job` gives the
  session a `PresentationClock` (`session.edit_clock`) that runs only that fraction
  of each chunk's physics steps (0 = the frame is held) and keeps
  `session.present_time()` (run time + the held time), which recordings are paced by;
  in a window a held chunk still takes its time on screen. The physics step sequence
  is unchanged (same results), only fewer steps per displayed frame.

### Props and contacts (`jobs/geometry.py`)

Contact bits: `FLY_BIT` = 8 and `TERRAIN_BIT` = 16 (docs/dev/API_NOTES.md §14), plus
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

`fly_simulator/jobs/kebab.py`, tests in `tests/test_jobs_kebab.py`.

**Scene.** A vertical spit (hinge about z, driven by a velocity actuator at 12 rpm)
stands 1.7 mm in front of the carving station. Its meat is an inverted cone 2.5 mm
tall (r 0.9 mm at the bottom, 1.25 mm at the top): 10 layers, 199 curved meat slabs
(visual mesh geoms on the spit body) over an inner core, under a browned crown with
the skewer tip poking out. Behind it is a burner with glowing ceramic tiles and a
warm light on the meat; below it a steel drip plate, motor housing and drip tray.
The shop has terracotta floor tiles and tiled walls. The fly wears a pleated chef's
toque. The props are fly-scale: the doner is about as tall as the fly is long.

**Looks (`jobs/kebab_assets.py`, visual only).** All meshes, textures and materials
are generated in code (numpy, OpenCV resizes) and passed to MjSpec directly; nothing
is written to disk. The slabs follow the cone (smooth surface noise shared by all of
them, a slight dome, rounded top / bottom edges and a wavy rim per layer, so the
cone reads as pressed, stacked layers), with meat textures (layered grain, fat
streaks, crispy dark and golden shreds). A cut slab is hidden and sinks into the
core; while it regrows it pushes back out and swaps texture raw (pink, white fat) →
seared → cooked. The carving test and the shaving launch use the former box
chunks' centres and frames (`chunk_centres()`), not the meshes, and the chunk
layout uses the same random draws as before, so carving is unchanged (15 s headless:
90 shavings before and after). Shavings are curled slice meshes; their physics is
still the hidden 3 µg ellipsoid. Lights: a key spot with shadows (`shadows=False`
turns the shadow map off; it costs ~6 ms per 960×640 frame), a cool rim light, the
warm burner light and a dimmer headlight. `visual.map.znear` is raised to 0.05 (×
extent 1 mm) so the spot-light shadow map has enough depth precision.

**Knife.** A chef's knife sits on the right front `rf_tarsus1`: a tapered 0.8 mm
blade mesh (spine, primary grind, bevelled edge, belly curving up to the point) in
polished steel (fake environment-reflection texture, high specular), a steel
bolster, and a black handle with three steel rivets. They are visual-only geoms with
no mass and no contacts, added to the fly spec just before `add_fly` (the extension
wraps `world.add_fly` once). They cannot change the dynamics: the fly still weighs
1.02 mg and no contact pair involves them. The cut test reads a hidden box
(`kebab_blade`, alpha 0, group 3) along the blade. I chose the knife's direction by
searching over the recorded stroke. With it, the tip is more than 1.6 mm ahead of
the thorax 58 % of the time and never goes below the floor.

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
test. When a chunk is cut, it is hidden (alpha, and sunk into the core), and a shaving from a pool
of 18 free bodies (3 µg ellipsoids drawn as curled slices, recycled oldest first) is launched from the
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

**Brain reactions (`--brain`, off by default).** The job turns on the full body
(proboscis joints). The proboscis follows the brain's MN9 rate: fully extended at
60 Hz, smoothed with τ 0.15 s. The job drives the proboscis itself, because the
brain-trigger path never interrupts a running action, and the carve action always
runs. Everything the brain gets is event driven (there is no food timer any more):

* **A blip per slice (touch).** Every cut sends a 40 ms pulse at 150 Hz to the
  **right-side `body_mech` set** (302 sensory-ascending mechanosensory afferents,
  sub-classes SA_DMT_*, SA_DLV, SA_MDA, SA_VTV_*; the set a whip hit on the right
  drives), labelled `KNIFE TOUCH (body_mech R)`. This is a stand-in: touch on the
  carving leg enters the VNC, which the model lacks. The brain's own leg afferents
  (`leg_sa`, SA_VTV_pro_meso_meta) are 74 of 86 gustatory, and only 2 `body_mech`
  neurons ride the prothoracic (front-leg segment) nerve. docs/SENSORY_SCREEN.md
  found that `body_mech` reaches no behaviour DN, so the touch only lights the brain
  up; it does not change behaviour. That is expected.
* **Taste on events.** A launched shaving is tracked until it lands on the tray.
  If it lands within `taste_radius` (1.5 mm) of the proboscis (the haustellum geom),
  or if the proboscis comes within `mouth_reach` (0.4 mm) of a ripe chunk or any
  shaving, the job sends a taste pulse: kind `taste`, sugar, for 0.5 s. The sugar set
  is Shiu et al.'s left labellar LB3 set, the same stand-in the taste patches use
  (docs/TASTE.md). The rate is `sugar_max_hz · hunger` (150 Hz when starving, at
  least 50 Hz). There is a 1.2 s refractory period between pulse onsets. In practice
  only landings trigger it: shavings land 1.3–2.3 mm from the proboscis (median 1.66
  mm, measured over 5 s of carving), and the proboscis never gets near the meat
  (≥ 1.6 mm). So a "taste" is "food landed within reach", not a contact. The fly
  can't reach the tray, and shavings don't collide with the fly. MN9, and with it
  the proboscis, follows these events. The food response is the connectome's.
  "Kebab" is only a label on sugar input.
* **Hunger / satiety (our phenomenological model, not connectome).** Satiety S
  (0–1, starts at 0.2) rises by 0.07 per taste pulse (a bite). It decays as
  `exp(−dt / 45 s)`, so hunger returns.
  Hunger `1 − S` scales the sugar rate, and carving speed is multiplied by
  `1 − 0.4 S`. Once S reaches 0.8 the fly is **sated**: taste events are ignored (no
  pulse, counted as `tastes_ignored_sated`) until S falls below 0.3. This
  hysteresis makes carving and feeding cycle (hungry → feeding → sated → hungry).
  Flies do lose sugar responsiveness when fed (e.g. Inagaki et al. 2012), but the
  numbers here are ours. The time constants are compressed for the show: a real fly
  takes hours to get hungry again. The HUD line reads, for example,
  `HUNGER 0.61 (model) feeding   carve x0.85   tastes 4   touches 41` (SATED while
  food is ignored).

The job trims `BrainLink.stim_log` to keep memory constant.

**Stress and startle (`--stress`).** `run_job.py --job kebab --brain --stress`
(`--stress` implies `--brain`) sets `cfg.stress.enabled`, and the Session installs
the stress layer (docs/STRESS.md). Carve speed is multiplied by
`1 + gain · (freq_mult − 1)` (gain 1: ×1.5 at arousal level 1), on top of the
satiety factor. Jobs have no whip, so **key S** (forwarded by `run_job.py` to
`job.handle_key`) **startles the chef**: `startle()` sends a `shove` on the thorax
(intensity 0.6, 0.2 s, label `STARTLE (poke)`). With stress on, the brain worker adds
the nociceptive relay (`an_walk` + `an_arousal` ascending neurons, a labelled VNC
stand-in). The relay recruits the OA neurons, the octopamine level rises, and the
carving speeds up, then calms down with the level's decay (τ 30 s). Without
`--stress` a poke only drives the right/left body afferents (no speed-up).
`startle_every_s > 0` pokes the chef automatically (headless demos).

**Verified with the real brain** (headless, 2026-09-26, FlyWire v783, 138,639
neurons, brain window off; frames rendered with `render_brain_frame` from the
captured states and checked by eye):

* *Blip per slice:* 180 s of carving gave 832 cuts and 832 touch pulses. Brain
  windows (0.1 s) containing a touch averaged **1551 spikes** against **363** for
  windows with no stimulus. The regions that rose were GNG (0.5 → 2.3 Hz mean),
  FLA_R, SAD and PRW. The raster shows a GNG burst per cut. No behaviour DN
  responded (walk / turn / MDN / GF stayed at 0), as the sensory screen predicts.
  The brain-window chips read `MANUAL BODY_MECH R`, because the window labels with
  the mapper label, not `details["label"]`.
* *MN9 follows shavings:* 35 taste pulses in 180 s, all triggered by landings.
  MN9 was 0 Hz in the 0.5 s before every pulse and peaked at **55–80 Hz** within
  0.6 s when hungry. It fell to 25–45 Hz near satiety, because the sugar rate
  scales with hunger. The proboscis extended to 40–80 %.
* *Satiety cycle* (default constants): feeding 0–33 s (S 0.2 → 0.81), sated
  33–78 s (S decays to 0.30, 138 food events ignored over the run), feeding
  78–99 s, sated 99–143 s, feeding 143–159 s, sated again. That is 3 sated bouts in
  180 s, a period of about 65 s. Carve speed ran between ×0.68 (full) and ×0.88
  (hungry). **Shaving yield hardly changes** (40–51 per 10 s): above speed 1 the
  yield is limited by the regrowth of the reachable band. Brain-off check, 15 s
  after a 5 s warm-up: speed 0.7 / 1.0 / 1.5 gave 62 / 78 / 76 shavings. What
  cycles visibly is the stroke tempo and the feeding.
* *Stress* (`--stress`, satiety speed factor off for isolation, pokes at 20, 22 and
  24 s): the arousal level was 0.12 before the pokes. It had crept up from 0
  during 20 s of touch and taste pulses: a little OA recruitment. After the pokes
  it went 0.45 → 0.74. Carve ×1.06 → **×1.37**, and the clip phase advanced at 1.34
  s/s instead of 1.05. Then it calmed down: level 0.43 and ×1.21 at 60 s (τ 30 s).
  The brain window showed walk / turn DN bursts at each poke and the PAIN/AROUSAL
  row at 0.74 with `+hit relay*`. Yield: 22 → 29 shavings per 5 s for 5 s, then
  back to 21–24 (regrowth-limited). RTF was 0.24 without stress and 0.10 with
  stress.

### mowing

`fly_simulator/jobs/mowing.py`, tests in `tests/test_jobs_lawn.py`.

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

`fly_simulator/jobs/raking.py`, tests in `tests/test_jobs_lawn.py`.

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

### dead_hang

`fly_simulator/jobs/dead_hang.py` (+ `dead_hang_assets.py`), tests in
`tests/test_jobs_dead_hang.py`. "The fly dead-hangs for its life, forever, from a
pull-up bar over a Venus flytrap."

![dead_hang](media/dead_hang.gif)

**Scene.** A fly-scale gym: a chrome pull-up bar (static capsule along x, R 0.15 mm,
8 mm long, at z = 11 mm) on black powder-coated posts, with chalk marks where the
hands go. Below it is a potted Venus flytrap (terracotta pot, peat with moss, a
rosette of small traps, a winged petiole). The main trap has two procedural lobes,
red inside with a green margin, green outside, and marginal teeth. The floor is a
black rubber gym mat, and there is a brick wall with posters ("NO PAIN / NO GAIN",
"DON'T LOOK DOWN", "HANG IN THERE"). The job camera frames the bar and the trap.

**How it hangs (physics, no welds).** The fly faces the camera, its body pitched
80° nose up, and holds on with its two front legs. The tarsi come up on the
camera side of the bar and reach over the top. The spawn / respawn pose is solved
by IK (`LegIK`: damped least squares on the 7 actuated DoFs of one leg, on a
scratch `MjData`). Targets: tarsus1 in front of the bar, tarsus5 on top. The pose
is then settled for 1 s on a scratch `MjData` (grip targets and adhesion held) and
written into FlyGym's "neutral" keyframe (free joint + leg angles, as the course
respawn does). `sim.warmup_s` is 0, so the standing warm-up never runs. The
posture is an `ActionManager` action (`HangGrip`, name `dead_hang`) that writes all
42 leg targets and the 6 adhesion commands every step. Grip = FlyGym's own tarsal
adhesion actuators (MuJoCo `adhesion` on tarsus5, gain 40: 40 µN per leg at ctrl 1 =
4 body weights; the fly weighs 10.05 µN) plus friction 1. Measured with this
posture (both legs, 10 s trials):

| adhesion ctrl per front leg | force per leg | result |
|---|---|---|
| 0 | 0 | the tarsi slide off, falls after 0.3 s |
| 0.05 / 0.10 | 2 / 4 µN | falls after 0.3 / 0.4 s |
| 0.15 / 0.3 / 1.0 | 6 / 12 / 40 µN | hangs the full 10 s |

So the hang needs roughly 1 body weight of total adhesion. With full adhesion it
hangs indefinitely (thorax 1.42 mm below the bar axis, sway < 0.02 mm). One arm is
precarious: the body swings and twists, and a leg coming back to the bar often
misses. The walking fall detector is paused for this job (it would read a vertical
fly as fallen). The job's own rule: no leg on the bar for 0.12 s and the thorax
2.4 mm below it (or 3.6 mm regardless) = a fall.

**Grip fatigue (phenomenological, ours).** Each front leg has a strength s (0–1).
The adhesion command *is* s, so the force is 40 µN × s. s drains by 0.011/s while
the leg carries load (× 2.5 when it holds alone; × (1 + stress level) with
`--stress`), capped by a capacity that decays by 0.002/s. Below ~0.86 (± jitter) the
fly **re-grips** the tired leg: it lifts it 0.15 mm off the bar and puts it back
(joint-space replay of the settled grip, 0.3 s; the other leg holds alone
meanwhile). That restores 60 % of (capacity − s) and costs 0.045 capacity. Random
**slips** (hazard 0.045/s × ((1 − s)/0.5)², × (1 + 2 × stress)) drop the weaker leg.
A gripping leg whose distal tarsi leave the bar for 0.25 s has physically slid off,
which also counts as a slip. The slipped leg hangs below the bar and flails, the
holding leg pulls up (femur–tibia +0.3 rad), the hind legs kick. After 0.8–1.8 s it
**re-grabs**: closed-loop IK, targets moving from above the bar in front down onto
the top, centred on where the body hangs now. A miss means it kicks and tries again.
Whether the fly falls is decided by the contacts: it drops when the adhesion is too
weak or a one-arm phase twists it off.

**The trap.** A trap head on a slide joint (the "lunge") carries two lobes on hinge
joints about the midrib, all driven by position servos (lobes: 12 Hz, critically
damped; props only, `gravcomp` on). Open = 55° from vertical. When the falling fly
enters the mouth, the trap **SNAPS** shut (counted CHOMP), holds 2 s, and the fly is
explicitly respawned on the bar (`job.recover("chomped")`, logged and counted). The
trap starts that respawn closed and reopens over 6 s. A fall that misses the mouth
lands on the soil or the floor ("missed the trap", an escape) and respawns after
1.5 s. Survival rate = escapes / falls. **Engineered and labelled:** the lobes are
visual meshes. The falling fly lands on an invisible static catch pad on the
midrib, and the lobes close around it without touching it (lobes that squeeze the
fly would need a multi-geom contact model and risk blow-ups). Legs and head can
stick out of the closed trap.

**Twitches and the brain (`--brain`, off by default).** Every ~14 s (random 0.6–1.4
×; key **T**) the trap **twitches**: it lunges 1.6 mm up and snaps 22° toward
closed in 70 ms, then relaxes. With a brain the trap is a `LoomingVision` source
(the lobes as spheres, the swatter's response curve for large slow objects), so the
twitch drives LC4 / LPLC2 through the looming machinery. Measured without a brain
(events captured): a twitch gives LC4 97–119 Hz and LPLC2 up to 28 Hz on both eyes
(θ up to 50°, dθ/dt ~1300 °/s); nothing is sent at rest. **Rule:** on the bar a jump
is suicide, so the giant fibre (DNp01 > 60 Hz, the jump rule's threshold) makes the
fly **FLINCH** instead: 0.35 s of full adhesion (clench), femur–tibia flexion (a
little pull-up) and hind-leg kicks, costing 0.02 strength, with a 1 s refractory
period. With `--brain-actions` the job replaces the trigger's giant-fibre check
(`_check_gf`), so no jump is ever started. `--habituation` (the brain's LC4 / LPLC2 →
DNp01 depression, docs/HABITUATION.md) should make the fly stop flinching at
repeated harmless twitches. `run_job.py` now has `--habituation`. The real-brain
path (whether a twitch's LC4 drive makes the GF cross 60 Hz) was **not run** in this
session. The flinch logic is tested with a fake brain state.

**HUD.** `DEAD HANG FLY - hanging on for dear life`, the hang streak and best,
CHOMPS, a grip bar (`GRIP [######----] 62 %`, L / R, capacity), re-grips / slips /
re-grabs / survival rate, and (with a brain) twitches / GF bursts / FLINCHES. A
banner flashes `*** CHOMP! ***`, `SLIP!`, `AAAAAAAAAH!`, `FLINCH!`.

### bowling

`fly_simulator/jobs/bowling.py` (scene, behaviour, `BowlingGame` scoring),
`fly_simulator/jobs/bowling_assets.py` (procedural meshes / textures), tests in
`tests/test_jobs_bowling.py`. HUD: `BOWLING FLY - league night, every night`.

**Scene.** A raised lane bed (1.6 mm; the fly walks on it) with a maple / pine board
texture, the black foul line, 7 target arrows, guide and approach dots and pin
spots, all drawn in one procedural texture. Semicircular gutters on both sides are
real channels (7 static facets each) that a stray ball drops into; kickbacks flank
the deck; the pit behind has a high-rolling-friction floor and a back cushion. A
masking unit ("FLY LANES / LEAGUE NIGHT - EVERY NIGHT") hides the pinsetter
housing; two neighbouring lanes with full racks, a ball-return hood and house
balls are decoration. Cosmic carpet on the floor. Scale: the ball is 3 mm (R 1.5,
0.3 mg); pins, spacing (4.2 mm) and lane width use the same scale as the ball
(8.5 in → 3 mm), but the lane is compressed ~11× (22 mm from foul line to head pin
instead of 250 mm).

**Pins.** 10 free bodies in the standard triangle (numbering 1 … 10, 7 on the
left). Collision: a flat foot cylinder, belly capsule, neck capsule, head sphere
(COM ~0.37 of the height); visual: a lathe mesh through the regulation USBC pin
profile, white with two red neck stripes. Mass 0.067 mg, the regulation ball / pin
ratio (~4.5). They touch the lane, the ball and each other, never the fly. Tuning
found: (1) with default max-mixing the ball's friction 1 and its spin pinned the
light pin to the deck and the ball stopped dead against it, so pin geoms have
priority 2 and friction 0.25 (lacquered pins); (2) FlyGym's stiff 0.2 ms contact
time constant blew up on fast pin-pin hits, so pins use 1 ms; (3) condim-6 contacts
between ball and pin (with the noslip solver) welded them together, so the ball
itself is condim 3 and its rolling friction comes from ball-only surfaces
(contype bit 64: the oiled lane 1e-4 mm, the dry approach 0.02 mm, the pit 0.25 mm).
A pin is **down** if it tilts more than 35° or is off the deck.

**The ramp (engineered, labelled).** A ball pushed at walking speed (~10 mm/s, ~100×
slower than a Froude-scaled real delivery) only nudged the head pin: measured, the
pin leaned on the ball and the ball stopped dead. So the approach is a plateau that
ends in a bowling ramp (the device kids use): 0.35 mm drop over 3 mm onto the lane
at the foul line. The fly pushes the ball over the top edge and the ball rolls down,
reaching ~70 mm/s (sqrt(10/7 g h)); from then on it is real rolling and contact
physics. Placed-ball sweeps (70–200 mm/s, aimed −3 … +3 mm) knocked down 4–6 pins on
average, at most 9; faster balls or lighter pins (0.03 mg) changed that little.

**Behaviour.** `PushPilot` (the mowing job's push loop) walks round the ball on the
approach and pushes it along the roll's target line; the waypoints are clamped
behind the ramp. When the ball is 0.5 mm past the ramp's top edge the fly stops
(**engineered stop rule**: steering speed 0 and a `freeze`) behind the foul line and
watches; a thorax past the foul line within 3 s of the release would be a **foul**
(the ball scores 0, as in real bowling; 0 so far). The ball rolls; the roll is over
when the ball is in the pit, in a gutter beside the deck, stalled (< 0.8 mm/s for
1.2 s: a dead ball) or after 15 s. The job then waits until the upright pins on the
deck are still (< 2 mm/s for 0.5 s, max 5 s) and scores the roll. Then the fly walks
back to its waiting spot (grooming first after a strike) and freezes until the
ball comes back. **Aim model** (documented "skill", default 0.75): roll 1 aims at the
pocket (y = −1 mm, between pins 1 and 3), roll 2 at the front-most standing pin;
aim error at the pins ~ N(0, 0.4 + 2 (1 − skill) mm), plus a wild throw (+N(0, 5 mm))
with probability 0.25 (1 − skill). The push itself adds its own scatter.

**Pinsetter (kinematic, labelled; every cycle counted).** After roll 1 the table
lifts the standing pins 2.6 mm, the sweep bar (mocap, visual) comes down in front of
the deck and sweeps back to the pit; every knocked pin it passes vanishes into the
housing (a teleport; stored pins are held there kinematically), and the standing
pins are put back down where they stood, upright. After a frame (or a strike, or a
10th-frame mark) it clears everything and lowers a fresh rack of 10 from the housing
onto the spots (`rack_resets`). **Ball return (labelled):** the ball is taken out of
the pit into the ball-return hood (a teleport) and pops out at the approach return
spot (x 4 mm, y ±1.5 mm) once the fly has walked clear. A reset during a roll voids
the roll (`voided`) and puts the rack back as it was before it.

**Scoring.** `BowlingGame`: standard ten-pin rules (strike = 10 + next two balls,
spare = 10 + next ball, the 10th frame's bonus balls with a fresh rack after a mark,
300 max), marks `X / - 7`, per-frame running totals (None while a bonus is
pending). HUD: game, frame / ball, score, last / best / average game, a two-line
scoreboard (`1: X =20 | 2: 7 / =30 | 3: - 4 =34 | …`), strikes / spares / gutters /
fouls / rolls / racks, and a message line (STRIKE!, DOUBLE!, TURKEY!, SPARE!,
GUTTER BALL, the 7-10 split!, GAME OVER).

**Verified** (headless, 2026-09-26, `run_job.py --job bowling --headless
--max-seconds 180 --timelapse 6 --record … --width 480 --height 320`): 23 rolls,
**52 pins (2.3 per roll)**, one complete game (**46**) and 5 balls of the next,
0 strikes, 0 spares, 3 gutter balls, 2 dead balls, 0 fouls, 23 pinsetter cycles,
11 rack resets, 22 ball returns, 19 pushes started, 9 back-away unsticks,
0 falls, 0 auto-recoveries, 0 instabilities, RTF 0.28. (That run used skill 0.6; with the
current default 0.75, a 40 s run with seed 3 gave 5 rolls, 8 pins, 0 falls, RTF 0.27.) About 7.5 s per roll, so a
game takes ~2.5 sim minutes. `run_sim.py --job bowling` runs too (3 s smoke test,
3 pins). Frames checked by eye (job renderer / recording): the push at the ramp, the
ball rolling down the lane, pins scattering, the sweep bar, the fly walking back.

**Limitations.** The fly bowls badly: strikes and spares are possible under the
scoring rules and the physics, but none happened in 180 s and no placed-ball test
reached 10 pins; slow, fly-scale pin action rarely clears the deck. The lane is
compressed, the ramp, stop rule, pinsetter and ball return are engineered (above),
and the pins are held kinematically while in the machine.

### broccoli_toss

`fly_simulator/jobs/broccoli_toss.py` (scene, viewer, sequence, blast),
`fly_simulator/jobs/broccoli_toss_assets.py` (procedural meshes / textures), tests in
`tests/test_jobs_broccoli_toss.py`. HUD: `BROCCOLI TOSS FLY - absolutely not`.

![broccoli_toss](media/broccoli_toss.gif)

A reaction-meme format at fly scale: someone is handed a plate of broccoli, looks at
it for a few seconds, then casually flicks the whole plate backward over their
shoulder without looking, and everything behind them explodes. No real people,
channels or logos: the flies are "the host fly" and "the viewer fly", the stream
layout is generic ("LIVE", "CHAT", "FOLLOW GOAL").

**Scene.** A fly-scale streamer's room (the fly is ~2.5 mm long; the seat is 2.1 mm
up, the desk 3.3 mm): dark wooden floor and a round neon rug, an acoustic-foam back
wall, a side wall with a (decorative) doorway, posters with generic art ("GG" over a
synthwave sunset, "LEVEL UP", "PRESS START"), RGB LED strips (emissive, hue cycling),
a ring light, a mic on an arm, a gaming desk with a glowing keyboard and two
monitors, a big stream TV on the side wall, a shelf with trophies and books, a
kitchenette (mini fridge, counter, microwave, a stack of plates) and a gaming chair.
The monitors show a generated stream layout (a game view, a webcam box with a cartoon
fly, a red LIVE badge, a chat column); the chat blocks are thin emissive bars that
scroll up over the chat column (six times faster after a blast) and the LIVE dot
blinks. Lights are moody purple / blue spots plus the monitor glow; the key light
casts shadows (`shadows=False` turns that off). All meshes and textures are made in
code, nothing is written to disk.

**The viewer fly (posed, kinematic: engineered, labelled).** A second NeuroMechFly
body (FlyGym's mesh model, `LEGS_ONLY` joints plus a 3-DoF neck, the passive tarsal
joints removed) is attached to a mocap mount on the seat with no free joint, pitched
50° nose-up as if leaning back. It has no actuators and no contacts, and every body is
gravity compensated; the job writes its 45 joint angles every 1 ms (and zeroes their
velocities). Its poses are joint-space keyframes solved once by damped least-squares
IK on a scratch `MjData` (rest: hind legs over the seat edge, mid legs on the
armrests; take; hold; windup; flick), blended with smoothstep. It is not a second
simulated walker, so it adds no contact physics. It wears a gaming headset (visual
geoms on its head). Its gaze is levelled at the monitors with a head pitch; it only
ever looks down at the plate during the ponder.

**The host fly** is the normal simulated fly (real walking physics, CPG controller,
fall detection). It walks the straight line from the kitchen spot to its spot beside
the chair (pure pursuit on the line), so it arrives facing the viewer: at this scale
the hybrid controller's "turn on the spot" is really arc walking with a ~2 mm radius
(measured: `turn_right` moved the fly ~2–3 mm while turning 90°), so the job plans
the facing into the path instead of turning at the end. The way back is a clockwise
loop under the desk edge that ends at the kitchen heading along the delivery line
again. The plate rides on the host's back (**kinematic carry**: the plate and florets
are placed on the thorax pose every 1 ms, no contacts) and appears there at the
kitchen.

**Sequence** (state machine in `update`, every 1 ms; one cycle is ~15 s):

1. `deliver`: host walks in with the plate; `face`: a 0.25 s stop (`freeze`); if it
   is more than 45° off it gets a few short turn bursts;
2. `present` (0.5 s) + `handover` (1.1 s): the plate lifts off the host's back and
   moves to the viewer's front "hands" (**kinematic handover**, labelled) while the
   viewer reaches for it;
3. `ponder`, **0.75–1 s** (uniform, seeded): the plate stays still in front of its head,
   the head tilts down at it and rocks (pitch +18°, roll ±14°), a mid leg taps the
   armrest, a hind leg swings; `hmm...`. The head swings back to the monitors 0.35 s
   before the end;
4. `flick` (0.17 s): hold → a tiny anticipation (`windup_s` 0.12 s, eased: the plate
   dips toward the chest and cocks a little, the torso turns 4° to its left and leans
   in, the head dips) → the **snap** (`throw_s` 0.05 s, u^2.2: velocity ~0 then a
   spike at the release): the right front leg whips up and back over the right
   shoulder (the picture's left, as in the meme), the torso twists 26° to its right
   (the viewer's mocap mount), leans back 8° into the chair and rolls the right
   shoulder up, the head jerks up and counter-turns (it never looks), the left front
   leg lets go. The plate hangs from the right front "hand", banks edge-first toward
   the throw and swings out beyond the hand; its face never turns to the webcam (no
   slow flip). At the release the plate and its 4 florets become free bodies with a
   **launch velocity set by the job (engineered, labelled)**: aimed to reach
   `target_xy` (-6.6, -5.0) ± 0.9 mm (the can pyramid) in `flight_s` = 50 ms, a fast
   flat throw (~210 mm/s; translation dominates), spinning 60 rad/s about its own
   axis; the florets get ±25 mm/s scatter. The leg does not exert this velocity.
   **Follow-through / recoil (engineered, labelled):** from the release damped
   springs (`_Springs`, stepped every 1 ms) take over: the arm overshoots and swings
   back to rest, the torso / lean / roll / head wobble back in 2–4 decaying swings,
   the hind legs kick, and the **chair recoils** (the chair is a mocap body, nudged
   back 0.1–0.2 mm and yawed, the viewer rides on it);
5. `flight`: real physics, but short: the blast goes off at the plate's first contact
   with anything but its own florets or at the `flight_s` timer, whichever comes
   first (47 ms in the seeded runs): at 30 fps that is 1–2 frames from release to
   boom, at the GIF's 15 fps one;
6. `boom` (2.6 s): **cartoon blast (engineered, labelled)**, near-instant: a white-hot
   core that is big at once and collapses in 30 ms; 80 lumpy fire ellipsoids (random
   axes and orientations, a billowy noise cube-map texture, each with its own
   direction, 0–35 ms onset, 12–30 ms growth, cooling white → yellow → orange → red →
   soot, rise, wobble and burn-out) that burst out to full size in a few tens of ms
   and fill the background on the picture's left and above; 24 darker smoke
   ellipsoids that roll up later; 40 spark streaks (capsules along their velocity),
   56 embers and a dust ring (all emissive primitives on a mocap body, animated
   vectorised); a point light at the blast and a warm **rim spot** from behind the
   viewer's right shoulder onto its back / side, the chair and the rug, both
   flickering with the fire; the viewer's materials glow for an instant. The key
   light's shadow is off while the room burns (MuJoCo's shadow map blacked out the
   emissive puffs). **One shockwave velocity kick** (an impulse) to every breakable
   prop within 9 mm: Δv = 230 mm/s × (2 mm / max(r, 2 mm))^0.6 × U(0.8, 1.2) along
   the outward direction with an upward bias, plus a random spin. From then on the
   props fly, tumble and collide with real contacts. The pool of 14 debris chips and
   the florets is launched from the blast centre (120–300 mm/s) and rains down with
   real physics. The plate itself is shattered (parked). The blast also shoves the
   viewer and the chair forward (a kick to the recoil springs);
7. `aftermath` (2.4 s): the viewer keeps watching the monitors, unbothered, the host
   stands there, the stream viewers go up (+80 + 12 % × U(0.6, 1.4)), chat scrolls
   fast;
8. `rebuild` (1.4 s, **kinematic, counted**): every prop lifts, glides and slerps
   back onto its spot; debris and florets go back to the pool behind the back wall.
   Meanwhile the host walks back to the kitchen (`walk_off`), gets the next plate
   (`fetch`, 0.6 s), and 1. again.

**Props.** 13 breakable free bodies behind the chair (3 cardboard "FRAGILE" boxes
stacked, a 3-can pyramid, a floor lamp, a speaker, a PC tower with RGB fans, a potted
plant, two trophies and an action figure on the shelf), 0.01–0.09 mg, friction 0.6,
contact time constant 1 ms (softer than FlyGym's 0.2 ms for fast hits); they never
touch the fly. While intact they are **parked kinematically**: contacts off, gravity
compensated, at rest on their spots, so the solver has no resting contacts to hold
(this took the step time from 0.5 ms to 0.19 ms: 134 → 14 contacts). They, the plate
and the florets go live (contacts + gravity) at the release; the debris goes live at
the blast. The camera is a "webcam" on the (shallow) desk under the monitors with a
wide 64° view. While the host walks it is wider and higher (azimuth 180°, elevation
-13°, 5 mm); from the handover to the end of the aftermath it glides (0.6 s) into the
meme's framing: chest-high (elevation -10°, 4.3 mm, look-at 3.45 mm up), the chair
centred with its whole back in view, slightly off-axis from the picture's left
(azimuth 172°) so the plate visibly flies back past the viewer. The blast (scaled ×3)
fills the background on the picture's left and above, but every puff is clamped to
stay behind the chair back, so the viewer stays in front of it, unbothered, and
glances off to the side afterwards, like the meme edits.

**Edit effects (a video edit, not physics; labelled `EDIT FX (not physics)` in the
HUD during the beat).** Matched against the meme edit frame by frame (a 10 fps clip:
hold, a one-frame anticipation, the arm up with the plate leaving the frame, then one
white "impact frame" with speed lines and a dark silhouette, two overexposed,
punched-in frames, then saturated fire behind the unbothered person, zoomed in ~1.4×
for the rest):

* **hit-stop / slow motion** (`time_scale`, presentation time): 0.07 s at ×0.02 (a
  near-freeze: 2 frames at 30 fps), then 0.4 s at ×0.3, ramping back to ×1 over
  0.25 s (~0.36 s of presentation time added per blast; recordings and the GIF show
  it, the physics sequence is unchanged);
* **impact flash** (counted in frames, so it never lingers): frame 1 is an "impact
  frame" (a high-contrast negative with a warm white burst out of the blast, streaked
  by the zoom blur), frames 2–4 are overexposed by ×3.4 / ×2 / ×1.3 plus a white /
  yellow lift strongest toward the blast (the viewer overexposed too), then the
  normal image; a chromatic fringe (R / B pulled apart 4 → 1 px) on frames 2–4;
* **zoom blur** out of the blast (5 taps, 8 % → 0 in ~0.2 s), **bloom** (bright pass,
  blurred at 1/4 and 1/8 size, added back warm: the glow engulfs the silhouette at
  the blast, a weaker halo while the room burns), a brief exposure lift and a warm
  grade that fades with the fire;
* **camera kick, damped shake and punch-in** (one `warpAffine`): a 3 %-of-width kick
  at the impact, the opposite smaller swing next, 7 / 5.5 Hz oscillations decaying
  with τ = 0.14 s (2–4 visible swings) plus a ±1.6° roll; the zoom punches to ×1.25
  and settles at ×1.14 while the room burns, then eases back as it rebuilds;
* **ghosting** at the hit (the previous output blended in at 45 / 30 / 15 %) and a
  **motion smear** on the throw's fastest frames (the snap and the flight: what moved
  since the last frame gets a directional blur along the throw, blended with the
  previous frame).

`post_process` costs 7–9 ms per 600×400 frame during the blast (numpy / OpenCV, measured
in the render loop), less on the smear and zoom-ease frames, nothing otherwise (identity). `edit_fx: false` turns all of it
off (no post-process, no hit-stop).

**Brain (`--brain`, off by default).** During the ponder the job sends a bitter taste
pulse every 0.5 s (0.4 s at 150 Hz: `StimulusEvent("taste", tastes=["bitter"])`, the
Shiu et al. labellar bitter GRN set LB1a–e, labelled `BROCCOLI: BITTER (stand-in:
LB1 bitter GRNs)`). **Stand-in:** the scene has one connectome brain and it belongs
to the simulated host fly; the job feeds it the viewer's broccoli so the brain window
shows the bitter pathway. Bitter drives no behaviour DN in the model (docs/TASTE.md),
so nothing moves because of it. Checked with the real FlyWire brain (138,639 neurons,
window off, 7.5 s): 8 bitter pulses in the ponder; brain states in the ponder averaged
335 spikes per window against 0 before it, and every descending group stayed at 0 Hz.
The HUD adds `bitter pulses N (stand-in ...)`.

**Counters / HUD.** Plates yeeted (the work counter), explosions, stream viewers (from
1,337, up after every blast, the last gain in brackets), `vegetables eaten: 0`
(always), room rebuilds, props launched, plus a message line (`the host fly: 'made you
broccoli'`, `hmm...`, `*flick*`, `KA-BOOM! (cartoon blast) chat goes wild: +N
viewers`, `rebuilding the room`) and the label line `(viewer: posed kinematic fly;
plate launch + blast: cartoon, see docs)`, plus `EDIT FX (not physics): hit-stop,
slow-mo x0.3, flash, bloom, shake, punch-in` during the impact's edit beat.

**Verified** (headless, 2026-09-27, `run_job.py --job broccoli_toss --headless
--max-seconds 150`): **10 plates yeeted, 10 explosions, 10 room rebuilds**, 130 props
launched, stream viewers 1,337 → 5,782, vegetables eaten 0, flights 55–64 ms, 0
flight timeouts, 0 plates lost, 4 back-away unsticks, **0 falls, 0 auto-recoveries,
0 instabilities, RTF 0.31**. (An earlier version of the walk loop orbited its end
point and needed one `stuck` recovery in 150 s; the path follower now lets the carrot
run on past the end.) `run_sim.py --job broccoli_toss --headless --max-seconds 4`
runs too. Frames checked by eye (job renderer): the host walking in with the plate
on its back, the handover, the ponder close-up, the plate on the right front leg over
the shoulder, the plate in the air over the chair, the fireball and dust ring with
props flying, the aftermath with the viewer still facing the monitors, the rebuilt
room. After the snap / edit rework (2026-09-27): `run_job.py --job broccoli_toss
--headless --max-seconds 20`: 2 plates yeeted, 2 explosions, 26 props launched,
flights 44–47 ms, 0 falls, 0 auto-recoveries, **0 instabilities**, RTF 0.26;
`run_sim.py --job broccoli_toss --headless --max-seconds 5` runs, and its `--record`
MP4 has the post-process and the slow motion in it. The throw → boom segment was
rendered at 30 fps (and 15 fps for the GIF) and checked frame by frame against the
meme clip: at 30 fps 5 frames of flick (4 of anticipation, 1 smeared snap), 2 of
flight, 2 hit-stop frames (the impact frame + an overexposed one), 2 more flash
frames, ~12 slow-motion frames while the fireball fills the background; at 15 fps
release → impact frame is one frame, as in the clip. Tests:
`tests/test_jobs_broccoli_toss.py` (10 tests, ~35 s: scene / viewer without free
joint, actuators or contacts; the full phase order and a 0.75–1 s ponder; the launch
goes backward and lands behind the chair, fast and flat, and the blast follows within
`flight_s`; the plate never turns face-on to the webcam, the torso twists and the
chair recoils, both back at rest after the cycle; the hit-stop / slow-mo schedule and
the presentation-clock lag of one blast; the post-process is identity when idle and
the impact frame is bright; the blast moves props and the rebuild puts them back;
counters and HUD; bitter pulses with a fake brain; a reset mid-cycle; the
`PresentationClock` step accounting).

**Limitations.** The viewer is posed, not simulated (no dynamics, it can't be
knocked over), the carry, handover, launch velocity, follow-through / recoil
springs, blast, debris launch and rebuild are engineered (above); the flash,
hit-stop, slow motion, shake, punch-in, blur and ghosting are edit effects. The
viewer is a fly, so the "arm" is a thin front leg: the snap reads mostly through the
plate, the smear and the torso twist. The blast's screen centre for the zoom blur /
flash is a fixed point of the webcam framing (`blast_screen`), not projected. In a
live window the hit-stop is honoured by waiting (the job runs at RTF ~0.3 anyway).
The host's arrival heading is within ~5–25° of the viewer, not exact. Blast-launched
props sometimes fly out toward the camera (invisible walls keep them in the room).
The chat on the monitors is geometry over a static texture (no text).

## Verification

The test file is `tests/test_jobs.py`: synthetic tests of the registry, steering and
recording helpers, plus short sisyphus and wheel runs covering contacts, pushing,
explicit recovery, NaN recovery, a lost boulder and the wheel spinning. `tests/test_jobs_lawn.py`
(9 tests, ~13 s) covers mowing and raking: contact bits, the grass pool (cut, stripe
colour, regrowth), row completion and the serpentine, the mower being pushed, the
mower surviving an explicit reset, rake carrying, pile counting, gusts, the constant
leaf pool, and the rake following the thorax. `tests/test_jobs_dead_hang.py` (7 tests, ~30 s)
covers dead_hang: registry, no weld constraints, the bar colliding with the fly and the trap
lobes being visual, a 6 s hang at full grip (the adhesion actuators give 40 µN × strength, no
applied forces), fatigue and a re-grip, a twitch moving the trap, GF → flinch (never a jump)
with a fake brain state, and grip 0 → the tarsi slide off → CHOMP → an explicit respawn with
the trap reopening.
Results of the long headless runs are below.

Measured on 2026-09-26 on the development machine (Apple M1), with other simulations running at
the same time:

| run | sim time | result | falls / auto-recoveries | RTF |
|---|---|---|---|---|
| `sisyphus` headless + 640×426 rolling MP4 (60 s segments, keep 1) + timelapse every 5 s | 300 s | **45 summits** (~6.7 s per cycle), 0.80 m pushed, 48 pushes started, 6 back-away unsticks, 0 boulders lost, 0 instabilities | 5 / 5 (all falls reset explicitly after 2.5 s) | 0.43 |
| `hamster_wheel` headless + 480×320 MP4 + timelapse every 10 s | 300 s | **99.4 revolutions**, 4.37 m, top speed 14.7 mm/s (~0.33 rev/s) | 0 / 0 | 0.41 |
| `--rotate` smoke test (both jobs, rebuilding the Session) | 2.5 s | the job switched as expected | 0 | – |
| `kebab` headless + 640×426 MP4 (20 s segments, keep 1) + timelapse every 3 s | 200 s | **1006 shavings**, 16 kebabs, 3.5 mg served, ~300 shavings/min steady (peak 382), ~88 of 199 chunks regrowing at any time, 0 repositions, 0 shavings lost, 0 instabilities | 0 / 0 | 0.26 |
| `kebab --brain` (full body, real FlyWire brain) | 12 s | 72 shavings; 3 food pulses, MN9 40–60 Hz on each, proboscis driven; 0 falls | 0 / 0 | 0.27 |
| `kebab --brain`, event-driven reactions (touch per cut, taste on landings, satiety) | 180 s | 832 shavings, 832 touch pulses, 35 taste pulses (MN9 55–80 Hz after each when hungry), 3 sated bouts | 0 / 0 | 0.24 |
| `kebab --brain --stress`, 3 startle pokes at 20–24 s | 60 s | arousal 0.12 → 0.74, carve ×1.06 → ×1.37 → ×1.21 at 60 s | 0 / 0 | 0.10 |
| `mowing` headless + 640×432 timelapse every 3 s (job renderer) | 185 s | **30 rows, 5 lawns**, 1251 mm² (0.00125 m²) mowed, 6627 blades cut, 30 pushes started, 1 back-away unstick, 0 instabilities | 0 / 0 | 0.47 |
| `dead_hang` headless + 640×426 event screenshots, shadows on | 180 s | **7 falls: 5 chomps, 2 missed the trap** (survival rate 29 %), best streak 39.3 s, 167 s on the bar, 5 re-grips, 3 slips (2 legs physically slid off), 8 re-grabs, 55 re-grab misses, 8 twitches, 0 instabilities | 7 explicit respawns (5 chomped, 2 missed_trap) | 0.31 with the screenshots (0.44–0.53 in earlier runs without rendering, shadows off) |
| `raking` headless + 640×432 timelapse every 3 s | 185 s | **46 leaves raked**, 2 piles completed, 2 gusts survived (47 leaves blown about, 14 out of the yard and back to the tree), 38 leaves fallen from the tree, 0 instabilities | 0 / 0 | 0.55 |
| `broccoli_toss` headless | 150 s | **10 plates yeeted, 10 explosions**, 10 rebuilds, 130 props launched, viewers 1,337 → 5,782, vegetables eaten 0, 0 flight timeouts, 4 back-away unsticks, 0 instabilities | 0 / 0 | 0.31 |

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

* dead_hang: most falls happen during one-arm phases (a re-grip or re-grab twists the
  body off the other leg) while the strength is still 0.7–0.9, not from the
  adhesion slowly running out. The fatigue drives the re-grips and slips that cause
  them. Re-grabs miss often (55 misses vs 8 re-grabs in 180 s). The trap lobes are
  visual and the catch pad is invisible (see above). The real-brain flinch path is
  untested.

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
  another respawn point can write it into the keyframe, as `fly_simulator/course` does.
* `RunMetrics` keeps one float per fall (`falls`, `recovery_times`). This grows very
  slowly and is the only per-event list; the job's own counters are O(1).
* The main app runs jobs too: `run_sim.py --job NAME [--job-config JSON]` (all app keys
  and flags available; the whip is turned off because the idle whip lies across the
  job scenes, so hit keys shove). `scripts/run_job.py` remains the standalone runner.
