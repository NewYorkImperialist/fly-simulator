# Obstacle-course mode

Endless mode is still the default. Course mode swaps the procedural terrain for a
hand-designed, timed course. It reuses the same pooled geoms, so the model is never
rebuilt and endless mode stays bit-identical until a course is installed.

```
.venv/bin/python scripts/run_course.py tutorial              # headless
.venv/bin/python scripts/run_course.py gauntlet --controller cpg
.venv/bin/python scripts/run_course.py slalom --window       # live window (Q quit, X reset, C camera)
.venv/bin/python scripts/run_course.py gauntlet --record g.mp4
.venv/bin/python scripts/run_course.py my.json --loop --laps 3
.venv/bin/python scripts/run_course.py --list                # built-ins + leaderboard
```

Other flags: `--frames-dir D --frame-x 30,80` (PNG with HUD when the thorax crosses
x), `--no-steer`, `--hit-mode shove`, `--timeout`, `--respawn-after`, `--no-log`,
`--runs-dir`, `--no-leaderboard`, `--brain` / `--brain-steer`.

## Built-in courses (`courses/*.json`)

| name | length | content |
|---|---|---|
| `tutorial` | 58 mm | low bumps, a mild ramp, two low steps, light rubble; 2 checkpoints |
| `gauntlet` | 115 mm | rubble, 3-step stairs up/down, gap, whip gauntlet (3 cracks at L2: left, right, front), ramp, low tunnel, dip, blocks; 3 checkpoints |
| `slalom` | 102 mm | 5 + 4 pillars. Pass each pillar on the side of the racing line (it alternates); 1 checkpoint |
| `brain_test` | 50 mm | two looming zones that send LC4 loom stimuli (to the brain if one is attached); 1 checkpoint |

## File format

JSON (or TOML with the same structure). Sections are laid out in order along +x,
starting `lead_in` mm after the spawn point. A `finish` is appended if it's missing.

```json
{
  "name": "my_course",
  "description": "...",
  "timeout_s": 120,
  "sections": [
    {"type": "flat", "length": 4},
    {"type": "rubble", "length": 14, "params": {"rocks": 6}},
    {"type": "checkpoint"},
    {"type": "stairs", "params": {"steps_up": 3, "step_height": 0.1}},
    {"type": "pillars", "params": {"count": 5, "spacing": 9, "offset": 2.6}},
    {"type": "finish"}
  ]
}
```

A section's `length` is a minimum. Sections whose geometry needs more room (stairs,
gap, dip, ramp, pillars) grow to fit. Unknown keys or params are rejected, which
catches typos.

Top-level options (`CourseSpec`, all optional):

| key | default | meaning |
|---|---|---|
| `seed` | 0 | random placement inside rubble / bumps / blocks |
| `track_half_width` | 8 | mm; full-width pieces (stairs, ramps, carpets) span \|y\| <= this |
| `chunk_length` | 6 | mm; chunk size while the course is installed (see "Terrain") |
| `start_x`, `lead_in` | 2, 4 | the start line (the timer starts when the thorax crosses it) and where the first section begins |
| `timeout_s` | 120 | race time limit; after it the run is DNF |
| `fall_penalty_s` | 3 | added per fall |
| `respawn_after_s` | 3 | how long the fly can stay FALLEN before it respawns at the last checkpoint |
| `respawn_penalty_s` | 0 | extra per respawn (the fall is already penalised) |
| `max_respawns` | 10 | more respawns than this in one lap = DNF |
| `gate_miss_penalty_s` | 2 | slalom pillar passed on the wrong side |
| `steer`, `steer_lookahead` | true, 6 | the heading-hold target follows the racing line (hybrid only); pure-pursuit look-ahead in mm |
| `after_finish` | stop | `stop` or `loop` (new lap from the start after `loop_delay_s`) |

Section types and params (defaults in `perpetualfly/course/spec.py: SECTION_DEFAULTS`):

| type | params | notes |
|---|---|---|
| `flat` | - | |
| `bumps` | count, height, radius, spread | gentle mounds (ellipsoids) |
| `rubble` | rocks, lumps, rock_height, rock_radius, lump_height, lump_radius, spread | rocks + dense small lumps |
| `blocks` | count, height, size, spread | yaw-rotated boxes |
| `stairs` | steps_up, steps_down, step_height, step_depth, landing | full-width steps up, a landing, equal steps down to the floor |
| `gap` / `dip` | depth, width, plateau, ramp_angle_deg | ramp up to a plateau, a slot down to the floor, ramp down |
| `ramp` | height, angle_deg, top | trapezoidal hill |
| `pillars` | count, spacing, offset, size, height, lead, first_side | slalom. Pillars sit on the centre line. The racing line passes pillar k at y = ±offset (first_side, then alternating), and each pillar is a gate judged by the fly's y when it crosses the pillar's x |
| `tunnel` | ceiling_z, thickness, wall_offset, piece | translucent roof (underside at ceiling_z, so it clears a thorax walking at ~1.1-1.5 mm) and side walls |
| `whip_gauntlet` | cracks, level, sides | red carpet. At `cracks` evenly spaced trigger points the whip cracks from `sides[k]` (left/right/front/rear/overhead/random) at `level`. In shove mode, or without a whip, you get an external shove in the matching direction instead |
| `loom_zone` | set, duration_s, repeats, interval_s | purple carpet. On entry it sends `StimulusEvent("manual", details={"set": "LC4"})` `repeats` times |
| `checkpoint` | pad | flat pad with a yellow gate. It becomes the respawn point once reached |
| `finish` | pad | red gate. Crossing it finishes the lap |

The start gate is green. Gates are two posts at |y| = 4.5 mm with a white crossbar
at 2.6 mm and a stripe on the floor.

## Race rules

* **Timer**: race time runs from crossing the start line (monotonic run time, so
  it keeps counting across respawns). The reported total is race time + penalties
  (falls, respawns, missed gates).
* **Checkpoints**: each one records a split and becomes the respawn point.
* **Respawn**: after the fly has been FALLEN (fall detector) for `respawn_after_s`,
  the course performs an **explicit reset** to the last checkpoint. The respawn
  point is written into FlyGym's "neutral" keyframe (free-joint x/y), so
  `sim.reset()` spawns the fly standing on the checkpoint's flat pad. Every reset
  hook (metrics, fall detector, whip, brain) sees a normal reset. The reset is
  logged as `course_respawn_reset` in events.csv, counted in `RunMetrics.n_resets`,
  and listed in the results (`respawns`: time, checkpoint, reason, how long the fly
  was down). Any other reset during the race (X key, the app's auto reset) also
  goes to the last checkpoint and is counted as `n_external_resets`.
* **Triggers fire once per lap.** They are not re-armed after a respawn, because
  the sim is deterministic: re-firing the crack that knocked the fly over replays
  the identical fall forever. We saw exactly that with L3 cracks, which gave 37
  respawns in a row.
* **DNF**: race time > `timeout_s`, or more than `max_respawns` respawns.
* **After the finish**: `stop` (`course.done` becomes true) or `loop`.
* **Steering** (hybrid controller only): every 2 ms the course sets the heading-hold
  target to `atan2(route_y(x + L) - y, L)` (pure pursuit on the racing line). The
  racing line is y = 0 outside the slaloms, which also keeps the fly in its lane.
  The CPG controller has no heading hold, so it can't steer. brain-steer composes
  its drive on top of the heading hold as usual.

## Outputs

* `<run dir>/course_results.json`: the course layout summary and one record per
  lap. Each record has: status (finished / dnf / aborted), time_s, penalty_s,
  total_time_s, splits, falls, respawns, gates (with the fly's lateral clearance),
  whip triggers, loom stimuli, progress, and the leaderboard rank.
* `<runs dir>/leaderboard.json`: the top 20 finished runs per course and
  controller mode (`hybrid`, `cpg`, `brain-steer`, later `policy`), sorted by
  total time.
* events.csv rows: `course_install`, `course_start`, `course_checkpoint`,
  `course_gate`, `course_trigger`, `course_loom`, `course_fall`,
  `course_respawn_reset`, `course_lap_reset`, `course_finished` / `course_dnf`.
* summary.json: a `course` block, via the chained `Session.summary_extra`.

## Terrain

`perpetualfly/terrain/sectioned.py: SectionedGenerator` is a drop-in replacement
for `ProceduralTerrain.generator`. It serves the fixed layout chunk by chunk: each
chunk gets the geoms whose centre lies in it. While a course is installed, the
chunk length is 6 mm and the window is 1 chunk behind and 5 ahead. That's the same
7 slots, and so the same pool: 10 boxes + 24 ellipsoids per slot, 12 + 8 for
spawns. Lateral re-centring is off. `validate()` rejects layouts that overflow a
slot or have geoms longer than a chunk. Long pieces (tunnel roofs, carpets) are
split automatically.

Additive changes to `chunks.py`:

* new `KIND_COLORS` entries (stairs, pillar, ceiling, tunnel_wall, start,
  checkpoint, finish, gate_bar, whip_zone, loom_zone);
* `OVERHEAD_KINDS` (`ceiling`, `gate_bar`): these are ignored by
  `ground_height_at()`, so the fall detector never mistakes a tunnel roof for the
  floor. The set is empty in endless mode, so nothing changes there.

## Integration API (for app.py)

```python
from perpetualfly.course import install_course, CourseOptions
course = install_course(session, args.course, cfg,
                        options=CourseOptions(after_finish="loop" if args.course_loop else None))
# with --course, set cfg.session.auto_reset_after_s = None (the course respawns instead)
hud += course.hud_lines()          # in hud_lines()
if course.done: quit               # in after_physics() / the loop
```

`install_course` chains `CourseRun.after_physics` onto `session.after_physics`
(respawns and lap restarts happen there, outside `sim.step`, under the physics lock
in threaded mode). It chains `finalize` onto `session.close` (writes the results;
an unfinished lap is recorded as aborted), and adds the course block to
`summary_extra`. It disables `[ ]` (difficulty) and `F` (flatten); spawn keys still
work. `session.course` points at the run, and `course.respawn_listeners` can reset
the camera. `course.uninstall()` returns to endless terrain. Looming stimuli go to
`stimulus_fn(ev)` if given, else to `session.brain.send`, else they are only
logged.

## Limitations

* The course is straight along +x. Slaloms only weave laterally (no turns or
  loops).
* Only the hybrid controller steers. On a slalom the CPG controller (or
  `--no-steer`) walks straight through and misses gates. For the hybrid
  controller with `--no-steer` we measured 5/9 gates and +8 s.
* Gates are judged by the thorax y when it crosses the pillar x. Grazing a pillar
  isn't penalised; the pillar itself is a physical obstacle.
* A respawn places the free-joint origin on the checkpoint, so the thorax ends up
  about 0.55 mm ahead of it.
* A course is sized for the fixed pool. Very dense sections need a shorter
  `chunk_length`, and `validate()` tells you when.
