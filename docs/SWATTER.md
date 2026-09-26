# Flyswatter: a threat the fly can see coming and dodge

The whip is now the pain / arousal stimulus. The **flyswatter** is the threat the fly
tries to escape with its brain: paddle → compound-eye looming → LC4 / LPLC2 → giant
fibre (DNp01, real FlyWire wiring) → escape jump + flight.

Files: `fly_simulator/interaction/swatter.py` (asset, swat, measurement, vision source,
`install_swatter`), `fly_simulator/vision/looming.py` (flat-surface sources, per-source
response, LC4 size gate; additive), `fly_simulator/actions/jump.py` (short-mode escape,
escape flight), `fly_simulator/actions/brain_triggers.py` (short mode / flight / threat
direction), `fly_simulator/brain_link.py` (low-latency pacing option),
`scripts/demo_swatter.py`, `tests/test_swatter.py`.

## 1. The asset

A classic swatter at fly scale (the fly is 2.5 mm long):

* **handle**: 14 mm yellow capsule (r 0.22 mm, 30 mg), a free body welded to the
  mocap body `swatter/hand` (the wrist). This is the whip's pattern: a body parented
  to a mocap body has no velocity, so it would hit nothing with momentum
  (docs/dev/API_NOTES.md §14).
* **paddle**: 8 × 7 × 0.15 mm red plate (6 mg), semi-transparent, with a 7 × 6 grid of
  red ribs and a yellow rim (visual geoms). It sits on a **neck hinge** (stiffness
  3·10⁴ µN·mm/rad, damping 40, ±45°), pre-bent 15° so it lands flat. The flex limits
  the crushing force and lets fast swats whip a little, like a plastic swatter.
* Only the plate collides: with the fly (`FLY_BIT`) and with the ground / terrain
  (`TERRAIN_BIT`), using FlyGym's stiff contact parameters and `priority 1`. It visibly
  slaps the floor when it misses.

**A swat** moves only the hand. The fly moves only through contact forces.

| phase | what happens |
|---|---|
| `raise` (0.30 s) | from the parked pose (up high, behind-right) to the raised pose: handle 75° up, the paddle ~23 mm above and ~18 mm behind the aim point |
| `pause` (per level) | the telegraph. The raised paddle hovers and tracks where the fly will be (see aim) |
| `slam` (per level) | the handle rotates about the wrist and speeds up the whole way (angle ∝ t²) until the plate is flat on the ground at the aim point. The aim is frozen at slam start |
| `press` (30 ms) | the hand is held 3° past flat and the neck flexes |
| `lift` (0.12 s), `back` (0.30 s) | lift to 55°, then return to the parked pose |

**Aim**: the ground point under the thorax, plus the fly's smoothed velocity × the time
left until impact (capped at 6 mm), so a walking fly is led. If the fly stops, turns or
jumps after the aim is frozen, it can escape. The hand follows a rate-limited tracker
(the whip's: ≤300 mm/s, ≤2·10⁴ mm/s²), so the mocap target never jumps.

**Levels** (`SwatLevel(name, slam_s, pause_s)`, tuned with the real brain, see §4):

| level | name | slam | pause |
|---|---|---|---|
| 1 | lazy | 0.50 s | 0.40 s |
| 2 | normal | 0.36 s | 0.25 s |
| 3 | quick | 0.22 s | 0.15 s |
| 4 | lightning | 0.07 s | 0.08 s |

Sides: `rear` (default: the handle comes from behind), `left`, `right`, `front`,
`random`.

**Measurement** (the whip's pattern). A post-step hook sums the world-frame contact
forces between the plate and the fly geoms (`mj_contactForce` rotated by the contact
frame) × dt, over slam + press + lift. It records the peak force, the bodies hit, the
first plate-ground contact (the slap) and its impulse, the plate speed and the fly
position at the first contact. At the end of the lift, `SwatEvent` goes to
`Swatter.listeners` with `outcome`:

* `hit`: impulse ≥ 0.02 µN·s.
* `grazed`: a hit below 30 µN·s while the fly was escaping (the plate caught a leg or
  the abdomen of a fly already on its way). A full swat on a standing fly is
  ~40–800 µN·s.
* `dodged`: no hit, and the fly jumped, or it was under the predicted footprint at
  slam start but not when the plate landed.
* `miss`: no hit, and the fly was never under the plate.

The fly survives any swat. There is no damage model: a squashed fly (peak forces
10³–10⁴ body weights) walks on.

## 2. Vision: the paddle as a flat looming source

`VisualSource(quads=...)` is new and additive. The callable returns parallelogram
corners `(Q, 4, 3)`. `quad_view` cuts each quad into 6 × 6 patches of two triangles.
Each triangle's solid angle is exact (Van Oosterom & Strackee 1983) and weighted by the
eye's field-of-view mask at its centroid, so a paddle that fills the dorsal sky, or
slides into the rear blind sector, is handled patch by patch. The distance is exact
(point to parallelogram). The test checks the solid angle against the analytic value
for an on-axis square: 4·asin(a²/(a²+d²)), to 1e-6.

`VisualSource(response=...)` gives a per-source response. The whip's `LoomResponse()`
only fires above 12 000 deg/s, which is tuned for a millisecond lash. The paddle is big
and comparatively slow: its expansion reaches ~500–5000 deg/s in the last
50–250 ms. That is the regime of the looming stimuli used to characterise LC4, LPLC2
and the giant fibre (von Reyn et al. 2017; Ache et al. 2019: l/v 10–80 ms, take-offs
at angular sizes of ~20–60°). `swatter_response()`: LC4 = 200·tanh((θ̇ − 300)/1500) Hz,
gated to θ ≥ 12°. The new `LoomResponse.lc4_theta0` defaults to 0, so the whip is
unchanged. The gate matters: while the fly walks, a far, edge-on paddle jitters across
the edge of the dorsal / rear field of view at up to 1500 deg/s with θ < 10°. LPLC2 =
200·ramp(θ; 18°, 55°)·ramp(θ̇; 150, 600 deg/s). Update cadence: 0.5 ms during the
slam, 2 ms while raised or returning.

Result: no loom events while raising or pausing. During the slam, both eyes are driven
to 150–200 Hz. The first loom event comes 0.27 s (L1), 0.20 s (L2), 0.09 s (L3) and
0.03 s (L4) before the plate lands, with the handle from behind.

## 3. Making dodging physically possible

The whip study found that the jump fired 0.08–0.19 s *after* contact. With the old
pipeline, a swat is never dodged (table in §4, row "old pacing"). Three changes were
needed:

**(a) Short-mode escape (`Jump(mode="short")`).** von Reyn et al. 2014 (Nat Neurosci
17:962) describe two take-off modes. In the **long mode**, the fly raises its wings and
adjusts its posture (~25–40+ ms, often more) before the TTM kick. In the **short
mode**, an early, strong giant-fibre spike drives the TTM directly: take-off comes
~5–10 ms after the GF spike, is less stable, and the flies often tumble. `Jump`'s
100 ms crouch is the long mode. In short mode the crouch shrinks to
`short_prep_s = 6 ms` of leg set-up and the stroke follows at once.

* Measured take-off after the trigger: **6.3–14.5 ms** (8 gait phases), against
  100–107 ms in long mode.
* Without flight, 7/8 short-mode jumps tumble (landing tilt 140–170°), like real
  short-mode escapes.
* `BrainActionTriggers` picks short mode when the GF rate is ≥ `jump_short_hz`.
  `install_swatter` sets this to 60 Hz, the jump threshold itself, so every
  GF-triggered escape runs in short mode while the swatter is installed. In our
  readout, the first GF window above threshold is at 75–200 Hz (20 ms windows). Set
  `short_hz` higher (e.g. 125) to keep long mode for weak bursts.

**Escape flight (`Jump(flight_assist=True)`; a documented physics emulation).** A jump
alone is a vertical 3 mm hop that lands ~1 mm from where it started, still under an
8 × 7 mm paddle. Real escapes continue in flight. Wings are not simulated
aerodynamically, so after take-off an external force on the thorax (`xfrc_applied`,
our contribution tracked and removed exactly on landing, end, cancel and reset) does
the following:

* It servoes the thorax velocity to 180 mm/s along `escape_dir` + 20 mm/s climb for
  0.1 s, then sinks at 50 mm/s.
* The force is capped at 4 body weights, gravity support included.
* A PD attitude stabiliser (15 Hz) stands in for the halteres and wing steering.
* The assist stops at the first foot contact after the powered phase, and the normal
  touchdown / landing logic takes over.

Result: 16/16 upright landings, 22–26 mm from the take-off point (~0.25 s airtime).
The trigger's `threat_fn` (set by `install_swatter` to the paddle centre while a swat
is in progress) sets `escape_dir` = away from the paddle, horizontally. This is
retinotopic information the looming detectors have; real flies jump away from the
threat (Card & Dickinson 2008). A threat straight overhead gives the heading. Both
options are off by default: the J key, and the O key without the swatter, are unchanged.

**(b) Brain latency (`brain_link.py`, pacing only).** Latency chain for a GF-triggered
jump:

1. Loom event: stamped at fly time t and applied by the brain exactly at t.
2. GF integrates: the first above-threshold GF window ends 20–80 ms after the first
   loom event. This is neural (the looming drive starts weak at θ ≈ 12°).
3. The window is published: its end is up to `window_s` after the spike.
4. The state reaches the body: the brain can only run to T once the fly chunk
   containing T has sent its clock mark, and the state is polled after the *next*
   chunk (+1–2 physics chunks).

Changes (all through existing / new `BrainLinkConfig` fields; defaults unchanged):

* `window_s = 0.02` instead of 0.1. The GF rate is read per window. At 0.02 s its
  resolution is 25 Hz per spike (2 DNp01 neurons), and the 60 Hz threshold needs 3
  spikes in 20 ms.
* New `sync_wait_s` (0.05 recommended) + `sync_loom_only` (default True): after the
  clock mark, `update()` waits up to that much wall time for the brain to publish the
  window ending in the last `window_s`. This removes one physics chunk of latency
  (measured: state arrival 20 ms → 10 ms after the window end at 10 ms chunks). It only
  waits while a visual `loom` stimulus is active, so normal running costs nothing.
  Measured wall cost: the demo trials ran as fast as before (~7 s per swat).
* **Early publish (done since): the brain fast path** (`BrainLinkConfig.fast_path`,
  default on; docs/BRAIN.md "Latency"). The worker sends a trigger the moment the
  GF's trailing 20 ms rate crosses the jump threshold, instead of at the window
  end. While a loom is active, the link waits until the brain has reached the
  clock mark it just sent. Measured over 42 trials (below): GF crossing → jump
  trigger went from 14–21 ms median (8–35 ms) to **2–3 ms (0–5 ms)**. The jump is
  still caused by the same real GF spikes over the same threshold.

  Real brain, `demo_real_vision.py --sim` (5 ms chunks), swatter L1–L3 from behind
  and from the left, 3 repeats each (the same start phase, so the repeats differ
  only by brain noise). Times are medians relative to slam start. GF crossing is
  the same in both runs up to noise; 18 swats per sense and configuration:

  | sense | configuration | GF crossing → jump | dodged | grazed | hit |
  |---|---|---|---|---|---|
  | geometric | window readout | 21 ms | 2 | 7 | 9 |
  | geometric | **fast path** | **2 ms** | **8** | 3 | 7 |
  | real vision | window readout | 14 ms | 0 | 1 | 17 |
  | real vision | fast path | 3 ms | 0 | 1 | 17 |

  Per level (geometric, window → fast): L1 rear 3 hits → 3 hits; L2 rear 1 dodge →
  2 dodges; L3 rear 2 grazes + 1 hit → **3 dodges**; L1 left 1 dodge + 2 grazes →
  3 hits; L2 left 3 grazes → **3 dodges**; L3 left 3 hits → 3 grazes. Earlier is
  not always better. At L1 the plate lands ~200 ms after the jump, and the short
  hop (no escape-flight emulation in this demo) can land the fly back under it.
  Real vision is not helped, because its loom events arrive at or after the plate
  lands from behind (the rear blind field, VISION.md §6). The fast path cannot fix
  that.

**(c) The telegraph.** Raise (0.3 s) + pause (0.08–0.4 s) + an accelerating slam gives
the fly its warning in the last part of the slam, like a real swatter. With the handle
from behind, the raised paddle sits at the edge of the rear blind sector, so the
visible warning is the slam itself. From the side, the raised paddle is visible, but
the loom comes later (lower elevation, see §4).

## 4. Results with the REAL brain (headless)

`scripts/demo_swatter.py` runs the full FlyWire brain (one worker, paced to the fly).
It uses steering, `--brain-actions` logic (`ActionManager` + `BrainActionTriggers`, GF >
60 Hz → jump), `window_s = 0.02` and `sync_wait_s = 0.05`. Each swat starts from a fresh
reset and 1 s of walking, at 4 gait phases (+10.5 ms each). The handle comes from
behind, on flat ground. All times are relative to the plate landing (first plate-fly
contact for hits, plate-ground contact otherwise).

    .venv/bin/python scripts/demo_swatter.py --levels 1,2,3,4 --phases 4 --compare --frames <dir>

| level (slam) | vision | hit | grazed | dodged | median loom onset | median GF > thr (window end) | median jump trigger | median take-off latency | median impulse µN·s |
|---|---|---|---|---|---|---|---|---|---|
| 1 lazy (0.50 s) | on | 0 | 0 | **4/4** | −0.274 s | −0.202 s | −0.192 s | 6 ms | 0 |
| 1 lazy | off | 4 | 0 | 0 | – | – | – | – | 42 |
| 2 normal (0.36 s) | on | 0 | 1 (0.9 µN·s) | **3/4** | −0.203 s | −0.149 s | −0.140 s | 13 ms | 0 |
| 2 normal | off | 4 | 0 | 0 | – | – | – | – | 157 |
| 3 quick (0.22 s) | on | 4 | 0 | 0 | −0.090 s | −0.054 s | −0.044 s | 11 ms | 85 (vs 260 blind) |
| 3 quick | off | 4 | 0 | 0 | – | – | – | – | 260 |
| 4 lightning (0.07 s) | on | 4 | 0 | 0 | −0.033 s | −0.003 s | +0.012 s | 20 ms | 251 |
| 4 lightning | off | 4 | 0 | 0 | – | – | – | – | 245 |

Per swat (vision on): GF peaks at 175–225 Hz. The GF-above-threshold state arrives at
the body 10 ms after its window ends. Dodging flies are 13–20 mm from the aim point
when the plate lands, after 23–25 mm escape flights, all landing upright. L3: the fly
takes off 23–43 ms before landing and is caught in the air, with less impulse (50–90
µN·s, once 440 µN·s when it jumped into the plate). L4: the jump fires after contact.
With vision off, nothing ever warns the fly: 16/16 hits.

**Which change matters.** L1 + L2, 4 phases each, vision on:

| configuration | L1 dodged | L2 dodged | median trigger (L1 / L2) |
|---|---|---|---|
| old pacing (window 0.1 s, no sync) + long-mode jump | 0/4 (4 hits) | 0/4 | −0.103 / −0.040 s, take-off +0.10 s later |
| old pacing + short mode + flight | 0/4 (1 graze) | 0/4 | −0.025 / −0.022 s |
| new pacing + long mode + flight | 0/4 (1 graze) | 0/4 | −0.139 / −0.110 s, take-off +0.10 s later |
| **new pacing + short mode, no flight (default)** | 2/4 | 4/4 | −0.180 / −0.140 s (the tumbling 2–6 mm hop apparently carries the fly out from under the 7 mm wide plate; not checked frame by frame) |
| new pacing + short mode + flight (emulation, opt-in) | **4/4** | **3/4** (+1 graze) | −0.192 / −0.140 s |

You need both the latency fix and the short mode. Flight makes the escape reliable and
directed.

**Sides** (left / right handle, levels 1–3, 2 phases each): L1 3 dodged + 1 grazed; L2
1 dodged, 2 grazed, 1 hit; L3 2 hits + 2 grazed. From the side, the loom comes later:
median −0.18 s (L1) and −0.10 s (L2), against −0.27 / −0.20 s from behind.

This is plausible for real flies: slow, telegraphed swats are usually dodged, and a fast
one lands. Drosophila's looming-escape latencies are ~100–200 ms from the stimulus (GF
short-mode take-offs faster).

**Frames checked** (renderer PNGs, follow camera at 18 mm, 480 × 320):

* Dodge (L1): at −62 ms the fly is airborne and heading away, with the paddle coming
  down behind it. At −2 / +18 ms the fly is in flight in open space and the paddle is
  out of frame behind it.
* Grazed (L2): the plate lands and the fly's legs stick out from under its edge.
* Hit (L3): the plate is flat on the fly (legs visible through the translucent mesh),
  slightly propped up by the body.
* Earlier run, hit (L2): the paddle comes down on the standing fly; after the lift the
  fly crawls out from the plate's edge.

The grid, the yellow rim and the handle read clearly as a flyswatter.

## 5. Integration API (app wiring later; app.py / config.py untouched)

```python
from fly_simulator.interaction.swatter import Swatter, SwatterConfig, install_swatter
sw = Swatter(SwatterConfig())                       # bodies must exist before add_fly:
session = Session(app_cfg, world_extensions=[sw.extension], brain=link)
h = install_swatter(session, swatter=sw)             # vision + brain + escape wiring
h.level = 2; h.swat()                                # or h.handle_key("v") / ("V")
sw.listeners.append(fn)                              # SwatEvent per swat
```

`install_swatter(session_or_parts, cfg=None, *, swatter, brain_link, vision=True,
looming=None, looming_cfg=None, escape=True, short_hz=60, flight=False,
hit_ref_impulse_uNs=100, say=None)` does the following:

* attaches the swatter;
* uses `session.ground_height` for the aim height and `session.actions` for the
  `jumped` flag;
* adds the paddle as a looming source, to `looming` (e.g. the whip's `LoomingVision`)
  or to a new one sending to the brain link;
* with `--brain-actions` triggers, sets short mode and `threat_fn` (the escape-flight emulation only with `flight=True`);
* sends hits to the brain as `StimulusEvent("hit", side="top")`;
* prints `format_swat_event` lines via `say`.

It raises if the extension is missing, and warns if the brain link uses
`window_s > 0.03` or no `sync_wait_s`.

Suggested app wiring:

* keys (`SWATTER_KEYS`): **V** swat from behind, **Shift+V** swat from a random side,
  1–4 strength while in swatter mode (or make it a third hit mode for H: whip → shove
  → swatter);
* config keys: `swatter.enabled`, `swatter.level`, `swatter.side`, plus
  `brain.window_s = 0.02` and `brain.sync_wait_s = 0.05` when the swatter is on;
* events.csv: a `swat` row per `SwatEvent` (`outcome`, impulse, `t_ground`, `jumped`);
* the auto perturber / fall-detector gate should treat a swat like a whip crack.

## 6. Limitations

* **Escape flight is an emulation**: an external force with a velocity servo and
  attitude stabiliser, not wing aerodynamics. Short-mode jumps without it tumble.
* **Short vs long mode is a rate threshold**, not the real spike-timing competition
  between the GF and the non-GF pathways. With `install_swatter`'s default (60 Hz)
  every GF escape is short mode.
* The GF latency after loom onset (20–80 ms) comes from our LC4 / LPLC2 response
  function feeding Poisson drive into the LIF model. The response constants are
  plausible, not fitted to recordings.
* The field-of-view model has a rear blind sector (az > 165°, below 70° elevation). A
  swatter raised straight behind the fly is barely visible until it comes down.
* No damage model: squashed flies walk on. Crushing contact forces reach 10⁴ body
  weights but stay numerically stable.
* The swatter does not re-aim during the slam and ignores terrain except for the aim
  height. Rough terrain was not tested.
* The brain wait (`sync_wait_s`) costs wall time when the brain is slower than the fly.
  In the live app it can lower the frame rate while a loom is active, so it is off by
  default.
* The results are from 4 gait phases per level on flat ground (one brain seed). Rates
  are indicative, not statistics.


## Default: no escape-flight emulation

The escape-flight push is an external force, not wing physics, so it is **off by default** (project rule: don't fake physics). Dodges come from the real short-mode jump alone (6/8 of L1–L2 swats in the table above). Opt in with `install_swatter(..., flight=True)` or `demo_swatter.py --flight`. Real escape flight should come from wing aerodynamics later.
