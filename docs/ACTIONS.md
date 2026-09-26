# Action library (`perpetualfly/actions/`)

Body behaviours beyond walking: jump, freeze, groom, back away, turn in place, wing
raise, proboscis extension. Keys can trigger them now; the connectome brain can
trigger them later. Every number below was measured on this machine (FlyGym 2.1.0,
MuJoCo 3.9, timestep 1e-4 s, flat ground, default config, hybrid controller) with
the scripts named in each section.

## 1. What FlyGym 2.1 / NeuroMechFly supports (research)

### Joints beyond the legs

* `flygym.anatomy.JointPreset` has `ALL_POSSIBLE`, `ALL_BIOLOGICAL`, `LEGS_ONLY` and
  `LEGS_ACTIVE_ONLY`. `ALL_BIOLOGICAL` = every connected segment pair in
  `ALL_CONNECTED_SEGMENT_PAIRS`, with 3 DoFs on the non-leg joints:
  `c_thorax-c_head`, `c_head-c_rostrum`, `c_rostrum-c_haustellum` (proboscis,
  `PROBOSCIS_LINKS = [rostrum, haustellum]`),
  `c_thorax-c_abdomen12-…-c_abdomen6` (`ABDOMEN_LINKS`), `c_head-{l,r}_eye`,
  `c_head-{l,r}_pedicel-funiculus-arista` (antennae, `ANTENNA_LINKS`),
  `c_thorax-{l,r}_wing`, `c_thorax-{l,r}_haltere`, and the legs.
  `ActuatedDOFPreset` has only `ALL`, `LEGS_ONLY` and `LEGS_ACTIVE_ONLY`. No preset
  actuates just "legs + wings".
* `make_locomotion_fly` (in `flygym_demo.complex_terrain.common`) = `LEGS_ONLY`
  skeleton, with `fly.add_actuators(LEGS_ACTIVE_ONLY, POSITION, kp=45, forcerange=±65)`
  and `add_leg_adhesion(gain=40)`. Head, proboscis, antennae, wings, halteres and
  abdomen are rigid. With `fusestatic` the head is fused into the thorax body.
* **Extra joints without breaking locomotion (done):** `Skeleton(anatomical_joints=
  LEGS_ONLY joints + extra AnatomicalJoints)` works. For the proboscis the skeleton
  also needs `AnatomicalJoint("c_thorax", "c_head", AxesSet([]))`. That edge has no
  DoF, so the head stays fused, but the tree stays connected for the DFS in
  `Skeleton.iter_jointdofs`. The trap: every actuator added through
  `fly.add_actuators(..., POSITION)` lands in
  `get_actuated_jointdofs_order("position")`, and the locomotion controller writes
  a 42-vector there, so it would break. `perpetualfly.actions.body.make_action_fly`
  therefore adds the wing and proboscis position actuators straight to the MjSpec
  (`flygym.utils.mjcf.add_actuator`), outside FlyGym's registries. The controller
  still sees exactly 42 leg actuators.
  Walking check: 2 s of walking gives x = 27.599 mm with the extra joints and
  27.603 mm without, mean thorax z 1.125 vs 1.124 mm (`nq` 73→81, `nu` 48→56).
* Wing and proboscis sign conventions (checked by rendering, with the right side
  mirrored by FlyGym): wing **yaw +** raises the wing vertically, wing **roll +**
  spreads it sideways, and wing pitch rotates it about its long axis. **Rostrum
  pitch −** swings the proboscis down, **haustellum pitch +** unfolds the tip.
  Zero = the mesh rest pose (wings folded, proboscis retracted).

### Recorded behaviour data

* FlyGym 2.1 ships only **walking** kinematics:
  `flygym_demo/complex_terrain/assets/single_steps_untethered.pkl` (the single-step
  splines behind `PreprogrammedSteps`, from v1's `210902_pr_fly1`),
  `single_steps_flybody.npz`, `spotlight_data/assets/spotlight_behavior_clip.npz`
  (2 s at 330 fps, Spotlight mocap, all 6 legs moving = walking),
  `ball_flybody_data/assets/ball_flybody_clip.npz` (1 s, generated from IK), and
  `muscle_imitation/assets/mocap/{qpos,qvel,xipos,xivel}/0002.npy` (one short
  walking clip for the musculoskeletal "FlyMimic" arm). Tutorial 2
  (`2_replaying_experimental_recordings.ipynb`, `scripts/replay_behavior_*.py`)
  replays the Spotlight walking clip. No grooming data ships with it. The legacy
  `flygym-gymnasium` repo has only walking (`data/behavior/210902_pr_fly1.pkl`,
  `single_steps_untethered.pkl`).
* **NeuroMechFly v1** (`NeLy-EPFL/NeuroMechFly`, Lobato-Rios et al. 2022) has a
  **grooming** recording:
  `data/joint_tracking/grooming/fly1/df3d/joint_angles__180921_aDN_CsCh_Fly6_003_SG1_behData_images_images.pkl`.
  It is a tethered fly with its antennal descending neurons (aDN) activated by
  CsChrimson, tracked with DeepFly3D, at 2 kHz, 9 s long. In 0–3 s the front legs
  move (7–12° std) while the mid and hind legs are still (<0.3°): front-leg rubbing
  below the head (tarsi ~1 mm below the thorax, crossing the midline), then
  head / antenna sweeps (tarsi up at head height, 1.5–2.5 s). From 6 s it walks.
  **We use this clip** (see §3 Groom).
* The conversion (`perpetualfly/actions/convert_grooming.py`): the DeepFly3D angles
  follow the legacy joint convention of
  `flygym/assets/model/neuromechfly/legacy/flygym1_deepfly3d_rollyawpitch.xml`
  (key mapping as in NeuroMechFly's `kinematic_replay.py`). Its leg body frames are
  identical to FlyGym 2.1's at zero angles (positions match to 1 µm, 0° rotation
  difference). For each frame, forward kinematics in the legacy model is followed
  by least squares for the 2.1 yaw-pitch-roll angles that reproduce every leg
  segment's orientation and position. Tarsus5 error ≤ 0.7 µm. The front legs from
  0–3 s at 200 Hz are stored in `perpetualfly/actions/data/grooming_front_legs.npz`
  (32 kB).

### flybody (FlyGym 2.1 experimental `FlyBody`, Vaxenburg et al. 2025)

Noted, not adopted. `flygym/assets/model/flybody` has 3-DoF wings
(`c_thorax-{l,r}_wing-{yaw,roll,pitch}`, motor gains 300/200/100), haltere, head,
antenna, rostrum/haustellum/labrum, abdomen joints, labrum adhesion, and MuJoCo
fluid parameters in `fruitfly.xml` (density 0.00128, viscosity 0.000185). The
original flybody flies with a learned wingbeat policy and the fluid model. FlyGym
2.1 exposes the body but no flight controller, and it is a different body (units,
morphology, gains) from our NeuroMechFly.

### Can the legs jump? (physics)

The giant fiber (DNp01) excites the TTM motor neuron. The TTM snaps the **middle**
legs' trochanter down (coxa-trochanter extension), and the leg extends within a
few ms (von Reyn et al. 2014). Our mid legs have the same joints
(`lm_coxa-lm_trochanterfemur-pitch`, `lm_trochanterfemur-lm_tibia-pitch`)
driven by position actuators with kp = 45 µN·mm/rad and a ±65 µN·mm torque limit
(it is a torque: hinge joints). The fly weighs 10.05 µN.
Measured during our default jump stroke (`JumpParams()`, 15 ms target step):
peak vertical ground reaction **148 µN = 14.7× body weight**, mid-leg torques
reach the 65 µN·mm limit (saturated in ~8 % of the stroke steps), peak
mechanical power of the mid legs ≈ 0.27 mW, vertical speed after the stroke
**~150 mm/s**. **The default actuators can produce a real jump.** With boost 2
(kp and force limit ×2), the peak reaction is 257 µN (25.6× BW), and with ×3 it
is 366 µN.

## 2. Framework

```python
from perpetualfly.actions import ActionManager, Jump, Freeze, Groom, make_action
mgr = ActionManager(sim)          # sim = perpetualfly.simulation.Simulation
mgr.trigger(Jump())               # starts at the next physics step
mgr.trigger(make_action("freeze", duration=2.0))   # replaces a running action
mgr.cancel()                      # blend back to walking now
mgr.busy, mgr.active_name, mgr.phase()             # e.g. True, "jump", "flight"
mgr.listeners.append(fn)          # fn(ActionEvent(kind="start"|"end"|"cancel", name, time, info))
```

* **How it takes over.** `Simulation.step` runs `controller.step_and_apply()`
  (CPG / hybrid targets → `data.ctrl`) and then the `pre_step_hooks`. The manager
  owns one pre-step hook. While an action runs, it overwrites the 42 leg position
  targets with `w·action + (1−w)·controller` and switches the 6 adhesion
  actuators (to the action's value while `w ≥ 0.5`). `w` ramps up over the
  action's `blend_in`, and after the action it ramps down over `blend_out`
  (smoothstep), so the fly hands back smoothly into the live gait. The CPG keeps
  running underneath, so walking resumes in whatever phase the rhythm has reached.
  Drive actions (back away, turn) instead replace the hybrid controller's
  descending signal `[left, right]`. The manager wraps
  `controller.descending_signal` with an instance attribute, and the brain link's
  `signal_filter` still runs inside it.
* **Never teleports.** Actions write only `data.ctrl`. The one documented
  exception is the optional jump boost (below), which temporarily scales
  `actuator_gainprm/biasprm` (kp) and `actuator_forcerange` of the stroke
  actuators. It is restored at the end of the stroke, and also on
  cancel / replace / `sim.reset()`.
* **Reset.** A `pre_reset_hook` aborts the running action at once: no fade,
  physics parameters restored, a `cancel` event is sent.
* Idle cost: one Python call per physics step. With the manager attached but idle,
  the trajectory is bit-identical to a sim without it
  (`test_idle_manager_does_not_change_walking`).
* `control_every_steps > 1`: the blend then mixes with the previous step's ctrl
  (an exponential smoothing), which still converges; the default is 1.
* Thread safety: call `trigger` / `cancel` in the physics thread or under the
  app's `runner.locked()`.

Files: `base.py` (manager, `Action`, `ActionCommand`, `BodyIndex` joint / contact
helpers), `jump.py`, `behaviours.py` (freeze, groom, drive, wings, proboscis),
`body.py` (`make_action_fly`), `registry.py` (`make_action`, `ACTION_KEYS`),
`convert_grooming.py`, `data/grooming_front_legs.npz`.

## 3. Actions and measured results

`scripts/demo_actions.py` runs every action during walking and prints the
metrics. It also takes `--record out.mp4` (20× slow motion during the jump, 2×
during other actions) and `--frames DIR` (PNG key frames, plus a front close-up
for the groom / wings / proboscis frames).

### Jump (escape), key J

Phases: **crouch** (100 ms: the targets blend from the controller's current pose
to a crouch, with mid/hind coxa-trochanter −20° and femur-tibia +20°; adhesion
on, so swing legs come down) → **stroke** (15 ms: step targets to mid-leg
extension, coxa-trochanter +45°, femur-tibia +25°, mid thorax-coxa +12°; adhesion
released on all legs) → **flight** (legs swing to a landing pose; waits for ≥3
legs or a body part on the ground, 400 ms max) → **landing** (150 ms: standing
pose, adhesion on) → 200 ms hand-back to walking.

**Postural feedback (important).** At stroke onset the jump reads where the mid
tarsi are relative to the centre of mass. The pre-jump posture depends on the gait
phase, because the stance feet adhere while the fly crouches. Without feedback,
the take-off pitch rate varied from −25 to +52 rad/s across gait phases (−375 rad/s
per mm of mid-foot position, r = −0.95). Most jumps flipped the fly: 1–3 of every
4 phases, and 0/4 upright at several settings. The fix shifts both mid coxae by
`posture_gain · (x_mid − foot_x_ref)` = 43°/mm · (x − 0.23 mm). This is the same
role as the fly's pre-take-off postural adjustment of the mid legs, which sets the
take-off direction (Card & Dickinson 2008). A left/right-*asymmetric* correction
made roll much worse, so it is symmetric.

Results over 16 gait phases (triggers 0.30–0.49 s after reset, flat ground,
scratch script `jump_stats.py`):

| variant | upright landings | apex Δz (thorax) | vz after stroke | airtime | horizontal distance | median max tilt | speed 0.5 s after landing |
|---|---|---|---|---|---|---|---|
| **default, no boost** | **16/16** | **3.04 ± 0.17 mm** (2.78–3.27) | 147 mm/s | 54 ms | 1.2 mm (−1.1 mm, slightly backward) | 37° | 10.1 mm/s (14 mm/s by 0.6–1 s) |
| boost ×2 (same trim) | 0/16 | 4.29 ± 0.21 mm | 187 mm/s | – (tumbles) | 3.3 mm | 170° | – |
| boost ×2, `mid_thc=24` | 12/16 | 3.89 ± 0.22 mm | 167 mm/s | 59 ms | 1.2 mm | 31° | 6.3 mm/s |
| boost ×3, `mid_thc=26` | 5/16 | 4.4 mm | 185 mm/s | – | 2.9 mm | 167° | – |

Apex Δz is the thorax apex relative to the thorax height at the trigger (standing
~1.1 mm), so the thorax reaches ~4.1 mm above the floor, about 1.5 body lengths
of height gain for a 2.5 mm fly.

**Boost (documented physics change, off by default).** `Jump(boost=b)` multiplies
kp (gain and matching bias) and the force range of the stroke actuators (4 by default: mid
coxa-trochanter + femur-tibia, more if hind / front legs are used) by `b` for the 15 ms stroke only. It is restored
right after the stroke, on cancel and on reset (`test_jump_boost_is_temporary_and_reset_restores`).
Why it exists: the real TTM is a very strong, fast muscle. Take-off speeds of real
GF escapes are ~0.3–1 m/s, versus ~0.15 m/s here, and a 45 µN·mm/rad servo is not
a model of it. Honest result: the boost adds only ~35 % height (4.3 vs 3.0 mm),
because leg length and extension range limit the stroke more than force. It also
multiplies every pitch imbalance, so without wings the fly tumbles. A real fly
stabilises with its wings right after take-off. With the stroke trim retuned
(`mid_thc=24`) boost ×2 lands upright in 12/16 phases. **Keep the default
unboosted.**

Not implemented: forward-directed jumps. Using the hind legs (`hind_frac > 0`)
gives more height and ~5 mm forward travel but pitches the fly nose-down (front
flips); a leg-driven stabiliser would be needed. Wing depression during the jump
(real GF escapes) is not modelled because the wings have no aerodynamics.

### Freeze, key Z

Stance legs (controller adhesion on at the trigger) keep their current targets, so
the fly stops mid-stride. Swing legs blend to the standing pose in 80 ms, all
adhesion is on, and the pose is held for `duration`. Release is a 250 ms blend back
into the live gait. Measured (1 s freeze): speed during the hold < 0.5 mm/s,
drift 0.04 mm, max tilt 10°, walking 14.1 mm/s right after.

### Groom, key Y: recorded kinematics

The front legs replay the converted NeuroMechFly grooming recording (§1). The
clip is linearly interpolated at 200 Hz and loops; `speed` scales playback. The mid
legs are extended (coxa-trochanter +20°, femur-tibia −20°) and the hind legs hold
the standing pose, all 4 adhering, and the front tarsi release. The mid-leg
extension matters. The recording is tethered and its rubbing happens ~1 mm below
the thorax. Without the extension the free fly's tarsi hit the floor 34 % of the
time, the thorax sank to 0.87 mm and pitched 8° nose-down. With +20° the thorax is
at 1.28 mm, the front tarsi never touch the floor, and pitch is −3°. Tracking:
mean |q − target| of the front legs 0.8°. Max tilt 13–17°, walking 14 mm/s after a
3 s groom. `Groom(source="synthetic")` is a hand-designed alternative (front legs
protracted and rubbed in antiphase at 6 Hz), kept for comparison. The recorded
clip is the default.

### Back away (spare key E) / turn in place

Timed descending-signal overrides; the CPG does the stepping.
`BackAway(duration=1, speed=1)` = `[-1, -1]`, measured −8.4 to −9.3 mm in 1 s.
`TurnInPlace("left"|"right")` = `[-1, +1]` / `[+1, -1]`, measured ~270°/s after a
~0.1 s ramp. After the action the heading hold (`ControllerConfig.heading_gain`)
turns the fly back toward `target_heading_deg`, like a fly returning to its goal
direction.

### Wing raise (W) and proboscis extension (N): extra joints, behind a flag

These need `Simulation(cfg, fly_factory=make_action_fly_factory(wings=True,
proboscis=True))`. The `fly_factory` constructor argument is a 6-line additive
change in `perpetualfly/simulation.py`, and the default model is unchanged. Without
the joints, `mgr.trigger(WingRaise())` raises `RuntimeError`.
`WingRaise`: wing yaw +1.2 rad, roll +0.35 rad (raised, slightly spread).
`ProboscisExtend`: rostrum −1.0 rad, haustellum +1.0 rad. Both blend in over
0.12 s, hold, and blend out over 0.2 s. The legs keep walking during both. Actuators:
wings kp 0.5 µN·mm/rad (±2), proboscis kp 2 (±5), joint damping 0.01. Wings reach
the target within 0.15 rad and return to ≤ 0.15 rad afterwards
(`test_extra_joints_keep_walking_and_move_wings_proboscis`). The wings and
proboscis have no contacts and no aerodynamics, so they are visual / behavioural
state only.

### Not feasible / not done

* Flight: there are no aerodynamics on NeuroMechFly wings (flybody has a fluid
  model and a flight policy, but it is a different body).
* Head / antenna movements: possible with the same `make_action_fly` mechanism
  (add `c_thorax-c_head` or `c_head-{l,r}_pedicel` DoFs), but not done. A head
  joint un-fuses the head from the thorax, which changes the walking body
  dynamics, and needs its own walking check.
* Abdomen bending (egg laying, etc.): same situation, not needed yet.

## 4. Integration (app keys, full body, brain triggers)

### App (`perpetualfly/app.py`)

* `Session.actions` is an `ActionManager`, created before the brain attaches.
  `Session.available_actions` = `registry.available_actions(sim)`.
  `Session.trigger_action(name, source="key", **params)` returns the terminal
  message; it refuses W / N without the extra joints.
* Keys (`ACTION_KEY_MAP`): **J** jump, **Z** freeze (1.5 s), **Y** groom (3 s),
  **E** back away (1 s), **,** / **.** turn in place left / right (1 s), **W** wing raise
  (1.5 s), **N** proboscis (1.5 s). In `--script-keys`, write `comma` / `period` for
  `,` / `.`. The `?` overlay has an "actions" group; W / N are drawn grey (and marked
  "(unavailable)" in the terminal help) when the body has no extra joints.
* HUD: `ACTION <NAME>  [<phase>]` while an action runs, and `ACTION BACK TO WALKING
  [handback]` during the blend.
* events.csv: `action_start` (with `source`: `key` / `brain` / `api`), then
  `action_end` / `action_cancel` with the action's metrics (jump: `apex_dz_mm`,
  `airtime_s`, `distance_mm`, `landed_upright`, ...). `brain_action` rows record
  brain triggers (action, rate). summary.json: `full_body`, `actions` (start counts),
  `brain.brain_actions`.
* Jump safety: the auto perturber's gate refuses hits while a jump runs
  (`auto_hit_skipped` with reason "jumping"). The fall detector's post-step hook is
  skipped during the jump, and its hold timers and progress window are cleared when
  it resumes. A jump that ends badly is therefore still counted as a fall, once the
  landing phase is over; this was checked with a hard whip crack at touchdown.
  During freeze / groom the detector's "no progress" window is kept empty, so
  standing still is not read as being stuck (without this, a > 3 s freeze could be
  counted as a `no_progress` fall).

### Full body (`--full-body`, `FlyConfig.extra_joints`)

`Simulation` builds `make_action_fly(wings=True, proboscis=True)` when
`cfg.fly.extra_joints` is true. The library default is **off**, so `AppConfig()`
and every reference-trajectory test stays bit-identical. The CLI turns it **on**
unless `--no-full-body` is given or a `--config` file is used. Checks: 3 s headless
run, forward displacement within 2 % and thorax height within 0.1 mm of the
legs-only model (`test_full_body_walking_stats_match_default`). The whip, brain,
brain-actions and all keys were run with it on (real brain, normal terrain: whip
cracks hit and are sent to the brain, O → jump).

### Brain triggers (`--brain-actions`, implies `--brain-steer`)

`perpetualfly/actions/brain_triggers.py` (`BrainActionTriggers`) is created by
`BrainLink.attach` when `BrainLinkConfig.actions` is set. It checks each new
BrainState in the physics thread:

| readout | threshold (BrainLinkConfig) | action |
|---|---|---|
| `escape` = DNp01 giant fiber (1 per side, mean rate) | `jump_escape_hz` 60 Hz | `Jump`, replaces any running action; `jump_refractory_s` 1.5 s |
| MN9 (`probes["MN9"]`) | `proboscis_mn9_hz` 30 Hz | `ProboscisExtend(proboscis_hold_s=0.5)`, prolonged while MN9 stays high; only with the full body; never interrupts another action |
| `groom` = DNg12_a–e (21 per side, both sides, mean rate) | `groom_hz` 20 Hz over consecutive states covering `groom_sustain_s` 0.1 s | `Groom(groom_duration_s=2)`, never interrupts; `groom_refractory_s` 1 s after it ends |
| freeze | not mapped (no unambiguous freezing DN) | – |

The refractory timers use sim time and are cleared on reset. The `groom` group is a
new entry at the end of `schema.DESCENDING_GROUPS`, with `mapping.DESCENDING_TYPES`
"groom" = DNg12_a–e. It adds `dn_groom` to metrics.csv, a GROOM trace (green,
"DNg12") to the brain window's descending panel, and `DNg12 ... Hz` to the HUD.

**Measured with the real brain** (FlyWire v783, headless, flat, `--brain-actions`,
full body):

| input | readout | result |
|---|---|---|
| O (LC4 looming, 200 Hz, 1 s) at t = 2.01 s / 1.50 s | GF 120 / 115 Hz in the first 0.1 s state, MDN 15-45 Hz | `brain_action` jump at t = 2.13 / 1.62 s (**0.12 s after the key**); apex +3.19 / +2.78 mm, airtime 57 / 48 ms, landed upright; one jump per loom (refractory); walking again afterwards |
| T (sugar GRNs, 1 s) at t = 1.51 s | MN9 55 Hz | proboscis extension from t = 1.62 s to 3.13 s (sugar 1 s + 0.5 s hold) |

**Does head / antennal stimulation reach DNg12?** Probe script, real brain, 1 s
stimuli, DNg12 group mean rate:

| stimulus | DNg12 | also |
|---|---|---|
| JO grooming neurons (`jo_grooming`, JO-C/E), 200 Hz, one side or both | **0 Hz** | – |
| JO wind / gravity, 200 Hz | 0 Hz | – |
| head bristles (`head_bristle`), 200 Hz, both sides | **10-13 Hz** (max 12.9) | MN9 120 Hz |
| head bristles + JO grooming | 12-17 Hz (max 17.1) | MN9 125 Hz |
| whip hit on the head (`whip_hit`, `body=Head`, intensity 1) | max 5 Hz | MN9 120 Hz |
| sugar | 0 Hz | MN9 95 Hz |
| positive control: DNg12_b driven directly at 100 Hz | 38-46 Hz | – |

So in this brain-only model, head mechanosensation reaches DNg12 weakly (below the
20 Hz threshold), and the JO grooming afferents do not reach it at all. Brain-driven
grooming therefore does not fire from natural input: honest negative. Lowering
`groom_hz` to ~10 Hz would make strong head-bristle input groom. Side finding: head
bristle and head whip input drive **MN9 to ~120 Hz**, so with `--brain-actions` and
the full body, a hard whip hit on the head would extend the proboscis. That is
the model's wiring, not a bug in the trigger.

## 5. Tests

`tests/test_actions.py` (18 tests, ~70 s). Library: the idle manager is
bit-transparent; the jump goes through all phases, raises the thorax ≥ 1.5 mm,
airtime 20–200 ms, lands upright, restores the actuator params and walks on at
> 8 mm/s; the boost is temporary and a mid-stroke reset cancels and restores
everything; freeze holds (< 0.5 mm/s) and resumes; groom tracks the recording
(< 5°), keeps the tarsi off the floor and resumes; back away goes ≥ 4 mm backward;
turns go > 45° each way; cancel / replace work; the jump is deterministic
(bitwise-equal `qpos`); the registry works; the extra joints keep walking within
3 % and move the wings / proboscis and back.
Integration: brain triggers on synthetic BrainStates (jump + refractory; proboscis
prolonged; groom only when idle and sustained; GF interrupts a groom; nothing without
proboscis joints); CLI flags (`--full-body` default on in the CLI and off in the
library, `--brain-actions` implies steering); greyed-out overlay rows; a headless app
run with J N W Z `comma` (events.csv sources, jump metrics, summary counts, no
falls); the session gate (no auto hits and no falls during a jump, and a 3.5 s groom
is not a fall); full-body walking stats vs the default; and a synthetic-brain session
that installs the triggers.
`tests/test_quick_wins.py` now checks that every key, including the action keys, is
bound exactly once.

## 6. Limitations

* Jump direction is mostly vertical (≈1 mm backward drift). Hind-leg or forward
  jumps pitch the fly over. Uneven terrain and slopes were not tested.
* Boosted jumps tumble without aerodynamic stabilisation (see above).
* Groom replays one 3 s bout from one tethered fly with the front legs only. There
  is no head or antenna motion (no joints), and no adaptation to where anything
  touches the head.
* After an action the CPG resumes wherever its phase is. The blend hides the jump
  in targets, but the first step can be a little irregular (speed dips to ~10 mm/s
  in the first 0.5 s after a jump).
* Turn-in-place is undone by the heading hold afterwards (by design of the app's
  heading hold).
