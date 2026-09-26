# Flight prototype: flapping-wing flight with MuJoCo aerodynamics

NeuroMechFly can now fly with flapping wings. The lift comes from MuJoCo's built-in
ellipsoid fluid model acting on the moving wings. No external "flight force" is
applied. The code is in `fly_simulator/flight/`, the demos are in
`scripts/demo_flight.py` and the tests in `tests/test_flight.py`. Flight is opt-in:
the default walking `Simulation` (model, timestep 1e-4 s, Euler integrator, no air)
does not change. `test_default_walking_model_is_unaffected` checks this.

Status: every step works, from 1 to 6: tethered lift, free hover (5 s tested),
forward flight up to about 190 mm/s, take-off from the escape jump, landing, and
rendered frames plus a clip. Section 7 wires it into the app (`--flight`, key L,
brain escape flight). The limits are listed at the end.

## 1. Model (`fly_simulator/flight/body.py`)

`make_flight_fly()` builds FlyGym's locomotion fly (legs-only joints, 42 leg
position actuators, adhesion) and adds the following.

* **Stroke-plane wing frames.** The wing body frame comes from the mesh vertices.
  The span runs along +y (left wing, 0 to 2.3 mm from the hinge) or −y (right
  wing). The chord runs along x, with the leading edge at +x. +z is the dorsal
  side. The rigging quaternion folds the wing over the abdomen. It is replaced by a
  rotation about the thorax y axis by `stroke_plane_deg` (β = 47.5°, as in flybody).
  At zero joint angles the wings are spread, flat, leading edge forward, in a stroke
  plane tilted nose-down by β relative to the body. When the body pitches nose-up by
  β (the hover posture), the stroke plane is horizontal. In that posture the wing
  hinge sits almost exactly above the centre of mass: the hinge is (0.15, 0, 0.19) mm
  from the COM in the body frame, which gives about (−0.04, 0, 0.24) mm in the world
  frame at 47.5°.
* **3 hinges per wing**, applied in this order (each axis is carried along by the
  previous joints):
  * `c_thorax-{l,r}_wing-stroke`: φ, + = forward;
  * `-deviation`: θ, + = tip up;
  * `-rotation`: ψ, + = leading edge up.

  The signs are mirrored so that + means the same motion on both wings. The kinematics
  were checked numerically: the wing tip sweeps a horizontal plane in the hover
  posture, from +1.8 mm to −1.8 mm along x.
* **Wing servos**: a position actuator `<joint>-flight` per hinge.
  * Gains: kp 60 µN·mm/rad (stroke and deviation) and 15 (rotation).
  * Servo damping: kv 0.012 and 0.002, implemented as *joint damping*, which
    MuJoCo's Euler integrator treats implicitly.
  * Velocity feed-forward: `ctrl = q_ref + kv/kp · dq_ref`.
  * Tracking: 1–2 steps (50–100 µs) of lag, stroke amplitude within about 1°. Under
    air load the deviation, commanded 0, wanders by 2–9°.
  * Net servo torque ("muscle", `FlightSimulation.wing_servo_torque`): at most
    **≈20 µN·mm at 218 Hz** and 30 µN·mm at 250 Hz / 170°. The earlier estimate
    was 11 for inertia plus 5–8 for air.
  * Why the damping is joint damping: actuator kv is explicit and blew up at
    dt 5e-5. The `implicitfast` integrator was stable, but about 6× slower
    (412 vs 43 µs/step), because it takes the derivatives of the fluid forces over
    every body.
* **Fluid**: an invisible ellipsoid geom per wing (group 3, mass 0, no contacts).
  * Semi-axes: 0.005 × 0.55 × 1.15 mm (thickness × half chord × half span).
  * Centre at mid span.
  * `fluidshape=ellipsoid` with flybody's `fluidcoef = [1.0, 0.5, 1.5, 1.7, 1.0]`.
  * The meshes stay the visuals and keep the 2.5 µg wing mass.

  `apply_air(model)` sets density 1.28e-6 g/mm³ and viscosity 1.85e-5 g/(mm·s).
  These are flybody's cm-g-s values converted to mm-g-s. The other bodies get
  MuJoCo's inertia-box drag, which is physical body drag.
* **The 42 leg actuators stay the only position actuators FlyGym knows**. The wing
  actuators are added straight to the MjSpec, as in `fly_simulator/actions/body.py`.

`FlightSimulation(cfg)` (`fly_simulator/flight/sim.py`) is a `Simulation` subclass.
It uses a copied config with dt = **5e-5 s**, the flight fly, and air.
* Its `step()` runs the wingbeat generator at every physics step, and the flight
  controller at 5 kHz.
* Leg modes:
  * `"walk"`: the gait controller runs, and actions such as the jump work on top of
    it;
  * `"flight"`: neutral pose, adhesion off;
  * `"stance"`: neutral pose, adhesion on.
* Tether: `tethered=True` pins the free joint after every step.
* Forces are read from `data.qfrc_fluid`, because **MuJoCo force sensors do not
  include fluid forces**:
  * its free-joint translational part is the total fluid force (world frame) on the
    fly;
  * the rotational part is the torque about the thorax origin, which
    `fluid_torque_com()` moves to the COM.

## 2. Wingbeat generator (`fly_simulator/flight/wingbeat.py`)

The phase p advances at 2π·f and is integrated, so frequency changes stay
continuous. Per wing:

* stroke `φ = φ0 + A/2 · cos p`. The upstroke (backward) is p ∈ (0, π); the
  downstroke (forward) is p ∈ (π, 2π);
* rotation: leading edge first, at geometric angle of attack `aoa_down` on the
  downstroke (ψ = aoa_down) and `aoa_up` on the upstroke (ψ = π − aoa_up). The flip is
  a smoothed square wave `tanh(k sin(p + rot_phase)) / tanh(k)`, with k = 3.
  `rot_phase > 0` means advanced rotation;
* deviation `θ = −atan(tan(tilt) · sin(φ − φ0))`. This rotates the effective stroke
  plane by `tilt` about the lateral axis. + means the front of the stroke is lower,
  which gives forward force.

Left/right parameters are independent. `start(ramp_s)` / `stop()` fade the pattern
in and out from the spread rest pose with a smoothstep envelope. Starting at full
amplitude from rest asks about 1e8 rad/s² of the rotation hinge. The explicit fluid
forces then blew up, so the fade is required.

Base wingbeat (`control.base_wingbeat()`): **218 Hz, 160° stroke, AoA 45°, rotation
phase −0.5 rad**. The lift it gives is in section 3.

## 3. Tethered measurements (`scripts/demo_flight.py tether`, `effect`)

The thorax is pinned in the hover posture (stroke plane horizontal). Values are
averaged over whole wingbeats. Weight = 10.05 µN.

**Lift / weight vs frequency and stroke amplitude** (AoA 45°, rotation phase −0.5):

| stroke (deg) | 180 Hz | 200 Hz | 218 Hz | 235 Hz | 250 Hz |
|---|---|---|---|---|---|
| 120 | 0.43 | 0.52 | 0.61 | 0.69 | 0.76 |
| 140 | 0.57 | 0.69 | 0.80 | 0.90 | 0.99 |
| 151 | 0.65 | 0.78 | 0.91 | 1.02 | 1.11 |
| 160 | 0.72 | 0.87 | **1.00** | 1.12 | 1.22 |
| 170 | 0.80 | 0.96 | 1.11 | 1.24 | 1.35 |

Lift scales roughly as f² and A². The flight-feasibility rebuild with flybody's
kinematics reached 1.04 at 218 Hz / 151°. With these kinematics 151° gives 0.91,
and a 160° stroke is needed for 1.0.

**Stroke-plane tilt** (both wings, through the deviation), 218 Hz / 160°:

| tilt (deg) | Fx / W (forward) | Fz / W | pitch torque (µN·mm, + nose-down) |
|---|---|---|---|
| −20 | −0.345 | 0.844 | −1.073 |
| −10 | −0.190 | 0.956 | −0.274 |
| 0 | −0.009 | 0.999 | +0.527 |
| +10 | +0.173 | 0.969 | +1.229 |
| +20 | +0.329 | 0.867 | +1.733 |

**Angle of attack and rotation timing** (lift / weight, 218 Hz / 160°):

| rotation phase (rad) | AoA 35° | AoA 45° | AoA 55° |
|---|---|---|---|
| +0.3 (advanced) | 0.65 | 0.72 | 0.70 |
| 0 | 0.82 | 0.87 | 0.82 |
| −0.5 (delayed) | 1.02 | 1.00 | 0.91 |
| −0.9 | 0.94 | 0.89 | 0.79 |

In MuJoCo's quasi-steady ellipsoid model, *delayed* rotation gives more lift than
advanced rotation. Real flies gain lift from advanced rotation through unsteady
mechanisms (rotational circulation, wake capture) that this model does not
include.

**Control effectiveness** (per rad, forces in µN, torques in µN·mm about the COM, in
the level stroke frame; used by the controller):

| input | Fz | Fx | roll Tx | pitch Ty | yaw Tz |
|---|---|---|---|---|---|
| amplitude, both | 6.14 | 0.09 | 0 | 0.41 | 0 |
| mean stroke φ0, both | 0 | 2.84 | 0 | −10.32 | 0 |
| amplitude L+ / R− | 0.10 | 0.01 | 5.18 | 0.02 | −0.17 |
| stroke-plane tilt L+ / R− | −2.12 | 0.03 | 0.59 | −0.29 | −12.14 |
| stroke-plane tilt, both | −1.72 | 10.48 | 0 | 4.02 | 0 |

Base wrench: Fz 10.03 µN, pitch torque +0.52 µN·mm (nose-down). The trim is about
+2.9° of mean stroke.

## 4. Free hover and forward flight (`fly_simulator/flight/control.py`)

`HoverController` runs at 5 kHz (every 4 physics steps). All loops work in the
*stroke frame*, which is the thorax frame rotated by β and is level in the hover
posture.

* **Attitude**: PID on the rotation error plus rate damping (ω_n 60 rad/s, ζ 0.8 for
  roll and pitch; 30 rad/s, ζ 0.9 for yaw). Rates are low-passed over half a
  wingbeat. The desired torque is τ = I·α + ω×Iω, using the composite inertia of the
  fly about the COM: diag ≈ (4.1, 5.7, 3.3)e-4 g·mm² in the stroke frame. The
  attitude reference starts at the measured attitude and slews at most 10 rad/s.
* **Altitude**: PID (ω_n 12 rad/s, anti-windup) sets the lift along the
  stroke-plane normal.
* **Horizontal**: PID (ω_n 5 rad/s).
  * Forward goes through the symmetric stroke-plane tilt. The forward force the
    tilted lift vector already gives is subtracted, so attitude errors do not push
    the fly around.
  * Lateral goes through a roll set point.
  * A velocity target moves the position set point, which gives slow forward
    flight.
* **Mixing**: the wrench (Fz, Fx, Tx, Ty, Tz) goes through the inverted
  effectiveness matrix, which gives:
  * lift → stroke amplitude;
  * pitch → mean stroke angle;
  * roll → left/right amplitude difference;
  * yaw → left/right stroke-plane tilt;
  * forward → symmetric tilt.

  The forward force has the lowest priority: it is solved first, clipped, and its
  side effects are fed into the other four. When amplitude alone cannot give enough
  lift, the wingbeat frequency rises (forces scale with f², capped at 260 Hz).

**Halteres analogy.** The fast rate (D) terms use the thorax angular velocity. Real
flies sense it with their halteres, Coriolis gyroscopes that beat in antiphase with
the wings. Haltere feedback reaches the wing steering muscles within about one
wingbeat and acts mostly as rate damping. The slower attitude and position terms
play the role of visual feedback (optomotor, horizon, optic flow). The steering
kinematics also match real flies: amplitude asymmetry for roll, mean stroke angle
for pitch, and stroke-plane and angle-of-attack asymmetries for yaw (Dickinson &
Muijres 2016; Muijres et al. 2014).

Measured with `demo_flight.py hover`, starting at z = 10 mm in the hover posture.
Statistics are taken from 0.3 s on, at 5 kHz. The attitude error includes the
intra-wingbeat body oscillation.

| run | result |
|---|---|
| hover 5 s | xy drift **0.29 mm**, max distance from set point 0.57 mm, z rms 0.05 mm; attitude error rms roll 0.01° / pitch 0.57° / yaw 0.06°, max 0.9° |
| hover 2 s | drift 0.71 mm, z rms 0.08 mm, pitch rms 0.57° |
| pitch gust 3 µN·mm × 5 ms | peak pitch error 11.4°, recovered, z rms 0.12 mm |
| roll gust 3 µN·mm × 5 ms | peak roll 15.4°, recovered, max displacement 2.3 mm |
| yaw gust 3 µN·mm × 5 ms | peak yaw 14.3°, recovered |
| forward 50 mm/s (1 s) | 44.8 mm/s mean over the last 0.5 s (still lagging the ramp), attitude rms 0.6°, altitude held |
| forward 200 mm/s (1.5 s) | 186 mm/s, stroke-plane tilt 11°, z within 0.2 mm, 255 mm flown |

Speed: **≈0.7–0.8× real time** (dt 5e-5, single core, controller included).
Tethered without the controller: about 43 µs/step, 1.15× real time.

## 5. Take-off from the jump and landing (`demo_flight.py takeoff [--at apex] [--land]`)

The sequence is walk 0.3 s → `Jump()` (default escape jump through `ActionManager`
on the flight model) → wings fade in over 10 ms → `HoverController` with a set point
3 mm above the start COM and the current heading. Before take-off the wings are held
spread and flat, the rest pose of the flight model.

* **Wings start at take-off** (the start of the jump's flight phase, vz 141 mm/s,
  body level): the lowest COM afterwards is 3.04 mm and the fly climbs. The body
  pitches up into the hover posture within about 0.15 s. Over the last 1 s: z rms
  0.05 mm, attitude rms 0.6° / yaw 0.75°.
* **Wings start at the apex** (COM 3.9 mm, vz 0): the fly first falls to a COM of
  1.99 mm (the legs hang about 1 mm lower, so this is close to the floor), then
  recovers and hovers. It works, but with little margin. Starting at take-off is
  the default.
* **Landing** (`--land`):
  1. descend at 15 mm/s;
  2. at the first leg contact, adhesion goes on (leg mode "stance"), and the wings
     keep beating for 0.2 s while the pitch reference slews to the walking posture;
  3. the wings fade out and the gait controller takes over.

  Result: the fly lands upright (tilt 2° when the wings stop, 6° while walking
  afterwards). It slides about 1.4 mm forward while pitching down. Without the
  pitch-down the fly touched down on its hind legs at 47° and flipped onto its back.

## 6. Frames and clip

Checked by eye; offscreen renderer PNGs are in `runs/flight/` (gitignored):
`hover.png`, `wings_closeup_a.png`, `wings_top.png` and `takeoff_hover.png`. They
show the body pitched nose-up with spread, beating wings, and a top view with both
wings spread laterally. `takeoff_hover_30x.mp4` (160 kB, 400×300, 30× slow motion,
about 4 frames per wingbeat) shows the jump crouch with spread wings → take-off →
flapping climb and pitch-up → hover.

```
.venv/bin/python scripts/demo_flight.py hover --duration 2 --frames runs/flight
.venv/bin/python scripts/demo_flight.py takeoff --duration 0.3 --slowmo 30 --size 400x300 --record runs/flight/takeoff.mp4
```

## 7. Flight in the app (`--flight`; `fly_simulator/flight/mode.py`)

`--flight` builds the `Session` on a `FlightSimulation`: the flight fly (stroke-plane
wing hinges, fluid ellipsoids, the same 42 leg actuators and adhesion), dt **5e-5 s**,
air on. The fly walks with the normal gait controller (leg mode `"walk"`, wings held
spread and flat) until a take-off. `FlightMode` runs the state machine in a post-step
hook:

    walking -> takeoff -> hovering / forward (manual or escape) -> landing -> touchdown -> walking

* **Take-off**: a `Jump` through the session's `ActionManager`. At the start of the
  jump's flight phase (end of the leg stroke) the jump hands back (`done`, its normal
  end event), the legs go to the flight pose, adhesion off, the wingbeat fades in over
  10 ms and a `HoverController` (section 4) takes over. If the fly is tumbling at that
  moment (tilt > 80°, e.g. knocked over while jumping) the wings stay off and the jump
  carries on as a plain jump.
* **No external force anywhere**: every newton of lift, thrust and steering torque
  comes from the flapping wings in MuJoCo's fluid model. The old `flight_assist`
  emulation stays off and is ignored in flight mode.
* **Landing**: section 5's sequence (descend at 15 mm/s, adhesion at the first leg
  contact, 0.2 s pitch-down with the wings beating, 10 ms fade-out), then
  `controller.reset()` and walking.
* **Crash**: a body (non-leg) contact or tilt > 120° for 50 ms while airborne (after a
  0.15 s grace period for the tumbling short-mode take-off) fades the wings out and
  hands the fly to the fall detector.
* The altitude set point follows the terrain under the fly (clearance above the local
  ground, slewed at most 50 mm/s).
* The fall detector, the auto perturber and the other action keys pause while the fly
  is airborne; `RunMetrics` does not count flights as walking. The swatter counts an
  escape flight like a jump (`jump_probe`).

**Keys**: **L** takes off (long-mode jump → hover at +2 mm; lands by itself after
`hover_s` = 4 s of hovering) or lands. While airborne the **arrows steer**: UP / DOWN
change the forward speed in 50 mm/s steps (negative = backward, up to 250 mm/s),
LEFT / RIGHT turn the heading by 30°. On the ground the arrows keep cracking the whip;
SPACE / U always do. The HUD gets a `FLIGHT` line: state (WALKING / TAKEOFF / HOVERING
/ FORWARD / ESCAPE / LANDING), wingbeat frequency, COM altitude above the ground,
horizontal speed. Flight events go to `events.csv` (`flight_takeoff`, `flight_land`,
`flight_crash`, `flight_abort`) and `summary.json` (`flight`).

**Brain** (`--flight --brain-actions`): `BrainActionTriggers.flight` is set. Giant
fibre > threshold while walking → the usual (short-mode with the swatter) `Jump`, and
`FlightMode.takeoff(jump, escape_dir)`: wings on at take-off, escape flight away from
the threat. The escape direction comes from `threat_fn` (the swatter's paddle centre,
as for the old emulation: the looming detectors are retinotopic); without a threat
position the fly escapes straight ahead. Escape flight (`FlightModeConfig`):

* velocity control only at 200 mm/s along the escape direction (no position loop
  pulling the fly back toward the take-off point, stiffer velocity loop
  `xy_zeta` 3), COM held 2 mm above the ground (a high fly meets the descending paddle
  sooner), 0.8 s of flight, 0.25 s braking, then landing (~1.3 s airtime, 90–155 mm);
* the heading turns toward the escape direction at 6 rad/s; an escape more than 100°
  off the heading is flown **backward** facing the threat (turning 180° while
  accelerating crashed the fly; real flies also take off backward from a frontal
  looming stimulus);
* a GF burst while already airborne re-directs the escape (no second jump).

**Walking at dt 5e-5** (flat ground, 2 s after 0.3 s settling): 14.14 mm/s vs 14.08
mm/s on the canonical model, heading within 0.3°, straight (lateral drift 0.02 mm). The
gait is unchanged; the cost is the timestep: bare sim **0.33× real time vs 0.70×**, the
app ~0.25–0.28× walking and ~0.35–0.40× hovering / flying (brain on: 0.2–0.37×;
`--real-vision`: ~0.12×). `render_every_steps` is doubled (300) to keep the frame and
brain-update rate per simulated second.

**Swatter escapes, real brain, headless** (FlyWire LIF brain, window 0.02 s, sync wait
0.05 s, brain update every 10 ms of fly time; flat ground; L1–L2 from behind and from
the left, 3 start times 30 ms apart = distinct gait phases and brain-window
alignments, 12 swats per row; each swat from a fresh reset after ~1 s of walking). The
baseline is the default short-mode jump without flight on the canonical dt 1e-4
model (`scripts/demo_flight_escape.py [--flight] --phases 3`):

| geometric looming | dodged | grazed | hit | falls | landed upright | notes |
|---|---|---|---|---|---|---|
| short jump only (dt 1e-4) | 5/12 | 4 (0.6–17 µN·s) | 3 (204–859 µN·s) | **9/12** | 3/12 | the short jump tumbles (landing tilt 144–171°) |
| **real flight** (defaults) | **6/12** | 4 (0.2–1.6 µN·s) | 2 (37, 56 µN·s) | **2/12** | **10/12** | L1 rear 3/3 dodged; both hits: GF 44–56 ms before landing |
| flight, first version (position loop, +1 mm climb) | 5/12 | 5 | 2 | 2/12 | 8/12 | 4 crashes after the paddle caught the fly |

Per flight: the wings start ~10 ms after the GF trigger (median 8–12 ms; take-off tilt 16–40°), the
fly lands upright 1.29–1.35 s later, 90–154 mm away (tilt at wing stop 1–2°), and walks
on. Dodged flies are 8–33 mm from the aim point when the plate lands. The remaining
failures are late triggers (GF within ~60 ms of the plate landing: a real fly can't
clear a 7 mm paddle in that time either) and grazes, where the plate edge catches
the fly on its way out, sometimes knocking it over (crash → fall). Trigger times vary
between runs by ±30 ms (brain), so these 12-swat counts are noisy (the three flight
runs gave 5, 4 and 6 dodges, the last one after the tuning above).

With `--real-vision` (8 swats L1–L2, rear + left): 1 dodged, 1 grazed, 6 hits. The
compound eyes see the paddle too late (GF +0.03–0.05 s *after* the plate lands; the
known real-vision latency, docs/VISION.md), so flight can't help; spontaneous GF
escapes while walking produced extra flights, which all landed. RTF ~0.12.

Other checks (headless, real brain unless noted):

* `--flight --brain-actions --stress --swatter --terrain normal`, keys
  `1:l,2:up,2.6:left,7:space,9.5:v`: take-off at 1.17 s (long mode, tilt 3°), hover,
  forward 50 mm/s, turned 30° left, flew 10 s over the terrain (~5.6 mm above it), the
  whip hit the flying fly without upsetting it, the swat while flying fired the GF →
  escape re-directed → **dodged** (16 mm from the aim), landed upright after 10.6 s
  airtime / 545 mm, walked on. 649 brain states, 0 dropped.
* Manual cycle (no brain): L at 0.5 s → hover 4 s →
  auto landing, tilt 1.8°, walking again at 14 mm/s.
* Whip and pain while walking (`--flight --brain-steer --stress`): cracks hit, the
  octopamine level rose to 0.30. Note: at dt 5e-5 L2 cracks topple the fly more often
  than at dt 1e-4 (5 identical cracks: 2 falls vs 0). The normal body at dt 5e-5
  gives the same 2/5, so it is the timestep (contact dynamics), not the wings; the whip
  is calibrated at 1e-4.
* Before the guards: a GF escape fired while the whip was knocking the fly over, and
  L pressed on an upside-down fly, both started the wings and crashed. Now L is refused
  above 45° of tilt and the wings don't start above 80°. A crash used to switch the
  wing servos off at full wing speed, which blew the model up (NaN in the wing rotation
  hinge); the wings now always fade out.

**Refused / not supported**: `--flight` with `--course` or `--job` is a `ConfigError`
(their scoring, respawns and props assume the canonical walking model). `--full-body`
is ignored (the flight body has its own wings and no proboscis joints: W and N are
greyed out, MN9 has no body effect). `--real-vision` works (eyes on the flight fly),
`--whip-vision`, `--stress`, terrain, whip and swatter work as usual.

Frames (renderer PNGs in `runs/flight/`, gitignored, checked by eye): `app_escape_1_takeoff.png`
(fly airborne, wings spread, the paddle coming down behind it),
`app_escape_2_paddle_lands.png` (plate flat on the ground, the fly just in front of
its edge flying away: a graze of 0.4 µN·s), `app_escape_3_touchdown.png`,
`app_escape_4_walking.png` (upright, walking, wings spread), `app_hover_hud.png` (HUD
line `FLIGHT HOVERING 218 Hz alt 4.9 mm v 11 mm/s`), and `app_escape_10x.mp4` (150 kB,
480×320, 10× slow motion, take-off → escape under the slamming paddle).

## Limitations

* **Aerodynamics**: MuJoCo's ellipsoid model is quasi-steady: a thin ellipsoid with
  blunt and slender drag, angular drag, Kutta lift and Magnus terms. It has no
  leading-edge vortex dynamics, wake capture or wing–wing interaction, and the wing
  is rigid, with no camber or twist. The lift numbers are therefore only
  order-of-magnitude real. Delayed rotation giving the most lift is an artifact of
  the model.
* **Kinematics**: prescribed sinusoidal stroke and square-wave rotation, not learned
  or measured wingbeats (flybody uses a learned policy). The steering maps are
  linear around hover, measured tethered, and hard-coded in `control.py`. They must
  be re-measured (`demo_flight.py effect`) if the wingbeat or the fluid parameters
  change.
* **Body**:
  * the legs are held at the neutral walking pose in flight, with no tucking;
  * the head and abdomen are rigid;
  * forward flight uses stroke-plane tilt with the body at hover pitch; real flies
    also pitch nose-down, which is not modelled.
* **Controller**: an engineered PID and not a neural model. The brain only decides
  *when* to escape (giant fibre) and the threat direction comes from the swatter's
  position, not from the retinotopic LC4 / LPLC2 activity. Landing, braking and the
  escape speed are fixed parameters.
* **Timing**: dt 5e-5 halves the RTF of the whole app in `--flight`, walking
  included (the gait itself is unchanged). A possible speed-up is switching the
  timestep to 1e-4 while walking and to 5e-5 only in the air (not done: the CPG
  integrator's control timestep is fixed at construction).
* **Wings while walking**: the flight model's wings rest spread and flat (no folding
  joint), so a walking `--flight` fly shows spread wings.
* **Landing**: a scripted sequence (descend, stick, pitch down, stop). It is not a
  visually triggered landing response with leg extension.

Recommended next steps: take the escape direction from the brain / visual system
(left vs right LC4 / LPLC2) instead of the swatter's position; a visually triggered
landing; wing folding on the ground; or replace the prescribed wingbeat with a small
learned policy on this model, flybody-style.
