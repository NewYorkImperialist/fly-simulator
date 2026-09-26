# Flight prototype: flapping-wing flight with MuJoCo aerodynamics

NeuroMechFly can now fly with flapping wings. The lift comes from MuJoCo's built-in
ellipsoid fluid model acting on the moving wings. No external "flight force" is
applied. The code is in `perpetualfly/flight/`, the demos are in
`scripts/demo_flight.py` and the tests in `tests/test_flight.py`. Flight is opt-in:
the default walking `Simulation` (model, timestep 1e-4 s, Euler integrator, no air)
does not change. `test_default_walking_model_is_unaffected` checks this.

Status: every step works, from 1 to 6: tethered lift, free hover (5 s tested),
forward flight up to about 190 mm/s, take-off from the escape jump, landing, and
rendered frames plus a clip. The limits are listed at the end.

## 1. Model (`perpetualfly/flight/body.py`)

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
  actuators are added straight to the MjSpec, as in `perpetualfly/actions/body.py`.

`FlightSimulation(cfg)` (`perpetualfly/flight/sim.py`) is a `Simulation` subclass.
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

## 2. Wingbeat generator (`perpetualfly/flight/wingbeat.py`)

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

## 4. Free hover and forward flight (`perpetualfly/flight/control.py`)

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
* **Controller**: an engineered PID and not a neural model. It is not connected to
  the connectome brain or to the app. Flight is a separate mode:
  `FlightSimulation` is not used by `perpetualfly/app.py` and has no key binding.
* **Timing**: dt 5e-5 halves the walking speed of the sim. The gait controller and
  the jump run at that dt in the take-off demo. It works, but it is not the
  canonical 1e-4 walking configuration.
* **Landing**: a scripted sequence (descend, stick, pitch down, stop). It is not a
  visually triggered landing response with leg extension.

Recommended next step: integrate flight into the app as a mode (key F), with the
escape jump starting the wings at take-off, and add a brain trigger (for example
giant fiber → take-off and flight). The alternative is to replace the prescribed
wingbeat with a small learned policy on this model, flybody-style.
