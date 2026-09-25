# The physical whip

`perpetualfly/interaction/whip.py` adds a visible whip to the world: a chain of capsules
on a scripted handle. A crack is scripted motion of the handle **only**; the fly is moved
exclusively by MuJoCo contact forces between the whip and fly geoms. The whip is the
app's default hit (`--hit-mode whip`); H switches to the thorax-force shove
(docs/PERTURBATION_CALIBRATION.md) and back.

Units as everywhere: mm, g, s, force uN, impulse uN*s (= g*mm/s). Fly: 1.024 mg, weight
10.05 uN, dt = 1e-4 s.

## Model

| part | what | numbers (`WhipConfig` defaults) |
|---|---|---|
| `whip/handle` | mocap body, the "hand" (no geoms) | moved every step during a crack, every 10th step when idle |
| `whip/grip` | free body welded to the handle (`mjEQ_WELD`, solref 4e-4 s, torquescale 1), wooden handle stock, visual only | 5 mg, capsule 1.6 mm x r 0.11 |
| `whip/seg0..11` | 12 capsules chained along the grip's +x, each with 2 hinges (local z = in the swing plane, local y = out of plane) | 6 mm total, r 0.09 -> 0.04 mm, 2 mg total (per segment ~ r^2), leather brown, red tip |
| hinges | stiffness 6000 -> 120 uN*mm/rad (geometric taper base -> tip), damping 8 -> 0.06 uN*mm*s/rad, out-of-plane (y) hinges x3, armature 8e-6 g*mm^2 | static sag < 0.1 mm; settles in ~20 ms |
| contacts | `contype 0, conaffinity FLY_BIT` (fly only; `collide_terrain=True` adds `TERRAIN_BIT`), `priority 1`, FlyGym `ContactParams` solref/solimp/margin, friction 0.2 | no whip self-collision |

**Why a weld and not a chain on the mocap body?** Mocap bodies have no velocity. A chain
parented to one is carried along kinematically, so it neither lags nor carries momentum.
Driving a free grip through a stiff weld gives the chain real dynamics.

**Why 2 mg?** A 0.4 mg chain was momentum-limited: its impulse saturated at about
0.15 uN*s whatever the swing speed. The gentle to hard levels need 0.15-0.6 uN*s
(docs/PERTURBATION_CALIBRATION.md), which takes an effective striking mass comparable
to the fly's. Stiffness and damping scale with the mass, which keeps the chain's
frequencies the same.

## A crack

`whip.crack(side, level, source)`, where `side` is one of `left, right, front, rear,
overhead, random`, relative to the fly's heading at crack time. The side names where the
whip **comes from**: a crack from the left pushes the fly to its right. Phases (sim time):

| phase | duration | handle motion |
|---|---|---|
| `to_up` | 60 ms | from the current pose to the raised pose: 3.5 mm above the strike pivot, whip pointing up and away from the fly |
| `down` | 50 ms | down to the cocked pose: whip beside the fly, `back_angle_deg` = 160 deg behind the impact line |
| `hold` | 60 ms | lets the chain settle (residual wobble otherwise randomises the strike) |
| `swing` | 2-10 ms | rotation about `k = u x s` (s = push direction, u = whip direction at impact), accelerating to omega over 6 ms, then **braking over the last 6 deg** at theta_stop |
| `follow` | 20 ms | handle held at the stop while the chain lashes |
| `lift` | 40 ms | whip raised again (no resting on the fly, no spring-back into it) |
| `back` | 90 ms | to the idle pose |

Strike geometry (fly heading frame; aim point = thorax frame + (-0.45, 0, +0.12) mm, the
side of the thorax above the femurs):
* left / right: pivot 4.6 mm behind the aim point; the whip lies along the body axis,
  tilted 8 deg down, and sweeps sideways.
* front / rear: pivot beside the fly; the whip lies across the body and sweeps along it.
  The aim is 0.2 / 0.35 mm lower (head front / abdomen tip).
* overhead: whip horizontal along the body axis, aim 0.4 mm higher; it chops down onto
  the thorax.
* random: random horizontal push azimuth, pivot on the rear half, 0-20 deg extra descent.

**The stop rule.** The handle stops at `theta_stop = -asin(stop / contact_radius)`, where
`stop` = the fly's extent from the aim point in the direction the whip comes from (front
0.95, rear 1.7, side 0.45 mm, elliptically blended) + 0.15 mm. The whip's rest line
therefore lies just outside the fly, and only the chain's dynamic overshoot, the lash,
reaches it. A stiff whip that keeps sweeping bulldozes the fly: 70-110 ms contacts that
tipped it even at the gentlest speed.

**Levels** (`WhipConfig.levels`, strength keys 1-4 shared with the shove): peak handle
angular speed omega = 60 / 120 / 200 / 1200 rad/s. Level 4 also has a 6 mm `lunge`:
the handle translates toward the fly during the swing, since rotation alone saturates
around 0.8 uN*s. Front cracks use 0.9 x omega (`side_omega_scale`): a push backward
tips the fly far more easily (the shove calibration found the same), and at 1.0 x the
front medium crack tipped 6/8.

**Tracking.** The handle frame follows an anchor that tracks the aim point with a
rate-limited follower: velocity feed-forward (no lag while walking), |v| <= 300 mm/s,
|a| <= 2e4 mm/s^2. During swing/follow/lift it extrapolates at constant velocity. The
first version snapped the anchor after a big knock. A few-mm jump of the mocap target
in one step makes the weld yank the chain to NaN, so the target must never jump.

**Idle.** The handle is parked at (-3.0, -3.0, +2.3) mm from the aim point (behind,
right, above), with the whip pointing forward-right-down. It follows the fly smoothly
and stays in the follow camera's view. Minimum whip-fly distance while walking is
> 1 mm (tests: 3 s flat, 2 s on normal terrain; calibration: 0 stray contacts for any
upright fly).

**Queueing.** A crack requested while another is between `to_up` and the end of `lift`
is queued (one slot; a newer request replaces it). Otherwise it starts immediately from
wherever the handle is.

## Measurement

A post-step hook scans `data.contact` for whip<->fly geom pairs. It rotates
`mj_contactForce` into the world frame (force on geom2 = `frame.T @ f[:3]`, sign flipped
if the fly is geom1) and sums the force x dt into the strike's impulse vector. It also
tracks the peak total force, first and last contact time and per-body impulse. Contacts
during `swing`, `follow` and `lift` belong to the strike. At the end of `lift` a
`WhipHitEvent` goes to `whip.listeners`:

`sim_time` (first contact), `step`, `source`, `body` (the fly body that took most of the
impulse), `direction_name` (`from_<side>`), `direction` (unit vector of the measured
impulse, world), `magnitude_uN` (peak force), `magnitude_bw`, `duration_s` (first to
last contact), `impulse_uNs`, `level`, `hit`, `side`, `impulse_vec`, `mean_force_uN`,
`commanded_direction` (s), `omega`, `crack_time`, `n_contact_steps`, `bodies`,
`kind="whip"`.

Contacts outside a strike window are counted in `whip.stray_contact_steps` /
`stray_impulse`. They only happened in L3/L4 calibration trials after the fly had been
knocked over or launched and tumbled into the returning whip.

## App integration

* `AppConfig.whip` (saved in `config.json["app"]["whip"]`), `AppConfig.session.hit_mode`
  (`"whip"` default). CLI: `--hit-mode whip|shove`. `whip.enabled=false` doesn't build
  the whip.
* Keys (whip mode): SPACE random, LEFT/RIGHT from the fly's left/right, UP from the
  front, DOWN from the rear, U overhead; 1-4 strength (shared); H toggles whip/shove;
  A toggles automatic hits, which use the current mode. The auto perturber's push
  directions map to the whip side that pushes that way (push left = crack from the right).
* Hits -> `metrics.record_hit` (magnitude = mean contact force, so the summary's impulse
  = magnitude x duration is the measured impulse). events.csv gets `whip` rows (details:
  measured impulse vector, `impulse_uNs`, peak `magnitude_uN`, bodies...). Misses are
  `whip_miss` rows and aren't counted as hits. Mode changes are `hit_mode` rows.
  summary.json has `hit_mode`, `n_whip_cracks`, `n_whip_hits`, `n_whip_misses`,
  `whip_stray_contact_steps`.
* `metrics/contacts.py` ignores contacts with `whip/...` geoms, so a strike on the
  thorax is not "body touching the ground" for the fall detector or metrics.csv.
* HUD line: `WHIP L2 medium  cracks 5 hit 5 miss 0  [swing]  auto on` (or `SHOVE ...`).
* Threading: `crack()` is called from the key handler inside `runner.locked()`. Hooks
  and listeners run in the physics thread.

## Calibration

`scripts/demo_whip.py --calibrate --phases 8 --post 2.0 --sides left right front rear
overhead random` (headless, flat ground, default hybrid controller). For each trial: a
fresh sim, walk 1.0 s + k x 10.5 ms (8 gait phases), crack, observe 2 s, compare with
the unperturbed run of the same phase. Columns as in docs/PERTURBATION_CALIBRATION.md.
Also: `cos(impulse, push)` = cosine between the measured impulse and the commanded push
direction; `push along dir.` = displacement along the push direction 0.6 s after the
crack, relative to the unperturbed run (lateral/longitudinal sides only). The omega
column is the level's nominal speed (front uses 0.9 x).

### Summary (flat, 48 cracks per level = 8 phases x 6 sides)

| level | omega | hit rate | mean impulse (uN*s) | tipped >60 deg | shove equivalent (impulse, tipped) |
|---|---|---|---|---|---|
| 1 gentle | 60 rad/s | 48/48 | 0.17 | 0/48 | 0.15 uN*s, 0/48 |
| 2 medium | 120 rad/s | 48/48 | 0.27 | 3/48 (overhead / random only) | 0.30 uN*s, 8/48 |
| 3 hard | 200 rad/s | 48/48 | 0.65 | 36/48 | 0.60 uN*s, 31/48 |
| 4 absurd | 1200 rad/s + 6 mm lunge | 48/48 | 1.04 | 42/48 | 2.5 uN*s, 48/48 |

On normal procedural terrain (`--terrain normal --pre 1.5`, 4 phases x 6 sides = 24 per
level): hit rate 24/24 at every level; tipped 0 / 2 / 20 / 20 of 24; mean impulse
0.16 / 0.27 / 0.64 / 1.00 uN*s. No `SimulationInstabilityError` or NaN in any of
the 288 cracks, nor in an earlier 192-crack run at other speeds.

What it looks like: L1 is a visible flinch (13-18 deg wobble, 0.2-0.6 mm off course,
keeps walking). L2 is a clear stumble (20-40 deg, 1-1.5 mm; overhead and random
sometimes flip it). L3 usually rolls the fly onto its side or back and knocks it
1-3 cm. L4 flings it 4-9 cm (front: up to 10 cm) with 1-6 mm of airtime, spinning.
The absurd whip is weaker than the absurd shove (2.5 uN*s, 30-60 cm), see Limitations.

### Full table (flat)

| level | omega (rad/s) | side | n | hit rate | impulse uN*s mean (max) | peak force uN mean (max) | cos(impulse, push) | push along dir. (mm) | deviation (mm) | extra tilt (deg) | tipped >60 deg | still down at end | walking again within 2 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 gentle | 60 | front | 8 | 8/8 | 0.158 (0.196) | 493 (743) | 0.98 | 0.06 (0.19) | 0.40 (0.55) | 13 (22) | 0/8 | 0/8 | 8/8 |
| 1 gentle | 60 | left | 8 | 8/8 | 0.151 (0.180) | 612 (834) | 0.85 | -0.04 (0.41) | 0.57 (0.91) | 18 (25) | 0/8 | 0/8 | 8/8 |
| 1 gentle | 60 | overhead | 8 | 8/8 | 0.249 (0.258) | 615 (806) | 0.90 | - | 0.17 (0.26) | 7 (10) | 0/8 | 0/8 | 8/8 |
| 1 gentle | 60 | random | 8 | 8/8 | 0.170 (0.271) | 480 (764) | 0.70 | - | 0.34 (0.87) | 8 (13) | 0/8 | 0/8 | 8/8 |
| 1 gentle | 60 | rear | 8 | 8/8 | 0.128 (0.163) | 539 (655) | 0.94 | 0.05 (0.28) | 0.23 (0.54) | 9 (13) | 0/8 | 0/8 | 8/8 |
| 1 gentle | 60 | right | 8 | 8/8 | 0.150 (0.187) | 600 (872) | 0.84 | -0.04 (0.37) | 0.57 (0.91) | 18 (25) | 0/8 | 0/8 | 8/8 |
| 2 medium | 120 | front | 8 | 8/8 | 0.212 (0.252) | 863 (1059) | 0.99 | 0.28 (0.75) | 0.86 (1.24) | 22 (36) | 0/8 | 0/8 | 8/8 |
| 2 medium | 120 | left | 8 | 8/8 | 0.223 (0.255) | 1164 (1416) | 0.85 | 0.68 (2.44) | 1.43 (2.47) | 26 (35) | 0/8 | 0/8 | 8/8 |
| 2 medium | 120 | overhead | 8 | 8/8 | 0.437 (0.590) | 926 (1019) | 0.89 | - | 4.01 (30.38) | 32 (172) | 1/8 | 1/8 | 7/8 |
| 2 medium | 120 | random | 8 | 8/8 | 0.252 (0.297) | 1038 (1417) | 0.52 | - | 5.20 (25.24) | 43 (167) | 2/8 | 1/8 | 7/8 |
| 2 medium | 120 | rear | 8 | 8/8 | 0.276 (0.314) | 1299 (1374) | 0.90 | 0.29 (0.73) | 0.68 (1.41) | 19 (29) | 0/8 | 0/8 | 8/8 |
| 2 medium | 120 | right | 8 | 8/8 | 0.219 (0.245) | 1188 (1434) | 0.84 | 0.65 (2.25) | 1.41 (2.27) | 27 (40) | 0/8 | 0/8 | 8/8 |
| 3 hard | 200 | front | 8 | 8/8 | 0.597 (0.634) | 2360 (3032) | 0.99 | 21.64 (28.09) | 32.38 (47.48) | 161 (173) | 8/8 | 4/8 | 4/8 |
| 3 hard | 200 | left | 8 | 8/8 | 0.509 (0.574) | 2608 (2979) | 0.91 | 10.18 (16.51) | 22.01 (31.09) | 167 (172) | 8/8 | 6/8 | 2/8 |
| 3 hard | 200 | overhead | 8 | 8/8 | 1.160 (1.208) | 2987 (3688) | 0.97 | - | 4.22 (28.23) | 37 (170) | 1/8 | 1/8 | 7/8 |
| 3 hard | 200 | random | 8 | 8/8 | 0.557 (0.858) | 2372 (2885) | 0.49 | - | 25.13 (38.93) | 142 (175) | 6/8 | 5/8 | 3/8 |
| 3 hard | 200 | rear | 8 | 8/8 | 0.542 (0.593) | 2600 (3225) | 0.88 | 1.21 (17.34) | 13.90 (25.18) | 98 (175) | 5/8 | 3/8 | 5/8 |
| 3 hard | 200 | right | 8 | 8/8 | 0.516 (0.577) | 2719 (3040) | 0.86 | 10.45 (15.22) | 25.16 (32.26) | 170 (175) | 8/8 | 8/8 | 0/8 |
| 4 absurd | 1200 | front | 8 | 8/8 | 0.993 (1.038) | 3389 (4313) | 0.99 | 73.82 (93.31) | 91.30 (104.17) | 169 (175) | 8/8 | 6/8 | 2/8 |
| 4 absurd | 1200 | left | 8 | 8/8 | 0.914 (1.012) | 3138 (3730) | 0.91 | 36.20 (55.10) | 40.39 (56.61) | 173 (176) | 8/8 | 3/8 | 5/8 |
| 4 absurd | 1200 | overhead | 8 | 8/8 | 1.610 (1.794) | 5337 (6058) | 0.95 | - | 6.62 (32.75) | 48 (175) | 2/8 | 0/8 | 8/8 |
| 4 absurd | 1200 | random | 8 | 8/8 | 0.872 (1.040) | 4406 (7393) | 0.71 | - | 52.00 (93.68) | 150 (175) | 8/8 | 4/8 | 4/8 |
| 4 absurd | 1200 | rear | 8 | 8/8 | 0.926 (1.026) | 5387 (6083) | 0.93 | 46.65 (56.86) | 59.16 (72.82) | 166 (175) | 8/8 | 5/8 | 3/8 |
| 4 absurd | 1200 | right | 8 | 8/8 | 0.937 (1.025) | 3133 (3412) | 0.93 | 36.57 (53.55) | 41.07 (57.93) | 173 (177) | 8/8 | 6/8 | 2/8 |

### Full table (normal terrain, 4 phases)

| level | omega (rad/s) | side | n | hit rate | impulse uN*s mean (max) | peak force uN mean (max) | cos(impulse, push) | push along dir. (mm) | deviation (mm) | extra tilt (deg) | tipped >60 deg | still down at end | walking again within 2 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 gentle | 60 | front | 4 | 4/4 | 0.162 (0.184) | 595 (710) | 0.98 | 0.05 (0.16) | 0.88 (2.26) | 15 (29) | 0/4 | 0/4 | 3/4 |
| 1 gentle | 60 | left | 4 | 4/4 | 0.138 (0.158) | 640 (834) | 0.80 | 0.18 (0.39) | 2.12 (4.64) | 22 (32) | 0/4 | 0/4 | 2/4 |
| 1 gentle | 60 | overhead | 4 | 4/4 | 0.252 (0.255) | 642 (873) | 0.92 | - | 0.32 (0.45) | 9 (13) | 0/4 | 0/4 | 4/4 |
| 1 gentle | 60 | random | 4 | 4/4 | 0.140 (0.187) | 501 (730) | 0.71 | - | 0.48 (0.89) | 9 (13) | 0/4 | 0/4 | 4/4 |
| 1 gentle | 60 | rear | 4 | 4/4 | 0.116 (0.167) | 574 (643) | 0.94 | 0.06 (0.33) | 0.44 (0.75) | 12 (15) | 0/4 | 0/4 | 4/4 |
| 1 gentle | 60 | right | 4 | 4/4 | 0.155 (0.171) | 535 (623) | 0.89 | -0.21 (-0.02) | 0.55 (0.65) | 21 (23) | 0/4 | 0/4 | 4/4 |
| 2 medium | 120 | front | 4 | 4/4 | 0.207 (0.237) | 769 (969) | 0.98 | 0.23 (0.51) | 1.89 (5.29) | 25 (36) | 0/4 | 0/4 | 3/4 |
| 2 medium | 120 | left | 4 | 4/4 | 0.217 (0.229) | 1147 (1380) | 0.82 | -0.04 (0.48) | 1.26 (1.91) | 27 (36) | 0/4 | 0/4 | 4/4 |
| 2 medium | 120 | overhead | 4 | 4/4 | 0.411 (0.427) | 919 (988) | 0.96 | - | 0.39 (0.50) | 12 (15) | 0/4 | 0/4 | 4/4 |
| 2 medium | 120 | random | 4 | 4/4 | 0.241 (0.294) | 1229 (1252) | 0.55 | - | 12.04 (23.73) | 94 (177) | 2/4 | 2/4 | 2/4 |
| 2 medium | 120 | rear | 4 | 4/4 | 0.280 (0.326) | 1318 (1415) | 0.94 | 0.34 (0.67) | 3.45 (4.91) | 32 (37) | 0/4 | 0/4 | 1/4 |
| 2 medium | 120 | right | 4 | 4/4 | 0.235 (0.250) | 1278 (1695) | 0.92 | 1.79 (3.66) | 2.97 (5.97) | 28 (30) | 0/4 | 0/4 | 3/4 |
| 3 hard | 200 | front | 4 | 4/4 | 0.604 (0.623) | 2620 (3097) | 0.98 | 19.37 (24.16) | 34.30 (44.03) | 168 (175) | 4/4 | 3/4 | 1/4 |
| 3 hard | 200 | left | 4 | 4/4 | 0.525 (0.558) | 3774 (4508) | 0.97 | 21.15 (26.23) | 28.61 (31.51) | 156 (175) | 4/4 | 2/4 | 2/4 |
| 3 hard | 200 | overhead | 4 | 4/4 | 1.099 (1.168) | 2897 (3687) | 0.99 | - | 1.18 (3.50) | 15 (29) | 0/4 | 0/4 | 3/4 |
| 3 hard | 200 | random | 4 | 4/4 | 0.549 (0.636) | 2535 (2861) | 0.54 | - | 27.48 (43.30) | 165 (177) | 4/4 | 3/4 | 1/4 |
| 3 hard | 200 | rear | 4 | 4/4 | 0.522 (0.582) | 2309 (2728) | 0.87 | -2.29 (8.56) | 22.70 (27.51) | 173 (177) | 4/4 | 4/4 | 0/4 |
| 3 hard | 200 | right | 4 | 4/4 | 0.527 (0.570) | 2940 (3323) | 0.81 | 6.20 (8.75) | 16.98 (21.49) | 163 (177) | 4/4 | 3/4 | 1/4 |
| 4 absurd | 1200 | front | 4 | 4/4 | 0.912 (0.982) | 3612 (4306) | 0.98 | 71.43 (93.21) | 88.75 (115.64) | 168 (175) | 4/4 | 3/4 | 1/4 |
| 4 absurd | 1200 | left | 4 | 4/4 | 0.955 (1.009) | 3020 (3085) | 0.94 | 37.47 (45.46) | 44.87 (50.41) | 171 (175) | 4/4 | 3/4 | 1/4 |
| 4 absurd | 1200 | overhead | 4 | 4/4 | 1.556 (1.802) | 5589 (6350) | 0.96 | - | 11.10 (31.72) | 90 (176) | 2/4 | 1/4 | 2/4 |
| 4 absurd | 1200 | random | 4 | 4/4 | 0.746 (1.028) | 3746 (5108) | 0.43 | - | 50.00 (92.22) | 109 (170) | 2/4 | 1/4 | 3/4 |
| 4 absurd | 1200 | rear | 4 | 4/4 | 0.896 (1.020) | 4615 (5663) | 0.93 | 45.63 (59.72) | 54.93 (63.41) | 169 (170) | 4/4 | 3/4 | 1/4 |
| 4 absurd | 1200 | right | 4 | 4/4 | 0.930 (0.966) | 3232 (3350) | 0.89 | 34.33 (36.26) | 39.63 (49.21) | 146 (171) | 4/4 | 1/4 | 3/4 |

## Tuning history (what didn't work)

1. Chain parented to the mocap body: no dynamics (see above). Switched to a weld.
2. Soft chain (400 -> 4 uN*mm/rad): a 1.5 mm gravity sag and an undamped 16 Hz wobble;
   the tip flailed by several mm even while idle. Stiffened x7.5 and damped.
3. Light chain (0.4 mg): impulse capped at ~0.15 uN*s. Increased to 2 mg.
4. Continuous sweep through the fly: a heavy, stiff whip bulldozes, and slow levels
   tipped the fly with 100 ms pushes. Added the stop rule.
5. Aim on the thorax top (z +0.22 mm): the whip rode over the rounded top, lay across
   the back and sprang back, so the impulse often pointed the wrong way. Fixes: aim
   lower (+0.12, still above the femurs, which reach ~1.2 mm), out-of-plane hinges x3,
   friction 0.2, lift right after the lash, lift contacts counted in the strike.
6. Anchor snap after a knock: the weld exploded (qvel 1e5 within 2 ms). Replaced it
   with the rate-limited follower.
7. 95 deg back-swing: at high omega the swing (3 ms) is much shorter than the chain's
   response, so the whip barely followed. Increased to 160 deg. Beyond ~1000 rad/s
   impulse stays at ~0.8-1.0 uN*s (2500 rad/s is no stronger). A 6 mm lunge gives
   the best absurd; 10 mm scatters the strike.

## Performance

Measured on an idle M-series Mac (best of 3): no whip 136 us/step; whip in the model
without hooks 162 us/step (+26 us, the 30 extra DoFs and the weld in `mj_step`); whip
attached and idle 166 us/step (+4 us of hooks, which run every 10th step when idle),
so about +22 % overall. During the ~0.33 s of a crack: 232 us/step (per-step handle
updates, contact scanning, whip-fly contacts).

## Limitations

* The absurd level delivers ~1 uN*s, not the shove's 2.5 uN*s. The fly leaves the
  contact at roughly the whip's local speed, and the chain can't be driven much
  faster at this timestep and stiffness. It still launches the fly 4-9 cm.
* Overhead impulses (0.25-1.6 uN*s) partly press the fly into the ground. The ground
  takes most of that, so they tip it far less than lateral cracks of the same impulse.
* `random` hits glance more (cos ~0.5-0.7): the extra descent and oblique azimuths put
  part of the impulse downward or sideways. The event reports the measured direction.
* Whip-ground collision is off by default: the whip can dip through bumps or rocks
  visually. `collide_terrain=True` enables it, but that setting is not calibrated.
* A crack takes ~0.17 s from request to strike (wind-up) and ~0.33 s in total. The
  strike can't be instantaneous.
* After the fly is knocked over or launched, the returning whip can touch it (counted
  as stray contacts, not as hits). A fly flung more than ~300 mm/s is followed at
  300 mm/s.
* The follow camera shows the whip, but the pivot of left/right cracks sits near the
  right image edge.

## Commands

```bash
.venv/bin/python scripts/demo_whip.py                       # one crack per level x side
.venv/bin/python scripts/demo_whip.py --calibrate --phases 8 --sides left right front rear overhead random
.venv/bin/python scripts/demo_whip.py --calibrate --phases 4 --terrain normal --pre 1.5
.venv/bin/python scripts/demo_whip.py --record whip.mp4 --frames-dir frames \
    --sequence 1:left 2:right 2:front 2:rear 2:overhead 3:left 4:front
.venv/bin/python scripts/run_sim.py --headless --terrain normal --auto-perturb --hit-mode whip --max-seconds 60
.venv/bin/python -m pytest -q tests/test_whip.py
```
