# Perturbation ("whip") calibration

Strength levels used by `fly_simulator.interaction.Perturbation` (defaults in
`perturbation.py::_default_levels`). Forces act on the thorax body COM
(`data.xfrc_applied`, world frame) as a constant force for the given duration.
Body weight (BW) = `fly_mass * 9810 mm/s^2` = 1.024e-3 g x 9810 = **10.05 uN**.

| level | force | duration | impulse | intended effect |
|---|---|---|---|---|
| 1 gentle | 0.75 BW = 7.5 uN | 20 ms | 0.15 uN*s | measurable wobble, never tipped in 96 trials |
| 2 medium | 1.5 BW = 15.1 uN | 20 ms | 0.30 uN*s | clear stumble / body roll, usually recovers (~17 % knocked over) |
| 3 hard | 3 BW = 30.1 uN | 20 ms | 0.60 uN*s | tips the fly over in ~2/3 of hits (lateral: 75 %); about half stay on their back |
| 4 absurd | 10 BW = 100.5 uN | 25 ms | 2.5 uN*s | sends it flying (0.3-0.6 m sideways, ~0.27 m up) |

## Method

`scripts/demo_perturbation.py --calibrate --phases 8 --post 2.0` (headless, flat ground,
default hybrid controller with heading hold, FlyGym timestep 1e-4 s):

* fresh `Simulation` per trial, walk 1.0 s (+ k x 10.5 ms, k = 0..7, i.e. 8 gait
  phases spanning half a ~83 ms tripod cycle; the gait is mirror-symmetric after half a
  cycle, so left at phase k equals right at phase k+4), hit, observe 2 s;
* each hit is compared with the *unperturbed* run of the same phase (the sim and
  controller are deterministic, so the difference is caused by the hit);
* "random" = random azimuth + 10-35 deg upward elevation, a different draw per phase.

Columns (mean over the 8 phases, max in parentheses):

* deviation: peak horizontal distance between the perturbed and unperturbed thorax (mm);
* peak lateral disp.: peak |thorax displacement| perpendicular to the pre-hit travel
  direction, *including* normal gait sway (unperturbed: 0.32-0.40 mm);
* extra tilt: peak (tilt - unperturbed tilt), tilt = angle between thorax up-axis and
  world up (unperturbed walking: 2.5-7.3 deg);
* peak rise: peak thorax height above its pre-hit height (mm);
* tipped: tilt ever > 60 deg (on its side/back); still down: tilt > 60 deg at the end;
* walking again: upright (tilt < 30 deg for the last 0.5 s) and moving > 7 mm/s.

Peak horizontal thorax speed within 0.3 s of the hit (1 ms samples; unperturbed gait
peak 33.6 mm/s): gentle 41 mm/s (max 60), medium 66 (max 205), hard 242 (max 604),
absurd ~2070 mm/s.

## Results (8 gait phases per row, 2 s observation)

| level | force | dur. | impulse | direction | n | deviation from unperturbed path (mm) | peak lateral disp. (mm) | extra tilt (deg) | peak rise (mm) | tipped >60 deg | still down at end | walking again within 2 s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 gentle | 0.75 BW = 7.5 uN | 20 ms | 151 nN*s | left | 8 | 0.14 (0.21) | 0.43 (0.54) | 3 (6) | 0.07 (0.12) | 0/8 | 0/8 | 8/8 |
| 1 gentle | 0.75 BW = 7.5 uN | 20 ms | 151 nN*s | right | 8 | 0.14 (0.21) | 0.43 (0.57) | 3 (6) | 0.07 (0.12) | 0/8 | 0/8 | 8/8 |
| 1 gentle | 0.75 BW = 7.5 uN | 20 ms | 151 nN*s | forward | 8 | 0.11 (0.15) | 0.46 (0.64) | 3 (4) | 0.06 (0.10) | 0/8 | 0/8 | 8/8 |
| 1 gentle | 0.75 BW = 7.5 uN | 20 ms | 151 nN*s | backward | 8 | 0.11 (0.13) | 0.40 (0.54) | 2 (3) | 0.07 (0.12) | 0/8 | 0/8 | 8/8 |
| 1 gentle | 0.75 BW = 7.5 uN | 20 ms | 151 nN*s | up | 8 | 0.19 (0.36) | 0.49 (0.57) | 5 (8) | 0.20 (0.39) | 0/8 | 0/8 | 8/8 |
| 1 gentle | 0.75 BW = 7.5 uN | 20 ms | 151 nN*s | random | 8 | 0.11 (0.22) | 0.44 (0.58) | 3 (9) | 0.09 (0.16) | 0/8 | 0/8 | 8/8 |
| 2 medium | 1.5 BW = 15.1 uN | 20 ms | 301 nN*s | left | 8 | 4.04 (30.13) | 0.96 (4.03) | 28 (171) | 0.30 (1.65) | 1/8 | 1/8 | 7/8 |
| 2 medium | 1.5 BW = 15.1 uN | 20 ms | 301 nN*s | right | 8 | 4.13 (30.03) | 1.10 (4.54) | 30 (166) | 0.33 (1.67) | 1/8 | 1/8 | 7/8 |
| 2 medium | 1.5 BW = 15.1 uN | 20 ms | 301 nN*s | forward | 8 | 0.16 (0.17) | 0.45 (0.65) | 5 (7) | 0.07 (0.10) | 0/8 | 0/8 | 8/8 |
| 2 medium | 1.5 BW = 15.1 uN | 20 ms | 301 nN*s | backward | 8 | 7.73 (30.41) | 1.27 (3.45) | 48 (170) | 0.37 (0.93) | 2/8 | 2/8 | 6/8 |
| 2 medium | 1.5 BW = 15.1 uN | 20 ms | 301 nN*s | up | 8 | 8.18 (32.02) | 1.49 (4.38) | 48 (169) | 0.45 (0.94) | 2/8 | 2/8 | 6/8 |
| 2 medium | 1.5 BW = 15.1 uN | 20 ms | 301 nN*s | random | 8 | 4.61 (32.78) | 1.03 (3.91) | 36 (171) | 0.36 (1.31) | 2/8 | 1/8 | 7/8 |
| 3 hard | 3 BW = 30.1 uN | 20 ms | 603 nN*s | left | 8 | 17.77 (35.69) | 13.90 (26.96) | 118 (172) | 0.78 (1.48) | 6/8 | 3/8 | 5/8 |
| 3 hard | 3 BW = 30.1 uN | 20 ms | 603 nN*s | right | 8 | 24.18 (34.07) | 13.75 (26.69) | 132 (176) | 0.78 (1.26) | 6/8 | 6/8 | 2/8 |
| 3 hard | 3 BW = 30.1 uN | 20 ms | 603 nN*s | forward | 8 | 12.26 (25.90) | 1.19 (2.22) | 98 (170) | 0.42 (0.87) | 4/8 | 4/8 | 4/8 |
| 3 hard | 3 BW = 30.1 uN | 20 ms | 603 nN*s | backward | 8 | 19.87 (48.07) | 2.51 (6.01) | 92 (171) | 0.62 (1.56) | 4/8 | 4/8 | 4/8 |
| 3 hard | 3 BW = 30.1 uN | 20 ms | 603 nN*s | up | 8 | 16.29 (34.23) | 2.20 (4.73) | 90 (176) | 0.81 (1.41) | 4/8 | 4/8 | 4/8 |
| 3 hard | 3 BW = 30.1 uN | 20 ms | 603 nN*s | random | 8 | 28.07 (56.42) | 8.97 (23.84) | 151 (174) | 1.04 (2.17) | 7/8 | 6/8 | 2/8 |
| 4 absurd | 10 BW = 100.5 uN | 25 ms | 2512 nN*s | left | 8 | 451.68 (617.18) | 447.79 (608.68) | 171 (177) | 23.26 (40.76) | 8/8 | 5/8 | 3/8 |
| 4 absurd | 10 BW = 100.5 uN | 25 ms | 2512 nN*s | right | 8 | 492.78 (635.80) | 484.64 (628.94) | 172 (173) | 21.00 (40.08) | 8/8 | 7/8 | 1/8 |
| 4 absurd | 10 BW = 100.5 uN | 25 ms | 2512 nN*s | forward | 8 | 336.12 (450.72) | 65.35 (103.52) | 171 (175) | 40.70 (47.44) | 8/8 | 7/8 | 1/8 |
| 4 absurd | 10 BW = 100.5 uN | 25 ms | 2512 nN*s | backward | 8 | 400.76 (471.64) | 68.70 (127.33) | 170 (176) | 28.65 (38.64) | 8/8 | 5/8 | 3/8 |
| 4 absurd | 10 BW = 100.5 uN | 25 ms | 2512 nN*s | up | 8 | 49.07 (91.73) | 16.25 (30.42) | 171 (174) | 273.16 (275.93) | 8/8 | 7/8 | 1/8 |
| 4 absurd | 10 BW = 100.5 uN | 25 ms | 2512 nN*s | random | 8 | 607.92 (710.64) | 336.22 (560.77) | 172 (175) | 45.19 (72.97) | 8/8 | 7/8 | 1/8 |

## Exploration that led to these values

Single-phase and 3-4-phase sweeps (see `--grid MAG:DUR`), 2 s or 1.5 s observation:

| force x duration | lateral (left/right) | forward | backward / up |
|---|---|---|---|
| 1 BW x 20 ms | never tipped (16/16 fine) | fine | fine, but **random 3/24 tipped** (upward + backward component at one fragile phase) |
| 0.75 BW x 20 ms | fine | fine | 0/48 tipped incl. 16 phases x (random, up, backward) |
| 1.5 BW x 15 ms | fine | fine | 1/4 flipped backwards |
| 2 BW x 20 ms | 2/8 tipped | fine | 2/8 tipped (used to be the medium candidate: too harsh) |
| 2.5-3 BW x 10-20 ms | tips ~50 % | tips ~50 % | tips ~50 % |
| 4 BW x 20 ms | 7/8 tipped, flung 30-50 mm | 4/8 | 8/8 |
| 16 BW x 20 ms | flung 0.77 m | - | 0.47 m up |

Observations:

* **The response is strongly phase dependent and close to all-or-nothing.** Below a
  threshold the adhesive tripod holds (the body just sways a few tenths of a mm and rolls
  a few degrees); above it the stance legs lose grip and the fly rolls onto its side or
  back. The threshold depends on which tripod is in stance and where the feet are.
* Pure horizontal pushes are much better tolerated than pushes with an upward or
  backward component: unloading the legs (less normal force/adhesion) lets the fly
  pitch over backwards. At one phase a 1 BW hit with 22 deg elevation from behind-left
  flipped it, while the same force horizontal did nothing (probe over 8 azimuths x 2
  elevations). Forward pushes are the most benign.
* The duration matters less than the force above ~10 ms (e.g. 3 BW for 10 or 20 ms
  both tip it), consistent with a grip-breaking threshold.
* The FlyGym hybrid controller has **no righting reflex**: once on its back the fly
  stays there, legs cycling ("still down at end"). Some tipped trials count as
  "walking again" because the fly rolled a full turn and landed on its feet.
* No `SimulationInstabilityError` in any trial (up to 15 BW x 30 ms, 4.5 uN*s).
