# Squash damage (`--injury`)

A swatted, whipped or crushed fly should not walk on unharmed. With `--injury` the
measured contact forces on the fly damage its legs, body and wings. The fly limps,
holds a lame leg up, walks slower and rests, lies stunned after a heavy hit, and is
squashed by a hard one (a flattened replica and a splat, then a counted respawn).
Everything heals over tens of seconds of sim time.

**This is a phenomenological model, not a biomechanical damage model.** There is
no cuticle, tissue or haemolymph mechanics. A measured force becomes a damage
number per body part (a force floor and a reference impulse, calibrated below). The
damage then sets a few bounded, reversible control and model parameters. The
squashed look is visual only.

Code: `fly_simulator/injury.py` (config, sensor, effects, decals), app wiring in
`fly_simulator/app.py` (`Session.injury`), tests in `tests/test_injury.py`. Off by
default: nothing is installed, and the default run is bit-identical (tested, and
checked against the previous commit: same qpos/qvel hash after 3 s on normal
terrain with 3 automatic whip hits).

```bash
fly-simulator --swatter --injury          # V swats, 1-4 set the level, TAB shows the bars
fly-simulator --injury --auto-perturb     # whip hits (they rarely hurt, see below)
fly-simulator --job dead_hang --injury    # any job / course / feature composes
```

## Damage

Parts: the 6 legs (coxa to tarsus5), the thorax (with the head, which is fused into
the thorax body), the abdomen, and the left and right wing.

A post-step hook reads every contact between a fly geom and a non-fly geom. It skips
tarsus-on-static-ground contacts, because that is walking. For each remaining
contact it takes the world-frame force (`mj_contactForce`) and adds its magnitude
to that part's total for the step. Two terms add damage (forces in body weights,
BW = 10.05 µN):

1. **Crushing** (every step): `D += max(0, F − floor) · dt / ref`.
   Floors: legs 15 BW, thorax / abdomen / wings 25 BW.
   Refs (excess impulse for D = 1): legs 5, thorax 25, abdomen 40, wings 20 BW·s.
2. **Sharp impact** (once per episode and part):
   `min(0.25, (peak − 150 BW) / 2000 BW)`. Whip lashes last 1–2 ms (0.01–0.2 BW·s)
   but reach 50–860 BW, so the impulse term alone would never register them.

**Wings** have no collision geoms in NeuroMechFly, so wing damage comes from
**dorsal crush**. Take the part of a thorax / abdomen contact force that pushes
from the dorsal side (body −z). It counts for the wing on the side of the contact
point.

An **impact episode** runs from the first force above a floor until 50 ms below
all floors. It becomes an `injury_impact` event with, per part: damage, peak force,
impulse, and the bodies that hit it. Moving bodies (swatter, whip, props) are
listed before the static ground. Each part's damage is capped at 1.5.

Note: when a plate presses the fly into the floor, the ground's reaction on the
body counts as well as the plate's force. Both are part of being crushed, and the
refs are calibrated on this total.

## Effects (all bounded, all undone on heal / reset / close)

| damage | effect |
|---|---|
| leg D | That leg's CPG stride amplitude × (1 − 0.6 D) and its 7 position actuators' kp × (1 − 0.5 D), both ≥ 0.35. The heading hold compensates the resulting turn, so the gait becomes asymmetric (a limp). |
| leg D ≥ 0.8 | **Lame**: the leg is held up (standing pose + coxa / femur / tibia pitch offsets, from a kinematic search for the highest tarsus) and its adhesion is off. It stays lame until D < 0.55 (hysteresis). |
| body `max(thorax, 0.7 abdomen)` | CPG frequency × (1 − 0.5 D_body), ≥ 0.5: slower stepping. |
| D_body ≥ 0.35 | **Rest bouts**: a freeze of 1.2 · (1 + 2 D_body) s after every 3 · (1 − (D_body − 0.35)) s of walking. |
| episode body damage (thorax + abdomen) ≥ 0.6 | **Stunned** for 3 s × min(2, body / 0.6): an action with leg kp × 0.3, all tarsi adhering and targets held, so the body sinks and the fly lies still. Then the ActionManager blends back to walking. |
| episode thorax / abdomen peak ≥ 4500 BW, or episode body damage ≥ 2.5, or cumulative thorax ≥ 1.3 | **Squashed**: the fly is hidden (its geoms go to render group 5), and a flattened fly replica and a goo splat (two mocap bodies, visual-only geoms, no mass, no collision) are shown for 2 s. Then comes an explicit reset (`Session.reset("squash")`, counted in `n_squash_resets`; `job.recover("squashed")` inside a job). The stain stays on the floor. |
| wing D (`--flight`) | That side's stroke amplitude is capped at base × (1 − 0.5 D): less lift on that side. Unit-tested on the hover controller's `_write`. Not flown in a measured run. |

The CPG hooks wrap `cpg_network.step`: the per-leg amplitude and the frequency
factor are multiplied in for that step only. This composes with `--stress`, which
scales `_base_intrinsic_freqs`. The kp change is written only while no action holds
boosted actuator parameters (the jump); the ActionManager restores the value it
saved.

## Healing

Starting 2 s after the last damage, every part heals linearly at 0.025 per s of
sim time: D = 1 heals in 40 s, and a limp from a lazy swat heals in ~25 s. A reset
or respawn gives a fresh fly (all damage cleared).

## Calibration (raw forces, damage off)

Headless, flat ground, default hybrid controller, no brain (the swatter's vision is
off, so the fly never dodges). Peak = max per-step force on the part; impulse =
∫F dt over the whole event, including the ground's reaction.

| event | peak (BW) | impulse (BW·s) |
|---|---|---|
| walking 4.5 s | none (no non-tarsus contact at all) | 0 |
| jump + landing | mid legs 5–7 | 0.009 |
| whip L1 (left / overhead / rear) | 55–71 (thorax, abdomen) | 0.01–0.03 |
| whip L2 | 69–127 | 0.02–0.03 |
| whip L3 | 200–270 (front leg 247) | 0.07–1.2 (incl. lying on its back afterwards, ~1 BW) |
| whip L4 | 410–860 (front leg 864 overhead) | 0.1–1.0 |
| swat L1 lazy (42 µN·s) | thorax 470–1050, hind leg 530–720 | thorax 4.2–4.7, rh 2.3, wings 0.4–3.4 |
| swat L2 normal (155–170 µN·s) | thorax 1400–1640, abdomen 670–1560, hind legs 770–1230 | thorax 16–19, abdomen 4–7, legs 3–6, wing 14–15 |
| swat L3 quick (250–280 µN·s) | thorax 1900–3400, abdomen 2500–3240 | thorax 11–12, abdomen 32–38, wing 19–23 |
| swat L4 lightning (245–260 µN·s) | **thorax 6000–8650**, abdomen 3700–3800, legs up to 6400 | thorax 8–12, abdomen 29–36, wing 17–19 |

The floors sit well above walking, jumps and gentle whip cracks. The refs were
chosen so that L1 limps, L2 / L3 stun (and cripple a hind leg), and L4 squashes. L3
and L4 carry the same impulse, and only the peak separates them, which is why
squashing is a peak criterion (L3 ≤ 3434 BW, L4 ≥ 5991 BW: threshold 4500 BW,
tested at 2 gait phases each).

## Measured results (`--injury`, same setup, flat ground)

Speed = thorax displacement over 1 s windows. Stride = the tarsus5 fore-aft range
in the thorax frame per leg. Before every hit: 14.1 mm/s, stride LF .81 LM .96 LH .78
RF .80 RM .96 RH .78 mm.

**Swatter, from behind:**

| level | episode | after | recovery |
|---|---|---|---|
| L1 lazy | +1.6 (rh 0.57, thorax 0.39, wings 0.30 / 0.15) | limping 9.4–10.2 mm/s, step ×0.81. RH stride 0.78 → 0.55 (−30 %), LH 0.62. One rest bout. | 11.3 mm/s at +10 s, 13.7 at +20 s, OK and 14.0–14.1 at +25–28 s |
| L2 normal | +4.9 (lh 1.05, rh 0.93, thorax 0.53, abdomen 0.78, r_wing 1.0) | **stunned 6 s** (0 mm/s), both hind legs **lame** (stride 0.10–0.12), rest bouts. 10–11 mm/s between rests. | lame until +22 s; limping 11.8 mm/s at +30 s (lh 0.37) |
| L3 quick | +5.0 (abdomen 1.02, thorax 0.69, lh 0.81, r_wing 1.28) | **stunned 6 s**, LH lame, repeated rests; 7–7.6 mm/s between them (step ×0.72) | 9.3 mm/s at +17 s |
| L4 lightning | peak 8551 BW (another run: 8473) | **squashed**: replica + splat for 2 s, then respawn | fresh fly: 14.1 mm/s |

In the app (`--swatter --injury`, scripted V at L4 and then L2): L4 → squashed →
`squash_reset` → walking at 14.1 mm/s. The following L2 swat → stunned 5.9 s, one
lame leg. events.csv had `injury_impact`, `injury_lame`, `injury_squashed`,
`injury_respawn`, `squash_reset` and `injury_stunned` rows, and summary.json an
`injury` block.

**Whip** (1 crack, from the left and overhead): L1: no damage. L2: ≤ 0.04 (an
overhead crack that tipped the fly). L3: +0.07 to +0.14 (lf 0.06), 13.7 mm/s for
~2 s, then OK. L4 overhead: +0.83 (lf 0.27, rh 0.26, thorax 0.19), limping at
12.0–12.3 mm/s, step ×0.90. L4 left: +0.2 (it also tipped the fly on its back,
which the default controller never recovers from; the app's auto reset does).
So the whip stays mostly a pain / arousal stimulus (docs/STRESS.md): its impulse is
100–1000× smaller than a swat's.

**Lame leg alone** (damage set to 1, healing off, 3 s): LF lame 13.4 mm/s, RH lame
12.5 mm/s (vs 14.1). Heading held.

**Other**: dead_hang job, 25 s: no damage (the bar is static and the tarsi on it
are skipped). With injury on but no hits, the fly's trajectory is bit-identical to
injury off (test). Cost: +8 % wall time per sim second (1.77 → 1.92 s, normal
terrain).

**Frames checked** (renderer PNGs, 480 × 320, follow camera 6–8 mm): stunned (body
lowered, wings splayed after the L2 swat); lame LF (held up in front of the head),
lame RH (held up behind the abdomen); squashed (a flat brown fly with splayed legs
and wings, red eyes, on a yellow-green splat with droplets, the swatter handle
leaving). The 2-line HUD shows the bars, e.g. `legs LF#####  LM#####  LHx.....`.

## HUD, logs

* TAB (full HUD), two lines:
  `INJURY LAME (model)  thorax ##... abdomen ####. wings #../##.  step x0.81  hits 1 stuns 1 squashed 0`
  and `legs LF#####  LM#####  LHx.....  RF#####  RM###..  RHx.....`
  (5 chars = health 1 − D, `x` = lame). Status: OK / limping / lame / resting /
  stunned / squashed.
* events.csv: `injury_impact` (total, damage / peak_bw / impulse_bws per part,
  sources, body damage and peak), `injury_lame`, `injury_recovered`, `injury_rest`,
  `injury_stunned`, `injury_squashed`, `injury_respawn`, and the reset row
  `squash_reset`.
* metrics.csv: `injury_status, injury_max_leg, injury_thorax, injury_abdomen,
  injury_wings, injury_freq_mult`.
* summary.json `injury`: damage, max_damage, n_impacts / n_minor_impacts / n_lame
  / n_rests / n_stuns / n_squashes / n_squash_resets, config.
* Config keys: `injury.*` in `AppConfig` (all fields of `InjuryConfig`, e.g.
  `heal_rate_per_s`, `squash_peak_bw`, `leg_ref_bws`, `splat_visual`).

## Limitations

* This is a phenomenological mapping with free parameters, not a validated injury
  model. Real flies lose tarsi / legs, and they bleed and die. Here everything
  heals.
* The thresholds come from 2 gait phases per swat level, from behind, on flat
  ground, with vision off (no dodging). Other sides and terrains give other forces.
* Wing damage is a proxy (dorsal crush), and its flight effect (the amplitude cap)
  is unit-tested only. No flight run was measured.
* `--stress` is not coupled. Damage does not add to the octopamine level, which
  comes from the brain model. The whip / swat hits reach it through the existing
  relay.
* Jobs: a stun replaces the job's running action (the dead_hang grip, for
  example), so a stunned hanging fly falls. Rests only start when no action is
  running. Job props (bowling pins, the flytrap) damage the fly only through
  measured contacts above the floors. Only dead_hang was run (no damage in 25 s).
* The squashed replica is a stylised stand-in (ellipsoids and capsules), not the
  deformed NeuroMechFly mesh.
* A fly knocked onto its back is not righted by injury; the app's auto reset (or X)
  handles it as before.
