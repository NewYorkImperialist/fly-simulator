# Taste patches: sugar and bitter spots tasted with the legs

`--taste-patches` puts coloured spots on the ground. When the fly's legs stand on one,
the legs "taste" it and the FlyWire brain model gets gustatory input. With
`--brain-actions`, the brain's proboscis motor neuron MN9 makes the fly stop and feed
(the proboscis extends with the full body, the CLI default). Bitter spots produce no
avoidance: the connectome gives none (see below). But bitter mixed into sugar
suppresses MN9, and with it the feeding.

* Code: `perpetualfly/senses/taste.py` (patches, sensing, feeding rule). There are
  also small additive hooks:
  * `perpetualfly/brain/mapping.py`: the `taste` stimulus kind and the named set
    `leg_gustatory`.
  * `perpetualfly/app.py` / `config.py`: `AppConfig.taste`, keys 5 / 6 / 7, the HUD
    line, the summary.
* Tests: `tests/test_taste.py`. It uses the synthetic brain or a fake MN9 and takes
  about 15 s.

```bash
.venv/bin/python scripts/run_sim.py --taste-patches --brain-actions            # patches + feeding
.venv/bin/python scripts/run_sim.py --taste-patches --taste-density 0 --brain-actions \
    --terrain flat --script-keys "0.2:5"                                          # one sugar spot ahead
```

## Patches (visual only)

| kind | colour | tastes |
|---|---|---|
| sugar | pale yellow / white with a sheen (emissive, specular material) | sugar |
| bitter | dark olive-brown, matte | bitter |
| mixed | amber | sugar + bitter |

* **Geoms.** The patches are MuJoCo cylinders (disc radius 1.8–2.6 mm, 0.03 mm
  thick, drawn 0.005 mm above the ground). They come from a small pool of their own:
  24 procedural + 6 spawned. A world extension adds the pool with `contype =
  conaffinity = 0`, so the discs never collide. `ProceduralTerrain` / `chunks.py`
  are untouched.
* **Physics check.** Walking with patches on (including over a patch) is
  bit-identical to walking without the flag (`test_patches_are_visual_only_...`).
* **Z-fighting.** Thinner or lower discs z-fight with the floor at the default
  camera distance, which shows as blotchy discs in rendered frames.
* **Procedural placement.** Patches are placed in 20 mm cells along x. The count
  per cell is Poisson with mean `density_per_cm * 2` (default 0.3 per 10 mm; set it
  with `--taste-density`, 0 = only spawned patches).
  * Positions are uniform within ±3 mm of the fly's path. The path is the fly's y
    when the cell is loaded, like the terrain's lateral re-centring.
  * Kinds are drawn 50 % sugar, 30 % bitter, 20 % mixed. Patches in a cell don't
    overlap, and there are none before x = 8 mm.
  * The layout is a pure function of (seed, cell, y).
  * Cells load from 15 mm behind to 40 mm ahead of the thorax and are recycled as
    the fly walks. A reset relays out everything and drops spawned patches.
* **Terrain.** Discs sit on `Session.ground_height` (terrain ray cast), re-seated
  every 0.2 s because terrain chunks recycle. A disc on a bump is partly hidden.
* **Keys.** 5 / 6 / 7 spawn a sugar / bitter / mixed patch whose near edge is 3 mm
  ahead of the thorax, along the heading. The oldest spawned patch is recycled.
  Every spawn writes a `taste_spawn` row to events.csv.

## Sensing

* **Which legs taste.** Every 10 ms of sim time, a leg tastes a patch when both of
  these hold:
  * the leg touches something: a tibia / tarsus contact, the same test the action
    library uses;
  * its tarsus tip (`<leg>_tarsus5`) lies inside the disc and within 0.3 mm above
    its surface.
* **Swing hold.** A leg keeps tasting for 0.1 s after it lifts off, so the count
  doesn't flicker with the gait.
* **What gets sent.** While any leg tastes something, the sensor sends
  `StimulusEvent("taste", side, intensity, duration_s=0.15, details=...)`:
  * `details["tastes"]` is ⊆ {sugar, bitter}, plus `<taste>_hz`, `<taste>_legs`, a
    label such as `TASTE SUGAR 2`, and `proxy`;
  * rate per taste = `200 Hz * min(n_legs / 3, 1)`: one leg 67 Hz, two 133 Hz, three
    or more 200 Hz;
  * `side` = the touching legs' side (`none` when both sides touch).
* **Sustained drive.** The event is re-sent every 0.1 s, and immediately when the
  leg count or side changes. The drive therefore lasts while the contact lasts and
  stops about 0.15 s after it ends.
* **Logging.** Only the onset of a taste (or a taste added or lost) goes through
  `BrainLink.send`: a brain-window chip and a `brain_stim` row. Refreshes go to the
  brain only.
* **events.csv rows:**
  * `taste` / `taste_off` when the tasting legs change;
  * `action_start` / `action_end` with `action=feed` for feeding bouts.
* **summary.json** gets a `taste` block: contact seconds per taste, event counts,
  feeding bouts, config.
* **HUD:** `TASTE sugar 2 legs 133Hz  (labellar GRN stand-in)  MN9 62Hz  FEEDING 1.3s`.

### Which neurons stand in for leg taste (honesty note)

FlyWire v783 gustatory neurons present in the brain dataset (cell_class `gustatory`):

| sub-class | n (L / R) | organ | reaches MN9 in the model? |
|---|---|---|---|
| sugar/water (LB3 64 / 58, LB2d 3 / 4) | 67 / 62 | labellum | **yes**: Shiu's 20 left LB3 give MN9 54 Hz at 100 Hz, 76 Hz at 200 Hz. The right LB3 set **ignites the runaway state** at 100 Hz |
| bitter (LB1a–e) | 32 / 33 | labellum | no (MN9 0); inhibits sugar → MN9 |
| taste peg | 36 / 35 | labellum (inner) | weakly (MN9 7–40) |
| low-salt | 9 / 10 | labellum | weakly (MN9 5–8) |
| pharyngeal / accessory pharyngeal nerve groups | ~24 / 26 | pharynx | MN9 5–18, often runaway |
| **SA_VTV_pro_meso_meta** (sensory_ascending, gustatory) | 37 / 37 | **legs** (pro-, meso-, metathoracic; ascend via the VTV tract) | **no** (MN9 0) |

* **Leg taste is mostly missing from the model.** Most leg (tarsal) gustatory
  neurons end in the VNC, which the model does not contain. Tarsal sugar would reach
  the feeding circuit through ascending neurons that are not identified in the
  brain-only model.
* **The ascending leg afferents are the only leg taste neurons here, and they are
  unusable.** Their taste modality is not annotated. They drive MN9 not at all and
  the walk / turn DNs only weakly (table below). The left set ignites the model's
  self-sustained runaway state at 100 Hz.
* **So the default uses the labellar GRNs as a stand-in for leg taste.** These are
  the Shiu et al. 2024 validated labellar sets: 20 sugar GRNs (LB3,
  `mapping.SUGAR_GRN_IDS`) and 20 bitter GRNs (LB1a–d, `mapping.BITTER_GRN_IDS`).
  * Both sets lie on the **fly's left** labellar nerve. The event's side does not
    change which GRNs are driven; it is only recorded.
  * The right-side LB3 population is not used because of its runaway.
  * The HUD and every event say so: `labellar GRN stand-in` / `proxy`.
* **Optional leg afferents.** `TasteConfig.leg_afferent_hz > 0` additionally drives
  the `leg_gustatory` set (the SA_VTV gustatory afferents) on the touching side. It
  is off by default for the reasons above.

## Feeding rule (ours) on real MN9 activity

With `--brain-actions`:

* **Start.** When the fly has tasted sugar within 0.3 s and the newest BrainState
  (0.1 s window) has MN9 > `feed_mn9_hz` (30 Hz), the fly starts **`Feed`**:
  * the freeze stance (legs in stance keep their targets, swing legs are put down,
    all tarsi adhere);
  * proboscis extension (rostrum −1 rad, haustellum +1 rad) when the body has the
    proboscis joints.
* **End.** Feeding ends in one of two ways:
  * MN9 has stayed ≤ 30 Hz for `feed_release_s` (0.3 s): reason `mn9_low`, then a
    0.5 s pause before the next bout;
  * after `feed_max_s` (4 s), a stand-in for satiation: reason `max_bout`. The fly
    then walks on, and no new bout starts for `feed_refractory_s` (2.5 s). This lets
    it leave the patch.
* **Priority.** It never interrupts another action (jump, groom, …) except the
  brain's own MN9 → `ProboscisExtend`, which it replaces. The fall detector treats
  `feed` as a stationary action, so standing still is not "stuck".
* **Latency.** The fast-path MN9 trigger (20 ms window, where one spike = 50 Hz) is
  deliberately **not** used to start feeding. In the sugar + bitter mixture, single
  MN9 spikes crossed it and caused 0.3 s "feeding" twitches while the 0.1 s rate
  stayed at 4–25 Hz. The brain's plain proboscis trigger still uses the fast path.

The MN9 activity is the model's. The stop, the bout cap and the refractory period are
our behavioural rule: the connectome model has no locomotor "stop" output for feeding
that the readout could use.

## Verification with the real brain (FlyWire v783, headless)

### Offline dose / mixture probe

Setup: in-process `LIFEngine`, Poisson input for 0.5 s, 2–3 trials, mean rates in
Hz. "Runaway" means more than 50k spikes/s persist after the input stops, as in
docs/SENSORY_SCREEN.md.

| stimulus | MN9 | walk L/R | turn L/R | MDN | GF | runaway |
|---|---|---|---|---|---|---|
| sugar (Shiu 20) 50 / 67 / 100 / 133 / 150 / 200 Hz | 11 / 36 / 54 / 63 / 66 / 76 | 0 | 0 | 0 | 0 | no |
| bitter (Shiu 20) 50 / 100 / 200 Hz | 0 | 0 | 0 | 0 | 0 | no |
| all bitter, left (32) 150 Hz | 0 | 5.0 / 6.7 | 6.5 / 2.0 | 0 | 0 | no |
| all bitter, right (33) 150 Hz | 0 | 4.3 / 6.0 | 3.5 / 1.0 | 0 | 0 | weak persistence (7k sp/s) |
| LB3 right (58) 50 / 100 Hz | 6 / 20 | 0 | 0 / 0, 26 / 3.5 | 0 | 0 | at 100 Hz: **yes** |
| sugar + bitter 67 + 67 / 100 + 100 / 200 + 200 | 3 / 7 / 9 | 0 | 0 | 0 | 0 | no |
| sugar + bitter 133 + 67 / 200 + 133 / 200 + 67 | 32 / 30 / 60 | 0 | 0 | 0 | 0 | no |
| leg gustatory SA_VTV left 50 / 100 Hz | 0 / 0 | 0 / 1.3–2 | 0 / 8, 1.5 | 0 | 0 | at 100 Hz: **yes** |
| leg gustatory SA_VTV right 100 Hz | 0 | 7 / 10 | 11 / 3 | 0 | 0 | no |
| sugar 150 + leg gustatory left 50 | 62.5 (sugar alone 66) | 0 | 0 | 0 | 0 | no |

What the probe shows:

1. **Sugar → MN9 is graded with the number of legs.** One leg (67 Hz) gives MN9
   about 36 Hz, just above the 30 Hz feeding threshold. Two or more legs give
   63–76 Hz. Sugar does not touch any locomotor DN.
2. **Bitter drives no avoidance DN.** The Shiu bitter set drives nothing downstream
   of the readout. The whole left bitter population gives only 5–7 Hz on the walk
   DNs and a 6.5 Hz left turn (DNa02_L, same side). There is no MDN (backing up) and
   no escape. So bitter patches have **no body effect**; the fly walks across them.
   This is an honest negative.
3. **Bitter suppresses sugar-evoked MN9.** This is the Shiu et al. result: equal
   sugar and bitter drive cut MN9 from 54–76 Hz to 3–9 Hz. A mixed patch (same legs
   on both tastes) therefore gives no feeding. Sugar on three legs with bitter on one
   (200 + 67) still reaches 60 Hz.

### In the app

Setup: headless, flat terrain, full body, `--brain-actions`. One patch of radius
2.4 mm, centred 4.5 mm ahead of the spawn point; the fly walks straight onto it.

| patch | legs on the patch | MN9 (BrainState) | body |
|---|---|---|---|
| sugar | first contact at t = 0.17 s (front legs) | 80–90 Hz from the first state after contact; mean 62, max 95 Hz during the stop | MN9 fast trigger → brain `ProboscisExtend` at 0.21 s; **FEED at 0.38 s** (stop + proboscis). Stood 4.0 s (max bout, drift 0.08 mm), then walked on at about 14 mm/s. While crossing, the brain re-extended the proboscis (MN9 still high); MN9 fell to 0 about 0.25 s after the last leg left |
| bitter | 0.17–0.77 s (walked across) | 0 | no DN response (walk / turn / MDN 0), no change in gait |
| mixed | 0.17–0.77 s (walked across) | mean 4, max 10 Hz | **no feeding stop, no proboscis**; walked across |

**Rendered frames** (renderer PNGs, read back and checked):

* The fly stands on a sugar spot with its front legs, proboscis swung down onto the
  spot (front three-quarter follow view during `FEED`).
* Sugar, bitter and mixed spots lie on normal terrain next to the walking fly. An
  earlier version drew the discs 0.006 mm above the floor and showed z-fighting
  blotches; they are now 0.02 mm up and 0.03 mm thick.

## Configuration (`AppConfig.taste`, `TasteConfig`)

| group | key (default) | meaning |
|---|---|---|
| patches | `density_per_cm` (0.3), `cell_mm` (20), `band_mm` (3), `radius_mm` (1.8, 2.6), `kind_weights` (sugar 0.5 / bitter 0.3 / mixed 0.2), `first_x_mm` (8), `ahead_mm` / `behind_mm` (40 / 15), `pool_size` / `spawn_pool` (24 / 6), `spawn_distance_mm` (3), `seed` (7) | procedural placement and pools |
| sensing | `sense_every_s` (0.01), `refresh_s` (0.1), `stim_duration_s` (0.15), `contact_height_mm` (0.3), `leg_hold_s` (0.1), `legs_full` (3), `max_rate_hz` (200), `leg_afferent_hz` (0 = off) | legs → brain |
| feeding | `feed` (true), `feed_mn9_hz` (30), `feed_release_s` (0.3), `feed_max_s` (4), `feed_refractory_s` (2.5), `feed_taste_window_s` (0.3) | the feeding rule (needs `--brain-actions`) |

## Limitations

* **Stand-in neurons.** Leg taste is represented by labellar GRNs on one side. Real
  tarsal taste input and its lateralisation are not in the brain-only model.
* **No bitter avoidance.** The readout DNs show no avoidance for bitter, so none is
  shown. A behavioural rule for it would be invented, not connectome-driven.
* **Our rule, not the brain's.** The feeding stop, the 4 s bout cap and the
  refractory period are our rules. The model has no satiety, no adaptation and no
  locomotor stop signal for feeding.
* **Visual only.** Patches don't change friction or adhesion, so the fly doesn't
  stick to sugar.
* **Terrain.** A disc can be partly hidden inside a terrain bump or rock. Patches are
  not available with `--course` / `--job` (ConfigError).
* **Brain required for any body effect.** Without `--brain`, patches and sensing
  still work (HUD, events.csv), but nothing drives the body. Without
  `--brain-actions`, the brain gets the taste input but the fly does not stop.
