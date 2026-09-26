# Brain playground: virtual optogenetics, virtual lesions, decision meters

The playground lets you work directly on the real FlyWire brain (Shiu et al. 2024 LIF
model, docs/BRAIN.md) while it drives the fly. You can switch neurons on
(optogenetics), switch them off (lesions), and watch the decision form in the
descending neurons before the body acts. It covers roadmap items C1 and C2.

Code: `perpetualfly/brain_viz/playground.py` (palette, spec parsers, decision meters,
clicks), `perpetualfly/brain/mapping.py` (`resolve_target`, the `opto` stimulus),
`perpetualfly/brain/engine.py` (`set_silenced`), `perpetualfly/brain/process.py`
(`set_lesions`, the `BrainState.playground` readout) and `perpetualfly/brain_link.py`
(scheduling, keys, window commands, logging). Tests: `tests/test_playground.py`.

## Using it

```bash
# brain window with the clickable palette; the brain steers and triggers actions
.venv/bin/python scripts/run_sim.py --brain-actions

# scripted: stimulate DNa02 (left) at 120 Hz for 1 s at t = 3 s, MDN at t = 6 s
.venv/bin/python scripts/run_sim.py --brain-steer --stim "DNa02_L:120:1.0@3,MDN@6"

# lesion the giant fibre, then show a looming stimulus at t = 3 s: no jump
.venv/bin/python scripts/run_sim.py --headless --brain-headless --brain-actions \
    --max-seconds 5.5 --terrain flat --lesion DNp01 --script-keys "3:o"
```

* **Brain window, mouse.** The window's bottom-left panel is the **palette** (it
  replaces the spike raster) and the top-right panel shows the **decision meters** (they
  replace the transmitter bars). Press **P** in the brain window to switch back to
  the classic panels, or start with `--no-playground`.
  * Left-click a target button to stimulate it at the selected rate and duration. The
    rate chips are 30 / 60 / 120 / 200 Hz and the duration chips are 0.5 / 1 / 2 / 5 s.
  * Right-click a button, or click its **LES** box, to lesion the target. Do it
    again to un-lesion it. Ctrl-click counts as a right-click (for trackpads).
  * Click a **region on the brain map** to stimulate every neuron whose main output
    neuropil it is. Right-click the region to lesion it.
  * **STOP STIM** ends every stimulation. **CLEAR LESIONS** removes every lesion.
* **Fly window, keys.** Every letter is already taken, so the playground uses
  **9** (select the next palette target; the HUD shows it), **0** (stimulate the
  selected target at 120 Hz for 1 s) and **-** (lesion or un-lesion it). They also
  work in `--script-keys` (`"3:9,3.1:0"`).
* **CLI.**
  * `--stim "TARGET[:RATE_HZ[:DURATION_S]][@RUN_TIME_S],..."`. The defaults are
    120 Hz and 1 s. Without `@T` the stimulus starts as soon as the brain is running.
  * `--lesion "T1,T2"` lesions from the start; `"MDN@4"` lesions from run time 4 s.
  * Both flags imply `--brain`. Config file equivalents:
    `{"brain": {"stim": "...", "lesion": "...", "stim_rate_hz": 120,
    "stim_duration_s": 1, "playground": true}}`.

### Target names (`mapping.resolve_target`)

The first match wins:

1. Aliases: `GF` / `giant_fiber` = DNp01, `BDN2` = DNg100, `oDN1` = DNg97, `P9` =
   DNp09, `MN9` (the proboscis motor neuron pair), `DNg12` (the whole DNg12_a–e
   grooming group), `OA` (all 43 `OA-*` neurons).
2. Descending groups: `walk`, `turn`, `backward`, `escape`, `groom`.
3. Named sensory sets: `sugar`, `bitter`, `LC4`, `LPLC2`, `body_mech`,
   `head_bristle`, `jo_wind_gravity`, `jo_auditory`, `an_walk`, ...
4. Regions: `region:GNG`, or a bare neuropil name (`GNG`, `AL_L`).
5. A FlyWire `cell_type` (`DNa02`, `OA-VUMa1`, `LC4`), then `hemibrain_type`.
   `PREFIX*` matches every type that starts with PREFIX.
6. `ids:ROOT_ID,ROOT_ID` for root ids.

A trailing `_L` / `_R` picks the fly's left or right side (FlyWire `side` column), as
in `DNa02_L`, `LC4_R` or `turn_L`. A name that matches nothing never crashes the brain:
the terminal and the experiment log print `no neurons match`.

The palette offers 16 targets: GF DNp01, MDN, BDN2, P9, DNa02 L, DNa02 R, DNa01+02 L,
DNg12, MN9, sugar GRNs, bitter GRNs, LC4 L, LPLC2, an_walk, OA-VUMa1 and JO wind.

## What "stimulate" and "lesion" mean in the model

* **Optogenetic stimulation (direct, not a natural sense).** Every targeted neuron
  gets Poisson input at the chosen rate, through the same mechanism Shiu et al. use
  for sensory drive. Each event adds 0.275 mV × 250 = 69 mV, far above the 7 mV gap
  to threshold, and driven neurons have no refractory period. So a targeted neuron
  fires at roughly the chosen rate, plus whatever synaptic input it receives. This
  resembles strong CsChrimson activation of a split-GAL4 line: the drive is direct,
  bypasses the sensory periphery, and is labelled "OPTO" everywhere (chips, log,
  events.csv).
  * Targets with more than 2,000 neurons (large regions) are driven through a fixed
    random sample of 2,000.
  * A fly reset clears active stimulation, together with every other stimulus.
* **Lesion (silencing).** The target's spike threshold is set to +∞ for the whole
  run (`LIFEngine.set_silenced`). The neurons still integrate their input but never
  fire, so they send no output and read 0 Hz. This is like expressing Kir2.1 or
  tetanus toxin in them. Stimulating a lesioned neuron does nothing.
  * Lesions **persist across fly and brain resets** until you remove them.
  * Lesions combine with the `--stress` octopamine layer (their threshold array is
    separate).
* **Bit-identical when unused.** With no lesion the engine passes exactly the same
  arguments to the compiled kernel as before. Adding a lesion and removing it again
  returns to that path. This is tested spike for spike
  (`test_engine_bit_identical_when_unused`). The `opto` stimulus kind is new, so
  nothing changes unless it is sent. `BrainState.playground` stays empty until the
  playground is used.

## Decision meters

These rows show what the brain is leaning toward, computed from the descending-neuron
readout. The white ticks mark the thresholds the app actually uses, so you can see a
decision building up before the body acts:

| meter | neurons | threshold marks (source) | body effect |
|---|---|---|---|
| ESCAPE | GF DNp01 | jump 60 Hz (`brain_triggers.jump_escape_hz`; lower under `--stress`) | jump (`--brain-actions`) |
| BACK UP | MDN mean | stop at ½·`backward_ref`, reverse at `backward_ref` (40 Hz; 20 Hz with `--brain-backup`) | stepping slows / reverses (`--brain-steer`) |
| WALK FASTER | BDN2, oDN1, P9 mean | `r_ref` 30 Hz (tanh scale of `descending_to_drive`) | up to +0.2 amplitude |
| TURN | DNa01/02 L − R (bipolar) | ±`r_ref` | steering (`--brain-steer`) |
| GROOM | DNg12 | 20 Hz for ≥ 100 ms | groom (`--brain-actions`; never interrupts another action) |
| FEED | MN9 | 30 Hz | proboscis extension (`--brain-actions`, full body) |
| AROUSAL | octopamine level (model) | 0.5 | only shown with `--stress` |

* The bar turns red and the lean word (JUMP, BACK UP / STOP, FASTER, LEFT / RIGHT,
  GROOM, EXTEND, AROUSED) appears once the threshold is crossed and that effect is
  active in this run.
* If the effect is not active in this run, the bar is amber and the word is in
  parentheses (for example jump without `--brain-actions`). Below threshold the row
  shows the percentage of the threshold reached.

## Record / log

* The experiment log sits at the bottom of the palette. It lists the most recent
  stimulations and lesions with their fly run times and sources (cli, key, window),
  for example `t=  3.00  STIM DNa02_L 120 Hz 1 s (cli)`. Clicks show `sent: ...`
  until the app confirms them. Errors (`no neurons match`) appear there too.
* `events.csv` rows:
  * `brain_opto`: target, rate_hz, duration_s, stim_time, source. `opto_stop` rows
    also have this event type.
  * `brain_lesion`: target, on, the active lesion set, stim_time, source.
* `summary.json` → `brain.playground` holds the final lesions and the whole log.
* The fly HUD shows `PLAYGROUND lesioned: … stim: … [9] <selected> [0] stim [-]
  lesion`. The brain map shows red `LESIONED: …` and blue `OPTO … Hz … s` pills,
  red × marks on silenced display neurons, rings on stimulated ones, and a red
  outline around lesioned regions. Screenshots (I) include the playground panels.

## Effects on the body (real brain, flat terrain, headless, one run each)

Command pattern: `scripts/run_sim.py --headless --brain-headless --max-seconds 5.5
--terrain flat <flags>`, stimulus at t = 3 s. Values are from `metrics.csv`:
* fwd = forward thorax velocity (mean over the second, with the minimum in brackets);
* yaw = heading change over the window (+ = left). The baseline drifts +5.3°/s;
* h = maximum thorax height (1.19 mm when walking);
* DN = peak group rates in 0.1 s states.

Before every stimulus: fwd 14.6 mm/s (min −2.4, stride oscillation), yaw +5.3°.

| experiment | flags | brain (peak) | body in the 1 s after the stimulus |
|---|---|---|---|
| stim DNa02 L 120 Hz 1 s | `--brain-steer --stim DNa02_L:120:1@3` | turn_L 90 Hz, turn_R 0 | **turns left**: yaw +19.5° (drift +5.3°); the heading hold pulls it back −9.3° the next second |
| stim MDN 120 Hz 1 s | `--brain-steer --stim MDN:120:1@3` | MDN L ≤ 175 Hz, drive −0.55 | **backs up**: fwd −5.0 mm/s (min −24.7) |
| stim BDN2 120 Hz 1 s | `--brain-steer --stim BDN2:120:1@3` | walk L/R 57/77 Hz, drive +1.14/+1.12 | **faster**: fwd 16.0 mm/s (+10 %) |
| stim GF 120 Hz 0.3 s | `--brain-actions --stim GF:120:0.3@3` | GF 150 Hz | **jump** 0.12 s after onset, h 3.55 mm |
| stim DNg12 60 Hz 1 s | `--brain-actions --stim DNg12:60:1@2.5` | DNg12 60 Hz | **groom** 0.1 s after onset |
| … + lesion DNg12_a | `--lesion DNg12_a` | DNg12 48 Hz (8 of 42 silenced) | still grooms |
| looming (O), control | `--brain-actions --script-keys 3:o` | GF 135–150 Hz | **jump** at +0.12 s, h 3.55 mm |
| looming + **lesion GF** | `… --lesion DNp01` | GF 0 Hz, MDN ~20 Hz | **no jump** (h 1.18 mm); MDN still slows it to 6.8 mm/s |
| looming, `--brain-backup` control | `--brain-backup --script-keys 3:o` | GF 150, MDN ≤ 35 Hz | fwd 0.5 mm/s (min −17.1), drive −0.12 |
| looming + **lesion MDN** | `… --lesion MDN` | MDN 0 Hz, GF 145 Hz | **no backing up, no slowing**: fwd 14.6 mm/s (min −2.5) |
| right-eye loom, control | `--brain-steer --stim "LC4_R:80:1@3,LPLC2_R:80:1@3"` | turn_L 55 Hz | turns left, away from the loom: yaw +17.9° |
| … + **lesion DNa02** | `--lesion DNa02` | turn_L 30 Hz | turn reduced: yaw +12.9° |
| … + lesion DNa01 + DNa02 | `--lesion DNa02,DNa01` | turn_L 0 Hz | no turn: yaw +3.0° (drift alone ≈ +5°) |

Other observations:

* sugar at 200 Hz together with DNg12 at 60 Hz: MN9 reaches 50 Hz and the proboscis
  extends. The GROOM meter reads GROOM, but the fly does not groom, because brain
  actions never interrupt a running action.
* Stimulating `region:GNG` at 60 Hz (2,000 neurons sampled): MN9 185 Hz, walk DNs
  ~25–60 Hz, DNg12 28 Hz. The brain slows to ~0.3× real time and lags up to 0.5 s
  while it lasts, then catches up.

Frames checked (Read as PNG): the palette + meters with a lesioned GF, a DNa02 L
stimulation during a loom (`LESIONED: DNp01`, meters BACK UP → STOP, TURN → LEFT);
region stimulation (`region:GNG`) with a lesioned `region:AL_L` (red outline) under
`--stress` (AROUSAL row); sugar + DNg12 stimulation with a lesioned MDN (GROOM and
EXTEND leaning). A windowed check opened a real OpenCV brain window
(`run_window`), injected a left-click on the MDN button and a right-click on the
lesioned GF button through `BrainWindow.on_mouse`, received
`{"op": "stim", "target": "MDN", ...}` and `{"op": "lesion", "target": "DNp01",
"on": false}` on the command queue, and closed itself after 3 s.

## Limitations

* **The body readout is ours.** It uses the same DN → CPG mapping and action
  thresholds as `--brain-steer` / `--brain-actions`, and the effect sizes follow from
  those gains. The walk gain is only +0.2, so BDN2 speeds the fly up by just ~10 %.
  The heading hold fights brain turns. There is no VNC, so a DN acts only through
  these mappings.
* **Stimulation is idealised.** Every targeted neuron fires at the set rate: there
  are no opsin kinetics, no light spread and no partial expression. Stimulating a
  sensory set is not the same as the natural sense (no receptor dynamics, no
  retinotopy).
* **Large regions are subsampled** (2,000 neurons) and can slow the brain below
  real time.
* **Lesions are all-or-none.** There is no partial knock-down and no developmental
  compensation. A lesioned DN's absence is felt only through the readouts listed
  above.
* **Window clicks travel through the app**, so they apply at the fly's current run
  time, ~0.1 s later in brain time. Lesions apply at the next brain chunk and are not
  scheduled in fly time.
* The palette's rate / duration apply to clicks only; keys and `--stim` use
  `stim_rate_hz` / `stim_duration_s` (120 Hz / 1 s by default).
