# Sensory screen of the whole-brain model: which inputs reach which descending neurons

This screen asks which sensory inputs in the FlyWire brain model (Shiu et al. 2024 LIF,
`docs/BRAIN.md`) drive which behaviour descending neurons (DNs). It was run to answer
two questions: does any touch-like input a whip could plausibly excite reach the
walking, turning or escape DNs? And which inputs make the fly *speed up*, for a
"pain" response to whip hits?

* Code: `fly_simulator/brain/screen.py` (group enumeration, stimulation, readout, hop
  analysis) and `scripts/screen_sensory.py` (runner and `--summarize` markdown tables).
  Test: `tests/test_screen.py` runs on the synthetic network.
* Data: `docs/sensory_screen.csv` covers the full screen: 516 groups × {100, 200} Hz,
  2 trials × 0.5 s, 198 KB. `docs/sensory_screen_touch_oa.csv` covers the touch,
  ascending and thermo/hygro groups at 200 Hz with an octopamine readout, 69 KB.
* Reproduce: `.venv/bin/python scripts/screen_sensory.py` takes 17 min in one
  process (`--workers 2`: 8 min). Then run
  `scripts/screen_sensory.py --summarize docs/sensory_screen.csv`.

## Method

* **Groups** (516) are built from FlyWire v783 annotations and split by the fly's
  side unless noted:
  * Johnston's organ (JO): fine types (JO-A, B, CA, CL, CM, DA, DP, ED, EV, FD, FV),
    letter groups, the functional sub-classes wind_gravity, auditory and grooming,
    and all JO.
  * Bristles: head, eye, and every `BM_*` type.
  * Taste-peg mechanosensory neurons (TPMN) and pharyngeal mechanosensory neurons
    (aPhM).
  * Gustatory: by sub-class and by type.
  * Olfactory: 53 ORN glomerular types, sides pooled, plus the pheromone ORNs.
  * Thermo/hygro: 8 types, sides pooled.
  * Unknown sensory neurons.
  * Photoreceptors: R1-6, R7, R8, DRA, and ocellar photoreceptors.
  * Sensory-ascending (VNC afferents): 8 sub-classes, 40 types and the app's
    `body_mech` set.
  * Ascending neurons (AN): 39 sub-classes, plus types with at least 8 neurons.
  * Visual projection neurons (VPN): 71 types with at least 20 neurons (LC, LPLC,
    LLPC, LPC, MeTu, …).
  * Combinations for the whip question: `all_touch` (all mechanosensory plus
    sensory-ascending), `body+headbristle+JOwind`, and `all_ascending`.
* **Stimulation**: the engine runs in-process (`LIFEngine`, reset to rest before
  each run). Each group gets Poisson input at 100 and 200 Hz for 0.5 s, 2 trials.
* **Readout**:
  * Mean rate of each DN group in `mapping.DESCENDING_TYPES`: walk = BDN2/DNg100,
    oDN1/DNg97 and P9/DNp09; turn = DNa01/02; backward = MDN; escape = giant fiber
    DNp01; groom = DNg12.
  * MN9, and each key DN type per side.
  * The 20 most active DN cell types.
  * The number of neurons recruited.
  * The named octopaminergic cell types (`OA-*`, 43 neurons), touch CSV only.
  * The minimum number of excitatory synaptic hops to each DN group (≥1 synapse,
    and ≥5 synapses).
* **Runaway flag** (`runaway`, `persist_sps`): the model can switch into a global
  self-sustaining state of about 475k spikes/s, which continues after the input
  stops and saturates hundreds of DNs. A run is flagged when more than 50k spikes/s
  persist 50–100 ms after stimulus offset. Every ORN, thermo/hygro and pheromone
  group ignites this state, at 100 Hz already; it starts in the antennal lobe and
  mushroom body. Some gustatory groups and a few AN groups ignite it
  stochastically. **Flagged runs are artefacts of the model** (it has no adaptation
  and no spontaneous inhibition). They are excluded from all tables below, and
  `docs/BRAIN.md`'s "quiet brain" assumption does not hold for olfactory input.

## Results at 200 Hz input (non-runaway runs)

Counts of groups with a DN group at ≥ 5 Hz:

| category | groups | walk | turn | backward | escape | groom | MN9 |
|---|---|---|---|---|---|---|---|
| visual projection (VPN) | 142 | 59 | 71 | 13 | 17 | 0 | 3 |
| ascending (AN) | 95 | 16 | 28 | 1 | 0 | 0 | 5 |
| sensory_ascending | 69 | 3 | 4 | 0 | 0 | 0 | 0 |
| mech: JO | 40 | 2 | 2 | 0 | 4 | 0 | 0 |
| mech: bristles | 34 | 0 | 3 | 0 | 0 | 3 | 0 |
| mech: TPMN / aPhM / other | 8 | 0 | 0 | 0 | 0 | 0 | 3 |
| gustatory | 28 | 2 | 3 | 0 | 0 | 0 | 10 |
| photoreceptors | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| olfactory, thermo/hygro | 63 | all runaway | | | | | |

**Vision dominates the behaviour DNs.** This is expected: the DNs in the readout were
characterised with visual stimuli, and the VPNs are one synapse from many DNs.

| behaviour | strongest inputs (group: DN rate, Hz) |
|---|---|
| walk (BDN2/oDN1/P9) | LC31a (P9 148, BDN2 ~120), LC9 (P9 110–160), LC17, LC15, LC18, LC11, LPLC1; AN_AVLP_PVLP and AN_AVLP (BDN2 80–97) |
| turn (DNa01/02) | LC10a/c/d (DNa02 ~140 ipsilateral), LLPC1/2 (DNa02 ~150 ipsilateral), AN_IPS_GNG_7 (DNa02 ~125 ipsilateral), LC9, LC17, LC6 |
| backward (MDN) | LC9:right (36), LPC1 (~30 both MDNs), LPLC4 (20), LC6 (18), LC16 (18); no touch input exceeds 8 Hz |
| escape (GF DNp01) | LC6 (121–144), LPLC2 (130–141), LC17 (~104), LC4 (86–90), LC15 (~80), LC31a (46); **JO_auditory:right 40, JO_all:right 32, JO-A:right 28, JO-B:right 24** |
| groom (DNg12) | only large bristle pools: all_touch:both 27, all_touch:left 16, BM_all:left 12, BM_all:right 8, BM_InOm (eye bristle) and head bristle 5 |
| feeding (MN9) | sugar GRNs (LB3) 88–95 Hz on the fly's left, an unassigned mechanosensory group 82–110, TPMN1 and taste-peg GRNs ~40, AN_GNG_162 35–46 |

## Whip-plausible touch pathways (mechanosensory, bristle, JO, sensory-ascending, AN)

Top rows at 200 Hz. Full list: `--summarize`, section "whip-plausible".

| group | n | walk L/R | turn L/R | backward | escape | groom | notes |
|---|---|---|---|---|---|---|---|
| AN_IPS_GNG(_7):left | 18 (6) | 17/22 | **70**/14 | 0 | 0 | 0 | DNa02_L 125 (ipsilateral); OA-VUMa1 ~100 Hz |
| AN_IPS_GNG(_7):right | 19 (6) | 25/20 | 22/**68** | 0 | 0 | 0 | mirror image |
| AN_AVLP:left | 92 | **55/42** | 2/**64** | 0 | 0 | 0 | BDN2 97/94, contralateral turn |
| AN_AVLP:right | 97 | 31/44 | **56**/2 | 2–4 | 0 | 0 | mirror image (weaker walk) |
| AN_AVLP_PVLP:left / right | 10 / 10 | **61/38**, 42/57 | 12/55, 49/18 | 0–3 | 0 | 0 | BDN2 80–90 bilaterally |
| JO_auditory:right, JO_all:right, JO-A:right | 175, 510, 51 | 0 | ≤6 | 0 | **28–40** | 0 | GF escape; the left JO does *not* reach the GF |
| JO_all:left | 582 | 19/0 | 1/20 | 0 | 0 | 0 | oDN1_L 52 |
| BM_Ant:right (antennal bristles) | 21 | 0 | 0/**26** | 0 | 0 | 0 | DNa01/02_R; the left side reaches only 3 |
| head_bristle:left / right | 150 / 155 | 0 | 9/7, 0/4 | 0 | 0 | 5, 1 | |
| BM_all (all head + eye bristles), left | 705 | 0 | 12/1 | 1 | 0 | 12 | |
| all_touch:both | 3237 | 0 | 0 | 4 | 0 | **27** | only input with DNg12 above the app's 20 Hz groom threshold |
| SA_VTV_5 / SA_VTV_pro_meso_meta (leg afferents) | 7 / 41 | 9–16 | 12–22 / 2–6 | 0 | 0 | 0 | tarsal (mostly gustatory) afferents |
| **body_mech** (app whip set), left / right | 193 / 302 | 0 | 0 / 3–4 | 0 | 0 | 0 | no behaviour DN |
| SA_DMT_* / SA_DLV / SA_MDA (notum/wing/haltere) | – | 0 | ≤6 | 0 | 0 | 0 | none reach behaviour DNs |

Findings:

1. **Body mechanosensory afferents (the app's `body_mech`) reach no behaviour DN.**
   This holds at 100 and 200 Hz, both sides, per type and in combination. It
   confirms the earlier finding. The notum, wing and haltere sensory-ascending
   classes all top out at ≤ 6 Hz on a turn DN.
2. **JO is not silent, but the whip's current JO input is too weak.** The app drives
   only JO wind_gravity at 0.5× rate, which gives about 0 Hz. The *right* auditory
   JO (JO-A/B) reaches the giant fiber at 24–40 Hz; the left does not, a
   connectome asymmetry. The whole left JO drives oDN1_L (52 Hz) and turn_R (20 Hz).
   None of this reaches the app's jump threshold (GF > 60 Hz).
3. **Head and antennal bristles reach grooming and some turning, weakly.** DNg12 needs
   essentially all head and eye bristles plus body afferents (`all_touch:both`,
   27 Hz) to exceed 20 Hz. Right antennal bristles (BM_Ant:right) drive a clean
   ipsilateral turn (26 Hz), but the left side does not.
4. **Ascending neurons are the only touch-adjacent class that strongly drives walk and
   turn DNs.** FlyWire does not annotate their modality. They carry VNC information
   (leg/body proprioception and mechanosensation, locomotor state; Chen et al. 2023,
   Nat Neurosci), and any somatosensory "pain" signal from the body would have to
   reach the brain through them. `AN_AVLP_PVLP` and `AN_AVLP` drive BDN2 and oDN1
   bilaterally at 80–97 Hz; `AN_IPS_GNG_7` drives ipsilateral DNa02 and octopamine
   neurons.
5. **Lateralisation.** Turn DNs are almost always driven *ipsilaterally* by
   visual-steering inputs (LC10, LLPC1/2: left input gives DNa02_L, a left turn).
   AN_IPS_GNG_7 and antennal bristles also turn ipsilaterally, toward the input.
   **AN_AVLP and AN_AVLP_PVLP turn contralaterally**: left input gives DNa01/02_R, a
   turn away from the stimulated side, together with bilateral BDN2. LC4, LC6,
   LPLC2 and LC17 also give contralateral turns alongside escape.
6. **Touch recruits inhibition onto the walk DNs.** When body_mech and head bristles
   are added to the AN sets (0.2 s pulses), BDN2 drops from about 100 Hz to about
   0 Hz and turning rises. Adding more touch input to a speed-up set is
   counter-productive in this model.

## Nociception, "pain" and speed-up

* **FlyWire v783 has no identified nociceptors or nociceptive ascending neurons.** No
  annotation column mentions noci-, ppk, multidendritic/md, noxious or pain, including
  the synonyms and matching notes. Class IV md / ppk nociceptors are larval or body-wall
  neurons that project to the VNC, which the model does not contain. The only
  literature-annotated ANs are courtship-related (vPR13, dMS9, vPr-g). The
  "heating" thermosensory neurons (TRN_VP2, arista hot cells) sense innocuous warmth,
  and they drive the model into runaway.
* **Octopamine (arousal / locomotor speed).** Among the touch-adjacent inputs, only
  a few recruit the named OA neurons (`OA_mean`, `top_oa` in
  `docs/sensory_screen_touch_oa.csv`):
  * AN_IPS_GNG_7 (OA-VUMa1 ~100–118 Hz; mean over all OA neurons 16–19 Hz)
  * AN_multi:left (OA-VUMa8 86 Hz)
  * all_touch:both (OA-AL2 types ~60 Hz)
  * BM_all:left and head_bristle:left (OA-AL2 ~25–34 Hz)
  * body_mech: ≤ 1 Hz

  The OA neurons act only as fast excitatory synapses in this model; there is no
  neuromodulation.
* **Inputs that most increase walk-DN firing** (mean of walk_L/R at 200 Hz, excluding
  runaway runs):
  * Visual: LC31a (~90), LC9 (46–66), LC17 (~50), LC15, LC11, LC18.
  * Ascending: **AN_AVLP_PVLP (50)**, **AN_AVLP (38–48)**, AN_IPS_GNG / _7 (20–22).
  * No bristle, JO or sensory-ascending group exceeds 10 Hz on either walk group,
    except JO_all:left at 19 Hz on the left walk group only.

## Recommended stimulus sets for the app

Added **additively** to `mapping.named_sets` and usable now via
`manual {"set": ..., "side": ..., "rate_hz": ...}`:

| set | cell types (count) | effect (0.2 s pulse, 150 Hz, both sides) |
|---|---|---|
| `an_walk` | `cell_sub_class == AN_AVLP_PVLP` (types AN_AVLP_PVLP_1–10; 10 L + 10 R) | walk 52/52 (BDN2 ~90, oDN1 ~55, P9 ~12); turn 28/42; no MDN or GF; no runaway in 30 pulses |
| `an_avlp` | `cell_sub_class == AN_AVLP` (92 L + 97 R) | walk 43/42 (BDN2 ~80); one side gives a strong *contralateral* turn (left: turn_R 55) |
| `an_arousal` | `cell_type == AN_IPS_GNG_7` (6 L + 6 R) | OA mean 12.6 Hz (OA-VUMa1 ~100); ipsilateral DNa02 turn; BDN2 ~65; walk group only 24 because it averages in oDN1 and P9, which stay silent |

Left-side `an_walk` ids (for reference): 720575940606663881, 720575940607909899,
720575940614298986, 720575940614633122, 720575940620192677, 720575940620887536,
720575940621493068, 720575940622160460, 720575940623324395, 720575940643683744.

**Suggested whip "pain" mapping.** Keep the current body_mech plus JO input, for
honest sensory activity. *Add* `an_walk` on both sides at `100 + 50*intensity` Hz
(100–150 Hz) for the hit duration plus about 0.2 s. If the hit side should steer the
fly away, add `an_avlp` on the hit side at about 100 Hz (turn away plus
walk). Optionally add `an_arousal` at 100 Hz for the OA-VUM "arousal" signature.
Measured together, `an_walk` + `an_arousal` on both sides at 150 Hz gave walk 71/68 Hz
(BDN2 ~120), OA 13 Hz, and no MDN, GF or runaway. With the default `DriveGains`, that
raises stepping amplitude by about +0.2, the maximum of `walk_gain`. The walk term saturates at about 30 Hz, and the
two turn groups roughly cancel when both sides are driven.

**Caveats.**
* These ANs are *not* known nociceptors. Their modality is unannotated, and their
  effect on BDN2 is a model prediction from connectivity. The Shiu model was
  validated on taste and grooming pathways, not on these.
* Driving them is a behavioural shortcut for "a whip hit makes the fly run". It is
  not a simulation of pain.
* The real touch afferents in the model do the opposite: they inhibit BDN2 (finding
  6). Driving the ANs *together with* body_mech and head bristles cancels much of
  the speed-up. For a speed-up, keep the body_mech and head bristle rates modest,
  or drive the ANs a little after the touch pulse.
* Rates are per-neuron Poisson with the model's strong input weight (68.75 mV per
  event). 100–150 Hz is within Shiu et al.'s 10–200 Hz range.
* Olfactory and thermo input must not be added for "pain": it ignites the runaway
  state.

## Honest bottom line

* No input that a whip hit directly stimulates in this brain-only model (body
  mechanosensory afferents, head/eye/antennal bristles, JO) reaches the walk DNs or
  MDN at useful rates.
* The right auditory JO reaches the giant fiber at 24–40 Hz, which is below the
  jump threshold. Only very large bristle pools reach DNg12 above 20 Hz.
* A behavioural speed-up can be obtained by driving specific ascending neurons.
  These are plausible carriers of body state, but the connectome and annotations
  cannot identify them as nociceptive.
