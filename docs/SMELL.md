# Smell without runaway, and odour tracking

In the whole-brain LIF model (Shiu et al. 2024; docs/BRAIN.md), olfactory input
ignites a global, self-sustaining state of about 475k spikes/s that outlasts the
input (docs/SENSORY_SCREEN.md). This page does three things:

1. It finds where that state starts for real ORN input, and why.
2. It adds optional model changes ("smell fixes") that stop it, each documented as
   a model change.
3. It tests whether an odour plume can steer the fly through the connectome.

**Short answer.**

* The runaway is an **excitatory local-neuron loop in the antennal lobe (AL)**. It
  is a sign and gain problem, not missing GABA gain.
* Correcting the AL local neurons' transmitter signs fixes it: 0 runaways in 72
  trials, activity back to 0 within 0.4 s, and odour-specific PN, KC and lateral
  horn (LH) patterns. Taste, looming and LC10a steering are unchanged.
* **Odour tracking does not work.** The model's turn DNs respond to vinegar with a
  left-turn bias that does not depend on which antenna smells it, so a fly steered
  by them circles and does not approach the source.

Where things are:

* Code:
  * `fly_simulator/brain/smell.py`: the fixes, odour → ORN rates, and the olfactory
    readout.
  * `LIFEngine.set_adaptation` in `fly_simulator/brain/engine.py`: spike-triggered
    threshold adaptation.
  * `fly_simulator/senses/plume.py`: the plume and the two antennae.
  * `scripts/smell_eval.py`: comparison of the fixes.
  * `scripts/smell_tracking.py`: open- and closed-loop steering.
* Tests: `tests/test_smell.py`.
* Data: [smell_eval_summary.csv](smell_eval_summary.csv), one row per fix.
* App: `--smell-fix NAME`, `--odor-plume [ODOR]`, `--plume-control ODOR` (see "App").

## 1. Diagnosis

### Where it starts

This test drives the 68 ORNs of glomerulus DM1 (both antennae) at **20 Hz**,
recorded in 2 ms bins. Timeline of the first spikes after stimulus onset:

| time after onset | first spikes (neuron class: types) |
|---|---|
| 5 ms | DM1 uniglomerular PNs (DM1_lPN) |
| 8 ms | ALLN lLN2T_c (predicted 5HT; excitatory in the model) |
| 10–14 ms | ALLNs lLN2T_b/c/e, lLN2X03, il3LN6, lLN2F_a, lLN2P_a/b; LHLNs (CB1058, LHAV4a1_b); LHCENT12b |
| 14–18 ms | lLN1_bc (14 neurons), lLN2X12, lLN2X03–05, lLN2P_a/c; first KCs |
| ~20 ms | whole brain at ~480k spikes/s. It stays there after the input stops |

Every glomerulus tested (DM1, DM2, VA2, DL5, DC1) ignites at 20–50 Hz within 20 ms.
The PNs of all other glomeruli are then driven to ~130 Hz, so odour identity is lost.

### Which loop carries it

Each condition ran 0.3 s of odour and 0.3 s off. Persistent activity was measured
0.1–0.3 s after offset, in whole-brain spikes/s:

| condition | DM1 50 Hz | VA2 100 Hz | own PNs (Hz) | other PNs (Hz) |
|---|---|---|---|---|
| unchanged model | 476k | 475k | 256 | 129 |
| silence the 166 model-excitatory ALLNs | **0** | 5.8k | 168 | **0** |
| excitatory ALLN output × 0.25 / × 0.1 | 62k / **3k** | 64k / **0** | 188 / 170 | 12 / 0 |
| GABA/Glu ALLN output × 2 / × 4 / × 8 | 328k / 257k / 145k | 322k / 232k / 106k | 176–242 | 40–95 |
| silence AL-MBDL1 (the KC-driven hub, docs/FEAR_LEARNING.md) | 477k | 475k | 260 | 130 |
| silence KCs | 285k | 283k | 258 | 131 |
| silence multiglomerular PNs | 408k | 406k | 258 | 129 |
| silence LH local neurons | 551k | 549k | 272 | 137 |
| **flip the 52 DA/5HT-predicted ALLNs to inhibitory** | **0** | 340 | 164 | 0 |

The same comparison in the input weights, as mean signed synapses onto one neuron
of each class (w_syn = 0.275 mV per synapse, 7 mV threshold gap):

| onto → | excitatory ALLN | inhibitory ALLN | uniglomerular PN |
|---|---|---|---|
| from excitatory ALLNs | **+1094** | +224 | +488 |
| from inhibitory ALLNs | −305 | −132 | −220 |
| from ORNs | +619 | +248 | +579 |

**Conclusion.** The self-sustaining loop is the **excitatory ALLN → ALLN network**.
An excitatory LN receives about 1100 excitatory LN synapses, so the loop gain is far
above 1 at the uniform synaptic weight. Scaling up inhibition does not stop it: at
8× the GABA/Glu output, the model still has a 100–145k spikes/s persistent state.
AL-MBDL1, the KCs, the multiglomerular PNs and the LH are downstream or
amplifiers, not the cause (for ORN input; AL-MBDL1 matters for KC input,
docs/FEAR_LEARNING.md). So this is a sign and gain problem.

### Why 166 of the 429 ALLNs are excitatory

The model's sign comes from the Shiu connectivity table: GABA and Glu are
inhibitory, everything else is excitatory. The FlyWire v783 annotations of these
166 neurons show:

* **40 are GABA or Glu in the v783 `top_nt`**, but excitatory in the Shiu table,
  whose per-connection signs differ from v783's per-neuron predictions. Of these,
  **16 are known to be inhibitory** from immunostaining (FlyWire `known_nt`):
  * lLN2P_b (12): GABA, Sizemore et al. 2023.
  * il3LN6 (2): GABA, Tanaka et al. 2012.
  * v2LN36 (2): Glu, Chou et al. 2022.
* **52 are predicted dopaminergic or serotonergic**, with low confidence (mean 0.29
  and 0.36). No AL local neuron is known to be monoaminergic: the AL's
  serotonergic neuron is CSD, which is not an ALLN. The lineage of these LNs (ALl1)
  produces GABAergic and cholinergic LNs.
* The remaining ~72 are cholinergic. Of these, lLN1_bc, lLN2X03, lLN2T_b and
  lLN2T_c are **known cholinergic excitatory LNs** (Shang et al. 2007). Note that
  lLN2T_b/c are *predicted* 5HT.

In the real AL, much of the lateral excitation from these eLNs is electrical, through
eLN–PN gap junctions (Yaksi & Wilson 2010). The model has no gap junctions. It also
has no GABA_B, no adaptation and no neuromodulation.

## 2. Fixes

All fixes are off by default (`--smell-fix none`). "Off" is bit-identical: the
engine code path and weights are unchanged. This is tested against the pre-change
engine on the full connectome, on DM1 input and on LC4 input.

| name | model change |
|---|---|
| `gaba` | GABA/Glu ALLN outputs × `gaba_gain` (4). Literature basis: presynaptic inhibition of ORNs, Olsen & Wilson 2008 |
| `adapt` | threshold adaptation of all 1152 AL neurons (PN/LN/ALIN/ALON): +`adapt_mv` (10) mV per spike, τ 0.2 s |
| `eln` | model-excitatory ALLN outputs × `eln_scale` (0.1); weakens the ignition hub |
| `mbdl1` | silence AL-MBDL1 |
| `global` | pooled AL inhibition (**last resort, not a mechanism**): every AL spike raises all AL thresholds by 0.05 mV, τ 50 ms |
| `sign` | ALLN signs from the v783 `top_nt`, and DA/5HT-predicted ALLNs made inhibitory. 101 LNs flipped |
| `sign_adapt` (**recommended**) | `sign`, but respecting `known_nt`: known ACh LNs stay excitatory and known GABA/Glu LNs become inhibitory (71 LNs flipped). Plus a milder AL adaptation of 4 mV per spike, τ 0.2 s |

### Before / after

Protocol (`scripts/smell_eval.py --trials 3`):

* **Odours**: 8 in total. Six single glomeruli (DM1, DM2, VA2, DL5, DC1, DA2), vinegar
  (DM1 + VA2 + ½ DM2 + ½ DM4) and a DL5 + DC1 + DA2 mixture.
* **Rates**: ORN rates of 20, 50 and 100 Hz, with 3 trials each. That makes 24
  trials per rate and 72 per fix; the unchanged model got 1 trial each.
* **Timing**: 0.5 s odour, then 0.5 s off.
* **Runaway**: more than 50k spikes/s 50–100 ms after offset.
* **Returned**: fewer than 1000 spikes/s 0.4–0.5 s after offset. For the fixes that
  work, the value is exactly 0.
* **Rates in the table**: means of the non-runaway trials during the odour.
* **KC**: the fraction of KCs with at least 2 Hz.
* **Specificity**: pattern correlation within the same odour (across trials) and
  between different odours, at 50 Hz, shown as within / between.

| fix | runaway | returned | own PN Hz (20/50/100) | other PNs @50 | KC frac @100 | LH Hz @100 | PN spec. | KC spec. | LH spec. |
|---|---|---|---|---|---|---|---|---|---|
| none | 23/24 | 1/24 | – | – | – | – | –/0.99 | –/0.98 | –/1.00 |
| gaba | 65/72 | 7/72 | 0 / 14 / 44 | 0 | 0 | 0.1 | 0.96/0.71 | 0.90/0.88 | 0.99/0.99 |
| adapt | 1/72 | 70/72 | 21 / 34 / 49 | 5.4 | 5.0 % | 3.5 | 0.91/0.71 | 0.92/0.83 | 0.97/0.85 |
| eln | **0/72** | 71/72 | 61 / 105 / 148 | 0.2 | 1.5 % | 2.7 | 0.99/0.10 | 0.87/0.09 | 0.97/0.18 |
| mbdl1 | 68/72 | 4/72 | – | – | – | – | – | – | – |
| global | 26/72 | 6/72 | 45 / 68 / 73 | 9.0 | 4.0 % | 10.5 | 0.98/0.65 | 0.92/0.75 | 0.98/0.91 |
| sign | **0/72** | **72/72** | 52 / 95 / 142 | 0.0 | 1.3 % | 2.4 | 1.00/0.10 | 0.91/0.09 | 1.00/0.15 |
| **sign_adapt** | **0/72** | **72/72** | 29 / 47 / 66 | 1.3 | 0.5 % | 1.8 | 0.99/0.25 | 0.84/0.28 | 0.98/0.51 |

Per odour, for `sign_adapt` (all rates pooled):

| odour | own PNs Hz | KC frac | LH Hz |
|---|---|---|---|
| DM1 | 81 | 0.5 % | 2.2 |
| vinegar | 41 | 0.9 % | 2.9 |
| DL5 + DC1 + DA2 | 21 | 0.2 % | 1.8 |
| DA2 | 6 | 0 | 0 |

(With `sign`: DM1 152 Hz / 1.6 %, vinegar 64 Hz / 2.3 %.)

**Other validated behaviours** (0.5 s drive, every fix; nothing persists afterwards):

| check | none | sign | sign_adapt | eln |
|---|---|---|---|---|
| sugar GRNs 200 Hz → MN9 (test: > 40 Hz) | 100 | 96 | 78 | 82 |
| left LC4 + LPLC2 150 Hz → GF | 140 | 136 | 136 | 133 |
| left LC10a 90 Hz → DNa02 L / R | 112 / 0 | 102 / 0 | 106 / 0 | 94 / 0 |

These pathways run outside the AL. The small differences come from the fixes
touching a few AL-projecting neurons and from Poisson noise.

**Choice.** `sign_adapt` is the recommended fix, and `--odor-plume` uses it by
default:

* It never ran away (0 of 72 trials).
* Activity returns exactly to rest.
* PN patterns are odour-specific.
* It is the most literature-consistent: it keeps the known cholinergic eLNs
  excitatory, and the adaptation stands in for the real PN/LN spike-frequency
  adaptation and GABA_B inhibition.

With the known eLNs kept excitatory, neither part is enough alone. In the
diagnosis runs, the known-NT sign correction alone left a ~66k spikes/s persistent
state, and adaptation alone left 54k spikes/s at 3 mV and 10–25k at 5 mV. Together
(sign correction plus 4 mV) they leave 0–2k.

`sign` gives the cleanest specificity and stronger responses. However, it makes 26
known-cholinergic LNs inhibitory. `eln` works too (0/72), but it just turns the
excitatory LNs down to 10 %.

## 3. Odour tracking: negative result

**Plume** (`fly_simulator/senses/plume.py`). A source with concentration
`c = 2 · exp(-r / 12 mm)` meanders sideways (±1.5 mm, 5 s period). The two antennae
sit 1 mm ahead of the thorax and 0.4 mm apart. Each antenna's concentration sets
the rate of **that side's** ORNs (FlyWire `side` = antennal nerve), using
`rate = 100 · c^1.5 / (c^1.5 + 0.5^1.5)` Hz, scaled by the glomerulus weight.

As in flies, every ORN projects to both ALs. Any ipsi/contra asymmetry therefore
comes from the connectome's own ORN synapses. Gaudry et al. 2013 reported larger
ipsilateral release; no extra asymmetry is added here. `contrast_gain` can
exaggerate the left/right difference as a modelling aid. It is off (1) by default.

**Attractive odour.** Vinegar, via DM1 and VA2 (Semmelhack & Wang 2009), with weaker
DM2 and DM4. Controls: DL5 alone, geosmin (DA2, aversive; Stensmyr et al. 2012), or
no odour.

**Open loop** (`scripts/smell_tracking.py`, 20 trials × 0.3 s). Turn-DN rates are
DNa01 + DNa02 per side:

| fix, odour | left antenna only | right antenna only | both |
|---|---|---|---|
| sign_adapt, vinegar | turn L 16.9 / R 3.3 | L 13.8 / R 4.3 | L 17.3 / R 4.4 |
| sign_adapt, DL5 | L 7.4 / R 2.1 | L 11.5 / R 3.2 | L 13.5 / R 3.2 |
| sign, vinegar | L 28.0 / R 0.8 | L 18.5 / R 3.8 | L 32.0 / R 1.2 |
| sign, geosmin (DA2) | 0 / 0 | 0 / 0 | 0 / 0 |

Odour on either antenna drives the **left** DNa02 (DNa02_L) more than the right,
whatever the side. Left-antenna vinegar gives a somewhat larger left bias than
right-antenna vinegar: with `sign`, L−R is +27 against +15 Hz, which is a real but
secondary side effect. A 20 % concentration difference changes nothing measurable.
Geosmin reaches no turn DN at all.

**Closed loop.** A kinematic point fly moves at 3 mm/s. Heading changes at
3 rad/s × (tanh(turn_L/30) − tanh(turn_R/30)), which is the app's `--brain-steer`
mapping, plus heading noise. The source is 20 mm ahead, the fly starts at 4
headings, and each run lasts 8 s.

| odour | final distance to source (mm) |
|---|---|
| vinegar | 18.7 (16.6–20.6) |
| DL5 | 14.7 |
| no odour | 13.7 |

With vinegar, the constant left-turn bias makes the fly circle, so it ends *farther*
from the source than with no odour. `contrast_gain` 20 did not help: 19.4 mm.

A 3 s app run (`--odor-plume --brain-steer`) shows the same thing: turn_L around
20–25 Hz and turn_R 0–10 Hz while the fly walks past.

**Conclusion.** In this connectome model, the olfactory → DN route (AL → LH / MB →
… → DNa02) carries an odour-driven, left-lateralised turn signal. It does not carry
a signal about which antenna smells more. So the model gives no tracking.

Real flies steer toward the antenna with more odour: they compare the two antennae
(Duistermars et al. 2009; Gaudry et al. 2013), and they use wind and plume
dynamics, which the model lacks. The deliverable here is therefore **smell without
runaway plus a brain readout**, not tracking.

## App

* `--smell-fix NAME` turns on `--brain` with that fix
  (`BrainLinkConfig.smell_fix` / `smell_fix_config` → `BrainConfig.smell_fix`).
  It can also be changed at run time with `BrainProcess.set_smell_fix(...)`.
* `BrainState.smell` carries:
  * the fix name and the number of flipped and adapting neurons;
  * the rates of ORNs, left and right uniglomerular PNs, ALLNs, KCs and the LH;
  * the fraction of active KCs;
  * the total spikes/s and a runaway flag;
  * the active odour labels.
* This shows up as a HUD `SMELL` line, a brain-window header line, and `smell` in
  `summary.json`.
* `--odor-plume [ODOR]` sets a plume source at (25, 6) mm with an amber marker. It
  implies `--brain` and `--smell-fix sign_adapt`. `--plume-control ODOR` adds a grey
  second source at (25, −6) mm.
* Stimulus kind `odor` has details `{"odor", "left", "right", "r_max_hz"}`.
* A HUD `PLUME` line shows the L/R concentration and the PN/KC/LH rates. `plume` in
  `summary.json` gives the minimum distance to each source.
* With `--brain-steer`, the turn DNs steer. See section 3 for what that does.

## Limitations

* The fixes are **model changes**, not fitted biology:
  * The sign rules use predicted transmitters (v783 `top_nt`) plus a few
    immunostained types. The DA/5HT → GABA rule is an argument from lineage, not a
    measurement.
  * The adaptation parameters (4 or 10 mV, 0.2 s) were chosen, not fitted.
  * With `sign_adapt`, ~0.5 % of KCs respond at 100 Hz. Real odours activate about
    5–10 % (Honegger et al. 2011), but they also recruit many more glomeruli than
    these 1–4.
* The model still has no gap junctions, no GABA_B, no ORN adaptation, no spontaneous
  ORN firing (ORNs are silent without odour) and no neuromodulation.
* `smell_eval.py` ran 72 trials per fix: 8 odours, 3 rates and 3 trials. The
  per-fix runaway statistics are exact for these conditions. Other glomeruli, much
  larger mixtures or long exposures were not tested.
* The fear-learning add-on (docs/FEAR_LEARNING.md) still uses KC codes and silences
  AL-MBDL1. Combining it with ORN odours under a smell fix was not evaluated.
* Tracking fails for a connectome reason: the turn DNs' odour response has no usable
  left/right sign. The plume itself is a simple radial field, with no wind, no
  filaments and no intermittency.
