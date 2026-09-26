# Fear learning: the mushroom body learns to fear an odour

With `--learning` you can whip the fly while it is inside an odour zone. After that,
the brain responds differently to that odour. **Dopamine-gated long-term depression**
weakens the Kenyon cell (KC) → mushroom-body output neuron (MBON) synapses of the
KCs that the odour activates. This happens only in the compartments whose
punishment dopaminergic neurons (PPL1 DANs) fired at the same time.

Everything is off by default. With it off, the engine is bit-identical to the
original model (tested).

Code:

* `fly_simulator/brain/plasticity.py`: the rule, the connectome-derived
  compartments and the odour KC codes.
* `fly_simulator/brain/process.py`: `BrainConfig.plasticity`, `_Model.configure_plasticity`
  and `BrainProcess.set_plasticity`.
* `fly_simulator/brain/schema.py`: `BrainState.learning`.
* `fly_simulator/senses/odor.py`: odour zones, sensing and punishment.
* App wiring (flags `--odor-zones` and `--learning`, keys 8 / =, HUD line):
  `fly_simulator/app.py`, `fly_simulator/config.py`, `fly_simulator/brain_link.py`.
* Brain window header: `fly_simulator/brain_viz/window.py`.
* Scripts: `scripts/demo_fear_learning.py` (offline, in-process engine) and
  `scripts/demo_fear_learning_app.py` (full app, headless).
* Tests: `tests/test_fear_learning.py`.

## Biology

Aversive olfactory conditioning in Drosophila works like this. An odour activates
a sparse set of KCs, about 5–10 % (Turner et al. 2008; Honegger et al. 2011).
Punishment activates PPL1 DANs (Claridge-Chang et al. 2009; Aso et al. 2010, 2012).
Dopamine in a mushroom-body compartment then depresses the synapses from the
co-active KCs onto that compartment's MBON. The classic example is
KC → MBON-γ1pedc>α/β after pairing with PPL1-γ1pedc (Hige et al. 2015; Cohn et al.
2015; Aso & Rubin 2016; timing: Handler et al. 2019). As a result, the MBON
responds less to the trained odour, and the balance of MBON output shifts towards
avoidance (Aso et al. 2014; Owald et al. 2015).

## What is connectome and what is modelled

| stage | status |
|---|---|
| odour → KCs | **modelled**: a fixed KC set per odour. The set is chosen from the connectome's PN → KC synapses, and the worker drives it with Poisson input (see "Odour representation") |
| KC → MBON, KC → APL → KC, KC → DAN, MBON → … → DNs | connectome (Shiu et al. LIF, FlyWire v783) |
| which DAN goes with which MBON (compartments) | derived from the connectome (below) |
| **KC → MBON efficacy** | **modelled**: coincidence of KC activity and DAN spikes → depression (parameters are free, not fitted) |
| whip → PPL1 | **stand-in**. The model's whip afferents do not reach PPL1 (measured below), so a hit inside a zone drives the PPL1 DANs directly |
| AL-MBDL1 (2 neurons) | **silenced** while the add-on is on (runaway guard, below) |

### Compartments (FlyWire v783, derived in `compartments()`)

A punishment DAN type is a PPL1 type with at least 100 synapses onto KCs. Its
MBONs are the MBON types it contacts with at least 50 synapses. An MBON type that
is contacted by several DAN types belongs to the DAN type with the most synapses
onto it.

| DAN | synapses onto KCs | compartment MBONs (DAN → MBON synapses) | compartment |
|---|---|---|---|
| PPL101 (2) | 3113 | MBON11 (894) | γ1pedc (MBON-γ1pedc>α/β, GABA) |
| PPL103 (2) | 7963 | MBON12 (449), MBON32 (264), MBON35 (228), MBON31 (221), MBON33 (87), MBON15-like (68) | γ2α'1 |
| PPL104 (2) | 456 | MBON16 (141), MBON28 (73) | α'3 |
| PPL105 (2) | 2405 | MBON13 (286), MBON18 (133), MBON23 (55) | α'2α2 |
| PPL106 (2) | 2322 | MBON14 (465) | α3 |

PPL102 (51 synapses onto MBON11) loses MBON11 to PPL101. PPL107 and PPL108 make
almost no synapses onto KCs. The plastic synapses are all 22,142 KC → MBON CSR
entries onto these MBONs. The HUD / readout "main" compartment is PPL101 (γ1pedc).

### Rule (per engine run, ≤ 20 ms of brain time)

```
KC eligibility  e_j <- e_j exp(-dt/tau_elig) + spikes_j            tau_elig = 1 s
KC activity     a_j  = min(e_j / e_sat, 1)                          e_sat = 5 spikes
compartment DA  D_c  = mean spike rate of the compartment's DANs in the run
LTD             x   <- x - lr a_j g(D_c) (x - x_min) dt             lr = 0.5 /s, x_min = 0
                g(D) = clip((D - 100 Hz) / (200 Hz - 100 Hz), 0, 1)
engine weight   w = w0 * x   (written into LIFEngine.weights; the kernel is untouched)
```

Forgetting (`tau_forget_s`) is off by default. While the whole brain fires more than
100k spikes/s (the runaway state), learning pauses.

**Why the dopamine threshold is 100 Hz.** In this model, KCs excite DANs strongly
(125k KC → DAN synapses). The odour alone drives PPL106 to about 85 Hz and PPL101
to about 47 Hz (odour A). With a low threshold, every odour would therefore
punish itself. So the threshold sits above the odour-evoked DAN rates, and the
punishment stand-in drives the DANs at 200 Hz. The absolute rates in this model are
inflated compared with in-vivo rates (MBONs fire at around 100 Hz here), so treat
the threshold as relative to the model.

## Odour representation (and the runaway problem)

Stimulating ORNs, thermo- or pheromone receptors ignites the model's
self-sustaining runaway state (docs/SENSORY_SCREEN.md). Measured here:

* **Uniglomerular PNs: not usable.** A random 10 % of them (27 PNs) at only 10 Hz
  ran away every time (≈ 480k spikes/s, 68 % of KCs active, no odour specificity).
* **Direct KC drive with the APL intact** works. A random 5 % at 10–20 Hz gave
  almost no MBON output. The problem is that the 2 APL neurons fire at 100–200 Hz
  and inhibit the MBONs (APL → MBON 3.8k synapses). At 10 % and 40 Hz the MBONs
  responded (MBON11 64 Hz, MBON14 68 Hz). Silencing the APL made KC drive run away.
* **But with KC drive the runaway was stochastic.** A 10 % KC code at 30–40 Hz ran
  away in 1–3 of 4 one-second trials, and a 5 % code at 25–30 Hz in 5 of 8
  two-second trials. The spike time course shows how it starts. The MB is active
  for about 0.25 s, then the antennal lobe ignites (ALLN, ALPN), and after that the
  AL / LH / KC loop sustains itself. The first AL spikes are driven mostly by
  **AL-MBDL1** (2 neurons, class ALIN), which receives KC input.
* **Silencing AL-MBDL1 removes the runaway.** 0 of 4 runs at 10 %/40 Hz ran away,
  against 3 of 4 before. 0 of 48 two-second runs ran away across both odours,
  5 % / 10 % codes and 30 / 40 Hz. In non-runaway trials it does not change the
  MBON responses (MBON14 about 106 Hz in both cases). Silencing all ALLNs had the
  same effect, and silencing the ALPNs alone did not. Because the odour input
  bypasses the AL anyway, the add-on silences AL-MBDL1 (`silence`, default
  `("AL-MBDL1",)`) whenever it is configured, even with learning off.

**Choice:** an odour is the top 10 % of KCs (518 of 5177), ranked by synapses from
the uniglomerular PNs of a glomerulus set. Those KCs are driven at 30 Hz. The
glomerulus sets are labels only; we do not claim that a real odorant activates
exactly these:

* odour A: DM1, DM2, DM3, DM4, VA2, DC2 (370 KCab, 135 KCγ-m, …)
* odour B: DL1, DL5, VA5, VM3, DA3, VC3 (335 KCγ-m, 179 KCab, …)

The two sets overlap in 39 KCs. No other KCs are recruited: 0 % of the non-odour
KCs exceed 2 Hz.

## Does MBON output reach descending neurons?

**Yes, for these odours, and only through the punished compartments.** Odour A
(10 % at 30 Hz) drives walk DNs (L 11 / R 17 Hz) and turn DNs (DNa01/02 L 25 /
R 6 Hz), which is a leftward bias.

* Silencing all 13 punished-compartment MBON types removes that DN activity
  completely (walk and turn → 0).
* Silencing all MBONs has the same effect.
* Silencing MBON11 alone changes little.
* Of the single types, only **MBON14 (α3, ACh)** silences the DN response
  completely. MBON18 (α2sc) reduces it (turn L 8 → 2.5 Hz). The others do not.
* The direct MBON → behaviour-DN synapses are few: MBON27 → MDN 44, and
  MBON27 → DNa03 230. MBON27 is not plastic here.

So learning can change behaviour in this model, but the change is **the loss of
the odour-evoked walk / left-turn drive**. It is not an active turn away.
Depressing the approach-type MBONs does not disinhibit any avoidance pathway to
the behaviour DNs this model reads out.

## Results

### Offline conditioning (in-process engine, `scripts/demo_fear_learning.py --seeds 0 1 2`)

The protocol is as follows. Each test is 1 s with plasticity frozen. Training is
4 trials of [odour A + PPL1 DANs at 200 Hz for 1 s, then 5 s rest, then odour B
alone for 1 s, then 5 s rest]. Values are mean ± SD over 3 engine seeds (the
Poisson streams). The KC codes are the same in every seed. 0 runaway chunks; the
persistent activity after the stimulus was ≤ 15k spikes/s.

| readout | odour A (paired) pre → post | odour B (unpaired) pre → post |
|---|---|---|
| KC>MBON efficacy γ1pedc (PPL101) | 1 → **0.25 ± 0.01** | 1 → 0.93 |
| efficacy α3 (PPL106) | 1 → **0.20** | 1 → 0.85 |
| efficacy α'2α2 / γ2α'1 / α'3 | 1 → 0.28 / 0.29 / 0.27 | 1 → 0.88 / 0.96 / 0.98 |
| MBON11 (γ1pedc>α/β) | 76.7 ± 1.0 → **23.2 ± 1.4 Hz** | 66.2 ± 1.7 → 52.7 ± 3.0 |
| MBON14 (α3) | 93.0 ± 0.7 → **30.0 ± 0.2** | 46.2 ± 0.4 → 39.6 ± 0.7 |
| MBON18 (α2sc) | 33.8 ± 0.8 → **0** | 6.8 → 6.0 |
| MBON31 (γ2α'1 group) | 18.2 → 0 | 11.5 → 8.8 |
| non-plastic MBON02 / MBON07 | 74.5 → 80.2 / 51.1 → 55.8 | 24.7 → 26.0 / 2.1 → 3.5 |
| walk DNs L / R | 11.0 / 16.7 → **0 / 0** | 8.8 / 12.9 → 5.3 / 7.8 |
| turn DNs L / R | 24.5 / 5.5 → **0 / 0** | 18.3 / 3.8 → 10.8 / 3.0 |
| MDN, GF, DNg12 | 0 → 0 | 0 → 0 |

B's residual depression (0.85–0.93) comes mostly from the 39 KCs that the two
odours share (7.5 %), i.e. generalisation. Without the 5 s gaps between trials,
B's KCs were still eligible at the next pairing and got depressed almost as much
as A's (0.31). The eligibility window matters, as in the fly.

**Whip → PPL1** (`--whip-check`, 0.5 s drives): body_mech at 200 Hz, JO
wind/gravity at 100 Hz, the nociceptive relay sets an_arousal (120 Hz) and
an_walk (150 Hz), and head bristles at 200 Hz all gave 0 Hz in every PPL1 type.
All of them combined at 150 Hz gave PPL101 2 Hz. **The modelled whip does not
reach the punishment DANs**, so the app uses a DAN stand-in.

### Behaviour in the full app (`scripts/demo_fear_learning_app.py`)

The run is headless, with the real brain, `--brain-steer` behaviour and flat
ground. Each test puts an odour zone (40 mm radius; the fly walks at about 14 mm/s)
around the fly for 2 s, then removes it for 3 s. Training is 4 × [zone A + one
real whip crack (level 1) → 1 s punishment-DAN stand-in; zone B without a crack],
with 3 s rests. The turn drive is the L − R amplitude command the brain sends to
the walking controller (negative = left). Yaw is the measured heading change.

| run | test | turn drive (L−R) | DNa01/02 L / R (Hz) | yaw (deg/s) | KC>MBON γ1pedc eff. A / B |
|---|---|---|---|---|---|
| `--learning` | pre A | **−0.41** | 28.4 / 6.2 | +8.0 | 0.99 / 1.00 |
| | pre B | −0.28 | 20.3 / 6.0 | −0.3 | |
| | post A | **0.00** | **0.0 / 0.0** | −2.1 | **0.22** / 0.91 |
| | post B | −0.16 | 9.5 / 2.7 | +7.1 | |
| control (`--no-learning`, same cracks) | pre A | −0.31 | 24.5 / 8.2 | +6.8 | 1 / 1 |
| | pre B | −0.26 | 20.3 / 7.0 | −2.0 | |
| | post A | −0.30 | 22.7 / 7.1 | −2.2 | 1 / 1 |
| | post B | −0.23 | 13.9 / 3.8 | +6.8 | |

Across the 4 training trials the γ1pedc efficacy of A's KCs fell
0.68 → 0.47 → 0.32 → 0.23, and α3 fell 0.62 → 0.40 → 0.25 → 0.16. B's fell to
0.91 / 0.84. All 4 real whip cracks hit, and each triggered the punishment. There
were 0 runaway chunks. Wall time was 167 s for about 70 s of sim time. With 3 s
instead of 5 s rests, B ended at 0.82 (more carry-over of eligibility). The
pre-test of A alone already moves A's efficacy to 0.99: the odour sometimes
drives the DANs just past the 100 Hz threshold by itself.

**Honest reading.** After conditioning, the brain's odour-A → walking command
effect disappears. This is the left-turn drive of about −0.4 that A evoked before.
B keeps part of its effect (−0.28 → −0.16, through the shared KCs), and the
control is unchanged. But the **measured yaw does not show a clean effect**: over
1.7 s windows it is dominated by gait noise (±8 deg/s), whatever the drive. So in
this model, fear learning is clearly visible in the brain and in the descending
command, and it is not a robust behavioural avoidance. The fly does not turn away
from or leave zone A.

Frames checked (renderer PNGs, not committed): the default row of small zones (a
magenta A column with the fly inside and a teal B column beyond it), a
test-protocol A zone around the fly, and a post-test B zone around the fly. The
translucent haze reads clearly, and the fly and the whip stay visible.

## Use

```bash
# odour zones only (the brain gets the odour KC codes; no learning)
.venv/bin/python scripts/run_sim.py --odor-zones --brain-steer
# fear learning: whip (SPACE / arrows) while the fly is in the magenta A haze
.venv/bin/python scripts/run_sim.py --learning --brain-steer
# offline experiment / behaviour protocol
.venv/bin/python scripts/demo_fear_learning.py --seeds 0 1 2 --whip-check --lesion-check
.venv/bin/python scripts/demo_fear_learning_app.py            # add --no-learning for the control
```

* Keys 8 / = put an odour A (magenta) / B (teal) haze zone around the fly. The
  default layout is 6 zones (A, B, A, …) every 14 mm along the path, starting at
  x = 12 mm, with a radius of 4 mm.
* With `--learning`, a whip hit while the fly is in a zone, or left it less than
  0.5 s ago, drives the PPL1 DANs at 200 Hz for 1 s. A hit outside every zone does
  nothing extra.
* HUD: `ODOUR in A  punish A3 B0  KC>MBON(PPL101) A 0.33 B 0.94  learning ON`
  and `LEARNING (model) KC>MBON efficacy [PPL101] A 0.33 B 0.94  DAN 0 Hz`.
  The brain window header shows `learning (model) KC>MBON PPL101: A 0.33 · B 0.94`
  when there is room.
* `summary.json` has an `odor` block (time in zones, entries, punishments) and a
  `learning` block (the last `BrainState.learning`).
* Config: `{"odor": {...OdorConfig}, "brain": {"odors": true, "learning": true,
  "plasticity_config": {...PlasticityConfig}}}`. At run time use
  `BrainProcess.set_plasticity(dict | None)`.

## Limitations

* **The punishment is a stand-in.** The model's whip afferents do not reach PPL1.
  A real shock pathway runs through the VNC and ascending neurons that this brain
  model lacks.
* **No real olfaction.** Odours are KC codes; the AL is bypassed, and AL-MBDL1 is
  silenced so that the model's AL runaway cannot ignite. The glomerulus → odour
  labels are arbitrary.
* **Only depression.** The rule has one timescale and only LTD. It has no
  backward-pairing potentiation (Handler et al. 2019), no DAN-alone potentiation,
  no reward (PAM) compartments (configurable through `dan_types` but untested), and
  no consolidation or extinction. The parameters are free and not fitted. The
  100 Hz dopamine threshold is relative to this model because KC → DAN excitation
  is strong here.
* **The behavioural effect is a loss, not avoidance.** Conditioning removes the
  odour-evoked walk / left-turn DN drive, which runs through MBON14 and MBON18. The
  fly does not steer away from odour A, because the model's MBON → DN routes carry
  no avoidance signal and the odour code is not lateralised (both MBs are driven).
* **The memory lives in the worker.** It persists over fly and brain resets
  (`persist_on_reset`) but not over app restarts.
