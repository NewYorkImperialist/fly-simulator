# Stress / arousal: an octopamine layer on top of the connectome brain

Whip hits make the fly speed up for a while, then calm down. The signal goes through
the nervous system: the hits drive the brain model, the model's octopaminergic (OA)
neurons fire, a slow octopamine level integrates their spikes, and the level sets
the walking vigour. The level also makes the fly jumpier and changes brain
excitability. Code: `perpetualfly/brain/neuromod.py` (brain side, runs in the brain
worker), `perpetualfly/stress.py` (body side), the PAIN/AROUSAL row of the brain
window, `scripts/demo_stress.py` and `tests/test_stress.py`.

## Biology, and which parts are real and which are modelled

Flies have no cortisol. Their counterpart of noradrenaline / fight-or-flight is
**octopamine**. OA neurons are recruited by threats and startle, and octopamine
raises arousal and locomotor vigour: Tbh mutants, which lack octopamine, walk less
and more slowly (Roeder 2005; Suver et al. 2012 for OA and visual-motion gain).
**DH44** neurons (the CRF homologue) are the insect stress-hormone axis. The model
reports their firing (`dh44_hz`); no stimulus here ever reached them (0 Hz).

| stage | status |
|---|---|
| hit → sensory neurons (body mechanosensory, head bristle, JO sets) | as in docs/BRAIN.md (`mapping.py`) |
| hit → **nociceptive relay**: ascending neurons `an_walk` (AN_AVLP_PVLP, 10/side) and `an_arousal` (AN_IPS_GNG_7, 6/side), both sides | **our choice** (a stand-in for the VNC, which the model lacks). The groups come from docs/SENSORY_SCREEN.md. Label: "VNC stand-in", chip `HIT_RELAY` |
| relay / loom / fall → OA-neuron and walk-DN spikes | **connectome** (Shiu et al. LIF on FlyWire v783) |
| OA spikes → octopamine level (leaky integrator) | **modelled** (phenomenological) |
| level → threshold shift on the OA neurons' synaptic partners | **modelled**. The targets come from the connectome, but receptor expression is not in the connectome |
| level → CPG frequency and stride amplitude, jump threshold | **modelled** |

**OA neurons used:** the 43 FlyWire neurons whose `cell_type` starts with `OA-`:
OA-VUMa1–8, OA-VPM3/4, OA-AL2b1/b2/i1–i4 and OA-ASM1–3 (Busch et al. 2009 names).
The predicted-transmitter label (`nt == OA`, 210 neurons) is not used. In v783 it
also tags 100 photoreceptors (R1-6/R7/R8, which are histaminergic; the classifier
has no histamine class), JO afferents and DH44 cells.

## Do the real OA neurons respond to our stimuli? (connectome only)

`scripts/demo_stress.py --part brain`, section 1. The engine runs in-process from
rest, and the window covers the stimulus plus 0.4 s.

| stimulus | OA-* mean (0.1 s peak) | OA neurons active | main responders |
|---|---|---|---|
| whip L1 / L4, thorax, left | **0.00 Hz** | 0 / 43 | none |
| whip L4, head | **0.00 Hz** | 0 / 43 | none |
| shove L4 | **0.00 Hz** | 0 / 43 | none |
| fall (both sides of body_mech + JO) | 0.51 Hz (1.4) | 7 | OA-VUMa1 4 Hz, OA-AL2b2 4 Hz |
| loom LC4 200 Hz 1 s (key O) | 5.1 Hz (9.3) | 9 | OA-AL2b2 39, OA-AL2i3 27–29 |
| loom LPLC2 200 Hz 1 s | 9.0 Hz (14.2) | 12 | OA-AL2b2 52, OA-AL2i3 51 |
| head bristles 200 Hz 1 s | 8.7 Hz (16.3) | 14 | OA-AL2i3 36, OA-AL2b2 36 |
| sugar 200 Hz 1 s | 0.00 Hz | 0 | none |

So, measured honestly: **whip hits and shoves do not reach the OA neurons at all in
this model**, at any strength. Ten L4 hits leave the level at 0.03. Falls barely
reach them (3 falls: 0.006). Looming and head-bristle input do reach them, through
the OA-AL2 cluster. docs/SENSORY_SCREEN.md reports the same: body_mech → OA ≤ 1 Hz,
and FlyWire has no annotated nociceptors, since class IV/ppk nociceptors end in the
VNC. For a whip hit to "hurt", the hit is relayed to ascending neurons that do reach
the OA neurons (the next section).

## Design

### Level (brain worker, `OctopamineModel`)

```
u_OA  = r_OA / ref_rate_hz                 r_OA = mean rate of the 43 OA-* neurons in the chunk
u     = min(u_OA [+ u_noc], max_drive)
dL/dt = u (1 - L) / tau_rise - L / tau_decay        L in [0, 1], integrated exactly per 20 ms chunk
```

Defaults: `tau_rise_s = 1`, `tau_decay_s = 30` (configurable, 20–60),
`ref_rate_hz = 10`. The level saturates toward 1, and repeated events accumulate
because the decay is slow. Each `BrainState.neuromod` carries `octopamine`,
`oa_rate_hz`, `dh44_hz`, `noci_hz`, `n_targets`, `vth_shift_mv`, `noci_relay` and
`runaway`. The level persists across fly and brain resets
(`persist_on_reset=True`), because a neuromodulator does not vanish when the fly is
put back.

### "Pain": the nociceptive relay (`noci_relay`, on with `StressConfig`)

Every `whip_hit` / `shove` event also drives, on both sides, for the hit duration
plus 0.2 s:

* `an_walk` at `100 + 50 * intensity` Hz. Through the connectome this reaches the
  walk DNs BDN2 / oDN1 / P9, so with `--brain-steer` the brain's own drive gives an
  immediate speed-up burst.
* `an_arousal` at `50 + 70 * intensity` Hz. Through the connectome this reaches
  OA-VUMa1 and the other OA neurons, and so the slow level.

*Which* ascending neurons carry the hit is our choice. Everything downstream of them
is connectome. An alternative, fully modelled link, `nociceptive_input` (off), feeds
the hit afferents' own spike rate straight into `u`.

### Brain effect (modelled)

The spike threshold of the **13,589 direct synaptic partners** of the OA-* neurons
(at least 5 synapses in total) is lowered by `vth_shift_mv * L`, 1 mV at L = 1
against the 7 mV rest-to-threshold gap. These partners stand in for the neurons near
OA release sites, because receptor expression is not in the connectome. Most are
optic-lobe neurons (9,481; OA-AL2i cells project there), which fits OA's known role
in visual gain. They also include the walk DNs, the turn DNs and MDN. The giant
fiber is not among them. The engine takes a per-neuron threshold array only while
the shift is non-zero. Otherwise it runs the scalar code path unchanged.
**Engine exactness:** with the layer absent, disabled or at level 0, the spike
trains are bit-identical to the plain engine (`test_engine_bit_identical_without_
modulation`), and the Brian2 spike-for-spike test still passes.

**Runaway guard.** The Shiu model can fall into a global self-sustaining state of
about 475k spikes/s (a model artefact; docs/SENSORY_SCREEN.md). It happened once in
our runs, during a series of 8 relayed hits at level ~0.5: the level pinned at 0.98
and never decayed. The guard now works like this: while the whole-brain rate is
above `runaway_sps` (100k/s), the level gets no input and only decays, the threshold
shift is removed, and the readout sets `runaway` (the gauge shows RUNAWAY). In
spot checks the shift did not change the runaway rate measurably: 0/36 runs at
level 0, 0/15 at level 0.6 with 1 mV, and 1/3 at level 0.6 with 0.5 mV. These are
small samples.

### Body effect (`perpetualfly/stress.py`, modelled, bounded)

* CPG intrinsic frequency × `1 + 0.5 L` (max ×1.5). Implemented by scaling the
  hybrid controller's `_base_intrinsic_freqs` in place, so it survives resets.
* Stride amplitude: the final `[left, right]` drive × `1 + 0.2 L`, applied after the
  heading hold and the brain's steering filter, sign kept, each side capped at 1.5.
* Jump threshold with `--brain-actions`: giant fiber > `60 Hz × (1 − 0.5 L)`, not
  below 20 Hz.
* Disabled (`enabled=False`): nothing is touched. No neuromod command reaches the
  worker, and the controller, filter, triggers and `BrainLink.update` are left as
  they are (tested). `close()` restores everything.

## Measured effects

### Speed vs. level (physics only, level set directly, heading hold on)

| frequency × / amplitude × | flat mm/s | stability |
|---|---|---|
| 1.0 / 1.0 (calm) | 14.1 | |
| 1.5 / 1.0 | 21.3 | |
| 1.0 / 1.3 | 17.2 | |
| **1.5 / 1.2 (level 1, default)** | **24.5** (normal terrain 24.5) | normal terrain 20 s: 0 falls. **hard** terrain 10 s: 1 flip (baseline 0) |
| 1.25 / 1.1 (level 0.5) | — | hard 10 s: 0 falls, 19.1 mm/s |
| 1.6 / 1.3 | 27.8 | not used |

### Whip hits → speed-up → calm down (real brain, full app wiring)

`scripts/demo_stress.py --part body --whip-level 2 --hits 8` runs the real brain
worker (`pace="sim"`) with `--brain-steer --brain-actions` wiring and
`install_stress`. It gives 8 L2 whip cracks, 1.5 s apart, alternating sides, on
flat ground. Speed is measured over 2 s windows.

| | level | step × / stride × | speed |
|---|---|---|---|
| calm | 0.00 | 1.00 / 1.00 | 14.1 mm/s |
| after hit 1 / 2 / 4 / 8 | 0.22 / 0.33 / 0.49 / 0.71 | → 1.36 / 1.14 | |
| 0–2 s after the last hit | 0.67 | 1.33 / 1.13 | **21.2 mm/s** |
| 8–10 s after | 0.51 | 1.26 / 1.10 | 19.5 |
| 20–22 s after | 0.34 | 1.17 / 1.07 | 17.7 |
| 40–42 s after | 0.18 | 1.09 / 1.04 | 15.9 |
| stress off (same hits) | 0 | 1 / 1 | 14.1 |

During each hit the OA neurons peaked at 5–13 Hz, all through the relay. With stress
on there was one fall in 8 hits (control: 0/8). With L3 hits the fly falls on almost
every hit (6/6 with and without stress). Falls add body-afferent input but hardly
any OA input.

Brain bench (in-process, from rest; each hit alone, with the relay):

| hit | OA-* peak | walk DN peak | level after one hit |
|---|---|---|---|
| L1 (0.15) | 4.4 Hz | 53 Hz | +0.10 |
| L2 (0.25) | 8.4 | 43 | +0.18 |
| L3 (0.6) | 14.0 | 65 | +0.27 |
| L4 (1.0) | 14.0 | 78 | +0.20 |
| L4 head | 8.4 | 70 | +0.17 |

GF stayed at 0 and MDN at ≤ 2 Hz, and nothing kept firing 0.5 s after the hit. The
per-hit increments are noisy, because OA-VUMa1 is only 2 neurons (L3 > L4 in this
run). Across 8 L2 hits the level went 0.13 → 0.25 → 0.46 → 0.73. It then decayed to
0.52 after 10 s, 0.27 after 30 s and 0.10 after 60 s (τ = 30 s).

### Looming (connectome only) and jumpiness

Three LC4 looms (key O), 3 s apart, took the level to 0.53 / 0.78 / 0.84. It then
decayed to 0.60 after 10 s, 0.31 after 30 s and 0.11 after 60 s.

**Weak loom → jump** (app, `--scenario jump`: LC4 at 42 Hz for 0.5 s, 6 trials per
condition, 2 s apart):

| | level | GF max (mean) | threshold | jumps |
|---|---|---|---|---|
| stress off, calm | 0 | 53 Hz | 60 | 0/6 |
| stress off, after 3 looms | 0 | 53 | 60 | 0/6 |
| stress on, calm | 0.01 | 51 | 60 | 0/6 |
| **stress on, after 3 looms** | 0.57–0.81 | 43 | 36–43 | **4/6** |

After the looms the stressed fly walked at 22.7 mm/s against 14.1 calm. 20 s later
the level was 0.28 and the speed 16.8 mm/s. At LC4 50 Hz the result was 1–2/5 jumps
calm and 5/5 stressed.

**Brain-side effect alone** (level fixed; weak loom LC4 at 40–60 Hz; max over
0.1 s windows): the threshold shift *lowers* the giant fiber response (LC4 50 Hz:
mean GF 49 → 41 → 31 Hz at L = 0 / 0.5 / 1), because inhibitory neurons upstream of
the GF are targets too. Turn DNs go up (2.5 → 12.5 Hz). Total activity changes by
+5 % or less. In the bench (8 trials), jump probability for LC4 50 Hz was 0/8 at
L = 0, 5/8 at L = 0.3 and 6/8 at L = 0.6. For 40 Hz it was 0/8, 0/8 and 3/8. So the
jumpiness comes from the body-side threshold, partly offset by the brain-side
effect. That offset is a prediction of our target choice, not a validated result.

## Brain window and HUD

The DESCENDING COMMANDS panel gains a **PAIN/AROUSAL · octopamine · (model)** row as
soon as a state carries an enabled `neuromod`. The row has a 10 s trace of the level
(0–1), the value, the OA-neuron rate, `+hit relay*` when the relay is on (`*` =
modelled path) or `RUNAWAY`, and a vertical gauge. There is no row without the
layer. Frames checked: during a loom (level 0.45 → 0.63 step visible) and after 8
relayed whip hits (walk-DN bursts at each hit, level 0.71, `HIT_RELAY AN_AROUSAL` /
`AN_WALK` chips). `StressHandle.hud_line()` gives, for example:

```
PAIN/AROUSAL 0.71 [octopamine (model)] <- OA neurons 0.0 Hz (hits via VNC-stand-in relay) | step x1.36 stride x1.14 | jump > 39 Hz
```

## Integration API (for app.py; not wired yet)

```python
from perpetualfly.stress import StressConfig, install_stress, METRIC_COLUMNS
# after Session(cfg, brain=link) (BrainLink.attach done):
session.stress = install_stress(session, cfg.stress)     # cfg.stress: StressConfig
# per physics chunk: automatic (it wraps link.update; pass hook_update=False and
# call session.stress.update() in Session.after_physics instead, if preferred)
hud += [session.stress.hud_line()]                      # "" when disabled
metrics: METRIC_COLUMNS / session.stress.metric_row()
summary["stress"] = session.stress.summary();  session.stress.close()  # before link.close()
```

Suggested CLI: `--stress` → `StressConfig(enabled=True)`, which implies `--brain`.
Recommend `--brain-steer` (the relay's walk-DN bursts) and `--brain-actions` (the
jump threshold). Config keys: `freq_gain`, `amp_gain`, `jump_threshold_drop`,
`noci_relay` (False = connectome-only input), `neuromod` (NeuromodConfig fields:
`tau_rise_s`, `tau_decay_s`, `ref_rate_hz`, `vth_shift_mv`, `target_min_syn`,
`relay_walk_hz`, `relay_arousal_hz`, `runaway_sps`, ...). The brain can also be
configured without the body layer: `BrainConfig(neuromod={...})` or
`BrainProcess.set_neuromod({...} | None)` at run time.

## Limitations

* The whip → OA route exists only through our relay. The connectome itself does not
  route body touch to OA neurons, because the VNC is missing.
* The level dynamics, the target set, the threshold-shift mechanism and all body
  gains are free, phenomenological choices. Octopamine's real actions (muscle,
  sensory and central receptors, metabolism) are much broader.
* The OA response is carried by a handful of neurons (OA-VUMa1 = 2 cells, OA-AL2
  ~10), so per-event increments are noisy.
* At full stress on hard terrain the faster gait flipped once in 10 s. Normal
  terrain was stable over 20 s.
* The level update reaches the body with up to one `BrainState` (0.1 s) of delay.
  That is negligible next to τ_rise = 1 s.

```bash
.venv/bin/python scripts/demo_stress.py --part brain            # sections 1-4 (~5 min)
.venv/bin/python scripts/demo_stress.py --part brain --skip-effect   # 1-2 only
.venv/bin/python scripts/demo_stress.py --part body --whip-level 2 --hits 8 --out <dir>
.venv/bin/python scripts/demo_stress.py --part body --scenario jump --jump-rate 42 --trials 6
.venv/bin/python -m pytest -q tests/test_stress.py
```
