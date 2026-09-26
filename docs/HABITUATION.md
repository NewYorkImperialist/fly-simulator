# Looming habituation: the giant-fibre escape tires of harmless looms

If you loom at a fly over and over and nothing hits it, it stops jumping. After a
rest it jumps again. `--habituation` adds this to the connectome brain as a
documented, phenomenological **short-term synaptic depression** on the giant fibre's
looming inputs. It is off by default, and when it is off the engine is bit-identical
to the original model (tested).

Code: `fly_simulator/brain/habituation.py` (config + worker-side model), the kernel
branch in `fly_simulator/brain/engine.py` (`LIFEngine.set_depression`), wiring in
`fly_simulator/brain/process.py` (`BrainConfig.habituation`,
`BrainProcess.set_habituation`), `fly_simulator/brain_link.py` (flag, HUD) and
`tests/test_habituation.py`.

## Biology and where the depression sits

* **The GF escape habituates, and the decrement is in its afferents.** Engel & Wu
  (1996, *J Neurosci* 16:3486) stimulated the giant-fibre (GF, DNp01) pathway
  repeatedly and found that the GF-mediated response habituated, recovered on its
  own, and could be dishabituated by a novel stimulus. They put the decrement in
  the afferent pathway onto the GF: the GF → TTMn / DLMn output follows high rates
  without failing.
* **The GF's looming afferents are LC4 and LPLC2.** von Reyn et al. 2017 (*Nat
  Neurosci*) and Ache et al. 2019 (*Curr Biol*) identify them. FlyWire v783 agrees:
  they are the two biggest identified inputs to the two GFs, with LPLC2 at 1080
  synapses (11.5 % of GF input) and LC4 at 805 (8.6 %). DNp70 is next with 587.
* **Model of the decrement.** We use the resource-depletion model of short-term
  depression (Tsodyks & Markram 1997 *PNAS*; Abbott et al. 1997 *Science*).
  Homosynaptic depression is the classic mechanism of short-term habituation
  (Castellucci & Kandel 1974, *Aplysia*).

We do not cite a behavioural dataset for looming habituation in Drosophila, so the
time course below is **illustrative, not fitted**.

| stage | status |
|---|---|
| loom → LC4 (key O) / LC4 + LPLC2 (eyes, swatter) spikes | as before (`mapping.py`) |
| **efficacy of the LC4 / LPLC2 → DNp01 synapses** | **modelled** (depletion model, parameters chosen here) |
| everything else (GF spikes, the 60 Hz jump rule, MDN, ...) | connectome / unchanged |

## Model

Each presynaptic neuron j of type `pre_types` (default LC4 and LPLC2) has an
efficacy x_j between 0 and 1 that is shared by its synapses onto `post_types`
(default DNp01; `()` means all of its targets):

```
at delivery of each of j's spikes:  x_j <- 1 - (1 - x_j) exp(-(t - t_last)/tau_rec)   (exact recovery)
                                     PSC onto each depressed target = w * x_j
                                     x_j <- x_j - U x_j
```

This runs inside the numba kernel, per spike (`use_std` branch). When it is off the
kernel takes the unchanged arithmetic path, and when it is on with U = 0 the spikes
are identical too (both tested). Synapses from j onto non-target neurons are not
touched (tested bit for bit). Defaults: **U = 0.006, tau_rec = 20 s**. These were
picked with the real brain so that the key-O loom (LC4 at 200 Hz for 1 s, every
2 s) stops triggering jumps after about 3 trials and recovers within 10-60 s.
With U = 0.002 the jumps never stopped in 8 trials (the onset burst stays above
60 Hz). With U = 0.01 they stopped after 2 trials.

**Dishabituation** (`dishabituate_frac`, default 0 = off): a whip hit or shove moves
every efficacy that fraction of the way back to 1. This is a phenomenological
shortcut. In the dual-process view (Groves & Thompson 1970), dishabituation is a
superimposed sensitisation rather than an undoing of the depression. The
octopamine layer (`--stress`) is the more principled sensitisation route, but in
this model hits don't reach the GF through it. A fly / brain reset keeps the
depression (`persist_on_reset`).

Readout (`BrainState.habituation`): `efficacy` is the mean efficacy of the most
depressed type, `by_type` gives the per-type means (LC4, LPLC2), plus `u`,
`tau_rec_s`, `n_pre` and `n_post`. The HUD line `HABITUATION (model) GF-input
efficacy LC4 0.11 LPLC2 1.00` and the brain window header show it. In the brain
window it appears only when the header has room, i.e. at the default 1280 px
width.

## Measured (real brain)

**Engine in-process.** 8 O-looms (LC4 200 Hz, 1 s) with a 2 s onset spacing, then
single test looms after rests. "Jump" uses the fast-path rule the body uses: the
GF trailing 20 ms rate goes above 60 Hz. Three seeds gave the same pattern:

| trial | off: GF 20 ms peak / mean | U 0.006: GF peak / mean | jump (off / on) |
|---|---|---|---|
| 1 | 200 / 136 Hz | 175 / 74-79 Hz | Y / Y (12-13 ms) |
| 2 | 200 / 131 | 100 / 7-9 | Y / Y (20-23 ms) |
| 3 | 200 / 134 | 50-75 / 1-2 | Y / Y in 2 of 3 seeds |
| 4 | 200 / 134 | 0-50 / 0-1 | Y / **n** (3 of 3) |
| 5-8 | (not run in-process; the app control below jumped 8 / 8) | 0-50 / 0-1.5 | - / **n** (all) |
| then 10 s rest | | 100-125 / 12-17 | Y (20-21 ms, weak) |
| then 30 s rest | | 150 / 54 | Y (12-14 ms) |
| then 60 s rest (2 seeds) | | 175 / 74-77 | Y (full recovery) |
| dishabituation (frac 0.8, seed 0): 12 habituated looms, then one whip hit | | 175 / 62 | n -> **Y** |

The in-process runs drive `process._Model` and its `StimulusMapper` the same way
the worker does, in 20 ms chunks. The scratch script is not in the repo; the
app run below reproduces the effect end to end. U = 0.006,
tau_rec = 20 s; seeds 0, 1 and 2.

**Full app, headless** (`--brain-actions --habituation --brain-record`, flat,
O at t = 2, 4, ..., 16 s, then at 26 s and 46 s; GF values are 0.1 s window
peaks / means from the brain recording). The jump count comes from the brain
action events:

```bash
.venv/bin/python scripts/run_sim.py --headless --brain-headless --brain-actions \
    --habituation --brain-record --terrain flat --max-seconds 47.5 \
    --script-keys "2:o,4:o,6:o,8:o,10:o,12:o,14:o,16:o,26:o,46:o"
```

| loom | t (s) | LC4 efficacy at onset | GF peak / mean (1 s) | outcome | control (no habituation) |
|---|---|---|---|---|---|
| 1 | 2 | 1.00 | 120 / 76 Hz | jump (30 ms) | jump, GF 155 / 136 |
| 2 | 4 | 0.36 | 35 / 8 | jump (45 ms) | jump, 140 / 130 |
| 3 | 6 | 0.18 | 20 / 2 | jump (45 ms) | jump, 150 / 132 |
| 4 | 8 | 0.12 | 10 / 1 | **no jump** | jump, 140 / 132 |
| 5 | 10 | 0.11 | 15 / 2 | **no jump** | jump |
| 6-8 | 12-16 | 0.10-0.11 | 5 / 0 | **no jump** | jump (8 / 8 overall) |
| 9 | 26 (9 s rest) | 0.40 | 30 / 6 | jump (45 ms) | |
| 10 | 46 (19 s more) | 0.67 | 95 / 43 | jump (30 ms) | |

With habituation the fly jumped 5 times in 10 looms; the control jumped 8 times in
8. Late in the habituated series the GF onset burst still reaches exactly 75 Hz in
a 20 ms window: three spikes across the two GFs, just over the 60 Hz rule. Whether
a trial jumps then depends on a single spike, which is why "jump probability"
falls in a step rather than smoothly. Wall time was 96 s for the 47.5 s run, with
1.4 GB peak RSS.

## Use / config

* `--habituation` (implies `--brain`). Pair it with `--brain-actions` to see the
  effect on jumps.
* Config: `{"brain": {"habituation": true, "habituation_config": {"u": 0.006,
  "tau_rec_s": 20, "pre_types": ["LC4", "LPLC2"], "post_types": ["DNp01"],
  "dishabituate_frac": 0.8}}}`.
* At run time: `BrainProcess.set_habituation(dict | None)`.

## Limitations

* The parameters are free and not fitted to fly data. The key-O stimulus (200 Hz
  for 1 s on every LC4) is itself an idealisation. Shorter visual looms (the
  swatter, `--whip-vision`) deplete less per trial.
* Only the direct LC4 / LPLC2 → GF synapses depress. The indirect routes (LC4 →
  PVLP / DNp70 → GF) and the LC4 → MDN pathway do not, so backing up (MDN) does not
  habituate unless you set `post_types: []` (all targets).
* There is a single depression pool: no separate fast and slow components, no
  long-term habituation across sessions, and no stimulus specificity beyond "which
  presynaptic neurons fired".
* Dishabituation is a reset shortcut, not a sensitisation process (see above).
