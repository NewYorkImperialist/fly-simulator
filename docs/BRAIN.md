# Brain engine: Shiu et al. 2024 whole-brain LIF model of FlyWire, in its own process

`perpetualfly/brain/` runs the leaky integrate-and-fire (LIF) model of the whole
adult *Drosophila* central brain from Shiu et al. (2024, *Nature* 634:210), which is
built on the FlyWire v783 connectome. It runs in a separate process. Body events
(whip hits, falls, and so on) become Poisson input to sensory neurons, and the
process publishes activity snapshots (`BrainState`) a few times per second for the
brain window and the app. The message contract is `perpetualfly/brain/schema.py`. The app integration
added optional fields to it (see *Integration* below).

## What the model is, and what it is not

* **It is** Shiu et al.'s model with their equations and parameters. There are
  138,639 neurons and 15,091,983 connections (signed synapse counts), with
  `dt = 0.1 ms`:
  ```
  dv/dt = (v_0 - v + g) / t_mbr        (unless refractory)
  dg/dt = -g / tau                     (unless refractory)
  v > v_th  ->  spike;  v = v_rst, g = 0;  refractory 2.2 ms
  pre spike ->  after 1.8 ms: g_post += 0.275 mV * sign * n_synapses
  Poisson input (sensory drive): v += 0.275 mV * 250 per event (targets: no refractory period)
  v_0 = v_rst = -52 mV, v_th = -45 mV, t_mbr = 20 ms, tau = 5 ms
  ```
  The sign of each connection comes from the predicted transmitter of the
  presynaptic neuron (Eckstein et al. 2024). ACh, DA, 5-HT and OA count as
  excitatory; GABA and Glu count as inhibitory.
* **It is not** a model of neuromodulation. Dopamine, serotonin and octopamine
  neurons act only as fast excitatory synapses, as in Shiu et al. The window colours
  neurons by predicted transmitter, but the dynamics do not distinguish them.
* There is **no VNC.** FlyWire covers the brain only, and most body mechanosensors
  and all leg motor circuits live in the ventral nerve cord. The descending neurons
  are therefore the output of the model, and they drive nothing downstream.
* All neurons share **uniform parameters**. There are no gap junctions, no
  plasticity, no spontaneous activity (the brain is silent until driven) and no
  morphology. Synapse counts are the weights.
* Shiu et al. validated the model on *specific* sensorimotor pathways (sugar and
  water to MN9 feeding, bitter inhibition, antennal grooming), at about 90 %
  accuracy for the predicted activations tested. Beyond those pathways, treat it as
  a plausible, connectome-constrained activity pattern, not a prediction.

## Engine: why a numba port, and how it was checked

`perpetualfly/brain/engine.py` is a from-scratch, event-driven reimplementation of
those equations, compiled with numba. It keeps state between calls, so it can
advance in arbitrary chunks, with stimuli changed between chunks.

* It integrates only neurons away from rest (an "active set"). A resting neuron
  (`v = v_0`, `g = 0`) is an exact fixed point, so skipping it is exact. A neuron
  whose `|v - v_0|` and `|g|` both fall below 1e-6 mV is snapped to rest. Spikes are
  delivered by walking the presynaptic CSR row. The cost per step is therefore
  proportional to active neurons plus spikes times out-degree, not to N.
* The per-step order follows Brian2's default schedule: state update, threshold,
  synaptic delivery and Poisson input, then reset. The integration is the exact
  solution of the linear ODE, which is what Brian2's `method='linear'` computes
  (checked against `scipy.linalg.expm` in the tests).
* **Brian2 subtlety, reproduced:** variables flagged `(unless refractory)` are
  *conditionally written* in Brian2. Any synaptic `g += w` or Poisson `v += ...`
  that reaches a refractory neuron is discarded, including in the step the neuron
  spikes. The first version of the port missed this, and its downstream rates came
  out 30-80 % too high; MN9 fired at 113 Hz where Brian2 gives 89 Hz. The engine now
  implements it.
* **Checks:**
  1. On a 40-neuron recurrent network, the engine produces the same spikes as
     Brian2 (numpy target), spike for spike
     (`tests/test_brain.py::test_engine_matches_brian2_spike_for_spike`). It also
     matches a dense numpy reference implementation.
  2. **Sanity check on the full connectome:** the Shiu et al. sugar GRNs (20 of
     their 21 v630 root ids still exist in v783) were driven at 200 Hz, 10 trials of
     1 s each, in both Brian2 (runtime, cython) and the engine. Results are in
     `scripts/bench_brain.py --validate 10`:

     | | Brian2 2.10.1 | numba engine |
     |---|---|---|
     | MN9 (720575940660219265) rate | 88.6 ± 6.3 Hz | 92.7 ± 4.2 Hz (88.0 ± 3.2 in an earlier run with another RNG stream) |
     | neurons active | 432 | 436 |
     | per-neuron mean rates | | Pearson r = 0.9996; 99.8 % of active neurons agree within 3 SEM |

     Shiu et al.'s own published run of this experiment (their `results/example/sugarR.parquet`,
     FlyWire v630, 30 trials at 200 Hz) has MN9 at 93 Hz and about 450 active
     neurons. That matches, so the model is wired correctly: sugar GRN input reaches
     the proboscis motor neuron MN9 through the GNG / PRW feeding circuit.

## Benchmark (Apple M1, 4 P-cores, single thread; `scripts/bench_brain.py`)

RTF means simulated brain seconds per wall second. The engine is single-threaded:
a 4-thread numba `prange` version of the update step reached only 0.64 against 0.69
serial on the heaviest case, because spike delivery dominates the cost, so it was
dropped.

| engine / mode | load | chunked stepping? | RTF (sugar 200 Hz) | notes |
|---|---|---|---|---|
| **numba port (chosen)** | 1.0 s load + 0.2 s JIT (cached) | yes, any chunk size, no overhead | **~4** (10/20/50 ms chunks alike) | quiet brain: >1000x real time; left whip (body + JO): ~2.5-5; heavy (sugar+bitter+all body mech+LC4 at 200 Hz): ~0.7 |
| Brian2 runtime (cython) | 1.5-4 s build + 46 s first compile | yes (`net.run()` repeatedly) | 0.31 in 1 s chunks, 0.16 in 50 ms chunks, 0.09 in 20 ms chunks | about 0.3 s fixed overhead per `run()`; RSS 1.6 GB |
| Brian2 cpp_standalone | 7 s compile | **no**: each `device.run()` restarts the binary, reloads 15 M synapses and starts from the initial state (run_args can only set initial values) | 0.25 (1 s run takes 4-5 s including start-up) | chunking would mean relaunching and restoring state every chunk; not viable |

These numbers come from `bench_brain.py --brian2 --standalone --validate 10`. The
engine uses about 400 MB RSS in the worker, of which 180 MB is CSR arrays.

**Realistic publish rate.** By default the worker paces brain time to wall time
(`realtime=True`). The fly app uses `pace="sim"` instead (see *Integration*). It publishes one `BrainState` per `window_s` of brain time, 0.1 s
by default, so **10 states/s** whenever the load is below real time: quiet,
whip hits, sugar. During heavy transients (fall: body + JO on both sides at
200 Hz; head hits; LC4 looming), the engine runs at 0.5-0.85x real time. The brain
then lags wall time and publishes at 5-8 states/s until the transient passes. It
does not try to catch up afterwards, and `BrainState.realtime_factor` shows the lag.
States tile brain time: every spike appears in exactly one state. A state pickles to
roughly 5-40 KB and the layout to about 110 KB.

## Data (`scripts/fetch_brain_data.py` -> `data/brain/`, 149 MB, gitignored)

| file | source | size | licence |
|---|---|---|---|
| `Completeness_783.csv`, `Connectivity_783.parquet` | [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) @ `91bdd1e` | 3.3 + 100.8 MB | code MIT; data derived from FlyWire |
| `flywire_neuron_annotations.tsv` (Supplemental file 1) | [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations) v3.1.0 (`8587524`); Schlegel et al. 2024, Berg et al. 2025 | 31.7 MB | FlyWire data, CC BY-NC 4.0 |
| `per_neuron_neuropil_count_pre_783.feather` | Zenodo [10.5281/zenodo.10676866](https://zenodo.org/records/10676866) (Dorkenwald et al. 2024) | 16.9 MB | the record states CC BY 4.0; FlyWire terms are CC BY-NC 4.0 |
| `neurons_783.npz` | built locally from the files above (model order: transmitter, sign, primary neuropil, positions, annotation columns) | 3.5 MB | derived |

Downloads are pinned to commits or the Zenodo record and checked against SHA-256
values. Nothing requires a login; the FlyWire Codex bulk downloads, which do need
one, were not used. **FlyWire data are for non-commercial use (CC BY-NC 4.0).**
Cite Dorkenwald et al. 2024 and Schlegel et al. 2024 (Nature 634), Eckstein et al.
2024 (Cell 187) for transmitters, and Shiu et al. 2024 (Nature 634) for the model.
The code here is original. `brian2_ref.py` restates Shiu's MIT-licensed `model.py`
for cross-checking. The GPL eonsystemspbc/fly-brain port was read for ideas only;
no code was copied from it.

Python dependencies: the `brain` extra (numba, pyarrow, scipy; 137 MB installed,
mostly pyarrow) and optionally `brain-ref` (brian2 plus cython, for the
cross-check). Total disk use for data plus new packages is about 290 MB.

## Stimulus mapping (`mapping.StimulusMapper`)

Sets are chosen from FlyWire annotations. "Side" is always the *fly's* side
(FlyWire `side` column: soma side, or nerve-entry side for sensory neurons).
Shiu et al.'s "right" sugar GRNs are annotated `left` in v783, because older FlyWire
views were mirrored.

| event | neurons driven | rate |
|---|---|---|
| `whip_hit` / `shove` on the body (default) | **body mechanosensory afferents** that ascend from the VNC directly into the brain: `super_class == sensory_ascending`, sub-classes `SA_DMT_DMetaN`, `SA_DMT_ADMN`, `SA_DLV`, `SA_MDA`, `SA_VTV_DProN`, `SA_VTV_PDMN` (495 neurons, 193 L / 302 R), on the hit side. Notum, wing and haltere nerves; Schlegel et al. 2024. | `max(30, 200 * intensity)` Hz for `duration_s` |
| … plus for `whip_hit` | Johnston's-organ wind/gravity neurons (`wind_gravity`, JO-C/E; Kamikouchi et al. 2009), on the hit side, for the air the lash moves | 0.5x |
| hit whose `details["body"]` names the head, eye, antenna or proboscis | head bristle mechanosensory neurons (`head bristle`, 305), on that side, plus the body set at 0.5x | 1x |
| side `front` / `rear` / `top` / `none` | both sides | 0.7x |
| `fall` | body set on both sides plus all JO wind/gravity neurons | 1x |
| `ground_contact` | **off by default**. FlyWire's brain has no identified leg proprioceptors. With `enable_ground_contact=True` it drives the leg afferents ascending in the VTV tract (`SA_VTV_pro_meso_meta`, mostly tarsal gustatory). | 20 Hz * intensity |
| `manual` | `details={"set": name}` with name one of `sugar`, `bitter` (Shiu et al. GRN ids), `body_mech`, `head_bristle`, `jo_wind_gravity`, `jo_auditory`, `jo_grooming`, `leg_sa`, `LC4`, `LPLC2`; or `{"cell_type": "..."}`; or `{"root_ids": [...]}`. Optional `side` and `rate_hz`. | |
| `reset` | clears active stimuli. `BrainProcess.reset_state()` also returns every neuron to rest. | |

**Honesty note.** Most body mechanosensation (leg, wing and body bristles, chordotonal
organs) enters the VNC, which the model does not contain. Only the afferents above
reach the brain directly, and they are an approximation. In the model the body set
has a low out-degree (about 20 connections each) and on its own recruits only about
80 downstream neurons. The JO wind component gives a whip hit its visible
AMMC/SAD/WED activity. No whip hit, at any intensity, produced walking-relevant
descending-neuron activity in our tests. In this model a whip shows up as
mechanosensory and auditory processing, not as an escape command. Looming input
(LC4/LPLC2) does drive the giant fiber at 100-190 Hz, together with MDN at about
20 Hz; this matches von Reyn et al. 2014 and Ache et al. 2019.

## Descending readout (`mapping.DESCENDING_TYPES`, `descending_to_drive`)

Groups are split by soma side. Each value in `BrainState.descending` is the group's
mean rate (Hz) over the window.

| group | FlyWire cell types (count) | source for the behavioural role |
|---|---|---|
| `walk_L` / `walk_R` | DNg100 = BDN2, DNg97 = oDN1, DNp09 = P9 (1 each per side) | BDN2 and oDN1 drive forward walking (Sapkal et al. 2024); P9 (Bidaye et al. 2020). The DNg97 = oDN1 identification follows Sapkal et al. 2024 and matches the "P9_oDN1" ids in the eonsystemspbc/fly-brain notebook. |
| `turn_L` / `turn_R` | DNa01, DNa02 (1 each per side) | ipsilateral steering (Rayshubskiy et al. 2020; Yang et al. 2023). FlyWire also has `DNae001` with hemibrain type "DNa01"; we use the FlyWire `cell_type == DNa01`. |
| `backward_L` / `backward_R` | MDN (2 per side) | moonwalker DNs, backward walking (Bidaye et al. 2014) |
| `escape` | DNp01 = giant fiber (1 per side) | take-off escape (von Reyn et al. 2014) |
| `groom` | DNg12_a–e (21 per side, 42 in the model; both sides pooled) | anterior grooming: front-leg rubbing and head sweeps (Guo, Zhang & Simpson 2022). Added for `--brain-actions`; appended at the end of `DESCENDING_GROUPS` (metrics column `dn_groom`, GROOM trace in the brain window) |

Every group was found in the annotations.

`descending_to_drive(state) -> np.ndarray[2]` maps these rates onto the hybrid CPG
drive `[left, right]`, where |v| is the per-side amplitude and the sign is the
stepping direction. The mapping is deliberately conservative. The app decides
whether to use it; `DriveGains` holds the gains.

* A quiet brain gives `[1, 1]`, so walking simply continues.
* Walk groups raise both amplitudes by up to +0.2, using `tanh(rate / 30 Hz)`.
* `turn_L - turn_R` lowers the left amplitude and raises the right by up to 0.4. The
  hybrid controller turns toward the smaller-amplitude side, so DNa01/02 activity on
  the left turns the fly left. Amplitudes are clipped to [0.3, 1.5].
* The mean MDN rate blends both sides toward -1 (backward stepping). Stepping is
  fully reversed at 40 Hz and stops at about 20 Hz.
* `escape` and `groom` do not enter the walking drive. With `--brain-actions`
  they trigger body actions instead (giant fiber > 60 Hz → jump; DNg12 > 20 Hz
  for ≥ 100 ms → groom; MN9 > 30 Hz → proboscis extension). See "Brain → actions"
  below and docs/ACTIONS.md §4.

## Process API (`perpetualfly/brain/process.py`)

```python
from perpetualfly.brain import BrainConfig, BrainProcess, StimulusEvent, descending_to_drive

brain = BrainProcess(BrainConfig(subscribers=("app", "window")))  # spawn-context worker
brain.wait_ready()                      # ~2 s (load 1 s + layout + cached JIT)
layout = brain.layout()                 # BrainLayout, once
brain.send(StimulusEvent("whip_hit", side="left", intensity=0.8, duration_s=0.05,
                         details={"body": "Thorax"}))           # never blocks
st = brain.latest("app")                # newest BrainState or None; drops older ones
states = brain.poll("window")           # all pending states for another subscriber
sub = brain.subscriber("window")        # picklable: pass to the window's own process,
                                        # then sub.layout(timeout) / sub.latest()
brain.stop()                            # also runs at exit

# pace brain time to an external clock instead of wall time (what the fly app does):
brain = BrainProcess(BrainConfig(pace="sim", subscribers=("app",)))
brain.clock(t)                          # the fly has simulated up to t; the brain may run to t
brain.send(StimulusEvent(..., sim_time=t_hit))   # applied when brain time reaches t_hit
brain.reset_state(sim_time=t)
```

* **Design.** There is one worker (a daemon, spawn context) and one command queue.
  Each subscriber gets its own bounded state queue (8 states by default) and a
  one-slot layout queue. The worker publishes every state to every subscriber and
  drops the oldest when a queue is full. A slow or absent reader therefore never
  blocks the worker or the fly app.
* **Stimulus timing.** A stimulus takes effect at the start of the next chunk
  (20 ms of brain time by default). Chunks are also cut exactly where a stimulus
  ends and at every publish boundary.
* **No orphans.** `stop()` sends a stop command, then terminates, then kills the
  worker. The worker also exits within one chunk when its parent process
  disappears, which is tested with a parent that calls `os._exit` without stopping.
* **Layout.** Coordinates are FlyWire µm projected frontally (`u = x_nm/1000`,
  `v = y_nm/1000`), the same convention as `brain_viz/atlas.py`: the fly's left is
  on the image left, and v increases ventrally. There are 79 regions: every
  neuropil that is some neuron's primary output neuropil (most presynapses), plus
  `OTHER`. `region_xy` is the median anchor position of each region's neurons. The
  3,513 display neurons are all 1,303 DNs, MN9, the stimulus sets (capped) and 16
  per region stratified by transmitter. Their positions are FlyWire anchor points.
  `region_outline_xy` is empty; the window has its own atlas.
* `BrainConfig(synthetic={"n": 50, ...})` runs a tiny random annotated network with
  no data files, for tests and UI work.

## Integration with the fly app (`perpetualfly/brain_link.py`)

`scripts/run_sim.py --brain` (window), `--brain-headless` (no window),
`--no-brain-window`, `--brain-steer` (implies `--brain`), `--brain-backup` (implies
`--brain-steer`, MDN reference 20 Hz, see below). `BrainLink` owns the brain
worker (`BrainProcess`, one subscriber `"app"`) and the brain window
(`BrainWindowProcess`). The app relays every `BrainState` to the window after
filling in `state.drive`, so the window's DRIVE panel shows the drive the app
computed, and whether it is applied. The config is `AppConfig.brain`
(`BrainLinkConfig`), written to `config.json` together with the `BrainConfig` and
`DriveGains` actually used.

### Schema additions (all optional, at the end of `BrainState`)

`seq` (detect dropped states), `sim_time` (the fly time the state's end corresponds
to), `compute_rtf` (brain s per wall s spent computing, i.e. the headroom), `probes`
(`{"MN9": Hz}`), `drive` (filled by the app). Older producers and consumers are
unaffected.

### Time: the brain follows fly simulation time

`BrainConfig(pace="sim")` (what the app uses) replaces wall-clock pacing with a
clock from the app. After each physics chunk (5-15 ms of sim time) the app sends
`clock(run_time)` (at most every 10 ms of sim time), where run time is the fly's
monotonic time across resets. The worker never simulates past the latest clock
mark: when it catches up it waits for the next one. So:

* pausing the fly (P) pauses the brain, and the fly's 0.6x real-time speed is the
  brain's too;
* stimuli are **scheduled**: each `StimulusEvent.sim_time` (fly run time) is applied
  when brain time reaches it, or immediately if it is already past. A whip hit's
  event is emitted when the strike window closes (~30-60 ms after first contact); if
  the brain has already passed the contact time, the hit is applied up to that much
  late;
* `reset_state(sim_time)` is scheduled the same way. A fly reset also resets the
  brain (the body is teleported, so leftover activity would not belong to it);
* if the brain can't keep up, it lags. The lag (`fly run time - state.sim_time`,
  HUD `lag`) is normally 0.02-0.12 s: states cover 0.1 s and arrive ~15 ms after
  the fly passes their end. Beyond `max_lag_s` (1 s) the
  worker stops trying to catch up (the clock offset grows), so the lag stays bounded
  and physics is never blocked.

Measured in 20 s headless runs (flat, `--brain-steer`, whip hits L1-L3, a shove, a
fall, a reset, two looms, sugar, bitter): the lag stayed at 0.10-0.14 s, including
during looming, when the engine ran at 0.8x real time against the fly's 0.5-0.64x.
No state was dropped (199 of 199).

### Body -> brain events

| app event | StimulusEvent |
|---|---|
| whip hit (`WhipHitEvent`, hits only) | `whip_hit`, intensity = measured impulse / 1.0 uN*s (mean L4 impulse, docs/WHIP.md), clipped to 1: L1-L4 give ~0.15 / 0.25 / 0.6 / 1.0. Side = the crack's side in the fly's frame (`overhead` -> `top`; `random` cracks are classified from the measured impulse direction). `details.body` = the body with most impulse (head hits -> head bristles). Duration max(50 ms, contact). |
| shove (`HitEvent`) | `shove`, intensity = impulse / level-4 impulse (2.5 uN*s); side = opposite to the push (push to the fly's right = hit on its left; up -> both sides) |
| fall (detector) | `fall`, intensity 1, 0.1 s |
| recovered | `recovered` (no mapping; window chip and events.csv only) |
| fly reset | `reset_state(sim_time)` |
| pause / resume | window chip and events.csv only (the brain simply waits) |
| O | `manual` `LC4` at 200 Hz for 1 s (looming) |
| T / K | `manual` `sugar` / `bitter` at 200 Hz for 1 s |

Everything sent is logged as `brain_stim` rows in `events.csv` (details: kind,
side, intensity, duration, stimulus time, body, impulse, level).

### Brain -> body (`--brain-steer`)

1. Each `BrainState` -> `b_target = descending_to_drive(state)` (unchanged mapping).
   It is held until the next state, and set to `[1, 1]` if the latest state is
   older than 2 s of fly time or the brain has died.
2. `b` = first-order low-pass of `b_target` in fly time, tau = 0.1 s. States are
   0.1 s averages of a few neurons, and without it the drive jumps every 0.1 s.
3. The heading hold (`LocomotionController.descending_signal`, `[1 + d, 1 - d]`) and
   `b` are combined by `combine_drive` through `LocomotionController.signal_filter`:

   ```
   w     = clip(mean(b), 0, 1)
   final = b + w * d * [+1, -1]         each side clipped to +/-1.5 (DriveGains.max_amp)
   ```

   Reasoning: the brain sets the speed and direction of stepping and its own turn.
   The heading hold is a corrective differential on top of it. For a forward-walking
   brain (`mean(b) >= 1`) this is exactly `hold + (b - [1, 1])`. A quiet brain returns
   the heading hold unchanged, bit for bit, so `--brain-steer` with a silent brain
   walks exactly like no brain. As MDN pulls the stepping toward -1, `w` goes to 0:
   the heading correction is tuned for forward walking, and its sign has no clear
   meaning for reversed stepping. Without `--brain-steer` no filter is installed, and
   the drive is only displayed.

FlyGym's `HybridTurningController` handles negative drive (reversed CPG frequency,
|v| amplitude). Measured on flat ground with a fixed signal: `[1, 1]` gives
+14.6 mm/s, `[0, 0]` stops the fly (0.0 mm/s after 0.2 s), `[-0.5, -0.5]` gives
-3.9 mm/s and `[-1, -1]` gives -11.7 mm/s. It walks backward.

**Looming with the real brain** (O at 200 Hz for 1 s, flat, `--brain-steer`, two
trials, forward thorax velocity from `metrics.csv`):

| window after O | GF (DNp01) | MDN L/R | control drive (mean) | forward velocity mean (min) |
|---|---|---|---|---|
| 1 s before | 0 | 0 | 1.00 | 14.6 / 13.0 mm/s |
| 0-0.5 s | 120-140 Hz | 15-45 Hz | 0.51-0.54 | 8.2 / 7.4 mm/s |
| 0.5-1.2 s | 125-150 Hz | 10-45 Hz | 0.08-0.13 | **0.9 / 1.5 mm/s** (-4.9 / -1.8) |
| 1.2-2.2 s after | 0-20 Hz | 0 | 0.91-0.95 | 12.6 / 13.2 mm/s |

With the default `backward_ref = 40 Hz`, MDN's ~20 Hz average in this model gives a
drive near 0, so **looming stops the fly** (with brief backward steps) for about a
second. With `backward_ref = 20 Hz` the same stimulus makes it **walk backward**:
mean -4.3 and -0.8 mm/s over 0.5-1.2 s in two trials, with peaks of -22.6 and
-15.6 mm/s. The default was kept: the in-vivo MDN rate for full backward walking is
not known, and 40 Hz is the conservative choice.

**`--brain-backup`** switches to the 20 Hz reference. It implies `--brain-steer`, and
the config equivalent is `{"brain": {"backup": true, "backup_ref_hz": 20}}`. It
overrides `gains.backward_ref`. Check (flat, O at t = 3 s, `metrics.csv` vx, one run
each, same seeds):

```bash
.venv/bin/python scripts/run_sim.py --headless --brain-headless --brain-backup \
    --max-seconds 6.5 --terrain flat --script-keys "3:o"
```

| window after O | `--brain-steer` vx mean (min) | control drive | `--brain-backup` vx mean (min) | control drive |
|---|---|---|---|---|
| 1 s before | 14.1 mm/s | +1.00 | 14.1 mm/s | +1.00 |
| 0-0.5 s | 8.0 (-2.7) | +0.51 | 4.4 (-6.8) | +0.19 |
| 0.5-1.2 s | 2.3 (-3.1), net +1.4 mm | +0.13 | **-4.2 (-21.0), net -1.8 mm** | **-0.43** |
| 1.2-2.2 s | 13.4 | +0.96 | 12.9 | +0.95 |

The brain status line during the loom reads `drive L-0.77 R-0.78` with backup and
`L+0.06 R+0.07` without it (GF 135 Hz, MDN 20 Hz in both). So the fly visibly backs
up for about 0.7 s, then walks on. Whip hits, shoves, falls and taste produced no
walk / turn / MDN activity in these runs, so with `--brain-steer` they don't change
the walk. That is the model's honest limitation (see the honesty note above). The
giant fiber's escape command does not steer, but with `--brain-actions` it makes the
fly jump (next section).

### Brain → actions (`--brain-actions`)

`BrainLinkConfig.actions` (the flag also implies `--brain-steer`) installs
`perpetualfly.actions.brain_triggers.BrainActionTriggers`, which checks every new
BrainState. Real brain, flat, headless, full body:

* **O (looming)**: the giant fiber reaches 115–120 Hz in the first 0.1 s state, and
  the fly **jumps 0.12 s after the key** (apex +2.8 to +3.2 mm, ~50 ms airtime, lands
  upright). There is one jump per loom (1.5 s refractory), and MDN then slows the
  walk as before.
* **T (sugar)**: MN9 reaches 55 Hz and the **proboscis extends** from 0.1 s after the
  key until 0.5 s after the stimulus ends.
* **Grooming**: DNg12 is not reached by the JO grooming afferents (`jo_grooming`,
  0 Hz). Head bristles at 200 Hz give 10–13 Hz, a head whip hit gives ≤ 5 Hz, and
  driving DNg12 directly gives 38–46 Hz. So with the default 20 Hz threshold,
  grooming never fires from natural input in this model. Head bristles and head whip
  hits also drive MN9 to ~120 Hz, so strong head input would extend the proboscis.
  Table in docs/ACTIONS.md §4.
* Freeze is not mapped: DNp09 is linked to freezing (Zacarias et al. 2018) but is
  also in our walk group.

### HUD, terminal and logs

* Fly window HUD: `BRAIN t <brain s> lag <s> x<realtime factor> real time (can
  x<compute_rtf>)`, the drive `L/R` with `(steering)` or `(view only)`, GF / MDN / MN9
  rates, and the O/T/K hint. The terminal prints the same line after every stat line,
  plus edge-triggered notes (`giant fiber (DNp01) firing 130 Hz`, `MDN ...`,
  `MN9 ... Hz`). The brain window shows the stimulus chips (LOOM LC4, SUGAR,
  WHIP L 0.25, PAUSED, ...) and the DRIVE panel ("from brain, applied to the fly" or
  "NOT applied").
* Screenshot key I: along with the fly frame, a brain-window frame
  (`shotNNN_..._brain.png`) is rendered *in the app process* with the brain window's
  headless `render_frame`, from the last 30 `BrainState`s (3 s; `BrainLink.recent`).
  This takes ~40-60 ms. It works with `--brain-headless` too. It is a re-render, so
  its spike animation can differ slightly from the live window's.
* `metrics.csv` extra columns: `brain_time, brain_lag, brain_drive_L/R` (smoothed
  brain drive), `ctrl_drive_L/R` (the signal actually given to the controller),
  `dn_walk_L ... dn_escape`, `dn_groom` (Hz, latest state) and `mn9_hz`. `events.csv`:
  `brain_stim`, `brain_reset`, `brain_pause` / `brain_resume`. `summary.json`:
  `brain` (states, dropped, stimuli, brain time, final lag, worker info).

### Lifecycle

The brain worker is spawned before the MuJoCo world is built, so the connectome
loads in parallel (about 3 s to ready). The window starts once the layout has
arrived. Both are started once, survive fly resets, and are stopped in `run()`'s
`finally` (Q / ESC, fly window closed, Ctrl-C, exceptions). Both children ignore
SIGINT, so Ctrl-C is handled by the app only. Both also exit by themselves when the
parent dies: the worker within one wait (50 ms) or chunk, the window within 0.5 s.
This was checked with `ps` after SIGKILL of the app (and in
`tests/test_brain_app.py`). If the brain window is closed or dies, the fly and the
brain keep running and the terminal says so once. If the worker dies, the drive
falls back to `[1, 1]`.

## Commands

```bash
.venv/bin/python -m pip install -e ".[brain]"        # or: uv pip install numba pyarrow scipy
.venv/bin/python scripts/fetch_brain_data.py          # ~153 MB download, checksums, builds neurons_783.npz
.venv/bin/python scripts/demo_brain.py                # headless: whip, sugar, LC4; prints BrainState summaries
.venv/bin/python scripts/demo_brain.py --synthetic    # same without data
.venv/bin/python scripts/bench_brain.py               # engine RTF
.venv/bin/python scripts/bench_brain.py --brian2 --standalone --validate 10   # needs brian2
.venv/bin/python -m pytest -q tests/test_brain.py     # data tests skip without data/brain
.venv/bin/python scripts/run_sim.py --brain             # fly + brain window (keys O / T / K)
.venv/bin/python scripts/run_sim.py --headless --brain-headless --brain-steer \
    --max-seconds 20 --terrain flat --script-keys "2:left,6:o,12:t"
.venv/bin/python -m pytest -q tests/test_brain_app.py # app integration (synthetic brain)
```

## References

Shiu et al. 2024 Nature 634:210 · Dorkenwald et al. 2024 Nature 634:124 ·
Schlegel et al. 2024 Nature 634:139 · Eckstein et al. 2024 Cell 187:2574 ·
Berg et al. 2025 bioRxiv 10.1101/2025.10.09.680999 · Kamikouchi et al. 2009 Nature
458:165 · Bidaye et al. 2014 Science 344:97 · Bidaye et al. 2020 Neuron 108:469 ·
Sapkal et al. 2024 bioRxiv (steering / BDN2, oDN1) · Rayshubskiy et al. 2020 bioRxiv
(DNa02) · Yang et al. 2023 bioRxiv (DNa01/DNa02) · von Reyn et al. 2014 Nat Neurosci
17:962 · Ache et al. 2019 Curr Biol 29:1073 · Namiki et al. 2018 eLife 7:e34272.
