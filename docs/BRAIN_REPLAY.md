# Brain replay: record a run's brain activity and watch it later

`--brain-record` saves the connectome brain's `BrainState`s together with the key
events (stimuli sent to the brain, brain-triggered actions, resets) into the run
directory. `scripts/brain_replay.py RUN_DIR` plays them back in the brain window,
with play / pause, speed control, seeking and a timeline bar. It can also render a
time range headlessly to PNG or MP4. Recording is off by default and implies
`--brain`.

Code: `perpetualfly/brain_viz/replay.py` (`BrainRecorder`, `BrainRecording`,
`ReplayPlayer`, `render_png`, `render_range`, `run_viewer`), the hooks in
`perpetualfly/brain_link.py`, and `tests/test_brain_replay.py`.

## Recording format (`<run dir>/brain_rec/`)

| file | content |
|---|---|
| `layout.npz` | the static `BrainLayout` (regions, outlines, display neurons), written once, ~54 KB for the real brain |
| `chunk_00000.npz`, ... | `np.savez_compressed` chunks of 100 stored states (10 s at 0.1 s) |
| `events.jsonl` | one JSON line per event, appended live: `{"kind": "stim" \| "action", "t": fly run time, "label", ...}` |
| `meta.json` | format `perpetualfly-brain-rec/1`, counts, bytes, time range, whether the size cap was hit |

Per stored state:

* **Downsampling.** Incoming states are merged until they cover at least
  `record_window_s` (0.1 s) of brain time. The low-latency pacing of the swatter
  and whip vision publishes 0.02 s states, so five are merged into one. Rates are
  window-weighted means and spike totals are summed.
* **Descending readouts.** The descending group means are stored, and so is their
  peak over the merged states (`desc_peak`), so a short GF burst still shows.
* **Raster.** Display-neuron spikes are stored as int16 indices plus uint16
  offsets in 0.1 ms steps. At most `raster_cap` (1500) spikes are kept per stored
  state (random subset; `meta.raster_dropped` counts the rest).
* **Numeric and JSON fields.** Region rates are float16, and the other arrays are
  float32. The small dicts (`drive`, `neuromod`, `habituation`, `playground`,
  recent stimulus labels) are stored as one JSON string per state.
* **Bounded size.** Once the files exceed `record_max_mb` (50 MB) the recorder
  stops, prints one message and marks `meta.stopped_at_limit`. A run can never
  fill the disk.

**Size measured** (real brain, headless, `--brain-actions`):

| run | length | states | size | MB / min |
|---|---|---|---|---|
| 10 looms, mostly walking (habituation run) | 47.4 s | 474 | 0.42 MB | **0.54** |
| 8 looms every 2 s, no habituation (dense GF/MDN activity) | 17.4 s | 174 | 0.32 MB | **1.11** (includes the fixed 54 KB layout) |

A chunk of 10 s with looms is ~134 KB, and a quiet 10 s chunk is ~11-45 KB. At
0.5-1 MB/min the 50 MB cap covers roughly 50-100 min.

## Replay viewer

```bash
.venv/bin/python scripts/brain_replay.py runs/<ts>             # opens the replay window
.venv/bin/python scripts/brain_replay.py runs/<ts> --info      # summary + event list
.venv/bin/python scripts/brain_replay.py runs/<ts> --png 12.4 --out frame.png
.venv/bin/python scripts/brain_replay.py runs/<ts> --t0 45.8 --t1 46.5 --out clip.mp4
.venv/bin/python scripts/brain_replay.py runs/<ts> --t0 2 --t1 3 --out frames/   # PNG sequence
```

Options: `--speed`, `--fps`, `--start T`, `--size WxH` and `--no-playground`
(classic raster / transmitter panels). Times are fly run times, the same clock as
`events.csv` and `BrainState.sim_time`.

The window is the normal brain window (`BrainRenderer`) fed from the file on a
virtual clock, with a 74 px timeline bar below it:

* **Keys.** **space** plays / pauses, **← / →** seek 2 s, **[ / ]** change the
  speed (0.1× ... 8×), **Home** restarts, **q / Esc** quit. A click on the bar
  seeks to that point.
* **Timeline bar.** It spans the whole recording. The red trace is the giant-fibre
  (DNp01) peak rate, with a grey line at the 60 Hz jump threshold. Stimuli show as
  a coloured tick plus a bar for their duration: orange for looms, red for whip /
  shove, green for sugar, purple for bitter, grey for resets. Brain actions (jump,
  proboscis, groom) are cyan triangles. The white line is the playhead.
* **Status line.** It shows `REPLAY t <now> / <end> x<speed> PLAYING|PAUSED` and,
  with `--habituation`, the current GF-input efficacy.
* **Seeking.** A seek builds a fresh renderer and pre-feeds the 3 s before the
  target, so the traces and raster look as they did live. Recorded events are
  re-injected as stimulus chips (`ACTION JUMP` for actions).

**Frames checked** (rendered headless and viewed, habituation run): t = 2.4 s
(loom 1: GF 100 Hz, JUMP meter, looms + jumps on the timeline); t = 12.4 s
(habituated: GF 0 Hz, header `habituation (model): LC4 0.09 · LPLC2 1.00`, MDN
still active); the 45.8-46.5 s clip at 960x600 (the post-rest loom that jumps
again); and the header at both sizes. At 960 px there is no room for the
habituation text in the brain header, so it is left out there. The replay status
line and the HUD still show it.

## Limitations

* This is a replay of what was *published*, downsampled to 0.1 s. Spike timing
  within the raster is exact to 0.1 ms for the stored display-neuron spikes, but
  rates are 0.1 s averages. The fast-path crossings themselves are not stored,
  only the resulting actions.
* Only the brain is replayed, not the fly's body. Use `--record` / the M key for the
  fly video.
* The GUI viewer's arrow-key codes are those of OpenCV's `waitKeyEx` on macOS
  Cocoa / GTK / Windows. The viewer was checked through its headless paths and
  unit tests (keys via `ReplayPlayer.handle_key`); the GUI window itself was not
  opened in this work.
* A crash loses at most the states of the unwritten chunk (≤ 10 s). `events.jsonl`
  is flushed per line.
