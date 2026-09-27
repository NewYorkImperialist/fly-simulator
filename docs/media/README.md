# README media

Everything here was rendered offscreen: `--record` MP4s, app screenshots (`I` key via
`--script-keys`), or `play.py --record`. No screen captures. Commands were run from the
repo root. `S` is a scratch directory. GIFs were made with ffmpeg's two-pass palette
(`palettegen stats_mode=diff` + `paletteuse`). The PNGs were quantized to 256 colours
with Pillow.

GIF recipe (`START`, `DUR`, `SPEED`, `W`, `FPS`, `N`, `DITHER` per file below):

```bash
ffmpeg -ss START -t DUR -i in.mp4 -vf "setpts=PTS/SPEED,fps=FPS,scale=W:-2:flags=lanczos,split[a][b];\
[a]palettegen=max_colors=N:stats_mode=diff[p];[b][p]paletteuse=dither=DITHER:diff_mode=rectangle" -loop 0 out.gif
```

`--record` frames never contain the HUD, so the app clips are clean.

| file | size | shows |
|---|---|---|
| `hero.gif` | 2.3 MB | Swatter slam: a lazy swat comes from behind. The real FlyWire brain's giant fibre fires, and the fly escapes on flapping wings (no external force) while the paddle lands behind it. 3x slow motion |
| `kebab.gif` | 1.5 MB | The doner kebab chef job: it carves the spit with the recorded grooming stroke |
| `dead_hang.gif` | 0.5 MB | The dead-hang job: the fly hangs from the pull-up bar, the flytrap twitches, the fly re-grips, then its grip is drained and it slides off into the trap: CHOMP, then the respawn |
| `jobs.gif` | 2.5 MB | 2x2 grid of the eternal jobs: Sisyphus, hamster wheel, lawn mowing, leaf raking |
| `whip.gif` | 2.0 MB | A level-3 physical whip crack from the right knocks the fly sideways. 4x slow motion |
| `brain_window.png` | 0.1 MB | The brain window with real FlyWire activity 0.1 s after a looming stimulus (giant fibre 115 Hz, JUMP), after two whip hits under `--stress`. Shows the decision meters, DN traces, pain/arousal and the playground panel |
| `rings.gif` | 2.2 MB | Game "Fly Through Rings": real flapping flight, steered by LC10a → DNa01/02 |
| `rings_brain_panel.png` | 0.2 MB | A full rings frame including the "THE BRAIN" side panel |
| `chase.gif` | 1.8 MB | Game "Follow the Leader": the brain-steered fly pursues a leader fly |
| `course.png` | 0.1 MB | The `gauntlet` obstacle course: gates, a ramp, a pebble field |
| `taste.png` | 0.1 MB | Feeding: the fly stands on a sugar spot and the real MN9 extends its proboscis (close side view) |
| `social_preview.png` | 0.2 MB | 1280×640 GitHub social card: a kebab frame plus the title and tagline |

The compound-eye figure is `docs/real_vision.png` (from `scripts/demo_real_vision.py --figure`).

## Commands

Scratch config files (`--config`; note a `--config` file turns the full body off unless
`--full-body` is passed):

```bash
echo '{"render":{"width":720,"height":480}}' > $S/r720.json
echo '{"render":{"width":480,"height":320}}' > $S/r480.json
echo '{"render":{"width":720,"height":480},"camera":{"distance":13.0,"elevation":-18.0,"follow_azimuth_offset":-155.0,"zoom_per_mm":0.0,"heading_tau_s":3.0},"whip":{"enabled":false}}' > $S/hero.json
echo '{"render":{"width":720,"height":480},"camera":{"distance":15.0,"elevation":-28.0,"follow_azimuth_offset":-60.0,"zoom_per_mm":0.5}}' > $S/whip.json
echo '{"render":{"width":960,"height":540},"camera":{"distance":14.0,"elevation":-28.0,"follow_azimuth_offset":-35.0}}' > $S/course.json
echo '{"render":{"width":720,"height":480},"camera":{"distance":4.2,"elevation":-6.0,"follow_azimuth_offset":-92.0},"whip":{"enabled":false}}' > $S/taste.json
echo '{"render":{"width":1280,"height":640}}' > $S/r1280.json
```

**hero.gif** (the result was DODGED: take-off 2.55 s, landed upright 134 mm away; GIF: start 1.9, dur 1.5, speed 0.333, 540 px, 12 fps, 96 colours, dither none)
```bash
.venv/bin/python scripts/run_sim.py --flight --brain-actions --swatter --brain-headless --strength 1 \
  --terrain flat --script-keys "1.5:v" --max-seconds 4 --record $S/hero.mp4 --no-log --config $S/hero.json
```

**kebab.gif** (start 0.5, dur 4, 540 px, 12 fps, 128 colours, bayer dither scale 4)
```bash
.venv/bin/python scripts/run_sim.py --job kebab --headless --max-seconds 5 --record $S/kebab.mp4 --no-log --config $S/r720.json
```

**dead_hang.gif** (real time, 9 s, 480 px, 12 fps, 128 colours, bayer dither scale 4). The
frames came from a small script that builds `create_job_session("dead_hang", cfg,
{"twitch_every_s": 0, "slip_rate_per_s": 0})` at 600×400 and steps a `JobRunner`. It calls
`job.twitch()` at 1.0 s and `job._start_reach("rf", t, "regrip")` at 2.2 s. At 3.6 s it
sets both strengths and the capacity to 0.1 (a demo shortcut for "tired": the adhesion
command becomes 4 µN per leg, and the fly was off the bar within 0.1 s). It renders a clean frame
every 1/12 s and draws a short HUD (title, streak, grip, banner) with `compose_frame`.
```bash
ffmpeg -framerate 12 -i $S/gif/f%04d.png -vf "scale=480:-2:flags=lanczos,split[a][b];\
[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" -loop 0 dead_hang.gif
```
Run it live: `python scripts/run_job.py --job dead_hang` (or `run_sim.py --job dead_hang`).

**jobs.gif**. For each job, the same command at 480×320 with `--max-seconds 6` (raking 10). For
sisyphus, hamster_wheel and raking the job camera was moved closer, using a small wrapper
that imports the job modules, overrides each `EternalJob` subclass's `camera_preset()`
(and for raking also `camera_target()` = the thorax), then calls `fly_simulator.app.main()`.
Presets (azimuth, elevation, distance): sisyphus 68/-30/17, hamster_wheel 102/-18/16,
raking 100/-28/13 (on the thorax). Mowing uses the stock preset.
```bash
.venv/bin/python scripts/run_sim.py --job mowing --headless --max-seconds 6 --record $S/mowing.mp4 --no-log --config $S/r480.json
```
3.5 s from each clip (sisyphus / wheel / mowing from 2.5 s, raking from 6.5 s) went into
a 642×430 grid, with labels drawn by OpenCV. GIF: 3.2 s, 10 fps, 112 colours, bayer dither.

**whip.gif** (the log gave `HIT c_thorax, impulse 496 nN*s`; GIF: start 0.95, dur 0.9, speed 0.25, 540 px, 12 fps, 96 colours, dither none)
```bash
.venv/bin/python scripts/run_sim.py --terrain flat --strength 3 --script-keys "1.0:right" --max-seconds 2.2 \
  --record $S/whip.mp4 --no-log --config $S/whip.json
```

**brain_window.png** (the `shot002_*_brain.png` of the run)
```bash
.venv/bin/python scripts/run_sim.py --headless --brain-headless --brain-actions --stress --terrain flat --strength 2 \
  --script-keys "3:left,5.5:right,8.0:o,8.1:i,8.17:i" --max-seconds 8.4 --runs-dir $S/runs --config $S/whip.json
```

**rings.gif / rings_brain_panel.png** (GIF: `crop=960:640:0:0`, start 4.4, dur 3.6, 540 px, 12 fps, 128 colours, dither none; the panel PNG is the frame at 5.9 s)
```bash
.venv/bin/python scripts/play.py --game rings --brain --record $S/rings.mp4 --max-seconds 8 --no-highscore
```

**chase.gif** (`crop=960:640:0:0`, start 5.2, dur 3.8, 480 px, 12 fps, 96 colours, dither none)
```bash
.venv/bin/python scripts/play.py --game chase --brain --record $S/chase.mp4 --max-seconds 10 --no-highscore
```

**course.png** (frame at 1.45 s, resized to 720×408)
```bash
.venv/bin/python scripts/run_sim.py --course gauntlet --headless --max-seconds 6 --record $S/course.mp4 --no-log --config $S/course.json
```

**taste.png** (frame at 2.2 s, cropped to `[140:480, 100:720]`, caption drawn by OpenCV)
```bash
.venv/bin/python scripts/run_sim.py --full-body --headless --brain-headless --brain-actions --taste-patches \
  --taste-density 0 --terrain flat --script-keys "0.5:5" --max-seconds 2.4 --record $S/taste.mp4 --no-log --config $S/taste.json
```

**social_preview.png**: the kebab frame at 2.2 s of the command below, columns 150–990,
blended into a 500 px dark left panel. The title, tagline and credits were drawn with
Pillow in Avenir Next (Bold / Medium), a macOS system font.
```bash
.venv/bin/python scripts/run_sim.py --job kebab --headless --max-seconds 3 --record $S/kb.mp4 --no-log --config $S/r1280.json
```
