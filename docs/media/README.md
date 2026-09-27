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
| `bowling.gif` | 2.1 MB | The bowling job: the fly pushes the ball into the ramp guide and over the ramp, the camera rides along down the lane and cuts to the deck, a strike (real contacts, labelled slow motion x0.25), the pinsetter sweeps and lowers a fresh rack, the cut back to the fly. Recorded with the HUD |
| `broccoli_toss.gif` | 2.3 MB | The broccoli toss job: the host fly brings a plate of broccoli, the posed viewer fly in the gaming chair ponders it, snaps it over its shoulder without looking, and the room behind it explodes, with the meme's edit effects (impact frame, flash, hit-stop and slow motion, shake, punch-in, bloom); the props fly, the room rebuilds. Compact HUD, `EDIT FX (not physics)` tag during the edit beat |
| `taste_tester.gif` | 1.2 MB | The taste tester job with the real FlyWire brain: a sugar drop on the leg -> MN9 at ~72 Hz on the meter, the proboscis extends, APPROVED (green lamp, tally); a mixed (sugar + bitter) drop -> MN9 19 Hz, REJECTED, the leg pushes the dish away; the stamper stamps the cards and the diverter sweeps samples into the green / red bins. No HUD |
| `pizza_chef.gif` | 2.4 MB | The pizza chef job, one pizza: the dough ball kneaded flat, the toss (a spinning free body, labelled slow motion x0.1), sauce, cheese / pepperoni / basil raining from the bowls (slow motion x0.3), into the brick oven on the peel, baked, sliced by the cutter wheel, boxed and served; the chalkboard counts. No HUD |
| `trampoline.gif` | 2.1 MB | The trampoline job: the fly stands on the spring mat, the first jump, the rebound jumps pumping the height up beside the ruler (red / gold markers, NEW BEST), a backflip (a real boosted asymmetric push) landed at the mat's edge, the next bounce sends it off onto the lawn, the counted respawn, and it pumps up again. Labelled slow motion x0.2, a one-line HUD |
| `delivery_pilot.gif` | 2.1 MB | The delivery pilot job on the real flight fly: the parcel slides onto the fly on the depot roof, the jump → wings take-off, the flight to house No. 1 with the parcel hanging under it, the landing on the roof terrace, DELIVERED on the doormat, the take-off and the flight back to the depot's loading mark. Real time, no HUD (the captions are the job's overlay) |
| `jobs.gif` | 2.3 MB | 2x2 grid of the eternal jobs: Sisyphus (Greek hillside at sunset), hamster wheel (pet cage), lawn mowing (front yard), leaf raking (autumn backyard) |
| `whip.gif` | 2.0 MB | A level-3 physical whip crack from the right knocks the fly sideways. 4x slow motion |
| `brain_window.png` | 0.1 MB | The brain window with real FlyWire activity 0.1 s after a looming stimulus (giant fibre 115 Hz, JUMP), after two whip hits under `--stress`. Shows the decision meters, DN traces, pain/arousal and the playground panel |
| `rings.gif` | 2.2 MB | Game "Fly Through Rings": real flapping flight, steered by LC10a → DNa01/02 |
| `rings_brain_panel.png` | 0.2 MB | A full rings frame including the "THE BRAIN" side panel |
| `chase.gif` | 1.8 MB | Game "Follow the Leader": the brain-steered fly pursues a leader fly |
| `pong.gif` | 2.0 MB | Game "Fly Pong": the real FlyWire brain moves the green paddle the fly stands on (ball → LC10a → DNa01/02 → paddle speed) against a scripted AI; returns as the ball speeds up, then a miss ("MISSED! AI SCORES") and the next serve |
| `canyon.gif` | 2.1 MB | Game "Canyon Run": real flapping flight through sandstone pillars; each pillar looms on the eye that sees it (LC4 / LPLC2 → the opposite DNa01/02), the real FlyWire brain turns away: a near miss (yellow pillar), dodges, then a nearly head-on pillar (both eyes loom equally) is hit: CRASH!, relaunch |
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

**broccoli_toss.gif** (one cycle, 166 frames, 15 fps, 420 px, 96 colours, bayer dither scale 5).
The frames came from a small script that builds `create_job_session("broccoli_toss", cfg,
{"seed": 0})` at 600×400 and steps a `JobRunner(chunk_steps=50)`. It renders a frame (the job
renderer, so with the job's post-process) every 1/15 s of **presentation time**
(`runner.present_time()`: the hit-stop and slow motion are in the frames, no extra slow-motion
pass), draws a 3-line HUD with `compose_frame` (title, plates / explosions / viewers, vegetables
eaten + the job message) plus `EDIT FX (not physics): hit-stop / slow-mo / flash / shake` while
`job.edit_beat()`, and stops 0.8 s into the host's walk back.
```bash
ffmpeg -framerate 15 -i $S/gif/f%04d.png -vf "scale=420:-2:flags=lanczos,split[a][b];\
[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" -loop 0 broccoli_toss.gif
```
Run it live: `python scripts/run_job.py --job broccoli_toss` (or `run_sim.py --job broccoli_toss`).

**taste_tester.gif** (real FlyWire brain, 6.8-16.2 s of the run, 640 px, 12 fps, 160 colours,
bayer dither scale 4). The job is deterministic with the default seed, so this run shows samples
#2 SUGAR (170 Hz: MN9 72 Hz, APPROVED), #1 MIXED (125 + 100 Hz: MN9 19 Hz, REJECTED) and #5 SUGAR
(105 Hz: MN9 65 Hz, APPROVED).
```bash
.venv/bin/python scripts/run_sim.py --job taste_tester --brain-headless --headless --max-seconds 16.5 \
  --record $S/tt.mp4 --no-log --config $S/r720.json
ffmpeg -ss 6.8 -t 9.4 -i $S/tt.mp4 -vf "fps=12,scale=640:-2:flags=lanczos,split[a][b];\
[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" -loop 0 taste_tester.gif
```
Run it live: `python scripts/run_job.py --job taste_tester --brain` (or `run_sim.py --job taste_tester`).

**pizza_chef.gif** (default seed, the first pizza: 0-31 s of the run, no brain). Clean frames
(`JobRunner.render(hud=False)`, so the job's slow-motion label is in them) at 420×280, one every
1/15 s of presentation time (`session.present_time()`, so the slow motion is in it), a montage by
frame index: the kneading at 2× then every other frame, the first toss at 1×, sauce / pours / oven /
bake / slicing / serving sped up 2-6× and every other frame (137 frames), then:
```bash
ffmpeg -framerate 8 -i sel/s%04d.png -vf "scale=380:-2:flags=lanczos,split[a][b];\
[a]palettegen=max_colors=56:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" -loop 0 pizza_chef.gif
```
(Every frame differs: the camera glides and the fire flickers, so the size scales with the frame
count; 400 px / 96 colours was 3.2 MB.) Run it live: `python scripts/run_job.py --job pizza_chef`
(or `run_sim.py --job pizza_chef`).

**trampoline.gif** (default config, 211 frames = 14 s of presentation time, 4.1 s of sim).
A small script builds `create_job_session("trampoline", cfg)` at 600×400 and steps a
`JobRunner(chunk_steps=50)`; at 0.95 s of run time it opens the trick window (`trick_p` 1, no
height / streak limits) until one backflip has been tried (a demo shortcut to get a flip into the
clip; this one landed, 335°). It renders a clean frame (`render(hud=False)`, so the slow-motion
label is in it) every 1/15 s of `session.present_time()` and draws a one-line HUD (bounces, best,
streak, the banner) with `compose_frame`, then:
```bash
ffmpeg -framerate 15 -i gif/f%04d.png -vf "scale=480:-2:flags=lanczos,split[a][b];\
[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" -loop 0 trampoline.gif
```
Run it live: `python scripts/run_job.py --job trampoline` (or `run_sim.py --job trampoline`).

**jobs.gif** (the jobs' own default job cameras, no preset overrides). For each job a
small script builds `create_job_session(job, cfg)` at 480×320 (whip off) and steps a
`JobRunner(chunk_steps=50)`; from 2.5 s of run time (raking 6.5 s) it renders a clean
frame (`render(hud=False)`) every 0.1 s, 36 frames. Each frame is resized to 320×213
(`INTER_AREA`), labelled with OpenCV (Hershey simplex 0.45 on a dark tab) and put into a
642×428 grid (2 px gaps), then:
```bash
ffmpeg -framerate 10 -i grid/%04d.png -vf "trim=end=3.2,split[a][b];\
[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" -loop 0 jobs.gif
```

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

**pong.gif** (start 6.0, dur 5.6, 540 px, 12 fps, 128 colours, dither none, speed 1)
```bash
.venv/bin/python scripts/play.py --game pong --brain --record $S/pong.mp4 --max-seconds 24 --seed 2 --no-highscore
```

**canyon.gif** (start 0.6, dur 4.2, 480 px, 10 fps, 96 colours, dither none, speed 1)
```bash
.venv/bin/python scripts/play.py --game canyon --brain --record $S/canyon.mp4 --max-seconds 16 --no-highscore
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

**bowling.gif** (the first 9 s of the recording, 480 px, 10 fps, 96 colours, no dither; the
default seed's first ball is a strike). `run_job.py --record` frames include the HUD
(scoreboard); the recording is paced by presentation time, so the job's labelled slow
motion (x0.25 during the roll and the first pin action) is in it:
```bash
.venv/bin/python scripts/run_job.py --job bowling --headless --max-seconds 9 --record $S/bowl --segment-s 60 --keep 1 \
  --width 720 --height 480
ffmpeg -t 9 -i $S/bowl/recording01_t00000.02s.mp4 -vf "fps=10,scale=480:-2:flags=lanczos,split[a][b];\
[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" -loop 0 bowling.gif
```

**delivery_pilot.gif** (default config, wind on; clean job-camera frames, no HUD, rendered with
`JobRunner(session, job, headless=True, chunk_steps=50).render(hud=False)` every 1/15 s of run time
from 0.2 to 9.2 s at 600×400 into `$S/rec/%04d.png`; GIF: the first 7.4 s, 480 px, 10 fps, 64
colours, no dither):
```bash
ffmpeg -framerate 15 -i $S/rec/%04d.png -vf "trim=end=7.4,fps=10,scale=480:-2:flags=lanczos,split[a][b];\
[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" -loop 0 delivery_pilot.gif
```
