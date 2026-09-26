# FLY BRAIN PLAYS: games controlled by the connectome brain

> **Brain responses are real connectome wiring; the game interface (what the brain
> sees, how its outputs map to controls) is designed by us.**
>
> This label is shown in the game HUD and in the brain panel. The brain is the Shiu et al.
> 2024 LIF model of the whole FlyWire v783 central brain (138,639 neurons, 15 M
> connections; docs/BRAIN.md). Nothing in it was trained or tuned for the game. We
> chose the input side (which neurons the rocks drive, and at what rates) and the
> output side (how descending-neuron rates move the body).

Files: `perpetualfly/games/` (`asteroids.py` rocks + game rules, `vision.py` the eyes,
`brain_io.py` brain worker + mapping, `session.py` wiring, `hud.py`, `runner.py`,
`experiment.py`), `scripts/play.py`, `tests/test_games.py`. Nothing in the app
(`app.py`, `config.py`) was changed. The game builds its own `Simulation`.

## Game 1: ASTEROID DODGE

### Embodiment (option A: the real fly in MuJoCo)

The NeuroMechFly walks on flat ground using the hybrid CPG controller, at the usual
~14 mm/s. Fly-scale boulders roll at it. Each boulder is a sphere of radius 1.3–1.9 mm
(the fly is 2.5 mm long), grey-brown with visual crust bumps and craters.

* **Spawning.** Rocks appear 26 mm ahead of the fly, at a random bearing within
  ±12° of its current heading. Each rock is aimed at where the fly will be when it
  arrives: the fly's forward velocity × travel time, plus a random lateral offset.
* **Rocks are mocap bodies.** They follow exact straight-line kinematics and roll
  visually (the rotation matches the distance travelled). They collide only with the fly
  (`conaffinity = FLY_BIT`), using FlyGym's contact parameters.
* **Hit detection.** A hit is any MuJoCo contact between a live rock and a fly geom.
  The rock turns red and keeps colliding for 30 ms, so the fly is physically bumped. It
  then becomes a ghost and rolls on.
* **Dodges.** A rock whose centre gets more than `r + 2.5 mm` past the fly, measured
  along its path, without touching it counts as dodged.
* **Flipped fly.** A fly lying on its back or side (tilt > 100°) for 1.2 s is respawned.
  The brain is reset with it.

Option B (a 2D arcade craft) was not needed.

**Game rules** (`AsteroidConfig`, `wave_params`):

* **Waves.** Wave k has 4 + 2(k−1) rocks, speed 10 + 1.5(k−1) mm/s (capped at
  22 mm/s), and a spawn interval of 2.2 − 0.15(k−1) s (at least 0.9 s, ±20 %).
* **Difficulty.** easy / normal / hard scales rock speed (×0.8 / 1 / 1.3) and the spawn
  interval (×1.25 / 1 / 0.75), and sets the maximum aim offset (4.0 / 3.5 / 3.0 mm; a rock whose path passes within about 3.5 mm of the thorax hits the body or legs).
* **Lives and score.** 3 lives. Each dodge scores 100 × the wave number, plus 10 points
  per second survived.
* **High scores.** Kept per control mode and difficulty (top 5 each) in
  `runs/games_highscores.json`, which stays under 4 KB.

### What the brain sees (input interface, ours)

Each rock is a visual looming source for the two compound eyes. The implementation is
`perpetualfly.vision.looming.LoomingVision`, reused: the same field of view, the
`loom` stimulus kind, event scheduling and brain mapping. Per eye, the rock's angular
size θ and expansion rate θ̇ drive **LC4** and **LPLC2**, the looming-detector visual
projection neurons of that eye's optic lobe:

* 54 LC4 and 108 LPLC2 on the left, 50 LC4 and 102 LPLC2 on the right (FlyWire
  `side`).
* Each set gets Poisson input through `StimulusEvent("loom", side=eye,
  details={lc4_hz, lplc2_hz})`.

We chose these neurons because they are the looming channel (von Reyn et al. 2017; Ache
et al. 2019). In the connectome, one eye's LC4 / LPLC2 drive the **contralateral**
DNa01/DNa02 steering pair. That is the turn-away response the game relies on. Direct
stimulation of the whole model (`perpetualfly.brain.process._Model`, 0.3 s):

| stimulus (per neuron Poisson) | turn_L | turn_R | walk L/R | MDN L/R | giant fibre |
|---|---|---|---|---|---|
| LC4+LPLC2 left, 40 Hz | 0 | 18 | 1/0 | 0/0 | 80 |
| LC4+LPLC2 left, 80 Hz | 0 | 25 | 4/0 | 0/0 | 110 |
| LC4+LPLC2 right, 80 Hz | 37 | 0 | 1/8 | 3/5 | 125 |
| LC4+LPLC2 both, 80 Hz | 0 | 0 | 1/1 | 3/5 | 147 |
| left 150 Hz + right 50 Hz | 0 | 13–15 | 4–8/1 | 0–2/0–3 | 158–165 |

(Group means in Hz. Turn = DNa01 + DNa02 of that side. The sensory screen gives the
same picture: LC4, LPLC2, LC6 and LC17 turn contralaterally, docs/SENSORY_SCREEN.md.)

**Neurons deliberately not driven.** LC10a/c/d and LLPC1/2 are the strongest turn
drivers in the sensory screen (DNa02 ~140 Hz), but they steer *ipsilaterally*, toward
the stimulus. LC10 is the object-tracking pathway used in courtship pursuit. Driving
them with rocks would make the fly chase the asteroids, and choosing them would amount
to choosing the game's outcome. We use only the looming channel.

**Three interface choices**, each made because of a measured artefact:

1. **Response function** (`asteroid_response`). The whip's defaults respond only above
   12,000 deg/s and the swatter's above 300 deg/s. A 1.5 mm rock closing at ~25 mm/s
   expands at only ~30 deg/s when it is 12 mm away, about 0.4 s before impact. That is
   the last moment a walking fly can still turn out of its path. The game therefore uses:

   * LC4 = 200 tanh(max(0, θ̇ − 12)/150) Hz, for θ ≥ 7°;
   * LPLC2 = 200·ramp(θ; 10°, 45°)·ramp(θ̇; 8, 60 deg/s);
   * a 20 ms photoreceptor/EMD low-pass on θ.

   These are plausible tunings, silent for small, static or receding objects, but they
   are not fits to recordings.
2. **Gaze stabilisation** (`GameVision.eyes`). The model's eyes are rigid on the
   thorax, and the tripod gait yaws the thorax by about ±7° at 12 Hz. The eye frames
   follow the body yaw low-passed at 80 ms and stay level. Real flies stabilise gaze with
   head movements.
3. **Expansion measured on the object, gated by visibility** (`GameVision.update`).
   The base class takes the size from the field-of-view-masked solid angle. A rock
   sliding across the soft edge of the frontal binocular zone therefore "expanded" and
   "contracted" with every step. The contralateral eye got up to ~200 Hz of spurious LC4
   drive, which would steer the fly *toward* the rock. Now each eye computes the rock's
   true θ = 2 asin(r/d), applies the response, and multiplies both rates by the eye's
   field-of-view weight in the rock's direction.

   The field of view itself is the default one: 15° frontal binocular overlap with
   8° soft edges, and a rear blind sector. A rock straight ahead drives both eyes
   equally, and in the connectome the two turn signals then cancel. Such a rock can
   only be dodged by luck, or with a jump.

### How the brain's outputs move the body (output interface, ours)

`GameMapping` → `perpetualfly.brain.mapping.descending_to_drive` (the app's mapping)
with game gains:

| DN group (FlyWire types) | control | mapping |
|---|---|---|
| turn_L / turn_R (DNa01, DNa02) | steer | drive = [1 − d, 1 + d], d = 0.6·(tanh(L/25 Hz) − tanh(R/25 Hz)); the CPG turns toward the smaller amplitude, so left DN activity turns left. Measured on the controller: d = 0.2 / 0.4 / 0.6 / 0.8 → 61 / 154 / 235 / 323 deg/s. A one-eyed loom (contralateral turn group at ~25 Hz) gives d ≈ 0.46, about 180 deg/s. |
| walk_L/R (BDN2 = DNg100, oDN1 = DNg97, P9 = DNp09) | speed | +0.25 amplitude at saturation |
| backward_L/R (MDN) | brake / back up | blends stepping toward −1 (fully reversed at 40 Hz) |
| escape (giant fibre DNp01) | jump (opt-in, `--jump`) | GF > 60 Hz → `Jump` (long mode by default), 1.5 s refractory |

* **Drive smoothing.** The drive is low-passed with τ = 60 ms. It replaces the
  controller's heading hold (`heading_gain = 0`), so turns persist, and new rocks spawn
  relative to the new heading.
* **Brain pacing.** The brain runs in its own process, paced to the fly's simulated
  time (`BrainConfig(pace="sim")`) with 20 ms state windows. While a loom is active, the
  game waits up to 50 ms of wall time for the newest window (the swatter's low-latency
  scheme, docs/SWATTER.md).

**Why jump is off by default.** The giant fibre fires at 100–225 Hz for nearly every
rock, because any looming at 40 Hz already drives it to 80 Hz. With jumps on, the fly
hops at every rock. The long-mode hop (100 ms crouch, then ~0.3 s airborne and
landing) interrupts steering, and it sometimes tumbles the fly or throws it far. In a
4-trial pilot with jumps on, every brain trial and every mirror trial jumped. Outcomes
became dominated by the hops (lateral excursions up to 26 mm, and one respawn), not by
steering. The GF rate is always shown in the brain panel; `--jump` turns the jump on.

### Scientific check: does the brain steer away more than chance?

Headless, real brain, one brain process for all conditions (`perpetualfly/games/experiment.py`):

```bash
.venv/bin/python scripts/play.py --game asteroids --brain --experiment 40 --seed 1 --json runs/asteroids_exp.json
```

**Design.** The trials are paired. For trial i, the rock's parameters are drawn once
from the seed and replayed under all three conditions, interleaved:

* **offset**: |offset| uniform in 0.5–3.0 mm, alternating sides;
* **speed**: 10 mm/s (closing speed ~24 mm/s with the fly walking);
* **radius**: one of the four slot sizes;
* **bearing**: ±6°;
* **gait phase**: the walk-in lasts 0.8 s + U(0, 83 ms).

Each trial resets the fly and the brain (all neurons at rest), walks, spawns one
rock, and runs until the rock hits or passes. Jumps were off.

The three conditions:

* `brain`: correct eye → optic lobe mapping;
* `mirror`: left eye → right LC4 / LPLC2 and vice versa, with the same brain and motor
  mapping;
* `none`: brain disconnected, constant drive [1, 1]. This is the chance / geometry
  baseline.

40 trials × 3 conditions took 417 s wall.

| condition | trials | dodged | dodge rate | turned away from the eye that saw more | mean heading change away from it | mean lateral escape (mm) | mean closest approach (mm) |
|---|---|---|---|---|---|---|---|
| **brain** | 40 | **13** | **0.33 ± 0.07** | **34 / 36** (p = 2e-8) | **+29.7°** | +1.20 | 2.89 |
| mirror | 40 | 3 | 0.07 ± 0.04 | 1 / 24 (p = 3e-6), i.e. toward it | −23.0° | −0.66 | 2.09 |
| none | 40 | 6 | 0.15 ± 0.06 | 19 / 31 (p = 0.28) | +0.3° | −0.19 | 2.36 |

The third column counts trials in which one eye's peak LC4 / LPLC2 drive was at least
1.25× the other's. p values are two-sided sign tests. Lateral escape is measured away
from the rock's side, relative to the fly's heading at spawn. The ± values are binomial
standard errors.

| paired comparison (McNemar, exact) | dodged only in first | dodged only in second | p |
|---|---|---|---|
| brain vs none | 7 | 0 | **0.016** |
| brain vs mirror | 10 | 0 | **0.002** |
| mirror vs none | 0 | 3 | 0.25 |

**Reading.**

* **The brain steers away.** With the correct wiring, the fly turned away from the eye
  that saw the rock in 34 of 36 lateralised trials. The peak contralateral DNa01/02
  group rate was ~60 Hz, and the giant fibre ran at ~180 Hz.
* **It dodges more than chance.** It dodged twice as often as the disconnected
  baseline (33 % vs 15 %). Every pair that differed went its way: 7–0 against `none`,
  10–0 against `mirror`.
* **Mirroring flips the turn.** With the eyes mirrored, the same brain turned toward the
  rock in 23 of 24 lateralised trials and dodged least (7.5 %).
* **So the steering direction comes from the wiring.** It is the contralateral LC4 /
  LPLC2 → DNa01/02 pathway, not something the interface imposes. The interface's gains
  and thresholds decide *how much* it helps. With this offset range most rocks are
  hard: many pass within 1.5 mm of the fly's axis, where they are nearly binocular
  until the last ~0.3 s.

Measured by offset:

| offset range | brain | mirror | none |
|---|---|---|---|
| \|offset\| ≥ 1.75 mm | 6/12 | 2/12 | 3/12 |
| \|offset\| < 1.75 mm | 7/28 | 1/28 | 3/28 |

**Pilots** (6 and 4 paired trials, |offset| 0.8–4 mm): brain 5/6 dodged, mirror 3/6,
none 4/6; the brain turned away every time (+40° on average) and the mirror toward
(−29°). With jumps on (4 trials), every brain and mirror trial jumped, and hops
dominated the outcomes (see above).

**Run-to-run variability.** The brain's Poisson input and the wall-clock-dependent sync
wait make games non-deterministic. The same seed (`--seed 3`, normal) gave 1 dodge and
3 hits in one run, and 3 dodges and 0 hits in 7 s in another. The experiment's
statistics are over trials, not over repeated seeds.

### HUD, window, controls

* **Game view.** The follow camera sits behind the fly (24 mm, 24° down), with the
  rocks coming from ahead.
* **HUD.** Top: the title and control mode (`CONTROLLER: FLYWIRE BRAIN` / `BRAIN, EYES
  MIRRORED (control)` / `NONE, constant walk (control)`), score, best score, lives
  (hearts), wave, difficulty, time, dodges and hits. Centre: events (DODGE +100, HIT!,
  WAVE n, JUMP), GET READY, GAME OVER with the restart hint, PAUSED. Bottom: the honest
  label and the key help.
* **Brain panel** (right, 400 px):
  * **SEES**: LC4 / LPLC2 Hz per eye, with the optic lobe each eye drives (swapped in
    mirror mode).
  * **DOES**: DNa01/02 L and R, walk, MDN and giant fibre rates, with the control each
    one maps to.
  * the resulting CPG drive and turn direction, plus the brain lag and real-time factor.
* **Keys (window).** SPACE pause (the brain is paced to the fly, so it pauses too),
  R restart, 1 / 2 / 3 easy / normal / hard (restarts), B open or close the full brain
  activity window (`brain_viz`, its own process), Q / ESC quit.

### Commands

```bash
# play (window; brain panel on the right)
.venv/bin/python scripts/play.py --game asteroids --brain --window
.venv/bin/python scripts/play.py --game asteroids --brain --window --brain-window --difficulty easy
# headless / recording / frames
.venv/bin/python scripts/play.py --game asteroids --brain --max-seconds 60
.venv/bin/python scripts/play.py --game asteroids --brain --record runs/asteroids.mp4
.venv/bin/python scripts/play.py --game asteroids --brain --frames runs/frames --frame-times 2,4,6
# controls
.venv/bin/python scripts/play.py --game asteroids --brain --control mirror
.venv/bin/python scripts/play.py --game asteroids --brain --control none
# the experiment (one brain process serves all three conditions)
.venv/bin/python scripts/play.py --game asteroids --brain --experiment 40 --seed 1 --json runs/asteroids_exp.json
# tests / no data
.venv/bin/python scripts/play.py --game asteroids --synthetic-brain --max-seconds 5
.venv/bin/python -m pytest -q tests/test_games.py
```

### Install / run API (for app integration later)

```python
from perpetualfly.games import AsteroidSession, GameBrain, AsteroidConfig
brain = GameBrain("brain").start()          # "mirror" | "none"; synthetic={"n": 60} for tests
s = AsteroidSession(brain, AsteroidConfig(difficulty="normal"), seed=0)
while s.game.state != "gameover":
    s.step()                                 # 5 ms physics chunk + brain + game logic
print(s.game.summary()); brain.close()
```

* `s.game.listeners` gets `GameEvent`s: spawn, hit, dodge, wave, jump, respawn,
  gameover, restart.
* `perpetualfly.games.hud.compose(frame_rgb, s)` draws the HUD and panel.
* `perpetualfly.games.runner.GameRunner` runs the headless, window and record loops.

The pieces also work on their own:

* `AsteroidField(cfg).extension` is a world extension: pass it before `add_fly`.
* `field.attach(sim)` moves the rocks and detects hits.
* `field.visual_sources()` returns looming sources for any `LoomingVision`.

To wire the game into the app, the app would build a `Session` with
`world_extensions=[field.extension]` and attach the field and a `GameVision` to
`session.sim`. It would send loom events through its `BrainLink`, use `GameMapping` in
place of the `--brain-steer` gains, and set `controller.heading_gain = 0`. That is not
done here: app.py and config.py belong to the integration work.

### Limitations

* **The interface is ours.** The response function, gaze stabilisation, visibility
  gating, turn gain and drive smoothing were chosen to make a walking fly able to
  dodge. With the app's default gains (`turn_gain` 0.4, 30 Hz) the same 25 Hz turn signal
  gives d ≈ 0.27, about 95 deg/s, roughly half the turn rate. The
  *direction* of the turn and which neurons carry it come from the connectome alone.
  The mirror control shows that the direction matters.
* **Only the looming channel is driven.** The fly has no motion vision, no optic flow
  from its own walking, and no object-tracking input.
* **Head-on rocks are symmetric.** Both eyes are driven equally, so the brain produces
  no net turn (only a giant-fibre escape).
* **Mocap rocks cannot be pushed.** They are unstoppable, hit or not. Contact bumps the
  fly for 30 ms, and the rock then ghosts. There is no damage model beyond the lost
  life.
* **Jumps are off by default** (see above). Escape flight (docs/SWATTER.md) is not used.
* **Speed.** Physics runs at ~0.66× real time on an M1. With the brain, the game plays
  at about 0.3–0.5× real time (the brain runs at ~0.8–1× real time while it is looming).
  The window shows game time, not wall time.
* **One experiment.** Results are from one brain seed, 40 paired trials and a single
  rock speed. Dodge rates depend strongly on the offset range and speed chosen. The
  synthetic brain (tests) has no LC4 / LPLC2 or real wiring, so it cannot play.

### Frames checked

These are renderer PNGs, read back from `--frames` and from frames of a `--record`
clip. The clip was 7 s at 720 × 480 plus the panel, 0.9 MB. It was deleted after
checking.

* **Wave 1 start.** A lumpy grey-brown boulder with craters rolls toward the walking
  fly, and its floor reflection is visible. The panel shows the first weak looming on
  both eyes (LC4 20 Hz, LPLC2 10 Hz) and the giant fibre starting (25 Hz).
* **Rock on the right** (t = 2.0 s, later dodged). Right eye: LC4 51 Hz, LPLC2 149 Hz.
  Brain: turn L DNa01/02 25–75 Hz, turn R 0 Hz, GF 150 Hz. Drive L +0.50 R +1.25
  ("<- turning left"). In the image the fly is turned left, away from the boulder.
* **Rock on the left** (t = 4.0 s and t = 6.2 s). Left eye: LC4 ~100–184 Hz, LPLC2
  120–200 Hz. Drive L +1.24–1.39, R +0.73–0.79, "turning right ->": the fly turns
  away. One of these rocks was dodged (DODGE +100).
* **Hit** (t = 4.2 s in another run). The boulder turned red, in contact with the
  fly's side; HIT!, and a lost heart.
* **Game over.** The overlay shows score, survival and dodges, with the R / 1-2-3 / Q
  hint. The honest label is at the bottom of every frame.
* **HUD fix.** The first HUD used thicker black text as an outline. OpenCV's Hershey
  advance widths depend on thickness, so the outline drifted and produced ghost
  letters. The outline is now drawn as offset copies at the same thickness. The layout
  was also made width-aware (at 480 px the header overlapped), and the brain panel is
  scaled to fit below 600 px height.

A windowed check used the synthetic brain and injected keys, then closed itself:
`--window --max-wall-seconds 14 --script-keys "3:space,4.5:space,6:3,8:b,10:b,12:q"`.
Pause, the difficulty restart, opening and closing the brain window, and quit all
worked, and no child processes were left.
