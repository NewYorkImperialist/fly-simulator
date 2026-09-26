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
`experiment.py`; game 2: `chase.py` leader fly + LC10a eyes + rules,
`chase_experiment.py`; game 3: `rings.py` hoops + flight pilot + rules,
`rings_experiment.py`), `scripts/play.py`, `tests/test_games.py`. Nothing in the app
(`app.py`, `config.py`) was changed. Each game builds its own `Simulation` (game 3: a `FlightSimulation`).

Games: **1. ASTEROID DODGE** (`--game asteroids`, the looming channel: turn *away*),
**2. FOLLOW THE LEADER** (`--game chase`, the pursuit channel: turn *toward*) and
**3. FLY THROUGH RINGS** (`--game rings`, the same pursuit channel piloting the real
flapping-wing flight model).

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

## Game 2: FOLLOW THE LEADER (chase)

ASTEROID DODGE uses the brain's looming channel, which turns the fly *away*. This
game uses a different pathway, the small-object "pursuit" channel, which turns it
*toward* something. A dark leader fly weaves ahead. Each eye's view of it drives that
eye's **LC10a** neurons, and in the connectome LC10a drives the same side's DNa01/02,
so the brain steers toward the leader. The idea is NeuroMechFly v2's fly-following
demo, but here the whole FlyWire brain sits between the eyes and the legs, and none
of it was trained.

### Connectome check first (why LC10a, and not "plane through rings")

Before writing the game we stimulated the whole model directly
(`perpetualfly.brain.engine.LIFEngine`, reset to rest, per-neuron Poisson input for
0.3 s, 2–3 trials). Turn = the DNa01 + DNa02 group mean of that side, in Hz.

| input (FlyWire type, side, rate) | turn_L | turn_R | walk L/R | MDN | GF |
|---|---|---|---|---|---|
| LC10a left 10 / 20 / 30 Hz | 14 / 29 / 43 | 0 | 0 / 0–7 / 0–12 | 0 | 0 |
| LC10a left 50 / 100 / 200 Hz | 47 / 67 / 78 | 0–1 | 2–13 / 16–41 | 0 | 0 |
| LC10a right 20 / 50 / 100 / 200 Hz | 0–2 | 13 / 11 / 40 / 51 | 7–28 / 0–11 | 0 | 0 |
| LC10a left 100 + right 50 Hz | 48 | 0 | 11/31 | 0 | 0 |
| LC10a left 50 + right 100 Hz | **22** | 3 | 18/7 | 0 | 0 |
| LC10a both 100 Hz | **47** | 0 | 16/29 | 0 | 0 |
| LC10a + LC10d left / right 100 Hz | 88 / 0 | 0 / 68 | 13/39, 31/11 | 0 | 0 |
| LLPC1 + LLPC2 left / right 100 Hz | 73 / 10 | 28 / 67 | 15/23, 17/20 | 0–2 | 0 |

(LC10a: 115 left, 119 right neurons. Walk = BDN2/oDN1/P9.)

What this shows:

* **LC10a turns ipsilaterally, cleanly.** Left LC10a drives only the left DNa01/02
  (there is essentially no contralateral leak), and the reverse holds for the right.
  It drives no MDN and no giant fibre. It also raises the walk DNs, mostly on the
  opposite side.
* **LLPC1/2** turn toward the stimulated side as well, but they leak into the other
  side's turn group (28 Hz at 100 Hz input). LC10a is also the better-established
  pursuit neuron (courtship tracking: Ribeiro et al. 2018; Hindmarsh Sten et al. 2021).
  **We drive LC10a only.**
* **The model is left-biased.** Left LC10a is 3–4× more effective than right at low
  rates (20 Hz input: 29 vs 13 Hz). With both eyes driven, the left wins even when the
  right gets twice the rate (L50 + R100 still turns left). A target seen equally by
  both eyes therefore turns the model left. The interface below keeps both eyes from
  being driven at once.

Because the pursuit wiring works, the walking game was built. The fallback, "plane
through rings" in flight mode, was not needed.

### Embodiment

* **Leader.** A mocap fly built from ellipsoids and capsules: a dark body with
  tergite bands, red eyes, translucent wings and six legs. It is purely visual and has
  no collision.
* **Leader path.** It walks at a difficulty speed (easy / normal / hard: 8 / 9.5 /
  11 mm/s; the follower walks about 13 mm/s). Its turn rate is an Ornstein–Uhlenbeck
  process with τ = 0.7 s and SD 45 / 65 / 90 deg/s, clipped at 160 deg/s.
* **Leader rules.** It walks at 0.6× speed while the follower is more than 12 mm
  behind. After a catch it turns by up to ±60° and dashes at 2.2× speed for 0.7 s.
* **Rules.**
  * **FOLLOWING** while the thorax-to-thorax distance is below 9 mm; this scores
    10 points/s.
  * **CATCH** below 3 mm scores 100 × level. Every 3 catches the level goes up:
    leader speed +0.7 mm/s and weave +15 %.
  * **LOST HIM!** when the leader is more than 20 mm away for 0.4 s. This costs one of
    3 lives, and the leader is put back 7 mm ahead at a random bearing within ±25°.
  * **Game over** at 0 lives.
  * In the first 1 s (GET READY) the leader walks straight and nothing is scored.
* **High scores.** Kept per control mode and difficulty in the same
  `runs/games_highscores.json`, under `"chase"`.
* **Same body and motor side as game 1.** The follower is the same NeuroMechFly and
  hybrid CPG controller, with `heading_gain = 0`. The same `GameMapping` (turn gain
  0.6, r_ref 25 Hz, walk +0.25, MDN brake, 60 ms drive low-pass) and the same brain
  pacing (sim-paced worker, 20 ms windows, sync wait while stimulated) are used. Jumps
  are off: the giant fibre is not driven by this input.

### What the brain sees (input interface, ours)

`PursuitVision` (`chase.py`) updates every 5 ms. It uses the gaze-stabilised eye frames
of game 1 and treats the leader as a sphere of radius 0.8 mm. For each eye:

* LC10a rate = 150 Hz · size(θ) · ecc(az) · fov;
* size = ramp(θ; 1°, 4°) · (1 − 0.5 · ramp(θ; 30°, 60°)). The response is silent
  for sub-degree specks and halved for objects that fill the eye;
* ecc = ramp(az; 0°, 30°), where az is the leader's azimuth in that eye's frame,
  positive toward the eye's own side. A leader straight ahead drives neither eye; one
  20° to the left drives only the left LC10a, at 100 Hz;
* fov = the default field-of-view weight, which is blind behind the fly.

Events are `StimulusEvent("manual", side=eye, details={"set": "LC10a", "rate_hz":
r})` lasting 40 ms, re-sent every 20 ms and at once on a rise. `LC10a` was added to
`mapping.named_sets` for this. The brain takes the max rate per neuron over overlapping
events, so a falling rate takes effect within 40 ms.

**Why the eccentricity ramp.** The error angle is the natural steering signal for a
chasing fly: male houseflies turn at a rate proportional to it (Land & Collett
1974). The ramp also keeps the model's left bias (see above) out of the frontal
zone, where both eyes would otherwise be driven. It is our design, not a model of
LC10a receptive fields. The connectome is pooled per side, so we cannot address the
retinotopic subsets that real LC10a use. The ramp decides *how strongly* the brain is
driven. Which way the fly then turns comes from the wiring, and the mirror control
tests exactly that.

### Scientific check: does the brain follow better than chance?

```bash
.venv/bin/python scripts/play.py --game chase --brain --experiment 24 --seed 1 --json runs/chase_exp.json
```

**Design** (`chase_experiment.py`). The trials are paired: for trial i these
parameters are drawn once and replayed under all three conditions, interleaved:

* the leader's start bearing: 10–35°, alternating left and right;
* its heading offset: ±20°;
* its weaving path (the OU noise seed);
* the follower's gait phase.

Each trial resets the fly and the brain (all neurons at rest) and walks 0.5 s with the
leader hidden. It then places the leader 7 mm ahead at the trial's bearing and runs
6 s of game time. No lives are used; the outcomes are only measured.

The three conditions:

* `brain`: left eye → left LC10a;
* `mirror`: left eye → right LC10a and vice versa, with the same brain and motor map;
* `none`: brain disconnected, constant drive [1, 1]. This is the geometry / chance
  baseline, and it walks straight.

24 paired trials × 3 conditions took 2230 s of wall time. Per condition that was
normally 16 s (brain), 11 s (mirror) and 10 s (none). One trial stalled for about 15
minutes while the machine was paused, which does not affect the results.

| condition | trials | time following (< 9 mm) | mean distance (mm) | mean target error (deg) | catches (per trial) | leader lost | initial turn toward the leader (sign test) | mean initial turn toward (deg, first 0.5 s) |
|---|---|---|---|---|---|---|---|---|
| **brain** | 24 | **0.95 ± 0.01** | **5.8** | **28** | **68 (2.8)** | **0/24** | **23/24** (p = 3e-6) | **+32.2** |
| mirror | 24 | 0.10 ± 0.00 | 25.7 | 160 | 0 | 24/24 | 0/24 (p = 1e-7), i.e. away | −61.2 |
| none | 24 | 0.27 ± 0.02 | 19.9 | 120 | 12 (0.5) | 24/24 | 15/24 (p = 0.31) | +0.5 |

| paired comparison | follow-time difference | first follows longer | sign test p | Wilcoxon p | lost only in the second |
|---|---|---|---|---|---|
| brain vs none | +0.68 | 24/24 | 1.2e-7 | 1.8e-5 | 24 (McNemar p = 1.2e-7) |
| brain vs mirror | +0.86 | 24/24 | 1.2e-7 | 1.2e-7 | 24 |
| mirror vs none | −0.18 | 0/24 | 1.2e-7 | 1.2e-7 | – |

Time following is the fraction of the 6 s with a distance below 9 mm; the ± values
are standard errors. The initial turn is the heading change over the first 0.5 s
after the leader appears, signed toward the side it appeared on.

**Reading.**

* **The brain follows.** With the correct wiring the fly kept the leader within 9 mm
  for 95 % of the time. Its worst trial was 72 %. It never lost the leader and caught
  it 2.8 times per 6 s.
* **Both sides work despite the asymmetry.** Leader starting on the left: 97 %
  following, +40° initial turn. On the right: 94 %, +25°. The right side is weaker, as
  the connectome check predicts.
* **It turns toward the leader from the first moment.** It did so in 23 of 24 trials.
  The exception (trial 3, −10°) still followed for 100 % of the trial.
* **The disconnected baseline loses the leader every time.** It walks straight, so
  the weaving leader leaves the 9 mm zone after a median of about 3 s (lost 24/24).
  Its 27 % following is just the time before the leader drifts off. The 12 catches
  happen when the leader happens to cross its path.
* **Mirroring flips the behaviour.** With the eyes mirrored, the same brain turned
  *away* in 24 of 24 trials (−61°), lost the leader fastest (median 1.6 s) and did
  worse than the disconnected fly in every pair.
* **So the pursuit comes from the wiring.** It is LC10a → ipsilateral DNa01/02 (peak
  turn-group rates ~110–130 Hz, walk DNs ~50 Hz; no MDN or giant fibre activity).
  The interface sets its gain. The mirror control shows that the same interface with
  the sides swapped produces flight from the leader, not pursuit.

### HUD, window, controls

* **Game view.** The follow camera sits 15 mm behind the fly, 30° down, so that both
  flies are large in the frame.
* **HUD.** Top:
  * title, control mode, score, hearts, level, difficulty and best score;
  * time, % of time following, catches and losses;
  * a distance meter with the catch zone (blue), the follow zone (green), the current
    distance, and the target error with FOLLOWING / TOO FAR;
  * a small radar with the follower at the centre facing up and the leader as a dot,
    so an off-screen leader can still be found.

  Centre: event banners (FOLLOW HIM!, CATCH +n, LEVEL n, LOST HIM!), GET READY, GAME
  OVER, PAUSED. Bottom: the honest label and the key help.
* **Brain panel.**
  * **SEES**: LC10a Hz per eye, with the optic lobe each eye drives (swapped in mirror
    mode), and the leader's distance and side.
  * **DOES**: DNa01/02 L and R, walk, MDN and giant fibre (not mapped here).
  * the CPG drive and turn direction, and the label "one eye's LC10a drives the same
    side's DNa01/02 (turn toward)".
* **Keys.** As in game 1: SPACE, R, 1/2/3, B (brain window), Q/ESC.

### Commands

```bash
.venv/bin/python scripts/play.py --game chase --brain --window
.venv/bin/python scripts/play.py --game chase --brain --window --difficulty easy --brain-window
.venv/bin/python scripts/play.py --game chase --brain --control mirror      # or: none
.venv/bin/python scripts/play.py --game chase --brain --frames runs/frames --frame-times 2,5
.venv/bin/python scripts/play.py --game chase --brain --record runs/chase.mp4 --max-seconds 20
.venv/bin/python scripts/play.py --game chase --brain --experiment 24 --seed 1 --json runs/chase_exp.json
.venv/bin/python scripts/play.py --game chase --synthetic-brain --max-seconds 5   # no data
```

API: `ChaseSession(GameBrain("brain").start(), ChaseConfig(difficulty="normal"))`,
then `.step()` repeatedly. It exposes `.game` (`ChaseGame`: `score`, `catches`,
`follow_frac`, `dist`, `error_deg`, `events`), `.leader` (`LeaderFly`) and
`.vision` (`PursuitVision`). `hud.compose(frame, session)` picks the chase HUD
automatically.

### Frames checked (game 2)

These are renderer PNGs from `--frames` (960 × 640 plus the panel), read back and
then deleted.

* **GET READY** (`none`, t = 0.5 s). The dark leader fly is 6 mm ahead and to the
  right; its wings, red eyes and legs are visible. The panel shows the right eye →
  right LC10a at 143 Hz, but DOES stays at 0 Hz because the brain is disconnected.
* **Brain following** (t = 2.0 s). The leader is 7.6 mm away, 42° to the right. Right
  LC10a is at 150 Hz, turn R DNa01/02 at 25 Hz, drive L +1.47 / R +0.69, "turning
  right ->", and the fly is turning toward the leader. The banners show FOLLOW HIM! and
  CATCH +100, and the radar dot sits inside the follow ring.
* **Catch and level up** (t = 5.0 s). The leader is directly in front of the follower
  at 3.4 mm; the banners show CATCH +100 and LEVEL 2, and following is 100 %.
* **`none` losing it** (t = 2.5 s). The leader is 19 mm away, 141° behind to the right
  and off screen; only the radar shows it. The meter reads TOO FAR.
* **Mirror** (t = 1.6 s). The leader is 17° to the left. The panel shows "left eye →
  right LC10a 44 Hz", turn R 25 Hz, "turning right ->": the fly turns away. LOST HIM!
  and one heart is gone.
* **Game over** (mirror, t = 5.6 s, 3 losses). The overlay shows score, % followed and
  catches, with the R / 1-2-3 / Q hint. The honest label is at the bottom of every
  frame.
* **HUD fix.** Two LOST HIM! banners stacked when the losses were 1.6 s apart. The
  banner window was shortened to 1.5 s.

### Limitations

* **The interface is ours.** The response function (the eccentricity ramp above all,
  and max 150 Hz), gaze stabilisation, the equivalent-sphere leader and the motor
  gains were chosen by us. The turn *direction* and the neurons carrying it are the
  connectome's, which the mirror control demonstrates.
* **Only LC10a is driven.** The leader does not also loom (no LC4 / LPLC2) and does
  not produce optic flow. The follower sees nothing else, including no self-motion
  flow. Real pursuit also uses LC9, LC11 and other channels.
* **Left-right asymmetry.** Right LC10a drives the right turn DNs 3–4× more weakly
  than left LC10a at low rates. The model's right turns toward the leader are
  therefore weaker, and at small errors it steers more strongly to the left.
* **Oscillation.** The loop (20 ms brain windows, 60 ms drive low-pass, near-saturating
  DN rates) weaves around the leader, with the error swinging about ±20–45°. It is a
  proportional-ish controller with delay, not a smooth pursuit.
* **Leader physics.** The leader is a mocap ghost: no contact and no leg motion (it
  slides). A "catch" is a distance threshold.
* **Speed.** With the brain the game runs at about 0.35–0.4× real time on an M1,
  because the brain is stimulated almost continuously and the sync wait keeps the loop
  closed.
* **One experiment.** Results are from one brain seed, 24 paired trials, normal
  difficulty and a 6 s horizon.

## Game 3: FLY THROUGH RINGS (real flight)

The fly **flies**: the flapping-wing flight model of docs/FLIGHT.md (MuJoCo fluid
forces on the beating wings, dt 5e-5 s, no external force anywhere) with
`FlightMode` (jump take-off, forward flight, crash detection) and its
`HoverController`. Orange hoops stand across the flight path at varying lateral
offsets. The next ring is the brain's target: its bearing drives the LC10a pursuit
neurons on the side where it is (the FOLLOW THE LEADER interface, unchanged), and
the left–right DNa01/02 difference sets the **heading rate** of the flight
controller. The connectome turns the fly toward the ring.

### Embodiment

* **Flight.** `RingsSession` builds a `FlightSimulation` + `ActionManager` +
  `FlightMode`. A new game starts with the real take-off: `FlightMode.takeoff()`
  (long-mode jump, wings on at the end of the leg stroke, ~0.17 s after the start),
  then forward flight at the level's speed (50 mm/s on normal) with the COM held
  6 mm above the ground. Relaunches after a crash, and experiment trials, use an
  **air start**: the fly is placed at 6 mm in the hover posture and held for 40 ms
  while the wingbeat fades in, then released. A free start drops ~2 mm before the
  wings reach full stroke.
* **Controller inputs.** The game uses only `FlightMode`'s own inputs: forward
  speed (`_speed`), heading goal (`_yaw_goal`, which the heading reference follows at
  ≤ 6 rad/s) and altitude clearance. Three game settings:
  * the heading loop is stiffened (`yaw_wn` 30 → 80 rad/s, ζ 1). With the default
    loop, the heading lagged the goal by ~0.15 s and overshot by ~18° in forward
    flight (measured on a scripted 150°/s turn);
  * velocity control only (`xy_zeta` 3; the position set point follows the fly, as
    in escape flight);
  * the altitude integral starts from a per-speed trim (`Z_TRIM`, measured over 3 s
    of level flight). Without the trim, the fly flies ~0.8 mm high for ~1.5 s.
* **Heading.** The eyes' gaze frame and the game use the yaw of the level stroke
  frame (`FlightPilot.true_yaw`). `Simulation.heading()` projects the thorax x axis,
  which points ~48° nose-up in the hover posture, so any bank leaks into it (up to
  ~24° at the 25° roll limit).
* **Rings.** `RingCourse` is a pool of 4 mocap hoops. Each hoop is 20 capsules on a
  vertical circle (radius 3 mm on normal, tube 0.22 mm) plus a thin post to the
  ground. They are purely visual (contype 0). Three are on screen: the target is
  orange, later ones pale blue, the one just passed turns green or red. Layout: the
  first ring is 30 mm ahead, then one every 32 mm along the course. Each ring steps
  laterally by ±U(2, max_shift) from the previous one (max_shift 6 mm on normal;
  course centre ±10 mm). A ring faces the direction from the previous ring.
* **Outcome.** When the thorax COM crosses the ring plane, it is **through** if it
  lies within `radius − 0.3 mm` of the centre. Otherwise it is **missed** ("clipped
  the rim" within ±0.8 mm of the hoop). A ring also counts as missed when it stays
  > 110° off the heading for 0.3 s ("turned away"), or when it is not reached within
  2.5× the straight-line flight time. After a turned-away miss, a new course is laid
  out ahead of the fly.
* **Crash.** A `FlightMode` crash (tilt > 120° or a body contact for 50 ms), or a
  model blow-up, costs a life. The fly falls for 0.4 s, then gets an air start.

### What the brain sees and does (interface, ours)

* **Input.** `RingVision` = `PursuitVision` (game 2) aimed at the target ring's
  centre, with the ring radius as the object size. It uses the same
  `pursuit_response`: rate = 150 Hz × size(θ) × ramp(azimuth 0→30°) × field of view
  → `StimulusEvent("manual", side=eye, details={"set": "LC10a", ...})`. A ring
  straight ahead drives neither eye. A near ring (θ > 30–60°) keeps half its drive.
  We know of no data on how LC10a responds to a hoop; the ring's centre is simply
  the pursuit target.
* **Output.** turn = tanh(turn_L / 25 Hz) − tanh(turn_R / 25 Hz) (DNa01/02 group
  means, as in `descending_to_drive`), low-passed with τ = 60 ms. The heading rate
  is 120°/s × turn, integrated into `FlightMode`'s heading goal. For `none`, turn = 0.
* **Gain choice (disclosed).** We first used 240°/s per unit turn, the walking
  games' measured CPG turn rate. With it, the flying fly oscillated ±40° around the
  ring direction and passed 3 of 5 rings in a 6 s game. In one pilot game each (seed
  0, normal), 180°/s passed 10 of 13 rings and 120°/s passed 12 of 14. We kept 120.
  The experiment below used a different seed (1). No other parameter was tuned on
  brain runs.
* **Not controlled by the brain:**
  * forward speed: fixed per level. The walk DNs (BDN2/oDN1/P9) and MDN are walking
    commands, and we found no principled flight-speed or brake mapping;
  * altitude: held by the controller. Rings have no vertical offset: LC10a gives no
    elevation signal we use, and the model's descending groups have no flight
    altitude channel;
  * giant fibre: not mapped. It stayed at 0 Hz in every trial.
* **Game.** Each ring through scores 100 × min(streak, 5) × level. Every 5 rings is a
  new level (+5 mm/s, +0.5 mm lateral step). A miss or a crash costs a life (3
  lives). High scores go to `runs/games_highscores.json` under `rings`. The
  difficulties differ as follows:

  | difficulty | speed (mm/s) | max lateral step (mm) | ring radius (mm) |
  |---|---|---|---|
  | easy | 40 | 4 | 3.5 |
  | normal | 50 | 6 | 3.0 |
  | hard | 60 | 8 | 2.6 |

  The hoop radius is compiled into the model, so R / 1 / 2 / 3 change speed and
  lateral steps but keep the session's radius.

### Scientific check: does the brain fly through rings better than chance?

`--experiment N` runs paired single-ring trials (`rings_experiment.py`):

* **Trial i.** The ring's lateral offset alternates sides, with magnitude
  U(3.5, 7) mm, so a straight flight always misses. A settling jitter of
  U(0, 20) ms is added. Both are replayed under brain / mirror / none.
* **Each trial:**
  1. reset the model and the brain (neurons at rest);
  2. air start at the origin heading +x;
  3. 0.5 s of straight flight with no ring;
  4. one ring appears 30 mm ahead;
  5. the trial ends at the ring plane, or on turned away / timeout / crash.

Real FlyWire brain, headless, 16 trials × 3 conditions, seed 1. This took 4.3 min
wall (≈ 2.3 s per trial; one trial stalled 140 s while the machine was busy):

| condition | trials | through the ring | mean distance from the ring centre (mm)¹ | turned away / crashed | initial turn toward the ring, first 0.3 s (p) | mean initial turn toward (deg) |
|---|---|---|---|---|---|---|
| **brain** | 16 | **15/16 (94%)** | **1.07 ± 0.22** | 0 / 0 | **16/16** (p = 3e-5) | +14.9 |
| mirror | 16 | 0/16 (0%) | 21.2 ± 0.9 | 13 / 0 | 0/16 (p = 3e-5) | −14.8 |
| none | 16 | 0/16 (0%) | 5.33 ± 0.28 | 0 / 0 | – | 0 |

| comparison | pairs | through only in first | through only in second | McNemar p | first closer to the centre | Wilcoxon p |
|---|---|---|---|---|---|---|
| brain vs none | 16 | 15 | 0 | 6e-5 | 16/16 | 3e-5 |
| brain vs mirror | 16 | 15 | 0 | 6e-5 | 16/16 | 3e-5 |

¹ At the ring plane, or where the trial ended for the mirror's 13 turned-away
trials.

* The brain turned toward the ring in every trial and flew through 15 of 16 rings.
  The single miss clipped the rim: 2.97 mm from the centre, with a pass limit of
  2.7 mm.
* The mirror turned away in every trial. In 13 trials it turned so far that the ring
  ended up behind the fly (max heading change ~79°).
* The disconnected fly crossed every ring plane at its offset.
* DN peaks per trial: DNa01/02 on the ring's side 25–125 Hz (group means per 20 ms
  window). MDN and the giant fibre were 0 Hz in all 48 trials.

This is the same connectome property as game 2 (LC10a → ipsilateral DNa01/02), now
closing the loop through a flying body. The flight dynamics sit between the DNs and
the path: heading tracking, the crab angle while the velocity catches up, and the
altitude loop.

In pilot games (normal difficulty, levels 1–3, 50–60 mm/s) the brain passed 12 of
14 and 10 of 13 rings (gains 120 and 180°/s). Most misses clip the rim after a
lateral step of 5–7 mm.

### Commands

```
# play (window; SPACE pause, R restart (take-off again), 1/2/3 difficulty, B brain window, Q quit)
.venv/bin/python scripts/play.py --game rings --brain --window
# headless frames / clip; air start instead of the take-off
.venv/bin/python scripts/play.py --game rings --brain --frames runs/rings --frame-times 1.2,1.6 --width 800 --height 600
.venv/bin/python scripts/play.py --game rings --brain --air-start --record runs/rings.mp4
# controls
.venv/bin/python scripts/play.py --game rings --brain --control mirror
.venv/bin/python scripts/play.py --game rings --brain --control none
# the experiment (numbers above)
.venv/bin/python scripts/play.py --game rings --brain --experiment 16 --seed 1 --json runs/rings_exp.json
# tests (no connectome needed)
.venv/bin/python -m pytest tests/test_games.py -k "ring or turn_command"
```

Speed: without the brain, ~0.65× real time. With the real brain, 0.3–0.6× (the
brain panel shows its real-time factor).

API: `RingsSession(GameBrain("brain").start(), RingsConfig(), seed=0)`, then
`step()` in a loop. `session.render(renderer)` gives the frame: the camera follows
the flying fly without the altitude zoom-out, and without the heading freeze that
the hover posture's pitch would trigger.

### Frames checked (game 3)

Renderer PNGs at 800×600 + panel, read by eye, then deleted (disk):

* **take-off (t = 0.1 s)**: the fly seen from behind, airborne, wings spread,
  legs hanging. HUD "GET READY", flight line `FLIGHT FORWARD 218 Hz alt 6.1 mm`.
* **approach (brain, 1.2 s)**: the orange ring ahead, slightly right. The right
  eye's LC10a is at 44 Hz, and the heading command is −32°/s ("turning right"). Two
  pale-blue rings behind it, the radar shows the ring bar ahead.
* **through (brain, 1.6 s)**: the fly just past a **green** (passed) ring, "THROUGH!
  +100". The next ring is 20° left: left LC10a 111 Hz, DNa01/02 L 75 Hz, R 0,
  heading command +100°/s ("turning left").
* **missed (none, 2.5 s)**: a **red** ring passing to the left of the fly, "MISSED",
  one heart gone. The panel reads "disconnected (control): heading rate 0" while the
  left LC10a still gets 44 Hz from the next ring.
* The window was checked with injected keys (pause, restart with take-off, hard) and
  closed.

### Limitations

* **Two thirds of the loop are ours.** The interface (ring centre → LC10a rate,
  DNa01/02 → heading rate, gain 120°/s chosen from pilot games) and the flight
  controller (an engineered PID; see docs/FLIGHT.md) are designed by us. The brain
  supplies only the side-to-side sign and the timing of the turn command.
* **Heading only.** Speed and altitude are fixed and the rings are at the flight
  altitude. No brain output is mapped to them.
* **The rings are not physical.** Passing is judged geometrically at the thorax COM
  (radius − 0.3 mm), so a wing tip can go through a hoop. Wing-tip contact would
  need colliding hoops, and a hard contact can blow up the dt 5e-5 fluid model.
* **Visual model.** The ring is treated as a small moving target (LC10a pursuit).
  Real flies would also see its optic flow, looming and edges. The eyes' gaze is
  level and follows the body yaw, low-passed with 80 ms (head stabilisation assumed).
* **Scope of the claim.** 16 paired trials give an unambiguous difference
  (15/16 vs 0/16), but they cover a single speed (50 mm/s), distance (30 mm) and
  offset range (3.5–7 mm).
* The model's left bias (game 2) probably shows here too: in the brain trials the
  per-trial LC10a peak averaged 26 Hz on the left vs 90 Hz on the right (8 rings on
  each side). This is consistent with left rings being centred faster, but we did
  not test it separately.
