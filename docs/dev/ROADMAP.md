# Fly Simulator roadmap

Everything planned so far, compared. Effort estimates are rough (one developer, Apple M1).
"Brain-real" = how much of the behaviour comes from the real FlyWire wiring
(★★★ = decision made by real wiring end to end, ☆ = hand-built / not brain).

> **Status 2026-09-25 (action library, docs/ACTIONS.md):** B3 escape jump is done
> (J key; the giant fiber triggers it with `--brain-actions`, 0.12 s after O). B4:
> the freeze action exists (Z), but it has no brain trigger (no clean freezing DN).
> B5: the groom action replays recorded NeuroMechFly grooming (Y) and DNg12 is read
> out, but in this model head / JO input reaches DNg12 at only ≤ 13 Hz, so the brain
> doesn't trigger it yet. B7: proboscis joints (`--full-body`, default in the CLI)
> and MN9 → proboscis extension (T) are done; sugar patches on the ground are still
> missing. Also new: back away (E), turn in place (, .), wing raise (W).

## A. Quick wins (≤ 1 h each, no physics changes) — all done 2026-09-25 (details in docs/dev/STATUS.md)
| # | Item | Effort | Needs code | Brain-real | Visible impact |
|---|---|---|---|---|---|
| A1 | Looming makes fly back up (`--brain-backup`, MDN threshold 40→20 Hz) — **done: `--brain-backup` (0.5–1.2 s after O: vx mean −4.2, min −21 mm/s vs +2.3 without it)** | minutes | tiny | ★★ | High |
| A2 | Fly window Retina scaling (half-size text fix) — **done: automatic ×2 on Retina, `FLY_SIMULATOR_FLY_SCALE` override, shared `fly_simulator/display.py`** | < 1 h | small | – | High |
| A3 | Floor-reflection toggle (14→8 ms/frame) — **done: `--no-reflections`, draw 10.6–11.2 → 5.5–6.4 ms** | < 1 h | small | – | Medium |
| A4 | Change terrain difficulty live (`[` / `]`) — **done: `[` / `]`, `ProceduralTerrain.set_difficulty()`** | < 1 h | small | – | Medium |
| A5 | Screenshot key (renderer frames, never screen capture) — **done: `I` saves the fly PNG (clean + HUD) and a brain PNG** | < 1 h | small | – | Medium |
| A6 | Record toggle key (live MP4) — **done: `M`, 30 fps of sim time, REC badge** | < 1 h | small | – | Medium |
| A7 | On-screen `?` help overlay — **done: `?` overlay + terminal** | < 1 h | small | – | Medium |
| A8 | Pause auto-hits while fallen — **done: skipped while down + 1 s, counted and logged** | < 1 h | small | – | Medium |
| A9 | Honest distance stats (exclude flights) — **done: `walked_distance_mm`, avg speed = walked / run time** | < 1 h | small | – | Low |
| A10 | Bundle brain atlas in package data — **done: `package-data`, wheel checked** | minutes | tiny | – | Low |

## B. Brain-driven reactions
| # | Item | Effort | Brain-real | Visible impact | Depends on |
|---|---|---|---|---|---|
| B1 | Fly "sees" the whip coming (geometric looming → LC4/LPLC2) | ~1 day | ★★ (sense computed, decision real) | Very high | – |
| B2 | Sensory screen: which touch/sensory neurons reach walk/turn/escape DNs | few hours | ★★★ (finding) | Low (knowledge) | – |
| B3 | Escape jump (giant fibre → mid-leg push) | 1–2 days | ★★ | Very high | B1 or O key |
| B4 | Freeze / startle on hard hit | hours–1 day | ★★ if via real DNs | High | B2 |
| B5 | Antennal grooming (JO → grooming DNs → front-leg sweep) | few days | ★★★ decision (Shiu-validated) | High | head-hit detection |
| B6 | Stress / octopamine state (real OA-neuron activity → slow level → excitability + gait vigour) | few days | ★★ trigger real, effect modelled | High | – |
| B7 | Feeding: sugar patches + proboscis joints (MN9 → proboscis) | ~1 week | ★★★ decision (Shiu-validated) | High | body joints |
| B8 | Real vision: compound eyes + flyvis + bridge to LC4/LPLC2 | 1–2 weeks | ★★★ | Very high | replaces B1 sense |
| B9 | Visual steering toward/away from objects (flyvis motion) | ~1 week | ★★★ | High | B8 |
| B10 | Nerve cord: BANC brain+VNC connectome (whip touch enters real VNC) | months | ★★★ | Very high | research |
| B11 | Motor neurons → muscles → legs (brain fully drives body) | months+ | ★★★ | Ultimate | B10, research |

## C. Experiments with what exists
| # | Item | Effort | Needs code | What you learn |
|---|---|---|---|---|
| C1 | Virtual optogenetics (switch on DNa02 / MDN / DNg100 → watch fly) | hours | small (key/CLI) | Which neurons cause which behaviour |
| C2 | Virtual lesions (silence MDN / giant fibre) | hours | small (engine) | What breaks without a neuron |
| C3 | Sugar vs bitter competition (T + K) | minutes | none | Reproduces Shiu result live |
| C4 | Brain tour: stimulate each sensory class, watch regions | hours | small | Map of the brain's responses |
| C5 | Weakest gait phase for hits | hours | none (`demo_perturbation.py --calibrate`) | Body vulnerability |
| C6 | Direction × strength survival map | hours | none | Most dangerous hit directions |
| C7 | Reflexes on/off (`--controller cpg` vs hybrid) on hard terrain | hours | none | Value of reflexes |
| C8 | Leg adhesion on/off | hours | config | Role of sticky feet |
| C9 | Terrain × hit-strength survival heatmap (`eval_robustness.py`) | hours (compute) | none | Robustness landscape |
| C10 | Walking speed vs stability (CPG frequency) | hours | config | Speed/robustness trade-off |
| C11 | Plot the 78 min of soak logs (`runs/soak/`) | < 1 h | small | Survival curves, fall causes |

## D. Learning
| # | Item | Effort | Brain-real | Visible impact |
|---|---|---|---|---|
| D1 | First PPO training run (recovery/righting; bigger residual, start-flipped episodes, obs history) | prep ~1 h + 4–8 h overnight | ☆ (artificial network) | Very high — "learns not to care" |

## Suggested agenda
1. **Session 1 (≈1–2 h):** A1, A2, A5, A7, A8 + C3 — polish + first reactive behaviour.
2. **Session 2 (1–2 days):** B1 + B2 in parallel, then B3 — the fly flinches/jumps at the whip it sees.
3. **Overnight:** D1 training run (+ C9 heatmap on spare cores).
4. **Session 3 (few days):** B6 stress state, B5 grooming, C1/C2 optogenetics + lesions.
5. **Bigger projects:** B8 real vision → B9 visual steering; B7 feeding.
6. **Research track:** B10 nerve cord → B11 full brain-to-muscle.

## Done since the roadmap was written (2026-09-26)
- **Real vision** (`--real-vision`): compound eyes → flyvis → LC4/LPLC2 bridge (docs/VISION.md).
- **Eternal jobs**: Sisyphus, hamster wheel, mowing, raking, doner kebab (`--job NAME`, docs/JOBS.md).
- **Fly Brain Plays: asteroids** (`scripts/play.py`, docs/GAMES.md).
- **Whip → pain → speed-up** (`--stress`, docs/STRESS.md) and **flyswatter dodging** (`--swatter`, docs/SWATTER.md).
- **Obstacle courses** (`--course NAME`, docs/COURSE.md), **sensory screen** (docs/SENSORY_SCREEN.md).
- **Real flight** (`--flight`, key L; brain escape flights; docs/FLIGHT.md).
- **Brain playground**: virtual optogenetics, lesions, decision meters (docs/PLAYGROUND.md).
- **Brain games 2 and 3**: Follow the Leader (`--game chase`, LC10a pursuit) and Fly Through Rings (`--game rings`, real flight).
- **Fast brain→body path** (GF crossing → jump in ~2–3 ms).
- **Taste patches** (`--taste-patches`: sugar → stop + proboscis via MN9; bitter suppresses feeding).
- **Realistic doner + knife** visuals, and **kebab brain reactions** (touch per cut, event-driven taste, hunger, startle key).
- **Looming habituation** (`--habituation`) and **brain replay** (`--brain-record`, `scripts/brain_replay.py`).
- **Mushroom-body fear learning** (`--learning`, `--odor-zones`): brain-level learning works; no avoidance behaviour in this model (docs/FEAR_LEARNING.md).

## Queue (next)
1. **RL training run** (D1): recovery and righting. Needs several GB of free disk and a night of CPU.
2. **Olfaction without runaway**: tune the model (or add inhibition) so real antennal-lobe input works, then real odour navigation.
3. **Research track**: BANC brain + nerve-cord connectome (whip touch through the real VNC; avoidance routes), motor neurons → muscles.
4. **Polish**: other job scenes to the kebab's visual standard; a startle key for kebab in the main app; `--game` inside `run_sim.py`.

## Next job (queued after bowling)
- **broccoli_toss** — done 2026-09-27 (docs/JOBS.md#broccoli_toss). a fly-scale streamer's room (desk, glowing monitors with a generic "LIVE" stream, RGB strips, ring light, mic, posters, gaming chair). The host fly (real physics, walking) brings a plate of broccoli to the viewer fly (posed / animated in the gaming chair, facing the monitors). The viewer takes it, ponders it for 0.75–1 s (head tilt, plate still; with `--brain` the bitter pathway fires), then in one sudden casual flick throws the whole plate over its shoulder without looking. The plate flies with real physics; where it lands behind, everything explodes (a labelled cartoon blast: flash, fireball, shockwave impulse; props then move with real physics; debris and broccoli rain). The viewer keeps watching, unbothered; the room rebuilds; repeat forever. Counters: plates yeeted, explosions, stream viewers, vegetables eaten: 0. No real streamer names or likenesses.
