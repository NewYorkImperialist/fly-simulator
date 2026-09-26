# PerpetualFly — Project Spec

> **Fruit fly. Runs forever. Terrain gets worse. I can smack it. It learns not to care.**

A physically simulated fruit fly (FlyGym / NeuroMechFly 2.x on MuJoCo) jogs forward indefinitely through endless procedural terrain (rocks, bumps, slopes, gaps). The user can interactively hit/"whip" the fly with MuJoCo external forces. Later, an RL controller (residual on top of FlyGym's CPG/hybrid controller) learns to recover.

Loop: jog → obstacle / external hit → destabilized → controller reacts → recover → continue jogging forever. No finish line.

## Core technology
Python, FlyGym / NeuroMechFly 2.x, MuJoCo, NumPy, FlyGym's standard visualization utilities, Gymnasium-style abstractions. RL (Stable-Baselines3 PPO) must be addable later but is NOT a dependency of the first milestones.

**FlyGym APIs changed between versions. Inspect the installed FlyGym 2.x API and official examples before writing code. Do not copy FlyGym v1 code from the internet.** When uncertain, read the installed source.

## Layout
```
fly_simulator/
    __init__.py
    app.py
    simulation.py
    config.py
    controllers/
    terrain/
    interaction/
    metrics/        (or similar)
    rl/
scripts/
    run_sim.py
tests/
README.md
pyproject.toml
```
Keep it clean; don't build a huge framework.

## Phases
1. **Simplest working fly**: NeuroMechFly on flat ground, tracking camera, FlyGym-supplied locomotion controller (CPG/hybrid), walks forward continuously, rendered, runs until quit.
2. **Perpetual locomotion**: no episode timeout in interactive mode. Runs until quit / unrecoverable sim error / explicit reset. Track sim time, distance traveled, average forward speed, current speed, #falls, #recoveries, #external hits, longest uninterrupted jogging interval. Print periodically in terminal.
3. **Endless procedural terrain**: chunk-based recycling, bounded number of chunks around fly; chunks far behind are recycled ahead; ~constant memory. Types: flat, small bumps, rough ground, isolated rocks, blocks, shallow dips, mild slopes up/down, small gaps. Start conservative (traversable by baseline controller). `TerrainGenerator(seed=42)`, difficulty presets EASY/NORMAL/HARD/CHAOS, configurable probabilities (e.g. flat 60, bumps 15, rocks 10, blocks 5, slopes 7, gaps 3).
4. **Interactive whip**: MuJoCo external wrench (`data.xfrc_applied` or equivalent), applied to thorax. `perturbation.apply_impulse(body="thorax", direction=..., magnitude=..., duration=...)`; force cleared after duration. Directions: left, right, forward, backward, up, random. Keys ≈ SPACE random, LEFT/RIGHT, UP forward/upward, DOWN backward. Strength levels 1 gentle / 2 medium / 3 hard / 4 absurd, physically scaled to fly mass (~1 mg scale; check model units!) and empirically calibrated: gentle perturbs gait without launching; hard can knock over.
5. **Fall detection**: `FallDetector` with states UPRIGHT / DESTABILIZED / FALLEN / RECOVERING, using thorax height, tilt, sustained horizontal body, lack of forward progress, body-ground contacts. Don't terminate on fall in interactive mode. Manual reset key.
6. **Obstacle spawning keys**: R rock ahead, B bump, S slope, G gap, F flatten next chunk, SPACE hit, P pause, X reset, ESC/Q quit (adjust to viewer constraints). Never spawn inside the fly; configurable spawn distance ahead.
7. **Camera**: smooth tracking camera, side/rear three-quarter default; follow / side / top-ish, switchable if easy.
8. **Logging**: `runs/<YYYY-MM-DD_HH-MM-SS>/{config.json, events.csv, metrics.csv, summary.json}`. Events: timestamp, event_type, force_direction, force_magnitude, terrain_type, fall_detected, recovered. Metrics: position, speed, orientation, angular velocity, height, joint state summary, contact summary. Configurable sampling frequency.
9. **Auto perturb**: `--auto-perturb` with min/max seconds between hits, magnitude range, direction distribution, deterministic seeds. e.g. `python scripts/run_sim.py --auto-perturb --terrain normal`.
10. **Future RL env** (only after 1–9 are stable): Gymnasium env. Obs: orientation, angular vel, linear vel, thorax height, joint pos/vel, contacts, maybe CPG phase. NO privileged hit info. Action = RESIDUAL: `final = baseline + scale * correction`. Reward: forward/target velocity + upright + recovery − energy − instability; don't over-reward survival (standing still must not be optimal). Curriculum later.

## Long-term eval metrics
Mean time to failure, longest uninterrupted run, P(survive T s), avg recovery time, max impulse survived, obstacle success rate, hits survived, falls per km, recovery percentage.

## Engineering requirements
1. Don't fake physics — hits use MuJoCo forces.
2. Never teleport the fly to look stable (reset key is an explicit reset, fine).
3. Use FlyGym/NeuroMechFly 2.x APIs; trust installed code over the internet.
4. Keep FlyGym's physics timestep.
5. Separate simulator / controller / terrain / perturbation / keyboard-UI / metrics / RL env.
6. Type hints where useful.
7. Comment non-obvious MuJoCo/FlyGym behavior.
8. README with exact install + run commands, kept up to date.
9. `--headless` mode.
10. Rendering separable from physics (many steps without rendering).
11. Config via dataclasses (+ JSON/TOML/YAML), not scattered magic constants.
12. Document why when physics params change.
13. Detect NaNs/instability and fail loudly with diagnostics.

## Milestones
- **M1**: `python scripts/run_sim.py` → NeuroMechFly walking continuously on flat terrain, tracking camera, terminal runtime+distance, runs until quit.
- **M2**: SPACE = visible physical shove; directions work; fly keeps locomoting.
- **M3**: rocks/bumps, procedural chunks, infinite recycling.
- **M4**: fall detection, interactive obstacle spawning, event logging, auto random perturbations.
Only after M1–M4 are reliable: RL.

## Working style
implement → run → inspect error → fix → run again. No untested speculative code dumps. At each milestone report: what works, files changed, exact run command, known limitations, next milestone.
