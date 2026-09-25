# Residual RL for PerpetualFly (spec phase 10)

Status: the environment, PPO training script and evaluation script are built and
tested. **No policy has been trained yet.** Only a smoke test has been run: 4096 steps,
which shows the pipeline works end to end.

```bash
uv pip install --python .venv/bin/python -e ".[rl,dev]"     # gymnasium, stable-baselines3 (+torch), tensorboard
.venv/bin/python -m pytest -q tests/test_rl_env.py
```

The core package never imports `perpetualfly.rl`. gymnasium, SB3 and torch are only
needed for RL (a test checks this).

## Files

| file | what |
|---|---|
| `perpetualfly/rl/env.py` | `PerpetualFlyEnv(gymnasium.Env)`, `EnvConfig`, `RewardConfig`, `CurriculumStage`, `default_curriculum()` |
| `perpetualfly/rl/wrappers.py` | `make_env(cfg, rank, seed)` (picklable thunk + SB3 `Monitor`), `make_vec_env(cfg, n_envs, seed)` (SubprocVecEnv, spawn) |
| `perpetualfly/rl/evaluation.py` | `evaluate(env, policy, sim_seconds)`: long-horizon metrics through the app's `RunMetrics` |
| `scripts/train_ppo.py` | SB3 PPO + VecNormalize, curriculum, checkpoints, periodic long evaluation, TensorBoard |
| `scripts/eval_policy.py` | headless long evaluation of a checkpoint or `--baseline` (zero residual) |
| `tests/test_rl_env.py` | check_env, exact baseline equivalence, obs sanity / no hit leakage, reward ordering, termination, seeding, stages, instability |

## Environment design

```python
from perpetualfly.rl import PerpetualFlyEnv, EnvConfig
env = PerpetualFlyEnv(EnvConfig(stage="normal"))       # render_mode="rgb_array" optional
obs, info = env.reset(seed=0, options={"stage": "flat"})
obs, reward, terminated, truncated, info = env.step(action)   # action in [-1, 1]^42
```

The env wraps one `Simulation` on `ProceduralTerrain`, plus the thorax-shove
`Perturbation` with an `AutoPerturber` and a `FallDetector`. There is no logging and no
rendering. The simulation is built once, in about 0.2 s. Each episode reuses it through
`sim.reset()`. Every terrain difficulty has the same geom pool size, so on reset the env
just swaps in a new `TerrainGenerator` (new seed and difficulty). ProceduralTerrain's
pre-reset hook then lays the chunks out again. **The model is never rebuilt, not even
when the curriculum stage changes.**

### Control and action

* The baseline (FlyGym hybrid controller with heading hold) still runs **every physics
  step** (1e-4 s), exactly as in the app.
* The policy acts every `control_every_steps = 50` physics steps, which is **200 Hz**
  (5 ms). One ~83 ms tripod cycle is then about 17 policy steps. That is fine enough to
  reshape swing and stance within a step, and it keeps episodes short in step count.
  (At 100 Hz a cycle would be only 8 decisions. 500 Hz would make every env step
  2.5x more expensive per sim second and stretch credit assignment.) Policy overhead is
  negligible, so env throughput per sim second barely depends on k. The episode *step
  count* does.
* **Residual:** `ctrl[42 joint targets] = baseline_targets(t) + action_scale * a`, with
  `action_scale = 0.15` rad. `a` is held between policy steps and the baseline keeps
  updating every physics step. It is applied through the controller's own actuator
  write (`LocomotionController.apply`). The env replaces the controller instance's
  `step_and_apply` with a wrapper, so no core file changes.
  **`a == 0` is bit-identical to the baseline.** The residual addition is skipped
  when the residual is zero. A test compares qpos/qvel/ctrl after every policy step
  against a plain Simulation + terrain + hits with the same seeds, on normal terrain
  with hits.
* Optional adhesion residual (`adhesion_residual=True`, off by default): 6 extra
  actions, `adhesion = clip(baseline_on_off + adhesion_scale * a, 0, 1)`.
* `baseline="stand"` swaps the gait for the neutral standing pose. It is used by the
  "standing still" reward test.

### Observation (156 floats, Box(-10, 10), float32; proprioceptive only)

| key | n | scaling |
|---|---|---|
| `gravity_body` | 3 | unit vector of world -z in the thorax frame (upright: (0, 0, -1)) |
| `heading_err_sincos` | 2 | sin/cos of yaw minus the target heading (+x). No absolute x/y |
| `angvel_body` | 3 | rad/s / 10 (walking std 6-7.5) |
| `linvel_body` | 3 | mm/s / 20 (walking: 14 mm/s forward, std 7-13) |
| `height` | 1 | (thorax z - local ground - 1.1 mm) / 0.2 mm. Semi-privileged (uses the terrain ray cast); `obs_height=False` removes it |
| `joint_pos` | 42 | (q - neutral pose) / 0.3 rad (walking std 0.06-0.37) |
| `joint_vel` | 42 | rad/s / 25 (walking std 7-36) |
| `leg_contact` | 6 | 0/1 per leg (any segment touching terrain, `ContactClassifier`) |
| `cpg_phase_sincos` | 12 | sin, cos of the 6 CPG oscillator phases |
| `prev_action` | 42 | last clipped action |

Scales come from 3 s of baseline walking on normal terrain, so walking values are O(1).
Falls and hits give O(10) values, which are clipped. For training, `train_ppo.py` also
applies SB3 `VecNormalize` (running mean/std, saved alongside every checkpoint).
**Hits are never in the observation.** Hit events, `hit_active` and the terrain type
are reported in `info` only. A test writes a large `xfrc_applied` and checks that the
observation of that state does not change. The policy can only feel a hit through its
consequences.

### Reward (`RewardConfig`)

Terms are rates per simulated second, multiplied by the policy dt (0.005 s). The
weights therefore don't depend on `control_every_steps`. `info["reward_terms"]` has
the per-step contribution of every term.

| term | weight | definition |
|---|---|---|
| velocity | +2.0 /s | `exp(-((v_fwd - 14) / 5)^2)`. v_fwd = displacement along the target heading over the last 0.1 s (about one stride; instantaneous speed swings 0-34 mm/s within a stride) |
| upright | +0.5 /s | `max(0, up_z)` (world-z component of the thorax up axis) |
| alive | +0.5 /s | while the FallDetector state is not FALLEN |
| energy | -0.5 /s | `mean(a^2)` |
| action_rate | -0.5 /s | `mean((a - a_prev)^2)` |
| instability | -0.2 /s | `(wx^2 + wy^2) / 10^2`, body roll/pitch rates (walking: about -0.2/s) |
| fallen | -3.0 /s | while FALLEN |
| backward | -2.0 /s | `max(0, -v_fwd) / 14` |
| recovery | +5 once | FallDetector `recovered` event |
| termination | -10 once | terminated after staying FALLEN |
| instability termination | -50 once | `SimulationInstabilityError` |

Standing still is never optimal. The best a standing fly can get is upright + alive =
**1.0 /s**. The baseline gait earns **2.7 /s** (measured: velocity about 1.8, upright
0.5, alive 0.5, instability -0.2). A standing fly actually gets 0.8 /s, because the
FallDetector soon calls a stalled fly FALLEN ("no progress"). The test
`test_standing_still_earns_less_than_walking` checks both numbers.

### Termination / truncation

* `terminated`: continuously FALLEN for more than `fall_terminate_after_s` (2 s). The
  timer restarts on "recovering" and "relapse". This is training-only
  (`terminate_on_fall=True`). The app's fall semantics are untouched.
* `terminated` with `info["instability"] = True`: `SimulationInstabilityError`. The
  full diagnostics are logged at ERROR level *and* printed to stderr. The observation
  is zeros, the reward is -50, and `reset()` recovers.
* `truncated`: `max_episode_s` (20 s of sim time, 4000 policy steps).
* `info["episode_stats"]` on the last step: sim time, path, forward distance, speed,
  hits, falls, recoveries, and whether the fly was fallen at the end.

### Seeding

`reset(seed=s)` seeds the env RNG. Each episode draws 4 sub-seeds from it: terrain
layout, CPG initial phases (controller seed), `Perturbation` RNG and `AutoPerturber` RNG.
They are returned in `info["seeds"]`. Later `reset()` calls without a seed draw new
sub-seeds, so every episode differs but the whole sequence is reproducible. Tests
check that the same seed gives identical rollouts, both in a reused env and in a fresh
one. `EnvConfig.terrain_seed` / `controller_seed` can pin those two.

### Curriculum (`default_curriculum()`, `env.set_stage(i | name)` or `reset(options={"stage": ...})`)

| # | name | terrain | hits (levels, weights) | interval |
|---|---|---|---|---|
| 0 | flat | flat | none | |
| 1 | flat_gentle | flat | gentle | 2-5 s |
| 2 | flat_medium | flat | gentle/medium 50/50 | 2-4 s |
| 3 | easy_terrain | easy | gentle/medium 60/40 | 2-5 s |
| 4 | normal | normal | gentle/medium 60/40 | 2-5 s |
| 5 | aggressive | hard | gentle/medium/hard 30/40/30 | 1.5-4 s |

Hit directions follow `AutoPerturbConfig`'s default mix (mostly lateral, some
forward/backward/up/random). A stage change takes effect at the next reset.
`train_ppo.py` advances the stage when, over the last 30 finished episodes at the
current stage, at least 80 % reached the time limit without a fall-termination *and*
the mean forward speed was at least 0.7 x 14 mm/s. The log goes to `curriculum.csv`.

## Commands

```bash
# smoke test (what was run): 2 envs, 4096 steps, checkpoints + eval + curriculum exercised
.venv/bin/python scripts/train_ppo.py --timesteps 4096 --n-envs 2 --n-steps 1024 --batch-size 256 \
    --max-episode-s 2 --curriculum-window 3 --checkpoint-every 2048 --eval-every 2048 \
    --eval-seconds 3 --tensorboard --out runs/rl/smoke

# first real run
.venv/bin/python scripts/train_ppo.py --timesteps 10000000 --n-envs 8 --tensorboard
tensorboard --logdir runs/rl

# evaluation
.venv/bin/python scripts/eval_policy.py --baseline --stage normal --sim-seconds 180 --seed 0
.venv/bin/python scripts/eval_policy.py --checkpoint runs/rl/<ts>/final_model.zip --stage normal --sim-seconds 180
```

`train_ppo.py` writes `runs/rl/<timestamp>/`: `env_config.json`, `train_args.json`,
`checkpoints/model_<t>.zip` + `vecnormalize_<t>.pkl`, `final_model.zip`,
`vecnormalize.pkl`, `curriculum.csv`, `eval.csv`, and `tb/` (with `--tensorboard`).
`eval_policy.py` picks up the matching VecNormalize file and `env_config.json`
(control rate, action scale, obs layout) automatically.

Smoke-test evidence (`runs/rl/smoke/`): 2 SubprocVecEnvs, 2 PPO iterations, 4096
steps in 30-47 s, depending on machine load. The first update's approx_kl was 0.07 with the final defaults. The curriculum advanced 0 -> 1 -> 2 (on the short 2 s smoke episodes).
Both checkpoints were saved with VecNormalize stats. The periodic eval ran (3 s,
normal stage, 1 hit survived). `eval_policy.py --checkpoint runs/rl/smoke/final_model.zip`
loads it and runs.

## Throughput (Apple M1, 4 performance + 4 efficiency cores, 8 logical CPUs)

Measured with `EnvConfig` defaults (k = 50, 200 Hz policy). The machine was **shared
with another agent's jobs** during most measurements (load average 5-10), so treat the
vectorised numbers as lower bounds.

| setup | env steps/s | physics steps/s | sim s per wall s |
|---|---|---|---|
| 1 env, walking, quiet machine | 130-135 | 6.5-6.8k | 0.65-0.68 |
| 1 env, walking, loaded machine | 70-125 | 3.5-6.2k | 0.35-0.62 |
| 1 env, fly lying on its back (many body contacts) | about 0.5x walking | | |
| SubprocVecEnv 1 env (loaded) | 118 | 5.9k | |
| SubprocVecEnv 2 envs | 219 | 10.9k | |
| SubprocVecEnv 4 envs | 380 | 19.0k | |
| SubprocVecEnv 6 envs | 380 | 19.0k | |
| SubprocVecEnv 8 envs | 457 | 22.9k | |
| PPO end-to-end, 2 envs (smoke, incl. updates + eval) | 136 | | |

The physics step (`mj_step` about 90 µs plus about 45 µs of controller) is the whole
cost. Policy inference and the observation are negligible at 200 Hz. Scaling is good
up to the 4 performance cores and flatter after that, since the efficiency cores are
about 2-3x slower.

**Training-time estimate** (8 envs, about 400-450 env steps/s raw, about 20 % PPO
update overhead, so about 350 steps/s effective):

| policy steps | sim time experienced | wall time |
|---|---|---|
| 1M | 1.4 h | about 50 min |
| 5M | 7 h | about 4 h |
| 10M | 14 h | about 8 h |
| 20M | 28 h | about 16 h |

A quiet machine should be about 20-30 % faster. A 5-10M step run is an overnight job.
Much larger budgets need a many-core Linux box. Throughput scales linearly with
physical cores, and there is no GPU dependence (the MLP runs on CPU).

## Baseline (zero residual) evaluation

`eval_policy.py --baseline`, 180 sim s per run. A failure means FALLEN for more than
5 s, followed by an explicit reset. JSON is in `runs/rl/baseline_eval_*.json`.

| metric | normal + gentle/medium hits, seed 0 | same, seed 1 | flat + gentle/medium 50/50 |
|---|---|---|---|
| episodes / failures | 3 / 2 | 4 / 3 | 6 / 5 |
| mean time to failure (total time / failures) | 90 s | 60 s | 36 s |
| longest uninterrupted run | 68.7 s | 86.2 s | 54.5 s |
| falls / recoveries | 3 / 1 (33 %) | 3 / 0 (0 %) | 5 / 0 (0 %) |
| falls per km (forward distance) | 1344 | 1318 | 2319 |
| hits survived | 49 / 52 (94 %) | 49 / 52 (94 %) | 56 / 61 (92 %) |
| avg forward speed | 12.4 mm/s | 12.7 mm/s | 12.0 mm/s |
| P(survive 5 / 10 / 30 / 60 s) | 1.0 / 1.0 / 1.0 / 0.33 | 1.0 / 0.75 / 0.5 / 0.33 | 0.67 / 0.5 / 0.5 / 0.0 |
| instabilities | 0 | 0 | 0 |

The baseline survives about 93 % of gentle/medium hits. Almost every fall is permanent:
there is no righting reflex. That gives a mean time to failure of roughly 1 minute and
about 1300 falls per km on normal terrain. Walking uninterrupted, it averages 14 mm/s.
The failures, not the terrain, pull the averages down to 12-12.7 mm/s. Flat ground
with more medium hits is *harder* than normal terrain with fewer (the hit mix
dominates). The numbers rest on only 3-5 failures per run, so they are rough. For a
policy comparison, use 10+ minutes of sim time per condition, fixed seeds, and the same
seeds for baseline and policy.

## Recommendations for the first real training run

1. `--n-envs 8 --timesteps 10000000` overnight with the defaults: PPO, MLP 256x256 for
   both pi and vf, lr 1e-4, n_steps 2048 per env (16k-sample rollouts), batch 512,
   10 epochs, gamma 0.995 (1 s horizon at 200 Hz), GAE 0.95, clip 0.2,
   log_std_init -1 (initial exploration about 0.055 rad), target_kl 0.1, VecNormalize on obs
   and reward. In the smoke test a target_kl of 0.03 stopped the very first epoch: KL
   is summed over 42 dimensions, so keep it loose.
2. Start at stage 0 with the automatic curriculum. Check early on in TensorBoard that
   `episode/forward_speed` stays around 13-14 and `episode/success` around 1 on flat
   ground. If the policy first *degrades* the gait, lower `--action-scale` to 0.1 or
   `--log-std-init` to -1.5.
3. What matters is recovery after medium/hard hits and the "no righting reflex"
   failure. A 0.15 rad residual on joint *targets* may be too small to right a fly on
   its back. If the stage 4/5 success rate plateaus, try a larger action scale (0.3)
   and/or `--adhesion-residual` (releasing adhesion is probably needed to roll over),
   and a longer `--fall-terminate-after` (3-5 s) so the policy gets to practise getting
   up.
4. Compare policy and baseline with `eval_policy.py` on identical seeds, at stages 4
   and 5, for at least 10 min of sim time each. The periodic `eval.csv` (60 s) is only
   a trend indicator.
5. If sample efficiency is poor, the next levers are: an observation history (the last
   2-3 frames) or a recurrent policy (RecurrentPPO in sb3-contrib) to feel the hit
   through velocity changes; asymmetric actor-critic (hit info for the critic only);
   and reward shaping for righting (bonus on `up_z` while FALLEN).
