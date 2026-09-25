"""PPO (Stable-Baselines3) on the residual PerpetualFly env.

    # smoke test (a few thousand steps, 2 envs): proves the pipeline end to end
    .venv/bin/python scripts/train_ppo.py --timesteps 4096 --n-envs 2 --n-steps 1024 \
        --eval-every 0 --checkpoint-every 2048

    # first real run (see docs/RL.md for the reasoning behind the defaults)
    .venv/bin/python scripts/train_ppo.py --timesteps 10_000_000 --n-envs 8 --tensorboard

Output: runs/rl/<YYYY-MM-DD_HH-MM-SS>/ with env_config.json, train_args.json,
checkpoints/ (model + VecNormalize stats), final_model.zip, vecnormalize.pkl,
curriculum.csv, eval.csv, optional tb/ (TensorBoard).

Curriculum: starts at --stage, and moves to the next stage once the last
--curriculum-window finished episodes at the current stage have a success rate
(episode reached the time limit without a fall-termination) >= --curriculum-success
and mean forward speed >= --curriculum-speed-frac * target speed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CallbackList  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

from perpetualfly.rl.env import EnvConfig, PerpetualFlyEnv  # noqa: E402
from perpetualfly.rl.evaluation import evaluate, format_report  # noqa: E402
from perpetualfly.rl.wrappers import make_vec_env  # noqa: E402


class CurriculumCallback(BaseCallback):
    def __init__(self, start_stage: int, stage_names: list[str], window: int, success: float,
                 speed: float, log_path: Path, verbose: int = 1):
        super().__init__(verbose)
        self.stage = start_stage
        self.stage_names = stage_names
        self.n_stages = len(stage_names)
        self.window, self.success, self.speed = window, success, speed
        self.recent: deque = deque(maxlen=window)
        self.log_path = log_path
        with open(log_path, "w", newline="") as fh:
            csv.writer(fh).writerow(["timesteps", "stage", "success_rate", "mean_speed"])

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            st = info.get("episode_stats")
            if not done or st is None:
                continue
            # success = reached the time limit (truncated), not fall-terminated
            ok = bool(info.get("TimeLimit.truncated", False)) and not st["fallen_at_end"]
            self.recent.append((st["stage"], ok, st["avg_forward_speed"]))
            self.logger.record_mean("episode/forward_speed", st["avg_forward_speed"])
            self.logger.record_mean("episode/falls", st["n_falls"])
            self.logger.record_mean("episode/hits", st["n_hits"])
            self.logger.record_mean("episode/success", float(ok))
        self.logger.record("curriculum/stage", self.stage)
        cur = [r for r in self.recent if r[0] == self._stage_name()]
        if len(cur) >= self.window and self.stage < self.n_stages - 1:
            rate = float(np.mean([r[1] for r in cur]))
            spd = float(np.mean([r[2] for r in cur]))
            if rate >= self.success and spd >= self.speed:
                self.stage += 1
                self.training_env.env_method("set_stage", self.stage)
                self.recent.clear()
                with open(self.log_path, "a", newline="") as fh:
                    csv.writer(fh).writerow([self.num_timesteps, self.stage, rate, spd])
                if self.verbose:
                    print(f"[curriculum] t={self.num_timesteps}: success {rate:.2f}, speed "
                          f"{spd:.1f} -> stage {self.stage} ({self._stage_name()})", flush=True)
        return True

    def _stage_name(self) -> str:
        return self.stage_names[self.stage]


class SaveCallback(BaseCallback):
    """Model + VecNormalize stats every ``every`` timesteps."""

    def __init__(self, every: int, folder: Path):
        super().__init__()
        self.every, self.folder, self._next = every, folder, every

    def _on_step(self) -> bool:
        if self.every > 0 and self.num_timesteps >= self._next:
            self._next += self.every
            self.folder.mkdir(parents=True, exist_ok=True)
            path = self.folder / f"model_{self.num_timesteps}.zip"
            self.model.save(path)
            vn = self.model.get_vec_normalize_env()
            if vn is not None:
                vn.save(str(self.folder / f"vecnormalize_{self.num_timesteps}.pkl"))
            print(f"[checkpoint] {path}", flush=True)
        return True


class LongEvalCallback(BaseCallback):
    """Every ``every`` timesteps: deterministic policy for ``sim_seconds`` in a separate
    in-process env at ``stage``; logs the spec's long-horizon metrics."""

    def __init__(self, every: int, env_cfg: EnvConfig, stage, sim_seconds: float,
                 csv_path: Path, seed: int = 10_000):
        super().__init__()
        self.every, self._next = every, every
        self.env_cfg, self.stage, self.sim_seconds, self.seed = env_cfg, stage, sim_seconds, seed
        self.csv_path = csv_path
        self._header = False

    def _on_step(self) -> bool:
        if self.every <= 0 or self.num_timesteps < self._next:
            return True
        self._next += self.every
        env = PerpetualFlyEnv(self.env_cfg)
        vn = self.model.get_vec_normalize_env()

        def policy(obs):
            o = vn.normalize_obs(obs[None]) if vn is not None else obs[None]
            return self.model.predict(o, deterministic=True)[0][0]

        r = evaluate(env, policy, self.sim_seconds, seed=self.seed, stage=self.stage,
                     verbose=False)
        env.close()
        print(f"[eval] t={self.num_timesteps}\n{format_report(r)}", flush=True)
        row = {"timesteps": self.num_timesteps}
        for k in ("mean_time_to_failure_s", "falls_per_km_forward", "recovery_percentage",
                  "hits_survived_fraction", "avg_forward_speed_mm_s", "failures", "n_hits"):
            row[k] = r[k]
            if r[k] is not None:
                self.logger.record(f"eval/{k}", r[k])
        for h, v in r["p_survive"].items():
            row[f"p_survive_{h}"] = v
        with open(self.csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if not self._header:
                w.writeheader()
                self._header = True
            w.writerow(row)
        return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps", type=int, default=5_000_000, help="policy (env) steps")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", default="0", help="initial curriculum stage (index or name)")
    p.add_argument("--no-curriculum", action="store_true", help="stay at --stage")
    p.add_argument("--curriculum-window", type=int, default=30)
    p.add_argument("--curriculum-success", type=float, default=0.8)
    p.add_argument("--curriculum-speed-frac", type=float, default=0.7)
    # env
    p.add_argument("--control-every", type=int, default=50, help="physics steps per policy step")
    p.add_argument("--action-scale", type=float, default=0.15, help="rad at |a| = 1")
    p.add_argument("--max-episode-s", type=float, default=20.0)
    p.add_argument("--fall-terminate-after", type=float, default=2.0)
    p.add_argument("--adhesion-residual", action="store_true")
    # PPO
    p.add_argument("--n-steps", type=int, default=2048, help="rollout length per env")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.995, help="0.995 at 200 Hz = 1 s horizon")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--target-kl", type=float, default=0.1,
                   help="KL is summed over 42 action dims; SB3 stops the epoch at 1.5x")
    p.add_argument("--log-std-init", type=float, default=-1.0,
                   help="initial exploration std exp(-1) = 0.37 -> 0.055 rad at scale 0.15")
    p.add_argument("--net", default="256,256", help="hidden sizes (pi and vf)")
    p.add_argument("--device", default="cpu")
    # io
    p.add_argument("--out", type=Path, default=None, help="default runs/rl/<timestamp>")
    p.add_argument("--checkpoint-every", type=int, default=200_000)
    p.add_argument("--eval-every", type=int, default=500_000, help="0 = off")
    p.add_argument("--eval-seconds", type=float, default=60.0)
    p.add_argument("--eval-stage", default="normal")
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--resume", type=Path, default=None, help="continue from a model .zip")
    args = p.parse_args(argv)

    out = args.out or Path("runs/rl") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out.mkdir(parents=True, exist_ok=True)
    stage = int(args.stage) if args.stage.isdigit() else args.stage
    cfg = EnvConfig(
        control_every_steps=args.control_every, action_scale=args.action_scale,
        max_episode_s=args.max_episode_s, fall_terminate_after_s=args.fall_terminate_after,
        adhesion_residual=args.adhesion_residual, stage=stage,
    )
    env_dict = {k: v for k, v in asdict(cfg).items() if k != "app"}
    (out / "env_config.json").write_text(json.dumps(env_dict, indent=2, default=str))
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    venv = make_vec_env(cfg, n_envs=args.n_envs, seed=args.seed)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=args.gamma)
    probe = PerpetualFlyEnv(cfg)
    start_stage = probe.stage_index
    print(f"out: {out}\nenvs: {args.n_envs}, policy rate {1 / probe.dt:.0f} Hz, obs "
          f"{probe.observation_space.shape}, act {probe.action_space.shape}, stage "
          f"{probe.stage.name}", flush=True)
    probe.close()

    hidden = [int(x) for x in args.net.split(",") if x]
    if args.resume:
        model = PPO.load(str(args.resume), env=venv, device=args.device)
    else:
        model = PPO(
            "MlpPolicy", venv, learning_rate=args.lr, n_steps=args.n_steps,
            batch_size=args.batch_size, n_epochs=args.n_epochs, gamma=args.gamma,
            gae_lambda=args.gae_lambda, clip_range=args.clip_range, ent_coef=args.ent_coef,
            target_kl=args.target_kl, seed=args.seed, device=args.device, verbose=1,
            tensorboard_log=str(out / "tb") if args.tensorboard else None,
            policy_kwargs=dict(net_arch=dict(pi=hidden, vf=hidden),
                               log_std_init=args.log_std_init),
        )
    cbs = [SaveCallback(args.checkpoint_every, out / "checkpoints")]
    if not args.no_curriculum:
        cbs.append(CurriculumCallback(start_stage, [st.name for st in cfg.curriculum],
                                      args.curriculum_window,
                                      args.curriculum_success,
                                      args.curriculum_speed_frac * cfg.reward.target_speed,
                                      out / "curriculum.csv"))
    if args.eval_every > 0:
        es = int(args.eval_stage) if args.eval_stage.isdigit() else args.eval_stage
        cbs.append(LongEvalCallback(args.eval_every, cfg, es, args.eval_seconds, out / "eval.csv"))

    t0 = time.perf_counter()
    try:
        model.learn(total_timesteps=args.timesteps, callback=CallbackList(cbs),
                    progress_bar=False, reset_num_timesteps=args.resume is None)
    finally:
        model.save(out / "final_model.zip")
        venv.save(str(out / "vecnormalize.pkl"))
        wall = time.perf_counter() - t0
        print(f"saved {out / 'final_model.zip'} and vecnormalize.pkl; {model.num_timesteps} "
              f"steps in {wall:.0f} s ({model.num_timesteps / wall:.0f} steps/s)", flush=True)
        venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
