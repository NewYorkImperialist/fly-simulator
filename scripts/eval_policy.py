"""Long headless evaluation of a residual policy (or the zero-residual baseline).

    # baseline (a = 0): the FlyGym hybrid controller alone
    .venv/bin/python scripts/eval_policy.py --baseline --stage normal --sim-seconds 120

    # a PPO checkpoint from scripts/train_ppo.py (VecNormalize stats are picked up
    # from the same folder automatically)
    .venv/bin/python scripts/eval_policy.py --checkpoint runs/rl/<ts>/final_model.zip

Reports mean time to failure, falls/km, recovery %, hits survived, average speed,
P(survive T). A failure = FALLEN for longer than --fall-terminate-after (then an
explicit reset); falls the fly recovers from in time are counted but not failures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_simulator.rl.env import EnvConfig, FlySimulatorEnv  # noqa: E402
from fly_simulator.rl.evaluation import evaluate, format_report, zero_policy  # noqa: E402


def load_policy(checkpoint: Path, vecnorm: Path | None, env: FlySimulatorEnv):
    from stable_baselines3 import PPO

    model = PPO.load(str(checkpoint), device="cpu")
    norm = None
    if vecnorm is None:
        for cand in (checkpoint.with_name(checkpoint.stem.replace("model", "vecnormalize") + ".pkl"),
                     checkpoint.parent / "vecnormalize.pkl"):
            if cand.exists():
                vecnorm = cand
                break
    if vecnorm is not None and vecnorm.exists():
        import pickle

        with open(vecnorm, "rb") as fh:
            norm = pickle.load(fh)  # VecNormalize (venv not pickled)
        norm.training = False
        print(f"obs normalisation from {vecnorm}")
    elif vecnorm is not None:
        raise FileNotFoundError(vecnorm)

    def policy(obs: np.ndarray) -> np.ndarray:
        o = norm.normalize_obs(obs[None]) if norm is not None else obs[None]
        a, _ = model.predict(o, deterministic=True)
        return a[0]

    return policy


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--baseline", action="store_true", help="zero residual (hybrid controller)")
    src.add_argument("--checkpoint", type=Path, help="SB3 PPO .zip")
    p.add_argument("--vecnormalize", type=Path, default=None, help="VecNormalize .pkl")
    p.add_argument("--stage", default="normal", help="curriculum stage name or index")
    p.add_argument("--sim-seconds", type=float, default=120.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fall-terminate-after", type=float, default=5.0,
                   help="sim seconds FALLEN before the episode counts as a failure")
    p.add_argument("--control-every", type=int, default=None,
                   help="physics steps per policy step (must match training)")
    p.add_argument("--action-scale", type=float, default=None)
    p.add_argument("--json", type=Path, default=None, help="write the metrics here")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    cfg = EnvConfig(fall_terminate_after_s=args.fall_terminate_after)
    if args.checkpoint is not None:
        env_json = args.checkpoint.parent / "env_config.json"
        if env_json.exists():  # same control rate / scale / obs layout as training
            saved = json.loads(env_json.read_text())
            for k in ("control_every_steps", "action_scale", "adhesion_residual",
                      "adhesion_scale", "obs_height"):
                if k in saved:
                    setattr(cfg, k, saved[k])
    if args.control_every is not None:
        cfg.control_every_steps = args.control_every
    if args.action_scale is not None:
        cfg.action_scale = args.action_scale
    stage = int(args.stage) if args.stage.isdigit() else args.stage
    env = FlySimulatorEnv(cfg)
    policy = (zero_policy(env.n_actions) if args.baseline
              else load_policy(args.checkpoint, args.vecnormalize, env))
    who = "baseline (zero residual)" if args.baseline else str(args.checkpoint)
    print(f"evaluating {who}: stage {stage}, {args.sim_seconds:g} sim s, seed {args.seed}, "
          f"policy {1 / env.dt:.0f} Hz", flush=True)
    res = evaluate(env, policy, args.sim_seconds, seed=args.seed, stage=stage,
                   verbose=not args.quiet)
    print(format_report(res))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2, default=float))
        print(f"wrote {args.json}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
