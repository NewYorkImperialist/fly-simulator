"""Env factories for vectorised training (SB3 ``SubprocVecEnv`` / ``DummyVecEnv``)."""

from __future__ import annotations

import copy
from typing import Callable

from fly_simulator.rl.env import EnvConfig, FlySimulatorEnv


def make_env(cfg: EnvConfig | None = None, rank: int = 0, seed: int = 0,
             monitor: bool = True, render_mode: str | None = None) -> Callable[[], FlySimulatorEnv]:
    """Picklable thunk building one env; env ``rank`` is seeded with ``seed + rank``.

    ``monitor=True`` wraps it in SB3's ``Monitor`` (episode return / length in
    ``info["episode"]``) and keeps ``episode_stats`` from the env's final info.
    """
    cfg = copy.deepcopy(cfg) if cfg is not None else EnvConfig()

    def _init():
        env = FlySimulatorEnv(cfg, render_mode=render_mode)
        env.reset(seed=seed + rank)
        if monitor:
            from stable_baselines3.common.monitor import Monitor

            env = Monitor(env, info_keywords=("episode_stats",))
        return env

    return _init


def make_vec_env(cfg: EnvConfig | None = None, n_envs: int = 1, seed: int = 0,
                 subprocess: bool | None = None, start_method: str = "spawn"):
    """``n_envs`` envs in a SubprocVecEnv (default when n_envs > 1) or DummyVecEnv.

    ``spawn`` (macOS default) re-imports the package in each worker; each worker
    builds its own MuJoCo model (~0.2-1 s).
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    fns = [make_env(cfg, rank=i, seed=seed) for i in range(n_envs)]
    if subprocess is None:
        subprocess = n_envs > 1
    if subprocess:
        return SubprocVecEnv(fns, start_method=start_method)
    return DummyVecEnv(fns)
