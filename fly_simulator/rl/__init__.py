"""Residual-RL Gymnasium env (spec phase 10). Needs the ``rl`` extra:
``uv pip install --python .venv/bin/python -e ".[rl,dev]"``. The core package never
imports this subpackage, so gymnasium / SB3 / torch stay optional."""

from fly_simulator.rl.env import (
    CurriculumStage,
    EnvConfig,
    FlySimulatorEnv,
    RewardConfig,
    default_curriculum,
)

__all__ = ["CurriculumStage", "EnvConfig", "FlySimulatorEnv", "RewardConfig",
           "default_curriculum"]
