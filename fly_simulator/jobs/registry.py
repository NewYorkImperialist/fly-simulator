"""Job registry: ``@register_job`` on an ``EternalJob`` subclass, then
``get_job(name)`` / ``make_job(name, cfg)`` / ``available_jobs()``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fly_simulator.jobs.base import EternalJob, JobConfig

_REGISTRY: dict[str, type["EternalJob"]] = {}


def register_job(cls: type["EternalJob"]) -> type["EternalJob"]:
    """Class decorator: register ``cls`` under ``cls.name``."""
    if not cls.name or cls.name in ("job",):
        raise ValueError(f"{cls.__name__} needs a unique `name`")
    other = _REGISTRY.get(cls.name)
    if other is not None and other is not cls:
        raise ValueError(f"job name {cls.name!r} already registered by {other.__name__}")
    _REGISTRY[cls.name] = cls
    return cls


def _load_builtin() -> None:
    # importing the modules runs their @register_job decorators
    from fly_simulator.jobs import sisyphus  # noqa: F401
    try:
        from fly_simulator.jobs import hamster_wheel  # noqa: F401
    except ImportError:
        pass
    from fly_simulator.jobs import mowing  # noqa: F401
    from fly_simulator.jobs import raking  # noqa: F401
    from fly_simulator.jobs import kebab  # noqa: F401
    from fly_simulator.jobs import dead_hang  # noqa: F401
    from fly_simulator.jobs import bowling  # noqa: F401
    from fly_simulator.jobs import broccoli_toss  # noqa: F401


def available_jobs() -> list[str]:
    _load_builtin()
    return list(_REGISTRY)


def get_job(name: str) -> type["EternalJob"]:
    _load_builtin()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown job {name!r}; available: {', '.join(_REGISTRY)}") from None


def make_job(name: str, cfg: "JobConfig | dict | None" = None) -> "EternalJob":
    """Instantiate a job; ``cfg`` may be its config dataclass or a dict of overrides."""
    cls = get_job(name)
    if isinstance(cfg, dict):
        cfg = cls.config_cls(**cfg)
    return cls(cfg)
