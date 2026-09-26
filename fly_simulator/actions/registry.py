"""Name / key -> action factory, for the app's key handler and brain triggers."""

from __future__ import annotations

from .base import Action
from .behaviours import BackAway, Freeze, Groom, ProboscisExtend, TurnInPlace, WingRaise
from .jump import Jump

# App keys (fly_simulator/app.py ACTION_KEY_MAP binds these, with durations).
ACTION_KEYS: dict[str, str] = {
    "j": "jump",
    "y": "groom",
    "z": "freeze",
    "w": "wings",
    "n": "proboscis",
    "e": "back_away",
    ",": "turn_left",
    ".": "turn_right",
}

_FACTORIES = {
    "jump": Jump,
    "freeze": Freeze,
    "groom": Groom,
    "back_away": BackAway,
    "turn_left": lambda **kw: TurnInPlace("left", **kw),
    "turn_right": lambda **kw: TurnInPlace("right", **kw),
    "wings": WingRaise,
    "proboscis": ProboscisExtend,
}

#: actions that need the extra joints from make_action_fly_factory()
NEEDS_EXTRA_JOINTS = {"wings", "proboscis"}


def make_action(name: str, **params) -> Action:
    """``make_action("jump", boost=2.0)``, ``make_action("freeze", duration=2)``..."""
    try:
        return _FACTORIES[name](**params)
    except KeyError:
        raise KeyError(f"unknown action {name!r}; known: {sorted(_FACTORIES)}") from None


def available_actions(sim) -> list[str]:
    """Action names usable on ``sim`` (wings / proboscis need the extra joints)."""
    import mujoco as mj

    m = sim.model
    have = {
        "wings": mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR,
                               f"{sim.fly_name}/c_thorax-l_wing-yaw-wingpos") >= 0,
        "proboscis": mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR,
                                   f"{sim.fly_name}/c_head-c_rostrum-pitch-proboscispos") >= 0,
    }
    return [n for n in _FACTORIES if have.get(n, True)]
