"""Action library: extra body behaviours beyond walking (see docs/ACTIONS.md).

    from fly_simulator.actions import ActionManager, Jump, Freeze, Groom
    mgr = ActionManager(sim)          # attaches hooks to a fly_simulator Simulation
    mgr.trigger(Jump())               # runs, then hands control back to walking
"""

from .base import Action, ActionCommand, ActionEvent, ActionManager, BodyIndex
from .behaviours import (
    BackAway,
    DriveOverride,
    Freeze,
    Groom,
    ProboscisExtend,
    TurnInPlace,
    WingRaise,
    load_grooming_clip,
)
from .body import make_action_fly, make_action_fly_factory
from .jump import Jump, JumpParams
from .registry import ACTION_KEYS, make_action

__all__ = [
    "ACTION_KEYS", "Action", "ActionCommand", "ActionEvent", "ActionManager", "BackAway",
    "BodyIndex", "DriveOverride", "Freeze", "Groom", "Jump", "JumpParams", "ProboscisExtend",
    "TurnInPlace", "WingRaise", "load_grooming_clip", "make_action", "make_action_fly",
    "make_action_fly_factory",
]
