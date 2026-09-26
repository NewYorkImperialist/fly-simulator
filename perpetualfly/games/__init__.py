"""FLY BRAIN PLAYS: games whose controller is the FlyWire connectome brain (docs/GAMES.md).

Brain responses are real connectome wiring (Shiu et al. 2024 LIF model of FlyWire
v783); the game interface (what the brain sees, how its outputs map to controls) is
designed by us.
"""

from .asteroids import (
    DIFFICULTIES,
    AsteroidConfig,
    AsteroidField,
    AsteroidGame,
    GameEvent,
    asteroid_response,
    wave_params,
)
from .brain_io import CONTROLS, GameBrain, GameMapping

HONEST_LABEL = ("Brain responses are real connectome wiring (FlyWire v783, Shiu et al. 2024 "
                "LIF model); the game interface (what the brain sees, how its outputs map to "
                "controls) is designed by us.")

GAMES = ("asteroids",)


def __getattr__(name):  # lazy: session pulls in MuJoCo / FlyGym
    if name in ("AsteroidSession", "make_renderer", "render_frame", "game_looming_config"):
        from . import session

        return getattr(session, name)
    raise AttributeError(name)


__all__ = [
    "CONTROLS", "DIFFICULTIES", "GAMES", "HONEST_LABEL", "AsteroidConfig", "AsteroidField",
    "AsteroidGame", "AsteroidSession", "GameBrain", "GameEvent", "GameMapping",
    "asteroid_response", "game_looming_config", "make_renderer", "render_frame", "wave_params",
]
