"""Flapping-wing flight prototype for NeuroMechFly (MuJoCo ellipsoid fluid model).

See docs/FLIGHT.md. Opt-in only: nothing here changes the default walking model.
"""

from .body import AIR_DENSITY, AIR_VISCOSITY, FLIGHT_TIMESTEP, apply_air, make_flight_fly, make_flight_fly_factory
from .mode import FlightMode, FlightModeConfig
from .sim import FlightSimulation, TetherResult, measure_tethered
from .wingbeat import WingbeatGenerator, WingbeatParams, WingParams

__all__ = [
    "AIR_DENSITY", "AIR_VISCOSITY", "FLIGHT_TIMESTEP", "apply_air", "make_flight_fly",
    "make_flight_fly_factory", "FlightMode", "FlightModeConfig", "FlightSimulation", "TetherResult", "measure_tethered",
    "WingbeatGenerator", "WingbeatParams", "WingParams",
]
