"""PerpetualFly: a NeuroMechFly (FlyGym 2.x) that jogs forever."""

from perpetualfly.config import AppConfig
from perpetualfly.simulation import Simulation, SimulationInstabilityError

__version__ = "0.1.0"
__all__ = ["AppConfig", "Simulation", "SimulationInstabilityError", "__version__"]
