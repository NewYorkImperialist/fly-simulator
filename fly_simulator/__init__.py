"""Fly Simulator: a NeuroMechFly (FlyGym 2.x) that jogs forever."""

from fly_simulator.config import AppConfig
from fly_simulator.simulation import Simulation, SimulationInstabilityError

__version__ = "0.1.0"
__all__ = ["AppConfig", "Simulation", "SimulationInstabilityError", "__version__"]
