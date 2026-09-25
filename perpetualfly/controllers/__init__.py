"""Locomotion controllers (wrappers around FlyGym's CPG / hybrid controllers)."""

from perpetualfly.controllers.hybrid import (
    FastHybridController,
    FastHybridTurningController,
    FastObservationBuilder,
    LocomotionController,
)

__all__ = ["FastHybridController", "FastHybridTurningController", "FastObservationBuilder", "LocomotionController"]
