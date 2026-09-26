"""Connectome-based brain model (Shiu et al. 2024 LIF on FlyWire v783) in its own process.

Quick use::

    from fly_simulator.brain import BrainConfig, BrainProcess, StimulusEvent, descending_to_drive

    brain = BrainProcess(BrainConfig())          # spawns the worker
    brain.wait_ready()
    layout = brain.layout()                      # BrainLayout (once)
    brain.send(StimulusEvent("whip_hit", side="left", intensity=0.7))
    state = brain.latest("app")                  # newest BrainState or None (non-blocking)
    drive = descending_to_drive(state) if state else None
    brain.stop()

See docs/BRAIN.md. Heavy imports (numba, pyarrow) happen lazily in the worker.
"""

from .schema import (DESCENDING_GROUPS, NEUROTRANSMITTERS, BrainLayout, BrainState,
                     StimulusEvent)

__all__ = ["DESCENDING_GROUPS", "NEUROTRANSMITTERS", "BrainLayout", "BrainState",
           "StimulusEvent", "BrainConfig", "BrainProcess", "BrainSubscriber",
           "descending_to_drive", "DriveGains"]


def __getattr__(name):
    if name in ("BrainConfig", "BrainProcess", "BrainSubscriber"):
        from . import process
        return getattr(process, name)
    if name in ("descending_to_drive", "DriveGains"):
        from . import mapping
        return getattr(mapping, name)
    raise AttributeError(name)
