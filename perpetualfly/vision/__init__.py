"""Geometric vision for the connectome brain (docs/VISION.md).

``looming``: angular size / expansion rate of registered objects (the whip, and any
future moving hazard) on each compound eye, mapped to the looming-detector neurons
LC4 and LPLC2 of that eye (``StimulusEvent(kind="loom")``).
"""

from .looming import (
    EyeLoom,
    LoomingConfig,
    LoomingVision,
    LoomResponse,
    VisualSource,
    install_whip_vision,
    loom_response,
)

__all__ = [
    "EyeLoom",
    "LoomingConfig",
    "LoomingVision",
    "LoomResponse",
    "VisualSource",
    "install_whip_vision",
    "loom_response",
]
