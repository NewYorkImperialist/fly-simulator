"""Vision for the connectome brain (docs/VISION.md).

Real vision (``--real-vision``): ``eyes`` (FlyGym compound eyes -> 721 ommatidia per
eye), ``flyvis_net`` (flyvis visual system stepped per frame; optional 'vision'
extra), ``bridge`` (T4 / T5 -> LC4 / LPLC2 model -> ``loom`` events; eyes-only
dark-expansion fallback), ``objects`` (a looming dark sphere test object). Import
those submodules directly (they are not imported here, so this package stays light).

Geometric vision:

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
