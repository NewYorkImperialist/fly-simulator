"""Neuropil atlas (outlines + centroids in a frontal 2D projection) and colour tables.

The atlas JSON (``assets/flywire_neuropils_frontal.json``) is generated once by
``build_atlas.py`` from real FlyWire neuropil meshes. Format (all coordinates in
micrometres of FlyWire space, projected along z: ``u = x_nm/1000``, ``v = y_nm/1000``;
``*_L`` neuropils are on the image left, dorsal is up)::

    {
      "format": "perpetualfly-neuropil-atlas-v1",
      "bounds": [u_min, v_min, u_max, v_max],
      "brain_outline": [ [[u, v], ...], ... ],          # silhouette polygon(s)
      "regions": {
        "AL_L": {"centroid": [u, v], "depth": z_um, "area_um2": a,
                  "group": "AL", "outline": [ [[u, v], ...], ... ]},
        ...
      }
    }

The brain engine can reuse ``centroid`` as ``BrainLayout.region_xy`` and project
FlyWire neuron positions (nm) the same way (``x/1000, y/1000``); the window does not
require it (it fits whatever coordinates it gets onto the atlas, see window.py).
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
ATLAS_PATH = ASSETS / "flywire_neuropils_frontal.json"

# Neuropil -> super-region (FlyWire / Ito 2014 hierarchy), used for grouping and labels.
_GROUPS = {
    "OL": ("LA", "ME", "AME", "LO", "LOP"),
    "AL": ("AL",),
    "MB": ("MB_CA", "MB_PED", "MB_VL", "MB_ML"),
    "LH": ("LH",),
    "CX": ("FB", "EB", "PB", "NO"),
    "LX": ("LAL", "BU", "GA"),
    "SNP": ("SMP", "SIP", "SLP"),
    "VLNP": ("AVLP", "PVLP", "PLP", "WED", "AOTU"),
    "INP": ("CRE", "SCL", "ICL", "IB", "ATL"),
    "VMNP": ("VES", "EPA", "GOR", "SPS", "IPS"),
    "PENP": ("AMMC", "SAD", "FLA", "CAN", "PRW"),
    "GNG": ("GNG",),
    "OCG": ("OCG",),
}
_BASE_TO_GROUP = {b: g for g, bases in _GROUPS.items() for b in bases}
GROUP_ORDER: tuple[str, ...] = tuple(_GROUPS)

_ALIASES = {"SEZ": "GNG", "CA": "MB_CA", "PED": "MB_PED", "VL": "MB_VL", "ML": "MB_ML",
            "MB_CALYX": "MB_CA", "PCB": "PB", "FSB": "FB"}

# Colours are BGR (OpenCV). Neurotransmitter colours are used everywhere (dots, bars,
# raster, legend). Chosen for separability on a near-black background.
NT_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "ACh": (80, 200, 255),    # amber / yellow-orange  (excitatory, most neurons)
    "GABA": (255, 150, 70),   # blue                    (inhibitory)
    "Glu": (120, 230, 110),   # green                   (mostly inhibitory in flies)
    "DA": (80, 80, 255),      # red                     (dopamine)
    "5HT": (255, 110, 230),   # magenta / violet        (serotonin)
    "OA": (230, 240, 90),     # cyan                    (octopamine)
}
UNKNOWN_NT_BGR = (170, 170, 170)
NT_DISPLAY = {"ACh": "ACh", "GABA": "GABA", "Glu": "Glu", "DA": "DA", "5HT": "5-HT",
              "OA": "OA"}


def normalize_region_name(name: str) -> str:
    """'mb-ca (L)' / 'MB_CA_L' / 'SEZ' -> canonical atlas key where possible."""
    s = re.sub(r"[\s\-()\[\]]+", "_", str(name).strip().upper()).strip("_")
    s = re.sub(r"_+", "_", s)
    side = ""
    m = re.match(r"^(.+?)_(L|R|LEFT|RIGHT)$", s)
    if m:
        s, side = m.group(1), "_" + m.group(2)[0]
    s = _ALIASES.get(s, s)
    return s + side


def region_base(name: str) -> str:
    n = normalize_region_name(name)
    return n[:-2] if n.endswith(("_L", "_R")) else n


def region_side(name: str) -> str:
    """'L', 'R' or '' (midline / unknown)."""
    n = normalize_region_name(name)
    return n[-1] if n.endswith(("_L", "_R")) else ""


def region_group(name: str) -> str:
    return _BASE_TO_GROUP.get(region_base(name), "other")


@functools.lru_cache(maxsize=4)
def load_atlas(path: str | None = None) -> dict | None:
    """Load the atlas JSON (cached). Returns None if missing/unreadable."""
    p = Path(path) if path else ATLAS_PATH
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    regions = {}
    for name, r in raw.get("regions", {}).items():
        regions[name] = {
            "centroid": np.asarray(r["centroid"], np.float64),
            "depth": float(r.get("depth", 0.0)),
            "area_um2": float(r.get("area_um2", 0.0)),
            "group": r.get("group", region_group(name)),
            "outline": [np.asarray(o, np.float64) for o in r.get("outline", [])
                        if len(o) >= 3],
        }
    return {
        "bounds": np.asarray(raw.get("bounds", [0, 0, 1, 1]), np.float64),
        "brain_outline": [np.asarray(o, np.float64) for o in raw.get("brain_outline", [])
                          if len(o) >= 3],
        "regions": regions,
    }


def atlas_region_xy(names: list[str], atlas: dict | None = None) -> np.ndarray:
    """Atlas centroids (um) for ``names`` (NaN where the atlas has no such region)."""
    atlas = atlas if atlas is not None else load_atlas()
    out = np.full((len(names), 2), np.nan)
    if atlas is None:
        return out
    for i, n in enumerate(names):
        r = atlas["regions"].get(normalize_region_name(n))
        if r is not None:
            out[i] = r["centroid"]
    return out
