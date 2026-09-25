"""BrainLayout for the brain window: regions, display-neuron subset, 2D positions.

Coordinates are FlyWire (FAFB14.1) micrometres projected along z (frontal view):
``u = x_nm / 1000``, ``v = y_nm / 1000`` -- the same convention as
``perpetualfly/brain_viz/atlas.py``. In this space the fly's left hemisphere has the
smaller u (``*_L`` neuropils on the image left) and v grows ventrally (flip it to
draw dorsal up).

* regions: every neuropil that is some neuron's primary output neuropil (most
  presynapses; Dorkenwald et al. 2024 per-neuron neuropil counts) + "OTHER" for
  neurons with no presynapses in any neuropil. ``region_xy`` is the median anchor
  position of the neurons assigned to the region (a proxy for the neuropil centroid).
* display neurons: all descending neurons, MN9, the stimulus sets used by
  ``mapping.StimulusMapper`` (capped per set) and a random sample per region,
  stratified by transmitter. Positions are the FlyWire anchor point (on the neuron's
  backbone, inside neuropil), falling back to soma, then region centroid.
"""

from __future__ import annotations

import numpy as np

from .data import NeuronTable
from .mapping import MN9_IDS, descending_indices, named_sets
from .schema import BrainLayout

MODEL_NAME = "Shiu2024-LIF FlyWire v783 (numba port)"

# stimulus sets always shown (cap per set)
DISPLAY_SETS = {"sugar": 50, "bitter": 50, "body_mech": 500, "head_bristle": 200,
                "jo_wind_gravity": 150, "LC4": 110, "LPLC2": 60}


def region_centroids(table: NeuronTable) -> np.ndarray:
    xy = table.pos_um[:, :2].astype(np.float64)
    bad = ~np.isfinite(xy).all(1)
    xy[bad] = table.soma_um[bad, :2]
    out = np.zeros((len(table.regions), 2), dtype=np.float32)
    for r in range(len(table.regions)):
        m = (table.region == r) & np.isfinite(xy).all(1)
        out[r] = np.median(xy[m], axis=0) if m.any() else np.nan
    return out


def choose_display(table: NeuronTable, n_per_region: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = [np.nonzero(table.col("super_class") == "descending")[0],
                                table.index_of(MN9_IDS)]
    for g in descending_indices(table).values():
        chosen.append(g)
    sets = named_sets(table)
    for name, cap in DISPLAY_SETS.items():
        idx = sets[name]
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        chosen.append(idx)
    for r in range(len(table.regions)):
        members = np.nonzero(table.region == r)[0]
        if len(members) <= n_per_region:
            chosen.append(members)
            continue
        # stratify by transmitter: at least one of each NT present, rest proportional
        picks = []
        nts = table.nt[members]
        for k in np.unique(nts):
            picks.append(rng.choice(members[nts == k], 1))
        rest = np.setdiff1d(members, np.concatenate(picks))
        n_more = max(n_per_region - len(picks), 0)
        if n_more and len(rest):
            picks.append(rng.choice(rest, min(n_more, len(rest)), replace=False))
        chosen.append(np.concatenate(picks))
    return np.unique(np.concatenate([np.asarray(c, dtype=np.int64) for c in chosen]))


def build_layout(table: NeuronTable, n_per_region: int = 16, seed: int = 0):
    """Return (BrainLayout, display_idx) where display_idx are model indices."""
    disp = choose_display(table, n_per_region, seed)
    rxy = region_centroids(table)
    xy = table.pos_um[disp, :2].astype(np.float32)
    bad = ~np.isfinite(xy).all(1)
    xy[bad] = table.soma_um[disp[bad], :2]
    bad = ~np.isfinite(xy).all(1)
    xy[bad] = rxy[table.region[disp[bad]]]
    labels = [str(c) for c in table.col("cell_type")[disp]]
    layout = BrainLayout(
        regions=list(table.regions),
        region_xy=rxy,
        region_outline_xy=[],
        display_neuron_ids=table.root_id[disp].astype(np.int64),
        display_neuron_region=table.region[disp].astype(np.int32),
        display_neuron_nt=table.nt[disp].astype(np.int32),
        display_neuron_xy=np.nan_to_num(xy).astype(np.float32),
        display_neuron_label=labels,
        n_neurons_total=int(table.n),
        model_name=MODEL_NAME,
    )
    return layout, disp
