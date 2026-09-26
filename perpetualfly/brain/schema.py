"""Messages exchanged between the fly app, the brain process and the brain window.

This module is the contract between three independently developed parts:

* the fly app (``perpetualfly.app``) sends ``StimulusEvent``s (e.g. "whipped on the
  left") and may read ``BrainState.descending`` to steer the walking controller;
* the brain process (``perpetualfly.brain``) runs the connectome model (Shiu et al.
  2024 LIF whole-brain model on FlyWire) and publishes ``BrainState`` snapshots;
* the brain window (``perpetualfly.brain_viz``) draws ``BrainLayout`` once and then
  each ``BrainState``.

Everything here is plain dataclasses of builtins / numpy arrays so it pickles
cheaply through ``multiprocessing`` queues. Keep it stable: if a field must change,
change it here first and update both sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Predicted neurotransmitters in FlyWire (Eckstein et al. 2024). Order is fixed so
# arrays indexed by transmitter line up on both sides.
NEUROTRANSMITTERS: tuple[str, ...] = ("ACh", "GABA", "Glu", "DA", "5HT", "OA")

# Descending-neuron groups the brain reads out for the fly body. Values in
# BrainState.descending use these keys; each maps to one or more FlyWire neurons
# (chosen and documented in perpetualfly/brain/, not here).
DESCENDING_GROUPS: tuple[str, ...] = (
    "walk_L", "walk_R",        # e.g. DNg100 / BDN2, oDN1: forward walking drive
    "turn_L", "turn_R",        # e.g. DNa02 / DNa01: ipsilateral steering
    "backward_L", "backward_R",  # MDN: backward walking
    "escape",                  # giant fiber (DNp01)
    "groom",                   # DNg12 (anterior / head grooming), both sides
)


@dataclass
class StimulusEvent:
    """Something the body experienced that should drive sensory neurons.

    kind: "whip_hit", "shove", "ground_contact", "fall", "reset", "manual", ...
    side: "left" | "right" | "front" | "rear" | "top" | "none"
    intensity: 0..1 normalised (e.g. measured impulse / absurd-level impulse).
    duration_s: how long the sensory drive should last (brain time).
    sim_time: fly simulation time when it happened (for display/sync only).
    """

    kind: str
    side: str = "none"
    intensity: float = 1.0
    duration_s: float = 0.05
    sim_time: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class BrainLayout:
    """Static description sent once when the brain process starts.

    regions: neuropil names shown in the window (e.g. "MB_L", "AL_R", "GNG", ...).
    region_xy: (n_regions, 2) float32 2D positions (projected neuropil centroids,
        arbitrary units; the window rescales) for the brain map.
    region_outline_xy: optional list of (k, 2) polylines per region, same units, for
        drawing outlines; may be empty.
    display_neuron_ids: FlyWire root ids of the subset of neurons whose spikes are
        streamed for the raster (a few hundred to a few thousand).
    display_neuron_region: (n_display,) int index into ``regions``.
    display_neuron_nt: (n_display,) int index into NEUROTRANSMITTERS (-1 unknown).
    display_neuron_xy: (n_display, 2) float32 projected soma/centroid positions.
    display_neuron_label: short labels (cell type or ""), same length.
    n_neurons_total: neurons in the simulated model.
    model_name: e.g. "Shiu2024-LIF FlyWire v783".
    """

    regions: list[str]
    region_xy: np.ndarray
    region_outline_xy: list[np.ndarray]
    display_neuron_ids: np.ndarray
    display_neuron_region: np.ndarray
    display_neuron_nt: np.ndarray
    display_neuron_xy: np.ndarray
    display_neuron_label: list[str]
    n_neurons_total: int
    model_name: str


@dataclass
class BrainState:
    """One snapshot of brain activity, published a few times per second.

    brain_time: seconds of simulated brain time since start.
    wall_time: time.time() when published.
    realtime_factor: simulated brain seconds per wall second, recent average.
    window_s: length of the averaging window used for the rates below.
    rate_by_nt: (len(NEUROTRANSMITTERS),) mean firing rate (Hz) over all neurons of
        each transmitter type in the window.
    active_frac_by_nt: fraction of neurons of each type that spiked in the window.
    rate_by_region: (len(layout.regions),) mean rate (Hz) per region.
    descending: {group: rate Hz} for DESCENDING_GROUPS.
    raster_idx / raster_t: spikes of display neurons in the window, as indices into
        layout.display_neuron_ids and times (brain_time seconds).
    total_spikes: spikes of all neurons in the window.
    recent_stimuli: kinds/sides of stimuli applied during the window (for labels).

    Optional fields (additive; defaults keep older producers/consumers working):
    seq: 1, 2, 3, ... per published state (gaps = states dropped by a full queue).
    sim_time: pace="sim" only: the fly run time the end of this window corresponds
        to (brain_time + clock offset); ``fly time - sim_time`` is the brain's lag.
    compute_rtf: brain seconds per wall second spent *computing* (excludes waiting
        for the fly clock or wall time): how fast the brain could run.
    probes: {name: rate Hz} of extra single-neuron readouts, e.g. "MN9".
    drive: filled in by the fly app, not the brain: the walking command derived
        from ``descending`` ({"left", "right", "forward", "turn", "escape",
        "applied", "lag_s"}); shown by the brain window (DRIVE panel, header).
    """

    brain_time: float
    wall_time: float
    realtime_factor: float
    window_s: float
    rate_by_nt: np.ndarray
    active_frac_by_nt: np.ndarray
    rate_by_region: np.ndarray
    descending: dict[str, float]
    raster_idx: np.ndarray
    raster_t: np.ndarray
    total_spikes: int
    recent_stimuli: list[str] = field(default_factory=list)
    seq: int = 0
    sim_time: float | None = None
    compute_rtf: float = 0.0
    probes: dict[str, float] = field(default_factory=dict)
    drive: dict | None = None
