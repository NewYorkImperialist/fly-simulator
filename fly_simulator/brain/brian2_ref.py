"""Reference Brian2 implementation of the Shiu et al. 2024 model (validation/bench only).

This restates ``create_model()`` / ``poi()`` from philshiu/Drosophila_brain_model
``model.py`` (MIT licence, Copyright (c) 2023 Philip Shiu) with the same equations,
parameters, schedule and connectivity, so the numba engine in ``engine.py`` can be
checked against it. Not used by the running app (it is far slower; see
docs/BRAIN.md). Requires ``brian2``.
"""

from __future__ import annotations

from textwrap import dedent

import numpy as np

from .engine import LIFParams


def build_network(indptr, indices, weights, poisson_idx, rate_hz: float,
                  params: LIFParams | None = None, seed: int | None = None):
    """Return (Network, NeuronGroup, SpikeMonitor, PoissonInputs)."""
    import brian2 as b2
    from brian2 import Hz, mV, ms

    p = params or LIFParams()
    if seed is not None:
        b2.seed(seed)
    ns = {
        "v_0": p.v_0 * mV, "v_rst": p.v_rst * mV, "v_th": p.v_th * mV,
        "t_mbr": p.t_mbr * ms, "tau": p.tau * ms,
    }
    eqs = dedent("""
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau               : volt (unless refractory)
        rfc                            : second
        """)
    n = len(indptr) - 1
    neu = b2.NeuronGroup(n, model=eqs, method="linear", threshold="v > v_th",
                         reset="v = v_rst; g = 0 * mV", refractory="rfc",
                         name="default_neurons", namespace=ns)
    neu.v = p.v_0 * mV
    neu.g = 0 * mV
    neu.rfc = p.t_rfc * ms
    syn = b2.Synapses(neu, neu, "w : volt", on_pre="g += w", delay=p.t_dly * ms,
                      name="default_synapses")
    pre = np.repeat(np.arange(n), np.diff(indptr))
    syn.connect(i=pre, j=np.asarray(indices))
    syn.w = np.asarray(weights, dtype=float) * p.w_syn * mV
    mon = b2.SpikeMonitor(neu)
    pois = []
    for i in poisson_idx:
        pi = b2.PoissonInput(target=neu[int(i)], target_var="v", N=1, rate=rate_hz * Hz,
                             weight=p.w_syn * p.f_poi * mV)
        neu[int(i)].rfc = 0 * ms
        pois.append(pi)
    net = b2.Network(neu, syn, mon, *pois)
    return net, neu, mon, pois
