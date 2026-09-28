"""Event-driven reimplementation of the Shiu et al. (2024) whole-brain LIF model.

This is a port of the *equations and parameters* of ``model.py`` in
github.com/philshiu/Drosophila_brain_model (MIT licence), written from scratch for
chunked, incremental simulation (Brian2's runtime mode is too slow for real time, and
its cpp_standalone mode cannot pause/resume; see docs/BRAIN.md for the benchmark).

Model (per neuron, all in mV / ms, exactly as in Shiu's ``default_params``)::

    dv/dt = (v_0 - v + g) / t_mbr     (unless refractory)
    dg/dt = -g / tau                  (unless refractory)
    spike when v > v_th (and not refractory) -> v = v_rst, g = 0, refractory t_rfc
    presynaptic spike -> after t_dly: g_post += w_syn * (sign * n_synapses)
        (discarded if the target is refractory: Brian2 "conditional write")
    Poisson drive (N=1, rate r) -> v += w_syn * f_poi  (targets have t_rfc = 0)

Integration is the exact solution of the linear ODE over one step (what Brian2's
``method='linear'`` does), dt = 0.1 ms (Brian2 default). Per step the order of
operations follows Brian2's default schedule: state update -> threshold ->
synaptic delivery (+ Poisson input) -> reset.

The inner loop is compiled with numba. Only neurons away from rest are integrated
(an "active set"; a neuron at rest is a fixed point), and spikes are propagated by
walking the CSR row of each spiking neuron, so the cost per step is
O(active neurons) + O(spikes x out-degree) instead of O(N).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:  # numba is part of the ``brain`` extra
    import numba
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without numba
    numba = None

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


@dataclass
class LIFParams:
    """Shiu et al. 2024 ``default_params`` (units: mV, ms, Hz)."""

    v_0: float = -52.0      # resting potential (Kakaria & de Bivort 2017)
    v_rst: float = -52.0    # reset potential
    v_th: float = -45.0     # spike threshold
    t_mbr: float = 20.0     # membrane time constant, ms
    tau: float = 5.0        # synaptic time constant, ms (Juergensen et al. 2021)
    t_rfc: float = 2.2      # refractory period, ms (Lazar et al. 2021)
    t_dly: float = 1.8      # synaptic delay, ms (Paul et al. 2015)
    w_syn: float = 0.275    # mV per synapse (free parameter)
    f_poi: float = 250.0    # Poisson synapse scale: one Poisson event = w_syn*f_poi mV
    dt: float = 0.1         # ms (Brian2 default clock)

    @property
    def delay_steps(self) -> int:
        return int(round(self.t_dly / self.dt))

    @property
    def refractory_steps(self) -> int:
        return int(round(self.t_rfc / self.dt))

    def propagator(self) -> tuple[float, float, float]:
        """Exact one-step propagator of the (v - v_0, g) linear system."""
        dt, tm, ts = self.dt, self.t_mbr, self.tau
        e11 = math.exp(-dt / tm)
        e22 = math.exp(-dt / ts)
        if abs(ts - tm) < 1e-12:
            e12 = dt / tm * e11
        else:
            e12 = ts / (ts - tm) * (e22 - e11)
        return e11, e12, e22


@njit(cache=True, fastmath=False)
def _run_steps(n_steps, step0, S,
               indptr, indices, weights, ring, ring_cnt, delay,
               pois_idx, pois_p, pois_w,
               e11, e12, e22, v_0, v_th, v_rst, eps,
               active, is_active, n_active,
               out_step, out_idx, counts, rng, vth, use_vth,
               use_std, std_pre, std_post, std_x, std_last, std_u, std_tau_steps,
               use_ad, ad_mask, ad_a, ad_last, ad_inc, ad_ginc, ad_inv_tau, ad_g):
    """Advance the network. Only neurons in the *active set* (state differs from
    rest by more than ``eps`` mV) are integrated; a neuron at exact rest
    (v = v_0, g = 0) is a fixed point of the dynamics, so skipping it is exact.
    Neurons whose |v - v_0| and |g| both decay below ``eps`` are snapped to rest and
    leave the set (error < eps mV, i.e. ~1e-6 of the 7 mV threshold gap).

    Refractoriness follows Brian2 exactly: v and g are flagged ``(unless
    refractory)``, which in Brian2 makes them *conditionally written* -- every write
    (integration, synaptic ``g += w``, PoissonInput ``v += ...``) is discarded while
    the neuron is refractory, including in the step it spikes (only the reset is
    applied).

    ``S`` is the packed per-neuron state (one cache line per neuron):
    ``S[i] = (v, g, blocked_until, refractory_steps)``; ``blocked_until`` is the first
    step at which neuron i may be written again (spike step + max(refractory, 1)).

    ``use_vth``: per-neuron thresholds ``vth[i]`` instead of the scalar ``v_th``
    (neuromodulation, fly_simulator/brain/neuromod.py). With ``use_vth`` False the
    arithmetic is exactly the unmodulated model's.

    ``use_std``: phenomenological short-term synaptic depression (habituation,
    fly_simulator/brain/habituation.py). Each presynaptic neuron j with
    ``std_pre[j]`` has an efficacy ``std_x[j]`` in (0, 1] (resource-depletion model,
    Tsodyks & Markram 1997; Abbott et al. 1997): at delivery of each of its spikes
    it first recovers exactly from its last update (``std_last[j]``, a step) as
    ``x = 1 - (1 - x) exp(-(t - last) / std_tau_steps)``, its synapses onto
    targets with ``std_post[post]`` are delivered scaled by x, then
    ``x -= std_u * x``. With ``use_std`` False the arithmetic is exactly the
    undepressed model's.

    ``use_ad``: spike-triggered threshold adaptation (smell fix, fly_simulator/
    brain/smell.py). For neurons with ``ad_mask[i]`` the threshold is raised by
    ``a_i + G``: ``a_i`` (per neuron) jumps by ``ad_inc`` at each of its spikes, the
    shared term ``G`` (``ad_g = [G, last step]``) jumps by ``ad_ginc`` at every
    spike of any masked neuron (a pooled, activity-dependent inhibition); both decay
    exactly with ``exp(-steps * ad_inv_tau)`` (updated lazily from ``ad_last``).
    With ``use_ad`` False the arithmetic is exactly the unadapted model's."""
    n_slots = ring.shape[0]
    cap = out_step.shape[0]
    n_out = 0
    n_dropped = 0
    for s in range(n_steps):
        t = step0 + s
        slot = t % n_slots
        # 1) state update + threshold for active, non-refractory neurons
        cnt = 0
        k = n_active - 1
        while k >= 0:
            i = active[k]
            if t >= S[i, 2]:
                gi = S[i, 1]
                dv = S[i, 0] - v_0
                vi = v_0 + e11 * dv + e12 * gi
                gi = e22 * gi
                S[i, 0] = vi
                S[i, 1] = gi
                thr = vth[i] if use_vth else v_th
                ai = 0.0
                if use_ad:
                    if ad_mask[i]:
                        ai = ad_a[i] * math.exp(-(t - ad_last[i]) * ad_inv_tau)
                        thr += ai + ad_g[0] * math.exp(-(t - ad_g[1]) * ad_inv_tau)
                if vi > thr:
                    ring[slot, cnt] = i
                    cnt += 1
                    r = S[i, 3]
                    S[i, 2] = t + (r if r > 1.0 else 1.0)
                    if use_ad:
                        if ad_mask[i]:
                            ad_a[i] = ai + ad_inc
                            ad_last[i] = t
                            ad_g[0] = ad_g[0] * math.exp(-(t - ad_g[1]) * ad_inv_tau) + ad_ginc
                            ad_g[1] = t
                elif abs(gi) < eps and abs(vi - v_0) < eps:
                    S[i, 0] = v_0
                    S[i, 1] = 0.0
                    is_active[i] = 0
                    n_active -= 1
                    active[k] = active[n_active]
            k -= 1
        ring_cnt[slot] = cnt
        # 2) synaptic delivery of spikes emitted `delay` steps ago
        ds = (t - delay) % n_slots
        for k in range(ring_cnt[ds]):
            j = ring[ds, k]
            dep = False
            xj = 1.0
            if use_std:
                if std_pre[j]:
                    dep = True
                    xj = 1.0 - (1.0 - std_x[j]) * math.exp(-(t - std_last[j]) / std_tau_steps)
                    std_x[j] = xj - std_u * xj
                    std_last[j] = t
            for p in range(indptr[j], indptr[j + 1]):
                post = indices[p]
                if t < S[post, 2]:
                    continue  # refractory target: write discarded (as in Brian2)
                if dep and std_post[post]:
                    S[post, 1] += weights[p] * xj
                else:
                    S[post, 1] += weights[p]
                if is_active[post] == 0:
                    is_active[post] = 1
                    active[n_active] = post
                    n_active += 1
        # 3) Poisson input (Brian2 PoissonInput, N=1 -> Bernoulli per step)
        for k in range(pois_idx.shape[0]):
            if rng.random() < pois_p[k]:
                i = pois_idx[k]
                if t < S[i, 2]:
                    continue
                S[i, 0] += pois_w
                if is_active[i] == 0:
                    is_active[i] = 1
                    active[n_active] = i
                    n_active += 1
        # 4) reset + record
        for k in range(cnt):
            i = ring[slot, k]
            S[i, 0] = v_rst
            S[i, 1] = 0.0
            counts[i] += 1
            if n_out < cap:
                out_step[n_out] = t
                out_idx[n_out] = i
                n_out += 1
            else:
                n_dropped += 1
    return n_out, n_dropped, n_active


class LIFEngine:
    """Stateful LIF network that can be advanced in arbitrary chunks.

    Parameters
    ----------
    indptr, indices, weights: CSR adjacency by *presynaptic* neuron; ``weights`` is the
        signed synapse count (Shiu's "Excitatory x Connectivity"); it is multiplied by
        ``params.w_syn`` here.
    """

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, weights: np.ndarray,
                 params: LIFParams | None = None, seed: int = 0,
                 spike_capacity: int = 2_000_000):
        if numba is None:
            raise ImportError("fly_simulator.brain.engine needs numba (pip install numba)")
        self.p = params or LIFParams()
        self.n = int(len(indptr) - 1)
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        self.weights = (np.asarray(weights, dtype=np.float64) * self.p.w_syn).astype(np.float32)
        # packed state, see _run_steps; the attributes below are views into it
        self._S = np.zeros((self.n, 4), dtype=np.float64)
        self.v = self._S[:, 0]
        self.g = self._S[:, 1]
        self.blocked_until = self._S[:, 2]
        self.ref_steps = self._S[:, 3]
        self.v[:] = self.p.v_0
        self._ref_default = self.p.refractory_steps
        self.ref_steps[:] = self._ref_default
        d = self.p.delay_steps
        self.delay = d
        self.ring = np.zeros((d + 1, self.n), dtype=np.int32)
        self.ring_cnt = np.zeros(d + 1, dtype=np.int64)
        self.counts = np.zeros(self.n, dtype=np.int64)  # cumulative spikes per neuron
        self.eps = 1e-6  # mV; see _run_steps
        self._active = np.zeros(self.n, dtype=np.int32)
        self._is_active = np.zeros(self.n, dtype=np.uint8)
        self.n_active = 0
        self.step_count = 0
        self._out_step = np.empty(spike_capacity, dtype=np.int64)
        self._out_idx = np.empty(spike_capacity, dtype=np.int32)
        self._e = self.p.propagator()
        self.pois_w = self.p.w_syn * self.p.f_poi
        self._pois_idx = np.zeros(0, dtype=np.int32)
        self._pois_p = np.zeros(0, dtype=np.float64)
        self.dropped_spikes = 0
        self._rng = np.random.default_rng(seed)  # per-engine stream (numba Generator)
        # optional per-neuron thresholds (neuromodulation); None = scalar v_th
        self.v_th_arr: np.ndarray | None = None
        self._vth_dummy = np.zeros(1, dtype=np.float64)
        # virtual lesions (brain playground, docs/PLAYGROUND.md): silenced neurons
        # get an infinite spike threshold for the run, so they never fire and send
        # no output. None = none silenced (the exact unlesioned code path).
        self.silenced: np.ndarray | None = None
        self._vth_eff: np.ndarray | None = None
        # short-term synaptic depression (habituation, fly_simulator/brain/habituation.py);
        # None = off (the exact undepressed code path). See set_depression().
        self.std_pre: np.ndarray | None = None
        self.std_post: np.ndarray | None = None
        self.std_x = np.ones(1, dtype=np.float64)
        self.std_last = np.zeros(1, dtype=np.float64)
        self.std_u = 0.0
        self.std_tau_s = 1.0
        self._std_dummy = np.zeros(1, dtype=np.uint8)
        # spike-triggered threshold adaptation (smell fix, fly_simulator/brain/smell.py);
        # None = off (the exact unadapted code path). See set_adaptation().
        self.ad_mask: np.ndarray | None = None
        self.ad_a = np.zeros(1, dtype=np.float64)
        self.ad_last = np.zeros(1, dtype=np.float64)
        self.ad_g = np.zeros(2, dtype=np.float64)
        self.ad_inc = 0.0
        self.ad_ginc = 0.0
        self.ad_tau_s = 0.1

    # ------------------------------------------------------------------ inputs
    @property
    def t(self) -> float:
        """Simulated time in seconds."""
        return self.step_count * self.p.dt * 1e-3

    def set_poisson(self, idx: np.ndarray, rates_hz: np.ndarray | float) -> None:
        """Replace the Poisson-driven set. Like Shiu's ``poi()``, driven neurons get
        no refractory period; neurons that stop being driven get it back."""
        idx = np.asarray(idx, dtype=np.int32)
        rates = np.broadcast_to(np.asarray(rates_hz, dtype=np.float64), idx.shape)
        keep = rates > 0
        idx, rates = idx[keep], rates[keep]
        self.ref_steps[self._pois_idx] = self._ref_default
        self.ref_steps[idx] = 0
        self._pois_idx = np.ascontiguousarray(idx)
        self._pois_p = np.ascontiguousarray(np.minimum(rates * self.p.dt * 1e-3, 1.0))

    def reset_state(self) -> None:
        self.v[:] = self.p.v_0
        self.g[:] = 0.0
        self.blocked_until[:] = 0
        self.ring_cnt[:] = 0
        self._is_active[:] = 0
        self.n_active = 0
        self.reset_adaptation()

    def threshold_array(self) -> np.ndarray:
        """Per-neuron spike thresholds (mV), created on first use as ``v_th``
        everywhere; write into it to modulate excitability. ``clear_threshold()``
        returns to the scalar threshold (the exact unmodulated code path)."""
        if self.v_th_arr is None:
            self.v_th_arr = np.full(self.n, self.p.v_th, dtype=np.float64)
        return self.v_th_arr

    def clear_threshold(self) -> None:
        self.v_th_arr = None

    def set_silenced(self, idx) -> None:
        """Virtual lesion: neurons ``idx`` can no longer spike (their threshold is
        +inf during ``run``; membrane / synaptic state still integrates, so they
        simply produce no output). ``None`` or empty = no lesion; the engine then
        runs the exact unlesioned code path. Independent of ``threshold_array()``
        (neuromodulation), which may be set or cleared at any time."""
        if idx is None or len(idx) == 0:
            self.silenced = None
            self._vth_eff = None
            return
        self.silenced = np.unique(np.asarray(idx, dtype=np.int64))
        if self._vth_eff is None:
            self._vth_eff = np.empty(self.n, dtype=np.float64)

    def _thresholds(self):
        """(array, use_array) for the kernel: modulated thresholds and lesions."""
        if self.silenced is None:
            if self.v_th_arr is None:
                return self._vth_dummy, False
            return self.v_th_arr, True
        buf = self._vth_eff
        if self.v_th_arr is None:
            buf.fill(self.p.v_th)
        else:
            np.copyto(buf, self.v_th_arr)
        buf[self.silenced] = np.inf
        return buf, True

    # ------------------------------------------------------------------ depression
    def set_depression(self, pre_idx, post_idx=None, u: float = 0.0,
                       tau_rec_s: float = 10.0) -> None:
        """Short-term depression of the synapses from ``pre_idx`` onto ``post_idx``
        (None = onto every target): each presynaptic spike scales that neuron's
        depressed synapses by its efficacy x, then x -= u x; x recovers to 1 with
        ``tau_rec_s``. ``pre_idx`` None / empty = off (exact undepressed path).
        Efficacies are kept when only ``u`` / ``tau_rec_s`` change (same sets)."""
        if pre_idx is None or len(pre_idx) == 0:
            self.std_pre = None
            self.std_post = None
            return
        pre = np.zeros(self.n, dtype=np.uint8)
        pre[np.asarray(pre_idx, dtype=np.int64)] = 1
        post = np.ones(self.n, dtype=np.uint8) if post_idx is None else np.zeros(self.n, np.uint8)
        if post_idx is not None:
            post[np.asarray(post_idx, dtype=np.int64)] = 1
        same = (self.std_pre is not None and np.array_equal(pre, self.std_pre)
                and np.array_equal(post, self.std_post))
        if not same:
            self.std_pre, self.std_post = pre, post
            self.std_x = np.ones(self.n, dtype=np.float64)
            self.std_last = np.full(self.n, float(self.step_count), dtype=np.float64)
        self.std_u = float(min(max(u, 0.0), 1.0))
        self.std_tau_s = float(max(tau_rec_s, 1e-6))

    def efficacy(self, idx=None) -> np.ndarray:
        """Current efficacies (recovered to now) of presynaptic neurons ``idx``
        (all when None); 1 where depression is off."""
        idx = np.arange(self.n) if idx is None else np.asarray(idx, dtype=np.int64)
        if self.std_pre is None:
            return np.ones(len(idx))
        tau_steps = self.std_tau_s * 1e3 / self.p.dt
        x, last = self.std_x[idx], self.std_last[idx]
        return 1.0 - (1.0 - x) * np.exp(-(self.step_count - last) / tau_steps)

    def scale_depression(self, frac: float, idx=None) -> None:
        """Move efficacies a fraction ``frac`` of the way back to 1 (dishabituation)."""
        if self.std_pre is None:
            return
        idx = np.nonzero(self.std_pre)[0] if idx is None else np.asarray(idx, dtype=np.int64)
        x = self.efficacy(idx)
        self.std_x[idx] = x + float(np.clip(frac, 0.0, 1.0)) * (1.0 - x)
        self.std_last[idx] = float(self.step_count)

    # ------------------------------------------------------------------ adaptation
    def set_adaptation(self, idx, inc_mv: float = 0.0, tau_s: float = 0.1,
                       global_inc_mv: float = 0.0) -> None:
        """Spike-triggered threshold adaptation of neurons ``idx``: each of their
        spikes raises that neuron's threshold by ``inc_mv`` and the threshold of
        *all* of them by ``global_inc_mv`` (pooled activity-dependent inhibition);
        both decay with ``tau_s``. ``idx`` None / empty = off (exact unadapted
        path). Adaptation state is reset whenever this is called."""
        if idx is None or len(idx) == 0 or (inc_mv <= 0 and global_inc_mv <= 0):
            self.ad_mask = None
            return
        mask = np.zeros(self.n, dtype=np.uint8)
        mask[np.asarray(idx, dtype=np.int64)] = 1
        self.ad_mask = mask
        self.ad_a = np.zeros(self.n, dtype=np.float64)
        self.ad_last = np.full(self.n, float(self.step_count), dtype=np.float64)
        self.ad_g = np.array([0.0, float(self.step_count)])
        self.ad_inc = float(max(inc_mv, 0.0))
        self.ad_ginc = float(max(global_inc_mv, 0.0))
        self.ad_tau_s = float(max(tau_s, 1e-6))

    def reset_adaptation(self) -> None:
        """Clear the adaptation state (thresholds back to baseline)."""
        if self.ad_mask is not None:
            self.ad_a[:] = 0.0
            self.ad_g[0] = 0.0

    def adaptation_mv(self, idx=None) -> np.ndarray:
        """Current threshold raise (mV, per-neuron + pooled) of neurons ``idx``."""
        idx = np.arange(self.n) if idx is None else np.asarray(idx, dtype=np.int64)
        if self.ad_mask is None:
            return np.zeros(len(idx))
        k = self.p.dt * 1e-3 / self.ad_tau_s
        a = self.ad_a[idx] * np.exp(-(self.step_count - self.ad_last[idx]) * k)
        g = self.ad_g[0] * math.exp(-(self.step_count - self.ad_g[1]) * k)
        return (a + g) * self.ad_mask[idx]

    def _ad_args(self):
        if self.ad_mask is None:
            d = self._std_dummy
            return False, d, self.ad_a, self.ad_last, 0.0, 0.0, 1.0, self.ad_g
        inv_tau = self.p.dt * 1e-3 / self.ad_tau_s
        return (True, self.ad_mask, self.ad_a, self.ad_last, self.ad_inc, self.ad_ginc,
                inv_tau, self.ad_g)

    # ------------------------------------------------------------------ running
    def run(self, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Advance ``n_steps`` of dt. Returns (spike_step, spike_neuron) arrays
        (copies) of all spikes in the chunk; steps are absolute step numbers."""
        e11, e12, e22 = self._e
        p = self.p
        vth, use_vth = self._thresholds()
        n_out, dropped, self.n_active = _run_steps(
            int(n_steps), int(self.step_count), self._S, self.indptr, self.indices, self.weights, self.ring,
            self.ring_cnt, self.delay, self._pois_idx, self._pois_p, self.pois_w,
            e11, e12, e22, p.v_0, p.v_th, p.v_rst, self.eps,
            self._active, self._is_active, self.n_active,
            self._out_step, self._out_idx, self.counts, self._rng,
            vth, use_vth, *self._std_args(), *self._ad_args())
        self.step_count += int(n_steps)
        self.dropped_spikes += int(dropped)
        return self._out_step[:n_out].copy(), self._out_idx[:n_out].copy()

    def _std_args(self):
        if self.std_pre is None:
            d = self._std_dummy
            return False, d, d, self.std_x, self.std_last, 0.0, 1.0
        return (True, self.std_pre, self.std_post, self.std_x, self.std_last, self.std_u,
                self.std_tau_s * 1e3 / self.p.dt)

    def run_seconds(self, seconds: float) -> tuple[np.ndarray, np.ndarray]:
        return self.run(int(round(seconds * 1e3 / self.p.dt)))


def csr_from_edges(pre: np.ndarray, post: np.ndarray, w: np.ndarray, n: int):
    """Build (indptr, indices, weights) sorted by presynaptic index."""
    pre = np.asarray(pre, dtype=np.int64)
    order = np.argsort(pre, kind="stable") if np.any(np.diff(pre) < 0) else None
    if order is not None:
        pre, post, w = pre[order], np.asarray(post)[order], np.asarray(w)[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    return indptr, np.asarray(post, dtype=np.int32), np.asarray(w, dtype=np.float64)


def random_network(n: int = 50, p_conn: float = 0.1, seed: int = 0,
                   frac_inhibitory: float = 0.2, max_syn: int = 20):
    """Tiny synthetic connectome for tests: returns (indptr, indices, weights, sign)."""
    rng = np.random.default_rng(seed)
    sign = np.where(rng.random(n) < frac_inhibitory, -1, 1)
    mask = rng.random((n, n)) < p_conn
    np.fill_diagonal(mask, False)
    pre, post = np.nonzero(mask)
    counts = rng.integers(1, max_syn + 1, size=len(pre))
    return (*csr_from_edges(pre, post, counts * sign[pre], n), sign)
