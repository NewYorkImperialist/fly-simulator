"""Looming-escape habituation: short-term synaptic depression on the giant fibre's
looming inputs (phenomenological engine add-on, off by default).

Biology
-------
Repeated harmless looming stops evoking escape and recovers after rest. For the
Drosophila giant-fibre (GF, DNp01) escape pathway, Engel & Wu (1996, J Neurosci
16:3486) showed that the GF-mediated response habituates with repeated stimulation
of the brain, recovers spontaneously within tens of seconds, can be dishabituated
by a novel stimulus, and that the decrement lies in the *afferent* pathway onto the
GF (the GF -> TTMn/DLMn output follows high rates without fatigue). The GF's main
looming afferents are the visual projection neurons LC4 and LPLC2 (von Reyn et al.
2017 Neuron; Ache et al. 2019 Curr Biol); in FlyWire v783 they are the two
largest identified inputs to DNp01 (LPLC2 1080 and LC4 805 synapses onto the two
GFs). (Looming-evoked escape in freely behaving flies: Card & Dickinson 2008 Curr
Biol; we cite no specific behavioural looming-habituation dataset, so the time
course here is illustrative, not fitted.)

Model (what is modelled and what is connectome)
-----------------------------------------------
The Shiu et al. LIF model has static synapses. Here the synapses **from every
LC4 / LPLC2 neuron onto the GF** (``post_types``; empty = onto all of their
targets) get the classic resource-depletion short-term depression (Tsodyks &
Markram 1997 PNAS; Abbott et al. 1997 Science), a standard phenomenological
account of habituation as homosynaptic depression (Castellucci & Kandel 1974 in
Aplysia)::

    each presynaptic spike: PSC *= x, then x -= U x
    between spikes:         dx/dt = (1 - x) / tau_rec

computed exactly per spike inside the engine kernel (``LIFEngine.set_depression``).
The *site* follows the literature (GF afferents); U and tau_rec are free
parameters chosen so that the key-O loom (LC4 at 200 Hz for 1 s) habituates over a
few trials at ~2 s spacing and recovers within ~30-60 s (docs/HABITUATION.md).

Dishabituation (optional, ``dishabituate_frac`` > 0): a whip hit / shove moves the
efficacies ``frac`` of the way back to 1. Honest caveat: in the dual-process view
(Groves & Thompson 1970) dishabituation is a superimposed sensitisation, not an
undoing of the depression; this reset is a phenomenological shortcut.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import numpy as np

MODEL_LABEL = "habituation (model)"
HIT_KINDS = ("whip_hit", "shove", "hit")


@dataclass
class HabituationConfig:
    enabled: bool = False
    pre_types: tuple = ("LC4", "LPLC2")   # presynaptic cell types (depressed synapses)
    post_types: tuple = ("DNp01",)        # their targets affected; () = all targets
    u: float = 0.006                      # fraction of efficacy used per spike
    tau_rec_s: float = 20.0               # recovery time constant
    dishabituate_frac: float = 0.0        # whip hit / shove: x += frac (1 - x); 0 = off
    persist_on_reset: bool = True         # a fly/brain reset keeps the depression

    @classmethod
    def from_dict(cls, d: dict | None) -> "HabituationConfig":
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        out = {k: v for k, v in d.items() if k in names}
        for k in ("pre_types", "post_types"):
            if isinstance(out.get(k), (list, str)):
                v = out[k]
                out[k] = tuple([v] if isinstance(v, str) else v)
        return cls(**out)

    def to_dict(self) -> dict:
        return asdict(self)


def select_types(table, types) -> np.ndarray:
    ct = table.col("cell_type")
    if not types:
        return np.zeros(0, dtype=np.int64)
    return np.nonzero(np.isin(ct.astype(str), [str(t) for t in types]))[0].astype(np.int64)


class LoomHabituation:
    """Lives in the brain worker next to the engine (``process._Model``)."""

    def __init__(self, table, engine, cfg: HabituationConfig | dict | None = None):
        self.table = table
        self.engine = engine
        self.cfg = HabituationConfig()
        self.pre = np.zeros(0, dtype=np.int64)
        self.post: np.ndarray | None = None
        self.n_dishab = 0
        self.configure(cfg if cfg is not None else HabituationConfig(enabled=True))

    def configure(self, cfg: HabituationConfig | dict | None) -> None:
        if not isinstance(cfg, HabituationConfig):
            cfg = HabituationConfig.from_dict(cfg)
        self.cfg = cfg
        self.pre = select_types(self.table, cfg.pre_types)
        self.post = select_types(self.table, cfg.post_types) if cfg.post_types else None
        if cfg.enabled and len(self.pre):
            self.engine.set_depression(self.pre, self.post, cfg.u, cfg.tau_rec_s)
        else:
            self.engine.set_depression(None)

    def on_stimulus(self, ev) -> bool:
        """Dishabituation by a hit (if configured). True if applied."""
        c = self.cfg
        if not c.enabled or c.dishabituate_frac <= 0 or ev.kind not in HIT_KINDS:
            return False
        self.engine.scale_depression(c.dishabituate_frac, self.pre)
        self.n_dishab += 1
        return True

    def on_reset(self) -> None:
        if not self.cfg.persist_on_reset:
            self.engine.scale_depression(1.0, self.pre)

    def efficacy(self) -> dict:
        e = self.engine
        ct = self.table.col("cell_type")
        out = {}
        for t in self.cfg.pre_types:
            idx = self.pre[ct[self.pre] == t]
            out[str(t)] = float(e.efficacy(idx).mean()) if len(idx) else 1.0
        return out

    def readout(self) -> dict:
        eff = self.efficacy()
        on = bool(self.cfg.enabled and self.engine.std_pre is not None)
        vals = list(eff.values())
        return {"enabled": on, "label": MODEL_LABEL,
                # the most depressed pathway (an undriven type stays at 1)
                "efficacy": float(min(vals)) if vals else 1.0, "by_type": eff,
                "u": float(self.cfg.u), "tau_rec_s": float(self.cfg.tau_rec_s),
                "n_pre": int(len(self.pre)),
                "n_post": None if self.post is None else int(len(self.post)),
                "dishabituations": int(self.n_dishab)}
