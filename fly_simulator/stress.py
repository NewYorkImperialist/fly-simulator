"""Stress / arousal: body-side effects of the octopamine level (docs/STRESS.md).

The level itself lives in the brain worker (``fly_simulator/brain/neuromod.py``): a
leaky integrator driven by the spikes of the connectome's octopaminergic neurons
(OA-VUM / OA-VPM / OA-AL2 / OA-ASM), published as ``BrainState.neuromod``. This
module maps that level onto the body. The mapping is phenomenological (a model
choice, not connectome data), bounded, and a no-op when disabled:

* **locomotor vigour ("pain makes it run"):** the CPG intrinsic frequency is
  multiplied by ``1 + freq_gain * level`` (capped at ``max_freq_mult``) and the
  final ``[left, right]`` drive (stride amplitude, after the heading hold and the
  brain's steering) by ``1 + amp_gain * level`` (each side capped at ``max_amp``).
  Octopamine is required for normal walking speed / vigour in flies (Tbh mutants
  walk less and slower). Defaults: 14.1 mm/s calm -> 24.5 mm/s at level 1.
* **input:** the spikes of the connectome's OA neurons. Looming, head bristles and
  falls reach them through the connectome; whip hits / shoves do not, so by
  default (``noci_relay``) each hit also drives a labelled "VNC stand-in" relay of
  ascending neurons that do reach them (and the walk DNs). See
  fly_simulator/brain/neuromod.py and docs/STRESS.md.
* **jumpiness:** with ``--brain-actions`` the giant-fiber rate needed to trigger a
  jump is ``jump_escape_hz * (1 - jump_threshold_drop * level)`` (not below
  ``min_jump_hz``), so a weaker looming stimulus makes a stressed fly take off.

Integration (the app wires it; nothing here edits app.py)::

    from fly_simulator.stress import StressConfig, install_stress
    stress = install_stress(session, StressConfig(enabled=True))   # after Session()
    ...                                    # per physics chunk: automatic (see hook)
    stress.hud_line(); stress.metric_row(); stress.summary(); stress.close()

``install_stress`` also accepts parts: ``install_stress({"controller": ctrl,
"link": brain_link, "triggers": triggers}, cfg)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np

from fly_simulator.brain.neuromod import MODEL_LABEL, NeuromodConfig

METRIC_COLUMNS = ("octopamine", "oa_rate_hz", "noci_hz", "stress_freq_mult",
                  "stress_amp_mult", "stress_jump_hz")


@dataclass
class StressConfig:
    enabled: bool = False             # --stress
    # measured, flat ground: level 1 -> x1.5 frequency, x1.2 amplitude = 24.5 mm/s
    # (calm 14.1 mm/s); 20 s on "normal" terrain without a fall (docs/STRESS.md)
    freq_gain: float = 0.5            # CPG frequency x (1 + freq_gain * level)
    max_freq_mult: float = 1.5
    amp_gain: float = 0.2             # stride amplitude x (1 + amp_gain * level)
    max_amp: float = 1.5              # |signal| cap per side after scaling
    # Whip hits do not reach the OA neurons in the connectome, so by default each hit
    # also drives the nociceptive relay (NeuromodConfig.noci_relay, a VNC stand-in:
    # ascending neurons an_walk + an_arousal; which ANs is our choice, everything
    # downstream is connectome). False = connectome-only input (looming, head
    # bristles, falls). nociceptive_input = the alternative, fully modelled link.
    noci_relay: bool = True
    nociceptive_input: bool = False
    jump_threshold_drop: float = 0.5  # GF jump threshold x (1 - drop * level)
    min_jump_hz: float = 20.0
    stale_after_s: float = 2.0        # no fresh brain state for this long -> level held
    hook_update: bool = True          # run update() after BrainLink.update() by itself
    # brain-side layer (NeuromodConfig fields: tau_rise_s, tau_decay_s, ref_rate_hz,
    # vth_shift_mv, target_min_syn, noci_gain, noci_ref_hz, ...)
    neuromod: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> "StressConfig":
        d = dict(d or {})
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in names})

    def neuromod_config(self) -> NeuromodConfig:
        d = {"noci_relay": bool(self.noci_relay),
             "nociceptive_input": bool(self.nociceptive_input), **self.neuromod,
             "enabled": bool(self.enabled)}
        return NeuromodConfig.from_dict(d)


def freq_multiplier(level: float, cfg: StressConfig) -> float:
    return float(min(1.0 + cfg.freq_gain * float(np.clip(level, 0.0, 1.0)), cfg.max_freq_mult))


def amp_multiplier(level: float, cfg: StressConfig) -> float:
    return float(1.0 + cfg.amp_gain * float(np.clip(level, 0.0, 1.0)))


def jump_threshold(level: float, base_hz: float, cfg: StressConfig) -> float:
    f = 1.0 - cfg.jump_threshold_drop * float(np.clip(level, 0.0, 1.0))
    return float(max(base_hz * f, min(cfg.min_jump_hz, base_hz)))


def _get(obj, name):
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


class StressHandle:
    """Applies the published octopamine level to the body. Disabled -> inert."""

    def __init__(self, cfg: StressConfig, controller=None, link=None, triggers=None,
                 say: Callable[[str], None] | None = None) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.enabled)
        self.controller = controller
        self.link = link
        self._triggers = triggers
        self.say = say or (lambda msg: None)
        self.level = 0.0
        self.oa_rate_hz = 0.0
        self.noci_hz = 0.0
        self.freq_mult = 1.0
        self.amp_mult = 1.0
        self.readout: dict = {}
        self.max_level = 0.0
        self._last_state = None
        self._base_freqs: np.ndarray | None = None
        self._freq_target: np.ndarray | None = None
        self._base_jump_hz: float | None = None
        self._orig_update = None
        self._prev_filter = None
        self._our_filter = None
        if not self.enabled:
            return
        if cfg.amp_gain and controller is not None and hasattr(controller, "signal_filter"):
            # stride amplitude: scale the final [left, right] drive (after the brain's
            # steering filter, if any), keeping its sign (stepping direction)
            self._prev_filter = controller.signal_filter
            prev = self._prev_filter
            cap = float(cfg.max_amp)

            def stress_filter(hold):
                sig = hold if prev is None else prev(hold)
                a = self.amp_mult
                if a == 1.0:
                    return sig
                return np.clip(np.asarray(sig, dtype=float) * a, -cap, cap)

            self._our_filter = stress_filter
            controller.signal_filter = stress_filter
        self._freq_target = self._find_freq_array()
        if self._freq_target is not None:
            self._base_freqs = self._freq_target.copy()
        trig = self.triggers
        if trig is not None:
            self._base_jump_hz = float(trig.p.jump_escape_hz)
        brain = _get(link, "brain") if link is not None else None
        if brain is not None and hasattr(brain, "set_neuromod"):
            brain.set_neuromod(cfg.neuromod_config().to_dict())
        else:
            self.say("[stress] no brain running: the octopamine level stays 0 "
                     "(it is driven by the connectome's OA neurons; run with --brain)")
        if cfg.hook_update and link is not None and hasattr(link, "update"):
            self._orig_update = link.update

            def update_then_stress(*a, **kw):
                out = self._orig_update(*a, **kw)
                self.update()
                return out

            link.update = update_then_stress  # instance attribute; undone in close()

    # ------------------------------------------------------------------ parts
    @property
    def triggers(self):
        if self._triggers is not None:
            return self._triggers
        return _get(self.link, "triggers") if self.link is not None else None

    def _find_freq_array(self) -> np.ndarray | None:
        impl = _get(self.controller, "impl") if self.controller is not None else None
        if impl is None:
            return None
        base = getattr(impl, "_base_intrinsic_freqs", None)  # hybrid (FlyGym)
        if isinstance(base, np.ndarray):
            return base
        net = getattr(impl, "cpg_network", None)  # plain CPG controller
        arr = getattr(net, "intrinsic_freqs", None)
        return arr if isinstance(arr, np.ndarray) else None

    # ------------------------------------------------------------------ update
    def update(self, state=None) -> float:
        """Read the newest BrainState (or ``state``) and apply the level. Cheap when
        nothing changed; call once per physics chunk (automatic with hook_update)."""
        if not self.enabled:
            return 0.0
        st = state if state is not None else (_get(self.link, "latest")
                                              if self.link is not None else None)
        if st is not None and st is not self._last_state:
            self._last_state = st
            nm = getattr(st, "neuromod", None) or {}
            if nm:
                self.readout = dict(nm)
                self.level = float(np.clip(nm.get("octopamine", 0.0), 0.0, 1.0))
                self.oa_rate_hz = float(nm.get("oa_rate_hz", 0.0))
                self.noci_hz = float(nm.get("noci_hz", 0.0))
                self.max_level = max(self.max_level, self.level)
        self.set_level(self.level)
        return self.level

    def set_level(self, level: float) -> None:
        """Apply ``level`` to the body (also usable directly, e.g. in tests)."""
        if not self.enabled:
            return
        self.level = float(np.clip(level, 0.0, 1.0))
        m = freq_multiplier(self.level, self.cfg)
        if self._freq_target is not None and abs(m - self.freq_mult) > 1e-6:
            np.multiply(self._base_freqs, m, out=self._freq_target)
        self.freq_mult = m
        self.amp_mult = amp_multiplier(self.level, self.cfg)
        trig = self.triggers
        if trig is not None:
            if self._base_jump_hz is None:
                self._base_jump_hz = float(trig.p.jump_escape_hz)
            trig.p.jump_escape_hz = jump_threshold(self.level, self._base_jump_hz, self.cfg)

    # ------------------------------------------------------------------ readouts
    @property
    def jump_hz(self) -> float | None:
        trig = self.triggers
        return None if trig is None else float(trig.p.jump_escape_hz)

    def hud_line(self) -> str:
        if not self.enabled:
            return ""
        j = self.jump_hz
        src = f"OA neurons {self.oa_rate_hz:.1f} Hz"
        if self.cfg.noci_relay:
            src += " (hits via VNC-stand-in relay)"
        if self.cfg.nociceptive_input:
            src += f" + hit afferents {self.noci_hz:.0f} Hz (modelled link)"
        return (f"PAIN/AROUSAL {self.level:.2f} [{MODEL_LABEL}] <- {src} | "
                f"step x{self.freq_mult:.2f} stride x{self.amp_mult:.2f}"
                + (f" | jump > {j:.0f} Hz" if j is not None else ""))

    def metric_row(self) -> tuple:
        j = self.jump_hz
        return (self.level, self.oa_rate_hz, self.noci_hz, self.freq_mult, self.amp_mult,
                np.nan if j is None else j)

    def summary(self) -> dict:
        return {"enabled": self.enabled, "level": self.level, "max_level": self.max_level,
                "freq_mult": self.freq_mult, "amp_mult": self.amp_mult,
                "jump_hz": self.jump_hz,
                "config": asdict(self.cfg)}

    def close(self) -> None:
        """Undo everything: frequencies, jump threshold, update hook, brain layer."""
        if not self.enabled:
            return
        if self._freq_target is not None and self._base_freqs is not None:
            self._freq_target[:] = self._base_freqs
        c = self.controller
        if self._our_filter is not None and c is not None and \
                getattr(c, "signal_filter", None) is self._our_filter:
            c.signal_filter = self._prev_filter
        self._our_filter = None
        self.amp_mult = 1.0
        trig = self.triggers
        if trig is not None and self._base_jump_hz is not None:
            trig.p.jump_escape_hz = self._base_jump_hz
        if self._orig_update is not None and self.link is not None:
            try:
                del self.link.update
            except AttributeError:
                pass
            self._orig_update = None
        brain = _get(self.link, "brain") if self.link is not None else None
        if brain is not None and hasattr(brain, "set_neuromod"):
            try:
                brain.set_neuromod(None)
            except Exception:
                pass
        self.freq_mult = 1.0
        self.enabled = False


def install_stress(session_or_parts: Any, cfg: StressConfig | dict | None = None,
                   say: Callable[[str], None] | None = None) -> StressHandle:
    """Wire the stress layer to a ``Session`` (or {"controller", "link", "triggers"}).

    Call after the Session (and its BrainLink.attach) exists. With
    ``cfg.enabled`` False this returns an inert handle and touches nothing.
    """
    if not isinstance(cfg, StressConfig):
        cfg = StressConfig.from_dict(cfg)
    s = session_or_parts
    if isinstance(s, dict):
        controller, link, triggers = s.get("controller"), s.get("link"), s.get("triggers")
    else:
        sim = getattr(s, "sim", None)
        controller = getattr(sim, "controller", None) if sim is not None else \
            getattr(s, "controller", None)
        link = getattr(s, "brain", None) or getattr(s, "link", None)
        triggers = None
        if say is None:
            say = getattr(s, "say", None)
    return StressHandle(cfg, controller=controller, link=link, triggers=triggers, say=say)
