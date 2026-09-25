"""Keyboard -> perturbation actions, plus a one-call installer for the app.

Key names come from ``perpetualfly.interaction.decode_key`` (what
``LiveViewer.poll_keys`` returns). Raw OpenCV ``waitKeyEx`` codes differ per
HighGUI backend; ``decode_key`` normalizes them. On macOS (Cocoa backend) the arrows
arrive as the NSEvent function-key unicode values:

    63232 (0xF700) up, 63233 (0xF701) down, 63234 (0xF702) left, 63235 (0xF703) right

SPACE is 32 and the digits are their ASCII codes (49..52 for "1".."4") everywhere.
``handle`` also accepts raw integer codes and decodes them itself.

Default bindings (hit mode "shove" = external force on the thorax; hit mode "whip" =
the physical whip of ``perpetualfly.interaction.whip`` cracks from that side)::

             shove mode                         whip mode
    SPACE    random direction (+ upward)        crack from a random side
    LEFT     shove to the fly's left            crack from the fly's left (pushes right)
    RIGHT    shove to the fly's right           crack from the fly's right (pushes left)
    UP       shove forward                      crack from the front (pushes backward)
    DOWN     shove backward                     crack from the rear (pushes forward)
    U        shove straight up                  overhead crack (chops down)
    1..4     strength gentle / medium / hard / absurd (both modes)
    H        toggle hit mode whip <-> shove (only if a Whip is installed)
    A        toggle auto-perturb (uses the current hit mode)

Integration in ``app.py`` (one call + one line in the key loop)::

    controls = install_perturbation(sim)            # or (sim, pert_cfg, auto_cfg)
    ...
    for k in keys:
        msg = controls.handle(k)
        if msg:
            print(msg, flush=True)
    hud.append(controls.hud_line())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from perpetualfly.interaction.keyboard import decode_key
from perpetualfly.interaction.perturbation import (
    AutoPerturbConfig,
    AutoPerturber,
    HitEvent,
    Perturbation,
    PerturbationConfig,
)

if TYPE_CHECKING:
    from perpetualfly.interaction.whip import Whip, WhipHitEvent
    from perpetualfly.simulation import Simulation

HIT_MODES = ("whip", "shove")
# Key binding "hit:<direction>" -> whip side in whip mode: the arrow names the side
# the whip comes *from* (LEFT = from the fly's left).
KEY_DIRECTION_TO_WHIP_SIDE = {
    "random": "random", "left": "left", "right": "right", "forward": "front",
    "backward": "rear", "up": "overhead",
}
# Auto-perturber push direction -> whip side producing a push that way (a crack from
# the right pushes the fly to its left, etc.), so AutoPerturbConfig.direction_weights
# keep their meaning in whip mode.
PUSH_DIRECTION_TO_WHIP_SIDE = {
    "random": "random", "left": "right", "right": "left", "forward": "rear",
    "backward": "front", "up": "overhead",
}


def _default_bindings() -> dict[str, str]:
    return {
        "space": "hit:random",
        "left": "hit:left",
        "right": "hit:right",
        "up": "hit:forward",
        "down": "hit:backward",
        "u": "hit:up",
        "1": "level:1",
        "2": "level:2",
        "3": "level:3",
        "4": "level:4",
        "a": "auto:toggle",
        "h": "mode:toggle",
    }


@dataclass
class PerturbationKeyConfig:
    # key name -> action ("hit:<direction>", "level:<n>", "auto:toggle")
    bindings: dict[str, str] = field(default_factory=_default_bindings)


def format_event(ev: HitEvent, names: list[str] | None = None) -> str:
    lvl = ""
    if ev.level is not None:
        lvl = f"L{ev.level}"
        if names:
            lvl += f" {names[ev.level - 1]}"
        lvl += " "
    return (f"[hit/{ev.source}] t={ev.sim_time:.3f}s {lvl}{ev.direction_name} "
            f"{ev.magnitude_uN:.1f} uN ({ev.magnitude_bw:g}x weight) for "
            f"{ev.duration_s * 1e3:.0f} ms, impulse {ev.impulse_uNs * 1e3:.2f} nN*s")


def format_whip_event(ev: "WhipHitEvent", names: list[str] | None = None) -> str:
    lvl = f"L{ev.level}" + (f" {names[ev.level - 1]}" if names and ev.level else "")
    if not ev.hit:
        return f"[whip/{ev.source}] t={ev.sim_time:.3f}s {lvl} {ev.direction_name}: MISS"
    return (f"[whip/{ev.source}] t={ev.sim_time:.3f}s {lvl} {ev.direction_name}: HIT "
            f"{ev.body.split('/')[-1]}, impulse {ev.impulse_uNs * 1e3:.0f} nN*s, peak "
            f"{ev.magnitude_uN:.0f} uN ({ev.magnitude_bw:.0f}x weight), contact "
            f"{ev.duration_s * 1e3:.1f} ms")


class HitRouter:
    """Stands in for the ``Perturbation`` inside the ``AutoPerturber`` so automatic
    hits use the current hit mode: shove -> ``Perturbation.apply_impulse``; whip ->
    ``Whip.crack`` from the side that pushes in the drawn direction."""

    def __init__(self, controls: "PerturbationControls") -> None:
        self.controls = controls

    @property
    def cfg(self) -> PerturbationConfig:  # AutoPerturber reads cfg.body
        return self.controls.perturbation.cfg

    def apply_impulse(self, body, direction, magnitude=None, duration=None, *,
                      level=None, source="auto", rng=None):
        c = self.controls
        if c.mode == "whip" and c.whip is not None:
            side = PUSH_DIRECTION_TO_WHIP_SIDE.get(direction, "random")
            lvl = level if level is not None else c.perturbation.level
            c.whip.crack(side, lvl, source=source, rng=rng)
            return None
        return c.perturbation.apply_impulse(body, direction, magnitude, duration,
                                            level=level, source=source, rng=rng)


class PerturbationControls:
    """Maps key presses to hits (shove or whip) / strength / hit mode / auto toggling."""

    def __init__(self, perturbation: Perturbation, auto: AutoPerturber | None = None,
                 cfg: PerturbationKeyConfig | None = None, whip: "Whip | None" = None,
                 mode: str = "shove") -> None:
        self.perturbation = perturbation
        self.auto = auto
        self.cfg = cfg or PerturbationKeyConfig()
        self.whip = whip
        if mode not in HIT_MODES:
            raise ValueError(f"hit mode must be one of {HIT_MODES}")
        self.mode = mode if whip is not None else "shove"
        if auto is not None:
            auto.perturbation = HitRouter(self)  # auto hits follow the hit mode

    def handle(self, key: str | int | None) -> str | None:
        """Handle one key (name or raw waitKeyEx code). Returns a message if the key
        was consumed, else None (so the app can process it itself)."""
        name = decode_key(key) if isinstance(key, int) else key
        action = self.cfg.bindings.get(name) if name else None
        if action is None:
            return None
        kind, _, arg = action.partition(":")
        p = self.perturbation
        names = [lv.name for lv in p.cfg.levels]
        if kind == "hit":
            if self.mode == "whip" and self.whip is not None:
                side = KEY_DIRECTION_TO_WHIP_SIDE[arg]
                state = self.whip.crack(side, p.level, source="key")
                return (f"[whip/key] crack L{p.level} {p.level_name} from_{side}"
                        + (" (queued after the current crack)" if state == "queued" else ""))
            return format_event(p.hit(arg, source="key"), names)
        if kind == "mode":
            if self.whip is None:
                return "[hit-mode] shove (no whip installed)"
            self.mode = "shove" if self.mode == "whip" else "whip"
            return f"[hit-mode] {self.mode}"
        if kind == "level":
            p.set_level(int(arg))
            lv = p.cfg.levels[p.level - 1]
            msg = (f"[strength] {p.level} {lv.name}: shove {lv.magnitude_bw:g}x weight = "
                   f"{lv.magnitude_bw * p.body_weight_uN:.1f} uN for "
                   f"{lv.duration_s * 1e3:.0f} ms")
            if self.whip is not None:
                wl = self.whip.cfg.levels[p.level - 1]
                msg += f"; whip swing {wl.omega:g} rad/s" + (
                    f" + {wl.lunge:g} mm lunge" if wl.lunge else "")
            return msg
        if kind == "auto":
            if self.auto is None:
                return None
            return f"[auto-perturb] {'on' if self.auto.toggle() else 'off'}"
        raise ValueError(f"unknown key action {action!r}")

    def hud_line(self) -> str:
        p = self.perturbation
        if self.mode == "whip" and self.whip is not None:
            w = self.whip
            n_hit = sum(e.hit for e in w.events)
            s = (f"WHIP L{p.level} {p.level_name}  cracks {w.n_cracks} hit {n_hit} "
                 f"miss {len(w.events) - n_hit}")
            if w.phase != "idle":
                s += f"  [{w.phase}]"
        else:
            s = f"SHOVE L{p.level} {p.level_name}  hits {len(p.events)}"
            if p.is_active:
                s += "  *HIT*"
        if self.auto is not None:
            s += f"  auto {'on' if self.auto.enabled else 'off'}"
        return s

    def detach(self) -> None:
        if self.auto is not None:
            self.auto.detach()
        self.perturbation.detach()
        if self.whip is not None:
            self.whip.detach()


def install_perturbation(
    sim: "Simulation",
    cfg: PerturbationConfig | None = None,
    auto_cfg: AutoPerturbConfig | None = None,
    key_cfg: PerturbationKeyConfig | None = None,
    whip: "Whip | None" = None,
    mode: str = "shove",
) -> PerturbationControls:
    """Attach a ``Perturbation`` (and an ``AutoPerturber`` if ``auto_cfg`` is given)
    to ``sim`` and return the key handler. ``controls.perturbation.listeners`` is
    where an event logger subscribes (``listener(HitEvent)``). With an attached
    ``whip`` (``Whip.attach(sim)`` already done), ``mode`` selects "whip" or "shove"
    for keys and automatic hits; whip results arrive via ``whip.listeners``."""
    pert = Perturbation(sim, cfg)
    auto = AutoPerturber(sim, pert, auto_cfg) if auto_cfg is not None else None
    return PerturbationControls(pert, auto, key_cfg, whip=whip, mode=mode)
