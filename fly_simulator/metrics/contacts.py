"""Cheap, terrain-agnostic contact summary for the fly.

FlyGym creates explicit ``<pair>``s between fly geoms and ``world.ground_geoms`` only
(there is no fly self-collision, see docs/API_NOTES.md §7). So *every* active contact
that involves a fly geom is a fly–terrain contact, whatever terrain geoms exist
(plane, pooled boxes, hfield, mocap rocks). That lets us classify contacts from
``data.contact.geom1/geom2`` alone, without the per-leg sensors (which only exist when
the world has exactly one ground geom).

Exception: geoms of interaction objects that hit the fly (the physical whip,
``fly_simulator.interaction.whip``, geom names ``whip/...``) are not terrain. Contacts
involving them are ignored here, so a whip strike on the thorax is never counted as
"body touches the ground" by the fall detector or in metrics.csv.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco as mj
import numpy as np

LEG_NAMES = ("lf", "lm", "lh", "rf", "rm", "rh")
# Geoms whose ground contact means "the body touches the ground", i.e. not a leg.
# ``c_head`` is fused into the thorax body (fusestatic) but keeps its own geom.
BODY_GEOM_PREFIXES = ("c_thorax", "c_head", "c_abdomen")
# Non-terrain geoms that touch the fly (contacts with them are ignored).
IGNORED_GEOM_PREFIXES = ("whip/",)

_NONE = -1
_IGNORE = -2
_BODY = 6  # geom class ids: 0..5 = legs (LEG_NAMES order), 6 = body


@dataclass
class ContactSummary:
    n_contacts: int  # fly–terrain contacts
    legs_in_contact: int  # number of legs (0..6) with any segment touching terrain
    leg_contact: np.ndarray  # bool (6,), LEG_NAMES order
    body_contact: bool  # thorax / head / abdomen touches terrain
    body_contacts: int  # number of body-geom contacts


class ContactClassifier:
    """Maps geom ids to {leg i, body, other} once; ``summarize(data)`` is vectorized."""

    def __init__(self, model: mj.MjModel, fly_name: str = "nmf") -> None:
        prefix = f"{fly_name}/"
        cls = np.full(model.ngeom, _NONE, dtype=np.int64)
        for g in range(model.ngeom):
            name = model.geom(g).name
            if name.startswith(IGNORED_GEOM_PREFIXES):
                cls[g] = _IGNORE
                continue
            if not name.startswith(prefix):
                continue
            short = name[len(prefix):]
            if short.startswith(BODY_GEOM_PREFIXES):
                cls[g] = _BODY
            elif short[:2] in LEG_NAMES and short[2:3] == "_":
                cls[g] = LEG_NAMES.index(short[:2])
        self.geom_class = cls
        self._has_ignored = bool(np.any(cls == _IGNORE))

    def summarize(self, data: mj.MjData) -> ContactSummary:
        n = int(data.ncon)
        if n == 0:
            return ContactSummary(0, 0, np.zeros(6, bool), False, 0)
        # data.contact.geom1/2 are numpy views of length ncon.
        c1 = self.geom_class[data.contact.geom1[:n]]
        c2 = self.geom_class[data.contact.geom2[:n]]
        if self._has_ignored:
            keep = (c1 != _IGNORE) & (c2 != _IGNORE)
            c1, c2 = c1[keep], c2[keep]
        classes = np.concatenate([c1, c2])
        fly = (c1 != _NONE) | (c2 != _NONE)
        leg_contact = np.zeros(6, bool)
        legs = classes[(classes >= 0) & (classes < 6)]
        leg_contact[legs] = True
        body_contacts = int(np.count_nonzero(classes == _BODY))
        return ContactSummary(
            n_contacts=int(np.count_nonzero(fly)),
            legs_in_contact=int(leg_contact.sum()),
            leg_contact=leg_contact,
            body_contact=body_contacts > 0,
            body_contacts=body_contacts,
        )
