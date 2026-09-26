"""Prop-building helpers for eternal jobs (world extensions, before compile).

Contact scheme (docs/API_NOTES.md section 14; ``FLY_BIT`` = 8, ``TERRAIN_BIT`` = 16)
plus one bit for jobs, ``PROP_BIT`` = 32, so moving props can also hit each other:

| kind (``collide=``) | contype | conaffinity | collides with |
|---|---|---|---|
| ``"static"``  | TERRAIN_BIT | FLY_BIT | fly, dynamic props (acts like terrain) |
| ``"dynamic"`` | PROP_BIT | FLY_BIT, TERRAIN_BIT, PROP_BIT | fly, ground plane, static props, other dynamic props |
| ``"fly"``     | 0 | FLY_BIT | the fly only |
| ``"visual"``  | 0 | 0 | nothing (decoration) |

Every colliding prop geom gets ``priority=1`` and FlyGym's stiff ground-contact
parameters (``flygym.compose.ContactParams``), otherwise a 1 mg fly sinks into it.
Give every body an explicit mass: the default density (1000) in model units is
1000 g/mm^3.

Static geoms (on the world body) must never be *moved* at runtime unless the
world-body BVH is refitted (API_NOTES section 8 "BVH gotcha"); geoms on jointed or
mocap bodies can move freely. For props that come and go (grass blades, leaves,
kebab slices), pre-allocate a fixed pool of jointed/mocap bodies and recycle them:
never grow the model.
"""

from __future__ import annotations

import math
from typing import Iterable

import mujoco as mj
import numpy as np

from perpetualfly.terrain import FLY_BIT, TERRAIN_BIT

PROP_BIT = 32

_BITS = {
    "static": (TERRAIN_BIT, FLY_BIT),
    "dynamic": (PROP_BIT, FLY_BIT | TERRAIN_BIT | PROP_BIT),
    "fly": (0, FLY_BIT),
    "visual": (0, 0),
}


def contact_kwargs(collide: str = "static", friction: float | None = None) -> dict:
    """``add_geom`` keyword arguments for the contact kind ``collide`` (see module
    doc). ``friction``: sliding friction (default FlyGym's 1.0)."""
    if collide not in _BITS:
        raise ValueError(f"collide must be one of {sorted(_BITS)}")
    contype, conaffinity = _BITS[collide]
    kw = {"contype": contype, "conaffinity": conaffinity}
    if collide == "visual":
        kw["group"] = 1
        return kw
    from flygym.compose import ContactParams

    cp = ContactParams()
    kw.update(
        priority=1, condim=3,
        friction=(cp.sliding_friction if friction is None else float(friction),
                  cp.torsional_friction, cp.rolling_friction),
        solref=cp.get_solref_tuple(), solimp=cp.get_solimp_tuple(), margin=cp.margin,
    )
    return kw


def quat_axis_angle(axis: Iterable[float], angle: float) -> tuple[float, float, float, float]:
    """(w, x, y, z) of a rotation by ``angle`` rad about ``axis``."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    s = math.sin(angle / 2)
    return (math.cos(angle / 2), float(a[0] * s), float(a[1] * s), float(a[2] * s))


def quat_mul(q1, q2) -> tuple[float, float, float, float]:
    out = np.empty(4)
    mj.mju_mulQuat(out, np.asarray(q1, float), np.asarray(q2, float))
    return tuple(float(v) for v in out)


def quat_pitch(angle: float) -> tuple[float, float, float, float]:
    """Rotation about +y by ``-angle``: a box's +x axis then points *up* at ``angle``
    rad (a ramp rising along +x)."""
    return quat_axis_angle((0.0, 1.0, 0.0), -angle)


def add_box(parent, name: str, half_size, pos, *, quat=(1.0, 0.0, 0.0, 0.0),
            rgba=(0.5, 0.5, 0.5, 1.0), collide: str = "static", friction: float | None = None,
            mass: float | None = None, **extra):
    """Box geom on ``parent`` (``spec.worldbody`` or a body). Returns the geom spec."""
    kw = contact_kwargs(collide, friction)
    if mass is not None:
        kw["mass"] = float(mass)
    kw.update(extra)
    return parent.add_geom(name=name, type=mj.mjtGeom.mjGEOM_BOX,
                           size=tuple(float(v) for v in half_size),
                           pos=tuple(float(v) for v in pos), quat=tuple(quat), rgba=rgba, **kw)


def add_slope(parent, name: str, x0: float, x1: float, z0: float, angle: float,
              half_width: float, *, y: float = 0.0, thickness: float = 1.0, **kw):
    """Box whose *top face* is the plane through (x0, y, z0) rising at ``angle`` rad
    along +x (negative angle: falling), spanning x0..x1 horizontally."""
    c, s = math.cos(angle), math.sin(angle)
    run = x1 - x0
    half_len = 0.5 * run / c
    mid = np.array([x0 + run / 2, y, z0 + run / 2 * math.tan(angle)])
    normal = np.array([-s, 0.0, c])
    center = mid - normal * thickness / 2
    return add_box(parent, name, (half_len, half_width, thickness / 2), center,
                   quat=quat_pitch(angle), **kw)


def wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(v) -> float:
    return math.atan2(float(v[1]), float(v[0]))


LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")
LEG_SEGMENTS = ("coxa", "trochanterfemur", "tibia", "tarsus1", "tarsus2", "tarsus3",
                "tarsus4", "tarsus5")


def exclude_fly_legs(spec: mj.MjSpec, body_name: str, fly_name: str = "nmf",
                     legs: Iterable[str] = LEGS, segments: Iterable[str] = LEG_SEGMENTS) -> int:
    """No contacts between prop body ``body_name`` and the given fly leg segments
    (``spec.add_exclude``; the fly bodies are resolved at compile time, so this works
    in a world extension before ``add_fly``). Use it for props the fly should push
    with its head / body only: sticky tarsi (adhesion) otherwise climb onto a
    light free body and flip the fly over. Returns the number of excludes."""
    n = 0
    segments = tuple(segments)
    for leg in legs:
        for seg in segments:
            spec.add_exclude(bodyname1=body_name, bodyname2=f"{fly_name}/{leg}_{seg}")
            n += 1
    return n


FLY_BODY_GEOMS = {  # contact geom -> the fly body it belongs to (head is fused to the thorax)
    "c_head": "c_thorax", "c_thorax": "c_thorax", "c_abdomen12": "c_abdomen12",
    "c_abdomen3": "c_abdomen3", "c_abdomen4": "c_abdomen4", "c_abdomen5": "c_abdomen5",
    "c_abdomen6": "c_abdomen6",
}


def slippery_body_contact(spec: mj.MjSpec, body_name: str, geom_name: str,
                          fly_name: str = "nmf", friction: float = 0.05,
                          legs: bool = False) -> None:
    """Make the fly's head / thorax / abdomen touch prop geom ``geom_name`` (on body
    ``body_name``) with low friction via explicit ``<pair>``s, and exclude the rest of
    the fly (legs too unless ``legs``) from dynamic contact with that body.

    Why: a prop the fly pushes with its head rolls, and with friction 1 its surface
    drags the head up (the fly rears and flips over backward). Legs are excluded
    because the sticky tarsi (adhesion) climb onto light props. The prop geom keeps
    its normal friction with the ground / static props.
    """
    from flygym.compose import ContactParams

    cp = ContactParams()
    for g, b in FLY_BODY_GEOMS.items():
        spec.add_pair(geomname1=geom_name, geomname2=f"{fly_name}/{g}", condim=3,
                      friction=[friction, friction, cp.torsional_friction,
                                cp.rolling_friction, cp.rolling_friction],
                      solref=list(cp.get_solref_tuple()), solimp=list(cp.get_solimp_tuple()),
                      margin=cp.margin)
    for b in sorted(set(FLY_BODY_GEOMS.values())):
        spec.add_exclude(bodyname1=body_name, bodyname2=f"{fly_name}/{b}")
    if not legs:
        exclude_fly_legs(spec, body_name, fly_name)


def add_plane_box(parent, name: str, top_point, x_axis, normal, half_x: float, half_y: float,
                  *, thickness: float = 1.0, **kw):
    """Box whose top face lies in the plane through ``top_point`` with ``normal``,
    centred on ``top_point``, its long side along ``x_axis`` (projected into the
    plane). For banked / tilted surfaces (e.g. a ramp with sloping side banks)."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    x = np.asarray(x_axis, float)
    x = x - (x @ n) * n
    x = x / np.linalg.norm(x)
    y = np.cross(n, x)
    q = np.empty(4)
    mj.mju_mat2Quat(q, np.column_stack([x, y, n]).ravel())
    center = np.asarray(top_point, float) - n * thickness / 2
    return add_box(parent, name, (half_x, half_y, thickness / 2), center,
                   quat=tuple(float(v) for v in q), **kw)
