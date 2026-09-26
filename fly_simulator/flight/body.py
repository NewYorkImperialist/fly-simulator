"""Flight-capable NeuroMechFly: stroke-plane wing hinges, strong wing servos and
MuJoCo's ellipsoid fluid model on the wings.

Wing frame (verified from the mesh vertices, ``scratch probe``): in the NMF wing
body frame the span runs along +y (left wing, 0..2.3 mm) / -y (right wing), the
chord along x with the *leading edge at +x* (costal edge, lateral when folded),
and +z is the dorsal surface. The rigging quaternion folds the wing over the
abdomen. Here the wing body quaternion is replaced by a rotation about the thorax
y axis by ``stroke_plane_deg`` (beta): at zero joint angles both wings are spread
laterally, flat, leading edge forward, in a stroke plane tilted nose-down by beta
relative to the thorax. With beta = 47.5 deg (flybody / real *Drosophila*) the
stroke plane is horizontal when the body pitches nose-up by 47.5 deg -- the hover
posture, in which the wing hinge sits almost exactly above the centre of mass.

Joints per wing (applied in this order, each axis carried along by the previous):

* ``stroke``    (phi): about the stroke-plane normal; + = wing forward (both sides);
* ``deviation`` (theta): about the chord; + = wing tip up (dorsal);
* ``rotation``  (psi): about the span; + = leading edge up (0 = flat).

All three have a position actuator ``<joint>-flight`` (kp, force range well
above what flapping at 200+ Hz needs: ~11 uN*mm inertia + 5-8 uN*mm air loads);
the servo damping kv is the joint damping (implicit in MuJoCo's Euler
integrator), and the wingbeat generator adds the velocity feed-forward
(ctrl = q_ref + kv/kp * dq_ref).
The actuators are added straight to the ``MjSpec`` like
``fly_simulator.actions.body``: FlyGym still sees exactly the 42 leg position
actuators, so the locomotion controller is untouched.

Aerodynamics: an invisible ellipsoid geom per wing (``fluidshape=ellipsoid``,
flybody's ``fluidcoef = [1.0, 0.5, 1.5, 1.7, 1.0]`` = blunt drag, slender drag,
angular drag, Kutta lift, Magnus lift), mass 0, no contacts, group 3 (hidden). The
wing meshes stay the visuals and carry the 2.5 ug wing mass. Air density /
viscosity are set on ``model.opt`` by :func:`apply_air` (mm-g-s units). With a
non-zero density, bodies without an ellipsoid geom (legs, body) get MuJoCo's
inertia-box drag model: a small, physical body drag.

MuJoCo force sensors do not include fluid forces; the aerodynamic force is read
from ``data.qfrc_fluid`` (see ``fly_simulator.flight.sim``).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from flygym.anatomy import (
    ActuatedDOFPreset,
    AxisOrder,
    BodySegment,
    JointPreset,
    PASSIVE_TARSAL_LINKS,
    Skeleton,
)
from flygym.compose import ActuatorType, KinematicPosePreset, NeuroMechFly
from flygym.utils.mjcf import add_actuator

import mujoco as mj

# Air in mm-g-s units: flybody's 0.00128 g/cm^3 and 0.000185 g/(cm*s) (cm-g-s).
AIR_DENSITY = 1.28e-6  # g/mm^3
AIR_VISCOSITY = 1.85e-5  # g/(mm*s)
# flybody's fluid coefficients: blunt drag, slender drag, angular drag, Kutta
# lift, Magnus lift.
WING_FLUIDCOEF = (1.0, 0.5, 1.5, 1.7, 1.0)
# Wing ellipsoid semi-axes (mm): thickness, half chord, half span. NMF wing mesh:
# 2.30 mm from the hinge, 1.08 mm chord.
WING_ELLIPSOID = (0.005, 0.55, 1.15)
# Ellipsoid centre in the wing body frame (the hinge is the origin): mid span,
# centred on the chord of the mesh (x from -0.68 to +0.41 mm).
WING_ELLIPSOID_POS = (-0.13, 1.15, 0.05)

FLIGHT_TIMESTEP = 5e-5  # s, as flybody
STROKE_PLANE_DEG = 47.5

WING_JOINTS = ("stroke", "deviation", "rotation")
# Joint axes in the wing body frame (see module doc for the sign conventions).
_AXES = {
    "l": {"stroke": (0, 0, -1), "deviation": (1, 0, 0), "rotation": (0, -1, 0)},
    "r": {"stroke": (0, 0, 1), "deviation": (-1, 0, 0), "rotation": (0, -1, 0)},
}
# Servo gains (uN*mm/rad, uN*mm*s/rad) and torque limits (uN*mm). The wing's
# rotational inertia about the hinge is ~1e-6 g*mm^2 (stroke) and ~1e-7
# (rotation): natural frequencies ~1.1-2 kHz, near-critically damped.
# The force range bounds the *actuator* output kp (q_ref - q) + kv dq_ref, which
# includes the feed-forward that cancels the joint damping; the net servo torque
# (actuator - kv dq, see FlightSimulation.wing_servo_torque) stays ~20 uN*mm.
WING_GAINS = {
    "stroke": (60.0, 0.012, 80.0),
    "deviation": (60.0, 0.012, 80.0),
    "rotation": (15.0, 0.002, 40.0),
}
WING_JOINT_DAMPING = 1e-4


def wing_joint_name(side: str, joint: str) -> str:
    return f"c_thorax-{side}_wing-{joint}"


def wing_actuator_name(side: str, joint: str) -> str:
    return f"{wing_joint_name(side, joint)}-flight"


def _quat_about_y(deg: float) -> tuple[float, float, float, float]:
    h = np.radians(deg) / 2.0
    return (float(np.cos(h)), 0.0, float(np.sin(h)), 0.0)


def make_flight_fly(
    name: str = "nmf",
    *,
    stroke_plane_deg: float = STROKE_PLANE_DEG,
    wing_gains: dict | None = None,
    ellipsoid: tuple[float, float, float] = WING_ELLIPSOID,
    ellipsoid_pos: tuple[float, float, float] = WING_ELLIPSOID_POS,
    fluidcoef: tuple[float, ...] = WING_FLUIDCOEF,
    joint_stiffness: float = 0.05,
    joint_damping: float = 0.06,
    passive_tarsus_stiffness: float = 7.5,
    passive_tarsus_damping: float = 1e-2,
    actuator_gain: float = 45.0,
    actuator_forcerange: tuple[float, float] = (-65.0, 65.0),
    add_adhesion: bool = True,
    adhesion_gain: float = 40.0,
    colorize: bool = False,
) -> NeuroMechFly:
    """FlyGym's locomotion fly (legs-only joints, 42 leg position actuators,
    adhesion) + stroke-plane wing hinges, wing servos and fluid ellipsoids."""
    gains = dict(WING_GAINS)
    gains.update(wing_gains or {})
    neutral_pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL,
                        anatomical_joints=JointPreset.LEGS_ONLY.to_joint_list())
    fly = NeuroMechFly(name=name)
    created = fly.add_joints(skeleton, neutral_pose=neutral_pose,
                             stiffness=joint_stiffness, damping=joint_damping)
    for jointdof, joint in created.items():
        if jointdof.child.link in PASSIVE_TARSAL_LINKS:
            joint.stiffness[0] = passive_tarsus_stiffness
            joint.damping[0] = passive_tarsus_damping
    leg_dofs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(leg_dofs, ActuatorType.POSITION, neutral_input=neutral_pose,
                      kp=actuator_gain, forcerange=actuator_forcerange)
    if add_adhesion:
        fly.add_leg_adhesion(gain=adhesion_gain)

    spec = fly.mjcf_root
    quat = _quat_about_y(stroke_plane_deg)
    for side in "lr":
        body = fly.bodyseg_to_mjcfbody[BodySegment(f"{side}_wing")]
        body.quat = quat
        for jn in WING_JOINTS:
            # The servo damping kv is joint damping: MuJoCo's Euler integrator
            # treats dof damping implicitly (actuator kv would be explicit and
            # blows up at dt 5e-5; implicitfast is ~6x slower with the fluid model).
            body.add_joint(name=wing_joint_name(side, jn), type=mj.mjtJoint.mjJNT_HINGE,
                           axis=_AXES[side][jn], pos=(0, 0, 0),
                           damping=gains[jn][1] + WING_JOINT_DAMPING)
        epos = (ellipsoid_pos[0], ellipsoid_pos[1] if side == "l" else -ellipsoid_pos[1],
                ellipsoid_pos[2])
        # semi-axes along the body frame: x = chord, y = span, z = thickness
        body.add_geom(
            name=f"{side}_wing_fluid", type=mj.mjtGeom.mjGEOM_ELLIPSOID,
            size=[ellipsoid[1], ellipsoid[2], ellipsoid[0]], pos=epos, mass=0.0,
            contype=0, conaffinity=0, group=3, rgba=(0.3, 0.6, 1.0, 0.3),
            fluid_ellipsoid=1, fluid_coefs=list(fluidcoef),
        )
        for jn in WING_JOINTS:
            kp, kv, fr = gains[jn]
            add_actuator(spec, "position", name=wing_actuator_name(side, jn),
                         joint=wing_joint_name(side, jn), kp=kp,
                         forcelimited=True, forcerange=(-fr, fr))
    if colorize:
        fly.colorize()
    return fly


def make_flight_fly_factory(**kwargs) -> Callable:
    """``fly_factory`` for ``fly_simulator.simulation.Simulation``."""

    def factory(fc):
        return make_flight_fly(name=fc.name, add_adhesion=fc.adhesion,
                               colorize=fc.colorize, **kwargs)

    return factory


def apply_air(model: mj.MjModel, density: float = AIR_DENSITY,
              viscosity: float = AIR_VISCOSITY) -> None:
    """Turn on MuJoCo's fluid forces (the NMF model ships with density 0)."""
    model.opt.density = density
    model.opt.viscosity = viscosity
