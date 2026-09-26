"""Optional extra body joints (wings, proboscis) for the locomotion fly.

FlyGym 2.1's ``make_locomotion_fly`` uses ``JointPreset.LEGS_ONLY``: only the legs
have joints; head, proboscis (``c_rostrum``, ``c_haustellum``), antennae, wings,
halteres and abdomen are rigid (with ``fusestatic`` the head is fused into the
thorax body; the others are separate welded bodies). ``make_action_fly`` builds the
same fly (same leg joints, stiffness / damping, 42 leg position actuators with
kp=45 and +-65 force range, 6 adhesion actuators, in the same order) and
additionally gives

* ``wings=True``: ``c_thorax-{l,r}_wing-{yaw,pitch,roll}`` hinge joints, each
  with a position actuator ``<joint>-wingpos``;
* ``proboscis=True``: ``c_head-c_rostrum-pitch`` and
  ``c_rostrum-c_haustellum-pitch`` hinges (with ``c_thorax-c_head`` present in
  the skeleton but without any DoF, so the head stays fused to the thorax), each
  with a position actuator ``<joint>-proboscispos``.

The extra actuators are added straight to the ``MjSpec`` (``flygym.utils.mjcf.
add_actuator``), *not* through ``fly.add_actuators``, so FlyGym's
``get_actuated_jointdofs_order("position")`` and
``Simulation._intern_actuatorids_by_type_by_fly[POSITION]`` still list exactly the
42 leg DoFs: the locomotion controller is untouched. Their keyframe ctrl and qpos
are 0 = the rest pose of the meshes (wings folded over the abdomen, proboscis
retracted). Walking with the extra joints is checked in tests/test_actions.py.

Use it with ``Simulation(cfg, fly_factory=make_action_fly_factory(wings=True,
proboscis=True))``; the default Simulation keeps FlyGym's model unchanged.
"""

from __future__ import annotations

from typing import Callable

from flygym.anatomy import (
    ActuatedDOFPreset,
    AnatomicalJoint,
    AxesSet,
    AxisOrder,
    JointPreset,
    PASSIVE_TARSAL_LINKS,
    Skeleton,
)
from flygym.compose import ActuatorType, KinematicPosePreset, NeuroMechFly
from flygym.utils.mjcf import add_actuator

WING_DOFS = tuple(f"c_thorax-{s}_wing-{ax}" for s in "lr" for ax in ("yaw", "pitch", "roll"))
PROBOSCIS_DOFS = ("c_head-c_rostrum-pitch", "c_rostrum-c_haustellum-pitch")

# Wing hinge: 1.3 ug wing, tiny inertia. kp in uN*mm/rad; stiff enough to hold the
# wing against gait accelerations, damping keeps the wing from ringing.
WING_KP = 0.5
WING_FORCE = 2.0
PROBOSCIS_KP = 2.0
PROBOSCIS_FORCE = 5.0
EXTRA_JOINT_DAMPING = 0.01
EXTRA_JOINT_STIFFNESS = 0.0


def make_action_fly(
    name: str = "nmf",
    *,
    wings: bool = False,
    proboscis: bool = False,
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
    """``flygym_demo.complex_terrain.make_locomotion_fly`` + optional wing /
    proboscis joints (see module doc). With both flags False the result is
    equivalent to ``make_locomotion_fly``."""
    neutral_pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(AxisOrder.YAW_PITCH_ROLL)
    joints = JointPreset.LEGS_ONLY.to_joint_list()
    extra: list[AnatomicalJoint] = []
    if proboscis:
        extra += [
            AnatomicalJoint("c_thorax", "c_head", AxesSet([])),  # no DoF: head stays fused
            AnatomicalJoint("c_head", "c_rostrum", AxesSet(["pitch"])),
            AnatomicalJoint("c_rostrum", "c_haustellum", AxesSet(["pitch"])),
        ]
    if wings:
        extra += [AnatomicalJoint("c_thorax", f"{s}_wing", AxesSet(["pitch", "roll", "yaw"]))
                  for s in "lr"]
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, anatomical_joints=joints + extra)
    fly = NeuroMechFly(name=name)
    created = fly.add_joints(skeleton, neutral_pose=neutral_pose,
                             stiffness=joint_stiffness, damping=joint_damping)
    for jointdof, joint in created.items():
        if jointdof.child.link in PASSIVE_TARSAL_LINKS:
            joint.stiffness[0] = passive_tarsus_stiffness
            joint.damping[0] = passive_tarsus_damping
        elif not jointdof.child.is_leg():
            joint.stiffness[0] = EXTRA_JOINT_STIFFNESS
            joint.damping[0] = EXTRA_JOINT_DAMPING
    leg_dofs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(leg_dofs, ActuatorType.POSITION, neutral_input=neutral_pose,
                      kp=actuator_gain, forcerange=actuator_forcerange)
    if add_adhesion:
        fly.add_leg_adhesion(gain=adhesion_gain)
    for jointdof in created:
        if jointdof.name in WING_DOFS:
            kp, fr, tag = WING_KP, WING_FORCE, "wingpos"
        elif jointdof.name in PROBOSCIS_DOFS:
            kp, fr, tag = PROBOSCIS_KP, PROBOSCIS_FORCE, "proboscispos"
        else:
            continue
        add_actuator(fly.mjcf_root, "position", name=f"{jointdof.name}-{tag}",
                     joint=jointdof.name, kp=kp, forcelimited=True, forcerange=(-fr, fr))
    if colorize:
        fly.colorize()
    return fly


def make_action_fly_factory(wings: bool = True, proboscis: bool = True) -> Callable:
    """``fly_factory`` for ``fly_simulator.simulation.Simulation``:
    ``factory(fly_cfg) -> NeuroMechFly``."""

    def factory(fc):
        return make_action_fly(name=fc.name, wings=wings, proboscis=proboscis,
                               add_adhesion=fc.adhesion, colorize=fc.colorize)

    return factory
