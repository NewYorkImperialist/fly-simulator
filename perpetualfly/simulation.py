"""Physics simulation of a walking NeuroMechFly (no rendering in here).

``Simulation`` owns FlyGym's ``flygym.Simulation`` plus the locomotion controller and
exposes what later modules (terrain, perturbation, fall detection, metrics, RL) need:

* ``model`` / ``data``: the raw ``mujoco.MjModel`` / ``mujoco.MjData``
  (FlyGym 2.x uses the native MuJoCo bindings, not dm_control).
* ``pre_step_hooks`` / ``post_step_hooks``: callables ``hook(sim)`` run around every
  ``mj_step`` (e.g. a perturbation module writes ``data.xfrc_applied`` in a pre-step
  hook and clears it once its duration elapsed).
* ``pre_reset_hooks``: run at the top of ``reset()``, before FlyGym's keyframe reset
  (e.g. terrain rebuilds its layout so the fly never settles onto stale features).
* ``reset_hooks``: run after ``reset()``.
* ``world_extensions``: callables ``ext(world)`` applied to the ``MjSpec`` world
  after ``world_factory()`` and *before* ``world.add_fly()`` (FlyGym writes the
  "neutral" keyframe in ``add_fly`` sized to the model at that moment, so extra
  bodies/joints must already exist). See docs/API_NOTES.md section 14.
* thorax pose / velocity accessors.

Contacts: the default world is ``perpetualfly.terrain.TerrainWorld`` without any
pooled geoms. The fly's contact geoms get ``contype = FLY_BIT`` (8),
``conaffinity = 0``; a geom with ``contype = 0, conaffinity = FLY_BIT`` therefore
collides with the fly (and with nothing else) without explicit pairs.
"""

from __future__ import annotations

from typing import Callable

import mujoco as mj
import numpy as np

from flygym import Simulation as FlyGymSimulation
from flygym.anatomy import ContactBodiesPreset
from flygym.compose import BaseWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import make_locomotion_fly

from perpetualfly.config import AppConfig
from perpetualfly.controllers import LocomotionController
from perpetualfly.terrain import GroundRecentering, TerrainWorld

Hook = Callable[["Simulation"], None]
WorldExtension = Callable[[BaseWorld], None]


_BAD_STATE_WARNINGS = (
    mj.mjtWarning.mjWARN_BADQACC, mj.mjtWarning.mjWARN_BADQPOS, mj.mjtWarning.mjWARN_BADQVEL,
)


class SimulationInstabilityError(RuntimeError):
    """Raised when the physics state becomes non-finite or blows up."""


class Simulation:
    def __init__(
        self,
        cfg: AppConfig | None = None,
        world_factory: Callable[[], BaseWorld] | None = None,
        world_extensions: list[WorldExtension] | tuple[WorldExtension, ...] = (),
    ) -> None:
        self.cfg = cfg or AppConfig()
        fc = self.cfg.fly

        # FlyGym's standard locomotion fly: legs-only joints, position actuators on
        # the 42 active leg DoFs, passive tarsi, adhesion actuators on tarsus5.
        self.fly = make_locomotion_fly(
            name=fc.name, add_adhesion=fc.adhesion, colorize=fc.colorize
        )
        if world_factory is None:
            # Flat ground (single plane, explicit FlyGym pairs) with the FLY_BIT
            # contype scheme on the fly geoms, so world extensions can collide
            # with the fly. Without pooled geoms this is physically identical to
            # FlyGym's FlatGroundWorld (no dynamic contacts can arise: the fly
            # geoms have conaffinity 0 and the plane has contype 0).
            tc = self.cfg.terrain
            self.world = TerrainWorld(tc.ground_half_size, tc.checker_size_mm)
        else:
            self.world = world_factory()
        # Extra bodies (e.g. a physical whip) must be in the spec before add_fly.
        self.world_extensions = list(world_extensions)
        for ext in self.world_extensions:
            ext(self.world)
        self.world.add_fly(
            self.fly,
            [0.0, 0.0, fc.spawn_height],
            Rotation3D("quat", [1, 0, 0, 0]),  # facing +x
            bodysegs_with_ground_contact=ContactBodiesPreset(fc.ground_contact),
            # Per-leg contact sensors only exist if the world has exactly one ground
            # geom; they are cheap, so request them (FlyGym silently skips otherwise).
            add_ground_contact_sensors=True,
        )
        # Compiles the MjSpec -> MjModel/MjData and resets to the "neutral" keyframe.
        self.fg = FlyGymSimulation(self.world, timestep=self.cfg.sim.timestep)
        self.model: mj.MjModel = self.fg.mj_model
        self.data: mj.MjData = self.fg.mj_data

        # By default MuJoCo silently calls mj_resetData() when qpos/qvel/qacc become
        # NaN/huge (mjWARN_BADQACC etc.), which would teleport the fly back to the
        # origin and hide the blow-up. Disable that so we can fail loudly instead.
        self.model.opt.disableflags |= mj.mjtDisableBit.mjDSBL_AUTORESET
        if not self.cfg.sim.compute_energy:
            # Energy is diagnostics only (data.energy); the dynamics are unaffected.
            self.model.opt.enableflags &= ~int(mj.mjtEnableBit.mjENBL_ENERGY)

        self.fly_name = fc.name
        # Compiled element names are prefixed with "<fly name>/".
        self.thorax_body_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, f"{self.fly_name}/c_thorax"
        )
        # The fly's free joint is named after the fly; qpos[adr:adr+7] = pos + quat.
        free_jnt = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, self.fly_name)
        self._free_qpos = int(self.model.jnt_qposadr[free_jnt])
        self._free_qvel = int(self.model.jnt_dofadr[free_jnt])
        self.fly_mass = float(self.model.body_subtreemass[self.thorax_body_id])  # g
        # DoFs of the fly (thorax subtree: free joint + leg joints). check_stability
        # applies max_abs_qvel to these; other DoFs (e.g. a whip added by a world
        # extension) get NaN checks and the looser max_abs_qvel_other.
        m = self.model
        fly_dof = m.body_rootid[m.dof_bodyid] == self.thorax_body_id
        self.fly_dofs = np.flatnonzero(fly_dof)
        self.other_dofs = np.flatnonzero(~fly_dof)
        self._all_dofs_fly = self.other_dofs.size == 0
        self._fix_neutral_keyframe_for_extensions()

        self.control_every_steps = max(1, int(self.cfg.sim.control_every_steps))
        self.controller = LocomotionController(
            self.fg, self.fly_name, self.cfg.controller,
            control_timestep=self.control_every_steps * self.timestep,
        )

        self.pre_step_hooks: list[Hook] = []
        self.post_step_hooks: list[Hook] = []
        self.pre_reset_hooks: list[Hook] = []
        self.reset_hooks: list[Hook] = []
        self.step_count = 0
        if world_factory is None:
            tc = self.cfg.terrain
            self.post_step_hooks.append(
                GroundRecentering("ground_plane", tc.ground_half_size, tc.checker_size_mm)
            )
        self.reset()

    def _fix_neutral_keyframe_for_extensions(self) -> None:
        """FlyGym's ``_rebuild_neutral_keyframe`` (run in ``add_fly``) fills only the
        fly's joints (and the fly free joint) into the "neutral" keyframe and leaves
        every other joint at qpos = 0: a free joint from a world extension would be
        reset to the origin with a zero quaternion. Use the compiled rest pose
        (``qpos0``, i.e. the body poses as written in the spec) for those joints,
        and the spec poses for mocap bodies. No-op without extension joints."""
        m = self.model
        key = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, "neutral")
        if key < 0:
            return
        for j in range(m.njnt):
            if m.body_rootid[m.jnt_bodyid[j]] == self.thorax_body_id:
                continue
            a = int(m.jnt_qposadr[j])
            n = {int(mj.mjtJoint.mjJNT_FREE): 7, int(mj.mjtJoint.mjJNT_BALL): 4}.get(
                int(m.jnt_type[j]), 1)
            m.key_qpos[key, a:a + n] = m.qpos0[a:a + n]
        for b in np.flatnonzero(m.body_mocapid >= 0):
            i = int(m.body_mocapid[b])
            # key_mpos / key_mquat are flat (nkey, 3*nmocap) / (nkey, 4*nmocap)
            m.key_mpos[key, 3 * i:3 * i + 3] = m.body_pos[b]
            m.key_mquat[key, 4 * i:4 * i + 4] = m.body_quat[b]

    # ------------------------------------------------------------------ control
    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def time(self) -> float:
        return float(self.data.time)

    def reset(self) -> None:
        """Explicit reset to FlyGym's neutral keyframe + standing warmup."""
        # mj_resetDataKeyframe("neutral"); clears xfrc_applied, ctrl, warnings. Model
        # fields (e.g. a re-centred ground geom_pos) are *not* reset.
        for hook in self.pre_reset_hooks:
            hook(self)
        self.fg.reset()
        self.controller.reset()
        self.controller.apply(self.controller.initial_action())
        self.fg.warmup(self.cfg.sim.warmup_s)  # plain mj_step()s with neutral pose
        self.step_count = 0
        self.check_stability()
        for hook in self.reset_hooks:
            hook(self)

    def step(self, n: int = 1) -> None:
        """Advance ``n`` physics steps.

        Per physics step: controller (every ``control_every_steps`` steps; default
        every step), pre-step hooks, one ``mj_step``, post-step hooks, and every
        ``check_every_steps`` a stability check. With no hooks registered, the
        steps between two controller updates are handed to MuJoCo in one
        ``mj_step(model, data, nstep)`` call (same result as nstep single calls).
        """
        m, d = self.model, self.data
        mj_step = mj.mj_step
        control = self.controller.step_and_apply
        pre, post = self.pre_step_hooks, self.post_step_hooks
        k = self.control_every_steps
        check_every = self.cfg.sim.check_every_steps
        remaining = n
        while remaining > 0:
            if self.step_count % k == 0:
                control()
            if pre or post:
                for hook in pre:
                    hook(self)
                mj_step(m, d)
                self.step_count += 1
                remaining -= 1
                for hook in post:
                    hook(self)
                if self.step_count % check_every == 0:
                    self.check_stability()
            else:
                # up to the next controller update, stability check or the end
                count = self.step_count
                nstep = min(remaining, k - count % k, check_every - count % check_every)
                if nstep == 1:
                    mj_step(m, d)
                else:
                    mj_step(m, d, nstep)
                self.step_count = count + nstep
                remaining -= nstep
                if self.step_count % check_every == 0:
                    self.check_stability()

    # --------------------------------------------------------------- stability
    def check_stability(self) -> None:
        d = self.data
        sc = self.cfg.sim
        # Fast path (runs every check_every_steps): sums propagate NaN/inf, one
        # reduction per array; the detailed diagnosis below only runs on failure.
        if (
            np.isfinite(d.qpos.sum() + d.qacc.sum())
            and self._qvel_ok()
            and d.xpos[self.thorax_body_id, 2] > sc.min_thorax_z
            and d.warning[_BAD_STATE_WARNINGS[0]].number == 0
            and d.warning[_BAD_STATE_WARNINGS[1]].number == 0
            and d.warning[_BAD_STATE_WARNINGS[2]].number == 0
        ):
            return
        problems = []
        if not np.all(np.isfinite(d.qpos)):
            problems.append("non-finite qpos")
        if not np.all(np.isfinite(d.qvel)):
            problems.append("non-finite qvel")
        if not np.all(np.isfinite(d.qacc)):
            problems.append("non-finite qacc")
        fly_qvel = np.abs(d.qvel[self.fly_dofs])
        max_qvel = float(np.nanmax(fly_qvel)) if fly_qvel.size else 0.0
        if max_qvel > sc.max_abs_qvel:
            problems.append(f"fly |qvel| {max_qvel:.3g} > {sc.max_abs_qvel:.3g}")
        if self.other_dofs.size:
            other = float(np.nanmax(np.abs(d.qvel[self.other_dofs])))
            if other > sc.max_abs_qvel_other:
                problems.append(f"non-fly |qvel| {other:.3g} > {sc.max_abs_qvel_other:.3g}")
        z = float(d.xpos[self.thorax_body_id, 2])
        if not z > self.cfg.sim.min_thorax_z:
            problems.append(f"thorax z {z:.3g} below {self.cfg.sim.min_thorax_z}")
        for w in _BAD_STATE_WARNINGS:
            if d.warning[w].number > 0:
                problems.append(f"MuJoCo warning {w.name} x{d.warning[w].number}")
        if problems:
            raise SimulationInstabilityError(self._diagnostics(problems))

    def _qvel_ok(self) -> bool:
        """|qvel| limits: max_abs_qvel on the fly's DoFs, max_abs_qvel_other on the
        rest (NaN compares False, so it fails here too)."""
        qvel, sc = self.data.qvel, self.cfg.sim
        if self._all_dofs_fly:
            return float(np.abs(qvel).max(initial=0.0)) <= sc.max_abs_qvel
        return (float(np.abs(qvel[self.fly_dofs]).max(initial=0.0)) <= sc.max_abs_qvel
                and float(np.abs(qvel[self.other_dofs]).max(initial=0.0))
                <= sc.max_abs_qvel_other)

    def _diagnostics(self, problems: list[str]) -> str:
        m, d = self.model, self.data
        lines = [
            "Physics became unstable: " + "; ".join(problems),
            f"  sim time {d.time:.5f} s, steps since reset {self.step_count}, dt {m.opt.timestep}",
            f"  thorax pos {np.round(d.xpos[self.thorax_body_id], 4)} quat "
            f"{np.round(d.xquat[self.thorax_body_id], 4)}",
            f"  ncon {d.ncon}, xfrc_applied on thorax {d.xfrc_applied[self.thorax_body_id]}",
        ]
        limit = np.full(m.nv, self.cfg.sim.max_abs_qvel_other)
        limit[self.fly_dofs] = self.cfg.sim.max_abs_qvel
        bad = np.flatnonzero(~np.isfinite(d.qvel) | (np.abs(d.qvel) > limit))
        if bad.size == 0 and np.all(np.isfinite(d.qvel)):
            bad = np.argsort(-np.abs(d.qvel))[:5]
        for dof in bad[:10]:
            jnt = int(m.dof_jntid[dof])
            who = "fly" if m.body_rootid[m.dof_bodyid[dof]] == self.thorax_body_id else "non-fly"
            lines.append(f"  dof {dof} ({who} joint '{m.joint(jnt).name}'): qvel={d.qvel[dof]:.4g}")
        return "\n".join(lines)

    # ---------------------------------------------------------------- accessors
    def thorax_position(self) -> np.ndarray:
        """World position of the thorax body frame (mm), copy."""
        return self.data.xpos[self.thorax_body_id].copy()

    def thorax_quat(self) -> np.ndarray:
        """World orientation (w, x, y, z) of the thorax, copy."""
        return self.data.xquat[self.thorax_body_id].copy()

    def thorax_rotmat(self) -> np.ndarray:
        """3x3 world rotation; columns are the thorax x (forward), y (left), z (up)."""
        return self.data.xmat[self.thorax_body_id].reshape(3, 3).copy()

    def heading(self) -> float:
        """Yaw angle (rad) of the thorax forward axis projected on the ground plane."""
        fwd = self.data.xmat[self.thorax_body_id].reshape(3, 3)[:, 0]
        return float(np.arctan2(fwd[1], fwd[0]))

    def thorax_linvel(self) -> np.ndarray:
        """World-frame linear velocity (mm/s) of the free-joint origin.

        MuJoCo free joint convention: qvel[0:3] is the linear velocity in the world
        frame, qvel[3:6] the angular velocity in the *body (local) frame*. The free
        joint sits on the thorax body.
        """
        a = self._free_qvel
        return self.data.qvel[a : a + 3].copy()

    def thorax_angvel_local(self) -> np.ndarray:
        """Angular velocity (rad/s) in the thorax frame."""
        a = self._free_qvel
        return self.data.qvel[a + 3 : a + 6].copy()

    def thorax_angvel_world(self) -> np.ndarray:
        return self.thorax_rotmat() @ self.thorax_angvel_local()

    def tilt_deg(self) -> float:
        """Angle between the thorax up-axis and world up (0 = upright)."""
        up = self.data.xmat[self.thorax_body_id].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))

    def close(self) -> None:
        self.fg.close()
