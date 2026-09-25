"""Locomotion controllers built on FlyGym's own CPG / hybrid controllers.

FlyGym 2.x ships its locomotion controllers in the ``flygym_demo.complex_terrain``
package (installed alongside ``flygym``), not in ``flygym`` itself.

``FastHybridController`` subclasses FlyGym's ``HybridController`` and produces the
*same* actions (verified in tests/test_sim.py) but avoids the per-step
construction/hashing of ~130 ``JointDOF`` objects and ``list.index`` lookups that
dominate the reference implementation's runtime (~4x slower than the physics step).
All the reflex logic (retraction / stumbling rules, CPG integration, preprogrammed
step splines) is still FlyGym's code, inherited unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco as mj
import numpy as np

from flygym.anatomy import JointDOF
from flygym.compose import ActuatorType
from flygym_demo.complex_terrain import (
    CPGController,
    HybridController,
    HybridControllerObservation,
    HybridTurningController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    dof_spec_to_jointdof,
    get_default_locomotion_dof_order,
    make_tripod_cpg_network,
)
from flygym_demo.complex_terrain.hybrid_controller import (
    _CORRECTION_VECTORS,
    _DETECTED_STUMBLING_LINKS,
    _RIGHT_LEG_CORRECTION_SIGN,
)

from perpetualfly.config import ControllerConfig

if TYPE_CHECKING:
    from flygym import Simulation as FlyGymSimulation


class FastObservationBuilder:
    """Builds ``HybridControllerObservation`` with precomputed indices.

    Numerically equivalent to ``HybridControllerObservation.from_sim``.
    """

    def __init__(self, sim: "FlyGymSimulation", fly_name: str, legs: tuple[str, ...]):
        fly = sim.world.fly_lookup[fly_name]
        seg_cls = type(fly).BODY_SEGMENT_CLASS
        body_order = fly.get_bodysegs_order()
        body_ids = sim._internal_bodyids_by_fly[fly_name]
        self._sim = sim
        self._thorax_body_id = int(body_ids[body_order.index(seg_cls("c_thorax"))])
        self._tarsus5_body_ids = np.array(
            [body_ids[body_order.index(seg_cls(f"{leg}_tarsus5"))] for leg in legs]
        )
        geom_by_seg = sim._internal_geomid_by_bodyseg_by_fly[fly_name]
        detected = [
            seg_cls(f"{leg}_{link}") for leg in legs for link in _DETECTED_STUMBLING_LINKS
        ]
        self._n_legs = len(legs)
        self._n_links = len(_DETECTED_STUMBLING_LINKS)
        self._geom_to_row = {int(geom_by_seg[s]): i for i, s in enumerate(detected)}
        # Boolean lookup tables over geom ids replace per-step np.isin() calls
        # (4x np.isin was ~2/3 of the observation cost).
        ngeom = sim.mj_model.ngeom
        self._is_detected = np.zeros(ngeom, dtype=bool)
        self._is_detected[list(self._geom_to_row)] = True
        self._is_ground = np.zeros(ngeom, dtype=bool)
        self._is_ground[np.asarray(sim._internal_ground_geom_ids, dtype=np.int64)] = True
        self._wrench = np.zeros(6)

    def build(self) -> HybridControllerObservation:
        m, d = self._sim.mj_model, self._sim.mj_data
        forces = np.zeros((self._n_legs * self._n_links, 3))
        ncon = d.ncon
        if ncon:
            con = d.contact
            geom = con.geom[:ncon]
            g1, g2 = geom[:, 0], geom[:, 1]
            det, gnd = self._is_detected, self._is_ground
            active = (det[g1] & gnd[g2]) | (det[g2] & gnd[g1])
            if active.any():
                active &= con.exclude[:ncon] == 0
                frames = con.frame
                for cid in np.flatnonzero(active):
                    mj.mj_contactForce(m, d, int(cid), self._wrench)
                    # contact frame rows are (normal, tangent1, tangent2) in world coords
                    world_force = frames[cid].reshape(3, 3).T @ self._wrench[:3]
                    a, b = int(g1[cid]), int(g2[cid])
                    if a in self._geom_to_row:
                        forces[self._geom_to_row[a]] -= world_force
                    if b in self._geom_to_row:
                        forces[self._geom_to_row[b]] += world_force
        return HybridControllerObservation(
            thorax_z=float(d.xpos[self._thorax_body_id, 2]),
            tarsus5_z=d.xpos[self._tarsus5_body_ids, 2],  # fancy indexing -> float64 copy
            stumbling_contact_forces=forces.reshape(self._n_legs, self._n_links, 3),
            fly_heading=d.xmat[self._thorax_body_id].reshape(3, 3)[:, 0].copy(),
        )


class _VectorizedSteps:
    """All six legs' preprogrammed-step splines evaluated in one shot.

    ``PreprogrammedSteps.get_joint_angles`` calls one scipy ``CubicSpline`` per leg
    (~10 us each, mostly Python overhead). This class stacks the splines' breakpoints
    and coefficients and repeats scipy's arithmetic *in the same floating-point
    order* (periodic wrap ``x0 + (x - x0) % (xN - x0)``, ``find_interval``, then
    ``evaluate_poly1``: ``res += c[k] * z; z *= s``), so the result is bitwise
    identical to FlyGym's path (checked in tests/test_sim.py).
    """

    def __init__(self, steps: PreprogrammedSteps, legs: tuple[str, ...]) -> None:
        splines = [steps._psi_funcs[leg] for leg in legs]
        x = splines[0].x
        for sp in splines:
            if not (np.array_equal(sp.x, x) and sp.c.shape[0] == 4 and sp.axis == 1
                    and sp.extrapolate == "periodic"):
                raise ValueError("unexpected spline layout in PreprogrammedSteps")
        self._x = np.ascontiguousarray(x)
        self._x0 = float(x[0])
        self._period = float(x[-1] - x[0])
        self._last_interval = len(x) - 2
        # _c[leg * n_intervals + interval] = (4, 7) coefficients; row 0 = cubic term
        c = np.stack([sp.c for sp in splines], axis=0)  # (leg, 4, interval, dof)
        self._n_int = c.shape[2]
        self._c = np.ascontiguousarray(c.transpose(0, 2, 1, 3).reshape(-1, 4, c.shape[3]))
        self._leg_offset = np.arange(len(legs)) * self._n_int
        self._neutral = np.stack([steps.neutral_pos[leg][:, 0] for leg in legs])  # (6, 7)

    def joint_angles(self, phases: np.ndarray, magnitudes: np.ndarray) -> np.ndarray:
        """(6, 7) joint angles, == ``get_joint_angles(leg, phase, magnitude)`` per leg."""
        xv = self._x0 + (phases - self._x0) % self._period
        # scipy find_interval: x[i] <= xv < x[i+1]; xv == x[-1] -> last interval
        # (xv >= x[0] after the periodic wrap, so idx >= 0 already)
        idx = np.searchsorted(self._x, xv, side="right") - 1
        np.minimum(idx, self._last_interval, out=idx)
        sd = (xv - self._x[idx])[:, None]
        c = self._c[self._leg_offset + idx]  # (6, 4, 7)
        z2 = sd * sd
        res = c[:, 3] + c[:, 2] * sd  # (0 + c3 * 1) == c3 exactly
        res += c[:, 1] * z2
        res += c[:, 0] * (z2 * sd)
        neutral = self._neutral
        return neutral + magnitudes[:, None] * (res - neutral)


class FastHybridController(HybridController):
    """FlyGym ``HybridController`` with cached bookkeeping, vectorised over legs.

    ``step`` reproduces ``HybridController.step`` bit for bit (same arithmetic, same
    order of floating-point operations per leg), just without the per-step
    ``JointDOF`` construction, the per-leg scipy spline calls and scalar numpy ops.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        out_order: list[JointDOF] = (
            self.output_dof_order
            if self.output_dof_order is not None
            else get_default_locomotion_dof_order()
        )
        # For each output DOF, which (leg, per-leg dof index) produces it.
        src = {}
        for leg_idx, leg in enumerate(self.legs):
            for dof_idx, spec in enumerate(self.preprogrammed_steps.dofs_per_leg):
                src[dof_spec_to_jointdof(leg, spec)] = (leg_idx, dof_idx)
        pairs = np.array([src[dof] for dof in out_order])
        self._out_leg_idx, self._out_dof_idx = pairs[:, 0], pairs[:, 1]
        # Correction vectors (right legs mirrored) and phase-gain interpolation knots
        # per leg (constants; same formulas as hybrid_controller._step_phase_gain).
        corr, knots, swing_end = [], [], []
        for leg in self.legs:
            v = _CORRECTION_VECTORS[leg[1]]
            if leg.startswith("r"):
                v = v * _RIGHT_LEG_CORRECTION_SIGN
            corr.append(v)
            s0, s1 = self.preprogrammed_steps.swing_period[leg]
            knots.append(
                [s0, np.mean([s0, s1]), s1 + self.swing_extension,
                 np.mean([s1, 2 * np.pi]), 2 * np.pi]
            )
            swing_end.append(s1 + self.swing_extension)
        self._corr_vecs = np.array(corr)  # (6, 7)
        self._gain_knots = np.array(knots)  # (6, 5)
        self._gain_vals = np.array([0.0, 0.8, 0.0, -0.1, 0.0])
        # np.interp is called per group of legs sharing the same knots (left/right
        # pairs): a hand-written vectorised interp is *not* bitwise identical because
        # numpy's C code is compiled with FMA contraction on arm64.
        groups: dict[bytes, list[int]] = {}
        for i, row in enumerate(self._gain_knots):
            groups.setdefault(row.tobytes(), []).append(i)
        self._gain_groups = [(np.array(ix), self._gain_knots[ix[0]]) for ix in groups.values()]
        self._swing_start = np.array(
            [self.preprogrammed_steps.swing_period[leg][0] for leg in self.legs]
        )
        self._swing_end = np.array(swing_end)
        self._steps_vec = _VectorizedSteps(self.preprogrammed_steps, self.legs)
        self._retract_up = self.retraction_rates[0] * self.timestep
        self._retract_down = self.retraction_rates[1] * self.timestep
        self._stumble_up = self.stumbling_rates[0] * self.timestep
        self._stumble_down = self.stumbling_rates[1] * self.timestep

    def _phase_gain(self, wrapped_phase: np.ndarray) -> np.ndarray:
        """Per-leg ``np.interp(phase, knots[leg], vals)`` (== ``_step_phase_gain``)."""
        gain = np.empty(len(wrapped_phase))
        for ix, knots in self._gain_groups:
            gain[ix] = np.interp(wrapped_phase[ix], knots, self._gain_vals)
        return gain

    def step(self, obs: HybridControllerObservation) -> LocomotionAction:
        # Mirrors HybridController.step(); only the bookkeeping differs.
        leg_to_correct_retraction = self._select_retraction_leg(obs)
        if leg_to_correct_retraction is not None:
            if (
                self.retraction_correction[leg_to_correct_retraction]
                > self.retraction_persistence_initiation_threshold
            ):
                self.retraction_persistence_counter[leg_to_correct_retraction] = 1
        self._update_persistence_counter()
        stumbling_mask = self._get_stumbling_mask(obs)
        self.cpg_network.step()

        # _update_retraction_correction / _update_stumbling_correction for all legs
        rc, sc = self.retraction_correction, self.stumbling_correction
        retract = self.retraction_persistence_counter > 0
        if leg_to_correct_retraction is not None:
            retract[leg_to_correct_retraction] = True
        rc[:] = np.where(retract, rc + self._retract_up, np.maximum(0.0, rc - self._retract_down))
        sc[:] = np.where(stumbling_mask, sc + self._stumble_up,
                         np.maximum(0.0, sc - self._stumble_down))
        retracting = rc > 0
        sc[retracting] = 0.0
        net_correction = np.where(retracting, rc, sc)

        phases = self.cpg_network.curr_phases
        angles = self._steps_vec.joint_angles(phases, self.cpg_network.curr_magnitudes)
        # == np.clip(net_correction, 0, max_correction) (np.clip has ~3 us overhead)
        net_correction = np.minimum(np.maximum(net_correction, 0.0), self.max_correction)
        wrapped = phases % (2 * np.pi)
        gain = self._phase_gain(wrapped)
        net_corrections = net_correction * gain
        leg_angles = angles + net_corrections[:, None] * self._corr_vecs
        if self.enable_adhesion:
            adhesion = ~((self._swing_start < wrapped) & (wrapped < self._swing_end))
        else:
            adhesion = np.zeros(len(self.legs), dtype=bool)

        joint_angles = leg_angles[self._out_leg_idx, self._out_dof_idx]
        self.last_info = {
            "net_corrections": net_corrections,
            "retraction_correction": rc.copy(),
            "stumbling_correction": sc.copy(),
            "stumbling_mask": stumbling_mask,
            "leg_to_correct_retraction": leg_to_correct_retraction,
        }
        return LocomotionAction(joint_angles=joint_angles, adhesion_onoff=adhesion)


class FastHybridTurningController(HybridTurningController, FastHybridController):
    """FlyGym ``HybridTurningController`` on top of ``FastHybridController``.

    ``step`` does what ``HybridTurningController.step`` does (left/right CPG
    amplitude = |signal|, frequency sign flipped for a negative signal) but writes
    into preallocated arrays, then calls ``FastHybridController.step``.
    """

    def step(self, descending_signal, obs: HybridControllerObservation) -> LocomotionAction:
        left, right = float(descending_signal[0]), float(descending_signal[1])
        net = self.cpg_network
        amps = np.empty(6)
        amps[:3] = abs(left)
        amps[3:] = abs(right)
        net.intrinsic_amps = amps
        freqs = self._base_intrinsic_freqs.copy()
        if left < 0:
            freqs[:3] *= -1
        if right < 0:
            freqs[3:] *= -1
        net.intrinsic_freqs = freqs
        return FastHybridController.step(self, obs)


class LocomotionController:
    """Thin adapter: FlyGym controller <-> our Simulation.

    Call ``reset()`` after the simulation was reset, then ``step_and_apply()`` once
    per physics step (FlyGym's tutorials run the controller at the physics rate),
    or once per ``control_timestep`` if decimated.
    """

    def __init__(
        self,
        sim: "FlyGymSimulation",
        fly_name: str,
        cfg: ControllerConfig,
        control_timestep: float | None = None,
    ):
        """``control_timestep``: interval between ``step_and_apply`` calls (default:
        the physics timestep, FlyGym's canonical loop). The CPG and the reflex
        rates integrate with this dt, so a decimated controller keeps the gait
        frequency."""
        self.cfg = cfg
        self._sim = sim
        self._fly_name = fly_name
        fly = sim.world.fly_lookup[fly_name]
        self.dof_order = fly.get_actuated_jointdofs_order("position")
        self.preprogrammed_steps = PreprogrammedSteps()
        self.control_timestep = float(control_timestep or sim.timestep)
        cpg = make_tripod_cpg_network(
            self.control_timestep, intrinsic_frequency=cfg.cpg_frequency, seed=cfg.seed
        )
        if cfg.kind == "hybrid":
            self.impl = FastHybridTurningController(
                timestep=self.control_timestep,
                cpg_network=cpg,
                preprogrammed_steps=self.preprogrammed_steps,
                output_dof_order=self.dof_order,
            )
            self._obs = FastObservationBuilder(sim, fly_name, self.impl.legs)
        elif cfg.kind == "cpg":
            self.impl = CPGController(
                cpg_network=cpg,
                preprogrammed_steps=self.preprogrammed_steps,
                output_dof_order=self.dof_order,
            )
            self._obs = None
        else:
            raise ValueError(f"Unknown controller kind {cfg.kind!r} (use 'hybrid' or 'cpg')")
        self.last_action: LocomotionAction | None = None
        # Precomputed ctrl indices == what apply_locomotion_action() /
        # Simulation.set_actuator_inputs() / set_leg_adhesion_states() write to.
        self._pos_ids = np.asarray(
            sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][fly_name], dtype=np.int64
        )
        adh = sim._intern_adhesionactuatorids_by_fly.get(fly_name)
        self._adh_ids = None if adh is None or len(adh) == 0 else np.asarray(adh, dtype=np.int64)
        self._target_heading = np.radians(cfg.target_heading_deg)
        self._ones2 = np.ones(2)
        # Optional ``fn(heading_hold_signal) -> signal`` applied in descending_signal()
        # (the app's --brain-steer installs the brain's descending drive here; see
        # perpetualfly/brain_link.py). None = heading hold only.
        self.signal_filter = None

    def initial_action(self) -> LocomotionAction:
        """Neutral standing pose with all tarsi adhering (used during warmup)."""
        return LocomotionAction(
            joint_angles=self.preprogrammed_steps.default_pose_by_dof_order(self.dof_order),
            adhesion_onoff=np.ones(6, dtype=bool),
        )

    def reset(self) -> None:
        if isinstance(self.impl, HybridController):
            self.impl.reset(seed=self.cfg.seed)
        else:
            self.impl.cpg_network.random_state = np.random.RandomState(self.cfg.seed)
            self.impl.cpg_network.reset()
        self.last_action = None

    def descending_signal(self) -> np.ndarray:
        """[left, right] CPG amplitude command. [1, 1] == plain straight walking.

        With heading hold enabled, a P-controller on the thorax yaw steers the fly
        back to ``target_heading_deg``: a larger left amplitude turns the fly right.
        """
        k = self.cfg.heading_gain
        if k <= 0:
            hold = self._ones2.copy()
        else:
            fwd = self._sim.mj_data.xmat[self._obs._thorax_body_id]  # row-major 3x3
            err = np.arctan2(fwd[3], fwd[0]) - self._target_heading  # column 0 = (m00, m10)
            err = (err + np.pi) % (2 * np.pi) - np.pi  # >0: fly points left of target
            lim = self.cfg.max_turn_signal
            delta = min(max(k * err, -lim), lim)  # == np.clip for scalars
            hold = np.array([1.0 + delta, 1.0 - delta])
        f = self.signal_filter
        return hold if f is None else f(hold)

    def compute_action(self) -> LocomotionAction:
        if self._obs is not None:
            return self.impl.step(self.descending_signal(), self._obs.build())
        return self.impl.step()

    def apply(self, action: LocomotionAction) -> None:
        # Same writes as flygym_demo's apply_locomotion_action(), minus the lookups.
        ctrl = self._sim.mj_data.ctrl
        ctrl[self._pos_ids] = action.joint_angles
        if action.adhesion_onoff is not None:
            if self._adh_ids is None:
                apply_locomotion_action(self._sim, self._fly_name, action)  # raises like FlyGym
            else:
                ctrl[self._adh_ids] = action.adhesion_onoff
        self.last_action = action

    def step_and_apply(self) -> LocomotionAction:
        action = self.compute_action()
        self.apply(action)
        return action
