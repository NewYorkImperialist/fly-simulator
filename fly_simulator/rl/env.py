"""Residual-RL Gymnasium environment for Fly Simulator (spec phase 10).

``FlySimulatorEnv`` wraps one ``fly_simulator.simulation.Simulation`` on
``ProceduralTerrain`` with the thorax-shove ``Perturbation`` + ``AutoPerturber`` and
a ``FallDetector``. No logging, no rendering unless ``render_mode="rgb_array"``.

Control scheme (see docs/RL.md):

* FlyGym's hybrid controller (the *baseline*) still runs at every physics step
  (1e-4 s), exactly as in the app.
* The policy acts every ``control_every_steps`` physics steps (default 50 -> 200 Hz)
  and outputs a residual ``a`` in [-1, 1]^42 that is held (zero-order hold) until the
  next policy step:  ``ctrl[joint targets] = baseline_targets + action_scale * a``.
  It goes through the same actuator write as the baseline controller
  (``LocomotionController.apply``). ``a == 0`` reproduces the baseline bit for bit
  (the residual addition is skipped entirely when the residual is zero).
* Optional residual on the six adhesion actuators (off by default).

Observations are proprioceptive only: no hit information, no absolute x/y.
Hit data is reported in ``info`` only (for logging / evaluation).

Rewards are *rates* (per simulated second) multiplied by the policy interval, so
weights don't depend on ``control_every_steps``. Per-term contributions are returned
in ``info["reward_terms"]``.

gymnasium is imported here only; the core package never imports ``fly_simulator.rl``.
"""

from __future__ import annotations

import copy
import logging
import math
import sys
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from flygym_demo.complex_terrain import LocomotionAction

from fly_simulator.config import AppConfig
from fly_simulator.interaction.perturbation import (
    AutoPerturbConfig,
    AutoPerturber,
    HitEvent,
    Perturbation,
)
from fly_simulator.metrics.contacts import ContactClassifier
from fly_simulator.metrics.falls import FallDetector, FallEvent, FallState
from fly_simulator.simulation import Simulation, SimulationInstabilityError
from fly_simulator.terrain import ProceduralTerrain, ProceduralTerrainConfig, TerrainGenerator

log = logging.getLogger(__name__)

N_JOINTS = 42  # position-actuated leg DoFs (7 per leg)
N_LEGS = 6

# Observation normalisation scales, from 3 s of baseline walking on normal terrain
# (std of each signal): angular velocity 6-7.5 rad/s per axis (peaks ~15), body-frame
# linear velocity 7-13 mm/s std around a 14 mm/s mean, joint deviation from the
# neutral pose 0.06-0.37 rad std, joint velocity 7-36 rad/s std, thorax height
# 1.07-1.19 mm. After scaling, walking signals are O(1); a fall or hit gives O(10)
# values, and everything is clipped to +-obs_clip.
ANGVEL_SCALE = 10.0  # rad/s
LINVEL_SCALE = 20.0  # mm/s
JOINT_POS_SCALE = 0.3  # rad (offset from the neutral pose)
JOINT_VEL_SCALE = 25.0  # rad/s
HEIGHT_REF = 1.1  # mm, walking thorax height above the local ground
HEIGHT_SCALE = 0.2  # mm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CurriculumStage:
    """Terrain difficulty + automatic thorax hits for one curriculum stage."""

    name: str
    terrain: str = "flat"  # ProceduralTerrain difficulty: flat | easy | normal | hard | chaos
    hits: bool = False
    hit_levels: tuple[int, ...] = (1,)  # Perturbation strength levels (1 gentle .. 4 absurd)
    hit_level_weights: tuple[float, ...] = (1.0,)
    hit_interval_s: tuple[float, float] = (2.0, 5.0)  # uniform, sim seconds
    first_hit_after_s: float = 1.0
    direction_weights: dict[str, float] | None = None  # None = AutoPerturbConfig default
    magnitude_bw_range: tuple[float, float] | None = None  # override level magnitudes

    def auto_perturb_config(self, seed: int) -> AutoPerturbConfig:
        kw: dict[str, Any] = dict(
            enabled=self.hits, min_interval_s=self.hit_interval_s[0],
            max_interval_s=self.hit_interval_s[1], first_hit_after_s=self.first_hit_after_s,
            levels=tuple(self.hit_levels), level_weights=tuple(self.hit_level_weights),
            magnitude_bw_range=self.magnitude_bw_range, seed=seed,
        )
        if self.direction_weights is not None:
            kw["direction_weights"] = dict(self.direction_weights)
        return AutoPerturbConfig(**kw)


def default_curriculum() -> list[CurriculumStage]:
    # Hit levels: docs/PERTURBATION_CALIBRATION.md (gentle never tips the baseline,
    # medium ~17 %, hard ~2/3).
    return [
        CurriculumStage("flat"),
        CurriculumStage("flat_gentle", "flat", True, (1,), (1.0,), (2.0, 5.0)),
        CurriculumStage("flat_medium", "flat", True, (1, 2), (0.5, 0.5), (2.0, 4.0)),
        CurriculumStage("easy_terrain", "easy", True, (1, 2), (0.6, 0.4), (2.0, 5.0)),
        CurriculumStage("normal", "normal", True, (1, 2), (0.6, 0.4), (2.0, 5.0)),
        CurriculumStage("aggressive", "hard", True, (1, 2, 3), (0.3, 0.4, 0.3), (1.5, 4.0)),
    ]


@dataclass
class RewardConfig:
    """Reward weights. Unless noted, each term is a rate per simulated second (the
    per-step reward is rate * policy dt)."""

    target_speed: float = 14.0  # mm/s, baseline walking speed
    speed_sigma: float = 5.0  # mm/s, width of the Gaussian velocity reward
    velocity_window_s: float = 0.1  # speed = displacement over this window (~1 stride)
    velocity: float = 2.0  # * exp(-((v_fwd - target) / sigma)^2)
    upright: float = 0.5  # * max(0, world-z component of the thorax up axis)
    alive: float = 0.5  # while not FALLEN
    energy: float = 0.5  # * mean(a^2)
    action_rate: float = 0.5  # * mean((a - a_prev)^2)
    instability: float = 0.2  # * (wx^2 + wy^2) / ANGVEL_SCALE^2 (body roll/pitch rates)
    fallen: float = 3.0  # while FALLEN
    backward: float = 2.0  # * max(0, -v_fwd) / target_speed
    # one-shot (not multiplied by dt)
    recovery_bonus: float = 5.0  # FallDetector "recovered" event
    fall_termination: float = 10.0  # episode terminated because it stayed down
    instability_termination: float = 50.0  # SimulationInstabilityError


@dataclass
class EnvConfig:
    app: AppConfig = field(default_factory=AppConfig)
    # Physics steps (1e-4 s) per policy step: 50 -> 200 Hz policy (5 ms), ~17 policy
    # steps per ~83 ms tripod cycle. See docs/RL.md.
    control_every_steps: int = 50
    action_scale: float = 0.15  # rad of joint-target residual at |a| = 1
    adhesion_residual: bool = False  # 6 extra actions added to the adhesion on/off (0..1)
    adhesion_scale: float = 1.0
    baseline: str = "hybrid"  # "hybrid" (FlyGym hybrid controller) | "stand" (neutral pose)
    max_episode_s: float = 20.0  # truncation (sim seconds after the reset warmup)
    terminate_on_fall: bool = True  # training semantics; the app never terminates
    fall_terminate_after_s: float = 2.0  # continuously FALLEN this long -> terminated
    obs_height: bool = True  # thorax height above local ground (semi-privileged)
    obs_clip: float = 10.0
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: list[CurriculumStage] = field(default_factory=default_curriculum)
    stage: int = 0
    # None = a new terrain layout seed per episode (drawn from the env RNG).
    terrain_seed: int | None = None
    # None = new CPG initial phases per episode; an int fixes them.
    controller_seed: int | None = None
    render_width: int = 480
    render_height: int = 320


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class FlySimulatorEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, cfg: EnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        # Private copy: the env mutates e.g. app.controller.seed per episode.
        self.cfg = cfg = copy.deepcopy(cfg) if cfg is not None else EnvConfig()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be None or 'rgb_array', got {render_mode!r}")
        if cfg.baseline not in ("hybrid", "stand"):
            raise ValueError(f"baseline must be 'hybrid' or 'stand', got {cfg.baseline!r}")
        self.render_mode = render_mode
        self.stage_index = self._stage_index(cfg.stage)
        stage = self.stage

        # --- world: procedural terrain (same pool size for every difficulty, so the
        # difficulty / seed can change on reset without rebuilding the model).
        app = cfg.app
        self.terrain = ProceduralTerrain(ProceduralTerrainConfig(
            difficulty=stage.terrain, seed=app.terrain.seed,
            ground_half_size=app.terrain.ground_half_size,
            checker_size_mm=app.terrain.checker_size_mm,
        ))
        self.sim = sim = Simulation(app, world_factory=self.terrain.build_world)
        self.terrain.attach(sim)
        self.perturbation = Perturbation(sim, replace(app.perturbation))
        self.perturbation.listeners.append(self._on_hit)
        self.auto: AutoPerturber | None = None
        self.detector = FallDetector(sim, app.falls, ground_height_fn=self.terrain.ground_height_at)
        self.detector.add_listener(self._on_fall_event)
        self.contacts = ContactClassifier(sim.model, sim.fly_name)

        # --- residual action path: wrap the controller's per-physics-step call.
        ctrl = sim.controller
        self._ctrl = ctrl
        self._residual = np.zeros(N_JOINTS)
        self._residual_on = False
        self._adh_residual = np.zeros(N_LEGS)
        self._adh_on = False
        self._stand_action = ctrl.initial_action()
        self.last_baseline_targets = np.zeros(N_JOINTS)
        ctrl.step_and_apply = self._controller_step  # instance attr shadows the method

        m = sim.model
        joints = m.actuator_trnid[ctrl._pos_ids, 0]
        self._jnt_qpos = np.asarray(m.jnt_qposadr[joints], dtype=np.int64)
        self._jnt_qvel = np.asarray(m.jnt_dofadr[joints], dtype=np.int64)
        self._neutral_pose = np.asarray(self._stand_action.joint_angles, dtype=float).copy()
        self._target_heading = math.radians(app.controller.target_heading_deg)
        self._fwd_dir = np.array([math.cos(self._target_heading), math.sin(self._target_heading)])

        # --- spaces
        self.n_actions = N_JOINTS + (N_LEGS if cfg.adhesion_residual else 0)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_actions,), dtype=np.float32)
        self.obs_layout = self._make_obs_layout()
        n_obs = sum(n for _, n in self.obs_layout)
        c = float(cfg.obs_clip)
        self.observation_space = spaces.Box(-c, c, shape=(n_obs,), dtype=np.float32)

        self.dt = cfg.control_every_steps * sim.timestep  # policy interval (s)
        self._max_policy_steps = int(round(cfg.max_episode_s / self.dt))
        self._renderer = None
        self._needs_reset = True
        self._episode_seeds: dict[str, int] = {}
        self._init_episode_state()

    # ------------------------------------------------------------------ layout
    def _make_obs_layout(self) -> list[tuple[str, int]]:
        layout = [
            ("gravity_body", 3),  # world -z expressed in the thorax frame
            ("heading_err_sincos", 2),  # yaw relative to the target heading
            ("angvel_body", 3),
            ("linvel_body", 3),
        ]
        if self.cfg.obs_height:
            layout.append(("height", 1))
        layout += [
            ("joint_pos", N_JOINTS),
            ("joint_vel", N_JOINTS),
            ("leg_contact", N_LEGS),
            ("cpg_phase_sincos", 2 * N_LEGS),
            ("prev_action", self.n_actions),
        ]
        return layout

    def obs_slices(self) -> dict[str, slice]:
        out, i = {}, 0
        for name, n in self.obs_layout:
            out[name] = slice(i, i + n)
            i += n
        return out

    # ------------------------------------------------------------------ stages
    @property
    def stage(self) -> CurriculumStage:
        return self.cfg.curriculum[self.stage_index]

    def _stage_index(self, stage: int | str) -> int:
        cur = self.cfg.curriculum
        if isinstance(stage, str):
            names = [s.name for s in cur]
            if stage not in names:
                raise ValueError(f"unknown stage {stage!r}; stages: {names}")
            return names.index(stage)
        i = int(stage)
        if not 0 <= i < len(cur):
            raise ValueError(f"stage index {i} out of range 0..{len(cur) - 1}")
        return i

    def set_stage(self, stage: int | str) -> int:
        """Select a curriculum stage (index or name); takes effect at the next reset."""
        self.stage_index = self._stage_index(stage)
        return self.stage_index

    def get_stage(self) -> int:
        return self.stage_index

    # ------------------------------------------------------------------ hooks
    def _controller_step(self) -> LocomotionAction:
        """Replaces ``LocomotionController.step_and_apply`` (called by Simulation.step
        every control step): baseline action + held residual -> same actuator write."""
        ctrl = self._ctrl
        if self.cfg.baseline == "stand":
            base = self._stand_action
        else:
            base = ctrl.compute_action()
        self.last_baseline_targets = base.joint_angles
        action = base
        if self._residual_on or self._adh_on:
            joints = base.joint_angles + self._residual if self._residual_on else base.joint_angles
            adh = base.adhesion_onoff
            if self._adh_on and adh is not None:
                adh = np.clip(adh.astype(float) + self._adh_residual, 0.0, 1.0)
            action = LocomotionAction(joint_angles=joints, adhesion_onoff=adh)
        ctrl.apply(action)
        return action

    def _on_hit(self, ev: HitEvent) -> None:
        self._step_hits.append(ev)
        self._episode_hits += 1

    def _on_fall_event(self, ev: FallEvent) -> None:
        self._step_fall_events.append(ev)
        if ev.kind in ("fall", "relapse"):
            self._down_since = ev.time
            if ev.kind == "fall":
                self._episode_falls += 1
        elif ev.kind in ("recovering", "recovered", "reset"):
            self._down_since = None
        if ev.kind == "recovered":
            self._episode_recoveries += 1

    # ------------------------------------------------------------------ episode
    def _init_episode_state(self) -> None:
        self._step_hits: list[HitEvent] = []
        self._step_fall_events: list[FallEvent] = []
        self._episode_hits = 0
        self._episode_falls = 0
        self._episode_recoveries = 0
        self._down_since: float | None = None
        self._policy_steps = 0
        self._prev_action = np.zeros(self.n_actions)
        self._residual[:] = 0.0
        self._adh_residual[:] = 0.0
        self._residual_on = self._adh_on = False
        self._pos_hist: deque = deque()
        self._t0 = 0.0
        self._start_xy = np.zeros(2)
        self._path = 0.0
        self._last_xy = np.zeros(2)

    def _configure_episode(self, rng: np.random.Generator) -> None:
        cfg, stage = self.cfg, self.stage
        draw = lambda: int(rng.integers(0, 2**31 - 1))  # noqa: E731
        seeds = dict(
            terrain=cfg.terrain_seed if cfg.terrain_seed is not None else draw(),
            controller=cfg.controller_seed if cfg.controller_seed is not None else draw(),
            perturbation=draw(),
            auto_perturb=draw(),
        )
        self._episode_seeds = seeds
        # Terrain: new generator (seed / difficulty); ProceduralTerrain's pre-reset
        # hook re-lays out the chunks around x = 0 with it during sim.reset().
        t = self.terrain
        t.cfg.seed, t.cfg.difficulty, t.cfg.weights = seeds["terrain"], stage.terrain, None
        t.generator = TerrainGenerator(seeds["terrain"], stage.terrain, cfg=t.cfg)
        # Controller: LocomotionController.reset() reseeds the CPG phases from here.
        self.sim.cfg.controller.seed = seeds["controller"]
        # Hits: fresh RNGs; the auto perturber is rebuilt for the stage's settings.
        self.perturbation.rng = np.random.default_rng(seeds["perturbation"])
        if self.auto is not None:
            self.auto.detach()
        self.auto = AutoPerturber(self.sim, self.perturbation,
                                  stage.auto_perturb_config(seeds["auto_perturb"]))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "stage" in options:
            self.set_stage(options["stage"])
        self._configure_episode(self.np_random)
        self._init_episode_state()
        self.sim.reset()  # terrain relayout -> keyframe -> controller reset -> warmup
        self._t0 = self.sim.time
        xy = self.sim.data.xpos[self.sim.thorax_body_id, :2].copy()
        self._start_xy, self._last_xy = xy.copy(), xy.copy()
        self._pos_hist.append((self._t0, xy))
        self._step_hits.clear()
        self._step_fall_events.clear()
        self._needs_reset = False
        if self._renderer is not None:
            self._renderer.camera.reset()
        obs = self._observation()
        info = {"stage": self.stage.name, "stage_index": self.stage_index,
                "seeds": dict(self._episode_seeds)}
        return obs, info

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("call reset() before step() (episode ended or never started)")
        cfg, rc = self.cfg, self.cfg.reward
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(self.n_actions), -1.0, 1.0)
        a = np.nan_to_num(a, nan=0.0)
        self._residual[:] = cfg.action_scale * a[:N_JOINTS]
        self._residual_on = bool(np.any(self._residual))
        if cfg.adhesion_residual:
            self._adh_residual[:] = cfg.adhesion_scale * a[N_JOINTS:]
            self._adh_on = bool(np.any(self._adh_residual))
        self._step_hits.clear()
        self._step_fall_events.clear()

        sim = self.sim
        try:
            sim.step(cfg.control_every_steps)
        except SimulationInstabilityError as e:
            return self._instability(e, a)
        self._policy_steps += 1

        # ---- kinematics
        tid = sim.thorax_body_id
        R = sim.data.xmat[tid].reshape(3, 3)
        t = sim.time
        xy = sim.data.xpos[tid, :2].copy()
        self._path += float(np.hypot(*(xy - self._last_xy)))
        self._last_xy = xy
        hist = self._pos_hist
        hist.append((t, xy))
        while len(hist) > 2 and t - hist[1][0] >= rc.velocity_window_s - 1e-9:
            hist.popleft()
        t_old, xy_old = hist[0]
        v_fwd = float((xy - xy_old) @ self._fwd_dir / (t - t_old)) if t > t_old else 0.0
        angvel = sim.data.qvel[sim._free_qvel + 3: sim._free_qvel + 6]
        state = self.detector.state

        # ---- reward (rates per second * dt, plus one-shot terms)
        dt = self.dt
        fallen = state == FallState.FALLEN
        terms = {
            "velocity": rc.velocity * math.exp(-((v_fwd - rc.target_speed) / rc.speed_sigma) ** 2),
            "upright": rc.upright * max(0.0, float(R[2, 2])),
            "alive": 0.0 if fallen else rc.alive,
            "energy": -rc.energy * float(np.mean(a[:N_JOINTS] ** 2)),
            "action_rate": -rc.action_rate * float(np.mean((a - self._prev_action) ** 2)),
            "instability": -rc.instability * float(angvel[0] ** 2 + angvel[1] ** 2)
            / ANGVEL_SCALE ** 2,
            "fallen": -rc.fallen if fallen else 0.0,
            "backward": -rc.backward * max(0.0, -v_fwd) / rc.target_speed,
        }
        terms = {k: v * dt for k, v in terms.items()}
        n_recovered = sum(ev.kind == "recovered" for ev in self._step_fall_events)
        terms["recovery"] = rc.recovery_bonus * n_recovered

        # ---- termination / truncation
        down_for = None if self._down_since is None else t - self._down_since
        terminated = bool(cfg.terminate_on_fall and down_for is not None
                          and down_for > cfg.fall_terminate_after_s)
        terms["termination"] = -rc.fall_termination if terminated else 0.0
        truncated = (not terminated) and self._policy_steps >= self._max_policy_steps
        reward = float(sum(terms.values()))

        self._prev_action = a
        obs = self._observation()
        info = {
            "reward_terms": terms,
            "fall_state": state.value,
            "down_for_s": down_for,
            "forward_speed": v_fwd,
            "sim_time": t - self._t0,
            # info only, never in the observation:
            "hits": [h.to_dict() for h in self._step_hits],
            "hit_active": self.perturbation.is_active,
            "fall_events": [ev.kind for ev in self._step_fall_events],
            "terrain_type": self.terrain.terrain_type_at(float(xy[0])),
            "instability": False,
        }
        if terminated or truncated:
            self._needs_reset = True
            info["episode_stats"] = self.episode_stats()
        return obs, reward, terminated, truncated, info

    def _instability(self, err: SimulationInstabilityError, a: np.ndarray):
        # Loud: logged at ERROR level *and* printed to stderr, never silently swallowed.
        msg = f"[FlySimulatorEnv] SimulationInstabilityError (stage {self.stage.name}, " \
              f"seeds {self._episode_seeds}, policy step {self._policy_steps}):\n{err}"
        log.error(msg)
        print(msg, file=sys.stderr, flush=True)
        self._needs_reset = True
        self._prev_action = a
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        rc = self.cfg.reward
        terms = {"termination": -rc.instability_termination}
        info = {"reward_terms": terms, "instability": True, "instability_msg": str(err),
                "hits": [h.to_dict() for h in self._step_hits], "fall_events": [],
                "episode_stats": self.episode_stats()}
        return obs, -rc.instability_termination, True, False, info

    def episode_stats(self) -> dict[str, Any]:
        t = self.sim.time - self._t0
        fwd = float((self._last_xy - self._start_xy) @ self._fwd_dir)
        return {
            "stage": self.stage.name,
            "sim_time_s": t,
            "path_mm": self._path,
            "forward_mm": fwd,
            "avg_forward_speed": fwd / t if t > 0 else 0.0,
            "n_hits": self._episode_hits,
            "n_falls": self._episode_falls,
            "n_recoveries": self._episode_recoveries,
            "fallen_at_end": self.detector.state == FallState.FALLEN,
        }

    # ------------------------------------------------------------------ obs
    def _observation(self) -> np.ndarray:
        sim, cfg = self.sim, self.cfg
        d = sim.data
        tid = sim.thorax_body_id
        R = d.xmat[tid].reshape(3, 3)
        a = sim._free_qvel
        parts = [
            -R[2, :],  # R^T @ (0, 0, -1): gravity direction in the thorax frame
        ]
        err = math.atan2(R[1, 0], R[0, 0]) - self._target_heading
        parts.append(np.array([math.sin(err), math.cos(err)]))
        parts.append(d.qvel[a + 3: a + 6] / ANGVEL_SCALE)  # free joint: body frame
        parts.append(R.T @ d.qvel[a: a + 3] / LINVEL_SCALE)  # world -> body frame
        if cfg.obs_height:
            x, y, z = d.xpos[tid]
            h = z - self.terrain.ground_height_at(float(x), float(y))
            parts.append(np.array([(h - HEIGHT_REF) / HEIGHT_SCALE]))
        parts.append((d.qpos[self._jnt_qpos] - self._neutral_pose) / JOINT_POS_SCALE)
        parts.append(d.qvel[self._jnt_qvel] / JOINT_VEL_SCALE)
        parts.append(self.contacts.summarize(d).leg_contact.astype(float))
        phases = self._ctrl.impl.cpg_network.curr_phases
        parts.append(np.sin(phases))
        parts.append(np.cos(phases))
        parts.append(self._prev_action)
        obs = np.concatenate(parts)
        c = cfg.obs_clip
        np.clip(obs, -c, c, out=obs)
        return np.nan_to_num(obs, nan=0.0).astype(np.float32)

    # ------------------------------------------------------------------ render
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            from fly_simulator.rendering import FrameRenderer

            app = self.cfg.app
            rcfg = replace(app.render, width=self.cfg.render_width, height=self.cfg.render_height)
            self._renderer = FrameRenderer(self.sim.model, rcfg, app.camera)
        sim = self.sim
        x, y, _ = sim.thorax_position()
        return self._renderer.render(
            sim.data, sim.time, sim.thorax_position(), sim.heading(),
            ground_z=self.terrain.ground_height_at(float(x), float(y)), tilt_deg=sim.tilt_deg(),
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        sim = getattr(self, "sim", None)
        if sim is not None:
            sim.close()
            self.sim = None
