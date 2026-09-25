"""Configuration dataclasses.

All tunable numbers live here instead of being scattered as magic constants.
Units follow FlyGym/MuJoCo model units: length mm, time s, mass g,
force uN (= g*mm/s^2). See docs/API_NOTES.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from perpetualfly.brain_link import BrainLinkConfig
from perpetualfly.interaction.perturbation import AutoPerturbConfig, PerturbationConfig
from perpetualfly.interaction.whip import WhipConfig
from perpetualfly.metrics.falls import FallDetectorConfig
from perpetualfly.metrics.run_logger import LoggingConfig


@dataclass
class FlyConfig:
    name: str = "nmf"
    # Spawn height of the fly's attachment frame (mm). The thorax sits ~1.3 mm above
    # this frame in the neutral pose; FlyGym's flat-ground tutorial uses 0.8.
    spawn_height: float = 0.8
    # Which body segments collide with the ground. FlyGym's walking tutorials use
    # "tibia_tarsus_only"; we default to legs+thorax+abdomen+head so that a fallen
    # fly lies *on* the ground instead of sinking its body through it (needed for
    # later fall detection). Verified that walking performance is unaffected.
    ground_contact: str = "legs_thorax_abdomen_head"
    colorize: bool = True
    # Adhesion (sticky tarsi) is part of FlyGym's standard locomotion setup.
    adhesion: bool = True


@dataclass
class TerrainConfig:
    # Used by the app (perpetualfly.app): procedural terrain difficulty preset
    # flat | easy | normal | hard | chaos (perpetualfly.terrain.DIFFICULTY_PRESETS).
    # "flat" is still the procedural terrain (all chunks flat), so obstacles can be
    # spawned with the keys. Simulation(cfg) alone (no world_factory) always builds
    # plain flat ground and ignores these three fields.
    difficulty: str = "flat"
    seed: int = 42
    weights: dict[str, float] | None = None  # optional override of the preset table
    ground_half_size: float = 1000.0  # mm, rendered extent of the flat plane
    checker_size_mm: float = 2.0  # rendered checker square size (visual only)


@dataclass
class ControllerConfig:
    # "hybrid" = FlyGym HybridController (CPG + stumbling/retraction reflexes).
    # "cpg"    = FlyGym CPGController (pure tripod CPG, no reflexes).
    kind: str = "hybrid"
    seed: int = 0
    # Tripod CPG intrinsic stepping frequency (Hz); FlyGym default is 12.
    cpg_frequency: float = 12.0
    # Heading hold (hybrid only). FlyGym's controller settles into a constant ~10 deg
    # heading offset to the left; since the world/terrain is laid out along +x, a
    # P-controller feeds FlyGym's HybridTurningController a [left, right] descending
    # signal = [1 + k*err, 1 - k*err]. heading_gain=0 disables it (signal [1, 1],
    # identical to the plain HybridController).
    heading_gain: float = 1.5  # per rad of heading error
    target_heading_deg: float = 0.0
    max_turn_signal: float = 0.4


@dataclass
class SimConfig:
    # None = keep FlyGym's physics timestep (1e-4 s from mujoco_globals.yaml).
    # The spec requires keeping FlyGym's timestep; only override for experiments.
    timestep: float | None = None
    # Settle time with the neutral pose + adhesion before the controller starts (s).
    warmup_s: float = 0.05
    # Stability checks (see Simulation.check_stability).
    check_every_steps: int = 10
    max_abs_qvel: float = 1e6  # fly DoFs, rad/s or mm/s; anything above is a blow-up
    # DoFs that don't belong to the fly (bodies added by world extensions, e.g. a
    # light whip tip may legitimately move very fast): NaN checks + this limit.
    max_abs_qvel_other: float = 1e9
    min_thorax_z: float = -50.0  # mm; below this the fly fell through the world
    # Run the locomotion controller every N physics steps and hold ctrl in between
    # (the CPG / reflex integrators then use dt = N * timestep). 1 = FlyGym's
    # canonical loop (controller at the physics rate). >1 is faster but changes
    # the trajectory (see docs/API_NOTES.md, Performance): keep 1 unless you
    # accept that.
    control_every_steps: int = 1
    # FlyGym's globals enable mjENBL_ENERGY (potential/kinetic energy computed in
    # every mj_step, never read by PerpetualFly). Turning it off does not change
    # the dynamics (verified bit-identical trajectory) and saves ~1-2 %.
    compute_energy: bool = False


@dataclass
class CameraConfig:
    mode: str = "follow"  # follow | side | top
    distance: float = 9.0  # mm from the look-at point
    elevation: float = -22.0  # deg, negative looks down
    # Azimuth offset relative to the fly's (smoothed) heading, degrees.
    # MuJoCo places a free camera at lookat - distance * forward(azimuth, elevation),
    # so -45 puts the camera behind and to the fly's left: rear three-quarter view
    # (verified from rendered frames; -135 would be front-left).
    follow_azimuth_offset: float = -45.0
    # Exponential smoothing time constants in *simulated* seconds, so the camera
    # motion is independent of the render rate.
    position_tau_s: float = 0.15
    heading_tau_s: float = 0.6
    # Catch-up after big shoves: the position time constant shrinks as
    # position_tau_s / (1 + (lag / catchup_distance)^2), where lag is the distance
    # between the smoothed look-at point and the fly, and the look-at point is
    # never allowed to trail the fly by more than max_lag mm.
    catchup_distance: float = 1.5  # mm
    max_lag: float = 3.0  # mm
    # Zoom out when the fly is high above the ground (launched by a hit) so the
    # ground stays in view: distance += zoom_per_mm * (height - zoom_above).
    zoom_above: float = 2.0  # mm of thorax height above the local ground
    zoom_per_mm: float = 1.5
    max_distance: float = 30.0
    # Don't follow the heading while the fly lies on its side / back (its forward
    # axis then points anywhere and the view would spin).
    freeze_heading_tilt_deg: float = 50.0
    fovy: float = 45.0


@dataclass
class RenderConfig:
    width: int = 960
    height: int = 640
    # Render one frame every N physics steps (1e-4 s each). 150 -> 66.7 frames per
    # simulated second. Physics is always stepped without rendering in between.
    render_every_steps: int = 150
    # Cap on sim-time / wall-time. The fly sim usually runs slower than real time,
    # in which case the cap never kicks in.
    max_realtime_factor: float = 1.0
    # Live window only: display frames at most this often (wall clock). Physics
    # runs in chunks of <= render_every_steps between frames, so display cost no
    # longer scales with the physics rate. Recording (--record) still grabs one
    # frame every render_every_steps simulated steps.
    target_fps: float = 30.0
    # Live window: step physics in a background thread (perpetualfly/physics_thread.py)
    # so rendering + the OpenCV event pump don't steal time from physics. Same
    # physics results; turn off to debug with the plain single-threaded loop.
    threaded_physics: bool = True
    # Physics steps the worker runs per lock acquisition (50 = 5 ms sim, ~7 ms
    # wall): bounds how long the display waits for the lock.
    thread_chunk_steps: int = 50
    window_title: str = "PerpetualFly"
    show_hud: bool = True


@dataclass
class StatsConfig:
    print_interval_s: float = 1.0  # simulated seconds between terminal prints
    speed_window_s: float = 0.5  # window for "current speed"


@dataclass
class SessionConfig:
    """App behaviour around falls (perpetualfly.app)."""

    # Print "press X to reset" once the fly has been down (fallen, not yet
    # recovered) for this many simulated seconds. 0 = never.
    fall_hint_after_s: float = 3.0
    # Unattended runs: explicitly reset the sim after the fly has been down this
    # long (None = never). Counted in metrics as a reset and logged as an
    # "auto_reset" event; nothing is hidden.
    auto_reset_after_s: float | None = None
    # Distance ahead of the thorax for key-spawned obstacles (None = terrain default).
    spawn_distance: float | None = None
    # What SPACE / arrows / U and the auto perturber do: "whip" = crack the physical
    # whip (AppConfig.whip; the fly moves only through contact forces), "shove" =
    # constant external force on the thorax (AppConfig.perturbation). H toggles.
    hit_mode: str = "whip"


def _app_auto_perturb() -> AutoPerturbConfig:
    # The auto perturber is always installed (so A can toggle it) but starts off
    # unless --auto-perturb is given.
    return AutoPerturbConfig(enabled=False)


@dataclass
class AppConfig:
    fly: FlyConfig = field(default_factory=FlyConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    whip: WhipConfig = field(default_factory=WhipConfig)
    auto_perturb: AutoPerturbConfig = field(default_factory=_app_auto_perturb)
    falls: FallDetectorConfig = field(default_factory=FallDetectorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    # Connectome brain (--brain, --brain-steer; perpetualfly/brain_link.py). Off by default.
    brain: BrainLinkConfig = field(default_factory=BrainLinkConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        return _from_dict(cls, data)

    @classmethod
    def load_json(cls, path: str | Path) -> "AppConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _from_dict(cls: type, data: dict[str, Any]):
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        default = f.default_factory() if callable(f.default_factory) else None  # type: ignore[misc]
        if is_dataclass(default) and isinstance(value, dict):
            sub = type(default)
            # Configs with their own from_dict (tuples, nested lists) use it.
            value = sub.from_dict(value) if hasattr(sub, "from_dict") else _from_dict(sub, value)
        elif isinstance(f.default, tuple) and isinstance(value, list):
            value = tuple(value)
        kwargs[f.name] = value
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**kwargs)
