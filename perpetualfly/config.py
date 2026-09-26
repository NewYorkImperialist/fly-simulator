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
from perpetualfly.senses.odor import OdorConfig  # --odor-zones (docs/FEAR_LEARNING.md)
from perpetualfly.senses.taste import TasteConfig  # --taste-patches (docs/TASTE.md)
from perpetualfly.stress import StressConfig


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
    # Extra wing + proboscis joints (perpetualfly.actions.make_action_fly; needed by
    # the W / N actions). Off in the library so FlyGym's model and the reference
    # trajectories stay bit-identical; the interactive CLI turns it on
    # (--full-body / --no-full-body). Walking is unchanged within ~0.1 %.
    extra_joints: bool = False


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
    # Floor reflections (MuJoCo's mjRND_REFLECTION scene flag; the checker floor has
    # reflectance 0.2). Off (--no-reflections) saves ~5 ms per 960x640 frame on an
    # M1 (draw 10.6-11.2 -> 5.5-6.4 ms, measured); visual only.
    reflections: bool = True
    # Live window: image upscale for imshow (None = auto: the display's backing
    # scale on a Retina Mac, env PERPETUALFLY_FLY_SCALE overrides; see
    # perpetualfly/display.py). The HUD is drawn after the upscale, so it stays sharp.
    display_scale: float | None = None
    # M key (live MP4 of the fly view, saved in the run dir): the video has this
    # frame rate and plays at *simulated* real time (like --record). A frame is
    # written whenever the run time has advanced by 1/record_fps: in the live
    # window the latest displayed frame is used (frames are dropped when the window
    # shows more than record_fps per simulated second, i.e. at RTF < 1), so
    # recording costs the physics thread nothing; headless, a frame is rendered
    # only when one is due.
    record_fps: float = 30.0


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
    # Automatic hits wait while the fly is FALLEN / RECOVERING and resume once it
    # has been UPRIGHT or DESTABILIZED for auto_perturb_resume_after_s (sim s).
    # Hits that would have fired meanwhile are counted as skipped (summary.json).
    auto_perturb_pause_when_down: bool = True
    auto_perturb_resume_after_s: float = 1.0


@dataclass
class SwatterAppConfig:
    """The flyswatter (--swatter; perpetualfly/interaction/swatter.py, docs/SWATTER.md).
    V swats from behind, Shift+V from a random side; 1-4 set ``level`` too."""

    enabled: bool = False
    level: int = 2  # 1 lazy, 2 normal, 3 quick, 4 lightning
    vision: bool = True  # the paddle is a looming source for the brain (LC4 / LPLC2)
    escape: bool = True  # with --brain-actions: GF >= short_hz -> short-mode jump
    short_hz: float = 60.0
    # escape-flight *emulation* (an external thorax force after take-off, not wing
    # physics): off, per the project rule; dodges come from the real jump alone
    flight: bool = False
    model: dict = field(default_factory=dict)  # SwatterConfig overrides (geometry, levels)


@dataclass
class WhipVisionConfig:
    """The fly sees the whip coming (--whip-vision; perpetualfly/vision/looming.py)."""

    enabled: bool = False
    looming: dict = field(default_factory=dict)  # LoomingConfig overrides


# --- real vision (--real-vision; perpetualfly/vision/{eyes,flyvis_net,bridge}.py) ---
@dataclass
class RealVisionAppConfig:
    """The fly actually sees (--real-vision; docs/VISION.md "Real vision"): FlyGym
    compound eyes -> flyvis visual system -> LC4 / LPLC2 bridge -> brain. Replaces the
    geometric looming sense (--whip-vision, the swatter's paddle source) when on.
    ``eyes_only``: compound eyes sampled at ``rate_hz`` without network / brain."""

    enabled: bool = False
    eyes_only: bool = False
    rate_hz: float = 100.0  # eye samples = flyvis steps per sim second
    backend: str = "auto"  # "flyvis" | "dark_expansion" (eyes-only fallback) | "auto"
    steer: bool = False  # also drive LC10a from small moving objects
    vision: dict = field(default_factory=dict)  # RealVisionConfig overrides
# --- end real vision ---


@dataclass
class FlightModeConfig:
    """Real flapping-wing flight as an app mode (``--flight``; perpetualfly/flight/mode.py,
    docs/FLIGHT.md section 7). Key L takes off / lands; with --brain-actions the giant
    fibre escape jump starts the wings and flies away from the threat."""

    enabled: bool = False
    climb_mm: float = 2.0  # hover set point above the take-off COM
    hover_s: float | None = 4.0  # manual take-off: land after this (None = wait for L)
    wing_ramp_s: float = 0.01  # wingbeat fade-in at take-off
    # escape flight (brain giant fibre -> jump -> wings)
    escape_speed: float = 200.0  # mm/s
    escape_ramp_s: float = 0.0  # velocity target ramp (0 = step; the accel cap limits it)
    escape_s: float = 0.8  # powered flight before braking (+ brake_s + landing ~ 1.3 s)
    escape_climb_mm: float = 1.0
    # COM height above the ground while escaping (None = take-off + escape_climb_mm)
    escape_altitude_mm: float | None = 2.0
    brake_s: float = 0.25
    # escape = velocity control only (no pull back toward the take-off point), with
    # a stiffer velocity loop (HoverGains.xy_zeta while escaping)
    escape_velocity_mode: bool = True
    escape_xy_zeta: float = 3.0
    turn_rate: float = 6.0  # rad/s, heading slew toward the flight direction
    # escape more than this off the heading: no turn, fly backward (facing the threat)
    max_turn_deg: float = 100.0
    max_accel_xy: float = 5000.0  # mm/s^2, HoverGains.max_accel_xy while flying
    # manual steering (arrow keys while airborne)
    speed_step: float = 50.0  # mm/s per UP / DOWN
    max_speed: float = 250.0
    turn_step_deg: float = 30.0  # per LEFT / RIGHT
    # landing
    land_speed: float = 15.0  # mm/s descent
    land_pitch_s: float = 0.2  # wings beat on after touchdown while pitching down
    wing_stop_s: float = 0.01
    max_land_s: float = 4.0  # still no touchdown after this: stop the wings anyway
    max_takeoff_tilt_deg: float = 45.0  # L refused when the fly is not upright
    max_wings_tilt_deg: float = 80.0  # tilt at the end of the leg stroke: no wings above
    crash_tilt_deg: float = 120.0  # thorax tilt (hover posture ~48 deg) = crash
    crash_grace_s: float = 0.15  # after take-off: the short-mode jump tumbles first
    manual_jump: dict = field(default_factory=dict)  # Jump overrides for the L take-off

    @classmethod
    def from_dict(cls, d: dict) -> "FlightModeConfig":
        known = set(cls.__dataclass_fields__)
        bad = set(d) - known
        if bad:
            raise ValueError(f"Unknown config keys for FlightModeConfig: {sorted(bad)}")
        return cls(**d)


@dataclass
class CourseAppConfig:
    """Obstacle course (--course NAME, --course-loop; perpetualfly/course, docs/COURSE.md)."""

    name: str | None = None  # built-in name or a .json / .toml path; None = endless mode
    loop: bool = False  # start a new lap after the finish (else the app quits)


@dataclass
class JobAppConfig:
    """Eternal job (--job NAME, --job-config JSON; perpetualfly/jobs, docs/JOBS.md)."""

    name: str | None = None
    config: dict = field(default_factory=dict)  # job config overrides


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
    # Optional features (all off by default; see perpetualfly/app.py Session)
    swatter: SwatterAppConfig = field(default_factory=SwatterAppConfig)
    stress: StressConfig = field(default_factory=StressConfig)  # --stress (docs/STRESS.md)
    whip_vision: WhipVisionConfig = field(default_factory=WhipVisionConfig)
    real_vision: RealVisionAppConfig = field(default_factory=RealVisionAppConfig)
    course: CourseAppConfig = field(default_factory=CourseAppConfig)
    job: JobAppConfig = field(default_factory=JobAppConfig)
    flight: FlightModeConfig = field(default_factory=FlightModeConfig)  # --flight
    # --- taste patches (--taste-patches; perpetualfly/senses/taste.py, docs/TASTE.md) ---
    taste: TasteConfig = field(default_factory=TasteConfig)
    # --- odour zones / fear learning (--odor-zones, --learning; docs/FEAR_LEARNING.md) ---
    odor: OdorConfig = field(default_factory=OdorConfig)

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
