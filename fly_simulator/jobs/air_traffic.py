"""AIR TRAFFIC CONTROLLER FLY: the fly is an air-traffic controller forever
(docs/JOBS.md, "air_traffic").

A control tower cab at night: a glass ring of windows, radar consoles, a flight-strip
board, the controller's swivel stool in the middle, and below it an airfield with a
lit runway (runway 09 at our own airport "FLY"). The controller is the simulated fly
(the real NeuroMechFly body), standing on the stool.

* **Planes (kinematic; labelled).** A pool of small airliners on mocap bodies (no
  contacts), with navigation lights and a strobe. They arrive from the left or the
  right approach corridor. If nobody is being served a new plane is **called** at
  once: it flies inbound to its calling point (world azimuth 30-55 deg left or right,
  16 mm out), loiters there in a small circle and requests clearance. Otherwise it
  joins its side's **holding stack** (circling far out, stacked in altitude); holding
  planes are called in arrival order.
* **What the controller sees (LC10a; the chase / rings mapping).** The requesting
  plane is a small moving target: ``PursuitVision`` (fly_simulator/games/chase.py)
  turns its angular size and azimuth in each (gaze-stabilised) eye into LC10a
  rates (150 Hz x size(theta) x ecc(azimuth) x fov), sent as ``manual`` LC10a
  events. Only the requesting plane is a visual source.
* **How it turns (a documented body-yaw mapping).** turn = tanh(turn_L / 25 Hz) -
  tanh(turn_R / 25 Hz) (the DNa01 / DNa02 groups, as in the games) sets the yaw rate
  of the **swivel stool** (a hinge with a velocity servo) the fly stands on in the
  standing pose with its tarsi adhering: 150 deg/s x min(|turn| / 0.8, 1), + = left;
  the fly turns with the stool by contact and adhesion. The walking CPG's own turn in
  place drifted 1-2 mm per quarter turn (docs). With ``--brain`` the turn comes from
  the connectome (LC10a -> ipsilateral DNa01/02, docs/GAMES.md). Without a brain a
  labelled scripted controller (a proportional turn toward the plane after a 0.15 s
  reaction time, only while it is in the field of view) feeds the same map.
  Scripted housekeeping (counted): between planes, or while the requesting plane is
  behind the fly (out of sight), the stool swivels back toward the runway.
* **Clearance.** When the fly's heading points at the requesting plane within
  ``tolerance_deg`` for ``hold_s`` the plane is CLEARED TO LAND: it flies a
  kinematic approach (a Bezier curve onto the extended centre line), touches down on
  the runway, rolls out and disappears (back in the pool). The next plane is called.
  A plane that waits ``divert_s`` diverts to another airport (counted).
* **Counters.** Planes cleared (the work counter), mean response time (request ->
  clearance), holding-pattern length (now / max), near misses (a plane arriving while
  ``crowded`` planes are already holding: the airspace conflict alert), diverted,
  and whether the first turn after a request went toward the plane.

Experiment (``fly_simulator/jobs/air_traffic_experiment.py``, ``python -m
fly_simulator.jobs.air_traffic_experiment --planes N``): brain vs mirror (the eyes
swapped onto the other side's LC10a) vs none (no turn drive), paired single planes at
+-40-90 deg.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import mujoco as mj
import numpy as np

from fly_simulator.jobs import air_traffic_assets as A
from fly_simulator.jobs.bouncer import BouncerStance
from fly_simulator.jobs.base import CameraPreset, EternalJob, JobConfig
from fly_simulator.jobs.geometry import contact_kwargs, quat_axis_angle, spot_or_directional, wrap_angle
from fly_simulator.jobs.registry import register_job

P = "atc/"
CONTROLS = ("auto", "brain", "mirror", "none", "script")


@dataclass
class AirTrafficConfig(JobConfig):
    # --- the cab / airfield (mm) ------------------------------------------------------
    cab_r: float = 6.8
    ground_z: float = -6.0  # the airfield below the tower (a fly-scale tower)
    runway_x0: float = 30.0
    runway_len: float = 70.0
    runway_w: float = 3.0
    # --- traffic -----------------------------------------------------------------------
    n_planes: int = 8  # pooled plane bodies
    spawn_every_s: float = 4.5  # mean arrival interval (x U(0.6, 1.4))
    call_az_deg: tuple = (30.0, 55.0)  # calling point: world azimuth range, left / right
    call_dist: float = 16.0
    call_z: tuple = (1.5, 3.5)
    loiter_r: float = 1.2  # the calling plane circles here while it waits
    loiter_period_s: float = 8.0
    call_fly_s: float = 2.5  # stack / arrival -> calling point
    stack_xy: float = 20.0  # holding fixes at (stack_x, +-stack_xy)
    stack_x: float = 34.0
    stack_r: float = 3.0
    stack_z0: float = 6.0
    stack_dz: float = 1.5
    alternate_p: float = 0.7  # a new arrival comes from the other corridor with this probability
    crowded: int = 3  # a new arrival with this many already holding = a near miss
    divert_s: float = 30.0
    visual_radius_mm: float = 1.4  # the plane's size for LC10a (3.3 mm long, 3.6 mm span)
    # --- clearance -----------------------------------------------------------------------
    tolerance_deg: float = 15.0
    hold_s: float = 0.15
    land_s: float = 5.0
    rollout_s: float = 3.0
    # --- the controller --------------------------------------------------------------------
    control: str = "auto"  # auto (brain if --brain else script) / brain / mirror / none / script
    turn_max_dps: float = 150.0  # the swivel stool's top yaw rate
    turn_deadband: float = 0.06
    turn_full: float = 0.8  # |turn| >= this = the top rate; below it proportional
    swivel_r: float = 3.0  # the stool top (mm); centred on the fly's spawn point
    swivel_x: float = 0.55
    swivel_kv: float = 0.2  # velocity servo gain of the stool (uN mm s / rad)
    face_runway_deg: float = 60.0  # between planes: facing farther away than this = turn back
    behind_deg: float = 150.0  # a calling plane farther behind than this: turn back too
    r_ref_hz: float = 25.0  # tanh(turn DN rate / r_ref)
    script_gain_deg: float = 50.0  # scripted: turn = clip(error / this, +-1)
    script_delay_s: float = 0.15
    first_turn_s: float = 0.6  # the "first turn" direction is measured over this
    # --- looks -------------------------------------------------------------------------------
    shadows: bool = False
    n_stars: int = 90


@dataclass
class Plane:
    k: int
    state: str = "hidden"  # hidden / arriving / holding / calling / landing / rollout
    side: int = 1  # +1 left (+y), -1 right
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    t0: float = 0.0
    path: tuple | None = None  # bezier control points
    T: float = 1.0
    call_pt: np.ndarray | None = None
    level: int = 0
    phase0: float = 0.0
    number: int = 0
    t_called: float = 0.0
    t_arrived: float = 0.0
    flight: str = ""


class _Target:
    """The calling plane as a ``PursuitVision`` leader (``active``, ``position3``,
    ``cfg.visual_radius_mm``)."""

    def __init__(self, job: "AirTrafficJob") -> None:
        self.job = job
        self.cfg = type("C", (), {"visual_radius_mm": job.cfg.visual_radius_mm})()

    @property
    def active(self) -> bool:
        return self.job.requesting() is not None

    def position3(self) -> np.ndarray:
        return self.job.planes[self.job.calling].pos.copy()


class _MirrorSink:
    """Swaps the eye: left eye -> right LC10a and vice versa (the mirror control)."""

    def __init__(self, link) -> None:
        self.link = link

    def send(self, ev, source: str = "") -> None:
        from dataclasses import replace

        side = {"left": "right", "right": "left"}.get(ev.side, ev.side)
        self.link.send(replace(ev, side=side), source=source)


@register_job
class AirTrafficJob(EternalJob):
    name = "air_traffic"
    znear = 0.1
    zfar = 300.0
    title = "AIR TRAFFIC CONTROLLER FLY"
    tagline = "the fly clears planes to land forever"
    work_label = "planes cleared"
    config_cls = AirTrafficConfig
    required_names = (P + "plane0", P + "runway", P + "sweep0")

    def __init__(self, cfg: AirTrafficConfig | None = None) -> None:
        super().__init__(cfg)
        c = self.cfg
        self.rng = np.random.default_rng(c.seed + 57)
        self.planes = [Plane(k) for k in range(c.n_planes)]
        self.calling: int | None = None
        self.holding: list[int] = []  # plane indices in call order
        self.n_cleared = self.n_diverted = self.n_near_miss = self.n_arrived = 0
        self.max_holding = 0
        self.resp_sum = 0.0
        self.resp_n = 0
        self.last_resp: float | None = None
        self.n_first_ok = self.n_first_trials = 0
        self._first: dict | None = None
        self.turn = 0.0  # the turn signal (+ = left) in use
        self.hk: str | None = None  # housekeeping (scripted): "face" = swivel back to the runway
        self.hk_turn = 0.0
        self.housekeeping = True  # walk back to the spot / face the runway (off in the experiment)
        self.n_face_idle = 0
        self.n_face_back = 0
        self.dn = (0.0, 0.0)
        self._aligned_since: float | None = None
        self._next_spawn = 1.0
        self.flash = ""
        self._flash_until = -1e9
        self._err_hist: deque = deque(maxlen=800)  # (t, error) every 1 ms: the scripted delay
        self.vision = None
        self._flight_no = 100
        self._last_side = 1
        self.events: deque = deque(maxlen=200)  # (kind, plane no, value) for tests / stats

    # ------------------------------------------------------------ build
    def configure_app(self, app_cfg) -> None:
        super().configure_app(app_cfg)
        app_cfg.render.reflections = False

    def extension(self, world) -> None:
        spec = world.mjcf_root
        self._materials(spec)
        self._add_cab(spec)
        self._add_airfield(spec)
        self._add_planes(spec)
        self._add_lights(spec)

    def _materials(self, spec) -> None:
        c = self.cfg
        T = A.add_textured_material
        A.add_texture(spec, P + "tex_radar", A.radar_texture())
        T(spec, P + "radar", P + "tex_radar", rgba=(1, 1, 1, 1), emission=0.9, specular=0.0)
        A.add_texture(spec, P + "tex_strips", A.strips_texture())
        T(spec, P + "strips", P + "tex_strips", rgba=(1, 1, 1, 1), emission=0.5)
        A.add_texture(spec, P + "tex_ground", A.ground_texture(c.seed))
        T(spec, P + "ground", P + "tex_ground", rgba=(1, 1, 1, 1), specular=0.0)
        A.add_texture(spec, P + "tex_runway", A.runway_texture())
        T(spec, P + "runway", P + "tex_runway", rgba=(1, 1, 1, 1), specular=0.2, emission=0.15)
        A.add_texture(spec, P + "tex_floor", A.floor_texture())
        T(spec, P + "floor", P + "tex_floor", rgba=(1, 1, 1, 1), specular=0.05)
        mats = {
            "console": ((0.16, 0.17, 0.2, 1), dict(specular=0.4)),
            "console_top": ((0.08, 0.08, 0.1, 1), dict(specular=0.6)),
            "glass": ((0.5, 0.7, 0.9, 0.08), dict(specular=0.9, shininess=0.9)),
            "mullion": ((0.2, 0.21, 0.24, 1), dict(specular=0.5)),
            "spot": ((0.12, 0.1, 0.08, 1), dict(specular=0.2)),
            "spot_ring": ((0.9, 0.7, 0.2, 1), dict(emission=0.4)),
            "plane": ((0.92, 0.93, 0.95, 1), dict(specular=0.6, shininess=0.6, emission=0.15)),
            "plane_wing": ((0.72, 0.74, 0.78, 1), dict(specular=0.6, emission=0.1)),
            "tail": ((0.1, 0.45, 0.85, 1), dict(specular=0.5, emission=0.2)),
            "nav_red": ((1.0, 0.1, 0.1, 1), dict(emission=1.0)),
            "nav_green": ((0.1, 1.0, 0.2, 1), dict(emission=1.0)),
            "strobe_on": ((1.0, 1.0, 1.0, 1), dict(emission=1.0)),
            "strobe_off": ((0.3, 0.3, 0.3, 1), dict(emission=0.0)),
            "edge_light": ((1.0, 0.9, 0.6, 1), dict(emission=1.0)),
            "approach_light": ((1.0, 1.0, 0.95, 1), dict(emission=1.0)),
            "taxi_light": ((0.2, 0.5, 1.0, 1), dict(emission=1.0)),
            "star": ((0.9, 0.9, 1.0, 1), dict(emission=1.0)),
            "city": ((1.0, 0.75, 0.4, 1), dict(emission=0.6)),
            "blip": ((0.5, 1.0, 0.5, 1), dict(emission=1.0)),
            "blip_call": ((1.0, 0.9, 0.3, 1), dict(emission=1.0)),
            "sweep": ((0.3, 1.0, 0.4, 0.8), dict(emission=1.0)),
            "tower": ((0.25, 0.25, 0.28, 1), dict(specular=0.2)),
            "hangar": ((0.18, 0.2, 0.22, 1), dict(specular=0.3)),
        }
        for nm, (rgba, kw) in mats.items():
            spec.add_material(name=P + nm, rgba=rgba, **kw)

    def _add_cab(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        A.add_mesh(spec, P + "floor_mesh", A.disc_mesh(c.cab_r, 0.1, 48))
        wb.add_geom(name=P + "floor", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "floor_mesh",
                    pos=(0, 0, -0.051), material=P + "floor", **vis)
        # the controller's swivel stool: a round top on a hinge (z) turned by a velocity
        # servo; the fly stands on it (real contact + its tarsal adhesion) and turns with it
        R, top = c.swivel_r, 0.08
        sw = wb.add_body(name=P + "swivel", pos=(c.swivel_x, 0.0, 0.0))
        sw.add_joint(name=P + "swivel", type=mj.mjtJoint.mjJNT_HINGE, axis=(0, 0, 1), damping=1e-4,
                     armature=1e-4)
        sw.add_geom(name=P + "swivel_top", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(R, top / 2, 0),
                    pos=(0, 0, top / 2), material=P + "spot", mass=2e-3, **contact_kwargs("fly", 1.0))
        ring = np.array([[(R - 0.05) * math.cos(a), (R - 0.05) * math.sin(a), top + 0.005]
                         for a in np.linspace(0, 2 * math.pi, 49)])
        A.add_mesh(spec, P + "spot_ring_mesh", A.polyline_tube(ring, 0.03, 6))
        sw.add_geom(name=P + "spot_ring", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "spot_ring_mesh",
                    material=P + "spot_ring", **vis)
        sw.add_geom(name=P + "spot_mark", type=mj.mjtGeom.mjGEOM_BOX, size=(0.35, 0.05, 0.004),
                    pos=(R - 0.45, 0, top + 0.004), material=P + "spot_ring", **vis)
        act = spec.add_actuator(name=P + "swivel_motor", target=P + "swivel", trntype=mj.mjtTrn.mjTRN_JOINT)
        act.set_to_velocity(kv=c.swivel_kv)
        act.ctrlrange = (-10.0, 10.0)
        # windows: 12 glass panels + mullions, a low sill wall, a roof rim
        n = 12
        for i in range(n):
            a0 = 2 * math.pi * i / n + math.pi / n  # no mullion straight ahead / behind
            am = a0 + math.pi / n
            w = 2 * c.cab_r * math.sin(math.pi / n)
            pos = (c.cab_r * math.cos(am), c.cab_r * math.sin(am))
            q = quat_axis_angle((0, 0, 1), am)
            wb.add_geom(name=P + f"glass{i}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.01, w / 2, 3.3),
                        pos=(*pos, 4.9), quat=q, material=P + "glass", **vis)
            wb.add_geom(name=P + f"sill{i}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.15, w / 2 + 0.05, 0.8),
                        pos=(*pos, 0.8), quat=q, material=P + "console", **vis)
            px, py = c.cab_r * math.cos(a0), c.cab_r * math.sin(a0)
            wb.add_geom(name=P + f"mullion{i}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.06, 0.06, 3.3),
                        pos=(px, py, 4.9), material=P + "mullion", **vis)
        ring = np.array([[c.cab_r * math.cos(a), c.cab_r * math.sin(a), 8.25] for a in np.linspace(0, 2 * math.pi, 49)])
        A.add_mesh(spec, P + "roof_rim_mesh", A.polyline_tube(ring, 0.12, 8))
        wb.add_geom(name=P + "roof_rim", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "roof_rim_mesh",
                    material=P + "mullion", **vis)
        # consoles along the front arc, with two radar scopes and a strip board
        for i, a in enumerate(np.radians([-75, -50, -25, 25, 50, 75])):  # the view ahead stays clear
            r = c.cab_r - 1.3
            q = quat_axis_angle((0, 0, 1), a)
            wb.add_geom(name=P + f"console{i}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.9, 1.05, 0.7),
                        pos=(r * math.cos(a), r * math.sin(a), 0.7), quat=q, material=P + "console", **vis)
            wb.add_geom(name=P + f"console_top{i}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.95, 1.1, 0.04),
                        pos=(r * math.cos(a), r * math.sin(a), 1.42), quat=q, material=P + "console_top", **vis)
        self.scopes = []
        for k, a_deg in enumerate((-25.0, 25.0)):
            a = math.radians(a_deg)
            r = c.cab_r - 1.35
            ctr = np.array([r * math.cos(a), r * math.sin(a), 1.47])
            A.add_mesh(spec, P + f"scope{k}_mesh", A.disc_mesh(0.75, 0.02, 40))
            wb.add_geom(name=P + f"scope{k}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + f"scope{k}_mesh",
                        pos=tuple(ctr), material=P + "radar", **vis)
            sw = wb.add_body(name=P + f"sweep{k}", mocap=True, pos=tuple(ctr + np.array([0, 0, 0.02])))
            sw.add_geom(name=P + f"sweep{k}_g", type=mj.mjtGeom.mjGEOM_BOX, size=(0.36, 0.012, 0.004),
                        pos=(0.36, 0, 0), material=P + "sweep", **vis)
            self.scopes.append(ctr)
            for j in range(c.n_planes):
                b = wb.add_body(name=P + f"blip{k}_{j}", mocap=True, pos=(0, 0, -60.0))
                b.add_geom(name=P + f"blip{k}_{j}_g", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.04, 0, 0),
                           material=P + "blip", **vis)
        a = math.radians(-75.0)
        r = c.cab_r - 1.35
        A.add_mesh(spec, P + "strips_mesh", A.panel_mesh(1.6, 0.8, 0.02))
        wb.add_geom(name=P + "strips", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "strips_mesh",
                    pos=(r * math.cos(a), r * math.sin(a), 1.9), quat=A.panel_quat((-math.cos(a), -math.sin(a), 0)),
                    material=P + "strips", **vis)

    def _add_airfield(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        gz = c.ground_z
        wb.add_geom(name=P + "ground", type=mj.mjtGeom.mjGEOM_BOX, size=(160.0, 160.0, 0.1),
                    pos=(30.0, 0.0, gz - 0.1), material=P + "ground", **vis)
        # the tower shaft below the cab
        wb.add_geom(name=P + "shaft", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(3.5, -gz / 2, 0),
                    pos=(0, 0, gz / 2 - 0.2), material=P + "tower", **vis)
        wb.add_geom(name=P + "cab_base", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(c.cab_r + 0.3, 0.2, 0),
                    pos=(0, 0, -0.35), material=P + "tower", **vis)
        L, W = c.runway_len, c.runway_w
        A.add_mesh(spec, P + "runway_mesh", A.panel_mesh(L, W, 0.02))  # textured face up
        wb.add_geom(name=P + "runway", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + "runway_mesh",
                    pos=(c.runway_x0 + L / 2, 0.0, gz + 0.01), quat=quat_axis_angle((1, 0, 0), -math.pi / 2),
                    material=P + "runway", **vis)
        # edge lights, the approach light bar, a taxiway with blue lights, hangars
        for i, x in enumerate(np.linspace(c.runway_x0, c.runway_x0 + L, 21)):
            for s in (-1, 1):
                wb.add_geom(name=P + f"edge{i}_{int(s > 0)}", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.12, 0, 0),
                            pos=(x, s * (W / 2 + 0.3), gz + 0.12), material=P + "edge_light", **vis)
        for i in range(8):
            x = c.runway_x0 - 1.5 - 1.6 * i
            wb.add_geom(name=P + f"approach{i}", type=mj.mjtGeom.mjGEOM_BOX, size=(0.08, 0.8 - 0.05 * i, 0.06),
                        pos=(x, 0.0, gz + 0.08), material=P + "approach_light", **vis)
        wb.add_geom(name=P + "taxiway", type=mj.mjtGeom.mjGEOM_BOX, size=(L / 2, 0.8, 0.01),
                    pos=(c.runway_x0 + L / 2, -7.0, gz + 0.01), material=P + "runway", **vis)
        for i, x in enumerate(np.linspace(c.runway_x0, c.runway_x0 + L, 17)):
            wb.add_geom(name=P + f"taxi{i}", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.1, 0, 0),
                        pos=(x, -6.0, gz + 0.1), material=P + "taxi_light", **vis)
        for i, x in enumerate((30.0, 42.0, 54.0)):
            wb.add_geom(name=P + f"hangar{i}", type=mj.mjtGeom.mjGEOM_BOX, size=(4.5, 3.0, 2.2),
                        pos=(x, -14.0, gz + 2.2), material=P + "hangar", **vis)
        # stars (a dome) and a far city glow
        rng = np.random.default_rng(c.seed + 5)
        for i in range(c.n_stars):
            az = rng.uniform(-math.pi, math.pi)
            el = math.asin(rng.uniform(0.08, 0.95))
            R = 220.0
            wb.add_geom(name=P + f"star{i}", type=mj.mjtGeom.mjGEOM_SPHERE, size=(rng.uniform(0.25, 0.6), 0, 0),
                        pos=(R * math.cos(el) * math.cos(az), R * math.cos(el) * math.sin(az), R * math.sin(el)),
                        material=P + "star", **vis)
        for i in range(60):
            az = rng.uniform(-1.3, 1.3)
            R = rng.uniform(150, 190)
            wb.add_geom(name=P + f"city{i}", type=mj.mjtGeom.mjGEOM_BOX,
                        size=(0.3, rng.uniform(0.2, 0.8), rng.uniform(0.15, 0.6)),
                        pos=(R * math.cos(az), R * math.sin(az), gz + 0.5), material=P + "city", **vis)
        sky = spec.texture("skybox")
        if sky is not None:
            sky.rgb1 = (0.02, 0.03, 0.08)
            sky.rgb2 = (0.0, 0.0, 0.01)

    def _add_planes(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        vis = dict(contact_kwargs("visual"), mass=0.0)
        meshes = A.plane_meshes(3.3, 3.6)
        for nm, md in meshes.items():
            A.add_mesh(spec, P + f"plane_{nm}_mesh", md)
        mat = {"body": "plane", "wings": "plane_wing", "fin": "tail"}
        for k in range(c.n_planes):
            b = wb.add_body(name=P + f"plane{k}", mocap=True, pos=(0.0, 0.0, -80.0 - 3 * k))
            for nm in meshes:
                b.add_geom(name=P + f"plane{k}_{nm}", type=mj.mjtGeom.mjGEOM_MESH, meshname=P + f"plane_{nm}_mesh",
                           material=P + mat[nm], **vis)
            b.add_geom(name=P + f"plane{k}_navL", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.06, 0, 0),
                       pos=(-0.5, 1.77, -0.03), material=P + "nav_red", **vis)
            b.add_geom(name=P + f"plane{k}_navR", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.06, 0, 0),
                       pos=(-0.5, -1.77, -0.03), material=P + "nav_green", **vis)
            b.add_geom(name=P + f"plane{k}_strobe", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.05, 0, 0),
                       pos=(-1.58, 0.0, 1.5), material=P + "strobe_off", **vis)
            b.add_geom(name=P + f"plane{k}_landing", type=mj.mjtGeom.mjGEOM_SPHERE, size=(0.07, 0, 0),
                       pos=(1.65, 0.0, -0.07), material=P + "strobe_off", **vis)

    def _add_lights(self, spec) -> None:
        c = self.cfg
        wb = spec.worldbody
        spec.visual.headlight.ambient = (0.2, 0.2, 0.25)
        spec.visual.headlight.diffuse = (0.3, 0.3, 0.35)
        spec.visual.headlight.specular = (0.05, 0.05, 0.05)
        key = np.array([-4.0, 3.0, 7.8])
        wb.add_light(name=P + "cab_light", type=spot_or_directional(c.shadows), pos=tuple(key),
                     dir=tuple(np.array([0.5, -0.3, 0.0]) - key), diffuse=(0.45, 0.42, 0.38),
                     specular=(0.2, 0.2, 0.2), cutoff=50.0, exponent=1.0, castshadow=bool(c.shadows))
        wb.add_light(name=P + "console_glow", type=mj.mjtLightType.mjLIGHT_POINT, pos=(5.5, 0.0, 2.2),
                     diffuse=(0.15, 0.35, 0.2), specular=(0.0, 0.0, 0.0), attenuation=(0.5, 0.1, 0.02),
                     castshadow=False)
        wb.add_light(name=P + "field", type=mj.mjtLightType.mjLIGHT_DIRECTIONAL, pos=(40.0, 0.0, 30.0),
                     dir=(0.0, 0.2, -1.0), diffuse=(0.12, 0.12, 0.16), specular=(0.0, 0.0, 0.0), castshadow=False)

    # ------------------------------------------------------------ attach
    def on_attach(self) -> None:
        m, s = self.sim.model, self.session
        c = self.cfg
        gp = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "ground_plane")
        if gp >= 0:  # the fly's floor stays (physics); the cab floor draws it
            m.geom_group[gp] = 3
            m.geom_rgba[gp, 3] = 0.0
        self.p_mocap = [int(m.body_mocapid[m.body(P + f"plane{k}").id]) for k in range(c.n_planes)]
        self.p_strobe = [m.geom(P + f"plane{k}_strobe").id for k in range(c.n_planes)]
        self.p_landing = [m.geom(P + f"plane{k}_landing").id for k in range(c.n_planes)]
        self.sweep_mocap = [int(m.body_mocapid[m.body(P + f"sweep{k}").id]) for k in range(2)]
        self.blip_mocap = [[int(m.body_mocapid[m.body(P + f"blip{k}_{j}").id]) for j in range(c.n_planes)]
                           for k in range(2)]
        self.blip_gid = [[m.geom(P + f"blip{k}_{j}_g").id for j in range(c.n_planes)] for k in range(2)]
        self.mat = {n: m.material(P + n).id for n in ("strobe_on", "strobe_off", "blip", "blip_call")}
        self.steering.set(None, 0.0)
        # the fly does not walk: it stands on the swivel stool (all tarsi adhering)
        # and the stool turns under it; the walking CPG idles underneath
        self.steering._prev = None
        self.sim.controller.signal_filter = lambda hold: np.zeros(2)
        self.swivel_act = m.actuator(P + "swivel_motor").id
        self.swivel_q = int(m.jnt_qposadr[m.joint(P + "swivel").id])
        self.acts = s.actions
        st = getattr(s, "STATIONARY_ACTIONS", None)
        if st is not None and BouncerStance.name not in st:
            s.STATIONARY_ACTIONS = (*st, BouncerStance.name)
        self._stance = None
        self._begin_stance()
        link = getattr(s, "brain", None)
        self._install_vision(link)
        self._write_planes()

    def control_mode(self) -> str:
        c = self.cfg.control
        if c == "auto":
            return "brain" if getattr(self.session, "brain", None) is not None else "script"
        return c

    def _install_vision(self, link) -> None:
        from fly_simulator.games.chase import PursuitVision

        mode = self.control_mode()
        sink = None
        if link is not None and mode in ("brain", "mirror"):
            sink = _MirrorSink(link) if mode == "mirror" else link
        tf = getattr(link, "_run_time", None) if link is not None else self.session.metrics.run_time_at
        self.vision = PursuitVision(self.sim, _Target(self), sink=sink, time_fn=tf).attach()

    def set_control(self, mode: str) -> None:
        """Switch the control condition (experiment): re-installs the vision sink."""
        if mode not in CONTROLS:
            raise ValueError(f"control must be one of {CONTROLS}")
        self.cfg.control = mode
        if self.vision is not None:
            self.vision.detach()
        self._install_vision(getattr(self.session, "brain", None))

    def _begin_stance(self) -> None:
        self._stance = BouncerStance()
        self.acts.trigger(self._stance, source="job")

    # ------------------------------------------------------------ the turn map
    def swivel_rate(self) -> float:
        """The stool's commanded yaw rate (rad/s, + = left) from ``self.turn``."""
        c = self.cfg
        u = float(self.hk_turn if self.hk is not None else self.turn)
        if abs(u) < c.turn_deadband:
            return 0.0
        return math.copysign(math.radians(c.turn_max_dps) * min(abs(u) / c.turn_full, 1.0), u)

    def turn_from_dn(self, turn_l: float, turn_r: float) -> float:
        r = self.cfg.r_ref_hz
        return math.tanh(max(turn_l, 0.0) / r) - math.tanh(max(turn_r, 0.0) / r)

    def bearing_error(self, k: int | None = None) -> float | None:
        """World bearing of plane k (default: the calling one) minus the fly's
        heading (rad, + = the plane is to the left)."""
        k = self.calling if k is None else k
        if k is None:
            return None
        p = self.planes[k].pos
        th = self.sim.thorax_position()
        b = math.atan2(p[1] - th[1], p[0] - th[0])
        return wrap_angle(b - self.sim.heading())

    def _update_turn(self, t: float) -> None:
        c = self.cfg
        mode = self.control_mode()
        err = self.bearing_error() if self.requesting() is not None else None
        self._err_hist.append((t, err))
        if mode in ("brain", "mirror"):
            link = getattr(self.session, "brain", None)
            st = getattr(link, "latest", None) if link is not None else None
            d = (getattr(st, "descending", None) or {}) if st is not None else {}
            self.dn = (float(d.get("turn_L", 0.0) or 0.0), float(d.get("turn_R", 0.0) or 0.0))
            self.turn = self.turn_from_dn(*self.dn) if self.requesting() is not None else 0.0
        elif mode == "script":
            e = None
            for tt, ee in reversed(self._err_hist):
                if t - tt >= c.script_delay_s:
                    e = ee
                    break
            if e is None or abs(math.degrees(e)) > self.vision.cfg.fov_rear_deg:
                self.turn = 0.0
            else:
                self.turn = float(np.clip(math.degrees(e) / c.script_gain_deg, -1.0, 1.0))
        else:
            self.turn = 0.0

    def _home(self, t: float) -> None:
        """Scripted housekeeping (labelled; not the brain): between planes a
        controller facing farther than ``face_runway_deg`` from the runway (+x)
        swivels back toward it; and while the calling plane is behind it (>
        ``behind_deg``, outside its field of view: it could not see it) it swivels
        back toward the runway first. ``self.hk`` = "face" while it does."""
        c = self.cfg
        if not self.housekeeping:
            self.hk = None
            return
        face_err = wrap_angle(-self.sim.heading())  # + = the runway is to the left
        if self.requesting() is not None:
            behind = abs(math.degrees(self.bearing_error()))
            if self.hk is None and behind > c.behind_deg:
                self.n_face_back += 1
                self.hk = "face"
            elif self.hk is not None and (behind < c.behind_deg - 40.0 or abs(math.degrees(face_err)) < 15.0):
                self.hk = None
        elif self.hk is None and abs(math.degrees(face_err)) > c.face_runway_deg:
            self.n_face_idle += 1
            self.hk = "face"
        elif self.hk is not None and abs(math.degrees(face_err)) < 15.0:
            self.hk = None
        if self.hk == "face":
            self.hk_turn = float(np.clip(math.degrees(face_err) / 40.0, -1.0, 1.0))

    # ------------------------------------------------------------ traffic
    def _free_plane(self) -> Plane | None:
        for p in self.planes:
            if p.state == "hidden":
                return p
        return None

    def spawn(self, side: int | None = None) -> Plane | None:
        """A new arrival from the left (+1) or right (-1) corridor."""
        c = self.cfg
        p = self._free_plane()
        if p is None:
            return None
        t = self.sim.time
        if side is None:  # the corridors take turns more often than not
            side = -self._last_side if self.rng.random() < self.cfg.alternate_p else self._last_side
        p.side = int(side)
        self._last_side = p.side
        self._flight_no += 7
        p.number = self._flight_no
        p.flight = f"FW{p.number % 1000:03d}"
        az = p.side * math.radians(self.rng.uniform(70.0, 85.0))
        p.pos = np.array([55.0 * math.cos(az), 55.0 * math.sin(az), 9.0 + self.rng.uniform(0, 3)])
        p.t_arrived = t
        self.n_arrived += 1
        if len(self.holding) >= c.crowded:
            self.n_near_miss += 1
            self._flash(f"CONFLICT ALERT: {p.flight} near miss ({len(self.holding)} holding)", 2.0)
            self.events.append(("near_miss", p.number, len(self.holding)))
        if self.calling is None and not self.holding:
            self._call(p, t)
        else:
            self._to_stack(p, t)
        return p

    def _stack_fix(self, side: int) -> np.ndarray:
        c = self.cfg
        return np.array([c.stack_x, side * c.stack_xy])

    def _to_stack(self, p: Plane, t: float) -> None:
        c = self.cfg
        p.state = "arriving"
        self.holding.append(p.k)
        lv = sum(1 for k in self.holding if self.planes[k].side == p.side) - 1
        p.level = lv
        fix = self._stack_fix(p.side)
        tgt = np.array([fix[0] + c.stack_r, fix[1], c.stack_z0 + c.stack_dz * lv])
        self._fly_to(p, tgt, t, 4.0)
        self.max_holding = max(self.max_holding, len(self.holding))

    def _fly_to(self, p: Plane, tgt: np.ndarray, t: float, T: float) -> None:
        d = tgt - p.pos
        hdg = np.array([math.cos(p.yaw), math.sin(p.yaw), 0.0])
        L = float(np.linalg.norm(d))
        p.path = (p.pos.copy(), p.pos + hdg * 0.3 * L, tgt - d / max(L, 1e-9) * 0.3 * L, tgt.copy())
        p.t0, p.T = t, max(T, 0.2)

    def requesting(self) -> int | None:
        """The plane requesting clearance now (at its calling point), else None."""
        k = self.calling
        return k if k is not None and self.planes[k].state == "calling" else None

    def _call(self, p: Plane, t: float) -> None:
        """Next in line: the plane flies inbound to its calling point (then calls)."""
        c = self.cfg
        if p.k in self.holding:
            self.holding.remove(p.k)
        az = p.side * math.radians(self.rng.uniform(*c.call_az_deg))
        p.call_pt = np.array([c.call_dist * math.cos(az), c.call_dist * math.sin(az),
                              self.rng.uniform(*c.call_z)])
        p.state = "inbound"
        self.calling = p.k
        p.phase0 = self.rng.uniform(0, 2 * math.pi)
        p.t_called = t + c.call_fly_s
        self._fly_to(p, p.call_pt + self._loiter_offset(p, t + c.call_fly_s), t, c.call_fly_s)

    def _request(self, p: Plane, t: float) -> None:
        """At the calling point: requesting clearance (the LC10a target from now)."""
        p.state = "calling"
        p.t_called = t
        self._aligned_since = None
        err = self.bearing_error(p.k)
        self._first = {"t": t, "h0": self.sim.heading(), "e0": err, "k": p.k}
        self._flash(f"{p.flight}: requesting clearance ({'left' if p.side > 0 else 'right'})", 2.0)
        self.events.append(("called", p.number, round(math.degrees(err), 1)))

    def _loiter_offset(self, p: Plane, t: float) -> np.ndarray:
        c = self.cfg
        a = p.phase0 + 2 * math.pi * (t - p.t_called) / c.loiter_period_s
        return np.array([c.loiter_r * math.cos(a), c.loiter_r * math.sin(a), 0.0])

    def clear(self, t: float) -> None:
        """Clear the calling plane to land (kinematic approach, touchdown, roll-out)."""
        c = self.cfg
        p = self.planes[self.calling]
        self.calling = None
        rt = t - p.t_called
        self.resp_sum += rt
        self.resp_n += 1
        self.last_resp = rt
        self.n_cleared += 1
        self.add_work(1)
        self._close_first(t)
        gz = c.ground_z
        td = np.array([c.runway_x0 + 4.0, 0.0, gz + 0.25])
        fin = np.array([c.runway_x0 - 14.0, 0.0, gz + 5.0])
        hdg = np.array([math.cos(p.yaw), math.sin(p.yaw), 0.0])
        p.path = (p.pos.copy(), p.pos + hdg * 6.0, fin, td)
        p.t0, p.T = t, c.land_s
        p.state = "landing"
        self._flash(f"{p.flight}: CLEARED TO LAND runway 09  ({rt:.1f} s)", 2.5)
        self.events.append(("cleared", p.number, round(rt, 2)))
        self.session.log_event("atc_cleared", flight=p.flight, response_s=round(rt, 2),
                               holding=len(self.holding), control=self.control_mode())
        self._call_next(t)

    def _call_next(self, t: float) -> None:
        if self.calling is None and self.holding:
            self._call(self.planes[self.holding[0]], t)

    def _close_first(self, t: float) -> None:
        f = self._first
        if f is None:
            return
        self._first = None
        e0 = f["e0"]
        if e0 is None or abs(math.degrees(e0)) <= self.cfg.tolerance_deg:
            return
        dh = wrap_angle(self.sim.heading() - f["h0"])
        self.n_first_trials += 1
        if dh * e0 > 0:
            self.n_first_ok += 1

    # ------------------------------------------------------------ job logic
    def update(self) -> None:
        c = self.cfg
        t = self.sim.time
        if t >= self._next_spawn:
            self.spawn()
            self._next_spawn = t + c.spawn_every_s * self.rng.uniform(0.6, 1.4)
        self._update_turn(t)
        self._home(t)
        self.sim.data.ctrl[self.swivel_act] = self.swivel_rate()
        if self._stance is None or (self.acts.action is not self._stance and self.acts.action is None
                                    and not self.fly_down()):
            self._begin_stance()
        if self._first is not None and t - self._first["t"] >= c.first_turn_s:
            self._close_first(t)
        # clearance: the heading points at the calling plane
        if self.requesting() is not None:
            p = self.planes[self.calling]
            err = self.bearing_error()
            if abs(math.degrees(err)) <= c.tolerance_deg:
                if self._aligned_since is None:
                    self._aligned_since = t
                if t - self._aligned_since >= c.hold_s:
                    self.clear(t)
            else:
                self._aligned_since = None
            if self.calling is not None and t - p.t_called >= c.divert_s:
                self._divert(p, t)
        self._move_planes(t)
        self._div = getattr(self, "_div", 0) + 1
        if self._div >= 10:
            self._div = 0
            self._write_planes()
        if self.flash and self.run_time() > self._flash_until:
            self.flash = ""
        v = self.vision
        if v is not None and len(v.sent) > 2000:
            del v.sent[:-500]

    def _divert(self, p: Plane, t: float) -> None:
        self.calling = None
        self._first = None
        self.n_diverted += 1
        self._flash(f"{p.flight}: DIVERTING (no clearance in {self.cfg.divert_s:.0f} s)", 2.0)
        self.events.append(("diverted", p.number, round(t - p.t_called, 1)))
        away = np.array([p.pos[0] + 30.0 * math.cos(p.yaw), p.pos[1] + 30.0 * math.sin(p.yaw), 14.0])
        self._fly_to(p, away, t, 4.0)
        p.state = "leaving"
        self._call_next(t)

    def _move_planes(self, t: float) -> None:
        c = self.cfg
        for p in self.planes:
            if p.state == "hidden":
                continue
            old = p.pos.copy()
            if p.path is not None:
                u = (t - p.t0) / p.T
                p.pos = A.bezier(*p.path, u)
                if u >= 1.0:
                    p.path = None
                    if p.state == "landing":
                        p.state = "rollout"
                        p.t0 = t
                    elif p.state == "leaving":
                        p.state = "hidden"
                    elif p.state == "arriving":
                        p.state = "holding"
                        p.t0 = t
                    elif p.state == "inbound":
                        self._request(p, t)
            elif p.state == "holding":
                fix = self._stack_fix(p.side)
                a = 2 * math.pi * (t - p.t0) / 12.0 * p.side
                p.pos = np.array([fix[0] + c.stack_r * math.cos(a), fix[1] + c.stack_r * math.sin(a),
                                  c.stack_z0 + c.stack_dz * p.level])
            elif p.state == "calling":
                p.pos = p.call_pt + self._loiter_offset(p, t)
            elif p.state == "rollout":
                u = min((t - p.t0) / c.rollout_s, 1.0)
                s = 1.0 - (1.0 - u) ** 2
                x0 = c.runway_x0 + 4.0
                p.pos = np.array([x0 + 30.0 * s, 0.0, c.ground_z + 0.25])
                if u >= 1.0:
                    p.state = "hidden"
            d = p.pos - old
            if float(np.hypot(d[0], d[1])) > 1e-5:
                p.yaw = math.atan2(d[1], d[0])
        # holding stack levels compact as planes leave
        for side in (1, -1):
            lv = 0
            for k in self.holding:
                q = self.planes[k]
                if q.side == side:
                    q.level = lv
                    lv += 1

    def _write_planes(self) -> None:
        m, d = self.sim.model, self.sim.data
        t = self.sim.time
        strobe = (t % 1.0) < 0.08
        for p in self.planes:
            mid = self.p_mocap[p.k]
            if p.state == "hidden":
                d.mocap_pos[mid] = (0.0, 0.0, -80.0 - 3 * p.k)
            else:
                d.mocap_pos[mid] = p.pos
                pitch = 0.0
                if p.state == "landing":
                    pitch = -0.06
                bank = 0.0
                if p.state in ("holding", "calling"):
                    bank = 0.3 * p.side if p.state == "holding" else 0.25
                q = quat_axis_angle((0, 0, 1), p.yaw)
                qb = quat_axis_angle((1, 0, 0), bank)
                qp = quat_axis_angle((0, 1, 0), -pitch)
                from fly_simulator.jobs.geometry import quat_mul

                d.mocap_quat[mid] = quat_mul(quat_mul(q, qp), qb)
            m.geom_matid[self.p_strobe[p.k]] = self.mat["strobe_on" if strobe else "strobe_off"]
            m.geom_matid[self.p_landing[p.k]] = self.mat["strobe_on" if p.state in ("landing", "calling")
                                                          else "strobe_off"]
        # the radar: sweep + blips (world xy scaled onto the scope, up = +x)
        for s in range(2):
            d.mocap_quat[self.sweep_mocap[s]] = quat_axis_angle((0, 0, 1), -2 * math.pi * t / 3.0)
            ctr = self.scopes[s]
            for p in self.planes:
                bid = self.blip_mocap[s][p.k]
                if p.state == "hidden":
                    d.mocap_pos[bid] = (0, 0, -60.0)
                    continue
                v = p.pos[:2] / 60.0 * 0.7
                if float(np.hypot(*v)) > 0.7:
                    v = v / float(np.hypot(*v)) * 0.7
                d.mocap_pos[bid] = (ctr[0] + v[0], ctr[1] + v[1], ctr[2] + 0.03)
                m.geom_matid[self.blip_gid[s][p.k]] = self.mat["blip_call" if p.k == self.calling else "blip"]

    # ------------------------------------------------------------ reset
    def on_reset(self) -> None:
        self.steering.set(None, 0.0)
        self.hk = None
        self.sim.data.ctrl[self.swivel_act] = 0.0
        self._begin_stance()
        self._aligned_since = None
        self._err_hist.clear()
        if self._first is not None:
            self._first["h0"] = self.sim.heading()

    def reset_props(self) -> None:
        self._write_planes()

    # ------------------------------------------------------------ HUD / camera
    def _flash(self, text: str, secs: float) -> None:
        self.flash = text
        self._flash_until = self.run_time() + secs

    def mean_response_s(self) -> float | None:
        return self.resp_sum / self.resp_n if self.resp_n else None

    def camera_target(self) -> np.ndarray:
        return np.array([7.0, 0.0, -0.3])

    def camera_preset(self) -> CameraPreset:
        return CameraPreset(azimuth=0.0, elevation=-19.0, distance=21.0, tau_s=0.3)

    def job_hud_lines(self) -> list[str]:
        mr = self.mean_response_s()
        err = self.bearing_error()
        call = self.planes[self.calling] if self.calling is not None else None
        lines = [
            f"CLEARED {self.n_cleared}   mean response {mr:.1f} s" if mr is not None else
            f"CLEARED {self.n_cleared}   mean response --",
            f"holding {len(self.holding)} (max {self.max_holding})   near misses {self.n_near_miss}   "
            f"diverted {self.n_diverted}   first turn toward the plane {self.n_first_ok}/{self.n_first_trials}",
            (f"calling: {call.flight} at {math.degrees(err):+.0f} deg ({'left' if err > 0 else 'right'})"
             if call is not None and call.state == "calling" else
             (f"inbound: {call.flight}" if call is not None else "calling: --"))
            + f"   control: {self.control_mode()}",
        ]
        mode = self.control_mode()
        if mode in ("brain", "mirror"):
            L = self.vision.eye_state[0].rate_hz if self.vision is not None else 0.0
            R = self.vision.eye_state[1].rate_hz if self.vision is not None else 0.0
            lines.append(f"SEES LC10a L {L:3.0f} R {R:3.0f} Hz{' (mirrored)' if mode == 'mirror' else ''}   "
                         f"DOES DNa01/02 L {self.dn[0]:3.0f} R {self.dn[1]:3.0f} Hz -> turn {self.turn:+.2f}")
        elif mode == "script":
            lines.append(f"(no brain: scripted orientation, turn {self.turn:+.2f})")
        if self.flash:
            lines.append(">> " + self.flash)
        return lines

    def job_stats(self) -> dict[str, Any]:
        mr = self.mean_response_s()
        return {
            "cleared": self.n_cleared, "mean_response_s": round(mr, 2) if mr is not None else None,
            "last_response_s": round(self.last_resp, 2) if self.last_resp is not None else None,
            "holding": len(self.holding), "max_holding": self.max_holding, "near_misses": self.n_near_miss,
            "diverted": self.n_diverted, "arrived": self.n_arrived, "first_turn_ok": self.n_first_ok,
            "first_turn_trials": self.n_first_trials, "control": self.control_mode(),
            "calling": self.planes[self.calling].flight if self.calling is not None else None,
            "swivel_back_idle": self.n_face_idle, "swivel_back_plane_behind": self.n_face_back,
        }
