"""Looming test object: a dark sphere on a mocap body (docs/VISION.md, "Real vision").

A classic looming stimulus in the physics scene: a matt black sphere (radius
``radius_mm``, visual only: no contacts, geom group 1) that is parked far away and,
on ``start(kind, ...)``, appears at a start point fixed relative to the fly's head
and heading, waits ``appear_s`` (so appearance and motion onset are separate
events), then moves in a straight line at constant speed:

* ``approach``: from ``dist_mm`` toward the head at ``speed = radius / l_over_v``
  (the looming parameter l/v of von Reyn et al. 2017), and stops ``stop_mm`` from
  the head (it never touches the fly). With ``track`` (default) it stays at a fixed
  bearing (azimuth / elevation relative to the fly's current head position and
  heading, like a closed-loop looming stimulus in a virtual-reality rig), so a
  walking or turning fly neither walks out from under it nor turns it out of the
  eye's field of view; the distance closes at ``speed``;
* ``recede``: from ``dist_mm`` (near) straight away from the head;
* ``pass``: along a line parallel to the fly's heading at lateral offset
  ``dist_mm``, from ``pass_mm`` ahead to ``pass_mm`` behind (a lateral moving
  object, no expansion).

``azimuth_deg`` (+ = the fly's left) / ``elevation_deg`` give the direction of the
start point from the head. ``geometric_source()`` gives the same object to the
calculated looming sense (``fly_simulator.vision.looming``), for comparisons.

World extension like the swatter: the body must be in the model before add_fly::

    obj = LoomingObject()
    session = Session(cfg, world_extensions=[obj.extension], ...)
    obj.attach(session.sim)
    obj.start("approach", azimuth_deg=90, l_over_v=0.04)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco as mj
import numpy as np

PREFIX = "loomobj/"
KINDS = ("approach", "recede", "pass")


@dataclass
class LoomingObjectConfig:
    radius_mm: float = 1.0
    rgba: tuple[float, float, float, float] = (0.03, 0.03, 0.03, 1.0)
    park: tuple[float, float, float] = (0.0, 0.0, -500.0)  # far below the ground
    dist_mm: float = 30.0
    stop_mm: float = 2.0  # approach stops this far from the head (centre distance)
    appear_s: float = 1.0
    # fade the sphere in (alpha 0 -> 1) over this long after it appears, so its
    # appearance is not itself a sudden dark "looming" flash for the eyes
    fade_s: float = 0.8  # (and out again over the same time before it parks)
    pass_mm: float = 25.0
    pass_speed_mm_s: float = 40.0
    recede_speed_mm_s: float = 25.0
    hold_s: float = 0.3  # stays at the end point, then parks
    track: bool = True  # approach: fixed bearing relative to the moving fly


class LoomingObject:
    def __init__(self, cfg: LoomingObjectConfig | None = None) -> None:
        self.cfg = cfg or LoomingObjectConfig()
        self.sim = None
        self.phase = "idle"  # idle | appear | move | hold | leave
        self.onset_t: float | None = None  # sim time motion starts
        self.end_t: float | None = None  # sim time motion ends (collision time for approach)
        self.kind: str | None = None
        self._p0 = self._p1 = None
        self._t_phase = 0.0
        self._attached = False
        self.onset_listeners: list = []  # fn(sim_time) when motion starts

    def extension(self, world) -> None:
        c = self.cfg
        spec = world.mjcf_root
        b = spec.worldbody.add_body(name=PREFIX + "body", pos=c.park, mocap=True)
        b.add_geom(name=PREFIX + "sphere", type=mj.mjtGeom.mjGEOM_SPHERE,
                   size=(c.radius_mm, 0, 0), rgba=c.rgba, contype=0, conaffinity=0,
                   group=1, mass=0)

    def attach(self, sim) -> "LoomingObject":
        m = sim.model
        bid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, PREFIX + "body")
        if bid < 0:
            raise RuntimeError("the looming object is not in the model: build with "
                               "world_extensions=[obj.extension]")
        self.sim = sim
        self.mocap_id = int(m.body_mocapid[bid])
        self.geom_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, PREFIX + "sphere")
        self._rgba = np.array(self.cfg.rgba, dtype=float)
        if not self._attached:
            sim.pre_step_hooks.append(self)
            sim.reset_hooks.append(self._on_reset)
            self._attached = True
        self._park()
        return self

    # ------------------------------------------------------------------ control
    def head_position(self) -> np.ndarray:
        m, d = self.sim.model, self.sim.data
        hid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"{self.sim.fly_name}/c_head")
        return d.xpos[hid].copy() if hid >= 0 else self.sim.thorax_position().copy()

    def start(self, kind: str = "approach", *, azimuth_deg: float = 90.0,
              elevation_deg: float = 10.0, l_over_v: float = 0.04,
              dist_mm: float | None = None, speed_mm_s: float | None = None) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        c = self.cfg
        h = self.head_position()
        yaw = self.sim.heading()
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
        u = math.cos(el) * (math.cos(az) * fwd + math.sin(az) * left) + math.sin(el) * np.array([0, 0, 1.0])
        D = c.dist_mm if dist_mm is None else dist_mm
        if kind == "approach":
            v = c.radius_mm / l_over_v if speed_mm_s is None else speed_mm_s
            self._p0, self._p1 = h + D * u, h + c.stop_mm * u
        elif kind == "recede":
            v = c.recede_speed_mm_s if speed_mm_s is None else speed_mm_s
            self._p0, self._p1 = h + D * u, h + (D + 30.0) * u
        else:  # pass: lateral line at distance D on the azimuth side, front -> back
            v = c.pass_speed_mm_s if speed_mm_s is None else speed_mm_s
            side = left if azimuth_deg >= 0 else -left
            base = h + D * side + math.tan(el) * D * np.array([0, 0, 1.0])
            self._p0, self._p1 = base + c.pass_mm * fwd, base - c.pass_mm * fwd
        self.speed = float(v)
        self._azel, self._D0 = (az, el), float(D)
        self.kind = kind
        self.duration = float(np.linalg.norm(self._p1 - self._p0) / max(v, 1e-9))
        t = self.sim.time
        self.onset_t = t + c.appear_s
        self.end_t = self.onset_t + self.duration
        # for an approach, the time it would reach the eye (centre) at constant speed
        self.collision_t = self.onset_t + D / v if kind == "approach" else None
        self.phase = "appear"
        self._t_phase = t
        self._set(self._p0)
        self._alpha(0.0 if self.cfg.fade_s > 0 else 1.0)

    def _bearing(self) -> np.ndarray:
        yaw = self.sim.heading()
        az, el = self._azel
        fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        return (math.cos(el) * (math.cos(az) * fwd + math.sin(az) * left)
                + math.sin(el) * np.array([0.0, 0.0, 1.0]))

    @property
    def busy(self) -> bool:
        return self.phase != "idle"

    def position(self) -> np.ndarray:
        return self.sim.data.mocap_pos[self.mocap_id].copy()

    def _set(self, p) -> None:
        self.sim.data.mocap_pos[self.mocap_id] = p

    def _park(self) -> None:
        self._set(np.array(self.cfg.park, dtype=float))
        self.phase = "idle"
        self._alpha(1.0)

    def _alpha(self, a: float) -> None:
        # rendered colour blends toward the background: fade = lighten + transparent
        rgba = self._rgba.copy()
        rgba[3] = self._rgba[3] * float(np.clip(a, 0.0, 1.0))
        self.sim.model.geom_rgba[self.geom_id] = rgba

    def _on_reset(self, sim) -> None:
        self._park()

    def __call__(self, sim) -> None:
        if self.phase == "idle":
            return
        t = sim.time
        if self.phase == "appear":
            f = self.cfg.fade_s
            self._alpha(1.0 if f <= 0 else (t - self._t_phase) / f)
            if t >= self.onset_t:
                self._alpha(1.0)
                self.phase = "move"
                for fn in list(self.onset_listeners):
                    fn(t)
            else:
                return
        if self.phase == "move":
            s = min(1.0, (t - self.onset_t) / max(self.duration, 1e-9))
            if self.kind == "approach" and self.cfg.track:
                # distance to the current head closes at constant speed along the
                D = self._D0 - s * (self._D0 - self.cfg.stop_mm)
                self._set(self.head_position() + D * self._bearing())
            else:
                self._set(self._p0 + s * (self._p1 - self._p0))
            if s >= 1.0:
                self.phase = "hold"
                self._t_phase = t
        elif self.phase == "hold":
            if self.kind == "approach" and self.cfg.track:  # keep the bearing while held
                self._set(self.head_position() + self.cfg.stop_mm * self._bearing())
            if t - self._t_phase >= self.cfg.hold_s:
                self.phase = "leave"  # fade out, then park
                self._t_phase = t
        elif self.phase == "leave":
            f = self.cfg.fade_s
            if self.kind == "approach" and self.cfg.track:
                self._set(self.head_position() + self.cfg.stop_mm * self._bearing())
            if f <= 0 or t - self._t_phase >= f:
                self._park()
            else:
                self._alpha(1.0 - (t - self._t_phase) / f)

    # ------------------------------------------------------------------ geometric sense
    def geometric_source(self, response=None):
        """The sphere as a ``VisualSource`` for ``LoomingVision`` (1 ms updates while
        moving). Response default: the swatter's (slow, large objects). The geometric
        sense cannot see a fade-in: it only gets the sphere from motion onset on;
        the source's filter is re-initialised at onset (``onset_listeners``) so the
        appearance is not read as expansion."""
        from fly_simulator.interaction.swatter import swatter_response
        from fly_simulator.vision.looming import VisualSource

        def shapes():
            if self.phase not in ("move", "hold"):
                return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0)
            p = self.position()[None, :]
            return p, np.array([[0, 0, 1.0]]), np.zeros(1), np.array([self.cfg.radius_mm])

        def period():
            return 1e-3 if self.phase in ("move", "hold") else None

        return VisualSource("loom_object", shapes=shapes, period=period,
                            response=response or swatter_response())
