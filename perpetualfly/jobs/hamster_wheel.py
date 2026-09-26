"""Hamster wheel: the fly runs inside a big wheel (hinge joint) and spins it with its
own legs. Forever.

The wheel (axle along world y, centred above the origin so the fly spawns standing
on the inside bottom) is a ring of ``n_slats`` box slats in alternating bright
colours (rotation is easy to see), with slippery inner lips on both sides that keep
the fly on the running surface, a decorative back disc with spokes, an axle and a
stand. Only the fly touches the wheel (contact kind "fly"). The hinge has a little
damping; the wheel turns only because the fly's feet push the slats backward.

Behaviour: heading hold along the wheel tangent (+x) with a small correction toward
the centre line (y = 0). Counters: revolutions, distance run (surface travel), top
speed. The fall detector's stall rule measures progress relative to the wheel
surface (``progress_xy``), so running on the spot is not "stuck".
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import mujoco as mj
import numpy as np

from perpetualfly.jobs.base import CameraPreset, EternalJob, JobConfig
from perpetualfly.jobs.geometry import add_box, quat_axis_angle
from perpetualfly.jobs.registry import register_job

P = "wheel/"
SLAT_A = (1.00, 0.78, 0.10, 1.0)  # sunny yellow
SLAT_B = (1.00, 0.45, 0.10, 1.0)  # orange
LIP = (0.15, 0.55, 0.95, 1.0)  # blue rims
LIP_NEAR = (0.15, 0.55, 0.95, 0.3)  # camera side: translucent, the fly stays visible
DISC = (0.20, 0.62, 1.00, 1.0)
STEEL = (0.75, 0.75, 0.78, 1.0)


@dataclass
class HamsterWheelConfig(JobConfig):
    inner_radius: float = 7.0  # running surface radius (14 mm wheel)
    width: float = 5.0  # running surface width between the lips (mm)
    n_slats: int = 60
    slat_thickness: float = 0.3
    bottom_height: float = 1.0  # running surface height at the bottom (mm above the floor)
    wheel_mass: float = 4e-3  # g (4 mg, the fly is 1.02 mg)
    damping: float = 2.0  # hinge damping (uN*mm*s/rad); with ~2 rad/s the fly sits ~3 deg forward
    armature: float = 0.0
    slat_friction: float = 1.0
    lip_height: float = 0.8  # inner lips, radially inward from the running surface
    lip_friction: float = 0.1  # slippery: sticky tarsi can't climb out
    # behaviour
    centre_gain: float = 0.35  # heading correction per mm of lateral offset (rad/mm)
    centre_max_deg: float = 20.0
    speed: float = 1.0
    speed_window_s: float = 1.0  # top speed = max over windows this long


@register_job
class HamsterWheelJob(EternalJob):
    name = "hamster_wheel"
    title = "HAMSTER WHEEL FLY"
    tagline = "the wheel is the destination"
    work_label = "revolutions"
    work_format = "{:.1f}"
    config_cls = HamsterWheelConfig
    required_names = (P + "wheel",)

    cfg: HamsterWheelConfig

    @property
    def centre_z(self) -> float:
        return self.cfg.bottom_height + self.cfg.inner_radius

    def configure_app(self, app_cfg) -> None:
        super().configure_app(app_cfg)
        # stand on the inside bottom of the wheel (FlyGym's flat-ground spawn is 0.8)
        app_cfg.fly.spawn_height = self.cfg.bottom_height + 0.8
        app_cfg.controller.target_heading_deg = 0.0

    def ground_height(self, x: float, y: float) -> float:
        R = self.cfg.inner_radius
        if abs(x) >= R * 0.95:
            return self.cfg.bottom_height
        return self.centre_z - math.sqrt(R * R - x * x)

    def extension(self, world) -> None:
        c = self.cfg
        wb = world.mjcf_root.worldbody
        R, t, hw = c.inner_radius, c.slat_thickness, c.width / 2
        zc = self.centre_z
        wheel = wb.add_body(name=P + "wheel", pos=(0.0, 0.0, zc))
        wheel.add_joint(name=P + "hinge", type=mj.mjtJoint.mjJNT_HINGE, axis=(0, 1, 0),
                        damping=c.damping, armature=c.armature)
        n = c.n_slats
        dth = 2 * math.pi / n
        slat_len = (R + t) * dth * 1.02 / 2  # half length, slight overlap: no gaps
        m_slat = c.wheel_mass * 0.8 / n
        for i in range(n):
            th = i * dth  # angle from the bottom, about +y
            # position of the slat centre: bottom is (0, 0, -R); rotate about y by th
            r = R + t / 2
            pos = (-r * math.sin(th), 0.0, -r * math.cos(th))
            q = quat_axis_angle((0, 1, 0), th)
            add_box(wheel, f"{P}slat{i}", (slat_len, hw + 0.3, t / 2), pos, quat=q,
                    rgba=SLAT_A if (i // 3) % 2 == 0 else SLAT_B, collide="fly",
                    friction=c.slat_friction, mass=m_slat)
            # lips on both sides (radially inward)
            rl = R - c.lip_height / 2
            lpos = (-rl * math.sin(th), 0.0, -rl * math.cos(th))
            for side, sgn in (("l", 1.0), ("r", -1.0)):
                add_box(wheel, f"{P}lip{i}{side}", (slat_len, 0.12, c.lip_height / 2),
                        (lpos[0], sgn * (hw + 0.12), lpos[2]), quat=q,
                        rgba=LIP if sgn > 0 else LIP_NEAR,
                        collide="fly", friction=c.lip_friction,
                        mass=c.wheel_mass * 0.1 / n)
        # decorative back disc + spokes + hub (far side, +y), no contacts
        wheel.add_geom(name=P + "disc", type=mj.mjtGeom.mjGEOM_CYLINDER,
                       size=(R + t, 0.05, 0), pos=(0, hw + 0.35, 0),
                       quat=quat_axis_angle((1, 0, 0), math.pi / 2), rgba=DISC,
                       contype=0, conaffinity=0, group=1, mass=0.0)
        for k in range(6):
            a = k * math.pi / 3
            add_box(wheel, f"{P}spoke{k}", (R / 2, 0.06, 0.18),
                    (-(R / 2) * math.sin(a), hw + 0.28, -(R / 2) * math.cos(a)),
                    quat=quat_axis_angle((0, 1, 0), a + math.pi / 2), rgba=(1, 1, 1, 1),
                    collide="visual", mass=0.0)
        wheel.add_geom(name=P + "hub", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.8, 0.3, 0),
                       pos=(0, hw + 0.4, 0), quat=quat_axis_angle((1, 0, 0), math.pi / 2),
                       rgba=STEEL, contype=0, conaffinity=0, group=1, mass=0.0)
        # axle + stand (static decoration behind the wheel)
        wb.add_geom(name=P + "axle", type=mj.mjtGeom.mjGEOM_CYLINDER, size=(0.25, 1.2, 0),
                    pos=(0, hw + 1.4, zc), quat=quat_axis_angle((1, 0, 0), math.pi / 2),
                    rgba=STEEL, contype=0, conaffinity=0, group=1)
        for sx in (-1.0, 1.0):
            leg_len = math.hypot(zc, 4.0)
            add_box(wb, f"{P}stand{int(sx > 0)}", (0.3, 0.3, leg_len / 2),
                    (sx * 2.0, hw + 2.3, zc / 2),
                    quat=quat_axis_angle((0, 1, 0), -sx * math.atan2(4.0, zc)),
                    rgba=STEEL, collide="visual")
        add_box(wb, P + "base", (5.5, 1.2, 0.2), (0, hw + 2.3, 0.2), rgba=(0.3, 0.3, 0.35, 1),
                collide="visual")

    # ------------------------------------------------------------ attach / state
    def on_attach(self) -> None:
        m = self.sim.model
        j = m.joint(P + "hinge").id
        self.qadr = int(m.jnt_qposadr[j])
        self.vadr = int(m.jnt_dofadr[j])
        self.revolutions = 0.0
        self.distance_mm = 0.0
        self.top_speed = 0.0
        self._speed_hist: deque = deque(maxlen=int(self.cfg.speed_window_s / 0.05) + 1)
        self._t_last_sample = -1.0
        self.on_reset()
        self.state = "running"

    def on_reset(self) -> None:
        self._theta_last = float(self.sim.data.qpos[self.qadr])
        self._theta_accum = 0.0  # surface travel since the last reset (rad)
        self._speed_hist.clear()

    def wheel_angle(self) -> float:
        return float(self.sim.data.qpos[self.qadr])

    def surface_speed(self) -> float:
        """mm/s of the running surface (positive = the fly runs forward)."""
        return float(self.sim.data.qvel[self.vadr]) * self.cfg.inner_radius

    def progress_xy(self, x: float, y: float):
        return (x + self._theta_accum * self.cfg.inner_radius, y)

    # ------------------------------------------------------------ behaviour
    def update(self) -> None:
        c = self.cfg
        th = self.wheel_angle()
        if not math.isfinite(th):
            return
        d = th - self._theta_last
        self._theta_last = th
        if abs(d) < 1.0:  # (a reset re-baselines; guard against jumps)
            self._theta_accum += d
            if d > 0:
                self.revolutions += d / (2 * math.pi)
                self.distance_mm += d * c.inner_radius
                self.add_work(d / (2 * math.pi))
                self.work = self.revolutions
        t = self.sim.time
        if t - self._t_last_sample >= 0.05 or t < self._t_last_sample:
            self._t_last_sample = t
            self._speed_hist.append((t, self._theta_accum))
            (t0, a0), (t1, a1) = self._speed_hist[0], self._speed_hist[-1]
            if t1 - t0 >= c.speed_window_s * 0.9:
                self.top_speed = max(self.top_speed, (a1 - a0) / (t1 - t0) * c.inner_radius)
        y = float(self.sim.data.xpos[self.sim.thorax_body_id, 1])
        lim = math.radians(c.centre_max_deg)
        self.steering.set(min(max(-c.centre_gain * y, -lim), lim), c.speed)
        self.state = "down" if self.fly_down() else "running"

    # ------------------------------------------------------------ view / HUD
    def camera_target(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.centre_z - 0.3 * self.cfg.inner_radius])

    def camera_preset(self) -> CameraPreset:
        # from the open side (-y), a little from the front and above
        return CameraPreset(azimuth=102.0, elevation=-24.0, distance=26.0, tau_s=0.3)

    def job_hud_lines(self) -> list[str]:
        return [f"distance {self.distance_mm / 1000:.3f} m   speed {self.surface_speed():5.1f} mm/s"
                f"   top {self.top_speed:5.1f} mm/s"]

    def job_stats(self) -> dict:
        return {"revolutions": self.revolutions, "distance_m": self.distance_mm / 1000,
                "top_speed_mm_s": self.top_speed, "unstuck": self.n_unstuck}
