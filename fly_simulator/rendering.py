"""Offscreen rendering with a smooth tracking camera (no window code here).

We render with ``mujoco.Renderer`` (the same class FlyGym's ``flygym.Renderer``
wraps) but drive a free ``mujoco.MjvCamera`` ourselves instead of using a camera
fixed in the model. FlyGym's ``fly.add_tracking_camera`` (MuJoCo "track" mode)
follows the thorax rigidly, so every bob of the body during the tripod gait shakes
the image; low-pass filtering the look-at point and heading gives a smooth view.
"""

from __future__ import annotations

import math

import mujoco as mj
import numpy as np

from fly_simulator.config import CameraConfig, RenderConfig

CAMERA_MODES = ("follow", "side", "top")


class SmoothFollowCamera:
    """Exponentially smoothed follow camera on top of ``mujoco.MjvCamera``.

    MuJoCo free-camera convention: the camera looks along
    ``(cos(el)cos(az), cos(el)sin(az), sin(el))`` (degrees) at ``lookat`` from
    ``distance`` away.
    """

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.mode = cfg.mode
        self.cam = mj.MjvCamera()
        self.cam.type = mj.mjtCamera.mjCAMERA_FREE
        self.cam.distance = cfg.distance
        self._lookat: np.ndarray | None = None
        self._heading: float | None = None
        self._last_t: float | None = None

    def reset(self) -> None:
        self._lookat = self._heading = self._last_t = None

    def cycle_mode(self) -> str:
        self.mode = CAMERA_MODES[(CAMERA_MODES.index(self.mode) + 1) % len(CAMERA_MODES)]
        return self.mode

    def update(self, t: float, target: np.ndarray, heading_rad: float, *,
               ground_z: float = 0.0, tilt_deg: float | None = None) -> mj.MjvCamera:
        """Advance the smoothing to sim time ``t``. ``ground_z`` (terrain height under
        the fly) drives the zoom-out when the fly is launched high; ``tilt_deg``
        freezes the heading while the fly lies on its side/back."""
        c = self.cfg
        target = np.asarray(target, dtype=float)
        if self._lookat is None or self._last_t is None or t < self._last_t:
            self._lookat, self._heading = target.copy(), heading_rad
        else:
            dt = t - self._last_t
            # Adaptive smoothing: a shoved/launched fly moves several mm within
            # ~50 ms, and a fixed 0.15 s low-pass would lose it off-screen. The
            # time constant shrinks with the lag, and the lag is capped.
            lag = float(np.linalg.norm(target - self._lookat))
            tau = c.position_tau_s / (1.0 + (lag / max(c.catchup_distance, 1e-9)) ** 2)
            a_pos = 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0
            self._lookat += a_pos * (target - self._lookat)
            err = target - self._lookat
            n = float(np.linalg.norm(err))
            if c.max_lag > 0 and n > c.max_lag:
                self._lookat += err * (1.0 - c.max_lag / n)
            if tilt_deg is None or tilt_deg < c.freeze_heading_tilt_deg:
                a_head = 1.0 - math.exp(-dt / c.heading_tau_s)
                # wrap the heading difference to (-pi, pi] before filtering
                dh = (heading_rad - self._heading + math.pi) % (2 * math.pi) - math.pi
                self._heading += a_head * dh
        self._last_t = t
        # Zoom out (smoothly, via the smoothed look-at height) when high up.
        height = float(self._lookat[2]) - float(ground_z)
        extra = max(0.0, height - c.zoom_above) * c.zoom_per_mm
        distance = min(c.distance + extra, max(c.max_distance, c.distance))

        heading_deg = math.degrees(self._heading)
        self.cam.lookat[:] = self._lookat
        self.cam.distance = distance
        if self.mode == "follow":
            self.cam.azimuth = heading_deg + self.cfg.follow_azimuth_offset
            self.cam.elevation = self.cfg.elevation
        elif self.mode == "side":
            # looking along -y of the fly frame, i.e. the camera sits on the fly's left
            self.cam.azimuth = heading_deg - 90.0
            self.cam.elevation = -8.0
        else:  # top
            self.cam.azimuth = heading_deg
            self.cam.elevation = -85.0
            self.cam.distance = distance * 1.3
        return self.cam


class FrameRenderer:
    """Renders RGB frames of the sim from a ``SmoothFollowCamera``."""

    def __init__(self, model: mj.MjModel, render_cfg: RenderConfig, cam_cfg: CameraConfig):
        # The offscreen buffer is capped by model.vis.global_.offwidth/offheight
        # (2048 in FlyGym's mujoco_globals.yaml).
        self.renderer = mj.Renderer(model, render_cfg.height, render_cfg.width)
        self.camera = SmoothFollowCamera(cam_cfg)
        # Model cameras have their own fovy; the free camera uses the global one.
        model.vis.global_.fovy = cam_cfg.fovy
        self.scene_option = mj.MjvOption()
        self.set_reflections(render_cfg.reflections)
        # optional screen-space post-process ``post_process(frame, t) -> frame``
        # (a job's edit effects); ``post_clock()`` gives its t (read in update_scene,
        # i.e. under the physics lock in threaded mode)
        self.post_process = None
        self.post_clock = None
        self._post_t = 0.0

    def set_reflections(self, on: bool) -> None:
        """Floor reflections on/off (a scene render flag; survives update_scene)."""
        self.renderer.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = int(bool(on))

    def render(self, data: mj.MjData, t: float, target: np.ndarray, heading: float,
               **cam_kw) -> np.ndarray:
        self.update_scene(data, t, target, heading, **cam_kw)
        return self.draw()

    # render() split in two for threaded use (see fly_simulator/physics_thread.py):
    # update_scene() reads MjData (hold the physics lock, ~0.1 ms); draw() only
    # uses the copied mjvScene, so physics can keep stepping meanwhile (~14 ms at
    # 960x640, MuJoCo releases the GIL while drawing).
    def update_scene(self, data: mj.MjData, t: float, target: np.ndarray, heading: float,
                     **cam_kw) -> None:
        """``cam_kw``: ``ground_z``, ``tilt_deg`` (see SmoothFollowCamera.update)."""
        cam = self.camera.update(t, target, heading, **cam_kw)
        self.renderer.update_scene(data, camera=cam, scene_option=self.scene_option)
        if self.post_process is not None:
            self._post_t = float(self.post_clock()) if self.post_clock is not None else float(t)

    def draw(self) -> np.ndarray:
        frame = self.renderer.render()  # (H, W, 3) uint8 RGB
        if self.post_process is not None:
            frame = self.post_process(frame, self._post_t)
        return frame

    def close(self) -> None:
        self.renderer.close()
