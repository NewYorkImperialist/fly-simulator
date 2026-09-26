"""Compound eyes: FlyGym's eye cameras -> 721 ommatidia per eye (docs/VISION.md).

"Real vision" part 1. FlyGym 2.1's ``fly.add_vision()`` adds one camera per eye
(``l_eye_cam`` / ``r_eye_cam``, 157 deg field of view, fisheye-corrected) and
``Simulation.get_ommatidia_readouts(fly)`` samples them onto the 721-ommatidium
hexagonal lattice of each eye (yellow / pale channels; NeuroMechFly v2, Wang-Chen
et al. 2024). The fly's own head / thorax / eyes / antennae / front coxae are hidden
from the eye cameras (FlyGym's vision.yaml), so the fly sees the world, not itself.

* ``make_eyes_fly_factory``: a ``fly_factory`` for ``fly_simulator.simulation.
  Simulation`` that builds the same fly the simulation would (standard locomotion
  fly, or the full-body action fly with ``fly.extra_joints``) and calls
  ``add_vision()`` on it before ``add_fly``. Adding the cameras adds two massless
  child bodies of the eyes (no joints, no collisions); the walking dynamics are
  unchanged, but the model is not bit-identical, so eyes are off by default.
* ``CompoundEyes``: a physics post-step hook that samples the ommatidia every
  ``1 / rate_hz`` sim seconds (rendering costs ~15-20 ms of wall time per sample,
  so it is not done every 0.1 ms physics step), keeps the latest frame and calls
  its listeners ``fn(t, frame)`` with ``frame`` (2, 721) luminance in [0, 1]
  (left, right; per ommatidium the channel of its type, i.e. ``readouts.max(-1)``,
  as in FlyGym v1's flyvis example).

Props in geom group 1 (the swatter, the whip, ``LoomingObject``) are made visible to
the eye renderer (``EyesConfig.see_group1``): FlyGym hides group 1 only because its
optional eye markers live there.

Rendering uses its own offscreen MuJoCo renderer (created lazily by FlyGym on the
first sample, in the thread that samples: the physics thread in the app).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

N_OMMATIDIA = 721


@dataclass
class EyesConfig:
    enabled: bool = False
    rate_hz: float = 100.0  # ommatidia samples per sim second
    history: int = 64  # frames kept in CompoundEyes.frames (figures / diagnostics)
    # show geom group 1 (props: swatter, whip, looming objects) to the eyes; FlyGym
    # hides it by default (see CompoundEyes.sample). Needs draw_markers=False.
    see_group1: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def make_eyes_fly_factory(base: Callable | None = None, *, extra_joints: bool | None = None,
                          draw_markers: bool = False) -> Callable:
    """``fly_factory(fly_cfg) -> NeuroMechFly`` with compound-eye cameras.

    ``base``: another fly factory to wrap (e.g. ``make_action_fly_factory()``); None =
    what ``Simulation`` would build for ``fly_cfg`` (``extra_joints`` overrides
    ``fly_cfg.extra_joints``)."""

    def factory(fc):
        if base is not None:
            fly = base(fc)
        else:
            ej = getattr(fc, "extra_joints", False) if extra_joints is None else extra_joints
            if ej:
                from fly_simulator.actions.body import make_action_fly_factory

                fly = make_action_fly_factory(wings=True, proboscis=True)(fc)
            else:
                from flygym_demo.complex_terrain import make_locomotion_fly

                fly = make_locomotion_fly(name=fc.name, add_adhesion=fc.adhesion,
                                          colorize=fc.colorize)
        fly.add_vision(draw_sensor_markers=draw_markers)
        return fly

    return factory


def has_eyes(sim) -> bool:
    """True if the simulation's fly has FlyGym eye cameras."""
    ids = getattr(sim.fg, "_intern_eye_camera_ids_by_fly", {})
    v = ids.get(sim.fly_name)
    return v is not None and len(v) > 0


def luminance(readouts: np.ndarray) -> np.ndarray:
    """(2, 721, 2) yellow / pale readouts -> (2, 721) luminance (the non-zero
    channel of each ommatidium)."""
    return np.asarray(readouts, dtype=np.float32).max(axis=-1)


class CompoundEyes:
    """Post-step hook sampling both compound eyes at ``cfg.rate_hz`` (sim time)."""

    def __init__(self, sim, cfg: EyesConfig | None = None) -> None:
        if not has_eyes(sim):
            raise RuntimeError("the fly has no eye cameras: build the Simulation with "
                               "fly_factory=make_eyes_fly_factory() (docs/VISION.md)")
        self.sim = sim
        self.cfg = cfg or EyesConfig(enabled=True)
        self.listeners: list[Callable[[float, np.ndarray], None]] = []
        self.frame: np.ndarray | None = None  # (2, 721) latest luminance
        self.readouts: np.ndarray | None = None  # (2, 721, 2) latest raw readouts
        self.t_frame: float | None = None
        self.frames: deque = deque(maxlen=max(1, int(self.cfg.history)))  # (t, frame)
        self.n_samples = 0
        self.wall_s = 0.0  # wall time spent rendering + sampling
        self._next_t: float | None = None
        self._attached = False
        self.enabled = True

    @property
    def period(self) -> float:
        return 1.0 / max(float(self.cfg.rate_hz), 1e-6)

    def attach(self) -> "CompoundEyes":
        if not self._attached:
            self.sim.post_step_hooks.append(self)
            self.sim.reset_hooks.append(self._on_reset)
            self._attached = True
        return self

    def detach(self) -> None:
        if self._attached:
            self.sim.post_step_hooks.remove(self)
            self.sim.reset_hooks.remove(self._on_reset)
            self._attached = False

    def _on_reset(self, sim) -> None:
        self._next_t = None
        self.frames.clear()

    def __call__(self, sim) -> None:
        if not self.enabled:
            return
        t = sim.time
        if self._next_t is not None and t < self._next_t - 1e-9:
            return
        self.sample()

    def sample(self) -> np.ndarray:
        """Render both eyes now; returns (2, 721) luminance and notifies listeners."""
        t0 = time.perf_counter()
        sim = self.sim
        fg = sim.fg
        if fg.eye_renderer is None:
            fg.get_raw_vision(sim.fly_name)  # FlyGym creates the renderer lazily
        if self.cfg.see_group1 and fg.eye_renderer_scene_option is not None:
            # FlyGym hides geom group 1 from the eyes only to hide its eye markers
            # (group 1 with draw_sensor_markers=True, else group 4). Our props (the
            # swatter, the whip, looming test objects) are visual geoms in group 1.
            fg.eye_renderer_scene_option.geomgroup[1] = 1
        r = fg.get_ommatidia_readouts(sim.fly_name)
        self.wall_s += time.perf_counter() - t0
        t = sim.time
        self.readouts = r
        self.frame = luminance(r)
        self.t_frame = t
        self.frames.append((t, self.frame))
        self.n_samples += 1
        # next sample one period after this one (never drifts: anchored on the grid)
        p = self.period
        self._next_t = t + p if self._next_t is None else max(self._next_t + p, t + 0.5 * p)
        for fn in list(self.listeners):
            fn(t, self.frame)
        return self.frame

    def ms_per_sample(self) -> float:
        return 1e3 * self.wall_s / max(self.n_samples, 1)

    def human_readable(self, frame: np.ndarray | None = None) -> np.ndarray:
        """(2, H, W) hexagonal images of both eyes (FlyGym's retina tools)."""
        from flygym.vision.retina import Retina

        ret = self.sim.fg.retina or Retina()
        f = self.frame if frame is None else frame
        return np.stack([ret.hex_pxls_to_human_readable(f[i][:, None])[..., 0]
                         for i in range(2)])
