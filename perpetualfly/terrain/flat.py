from __future__ import annotations

import numpy as np
from flygym.compose import FlatGroundWorld


def build_flat_world(half_size: float = 1000.0, checker_size_mm: float = 2.0) -> FlatGroundWorld:
    """FlyGym's flat ground: a single plane geom named ``ground_plane``.

    ``half_size`` (mm) only limits the *rendered* area: MuJoCo planes are infinite for
    collision. At ~14 mm/s the fly needs ~70 simulated seconds to walk off the drawn
    1000 mm checkerboard, so ``GroundRecentering`` shifts the plane along with it.

    FlyGym's material uses ``texrepeat=250`` over the plane, i.e. 4 mm squares for the
    default size, which is larger than the ~2.5 mm fly. We rescale the (visual-only)
    texture repeat so squares are ``checker_size_mm`` wide; one texture repeat of the
    builtin checker contains 2x2 squares.
    """
    world = FlatGroundWorld(half_size=half_size)
    repeat = (2 * half_size) / (2 * checker_size_mm)
    world.mjcf_root.material("grid").texrepeat = [repeat, repeat]
    return world


class GroundRecentering:
    """Post-step hook that keeps the drawn flat plane centred under the fly.

    A MuJoCo plane is infinite for collision and its contact depends only on the
    plane's normal and height, so shifting it within its own plane changes nothing
    physically. The shift is snapped to whole texture periods (2 checker squares),
    so the checkerboard does not visibly jump. Without this the fly would walk off
    the rendered area after ~70 simulated seconds.
    """

    def __init__(self, geom_name: str, half_size: float, checker_size_mm: float,
                 every_steps: int = 1000) -> None:
        self.geom_name = geom_name
        self.period = 2 * checker_size_mm
        self.threshold = 0.25 * half_size
        self.every_steps = every_steps
        self._geom_id: int | None = None

    def __call__(self, sim) -> None:
        if sim.step_count % self.every_steps:
            return
        import mujoco as mj

        m = sim.model
        if self._geom_id is None:
            self._geom_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, self.geom_name)
            # The plane is compiled at the world origin, so MuJoCo marks it
            # geom_sameframe = 1 and mj_kinematics copies the (world) body frame
            # instead of reading geom_pos: moving geom_pos alone did nothing, and the
            # fly walked off the drawn checkerboard after ~1 m (collision was fine:
            # the plane is infinite). Clearing the flag makes geom_pos count.
            m.geom_sameframe[self._geom_id] = 0
        center = m.geom_pos[self._geom_id, :2]
        fly_xy = sim.data.xpos[sim.thorax_body_id, :2]
        if np.any(np.abs(fly_xy - center) > self.threshold):
            new = np.round(fly_xy / self.period) * self.period
            m.geom_pos[self._geom_id, :2] = new
            sim.data.geom_xpos[self._geom_id, :2] = new  # visible before the next step
