"""Obstacle-course mode (alongside endless mode). See docs/COURSE.md.

    from fly_simulator.course import install_course
    session = Session(cfg)                     # fly_simulator.app.Session
    course = install_course(session, "gauntlet")
    ...                                        # the app loop runs as usual
    course.hud_lines(); course.done; course.result()

``install_course`` swaps the session's terrain generator for the course layout
(same pooled geoms, no model rebuild; endless mode is untouched until then),
installs the race hooks and chains ``CourseRun.after_physics`` / ``finalize`` onto
``session.after_physics`` / ``session.close``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from fly_simulator.course.layout import CourseLayout, build_layout
from fly_simulator.course.leaderboard import controller_mode
from fly_simulator.course.race import CourseOptions, CourseRun
from fly_simulator.course.spec import COURSES_DIR, CourseSpec, SectionSpec, builtin_courses

if TYPE_CHECKING:
    from fly_simulator.app import Session
    from fly_simulator.config import AppConfig


def install_course(session: "Session", course: str | Path | CourseSpec,
                   cfg: "AppConfig | None" = None, *, options: CourseOptions | None = None,
                   stimulus_fn: Callable | None = None, reset: bool | None = None) -> CourseRun:
    """Turn ``session`` into an obstacle-course run and return the ``CourseRun``.

    ``course``: built-in name (``builtin_courses()``), a .json/.toml path or a
    ``CourseSpec``. ``cfg``: the AppConfig (default ``session.cfg``; used for the
    leaderboard's controller mode and runs dir). ``stimulus_fn(StimulusEvent)``:
    receives looming stimuli (default: ``session.brain.send`` if a brain is
    attached, else they are only logged). ``reset``: explicit reset to the start
    first (default: only if the fly is not at the spawn point). Threaded window
    mode: call under ``runner.locked()``.
    """
    spec = course if isinstance(course, CourseSpec) else CourseSpec.load(course)
    opts = options or CourseOptions()
    if cfg is not None and opts.controller_mode is None:
        opts.controller_mode = controller_mode(cfg)
    run = CourseRun(session, spec, build_layout(spec), opts, stimulus_fn=stimulus_fn)
    run.install()
    x, y, _ = session.sim.thorax_position()
    if reset or (reset is None and (abs(x) > 1.5 or abs(y) > 1.0)):  # thorax ~0.55 at spawn
        run._reset("course_start_reset")
    return run


__all__ = ["COURSES_DIR", "CourseLayout", "CourseOptions", "CourseRun", "CourseSpec",
           "SectionSpec", "build_layout", "builtin_courses", "controller_mode",
           "install_course"]
