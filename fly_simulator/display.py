"""Display-size helpers shared by the fly window and the brain window (no cv2 import).

OpenCV's Cocoa backend (macOS) maps one image pixel to one *physical* pixel, so on a
Retina display (backing scale 2) a 960x640 frame shows at 480x320 pt with 6-pt text.
``auto_display_scale`` returns the factor to upscale the image by before ``imshow``.
"""

from __future__ import annotations

import os
import sys


def screen_info() -> tuple[float, int, int] | None:
    """(backing scale, width pt, height pt) of the main display on macOS, else None."""
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        cg = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        cg.CGMainDisplayID.restype = ctypes.c_uint32
        cg.CGDisplayCopyDisplayMode.restype = ctypes.c_void_p
        cg.CGDisplayCopyDisplayMode.argtypes = [ctypes.c_uint32]
        for f in ("CGDisplayModeGetPixelWidth", "CGDisplayModeGetWidth",
                  "CGDisplayModeGetHeight"):
            getattr(cg, f).restype = ctypes.c_size_t
            getattr(cg, f).argtypes = [ctypes.c_void_p]
        cg.CGDisplayModeRelease.argtypes = [ctypes.c_void_p]
        m = cg.CGDisplayCopyDisplayMode(cg.CGMainDisplayID())
        if not m:
            return None
        px, w, h = (cg.CGDisplayModeGetPixelWidth(m), cg.CGDisplayModeGetWidth(m),
                    cg.CGDisplayModeGetHeight(m))
        cg.CGDisplayModeRelease(m)
        return (px / w if w else 1.0), int(w), int(h)
    except Exception:
        return None


def auto_display_scale(size: tuple[int, int], max_frac_w: float = 0.72,
                       max_frac_h: float = 0.85, env_var: str | None = None,
                       info: tuple[float, int, int] | None = None) -> float:
    """Image upscale for ``imshow`` so a ``size`` = (w, h) px frame is legible.

    Upscales by the display's backing scale, capped so the window uses at most
    ``max_frac_w`` / ``max_frac_h`` of the screen (in points). 1.0 off macOS.
    ``env_var`` (if set in the environment) overrides the result, e.g.
    ``FLY_SIMULATOR_FLY_SCALE=1.5``. ``info`` = ``screen_info()`` result (tests).
    """
    env = os.environ.get(env_var) if env_var else None
    if env:
        try:
            return max(0.25, float(env))
        except ValueError:
            pass
    info = screen_info() if info is None else info
    if info is None:
        return 1.0
    backing, wpt, hpt = info
    fit = min(1.0, max_frac_w * wpt / size[0], max_frac_h * hpt / size[1])
    return float(max(0.5, backing * fit))
