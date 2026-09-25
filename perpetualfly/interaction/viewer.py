"""OpenCV live window that shows offscreen-rendered frames and polls the keyboard.

Why OpenCV rather than ``mujoco.viewer.launch_passive``: on macOS the passive viewer
only works when the script is started with ``mjpython`` (plain ``python`` raises
RuntimeError), and its built-in key bindings (space, many letters, ESC) collide
with the controls this project needs. An OpenCV window runs under plain ``python``
on the main thread, gives us every key press via ``cv2.waitKeyEx`` and lets us own
the (smoothed) camera completely. See docs/API_NOTES.md.
"""

from __future__ import annotations

import cv2
import numpy as np

from perpetualfly.interaction.keyboard import decode_key


class LiveViewer:
    def __init__(self, title: str = "PerpetualFly") -> None:
        self.title = title
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
        self._shown_once = False

    def show(self, rgb: np.ndarray, hud_lines: list[str] | None = None) -> None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if hud_lines:
            font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.55
            # Translucent dark panel behind white text: readable on the white sky and
            # the grey floor. (A thick black "outline" pass doesn't work: Hershey
            # glyph advance grows with thickness, so the outline drifts sideways.)
            w = max(cv2.getTextSize(line, font, scale, 1)[0][0] for line in hud_lines)
            h = 22 * len(hud_lines) + 10
            x1, y1 = min(bgr.shape[1], 20 + w), min(bgr.shape[0], h)
            panel = bgr[0:y1, 0:x1]
            panel[:] = (panel * 0.45).astype(bgr.dtype)
            for i, line in enumerate(hud_lines):
                cv2.putText(bgr, line, (10, 24 + 22 * i), font, scale, (255, 255, 255), 1,
                            cv2.LINE_AA)
        cv2.imshow(self.title, bgr)
        self._shown_once = True

    def poll_keys(self, wait_ms: int = 1) -> list[str]:
        """Pump the GUI event loop and return pressed keys (usually 0 or 1).

        ``waitKeyEx`` must be called regularly or the window freezes. On macOS a
        call takes ~10-15 ms even with ``wait_ms=1``.
        """
        keys = []
        code = cv2.waitKeyEx(wait_ms)
        while code != -1:
            name = decode_key(code)
            if name:
                keys.append(name)
            code = cv2.waitKeyEx(1) if len(keys) < 8 else -1
        return keys

    def is_open(self) -> bool:
        """False once the user closed the window with the title-bar button."""
        if not self._shown_once:
            return True
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.title)
            cv2.waitKey(1)
        except cv2.error:
            pass
