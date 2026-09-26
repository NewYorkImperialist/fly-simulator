"""OpenCV live window that shows offscreen-rendered frames and polls the keyboard.

Why OpenCV rather than ``mujoco.viewer.launch_passive``: on macOS the passive viewer
only works when the script is started with ``mjpython`` (plain ``python`` raises
RuntimeError), and its built-in key bindings (space, many letters, ESC) collide
with the controls this project needs. An OpenCV window runs under plain ``python``
on the main thread, gives us every key press via ``cv2.waitKeyEx`` and lets us own
the (smoothed) camera completely. See docs/API_NOTES.md.

Retina: OpenCV's Cocoa backend shows one image pixel per *physical* pixel, so the
frame is upscaled by ``display_scale`` (auto: the display's backing scale, see
``perpetualfly.display``; env ``PERPETUALFLY_FLY_SCALE`` overrides) and the HUD is
drawn *after* the upscale with a proportionally larger font, so text stays sharp.
"""

from __future__ import annotations

import cv2
import numpy as np

from perpetualfly.display import auto_display_scale
from perpetualfly.interaction.keyboard import decode_key

FLY_SCALE_ENV = "PERPETUALFLY_FLY_SCALE"
# (group title, [(keys, description), ...]) as drawn by draw_help_overlay
HelpGroups = list[tuple[str, list[tuple[str, str]]]]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_REC_RED = (40, 40, 230)  # BGR
_GREY = (115, 115, 115)
# help rows whose description starts with this are drawn greyed out (unavailable)
UNAVAILABLE_MARK = "~"


def _darken(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, k: float) -> None:
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 > x0 and y1 > y0:
        panel = img[y0:y1, x0:x1]
        panel[:] = (panel * k).astype(img.dtype)


def draw_hud(bgr: np.ndarray, lines: list[str], scale: float = 1.0) -> None:
    """HUD text (top left) on a translucent dark panel, in place. ``scale`` = the
    image's display scale (font, spacing and stroke grow with it)."""
    if not lines:
        return
    fs, step, th = 0.55 * scale, int(round(22 * scale)), max(1, int(round(scale)))
    # Translucent dark panel behind white text: readable on the white sky and the
    # grey floor. (A thick black "outline" pass doesn't work: Hershey glyph advance
    # grows with thickness, so the outline drifts sideways.)
    w = max(cv2.getTextSize(line, _FONT, fs, th)[0][0] for line in lines)
    _darken(bgr, 0, 0, int(20 * scale) + w, step * len(lines) + int(10 * scale), 0.45)
    x = int(round(10 * scale))
    for i, line in enumerate(lines):
        cv2.putText(bgr, line, (x, int(round(24 * scale)) + step * i), _FONT, fs, _WHITE,
                    th, cv2.LINE_AA)


def draw_rec_badge(bgr: np.ndarray, text: str = "REC", scale: float = 1.0) -> None:
    """Red dot + text in the top-right corner (recording indicator)."""
    fs, th = 0.6 * scale, max(1, int(round(2 * scale)))
    (tw, tht), _ = cv2.getTextSize(text, _FONT, fs, th)
    pad, r = int(10 * scale), int(7 * scale)
    x1 = bgr.shape[1] - pad
    x0 = x1 - tw - 3 * r - pad
    _darken(bgr, x0 - pad, 0, bgr.shape[1], tht + 3 * pad, 0.45)
    cy = pad + tht // 2 + int(4 * scale)
    cv2.circle(bgr, (x0 + r, cy), r, _REC_RED, -1, cv2.LINE_AA)
    cv2.putText(bgr, text, (x0 + 3 * r, cy + tht // 2), _FONT, fs, _WHITE, th, cv2.LINE_AA)


def draw_help_overlay(bgr: np.ndarray, groups: HelpGroups, scale: float = 1.0,
                      title: str = "Keys  (? closes this help)") -> None:
    """Centered key-help panel, in place. Font shrinks if the table would not fit."""
    rows: list[tuple[str, str, str]] = [("title", title, "")]
    for name, items in groups:
        rows.append(("group", name, ""))
        rows.extend(("key", k, d) for k, d in items)
    H, W = bgr.shape[:2]
    fs, th = 0.45 * scale, max(1, int(round(scale)))
    for _ in range(8):  # shrink until it fits
        step = int(round(fs / 0.45 * 18))
        kw = max(cv2.getTextSize(k, _FONT, fs, th)[0][0] for kind, k, _ in rows if kind == "key")
        dw = max(cv2.getTextSize(d, _FONT, fs, th)[0][0] for kind, _, d in rows if kind == "key")
        gap = int(step * 0.9)
        pw, ph = kw + dw + 3 * gap, step * (len(rows) + len(groups) * 0.4) + 2 * gap
        if pw <= W * 0.97 and ph <= H * 0.97:
            break
        fs *= 0.9
    x0, y0 = int((W - pw) / 2), int((H - ph) / 2)
    _darken(bgr, x0, y0, int(x0 + pw), int(y0 + ph), 0.22)
    cv2.rectangle(bgr, (x0, y0), (int(x0 + pw), int(y0 + ph)), (120, 120, 120), th)
    y = y0 + gap + int(step * 0.7)
    for kind, a, b in rows:
        if kind == "title":
            cv2.putText(bgr, a, (x0 + gap, y), _FONT, fs * 1.15, _WHITE, th + 1, cv2.LINE_AA)
        elif kind == "group":
            y += int(step * 0.4)
            cv2.putText(bgr, a, (x0 + gap, y), _FONT, fs, (120, 210, 255), th + 1,
                        cv2.LINE_AA)
        elif b.startswith(UNAVAILABLE_MARK):  # greyed out: not available right now
            b = b[len(UNAVAILABLE_MARK):]
            cv2.putText(bgr, a, (x0 + gap, y), _FONT, fs, _GREY, th, cv2.LINE_AA)
            cv2.putText(bgr, b, (x0 + 2 * gap + kw, y), _FONT, fs, _GREY, th, cv2.LINE_AA)
        else:
            cv2.putText(bgr, a, (x0 + gap, y), _FONT, fs, (140, 230, 140), th, cv2.LINE_AA)
            cv2.putText(bgr, b, (x0 + 2 * gap + kw, y), _FONT, fs, _WHITE, th, cv2.LINE_AA)
        y += step


def compose_frame(rgb: np.ndarray, hud_lines: list[str] | None = None, *,
                  scale: float = 1.0, help_groups: HelpGroups | None = None,
                  rec: str | None = None) -> np.ndarray:
    """BGR image as shown in the window: upscaled by ``scale``, then HUD / REC badge
    (``rec`` = its text, e.g. "REC 12s") / help overlay drawn at that resolution.
    Also used for screenshots (scale 1)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if abs(scale - 1.0) > 0.02:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
    if hud_lines:
        draw_hud(bgr, hud_lines, scale)
    if rec:
        draw_rec_badge(bgr, rec, scale)
    if help_groups:
        draw_help_overlay(bgr, help_groups, scale)
    return bgr


class LiveViewer:
    def __init__(self, title: str = "PerpetualFly", display_scale: float | None = None,
                 frame_size: tuple[int, int] = (960, 640)) -> None:
        """``display_scale``: None = auto for a ``frame_size`` = (w, h) frame
        (backing scale on Retina, capped to ~2/3 of the screen width; env
        ``PERPETUALFLY_FLY_SCALE`` overrides)."""
        self.title = title
        self.shift_names = False  # True: upper-case letters -> "shift+<k>" (decode_key)
        self.display_scale = (auto_display_scale(frame_size, 0.67, 0.8, env_var=FLY_SCALE_ENV)
                              if display_scale is None else float(display_scale))
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
        self._shown_once = False
        self.last_shown: np.ndarray | None = None  # last BGR image (tests / debugging)

    def show(self, rgb: np.ndarray, hud_lines: list[str] | None = None, *,
             help_groups: HelpGroups | None = None, rec: str | None = None) -> None:
        bgr = compose_frame(rgb, hud_lines, scale=self.display_scale,
                            help_groups=help_groups, rec=rec)
        cv2.imshow(self.title, bgr)
        self.last_shown = bgr
        self._shown_once = True

    def poll_keys(self, wait_ms: int = 1) -> list[str]:
        """Pump the GUI event loop and return pressed keys (usually 0 or 1).

        ``waitKeyEx`` must be called regularly or the window freezes. On macOS a
        call takes ~10-15 ms even with ``wait_ms=1``.
        """
        keys = []
        code = cv2.waitKeyEx(wait_ms)
        while code != -1:
            name = decode_key(code, self.shift_names)
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
