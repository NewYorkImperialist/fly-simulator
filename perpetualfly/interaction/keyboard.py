"""Normalize OpenCV ``waitKeyEx`` codes to key names.

Arrow keys have different codes per HighGUI backend, so all known ones are mapped.
"""

from __future__ import annotations

_SPECIAL = {
    27: "esc",
    32: "space",
    13: "enter",
    9: "tab",
    # macOS (Cocoa backend): NSUpArrowFunctionKey etc.
    63232: "up", 63233: "down", 63234: "left", 63235: "right",
    # Linux (GTK/Qt) keysyms
    65361: "left", 65362: "up", 65363: "right", 65364: "down",
    # Windows (virtual-key << 16)
    2424832: "left", 2490368: "up", 2555904: "right", 2621440: "down",
}


def decode_key(code: int) -> str | None:
    """Return a lowercase key name ('q', 'space', 'left', ...) or None for no key."""
    if code is None or code < 0:
        return None
    if code in _SPECIAL:
        return _SPECIAL[code]
    low = code & 0xFF
    if code < 256 or (code & 0xFFFF) < 256:  # some backends set modifier bits high
        if low in _SPECIAL:
            return _SPECIAL[low]
        if 32 < low < 127:
            return chr(low).lower()
    return f"code{code}"
