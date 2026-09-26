"""Live brain-activity window (OpenCV) for the PerpetualFly connectome model.

Three layers:

* ``BrainRenderer`` -- pure numpy/OpenCV drawing, no GUI. Ingests ``BrainState`` /
  ``StimulusEvent`` / ``BrainLayout`` messages, animates between them on its own
  clock and draws a BGR frame. Used by everything below and by the headless
  ``render_frame`` (tests, PNG/MP4).
* ``BrainWindow`` -- a named OpenCV window around a renderer (``tick()`` renders,
  shows and pumps the GUI event loop; must run on the process's main thread).
* ``run_window`` -- ``multiprocessing`` target that owns a ``BrainWindow`` and drains
  a queue; ``BrainWindowProcess`` is the parent-side handle (spawn context, bounded
  queue, never blocks the caller, closing the window never affects the parent).

Smoothness when the brain is slow: every ``BrainState`` covers ``window_s`` seconds of
brain time. The renderer keeps a *playhead* in brain time that is advanced at the
wall-clock pace needed to reach the newest state's ``brain_time`` just as the next
state is expected, so the spikes of each window are replayed progressively (neurons
flash, the raster scrolls) instead of appearing in one lump once a second. Region /
transmitter / descending values ease towards their newest values meanwhile.
"""

from __future__ import annotations

import math
import os
import queue as queue_mod
import re
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from perpetualfly.brain.schema import (
    DESCENDING_GROUPS,
    NEUROTRANSMITTERS,
    BrainLayout,
    BrainState,
    StimulusEvent,
)
from perpetualfly.brain_viz.atlas import (
    GROUP_ORDER,
    NT_COLORS_BGR,
    NT_DISPLAY,
    UNKNOWN_NT_BGR,
    load_atlas,
    normalize_region_name,
    region_base,
    region_group,
)
from perpetualfly.display import auto_display_scale as _auto_display_scale
from perpetualfly.display import screen_info

# ASCII on purpose: OpenCV's Cocoa backend mangles non-ASCII window titles (the em dash
# came out as ",Äî"). The in-image header uses the proper "PerpetualFly — Brain".
WINDOW_TITLE = "PerpetualFly - Brain"
DEFAULT_SIZE = (1280, 800)

# ----------------------------------------------------------------------------- style
BG = (17, 13, 12)
PANEL = (27, 22, 20)
BORDER = (60, 52, 47)
TEXT = (240, 238, 236)
DIM = (165, 156, 150)
FAINT = (105, 97, 92)
GRID = (46, 40, 37)
L_COL = (255, 190, 105)    # fly's left: light blue
R_COL = (150, 120, 255)    # fly's right: rose
ESC_COL = (70, 70, 250)    # escape: red
GROOM_COL = (120, 220, 120)  # grooming DNs: green
GOOD = (120, 210, 110)
WARN = (60, 170, 255)
BAD = (80, 80, 240)

NT_BGR = np.array([NT_COLORS_BGR[n] for n in NEUROTRANSMITTERS] + [UNKNOWN_NT_BGR],
                  np.float32)  # index -1 -> unknown (last row)

_HEAT = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None],
                          cv2.COLORMAP_INFERNO)[:, 0, :].astype(np.float32)
# map LUT: heat colour with brightness ~ level^2 so baseline activity stays a dim haze
_lv = np.linspace(0, 1, 256, dtype=np.float32)[:, None]
_HEAT_LUT = np.clip(_HEAT * (0.12 + 0.88 * _lv ** 2) * 0.85, 0, 255).astype(np.uint8)[None]
del _lv

_DN_ROWS = (("WALK", "walk_L", "walk_R"), ("TURN", "turn_L", "turn_R"),
            ("BACKWARD", "backward_L", "backward_R"), ("ESCAPE", "escape", None),
            ("GROOM", "groom", None))
_DN_FLOOR_HZ = {"WALK": 40.0, "TURN": 40.0, "BACKWARD": 40.0, "ESCAPE": 60.0, "GROOM": 30.0}
_DN_SINGLE_LABEL = {"escape": "GF", "groom": "DNg12"}


# ----------------------------------------------------------------------------- text
_FONT: object = None


def _font():
    global _FONT
    if _FONT is None:
        try:
            _FONT = cv2.FontFace("sans")
        except Exception:  # OpenCV < 5: Hershey fallback
            _FONT = False
    return _FONT or None


def text_width(s: str, size: int = 13, weight: int = 400) -> int:
    ff = _font()
    if ff is not None:
        r = cv2.getTextSize((4096, 256), s, (0, 0), ff, size, weight)
        return int(r[2])
    (w, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, size / 30.0,
                                2 if weight >= 600 else 1)
    return int(w)


def put_text(img: np.ndarray, s: str, org, color, size: int = 13, weight: int = 400,
             align: str = "left") -> int:
    """Draw ``s`` with baseline-left at ``org`` (or right/center aligned). Returns x end."""
    x, y = int(org[0]), int(org[1])
    if align != "left":
        w = text_width(s, size, weight)
        x -= w if align == "right" else w // 2
    color = tuple(int(c) for c in color)
    ff = _font()
    if ff is not None:
        cv2.putText(img, s, (x, y), color, ff, size, weight)
    else:
        s = s.replace("—", "-").replace("×", "x").replace("…", "...")
        cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, color,
                    2 if weight >= 600 else 1, cv2.LINE_AA)
    return x + text_width(s, size, weight)


def _pill(img, x, y, s, fg, bg, size=12, weight=600, pad=6, h=None) -> int:
    """Rounded-ish label box with top-left (x, y). Returns right edge."""
    w = text_width(s, size, weight) + 2 * pad
    h = h or size + 10
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1, cv2.LINE_AA)
    put_text(img, s, (x + pad, y + h - (h - size) // 2 - 2), fg, size, weight)
    return x + w


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


def _panel(img, r: Rect, title: str = "", subtitle: str = "") -> None:
    cv2.rectangle(img, (r.x, r.y), (r.x2, r.y2), PANEL, -1)
    cv2.rectangle(img, (r.x, r.y), (r.x2, r.y2), BORDER, 1)
    if title:
        x = put_text(img, title, (r.x + 12, r.y + 21), TEXT, 13, 700)
        if subtitle:
            put_text(img, subtitle, (x + 10, r.y + 21), DIM, 12, 400)


# ----------------------------------------------------------------------------- helpers
def short_stimulus(s) -> str:
    """'whip_hit left 0.6' / StimulusEvent -> 'WHIP L 0.60'."""
    if isinstance(s, StimulusEvent):
        lab = (s.details or {}).get("label") if isinstance(s.details, dict) else None
        if lab:  # e.g. the app's "LOOM" / "SUGAR" keys
            return str(lab).upper()[:28]
        s = f"{s.kind} {s.side} {s.intensity}"
    kinds = {"whip_hit": "WHIP", "whip": "WHIP", "shove": "SHOVE", "fall": "FALL",
             "ground_contact": "CONTACT", "reset": "RESET", "manual": "MANUAL"}
    sides = {"left": "L", "right": "R", "front": "FRONT", "rear": "REAR", "top": "TOP",
             "none": "", "l": "L", "r": "R"}
    out = []
    for tok in re.split(r"[\s:,/|=]+", str(s).strip()):
        if not tok:
            continue
        low = tok.lower()
        if low in kinds:
            out.append(kinds[low])
        elif low in sides:
            if sides[low]:
                out.append(sides[low])
        else:
            try:
                out.append(f"{float(tok):.2f}")
            except ValueError:
                out.append(tok.upper()[:12])
    return " ".join(out)[:28] or "?"


def _stim_color(label: str):
    if label.startswith("WHIP"):
        return (60, 60, 225)
    if label.startswith(("SHOVE", "FALL")):
        return (40, 140, 235)
    return (150, 110, 70)


def _arr(x, n: int | None = None, dtype=np.float64) -> np.ndarray:
    """Robust array coercion: None/garbage -> zeros(n)."""
    try:
        a = np.asarray(x if x is not None else [], dtype=dtype).ravel()
    except (TypeError, ValueError):
        a = np.zeros(0, dtype)
    a = np.where(np.isfinite(a), a, 0) if a.dtype.kind == "f" else a
    if n is not None and len(a) != n:
        b = np.zeros(n, dtype)
        b[: min(n, len(a))] = a[: min(n, len(a))]
        a = b
    return a


def _fit_axes(src: np.ndarray, dst: np.ndarray):
    """Per-axis affine (optionally swapped) map src->dst. Returns (fn, rel_rms)."""
    best = None
    for swap in (False, True):
        s = src[:, ::-1] if swap else src
        coefs, res = [], 0.0
        for d in range(2):
            A = np.stack([s[:, d], np.ones(len(s))], 1)
            c, *_ = np.linalg.lstsq(A, dst[:, d], rcond=None)
            res += float(((A @ c - dst[:, d]) ** 2).sum())
            coefs.append(c)
        if best is None or res < best[0]:
            best = (res, swap, coefs)
    res, swap, coefs = best
    spread = float(np.sqrt(((dst - dst.mean(0)) ** 2).sum(1).mean())) or 1.0
    rel = math.sqrt(res / len(src)) / spread

    def fn(p: np.ndarray, swap=swap, coefs=coefs) -> np.ndarray:
        p = np.asarray(p, np.float64)
        p = p[..., ::-1] if swap else p
        return np.stack([p[..., 0] * coefs[0][0] + coefs[0][1],
                         p[..., 1] * coefs[1][0] + coefs[1][1]], -1)

    return fn, rel


# ============================================================================ renderer
class BrainRenderer:
    """Draws the brain window into a BGR ``np.ndarray`` (no GUI calls)."""

    def __init__(self, layout: BrainLayout, size: tuple[int, int] = DEFAULT_SIZE,
                 rate_range: tuple[float, float] = (1.0, 150.0),
                 trace_span_s: float = 10.0, group_by: str = "region",
                 atlas: dict | None = None) -> None:
        self.W, self.H = int(size[0]), int(size[1])
        self.rate_lo, self.rate_hi = rate_range
        self.span = float(trace_span_s)
        self.group_by = group_by
        self._atlas = atlas if atlas is not None else load_atlas()
        # octopamine stress gauge row (DESCENDING panel): shown once a state carries
        # an enabled BrainState.neuromod (perpetualfly/brain/neuromod.py)
        self._show_nm = False
        self._layout_panels()
        self.set_layout(layout)

    # ------------------------------------------------------------------ setup
    def _layout_panels(self) -> None:
        W, H, p = self.W, self.H, 10
        hh = 64
        lw = int(W * 0.61)
        map_h = min(int(lw / 1.9), int((H - hh) * 0.6))
        self.r_header = Rect(0, 0, W, hh)
        self.r_map = Rect(p, hh, lw, map_h)
        self.r_raster = Rect(p, hh + map_h + p, lw, H - (hh + map_h + p) - p)
        rx = p + lw + p
        rw = W - rx - p
        nt_h = int((H - hh - p) * 0.40)
        self.r_nt = Rect(rx, hh, rw, nt_h)
        self.r_dn = Rect(rx, hh + nt_h + p, rw, H - (hh + nt_h + p) - p)

    def set_layout(self, layout: BrainLayout) -> None:
        self.layout = layout
        self.n_regions = len(layout.regions)
        self.n_neurons = len(_arr(layout.display_neuron_ids, dtype=np.float64))
        nt = _arr(layout.display_neuron_nt, self.n_neurons, np.int64)
        nt = np.where((nt >= 0) & (nt < len(NEUROTRANSMITTERS)), nt, -1)
        self._nt = nt
        self._neuron_col = NT_BGR[nt]  # -1 -> last row (unknown)
        reg = _arr(layout.display_neuron_region, self.n_neurons, np.int64)
        self._nreg = np.where((reg >= 0) & (reg < max(self.n_regions, 1)), reg, 0)
        self._build_map_geometry()
        self._build_raster_rows()
        self._reset_dynamics()
        self._static = self._draw_static()

    def _reset_dynamics(self) -> None:
        R, N, D = self.n_regions, self.n_neurons, len(DESCENDING_GROUPS)
        self.state: BrainState | None = None
        self._disp_region = np.zeros(R)
        self._tgt_region = np.zeros(R)
        self._disp_nt = np.zeros((2, len(NEUROTRANSMITTERS)))
        self._tgt_nt = np.zeros((2, len(NEUROTRANSMITTERS)))
        self._disp_dn = np.zeros(D)
        self._tgt_dn = np.zeros(D)
        self._flash = np.zeros(N, np.float32)
        self._pending_t = np.zeros(0)
        self._pending_i = np.zeros(0, np.int64)
        self._seg = None  # (wall0, bt0, bt1, dur)
        self._interval = None
        self._last_arrival = None
        self._last_wall = None
        self._raster_chunks: deque = deque()
        self._dn_hist: deque = deque()
        self._bt_hist: deque = deque()  # (wall, playhead brain time)
        self._stims: deque = deque(maxlen=12)    # (wall, label)
        self._markers: deque = deque(maxlen=64)  # (wall, label)
        self._direct_stim_wall = -1e9
        self._rtf = None
        self._drive_ext = None
        self._nm: dict = {}
        self._tgt_nm = 0.0
        self._disp_nm = 0.0
        self._nm_hist: deque = deque()  # (wall, displayed level)

    # .................................................................. map geometry
    def _map_transform(self, lo: np.ndarray, hi: np.ndarray, flip_y: bool = False):
        r = self.r_map
        m_top, m_side, m_bot = 34, 26, 30
        aw, ah = r.w - 2 * m_side, r.h - m_top - m_bot
        span = np.maximum(hi - lo, 1e-9)
        s = min(aw / span[0], ah / span[1])
        ox = r.x + m_side + (aw - s * span[0]) / 2
        oy = r.y + m_top + (ah - s * span[1]) / 2

        def to_px(p: np.ndarray) -> np.ndarray:
            p = np.asarray(p, np.float64)
            y = (hi[1] - p[..., 1]) if flip_y else (p[..., 1] - lo[1])
            return np.stack([ox + (p[..., 0] - lo[0]) * s, oy + y * s], -1)

        return to_px, s

    def _build_map_geometry(self) -> None:
        L, atlas = self.layout, self._atlas
        R, N = self.n_regions, self.n_neurons
        rxy = np.full((R, 2), np.nan)
        try:
            a = np.asarray(L.region_xy, np.float64).reshape(R, 2)
            rxy[:] = a
        except (TypeError, ValueError):
            pass
        rxy[~np.isfinite(rxy).all(1)] = np.nan
        # optional layout outlines, one list entry per region
        louts: list[np.ndarray | None] = [None] * R
        if L.region_outline_xy is not None and len(L.region_outline_xy) == R:
            for i, o in enumerate(L.region_outline_xy):
                try:
                    o = np.asarray(o, np.float64).reshape(-1, 2)
                except (TypeError, ValueError):
                    continue
                if len(o) >= 3 and np.isfinite(o).all():
                    louts[i] = o
        nxy = None
        try:
            nxy = np.asarray(L.display_neuron_xy, np.float64).reshape(N, 2)
            ok = np.isfinite(nxy).all(1)
            if ok.sum() < max(1, N // 2) or np.ptp(nxy[ok], 0).min() <= 0:
                nxy = None
        except (TypeError, ValueError):
            nxy = None

        self.mode = "layout"
        keys = [normalize_region_name(n) for n in L.regions]
        amatch = [(k in atlas["regions"]) if atlas else False for k in keys]
        fit = None
        if atlas is not None and sum(amatch) >= 3:
            idx = [i for i in range(R) if amatch[i] and np.isfinite(rxy[i]).all()]
            if len(idx) >= 3 and np.ptp(rxy[idx], 0).min() > 0:
                dst = np.array([atlas["regions"][keys[i]]["centroid"] for i in idx])
                fn, rel = _fit_axes(rxy[idx], dst)
                if rel >= 0.25 and len(idx) >= 8:
                    # robust refit: region_xy may be e.g. the median position of each
                    # region's neurons, which for a few regions (EPA, FLA, GA with the
                    # real engine) lies far from the neuropil; fit on the best 75 %
                    src = rxy[idx]
                    r0 = np.linalg.norm(fn(src) - dst, axis=1)
                    keep = r0 <= np.quantile(r0, 0.75)
                    fn2, rel2 = _fit_axes(src[keep], dst[keep])
                    if rel2 < rel:
                        fn, rel = fn2, rel2
                if rel < 0.25:
                    fit = fn
            elif sum(amatch) >= 0.5 * R:
                # names match but no usable region_xy: use atlas centroids, and since
                # neuron xy can't be registered, scatter neurons inside their regions
                rxy[:] = np.nan
                nxy = None
                fit = lambda p: np.asarray(p, np.float64)  # noqa: E731
        if fit is not None:
            self.mode = "atlas"
            to_atlas = fit
            lo, hi = atlas["bounds"][:2], atlas["bounds"][2:]
            bo = atlas["brain_outline"]
            if bo:
                allb = np.concatenate(bo)
                lo, hi = allb.min(0) - 6, allb.max(0) + 6
            to_px, scale = self._map_transform(lo, hi)
            px_of = lambda p: to_px(to_atlas(p))  # noqa: E731
            cent = np.full((R, 2), np.nan)
            for i in range(R):
                if np.isfinite(rxy[i]).all():
                    cent[i] = px_of(rxy[i])
                elif amatch[i]:
                    cent[i] = to_px(atlas["regions"][keys[i]]["centroid"])
            polys = []
            for i in range(R):
                if louts[i] is not None:
                    polys.append([px_of(louts[i])])
                elif amatch[i]:
                    polys.append([to_px(o) for o in atlas["regions"][keys[i]]["outline"]])
                else:
                    polys.append([])
            self._silhouette = [to_px(o) for o in bo]
            used = {keys[i] for i in range(R) if amatch[i]}
            self._context = [to_px(o) for k, r in atlas["regions"].items() if k not in used
                             for o in r["outline"]]
            self._scale_um = scale
        else:
            pts = [rxy[np.isfinite(rxy).all(1)]]
            pts += [o for o in louts if o is not None]
            if nxy is not None:
                pts.append(nxy[np.isfinite(nxy).all(1)])
            pts = np.concatenate([p for p in pts if len(p)]) if any(len(p) for p in pts) \
                else np.zeros((0, 2))
            if len(pts) < 2 or np.ptp(pts, 0).max() <= 0:
                # nothing usable: arrange regions on a ring
                ang = np.linspace(0, 2 * np.pi, max(R, 1), endpoint=False)
                rxy = np.stack([np.cos(ang), 0.5 * np.sin(ang)], 1)[:R]
                pts = rxy
                nxy = None
            lo, hi = pts.min(0), pts.max(0)
            pad = 0.06 * np.maximum(hi - lo, 1e-9).max()
            to_px, scale = self._map_transform(lo - pad, hi + pad)
            px_of = to_px
            cent = np.array([px_of(p) if np.isfinite(p).all() else (np.nan, np.nan)
                             for p in rxy]).reshape(R, 2)
            polys = [[px_of(o)] if o is not None else [] for o in louts]
            self._silhouette = []
            self._context = []
            self._scale_um = scale
        # regions without any position: park them along the bottom edge
        miss = ~np.isfinite(cent).all(1)
        if miss.any():
            k = np.nonzero(miss)[0]
            xs = np.linspace(self.r_map.x + 40, self.r_map.x2 - 40, len(k) + 2)[1:-1]
            cent[k] = np.stack([xs, np.full(len(k), self.r_map.y2 - 40)], 1)
        self._reg_px = cent
        self._reg_polys = polys
        # neuron positions
        rng = np.random.default_rng(1)
        if nxy is not None:
            npx = px_of(nxy)
            bad = ~np.isfinite(npx).all(1)
        else:
            npx = np.zeros((N, 2))
            bad = np.ones(N, bool)
        if bad.any():
            for ri in np.unique(self._nreg[bad]):
                sel = np.nonzero(bad & (self._nreg == ri))[0]
                npx[sel] = self._sample_region_px(rng, ri, len(sel))
        r = self.r_map
        npx[:, 0] = np.clip(npx[:, 0], r.x + 1, r.x2 - 2)
        npx[:, 1] = np.clip(npx[:, 1], r.y + 1, r.y2 - 2)
        self._neuron_px = np.round(npx).astype(np.int32)
        # region masks (map-local coordinates, anti-aliased)
        self._masks = []
        for i in range(R):
            self._masks.append(self._region_mask(i))

    def _sample_region_px(self, rng, ri: int, n: int) -> np.ndarray:
        polys = self._reg_polys[ri] if ri < len(self._reg_polys) else []
        c = self._reg_px[ri] if ri < len(self._reg_px) else np.array(
            [self.r_map.x + self.r_map.w / 2, self.r_map.y + self.r_map.h / 2])
        if polys:
            allp = np.concatenate(polys)
            lo = np.floor(allp.min(0)).astype(int)
            hi = np.ceil(allp.max(0)).astype(int)
            m = np.zeros((hi[1] - lo[1] + 1, hi[0] - lo[0] + 1), np.uint8)
            cv2.fillPoly(m, [np.round(p - lo).astype(np.int32) for p in polys], 255)
            ys, xs = np.nonzero(m)
            if len(xs):
                k = rng.integers(0, len(xs), n)
                return np.stack([xs[k], ys[k]], 1) + lo
        return c + rng.normal(0, 7, (n, 2))

    def _region_mask(self, i: int):
        r = self.r_map
        polys = self._reg_polys[i]
        if polys:
            allp = np.concatenate(polys)
            x0, y0 = np.floor(allp.min(0)).astype(int) - 2
            x1, y1 = np.ceil(allp.max(0)).astype(int) + 2
        else:
            cx, cy = self._reg_px[i]
            rad = 9
            x0, y0, x1, y1 = int(cx - rad - 2), int(cy - rad - 2), int(cx + rad + 2), \
                int(cy + rad + 2)
        x0, y0 = max(x0, r.x), max(y0, r.y)
        x1, y1 = min(x1, r.x2), min(y1, r.y2)
        if x1 <= x0 or y1 <= y0:
            return None
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        if polys:
            cv2.fillPoly(m, [np.round((p - (x0, y0)) * 8).astype(np.int32) for p in polys],
                         255, cv2.LINE_AA, shift=3)
        else:
            cx, cy = self._reg_px[i]
            cv2.circle(m, (int((cx - x0) * 8), int((cy - y0) * 8)), 9 * 8, 255, -1,
                       cv2.LINE_AA, shift=3)
        return (x0 - r.x, y0 - r.y, m.astype(np.float32) / 255.0)

    # .................................................................. raster rows
    def _build_raster_rows(self) -> None:
        N = self.n_neurons
        names = self.layout.regions
        grp = np.array([region_group(n) for n in names] or ["other"])
        gorder = {g: i for i, g in enumerate(GROUP_ORDER + ("other",))}
        greg = np.array([gorder.get(g, len(gorder)) for g in grp])
        nreg = self._nreg
        ntk = np.where(self._nt >= 0, self._nt, len(NEUROTRANSMITTERS))
        if self.group_by == "nt":
            order = np.lexsort((nreg, greg[nreg] if N else nreg, ntk))
            key = ntk[order]
            label_of = lambda k: NT_DISPLAY.get(  # noqa: E731
                NEUROTRANSMITTERS[k], "?") if k < len(NEUROTRANSMITTERS) else "?"
        elif N and (grp == "other").all():  # unknown region names: group by region
            order = np.lexsort((ntk, nreg))
            key = nreg[order]
            label_of = lambda k: str(names[k])[:9] if k < len(names) else "?"  # noqa: E731
        else:
            order = np.lexsort((ntk, nreg, greg[nreg] if N else nreg))
            key = greg[nreg[order]] if N else np.zeros(0, int)
            inv = {i: g for g, i in gorder.items()}
            label_of = lambda k: inv.get(k, "?")  # noqa: E731
        r = self.r_raster
        self._rr = Rect(r.x + 62, r.y + 32, r.w - 62 - 12, r.h - 32 - 24)
        rank = np.empty(N, np.int64)
        rank[order] = np.arange(N)
        self._raster_row = (self._rr.y + (rank * self._rr.h) // max(N, 1)).astype(np.int32)
        bands = []
        if N:
            start = 0
            for j in range(1, N + 1):
                if j == N or key[j] != key[start]:
                    bands.append((label_of(int(key[start])),
                                  self._rr.y + start * self._rr.h // N,
                                  self._rr.y + j * self._rr.h // N, int(key[start])))
                    start = j
        self._bands = bands

    def toggle_grouping(self) -> None:
        self.group_by = "nt" if self.group_by == "region" else "region"
        self._build_raster_rows()
        self._raster_chunks.clear()
        self._static = self._draw_static()

    # .................................................................. static art
    def _rate_to_unit(self, r) -> np.ndarray:
        r = np.asarray(r, np.float64)
        return np.clip(np.log(np.maximum(r, 1e-9) / self.rate_lo) /
                       math.log(self.rate_hi / self.rate_lo), 0.0, 1.0)

    def _draw_static(self) -> np.ndarray:
        img = np.full((self.H, self.W, 3), BG, np.uint8)
        self._draw_static_map(img)
        self._draw_static_nt(img)
        self._draw_static_dn(img)
        self._draw_static_raster(img)
        cv2.line(img, (0, self.r_header.y2 - 1), (self.W, self.r_header.y2 - 1), BORDER, 1)
        return img

    def _draw_static_map(self, img) -> None:
        r = self.r_map
        _panel(img, r)
        sub = img[r.y:r.y2, r.x:r.x2]
        # soft radial vignette
        yy, xx = np.mgrid[0:r.h, 0:r.w]
        d = np.sqrt(((xx - r.w / 2) / (r.w * 0.6)) ** 2 + ((yy - r.h / 2) / (r.h * 0.6)) ** 2)
        glow = np.clip(1 - d, 0, 1)[..., None] * np.array([14, 9, 6], np.float32)
        sub[:] = np.clip(sub.astype(np.float32) + glow, 0, 255).astype(np.uint8)
        for o in self._silhouette:
            cv2.fillPoly(img, [np.round(o * 8).astype(np.int32)], (38, 31, 28), cv2.LINE_AA,
                         shift=3)
        for o in self._context:
            cv2.polylines(img, [np.round(o * 8).astype(np.int32)], True, (52, 45, 41), 1,
                          cv2.LINE_AA, shift=3)
        for polys in self._reg_polys:
            for o in polys:
                cv2.fillPoly(img, [np.round(o * 8).astype(np.int32)], (44, 37, 33),
                             cv2.LINE_AA, shift=3)
        for polys in self._reg_polys:
            for o in polys:
                cv2.polylines(img, [np.round(o * 8).astype(np.int32)], True, (84, 74, 68), 1,
                              cv2.LINE_AA, shift=3)
        for o in self._silhouette:
            cv2.polylines(img, [np.round(o * 8).astype(np.int32)], True, (120, 106, 98), 1,
                          cv2.LINE_AA, shift=3)
        if self.mode == "layout":
            for i, polys in enumerate(self._reg_polys):
                if not polys and self._masks[i] is not None:
                    c = self._reg_px[i]
                    cv2.circle(img, (int(c[0] * 8), int(c[1] * 8)), 9 * 8, (84, 74, 68), 1,
                               cv2.LINE_AA, shift=3)
        # faint resting neuron dots
        if self.n_neurons:
            p = self._neuron_px
            col = (self._neuron_col * 0.28 + 12).astype(np.uint8)
            img[p[:, 1], p[:, 0]] = np.maximum(img[p[:, 1], p[:, 0]], col)
        # group labels at the area-weighted centre of each super-region
        self._draw_group_labels(img)
        put_text(img, "BRAIN MAP", (r.x + 12, r.y + 21), TEXT, 13, 700)
        mode = "FlyWire neuropils" if self.mode == "atlas" else "layout positions"
        put_text(img, f"region glow = mean rate · dots = display-neuron spikes · {mode}",
                 (r.x + 96, r.y + 21), DIM, 12)
        put_text(img, "fly L", (r.x + 12, r.y2 - 12), L_COL, 12, 600)
        put_text(img, "fly R", (r.x + 12 + 44, r.y2 - 12), R_COL, 12, 600)
        put_text(img, "dorsal up", (r.x + 100, r.y2 - 12), FAINT, 11)
        # colour bar
        cb = Rect(r.x2 - 230, r.y2 - 22, 150, 8)
        bar = _HEAT[np.linspace(0, 255, cb.w).astype(int)][None].repeat(cb.h, 0)
        img[cb.y:cb.y2, cb.x:cb.x2] = bar.astype(np.uint8)
        cv2.rectangle(img, (cb.x - 1, cb.y - 1), (cb.x2, cb.y2), BORDER, 1)
        for hz in (1, 10, 100):
            u = float(self._rate_to_unit(hz))
            x = int(cb.x + u * (cb.w - 1))
            cv2.line(img, (x, cb.y2), (x, cb.y2 + 3), DIM, 1)
            put_text(img, f"{hz}", (x, cb.y - 4), DIM, 10, 400, "center")
        put_text(img, "Hz", (cb.x2 + 8, cb.y2), DIM, 11)
        put_text(img, "region rate", (cb.x - 8, cb.y2), DIM, 11, 400, "right")

    def _draw_group_labels(self, img) -> None:
        if self.mode != "atlas":
            return
        names = self.layout.regions
        pretty = {"OL": "optic lobe", "AL": "AL", "MB": "MB", "LH": "LH", "CX": "CX",
                  "LX": "LAL", "SNP": "SMP/SIP/SLP", "VLNP": "VLNP", "GNG": "GNG",
                  "PENP": "AMMC/SAD", "VMNP": "VES/IPS/SPS", "INP": "INP"}
        placed: list[np.ndarray] = []
        for side in ("L", "R", ""):
            for g in ("OL", "AL", "MB", "LH", "CX", "LX", "GNG"):
                idx = [i for i, n in enumerate(names) if region_group(n) == g and
                       (normalize_region_name(n).endswith("_" + side) if side else
                        not normalize_region_name(n).endswith(("_L", "_R")))]
                if g == "OL":
                    idx = [i for i in idx if region_base(names[i]) in ("ME", "LO")]
                if g == "MB":
                    idx = [i for i in idx if region_base(names[i]) == "MB_CA"]
                if not idx:
                    continue
                c = np.nanmean(self._reg_px[idx], 0)
                if not np.isfinite(c).all():
                    continue
                if g == "OL":
                    c = c + (0, -self.r_map.h * 0.30)
                elif g == "MB":
                    c = c + (0, -16)
                elif g == "GNG":
                    c = c + (0, 30)
                elif g == "CX":
                    c = c + (0, -44)
                elif g == "AL":
                    c = c + (0, 22)
                if any(np.hypot(*(c - q)) < 30 for q in placed):
                    continue
                placed.append(c)
                put_text(img, pretty[g], (c[0], c[1]), (150, 140, 132), 11, 600, "center")

    def _draw_static_nt(self, img) -> None:
        r = self.r_nt
        _panel(img, r, "NEUROTRANSMITTER ACTIVITY", "by predicted transmitter")
        top = r.y + 44
        rows = len(NEUROTRANSMITTERS)
        foot = 36
        self._nt_row_h = (r.h - (top - r.y) - foot - 6) / rows
        self._nt_bar = Rect(r.x + 74, top, r.w - 74 - 150, int(self._nt_row_h * rows))
        b = self._nt_bar
        put_text(img, "mean rate (log)", (b.x, top - 6), FAINT, 10)
        put_text(img, "rate", (r.x2 - 112, top - 6), FAINT, 10, 400, "right")
        put_text(img, "% active", (r.x2 - 14, top - 6), FAINT, 10, 400, "right")
        for hz in (0.1, 1, 10, 100):
            u = float(self._nt_unit(hz))
            x = int(b.x + u * b.w)
            cv2.line(img, (x, b.y), (x, b.y2 - 4), GRID, 1)
            put_text(img, f"{hz:g}", (x, b.y2 + 8), FAINT, 9, 400, "center")
        for k, nt in enumerate(NEUROTRANSMITTERS):
            y = int(top + k * self._nt_row_h + self._nt_row_h / 2)
            cv2.circle(img, (r.x + 18, y - 4), 5, NT_COLORS_BGR[nt], -1, cv2.LINE_AA)
            put_text(img, NT_DISPLAY[nt], (r.x + 30, y + 1), TEXT, 13, 600)
        put_text(img, "DA / 5-HT / OA = spiking of those neurons,", (r.x + 12, r.y2 - 22),
                 DIM, 11)
        put_text(img, "not neuromodulator concentration or release.", (r.x + 12, r.y2 - 8),
                 DIM, 11)

    def _nt_unit(self, hz):
        lo, hi = 0.05, 150.0
        return np.clip(np.log(np.maximum(np.asarray(hz, np.float64), 1e-9) / lo) /
                       math.log(hi / lo), 0, 1)

    def _draw_static_dn(self, img) -> None:
        r = self.r_dn
        _panel(img, r, "DESCENDING COMMANDS", f"DN firing rate, last {self.span:g} s")
        drive_h = 98
        top = r.y + 34
        rows = len(_DN_ROWS) + (1 if self._show_nm else 0)
        rh = (r.h - 34 - drive_h - 8) / rows
        self._dn_rows = []
        self._nm_row = None
        if self._show_nm:
            k = len(_DN_ROWS)
            y0 = int(top + k * rh)
            pr = Rect(r.x + 86, y0 + 4, r.w - 86 - 76, int(rh) - 10)
            self._nm_row = pr
            col = NT_COLORS_BGR["OA"]
            cv2.rectangle(img, (pr.x, pr.y), (pr.x2, pr.y2), (22, 18, 16), -1)
            yy = int(pr.y2 - 0.5 * pr.h)
            cv2.line(img, (pr.x, yy), (pr.x2, yy), GRID, 1)
            cv2.line(img, (pr.x, pr.y2), (pr.x2, pr.y2), BORDER, 1)
            put_text(img, "1", (pr.x + 4, pr.y + 11), FAINT, 9)
            put_text(img, "PAIN/AROUSAL", (r.x + 12, y0 + int(rh / 2) - 6), col, 10, 700)
            put_text(img, "octopamine", (r.x + 12, y0 + int(rh / 2) + 7), DIM, 10)
            put_text(img, "(model)", (r.x + 12, y0 + int(rh / 2) + 19), FAINT, 10)
        for k, (name, gl, gr) in enumerate(_DN_ROWS):
            y0 = int(top + k * rh)
            pr = Rect(r.x + 86, y0 + 4, r.w - 86 - 76, int(rh) - 10)
            self._dn_rows.append((name, gl, gr, pr))
            cv2.rectangle(img, (pr.x, pr.y), (pr.x2, pr.y2), (22, 18, 16), -1)
            for f in (0.5,):
                yy = int(pr.y2 - f * pr.h)
                cv2.line(img, (pr.x, yy), (pr.x2, yy), GRID, 1)
            cv2.line(img, (pr.x, pr.y2), (pr.x2, pr.y2), BORDER, 1)
            put_text(img, name, (r.x + 12, y0 + int(rh / 2) + 1), TEXT, 12, 700)
            put_text(img, _DN_SINGLE_LABEL.get(gl, gl) if gr is None else "L / R", (r.x + 12, y0 + int(rh / 2) + 15),
                     FAINT, 10)
        self._dn_drive = Rect(r.x + 10, r.y2 - drive_h, r.w - 20, drive_h - 10)
        d = self._dn_drive
        cv2.line(img, (d.x, d.y - 4), (d.x2, d.y - 4), BORDER, 1)
        # time axis labels on the last row
        pr = self._nm_row if self._nm_row is not None else self._dn_rows[-1][3]
        put_text(img, f"-{self.span:g} s", (pr.x, pr.y2 + 11), FAINT, 9)
        put_text(img, "now", (pr.x2, pr.y2 + 11), FAINT, 9, 400, "right")

    def _draw_static_raster(self, img) -> None:
        r = self.r_raster
        how = "by region" if self.group_by == "region" else "by transmitter"
        _panel(img, r, "SPIKE RASTER",
               f"{self.n_neurons:,} display neurons {how}  ·  [G] regroup")
        rr = self._rr
        cv2.rectangle(img, (rr.x, rr.y), (rr.x2, rr.y2), (20, 16, 15), -1)
        for j, (lab, y0, y1, key) in enumerate(self._bands):
            if j % 2:
                cv2.rectangle(img, (rr.x, y0), (rr.x2, y1 - 1), (25, 21, 19), -1)
            if j:
                cv2.line(img, (rr.x - 6, y0), (rr.x2, y0), GRID, 1)
            if y1 - y0 >= 7:
                col = DIM
                if self.group_by == "nt" and key < len(NEUROTRANSMITTERS):
                    col = NT_COLORS_BGR[NEUROTRANSMITTERS[key]]
                put_text(img, lab, (rr.x - 8, (y0 + y1) // 2 + 4), col,
                         10 if y1 - y0 >= 10 else 9, 600, "right")
        for k in range(0, int(self.span) + 1, 2):
            x = int(rr.x2 - k / self.span * rr.w)
            cv2.line(img, (x, rr.y2), (x, rr.y2 + 3), FAINT, 1)
            put_text(img, "now" if k == 0 else f"-{k} s", (x, rr.y2 + 15), FAINT, 10, 400,
                     "right" if k == 0 else "left" if k >= self.span else "center")
        put_text(img, "wall time", (rr.x - 8, rr.y2 + 15), FAINT, 9, 400, "right")
        # NT legend (colours used everywhere)
        x = r.x2 - 12
        for nt in reversed(NEUROTRANSMITTERS):
            w = text_width(NT_DISPLAY[nt], 11, 600)
            x -= w
            put_text(img, NT_DISPLAY[nt], (x, r.y + 21), NT_COLORS_BGR[nt], 11, 600)
            x -= 14
            cv2.circle(img, (x + 6, r.y + 17), 4, NT_COLORS_BGR[nt], -1, cv2.LINE_AA)
            x -= 8

    # ------------------------------------------------------------------ ingest
    def ingest(self, item, wall: float | None = None) -> None:
        """Feed a ``BrainState``, ``StimulusEvent`` or ``BrainLayout``."""
        wall = time.time() if wall is None else wall
        if isinstance(item, BrainLayout):
            self.set_layout(item)
        elif isinstance(item, StimulusEvent):
            lab = short_stimulus(item)
            self._stims.append((wall, lab))
            self._markers.append((wall, lab))
            self._direct_stim_wall = wall
        elif isinstance(item, BrainState):
            self._ingest_state(item, wall)

    def _ingest_state(self, s: BrainState, wall: float) -> None:
        R = self.n_regions
        bt1 = float(s.brain_time) if np.isfinite(s.brain_time) else 0.0
        win = float(s.window_s) if (s.window_s and np.isfinite(s.window_s)) else 0.0
        win = max(win, 0.0)
        # arrival interval estimate (EMA of wall gaps)
        if self._last_arrival is not None:
            gap = wall - self._last_arrival
            if gap > 1e-3:
                self._interval = gap if self._interval is None else \
                    0.7 * self._interval + 0.3 * min(gap, 10.0)
        self._last_arrival = wall
        interval = self._interval
        rtf = float(s.realtime_factor) if (s.realtime_factor is not None and
                                           np.isfinite(s.realtime_factor)) else 0.0
        if rtf <= 0 and self.state is not None and interval:
            dbt = bt1 - float(self.state.brain_time)
            rtf = max(dbt, 0.0) / interval
        self._rtf = rtf if rtf > 0 else self._rtf
        if interval is None:
            interval = win / rtf if rtf > 0 and win > 0 else 0.25
        interval = float(np.clip(interval, 0.02, 5.0))
        # playhead segment
        cur = self._playhead(wall)
        bt0 = bt1 - win
        if cur is None or cur < bt0 - 1e-9 or cur > bt1:
            start = bt0
        else:
            start = cur
        # if we're lagging far behind (backlog), jump
        if cur is not None and bt1 - cur > 3 * max(win, 1e-3):
            start = bt0
        self._seg = (wall, start, bt1, interval)
        # queue spikes
        idx = _arr(s.raster_idx, dtype=np.int64)
        t = _arr(s.raster_t, dtype=np.float64)
        n = min(len(idx), len(t))
        idx, t = idx[:n], t[:n]
        ok = (idx >= 0) & (idx < self.n_neurons)
        idx, t = idx[ok], t[ok]
        if len(self._pending_t):
            idx = np.concatenate([self._pending_i, idx])
            t = np.concatenate([self._pending_t, t])
        o = np.argsort(t, kind="stable")
        self._pending_t, self._pending_i = t[o], idx[o]
        # targets
        self._tgt_region = _arr(s.rate_by_region, R)
        nt = len(NEUROTRANSMITTERS)
        self._tgt_nt = np.stack([_arr(s.rate_by_nt, nt), _arr(s.active_frac_by_nt, nt)])
        desc = s.descending if isinstance(s.descending, dict) else {}
        self._tgt_dn = np.array([float(desc.get(g, 0.0) or 0.0) for g in DESCENDING_GROUPS])
        self._tgt_dn = np.where(np.isfinite(self._tgt_dn), self._tgt_dn, 0.0)
        drive = getattr(s, "drive", None)
        self._drive_ext = drive if isinstance(drive, dict) else None
        nm = getattr(s, "neuromod", None)
        if isinstance(nm, dict) and nm.get("enabled"):
            self._nm = nm
            try:
                lv = float(nm.get("octopamine", 0.0))
            except (TypeError, ValueError):
                lv = 0.0
            self._tgt_nm = float(np.clip(lv, 0.0, 1.0)) if np.isfinite(lv) else 0.0
            if not self._show_nm:
                self._show_nm = True
                self._static = self._draw_static()
                self._disp_nm = self._tgt_nm
        # stimuli
        stims = list(getattr(s, "recent_stimuli", None) or [])
        for st in stims:
            lab = short_stimulus(st)
            if wall - self._direct_stim_wall > 5.0:
                self._markers.append((wall, lab))
                self._stims.append((wall, lab))
            elif not any(lab == l2 and wall - w2 < 5.0 for w2, l2 in self._stims):
                self._stims.append((wall, lab))
        if self.state is None:  # first state: jump instead of easing from zero
            self._disp_region = self._tgt_region.copy()
            self._disp_nt = self._tgt_nt.copy()
            self._disp_dn = self._tgt_dn.copy()
        self.state = s

    def _playhead(self, wall: float) -> float | None:
        if self._seg is None:
            return None
        w0, b0, b1, dur = self._seg
        f = min(max((wall - w0) / dur, 0.0), 1.0)
        return b0 + (b1 - b0) * f

    def _wall_of_bt(self, bt: np.ndarray) -> np.ndarray:
        w0, b0, b1, dur = self._seg
        if b1 - b0 <= 1e-12:
            return np.full(len(bt), w0 + dur)
        return w0 + np.clip((bt - b0) / (b1 - b0), 0, 1) * dur

    # ------------------------------------------------------------------ animate
    def step(self, wall: float | None = None) -> None:
        wall = time.time() if wall is None else wall
        dt = 0.0 if self._last_wall is None else max(0.0, wall - self._last_wall)
        self._last_wall = wall
        tau = float(np.clip(0.35 * (self._interval or 0.25), 0.06, 0.45))
        a = 1.0 - math.exp(-dt / tau) if dt > 0 else 0.0
        self._disp_region += a * (self._tgt_region - self._disp_region)
        self._disp_nt += a * (self._tgt_nt - self._disp_nt)
        self._disp_dn += a * (self._tgt_dn - self._disp_dn)
        self._disp_nm += a * (self._tgt_nm - self._disp_nm)
        # spike flashes decay (wall time constant)
        ftau = 0.16
        self._flash *= math.exp(-dt / ftau) if dt > 0 else 1.0
        bt = self._playhead(wall)
        if bt is not None and len(self._pending_t):
            k = int(np.searchsorted(self._pending_t, bt, side="right"))
            if k:
                t, idx = self._pending_t[:k], self._pending_i[:k]
                self._pending_t, self._pending_i = self._pending_t[k:], self._pending_i[k:]
                tw = np.minimum(self._wall_of_bt(t), wall)
                f = np.exp(-(wall - tw) / ftau).astype(np.float32)
                np.maximum.at(self._flash, idx, f)
                self._raster_chunks.append((tw, idx))
        # trim histories
        cutoff = wall - self.span - 0.5
        while self._raster_chunks and self._raster_chunks[0][0][-1] < cutoff:
            self._raster_chunks.popleft()
        if self.state is not None:
            self._dn_hist.append((wall, self._disp_dn.copy()))
            if self._show_nm:
                self._nm_hist.append((wall, self._disp_nm))
        while self._nm_hist and self._nm_hist[0][0] < cutoff:
            self._nm_hist.popleft()
            if bt is not None:
                self._bt_hist.append((wall, bt))
        while self._dn_hist and self._dn_hist[0][0] < cutoff:
            self._dn_hist.popleft()
        while self._bt_hist and self._bt_hist[0][0] < cutoff:
            self._bt_hist.popleft()
        while self._markers and self._markers[0][0] < cutoff:
            self._markers.popleft()

    def settle(self) -> None:
        """Jump animated values to their targets (for static screenshots)."""
        self._disp_region = self._tgt_region.copy()
        self._disp_nt = self._tgt_nt.copy()
        self._disp_dn = self._tgt_dn.copy()
        self._disp_nm = self._tgt_nm

    # ------------------------------------------------------------------ draw
    def draw(self, wall: float | None = None) -> np.ndarray:
        wall = time.time() if wall is None else wall
        img = self._static.copy()
        self._draw_map(img)
        self._draw_nt(img)
        self._draw_dn(img, wall)
        self._draw_raster(img, wall)
        self._draw_header(img, wall)
        return img

    def render(self, wall: float | None = None) -> np.ndarray:
        wall = time.time() if wall is None else wall
        self.step(wall)
        return self.draw(wall)

    def _draw_map(self, img) -> None:
        r = self.r_map
        view = img[r.y:r.y2, r.x:r.x2]
        # per-pixel activity level in [0, 1] (max over overlapping neuropils) ...
        act = np.zeros((r.h, r.w), np.float32)
        u = self._rate_to_unit(self._disp_region)
        for i, m in enumerate(self._masks):
            if m is None or u[i] < 0.02:
                continue
            x0, y0, mk = m
            sub = act[y0:y0 + mk.shape[0], x0:x0 + mk.shape[1]]
            np.maximum(sub, mk * np.float32(u[i]), out=sub)
        # ... coloured through the heat LUT (brightness baked in), then glow
        idx = (act * 255.0).astype(np.uint8)
        col = cv2.LUT(cv2.merge([idx, idx, idx]), _HEAT_LUT)
        out = cv2.max(view, col)
        small = cv2.resize(col, (max(1, r.w // 4), max(1, r.h // 4)),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 3.0)
        glow = cv2.resize(small, (r.w, r.h), interpolation=cv2.INTER_LINEAR)
        out = cv2.addWeighted(out, 1.0, glow, 1.1, 0.0)
        # spike flashes: NT-coloured dots, white-hot when fresh, with a soft halo
        f = self._flash
        on = np.nonzero(f > 0.04)[0]
        if len(on):
            pts = self._neuron_px[on] - (r.x, r.y)
            ff = f[on][:, None]
            core = self._neuron_col[on] * ff + 255.0 * np.clip(ff - 0.7, 0, 1) * 0.8
            layer = np.zeros_like(out)
            layer[pts[:, 1], pts[:, 0]] = np.clip(core, 0, 255).astype(np.uint8)
            halo = cv2.GaussianBlur(layer, (0, 0), 2.0)
            dil = cv2.dilate(layer, np.ones((2, 2), np.uint8))
            out = cv2.addWeighted(cv2.max(out, dil), 1.0, halo, 4.0, 0.0)
        view[:] = out
        self._draw_stim_arrows(img)
        # hot-spot labels for the most active regions
        if self.n_regions:
            order = np.argsort(-self._disp_region)[:6]
            placed: list[tuple[int, int, int, int]] = []
            for i in order:
                rate = self._disp_region[i]
                if rate < 12.0 or len(placed) >= 4:
                    break
                c = self._reg_px[i]
                if not np.isfinite(c).all():
                    continue
                s = f"{self.layout.regions[i]}  {rate:.0f} Hz"
                w = text_width(s, 11, 600) + 10
                x = int(np.clip(c[0] - w / 2, r.x + 4, r.x2 - w - 4))
                y = int(np.clip(c[1] - 26, r.y + 30, r.y2 - 40))
                box = (x - 3, y - 3, x + w + 3, y + 20)
                if any(box[0] < q[2] and q[0] < box[2] and box[1] < q[3] and q[1] < box[3]
                       for q in placed):
                    continue
                placed.append(box)
                cv2.rectangle(img, (x, y), (x + w, y + 17), (18, 14, 12), -1)
                cv2.rectangle(img, (x, y), (x + w, y + 17), (90, 170, 250), 1)
                put_text(img, s, (x + 5, y + 13), TEXT, 11, 600)

    def _draw_stim_arrows(self, img) -> None:
        """Fading arrow at the map edge pointing at the side that was just hit."""
        r = self.r_map
        wall = self._last_wall if self._last_wall is not None else 0.0
        for w, lab in list(self._stims)[-3:]:
            age = wall - w
            if age < 0 or age > 2.5:
                continue
            a = float(np.clip(1 - age / 2.5, 0, 1))
            toks = lab.split()
            side = toks[1] if len(toks) > 1 else ""
            col = tuple(int(c * a + BG[i] * (1 - a)) for i, c in enumerate(_stim_color(lab)))
            yc = r.y + r.h // 2
            if side in ("L", "R"):
                left = side == "L"
                x0 = r.x + 8 if left else r.x2 - 8
                d = 1 if left else -1
                pts = np.array([[x0, yc - 16], [x0 + d * 22, yc], [x0, yc + 16]], np.int32)
                cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
                put_text(img, toks[0], (x0 + d * 26, yc + 5), col, 12, 700,
                         "left" if left else "right")
            elif side in ("FRONT", "REAR", "TOP"):
                put_text(img, f"{toks[0]} {side}", (r.x + r.w // 2, r.y + 44), col, 12, 700,
                         "center")

    def _draw_nt(self, img) -> None:
        r, b = self.r_nt, self._nt_bar
        rates, frac = self._disp_nt
        for k, nt in enumerate(NEUROTRANSMITTERS):
            yc = int(b.y + k * self._nt_row_h + self._nt_row_h / 2)
            col = NT_COLORS_BGR[nt]
            h = max(6, int(self._nt_row_h * 0.38))
            y0 = yc - h // 2 - 4
            u = float(self._nt_unit(rates[k])) if rates[k] > 0 else 0.0
            w = int(u * b.w)
            cv2.rectangle(img, (b.x, y0), (b.x2, y0 + h), (36, 30, 28), -1)
            if w > 0:
                dimc = tuple(int(c * 0.55) for c in col)
                cv2.rectangle(img, (b.x, y0), (b.x + w, y0 + h), dimc, -1)
                cv2.rectangle(img, (b.x + max(0, w - 3), y0), (b.x + w, y0 + h), col, -1)
            # active-fraction micro bar under the rate bar
            fw = int(np.clip(frac[k], 0, 1) * b.w)
            cv2.rectangle(img, (b.x, y0 + h + 3), (b.x2, y0 + h + 5), (36, 30, 28), -1)
            if fw > 0:
                cv2.rectangle(img, (b.x, y0 + h + 3), (b.x + fw, y0 + h + 5), col, -1)
            put_text(img, f"{rates[k]:.1f} Hz" if rates[k] < 100 else f"{rates[k]:.0f} Hz",
                     (r.x2 - 112, yc + 1), TEXT, 12, 600, "right")
            put_text(img, f"{100 * frac[k]:.0f}%", (r.x2 - 14, yc + 1), col, 12, 600, "right")

    def _draw_dn(self, img, wall: float) -> None:
        idx = {g: i for i, g in enumerate(DESCENDING_GROUPS)}
        if self._dn_hist:
            tw = np.array([h[0] for h in self._dn_hist])
            vals = np.stack([h[1] for h in self._dn_hist])
        else:
            tw, vals = np.zeros(0), np.zeros((0, len(DESCENDING_GROUPS)))
        for name, gl, gr, pr in self._dn_rows:
            groups = [(gl, L_COL if gr else (GROOM_COL if gl == "groom" else ESC_COL))] + (
                [(gr, R_COL)] if gr else [])
            cols = [idx[g] for g, _ in groups]
            vmax = max(_DN_FLOOR_HZ[name], float(vals[:, cols].max()) * 1.15
                       if len(vals) else 0.0)
            put_text(img, f"{vmax:.0f} Hz", (pr.x + 4, pr.y + 11), FAINT, 9)
            for mw, lab in self._markers:
                x = int(pr.x2 - (wall - mw) / self.span * pr.w)
                if pr.x <= x <= pr.x2:
                    cv2.line(img, (x, pr.y), (x, pr.y2), (50, 50, 120), 1)
            for g, col in groups:
                if len(tw) >= 2:
                    x = pr.x2 - (wall - tw) / self.span * pr.w
                    y = pr.y2 - np.clip(vals[:, idx[g]] / vmax, 0, 1) * (pr.h - 2)
                    m = x >= pr.x
                    pts = np.stack([x[m], y[m]], 1)
                    if len(pts) >= 2:
                        cv2.polylines(img, [np.round(pts * 4).astype(np.int32)], False, col,
                                      2 if g == "escape" else 1, cv2.LINE_AA, shift=2)
            # current values
            vx = pr.x2 + 10
            for j, (g, col) in enumerate(groups):
                v = self._disp_dn[idx[g]]
                side = "" if gr is None else ("L " if j == 0 else "R ")
                put_text(img, f"{side}{v:.0f}", (vx, pr.y + 14 + 16 * j), col, 12, 600)
        self._draw_nm(img, wall)
        self._draw_drive(img)

    def _draw_nm(self, img, wall: float) -> None:
        """Octopamine level (0..1, a model quantity driven by the OA neurons' spikes
        and, if on, the modelled hit-afferent link, marked "*"): trace over the last
        ``span`` s, value, OA-neuron / hit-afferent rates and a vertical gauge."""
        pr = self._nm_row
        if pr is None:
            return
        col = NT_COLORS_BGR["OA"]
        for mw, lab in self._markers:
            x = int(pr.x2 - (wall - mw) / self.span * pr.w)
            if pr.x <= x <= pr.x2:
                cv2.line(img, (x, pr.y), (x, pr.y2), (50, 50, 120), 1)
        if len(self._nm_hist) >= 2:
            tw = np.array([h[0] for h in self._nm_hist])
            v = np.array([h[1] for h in self._nm_hist])
            x = pr.x2 - (wall - tw) / self.span * pr.w
            y = pr.y2 - np.clip(v, 0, 1) * (pr.h - 2)
            m = x >= pr.x
            if m.sum() >= 2:
                pts = np.stack([x[m], y[m]], 1)
                poly = np.vstack([pts, [[pts[-1, 0], pr.y2], [pts[0, 0], pr.y2]]])
                layer = img.copy()
                cv2.fillPoly(layer, [np.round(poly * 4).astype(np.int32)],
                             tuple(int(c * 0.45) for c in col), cv2.LINE_AA, shift=2)
                img[pr.y:pr.y2 + 1, pr.x:pr.x2 + 1] = layer[pr.y:pr.y2 + 1, pr.x:pr.x2 + 1]
                cv2.polylines(img, [np.round(pts * 4).astype(np.int32)], False, col, 2,
                              cv2.LINE_AA, shift=2)
        lv = float(np.clip(self._disp_nm, 0, 1))
        vx = pr.x2 + 10
        put_text(img, f"{lv:.2f}", (vx, pr.y + 14), col, 13, 700)
        try:
            oa = float(self._nm.get("oa_rate_hz", 0.0))
        except (TypeError, ValueError):
            oa = 0.0
        put_text(img, f"OA {oa:.1f} Hz", (vx, pr.y + 29), DIM, 9)
        if self._nm.get("runaway"):
            put_text(img, "RUNAWAY", (vx, pr.y + 41), BAD, 9, 700)
        elif self._nm.get("nociceptive_input"):
            try:
                hit = float(self._nm.get("noci_hz", 0.0))
            except (TypeError, ValueError):
                hit = 0.0
            put_text(img, f"hit {hit:.0f} Hz*", (vx, pr.y + 41), WARN, 9)
        elif self._nm.get("noci_relay"):
            put_text(img, "+hit relay*", (vx, pr.y + 41), WARN, 9)
        # vertical gauge at the panel's right edge
        gx, gw = self.r_dn.x2 - 14, 7
        cv2.rectangle(img, (gx, pr.y), (gx + gw, pr.y2), (36, 30, 28), -1)
        gy = int(pr.y2 - lv * pr.h)
        if lv > 0:
            cv2.rectangle(img, (gx, gy), (gx + gw, pr.y2), col, -1)
        cv2.rectangle(img, (gx, pr.y), (gx + gw, pr.y2), BORDER, 1)

    def preview_drive(self) -> dict:
        """Simple readout of DN rates -> body command (display only, 'preview')."""
        g = dict(zip(DESCENDING_GROUPS, self._disp_dn))
        walk = 0.5 * (g["walk_L"] + g["walk_R"])
        back = 0.5 * (g["backward_L"] + g["backward_R"])
        fwd = math.tanh((walk - 1.5 * back) / 40.0)
        turn = math.tanh((g["turn_L"] - g["turn_R"]) / 30.0)  # + = turn left
        esc = float(np.clip(g["escape"] / 60.0, 0, 1))
        return {"forward": fwd, "turn": turn, "escape": esc}

    def _draw_drive(self, img) -> None:
        d = self._dn_drive
        ext = self._drive_ext
        drv = self.preview_drive()
        if ext:
            for k in ("forward", "turn", "escape"):
                if k in ext:
                    try:
                        drv[k] = float(ext[k])
                    except (TypeError, ValueError):
                        pass
        title = "DRIVE" if ext else "DRIVE PREVIEW"
        if ext:
            sub = ("from brain, applied to the fly" if ext.get("applied") else
                   "from brain, NOT applied (run with --brain-steer)")
        else:
            sub = "simple DN mapping, not what the fly uses"
        x = put_text(img, title, (d.x + 2, d.y + 14), TEXT, 12, 700)
        put_text(img, sub, (x + 8, d.y + 14), FAINT, 10)
        bw = d.w - 84 - 40 - 52

        def bipolar(y, label, v, neg, pos, col):
            put_text(img, label, (d.x + 2, y + 9), DIM, 11, 600)
            bx = d.x + 84
            cx = bx + bw // 2
            cv2.rectangle(img, (bx, y), (bx + bw, y + 10), (36, 30, 28), -1)
            cv2.line(img, (cx, y - 2), (cx, y + 12), FAINT, 1)
            e = int(cx + np.clip(v, -1, 1) * bw / 2)
            cv2.rectangle(img, (min(cx, e), y + 1), (max(cx, e), y + 9), col, -1)
            put_text(img, neg, (bx - 4, y + 9), FAINT, 9, 400, "right")
            put_text(img, pos, (bx + bw + 5, y + 9), FAINT, 9)
            put_text(img, f"{v + 0.0:+.2f}".replace("-0.00", "+0.00"), (d.x2 - 2, y + 10),
                     TEXT, 11, 600, "right")

        bipolar(d.y + 28, "forward", drv["forward"], "back", "fwd", (150, 210, 120))
        tcol = L_COL if drv["turn"] >= 0 else R_COL
        bipolar(d.y + 50, "turn", -drv["turn"], "L", "R", tcol)
        y = d.y + 72
        put_text(img, "escape", (d.x + 2, y + 9), DIM, 11, 600)
        e = float(np.clip(drv["escape"], 0, 1))
        on = e > 0.5
        c = (int(40 + 30 * e), int(40 + 30 * e), int(60 + 195 * e))
        cv2.circle(img, (d.x + 90, y + 5), 6, c, -1, cv2.LINE_AA)
        put_text(img, "GIANT FIBRE FIRING — JUMP" if on else "quiet", (d.x + 104, y + 10),
                 ESC_COL if on else FAINT, 11, 700 if on else 400)

    def _draw_raster(self, img, wall: float) -> None:
        rr = self._rr
        for mw, lab in self._markers:
            x = int(rr.x2 - (wall - mw) / self.span * rr.w)
            if rr.x <= x <= rr.x2:
                cv2.line(img, (x, rr.y), (x, rr.y2), (60, 60, 150), 1)
                tx = min(x + 3, rr.x2 - 74)
                cv2.rectangle(img, (tx - 2, rr.y2 - 16), (tx + text_width(lab, 10, 600) + 3,
                                                          rr.y2 - 2), (20, 16, 15), -1)
                put_text(img, lab, (tx, rr.y2 - 5), (120, 120, 245), 10, 600)
        if self._raster_chunks:
            tw = np.concatenate([c[0] for c in self._raster_chunks])
            idx = np.concatenate([c[1] for c in self._raster_chunks])
            x = (rr.x2 - 1 - (wall - tw) / self.span * rr.w).astype(np.int32)
            m = x >= rr.x
            x, idx = x[m], idx[m]
            y = self._raster_row[idx]
            img[y, x] = (self._neuron_col[idx] * 0.9).astype(np.uint8)
        # playhead line
        cv2.line(img, (rr.x2 - 1, rr.y), (rr.x2 - 1, rr.y2), (90, 80, 74), 1)
        # brain time covered by the visible span
        if len(self._bt_hist) >= 2:
            covered = self._bt_hist[-1][1] - self._bt_hist[0][1]
            s = (f"{self.span:g} s wall = {covered * 1000:.0f} ms brain" if covered < 2 else
                 f"{self.span:g} s wall = {covered:.1f} s brain")
            w = text_width(s, 10, 600)
            cv2.rectangle(img, (rr.x2 - w - 14, rr.y + 2), (rr.x2 - 3, rr.y + 17), (20, 16, 15),
                          -1)
            put_text(img, s, (rr.x2 - 8, rr.y + 13), DIM, 10, 600, "right")

    def _draw_header(self, img, wall: float) -> None:
        L, s = self.layout, self.state
        put_text(img, "PerpetualFly — Brain", (14, 27), TEXT, 20, 700)
        info = f"{L.model_name}  ·  {int(L.n_neurons_total):,} neurons simulated  ·  " \
               f"{self.n_neurons:,} displayed"
        put_text(img, info, (14, 50), DIM, 12)
        # right: realtime badge
        stale = self._last_arrival is None or \
            wall - self._last_arrival > max(3.0, 3.0 * (self._interval or 1.0))
        rtf = self._rtf
        if s is None:
            badge, col = "WAITING FOR BRAIN…", (90, 84, 80)
        elif stale:
            badge, col = f"NO BRAIN DATA FOR {wall - self._last_arrival:.0f} s", BAD
        elif getattr(s, "sim_time", None) is not None:
            # paced to the fly's simulated time (the app): wall-clock speed is the
            # fly's; what matters is whether the brain keeps up with the fly
            lag = (s.drive or {}).get("lag_s") if isinstance(s.drive, dict) else None
            if lag is not None and lag > 0.3:
                badge, col = f"BRAIN {lag:.1f} s BEHIND THE FLY", (30, 120, 200)
            else:
                badge, col = "IN STEP WITH THE FLY" + (
                    f"  lag {lag:.2f} s" if lag is not None else ""), (60, 140, 60)
        elif rtf is None or rtf <= 0:
            badge, col = "SPEED UNKNOWN", (90, 84, 80)
        elif rtf >= 0.95:
            badge, col = f"REAL TIME  {rtf:.2f}×", (60, 140, 60)
        else:
            slow = 1.0 / rtf
            badge = f"BRAIN {slow:.1f}× SLOWER THAN REAL TIME" if slow < 10 else \
                f"BRAIN {slow:.0f}× SLOWER THAN REAL TIME"
            col = (30, 120, 200) if slow < 10 else (50, 60, 190)
        bw = text_width(badge, 13, 700) + 20
        bx = self.W - 14 - bw
        cv2.rectangle(img, (bx, 10), (bx + bw, 34), col, -1)
        put_text(img, badge, (bx + 10, 27), (255, 255, 255), 13, 700)
        if s is not None:
            bt = self._playhead(wall)
            bt = float(s.brain_time) if bt is None else bt
            sub = f"brain t = {bt:.3f} s"
            if rtf and rtf > 0:
                sub += f"  ·  {rtf:.3g}× real time"
            if self._interval:
                sub += f"  ·  update every {self._interval:.2g} s"
            put_text(img, sub, (self.W - 14, 52), DIM, 12, 400, "right")
            if s.window_s and s.window_s > 0 and s.total_spikes:
                sps = s.total_spikes / s.window_s
                put_text(img, f"{sps / 1e6:.2f} M spikes / brain-s" if sps >= 1e5 else
                         f"{sps:,.0f} spikes / brain-s", (self.W // 2 + 60, 52), DIM, 12)
        # stimulus chips (fade after 6 s)
        x = int(self.W * 0.40)
        put_text(img, "stimuli", (x, 27), FAINT, 11)
        x += 50
        shown = [(w, lab) for w, lab in self._stims if wall - w < 8.0][-4:]
        if not shown:
            put_text(img, "none yet", (x, 27), FAINT, 11)
        for w, lab in reversed(shown):
            age = wall - w
            fade = float(np.clip(1.0 - (age - 3.0) / 5.0, 0.25, 1.0))
            bg = tuple(int(c * fade) for c in _stim_color(lab))
            fg = tuple(int(255 * max(fade, 0.55)) for _ in range(3))
            if x + text_width(lab, 12, 700) + 14 > bx - 10:
                break
            x = _pill(img, x, 12, lab, fg, bg, 12, 700) + 6


# ============================================================================ headless
def render_frame(layout: BrainLayout, state: BrainState | list | None = None,
                 size: tuple[int, int] = DEFAULT_SIZE, interval: float = 0.25,
                 fps: float = 30.0, **kw) -> np.ndarray:
    """Render one BGR frame for ``state`` (or a list of states, oldest first).

    A synthetic clock ingests the states ``interval`` s apart and animates at ``fps``,
    so the result looks like the live window at the moment the last state has been
    fully replayed (spike flashes, raster and DN traces included).
    """
    r = BrainRenderer(layout, size=size, **kw)
    states = [] if state is None else (list(state) if isinstance(state, (list, tuple))
                                       else [state])
    wall = 1000.0
    dt = 1.0 / fps
    r._interval = interval
    r.step(wall)
    for s in states:
        r.ingest(s, wall)
        for _ in range(max(1, int(round(interval / dt)))):
            wall += dt
            r.step(wall)
    if states:
        r.settle()
    return r.draw(wall)


# ============================================================================ GUI window
def _placeholder(size, msg: str) -> np.ndarray:
    img = np.full((size[1], size[0], 3), BG, np.uint8)
    put_text(img, "PerpetualFly — Brain", (20, 36), TEXT, 20, 700)
    put_text(img, msg, (size[0] // 2, size[1] // 2), DIM, 16, 400, "center")
    return img


def _screen_info() -> tuple[float, int, int] | None:
    """(backing scale, width pt, height pt) of the main display on macOS, else None."""
    return screen_info()


def auto_display_scale(size: tuple[int, int], max_frac_w: float = 0.72,
                       max_frac_h: float = 0.85) -> float:
    """Image upscale for imshow so the window is legible but fits next to the fly window.

    OpenCV's Cocoa backend maps one image pixel to one *physical* pixel, so on a Retina
    display a 1280x800 frame shows at 640x400 pt with 6-pt text. We upscale by the
    backing scale, capped so the window uses at most ~72% of the screen width.
    Override with env ``PERPETUALFLY_BRAIN_SCALE`` (shared helper:
    ``perpetualfly.display.auto_display_scale``).
    """
    return _auto_display_scale(size, max_frac_w, max_frac_h,
                               env_var="PERPETUALFLY_BRAIN_SCALE")


class BrainWindow:
    """OpenCV window showing a ``BrainRenderer`` (call ``tick()`` from the main thread).

    ``display_scale``: None = auto (see ``auto_display_scale``); the frame is rendered
    at ``size`` and resized for display only.
    """

    def __init__(self, layout: BrainLayout | None = None, title: str = WINDOW_TITLE,
                 size: tuple[int, int] = DEFAULT_SIZE, display_scale: float | None = None,
                 **renderer_kw) -> None:
        self.title = title
        self.size = size
        self.display_scale = auto_display_scale(size) if display_scale is None \
            else float(display_scale)
        self._kw = renderer_kw
        self.renderer = BrainRenderer(layout, size, **renderer_kw) if layout else None
        cv2.namedWindow(self.title, cv2.WINDOW_AUTOSIZE)
        self._shown = False
        self.last_render_ms = 0.0

    def feed(self, item) -> None:
        if isinstance(item, BrainLayout) and self.renderer is None:
            self.renderer = BrainRenderer(item, self.size, **self._kw)
        elif self.renderer is not None:
            self.renderer.ingest(item)

    def tick(self, wait_ms: int = 1) -> list[str]:
        """Render + show one frame, pump events; returns pressed keys ('q', 'esc', ...)."""
        t0 = time.perf_counter()
        if self.renderer is None:
            frame = _placeholder(self.size, "waiting for brain layout…")
        else:
            frame = self.renderer.render()
        if abs(self.display_scale - 1.0) > 0.02:
            frame = cv2.resize(frame, None, fx=self.display_scale, fy=self.display_scale,
                               interpolation=cv2.INTER_CUBIC if self.display_scale > 1
                               else cv2.INTER_AREA)
        self.last_render_ms = (time.perf_counter() - t0) * 1000
        cv2.imshow(self.title, frame)
        self._shown = True
        keys = []
        code = cv2.waitKeyEx(wait_ms)
        while code != -1 and len(keys) < 8:
            c = code & 0xFF
            keys.append("esc" if c == 27 else chr(c).lower() if 32 <= c < 127 else "")
            code = cv2.waitKeyEx(1)
        keys = [k for k in keys if k]
        if "g" in keys and self.renderer is not None:
            self.renderer.toggle_grouping()
        return keys

    def is_open(self) -> bool:
        if not self._shown:
            return True
        try:
            return cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.title)
            for _ in range(3):
                cv2.waitKey(1)
        except cv2.error:
            pass


def run_window(state_queue, layout: BrainLayout | None = None, *,
               title: str = WINDOW_TITLE, size: tuple[int, int] = DEFAULT_SIZE,
               fps: float = 30.0, max_seconds: float | None = None,
               parent_pid: int | None = None, display_scale: float | None = None,
               **renderer_kw) -> None:
    """Window main loop; use as a ``multiprocessing`` (spawn) target.

    Drains ``state_queue`` without blocking (``BrainState``, ``StimulusEvent`` or a new
    ``BrainLayout``; ``None`` = quit). Exits when the user closes the window or presses
    q/Esc, when the parent process dies, after ``max_seconds``, or on ``None``.
    """
    import multiprocessing as mp
    import signal

    try:  # Ctrl-C in the terminal belongs to the parent; it will send None / die.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except ValueError:
        pass
    parent = mp.parent_process()
    ppid0 = parent_pid or os.getppid()
    win = BrainWindow(layout, title=title, size=size, display_scale=display_scale,
                      **renderer_kw)
    t_start = time.monotonic()
    next_parent_check = 0.0
    period = 1.0 / max(fps, 1.0)
    try:
        while True:
            t0 = time.monotonic()
            for _ in range(1000):
                try:
                    item = state_queue.get_nowait()
                except queue_mod.Empty:
                    break
                except (EOFError, OSError, ValueError):
                    return
                if item is None:
                    return
                win.feed(item)
            keys = win.tick(1)
            if "q" in keys or "esc" in keys or not win.is_open():
                return
            if max_seconds is not None and t0 - t_start > max_seconds:
                return
            if t0 >= next_parent_check:
                next_parent_check = t0 + 0.5
                if (parent is not None and not parent.is_alive()) or os.getppid() != ppid0:
                    return
            rest = period - (time.monotonic() - t0)
            if rest > 0.002:
                time.sleep(rest)
    finally:
        win.close()


class BrainWindowProcess:
    """Parent-side handle: brain window in its own (spawned) process.

    ``send()`` never blocks and silently drops when the window is gone or slow, so the
    fly app is unaffected by closing (or crashing) the brain window.
    """

    def __init__(self, layout: BrainLayout | None = None, maxsize: int = 64,
                 **window_kw) -> None:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.queue = ctx.Queue(maxsize=maxsize)
        self.queue.cancel_join_thread()  # never hang the parent at exit on a full pipe
        window_kw.setdefault("parent_pid", os.getpid())
        self.process = ctx.Process(target=run_window, args=(self.queue, layout),
                                   kwargs=window_kw, name="perpetualfly-brain-window",
                                   daemon=True)
        self.dropped = 0

    def start(self) -> "BrainWindowProcess":
        self.process.start()
        return self

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def send(self, item) -> bool:
        """Queue a BrainState / StimulusEvent / BrainLayout. False if dropped."""
        if not self.process.is_alive():
            return False
        try:
            self.queue.put_nowait(item)
            return True
        except (queue_mod.Full, ValueError, OSError):
            self.dropped += 1
            return False

    def close(self, timeout: float = 3.0) -> None:
        if self.process.is_alive():
            try:
                self.queue.put(None, timeout=0.2)
            except (queue_mod.Full, ValueError, OSError):
                pass
            self.process.join(timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1.0)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(1.0)
        try:
            self.queue.close()
        except (ValueError, OSError):
            pass

    def __enter__(self) -> "BrainWindowProcess":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()


def render_timeline(layout: BrainLayout, events, seconds: float, fps: float = 30.0,
                    size: tuple[int, int] = DEFAULT_SIZE, **kw):
    """Yield BGR frames of a scripted run: ``events`` = [(wall_s, item), ...] sorted.

    Deterministic, headless (MP4 export, tests). ``wall_s`` is relative to the start.
    """
    r = BrainRenderer(layout, size=size, **kw)
    ev = list(events)
    j = 0
    for k in range(int(round(seconds * fps))):
        wall = k / fps
        while j < len(ev) and ev[j][0] <= wall:
            r.ingest(ev[j][1], ev[j][0])
            j += 1
        yield r.render(wall)
