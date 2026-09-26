"""Brain playground (docs/PLAYGROUND.md): virtual optogenetics, virtual lesions and
decision meters, on top of the brain window.

* **Pure helpers** (unit-tested, no GUI): the palette ``PRESETS``, the CLI spec
  parsers ``parse_stim_specs`` (``"DNa02_L:120:1.0@3"``) and ``parse_lesion_specs``
  (``"DNp01,MDN"``), and ``decision_meters`` (DN readouts -> "what the brain is
  leaning toward", with the thresholds the body uses).
* **Drawing / hit-testing** (``PlaygroundUI``), used by ``BrainRenderer`` when
  ``playground=True``: the PALETTE panel (replaces the spike raster), the DECISION
  METERS panel (replaces the transmitter panel), map overlays (silenced neurons,
  stimulated neurons, lesioned regions) and ``click(x, y, button)`` which turns a
  mouse click into a command dict for the app::

      {"op": "stim", "target": "MDN", "rate_hz": 120.0, "duration_s": 1.0}
      {"op": "lesion", "target": "DNp01", "on": True}
      {"op": "clear_lesions"} / {"op": "stop_stim"}

  The window process puts these on its command queue; ``BrainLink`` executes them
  (so they are time-stamped in fly time and logged to events.csv).

Stimulation here is *optogenetic-style*: every targeted neuron gets Poisson input
whose events each push it over threshold, so it fires at about the chosen rate
(like a strong CsChrimson activation). It is a direct drive of the neurons, not a
natural sense. A lesion sets the neurons' spike threshold to infinity: they
integrate input but never fire, so they send no output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

OPTO_NOTE = "optogenetic stimulation (direct, not a natural sense)"


@dataclass(frozen=True)
class Preset:
    target: str   # mapping.resolve_target name
    label: str    # palette button text
    role: str     # what it does in flies / in this model
    rate_hz: float = 120.0


# palette: 4 x 4 grid (row-major)
PRESETS: tuple[Preset, ...] = (
    Preset("DNp01", "GF DNp01", "giant fibre: escape jump"),
    Preset("MDN", "MDN", "moonwalker: back up"),
    Preset("DNg100", "BDN2 DNg100", "walk forward / faster"),
    Preset("DNp09", "P9 DNp09", "walk forward"),
    Preset("DNa02_L", "DNa02 L", "steer: turn left"),
    Preset("DNa02_R", "DNa02 R", "steer: turn right"),
    Preset("turn_L", "DNa01+02 L", "both left steering DNs"),
    Preset("DNg12", "DNg12", "anterior grooming"),
    Preset("MN9", "MN9", "proboscis motor neuron"),
    Preset("sugar", "sugar GRNs", "taste -> MN9 feeding"),
    Preset("bitter", "bitter GRNs", "taste (aversive)"),
    Preset("LC4_L", "LC4 L", "looming, left eye -> GF"),
    Preset("LPLC2", "LPLC2", "looming, both eyes"),
    Preset("an_walk", "an_walk", "ascending -> walk DNs"),
    Preset("OA-VUMa1", "OA-VUMa1", "octopamine neurons"),
    Preset("jo_wind_gravity", "JO wind", "antennal wind/gravity"),
)
RATES_HZ: tuple[float, ...] = (30.0, 60.0, 120.0, 200.0)
DURATIONS_S: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)


# ----------------------------------------------------------------------------- specs
@dataclass
class StimSpec:
    target: str
    rate_hz: float = 120.0
    duration_s: float = 1.0
    at_s: float | None = None  # fly run time; None = as soon as the brain is up


def parse_stim_specs(text: str | None, default_rate: float = 120.0,
                     default_dur: float = 1.0) -> list[StimSpec]:
    """``"DNa02_L:120:1.0@3, MDN@6"`` -> StimSpecs (``TARGET[:RATE[:DURATION]][@T]``;
    ``;`` also separates entries). Raises ValueError on a malformed entry."""
    out = []
    for part in re.split(r"[,;]", text or ""):
        part = part.strip()
        if not part:
            continue
        at = None
        if "@" in part:
            part, _, t = part.rpartition("@")
            at = float(t)
        # the target itself may contain ':' (region:GNG, ids:...): numbers are
        # taken from the right
        toks = part.split(":")
        nums: list[float] = []
        while len(toks) > 1 and len(nums) < 2:
            try:
                nums.insert(0, float(toks[-1]))
            except ValueError:
                break
            toks.pop()
        target = ":".join(toks).strip()
        if not target:
            raise ValueError(f"bad --stim entry {part!r} (want TARGET[:RATE[:DUR]][@T])")
        rate = nums[0] if nums else default_rate
        dur = nums[1] if len(nums) > 1 else default_dur
        if rate <= 0 or dur <= 0 or (at is not None and at < 0):
            raise ValueError(f"bad --stim entry {part!r}: rate, duration and time must be > 0")
        out.append(StimSpec(target, float(rate), float(dur), at))
    return sorted(out, key=lambda s: -1.0 if s.at_s is None else s.at_s)


def parse_lesion_specs(text: str | None) -> list[tuple[str, float | None]]:
    """``"DNp01,MDN@3"`` -> [("DNp01", None), ("MDN", 3.0)] (None = from the start)."""
    out = []
    for part in re.split(r"[,;]", text or ""):
        part = part.strip()
        if not part:
            continue
        at = None
        if "@" in part:
            part, _, t = part.rpartition("@")
            at = float(t)
            if at < 0:
                raise ValueError(f"bad --lesion time in {part!r}")
        if not part.strip():
            raise ValueError("empty --lesion target")
        out.append((part.strip(), at))
    return out


# ----------------------------------------------------------------------------- meters
DEFAULT_THRESHOLDS = {"jump_hz": 60.0, "groom_hz": 20.0, "mn9_hz": 30.0,
                      "backward_ref_hz": 40.0, "r_ref_hz": 30.0,
                      "actions": False, "steer": False}


@dataclass
class Meter:
    key: str
    label: str
    neurons: str
    value: float          # Hz (turn: L - R Hz; arousal: level 0..1)
    vmax: float           # bar full scale
    marks: tuple          # ((value, label), ...) thresholds drawn on the bar
    lean: str             # "" or e.g. "JUMP"
    frac: float           # value / main threshold (>= 1: the body acts)
    active: bool          # does this threshold act on the body in this run?
    bipolar: bool = False


def decision_meters(descending: dict, probes: dict | None = None,
                    neuromod: dict | None = None, thr: dict | None = None) -> list[Meter]:
    """What the brain is leaning toward, from the DN readouts, against the
    thresholds the app uses (``perpetualfly.actions.brain_triggers`` for jump /
    groom / proboscis; ``mapping.descending_to_drive`` for back-up / walk / turn)."""
    t = dict(DEFAULT_THRESHOLDS)
    t.update(thr or {})
    d = {k: float(v or 0.0) for k, v in (descending or {}).items()}
    p = {k: float(v or 0.0) for k, v in (probes or {}).items()}
    acts, steer = bool(t["actions"]), bool(t["steer"])
    gf, jh = d.get("escape", 0.0), float(t["jump_hz"])
    mdn = 0.5 * (d.get("backward_L", 0.0) + d.get("backward_R", 0.0))
    bref = float(t["backward_ref_hz"])
    walk = 0.5 * (d.get("walk_L", 0.0) + d.get("walk_R", 0.0))
    rr = float(t["r_ref_hz"])
    turn = d.get("turn_L", 0.0) - d.get("turn_R", 0.0)
    groom, gh = d.get("groom", 0.0), float(t["groom_hz"])
    mn9, mh = p.get("MN9", 0.0), float(t["mn9_hz"])
    out = [
        Meter("escape", "ESCAPE", "GF DNp01", gf, max(150.0, 2.2 * jh),
              ((jh, "jump"),), "JUMP" if gf > jh else "", gf / max(jh, 1e-9), acts),
        Meter("backup", "BACK UP", "MDN", mdn, max(60.0, 1.4 * bref),
              ((0.5 * bref, "stop"), (bref, "reverse")),
              "BACK UP" if mdn >= bref else "STOP" if mdn >= 0.5 * bref else "",
              mdn / max(0.5 * bref, 1e-9), steer),
        Meter("walk", "WALK FASTER", "BDN2 oDN1 P9", walk, 90.0, ((rr, "strong"),),
              "FASTER" if walk >= rr else "", walk / max(rr, 1e-9), steer),
        Meter("turn", "TURN", "DNa01/02 L-R", turn, 90.0, ((-rr, ""), (rr, "")),
              ("LEFT" if turn > 0 else "RIGHT") if abs(turn) >= rr else "",
              abs(turn) / max(rr, 1e-9), steer, bipolar=True),
        Meter("groom", "GROOM", "DNg12", groom, max(60.0, 2.5 * gh), ((gh, "groom"),),
              "GROOM" if groom > gh else "", groom / max(gh, 1e-9), acts),
        Meter("feed", "FEED", "MN9", mn9, max(120.0, 2.5 * mh), ((mh, "extend"),),
              "EXTEND" if mn9 > mh else "", mn9 / max(mh, 1e-9), acts),
    ]
    if isinstance(neuromod, dict) and neuromod.get("enabled"):
        lv = float(np.clip(float(neuromod.get("octopamine", 0.0) or 0.0), 0.0, 1.0))
        out.append(Meter("arousal", "AROUSAL", "octopamine (model)", lv, 1.0,
                         ((0.5, "high"),), "AROUSED" if lv >= 0.5 else "", lv / 0.5, True))
    return out


# ----------------------------------------------------------------------------- UI
class PlaygroundUI:
    """Palette + meters drawing and hit-testing for ``BrainRenderer`` (imports the
    window's drawing helpers lazily, so the pure helpers above stay GUI-free)."""

    def __init__(self, renderer) -> None:
        self.r = renderer
        self.rate_hz = 120.0
        self.duration_s = 1.0
        self._hits: list[tuple[tuple[int, int, int, int], dict]] = []
        self.last_command: dict | None = None
        self._flash: tuple[float, str] | None = None  # (wall, text) click feedback

    # ................................................................ state
    @property
    def pg(self) -> dict:
        s = self.r.state
        pg = getattr(s, "playground", None) if s is not None else None
        return pg if isinstance(pg, dict) else {}

    def lesioned_targets(self) -> set[str]:
        pg = self.pg
        if isinstance(pg.get("active_lesions"), list):  # the app's list (authoritative)
            return {str(x) for x in pg["active_lesions"]}
        return {str(x.get("target")) for x in pg.get("lesions", []) if isinstance(x, dict)}

    # ................................................................ clicks
    def click(self, x: float, y: float, button: str = "left", wall: float | None = None):
        """Frame-pixel click -> command dict (or None). ``button``: left / right."""
        import time

        wall = time.time() if wall is None else wall
        for (x0, y0, x1, y1), act in reversed(self._hits if self.r.show_playground else []):
            if x0 <= x <= x1 and y0 <= y <= y1:
                return self._do(act, button, wall)
        # the brain map: a region
        r = self.r.r_map
        if r.x <= x < r.x2 and r.y <= y < r.y2:
            ri = self.r.region_at(x, y)
            if ri is not None:
                name = str(self.r.layout.regions[ri])
                return self._do({"kind": "target", "target": f"region:{name}"}, button, wall)
        return None

    def _do(self, act: dict, button: str, wall: float):
        k = act["kind"]
        cmd = None
        if k == "rate":
            self.rate_hz = act["value"]
        elif k == "dur":
            self.duration_s = act["value"]
        elif k == "stop":
            cmd = {"op": "stop_stim"}
        elif k == "clear":
            cmd = {"op": "clear_lesions"}
        elif k in ("target", "lesion", "stim"):
            tg = act["target"]
            if k == "lesion" or (k == "target" and button == "right"):
                cmd = {"op": "lesion", "target": tg, "on": tg not in self.lesioned_targets()}
            else:
                cmd = {"op": "stim", "target": tg, "rate_hz": float(self.rate_hz),
                       "duration_s": float(self.duration_s)}
        if cmd is not None:
            self.last_command = cmd
            self._flash = (wall, describe_command(cmd))
        return cmd

    # ................................................................ drawing
    def draw_static_palette(self, img) -> None:
        from .window import BORDER, DIM, _panel

        r = self.r.r_raster
        _panel(img, r, "BRAIN PLAYGROUND", OPTO_NOTE + "  ·  lesions")
        cv2 = _cv2()
        g = self._grid()
        for (x0, y0, x1, y1), _ in g:
            cv2.rectangle(img, (x0, y0), (x1, y1), (34, 28, 26), -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), BORDER, 1)
        from .window import put_text

        y_hint = max(y1 for (_, _, _, y1), _ in g) + 16 if g else r.y + 240
        put_text(img, "click a button: stimulate · right-click or LES: silence (lesion) · "
                 "map: click a region (right: lesion)", (r.x + 12, y_hint), DIM, 10)

    def _grid(self):
        r = self.r.r_raster
        cols, rows = 4, 4
        gx, gy = r.x + 10, r.y + 60
        gw = (r.w - 20) // cols
        gh = min(40, (r.h - 60 - 74) // rows)
        out = []
        for k, p in enumerate(PRESETS[: cols * rows]):
            c, rr = k % cols, k // cols
            x0, y0 = gx + c * gw, gy + rr * gh
            out.append(((x0, y0, x0 + gw - 6, y0 + gh - 5), p))
        return out

    def draw_palette(self, img, wall: float) -> None:
        from .window import BAD, DIM, FAINT, TEXT, put_text, text_width

        cv2 = _cv2()
        r = self.r.r_raster
        self._hits = []
        pg = self.pg
        les = self.lesioned_targets()
        opto_now = {str(o.get("label", "")) for o in pg.get("opto", []) if isinstance(o, dict)}
        sel = pg.get("selected")
        # controls row
        y = r.y + 32
        x = put_text(img, "rate", (r.x + 12, y + 15), DIM, 11) + 6
        for v in RATES_HZ:
            x = self._chip(img, x, y, f"{v:g} Hz", v == self.rate_hz, {"kind": "rate", "value": v})
        x = put_text(img, "duration", (x + 10, y + 15), DIM, 11) + 6
        for v in DURATIONS_S:
            x = self._chip(img, x, y, f"{v:g} s", v == self.duration_s, {"kind": "dur", "value": v})
        x2 = r.x2 - 10
        w = text_width("CLEAR LESIONS", 11, 700) + 14
        self._button(img, x2 - w, y, w, "CLEAR LESIONS", BAD, {"kind": "clear"})
        w2 = text_width("STOP STIM", 11, 700) + 14
        self._button(img, x2 - w - 6 - w2, y, w2, "STOP STIM", (200, 170, 60), {"kind": "stop"})
        # target grid
        for (x0, y0, x1, y1), p in self._grid():
            on_les = p.target in les
            on_opto = canonical_label(p.target) in opto_now
            if on_opto:
                cv2.rectangle(img, (x0, y0), (x1, y1), (110, 80, 20), -1)
            if on_les:
                cv2.rectangle(img, (x0, y0), (x1, y1), (40, 30, 90), -1)
            if sel is not None and sel == p.target:
                cv2.rectangle(img, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), (240, 240, 240), 1)
            self._hits.append(((x0, y0, x1, y1), {"kind": "target", "target": p.target}))
            put_text(img, p.label, (x0 + 6, y0 + 15), TEXT, 12, 700)
            put_text(img, p.role[:26], (x0 + 6, y0 + 29), FAINT, 9)
            bw = 30
            bx = x1 - bw - 4
            self._button(img, bx, y0 + 4, bw, "LES", BAD if on_les else (70, 60, 110),
                         {"kind": "lesion", "target": p.target}, size=9, h=15)
            if on_les:
                put_text(img, "SILENCED", (bx - 2, y1 - 5), BAD, 9, 700, "right")
            elif on_opto:
                put_text(img, "STIM ON", (bx - 2, y1 - 5), (240, 210, 90), 9, 700, "right")
        # experiment log
        self._draw_log(img, wall)

    def _draw_log(self, img, wall: float) -> None:
        from .window import DIM, TEXT, put_text

        r = self.r.r_raster
        pg = self.pg
        y = r.y + r.h - 50
        put_text(img, "EXPERIMENT LOG", (r.x + 12, y), TEXT, 10, 700)
        lines = [str(s) for s in (pg.get("log") or [])][-3:]
        if self._flash is not None and wall - self._flash[0] < 2.0 and (
                not lines or self._flash[1] not in lines[-1]):
            lines = (lines + [f"sent: {self._flash[1]} ..."])[-3:]
        for e in pg.get("errors", []) or []:
            lines = (lines + [f"! {e}"])[-3:]
        if not lines:
            lines = ["nothing yet: click a target (or --stim / --lesion, keys 9 0 -)"]
        for k, ln in enumerate(lines):
            last = k == len(lines) - 1
            put_text(img, ln[:120], (r.x + 12, y + 14 + 13 * k), TEXT if last else DIM, 10)

    def _chip(self, img, x, y, s, on, act) -> int:
        from .window import BORDER, TEXT, put_text, text_width

        cv2 = _cv2()
        w = text_width(s, 11, 600) + 12
        cv2.rectangle(img, (x, y + 2), (x + w, y + 20), (150, 110, 40) if on else (36, 30, 28), -1)
        cv2.rectangle(img, (x, y + 2), (x + w, y + 20), BORDER, 1)
        put_text(img, s, (x + 6, y + 15), TEXT, 11, 600)
        self._hits.append(((x, y + 2, x + w, y + 20), act))
        return x + w + 4

    def _button(self, img, x, y, w, s, col, act, size=11, h=18) -> None:
        from .window import put_text

        cv2 = _cv2()
        cv2.rectangle(img, (x, y + 2), (x + w, y + 2 + h), col, -1)
        put_text(img, s, (x + w // 2, y + h - (h - size) // 2), (255, 255, 255), size, 700,
                 "center")
        self._hits.append(((x, y + 2, x + w, y + 2 + h), act))

    # ................................................................ meters
    def draw_static_meters(self, img) -> None:
        from .window import _panel

        _panel(img, self.r.r_nt, "DECISION METERS", "what the brain is leaning toward")

    def draw_meters(self, img) -> None:
        from .window import BAD, DIM, FAINT, GRID, TEXT, WARN, put_text

        cv2 = _cv2()
        R = self.r
        r = R.r_nt
        s = R.state
        from perpetualfly.brain.schema import DESCENDING_GROUPS

        desc = dict(zip(DESCENDING_GROUPS, R._disp_dn))
        probes = dict(getattr(s, "probes", None) or {}) if s is not None else {}
        nm = getattr(s, "neuromod", None) if s is not None else None
        thr = self.pg.get("thresholds") or {}
        meters = decision_meters(desc, probes, nm, thr)
        top = r.y + 34
        foot = 16
        rh = min(36, (r.h - (top - r.y) - foot) / max(len(meters), 1))
        bx0, bx1 = r.x + 118, r.x2 - 146
        for k, m in enumerate(meters):
            y = int(top + k * rh)
            yc = y + int(rh * 0.45)
            over = m.frac >= 1.0
            col = BAD if (over and m.active) else WARN if over else (150, 150, 150)
            put_text(img, m.label, (r.x + 12, yc + 1), TEXT if over else DIM, 12, 700)
            put_text(img, m.neurons, (r.x + 12, yc + 14), FAINT, 9)
            cv2.rectangle(img, (bx0, yc - 6), (bx1, yc + 6), (36, 30, 28), -1)
            span = bx1 - bx0
            if m.bipolar:
                cx = bx0 + span // 2
                v = float(np.clip(m.value / m.vmax, -1, 1))
                e = int(cx - v * span / 2)  # turn left = bar to the left
                cv2.rectangle(img, (min(cx, e), yc - 5), (max(cx, e), yc + 5), col, -1)
                cv2.line(img, (cx, yc - 9), (cx, yc + 9), FAINT, 1)
                for tv, _ in m.marks:
                    tx = int(cx - tv / m.vmax * span / 2)
                    cv2.line(img, (tx, yc - 9), (tx, yc + 9), (230, 230, 230), 1)
                put_text(img, "L", (bx0 - 10, yc + 4), FAINT, 9)
                put_text(img, "R", (bx1 + 3, yc + 4), FAINT, 9)
            else:
                f = float(np.clip(m.value / m.vmax, 0, 1))
                if f > 0:
                    cv2.rectangle(img, (bx0, yc - 5), (bx0 + int(f * span), yc + 5), col, -1)
                for tv, tl in m.marks:
                    tx = int(bx0 + np.clip(tv / m.vmax, 0, 1) * span)
                    cv2.line(img, (tx, yc - 9), (tx, yc + 9), (230, 230, 230), 1)
                    if tl:
                        put_text(img, f"{tl} {tv:g}" if m.key != "arousal" else tl,
                                 (tx, yc - 10), FAINT, 8, 400, "center")
            unit = "" if m.key == "arousal" else " Hz"
            val = f"{m.value:+.0f}{unit}" if m.bipolar else (
                f"{m.value:.2f}" if m.key == "arousal" else f"{m.value:.0f}{unit}")
            put_text(img, val, (bx1 + 14, yc + 4), TEXT, 12, 600)
            if m.lean:
                put_text(img, m.lean if m.active else f"({m.lean})", (r.x2 - 10, yc + 4), col,
                         11, 700, "right")
            else:
                # how close to the threshold
                pct = int(round(100 * min(m.frac, 0.99)))
                put_text(img, f"{pct}%", (r.x2 - 10, yc + 4), FAINT, 10, 400, "right")
            cv2.line(img, (r.x + 8, int(y + rh) - 1), (r.x2 - 8, int(y + rh) - 1), GRID, 1)
        acts = bool(thr.get("actions")) if thr else False
        steer = bool(thr.get("steer")) if thr else False
        note = ("marks = body thresholds; " + ("jump/groom/feed act (--brain-actions)" if acts
                else "jump/groom/feed need --brain-actions") + ", " +
                ("walk/turn/back up steer the fly" if steer else "walk/turn/back up need --brain-steer"))
        put_text(img, note, (r.x + 12, r.y2 - 6), FAINT, 9)

    # ................................................................ map overlays
    def draw_map_overlays(self, img) -> None:
        from .window import BAD, _pill

        cv2 = _cv2()
        R = self.r
        pg = self.pg
        r = R.r_map
        # lesioned regions: red outline
        for les in pg.get("lesions", []) or []:
            lab = str(les.get("label", ""))
            if lab.startswith("region "):
                nm = lab[7:]
                if nm in R.layout.regions:
                    for o in R._reg_polys[R.layout.regions.index(nm)]:
                        cv2.polylines(img, [np.round(o * 8).astype(np.int32)], True, BAD, 2,
                                      cv2.LINE_AA, shift=3)
        for key, col in (("opto_disp", (240, 200, 60)), ("lesion_disp", BAD)):
            v = pg.get(key)
            idx = np.asarray([] if v is None else v, dtype=np.int64).ravel()
            idx = idx[(idx >= 0) & (idx < R.n_neurons)]
            ring = 4 if len(idx) <= 40 else 2
            for i in idx[:150]:
                x, y = R._neuron_px[i]
                if key == "lesion_disp":
                    cv2.line(img, (x - 3, y - 3), (x + 3, y + 3), col, 1, cv2.LINE_AA)
                    cv2.line(img, (x - 3, y + 3), (x + 3, y - 3), col, 1, cv2.LINE_AA)
                else:
                    cv2.circle(img, (int(x), int(y)), ring, col, 1, cv2.LINE_AA)
        # pills: active lesions / stimulation
        x = r.x + 12
        y = r.y + 28
        les = [str(x_.get("label", "")) for x_ in pg.get("lesions", []) or []]
        if les:
            x = _pill(img, x, y, "LESIONED: " + ", ".join(les)[:60], (255, 255, 255), BAD, 11, 700) + 6
        for o in (pg.get("opto", []) or [])[:3]:
            s = f"OPTO {o.get('label', '')} {float(o.get('rate_hz', 0)):.0f} Hz " \
                f"{float(o.get('t_left', 0)):.1f} s"
            if x + 200 > r.x2:
                break
            x = _pill(img, x, y, s, (20, 20, 20), (240, 200, 60), 11, 700) + 6


def canonical_label(target: str) -> str:
    """The worker's label for a palette target (``DNa02_L`` -> ``DNa02 L``; aliases
    as in ``mapping.TARGET_ALIASES`` for the palette's names)."""
    t = str(target)
    t = {"GF": "DNp01", "BDN2": "DNg100", "P9": "DNp09", "oDN1": "DNg97"}.get(t, t)
    if t.endswith(("_L", "_R")) and len(t) > 2:
        return f"{t[:-2]} {t[-1]}"
    return t


def describe_command(cmd: dict) -> str:
    op = cmd.get("op")
    if op == "stim":
        return (f"STIM {cmd.get('target')} {float(cmd.get('rate_hz', 0)):g} Hz "
                f"{float(cmd.get('duration_s', 0)):g} s")
    if op == "lesion":
        return f"{'LESION' if cmd.get('on') else 'UNLESION'} {cmd.get('target')}"
    if op == "clear_lesions":
        return "CLEAR LESIONS"
    if op == "stop_stim":
        return "STOP STIM"
    return str(cmd)


def _cv2():
    import cv2

    return cv2

