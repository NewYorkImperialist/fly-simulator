"""HUD and brain panel for FLY BRAIN PLAYS (OpenCV drawing on renderer frames).

``compose(frame_rgb, session, ...)`` returns one BGR image: the MuJoCo frame with the
game HUD (score, lives, wave, time, events, honest label) and, on the right, a
panel with what the brain sees (LC4 / LPLC2 drive per eye) and what it does
(DNa01/02 turn, walk, MDN, giant fibre rates, the resulting drive). FOLLOW THE
LEADER (``session.game_name == "chase"``) uses ``draw_chase_hud`` (following time,
catches, distance meter, radar) and ``chase_panel`` (LC10a drive per eye).
"""

from __future__ import annotations

import cv2
import numpy as np

from perpetualfly.games import HONEST_LABEL

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (240, 240, 240)
GREY = (150, 150, 150)
YELLOW = (60, 210, 250)
RED = (60, 60, 230)
GREEN = (110, 210, 110)
CYAN = (220, 200, 90)
ORANGE = (40, 150, 250)
PANEL_BG = (24, 20, 18)

CONTROL_LABEL = {
    "brain": "CONTROLLER: FLYWIRE BRAIN",
    "mirror": "CONTROLLER: BRAIN, EYES MIRRORED (control)",
    "none": "CONTROLLER: NONE, constant walk (control)",
}


def text(img, s, org, scale=0.5, color=WHITE, thick=1, outline=True):
    if outline:  # same thickness (Hershey advance widths depend on thickness)
        x, y = org
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (2, 2)):
            cv2.putText(img, s, (x + dx, y + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def _heart(img, cx, cy, r, color):
    cv2.circle(img, (cx - r // 2, cy), r // 2 + 1, color, -1, cv2.LINE_AA)
    cv2.circle(img, (cx + r // 2, cy), r // 2 + 1, color, -1, cv2.LINE_AA)
    pts = np.array([[cx - r, cy + 1], [cx + r, cy + 1], [cx, cy + r + 2]], np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


def _wrap(s: str, width: int) -> list[str]:
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])


def draw_hud(img: np.ndarray, session, *, high_score: int | None = None, paused: bool = False,
             recent_events: list | None = None, hint: str | None = None) -> np.ndarray:
    g = session.game
    H, W = img.shape[:2]
    # top bar (left: title / controller / stats / wave; right: score, lives)
    small = W < 800
    over = img.copy()
    cv2.rectangle(over, (0, 0), (W, 104 if small else 84), (0, 0, 0), -1)
    cv2.addWeighted(over, 0.45, img, 0.55, 0, img)
    text(img, "FLY BRAIN PLAYS: ASTEROID DODGE" if not small else "FLY BRAIN PLAYS",
         (12, 24), 0.62, YELLOW, 2)
    text(img, CONTROL_LABEL.get(session.brain.control, session.brain.control), (12, 50), 0.48,
         GREEN if session.brain.control == "brain" else ORANGE, 1)
    score = f"SCORE {int(g.score)}"
    (sw, _), _ = cv2.getTextSize(score, FONT, 0.62, 2)
    text(img, score, (W - sw - 14, 24), 0.62, WHITE, 2)
    for k in range(g.cfg.lives):
        _heart(img, W - sw - 40 - 26 * k, 18, 9, RED if k < g.lives else (70, 70, 70))
    hs = f"BEST {high_score}" if high_score is not None else ""
    wave = f"WAVE {g.wave}  {g.cfg.difficulty}  {hs}"
    stats = (f"time {g.survival_s:5.1f}s   dodged {g.dodges}   hits {g.hits}   "
             f"jumps {len(session.jump_times)}")
    text(img, stats, (12, 74), 0.45, GREY)
    if small:
        text(img, wave, (12, 96), 0.45, GREY)
    else:
        (ww, _), _ = cv2.getTextSize(wave, FONT, 0.45, 1)
        text(img, wave, (W - ww - 14, 50), 0.45, GREY)
    # event feed (last 2.5 s)
    now = g.time()
    evs = [e for e in (recent_events if recent_events is not None else g.events)
           if e.kind in ("dodge", "hit", "wave", "jump", "respawn") and now - e.t < 2.5]
    y = 120
    for e in evs[-4:]:
        col = {"dodge": GREEN, "hit": RED, "wave": YELLOW, "jump": CYAN}.get(e.kind, WHITE)
        text(img, e.text, (W // 2 - 90, y), 0.9 if e.kind in ("hit", "wave") else 0.7, col, 2)
        y += 34
    # center banners
    if g.state == "gameover":
        over = img.copy()
        cv2.rectangle(over, (W // 2 - 260, H // 2 - 80), (W // 2 + 260, H // 2 + 70), (0, 0, 0), -1)
        cv2.addWeighted(over, 0.6, img, 0.4, 0, img)
        text(img, "GAME OVER", (W // 2 - 120, H // 2 - 30), 1.3, RED, 3)
        text(img, f"score {int(g.score)}   survived {g.survival_s:.1f} s   dodged {g.dodges}",
             (W // 2 - 225, H // 2 + 10), 0.6, WHITE)
        text(img, "R restart   1/2/3 difficulty   Q quit", (W // 2 - 175, H // 2 + 45), 0.55, GREY)
    elif g.state == "ready":
        text(img, "GET READY", (W // 2 - 90, H // 2 - 60), 1.0, YELLOW, 2)
    if paused:
        text(img, "PAUSED (space)", (W // 2 - 110, H // 2 + 110), 0.9, YELLOW, 2)
    _footer(img, hint)
    return img


def _footer(img: np.ndarray, hint: str | None) -> None:
    """The honest label (and the key help) at the bottom of every game frame."""
    H, W = img.shape[:2]
    lines = _wrap(HONEST_LABEL, int(W / 7.6))
    over = img.copy()
    y0 = H - 12 - 18 * len(lines) - (18 if hint else 0)
    cv2.rectangle(over, (0, y0 - 6), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(over, 0.5, img, 0.5, 0, img)
    y = y0 + 10
    for ln in lines:
        text(img, ln, (12, y), 0.45, (210, 230, 210))
        y += 18
    if hint:
        text(img, hint, (12, y), 0.42, GREY)


def draw_chase_hud(img: np.ndarray, session, *, high_score: int | None = None,
                   paused: bool = False, recent_events: list | None = None,
                   hint: str | None = None) -> np.ndarray:
    """HUD of FOLLOW THE LEADER: score, lives, level, following time, distance meter."""
    g = session.game
    c = g.cfg
    H, W = img.shape[:2]
    small = W < 800
    over = img.copy()
    cv2.rectangle(over, (0, 0), (W, 104 if small else 84), (0, 0, 0), -1)
    cv2.addWeighted(over, 0.45, img, 0.55, 0, img)
    text(img, "FLY BRAIN PLAYS: FOLLOW THE LEADER" if not small else "FOLLOW THE LEADER",
         (12, 24), 0.62, YELLOW, 2)
    text(img, CONTROL_LABEL.get(session.brain.control, session.brain.control), (12, 50), 0.48,
         GREEN if session.brain.control == "brain" else ORANGE, 1)
    score = f"SCORE {int(g.score)}"
    (sw, _), _ = cv2.getTextSize(score, FONT, 0.62, 2)
    text(img, score, (W - sw - 14, 24), 0.62, WHITE, 2)
    for k in range(c.lives):
        _heart(img, W - sw - 40 - 26 * k, 18, 9, RED if k < g.lives else (70, 70, 70))
    hs = f"BEST {high_score}" if high_score is not None else ""
    lvl = f"LEVEL {g.level}  {c.difficulty}  {hs}"
    stats = (f"time {g.survival_s:5.1f}s   following {100 * g.follow_frac:3.0f}%   "
             f"catches {g.catches}   lost {g.losses}")
    text(img, stats, (12, 74), 0.45, GREY)
    if small:
        text(img, lvl, (12, 96), 0.45, GREY)
    else:
        (ww, _), _ = cv2.getTextSize(lvl, FONT, 0.45, 1)
        text(img, lvl, (W - ww - 14, 50), 0.45, GREY)
    # distance meter (right, under the bar): catch | follow zone | lose
    if g.state in ("playing", "trial") and np.isfinite(g.dist):
        x0, y0, mw = W - 234, (118 if small else 100), 220
        dmax = c.lose_dist_mm

        def xp(d):
            return x0 + int(mw * min(max(d, 0.0), dmax) / dmax)

        cv2.rectangle(img, (x0, y0), (x0 + mw, y0 + 10), (40, 40, 40), -1)
        cv2.rectangle(img, (xp(0), y0), (xp(c.follow_dist_mm), y0 + 10), (40, 90, 40), -1)
        cv2.rectangle(img, (xp(0), y0), (xp(c.catch_dist_mm), y0 + 10), (60, 160, 200), -1)
        cv2.rectangle(img, (x0, y0), (x0 + mw, y0 + 10), (120, 120, 120), 1)
        cv2.line(img, (xp(g.dist), y0 - 4), (xp(g.dist), y0 + 14), WHITE, 2)
        lab = "FOLLOWING" if g.following else ("TOO FAR" if g.dist > c.follow_dist_mm else "")
        text(img, f"leader {g.dist:4.1f} mm  {g.error_deg:+4.0f} deg  {lab}", (x0, y0 + 30), 0.45,
             GREEN if g.following else ORANGE)
        # radar: follower at the centre facing up, the leader as a dot (range = lose dist)
        cx, cy, rr = x0 + mw - 46, y0 + 90, 44
        cv2.circle(img, (cx, cy), rr, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), int(rr * c.follow_dist_mm / dmax), (40, 110, 40), 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), rr, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.fillPoly(img, [np.array([[cx, cy - 6], [cx - 4, cy + 4], [cx + 4, cy + 4]], np.int32)],
                     (90, 170, 230), cv2.LINE_AA)
        a = np.radians(g.error_deg)
        rd = rr * min(g.dist, dmax) / dmax
        lx, ly = int(round(cx - rd * np.sin(a))), int(round(cy - rd * np.cos(a)))
        cv2.circle(img, (lx, ly), 4, WHITE if g.following else ORANGE, -1, cv2.LINE_AA)
    now = g.time()
    evs = [e for e in (recent_events if recent_events is not None else g.events)
           if e.kind in ("catch", "lost", "level", "go", "respawn") and now - e.t < 1.5]
    y = 150 if small else 132
    for e in evs[-3:]:
        col = {"catch": GREEN, "lost": RED, "level": YELLOW, "go": YELLOW}.get(e.kind, WHITE)
        text(img, e.text, (W // 2 - 90, y), 0.9 if e.kind in ("lost", "level", "go") else 0.8,
             col, 2)
        y += 34
    if g.state == "gameover":
        over = img.copy()
        cv2.rectangle(over, (W // 2 - 260, H // 2 - 80), (W // 2 + 260, H // 2 + 70), (0, 0, 0), -1)
        cv2.addWeighted(over, 0.6, img, 0.4, 0, img)
        text(img, "GAME OVER", (W // 2 - 120, H // 2 - 30), 1.3, RED, 3)
        text(img, f"score {int(g.score)}   followed {100 * g.follow_frac:.0f}%   "
                  f"catches {g.catches}", (W // 2 - 225, H // 2 + 10), 0.6, WHITE)
        text(img, "R restart   1/2/3 difficulty   Q quit", (W // 2 - 175, H // 2 + 45), 0.55, GREY)
    elif g.state == "ready":
        text(img, "GET READY: follow the dark fly", (W // 2 - 190, H // 2 - 60), 0.9, YELLOW, 2)
    if paused:
        text(img, "PAUSED (space)", (W // 2 - 110, H // 2 + 110), 0.9, YELLOW, 2)
    _footer(img, hint)
    return img


def _bar(img, x, y, w, h, value, vmax, color, label, unit="Hz"):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 55, 50), 1)
    f = 0.0 if vmax <= 0 else max(0.0, min(1.0, value / vmax))
    if f > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + 1 + int((w - 2) * f), y + h - 1), color, -1)
    text(img, label, (x, y - 4), 0.42, GREY, 1, outline=False)
    text(img, f"{value:4.0f} {unit}", (x + w - 64, y - 4), 0.42, WHITE, 1, outline=False)


def brain_panel(session, size=(400, 640)) -> np.ndarray:
    W, H = size
    img = np.full((H, W, 3), PANEL_BG, np.uint8)
    b = session.brain
    rt = session.game.time()
    text(img, "THE BRAIN", (14, 28), 0.7, YELLOW, 2, outline=False)
    if b.control == "none":
        text(img, "disconnected (control): drive [1, 1]", (14, 52), 0.45, ORANGE, 1, outline=False)
    else:
        lag = b.lag(rt)
        st = b.latest
        rtf = f"x{st.realtime_factor:.2f}" if st is not None else "-"
        text(img, f"FlyWire v783 LIF, 138,639 neurons  lag {lag or 0:.2f}s {rtf}",
             (14, 52), 0.4, GREY, 1, outline=False)
    # --- senses
    y = 86
    text(img, "SEES (looming -> LC4 / LPLC2 Poisson Hz)", (14, y), 0.45, CYAN, 1, outline=False)
    eyes = b.eye_drive(rt)
    mir = b.control == "mirror"
    y += 30
    for eye in ("left", "right"):
        lc4, lp = eyes[eye]
        tgt = {"left": "right", "right": "left"}[eye] if mir else eye
        _bar(img, 14, y, W - 28, 12, lc4, 200, CYAN, f"{eye} eye -> {tgt} LC4")
        y += 30
        _bar(img, 14, y, W - 28, 12, lp, 200, CYAN, f"{eye} eye -> {tgt} LPLC2")
        y += 34
    # --- outputs
    r = b.rates if b.control != "none" else {}
    text(img, "DOES (descending neurons, Hz)", (14, y), 0.45, GREEN, 1, outline=False)
    y += 30
    rows = [
        ("turn L  DNa01/02 L -> steer left", r.get("turn_L", 0.0), 80, GREEN),
        ("turn R  DNa01/02 R -> steer right", r.get("turn_R", 0.0), 80, GREEN),
        ("walk  BDN2/oDN1/P9 -> speed", 0.5 * (r.get("walk_L", 0) + r.get("walk_R", 0)), 80, GREEN),
        ("MDN -> brake / back up", 0.5 * (r.get("backward_L", 0) + r.get("backward_R", 0)), 40,
         ORANGE),
        ("giant fibre DNp01 -> " + ("JUMP" if b.map.jump else "escape (jump off)"),
         r.get("escape", 0.0), 200, RED),
    ]
    for label, v, vmax, col in rows:
        _bar(img, 14, y, W - 28, 12, float(v), vmax, col, label)
        y += 32
    # --- drive
    y += 4
    d = b.drive
    text(img, f"CPG drive  L {d[0]:+.2f}  R {d[1]:+.2f}", (14, y), 0.5, WHITE, 1, outline=False)
    turn = -(d[0] - d[1]) / 2
    arrow = "<- turning left" if turn > 0.05 else ("turning right ->" if turn < -0.05 else "straight")
    text(img, arrow, (14, y + 24), 0.5, YELLOW, 1, outline=False)
    y += 52
    for ln in _wrap("Brain responses are real connectome wiring. What it sees and how its "
                    "outputs map to controls is our game interface.", 48):
        text(img, ln, (14, y), 0.4, (190, 210, 190), 1, outline=False)
        y += 17
    return img


def _dn_rows(img, b, r, y, W) -> int:
    rows = [
        ("turn L  DNa01/02 L -> steer left", r.get("turn_L", 0.0), 80, GREEN),
        ("turn R  DNa01/02 R -> steer right", r.get("turn_R", 0.0), 80, GREEN),
        ("walk  BDN2/oDN1/P9 -> speed", 0.5 * (r.get("walk_L", 0) + r.get("walk_R", 0)), 80, GREEN),
        ("MDN -> brake / back up", 0.5 * (r.get("backward_L", 0) + r.get("backward_R", 0)), 40,
         ORANGE),
        ("giant fibre DNp01 (not mapped here)", r.get("escape", 0.0), 200, RED),
    ]
    for label, v, vmax, col in rows:
        _bar(img, 14, y, W - 28, 12, float(v), vmax, col, label)
        y += 32
    return y


def chase_panel(session, size=(400, 640)) -> np.ndarray:
    """What the brain sees (the leader -> LC10a per eye) and what it does."""
    W, H = size
    img = np.full((H, W, 3), PANEL_BG, np.uint8)
    b = session.brain
    g = session.game
    rt = g.time()
    text(img, "THE BRAIN", (14, 28), 0.7, YELLOW, 2, outline=False)
    if b.control == "none":
        text(img, "disconnected (control): drive [1, 1]", (14, 52), 0.45, ORANGE, 1, outline=False)
    else:
        lag = b.lag(rt)
        st = b.latest
        rtf = f"x{st.realtime_factor:.2f}" if st is not None else "-"
        text(img, f"FlyWire v783 LIF, 138,639 neurons  lag {lag or 0:.2f}s {rtf}",
             (14, 52), 0.4, GREY, 1, outline=False)
    y = 86
    text(img, "SEES (leader -> LC10a pursuit neurons, Hz)", (14, y), 0.45, CYAN, 1, outline=False)
    y += 30
    pur = b.pursuit_drive(rt)
    mir = b.control == "mirror"
    for eye in ("left", "right"):
        tgt = {"left": "right", "right": "left"}[eye] if mir else eye
        _bar(img, 14, y, W - 28, 12, pur[eye], 150, CYAN, f"{eye} eye -> {tgt} LC10a")
        y += 32
    side = "left" if g.error_deg > 0 else "right"
    text(img, f"leader {g.dist:4.1f} mm, {abs(g.error_deg):3.0f} deg to the {side}"
         if np.isfinite(g.dist) else "leader -", (14, y), 0.45, WHITE, 1, outline=False)
    y += 30
    r = b.rates if b.control != "none" else {}
    text(img, "DOES (descending neurons, Hz)", (14, y), 0.45, GREEN, 1, outline=False)
    y += 30
    y = _dn_rows(img, b, r, y, W)
    y += 4
    d = b.drive
    text(img, f"CPG drive  L {d[0]:+.2f}  R {d[1]:+.2f}", (14, y), 0.5, WHITE, 1, outline=False)
    turn = -(d[0] - d[1]) / 2
    arrow = "<- turning left" if turn > 0.05 else ("turning right ->" if turn < -0.05 else "straight")
    text(img, arrow, (14, y + 24), 0.5, YELLOW, 1, outline=False)
    y += 52
    for ln in _wrap("Brain responses are real connectome wiring: one eye's LC10a drives the "
                    "same side's DNa01/02 (turn toward). What it sees and how its outputs "
                    "map to controls is our game interface.", 48):
        text(img, ln, (14, y), 0.4, (190, 210, 190), 1, outline=False)
        y += 17
    return img


def compose(frame_rgb: np.ndarray, session, *, panel: bool = True, **hud_kw) -> np.ndarray:
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    chase = getattr(session, "game_name", "asteroids") == "chase"
    (draw_chase_hud if chase else draw_hud)(img, session, **hud_kw)
    if not panel:
        return img
    H = img.shape[0]
    panel_fn = chase_panel if chase else brain_panel
    if H >= 600:
        p = panel_fn(session, (400, H))
    else:  # the panel needs ~600 px: draw it at 640 and scale it to fit
        p = cv2.resize(panel_fn(session, (400, 640)), (int(round(400 * H / 640)), H),
                       interpolation=cv2.INTER_AREA)
    return np.concatenate([img, p], axis=1)
