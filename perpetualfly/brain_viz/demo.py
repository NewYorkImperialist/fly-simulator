"""Brain window demo on the mock brain: live window, separate process, or MP4/PNG.

    python -m perpetualfly.brain_viz                 # live window, in this process
    python -m perpetualfly.brain_viz --process       # window in a spawned child process
    python -m perpetualfly.brain_viz --record out.mp4 --seconds 12   # headless video
    python -m perpetualfly.brain_viz --record frame.png --seconds 3  # last frame only

Keys (live, in-process): l / r = whip left / right, f = shove front, x = fall,
g = regroup raster (region <-> transmitter), q / Esc = quit.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from perpetualfly.brain.schema import StimulusEvent
from perpetualfly.brain_viz.mock import MockBrain
from perpetualfly.brain_viz.window import (
    DEFAULT_SIZE,
    BrainWindow,
    BrainWindowProcess,
    render_timeline,
)

_KEY_STIM = {"l": ("whip_hit", "left"), "r": ("whip_hit", "right"),
             "f": ("shove", "front"), "x": ("fall", "none")}


def _random_stim(rng) -> StimulusEvent:
    kind, side = [("whip_hit", "left"), ("whip_hit", "right"), ("whip_hit", "left"),
                  ("whip_hit", "right"), ("shove", "front"), ("ground_contact", "none")][
        rng.integers(6)]
    return StimulusEvent(kind, side, float(rng.uniform(0.35, 1.0)))


def mock_timeline(seconds: float, rtf: float, publish_hz: float, seed: int = 0,
                  script=None) -> tuple:
    """(layout, [(wall_s, BrainState), ...]) for a scripted mock run."""
    mb = MockBrain(realtime_factor=rtf, seed=seed)
    script = sorted(script if script is not None else [
        (1.5, StimulusEvent("whip_hit", "left", 0.8)),
        (4.5, StimulusEvent("whip_hit", "right", 0.5)),
        (7.5, StimulusEvent("shove", "front", 0.7)),
        (10.0, StimulusEvent("whip_hit", "left", 1.0)),
    ], key=lambda e: e[0])
    period = 1.0 / publish_hz
    out, j, t = [], 0, period
    while t <= seconds + 1e-9:
        while j < len(script) and script[j][0] <= t:
            mb.stimulate(script[j][1])
            j += 1
        out.append((t, mb.advance(rtf * period)))
        t += period
    return mb.layout, out


def record(path: str, seconds: float, fps: float, rtf: float, publish_hz: float,
           size, seed: int = 0) -> None:
    layout, events = mock_timeline(seconds, rtf, publish_hz, seed)
    frames = render_timeline(layout, events, seconds, fps, size=size)
    if path.lower().endswith((".png", ".jpg")):
        last = None
        for last in frames:
            pass
        cv2.imwrite(path, last)
        print(f"wrote {path}")
        return
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, tuple(size))
    n = 0
    t0 = time.perf_counter()
    for fr in frames:
        vw.write(fr)
        n += 1
    vw.release()
    print(f"wrote {path}: {n} frames, {1000 * (time.perf_counter() - t0) / max(n, 1):.1f} ms/frame")


def run_live(args) -> None:
    rng = np.random.default_rng(args.seed)
    mb = MockBrain(realtime_factor=args.rtf, seed=args.seed)
    period = 1.0 / args.publish_hz
    t0 = time.monotonic()
    next_pub, next_auto = t0 + period, t0 + 2.0
    proc = win = None
    if args.process:
        proc = BrainWindowProcess(mb.layout, size=args.size, fps=args.fps).start()
    else:
        win = BrainWindow(mb.layout, size=args.size)
    print(__doc__.split("Keys")[1].strip() if win else "window runs in a child process")
    try:
        while True:
            now = time.monotonic()
            if args.seconds and now - t0 > args.seconds:
                break
            if args.auto and now >= next_auto:
                mb.stimulate(_random_stim(rng))
                next_auto = now + rng.uniform(2.5, 5.0)
            if now >= next_pub:
                st = mb.advance(args.rtf * period)
                if proc is not None:
                    proc.send(st)
                else:
                    win.feed(st)
                next_pub = max(next_pub + period, now - period)
            if proc is not None:
                if not proc.is_alive():
                    print("brain window closed")
                    break
                time.sleep(0.01)
                continue
            keys = win.tick(1)
            for k in keys:
                if k in _KEY_STIM:
                    mb.stimulate(StimulusEvent(*_KEY_STIM[k], intensity=0.8))
            if "q" in keys or "esc" in keys or not win.is_open():
                break
            rest = 1.0 / args.fps - (time.monotonic() - now)
            if rest > 0.002:
                time.sleep(rest)
    except KeyboardInterrupt:
        pass
    finally:
        if proc is not None:
            proc.close()
        if win is not None:
            win.close()


def _size(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="PerpetualFly brain window (mock brain)")
    ap.add_argument("--mock", action="store_true", default=True,
                    help="drive the window with the mock brain (the only source for now)")
    ap.add_argument("--process", action="store_true",
                    help="open the window in a spawned child process (integration path)")
    ap.add_argument("--rtf", type=float, default=0.25,
                    help="mock brain speed, brain seconds per wall second")
    ap.add_argument("--publish-hz", type=float, default=4.0, help="BrainState rate")
    ap.add_argument("--no-auto", dest="auto", action="store_false",
                    help="no random stimuli (use keys l/r/f/x)")
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-quit after N s")
    ap.add_argument("--record", metavar="PATH", help="headless: write .mp4 (or last .png)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--size", type=_size, default=DEFAULT_SIZE, help="WxH, e.g. 1280x800")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.record:
        record(args.record, args.seconds or 12.0, args.fps, args.rtf, args.publish_hz,
               args.size, args.seed)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
