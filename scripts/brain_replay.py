"""Replay a brain recording (``--brain-record``) in the brain window, or export it.

    .venv/bin/python scripts/brain_replay.py runs/<ts>                  # viewer
    .venv/bin/python scripts/brain_replay.py runs/<ts> --info           # summary only
    .venv/bin/python scripts/brain_replay.py runs/<ts> --png 12.5 --out f.png
    .venv/bin/python scripts/brain_replay.py runs/<ts> --t0 10 --t1 16 --out clip.mp4

Viewer keys: space play/pause, left/right seek 2 s, [ ] speed, Home restart, click the
timeline bar to seek, q / Esc quit. Times are fly run times (s). docs/BRAIN_REPLAY.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perpetualfly.brain_viz.replay import (BrainRecording, event_class, render_png,  # noqa: E402
                                           render_range, run_viewer)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path, help="run directory (or its brain_rec/)")
    ap.add_argument("--info", action="store_true", help="print a summary and the events")
    ap.add_argument("--png", type=float, default=None, metavar="T",
                    help="headless: render the frame at fly time T to --out")
    ap.add_argument("--t0", type=float, default=None, help="headless range start (s)")
    ap.add_argument("--t1", type=float, default=None, help="headless range end (s)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output .png (--png), .mp4 or a directory of PNGs (--t0/--t1)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--start", type=float, default=None, help="viewer: start at this time")
    ap.add_argument("--size", type=str, default=None, help="frame size WxH (default 1280x800)")
    ap.add_argument("--no-playground", action="store_true",
                    help="classic panels (raster + transmitters) instead of the playground")
    a = ap.parse_args(argv)
    rec = BrainRecording(a.run_dir)
    s = rec.summary()
    size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
    pg = not a.no_playground
    print(f"brain recording {rec.dir}: {s['states']} states, {s['events']} events, "
          f"t {s['t_start']:.2f}-{s['t_end']:.2f} s, {s['mb']:.2f} MB"
          + (f" ({s['mb_per_min']:.2f} MB/min)" if s["mb_per_min"] else ""), flush=True)
    if a.info:
        for e in rec.events:
            extra = f" {e['rate_hz']:.0f} Hz" if "rate_hz" in e else ""
            print(f"  t={e['t']:8.3f}  {event_class(e):7s} {e.get('label', '')}{extra}")
        return 0
    if a.png is not None:
        out = a.out or Path(f"brain_replay_{a.png:.2f}.png")
        render_png(rec, a.png, out, size=size, playground=pg)
        print(f"wrote {out}")
        return 0
    if a.t0 is not None or a.t1 is not None:
        t0 = rec.t_start if a.t0 is None else a.t0
        t1 = rec.t_end if a.t1 is None else a.t1
        out = a.out or Path("brain_replay.mp4")
        n = render_range(rec, t0, t1, out, fps=a.fps, speed=a.speed, size=size, playground=pg)
        print(f"wrote {n} frames -> {out}")
        return 0
    run_viewer(rec, speed=a.speed, start=a.start, playground=pg, fps=a.fps, size=size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
