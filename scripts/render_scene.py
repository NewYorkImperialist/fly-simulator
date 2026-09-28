#!/usr/bin/env python
"""Render a scripted cinematic scene (docs/SCENES.md) offscreen to an MP4.

    python scripts/render_scene.py --scene temple_standoff --out runs/scenes/temple_standoff.mp4
    python scripts/render_scene.py --scene temple_standoff --out S/low.mp4 --width 480 --height 270 \
        --png-dir S/frames --png-every 6          # a quick low-res pass with PNG stills
    python scripts/render_scene.py --scene temple_standoff --out S/cut.mp4 --t0 8.5 --t1 11.5

No window is opened (MuJoCo's offscreen renderer). The audio is synthesized and
muxed with ffmpeg unless --no-audio.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_simulator.scenes import SCENES, load_scene  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="temple_standoff", choices=sorted(SCENES))
    ap.add_argument("--out", required=True, help="output MP4")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--t0", type=float, default=0.0, help="start time (s)")
    ap.add_argument("--t1", type=float, default=None, help="end time (s; default: the whole scene)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-letterbox", action="store_true", help="full 16:9 frame (no 2.39:1 bars)")
    ap.add_argument("--no-shadows", action="store_true", help="faster preview")
    ap.add_argument("--grain", type=float, default=3.0, help="film grain (0 = off)")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--png-dir", default=None, help="also write PNG stills here")
    ap.add_argument("--png-every", type=int, default=0, help="every N-th frame to --png-dir")
    ap.add_argument("--caption", default=None,
                    help="subtitle text for the lead little fly (default: the scene's own line)")
    ap.add_argument("--caption-at", type=float, nargs=2, metavar=("START", "END"), default=None,
                    help="caption start / end time in seconds (default 6.6 8.9)")
    ap.add_argument("--no-caption", action="store_true", help="no subtitle")
    a = ap.parse_args(argv)
    mod = load_scene(a.scene)
    opt = mod.RenderOptions(width=a.width, height=a.height, fps=a.fps, letterbox=not a.no_letterbox,
                            audio=not a.no_audio, shadows=not a.no_shadows, grain=a.grain, t0=a.t0,
                            t1=mod.T_END if a.t1 is None else a.t1, png_dir=a.png_dir,
                            png_every=a.png_every, crf=a.crf)
    if a.no_caption:
        opt.caption = None
    elif a.caption is not None:
        opt.caption = a.caption
    if a.caption_at is not None:
        opt.caption_t = tuple(a.caption_at)
    mod.render(a.out, opt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
