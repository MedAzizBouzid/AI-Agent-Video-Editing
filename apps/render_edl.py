"""Render a hand-written EDL JSON file.

    python -m apps.render_edl examples/demo.edl.json out.mp4 --font "Arial"

Paths inside the EDL (clip sources, logo, music) are resolved relative to the EDL file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from videoagent.console import use_utf8_console
from videoagent.edl import EDLValidationError, render_edl
from videoagent.schemas import EDL


def main(argv: list[str] | None = None) -> int:
    use_utf8_console()
    p = argparse.ArgumentParser(description="Render an EDL JSON file to a video")
    p.add_argument("edl", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--font", default="Arial", help="caption font (needs Arabic glyphs for Arabic)")
    p.add_argument("--fonts-dir", type=Path, default=None)
    p.add_argument("--reframe", choices=["auto", "crop", "pad", "blur"], default="auto")
    args = p.parse_args(argv)

    edl = EDL(**json.loads(args.edl.read_text(encoding="utf-8")))
    try:
        out = render_edl(
            edl, args.output, base_dir=args.edl.parent, font=args.font,
            fonts_dir=args.fonts_dir, reframe_mode=args.reframe,
        )
    except EDLValidationError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
