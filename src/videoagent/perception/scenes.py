"""Shot (scene) detection with FFmpeg's `scene` score.

For each sampled frame FFmpeg computes how different it is from the previous one (0 = identical,
1 = completely different). A score above `threshold` marks a cut. We sample at a low fps and a
small size first: cuts are still obvious, and a multi-hour recording is analysed in minutes.

Note for screen recordings: they rarely have hard cuts, so you may get one or two big shots.
That is correct, and why silences and transcript matter more than shots for this content.
"""

from __future__ import annotations

import re

from ..schemas.perception import Shot
from ..tools._ffmpeg import PathLike, probe, run

_PTS = re.compile(r"pts_time:\s*([\d.]+)")


def parse_showinfo(stderr: str) -> list[float]:
    return sorted({float(m.group(1)) for line in stderr.splitlines() if "Parsed_showinfo" in line
                   and (m := _PTS.search(line))})


def cuts_to_shots(cuts: list[float], duration: float, min_shot_s: float = 1.0) -> list[Shot]:
    """Boundaries -> contiguous shots covering [0, duration]. Cuts that would make a shot shorter
    than `min_shot_s` are dropped (flashes and fades otherwise produce many tiny shots)."""
    kept: list[float] = []
    last = 0.0
    for t in sorted(cuts):
        if t - last >= min_shot_s and duration - t >= min_shot_s:
            kept.append(t)
            last = t
    bounds = [0.0, *kept, duration]
    return [Shot(index=i, start=a, end=b) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]


def detect_shots(
    src: PathLike,
    threshold: float = 0.3,
    min_shot_s: float = 1.0,
    sample_fps: float = 4.0,
) -> list[Shot]:
    """Split `src` into shots. Lower `threshold` = more sensitive."""
    info = probe(src)
    if info.duration <= 0:
        return []
    proc = run(
        ["-i", str(src), "-an",
         "-vf", f"fps={sample_fps:g},scale=320:-2,select='gt(scene,{threshold})',showinfo",
         "-f", "null", "-"],
        loglevel="info",
    )
    return cuts_to_shots(parse_showinfo(proc.stderr), info.duration, min_shot_s)
