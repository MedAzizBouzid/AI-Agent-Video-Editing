"""Silence detection with FFmpeg's silencedetect filter."""

from __future__ import annotations

import re

from ..schemas.perception import Silence
from ..tools._ffmpeg import PathLike, probe, run

_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def parse_silencedetect(stderr: str, duration: float) -> list[Silence]:
    """Turn silencedetect log lines into Silence spans.

    A silence that is still open when the file ends has a start but no end: close it at `duration`.
    """
    spans: list[Silence] = []
    open_start: float | None = None
    for line in stderr.splitlines():
        if m := _START.search(line):
            open_start = max(0.0, float(m.group(1)))
        elif (m := _END.search(line)) and open_start is not None:
            end = min(float(m.group(1)), duration)
            if end > open_start:
                spans.append(Silence(start=open_start, end=end))
            open_start = None
    if open_start is not None and duration > open_start:
        spans.append(Silence(start=open_start, end=duration))
    return spans


def detect_silences(src: PathLike, noise_db: float = -35.0, min_duration_s: float = 0.5) -> list[Silence]:
    """Find stretches where the audio stays below `noise_db` for at least `min_duration_s`.

    Returns [] for files without audio. -35 dB suits normal speech recorded close to a mic;
    raise it (e.g. -30) for noisy rooms where background noise hides the pauses.
    """
    info = probe(src)
    if not info.has_audio:
        return []
    proc = run(
        ["-i", str(src), "-vn", "-af", f"silencedetect=noise={noise_db}dB:d={min_duration_s}",
         "-f", "null", "-"],
        loglevel="info",
    )
    return parse_silencedetect(proc.stderr, info.duration)
