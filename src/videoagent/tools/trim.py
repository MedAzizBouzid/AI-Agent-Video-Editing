from __future__ import annotations

from pathlib import Path

from ._ffmpeg import PathLike, ensure_parent, probe, run, video_encoder_args


def atempo_chain(speed: float) -> list[str]:
    """atempo only accepts 0.5-2.0 per instance, so chain filters for larger changes."""
    filters: list[str] = []
    remaining = speed
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")
    return filters


def trim(src: PathLike, dst: PathLike, start: float, end: float, speed: float = 1.0) -> Path:
    """Cut [start, end] seconds from `src` (frame-accurate, re-encoded) and optionally change speed.

    Output duration is (end - start) / speed.
    """
    if start < 0 or end <= start:
        raise ValueError(f"invalid range: start={start}, end={end}")
    if not 0.25 <= speed <= 4.0:
        raise ValueError(f"speed must be between 0.25 and 4.0, got {speed}")
    info = probe(src)
    if end > info.duration + 0.05:
        raise ValueError(f"end={end} is beyond source duration {info.duration:.2f}s")

    out = ensure_parent(dst)
    args = ["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src)]
    if speed != 1.0:
        args += ["-vf", f"setpts=PTS/{speed}"]
    if info.has_audio:
        if speed != 1.0:
            args += ["-af", ",".join(atempo_chain(speed))]
        args += ["-c:a", "aac", "-b:a", "192k"]
    else:
        args += ["-an"]
    args += video_encoder_args() + ["-movflags", "+faststart", str(out)]
    run(args)
    return out
