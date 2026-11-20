"""Keyframe sampling: a few small JPEGs the critic (a vision LLM) can look at later."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..schemas.perception import Shot
from ..tools._ffmpeg import PathLike, run


def pick_timestamps(shots: Sequence[Shot], duration: float, max_frames: int) -> list[float]:
    """Pick up to `max_frames` timestamps: the middle of each shot, spread evenly if there are too
    many shots. With no shots, spread evenly over the whole file."""
    if max_frames <= 0 or duration <= 0:
        return []
    if not shots:
        return [duration * (i + 0.5) / max_frames for i in range(max_frames)]
    mids = [(s.start + s.end) / 2 for s in shots]
    if len(mids) <= max_frames:
        return mids
    step = len(mids) / max_frames
    return [mids[int(i * step + step / 2)] for i in range(max_frames)]


def sample_keyframes(
    src: PathLike,
    shots: Sequence[Shot],
    duration: float,
    out_dir: PathLike,
    max_frames: int = 12,
    width: int = 512,
) -> list[str]:
    """Write JPEGs (scaled to `width`) into `out_dir`; returns their paths in time order."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, t in enumerate(pick_timestamps(shots, duration, max_frames)):
        dst = out / f"kf_{i:03d}_{t:08.2f}.jpg"
        run(["-ss", f"{max(0.0, t):.3f}", "-i", str(src), "-frames:v", "1",
             "-vf", f"scale={width}:-2", "-q:v", "4", str(dst)])
        paths.append(str(dst))
    return paths
