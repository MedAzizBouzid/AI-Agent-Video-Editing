from __future__ import annotations

from pathlib import Path

from ..config import load_platforms
from ._ffmpeg import PathLike, ensure_parent, probe, run, video_encoder_args
from .audio import loudnorm_filter, measure_loudness
from .reframe import Mode, reframe_graph


def export(
    src: PathLike,
    dst: PathLike,
    platform: str,
    mode: Mode = "crop",
    crop_x: float = 0.5,
    platforms_path: PathLike | None = None,
) -> Path:
    """Final render for a platform: resolution, fps, bitrate, loudness and faststart per platforms.yaml."""
    platforms = load_platforms(platforms_path)
    if platform not in platforms:
        raise ValueError(f"unknown platform '{platform}', expected one of {sorted(platforms)}")
    spec = platforms[platform]
    info = probe(src)
    out = ensure_parent(dst)

    args = [
        "-i", str(src),
        "-filter_complex",
        reframe_graph(spec.width, spec.height, mode, crop_x, fps=spec.fps),
        "-map", "[vout]",
    ]
    if info.has_audio:
        measured = measure_loudness(src, spec.loudness_lufs)
        args += [
            "-map", "0:a",
            "-af", loudnorm_filter(spec.loudness_lufs, measured) + ",aresample=48000",
            "-c:a", "aac", "-b:a", spec.audio_bitrate,
        ]
    args += video_encoder_args(spec.video_bitrate) + ["-movflags", "+faststart", str(out)]
    run(args)
    return out
