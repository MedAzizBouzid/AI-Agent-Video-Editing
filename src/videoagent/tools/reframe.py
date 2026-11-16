from __future__ import annotations

from pathlib import Path
from typing import Literal

from ._ffmpeg import PathLike, ensure_parent, run, video_encoder_args

Mode = Literal["crop", "pad", "blur"]


def reframe_graph(
    width: int,
    height: int,
    mode: Mode = "crop",
    crop_x: float = 0.5,
    fps: float | None = None,
    in_label: str = "0:v",
    out_label: str = "vout",
) -> str:
    """Build a filter_complex fragment that turns `[in_label]` into `[out_label]` at width x height.

    crop: fill the frame, cutting the overflow (crop_x 0..1 picks the horizontal position)
    pad:  fit inside the frame with black bars
    blur: fit inside the frame over a blurred, enlarged copy of itself (good for 16:9 -> 9:16)
    """
    if width % 2 or height % 2 or width <= 0 or height <= 0:
        raise ValueError("width and height must be positive even numbers")
    if not 0.0 <= crop_x <= 1.0:
        raise ValueError("crop_x must be between 0 and 1")
    tail = f",fps={fps:g}" if fps else ""
    w, h = width, height
    if mode == "crop":
        return (
            f"[{in_label}]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h}:(iw-ow)*{crop_x:.4f}:(ih-oh)/2,setsar=1{tail}[{out_label}]"
        )
    if mode == "pad":
        return (
            f"[{in_label}]scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1{tail}[{out_label}]"
        )
    if mode == "blur":
        o = out_label
        return (
            f"[{in_label}]split=2[{o}_bg][{o}_fg];"
            f"[{o}_bg]scale={w}:{h}:force_original_aspect_ratio=increase,"
            f"crop={w}:{h},boxblur=20:2[{o}_bgb];"
            f"[{o}_fg]scale={w}:{h}:force_original_aspect_ratio=decrease[{o}_fgs];"
            f"[{o}_bgb][{o}_fgs]overlay=(W-w)/2:(H-h)/2,setsar=1{tail}[{out_label}]"
        )
    raise ValueError(f"unknown reframe mode: {mode}")


def reframe(
    src: PathLike,
    dst: PathLike,
    width: int,
    height: int,
    mode: Mode = "crop",
    crop_x: float = 0.5,
) -> Path:
    """Re-render `src` at width x height using the chosen mode. Audio is kept as is."""
    out = ensure_parent(dst)
    graph = reframe_graph(width, height, mode, crop_x)
    args = [
        "-i", str(src), "-filter_complex", graph,
        "-map", "[vout]", "-map", "0:a?", "-c:a", "aac", "-b:a", "192k",
        *video_encoder_args(), "-movflags", "+faststart", str(out),
    ]
    run(args)
    return out
