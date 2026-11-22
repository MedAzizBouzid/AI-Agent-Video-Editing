"""Shared helpers: subprocess runner, ffprobe wrapper, encoder selection."""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

PathLike = str | os.PathLike[str]


class FFmpegError(RuntimeError):
    """ffmpeg/ffprobe failed. The message includes the command and the stderr tail."""


def _bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegError(f"{name} not found on PATH")
    return path


def run(args: list[str], *, loglevel: str = "error") -> subprocess.CompletedProcess[str]:
    """Run ffmpeg with the given args. Raises FFmpegError on a non-zero exit."""
    cmd = [_bin("ffmpeg"), "-hide_banner", "-nostdin", "-y", "-loglevel", loglevel, *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise FFmpegError(
            f"ffmpeg failed (exit {proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-2000:]}"
        )
    return proc


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


def _parse_rate(rate: str) -> float:
    num, _, den = rate.partition("/")
    try:
        return float(num) / float(den) if den and float(den) else float(num)
    except ValueError:
        return 0.0


def probe(path: PathLike) -> MediaInfo:
    proc = subprocess.run(
        [_bin("ffprobe"), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed for {path}: {proc.stderr[-1000:]}")
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    return MediaInfo(
        duration=duration,
        width=int(video["width"]) if video else 0,
        height=int(video["height"]) if video else 0,
        fps=_parse_rate(video.get("r_frame_rate", "0/1")) if video else 0.0,
        has_audio=has_audio,
    )


@functools.lru_cache(maxsize=1)
def _nvenc_works() -> bool:
    """True if a 1-frame NVENC encode succeeds (listing the encoder is not enough)."""
    try:
        run(["-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
             "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"])
        return True
    except FFmpegError:
        return False


def video_encoder_args(bitrate: str | None = None) -> list[str]:
    """H.264 encoder args. VIDEOAGENT_ENCODER = auto (default) | nvenc | libx264.

    With `bitrate` (e.g. "8M") the encode is rate-controlled; otherwise quality-based.
    """
    choice = os.environ.get("VIDEOAGENT_ENCODER", "auto").lower()
    use_nvenc = choice == "nvenc" or (choice == "auto" and _nvenc_works())
    if use_nvenc:
        args = ["-c:v", "h264_nvenc", "-preset", "p5"]
        quality = ["-cq", "23"]
    else:
        args = ["-c:v", "libx264", "-preset", "veryfast"]
        quality = ["-crf", "20"]
    if bitrate:
        value = float(bitrate.rstrip("MmKk"))
        unit = bitrate[-1]
        args += ["-b:v", bitrate, "-maxrate", bitrate, "-bufsize", f"{value * 2:g}{unit}"]
    else:
        args += quality
    return args + ["-pix_fmt", "yuv420p"]


def ensure_parent(dst: PathLike) -> Path:
    path = Path(dst)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
