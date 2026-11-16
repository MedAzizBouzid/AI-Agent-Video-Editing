from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

from ._ffmpeg import PathLike, ensure_parent, probe, run, video_encoder_args


def concat(clips: Sequence[PathLike], dst: PathLike) -> Path:
    """Join clips in order. Everything is normalised to the first clip's size and fps.

    All clips must either have audio or all lack it.
    """
    if not clips:
        raise ValueError("concat needs at least one clip")
    out = ensure_parent(dst)
    if len(clips) == 1:
        shutil.copyfile(clips[0], out)
        return out

    infos = [probe(c) for c in clips]
    if len({i.has_audio for i in infos}) > 1:
        raise ValueError("cannot concat clips where only some have audio")
    has_audio = infos[0].has_audio
    width, height, fps = infos[0].width, infos[0].height, infos[0].fps

    args: list[str] = []
    for clip in clips:
        args += ["-i", str(clip)]

    parts: list[str] = []
    joined = ""
    for i in range(len(clips)):
        parts.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps:.3f}[v{i}]"
        )
        joined += f"[v{i}]"
        if has_audio:
            parts.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]")
            joined += f"[a{i}]"
    n_audio = 1 if has_audio else 0
    labels = "[vout][aout]" if has_audio else "[vout]"
    parts.append(f"{joined}concat=n={len(clips)}:v=1:a={n_audio}{labels}")

    args += ["-filter_complex", ";".join(parts), "-map", "[vout]"]
    if has_audio:
        args += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
    args += video_encoder_args() + ["-movflags", "+faststart", str(out)]
    run(args)
    return out
