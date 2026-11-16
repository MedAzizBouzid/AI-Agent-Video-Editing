from __future__ import annotations

import json
from pathlib import Path

from ._ffmpeg import PathLike, ensure_parent, probe, run

TRUE_PEAK = -1.5
LRA = 11.0


def measure_loudness(src: PathLike, lufs: float = -14.0) -> dict[str, float]:
    """First loudnorm pass. Returns input_i, input_tp, input_lra, input_thresh, target_offset."""
    if not probe(src).has_audio:
        raise ValueError(f"{src} has no audio stream")
    proc = run(
        ["-i", str(src), "-vn",
         "-af", f"loudnorm=I={lufs}:TP={TRUE_PEAK}:LRA={LRA}:print_format=json",
         "-f", "null", "-"],
        loglevel="info",
    )
    text = proc.stderr
    block = json.loads(text[text.rfind("{"): text.rfind("}") + 1])
    return {k: float(v) for k, v in block.items() if k != "normalization_type"}


def loudnorm_filter(lufs: float = -14.0, measured: dict[str, float] | None = None) -> str:
    """Build a loudnorm filter string; with `measured` it runs in accurate linear mode."""
    base = f"loudnorm=I={lufs}:TP={TRUE_PEAK}:LRA={LRA}"
    if not measured:
        return base
    return (
        f"{base}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )


def normalize_loudness(src: PathLike, dst: PathLike, lufs: float = -14.0) -> Path:
    """Two-pass loudness normalisation to `lufs` integrated loudness. Video is copied untouched."""
    out = ensure_parent(dst)
    measured = measure_loudness(src, lufs)
    run([
        "-i", str(src), "-map", "0:v?", "-map", "0:a",
        "-af", loudnorm_filter(lufs, measured) + ",aresample=48000",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out),
    ])
    return out


def mix_music(
    src: PathLike,
    music: PathLike,
    dst: PathLike,
    music_volume: float = 0.15,
) -> Path:
    """Mix a music track under the video's audio. Music loops or is cut to the video length."""
    if not 0.0 <= music_volume <= 1.0:
        raise ValueError("music_volume must be between 0 and 1")
    out = ensure_parent(dst)
    info = probe(src)
    music_chain = f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={music_volume}"
    if info.has_audio:
        graph = (
            f"{music_chain}[m];[0:a]aresample=48000,aformat=channel_layouts=stereo[a0];"
            "[a0][m]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )
    else:
        graph = f"{music_chain}[aout]"
    run([
        "-i", str(src), "-stream_loop", "-1", "-i", str(music),
        "-filter_complex", graph, "-map", "0:v?", "-map", "[aout]",
        "-t", f"{info.duration:.3f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(out),
    ])
    return out
