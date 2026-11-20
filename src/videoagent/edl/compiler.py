"""EDL -> one FFmpeg command.

The whole edit is a single `filter_complex` graph, so every frame is decoded and encoded once
and there are no intermediate files to keep in sync:

    per clip:   trim -> speed -> reframe to platform size -> fade        (video + audio)
    then:       concat all clips
    then:       burn captions + text overlays (one .ass file), overlay logos
    audio:      optional music mix under the clips
    master.mp4 -> export() does the final two-pass loudness normalisation and platform bitrate

`compile_edl` is pure apart from writing the .ass file, so the graph can be unit-tested without
running FFmpeg. `render_edl` is the end-to-end entry point.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..config import PlatformSpec, load_platforms
from ..schemas.edl import EDL
from ..tools._ffmpeg import MediaInfo, PathLike, probe, run, video_encoder_args
from ..tools.captions import (
    Layout, TextStyle, ass_filter, build_ass, caption_events, write_ass,
)
from ..tools.export import export
from ..tools.overlay import OVERLAY_STYLES, logo_filter, logo_overlays, text_overlay_events
from ..tools.reframe import reframe_graph
from ..tools.trim import atempo_chain
from .validator import EDLValidationError, validate_edl

ReframeMode = Literal["auto", "crop", "pad", "blur"]
ASPECT_TOLERANCE = 0.02  # source and target aspect ratios within 2% count as "the same"


@dataclass
class CompiledEDL:
    """Everything needed to run FFmpeg for the master render, minus the output path."""

    input_args: list[str]  # -i ... for every input, in index order
    filter_graph: str
    maps: list[str]  # -map arguments
    duration: float
    has_audio: bool
    ass_path: Path | None = None
    ass_text: str = ""
    notes: list[str] = field(default_factory=list)


def _resolve(path: str, base_dir: PathLike | None) -> str:
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / p
    return str(p)


def pick_reframe_mode(info: MediaInfo, spec: PlatformSpec, requested: ReframeMode) -> str:
    """'auto': keep the full frame (blur) when the aspect differs, otherwise nothing is lost by crop.

    Screen recordings usually have UI at the edges, so cropping 16:9 into 9:16 would cut it off.
    """
    if requested != "auto":
        return requested
    if info.height == 0:
        return "crop"
    src_aspect = info.width / info.height
    dst_aspect = spec.width / spec.height
    return "crop" if abs(src_aspect - dst_aspect) / dst_aspect <= ASPECT_TOLERANCE else "blur"


def compile_edl(
    edl: EDL,
    spec: PlatformSpec,
    infos: Mapping[str, MediaInfo],
    workdir: PathLike,
    *,
    base_dir: PathLike | None = None,
    reframe_mode: ReframeMode = "auto",
    font: str = "Arial",
    fonts_dir: PathLike | None = None,
) -> CompiledEDL:
    """Build the FFmpeg inputs and filter graph for `edl`. Writes `captions.ass` into `workdir`.

    infos: clip.source -> MediaInfo, for every source used (from ffprobe or a PerceptionDoc)
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    w, h, fps = spec.width, spec.height, spec.fps

    # Unique sources, in order of first use -> FFmpeg input index.
    sources: list[str] = []
    for clip in edl.clips:
        if clip.source not in sources:
            sources.append(clip.source)
    src_index = {s: i for i, s in enumerate(sources)}
    input_args: list[str] = []
    for s in sources:
        input_args += ["-i", _resolve(s, base_dir)]
    n_inputs = len(sources)

    any_audio = any(infos[c.source].has_audio for c in edl.clips) and edl.audio.keep_original
    graph: list[str] = []
    concat_in = ""

    # ---- per clip -------------------------------------------------------------------------
    for i, clip in enumerate(edl.clips):
        k = src_index[clip.source]
        info = infos[clip.source]
        dur = clip.duration
        speed_expr = "" if clip.speed == 1.0 else f"/{clip.speed:g}"

        graph.append(
            f"[{k}:v]trim=start={clip.start:.3f}:end={clip.end:.3f},"
            f"setpts=(PTS-STARTPTS){speed_expr}[t{i}]"
        )
        mode = pick_reframe_mode(info, spec, reframe_mode)
        graph.append(reframe_graph(w, h, mode, 0.5, fps=fps, in_label=f"t{i}", out_label=f"r{i}"))  # type: ignore[arg-type]

        fades = ""
        if clip.fade_in_s > 0:
            fades += f",fade=t=in:st=0:d={clip.fade_in_s:.3f}"
        if clip.fade_out_s > 0:
            fades += f",fade=t=out:st={dur - clip.fade_out_s:.3f}:d={clip.fade_out_s:.3f}"
        graph.append(f"[r{i}]null{fades}[v{i}]")
        concat_in += f"[v{i}]"

        if any_audio:
            if info.has_audio:
                tempo = "" if clip.speed == 1.0 else "," + ",".join(atempo_chain(clip.speed))
                a_fades = ""
                if clip.fade_in_s > 0:
                    a_fades += f",afade=t=in:st=0:d={clip.fade_in_s:.3f}"
                if clip.fade_out_s > 0:
                    a_fades += f",afade=t=out:st={dur - clip.fade_out_s:.3f}:d={clip.fade_out_s:.3f}"
                graph.append(
                    f"[{k}:a]atrim=start={clip.start:.3f}:end={clip.end:.3f},"
                    f"asetpts=PTS-STARTPTS{tempo},aresample=48000,"
                    f"aformat=channel_layouts=stereo{a_fades}[a{i}]"
                )
            else:  # keep every clip's audio the same length: silence for clips without audio
                graph.append(
                    f"anullsrc=r=48000:cl=stereo,atrim=duration={dur:.3f},asetpts=PTS-STARTPTS[a{i}]"
                )
            concat_in += f"[a{i}]"

    n = len(edl.clips)
    if any_audio:
        graph.append(f"{concat_in}concat=n={n}:v=1:a=1[vcat][acat]")
    else:
        graph.append(f"{concat_in}concat=n={n}:v=1:a=0[vcat]")
    video = "vcat"

    # ---- captions and text overlays (one .ass file) -------------------------------------
    layout = Layout(w, h, spec.caption_safe_zone_bottom_px)
    events = caption_events(edl.captions) + text_overlay_events(edl.overlays, layout)
    ass_path: Path | None = None
    ass_text = ""
    if events:
        ass_text = build_ass(
            events, layout, caption_style=TextStyle(font=font),
            extra_styles={name: _with_font(style, font) for name, style in OVERLAY_STYLES.items()},
        )
        ass_path = write_ass(workdir / "captions.ass", ass_text)
        graph.append(f"[{video}]{ass_filter(ass_path, fonts_dir)}[vtext]")
        video = "vtext"

    # ---- logos ----------------------------------------------------------------------------
    for j, ov in enumerate(logo_overlays(edl.overlays)):
        input_args += ["-loop", "1", "-i", _resolve(ov.asset or "", base_dir)]
        logo_idx = n_inputs
        n_inputs += 1
        out = f"vlogo{j}"
        graph.append(logo_filter(video, f"{logo_idx}:v", out, ov, layout))
        video = out

    graph.append(f"[{video}]null[vout]")

    # ---- audio: original clips + optional music -----------------------------------------
    total = edl.total_duration
    audio = "acat" if any_audio else None
    if audio and not edl.audio.keep_original:  # defensive; any_audio already honours keep_original
        graph.append(f"[{audio}]volume=0[amute]")
        audio = "amute"
    if edl.audio.music:
        input_args += ["-stream_loop", "-1", "-i", _resolve(edl.audio.music, base_dir)]
        music_idx = n_inputs
        n_inputs += 1
        graph.append(
            f"[{music_idx}:a]aresample=48000,aformat=channel_layouts=stereo,"
            f"volume={edl.audio.music_volume:g},atrim=duration={total:.3f}[music]"
        )
        if audio:
            graph.append(
                f"[{audio}][music]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
            )
        else:
            graph.append("[music]anull[aout]")
        audio = "aout"
    elif audio:
        graph.append(f"[{audio}]anull[aout]")
        audio = "aout"

    maps = ["-map", "[vout]"] + (["-map", f"[{audio}]"] if audio else [])
    return CompiledEDL(
        input_args=input_args,
        filter_graph=";".join(graph),
        maps=maps,
        duration=total,
        has_audio=audio is not None,
        ass_path=ass_path,
        ass_text=ass_text,
    )


def _with_font(style: TextStyle, font: str) -> TextStyle:
    from dataclasses import replace

    return replace(style, font=font)


def render_edl(
    edl: EDL,
    dst: PathLike,
    *,
    platforms_path: PathLike | None = None,
    base_dir: PathLike | None = None,
    workdir: PathLike | None = None,
    reframe_mode: ReframeMode = "auto",
    font: str = "Arial",
    fonts_dir: PathLike | None = None,
    keep_master: bool = False,
) -> Path:
    """Validate, compile and render `edl` to `dst`. Raises EDLValidationError on an invalid EDL.

    Two stages: (1) one filter_complex render to a high-quality master, (2) `export()` for
    two-pass loudness normalisation and the platform bitrate.
    """
    platforms = load_platforms(platforms_path)
    if edl.platform not in platforms:
        raise ValueError(f"unknown platform '{edl.platform}', expected one of {sorted(platforms)}")
    spec = platforms[edl.platform]

    sources = {c.source for c in edl.clips}
    infos = {s: probe(_resolve(s, base_dir)) for s in sources}
    result = validate_edl(
        edl, {s: i.duration for s, i in infos.items()}, spec, base_dir=base_dir
    )
    if not result.ok:
        raise EDLValidationError(result)

    cleanup = None
    if workdir is None:
        cleanup = tempfile.TemporaryDirectory(prefix="videoagent_")
        workdir = cleanup.name
    try:
        workdir = Path(workdir)
        compiled = compile_edl(
            edl, spec, infos, workdir,
            base_dir=base_dir, reframe_mode=reframe_mode, font=font, fonts_dir=fonts_dir,
        )
        master = workdir / "master.mp4"
        run([
            *compiled.input_args,
            "-filter_complex", compiled.filter_graph,
            *compiled.maps,
            *(["-c:a", "aac", "-b:a", "256k"] if compiled.has_audio else []),
            "-t", f"{compiled.duration:.3f}",
            *video_encoder_args(), "-movflags", "+faststart", str(master),
        ])
        out = export(master, dst, edl.platform, mode="crop", platforms_path=platforms_path,
                     lufs=edl.audio.normalize_lufs)
        if keep_master:
            keep = Path(dst).with_suffix(".master.mp4")
            keep.write_bytes(master.read_bytes())
        return out
    finally:
        if cleanup is not None:
            cleanup.cleanup()
