"""Deterministic quality checks on a rendered video. No LLM, so they are fast, free and repeatable.

Each `check_*` returns a CheckResult. `issues_from_checks` turns failures into Issues with a
severity and a concrete fix the planner can act on.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..config import PlatformSpec
from ..perception.silence import detect_silences
from ..schemas.critic import CheckResult, Issue, Severity
from ..schemas.edl import EDL
from ..schemas.perception import PerceptionDoc
from ..tools._ffmpeg import FFmpegError, MediaInfo, PathLike, probe, run
from ..tools.audio import measure_loudness

_BLACK = re.compile(r"black_start:\s*([\d.]+)\s+black_end:\s*([\d.]+)")
_FREEZE_START = re.compile(r"freeze_start:\s*([\d.]+)")
_FREEZE_END = re.compile(r"freeze_end:\s*([\d.]+)")


@dataclass(frozen=True)
class Thresholds:
    loudness_tolerance_lu: float = 1.5  # integrated loudness may be this far from the target
    max_true_peak_db: float = -1.0  # above this, audio can clip after platform re-encoding
    black_edge_s: float = 0.5  # black this close to the start or end is a normal fade in/out
    max_black_s: float = 0.5  # total black allowed elsewhere
    max_dead_air_s: float = 3.0  # silence at least this long counts as dead air
    max_freeze_s: float = 5.0  # frozen picture at least this long
    duration_tolerance: float = 0.15  # allowed deviation from the brief's target duration
    min_duration_s: float = 1.0
    fps_tolerance: float = 0.5


# How serious a failure is, and what to do about it. "blocker": re-planning cannot fix it.
SEVERITY: dict[str, Severity] = {
    "playable": "blocker", "resolution": "blocker", "duration": "major", "audio_present": "major",
    "loudness": "major", "black_frames": "major", "dead_air": "minor", "frozen_video": "minor",
    "captions_present": "major", "caption_safe_zone": "major", "clean_cuts": "major",
}
FIX: dict[str, str] = {
    "playable": "The render is broken: re-run the render; if it persists, report a compiler bug.",
    "resolution": "Output size does not match the platform: this is a render/export bug, not a plan problem.",
    "duration": "Change the total length: add or remove clips, or trim clip start/end times.",
    "audio_present": "Keep the original audio (audio.keep_original=true) or add music or voiceover.",
    "loudness": "Re-run the export loudness normalisation; check the source audio is not silent or clipped.",
    "black_frames": "Remove or trim the clip containing black frames; check clip start/end times.",
    "dead_air": "Cut the silent stretch: split the clip around it or raise the clip speed.",
    "frozen_video": "Cut the static section, or speed it up if it is a deliberate screen pause.",
    "captions_present": "Add captions covering the speech (captions list is empty but captions are required).",
    "caption_safe_zone": "Captions overlap the platform UI zone: use the standard caption style and margins.",
    "clean_cuts": "Move the clip start/end to a sentence boundary or into a pause (see SILENCES in the footage notes).",
}


def _result(name: str, passed: bool, detail: str, value: float | None = None) -> CheckResult:
    return CheckResult(name=name, passed=passed, detail=detail, value=value)


# ---- file-level checks (from ffprobe) --------------------------------------------------------

def check_playable(info: MediaInfo) -> CheckResult:
    ok = info.width > 0 and info.height > 0 and info.duration > 0
    return _result("playable", ok, "has a video stream and a duration" if ok
                   else "no decodable video stream or zero duration", info.duration)


def check_resolution(info: MediaInfo, spec: PlatformSpec, t: Thresholds) -> CheckResult:
    size_ok = (info.width, info.height) == (spec.width, spec.height)
    fps_ok = abs(info.fps - spec.fps) <= t.fps_tolerance
    detail = (f"{info.width}x{info.height} @ {info.fps:.1f}fps, expected "
              f"{spec.width}x{spec.height} @ {spec.fps}fps")
    return _result("resolution", size_ok and fps_ok, detail)


def check_duration(
    info: MediaInfo, spec: PlatformSpec, t: Thresholds, target_s: float | None = None
) -> CheckResult:
    d = info.duration
    if d > spec.max_duration_s + 0.1:
        return _result("duration", False,
                       f"{d:.1f}s exceeds the {spec.max_duration_s:.0f}s platform maximum", d)
    if d < t.min_duration_s:
        return _result("duration", False, f"{d:.1f}s is shorter than the {t.min_duration_s}s minimum", d)
    if target_s and abs(d - target_s) > target_s * t.duration_tolerance:
        return _result("duration", False,
                       f"{d:.1f}s is more than {t.duration_tolerance:.0%} away from the target {target_s:.1f}s", d)
    return _result("duration", True, f"{d:.1f}s", d)


def check_audio_present(info: MediaInfo, expect_audio: bool = True) -> CheckResult:
    if not expect_audio:
        return _result("audio_present", True, "audio not required")
    return _result("audio_present", info.has_audio,
                   "has an audio track" if info.has_audio else "no audio track")


# ---- signal-level checks (run FFmpeg filters over the output) --------------------------------

def check_loudness(video: PathLike, spec: PlatformSpec, t: Thresholds) -> CheckResult:
    m = measure_loudness(video, spec.loudness_lufs)
    lufs, peak = m["input_i"], m["input_tp"]
    if lufs == float("-inf") or lufs < -69:
        return _result("loudness", False, "audio is silent", lufs if lufs > -1e9 else None)
    off = abs(lufs - spec.loudness_lufs)
    ok = off <= t.loudness_tolerance_lu and peak <= t.max_true_peak_db
    return _result("loudness", ok,
                   f"{lufs:.1f} LUFS (target {spec.loudness_lufs}, tolerance ±{t.loudness_tolerance_lu}), "
                   f"true peak {peak:.1f} dBTP (max {t.max_true_peak_db})", lufs)


def parse_blackdetect(stderr: str) -> list[tuple[float, float]]:
    return [(float(a), float(b)) for a, b in _BLACK.findall(stderr)]


def black_outside_edges(
    segments: Sequence[tuple[float, float]], duration: float, edge_s: float
) -> float:
    """Total black time that is not within `edge_s` of the start or end (those are fades)."""
    lo, hi = edge_s, duration - edge_s
    return sum(max(0.0, min(end, hi) - max(start, lo)) for start, end in segments)


def check_black_frames(video: PathLike, info: MediaInfo, t: Thresholds) -> CheckResult:
    proc = run(["-i", str(video), "-an", "-vf", "blackdetect=d=0.2:pix_th=0.10", "-f", "null", "-"],
               loglevel="info")
    segments = parse_blackdetect(proc.stderr)
    total = black_outside_edges(segments, info.duration, t.black_edge_s)
    if total > t.max_black_s:
        spans = ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in segments)
        return _result("black_frames", False, f"{total:.1f}s of black frames inside the video ({spans})", total)
    return _result("black_frames", True, "no unexpected black frames", total)


def check_dead_air(video: PathLike, info: MediaInfo, t: Thresholds) -> CheckResult:
    if not info.has_audio:
        return _result("dead_air", True, "no audio to check")
    # -40 dB is meant for the final, loudness-normalised render. On a very quiet file this
    # reports the whole file as silent, which is also worth knowing.
    long = detect_silences(video, -40.0, t.max_dead_air_s)
    if long:
        spans = ", ".join(f"{s.start:.1f}-{s.end:.1f}s" for s in long)
        return _result("dead_air", False, f"silence of {t.max_dead_air_s:g}s or more at {spans}",
                       max(s.duration for s in long))
    return _result("dead_air", True, "no long silences")


def parse_freezedetect(stderr: str, duration: float) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start: float | None = None
    for line in stderr.splitlines():
        if m := _FREEZE_START.search(line):
            start = float(m.group(1))
        elif (m := _FREEZE_END.search(line)) and start is not None:
            spans.append((start, float(m.group(1))))
            start = None
    if start is not None:
        spans.append((start, duration))
    return spans


def check_frozen_video(video: PathLike, info: MediaInfo, t: Thresholds) -> CheckResult:
    proc = run(["-i", str(video), "-an", "-vf", f"freezedetect=n=-60dB:d={t.max_freeze_s}",
                "-f", "null", "-"], loglevel="info")
    spans = parse_freezedetect(proc.stderr, info.duration)
    if spans:
        txt = ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in spans)
        return _result("frozen_video", False, f"picture does not change for {t.max_freeze_s:g}s+ at {txt}",
                       max(b - a for a, b in spans))
    return _result("frozen_video", True, "no long frozen sections")


# ---- plan-level checks (from the EDL and the generated .ass file) ----------------------------

def check_captions_present(edl: EDL | None, expected: bool) -> CheckResult:
    if not expected:
        return _result("captions_present", True, "captions not required")
    n = len(edl.captions) if edl else 0
    return _result("captions_present", n > 0, f"{n} captions" if n else "captions are required but there are none", n)


def check_caption_safe_zone(ass_text: str | None, spec: PlatformSpec) -> CheckResult:
    """Bottom-aligned text must keep at least `caption_safe_zone_bottom_px` clear (platform UI)."""
    need = spec.caption_safe_zone_bottom_px
    if need <= 0 or not ass_text:
        return _result("caption_safe_zone", True, "no safe zone required" if need <= 0 else "no text rendered")
    styles: dict[str, tuple[int, int]] = {}  # name -> (alignment, marginV)
    worst = None
    for line in ass_text.splitlines():
        if line.startswith("Style:"):
            f = line[len("Style:"):].split(",")
            styles[f[0].strip()] = (int(f[18]), int(f[21]))
        elif line.startswith("Dialogue:"):
            f = line[len("Dialogue:"):].split(",", 9)
            style_align, style_margin = styles.get(f[3], (2, 0))
            override = re.match(r"\{\\an(\d)\}", f[9])
            align = int(override.group(1)) if override else style_align
            margin = int(f[7]) or style_margin
            if align in (1, 2, 3) and margin < need:
                worst = margin if worst is None else min(worst, margin)
    if worst is not None:
        return _result("caption_safe_zone", False,
                       f"text sits {worst}px from the bottom; the platform UI needs {need}px clear", worst)
    return _result("caption_safe_zone", True, f"all bottom text is at least {need}px from the bottom")


def check_clean_cuts(
    edl: EDL | None, docs: Mapping[str, PerceptionDoc] | None, margin_s: float = 0.3
) -> CheckResult:
    """Flag clip boundaries that fall in the middle of a spoken sentence.

    A cut is fine at a pause (inside a detected silence) or near a segment edge. `margin_s` absorbs
    Whisper's timestamp imprecision (about ±0.3 s).
    """
    if not edl or not docs:
        return _result("clean_cuts", True, "no transcript to check against")
    problems: list[str] = []
    out_t = 0.0
    for i, clip in enumerate(edl.clips):
        doc = docs.get(clip.source)
        if doc:
            for label, t, at in (("start", clip.start, out_t), ("end", clip.end, out_t + clip.duration)):
                if any(sil.start <= t <= sil.end for sil in doc.silences):
                    continue
                for seg in doc.transcript:
                    if seg.start + margin_s < t < seg.end - margin_s:
                        problems.append(f"clips[{i}] {label} at {t:.1f}s (output {at:.1f}s) cuts through "
                                        f"\"{seg.text[:40]}\" ({seg.start:.1f}-{seg.end:.1f}s)")
                        break
        out_t += clip.duration
    if problems:
        return _result("clean_cuts", False, "; ".join(problems), float(len(problems)))
    return _result("clean_cuts", True, "all cuts land on sentence boundaries or pauses", 0.0)


# ---- orchestration ---------------------------------------------------------------------------

def run_checks(
    video: PathLike,
    spec: PlatformSpec,
    *,
    target_duration_s: float | None = None,
    expect_audio: bool = True,
    expect_captions: bool = False,
    edl: EDL | None = None,
    ass_text: str | None = None,
    docs: Mapping[str, PerceptionDoc] | None = None,
    thresholds: Thresholds | None = None,
) -> list[CheckResult]:
    """Run every applicable check. A check that cannot run is reported as failed, not skipped."""
    t = thresholds or Thresholds()
    try:
        info = probe(video)
    except FFmpegError as exc:
        return [_result("playable", False, f"cannot read the file: {str(exc)[:200]}")]

    results = [check_playable(info)]
    if not results[0].passed:
        return results  # nothing else is meaningful on a broken file

    def guarded(name: str, fn, *args) -> CheckResult:
        try:
            return fn(*args)
        except (FFmpegError, ValueError, KeyError) as exc:
            return _result(name, False, f"check could not run: {str(exc)[:200]}")

    results += [
        check_resolution(info, spec, t),
        check_duration(info, spec, t, target_duration_s),
        check_audio_present(info, expect_audio),
    ]
    if info.has_audio:
        results.append(guarded("loudness", check_loudness, video, spec, t))
        results.append(guarded("dead_air", check_dead_air, video, info, t))
    results += [
        guarded("black_frames", check_black_frames, video, info, t),
        guarded("frozen_video", check_frozen_video, video, info, t),
        check_captions_present(edl, expect_captions),
        check_caption_safe_zone(ass_text, spec),
        check_clean_cuts(edl, docs),
    ]
    return results


def issues_from_checks(results: Sequence[CheckResult]) -> list[Issue]:
    return [
        Issue(severity=SEVERITY[r.name], description=f"{r.name}: {r.detail}",
              suggested_fix=FIX[r.name], source="check")
        for r in results if not r.passed
    ]


def score_from_checks(results: Sequence[CheckResult]) -> float:
    """1.0 when everything passes; each failure subtracts its weight (blocker 3, major 2, minor 1)."""
    weight = {"blocker": 3, "major": 2, "minor": 1}
    total = sum(weight[SEVERITY[r.name]] for r in results) or 1
    lost = sum(weight[SEVERITY[r.name]] for r in results if not r.passed)
    return max(0.0, 1.0 - lost / total)
