"""Rule-based EDL validation.

Pydantic (schemas/edl.py) already guarantees the *shape* of an EDL. This module checks what a
schema cannot: does each clip exist, are the cut points inside the source, does the result fit
the platform, do captions overlap. Every message says what is wrong AND how to fix it, because
in the agent loop these messages are fed back to the LLM so it can repair its own plan.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..config import PlatformSpec
from ..schemas.edl import EDL

# Tolerances (seconds)
EPS = 0.05  # float / frame rounding slack on time comparisons
MIN_CLIP_S = 0.2
MIN_CAPTION_S = 0.3
MAX_CAPTION_CPS = 25.0  # characters per second; faster is unreadable


class ValidationIssue(BaseModel):
    level: Literal["error", "warning"]
    code: str  # stable machine-readable id, e.g. "clip_out_of_range"
    message: str
    path: str = ""  # where in the EDL, e.g. "clips[2]"


class ValidationResult(BaseModel):
    issues: list[ValidationIssue] = []

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def format_for_llm(self) -> str:
        """A compact list the planner can act on."""
        if not self.issues:
            return "EDL is valid."
        lines = [f"- [{i.level}] {i.path + ': ' if i.path else ''}{i.message}" for i in self.issues]
        return "\n".join(lines)


class EDLValidationError(ValueError):
    def __init__(self, result: ValidationResult):
        self.result = result
        super().__init__("EDL is invalid:\n" + result.format_for_llm())


def validate_edl(
    edl: EDL,
    source_durations: Mapping[str, float],
    platform: PlatformSpec | None = None,
    *,
    base_dir: str | Path | None = None,
) -> ValidationResult:
    """Check `edl` against the known sources and (optionally) its platform spec.

    source_durations: clip.source string -> duration in seconds of that file
    platform: the PlatformSpec for edl.platform; enables the max-duration check
    base_dir: relative asset paths (logo, music) are resolved against this
    """
    issues: list[ValidationIssue] = []

    def add(level: str, code: str, message: str, path: str = "") -> None:
        issues.append(ValidationIssue(level=level, code=code, message=message, path=path))  # type: ignore[arg-type]

    if not edl.title.strip():
        add("error", "empty_title", "title is empty; give the video a short title.", "title")

    # ---- clips ----------------------------------------------------------------------------
    known = sorted(source_durations)
    for i, clip in enumerate(edl.clips):
        where = f"clips[{i}]"
        duration = source_durations.get(clip.source)
        if duration is None:
            add("error", "unknown_source",
                f"source '{clip.source}' is not one of the available files: {known}. "
                "Use one of those exact strings.", where)
            continue
        if clip.start >= duration - EPS:
            add("error", "clip_start_beyond_source",
                f"start={clip.start:.2f}s is at or past the end of '{clip.source}' "
                f"({duration:.2f}s long). Choose a start inside the file.", where)
        elif clip.end > duration + EPS:
            add("error", "clip_out_of_range",
                f"end={clip.end:.2f}s is past the end of '{clip.source}' ({duration:.2f}s long). "
                f"Set end <= {duration:.2f}.", where)
        if clip.duration < MIN_CLIP_S:
            add("error", "clip_too_short",
                f"clip lasts {clip.duration:.2f}s; the minimum is {MIN_CLIP_S}s. "
                "Lengthen it or drop it.", where)
        if clip.fade_in_s + clip.fade_out_s > clip.duration + EPS:
            add("error", "fades_too_long",
                f"fade_in_s + fade_out_s = {clip.fade_in_s + clip.fade_out_s:.2f}s exceeds the "
                f"clip duration {clip.duration:.2f}s. Shorten the fades.", where)

    total = edl.total_duration

    # ---- platform -------------------------------------------------------------------------
    if platform is not None and total > platform.max_duration_s + EPS:
        add("error", "too_long",
            f"total duration {total:.1f}s exceeds the {edl.platform} maximum of "
            f"{platform.max_duration_s:.0f}s. Remove or shorten clips by about "
            f"{total - platform.max_duration_s:.1f}s.", "clips")

    # ---- captions -------------------------------------------------------------------------
    ordered = sorted(enumerate(edl.captions), key=lambda p: p[1].start)
    previous_end = 0.0
    previous_index = -1
    for i, cap in ordered:
        where = f"captions[{i}]"
        if not cap.text.strip():
            add("error", "empty_caption", "caption text is empty; remove it.", where)
            continue
        if cap.end <= cap.start:
            add("error", "caption_bad_range",
                f"end={cap.end:.2f}s must be after start={cap.start:.2f}s.", where)
            continue
        if cap.end > total + EPS:
            add("error", "caption_beyond_video",
                f"caption ends at {cap.end:.2f}s but the video is only {total:.2f}s long. "
                "Caption times are on the OUTPUT timeline, not the source timeline.", where)
        if cap.start < previous_end - EPS and previous_index >= 0:
            add("error", "caption_overlap",
                f"starts at {cap.start:.2f}s, before captions[{previous_index}] ends "
                f"({previous_end:.2f}s). Captions must not overlap.", where)
        length = cap.end - cap.start
        if length < MIN_CAPTION_S:
            add("warning", "caption_too_brief",
                f"shown for only {length:.2f}s; viewers cannot read it.", where)
        elif len(cap.text) / length > MAX_CAPTION_CPS:
            add("warning", "caption_too_fast",
                f"{len(cap.text) / length:.0f} characters/second is too fast to read "
                f"(limit ~{MAX_CAPTION_CPS:.0f}). Show it longer or split the text.", where)
        previous_end = max(previous_end, cap.end)
        previous_index = i

    # ---- overlays -------------------------------------------------------------------------
    for i, ov in enumerate(edl.overlays):
        where = f"overlays[{i}]"
        if ov.end <= ov.start:
            add("error", "overlay_bad_range",
                f"end={ov.end:.2f}s must be after start={ov.start:.2f}s.", where)
        elif ov.end > total + EPS:
            add("error", "overlay_beyond_video",
                f"ends at {ov.end:.2f}s but the video is only {total:.2f}s long.", where)
        if ov.kind == "logo":
            if not ov.asset:
                add("error", "logo_without_asset", "a logo overlay needs `asset` (an image path).", where)
            elif not _exists(ov.asset, base_dir):
                add("error", "asset_missing", f"logo file '{ov.asset}' does not exist.", where)
        elif not (ov.text or "").strip():
            add("error", "overlay_without_text", f"a {ov.kind} overlay needs `text`.", where)

    # ---- voiceover ------------------------------------------------------------------------
    for i, line in enumerate(edl.voiceover):
        where = f"voiceover[{i}]"
        if not line.text.strip():
            add("error", "empty_voiceover", "voiceover text is empty; remove it.", where)
        if line.start > total + EPS:
            add("error", "voiceover_beyond_video",
                f"starts at {line.start:.2f}s but the video is only {total:.2f}s long.", where)
    if edl.voiceover:
        add("warning", "voiceover_not_rendered",
            "voiceover lines are accepted but not rendered yet (text-to-speech arrives in step 7).",
            "voiceover")

    # ---- audio ----------------------------------------------------------------------------
    if edl.audio.music and not _exists(edl.audio.music, base_dir):
        add("error", "asset_missing", f"music file '{edl.audio.music}' does not exist.", "audio.music")

    return ValidationResult(issues=issues)


def _exists(path: str, base_dir: str | Path | None) -> bool:
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / p
    return p.is_file()
