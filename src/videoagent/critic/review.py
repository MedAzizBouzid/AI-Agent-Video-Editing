"""LLM review of a rendered video: does it deliver the brief?

The model sees (1) the brief, (2) a text description of the rendered edit with the speech inside
each clip, and (3) a few frames if a vision-capable provider is configured. If frames cannot be
used, the review still runs on the text alone and says so; if the LLM is unavailable entirely,
the deterministic checks stand on their own. A failure here never breaks the critique.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ..config import PlatformSpec
from ..llm.client import LLMClient
from ..llm.errors import AllProvidersFailed, StructuredOutputError
from ..llm.images import user_message_with_images
from ..llm.prompts import load_prompt
from ..perception.frames import pick_timestamps, sample_keyframes
from ..schemas.brief import BriefSpec
from ..schemas.critic import Issue
from ..schemas.edl import EDL
from ..schemas.perception import PerceptionDoc
from ..tools._ffmpeg import PathLike, probe

log = logging.getLogger("videoagent.critic")
MAX_SPEECH_CHARS_PER_CLIP = 500
TIMESTAMP_SLACK_S = 0.5


class ReviewIssue(BaseModel):
    """What the LLM may report. No "blocker": only measurements can declare a render broken."""

    severity: Literal["minor", "major"]
    timestamp_s: float | None = Field(default=None, description="seconds on the OUTPUT timeline")
    description: str
    suggested_fix: str = ""


class ReviewOutput(BaseModel):
    summary: str
    score: float = Field(ge=0, le=1)
    issues: list[ReviewIssue] = Field(default_factory=list)


def _speech_in(doc: PerceptionDoc | None, start: float, end: float) -> str:
    if doc is None:
        return ""
    words = " ".join(s.text for s in doc.transcript if s.end > start and s.start < end)
    return words if len(words) <= MAX_SPEECH_CHARS_PER_CLIP else words[:MAX_SPEECH_CHARS_PER_CLIP] + "..."


def describe_edit(
    edl: EDL, docs: Mapping[str, PerceptionDoc] | None = None
) -> str:
    """The rendered edit as text: clips on the output timeline with the speech inside each one,
    then captions and overlays. This is the model's main evidence when it cannot see frames."""
    lines = ["CLIPS (output timeline):"]
    t = 0.0
    for i, clip in enumerate(edl.clips, 1):
        speech = _speech_in((docs or {}).get(clip.source), clip.start, clip.end)
        speed = f" at {clip.speed:g}x" if clip.speed != 1 else ""
        lines.append(
            f"{i}. {t:.1f}-{t + clip.duration:.1f}s  {clip.source} "
            f"[source {clip.start:.1f}-{clip.end:.1f}s]{speed}"
            + (f' | speech: "{speech}"' if speech else " | no speech")
        )
        t += clip.duration
    lines.append("CAPTIONS:" + ("" if edl.captions else " none"))
    lines += [f"[{c.start:.1f}-{c.end:.1f}s] {c.text}" for c in edl.captions]
    lines.append("OVERLAYS:" + ("" if edl.overlays else " none"))
    for o in edl.overlays:
        what = o.text or o.asset or ""
        lines.append(f"{o.kind} {what!r} {o.start:.1f}-{o.end:.1f}s at {o.position}")
    if edl.audio.music:
        lines.append(f"MUSIC: {edl.audio.music} at volume {edl.audio.music_volume:g}")
    return "\n".join(lines)


def describe_brief(brief: BriefSpec | None, spec: PlatformSpec, platform: str, duration: float) -> str:
    lines = ["BRIEF:"]
    if brief:
        lines += [
            f"goal: {brief.goal}", f"language: {brief.language}", f"tone: {brief.tone}",
            f"captions required: {'yes' if brief.captions else 'no'}",
            f"voiceover: {'yes' if brief.voiceover else 'no'}",
        ]
        if brief.target_duration_s:
            lines.append(f"target duration: {brief.target_duration_s:g}s")
        if brief.cta:
            lines.append(f"call to action: {brief.cta}")
        if brief.notes:
            lines.append(f"notes: {brief.notes}")
    else:
        lines.append("(no brief provided: judge general quality only)")
    lines.append(f"PLATFORM: {platform} ({spec.aspect}, max {spec.max_duration_s:.0f}s)")
    lines.append(f"RENDERED DURATION: {duration:.1f}s")
    return "\n".join(lines)


def review_video(
    client: LLMClient,
    video: PathLike,
    spec: PlatformSpec,
    *,
    platform: str,
    brief: BriefSpec | None = None,
    edl: EDL | None = None,
    docs: Mapping[str, PerceptionDoc] | None = None,
    max_frames: int = 6,
    frame_width: int = 512,
    workdir: PathLike | None = None,
) -> tuple[ReviewOutput | None, list[str]]:
    """Returns (review or None, notes). None means the review could not be done; see the notes."""
    notes: list[str] = []
    duration = probe(video).duration
    text = describe_brief(brief, spec, platform, duration)
    if edl is not None:
        text += "\n\n" + describe_edit(edl, docs)

    use_vision = client.supports_vision and max_frames > 0
    if max_frames > 0 and not client.supports_vision:
        notes.append("frame review skipped: no vision-capable provider is configured")
    frames: list[str] = []
    captions: list[str] = []
    cleanup = None
    if use_vision:
        if workdir is None:
            cleanup = tempfile.TemporaryDirectory(prefix="videoagent_review_")
            workdir = cleanup.name
        times = pick_timestamps([], duration, max_frames)
        frames = sample_keyframes(video, [], duration, Path(workdir) / "frames", max_frames, frame_width)
        captions = [f"Frame {i + 1} at {t:.1f}s:" for i, t in enumerate(times)]
        text += "\n\nFRAMES: sampled evenly from the rendered video, with their timestamps."
    else:
        text += "\n\nNo frames are provided: do not comment on visuals."

    system = load_prompt("critic")
    try:
        message = user_message_with_images(text, frames, captions) if frames else \
            {"role": "user", "content": text}
        try:
            out = client.complete_json([message], ReviewOutput, system=system, tag="critic",
                                       max_repairs=1, needs_vision=use_vision)
        except AllProvidersFailed as exc:
            if not use_vision:
                raise
            notes.append(f"frame review failed ({exc}); retried on text only")
            text_only = text.split("\n\nFRAMES:")[0] + "\n\nNo frames are provided: do not comment on visuals."
            out = client.complete_json([{"role": "user", "content": text_only}], ReviewOutput,
                                       system=system, tag="critic", max_repairs=1)
    except (AllProvidersFailed, StructuredOutputError) as exc:
        log.warning("LLM review unavailable: %s", exc)
        notes.append(f"LLM review skipped: {str(exc)[:300]}")
        return None, notes
    finally:
        if cleanup is not None:
            cleanup.cleanup()

    return _sanitize(out, duration, notes), notes


def _sanitize(out: ReviewOutput, duration: float, notes: list[str]) -> ReviewOutput:
    """Guard against hallucinated issues: drop empty ones and ones pointing past the end."""
    kept: list[ReviewIssue] = []
    dropped = 0
    for issue in out.issues:
        if not issue.description.strip():
            dropped += 1
        elif issue.timestamp_s is not None and issue.timestamp_s > duration + TIMESTAMP_SLACK_S:
            dropped += 1
        else:
            if issue.timestamp_s is not None:
                issue = issue.model_copy(update={"timestamp_s": max(0.0, issue.timestamp_s)})
            kept.append(issue)
    if dropped:
        notes.append(f"{dropped} LLM issue(s) dropped: empty or timestamped past the end of the video")
    return out.model_copy(update={"issues": kept})


def to_issues(review: ReviewOutput) -> list[Issue]:
    return [Issue(severity=i.severity, timestamp_s=i.timestamp_s, description=i.description,
                  suggested_fix=i.suggested_fix, source="llm") for i in review.issues]
