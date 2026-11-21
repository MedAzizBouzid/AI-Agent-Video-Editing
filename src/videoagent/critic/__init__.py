"""The critic: measure the render, optionally have an LLM review it, return one CriticReport."""

from __future__ import annotations

from collections.abc import Mapping

from ..config import PlatformSpec
from ..llm.client import LLMClient
from ..schemas.brief import BriefSpec
from ..schemas.critic import CriticReport
from ..schemas.edl import EDL
from ..schemas.perception import PerceptionDoc
from ..tools._ffmpeg import PathLike
from .checks import SEVERITY, Thresholds, issues_from_checks, run_checks, score_from_checks
from .review import review_video, to_issues

LLM_WEIGHT = 0.4  # share of the final score that comes from the LLM review


def critique(
    video: PathLike,
    spec: PlatformSpec,
    *,
    platform: str,
    brief: BriefSpec | None = None,
    edl: EDL | None = None,
    docs: Mapping[str, PerceptionDoc] | None = None,
    client: LLMClient | None = None,
    ass_text: str | None = None,
    thresholds: Thresholds | None = None,
    review_frames: int = 6,
    workdir: PathLike | None = None,
) -> CriticReport:
    """Judge a rendered video.

    Always runs the deterministic checks. If `client` is given and the render is not broken, an
    LLM also reviews it against the brief (frames + the edit described as text). A render with a
    blocker is not sent to the LLM: it would only spend tokens on a file that cannot ship.

    verdict: "fail" if any blocker, "revise" if any major issue, otherwise "pass".
    """
    results = run_checks(
        video, spec,
        target_duration_s=brief.target_duration_s if brief else None,
        expect_captions=brief.captions if brief else False,
        edl=edl, ass_text=ass_text, docs=docs, thresholds=thresholds,
    )
    issues = issues_from_checks(results)
    score = score_from_checks(results)
    notes: list[str] = []
    summary, reviewed = "", False

    broken = any(not r.passed and SEVERITY[r.name] == "blocker" for r in results)
    if client is not None and not broken:
        review, notes = review_video(
            client, video, spec, platform=platform, brief=brief, edl=edl, docs=docs,
            max_frames=review_frames, workdir=workdir,
        )
        if review is not None:
            issues += to_issues(review)
            score = (1 - LLM_WEIGHT) * score + LLM_WEIGHT * review.score
            summary, reviewed = review.summary, True
    elif client is not None:
        notes.append("LLM review skipped: the render has a blocking problem")

    severities = {i.severity for i in issues}
    verdict = "fail" if "blocker" in severities else "revise" if "major" in severities else "pass"
    return CriticReport(
        checks=results, issues=issues, score=round(score, 3), verdict=verdict,
        summary=summary, llm_reviewed=reviewed, notes=notes,
    )


__all__ = ["critique", "Thresholds"]
