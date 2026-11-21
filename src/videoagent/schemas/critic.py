from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["minor", "major", "blocker"]


class CheckResult(BaseModel):
    name: str  # e.g. loudness, duration, caption_safe_zone, black_frames
    passed: bool
    detail: str = ""
    value: float | None = None


class Issue(BaseModel):
    severity: Severity
    timestamp_s: float | None = None  # on the OUTPUT timeline
    description: str
    suggested_fix: str = ""
    source: Literal["check", "llm"] = "check"  # who found it: a measurement or the LLM reviewer


class CriticReport(BaseModel):
    """Combines deterministic checks (checks.py) and LLM review (review.py).

    verdict: "pass"   no blocker or major issue
             "revise" the plan can be fixed by editing the EDL (too long, bad cut, missing captions)
             "fail"   something is broken that re-planning cannot fix (unplayable, wrong size)
    """

    checks: list[CheckResult] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0, le=1)
    verdict: Literal["pass", "revise", "fail"] = "revise"
    summary: str = ""
    llm_reviewed: bool = False
    notes: list[str] = Field(default_factory=list)  # e.g. "frame review skipped: no vision provider"

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def format_for_planner(self) -> str:
        """The text sent back to the planner LLM when the verdict is not "pass"."""
        lines = [f"Verdict: {self.verdict} (score {self.score:.2f})"]
        if self.summary:
            lines.append(f"Summary: {self.summary}")
        order = {"blocker": 0, "major": 1, "minor": 2}
        for issue in sorted(self.issues, key=lambda i: (order[i.severity], i.timestamp_s or 0)):
            at = f" @{issue.timestamp_s:.1f}s" if issue.timestamp_s is not None else ""
            fix = f" -> Fix: {issue.suggested_fix}" if issue.suggested_fix else ""
            lines.append(f"- [{issue.severity}]{at} {issue.description}{fix}")
        if not self.issues:
            lines.append("- no issues found")
        return "\n".join(lines)
