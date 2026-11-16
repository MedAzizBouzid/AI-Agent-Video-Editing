from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CheckResult(BaseModel):
    name: str  # e.g. loudness, duration, caption_safe_zone, black_frames
    passed: bool
    detail: str = ""
    value: float | None = None


class Issue(BaseModel):
    severity: Literal["minor", "major", "blocker"]
    timestamp_s: float | None = None
    description: str
    suggested_fix: str = ""


class CriticReport(BaseModel):
    """Combines deterministic checks (checks.py) and LLM review (review.py)."""

    checks: list[CheckResult] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0, le=1)
    verdict: Literal["pass", "revise", "fail"] = "revise"

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"
