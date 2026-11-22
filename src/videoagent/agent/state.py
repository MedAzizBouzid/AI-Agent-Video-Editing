"""What flows through the agent graph: configuration, injected dependencies, and the state.

The graph is a loop, not a line:

    perceive -> plan -> validate -> render -> critique -> copy -> finish
                  ^        |                    |
                  |        '-- errors ----------'   (the exact messages go back to the planner)
                  '-------- "revise" verdict -------'

State is a plain dict (LangGraph merges what each node returns into it). Every node reads what it
needs from the state and returns only the keys it changes.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

from ..config import PlatformSpec
from ..llm.client import LLMClient
from ..schemas import EDL, BriefSpec, CriticReport, PerceptionDoc
from ..schemas.brief import Platform

Status = Literal["running", "ready", "needs_review", "failed", "planned"]


class IntakeResult(BaseModel):
    """What the intake LLM extracts from a free-text request (the input files are not its business)."""

    goal: str
    platforms: list[Platform] = Field(default_factory=list)
    language: str | None = None
    tone: str = "friendly, concise"
    target_duration_s: float | None = Field(default=None, gt=0)
    voiceover: bool = False
    captions: bool = True
    cta: str | None = None
    notes: str = ""


class PublishCopy(BaseModel):
    title: str
    description: str = ""
    hashtags: list[str] = Field(default_factory=list)


@dataclass
class AgentConfig:
    """Limits and switches. The defaults keep a run cheap and bounded."""

    max_plan_repairs: int = 3  # validator failures the planner may fix before the run gives up
    max_revisions: int = 2  # re-plans after the critic says "revise"
    max_cost_usd: float = 0.50  # stop re-planning once the LLM spend passes this
    llm_review: bool = True  # let the critic's LLM look at the render (costs tokens)
    review_frames: int = 6
    write_copy: bool = True  # generate title / description / hashtags
    plan_only: bool = False  # stop after a valid EDL: no render (cheap way to inspect the plan)
    max_context_chars: int = 24000  # transcript text budget shared by all sources
    # rendering
    font: str = "Arial"
    fonts_dir: Path | None = None
    reframe_mode: str = "auto"
    # perception
    whisper_model: str = "small"
    speech_language: str | None = None  # None = let Whisper detect
    cache_dir: Path = Path("data/cache/perception")


# The injected functions. Production versions are built by `build_deps`; tests pass fakes.
PerceiveFn = Callable[[Path, str], PerceptionDoc]  # (real path, short name) -> doc
RenderFn = Callable[[EDL, Path, BriefSpec], Path]  # EDL with real source paths -> written file
CritiqueFn = Callable[..., CriticReport]  # (video, edl, brief, docs, platform) -> report


@dataclass
class Deps:
    client: LLMClient
    platforms: dict[str, PlatformSpec]
    perceive: PerceiveFn
    render: RenderFn
    critique: CritiqueFn
    voiceover_available: bool = False
    clock: Callable[[], float] = time.monotonic


class AgentState(TypedDict, total=False):
    # inputs
    brief: BriefSpec
    platform: str
    inputs: dict[str, str]  # short source name -> real file path
    out_dir: str
    # working data
    docs: dict[str, PerceptionDoc]  # short source name -> analysis
    edl: EDL | None
    feedback: str  # what to tell the planner next time ("" on the first pass)
    feedback_kind: Literal["", "validation", "review"]
    review_feedback: str  # the critic's last review; kept while the planner also fixes validator errors
    repair_rounds: int
    revisions: int
    video: str | None
    report: CriticReport | None
    copy: PublishCopy | None
    # bookkeeping
    status: Status
    reason: str
    attempts: Annotated[list[dict[str, Any]], operator.add]  # every render + its verdict
    events: Annotated[list[str], operator.add]  # human-readable trace of the run
    outputs: dict[str, str]  # label -> path of what finish wrote


@dataclass
class RunResult:
    platform: str
    status: Status
    reason: str
    outputs: dict[str, str]
    events: list[str]
    edl: EDL | None
    report: CriticReport | None
    copy: PublishCopy | None
