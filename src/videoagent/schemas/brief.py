from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Platform = Literal["youtube", "shorts", "tiktok"]


class BriefSpec(BaseModel):
    """What the client/user wants from the raw footage."""

    inputs: list[str] = Field(min_length=1, description="Paths to raw video files")
    goal: str = Field(description="One-sentence description of the desired video")
    platforms: list[Platform] = Field(default_factory=lambda: ["youtube"])
    language: str = "en"
    tone: str = "friendly, concise"
    target_duration_s: float | None = Field(default=None, gt=0)
    voiceover: bool = False
    captions: bool = True
    cta: str | None = None
    notes: str = ""
