from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .brief import Platform


class Clip(BaseModel):
    """A piece of a source file placed on the timeline, in order."""

    source: str = Field(description="Path matching a PerceptionDoc.source.path")
    start: float = Field(ge=0, description="In-point in the source, seconds")
    end: float = Field(gt=0, description="Out-point in the source, seconds")
    speed: float = Field(default=1.0, gt=0, le=4.0)
    fade_in_s: float = Field(default=0.0, ge=0)
    fade_out_s: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "Clip":
        if self.end <= self.start:
            raise ValueError("clip end must be > start")
        return self

    @property
    def duration(self) -> float:
        return (self.end - self.start) / self.speed


class Caption(BaseModel):
    """Timed on the OUTPUT timeline."""

    start: float = Field(ge=0)
    end: float = Field(gt=0)
    text: str


class Overlay(BaseModel):
    kind: Literal["logo", "title", "lower_third", "cta"]
    text: str | None = None
    asset: str | None = None
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    position: Literal["top", "center", "bottom", "top_left", "top_right"] = "bottom"


class VoiceoverLine(BaseModel):
    text: str
    start: float = Field(ge=0)
    voice: str | None = None


class AudioSpec(BaseModel):
    keep_original: bool = True
    music: str | None = None
    music_volume: float = Field(default=0.15, ge=0, le=1)
    normalize_lufs: float = -14.0


class EDL(BaseModel):
    """Edit Decision List: the only thing the LLM produces for rendering."""

    version: int = 1
    platform: Platform
    title: str
    description: str = ""
    clips: list[Clip] = Field(min_length=1)
    captions: list[Caption] = Field(default_factory=list)
    overlays: list[Overlay] = Field(default_factory=list)
    voiceover: list[VoiceoverLine] = Field(default_factory=list)
    audio: AudioSpec = Field(default_factory=AudioSpec)

    @property
    def total_duration(self) -> float:
        return sum(c.duration for c in self.clips)
