from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Span(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "Span":
        if self.end < self.start:
            raise ValueError("end must be >= start")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class TranscriptSegment(Span):
    text: str
    confidence: float | None = None


class Shot(Span):
    index: int


class Silence(Span):
    pass


class SourceInfo(BaseModel):
    path: str
    sha256: str
    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool = True


class PerceptionDoc(BaseModel):
    """Compact, text-only description of a source file. This is what the LLM reads."""

    source: SourceInfo
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    shots: list[Shot] = Field(default_factory=list)
    silences: list[Silence] = Field(default_factory=list)
    keyframes: list[str] = Field(default_factory=list, description="Paths to sampled frames")
