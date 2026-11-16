from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class PlatformSpec(BaseModel):
    aspect: str
    width: int
    height: int
    fps: int
    video_bitrate: str
    audio_bitrate: str
    loudness_lufs: float
    max_duration_s: float
    caption_safe_zone_bottom_px: int = 0


def config_dir() -> Path:
    override = os.environ.get("VIDEOAGENT_CONFIG_DIR")
    return Path(override) if override else Path(__file__).resolve().parents[2] / "configs"


def load_platforms(path: str | os.PathLike[str] | None = None) -> dict[str, PlatformSpec]:
    file = Path(path) if path else config_dir() / "platforms.yaml"
    raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    return {name: PlatformSpec(**spec) for name, spec in raw.items()}
