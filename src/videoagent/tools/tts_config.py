"""Build a RoutedTTS from configs/tts.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..config import config_dir
from .voiceover import EdgeBackend, KokoroBackend, PiperBackend, Route, RoutedTTS, TTSBackend, TTSError


class LanguageRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str
    voice: str | None = None


class PiperSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    binary: str = "piper"
    args: list[str] = Field(default_factory=lambda: ["-m", "{model}", "-f", "{out}"])
    timeout_s: float = 120.0


class TTSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    default_language: str = "en"
    languages: dict[str, LanguageRoute]
    piper: PiperSettings = Field(default_factory=PiperSettings)


def load_tts_config(path: str | os.PathLike[str] | None = None) -> TTSConfig:
    file = Path(path) if path else config_dir() / "tts.yaml"
    return TTSConfig(**yaml.safe_load(file.read_text(encoding="utf-8")))


def build_tts(
    config: TTSConfig | str | os.PathLike[str] | None = None, *, manager: Any | None = None,
    base_dir: str | os.PathLike[str] | None = None,
) -> TTSBackend:
    """Create only the backends the config actually routes to. Relative piper model paths are
    resolved against `base_dir` (default: the current directory)."""
    cfg = config if isinstance(config, TTSConfig) else load_tts_config(config)
    base = Path(base_dir) if base_dir else Path.cwd()
    used = {r.backend for r in cfg.languages.values()}
    unknown = used - {"piper", "kokoro", "edge"}
    if unknown:
        raise TTSError(f"unknown TTS backend(s) in tts.yaml: {sorted(unknown)}")

    def voices(backend: str) -> dict[str, str]:
        out = {}
        for lang, r in cfg.languages.items():
            if r.backend == backend and r.voice:
                out[lang] = str(base / r.voice) if backend == "piper" and not Path(r.voice).is_absolute() else r.voice
        return out

    backends: dict[str, TTSBackend] = {}
    if "piper" in used:
        backends["piper"] = PiperBackend(voices("piper"), binary=cfg.piper.binary, args=cfg.piper.args,
                                         timeout_s=cfg.piper.timeout_s, default_language=cfg.default_language)
    if "kokoro" in used:
        backends["kokoro"] = KokoroBackend(voices("kokoro"), manager=manager,
                                           default_language=cfg.default_language)
    if "edge" in used:
        backends["edge"] = EdgeBackend(voices("edge"), default_language=cfg.default_language)
    routes = {lang: Route(r.backend, r.voice) for lang, r in cfg.languages.items()}
    return RoutedTTS(backends, routes, cfg.default_language)
