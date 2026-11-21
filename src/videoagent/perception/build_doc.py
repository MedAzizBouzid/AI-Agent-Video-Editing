"""Merge all perception results into one PerceptionDoc, cached on disk.

Perception is the slow, expensive part (Whisper on a 2 h recording takes minutes). The cache key
is the file's fingerprint plus every setting that changes the result, so re-running the agent on
the same footage, or iterating on the planner prompt, never repeats this work.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any
from pathlib import Path

from ..schemas.perception import PerceptionDoc, SourceInfo, TranscriptSegment
from ..tools._ffmpeg import PathLike, probe
from .frames import sample_keyframes
from .scenes import detect_shots
from .silence import detect_silences
from .transcribe import DEFAULT_MODEL, register_whisper, transcribe

TranscribeFn = Callable[[PathLike], tuple[list[TranscriptSegment], str | None]]
_CHUNK = 4 * 1024 * 1024


@dataclass(frozen=True)
class PerceptionConfig:
    """Every setting that changes the output. All of them go into the cache key."""

    whisper_model: str = DEFAULT_MODEL
    language: str | None = None
    silence_db: float = -35.0
    silence_min_s: float = 0.5
    scene_threshold: float = 0.3
    min_shot_s: float = 1.0
    max_keyframes: int = 12

    def key(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def file_fingerprint(path: PathLike) -> str:
    """Fast identity for a (possibly multi-GB) video: size + hash of the first, middle and last 4 MB.

    Hashing a whole 5 GB recording takes long; sampling three places detects any re-export, trim
    or replacement in practice. It is NOT a security hash: two files that differ only outside
    the sampled regions would collide.
    """
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with p.open("rb") as f:
        for offset in sorted({0, max(0, size // 2 - _CHUNK // 2), max(0, size - _CHUNK)}):
            f.seek(offset)
            h.update(f.read(_CHUNK))
    return h.hexdigest()[:32]


def _cache_file(cache_dir: Path, fingerprint: str, cfg: PerceptionConfig) -> Path:
    return cache_dir / f"{fingerprint}-{cfg.key()}.json"


def _load_cached(file: Path) -> PerceptionDoc | None:
    if not file.is_file():
        return None
    try:
        doc = PerceptionDoc.model_validate_json(file.read_text(encoding="utf-8"))
    except ValueError:  # corrupt or from an older schema: just rebuild
        return None
    return doc if all(Path(k).is_file() for k in doc.keyframes) else None


def build_perception_doc(
    src: PathLike,
    cfg: PerceptionConfig | None = None,
    cache_dir: PathLike = "data/cache/perception",
    *,
    transcribe_fn: TranscribeFn | None = None,
    use_cache: bool = True,
    source_name: str | None = None,
    model_manager: Any | None = None,
) -> PerceptionDoc:
    """Analyse `src` (or load the cached analysis).

    transcribe_fn: replaces Whisper (tests, a hosted STT API, or a model owned by model_manager).
        Default loads faster-whisper lazily on first use.
    model_manager: a runtime.ModelManager. When given (and no transcribe_fn), Whisper is loaded
        through it, so it shares the GPU safely with other models and stays loaded across files.
    source_name: what to store in `source.path`. The planner must use this exact string as
        `clip.source` in the EDL, so prefer a short stable name; defaults to `str(src)`.
    """
    cfg = cfg or PerceptionConfig()
    cache = Path(cache_dir)
    fingerprint = file_fingerprint(src)
    cache_file = _cache_file(cache, fingerprint, cfg)

    if use_cache and (doc := _load_cached(cache_file)):
        return doc.model_copy(update={"source": doc.source.model_copy(
            update={"path": source_name or str(src)})})

    info = probe(src)
    shots = detect_shots(src, cfg.scene_threshold, cfg.min_shot_s)
    silences = detect_silences(src, cfg.silence_db, cfg.silence_min_s) if info.has_audio else []
    transcript: list[TranscriptSegment] = []
    language: str | None = None
    if info.has_audio:
        if transcribe_fn is None and model_manager is not None:
            name = register_whisper(model_manager, cfg.whisper_model)
            with model_manager.use(name) as model:
                transcript, language = transcribe(src, model, cfg.language)
        elif transcribe_fn is None:
            from .transcribe import load_model

            model = load_model(cfg.whisper_model)
            transcript, language = transcribe(src, model, cfg.language)
        else:
            transcript, language = transcribe_fn(src)
    keyframes = sample_keyframes(
        src, shots, info.duration, cache / "frames" / fingerprint, cfg.max_keyframes
    )

    doc = PerceptionDoc(
        source=SourceInfo(
            path=source_name or str(src), sha256=fingerprint, duration_s=info.duration,
            width=info.width, height=info.height, fps=info.fps, has_audio=info.has_audio,
        ),
        transcript=transcript, shots=shots, silences=silences,
        keyframes=keyframes, language=language or cfg.language,
    )
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(".tmp")
    tmp.write_text(doc.model_dump_json(indent=1), encoding="utf-8")
    os.replace(tmp, cache_file)  # atomic: a crash never leaves a half-written cache entry
    return doc
