"""Voiceover: text -> speech audio, placed on the timeline.

Three backends behind one interface, chosen per language by `RoutedTTS` (see configs/tts.yaml):

  piper  offline CLI, CPU, many languages. Arabic: only ar_JO (Jordanian), no Tunisian voice.
  kokoro higher quality, small GPU/CPU model. en, fr, es, hi, it, pt, ja, zh. NO Arabic.
  edge   free online Microsoft voices incl. Tunisian Arabic. Unofficial endpoint with unclear
         commercial terms: check them (or use Azure AI Speech) before using it for paying clients.

Synthesised audio is cached by (backend, voice, language, text), so re-rendering a video never
re-synthesises lines that did not change.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..schemas.edl import EDL, VoiceoverLine
from ._ffmpeg import PathLike, probe


class TTSError(RuntimeError):
    """A backend failed or is not set up (message says how to fix it)."""


class VoiceoverError(ValueError):
    """The voiceover cannot be placed on the video. The message is written for the planner."""


class VoiceoverUnavailable(VoiceoverError):
    """The EDL has voiceover lines but no TTS backend is configured."""


def base_language(code: str | None) -> str | None:
    """'ar-TN' / 'ar_TN' / 'AR' -> 'ar'."""
    return re.split(r"[-_]", code.strip().lower())[0] if code and code.strip() else None


def resolve_voice(
    voices: Mapping[str, str], voice: str | None, language: str | None, default_language: str | None
) -> str:
    """Pick a voice: an explicit `voice` (an alias from `voices`, or used literally), else the one
    configured for the language ('ar-TN' falls back to 'ar'). The default language is used only
    when no language is given: reading German text with an English voice would be gibberish."""
    if voice:
        return voices.get(voice, voice)
    keys = (language, base_language(language)) if language else (default_language,)
    for key in keys:
        if key and key in voices:
            return voices[key]
    raise TTSError(f"no voice configured for language {language!r}; configured: {sorted(voices)}")


class TTSBackend(Protocol):
    name: str

    def resolve(self, voice: str | None, language: str | None) -> str:
        """The concrete voice that would be used (also part of the cache key)."""

    def synthesize(
        self, text: str, dst_stem: Path, *, voice: str | None = None, language: str | None = None
    ) -> Path:
        """Write speech for `text` to `dst_stem` plus the backend's own suffix; return that path."""


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---- Piper -----------------------------------------------------------------------------------

class PiperBackend:
    """Runs the `piper` CLI (pip install piper-tts). `voices` maps language -> path of a .onnx voice
    model (download them from https://huggingface.co/rhasspy/piper-voices)."""

    name = "piper"

    def __init__(
        self,
        voices: Mapping[str, str],
        *,
        binary: str = "piper",
        default_language: str | None = None,
        args: Sequence[str] = ("-m", "{model}", "-f", "{out}"),
        timeout_s: float = 120.0,
    ):
        self.voices, self.binary, self.default_language = dict(voices), binary, default_language
        self.args, self.timeout_s = list(args), timeout_s

    def resolve(self, voice: str | None, language: str | None) -> str:
        return resolve_voice(self.voices, voice, language, self.default_language)

    def synthesize(self, text, dst_stem, *, voice=None, language=None) -> Path:
        model = self.resolve(voice, language)
        if not Path(model).is_file():
            raise TTSError(f"piper voice model not found: {model} "
                           "(download the .onnx and .onnx.json from huggingface.co/rhasspy/piper-voices)")
        out = dst_stem.with_suffix(".wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [self.binary] + [a.format(model=model, out=out) for a in self.args]
        try:
            proc = subprocess.run(cmd, input=_squash(text), capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=self.timeout_s)
        except FileNotFoundError as exc:
            raise TTSError(f"'{self.binary}' not found: pip install piper-tts") from exc
        except subprocess.TimeoutExpired as exc:
            raise TTSError(f"piper timed out after {self.timeout_s:.0f}s") from exc
        if proc.returncode != 0:
            raise TTSError(f"piper failed (exit {proc.returncode}): {proc.stderr[-500:]}")
        if not out.is_file() or out.stat().st_size <= 44:  # 44 bytes = an empty WAV header
            raise TTSError("piper produced no audio")
        return out


# ---- Kokoro ----------------------------------------------------------------------------------

KOKORO_LANG_CODES = {"en": "a", "fr": "f", "es": "e", "hi": "h", "it": "i", "pt": "p",
                     "ja": "j", "zh": "z"}
KOKORO_SAMPLE_RATE = 24000
KOKORO_VRAM_MB = 600  # estimate for the 82M-parameter model


def write_wav(path: Path, samples: Any, sample_rate: int) -> None:
    """Mono float samples in [-1, 1] -> 16-bit PCM WAV (stdlib only, so no soundfile needed)."""
    import numpy as np

    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


class KokoroBackend:
    """Kokoro via `pip install kokoro soundfile` (needs espeak-ng for non-English).

    `voices` maps language -> Kokoro voice id (e.g. fr -> ff_siwis). With a ModelManager the
    pipeline is loaded through it, so it shares the GPU with Whisper."""

    name = "kokoro"

    def __init__(
        self,
        voices: Mapping[str, str],
        *,
        default_language: str | None = None,
        manager: Any | None = None,
        pipeline_factory: Callable[[str], Any] | None = None,
        speed: float = 1.0,
    ):
        self.voices, self.default_language = dict(voices), default_language
        self.manager, self.speed = manager, speed
        self._factory = pipeline_factory or self._default_factory
        self._pipelines: dict[str, Any] = {}

    @staticmethod
    def _default_factory(lang_code: str) -> Any:
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise TTSError("kokoro is not installed: pip install kokoro soundfile") from exc
        return KPipeline(lang_code=lang_code)

    def resolve(self, voice: str | None, language: str | None) -> str:
        return resolve_voice(self.voices, voice, language, self.default_language)

    def _lang_code(self, language: str | None) -> str:
        lang = base_language(language) or self.default_language
        code = KOKORO_LANG_CODES.get(lang or "")
        if code is None:
            raise TTSError(f"Kokoro does not support language {language!r} "
                           f"(supported: {sorted(KOKORO_LANG_CODES)}); route it to piper or edge")
        return code

    def synthesize(self, text, dst_stem, *, voice=None, language=None) -> Path:
        import numpy as np

        code = self._lang_code(language)
        voice_id = self.resolve(voice, language)
        if self.manager is not None:
            name = f"kokoro:{code}"
            self.manager.register(name, lambda: self._factory(code), vram_mb=KOKORO_VRAM_MB)
            with self.manager.use(name) as pipeline:
                chunks = self._run(pipeline, text, voice_id)
        else:
            pipeline = self._pipelines.setdefault(code, self._factory(code))
            chunks = self._run(pipeline, text, voice_id)
        if not chunks:
            raise TTSError("kokoro produced no audio")
        out = dst_stem.with_suffix(".wav")
        out.parent.mkdir(parents=True, exist_ok=True)
        write_wav(out, np.concatenate(chunks), KOKORO_SAMPLE_RATE)
        return out

    def _run(self, pipeline: Any, text: str, voice_id: str) -> list[Any]:
        import numpy as np

        chunks = []
        for result in pipeline(_squash(text), voice=voice_id, speed=self.speed):
            audio = getattr(result, "audio", None)
            if audio is None:  # older kokoro versions yield (graphemes, phonemes, audio) tuples
                audio = result[2]
            if audio is None:
                continue
            if hasattr(audio, "detach"):  # a torch tensor
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        return chunks


# ---- Edge (online) ---------------------------------------------------------------------------

class EdgeBackend:
    """Microsoft Edge online voices via `pip install edge-tts`. List voices with `edge-tts --list-voices`.

    Unofficial service: confirm its terms permit your use before selling videos made with it."""

    name = "edge"

    def __init__(
        self,
        voices: Mapping[str, str],
        *,
        default_language: str | None = None,
        save_fn: Callable[[str, str, Path], None] | None = None,
    ):
        self.voices, self.default_language = dict(voices), default_language
        self._save = save_fn or self._default_save

    @staticmethod
    def _default_save(text: str, voice: str, out: Path) -> None:
        import asyncio

        try:
            import edge_tts
        except ImportError as exc:
            raise TTSError("edge-tts is not installed: pip install edge-tts") from exc
        asyncio.run(edge_tts.Communicate(text, voice).save(str(out)))

    def resolve(self, voice: str | None, language: str | None) -> str:
        return resolve_voice(self.voices, voice, language, self.default_language)

    def synthesize(self, text, dst_stem, *, voice=None, language=None) -> Path:
        voice_id = self.resolve(voice, language)
        out = dst_stem.with_suffix(".mp3")
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._save(_squash(text), voice_id, out)
        except TTSError:
            raise
        except Exception as exc:  # noqa: BLE001 - network and service errors vary widely
            raise TTSError(f"edge-tts failed: {type(exc).__name__}: {str(exc)[:200]}") from exc
        if not out.is_file() or out.stat().st_size == 0:
            raise TTSError("edge-tts produced no audio")
        return out


# ---- routing by language ----------------------------------------------------------------------

@dataclass(frozen=True)
class Route:
    backend: str
    voice: str | None = None


class RoutedTTS:
    """Picks the backend (and its voice) from the language, e.g. fr -> kokoro, ar -> piper."""

    def __init__(
        self,
        backends: Mapping[str, TTSBackend],
        routes: Mapping[str, Route],
        default_language: str | None = None,
    ):
        for lang, route in routes.items():
            if route.backend not in backends:
                raise TTSError(f"language '{lang}' is routed to unknown backend '{route.backend}'")
        self.backends, self.routes, self.default_language = dict(backends), dict(routes), default_language
        self.name = "routed"

    def _route(self, language: str | None) -> Route:  # the backend resolves the voice by language
        keys = (language, base_language(language)) if language else (self.default_language,)
        for key in keys:
            if key and key in self.routes:
                return self.routes[key]
        raise TTSError(f"no TTS route for language {language!r}; configured: {sorted(self.routes)}")

    def resolve(self, voice: str | None, language: str | None) -> str:
        route = self._route(language)
        backend = self.backends[route.backend]
        return f"{route.backend}:{backend.resolve(voice, language)}"

    def synthesize(self, text, dst_stem, *, voice=None, language=None) -> Path:
        route = self._route(language)
        return self.backends[route.backend].synthesize(
            text, dst_stem, voice=voice, language=language
        )


# ---- cache -----------------------------------------------------------------------------------

def synthesize_cached(
    tts: TTSBackend, text: str, cache_dir: PathLike, *, voice: str | None = None,
    language: str | None = None,
) -> Path:
    """Synthesise once per distinct (backend, resolved voice, language, text)."""
    key = hashlib.sha256(json.dumps(
        [tts.name, tts.resolve(voice, language), base_language(language) or "", _squash(text)],
        ensure_ascii=False).encode()).hexdigest()[:24]
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for hit in cache.glob(f"{key}.*"):
        if hit.stat().st_size > 0:
            return hit
    tmp_stem = cache / f".tmp-{key}"
    produced = tts.synthesize(text, tmp_stem, voice=voice, language=language)
    final = cache / f"{key}{produced.suffix}"
    os.replace(produced, final)  # atomic: a crash never leaves a half-written cache entry
    return final


# ---- scheduling ------------------------------------------------------------------------------

@dataclass(frozen=True)
class ScheduledLine:
    index: int  # position in EDL.voiceover
    text: str
    path: Path
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


def schedule_lines(
    lines: Sequence[VoiceoverLine],
    paths: Sequence[Path],
    durations: Sequence[float],
    total_duration: float,
    *,
    gap_s: float = 0.25,
    tolerance_s: float = 0.3,
) -> tuple[list[ScheduledLine], list[str]]:
    """Place lines on the output timeline in start order. A line that would overlap the previous
    one is pushed back to follow it (with `gap_s`); a line that then runs past the end of the video
    is an error, because speech cut off mid-sentence is never acceptable. Returns (lines, notes)."""
    scheduled: list[ScheduledLine] = []
    notes: list[str] = []
    cursor = 0.0
    for i in sorted(range(len(lines)), key=lambda k: lines[k].start):
        want, dur = lines[i].start, durations[i]
        start = max(want, cursor)
        if start > want + 1e-6:
            notes.append(f"voiceover[{i}] moved from {want:.1f}s to {start:.1f}s to follow the previous line")
        end = start + dur
        if end > total_duration + tolerance_s:
            snippet = lines[i].text.strip()[:40]
            raise VoiceoverError(
                f"voiceover[{i}] (\"{snippet}\") would end at {end:.1f}s but the video is only "
                f"{total_duration:.1f}s long ({end - total_duration:.1f}s too long). Shorten the text, "
                "start it earlier, or make the video longer."
            )
        scheduled.append(ScheduledLine(i, lines[i].text, paths[i], start, dur))
        cursor = end + gap_s
    return scheduled, notes


def prepare_voiceover(
    edl: EDL,
    tts: TTSBackend,
    cache_dir: PathLike,
    *,
    language: str | None = None,
    duration_fn: Callable[[Path], float] | None = None,
) -> tuple[list[ScheduledLine], list[str]]:
    """Synthesise (cached) every line of `edl.voiceover` and place them on the timeline."""
    get_duration = duration_fn or (lambda p: probe(p).duration)
    paths = [synthesize_cached(tts, line.text, cache_dir, voice=line.voice, language=language)
             for line in edl.voiceover]
    durations = [get_duration(p) for p in paths]
    return schedule_lines(edl.voiceover, paths, durations, edl.total_duration)
