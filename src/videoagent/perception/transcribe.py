"""Speech-to-text with faster-whisper (CTranslate2). Runs locally; fits a 6 GB GPU.

The model is passed in (or loaded lazily) so that `runtime/model_manager.py` can later own the
GPU lifecycle: load Whisper, transcribe, unload, then load TTS. Importing this module does not
import faster-whisper, so everything else works without it installed.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless noise on Windows
log = logging.getLogger("videoagent.perception")

from ..schemas.perception import TranscriptSegment
from ..tools._ffmpeg import PathLike

DEFAULT_MODEL = os.environ.get("VIDEOAGENT_WHISPER_MODEL", "small")


def load_model(size: str = DEFAULT_MODEL, device: str = "auto", compute_type: str = "auto") -> Any:
    """Load a faster-whisper model. First use downloads the weights from Hugging Face.

    On an RTX 3050 (6 GB): "small" in float16 uses ~2 GB; "medium" ~5 GB (tight, but possible if
    nothing else is on the GPU). compute_type="int8_float16" roughly halves memory.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - exercised only without the package
        raise RuntimeError("faster-whisper is not installed: pip install faster-whisper") from exc
    model = WhisperModel(size, device=device, compute_type=compute_type)
    if device != "auto":
        return model  # the caller asked for a specific device: let any failure be loud
    try:
        _probe(model)
    except RuntimeError as exc:
        if not gpu_libraries_missing(exc):
            raise
        log.warning(
            "Whisper found a GPU but its CUDA libraries are missing (%s); using the CPU instead "
            "(slower; medium and large models can take many minutes). For GPU speed install cuBLAS (CUDA 12) and cuDNN 9: on Windows use the libs release at github.com/Purfview/whisper-standalone-win and put the DLLs in a folder on PATH.", exc)
        model = WhisperModel(size, device="cpu", compute_type="int8")
    return model


def gpu_libraries_missing(exc: BaseException) -> bool:
    """True for the errors CTranslate2 raises when a CUDA library cannot be loaded."""
    text = str(exc).lower()
    return any(w in text for w in ("cublas", "cudnn", "cudart", "cuda", "dll")) and any(
        w in text for w in ("not found", "cannot be loaded", "cannot load", "could not locate", "failed to load"))


def _probe(model: Any) -> None:
    """Run one second of silence through the model. CTranslate2 loads the CUDA libraries lazily, on
    the first real computation, so a broken GPU setup would otherwise only show up mid-job."""
    import numpy as np

    segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language="en", vad_filter=False)
    for _ in segments:
        pass


# Approximate VRAM in MB for faster-whisper in float16 (decoder state and activations included).
# These are estimates used for budgeting, not measurements: verify on your card with nvidia-smi.
WHISPER_VRAM_MB = {"tiny": 500, "base": 700, "small": 1500, "medium": 3500,
                   "large-v2": 5500, "large-v3": 5500}


# The same models quantised to int8 weights (compute_type "int8_float16"): roughly 60% of the memory,
# usually with no audible loss of accuracy. Also estimates.
WHISPER_VRAM_MB_INT8 = {"tiny": 300, "base": 450, "small": 900, "medium": 2000,
                        "large-v2": 3300, "large-v3": 3300}


def choose_compute_type(size: str, budget_mb: int, compute_type: str = "auto") -> tuple[str, int]:
    """(compute_type, declared VRAM in MB) for loading `size` on a card with `budget_mb` to spend.

    "auto" means: full precision if it fits the budget, otherwise int8_float16 so that a large
    model still runs on a small card. An explicit choice is respected, never overridden.
    """
    full = WHISPER_VRAM_MB.get(size, 3000)
    quantised = WHISPER_VRAM_MB_INT8.get(size, 1800)
    if compute_type == "auto":
        return ("auto", full) if full <= budget_mb else ("int8_float16", quantised)
    return compute_type, quantised if compute_type.startswith("int8") else full


def register_whisper(
    manager: Any, size: str = DEFAULT_MODEL, device: str = "auto", compute_type: str = "auto"
) -> str:
    """Register Whisper with a ModelManager and return the name to pass to `manager.use()`."""
    name = f"whisper:{size}"
    compute_type, vram_mb = choose_compute_type(size, manager.budget_mb, compute_type)
    if compute_type != "auto":
        log.info("Whisper %s: using compute_type=%s (~%d MB)", size, compute_type, vram_mb)
    manager.register(name, lambda: load_model(size, device, compute_type), vram_mb=vram_mb)
    return name


def confidence_from_logprob(avg_logprob: float | None) -> float | None:
    """avg_logprob is the mean token log-probability; exp() turns it into a 0..1 probability."""
    if avg_logprob is None:
        return None
    return max(0.0, min(1.0, math.exp(avg_logprob)))


def transcribe(
    src: PathLike,
    model: Any | None = None,
    language: str | None = None,
    *,
    vad_filter: bool = True,
    drop_no_speech_above: float = 0.9,
) -> tuple[list[TranscriptSegment], str | None]:
    """Transcribe `src`. Returns (segments, detected_language).

    language: ISO code like "ar", "fr", "en"; None = auto-detect (less reliable on short or
        mixed-language clips: set it when you know it).
    vad_filter: skip non-speech with a voice-activity detector. This is the main defence against
        Whisper inventing text ("Thanks for watching") over silence and music.
    drop_no_speech_above: drop segments Whisper itself judges to be non-speech with at least this
        probability.
    """
    whisper = model if model is not None else load_model()
    segments, info = whisper.transcribe(
        str(src),
        language=language,
        vad_filter=vad_filter,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        # Feeding earlier text back in is what makes Whisper loop and repeat itself.
        condition_on_previous_text=False,
    )
    out: list[TranscriptSegment] = []
    for seg in segments:  # a generator: decoding happens while we iterate
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        if (getattr(seg, "no_speech_prob", 0.0) or 0.0) > drop_no_speech_above:
            continue
        start, end = float(seg.start), float(seg.end)
        if end <= start:
            continue
        out.append(TranscriptSegment(
            start=start, end=end, text=text,
            confidence=confidence_from_logprob(getattr(seg, "avg_logprob", None)),
        ))
    return out, getattr(info, "language", None)
