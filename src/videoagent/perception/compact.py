"""The text the planner LLM actually reads.

A full PerceptionDoc as JSON repeats keys on every segment and wastes tokens. This renders the
same facts as terse timestamped lines. Roughly 1 token per 3-4 characters, so the default
`max_chars` of 24000 is about 6-8k tokens: small enough for free-tier context and rate limits.
"""

from __future__ import annotations

from ..schemas.perception import PerceptionDoc


def _t(seconds: float) -> str:
    return f"{seconds:.1f}"


def compact_for_llm(
    doc: PerceptionDoc,
    *,
    max_chars: int = 24000,
    min_silence_s: float = 1.0,
    start: float = 0.0,
    end: float | None = None,
) -> str:
    """Render `doc` as text. `start`/`end` select a time window (for planning long footage in
    chunks). If the transcript does not fit in `max_chars`, it is cut and the last line says how
    many segments were left out, so the model knows it is not seeing everything."""
    src = doc.source
    stop = end if end is not None else src.duration_s
    head = [
        f"SOURCE {src.path} | {src.width}x{src.height} | {src.fps:.0f}fps | "
        f"duration {_t(src.duration_s)}s | language {doc.language or 'unknown'}",
        f"WINDOW {_t(start)}-{_t(stop)}s",
    ]
    shots = [s for s in doc.shots if s.end > start and s.start < stop]
    head.append("SHOTS: " + (" ".join(f"{_t(s.start)}-{_t(s.end)}" for s in shots) or "none"))
    silences = [
        s for s in doc.silences
        if s.duration >= min_silence_s and s.end > start and s.start < stop
    ]
    head.append(
        f"SILENCES >={min_silence_s:g}s: "
        + (" ".join(f"{_t(s.start)}-{_t(s.end)}" for s in silences) or "none")
    )
    head.append("TRANSCRIPT:")

    segments = [t for t in doc.transcript if t.end > start and t.start < stop]
    used = sum(len(line) + 1 for line in head)
    lines: list[str] = []
    for i, seg in enumerate(segments):
        line = f"[{_t(seg.start)}-{_t(seg.end)}] {seg.text}"
        if used + len(line) + 1 > max_chars:
            lines.append(f"... {len(segments) - i} more segments not shown")
            break
        lines.append(line)
        used += len(line) + 1
    if not segments:
        lines.append("(no speech detected)")
    return "\n".join(head + lines)
