"""Build captions from the transcript instead of asking the LLM to compute timings.

A planner that cuts a 10-minute recording into 40 seconds has to map "the word said at source
time 312.4 s" to "output time 17.9 s" for every caption, through cuts, speed changes and
gaps. That is arithmetic, and LLMs get it wrong in ways nobody notices until the captions drift
out of sync. So the planner only chooses WHICH parts of the footage to keep (clips); this module
derives the captions from the transcript for exactly those parts.

The planner can still write its own captions when it must change the words (translation, a
shorter hook). If it supplies any, they are kept and nothing here runs.

Limits: Whisper gives segment-level timings, not word-level, so a long segment is split into
readable chunks with its time shared out in proportion to their length. That is accurate to a
fraction of a second for normal speech, which is enough for captions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from ..schemas.edl import EDL, Caption
from ..schemas.perception import PerceptionDoc

MAX_CHARS = 42  # per caption; two short lines at shorts font sizes
MIN_SHOW_S = 0.6  # a caption shown for less than this cannot be read
MIN_OVERLAP_S = 0.3  # ignore a transcript segment that only grazes the clip edge
GAP_S = 0.05  # captions never touch, so the validator's overlap rule cannot trip on rounding


def split_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Break `text` into chunks of at most `max_chars`, at spaces (never inside a word).

    Prefers to break after punctuation, so a chunk reads as a phrase. A single word longer than
    `max_chars` is kept whole rather than cut (cutting Arabic mid-word would break its shaping).
    """
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and length + extra > max_chars:
            chunks.append(" ".join(current))
            current, length = [], 0
            extra = len(word)
        current.append(word)
        length += extra
        if re.search(r"[.!?؟،,;:…]$", word) and length >= max_chars * 0.5:
            chunks.append(" ".join(current))
            current, length = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks


def captions_from_transcript(
    edl: EDL, docs: Mapping[str, PerceptionDoc], *, max_chars: int = MAX_CHARS
) -> list[Caption]:
    """Captions on the OUTPUT timeline for the speech inside `edl.clips`.

    Handles cuts, speed changes and the gap between clips. Sources with no PerceptionDoc, and
    clips with no speech, simply produce no captions.
    """
    captions: list[Caption] = []
    offset = 0.0  # where the current clip starts on the output timeline
    for clip in edl.clips:
        doc = docs.get(clip.source)
        if doc is not None:
            for seg in doc.transcript:
                lo, hi = max(seg.start, clip.start), min(seg.end, clip.end)
                if hi - lo < MIN_OVERLAP_S or not seg.text.strip():
                    continue
                # Keep the part of the text that falls inside the clip when a cut slices a segment
                # (proportional to time; exact words are unknowable without word timestamps).
                seg_len = max(seg.end - seg.start, 1e-6)
                words = seg.text.split()
                a = round((lo - seg.start) / seg_len * len(words))
                b = round((hi - seg.start) / seg_len * len(words))
                kept = " ".join(words[a:b] if (lo > seg.start or hi < seg.end) else words)
                if not kept:
                    continue
                out_lo = offset + (lo - clip.start) / clip.speed
                out_hi = offset + (hi - clip.start) / clip.speed
                captions.extend(_spread(kept, out_lo, out_hi, max_chars))
        offset += clip.duration
    return _tidy(captions, edl.total_duration)


def _spread(text: str, start: float, end: float, max_chars: int) -> list[Caption]:
    """Share [start, end] between the chunks of `text` in proportion to their length."""
    chunks = split_text(text, max_chars)
    total_chars = sum(len(c) for c in chunks) or 1
    out, t = [], start
    for chunk in chunks:
        share = (end - start) * len(chunk) / total_chars
        out.append(Caption(start=t, end=t + share, text=chunk))
        t += share
    return out


def _tidy(captions: list[Caption], total: float) -> list[Caption]:
    """Sort, clamp to the video, drop what cannot be read, and remove any overlap."""
    result: list[Caption] = []
    previous_end = 0.0
    for cap in sorted(captions, key=lambda c: c.start):
        start = max(cap.start, previous_end + (GAP_S if result else 0.0))
        end = min(cap.end, total)
        if end - start < MIN_SHOW_S:
            # too short to read: stretch it if there is room before the video ends, else drop it
            end = min(start + MIN_SHOW_S, total)
            if end - start < MIN_SHOW_S - 1e-6:
                continue
        result.append(Caption(start=round(start, 3), end=round(end, 3), text=cap.text))
        previous_end = end
    return result
