"""House style: edit rules the pipeline applies itself instead of hoping the LLM follows them.

A planner prompt is a request; this is a guarantee. Each rule below fixes a mistake that a real
model made on real footage:

  * back-to-back clips of the same source are ONE clip. Two clips that meet at 30.2 s with a fade
    out and a fade in make the picture dip to black in the middle of a sentence.
  * fades only at the very start and end of the video. A dip at every jump cut looks amateur.
  * the call to action comes from the brief, never from the model's imagination, and sits in the
    last seconds of the OUTPUT (the model put one at 20-23 s of a 25.6 s video, and invented its text).
"""

from __future__ import annotations

from ..schemas import EDL, BriefSpec, Clip
from ..schemas.edl import Overlay

SEAM_GAP_S = 0.05  # clips closer than this in the source are treated as touching
CTA_SECONDS = 3.0


def merge_contiguous(clips: list[Clip]) -> list[Clip]:
    """Join consecutive clips that continue each other (same source, same speed, no gap)."""
    merged: list[Clip] = []
    for clip in clips:
        prev = merged[-1] if merged else None
        if (prev is not None and prev.source == clip.source and prev.speed == clip.speed
                and 0 <= clip.start - prev.end <= SEAM_GAP_S):
            merged[-1] = prev.model_copy(update={"end": clip.end, "fade_out_s": clip.fade_out_s})
        else:
            merged.append(clip)
    return merged


def only_edge_fades(clips: list[Clip]) -> list[Clip]:
    """Keep the first clip's fade-in and the last clip's fade-out; hard cuts everywhere else."""
    last = len(clips) - 1
    return [c.model_copy(update={"fade_in_s": c.fade_in_s if i == 0 else 0.0,
                                 "fade_out_s": c.fade_out_s if i == last else 0.0})
            for i, c in enumerate(clips)]


def place_cta(overlays: list[Overlay], cta: str | None, total: float) -> list[Overlay]:
    """Drop whatever CTA overlays the model wrote; add the brief's CTA over the last seconds."""
    kept = [o for o in overlays if o.kind != "cta"]
    if cta and cta.strip():
        kept.append(Overlay(kind="cta", text=cta.strip(), start=max(0.0, total - CTA_SECONDS), end=total))
    return kept


def apply_house_style(edl: EDL, brief: BriefSpec) -> tuple[EDL, list[str]]:
    """Returns the tidied EDL and a short note for each rule that changed something."""
    notes: list[str] = []
    clips = merge_contiguous(edl.clips)
    if len(clips) < len(edl.clips):
        notes.append(f"merged {len(edl.clips) - len(clips)} back-to-back clip(s)")
    faded = only_edge_fades(clips)
    if faded != clips:
        notes.append("removed fades between clips")
    edl = edl.model_copy(update={"clips": faded})
    overlays = place_cta(edl.overlays, brief.cta, edl.total_duration)
    if overlays != edl.overlays:
        notes.append("call to action " + ("placed from the brief" if brief.cta else "removed: the brief has none"))
    return edl.model_copy(update={"overlays": overlays}), notes
