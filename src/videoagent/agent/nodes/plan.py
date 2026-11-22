"""The planner: an LLM chooses the clips; the pipeline builds the captions from the transcript."""

from __future__ import annotations

import json

from ...config import PlatformSpec
from ...edl.auto_captions import captions_from_transcript
from ...llm.errors import LLMError
from ...llm.prompts import load_prompt
from ...perception.compact import compact_for_llm
from ...schemas import EDL, BriefSpec, PerceptionDoc
from ..state import AgentConfig, AgentState, Deps


def build_plan_message(
    brief: BriefSpec,
    platform: str,
    spec: PlatformSpec,
    docs: dict[str, PerceptionDoc],
    *,
    max_chars: int,
    previous: EDL | None = None,
    feedback: str = "",
    feedback_kind: str = "",
) -> str:
    per_doc = max(4000, max_chars // max(1, len(docs)))
    sources = "\n\n".join(compact_for_llm(d, max_chars=per_doc) for d in docs.values())
    brief_json = brief.model_dump(exclude={"inputs"}, exclude_none=True)
    parts = [
        "BRIEF:\n" + json.dumps(brief_json, ensure_ascii=False, indent=1),
        f"PLATFORM: {platform} ({spec.width}x{spec.height}, {spec.fps}fps, "
        f"maximum {spec.max_duration_s:.0f} seconds). Set \"platform\" to \"{platform}\".",
        "SOURCES:\n" + sources,
    ]
    if previous is not None and feedback:
        label = ("PROBLEMS found by the validator in your previous EDL"
                 if feedback_kind == "validation"
                 else "REVIEW of the video rendered from your previous EDL")
        parts += [
            "YOUR PREVIOUS EDL:\n" + previous.model_dump_json(indent=1, exclude_defaults=True),
            f"{label}:\n{feedback}\n\nReturn the complete corrected EDL.",
        ]
    return "\n\n".join(parts)


def make_plan(deps: Deps, cfg: AgentConfig):
    def plan(state: AgentState) -> dict:
        brief, platform = state["brief"], state["platform"]
        feedback = state.get("feedback", "")
        message = build_plan_message(
            brief, platform, deps.platforms[platform], state["docs"],
            max_chars=cfg.max_context_chars, previous=state.get("edl"),
            feedback=feedback, feedback_kind=state.get("feedback_kind", ""),
        )
        tag = "planner" if not feedback else f"planner:{state.get('feedback_kind') or 'fix'}"
        try:
            edl = deps.client.complete_json(message, EDL, system=load_prompt("planner"), tag=tag)
        except LLMError as exc:
            return {"status": "failed", "reason": f"planner failed: {exc}",
                    "events": [f"plan: failed ({type(exc).__name__})"]}

        edl = edl.model_copy(update={"platform": platform})  # the graph, not the LLM, owns this
        notes = []
        if not brief.captions:
            edl = edl.model_copy(update={"captions": []})
        elif not edl.captions:
            captions = captions_from_transcript(edl, state["docs"])
            edl = edl.model_copy(update={"captions": captions})
            notes.append(f"{len(captions)} caption(s) built from the transcript")
        if not brief.voiceover and edl.voiceover:
            edl = edl.model_copy(update={"voiceover": []})
            notes.append("dropped voiceover the brief did not ask for")
        return {"edl": edl, "events": [
            f"plan: {len(edl.clips)} clip(s), {edl.total_duration:.1f}s"
            + (f" [{tag}]" if feedback else "") + (f"; {'; '.join(notes)}" if notes else "")]}

    return plan
