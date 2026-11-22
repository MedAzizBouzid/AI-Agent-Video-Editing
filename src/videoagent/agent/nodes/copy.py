"""Title, description and hashtags for publishing. A failure here never fails the run."""

from __future__ import annotations

import json

from ...llm.errors import LLMError
from ...llm.prompts import load_prompt
from ..state import AgentConfig, AgentState, Deps, PublishCopy


def make_copy(deps: Deps, cfg: AgentConfig):
    def write_copy(state: AgentState) -> dict:
        edl, brief = state["edl"], state["brief"]
        fallback = PublishCopy(title=edl.title, description=edl.description)
        if not cfg.write_copy:
            return {"copy": fallback, "events": ["copy: skipped (using the planner's title)"]}
        spoken = " ".join(c.text for c in edl.captions)[:3000]
        message = (
            "BRIEF:\n" + json.dumps(brief.model_dump(exclude={"inputs"}, exclude_none=True),
                                    ensure_ascii=False, indent=1)
            + f"\n\nPLATFORM: {state['platform']}\n\nVIDEO TEXT:\n{spoken or '(no captions)'}"
            + f"\n\nCURRENT TITLE: {edl.title}"
        )
        try:
            result = deps.client.complete_json(message, PublishCopy, system=load_prompt("copywriter"),
                                               tag="copywriter")
        except LLMError as exc:
            return {"copy": fallback, "events": [f"copy: failed ({type(exc).__name__}), using the planner's title"]}
        return {"copy": result, "events": ["copy: ok"]}

    return write_copy
