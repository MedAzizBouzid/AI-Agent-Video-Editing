"""Check the EDL with the rules in edl/validator.py. Errors go back to the planner verbatim."""

from __future__ import annotations

from ...edl.validator import validate_edl
from ..state import AgentConfig, AgentState, Deps


def make_validate(deps: Deps, cfg: AgentConfig):
    def validate(state: AgentState) -> dict:
        edl = state["edl"]
        durations = {name: doc.source.duration_s for name, doc in state["docs"].items()}
        result = validate_edl(
            edl, durations, deps.platforms[state["platform"]],
            voiceover_available=deps.voiceover_available,
        )
        if result.ok:
            return {"repair_rounds": 0, "feedback": "", "feedback_kind": "", "events": [
                "validate: ok" + (f" ({len(result.warnings)} warning(s))" if result.warnings else "")]}
        rounds = state.get("repair_rounds", 0) + 1
        feedback = result.format_for_llm()
        if state.get("review_feedback"):  # do not let the validator's errors erase the review
            feedback += ("\n\nAlso still to fix, from the review of the previous render "
                         "(times are OUTPUT time):\n" + state["review_feedback"])
        update = {"repair_rounds": rounds, "feedback": feedback,
                  "feedback_kind": "validation",
                  "events": [f"validate: {len(result.errors)} error(s), repair round {rounds}"]}
        if rounds > cfg.max_plan_repairs:
            update.update(status="failed", reason=(
                f"the planner could not produce a valid EDL in {cfg.max_plan_repairs} repairs; "
                f"last problems:\n{result.format_for_llm()}"))
        return update

    return validate
