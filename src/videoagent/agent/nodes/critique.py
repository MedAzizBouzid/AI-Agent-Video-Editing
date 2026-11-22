"""Judge the render and decide what happens next. Routing lives in graph.py; this only measures."""

from __future__ import annotations

from ..state import AgentConfig, AgentState, Deps


def make_critique(deps: Deps, cfg: AgentConfig):
    def critique(state: AgentState) -> dict:
        report = deps.critique(
            state["video"], state["edl"], state["brief"], state["docs"], state["platform"],
        )
        attempt = {"video": state["video"], "edl": state["edl"], "report": report,
                   "score": report.score, "verdict": report.verdict}
        update: dict = {"report": report, "attempts": [attempt], "events": [
            f"critique: {report.verdict} (score {report.score:.2f}, "
            f"{sum(1 for i in report.issues if i.severity == 'major')} major issue(s))"]}
        if report.verdict == "pass":
            update.update(feedback="", feedback_kind="", review_feedback="")
        elif report.verdict == "revise":
            update.update(feedback=report.format_for_planner(), feedback_kind="review",
                          review_feedback=report.format_for_planner(),
                          revisions=state.get("revisions", 0) + 1)
            spent = deps.client.usage.cost_usd
            if spent > cfg.max_cost_usd:
                update["reason"] = f"LLM spend cap reached (${spent:.2f} > ${cfg.max_cost_usd:.2f})"
        else:
            update.update(status="failed", reason="the render is broken:\n" + report.format_for_planner())
        return update

    return critique
