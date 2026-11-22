"""Wire the nodes into a LangGraph state machine and expose `run_pipeline`.

Why a graph and not a script: the interesting behaviour is the LOOPS. A bad plan goes back to the
planner with the validator's exact messages; a render the critic dislikes goes back with the review.
Each loop is bounded (AgentConfig), so a run always ends: ready, needs_review or failed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from ..schemas import BriefSpec
from .nodes.copy import make_copy
from .nodes.finish import make_finish
from .nodes.intake import parse_brief
from .nodes.perceive import make_perceive, short_names
from .nodes.plan import make_plan
from .nodes.render import make_render
from .nodes.critique import make_critique
from .nodes.validate import make_validate
from .state import AgentConfig, AgentState, Deps, RunResult

log = logging.getLogger("videoagent.agent")
RECURSION_LIMIT = 80  # generous: the loops are bounded by AgentConfig, this only stops a wiring bug


def _failed(state: AgentState) -> bool:
    return state.get("status") == "failed"


def build_graph(deps: Deps, cfg: AgentConfig | None = None):
    cfg = cfg or AgentConfig()

    def timed(name: str, fn):
        """Add the node's wall time and LLM spend to the run's event trace."""
        def wrapper(state: AgentState) -> dict:
            t0, calls0 = deps.clock(), deps.client.usage.calls
            update = dict(fn(state))
            took, calls = deps.clock() - t0, deps.client.usage.calls - calls0
            extra = f" [{took:.1f}s" + (f", {calls} LLM call(s)" if calls else "") + "]"
            if update.get("events"):
                update["events"] = [*update["events"][:-1], update["events"][-1] + extra]
            return update
        return wrapper

    g = StateGraph(AgentState)
    g.add_node("perceive", timed("perceive", make_perceive(deps)))
    g.add_node("plan", timed("plan", make_plan(deps, cfg)))
    g.add_node("validate", timed("validate", make_validate(deps, cfg)))
    g.add_node("render", timed("render", make_render(deps, cfg)))
    g.add_node("critique", timed("critique", make_critique(deps, cfg)))
    g.add_node("copy", timed("copy", make_copy(deps, cfg)))
    g.add_node("finish", make_finish(deps))

    g.add_edge(START, "perceive")
    g.add_conditional_edges("perceive", lambda s: "finish" if _failed(s) else "plan",
                            {"plan": "plan", "finish": "finish"})
    g.add_conditional_edges("plan", lambda s: "finish" if _failed(s) else "validate",
                            {"validate": "validate", "finish": "finish"})

    def after_validate(s: AgentState) -> str:
        if _failed(s):
            return "finish"
        if s.get("feedback_kind") == "validation":  # errors: the planner gets another try
            return "plan"
        return "finish" if cfg.plan_only else "render"

    g.add_conditional_edges("validate", after_validate,
                            {"plan": "plan", "render": "render", "finish": "finish"})
    g.add_conditional_edges("render", lambda s: "finish" if _failed(s) else "critique",
                            {"critique": "critique", "finish": "finish"})

    def after_critique(s: AgentState) -> str:
        if _failed(s):
            return "finish"
        report = s["report"]
        if report.verdict == "pass":
            return "copy"
        within_budget = (s.get("revisions", 0) <= cfg.max_revisions
                         and deps.client.usage.cost_usd <= cfg.max_cost_usd)
        return "plan" if within_budget else "finish"

    g.add_conditional_edges("critique", after_critique,
                            {"copy": "copy", "plan": "plan", "finish": "finish"})
    g.add_edge("copy", "finish")
    g.add_edge("finish", END)
    return g.compile()




def run_pipeline(
    inputs: Sequence[str | Path],
    deps: Deps,
    cfg: AgentConfig | None = None,
    *,
    out_dir: str | Path = "out",
    brief_text: str | None = None,
    brief: BriefSpec | None = None,
    platforms: Sequence[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> list[RunResult]:
    """Produce one video per platform from `inputs`.

    Give either `brief` (already structured) or `brief_text` (free text for the intake LLM).
    `platforms` overrides the brief's platforms; `overrides` are brief fields that beat the LLM's reading.
    """
    cfg = cfg or AgentConfig()
    paths = [str(p) for p in inputs]
    names = short_names(paths)
    if brief is None:
        if not brief_text:
            raise ValueError("give a brief (BriefSpec) or brief_text")
        brief = parse_brief(deps.client, brief_text, paths, overrides=overrides,
                            default_language=None)
    elif overrides:
        brief = brief.model_copy(update={k: v for k, v in overrides.items() if v is not None})
    targets = list(platforms) if platforms else list(brief.platforms)
    unknown = [p for p in targets if p not in deps.platforms]
    if unknown:
        raise ValueError(f"unknown platform(s) {unknown}; known: {sorted(deps.platforms)}")

    graph = build_graph(deps, cfg)
    results = []
    for platform in targets:
        log.info("running pipeline for %s", platform)
        state = graph.invoke(
            {"brief": brief.model_copy(update={"platforms": [platform]}), "platform": platform,
             "inputs": names, "out_dir": str(out_dir), "attempts": [], "events": [],
             "repair_rounds": 0, "revisions": 0, "review_feedback": "", "status": "planned" if cfg.plan_only else "running"},
            {"recursion_limit": RECURSION_LIMIT},
        )
        results.append(RunResult(
            platform=platform, status=state["status"], reason=state.get("reason", ""),
            outputs=state.get("outputs", {}), events=state.get("events", []),
            edl=state.get("edl"), report=state.get("report"), copy=state.get("copy"),
        ))
    return results
