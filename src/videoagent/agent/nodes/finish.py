"""Write the deliverables for the human reviewer, whatever the outcome."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..state import AgentState, Deps


def _write_json(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def pick_best(attempts: list[dict]) -> dict | None:
    """The best render that is usable (not 'fail'); ties go to the later attempt."""
    usable = [a for a in attempts if a["verdict"] != "fail"]
    return max(usable, key=lambda a: (a["score"], attempts.index(a)), default=None)


def make_finish(deps: Deps):
    def finish(state: AgentState) -> dict:
        out = Path(state["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        stem = f"{Path(next(iter(state['inputs'].values()))).stem}.{state['platform']}"
        status, reason = state.get("status", "running"), state.get("reason", "")
        outputs: dict[str, str] = {}
        edl, report = state.get("edl"), state.get("report")

        best = None
        if status == "running":  # reached the end through a verdict: decide now
            best = pick_best(state.get("attempts", []))
            if report is not None and report.verdict == "pass":
                status, best = "ready", state["attempts"][-1]
            else:
                status = "needs_review" if best else "failed"
                reason = reason or "the critic still found problems after the allowed revisions"
        elif status == "failed" and (best := pick_best(state.get("attempts", []))):
            # a later step broke, but an earlier render was usable: hand that to the reviewer
            status, reason = "needs_review", f"stopped early ({reason.splitlines()[0][:200]}); best earlier render attached"
        if best:
            edl, report = best["edl"], best["report"]
            final = out / f"{stem}.mp4"
            shutil.copy2(best["video"], final)
            outputs["video"] = str(final)
        if status == "planned":
            reason = reason or "plan-only run: nothing was rendered"

        if edl is not None:
            _write_json(out / f"{stem}.edl.json", edl.model_dump_json(indent=1))
            outputs["edl"] = str(out / f"{stem}.edl.json")
        if report is not None:
            _write_json(out / f"{stem}.report.json", report.model_dump_json(indent=1))
            outputs["report"] = str(out / f"{stem}.report.json")
        if state.get("copy") is not None:
            _write_json(out / f"{stem}.copy.json", state["copy"].model_dump_json(indent=1))
            outputs["copy"] = str(out / f"{stem}.copy.json")
        usage = deps.client.usage
        run = {"platform": state["platform"], "status": status, "reason": reason,
               "events": state.get("events", []),
               "llm": {"calls": usage.calls, "failures": usage.failures,
                       "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                       "cost_usd": usage.cost_usd}}
        _write_json(out / f"{stem}.run.json", json.dumps(run, ensure_ascii=False, indent=1))
        outputs["run"] = str(out / f"{stem}.run.json")
        return {"status": status, "reason": reason, "outputs": outputs, "edl": edl, "report": report,
                "events": [f"finish: {status}"]}

    return finish
