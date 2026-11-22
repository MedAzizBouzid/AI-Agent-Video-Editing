"""Render the EDL with FFmpeg. The planner works with short names; the renderer needs real paths."""

from __future__ import annotations

from pathlib import Path

from ...schemas import EDL
from ..state import AgentConfig, AgentState, Deps


def with_real_paths(edl: EDL, inputs: dict[str, str]) -> EDL:
    return edl.model_copy(update={"clips": [
        c.model_copy(update={"source": str(Path(inputs[c.source]).resolve())}) for c in edl.clips]})


def make_render(deps: Deps, cfg: AgentConfig):
    def render(state: AgentState) -> dict:
        n = len(state.get("attempts", [])) + 1
        work = Path(state["out_dir"]) / ".work"
        work.mkdir(parents=True, exist_ok=True)
        dst = work / f"attempt{n}.{state['platform']}.mp4"
        try:
            deps.render(with_real_paths(state["edl"], state["inputs"]), dst, state["brief"])
        except Exception as exc:  # noqa: BLE001 - FFmpeg, validation and TTS errors all end the run
            return {"status": "failed", "reason": f"render failed: {exc}"[:1500],
                    "events": [f"render: failed ({type(exc).__name__})"]}
        return {"video": str(dst), "events": [f"render: attempt {n} -> {dst.name}"]}

    return render
