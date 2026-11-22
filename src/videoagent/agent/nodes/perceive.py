"""Analyse the raw files: transcript, shots, silences. No LLM here, and results are cached."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ..state import AgentState, Deps


def short_names(paths: list[str]) -> dict[str, str]:
    """Stable short names the planner can copy exactly: the file name, made unique if two files
    share one ('a/clip.mp4' and 'b/clip.mp4' become 'clip.mp4' and 'clip_2.mp4')."""
    names: dict[str, str] = {}
    for p in paths:
        base = Path(p).name
        name, n = base, 1
        while name in names:
            n += 1
            name = f"{Path(base).stem}_{n}{Path(base).suffix}"
        names[name] = str(p)
    return names


def make_perceive(deps: Deps):
    def perceive(state: AgentState) -> dict:
        inputs: Mapping[str, str] = state["inputs"]
        docs = {}
        for name, path in inputs.items():
            if not Path(path).is_file():
                return {"status": "failed", "reason": f"input file not found: {path}",
                        "events": [f"perceive: missing file {path}"]}
            docs[name] = deps.perceive(Path(path), name)
        speech = sum(len(d.transcript) for d in docs.values())
        return {"docs": docs, "events": [
            f"perceive: {len(docs)} file(s), {speech} speech segment(s)"
            + ("" if speech else " (no speech detected: planning from shots only)")]}

    return perceive
