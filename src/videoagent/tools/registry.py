"""Tool list exposed to the agent (and to MCP later). Add new tools here."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .audio import measure_loudness, mix_music, normalize_loudness
from .concat import concat
from .export import export
from .reframe import reframe
from .trim import trim


@dataclass(frozen=True)
class Tool:
    name: str
    fn: Callable[..., Any]
    description: str


def _tool(fn: Callable[..., Any]) -> Tool:
    doc = (fn.__doc__ or "").strip().splitlines()
    return Tool(name=fn.__name__, fn=fn, description=doc[0] if doc else "")


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in map(_tool, [trim, concat, reframe, normalize_loudness, measure_loudness, mix_music, export])
}


def get_tool(name: str) -> Tool:
    try:
        return TOOLS[name]
    except KeyError:
        raise KeyError(f"unknown tool '{name}', available: {sorted(TOOLS)}") from None
