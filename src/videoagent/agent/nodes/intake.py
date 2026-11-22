"""Turn the client's free-text request into a BriefSpec (runs once per job, before the graph)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...llm.client import LLMClient
from ...llm.prompts import load_prompt
from ...schemas import BriefSpec
from ..state import IntakeResult

DEFAULT_PLATFORM = "shorts"


def parse_brief(
    client: LLMClient,
    text: str,
    inputs: Sequence[str],
    *,
    overrides: Mapping[str, Any] | None = None,
    default_language: str | None = None,
    footage_language: str | None = None,
) -> BriefSpec:
    """Ask the LLM to extract the structured brief from `text`.

    `overrides` are explicit choices (CLI flags) that always win over what the LLM inferred: a
    flag is the user's exact word, the LLM only a reading of free text. Values of None are ignored.
    """
    hints = [f"Default platform if none is named: {DEFAULT_PLATFORM}."]
    if footage_language:
        hints.append(f"The footage's spoken language is: {footage_language}.")
    if default_language:
        hints.append(f"Default output language if the request does not say: {default_language}.")
    result = client.complete_json(
        f"CLIENT REQUEST:\n{text.strip()}\n\n" + "\n".join(hints),
        IntakeResult, system=load_prompt("intake"), tag="intake",
    )
    data = result.model_dump()
    data["platforms"] = data["platforms"] or [DEFAULT_PLATFORM]
    data["language"] = data["language"] or default_language or footage_language or "en"
    data.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return BriefSpec(inputs=list(inputs), **data)
