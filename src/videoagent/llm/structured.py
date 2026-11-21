"""Getting reliable structured output from models that are only *mostly* reliable.

Free-tier and small models often wrap JSON in markdown fences, add a sentence before it, or emit a
`<think>` block first. `extract_json` tolerates all of that. Whatever is found is then checked by
Pydantic, and the exact validation errors are fed back to the model for a repair attempt
(see LLMClient.complete_json).
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ValidationError

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse the first JSON object or array in `text`. Raises ValueError if there is none."""
    cleaned = _THINK.sub("", text)
    candidates = [m.group(1) for m in _FENCE.finditer(cleaned)] + [cleaned]
    decoder = json.JSONDecoder()
    last_error: Exception | None = None
    for candidate in candidates:
        for i, ch in enumerate(candidate):
            if ch not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[i:])
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            return value
    raise ValueError(f"no valid JSON found in the reply ({last_error or 'no { or [ present'})")


def json_schema_of(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema()


def schema_instructions(model: type[BaseModel]) -> str:
    """The system-prompt addendum that tells the model what JSON to produce."""
    schema = json.dumps(json_schema_of(model), separators=(",", ":"), ensure_ascii=False)
    return (
        "Respond with a single JSON object and nothing else: no markdown fences, no commentary. "
        f"It must validate against this JSON Schema:\n{schema}"
    )


def format_validation_error(exc: ValidationError | ValueError, limit: int = 10) -> str:
    """A short, model-readable list of what was wrong."""
    if isinstance(exc, ValidationError):
        errors = exc.errors()
        lines = [
            f"- {'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in errors[:limit]
        ]
        if len(errors) > limit:
            lines.append(f"- ... and {len(errors) - limit} more")
        return "\n".join(lines)
    return f"- {exc}"
