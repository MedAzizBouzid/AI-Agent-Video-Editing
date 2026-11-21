"""Prompt files live next to this module in prompts/*.md so they can be edited and diffed as text."""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt named '{name}' in {PROMPT_DIR}")
    return path.read_text(encoding="utf-8").strip()
