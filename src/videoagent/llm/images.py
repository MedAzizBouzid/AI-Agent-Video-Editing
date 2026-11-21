"""Image input for vision-capable models, in the OpenAI Chat Completions format: a user message
whose `content` is a list of parts, some text and some `image_url` (here a base64 data URL)."""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Rough token cost of one small (~512 px) image, used only when a provider reports no usage.
IMAGE_TOKEN_ESTIMATE = 700


def image_part(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def user_message_with_images(
    text: str, images: Sequence[str | Path], captions: Sequence[str] | None = None
) -> dict[str, Any]:
    """One user message: the text first, then each image (optionally preceded by its caption, so
    the model knows e.g. "Frame 2 at 4.5s")."""
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for i, img in enumerate(images):
        if captions and i < len(captions):
            parts.append({"type": "text", "text": captions[i]})
        parts.append(image_part(img))
    return {"role": "user", "content": parts}


def content_size(content: str | list[dict[str, Any]]) -> tuple[int, int]:
    """(characters of text, number of images) in a message's content."""
    if isinstance(content, str):
        return len(content), 0
    chars = sum(len(p.get("text", "")) for p in content if p.get("type") == "text")
    return chars, sum(1 for p in content if p.get("type") == "image_url")
