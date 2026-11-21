"""One backend = one way of talking to one provider. All the providers we use (Cerebras,
OpenRouter, Gemini's OpenAI-compatible endpoint, Groq, Cloudflare, ...) speak the OpenAI Chat
Completions format, so a single OpenAIBackend with a different `base_url` covers all of them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .errors import (
    AuthError, BadRequestError, LLMError, ModelNotFoundError, RateLimitError, TransientError,
)
from .providers import ProviderConfig
from .usage import estimate_tokens

Message = dict[str, str]


@dataclass(frozen=True)
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = False  # token counts were guessed, not reported by the provider
    finish_reason: str | None = None  # "stop", "length" (reply was cut off), ...
    provider: str = ""
    model: str = ""
    latency_s: float = 0.0


class Backend(Protocol):
    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, Any] | None,
    ) -> ChatResult: ...

    def list_models(self) -> list[str]: ...


def _retry_after(exc: Any) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    value = headers.get("retry-after") if headers else None
    try:
        return float(value) if value is not None else None
    except ValueError:  # an HTTP-date; we just fall back to the default cooldown
        return None


def translate_openai_error(exc: Exception) -> LLMError:
    """Map an openai-SDK exception to our error kinds (see errors.py)."""
    import openai

    detail = str(exc)[:300]
    if isinstance(exc, openai.RateLimitError):
        return RateLimitError(detail, retry_after=_retry_after(exc))
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return AuthError(detail)
    if isinstance(exc, openai.NotFoundError):
        return ModelNotFoundError(detail)
    if isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError)):
        return BadRequestError(detail)
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        if status == 402:  # OpenRouter and others: out of credits
            return AuthError(f"payment required: {detail}")
        if status >= 500 or status in (408, 409, 425):
            return TransientError(f"HTTP {status}: {detail}")
        return BadRequestError(f"HTTP {status}: {detail}")
    # timeouts, connection errors, malformed bodies from a flaky proxy...
    return TransientError(f"{type(exc).__name__}: {detail}")


class OpenAIBackend:
    def __init__(self, cfg: ProviderConfig, api_key: str, *, http_client: Any | None = None):
        import openai

        self.cfg = cfg
        # max_retries=0: retrying is the fallback chain's job, which can also switch provider.
        self._client = openai.OpenAI(
            base_url=cfg.base_url, api_key=api_key, timeout=cfg.timeout_s, max_retries=0,
            default_headers=cfg.headers or None, http_client=http_client,
        )

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        import time

        kwargs: dict[str, Any] = {
            "model": self.cfg.model, "messages": messages, "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if response_format is not None:
            kwargs["response_format"] = response_format
        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - everything is translated, nothing leaks out
            raise translate_openai_error(exc) from exc
        latency = time.monotonic() - started

        choice = response.choices[0] if response.choices else None
        text = (choice.message.content or "") if choice and choice.message else ""
        if not text.strip():
            raise TransientError("empty response")
        usage = getattr(response, "usage", None)
        if usage and usage.prompt_tokens is not None and usage.completion_tokens is not None:
            prompt, completion, estimated = usage.prompt_tokens, usage.completion_tokens, False
        else:
            prompt = estimate_tokens("".join(m["content"] for m in messages))
            completion, estimated = estimate_tokens(text), True
        return ChatResult(
            text=text, prompt_tokens=prompt, completion_tokens=completion, estimated=estimated,
            finish_reason=getattr(choice, "finish_reason", None), provider=self.cfg.name,
            model=self.cfg.model, latency_s=latency,
        )

    def list_models(self) -> list[str]:
        try:
            return sorted(m.id for m in self._client.models.list())
        except Exception as exc:  # noqa: BLE001
            raise translate_openai_error(exc) from exc
