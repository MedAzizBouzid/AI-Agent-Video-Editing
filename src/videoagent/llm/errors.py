"""Error types. Backends translate every SDK/HTTP failure into one of these, so the fallback
chain decides what to do from the *kind* of failure and never has to know about HTTP details."""

from __future__ import annotations


class LLMError(Exception):
    """Base class for everything this package raises."""


class RateLimitError(LLMError):
    """HTTP 429. `retry_after` (seconds) is set when the provider told us how long to wait."""

    def __init__(self, message: str = "rate limited", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class TransientError(LLMError):
    """Worth retrying: timeout, connection reset, HTTP 5xx, empty response."""


class AuthError(LLMError):
    """HTTP 401/403: the key is wrong or lacks access. Retrying cannot help."""


class ModelNotFoundError(LLMError):
    """HTTP 404: the model ID does not exist (free-tier IDs expire). Fix configs/llm_providers.yaml."""


class BadRequestError(LLMError):
    """HTTP 400/422: the request was rejected (unsupported option, prompt too long, ...)."""


class AllProvidersFailed(LLMError):
    """Every provider failed or is unavailable. `failures` maps provider name -> reason."""

    def __init__(self, failures: dict[str, str], retry_after: float | None = None):
        self.failures = failures
        self.retry_after = retry_after
        detail = "; ".join(f"{k}: {v}" for k, v in failures.items()) or "no provider configured"
        hint = f" (earliest retry in {retry_after:.0f}s)" if retry_after else ""
        super().__init__(f"all LLM providers failed{hint}: {detail}")


class StructuredOutputError(LLMError):
    """The model never produced JSON that matches the requested schema, even after repair tries."""

    def __init__(self, message: str, last_text: str = ""):
        super().__init__(message)
        self.last_text = last_text
