"""The provider fallback chain: try providers in order, and decide from the *kind* of failure
what to do next.

    429 rate limit  -> put that provider on cooldown (Retry-After, or exponential if it keeps
                       happening: that is what an exhausted daily quota looks like) and move on
    timeout / 5xx   -> retry the same provider with backoff, then move on
    401/402/403     -> the key or account is the problem: disable the provider for this session
    404             -> the model id no longer exists: disable it and say so
    400             -> if we asked for JSON mode, assume it is unsupported, turn it off for this
                       provider and retry; otherwise skip the provider for this call only

If every provider is cooling down for less than `max_wait_s` we sleep, otherwise we raise
AllProvidersFailed with the earliest retry time, so the caller can decide what to do.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .backends import Backend, ChatResult, Message
from .errors import (
    AllProvidersFailed, AuthError, BadRequestError, ModelNotFoundError, RateLimitError,
    TransientError,
)
from .providers import ProviderConfig, RetryConfig
from .usage import UsageLog, UsageRecord, cost_usd

log = logging.getLogger("videoagent.llm")
MAX_COOLDOWN_S = 3600.0
MAX_PASSES = 3


@dataclass
class Provider:
    cfg: ProviderConfig
    backend: Backend | None
    disabled: str | None = None  # reason, when permanently unusable this session
    cooldown_until: float = 0.0
    rate_limit_streak: int = 0
    json_mode: str = field(init=False)

    def __post_init__(self) -> None:
        self.json_mode = self.cfg.json_mode

    @property
    def name(self) -> str:
        return self.cfg.name


class _Skip(Exception):
    """Internal: give up on this provider for now; the message is the reason."""


class FallbackChain:
    def __init__(
        self,
        providers: list[Provider],
        retry: RetryConfig | None = None,
        usage: UsageLog | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.providers = providers
        self.retry = retry or RetryConfig()
        self.usage = usage if usage is not None else UsageLog()
        self._clock, self._sleep = clock, sleep

    # ---- public ---------------------------------------------------------------------------

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        want_json: bool = False,
        json_schema: dict[str, Any] | None = None,
        tag: str = "",
    ) -> ChatResult:
        failures = {p.name: p.disabled for p in self.providers if p.disabled}
        passes = 0
        while True:
            active = [p for p in self.providers if p.disabled is None]
            if not active:
                raise AllProvidersFailed(failures)
            now = self._clock()
            ready = [p for p in active if p.cooldown_until <= now]
            if not ready:
                wait = min(p.cooldown_until for p in active) - now
                if wait > self.retry.max_wait_s:
                    raise AllProvidersFailed(failures, retry_after=wait)
                log.info("all providers cooling down; waiting %.1fs", wait)
                self._sleep(wait)
                continue
            for provider in ready:
                try:
                    return self._try(provider, messages, temperature, max_tokens, want_json,
                                     json_schema, tag)
                except _Skip as skip:
                    failures[provider.name] = str(skip)
                    log.warning("provider %s skipped: %s", provider.name, skip)
            passes += 1
            now = self._clock()
            cooling = [p.cooldown_until - now for p in active if p.cooldown_until > now]
            if not cooling or passes >= MAX_PASSES:
                raise AllProvidersFailed(failures, retry_after=min(cooling) if cooling else None)

    def status(self) -> dict[str, str]:
        """provider name -> 'ready' | 'cooling down (Ns)' | disabled reason (for the CLI)."""
        now, out = self._clock(), {}
        for p in self.providers:
            if p.disabled:
                out[p.name] = p.disabled
            elif p.cooldown_until > now:
                out[p.name] = f"cooling down ({p.cooldown_until - now:.0f}s)"
            else:
                out[p.name] = "ready"
        return out

    # ---- internals ------------------------------------------------------------------------

    @staticmethod
    def _response_format(p: Provider, want_json: bool, schema: dict[str, Any] | None):
        if not want_json or p.json_mode == "none":
            return None
        if p.json_mode == "json_schema" and schema is not None:
            return {"type": "json_schema", "json_schema": {"name": "result", "schema": schema}}
        return {"type": "json_object"}

    def _record_failure(self, p: Provider, tag: str, error: str) -> None:
        self.usage.add(UsageRecord(provider=p.name, model=p.cfg.model, tag=tag, ok=False, error=error))

    def _try(self, p: Provider, messages, temperature, max_tokens, want_json, schema, tag) -> ChatResult:
        assert p.backend is not None
        attempt = 0
        while attempt < self.retry.max_attempts_per_provider:
            fmt = self._response_format(p, want_json, schema)
            try:
                result = p.backend.chat(
                    messages, temperature=temperature, max_tokens=max_tokens, response_format=fmt
                )
            except RateLimitError as exc:
                p.rate_limit_streak += 1
                cooldown = exc.retry_after
                if cooldown is None:  # repeated 429s usually mean a spent daily quota: back off harder
                    cooldown = self.retry.default_cooldown_s * 2 ** (p.rate_limit_streak - 1)
                cooldown = min(cooldown, MAX_COOLDOWN_S)
                p.cooldown_until = self._clock() + cooldown
                self._record_failure(p, tag, "rate limited")
                raise _Skip(f"rate limited (cooling down {cooldown:.0f}s)") from exc
            except AuthError as exc:
                p.disabled = f"auth/payment error, check the key or account: {exc}"
                self._record_failure(p, tag, "auth")
                raise _Skip(p.disabled) from exc
            except ModelNotFoundError as exc:
                p.disabled = (f"model '{p.cfg.model}' not found; pick a current id with "
                              f"`python -m apps.llm_check`: {exc}")
                self._record_failure(p, tag, "model not found")
                raise _Skip(p.disabled) from exc
            except BadRequestError as exc:
                if fmt is not None:  # most likely "response_format not supported"
                    log.warning("%s rejected JSON mode; continuing without it", p.name)
                    p.json_mode = "none"
                    continue  # not counted as an attempt
                self._record_failure(p, tag, "bad request")
                raise _Skip(f"request rejected: {exc}") from exc
            except TransientError as exc:
                attempt += 1
                self._record_failure(p, tag, str(exc)[:120])
                if attempt < self.retry.max_attempts_per_provider:
                    self._sleep(self.retry.backoff_seconds * 2 ** (attempt - 1))
                    continue
                p.cooldown_until = self._clock() + self.retry.backoff_seconds * 4
                raise _Skip(f"failed after {attempt} attempts: {exc}") from exc

            p.rate_limit_streak = 0
            self.usage.add(UsageRecord(
                provider=p.name, model=p.cfg.model, tag=tag, ok=True,
                prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                latency_s=result.latency_s, estimated=result.estimated,
                cost_usd=cost_usd(result.prompt_tokens, result.completion_tokens,
                                  p.cfg.price_in_per_mtok, p.cfg.price_out_per_mtok),
            ))
            return result
        raise _Skip("no attempts left")  # unreachable unless max_attempts_per_provider < 1
