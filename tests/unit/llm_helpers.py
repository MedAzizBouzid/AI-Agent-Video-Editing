"""Shared fakes for the LLM tests."""

from __future__ import annotations

from videoagent.llm.backends import ChatResult
from videoagent.llm.fallback import FallbackChain, Provider
from videoagent.llm.providers import ProviderConfig, RetryConfig


class Clock:
    """Fake time: sleep() just advances now, so tests of waiting are instant."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def ok(text: str = "ok", *, prompt: int = 10, completion: int = 5, finish: str | None = "stop") -> ChatResult:
    return ChatResult(text=text, prompt_tokens=prompt, completion_tokens=completion, finish_reason=finish)


class FakeBackend:
    """Plays back a script: each call consumes one item; an Exception item is raised."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, messages, *, temperature, max_tokens, response_format):
        self.calls.append({"messages": messages, "response_format": response_format,
                           "temperature": temperature, "max_tokens": max_tokens})
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item

    def list_models(self):
        return ["fake-model"]


def cfg(name: str, **kw) -> ProviderConfig:
    base = dict(name=name, base_url="http://x", api_key_env="K", model=f"{name}-model")
    base.update(kw)
    return ProviderConfig(**base)


def make_chain(*specs, retry: RetryConfig | None = None, clock: Clock | None = None):
    """specs: (name, backend) or (name, backend, config_kwargs). Returns (chain, clock)."""
    clock = clock or Clock()
    providers = []
    for spec in specs:
        name, backend, *rest = spec
        providers.append(Provider(cfg(name, **(rest[0] if rest else {})), backend))
    chain = FallbackChain(providers, retry or RetryConfig(), clock=clock, sleep=clock.sleep)
    return chain, clock

MSG = [{"role": "user", "content": "hi"}]
