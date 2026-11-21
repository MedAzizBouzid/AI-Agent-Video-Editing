"""Token, latency and cost accounting. The success metric is "cents per video", so every call is
recorded with a `tag` (planner, critic, ...) that lets evals break the cost down per stage."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class UsageRecord:
    provider: str
    model: str
    tag: str
    ok: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    estimated: bool = False  # True when the provider reported no token counts
    error: str = ""


@dataclass
class UsageLog:
    records: list[UsageRecord] = field(default_factory=list)

    def add(self, record: UsageRecord) -> None:
        self.records.append(record)

    @property
    def calls(self) -> int:
        return sum(1 for r in self.records if r.ok)

    @property
    def failures(self) -> int:
        return sum(1 for r in self.records if not r.ok)

    @property
    def prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    @property
    def completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    @property
    def cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.records)

    def by(self, key: str) -> dict[str, dict[str, float]]:
        """Totals grouped by "provider", "tag" or "model"."""
        groups: dict[str, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "failures": 0, "tokens": 0, "cost_usd": 0.0}
        )
        for r in self.records:
            g = groups[getattr(r, key)]
            g["calls" if r.ok else "failures"] += 1
            g["tokens"] += r.prompt_tokens + r.completion_tokens
            g["cost_usd"] += r.cost_usd
        return dict(groups)

    def summary(self) -> str:
        return (
            f"{self.calls} calls, {self.failures} failed attempts, "
            f"{self.prompt_tokens} in / {self.completion_tokens} out tokens, "
            f"${self.cost_usd:.4f}"
        )


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token) for providers that return no usage."""
    return max(1, len(text) // 4)


def cost_usd(prompt_tokens: int, completion_tokens: int, price_in: float, price_out: float) -> float:
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000
