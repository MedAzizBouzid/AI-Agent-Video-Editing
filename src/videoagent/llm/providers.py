"""Provider configuration (configs/llm_providers.yaml)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..config import config_dir

PLACEHOLDER = "REPLACE_WITH"


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in the YAML should fail loudly

    name: str
    base_url: str
    api_key_env: str
    model: str
    trains_on_data: bool = False
    # How to ask for JSON: "json_object" (widely supported), "json_schema" (strict, fewer
    # providers), or "none" (rely on the prompt alone). Downgraded automatically if rejected.
    json_mode: Literal["json_object", "json_schema", "none"] = "json_object"
    timeout_s: float = 60.0
    price_in_per_mtok: float = 0.0  # USD per million input tokens (0 for free tiers)
    price_out_per_mtok: float = 0.0
    headers: dict[str, str] = Field(default_factory=dict)  # extra HTTP headers


class RetryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_attempts_per_provider: int = Field(default=2, ge=1)
    backoff_seconds: float = Field(default=2.0, ge=0)  # doubles on every retry
    max_wait_s: float = Field(default=30.0, ge=0)  # longest we will sleep waiting for a cooldown
    default_cooldown_s: float = Field(default=60.0, ge=0)  # 429 without a Retry-After header


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    providers: list[ProviderConfig]
    retry: RetryConfig = Field(default_factory=RetryConfig)


def load_llm_config(path: str | os.PathLike[str] | None = None) -> LLMConfig:
    file = Path(path) if path else config_dir() / "llm_providers.yaml"
    return LLMConfig(**yaml.safe_load(file.read_text(encoding="utf-8")))


def unavailable_reason(
    cfg: ProviderConfig, env: Mapping[str, str], *, strict_privacy: bool = False
) -> str | None:
    """Why this provider cannot be used right now, or None if it can. Checked once at startup so
    the user sees a clear message instead of a 401 deep inside a run."""
    if strict_privacy and cfg.trains_on_data:
        return "skipped by PRIVACY_STRICT (provider may train on prompts)"
    if PLACEHOLDER in cfg.model:
        return "model id is still a placeholder in configs/llm_providers.yaml"
    if not env.get(cfg.api_key_env):
        return f"environment variable {cfg.api_key_env} is not set"
    return None
