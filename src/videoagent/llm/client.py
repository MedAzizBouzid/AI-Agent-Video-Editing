"""LLMClient: the one object the rest of the agent talks to.

    client = build_client()
    edl = client.complete_json(messages, EDL, system=planner_prompt, tag="planner")

Nothing outside this package knows which provider answered.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from pydantic import BaseModel

from .backends import Backend, ChatResult, Message, OpenAIBackend
from .errors import StructuredOutputError
from .fallback import FallbackChain, Provider
from .providers import ProviderConfig, load_llm_config, unavailable_reason
from .structured import extract_json, format_validation_error, json_schema_of, schema_instructions
from .usage import UsageLog

log = logging.getLogger("videoagent.llm")
T = TypeVar("T", bound=BaseModel)
MAX_ECHO_CHARS = 6000  # how much of a bad reply we send back to the model when asking it to repair


def _as_messages(messages: list[Message] | str) -> list[Message]:
    return [{"role": "user", "content": messages}] if isinstance(messages, str) else list(messages)


def _with_system(messages: list[Message], system: str | None) -> list[Message]:  # noqa: D401
    """Merge `system` into the conversation: appended to an existing system message, else first."""
    if not system:
        return messages
    if messages and messages[0]["role"] == "system":
        return [{"role": "system", "content": messages[0]["content"] + "\n\n" + system}, *messages[1:]]
    return [{"role": "system", "content": system}, *messages]


class LLMClient:
    def __init__(self, chain: FallbackChain, *, max_repairs: int = 2):
        self.chain = chain
        self.max_repairs = max_repairs

    @property
    def usage(self) -> UsageLog:
        return self.chain.usage

    @property
    def supports_vision(self) -> bool:
        return self.chain.supports_vision()

    def complete(
        self,
        messages: list[Message] | str,
        *,
        system: str | None = None,
        temperature: float = 0.4,
        max_tokens: int | None = None,
        tag: str = "",
    ) -> ChatResult:
        """Free-text completion (titles, descriptions, ...)."""
        return self.chain.chat(
            _with_system(_as_messages(messages), system),
            temperature=temperature, max_tokens=max_tokens, tag=tag,
        )

    def complete_json(
        self,
        messages: list[Message] | str,
        schema: type[T],
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        tag: str = "",
        max_repairs: int | None = None,
        needs_vision: bool = False,
    ) -> T:
        """Ask for JSON matching `schema` (a Pydantic model) and return the validated object.

        If the reply is not valid JSON or fails validation, the exact problems are sent back and
        the model gets up to `max_repairs` chances to fix them. Raises StructuredOutputError if
        it never does. Repair calls are tagged "<tag>:repair" in the usage log.
        """
        repairs = self.max_repairs if max_repairs is None else max_repairs
        instructions = schema_instructions(schema)
        convo = _with_system(_as_messages(messages), (system + "\n\n" if system else "") + instructions)
        schema_dict = json_schema_of(schema)

        last = ""
        for attempt in range(repairs + 1):
            result = self.chain.chat(
                convo, temperature=temperature, max_tokens=max_tokens, want_json=True,
                json_schema=schema_dict, tag=tag if attempt == 0 else f"{tag}:repair",
                needs_vision=needs_vision,
            )
            last = result.text
            try:
                return schema.model_validate(extract_json(result.text))
            except ValueError as exc:  # JSON errors and pydantic.ValidationError both land here
                problems = format_validation_error(exc)
                log.info("structured output invalid (attempt %d/%d):\n%s", attempt + 1, repairs + 1, problems)
                if attempt == repairs:
                    break
                note = "Your reply was cut off because it was too long. Make it shorter.\n" \
                    if result.finish_reason == "length" else ""
                convo = convo + [
                    {"role": "assistant", "content": result.text[:MAX_ECHO_CHARS]},
                    {"role": "user", "content": (
                        f"{note}That reply is not valid. Problems:\n{problems}\n"
                        "Return the corrected JSON object only."
                    )},
                ]
        raise StructuredOutputError(
            f"no valid {schema.__name__} after {repairs + 1} attempts; last problems:\n"
            f"{format_validation_error(_last_error(last, schema))}",
            last_text=last,
        )


def _last_error(text: str, schema: type[BaseModel]) -> ValueError:
    try:
        schema.model_validate(extract_json(text))
    except ValueError as exc:
        return exc
    return ValueError("unknown validation failure")  # pragma: no cover


BackendFactory = Callable[[ProviderConfig, str], Backend]


def build_client(
    config_path: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    strict_privacy: bool | None = None,
    backend_factory: BackendFactory | None = None,
    usage: UsageLog | None = None,
    load_dotenv_file: bool = True,
    max_repairs: int = 2,
    **chain_kwargs: Any,
) -> LLMClient:
    """Build a client from configs/llm_providers.yaml and the environment (.env is loaded too).

    Providers that cannot be used (no key, placeholder model id, PRIVACY_STRICT) are kept in the
    chain as disabled with the reason, so errors and `status()` can say exactly what is wrong.
    """
    if load_dotenv_file:
        try:
            from dotenv import load_dotenv

            load_dotenv()  # never overrides variables that are already set
        except ImportError:  # pragma: no cover
            pass
    environ: Mapping[str, str] = os.environ if env is None else env
    if strict_privacy is None:
        strict_privacy = environ.get("PRIVACY_STRICT", "0").lower() in ("1", "true", "yes")
    factory = backend_factory or (lambda cfg, key: OpenAIBackend(cfg, key))

    config = load_llm_config(config_path)
    providers: list[Provider] = []
    for cfg in config.providers:
        reason = unavailable_reason(cfg, environ, strict_privacy=strict_privacy)
        if reason:
            log.info("provider %s unavailable: %s", cfg.name, reason)
            providers.append(Provider(cfg, backend=None, disabled=reason))
        else:
            providers.append(Provider(cfg, backend=factory(cfg, environ[cfg.api_key_env])))
    chain = FallbackChain(providers, config.retry, usage, **chain_kwargs)
    return LLMClient(chain, max_repairs=max_repairs)
