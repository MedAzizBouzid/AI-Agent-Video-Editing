"""Check your LLM providers: which keys are set, which model IDs exist, and whether they answer.

    python -m apps.llm_check                 # status of each provider
    python -m apps.llm_check --models        # list the model IDs each provider offers now
    python -m apps.llm_check --models --grep free
    python -m apps.llm_check --ping          # send a tiny test prompt through the fallback chain

Free-tier model IDs change often. Use --models to find a current one, then put it in
configs/llm_providers.yaml.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from videoagent.console import use_utf8_console
from videoagent.llm import LLMError, build_client
from videoagent.llm.backends import OpenAIBackend
from videoagent.llm.client import find_env_file


_AUTO = object()  # "look for the .env yourself"


def describe_env(environ=None, env_file=_AUTO) -> list[str]:
    """Say which .env was read and which keys are set (never the values)."""
    environ = os.environ if environ is None else environ
    path = find_env_file() if env_file is _AUTO else env_file
    lines = []
    if path is None:
        lines.append(".env: NOT FOUND. Create a file named exactly .env (not .env.txt, not .env.example) "
                     "next to pyproject.toml, or run from that folder.")
    else:
        lines.append(f".env: {path}")
        head = Path(path).read_bytes()[:2]
        if head in (b"\xff\xfe", b"\xfe\xff"):
            lines.append("  WARNING: this file is saved as UTF-16 (PowerShell `>` does that), so it cannot be "
                         "read. Re-save it as UTF-8 (in Notepad: Save As > Encoding: UTF-8).")
    for name in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        lines.append(f"  {name}: {'set' if environ.get(name) else 'EMPTY or missing'}")
    print("\n".join(lines))
    return lines


def listing_backend(prov, environ=None):
    """A backend just for listing model IDs, or None if the provider's key is not set.

    Listing must work BEFORE a model ID is chosen (that is what it is for), so a placeholder model
    does not matter here: only the API key does. PRIVACY_STRICT does not matter either, since
    listing sends no prompt.
    """
    environ = os.environ if environ is None else environ
    key = environ.get(prov.cfg.api_key_env)
    return OpenAIBackend(prov.cfg, key) if key else None


def main(argv: list[str] | None = None) -> int:
    use_utf8_console()
    p = argparse.ArgumentParser(description="Check configured LLM providers")
    p.add_argument("--config", default=None, help="path to llm_providers.yaml")
    p.add_argument("--models", action="store_true", help="list model IDs each provider offers")
    p.add_argument("--grep", default=None, help="only show model IDs containing this text")
    p.add_argument("--ping", action="store_true", help="send a tiny prompt through the chain")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    client = build_client(args.config)  # loads .env into the environment
    describe_env()  # so this runs after it: it reports what the program actually sees
    for name, state in client.chain.status().items():
        print(f"{name:12} {state}")

    if args.models:
        for prov in client.chain.providers:
            backend = listing_backend(prov)
            if backend is None:
                print(f"\n{prov.name}: no API key. Set {prov.cfg.api_key_env} in .env to list its models.")
                continue
            print(f"\n{prov.name}: configured model = {prov.cfg.model}")
            try:
                ids = backend.list_models()
            except LLMError as exc:
                print(f"  could not list models: {exc}")
                continue
            shown = [m for m in ids if not args.grep or args.grep.lower() in m.lower()]
            for model_id in shown:
                print(f"  {model_id}")
            if not shown:
                print(f"  (no model matches '{args.grep}'; {len(ids)} models in total)")

    if args.ping:
        try:
            res = client.complete("Reply with the single word: OK", max_tokens=16, tag="ping")
        except LLMError as exc:
            print(f"\nping failed: {exc}")
            return 1
        print(f"\nping ok via {res.provider} ({res.model}) in {res.latency_s:.2f}s: {res.text.strip()!r}")
        print(client.usage.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
