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

from videoagent.llm import LLMError, build_client


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Check configured LLM providers")
    p.add_argument("--config", default=None, help="path to llm_providers.yaml")
    p.add_argument("--models", action="store_true", help="list model IDs each provider offers")
    p.add_argument("--grep", default=None, help="only show model IDs containing this text")
    p.add_argument("--ping", action="store_true", help="send a tiny prompt through the chain")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    client = build_client(args.config)
    for name, state in client.chain.status().items():
        print(f"{name:12} {state}")

    if args.models:
        for prov in client.chain.providers:
            if prov.backend is None:
                continue
            print(f"\n{prov.name}: configured model = {prov.cfg.model}")
            try:
                ids = prov.backend.list_models()
            except LLMError as exc:
                print(f"  could not list models: {exc}")
                continue
            for model_id in ids:
                if not args.grep or args.grep.lower() in model_id.lower():
                    print(f"  {model_id}")

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
