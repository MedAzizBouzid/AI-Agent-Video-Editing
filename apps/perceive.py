"""Analyse a video and print what the planner LLM will see.

    python -m apps.perceive raw.mp4 --model small --language ar

First run downloads the Whisper weights. Results are cached in data/cache/perception.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from videoagent.perception import PerceptionConfig, build_perception_doc, compact_for_llm


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a PerceptionDoc for a video")
    p.add_argument("video", type=Path)
    p.add_argument("--model", default=PerceptionConfig().whisper_model,
                   help="faster-whisper size: tiny, base, small, medium, large-v3")
    p.add_argument("--language", default=None, help="ISO code (ar, fr, en); default auto-detect")
    p.add_argument("--cache-dir", type=Path, default=Path("data/cache/perception"))
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--json", action="store_true", help="print the full PerceptionDoc as JSON")
    args = p.parse_args(argv)

    cfg = PerceptionConfig(whisper_model=args.model, language=args.language)
    doc = build_perception_doc(
        args.video, cfg, args.cache_dir, use_cache=not args.no_cache, source_name=args.video.name
    )
    print(doc.model_dump_json(indent=1) if args.json else compact_for_llm(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
