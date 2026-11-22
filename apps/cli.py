"""The product entry point: raw footage + a short brief in, publish-ready video(s) out.

    python -m apps.cli run raw.mp4 --brief "30s teaser for our new sneakers, in Arabic" --platform shorts
    python -m apps.cli run a.mp4 b.mp4 --brief @brief.txt --platform shorts --platform youtube --out out/
    python -m apps.cli run raw.mp4 --brief "..." --plan-only          # just show the plan (no render)

For each platform it writes, in --out (default ./out):
    <name>.<platform>.mp4          the video            <name>.<platform>.edl.json     the edit plan
    <name>.<platform>.report.json  the critic's report  <name>.<platform>.copy.json    title/description/hashtags
    <name>.<platform>.run.json     what happened, how long, and what the LLM calls cost

Exit code: 0 = ready to publish, 1 = needs human review, 2 = failed.
A human always approves the result: "ready" means the checks and the reviewer found nothing wrong.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from videoagent.agent import AgentConfig, Deps, RunResult, run_pipeline
from videoagent.agent.deps import build_deps
from videoagent.console import use_utf8_console
from videoagent.llm import LLMError

EXIT = {"ready": 0, "planned": 0, "needs_review": 1, "failed": 2}
DepsFactory = Callable[[AgentConfig], Deps]


def read_brief(value: str) -> str:
    """`--brief "text"` or `--brief @file.txt` (a file is handy for long Arabic briefs)."""
    if value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="videoagent", description="AI video production agent")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="produce video(s) from raw footage and a brief")
    r.add_argument("inputs", nargs="+", type=Path, help="raw video file(s)")
    r.add_argument("--brief", required=True, help='what you want, in plain words, or @file.txt')
    r.add_argument("--platform", action="append", choices=["youtube", "shorts", "tiktok"],
                   help="repeat for several; default: what the brief says, else shorts")
    r.add_argument("--out", type=Path, default=Path("out"))
    r.add_argument("--language", default=None, help="output language: en, fr, ar, ar-TN")
    r.add_argument("--duration", type=float, default=None, help="target length in seconds")
    r.add_argument("--tone", default=None)
    r.add_argument("--cta", default=None, help="call-to-action text for the end of the video")
    r.add_argument("--voiceover", action="store_true", default=None, help="add a generated voice")
    r.add_argument("--no-captions", dest="captions", action="store_false", default=None)
    r.add_argument("--plan-only", action="store_true", help="stop after a valid plan; no render")
    r.add_argument("--no-review", action="store_true", help="skip the LLM review (checks still run)")
    r.add_argument("--no-copy", action="store_true", help="skip title/description generation")
    r.add_argument("--max-revisions", type=int, default=2)
    r.add_argument("--max-cost", type=float, default=0.50, help="LLM spend cap in USD")
    r.add_argument("--font", default="Arial", help="caption font (must contain Arabic glyphs for Arabic)")
    r.add_argument("--fonts-dir", type=Path, default=None)
    r.add_argument("--whisper-model", default="small")
    r.add_argument("--speech-language", default=None, help="language spoken in the footage (default: detect)")
    r.add_argument("--cache-dir", type=Path, default=Path("data/cache/perception"))
    r.add_argument("-v", "--verbose", action="store_true")
    return p


def print_result(res: RunResult) -> None:
    print(f"\n=== {res.platform}: {res.status.upper()} ===")
    if res.reason:
        print(res.reason)
    for event in res.events:
        print(f"  {event}")
    if res.copy:
        print(f"  title: {res.copy.title}")
        if res.copy.hashtags:
            print("  hashtags: " + " ".join("#" + h.lstrip("#") for h in res.copy.hashtags))
    if res.report:
        print(f"  review: {res.report.verdict}, score {res.report.score:.2f}")
    for label, path in res.outputs.items():
        print(f"  {label}: {path}")


def main(argv: list[str] | None = None, *, deps_factory: DepsFactory = build_deps) -> int:
    use_utf8_console()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    cfg = AgentConfig(
        max_revisions=args.max_revisions, max_cost_usd=args.max_cost, llm_review=not args.no_review,
        write_copy=not args.no_copy, plan_only=args.plan_only, font=args.font, fonts_dir=args.fonts_dir,
        whisper_model=args.whisper_model, speech_language=args.speech_language, cache_dir=args.cache_dir,
    )
    missing = [str(p) for p in args.inputs if not p.is_file()]
    if missing:
        print(f"input file not found: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        deps = deps_factory(cfg)
        results = run_pipeline(
            args.inputs, deps, cfg, out_dir=args.out, brief_text=read_brief(args.brief),
            platforms=args.platform,
            overrides={"language": args.language, "target_duration_s": args.duration, "tone": args.tone,
                       "cta": args.cta, "voiceover": args.voiceover, "captions": args.captions},
        )
    except LLMError as exc:
        print(f"could not reach any LLM provider: {exc}\nCheck: python -m apps.llm_check", file=sys.stderr)
        return 2
    for res in results:
        print_result(res)
    print("\n" + deps.client.usage.summary())
    return max(EXIT[r.status] for r in results)


def entrypoint() -> None:  # the `videoagent` console script
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
