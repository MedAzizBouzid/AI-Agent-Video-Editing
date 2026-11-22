"""Judge a rendered video.

    python -m apps.critique out.mp4 --platform shorts                 # measurements only (free)
    python -m apps.critique out.mp4 --platform shorts --llm --goal "30s product teaser in Arabic"
    python -m apps.critique out.mp4 --platform shorts --edl edl.json  # also checks captions and cuts

Exit code: 0 = pass, 1 = revise (fixable by re-planning), 2 = fail (broken render).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from videoagent.console import use_utf8_console
from videoagent.config import load_platforms
from videoagent.critic import critique
from videoagent.llm import build_client
from videoagent.schemas import EDL, BriefSpec

EXIT = {"pass": 0, "revise": 1, "fail": 2}


def main(argv: list[str] | None = None) -> int:
    use_utf8_console()
    p = argparse.ArgumentParser(description="Critique a rendered video")
    p.add_argument("video", type=Path)
    p.add_argument("--platform", required=True, choices=sorted(load_platforms()))
    p.add_argument("--edl", type=Path, default=None, help="EDL JSON used to render the video")
    p.add_argument("--goal", default=None, help="the brief's goal (enables brief-aware review)")
    p.add_argument("--language", default="en")
    p.add_argument("--target-duration", type=float, default=None)
    p.add_argument("--llm", action="store_true", help="also run the LLM review (uses API tokens)")
    p.add_argument("--frames", type=int, default=6, help="frames shown to the LLM (0 = text only)")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    args = p.parse_args(argv)

    spec = load_platforms()[args.platform]
    edl = EDL(**json.loads(args.edl.read_text(encoding="utf-8"))) if args.edl else None
    brief = BriefSpec(
        inputs=[str(args.video)], goal=args.goal, platforms=[args.platform], language=args.language,
        target_duration_s=args.target_duration, captions=bool(edl and edl.captions) or False,
    ) if args.goal else None
    client = build_client() if args.llm else None

    report = critique(args.video, spec, platform=args.platform, brief=brief, edl=edl,
                      client=client, review_frames=args.frames)
    if args.json:
        print(report.model_dump_json(indent=1))
    else:
        for c in report.checks:
            print(f"{'PASS' if c.passed else 'FAIL'}  {c.name:18} {c.detail}")
        print()
        print(report.format_for_planner())
        for note in report.notes:
            print(f"note: {note}")
        if client:
            print(client.usage.summary())
    return EXIT[report.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
