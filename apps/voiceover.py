"""Speak one line of text with the configured voice (a quick way to test TTS setup).

    python -m apps.voiceover "Bonjour tout le monde" hello --language fr
    python -m apps.voiceover "مرحبا بكم" hello --language ar

Writes hello.wav (or .mp3 for the edge backend) next to where you run it, then prints its length.
Voices come from configs/tts.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from videoagent.console import use_utf8_console
from videoagent.runtime import ModelManager
from videoagent.tools._ffmpeg import probe
from videoagent.tools.tts_config import build_tts
from videoagent.tools.voiceover import TTSError


def main(argv: list[str] | None = None) -> int:
    use_utf8_console()
    p = argparse.ArgumentParser(description="Synthesise a line of speech")
    p.add_argument("text")
    p.add_argument("output", type=Path, help="output path without extension")
    p.add_argument("--language", default=None, help="e.g. en, fr, ar, ar-TN (default: config default)")
    p.add_argument("--voice", default=None, help="override the configured voice")
    p.add_argument("--config", default=None, help="path to tts.yaml")
    args = p.parse_args(argv)

    tts = build_tts(args.config, manager=ModelManager())
    try:
        out = tts.synthesize(args.text, args.output.with_suffix(""), voice=args.voice,
                             language=args.language)
    except TTSError as exc:
        print(f"TTS failed: {exc}")
        return 1
    print(f"wrote {out} ({probe(out).duration:.2f}s, voice {tts.resolve(args.voice, args.language)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
