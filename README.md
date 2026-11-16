# video-agent

An AI video production agent. It turns raw recordings (screen recordings or product footage) plus a short brief into publish-ready videos: edited, captioned, voiced, titled and formatted per platform, with a human approving the result.

**Success metric:** cut a 2-3 hour editing job to ~15 minutes of review, at a cost of cents per video.

## Architecture

- **Brain (planning, copy, critic):** hosted LLM API with a provider fallback chain (`configs/llm_providers.yaml`).
- **Local (your PC):** faster-whisper transcription, FFmpeg (NVENC) rendering, Piper/Kokoro voiceover.
- **Contract:** the LLM never touches FFmpeg. It outputs an EDL, `edl/validator.py` checks it, `edl/compiler.py` renders it.

```
intake -> perceive -> plan -> validate -> render -> critique -> approve
```

## Build order

1. `schemas/` and `tools/` with tests
2. `edl/` (hand-write an EDL and render it)
3. `perception/`
4. `llm/` and `agent/` (planner + validate loop)
5. `critic/` and `evals/`
6. `apps/cli.py`, then API, worker, bot

## Setup

```bash
uv sync            # or: pip install -e ".[dev]"
cp .env.example .env
pytest
```

## Eval results

_To be filled in once the golden set exists (valid-EDL rate, critic pass rate, loops, cost/video, time/video)._
