# Roadmap

Each step fits in one Claude session. Start a session by reading this file; end it with tests green, the box ticked and a commit.

- [x] **0. Skeleton**: folders, configs, pyproject, schemas (`BriefSpec`, `PerceptionDoc`, `EDL`, `CriticReport`)
- [x] **1. FFmpeg tools**: `trim`, `concat`, `reframe` (crop/pad/blur), `audio` (loudness, music mix), `export` (per platform), `registry`. 41 tests on a generated clip.
- [x] **2. EDL validator and compiler**: `edl/validator.py` (rules with LLM-readable messages), `edl/compiler.py` (EDL -> one `filter_complex`), `tools/captions.py` + `tools/overlay.py` (ASS/libass text, logos). Arabic RTL shaping verified on rendered frames. 107 tests. Try it: `python -m apps.render_edl examples/demo.edl.json out.mp4 --font "Arial"` (put a `raw.mp4` next to the EDL).
- [x] **3. Perception**: `perception/` silence (FFmpeg silencedetect), shots (FFmpeg scene score, no OpenCV), keyframes, transcription (faster-whisper, injectable), `build_doc` (fingerprint + config cache), `compact_for_llm` (token-lean text for the planner). 139 tests. **Whisper is only tested with a fake model here (no model download in the sandbox): run `python -m apps.perceive raw.mp4 --model small --language ar` on your PC to confirm it with real speech.**
- [x] **4. LLM client**: `llm/` OpenAI-compatible backend, `FallbackChain` (429 cooldowns with escalation, retry/backoff, auth/model-id disabling, JSON-mode auto-downgrade, `PRIVACY_STRICT`), `LLMClient.complete` / `complete_json` (extracts JSON from noisy replies, validates with Pydantic, sends the exact errors back for up to 2 repairs), per-call token/cost `UsageLog` with tags. 213 tests incl. the real SDK against a local HTTP server. **Not yet tested against a real provider: put a key in `.env`, pick a current model id with `python -m apps.llm_check --models`, edit `configs/llm_providers.yaml`, then run `python -m apps.llm_check --ping`.**
- [ ] **5. Planner and CLI**: `agent/` graph, planner prompt, validate and repair loop, `videoagent run`
- [x] **6. Critic**: `critic/checks.py` (11 measurements: playable, resolution/fps, duration, audio, loudness + true peak, black frames, dead air, frozen picture, captions present, caption safe zone, cuts through sentences) and `critic/review.py` (LLM judges the render against the brief from the edit-as-text plus sampled frames; text-only fallback when no provider has `vision: true`; hallucination guards; the LLM can never declare a blocker). `critique()` returns a `CriticReport` (pass / revise / fail) with `format_for_planner()` for the repair loop. 263 tests. **The loop that sends a "revise" verdict back to the planner is part of step 5 (not built yet). Not yet tried with a real vision model: set `vision: true` on a provider that supports images, then `python -m apps.critique out.mp4 --platform shorts --llm --goal "..."`.**
- [ ] **7. Voiceover and GPU manager** (needs your PC): `voiceover.py`, `model_manager.py`
- [ ] **8. Memory and observability**: SQLite store, approval feedback, cost logging
- [ ] **9. Evals**: golden set format, `metrics.py`, `run_evals.py`
- [ ] **10. API, worker, bot**: FastAPI, queue, `docker-compose.yml`, Telegram bot, final README

## Notes

- Text rendering uses libass (HarfBuzz + FriBidi), which shapes Arabic and handles RTL. The font must contain Arabic glyphs (Arial on Windows does; on Linux use `DejaVu Sans` or Noto Sans Arabic). Check your FFmpeg has libass: `ffmpeg -filters | grep ass`.
- Perception cache lives in `data/cache/perception` (gitignored). The planner must use `PerceptionDoc.source.path` as `clip.source`, so pass a short stable `source_name`.
- LLM providers: `vision: true` in `configs/llm_providers.yaml` marks a model that accepts images (the critic's frame review needs one). Without it the review runs on text only.
- Voiceover lines are validated but not rendered until step 7.

- Encoder: `VIDEOAGENT_ENCODER=auto|nvenc|libx264` (default `auto` tries NVENC and falls back to libx264). Set `nvenc` on your RTX 3050 machine.
- Tests need `ffmpeg` and `ffprobe` on PATH. They generate their own test clips.
- Run tests: `PYTHONPATH=src python -m pytest -q tests`
