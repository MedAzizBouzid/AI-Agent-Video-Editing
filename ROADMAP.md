# Roadmap

Each step fits in one Claude session. Start a session by reading this file; end it with tests green, the box ticked and a commit.

- [x] **0. Skeleton**: folders, configs, pyproject, schemas (`BriefSpec`, `PerceptionDoc`, `EDL`, `CriticReport`)
- [x] **1. FFmpeg tools**: `trim`, `concat`, `reframe` (crop/pad/blur), `audio` (loudness, music mix), `export` (per platform), `registry`. 41 tests on a generated clip.
- [ ] **2. EDL validator and compiler**: `edl/validator.py`, `edl/compiler.py`, plus `captions.py` and `overlay.py`. Done when a hand-written EDL renders to a playable MP4. Test with a real Arabic caption sample (RTL shaping, fonts).
- [ ] **3. Perception** (needs your PC): transcription, scene detection, silence detection, `PerceptionDoc` cache
- [ ] **4. LLM client**: `llm/client.py`, `llm/fallback.py`, structured output with retry (needs one API key for a live test)
- [ ] **5. Planner and CLI**: `agent/` graph, planner prompt, validate and repair loop, `videoagent run`
- [ ] **6. Critic** (needs your PC): `critic/checks.py` and LLM review
- [ ] **7. Voiceover and GPU manager** (needs your PC): `voiceover.py`, `model_manager.py`
- [ ] **8. Memory and observability**: SQLite store, approval feedback, cost logging
- [ ] **9. Evals**: golden set format, `metrics.py`, `run_evals.py`
- [ ] **10. API, worker, bot**: FastAPI, queue, `docker-compose.yml`, Telegram bot, final README

## Notes

- Encoder: `VIDEOAGENT_ENCODER=auto|nvenc|libx264` (default `auto` tries NVENC and falls back to libx264). Set `nvenc` on your RTX 3050 machine.
- Tests need `ffmpeg` and `ffprobe` on PATH. They generate their own test clips.
- Run tests: `PYTHONPATH=src python -m pytest -q tests`
