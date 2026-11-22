"""Production wiring: the real Whisper, FFmpeg, TTS and critic behind the Deps interface."""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import load_platforms
from ..critic import critique as run_critic
from ..edl.compiler import render_edl
from ..llm.client import LLMClient, build_client
from ..perception import PerceptionConfig, build_perception_doc
from ..runtime import ModelManager
from ..schemas import EDL, BriefSpec
from .state import AgentConfig, Deps

log = logging.getLogger("videoagent.agent")


def build_deps(cfg: AgentConfig | None = None, client: LLMClient | None = None) -> Deps:
    cfg = cfg or AgentConfig()
    client = client or build_client()
    platforms = load_platforms()
    manager = ModelManager()  # one GPU budget shared by Whisper and the TTS voices

    tts = None
    try:
        from ..tools.tts_config import build_tts

        tts = build_tts(manager=manager)  # builds backends only; no model is loaded until a line is spoken
    except Exception as exc:  # noqa: BLE001 - a bad tts.yaml must not stop videos that need no voice
        log.warning("voiceover unavailable: %s", exc)

    def perceive(path: Path, name: str):
        pcfg = PerceptionConfig(whisper_model=cfg.whisper_model, language=cfg.speech_language)
        return build_perception_doc(path, pcfg, cfg.cache_dir, source_name=name, model_manager=manager)

    def render(edl: EDL, dst: Path, brief: BriefSpec) -> Path:
        return render_edl(
            edl, dst, font=cfg.font, fonts_dir=cfg.fonts_dir, reframe_mode=cfg.reframe_mode,  # type: ignore[arg-type]
            tts=tts if edl.voiceover else None, language=brief.language,
        )

    def critique(video, edl, brief, docs, platform):
        return run_critic(
            video, platforms[platform], platform=platform, brief=brief, edl=edl, docs=docs,
            client=client if cfg.llm_review else None, review_frames=cfg.review_frames,
        )

    return Deps(client=client, platforms=platforms, perceive=perceive, render=render,
                critique=critique, voiceover_available=tts is not None)
