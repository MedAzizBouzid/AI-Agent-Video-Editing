"""The whole pipeline with real FFmpeg and the real critic; only the LLM and Whisper are scripted."""

import shutil
from pathlib import Path

import pytest

from videoagent.agent import AgentConfig, Deps, run_pipeline
from videoagent.critic import critique as run_critic
from videoagent.edl import render_edl
from videoagent.llm.client import LLMClient
from videoagent.schemas import EDL
from videoagent.tools._ffmpeg import probe

from .agent_helpers import PLATFORMS, copy_json, edl_json, make_doc, report
from .llm_helpers import FakeBackend, make_chain, ok

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")

SPEECH = [(1.0, 4.0, "مرحبا بكم في عرضنا الجديد"), (6.0, 9.0, "Découvrez la nouvelle collection")]


def real_deps(client, doc, font="DejaVu Sans"):
    return Deps(
        client=client, platforms=PLATFORMS,
        perceive=lambda path, name: doc.model_copy(update={"source": doc.source.model_copy(update={"path": name})}),
        render=lambda edl, dst, brief: render_edl(edl, dst, font=font, reframe_mode="blur"),
        critique=lambda video, edl, brief, docs, platform: run_critic(
            video, PLATFORMS[platform], platform=platform, brief=brief, edl=edl, docs=docs, client=None),
        voiceover_available=False,
    )


def test_footage_to_a_checked_vertical_video_with_burned_in_arabic_captions(sample_clip, tmp_path):
    chain, _ = make_chain(("main", FakeBackend(ok(edl_json(clips=[(1, 4), (6, 9)], source="sample.mp4")), ok(copy_json("عرض"))), {}))
    doc = make_doc(duration=10.0, speech=SPEECH)
    from videoagent.schemas import BriefSpec
    brief = BriefSpec(inputs=[str(sample_clip)], goal="teaser", platforms=["shorts"], language="ar")

    [res] = run_pipeline([sample_clip], real_deps(LLMClient(chain), doc), AgentConfig(), out_dir=tmp_path / "out",
                         brief=brief)

    assert res.status in ("ready", "needs_review"), f"{res.reason}\n{res.report.format_for_planner() if res.report else res.events}"
    video = Path(res.outputs["video"])
    info = probe(video)
    assert (info.width, info.height) == (1080, 1920)                 # reframed to the shorts format
    assert 5.5 < info.duration < 6.5                                  # two 3-second clips
    assert [c.text for c in res.edl.captions] == [s[2] for s in SPEECH]   # captions came from the transcript
    assert next(c for c in res.report.checks if c.name == "captions_present").passed
    saved = EDL.model_validate_json(Path(res.outputs["edl"]).read_text(encoding="utf-8"))
    assert saved.clips[0].source == "sample.mp4"                      # the plan keeps the short name
