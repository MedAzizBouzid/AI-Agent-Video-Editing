import json
import shutil
import subprocess

import pytest

from videoagent.config import load_platforms
from videoagent.critic import critique
from videoagent.critic.review import describe_brief, describe_edit, review_video
from videoagent.edl import render_edl
from videoagent.llm.client import LLMClient
from videoagent.llm.errors import AuthError
from videoagent.schemas import EDL, AudioSpec, BriefSpec, Caption, Clip, Overlay
from videoagent.schemas.critic import CriticReport, Issue
from videoagent.schemas.perception import PerceptionDoc, SourceInfo, TranscriptSegment

from .llm_helpers import FakeBackend, make_chain, ok

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
YT = load_platforms()["youtube"]


def client_with(*replies, vision=True, extra=()):
    backend = FakeBackend(*replies)
    chain, _ = make_chain(("main", backend, {"vision": vision}), *extra)
    return LLMClient(chain), backend


def review_json(score=0.9, issues=(), summary="Looks fine."):
    return ok(json.dumps({"summary": summary, "score": score, "issues": list(issues)}))


DOC = PerceptionDoc(
    source=SourceInfo(path="raw.mp4", sha256="x", duration_s=10, width=1280, height=720, fps=30),
    transcript=[TranscriptSegment(start=0, end=4, text="Welcome to our new product"),
                TranscriptSegment(start=4, end=9, text="It saves you two hours a day")],
)
EDL_ = EDL(platform="youtube", title="Demo",
           clips=[Clip(source="raw.mp4", start=0, end=4), Clip(source="raw.mp4", start=4, end=8, speed=2.0)],
           captions=[Caption(start=0, end=3, text="Welcome!")],
           overlays=[Overlay(kind="title", text="NEW", start=0, end=2, position="top")],
           audio=AudioSpec(music="song.mp3", music_volume=0.2))
BRIEF = BriefSpec(inputs=["raw.mp4"], goal="30s teaser for our product", language="en",
                  target_duration_s=6.0, cta="Buy now")


# ---- text the model reads --------------------------------------------------

def test_describe_edit_lists_clips_speech_captions_overlays_music():
    text = describe_edit(EDL_, {"raw.mp4": DOC})
    assert '1. 0.0-4.0s  raw.mp4 [source 0.0-4.0s] | speech: "Welcome to our new product"' in text
    assert "2. 4.0-6.0s" in text and "at 2x" in text            # second clip is sped up: 4 s -> 2 s
    assert "[0.0-3.0s] Welcome!" in text
    assert "title 'NEW' 0.0-2.0s at top" in text
    assert "MUSIC: song.mp3 at volume 0.2" in text


def test_describe_edit_without_transcripts_says_no_speech():
    assert "no speech" in describe_edit(EDL_, None)


def test_describe_brief():
    text = describe_brief(BRIEF, YT, "youtube", 6.2)
    assert "goal: 30s teaser for our product" in text and "call to action: Buy now" in text
    assert "target duration: 6s" in text and "RENDERED DURATION: 6.2s" in text
    assert "no brief provided" in describe_brief(None, YT, "youtube", 6.2)


# ---- reviewing with and without frames ------------------------------------

def test_frames_are_sent_as_images_with_labels(sample_clip, tmp_path):
    client, backend = client_with(review_json())
    out, notes = review_video(client, sample_clip, YT, platform="youtube", brief=BRIEF, edl=EDL_,
                              docs={"raw.mp4": DOC}, max_frames=3, workdir=tmp_path)
    assert out is not None and notes == []
    sent = backend.calls[0]
    content = sent["messages"][-1]["content"]
    assert isinstance(content, list)
    assert sum(1 for p in content if p["type"] == "image_url") == 3
    labels = [p["text"] for p in content if p["type"] == "text" and p["text"].startswith("Frame")]
    assert labels[0].startswith("Frame 1 at ") and len(labels) == 3
    assert content[-1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    system = sent["messages"][0]["content"]
    assert "quality reviewer" in system and "JSON Schema" in system
    assert "30s teaser for our product" in content[0]["text"]
    assert "Welcome to our new product" in content[0]["text"]
    assert sent["response_format"] == {"type": "json_object"}


def test_without_a_vision_provider_the_review_runs_on_text_and_says_so(sample_clip):
    client, backend = client_with(review_json(), vision=False)
    out, notes = review_video(client, sample_clip, YT, platform="youtube", brief=BRIEF, edl=EDL_)
    assert out is not None
    assert any("no vision-capable provider" in n for n in notes)
    content = backend.calls[0]["messages"][-1]["content"]
    assert isinstance(content, str) and "do not comment on visuals" in content


def test_zero_frames_means_text_only_without_a_warning(sample_clip):
    client, backend = client_with(review_json())
    _, notes = review_video(client, sample_clip, YT, platform="youtube", max_frames=0)
    assert notes == [] and isinstance(backend.calls[0]["messages"][-1]["content"], str)


def test_vision_failure_falls_back_to_text_only(sample_clip):
    bad_vision = FakeBackend(AuthError("bad key"))
    text_backend = FakeBackend(review_json(issues=[{"severity": "major", "timestamp_s": 1.0,
                                                    "description": "hook is weak", "suggested_fix": "start with the demo"}]))
    chain, _ = make_chain(("eyes", bad_vision, {"vision": True}), ("text", text_backend))
    out, notes = review_video(LLMClient(chain), sample_clip, YT, platform="youtube", max_frames=2)
    assert out is not None and out.issues[0].description == "hook is weak"
    assert any("retried on text only" in n for n in notes)
    assert isinstance(text_backend.calls[0]["messages"][-1]["content"], str)


def test_unusable_llm_gives_none_and_a_note_not_an_exception(sample_clip):
    client, _ = client_with(ok("I refuse"), vision=False)
    out, notes = review_video(client, sample_clip, YT, platform="youtube", max_frames=0)
    assert out is None and any("LLM review skipped" in n for n in notes)


def test_hallucinated_issues_are_dropped(sample_clip):
    dur = 10.0
    issues = [
        {"severity": "major", "timestamp_s": 4.0, "description": "real one"},
        {"severity": "major", "timestamp_s": 400.0, "description": "past the end"},
        {"severity": "minor", "description": "   "},
    ]
    client, _ = client_with(review_json(issues=issues), vision=False)
    out, notes = review_video(client, sample_clip, YT, platform="youtube", max_frames=0)
    assert [i.description for i in out.issues] == ["real one"]
    assert any("2 LLM issue(s) dropped" in n for n in notes)


def test_llm_cannot_declare_a_blocker(sample_clip):
    bad = review_json(issues=[{"severity": "blocker", "description": "x"}])
    good = review_json()
    client, backend = client_with(bad, good, vision=False)
    out, _ = review_video(client, sample_clip, YT, platform="youtube", max_frames=0)
    assert out.issues == [] and len(backend.calls) == 2           # one repair, then valid
    assert "severity" in backend.calls[1]["messages"][-1]["content"]


# ---- critique(): checks + review together ---------------------------------

@pytest.fixture(scope="module")
def render_dir(tmp_path_factory, sample_clip):
    d = tmp_path_factory.mktemp("critique")
    shutil.copy(sample_clip, d / "raw.mp4")
    # a source with 3 s of black in the middle, tone throughout
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:d=3",
         "-f", "lavfi", "-i", "color=c=black:s=640x360:r=30:d=3",
         "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:d=3",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:d=9",
         "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]", "-map", "3:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "withblack.mp4")],
        check=True,
    )
    return d


@pytest.fixture(scope="module")
def good_render(render_dir):
    edl = EDL(platform="youtube", title="Good", clips=[Clip(source="raw.mp4", start=0, end=5)],
              captions=[Caption(start=0.5, end=4.5, text="A clean, short caption")])
    out = render_edl(edl, render_dir / "good.mp4", base_dir=render_dir, font="DejaVu Sans")
    return out, edl


def test_a_good_render_passes_every_check(good_render):
    video, edl = good_render
    brief = BriefSpec(inputs=["raw.mp4"], goal="demo", platforms=["youtube"], target_duration_s=5.0)
    report = critique(video, YT, platform="youtube", brief=brief, edl=edl)
    failed = [c.name for c in report.checks if not c.passed]
    assert failed == [] and report.verdict == "pass" and report.score == 1.0
    assert report.passed and not report.llm_reviewed and report.issues == []


def test_a_fixable_problem_gives_revise_with_an_actionable_issue(render_dir):
    edl = EDL(platform="youtube", title="Black", clips=[Clip(source="withblack.mp4", start=0, end=9)])
    video = render_edl(edl, render_dir / "black_out.mp4", base_dir=render_dir)
    report = critique(video, YT, platform="youtube", edl=edl)
    assert report.verdict == "revise"
    assert [(i.severity, i.source) for i in report.issues] == [("major", "check")]
    text = report.format_for_planner()
    assert "black_frames" in text and "3.0-6.0s" in text and "Fix:" in text and "Verdict: revise" in text


def test_a_broken_render_fails_and_does_not_spend_llm_tokens(render_dir):
    client, backend = client_with(review_json())
    wrong_size = render_dir / "raw.mp4"            # 640x360, but YouTube needs 1920x1080
    report = critique(wrong_size, YT, platform="youtube", client=client)
    assert report.verdict == "fail" and backend.calls == []
    assert any("blocking problem" in n for n in report.notes) and not report.llm_reviewed


def test_llm_findings_are_merged_and_can_force_a_revision(good_render):
    video, edl = good_render
    issue = {"severity": "major", "timestamp_s": 1.0, "description": "No hook in the first seconds",
             "suggested_fix": "Open with the product shot"}
    client, backend = client_with(review_json(score=0.5, issues=[issue], summary="Weak opening."), vision=False)
    report = critique(video, YT, platform="youtube", edl=edl, client=client, review_frames=0)
    assert report.llm_reviewed and report.summary == "Weak opening."
    assert report.verdict == "revise"
    llm_issue = next(i for i in report.issues if i.source == "llm")
    assert llm_issue.timestamp_s == 1.0 and "hook" in llm_issue.description
    assert report.score == pytest.approx(0.6 * 1.0 + 0.4 * 0.5, abs=0.01)
    assert "@1.0s" in report.format_for_planner() and "Weak opening." in report.format_for_planner()
    assert client.usage.records[0].tag == "critic"


def test_clean_llm_review_keeps_a_pass(good_render):
    video, edl = good_render
    client, _ = client_with(review_json(score=0.95), vision=False)
    report = critique(video, YT, platform="youtube", edl=edl, client=client, review_frames=0)
    assert report.verdict == "pass" and report.llm_reviewed and report.score > 0.95


def test_dead_llm_does_not_break_the_critique(good_render):
    video, edl = good_render
    client, _ = client_with(AuthError("nope"), vision=False)
    report = critique(video, YT, platform="youtube", edl=edl, client=client, review_frames=0)
    assert report.verdict == "pass" and not report.llm_reviewed
    assert any("LLM review skipped" in n for n in report.notes)


def test_report_round_trips_through_json():
    report = CriticReport(issues=[Issue(severity="minor", description="x", timestamp_s=2.0)], score=0.8,
                          verdict="pass", notes=["n"])
    assert CriticReport.model_validate_json(report.model_dump_json()) == report
    assert "- no issues found" in CriticReport(verdict="pass", score=1.0).format_for_planner()


def test_format_for_planner_sorts_by_severity():
    r = CriticReport(issues=[Issue(severity="minor", description="polish"),
                             Issue(severity="blocker", description="broken"),
                             Issue(severity="major", description="cut", timestamp_s=3.0)])
    lines = r.format_for_planner().splitlines()
    assert [l.split("]")[0] for l in lines[1:]] == ["- [blocker", "- [major", "- [minor"]
