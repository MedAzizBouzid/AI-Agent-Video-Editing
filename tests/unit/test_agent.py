import json
from pathlib import Path

import pytest

from videoagent.agent import AgentConfig, parse_brief, run_pipeline
from videoagent.agent.nodes.perceive import short_names
from videoagent.agent.nodes.plan import build_plan_message
from videoagent.llm.errors import AuthError
from videoagent.llm.prompts import load_prompt
from videoagent.llm.usage import UsageLog
from videoagent.schemas import EDL, Caption

from .agent_helpers import (
    PLATFORMS, Harness, copy_json, edl_json, intake_json, major, make_doc, replies, report,
)
from .llm_helpers import ok


def h(tmp_path, *texts, critiques=(report(),), **kw):
    return Harness(tmp_path, replies(*texts), critiques, **kw)


# ---- the happy path --------------------------------------------------------------------------

def test_happy_path_renders_critiques_and_writes_everything(tmp_path):
    t = h(tmp_path, edl_json(), copy_json())
    [res] = t.run()
    assert res.status == "ready" and res.reason == ""
    assert Path(res.outputs["video"]).read_bytes() == b"render 1"
    for key in ("edl", "report", "copy", "run"):
        assert Path(res.outputs[key]).is_file()
    assert res.copy.title == "Great title"
    assert [e.split(":")[0] for e in res.events] == [
        "perceive", "plan", "validate", "render", "critique", "copy", "finish"]
    run = json.loads(Path(res.outputs["run"]).read_text(encoding="utf-8"))
    assert run["status"] == "ready" and run["llm"]["calls"] == 2


def test_captions_come_from_the_transcript_not_from_the_llm(tmp_path):
    [res] = h(tmp_path, edl_json(), copy_json()).run()
    # kept source 2-6 and 10-14 -> output 0-4 and 4-8; the third sentence (30-34) was cut
    assert [(c.start, c.end, c.text) for c in res.edl.captions] == [
        (0.0, 4.0, "first thing the client says"), (4.05, 8.0, "second thing worth keeping")]
    saved = EDL.model_validate_json(Path(res.outputs["edl"]).read_text(encoding="utf-8"))
    assert saved.captions == res.edl.captions


def test_the_graph_forces_the_platform_and_hands_the_renderer_real_paths(tmp_path):
    t = h(tmp_path, edl_json(platform="youtube"), copy_json())
    [res] = t.run()
    assert res.edl.platform == "shorts"                         # the LLM said youtube; the run is shorts
    assert t.renders[0].clips[0].source == str(t.input.resolve())  # short name replaced by the real path
    assert t.critique_calls[0][1].clips[0].source == "raw.mp4"     # the critic sees the names it has docs for


def test_planner_captions_are_kept_and_the_brief_can_switch_them_off(tmp_path):
    own = [{"start": 0, "end": 3, "text": "My own hook"}]
    [kept] = h(tmp_path, edl_json(captions=own), copy_json()).run()
    assert [c.text for c in kept.edl.captions] == ["My own hook"]
    t = h(tmp_path / "b", edl_json(captions=own), copy_json())
    [off] = t.run(brief=t.brief(captions=False))
    assert off.edl.captions == []


def test_voiceover_the_brief_did_not_ask_for_is_dropped(tmp_path):
    vo = [{"text": "hello", "start": 0}]
    [res] = h(tmp_path, edl_json(voiceover=vo), copy_json()).run()
    assert res.edl.voiceover == [] and "dropped voiceover" in " ".join(res.events)


def test_a_planner_that_returns_prose_around_json_is_repaired_by_the_client(tmp_path):
    t = h(tmp_path, "Sure! here you go: " + edl_json(clips=[(2, 6)]) + " enjoy", copy_json())
    [res] = t.run()
    assert res.status == "ready" and len(res.edl.clips) == 1


# ---- the validate loop -----------------------------------------------------------------------

def test_validator_errors_are_sent_back_verbatim_and_the_second_plan_succeeds(tmp_path):
    bad = edl_json(clips=[(2, 90)])                              # source is only 60 s long
    t = h(tmp_path, bad, edl_json(), copy_json())
    [res] = t.run()
    assert res.status == "ready"
    second = t.messages(1)
    assert "PROBLEMS found by the validator" in second
    assert "end=90.00s is past the end of 'raw.mp4' (60.00s long)" in second
    assert '"end": 90' in second                                 # the planner sees its own previous EDL
    assert len(t.renders) == 1                                   # the bad EDL was never rendered
    assert [e.split(":")[0] for e in res.events][:5] == ["perceive", "plan", "validate", "plan", "validate"]


def test_the_planner_gives_up_after_the_allowed_repairs_without_rendering(tmp_path):
    t = h(tmp_path, edl_json(clips=[(2, 90)]))                  # the same bad plan forever
    [res] = t.run(AgentConfig(max_plan_repairs=2))
    assert res.status == "failed" and "could not produce a valid EDL" in res.reason
    assert "past the end" in res.reason
    assert t.renders == [] and "video" not in res.outputs
    assert len(t.backend.calls) == 3                             # first try + 2 repairs
    assert Path(res.outputs["run"]).is_file()                    # a failed run still leaves a record


def test_planner_unavailable_fails_the_run_with_the_reason(tmp_path):
    t = h(tmp_path, AuthError("bad key"))
    t.backend.script = [AuthError("bad key")]
    [res] = t.run()
    assert res.status == "failed" and "planner failed" in res.reason and t.renders == []


# ---- the review loop -------------------------------------------------------------------------

def test_a_revise_verdict_sends_the_review_back_and_the_second_render_wins(tmp_path):
    t = h(tmp_path, edl_json(), edl_json(clips=[(10, 14), (2, 6)]), copy_json(),
          critiques=[report("revise", 0.5, [major("weak hook", 1.0, "open with the result")]), report()])
    [res] = t.run()
    assert res.status == "ready" and len(t.renders) == 2
    second = t.messages(1)
    assert "REVIEW of the video rendered from your previous EDL" in second
    assert "weak hook" in second and "open with the result" in second
    assert Path(res.outputs["video"]).read_bytes() == b"render 2"


def test_validation_errors_during_a_revision_do_not_erase_the_review(tmp_path):
    t = h(tmp_path, edl_json(), edl_json(clips=[(2, 90)]), edl_json(), copy_json(),
          critiques=[report("revise", 0.5, [major("weak hook")]), report()])
    t.run()
    third = t.messages(2)                                        # the repair of the revision
    assert "past the end" in third and "weak hook" in third


def test_after_the_allowed_revisions_the_best_render_is_delivered_for_human_review(tmp_path):
    scores = [0.5, 0.7, 0.6]
    t = h(tmp_path, edl_json(), critiques=[report("revise", s, [major()]) for s in scores])
    [res] = t.run(AgentConfig(max_revisions=2))
    assert res.status == "needs_review" and len(t.renders) == 3
    assert Path(res.outputs["video"]).read_bytes() == b"render 2"   # score 0.7, not the last one
    assert res.report.score == 0.7 and "copy" not in res.outputs


def test_a_broken_render_fails_without_replanning(tmp_path):
    t = h(tmp_path, edl_json(), critiques=[report("fail", 0.0, [major("file is unplayable")])])
    [res] = t.run()
    assert res.status == "failed" and "unplayable" in res.reason and len(t.renders) == 1


def test_the_spend_cap_stops_further_replanning(tmp_path, monkeypatch):
    monkeypatch.setattr(UsageLog, "cost_usd", property(lambda self: 9.99))
    t = h(tmp_path, edl_json(), critiques=[report("revise", 0.5, [major()])])
    [res] = t.run(AgentConfig(max_cost_usd=1.0))
    assert res.status == "needs_review" and len(t.renders) == 1
    assert "spend cap" in res.reason


# ---- render failures -------------------------------------------------------------------------

def test_a_render_error_fails_the_run_with_the_message(tmp_path):
    t = h(tmp_path, edl_json(), render_errors=[RuntimeError("ffmpeg exploded")])
    [res] = t.run()
    assert res.status == "failed" and "ffmpeg exploded" in res.reason and "video" not in res.outputs


def test_a_render_error_on_a_later_attempt_still_delivers_the_earlier_good_render(tmp_path):
    t = h(tmp_path, edl_json(), edl_json(), critiques=[report("revise", 0.6, [major()])],
          render_errors=[None, RuntimeError("disk full")])
    [res] = t.run()
    assert res.status == "needs_review" and "disk full" in res.reason
    assert Path(res.outputs["video"]).read_bytes() == b"render 1"


# ---- other modes -----------------------------------------------------------------------------

def test_plan_only_stops_before_rendering(tmp_path):
    t = h(tmp_path, edl_json())
    [res] = t.run(AgentConfig(plan_only=True))
    assert res.status == "planned" and t.renders == [] and t.critique_calls == []
    assert set(res.outputs) == {"edl", "run"} and len(res.edl.captions) == 2


def test_a_missing_input_file_fails_early(tmp_path):
    t = h(tmp_path, edl_json())
    t.input.unlink()
    [res] = t.run()
    assert res.status == "failed" and "input file not found" in res.reason
    assert t.backend.calls == []                                 # no tokens spent


def test_one_video_per_platform_with_the_right_platform_in_each_prompt(tmp_path):
    t = h(tmp_path, edl_json(), copy_json(), edl_json(), copy_json(), critiques=[report()])
    res = t.run(platforms=["shorts", "youtube"])
    assert [r.platform for r in res] == ["shorts", "youtube"]
    assert "PLATFORM: youtube (1920x1080" in t.messages(2)
    assert Path(res[0].outputs["video"]).name == "raw.shorts.mp4"
    assert Path(res[1].outputs["video"]).name == "raw.youtube.mp4"


def test_unknown_platform_is_rejected_before_any_work(tmp_path):
    t = h(tmp_path, edl_json())
    with pytest.raises(ValueError, match="unknown platform"):
        t.run(platforms=["myspace"])
    assert t.backend.calls == []


def test_copy_failure_does_not_fail_the_run(tmp_path):
    t = h(tmp_path, edl_json(), "this is not json at all")
    [res] = t.run()
    assert res.status == "ready" and res.copy.title == "Demo"
    assert "copy: failed" in " ".join(res.events)


# ---- intake ----------------------------------------------------------------------------------

def test_intake_extracts_the_brief_and_explicit_flags_win(tmp_path):
    t = h(tmp_path, intake_json(platforms=["tiktok"], language="fr", target_duration_s=30, cta="Commandez"))
    brief = parse_brief(t.client, "vidéo de 30s pour tiktok", ["a.mp4"],
                        overrides={"language": "ar", "tone": None})
    assert brief.platforms == ["tiktok"] and brief.language == "ar"        # flag beat the LLM's "fr"
    assert brief.target_duration_s == 30 and brief.cta == "Commandez" and brief.inputs == ["a.mp4"]
    assert brief.tone == "friendly, concise"                                # None override ignored


def test_intake_defaults_when_the_llm_names_no_platform_or_language(tmp_path):
    t = h(tmp_path, intake_json())
    brief = parse_brief(t.client, "make a video", ["a.mp4"])
    assert brief.platforms == ["shorts"] and brief.language == "en"


def test_run_pipeline_runs_intake_once_for_all_platforms(tmp_path):
    t = h(tmp_path, intake_json(platforms=["shorts", "youtube"]), edl_json(), copy_json(),
          edl_json(), copy_json())
    res = t.run(brief=None, brief_text="teaser for shorts and youtube")
    assert [r.platform for r in res] == ["shorts", "youtube"]
    assert sum("CLIENT REQUEST" in t.messages(i) for i in range(len(t.backend.calls))) == 1


# ---- pieces ----------------------------------------------------------------------------------

def test_short_names_are_unique_and_keep_the_extension():
    names = short_names(["/a/clip.mp4", "/b/clip.mp4", "/c/other.mov"])
    assert list(names) == ["clip.mp4", "clip_2.mp4", "other.mov"]
    assert names["clip_2.mp4"] == "/b/clip.mp4"


def test_the_plan_message_carries_brief_platform_sources_and_feedback(tmp_path):
    t = h(tmp_path, edl_json())
    prev = EDL(platform="shorts", title="T", clips=[{"source": "raw.mp4", "start": 0, "end": 5}])
    msg = build_plan_message(t.brief(tone="bold"), "shorts", PLATFORMS["shorts"], {"raw.mp4": make_doc()},
                             max_chars=10000, previous=prev, feedback="- [error] x", feedback_kind="validation")
    assert '"tone": "bold"' in msg and "maximum 60 seconds" in msg and "[2.0-6.0] first thing" in msg
    assert "inputs" not in msg.split("SOURCES")[0]               # file paths are not the planner's business
    assert "YOUR PREVIOUS EDL" in msg and "- [error] x" in msg


@pytest.mark.parametrize("name", ["planner", "intake", "copywriter", "critic"])
def test_prompts_are_written(name):
    text = load_prompt(name)
    assert "TODO" not in text and len(text) > 400


def test_the_planner_prompt_explains_the_timelines_and_the_hard_rules():
    p = load_prompt("planner")
    for needle in ("SOURCE time", "OUTPUT time", "leave `captions` as `[]`", "exact file name",
                   "Do not add a `cta` overlay", "hard cuts", "brief's `language`"):
        assert needle in p
