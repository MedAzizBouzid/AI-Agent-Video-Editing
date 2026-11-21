import json
import shutil
import subprocess

import pytest

from videoagent.config import PlatformSpec, load_platforms
from videoagent.critic.checks import (
    SEVERITY, Thresholds, black_outside_edges, check_audio_present, check_captions_present,
    check_caption_safe_zone, check_clean_cuts, check_duration, check_playable, check_resolution,
    issues_from_checks, parse_blackdetect, parse_freezedetect, run_checks, score_from_checks,
)
from videoagent.edl import compile_edl, render_edl
from videoagent.schemas import EDL, Caption, Clip, Overlay
from videoagent.schemas.critic import CheckResult
from videoagent.schemas.perception import PerceptionDoc, Silence, SourceInfo, TranscriptSegment
from videoagent.tools._ffmpeg import MediaInfo, probe

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
T = Thresholds()
YT = load_platforms()["youtube"]
SHORTS = load_platforms()["shorts"]


def info(w=1920, h=1080, fps=30.0, dur=10.0, audio=True):
    return MediaInfo(duration=dur, width=w, height=h, fps=fps, has_audio=audio)


def ff(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


# ---- parsing helpers -------------------------------------------------------

def test_parse_blackdetect():
    log = "[blackdetect @ 0x1] black_start:2 black_end:5.04 black_duration:3.04\n"
    assert parse_blackdetect(log) == [(2.0, 5.04)]
    assert parse_blackdetect("nothing") == []


def test_black_at_the_edges_is_a_fade_not_a_problem():
    assert black_outside_edges([(0.0, 0.4), (9.6, 10.0)], 10.0, 0.5) == 0.0
    assert black_outside_edges([(0.0, 2.0)], 10.0, 0.5) == pytest.approx(1.5)   # only the part past the edge
    assert black_outside_edges([(4.0, 6.0)], 10.0, 0.5) == pytest.approx(2.0)


def test_parse_freezedetect_including_an_open_freeze():
    log = ("lavfi.freezedetect.freeze_start: 3\n lavfi.freezedetect.freeze_end: 9.5\n"
           "lavfi.freezedetect.freeze_start: 12\n")
    assert parse_freezedetect(log, 15.0) == [(3.0, 9.5), (12.0, 15.0)]


# ---- file-level checks (no FFmpeg run) ------------------------------------

def test_playable_and_resolution_and_audio():
    assert check_playable(info()).passed
    assert not check_playable(info(w=0, h=0)).passed
    assert check_resolution(info(), YT, T).passed
    assert not check_resolution(info(w=1280, h=720), YT, T).passed
    assert not check_resolution(info(fps=24), YT, T).passed
    assert check_resolution(info(fps=29.97), YT, T).passed   # within tolerance
    assert not check_audio_present(info(audio=False)).passed
    assert check_audio_present(info(audio=False), expect_audio=False).passed


def test_duration_rules():
    assert check_duration(info(dur=30), SHORTS, T).passed is True
    too_long = check_duration(info(dur=75), SHORTS, T)
    assert not too_long.passed and "60s" in too_long.detail
    assert not check_duration(info(dur=0.4), SHORTS, T).passed
    assert check_duration(info(dur=31), SHORTS, T, target_s=30).passed       # within 15%
    off = check_duration(info(dur=45), SHORTS, T, target_s=30)
    assert not off.passed and "target 30.0s" in off.detail


def test_captions_present():
    edl = EDL(platform="shorts", title="T", clips=[Clip(source="a", start=0, end=2)])
    assert check_captions_present(edl, expected=False).passed
    assert not check_captions_present(edl, expected=True).passed
    assert not check_captions_present(None, expected=True).passed
    edl.captions.append(Caption(start=0, end=1, text="hi"))
    assert check_captions_present(edl, expected=True).passed


# ---- caption safe zone -----------------------------------------------------

def _ass_for(platform_spec, tmp_path):
    edl = EDL(platform="shorts", title="T", clips=[Clip(source="a.mp4", start=0, end=4)],
              captions=[Caption(start=0, end=2, text="hello")],
              overlays=[Overlay(kind="cta", text="Buy", start=2, end=4, position="bottom"),
                        Overlay(kind="title", text="Top", start=0, end=2, position="top")])
    media = MediaInfo(duration=10, width=1280, height=720, fps=30, has_audio=True)
    return compile_edl(edl, platform_spec, {"a.mp4": media}, tmp_path).ass_text


def test_compiler_output_respects_the_safe_zone(tmp_path):
    assert check_caption_safe_zone(_ass_for(SHORTS, tmp_path), SHORTS).passed


def test_text_inside_the_ui_zone_is_flagged(tmp_path):
    ass = _ass_for(SHORTS, tmp_path)
    bad = ass.replace("Dialogue: 0,0:00:00.00,0:00:02.00,Caption,,0,0,0,,hello",
                      "Dialogue: 0,0:00:00.00,0:00:02.00,Caption,,0,0,50,,hello")
    result = check_caption_safe_zone(bad, SHORTS)
    assert not result.passed and "50px" in result.detail and "380px" in result.detail


def test_top_aligned_text_is_not_judged_against_the_bottom_zone():
    ass = ("Style: Caption,Arial,40,&H0,&H0,&H0,&H0,-1,0,0,0,100,100,0,0,1,2,0,2,10,10,400,1\n"
           "Dialogue: 0,0:00:00.00,0:00:02.00,Caption,,0,0,20,,{\\an8}top text\n")
    assert check_caption_safe_zone(ass, SHORTS).passed


def test_no_safe_zone_means_nothing_to_check():
    assert check_caption_safe_zone("garbage", YT).passed
    assert check_caption_safe_zone(None, SHORTS).passed


# ---- clean cuts ------------------------------------------------------------

def doc_with_sentence(silence=None):
    return PerceptionDoc(
        source=SourceInfo(path="a.mp4", sha256="x", duration_s=30, width=1280, height=720, fps=30),
        transcript=[TranscriptSegment(start=0, end=10, text="a long spoken sentence about the product"),
                    TranscriptSegment(start=12, end=20, text="the second sentence")],
        silences=[silence] if silence else [],
    )


def edl_cut(start, end):
    return EDL(platform="youtube", title="T", clips=[Clip(source="a.mp4", start=start, end=end)])


def test_cut_through_a_sentence_is_flagged_with_where_and_what():
    r = check_clean_cuts(edl_cut(0, 4), {"a.mp4": doc_with_sentence()})
    assert not r.passed
    assert "clips[0] end at 4.0s" in r.detail and "long spoken sentence" in r.detail


def test_cuts_at_boundaries_or_pauses_are_fine():
    docs = {"a.mp4": doc_with_sentence(Silence(start=5.0, end=5.6))}
    assert check_clean_cuts(edl_cut(0, 5.3), docs).passed          # inside a pause
    assert check_clean_cuts(edl_cut(0, 10.1), docs).passed         # at the sentence end
    assert check_clean_cuts(edl_cut(10.5, 12.2), docs).passed      # in the gap between sentences
    assert check_clean_cuts(edl_cut(0, 9.8), docs).passed          # within the timestamp margin


def test_clean_cuts_without_evidence_passes():
    assert check_clean_cuts(None, {"a.mp4": doc_with_sentence()}).passed
    assert check_clean_cuts(edl_cut(0, 4), None).passed
    assert check_clean_cuts(edl_cut(0, 4), {}).passed


def test_second_clip_reports_output_time():
    edl = EDL(platform="youtube", title="T", clips=[Clip(source="a.mp4", start=10.5, end=12.0),
                                                     Clip(source="a.mp4", start=14.0, end=16.0)])
    r = check_clean_cuts(edl, {"a.mp4": doc_with_sentence()})
    assert not r.passed and "clips[1] start at 14.0s (output 1.5s)" in r.detail


# ---- signal checks on real videos -----------------------------------------

@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("critic_media")
    tone = ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:d=8"]
    # black in the middle (3 s) with continuous tone
    ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:d=2.5", "-f", "lavfi", "-i",
       "color=c=black:s=640x360:r=30:d=3", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:d=2.5",
       *tone, "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]", "-map", "3:a",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "black.mp4"))
    # a static picture for 8 s
    ff("-f", "lavfi", "-i", "color=c=blue:s=640x360:r=30:d=8", *tone, "-c:v", "libx264",
       "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(d / "frozen.mp4"))
    # moving picture, 2 s tone + 4 s silence + 2 s tone
    ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:d=8",
       "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:d=2",
       "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=4",
       "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:d=2",
       "-filter_complex", "[1:a][2:a][3:a]concat=n=3:v=0:a=1[a]", "-map", "0:v", "-map", "[a]",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "deadair.mp4"))
    return d


def by_name(results):
    return {r.name: r for r in results}


def test_black_frames_in_the_middle_are_found(media):
    r = by_name(run_checks(media / "black.mp4", YT))["black_frames"]
    assert not r.passed and "2.5-5.5s" in r.detail


def test_frozen_picture_is_found(media):
    r = by_name(run_checks(media / "frozen.mp4", YT))["frozen_video"]
    assert not r.passed and r.value >= 5


def test_dead_air_is_found(media):
    r = by_name(run_checks(media / "deadair.mp4", YT))["dead_air"]
    assert not r.passed and "2.0-6.0s" in r.detail


def test_clean_video_passes_those_signal_checks(sample_clip):
    r = by_name(run_checks(sample_clip, load_platforms()["youtube"]))
    assert r["black_frames"].passed and r["dead_air"].passed and r["frozen_video"].passed


def test_unnormalised_audio_fails_loudness(sample_clip):
    r = by_name(run_checks(sample_clip, YT))["loudness"]
    assert not r.passed and "LUFS" in r.detail


def test_unreadable_file_reports_only_playable(tmp_path):
    (tmp_path / "x.mp4").write_bytes(b"not a video")
    results = run_checks(tmp_path / "x.mp4", YT)
    assert [r.name for r in results] == ["playable"] and not results[0].passed
    assert not run_checks(tmp_path / "missing.mp4", YT)[0].passed


def test_silent_audio_is_reported_as_such(tmp_path):
    p = tmp_path / "s.mp4"
    ff("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30:d=3", "-f", "lavfi", "-i",
       "anullsrc=r=48000:cl=mono:d=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(p))
    assert "silent" in by_name(run_checks(p, YT))["loudness"].detail


# ---- issues and score ------------------------------------------------------

def test_issues_carry_severity_and_fix():
    results = [CheckResult(name="black_frames", passed=False, detail="3s black"),
               CheckResult(name="duration", passed=True, detail="ok"),
               CheckResult(name="resolution", passed=False, detail="wrong size")]
    issues = issues_from_checks(results)
    assert [(i.severity, i.source) for i in issues] == [("major", "check"), ("blocker", "check")]
    assert all(i.suggested_fix for i in issues)


def test_score_weights_blockers_more_than_minor_problems():
    names = list(SEVERITY)
    all_ok = [CheckResult(name=n, passed=True) for n in names]
    assert score_from_checks(all_ok) == 1.0
    minor_fail = [CheckResult(name=n, passed=(n != "dead_air")) for n in names]
    blocker_fail = [CheckResult(name=n, passed=(n != "resolution")) for n in names]
    assert 1.0 > score_from_checks(minor_fail) > score_from_checks(blocker_fail) > 0
    assert set(SEVERITY) == set(__import__("videoagent.critic.checks", fromlist=["FIX"]).FIX)


def test_every_check_name_has_a_severity_and_a_fix():
    results = run_checks(__import__("pathlib").Path("/nonexistent.mp4"), YT)
    assert results  # smoke: the failure path still builds issues without a KeyError
    assert issues_from_checks(results)
