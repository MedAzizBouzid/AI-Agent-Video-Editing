import pytest

from videoagent.config import PlatformSpec
from videoagent.edl.validator import EDLValidationError, validate_edl
from videoagent.schemas.edl import EDL, AudioSpec, Caption, Clip, Overlay, VoiceoverLine

SOURCES = {"a.mp4": 10.0, "b.mp4": 20.0}
SHORTS = PlatformSpec(aspect="9:16", width=1080, height=1920, fps=30, video_bitrate="6M",
                      audio_bitrate="192k", loudness_lufs=-14, max_duration_s=60)


def make(**kw) -> EDL:
    base = dict(platform="shorts", title="T", clips=[Clip(source="a.mp4", start=0, end=5)])
    base.update(kw)
    return EDL(**base)


def codes(result):
    return {i.code for i in result.issues}


def test_valid_edl_has_no_issues():
    result = validate_edl(make(), SOURCES, SHORTS)
    assert result.ok and result.issues == []
    assert result.format_for_llm() == "EDL is valid."


def test_unknown_source_lists_available_files():
    r = validate_edl(make(clips=[Clip(source="zzz.mp4", start=0, end=1)]), SOURCES)
    assert codes(r) == {"unknown_source"}
    assert "a.mp4" in r.errors[0].message and "b.mp4" in r.errors[0].message


def test_clip_out_of_range_says_how_to_fix():
    r = validate_edl(make(clips=[Clip(source="a.mp4", start=2, end=12)]), SOURCES)
    assert codes(r) == {"clip_out_of_range"}
    assert "end <= 10.00" in r.errors[0].message


def test_clip_start_beyond_source():
    r = validate_edl(make(clips=[Clip(source="a.mp4", start=10, end=11)]), SOURCES)
    assert "clip_start_beyond_source" in codes(r)


def test_clip_end_within_epsilon_is_ok():
    r = validate_edl(make(clips=[Clip(source="a.mp4", start=5, end=10.03)]), SOURCES)
    assert r.ok


def test_clip_too_short():
    r = validate_edl(make(clips=[Clip(source="a.mp4", start=0, end=0.1)]), SOURCES)
    assert "clip_too_short" in codes(r)


def test_speed_affects_min_length_and_total():
    # 8s of source at 4x = 2s output
    edl = make(clips=[Clip(source="a.mp4", start=0, end=8, speed=4.0)])
    assert validate_edl(edl, SOURCES, SHORTS).ok
    assert edl.total_duration == 2.0


def test_fades_longer_than_clip():
    clip = Clip(source="a.mp4", start=0, end=1, fade_in_s=0.8, fade_out_s=0.8)
    assert "fades_too_long" in codes(validate_edl(make(clips=[clip]), SOURCES))


def test_platform_max_duration():
    clips = [Clip(source="b.mp4", start=0, end=20)] * 4  # 80 s > 60 s
    r = validate_edl(make(clips=clips), SOURCES, SHORTS)
    assert "too_long" in codes(r)
    assert "20.0s" in r.errors[0].message  # tells the planner how much to cut


def test_platform_check_skipped_without_spec():
    clips = [Clip(source="b.mp4", start=0, end=20)] * 4
    assert validate_edl(make(clips=clips), SOURCES).ok


def test_empty_title():
    assert "empty_title" in codes(validate_edl(make(title="  "), SOURCES))


# ---- captions --------------------------------------------------------------

def test_caption_beyond_video_mentions_output_timeline():
    r = validate_edl(make(captions=[Caption(start=4, end=9, text="hello there")]), SOURCES)
    assert "caption_beyond_video" in codes(r)
    assert "OUTPUT timeline" in r.errors[0].message


def test_caption_overlap_detected_regardless_of_list_order():
    caps = [Caption(start=2, end=4, text="second one"), Caption(start=0, end=3, text="first one")]
    r = validate_edl(make(captions=caps), SOURCES)
    assert "caption_overlap" in codes(r)


def test_adjacent_captions_are_fine():
    caps = [Caption(start=0, end=2, text="one two"), Caption(start=2, end=4, text="three four")]
    assert validate_edl(make(captions=caps), SOURCES).ok


def test_caption_reading_speed_and_brevity_are_warnings_not_errors():
    long_fast = Caption(start=0, end=1, text="x" * 60)
    brief = Caption(start=2, end=2.1, text="hi")
    r = validate_edl(make(captions=[long_fast, brief]), SOURCES)
    assert r.ok
    assert codes(r) == {"caption_too_fast", "caption_too_brief"}


def test_empty_caption_text():
    r = validate_edl(make(captions=[Caption(start=0, end=1, text=" ")]), SOURCES)
    assert "empty_caption" in codes(r)


# ---- overlays / assets / audio / voiceover --------------------------------

def test_overlay_rules(tmp_path):
    logo = tmp_path / "logo.png"
    logo.write_bytes(b"x")
    ok = make(overlays=[Overlay(kind="logo", asset="logo.png", start=0, end=3)])
    assert validate_edl(ok, SOURCES, base_dir=tmp_path).ok
    assert "asset_missing" in codes(validate_edl(ok, SOURCES, base_dir=tmp_path / "nope"))

    bad = make(overlays=[
        Overlay(kind="title", start=0, end=2),                       # no text
        Overlay(kind="logo", start=0, end=2),                        # no asset
        Overlay(kind="cta", text="Buy", start=3, end=9),             # beyond video (5 s)
    ])
    assert codes(validate_edl(bad, SOURCES)) == {
        "overlay_without_text", "logo_without_asset", "overlay_beyond_video",
    }


def test_missing_music_file(tmp_path):
    edl = make(audio=AudioSpec(music="song.mp3"))
    assert "asset_missing" in codes(validate_edl(edl, SOURCES, base_dir=tmp_path))
    (tmp_path / "song.mp3").write_bytes(b"x")
    assert validate_edl(edl, SOURCES, base_dir=tmp_path).ok


def test_voiceover_is_warned_not_rendered():
    r = validate_edl(make(voiceover=[VoiceoverLine(text="hello", start=0)]), SOURCES)
    assert r.ok and codes(r) == {"voiceover_not_rendered"}
    late = validate_edl(make(voiceover=[VoiceoverLine(text="hello", start=99)]), SOURCES)
    assert "voiceover_beyond_video" in codes(late)


def test_format_for_llm_lists_paths():
    r = validate_edl(make(clips=[Clip(source="zzz.mp4", start=0, end=1)]), SOURCES)
    assert r.format_for_llm().startswith("- [error] clips[0]:")


def test_error_exception_carries_result():
    r = validate_edl(make(title=""), SOURCES)
    err = EDLValidationError(r)
    assert err.result is r and "empty" in str(err)
