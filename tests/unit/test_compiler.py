import json
import shutil
import subprocess

import pytest

from videoagent.config import PlatformSpec, load_platforms
from videoagent.edl import EDLValidationError, compile_edl, render_edl
from videoagent.edl.compiler import pick_reframe_mode
from videoagent.schemas.edl import EDL, AudioSpec, Caption, Clip, Overlay
from videoagent.tools._ffmpeg import MediaInfo, probe

SHORTS = PlatformSpec(aspect="9:16", width=1080, height=1920, fps=30, video_bitrate="6M",
                      audio_bitrate="192k", loudness_lufs=-14, max_duration_s=60,
                      caption_safe_zone_bottom_px=380)
YOUTUBE = PlatformSpec(aspect="16:9", width=1920, height=1080, fps=30, video_bitrate="8M",
                       audio_bitrate="192k", loudness_lufs=-14, max_duration_s=1200)
LANDSCAPE = MediaInfo(duration=10, width=1280, height=720, fps=30, has_audio=True)
SILENT = MediaInfo(duration=10, width=1280, height=720, fps=30, has_audio=False)


def edl(**kw) -> EDL:
    base = dict(platform="shorts", title="T", clips=[Clip(source="a.mp4", start=1, end=5)])
    base.update(kw)
    return EDL(**base)


# ---- graph structure (no FFmpeg needed) ------------------------------------

def test_reframe_mode_auto():
    assert pick_reframe_mode(LANDSCAPE, SHORTS, "auto") == "blur"      # 16:9 -> 9:16 keeps full frame
    assert pick_reframe_mode(LANDSCAPE, YOUTUBE, "auto") == "crop"     # same aspect: crop is a no-op
    assert pick_reframe_mode(LANDSCAPE, SHORTS, "pad") == "pad"        # explicit wins


def test_graph_has_trim_concat_and_maps(tmp_path):
    e = edl(clips=[Clip(source="a.mp4", start=1, end=5), Clip(source="a.mp4", start=6, end=9)])
    c = compile_edl(e, SHORTS, {"a.mp4": LANDSCAPE}, tmp_path)
    g = c.filter_graph
    assert "trim=start=1.000:end=5.000" in g and "trim=start=6.000:end=9.000" in g
    assert "concat=n=2:v=1:a=1[vcat][acat]" in g
    assert c.maps == ["-map", "[vout]", "-map", "[aout]"]
    assert c.duration == 7.0
    assert c.input_args == ["-i", "a.mp4"]  # one input even though used twice


def test_graph_speed_changes_pts_and_tempo(tmp_path):
    c = compile_edl(edl(clips=[Clip(source="a.mp4", start=0, end=4, speed=2.0)]),
                    SHORTS, {"a.mp4": LANDSCAPE}, tmp_path)
    assert "setpts=(PTS-STARTPTS)/2" in c.filter_graph
    assert "atempo=2.0000" in c.filter_graph
    assert c.duration == 2.0


def test_graph_fades(tmp_path):
    clip = Clip(source="a.mp4", start=0, end=4, fade_in_s=0.5, fade_out_s=1.0)
    g = compile_edl(edl(clips=[clip]), SHORTS, {"a.mp4": LANDSCAPE}, tmp_path).filter_graph
    assert "fade=t=in:st=0:d=0.500" in g and "fade=t=out:st=3.000:d=1.000" in g
    assert "afade=t=in" in g and "afade=t=out:st=3.000" in g


def test_silent_source_has_no_audio_mapped(tmp_path):
    c = compile_edl(edl(), SHORTS, {"a.mp4": SILENT}, tmp_path)
    assert not c.has_audio and c.maps == ["-map", "[vout]"]
    assert "a=0[vcat]" in c.filter_graph


def test_mixed_sources_pad_silence_for_clip_without_audio(tmp_path):
    e = edl(clips=[Clip(source="a.mp4", start=0, end=2), Clip(source="b.mp4", start=0, end=3)])
    c = compile_edl(e, SHORTS, {"a.mp4": LANDSCAPE, "b.mp4": SILENT}, tmp_path)
    assert "anullsrc" in c.filter_graph and "atrim=duration=3.000" in c.filter_graph
    assert c.has_audio


def test_keep_original_false_drops_clip_audio(tmp_path):
    c = compile_edl(edl(audio=AudioSpec(keep_original=False)), SHORTS, {"a.mp4": LANDSCAPE}, tmp_path)
    assert not c.has_audio


def test_captions_and_overlays_write_one_ass_file(tmp_path):
    e = edl(
        captions=[Caption(start=0, end=2, text="Hello")],
        overlays=[Overlay(kind="title", text="Big", start=0, end=2, position="top")],
    )
    c = compile_edl(e, SHORTS, {"a.mp4": LANDSCAPE}, tmp_path)
    assert c.ass_path == tmp_path / "captions.ass" and c.ass_path.exists()
    assert "Hello" in c.ass_text and "Big" in c.ass_text and "Style: Title," in c.ass_text
    assert "ass=filename=" in c.filter_graph


def test_no_text_means_no_ass_filter(tmp_path):
    c = compile_edl(edl(), SHORTS, {"a.mp4": LANDSCAPE}, tmp_path)
    assert c.ass_path is None and "ass=" not in c.filter_graph


def test_logo_adds_looped_input(tmp_path):
    e = edl(overlays=[Overlay(kind="logo", asset="logo.png", start=0, end=3)])
    c = compile_edl(e, SHORTS, {"a.mp4": LANDSCAPE}, tmp_path)
    assert c.input_args[-4:] == ["-loop", "1", "-i", "logo.png"]
    assert "[1:v]scale=" in c.filter_graph


def test_music_input_is_looped_and_trimmed_to_total(tmp_path):
    e = edl(audio=AudioSpec(music="m.wav", music_volume=0.3))
    c = compile_edl(e, SHORTS, {"a.mp4": LANDSCAPE}, tmp_path)
    assert c.input_args[-4:] == ["-stream_loop", "-1", "-i", "m.wav"]
    assert "volume=0.3" in c.filter_graph and "atrim=duration=4.000" in c.filter_graph
    assert "amix=inputs=2" in c.filter_graph


def test_music_only_when_video_is_silent(tmp_path):
    c = compile_edl(edl(audio=AudioSpec(music="m.wav")), SHORTS, {"a.mp4": SILENT}, tmp_path)
    assert c.has_audio and "amix" not in c.filter_graph and "[music]anull[aout]" in c.filter_graph


def test_every_label_is_consumed_exactly_once(tmp_path):
    """FFmpeg errors if a filter output label is unused or used twice: catch it without running it."""
    import re

    e = edl(
        clips=[Clip(source="a.mp4", start=0, end=2, fade_in_s=0.2), Clip(source="a.mp4", start=3, end=5)],
        captions=[Caption(start=0, end=1, text="x")],
        overlays=[Overlay(kind="logo", asset="l.png", start=0, end=2)],
        audio=AudioSpec(music="m.wav"),
    )
    g = compile_edl(e, SHORTS, {"a.mp4": LANDSCAPE}, tmp_path).filter_graph
    produced, consumed = [], []
    for chain in g.split(";"):
        outs = re.findall(r"\[([^\]]+)\]", chain)
        # leading labels are inputs, trailing labels are outputs
        head = re.match(r"^((?:\[[^\]]+\])+)", chain)
        tail = re.search(r"((?:\[[^\]]+\])+)$", chain)
        ins = re.findall(r"\[([^\]]+)\]", head.group(1)) if head else []
        out = re.findall(r"\[([^\]]+)\]", tail.group(1)) if tail and tail.group(1) != (head.group(1) if head else None) else []
        consumed += [i for i in ins if not re.fullmatch(r"\d+:[av]", i)]
        produced += out
    produced = [p for p in produced if p not in ("vout", "aout")]
    assert sorted(produced) == sorted(consumed), (sorted(produced), sorted(consumed))
    assert len(set(produced)) == len(produced)


# ---- real renders ----------------------------------------------------------

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@pytest.fixture
def workspace(tmp_path, sample_clip, music_file):
    shutil.copy(sample_clip, tmp_path / "raw.mp4")
    shutil.copy(music_file, tmp_path / "music.wav")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "color=c=orange:s=200x100:d=1", "-frames:v", "1", str(tmp_path / "logo.png")],
        check=True,
    )
    return tmp_path


@needs_ffmpeg
def test_render_full_edl_to_shorts(workspace):
    e = EDL(
        platform="shorts", title="Demo",
        clips=[Clip(source="raw.mp4", start=1, end=4, fade_in_s=0.2),
               Clip(source="raw.mp4", start=5, end=9, speed=2.0)],
        captions=[Caption(start=0.2, end=2.5, text="Hello world"),
                  Caption(start=3.0, end=4.9, text="مرحبا بكم في العرض")],
        overlays=[Overlay(kind="title", text="Demo", start=0, end=2, position="top"),
                  Overlay(kind="logo", asset="logo.png", start=0, end=5, position="top_right"),
                  Overlay(kind="cta", text="Subscribe", start=3, end=5)],
        audio=AudioSpec(music="music.wav", music_volume=0.2),
    )
    out = render_edl(e, workspace / "out" / "final.mp4", base_dir=workspace, font="DejaVu Sans")
    info = probe(out)
    assert (info.width, info.height) == (1080, 1920)
    assert info.duration == pytest.approx(5.0, abs=0.3)  # 3 s + 4 s / 2x
    assert info.has_audio and info.fps == pytest.approx(30, abs=0.1)


@needs_ffmpeg
def test_burned_in_text_actually_changes_pixels(workspace):
    """A caption must really be drawn: compare the same frame with and without it."""
    def render(captions, name):
        e = EDL(platform="youtube", title="T", clips=[Clip(source="raw.mp4", start=0, end=3)],
                captions=captions)
        out = render_edl(e, workspace / name, base_dir=workspace, font="DejaVu Sans")
        raw = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", "1", "-i", str(out),
             "-frames:v", "1", "-vf", "crop=iw:ih/4:0:ih*3/4", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, check=True,
        ).stdout
        return raw

    plain = render([], "plain.mp4")
    for text in ("Hello world", "مرحبا بكم في العرض"):
        with_text = render([Caption(start=0, end=3, text=text)], "text.mp4")
        diff = sum(abs(a - b) for a, b in zip(plain, with_text)) / len(plain)
        assert diff > 0.5, f"caption '{text}' did not change the bottom of the frame"


@needs_ffmpeg
def test_render_rejects_invalid_edl_before_running_ffmpeg(workspace):
    e = EDL(platform="shorts", title="T", clips=[Clip(source="raw.mp4", start=0, end=99)])
    with pytest.raises(EDLValidationError) as exc:
        render_edl(e, workspace / "x.mp4", base_dir=workspace)
    assert "clip_out_of_range" in {i.code for i in exc.value.result.errors}
    assert not (workspace / "x.mp4").exists()


@needs_ffmpeg
def test_render_youtube_same_aspect_no_blur(workspace):
    e = EDL(platform="youtube", title="T", clips=[Clip(source="raw.mp4", start=0, end=2)])
    out = render_edl(e, workspace / "yt.mp4", base_dir=workspace)
    info = probe(out)
    assert (info.width, info.height) == (1920, 1080)


@needs_ffmpeg
def test_render_keeps_master_when_asked(workspace):
    e = EDL(platform="youtube", title="T", clips=[Clip(source="raw.mp4", start=0, end=2)])
    out = render_edl(e, workspace / "k.mp4", base_dir=workspace, keep_master=True)
    assert (workspace / "k.master.mp4").exists() and out.exists()


@needs_ffmpeg
def test_hand_written_json_edl_roundtrip(workspace):
    data = {"platform": "youtube", "title": "T",
            "clips": [{"source": "raw.mp4", "start": 0, "end": 2}]}
    e = EDL(**json.loads(json.dumps(data)))
    assert probe(render_edl(e, workspace / "j.mp4", base_dir=workspace)).duration == pytest.approx(2, abs=0.3)


def test_platform_config_loads():
    assert {"youtube", "shorts", "tiktok"} <= load_platforms().keys()
