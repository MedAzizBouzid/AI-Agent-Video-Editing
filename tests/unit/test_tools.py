import math
import shutil

import pytest

from videoagent.tools._ffmpeg import FFmpegError, probe, video_encoder_args
from videoagent.tools.audio import measure_loudness, mix_music, normalize_loudness
from videoagent.tools.concat import concat
from videoagent.tools.export import export
from videoagent.tools.reframe import reframe, reframe_graph
from videoagent.tools.registry import TOOLS, get_tool
from videoagent.tools.trim import atempo_chain, trim

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


# ---- probe / encoder -------------------------------------------------------

def test_probe_reads_metadata(sample_clip):
    info = probe(sample_clip)
    assert (info.width, info.height) == (640, 360)
    assert info.fps == pytest.approx(30, abs=0.01)
    assert info.duration == pytest.approx(10, abs=0.2)
    assert info.has_audio


def test_probe_missing_file_raises(tmp_path):
    with pytest.raises(FFmpegError):
        probe(tmp_path / "nope.mp4")


def test_encoder_env_override(monkeypatch):
    monkeypatch.setenv("VIDEOAGENT_ENCODER", "libx264")
    assert "libx264" in video_encoder_args()
    monkeypatch.setenv("VIDEOAGENT_ENCODER", "nvenc")
    assert "h264_nvenc" in video_encoder_args("8M")


def test_encoder_bitrate_args(monkeypatch):
    monkeypatch.setenv("VIDEOAGENT_ENCODER", "libx264")
    args = video_encoder_args("8M")
    assert args[args.index("-b:v") + 1] == "8M"
    assert args[args.index("-bufsize") + 1] == "16M"


# ---- trim ------------------------------------------------------------------

def test_trim_duration(sample_clip, tmp_path):
    out = trim(sample_clip, tmp_path / "t.mp4", 2, 5)
    info = probe(out)
    assert info.duration == pytest.approx(3, abs=0.2)
    assert info.has_audio


def test_trim_speed_up(sample_clip, tmp_path):
    out = trim(sample_clip, tmp_path / "t.mp4", 0, 4, speed=2.0)
    assert probe(out).duration == pytest.approx(2, abs=0.2)


def test_trim_silent_source(silent_clip, tmp_path):
    out = trim(silent_clip, tmp_path / "t.mp4", 0, 1)
    assert not probe(out).has_audio


@pytest.mark.parametrize("start,end", [(5, 2), (-1, 3), (0, 99)])
def test_trim_rejects_bad_ranges(sample_clip, tmp_path, start, end):
    with pytest.raises(ValueError):
        trim(sample_clip, tmp_path / "t.mp4", start, end)


def test_trim_rejects_bad_speed(sample_clip, tmp_path):
    with pytest.raises(ValueError):
        trim(sample_clip, tmp_path / "t.mp4", 0, 1, speed=10)


@pytest.mark.parametrize("speed", [0.25, 0.5, 1.5, 2.0, 3.0, 4.0])
def test_atempo_chain_multiplies_to_speed(speed):
    filters = atempo_chain(speed)
    factors = [float(f.split("=")[1]) for f in filters]
    assert all(0.5 <= f <= 2.0 for f in factors)
    assert math.prod(factors) == pytest.approx(speed, rel=1e-3)


# ---- concat ----------------------------------------------------------------

def test_concat_sums_durations(sample_clip, tmp_path):
    a = trim(sample_clip, tmp_path / "a.mp4", 0, 2)
    b = trim(sample_clip, tmp_path / "b.mp4", 3, 6)
    out = concat([a, b], tmp_path / "c.mp4")
    info = probe(out)
    assert info.duration == pytest.approx(5, abs=0.3)
    assert info.has_audio
    assert (info.width, info.height) == (640, 360)


def test_concat_normalises_size(sample_clip, tmp_path):
    a = trim(sample_clip, tmp_path / "a.mp4", 0, 1)
    small = reframe(sample_clip, tmp_path / "small.mp4", 320, 180)
    out = concat([a, small], tmp_path / "c.mp4")
    assert (probe(out).width, probe(out).height) == (640, 360)


def test_concat_single_clip_copies(sample_clip, tmp_path):
    out = concat([sample_clip], tmp_path / "one.mp4")
    assert probe(out).duration == pytest.approx(10, abs=0.2)


def test_concat_rejects_mixed_audio(sample_clip, silent_clip, tmp_path):
    with pytest.raises(ValueError, match="audio"):
        concat([sample_clip, silent_clip], tmp_path / "c.mp4")


def test_concat_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        concat([], tmp_path / "c.mp4")


# ---- reframe ---------------------------------------------------------------

@pytest.mark.parametrize("mode", ["crop", "pad", "blur"])
def test_reframe_to_vertical(sample_clip, tmp_path, mode):
    out = reframe(sample_clip, tmp_path / f"{mode}.mp4", 360, 640, mode=mode)
    info = probe(out)
    assert (info.width, info.height) == (360, 640)
    assert info.has_audio
    assert info.duration == pytest.approx(10, abs=0.3)


def test_reframe_rejects_odd_size():
    with pytest.raises(ValueError):
        reframe_graph(361, 640)


def test_reframe_rejects_bad_mode():
    with pytest.raises(ValueError):
        reframe_graph(360, 640, mode="stretch")  # type: ignore[arg-type]


def test_reframe_rejects_bad_crop_x():
    with pytest.raises(ValueError):
        reframe_graph(360, 640, crop_x=1.5)


# ---- audio -----------------------------------------------------------------

def test_measure_loudness_returns_floats(sample_clip):
    m = measure_loudness(sample_clip)
    assert {"input_i", "input_tp", "input_lra", "input_thresh", "target_offset"} <= m.keys()
    assert m["input_i"] < 0


def test_measure_loudness_no_audio(silent_clip):
    with pytest.raises(ValueError):
        measure_loudness(silent_clip)


def test_normalize_loudness_hits_target(sample_clip, tmp_path):
    out = normalize_loudness(sample_clip, tmp_path / "n.mp4", lufs=-14.0)
    assert measure_loudness(out)["input_i"] == pytest.approx(-14.0, abs=1.5)
    assert probe(out).duration == pytest.approx(10, abs=0.3)


def test_mix_music_keeps_video_length(sample_clip, music_file, tmp_path):
    out = mix_music(sample_clip, music_file, tmp_path / "m.mp4", music_volume=0.2)
    info = probe(out)
    assert info.has_audio
    assert info.duration == pytest.approx(10, abs=0.3)


def test_mix_music_on_silent_video(silent_clip, music_file, tmp_path):
    out = mix_music(silent_clip, music_file, tmp_path / "m.mp4")
    info = probe(out)
    assert info.has_audio
    assert info.duration == pytest.approx(3, abs=0.3)


def test_mix_music_rejects_bad_volume(sample_clip, music_file, tmp_path):
    with pytest.raises(ValueError):
        mix_music(sample_clip, music_file, tmp_path / "m.mp4", music_volume=2)


# ---- export ----------------------------------------------------------------

def test_export_shorts(sample_clip, tmp_path):
    short = trim(sample_clip, tmp_path / "s.mp4", 0, 3)
    out = export(short, tmp_path / "shorts.mp4", "shorts", mode="blur")
    info = probe(out)
    assert (info.width, info.height) == (1080, 1920)
    assert info.fps == pytest.approx(30, abs=0.1)
    assert info.has_audio
    assert measure_loudness(out)["input_i"] == pytest.approx(-14.0, abs=2.0)


def test_export_unknown_platform(sample_clip, tmp_path):
    with pytest.raises(ValueError, match="unknown platform"):
        export(sample_clip, tmp_path / "x.mp4", "myspace")


# ---- registry --------------------------------------------------------------

def test_registry_lists_tools_with_descriptions():
    assert {"trim", "concat", "reframe", "export", "normalize_loudness", "mix_music"} <= TOOLS.keys()
    assert all(t.description for t in TOOLS.values())
    assert get_tool("trim").fn is trim


def test_registry_unknown_tool():
    with pytest.raises(KeyError):
        get_tool("teleport")
