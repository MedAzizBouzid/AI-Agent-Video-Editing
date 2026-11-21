import re
import shutil
import subprocess
from pathlib import Path

import pytest

from videoagent.config import load_platforms
from videoagent.edl import compile_edl, render_edl, validate_edl
from videoagent.schemas import EDL, AudioSpec, Clip, VoiceoverLine
from videoagent.tools._ffmpeg import MediaInfo, probe
from videoagent.tools.voiceover import (
    PiperBackend, Route, RoutedTTS, ScheduledLine, VoiceoverError, VoiceoverUnavailable,
)

from .graph_helpers import assert_labels_balanced
from .test_voiceover import fake_piper  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
YT = load_platforms()["youtube"]
TONE = MediaInfo(duration=10, width=1920, height=1080, fps=30, has_audio=True)
SILENT = MediaInfo(duration=10, width=1920, height=1080, fps=30, has_audio=False)


def line(text, start):
    return VoiceoverLine(text=text, start=start)


def edl(**kw):
    base = dict(platform="youtube", title="T", clips=[Clip(source="a.mp4", start=0, end=4)])
    base.update(kw)
    return EDL(**base)


def scheduled(*items):
    return [ScheduledLine(i, "t", Path(f"/v/{i}.wav"), start, dur) for i, (start, dur) in enumerate(items)]


# ---- graph structure (no rendering) ----------------------------------------

def test_voice_is_delayed_and_the_background_is_ducked(tmp_path):
    c = compile_edl(edl(), YT, {"a.mp4": TONE}, tmp_path, voiceover=scheduled((1.5, 1.0)))
    g = c.filter_graph
    assert "adelay=1500|1500" in g
    assert "[vo]asplit=2[vo_sc][vo_mix]" in g
    assert "[acat][vo_sc]sidechaincompress=threshold=0.04:ratio=8" in g
    assert "[ducked][vo_mix]amix=inputs=2:duration=first" in g
    assert g.endswith("[avoiced]anull[aout]") or "[avoiced]anull[aout]" in g
    assert c.input_args[-2:] == ["-i", "/v/0.wav"] and c.has_audio
    assert_labels_balanced(g)


def test_several_lines_are_mixed_before_ducking(tmp_path):
    c = compile_edl(edl(), YT, {"a.mp4": TONE}, tmp_path, voiceover=scheduled((0.5, 1.0), (2.5, 1.0)))
    assert "[vo0][vo1]amix=inputs=2:duration=longest" in c.filter_graph
    assert "adelay=500|500" in c.filter_graph and "adelay=2500|2500" in c.filter_graph
    assert c.input_args[-4:] == ["-i", "/v/0.wav", "-i", "/v/1.wav"]
    assert_labels_balanced(c.filter_graph)


def test_music_and_voice_both_sit_under_the_voice_ducking(tmp_path):
    e = edl(audio=AudioSpec(music="m.wav"))
    c = compile_edl(e, YT, {"a.mp4": TONE}, tmp_path, voiceover=scheduled((1.0, 1.0)))
    assert "[amixed][vo_sc]sidechaincompress" in c.filter_graph        # clips + music are the ducked bed
    assert c.input_args[-6:] == ["-stream_loop", "-1", "-i", "m.wav", "-i", "/v/0.wav"]
    assert_labels_balanced(c.filter_graph)


def test_voice_is_the_soundtrack_when_the_footage_has_no_audio(tmp_path):
    c = compile_edl(edl(), YT, {"a.mp4": SILENT}, tmp_path, voiceover=scheduled((0.2, 1.0)))
    assert "sidechaincompress" not in c.filter_graph
    assert "[vo]apad=whole_dur=4.000,atrim=duration=4.000[avoiced]" in c.filter_graph
    assert c.has_audio and c.maps == ["-map", "[vout]", "-map", "[aout]"]
    assert_labels_balanced(c.filter_graph)


def test_dropping_original_audio_leaves_just_the_voice(tmp_path):
    c = compile_edl(edl(audio=AudioSpec(keep_original=False)), YT, {"a.mp4": TONE}, tmp_path,
                    voiceover=scheduled((1.0, 1.0)))
    assert "sidechaincompress" not in c.filter_graph and "apad" in c.filter_graph


def test_no_voiceover_means_no_voice_filters(tmp_path):
    c = compile_edl(edl(), YT, {"a.mp4": TONE}, tmp_path)
    assert "sidechain" not in c.filter_graph and "adelay" not in c.filter_graph
    assert_labels_balanced(c.filter_graph)


def test_validator_only_warns_when_there_is_no_tts():
    e = edl(voiceover=[line("hello", 0)])
    no_tts = validate_edl(e, {"a.mp4": 10.0})
    assert no_tts.ok and [i.code for i in no_tts.warnings] == ["voiceover_not_rendered"]
    assert "no text-to-speech backend" in no_tts.warnings[0].message
    assert validate_edl(e, {"a.mp4": 10.0}, voiceover_available=True).issues == []


# ---- real renders ----------------------------------------------------------

@pytest.fixture
def ws(tmp_path, sample_clip, silent_clip):
    shutil.copy(sample_clip, tmp_path / "raw.mp4")        # 10 s, steady 440 Hz tone
    shutil.copy(silent_clip, tmp_path / "silent.mp4")     # 3 s, no audio
    return tmp_path


@pytest.fixture
def tts(fake_piper):  # noqa: F811
    return PiperBackend({"en": fake_piper.en, "ar": fake_piper.ar}, binary=fake_piper.binary,
                        default_language="en")


def band_db(path, hz, start, dur):
    """Mean level (dB) of the audio in a narrow band around `hz` during [start, start+dur]."""
    p = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-ss", str(start), "-t", str(dur), "-i", str(path), "-vn",
         "-af", f"bandpass=f={hz}:width_type=h:w=60,volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    m = re.search(r"mean_volume:\s*(-inf|-?[\d.]+) dB", p.stderr)
    assert m, p.stderr[-300:]
    return float("-inf") if m.group(1) == "-inf" else float(m.group(1))


def test_the_voice_is_audible_and_the_background_ducks_then_recovers(ws, tts):
    lines = [line("Hello this is a test line", 1.0), line("Another line of speech here", 5.0)]
    with_voice = EDL(platform="youtube", title="T", clips=[Clip(source="raw.mp4", start=0, end=8)],
                     voiceover=lines)
    without = with_voice.model_copy(update={"voiceover": []})
    render_edl(with_voice, ws / "v.mp4", base_dir=ws, tts=tts, language="en", keep_master=True)
    render_edl(without, ws / "n.mp4", base_dir=ws, keep_master=True)
    v, n = ws / "v.master.mp4", ws / "n.master.mp4"            # masters: before loudness normalisation

    speaking = (1.2, 1.0)                                       # inside the first line (1.0 - 2.7 s)
    assert band_db(v, 1000, *speaking) > band_db(n, 1000, *speaking) + 15       # the voice is there
    assert band_db(v, 440, *speaking) < band_db(n, 440, *speaking) - 3          # the tone is ducked
    quiet = (3.6, 1.0)                                          # between the lines
    assert abs(band_db(v, 440, *quiet) - band_db(n, 440, *quiet)) < 1.5         # ...and back to normal
    assert band_db(v, 1000, *quiet) < band_db(v, 1000, *speaking) - 20          # no voice between lines
    assert band_db(v, 1000, 5.2, 1.0) > band_db(n, 1000, 5.2, 1.0) + 15        # second line placed at 5 s


def test_footage_without_audio_gets_a_voice_track(ws, tts):
    e = EDL(platform="youtube", title="T", clips=[Clip(source="silent.mp4", start=0, end=3)],
            voiceover=[line("Hello there", 0.2)])
    out = render_edl(e, ws / "o.mp4", base_dir=ws, tts=tts, language="en", keep_master=True)
    info = probe(out)
    assert info.has_audio and info.duration == pytest.approx(3.0, abs=0.3)
    assert band_db(ws / "o.master.mp4", 1000, 0.3, 0.5) > -45


def test_voiceover_without_a_tts_backend_fails_loudly(ws):
    e = EDL(platform="youtube", title="T", clips=[Clip(source="raw.mp4", start=0, end=3)],
            voiceover=[line("hello", 0)])
    with pytest.raises(VoiceoverUnavailable, match="no TTS backend"):
        render_edl(e, ws / "x.mp4", base_dir=ws)
    assert not (ws / "x.mp4").exists()


def test_speech_that_does_not_fit_is_rejected_before_rendering(ws, tts):
    e = EDL(platform="youtube", title="T", clips=[Clip(source="raw.mp4", start=0, end=3)],
            voiceover=[line("A very long sentence that cannot possibly fit before the video ends", 2.0)])
    with pytest.raises(VoiceoverError, match="too long"):
        render_edl(e, ws / "x.mp4", base_dir=ws, tts=tts, language="en")
    assert not (ws / "x.mp4").exists()


def test_rerendering_reuses_synthesised_audio(ws, tts, fake_piper):  # noqa: F811
    e = EDL(platform="youtube", title="T", clips=[Clip(source="raw.mp4", start=0, end=6)],
            voiceover=[line("First line", 0.5), line("Second line", 3.0)])
    cache = ws / "tts-cache"
    render_edl(e, ws / "1.mp4", base_dir=ws, tts=tts, language="en", tts_cache_dir=cache)
    assert len(fake_piper.calls()) == 2
    edited = e.model_copy(update={"voiceover": [line("First line", 0.5), line("Changed second line", 3.0)]})
    render_edl(edited, ws / "2.mp4", base_dir=ws, tts=tts, language="en", tts_cache_dir=cache)
    assert len(fake_piper.calls()) == 3                          # only the changed line was re-synthesised


def test_the_brief_language_selects_the_voice(ws, fake_piper):  # noqa: F811
    piper = PiperBackend({"en": fake_piper.en, "ar": fake_piper.ar}, binary=fake_piper.binary)
    routed = RoutedTTS({"piper": piper}, {"en": Route("piper"), "ar": Route("piper")}, "en")
    e = EDL(platform="youtube", title="T", clips=[Clip(source="raw.mp4", start=0, end=5)],
            voiceover=[line("مرحبا بكم", 0.5)])
    render_edl(e, ws / "ar.mp4", base_dir=ws, tts=routed, language="ar-TN")
    assert fake_piper.calls()[0].split("|")[0] == fake_piper.ar
