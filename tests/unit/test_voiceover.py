import os
import stat
import sys
import textwrap
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from videoagent.runtime import ModelManager
from videoagent.schemas import EDL, Clip, VoiceoverLine
from videoagent.tools.tts_config import TTSConfig, build_tts, load_tts_config
from videoagent.tools.voiceover import (
    EdgeBackend, KokoroBackend, PiperBackend, Route, RoutedTTS, ScheduledLine, TTSError,
    VoiceoverError, base_language, prepare_voiceover, resolve_voice, schedule_lines,
    synthesize_cached, write_wav,
)


# ---- helpers ---------------------------------------------------------------

def wav_info(path):
    with wave.open(str(path)) as f:
        return f.getframerate(), f.getnframes() / f.getframerate(), f.getnchannels()


@pytest.fixture
def fake_piper(tmp_path, monkeypatch):
    """A stand-in `piper` executable: honours -m/-f, reads stdin, makes a WAV whose length depends on
    the text, and logs 'model|text' so tests can see what it was asked."""
    script = tmp_path / "fakepiper.py"
    script.write_text(textwrap.dedent("""\
        import os, sys, time, wave, struct
        args = sys.argv[1:]
        model = args[args.index("-m") + 1]
        out = args[args.index("-f") + 1]
        text = sys.stdin.read()
        with open(os.environ["FAKE_PIPER_LOG"], "a") as log:
            log.write(model + "|" + text + "\\n")
        if os.environ.get("FAKE_PIPER_SLEEP"):
            time.sleep(float(os.environ["FAKE_PIPER_SLEEP"]))
        if os.environ.get("FAKE_PIPER_FAIL"):
            sys.stderr.write("boom: bad phonemes"); sys.exit(3)
        n = 0 if os.environ.get("FAKE_PIPER_EMPTY") else int(22050 * max(0.5, len(text) / 15))
        with wave.open(out, "wb") as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(22050)
            f.writeframes(b"".join(struct.pack("<h", int(8000 * (1 if (i // 11) % 2 else -1))) for i in range(n)))
        """))
    if sys.platform == "win32":  # no shebangs on Windows: a .cmd launcher runs the script
        launcher = tmp_path / "fakepiper.cmd"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\r\n')
    else:
        launcher = tmp_path / "fakepiper"
        launcher.write_text(f"#!{sys.executable}\n" + script.read_text())
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    log = tmp_path / "piper.log"
    monkeypatch.setenv("FAKE_PIPER_LOG", str(log))
    voice_en, voice_ar = tmp_path / "en.onnx", tmp_path / "ar.onnx"
    voice_en.write_bytes(b"x")
    voice_ar.write_bytes(b"x")
    return SimpleNamespace(binary=str(launcher), log=log, en=str(voice_en), ar=str(voice_ar),
                           calls=lambda: log.read_text(encoding="utf-8").splitlines() if log.exists() else [])


# ---- languages and voices --------------------------------------------------

@pytest.mark.parametrize("code,expected", [("ar-TN", "ar"), ("fr_FR", "fr"), ("EN", "en"),
                                           (None, None), ("", None), (" ar ", "ar")])
def test_base_language(code, expected):
    assert base_language(code) == expected


def test_resolve_voice_rules():
    voices = {"en": "v-en", "ar": "v-ar", "fr-CA": "v-frca"}
    assert resolve_voice(voices, "alias", "en", "en") == "alias"            # literal
    assert resolve_voice({"deep": "path/deep.onnx"}, "deep", "en", None) == "path/deep.onnx"  # alias
    assert resolve_voice(voices, None, "ar-TN", "en") == "v-ar"             # base-language fallback
    assert resolve_voice(voices, None, "fr-CA", "en") == "v-frca"           # exact match first
    assert resolve_voice(voices, None, None, "en") == "v-en"                # default only when no language


def test_an_unconfigured_language_is_an_error_not_the_default_voice():
    with pytest.raises(TTSError, match="no voice configured for language 'de'"):
        resolve_voice({"en": "v-en"}, None, "de", "en")


# ---- WAV writing -----------------------------------------------------------

def test_write_wav_roundtrip(tmp_path):
    t = np.arange(12000) / 24000
    write_wav(tmp_path / "t.wav", 0.5 * np.sin(2 * np.pi * 440 * t), 24000)
    rate, seconds, channels = wav_info(tmp_path / "t.wav")
    assert (rate, channels) == (24000, 1) and seconds == pytest.approx(0.5)


def test_write_wav_clips_out_of_range_samples(tmp_path):
    write_wav(tmp_path / "t.wav", np.array([5.0, -5.0]), 8000)
    with wave.open(str(tmp_path / "t.wav")) as f:
        frames = np.frombuffer(f.readframes(2), dtype="<i2")
    assert list(frames) == [32767, -32767]


# ---- Piper (real subprocess against a fake executable) ---------------------

def test_piper_runs_the_cli_and_returns_the_wav(fake_piper, tmp_path):
    tts = PiperBackend({"en": fake_piper.en, "ar": fake_piper.ar}, binary=fake_piper.binary, default_language="en")
    out = tts.synthesize("Hello   there\nfriend", tmp_path / "out" / "line", language="en")
    assert out == tmp_path / "out" / "line.wav" and out.stat().st_size > 44
    assert fake_piper.calls() == [f"{fake_piper.en}|Hello there friend"]       # whitespace squashed


def test_piper_picks_the_model_by_language_and_explicit_voice(fake_piper, tmp_path):
    tts = PiperBackend({"en": fake_piper.en, "ar": fake_piper.ar}, binary=fake_piper.binary)
    tts.synthesize("مرحبا", tmp_path / "a", language="ar-TN")
    tts.synthesize("hi", tmp_path / "b", voice=fake_piper.en)
    assert [c.split("|")[0] for c in fake_piper.calls()] == [fake_piper.ar, fake_piper.en]
    assert "مرحبا" in fake_piper.calls()[0]                                     # Arabic text passes intact


def test_piper_errors_say_how_to_fix_them(fake_piper, tmp_path, monkeypatch):
    voices = {"en": fake_piper.en}
    with pytest.raises(TTSError, match="not found: pip install piper-tts"):
        PiperBackend(voices, binary="/no/such/piper").synthesize("x", tmp_path / "a", language="en")
    with pytest.raises(TTSError, match="voice model not found.*huggingface"):
        PiperBackend({"en": str(tmp_path / "missing.onnx")}, binary=fake_piper.binary).synthesize(
            "x", tmp_path / "a", language="en")
    tts = PiperBackend(voices, binary=fake_piper.binary)
    monkeypatch.setenv("FAKE_PIPER_FAIL", "1")
    with pytest.raises(TTSError, match="exit 3.*bad phonemes"):
        tts.synthesize("x", tmp_path / "a", language="en")
    monkeypatch.delenv("FAKE_PIPER_FAIL")
    monkeypatch.setenv("FAKE_PIPER_EMPTY", "1")
    with pytest.raises(TTSError, match="no audio"):
        tts.synthesize("x", tmp_path / "a", language="en")


def test_piper_timeout(fake_piper, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PIPER_SLEEP", "5")
    tts = PiperBackend({"en": fake_piper.en}, binary=fake_piper.binary, timeout_s=0.5)
    with pytest.raises(TTSError, match="timed out"):
        tts.synthesize("x", tmp_path / "a", language="en")


# ---- Kokoro (injected pipeline: the real model needs a download) -----------

def fake_pipeline(chunks, seen=None):
    def pipeline(text, voice, speed):
        if seen is not None:
            seen.append((text, voice, speed))
        for c in chunks:
            yield c
    return pipeline


def test_kokoro_joins_chunks_into_a_24khz_wav(tmp_path):
    chunks = [SimpleNamespace(audio=np.full(12000, 0.1, dtype=np.float32)),
              SimpleNamespace(audio=np.full(12000, 0.1, dtype=np.float32))]
    seen = []
    tts = KokoroBackend({"fr": "ff_siwis"}, default_language="fr",
                        pipeline_factory=lambda code: fake_pipeline(chunks, seen))
    out = tts.synthesize("Bonjour   le monde", tmp_path / "x", language="fr-FR")
    rate, seconds, _ = wav_info(out)
    assert (rate, seconds) == (24000, pytest.approx(1.0)) and out.suffix == ".wav"
    assert seen == [("Bonjour le monde", "ff_siwis", 1.0)]


def test_kokoro_accepts_old_tuple_results_and_torch_like_tensors(tmp_path):
    class Tensor:
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.full(2400, 0.2, dtype=np.float32)

    tts = KokoroBackend({"en": "af_heart"}, default_language="en",
                        pipeline_factory=lambda code: fake_pipeline([("gs", "ps", Tensor()), ("gs", "ps", None)]))
    out = tts.synthesize("hi", tmp_path / "x", language="en")
    assert wav_info(out)[1] == pytest.approx(0.1)


def test_kokoro_rejects_arabic_with_a_pointer_to_the_alternatives(tmp_path):
    tts = KokoroBackend({"ar": "x"}, pipeline_factory=lambda c: fake_pipeline([]))
    with pytest.raises(TTSError, match="does not support language 'ar'.*piper or edge"):
        tts.synthesize("مرحبا", tmp_path / "x", language="ar")


def test_kokoro_without_audio_is_an_error(tmp_path):
    tts = KokoroBackend({"en": "v"}, default_language="en", pipeline_factory=lambda c: fake_pipeline([]))
    with pytest.raises(TTSError, match="no audio"):
        tts.synthesize("hi", tmp_path / "x", language="en")


def test_kokoro_loads_through_the_model_manager_once_per_language(tmp_path):
    created = []
    chunks = [SimpleNamespace(audio=np.zeros(2400, dtype=np.float32))]
    manager = ModelManager(5000)
    tts = KokoroBackend({"en": "af_heart", "fr": "ff_siwis"}, default_language="en", manager=manager,
                        pipeline_factory=lambda code: created.append(code) or fake_pipeline(chunks))
    tts.synthesize("a", tmp_path / "1", language="en")
    tts.synthesize("b", tmp_path / "2", language="en")
    tts.synthesize("c", tmp_path / "3", language="fr")
    assert created == ["a", "f"] and manager.loaded() == ["kokoro:a", "kokoro:f"]


# ---- Edge (injected save function: the real one needs the network) ---------

def test_edge_writes_mp3_and_wraps_failures(tmp_path):
    saved = []

    def save(text, voice, out):
        saved.append((text, voice))
        out.write_bytes(b"ID3fake")

    tts = EdgeBackend({"ar-TN": "ar-TN-HediNeural"}, save_fn=save)
    out = tts.synthesize("مرحبا   بكم", tmp_path / "x", language="ar-TN")
    assert out.suffix == ".mp3" and saved == [("مرحبا بكم", "ar-TN-HediNeural")]

    def broken(text, voice, out):
        raise ConnectionError("no network")

    with pytest.raises(TTSError, match="edge-tts failed: ConnectionError"):
        EdgeBackend({"ar": "v"}, save_fn=broken).synthesize("x", tmp_path / "y", language="ar")
    with pytest.raises(TTSError, match="no audio"):
        EdgeBackend({"ar": "v"}, save_fn=lambda t, v, o: o.write_bytes(b"")).synthesize(
            "x", tmp_path / "z", language="ar")


# ---- routing ---------------------------------------------------------------

class Stub:
    def __init__(self, name, voices):
        self.name, self.voices, self.calls = name, voices, []

    def resolve(self, voice, language):
        return resolve_voice(self.voices, voice, language, "en")

    def synthesize(self, text, dst_stem, *, voice=None, language=None):
        self.calls.append((text, voice, language))
        out = dst_stem.with_suffix(".wav")
        out.write_bytes(b"RIFF" + b"0" * 60)
        return out


def router():
    a, b = Stub("a", {"en": "a-en", "fr": "a-fr"}), Stub("b", {"ar": "b-ar"})
    routes = {"en": Route("a"), "fr": Route("a"), "ar": Route("b")}
    return RoutedTTS({"a": a, "b": b}, routes, default_language="en"), a, b


def test_router_dispatches_by_language(tmp_path):
    tts, a, b = router()
    tts.synthesize("salut", tmp_path / "1", language="fr")
    tts.synthesize("مرحبا", tmp_path / "2", language="ar-TN")
    tts.synthesize("hello", tmp_path / "3")                      # no language: default
    assert [c[0] for c in a.calls] == ["salut", "hello"] and [c[0] for c in b.calls] == ["مرحبا"]
    assert tts.resolve(None, "ar-TN") == "b:b-ar" and tts.resolve(None, "fr") == "a:a-fr"


def test_router_refuses_unrouted_languages_and_unknown_backends():
    tts, _, _ = router()
    with pytest.raises(TTSError, match="no TTS route for language 'de'"):
        tts.resolve(None, "de")
    with pytest.raises(TTSError, match="unknown backend 'nope'"):
        RoutedTTS({}, {"en": Route("nope")})


def test_router_passes_an_explicit_voice_through(tmp_path):
    tts, a, _ = router()
    tts.synthesize("x", tmp_path / "1", voice="custom", language="en")
    assert a.calls[0][1] == "custom"


# ---- cache -----------------------------------------------------------------

def test_cache_synthesises_once_per_distinct_input(tmp_path):
    tts, a, b = router()
    first = synthesize_cached(tts, "Hello  world", tmp_path / "c", language="en")
    again = synthesize_cached(tts, "Hello world", tmp_path / "c", language="en")   # same after squashing
    assert first == again and len(a.calls) == 1
    synthesize_cached(tts, "Hello world", tmp_path / "c", language="fr")           # other voice
    synthesize_cached(tts, "Different", tmp_path / "c", language="en")             # other text
    synthesize_cached(tts, "مرحبا", tmp_path / "c", language="ar")                  # other backend
    assert len(a.calls) == 3 and len(b.calls) == 1
    assert not list((tmp_path / "c").glob(".tmp-*"))


def test_a_failed_synthesis_leaves_nothing_in_the_cache(tmp_path):
    class Failing(Stub):
        def synthesize(self, *a, **k):
            raise TTSError("nope")

    tts = RoutedTTS({"a": Failing("a", {"en": "v"})}, {"en": Route("a")}, "en")
    with pytest.raises(TTSError):
        synthesize_cached(tts, "x", tmp_path / "c", language="en")
    assert list((tmp_path / "c").iterdir()) == []


def test_cache_key_changes_with_the_configured_voice(tmp_path):
    one, _, _ = router()
    a2 = Stub("a", {"en": "a-en-NEW", "fr": "a-fr"})
    two = RoutedTTS({"a": a2, "b": Stub("b", {"ar": "b-ar"})},
                    {"en": Route("a"), "fr": Route("a"), "ar": Route("b")}, "en")
    synthesize_cached(one, "same text", tmp_path / "c", language="en")
    synthesize_cached(two, "same text", tmp_path / "c", language="en")
    assert len(a2.calls) == 1            # a new voice is not served from the old voice's cache


# ---- scheduling ------------------------------------------------------------

def L(text, start):
    return VoiceoverLine(text=text, start=start)


def sched(lines, durations, total=30.0, **kw):
    paths = [Path(f"/x/{i}.wav") for i in range(len(lines))]
    return schedule_lines(lines, paths, durations, total, **kw)


def test_lines_keep_their_start_times_when_there_is_room():
    out, notes = sched([L("a", 1.0), L("b", 10.0)], [3.0, 4.0])
    assert [(s.start, s.end) for s in out] == [(1.0, 4.0), (10.0, 14.0)] and notes == []


def test_an_overlapping_line_is_pushed_back_with_a_gap_and_a_note():
    out, notes = sched([L("a", 1.0), L("b", 2.0)], [3.0, 2.0], gap_s=0.25)
    assert out[1].start == pytest.approx(4.25) and "voiceover[1] moved from 2.0s to 4.2s" in notes[0]


def test_lines_are_scheduled_in_start_order_but_keep_their_index():
    out, _ = sched([L("late", 8.0), L("early", 1.0)], [2.0, 2.0])
    assert [(s.index, s.text) for s in out] == [(1, "early"), (0, "late")]


def test_a_line_that_runs_past_the_end_is_an_error_the_planner_can_act_on():
    with pytest.raises(VoiceoverError) as exc:
        sched([L("This sentence is too long for the remaining time", 8.0)], [5.0], total=10.0)
    msg = str(exc.value)
    assert "voiceover[0]" in msg and "This sentence is too long" in msg and "3.0s too long" in msg
    assert "Shorten the text" in msg


def test_a_tiny_overrun_within_tolerance_is_accepted():
    out, _ = sched([L("a", 8.0)], [2.2], total=10.0)
    assert out[0].end == pytest.approx(10.2)


def test_no_lines_no_schedule():
    assert sched([], []) == ([], [])


def test_prepare_voiceover_synthesises_caches_and_schedules(tmp_path):
    tts, a, _ = router()
    edl = EDL(platform="youtube", title="T", clips=[Clip(source="a.mp4", start=0, end=20)],
              voiceover=[L("first", 1.0), L("second", 6.0), L("first", 12.0)])
    scheduled, notes = prepare_voiceover(edl, tts, tmp_path / "c", language="en", duration_fn=lambda p: 2.0)
    assert [s.start for s in scheduled] == [1.0, 6.0, 12.0] and notes == []
    assert len(a.calls) == 2                  # "first" appeared twice but was synthesised once
    assert isinstance(scheduled[0], ScheduledLine) and scheduled[0].path.exists()


# ---- config ----------------------------------------------------------------

def test_shipped_tts_config_builds_without_network():
    tts = build_tts(base_dir="/models-root")
    assert tts.resolve(None, "fr") == "kokoro:ff_siwis"
    assert tts.resolve(None, "ar-TN") == "piper:" + str(Path("/models-root") / "models/piper/ar_JO-kareem-medium.onnx")
    assert tts.resolve(None, None) == "kokoro:af_heart"


def test_only_routed_backends_are_created():
    cfg = TTSConfig(languages={"en": {"backend": "piper", "voice": "/v/en.onnx"}}, default_language="en")
    tts = build_tts(cfg)
    assert sorted(tts.backends) == ["piper"]


def test_unknown_backend_and_unknown_keys_are_rejected():
    with pytest.raises(TTSError, match="unknown TTS backend"):
        build_tts(TTSConfig(languages={"en": {"backend": "espeak", "voice": "x"}}))
    with pytest.raises(Exception):
        TTSConfig(languages={}, defualt_language="en")


def test_edge_route_can_be_added_for_tunisian_arabic():
    cfg = TTSConfig(languages={"ar-TN": {"backend": "edge", "voice": "ar-TN-HediNeural"},
                               "en": {"backend": "piper", "voice": "/v/en.onnx"}})
    tts = build_tts(cfg)
    assert tts.resolve(None, "ar-TN") == "edge:ar-TN-HediNeural"
    assert load_tts_config().languages["fr"].backend == "kokoro"


def test_cli_reports_the_written_file(monkeypatch, capsys, tmp_path):
    import apps.voiceover as cli

    class Fake:
        def synthesize(self, text, stem, *, voice, language):
            out = stem.with_suffix(".wav")
            write_wav(out, np.zeros(2400), 24000)
            return out

        def resolve(self, voice, language):
            return "fake:voice"

    monkeypatch.setattr(cli, "build_tts", lambda *a, **k: Fake())
    assert cli.main(["hello", str(tmp_path / "out"), "--language", "en"]) == 0
    assert "out.wav (0.10s, voice fake:voice)" in capsys.readouterr().out
