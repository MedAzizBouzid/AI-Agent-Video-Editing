import shutil
from types import SimpleNamespace

import pytest

from videoagent.perception import PerceptionConfig, build_perception_doc, compact_for_llm, file_fingerprint
from videoagent.perception.frames import pick_timestamps, sample_keyframes
from videoagent.perception.scenes import cuts_to_shots, detect_shots, parse_showinfo
from videoagent.perception.silence import detect_silences, parse_silencedetect
from videoagent.perception.transcribe import confidence_from_logprob, transcribe
from videoagent.schemas.perception import (
    PerceptionDoc, Shot, Silence, SourceInfo, TranscriptSegment,
)

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


# ---- silence ---------------------------------------------------------------

def test_parse_silencedetect_closed_open_and_clamped():
    log = (
        "[silencedetect @ 0x1] silence_start: -0.0012\n"
        "[silencedetect @ 0x1] silence_end: 2.5 | silence_duration: 2.5\n"
        "[silencedetect @ 0x1] silence_start: 8\n"          # never closed: runs to the end
    )
    spans = parse_silencedetect(log, duration=10.0)
    assert [(s.start, s.end) for s in spans] == [(0.0, 2.5), (8.0, 10.0)]


def test_parse_silencedetect_empty():
    assert parse_silencedetect("nothing here", 5.0) == []


def test_detect_silences_finds_the_gap(scene_clip):
    spans = detect_silences(scene_clip)
    assert len(spans) == 1
    assert spans[0].start == pytest.approx(2.0, abs=0.15)
    assert spans[0].end == pytest.approx(4.0, abs=0.15)


def test_detect_silences_no_audio_returns_empty(silent_clip):
    assert detect_silences(silent_clip) == []


def test_continuous_tone_has_no_silence(sample_clip):
    assert detect_silences(sample_clip) == []


# ---- scenes ----------------------------------------------------------------

def test_parse_showinfo_only_reads_showinfo_lines():
    log = (
        "[Parsed_showinfo_3 @ 0x1] n:   0 pts:  60 pts_time:2.0 pos: 1\n"
        "[Parsed_showinfo_3 @ 0x1] n:   1 pts: 120 pts_time:4.0 pos: 2\n"
        "[other @ 0x1] pts_time:99.0\n"
    )
    assert parse_showinfo(log) == [2.0, 4.0]


def test_cuts_to_shots_are_contiguous_and_cover_everything():
    shots = cuts_to_shots([3.0, 7.0], duration=10.0)
    assert [(s.start, s.end) for s in shots] == [(0.0, 3.0), (3.0, 7.0), (7.0, 10.0)]
    assert [s.index for s in shots] == [0, 1, 2]


def test_cuts_to_shots_drops_tiny_shots():
    # 0.4 s after the first cut, and 0.5 s before the end: both would create shots < 1 s
    shots = cuts_to_shots([3.0, 3.4, 9.5], duration=10.0, min_shot_s=1.0)
    assert [(s.start, s.end) for s in shots] == [(0.0, 3.0), (3.0, 10.0)]


def test_cuts_to_shots_without_cuts_is_one_shot():
    assert len(cuts_to_shots([], 8.0)) == 1


def test_detect_shots_finds_color_changes(scene_clip):
    shots = detect_shots(scene_clip)
    assert len(shots) == 3
    assert shots[1].start == pytest.approx(2.0, abs=0.3)
    assert shots[2].start == pytest.approx(4.0, abs=0.3)
    assert shots[-1].end == pytest.approx(6.0, abs=0.1)


def test_detect_shots_without_cuts(sample_clip):
    assert len(detect_shots(sample_clip)) == 1


# ---- keyframes -------------------------------------------------------------

def test_pick_timestamps_variants():
    assert pick_timestamps([], 10.0, 2) == [2.5, 7.5]                       # no shots: even spread
    few = [Shot(index=0, start=0, end=4), Shot(index=1, start=4, end=10)]
    assert pick_timestamps(few, 10.0, 5) == [2.0, 7.0]                      # shot midpoints
    many = [Shot(index=i, start=i, end=i + 1) for i in range(100)]
    picked = pick_timestamps(many, 100.0, 10)
    assert len(picked) == 10 and picked == sorted(picked) and 0 < picked[0] and picked[-1] < 100
    assert pick_timestamps(few, 10.0, 0) == []


def test_sample_keyframes_writes_jpegs(scene_clip, tmp_path):
    shots = detect_shots(scene_clip)
    paths = sample_keyframes(scene_clip, shots, 6.0, tmp_path / "frames", max_frames=3, width=160)
    assert len(paths) == 3
    assert all(p.endswith(".jpg") for p in paths)
    assert all((tmp_path / "frames").joinpath(p.split("/")[-1]).stat().st_size > 100 for p in paths)


# ---- transcription (fake model: real Whisper needs a model download) -------

def seg(start, end, text, lp=-0.2, nsp=0.0):
    return SimpleNamespace(start=start, end=end, text=text, avg_logprob=lp, no_speech_prob=nsp)


class FakeWhisper:
    def __init__(self, segments, language="ar"):
        self.segments, self.language, self.calls = segments, language, []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return iter(self.segments), SimpleNamespace(language=self.language)


def test_transcribe_converts_and_filters():
    fake = FakeWhisper([
        seg(0, 2, "  مرحبا  ", lp=-0.1),
        seg(2, 3, "   "),                       # empty text
        seg(3, 4, "ghost", nsp=0.99),           # Whisper thinks it is not speech
        seg(5, 5, "zero length"),               # end == start
        seg(6, 8, "hello", lp=-1.0),
    ])
    out, lang = transcribe("x.mp4", fake, language="ar")
    assert lang == "ar"
    assert [s.text for s in out] == ["مرحبا", "hello"]
    assert out[0].confidence == pytest.approx(0.905, abs=0.01)
    assert out[1].confidence == pytest.approx(0.368, abs=0.01)
    assert all(isinstance(s, TranscriptSegment) for s in out)


def test_transcribe_passes_safe_decoding_options():
    fake = FakeWhisper([])
    transcribe("x.mp4", fake, language="fr")
    _, kw = fake.calls[0]
    assert kw["language"] == "fr" and kw["vad_filter"] is True
    assert kw["condition_on_previous_text"] is False  # prevents Whisper repeating itself


def test_confidence_bounds():
    assert confidence_from_logprob(None) is None
    assert confidence_from_logprob(0.5) == 1.0
    assert 0 <= confidence_from_logprob(-50) < 1e-6


# ---- fingerprint -----------------------------------------------------------

def test_fingerprint_stable_and_sensitive(tmp_path):
    a, b, c = tmp_path / "a.bin", tmp_path / "b.bin", tmp_path / "c.bin"
    a.write_bytes(b"x" * 1000)
    b.write_bytes(b"x" * 1000)
    c.write_bytes(b"x" * 999 + b"y")
    assert file_fingerprint(a) == file_fingerprint(b)
    assert file_fingerprint(a) != file_fingerprint(c)


def test_fingerprint_changes_with_size_and_middle(tmp_path):
    big = tmp_path / "big.bin"
    big.write_bytes(b"\0" * (10 * 1024 * 1024))
    before = file_fingerprint(big)
    data = bytearray(big.read_bytes())
    data[5 * 1024 * 1024] = 1  # middle byte, inside a sampled region
    big.write_bytes(bytes(data))
    assert file_fingerprint(big) != before
    big.write_bytes(bytes(data) + b"\0")  # same samples, different size
    assert file_fingerprint(big) != before


# ---- build_perception_doc --------------------------------------------------

class Counter:
    def __init__(self, language="ar"):
        self.calls, self.language = 0, language

    def __call__(self, path):
        self.calls += 1
        return [TranscriptSegment(start=0.1, end=1.9, text="hello", confidence=0.9)], self.language


def test_build_doc_merges_everything(scene_clip, tmp_path):
    fn = Counter()
    doc = build_perception_doc(scene_clip, cache_dir=tmp_path, transcribe_fn=fn, source_name="scene.mp4")
    assert doc.source.path == "scene.mp4"
    assert (doc.source.width, doc.source.height) == (640, 360)
    assert doc.source.duration_s == pytest.approx(6, abs=0.2)
    assert len(doc.shots) == 3 and len(doc.silences) == 1
    assert doc.transcript[0].text == "hello" and doc.language == "ar"
    assert 1 <= len(doc.keyframes) <= 12
    assert fn.calls == 1


def test_second_run_is_served_from_cache(scene_clip, tmp_path):
    fn = Counter()
    first = build_perception_doc(scene_clip, cache_dir=tmp_path, transcribe_fn=fn)
    second = build_perception_doc(scene_clip, cache_dir=tmp_path, transcribe_fn=fn)
    assert fn.calls == 1
    assert second.model_dump() == first.model_dump()


def test_cache_hit_still_honours_source_name(scene_clip, tmp_path):
    fn = Counter()
    build_perception_doc(scene_clip, cache_dir=tmp_path, transcribe_fn=fn, source_name="a.mp4")
    renamed = build_perception_doc(scene_clip, cache_dir=tmp_path, transcribe_fn=fn, source_name="b.mp4")
    assert renamed.source.path == "b.mp4" and fn.calls == 1


def test_changed_setting_misses_cache(scene_clip, tmp_path):
    fn = Counter()
    build_perception_doc(scene_clip, PerceptionConfig(language="ar"), tmp_path, transcribe_fn=fn)
    build_perception_doc(scene_clip, PerceptionConfig(language="fr"), tmp_path, transcribe_fn=fn)
    assert fn.calls == 2


def test_use_cache_false_recomputes(scene_clip, tmp_path):
    fn = Counter()
    build_perception_doc(scene_clip, cache_dir=tmp_path, transcribe_fn=fn)
    build_perception_doc(scene_clip, cache_dir=tmp_path, transcribe_fn=fn, use_cache=False)
    assert fn.calls == 2


def test_corrupt_cache_is_rebuilt(scene_clip, tmp_path):
    fn = Counter()
    build_perception_doc(scene_clip, cache_dir=tmp_path, transcribe_fn=fn)
    for f in tmp_path.glob("*.json"):
        f.write_text("{not json")
    build_perception_doc(scene_clip, cache_dir=tmp_path, transcribe_fn=fn)
    assert fn.calls == 2


def test_missing_keyframe_invalidates_cache(scene_clip, tmp_path):
    fn = Counter()
    doc = build_perception_doc(scene_clip, cache_dir=tmp_path, transcribe_fn=fn)
    from pathlib import Path
    Path(doc.keyframes[0]).unlink()
    build_perception_doc(scene_clip, cache_dir=tmp_path, transcribe_fn=fn)
    assert fn.calls == 2


def test_video_without_audio_skips_transcription(silent_clip, tmp_path):
    fn = Counter()
    doc = build_perception_doc(silent_clip, cache_dir=tmp_path, transcribe_fn=fn)
    assert fn.calls == 0
    assert doc.transcript == [] and doc.silences == [] and not doc.source.has_audio


# ---- compact view ----------------------------------------------------------

def make_doc(n_segments=3) -> PerceptionDoc:
    return PerceptionDoc(
        source=SourceInfo(path="raw.mp4", sha256="x", duration_s=100, width=1280, height=720, fps=30),
        transcript=[TranscriptSegment(start=i * 10, end=i * 10 + 5, text=f"sentence {i}")
                    for i in range(n_segments)],
        shots=[Shot(index=0, start=0, end=60), Shot(index=1, start=60, end=100)],
        silences=[Silence(start=5, end=5.4), Silence(start=20, end=23)],
        language="ar",
    )


def test_compact_contains_the_facts():
    text = compact_for_llm(make_doc())
    assert "raw.mp4 | 1280x720 | 30fps | duration 100.0s | language ar" in text
    assert "SHOTS: 0.0-60.0 60.0-100.0" in text
    assert "SILENCES >=1s: 20.0-23.0" in text          # the 0.4 s pause is filtered out
    assert "[10.0-15.0] sentence 1" in text


def test_compact_window_selects_time_range():
    text = compact_for_llm(make_doc(10), start=30, end=55)
    assert "sentence 3" in text and "sentence 5" in text
    assert "sentence 0" not in text and "sentence 7" not in text
    assert "SHOTS: 0.0-60.0" in text and "60.0-100.0" not in text


def test_compact_truncates_and_says_so():
    text = compact_for_llm(make_doc(500), max_chars=600, end=10_000)  # window covers all 500 segments
    assert len(text) <= 700
    assert "more segments not shown" in text.splitlines()[-1]


def test_compact_without_speech():
    doc = make_doc(0)
    assert "(no speech detected)" in compact_for_llm(doc)


def test_compact_is_much_smaller_than_json():
    doc = make_doc(200)
    assert len(compact_for_llm(doc, max_chars=10**6)) < len(doc.model_dump_json()) * 0.6


# ---- cli -------------------------------------------------------------------

def test_cli_prints_compact_view(monkeypatch, capsys, tmp_path):
    import apps.perceive as cli

    monkeypatch.setattr(cli, "build_perception_doc", lambda *a, **k: make_doc())
    assert cli.main([str(tmp_path / "raw.mp4")]) == 0
    assert "SOURCE raw.mp4" in capsys.readouterr().out
