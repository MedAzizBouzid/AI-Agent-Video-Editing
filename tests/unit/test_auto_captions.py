from videoagent.edl.auto_captions import captions_from_transcript, split_text
from videoagent.edl.validator import validate_edl
from videoagent.schemas import EDL, Clip, PerceptionDoc, SourceInfo, TranscriptSegment


def doc(*segs, path="a.mp4", duration=100.0):
    return PerceptionDoc(
        source=SourceInfo(path=path, sha256="x", duration_s=duration, width=1920, height=1080, fps=30),
        transcript=[TranscriptSegment(start=s, end=e, text=t) for s, e, t in segs],
    )


def edl(*clips):
    return EDL(platform="youtube", title="T", clips=[Clip(source="a.mp4", start=s, end=e, **kw)
                                                      for s, e, kw in clips])


def test_times_are_mapped_through_cuts_to_the_output_timeline():
    d = doc((10, 14, "first sentence here"), (50, 54, "second sentence there"))
    e = edl((10, 14, {}), (50, 54, {}))                     # source 10-14 -> out 0-4, 50-54 -> out 4-8
    caps = captions_from_transcript(e, {"a.mp4": d})
    assert [(c.start, c.end, c.text) for c in caps] == [
        (0.0, 4.0, "first sentence here"), (4.05, 8.0, "second sentence there")]


def test_speed_changes_compress_the_caption_times():
    d = doc((0, 8, "hello there my friend"))
    caps = captions_from_transcript(edl((0, 8, {"speed": 2.0})), {"a.mp4": d})
    assert (caps[0].start, caps[0].end) == (0.0, 4.0)


def test_speech_outside_the_kept_clips_is_not_captioned():
    d = doc((0, 5, "cut out"), (20, 25, "kept words here"))
    caps = captions_from_transcript(edl((20, 25, {})), {"a.mp4": d})
    assert [c.text for c in caps] == ["kept words here"]


def test_a_cut_through_a_segment_keeps_only_the_part_inside():
    d = doc((0, 10, "one two three four five six seven eight nine ten"))
    caps = captions_from_transcript(edl((5, 10, {})), {"a.mp4": d})
    assert caps[0].text == "six seven eight nine ten"
    assert (caps[0].start, caps[0].end) == (0.0, 5.0)


def test_long_segments_are_split_into_readable_chunks_that_cover_the_time():
    text = "this is a rather long sentence that must be broken into several shorter caption lines"
    caps = captions_from_transcript(edl((0, 12, {})), {"a.mp4": doc((0, 12, text))})
    assert len(caps) >= 2 and all(len(c.text) <= 42 for c in caps)
    assert " ".join(c.text for c in caps) == text
    assert caps[0].start == 0.0 and abs(caps[-1].end - 12.0) < 0.1


def test_arabic_words_are_never_split_inside_a_word():
    text = "مرحبا بكم في هذا الفيديو الجديد حول منتجاتنا الرائعة والمميزة جدا"
    chunks = split_text(text, 20)
    assert " ".join(chunks) == text and all(len(c) <= 20 for c in chunks)
    assert split_text("x" * 60, 20) == ["x" * 60]          # one huge word is kept whole


def test_output_passes_the_validator_even_for_messy_transcripts():
    d = doc((0, 0.4, "tiny"), (0.5, 3, "overlapping start"), (2.9, 6, "and next one"), (6, 6.2, "x"))
    e = edl((0, 6, {}))
    caps = captions_from_transcript(e, {"a.mp4": d})
    result = validate_edl(e.model_copy(update={"captions": caps}), {"a.mp4": 100.0})
    assert result.ok, result.format_for_llm()
    assert all(b.start >= a.end for a, b in zip(caps, caps[1:]))


def test_missing_doc_or_no_speech_gives_no_captions():
    assert captions_from_transcript(edl((0, 5, {})), {}) == []
    assert captions_from_transcript(edl((0, 5, {})), {"a.mp4": doc()}) == []


def test_clips_from_two_sources_use_the_right_transcript():
    a, b = doc((0, 4, "from a"), path="a.mp4"), doc((0, 4, "from b"), path="b.mp4")
    e = EDL(platform="youtube", title="T", clips=[Clip(source="a.mp4", start=0, end=4),
                                                  Clip(source="b.mp4", start=0, end=4)])
    caps = captions_from_transcript(e, {"a.mp4": a, "b.mp4": b})
    assert [(c.text, c.start) for c in caps] == [("from a", 0.0), ("from b", 4.05)]
