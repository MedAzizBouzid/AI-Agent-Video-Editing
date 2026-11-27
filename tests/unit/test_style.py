import json

from videoagent.edl.style import apply_house_style, merge_contiguous, only_edge_fades, place_cta
from videoagent.schemas import EDL, BriefSpec, Clip
from videoagent.schemas.edl import Overlay

from .agent_helpers import Harness, copy_json, make_doc, replies, report


def clip(s, e, source="a.mp4", speed=1.0, fi=0.0, fo=0.0):
    return Clip(source=source, start=s, end=e, speed=speed, fade_in_s=fi, fade_out_s=fo)


def brief(**kw):
    return BriefSpec(inputs=["a.mp4"], goal="g", platforms=["shorts"], **kw)


def test_back_to_back_clips_become_one_and_keep_the_outer_fades():
    out = merge_contiguous([clip(22.3, 30.2, fi=0.15, fo=0.15), clip(30.2, 36.2, fi=0.15, fo=0.4)])
    assert len(out) == 1 and (out[0].start, out[0].end) == (22.3, 36.2)
    assert (out[0].fade_in_s, out[0].fade_out_s) == (0.15, 0.4)


def test_clips_that_do_not_continue_each_other_are_left_alone():
    base = [clip(0, 5)]
    for other in (clip(5.5, 9),                       # a real gap
                  clip(4, 9),                         # overlaps: would repeat footage
                  clip(5, 9, speed=1.2),              # different speed
                  clip(5, 9, source="b.mp4")):        # different file
        assert len(merge_contiguous(base + [other])) == 2


def test_a_tiny_gap_is_still_a_continuation_and_chains_collapse():
    out = merge_contiguous([clip(0, 5), clip(5.03, 8), clip(8, 10)])
    assert len(out) == 1 and (out[0].start, out[0].end) == (0, 10)


def test_only_the_first_fade_in_and_last_fade_out_survive():
    out = only_edge_fades([clip(0, 3, fi=0.2, fo=0.2), clip(10, 13, fi=0.2, fo=0.2), clip(20, 23, fi=0.2, fo=0.2)])
    assert [(c.fade_in_s, c.fade_out_s) for c in out] == [(0.2, 0), (0, 0), (0, 0.2)]


def test_the_cta_comes_from_the_brief_and_sits_in_the_last_three_seconds():
    invented = [Overlay(kind="cta", text="invented", start=20, end=23), Overlay(kind="title", text="T", start=0, end=2)]
    out = place_cta(invented, "Commandez maintenant", 25.6)
    assert [(o.kind, o.text) for o in out] == [("title", "T"), ("cta", "Commandez maintenant")]
    cta = out[1]
    assert round(cta.start, 1) == 22.6 and cta.end == 25.6


def test_without_a_brief_cta_the_models_invented_one_is_removed():
    out = place_cta([Overlay(kind="cta", text="invented", start=1, end=3)], None, 10)
    assert out == []
    assert place_cta([], "  ", 10) == []


def test_a_video_shorter_than_the_cta_window_shows_it_from_the_start():
    [o] = place_cta([], "Go", 2.0)
    assert (o.start, o.end) == (0.0, 2.0)


def test_apply_house_style_reports_what_it_changed_and_is_idempotent():
    edl = EDL(platform="shorts", title="T", clips=[clip(0, 5, fi=.2, fo=.2), clip(5, 9, fi=.2, fo=.2)],
              overlays=[Overlay(kind="cta", text="x", start=1, end=3)])
    out, notes = apply_house_style(edl, brief())
    assert notes == ["merged 1 back-to-back clip(s)", "call to action removed: the brief has none"]
    again, notes2 = apply_house_style(out, brief())
    assert again == out and notes2 == []


def test_the_real_model_output_that_motivated_this_is_fixed_end_to_end(tmp_path):
    """Replay of a real planner reply: fades on every clip, two touching clips, an invented CTA."""
    fade = {"speed": 1.0, "fade_in_s": 0.15, "fade_out_s": 0.15}
    reply = json.dumps({
        "platform": "shorts", "title": "T",
        "clips": [dict(source="raw.mp4", start=0.0, end=6.6, **fade), dict(source="raw.mp4", start=22.3, end=30.2, **fade),
                  dict(source="raw.mp4", start=30.2, end=36.2, **fade), dict(source="raw.mp4", start=51.3, end=56.4, **fade)],
        "overlays": [{"kind": "cta", "text": "اطلب المنتج الآن", "start": 20.0, "end": 23.0, "position": "bottom"}]})
    doc = make_doc(duration=60.0, speech=[(0, 6.6, "hook sentence here"), (22.3, 30.2, "middle part one"),
                                          (30.2, 36.2, "middle part two"), (51.3, 56.4, "closing words")])
    t = Harness(tmp_path, replies(reply, copy_json()), [report()], doc=doc)
    [res] = t.run()
    clips = res.edl.clips
    assert [(c.start, c.end) for c in clips] == [(0.0, 6.6), (22.3, 36.2), (51.3, 56.4)]
    assert [(c.fade_in_s, c.fade_out_s) for c in clips] == [(0.15, 0), (0, 0), (0, 0.15)]
    assert res.edl.overlays == []                                   # the brief has no CTA
    assert abs(res.edl.total_duration - 25.6) < 1e-6                # merging changed no output time
    assert [c.text for c in res.edl.captions] == [
        "hook sentence here", "middle part one", "middle part two", "closing words"]
    assert "merged 1 back-to-back clip(s)" in " ".join(res.events)


def test_the_briefs_cta_is_placed_by_the_pipeline(tmp_path):
    t = Harness(tmp_path, replies('{"platform":"shorts","title":"T","clips":[{"source":"raw.mp4","start":2,"end":14}]}',
                                  copy_json()), [report()])
    [res] = t.run(brief=t.brief(cta="Commandez maintenant"))
    [cta] = res.edl.overlays
    assert cta.text == "Commandez maintenant" and cta.end == 12.0 and cta.start == 9.0
