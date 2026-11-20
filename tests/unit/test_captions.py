import pytest

from videoagent.schemas.edl import Caption, Overlay
from videoagent.tools.captions import (
    Event, Layout, TextStyle, ass_filter, ass_timestamp, build_ass, caption_events,
    escape_ass, escape_filter_value, hex_to_ass,
)
from videoagent.tools.overlay import (
    OVERLAY_STYLES, logo_filter, logo_overlays, logo_position, logo_width, text_overlay_events,
)

SHORTS = Layout(1080, 1920, safe_bottom_px=380)
YOUTUBE = Layout(1920, 1080)


@pytest.mark.parametrize("seconds,expected", [
    (0, "0:00:00.00"), (1.5, "0:00:01.50"), (59.999, "0:01:00.00"),
    (61.234, "0:01:01.23"), (3661.0, "1:01:01.00"),
])
def test_ass_timestamp(seconds, expected):
    assert ass_timestamp(seconds) == expected


def test_ass_timestamp_rejects_negative():
    with pytest.raises(ValueError):
        ass_timestamp(-1)


def test_hex_to_ass_is_bgr_order():
    assert hex_to_ass("#1F6FEB") == "&H00EB6F1F"
    assert hex_to_ass("#FFFFFF", alpha=128) == "&H80FFFFFF"


@pytest.mark.parametrize("bad", ["red", "#FFF", "#GGGGGG", ""])
def test_hex_to_ass_rejects_bad_color(bad):
    with pytest.raises(ValueError):
        hex_to_ass(bad)


def test_escape_ass_neutralises_override_tags_and_newlines():
    assert escape_ass("a{\\b1}b") == "a\\{\\\\b1\\}b"
    assert escape_ass("line1\nline2") == "line1\\Nline2"
    assert escape_ass("bad\x07char") == "badchar"


def test_escape_ass_keeps_arabic_untouched():
    text = "مرحبا بكم"
    assert escape_ass(text) == text


def test_build_ass_header_and_events():
    events = caption_events([Caption(start=1, end=2.5, text="Hello"), Caption(start=3, end=4, text="Bye")])
    ass = build_ass(events, SHORTS)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "Dialogue: 0,0:00:01.00,0:00:02.50,Caption,,0,0,0,,Hello" in ass
    assert ass.count("Dialogue:") == 2


def test_build_ass_sorts_events_by_start():
    events = [Event(5, 6, "late", "Caption"), Event(1, 2, "early", "Caption")]
    ass = build_ass(events, YOUTUBE)
    assert ass.index("early") < ass.index("late")


def test_build_ass_caption_margin_uses_safe_zone():
    assert "Style: Caption" in build_ass([], SHORTS)
    style = next(l for l in build_ass([], SHORTS).splitlines() if l.startswith("Style: Caption"))
    assert style.split(",")[21] == "380"  # MarginV
    style = next(l for l in build_ass([], YOUTUBE).splitlines() if l.startswith("Style: Caption"))
    assert style.split(",")[21] == str(round(1080 * 0.06))  # no safe zone: 6% of height


def test_build_ass_rejects_unknown_style_and_bad_range():
    with pytest.raises(ValueError, match="unknown style"):
        build_ass([Event(0, 1, "x", "Nope")], YOUTUBE)
    with pytest.raises(ValueError, match="end must be > start"):
        build_ass([Event(2, 1, "x", "Caption")], YOUTUBE)


def test_build_ass_box_style_uses_border_style_3():
    ass = build_ass([], YOUTUBE, extra_styles={"LowerThird": TextStyle(box=True)})
    line = next(l for l in ass.splitlines() if l.startswith("Style: LowerThird"))
    assert line.split(",")[15] == "3"


def test_empty_captions_are_skipped():
    assert caption_events([Caption(start=0, end=1, text="   ")]) == []


def test_escape_filter_value_escapes_special_chars():
    assert escape_filter_value("C:\\tmp\\a b.ass") == "C\\:/tmp/a\\ b.ass"
    assert ass_filter("/tmp/x.ass") == "ass=filename=/tmp/x.ass"
    assert "fontsdir=/fonts" in ass_filter("/tmp/x.ass", "/fonts")


# ---- overlays --------------------------------------------------------------

def test_text_overlay_events_alignment_and_margins():
    ovs = [
        Overlay(kind="title", text="Hi", start=0, end=2, position="top"),
        Overlay(kind="cta", text="Buy", start=2, end=4, position="bottom"),
        Overlay(kind="lower_third", text="Name", start=1, end=3, position="top_left"),
    ]
    events = text_overlay_events(ovs, SHORTS)
    by_style = {e.style: e for e in events}
    assert by_style["Title"].alignment == 8
    assert by_style["Cta"].alignment == 2
    assert by_style["LowerThird"].alignment == 7
    # bottom overlays sit above the captions; top overlays use the small top margin
    assert by_style["Cta"].margin_v > SHORTS.caption_margin_v
    assert by_style["Title"].margin_v == SHORTS.top_margin_v


def test_text_overlay_events_skip_logos_and_blank_text():
    ovs = [
        Overlay(kind="logo", asset="l.png", start=0, end=1),
        Overlay(kind="title", text="  ", start=0, end=1),
    ]
    assert text_overlay_events(ovs, YOUTUBE) == []


def test_overlay_styles_register_in_ass():
    ass = build_ass([], YOUTUBE, extra_styles=OVERLAY_STYLES)
    for name in OVERLAY_STYLES:
        assert f"Style: {name}," in ass


def test_logo_helpers():
    ovs = [Overlay(kind="logo", asset="l.png", start=0, end=1),
           Overlay(kind="logo", start=0, end=1)]  # no asset: ignored
    assert len(logo_overlays(ovs)) == 1
    assert logo_width(SHORTS) % 2 == 0
    assert logo_position("top_right", SHORTS)[0].startswith("W-w")
    assert logo_position("bottom", SHORTS)[1].startswith("H-h")


def test_logo_filter_has_enable_window():
    ov = Overlay(kind="logo", asset="l.png", start=1, end=3, position="top_left")
    f = logo_filter("vin", "2:v", "vout", ov, SHORTS)
    assert "between(t,1.000,3.000)" in f
    assert f.startswith("[2:v]scale=") and "[vin]" in f and f.endswith("[vout]")
