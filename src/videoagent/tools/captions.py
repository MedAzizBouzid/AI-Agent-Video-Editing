"""Caption rendering via ASS subtitles (libass).

Why ASS and not drawtext: libass runs HarfBuzz + FriBidi, so Arabic letters are joined correctly
and right-to-left text is laid out correctly. drawtext does neither reliably. libass also wraps
long lines for us (WrapStyle 0), so we do no manual wrapping.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..schemas.edl import Caption
from ._ffmpeg import PathLike, ensure_parent

# numpad alignment codes used by ASS (\an)
ALIGN = {"top": 8, "center": 5, "bottom": 2, "top_left": 7, "top_right": 9}

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def hex_to_ass(color: str, alpha: int = 0) -> str:
    """'#RRGGBB' -> ASS '&HAABBGGRR'. alpha: 0 = opaque, 255 = transparent."""
    value = color.lstrip("#")
    if len(value) != 6 or not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        raise ValueError(f"expected #RRGGBB, got {color!r}")
    r, g, b = value[0:2], value[2:4], value[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def ass_timestamp(seconds: float) -> str:
    """Seconds -> H:MM:SS.cc (centiseconds). Rounds first so 59.999 carries into the minute."""
    if seconds < 0:
        raise ValueError("timestamp must be >= 0")
    total_cs = round(seconds * 100)
    cs = total_cs % 100
    total_s = total_cs // 100
    return f"{total_s // 3600}:{(total_s // 60) % 60:02d}:{total_s % 60:02d}.{cs:02d}"


def escape_ass(text: str) -> str:
    """Make arbitrary text safe inside an ASS Dialogue line.

    `{...}` would be parsed as override tags, and newlines must become \\N.
    """
    text = _CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return text.replace("\n", "\\N")


@dataclass(frozen=True)
class TextStyle:
    """Visual style, with sizes given relative to the frame height so it works at any resolution."""

    font: str = "Arial"
    size_ratio: float = 0.045  # font size / frame height
    color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline_ratio: float = 0.0035
    bold: bool = True
    box: bool = False  # True: opaque box behind the text instead of an outline
    box_color: str = "#000000"
    box_alpha: int = 96  # 0 opaque .. 255 transparent


@dataclass(frozen=True)
class Layout:
    """Frame size and the vertical room reserved for platform UI at the bottom."""

    width: int
    height: int
    safe_bottom_px: int = 0

    @property
    def caption_margin_v(self) -> int:
        """Bottom margin for captions: the platform safe zone, or 6% of the height if larger."""
        return max(self.safe_bottom_px, round(self.height * 0.06))

    @property
    def overlay_bottom_margin_v(self) -> int:
        """Bottom-anchored overlays sit above the captions."""
        return self.caption_margin_v + round(self.height * 0.09)

    @property
    def top_margin_v(self) -> int:
        return round(self.height * 0.06)


def _style_line(name: str, style: TextStyle, layout: Layout, alignment: int, margin_v: int) -> str:
    size = max(8, round(layout.height * style.size_ratio))
    outline = max(1, round(layout.height * style.outline_ratio))
    border_style = 3 if style.box else 1
    outline_value = round(size * 0.25) if style.box else outline
    back = hex_to_ass(style.box_color, style.box_alpha)
    side = round(layout.width * 0.06)
    fields = [
        name, style.font, str(size),
        hex_to_ass(style.color), hex_to_ass(style.color),
        hex_to_ass(style.outline_color), back,
        "-1" if style.bold else "0", "0", "0", "0",
        "100", "100", "0", "0",
        str(border_style), str(outline_value), "0",
        str(alignment), str(side), str(side), str(margin_v), "1",
    ]
    return "Style: " + ",".join(fields)


@dataclass(frozen=True)
class Event:
    """One timed piece of text. `style` must be a name registered with `build_ass`."""

    start: float
    end: float
    text: str
    style: str
    alignment: int | None = None  # override the style's alignment (\an)
    margin_v: int | None = None


def caption_events(captions: Iterable[Caption]) -> list[Event]:
    return [Event(c.start, c.end, c.text, "Caption") for c in captions if c.text.strip()]


def build_ass(
    events: Iterable[Event],
    layout: Layout,
    caption_style: TextStyle | None = None,
    extra_styles: dict[str, TextStyle] | None = None,
) -> str:
    """Build the full text of an .ass file.

    Always registers the style "Caption". `extra_styles` adds more (title, lower_third, ...).
    Each style is bottom-center aligned by default; an Event may override alignment and margin.
    """
    styles = {"Caption": caption_style or TextStyle()}
    styles.update(extra_styles or {})

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {layout.width}",
        f"PlayResY: {layout.height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    ]
    for name, style in styles.items():
        lines.append(_style_line(name, style, layout, 2, layout.caption_margin_v))
    lines += [
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for ev in sorted(events, key=lambda e: (e.start, e.end)):
        if ev.style not in styles:
            raise ValueError(f"event uses unknown style '{ev.style}'")
        if ev.end <= ev.start:
            raise ValueError(f"event end must be > start: {ev}")
        prefix = f"{{\\an{ev.alignment}}}" if ev.alignment else ""
        margin = ev.margin_v if ev.margin_v is not None else 0  # 0 = use the style's margin
        lines.append(
            f"Dialogue: 0,{ass_timestamp(ev.start)},{ass_timestamp(ev.end)},{ev.style},,"
            f"0,0,{margin},,{prefix}{escape_ass(ev.text)}"
        )
    return "\n".join(lines) + "\n"


def write_ass(path: PathLike, content: str) -> Path:
    out = ensure_parent(path)
    out.write_text(content, encoding="utf-8")
    return out


def ass_filter(ass_path: PathLike, fonts_dir: PathLike | None = None) -> str:
    """The FFmpeg filter fragment that burns an .ass file. Paths are escaped for filtergraphs."""
    parts = [f"filename={escape_filter_value(ass_path)}"]
    if fonts_dir:
        parts.append(f"fontsdir={escape_filter_value(fonts_dir)}")
    return "ass=" + ":".join(parts)


def escape_filter_value(value: PathLike) -> str:
    """Escape a path for use as a filter option value (handles ':' ',' quotes and backslashes)."""
    text = str(value).replace("\\", "/")
    for ch in ("\\", "'", ":", ",", "[", "]", ";", " "):
        text = text.replace(ch, "\\" + ch)
    return text
