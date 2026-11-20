"""Overlays: text (title, lower_third, cta) become ASS events; logos become an `overlay` filter."""

from __future__ import annotations

from collections.abc import Sequence

from ..schemas.edl import Overlay
from .captions import ALIGN, Event, Layout, TextStyle

# Styles registered in the .ass file for each text overlay kind.
OVERLAY_STYLES: dict[str, TextStyle] = {
    "Title": TextStyle(size_ratio=0.07, bold=True),
    "LowerThird": TextStyle(size_ratio=0.04, box=True, box_color="#0D1117", box_alpha=64),
    "Cta": TextStyle(size_ratio=0.045, box=True, box_color="#1F6FEB", box_alpha=32),
}
_STYLE_FOR_KIND = {"title": "Title", "lower_third": "LowerThird", "cta": "Cta"}

LOGO_WIDTH_RATIO = 0.15
LOGO_MARGIN_RATIO = 0.04


def text_overlay_events(overlays: Sequence[Overlay], layout: Layout) -> list[Event]:
    """Convert title / lower_third / cta overlays into ASS events (logos are skipped)."""
    events: list[Event] = []
    for ov in overlays:
        style = _STYLE_FOR_KIND.get(ov.kind)
        if style is None or not (ov.text or "").strip():
            continue
        top = ov.position in ("top", "top_left", "top_right")
        margin = layout.top_margin_v if top else layout.overlay_bottom_margin_v
        if ov.position == "center":
            margin = 0
        events.append(
            Event(ov.start, ov.end, ov.text or "", style, alignment=ALIGN[ov.position], margin_v=margin)
        )
    return events


def logo_overlays(overlays: Sequence[Overlay]) -> list[Overlay]:
    return [o for o in overlays if o.kind == "logo" and o.asset]


def logo_width(layout: Layout) -> int:
    """Logo width in pixels (even, as required by yuv420p scaling)."""
    return max(2, round(layout.width * LOGO_WIDTH_RATIO) // 2 * 2)


def logo_position(position: str, layout: Layout) -> tuple[str, str]:
    """x/y expressions for the `overlay` filter (W,H = main frame; w,h = logo)."""
    mx = f"W*{LOGO_MARGIN_RATIO}"
    my = f"H*{LOGO_MARGIN_RATIO}"
    x = {
        "top_left": mx,
        "top_right": f"W-w-{mx}",
    }.get(position, "(W-w)/2")
    y = {
        "top": my, "top_left": my, "top_right": my,
        "bottom": f"H-h-{my}", "center": "(H-h)/2",
    }[position]
    return x, y


def logo_filter(
    in_label: str, logo_label: str, out_label: str, ov: Overlay, layout: Layout
) -> str:
    """Filter fragment: scale the logo input and overlay it on `in_label` between ov.start and ov.end."""
    x, y = logo_position(ov.position, layout)
    return (
        f"[{logo_label}]scale={logo_width(layout)}:-2,format=rgba[{out_label}_l];"
        f"[{in_label}][{out_label}_l]overlay=x={x}:y={y}:"
        f"enable='between(t,{ov.start:.3f},{ov.end:.3f})'[{out_label}]"
    )
