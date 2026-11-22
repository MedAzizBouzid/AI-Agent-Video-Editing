"""Console setup shared by the command-line tools."""

from __future__ import annotations

import sys


def use_utf8_console() -> None:
    """Make print() safe for Arabic and accented text.

    On Windows the console encoding is often cp1252, which cannot represent Arabic: printing a
    transcript or a title would raise UnicodeEncodeError. Reconfigure the streams to UTF-8 (and
    never raise on a character the terminal font lacks: it is replaced instead).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):  # a stream that cannot be reconfigured: keep it as is
                pass
