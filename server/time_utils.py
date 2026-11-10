"""Normalize ISO-8601 timestamps emitted by external market-data feeds.

Alpaca returns nanosecond precision (nine fractional digits).  Python 3.10's
``datetime.fromisoformat`` accepts only zero, three, or six fractional digits,
so the autonomous runner crashed with ``Invalid isoformat string`` on every
cycle.  Normalizing the fraction to exactly six digits before parsing keeps the
parser compatible with every supported runtime and every feed precision.

Truncation (not rounding) is deliberate: it preserves chronological ordering at
microsecond resolution and can never move a quote into the future.
"""
from __future__ import annotations

import re
from datetime import datetime

_FRACTION = re.compile(r"\.(\d+)")


def _normalize_fraction(match: re.Match[str]) -> str:
    digits = (match.group(1) + "000000")[:6]
    return f".{digits}"


def parse_iso_timestamp(value: str | datetime) -> datetime:
    """Parse an ISO-8601 timestamp, tolerating any fractional-second precision."""
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("Timestamp is empty.")
    text = _FRACTION.sub(_normalize_fraction, text, count=1)
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)
