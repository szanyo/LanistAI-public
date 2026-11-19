# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Stable timestamp formatting at the GUI boundary."""

from datetime import datetime


def format_local_timestamp(raw_timestamp: str | None) -> str:
    """Format an ISO timestamp in local time, preserving invalid source text."""

    if not raw_timestamp:
        return "-"

    timestamp = str(raw_timestamp).strip()
    if not timestamp:
        return "-"

    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone()
    except (OSError, OverflowError, ValueError):
        return timestamp
    return parsed.strftime("%Y-%m-%d %H:%M:%S %Z")
