# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Own dependency-light timestamp formatting for RL-model metadata."""

from datetime import UTC, datetime


def utc_now_iso() -> str:
    """Return the current UTC instant at second precision with a ``Z`` suffix."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
