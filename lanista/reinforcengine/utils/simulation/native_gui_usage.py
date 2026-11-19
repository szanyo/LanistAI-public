# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Process-scoped evidence for successful embedded SUMO GUI startup."""

from __future__ import annotations

from threading import Event


class NativeGuiUsageTracker:
    """Remember whether this process image started an embedded native GUI."""

    def __init__(self) -> None:
        """Initialize an unarmed process-use marker."""
        self._started = Event()

    def mark_started(self) -> None:
        """Record successful native GUI startup for the lifetime of the process."""
        self._started.set()

    @property
    def started(self) -> bool:
        """Return whether an embedded native GUI has started in this process."""
        return self._started.is_set()


PROCESS_NATIVE_GUI_USAGE = NativeGuiUsageTracker()

