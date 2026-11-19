# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Recover the committed TensorBoard step floor for one logical session."""

from __future__ import annotations

from pathlib import Path


class TensorboardStepContinuityService:
    """Bound event inspection used when an existing session opens a new segment."""

    _MAX_EVENT_FILES = 32
    _MAX_EVENTS_PER_FILE = 250_000

    def last_committed_step(
        self,
        log_dir: str,
        *,
        canonical_step: int = 0,
    ) -> int:
        """Return the greatest trusted canonical or recoverable event step."""

        maximum = max(0, int(canonical_step or 0))
        root = Path(log_dir)
        if not root.is_dir():
            return maximum
        try:
            from tensorboard.backend.event_processing.event_file_loader import (
                EventFileLoader,
            )
        except ImportError:
            return maximum

        event_file_rows: list[tuple[int, str, Path]] = []
        for event_file in root.glob("events.out.tfevents.*"):
            try:
                if event_file.is_symlink() or not event_file.is_file():
                    continue
                event_file_rows.append(
                    (event_file.stat().st_mtime_ns, event_file.name, event_file)
                )
            except OSError:
                # A writer may rotate or replace a segment while recovery is
                # taking its bounded inventory.
                continue
        event_files = [
            row[2]
            for row in sorted(event_file_rows)[-self._MAX_EVENT_FILES :]
        ]
        for event_file in event_files:
            try:
                loader = EventFileLoader(str(event_file))
                for event_index, event in enumerate(loader.Load()):
                    if event_index >= self._MAX_EVENTS_PER_FILE:
                        break
                    if not getattr(getattr(event, "summary", None), "value", None):
                        continue
                    maximum = max(maximum, max(0, int(getattr(event, "step", 0) or 0)))
            except (OSError, RuntimeError, TypeError, ValueError):
                # One malformed segment must not prevent a later valid segment
                # from recovering the logical session floor.
                continue
        return maximum
