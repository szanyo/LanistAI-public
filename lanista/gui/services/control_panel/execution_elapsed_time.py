# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Track active runtime duration for the ControlPanel presentation."""

from __future__ import annotations

from dataclasses import dataclass

from lanista.reinforcengine.rl_model.execution_contracts import (
    ExecutionSnapshot,
    RLModelState,
)


@dataclass(slots=True)
class ExecutionElapsedTime:
    """Accumulate RUNNING-only duration for the currently observed run."""

    _run_id: str | None = None
    _accumulated_seconds: float = 0.0
    _running_started_at: float | None = None

    def observe(self, snapshot: ExecutionSnapshot, *, now: float) -> None:
        """Apply one execution snapshot at the supplied monotonic timestamp."""

        if snapshot.run_id != self._run_id:
            self._run_id = snapshot.run_id
            self._accumulated_seconds = 0.0
            self._running_started_at = None

        if snapshot.run_id is None:
            return
        if snapshot.state is RLModelState.RUNNING:
            if self._running_started_at is None:
                self._running_started_at = now
            return
        self._stop_running_clock(now)

    def display(self, *, now: float) -> str:
        """Return the UI-ready active duration for the current run."""

        if self._run_id is None:
            return "Elapsed: —"
        return f"Elapsed: {_format_duration(self._elapsed_seconds(now))}"

    def _elapsed_seconds(self, now: float) -> float:
        """Return accumulated duration including the active RUNNING interval."""

        if self._running_started_at is None:
            return self._accumulated_seconds
        return self._accumulated_seconds + max(0.0, now - self._running_started_at)

    def _stop_running_clock(self, now: float) -> None:
        """Commit the active RUNNING interval before a non-running state."""

        if self._running_started_at is None:
            return
        self._accumulated_seconds = self._elapsed_seconds(now)
        self._running_started_at = None


def _format_duration(seconds: float) -> str:
    """Format non-negative active seconds as a fixed-width clock value."""

    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
