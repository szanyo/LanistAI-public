# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Stable contracts for model execution commands and state publication."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import Event
from typing import Literal, Protocol

EvaluationVisualization = Literal["gui", "headless"]
EvaluationPacing = Literal["maximum-throughput", "realtime"]


class RLModelState(Enum):
    """Externally visible lifecycle state of a model runtime."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"
    LOADING = "loading"


ACTIVE_EXECUTION_STATES = frozenset(
    {
        RLModelState.STARTING,
        RLModelState.RUNNING,
        RLModelState.PAUSED,
        RLModelState.STOPPING,
    }
)


class ExecutionRejectedError(RuntimeError):
    """Base class for typed lifecycle command rejections."""

    def __init__(self, message: str, *, state: RLModelState, run_id: str | None) -> None:
        """Initialize the execution rejected error and its owned state."""
        super().__init__(message)
        self.state = state
        self.run_id = run_id


class ExecutionAlreadyRunningError(ExecutionRejectedError):
    """Raised when a second worker is requested for an active runtime."""


class InvalidExecutionStateError(ExecutionRejectedError):
    """Raised when a command is not valid for the current lifecycle state."""


class ExecutionStopTimeoutError(ExecutionRejectedError):
    """Raised when a worker does not acknowledge a stop within its deadline."""

    def __init__(
        self,
        *,
        state: RLModelState,
        run_id: str | None,
        timeout_seconds: float,
    ) -> None:
        """Initialize the execution stop timeout error and its owned state."""
        super().__init__(
            f"Execution run {run_id or '<none>'} did not stop within "
            f"{timeout_seconds:.3f} seconds.",
            state=state,
            run_id=run_id,
        )
        self.timeout_seconds = timeout_seconds


class ExecutionLease(Protocol):
    """Exclusive policy lease owned by one execution run."""

    @property
    def run_id(self) -> str:
        """Return the immutable run identity that owns the lease."""

    def mark_reload_required(self) -> None:
        """Idempotently keep the policy gate closed after execution failure."""

    def release(self) -> None:
        """Release the lease idempotently."""


class ExecutionLeaseFactory(Protocol):
    """Create one exclusive policy lease before a run becomes visible."""

    def acquire_execution_lease(self, run_id: str) -> ExecutionLease:
        """Acquire and return the lease for ``run_id``."""


@dataclass(frozen=True)
class ExecutionSnapshot:
    """Immutable state published to application and GUI adapters."""

    run_id: str | None
    state: RLModelState
    worker_alive: bool
    stop_requested: bool = False
    failure_type: str | None = None


@dataclass(frozen=True)
class ExecutionContext:
    """Cooperative control surface visible to one model operation."""

    run_id: str
    _pause_event: Event = field(repr=False)
    _stop_event: Event = field(repr=False)

    @property
    def stop_requested(self) -> bool:
        """Return the stop requested exposed by this execution context."""
        return self._stop_event.is_set()

    def check_stop(self) -> None:
        """Raise at a cooperative cancellation point."""
        if self._stop_event.is_set():
            raise InterruptedError("Operation stopped by user")

    def wait_if_paused(self, poll_interval_seconds: float = 0.05) -> None:
        """Block cooperatively while paused and remain responsive to stop."""
        self.check_stop()
        while self._pause_event.is_set():
            if self._stop_event.wait(poll_interval_seconds):
                raise InterruptedError("Operation stopped by user")
        self.check_stop()
