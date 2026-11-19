# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Immutable contracts for ControlPanel execution ownership transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lanista.gui.services.control_panel.control_panel_execution_state_service import (
    ControlPanelExecutionProjection,
)
from lanista.reinforcengine.rl_model.execution_contracts import (
    ExecutionSnapshot,
    RLModelState,
)


class ExecutionWorkflowRuntime(Protocol):
    """Runtime identity and snapshot surface needed by the registry."""

    def get_execution_snapshot(self) -> ExecutionSnapshot:
        """Return the authoritative execution snapshot."""


class ExecutionWorkflowHandle(Protocol):
    """Runtime-owned completion handle tracked independently of GUI tasks."""

    @property
    def run_id(self) -> str:
        """Return the immutable run identity."""

    def snapshot(self) -> ExecutionSnapshot:
        """Return the handle's current execution snapshot."""

    async def wait(
        self,
        timeout_seconds: float | None = None,
    ) -> ExecutionSnapshot:
        """Await runtime-owned completion without owning the worker."""


@dataclass(frozen=True, slots=True)
class ExecutionWorkflowContext:
    """Complete immutable ownership state for one coordinator generation."""

    generation: int
    active: bool
    runtime: ExecutionWorkflowRuntime | None
    run_id: str | None
    handle: ExecutionWorkflowHandle | None
    originating_session: str | None
    snapshot: ExecutionSnapshot
    terminal_claim: tuple[str | None, RLModelState] | None
    begin_pending: bool


@dataclass(frozen=True, slots=True)
class ExecutionBeginCommand:
    """Ownership captured immediately before a runtime begin call."""

    generation: int
    runtime: ExecutionWorkflowRuntime
    previous_run_id: str | None
    previous_session: str | None
    previous_snapshot: ExecutionSnapshot


@dataclass(frozen=True, slots=True)
class ExecutionWorkflowLease:
    """Current runtime/run/handle identity carried by completion observers."""

    generation: int
    runtime: ExecutionWorkflowRuntime
    run_id: str
    handle: ExecutionWorkflowHandle


@dataclass(frozen=True, slots=True)
class ExecutionCommand:
    """Pause or resume command bound to one coordinator generation and run."""

    generation: int
    runtime: ExecutionWorkflowRuntime
    run_id: str


@dataclass(frozen=True, slots=True)
class ExecutionStopCommand:
    """Stop confirmation data frozen before a modal can become stale."""

    generation: int
    runtime: ExecutionWorkflowRuntime
    run_id: str
    lease: ExecutionWorkflowLease | None
    close_training_session: bool


@dataclass(frozen=True, slots=True)
class ExecutionTransition:
    """Accepted projection and ownership side effects for the UI adapter."""

    context: ExecutionWorkflowContext
    projection: ControlPanelExecutionProjection
    release_observer: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionAdoption:
    """Validated handle adoption result."""

    lease: ExecutionWorkflowLease | None
    transition: ExecutionTransition
    contract_matches: bool


@dataclass(frozen=True, slots=True)
class BeginFailureReconciliation:
    """Result of inspecting runtime state after begin raised."""

    adopted_new_run: bool
    transition: ExecutionTransition | None


@dataclass(frozen=True, slots=True)
class TerminalSessionEffect:
    """Exactly-once terminal persistence decision for the originating session."""

    state: RLModelState
    session_name: str | None
