# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Pure execution-snapshot projection for the ControlPanel adapter."""

from __future__ import annotations

from dataclasses import dataclass

from lanista.reinforcengine.rl_model.execution_contracts import (
    ExecutionSnapshot,
    RLModelState,
)

_TERMINAL_STATES = frozenset(
    {
        RLModelState.STOPPED,
        RLModelState.COMPLETED,
        RLModelState.FAILED,
    }
)
_COMMAND_LOCK_STATES = frozenset(
    {
        RLModelState.STARTING,
        RLModelState.STOPPING,
        RLModelState.LOADING,
    }
)
_COMMANDABLE_WORKER_STATES = frozenset(
    {
        RLModelState.RUNNING,
        RLModelState.PAUSED,
    }
)


@dataclass(frozen=True, slots=True)
class ControlPanelExecutionProjection:
    """Accepted lifecycle view derived from one immutable runtime snapshot."""

    snapshot: ExecutionSnapshot
    tracked_run_id: str | None
    task_release_eligible: bool
    commands_locked: bool

    @property
    def state(self) -> RLModelState:
        """Expose the coordinator state without translating terminal values."""

        return self.snapshot.state


class ControlPanelExecutionStateService:
    """Validate run identity and derive ControlPanel lifecycle policy."""

    @staticmethod
    def project(
        snapshot: ExecutionSnapshot,
        *,
        tracked_run_id: str | None = None,
    ) -> ControlPanelExecutionProjection | None:
        """Return an accepted projection, or ``None`` for a stale run snapshot."""

        if tracked_run_id is not None and snapshot.run_id != tracked_run_id:
            return None

        resolved_run_id = tracked_run_id or snapshot.run_id
        matching_tracked_run = (
            tracked_run_id is not None
            and snapshot.run_id == tracked_run_id
        )
        terminal = snapshot.state in _TERMINAL_STATES
        task_release_eligible = (
            matching_tracked_run
            and terminal
            and not snapshot.worker_alive
        )
        commands_locked = (
            snapshot.state in _COMMAND_LOCK_STATES
            or (
                snapshot.worker_alive
                and snapshot.state not in _COMMANDABLE_WORKER_STATES
            )
        )

        return ControlPanelExecutionProjection(
            snapshot=snapshot,
            tracked_run_id=resolved_run_id,
            task_release_eligible=task_release_eligible,
            commands_locked=commands_locked,
        )
