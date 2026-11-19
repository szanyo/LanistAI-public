# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Synchronize a ControlPanel stop request with its owned worker task."""

from __future__ import annotations

import asyncio
from typing import Protocol

from lanista.reinforcengine.rl_model.execution_contracts import (
    ACTIVE_EXECUTION_STATES,
    ExecutionSnapshot,
    InvalidExecutionStateError,
    RLModelState,
)


class ExecutionStopPort(Protocol):
    """Narrow model-runtime contract required by the GUI stop workflow."""

    def get_execution_snapshot(self) -> ExecutionSnapshot:
        """Return the current immutable execution state."""

    async def stop_and_wait(self, timeout_seconds: float) -> ExecutionSnapshot:
        """Request cooperative stop and await worker cleanup."""


class StopSynchronizationError(RuntimeError):
    """Raised when the operation task and runtime report different runs."""


class ControlPanelStopService:
    """Await a runtime stop without cancelling its cleanup-owning task."""

    async def stop(
        self,
        *,
        model: ExecutionStopPort,
        expected_run_id: str,
        operation_task: asyncio.Future[ExecutionSnapshot] | None,
        timeout_seconds: float,
    ) -> ExecutionSnapshot:
        """Stop the workflow owned by this control panel stop service."""
        initial = model.get_execution_snapshot()
        self._require_owned_run(
            initial,
            expected_run_id=expected_run_id,
            source="initial runtime snapshot",
        )
        if initial.state in ACTIVE_EXECUTION_STATES:
            stopped = await model.stop_and_wait(timeout_seconds)
        elif initial.state in {
            RLModelState.STOPPED,
            RLModelState.COMPLETED,
            RLModelState.FAILED,
        } and not initial.worker_alive:
            stopped = initial
        else:
            raise InvalidExecutionStateError(
                f"Cannot stop execution while it is {initial.state.value}.",
                state=initial.state,
                run_id=initial.run_id,
            )

        self._require_owned_run(
            stopped,
            expected_run_id=expected_run_id,
            source="stopped runtime snapshot",
        )
        task_result = await self._wait_for_operation_task(operation_task)
        if task_result is not None:
            self._require_owned_run(
                task_result,
                expected_run_id=expected_run_id,
                source="completed operation task",
            )
        if stopped.worker_alive:
            raise StopSynchronizationError(
                "The execution coordinator reported completion while its worker was alive."
            )
        return stopped

    @staticmethod
    def _require_owned_run(
        snapshot: ExecutionSnapshot,
        *,
        expected_run_id: str,
        source: str,
    ) -> None:
        """Return the required owned run for this control panel stop service."""
        if snapshot.run_id != expected_run_id:
            raise StopSynchronizationError(
                f"The {source} belongs to run {snapshot.run_id!r}, "
                f"not the expected run {expected_run_id!r}."
            )

    @staticmethod
    async def _wait_for_operation_task(
        operation_task: asyncio.Future[ExecutionSnapshot] | None,
    ) -> ExecutionSnapshot | None:
        """Run the wait for operation task responsibility for this control panel stop service
        asynchronously.
        """
        if operation_task is None:
            return None
        try:
            return await asyncio.shield(operation_task)
        except asyncio.CancelledError:
            if operation_task.cancelled():
                # The coordinator snapshot is authoritative. A cancelled GUI
                # observer must not reclassify a fully stopped runtime.
                return None
            raise
