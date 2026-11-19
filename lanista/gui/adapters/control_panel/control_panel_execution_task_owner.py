# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Own cancellable GUI tasks for ControlPanel execution."""

from __future__ import annotations

import asyncio
from asyncio import Task

from lanista.reinforcengine.rl_model.execution_contracts import ExecutionSnapshot


class ControlPanelExecutionTaskOwner:
    """Track GUI observers without claiming ownership of the runtime worker."""

    def __init__(self) -> None:
        """Initialize the control panel execution task owner and its owned state."""
        self._begin_task: Task[None] | None = None
        self._operation_task: Task[ExecutionSnapshot] | None = None
        self._stop_task: Task[None] | None = None

    @property
    def begin_task(self) -> Task[None] | None:
        """Return the background runtime-begin observer."""

        return self._begin_task

    @property
    def operation_task(self) -> Task[ExecutionSnapshot] | None:
        """Return the operation task exposed by this control panel execution task owner."""
        return self._operation_task

    @property
    def stop_task(self) -> Task[None] | None:
        """Return the stop task exposed by this control panel execution task owner."""
        return self._stop_task

    def track_operation(self, task: Task[ExecutionSnapshot] | None) -> None:
        """Track the operation owned by this control panel execution task
        owner."""
        self._operation_task = task

    def track_begin(self, task: Task[None] | None) -> None:
        """Track preparation that may outlive the initiating UI callback."""

        self._begin_task = task

    def track_stop(self, task: Task[None] | None) -> None:
        """Track the stop owned by this control panel execution task owner."""
        self._stop_task = task

    def operation_is_running(self) -> bool:
        """Return whether operation is running applies to this control panel execution task owner."""
        task = self._operation_task
        return task is not None and not task.done()

    def begin_is_running(self) -> bool:
        """Return whether runtime preparation is still pending."""

        task = self._begin_task
        return task is not None and not task.done()

    def stop_is_running(self) -> bool:
        """Return whether stop is running applies to this control panel execution task owner."""
        task = self._stop_task
        return task is not None and not task.done()

    def release_operation(self, expected: Task[ExecutionSnapshot] | None) -> bool:
        """Release the operation owned by this control panel execution task owner."""
        if expected is None or self._operation_task is not expected:
            return False
        self._operation_task = None
        return True

    def release_begin(self, expected: Task[None] | None) -> bool:
        """Release only the runtime-begin task that still owns the slot."""

        if expected is None or self._begin_task is not expected:
            return False
        self._begin_task = None
        return True

    def release_stop(self, expected: Task[None] | None) -> bool:
        """Release the stop owned by this control panel execution task owner."""
        if expected is None or self._stop_task is not expected:
            return False
        self._stop_task = None
        return True

    async def close(self) -> None:
        """Cancel GUI observers without pretending to cancel blocking preparation."""

        owned = tuple(task for task in (self._operation_task, self._stop_task) if task is not None)
        self._begin_task = None
        for task in owned:
            if not task.done():
                task.cancel()
        if owned:
            await asyncio.gather(*owned, return_exceptions=True)
        if self._operation_task in owned:
            self._operation_task = None
        if self._stop_task in owned:
            self._stop_task = None
