# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Await a runtime-owned run after a ControlPanel UI lifetime is replaced."""

from __future__ import annotations

import asyncio
from typing import Protocol

from lanista.reinforcengine.rl_model.execution_contracts import (
    ACTIVE_EXECUTION_STATES,
    ExecutionSnapshot,
)


class AttachedExecutionRuntimePort(Protocol):
    """Snapshot surface needed to observe, but never own, an existing worker."""

    def get_execution_snapshot(self) -> ExecutionSnapshot:
        """Return the execution snapshot exposed by this attached execution runtime port."""
        ...


class ControlPanelAttachedExecutionHandle:
    """Poll one immutable run identity until its runtime publishes a terminal state."""

    def __init__(
        self,
        runtime: AttachedExecutionRuntimePort,
        run_id: str,
        *,
        poll_interval_seconds: float = 0.01,
    ) -> None:
        """Initialize the control panel attached execution handle and its owned state."""
        self._runtime = runtime
        self._run_id = run_id
        self._poll_interval_seconds = poll_interval_seconds

    @property
    def run_id(self) -> str:
        """Return the run id exposed by this control panel attached execution handle."""
        return self._run_id

    def snapshot(self) -> ExecutionSnapshot:
        """Return the snapshot value owned by this control panel attached execution handle."""
        return self._runtime.get_execution_snapshot()

    async def wait(
        self,
        timeout_seconds: float | None = None,
    ) -> ExecutionSnapshot:
        """Wait for the operation owned by this control panel attached execution handle to finish.
        """
        loop = asyncio.get_running_loop()
        deadline = None if timeout_seconds is None else loop.time() + timeout_seconds
        while True:
            snapshot = self.snapshot()
            if (
                snapshot.run_id != self._run_id
                or snapshot.state not in ACTIVE_EXECUTION_STATES
            ):
                return snapshot
            if deadline is not None and loop.time() >= deadline:
                raise TimeoutError
            await asyncio.sleep(self._poll_interval_seconds)
