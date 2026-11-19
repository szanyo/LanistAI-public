# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Own the lifetime of one runtime-to-ControlPanel progress binding."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from lanista.reinforcengine.rl_model.execution_contracts import (
    ExecutionSnapshot,
    RLModelState,
)

ProgressCallback = Callable[
    [float | None, float | None, RLModelState, str],
    None,
]


class ExecutionUpdatePort(Protocol):
    """Run identity exposed by the latest runtime progress update."""

    @property
    def run_id(self) -> str | None:
        """Return the run that emitted the update."""


class ProgressRuntimePort(Protocol):
    """Narrow runtime surface required for progress observation."""

    def get_execution_snapshot(self) -> ExecutionSnapshot:
        """Return the authoritative immutable execution snapshot."""

    def get_last_execution_update(self) -> ExecutionUpdatePort | None:
        """Return the latest progress update, when one exists."""

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """Install one progress callback."""

    def clear_progress_callback(self, callback: ProgressCallback) -> bool:
        """Clear ``callback`` only when it is still the installed callback."""


@dataclass(frozen=True, slots=True)
class ControlPanelProgressEvent:
    """Immutable progress event accepted from one binding generation."""

    runtime: ProgressRuntimePort
    binding_generation: int
    snapshot: ExecutionSnapshot
    total: float | None
    progress: float | None
    reported_state: RLModelState
    message: str


class ProgressEventSink(Protocol):
    """UI-side consumer of validated progress events."""

    def accept_progress_event(self, event: ControlPanelProgressEvent) -> None:
        """Apply an event after UI-thread dispatch."""


class UiDispatchPort(Protocol):
    """Dispatcher used to cross into the Textual application thread."""

    def dispatch_progress_to_ui(
        self,
        callback: Callable[[], None],
    ) -> None:
        """Schedule or immediately run one UI callback."""


@dataclass(frozen=True, slots=True)
class _ProgressBinding:
    """Carry the data required for progress binding."""

    generation: int
    runtime: ProgressRuntimePort
    callback: ProgressCallback


class ControlPanelProgressCoordinator:
    """Bind one runtime callback and reject stale work around UI dispatch."""

    def __init__(
        self,
        *,
        dispatcher: UiDispatchPort,
        sink: ProgressEventSink,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the control panel progress coordinator and its owned state."""
        self._dispatcher: UiDispatchPort | None = dispatcher
        self._sink: ProgressEventSink | None = sink
        self._logger = logger or logging.getLogger(__name__)
        self._generation = 0
        self._binding: _ProgressBinding | None = None
        self._active = True

    def activate(
        self,
        *,
        dispatcher: UiDispatchPort,
        sink: ProgressEventSink,
    ) -> None:
        """Open dispatch after a mount, invalidating any former lifetime."""

        if (
            self._active
            and self._dispatcher is dispatcher
            and self._sink is sink
        ):
            return
        self.detach()
        self._generation += 1
        self._dispatcher = dispatcher
        self._sink = sink
        self._active = True

    def bind(self, runtime: ProgressRuntimePort) -> None:
        """Replace the current callback with one owned by a new generation."""

        if not self._active or self._dispatcher is None or self._sink is None:
            raise RuntimeError("Cannot bind progress after the coordinator is closed.")

        self.detach()
        self._generation += 1
        generation = self._generation

        def runtime_bound_progress(
            total: float | None,
            progress: float | None,
            reported_state: RLModelState,
            message: str,
        ) -> None:
            """Perform the runtime bound progress step within the enclosing bind workflow."""
            self._receive_runtime_progress(
                runtime=runtime,
                generation=generation,
                total=total,
                progress=progress,
                reported_state=reported_state,
                message=message,
            )

        binding = _ProgressBinding(
            generation=generation,
            runtime=runtime,
            callback=runtime_bound_progress,
        )
        self._binding = binding
        try:
            runtime.set_progress_callback(runtime_bound_progress)
        except Exception:
            if self._binding is binding:
                self._binding = None
                self._generation += 1
            self._clear_exact_callback(runtime, runtime_bound_progress)
            raise

    def detach(self) -> None:
        """Invalidate and clear only the callback owned by this coordinator."""

        binding = self._binding
        if binding is None:
            return
        self._binding = None
        self._generation += 1
        self._clear_exact_callback(binding.runtime, binding.callback)

    def close(self) -> None:
        """Reject queued and future events, then release the exact callback."""

        self._active = False
        self.detach()
        self._generation += 1
        self._dispatcher = None
        self._sink = None

    def _receive_runtime_progress(
        self,
        *,
        runtime: ProgressRuntimePort,
        generation: int,
        total: float | None,
        progress: float | None,
        reported_state: RLModelState,
        message: str,
    ) -> None:
        """Handle the receive runtime progress operation owned by this control panel progress
        coordinator.
        """
        if not self._owns(runtime, generation):
            return

        execution_update = runtime.get_last_execution_update()
        snapshot = runtime.get_execution_snapshot()
        callback_run_id = (
            execution_update.run_id
            if execution_update is not None
            else snapshot.run_id
        )
        if callback_run_id != snapshot.run_id:
            return

        event = ControlPanelProgressEvent(
            runtime=runtime,
            binding_generation=generation,
            snapshot=snapshot,
            total=total,
            progress=progress,
            reported_state=reported_state,
            message=message,
        )
        dispatcher = self._dispatcher
        if dispatcher is None:
            return
        dispatcher.dispatch_progress_to_ui(
            lambda: self._deliver_if_current(event)
        )

    def _deliver_if_current(self, event: ControlPanelProgressEvent) -> None:
        """Return whether deliver if current applies to this control panel progress coordinator."""
        if not self._owns(event.runtime, event.binding_generation):
            return
        if event.runtime.get_execution_snapshot() != event.snapshot:
            return
        sink = self._sink
        if sink is not None:
            sink.accept_progress_event(event)

    def _owns(
        self,
        runtime: ProgressRuntimePort,
        generation: int,
    ) -> bool:
        """Return the owns value owned by this control panel progress coordinator."""
        binding = self._binding
        return bool(
            self._active
            and binding is not None
            and binding.runtime is runtime
            and binding.generation == generation
        )

    def _clear_exact_callback(
        self,
        runtime: ProgressRuntimePort,
        callback: ProgressCallback,
    ) -> None:
        """Clear the exact callback owned by this control panel progress coordinator."""
        try:
            runtime.clear_progress_callback(callback)
        except Exception:
            self._logger.exception(
                "Could not detach the ControlPanel progress adapter."
            )
