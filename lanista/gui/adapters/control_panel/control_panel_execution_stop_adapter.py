# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Own ControlPanel stop confirmation and asynchronous cleanup observation."""

from __future__ import annotations

import asyncio
import logging
from asyncio import Task
from dataclasses import dataclass
from typing import cast

from lanista.gui.adapters.control_panel.control_panel_execution_metadata_effect_adapter import (
    ControlPanelExecutionMetadataEffectAdapter,
)
from lanista.gui.adapters.control_panel.control_panel_execution_presentation_adapter import (
    ControlPanelExecutionPresentationAdapter,
)
from lanista.gui.adapters.control_panel.control_panel_execution_projection import (
    TERMINAL_EXECUTION_STATES,
    ControlPanelExecutionProjection,
)
from lanista.gui.adapters.control_panel.control_panel_execution_task_owner import (
    ControlPanelExecutionTaskOwner,
)
from lanista.gui.adapters.control_panel.control_panel_execution_view_port import (
    ControlPanelExecutionViewPort,
)
from lanista.gui.services.control_panel.control_panel_execution_workflow import (
    ControlPanelExecutionWorkflowCoordinator,
)
from lanista.gui.services.control_panel.control_panel_execution_workflow_contracts import (
    ExecutionStopCommand,
)
from lanista.gui.services.control_panel.control_panel_stop_service import (
    ControlPanelStopService,
    StopSynchronizationError,
)
from lanista.reinforcengine.rl_model.execution_contracts import (
    ExecutionRejectedError,
    ExecutionSnapshot,
    ExecutionStopTimeoutError,
    RLModelState,
)
from lanista.reinforcengine.rl_model.model import BaseModelRuntime, RLModelRunningModes


@dataclass(frozen=True, slots=True)
class StopObserverRequest:
    """GUI observer paired with an immutable workflow stop command."""

    command: ExecutionStopCommand
    operation_task: Task[ExecutionSnapshot] | None


class ControlPanelExecutionStopAdapter:
    """Run stop effects while the workflow coordinator validates ownership."""

    def __init__(
        self,
        *,
        view: ControlPanelExecutionViewPort,
        workflow: ControlPanelExecutionWorkflowCoordinator,
        projection: ControlPanelExecutionProjection,
        tasks: ControlPanelExecutionTaskOwner,
        presentation: ControlPanelExecutionPresentationAdapter,
        logger: logging.Logger,
        timeout_seconds: float,
        metadata: ControlPanelExecutionMetadataEffectAdapter,
    ) -> None:
        """Initialize the control panel execution stop adapter and its owned state."""
        self._view = view
        self._workflow = workflow
        self._projection = projection
        self._tasks = tasks
        self._presentation = presentation
        self._logger = logger
        self._timeout_seconds = timeout_seconds
        self._metadata = metadata
        self._stop_service = ControlPanelStopService()

    def request_stop(self) -> None:
        """Request the stop owned by this control panel execution stop
adapter."""
        request = self.capture_stop_request()
        if request is None:
            return

        def confirm(confirmed: bool | None) -> None:
            """Confirm the operation owned by this control panel execution stop adapter."""
            if confirmed:
                self.reset(request)

        self._view.request_stop_confirmation(confirm)

    def capture_stop_request(self) -> StopObserverRequest | None:
        """Capture the stop request required by this control panel execution stop adapter."""
        runtime = self._projection.current_runtime()
        self._projection.synchronize()
        if runtime is None:
            self._reject("Stop", "No owned execution run is available.")
            return None
        command = self._workflow.capture_stop(
            runtime,
            close_training_session=(
                self._view.selected_execution_mode() is RLModelRunningModes.TRAINING
            ),
        )
        if command is None:
            self._reject("Stop", "No owned execution run is available.")
            return None
        return StopObserverRequest(command, self._tasks.operation_task)

    def reset(self, request: StopObserverRequest) -> None:
        """Reset this control panel execution stop adapter to its initial operational state."""
        if self._tasks.stop_is_running():
            return
        command = request.command
        runtime = cast(BaseModelRuntime, command.runtime)
        snapshot = runtime.get_execution_snapshot()
        if not self._workflow.stop_is_current(
            command,
            self._projection.current_runtime(),
            snapshot,
        ):
            self._view.notify_execution(
                "The selected execution changed before stop confirmation.",
                title="Stop blocked",
                severity="warning",
            )
            return
        try:
            stopping = runtime.stop()
        except ExecutionRejectedError as error:
            self._reject("Stop", str(error))
            return
        if not self._projection.apply_transition(
            self._workflow.accept_stop_result(command, stopping)
        ):
            return
        self._view.set_execution_status("stopping", "Waiting for worker cleanup")
        self._view.refresh_execution_buttons()
        self._tasks.track_stop(asyncio.create_task(self._stop_and_reset(request)))

    async def _stop_and_reset(self, request: StopObserverRequest) -> None:
        """Stop the and reset workflow owned by this control panel execution stop adapter."""
        command = request.command
        runtime = cast(BaseModelRuntime, command.runtime)
        current = cast(Task[None] | None, asyncio.current_task())
        try:
            snapshot = await self._stop_service.stop(
                model=runtime,
                expected_run_id=command.run_id,
                operation_task=request.operation_task,
                timeout_seconds=self._timeout_seconds,
            )
        except ExecutionStopTimeoutError:
            self.present_stop_failure(
                "Stop timed out; cleanup is still running and administration remains locked.",
                runtime=runtime,
                snapshot=runtime.get_execution_snapshot(),
            )
            return
        except (StopSynchronizationError, RuntimeError) as error:
            self.present_stop_failure(
                str(error),
                runtime=runtime,
                snapshot=runtime.get_execution_snapshot(),
            )
            return
        finally:
            self._tasks.release_stop(current)
        transition = (
            self._workflow.observe_completion(command.lease, snapshot)
            if command.lease is not None
            else None
        )
        transition = transition or self._workflow.accept_stop_result(command, snapshot)
        if not self._projection.apply_transition(
            transition,
            expected_task=request.operation_task,
        ):
            return
        if snapshot.state not in TERMINAL_EXECUTION_STATES:
            self._view.refresh_execution_buttons()
            return
        self._projection.apply_terminal_effects_once(snapshot, "")
        self._view.refresh_execution_buttons()
        self._view.publish_execution_context()
        if snapshot.state is RLModelState.STOPPED:
            self.complete_stopped_progress(command)

    def complete_stopped_progress(self, command: ExecutionStopCommand) -> None:
        """Complete the stopped progress owned by this control panel
execution stop adapter."""
        self._projection.apply_terminal_effects_once(
            self._workflow.context.snapshot,
            "",
        )
        if command.close_training_session:
            self._view.set_execution_selection(None, None)
            try:
                self._metadata.persist_cleared_selection(
                    action_summary="Execution stopped"
                )
            finally:
                self._present_stopped_state()
        else:
            self._present_stopped_state()
        self._presentation.notify_feedback("reset")

    def _present_stopped_state(self) -> None:
        """Present the stopped state owned by this control panel execution
stop adapter."""
        self._view.set_execution_status(
            "stopped",
            "Create a new session to continue training",
            0,
            None,
        )
        self._view.publish_execution_context()
        self._view.refresh_execution_buttons()

    def present_stop_failure(
        self,
        message: str,
        *,
        runtime: BaseModelRuntime | None,
        snapshot: ExecutionSnapshot,
    ) -> None:
        """Present the stop failure owned by this control panel execution
stop adapter."""
        if not self._projection.apply_snapshot(runtime, snapshot):
            self._logger.debug(
                "Ignored stop failure from a stale runtime/run [run_id=%s; state=%s].",
                snapshot.run_id,
                snapshot.state.value,
            )
            return
        if snapshot.state in TERMINAL_EXECUTION_STATES:
            self._projection.apply_terminal_effects_once(snapshot, "")
        else:
            self._view.set_execution_status(snapshot.state.value, message)
        self._view.refresh_execution_buttons()
        self._view.publish_execution_context()
        self._presentation.notify_stop_failure(message)

    def _reject(self, action: str, message: str) -> None:
        """Reject the operation owned by this control panel execution stop adapter."""
        self._projection.synchronize()
        self._view.refresh_execution_buttons()
        self._presentation.notify_command_rejection(action, message)
