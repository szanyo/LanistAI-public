# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Compose focused ControlPanel execution effect components."""

from __future__ import annotations

import logging
from asyncio import Task

from lanista.gui.adapters.control_panel.control_panel_execution_command_adapter import (
    ControlPanelExecutionCommandAdapter,
)
from lanista.gui.adapters.control_panel.control_panel_execution_metadata_effect_adapter import (
    ControlPanelExecutionMetadataEffectAdapter,
)
from lanista.gui.adapters.control_panel.control_panel_execution_presentation_adapter import (
    ControlPanelExecutionPresentationAdapter,
)
from lanista.gui.adapters.control_panel.control_panel_execution_projection import (
    ControlPanelExecutionProjection,
)
from lanista.gui.adapters.control_panel.control_panel_execution_stop_adapter import (
    ControlPanelExecutionStopAdapter,
    StopObserverRequest,
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
    ExecutionTransition,
    ExecutionWorkflowLease,
)
from lanista.gui.services.control_panel.control_panel_progress_coordinator import (
    ControlPanelProgressCoordinator,
    ControlPanelProgressEvent,
)
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.execution_contracts import (
    ExecutionSnapshot,
    RLModelState,
)
from lanista.reinforcengine.rl_model.model import BaseModelRuntime, RLModelRunningModes
from lanista.reinforcengine.rl_model.operation_presentation_service import (
    QuickStartPresentation,
)


class ControlPanelExecutionEffectAdapter:
    """Expose one execution boundary backed by one workflow state source."""

    def __init__(
        self,
        *,
        view: ControlPanelExecutionViewPort,
        services: ApplicationServices,
        workflow: ControlPanelExecutionWorkflowCoordinator,
        progress: ControlPanelProgressCoordinator,
        logger: logging.Logger,
        stop_timeout_seconds: float,
    ) -> None:
        """Initialize the control panel execution effect adapter and its owned state."""
        self._workflow = workflow
        self._tasks = ControlPanelExecutionTaskOwner()
        metadata = ControlPanelExecutionMetadataEffectAdapter(
            view=view,
            services=services,
            logger=logger,
        )
        self._presentation = ControlPanelExecutionPresentationAdapter(view)
        self._projection = ControlPanelExecutionProjection(
            view=view,
            workflow=workflow,
            tasks=self._tasks,
            presentation=self._presentation,
            metadata=metadata,
        )
        self._commands = ControlPanelExecutionCommandAdapter(
            view=view,
            services=services,
            workflow=workflow,
            projection=self._projection,
            tasks=self._tasks,
            presentation=self._presentation,
            progress=progress,
            logger=logger,
            metadata=metadata,
        )
        self._stop = ControlPanelExecutionStopAdapter(
            view=view,
            workflow=workflow,
            projection=self._projection,
            tasks=self._tasks,
            presentation=self._presentation,
            logger=logger,
            timeout_seconds=stop_timeout_seconds,
            metadata=metadata,
        )

    @property
    def workflow(self) -> ControlPanelExecutionWorkflowCoordinator:
        """Return the workflow exposed by this control panel execution effect adapter."""
        return self._workflow

    @property
    def operation_task(self) -> Task[ExecutionSnapshot] | None:
        """Return the operation task exposed by this control panel execution effect adapter."""
        return self._tasks.operation_task

    @operation_task.setter
    def operation_task(self, task: Task[ExecutionSnapshot] | None) -> None:
        """Handle the operation task operation owned by this control panel execution effect
        adapter.
        """
        self._tasks.track_operation(task)

    @property
    def stop_task(self) -> Task[None] | None:
        """Return the stop task exposed by this control panel execution effect adapter."""
        return self._tasks.stop_task

    @stop_task.setter
    def stop_task(self, task: Task[None] | None) -> None:
        """Stop the task workflow owned by this control panel execution effect adapter."""
        self._tasks.track_stop(task)

    def activate(self) -> None:
        """Activate the operation owned by this control panel execution effect adapter."""
        self._workflow.activate()

    async def close(self) -> None:
        """Close resources owned by this control panel execution effect adapter."""
        self._workflow.close()
        await self._tasks.close()

    def current_runtime(self) -> BaseModelRuntime | None:
        """Return the current runtime value owned by this control panel execution effect adapter."""
        return self._projection.current_runtime()

    def attach_runtime(
        self,
        runtime: BaseModelRuntime | None,
        snapshot: ExecutionSnapshot,
    ) -> bool:
        """Return the attach runtime value owned by this control panel execution effect adapter."""
        detached_run_matches = self._workflow.detached_run_matches(snapshot)
        accepted = self._projection.attach_runtime(runtime, snapshot)
        if accepted and runtime is not None:
            self._commands.reconcile_attached_runtime(
                runtime,
                snapshot,
                project_detached_terminal=detached_run_matches,
            )
        return accepted

    def apply_snapshot(
        self,
        runtime: BaseModelRuntime | None,
        snapshot: ExecutionSnapshot,
        *,
        expected_task: Task[ExecutionSnapshot] | None = None,
        lease: ExecutionWorkflowLease | None = None,
    ) -> bool:
        """Apply the snapshot transition owned by this control panel execution effect adapter."""
        return self._projection.apply_snapshot(
            runtime,
            snapshot,
            expected_task=expected_task,
            lease=lease,
        )

    def apply_transition(
        self,
        transition: ExecutionTransition | None,
        *,
        expected_task: Task[ExecutionSnapshot] | None = None,
    ) -> bool:
        """Apply the transition transition owned by this control panel execution effect adapter."""
        return self._projection.apply_transition(
            transition,
            expected_task=expected_task,
        )

    def synchronize(self) -> ExecutionSnapshot:
        """Return the synchronize value owned by this control panel execution effect adapter."""
        return self._projection.synchronize()

    def build_quick_start_presentation(
        self,
        *,
        mode: RLModelRunningModes,
        panel_state: RLModelState,
        has_current_model: bool,
        can_evaluate: bool,
        can_infer: bool,
        has_training_session: bool,
        inspect_is_running: bool,
    ) -> QuickStartPresentation:
        """Build the quick start presentation required by this control panel execution effect
        adapter.
        """
        return self._presentation.build_quick_start(
            mode=mode,
            panel_state=panel_state,
            has_current_model=has_current_model,
            can_evaluate=can_evaluate,
            can_infer=can_infer,
            has_training_session=has_training_session,
            inspect_is_running=inspect_is_running,
        )

    def start(self) -> None:
        """Start the workflow owned by this control panel execution effect adapter."""
        self._commands.start()

    def resume(self) -> None:
        """Resume the operation owned by this control panel execution effect adapter."""
        self._commands.resume()

    def pause(self) -> None:
        """Handle the pause operation owned by this control panel execution effect adapter."""
        self._commands.pause()

    async def observe_completion(
        self,
        lease: ExecutionWorkflowLease,
    ) -> ExecutionSnapshot:
        """Run the observe completion responsibility for this control panel execution effect
        adapter asynchronously.
        """
        return await self._commands.observe_completion(lease)

    def configure_running_mode(self) -> None:
        """Configure the running mode owned by this control panel execution effect adapter."""
        self._commands.configure_running_mode()

    def request_stop(self) -> None:
        """Request the stop owned by this control panel execution effect
adapter."""
        self._stop.request_stop()

    def capture_stop_request(self) -> StopObserverRequest | None:
        """Capture the stop request required by this control panel execution effect adapter."""
        return self._stop.capture_stop_request()

    def reset(self, request: StopObserverRequest) -> None:
        """Reset this control panel execution effect adapter to its initial operational state."""
        self._stop.reset(request)

    def complete_stopped_progress(self, command: ExecutionStopCommand) -> None:
        """Complete the stopped progress owned by this control panel
execution effect adapter."""
        self._stop.complete_stopped_progress(command)

    def present_stop_failure(
        self,
        message: str,
        *,
        runtime: BaseModelRuntime | None,
        snapshot: ExecutionSnapshot,
    ) -> None:
        """Present the stop failure owned by this control panel execution
effect adapter."""
        self._stop.present_stop_failure(
            message,
            runtime=runtime,
            snapshot=snapshot,
        )

    def apply_terminal_progress(
        self,
        runtime: BaseModelRuntime,
        snapshot: ExecutionSnapshot,
        additional_message: str = "",
        *,
        expected_task: Task[ExecutionSnapshot] | None = None,
        lease: ExecutionWorkflowLease | None = None,
    ) -> None:
        """Apply the terminal progress transition owned by this control panel execution effect
        adapter.
        """
        self._projection.apply_terminal_progress(
            runtime,
            snapshot,
            additional_message,
            expected_task=expected_task,
            lease=lease,
        )

    def accept_progress_event(self, event: ControlPanelProgressEvent) -> None:
        """Accept the progress event owned by this control panel execution
effect adapter."""
        self._projection.accept_progress_event(event)
