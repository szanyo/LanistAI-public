# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Project coordinator-approved execution state onto the ControlPanel view."""

from __future__ import annotations

from asyncio import Task
from functools import partial
from typing import cast

from lanista.gui.adapters.control_panel.control_panel_execution_metadata_effect_adapter import (
    ControlPanelExecutionMetadataEffectAdapter,
)
from lanista.gui.adapters.control_panel.control_panel_execution_presentation_adapter import (
    ControlPanelExecutionPresentationAdapter,
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
    ExecutionTransition,
    ExecutionWorkflowLease,
)
from lanista.gui.services.control_panel.control_panel_progress_coordinator import (
    ControlPanelProgressEvent,
)
from lanista.reinforcengine.rl_model.execution_contracts import (
    ExecutionSnapshot,
    RLModelState,
)
from lanista.reinforcengine.rl_model.model import BaseModelRuntime, RLModelRunningModes

TERMINAL_EXECUTION_STATES = frozenset(
    {RLModelState.STOPPED, RLModelState.COMPLETED, RLModelState.FAILED}
)


class ControlPanelExecutionProjection:
    """Accept effects only through the single workflow coordinator state source."""

    def __init__(
        self,
        *,
        view: ControlPanelExecutionViewPort,
        workflow: ControlPanelExecutionWorkflowCoordinator,
        tasks: ControlPanelExecutionTaskOwner,
        presentation: ControlPanelExecutionPresentationAdapter,
        metadata: ControlPanelExecutionMetadataEffectAdapter,
    ) -> None:
        """Initialize the control panel execution projection and its owned state."""
        self._view = view
        self._workflow = workflow
        self._tasks = tasks
        self._presentation = presentation
        self._metadata = metadata

    def current_runtime(self) -> BaseModelRuntime | None:
        """Return the current runtime value owned by this control panel execution projection."""
        return cast(BaseModelRuntime | None, self._workflow.context.runtime)

    def attach_runtime(
        self,
        runtime: BaseModelRuntime | None,
        snapshot: ExecutionSnapshot,
    ) -> bool:
        """Return the attach runtime value owned by this control panel execution projection."""
        transition = self._workflow.attach_runtime(
            runtime,
            snapshot,
            selected_session=self._view.selected_execution_session(),
        )
        return self.apply_transition(transition)

    def apply_snapshot(
        self,
        runtime: BaseModelRuntime | None,
        snapshot: ExecutionSnapshot,
        *,
        expected_task: Task[ExecutionSnapshot] | None = None,
        lease: ExecutionWorkflowLease | None = None,
    ) -> bool:
        """Apply the snapshot transition owned by this control panel execution projection."""
        if not self._view.execution_effects_available() or runtime is None:
            return False
        effective_lease = lease
        if (
            effective_lease is None
            and expected_task is not None
            and self._tasks.operation_task is expected_task
        ):
            candidate = self._workflow.current_lease()
            if candidate is not None and candidate.runtime is runtime:
                effective_lease = candidate
        if effective_lease is not None and effective_lease.runtime is not runtime:
            return False
        transition = (
            self._workflow.observe_completion(effective_lease, snapshot)
            if effective_lease is not None
            else self._workflow.accept_snapshot(
                runtime,
                snapshot,
                selected_session=self._view.selected_execution_session(),
            )
        )
        return self.apply_transition(transition, expected_task=expected_task)

    def apply_transition(
        self,
        transition: ExecutionTransition | None,
        *,
        expected_task: Task[ExecutionSnapshot] | None = None,
    ) -> bool:
        """Apply the transition transition owned by this control panel execution projection."""
        if transition is None:
            return False
        projection = transition.projection
        self._presentation.apply_projection(
            projection.snapshot,
            commands_locked=projection.commands_locked,
        )
        if transition.release_observer:
            self._tasks.release_operation(expected_task)
        return True

    def synchronize(self) -> ExecutionSnapshot:
        """Return the synchronize value owned by this control panel execution projection."""
        if self._workflow.context.begin_pending:
            return self._workflow.context.snapshot
        runtime = self.current_runtime()
        if runtime is None:
            return self._workflow.context.snapshot
        self.apply_snapshot(runtime, runtime.get_execution_snapshot())
        return self._workflow.context.snapshot

    def apply_terminal_progress(
        self,
        runtime: BaseModelRuntime,
        snapshot: ExecutionSnapshot,
        additional_message: str = "",
        *,
        expected_task: Task[ExecutionSnapshot] | None = None,
        lease: ExecutionWorkflowLease | None = None,
    ) -> None:
        """Apply the terminal progress transition owned by this control panel execution projection."""
        self._view.dispatch_execution_ui(
            partial(
                self._apply_terminal_progress_on_ui,
                runtime,
                snapshot,
                additional_message,
                expected_task=expected_task,
                lease=lease,
            )
        )

    def _apply_terminal_progress_on_ui(
        self,
        runtime: BaseModelRuntime,
        snapshot: ExecutionSnapshot,
        additional_message: str,
        *,
        expected_task: Task[ExecutionSnapshot] | None = None,
        lease: ExecutionWorkflowLease | None = None,
    ) -> None:
        """Apply the terminal progress on ui transition owned by this control panel execution
        projection.
        """
        if not self.apply_snapshot(
            runtime,
            snapshot,
            expected_task=expected_task,
            lease=lease,
        ):
            return
        self.apply_terminal_effects_once(snapshot, additional_message)
        self._view.refresh_execution_buttons()
        self._view.publish_execution_context()

    def apply_terminal_effects_once(
        self,
        snapshot: ExecutionSnapshot,
        additional_message: str,
    ) -> None:
        """Apply the terminal effects once transition owned by this control panel execution
        projection.
        """
        effect = self._workflow.claim_terminal(
            snapshot,
            fallback_session=self._view.selected_execution_session(),
        )
        if effect is None:
            return
        self._presentation.render_terminal(
            snapshot,
            additional_message=additional_message,
        )
        self._metadata.persist_session_status(
            effect.session_name,
            effect.state,
            action_summary=f"Execution {effect.state.value}",
        )
        runtime = self.current_runtime()
        running_mode_provider = getattr(runtime, "get_running_mode", None)
        running_mode = (
            running_mode_provider()
            if callable(running_mode_provider)
            else None
        )
        if (
            effect.state is RLModelState.COMPLETED
            and running_mode is RLModelRunningModes.TRAINING
        ):
            checkpoint_name = (
                self._metadata.promote_latest_training_checkpoint_selection(
                    effect.session_name,
                    action_summary="Training completed",
                )
            )
            if checkpoint_name is not None:
                self._view.set_execution_selection(
                    effect.session_name,
                    checkpoint_name,
                )

    def accept_progress_event(self, event: ControlPanelProgressEvent) -> None:
        """Accept the progress event owned by this control panel execution
        projection."""
        runtime = cast(BaseModelRuntime, event.runtime)
        if event.snapshot.state in TERMINAL_EXECUTION_STATES:
            self._apply_terminal_progress_on_ui(runtime, event.snapshot, event.message)
            return
        if not self.apply_snapshot(runtime, event.snapshot):
            return
        self._presentation.render_live(event)
