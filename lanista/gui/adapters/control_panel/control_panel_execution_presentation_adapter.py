# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Project accepted execution effects onto the ControlPanel view port."""

from __future__ import annotations

from lanista.gui.adapters.control_panel.control_panel_execution_view_port import (
    ControlPanelExecutionViewPort,
)
from lanista.gui.contracts import normalize_ui_severity
from lanista.gui.services.control_panel.control_panel_progress_coordinator import (
    ControlPanelProgressEvent,
)
from lanista.reinforcengine.rl_model.execution_contracts import ExecutionSnapshot, RLModelState
from lanista.reinforcengine.rl_model.model import RLModelRunningModes
from lanista.reinforcengine.rl_model.operation_presentation_service import (
    OperationPresentationService,
    QuickStartPresentation,
)


class ControlPanelExecutionPresentationAdapter:
    """Render workflow-approved state without making ownership decisions."""

    def __init__(
        self,
        view: ControlPanelExecutionViewPort,
        presentation: OperationPresentationService | None = None,
    ) -> None:
        """Initialize the control panel execution presentation adapter and its owned state."""
        self._view = view
        self._presentation = presentation or OperationPresentationService()

    def apply_projection(
        self,
        snapshot: ExecutionSnapshot,
        *,
        commands_locked: bool,
    ) -> None:
        """Apply the projection transition owned by this control panel execution presentation
        adapter.
        """
        self._view.apply_execution_projection(snapshot, commands_locked)

    def build_quick_start(
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
        """Return the operation-core projection used by the Start button."""

        return self._presentation.get_quick_start_presentation(
            mode=mode,
            panel_state=panel_state,
            has_current_model=has_current_model,
            can_evaluate=can_evaluate,
            can_infer=can_infer,
            has_training_session=has_training_session,
            inspect_is_running=inspect_is_running,
        )

    def render_live(self, event: ControlPanelProgressEvent) -> None:
        """Handle the render live operation owned by this control panel execution presentation
        adapter.
        """
        if event.total is not None and event.total >= 0:
            total = event.total
            reset_total = False
        elif event.total is None:
            total = None
            reset_total = True
        else:
            total = None
            reset_total = False
        progress = event.progress if event.progress is not None and event.progress >= 0 else None
        self._view.set_execution_progress(
            total=total,
            progress=progress,
            reset_total=reset_total,
        )
        presentation = self._presentation.get_operation_live_state_presentation(
            panel_state=event.snapshot.state,
            additional_message=event.message,
        )
        self._view.set_execution_status(
            presentation.activity,
            presentation.step,
        )
        self._view.refresh_execution_buttons()

    def render_terminal(
        self,
        snapshot: ExecutionSnapshot,
        *,
        additional_message: str,
    ) -> None:
        """Handle the render terminal operation owned by this control panel execution presentation
        adapter.
        """
        presentation = self._presentation.get_operation_terminal_state_presentation(
            status=snapshot.state,
            additional_message=additional_message,
        )
        self._view.set_execution_progress(
            total=None,
            progress=0,
            reset_total=True,
        )
        self._view.render_execution_terminal(
            presentation.activity,
            presentation.step,
        )

    def notify_feedback(self, action: str) -> None:
        """Notify the caller about the feedback outcome."""
        feedback = self._presentation.get_operation_feedback(action)
        self._view.notify_execution(
            feedback.message,
            title=feedback.title,
            severity=normalize_ui_severity(feedback.severity),
        )

    def notify_command_rejection(self, action: str, message: str) -> None:
        """Notify the caller about the command rejection outcome."""
        self._view.notify_execution(
            message,
            title=f"{action} blocked",
            severity="warning",
        )

    def notify_stop_failure(self, message: str) -> None:
        """Notify the caller about the stop failure outcome."""
        self._view.notify_execution(
            message,
            title="Execution stop failed",
            severity="error",
        )
