# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Narrow Textual view contract for ControlPanel execution effects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from lanista.gui.contracts import UiSeverity
from lanista.reinforcengine.rl_model.evaluation.contracts import LibraryEvaluationScope
from lanista.reinforcengine.rl_model.evaluation_restart_contracts import (
    GuiEvaluationRestartContinuation,
)
from lanista.reinforcengine.rl_model.execution_contracts import (
    EvaluationPacing,
    EvaluationVisualization,
    ExecutionSnapshot,
)
from lanista.reinforcengine.rl_model.model import RLModelRunningModes


class ControlPanelExecutionViewPort(Protocol):
    """Render and route execution effects without owning workflow state."""

    def execution_effects_available(self) -> bool:
        """Return whether execution effects available applies to this control panel execution view
        port.
        """
        ...

    def refresh_execution_runtime(self) -> None:
        """Refresh the execution runtime owned by this control panel
execution view port."""
        ...

    def selected_execution_mode(self) -> RLModelRunningModes:
        """Return the selected execution mode value provided by this control panel execution view
        port.
        """
        ...

    def selected_execution_session(self) -> str | None:
        """Return the selected execution session value provided by this control panel execution
        view port.
        """
        ...

    def selected_execution_checkpoint(self) -> str | None:
        """Return the selected execution checkpoint value provided by this control panel execution
        view port.
        """
        ...

    def selected_evaluation_visualization(self) -> EvaluationVisualization:
        """Return the selected evaluation visualization value provided by this control panel
        execution view port.
        """
        ...

    def selected_evaluation_pacing(self) -> EvaluationPacing:
        """Return the selected evaluation pacing value provided by this control panel execution
        view port.
        """
        ...

    def selected_evaluation_scenario_index(self) -> int | None:
        """Return the scenario index selected for Evaluation."""

        ...

    def selected_evaluation_library_scope(self) -> LibraryEvaluationScope | None:
        """Return the selected grouped-library evaluation scope."""
        ...

    def evaluation_restart_continuation(
        self,
    ) -> GuiEvaluationRestartContinuation | None:
        """Return the evaluation restart continuation value provided by this control panel
        execution view port.
        """
        ...

    def native_gui_restart_required(self) -> bool:
        """Return whether this process must be replaced before another run."""
        ...

    def set_post_run_continuation(
        self,
        continuation: GuiEvaluationRestartContinuation | None,
    ) -> None:
        """Store the completed GUI run for a later user action."""
        ...

    def set_execution_selection(
        self,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> None:
        """Set the execution selection owned by this control panel execution view port."""
        ...

    def apply_execution_projection(
        self,
        snapshot: ExecutionSnapshot,
        commands_locked: bool,
    ) -> None:
        """Apply the execution projection transition owned by this control panel execution view
        port.
        """
        ...

    def set_execution_status(
        self,
        activity: str,
        step: str,
        progress: float | None = None,
        total: float | None = None,
    ) -> None:
        """Set the execution status owned by this control panel execution view port."""
        ...

    def set_execution_progress(
        self,
        *,
        total: float | None,
        progress: float | None,
        reset_total: bool,
    ) -> None:
        """Set the execution progress owned by this control panel execution view port."""
        ...

    def render_execution_terminal(self, activity: str, step: str) -> None:
        """Handle the render execution terminal operation owned by this control panel execution
        view port.
        """
        ...

    def refresh_execution_buttons(self) -> None:
        """Refresh the execution buttons owned by this control panel
execution view port."""
        ...

    def publish_execution_context(self) -> None:
        """Publish the execution context owned by this control panel execution view port."""
        ...

    def notify_execution(
        self,
        message: str,
        *,
        title: str,
        severity: UiSeverity,
    ) -> None:
        """Notify the caller about the execution outcome."""
        ...

    def request_stop_confirmation(
        self,
        callback: Callable[[bool | None], None],
    ) -> None:
        """Request the stop confirmation owned by this control panel
execution view port."""
        ...

    def request_model_load(self) -> None:
        """Request the model load owned by this control panel execution view
port."""
        ...

    def request_configuration_apply(self) -> None:
        """Request the configuration apply owned by this control panel
execution view port."""
        ...

    def request_configuration_dialog(self) -> None:
        """Request the configuration dialog owned by this control panel
execution view port."""
        ...

    def start_tensorboard_for_execution(self) -> None:
        """Start the tensorboard for execution workflow owned by this control panel execution view
        port.
        """
        ...

    def dispatch_execution_ui(self, callback: Callable[[], None]) -> None:
        """Dispatch the execution ui owned by this control panel execution view port."""
        ...
