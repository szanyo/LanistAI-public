# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Textual adapter for ControlPanel configuration dialog effects."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from lanista.gui.services.configuration.control_panel_configuration_contracts import (
    ConfigurationCommand,
    ConfigurationDialogSelection,
    ControlPanelConfigurationDialogPlan,
    EvaluationTarget,
)
from lanista.gui.utils.navigation import route_ids as nav_routes
from lanista.gui.widgets.dialogs.configuration_dialog import RunConfigurationDialog
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.evaluation.contracts import LibraryEvaluationScope
from lanista.reinforcengine.rl_model.execution_contracts import (
    EvaluationPacing,
    EvaluationVisualization,
)
from lanista.reinforcengine.rl_model.model import RLModelRunningModes
from lanista.reinforcengine.rl_model.runtime.configuration_state_service import (
    ConfigurationRuntimeState,
)


class ConfigurationScreenActionsPort(Protocol):
    """HomeScreen actions used by the configuration workflow."""

    def action_apply_configuration(
        self,
        start_after_apply: bool = False,
        expected_model_uuid: str | None = None,
    ) -> None:
        """Handle the apply configuration user action for this configuration screen actions port."""
        ...

    def action_load_model(self, reopen_startup_dialog: bool = False) -> None:
        """Handle the load model user action for this configuration screen actions port."""
        ...

    def action_open_tensorboard_pack_selector(
        self,
        reopen_startup_dialog: bool = False,
    ) -> None:
        """Handle the open tensorboard pack selector user action for this configuration screen
        actions port.
        """
        ...

    def action_open_compare_builder(self) -> None:
        """Handle the open compare builder user action for this configuration screen actions port.
        """
        ...

    def action_edit_model_details(
        self,
        reopen_route_on_close: str | None = None,
        reopen_opener_on_close: Callable[[], None] | None = None,
        open_details_directly: bool = False,
        selected_session_name: str | None = None,
        selected_checkpoint_name: str | None = None,
        expected_model_uuid: str | None = None,
    ) -> None:
        """Handle the edit model details user action for this configuration screen actions port."""
        ...

    def action_create_new_model(
        self,
        framework_uuid: str | None = None,
        reopen_startup_dialog: bool = False,
    ) -> None:
        """Handle the create new model user action for this configuration screen actions port."""
        ...

    def action_create_new_session(
        self,
        reopen_startup_dialog: bool = False,
        selected_running_mode: RLModelRunningModes | None = None,
        selected_framework_uuid: str | None = None,
        selected_session_name: str | None = None,
        expected_model_uuid: str | None = None,
    ) -> None:
        """Handle the create new session user action for this configuration screen actions port."""
        ...


class ControlPanelConfigurationDialogFactory:
    """Map application runtime state into one Textual dialog instance."""

    @staticmethod
    def build(
        *,
        runtime_state: ConfigurationRuntimeState,
        framework_options: list[tuple[str, str]],
        inspect_is_running: bool,
        application_services: ApplicationServices,
        auto_open_session_selector: bool,
        auto_open_checkpoint_selector: bool,
        selected_evaluation_visualization: EvaluationVisualization = "gui",
        selected_evaluation_pacing: EvaluationPacing = "maximum-throughput",
        evaluation_scenario_options: list[tuple[str, int]] | None = None,
        selected_evaluation_scenario_index: int | None = None,
        selected_evaluation_library_scope: LibraryEvaluationScope | None = None,
        selected_evaluation_target: EvaluationTarget = "single",
    ) -> RunConfigurationDialog:
        """Build the configured result owned by this control panel configuration dialog factory."""
        return RunConfigurationDialog(
            has_model=runtime_state.has_valid_model,
            loaded_model_name=runtime_state.loaded_model_name,
            expected_model_uuid=runtime_state.loaded_model_uuid,
            can_evaluate=runtime_state.can_evaluate,
            can_new_session=runtime_state.can_new_session,
            can_confirm=runtime_state.can_confirm,
            can_start_run=runtime_state.can_start_run,
            configuration_locked=runtime_state.configuration_locked,
            selected_mode=runtime_state.selected_mode,
            framework_options=framework_options,
            selected_framework_uuid=runtime_state.selected_framework_uuid,
            session_options=runtime_state.session_options,
            selected_session_name=runtime_state.selected_session_name,
            checkpoint_options=runtime_state.checkpoint_options,
            selected_checkpoint_name=runtime_state.selected_checkpoint_name,
            selected_evaluation_visualization=selected_evaluation_visualization,
            selected_evaluation_pacing=selected_evaluation_pacing,
            evaluation_scenario_options=evaluation_scenario_options or [],
            selected_evaluation_scenario_index=selected_evaluation_scenario_index,
            selected_evaluation_library_scope=selected_evaluation_library_scope,
            selected_evaluation_target=selected_evaluation_target,
            inspect_is_running=inspect_is_running,
            configuration_status_items=runtime_state.configuration_status_items,
            framework_statuses=runtime_state.framework_statuses,
            structure_check_items=runtime_state.structure_check_items,
            runtime_check_items=runtime_state.runtime_check_items,
            application_services=application_services,
            auto_open_session_selector=auto_open_session_selector,
            auto_open_checkpoint_selector=auto_open_checkpoint_selector,
        )


class ControlPanelConfigurationCommandAdapter:
    """Dispatch typed configuration commands to the containing HomeScreen."""

    _EDIT_COMMANDS = frozenset(
        {
            ConfigurationCommand.EDIT_MODEL,
            ConfigurationCommand.EDIT_SESSION,
            ConfigurationCommand.EDIT_CHECKPOINT,
        }
    )

    def __init__(self, screen: ConfigurationScreenActionsPort) -> None:
        """Initialize the control panel configuration command adapter and its owned state."""
        self._screen = screen

    def dispatch(
        self,
        plan: ControlPanelConfigurationDialogPlan,
        *,
        reopen_session_details: Callable[[], None],
        reopen_checkpoint_details: Callable[[], None],
        prepare_new_session: Callable[[ConfigurationDialogSelection], None],
    ) -> None:
        """Dispatch the operation owned by this control panel configuration command adapter."""
        selection = plan.selection
        if selection is None or plan.command is ConfigurationCommand.NONE:
            return
        if plan.command in self._EDIT_COMMANDS:
            self._dispatch_edit(
                plan.command,
                selection,
                reopen_session_details=reopen_session_details,
                reopen_checkpoint_details=reopen_checkpoint_details,
            )
            return
        self._dispatch_action(plan, selection, prepare_new_session)

    def _dispatch_action(
        self,
        plan: ControlPanelConfigurationDialogPlan,
        selection: ConfigurationDialogSelection,
        prepare_new_session: Callable[[ConfigurationDialogSelection], None],
    ) -> None:
        """Dispatch the action owned by this control panel configuration command adapter."""
        command = plan.command
        if command is ConfigurationCommand.APPLY_CONFIGURATION:
            self._screen.action_apply_configuration(
                start_after_apply=selection.mode is RLModelRunningModes.TRAINING,
                expected_model_uuid=selection.expected_model_uuid,
            )
        elif command is ConfigurationCommand.LOAD_MODEL:
            self._screen.action_load_model(reopen_startup_dialog=plan.reopen_startup_dialog)
        elif command is ConfigurationCommand.INSPECT_TENSORBOARD:
            self._screen.action_open_tensorboard_pack_selector(
                reopen_startup_dialog=plan.reopen_startup_dialog
            )
        elif command is ConfigurationCommand.CREATE_COMPARISON_PACK:
            self._screen.action_open_compare_builder()
        elif command is ConfigurationCommand.CREATE_MODEL:
            self._screen.action_create_new_model(
                selection.framework_uuid,
                reopen_startup_dialog=plan.reopen_startup_dialog,
            )
        elif command is ConfigurationCommand.CREATE_SESSION:
            prepare_new_session(selection)
            self._screen.action_create_new_session(
                reopen_startup_dialog=plan.reopen_startup_dialog,
                selected_running_mode=selection.mode,
                selected_framework_uuid=selection.framework_uuid,
                selected_session_name=selection.session_name,
                expected_model_uuid=selection.expected_model_uuid,
            )

    def _dispatch_edit(
        self,
        command: ConfigurationCommand,
        selection: ConfigurationDialogSelection,
        *,
        reopen_session_details: Callable[[], None],
        reopen_checkpoint_details: Callable[[], None],
    ) -> None:
        """Dispatch the edit owned by this control panel configuration command adapter."""
        if command is ConfigurationCommand.EDIT_MODEL:
            self._screen.action_edit_model_details(
                open_details_directly=True,
                selected_session_name=selection.session_name,
                selected_checkpoint_name=selection.checkpoint_name,
                expected_model_uuid=selection.expected_model_uuid,
            )
        elif command is ConfigurationCommand.EDIT_SESSION:
            self._screen.action_edit_model_details(
                reopen_route_on_close=nav_routes.CONFIGURATION_DIALOG,
                reopen_opener_on_close=reopen_session_details,
                open_details_directly=True,
                selected_session_name=selection.session_name,
                selected_checkpoint_name=None,
                expected_model_uuid=selection.expected_model_uuid,
            )
        elif command is ConfigurationCommand.EDIT_CHECKPOINT:
            self._screen.action_edit_model_details(
                reopen_route_on_close=nav_routes.CONFIGURATION_DIALOG,
                reopen_opener_on_close=reopen_checkpoint_details,
                open_details_directly=True,
                selected_session_name=selection.session_name,
                selected_checkpoint_name=selection.checkpoint_name,
                expected_model_uuid=selection.expected_model_uuid,
            )
