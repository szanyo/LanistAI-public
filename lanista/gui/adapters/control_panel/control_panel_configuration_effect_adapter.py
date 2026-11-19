# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Own ControlPanel configuration-modal effects and result generations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Protocol

from lanista.gui.adapters.control_panel.control_panel_configuration_dialog_adapter import (
    ConfigurationScreenActionsPort,
    ControlPanelConfigurationCommandAdapter,
    ControlPanelConfigurationDialogFactory,
)
from lanista.gui.contracts import UiNotification
from lanista.gui.services.configuration.control_panel_configuration_contracts import (
    ConfigurationDialogResult,
    ConfigurationDialogSelection,
    ControlPanelConfigurationDialogPlan,
)
from lanista.gui.services.configuration.control_panel_configuration_flow_service import (
    ControlPanelConfigurationFlowService,
)
from lanista.gui.widgets.dialogs.configuration_dialog import RunConfigurationDialog
from lanista.reinforcengine.application_errors import FailureCode
from lanista.reinforcengine.rl_model.administration_contracts import ModelAdministrationOwner
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.model import BaseModelRuntime, RLModelRunningModes
from lanista.reinforcengine.rl_model.runtime.active_selection_service import (
    ActiveRuntimeSelectionRequest,
)
from lanista.reinforcengine.rl_model.runtime.configuration_state_service import (
    ConfigurationRuntimeState,
    ConfigurationSelectionPreference,
)
from lanista.reinforcengine.utils.scenarios.scenario_runtime import (
    discover_evaluation_scenario_inventory,
)


@dataclass(frozen=True, slots=True)
class ConfigurationDialogOwner:
    """Runtime identity and generation that own one modal callback."""

    generation: int
    runtime: BaseModelRuntime | None


class ControlPanelConfigurationViewPort(Protocol):
    """Minimal ControlPanel view surface used by configuration effects."""

    def configuration_effects_available(self) -> bool:
        """Return whether configuration effects available applies to this control panel
        configuration view port.
        """
        ...

    def refresh_configuration_runtime(self) -> None:
        """Refresh the configuration runtime owned by this control panel
configuration view port."""
        ...

    def current_configuration_runtime(self) -> BaseModelRuntime | None:
        """Return the current configuration runtime value provided by this control panel
        configuration view port.
        """
        ...

    def current_configuration_selection(
        self,
        preferred_mode: RLModelRunningModes | None,
    ) -> ConfigurationDialogSelection:
        """Return the current configuration selection value provided by this control panel
        configuration view port.
        """
        ...

    def apply_configuration_selection(
        self,
        selection: ConfigurationDialogSelection,
    ) -> None:
        """Apply the configuration selection transition owned by this control panel configuration
        view port.
        """
        ...

    def prepare_new_session_selection(
        self,
        selection: ConfigurationDialogSelection,
    ) -> None:
        """Prepare the new session selection required by this control panel configuration view
        port.
        """
        ...

    def configuration_screen_actions(self) -> ConfigurationScreenActionsPort:
        """Return the configuration screen actions value provided by this control panel
        configuration view port.
        """
        ...

    def push_configuration_dialog(
        self,
        dialog: RunConfigurationDialog,
        callback: Callable[[ConfigurationDialogResult | None], None],
    ) -> None:
        """Push the configuration dialog owned by this control panel configuration view port."""
        ...

    def notify_configuration(self, notification: UiNotification) -> None:
        """Notify the caller about the configuration outcome."""
        ...

    def publish_configuration_context(self) -> None:
        """Publish the configuration context owned by this control panel configuration view port."""
        ...

    def refresh_configuration_buttons(self) -> None:
        """Refresh the configuration buttons owned by this control panel
configuration view port."""
        ...

    def tensorboard_inspection_is_running(self) -> bool:
        """Return whether tensorboard inspection is running applies to this control panel
        configuration view port.
        """
        ...


class ControlPanelConfigurationEffectAdapter:
    """Execute modal effects without becoming a second selection reducer."""

    def __init__(
        self,
        *,
        view: ControlPanelConfigurationViewPort,
        services: ApplicationServices,
        framework_options: list[tuple[str, str]],
    ) -> None:
        """Initialize the control panel configuration effect adapter and its owned state."""
        self._view = view
        self._services = services
        self._framework_options = framework_options
        self._flow: ControlPanelConfigurationFlowService | None = None
        self._generation = 0
        self._owner: ConfigurationDialogOwner | None = None

    @property
    def owner(self) -> ConfigurationDialogOwner | None:
        """Return the active modal owner for diagnostics and tests."""

        return self._owner

    def close(self) -> None:
        """Invalidate callbacks captured by the closing widget lifetime."""

        self.invalidate()

    def invalidate(self) -> None:
        """Expire the active modal and every delayed reopen callback."""

        self._owner = None
        self._generation += 1

    def open(
        self,
        preferred_mode: RLModelRunningModes | None = None,
        *,
        selection: ConfigurationDialogSelection | None = None,
        auto_open_session_selector: bool = False,
        auto_open_checkpoint_selector: bool = False,
    ) -> None:
        """Build and push one generation-owned configuration modal."""

        if not self._view.configuration_effects_available() or self._owner is not None:
            return
        self._view.refresh_configuration_runtime()
        owner = self._claim()
        if owner is None:
            return
        try:
            requested = selection or self._view.current_configuration_selection(preferred_mode)
            preference = (
                ConfigurationSelectionPreference(
                    session_name=selection.session_name,
                    checkpoint_name=selection.checkpoint_name,
                )
                if selection is not None
                else None
            )
            runtime_state = self._services.runtime.configuration_state.build(
                preferred_mode=requested.mode,
                preferred_framework_uuid=requested.framework_uuid,
                framework_options=self._framework_options,
                selection_preference=preference,
            )
            scenario_options, scenario_index = self._evaluation_scenario_selection(
                owner.runtime,
                runtime_state=runtime_state,
                requested=requested,
            )
            dialog = ControlPanelConfigurationDialogFactory.build(
                runtime_state=runtime_state,
                framework_options=self._framework_options,
                inspect_is_running=(self._view.tensorboard_inspection_is_running()),
                application_services=self._services,
                auto_open_session_selector=auto_open_session_selector,
                auto_open_checkpoint_selector=auto_open_checkpoint_selector,
                selected_evaluation_visualization=(requested.evaluation_visualization),
                selected_evaluation_pacing=requested.evaluation_pacing,
                evaluation_scenario_options=scenario_options,
                selected_evaluation_scenario_index=scenario_index,
                selected_evaluation_library_scope=requested.evaluation_library_scope,
                selected_evaluation_target=requested.evaluation_target,
            )
            self._view.push_configuration_dialog(
                dialog,
                partial(self._handle_result, owner),
            )
        except Exception:
            self._release(owner)
            raise

    def _evaluation_scenario_selection(
        self,
        runtime: BaseModelRuntime | None,
        *,
        runtime_state: ConfigurationRuntimeState,
        requested: ConfigurationDialogSelection,
    ) -> tuple[list[tuple[str, int]], int | None]:
        """Resolve available scenarios and the checkpoint-backed Evaluation default."""

        if runtime_state.selected_mode is not RLModelRunningModes.EVALUATION:
            return [], None
        if runtime is not None:
            available_indices = runtime.get_available_evaluation_scenario_indices()
            configured_index = runtime.get_evaluation_scenario_index()
        else:
            staged_manifest = self._services.runtime.staging.get_selected_manifest()
            if staged_manifest is None:
                return [], None
            available_indices, configured_index = discover_evaluation_scenario_inventory(
                staged_manifest,
                runtime_state.selected_framework_uuid,
            )
        options = [(f"Scenario index {index}", index) for index in available_indices]
        if not options:
            return [], None
        available = set(available_indices)
        requested_index = requested.evaluation_scenario_index
        if requested_index in available:
            return options, requested_index

        detail = self._services.catalog.checkpoints.build_selection_detail(
            runtime_state.selected_session_name,
            runtime_state.selected_checkpoint_name,
        )
        if detail.scenario_index in available:
            return options, detail.scenario_index

        if configured_index in available:
            return options, configured_index
        return options, options[0][1]

    def _claim(self) -> ConfigurationDialogOwner | None:
        """Return the claim value owned by this control panel configuration effect adapter."""
        if not self._view.configuration_effects_available() or self._owner is not None:
            return None
        self._generation += 1
        owner = ConfigurationDialogOwner(
            generation=self._generation,
            runtime=self._view.current_configuration_runtime(),
        )
        self._owner = owner
        return owner

    def _release(self, owner: ConfigurationDialogOwner) -> None:
        """Release the operation owned by this control panel configuration effect adapter."""
        active = self._owner
        if active is not None and active.generation == owner.generation:
            self._owner = None

    def _complete(self, owner: ConfigurationDialogOwner) -> bool:
        """Return the complete value owned by this control panel configuration effect adapter."""
        active = self._owner
        if active is None or active.generation != owner.generation:
            return False
        self._owner = None
        return bool(
            self._view.configuration_effects_available()
            and owner.runtime is self._view.current_configuration_runtime()
        )

    def _handle_result(
        self,
        owner: ConfigurationDialogOwner,
        result: ConfigurationDialogResult | None,
    ) -> None:
        """Handle the result workflow for this control panel configuration effect adapter."""
        if not self._complete(owner):
            return
        if result is not None and not self._result_owner_is_current(result):
            return
        plan = self._get_flow().build_result_plan(result)
        selection = plan.selection
        if selection is None:
            return
        if plan.blocked_notification is not None:
            self._view.notify_configuration(plan.blocked_notification)
            return
        if plan.commit_runtime_selection and not self._commit(selection):
            return
        if plan.reopen_dialog:
            self.open(
                selection=selection,
                auto_open_session_selector=plan.auto_open_session_selector,
                auto_open_checkpoint_selector=plan.auto_open_checkpoint_selector,
            )
            return
        self._dispatch(owner, plan)
        if not plan.commit_runtime_selection:
            # Child workflows do not mutate the active runtime selection. Only
            # project selection state after this adapter committed it.
            return
        self._view.publish_configuration_context()
        self._view.refresh_configuration_buttons()

    def _get_flow(self) -> ControlPanelConfigurationFlowService:
        """Return the flow exposed by this control panel configuration effect adapter."""
        if self._flow is None:
            self._flow = ControlPanelConfigurationFlowService(self._services.runtime.run_precheck)
        return self._flow

    def _result_owner_is_current(self, result: ConfigurationDialogResult) -> bool:
        """Return whether result owner is current applies to this control panel configuration
        effect adapter.
        """
        expected_model_uuid = str(result.expected_model_uuid or "").strip()
        if not expected_model_uuid:
            return True
        return self._services.administration.models.is_owner_current(
            ModelAdministrationOwner(expected_model_uuid)
        )

    def _commit(self, selection: ConfigurationDialogSelection) -> bool:
        """Commit the operation owned by this control panel configuration effect adapter."""
        if selection.evaluation_library_scope is not None:
            try:
                self._services.evaluation_libraries.resolve_selection(
                    selection.evaluation_library_scope
                )
            except Exception as error:
                self._view.notify_configuration(
                    UiNotification(
                        message=str(error), title="Library selection expired", severity="error"
                    )
                )
                return False
        result = self._services.runtime.active_selection.set_selection(
            ActiveRuntimeSelectionRequest(
                session_name=selection.session_name,
                checkpoint_name=selection.checkpoint_name,
                framework_uuid=selection.framework_uuid,
                persist=True,
                expected_model_uuid=selection.expected_model_uuid,
            )
        )
        if not result.succeeded or result.value is None:
            failure = result.failure
            if failure is None or failure.code in {
                FailureCode.OPERATION_CONFLICT,
                FailureCode.RUNTIME_UNAVAILABLE,
            }:
                notification = UiNotification(
                    message=("The active model changed before configuration was applied."),
                    title="Configuration expired",
                    severity="warning",
                )
            else:
                notification = UiNotification(
                    message=failure.message,
                    title="Configuration failed",
                    severity="error",
                )
            self._view.notify_configuration(notification)
            return False
        self._view.apply_configuration_selection(selection)
        return True

    def _dispatch(
        self,
        owner: ConfigurationDialogOwner,
        plan: ControlPanelConfigurationDialogPlan,
    ) -> None:
        """Dispatch the operation owned by this control panel configuration effect adapter."""
        selection = plan.selection
        if selection is None:
            return
        adapter = ControlPanelConfigurationCommandAdapter(self._view.configuration_screen_actions())
        adapter.dispatch(
            plan,
            reopen_session_details=partial(
                self._reopen_details,
                owner,
                selection,
                auto_open_session_selector=True,
            ),
            reopen_checkpoint_details=partial(
                self._reopen_details,
                owner,
                selection,
                auto_open_checkpoint_selector=True,
            ),
            prepare_new_session=self._view.prepare_new_session_selection,
        )

    def _reopen_details(
        self,
        owner: ConfigurationDialogOwner,
        selection: ConfigurationDialogSelection,
        *,
        auto_open_session_selector: bool = False,
        auto_open_checkpoint_selector: bool = False,
    ) -> None:
        """Reopen the details owned by this control panel configuration effect adapter."""
        if (
            not self._view.configuration_effects_available()
            or owner.runtime is not self._view.current_configuration_runtime()
            or owner.generation != self._generation
        ):
            return
        self.open(
            selection=selection,
            auto_open_session_selector=auto_open_session_selector,
            auto_open_checkpoint_selector=auto_open_checkpoint_selector,
        )
