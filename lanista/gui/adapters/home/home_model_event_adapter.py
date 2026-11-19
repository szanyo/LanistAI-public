# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Apply model-runtime events to the Home screen through narrow UI ports."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from lanista.gui.adapters.model_runtime_events import ModelChangedMessage
from lanista.gui.contracts import UiNotification
from lanista.gui.services.home.home_model_event_presentation_service import (
    HomeModelEventPresentationService,
    HomeModelFailedPresentationPlan,
    HomeModelLoadedPresentationPlan,
    HomeModelLoadingPresentationPlan,
)
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.execution_contracts import RLModelState
from lanista.reinforcengine.rl_model.model import RLModelRunningModes


class _ControlPanelPort(Protocol):
    """Define the typed boundary for control panel."""

    def set_actions_locked(self, locked: bool) -> None:
        """Set the actions locked owned by this control panel port."""
        ...

    def set_main_status(
        self,
        activity: str,
        step: str,
        progress: float,
        total: float | None,
    ) -> None:
        """Set the main status owned by this control panel port."""
        ...

    def refresh_runtime_context_from_model(self) -> None:
        """Refresh the runtime context from model owned by this control
panel port."""
        ...


class _InfoPanelPort(Protocol):
    """Define the typed boundary for info panel."""

    def set_model_info(
        self,
        model_name: str | None,
        framework_uuid: str | None,
    ) -> None:
        """Set the model info owned by this info panel port."""
        ...

    def set_runtime_info(
        self,
        mode: str,
        session_name: str | None,
        parent_session_name: str | None,
        checkpoint_name: str | None,
        tensorboard_status: str,
        session_created_at_raw: str | None,
    ) -> None:
        """Set the runtime info owned by this info panel port."""
        ...


class _RuntimeContextEvent(Protocol):
    """Define the typed boundary for runtime context event."""

    mode: RLModelRunningModes
    session_name: str | None
    parent_session_name: str | None
    checkpoint_name: str | None
    tensorboard_status: str
    session_created_at_raw: str | None


class HomeModelEventAdapter:
    """Own Home's deterministic projection of runtime slot events."""

    def __init__(
        self,
        *,
        services: ApplicationServices,
        presentation: HomeModelEventPresentationService,
        control_panel_provider: Callable[[], _ControlPanelPort],
        info_panel_provider: Callable[[], _InfoPanelPort],
        model_loading_provider: Callable[[], bool],
        model_loading_committer: Callable[[bool], None],
        startup_autoload_provider: Callable[[], bool],
        stop_tensorboard: Callable[[], None],
        notify_ui: Callable[[UiNotification], None],
    ) -> None:
        """Initialize the home model event adapter and its owned state."""
        self._services = services
        self._presentation = presentation
        self._control_panel_provider = control_panel_provider
        self._info_panel_provider = info_panel_provider
        self._model_loading_provider = model_loading_provider
        self._model_loading_committer = model_loading_committer
        self._startup_autoload_provider = startup_autoload_provider
        self._stop_tensorboard = stop_tensorboard
        self._notify_ui = notify_ui

    def handle(self, event: ModelChangedMessage) -> None:
        """Project one model state without exposing manager ownership to the GUI."""

        control_panel = self._control_panel_provider()
        info_panel = self._info_panel_provider()
        if event.state is RLModelState.LOADING:
            self._apply_loading(control_panel, info_panel)
            return
        if event.state is RLModelState.FAILED:
            self._apply_failure(
                control_panel,
                info_panel,
                active_model_preserved=event.model is not None,
            )
            return
        if event.model is not None:
            self._apply_loaded(control_panel, info_panel)
            return
        self._apply_unavailable(control_panel, info_panel)

    def handle_runtime_context(self, event: _RuntimeContextEvent) -> None:
        """Refresh runtime labels from typed facades without manager access."""

        info_panel = self._info_panel_provider()
        has_model_context = (
            self._services.runtime.staging.get_selected_manifest() is not None
            or self._services.runtime.status.has_loaded_runtime()
        )
        model_name = (
            self._services.inspection.presentation.get_current_model_display_name()
            if has_model_context
            else None
        )
        framework_uuid = (
            self._services.runtime.context.get_current_framework_uuid()
            if has_model_context
            else None
        )
        info_panel.set_model_info(model_name, framework_uuid or "-")
        info_panel.set_runtime_info(
            event.mode.value if has_model_context else "-",
            event.session_name,
            event.parent_session_name,
            event.checkpoint_name,
            event.tensorboard_status,
            event.session_created_at_raw,
        )

    def _apply_loading(
        self,
        control_panel: _ControlPanelPort,
        info_panel: _InfoPanelPort,
    ) -> None:
        """Apply the loading transition owned by this home model event adapter."""
        self._model_loading_committer(True)
        plan = self._presentation.build_loading_plan()
        control_panel.set_actions_locked(plan.actions_locked)
        self._set_main_status(control_panel, plan)
        info_panel.set_model_info(plan.model_name, plan.framework_uuid)
        self._notify_ui(plan.notification)

    def _apply_failure(
        self,
        control_panel: _ControlPanelPort,
        info_panel: _InfoPanelPort,
        *,
        active_model_preserved: bool,
    ) -> None:
        """Apply the failure transition owned by this home model event adapter."""
        was_loading = self._model_loading_provider()
        self._model_loading_committer(False)
        plan = self._presentation.build_failed_plan(
            was_loading=was_loading,
            startup_autoload_in_progress=self._startup_autoload_provider(),
            active_model_preserved=active_model_preserved,
        )
        self._apply_failure_plan(
            plan,
            control_panel,
            info_panel,
            active_model_preserved=active_model_preserved,
        )

    def _apply_unavailable(
        self,
        control_panel: _ControlPanelPort,
        info_panel: _InfoPanelPort,
    ) -> None:
        """Apply the unavailable transition owned by this home model event adapter."""
        was_loading = self._model_loading_provider()
        self._model_loading_committer(False)
        plan = self._presentation.build_failed_plan(
            was_loading=was_loading,
            startup_autoload_in_progress=self._startup_autoload_provider(),
        )
        self._apply_failure_plan(
            plan,
            control_panel,
            info_panel,
            active_model_preserved=False,
        )

    def _apply_failure_plan(
        self,
        plan: HomeModelFailedPresentationPlan,
        control_panel: _ControlPanelPort,
        info_panel: _InfoPanelPort,
        *,
        active_model_preserved: bool,
    ) -> None:
        """Apply the failure plan transition owned by this home model event adapter."""
        control_panel.set_actions_locked(plan.actions_locked)
        if plan.clear_info:
            info_panel.set_model_info(None, None)
            info_panel.set_runtime_info(*plan.runtime_info)
        elif active_model_preserved:
            info_panel.set_model_info(
                self._services.inspection.presentation.get_current_model_display_name(),
                self._services.runtime.context.get_current_framework_uuid() or "-",
            )
        if plan.clear_last_successful_model_name:
            self._services.runtime.selection.clear_last_successful_runtime_state()
        self._set_main_status(control_panel, plan)
        if plan.refresh_runtime_context:
            control_panel.refresh_runtime_context_from_model()
        if plan.stop_tensorboard:
            self._stop_tensorboard()
        if plan.notification is not None:
            self._notify_ui(plan.notification)

    def _apply_loaded(
        self,
        control_panel: _ControlPanelPort,
        info_panel: _InfoPanelPort,
    ) -> None:
        """Apply the loaded transition owned by this home model event adapter."""
        self._model_loading_committer(False)
        plan = self._presentation.build_loaded_plan()
        control_panel.set_actions_locked(plan.actions_locked)
        info_panel.set_model_info(plan.model_name, plan.framework_uuid or "-")
        self._set_main_status(control_panel, plan)
        if plan.refresh_runtime_context:
            control_panel.refresh_runtime_context_from_model()
        self._notify_ui(
            UiNotification(
                message=f"Model loaded: {plan.model_name}",
                title=plan.notification.title,
                severity=plan.notification.severity,
            )
        )
        if plan.warning_notification is not None:
            self._notify_ui(plan.warning_notification)

    @staticmethod
    def _set_main_status(
        control_panel: _ControlPanelPort,
        plan: (
            HomeModelLoadingPresentationPlan
            | HomeModelLoadedPresentationPlan
            | HomeModelFailedPresentationPlan
        ),
    ) -> None:
        """Set the main status owned by this home model event adapter."""
        control_panel.set_main_status(
            plan.main_status_activity,
            plan.main_status_step,
            plan.main_status_progress,
            plan.main_status_total,
        )
