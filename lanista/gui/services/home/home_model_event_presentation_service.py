# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate home model event presentation."""

from __future__ import annotations

from dataclasses import dataclass, field

from lanista.gui.contracts import UiNotification
from lanista.reinforcengine.rl_model.application_services import ApplicationServices


@dataclass(frozen=True)
class HomeModelLoadingPresentationPlan:
    """Describe the executable plan for home model loading presentation."""

    actions_locked: bool = True
    model_name: str = "Loading model..."
    framework_uuid: str | None = None
    main_status_activity: str = "Loading model"
    main_status_step: str = "Starting"
    main_status_progress: int = 0
    main_status_total: int = 100
    notification: UiNotification = field(
        default_factory=lambda: UiNotification(
            message="Model loading started.",
            title="Model import",
            severity="information",
        )
    )


@dataclass(frozen=True)
class HomeModelLoadedPresentationPlan:
    """Describe the executable plan for home model loaded presentation."""

    actions_locked: bool = False
    model_name: str = ""
    framework_uuid: str = ""
    main_status_activity: str = ""
    main_status_step: str = ""
    main_status_progress: int = 100
    main_status_total: int | None = 100
    notification: UiNotification = field(
        default_factory=lambda: UiNotification(
            message="Model loaded.",
            title="Model import",
            severity="information",
        )
    )
    warning_notification: UiNotification | None = None
    refresh_runtime_context: bool = True


@dataclass(frozen=True)
class HomeModelFailedPresentationPlan:
    """Describe the executable plan for home model failed presentation."""

    actions_locked: bool = False
    clear_info: bool = True
    runtime_info: tuple[str, str | None, str | None, str | None, str, str | None] = (
        "-",
        None,
        None,
        None,
        "offline",
        None,
    )
    main_status_activity: str = "Idle"
    main_status_step: str = "-"
    main_status_progress: int = 0
    main_status_total: int | None = None
    refresh_runtime_context: bool = True
    stop_tensorboard: bool = True
    clear_last_successful_model_name: bool = False
    notification: UiNotification | None = None


class HomeModelEventPresentationService:
    """Build Home-screen presentation plans from typed application services."""

    def __init__(self, services: ApplicationServices) -> None:
        """Initialize the home model event presentation service and its owned state."""
        self._services = services

    def build_loading_plan(self) -> HomeModelLoadingPresentationPlan:
        """Build the loading plan required by this home model event presentation service."""
        return HomeModelLoadingPresentationPlan()

    def build_loaded_plan(self) -> HomeModelLoadedPresentationPlan:
        """Build the loaded plan required by this home model event presentation service."""
        model_name = (
            self._services.inspection.presentation.get_current_model_display_name()
            or "No model selected"
        )
        framework_uuid = self._services.runtime.context.get_current_framework_uuid() or ""
        if self._services.runtime.status.current_model_has_valid_runtime():
            ready_presentation = self._services.runtime.status.get_model_ready_presentation()
            if ready_presentation is not None:
                warning_notification = None
                if getattr(ready_presentation, "training_only_warning", False):
                    warning_notification = UiNotification(
                        message=(
                            "Model loaded, but evaluation is unavailable until valid session + checkpoint exists."
                        ),
                        title="Training-only model",
                        severity="warning",
                    )
                return HomeModelLoadedPresentationPlan(
                    model_name=model_name,
                    framework_uuid=framework_uuid,
                    main_status_activity=ready_presentation.activity,
                    main_status_step=ready_presentation.step,
                    warning_notification=warning_notification,
                )

        return HomeModelLoadedPresentationPlan(
            model_name=model_name,
            framework_uuid=framework_uuid or "-",
            main_status_activity="Invalid model runtime",
            main_status_step="Load a valid session/checkpoint from Configure",
            main_status_progress=0,
            main_status_total=None,
        )

    def build_failed_plan(
        self,
        *,
        was_loading: bool,
        startup_autoload_in_progress: bool,
        active_model_preserved: bool = False,
    ) -> HomeModelFailedPresentationPlan:
        """Build the failed plan required by this home model event presentation service."""
        load_issue = self._services.runtime.status.get_last_model_load_issue()
        if was_loading and startup_autoload_in_progress and not active_model_preserved:
            return HomeModelFailedPresentationPlan(
                clear_last_successful_model_name=True,
                notification=None,
            )

        notification = None
        if was_loading:
            notification = UiNotification(
                message=load_issue or "Model loading failed. Check logs for details.",
                title="Model import failed",
                severity="error",
            )

        return HomeModelFailedPresentationPlan(
            clear_info=not active_model_preserved,
            main_status_activity="Model import failed" if was_loading else "Idle",
            main_status_step=load_issue or "-",
            stop_tensorboard=not active_model_preserved,
            notification=notification,
        )
