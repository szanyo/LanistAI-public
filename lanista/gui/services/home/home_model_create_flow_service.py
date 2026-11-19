# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate home model create flow."""

from __future__ import annotations

from dataclasses import dataclass

from lanista.gui.contracts import UiNotification, normalize_ui_severity
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.model import RLModelRunningModes


@dataclass(frozen=True)
class HomeModelCreateDialogContext:
    """Carry the runtime context for home model create dialog."""

    blocked_notification: UiNotification | None
    selected_framework_uuid: str
    blocked_by_loading: bool = False

    @property
    def blocked(self) -> bool:
        """Return the blocked exposed by this home model create dialog context."""
        return self.blocked_notification is not None or self.blocked_by_loading


@dataclass(frozen=True)
class HomeModelCreateResult:
    """Report the result of home model create."""

    canceled: bool = False
    reopen_creation_parent: bool = False
    refresh_control_panel: bool = False
    notification: UiNotification | None = None


class HomeModelCreateFlowService:
    """Decision helpers for HomeScreen model creation orchestration."""

    def __init__(self, services: ApplicationServices) -> None:
        """Initialize the home model create flow service and its owned state."""
        self._services = services

    def build_dialog_context(
        self,
        *,
        model_loading: bool,
        selected_running_mode: RLModelRunningModes,
        requested_framework_uuid: str | None,
        control_panel_framework_uuid: str | None,
    ) -> HomeModelCreateDialogContext:
        """Build the dialog context required by this home model create flow service."""
        if model_loading:
            return HomeModelCreateDialogContext(
                blocked_notification=None,
                selected_framework_uuid=str(requested_framework_uuid or control_panel_framework_uuid or ""),
                blocked_by_loading=True,
            )

        precheck = self._services.runtime.run_precheck.evaluate_create_model(
            selected_running_mode
        )
        if not precheck.allowed:
            return HomeModelCreateDialogContext(
                blocked_notification=UiNotification(
                    message=precheck.message or "Action blocked.",
                    title=precheck.title,
                    severity=normalize_ui_severity(precheck.severity),
                ),
                selected_framework_uuid=str(requested_framework_uuid or control_panel_framework_uuid or ""),
            )

        selected_framework_uuid = requested_framework_uuid or control_panel_framework_uuid
        return HomeModelCreateDialogContext(
            blocked_notification=None,
            selected_framework_uuid=str(selected_framework_uuid or ""),
        )

    def handle_creation_result(
        self,
        *,
        target_model_folder_path: str | None,
        target_model_name: str | None,
        creation_strategy: str,
        auto_load: bool,
        reopen_startup_dialog: bool,
    ) -> HomeModelCreateResult:
        """Handle the creation result workflow for this home model create flow service."""
        if not target_model_folder_path:
            return HomeModelCreateResult(
                canceled=True,
                reopen_creation_parent=reopen_startup_dialog,
            )
        assert target_model_name is not None

        manifest = self._services.catalog.models.load_model_manifest_from_folder(
            target_model_folder_path
        )
        if manifest is None:
            return HomeModelCreateResult(
                reopen_creation_parent=reopen_startup_dialog,
                notification=UiNotification(
                    message="Model was created, but it could not be loaded automatically.",
                    title="Create model",
                    severity="error",
                ),
            )

        display_name = self._services.catalog.creation.build_model_display_name(
            target_model_name,
            manifest.get_configuration().model_version,
        )
        if not auto_load:
            return HomeModelCreateResult(
                reopen_creation_parent=reopen_startup_dialog,
                notification=UiNotification(
                    message=f"{self._creation_text(creation_strategy, selected=False)}: {display_name}",
                    title=self._creation_title(creation_strategy),
                    severity="information",
                ),
            )

        self._services.runtime.staging.stage_manifest(manifest)
        return HomeModelCreateResult(
            reopen_creation_parent=True,
            refresh_control_panel=True,
            notification=UiNotification(
                message=f"{self._creation_text(creation_strategy, selected=True)}: {display_name}",
                title=self._creation_title(creation_strategy),
                severity="information",
            ),
        )

    @staticmethod
    def _creation_title(creation_strategy: str) -> str:
        """Return the creation title value owned by this home model create flow service."""
        return "Clone model" if creation_strategy == "clone" else "Create model"

    @staticmethod
    def _creation_text(creation_strategy: str, *, selected: bool) -> str:
        """Return the creation text value owned by this home model create flow service."""
        if creation_strategy == "clone":
            return "Model cloned and selected" if selected else "Cloned model created"
        return "Model created and selected" if selected else "Empty model scaffold created"
