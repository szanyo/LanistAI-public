# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate home model load flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lanista.gui.contracts import UiNotification
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest

type HomeModelLoadAction = Literal["edit", "load"]


@dataclass(frozen=True)
class HomeModelLoadDialogContext:
    """Carry the runtime context for home model load dialog."""

    blocked_notification: UiNotification | None
    current_model_uuid: str = ""
    blocked_by_loading: bool = False

    @property
    def blocked(self) -> bool:
        """Return the blocked exposed by this home model load dialog context."""
        return self.blocked_notification is not None or self.blocked_by_loading


@dataclass(frozen=True)
class HomeModelLoadResult:
    """Report the result of home model load."""

    canceled: bool = False
    open_edit_details: bool = False
    reopen_load_parent: bool = False
    reopen_configuration: bool = False
    notification: UiNotification | None = None


class HomeModelLoadFlowService:
    """Decision helpers for HomeScreen model load dialog orchestration."""

    def __init__(self, services: ApplicationServices) -> None:
        """Initialize the home model load flow service and its owned state."""
        self._services = services

    def build_dialog_context(
        self,
        *,
        is_critical_phase: bool,
        model_loading: bool,
    ) -> HomeModelLoadDialogContext:
        """Build the dialog context required by this home model load flow service."""
        if is_critical_phase:
            return HomeModelLoadDialogContext(
                blocked_notification=UiNotification(
                    message="Load is disabled while training is running.",
                    title="Load blocked",
                    severity="warning",
                )
            )

        if model_loading:
            return HomeModelLoadDialogContext(
                blocked_notification=None,
                blocked_by_loading=True,
            )

        current_manifest = self._services.runtime.staging.get_selected_manifest()
        current_model_uuid = (
            str(getattr(current_manifest.get_configuration(), "model_uuid", "") or "").strip()
            if current_manifest is not None
            else ""
        )
        return HomeModelLoadDialogContext(
            blocked_notification=None,
            current_model_uuid=current_model_uuid,
        )

    def handle_dialog_result(
        self,
        *,
        action: HomeModelLoadAction | None,
        model_manifest: ModelManifest | None,
        reopen_startup_dialog: bool,
    ) -> HomeModelLoadResult:
        """Handle the dialog result workflow for this home model load flow service."""
        if model_manifest is None:
            return HomeModelLoadResult(
                canceled=True,
                reopen_load_parent=reopen_startup_dialog,
            )

        self._services.runtime.staging.stage_manifest(model_manifest)
        if action == "edit":
            return HomeModelLoadResult(open_edit_details=True)

        configuration = model_manifest.get_configuration()
        display_name = self._services.catalog.creation.build_model_display_name(
            str(getattr(configuration, "model_name", "") or ""),
            str(getattr(configuration, "model_version", "") or ""),
        )
        return HomeModelLoadResult(
            reopen_configuration=True,
            notification=UiNotification(
                message=f"Model selected: {display_name}",
                title="Load model",
                severity="information",
            ),
        )
