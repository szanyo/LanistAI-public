# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate home startup selection flow."""

from __future__ import annotations

from dataclasses import dataclass

from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest


@dataclass(frozen=True)
class HomeStartupSelectionPlan:
    """Describe the executable plan for home startup selection."""

    log_message: str
    selected_manifest: ModelManifest | None = None
    selected_framework_uuid: str | None = None
    refresh_control_panel: bool = False


class HomeStartupSelectionFlowService:
    """Decision helpers for HomeScreen startup model selection."""

    def __init__(self, services: ApplicationServices) -> None:
        """Initialize the home startup selection flow service and its owned state."""
        self._services = services

    def build_startup_selection_plan(self) -> HomeStartupSelectionPlan:
        """Build the startup selection plan required by this home startup selection flow service."""
        if self._services.runtime.status.has_loaded_runtime():
            return HomeStartupSelectionPlan(
                log_message="Startup check: active model already present.",
            )

        selection = self._services.runtime.selection
        selected_manifest = selection.get_last_successful_model_manifest()
        if selected_manifest is not None:
            return HomeStartupSelectionPlan(
                log_message="Startup check: previously successful model found, selecting it.",
                selected_manifest=selected_manifest,
                selected_framework_uuid=(
                    selection.get_last_successful_framework_uuid()
                    or selection.resolve_manifest_bootstrap_framework_uuid(
                        selected_manifest
                    )
                ),
                refresh_control_panel=True,
            )

        fallback_manifest = (
            self._services.catalog.models.get_latest_available_model_manifest()
        )
        if fallback_manifest is not None:
            return HomeStartupSelectionPlan(
                log_message="Startup check: no previous successful model, selecting latest valid model.",
                selected_manifest=fallback_manifest,
                selected_framework_uuid=(
                    selection.resolve_manifest_bootstrap_framework_uuid(
                        fallback_manifest
                    )
                ),
                refresh_control_panel=True,
            )

        return HomeStartupSelectionPlan(
            log_message="Startup check: no selectable model found.",
        )
