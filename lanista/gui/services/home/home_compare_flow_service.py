# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate home compare flow."""

from __future__ import annotations

from dataclasses import dataclass

from lanista.gui.contracts import UiNotification
from lanista.reinforcengine.rl_model.application_services import ApplicationServices


@dataclass(frozen=True)
class CompareBuilderContext:
    """Carry the runtime context for compare builder."""

    notification: UiNotification | None = None

    @property
    def blocked(self) -> bool:
        """Return the blocked exposed by this compare builder context."""
        return self.notification is not None


class HomeCompareFlowService:
    """Decision helpers for HomeScreen comparison-pack orchestration."""

    def __init__(self, services: ApplicationServices) -> None:
        """Initialize the home compare flow service and its owned state."""
        self._services = services

    def build_builder_context(self, *, is_critical_phase: bool) -> CompareBuilderContext:
        """Build the builder context required by this home compare flow service."""
        if is_critical_phase:
            return CompareBuilderContext(
                notification=UiNotification(
                    message="Compare pack creation is disabled while training is running.",
                    title="Compare blocked",
                    severity="warning",
                )
            )

        selected_manifest = self._services.runtime.staging.get_selected_manifest()
        structure_is_valid = (
            selected_manifest is not None
            and self._services.runtime.checklist.build_model_load_checklist(
                selected_manifest
            ).valid
        )
        if not structure_is_valid:
            return CompareBuilderContext(
                notification=UiNotification(
                    message="Compare pack creation requires a structurally valid selected model.",
                    title="Compare blocked",
                    severity="warning",
                )
            )

        return CompareBuilderContext()

    def build_result_notification(self, result: object | None) -> UiNotification | None:
        """Build the result notification required by this home compare flow service."""
        if result is None:
            return None

        action = getattr(result, "action", None)
        pack_name = str(getattr(result, "pack_name", "") or "")
        source_count = getattr(result, "source_count", 0)

        if action == "update_sources":
            message = f"Comparison pack sources updated: {pack_name} ({source_count} source logs)"
        elif action == "edit_description":
            message = f"Comparison pack description saved: {pack_name}"
        elif action == "rename":
            message = f"Comparison pack renamed: {pack_name}"
        elif action == "delete":
            message = f"Comparison pack deleted: {pack_name}"
        else:
            message = f"Comparison pack created: {pack_name} ({source_count} source logs)"

        return UiNotification(
            message=message,
            title="Compare",
            severity="information",
        )
