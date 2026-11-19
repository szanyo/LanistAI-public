# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate home edit flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lanista.gui.contracts import UiNotification
from lanista.reinforcengine.application_errors import OperationResult
from lanista.reinforcengine.rl_model.administration_contracts import ModelAdministrationOwner
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.metadata.edit_service import (
    MetadataEditOutcome,
    MetadataEditRequest,
    MetadataEditSnapshot,
)
from lanista.reinforcengine.rl_model.runtime.active_selection_service import (
    ActiveRuntimeSelectionRequest,
)
from lanista.reinforcengine.rl_model.runtime.selection_service import (
    RuntimeSelectionUpdate,
)


class HomeEditPayloadView(Protocol):
    """Read-only model detail fields consumed by the edit workflow."""

    @property
    def model_name(self) -> str:
        """Return the model name exposed by this home edit payload view."""
        ...

    @property
    def model_uuid(self) -> str:
        """Return the model uuid exposed by this home edit payload view."""
        ...

    @property
    def model_version(self) -> str:
        """Return the model version exposed by this home edit payload view."""
        ...

    @property
    def model_description(self) -> str:
        """Return the model description exposed by this home edit payload view."""
        ...

    @property
    def framework_uuid(self) -> str:
        """Return the framework uuid exposed by this home edit payload view."""
        ...

    @property
    def current_session_uuid(self) -> str | None:
        """Return the current session uuid exposed by this home edit payload view."""
        ...

    @property
    def current_session_display_name(self) -> str | None:
        """Return the current session display name exposed by this home edit payload view."""
        ...

    @property
    def current_session_name(self) -> str:
        """Return the current session name exposed by this home edit payload view."""
        ...

    @property
    def current_session_description(self) -> str:
        """Return the current session description exposed by this home edit payload view."""
        ...


@dataclass(frozen=True, slots=True)
class HomeEditPayload:
    """Immutable snapshot retained while modal callbacks are outstanding."""

    model_name: str
    model_uuid: str
    model_version: str
    model_description: str
    framework_uuid: str
    current_session_uuid: str | None
    current_session_display_name: str | None
    current_session_name: str
    current_session_description: str

    @classmethod
    def from_view(cls, payload: HomeEditPayloadView) -> HomeEditPayload:
        """Copy a presentation DTO so later mutations cannot alter workflow state."""

        return cls(
            model_name=str(getattr(payload, "model_name", "") or ""),
            model_uuid=str(getattr(payload, "model_uuid", "") or ""),
            model_version=str(getattr(payload, "model_version", "") or ""),
            model_description=str(getattr(payload, "model_description", "") or ""),
            framework_uuid=str(getattr(payload, "framework_uuid", "") or ""),
            current_session_uuid=getattr(payload, "current_session_uuid", None),
            current_session_display_name=getattr(
                payload,
                "current_session_display_name",
                None,
            ),
            current_session_name=str(getattr(payload, "current_session_name", "") or ""),
            current_session_description=str(
                getattr(payload, "current_session_description", "") or ""
            ),
        )


@dataclass(frozen=True, slots=True)
class HomeEditTargetContext:
    """Carry the runtime context for home edit target."""

    blocked_notification: UiNotification | None
    payload: HomeEditPayload | None = None
    target_session_name: str | None = None
    target_checkpoint_name: str | None = None

    @property
    def blocked(self) -> bool:
        """Return the blocked exposed by this home edit target context."""
        return self.blocked_notification is not None


@dataclass(frozen=True, slots=True)
class HomeEditTargetSelectorContext:
    """Carry the runtime context for home edit target selector."""

    loaded_model_name: str | None
    model_uuid: str | None
    session_options: tuple[tuple[str, str], ...]
    selected_session_name: str | None
    checkpoint_options: tuple[tuple[str, str], ...]
    selected_checkpoint_name: str | None


@dataclass(frozen=True, slots=True)
class HomeEditDetailDialogContext:
    """Carry the runtime context for home edit detail dialog."""

    model_name: str
    model_uuid: str
    model_version: str
    framework_uuid: str
    model_description: str
    session_uuid: str | None
    session_name: str
    session_description: str
    checkpoint_name: str | None
    checkpoint_annotation: str
    snapshot: MetadataEditSnapshot | None
    snapshot_failure: UiNotification | None = None


@dataclass(frozen=True, slots=True)
class HomeEditSaveResult:
    """Report the result of home edit save."""

    success: bool
    notification: UiNotification
    model_display_name: str | None = None
    framework_uuid: str | None = None
    requires_reload: bool = False
    storage_recovery_required: bool = False


class HomeEditFlowService:
    """Decision and persistence workflow helpers for HomeScreen edit dialogs."""

    def __init__(self, services: ApplicationServices) -> None:
        """Initialize the home edit flow service and its owned state."""
        self._services = services

    def build_target_context(
        self,
        *,
        is_critical_phase: bool,
        selected_session_name: str | None,
        selected_checkpoint_name: str | None,
        control_panel_session_name: str | None,
        expected_model_uuid: str | None = None,
    ) -> HomeEditTargetContext:
        """Build the target context required by this home edit flow service."""
        if is_critical_phase:
            return HomeEditTargetContext(
                blocked_notification=UiNotification(
                    message="Edit is disabled while training is running.",
                    title="Edit blocked",
                    severity="warning",
                )
            )

        normalized_model_uuid = str(expected_model_uuid or "").strip()
        if normalized_model_uuid and not (
            self._services.administration.models.is_owner_current(
                ModelAdministrationOwner(normalized_model_uuid)
            )
        ):
            return HomeEditTargetContext(
                blocked_notification=UiNotification(
                    message="The active model changed before editing started.",
                    title="Edit expired",
                    severity="warning",
                )
            )

        runtime_selection = self._services.runtime.context.get_current_runtime_selection_state()
        target_session_name = (
            selected_session_name
            or control_panel_session_name
            or runtime_selection.session_name
            or self._services.catalog.session_checkpoints.get_latest_session_for_current_model()
        )
        target_checkpoint_name = selected_checkpoint_name
        if target_checkpoint_name is None and target_session_name == runtime_selection.session_name:
            target_checkpoint_name = runtime_selection.checkpoint_name

        payload = self._services.inspection.presentation.build_model_detail_payload(
            target_session_name
        )
        if payload is None:
            return HomeEditTargetContext(
                blocked_notification=UiNotification(
                    message="Load a model before editing metadata.",
                    title="Action blocked",
                    severity="warning",
                ),
                target_session_name=target_session_name,
                target_checkpoint_name=target_checkpoint_name,
            )

        resolved_model_uuid = str(getattr(payload, "model_uuid", "") or "").strip()
        if normalized_model_uuid and resolved_model_uuid != normalized_model_uuid:
            return HomeEditTargetContext(
                blocked_notification=UiNotification(
                    message="The active model changed before editing started.",
                    title="Edit expired",
                    severity="warning",
                ),
                target_session_name=target_session_name,
                target_checkpoint_name=target_checkpoint_name,
            )

        return HomeEditTargetContext(
            blocked_notification=None,
            payload=HomeEditPayload.from_view(payload),
            target_session_name=target_session_name,
            target_checkpoint_name=target_checkpoint_name,
        )

    def build_target_selector_context(
        self,
        *,
        payload: HomeEditPayloadView,
        preferred_session_name: str | None,
        preferred_checkpoint_name: str | None,
        fallback_session_name: str | None,
    ) -> HomeEditTargetSelectorContext:
        """Build the target selector context required by this home edit flow service."""
        runtime_selection = self._services.runtime.context.get_current_runtime_selection_state()
        framework_uuid = self._services.runtime.context.get_current_framework_uuid()
        active_session_name = (
            preferred_session_name if preferred_session_name is not None else fallback_session_name
        )

        session_options = tuple(
            self._services.sessions.selection.list_session_options_for_current_model(framework_uuid)
        )
        valid_session_values = {value for _, value in session_options}
        if active_session_name not in valid_session_values:
            active_session_name = session_options[0][1] if session_options else None

        checkpoint_options = tuple(
            self._services.sessions.selection.list_checkpoint_options_for_current_model(
                active_session_name,
                framework_uuid=framework_uuid,
            )
        )
        valid_checkpoint_values = {value for _, value in checkpoint_options}
        active_checkpoint_name = preferred_checkpoint_name
        if active_checkpoint_name is None and active_session_name == runtime_selection.session_name:
            active_checkpoint_name = runtime_selection.checkpoint_name
        if active_checkpoint_name and active_checkpoint_name not in valid_checkpoint_values:
            active_checkpoint_name = None

        return HomeEditTargetSelectorContext(
            loaded_model_name=(
                self._services.inspection.presentation.get_current_model_display_name()
            ),
            model_uuid=getattr(payload, "model_uuid", None),
            session_options=session_options,
            selected_session_name=active_session_name,
            checkpoint_options=checkpoint_options,
            selected_checkpoint_name=active_checkpoint_name,
        )

    def select_edit_target(
        self,
        *,
        session_name: str | None,
        checkpoint_name: str | None,
        expected_model_uuid: str,
    ) -> OperationResult[RuntimeSelectionUpdate]:
        """Select the edit target owned by this home edit flow service."""
        return self._services.runtime.active_selection.set_selection(
            ActiveRuntimeSelectionRequest(
                session_name=session_name,
                checkpoint_name=checkpoint_name,
                persist=True,
                expected_model_uuid=expected_model_uuid,
            )
        )

    def build_detail_dialog_context(
        self,
        *,
        payload: HomeEditPayloadView,
        target_checkpoint_name: str | None,
    ) -> HomeEditDetailDialogContext:
        """Build the detail dialog context required by this home edit flow service."""
        requested_session_name = getattr(payload, "current_session_uuid", None)
        context_result = self._services.administration.metadata_edits.capture_edit_context(
            session_name=requested_session_name,
            checkpoint_name=target_checkpoint_name,
            expected_model_uuid=str(getattr(payload, "model_uuid", "") or ""),
        )
        snapshot_failure = None
        edit_context = context_result.value
        if not context_result.succeeded or edit_context is None:
            message = (
                context_result.failure.message
                if context_result.failure is not None
                else "Failed to capture a safe metadata edit snapshot."
            )
            snapshot_failure = UiNotification(
                message=message,
                title="Edit unavailable",
                severity="error",
            )
        return HomeEditDetailDialogContext(
            model_name=(
                edit_context.model_name
                if edit_context is not None
                else getattr(payload, "model_name", "")
            ),
            model_uuid=(
                edit_context.snapshot.model_uuid
                if edit_context is not None
                else getattr(payload, "model_uuid", "")
            ),
            model_version=(
                edit_context.model_version
                if edit_context is not None
                else getattr(payload, "model_version", "")
            ),
            framework_uuid=getattr(payload, "framework_uuid", ""),
            model_description=(
                edit_context.model_description
                if edit_context is not None
                else getattr(payload, "model_description", "")
            ),
            session_uuid=(
                edit_context.session_name if edit_context is not None else requested_session_name
            ),
            session_name=(
                edit_context.session_display_name
                if edit_context is not None
                else getattr(payload, "current_session_name", "")
            ),
            session_description=(
                edit_context.session_description
                if edit_context is not None
                else getattr(payload, "current_session_description", "")
            ),
            checkpoint_name=(
                edit_context.checkpoint_name if edit_context is not None else target_checkpoint_name
            ),
            checkpoint_annotation=(
                edit_context.checkpoint_annotation if edit_context is not None else ""
            ),
            snapshot=(edit_context.snapshot if edit_context is not None else None),
            snapshot_failure=snapshot_failure,
        )

    def save_detail_updates(
        self,
        *,
        payload: HomeEditPayloadView,
        model_name: str | None,
        model_version: str | None,
        model_description: str | None,
        session_uuid: str | None,
        session_name: str | None,
        session_description: str | None,
        checkpoint_name: str | None,
        checkpoint_annotation: str | None,
        snapshot: MetadataEditSnapshot,
    ) -> HomeEditSaveResult:
        """Persist the detail updates owned by this home edit flow service."""
        save_result = self._services.administration.metadata_edits.save(
            MetadataEditRequest(
                expected_model_uuid=snapshot.model_uuid,
                expected_model_name=snapshot.model_name,
                expected_model_version=snapshot.model_version,
                snapshot=snapshot,
                model_name=model_name,
                model_version=model_version,
                model_description=model_description,
                session_name=session_uuid,
                session_display_name=session_name,
                session_description=session_description,
                checkpoint_name=checkpoint_name,
                checkpoint_annotation=checkpoint_annotation,
            )
        )
        if not save_result.succeeded or save_result.value is None:
            message = (
                save_result.failure.message
                if save_result.failure is not None
                else "Failed to save metadata changes."
            )
            return HomeEditSaveResult(
                success=False,
                notification=UiNotification(
                    message=message,
                    title="Save failed",
                    severity="error",
                ),
            )

        return self._build_successful_save_result(
            payload=payload,
            outcome=save_result.value,
        )

    def _build_successful_save_result(
        self,
        *,
        payload: HomeEditPayloadView,
        outcome: MetadataEditOutcome,
    ) -> HomeEditSaveResult:
        """Build the successful save result required by this home edit flow service."""
        projection_synchronized = outcome.projection_synchronized
        storage_cleanup_synchronized = outcome.storage_cleanup_synchronized
        requires_reload = not projection_synchronized or not storage_cleanup_synchronized
        notification = self._successful_save_notification(
            projection_synchronized=projection_synchronized,
            storage_cleanup_synchronized=storage_cleanup_synchronized,
        )
        return HomeEditSaveResult(
            success=True,
            notification=notification,
            model_display_name=self._services.catalog.creation.build_model_display_name(
                outcome.model_name,
                outcome.model_version,
            ),
            framework_uuid=getattr(payload, "framework_uuid", "") or "-",
            requires_reload=requires_reload,
            storage_recovery_required=not storage_cleanup_synchronized,
        )

    @staticmethod
    def _successful_save_notification(
        *,
        projection_synchronized: bool,
        storage_cleanup_synchronized: bool,
    ) -> UiNotification:
        """Return the successful save notification value owned by this home edit flow service."""
        if not storage_cleanup_synchronized:
            return UiNotification(
                message=(
                    "Metadata changes were committed durably, but storage cleanup "
                    "or lock release did not complete. Reload the model and run "
                    "storage recovery before further administration."
                ),
                title="Saved; recovery required",
                severity="warning",
            )
        if not projection_synchronized:
            return UiNotification(
                message=(
                    "Metadata changes saved. Reload the active model before "
                    "continuing because the in-memory view could not be updated."
                ),
                title="Saved; reload required",
                severity="warning",
            )
        return UiNotification(
            message="Metadata changes saved.",
            title="Saved",
            severity="information",
        )
