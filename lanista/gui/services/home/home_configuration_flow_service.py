# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate home configuration flow."""

from __future__ import annotations

from dataclasses import dataclass

from lanista.gui.contracts import UiNotification, normalize_ui_severity
from lanista.reinforcengine.application_errors import ApplicationError
from lanista.reinforcengine.rl_model.administration_contracts import ModelAdministrationOwner
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.model import RLModelRunningModes
from lanista.reinforcengine.rl_model.runtime.active_selection_service import (
    ActiveRuntimeSelectionRequest,
)
from lanista.reinforcengine.rl_model.runtime.context_service import RuntimeSelectionState


@dataclass(frozen=True)
class SessionSelectorContext:
    """Carry the runtime context for session selector."""

    blocked_notification: UiNotification | None
    owner: ModelAdministrationOwner | None = None
    current_session_uuid: str | None = None
    session_options: tuple[tuple[str, str], ...] = ()

    @property
    def blocked(self) -> bool:
        """Return the blocked exposed by this session selector context."""
        return self.blocked_notification is not None


@dataclass(frozen=True)
class CheckpointSelectorContext:
    """Carry the runtime context for checkpoint selector."""

    blocked_notification: UiNotification | None
    owner: ModelAdministrationOwner | None = None
    session_uuid: str | None = None
    selected_checkpoint_name: str | None = None
    checkpoint_options: tuple[tuple[str, str], ...] = ()
    allow_fresh_training: bool = False
    fresh_selection_label: str = "Start fresh (no checkpoint)"
    fresh_selection_mode_label: str = "Start fresh"
    fresh_selection_search_text: str = "start fresh no checkpoint fresh training"
    allow_delete: bool = True
    allow_rename: bool = False
    allow_edit: bool = False

    @property
    def blocked(self) -> bool:
        """Return the blocked exposed by this checkpoint selector context."""
        return self.blocked_notification is not None


@dataclass(frozen=True)
class ConfigurationApplyPlan:
    """Describe the executable plan for configuration apply."""

    blocked_by_loading: bool = False
    refresh_control_panel: bool = False
    import_required: bool = False
    notification: UiNotification | None = None
    selection_log_message: str | None = None
    result_log_message: str | None = None
    staged_manifest: ModelManifest | None = None
    selected_framework_uuid: str | None = None
    import_model_name: str = "selected model"


class HomeConfigurationFlowService:
    """Pure-ish configuration dialog workflow decisions for HomeScreen."""

    def __init__(self, services: ApplicationServices) -> None:
        """Initialize the home configuration flow service and its owned state."""
        self._services = services

    def _capture_administration_owner(
        self,
        *,
        blocked_title: str,
    ) -> tuple[ModelAdministrationOwner | None, UiNotification | None]:
        """Freeze the model identity before a selector can outlive its opener."""

        try:
            return self._services.administration.models.capture_owner(), None
        except ApplicationError as error:
            return (
                None,
                UiNotification(
                    message=str(error),
                    title=blocked_title,
                    severity="error",
                ),
            )

    @staticmethod
    def _selection_log_message(
        selected_mode: RLModelRunningModes,
        selected_framework_uuid: str | None,
        runtime_selection: RuntimeSelectionState,
    ) -> str:
        """Return the selection log message value owned by this home configuration flow service."""
        return (
            "Configuration confirm selection: "
            f"mode={getattr(selected_mode, 'value', selected_mode)}, "
            f"framework={selected_framework_uuid or '-'}, "
            f"session={runtime_selection.session_name or '-'}, "
            f"checkpoint={runtime_selection.checkpoint_name or '-'}"
        )

    @staticmethod
    def _manifest_model_name(staged_manifest: ModelManifest) -> str:
        """Return the manifest model name value owned by this home configuration flow service."""
        return str(
            getattr(staged_manifest.get_configuration(), "model_name", "") or "selected model"
        )

    def _build_import_plan(
        self,
        *,
        staged_manifest: ModelManifest,
        selected_framework_uuid: str | None,
        selection_log_message: str,
        result_log_message: str | None = None,
    ) -> ConfigurationApplyPlan:
        """Build the import plan required by this home configuration flow service."""
        return ConfigurationApplyPlan(
            import_required=True,
            selection_log_message=selection_log_message,
            result_log_message=result_log_message,
            staged_manifest=staged_manifest,
            selected_framework_uuid=selected_framework_uuid,
            import_model_name=self._manifest_model_name(staged_manifest),
        )

    def _build_loaded_runtime_plan(
        self,
        *,
        staged_manifest: ModelManifest,
        selected_framework_uuid: str | None,
        runtime_selection: RuntimeSelectionState,
        selection_log_message: str,
    ) -> ConfigurationApplyPlan:
        """Build the loaded runtime plan required by this home configuration flow service."""
        same_loaded_model = self._services.runtime.status.is_same_loaded_model(
            staged_manifest,
            framework_uuid=selected_framework_uuid,
        )
        if (
            same_loaded_model
            and self._services.runtime.context.is_current_runtime_selection_applied(
                session_name=runtime_selection.session_name,
                checkpoint_name=runtime_selection.checkpoint_name,
            )
        ):
            return ConfigurationApplyPlan(
                refresh_control_panel=True,
                selection_log_message=selection_log_message,
                result_log_message=(
                    "Configure confirm is a no-op: same model/session/checkpoint already active."
                ),
            )
        if same_loaded_model:
            return ConfigurationApplyPlan(
                refresh_control_panel=True,
                selection_log_message=selection_log_message,
                result_log_message=(
                    "Configure confirm reused loaded framework runtime; "
                    "only selection/context changed."
                ),
            )
        return self._build_import_plan(
            staged_manifest=staged_manifest,
            selected_framework_uuid=selected_framework_uuid,
            selection_log_message=selection_log_message,
        )

    def build_apply_plan(
        self,
        *,
        model_loading: bool,
        selected_mode: RLModelRunningModes,
        selected_framework_uuid: str | None,
    ) -> ConfigurationApplyPlan:
        """Build the apply plan required by this home configuration flow service."""
        if model_loading:
            return ConfigurationApplyPlan(blocked_by_loading=True)

        staged_manifest = self._services.runtime.staging.get_selected_manifest()
        runtime_selection = self._services.runtime.context.get_current_runtime_selection_state()

        if staged_manifest is None:
            return ConfigurationApplyPlan(refresh_control_panel=True)

        selection_log_message = self._selection_log_message(
            selected_mode,
            selected_framework_uuid,
            runtime_selection,
        )
        confirm_precheck = self._services.runtime.run_precheck.evaluate_configuration_confirm(
            selected_mode=selected_mode,
            selected_framework_uuid=selected_framework_uuid or "",
            selected_session_name=runtime_selection.session_name,
            selected_checkpoint_name=runtime_selection.checkpoint_name,
        )
        if not confirm_precheck.allowed:
            return ConfigurationApplyPlan(
                refresh_control_panel=True,
                selection_log_message=selection_log_message,
                notification=UiNotification(
                    message=confirm_precheck.message or "Configuration blocked.",
                    title=confirm_precheck.title,
                    severity=normalize_ui_severity(confirm_precheck.severity),
                ),
            )

        # A staged manifest only describes the requested model. It does not
        # mean that a framework runtime/model object has already been imported.
        if not self._services.runtime.status.has_loaded_runtime():
            return self._build_import_plan(
                staged_manifest=staged_manifest,
                selected_framework_uuid=selected_framework_uuid,
                selection_log_message=selection_log_message,
                result_log_message=(
                    "Configure confirm requires import: a model manifest is selected, "
                    "but no runtime model is loaded."
                ),
            )

        return self._build_loaded_runtime_plan(
            staged_manifest=staged_manifest,
            selected_framework_uuid=selected_framework_uuid,
            runtime_selection=runtime_selection,
            selection_log_message=selection_log_message,
        )

    def build_session_selector_context(self, *, is_critical_phase: bool) -> SessionSelectorContext:
        """Build the session selector context required by this home configuration flow service."""
        if is_critical_phase:
            return SessionSelectorContext(
                blocked_notification=UiNotification(
                    message="View is disabled while training is running.",
                    title="View blocked",
                    severity="warning",
                )
            )
        if self._services.runtime.staging.get_selected_manifest() is None:
            return SessionSelectorContext(
                blocked_notification=UiNotification(
                    message="Session selector requires a selected model.",
                    title="View blocked",
                    severity="warning",
                )
            )

        owner, owner_blocker = self._capture_administration_owner(
            blocked_title="Session selector blocked",
        )
        if owner_blocker is not None:
            return SessionSelectorContext(blocked_notification=owner_blocker)
        assert owner is not None

        runtime_selection = self._services.runtime.context.get_current_runtime_selection_state()
        current_session_uuid = (
            runtime_selection.session_name
            or self._services.catalog.session_checkpoints.get_latest_session_for_current_model()
        )
        session_options = tuple(
            (
                self._services.sessions.metadata.get_session_reference_display(session_uuid),
                session_uuid,
            )
            for session_uuid in (
                self._services.catalog.session_checkpoints.list_sessions_for_current_model()
            )
        )
        return SessionSelectorContext(
            blocked_notification=None,
            owner=owner,
            current_session_uuid=current_session_uuid,
            session_options=session_options,
        )

    def select_session(
        self,
        owner: ModelAdministrationOwner,
        session_uuid: str,
    ) -> tuple[bool, UiNotification]:
        """Select the session owned by this home configuration flow service."""
        replacement_checkpoint = (
            self._services.catalog.session_checkpoints.get_latest_checkpoint_for_session(
                session_uuid
            )
        )
        result = self._services.runtime.active_selection.set_selection(
            ActiveRuntimeSelectionRequest(
                session_name=session_uuid,
                checkpoint_name=replacement_checkpoint,
                persist=True,
                expected_model_uuid=owner.model_uuid,
            )
        )
        if not result.succeeded or result.value is None:
            return (
                False,
                UiNotification(
                    message=(
                        result.failure.message
                        if result.failure is not None
                        else "The session selection could not be saved."
                    ),
                    title="Session selection failed",
                    severity="error",
                ),
            )
        return (
            True,
            UiNotification(
                message=(
                    "Session selected: "
                    f"{self._services.sessions.metadata.get_session_reference_display(session_uuid)}"
                ),
                title="Session view",
                severity="information",
            ),
        )

    def delete_session(
        self,
        owner: ModelAdministrationOwner,
        session_uuid: str,
    ) -> tuple[bool, UiNotification]:
        """Delete the session through this home configuration flow service's safety boundary."""
        session_display = self._services.sessions.metadata.get_session_reference_display(
            session_uuid
        )
        result = self._services.administration.sessions.delete_session(
            owner,
            session_uuid,
        )
        if not result.succeeded or result.value is None:
            message = (
                result.failure.message
                if result.failure is not None
                else "Failed to delete session."
            )
            return (
                False,
                UiNotification(
                    message=message,
                    title="Delete failed",
                    severity="error",
                ),
            )
        if result.value.degraded:
            return (
                True,
                UiNotification(
                    message=(
                        f"Session deleted: {session_display}. "
                        "Reload the active model before continuing."
                    ),
                    title="Session deleted; reload required",
                    severity="warning",
                ),
            )
        return (
            True,
            UiNotification(
                message=f"Session deleted: {session_display}",
                title="Session deleted",
                severity="information",
            ),
        )

    def build_checkpoint_selector_context(
        self, *, is_critical_phase: bool
    ) -> CheckpointSelectorContext:
        """Build the checkpoint selector context required by this home configuration flow service.
        """
        if is_critical_phase:
            return CheckpointSelectorContext(
                blocked_notification=UiNotification(
                    message="Checkpoint selection is disabled while training is running.",
                    title="View blocked",
                    severity="warning",
                )
            )
        if self._services.runtime.staging.get_selected_manifest() is None:
            return CheckpointSelectorContext(
                blocked_notification=UiNotification(
                    message="Checkpoint selector requires a selected model.",
                    title="View blocked",
                    severity="warning",
                )
            )

        owner, owner_blocker = self._capture_administration_owner(
            blocked_title="Checkpoint selector blocked",
        )
        if owner_blocker is not None:
            return CheckpointSelectorContext(blocked_notification=owner_blocker)
        assert owner is not None

        runtime_selection = self._services.runtime.context.get_current_runtime_selection_state()
        session_uuid = (
            runtime_selection.session_name
            or self._services.catalog.session_checkpoints.get_latest_session_for_current_model()
        )
        if not session_uuid:
            return CheckpointSelectorContext(
                blocked_notification=UiNotification(
                    message="Select or create a session before choosing a checkpoint.",
                    title="Checkpoint blocked",
                    severity="warning",
                )
            )

        framework_uuid = self._services.runtime.context.get_current_framework_uuid()
        checkpoint_options = tuple(
            self._services.sessions.selection.list_checkpoint_options_for_current_model(
                session_uuid,
                framework_uuid=framework_uuid,
            )
        )
        return CheckpointSelectorContext(
            blocked_notification=None,
            owner=owner,
            session_uuid=session_uuid,
            selected_checkpoint_name=runtime_selection.checkpoint_name,
            checkpoint_options=checkpoint_options,
        )

    def build_checkpoint_dialog_context(
        self,
        *,
        is_critical_phase: bool,
        selected_running_mode: RLModelRunningModes,
    ) -> CheckpointSelectorContext:
        """Build the checkpoint dialog context required by this home configuration flow service."""
        context = self.build_checkpoint_selector_context(is_critical_phase=is_critical_phase)
        if context.blocked:
            return context

        fresh_selection_label = (
            "Constant-speed diagnostic (no checkpoint)"
            if selected_running_mode == RLModelRunningModes.EVALUATION
            else "Start fresh (no checkpoint)"
        )
        fresh_selection_mode_label = (
            "Constant-speed diagnostic"
            if selected_running_mode == RLModelRunningModes.EVALUATION
            else "Start fresh"
        )
        fresh_selection_search_text = (
            "constant speed diagnostic no checkpoint evaluation"
            if selected_running_mode == RLModelRunningModes.EVALUATION
            else "start fresh no checkpoint fresh training"
        )
        return CheckpointSelectorContext(
            blocked_notification=context.blocked_notification,
            owner=context.owner,
            session_uuid=context.session_uuid,
            selected_checkpoint_name=context.selected_checkpoint_name,
            checkpoint_options=context.checkpoint_options,
            allow_fresh_training=selected_running_mode != RLModelRunningModes.INFERENCE,
            fresh_selection_label=fresh_selection_label,
            fresh_selection_mode_label=fresh_selection_mode_label,
            fresh_selection_search_text=fresh_selection_search_text,
            allow_delete=True,
            allow_rename=False,
            allow_edit=True,
        )

    def select_checkpoint(
        self,
        owner: ModelAdministrationOwner,
        *,
        session_uuid: str,
        checkpoint_name: str | None,
    ) -> tuple[bool, UiNotification]:
        """Select the checkpoint owned by this home configuration flow service."""
        result = self._services.runtime.active_selection.set_selection(
            ActiveRuntimeSelectionRequest(
                session_name=session_uuid,
                checkpoint_name=checkpoint_name,
                persist=True,
                expected_model_uuid=owner.model_uuid,
            )
        )
        if not result.succeeded or result.value is None:
            return (
                False,
                UiNotification(
                    message=(
                        result.failure.message
                        if result.failure is not None
                        else "The checkpoint selection could not be saved."
                    ),
                    title="Checkpoint selection failed",
                    severity="error",
                ),
            )
        return (
            True,
            UiNotification(
                message=f"Checkpoint selected: {checkpoint_name or 'fresh training'}",
                title="Checkpoint view",
                severity="information",
            ),
        )

    def delete_checkpoint(
        self,
        owner: ModelAdministrationOwner,
        *,
        session_uuid: str,
        checkpoint_name: str,
    ) -> tuple[bool, UiNotification]:
        """Delete the checkpoint through this home configuration flow service's safety boundary."""
        result = self._services.administration.checkpoints.delete_checkpoint(
            owner,
            session_uuid,
            checkpoint_name,
        )
        if not result.succeeded or result.value is None:
            message = (
                result.failure.message
                if result.failure is not None
                else "Failed to delete checkpoint."
            )
            return (
                False,
                UiNotification(
                    message=message,
                    title="Delete failed",
                    severity="error",
                ),
            )
        if result.value.degraded:
            return (
                True,
                UiNotification(
                    message=(
                        f"Checkpoint deleted: {result.value.checkpoint_name}. "
                        "Reload the active model before continuing."
                    ),
                    title="Checkpoint deleted; reload required",
                    severity="warning",
                ),
            )
        return (
            True,
            UiNotification(
                message=f"Checkpoint deleted: {result.value.checkpoint_name}",
                title="Checkpoint deleted",
                severity="information",
            ),
        )

    def rename_checkpoint(
        self,
        owner: ModelAdministrationOwner,
        *,
        session_uuid: str,
        checkpoint_name: str,
        renamed_checkpoint_name: str,
    ) -> tuple[bool, UiNotification]:
        """Rename the checkpoint owned by this home configuration flow service."""
        result = self._services.administration.checkpoints.rename_checkpoint(
            owner,
            session_uuid,
            checkpoint_name,
            renamed_checkpoint_name,
        )
        if not result.succeeded or result.value is None:
            message = (
                result.failure.message
                if result.failure is not None
                else "Failed to rename checkpoint."
            )
            return (
                False,
                UiNotification(
                    message=message,
                    title="Rename failed",
                    severity="error",
                ),
            )
        if result.value.degraded:
            return (
                True,
                UiNotification(
                    message=(
                        f"Checkpoint renamed to: {result.value.checkpoint_name}. "
                        "Reload the active model before continuing."
                    ),
                    title="Checkpoint renamed; reload required",
                    severity="warning",
                ),
            )
        return (
            True,
            UiNotification(
                message=f"Checkpoint renamed to: {result.value.checkpoint_name}",
                title="Checkpoint renamed",
                severity="information",
            ),
        )
