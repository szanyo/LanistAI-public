# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate home new session flow."""

from __future__ import annotations

from dataclasses import dataclass

from lanista.gui.contracts import UiNotification, normalize_ui_severity
from lanista.reinforcengine.application_errors import ApplicationError
from lanista.reinforcengine.rl_model.administration_contracts import ModelAdministrationOwner
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.model import RLModelRunningModes


@dataclass(frozen=True)
class HomeNewSessionDialogContext:
    """Carry the runtime context for home new session dialog."""

    blocked_notification: UiNotification | None
    current_session_uuid: str | None = None
    current_session_display_name: str | None = None
    framework_uuid: str | None = None
    expected_model_uuid: str | None = None
    blocked_by_loading: bool = False

    @property
    def blocked(self) -> bool:
        """Return the blocked exposed by this home new session dialog context."""
        return self.blocked_notification is not None or self.blocked_by_loading


@dataclass(frozen=True)
class HomeNewSessionCreateResult:
    """Report the result of home new session create."""

    session_name: str | None = None
    operation_id: str | None = None
    refresh_control_panel: bool = False
    reopen_session_parent: bool = False
    notification: UiNotification | None = None


class HomeNewSessionFlowService:
    """Decision helpers for HomeScreen new-session orchestration."""

    def __init__(self, services: ApplicationServices) -> None:
        """Initialize the home new session flow service and its owned state."""
        self._services = services

    @staticmethod
    def _normalize_framework_uuid(framework_uuid: str | None) -> str | None:
        """Normalize the framework uuid accepted by this home new session flow service."""
        return str(framework_uuid or "").strip().lower() or None

    def _build_available_session_context(
        self,
        *,
        control_panel_session_uuid: str | None,
        control_panel_framework_uuid: str | None,
        expected_model_uuid: str | None,
    ) -> HomeNewSessionDialogContext:
        """Build the available session context required by this home new session flow service."""
        runtime_context = self._services.runtime.context
        framework_uuid = self._normalize_framework_uuid(
            control_panel_framework_uuid
        ) or self._normalize_framework_uuid(
            runtime_context.get_current_framework_uuid()
        )
        if framework_uuid is None:
            return HomeNewSessionDialogContext(
                blocked_notification=None,
                framework_uuid=None,
                expected_model_uuid=expected_model_uuid,
            )

        try:
            session_uuids = list(
                self._services.catalog.session_checkpoints.list_sessions_for_current_model(
                    framework_uuid,
                )
            )
            valid_session_uuids = set(session_uuids)
            current_session_uuid: str | None
            if control_panel_session_uuid in valid_session_uuids:
                current_session_uuid = control_panel_session_uuid
            else:
                runtime_session_candidate = (
                    runtime_context.get_selected_session_candidate()
                )
                current_session_uuid = (
                    runtime_session_candidate
                    if runtime_session_candidate in valid_session_uuids
                    else (session_uuids[0] if session_uuids else None)
                )
            current_session_display = (
                self._services.sessions.metadata.get_session_reference_display(
                    current_session_uuid
                )
                if current_session_uuid
                else None
            )
        except (ApplicationError, OSError) as error:
            safe_message = str(error).strip() if isinstance(error, ApplicationError) else ""
            return HomeNewSessionDialogContext(
                blocked_notification=UiNotification(
                    message=(
                        safe_message
                        or "The available session metadata could not be loaded."
                    ),
                    title="Session metadata unavailable",
                    severity="error",
                ),
                framework_uuid=framework_uuid,
                expected_model_uuid=expected_model_uuid,
            )

        return HomeNewSessionDialogContext(
            blocked_notification=None,
            current_session_uuid=current_session_uuid,
            current_session_display_name=current_session_display,
            framework_uuid=framework_uuid,
            expected_model_uuid=expected_model_uuid,
        )

    def build_dialog_context(
        self,
        *,
        model_loading: bool,
        selected_running_mode: RLModelRunningModes,
        control_panel_session_uuid: str | None,
        control_panel_framework_uuid: str | None,
        expected_model_uuid: str | None = None,
    ) -> HomeNewSessionDialogContext:
        """Build the dialog context required by this home new session flow service."""
        if model_loading:
            return HomeNewSessionDialogContext(
                blocked_notification=None,
                framework_uuid=control_panel_framework_uuid,
                expected_model_uuid=expected_model_uuid,
                blocked_by_loading=True,
            )

        normalized_model_uuid = str(expected_model_uuid or "").strip()
        if normalized_model_uuid and not self._services.administration.models.is_owner_current(
            ModelAdministrationOwner(normalized_model_uuid)
        ):
            return HomeNewSessionDialogContext(
                blocked_notification=UiNotification(
                    message="The active model changed before session creation started.",
                    title="New session expired",
                    severity="warning",
                ),
                framework_uuid=control_panel_framework_uuid,
                expected_model_uuid=normalized_model_uuid,
            )

        precheck = self._services.runtime.run_precheck.evaluate_create_session(
            selected_running_mode
        )
        if not precheck.allowed:
            return HomeNewSessionDialogContext(
                blocked_notification=UiNotification(
                    message=precheck.message or "Action blocked.",
                    title=precheck.title,
                    severity=normalize_ui_severity(precheck.severity),
                ),
                framework_uuid=control_panel_framework_uuid,
                expected_model_uuid=normalized_model_uuid or None,
            )

        return self._build_available_session_context(
            control_panel_session_uuid=control_panel_session_uuid,
            control_panel_framework_uuid=control_panel_framework_uuid,
            expected_model_uuid=normalized_model_uuid or None,
        )

    def build_cancel_result(self, *, reopen_startup_dialog: bool) -> HomeNewSessionCreateResult:
        """Build the cancel result required by this home new session flow service."""
        return HomeNewSessionCreateResult(reopen_session_parent=reopen_startup_dialog)

    def create_session(
        self,
        *,
        strategy: str,
        parent_session_uuid: str | None,
        parent_checkpoint_name: str | None,
        framework_uuid: str | None,
        session_display_name: str | None,
        session_description: str | None,
        reopen_startup_dialog: bool,
        expected_model_uuid: str | None = None,
    ) -> HomeNewSessionCreateResult:
        """Create the session owned by this home new session flow service."""
        runtime_selection = self._services.runtime.context.get_current_runtime_selection_state()
        creation = self._services.sessions.creation.create_new_session_for_current_model(
            strategy=strategy,
            parent_session_uuid=parent_session_uuid,
            parent_checkpoint_name=parent_checkpoint_name,
            framework_uuid=framework_uuid,
            session_display_name=session_display_name,
            session_description=session_description,
            selected_session_name=runtime_selection.session_name,
            expected_model_uuid=expected_model_uuid,
        )
        if not creation.succeeded:
            assert creation.failure is not None
            return HomeNewSessionCreateResult(
                operation_id=creation.operation_id,
                reopen_session_parent=reopen_startup_dialog,
                notification=UiNotification(
                    message=creation.failure.message,
                    title="Session creation failed",
                    severity="error",
                ),
            )
        outcome = creation.value
        assert outcome is not None
        session_display = self._services.sessions.metadata.get_session_reference_display(
            outcome.session_name
        )
        strategy_suffix = " (fresh)" if strategy == "fresh" else " (fork)"
        if outcome.degraded:
            notification = UiNotification(
                message=(
                    f"Session storage was created as {session_display}, but "
                    "runtime reload or storage recovery is required."
                ),
                title="Session created with recovery required",
                severity="warning",
            )
        else:
            notification = UiNotification(
                message=(
                    f"New session created: {session_display}{strategy_suffix}. "
                    "Press Start to begin training."
                ),
                title="Session created",
                severity="information",
            )
        return HomeNewSessionCreateResult(
            session_name=outcome.session_name,
            operation_id=creation.operation_id,
            refresh_control_panel=True,
            # Session creation is already confirmed in NewSessionStrategyDialog.
            # Returning to Configure after success only repeats the validation/UI
            # pass. Cancel and failure paths still reopen the parent dialog.
            reopen_session_parent=False,
            notification=notification,
        )
