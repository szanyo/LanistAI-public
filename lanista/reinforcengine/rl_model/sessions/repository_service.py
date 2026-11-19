# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate the training-session repository."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from typing import Protocol
from uuid import uuid4

from lanista.foundation.configuration import Configuration
from lanista.reinforcengine.application_errors import (
    ApplicationError,
    ApplicationValidationError,
    OperationResult,
    PersistenceError,
    ResourceNotFoundError,
    RetryPolicy,
    RuntimeUnavailableError,
    exception_from_failure,
    failure_from_exception,
)
from lanista.reinforcengine.rl_model.execution_contracts import RLModelState
from lanista.reinforcengine.rl_model.metadata.timestamps import utc_now_iso
from lanista.reinforcengine.rl_model.sessions.repository_contracts import (
    SessionManifestSummary,
    SessionMetadataUpdate,
    SessionMetadataUpdateOutcome,
)
from lanista.reinforcengine.rl_model.storage.layout import find_existing_session_manifest_path


class SessionManagerPort(Protocol):
    """Strict storage operations required by the repository facade."""

    def get_session_path(self, session_name: str) -> str:
        """Return the session path exposed by this session manager port."""
        ...

    def get_session_config_path(self, session_name: str) -> str:
        """Return the session config path exposed by this session manager port."""
        ...

    def load_session_configuration(self, session_name: str) -> Configuration:
        """Load the session configuration owned by this session manager port."""
        ...

    def update_session_metadata(
        self,
        session_name: str,
        update: SessionMetadataUpdate,
    ) -> None:
        """Update the session metadata owned by this session manager port."""
        ...


class SessionSummaryPort(Protocol):
    """Cache/projection operations required after repository access."""

    def load(
        self,
        session_name: str | None,
    ) -> OperationResult[SessionManifestSummary]:
        """Load and return the state owned by this session summary port."""
        ...

    def reference_display(
        self,
        session_uuid: str,
        summary: SessionManifestSummary,
    ) -> str:
        """Return the reference display value provided by this session summary port."""
        ...

    def invalidate(self, session_name: str | None) -> None:
        """Invalidate the operation owned by this session summary port."""
        ...


class SessionRepositoryLogger(Protocol):
    """Minimal structured logging surface used by the facade."""

    def warning(self, message: str) -> object:
        """Publish a warning record through this session repository logger."""
        ...

    def exception(self, message: str) -> object:
        """Publish a exception record through this session repository logger."""
        ...


class SessionRepositoryService:
    """Filesystem-backed session repository helpers for the active model."""

    def __init__(
        self,
        *,
        sessions_root_provider: Callable[[], str],
        session_manager_provider: Callable[[], SessionManagerPort | None],
        session_summary_service_provider: Callable[[], SessionSummaryPort],
        framework_normalizer: Callable[[str | None], str],
        logger: SessionRepositoryLogger | None = None,
    ) -> None:
        """Initialize the session repository service and its owned state."""
        self._sessions_root_provider = sessions_root_provider
        self._session_manager_provider = session_manager_provider
        self._session_summary_service_provider = session_summary_service_provider
        self._framework_normalizer = framework_normalizer
        self._logger = logger

    def _warning(self, message: str) -> None:
        """Publish a warning record through this session repository service."""
        if self._logger is not None:
            self._logger.warning(message)

    def _exception(self, message: str) -> None:
        """Publish a exception record through this session repository service."""
        if self._logger is not None:
            self._logger.exception(message)

    def _get_sessions_root(self) -> str:
        """Return the sessions root exposed by this session repository service."""
        return self._sessions_root_provider()

    def _build_training_session_manager(self) -> SessionManagerPort | None:
        """Build the training session manager required by this session repository service."""
        return self._session_manager_provider()

    def _get_session_summary_service(self) -> SessionSummaryPort:
        """Return the session summary service exposed by this session repository service."""
        return self._session_summary_service_provider()

    def _normalize_framework_uuid(self, framework_uuid: object) -> str:
        """Normalize the framework uuid accepted by this session repository service."""
        normalized_input = str(framework_uuid) if framework_uuid is not None else None
        return self._framework_normalizer(normalized_input)

    def get_session_path(self, session_name: str) -> str:
        """Return the session path exposed by this session repository service."""
        session_manager = self._build_training_session_manager()
        if session_manager is None:
            return ""
        return session_manager.get_session_path(session_name)

    def get_session_config_path(self, session_name: str) -> str:
        """Return the session config path exposed by this session repository service."""
        session_manager = self._build_training_session_manager()
        if session_manager is None:
            return ""
        return session_manager.get_session_config_path(session_name)

    def load_session_manifest_summary(
        self,
        session_name: str | None,
    ) -> OperationResult[SessionManifestSummary]:
        """Load the session manifest summary owned by this session repository service."""
        return self._get_session_summary_service().load(session_name)

    def _require_session_manifest_summary(
        self,
        session_name: str,
    ) -> SessionManifestSummary:
        """Return the required session manifest summary for this session repository service."""
        result = self.load_session_manifest_summary(session_name)
        if result.failure is not None:
            raise exception_from_failure(result.failure)
        if result.value is None:
            raise RuntimeUnavailableError("The session summary is unavailable.")
        return result.value

    def get_session_reference_display_from_summary(
        self,
        session_uuid: str,
        summary: SessionManifestSummary | None = None,
    ) -> str:
        """Return the session reference display from summary exposed by this session repository
        service.
        """
        resolved_summary = summary or self._require_session_manifest_summary(session_uuid)
        return self._get_session_summary_service().reference_display(
            session_uuid,
            resolved_summary,
        )

    def load_session_configuration(
        self,
        session_name: str,
    ) -> OperationResult[Configuration]:
        """Load the session configuration owned by this session repository service."""
        operation_id = uuid4().hex
        if not session_name.strip():
            validation_error = ApplicationValidationError(
                "A session name is required."
            )
            return OperationResult.failed(
                failure_from_exception(validation_error, operation_id=operation_id)
            )

        session_manager = self._build_training_session_manager()
        if session_manager is None:
            unavailable_error = RuntimeUnavailableError(
                "Session storage is not available."
            )
            return OperationResult.failed(
                failure_from_exception(
                    unavailable_error,
                    operation_id=operation_id,
                    retry_policy=RetryPolicy.SAFE,
                )
            )

        session_config_path = self.get_session_config_path(session_name)
        try:
            session_configuration = session_manager.load_session_configuration(session_name)
        except FileNotFoundError:
            failure_error: ApplicationError = ResourceNotFoundError(
                "The selected session no longer exists."
            )
            self._warning(
                f"Session configuration missing at {session_config_path!r}; "
                f"operation_id={operation_id}."
            )
            return OperationResult.failed(
                failure_from_exception(failure_error, operation_id=operation_id)
            )
        except (OSError, TypeError, ValueError) as error:
            failure_error = PersistenceError("The session configuration could not be loaded.")
            self._exception(
                f"Session configuration read failed at {session_config_path!r}; "
                f"operation_id={operation_id}; error={error!r}."
            )
            return OperationResult.failed(
                failure_from_exception(
                    failure_error,
                    operation_id=operation_id,
                    retry_policy=RetryPolicy.SAFE,
                )
            )
        except Exception as error:
            self._exception(
                f"Unexpected session configuration failure at {session_config_path!r}; "
                f"operation_id={operation_id}; error={error!r}."
            )
            return OperationResult.failed(
                failure_from_exception(error, operation_id=operation_id)
            )

        return OperationResult.success(session_configuration, operation_id=operation_id)

    def update_session_metadata(
        self,
        session_name: str | None,
        update: SessionMetadataUpdate,
    ) -> OperationResult[SessionMetadataUpdateOutcome]:
        """Update the session metadata owned by this session repository service."""
        operation_id = uuid4().hex
        if not session_name or not session_name.strip():
            validation_error = ApplicationValidationError(
                "A session name is required."
            )
            return OperationResult.failed(
                failure_from_exception(validation_error, operation_id=operation_id)
            )

        session_manager = self._build_training_session_manager()
        if session_manager is None:
            unavailable_error = RuntimeUnavailableError(
                "Session storage is not available."
            )
            return OperationResult.failed(
                failure_from_exception(
                    unavailable_error,
                    operation_id=operation_id,
                    retry_policy=RetryPolicy.SAFE,
                )
            )

        try:
            session_manager.update_session_metadata(session_name, update)
        except FileNotFoundError:
            failure_error: ApplicationError = ResourceNotFoundError(
                "The selected session no longer exists."
            )
        except (OSError, TypeError, ValueError) as error:
            failure_error = PersistenceError("The session metadata could not be saved.")
            self._exception(
                f"Session metadata write failed for {session_name!r}; "
                f"operation_id={operation_id}; error={error!r}."
            )
        except Exception as error:
            self._exception(
                f"Unexpected session metadata failure for {session_name!r}; "
                f"operation_id={operation_id}; error={error!r}."
            )
            return OperationResult.failed(
                failure_from_exception(error, operation_id=operation_id)
            )
        else:
            return self._finish_metadata_update(
                session_name,
                update,
                operation_id=operation_id,
            )

        return OperationResult.failed(
            failure_from_exception(
                failure_error,
                operation_id=operation_id,
                retry_policy=(
                    RetryPolicy.SAFE
                    if isinstance(failure_error, PersistenceError)
                    else RetryPolicy.NEVER
                ),
            )
        )

    def _finish_metadata_update(
        self,
        session_name: str,
        update: SessionMetadataUpdate,
        *,
        operation_id: str,
    ) -> OperationResult[SessionMetadataUpdateOutcome]:
        """Return the finish metadata update value owned by this session repository service."""
        summary_invalidated = True
        try:
            self._get_session_summary_service().invalidate(session_name)
        except Exception as error:
            summary_invalidated = False
            self._exception(
                f"Session metadata committed but summary invalidation failed for "
                f"{session_name!r}; operation_id={operation_id}; error={error!r}."
            )
        outcome = SessionMetadataUpdateOutcome(
            session_name=session_name,
            fields=update.field_names,
            summary_invalidated=summary_invalidated,
        )
        return OperationResult.success(outcome, operation_id=operation_id)

    def set_session_status(
        self,
        session_name: str | None,
        status: RLModelState,
    ) -> OperationResult[SessionMetadataUpdateOutcome]:
        """Set the session status owned by this session repository service."""
        return self.update_session_metadata(
            session_name,
            SessionMetadataUpdate.status(status, updated_at=utc_now_iso()),
        )

    def get_session_status(self, session_name: str | None) -> RLModelState | None:
        """Return the session status exposed by this session repository service."""
        if not session_name:
            return None

        load_result = self.load_session_configuration(session_name)
        session_configuration = load_result.value if load_result.succeeded else None
        if session_configuration is None:
            return None

        status_value = str(getattr(session_configuration, "status", "") or "").strip().lower()
        if not status_value:
            session_section = getattr(session_configuration, "session", None)
            if isinstance(session_section, dict):
                status_value = str(session_section.get("status", "") or "").strip().lower()
            elif session_section is not None:
                status_value = str(getattr(session_section, "status", "") or "").strip().lower()

        if not status_value:
            return None

        try:
            return RLModelState(status_value)
        except ValueError:
            return None

    def session_is_closed_for_training(self, session_name: str | None) -> bool:
        """Return the session is closed for training value owned by this session repository
        service.
        """
        return self.get_session_status(session_name) in (
            RLModelState.COMPLETED,
            RLModelState.FAILED,
            RLModelState.STOPPED,
        )

    def infer_framework_uuid_from_session_checkpoints(self, session_name: str) -> str:
        """Return the infer framework uuid from session checkpoints value owned by this session
        repository service.
        """
        try:
            checkpoints_root = os.path.join(self._get_sessions_root(), session_name, "checkpoints")
        except RuntimeError:
            return ""
        if not os.path.isdir(checkpoints_root):
            return ""

        has_directory_checkpoint = False
        has_zip_checkpoint = False
        stablebaselines_root = os.path.join(checkpoints_root, "stablebaselines")
        if os.path.isdir(stablebaselines_root) and os.listdir(stablebaselines_root):
            has_zip_checkpoint = True
        for entry in os.listdir(checkpoints_root):
            if entry == "stablebaselines":
                continue
            full_path = os.path.join(checkpoints_root, entry)
            if os.path.isdir(full_path):
                has_directory_checkpoint = True
            elif os.path.isfile(full_path) and str(entry).lower().endswith(".zip"):
                has_zip_checkpoint = True

        if has_directory_checkpoint and not has_zip_checkpoint:
            return "ray"
        if has_zip_checkpoint and not has_directory_checkpoint:
            return "sb3"
        return ""

    def get_session_framework_uuid(self, session_name: str) -> str:
        """Return the session framework uuid exposed by this session repository service."""
        summary = self._require_session_manifest_summary(session_name)
        return self._session_framework_uuid(session_name, summary)

    def list_sessions_for_current_model(self, framework_uuid: str | None = None) -> list[str]:
        """Return the sessions for current model available to this session repository service."""
        summaries = self.list_session_summaries_for_current_model()
        normalized_framework_uuid = self._normalize_framework_uuid(framework_uuid)
        if normalized_framework_uuid:
            summaries = tuple(
                summary
                for summary in summaries
                if self._session_framework_uuid(summary.storage_name, summary)
                == normalized_framework_uuid
            )
        return [summary.storage_name for summary in summaries]

    def list_session_summaries_for_current_model(
        self,
    ) -> tuple[SessionManifestSummary, ...]:
        """Return one ordered immutable read projection for the active model."""

        try:
            sessions_root = self._get_sessions_root()
        except RuntimeError:
            return ()
        if not os.path.isdir(sessions_root):
            return ()

        session_summaries = self._discover_session_summaries(sessions_root)
        ordered_names = sorted(
            session_summaries,
            key=lambda name: self._session_sort_key(
                sessions_root,
                name,
                session_summaries[name],
            ),
            reverse=True,
        )
        return tuple(session_summaries[name] for name in ordered_names)

    def _discover_session_summaries(
        self,
        sessions_root: str,
    ) -> dict[str, SessionManifestSummary]:
        """Return the discover session summaries value owned by this session repository service."""
        summaries: dict[str, SessionManifestSummary] = {}
        for entry in os.scandir(sessions_root):
            if not entry.is_dir():
                continue
            manifest_path = find_existing_session_manifest_path(entry.path)
            if not os.path.isfile(manifest_path):
                continue
            summaries[entry.name] = self._require_session_manifest_summary(entry.name)
        return summaries

    def _session_framework_uuid(
        self,
        session_name: str,
        summary: SessionManifestSummary,
    ) -> str:
        """Return the session framework uuid value owned by this session repository service."""
        framework_uuid = self._normalize_framework_uuid(summary.framework_uuid)
        if framework_uuid:
            return framework_uuid

        framework_uuid = self.infer_framework_uuid_from_session_checkpoints(
            session_name
        )
        if framework_uuid:
            update_result = self.update_session_metadata(
                session_name,
                SessionMetadataUpdate.framework(framework_uuid),
            )
            if update_result.failure is not None:
                raise exception_from_failure(update_result.failure)
        return framework_uuid

    @staticmethod
    def _session_sort_key(
        sessions_root: str,
        session_name: str,
        summary: SessionManifestSummary,
    ) -> tuple[float, float, str]:
        """Return the session sort key value owned by this session repository service."""
        created_at = summary.created_at
        timestamp_value = -1.0
        if created_at:
            try:
                timestamp_value = datetime.fromisoformat(
                    created_at.replace("Z", "+00:00")
                ).timestamp()
            except ValueError:
                timestamp_value = -1.0

        mtime = os.path.getmtime(os.path.join(sessions_root, session_name))
        return timestamp_value, mtime, session_name
