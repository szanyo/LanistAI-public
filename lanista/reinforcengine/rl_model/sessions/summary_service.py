# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate training-session summaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from yaml import YAMLError

from lanista.reinforcengine.application_errors import (
    ApplicationValidationError,
    OperationResult,
    PersistenceError,
    ResourceNotFoundError,
    RetryPolicy,
    RuntimeUnavailableError,
    failure_from_exception,
)
from lanista.reinforcengine.rl_model.metadata.schema import (
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.sessions.repository_contracts import (
    SessionManifestSummary,
)
from lanista.reinforcengine.rl_model.storage.names import InvalidStorageName, SessionStorageName
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.storage.persistence import read_yaml_mapping_file


class SessionSummaryManagerPort(Protocol):
    """Path boundary required by the summary reader."""

    def get_sessions_root(self) -> str:
        """Return the sessions root exposed by this session summary manager port."""
        ...

    def get_session_config_path(self, session_name: str) -> str:
        """Return the session config path exposed by this session summary manager port."""
        ...


class SessionSummaryLogger(Protocol):
    """Diagnostic boundary that keeps sensitive paths out of UI failures."""

    def exception(self, message: str) -> object:
        """Publish a exception record through this session summary logger."""
        ...


class SessionSummaryService:
    """Load immutable summaries only from canonical schema-v1 manifests."""

    def __init__(
        self,
        session_manager_provider: Callable[[], SessionSummaryManagerPort | None],
        *,
        logger: SessionSummaryLogger | None = None,
    ) -> None:
        """Initialize the session summary service and its owned state."""
        self._session_manager_provider = session_manager_provider
        self._logger = logger
        self._cache: dict[Path, tuple[int, SessionManifestSummary]] = {}

    @staticmethod
    def normalize_framework_uuid(framework_uuid: str | None) -> str:
        """Normalize the framework uuid accepted by this session summary service."""
        return str(framework_uuid or "").strip().lower()

    def _exception(self, message: str) -> None:
        """Publish a exception record through this session summary service."""
        if self._logger is None:
            return
        try:
            self._logger.exception(message)
        except Exception:
            # Diagnostics must never replace the correlated read result.
            return

    @staticmethod
    def _session_storage_name(session_name: str | None) -> SessionStorageName:
        """Return the session storage name value owned by this session summary service."""
        try:
            return SessionStorageName.parse(session_name or "")
        except InvalidStorageName as error:
            raise ApplicationValidationError(
                "Select a valid managed session."
            ) from error

    def _get_manifest_path(
        self,
        manager: SessionSummaryManagerPort,
        candidate: str,
    ) -> Path:
        """Return the manifest path exposed by this session summary service."""
        sessions_root = manager.get_sessions_root()
        return StoragePathGuard(sessions_root).require_descendant(
            candidate,
            must_exist=True,
        )

    def invalidate(self, session_name: str | None) -> None:
        """Invalidate the operation owned by this session summary service."""
        storage_name = self._session_storage_name(session_name)
        session_manager = self._session_manager_provider()
        if session_manager is None:
            return
        manifest_path = self._get_manifest_path(
            session_manager,
            session_manager.get_session_config_path(str(storage_name)),
        )
        self._cache.pop(manifest_path, None)

    def load(
        self,
        session_name: str | None,
    ) -> OperationResult[SessionManifestSummary]:
        """Load and return the state owned by this session summary service."""
        operation_id = uuid4().hex
        try:
            storage_name = self._session_storage_name(session_name)
        except ApplicationValidationError as error:
            return OperationResult.failed(
                failure_from_exception(error, operation_id=operation_id)
            )

        manager = self._session_manager_provider()
        if manager is None:
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

        return self._load_from_manager(manager, storage_name, operation_id)

    def _load_from_manager(
        self,
        manager: SessionSummaryManagerPort,
        storage_name: SessionStorageName,
        operation_id: str,
    ) -> OperationResult[SessionManifestSummary]:
        """Load the from manager owned by this session summary service."""
        manifest_path_hint = "<unresolved>"
        try:
            manifest_path_hint = manager.get_session_config_path(str(storage_name))
            manifest_path = self._get_manifest_path(manager, manifest_path_hint)
            modified_ns = manifest_path.stat().st_mtime_ns
            cached = self._cache.get(manifest_path)
            if cached is not None and cached[0] == modified_ns:
                return OperationResult.success(cached[1], operation_id=operation_id)

            payload = validate_versioned_metadata_payload(
                MetadataKind.SESSION_MANIFEST,
                read_yaml_mapping_file(str(manifest_path)),
            )
            summary = self._build_summary(storage_name, payload)
        except FileNotFoundError as error:
            self._exception(
                "Session summary resource disappeared "
                f"for session={storage_name!s}; operation_id={operation_id}; "
                f"path={manifest_path_hint!r}; error={error!r}."
            )
            missing_failure = ResourceNotFoundError(
                "The selected session no longer exists."
            )
            return OperationResult.failed(
                failure_from_exception(missing_failure, operation_id=operation_id)
            )
        except (
            OSError,
            StoragePathViolation,
            TypeError,
            ValueError,
            YAMLError,
        ) as error:
            self._exception(
                "Session summary read failed "
                f"for session={storage_name!s}; operation_id={operation_id}; "
                f"path={manifest_path_hint!r}; error={error!r}."
            )
            persistence_failure = PersistenceError(
                "The session summary is unreadable or uses an unsupported schema."
            )
            return OperationResult.failed(
                failure_from_exception(
                    persistence_failure,
                    operation_id=operation_id,
                    retry_policy=RetryPolicy.SAFE,
                )
            )
        except Exception as error:
            self._exception(
                "Unexpected session summary failure "
                f"for session={storage_name!s}; operation_id={operation_id}; "
                f"path={manifest_path_hint!r}; error={error!r}."
            )
            return OperationResult.failed(
                failure_from_exception(error, operation_id=operation_id)
            )

        self._cache[manifest_path] = (modified_ns, summary)
        return OperationResult.success(summary, operation_id=operation_id)

    @classmethod
    def _build_summary(
        cls,
        storage_name: SessionStorageName,
        payload: Mapping[str, object],
    ) -> SessionManifestSummary:
        """Build the summary required by this session summary service."""
        session_section = payload.get("session")
        if not isinstance(session_section, Mapping):
            raise ValueError("Session metadata is missing its canonical session section.")

        def pick(*keys: str) -> str:
            """Perform the pick step within the enclosing build summary workflow."""
            for key in keys:
                value = payload.get(key)
                if value is None:
                    value = session_section.get(key)
                text = str(value or "").strip()
                if text:
                    return text
            return ""

        session_version_text = pick("session_version")
        try:
            session_version = max(0, int(session_version_text or 0))
        except (TypeError, ValueError):
            session_version = 0

        return SessionManifestSummary(
            storage_name=str(storage_name),
            identity_uuid=pick("uuid"),
            display_name=pick("display_name"),
            session_name=pick("session_name", "name"),
            session_version=session_version,
            framework_uuid=cls.normalize_framework_uuid(pick("framework_uuid")) or None,
            parent_session_uuid=pick("parent_session_uuid") or None,
            description=pick("description"),
            created_at=pick("created_at"),
            updated_at=pick("updated_at", "modified_at"),
            status=pick("status"),
        )

    @staticmethod
    def reference_display(
        session_uuid: str,
        summary: SessionManifestSummary,
    ) -> str:
        """Return the reference display value owned by this session summary service."""
        display_name = summary.display_name
        identity_uuid = summary.identity_uuid or session_uuid
        if display_name:
            return f"{display_name} [{identity_uuid[:8]}]"

        if summary.session_name and summary.session_version > 0:
            return (
                f"{summary.session_name}.v{summary.session_version} "
                f"[{identity_uuid[:8]}]"
            )

        return session_uuid
