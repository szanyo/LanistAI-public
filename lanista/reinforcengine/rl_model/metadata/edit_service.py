# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Crash-recoverable application boundary for Home metadata edits."""

from __future__ import annotations

import sys
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from lanista.reinforcengine.application_errors import (
    ApplicationError,
    ApplicationValidationError,
    OperationConflict,
    OperationResult,
    PersistenceError,
    ResourceNotFoundError,
    RuntimeUnavailableError,
    failure_from_exception,
)
from lanista.reinforcengine.rl_model.checkpoints.catalog_contracts import (
    CheckpointCatalogQueryPort,
)
from lanista.reinforcengine.rl_model.metadata.edit_committer import (
    MetadataEditCommitter,
)
from lanista.reinforcengine.rl_model.metadata.edit_contracts import (
    ABSENT_METADATA_DIGEST,
    MetadataDegradedStateReporter,
    MetadataEditContext,
    MetadataEditLockFactory,
    MetadataEditOutcome,
    MetadataEditRequest,
    MetadataEditSnapshot,
    MetadataEditTransactionFactory,
)
from lanista.reinforcengine.rl_model.metadata.edit_preparer import (
    MetadataEditPreparer,
)
from lanista.reinforcengine.rl_model.metadata.edit_revision_service import (
    MetadataEditRevisionService,
)
from lanista.reinforcengine.rl_model.metadata.projection_service import (
    capture_projection,
    restore_projection,
    synchronize_model_projection,
)
from lanista.reinforcengine.rl_model.storage.mutation_lock import (
    StorageMutationLock,
    StorageMutationLockError,
    StorageMutationLockTimeout,
)
from lanista.reinforcengine.rl_model.storage.mutation_scope import (
    ModelsStorageMutationScope,
)
from lanista.reinforcengine.rl_model.storage.names import InvalidStorageName
from lanista.reinforcengine.rl_model.storage.path_guard import StoragePathViolation
from lanista.reinforcengine.rl_model.storage.transaction import (
    StorageTransaction,
    StorageTransactionError,
)

__all__ = [
    "ABSENT_METADATA_DIGEST",
    "MetadataDegradedStateReporter",
    "MetadataEditContext",
    "MetadataEditLockFactory",
    "MetadataEditOutcome",
    "MetadataEditRequest",
    "MetadataEditService",
    "MetadataEditSnapshot",
    "MetadataEditTransactionFactory",
]


class MetadataEditService:
    """Coordinate one metadata edit through focused application components."""

    def __init__(
        self,
        *,
        active_configuration_provider: Callable[[], object | None],
        models_root_provider: Callable[[], str],
        checkpoint_catalog_provider: Callable[[], CheckpointCatalogQueryPort],
        transaction_factory: MetadataEditTransactionFactory = StorageTransaction,
        lock_factory: MetadataEditLockFactory = StorageMutationLock,
        lock_timeout_seconds: float = 5.0,
        administration_operation_factory: Callable[
            [], AbstractContextManager[None]
        ] = nullcontext,
        degraded_state_reporter: MetadataDegradedStateReporter | None = None,
        logger: object | None = None,
    ) -> None:
        """Initialize the metadata edit service and its owned state."""
        self._active_configuration_provider = active_configuration_provider
        self._committer = MetadataEditCommitter(transaction_factory)
        self._revisions = MetadataEditRevisionService(
            models_root_provider=models_root_provider,
            checkpoint_catalog_provider=checkpoint_catalog_provider,
            degraded_state_reporter=degraded_state_reporter,
        )
        self._lock_factory = lock_factory
        self._lock_timeout_seconds = max(0.0, float(lock_timeout_seconds))
        self._administration_operation_factory = (
            administration_operation_factory
        )
        self._degraded_state_reporter = degraded_state_reporter
        self._logger = logger

    def save(
        self,
        request: MetadataEditRequest,
        *,
        operation_id: str | None = None,
    ) -> OperationResult[MetadataEditOutcome]:
        """Commit the edit or return a safe typed failure without partial writes."""

        correlation_id = operation_id or uuid4().hex
        try:
            with self._administration_operation_factory():
                storage_scope = self._revisions.capture_storage_scope()
                return self._save_under_administration(
                    request,
                    correlation_id,
                    storage_scope,
                )
        except Exception as exc:
            self._log_failure(correlation_id, exc)
            return OperationResult.failed(
                failure_from_exception(
                    self._classify_failure(exc),
                    operation_id=correlation_id,
                )
            )

    def _save_under_administration(
        self,
        request: MetadataEditRequest,
        correlation_id: str,
        storage_scope: ModelsStorageMutationScope,
    ) -> OperationResult[MetadataEditOutcome]:
        """Commit while the shared runtime/admin transition gate is held."""

        durably_committed = False
        outcome: MetadataEditOutcome | None = None
        try:
            configuration, model_root = self._active_context(storage_scope)
            with self._mutation_lock(storage_scope):
                configuration = self._revalidate_active_context(
                    configuration,
                    model_root,
                    storage_scope,
                )
                prepared = self._preparer_for_scope(storage_scope).prepare(
                    configuration,
                    model_root,
                    request,
                )
                cleanup_error = self._committer.commit(prepared)
                durably_committed = True
                projection_synchronized = self._synchronize_projection_safely(
                    prepared.configuration,
                    prepared.outcome,
                    correlation_id,
                )
                outcome = replace(
                    prepared.outcome,
                    projection_synchronized=projection_synchronized,
                    storage_cleanup_synchronized=(cleanup_error is None),
                )
                if cleanup_error is not None:
                    self._log_post_commit_failure(
                        correlation_id,
                        cleanup_error,
                    )
                self._record_degraded_state(outcome, correlation_id)
            return OperationResult.success(
                outcome,
                operation_id=correlation_id,
            )
        except Exception as exc:
            if durably_committed and outcome is not None:
                self._log_post_commit_failure(correlation_id, exc)
                degraded_outcome = replace(
                    outcome,
                    storage_cleanup_synchronized=False,
                )
                self._record_degraded_state(
                    degraded_outcome,
                    correlation_id,
                )
                return OperationResult.success(
                    degraded_outcome,
                    operation_id=correlation_id,
                )
            raise

    def capture_snapshot(
        self,
        *,
        expected_model_uuid: str,
        session_name: str | None,
        checkpoint_name: str | None,
        operation_id: str | None = None,
    ) -> OperationResult[MetadataEditSnapshot]:
        """Capture strict durable revisions for a later compare-and-swap save."""

        correlation_id = operation_id or uuid4().hex
        try:
            with self._administration_operation_factory():
                storage_scope = self._revisions.capture_storage_scope()
                with self._mutation_lock(storage_scope):
                    configuration, model_root = self._capture_context_for_model(
                        expected_model_uuid,
                        storage_scope,
                    )
                    snapshot = self._revisions.capture_snapshot(
                        configuration,
                        model_root,
                        session_name=session_name,
                        checkpoint_name=checkpoint_name,
                    )
            return OperationResult.success(
                snapshot,
                operation_id=correlation_id,
            )
        except Exception as exc:
            self._log_failure(correlation_id, exc)
            return OperationResult.failed(
                failure_from_exception(
                    self._classify_failure(exc),
                    operation_id=correlation_id,
                )
            )

    def capture_edit_context(
        self,
        *,
        expected_model_uuid: str,
        session_name: str | None,
        checkpoint_name: str | None,
        operation_id: str | None = None,
    ) -> OperationResult[MetadataEditContext]:
        """Read dialog values and their CAS revisions under the same lock."""

        correlation_id = operation_id or uuid4().hex
        try:
            with self._administration_operation_factory():
                storage_scope = self._revisions.capture_storage_scope()
                with self._mutation_lock(storage_scope):
                    configuration, model_root = self._capture_context_for_model(
                        expected_model_uuid,
                        storage_scope,
                    )
                    context = self._revisions.capture_context(
                        configuration,
                        model_root,
                        session_name=session_name,
                        checkpoint_name=checkpoint_name,
                    )
            return OperationResult.success(
                context,
                operation_id=correlation_id,
            )
        except Exception as exc:
            self._log_failure(correlation_id, exc)
            return OperationResult.failed(
                failure_from_exception(
                    self._classify_failure(exc),
                    operation_id=correlation_id,
                )
            )

    def _active_context(
        self,
        storage_scope: ModelsStorageMutationScope,
    ) -> tuple[object, Path]:
        """Return the active context value owned by this metadata edit service."""
        configuration = self._active_configuration_provider()
        if configuration is None:
            raise RuntimeUnavailableError("Load a model before editing metadata.")
        return configuration, self._revisions.resolve_model_root(
            configuration,
            storage_scope=storage_scope,
        )

    def _capture_context_for_model(
        self,
        expected_model_uuid: str,
        storage_scope: ModelsStorageMutationScope,
    ) -> tuple[object, Path]:
        """Resolve a capture target only after its model owner is validated."""

        expected_uuid = expected_model_uuid.strip()
        if not expected_uuid:
            raise OperationConflict(
                "The active model changed. Reopen the edit dialog."
            )
        configuration = self._active_configuration_provider()
        if configuration is None:
            raise RuntimeUnavailableError("Load a model before editing metadata.")
        current_uuid = str(
            getattr(configuration, "model_uuid", "") or ""
        ).strip()
        if not current_uuid or current_uuid != expected_uuid:
            raise OperationConflict(
                "The active model changed. Reopen the edit dialog."
            )
        return configuration, self._revisions.resolve_model_root(
            configuration,
            storage_scope=storage_scope,
        )

    def _mutation_lock(
        self,
        storage_scope: ModelsStorageMutationScope,
    ) -> StorageMutationLock:
        """Return the mutation lock value owned by this metadata edit service."""
        return storage_scope.lock(
            lock_factory=self._lock_factory,
            timeout_seconds=self._lock_timeout_seconds,
        )

    def _preparer_for_scope(
        self,
        storage_scope: ModelsStorageMutationScope,
    ) -> MetadataEditPreparer:
        """Return the preparer for scope value owned by this metadata edit service."""
        models_root = str(storage_scope.models_root)
        return MetadataEditPreparer(
            models_root_provider=lambda: models_root,
            revision_service=self._revisions,
        )

    def _revalidate_active_context(
        self,
        expected_configuration: object,
        expected_root: Path,
        storage_scope: ModelsStorageMutationScope,
    ) -> object:
        """Return the revalidate active context value owned by this metadata edit service."""
        configuration = self._active_configuration_provider()
        if configuration is None:
            raise OperationConflict(
                "The active model changed. Reopen the edit dialog."
            )
        current_root = self._revisions.resolve_model_root(
            configuration,
            storage_scope=storage_scope,
        )
        expected_uuid = str(
            getattr(expected_configuration, "model_uuid", "") or ""
        ).strip()
        current_uuid = str(
            getattr(configuration, "model_uuid", "") or ""
        ).strip()
        if (
            current_root != expected_root
            or not expected_uuid
            or current_uuid != expected_uuid
        ):
            raise OperationConflict(
                "The active model changed. Reopen the edit dialog."
            )
        return configuration

    def _synchronize_projection_safely(
        self,
        configuration: object,
        outcome: MetadataEditOutcome,
        operation_id: str,
    ) -> bool:
        """Return the synchronize projection safely value owned by this metadata edit service."""
        try:
            snapshot = capture_projection(configuration)
        except Exception as exc:
            self._log_projection_failure(operation_id, exc, restored=True)
            return False
        try:
            synchronize_model_projection(configuration, outcome)
        except Exception as exc:
            restored = True
            try:
                restore_projection(snapshot)
            except Exception as restore_exc:
                restored = False
                self._log_projection_failure(
                    operation_id,
                    restore_exc,
                    restored=False,
                )
            self._log_projection_failure(operation_id, exc, restored=restored)
            return False
        return True

    @staticmethod
    def _classify_failure(error: Exception) -> Exception:
        """Classify the failure owned by this metadata edit service."""
        if isinstance(error, ApplicationError):
            return error
        if isinstance(error, StorageMutationLockTimeout):
            return OperationConflict(
                "Another metadata update is in progress. Reopen the edit dialog."
            )
        if isinstance(error, InvalidStorageName | StoragePathViolation):
            return ApplicationValidationError(str(error))
        if isinstance(error, FileNotFoundError):
            return ResourceNotFoundError(
                "A metadata target no longer exists. Reopen the edit dialog."
            )
        if isinstance(
            error,
            StorageMutationLockError | StorageTransactionError | OSError,
        ):
            return PersistenceError(
                "Metadata could not be read or committed safely."
            )
        return error

    def _record_degraded_state(
        self,
        outcome: MetadataEditOutcome,
        operation_id: str,
    ) -> None:
        """Record the degraded state owned by this metadata edit service."""
        reload_required = not outcome.projection_synchronized
        storage_recovery_required = not outcome.storage_cleanup_synchronized
        if not reload_required and not storage_recovery_required:
            return
        if self._degraded_state_reporter is None:
            return
        try:
            self._degraded_state_reporter(
                reload_required=reload_required,
                storage_recovery_required=storage_recovery_required,
            )
        except Exception as exc:
            self._log_post_commit_failure(operation_id, exc)

    def _log_projection_failure(
        self,
        operation_id: str,
        error: BaseException,
        *,
        restored: bool,
    ) -> None:
        """Log the projection failure owned by this metadata edit service."""
        self._emit_log(
            "warning",
            "Metadata was committed but the active projection could not be "
            "synchronized (operation_id=%s, restored=%s): %s",
            operation_id,
            restored,
            error,
        )

    def _log_post_commit_failure(
        self,
        operation_id: str,
        error: BaseException,
    ) -> None:
        """Log the post commit failure owned by this metadata edit service."""
        self._emit_log(
            "warning",
            "Metadata commit is durable but cleanup or lock release requires "
            "recovery (operation_id=%s): %s",
            operation_id,
            error,
        )

    def _log_failure(self, operation_id: str, error: BaseException) -> None:
        """Log the failure owned by this metadata edit service."""
        self._emit_log(
            "exception",
            "Metadata edit failed (operation_id=%s).",
            operation_id,
            exc_info=error,
        )

    def _emit_log(
        self,
        level: str,
        message: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Emit the log owned by this metadata edit service."""
        if self._logger is None:
            return
        emitter = getattr(self._logger, level, None)
        if not callable(emitter):
            return
        try:
            emitter(message, *args, **kwargs)
        except Exception as exc:
            # Logging is an outbound diagnostic adapter. Its failure cannot
            # change a durable application's operation result.
            try:
                stderr = sys.__stderr__
                if stderr is None:
                    return
                stderr.write(
                    "Metadata diagnostic logging failed "
                    f"({type(exc).__name__}).\n"
                )
            except (AttributeError, OSError, ValueError):
                return
