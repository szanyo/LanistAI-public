# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Model identity and durable model-storage deletion commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from lanista.reinforcengine.application_errors import (
    ApplicationError,
    ApplicationValidationError,
    OperationConflict,
    OperationResult,
    OperationTimeout,
    PersistenceError,
    ResourceNotFoundError,
    RetryPolicy,
    RuntimeUnavailableError,
    failure_from_exception,
)
from lanista.reinforcengine.rl_model.administration_command_policy import (
    AdministrationCommandPolicy,
)
from lanista.reinforcengine.rl_model.administration_contracts import (
    ModelAdministrationOwner,
    ModelDeletionOutcome,
)
from lanista.reinforcengine.rl_model.storage.mutation_lock import (
    StorageMutationLockError,
    StorageMutationLockTimeout,
)
from lanista.reinforcengine.rl_model.storage.mutation_scope import ModelsStorageMutationScope
from lanista.reinforcengine.rl_model.storage.names import InvalidStorageName
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.storage.transaction import (
    StorageTransaction,
    StorageTransactionError,
)


class ModelStorageCommandService:
    """Own active-model identity checks and model-root deletion."""

    def __init__(
        self,
        *,
        policy: AdministrationCommandPolicy,
        models_root_provider: Callable[[], str],
        clear_current_model_state_provider: Callable[[], None],
    ) -> None:
        """Initialize the model storage command service and its owned state."""
        self._policy = policy
        self._models_root_provider = models_root_provider
        self._clear_current_model_state_provider = clear_current_model_state_provider

    def capture_owner(self) -> ModelAdministrationOwner:
        """Capture the active model identity under the administration gate."""

        return self._policy.capture_owner()

    def is_owner_current(self, owner: ModelAdministrationOwner) -> bool:
        """Return an advisory identity snapshot for UI workflow guards."""

        return self._policy.is_owner_current(owner)

    def delete_current_model_storage(
        self,
        *,
        expected_model_uuid: str | None = None,
        operation_id: str | None = None,
    ) -> OperationResult[ModelDeletionOutcome]:
        """Delete active-model storage and retain any committed degradation."""

        correlation_id = operation_id or uuid4().hex
        try:
            outcome = self._policy.execute(
                lambda: self._delete_under_policy(
                    expected_model_uuid=expected_model_uuid,
                    operation_id=correlation_id,
                ),
                validate=lambda: self._require_deletion(expected_model_uuid),
                context_exit_recovery=lambda committed, error: (
                    self._recover_context_exit(
                        committed,
                        error,
                        operation_id=correlation_id,
                    )
                ),
            )
            return OperationResult.success(outcome, operation_id=correlation_id)
        except Exception as error:
            return self._failure_result(error, operation_id=correlation_id)

    def _require_deletion(self, expected_model_uuid: str | None) -> None:
        """Return the required deletion for this model storage command service."""
        self._policy.require_available()
        configuration = self._policy.active_configuration()
        if configuration is None:
            raise RuntimeUnavailableError(
                "Load a model before deleting model storage."
            )
        active_uuid = str(getattr(configuration, "model_uuid", "") or "").strip()
        if not active_uuid:
            raise RuntimeUnavailableError("The active model identity is unavailable.")
        expected_uuid = str(expected_model_uuid or "").strip()
        if expected_uuid and active_uuid != expected_uuid:
            raise OperationConflict(
                "The active model changed before deletion was committed."
            )

    def _delete_under_policy(
        self,
        *,
        expected_model_uuid: str | None,
        operation_id: str,
    ) -> ModelDeletionOutcome:
        """Delete the under policy through this model storage command service's safety boundary."""
        configuration = self._policy.active_configuration()
        if configuration is None:
            raise RuntimeUnavailableError(
                "Load a model before deleting model storage."
            )

        expected_uuid = str(expected_model_uuid or "").strip()
        active_uuid = str(getattr(configuration, "model_uuid", "") or "").strip()
        if expected_uuid and active_uuid != expected_uuid:
            raise OperationConflict(
                "The active model changed before deletion was committed."
            )

        model_folder_path = str(
            getattr(configuration, "model_folder_path", "") or ""
        ).strip()
        models_root = str(self._models_root_provider() or "").strip()
        if not model_folder_path or not models_root:
            raise RuntimeUnavailableError(
                "The active model storage context is unavailable."
            )

        scope = ModelsStorageMutationScope.from_models_root(models_root)
        with scope.lock():
            guarded_model_path = StoragePathGuard(
                scope.models_root
            ).require_descendant(model_folder_path, must_exist=True)
            if guarded_model_path.parent != scope.models_root:
                raise StoragePathViolation(
                    "Model deletion requires a direct managed child."
                )
            if not guarded_model_path.is_dir():
                raise NotADirectoryError(guarded_model_path)
            post_commit_error = self._commit_storage_deletion(
                scope.models_root,
                guarded_model_path,
            )

        runtime_state_cleared = self._clear_runtime_state(
            operation_id=operation_id,
        )
        storage_recovery_required = post_commit_error is not None
        reload_required = not runtime_state_cleared or storage_recovery_required
        if post_commit_error is not None:
            self._log_post_commit_error(
                post_commit_error,
                operation_id=operation_id,
                phase="storage_cleanup",
            )
        return ModelDeletionOutcome(
            model_uuid=active_uuid,
            model_name=str(getattr(configuration, "model_name", "") or "").strip(),
            committed=True,
            degraded=reload_required or storage_recovery_required,
            reload_required=reload_required,
            storage_recovery_required=storage_recovery_required,
            runtime_state_cleared=runtime_state_cleared,
        )

    @staticmethod
    def _commit_storage_deletion(
        models_root: Path,
        model_path: Path,
    ) -> Exception | None:
        """Commit the storage deletion owned by this model storage command service."""
        transaction = StorageTransaction(
            models_root,
            operation="delete-model-storage",
        )
        try:
            with transaction:
                transaction.quarantine(model_path)
                transaction.commit()
        except Exception as error:
            if not transaction.committed:
                raise
            return error
        return None

    def _clear_runtime_state(self, *, operation_id: str) -> bool:
        """Clear the runtime state owned by this model storage command service."""
        try:
            self._clear_current_model_state_provider()
        except Exception as error:
            self._log_post_commit_error(
                error,
                operation_id=operation_id,
                phase="runtime_state_clear",
            )
            return False
        return True

    def _recover_context_exit(
        self,
        committed: ModelDeletionOutcome,
        error: Exception,
        *,
        operation_id: str,
    ) -> ModelDeletionOutcome:
        """Return the recover context exit value owned by this model storage command service."""
        self._log_post_commit_error(
            error,
            operation_id=operation_id,
            phase="context_exit",
        )
        return replace(
            committed,
            degraded=True,
            reload_required=True,
            storage_recovery_required=True,
        )

    def _log_post_commit_error(
        self,
        error: Exception,
        *,
        operation_id: str,
        phase: str,
    ) -> None:
        """Log the post commit error owned by this model storage command service."""
        if isinstance(error, ExceptionGroup):
            for nested_error in error.exceptions:
                if isinstance(nested_error, Exception):
                    self._log_post_commit_error(
                        nested_error,
                        operation_id=operation_id,
                        phase=phase,
                    )
            return
        self._policy.warning(
            "model_deletion_post_commit_failed "
            f"operation_id={operation_id} phase={phase} "
            f"exception_type={type(error).__name__}"
        )
        self._policy.exception(error)

    def _failure_result(
        self,
        error: Exception,
        *,
        operation_id: str,
    ) -> OperationResult[ModelDeletionOutcome]:
        """Return the failure result value owned by this model storage command service."""
        classified = self._classify_failure(error)
        failure = failure_from_exception(
            classified,
            operation_id=operation_id,
            retry_policy=(
                RetryPolicy.SAFE if isinstance(classified, OperationTimeout) else None
            ),
        )
        self._policy.warning(
            "model_deletion_failed "
            f"operation_id={operation_id} failure_code={failure.code} "
            f"exception_type={type(error).__name__}"
        )
        if not isinstance(error, ApplicationError):
            self._policy.exception(error)
        return OperationResult.failed(failure)

    @staticmethod
    def _classify_failure(error: Exception) -> Exception:
        """Classify the failure owned by this model storage command service."""
        if isinstance(error, ApplicationError):
            return error
        if isinstance(error, StorageMutationLockTimeout):
            return OperationTimeout("Model storage is busy. Try again.")
        if isinstance(error, StoragePathViolation | InvalidStorageName):
            return ApplicationValidationError(
                "The model deletion request references invalid managed storage."
            )
        if isinstance(error, FileNotFoundError | NotADirectoryError):
            return ResourceNotFoundError(
                "The model storage no longer exists. Refresh the model list."
            )
        if isinstance(
            error,
            StorageMutationLockError | StorageTransactionError | OSError,
        ):
            return PersistenceError("Model storage could not be deleted safely.")
        return error
