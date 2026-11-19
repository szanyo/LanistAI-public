# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Own the durable training-session deletion command boundary."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

from lanista.foundation.configuration import Configuration
from lanista.reinforcengine.application_errors import (
    ApplicationError,
    ApplicationValidationError,
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
    SessionDeletionOutcome,
)
from lanista.reinforcengine.rl_model.sessions.deletion_unit_of_work import (
    SessionDeletionRecoveryRequiredError,
    SessionDeletionRequest,
    SessionDeletionResult,
    SessionDeletionSelection,
    SessionDeletionUnitOfWork,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    TENSORBOARD_DIR_NAME,
    TRAINING_SESSIONS_DIR_NAME,
)
from lanista.reinforcengine.rl_model.storage.mutation_lock import (
    StorageMutationLockError,
    StorageMutationLockTimeout,
)
from lanista.reinforcengine.rl_model.storage.names import InvalidStorageName, SessionStorageName
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.storage.transaction import StorageTransactionError


class SessionPathResolver(Protocol):
    """Resolve one managed session path without owning deletion semantics."""

    def get_session_path(
        self,
        session_name: str,
    ) -> str | os.PathLike[str]:
        """Return the session path exposed by this session path resolver."""
        ...


@dataclass(frozen=True, slots=True)
class _SessionDeletionPaths:
    """Represent configuration for session deletion."""

    model_root: Path
    session: Path
    tensorboard: Path


class SessionDeletionCommandService:
    """Own session deletion, replacement selection, and runtime projection."""

    def __init__(
        self,
        *,
        policy: AdministrationCommandPolicy,
        models_root_provider: Callable[[], str],
        session_manager_provider: Callable[[], SessionPathResolver | None],
        selected_session_provider: Callable[[], str | None],
        selected_checkpoint_provider: Callable[[], str | None],
        get_replacement_session_provider: Callable[[str], str | None],
        get_latest_checkpoint_provider: Callable[[str], str | None],
        set_current_model_runtime_selection_provider: Callable[
            [str | None, str | None, bool], None
        ],
    ) -> None:
        """Initialize the session deletion command service and its owned state."""
        self._policy = policy
        self._models_root_provider = models_root_provider
        self._session_manager_provider = session_manager_provider
        self._selected_session_provider = selected_session_provider
        self._selected_checkpoint_provider = selected_checkpoint_provider
        self._get_replacement_session_provider = get_replacement_session_provider
        self._get_latest_checkpoint_provider = get_latest_checkpoint_provider
        self._set_current_model_runtime_selection_provider = (
            set_current_model_runtime_selection_provider
        )

    def delete_session(
        self,
        owner: ModelAdministrationOwner,
        session_uuid: str | None,
        *,
        operation_id: str | None = None,
    ) -> OperationResult[SessionDeletionOutcome]:
        """Delete a captured model's session and expose degradation."""

        correlation_id = operation_id or uuid4().hex
        try:
            outcome = self._policy.execute(
                lambda: self._delete_under_policy(
                    session_uuid,
                    operation_id=correlation_id,
                ),
                validate=lambda: self._require_administration(owner),
                context_exit_recovery=lambda committed, error: (
                    self._recover_context_exit(
                        committed,
                        error,
                        operation_id=correlation_id,
                    )
                ),
                recovery_required_error_types=(
                    SessionDeletionRecoveryRequiredError,
                ),
            )
            return OperationResult.success(outcome, operation_id=correlation_id)
        except Exception as error:
            classified = self._classify_failure(error)
            failure = failure_from_exception(
                classified,
                operation_id=correlation_id,
                retry_policy=(
                    RetryPolicy.SAFE
                    if isinstance(classified, OperationTimeout)
                    else None
                ),
            )
            self._policy.warning(
                "session_deletion_failed "
                f"operation_id={correlation_id} failure_code={failure.code} "
                f"exception_type={type(error).__name__}"
            )
            self._policy.exception(error)
            return OperationResult.failed(failure)

    def _require_administration(self, owner: ModelAdministrationOwner) -> None:
        """Return the required administration for this session deletion command service."""
        self._policy.require_available()
        self._policy.require_current_owner(owner)

    def _delete_under_policy(
        self,
        session_uuid: str | None,
        *,
        operation_id: str,
    ) -> SessionDeletionOutcome:
        """Delete the under policy through this session deletion command service's safety boundary.
        """
        configuration = self._policy.active_configuration()
        session_manager = self._session_manager_provider()
        if configuration is None or session_manager is None:
            raise RuntimeUnavailableError(
                "Load a model before deleting a training session."
            )

        typed_configuration = cast(Configuration, configuration)
        session_name = SessionStorageName.parse(session_uuid or "")
        paths = self._resolve_paths(typed_configuration, session_manager, session_name)
        indexed_sessions = list(getattr(typed_configuration, "training_sessions", []) or [])
        selected_session = self._selected_session_provider()
        selected_checkpoint = self._selected_checkpoint_provider()
        captured_replacement = self._resolve_replacement(
            session_name,
            selected_session=selected_session,
        )

        try:
            commit = SessionDeletionUnitOfWork(
                paths.model_root,
                models_root=self._models_root_provider(),
            ).execute(
                SessionDeletionRequest(
                    session_name=session_name,
                    session_path=paths.session,
                    tensorboard_path=paths.tensorboard,
                    indexed_sessions=tuple(indexed_sessions),
                    current_selection=SessionDeletionSelection.normalized(
                        selected_session,
                        selected_checkpoint,
                    ),
                ),
                replacement_selection_provider=lambda: captured_replacement,
            )
        except SessionDeletionRecoveryRequiredError as error:
            self._record_degradation(
                operation_id,
                reload_required=True,
                storage_recovery_required=True,
            )
            self._policy.warning(
                "session_deletion_recovery_required "
                f"operation_id={operation_id} "
                f"transaction_id={error.evidence.transaction_id} "
                "recovery_error_type="
                f"{error.evidence.recovery_error_type or 'none'}"
            )
            raise

        projection_synchronized = self._publish_projection(
            typed_configuration,
            commit.value,
            operation_id=operation_id,
        )
        return self._build_outcome(
            session_name=str(session_name),
            result=commit.value,
            projection_synchronized=projection_synchronized,
            post_commit_error=commit.post_commit_error,
            operation_id=operation_id,
        )

    def _resolve_paths(
        self,
        configuration: Configuration,
        session_manager: SessionPathResolver,
        session_name: SessionStorageName,
    ) -> _SessionDeletionPaths:
        """Resolve the paths required by this session deletion command service."""
        models_root = str(self._models_root_provider() or "").strip()
        model_folder = str(
            getattr(configuration, "model_folder_path", "") or ""
        ).strip()
        if not models_root or not model_folder:
            raise StoragePathViolation("Model storage context is incomplete.")

        guarded_model_root = StoragePathGuard(models_root).require_descendant(
            model_folder,
            must_exist=True,
        )
        lexical_model_root = Path(os.path.abspath(model_folder))
        if lexical_model_root != guarded_model_root:
            raise StoragePathViolation(
                "Model storage aliases are forbidden for deletion."
            )
        if not guarded_model_root.is_dir():
            raise NotADirectoryError(guarded_model_root)

        model_guard = StoragePathGuard(guarded_model_root)
        session_root = guarded_model_root / TRAINING_SESSIONS_DIR_NAME
        tensorboard_root = guarded_model_root / TENSORBOARD_DIR_NAME
        expected_session_path = session_root / str(session_name)
        tensorboard_path = tensorboard_root / str(session_name)
        provided_session = session_manager.get_session_path(str(session_name))
        try:
            provided_session_path = os.fspath(provided_session)
        except TypeError as error:
            raise StoragePathViolation(
                "Session storage adapter returned an invalid path."
            ) from error
        provided_lexical_path = Path(os.path.abspath(provided_session_path))
        if provided_lexical_path != expected_session_path:
            raise StoragePathViolation(
                "Session storage path does not match its managed location."
            )

        guarded_paths = model_guard.validate_all(
            (expected_session_path, tensorboard_path),
        )
        if guarded_paths != (expected_session_path, tensorboard_path):
            raise StoragePathViolation(
                "Session storage aliases are forbidden for deletion."
            )
        if session_root.is_symlink() or tensorboard_root.is_symlink():
            raise StoragePathViolation(
                "Session storage roots cannot be symbolic links."
            )
        for candidate in (expected_session_path, tensorboard_path):
            if candidate.is_symlink():
                raise StoragePathViolation(
                    "Session storage entries cannot be symbolic links."
                )
            if os.path.lexists(candidate) and not candidate.is_dir():
                raise NotADirectoryError(candidate)
        return _SessionDeletionPaths(
            model_root=guarded_model_root,
            session=expected_session_path,
            tensorboard=tensorboard_path,
        )

    def _resolve_replacement(
        self,
        session_name: SessionStorageName,
        *,
        selected_session: str | None,
    ) -> SessionDeletionSelection:
        """Resolve the replacement required by this session deletion command service."""
        if selected_session != str(session_name):
            return SessionDeletionSelection.normalized(None, None)
        try:
            replacement_session = self._get_replacement_session_provider(
                str(session_name)
            )
            replacement_checkpoint = (
                self._get_latest_checkpoint_provider(replacement_session)
                if replacement_session
                else None
            )
            return SessionDeletionSelection.normalized(
                replacement_session,
                replacement_checkpoint,
            )
        except Exception as error:
            raise RuntimeUnavailableError(
                "A replacement runtime selection could not be resolved."
            ) from error

    def _publish_projection(
        self,
        configuration: Configuration,
        result: SessionDeletionResult,
        *,
        operation_id: str,
    ) -> bool:
        """Publish the projection owned by this session deletion command service."""
        synchronized = self._run_projection_step(
            operation_id,
            "training_sessions",
            lambda: setattr(
                configuration,
                "training_sessions",
                list(result.remaining_sessions),
            ),
        )
        if not result.selection_changed:
            return synchronized

        runtime_selection_synchronized = self._run_projection_step(
            operation_id,
            "runtime_selection",
            lambda: self._set_current_model_runtime_selection_provider(
                result.selection.session_name,
                result.selection.checkpoint_name,
                False,
            ),
        )
        configuration_selection_synchronized = self._run_projection_step(
            operation_id,
            "configuration_selection",
            lambda: self._synchronize_configuration_projection(
                configuration,
                result.selection,
            ),
        )
        return (
            synchronized
            and runtime_selection_synchronized
            and configuration_selection_synchronized
        )

    def _run_projection_step(
        self,
        operation_id: str,
        phase: str,
        action: Callable[[], None],
    ) -> bool:
        """Return the run projection step value owned by this session deletion command service."""
        try:
            action()
        except Exception as error:
            self._policy.warning(
                "session_deletion_projection_failed "
                f"operation_id={operation_id} phase={phase} "
                f"exception_type={type(error).__name__}"
            )
            self._policy.exception(error)
            return False
        return True

    @staticmethod
    def _synchronize_configuration_projection(
        configuration: Configuration,
        selection: SessionDeletionSelection,
    ) -> None:
        """Synchronize the configuration projection owned by this session deletion command service.
        """
        session_name = selection.session_name or ""
        checkpoint_name = selection.checkpoint_name or ""
        configuration.selected_default_model_name = session_name
        configuration.selected_default_checkpoint_name = checkpoint_name

        model_configuration = getattr(configuration, "model_configuration", None)
        if isinstance(model_configuration, dict):
            model_configuration["selected_default_model_name"] = session_name
            model_configuration["selected_default_checkpoint_name"] = checkpoint_name
        elif model_configuration is not None:
            model_configuration.selected_default_model_name = session_name
            model_configuration.selected_default_checkpoint_name = checkpoint_name

        schema = getattr(configuration, "model_configuration_schema", None)
        model_copy = getattr(schema, "model_copy", None)
        if callable(model_copy):
            configuration.model_configuration_schema = model_copy(
                update={
                    "selected_default_model_name": session_name,
                    "selected_default_checkpoint_name": checkpoint_name,
                }
            )

    def _build_outcome(
        self,
        *,
        session_name: str,
        result: SessionDeletionResult,
        projection_synchronized: bool,
        post_commit_error: Exception | None,
        operation_id: str,
    ) -> SessionDeletionOutcome:
        """Build the outcome required by this session deletion command service."""
        storage_recovery_required = post_commit_error is not None
        reload_required = not projection_synchronized or storage_recovery_required
        if reload_required or storage_recovery_required:
            self._record_degradation(
                operation_id,
                reload_required=reload_required,
                storage_recovery_required=storage_recovery_required,
            )
        if post_commit_error is not None:
            self._log_post_commit_errors(
                post_commit_error,
                operation_id=operation_id,
            )
        return SessionDeletionOutcome(
            session_name=session_name,
            remaining_sessions=result.remaining_sessions,
            selection=result.selection,
            committed=True,
            degraded=reload_required or storage_recovery_required,
            reload_required=reload_required,
            storage_recovery_required=storage_recovery_required,
            projection_synchronized=projection_synchronized,
        )

    def _recover_context_exit(
        self,
        committed: SessionDeletionOutcome,
        error: Exception,
        *,
        operation_id: str,
    ) -> SessionDeletionOutcome:
        """Return the recover context exit value owned by this session deletion command service."""
        self._record_degradation(
            operation_id,
            reload_required=True,
            storage_recovery_required=True,
        )
        self._log_post_commit_errors(error, operation_id=operation_id)
        return replace(
            committed,
            degraded=True,
            reload_required=True,
            storage_recovery_required=True,
        )

    def _log_post_commit_errors(
        self,
        error: Exception,
        *,
        operation_id: str,
    ) -> None:
        """Log the post commit errors owned by this session deletion command service."""
        if isinstance(error, ExceptionGroup):
            for nested_error in error.exceptions:
                if isinstance(nested_error, Exception):
                    self._log_post_commit_errors(
                        nested_error,
                        operation_id=operation_id,
                    )
            return
        self._policy.warning(
            "session_deletion_post_commit_failed "
            f"operation_id={operation_id} "
            f"exception_type={type(error).__name__}"
        )
        self._policy.exception(error)

    def _record_degradation(
        self,
        operation_id: str,
        *,
        reload_required: bool,
        storage_recovery_required: bool,
    ) -> None:
        """Record the degradation owned by this session deletion command service."""
        self._policy.report_degradation(
            operation_id=operation_id,
            failure_event="session_deletion_degraded_state_report_failed",
            reload_required=reload_required,
            storage_recovery_required=storage_recovery_required,
        )

    @staticmethod
    def _classify_failure(error: Exception) -> Exception:
        """Classify the failure owned by this session deletion command service."""
        if isinstance(error, ApplicationError):
            return error
        if isinstance(error, StorageMutationLockTimeout):
            return OperationTimeout("Session storage is busy. Try again.")
        if isinstance(error, InvalidStorageName | StoragePathViolation):
            return ApplicationValidationError(
                "The session deletion request is invalid."
            )
        if isinstance(error, FileNotFoundError | NotADirectoryError):
            return ResourceNotFoundError(
                "The session storage no longer exists. Refresh the session list."
            )
        if isinstance(
            error,
            StorageMutationLockError | StorageTransactionError | OSError,
        ):
            return PersistenceError("Session storage could not be deleted safely.")
        return error
