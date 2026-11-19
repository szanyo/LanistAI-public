# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Crash-recoverable unit of work for deleting one training session."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Never, Protocol

from lanista.reinforcengine.rl_model.metadata.timestamps import utc_now_iso
from lanista.reinforcengine.rl_model.sessions.deletion_repository import (
    MetadataReplacement,
    SessionDeletionMetadataRepository,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    TENSORBOARD_DIR_NAME,
    TRAINING_SESSIONS_DIR_NAME,
)
from lanista.reinforcengine.rl_model.storage.mutation_lock import StorageMutationLock
from lanista.reinforcengine.rl_model.storage.mutation_scope import (
    ModelsStorageMutationScope,
    StorageMutationLockFactory,
)
from lanista.reinforcengine.rl_model.storage.names import SessionStorageName
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.storage.transaction import (
    StorageRecoveryReport,
    StorageTransaction,
    StorageTransactionError,
)


class StorageTransactionFactory(Protocol):
    """Define the typed boundary for storage transaction factory."""

    def __call__(
        self,
        root: str | os.PathLike[str],
        *,
        operation: str,
    ) -> StorageTransaction:
        """Invoke this storage transaction factory as a callable."""
        ...


@dataclass(frozen=True, slots=True)
class SessionDeletionRecoveryEvidence:
    """Describe an unsafe or ambiguous immediate journal recovery."""

    transaction_id: str
    recovery_report: StorageRecoveryReport | None
    recovery_error_type: str | None


class SessionDeletionRecoveryRequiredError(StorageTransactionError):
    """Raised when a failed deletion still requires validated recovery."""

    def __init__(
        self,
        message: str,
        evidence: SessionDeletionRecoveryEvidence,
    ) -> None:
        """Initialize the session deletion recovery required error and its owned state."""
        super().__init__(message)
        self.evidence = evidence


@dataclass(frozen=True, slots=True)
class SessionDeletionSelection:
    """Carry the data required for session deletion selection."""

    session_name: str | None
    checkpoint_name: str | None

    @classmethod
    def normalized(
        cls,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> SessionDeletionSelection:
        """Return the normalized value owned by this session deletion selection."""
        normalized_session = str(session_name or "").strip() or None
        normalized_checkpoint = (
            str(checkpoint_name or "").strip() or None
            if normalized_session
            else None
        )
        return cls(normalized_session, normalized_checkpoint)


@dataclass(frozen=True, slots=True)
class SessionDeletionRequest:
    """Describe a request to perform session deletion."""

    session_name: SessionStorageName
    session_path: Path
    tensorboard_path: Path
    indexed_sessions: tuple[str, ...]
    current_selection: SessionDeletionSelection


@dataclass(frozen=True, slots=True)
class SessionDeletionResult:
    """Report the result of session deletion."""

    remaining_sessions: tuple[str, ...]
    selection: SessionDeletionSelection
    selection_changed: bool


@dataclass(frozen=True, slots=True)
class SessionDeletionCommit:
    """Carry a durable deletion result and deferred recovery evidence."""

    value: SessionDeletionResult
    post_commit_error: Exception | None = None


class SessionDeletionUnitOfWork:
    """Commit storage, lineage, and persistent selection under one journal."""

    def __init__(
        self,
        model_root: str | Path,
        *,
        models_root: str | Path | None = None,
        transaction_factory: StorageTransactionFactory = StorageTransaction,
        lock_factory: StorageMutationLockFactory = StorageMutationLock,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the session deletion unit of work and its owned state."""
        self._guard = StoragePathGuard(model_root)
        self._repository = SessionDeletionMetadataRepository(self._guard.root)
        self._transaction_factory = transaction_factory
        self._mutation_scope = ModelsStorageMutationScope.for_model_root(
            self._guard.root,
            models_root=models_root,
        )
        self._lock_factory = lock_factory
        self._lock_timeout_seconds = lock_timeout_seconds

    @property
    def model_root(self) -> Path:
        """Return the model root exposed by this session deletion unit of work."""
        return self._guard.root

    def recover(self) -> StorageRecoveryReport:
        """Recover abandoned journals before allowing another mutation."""

        with self._mutation_scope.lock(
            lock_factory=self._lock_factory,
            timeout_seconds=self._lock_timeout_seconds,
        ):
            return self._recover_locked()

    def _recover_locked(self) -> StorageRecoveryReport:
        """Recover journals while the shared models lock is already held."""

        report = StorageTransaction.recover(self._guard.root)
        if report.failed:
            failed_ids = ", ".join(report.failed)
            raise StorageTransactionError(
                f"Session storage recovery failed for transaction(s): {failed_ids}."
            )
        return report

    def execute(
        self,
        request: SessionDeletionRequest,
        *,
        replacement_selection_provider: Callable[[], SessionDeletionSelection],
    ) -> SessionDeletionCommit:
        """Delete a session and atomically publish all persistent references."""

        committed: SessionDeletionCommit | None = None
        try:
            with self._mutation_scope.lock(
                lock_factory=self._lock_factory,
                timeout_seconds=self._lock_timeout_seconds,
            ):
                committed = self._execute_locked(
                    request,
                    replacement_selection_provider=replacement_selection_provider,
                )
                return committed
        except Exception as error:
            if committed is None:
                recovery_required = self._find_recovery_required_error(error)
                if recovery_required is not None and recovery_required is not error:
                    raise SessionDeletionRecoveryRequiredError(
                        str(recovery_required),
                        recovery_required.evidence,
                    ) from error
                raise
            return SessionDeletionCommit(
                value=committed.value,
                post_commit_error=self._combine_post_commit_errors(
                    committed.post_commit_error,
                    error,
                ),
            )

    def _execute_locked(
        self,
        request: SessionDeletionRequest,
        *,
        replacement_selection_provider: Callable[[], SessionDeletionSelection],
    ) -> SessionDeletionCommit:
        """Run preflight, recovery, and commit under the canonical lock."""

        self._recover_locked()
        session_path, tensorboard_path = self._validate_request_paths(request)
        lineage_replacements = self._repository.build_lineage_replacements(
            request.session_name,
            modified_at=utc_now_iso(),
        )
        persisted_selection = self._repository.load_runtime_selection()
        selection = SessionDeletionSelection.normalized(
            request.current_selection.session_name,
            request.current_selection.checkpoint_name,
        )
        selection_changed = selection.session_name == str(request.session_name)
        persisted_selection_changed = (
            persisted_selection.session_name == str(request.session_name)
        )

        transaction = self._transaction_factory(
            self._guard.root,
            operation="delete-session-unit-of-work",
        )
        return self._commit_deletion(
            transaction,
            request,
            session_path=session_path,
            tensorboard_path=tensorboard_path,
            lineage_replacements=lineage_replacements,
            selection=selection,
            selection_changed=selection_changed,
            persisted_selection_changed=persisted_selection_changed,
            replacement_selection_provider=replacement_selection_provider,
        )

    def _commit_deletion(
        self,
        transaction: StorageTransaction,
        request: SessionDeletionRequest,
        *,
        session_path: Path,
        tensorboard_path: Path,
        lineage_replacements: tuple[MetadataReplacement, ...],
        selection: SessionDeletionSelection,
        selection_changed: bool,
        persisted_selection_changed: bool,
        replacement_selection_provider: Callable[[], SessionDeletionSelection],
    ) -> SessionDeletionCommit:
        """Return the committed value even when terminal cleanup reports an error."""

        result: SessionDeletionResult | None = None
        try:
            with transaction:
                staged_replacements = self._repository.stage(
                    transaction,
                    lineage_replacements,
                )
                transaction.quarantine(session_path)
                if os.path.lexists(tensorboard_path):
                    transaction.quarantine(tensorboard_path)

                if selection_changed:
                    selection = replacement_selection_provider()
                    selection = SessionDeletionSelection.normalized(
                        selection.session_name,
                        selection.checkpoint_name,
                    )
                    if selection.session_name == str(request.session_name):
                        raise StorageTransactionError(
                            "Replacement runtime selection still references the deleted session."
                        )
                if selection_changed or persisted_selection_changed:
                    selection_replacement = (
                        self._repository.build_runtime_selection_replacement(
                            session_name=selection.session_name,
                            checkpoint_name=selection.checkpoint_name,
                        ),
                    )
                    staged_replacements += self._repository.stage(
                        transaction,
                        selection_replacement,
                    )

                self._repository.publish(transaction, staged_replacements)
                result = self._result_for(request, selection, selection_changed)
                transaction.commit()
        except Exception as error:
            if not transaction.committed or result is None:
                self._raise_precommit_failure_after_recovery(
                    transaction,
                    error,
                )
            return SessionDeletionCommit(
                value=result,
                post_commit_error=error,
            )
        if result is None:  # pragma: no cover - guarded by the commit order
            raise StorageTransactionError(
                "Session deletion committed without an authoritative result."
            )
        return SessionDeletionCommit(value=result)

    def _raise_precommit_failure_after_recovery(
        self,
        transaction: StorageTransaction,
        error: Exception,
    ) -> Never:
        """Recover an abandoned journal and reject ambiguous recovery evidence."""

        try:
            recovery = StorageTransaction.recover(self._guard.root)
        except Exception as recovery_error:
            evidence = SessionDeletionRecoveryEvidence(
                transaction_id=transaction.transaction_id,
                recovery_report=None,
                recovery_error_type=type(recovery_error).__name__,
            )
            raise SessionDeletionRecoveryRequiredError(
                "Session deletion rollback could not be recovered safely.",
                evidence,
            ) from ExceptionGroup(
                "Session deletion and immediate recovery both failed.",
                (error, recovery_error),
            )
        expected_rollback = (transaction.transaction_id,)
        if (
            recovery.failed
            or recovery.finalized
            or recovery.rolled_back not in {(), expected_rollback}
        ):
            evidence = SessionDeletionRecoveryEvidence(
                transaction_id=transaction.transaction_id,
                recovery_report=recovery,
                recovery_error_type=None,
            )
            raise SessionDeletionRecoveryRequiredError(
                "Session deletion rollback produced ambiguous recovery evidence.",
                evidence,
            ) from error
        if recovery.rolled_back == expected_rollback:
            raise self._precommit_operation_error(error)
        raise error

    @staticmethod
    def _precommit_operation_error(context_exit_error: Exception) -> Exception:
        """Return the precommit operation error value owned by this session deletion unit of work.
        """
        operation_error = context_exit_error.__context__
        if (
            context_exit_error.__cause__ is None
            and isinstance(operation_error, Exception)
        ):
            return operation_error
        return context_exit_error

    @staticmethod
    def _find_recovery_required_error(
        error: Exception,
    ) -> SessionDeletionRecoveryRequiredError | None:
        """Find safety evidence hidden by a later context-manager failure."""

        pending: list[BaseException] = [error]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
            if isinstance(current, SessionDeletionRecoveryRequiredError):
                return current
            if isinstance(current, BaseExceptionGroup):
                pending.extend(current.exceptions)
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)
        return None

    @staticmethod
    def _result_for(
        request: SessionDeletionRequest,
        selection: SessionDeletionSelection,
        selection_changed: bool,
    ) -> SessionDeletionResult:
        """Return the result for value owned by this session deletion unit of work."""
        return SessionDeletionResult(
            remaining_sessions=tuple(
                session
                for session in request.indexed_sessions
                if session != str(request.session_name)
            ),
            selection=selection,
            selection_changed=selection_changed,
        )

    @staticmethod
    def _combine_post_commit_errors(
        existing: Exception | None,
        additional: Exception,
    ) -> Exception:
        """Return the combine post commit errors value owned by this session deletion unit of work.
        """
        if existing is None:
            return additional
        if existing is additional:
            return existing
        return ExceptionGroup(
            "Multiple failures followed a durable session deletion.",
            (existing, additional),
        )

    def _validate_request_paths(
        self,
        request: SessionDeletionRequest,
    ) -> tuple[Path, Path]:
        """Validate the request paths required by this session deletion unit of work."""
        expected_session_path = (
            self._guard.root
            / TRAINING_SESSIONS_DIR_NAME
            / str(request.session_name)
        )
        expected_tensorboard_path = (
            self._guard.root
            / TENSORBOARD_DIR_NAME
            / str(request.session_name)
        )
        supplied_paths = (request.session_path, request.tensorboard_path)
        expected_paths = (expected_session_path, expected_tensorboard_path)
        for supplied, expected in zip(supplied_paths, expected_paths, strict=True):
            lexical_path = Path(supplied).absolute()
            guarded_path = self._guard.require_descendant(supplied)
            if lexical_path != expected or guarded_path != expected:
                raise StoragePathViolation(
                    "Session deletion paths do not match the managed storage layout."
                )
            if supplied.is_symlink():
                raise StoragePathViolation(
                    "Session deletion paths cannot be symbolic links."
                )
        if not expected_session_path.is_dir():
            raise FileNotFoundError(expected_session_path)
        return expected_session_path, expected_tensorboard_path
