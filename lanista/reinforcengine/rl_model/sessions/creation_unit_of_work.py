# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Own the crash-recoverable transaction boundary for session creation."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lanista.reinforcengine.rl_model.storage.mutation_lock import StorageMutationLock
from lanista.reinforcengine.rl_model.storage.mutation_scope import (
    ModelsStorageMutationScope,
    StorageMutationLockFactory,
)
from lanista.reinforcengine.rl_model.storage.path_guard import StoragePathGuard
from lanista.reinforcengine.rl_model.storage.transaction import (
    StorageRecoveryReport,
    StorageTransaction,
    StorageTransactionError,
)


class SessionCreationTransactionFactory(Protocol):
    """Construct one recoverable transaction rooted at the active model."""

    def __call__(
        self,
        root: str | os.PathLike[str],
        *,
        operation: str,
    ) -> StorageTransaction:
        """Invoke this session creation transaction factory as a callable."""
        ...


class SessionCreationRecoveryError(StorageTransactionError):
    """Raised when an abandoned creation journal cannot be recovered."""


@dataclass(frozen=True, slots=True)
class SessionCreationCommitOutcome[ResultT]:
    """Return a committed result and any best-effort cleanup failure."""

    result: ResultT
    post_commit_error: Exception | None = None


class SessionCreationUnitOfWork:
    """Serialize preparation and transaction commit under the models-root lock."""

    _OPERATION = "create-session-unit-of-work"

    def __init__(
        self,
        model_root: str | os.PathLike[str],
        *,
        models_root: str | os.PathLike[str] | None = None,
        transaction_factory: SessionCreationTransactionFactory = StorageTransaction,
        lock_factory: StorageMutationLockFactory = StorageMutationLock,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the session creation unit of work and its owned state."""
        self._mutation_scope = ModelsStorageMutationScope.for_model_root(
            model_root,
            models_root=models_root,
        )
        self._model_root = StoragePathGuard(model_root).root
        self._transaction_factory = transaction_factory
        self._lock_factory = lock_factory
        self._lock_timeout_seconds = lock_timeout_seconds

    @property
    def model_root(self) -> Path:
        """Return the canonical model root owned by this unit of work."""

        return self._model_root

    def recover(self) -> StorageRecoveryReport:
        """Recover abandoned creation journals under the shared root lock."""

        with self._mutation_scope.lock(
            lock_factory=self._lock_factory,
            timeout_seconds=self._lock_timeout_seconds,
        ):
            return self._recover_locked()

    def execute[PreparedT, ResultT](
        self,
        prepare: Callable[[], PreparedT],
        commit: Callable[[StorageTransaction, PreparedT], ResultT],
    ) -> SessionCreationCommitOutcome[ResultT]:
        """Prepare current state, then commit it through one durable journal."""

        with self._mutation_scope.lock(
            lock_factory=self._lock_factory,
            timeout_seconds=self._lock_timeout_seconds,
        ):
            self._recover_locked()
            prepared = prepare()
            transaction = self._create_transaction_locked()
            return self._commit_locked(transaction, prepared, commit)

    def _recover_locked(self) -> StorageRecoveryReport:
        """Recover model-root journals while the models-root lock is held."""

        report = StorageTransaction.recover(self._model_root)
        if report.failed:
            failed_ids = ", ".join(report.failed)
            raise SessionCreationRecoveryError(
                "Session creation recovery failed for transaction(s): "
                f"{failed_ids}."
            )
        return report

    def _create_transaction_locked(self) -> StorageTransaction:
        """Construct a transaction and recover a partially initialized journal."""

        try:
            return self._transaction_factory(
                self._model_root,
                operation=self._OPERATION,
            )
        except Exception:
            # Construction can fail after the workspace or journal appears.
            # Immediate recovery prevents a live process from leaving that
            # partial transaction to an unrelated future operation.
            self._recover_locked()
            raise

    @staticmethod
    def _commit_locked[PreparedT, ResultT](
        transaction: StorageTransaction,
        prepared: PreparedT,
        commit: Callable[[StorageTransaction, PreparedT], ResultT],
    ) -> SessionCreationCommitOutcome[ResultT]:
        """Distinguish rollback-safe failures from post-marker cleanup errors."""

        try:
            with transaction:
                result = commit(transaction, prepared)
                transaction.commit()
        except Exception as error:
            if not transaction.committed:
                raise
            return SessionCreationCommitOutcome(
                result=result,
                post_commit_error=error,
            )
        return SessionCreationCommitOutcome(result=result)


__all__ = [
    "SessionCreationCommitOutcome",
    "SessionCreationRecoveryError",
    "SessionCreationTransactionFactory",
    "SessionCreationUnitOfWork",
]
