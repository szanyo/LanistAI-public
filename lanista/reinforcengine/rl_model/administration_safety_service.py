# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Serialize runtime transitions with model-storage administration."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock, RLock
from uuid import uuid4

from lanista.reinforcengine.application_errors import OperationConflict, OperationTimeout
from lanista.reinforcengine.rl_model.execution_contracts import ExecutionLease
from lanista.reinforcengine.rl_model.storage.recovery_service import (
    ValidatedStorageRecovery,
    is_validated_storage_recovery,
)

_RUNTIME_RECOVERY_ACKNOWLEDGEMENT_MARKER = object()


class RuntimeRecoveryAcknowledgement:
    """Context-bound capability for proven storage recovery."""

    __slots__ = ("_active", "_marker", "_storage_recovery_validated")

    _active: bool
    _marker: object
    _storage_recovery_validated: bool

    def __init__(self) -> None:
        """Initialize the runtime recovery acknowledgement and its owned state."""
        raise TypeError(
            "Runtime recovery acknowledgement is issued only by its safety gate."
        )

    def __setattr__(self, name: str, value: object) -> None:
        """Assign an attribute while preserving this runtime recovery acknowledgement's invariants.
        """
        raise AttributeError("Runtime recovery acknowledgement is immutable.")

    def acknowledge_storage_recovery(
        self,
        proof: ValidatedStorageRecovery,
    ) -> None:
        """Remember active storage proof for the enclosing successful commit."""

        if (
            self._marker is not _RUNTIME_RECOVERY_ACKNOWLEDGEMENT_MARKER
            or not self._active
        ):
            raise OperationConflict(
                "Runtime recovery acknowledgement is no longer active."
            )
        if not is_validated_storage_recovery(proof):
            raise OperationConflict(
                "Runtime recovery did not provide valid storage proof."
            )
        object.__setattr__(self, "_storage_recovery_validated", True)


def _new_runtime_recovery_acknowledgement() -> RuntimeRecoveryAcknowledgement:
    """Return the new runtime recovery acknowledgement value owned by this administration safety
    service module.
    """
    acknowledgement = object.__new__(RuntimeRecoveryAcknowledgement)
    object.__setattr__(
        acknowledgement,
        "_marker",
        _RUNTIME_RECOVERY_ACKNOWLEDGEMENT_MARKER,
    )
    object.__setattr__(acknowledgement, "_active", True)
    object.__setattr__(
        acknowledgement,
        "_storage_recovery_validated",
        False,
    )
    return acknowledgement


def _invalidate_runtime_recovery_acknowledgement(
    acknowledgement: RuntimeRecoveryAcknowledgement,
) -> None:
    """Invalidate the runtime recovery acknowledgement owned by this administration safety service
    module.
    """
    object.__setattr__(acknowledgement, "_active", False)


@dataclass(frozen=True, slots=True)
class AdministrationSafetySnapshot:
    """Current execution and post-commit recovery requirements."""

    execution_active: bool
    execution_run_id: str | None
    reload_required: bool
    storage_recovery_required: bool
    runtime_load_active: bool = False
    runtime_load_operation_id: str | None = None
    runtime_recovery_active: bool = False
    administration_active: bool = False
    administration_completion_pending: bool = False

    @property
    def blocked(self) -> bool:
        """Return the blocked exposed by this administration safety snapshot."""
        return (
            self.execution_active
            or self.runtime_load_active
            or self.runtime_recovery_active
            or self.administration_active
            or self.administration_completion_pending
            or self.reload_required
            or self.storage_recovery_required
        )


class AdministrationCompletionGuard:
    """Bridge a durable commit across administration-context release.

    An armed guard keeps runtime transitions blocked after the transition lock
    is released. The caller must then confirm a clean context exit or fail the
    completion, which atomically latches recovery requirements before removing
    the transient blocker.
    """

    __slots__ = ("_completion_id", "_service")

    def __init__(
        self,
        service: AdministrationSafetyService,
        completion_id: str,
    ) -> None:
        """Initialize the administration completion guard and its owned state."""
        self._service = service
        self._completion_id = completion_id

    def arm_committed_completion(
        self,
        *,
        reload_required: bool,
        storage_recovery_required: bool,
    ) -> None:
        """Block transitions until the caller observes context release."""

        self._service._arm_administration_completion(
            self._completion_id,
            reload_required=reload_required,
            storage_recovery_required=storage_recovery_required,
        )

    def confirm_successful_exit(self) -> None:
        """Remove only this guard's transient completion blocker."""

        self._service._resolve_administration_completion(
            self._completion_id,
            reload_required=False,
            storage_recovery_required=False,
        )

    def latch_recovery_required(self) -> None:
        """Fail closed inside the active context before any unlock."""

        self._service._latch_administration_recovery(self._completion_id)

    def fail_exit(self) -> None:
        """Latch context-exit degradation and remove the transient blocker."""

        self._service._resolve_administration_completion(
            self._completion_id,
            reload_required=True,
            storage_recovery_required=True,
        )


class _AdministrationExecutionLease:
    """Idempotent handle for one safety-service execution owner."""

    def __init__(
        self,
        service: AdministrationSafetyService,
        run_id: str,
    ) -> None:
        """Initialize the administration execution lease and its owned state."""
        self._service = service
        self._run_id = run_id
        self._release_lock = Lock()
        self._released = False

    @property
    def run_id(self) -> str:
        """Return the run id exposed by this administration execution lease."""
        return self._run_id

    def mark_reload_required(self) -> None:
        """Latch recovery before a timed-out run can release ownership."""

        self._service.mark_degraded(
            reload_required=True,
            storage_recovery_required=False,
        )

    def release(self) -> None:
        """Release the operation owned by this administration execution lease."""
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._service._release_execution_lease(self._run_id)


class _AdministrationRuntimeLoadLease:
    """Idempotent handle for one candidate-load transition owner."""

    def __init__(
        self,
        service: AdministrationSafetyService,
        operation_id: str,
    ) -> None:
        """Initialize the administration runtime load lease and its owned state."""
        self._service = service
        self._operation_id = operation_id
        self._release_lock = Lock()
        self._released = False

    @property
    def operation_id(self) -> str:
        """Return the operation id exposed by this administration runtime load lease."""
        return self._operation_id

    def release(self) -> None:
        """Release the operation owned by this administration runtime load lease."""
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._service._release_runtime_load_lease(self._operation_id)


class AdministrationSafetyService:
    """Own the transition lock shared by run start and administration.

    Administration holds the transition lock for its complete mutation. A run
    may start only after that context exits, at which point it also observes
    any reload/recovery requirement recorded by the mutation.
    """

    def __init__(
        self,
        *,
        transition_timeout_seconds: float = 5.0,
        native_gui_mutation_guard: Callable[[], None] = lambda: None,
    ) -> None:
        """Initialize the administration safety service and its owned state."""
        self._transition_lock = RLock()
        self._transition_timeout_seconds = max(
            0.0,
            float(transition_timeout_seconds),
        )
        self._execution_run_id: str | None = None
        self._runtime_load_operation_id: str | None = None
        self._runtime_recovery_active = False
        self._active_administration_context_ids: set[str] = set()
        self._pending_administration_completion_ids: set[str] = set()
        self._reload_required = False
        self._storage_recovery_required = False
        self._reload_degradation_generation = 0
        self._storage_degradation_generation = 0
        self._native_gui_mutation_guard = native_gui_mutation_guard

    def snapshot(self) -> AdministrationSafetySnapshot:
        """Return the snapshot value owned by this administration safety service."""
        with self._transition_lock:
            return self._snapshot_locked()

    @contextmanager
    def administration_operation(
        self,
        owner_operation_id: str | None = None,
    ) -> Iterator[AdministrationCompletionGuard]:
        """Exclude run transitions and reject unresolved degraded state."""

        if not self._transition_lock.acquire(
            timeout=self._transition_timeout_seconds
        ):
            raise OperationTimeout(
                "Administration could not acquire the runtime transition gate."
            )
        completion_id = uuid4().hex
        completion_guard = AdministrationCompletionGuard(
            self,
            completion_id,
        )
        try:
            self._raise_if_administration_blocked(
                self._snapshot_locked(),
                owner_operation_id=owner_operation_id,
            )
            self._active_administration_context_ids.add(completion_id)
            try:
                yield completion_guard
            finally:
                self._active_administration_context_ids.discard(completion_id)
        finally:
            self._transition_lock.release()

    @contextmanager
    def runtime_recovery_operation(
        self,
        owner_operation_id: str | None = None,
    ) -> Iterator[RuntimeRecoveryAcknowledgement]:
        """Commit a recovered runtime slot and acknowledge degradation atomically.

        A recovery load is allowed while reload or storage recovery is pending,
        but never while a worker owns the execution lease. Recovery flags are
        cleared only after the guarded slot commit exits successfully. A plain
        runtime reload acknowledges ``reload_required`` only; clearing the
        storage flag additionally requires active service-issued proof
        acknowledged by the guarded slot transaction.
        """

        if not self._transition_lock.acquire(
            timeout=self._transition_timeout_seconds
        ):
            raise OperationTimeout(
                "Runtime recovery could not acquire the transition gate."
            )
        acknowledgement = _new_runtime_recovery_acknowledgement()
        try:
            snapshot = self._snapshot_locked()
            if snapshot.execution_active:
                raise OperationConflict(
                    "Runtime recovery is disabled while execution is active."
                )
            if snapshot.runtime_recovery_active:
                raise OperationConflict(
                    "Another runtime recovery operation is already active."
                )
            if snapshot.administration_active:
                raise OperationConflict(
                    "Runtime recovery is disabled during administration."
                )
            if snapshot.administration_completion_pending:
                raise OperationConflict(
                    "Runtime recovery is disabled while administration is finalizing."
                )
            if (
                snapshot.runtime_load_active
                and snapshot.runtime_load_operation_id != owner_operation_id
            ):
                raise OperationConflict(
                    "Runtime recovery is owned by another load operation."
                )
            self._runtime_recovery_active = True
            reload_generation = self._reload_degradation_generation
            storage_generation = self._storage_degradation_generation
            try:
                yield acknowledgement
                self._acknowledge_successful_reload_locked(
                    storage_recovery_validated=(
                        acknowledgement._storage_recovery_validated
                    ),
                    expected_reload_generation=reload_generation,
                    expected_storage_generation=storage_generation,
                )
            finally:
                self._runtime_recovery_active = False
        finally:
            _invalidate_runtime_recovery_acknowledgement(acknowledgement)
            self._transition_lock.release()

    def acquire_execution_lease(self, run_id: str) -> ExecutionLease:
        """Acquire the exclusive administration boundary for one run ID."""

        self._native_gui_mutation_guard()
        normalized_run_id = str(run_id).strip()
        if not normalized_run_id:
            raise ValueError("Execution lease run_id must not be empty.")
        if not self._transition_lock.acquire(
            timeout=self._transition_timeout_seconds
        ):
            raise OperationTimeout(
                "Execution could not acquire the runtime transition gate."
            )
        try:
            self._activate_execution_locked(normalized_run_id)
            return _AdministrationExecutionLease(self, normalized_run_id)
        finally:
            self._transition_lock.release()

    def acquire_runtime_load_lease(
        self,
        operation_id: str,
    ) -> _AdministrationRuntimeLoadLease:
        """Reserve candidate preparation and slot commit against execution."""

        self._native_gui_mutation_guard()
        normalized_operation_id = str(operation_id).strip()
        if not normalized_operation_id:
            raise ValueError("Runtime load operation_id must not be empty.")
        if not self._transition_lock.acquire(
            timeout=self._transition_timeout_seconds
        ):
            raise OperationTimeout(
                "Runtime loading could not acquire the transition gate."
            )
        try:
            snapshot = self._snapshot_locked()
            if snapshot.execution_active:
                raise OperationConflict(
                    "Runtime loading is disabled while execution is active."
                )
            if snapshot.runtime_recovery_active:
                raise OperationConflict(
                    "Runtime loading is disabled during runtime recovery."
                )
            if snapshot.administration_active:
                raise OperationConflict(
                    "Runtime loading is disabled during administration."
                )
            if snapshot.administration_completion_pending:
                raise OperationConflict(
                    "Runtime loading is disabled while administration is finalizing."
                )
            if snapshot.runtime_load_active:
                raise OperationConflict(
                    "Another runtime load operation is already active."
                )
            self._runtime_load_operation_id = normalized_operation_id
            return _AdministrationRuntimeLoadLease(
                self,
                normalized_operation_id,
            )
        finally:
            self._transition_lock.release()

    def mark_degraded(
        self,
        *,
        reload_required: bool,
        storage_recovery_required: bool,
    ) -> None:
        """Persist post-commit restrictions before administration unlocks."""

        with self._transition_lock:
            self._mark_degraded_locked(
                reload_required=reload_required,
                storage_recovery_required=storage_recovery_required,
            )

    def clear_after_successful_reload(self) -> None:
        """A successful load/readback acknowledges only runtime reload state."""

        with self._transition_lock:
            if (
                self._execution_run_id is not None
                or self._runtime_load_operation_id is not None
                or self._runtime_recovery_active
                or self._active_administration_context_ids
                or self._pending_administration_completion_ids
            ):
                raise OperationConflict(
                    "Recovery state cannot be cleared during a runtime transition."
                )
            self._acknowledge_successful_reload_locked()

    def run_start_block_reason(self) -> str | None:
        """Return a stable user-facing blocker for the run precheck."""

        snapshot = self.snapshot()
        if snapshot.runtime_load_active:
            return "Runtime loading is in progress."
        if snapshot.runtime_recovery_active:
            return "Runtime recovery is in progress."
        if snapshot.administration_active:
            return "Model administration is in progress."
        if snapshot.administration_completion_pending:
            return "Model administration is still finalizing."
        if snapshot.storage_recovery_required:
            return "Storage recovery and model reload are required before starting."
        if snapshot.reload_required:
            return "Reload the active model before starting."
        return None

    def _snapshot_locked(self) -> AdministrationSafetySnapshot:
        """Return the snapshot locked value owned by this administration safety service."""
        return AdministrationSafetySnapshot(
            execution_active=self._execution_run_id is not None,
            execution_run_id=self._execution_run_id,
            reload_required=self._reload_required,
            storage_recovery_required=self._storage_recovery_required,
            runtime_load_active=self._runtime_load_operation_id is not None,
            runtime_load_operation_id=self._runtime_load_operation_id,
            runtime_recovery_active=self._runtime_recovery_active,
            administration_active=bool(
                self._active_administration_context_ids
            ),
            administration_completion_pending=bool(
                self._pending_administration_completion_ids
            ),
        )

    def _activate_execution_locked(self, run_id: str) -> None:
        """Activate the execution locked owned by this administration safety service."""
        snapshot = self._snapshot_locked()
        if snapshot.execution_active:
            raise OperationConflict(
                f"Execution run {snapshot.execution_run_id} is already active."
            )
        if snapshot.runtime_load_active:
            raise OperationConflict(
                "Runtime loading is in progress."
            )
        if snapshot.runtime_recovery_active:
            raise OperationConflict(
                "Execution cannot start during runtime recovery."
            )
        if snapshot.administration_active:
            raise OperationConflict(
                "Execution cannot start during model administration."
            )
        if snapshot.administration_completion_pending:
            raise OperationConflict(
                "Model administration is still finalizing."
            )
        if snapshot.storage_recovery_required:
            raise OperationConflict(
                "Storage recovery is required before starting execution."
            )
        if snapshot.reload_required:
            raise OperationConflict(
                "Reload the active model before starting execution."
            )
        self._execution_run_id = run_id

    def _release_execution_lease(self, run_id: str) -> None:
        """Clear only the matching owner so a stale release cannot unlock a new run."""

        with self._transition_lock:
            if self._execution_run_id == run_id:
                self._execution_run_id = None

    def _release_runtime_load_lease(self, operation_id: str) -> None:
        """Clear only the matching load owner so stale releases stay harmless."""

        with self._transition_lock:
            if self._runtime_load_operation_id == operation_id:
                self._runtime_load_operation_id = None

    def _acknowledge_successful_reload_locked(
        self,
        *,
        storage_recovery_validated: bool = False,
        expected_reload_generation: int | None = None,
        expected_storage_generation: int | None = None,
    ) -> None:
        """Acknowledge a successful reload while holding the administration safety lock."""
        if (
            expected_reload_generation is None
            or self._reload_degradation_generation
            == expected_reload_generation
        ):
            self._reload_required = False
        if (
            storage_recovery_validated
            and (
                expected_storage_generation is None
                or self._storage_degradation_generation
                == expected_storage_generation
            )
        ):
            self._storage_recovery_required = False

    def _arm_administration_completion(
        self,
        completion_id: str,
        *,
        reload_required: bool,
        storage_recovery_required: bool,
    ) -> None:
        """Arm one active context before its durable result crosses unlock."""

        with self._transition_lock:
            if completion_id not in self._active_administration_context_ids:
                raise OperationConflict(
                    "Administration completion can only be armed in its active context."
                )
            self._pending_administration_completion_ids.add(completion_id)
            self._mark_degraded_locked(
                reload_required=reload_required,
                storage_recovery_required=storage_recovery_required,
            )

    def _resolve_administration_completion(
        self,
        completion_id: str,
        *,
        reload_required: bool,
        storage_recovery_required: bool,
    ) -> None:
        """Resolve only a matching post-context completion token."""

        with self._transition_lock:
            if completion_id in self._active_administration_context_ids:
                raise OperationConflict(
                    "Administration completion cannot finish before context exit."
                )
            if completion_id not in self._pending_administration_completion_ids:
                return
            self._mark_degraded_locked(
                reload_required=reload_required,
                storage_recovery_required=storage_recovery_required,
            )
            self._pending_administration_completion_ids.remove(completion_id)

    def _latch_administration_recovery(self, completion_id: str) -> None:
        """Latch recovery only from the context that owns the capability."""

        with self._transition_lock:
            if completion_id not in self._active_administration_context_ids:
                raise OperationConflict(
                    "Administration recovery can only be latched in its active context."
                )
            self._mark_degraded_locked(
                reload_required=True,
                storage_recovery_required=True,
            )

    def _mark_degraded_locked(
        self,
        *,
        reload_required: bool,
        storage_recovery_required: bool,
    ) -> None:
        """Mark the degraded locked owned by this administration safety
service."""
        if reload_required:
            self._reload_required = True
            self._reload_degradation_generation += 1
        if storage_recovery_required:
            self._storage_recovery_required = True
            self._storage_degradation_generation += 1

    @staticmethod
    def _raise_if_administration_blocked(
        snapshot: AdministrationSafetySnapshot,
        *,
        owner_operation_id: str | None = None,
    ) -> None:
        """Raise the if administration blocked owned by this administration
safety service."""
        if snapshot.execution_active:
            raise OperationConflict(
                "Administration is disabled while execution is active."
            )
        if snapshot.runtime_recovery_active:
            raise OperationConflict(
                "Administration is disabled during runtime recovery."
            )
        if snapshot.administration_completion_pending:
            raise OperationConflict(
                "Administration is disabled while another operation is finalizing."
            )
        if (
            snapshot.runtime_load_active
            and snapshot.runtime_load_operation_id != owner_operation_id
        ):
            raise OperationConflict(
                "Administration is disabled while runtime loading is active."
            )
        if snapshot.storage_recovery_required:
            raise OperationConflict(
                "Storage recovery and model reload are required before administration."
            )
        if snapshot.reload_required:
            raise OperationConflict(
                "Reload the active model before further administration."
            )
