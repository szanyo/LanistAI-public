# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Short, synchronous commit boundary for the active runtime slot."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import replace
from typing import TYPE_CHECKING, Protocol

from lanista.reinforcengine.application_errors import OperationCancelled, OperationConflict
from lanista.reinforcengine.rl_model.administration_safety_service import (
    RuntimeRecoveryAcknowledgement,
)
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_model.runtime_slot_store import RuntimeSlotStore
from lanista.reinforcengine.rl_model.storage.recovery_service import (
    StorageRecoveryReceipt,
    ValidatedStorageRecovery,
)

if TYPE_CHECKING:
    from lanista.reinforcengine.rl_model.runtime_slot_store import RuntimeSlotState


class _RuntimeRecoveryOperationFactory(Protocol):
    """Open the safety gate for one runtime-slot transaction."""

    def __call__(
        self,
        owner_operation_id: str | None = None,
    ) -> AbstractContextManager[RuntimeRecoveryAcknowledgement]:
        """Invoke this runtime recovery operation factory as a callable."""
        ...


_StorageRecoveryOperationFactory = Callable[
    [StorageRecoveryReceipt, ModelManifest],
    AbstractContextManager[ValidatedStorageRecovery],
]


class RuntimeSlotCommitService:
    """Apply or clear the active runtime without exposing a partial slot."""

    def __init__(
        self,
        *,
        slot_store: RuntimeSlotStore,
        administration_operation_factory: Callable[
            [str | None], AbstractContextManager[object]
        ],
        recovery_operation_factory: _RuntimeRecoveryOperationFactory,
        storage_recovery_operation_factory: _StorageRecoveryOperationFactory,
    ) -> None:
        """Initialize the runtime slot commit service and its owned state."""
        self._slot_store = slot_store
        self._administration_operation_factory = administration_operation_factory
        self._recovery_operation_factory = recovery_operation_factory
        self._storage_recovery_operation_factory = (
            storage_recovery_operation_factory
        )

    def commit_loaded(
        self,
        state: RuntimeSlotState,
        *,
        expected_revision: int,
        after_commit: Callable[[], None] | None = None,
        transition_owner_id: str | None = None,
        storage_recovery_receipt: StorageRecoveryReceipt | None = None,
    ) -> BaseModelRuntime | None:
        """CAS-commit a candidate and acknowledge only proven recovery."""

        storage_recovery_operation = None
        if storage_recovery_receipt is not None:
            if state.manifest is None:
                raise OperationConflict(
                    "Storage recovery evidence requires a committed model manifest."
                )
            storage_recovery_operation = (
                self._storage_recovery_operation_factory(
                    storage_recovery_receipt,
                    state.manifest,
                )
            )
        with (
            self._recovery_operation_factory(
                transition_owner_id,
            ) as recovery_acknowledgement,
            self._slot_store.operation(),
        ):
            previous = self._slot_store.snapshot()
            committed = False
            try:
                if storage_recovery_operation is None:
                    result = self._commit(
                        state,
                        expected_revision=expected_revision,
                        after_commit=after_commit,
                        previous=previous,
                    )
                    committed = True
                else:
                    with storage_recovery_operation as proof:
                        result = self._commit(
                            state,
                            expected_revision=expected_revision,
                            after_commit=after_commit,
                            previous=previous,
                        )
                        committed = True
                        recovery_acknowledgement.acknowledge_storage_recovery(
                            proof
                        )
            except Exception:
                if committed:
                    self._slot_store.publish(previous)
                raise
            return result

    def clear(
        self,
        *,
        expected_revision: int,
        transition_owner_id: str | None = None,
    ) -> None:
        """Clear a healthy active slot while excluding execution and recovery."""

        with (
            self._administration_operation_factory(transition_owner_id),
            self._slot_store.operation(),
        ):
            previous = self._slot_store.snapshot()
            cleared = replace(
                previous,
                model=None,
                manifest=None,
                framework_uuid=None,
                session_name=None,
                checkpoint_name=None,
                revision=expected_revision,
            )
            self._commit(
                cleared,
                expected_revision=expected_revision,
                previous=previous,
            )

    def _commit(
        self,
        state: RuntimeSlotState,
        *,
        expected_revision: int,
        after_commit: Callable[[], None] | None = None,
        previous: RuntimeSlotState | None = None,
    ) -> BaseModelRuntime | None:
        """Commit the operation owned by this runtime slot commit service."""
        previous_state = previous or self._slot_store.snapshot()
        if previous_state.revision != expected_revision:
            raise OperationCancelled(
                "Runtime selection changed while the candidate was loading."
            )

        committed = replace(state, revision=expected_revision + 1)
        try:
            self._slot_store.publish(committed)
            if after_commit is not None:
                after_commit()
        except Exception:
            self._slot_store.publish(previous_state)
            raise
        return committed.model
