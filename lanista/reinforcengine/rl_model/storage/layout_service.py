# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate storage layout."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from lanista.reinforcengine.application_errors import ApplicationError, PersistenceError
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.storage.mutation_lock import StorageMutationLock
from lanista.reinforcengine.rl_model.storage.mutation_scope import (
    ModelsStorageMutationScope,
    StorageMutationLockFactory,
)
from lanista.reinforcengine.rl_model.storage.transaction import (
    StorageTransaction,
    StorageTransactionError,
)


class StorageLayoutDegradedStateReporter(Protocol):
    """Latch the shared administration safety boundary after partial mutation."""

    def __call__(
        self,
        *,
        reload_required: bool,
        storage_recovery_required: bool,
    ) -> None:
        """Invoke this storage layout degraded state reporter as a callable."""
        ...


StorageLayoutFailureInjector = Callable[[str], None]


@dataclass(slots=True)
class _StorageLayoutMutationState:
    """Track whether a failed preparation may have changed durable storage."""

    phase: str = "preflight"
    mutation_may_have_occurred: bool = False


class StorageLayoutService:
    """Recover model storage and ensure the expected TensorBoard layout."""

    def __init__(
        self,
        *,
        tensorboard_structure_ensurer: Callable[[str], None],
        models_root_provider: Callable[[], str] | None = None,
        logger: object | None = None,
        degraded_state_reporter: StorageLayoutDegradedStateReporter | None = None,
        failure_injector: StorageLayoutFailureInjector | None = None,
        lock_factory: StorageMutationLockFactory = StorageMutationLock,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the storage layout service and its owned state."""
        self._tensorboard_structure_ensurer = tensorboard_structure_ensurer
        self._models_root_provider = models_root_provider
        self._logger = logger
        self._degraded_state_reporter = degraded_state_reporter
        self._failure_injector = failure_injector
        self._lock_factory = lock_factory
        self._lock_timeout_seconds = lock_timeout_seconds

    def _warning(self, message: str) -> None:
        """Publish a warning record through this storage layout service."""
        if self._logger is not None and hasattr(self._logger, "warning"):
            self._logger.warning(message)

    def _exception(self, error: Exception) -> None:
        """Publish a exception record through this storage layout service."""
        if self._logger is not None and hasattr(self._logger, "exception"):
            self._logger.exception(error)

    def _inject(self, point: str) -> None:
        """Inject the operation owned by this storage layout service."""
        if self._failure_injector is not None:
            self._failure_injector(point)

    @staticmethod
    def _resolve_model_folder_path(manifest: ModelManifest) -> str:
        """Resolve the model folder path required by this storage layout service."""
        configuration = getattr(manifest, "get_configuration", None)
        if not callable(configuration):
            return ""

        cfg = configuration()
        if cfg is None:
            return ""

        return str(getattr(cfg, "model_folder_path", "") or "").strip()

    def ensure_manifest_storage_layout(self, manifest: ModelManifest | None) -> None:
        """Ensure the manifest storage layout owned by this storage layout service."""
        if manifest is None:
            return

        state = _StorageLayoutMutationState()
        try:
            model_folder_path = self._resolve_model_folder_path(manifest)
            if not model_folder_path:
                raise ValueError("The model manifest has no storage root.")
            if not os.path.isdir(model_folder_path):
                raise FileNotFoundError("The model storage root is unavailable.")

            explicit_models_root = (
                self._models_root_provider() if self._models_root_provider is not None else None
            )
            scope = ModelsStorageMutationScope.for_model_root(
                model_folder_path,
                models_root=explicit_models_root,
            )
            with scope.lock(
                lock_factory=self._lock_factory,
                timeout_seconds=self._lock_timeout_seconds,
            ):
                ModelsStorageMutationScope.for_model_root(
                    model_folder_path,
                    models_root=scope.models_root,
                )
                self._ensure_manifest_storage_layout_locked(
                    manifest,
                    model_folder_path,
                    state,
                )
        except Exception as error:
            self._raise_preparation_failure(error, state)

    def _raise_preparation_failure(
        self,
        error: Exception,
        state: _StorageLayoutMutationState,
    ) -> None:
        """Translate preparation failures after releasing the storage lock."""

        recovery_required = state.mutation_may_have_occurred
        if recovery_required and self._degraded_state_reporter is not None:
            self._degraded_state_reporter(
                reload_required=True,
                storage_recovery_required=True,
            )
        self._warning(
            "storage_layout_preparation_failed "
            f"phase={state.phase} "
            f"storage_recovery_required={str(recovery_required).lower()} "
            f"error_type={type(error).__name__}"
        )
        self._exception(error)
        message = "Model storage layout could not be prepared safely."
        if recovery_required:
            message += " Storage recovery and model reload are required."
        raise PersistenceError(message) from error

    def _ensure_manifest_storage_layout_locked(
        self,
        manifest: ModelManifest,
        model_folder_path: str,
        state: _StorageLayoutMutationState,
    ) -> None:
        """Recover storage and ensure TensorBoard layout while model deletion is excluded."""

        state.phase = "before_storage_recovery"
        self._inject(state.phase)
        state.phase = "storage_recovery"
        try:
            recovery = StorageTransaction.recover(model_folder_path)
        except Exception:
            # Recovery can fail after replaying only part of an abandoned
            # journal, so the durable tree must be treated as uncertain.
            state.mutation_may_have_occurred = True
            raise
        if recovery.failed:
            state.mutation_may_have_occurred = True
            raise StorageTransactionError(
                "Storage recovery did not complete for every transaction."
            )

        state.mutation_may_have_occurred = bool(recovery.rolled_back or recovery.finalized)
        state.phase = "after_storage_recovery"
        self._inject(state.phase)

        if recovery.rolled_back or recovery.finalized:
            state.phase = "manifest_readback"
            try:
                manifest.load()
            except ApplicationError as error:
                raise StorageTransactionError(
                    "Recovered model metadata could not be reloaded."
                ) from error

        state.phase = "before_tensorboard_structure"
        self._inject(state.phase)
        state.phase = "tensorboard_structure"
        state.mutation_may_have_occurred = True
        self._tensorboard_structure_ensurer(model_folder_path)
        state.phase = "after_tensorboard_structure"
        self._inject(state.phase)
