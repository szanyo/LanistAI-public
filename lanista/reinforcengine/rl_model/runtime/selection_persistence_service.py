# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate runtime selection persistence."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from yaml import YAMLError

from lanista.reinforcengine.application_errors import PersistenceError
from lanista.reinforcengine.rl_model.metadata.schema import (
    CURRENT_METADATA_SCHEMA_VERSION,
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.storage.mutation_lock import StorageMutationLock
from lanista.reinforcengine.rl_model.storage.mutation_scope import (
    ModelsStorageMutationScope,
    StorageMutationLockFactory,
)
from lanista.reinforcengine.rl_model.storage.persistence import (
    atomic_write_validated_yaml_file,
    read_yaml_mapping_file,
)


class RuntimeSelectionPersistenceService:
    """Coordinate runtime selection persistence operations."""

    def __init__(
        self,
        *,
        runtime_state_path_provider: Callable[[], str],
        models_root_provider: Callable[[], str] | None = None,
        lock_factory: StorageMutationLockFactory = StorageMutationLock,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the runtime selection persistence service and its owned state."""
        self._runtime_state_path_provider = runtime_state_path_provider
        self._models_root_provider = models_root_provider
        self._lock_factory = lock_factory
        self._lock_timeout_seconds = lock_timeout_seconds

    def _storage_mutation(
        self,
    ) -> AbstractContextManager[object]:
        """Return the storage mutation value owned by this runtime selection persistence service."""
        if self._models_root_provider is not None:
            models_root = self._models_root_provider()
            scope = ModelsStorageMutationScope.from_models_root(models_root)
        else:
            # Isolated consumers without model-storage context cannot
            # coordinate the application-level runtime-state file.
            return nullcontext()
        return scope.lock(
            lock_factory=self._lock_factory,
            timeout_seconds=self._lock_timeout_seconds,
        )

    def load_last_successful(self) -> dict[str, Any] | None:
        """Read a canonical runtime-state document, if one exists."""

        path = self._runtime_state_path_provider()
        if not os.path.isfile(path):
            return None
        try:
            payload = read_yaml_mapping_file(path)
            return validate_versioned_metadata_payload(
                MetadataKind.RUNTIME_STATE,
                payload,
            )
        except (OSError, ValueError, YAMLError) as error:
            raise PersistenceError(
                "Runtime selection state is unreadable or uses an unsupported "
                "schema. Run the metadata migration before starting the application."
            ) from error

    def save_last_successful(
        self,
        *,
        model_name: str | None,
        framework_uuid: str | None,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> None:
        """Persist the last successful owned by this runtime selection persistence service."""
        with self._storage_mutation():
            path = self._runtime_state_path_provider()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            payload = {
                "schema_version": CURRENT_METADATA_SCHEMA_VERSION,
                "runtime_state": {
                    "last_successful_model_name": model_name or "",
                    "last_successful_framework_uuid": framework_uuid or "",
                    "selected_session_name": session_name or "",
                    "selected_checkpoint_name": checkpoint_name or "",
                },
            }
            atomic_write_validated_yaml_file(
                path,
                payload,
                validator=lambda candidate: validate_versioned_metadata_payload(
                    MetadataKind.RUNTIME_STATE,
                    candidate,
                ),
                verify_reader=read_yaml_mapping_file,
                smoke_check_reader=read_yaml_mapping_file,
            )

    def persist_last_successful_runtime_state(
        self,
        *,
        model_name: str | None,
        framework_uuid: str | None,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> None:
        """Persist the last successful runtime state owned by this runtime selection persistence
        service.
        """
        self.save_last_successful(
            model_name=model_name,
            framework_uuid=framework_uuid,
            session_name=session_name,
            checkpoint_name=checkpoint_name,
        )

    def clear_last_successful(self) -> None:
        """Clear the last successful owned by this runtime selection persistence service."""
        self.save_last_successful(
            model_name=None,
            framework_uuid=None,
            session_name=None,
            checkpoint_name=None,
        )

    def clear_last_successful_runtime_state(self) -> None:
        """Clear the last successful runtime state owned by this runtime selection persistence
        service.
        """
        self.clear_last_successful()
