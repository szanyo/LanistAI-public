# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Apply and atomically commit the active runtime selection."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from lanista.reinforcengine.application_errors import (
    ApplicationError,
    OperationConflict,
    OperationResult,
    PersistenceError,
    RetryPolicy,
    RuntimeUnavailableError,
    failure_from_exception,
)
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.runtime.selection_service import (
    RuntimeSelectionService,
    RuntimeSelectionUpdate,
)


@dataclass(frozen=True, slots=True)
class ActiveRuntimeSelectionRequest:
    """One ownership-checked session and checkpoint selection command."""

    session_name: str | None
    checkpoint_name: str | None
    framework_uuid: str | None = None
    persist: bool = False
    expected_model_uuid: str | None = None


class ActiveRuntimeSelectionLogger(Protocol):
    """Diagnostic boundary that keeps exception details out of results."""

    def warning(self, message: str) -> object:
        """Publish a warning record through this active runtime selection logger."""
        ...

    def exception(self, message: str) -> object:
        """Publish a exception record through this active runtime selection logger."""
        ...


class ActiveRuntimeSelectionService:
    """Coordinate local persistence with the in-memory selection state."""

    def __init__(
        self,
        *,
        active_manifest_provider: Callable[[], ModelManifest | None],
        runtime_selection_service: RuntimeSelectionService,
        current_framework_provider: Callable[[], str | None],
        state_provider: Callable[[], RuntimeSelectionUpdate],
        state_committer: Callable[[RuntimeSelectionUpdate], None],
        administration_operation_factory: Callable[
            [], AbstractContextManager[object]
        ] = nullcontext,
        logger: ActiveRuntimeSelectionLogger | None = None,
        native_gui_mutation_guard: Callable[[], None] = lambda: None,
    ) -> None:
        """Initialize the active runtime selection service and its owned state."""
        self._active_manifest_provider = active_manifest_provider
        self._runtime_selection_service = runtime_selection_service
        self._current_framework_provider = current_framework_provider
        self._state_provider = state_provider
        self._state_committer = state_committer
        self._administration_operation_factory = administration_operation_factory
        self._logger = logger
        self._native_gui_mutation_guard = native_gui_mutation_guard

    def _warning(self, message: str) -> None:
        """Publish a warning record through this active runtime selection service."""
        if self._logger is None:
            return
        try:
            self._logger.warning(message)
        except Exception:
            # Diagnostic failure must not replace the correlated result.
            return

    def _exception(self, message: str) -> None:
        """Publish a exception record through this active runtime selection service."""
        if self._logger is None:
            return
        try:
            self._logger.exception(message)
        except Exception:
            # Diagnostic failure must not replace the correlated result.
            return

    def set_selection(
        self,
        request: ActiveRuntimeSelectionRequest,
    ) -> OperationResult[RuntimeSelectionUpdate]:
        """Apply one coherent session/checkpoint selection to the active model."""

        operation_id = uuid4().hex
        try:
            self._native_gui_mutation_guard()
            with self._administration_operation_factory():
                return self._set_selection_locked(request, operation_id)
        except ApplicationError as error:
            self._warning(
                "Active runtime selection failed with an expected error; "
                f"operation_id={operation_id}; error={error!r}."
            )
            return OperationResult.failed(
                failure_from_exception(
                    error,
                    operation_id=operation_id,
                    retry_policy=RetryPolicy.SAFE,
                )
            )
        except (OSError, TypeError, ValueError) as error:
            self._exception(
                "Active runtime selection persistence failed; "
                f"operation_id={operation_id}; error={error!r}."
            )
            persistence_error = PersistenceError("The runtime selection could not be saved.")
            return OperationResult.failed(
                failure_from_exception(
                    persistence_error,
                    operation_id=operation_id,
                    retry_policy=RetryPolicy.SAFE,
                )
            )
        except Exception as error:
            self._exception(
                "Unexpected active runtime selection failure; "
                f"operation_id={operation_id}; error={error!r}."
            )
            return OperationResult.failed(failure_from_exception(error, operation_id=operation_id))

    def _set_selection_locked(
        self,
        request: ActiveRuntimeSelectionRequest,
        operation_id: str,
    ) -> OperationResult[RuntimeSelectionUpdate]:
        """Set the selection locked owned by this active runtime selection service."""
        manifest = self._active_manifest_provider()
        if manifest is None:
            raise RuntimeUnavailableError("No active model is available.")

        configuration = manifest.get_configuration()
        if configuration is None:
            raise RuntimeUnavailableError("The active model configuration is unavailable.")

        expected_uuid = str(request.expected_model_uuid or "").strip()
        active_uuid = str(getattr(configuration, "model_uuid", "") or "").strip()
        if expected_uuid and active_uuid != expected_uuid:
            raise OperationConflict(
                "The active model changed before the runtime selection was saved."
            )

        requested_framework_uuid = str(request.framework_uuid or "").strip() or None
        current_framework_uuid = (
            requested_framework_uuid or self._current_framework_provider()
        )
        update = self._runtime_selection_service.prepare_runtime_selection(
            model_manifest=manifest,
            session_name=request.session_name,
            checkpoint_name=request.checkpoint_name,
            current_runtime_framework_uuid=current_framework_uuid,
        )
        previous_state = self._state_provider()
        self._state_committer(update)
        try:
            if request.persist:
                self._runtime_selection_service.persist_active_runtime_selection(
                    model_manifest=manifest,
                    update=update,
                )
        except Exception:
            self._state_committer(previous_state)
            raise
        return OperationResult.success(update, operation_id=operation_id)
