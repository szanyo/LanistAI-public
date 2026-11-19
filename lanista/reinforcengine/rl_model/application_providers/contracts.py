# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed external dependencies and state operations for application providers."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol, cast

from lanista.foundation.configuration import Configuration
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.metadata.model_manifest_writer import ModelManifestWriter
from lanista.reinforcengine.rl_model.runtime.events import (
    ModelLoadProgressEvent,
    RuntimeEventPublisher,
)
from lanista.reinforcengine.rl_model.runtime.selection_service import (
    LastSuccessfulRuntimeState,
    RuntimeSelectionUpdate,
)
from lanista.reinforcengine.rl_model.runtime_slot_store import RuntimeSlotStore
from lanista.reinforcengine.rl_model.sessions.training_session_manager import (
    TrainingSessionManager,
)
from lanista.reinforcengine.rl_model.staged_model_selection_service import (
    StagedModelSelection,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    SESSION_MANIFEST_FILE_NAME,
    TRAINING_SESSIONS_DIR_NAME,
)


class ProviderLogger(Protocol):
    """Structured logger required by the provider graph."""

    name: str

    def info(self, message: str) -> None:
        """Publish a info record through this provider logger."""
        ...

    def warning(self, message: str) -> None:
        """Publish a warning record through this provider logger."""
        ...

    def error(self, message: str) -> None:
        """Publish a error record through this provider logger."""
        ...

    def debug(self, message: str) -> None:
        """Publish a debug record through this provider logger."""
        ...

    def exception(
        self,
        message: object,
        *,
        exc_info: object | None = None,
    ) -> None:
        """Publish a exception record through this provider logger."""
        ...


@dataclass(frozen=True, slots=True)
class ApplicationProviderContext:
    """Hold the external state and filesystem dependencies of the provider DAG."""

    logger: ProviderLogger
    slot_store: RuntimeSlotStore
    models_root_provider: Callable[[], str]
    runtime_state_path_provider: Callable[[], str]
    log_level_provider: Callable[[], str]
    manifest_cls: type[ModelManifest] = ModelManifest
    session_manager_cls: type[TrainingSessionManager] = TrainingSessionManager
    manifest_writer_factory: Callable[[], ModelManifestWriter] = ModelManifestWriter
    native_gui_mutation_guard: Callable[[], None] = lambda: None

    def __post_init__(self) -> None:
        """Reject incomplete dependency bundles before any service is constructed."""

        if not isinstance(self.slot_store, RuntimeSlotStore):
            raise TypeError("Application providers require one RuntimeSlotStore instance.")
        logger_name = getattr(self.logger, "name", None)
        if not isinstance(logger_name, str) or not logger_name.strip():
            raise TypeError("Application provider logger is missing: name.")
        for dependency_name in (
            "models_root_provider",
            "runtime_state_path_provider",
            "log_level_provider",
            "manifest_cls",
            "session_manager_cls",
            "manifest_writer_factory",
            "native_gui_mutation_guard",
        ):
            if not callable(getattr(self, dependency_name)):
                raise TypeError(f"Application provider dependency is not callable: {dependency_name}.")
        for method_name in ("info", "warning", "error", "debug", "exception"):
            if not callable(getattr(self.logger, method_name, None)):
                raise TypeError(f"Application provider logger is missing: {method_name}.")

    def loaded_manifest(self) -> ModelManifest | None:
        """Return the manifest owned by the active runtime, when one exists."""

        current_model = self.slot_store.snapshot().model
        if current_model is None:
            return None
        manifest_provider = getattr(current_model, "get_manifest", None)
        if not callable(manifest_provider):
            return None
        return cast(ModelManifest | None, manifest_provider())

    def active_manifest(self) -> ModelManifest | None:
        """Prefer the loaded runtime manifest over the staged selection."""

        return self.loaded_manifest() or self.slot_store.snapshot().manifest

    def active_configuration(self) -> Configuration | None:
        """Return the active manifest configuration without exposing the slot."""

        manifest = self.active_manifest()
        if manifest is None:
            return None
        return cast(Configuration | None, manifest.get_configuration())

    @staticmethod
    def normalize_framework_uuid(framework_uuid: str | None) -> str:
        """Normalize framework identifiers once for the complete provider graph."""

        return str(framework_uuid or "").strip().lower()

    def sessions_root(self) -> str:
        """Resolve the active model's managed training-session root."""

        configuration = self.active_configuration()
        if configuration is None:
            raise RuntimeError("No current model selected.")
        return os.path.join(
            str(configuration.model_folder_path),
            TRAINING_SESSIONS_DIR_NAME,
        )

    def session_manager_for_model(
        self,
        model_folder_path: str,
        *,
        storage_lock_already_held: bool = False,
    ) -> TrainingSessionManager:
        """Create a session manager with explicit root-lock ownership."""

        return self.session_manager_cls(
            model_folder_path=model_folder_path,
            sessions_dir_name=TRAINING_SESSIONS_DIR_NAME,
            session_file_name=SESSION_MANIFEST_FILE_NAME,
            models_root_provider=self.models_root_provider,
            storage_lock_already_held=storage_lock_already_held,
        )

    def active_session_manager(self) -> TrainingSessionManager | None:
        """Create a manager for the active model, or report that none is selected."""

        configuration = self.active_configuration()
        if configuration is None:
            return None
        model_folder_path = str(
            getattr(configuration, "model_folder_path", "") or ""
        )
        if not model_folder_path:
            return None
        return self.session_manager_for_model(model_folder_path)

    def save_minimal_model_manifest(self, configuration: Configuration) -> None:
        """Persist the minimal schema-v1 projection through its focused writer."""

        self.manifest_writer_factory().save_minimal_manifest(configuration)

    def clear_current_model_state(self) -> None:
        """Clear deleted-model state while retaining only the restart cache."""

        self.slot_store.update(
            lambda previous: replace(
                previous,
                model=None,
                manifest=None,
                framework_uuid=None,
                session_name=None,
                checkpoint_name=None,
                revision=previous.revision + 1,
            )
        )

    def commit_staged_selection(self, selection: StagedModelSelection) -> None:
        """Atomically project one staged manifest and its stored selection."""

        self.slot_store.update(
            lambda previous: replace(
                previous,
                manifest=selection.manifest,
                session_name=selection.session_name,
                checkpoint_name=selection.checkpoint_name,
                revision=previous.revision + 1,
            )
        )

    def active_selection_state(self) -> RuntimeSelectionUpdate:
        """Project the active slot into the runtime-selection DTO."""

        state = self.slot_store.snapshot()
        return RuntimeSelectionUpdate(
            session_name=state.session_name,
            checkpoint_name=state.checkpoint_name,
            active_framework_uuid=state.framework_uuid,
        )

    def commit_active_selection(self, selection: RuntimeSelectionUpdate) -> None:
        """Atomically publish one current session/checkpoint/framework selection."""

        self.slot_store.update(
            lambda previous: replace(
                previous,
                session_name=selection.session_name,
                checkpoint_name=selection.checkpoint_name,
                framework_uuid=selection.active_framework_uuid,
                revision=previous.revision + 1,
            )
        )

    def last_successful_runtime_state(self) -> LastSuccessfulRuntimeState:
        """Project the restart cache without reading mutable configuration."""

        state = self.slot_store.snapshot()
        return LastSuccessfulRuntimeState(
            model_name=state.last_successful_model_name,
            framework_uuid=state.last_successful_framework_uuid,
            session_name=state.last_successful_session_name,
            checkpoint_name=state.last_successful_checkpoint_name,
        )

    def commit_last_successful_runtime_state(
        self,
        state: LastSuccessfulRuntimeState,
    ) -> None:
        """Update restart metadata without changing the active-slot revision."""

        self.slot_store.update(
            lambda previous: replace(
                previous,
                last_successful_model_name=state.model_name,
                last_successful_framework_uuid=state.framework_uuid,
                last_successful_session_name=state.session_name,
                last_successful_checkpoint_name=state.checkpoint_name,
            )
        )

    @staticmethod
    def post_model_load_progress(
        publisher: RuntimeEventPublisher | None,
        activity: str,
        step: str,
        progress: float,
        total: float = 100.0,
    ) -> None:
        """Publish one typed progress event when a request supplied a sink."""

        if publisher is not None:
            publisher.publish(ModelLoadProgressEvent(activity, step, progress, total))

    def log_runtime_event_publish_failure(
        self,
        operation_id: str,
        error: Exception,
    ) -> None:
        """Record event delivery failure without altering runtime ownership."""

        self.logger.error(
            "runtime_event_publish_failed "
            f"operation_id={operation_id} error_type={type(error).__name__}"
        )
        self.logger.exception(error)
