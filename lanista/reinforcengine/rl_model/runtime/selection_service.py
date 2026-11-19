# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate runtime selection."""

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from lanista.reinforcengine.application_errors import PersistenceError
from lanista.reinforcengine.rl_model.metadata.configuration_schema import ConfigurationSchema
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.metadata.schema import CURRENT_METADATA_SCHEMA_VERSION
from lanista.reinforcengine.rl_model.model import RLModelRunningModes
from lanista.reinforcengine.rl_model.runtime.selection_catalog_service import (
    RuntimeSelectionCatalogService,
)
from lanista.reinforcengine.rl_model.runtime.selection_persistence_service import (
    RuntimeSelectionPersistenceService,
)


@dataclass(frozen=True, slots=True)
class LastSuccessfulRuntimeState:
    """Represent runtime state for last successful runtime."""

    model_name: str | None = None
    framework_uuid: str | None = None
    session_name: str | None = None
    checkpoint_name: str | None = None


@dataclass
class RuntimeSelection:
    """Carry the data required for runtime selection."""

    session_name: str | None = None
    checkpoint_name: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeSelectionUpdate:
    """Carry the data required for runtime selection update."""

    session_name: str | None = None
    checkpoint_name: str | None = None
    active_framework_uuid: str | None = None


@dataclass
class TensorboardRestartContext:
    """Carry the runtime context for tensorboard restart."""

    mode: RLModelRunningModes
    framework_uuid: str
    selected_session: str | None


class RuntimeSelectionService:
    """Coordinate runtime selection operations."""

    def __init__(
        self,
        *,
        runtime_state_path_provider: Callable[[], str],
        framework_normalizer: Callable[[str | None], str | None],
        models_root_provider: Callable[[], str] | None = None,
        last_successful_state_provider: (Callable[[], LastSuccessfulRuntimeState] | None) = None,
        last_successful_state_committer: (
            Callable[[LastSuccessfulRuntimeState], None] | None
        ) = None,
        manifest_factory: Callable[[str], ModelManifest | None] | None = None,
        selection_catalog: RuntimeSelectionCatalogService | None = None,
    ) -> None:
        """Initialize the runtime selection service and its owned state."""
        self._runtime_state_path_provider = runtime_state_path_provider
        self._framework_normalizer = framework_normalizer
        self._models_root_provider = models_root_provider
        self._manifest_factory = manifest_factory
        self._selection_catalog = selection_catalog
        self._local_last_successful_state = LastSuccessfulRuntimeState()
        self._last_successful_state_provider = (
            last_successful_state_provider or self._provide_local_last_successful_state
        )
        self._last_successful_state_committer = (
            last_successful_state_committer or self._commit_local_last_successful_state
        )
        self._persistence_service = RuntimeSelectionPersistenceService(
            runtime_state_path_provider=runtime_state_path_provider,
            models_root_provider=models_root_provider,
        )

    def _provide_local_last_successful_state(self) -> LastSuccessfulRuntimeState:
        """Return the provide local last successful state value owned by this runtime selection
        service.
        """
        return self._local_last_successful_state

    def _commit_local_last_successful_state(
        self,
        state: LastSuccessfulRuntimeState,
    ) -> None:
        """Commit the local last successful state owned by this runtime selection service."""
        self._local_last_successful_state = state

    def _current_last_successful_state(self) -> LastSuccessfulRuntimeState:
        """Return the current last successful state value owned by this runtime selection service.
        """
        state = self._last_successful_state_provider()
        return LastSuccessfulRuntimeState(
            model_name=self._normalize_optional_text(state.model_name),
            framework_uuid=(self._framework_normalizer(state.framework_uuid) or None),
            session_name=self._normalize_optional_text(state.session_name),
            checkpoint_name=self._normalize_optional_text(state.checkpoint_name),
        )

    def load_last_successful(self) -> LastSuccessfulRuntimeState:
        """Load the last successful owned by this runtime selection service."""
        payload = self._persistence_service.load_last_successful()
        if payload is None:
            resolved = LastSuccessfulRuntimeState()
        else:
            state = payload["runtime_state"]
            resolved = LastSuccessfulRuntimeState(
                model_name=self._normalize_optional_text(state.get("last_successful_model_name")),
                framework_uuid=(
                    self._framework_normalizer(state.get("last_successful_framework_uuid")) or None
                ),
                session_name=self._normalize_optional_text(
                    state.get("selected_session_name")
                ),
                checkpoint_name=self._normalize_optional_text(
                    state.get("selected_checkpoint_name")
                ),
            )
        self._last_successful_state_committer(resolved)
        return resolved

    def save_last_successful(self, state: LastSuccessfulRuntimeState) -> None:
        """Persist the last successful owned by this runtime selection service."""
        self.persist_last_successful_runtime_state(
            model_name=state.model_name,
            framework_uuid=state.framework_uuid,
            session_name=state.session_name,
            checkpoint_name=state.checkpoint_name,
        )

    def persist_last_successful_runtime_state(
        self,
        *,
        model_name: str | None,
        framework_uuid: str | None,
        session_name: str | None = None,
        checkpoint_name: str | None = None,
    ) -> None:
        """Persist the last successful runtime state owned by this runtime
selection service."""
        state = LastSuccessfulRuntimeState(
            model_name=self._normalize_optional_text(model_name),
            framework_uuid=self._framework_normalizer(framework_uuid) or None,
            session_name=self._normalize_optional_text(session_name),
            checkpoint_name=self._normalize_optional_text(checkpoint_name),
        )
        self._persist_state(state)

    def persist_loaded_runtime_state(
        self,
        *,
        model_name: str | None,
        framework_uuid: str | None,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> None:
        """Persist state already published inside a rollback-capable slot commit."""

        self._save_state(
            LastSuccessfulRuntimeState(
                model_name=self._normalize_optional_text(model_name),
                framework_uuid=self._framework_normalizer(framework_uuid) or None,
                session_name=self._normalize_optional_text(session_name),
                checkpoint_name=self._normalize_optional_text(checkpoint_name),
            )
        )

    def _persist_state(self, state: LastSuccessfulRuntimeState) -> None:
        """Persist the state owned by this runtime selection service."""
        previous = self._current_last_successful_state()
        self._save_state(state)
        try:
            self._last_successful_state_committer(state)
        except Exception as commit_error:
            try:
                self._save_state(previous)
            except Exception as rollback_error:
                raise PersistenceError(
                    "Runtime state cache commit and storage rollback both failed."
                ) from rollback_error
            raise PersistenceError("Runtime state cache commit failed.") from commit_error

    def _save_state(self, state: LastSuccessfulRuntimeState) -> None:
        """Persist the state owned by this runtime selection service."""
        self._persistence_service.persist_last_successful_runtime_state(
            model_name=state.model_name,
            framework_uuid=state.framework_uuid,
            session_name=state.session_name,
            checkpoint_name=state.checkpoint_name,
        )

    def clear_last_successful(self) -> None:
        """Clear the last successful owned by this runtime selection service."""
        self.clear_last_successful_runtime_state()

    def clear_last_successful_runtime_state(self) -> None:
        """Clear the last successful runtime state owned by this runtime selection service."""
        self._persist_state(LastSuccessfulRuntimeState())

    def get_last_successful_framework_uuid(self) -> str | None:
        """Return the cached framework, falling back to the saved model manifest."""

        state = self._current_last_successful_state()
        return self.resolve_last_successful_framework_uuid(
            last_successful_framework_uuid=state.framework_uuid,
            last_successful_model_manifest_provider=(self.get_last_successful_model_manifest),
        )

    def get_last_successful_model_manifest(self) -> ModelManifest | None:
        """Resolve the cached model and clear both state copies if it is stale."""

        state = self._current_last_successful_state()
        if not state.model_name:
            return None
        if self._models_root_provider is None or self._manifest_factory is None:
            raise RuntimeError(
                "Last-successful manifest resolution requires model storage "
                "and manifest factory dependencies."
            )
        return self.resolve_last_successful_model_manifest(
            last_successful_model_name=state.model_name,
            models_root_provider=self._models_root_provider,
            manifest_provider=self._manifest_factory,
            clear_last_successful_callback=(self.clear_last_successful_runtime_state),
        )

    @staticmethod
    def _normalize_optional_text(value: str | None) -> str | None:
        """Normalize the optional text accepted by this runtime selection service."""
        text = str(value or "").strip()
        return text or None

    def load_model_configuration_selection(self, configuration: object) -> RuntimeSelection:
        """Read legacy schema-v1 selection defaults without mutating the model."""

        schema_version = getattr(
            configuration,
            "model_configuration_schema_version",
            None,
        )
        try:
            if isinstance(configuration, Mapping):
                schema = ConfigurationSchema.from_model_configuration_payload(
                    configuration,
                )
            elif type(schema_version) is int and schema_version == CURRENT_METADATA_SCHEMA_VERSION:
                schema_source = getattr(configuration, "model_configuration", configuration)
                schema = ConfigurationSchema.from_configuration(schema_source)
            else:
                raise ValueError(f"Unsupported model configuration schema: {schema_version!r}.")
        except ValueError as error:
            raise PersistenceError(
                "Model configuration is unreadable or uses an unsupported schema. "
                "Run the metadata migration before loading the model."
            ) from error
        return RuntimeSelection(
            session_name=self._normalize_optional_text(
                getattr(configuration, "selected_default_model_name", None)
                or schema.selected_default_model_name
            ),
            checkpoint_name=self._normalize_optional_text(
                getattr(configuration, "selected_default_checkpoint_name", None)
                or schema.selected_default_checkpoint_name
            ),
        )

    def load_staged_selection(
        self,
        model_manifest: ModelManifest,
    ) -> RuntimeSelection:
        """Restore a guarded local selection, then fall back to legacy defaults."""

        catalog = self._require_selection_catalog()
        configuration = model_manifest.get_configuration()
        model_name = catalog.model_name(model_manifest)
        cached = self._current_last_successful_state()
        use_local_state = bool(
            model_name
            and cached.model_name == model_name
            and cached.framework_uuid
        )
        if use_local_state:
            framework_uuid = catalog.supported_framework(
                model_manifest,
                cached.framework_uuid,
            )
            if framework_uuid is None:
                return RuntimeSelection()
            candidate = RuntimeSelection(
                session_name=cached.session_name,
                checkpoint_name=cached.checkpoint_name,
            )
        else:
            framework_uuid = self.resolve_manifest_bootstrap_framework_uuid(
                model_manifest
            )
            candidate = self.load_model_configuration_selection(configuration)

        session_name, checkpoint_name = catalog.validate_selection(
            model_manifest,
            framework_uuid=framework_uuid,
            session_name=candidate.session_name,
            checkpoint_name=candidate.checkpoint_name,
        )
        return RuntimeSelection(
            session_name=session_name,
            checkpoint_name=checkpoint_name,
        )

    def persist_active_runtime_selection(
        self,
        *,
        model_manifest: ModelManifest,
        update: RuntimeSelectionUpdate,
    ) -> None:
        """Persist one complete model/framework/session/checkpoint association."""

        catalog = self._require_selection_catalog()
        model_name = catalog.model_name(model_manifest)
        framework_uuid = catalog.supported_framework(
            model_manifest,
            update.active_framework_uuid,
        )
        if model_name is None or framework_uuid is None:
            raise PersistenceError(
                "The active model or framework identity is no longer valid."
            )
        self.persist_last_successful_runtime_state(
            model_name=model_name,
            framework_uuid=framework_uuid,
            session_name=update.session_name,
            checkpoint_name=update.checkpoint_name,
        )

    def _require_selection_catalog(self) -> RuntimeSelectionCatalogService:
        """Return the required selection catalog for this runtime selection service."""
        if self._selection_catalog is None:
            raise RuntimeError("Runtime selection catalog is not configured.")
        return self._selection_catalog

    @staticmethod
    def normalize_selection(
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> RuntimeSelection:
        """Normalize the selection accepted by this runtime selection service."""
        return RuntimeSelection(
            session_name=str(session_name or "").strip() or None,
            checkpoint_name=str(checkpoint_name or "").strip() or None,
        )

    def resolve_active_framework(
        self,
        *,
        configuration: object,
        current_runtime_framework_uuid: str | None,
    ) -> str:
        """Resolve the active framework required by this runtime selection service."""
        schema_source = getattr(configuration, "model_configuration", configuration)
        schema = ConfigurationSchema.from_configuration(schema_source)
        configured_framework_uuid = self._framework_normalizer(
            getattr(configuration, "active_runtime_framework_uuid", None)
        )
        return (
            self._framework_normalizer(current_runtime_framework_uuid)
            or self._framework_normalizer(schema.preferred_framework_uuid)
            or configured_framework_uuid
            or self._framework_normalizer(schema.active_runtime_framework_uuid)
            or "ray"
        )

    def resolve_current_model_framework_uuid(
        self,
        *,
        configuration: object | None,
        current_runtime_framework_uuid: str | None,
    ) -> str | None:
        """Resolve the current model framework uuid required by this runtime selection service."""
        normalized_current_framework_uuid = self._framework_normalizer(
            current_runtime_framework_uuid
        )
        if normalized_current_framework_uuid:
            return normalized_current_framework_uuid

        if configuration is None:
            return None

        preferred_framework_uuid = self._framework_normalizer(
            getattr(configuration, "preferred_framework_uuid", None)
        )
        if preferred_framework_uuid:
            return preferred_framework_uuid

        configured_framework_uuid = self._framework_normalizer(
            getattr(configuration, "active_runtime_framework_uuid", None)
        )
        if configured_framework_uuid:
            return configured_framework_uuid

        engine_framework_uuid = self._framework_normalizer(
            getattr(getattr(configuration, "rl_engine_framework", None), "uuid", None)
        )
        return engine_framework_uuid or None

    def build_tensorboard_restart_context(
        self,
        *,
        mode: RLModelRunningModes,
        selected_session: str | None,
        framework_uuid: str | None = None,
        configuration: object | None = None,
        current_runtime_framework_uuid: str | None = None,
    ) -> TensorboardRestartContext | None:
        """Build the tensorboard restart context required by this runtime selection service."""
        resolved_framework_uuid = (
            self._framework_normalizer(framework_uuid)
            or self.resolve_current_model_framework_uuid(
                configuration=configuration,
                current_runtime_framework_uuid=current_runtime_framework_uuid,
            )
            or ""
        )
        if not resolved_framework_uuid:
            return None

        return TensorboardRestartContext(
            mode=mode,
            framework_uuid=resolved_framework_uuid,
            selected_session=selected_session,
        )

    def resolve_manifest_bootstrap_framework_uuid(
        self, model_manifest: ModelManifest | None
    ) -> str | None:
        """Resolve the manifest bootstrap framework uuid required by this runtime selection
        service.
        """
        if model_manifest is None:
            return None

        bootstrap_framework_uuid = getattr(model_manifest, "get_bootstrap_framework_uuid", None)
        if callable(bootstrap_framework_uuid):
            resolved_framework_uuid = self._framework_normalizer(bootstrap_framework_uuid())
            if resolved_framework_uuid:
                return resolved_framework_uuid

        configuration = model_manifest.get_configuration()
        return (
            self._framework_normalizer(
                getattr(configuration, "preferred_framework_uuid", None)
            )
            or self._framework_normalizer(
                getattr(configuration, "active_runtime_framework_uuid", None)
            )
            or self._framework_normalizer(
                getattr(getattr(configuration, "rl_engine_framework", None), "uuid", None)
            )
            or None
        )

    def resolve_last_successful_framework_uuid(
        self,
        *,
        last_successful_framework_uuid: str | None,
        last_successful_model_manifest_provider: Callable[[], ModelManifest | None],
    ) -> str | None:
        """Resolve the last successful framework uuid required by this runtime selection service."""
        if last_successful_framework_uuid:
            return self._framework_normalizer(last_successful_framework_uuid) or None

        model_manifest = last_successful_model_manifest_provider()
        return self.resolve_manifest_bootstrap_framework_uuid(model_manifest)

    def resolve_last_successful_model_manifest(
        self,
        *,
        last_successful_model_name: str | None,
        models_root_provider: Callable[[], str],
        manifest_provider: Callable[[str], ModelManifest | None],
        clear_last_successful_callback: Callable[[], None],
    ) -> ModelManifest | None:
        """Resolve the last successful model manifest required by this runtime selection service."""
        model_folder_name = str(last_successful_model_name or "").strip()
        if not model_folder_name:
            return None

        model_folder = os.path.join(models_root_provider(), model_folder_name)
        if not os.path.isdir(model_folder):
            clear_last_successful_callback()
            return None

        model_manifest = manifest_provider(model_folder)
        if model_manifest is not None:
            model_manifest.load()
            model_manifest.ensure_valid()
            return model_manifest

        clear_last_successful_callback()
        return None

    def prepare_runtime_selection(
        self,
        *,
        model_manifest: ModelManifest,
        session_name: str | None,
        checkpoint_name: str | None,
        current_runtime_framework_uuid: str | None,
    ) -> RuntimeSelectionUpdate:
        """Resolve and validate a selection without mutating storage."""

        selection = self.normalize_selection(session_name, checkpoint_name)
        active_framework_uuid = self.resolve_active_framework(
            configuration=model_manifest.get_configuration(),
            current_runtime_framework_uuid=current_runtime_framework_uuid,
        )
        catalog = self._require_selection_catalog()
        framework_uuid = catalog.supported_framework(
            model_manifest,
            active_framework_uuid,
        )
        if framework_uuid is None:
            raise PersistenceError("The selected framework is not available for this model.")
        validated_session, validated_checkpoint = catalog.validate_selection(
            model_manifest,
            framework_uuid=framework_uuid,
            session_name=selection.session_name,
            checkpoint_name=selection.checkpoint_name,
        )
        if selection.session_name and validated_session is None:
            raise PersistenceError("The selected session is no longer available.")
        if selection.checkpoint_name and validated_checkpoint is None:
            raise PersistenceError("The selected checkpoint is no longer available.")

        return RuntimeSelectionUpdate(
            session_name=validated_session,
            checkpoint_name=validated_checkpoint,
            active_framework_uuid=framework_uuid,
        )
