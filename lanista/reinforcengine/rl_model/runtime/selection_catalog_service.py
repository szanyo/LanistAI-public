# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Guarded read-side validation for restored runtime selections."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from lanista.reinforcengine.rl_model.checkpoints.catalog_repository import (
    CheckpointCatalogRepository,
)
from lanista.reinforcengine.rl_model.metadata.configuration_schema import ConfigurationSchema
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.sessions.training_session_manager import (
    TrainingSessionManager,
)
from lanista.reinforcengine.rl_model.storage.names import (
    CheckpointStorageName,
    InvalidStorageName,
    ModelStorageName,
    SessionStorageName,
)
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)


class RuntimeSelectionCatalogService:
    """Validate model, framework, session, and checkpoint identities together."""

    def __init__(
        self,
        *,
        models_root_provider: Callable[[], str],
        session_manager_factory: Callable[[str], TrainingSessionManager],
        framework_normalizer: Callable[[str | None], str | None],
    ) -> None:
        """Initialize the runtime selection catalog service and its owned state."""
        self._models_root_provider = models_root_provider
        self._session_manager_factory = session_manager_factory
        self._framework_normalizer = framework_normalizer

    def model_name(self, manifest: ModelManifest) -> str | None:
        """Return a guarded direct-child model identity."""

        model_folder = self._model_folder(manifest)
        if model_folder is None:
            return None
        try:
            return ModelStorageName.parse(model_folder.name).value
        except InvalidStorageName:
            return None

    def supported_framework(
        self,
        manifest: ModelManifest,
        framework_uuid: str | None,
    ) -> str | None:
        """Return a normalized framework only when its model resource exists."""

        normalized = self._framework_normalizer(framework_uuid)
        model_folder = self._model_folder(manifest)
        if not normalized or model_folder is None:
            return None
        try:
            configuration = manifest.get_configuration()
            schema_source = getattr(
                configuration,
                "model_configuration",
                configuration,
            )
            schema = ConfigurationSchema.from_configuration(
                schema_source,
                model_folder_path=str(model_folder),
            )
            resource_path = schema.framework_resource_paths_as_dict.get(normalized)
            if not resource_path:
                return None
            guarded_resource = StoragePathGuard(model_folder).require_descendant(
                resource_path,
                must_exist=True,
            )
            return normalized if guarded_resource.is_dir() else None
        except (OSError, ValueError, StoragePathViolation):
            return None

    def validate_selection(
        self,
        manifest: ModelManifest,
        *,
        framework_uuid: str | None,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> tuple[str | None, str | None]:
        """Fail closed when a restored session or checkpoint is not catalogued."""

        framework = self.supported_framework(manifest, framework_uuid)
        model_folder = self._model_folder(manifest)
        if framework is None or model_folder is None or not session_name:
            return None, None
        try:
            session = SessionStorageName.parse(session_name).value
            manager = self._session_manager_factory(str(model_folder))
            if session not in manager.list_sessions():
                return None, None
            if self._session_framework(manager, session) != framework:
                return None, None
            if not checkpoint_name:
                return session, None
            checkpoint = CheckpointStorageName.parse(checkpoint_name).value
            repository = self._checkpoint_repository(manager)
            if checkpoint not in repository.list_for_session(session, framework):
                return session, None
            return session, checkpoint
        except (FileNotFoundError, InvalidStorageName, OSError, ValueError):
            return None, None

    def _model_folder(self, manifest: ModelManifest) -> Path | None:
        """Return the model folder value owned by this runtime selection catalog service."""
        try:
            configuration = manifest.get_configuration()
            model_folder_path = str(
                getattr(configuration, "model_folder_path", "") or ""
            ).strip()
            guard = StoragePathGuard(self._models_root_provider())
            model_folder = guard.require_descendant(
                model_folder_path,
                must_exist=True,
            )
            return model_folder if model_folder.parent == guard.root else None
        except (OSError, ValueError, StoragePathViolation):
            return None

    def _session_framework(
        self,
        manager: TrainingSessionManager,
        session_name: str,
    ) -> str:
        """Return the session framework value owned by this runtime selection catalog service."""
        configuration = manager.load_session_configuration(session_name)
        return self._framework_normalizer(
            getattr(configuration, "framework_uuid", None)
        ) or ""

    def _checkpoint_repository(
        self,
        manager: TrainingSessionManager,
    ) -> CheckpointCatalogRepository:
        """Checkpoint the repository owned by this runtime selection catalog service."""
        return CheckpointCatalogRepository(
            sessions_root_provider=manager.get_sessions_root,
            session_framework_provider=(
                lambda session_name: self._session_framework(manager, session_name)
            ),
            framework_normalizer=self._framework_normalizer,
        )
