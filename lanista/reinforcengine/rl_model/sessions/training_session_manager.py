# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate the lifecycle of training-session manifests."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from datetime import datetime

import yaml

from lanista.foundation.configuration import Configuration
from lanista.reinforcengine.rl_model.sessions.configuration_wrapper import (
    SessionConfigurationWrapper,
)
from lanista.reinforcengine.rl_model.sessions.repository_contracts import SessionMetadataUpdate
from lanista.reinforcengine.rl_model.storage.layout import (
    SESSION_MANIFEST_FILE_NAME,
    TRAINING_SESSIONS_DIR_NAME,
    find_existing_session_manifest_path,
)
from lanista.reinforcengine.rl_model.storage.mutation_lock import StorageMutationLock
from lanista.reinforcengine.rl_model.storage.mutation_scope import (
    ModelsStorageMutationScope,
    StorageMutationLockFactory,
)


class TrainingSessionManager:
    """Manage session manifests below a model's ``training_sessions`` root."""

    def __init__(
        self,
        model_folder_path: str,
        sessions_dir_name: str = TRAINING_SESSIONS_DIR_NAME,
        session_file_name: str = SESSION_MANIFEST_FILE_NAME,
        *,
        models_root_provider: Callable[[], str] | None = None,
        storage_lock_already_held: bool = False,
        lock_factory: StorageMutationLockFactory = StorageMutationLock,
        lock_timeout_seconds: float = 5.0,
    ):
        """Initialize the training session manager and its owned state."""
        self._model_folder_path = str(model_folder_path)
        self._sessions_dir_name = str(sessions_dir_name)
        self._session_file_name = str(session_file_name)
        self._models_root_provider = models_root_provider
        self._storage_lock_already_held = bool(storage_lock_already_held)
        self._lock_factory = lock_factory
        self._lock_timeout_seconds = lock_timeout_seconds
        if self._models_root_provider is None and not self._storage_lock_already_held:
            raise ValueError(
                "A models-root provider or an explicit enclosing storage lock "
                "is required for session writers."
            )

    def _storage_mutation(self) -> AbstractContextManager[object]:
        """Return the root lock or reuse an explicitly declared outer owner."""
        if self._storage_lock_already_held:
            return nullcontext()
        if self._models_root_provider is None:  # pragma: no cover - constructor invariant
            raise RuntimeError("Session storage lock context is unavailable.")
        return ModelsStorageMutationScope.for_model_root(
            self._model_folder_path,
            models_root=self._models_root_provider(),
        ).lock(
            lock_factory=self._lock_factory,
            timeout_seconds=self._lock_timeout_seconds,
        )

    def get_sessions_root(self) -> str:
        """Return the sessions root exposed by this training session manager."""
        return os.path.join(self._model_folder_path, self._sessions_dir_name)

    def get_session_path(self, session_uuid: str) -> str:
        """Return the session path exposed by this training session manager."""
        return os.path.join(self.get_sessions_root(), str(session_uuid))

    def get_session_config_path(self, session_uuid: str) -> str:
        """Return the session config path exposed by this training session manager."""
        session_path = self.get_session_path(session_uuid)
        return find_existing_session_manifest_path(session_path)

    def ensure_session_structure(self, session_uuid: str) -> None:
        """Ensure the session structure owned by this training session manager."""
        with self._storage_mutation():
            self._ensure_session_structure_locked(session_uuid)

    def _ensure_session_structure_locked(self, session_uuid: str) -> None:
        """Ensure the session structure locked owned by this training session manager."""
        session_path = self.get_session_path(session_uuid)
        os.makedirs(os.path.join(session_path, "checkpoints"), exist_ok=True)

    def list_sessions(self) -> list[str]:
        """Return the sessions available to this training session manager."""
        sessions_root = self.get_sessions_root()
        if not os.path.isdir(sessions_root):
            return []

        sessions: list[str] = []
        session_sort_values: dict[str, float] = {}
        for entry in os.scandir(sessions_root):
            if not entry.is_dir():
                continue
            manifest_path = self.get_session_config_path(entry.name)
            if not os.path.isfile(manifest_path):
                continue
            sessions.append(entry.name)
            session_sort_values[entry.name] = self._read_created_at_timestamp(manifest_path)

        sessions.sort(
            key=lambda s: (
                session_sort_values.get(s, float("-inf")),
                os.path.getmtime(self.get_session_path(s)),
                s,
            ),
            reverse=True,
        )
        return sessions

    @staticmethod
    def _read_created_at_timestamp(manifest_path: str) -> float:
        """Read the created at timestamp without changing this training session manager's state."""
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except Exception:
            return float("-inf")

        if not isinstance(payload, dict):
            return float("-inf")

        session_section = payload.get("session")
        created_at = str(payload.get("created_at") or "").strip()
        if not created_at and isinstance(session_section, dict):
            created_at = str(session_section.get("created_at") or "").strip()
        if not created_at:
            return float("-inf")

        try:
            return datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return float("-inf")

    def load_session_configuration(self, session_uuid: str) -> Configuration:
        """Load a session manifest or propagate its exact read/validation error."""

        wrapper = SessionConfigurationWrapper(self.get_session_config_path(session_uuid))
        if not wrapper.exists():
            raise FileNotFoundError(f"Session manifest not found for {session_uuid!r}.")
        return wrapper.load_configuration()

    def update_session_metadata(
        self,
        session_uuid: str,
        update: SessionMetadataUpdate,
    ) -> None:
        """Commit one typed metadata update or propagate the exact failure."""

        with self._storage_mutation():
            self._update_session_metadata_locked(session_uuid, update)

    def _update_session_metadata_locked(
        self,
        session_uuid: str,
        update: SessionMetadataUpdate,
    ) -> None:
        """Update the session metadata locked owned by this training session manager."""
        wrapper = SessionConfigurationWrapper(self.get_session_config_path(session_uuid))
        if not wrapper.exists():
            raise FileNotFoundError(f"Session manifest not found for {session_uuid!r}.")

        wrapper.load_configuration()

        changed = wrapper.apply_fields(update.to_storage_fields())
        if not changed:
            return

        wrapper.save()

    def create_session_configuration(
        self,
        session_uuid: str,
        initializer: dict[str, object],
    ) -> None:
        """Create one session manifest or propagate its exact storage failure."""

        with self._storage_mutation():
            self._ensure_session_structure_locked(session_uuid)
            session_path = self.get_session_config_path(session_uuid)
            wrapper = SessionConfigurationWrapper(session_path)
            wrapper.apply_fields(initializer)
            wrapper.save()
