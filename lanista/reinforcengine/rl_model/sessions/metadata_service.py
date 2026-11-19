# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate training-session metadata."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from lanista.foundation.configuration import Configuration
from lanista.reinforcengine.rl_model.empty_model_generator import EmptyModelGenerator
from lanista.reinforcengine.rl_model.name_formatting_service import NameFormattingService
from lanista.reinforcengine.rl_model.storage.mutation_lock import (
    StorageMutationLock,
    StorageMutationLockError,
)
from lanista.reinforcengine.rl_model.storage.mutation_scope import (
    ModelsStorageMutationScope,
    StorageMutationLockFactory,
)
from lanista.reinforcengine.rl_model.storage.path_guard import StoragePathViolation


class _SupportsSave(Protocol):
    """Define the typed boundary for supports save."""

    def save(self) -> None:
        """Persist the current state owned by this supports save."""
        ...


class _SessionPathPort(Protocol):
    """Define the typed boundary for session path."""

    def get_session_config_path(self, session_name: str) -> str:
        """Return the session config path exposed by this session path port."""
        ...


class _WarningLogger(Protocol):
    """Define the typed boundary for warning logger."""

    def warning(self, message: str) -> object:
        """Publish a warning record through this warning logger."""
        ...


@dataclass
class SessionChainNode:
    """Carry the data required for session chain node."""

    session_uuid: str
    display_name: str
    parent_session_uuid: str | None
    session_description: str
    is_current: bool


class SessionMetadataService:
    """Coordinate session metadata operations."""

    def __init__(
        self,
        *,
        active_configuration_provider: Callable[[], Configuration | None],
        session_configuration_loader: Callable[[str], Configuration | None],
        sessions_provider: Callable[[], list[str]],
        session_manager_provider: Callable[[], _SessionPathPort | None],
        model_manifest_saver: Callable[[Configuration], None] | None = None,
        models_root_provider: Callable[[], str] | None = None,
        administration_operation_factory: Callable[
            [], AbstractContextManager[object]
        ] = nullcontext,
        logger: _WarningLogger | None = None,
        lock_factory: StorageMutationLockFactory = StorageMutationLock,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the session metadata service and its owned state."""
        self._active_configuration_provider = active_configuration_provider
        self._session_configuration_loader = session_configuration_loader
        self._sessions_provider = sessions_provider
        self._session_manager_provider = session_manager_provider
        self._model_manifest_saver = model_manifest_saver
        self._models_root_provider = models_root_provider
        self._administration_operation_factory = administration_operation_factory
        self._logger = logger
        self._lock_factory = lock_factory
        self._lock_timeout_seconds = lock_timeout_seconds

    def _warning(self, message: str) -> None:
        """Publish a warning record through this session metadata service."""
        if self._logger is not None:
            self._logger.warning(message)

    def _active_configuration(self) -> Configuration | None:
        """Return the active configuration value owned by this session metadata service."""
        return self._active_configuration_provider()

    def _load_session_configuration(self, session_name: str) -> Configuration | None:
        """Load the session configuration owned by this session metadata service."""
        return self._session_configuration_loader(session_name)

    def _sessions(self) -> list[str]:
        """Return the sessions value owned by this session metadata service."""
        return list(self._sessions_provider() or [])

    def _build_training_session_manager(self) -> _SessionPathPort | None:
        """Build the training session manager required by this session metadata service."""
        return self._session_manager_provider()

    def _storage_mutation(self) -> AbstractContextManager[object]:
        """Return the storage mutation value owned by this session metadata service."""
        if self._models_root_provider is None:
            return nullcontext()
        scope = ModelsStorageMutationScope.from_models_root(self._models_root_provider())
        return scope.lock(
            lock_factory=self._lock_factory,
            timeout_seconds=self._lock_timeout_seconds,
        )

    def _save_model_manifest(self, configuration: Configuration) -> None:
        """Persist the model manifest owned by this session metadata service."""
        if self._model_manifest_saver is None:
            return
        self._model_manifest_saver(configuration)

    def default_session_base_name(self) -> str:
        """Return the default session base name value owned by this session metadata service."""
        configuration = self._active_configuration()
        if configuration is None:
            return "session"

        model_name = str(getattr(configuration, "model_name", "") or "model")
        return NameFormattingService.sanitize_name(model_name, fallback="session")

    def get_next_session_version(self, base_name: str) -> int:
        """Return the next session version exposed by this session metadata service."""
        if self._active_configuration() is None:
            return 1

        max_version = 0
        for existing_session in self._sessions():
            session_configuration = self._load_session_configuration(existing_session)
            if session_configuration is None:
                continue

            existing_base_name = str(
                getattr(session_configuration, "session_name", "") or ""
            ).strip()
            if existing_base_name != base_name:
                continue

            try:
                existing_version = int(getattr(session_configuration, "session_version", 0) or 0)
            except (TypeError, ValueError):
                existing_version = 0

            max_version = max(max_version, existing_version)

        return max_version + 1

    def get_session_display_name(self, session_uuid: str | None) -> str | None:
        """Return the session display name exposed by this session metadata service."""
        if not session_uuid:
            return None

        session_configuration = self._load_session_configuration(session_uuid)
        if session_configuration is None:
            return session_uuid

        display_name = str(getattr(session_configuration, "display_name", "") or "").strip()
        identity_uuid = self.get_session_identity_uuid(session_uuid) or session_uuid
        if display_name:
            return f"{display_name} [{identity_uuid[:8]}]"

        session_name = str(getattr(session_configuration, "session_name", "") or "").strip()
        try:
            session_version = int(getattr(session_configuration, "session_version", 0) or 0)
        except (TypeError, ValueError):
            session_version = 0

        if session_name and session_version > 0:
            return f"{session_name}.v{session_version} [{identity_uuid[:8]}]"

        return session_uuid

    def get_session_identity_uuid(self, session_uuid: str | None) -> str | None:
        """Return the session identity uuid exposed by this session metadata service."""
        if not session_uuid:
            return None

        session_configuration = self._load_session_configuration(session_uuid)
        if session_configuration is None:
            return session_uuid

        identity_uuid = str(getattr(session_configuration, "uuid", "") or "").strip()
        return identity_uuid or session_uuid

    def get_session_display_name_without_uuid(self, session_uuid: str | None) -> str | None:
        """Return the session display name without uuid exposed by this session metadata service."""
        if not session_uuid:
            return None

        session_configuration = self._load_session_configuration(session_uuid)
        if session_configuration is None:
            return session_uuid

        display_name = str(getattr(session_configuration, "display_name", "") or "").strip()
        if display_name:
            return display_name

        session_name = str(getattr(session_configuration, "session_name", "") or "").strip()
        try:
            session_version = int(getattr(session_configuration, "session_version", 0) or 0)
        except (TypeError, ValueError):
            session_version = 0

        if session_name and session_version > 0:
            return f"{session_name}.v{session_version}"

        return session_name or session_uuid

    def bump_session_minor_version(self) -> str:
        """Return the bump session minor version value owned by this session metadata service."""
        try:
            with (
                self._administration_operation_factory(),
                self._storage_mutation(),
            ):
                return self._bump_session_minor_version_locked()
        except (
            FileNotFoundError,
            StorageMutationLockError,
            StoragePathViolation,
            OSError,
        ) as exc:
            self._warning(f"Failed to lock model version storage: {exc}")
            configuration = self._active_configuration()
            if configuration is None:
                return "v1.1.1"
            return str(getattr(configuration, "model_version", "") or "")

    def _bump_session_minor_version_locked(self) -> str:
        """Return the bump session minor version locked value owned by this session metadata
        service.
        """
        configuration = self._active_configuration()
        if configuration is None:
            return "v1.1.1"

        current_version_text = str(getattr(configuration, "model_version", "") or "")
        parsed_version = EmptyModelGenerator.parse_composite_version(current_version_text)
        if parsed_version is None:
            return current_version_text

        manual_major, model_number, _ = parsed_version

        highest_session_minor = 0
        for existing_session in self._sessions():
            session_configuration = self._load_session_configuration(existing_session)
            if session_configuration is None:
                continue
            try:
                existing_version = int(getattr(session_configuration, "session_version", 0) or 0)
            except (TypeError, ValueError):
                existing_version = 0
            highest_session_minor = max(highest_session_minor, existing_version)

        next_version = EmptyModelGenerator.build_composite_version(
            manual_major, model_number, highest_session_minor
        )

        configuration.model_version = next_version
        try:
            self._save_model_manifest(configuration)
        except Exception as exc:
            self._warning(f"Failed to persist model composite version '{next_version}': {exc}")

        return next_version

    def get_session_parent_uuid(self, session_uuid: str | None) -> str | None:
        """Return the session parent uuid exposed by this session metadata service."""
        if not session_uuid:
            return None

        session_configuration = self._load_session_configuration(session_uuid)
        if session_configuration is None:
            return None

        parent_uuid = str(getattr(session_configuration, "parent_session_uuid", "") or "").strip()
        return parent_uuid or None

    def get_session_description(self, session_uuid: str | None) -> str:
        """Return the session description exposed by this session metadata service."""
        if not session_uuid:
            return ""

        session_configuration = self._load_session_configuration(session_uuid)
        if session_configuration is None:
            return ""

        return str(getattr(session_configuration, "description", "") or "").strip()

    def get_session_created_at(self, session_name: str | None) -> str | None:
        """Return the session created at exposed by this session metadata service."""
        if not session_name:
            return None

        session_configuration = self._load_session_configuration(session_name)
        if session_configuration is None:
            return None

        created_at = str(getattr(session_configuration, "created_at", "") or "").strip()
        return created_at or None

    def get_session_updated_at(self, session_name: str | None) -> str | None:
        """Return the session updated at exposed by this session metadata service."""
        if not session_name:
            return None

        session_configuration = self._load_session_configuration(session_name)
        if session_configuration is None:
            return None

        updated_at = str(getattr(session_configuration, "updated_at", "") or "").strip()
        return updated_at or None

    def get_session_last_modified_at(self, session_name: str | None) -> str | None:
        """Return the session last modified at exposed by this session metadata service."""
        updated_at = self.get_session_updated_at(session_name)
        if updated_at:
            return updated_at

        session_manager = self._build_training_session_manager()
        if session_manager is None or not session_name:
            return None

        session_config_path = session_manager.get_session_config_path(session_name)
        if not session_config_path or not os.path.isfile(session_config_path):
            return None

        try:
            return (
                datetime.fromtimestamp(
                    os.path.getmtime(session_config_path),
                    UTC,
                )
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except Exception:
            return None

    def get_session_child_uuids(self, session_uuid: str | None) -> list[str]:
        """Return the session child uuids exposed by this session metadata service."""
        if not session_uuid:
            return []

        session_configuration = self._load_session_configuration(session_uuid)
        explicit_children = (
            list(getattr(session_configuration, "child_session_uuids", []) or [])
            if session_configuration is not None
            else []
        )
        explicit_children = [
            str(child_uuid).strip() for child_uuid in explicit_children if str(child_uuid).strip()
        ]
        if explicit_children:
            return explicit_children

        children: list[str] = []
        for candidate_uuid in self._sessions():
            if self.get_session_parent_uuid(candidate_uuid) == session_uuid:
                children.append(candidate_uuid)
        return children

    def get_session_reference_display(self, session_uuid: str | None) -> str:
        """Return the session reference display exposed by this session metadata service."""
        if not session_uuid:
            return "-"
        return self.get_session_display_name(session_uuid) or session_uuid

    def build_session_ancestry_chain(
        self,
        session_uuid: str | None,
        max_depth: int = 128,
    ) -> list[SessionChainNode]:
        """Build the session ancestry chain required by this session metadata service."""
        if not session_uuid:
            return []

        chain_reversed: list[SessionChainNode] = []
        visited: set[str] = set()
        current_uuid: str | None = session_uuid
        depth = 0

        while current_uuid and depth < max_depth:
            if current_uuid in visited:
                self._warning(f"Cycle detected in session ancestry chain at '{current_uuid}'.")
                break
            visited.add(current_uuid)

            parent_uuid = self.get_session_parent_uuid(current_uuid)
            chain_reversed.append(
                SessionChainNode(
                    session_uuid=current_uuid,
                    display_name=self.get_session_reference_display(current_uuid),
                    parent_session_uuid=parent_uuid,
                    session_description=self.get_session_description(current_uuid),
                    is_current=(current_uuid == session_uuid),
                )
            )

            current_uuid = parent_uuid
            depth += 1

        chain_reversed.reverse()
        return chain_reversed
