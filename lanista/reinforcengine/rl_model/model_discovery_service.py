# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate model discovery."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from yaml import YAMLError

from lanista.reinforcengine.application_errors import (
    ApplicationError,
    ApplicationValidationError,
    PersistenceError,
    ResourceNotFoundError,
)
from lanista.reinforcengine.rl_model.empty_model_generator import EmptyModelGenerator
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.sessions.configuration_wrapper import (
    SessionConfigurationWrapper,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    TRAINING_SESSIONS_DIR_NAME,
    find_existing_model_manifest_path,
    find_existing_session_manifest_path,
)
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)

_PathSignature = tuple[int, int, int, int, int] | None
_SessionEntrySignature = tuple[
    str,
    _PathSignature,
    _PathSignature,
]
_SessionRootSignature = tuple[
    _PathSignature,
    tuple[_SessionEntrySignature, ...] | None,
]
_ModelManifestSignature = tuple[
    str,
    _PathSignature,
    _PathSignature,
]
_ModelsRootSignature = tuple[_ModelManifestSignature, ...]
_ModelDiscoverySignature = tuple[
    str,
    _PathSignature,
    _PathSignature,
    _SessionRootSignature,
]
_DiscoveryRootSignature = tuple[_ModelDiscoverySignature, ...]


class DiscoveredSessionMetadataStatus(StrEnum):
    """Quality of the session metadata projected for one discovered model."""

    CLEAN = "clean"
    INVALID_METADATA = "invalid_metadata"
    UNSAFE_STORAGE = "unsafe_storage"
    UNREADABLE_STORAGE = "unreadable_storage"


@dataclass(frozen=True, slots=True)
class DiscoveredValidModelEntry:
    """Project one selectable model and the quality of its session inventory."""

    manifest: ModelManifest
    display_name: str
    effective_version: str
    short_uuid: str
    created_at_display: str
    session_count: int
    invalid_session_count: int
    session_metadata_status: DiscoveredSessionMetadataStatus
    is_current: bool
    search_text: str


@dataclass(frozen=True, slots=True)
class DiscoveredInvalidModelEntry:
    """Describe a model-root entry that is unsafe or cannot be loaded."""

    folder_name: str
    reason: str
    search_text: str


@dataclass(frozen=True, slots=True)
class _DiscoveryEntryResult:
    """Hold exactly one classified model-folder discovery result."""

    valid: DiscoveredValidModelEntry | None = None
    invalid: DiscoveredInvalidModelEntry | None = None


@dataclass(frozen=True, slots=True)
class _DiscoveredSessionInventory:
    """Summarize strict session reads without exposing filesystem details."""

    session_count: int = 0
    invalid_session_count: int = 0
    highest_valid_minor: int = 0
    status: DiscoveredSessionMetadataStatus = DiscoveredSessionMetadataStatus.CLEAN


@dataclass(frozen=True, slots=True)
class _DiscoveredSessionEntry:
    """Classify one directory below a model's managed session root."""

    discovered: bool = False
    valid_minor: int | None = None
    status: DiscoveredSessionMetadataStatus = DiscoveredSessionMetadataStatus.CLEAN


class ModelDiscoveryService:
    """Coordinate model discovery operations."""

    def __init__(
        self,
        *,
        models_root_provider: Callable[[], str],
        logger_name_provider: Callable[[], str],
        logger_level_provider: Callable[[], str],
        manifest_cls: type[ModelManifest] = ModelManifest,
    ) -> None:
        """Initialize the model discovery service and its owned state."""
        self._models_root_provider = models_root_provider
        self._logger_name_provider = logger_name_provider
        self._logger_level_provider = logger_level_provider
        self._manifest_cls = manifest_cls
        self._available_manifest_cache_signature: _ModelsRootSignature | None = None
        self._available_manifest_cache: list[ModelManifest] = []
        self._discovery_cache_signature: _DiscoveryRootSignature | None = None
        self._discovery_cache_current_uuid: str = ""
        self._discovery_cache_valid_entries: list[DiscoveredValidModelEntry] = []
        self._discovery_cache_invalid_entries: list[DiscoveredInvalidModelEntry] = []

    def _manifest_logger_kwargs(self, child_logger_name: str) -> dict[str, str]:
        """Return the manifest logger kwargs value owned by this model discovery service."""
        return {
            "parent_logger_name": self._logger_name_provider(),
            "child_logger_name": child_logger_name,
            "child_logger_level": self._logger_level_provider(),
        }

    @staticmethod
    def _path_signature(path: str | os.PathLike[str]) -> _PathSignature:
        """Return the path signature value owned by this model discovery service."""
        try:
            stat_result = os.lstat(path)
        except OSError:
            return None
        return (
            int(stat_result.st_mode),
            int(stat_result.st_size),
            int(stat_result.st_mtime_ns),
            int(stat_result.st_ctime_ns),
            int(stat_result.st_ino),
        )

    @staticmethod
    def _resolve_safe_model_manifest_path(
        model_folder_path: str | os.PathLike[str],
    ) -> Path | None:
        """Return a direct, regular model manifest without following aliases."""

        raw_model_folder = Path(model_folder_path).expanduser()
        if raw_model_folder.is_symlink() or not raw_model_folder.is_dir():
            return None

        raw_manifest = Path(find_existing_model_manifest_path(str(raw_model_folder)))
        if not os.path.lexists(raw_manifest) or raw_manifest.is_symlink():
            return None

        try:
            guard = StoragePathGuard(raw_model_folder)
            manifest_path = guard.require_descendant(raw_manifest, must_exist=True)
        except (FileNotFoundError, OSError, StoragePathViolation):
            return None
        if manifest_path.parent != guard.root or not manifest_path.is_file():
            return None
        return manifest_path

    @classmethod
    def _build_session_root_signature(
        cls,
        model_folder_path: str,
    ) -> _SessionRootSignature:
        """Build the session root signature required by this model discovery service."""
        sessions_root = os.path.join(model_folder_path, TRAINING_SESSIONS_DIR_NAME)
        root_signature = cls._path_signature(sessions_root)
        if (
            root_signature is None
            or os.path.islink(sessions_root)
            or not os.path.isdir(sessions_root)
        ):
            return root_signature, ()

        try:
            with os.scandir(sessions_root) as scanner:
                entries = sorted(scanner, key=lambda item: item.name.lower())
        except OSError:
            return root_signature, None

        signatures: list[_SessionEntrySignature] = []
        for entry in entries:
            entry_signature = cls._path_signature(entry.path)
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError:
                is_directory = False
            canonical_manifest = (
                find_existing_session_manifest_path(entry.path)
                if is_directory
                else ""
            )
            signatures.append(
                (
                    entry.name,
                    entry_signature,
                    cls._path_signature(canonical_manifest)
                    if canonical_manifest
                    else None,
                )
            )
        return root_signature, tuple(signatures)

    @classmethod
    def build_models_root_signature(
        cls,
        root_path: str,
    ) -> _ModelsRootSignature | None:
        """Build the models root signature required by this model discovery service."""
        if not os.path.isdir(root_path):
            return None

        try:
            with os.scandir(root_path) as scanner:
                entries = sorted(scanner, key=lambda item: item.name.lower())
        except OSError:
            return None

        signature_parts: list[_ModelManifestSignature] = []
        for entry in entries:
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError:
                is_directory = False
            if not is_directory:
                continue
            manifest_path = find_existing_model_manifest_path(entry.path)
            signature_parts.append(
                (
                    entry.name,
                    cls._path_signature(entry.path),
                    cls._path_signature(manifest_path),
                )
            )

        return tuple(signature_parts)

    @classmethod
    def _build_discovery_root_signature(
        cls,
        root_path: str,
    ) -> _DiscoveryRootSignature | None:
        """Build the discovery root signature required by this model discovery service."""
        if not os.path.isdir(root_path):
            return None
        try:
            with os.scandir(root_path) as scanner:
                entries = sorted(scanner, key=lambda item: item.name.lower())
        except OSError:
            return None

        signature_parts: list[_ModelDiscoverySignature] = []
        for entry in entries:
            try:
                is_directory = entry.is_dir(follow_symlinks=False)
            except OSError:
                is_directory = False
            manifest_path = (
                find_existing_model_manifest_path(entry.path)
                if is_directory
                else ""
            )
            signature_parts.append(
                (
                    entry.name,
                    cls._path_signature(entry.path),
                    cls._path_signature(manifest_path) if manifest_path else None,
                    cls._build_session_root_signature(entry.path)
                    if is_directory
                    else (None, ()),
                )
            )
        return tuple(signature_parts)

    @staticmethod
    def format_discovered_created_at(value: str) -> str:
        """Format the discovered created at for presentation by this model discovery service."""
        text = str(value or "").strip()
        if not text:
            return "-"
        normalized = text.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return text[:19] if len(text) >= 19 else text

    @staticmethod
    def _resolve_discovered_effective_version(
        raw_version: str,
        highest_valid_session_minor: int,
    ) -> str:
        """Resolve the discovered effective version required by this model discovery service."""
        parsed_version = EmptyModelGenerator.parse_composite_version(raw_version)
        if parsed_version is None:
            return raw_version

        manual_major, model_number, session_minor = parsed_version
        highest_minor = max(int(session_minor), highest_valid_session_minor)
        return EmptyModelGenerator.build_composite_version(
            manual_major,
            model_number,
            highest_minor,
        )

    @classmethod
    def _discover_session_inventory(
        cls,
        model_folder_path: str,
    ) -> _DiscoveredSessionInventory:
        """Return the discover session inventory value owned by this model discovery service."""
        raw_sessions_root = Path(model_folder_path) / TRAINING_SESSIONS_DIR_NAME
        if not os.path.lexists(raw_sessions_root):
            return _DiscoveredSessionInventory()
        if raw_sessions_root.is_symlink():
            return _DiscoveredSessionInventory(
                status=DiscoveredSessionMetadataStatus.UNSAFE_STORAGE,
            )

        guard = StoragePathGuard(model_folder_path)
        try:
            sessions_root = guard.require_descendant(
                raw_sessions_root,
                must_exist=True,
            )
        except StoragePathViolation:
            return _DiscoveredSessionInventory(
                status=DiscoveredSessionMetadataStatus.UNSAFE_STORAGE,
            )
        except (FileNotFoundError, OSError):
            return _DiscoveredSessionInventory(
                status=DiscoveredSessionMetadataStatus.UNREADABLE_STORAGE,
            )
        if not sessions_root.is_dir():
            return _DiscoveredSessionInventory(
                status=DiscoveredSessionMetadataStatus.INVALID_METADATA,
            )

        try:
            with os.scandir(sessions_root) as scanner:
                entries = sorted(scanner, key=lambda item: item.name.lower())
        except OSError:
            return _DiscoveredSessionInventory(
                status=DiscoveredSessionMetadataStatus.UNREADABLE_STORAGE,
            )

        session_count = 0
        invalid_session_count = 0
        highest_valid_minor = 0
        status = DiscoveredSessionMetadataStatus.CLEAN
        for entry in entries:
            inspected = cls._inspect_session_entry(
                entry,
                guard=guard,
                sessions_root=sessions_root,
            )
            if not inspected.discovered:
                continue
            session_count += 1
            if inspected.valid_minor is None:
                invalid_session_count += 1
            else:
                highest_valid_minor = max(
                    highest_valid_minor,
                    inspected.valid_minor,
                )
            status = cls._more_severe_session_status(status, inspected.status)

        return _DiscoveredSessionInventory(
            session_count=session_count,
            invalid_session_count=invalid_session_count,
            highest_valid_minor=highest_valid_minor,
            status=status,
        )

    @classmethod
    def _inspect_session_entry(
        cls,
        entry: os.DirEntry[str],
        *,
        guard: StoragePathGuard,
        sessions_root: Path,
    ) -> _DiscoveredSessionEntry:
        """Return the inspect session entry value owned by this model discovery service."""
        try:
            is_symlink = entry.is_symlink()
        except OSError:
            return _DiscoveredSessionEntry(
                discovered=True,
                status=DiscoveredSessionMetadataStatus.UNREADABLE_STORAGE,
            )
        if is_symlink:
            return _DiscoveredSessionEntry(
                discovered=True,
                status=DiscoveredSessionMetadataStatus.UNSAFE_STORAGE,
            )
        try:
            if not entry.is_dir(follow_symlinks=False):
                return _DiscoveredSessionEntry()
            session_path = guard.require_descendant(entry.path, must_exist=True)
        except StoragePathViolation:
            return _DiscoveredSessionEntry(
                discovered=True,
                status=DiscoveredSessionMetadataStatus.UNSAFE_STORAGE,
            )
        except (FileNotFoundError, OSError):
            return _DiscoveredSessionEntry(
                discovered=True,
                status=DiscoveredSessionMetadataStatus.UNREADABLE_STORAGE,
            )
        if session_path.parent != sessions_root:
            return _DiscoveredSessionEntry(
                discovered=True,
                status=DiscoveredSessionMetadataStatus.UNSAFE_STORAGE,
            )

        return cls._inspect_session_manifest(session_path, guard=guard)

    @classmethod
    def _inspect_session_manifest(
        cls,
        session_path: Path,
        *,
        guard: StoragePathGuard,
    ) -> _DiscoveredSessionEntry:
        """Validate one canonical manifest and extract its session minor."""

        manifest_path = Path(find_existing_session_manifest_path(str(session_path)))
        if not os.path.lexists(manifest_path):
            return _DiscoveredSessionEntry(
                discovered=True,
                status=DiscoveredSessionMetadataStatus.INVALID_METADATA,
            )
        if manifest_path.is_symlink():
            return _DiscoveredSessionEntry(
                discovered=True,
                status=DiscoveredSessionMetadataStatus.UNSAFE_STORAGE,
            )

        try:
            validated_manifest = guard.require_descendant(
                manifest_path,
                must_exist=True,
            )
            if (
                validated_manifest.parent != session_path
                or not validated_manifest.is_file()
            ):
                raise StoragePathViolation("Session manifest is not a direct regular file.")
            session_configuration = SessionConfigurationWrapper(
                str(validated_manifest)
            ).load_configuration()
            valid_minor = cls._parse_session_minor(session_configuration)
        except StoragePathViolation:
            return _DiscoveredSessionEntry(
                discovered=True,
                status=DiscoveredSessionMetadataStatus.UNSAFE_STORAGE,
            )
        except OSError:
            return _DiscoveredSessionEntry(
                discovered=True,
                status=DiscoveredSessionMetadataStatus.UNREADABLE_STORAGE,
            )
        except (AttributeError, TypeError, ValueError, YAMLError):
            return _DiscoveredSessionEntry(
                discovered=True,
                status=DiscoveredSessionMetadataStatus.INVALID_METADATA,
            )
        return _DiscoveredSessionEntry(
            discovered=True,
            valid_minor=valid_minor,
        )

    @staticmethod
    def _parse_session_minor(session_configuration: object) -> int:
        """Parse the session minor owned by this model discovery service."""
        raw_version = getattr(session_configuration, "session_version", None)
        if type(raw_version) is int and raw_version >= 0:
            return raw_version
        if isinstance(raw_version, str):
            normalized = raw_version.strip()
            if normalized.isdigit():
                return int(normalized)
        raise ValueError("Session version must be a non-negative integer.")

    @staticmethod
    def _more_severe_session_status(
        current: DiscoveredSessionMetadataStatus,
        candidate: DiscoveredSessionMetadataStatus,
    ) -> DiscoveredSessionMetadataStatus:
        """Return the more severe session status value owned by this model discovery service."""
        priority = {
            DiscoveredSessionMetadataStatus.CLEAN: 0,
            DiscoveredSessionMetadataStatus.INVALID_METADATA: 1,
            DiscoveredSessionMetadataStatus.UNREADABLE_STORAGE: 2,
            DiscoveredSessionMetadataStatus.UNSAFE_STORAGE: 3,
        }
        return candidate if priority[candidate] > priority[current] else current

    def discover_model_candidates(
        self,
        current_model_uuid: str | None = None,
    ) -> tuple[list[DiscoveredValidModelEntry], list[DiscoveredInvalidModelEntry]]:
        """Return the discover model candidates value owned by this model discovery service."""
        root_path = self._models_root_provider()
        normalized_current_uuid = str(current_model_uuid or "").strip()
        discovery_signature = self._build_discovery_root_signature(root_path)
        cached_result = self._cached_discovery_result(
            discovery_signature,
            normalized_current_uuid,
        )
        if cached_result is not None:
            return cached_result

        valid_entries: list[DiscoveredValidModelEntry] = []
        invalid_entries: list[DiscoveredInvalidModelEntry] = []
        if not os.path.isdir(root_path):
            return valid_entries, invalid_entries

        manifest_logger_kwargs = self._manifest_logger_kwargs("model_manifest")
        try:
            with os.scandir(root_path) as scanner:
                entries = sorted(scanner, key=lambda item: item.name.lower())
        except OSError:
            return valid_entries, invalid_entries

        for entry in entries:
            result = self._classify_discovery_entry(
                entry,
                manifest_logger_kwargs,
                normalized_current_uuid,
            )
            if result.valid is not None:
                valid_entries.append(result.valid)
            if result.invalid is not None:
                invalid_entries.append(result.invalid)

        self._publish_discovery_cache(
            discovery_signature,
            normalized_current_uuid,
            valid_entries,
            invalid_entries,
        )
        return valid_entries, invalid_entries

    def _cached_discovery_result(
        self,
        signature: _DiscoveryRootSignature | None,
        current_model_uuid: str,
    ) -> tuple[list[DiscoveredValidModelEntry], list[DiscoveredInvalidModelEntry]] | None:
        """Return the cached discovery result value owned by this model discovery service."""
        if (
            signature is None
            or signature != self._discovery_cache_signature
            or current_model_uuid != self._discovery_cache_current_uuid
        ):
            return None
        return (
            list(self._discovery_cache_valid_entries),
            list(self._discovery_cache_invalid_entries),
        )

    def _classify_discovery_entry(
        self,
        entry: os.DirEntry[str],
        manifest_logger_kwargs: dict[str, str],
        current_model_uuid: str,
    ) -> _DiscoveryEntryResult:
        """Classify the discovery entry owned by this model discovery service."""
        try:
            is_symlink = entry.is_symlink()
            is_directory = entry.is_dir(follow_symlinks=False)
        except OSError:
            return _DiscoveryEntryResult(
                invalid=self._invalid_discovery_entry(
                    entry.name,
                    "manifest discovery failed",
                )
            )
        if is_symlink:
            return _DiscoveryEntryResult(
                invalid=self._invalid_discovery_entry(
                    entry.name,
                    "unsafe model storage path",
                )
            )
        if not is_directory:
            return _DiscoveryEntryResult(
                invalid=self._invalid_discovery_entry(
                    entry.name,
                    "manifest discovery failed",
                )
            )

        raw_manifest_path = Path(find_existing_model_manifest_path(entry.path))
        if not os.path.lexists(raw_manifest_path):
            return _DiscoveryEntryResult(
                invalid=self._invalid_discovery_entry(
                    entry.name,
                    "missing model_manifest.yaml",
                )
            )
        if self._resolve_safe_model_manifest_path(entry.path) is None:
            return _DiscoveryEntryResult(
                invalid=self._invalid_discovery_entry(
                    entry.name,
                    "unsafe model manifest path",
                )
            )
        return self._load_discovery_manifest(
            entry,
            manifest_logger_kwargs,
            current_model_uuid,
        )

    def _load_discovery_manifest(
        self,
        entry: os.DirEntry[str],
        manifest_logger_kwargs: dict[str, str],
        current_model_uuid: str,
    ) -> _DiscoveryEntryResult:
        """Load the discovery manifest owned by this model discovery service."""
        try:
            manifest = self._manifest_cls(entry.path, **manifest_logger_kwargs)
            manifest.load()
            manifest.ensure_valid()
            return _DiscoveryEntryResult(
                valid=self._build_valid_discovery_entry(
                    manifest,
                    current_model_uuid,
                    entry.path,
                )
            )
        except ApplicationValidationError:
            return _DiscoveryEntryResult(
                invalid=self._invalid_discovery_entry(
                    entry.name,
                    "invalid model manifest",
                )
            )
        except (PersistenceError, ResourceNotFoundError):
            return _DiscoveryEntryResult(
                invalid=self._invalid_discovery_entry(
                    entry.name,
                    "unreadable model manifest",
                )
            )
        except (
            ApplicationError,
            OSError,
            AttributeError,
            TypeError,
            ValueError,
            YAMLError,
        ):
            return _DiscoveryEntryResult(
                invalid=self._invalid_discovery_entry(
                    entry.name,
                    "manifest discovery failed",
                )
            )

    def _build_valid_discovery_entry(
        self,
        manifest: ModelManifest,
        current_model_uuid: str,
        model_folder_path: str,
    ) -> DiscoveredValidModelEntry:
        """Build the valid discovery entry required by this model discovery service."""
        cfg = manifest.get_configuration()
        model_name = str(getattr(cfg, "model_name", "") or "Unnamed model")
        raw_version = str(getattr(cfg, "model_version", "") or "").strip()
        session_inventory = self._discover_session_inventory(model_folder_path)
        effective_version = self._resolve_discovered_effective_version(
            raw_version,
            session_inventory.highest_valid_minor,
        )
        model_uuid = str(getattr(cfg, "model_uuid", "") or "")
        created_at_display = self.format_discovered_created_at(
            str(getattr(cfg, "created_at", "") or "")
        )
        return DiscoveredValidModelEntry(
            manifest=manifest,
            display_name=model_name,
            effective_version=effective_version,
            short_uuid=EmptyModelGenerator.shorten_uuid(model_uuid),
            created_at_display=created_at_display,
            session_count=session_inventory.session_count,
            invalid_session_count=session_inventory.invalid_session_count,
            session_metadata_status=session_inventory.status,
            is_current=bool(model_uuid) and model_uuid == current_model_uuid,
            search_text=" ".join(
                [
                    model_name,
                    effective_version,
                    model_uuid,
                    created_at_display,
                    str(session_inventory.session_count),
                    str(session_inventory.invalid_session_count),
                    session_inventory.status.value,
                ]
            ).lower(),
        )

    @staticmethod
    def _invalid_discovery_entry(
        folder_name: str,
        reason: str,
    ) -> DiscoveredInvalidModelEntry:
        """Return the invalid discovery entry value owned by this model discovery service."""
        return DiscoveredInvalidModelEntry(
            folder_name=folder_name,
            reason=reason,
            search_text=f"{folder_name} {reason}".lower(),
        )

    def _publish_discovery_cache(
        self,
        signature: _DiscoveryRootSignature | None,
        current_model_uuid: str,
        valid_entries: list[DiscoveredValidModelEntry],
        invalid_entries: list[DiscoveredInvalidModelEntry],
    ) -> None:
        """Publish the discovery cache owned by this model discovery service."""
        self._discovery_cache_signature = signature
        self._discovery_cache_current_uuid = current_model_uuid
        self._discovery_cache_valid_entries = list(valid_entries)
        self._discovery_cache_invalid_entries = list(invalid_entries)

    def list_available_model_manifests(self) -> list[ModelManifest]:
        """Return the available model manifests available to this model discovery service."""
        root = self._models_root_provider()
        if not os.path.isdir(root):
            return []

        signature = self.build_models_root_signature(root)
        if signature is not None and signature == self._available_manifest_cache_signature:
            return list(self._available_manifest_cache)

        try:
            with os.scandir(root) as scanner:
                entries = sorted(scanner, key=lambda item: item.name.lower())
        except OSError:
            return []

        manifests: list[ModelManifest] = []
        for entry in entries:
            if self._resolve_safe_model_manifest_path(entry.path) is None:
                continue
            try:
                model_manifest = self._manifest_cls(
                    entry.path,
                    **self._manifest_logger_kwargs("model_manifest_discovery"),
                )
                model_manifest.load()
                model_manifest.ensure_valid()
                manifests.append(model_manifest)
            except (
                ApplicationError,
                OSError,
                AttributeError,
                TypeError,
                ValueError,
                YAMLError,
            ):
                # Catalog discovery isolates invalid metadata per model. An
                # explicit load still propagates the typed failure to its facade.
                continue

        self._available_manifest_cache_signature = signature
        self._available_manifest_cache = list(manifests)
        return manifests

    def load_model_manifest_from_folder(
        self,
        model_folder_path: str,
    ) -> ModelManifest | None:
        """Load one direct child of the configured model root if it is valid."""

        if Path(model_folder_path).expanduser().is_symlink():
            return None
        guard = StoragePathGuard(self._models_root_provider())
        try:
            model_folder = guard.require_descendant(
                model_folder_path,
                must_exist=True,
            )
        except (FileNotFoundError, StoragePathViolation):
            return None
        if model_folder.parent != guard.root:
            return None
        if self._resolve_safe_model_manifest_path(model_folder) is None:
            return None

        manifest = self._manifest_cls(
            str(model_folder),
            **self._manifest_logger_kwargs("model_manifest_discovery"),
        )
        manifest.load()
        manifest.ensure_valid()
        return manifest

    def has_available_models(self) -> bool:
        """Return whether available models applies to this model discovery service."""
        return len(self.list_available_model_manifests()) > 0

    def get_latest_available_model_manifest(self) -> ModelManifest | None:
        """Return the latest available model manifest exposed by this model discovery service."""
        manifests = self.list_available_model_manifests()
        if not manifests:
            return None

        def key(item: ModelManifest) -> float:
            """Perform the key step within the enclosing get latest available model manifest
            workflow.
            """
            try:
                folder = str(item.get_configuration().model_folder_path)
                return os.path.getmtime(folder)
            except (OSError, AttributeError, TypeError, ValueError):
                return -1.0

        manifests.sort(key=key, reverse=True)
        return manifests[0]
