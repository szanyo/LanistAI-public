# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Read-only validation of model packages before clone staging begins."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from yaml import YAMLError

from lanista.reinforcengine.rl_model.metadata.configuration_schema import (
    VALID_FRAMEWORK_UUIDS,
    ConfigurationSchema,
)
from lanista.reinforcengine.rl_model.metadata.schema import (
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    FRAMEWORK_MODEL_CONFIGURATION_FILE_NAME,
    MODEL_CONFIGURATION_FILE_NAME,
    MODEL_MANIFEST_FILE_NAME,
    RESOURCES_DIR_NAME,
)
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.storage.persistence import read_yaml_mapping_file


class CloneSourceValidationError(ValueError):
    """Raised when a source package is not a canonical schema-v1 model."""


@dataclass(frozen=True, slots=True)
class CloneSourceMetadata:
    """Validated metadata needed to construct an immutable clone plan."""

    model_uuid: str | None
    model_version: str
    description: str
    preferred_framework_uuid: str


class ModelClonePreflight:
    """Validate clone inputs without creating storage or changing metadata."""

    _NON_CLONABLE_ROOTS = frozenset({"tensorboard", "training_sessions"})
    _IGNORED_NAMES = frozenset({"__pycache__"})

    def __init__(self, source_folder: Path) -> None:
        """Initialize the model clone preflight and its owned state."""
        self._source_folder = source_folder
        self._guard = StoragePathGuard(source_folder)

    def inspect(
        self,
        selected_relative_paths: tuple[str, ...] | None,
    ) -> CloneSourceMetadata:
        """Return canonical source metadata or fail before transaction staging."""

        self._validate_cloneable_links(selected_relative_paths)
        model_payload = self._read_versioned(
            MODEL_MANIFEST_FILE_NAME,
            MetadataKind.MODEL_MANIFEST,
            label="model manifest",
        )
        configuration_payload = self._read_versioned(
            MODEL_CONFIGURATION_FILE_NAME,
            MetadataKind.MODEL_CONFIGURATION,
            label="model configuration",
        )
        configuration = self._validate_model_configuration(configuration_payload)
        frameworks = self._frameworks_to_validate(
            configuration.preferred_framework_uuid,
            selected_relative_paths,
        )
        for framework_uuid in frameworks:
            self._validate_framework_configuration(framework_uuid)

        model = model_payload.get("model")
        if not isinstance(model, Mapping):
            raise CloneSourceValidationError("Source model manifest is invalid.")
        return CloneSourceMetadata(
            model_uuid=str(model.get("uuid") or "") or None,
            model_version=str(model.get("version") or ""),
            description=str(model.get("description") or ""),
            preferred_framework_uuid=configuration.preferred_framework_uuid,
        )

    def _read_versioned(
        self,
        relative_path: str,
        kind: MetadataKind,
        *,
        label: str,
    ) -> dict[str, object]:
        """Read the versioned without changing this model clone preflight's state."""
        try:
            path = self._guard.resolve(relative_path, must_exist=True)
            payload = read_yaml_mapping_file(os.fspath(path))
            return validate_versioned_metadata_payload(kind, payload)
        except FileNotFoundError as error:
            raise CloneSourceValidationError(
                f"Source {label} is missing. Run the metadata migration before cloning."
            ) from error
        except (OSError, TypeError, ValueError, YAMLError) as error:
            raise CloneSourceValidationError(
                f"Source {label} is invalid. Run the metadata migration before cloning."
            ) from error

    def _validate_model_configuration(
        self,
        payload: Mapping[str, object],
    ) -> ConfigurationSchema:
        """Validate the model configuration required by this model clone preflight."""
        section = payload.get("model_configuration")
        if not isinstance(section, Mapping):
            raise CloneSourceValidationError("Source model configuration is invalid.")

        preferred = self._require_framework_uuid(
            section.get("preferred_framework_uuid"),
            field_name="preferred framework",
        )
        self._require_framework_uuid(
            section.get("active_runtime_framework_uuid"),
            field_name="active runtime framework",
        )
        try:
            configuration = ConfigurationSchema.from_model_configuration_payload(payload)
        except (TypeError, ValueError) as error:
            raise CloneSourceValidationError("Source model configuration is invalid.") from error
        if configuration.preferred_framework_uuid != preferred:
            raise CloneSourceValidationError("Source preferred framework is invalid.")
        self._validate_framework_resource_paths(configuration)
        return configuration

    @staticmethod
    def _require_framework_uuid(value: object, *, field_name: str) -> str:
        """Return the required framework uuid for this model clone preflight."""
        normalized = str(value or "").strip().lower()
        if normalized not in VALID_FRAMEWORK_UUIDS:
            raise CloneSourceValidationError(f"Source {field_name} is invalid.")
        return normalized

    def _validate_framework_resource_paths(
        self,
        configuration: ConfigurationSchema,
    ) -> None:
        """Validate the framework resource paths required by this model clone preflight."""
        for framework_uuid, configured_path in (
            configuration.framework_resource_paths_as_dict.items()
        ):
            expected_path = f"{RESOURCES_DIR_NAME}/{framework_uuid}"
            if str(configured_path).replace("\\", "/").rstrip("/") != expected_path:
                raise CloneSourceValidationError(
                    "Source framework resource paths must use the canonical model layout."
                )
            self._guard.resolve(expected_path, must_exist=True)

    def _frameworks_to_validate(
        self,
        preferred_framework_uuid: str,
        selected_relative_paths: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        """Return the frameworks to validate value owned by this model clone preflight."""
        frameworks = {preferred_framework_uuid}
        for framework_uuid in sorted(VALID_FRAMEWORK_UUIDS):
            manifest_path = self._framework_manifest_relative_path(framework_uuid)
            if self._selection_contains(manifest_path, selected_relative_paths):
                frameworks.add(framework_uuid)
        return tuple(sorted(frameworks))

    @staticmethod
    def _selection_contains(
        candidate: str,
        selected_relative_paths: tuple[str, ...] | None,
    ) -> bool:
        """Return the selection contains value owned by this model clone preflight."""
        if selected_relative_paths is None:
            return True
        return any(
            candidate == selected or candidate.startswith(f"{selected}/")
            for selected in selected_relative_paths
        )

    def _validate_framework_configuration(self, framework_uuid: str) -> None:
        """Validate the framework configuration required by this model clone preflight."""
        payload = self._read_versioned(
            self._framework_manifest_relative_path(framework_uuid),
            MetadataKind.FRAMEWORK_CONFIGURATION,
            label=f"{framework_uuid} framework manifest",
        )
        engine = payload.get("rl_engine_framework")
        implementation = payload.get("model_implementation")
        if not isinstance(engine, Mapping) or not isinstance(implementation, Mapping):
            raise CloneSourceValidationError(
                f"Source {framework_uuid} framework manifest is incomplete."
            )
        actual_framework = str(engine.get("uuid") or "").strip().lower()
        module = str(implementation.get("module") or "").strip()
        object_name = str(implementation.get("object") or "").strip()
        if actual_framework != framework_uuid or not module or not object_name:
            raise CloneSourceValidationError(
                f"Source {framework_uuid} framework manifest is incomplete."
            )

    @staticmethod
    def _framework_manifest_relative_path(framework_uuid: str) -> str:
        """Return the framework manifest relative path value owned by this model clone preflight."""
        return (
            f"{RESOURCES_DIR_NAME}/{framework_uuid}/"
            f"{FRAMEWORK_MODEL_CONFIGURATION_FILE_NAME}"
        )

    def _validate_cloneable_links(
        self,
        selected_relative_paths: tuple[str, ...] | None,
    ) -> None:
        """Validate the cloneable links required by this model clone preflight."""
        roots = self._inspection_roots(selected_relative_paths)
        for root in roots:
            if root.is_dir():
                self._inspect_tree_links(root)
            elif root.is_symlink():
                self._guard.require_descendant(root, must_exist=True)

    def _inspection_roots(
        self,
        selected_relative_paths: tuple[str, ...] | None,
    ) -> tuple[Path, ...]:
        """Return the inspection roots value owned by this model clone preflight."""
        if selected_relative_paths is None:
            return (self._source_folder,)
        return tuple(
            self._guard.resolve(relative_path, must_exist=True)
            for relative_path in selected_relative_paths
            if self._is_cloneable_relative_path(relative_path)
        )

    def _inspect_tree_links(self, root: Path) -> None:
        """Inspect the tree links owned by this model clone preflight."""
        for directory, directory_names, file_names in os.walk(root, followlinks=False):
            current = Path(directory)
            directory_names[:] = [
                name
                for name in directory_names
                if self._is_cloneable_relative_path(
                    Path(current, name).relative_to(self._source_folder).as_posix()
                )
            ]
            for name in tuple(directory_names):
                candidate = Path(current, name)
                if candidate.is_symlink():
                    self._guard.require_descendant(candidate, must_exist=True)
                    raise StoragePathViolation(
                        "Directory symlinks cannot be cloned safely."
                    )
            for name in file_names:
                candidate = Path(current, name)
                if candidate.is_symlink():
                    self._guard.require_descendant(candidate, must_exist=True)

    @classmethod
    def _is_cloneable_relative_path(cls, relative_path: str) -> bool:
        """Return whether cloneable relative path applies to this model clone preflight."""
        parts = Path(relative_path).parts
        if not parts:
            return True
        return parts[0] not in cls._NON_CLONABLE_ROOTS and not any(
            part in cls._IGNORED_NAMES for part in parts
        )
