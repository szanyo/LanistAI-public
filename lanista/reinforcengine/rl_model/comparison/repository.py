# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Own persistent paths and versioned manifests for comparison packs."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from lanista.reinforcengine.application_errors import PersistenceError
from lanista.reinforcengine.rl_model.metadata.schema import (
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.metadata.timestamps import utc_now_iso
from lanista.reinforcengine.rl_model.storage.layout import (
    TENSORBOARD_COMPARISON_DIR_NAME,
    TENSORBOARD_DIR_NAME,
)
from lanista.reinforcengine.rl_model.storage.names import (
    ComparisonPackStorageName,
    InvalidStorageName,
)
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.storage.persistence import (
    atomic_write_yaml_file,
    read_yaml_mapping_file,
)

COMPARISON_PACK_MANIFEST_FILE_NAME = "comparison_pack_manifest.yml"


class ComparisonPackMetadataError(PersistenceError):
    """Report an existing manifest that is not readable canonical metadata."""


@dataclass(frozen=True, slots=True)
class ComparisonPackRecord:
    """Describe one guarded comparison directory and its optional metadata."""

    name: str
    path: str
    annotation: str | None
    source_paths: tuple[str, ...]


class ComparisonPackRepository:
    """Own guarded comparison-pack paths and canonical manifest persistence."""

    def __init__(
        self,
        *,
        active_model_folder_path_provider: Callable[[], str],
        logger: object,
    ) -> None:
        """Initialize the comparison pack repository and its owned state."""
        self._active_model_folder_path_provider = active_model_folder_path_provider
        self._logger = logger

    def get_root(self) -> str | None:
        """Return the guarded comparison root for the active model, if available."""

        model_folder_path = str(self._active_model_folder_path_provider() or "")
        if not model_folder_path:
            return None
        try:
            comparison_root = StoragePathGuard(model_folder_path).resolve(
                TENSORBOARD_DIR_NAME,
                TENSORBOARD_COMPARISON_DIR_NAME,
            )
        except StoragePathViolation as error:
            self._warning(
                "Rejected unsafe comparison-pack root for "
                f"'{model_folder_path}': {error}"
            )
            return None
        return str(comparison_root)

    @staticmethod
    def resolve_pack_path(
        comparison_root: str,
        pack_name: ComparisonPackStorageName,
    ) -> Path:
        """Resolve one validated pack name beneath the comparison root."""

        return StoragePathGuard(comparison_root).resolve(pack_name)

    def list_packs(self) -> tuple[ComparisonPackRecord, ...]:
        """List guarded pack directories without following symbolic links."""

        comparison_root = self.get_root()
        if comparison_root is None or not os.path.isdir(comparison_root):
            return ()

        records: list[ComparisonPackRecord] = []
        with os.scandir(comparison_root) as children:
            for child in children:
                if not child.is_dir(follow_symlinks=False):
                    continue
                raw_name = str(child.name or "").strip()
                try:
                    pack_name = ComparisonPackStorageName.parse(raw_name)
                    pack_path = self.resolve_pack_path(
                        comparison_root,
                        pack_name,
                    )
                except (InvalidStorageName, StoragePathViolation) as error:
                    self._warning(
                        "Skipped unsafe comparison-pack directory "
                        f"'{raw_name}': {error}"
                    )
                    continue

                manifest = self.read_manifest(str(pack_path))
                annotation = None
                source_paths: tuple[str, ...] = ()
                if manifest is not None:
                    annotation = (
                        str(manifest.get("annotation") or "").strip() or None
                    )
                    source_paths = self.source_paths_from_manifest(manifest)
                records.append(
                    ComparisonPackRecord(
                        name=str(pack_name),
                        path=str(pack_path),
                        annotation=annotation,
                        source_paths=source_paths,
                    )
                )
        return tuple(records)

    @staticmethod
    def manifest_path(pack_path: str) -> str:
        """Return the canonical manifest sidecar path for one pack."""

        return os.path.join(pack_path, COMPARISON_PACK_MANIFEST_FILE_NAME)

    @classmethod
    def load_annotation(cls, pack_path: str) -> str | None:
        """Load the optional annotation from a canonical manifest."""

        comparison_pack = cls.read_manifest(pack_path)
        if comparison_pack is None:
            return None
        return str(comparison_pack.get("annotation") or "").strip() or None

    @classmethod
    def load_source_paths(cls, pack_path: str) -> list[str] | None:
        """Load source paths recorded by a canonical manifest."""

        comparison_pack = cls.read_manifest(pack_path)
        if comparison_pack is None:
            return None
        source_paths = cls.source_paths_from_manifest(comparison_pack)
        return list(source_paths) or None

    @staticmethod
    def source_paths_from_manifest(
        comparison_pack: dict[str, object],
    ) -> tuple[str, ...]:
        """Normalize the source path collection in a validated payload."""

        raw_source_paths = comparison_pack.get("source_paths")
        if not isinstance(raw_source_paths, list):
            return ()

        source_paths: list[str] = []
        for source_path in raw_source_paths:
            cleaned_source_path = str(source_path or "").strip()
            if cleaned_source_path:
                source_paths.append(cleaned_source_path)
        return tuple(source_paths)

    @classmethod
    def read_manifest(cls, pack_path: str) -> dict[str, object] | None:
        """Read optional metadata and fail closed for an existing bad sidecar."""

        manifest_path = cls.manifest_path(pack_path)
        if not os.path.isfile(manifest_path):
            return None
        try:
            payload = read_yaml_mapping_file(manifest_path)
            validated_payload = validate_versioned_metadata_payload(
                MetadataKind.COMPARISON_PACK_MANIFEST,
                payload,
            )
        except Exception as error:
            raise ComparisonPackMetadataError(
                "Comparison-pack metadata is unreadable or uses an "
                "unsupported schema."
            ) from error
        return dict(validated_payload["comparison_pack"])

    @classmethod
    def write_manifest(
        cls,
        pack_path: str,
        pack_name: str,
        annotation: str | None,
        source_paths: Sequence[str] | None = None,
    ) -> None:
        """Atomically write and verify one versioned comparison-pack manifest."""

        os.makedirs(pack_path, exist_ok=True)
        comparison_pack_payload: dict[str, object] = {
            "name": str(pack_name or "").strip(),
            "annotation": str(annotation or "").strip() or None,
            "updated_at": utc_now_iso(),
        }
        cleaned_source_paths = [
            cleaned_path
            for cleaned_path in (
                str(source_path or "").strip()
                for source_path in (source_paths or ())
            )
            if cleaned_path
        ]
        if cleaned_source_paths:
            comparison_pack_payload["source_paths"] = cleaned_source_paths
        payload = {
            "schema_version": 1,
            "comparison_pack": comparison_pack_payload,
        }
        atomic_write_yaml_file(
            cls.manifest_path(pack_path),
            payload,
            verify_reader=read_yaml_mapping_file,
            smoke_check_reader=read_yaml_mapping_file,
        )

    @classmethod
    def confirm_annotation_readback(
        cls,
        pack_path: str,
        *,
        pack_name: str,
        annotation: str | None,
        source_paths: Sequence[str] | None,
    ) -> None:
        """Fail if a just-written annotation manifest cannot be read back exactly."""

        comparison_pack = cls.read_manifest(pack_path)
        if comparison_pack is None:
            raise ComparisonPackMetadataError(
                "Comparison-pack metadata disappeared during annotation update."
            )
        readback_name = str(comparison_pack.get("name") or "").strip()
        readback_annotation = (
            str(comparison_pack.get("annotation") or "").strip() or None
        )
        readback_sources = cls.source_paths_from_manifest(comparison_pack)
        if (
            readback_name != pack_name
            or readback_annotation != annotation
            or readback_sources != tuple(source_paths or ())
        ):
            raise ComparisonPackMetadataError(
                "Comparison-pack annotation readback did not match the write."
            )

    def _warning(self, message: str) -> None:
        """Publish a warning record through this comparison pack repository."""
        warning = getattr(self._logger, "warning", None)
        if not callable(warning):
            return
        try:
            warning(message)
        except Exception:
            return


__all__ = [
    "COMPARISON_PACK_MANIFEST_FILE_NAME",
    "ComparisonPackMetadataError",
    "ComparisonPackRecord",
    "ComparisonPackRepository",
]
