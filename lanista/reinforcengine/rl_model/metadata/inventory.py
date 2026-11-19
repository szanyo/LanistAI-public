# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Scoped inventory of versioned model-storage metadata files."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from lanista.reinforcengine.rl_model.metadata.schema import MetadataKind
from lanista.reinforcengine.rl_model.storage.layout import (
    CHECKPOINT_MANIFEST_FILE_NAME,
    CLONE_MANIFEST_FILE_NAME,
    FRAMEWORK_MODEL_CONFIGURATION_FILE_NAME,
    MODEL_CONFIGURATION_FILE_NAME,
    MODEL_MANIFEST_FILE_NAME,
    SESSION_MANIFEST_FILE_NAME,
)
from lanista.reinforcengine.rl_model.storage.path_guard import StoragePathGuard

COMPARISON_PACK_MANIFEST_FILE_NAME = "comparison_pack_manifest.yml"
RUNTIME_STATE_FILE_NAME = "runtime_state.yaml"


@dataclass(frozen=True, slots=True)
class MetadataInventoryEntry:
    """One first-party YAML metadata file selected for validation."""

    path: Path
    relative_path: str
    kind: MetadataKind
    byte_size: int
    sha256: str
    migrated_sha256: str = ""


class MetadataInventoryService:
    """Find only known metadata locations without reading checkpoint binaries."""

    def inventory(self, root: str | os.PathLike[str]) -> tuple[MetadataInventoryEntry, ...]:
        """Return the inventory value owned by this metadata inventory service."""
        guard = StoragePathGuard(root)
        if not guard.root.is_dir():
            raise NotADirectoryError(guard.root)

        candidates: set[Path] = set()
        model_roots = self._model_roots(guard.root)
        self._add_direct_metadata(guard.root, candidates)
        for model_root in model_roots:
            self._add_model_metadata(model_root, candidates)

        entries: list[MetadataInventoryEntry] = []
        for candidate in sorted(candidates):
            guarded_path = guard.require_descendant(candidate, must_exist=True)
            kind = self.classify(guarded_path)
            entries.append(
                MetadataInventoryEntry(
                    path=guarded_path,
                    relative_path=guard.relative_path(guarded_path),
                    kind=kind,
                    byte_size=guarded_path.stat().st_size,
                    sha256=sha256_file(guarded_path),
                )
            )
        return tuple(entries)

    @staticmethod
    def classify(path: Path) -> MetadataKind:
        """Classify the operation owned by this metadata inventory service."""
        name = path.name
        if name == MODEL_MANIFEST_FILE_NAME:
            return MetadataKind.MODEL_MANIFEST
        if name == MODEL_CONFIGURATION_FILE_NAME:
            return MetadataKind.MODEL_CONFIGURATION
        if name == FRAMEWORK_MODEL_CONFIGURATION_FILE_NAME:
            return MetadataKind.FRAMEWORK_CONFIGURATION
        if name == SESSION_MANIFEST_FILE_NAME:
            return MetadataKind.SESSION_MANIFEST
        if name == CHECKPOINT_MANIFEST_FILE_NAME or name.endswith(f".{CHECKPOINT_MANIFEST_FILE_NAME}"):
            return MetadataKind.CHECKPOINT_MANIFEST
        if name == COMPARISON_PACK_MANIFEST_FILE_NAME:
            return MetadataKind.COMPARISON_PACK_MANIFEST
        if name == CLONE_MANIFEST_FILE_NAME:
            return MetadataKind.CLONE_MANIFEST
        if name == RUNTIME_STATE_FILE_NAME:
            return MetadataKind.RUNTIME_STATE
        raise ValueError(f"Unsupported metadata file: {name}")

    @staticmethod
    def _model_roots(root: Path) -> tuple[Path, ...]:
        """Return the model roots value owned by this metadata inventory service."""
        if MetadataInventoryService._looks_like_model_root(root):
            return (root,)
        return tuple(
            entry
            for entry in sorted(root.iterdir())
            if entry.is_dir() and MetadataInventoryService._looks_like_model_root(entry)
        )

    @staticmethod
    def _looks_like_model_root(path: Path) -> bool:
        """Return the looks like model root value owned by this metadata inventory service."""
        return any(
            (path / marker).exists()
            for marker in (
                MODEL_MANIFEST_FILE_NAME,
                MODEL_CONFIGURATION_FILE_NAME,
                "training_sessions",
                "resources",
                "tensorboard",
            )
        )

    @staticmethod
    def _add_direct_metadata(root: Path, candidates: set[Path]) -> None:
        """Add the direct metadata owned by this metadata inventory service."""
        for name in (
            MODEL_MANIFEST_FILE_NAME,
            MODEL_CONFIGURATION_FILE_NAME,
            CLONE_MANIFEST_FILE_NAME,
            RUNTIME_STATE_FILE_NAME,
        ):
            candidate = root / name
            if candidate.is_file():
                candidates.add(candidate)

    @classmethod
    def _add_model_metadata(cls, model_root: Path, candidates: set[Path]) -> None:
        """Add the model metadata owned by this metadata inventory service."""
        cls._add_direct_metadata(model_root, candidates)
        candidates.update(
            path
            for path in (model_root / "resources").glob(f"*/{FRAMEWORK_MODEL_CONFIGURATION_FILE_NAME}")
            if path.is_file()
        )
        sessions_root = model_root / "training_sessions"
        if sessions_root.is_dir():
            candidates.update(path for path in sessions_root.glob(f"*/{SESSION_MANIFEST_FILE_NAME}") if path.is_file())
            candidates.update(
                path
                for path in sessions_root.rglob(f"*{CHECKPOINT_MANIFEST_FILE_NAME}")
                if path.is_file()
            )
        comparison_root = model_root / "tensorboard" / "comparison"
        if comparison_root.is_dir():
            candidates.update(
                path
                for path in comparison_root.glob(f"*/{COMPARISON_PACK_MANIFEST_FILE_NAME}")
                if path.is_file()
            )


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash one metadata or explicitly selected verification file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
