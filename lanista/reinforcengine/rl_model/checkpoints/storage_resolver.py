# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Single-owner checkpoint layout and guarded path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from lanista.reinforcengine.rl_model.storage.layout import CHECKPOINT_MANIFEST_FILE_NAME
from lanista.reinforcengine.rl_model.storage.names import (
    CheckpointStorageName,
    InvalidStorageName,
    SessionStorageName,
)
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)


@dataclass(frozen=True, slots=True)
class CheckpointStoragePaths:
    """Guarded direct and Stable-Baselines paths for one logical checkpoint."""

    direct: Path
    organized_flat: Path
    organized_root: Path
    organized_file: Path


class CheckpointStorageResolver:
    """Resolve every supported checkpoint layout through one path guard."""

    @staticmethod
    def checkpoints_root(
        sessions_root: str | os.PathLike[str],
        session_name: str,
    ) -> Path:
        """Resolve one validated session's canonical checkpoint directory."""

        guard = StoragePathGuard(sessions_root)
        session_storage_name = SessionStorageName.parse(session_name)
        return guard.resolve(session_storage_name, "checkpoints")

    @staticmethod
    def storage_paths(
        checkpoints_root: str | os.PathLike[str],
        checkpoint_name: str,
    ) -> tuple[StoragePathGuard, CheckpointStoragePaths]:
        """Build every supported guarded path for one logical artifact."""

        guard = StoragePathGuard(checkpoints_root)
        storage_name = CheckpointStorageName.parse(checkpoint_name)
        stem_name = CheckpointStorageName.parse(Path(storage_name.value).stem)
        return guard, CheckpointStoragePaths(
            direct=guard.resolve(storage_name),
            organized_flat=guard.resolve("stablebaselines", storage_name),
            organized_root=guard.resolve("stablebaselines", stem_name),
            organized_file=guard.resolve(
                "stablebaselines",
                stem_name,
                storage_name,
            ),
        )

    @classmethod
    def resolve_candidate_from_root(
        cls,
        checkpoints_root: str | os.PathLike[str],
        checkpoint_name: str,
    ) -> Path:
        """Resolve the current artifact or its guarded direct fallback path."""

        guard, paths = cls.storage_paths(checkpoints_root, checkpoint_name)
        for candidate in (
            paths.direct,
            paths.organized_flat,
            paths.organized_file,
        ):
            if os.path.lexists(candidate):
                return candidate

        if not checkpoint_name.lower().endswith(".zip"):
            zipped_name = CheckpointStorageName.parse(f"{checkpoint_name}.zip")
            organized_flat_zip = guard.resolve("stablebaselines", zipped_name)
            organized_zip = guard.resolve(
                "stablebaselines",
                CheckpointStorageName.parse(Path(checkpoint_name).stem),
                zipped_name,
            )
            if os.path.lexists(organized_flat_zip):
                return organized_flat_zip
            if os.path.lexists(organized_zip):
                return organized_zip

        return paths.direct

    @classmethod
    def try_resolve_candidate_from_root(
        cls,
        checkpoints_root: str | os.PathLike[str],
        checkpoint_name: str,
    ) -> Path | None:
        """Return ``None`` when untrusted input cannot form a guarded path."""

        try:
            return cls.resolve_candidate_from_root(
                checkpoints_root,
                checkpoint_name,
            )
        except (InvalidStorageName, StoragePathViolation):
            return None

    @staticmethod
    def manifest_sidecar(
        guard: StoragePathGuard,
        checkpoint_path: Path,
    ) -> Path:
        """Return the guarded sidecar used by file checkpoint artifacts."""

        return guard.require_descendant(
            f"{checkpoint_path}.{CHECKPOINT_MANIFEST_FILE_NAME}"
        )
