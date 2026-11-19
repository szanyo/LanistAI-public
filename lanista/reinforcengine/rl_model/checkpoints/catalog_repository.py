# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Read-only filesystem repository for checkpoint catalog artifacts."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from lanista.reinforcengine.rl_model.checkpoints.catalog_contracts import (
    CheckpointArtifactSnapshot,
)
from lanista.reinforcengine.rl_model.checkpoints.metadata import load_checkpoint_manifest
from lanista.reinforcengine.rl_model.checkpoints.name_metadata import (
    extract_checkpoint_steps,
)
from lanista.reinforcengine.rl_model.checkpoints.storage_resolver import (
    CheckpointStorageResolver,
)
from lanista.reinforcengine.rl_model.storage.layout import CHECKPOINT_MANIFEST_FILE_NAME
from lanista.reinforcengine.rl_model.storage.names import InvalidStorageName
from lanista.reinforcengine.rl_model.storage.path_guard import StoragePathViolation
from lanista.reinforcengine.rl_model.storage.transaction import TRANSACTION_DIRECTORY_NAME


class CheckpointManifestLoader(Protocol):
    """Read optional presentation metadata for one checkpoint artifact."""

    def __call__(
        self,
        checkpoint_path: str | os.PathLike[str],
    ) -> Mapping[str, object]:
        """Invoke this checkpoint manifest loader as a callable."""
        ...


class CheckpointMtimeProvider(Protocol):
    """Return one artifact modification time for sorting or projection."""

    def __call__(self, checkpoint_path: str | os.PathLike[str]) -> float:
        """Invoke this checkpoint mtime provider as a callable."""
        ...


def checkpoint_mtime(checkpoint_path: str | os.PathLike[str]) -> float:
    """Return an artifact mtime through a narrowly typed test seam."""

    return os.path.getmtime(checkpoint_path)


class CheckpointCatalogRepository:
    """Discover checkpoint artifacts without acquiring mutation capabilities."""

    def __init__(
        self,
        *,
        sessions_root_provider: Callable[[], str],
        session_framework_provider: Callable[[str], str],
        framework_normalizer: Callable[[str | None], str | None],
        step_extractor: Callable[[str], int] = extract_checkpoint_steps,
        manifest_loader: CheckpointManifestLoader = load_checkpoint_manifest,
        mtime_provider: CheckpointMtimeProvider = checkpoint_mtime,
        resolver: CheckpointStorageResolver | None = None,
    ) -> None:
        """Initialize the checkpoint catalog repository and its owned state."""
        self._sessions_root_provider = sessions_root_provider
        self._session_framework_provider = session_framework_provider
        self._framework_normalizer = framework_normalizer
        self._step_extractor = step_extractor
        self._manifest_loader = manifest_loader
        self._mtime_provider = mtime_provider
        self._resolver = resolver or CheckpointStorageResolver()

    def list_for_session(
        self,
        session_name: str,
        framework_uuid: str | None = None,
    ) -> list[str]:
        """Return existing checkpoint names in established newest-first order."""

        if not session_name or not self._framework_matches(
            session_name,
            framework_uuid,
        ):
            return []

        checkpoints_root = self._soft_checkpoints_root(session_name)
        if checkpoints_root is None or not checkpoints_root.is_dir():
            return []

        checkpoints = self._organized_checkpoint_names(checkpoints_root)
        checkpoints.extend(self._direct_checkpoint_names(checkpoints_root))
        existing_names = [
            name
            for name in dict.fromkeys(checkpoints)
            if self._resolved_artifact_exists(checkpoints_root, name)
        ]
        existing_names.sort(
            key=lambda name: self._checkpoint_sort_key(
                checkpoints_root,
                name,
            ),
            reverse=True,
        )
        return existing_names

    def has_checkpoints(
        self,
        session_name: str | None,
        framework_uuid: str | None = None,
    ) -> bool:
        """Return whether a valid session contains a matching artifact."""

        if not session_name:
            return False
        return bool(
            self.list_for_session(
                session_name,
                framework_uuid=framework_uuid,
            )
        )

    def read_artifact(
        self,
        session_name: str,
        checkpoint_name: str,
    ) -> CheckpointArtifactSnapshot:
        """Read one path/stat/manifest snapshot with presentation fallbacks."""

        checkpoints_root = self._soft_checkpoints_root(session_name)
        checkpoint_path = (
            self._resolver.try_resolve_candidate_from_root(
                checkpoints_root,
                checkpoint_name,
            )
            if checkpoints_root is not None
            else None
        )
        exists = bool(checkpoint_path and os.path.exists(checkpoint_path))
        manifest: Mapping[str, object] = (
            dict(self._manifest_loader(checkpoint_path))
            if checkpoint_path is not None
            else {}
        )
        return CheckpointArtifactSnapshot(
            path=checkpoint_path,
            exists=exists,
            saved_at_raw=self._saved_at_from_stat(
                checkpoint_path,
                exists=exists,
            ),
            checkpoint_kind=(
                "file"
                if checkpoint_path is not None
                and os.path.isfile(checkpoint_path)
                else "directory"
            ),
            manifest=manifest,
        )

    def _framework_matches(
        self,
        session_name: str,
        framework_uuid: str | None,
    ) -> bool:
        """Return the framework matches value owned by this checkpoint catalog repository."""
        normalized_framework = self._framework_normalizer(framework_uuid)
        return not normalized_framework or (
            self._session_framework_provider(session_name)
            == normalized_framework
        )

    def _soft_checkpoints_root(self, session_name: str) -> Path | None:
        """Return the soft checkpoints root value owned by this checkpoint catalog repository."""
        try:
            return self._resolver.checkpoints_root(
                self._sessions_root_provider(),
                session_name,
            )
        except (InvalidStorageName, StoragePathViolation, OSError):
            return None

    def _resolved_artifact_exists(
        self,
        checkpoints_root: Path,
        checkpoint_name: str,
    ) -> bool:
        """Return the resolved artifact exists value owned by this checkpoint catalog repository."""
        resolved = self._resolver.try_resolve_candidate_from_root(
            checkpoints_root,
            checkpoint_name,
        )
        return resolved is not None and os.path.lexists(resolved)

    def _checkpoint_sort_key(
        self,
        checkpoints_root: Path,
        checkpoint_name: str,
    ) -> tuple[int, float, str]:
        """Checkpoint the sort key owned by this checkpoint catalog repository."""
        checkpoint_path = self._resolver.resolve_candidate_from_root(
            checkpoints_root,
            checkpoint_name,
        )
        mtime = (
            self._mtime_provider(checkpoint_path)
            if os.path.exists(checkpoint_path)
            else 0
        )
        return self._step_extractor(checkpoint_name), mtime, checkpoint_name

    def _saved_at_from_stat(
        self,
        checkpoint_path: Path | None,
        *,
        exists: bool,
    ) -> str | None:
        """Return the saved at from stat value owned by this checkpoint catalog repository."""
        if not exists or checkpoint_path is None:
            return None
        try:
            return (
                datetime.fromtimestamp(
                    self._mtime_provider(checkpoint_path),
                    UTC,
                )
                .replace(microsecond=0)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except Exception:
            # A failed optional stat must not make catalog inspection fatal.
            return None

    @staticmethod
    def _organized_checkpoint_names(checkpoints_root: Path) -> list[str]:
        """Return the organized checkpoint names value owned by this checkpoint catalog repository.
        """
        checkpoints: list[str] = []
        organized_root = checkpoints_root / "stablebaselines"
        if organized_root.is_symlink() or not organized_root.is_dir():
            return checkpoints

        for entry in os.listdir(organized_root):
            full = organized_root / entry
            if full.is_symlink():
                continue
            if full.is_file() and entry.lower().endswith(".zip"):
                checkpoints.append(entry)
                continue
            if not full.is_dir():
                continue
            expected_name = f"{entry}.zip"
            if (full / expected_name).is_file():
                checkpoints.append(expected_name)
                continue
            zip_entries = sorted(
                child
                for child in os.listdir(full)
                if child.lower().endswith(".zip")
                and (full / child).is_file()
            )
            if zip_entries:
                checkpoints.append(zip_entries[-1])
        return checkpoints

    @staticmethod
    def _direct_checkpoint_names(checkpoints_root: Path) -> list[str]:
        """Return the direct checkpoint names value owned by this checkpoint catalog repository."""
        checkpoints: list[str] = []
        for entry in os.listdir(checkpoints_root):
            if (
                entry == CHECKPOINT_MANIFEST_FILE_NAME
                or entry.endswith(f".{CHECKPOINT_MANIFEST_FILE_NAME}")
                or entry in {"stablebaselines", TRANSACTION_DIRECTORY_NAME}
            ):
                continue
            full = checkpoints_root / entry
            if full.is_symlink():
                continue
            if full.is_dir() or full.is_file():
                checkpoints.append(entry)
        return checkpoints
