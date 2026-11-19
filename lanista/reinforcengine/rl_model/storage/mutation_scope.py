# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Canonical lock scope shared by every model-storage writer."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lanista.reinforcengine.rl_model.storage.layout import (
    MODEL_MANIFEST_FILE_NAME,
    TRAINING_SESSIONS_DIR_NAME,
)
from lanista.reinforcengine.rl_model.storage.mutation_lock import StorageMutationLock
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)


class StorageMutationLockFactory(Protocol):
    """Construct a lock for one canonical models-storage root."""

    def __call__(
        self,
        root: str | os.PathLike[str],
        *,
        timeout_seconds: float,
    ) -> StorageMutationLock:
        """Invoke this storage mutation lock factory as a callable."""
        ...


def _existing_lexical_directory(
    path: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    """Return the existing lexical directory value owned by this storage mutation scope module."""
    raw_path = Path(path).expanduser()
    lexical_path = Path(os.path.abspath(raw_path))
    canonical_path = StoragePathGuard(lexical_path).root
    if lexical_path != canonical_path or lexical_path.is_symlink():
        raise StoragePathViolation(f"{label} aliases and symbolic links are forbidden.")
    if not lexical_path.is_dir():
        raise FileNotFoundError(f"{label} is unavailable: {lexical_path}")
    return canonical_path


@dataclass(frozen=True, slots=True)
class ModelsStorageMutationScope:
    """Resolve every nested mutation to one stable ``models_root`` lock.

    Writers acquire this scope before filesystem preflight and retain it through
    compare-and-swap validation, transaction commit, and recovery. Locking only
    the nested model/session/checkpoint directory would let model deletion race
    those operations, so mixed lock scopes are deliberately unsupported.
    """

    models_root: Path

    @classmethod
    def from_models_root(
        cls,
        models_root: str | os.PathLike[str],
    ) -> ModelsStorageMutationScope:
        """Create this models storage mutation scope's value from models root."""
        return cls(
            _existing_lexical_directory(
                models_root,
                label="Models storage root",
            )
        )

    @classmethod
    def for_model_root(
        cls,
        model_root: str | os.PathLike[str],
        *,
        models_root: str | os.PathLike[str] | None = None,
    ) -> ModelsStorageMutationScope:
        """Return the for model root value owned by this models storage mutation scope."""
        canonical_model_root = _existing_lexical_directory(
            model_root,
            label="Model storage root",
        )
        scope = cls.from_models_root(
            models_root if models_root is not None else canonical_model_root.parent
        )
        guarded_model_root = StoragePathGuard(scope.models_root).require_descendant(
            canonical_model_root,
            must_exist=True,
        )
        if (
            guarded_model_root != canonical_model_root
            or canonical_model_root.parent != scope.models_root
        ):
            raise StoragePathViolation(
                "A model storage root must be a direct child of models storage."
            )
        return scope

    @classmethod
    def for_sessions_root(
        cls,
        sessions_root: str | os.PathLike[str],
        *,
        models_root: str | os.PathLike[str] | None = None,
    ) -> ModelsStorageMutationScope:
        """Return the for sessions root value owned by this models storage mutation scope."""
        canonical_sessions_root = _existing_lexical_directory(
            sessions_root,
            label="Training sessions root",
        )
        if canonical_sessions_root.name != TRAINING_SESSIONS_DIR_NAME:
            raise StoragePathViolation(
                "Training sessions storage must use the managed directory name."
            )
        return cls.for_model_root(
            canonical_sessions_root.parent,
            models_root=models_root,
        )

    @classmethod
    def for_migration_root(
        cls,
        migration_root: str | os.PathLike[str],
        *,
        models_root: str | os.PathLike[str] | None = None,
    ) -> ModelsStorageMutationScope:
        """Accept the CLI's documented model-root or models-root scope."""

        canonical_migration_root = _existing_lexical_directory(
            migration_root,
            label="Metadata migration root",
        )
        if models_root is not None:
            explicit_scope = cls.from_models_root(models_root)
            if canonical_migration_root == explicit_scope.models_root:
                return explicit_scope
            return cls.for_model_root(
                canonical_migration_root,
                models_root=explicit_scope.models_root,
            )

        if os.path.lexists(canonical_migration_root / MODEL_MANIFEST_FILE_NAME):
            return cls.for_model_root(canonical_migration_root)
        return cls.from_models_root(canonical_migration_root)

    def lock(
        self,
        *,
        lock_factory: StorageMutationLockFactory = StorageMutationLock,
        timeout_seconds: float = 5.0,
    ) -> StorageMutationLock:
        """Create the sole lock a mutation may hold for this storage scope."""

        return lock_factory(
            self.models_root,
            timeout_seconds=timeout_seconds,
        )
