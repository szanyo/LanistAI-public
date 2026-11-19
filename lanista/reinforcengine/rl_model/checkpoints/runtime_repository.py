# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Serialized runtime checkpoint persistence below one model-storage root."""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from lanista.reinforcengine.rl_model.checkpoints.metadata import (
    update_last_checkpoint_metadata,
    write_checkpoint_manifest,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    CHECKPOINT_MANIFEST_FILE_NAME,
    TRAINING_SESSIONS_DIR_NAME,
    find_existing_session_manifest_path,
)
from lanista.reinforcengine.rl_model.storage.mutation_lock import StorageMutationLock
from lanista.reinforcengine.rl_model.storage.mutation_scope import (
    ModelsStorageMutationScope,
    StorageMutationLockFactory,
)
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.storage.transaction import (
    StorageTransaction,
    StorageTransactionError,
)


class CheckpointArtifactKind(StrEnum):
    """Declare the on-disk shape and publication boundary of an artifact."""

    DIRECTORY = "directory"
    FILE = "file"
    FILE_BUNDLE = "file-bundle"


class _WarningLogger(Protocol):
    """Record a non-fatal metadata projection failure."""

    def warning(self, message: str) -> object:
        """Record one warning message."""
        ...


@dataclass(slots=True)
class _StagedCheckpoint:
    """Carry the data required for staged checkpoint."""

    artifact_kind: CheckpointArtifactKind
    final_path: Path
    staging_path: Path
    publication_final_path: Path
    publication_staging_path: Path
    staged_manifest_path: Path | None = None
    final_manifest_path: Path | None = None


@dataclass(slots=True)
class RuntimeCheckpointMutation:
    """Checkpoint writer available only while its repository transaction is active."""

    session_root: Path
    _transaction: StorageTransaction
    _failure: Exception | None = None
    _checkpoints: dict[Path, _StagedCheckpoint] = field(
        default_factory=dict
    )
    _staged_session_manifest: Path | None = None
    _final_session_manifest: Path | None = None

    def _mark_failed(self, exc: Exception) -> None:
        """Mark the failed owned by this runtime checkpoint mutation."""
        if self._failure is None:
            self._failure = exc

    def _raise_if_failed(self) -> None:
        """Raise the if failed owned by this runtime checkpoint mutation."""
        if self._failure is not None:
            raise self._failure

    def _checkpoints_root(self) -> Path:
        """Return the checkpoints root value owned by this runtime checkpoint mutation."""
        checkpoints_root = self.session_root / "checkpoints"
        if checkpoints_root.is_symlink() or not checkpoints_root.is_dir():
            raise StoragePathViolation(
                "Runtime checkpoint storage is unavailable or aliased."
            )
        return checkpoints_root

    def _checkpoint_path(
        self,
        checkpoint_path: str,
        *,
        must_exist: bool,
    ) -> Path:
        """Checkpoint the path owned by this runtime checkpoint mutation."""
        checkpoints_root = self._checkpoints_root()
        raw_path = Path(checkpoint_path).expanduser()
        if ".." in raw_path.parts:
            raise StoragePathViolation(
                "Runtime checkpoint targets cannot contain parent traversal."
            )
        lexical_path = Path(os.path.abspath(raw_path))
        guarded_path = StoragePathGuard(checkpoints_root).require_descendant(
            lexical_path,
            must_exist=must_exist,
        )
        if lexical_path != guarded_path or lexical_path.is_symlink():
            raise StoragePathViolation(
                "Runtime checkpoint artifacts cannot use storage aliases."
            )
        return guarded_path

    @staticmethod
    def _publication_paths(
        target: Path,
        staging_path: Path,
        checkpoints_root: Path,
        artifact_kind: CheckpointArtifactKind,
    ) -> tuple[Path, Path]:
        """Return the publication paths value owned by this runtime checkpoint mutation."""
        if artifact_kind is CheckpointArtifactKind.FILE_BUNDLE:
            if target.parent == checkpoints_root:
                raise StoragePathViolation(
                    "A file checkpoint bundle must be nested below its "
                    "checkpoint directory."
                )
            if os.path.lexists(target.parent):
                raise FileExistsError(target.parent)
            return target.parent, staging_path.parent
        return target, staging_path

    def prepare_checkpoint_target(
        self,
        checkpoint_path: str,
        *,
        artifact_kind: CheckpointArtifactKind,
    ) -> str:
        """Return a hidden staging path while reserving the final target."""

        checkpoints_root = self._checkpoints_root()
        target = self._checkpoint_path(checkpoint_path, must_exist=False)
        try:
            if not isinstance(artifact_kind, CheckpointArtifactKind):
                raise TypeError(
                    "Runtime checkpoint artifact kind must use "
                    "CheckpointArtifactKind."
                )
            self._first_missing_parent(target.parent, checkpoints_root)
            staging_path = self._transaction.reserve_staging_path(target)
            publication_final_path, publication_staging_path = (
                self._publication_paths(
                    target,
                    staging_path,
                    checkpoints_root,
                    artifact_kind,
                )
            )

            missing_root = self._first_missing_parent(
                publication_final_path.parent,
                checkpoints_root,
            )
            if missing_root is not None:
                self._transaction.track_created(missing_root)

            self._checkpoints[staging_path] = _StagedCheckpoint(
                artifact_kind=artifact_kind,
                final_path=target,
                staging_path=staging_path,
                publication_final_path=publication_final_path,
                publication_staging_path=publication_staging_path,
            )
            return str(staging_path)
        except Exception as exc:
            self._mark_failed(exc)
            raise

    def _staged_checkpoint(
        self,
        checkpoint_path: str,
        *,
        must_exist: bool,
    ) -> _StagedCheckpoint:
        """Return the staged checkpoint value owned by this runtime checkpoint mutation."""
        lexical_path = Path(
            os.path.abspath(Path(checkpoint_path).expanduser())
        )
        checkpoint = self._checkpoints.get(lexical_path)
        if checkpoint is None or lexical_path.is_symlink():
            raise StoragePathViolation(
                "Runtime checkpoint writes require a repository staging path."
            )
        if must_exist and not os.path.lexists(lexical_path):
            raise FileNotFoundError(lexical_path)
        return checkpoint

    @staticmethod
    def _manifest_paths(
        checkpoint: _StagedCheckpoint,
    ) -> tuple[Path, Path]:
        """Return the manifest paths value owned by this runtime checkpoint mutation."""
        if checkpoint.artifact_kind is CheckpointArtifactKind.DIRECTORY:
            return (
                checkpoint.staging_path / CHECKPOINT_MANIFEST_FILE_NAME,
                checkpoint.final_path / CHECKPOINT_MANIFEST_FILE_NAME,
            )
        return (
            Path(
                f"{checkpoint.staging_path}."
                f"{CHECKPOINT_MANIFEST_FILE_NAME}"
            ),
            Path(
                f"{checkpoint.final_path}."
                f"{CHECKPOINT_MANIFEST_FILE_NAME}"
            ),
        )

    @staticmethod
    def _validate_artifact_shape(checkpoint: _StagedCheckpoint) -> None:
        """Validate the artifact shape required by this runtime checkpoint mutation."""
        if checkpoint.artifact_kind is CheckpointArtifactKind.DIRECTORY:
            valid = checkpoint.staging_path.is_dir()
        else:
            valid = checkpoint.staging_path.is_file()
        if not valid:
            raise StorageTransactionError(
                "Runtime checkpoint artifact does not match its declared "
                f"{checkpoint.artifact_kind.value} kind."
            )

    @staticmethod
    def _first_missing_parent(
        parent: Path,
        checkpoints_root: Path,
    ) -> Path | None:
        """Return the first missing parent value owned by this runtime checkpoint mutation."""
        missing: list[Path] = []
        current = parent
        while current != checkpoints_root and not os.path.lexists(current):
            missing.append(current)
            current = current.parent
        if current != checkpoints_root and (
            current.is_symlink() or not current.is_dir()
        ):
            raise StoragePathViolation(
                "Runtime checkpoint parents are unavailable or aliased."
            )
        return missing[-1] if missing else None

    @staticmethod
    def _guarded_lexical_path(
        root: Path,
        candidate: Path,
        *,
        must_exist: bool,
        error_message: str,
    ) -> Path:
        """Return the guarded lexical path value owned by this runtime checkpoint mutation."""
        lexical_path = Path(os.path.abspath(candidate))
        guarded_path = StoragePathGuard(root).require_descendant(
            lexical_path,
            must_exist=must_exist,
        )
        if lexical_path != guarded_path or lexical_path.is_symlink():
            raise StoragePathViolation(error_message)
        return guarded_path

    def _prepare_final_replacement(
        self,
        root: Path,
        path: Path,
    ) -> None:
        """Prepare the final replacement required by this runtime checkpoint mutation."""
        guarded_path = self._guarded_lexical_path(
            root,
            path,
            must_exist=False,
            error_message=(
                "Runtime checkpoint publication cannot use storage aliases."
            ),
        )
        if os.path.lexists(guarded_path):
            self._transaction.quarantine(guarded_path)

    @staticmethod
    def _manifest_is_bundled(checkpoint: _StagedCheckpoint) -> bool:
        """Return the manifest is bundled value owned by this runtime checkpoint mutation."""
        if (
            checkpoint.staged_manifest_path is None
            or checkpoint.final_manifest_path is None
        ):
            return False
        try:
            staged_relative = checkpoint.staged_manifest_path.relative_to(
                checkpoint.publication_staging_path
            )
            final_relative = checkpoint.final_manifest_path.relative_to(
                checkpoint.publication_final_path
            )
        except ValueError:
            return False
        return staged_relative == final_relative

    def write_checkpoint_manifest(
        self,
        checkpoint_path: str,
        *,
        checkpoint_name: str,
        checkpoint_kind: str,
        framework_uuid: str,
        scenario_index: int | None = None,
        curriculum_state: object = None,
        step_count: int | None = None,
        parent_checkpoint_name: str | None = None,
        saved_at: str | None = None,
        annotation: str | None = None,
    ) -> str:
        """Write a sidecar beside the hidden staged checkpoint artifact."""

        checkpoint = self._staged_checkpoint(
            checkpoint_path,
            must_exist=True,
        )
        try:
            self._validate_artifact_shape(checkpoint)
            staged_manifest_path, final_manifest_path = self._manifest_paths(
                checkpoint
            )
            staged_manifest = self._guarded_lexical_path(
                self.session_root,
                staged_manifest_path,
                must_exist=False,
                error_message=(
                    "Runtime checkpoint staging metadata escaped its "
                    "transaction storage."
                ),
            )
            final_manifest = self._guarded_lexical_path(
                self._checkpoints_root(),
                final_manifest_path,
                must_exist=False,
                error_message=(
                    "Runtime checkpoint metadata cannot use storage aliases."
                )
            )
            manifest_path = write_checkpoint_manifest(
                str(checkpoint.staging_path),
                checkpoint_name=checkpoint_name,
                checkpoint_kind=checkpoint_kind,
                framework_uuid=framework_uuid,
                scenario_index=scenario_index,
                curriculum_state=curriculum_state,
                step_count=step_count,
                parent_checkpoint_name=parent_checkpoint_name,
                saved_at=saved_at,
                annotation=annotation,
            )
            if Path(os.path.abspath(manifest_path)) != staged_manifest:
                raise StoragePathViolation(
                    "Runtime checkpoint metadata escaped its staging storage."
                )
            guarded_manifest = StoragePathGuard(
                self.session_root
            ).require_descendant(
                staged_manifest,
                must_exist=True,
            )
            self._transaction.confirm_path(guarded_manifest)
            checkpoint.staged_manifest_path = guarded_manifest
            checkpoint.final_manifest_path = final_manifest
            return str(final_manifest)
        except Exception as exc:
            self._mark_failed(exc)
            raise

    def update_last_checkpoint_metadata(
        self,
        checkpoint_name: str,
        checkpoint_kind: str,
        *,
        logger: _WarningLogger | None = None,
        raise_on_error: bool = False,
    ) -> None:
        """Prepare updated session metadata without exposing it before commit."""

        _ = raise_on_error
        manifest_path = Path(
            os.path.abspath(
                find_existing_session_manifest_path(str(self.session_root))
            )
        )
        try:
            guarded_manifest = StoragePathGuard(
                self.session_root
            ).require_descendant(
                manifest_path,
                must_exist=True,
            )
            if (
                manifest_path != guarded_manifest
                or manifest_path.is_symlink()
                or not guarded_manifest.is_file()
            ):
                raise StoragePathViolation(
                    "Runtime checkpoint session metadata is unavailable or aliased."
                )
            if self._staged_session_manifest is None:
                staged_manifest = self._transaction.reserve_staging_path(
                    guarded_manifest
                )
                shutil.copy2(
                    guarded_manifest,
                    staged_manifest,
                    follow_symlinks=False,
                )
                self._transaction.confirm_path(staged_manifest)
                self._staged_session_manifest = staged_manifest
                self._final_session_manifest = guarded_manifest

            update_last_checkpoint_metadata(
                str(self._staged_session_manifest.parent),
                checkpoint_name,
                checkpoint_kind,
                logger=logger,
                raise_on_error=True,
            )
            self._transaction.confirm_path(self._staged_session_manifest)
        except Exception as exc:
            self._mark_failed(exc)
            raise

    def _publish_checkpoint(self, checkpoint: _StagedCheckpoint) -> None:
        """Publish the checkpoint owned by this runtime checkpoint mutation."""
        staged_manifest = checkpoint.staged_manifest_path
        final_manifest = checkpoint.final_manifest_path
        if staged_manifest is None:
            raise StorageTransactionError(
                "Runtime checkpoint sidecar was not prepared."
            )
        if not self._manifest_is_bundled(checkpoint):
            if final_manifest is None:
                raise StorageTransactionError(
                    "Runtime checkpoint sidecar target is unavailable."
                )
            self._transaction.publish(staged_manifest, final_manifest)
        self._transaction.publish(
            checkpoint.publication_staging_path,
            checkpoint.publication_final_path,
        )

    def _finalize(self) -> None:
        """Publish only complete, durable checkpoint bundles at final paths."""

        self._raise_if_failed()
        if not self._checkpoints and self._staged_session_manifest is None:
            return
        try:
            checkpoints_root = self._checkpoints_root()
            for checkpoint in self._checkpoints.values():
                if checkpoint.staged_manifest_path is None:
                    raise StorageTransactionError(
                        "Runtime checkpoint sidecar was not prepared."
                    )
                self._transaction.confirm_path(checkpoint.staging_path)

                self._prepare_final_replacement(
                    checkpoints_root,
                    checkpoint.publication_final_path,
                )
                if not self._manifest_is_bundled(checkpoint):
                    if checkpoint.final_manifest_path is None:
                        raise StorageTransactionError(
                            "Runtime checkpoint sidecar target is unavailable."
                        )
                    self._prepare_final_replacement(
                        checkpoints_root,
                        checkpoint.final_manifest_path,
                    )

            for checkpoint in self._checkpoints.values():
                self._publish_checkpoint(checkpoint)

            # The session manifest is the durable selection pointer.  Publish
            # it only after every referenced checkpoint bundle is complete so
            # readers can observe the old selection or the new complete one,
            # never a selection that points at a missing artifact.
            if self._staged_session_manifest is not None:
                if self._final_session_manifest is None:
                    raise StorageTransactionError(
                        "Runtime checkpoint session metadata target is unavailable."
                    )
                self._prepare_final_replacement(
                    self.session_root,
                    self._final_session_manifest,
                )
                self._transaction.publish(
                    self._staged_session_manifest,
                    self._final_session_manifest,
                )
        except Exception as exc:
            self._mark_failed(exc)
            raise


class RuntimeCheckpointRepository:
    """Own the stable ``models_root`` lock for one runtime checkpoint sequence."""

    def __init__(
        self,
        session_root: str | os.PathLike[str],
        *,
        lock_factory: StorageMutationLockFactory = StorageMutationLock,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the runtime checkpoint repository and its owned state."""
        self._session_root = Path(os.path.abspath(Path(session_root).expanduser()))
        self._lock_factory = lock_factory
        self._lock_timeout_seconds = max(0.0, float(lock_timeout_seconds))

    @classmethod
    def for_checkpoint_directory(
        cls,
        checkpoint_directory: str | os.PathLike[str],
        *,
        lock_factory: StorageMutationLockFactory = StorageMutationLock,
        lock_timeout_seconds: float = 5.0,
    ) -> RuntimeCheckpointRepository:
        """Build from the managed ``<session>/checkpoints`` directory."""

        checkpoint_root = Path(
            os.path.abspath(Path(checkpoint_directory).expanduser())
        )
        return cls(
            checkpoint_root.parent,
            lock_factory=lock_factory,
            lock_timeout_seconds=lock_timeout_seconds,
        )

    def _validated_scope(self) -> ModelsStorageMutationScope:
        """Return the validated scope value owned by this runtime checkpoint repository."""
        sessions_root = self._session_root.parent
        if sessions_root.name != TRAINING_SESSIONS_DIR_NAME:
            raise StoragePathViolation(
                "Runtime checkpoints must belong to managed training-session storage."
            )
        scope = ModelsStorageMutationScope.for_sessions_root(sessions_root)
        guarded_session = StoragePathGuard(sessions_root).require_descendant(
            self._session_root,
            must_exist=True,
        )
        if (
            guarded_session != self._session_root
            or self._session_root.parent != sessions_root
            or self._session_root.is_symlink()
            or not self._session_root.is_dir()
        ):
            raise StoragePathViolation(
                "Runtime checkpoint session storage is unavailable or aliased."
            )
        return scope

    @contextmanager
    def mutation(self) -> Iterator[RuntimeCheckpointMutation]:
        """Atomically commit or roll back one complete runtime checkpoint write."""

        scope = self._validated_scope()
        with scope.lock(
            lock_factory=self._lock_factory,
            timeout_seconds=self._lock_timeout_seconds,
        ):
            # Revalidate after lock acquisition because model/session deletion
            # uses the same root lock and may have won the preceding race.
            self._validated_scope()
            recovery = StorageTransaction.recover(self._session_root)
            if recovery.failed:
                failed = ", ".join(recovery.failed)
                raise StorageTransactionError(
                    "Runtime checkpoint recovery failed for transaction(s): "
                    f"{failed}."
                )
            with StorageTransaction(
                self._session_root,
                operation="runtime-checkpoint-save",
            ) as transaction:
                mutation = RuntimeCheckpointMutation(
                    self._session_root,
                    transaction,
                )
                yield mutation
                mutation._finalize()
                transaction.commit()
