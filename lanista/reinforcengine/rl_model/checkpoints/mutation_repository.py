# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Locked and recoverable filesystem adapter for checkpoint mutations."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Never, Protocol

from lanista.reinforcengine.application_errors import (
    ApplicationError,
    ApplicationValidationError,
    OperationConflict,
    OperationTimeout,
    PersistenceError,
    ResourceNotFoundError,
)
from lanista.reinforcengine.rl_model.checkpoints.metadata import (
    load_checkpoint_manifest,
    write_checkpoint_manifest,
)
from lanista.reinforcengine.rl_model.checkpoints.mutation_contracts import (
    CheckpointMutationCommit,
)
from lanista.reinforcengine.rl_model.checkpoints.name_metadata import (
    extract_checkpoint_steps,
)
from lanista.reinforcengine.rl_model.checkpoints.storage_resolver import (
    CheckpointStoragePaths,
    CheckpointStorageResolver,
)
from lanista.reinforcengine.rl_model.storage.layout import CHECKPOINT_MANIFEST_FILE_NAME
from lanista.reinforcengine.rl_model.storage.mutation_lock import (
    StorageMutationLock,
    StorageMutationLockTimeout,
)
from lanista.reinforcengine.rl_model.storage.mutation_scope import (
    ModelsStorageMutationScope,
    StorageMutationLockFactory,
)
from lanista.reinforcengine.rl_model.storage.names import (
    CheckpointStorageName,
    InvalidStorageName,
)
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.storage.transaction import StorageTransaction


class CheckpointTransactionFactory(Protocol):
    """Construct one checkpoint-root storage transaction."""

    def __call__(
        self,
        root: str | os.PathLike[str],
        *,
        operation: str,
    ) -> StorageTransaction:
        """Invoke this checkpoint transaction factory as a callable."""
        ...


class CheckpointManifestLoader(Protocol):
    """Read optional metadata used to prepare annotation updates."""

    def __call__(
        self,
        checkpoint_path: str | os.PathLike[str],
    ) -> Mapping[str, object]:
        """Invoke this checkpoint manifest loader as a callable."""
        ...


class CheckpointManifestWriter(Protocol):
    """Persist one canonical checkpoint manifest."""

    def __call__(
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
        """Invoke this checkpoint manifest writer as a callable."""
        ...


class _WarningLogger(Protocol):
    """Define the typed boundary for warning logger."""

    def warning(self, message: str) -> object:
        """Record a non-fatal storage diagnostic."""


@dataclass(frozen=True, slots=True)
class _CheckpointAnnotationTarget:
    """Carry the data required for checkpoint annotation target."""

    checkpoints_root: Path
    checkpoint_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class _CheckpointAnnotationWrite:
    """Carry the data required for checkpoint annotation write."""

    target: _CheckpointAnnotationTarget
    checkpoint_name: str
    checkpoint_kind: str
    framework_uuid: str
    scenario_index: int | None
    curriculum_state: object
    step_count: int | None
    parent_checkpoint_name: str | None
    saved_at: str | None
    annotation: str | None


class CheckpointMutationRepository:
    """Serialize checkpoint delete, rename, and annotation transactions."""

    def __init__(
        self,
        *,
        sessions_root_provider: Callable[[], str],
        session_framework_provider: Callable[[str], str],
        current_framework_provider: Callable[[], str | None],
        models_root_provider: Callable[[], str] | None = None,
        step_extractor: Callable[[str], int] = extract_checkpoint_steps,
        logger: _WarningLogger | None = None,
        lock_factory: StorageMutationLockFactory = StorageMutationLock,
        lock_timeout_seconds: float = 5.0,
        transaction_factory: CheckpointTransactionFactory = StorageTransaction,
        manifest_loader: CheckpointManifestLoader = load_checkpoint_manifest,
        manifest_writer: CheckpointManifestWriter = write_checkpoint_manifest,
        resolver: CheckpointStorageResolver | None = None,
    ) -> None:
        """Initialize the checkpoint mutation repository and its owned state."""
        self._sessions_root_provider = sessions_root_provider
        self._session_framework_provider = session_framework_provider
        self._current_framework_provider = current_framework_provider
        self._models_root_provider = models_root_provider
        self._step_extractor = step_extractor
        self._logger = logger
        self._lock_factory = lock_factory
        self._lock_timeout_seconds = lock_timeout_seconds
        self._transaction_factory = transaction_factory
        self._manifest_loader = manifest_loader
        self._manifest_writer = manifest_writer
        self._resolver = resolver or CheckpointStorageResolver()

    def delete(
        self,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> CheckpointMutationCommit[str]:
        """Delete one checkpoint and its sidecar as a recoverable transaction."""

        if not session_name or not checkpoint_name:
            raise ApplicationValidationError(
                "Checkpoint deletion requires a session and checkpoint."
            )
        return self._execute_mutation(
            checkpoint_name,
            lambda: self._delete_locked(session_name, checkpoint_name),
        )

    def rename(
        self,
        session_name: str | None,
        checkpoint_name: str | None,
        renamed_checkpoint_name: str | None,
    ) -> CheckpointMutationCommit[str]:
        """Rename one direct or organized checkpoint without losing auxiliaries."""

        if not session_name or not checkpoint_name:
            raise ApplicationValidationError(
                "Checkpoint rename requires a session and checkpoint."
            )
        return self._execute_mutation(
            checkpoint_name,
            lambda: self._rename_locked(
                session_name,
                checkpoint_name,
                renamed_checkpoint_name,
            ),
        )

    def update_annotation(
        self,
        session_name: str | None,
        checkpoint_name: str | None,
        annotation: str | None,
    ) -> CheckpointMutationCommit[str | None]:
        """Atomically replace one checkpoint's normalized annotation metadata."""

        if not session_name or not checkpoint_name:
            raise ApplicationValidationError(
                "Checkpoint annotation requires a session and checkpoint."
            )
        return self._execute_mutation(
            checkpoint_name,
            lambda: self._update_annotation_locked(
                session_name,
                checkpoint_name,
                annotation,
            ),
        )

    def _mutation_scope(self) -> ModelsStorageMutationScope:
        """Return the mutation scope value owned by this checkpoint mutation repository."""
        explicit_models_root = (
            self._models_root_provider()
            if self._models_root_provider is not None
            else None
        )
        return ModelsStorageMutationScope.for_sessions_root(
            self._sessions_root_provider(),
            models_root=explicit_models_root,
        )

    def _execute_mutation[ResultT](
        self,
        checkpoint_name: str,
        mutation: Callable[[], CheckpointMutationCommit[ResultT]],
    ) -> CheckpointMutationCommit[ResultT]:
        """Execute the mutation owned by this checkpoint mutation repository."""
        committed: CheckpointMutationCommit[ResultT] | None = None
        try:
            with self._mutation_scope().lock(
                lock_factory=self._lock_factory,
                timeout_seconds=self._lock_timeout_seconds,
            ):
                committed = mutation()
                return committed
        except Exception as error:
            if committed is not None:
                return CheckpointMutationCommit(
                    value=committed.value,
                    post_commit_error=self._combine_post_commit_errors(
                        committed.post_commit_error,
                        error,
                    ),
                )
            self._raise_mutation_failure(checkpoint_name, error)

    def _raise_mutation_failure(
        self,
        checkpoint_name: str,
        error: Exception,
    ) -> Never:
        """Return the raise mutation failure value provided by this checkpoint mutation repository.
        """
        if isinstance(error, ApplicationError):
            raise error
        if isinstance(error, StorageMutationLockTimeout):
            self._warn_mutation_failure(checkpoint_name, error, action="lock")
            raise OperationTimeout(
                "Checkpoint storage is busy. Try again."
            ) from error
        if isinstance(error, FileNotFoundError):
            self._warn_mutation_failure(checkpoint_name, error, action="locate")
            raise ResourceNotFoundError(
                "Checkpoint storage is unavailable."
            ) from error
        if isinstance(error, InvalidStorageName | StoragePathViolation):
            self._warn_mutation_failure(
                checkpoint_name,
                error,
                action="validate",
            )
            raise ApplicationValidationError(
                "Session and checkpoint names must stay within managed storage."
            ) from error
        self._warn_mutation_failure(checkpoint_name, error, action="lock")
        raise PersistenceError(
            "Checkpoint storage could not be locked."
        ) from error

    def _warn_mutation_failure(
        self,
        checkpoint_name: str,
        error: Exception,
        *,
        action: str,
    ) -> None:
        """Report the mutation failure owned by this checkpoint mutation
repository."""
        if self._logger is not None:
            self._logger.warning(
                f"Failed to {action} checkpoint storage for "
                f"'{checkpoint_name}': {error}"
            )

    @staticmethod
    def _combine_post_commit_errors(
        existing: Exception | None,
        additional: Exception,
    ) -> Exception:
        """Return the combine post commit errors value owned by this checkpoint mutation
        repository.
        """
        if existing is None:
            return additional
        if existing is additional:
            return existing
        return ExceptionGroup(
            "Multiple failures followed a durable checkpoint commit.",
            (existing, additional),
        )

    @staticmethod
    def _commit_storage_mutation[ResultT](
        transaction: StorageTransaction,
        *,
        value: ResultT,
        mutation: Callable[[StorageTransaction], None],
    ) -> CheckpointMutationCommit[ResultT]:
        """Commit the storage mutation owned by this checkpoint mutation repository."""
        try:
            with transaction:
                mutation(transaction)
                transaction.commit()
        except Exception as error:
            if not transaction.committed:
                raise
            return CheckpointMutationCommit(
                value=value,
                post_commit_error=error,
            )
        return CheckpointMutationCommit(value=value)

    def _mutation_checkpoints_root(self, session_name: str) -> Path:
        """Return the mutation checkpoints root value owned by this checkpoint mutation repository.
        """
        try:
            return self._resolver.checkpoints_root(
                self._sessions_root_provider(),
                session_name,
            )
        except (InvalidStorageName, StoragePathViolation) as error:
            raise ApplicationValidationError(
                "Session and checkpoint names must stay within managed storage."
            ) from error
        except OSError as error:
            raise PersistenceError(
                "Checkpoint storage could not be resolved."
            ) from error

    def _delete_locked(
        self,
        session_name: str,
        checkpoint_name: str,
    ) -> CheckpointMutationCommit[str]:
        """Delete the locked through this checkpoint mutation repository's safety boundary."""
        normalized_name = self._parse_checkpoint_name(
            checkpoint_name,
            message="Checkpoint name must be one safe, portable path component.",
        )
        checkpoints_root = self._mutation_checkpoints_root(session_name)
        if not checkpoints_root.is_dir():
            raise ResourceNotFoundError("Checkpoint storage is unavailable.")
        return self._delete_checkpoint_storage(checkpoints_root, normalized_name)

    def _delete_checkpoint_storage(
        self,
        checkpoints_root: Path,
        normalized_name: CheckpointStorageName,
    ) -> CheckpointMutationCommit[str]:
        """Delete the checkpoint storage through this checkpoint mutation repository's safety
        boundary.
        """
        try:
            guard, paths = self._resolver.storage_paths(
                checkpoints_root,
                normalized_name.value,
            )
            checkpoint_path = self._resolver.resolve_candidate_from_root(
                checkpoints_root,
                normalized_name.value,
            )
            if not os.path.lexists(checkpoint_path):
                raise ResourceNotFoundError(
                    "Selected checkpoint no longer exists."
                )
            guarded_path = guard.require_descendant(
                checkpoint_path,
                must_exist=True,
            )
            sidecar = self._resolver.manifest_sidecar(guard, guarded_path)
            transaction = self._transaction_factory(
                checkpoints_root,
                operation="checkpoint-delete",
            )
            return self._commit_storage_mutation(
                transaction,
                value=normalized_name.value,
                mutation=lambda active: self._quarantine_checkpoint_paths(
                    active,
                    paths=paths,
                    checkpoint_path=guarded_path,
                    sidecar_path=sidecar,
                ),
            )
        except ResourceNotFoundError:
            raise
        except (InvalidStorageName, StoragePathViolation) as error:
            self._warning(
                "Rejected unsafe checkpoint deletion for "
                f"'{normalized_name.value}': {error}"
            )
            raise ApplicationValidationError(
                "Checkpoint path must stay within managed storage."
            ) from error
        except FileNotFoundError as error:
            self._warning(
                "Checkpoint disappeared during deletion for "
                f"'{normalized_name.value}': {error}"
            )
            raise ResourceNotFoundError(
                "Selected checkpoint no longer exists."
            ) from error
        except Exception as error:
            self._warning(
                "Failed to delete checkpoint storage for "
                f"'{normalized_name.value}': {error}"
            )
            raise PersistenceError(
                "Checkpoint deletion could not be committed."
            ) from error

    @staticmethod
    def _quarantine_checkpoint_paths(
        transaction: StorageTransaction,
        *,
        paths: CheckpointStoragePaths,
        checkpoint_path: Path,
        sidecar_path: Path,
    ) -> None:
        """Quarantine the checkpoint paths owned by this checkpoint mutation
repository."""
        if paths.organized_root.is_dir():
            transaction.quarantine(paths.organized_root)
        if os.path.lexists(checkpoint_path):
            transaction.quarantine(checkpoint_path)
        if sidecar_path.is_file():
            transaction.quarantine(sidecar_path)

    def _rename_locked(
        self,
        session_name: str,
        checkpoint_name: str,
        renamed_checkpoint_name: str | None,
    ) -> CheckpointMutationCommit[str]:
        """Rename the locked owned by this checkpoint mutation repository."""
        source_name, candidate_name = self._normalize_rename_names(
            checkpoint_name,
            renamed_checkpoint_name,
        )
        if candidate_name == source_name:
            raise OperationConflict("Checkpoint name is unchanged.")
        checkpoints_root = self._mutation_checkpoints_root(session_name)
        if not checkpoints_root.is_dir():
            raise ResourceNotFoundError("Checkpoint storage is unavailable.")
        return self._rename_checkpoint_storage(
            checkpoints_root,
            source_name,
            candidate_name,
        )

    def _normalize_rename_names(
        self,
        checkpoint_name: str,
        renamed_checkpoint_name: str | None,
    ) -> tuple[CheckpointStorageName, CheckpointStorageName]:
        """Normalize the rename names accepted by this checkpoint mutation repository."""
        candidate_raw = str(renamed_checkpoint_name or "").strip()
        if not candidate_raw:
            raise ApplicationValidationError("Enter a valid checkpoint name.")
        if (
            checkpoint_name.lower().endswith(".zip")
            and not candidate_raw.lower().endswith(".zip")
        ):
            candidate_raw = f"{candidate_raw}.zip"
        return (
            self._parse_checkpoint_name(
                checkpoint_name,
                message=(
                    "Checkpoint name must be one safe, portable path "
                    "component."
                ),
            ),
            self._parse_checkpoint_name(
                candidate_raw,
                message=(
                    "Checkpoint name must be one safe, portable path "
                    "component."
                ),
            ),
        )

    def _rename_checkpoint_storage(
        self,
        checkpoints_root: Path,
        source_name: CheckpointStorageName,
        candidate_name: CheckpointStorageName,
    ) -> CheckpointMutationCommit[str]:
        """Rename the checkpoint storage owned by this checkpoint mutation repository."""
        try:
            source_path = self._resolver.resolve_candidate_from_root(
                checkpoints_root,
                source_name.value,
            )
            if not os.path.lexists(source_path):
                raise ResourceNotFoundError(
                    "Selected checkpoint no longer exists."
                )
            guard, source_paths = self._resolver.storage_paths(
                checkpoints_root,
                source_name.value,
            )
            _, target_paths = self._resolver.storage_paths(
                checkpoints_root,
                candidate_name.value,
            )
            guarded_source = guard.require_descendant(
                source_path,
                must_exist=True,
            )
            if any(
                os.path.lexists(path)
                for path in (
                    target_paths.direct,
                    target_paths.organized_flat,
                    target_paths.organized_root,
                )
            ):
                raise OperationConflict(
                    f"Checkpoint '{candidate_name.value}' already exists."
                )
            transaction = self._transaction_factory(
                checkpoints_root,
                operation="checkpoint-rename",
            )
            return self._commit_storage_mutation(
                transaction,
                value=candidate_name.value,
                mutation=lambda active: self._rename_checkpoint_paths(
                    active,
                    guard=guard,
                    source_paths=source_paths,
                    source_checkpoint_path=guarded_source,
                    target_paths=target_paths,
                    candidate_name=candidate_name,
                ),
            )
        except ApplicationError:
            raise
        except (InvalidStorageName, StoragePathViolation) as error:
            self._warning(
                f"Rejected unsafe checkpoint rename '{source_name.value}' -> "
                f"'{candidate_name.value}': {error}"
            )
            raise ApplicationValidationError(
                "Checkpoint path must stay within managed storage."
            ) from error
        except FileNotFoundError as error:
            self._warning(
                f"Checkpoint disappeared during rename '{source_name.value}' "
                f"-> '{candidate_name.value}': {error}"
            )
            raise ResourceNotFoundError(
                "Selected checkpoint no longer exists."
            ) from error
        except Exception as error:
            self._warning(
                f"Failed to rename checkpoint '{source_name.value}' -> "
                f"'{candidate_name.value}': {error}"
            )
            raise PersistenceError(
                "Checkpoint rename could not be committed."
            ) from error

    def _rename_checkpoint_paths(
        self,
        transaction: StorageTransaction,
        *,
        guard: StoragePathGuard,
        source_paths: CheckpointStoragePaths,
        source_checkpoint_path: Path,
        target_paths: CheckpointStoragePaths,
        candidate_name: CheckpointStorageName,
    ) -> None:
        """Rename the checkpoint paths owned by this checkpoint mutation repository."""
        if source_checkpoint_path.is_relative_to(source_paths.organized_root):
            self._rename_organized_checkpoint(
                transaction,
                guard,
                source_paths.organized_root,
                source_checkpoint_path,
                target_paths.organized_root,
                candidate_name,
            )
        elif source_checkpoint_path == source_paths.organized_flat:
            self._rename_direct_checkpoint(
                transaction,
                guard,
                source_checkpoint_path,
                target_paths.organized_flat,
            )
        else:
            self._rename_direct_checkpoint(
                transaction,
                guard,
                source_checkpoint_path,
                target_paths.direct,
            )

    def _rename_organized_checkpoint(
        self,
        transaction: StorageTransaction,
        guard: StoragePathGuard,
        source_root: Path,
        source_file: Path,
        target_root: Path,
        target_name: CheckpointStorageName,
    ) -> None:
        """Rename the organized checkpoint owned by this checkpoint mutation repository."""
        transaction.move(source_root, target_root)
        moved_source = guard.require_descendant(
            target_root / source_file.name,
            must_exist=True,
        )
        target_file = guard.require_descendant(target_root / target_name.value)
        source_sidecar = self._resolver.manifest_sidecar(guard, moved_source)
        target_sidecar = self._resolver.manifest_sidecar(guard, target_file)
        if moved_source != target_file:
            transaction.move(moved_source, target_file)
        if source_sidecar.is_file():
            transaction.move(source_sidecar, target_sidecar)

    def _rename_direct_checkpoint(
        self,
        transaction: StorageTransaction,
        guard: StoragePathGuard,
        source_file: Path,
        target_file: Path,
    ) -> None:
        """Rename the direct checkpoint owned by this checkpoint mutation repository."""
        source_sidecar = self._resolver.manifest_sidecar(guard, source_file)
        target_sidecar = self._resolver.manifest_sidecar(guard, target_file)
        transaction.move(source_file, target_file)
        if source_sidecar.is_file():
            transaction.move(source_sidecar, target_sidecar)

    def _update_annotation_locked(
        self,
        session_name: str,
        checkpoint_name: str,
        annotation: str | None,
    ) -> CheckpointMutationCommit[str | None]:
        """Update the annotation locked owned by this checkpoint mutation repository."""
        target = self._resolve_annotation_target(
            session_name,
            checkpoint_name,
        )
        prepared = self._prepare_annotation_write(
            session_name,
            checkpoint_name,
            annotation,
            target,
            self._manifest_loader(target.checkpoint_path),
        )
        transaction = self._transaction_factory(
            target.checkpoints_root,
            operation="checkpoint-annotation",
        )
        try:
            return self._commit_storage_mutation(
                transaction,
                value=prepared.annotation,
                mutation=lambda active: self._write_annotation_manifest(
                    active,
                    prepared,
                ),
            )
        except Exception as error:
            self._warning(
                "Failed to update checkpoint annotation for "
                f"'{checkpoint_name}': {error}"
            )
            raise PersistenceError(
                "Checkpoint annotation could not be committed."
            ) from error

    def _resolve_annotation_target(
        self,
        session_name: str,
        checkpoint_name: str,
    ) -> _CheckpointAnnotationTarget:
        """Resolve the annotation target required by this checkpoint mutation repository."""
        normalized_name = self._parse_checkpoint_name(
            checkpoint_name,
            message="Checkpoint name must be one safe, portable path component.",
        )
        checkpoints_root = self._mutation_checkpoints_root(session_name)
        if not checkpoints_root.is_dir():
            raise ResourceNotFoundError("Checkpoint storage is unavailable.")
        try:
            guard, _paths = self._resolver.storage_paths(
                checkpoints_root,
                normalized_name.value,
            )
            checkpoint_path = self._resolver.resolve_candidate_from_root(
                checkpoints_root,
                normalized_name.value,
            )
        except (InvalidStorageName, StoragePathViolation) as error:
            raise ApplicationValidationError(
                "Checkpoint path must stay within managed storage."
            ) from error
        if not checkpoint_path.exists():
            raise ResourceNotFoundError(
                "Selected checkpoint no longer exists."
            )
        manifest_path = (
            guard.require_descendant(
                checkpoint_path / CHECKPOINT_MANIFEST_FILE_NAME
            )
            if checkpoint_path.is_dir()
            else self._resolver.manifest_sidecar(guard, checkpoint_path)
        )
        return _CheckpointAnnotationTarget(
            checkpoints_root=checkpoints_root,
            checkpoint_path=checkpoint_path,
            manifest_path=manifest_path,
        )

    def _prepare_annotation_write(
        self,
        session_name: str,
        checkpoint_name: str,
        annotation: str | None,
        target: _CheckpointAnnotationTarget,
        manifest: Mapping[str, object],
    ) -> _CheckpointAnnotationWrite:
        """Prepare the annotation write required by this checkpoint mutation repository."""
        checkpoint_kind = (
            str(
                manifest.get("kind")
                or ("file" if target.checkpoint_path.is_file() else "directory")
            ).strip()
            or "checkpoint"
        )
        framework_uuid = (
            str(
                manifest.get("framework_uuid")
                or self._session_framework_provider(session_name)
                or self._current_framework_provider()
                or "-"
            ).strip()
            or "-"
        )
        step_count = self._optional_int(manifest.get("step_count"))
        if step_count is None:
            step_count = self._step_extractor(checkpoint_name)
        return _CheckpointAnnotationWrite(
            target=target,
            checkpoint_name=str(manifest.get("name") or checkpoint_name),
            checkpoint_kind=checkpoint_kind,
            framework_uuid=framework_uuid,
            scenario_index=self._optional_int(manifest.get("scenario_index")),
            curriculum_state=manifest.get("curriculum_state"),
            step_count=step_count,
            parent_checkpoint_name=(
                str(manifest.get("parent_checkpoint_name") or "").strip()
                or None
            ),
            saved_at=(
                str(manifest.get("saved_at") or "").strip() or None
            ),
            annotation=str(annotation or "").strip() or None,
        )

    def _write_annotation_manifest(
        self,
        transaction: StorageTransaction,
        prepared: _CheckpointAnnotationWrite,
    ) -> None:
        """Persist the annotation manifest owned by this checkpoint mutation repository."""
        target = prepared.target
        if target.manifest_path.is_file():
            transaction.snapshot_file(target.manifest_path)
        else:
            transaction.track_created(target.manifest_path)
        self._manifest_writer(
            str(target.checkpoint_path),
            checkpoint_name=prepared.checkpoint_name,
            checkpoint_kind=prepared.checkpoint_kind,
            framework_uuid=prepared.framework_uuid,
            scenario_index=prepared.scenario_index,
            curriculum_state=prepared.curriculum_state,
            step_count=prepared.step_count,
            parent_checkpoint_name=prepared.parent_checkpoint_name,
            saved_at=prepared.saved_at,
            annotation=prepared.annotation,
        )
        transaction.confirm_path(target.manifest_path)

    @staticmethod
    def _optional_int(value: object) -> int | None:
        """Return the optional int value owned by this checkpoint mutation repository."""
        if not isinstance(value, str | int | float):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_checkpoint_name(
        checkpoint_name: str,
        *,
        message: str,
    ) -> CheckpointStorageName:
        """Parse the checkpoint name owned by this checkpoint mutation repository."""
        try:
            return CheckpointStorageName.parse(checkpoint_name)
        except InvalidStorageName as error:
            raise ApplicationValidationError(message) from error

    def _warning(self, message: str) -> None:
        """Publish a warning record through this checkpoint mutation repository."""
        if self._logger is not None:
            self._logger.warning(message)
