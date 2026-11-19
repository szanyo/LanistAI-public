# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Transactional migration of legacy TensorBoard framework-root artifacts."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self
from uuid import uuid4

from lanista.reinforcengine.rl_model.storage.mutation_scope import (
    ModelsStorageMutationScope,
)
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.storage.transaction import StorageTransaction

_LOGGER = logging.getLogger(__name__)
_KNOWN_ROOT_ARTIFACTS = frozenset(
    {
        "params.json",
        "params.pkl",
        "progress.csv",
        "result.json",
    }
)


class _StorageMutationScope(Protocol):
    """Define the typed boundary for storage mutation scope."""

    def lock(self) -> AbstractContextManager[object]:
        """Return the lock value provided by this storage mutation scope."""
        ...


class _StorageTransaction(Protocol):
    """Define the typed boundary for storage transaction."""

    @property
    def committed(self) -> bool:
        """Return the committed exposed by this storage transaction."""
        ...

    @property
    def rolled_back(self) -> bool:
        """Return the rolled back exposed by this storage transaction."""
        ...

    def __enter__(self) -> Self:
        """Enter this storage transaction's managed context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Release owned resources when leaving this storage transaction's managed context."""
        ...

    def track_created(self, candidate: str | os.PathLike[str]) -> Path:
        """Return the track created value provided by this storage transaction."""
        ...

    def move(
        self,
        source: str | os.PathLike[str],
        target: str | os.PathLike[str],
    ) -> Path:
        """Move the operation owned by this storage transaction."""
        ...

    def quarantine(self, candidate: str | os.PathLike[str]) -> Path:
        """Return the quarantine value provided by this storage transaction."""
        ...

    def commit(self) -> None:
        """Commit the operation owned by this storage transaction."""
        ...


class _StorageTransactionFactory(Protocol):
    """Define the typed boundary for storage transaction factory."""

    def __call__(
        self,
        root: str,
        *,
        operation: str,
    ) -> _StorageTransaction:
        """Invoke this storage transaction factory as a callable."""
        ...


StorageMutationScopeFactory = Callable[[str], _StorageMutationScope]


def _mutation_scope_factory(model_folder_path: str) -> _StorageMutationScope:
    """Return the mutation scope factory value owned by this tensorboard artifact normalizer
    module.
    """
    return ModelsStorageMutationScope.for_model_root(model_folder_path)


def _transaction_factory(
    root: str,
    *,
    operation: str,
) -> _StorageTransaction:
    """Return the transaction factory value owned by this tensorboard artifact normalizer module."""
    return StorageTransaction(root, operation=operation)


@dataclass(frozen=True, slots=True)
class TensorboardArtifactNormalizationRequest:
    """Confined source and target coordinates for one normalization batch."""

    model_folder_path: str
    framework_root: str
    run_dir: str
    ensure_run_directory: bool = False


@dataclass(frozen=True, slots=True)
class TensorboardArtifactNormalizationResult:
    """Observable outcome of an optional legacy-artifact migration."""

    operation_id: str
    moved_artifacts: tuple[str, ...] = ()
    discarded_duplicates: tuple[str, ...] = ()
    failed_artifacts: tuple[str, ...] = ()
    committed: bool = False
    rolled_back: bool = False
    recovery_required: bool = False
    coordinates_validated: bool = False
    degradation_message: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether the optional normalization completed cleanly."""

        return self.degradation_message is None


class TensorboardArtifactNormalizer:
    """Move legacy artifacts as one locked, journaled filesystem batch."""

    def __init__(
        self,
        *,
        mutation_scope_factory: StorageMutationScopeFactory = (_mutation_scope_factory),
        transaction_factory: _StorageTransactionFactory = _transaction_factory,
    ) -> None:
        """Initialize the tensorboard artifact normalizer and its owned state."""
        self._mutation_scope_factory = mutation_scope_factory
        self._transaction_factory = transaction_factory

    def normalize(
        self,
        request: TensorboardArtifactNormalizationRequest,
    ) -> TensorboardArtifactNormalizationResult:
        """Normalize known files, retaining all sources on pre-commit failure."""

        operation_id = uuid4().hex
        artifact_names: tuple[str, ...] = ()
        try:
            guard, framework_root, run_dir = self._validated_paths(request)
            if not os.path.lexists(framework_root) and not request.ensure_run_directory:
                return TensorboardArtifactNormalizationResult(
                    operation_id,
                    coordinates_validated=True,
                )
            if os.path.lexists(framework_root) and not framework_root.is_dir():
                raise NotADirectoryError(framework_root)
            artifact_names = self._artifact_names(framework_root) if framework_root.is_dir() else ()
            if not artifact_names and run_dir.is_dir():
                return TensorboardArtifactNormalizationResult(
                    operation_id,
                    coordinates_validated=True,
                )
        except Exception as error:
            return self._degraded_result(
                operation_id=operation_id,
                artifact_names=artifact_names,
                moved=[],
                discarded=[],
                transaction=None,
                error=error,
            )

        return self._normalize_locked(
            operation_id=operation_id,
            guard=guard,
            mutation_root=request.model_folder_path,
            framework_root=framework_root,
            run_dir=run_dir,
            artifact_names=artifact_names,
        )

    def _normalize_locked(
        self,
        *,
        operation_id: str,
        guard: StoragePathGuard,
        mutation_root: str,
        framework_root: Path,
        run_dir: Path,
        artifact_names: tuple[str, ...],
    ) -> TensorboardArtifactNormalizationResult:
        """Normalize the locked accepted by this tensorboard artifact normalizer."""
        moved: list[str] = []
        discarded: list[str] = []
        transaction: _StorageTransaction | None = None
        transaction_attempted = False
        try:
            scope = self._mutation_scope_factory(mutation_root)
            with scope.lock():
                if os.path.lexists(framework_root) and not framework_root.is_dir():
                    raise NotADirectoryError(framework_root)
                artifact_names = (
                    self._artifact_names(framework_root) if framework_root.is_dir() else ()
                )
                if not artifact_names and run_dir.is_dir():
                    return TensorboardArtifactNormalizationResult(
                        operation_id,
                        coordinates_validated=True,
                    )
                transaction_attempted = True
                transaction = self._transaction_factory(
                    str(guard.root),
                    operation="normalize-tensorboard-artifacts",
                )
                with transaction:
                    self._prepare_directory_chain(
                        transaction,
                        guard,
                        framework_root,
                    )
                    self._prepare_run_directory(transaction, guard, run_dir)
                    self._apply_artifacts(
                        transaction,
                        guard,
                        framework_root,
                        run_dir,
                        artifact_names,
                        moved,
                        discarded,
                    )
                    transaction.commit()
        except Exception as error:
            return self._degraded_result(
                operation_id=operation_id,
                artifact_names=artifact_names,
                moved=moved,
                discarded=discarded,
                transaction=transaction,
                transaction_attempted=transaction_attempted,
                coordinates_validated=True,
                error=error,
            )

        return TensorboardArtifactNormalizationResult(
            operation_id=operation_id,
            moved_artifacts=tuple(moved),
            discarded_duplicates=tuple(discarded),
            committed=True,
            coordinates_validated=True,
        )

    @classmethod
    def _validated_paths(
        cls,
        request: TensorboardArtifactNormalizationRequest,
    ) -> tuple[StoragePathGuard, Path, Path]:
        """Return the validated paths value owned by this tensorboard artifact normalizer."""
        if not str(request.model_folder_path).strip():
            raise StoragePathViolation("The model storage root is required.")
        guard = StoragePathGuard(request.model_folder_path)
        declared_root = Path(request.model_folder_path).expanduser().absolute()
        if declared_root != guard.root or declared_root.is_symlink():
            raise StoragePathViolation("Model storage aliases are forbidden during normalization.")
        framework_root = cls._validated_request_path(
            guard,
            declared_root,
            request.framework_root,
        )
        run_dir = cls._validated_request_path(
            guard,
            declared_root,
            request.run_dir,
        )
        if run_dir.parent != framework_root:
            raise StoragePathViolation(
                "The TensorBoard run directory must be a direct framework child."
            )
        return guard, framework_root, run_dir

    @staticmethod
    def _validated_request_path(
        guard: StoragePathGuard,
        declared_root: Path,
        candidate: str,
    ) -> Path:
        """Return the validated request path value owned by this tensorboard artifact normalizer."""
        raw_text = os.fspath(candidate)
        raw_candidate = Path(raw_text).expanduser()
        if "\\" in raw_text or ".." in raw_candidate.parts:
            raise StoragePathViolation("Relative path markers and backslashes are forbidden.")
        lexical_path = (
            raw_candidate if raw_candidate.is_absolute() else declared_root / raw_candidate
        ).absolute()
        try:
            relative_path = lexical_path.relative_to(declared_root)
        except ValueError as exc:
            raise StoragePathViolation(
                "TensorBoard coordinates must remain below the model root."
            ) from exc
        canonical_lexical_path = guard.root.joinpath(relative_path)
        validated_path = guard.require_descendant(canonical_lexical_path)
        if canonical_lexical_path != validated_path:
            raise StoragePathViolation("Storage aliases are forbidden in TensorBoard coordinates.")
        return validated_path

    @staticmethod
    def _prepare_directory_chain(
        transaction: _StorageTransaction,
        guard: StoragePathGuard,
        target: Path,
    ) -> None:
        """Prepare the directory chain required by this tensorboard artifact normalizer."""
        missing_paths: list[Path] = []
        cursor = target
        while not os.path.lexists(cursor):
            TensorboardArtifactNormalizer._require_unaliased(guard, cursor)
            missing_paths.append(cursor)
            cursor = cursor.parent
        if cursor != guard.root:
            TensorboardArtifactNormalizer._require_unaliased(
                guard,
                cursor,
                must_exist=True,
            )
        if not cursor.is_dir():
            raise NotADirectoryError(cursor)
        for missing_path in reversed(missing_paths):
            transaction.track_created(missing_path)
            missing_path.mkdir()

    @staticmethod
    def _prepare_run_directory(
        transaction: _StorageTransaction,
        guard: StoragePathGuard,
        run_dir: Path,
    ) -> None:
        """Prepare the run directory required by this tensorboard artifact normalizer."""
        if not os.path.lexists(run_dir):
            TensorboardArtifactNormalizer._require_unaliased(
                guard,
                run_dir,
            )
            transaction.track_created(run_dir)
            run_dir.mkdir()
            return
        TensorboardArtifactNormalizer._require_unaliased(
            guard,
            run_dir,
            must_exist=True,
        )
        if not run_dir.is_dir():
            raise NotADirectoryError(run_dir)

    @staticmethod
    def _apply_artifacts(
        transaction: _StorageTransaction,
        guard: StoragePathGuard,
        framework_root: Path,
        run_dir: Path,
        artifact_names: tuple[str, ...],
        moved: list[str],
        discarded: list[str],
    ) -> None:
        """Apply the artifacts transition owned by this tensorboard artifact normalizer."""
        for entry_name in artifact_names:
            source = TensorboardArtifactNormalizer._require_unaliased(
                guard,
                framework_root / entry_name,
                must_exist=True,
            )
            if not source.is_file():
                raise OSError("A legacy TensorBoard artifact is not a regular file.")
            target = TensorboardArtifactNormalizer._require_unaliased(
                guard,
                run_dir / entry_name,
                must_exist=os.path.lexists(run_dir / entry_name),
            )
            if os.path.lexists(target):
                if not target.is_file():
                    raise OSError("A TensorBoard artifact collision is not a regular file.")
                transaction.quarantine(source)
                discarded.append(entry_name)
            else:
                transaction.move(source, target)
                moved.append(entry_name)

    @staticmethod
    def _require_unaliased(
        guard: StoragePathGuard,
        candidate: Path,
        *,
        must_exist: bool = False,
    ) -> Path:
        """Return the required unaliased for this tensorboard artifact normalizer."""
        lexical_path = candidate.absolute()
        validated_path = guard.require_descendant(
            lexical_path,
            must_exist=must_exist,
        )
        if lexical_path != validated_path or lexical_path.is_symlink():
            raise StoragePathViolation(
                "Storage aliases are forbidden during artifact normalization."
            )
        return validated_path

    @staticmethod
    def _artifact_names(framework_root: Path) -> tuple[str, ...]:
        """Return the artifact names value owned by this tensorboard artifact normalizer."""
        return tuple(
            entry.name
            for entry in sorted(framework_root.iterdir(), key=lambda path: path.name)
            if (entry.is_file() or entry.is_symlink())
            and (
                entry.name in _KNOWN_ROOT_ARTIFACTS or entry.name.startswith("events.out.tfevents.")
            )
        )

    @staticmethod
    def _degraded_result(
        *,
        operation_id: str,
        artifact_names: tuple[str, ...],
        moved: list[str],
        discarded: list[str],
        transaction: _StorageTransaction | None,
        transaction_attempted: bool = False,
        coordinates_validated: bool = False,
        error: Exception,
    ) -> TensorboardArtifactNormalizationResult:
        """Return the degraded result value owned by this tensorboard artifact normalizer."""
        committed = transaction is not None and transaction.committed
        rolled_back = transaction is not None and transaction.rolled_back
        recovery_required = transaction_attempted and not (committed or rolled_back)
        message = TensorboardArtifactNormalizer._degradation_message(
            committed=committed,
            rolled_back=rolled_back,
            recovery_required=recovery_required,
        )
        _LOGGER.warning(
            "tensorboard_artifact_normalization_degraded "
            f"operation_id={operation_id} committed={committed} "
            f"rolled_back={rolled_back} recovery_required={recovery_required} "
            f"exception_type={type(error).__name__}",
            exc_info=(type(error), error, error.__traceback__),
        )
        return TensorboardArtifactNormalizationResult(
            operation_id=operation_id,
            moved_artifacts=tuple(moved) if committed else (),
            discarded_duplicates=tuple(discarded) if committed else (),
            failed_artifacts=() if committed else artifact_names,
            committed=committed,
            rolled_back=rolled_back,
            recovery_required=recovery_required,
            coordinates_validated=coordinates_validated,
            degradation_message=message,
        )

    @staticmethod
    def _degradation_message(
        *,
        committed: bool,
        rolled_back: bool,
        recovery_required: bool,
    ) -> str:
        """Return the degradation message value owned by this tensorboard artifact normalizer."""
        if committed:
            return "TensorBoard legacy artifacts were committed, but transaction cleanup degraded."
        if rolled_back:
            return "TensorBoard legacy artifact normalization was rolled back."
        if recovery_required:
            return "TensorBoard legacy artifact normalization requires storage recovery."
        return "TensorBoard legacy artifact normalization did not start."
