# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Administration-aware commands for TensorBoard comparison packs."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Never

from lanista.reinforcengine.application_errors import (
    ApplicationError,
    ApplicationValidationError,
    OperationConflict,
    OperationTimeout,
    PersistenceError,
    ResourceNotFoundError,
    RuntimeUnavailableError,
)
from lanista.reinforcengine.rl_model.administration_safety_service import (
    AdministrationCompletionGuard,
    AdministrationSafetyService,
)
from lanista.reinforcengine.rl_model.comparison.contracts import (
    ComparisonPackCreationOutcome,
    ComparisonPackMutationOutcome,
    ComparisonPackTransactionFactory,
)
from lanista.reinforcengine.rl_model.comparison.materializer import (
    ComparisonPackMaterializer,
)
from lanista.reinforcengine.rl_model.comparison.repository import (
    ComparisonPackRepository,
)
from lanista.reinforcengine.rl_model.comparison.tensorboard_entry import (
    TensorboardPackEntry,
)
from lanista.reinforcengine.rl_model.name_formatting_service import NameSanitizer
from lanista.reinforcengine.rl_model.storage.mutation_lock import (
    StorageMutationLock,
    StorageMutationLockError,
    StorageMutationLockTimeout,
)
from lanista.reinforcengine.rl_model.storage.mutation_scope import (
    ModelsStorageMutationScope,
    StorageMutationLockFactory,
)
from lanista.reinforcengine.rl_model.storage.names import (
    ComparisonPackStorageName,
    InvalidStorageName,
)
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.storage.transaction import (
    StorageTransaction,
    StorageTransactionError,
)


@dataclass(frozen=True, slots=True)
class _ComparisonPackRenamePlan:
    """Describe the executable plan for comparison pack rename."""

    current_name: str
    candidate_name: str
    source_path: Path
    target_path: Path
    annotation: str | None
    source_paths: tuple[str, ...]
    had_manifest: bool


class ComparisonPackCommandService:
    """Coordinate durable comparison-pack mutations through one safety gate."""

    def __init__(
        self,
        *,
        active_model_folder_path_provider: Callable[[], str],
        repository: ComparisonPackRepository,
        materializer: ComparisonPackMaterializer,
        sanitize_name: NameSanitizer,
        logger: object,
        models_root_provider: Callable[[], str] | None = None,
        lock_factory: StorageMutationLockFactory = StorageMutationLock,
        lock_timeout_seconds: float = 5.0,
        transaction_factory: ComparisonPackTransactionFactory = StorageTransaction,
        administration_operation_factory: Callable[
            [], AbstractContextManager[AdministrationCompletionGuard]
        ]
        | None = None,
    ) -> None:
        """Initialize the comparison pack command service and its owned state."""
        self._active_model_folder_path_provider = active_model_folder_path_provider
        self._repository = repository
        self._materializer = materializer
        self._sanitize_name = sanitize_name
        self._logger = logger
        self._models_root_provider = models_root_provider
        self._lock_factory = lock_factory
        self._lock_timeout_seconds = lock_timeout_seconds
        self._transaction_factory = transaction_factory
        self._completion_reporting_failed = False
        self._failed_closed_model_root: str | None = None
        self._owned_administration_safety: AdministrationSafetyService | None
        self._administration_operation_factory: Callable[
            [], AbstractContextManager[AdministrationCompletionGuard]
        ]
        if administration_operation_factory is None:
            self._owned_administration_safety = AdministrationSafetyService()
            self._administration_operation_factory = (
                self._owned_administration_safety.administration_operation
            )
        else:
            self._owned_administration_safety = None
            self._administration_operation_factory = administration_operation_factory

    def create_comparison_pack(
        self,
        pack_name: str,
        source_paths: list[str],
    ) -> ComparisonPackCreationOutcome:
        """Create and durably publish a comparison pack."""

        return self._execute_mutation(
            operation="create",
            mutation=lambda: self._create_comparison_pack_locked(
                pack_name,
                source_paths,
            ),
        )

    def rename_comparison_pack(
        self,
        pack_name: str | None,
        renamed_pack_name: str | None,
    ) -> ComparisonPackMutationOutcome:
        """Rename a pack and update its existing manifest atomically."""

        return self._execute_mutation(
            operation="rename",
            mutation=lambda: self._rename_comparison_pack_locked(
                pack_name,
                renamed_pack_name,
            ),
        )

    def delete_comparison_pack(
        self,
        pack_name: str | None,
    ) -> ComparisonPackMutationOutcome:
        """Quarantine and durably delete one comparison pack."""

        return self._execute_mutation(
            operation="delete",
            mutation=lambda: self._delete_comparison_pack_locked(pack_name),
        )

    def update_comparison_pack_annotation(
        self,
        pack_name: str | None,
        annotation: str | None,
    ) -> ComparisonPackMutationOutcome:
        """Update manifest annotation with exact transactional readback."""

        return self._execute_mutation(
            operation="annotate",
            mutation=lambda: self._update_comparison_pack_annotation_locked(
                pack_name,
                annotation,
            ),
        )

    def replace_comparison_pack_sources(
        self,
        pack_name: str | None,
        source_paths: list[str],
    ) -> ComparisonPackMutationOutcome:
        """Replace copied sources while preserving the pack annotation."""

        return self._execute_mutation(
            operation="replace-sources",
            mutation=lambda: self._replace_comparison_pack_sources_locked(
                pack_name,
                source_paths,
            ),
        )

    def _mutation_scope(self) -> tuple[ModelsStorageMutationScope, Path]:
        """Return the mutation scope value owned by this comparison pack command service."""
        model_root = self._active_model_folder_path()
        explicit_models_root = (
            self._models_root_provider()
            if self._models_root_provider is not None
            else None
        )
        scope = ModelsStorageMutationScope.for_model_root(
            model_root,
            models_root=explicit_models_root,
        )
        guarded_model_root = StoragePathGuard(
            scope.models_root
        ).require_descendant(
            model_root,
            must_exist=True,
        )
        return scope, guarded_model_root

    def _active_model_is_current(
        self,
        scope: ModelsStorageMutationScope,
        expected_model_root: Path,
    ) -> bool:
        """Return whether active model is current applies to this comparison pack command service.
        """
        try:
            active_model_root = StoragePathGuard(
                scope.models_root
            ).require_descendant(
                self._active_model_folder_path(),
                must_exist=True,
            )
            active_scope = ModelsStorageMutationScope.for_model_root(
                active_model_root,
                models_root=scope.models_root,
            )
        except (FileNotFoundError, StoragePathViolation, OSError):
            return False
        return (
            active_scope.models_root == scope.models_root
            and active_model_root == expected_model_root
        )

    def _execute_mutation[
        OutcomeT: (ComparisonPackCreationOutcome, ComparisonPackMutationOutcome)
    ](
        self,
        *,
        operation: str,
        mutation: Callable[[], OutcomeT],
    ) -> OutcomeT:
        """Execute the mutation owned by this comparison pack command service."""
        if not self._active_model_folder_path():
            raise RuntimeUnavailableError("Selected model folder is unavailable.")
        if self._locally_failed_closed():
            raise OperationConflict(
                "Comparison-pack administration is blocked until the active "
                "model is reloaded."
            )

        committed: OutcomeT | None = None
        completion_guard: AdministrationCompletionGuard | None = None
        try:
            with self._administration_operation_factory() as context_value:
                completion_guard = self._require_completion_guard(context_value)
                try:
                    scope, expected_model_root = self._mutation_scope()
                    with scope.lock(
                        lock_factory=self._lock_factory,
                        timeout_seconds=self._lock_timeout_seconds,
                    ):
                        if not self._active_model_is_current(
                            scope,
                            expected_model_root,
                        ):
                            raise RuntimeUnavailableError(
                                "Selected model folder is unavailable."
                            )
                        committed = mutation()
                except BaseException as operation_error:
                    if not isinstance(operation_error, Exception):
                        completion_guard.arm_committed_completion(
                            reload_required=True,
                            storage_recovery_required=True,
                        )
                    raise
                if committed is None:
                    raise OperationConflict(
                        "Comparison-pack administration did not produce a "
                        "durable result."
                    )
                self._arm_committed_completion(completion_guard, committed)
            if committed is None:
                raise OperationConflict(
                    "Administration context suppressed an incomplete "
                    "comparison-pack operation."
                )
            completion_guard.confirm_successful_exit()
            return committed
        except BaseException as error:
            self._fail_administration_completion(completion_guard)
            if committed is not None and isinstance(error, Exception):
                return self._degraded_outcome(committed, operation, error)
            if isinstance(error, Exception):
                self._raise_mutation_boundary_failure(error)
            raise

    @staticmethod
    def _require_completion_guard(
        context_value: object,
    ) -> AdministrationCompletionGuard:
        """Return the required completion guard for this comparison pack command service."""
        if not isinstance(context_value, AdministrationCompletionGuard):
            raise OperationConflict(
                "Comparison-pack administration requires a completion-aware "
                "safety gate."
            )
        return context_value

    def _arm_committed_completion[
        OutcomeT: (ComparisonPackCreationOutcome, ComparisonPackMutationOutcome)
    ](
        self,
        completion_guard: AdministrationCompletionGuard,
        outcome: OutcomeT,
    ) -> None:
        """Arm the committed completion owned by this comparison pack
command service."""
        try:
            completion_guard.arm_committed_completion(
                reload_required=outcome.reload_required,
                storage_recovery_required=outcome.storage_recovery_required,
            )
        except BaseException:
            try:
                completion_guard.latch_recovery_required()
            except Exception as reporting_error:
                self._latch_local_completion_failure()
                self._warning(
                    "comparison_pack_completion_reporting_failed "
                    f"exception_type={type(reporting_error).__name__}"
                )
                self._exception(reporting_error)
            raise

    def _fail_administration_completion(
        self,
        completion_guard: AdministrationCompletionGuard | None,
    ) -> None:
        """Fail the administration completion owned by this comparison pack
command service."""
        if completion_guard is None:
            return
        try:
            completion_guard.fail_exit()
        except Exception as error:
            self._latch_local_completion_failure()
            self._warning(
                "comparison_pack_completion_fail_closed "
                f"exception_type={type(error).__name__}"
            )
            self._exception(error)

    def _latch_local_completion_failure(self) -> None:
        """Latch the local completion failure owned by this comparison pack
command service."""
        self._completion_reporting_failed = True
        try:
            self._failed_closed_model_root = self._active_model_folder_path()
        except Exception:
            self._failed_closed_model_root = None

    def _locally_failed_closed(self) -> bool:
        """Return the locally failed closed value owned by this comparison pack command service."""
        if not self._completion_reporting_failed:
            return False
        try:
            active_model_root = self._active_model_folder_path()
        except Exception:
            return True
        if (
            self._failed_closed_model_root is None
            or active_model_root == self._failed_closed_model_root
        ):
            return True
        self._completion_reporting_failed = False
        self._failed_closed_model_root = None
        return False

    def _run_storage_transaction[
        OutcomeT: (ComparisonPackCreationOutcome, ComparisonPackMutationOutcome)
    ](
        self,
        *,
        root: str,
        operation: str,
        persistence_message: str,
        outcome: OutcomeT,
        mutation: Callable[[StorageTransaction], None],
    ) -> OutcomeT:
        """Return the run storage transaction value owned by this comparison pack command service.
        """
        transaction: StorageTransaction | None = None
        try:
            transaction = self._transaction_factory(root, operation=operation)
            with transaction:
                mutation(transaction)
                transaction.commit()
        except Exception as error:
            if transaction is not None and transaction.committed:
                return self._degraded_outcome(outcome, operation, error)
            self._warning(
                "comparison_pack_storage_mutation_failed "
                f"operation={operation} exception_type={type(error).__name__}"
            )
            raise PersistenceError(persistence_message) from error
        return outcome

    def _degraded_outcome[
        OutcomeT: (ComparisonPackCreationOutcome, ComparisonPackMutationOutcome)
    ](
        self,
        outcome: OutcomeT,
        operation: str,
        error: Exception,
    ) -> OutcomeT:
        """Return the degraded outcome value owned by this comparison pack command service."""
        self._warning(
            "comparison_pack_post_commit_failed "
            f"operation={operation} exception_type={type(error).__name__}"
        )
        self._exception(error)
        return replace(
            outcome,
            degraded=True,
            reload_required=True,
            storage_recovery_required=True,
        )

    def _raise_mutation_boundary_failure(self, error: Exception) -> Never:
        """Return the raise mutation boundary failure value provided by this comparison pack
        command service.
        """
        if isinstance(error, ApplicationError):
            raise error
        if isinstance(error, StorageMutationLockTimeout):
            raise OperationTimeout("Model storage is busy. Try again.") from error
        if isinstance(error, FileNotFoundError | NotADirectoryError):
            raise RuntimeUnavailableError(
                "Selected model folder is unavailable."
            ) from error
        if isinstance(error, StoragePathViolation):
            raise RuntimeUnavailableError(
                "Selected model folder is unavailable."
            ) from error
        if isinstance(
            error,
            StorageMutationLockError | StorageTransactionError | OSError,
        ):
            raise PersistenceError(
                "Comparison-pack storage could not be changed safely."
            ) from error
        raise error

    def _create_comparison_pack_locked(
        self,
        pack_name: str,
        source_paths: list[str],
    ) -> ComparisonPackCreationOutcome:
        """Create the comparison pack locked owned by this comparison pack command service."""
        normalized_paths, selected_entries = self._select_sources(source_paths)
        resolved_pack_name = self._sanitize_name(pack_name, limit=80)
        if not resolved_pack_name:
            resolved_pack_name = (
                f"compare_{datetime.now(UTC).strftime('%Y-%m-%d_%H-%M-%S')}"
            )

        comparison_root = self._require_comparison_root()
        try:
            validated_name = ComparisonPackStorageName.parse(resolved_pack_name)
            target_pack_root = self._repository.resolve_pack_path(
                comparison_root,
                validated_name,
            )
        except (InvalidStorageName, StoragePathViolation) as error:
            self._warning(
                "Rejected invalid comparison pack name "
                f"'{resolved_pack_name}': {error}"
            )
            raise ApplicationValidationError(
                "Enter a valid comparison pack name."
            ) from error
        if os.path.exists(target_pack_root):
            raise OperationConflict(
                f"Comparison pack '{resolved_pack_name}' already exists."
            )

        os.makedirs(comparison_root, exist_ok=True)
        outcome = ComparisonPackCreationOutcome(
            pack_name=str(validated_name),
            log_directory=str(target_pack_root),
            source_paths=normalized_paths,
            committed=True,
        )

        def create(transaction: StorageTransaction) -> None:
            """Create a validated value owned by this comparison pack command service."""
            staged_pack_root = transaction.stage_directory(target_pack_root)
            self._materializer.copy_selected_entries(
                str(staged_pack_root),
                selected_entries,
            )
            self._repository.write_manifest(
                str(staged_pack_root),
                str(validated_name),
                None,
                normalized_paths,
            )
            transaction.confirm_path(staged_pack_root)
            transaction.publish(staged_pack_root, target_pack_root)
            transaction.confirm_path(target_pack_root)

        return self._run_storage_transaction(
            root=comparison_root,
            operation="create-comparison-pack",
            persistence_message=(
                "Failed to create comparison pack. Check logs for details."
            ),
            outcome=outcome,
            mutation=create,
        )

    def _rename_comparison_pack_locked(
        self,
        pack_name: str | None,
        renamed_pack_name: str | None,
    ) -> ComparisonPackMutationOutcome:
        """Rename the comparison pack locked owned by this comparison pack command service."""
        comparison_root = self._require_comparison_root()
        plan = self._prepare_rename_plan(
            comparison_root,
            pack_name,
            renamed_pack_name,
        )
        outcome = ComparisonPackMutationOutcome(
            pack_name=plan.candidate_name,
            previous_pack_name=plan.current_name,
            log_directory=str(plan.target_path),
            annotation=plan.annotation,
            source_paths=plan.source_paths,
        )
        return self._run_storage_transaction(
            root=comparison_root,
            operation="rename-comparison-pack",
            persistence_message=(
                "Failed to rename comparison pack. Check logs for details."
            ),
            outcome=outcome,
            mutation=lambda transaction: self._rename_pack_in_transaction(
                transaction,
                plan,
            ),
        )

    def _prepare_rename_plan(
        self,
        comparison_root: str,
        pack_name: str | None,
        renamed_pack_name: str | None,
    ) -> _ComparisonPackRenamePlan:
        """Prepare the rename plan required by this comparison pack command service."""
        raw_current_name = str(pack_name or "").strip()
        raw_candidate_name = self._sanitize_name(
            str(renamed_pack_name or "").strip(),
            limit=80,
        )
        if not raw_current_name:
            raise ApplicationValidationError(
                "Select a comparison pack to rename."
            )
        if not raw_candidate_name:
            raise ApplicationValidationError(
                "Enter a valid comparison pack name."
            )
        try:
            current_name = ComparisonPackStorageName.parse(raw_current_name)
            candidate_name = ComparisonPackStorageName.parse(raw_candidate_name)
        except InvalidStorageName as error:
            raise ApplicationValidationError(
                "Enter a valid comparison pack name."
            ) from error
        if candidate_name == current_name:
            raise OperationConflict("Comparison pack name is unchanged.")

        try:
            source_path = self._repository.resolve_pack_path(
                comparison_root,
                current_name,
            )
            target_path = self._repository.resolve_pack_path(
                comparison_root,
                candidate_name,
            )
        except StoragePathViolation as error:
            raise ApplicationValidationError(
                "Select a valid comparison pack to rename."
            ) from error
        if not os.path.isdir(source_path):
            raise ResourceNotFoundError(
                "Selected comparison pack no longer exists."
            )
        if os.path.exists(target_path):
            raise OperationConflict(
                f"Comparison pack '{candidate_name}' already exists."
            )

        annotation = self._repository.load_annotation(str(source_path))
        source_paths = self._repository.load_source_paths(str(source_path))
        had_manifest = os.path.isfile(
            self._repository.manifest_path(str(source_path))
        )
        return _ComparisonPackRenamePlan(
            current_name=str(current_name),
            candidate_name=str(candidate_name),
            source_path=source_path,
            target_path=target_path,
            annotation=annotation,
            source_paths=tuple(source_paths or ()),
            had_manifest=had_manifest,
        )

    def _rename_pack_in_transaction(
        self,
        transaction: StorageTransaction,
        plan: _ComparisonPackRenamePlan,
    ) -> None:
        """Rename the pack in transaction owned by this comparison pack command service."""
        transaction.move(plan.source_path, plan.target_path)
        if not plan.had_manifest:
            transaction.confirm_path(plan.target_path)
            return
        manifest_path = self._repository.manifest_path(str(plan.target_path))
        transaction.quarantine(manifest_path)
        transaction.track_created(manifest_path)
        self._repository.write_manifest(
            str(plan.target_path),
            plan.candidate_name,
            plan.annotation,
            plan.source_paths or None,
        )
        transaction.confirm_path(manifest_path)

    def _delete_comparison_pack_locked(
        self,
        pack_name: str | None,
    ) -> ComparisonPackMutationOutcome:
        """Delete the comparison pack locked through this comparison pack command service's safety
        boundary.
        """
        comparison_root = self._require_comparison_root()
        current_name, pack_path = self._require_existing_pack(
            comparison_root,
            pack_name,
            missing_selection_message="Select a comparison pack to delete.",
            invalid_selection_message=(
                "Select a valid comparison pack to delete."
            ),
        )
        outcome = ComparisonPackMutationOutcome(pack_name=str(current_name))

        def delete(transaction: StorageTransaction) -> None:
            """Perform the delete step within the enclosing delete comparison pack locked workflow.
            """
            transaction.quarantine(pack_path)

        return self._run_storage_transaction(
            root=comparison_root,
            operation="delete-comparison-pack",
            persistence_message=(
                "Failed to delete comparison pack. Check logs for details."
            ),
            outcome=outcome,
            mutation=delete,
        )

    def _update_comparison_pack_annotation_locked(
        self,
        pack_name: str | None,
        annotation: str | None,
    ) -> ComparisonPackMutationOutcome:
        """Update the comparison pack annotation locked owned by this comparison pack command
        service.
        """
        comparison_root = self._require_comparison_root()
        current_name, pack_path = self._require_existing_pack(
            comparison_root,
            pack_name,
            missing_selection_message="Select a comparison pack to annotate.",
            invalid_selection_message=(
                "Select a valid comparison pack to annotate."
            ),
        )
        cleaned_annotation = str(annotation or "").strip() or None
        source_paths = self._repository.load_source_paths(str(pack_path))
        manifest_path = self._repository.manifest_path(str(pack_path))
        outcome = ComparisonPackMutationOutcome(
            pack_name=str(current_name),
            log_directory=str(pack_path),
            annotation=cleaned_annotation,
            source_paths=tuple(source_paths or ()),
        )

        def update_annotation(transaction: StorageTransaction) -> None:
            """Update the annotation owned by this comparison pack command service."""
            if os.path.isfile(manifest_path):
                transaction.snapshot_file(manifest_path)
            else:
                transaction.track_created(manifest_path)
            self._repository.write_manifest(
                str(pack_path),
                str(current_name),
                cleaned_annotation,
                source_paths,
            )
            self._repository.confirm_annotation_readback(
                str(pack_path),
                pack_name=str(current_name),
                annotation=cleaned_annotation,
                source_paths=source_paths,
            )
            transaction.confirm_path(manifest_path)

        return self._run_storage_transaction(
            root=comparison_root,
            operation="annotate-comparison-pack",
            persistence_message=(
                "Failed to update comparison pack annotation. "
                "Check logs for details."
            ),
            outcome=outcome,
            mutation=update_annotation,
        )

    def _replace_comparison_pack_sources_locked(
        self,
        pack_name: str | None,
        source_paths: list[str],
    ) -> ComparisonPackMutationOutcome:
        """Replace the comparison pack sources locked owned by this comparison pack command
        service.
        """
        comparison_root = self._require_comparison_root()
        current_name, pack_path = self._require_existing_pack(
            comparison_root,
            pack_name,
            missing_selection_message=(
                "Select a comparison pack to update sources."
            ),
            invalid_selection_message=(
                "Select a valid comparison pack to update sources."
            ),
        )
        normalized_paths, selected_entries = self._select_sources(source_paths)
        annotation = self._repository.load_annotation(str(pack_path))
        outcome = ComparisonPackMutationOutcome(
            pack_name=str(current_name),
            log_directory=str(pack_path),
            annotation=annotation,
            source_paths=normalized_paths,
        )

        def replace_sources(transaction: StorageTransaction) -> None:
            """Replace the sources owned by this comparison pack command service."""
            transaction.quarantine(pack_path)
            staged_pack_path = transaction.stage_directory(pack_path)
            self._materializer.copy_selected_entries(
                str(staged_pack_path),
                selected_entries,
            )
            self._repository.write_manifest(
                str(staged_pack_path),
                str(current_name),
                annotation,
                normalized_paths,
            )
            transaction.confirm_path(staged_pack_path)
            transaction.publish(staged_pack_path, pack_path)
            transaction.confirm_path(pack_path)

        return self._run_storage_transaction(
            root=comparison_root,
            operation="replace-comparison-pack-sources",
            persistence_message=(
                "Failed to update comparison pack sources. "
                "Check logs for details."
            ),
            outcome=outcome,
            mutation=replace_sources,
        )

    def _require_comparison_root(self) -> str:
        """Return the required comparison root for this comparison pack command service."""
        comparison_root = self._repository.get_root()
        if comparison_root is None:
            raise RuntimeUnavailableError("Selected model folder is unavailable.")
        return comparison_root

    def _require_existing_pack(
        self,
        comparison_root: str,
        pack_name: str | None,
        *,
        missing_selection_message: str,
        invalid_selection_message: str,
    ) -> tuple[ComparisonPackStorageName, Path]:
        """Return the required existing pack for this comparison pack command service."""
        raw_current_name = str(pack_name or "").strip()
        if not raw_current_name:
            raise ApplicationValidationError(missing_selection_message)
        try:
            current_name = ComparisonPackStorageName.parse(raw_current_name)
            pack_path = self._repository.resolve_pack_path(
                comparison_root,
                current_name,
            )
        except (InvalidStorageName, StoragePathViolation) as error:
            raise ApplicationValidationError(invalid_selection_message) from error
        if not os.path.isdir(pack_path):
            raise ResourceNotFoundError(
                "Selected comparison pack no longer exists."
            )
        return current_name, pack_path

    def _select_sources(
        self,
        source_paths: list[str],
    ) -> tuple[tuple[str, ...], tuple[TensorboardPackEntry, ...]]:
        """Select the sources owned by this comparison pack command service."""
        normalized_paths = self._materializer.normalize_source_paths(source_paths)
        if len(normalized_paths) < 2:
            raise ApplicationValidationError(
                "Select at least two TensorBoard packs."
            )
        selected_entries = self._materializer.resolve_selected_entries(
            normalized_paths
        )
        if len(selected_entries) < 2:
            raise ResourceNotFoundError(
                "Selected packs are missing or no longer available."
            )
        return normalized_paths, selected_entries

    def _active_model_folder_path(self) -> str:
        """Return the active model folder path value owned by this comparison pack command service.
        """
        return str(self._active_model_folder_path_provider() or "")

    def _warning(self, message: str) -> None:
        """Publish a warning record through this comparison pack command service."""
        warning = getattr(self._logger, "warning", None)
        if not callable(warning):
            return
        try:
            warning(message)
        except Exception:
            return

    def _exception(self, error: BaseException) -> None:
        """Publish a exception record through this comparison pack command service."""
        exception = getattr(self._logger, "exception", None)
        if not callable(exception):
            return
        exception_info = (type(error), error, error.__traceback__)
        try:
            exception(error, exc_info=exception_info)
        except TypeError:
            try:
                exception(error)
            except Exception:
                return
        except Exception:
            return


__all__ = ["ComparisonPackCommandService"]
