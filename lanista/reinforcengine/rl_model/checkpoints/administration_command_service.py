# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Checkpoint delete, rename, annotation, and runtime-projection commands."""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from lanista.reinforcengine.application_errors import (
    ApplicationError,
    OperationResult,
    OperationTimeout,
    RetryPolicy,
    RuntimeUnavailableError,
    failure_from_exception,
)
from lanista.reinforcengine.rl_model.administration_command_policy import (
    AdministrationCommandPolicy,
)
from lanista.reinforcengine.rl_model.administration_contracts import (
    CheckpointAnnotationOutcome,
    CheckpointDeletionOutcome,
    CheckpointRenameOutcome,
    ModelAdministrationOwner,
)
from lanista.reinforcengine.rl_model.checkpoints.mutation_contracts import (
    CheckpointMutationPort,
)


class CheckpointAdministrationCommandService:
    """Own durable checkpoint mutations and active-selection projection."""

    def __init__(
        self,
        *,
        policy: AdministrationCommandPolicy,
        checkpoint_mutations: CheckpointMutationPort,
        selected_session_provider: Callable[[], str | None],
        selected_checkpoint_provider: Callable[[], str | None],
        get_latest_checkpoint_provider: Callable[[str], str | None],
        set_current_model_runtime_selection_provider: Callable[
            [str | None, str | None, bool], None
        ],
    ) -> None:
        """Initialize the checkpoint administration command service and its owned state."""
        self._policy = policy
        self._checkpoint_mutations = checkpoint_mutations
        self._selected_session_provider = selected_session_provider
        self._selected_checkpoint_provider = selected_checkpoint_provider
        self._get_latest_checkpoint_provider = get_latest_checkpoint_provider
        self._set_current_model_runtime_selection_provider = (
            set_current_model_runtime_selection_provider
        )

    def delete_checkpoint(
        self,
        owner: ModelAdministrationOwner,
        session_name: str | None,
        checkpoint_name: str | None,
        *,
        operation_id: str | None = None,
    ) -> OperationResult[CheckpointDeletionOutcome]:
        """Delete an owned checkpoint and report projection degradation."""

        correlation_id = operation_id or uuid4().hex
        try:
            outcome = self._policy.execute(
                lambda: self._delete_under_policy(
                    session_name,
                    checkpoint_name,
                    operation_id=correlation_id,
                ),
                validate=lambda: self._require_runtime(owner),
                context_exit_recovery=lambda committed, error: (
                    self._deletion_outcome(
                        session_name=committed.session_name,
                        checkpoint_name=committed.checkpoint_name,
                        replacement_checkpoint_name=(
                            committed.replacement_checkpoint_name
                        ),
                        projection_synchronized=committed.projection_synchronized,
                        post_commit_error=error,
                        operation_id=correlation_id,
                    )
                ),
            )
            return OperationResult.success(outcome, operation_id=correlation_id)
        except Exception as error:
            return self._failure_result(
                error,
                operation_id=correlation_id,
                operation="delete",
            )

    def rename_checkpoint(
        self,
        owner: ModelAdministrationOwner,
        session_name: str | None,
        checkpoint_name: str | None,
        renamed_checkpoint_name: str | None,
        *,
        operation_id: str | None = None,
    ) -> OperationResult[CheckpointRenameOutcome]:
        """Rename an owned checkpoint and report projection degradation."""

        correlation_id = operation_id or uuid4().hex
        try:
            outcome = self._policy.execute(
                lambda: self._rename_under_policy(
                    session_name,
                    checkpoint_name,
                    renamed_checkpoint_name,
                    operation_id=correlation_id,
                ),
                validate=lambda: self._require_runtime(owner),
                context_exit_recovery=lambda committed, error: (
                    self._rename_outcome(
                        session_name=committed.session_name,
                        previous_checkpoint_name=(
                            committed.previous_checkpoint_name
                        ),
                        checkpoint_name=committed.checkpoint_name,
                        projection_synchronized=committed.projection_synchronized,
                        post_commit_error=error,
                        operation_id=correlation_id,
                    )
                ),
            )
            return OperationResult.success(outcome, operation_id=correlation_id)
        except Exception as error:
            return self._failure_result(
                error,
                operation_id=correlation_id,
                operation="rename",
            )

    def update_checkpoint_annotation(
        self,
        owner: ModelAdministrationOwner,
        session_name: str | None,
        checkpoint_name: str | None,
        annotation: str | None,
        *,
        operation_id: str | None = None,
    ) -> OperationResult[CheckpointAnnotationOutcome]:
        """Update checkpoint annotation without changing runtime selection."""

        correlation_id = operation_id or uuid4().hex
        try:
            outcome = self._policy.execute(
                lambda: self._update_annotation_under_policy(
                    session_name,
                    checkpoint_name,
                    annotation,
                    operation_id=correlation_id,
                ),
                validate=lambda: self._require_runtime(owner),
                context_exit_recovery=lambda committed, error: (
                    self._annotation_outcome(
                        session_name=committed.session_name,
                        checkpoint_name=committed.checkpoint_name,
                        annotation=committed.annotation,
                        post_commit_error=error,
                        operation_id=correlation_id,
                    )
                ),
            )
            return OperationResult.success(outcome, operation_id=correlation_id)
        except Exception as error:
            return self._failure_result(
                error,
                operation_id=correlation_id,
                operation="annotate",
            )

    def _require_runtime(self, owner: ModelAdministrationOwner) -> None:
        """Return the required runtime for this checkpoint administration command service."""
        self._policy.require_available()
        if self._policy.active_configuration() is None:
            raise RuntimeUnavailableError(
                "Load a model before changing checkpoint storage."
            )
        self._policy.require_current_owner(owner)

    def _delete_under_policy(
        self,
        session_name: str | None,
        checkpoint_name: str | None,
        *,
        operation_id: str,
    ) -> CheckpointDeletionOutcome:
        """Delete the under policy through this checkpoint administration command service's safety
        boundary.
        """
        commit = self._checkpoint_mutations.delete(
            session_name,
            checkpoint_name,
        )
        replacement_checkpoint_name, projection_synchronized = (
            self._project_deletion(
                session_name,
                checkpoint_name,
                operation_id=operation_id,
            )
        )
        return self._deletion_outcome(
            session_name=session_name,
            checkpoint_name=commit.value,
            replacement_checkpoint_name=replacement_checkpoint_name,
            projection_synchronized=projection_synchronized,
            post_commit_error=commit.post_commit_error,
            operation_id=operation_id,
        )

    def _rename_under_policy(
        self,
        session_name: str | None,
        checkpoint_name: str | None,
        renamed_checkpoint_name: str | None,
        *,
        operation_id: str,
    ) -> CheckpointRenameOutcome:
        """Rename the under policy owned by this checkpoint administration command service."""
        commit = self._checkpoint_mutations.rename(
            session_name,
            checkpoint_name,
            renamed_checkpoint_name,
        )
        projection_synchronized = self._project_rename(
            session_name,
            checkpoint_name,
            commit.value,
            operation_id=operation_id,
        )
        return self._rename_outcome(
            session_name=session_name,
            previous_checkpoint_name=checkpoint_name,
            checkpoint_name=commit.value,
            projection_synchronized=projection_synchronized,
            post_commit_error=commit.post_commit_error,
            operation_id=operation_id,
        )

    def _update_annotation_under_policy(
        self,
        session_name: str | None,
        checkpoint_name: str | None,
        annotation: str | None,
        *,
        operation_id: str,
    ) -> CheckpointAnnotationOutcome:
        """Update the annotation under policy owned by this checkpoint administration command
        service.
        """
        commit = self._checkpoint_mutations.update_annotation(
            session_name,
            checkpoint_name,
            annotation,
        )
        return self._annotation_outcome(
            session_name=session_name,
            checkpoint_name=checkpoint_name,
            annotation=commit.value,
            post_commit_error=commit.post_commit_error,
            operation_id=operation_id,
        )

    def _deletion_outcome(
        self,
        *,
        session_name: str | None,
        checkpoint_name: str,
        replacement_checkpoint_name: str | None,
        projection_synchronized: bool,
        post_commit_error: Exception | None,
        operation_id: str,
    ) -> CheckpointDeletionOutcome:
        """Return the deletion outcome value owned by this checkpoint administration command
        service.
        """
        reload_required, storage_recovery_required = (
            self._degradation_requirements(
                projection_synchronized=projection_synchronized,
                post_commit_error=post_commit_error,
                operation_id=operation_id,
                operation="delete",
            )
        )
        degraded = reload_required or storage_recovery_required
        if degraded:
            self._record_degradation(
                operation_id,
                "delete",
                reload_required=reload_required,
                storage_recovery_required=storage_recovery_required,
            )
        return CheckpointDeletionOutcome(
            session_name=str(session_name),
            checkpoint_name=checkpoint_name,
            replacement_checkpoint_name=replacement_checkpoint_name,
            committed=True,
            degraded=degraded,
            reload_required=reload_required,
            storage_recovery_required=storage_recovery_required,
            projection_synchronized=projection_synchronized,
        )

    def _rename_outcome(
        self,
        *,
        session_name: str | None,
        previous_checkpoint_name: str | None,
        checkpoint_name: str,
        projection_synchronized: bool,
        post_commit_error: Exception | None,
        operation_id: str,
    ) -> CheckpointRenameOutcome:
        """Rename the outcome owned by this checkpoint administration command service."""
        reload_required, storage_recovery_required = (
            self._degradation_requirements(
                projection_synchronized=projection_synchronized,
                post_commit_error=post_commit_error,
                operation_id=operation_id,
                operation="rename",
            )
        )
        degraded = reload_required or storage_recovery_required
        if degraded:
            self._record_degradation(
                operation_id,
                "rename",
                reload_required=reload_required,
                storage_recovery_required=storage_recovery_required,
            )
        return CheckpointRenameOutcome(
            session_name=str(session_name),
            previous_checkpoint_name=str(previous_checkpoint_name),
            checkpoint_name=checkpoint_name,
            committed=True,
            degraded=degraded,
            reload_required=reload_required,
            storage_recovery_required=storage_recovery_required,
            projection_synchronized=projection_synchronized,
        )

    def _annotation_outcome(
        self,
        *,
        session_name: str | None,
        checkpoint_name: str | None,
        annotation: str | None,
        post_commit_error: Exception | None,
        operation_id: str,
    ) -> CheckpointAnnotationOutcome:
        """Return the annotation outcome value owned by this checkpoint administration command
        service.
        """
        reload_required, storage_recovery_required = (
            self._degradation_requirements(
                projection_synchronized=True,
                post_commit_error=post_commit_error,
                operation_id=operation_id,
                operation="annotate",
            )
        )
        degraded = reload_required or storage_recovery_required
        if degraded:
            self._record_degradation(
                operation_id,
                "annotate",
                reload_required=reload_required,
                storage_recovery_required=storage_recovery_required,
            )
        return CheckpointAnnotationOutcome(
            session_name=str(session_name),
            checkpoint_name=str(checkpoint_name),
            annotation=annotation,
            committed=True,
            degraded=degraded,
            reload_required=reload_required,
            storage_recovery_required=storage_recovery_required,
            projection_synchronized=True,
        )

    def _degradation_requirements(
        self,
        *,
        projection_synchronized: bool,
        post_commit_error: Exception | None,
        operation_id: str,
        operation: str,
    ) -> tuple[bool, bool]:
        """Return the degradation requirements value owned by this checkpoint administration
        command service.
        """
        storage_recovery_required = post_commit_error is not None
        reload_required = not projection_synchronized or storage_recovery_required
        if post_commit_error is not None:
            self._log_post_commit_errors(
                post_commit_error,
                operation_id=operation_id,
                operation=operation,
            )
        return reload_required, storage_recovery_required

    def _log_post_commit_errors(
        self,
        error: Exception,
        *,
        operation_id: str,
        operation: str,
    ) -> None:
        """Log the post commit errors owned by this checkpoint administration command service."""
        if isinstance(error, ExceptionGroup):
            for nested_error in error.exceptions:
                if isinstance(nested_error, Exception):
                    self._log_post_commit_errors(
                        nested_error,
                        operation_id=operation_id,
                        operation=operation,
                    )
            return
        self._policy.warning(
            "checkpoint_post_commit_failed "
            f"operation_id={operation_id} operation={operation} "
            f"exception_type={type(error).__name__}"
        )
        self._policy.exception(error)

    def _project_deletion(
        self,
        session_name: str | None,
        checkpoint_name: str | None,
        *,
        operation_id: str,
    ) -> tuple[str | None, bool]:
        """Project the deletion owned by this checkpoint administration command service."""
        replacement_checkpoint_name: str | None = None
        try:
            if (
                self._selected_session_provider() != session_name
                or self._selected_checkpoint_provider() != checkpoint_name
            ):
                return None, True
            replacement_checkpoint_name = (
                self._get_latest_checkpoint_provider(session_name)
                if session_name
                else None
            )
            self._set_current_model_runtime_selection_provider(
                session_name,
                replacement_checkpoint_name,
                True,
            )
        except Exception as error:
            self._log_projection_failure(
                error,
                operation_id=operation_id,
                operation="delete",
            )
            return replacement_checkpoint_name, False
        return replacement_checkpoint_name, True

    def _project_rename(
        self,
        session_name: str | None,
        previous_checkpoint_name: str | None,
        checkpoint_name: str,
        *,
        operation_id: str,
    ) -> bool:
        """Project the rename owned by this checkpoint administration command service."""
        try:
            if (
                self._selected_session_provider() != session_name
                or self._selected_checkpoint_provider() != previous_checkpoint_name
            ):
                return True
            self._set_current_model_runtime_selection_provider(
                session_name,
                checkpoint_name,
                True,
            )
        except Exception as error:
            self._log_projection_failure(
                error,
                operation_id=operation_id,
                operation="rename",
            )
            return False
        return True

    def _log_projection_failure(
        self,
        error: Exception,
        *,
        operation_id: str,
        operation: str,
    ) -> None:
        """Log the projection failure owned by this checkpoint administration command service."""
        self._policy.warning(
            "checkpoint_projection_failed "
            f"operation_id={operation_id} operation={operation} "
            f"exception_type={type(error).__name__}"
        )
        self._policy.exception(error)

    def _record_degradation(
        self,
        operation_id: str,
        operation: str,
        *,
        reload_required: bool,
        storage_recovery_required: bool,
    ) -> None:
        """Record the degradation owned by this checkpoint administration command service."""
        self._policy.report_degradation(
            operation_id=operation_id,
            failure_event="checkpoint_degraded_state_report_failed",
            reload_required=reload_required,
            storage_recovery_required=storage_recovery_required,
            detail=f"operation={operation}",
        )

    def _failure_result[OutcomeT](
        self,
        error: Exception,
        *,
        operation_id: str,
        operation: str,
    ) -> OperationResult[OutcomeT]:
        """Return the failure result value owned by this checkpoint administration command service.
        """
        failure = failure_from_exception(
            error,
            operation_id=operation_id,
            retry_policy=(
                RetryPolicy.SAFE if isinstance(error, OperationTimeout) else None
            ),
        )
        self._policy.warning(
            "checkpoint_mutation_failed "
            f"operation_id={operation_id} operation={operation} "
            f"failure_code={failure.code} "
            f"exception_type={type(error).__name__}"
        )
        if not isinstance(error, ApplicationError):
            self._policy.exception(error)
        return OperationResult.failed(failure)
