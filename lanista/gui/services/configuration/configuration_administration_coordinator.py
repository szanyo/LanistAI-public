# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Deterministic session/checkpoint administration for configuration UI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from lanista.gui.contracts import UiNotification
from lanista.gui.services.configuration.configuration_administration_contracts import (
    CheckpointAdministrationAction,
    CheckpointAdministrationEvent,
    CheckpointDeletionRequest,
    CheckpointSelectorIdentity,
    ConfigurationAdministrationBoundary,
    ConfigurationAdministrationEffect,
    ConfigurationAdministrationState,
    ConfigurationAdministrationTransition,
    ConfirmCheckpointDeletion,
    ConfirmSessionDeletion,
    DismissConfiguration,
    OpenCheckpointSelector,
    OpenSessionSelector,
    PresentAdministrationNotification,
    ReopenCheckpointSelector,
    ReopenSessionSelector,
    SessionAdministrationAction,
    SessionAdministrationEvent,
    SessionDeletionRequest,
    SessionSelectorIdentity,
)
from lanista.gui.services.configuration.control_panel_configuration_contracts import (
    ConfigurationDialogAction,
)
from lanista.reinforcengine.application_errors import FailureCode, OperationFailure
from lanista.reinforcengine.rl_model.administration_contracts import ModelAdministrationOwner


class ConfigurationAdministrationCoordinator:
    """Own immutable workflow snapshots and application administration calls."""

    def __init__(
        self,
        *,
        owner: ModelAdministrationOwner | None,
        session_name: str | None,
        checkpoint_name: str | None,
        boundary_provider: Callable[[], ConfigurationAdministrationBoundary],
    ) -> None:
        """Initialize the configuration administration coordinator and its owned state."""
        self._state = ConfigurationAdministrationState(
            owner=owner,
            generation=0,
            session_name=session_name,
            checkpoint_name=checkpoint_name,
        )
        self._boundary_provider = boundary_provider

    @property
    def state(self) -> ConfigurationAdministrationState:
        """Return the state exposed by this configuration administration coordinator."""
        return self._state

    def project_session_name(self, session_name: str | None) -> None:
        """Synchronize a UI selection projection without changing identity."""

        self._state = replace(self._state, session_name=session_name)

    def project_checkpoint_name(self, checkpoint_name: str | None) -> None:
        """Synchronize a UI selection projection without changing identity."""

        self._state = replace(self._state, checkpoint_name=checkpoint_name)

    def invalidate(self) -> ConfigurationAdministrationTransition:
        """Reject every callback issued by an earlier child workflow."""

        return self._commit(replace(self._state, generation=self._state.generation + 1))

    def open_session_selector(
        self,
        *,
        enabled: bool,
        active: bool,
    ) -> ConfigurationAdministrationTransition:
        """Open the session selector workflow owned by this configuration administration
        coordinator.
        """
        if not enabled:
            return self._unchanged()
        claimed = self._claim_owner(active=active)
        if claimed is None:
            return self._unchanged()
        owner, generation = claimed
        identity = SessionSelectorIdentity(owner=owner, generation=generation)
        return self._transition(OpenSessionSelector(identity))

    def open_checkpoint_selector(
        self,
        *,
        enabled: bool,
        active: bool,
    ) -> ConfigurationAdministrationTransition:
        """Open the checkpoint selector workflow owned by this configuration administration
        coordinator.
        """
        session_name = self._state.session_name
        if not enabled or session_name is None:
            return self._unchanged()
        claimed = self._claim_owner(active=active)
        if claimed is None:
            return self._unchanged()
        owner, generation = claimed
        identity = CheckpointSelectorIdentity(
            owner=owner,
            generation=generation,
            session_name=session_name,
        )
        return self._transition(OpenCheckpointSelector(identity))

    def handle_session_result(
        self,
        identity: SessionSelectorIdentity,
        event: SessionAdministrationEvent,
        *,
        active: bool,
    ) -> ConfigurationAdministrationTransition:
        """Handle the session result workflow for this configuration administration coordinator."""
        boundary = self._current_boundary(identity.owner, identity.generation, active=active)
        if boundary is None:
            return self._unchanged()
        if event.action is SessionAdministrationAction.SELECT:
            return self._select_session(event.session_name)
        if event.action is SessionAdministrationAction.EDIT:
            return self._edit_session(event.session_name)
        if event.action is SessionAdministrationAction.DELETE and event.session_name:
            display_name = boundary.session_references.get_session_reference_display(
                event.session_name
            )
            return self._transition(
                ConfirmSessionDeletion(
                    SessionDeletionRequest(
                        identity=identity,
                        session_name=event.session_name,
                        display_name=display_name,
                    )
                )
            )
        return self._unchanged()

    def confirm_session_deletion(
        self,
        request: SessionDeletionRequest,
        confirmed: bool | None,
        *,
        active: bool,
    ) -> ConfigurationAdministrationTransition:
        """Confirm the session deletion owned by this configuration administration coordinator."""
        identity = request.identity
        boundary = self._current_boundary(identity.owner, identity.generation, active=active)
        if boundary is None:
            return self._unchanged()
        if not confirmed:
            return self._transition(ReopenSessionSelector())

        result = boundary.session_commands.delete_session(
            identity.owner,
            request.session_name,
        )
        if not result.succeeded or result.value is None:
            if self._is_conflict(result.failure):
                return self._unchanged()
            return self._transition(
                self._failure_notification(
                    result.failure,
                    fallback="Failed to delete session.",
                    title="Delete failed",
                ),
                ReopenSessionSelector(),
            )

        outcome = result.value
        next_state = replace(
            self._state,
            session_name=outcome.selection.session_name,
            checkpoint_name=outcome.selection.checkpoint_name,
        )
        notification = self._session_deleted_notification(
            request.display_name,
            degraded=outcome.degraded,
        )
        return self._commit(
            next_state,
            PresentAdministrationNotification(notification),
            DismissConfiguration(ConfigurationDialogAction.REFRESH_SESSION_SELECTOR),
        )

    def handle_checkpoint_result(
        self,
        identity: CheckpointSelectorIdentity,
        event: CheckpointAdministrationEvent,
        *,
        active: bool,
    ) -> ConfigurationAdministrationTransition:
        """Handle the checkpoint result workflow for this configuration administration coordinator.
        """
        boundary = self._current_boundary(identity.owner, identity.generation, active=active)
        if boundary is None:
            return self._unchanged()
        if event.action is CheckpointAdministrationAction.SELECT:
            return self._select_checkpoint(identity.session_name, event.checkpoint_name)
        if event.action is CheckpointAdministrationAction.EDIT:
            return self._edit_checkpoint(identity.session_name, event.checkpoint_name)
        if event.action is CheckpointAdministrationAction.DELETE and event.checkpoint_name:
            return self._transition(
                ConfirmCheckpointDeletion(
                    CheckpointDeletionRequest(
                        identity=identity,
                        checkpoint_name=event.checkpoint_name,
                    )
                )
            )
        if (
            event.action is CheckpointAdministrationAction.RENAME
            and event.checkpoint_name
            and event.renamed_checkpoint_name
        ):
            return self._rename_checkpoint(
                boundary,
                identity,
                event.checkpoint_name,
                event.renamed_checkpoint_name,
            )
        return self._unchanged()

    def confirm_checkpoint_deletion(
        self,
        request: CheckpointDeletionRequest,
        confirmed: bool | None,
        *,
        active: bool,
    ) -> ConfigurationAdministrationTransition:
        """Confirm the checkpoint deletion owned by this configuration administration coordinator.
        """
        identity = request.identity
        boundary = self._current_boundary(identity.owner, identity.generation, active=active)
        if boundary is None:
            return self._unchanged()
        if not confirmed:
            return self._transition(ReopenCheckpointSelector())

        result = boundary.checkpoint_commands.delete_checkpoint(
            identity.owner,
            identity.session_name,
            request.checkpoint_name,
        )
        if not result.succeeded or result.value is None:
            effects: tuple[ConfigurationAdministrationEffect, ...] = (
                self._failure_notification(
                    result.failure,
                    fallback="Failed to delete checkpoint.",
                    title="Delete failed",
                ),
            )
            if not self._is_conflict(result.failure):
                effects += (ReopenCheckpointSelector(),)
            return self._commit(self._state, *effects)

        outcome = result.value
        next_state = self._state
        if (
            identity.session_name == self._state.session_name
            and request.checkpoint_name == self._state.checkpoint_name
        ):
            next_state = replace(
                next_state,
                checkpoint_name=outcome.replacement_checkpoint_name,
            )
        notification = self._checkpoint_deleted_notification(
            outcome.checkpoint_name,
            degraded=outcome.degraded,
        )
        return self._commit(
            next_state,
            PresentAdministrationNotification(notification),
            DismissConfiguration(ConfigurationDialogAction.REFRESH_CHECKPOINT_SELECTOR),
        )

    def _rename_checkpoint(
        self,
        boundary: ConfigurationAdministrationBoundary,
        identity: CheckpointSelectorIdentity,
        checkpoint_name: str,
        renamed_checkpoint_name: str,
    ) -> ConfigurationAdministrationTransition:
        """Rename the checkpoint owned by this configuration administration coordinator."""
        result = boundary.checkpoint_commands.rename_checkpoint(
            identity.owner,
            identity.session_name,
            checkpoint_name,
            renamed_checkpoint_name,
        )
        if not result.succeeded or result.value is None:
            return self._transition(
                self._failure_notification(
                    result.failure,
                    fallback="Failed to rename checkpoint.",
                    title="Rename failed",
                )
            )

        outcome = result.value
        next_state = self._state
        if (
            identity.session_name == self._state.session_name
            and checkpoint_name == self._state.checkpoint_name
        ):
            next_state = replace(next_state, checkpoint_name=outcome.checkpoint_name)
        notification = self._checkpoint_renamed_notification(
            outcome.checkpoint_name,
            degraded=outcome.degraded,
        )
        return self._commit(
            next_state,
            PresentAdministrationNotification(notification),
            DismissConfiguration(ConfigurationDialogAction.REFRESH),
        )

    def _select_session(self, session_name: str | None) -> ConfigurationAdministrationTransition:
        """Select the session owned by this configuration administration coordinator."""
        if session_name == self._state.session_name:
            return self._unchanged()
        next_state = replace(
            self._state,
            session_name=session_name,
            checkpoint_name=None,
        )
        return self._commit(
            next_state,
            DismissConfiguration(ConfigurationDialogAction.REFRESH),
        )

    def _edit_session(self, session_name: str | None) -> ConfigurationAdministrationTransition:
        """Return the edit session value owned by this configuration administration coordinator."""
        next_state = replace(
            self._state,
            session_name=session_name,
            checkpoint_name=None,
        )
        return self._commit(
            next_state,
            DismissConfiguration(ConfigurationDialogAction.EDIT_SESSION),
        )

    def _select_checkpoint(
        self,
        session_name: str,
        checkpoint_name: str | None,
    ) -> ConfigurationAdministrationTransition:
        """Select the checkpoint owned by this configuration administration coordinator."""
        if session_name != self._state.session_name:
            return self._unchanged()
        if checkpoint_name == self._state.checkpoint_name:
            return self._unchanged()
        return self._commit(
            replace(self._state, checkpoint_name=checkpoint_name),
            DismissConfiguration(ConfigurationDialogAction.REFRESH),
        )

    def _edit_checkpoint(
        self,
        session_name: str,
        checkpoint_name: str | None,
    ) -> ConfigurationAdministrationTransition:
        """Return the edit checkpoint value owned by this configuration administration coordinator.
        """
        if session_name != self._state.session_name:
            return self._unchanged()
        next_state = self._state
        if checkpoint_name is not None:
            next_state = replace(next_state, checkpoint_name=checkpoint_name)
        return self._commit(
            next_state,
            DismissConfiguration(ConfigurationDialogAction.EDIT_CHECKPOINT),
        )

    def _claim_owner(self, *, active: bool) -> tuple[ModelAdministrationOwner, int] | None:
        """Return the claim owner value owned by this configuration administration coordinator."""
        owner = self._state.owner
        if owner is None or not active:
            return None
        if not self._boundary_provider().models.is_owner_current(owner):
            return None
        generation = self._state.generation + 1
        self._state = replace(self._state, generation=generation)
        return owner, generation

    def _current_boundary(
        self,
        owner: ModelAdministrationOwner,
        generation: int,
        *,
        active: bool,
    ) -> ConfigurationAdministrationBoundary | None:
        """Return the current boundary value owned by this configuration administration
        coordinator.
        """
        if not active or self._state.owner != owner or self._state.generation != generation:
            return None
        boundary = self._boundary_provider()
        return boundary if boundary.models.is_owner_current(owner) else None

    def _transition(
        self,
        *effects: ConfigurationAdministrationEffect,
    ) -> ConfigurationAdministrationTransition:
        """Return the transition value owned by this configuration administration coordinator."""
        return self._commit(self._state, *effects)

    def _unchanged(self) -> ConfigurationAdministrationTransition:
        """Return the unchanged value owned by this configuration administration coordinator."""
        return ConfigurationAdministrationTransition(self._state)

    def _commit(
        self,
        state: ConfigurationAdministrationState,
        *effects: ConfigurationAdministrationEffect,
    ) -> ConfigurationAdministrationTransition:
        """Commit the operation owned by this configuration administration coordinator."""
        self._state = state
        return ConfigurationAdministrationTransition(state=state, effects=effects)

    @staticmethod
    def _is_conflict(failure: OperationFailure | None) -> bool:
        """Return whether conflict applies to this configuration administration coordinator."""
        return failure is not None and failure.code is FailureCode.OPERATION_CONFLICT

    @staticmethod
    def _failure_notification(
        failure: OperationFailure | None,
        *,
        fallback: str,
        title: str,
    ) -> PresentAdministrationNotification:
        """Return the failure notification value owned by this configuration administration
        coordinator.
        """
        return PresentAdministrationNotification(
            UiNotification(
                message=failure.message if failure is not None else fallback,
                title=title,
                severity="error",
            )
        )

    @staticmethod
    def _session_deleted_notification(
        display_name: str,
        *,
        degraded: bool,
    ) -> UiNotification:
        """Return the session deleted notification value owned by this configuration administration
        coordinator.
        """
        if degraded:
            return UiNotification(
                message=(
                    f"Session deleted: {display_name}. Reload the active model before continuing."
                ),
                title="Session deleted; reload required",
                severity="warning",
            )
        return UiNotification(
            message=f"Session deleted: {display_name}",
            title="Session deleted",
        )

    @staticmethod
    def _checkpoint_deleted_notification(
        checkpoint_name: str,
        *,
        degraded: bool,
    ) -> UiNotification:
        """Checkpoint the deleted notification owned by this configuration administration
        coordinator.
        """
        if degraded:
            return UiNotification(
                message=(
                    f"Checkpoint deleted: {checkpoint_name}. "
                    "Reload the active model before continuing."
                ),
                title="Checkpoint deleted; reload required",
                severity="warning",
            )
        return UiNotification(
            message=f"Checkpoint deleted: {checkpoint_name}",
            title="Checkpoint deleted",
        )

    @staticmethod
    def _checkpoint_renamed_notification(
        checkpoint_name: str,
        *,
        degraded: bool,
    ) -> UiNotification:
        """Checkpoint the renamed notification owned by this configuration administration
        coordinator.
        """
        if degraded:
            return UiNotification(
                message=(
                    f"Checkpoint renamed to: {checkpoint_name}. "
                    "Reload the active model before continuing."
                ),
                title="Checkpoint renamed; reload required",
                severity="warning",
            )
        return UiNotification(
            message=f"Checkpoint renamed to: {checkpoint_name}",
            title="Checkpoint renamed",
        )
