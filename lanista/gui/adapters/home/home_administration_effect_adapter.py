# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Own Home session/checkpoint modal callbacks across one screen lifetime."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from lanista.gui.contracts import UiNotification
from lanista.gui.predefs.secure_delete_dialog_definitions import (
    secure_delete_checkpoint_dialog,
    secure_delete_session_dialog,
)
from lanista.gui.services.home.home_configuration_flow_service import (
    HomeConfigurationFlowService,
)
from lanista.gui.widgets.dialogs.checkpoint_configuration_dialog import (
    CheckpointConfigurationDialog,
    CheckpointConfigurationDialogResult,
)
from lanista.gui.widgets.dialogs.secure_yes_no_dialog import SecureYesNoDialog
from lanista.gui.widgets.dialogs.session_configuration_dialog import (
    SessionConfigurationDialog,
    SessionConfigurationDialogResult,
)
from lanista.reinforcengine.rl_model.administration_contracts import ModelAdministrationOwner
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.model import RLModelRunningModes


class HomeAdministrationViewPort(Protocol):
    """Narrow HomeScreen surface used by administration effects."""

    def administration_effects_available(self) -> bool:
        """Return whether administration effects available applies to this home administration view
        port.
        """
        ...

    def push_session_administration_dialog(
        self,
        dialog: SessionConfigurationDialog,
        callback: Callable[[SessionConfigurationDialogResult | None], None],
    ) -> None:
        """Push the session administration dialog owned by this home administration view port."""
        ...

    def push_checkpoint_administration_dialog(
        self,
        dialog: CheckpointConfigurationDialog,
        callback: Callable[[CheckpointConfigurationDialogResult | None], None],
    ) -> None:
        """Push the checkpoint administration dialog owned by this home administration view port."""
        ...

    def push_administration_confirmation(
        self,
        dialog: SecureYesNoDialog,
        callback: Callable[[bool | None], None],
    ) -> None:
        """Push the administration confirmation owned by this home administration view port."""
        ...

    def refresh_administration_runtime(self) -> None:
        """Refresh the administration runtime owned by this home
administration view port."""
        ...

    def notify_administration(self, notification: UiNotification) -> None:
        """Notify the caller about the administration outcome."""
        ...

    def open_administration_edit(
        self,
        *,
        session_name: str,
        checkpoint_name: str | None,
        expected_model_uuid: str,
        reopen_opener: Callable[[], None],
    ) -> None:
        """Open the administration edit workflow owned by this home administration view port."""
        ...

    def reopen_session_administration(self) -> None:
        """Reopen the session administration owned by this home administration view port."""
        ...

    def reopen_checkpoint_administration(self) -> None:
        """Reopen the checkpoint administration owned by this home administration view port."""
        ...


class HomeAdministrationEffectAdapter:
    """Execute owner-checked session/checkpoint modal effects exactly once."""

    def __init__(
        self,
        *,
        view: HomeAdministrationViewPort,
        services: ApplicationServices,
        flow: HomeConfigurationFlowService,
        is_critical_phase: Callable[[], bool],
        selected_mode: Callable[[], RLModelRunningModes],
    ) -> None:
        """Initialize the home administration effect adapter and its owned state."""
        self._view = view
        self._services = services
        self._flow = flow
        self._is_critical_phase = is_critical_phase
        self._selected_mode = selected_mode
        self._generation = 0
        self._active = True

    def activate(self) -> None:
        """Activate the operation owned by this home administration effect adapter."""
        if self._active:
            return
        self._generation += 1
        self._active = True

    def close(self) -> None:
        """Close resources owned by this home administration effect adapter."""
        self._generation += 1
        self._active = False

    def invalidate(self) -> None:
        """Invalidate the operation owned by this home administration effect adapter."""
        self._generation += 1

    def _begin(self) -> int:
        """Begin the operation owned by this home administration effect adapter."""
        self._generation += 1
        return self._generation

    def _is_current(self, generation: int, owner: ModelAdministrationOwner) -> bool:
        """Return whether current applies to this home administration effect adapter."""
        return generation == self._generation and self._effects_allowed(owner)

    def _effects_allowed(self, owner: ModelAdministrationOwner) -> bool:
        """Return the effects allowed value owned by this home administration effect adapter."""
        return bool(
            self._active
            and self._view.administration_effects_available()
            and not self._is_critical_phase()
            and self._services.administration.models.is_owner_current(owner)
        )

    def _consume(self, generation: int, owner: ModelAdministrationOwner) -> bool:
        """Consume one modal callback before allowing any nested effect."""

        if generation != self._generation:
            return False
        is_current = self._is_current(generation, owner)
        self._generation += 1
        return is_current

    def open_sessions(self) -> None:
        """Open the sessions workflow owned by this home administration effect adapter."""
        context = self._flow.build_session_selector_context(
            is_critical_phase=self._is_critical_phase(),
        )
        if context.blocked_notification is not None:
            self._view.notify_administration(context.blocked_notification)
            return
        owner = context.owner
        if owner is None:
            return
        generation = self._begin()

        def on_result(result: SessionConfigurationDialogResult | None) -> None:
            """Handle the result event for this home administration effect adapter."""
            if not self._consume(generation, owner):
                return
            self._handle_session_result(owner, result)

        dialog = SessionConfigurationDialog(
            session_options=list(context.session_options),
            selected_session_name=context.current_session_uuid,
            allow_delete=True,
            allow_edit=True,
        )
        if self._is_current(generation, owner):
            self._view.push_session_administration_dialog(dialog, on_result)

    def _handle_session_result(
        self,
        owner: ModelAdministrationOwner,
        result: SessionConfigurationDialogResult | None,
    ) -> None:
        """Handle the session result workflow for this home administration effect adapter."""
        if result is None or not result.session_name:
            return
        session_name = result.session_name
        if result.action == "select":
            success, notification = self._flow.select_session(owner, session_name)
            if not self._effects_allowed(owner):
                return
            if success:
                self._view.refresh_administration_runtime()
                if not self._effects_allowed(owner):
                    return
            self._view.notify_administration(notification)
        elif result.action == "edit" and self._effects_allowed(owner):
            self._view.open_administration_edit(
                session_name=session_name,
                checkpoint_name=None,
                expected_model_uuid=owner.model_uuid,
                reopen_opener=self._view.reopen_session_administration,
            )
        elif result.action == "delete":
            self._request_session_deletion(self._begin(), owner, session_name)

    def _request_session_deletion(
        self,
        generation: int,
        owner: ModelAdministrationOwner,
        session_name: str,
    ) -> None:
        """Request the session deletion owned by this home administration
effect adapter."""
        display_name = self._services.sessions.metadata.get_session_reference_display(
            session_name
        )

        def confirm(confirmed: bool | None) -> None:
            """Confirm the operation owned by this home administration effect adapter."""
            if not self._consume(generation, owner):
                return
            if not confirmed:
                self._view.reopen_session_administration()
                return
            success, notification = self._flow.delete_session(owner, session_name)
            if not self._effects_allowed(owner):
                return
            if success:
                self._view.refresh_administration_runtime()
                if not self._effects_allowed(owner):
                    return
            self._view.notify_administration(notification)
            if not self._effects_allowed(owner):
                return
            self._view.reopen_session_administration()

        dialog = secure_delete_session_dialog(display_name)
        if self._is_current(generation, owner):
            self._view.push_administration_confirmation(dialog, confirm)

    def open_checkpoints(self) -> None:
        """Open the checkpoints workflow owned by this home administration effect adapter."""
        context = self._flow.build_checkpoint_dialog_context(
            is_critical_phase=self._is_critical_phase(),
            selected_running_mode=self._selected_mode(),
        )
        if context.blocked_notification is not None:
            self._view.notify_administration(context.blocked_notification)
            return
        owner = context.owner
        session_name = context.session_uuid
        if owner is None or session_name is None:
            return
        generation = self._begin()

        def on_result(result: CheckpointConfigurationDialogResult | None) -> None:
            """Handle the result event for this home administration effect adapter."""
            if not self._consume(generation, owner) or result is None:
                return
            self._handle_checkpoint_result(owner, session_name, result)

        dialog = CheckpointConfigurationDialog(
            session_name=session_name,
            checkpoint_options=list(context.checkpoint_options),
            selected_checkpoint_name=context.selected_checkpoint_name,
            allow_fresh_training=context.allow_fresh_training,
            fresh_selection_label=context.fresh_selection_label,
            fresh_selection_mode_label=context.fresh_selection_mode_label,
            fresh_selection_search_text=context.fresh_selection_search_text,
            allow_delete=context.allow_delete,
            allow_rename=context.allow_rename,
            allow_edit=context.allow_edit,
        )
        if self._is_current(generation, owner):
            self._view.push_checkpoint_administration_dialog(dialog, on_result)

    def _handle_checkpoint_result(
        self,
        owner: ModelAdministrationOwner,
        session_name: str,
        result: CheckpointConfigurationDialogResult,
    ) -> None:
        """Handle the checkpoint result workflow for this home administration effect adapter."""
        checkpoint_name = result.checkpoint_name
        if result.action == "select":
            self._select_checkpoint(owner, session_name, checkpoint_name)
        elif result.action == "edit" and checkpoint_name:
            self._edit_checkpoint(owner, session_name, checkpoint_name)
        elif result.action == "delete" and checkpoint_name:
            self._request_checkpoint_deletion(
                self._begin(),
                owner,
                session_name,
                checkpoint_name,
            )
        elif (
            result.action == "rename"
            and checkpoint_name
            and result.renamed_checkpoint_name
        ):
            self._rename_checkpoint(
                owner,
                session_name,
                checkpoint_name,
                result.renamed_checkpoint_name,
            )

    def _select_checkpoint(
        self,
        owner: ModelAdministrationOwner,
        session_name: str,
        checkpoint_name: str | None,
    ) -> None:
        """Select the checkpoint owned by this home administration effect adapter."""
        success, notification = self._flow.select_checkpoint(
            owner,
            session_uuid=session_name,
            checkpoint_name=checkpoint_name,
        )
        if not self._effects_allowed(owner):
            return
        if success:
            self._view.refresh_administration_runtime()
            if not self._effects_allowed(owner):
                return
        self._view.notify_administration(notification)

    def _edit_checkpoint(
        self,
        owner: ModelAdministrationOwner,
        session_name: str,
        checkpoint_name: str,
    ) -> None:
        """Handle the edit checkpoint operation owned by this home administration effect
adapter."""
        success, notification = self._flow.select_checkpoint(
            owner,
            session_uuid=session_name,
            checkpoint_name=checkpoint_name,
        )
        if not self._effects_allowed(owner):
            return
        self._view.notify_administration(notification)
        if not success or not self._effects_allowed(owner):
            return
        self._view.refresh_administration_runtime()
        if self._effects_allowed(owner):
            self._view.open_administration_edit(
                session_name=session_name,
                checkpoint_name=checkpoint_name,
                expected_model_uuid=owner.model_uuid,
                reopen_opener=self._view.reopen_checkpoint_administration,
            )

    def _rename_checkpoint(
        self,
        owner: ModelAdministrationOwner,
        session_name: str,
        checkpoint_name: str,
        renamed_checkpoint_name: str,
    ) -> None:
        """Rename the checkpoint owned by this home administration effect adapter."""
        success, notification = self._flow.rename_checkpoint(
            owner,
            session_uuid=session_name,
            checkpoint_name=checkpoint_name,
            renamed_checkpoint_name=renamed_checkpoint_name,
        )
        if not self._effects_allowed(owner):
            return
        if success:
            self._view.refresh_administration_runtime()
            if not self._effects_allowed(owner):
                return
            self._view.reopen_checkpoint_administration()
            if not self._effects_allowed(owner):
                return
        self._view.notify_administration(notification)

    def _request_checkpoint_deletion(
        self,
        generation: int,
        owner: ModelAdministrationOwner,
        session_name: str,
        checkpoint_name: str,
    ) -> None:
        """Request the checkpoint deletion owned by this home administration
effect adapter."""
        def confirm(confirmed: bool | None) -> None:
            """Confirm the operation owned by this home administration effect adapter."""
            if not self._consume(generation, owner):
                return
            if not confirmed:
                self._view.reopen_checkpoint_administration()
                return
            success, notification = self._flow.delete_checkpoint(
                owner,
                session_uuid=session_name,
                checkpoint_name=checkpoint_name,
            )
            if not self._effects_allowed(owner):
                return
            if success:
                self._view.refresh_administration_runtime()
                if not self._effects_allowed(owner):
                    return
            self._view.notify_administration(notification)
            if not self._effects_allowed(owner):
                return
            self._view.reopen_checkpoint_administration()

        dialog = secure_delete_checkpoint_dialog(checkpoint_name, session_name)
        if self._is_current(generation, owner):
            self._view.push_administration_confirmation(dialog, confirm)
