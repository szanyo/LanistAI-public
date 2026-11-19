# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Prepare presentation data for control panel operation."""

from __future__ import annotations

from dataclasses import dataclass, replace

from lanista.gui.contracts import UiNotification
from lanista.reinforcengine.rl_model.execution_contracts import (
    ExecutionSnapshot,
    RLModelState,
)
from lanista.reinforcengine.rl_model.model import RLModelRunningModes


@dataclass(frozen=True)
class ControlPanelButtonPresentation:
    """Prepare presentation data for control panel button."""

    display: bool
    disabled: bool = False
    label: str | None = None


@dataclass(frozen=True)
class ControlPanelButtonsPresentation:
    """Prepare presentation data for control panel buttons."""

    container: ControlPanelButtonPresentation
    start: ControlPanelButtonPresentation
    resume: ControlPanelButtonPresentation
    pause: ControlPanelButtonPresentation
    stop: ControlPanelButtonPresentation
    finish_inspection: ControlPanelButtonPresentation
    run_again: ControlPanelButtonPresentation
    change_configuration: ControlPanelButtonPresentation


@dataclass(frozen=True)
class ControlPanelStartContext:
    """Carry the runtime context for control panel start."""

    selected_session: str | None
    selected_checkpoint: str | None
    blocked_notification: UiNotification | None = None

    @property
    def blocked(self) -> bool:
        """Return the blocked exposed by this control panel start context."""
        return self.blocked_notification is not None


class ControlPanelOperationPresenter:
    """Presentation decisions for the ControlPanel operation controls."""

    @staticmethod
    def build_buttons(
        *,
        has_valid_model: bool,
        actions_locked: bool,
        inspection_running: bool,
        current_mode: RLModelRunningModes,
        execution_snapshot: ExecutionSnapshot,
        selected_session: str | None,
        selected_session_has_checkpoints: bool,
        selected_session_closed_for_training: bool,
        quick_start_display: bool,
        quick_start_label: str,
        native_gui_restart_required: bool = False,
    ) -> ControlPanelButtonsPresentation:
        """Build the buttons required by this control panel operation presenter."""
        hidden_buttons = ControlPanelOperationPresenter._build_hidden_buttons(quick_start_label)

        if inspection_running:
            return ControlPanelOperationPresenter._build_inspection_buttons(
                hidden_buttons,
                actions_locked=actions_locked,
            )

        if actions_locked:
            return hidden_buttons

        if (
            native_gui_restart_required
            and execution_snapshot.state
            in {RLModelState.COMPLETED, RLModelState.STOPPED, RLModelState.FAILED}
            and not execution_snapshot.worker_alive
        ):
            return replace(
                hidden_buttons,
                container=ControlPanelButtonPresentation(display=True),
                run_again=ControlPanelButtonPresentation(
                    display=True,
                    label="Run again",
                ),
                change_configuration=ControlPanelButtonPresentation(
                    display=True,
                    label="Change configuration",
                ),
            )

        lifecycle_presentation = ControlPanelOperationPresenter._build_lifecycle_buttons(
            execution_snapshot=execution_snapshot,
            hidden_buttons=hidden_buttons,
        )
        if lifecycle_presentation is not None:
            return lifecycle_presentation

        if not has_valid_model or execution_snapshot.state == RLModelState.LOADING:
            return hidden_buttons

        quick_start_allowed = ControlPanelOperationPresenter._quick_start_allowed(
            current_mode=current_mode,
            selected_session=selected_session,
            selected_session_has_checkpoints=selected_session_has_checkpoints,
            selected_session_closed_for_training=selected_session_closed_for_training,
            quick_start_display=quick_start_display,
        )
        if not quick_start_allowed:
            return hidden_buttons

        return replace(
            hidden_buttons,
            container=ControlPanelButtonPresentation(display=True),
            start=replace(
                hidden_buttons.start,
                display=True,
            ),
        )

    @staticmethod
    def _build_hidden_buttons(
        quick_start_label: str,
    ) -> ControlPanelButtonsPresentation:
        """Build the hidden buttons required by this control panel operation presenter."""
        hidden = ControlPanelButtonPresentation(display=False)
        return ControlPanelButtonsPresentation(
            container=hidden,
            start=ControlPanelButtonPresentation(
                display=False,
                label=quick_start_label,
            ),
            resume=hidden,
            pause=hidden,
            stop=hidden,
            finish_inspection=hidden,
            run_again=hidden,
            change_configuration=hidden,
        )

    @staticmethod
    def _build_inspection_buttons(
        hidden_buttons: ControlPanelButtonsPresentation,
        *,
        actions_locked: bool,
    ) -> ControlPanelButtonsPresentation:
        """Build the inspection buttons required by this control panel operation presenter."""
        return replace(
            hidden_buttons,
            container=ControlPanelButtonPresentation(display=True),
            finish_inspection=ControlPanelButtonPresentation(
                display=True,
                disabled=actions_locked,
            ),
        )

    @staticmethod
    def _quick_start_allowed(
        *,
        current_mode: RLModelRunningModes,
        selected_session: str | None,
        selected_session_has_checkpoints: bool,
        selected_session_closed_for_training: bool,
        quick_start_display: bool,
    ) -> bool:
        """Return the quick start allowed value owned by this control panel operation presenter."""
        if not quick_start_display:
            return False
        if current_mode != RLModelRunningModes.TRAINING:
            return True
        return bool(selected_session) and not (
            selected_session_has_checkpoints or selected_session_closed_for_training
        )

    @staticmethod
    def _build_lifecycle_buttons(
        *,
        execution_snapshot: ExecutionSnapshot,
        hidden_buttons: ControlPanelButtonsPresentation,
    ) -> ControlPanelButtonsPresentation | None:
        """Return the explicit button matrix for an owned execution run."""

        enabled_control = ControlPanelButtonPresentation(display=True)
        disabled_control = ControlPanelButtonPresentation(display=True, disabled=True)
        state_controls = {
            RLModelState.STARTING: (
                hidden_buttons.resume,
                hidden_buttons.pause,
                disabled_control,
            ),
            RLModelState.RUNNING: (
                hidden_buttons.resume,
                enabled_control,
                enabled_control,
            ),
            RLModelState.PAUSED: (
                enabled_control,
                hidden_buttons.pause,
                enabled_control,
            ),
            RLModelState.STOPPING: (
                hidden_buttons.resume,
                hidden_buttons.pause,
                disabled_control,
            ),
        }
        controls = state_controls.get(execution_snapshot.state)
        if controls is None:
            if not execution_snapshot.worker_alive:
                return None
            controls = (
                hidden_buttons.resume,
                hidden_buttons.pause,
                disabled_control,
            )

        resume, pause, stop = controls
        return replace(
            hidden_buttons,
            container=ControlPanelButtonPresentation(display=True),
            resume=resume,
            pause=pause,
            stop=stop,
        )

    @staticmethod
    def build_start_context(
        *,
        current_mode: RLModelRunningModes,
        selected_session: str | None,
        selected_checkpoint: str | None,
        runtime_session: str | None,
        runtime_checkpoint: str | None,
    ) -> ControlPanelStartContext:
        """Build the start context required by this control panel operation presenter."""
        effective_selected_session = selected_session or runtime_session
        effective_selected_checkpoint = selected_checkpoint or runtime_checkpoint

        return ControlPanelStartContext(
            selected_session=effective_selected_session,
            selected_checkpoint=effective_selected_checkpoint,
        )
