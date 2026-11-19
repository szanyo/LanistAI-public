# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate operation presentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OperationButtonsPresentation:
    """Prepare presentation data for operation buttons."""

    restart_display: bool
    pause_display: bool
    stop_display: bool
    container_display: bool
    management_disabled: bool


@dataclass
class OperationFeedback:
    """Carry the data required for operation feedback."""

    message: str
    title: str
    severity: str


@dataclass
class OperationTerminalStatePresentation:
    """Prepare presentation data for operation terminal state."""

    activity: str
    step: str
    session_status: object


@dataclass
class OperationLiveStatePresentation:
    """Prepare presentation data for operation live state."""

    activity: str
    step: str


@dataclass
class QuickStartPresentation:
    """Prepare presentation data for quick start."""

    display: bool
    label: str


class OperationPresentationService:
    """Coordinate operation presentation operations."""

    @staticmethod
    def _state_value(state: object) -> str:
        """Return the state value value owned by this operation presentation service."""
        return str(getattr(state, "value", state) or "").strip().lower()

    @classmethod
    def get_operation_buttons_presentation(
        cls,
        panel_state: object,
        has_current_model: bool,
        has_previous_models: bool,
    ) -> OperationButtonsPresentation:
        """Return the operation buttons presentation exposed by this operation presentation
        service.
        """
        restart_display = panel_state in ("paused", "failed") or cls._state_value(panel_state) in (
            "paused",
            "failed",
        )
        pause_display = panel_state == "running" or cls._state_value(panel_state) == "running"
        stop_display = panel_state in ("running", "paused") or cls._state_value(panel_state) in (
            "running",
            "paused",
        )

        if not has_current_model and not has_previous_models:
            restart_display = False
            pause_display = False
            stop_display = False

        return OperationButtonsPresentation(
            restart_display=restart_display,
            pause_display=pause_display,
            stop_display=stop_display,
            container_display=(restart_display or pause_display or stop_display),
            management_disabled=(
                panel_state in ("running", "paused")
                or cls._state_value(panel_state) in ("running", "paused")
            ),
        )

    @staticmethod
    def get_operation_feedback(action: str) -> OperationFeedback:
        """Return the operation feedback exposed by this operation presentation service."""
        feedback_by_action: dict[str, OperationFeedback] = {
            "started": OperationFeedback("Progress started.", "Progress", "information"),
            "resumed": OperationFeedback("Progress resumed.", "Progress", "information"),
            "paused": OperationFeedback("Progress paused.", "Progress", "information"),
            "reset": OperationFeedback("Progress has been reset.", "Progress", "information"),
        }
        return feedback_by_action.get(
            str(action or ""),
            OperationFeedback("Operation updated.", "Progress", "information"),
        )

    @classmethod
    def get_operation_terminal_state_presentation(
        cls,
        status: object,
        additional_message: str = "",
    ) -> OperationTerminalStatePresentation:
        """Return the operation terminal state presentation exposed by this operation presentation
        service.
        """
        status_value = cls._state_value(status)
        step_text = additional_message or "-"

        if status_value in ("completed", "failed"):
            return OperationTerminalStatePresentation(
                activity=status_value,
                step=step_text,
                session_status=status,
            )

        return OperationTerminalStatePresentation(
            activity=status_value,
            step=step_text,
            session_status=status,
        )

    @classmethod
    def get_operation_live_state_presentation(
        cls,
        panel_state: object,
        additional_message: str | None = None,
    ) -> OperationLiveStatePresentation:
        """Return the operation live state presentation exposed by this operation presentation
        service.
        """
        return OperationLiveStatePresentation(
            activity=cls._state_value(panel_state),
            step=additional_message or "-",
        )

    @classmethod
    def get_quick_start_presentation(
        cls,
        mode: object,
        panel_state: object,
        has_current_model: bool,
        can_evaluate: bool,
        can_infer: bool,
        has_training_session: bool,
        inspect_is_running: bool,
    ) -> QuickStartPresentation:
        """Return the quick start presentation exposed by this operation presentation service."""
        _ = inspect_is_running
        if cls._state_value(panel_state) == "paused":
            return QuickStartPresentation(display=has_current_model, label="Resume")

        if not has_current_model:
            return QuickStartPresentation(display=False, label="Start")

        if cls._state_value(mode) == "inference":
            return QuickStartPresentation(
                display=can_infer,
                label="Start Inference",
            )

        if cls._state_value(mode) == "evaluation":
            return QuickStartPresentation(
                display=can_evaluate,
                label="Start Evaluation",
            )

        return QuickStartPresentation(
            display=has_training_session,
            label="Start Training",
        )
