# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Own non-critical ControlPanel execution metadata persistence effects."""

from __future__ import annotations

import logging
from typing import Protocol

from lanista.gui.contracts import UiSeverity
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.execution_contracts import RLModelState
from lanista.reinforcengine.rl_model.runtime.active_selection_service import (
    ActiveRuntimeSelectionRequest,
)


class ControlPanelExecutionMetadataViewPort(Protocol):
    """Presentation-only view surface for metadata persistence outcomes."""

    def notify_execution(
        self,
        message: str,
        *,
        title: str,
        severity: UiSeverity,
    ) -> None:
        """Notify the caller about the execution outcome."""
        ...


class ControlPanelExecutionMetadataEffectAdapter:
    """Persist execution metadata without making the widget an application owner."""

    def __init__(
        self,
        *,
        view: ControlPanelExecutionMetadataViewPort,
        services: ApplicationServices,
        logger: logging.Logger,
    ) -> None:
        """Initialize the control panel execution metadata effect adapter and its owned state."""
        self._view = view
        self._services = services
        self._logger = logger

    def persist_session_status(
        self,
        session_name: str | None,
        status: RLModelState,
        *,
        action_summary: str,
    ) -> bool:
        """Persist display metadata without disrupting the runtime transition."""

        status_saved, degraded = self._attempt_session_status_persistence(
            session_name,
            status,
            action_summary=action_summary,
        )
        if not status_saved:
            self._view.notify_execution(
                f"{action_summary}, but its session status could not be saved.",
                title="Session metadata warning",
                severity="warning",
            )
        elif degraded:
            self._view.notify_execution(
                f"{action_summary}; session metadata was saved but its summary "
                "needs a reload.",
                title="Session metadata warning",
                severity="warning",
            )
        return status_saved

    def _attempt_session_status_persistence(
        self,
        session_name: str | None,
        status: RLModelState,
        *,
        action_summary: str,
    ) -> tuple[bool, bool]:
        """Return the attempt session status persistence value owned by this control panel
        execution metadata effect adapter.
        """
        try:
            result = self._services.sessions.repository.set_session_status(
                session_name,
                status,
            )
        except Exception:
            self._logger.exception(
                "%s, but session status %s could not be persisted for %r.",
                action_summary,
                status.value,
                session_name,
            )
            status_saved = False
            degraded = False
        else:
            outcome = result.value if result.succeeded else None
            status_saved = outcome is not None and outcome.committed
            degraded = outcome.degraded if outcome is not None else False
            if not status_saved:
                self._logger.warning(
                    "%s, but session status %s was not persisted for %r "
                    "(operation_id=%s, failure=%s).",
                    action_summary,
                    status.value,
                    session_name,
                    result.operation_id,
                    (
                        result.failure.code.value
                        if result.failure is not None
                        else "unknown"
                    ),
                )
            elif degraded:
                self._logger.warning(
                    "%s and session status %s was committed for %r, but the "
                    "session summary cache was not invalidated "
                    "(operation_id=%s).",
                    action_summary,
                    status.value,
                    session_name,
                    result.operation_id,
                )

        return bool(status_saved), bool(degraded)

    def persist_cleared_selection(self, *, action_summary: str) -> None:
        """Persist the cleared selection as non-critical execution metadata."""

        try:
            result = self._services.runtime.active_selection.set_selection(
                ActiveRuntimeSelectionRequest(
                    session_name=None,
                    checkpoint_name=None,
                    persist=True,
                )
            )
        except Exception:
            self._logger.exception(
                "%s, but the cleared runtime selection could not be persisted.",
                action_summary,
            )
            self._notify_cleared_selection_failure(action_summary)
            return
        if result.succeeded and result.value is not None:
            return
        self._logger.error(
            "%s, but the cleared runtime selection could not be persisted; "
            "operation_id=%s; code=%s.",
            action_summary,
            result.operation_id,
            (
                result.failure.code.value
                if result.failure is not None
                else "missing_result"
            ),
        )
        self._notify_cleared_selection_failure(action_summary)

    def promote_latest_training_checkpoint_selection(
        self,
        session_name: str | None,
        *,
        action_summary: str,
    ) -> str | None:
        """Select the checkpoint written by a naturally completed Training run."""

        normalized_session = str(session_name or "").strip()
        if not normalized_session:
            self._logger.warning(
                "%s, but no originating session was available for checkpoint selection.",
                action_summary,
            )
            self._notify_checkpoint_promotion_failure(action_summary)
            return None
        try:
            checkpoint_name = (
                self._services.catalog.session_checkpoints.get_latest_checkpoint_for_session(
                    normalized_session
                )
            )
            if not checkpoint_name:
                self._logger.warning(
                    "%s, but session %r contains no checkpoint to select.",
                    action_summary,
                    normalized_session,
                )
                self._notify_checkpoint_promotion_failure(action_summary)
                return None
            result = self._services.runtime.active_selection.set_selection(
                ActiveRuntimeSelectionRequest(
                    session_name=normalized_session,
                    checkpoint_name=checkpoint_name,
                    persist=True,
                )
            )
        except Exception:
            self._logger.exception(
                "%s, but the latest checkpoint selection could not be reconciled for %r.",
                action_summary,
                normalized_session,
            )
            self._notify_checkpoint_promotion_failure(action_summary)
            return None
        if result.succeeded and result.value is not None:
            return checkpoint_name
        self._logger.error(
            "%s, but the latest checkpoint could not be selected; "
            "operation_id=%s; code=%s.",
            action_summary,
            result.operation_id,
            (
                result.failure.code.value
                if result.failure is not None
                else "missing_result"
            ),
        )
        self._notify_checkpoint_promotion_failure(action_summary)
        return None

    def _notify_cleared_selection_failure(self, action_summary: str) -> None:
        """Notify the caller about the cleared selection failure outcome."""
        self._view.notify_execution(
            f"{action_summary}, but its cleared runtime selection could not be saved.",
            title="Session metadata warning",
            severity="warning",
        )

    def _notify_checkpoint_promotion_failure(self, action_summary: str) -> None:
        """Warn when a completed Training run cannot hand off its checkpoint."""

        self._view.notify_execution(
            f"{action_summary}, but its latest checkpoint could not be selected "
            "automatically. Choose a checkpoint in Configure before Evaluation.",
            title="Checkpoint selection warning",
            severity="warning",
        )
