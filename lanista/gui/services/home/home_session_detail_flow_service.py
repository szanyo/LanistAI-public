# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate home session detail flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lanista.gui.contracts import UiNotification
from lanista.reinforcengine.rl_model.metadata.timestamps import utc_now_iso
from lanista.reinforcengine.rl_model.sessions.metadata_service import SessionMetadataService
from lanista.reinforcengine.rl_model.sessions.repository_contracts import SessionMetadataUpdate
from lanista.reinforcengine.rl_model.sessions.repository_service import SessionRepositoryService
from lanista.reinforcengine.rl_model.sessions.selection_service import SessionSelectionService


class HomeSessionServices(Protocol):
    """Define the typed boundary for home session services."""

    @property
    def selection(self) -> SessionSelectionService:
        """Return the selection exposed by this home session services."""
        ...

    @property
    def repository(self) -> SessionRepositoryService:
        """Return the repository exposed by this home session services."""
        ...

    @property
    def metadata(self) -> SessionMetadataService:
        """Return the metadata exposed by this home session services."""
        ...


@dataclass(frozen=True)
class HomeSessionDetailContext:
    """Carry the runtime context for home session detail."""

    blocked_notification: UiNotification | None
    session_uuid: str | None = None

    @property
    def blocked(self) -> bool:
        """Return the blocked exposed by this home session detail context."""
        return self.blocked_notification is not None


@dataclass(frozen=True)
class HomeSessionDetailSaveResult:
    """Report the result of home session detail save."""

    success: bool
    notification: UiNotification


class HomeSessionDetailFlowService:
    """Decision and persistence helpers for the session-only edit flow."""

    def __init__(self, sessions: HomeSessionServices) -> None:
        """Initialize the home session detail flow service and its owned state."""
        self._sessions = sessions

    def build_context(
        self,
        *,
        is_critical_phase: bool,
        session_uuid: str | None,
    ) -> HomeSessionDetailContext:
        """Build the context required by this home session detail flow service."""
        if is_critical_phase:
            return HomeSessionDetailContext(
                blocked_notification=UiNotification(
                    message="Edit is disabled while training is running.",
                    title="Edit blocked",
                    severity="warning",
                )
            )

        if not session_uuid:
            return HomeSessionDetailContext(
                blocked_notification=UiNotification(
                    message="Select a session before editing metadata.",
                    title="Action blocked",
                    severity="warning",
                )
            )

        if self._sessions.selection.build_session_selection_detail(session_uuid) is None:
            return HomeSessionDetailContext(
                blocked_notification=UiNotification(
                    message="The selected session could not be loaded.",
                    title="Action blocked",
                    severity="warning",
                ),
                session_uuid=session_uuid,
            )

        return HomeSessionDetailContext(
            blocked_notification=None,
            session_uuid=session_uuid,
        )

    def save_session_details(
        self,
        *,
        session_uuid: str | None,
        session_name: str | None,
        session_description: str | None,
    ) -> HomeSessionDetailSaveResult:
        """Persist the session details owned by this home session detail flow service."""
        result = self._sessions.repository.update_session_metadata(
            session_uuid,
            SessionMetadataUpdate.details(
                display_name=session_name,
                description=session_description,
                updated_at=utc_now_iso(),
            ),
        )
        outcome = result.value if result.succeeded else None
        if outcome is None or not outcome.committed:
            message = (
                result.failure.message
                if result.failure is not None
                else "Failed to save session metadata changes."
            )
            return HomeSessionDetailSaveResult(
                success=False,
                notification=UiNotification(
                    message=message,
                    title="Save failed",
                    severity="error",
                ),
            )

        if outcome.degraded:
            return HomeSessionDetailSaveResult(
                success=True,
                notification=UiNotification(
                    message=(
                        "Session metadata was saved, but cached session views "
                        "could not be refreshed. Reload the model before relying "
                        "on the displayed summary."
                    ),
                    title="Saved with warning",
                    severity="warning",
                ),
            )

        return HomeSessionDetailSaveResult(
            success=True,
            notification=UiNotification(
                message=(
                    "Session metadata saved: "
                    f"{self._sessions.metadata.get_session_reference_display(session_uuid)}"
                ),
                title="Saved",
                severity="information",
            ),
        )
