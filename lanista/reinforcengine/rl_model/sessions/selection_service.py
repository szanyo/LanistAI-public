# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate training-session selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class SessionSelectionDetail:
    """Carry the data required for session selection detail."""

    session_uuid: str
    display_name: str
    framework_uuid: str
    parent_display_name: str
    fork_source_session_display_name: str
    fork_source_checkpoint_name: str | None
    base_step_offset: int
    child_count: int
    checkpoint_count: int
    latest_checkpoint_name: str | None
    created_at_raw: str | None
    modified_at_raw: str | None
    description: str


class SessionSelectionService:
    """Coordinate session selection operations."""

    def __init__(
        self,
        *,
        session_configuration_loader: Callable[[str], object | None],
        session_reference_display_provider: Callable[[str], str],
        session_identity_uuid_provider: Callable[[str], str | None],
        session_display_name_without_uuid_provider: Callable[[str | None], str | None],
        session_parent_uuid_provider: Callable[[str | None], str | None],
        session_created_at_provider: Callable[[str | None], str | None],
        session_last_modified_at_provider: Callable[[str | None], str | None],
        session_description_provider: Callable[[str | None], str],
        session_child_uuids_provider: Callable[[str | None], list[str]],
        session_framework_uuid_provider: Callable[[str], str],
        sessions_provider: Callable[[str | None], list[str]],
        checkpoint_names_provider: Callable[[str, str | None], list[str]],
    ) -> None:
        """Initialize the session selection service and its owned state."""
        self._session_configuration_loader = session_configuration_loader
        self._session_reference_display_provider = session_reference_display_provider
        self._session_identity_uuid_provider = session_identity_uuid_provider
        self._session_display_name_without_uuid_provider = session_display_name_without_uuid_provider
        self._session_parent_uuid_provider = session_parent_uuid_provider
        self._session_created_at_provider = session_created_at_provider
        self._session_last_modified_at_provider = session_last_modified_at_provider
        self._session_description_provider = session_description_provider
        self._session_child_uuids_provider = session_child_uuids_provider
        self._session_framework_uuid_provider = session_framework_uuid_provider
        self._sessions_provider = sessions_provider
        self._checkpoint_names_provider = checkpoint_names_provider

    def build_session_selection_detail(self, session_uuid: str | None) -> SessionSelectionDetail | None:
        """Build the session selection detail required by this session selection service."""
        if not session_uuid:
            return None

        session_configuration = self._session_configuration_loader(session_uuid)
        if session_configuration is None:
            return None

        framework_uuid = self._session_framework_uuid_provider(session_uuid) or "-"
        parent_uuid = self._session_parent_uuid_provider(session_uuid)
        checkpoints = self._checkpoint_names_provider(session_uuid, framework_uuid)
        identity_uuid = self._session_identity_uuid_provider(session_uuid) or session_uuid
        source_session_uuid = str(getattr(session_configuration, "load_checkpoint_from_session", "") or "").strip() or None
        source_checkpoint_name = str(getattr(session_configuration, "load_checkpoint_name", "") or "").strip() or None
        try:
            base_step_offset = int(getattr(session_configuration, "base_step_offset", 0) or 0)
        except (TypeError, ValueError):
            base_step_offset = 0

        return SessionSelectionDetail(
            session_uuid=identity_uuid,
            display_name=self._session_display_name_without_uuid_provider(session_uuid) or session_uuid,
            framework_uuid=framework_uuid,
            parent_display_name=self._session_reference_display_provider(parent_uuid) if parent_uuid else "-",
            fork_source_session_display_name=(
                self._session_reference_display_provider(source_session_uuid) if source_session_uuid else "-"
            ),
            fork_source_checkpoint_name=source_checkpoint_name,
            base_step_offset=max(0, base_step_offset),
            child_count=len(self._session_child_uuids_provider(session_uuid)),
            checkpoint_count=len(checkpoints),
            latest_checkpoint_name=checkpoints[0] if checkpoints else None,
            created_at_raw=self._session_created_at_provider(session_uuid),
            modified_at_raw=self._session_last_modified_at_provider(session_uuid),
            description=self._session_description_provider(session_uuid),
        )

    def list_session_options_for_current_model(self, framework_uuid: str | None = None) -> list[tuple[str, str]]:
        """Return the session options for current model available to this session selection
        service.
        """
        return [
            (self._session_reference_display_provider(session_uuid), session_uuid)
            for session_uuid in self._sessions_provider(framework_uuid)
        ]

    def list_checkpoint_options_for_current_model(
        self,
        session_name: str | None,
        framework_uuid: str | None = None,
    ) -> list[tuple[str, str]]:
        """Return the checkpoint options for current model available to this session selection
        service.
        """
        if not session_name:
            return []
        return [
            (checkpoint_name, checkpoint_name)
            for checkpoint_name in self._checkpoint_names_provider(session_name, framework_uuid)
        ]
