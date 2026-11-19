# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate runtime context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from lanista.foundation.configuration import Configuration
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest


@dataclass
class RuntimeSelectionState:
    """Represent runtime state for runtime selection."""

    session_name: str | None
    checkpoint_name: str | None
    has_training_session: bool
    has_evaluation_runtime: bool


@dataclass
class RuntimeContextPayload:
    """Carry the data required for runtime context payload."""

    session_name: str | None
    parent_session_name: str | None
    checkpoint_name: str | None
    session_created_at_raw: str | None


class RuntimeContextService:
    """Coordinate runtime context operations."""

    def __init__(
        self,
        *,
        active_manifest_provider: Callable[[], ModelManifest | None],
        current_framework_provider: Callable[[], str | None],
        selected_session_provider: Callable[[], str | None],
        selected_checkpoint_provider: Callable[[], str | None],
        sessions_provider: Callable[[str | None], list[str]],
        checkpoints_provider: Callable[[str, str | None], list[str]],
        current_model_valid_runtime_provider: Callable[[str | None], bool],
        session_display_name_provider: Callable[[str | None], str | None],
        session_parent_uuid_provider: Callable[[str | None], str | None],
        session_created_at_provider: Callable[[str | None], str | None],
    ) -> None:
        """Initialize the runtime context service and its owned state."""
        self._active_manifest_provider = active_manifest_provider
        self._current_framework_provider = current_framework_provider
        self._selected_session_provider = selected_session_provider
        self._selected_checkpoint_provider = selected_checkpoint_provider
        self._sessions_provider = sessions_provider
        self._checkpoints_provider = checkpoints_provider
        self._current_model_valid_runtime_provider = current_model_valid_runtime_provider
        self._session_display_name_provider = session_display_name_provider
        self._session_parent_uuid_provider = session_parent_uuid_provider
        self._session_created_at_provider = session_created_at_provider

    def get_current_framework_uuid(self) -> str | None:
        """Return the framework selected by the active runtime."""

        return self._current_framework_provider()

    def get_selected_session_candidate(self) -> str | None:
        """Return the unvalidated in-memory session selected by the runtime."""

        return self._selected_session_provider()

    def get_active_configuration(self) -> Configuration | None:
        """Return the active model configuration without exposing its owner."""

        active_manifest = self._active_manifest_provider()
        return (
            active_manifest.get_configuration()
            if active_manifest is not None
            else None
        )

    def get_current_runtime_selection_state(self) -> RuntimeSelectionState:
        """Return the current runtime selection state exposed by this runtime context service."""
        if self._active_manifest_provider() is None:
            return RuntimeSelectionState(
                session_name=None,
                checkpoint_name=None,
                has_training_session=False,
                has_evaluation_runtime=False,
            )

        session_name = self._selected_session_provider()
        checkpoint_name = self._selected_checkpoint_provider()
        framework_uuid = self._current_framework_provider()

        sessions = self._sessions_provider(framework_uuid)
        valid_session_values = set(sessions)
        if session_name not in valid_session_values:
            session_name = None
            checkpoint_name = None

        if session_name is not None:
            checkpoints = self._checkpoints_provider(session_name, framework_uuid)
            if checkpoint_name not in set(checkpoints):
                checkpoint_name = None

        return RuntimeSelectionState(
            session_name=session_name,
            checkpoint_name=checkpoint_name,
            has_training_session=bool(sessions),
            has_evaluation_runtime=self._current_model_valid_runtime_provider(framework_uuid),
        )

    def is_current_runtime_selection_applied(
        self,
        *,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> bool:
        """Return whether current runtime selection applied applies to this runtime context
        service.
        """
        state = self.get_current_runtime_selection_state()
        normalized_session_name = session_name or None
        normalized_checkpoint_name = checkpoint_name or None
        return (
            state.session_name == normalized_session_name
            and state.checkpoint_name == normalized_checkpoint_name
        )

    def build_runtime_context_payload(
        self,
        *,
        selected_session: str | None,
        selected_checkpoint: str | None,
    ) -> RuntimeContextPayload:
        """Build the runtime context payload required by this runtime context service."""
        session_created_at_raw = self._session_created_at_provider(selected_session)
        session_display_name = self._session_display_name_provider(selected_session) or selected_session
        parent_session_uuid = self._session_parent_uuid_provider(selected_session)
        parent_session_display_name = (
            self._session_display_name_provider(parent_session_uuid) if parent_session_uuid else None
        )

        return RuntimeContextPayload(
            session_name=session_display_name,
            parent_session_name=parent_session_display_name,
            checkpoint_name=selected_checkpoint,
            session_created_at_raw=session_created_at_raw,
        )
