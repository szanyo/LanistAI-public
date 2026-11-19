# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Training-session checkpoint queries assembled from focused providers."""

from __future__ import annotations

from collections.abc import Callable

from lanista.reinforcengine.rl_model.checkpoints.catalog_contracts import CheckpointTreeEntry
from lanista.reinforcengine.rl_model.runtime.status_service import RuntimeCapabilities


class SessionCheckpointCatalogService:
    """Coordinate session-level checkpoint queries and runtime capabilities."""

    def __init__(
        self,
        *,
        session_status_provider: Callable[[str | None], object | None],
        sessions_provider: Callable[[str | None], list[str]],
        checkpoint_names_provider: Callable[[str, str | None], list[str]],
        checkpoint_tree_entries_provider: Callable[[str, str | None], list[CheckpointTreeEntry]],
        latest_session_provider: Callable[[], str | None],
        latest_checkpoint_provider: Callable[[str], str | None],
        current_model_can_evaluate_provider: Callable[[str | None], bool],
        current_model_has_training_session_provider: Callable[[], bool],
        session_has_checkpoints_provider: Callable[[str | None, str | None], bool],
    ) -> None:
        """Initialize the session checkpoint catalog service and its owned state."""
        self._session_status_provider = session_status_provider
        self._sessions_provider = sessions_provider
        self._checkpoint_names_provider = checkpoint_names_provider
        self._checkpoint_tree_entries_provider = checkpoint_tree_entries_provider
        self._latest_session_provider = latest_session_provider
        self._latest_checkpoint_provider = latest_checkpoint_provider
        self._current_model_can_evaluate_provider = current_model_can_evaluate_provider
        self._current_model_has_training_session_provider = current_model_has_training_session_provider
        self._session_has_checkpoints_provider = session_has_checkpoints_provider

    def session_is_closed_for_training(self, session_name: str | None) -> bool:
        """Return whether the session reached a terminal training state."""

        status = self._session_status_provider(session_name)
        return bool(getattr(status, "value", status) in {"completed", "failed", "stopped"})

    def list_sessions_for_current_model(self, framework_uuid: str | None = None) -> list[str]:
        """List current-model sessions, optionally limited to one framework."""

        return self._sessions_provider(framework_uuid)

    def list_checkpoints_for_session(self, session_name: str) -> list[str]:
        """List all checkpoint names for a session."""

        return self._checkpoint_names_provider(session_name, None)

    def list_checkpoints_for_session_for_framework(
        self,
        session_name: str,
        framework_uuid: str | None = None,
    ) -> list[str]:
        """List checkpoint names compatible with the requested framework."""

        return self._checkpoint_names_provider(session_name, framework_uuid)

    def build_checkpoint_tree_entries_for_session(
        self,
        session_name: str | None,
        framework_uuid: str | None = None,
    ) -> list[CheckpointTreeEntry]:
        """Build searchable checkpoint rows for a session-selection view."""

        if not session_name:
            return []
        return self._checkpoint_tree_entries_provider(session_name, framework_uuid)

    def get_latest_session_for_current_model(self) -> str | None:
        """Return the newest session selected by the runtime status policy."""

        return self._latest_session_provider()

    def get_latest_checkpoint_for_session(self, session_name: str) -> str | None:
        """Return the newest checkpoint selected by the runtime status policy."""

        return self._latest_checkpoint_provider(session_name)

    def session_has_checkpoints(
        self,
        session_name: str | None,
        framework_uuid: str | None = None,
    ) -> bool:
        """Report whether a session owns a compatible checkpoint artifact."""

        return self._session_has_checkpoints_provider(session_name, framework_uuid)

    def current_model_can_evaluate(self, framework_uuid: str | None = None) -> bool:
        """Report whether the current model can evaluate in the framework."""

        return self._current_model_can_evaluate_provider(framework_uuid)

    def current_model_has_training_session(self) -> bool:
        """Report whether the current model has at least one training session."""

        return self._current_model_has_training_session_provider()

    def get_current_model_runtime_capabilities(self) -> RuntimeCapabilities:
        """Return the training/evaluation capability snapshot used by the UI."""

        return RuntimeCapabilities(
            has_training_session=self.current_model_has_training_session(),
            has_evaluation_runtime=self.current_model_can_evaluate(),
        )
