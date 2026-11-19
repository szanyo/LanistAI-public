# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Read-only application boundary for checkpoint catalog queries."""

from __future__ import annotations

from collections.abc import Callable

from lanista.reinforcengine.rl_model.checkpoints.catalog_contracts import (
    CheckpointSelectionDetail,
    CheckpointTreeEntry,
)
from lanista.reinforcengine.rl_model.checkpoints.catalog_projection import (
    CheckpointCatalogProjector,
)
from lanista.reinforcengine.rl_model.checkpoints.catalog_repository import (
    CheckpointCatalogRepository,
)


class CheckpointCatalogReader:
    """Expose catalog queries without checkpoint mutation capabilities."""

    __slots__ = (
        "_projector",
        "_repository",
        "_session_framework_provider",
        "_session_reference_display_provider",
    )

    def __init__(
        self,
        *,
        repository: CheckpointCatalogRepository,
        projector: CheckpointCatalogProjector,
        session_reference_display_provider: Callable[[str | None], str],
        session_framework_provider: Callable[[str], str],
    ) -> None:
        """Initialize the checkpoint catalog reader and its owned state."""
        self._repository = repository
        self._projector = projector
        self._session_reference_display_provider = (
            session_reference_display_provider
        )
        self._session_framework_provider = session_framework_provider

    def list_for_session(
        self,
        session_name: str,
        framework_uuid: str | None = None,
    ) -> list[str]:
        """List matching checkpoint artifacts in stable newest-first order."""

        return self._repository.list_for_session(session_name, framework_uuid)

    def has_checkpoints(
        self,
        session_name: str | None,
        framework_uuid: str | None = None,
    ) -> bool:
        """Return whether the selected session contains a matching artifact."""

        return self._repository.has_checkpoints(session_name, framework_uuid)

    def build_selection_detail(
        self,
        session_uuid: str | None,
        checkpoint_name: str | None,
    ) -> CheckpointSelectionDetail:
        """Return immutable presentation data for the selected checkpoint."""

        projector = self._projector
        session_display_name = (
            self._session_reference_display_provider(session_uuid)
            if session_uuid
            else "-"
        )
        framework_uuid = (
            self._session_framework_provider(session_uuid)
            if session_uuid
            else ""
        )
        if not session_uuid or not checkpoint_name:
            return projector.build_fresh_selection_detail(
                session_uuid=session_uuid,
                session_display_name=session_display_name,
                framework_uuid=framework_uuid,
            )

        return projector.build_selection_detail(
            session_uuid=session_uuid,
            checkpoint_name=checkpoint_name,
            session_display_name=session_display_name,
            framework_uuid=framework_uuid,
            snapshot=self._repository.read_artifact(
                session_uuid,
                checkpoint_name,
            ),
        )

    def build_tree_entries(
        self,
        session_name: str | None,
        framework_uuid: str | None = None,
    ) -> list[CheckpointTreeEntry]:
        """Build sorted searchable tree entries for one session."""

        if not session_name:
            return []

        projector = self._projector
        entries = [
            projector.build_tree_entry(
                checkpoint_name,
                self.build_selection_detail(session_name, checkpoint_name),
            )
            for checkpoint_name in self.list_for_session(
                session_name,
                framework_uuid,
            )
        ]
        projector.sort_tree_entries(entries)
        return entries
