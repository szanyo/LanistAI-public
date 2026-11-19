# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate model and training-session lineage."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from lanista.reinforcengine.rl_model.sessions.repository_contracts import (
    SessionManifestSummary,
)


class ModelManifestProvider(Protocol):
    """Define the typed boundary for model manifest provider."""

    def get_configuration(self) -> object:
        """Return the configuration exposed by this model manifest provider."""
        ...


class ModelLineageCandidate(Protocol):
    """Define the typed boundary for model lineage candidate."""

    manifest: ModelManifestProvider
    display_name: str
    effective_version: str
    is_current: bool


class WarningLogger(Protocol):
    """Define the typed boundary for warning logger."""

    def warning(self, message: str) -> object:
        """Publish a warning record through this warning logger."""
        ...


@dataclass
class LineageNode:
    """Carry the data required for lineage node."""

    model_uuid: str
    display_name: str
    model_version: str
    parent_model_uuid: str | None
    description: str
    created_at_raw: str | None
    modified_at_raw: str | None
    is_current: bool
    child_count: int


@dataclass(frozen=True, slots=True)
class SessionLineageNode:
    """Immutable UI-neutral session lineage projection."""

    session_uuid: str
    display_name: str
    identity_uuid: str
    framework_uuid: str
    parent_session_uuid: str | None
    description: str
    created_at_raw: str | None
    modified_at_raw: str | None
    child_count: int


class LineageService:
    """Coordinate lineage operations."""

    def __init__(
        self,
        *,
        model_candidates_provider: Callable[
            [],
            tuple[Sequence[ModelLineageCandidate], Sequence[object]],
        ],
        session_summaries_provider: Callable[[], Sequence[SessionManifestSummary]] = tuple,
        session_reference_display_provider: Callable[[str, SessionManifestSummary], str]
        | None = None,
        logger: WarningLogger | None = None,
    ) -> None:
        """Initialize the lineage service and its owned state."""
        self._model_candidates_provider = model_candidates_provider
        self._session_summaries_provider = session_summaries_provider
        self._session_reference_display_provider = session_reference_display_provider
        self._logger = logger

    def _warning(self, message: str) -> None:
        """Publish a warning record through this lineage service."""
        if self._logger is not None and hasattr(self._logger, "warning"):
            self._logger.warning(message)

    def build_model_lineage_index(
        self,
    ) -> tuple[dict[str, LineageNode], dict[str | None, list[str]]]:
        """Build the model lineage index required by this lineage service."""
        valid_entries, _ = self._model_candidates_provider()
        nodes: dict[str, LineageNode] = {}
        children_by_parent: dict[str | None, list[str]] = {}

        for entry in valid_entries:
            cfg = entry.manifest.get_configuration()
            model_uuid = str(getattr(cfg, "model_uuid", "") or "").strip()
            if not model_uuid:
                continue
            parent_model_uuid = str(getattr(cfg, "parent_model_uuid", "") or "").strip() or None
            nodes[model_uuid] = LineageNode(
                model_uuid=model_uuid,
                display_name=str(
                    getattr(cfg, "model_name", "") or entry.display_name or model_uuid
                ),
                model_version=str(
                    getattr(cfg, "model_version", "") or entry.effective_version or ""
                ),
                parent_model_uuid=parent_model_uuid,
                description=str(getattr(cfg, "description", "") or "").strip(),
                created_at_raw=str(getattr(cfg, "created_at", "") or "").strip() or None,
                modified_at_raw=str(getattr(cfg, "modified_at", "") or "").strip() or None,
                is_current=bool(entry.is_current),
                child_count=0,
            )

        for model_uuid, node in list(nodes.items()):
            parent_uuid = node.parent_model_uuid
            if parent_uuid and parent_uuid in nodes:
                children_by_parent.setdefault(parent_uuid, []).append(model_uuid)

        for parent_uuid, child_uuids in children_by_parent.items():
            child_uuids.sort(
                key=lambda child_uuid: (
                    nodes[child_uuid].display_name.lower(),
                    nodes[child_uuid].model_version.lower(),
                    child_uuid,
                )
            )
            if parent_uuid in nodes:
                nodes[parent_uuid].child_count = len(child_uuids)

        return nodes, children_by_parent

    def build_model_ancestry_chain(
        self,
        model_uuid: str | None,
        max_depth: int = 128,
    ) -> list[LineageNode]:
        """Build the model ancestry chain required by this lineage service."""
        if not model_uuid:
            return []

        nodes, _ = self.build_model_lineage_index()
        if model_uuid not in nodes:
            return []

        chain_reversed: list[LineageNode] = []
        visited: set[str] = set()
        current_uuid: str | None = model_uuid
        depth = 0

        while current_uuid and depth < max_depth:
            if current_uuid in visited:
                self._warning(f"Cycle detected in model ancestry chain at '{current_uuid}'.")
                break
            visited.add(current_uuid)

            node = nodes.get(current_uuid)
            if node is None:
                break
            chain_reversed.append(node)
            current_uuid = node.parent_model_uuid
            depth += 1

        chain_reversed.reverse()
        return chain_reversed

    def build_session_lineage_index(
        self,
    ) -> tuple[
        dict[str, SessionLineageNode],
        dict[str | None, list[str]],
    ]:
        """Build the complete session lineage from one summary collection."""

        summaries = tuple(self._session_summaries_provider())
        summaries_by_name = {summary.storage_name: summary for summary in summaries}
        children_by_parent: dict[str | None, list[str]] = {}

        for summary in summaries:
            parent_uuid = summary.parent_session_uuid
            if parent_uuid in summaries_by_name:
                children_by_parent.setdefault(parent_uuid, []).append(summary.storage_name)

        nodes: dict[str, SessionLineageNode] = {}
        for summary in summaries:
            session_uuid = summary.storage_name
            parent_uuid = summary.parent_session_uuid
            if parent_uuid not in summaries_by_name:
                parent_uuid = None
            display_name = session_uuid
            if self._session_reference_display_provider is not None:
                display_name = self._session_reference_display_provider(
                    session_uuid,
                    summary,
                )
            nodes[session_uuid] = SessionLineageNode(
                session_uuid=session_uuid,
                display_name=display_name,
                identity_uuid=summary.identity_uuid or session_uuid,
                framework_uuid=str(summary.framework_uuid or ""),
                parent_session_uuid=parent_uuid,
                description=summary.description,
                created_at_raw=summary.created_at or None,
                modified_at_raw=summary.updated_at or None,
                child_count=len(children_by_parent.get(session_uuid, ())),
            )

        return nodes, children_by_parent
