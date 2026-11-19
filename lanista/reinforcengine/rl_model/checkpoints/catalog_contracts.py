# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed records and the read-only port for checkpoint catalog queries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CheckpointArtifactSnapshot:
    """Capture one path and its optional manifest during a catalog read."""

    path: Path | None
    exists: bool
    saved_at_raw: str | None
    checkpoint_kind: str
    manifest: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CheckpointSelectionDetail:
    """Immutable presentation data for one checkpoint selection."""

    checkpoint_name: str | None
    session_uuid: str | None
    session_display_name: str
    framework_uuid: str
    checkpoint_kind: str
    step_count: int | None
    saved_at_raw: str | None
    scenario_index: int | None
    curriculum_label: str
    parent_checkpoint_name: str | None
    annotation: str | None
    path: str | None
    exists: bool


@dataclass
class CheckpointTreeEntry:
    """Searchable checkpoint row used by session-selection trees."""

    checkpoint_name: str
    display_name: str
    scenario_label: str
    curriculum_label: str
    step_count: int | None
    parent_checkpoint_name: str | None
    search_text: str


class CheckpointCatalogQueryPort(Protocol):
    """Read-only checkpoint operations available to application consumers."""

    def list_for_session(
        self,
        session_name: str,
        framework_uuid: str | None = None,
    ) -> list[str]:
        """List checkpoint artifact names for one compatible session."""

        ...

    def has_checkpoints(
        self,
        session_name: str | None,
        framework_uuid: str | None = None,
    ) -> bool:
        """Report whether a session contains at least one matching artifact."""

        ...

    def build_selection_detail(
        self,
        session_uuid: str | None,
        checkpoint_name: str | None,
    ) -> CheckpointSelectionDetail:
        """Build display-ready metadata for a checkpoint or fresh selection."""

        ...

    def build_tree_entries(
        self,
        session_name: str | None,
        framework_uuid: str | None = None,
    ) -> list[CheckpointTreeEntry]:
        """Build sorted, searchable rows for a checkpoint selection tree."""

        ...
