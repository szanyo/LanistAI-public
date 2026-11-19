# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Immutable contracts shared by clone-selection reducers and adapters."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from lanista.gui.services.model_clone.clone_source_catalog import (
    CloneSourceCatalog,
    CloneTreeEntry,
)


@dataclass(frozen=True, slots=True)
class CloneSelectionPolicy:
    """Filesystem entries excluded from a clone candidate inventory."""

    non_clonable_root_dirs: frozenset[str]
    generated_relative_paths: frozenset[str]


@dataclass(frozen=True, slots=True)
class CloneSelectionInventory:
    """Normalized source tree used by every clone-selection decision."""

    entries: tuple[CloneTreeEntry, ...]
    all_paths: frozenset[str]
    directory_paths: frozenset[str]

    def __post_init__(self) -> None:
        """Validate and finalize the initialized clone selection inventory state."""
        paths = tuple(entry.relative_path for entry in self.entries)
        if any(
            not path or CloneSourceCatalog.normalize_relative_path(path) != path for path in paths
        ):
            raise ValueError("Clone inventory paths must be normalized and relative.")
        if len(set(paths)) != len(paths):
            raise ValueError("Clone inventory paths must be unique.")
        expected_entries = tuple(sorted(self.entries, key=_clone_tree_entry_order))
        if self.entries != expected_entries:
            raise ValueError("Clone inventory entries must use deterministic tree order.")
        expected_paths = frozenset(paths)
        expected_directories = frozenset(
            entry.relative_path for entry in self.entries if entry.is_dir
        )
        if self.all_paths != expected_paths or self.directory_paths != expected_directories:
            raise ValueError("Clone inventory indexes do not match its entries.")
        entries_by_path = {entry.relative_path: entry for entry in self.entries}
        for relative_path in paths:
            if "/" not in relative_path:
                continue
            parent_path = relative_path.rsplit("/", 1)[0]
            parent = entries_by_path.get(parent_path)
            if parent is None or not parent.is_dir:
                raise ValueError("Clone inventory parents must be present directories.")

    @classmethod
    def from_entries(
        cls,
        entries: Iterable[CloneTreeEntry],
    ) -> CloneSelectionInventory:
        """Create this clone selection inventory's value from entries."""
        normalized_entries = tuple(sorted(entries, key=_clone_tree_entry_order))
        return cls(
            entries=normalized_entries,
            all_paths=frozenset(entry.relative_path for entry in normalized_entries),
            directory_paths=frozenset(
                entry.relative_path for entry in normalized_entries if entry.is_dir
            ),
        )


def _clone_tree_entry_order(entry: CloneTreeEntry) -> tuple[int, str, str]:
    """Return the clone tree entry order value owned by this model clone selection contracts
    module.
    """
    return (
        entry.relative_path.count("/"),
        entry.relative_path.lower(),
        entry.relative_path,
    )


class CloneSelectionSide(StrEnum):
    """Enumerate the source and target sides of clone selection."""

    SOURCE = "source"
    TARGET = "target"


class CloneSelectionPhase(StrEnum):
    """Enumerate the phases of the clone-selection workflow."""

    ACTIVE = "active"
    CANCELLED = "cancelled"
    CONFIRMED = "confirmed"


class ClonePathSelectionState(StrEnum):
    """Enumerate the selectable states of one clone path."""

    UNSELECTED = "unselected"
    PARTIAL = "partial"
    SELECTED = "selected"


@dataclass(frozen=True, slots=True)
class CloneSelectionState:
    """Single immutable source of truth for one clone-selection modal."""

    workflow_id: str
    version: int
    phase: CloneSelectionPhase
    inventory: CloneSelectionInventory
    approved_paths: frozenset[str] = frozenset()
    source_selected_paths: frozenset[str] = frozenset()
    target_selected_paths: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class CloneSelectionEvent:
    """Represent the clone selection lifecycle event."""

    workflow_id: str
    expected_version: int


@dataclass(frozen=True, slots=True)
class ToggleClonePath(CloneSelectionEvent):
    """Carry toggle clone path data through the workflow event stream."""

    side: CloneSelectionSide
    relative_path: str


@dataclass(frozen=True, slots=True)
class ApproveCloneSelection(CloneSelectionEvent):
    """Carry approve clone selection data through the workflow event stream."""

    pass


@dataclass(frozen=True, slots=True)
class RemoveClonePaths(CloneSelectionEvent):
    """Carry remove clone paths data through the workflow event stream."""

    relative_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CancelCloneSelection(CloneSelectionEvent):
    """Carry cancel clone selection data through the workflow event stream."""

    pass


@dataclass(frozen=True, slots=True)
class ConfirmCloneSelection(CloneSelectionEvent):
    """Carry confirm clone selection data through the workflow event stream."""

    pass


@dataclass(frozen=True, slots=True)
class CloneSelectionEffect:
    """Represent the clone selection workflow operation."""

    workflow_id: str
    version: int


@dataclass(frozen=True, slots=True)
class RenderCloneSelection(CloneSelectionEffect):
    """Describe the render clone selection workflow effect."""

    source_tree: bool
    target_tree: bool


@dataclass(frozen=True, slots=True)
class ShowCloneSelectionHint(CloneSelectionEffect):
    """Describe the show clone selection hint workflow effect."""

    message: str


@dataclass(frozen=True, slots=True)
class DismissCloneSelection(CloneSelectionEffect):
    """Describe the dismiss clone selection workflow effect."""

    selected_relative_paths: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class StaleCloneSelectionIgnored(CloneSelectionEffect):
    """Describe the stale clone selection ignored workflow effect."""

    received_workflow_id: str
    received_version: int


@dataclass(frozen=True, slots=True)
class CloneSelectionTransition:
    """Represent the clone selection workflow operation."""

    state: CloneSelectionState
    effects: tuple[CloneSelectionEffect, ...] = ()


@dataclass(frozen=True, slots=True)
class CloneTreeNodeProjection:
    """Carry the data required for clone tree node projection."""

    relative_path: str
    parent_path: str | None
    label: str
    is_dir: bool


@dataclass(frozen=True, slots=True)
class CloneSelectionProjection:
    """Carry the data required for clone selection projection."""

    source_nodes: tuple[CloneTreeNodeProjection, ...]
    target_nodes: tuple[CloneTreeNodeProjection, ...]
    source_empty_message: str | None
    target_empty_message: str | None
    can_approve: bool
    can_remove: bool
    can_confirm: bool
