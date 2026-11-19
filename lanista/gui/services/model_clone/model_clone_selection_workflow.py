# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Immutable model-clone selection state, transitions, and tree projections."""

from __future__ import annotations

from dataclasses import replace

from lanista.gui.services.model_clone.clone_source_catalog import (
    CloneSourceCatalog,
    CloneTreeEntry,
)
from lanista.gui.services.model_clone.model_clone_selection_contracts import (
    ApproveCloneSelection,
    CancelCloneSelection,
    ClonePathSelectionState,
    CloneSelectionEffect,
    CloneSelectionEvent,
    CloneSelectionInventory,
    CloneSelectionPhase,
    CloneSelectionProjection,
    CloneSelectionSide,
    CloneSelectionState,
    CloneSelectionTransition,
    CloneTreeNodeProjection,
    ConfirmCloneSelection,
    DismissCloneSelection,
    RemoveClonePaths,
    RenderCloneSelection,
    ShowCloneSelectionHint,
    StaleCloneSelectionIgnored,
    ToggleClonePath,
)


def normalize_clone_path(value: str) -> str:
    """Return the canonical slash-separated clone-relative path."""

    return CloneSourceCatalog.normalize_relative_path(value)


def clone_parent_path(relative_path: str) -> str | None:
    """Return the normalized parent path, if the entry is not at the root."""

    normalized = normalize_clone_path(relative_path)
    if "/" not in normalized:
        return None
    return normalized.rsplit("/", 1)[0]


def clone_entry_name(relative_path: str) -> str:
    """Return the final normalized path component."""

    return normalize_clone_path(relative_path).rsplit("/", 1)[-1]


def start_clone_selection(
    inventory: CloneSelectionInventory,
    *,
    workflow_id: str,
) -> CloneSelectionState:
    """Create one active workflow over a frozen source inventory."""

    return CloneSelectionState(
        workflow_id=workflow_id,
        version=0,
        phase=CloneSelectionPhase.ACTIVE,
        inventory=inventory,
    )


def _selection_state(
    relative_path: str,
    active_paths: frozenset[str],
    selected_paths: frozenset[str],
) -> ClonePathSelectionState:
    """Return the selection state value owned by this model clone selection workflow module."""
    descendants = frozenset(path for path in active_paths if path.startswith(f"{relative_path}/"))
    own_selected = relative_path in selected_paths
    if not descendants:
        return (
            ClonePathSelectionState.SELECTED if own_selected else ClonePathSelectionState.UNSELECTED
        )
    states = (own_selected, *(path in selected_paths for path in descendants))
    if all(states):
        return ClonePathSelectionState.SELECTED
    if not any(states):
        return ClonePathSelectionState.UNSELECTED
    return ClonePathSelectionState.PARTIAL


def _selection_label(
    relative_path: str,
    is_dir: bool,
    active_paths: frozenset[str],
    selected_paths: frozenset[str],
) -> str:
    """Return the selection label value owned by this model clone selection workflow module."""
    markers = {
        ClonePathSelectionState.SELECTED: "[•]",
        ClonePathSelectionState.PARTIAL: "[~]",
        ClonePathSelectionState.UNSELECTED: "[ ]",
    }
    marker = markers[_selection_state(relative_path, active_paths, selected_paths)]
    suffix = "/" if is_dir else ""
    return f"{marker} {clone_entry_name(relative_path)}{suffix}"


def _clear_selected_paths(
    selected_paths: frozenset[str],
    paths_to_clear: frozenset[str],
) -> frozenset[str]:
    """Clear the selected paths owned by this model clone selection workflow module."""
    remaining = set(selected_paths)
    for relative_path in paths_to_clear:
        remaining.discard(relative_path)
        parent_path = clone_parent_path(relative_path)
        while parent_path:
            remaining.discard(parent_path)
            parent_path = clone_parent_path(parent_path)
    return frozenset(remaining)


def _toggle_selected_path(
    selected_paths: frozenset[str],
    relative_path: str,
    active_paths: frozenset[str],
    directory_paths: frozenset[str],
) -> frozenset[str]:
    """Return the toggle selected path value owned by this model clone selection workflow module."""
    normalized = normalize_clone_path(relative_path)
    if normalized not in active_paths:
        return selected_paths
    paths_to_change = {normalized}
    if normalized in directory_paths:
        paths_to_change.update(path for path in active_paths if path.startswith(f"{normalized}/"))
    frozen_changes = frozenset(paths_to_change)
    if (
        _selection_state(normalized, active_paths, selected_paths)
        is not ClonePathSelectionState.SELECTED
    ):
        return selected_paths | frozen_changes
    return _clear_selected_paths(selected_paths - frozen_changes, frozen_changes)


def _selected_roots(
    selected_paths: frozenset[str],
    active_paths: frozenset[str],
) -> tuple[str, ...]:
    """Return the selected roots value owned by this model clone selection workflow module."""
    active_selection = selected_paths & active_paths
    roots: list[str] = []
    for relative_path in sorted(active_selection):
        parent_path = clone_parent_path(relative_path)
        while parent_path and parent_path not in active_selection:
            parent_path = clone_parent_path(parent_path)
        if parent_path is None:
            roots.append(relative_path)
    return tuple(roots)


def _is_path_approved(
    relative_path: str,
    approved_paths: frozenset[str],
) -> bool:
    """Return whether path approved applies to this model clone selection workflow module."""
    normalized = normalize_clone_path(relative_path)
    if not normalized:
        return False
    if normalized in approved_paths:
        return True
    parent_path = clone_parent_path(normalized)
    while parent_path:
        if parent_path in approved_paths:
            return True
        parent_path = clone_parent_path(parent_path)
    return False


def _visible_side_paths(
    state: CloneSelectionState,
    side: CloneSelectionSide,
) -> frozenset[str]:
    """Return the visible side paths value owned by this model clone selection workflow module."""
    target_side = side is CloneSelectionSide.TARGET
    path_state = {
        entry.relative_path: _is_path_approved(
            entry.relative_path,
            state.approved_paths,
        )
        for entry in state.inventory.entries
    }
    visible_paths: set[str] = set()
    for entry in state.inventory.entries:
        prefix = f"{entry.relative_path}/"
        descendants = {
            path: approved for path, approved in path_state.items() if path.startswith(prefix)
        }
        entry_on_side = path_state[entry.relative_path] == target_side
        descendant_on_side = any(approved == target_side for approved in descendants.values())
        if descendant_on_side or (entry_on_side and not descendants):
            visible_paths.add(entry.relative_path)
    return frozenset(visible_paths)


def selected_clone_roots(
    state: CloneSelectionState,
    side: CloneSelectionSide,
) -> tuple[str, ...]:
    """Return the minimal selected paths currently actionable on one side."""

    active_paths = _visible_side_paths(state, side)
    selected_paths = (
        state.source_selected_paths
        if side is CloneSelectionSide.SOURCE
        else state.target_selected_paths
    )
    return _selected_roots(selected_paths, active_paths)


def _expand_source_paths(
    relative_paths: tuple[str, ...],
    inventory: CloneSelectionInventory,
) -> frozenset[str]:
    """Return the expand source paths value owned by this model clone selection workflow module."""
    expanded_paths: set[str] = set()
    for relative_path in relative_paths:
        normalized = normalize_clone_path(relative_path)
        if not normalized:
            continue
        expanded_paths.add(normalized)
        if normalized in inventory.directory_paths:
            expanded_paths.update(
                path
                for path in inventory.all_paths
                if path == normalized or path.startswith(f"{normalized}/")
            )
    return frozenset(expanded_paths)


def _same_or_descendant(relative_path: str, parent_path: str) -> bool:
    """Return the same or descendant value owned by this model clone selection workflow module."""
    normalized_path = normalize_clone_path(relative_path)
    normalized_parent = normalize_clone_path(parent_path)
    return normalized_path == normalized_parent or normalized_path.startswith(
        f"{normalized_parent}/"
    )


def _covering_approved_ancestors(
    targets: frozenset[str],
    approved_paths: frozenset[str],
) -> frozenset[str]:
    """Return the covering approved ancestors value owned by this model clone selection workflow
    module.
    """
    covering_paths: set[str] = set()
    for relative_path in targets:
        parent_path = clone_parent_path(relative_path)
        while parent_path:
            if parent_path in approved_paths:
                covering_paths.add(parent_path)
            parent_path = clone_parent_path(parent_path)
    return frozenset(covering_paths)


def _preserved_descendants(
    covering_ancestors: frozenset[str],
    targets: frozenset[str],
    inventory: CloneSelectionInventory,
) -> frozenset[str]:
    """Return the preserved descendants value owned by this model clone selection workflow module.
    """
    preserved_paths: set[str] = set()
    for ancestor_path in covering_ancestors:
        preserved_paths.update(
            entry.relative_path
            for entry in inventory.entries
            if entry.relative_path.startswith(f"{ancestor_path}/")
            and not any(_same_or_descendant(entry.relative_path, target) for target in targets)
            and not any(_same_or_descendant(target, entry.relative_path) for target in targets)
        )
    return frozenset(preserved_paths)


def _remove_approved_paths(
    approved_paths: frozenset[str],
    relative_paths: tuple[str, ...],
    inventory: CloneSelectionInventory,
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Remove the approved paths owned by this model clone selection workflow module."""
    targets = frozenset(
        normalized
        for relative_path in relative_paths
        if (normalized := normalize_clone_path(relative_path))
    )
    if not targets:
        return approved_paths, frozenset(), targets
    paths_to_remove = frozenset(
        path
        for path in approved_paths
        if any(_same_or_descendant(path, target) for target in targets)
    )
    covering_ancestors = _covering_approved_ancestors(targets, approved_paths)
    paths_to_remove |= covering_ancestors
    if not paths_to_remove:
        return approved_paths, frozenset(), targets
    preserved = _preserved_descendants(covering_ancestors, targets, inventory)
    remaining = (approved_paths - paths_to_remove) | preserved
    return remaining, approved_paths - remaining, targets


def _source_visible_entries(
    state: CloneSelectionState,
    active_paths: frozenset[str],
    filter_text: str,
) -> tuple[CloneTreeEntry, ...]:
    """Return the source visible entries value owned by this model clone selection workflow module.
    """
    available = tuple(
        entry for entry in state.inventory.entries if entry.relative_path in active_paths
    )
    normalized_filter = filter_text.strip().lower()
    if not normalized_filter:
        return available
    matched_paths = {
        entry.relative_path
        for entry in available
        if normalized_filter in entry.relative_path.lower()
    }
    visible_paths = set(matched_paths)
    for matched_path in matched_paths:
        parent_path = clone_parent_path(matched_path)
        while parent_path:
            if parent_path in active_paths:
                visible_paths.add(parent_path)
            parent_path = clone_parent_path(parent_path)
    return tuple(entry for entry in available if entry.relative_path in visible_paths)


def _target_visible_paths(
    state: CloneSelectionState,
    display_paths: frozenset[str],
    filter_text: str,
) -> frozenset[str]:
    """Return the target visible paths value owned by this model clone selection workflow module."""
    normalized_filter = filter_text.strip().lower()
    if not normalized_filter:
        return display_paths
    matched_paths = frozenset(
        path for path in state.approved_paths if normalized_filter in path.lower()
    )
    return frozenset(
        path
        for path in display_paths
        if path in matched_paths
        or any(path.startswith(f"{matched}/") for matched in matched_paths)
        or any(matched.startswith(f"{path}/") for matched in matched_paths)
    )


def project_clone_selection(
    state: CloneSelectionState,
    *,
    source_filter: str = "",
    target_filter: str = "",
) -> CloneSelectionProjection:
    """Build both tree projections and control enablement from one state."""

    source_active = _visible_side_paths(state, CloneSelectionSide.SOURCE)
    target_active = _visible_side_paths(state, CloneSelectionSide.TARGET)
    source_entries = _source_visible_entries(state, source_active, source_filter)
    target_paths = _target_visible_paths(state, target_active, target_filter)
    source_nodes = tuple(
        CloneTreeNodeProjection(
            entry.relative_path,
            clone_parent_path(entry.relative_path),
            _selection_label(
                entry.relative_path,
                entry.is_dir,
                source_active,
                state.source_selected_paths,
            ),
            entry.is_dir,
        )
        for entry in source_entries
    )
    target_nodes = tuple(
        CloneTreeNodeProjection(
            path,
            clone_parent_path(path),
            _selection_label(
                path,
                path in state.inventory.directory_paths,
                target_active,
                state.target_selected_paths,
            ),
            path in state.inventory.directory_paths,
        )
        for path in sorted(target_paths)
    )
    target_empty_message = None
    if not state.approved_paths:
        target_empty_message = "No approved clone entries yet."
    elif not target_nodes:
        target_empty_message = "No approved entries match search."
    return CloneSelectionProjection(
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        source_empty_message=None if source_nodes else "No matching source entries.",
        target_empty_message=target_empty_message,
        can_approve=bool(selected_clone_roots(state, CloneSelectionSide.SOURCE)),
        can_remove=bool(selected_clone_roots(state, CloneSelectionSide.TARGET)),
        can_confirm=bool(state.approved_paths),
    )


def _toggle_path(
    state: CloneSelectionState,
    event: ToggleClonePath,
) -> CloneSelectionTransition:
    """Return the toggle path value owned by this model clone selection workflow module."""
    active_paths = _visible_side_paths(state, event.side)
    selected_paths = (
        state.source_selected_paths
        if event.side is CloneSelectionSide.SOURCE
        else state.target_selected_paths
    )
    next_selection = _toggle_selected_path(
        selected_paths,
        event.relative_path,
        active_paths,
        state.inventory.directory_paths,
    )
    if next_selection == selected_paths:
        return CloneSelectionTransition(state)
    if event.side is CloneSelectionSide.SOURCE:
        next_state = replace(
            state,
            version=state.version + 1,
            source_selected_paths=next_selection,
        )
        render = RenderCloneSelection(next_state.workflow_id, next_state.version, True, False)
    else:
        next_state = replace(
            state,
            version=state.version + 1,
            target_selected_paths=next_selection,
        )
        render = RenderCloneSelection(next_state.workflow_id, next_state.version, False, True)
    return CloneSelectionTransition(next_state, (render,))


def _approve_selection(
    state: CloneSelectionState,
) -> CloneSelectionTransition:
    """Return the approve selection value owned by this model clone selection workflow module."""
    available_paths = _visible_side_paths(state, CloneSelectionSide.SOURCE)
    selected_roots = _selected_roots(state.source_selected_paths, available_paths)
    if not selected_roots:
        hint = ShowCloneSelectionHint(
            state.workflow_id,
            state.version,
            "Select at least one source entry before approval.",
        )
        return CloneSelectionTransition(state, (hint,))
    expanded_paths = _expand_source_paths(selected_roots, state.inventory)
    next_state = replace(
        state,
        version=state.version + 1,
        approved_paths=state.approved_paths | expanded_paths,
        source_selected_paths=_clear_selected_paths(
            state.source_selected_paths,
            expanded_paths,
        ),
        target_selected_paths=_clear_selected_paths(
            state.target_selected_paths,
            expanded_paths,
        ),
    )
    noun = "entry" if len(selected_roots) == 1 else "entries"
    effects: tuple[CloneSelectionEffect, ...] = (
        RenderCloneSelection(next_state.workflow_id, next_state.version, True, True),
        ShowCloneSelectionHint(
            next_state.workflow_id,
            next_state.version,
            f"Approved {len(selected_roots)} source {noun}.",
        ),
    )
    return CloneSelectionTransition(next_state, effects)


def _remove_paths(
    state: CloneSelectionState,
    event: RemoveClonePaths,
) -> CloneSelectionTransition:
    """Remove the paths owned by this model clone selection workflow module."""
    remaining, removed_paths, normalized_targets = _remove_approved_paths(
        state.approved_paths,
        event.relative_paths,
        state.inventory,
    )
    if not removed_paths:
        hint = ShowCloneSelectionHint(
            state.workflow_id,
            state.version,
            "Select an approved entry on the right tree to remove it.",
        )
        return CloneSelectionTransition(state, (hint,))
    next_state = replace(
        state,
        version=state.version + 1,
        approved_paths=remaining,
        source_selected_paths=_clear_selected_paths(
            state.source_selected_paths,
            removed_paths,
        ),
        target_selected_paths=_clear_selected_paths(
            state.target_selected_paths,
            removed_paths | normalized_targets,
        ),
    )
    noun = "entry" if len(event.relative_paths) == 1 else "entries"
    effects: tuple[CloneSelectionEffect, ...] = (
        RenderCloneSelection(next_state.workflow_id, next_state.version, True, True),
        ShowCloneSelectionHint(
            next_state.workflow_id,
            next_state.version,
            f"Removed {len(event.relative_paths)} approved {noun} from clone list.",
        ),
    )
    return CloneSelectionTransition(next_state, effects)


def _cancel_selection(state: CloneSelectionState) -> CloneSelectionTransition:
    """Cancel the selection owned by this model clone selection workflow module."""
    next_state = replace(
        state,
        version=state.version + 1,
        phase=CloneSelectionPhase.CANCELLED,
    )
    effect = DismissCloneSelection(next_state.workflow_id, next_state.version, None)
    return CloneSelectionTransition(next_state, (effect,))


def _confirm_selection(state: CloneSelectionState) -> CloneSelectionTransition:
    """Confirm the selection owned by this model clone selection workflow module."""
    if not state.approved_paths:
        hint = ShowCloneSelectionHint(
            state.workflow_id,
            state.version,
            "No approved clone entries selected.",
        )
        return CloneSelectionTransition(state, (hint,))
    next_state = replace(
        state,
        version=state.version + 1,
        phase=CloneSelectionPhase.CONFIRMED,
    )
    effect = DismissCloneSelection(
        next_state.workflow_id,
        next_state.version,
        tuple(sorted(next_state.approved_paths)),
    )
    return CloneSelectionTransition(next_state, (effect,))


def reduce_clone_selection(
    state: CloneSelectionState,
    event: CloneSelectionEvent,
) -> CloneSelectionTransition:
    """Apply one versioned event and reject late workflow callbacks."""

    if event.workflow_id != state.workflow_id or event.expected_version != state.version:
        stale = StaleCloneSelectionIgnored(
            state.workflow_id,
            state.version,
            event.workflow_id,
            event.expected_version,
        )
        return CloneSelectionTransition(state, (stale,))
    if state.phase is not CloneSelectionPhase.ACTIVE:
        return CloneSelectionTransition(state)
    if isinstance(event, ToggleClonePath):
        return _toggle_path(state, event)
    if isinstance(event, ApproveCloneSelection):
        return _approve_selection(state)
    if isinstance(event, RemoveClonePaths):
        return _remove_paths(state, event)
    if isinstance(event, CancelCloneSelection):
        return _cancel_selection(state)
    if isinstance(event, ConfirmCloneSelection):
        return _confirm_selection(state)
    return CloneSelectionTransition(state)
