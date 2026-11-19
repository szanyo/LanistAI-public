# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Keep grouped-library form edits pure and separate from generation effects."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

from lanista.reinforcengine.utils.scenarios.evaluation_library.contracts import (
    EvaluationGroupDefinition,
    validate_library_id,
    validate_utc_timestamp,
)
from lanista.reinforcengine.utils.scenarios.evaluation_library.generation import (
    GroupedEvaluationLibraryParameters,
)


@dataclass(frozen=True, slots=True)
class EvaluationGroupDraft:
    """Retain editable text while a user is still filling a group row."""

    group_id: str
    name: str
    vehicle_count: str = "0"
    pedestrian_count: str = "0"
    scenario_count: str = "1"


@dataclass(frozen=True, slots=True)
class EvaluationLibraryDraft:
    """Bind one mutable-in-the-GUI form to its original model identity."""

    model_root: Path
    library_id: str
    name: str
    description: str
    created_at: str
    master_seed: str
    groups: tuple[EvaluationGroupDraft, ...] = ()
    next_group_number: int = 1
    selected_library_id: str | None = None
    selected_manifest_sha256: str | None = None


def _number(label: str, raw: str, *, positive: bool = False) -> int:
    """Parse one strict decimal count from form text."""

    text = str(raw).strip()
    if not text or not text.isdecimal():
        raise ValueError(f"{label} must be a nonnegative decimal integer.")
    value = int(text)
    if positive and value == 0:
        raise ValueError(f"{label} must be positive.")
    return value


class EvaluationLibraryDraftService:
    """Apply form edits while keeping IDs stable and rejecting invalid publication."""

    @staticmethod
    def add_group(draft: EvaluationLibraryDraft) -> EvaluationLibraryDraft:
        number = draft.next_group_number
        group = EvaluationGroupDraft(group_id=f"group-{number:03d}", name=f"Group {number}")
        return replace(
            draft,
            groups=(*draft.groups, group),
            next_group_number=number + 1,
        )

    @staticmethod
    def remove_group(draft: EvaluationLibraryDraft, group_id: str) -> EvaluationLibraryDraft:
        groups = tuple(group for group in draft.groups if group.group_id != group_id)
        if len(groups) == len(draft.groups):
            raise KeyError(group_id)
        return replace(draft, groups=groups)

    @staticmethod
    def update_group(
        draft: EvaluationLibraryDraft,
        group_id: str,
        *,
        name: str | None = None,
        vehicle_count: str | None = None,
        pedestrian_count: str | None = None,
        scenario_count: str | None = None,
    ) -> EvaluationLibraryDraft:
        updated = tuple(
            replace(
                group,
                name=group.name if name is None else name,
                vehicle_count=group.vehicle_count if vehicle_count is None else vehicle_count,
                pedestrian_count=(
                    group.pedestrian_count if pedestrian_count is None else pedestrian_count
                ),
                scenario_count=group.scenario_count if scenario_count is None else scenario_count,
            )
            if group.group_id == group_id
            else group
            for group in draft.groups
        )
        if not any(group.group_id == group_id for group in draft.groups):
            raise KeyError(group_id)
        return replace(draft, groups=updated)

    @staticmethod
    def for_model(
        draft: EvaluationLibraryDraft,
        model_root: Path,
    ) -> EvaluationLibraryDraft:
        """Discard a stale draft and selected library when the model changes."""

        selected = Path(model_root)
        if selected == draft.model_root:
            return draft
        return replace(
            draft,
            model_root=selected,
            library_id="",
            name="",
            description="",
            groups=(),
            next_group_number=1,
            selected_library_id=None,
            selected_manifest_sha256=None,
        )

    @staticmethod
    def validate(draft: EvaluationLibraryDraft) -> GroupedEvaluationLibraryParameters:
        """Convert a complete form to validated, filesystem-free generator parameters."""

        validate_library_id(draft.library_id)
        validate_utc_timestamp(draft.created_at)
        name = draft.name.strip()
        if not name or not draft.groups:
            raise ValueError("Library name and at least one group are required.")
        description = draft.description.strip() or None
        groups = tuple(
            EvaluationGroupDefinition(
                group_id=group.group_id,
                name=group.name.strip(),
                vehicle_count=_number("vehicle_count", group.vehicle_count),
                pedestrian_count=_number("pedestrian_count", group.pedestrian_count),
                scenario_count=_number("scenario_count", group.scenario_count, positive=True),
            )
            for group in draft.groups
        )
        ids = tuple(group.group_id for group in groups)
        if ids != tuple(sorted(set(ids), key=lambda value: int(value[6:]))):
            raise ValueError("Group IDs must be unique and ordered by creation index.")
        names = tuple(unicodedata.normalize("NFKC", group.name).casefold() for group in groups)
        if len(names) != len(set(names)):
            raise ValueError("Group names must be unique after case-insensitive normalization.")
        return GroupedEvaluationLibraryParameters(
            library_id=draft.library_id,
            name=name,
            description=description,
            created_at=draft.created_at,
            groups=groups,
            master_seed=_number("master_seed", draft.master_seed),
        )


__all__ = ["EvaluationGroupDraft", "EvaluationLibraryDraft", "EvaluationLibraryDraftService"]
