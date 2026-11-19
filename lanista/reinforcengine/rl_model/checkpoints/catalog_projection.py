# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Pure presentation projection for checkpoint details and tree rows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from lanista.reinforcengine.rl_model.checkpoints.catalog_contracts import (
    CheckpointArtifactSnapshot,
    CheckpointSelectionDetail,
    CheckpointTreeEntry,
)
from lanista.reinforcengine.rl_model.checkpoints.name_metadata import (
    extract_checkpoint_steps,
)


@dataclass(frozen=True, slots=True)
class _NormalizedSelectionMetadata:
    """Describe normalized selection data."""

    checkpoint_kind: str
    step_count: int | None
    saved_at_raw: str | None
    scenario_index: int | None
    curriculum_label: str
    parent_checkpoint_name: str | None
    annotation: str | None


class CheckpointCatalogProjector:
    """Convert repository snapshots into stable UI-facing records."""

    def __init__(
        self,
        *,
        step_extractor: Callable[[str], int] = extract_checkpoint_steps,
    ) -> None:
        """Initialize the checkpoint catalog projector and its owned state."""
        self._step_extractor = step_extractor

    @staticmethod
    def build_fresh_selection_detail(
        *,
        session_uuid: str | None,
        session_display_name: str,
        framework_uuid: str,
    ) -> CheckpointSelectionDetail:
        """Build the established no-checkpoint presentation record."""

        return CheckpointSelectionDetail(
            checkpoint_name=None,
            session_uuid=session_uuid,
            session_display_name=session_display_name,
            framework_uuid=framework_uuid or "-",
            checkpoint_kind="fresh",
            step_count=None,
            saved_at_raw=None,
            scenario_index=None,
            curriculum_label="-",
            parent_checkpoint_name=None,
            annotation=None,
            path=None,
            exists=True,
        )

    def build_selection_detail(
        self,
        *,
        session_uuid: str,
        checkpoint_name: str,
        session_display_name: str,
        framework_uuid: str,
        snapshot: CheckpointArtifactSnapshot,
    ) -> CheckpointSelectionDetail:
        """Normalize optional metadata without changing storage semantics."""

        metadata = self._normalize_selection_metadata(
            checkpoint_name,
            snapshot.manifest,
            snapshot,
        )
        return CheckpointSelectionDetail(
            checkpoint_name=checkpoint_name,
            session_uuid=session_uuid,
            session_display_name=session_display_name,
            framework_uuid=framework_uuid or "-",
            checkpoint_kind=metadata.checkpoint_kind,
            step_count=metadata.step_count,
            saved_at_raw=metadata.saved_at_raw,
            scenario_index=metadata.scenario_index,
            curriculum_label=metadata.curriculum_label,
            parent_checkpoint_name=metadata.parent_checkpoint_name,
            annotation=metadata.annotation,
            path=str(snapshot.path) if snapshot.path is not None else None,
            exists=snapshot.exists,
        )

    @staticmethod
    def build_tree_entry(
        checkpoint_name: str,
        detail: CheckpointSelectionDetail,
    ) -> CheckpointTreeEntry:
        """Build one searchable tree row from a normalized detail record."""

        scenario_label = (
            f"Scenario {detail.scenario_index}"
            if detail.scenario_index is not None
            else "Scenario unknown"
        )
        curriculum_label = detail.curriculum_label or "full"
        step_text = (
            str(detail.step_count)
            if detail.step_count is not None and detail.step_count >= 0
            else "-"
        )
        search_text = " ".join(
            [
                checkpoint_name,
                scenario_label,
                curriculum_label,
                detail.parent_checkpoint_name or "",
                step_text,
                detail.saved_at_raw or "",
            ]
        ).lower()
        return CheckpointTreeEntry(
            checkpoint_name=checkpoint_name,
            display_name=f"{checkpoint_name}  steps={step_text}",
            scenario_label=scenario_label,
            curriculum_label=curriculum_label,
            step_count=detail.step_count,
            parent_checkpoint_name=detail.parent_checkpoint_name,
            search_text=search_text,
        )

    @staticmethod
    def sort_tree_entries(
        entries: list[CheckpointTreeEntry],
    ) -> None:
        """Apply the existing reverse scenario/curriculum/step/name order."""

        entries.sort(
            key=lambda entry: (
                entry.scenario_label,
                entry.curriculum_label,
                entry.step_count if entry.step_count is not None else -1,
                entry.checkpoint_name,
            ),
            reverse=True,
        )

    def _normalize_selection_metadata(
        self,
        checkpoint_name: str,
        manifest: Mapping[str, object],
        snapshot: CheckpointArtifactSnapshot,
    ) -> _NormalizedSelectionMetadata:
        """Normalize the selection metadata accepted by this checkpoint catalog projector."""
        saved_at_raw = (
            str(manifest.get("saved_at") or "").strip() or None
            if "saved_at" in manifest
            else snapshot.saved_at_raw
        )
        checkpoint_kind = (
            str(manifest.get("kind") or snapshot.checkpoint_kind or "").strip()
            or snapshot.checkpoint_kind
        )
        step_count = self._normalized_step_count(
            manifest.get("step_count"),
            checkpoint_name,
        )
        return _NormalizedSelectionMetadata(
            checkpoint_kind=checkpoint_kind,
            step_count=step_count,
            saved_at_raw=saved_at_raw,
            scenario_index=self._optional_int(manifest.get("scenario_index")),
            curriculum_label=self._format_curriculum_state_label(
                manifest.get("curriculum_state")
            ),
            parent_checkpoint_name=(
                str(manifest.get("parent_checkpoint_name") or "").strip()
                or None
            ),
            annotation=(
                str(manifest.get("annotation") or "").strip() or None
            ),
        )

    def _normalized_step_count(
        self,
        value: object,
        checkpoint_name: str,
    ) -> int | None:
        """Return the normalized step count value owned by this checkpoint catalog projector."""
        if value is None:
            return self._step_extractor(checkpoint_name)
        converted = self._optional_int(value)
        return (
            converted
            if converted is not None
            else self._step_extractor(checkpoint_name)
        )

    @staticmethod
    def _optional_int(value: object) -> int | None:
        """Return the optional int value owned by this checkpoint catalog projector."""
        if not isinstance(value, str | int | float):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_curriculum_state_label(curriculum_state: object) -> str:
        """Format the curriculum state label for presentation by this checkpoint catalog projector.
        """
        if curriculum_state is None or curriculum_state == "":
            return "full"
        if isinstance(curriculum_state, dict):
            name = str(
                curriculum_state.get("name")
                or curriculum_state.get("state_name")
                or ""
            ).strip()
            if name:
                return name
            target = curriculum_state.get(
                "target_intersection",
                curriculum_state.get("target_value"),
            )
            if target is not None:
                return f"target {target}"
            return "curriculum"
        return str(curriculum_state)
