# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Atomic preparation and commit of a staged model/session selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.runtime.selection_service import RuntimeSelection


@dataclass(frozen=True, slots=True)
class StagedModelSelection:
    """One internally consistent staged model and guarded local selection."""

    manifest: ModelManifest | None
    session_name: str | None
    checkpoint_name: str | None


class StagedModelSelectionService:
    """Resolve and commit all staged-selection fields together."""

    def __init__(
        self,
        *,
        selected_manifest_provider: Callable[[], ModelManifest | None],
        runtime_selection_loader: Callable[[ModelManifest], RuntimeSelection],
        empty_selection_factory: Callable[[], RuntimeSelection],
        state_committer: Callable[[StagedModelSelection], None],
        native_gui_mutation_guard: Callable[[], None] = lambda: None,
    ) -> None:
        """Initialize the staged model selection service and its owned state."""
        self._selected_manifest_provider = selected_manifest_provider
        self._runtime_selection_loader = runtime_selection_loader
        self._empty_selection_factory = empty_selection_factory
        self._state_committer = state_committer
        self._native_gui_mutation_guard = native_gui_mutation_guard

    def get_selected_manifest(self) -> ModelManifest | None:
        """Return the currently staged manifest without exposing its owner."""

        return self._selected_manifest_provider()

    def stage_manifest(
        self,
        model_manifest: ModelManifest | None,
    ) -> StagedModelSelection:
        """Resolve defaults, then commit one coherent snapshot."""

        self._native_gui_mutation_guard()
        runtime_selection = (
            self._runtime_selection_loader(model_manifest)
            if model_manifest is not None
            else self._empty_selection_factory()
        )
        staged_selection = StagedModelSelection(
            manifest=model_manifest,
            session_name=runtime_selection.session_name,
            checkpoint_name=runtime_selection.checkpoint_name,
        )
        self._state_committer(staged_selection)
        return staged_selection
