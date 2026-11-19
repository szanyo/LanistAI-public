# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""State owner connecting clone inventory discovery to the pure reducer."""

from __future__ import annotations

from uuid import uuid4

from lanista.gui.services.model_clone.clone_source_catalog import CloneSourceCatalog
from lanista.gui.services.model_clone.model_clone_selection_contracts import (
    ApproveCloneSelection,
    CancelCloneSelection,
    CloneSelectionEvent,
    CloneSelectionInventory,
    CloneSelectionPolicy,
    CloneSelectionProjection,
    CloneSelectionSide,
    CloneSelectionState,
    CloneSelectionTransition,
    ConfirmCloneSelection,
    RemoveClonePaths,
    ToggleClonePath,
)
from lanista.gui.services.model_clone.model_clone_selection_workflow import (
    project_clone_selection,
    reduce_clone_selection,
    selected_clone_roots,
    start_clone_selection,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    CLONE_MANIFEST_FILE_NAME,
    MODEL_CONFIGURATION_FILE_NAME,
    MODEL_MANIFEST_FILE_NAME,
    RESOURCES_DIR_NAME,
    TENSORBOARD_DIR_NAME,
    TRAINING_SESSIONS_DIR_NAME,
)

DEFAULT_MODEL_CLONE_SELECTION_POLICY = CloneSelectionPolicy(
    non_clonable_root_dirs=frozenset(
        {
            TENSORBOARD_DIR_NAME,
            TRAINING_SESSIONS_DIR_NAME,
        }
    ),
    generated_relative_paths=frozenset(
        {
            MODEL_MANIFEST_FILE_NAME,
            MODEL_CONFIGURATION_FILE_NAME,
            CLONE_MANIFEST_FILE_NAME,
            "__init__.py",
            f"{RESOURCES_DIR_NAME}/__init__.py",
        }
    ),
)


class CloneSelectionCoordinator:
    """Own immutable clone-selection generations for a thin UI adapter."""

    def __init__(self, state: CloneSelectionState) -> None:
        """Initialize the clone selection coordinator and its owned state."""
        self._state = state

    @classmethod
    def from_source(
        cls,
        source_model_folder_path: str,
        policy: CloneSelectionPolicy | None = None,
        *,
        workflow_id: str | None = None,
    ) -> CloneSelectionCoordinator:
        """Create this clone selection coordinator's value from source."""
        resolved_policy = policy or DEFAULT_MODEL_CLONE_SELECTION_POLICY
        entries = CloneSourceCatalog.discover(
            source_model_folder_path,
            root_dirs=set(resolved_policy.non_clonable_root_dirs),
            generated_paths=set(resolved_policy.generated_relative_paths),
        )
        return cls.from_inventory(
            CloneSelectionInventory.from_entries(entries),
            workflow_id=workflow_id,
        )

    @classmethod
    def from_inventory(
        cls,
        inventory: CloneSelectionInventory,
        *,
        workflow_id: str | None = None,
    ) -> CloneSelectionCoordinator:
        """Create this clone selection coordinator's value from inventory."""
        resolved_workflow_id = workflow_id or uuid4().hex
        return cls(
            start_clone_selection(
                inventory,
                workflow_id=resolved_workflow_id,
            )
        )

    @property
    def state(self) -> CloneSelectionState:
        """Return the state exposed by this clone selection coordinator."""
        return self._state

    def project(
        self,
        *,
        source_filter: str = "",
        target_filter: str = "",
    ) -> CloneSelectionProjection:
        """Project the operation owned by this clone selection coordinator."""
        return project_clone_selection(
            self._state,
            source_filter=source_filter,
            target_filter=target_filter,
        )

    def dispatch(self, event: CloneSelectionEvent) -> CloneSelectionTransition:
        """Dispatch the operation owned by this clone selection coordinator."""
        transition = reduce_clone_selection(self._state, event)
        self._state = transition.state
        return transition

    def toggle(
        self,
        side: CloneSelectionSide,
        relative_path: str,
    ) -> CloneSelectionTransition:
        """Return the toggle value owned by this clone selection coordinator."""
        return self.dispatch(
            ToggleClonePath(
                self._state.workflow_id,
                self._state.version,
                side,
                relative_path,
            )
        )

    def approve(self) -> CloneSelectionTransition:
        """Return the approve value owned by this clone selection coordinator."""
        return self.dispatch(
            ApproveCloneSelection(
                self._state.workflow_id,
                self._state.version,
            )
        )

    def remove(self, relative_paths: tuple[str, ...]) -> CloneSelectionTransition:
        """Return the remove value owned by this clone selection coordinator."""
        return self.dispatch(
            RemoveClonePaths(
                self._state.workflow_id,
                self._state.version,
                relative_paths,
            )
        )

    def remove_selected(self) -> CloneSelectionTransition:
        """Remove the selected owned by this clone selection coordinator."""
        return self.remove(
            selected_clone_roots(
                self._state,
                CloneSelectionSide.TARGET,
            )
        )

    def cancel(self) -> CloneSelectionTransition:
        """Cancel the operation owned by this clone selection coordinator."""
        return self.dispatch(
            CancelCloneSelection(
                self._state.workflow_id,
                self._state.version,
            )
        )

    def confirm(self) -> CloneSelectionTransition:
        """Confirm the operation owned by this clone selection coordinator."""
        return self.dispatch(
            ConfirmCloneSelection(
                self._state.workflow_id,
                self._state.version,
            )
        )
