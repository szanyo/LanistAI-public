# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Textual adapter for the immutable model-clone selection workflow."""

from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Tree
from textual.widgets._tree import TreeNode

from lanista.gui.services.model_clone.model_clone_selection_contracts import (
    CloneSelectionEffect,
    CloneSelectionProjection,
    CloneSelectionSide,
    CloneSelectionTransition,
    CloneTreeNodeProjection,
    DismissCloneSelection,
    RenderCloneSelection,
    ShowCloneSelectionHint,
    StaleCloneSelectionIgnored,
)
from lanista.gui.services.model_clone.model_clone_selection_coordinator import (
    CloneSelectionCoordinator,
)
from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.utils.tree_nodes import first_string_data_node
from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.locations import get_model_clone_selection_dialog_css_path


@dataclass(frozen=True, slots=True)
class ModelCloneSelectionResult:
    """Report the result of model clone selection."""

    selected_relative_paths: tuple[str, ...]


class CloneTransferTree(Tree[str | None]):
    """Store and navigate the clone transfer structure."""

    def __init__(self, label: str, *, id: str) -> None:
        """Initialize the clone transfer tree and its owned state."""
        super().__init__(label, id=id)
        self.show_root = False
        self.show_guides = True


class ModelCloneSelectionDialog(
    DirectionalFocusMixin,
    HelpEnabledModalMixin,
    ModalScreen[ModelCloneSelectionResult | None],
):
    """Translate Textual input and workflow effects without owning decisions."""

    _HELP_TOPIC = "model_clone"
    CSS_PATH = get_model_clone_selection_dialog_css_path()
    _TREE_SIDES = {
        "clone_selection_source_tree": CloneSelectionSide.SOURCE,
        "clone_selection_target_tree": CloneSelectionSide.TARGET,
    }

    _FOOTER_BINDINGS: ClassVar[tuple[Binding, ...]] = (
        Binding("f3", "shortcut_approve", "Approve", key_display="f3"),
        Binding("delete", "shortcut_remove", "Remove", key_display="del"),
        Binding("f4,enter", "shortcut_confirm", "Confirm"),
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
        Binding("escape", "shortcut_cancel", "Cancel", key_display="esc"),
    )
    BINDINGS = list(_FOOTER_BINDINGS)

    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "clone_selection_source_search": {
            "down": "clone_selection_source_tree",
            "right": "clone_selection_source_tree",
        },
        "clone_selection_source_tree": {
            "up": "clone_selection_source_search",
            "right": "clone_selection_approve",
            "down": "clone_selection_confirm",
        },
        "clone_selection_approve": {
            "left": "clone_selection_source_tree",
            "right": "clone_selection_target_search",
            "down": "clone_selection_remove",
        },
        "clone_selection_remove": {
            "left": "clone_selection_source_tree",
            "right": "clone_selection_target_search",
            "up": "clone_selection_approve",
            "down": "clone_selection_confirm",
        },
        "clone_selection_target_search": {
            "left": "clone_selection_remove",
            "down": "clone_selection_target_tree",
        },
        "clone_selection_target_tree": {
            "up": "clone_selection_target_search",
            "left": "clone_selection_remove",
            "down": "clone_selection_confirm",
        },
        "clone_selection_confirm": {
            "up": "clone_selection_source_tree",
            "right": "clone_selection_cancel",
        },
        "clone_selection_cancel": {
            "up": "clone_selection_target_tree",
            "left": "clone_selection_confirm",
        },
    }

    def __init__(
        self,
        source_model_folder_path: str,
        source_display_name: str,
    ) -> None:
        """Initialize the model clone selection dialog and its owned state."""
        super().__init__()
        self._source_model_folder_path = str(source_model_folder_path or "").strip()
        self._source_display_name = str(source_display_name or "-").strip() or "-"
        self._coordinator: CloneSelectionCoordinator | None = None
        self._last_tree_select_path: dict[str, str] = {}
        self._last_tree_select_ts: dict[str, float] = {}

    def _workflow(self) -> CloneSelectionCoordinator:
        """Return the workflow value owned by this model clone selection dialog."""
        coordinator = self._coordinator
        if coordinator is None:
            raise RuntimeError("Clone-selection workflow is not mounted.")
        return coordinator

    def _projection(
        self,
        *,
        source_filter: str | None = None,
        target_filter: str | None = None,
    ) -> CloneSelectionProjection:
        """Return the projection value owned by this model clone selection dialog."""
        resolved_source_filter = source_filter
        if resolved_source_filter is None:
            resolved_source_filter = self.query_one(
                "#clone_selection_source_search",
                Input,
            ).value
        resolved_target_filter = target_filter
        if resolved_target_filter is None:
            resolved_target_filter = self.query_one(
                "#clone_selection_target_search",
                Input,
            ).value
        return self._workflow().project(
            source_filter=resolved_source_filter,
            target_filter=resolved_target_filter,
        )

    @staticmethod
    def _expanded_tree_paths(tree: Tree[str | None]) -> set[str]:
        """Return the expanded tree paths value owned by this model clone selection dialog."""
        expanded_paths: set[str] = set()
        nodes_to_visit = list(tree.root.children)
        while nodes_to_visit:
            node = nodes_to_visit.pop()
            if isinstance(node.data, str) and node.is_expanded:
                expanded_paths.add(node.data)
            nodes_to_visit.extend(node.children)
        return expanded_paths

    @staticmethod
    def _restore_expanded_tree_paths(
        tree: Tree[str | None],
        expanded_paths: set[str],
    ) -> None:
        """Restore the expanded tree paths owned by this model clone selection dialog."""
        nodes_to_visit = list(tree.root.children)
        while nodes_to_visit:
            node = nodes_to_visit.pop()
            if isinstance(node.data, str) and node.data in expanded_paths and node.allow_expand:
                node.expand()
            nodes_to_visit.extend(node.children)

    def _select_first_tree_node(self, tree: Tree[str | None]) -> None:
        """Select the first tree node owned by this model clone selection dialog."""
        first_node = first_string_data_node(tree.root)
        if first_node is None:
            return
        if tree.id:
            self._last_tree_select_path.pop(tree.id, None)
            self._last_tree_select_ts.pop(tree.id, None)
        tree.select_node(first_node)

    @staticmethod
    def _populate_tree(
        tree: Tree[str | None],
        nodes: tuple[CloneTreeNodeProjection, ...],
    ) -> None:
        """Populate the tree owned by this model clone selection dialog."""
        node_lookup: dict[str, TreeNode[str | None]] = {}
        for entry in nodes:
            parent_node = (
                node_lookup.get(entry.parent_path, tree.root) if entry.parent_path else tree.root
            )
            node = parent_node.add(
                entry.label,
                data=entry.relative_path,
                expand=False,
            )
            node.allow_expand = entry.is_dir
            node_lookup[entry.relative_path] = node

    def _sync_transfer_buttons(
        self,
        projection: CloneSelectionProjection | None = None,
    ) -> None:
        """Synchronize the transfer buttons owned by this model clone selection
dialog."""
        approve_button = self.query_one("#clone_selection_approve", Button)
        remove_button = self.query_one("#clone_selection_remove", Button)
        resolved_projection = projection or self._projection()
        approve_button.disabled = not resolved_projection.can_approve
        remove_button.disabled = not resolved_projection.can_remove

    def _rebuild_source_tree(self, filter_text: str = "") -> None:
        """Rebuild the source tree owned by this model clone selection dialog."""
        tree = self.query_one("#clone_selection_source_tree", CloneTransferTree)
        expanded_paths = self._expanded_tree_paths(tree)
        projection = self._projection(source_filter=filter_text)
        tree.clear()
        tree.root.expand()
        if not projection.source_nodes:
            empty_node = tree.root.add_leaf(
                projection.source_empty_message or "No matching source entries.",
                data=None,
            )
            empty_node.allow_expand = False
        else:
            self._populate_tree(tree, projection.source_nodes)
            self._restore_expanded_tree_paths(tree, expanded_paths)
            self._select_first_tree_node(tree)
        self._sync_transfer_buttons(projection)

    def _rebuild_target_tree(self) -> None:
        """Rebuild the target tree owned by this model clone selection dialog."""
        tree = self.query_one("#clone_selection_target_tree", CloneTransferTree)
        expanded_paths = self._expanded_tree_paths(tree)
        projection = self._projection()
        tree.clear()
        tree.root.expand()
        if not projection.target_nodes:
            empty_node = tree.root.add_leaf(
                projection.target_empty_message or "No approved clone entries yet.",
                data=None,
            )
            empty_node.allow_expand = False
        else:
            self._populate_tree(tree, projection.target_nodes)
            self._restore_expanded_tree_paths(tree, expanded_paths)
            self._select_first_tree_node(tree)
        self.query_one("#clone_selection_confirm", Button).disabled = not projection.can_confirm
        self._sync_transfer_buttons(projection)

    def _effect_is_current(self, effect: CloneSelectionEffect) -> bool:
        """Return whether effect is current applies to this model clone selection dialog."""
        state = self._workflow().state
        return effect.workflow_id == state.workflow_id and effect.version == state.version

    def _execute_effect(self, effect: CloneSelectionEffect) -> None:
        """Execute the effect owned by this model clone selection dialog."""
        if isinstance(effect, StaleCloneSelectionIgnored):
            return
        if not self._effect_is_current(effect):
            return
        if isinstance(effect, RenderCloneSelection):
            if effect.source_tree:
                source_filter = self.query_one(
                    "#clone_selection_source_search",
                    Input,
                ).value
                self._rebuild_source_tree(source_filter)
            if effect.target_tree:
                self._rebuild_target_tree()
            return
        if isinstance(effect, ShowCloneSelectionHint):
            self._update_hint(effect.message)
            return
        if isinstance(effect, DismissCloneSelection):
            result = None
            if effect.selected_relative_paths is not None:
                result = ModelCloneSelectionResult(effect.selected_relative_paths)
            self.dismiss(result)

    def _execute_transition(self, transition: CloneSelectionTransition) -> None:
        """Execute the transition owned by this model clone selection dialog."""
        for effect in transition.effects:
            self._execute_effect(effect)

    def _toggle_tree_selection(self, tree_id: str, relative_path: str) -> None:
        """Handle the toggle tree selection operation owned by this model clone selection
dialog."""
        side = self._TREE_SIDES.get(tree_id)
        if side is None:
            return
        self._execute_transition(self._workflow().toggle(side, relative_path))

    def _remove_selected_target_paths(
        self,
        selected_target_paths: tuple[str, ...],
    ) -> bool:
        """Remove the selected target paths owned by this model clone selection dialog."""
        transition = self._workflow().remove(selected_target_paths)
        changed = any(isinstance(effect, RenderCloneSelection) for effect in transition.effects)
        self._execute_transition(transition)
        return changed

    def _update_hint(self, text: str) -> None:
        """Update the hint owned by this model clone selection dialog."""
        _ = text

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this model clone selection dialog."""
        with Vertical(id="model_clone_selection_dialog_container") as container:
            container.border_title = "Clone Selection"
            with (
                ScrollableContainer(id="model_clone_selection_dialog_scroll"),
                Vertical(id="model_clone_selection_dialog_content"),
            ):
                with Horizontal(id="clone_selection_workspace"):
                    with Vertical(id="clone_selection_source_column"):
                        yield Input(
                            placeholder="Search source entries...",
                            id="clone_selection_source_search",
                        )
                        yield CloneTransferTree(
                            "Source model tree",
                            id="clone_selection_source_tree",
                        )
                    with Vertical(id="clone_selection_transfer_column"):
                        yield Button(
                            ">>",
                            id="clone_selection_approve",
                            variant="primary",
                            disabled=True,
                        )
                        yield Button(
                            "<<",
                            id="clone_selection_remove",
                            variant="warning",
                            disabled=True,
                        )
                    with Vertical(id="clone_selection_target_column"):
                        yield Input(
                            placeholder="Search approved entries...",
                            id="clone_selection_target_search",
                        )
                        yield CloneTransferTree(
                            "Approved clone entries",
                            id="clone_selection_target_tree",
                        )
                with Horizontal(id="clone_selection_actions"):
                    yield Button(
                        "Confirm Clone Selection",
                        id="clone_selection_confirm",
                        variant="success",
                        disabled=True,
                    )
                    yield Button(
                        "Cancel",
                        id="clone_selection_cancel",
                        variant="default",
                    )
            yield ModalFooter.from_binding_rows(
                self._FOOTER_BINDINGS,
                [
                    ["shortcut_approve", "shortcut_remove"],
                    ["shortcut_confirm", "shortcut_cancel"],
                    ["open_help_dialog"],
                ],
                id="model_clone_selection_modal_footer",
                label_overrides={
                    "shortcut_approve": "Approve",
                    "shortcut_remove": "Remove",
                    "shortcut_confirm": "Confirm",
                    "shortcut_cancel": "Cancel / Close",
                },
            )

    def on_mount(self) -> None:
        """Handle the mount event for this model clone selection dialog."""
        self._coordinator = CloneSelectionCoordinator.from_source(
            self._source_model_folder_path,
        )
        self._rebuild_source_tree()
        self._rebuild_target_tree()
        self._focus_if_navigable("clone_selection_source_search")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle the input changed event for this model clone selection dialog."""
        if event.input.id == "clone_selection_source_search":
            self._rebuild_source_tree(event.value)
            return
        if event.input.id == "clone_selection_target_search":
            self._rebuild_target_tree()

    def on_tree_node_selected(
        self,
        event: Tree.NodeSelected[str | None],
    ) -> None:
        """Handle the tree node selected event for this model clone selection dialog."""
        tree_widget = event.control or event.node.tree
        tree_id = tree_widget.id
        if tree_id not in self._TREE_SIDES:
            return
        relative_path = event.node.data
        if not isinstance(relative_path, str) or tree_id is None:
            return
        now = monotonic()
        last_path = self._last_tree_select_path.get(tree_id)
        last_ts = self._last_tree_select_ts.get(tree_id, 0.0)
        if last_path == relative_path and (now - last_ts) <= 0.45:
            self._toggle_tree_selection(tree_id, relative_path)
            self._last_tree_select_path.pop(tree_id, None)
            self._last_tree_select_ts.pop(tree_id, None)
            return
        self._last_tree_select_path[tree_id] = relative_path
        self._last_tree_select_ts[tree_id] = now

    def on_key(self, event: Key) -> None:
        """Handle the key event for this model clone selection dialog."""
        focused_id = self.focused.id if self.focused is not None else None
        tree_ids = set(self._TREE_SIDES)
        if focused_id in tree_ids and event.key == "space":
            tree = self.query_one(f"#{focused_id}", CloneTransferTree)
            selected_node = tree.cursor_node or tree.root
            if isinstance(selected_node.data, str):
                self._toggle_tree_selection(focused_id, selected_node.data)
                event.stop()
                event.prevent_default()
            return
        if focused_id == "clone_selection_target_tree" and event.key == "delete":
            target_tree = self.query_one(
                "#clone_selection_target_tree",
                CloneTransferTree,
            )
            selected_node = target_tree.cursor_node or target_tree.root
            if isinstance(selected_node.data, str):
                self._remove_selected_target_paths((selected_node.data,))
            event.stop()
            event.prevent_default()
            return
        if event.key not in {"up", "left", "down", "right"}:
            return
        if focused_id in {
            "clone_selection_source_search",
            "clone_selection_target_search",
        } and event.key not in {"down", "left", "right"}:
            return
        if focused_id in tree_ids and event.key in {"up", "down"}:
            return
        if self._move_focus_in_direction(event.key):
            event.stop()
            event.prevent_default()

    def _press_button_if_enabled(self, button_id: str) -> None:
        """Return whether press button if enabled applies to this model clone selection dialog."""
        button = self.query_one(f"#{button_id}", Button)
        if button.display and not button.disabled:
            button.press()

    def action_shortcut_approve(self) -> None:
        """Handle the shortcut approve user action for this model clone selection dialog."""
        self._press_button_if_enabled("clone_selection_approve")

    def action_shortcut_remove(self) -> None:
        """Handle the shortcut remove user action for this model clone selection dialog."""
        focused_id = self.focused.id if self.focused is not None else None
        if focused_id == "clone_selection_target_tree":
            target_tree = self.query_one(
                "#clone_selection_target_tree",
                CloneTransferTree,
            )
            selected_node = target_tree.cursor_node or target_tree.root
            if isinstance(selected_node.data, str):
                self._remove_selected_target_paths((selected_node.data,))
                return
        self._press_button_if_enabled("clone_selection_remove")

    def action_shortcut_confirm(self) -> None:
        """Handle the shortcut confirm user action for this model clone selection dialog."""
        self._press_button_if_enabled("clone_selection_confirm")

    def action_shortcut_cancel(self) -> None:
        """Handle the shortcut cancel user action for this model clone selection dialog."""
        self._execute_transition(self._workflow().cancel())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this model clone selection dialog."""
        button_id = event.button.id
        if button_id == "clone_selection_cancel":
            self._execute_transition(self._workflow().cancel())
            return
        if button_id == "clone_selection_approve":
            self._execute_transition(self._workflow().approve())
            return
        if button_id == "clone_selection_remove":
            self._execute_transition(self._workflow().remove_selected())
            return
        if button_id == "clone_selection_confirm":
            self._execute_transition(self._workflow().confirm())
