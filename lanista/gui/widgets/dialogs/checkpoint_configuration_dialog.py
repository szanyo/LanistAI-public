# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for checkpoint configuration."""

import asyncio
from dataclasses import dataclass

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Tree
from textual.widgets._tree import TreeNode

from lanista.application import LanistAIApplication
from lanista.gui.services.async_collection_loader import AsyncCollectionLoader
from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.utils.timestamps import format_local_timestamp
from lanista.gui.utils.tree_nodes import first_string_data_node
from lanista.gui.widgets.dialogs.button_templates import primary_button
from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.fragments.selection_details_panel import SelectionDetailsPanel
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.locations import get_checkpoint_configuration_dialog_css_path
from lanista.reinforcengine.rl_model.checkpoints.name_metadata import extract_checkpoint_steps


@dataclass
class CheckpointConfigurationDialogResult:
    """Report the result of checkpoint configuration dialog."""

    action: str
    checkpoint_name: str | None
    renamed_checkpoint_name: str | None = None


@dataclass(frozen=True)
class _CheckpointListItem:
    """Carry the data required for checkpoint list item."""

    label: str
    value: str
    step_count: int
    search_text: str


class CheckpointConfigurationDialog(
    DirectionalFocusMixin,
    HelpEnabledModalMixin,
    ModalScreen[CheckpointConfigurationDialogResult | None],
):
    """Present checkpoint configuration in the Textual interface."""

    _HELP_TOPIC = "checkpoint_configuration"
    CSS_PATH = get_checkpoint_configuration_dialog_css_path()
    _APPLICATION = LanistAIApplication()
    _MAX_RENDERED_CHECKPOINTS = 100

    BINDINGS = [
        Binding("f3,e", "shortcut_edit", "Edit", key_display="f3"),
        Binding("r", "shortcut_rename", "Rename"),
        Binding("delete", "shortcut_delete", "Delete", key_display="del"),
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
        Binding("enter", "shortcut_confirm", "Confirm"),
        Binding("escape", "dismiss(None)", "Cancel", key_display="esc"),
    ]

    _FRESH_TRAINING_VALUE = "__fresh_training__"

    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "checkpoint_configuration_search": {
            "down": "checkpoint_configuration_tree",
            "right": "checkpoint_configuration_tree",
        },
        "checkpoint_configuration_tree": {
            "up": "checkpoint_configuration_search",
            "down": "checkpoint_configuration_edit",
            "right": "checkpoint_configuration_edit",
        },
        "checkpoint_configuration_edit": {
            "up": "checkpoint_configuration_tree",
            "down": "checkpoint_configuration_delete",
            "right": "checkpoint_configuration_delete",
        },
        "checkpoint_configuration_delete": {
            "up": "checkpoint_configuration_edit",
            "left": "checkpoint_configuration_edit",
            "right": "checkpoint_configuration_confirm",
            "down": "checkpoint_configuration_confirm",
        },
        "checkpoint_configuration_rename": {
            "up": "checkpoint_configuration_edit",
            "down": "checkpoint_configuration_delete",
            "right": "checkpoint_configuration_delete",
        },
        "checkpoint_configuration_confirm": {
            "up": "checkpoint_configuration_delete",
            "left": "checkpoint_configuration_delete",
            "right": "checkpoint_configuration_cancel",
        },
        "checkpoint_configuration_cancel": {
            "up": "checkpoint_configuration_delete",
            "left": "checkpoint_configuration_confirm",
        },
    }

    def __init__(
        self,
        *,
        session_name: str | None,
        checkpoint_options: list[tuple[str, str]],
        selected_checkpoint_name: str | None,
        allow_fresh_training: bool,
        fresh_selection_label: str = "Start fresh (no checkpoint)",
        fresh_selection_mode_label: str = "Start fresh",
        fresh_selection_search_text: str = "start fresh no checkpoint fresh training",
        allow_delete: bool = True,
        allow_rename: bool = False,
        allow_edit: bool = False,
    ) -> None:
        """Initialize the checkpoint configuration dialog and its owned state."""
        super().__init__()
        self._session_name = session_name
        self._allow_fresh_training = allow_fresh_training
        self._fresh_selection_label = fresh_selection_label
        self._fresh_selection_mode_label = fresh_selection_mode_label
        self._fresh_selection_search_text = fresh_selection_search_text
        self._allow_delete = allow_delete
        self._allow_rename = allow_rename
        self._allow_edit = allow_edit
        self._checkpoint_options = list(checkpoint_options)
        self._selected_checkpoint_name = selected_checkpoint_name if selected_checkpoint_name is not None else (
            self._FRESH_TRAINING_VALUE if allow_fresh_training else None
        )
        self._known_checkpoint_values = {value for _, value in self._checkpoint_options}
        self._checkpoint_items: tuple[_CheckpointListItem, ...] = ()
        self._loading_checkpoints = False
        self._checkpoint_load_error: BaseException | None = None
        self._checkpoint_loader: AsyncCollectionLoader[_CheckpointListItem] = AsyncCollectionLoader(
            self._build_checkpoint_items
        )
        self._checkpoint_load_task: asyncio.Task | None = None

    def _build_checkpoint_items(self) -> list[_CheckpointListItem]:
        """Build the checkpoint items required by this checkpoint configuration dialog."""
        items: list[_CheckpointListItem] = []
        for label, value in self._checkpoint_options:
            checkpoint_name = str(value or "").strip()
            if not checkpoint_name:
                continue
            display_label = str(label or checkpoint_name).strip() or checkpoint_name
            step_count = extract_checkpoint_steps(checkpoint_name)
            search_text = " ".join(
                [
                    display_label,
                    checkpoint_name,
                    str(step_count) if step_count >= 0 else "",
                ]
            ).lower()
            items.append(
                _CheckpointListItem(
                    label=display_label,
                    value=checkpoint_name,
                    step_count=step_count,
                    search_text=search_text,
                )
            )
        items.sort(key=lambda item: (item.step_count, item.label.lower(), item.value), reverse=True)
        return items

    def _show_tree_message(self, message: str) -> None:
        """Show the tree message owned by this checkpoint configuration dialog."""
        tree = self.query_one("#checkpoint_configuration_tree", Tree)
        tree.clear()
        tree.root.expand()
        node = tree.root.add_leaf(Text(message, style="dim"), data=None)
        node.allow_expand = False
        self.query_one("#checkpoint_configuration_details", SelectionDetailsPanel).show_message(message)
        self._update_action_buttons()

    def _start_checkpoint_load(self) -> None:
        """Start the checkpoint load workflow owned by this checkpoint configuration dialog."""
        self._loading_checkpoints = True
        self._checkpoint_load_error = None
        request_id = self._checkpoint_loader.begin_request()
        self._show_tree_message("Loading checkpoints...")

        async def _load() -> None:
            """Load and return the state owned by this checkpoint configuration dialog."""
            await asyncio.sleep(0)
            try:
                items = tuple(self._build_checkpoint_items())
                error = None
            except BaseException as exc:
                items = ()
                error = exc
            if not self._checkpoint_loader.is_current(request_id):
                return
            self._loading_checkpoints = False
            self._checkpoint_load_error = error
            self._checkpoint_items = items if error is None else ()
            try:
                self._rebuild_tree(self.query_one("#checkpoint_configuration_search", Input).value)
            except NoMatches:
                return

        self._checkpoint_load_task = asyncio.create_task(_load())

    def on_key(self, event: Key) -> None:
        """Handle the key event for this checkpoint configuration dialog."""
        if event.key in {"up", "left", "down", "right"} and self._move_focus_in_direction(event.key):
            event.stop()
            event.prevent_default()

    def action_shortcut_confirm(self) -> None:
        """Handle the shortcut confirm user action for this checkpoint configuration dialog."""
        button = self.query_one("#checkpoint_configuration_confirm", Button)
        if button.display and not button.disabled:
            button.press()

    def action_shortcut_delete(self) -> None:
        """Handle the shortcut delete user action for this checkpoint configuration dialog."""
        button = self.query_one("#checkpoint_configuration_delete", Button)
        if button.display and not button.disabled:
            button.press()

    def action_shortcut_edit(self) -> None:
        """Handle the shortcut edit user action for this checkpoint configuration dialog."""
        button = self.query_one("#checkpoint_configuration_edit", Button)
        if button.display and not button.disabled:
            button.press()

    def action_shortcut_rename(self) -> None:
        """Handle the shortcut rename user action for this checkpoint configuration dialog."""
        button = self.query_one("#checkpoint_configuration_rename", Button)
        if button.display and not button.disabled:
            button.press()

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this checkpoint configuration dialog."""
        with Vertical(id="checkpoint_configuration_dialog_container") as container:
            container.border_title = "Checkpoint Configuration"
            with (
                ScrollableContainer(id="checkpoint_configuration_dialog_scroll"),
                Vertical(id="checkpoint_configuration_dialog_content"),
            ):
                    yield Input(placeholder="Search checkpoints...", id="checkpoint_configuration_search")
                    tree = Tree("Checkpoints", id="checkpoint_configuration_tree")
                    tree.show_root = False
                    tree.show_guides = True
                    yield tree
                    yield SelectionDetailsPanel(
                        title="Selected Checkpoint",
                        empty_message="No checkpoint selected.",
                        id="checkpoint_configuration_details",
                    )
                    yield primary_button(
                        "Edit",
                        id="checkpoint_configuration_edit",
                        disabled=True,
                    )
                    yield Button(
                        "Delete",
                        id="checkpoint_configuration_delete",
                        variant="error",
                        disabled=True,
                    )
                    rename_input = Input(
                        placeholder="New checkpoint name...",
                        id="checkpoint_configuration_rename_input",
                    )
                    rename_input.display = self._allow_rename
                    yield rename_input
                    rename_button = Button(
                        "Rename",
                        id="checkpoint_configuration_rename",
                        variant="primary",
                        disabled=True,
                    )
                    rename_button.display = self._allow_rename
                    yield rename_button
                    with Horizontal(id="checkpoint_configuration_actions"):
                        yield Button("Confirm", id="checkpoint_configuration_confirm", variant="success", disabled=True)
                        yield Button("Cancel", id="checkpoint_configuration_cancel", variant="default")
            footer_rows = [
                ["shortcut_edit", "shortcut_delete", "open_help_dialog"],
                ["shortcut_confirm", "dismiss(None)"],
            ]
            if self._allow_rename:
                footer_rows[0].insert(0, "shortcut_rename")
            yield ModalFooter.from_binding_rows(
                self.BINDINGS,
                footer_rows,
                id="checkpoint_configuration_modal_footer",
                label_overrides={
                    "shortcut_edit": "Edit",
                    "shortcut_rename": "Rename",
                    "shortcut_delete": "Delete",
                    "shortcut_confirm": "Confirm",
                    "dismiss(None)": "Cancel / Close",
                },
            )

    def _update_action_buttons(self) -> None:
        """Update the action buttons owned by this checkpoint configuration dialog."""
        selected_checkpoint = self._selected_checkpoint_name
        can_target_checkpoint = bool(selected_checkpoint and selected_checkpoint != self._FRESH_TRAINING_VALUE)

        self.query_one("#checkpoint_configuration_confirm", Button).disabled = selected_checkpoint is None
        self.query_one("#checkpoint_configuration_delete", Button).disabled = (
            (not self._allow_delete) or (not can_target_checkpoint)
        )
        self.query_one("#checkpoint_configuration_edit", Button).disabled = (
            (not self._allow_edit) or (not can_target_checkpoint)
        )

        rename_input = self.query_one("#checkpoint_configuration_rename_input", Input)
        rename_button = self.query_one("#checkpoint_configuration_rename", Button)
        rename_candidate = str(rename_input.value or "").strip()
        rename_button.disabled = (
            (not self._allow_rename)
            or (not can_target_checkpoint)
            or (not rename_candidate)
            or rename_candidate == selected_checkpoint
        )

    def _rebuild_tree(self, filter_text: str = "") -> None:
        """Rebuild the tree owned by this checkpoint configuration dialog."""
        tree = self.query_one("#checkpoint_configuration_tree", Tree)
        tree.clear()
        tree.root.expand()

        if self._loading_checkpoints:
            self._show_tree_message("Loading checkpoints...")
            return
        if self._checkpoint_load_error is not None:
            self._show_tree_message("Failed to load checkpoints.")
            return

        normalized_filter = str(filter_text or "").strip().lower()
        checkpoint_entries = [
            item
            for item in self._checkpoint_items
            if item.value in self._known_checkpoint_values
            and (not normalized_filter or normalized_filter in item.search_text)
        ]

        if self._allow_fresh_training and (
            not normalized_filter or normalized_filter in self._fresh_selection_search_text
        ):
            fresh_node = tree.root.add_leaf(
                Text(self._fresh_selection_label, style="bold green"),
                data=self._FRESH_TRAINING_VALUE,
            )
            fresh_node.allow_expand = False

        if not checkpoint_entries and not tree.root.children:
            empty_node = tree.root.add_leaf(Text("No matching checkpoints.", style="dim"), data=None)
            empty_node.allow_expand = False
            self._selected_checkpoint_name = None
            self._update_action_buttons()
            self.query_one("#checkpoint_configuration_details", SelectionDetailsPanel).show_message("No matching checkpoints.")
            return

        selected_node: TreeNode | None = None
        rendered_entries = checkpoint_entries[: self._MAX_RENDERED_CHECKPOINTS]
        for entry in rendered_entries:
            label = Text(entry.label)
            if entry.step_count >= 0:
                label.append(f"  steps={entry.step_count}", style="dim")
            node = tree.root.add_leaf(label, data=entry.value)
            node.allow_expand = False
            if entry.value == self._selected_checkpoint_name:
                selected_node = node
        remaining_count = len(checkpoint_entries) - len(rendered_entries)
        if remaining_count > 0:
            more_node = tree.root.add_leaf(
                Text(f"Showing first {len(rendered_entries)} checkpoints. Narrow search for {remaining_count} more.", style="dim"),
                data=None,
            )
            more_node.allow_expand = False

        if selected_node is None and tree.root.children:
            selected_node = first_string_data_node(tree.root)

        if selected_node is not None and isinstance(selected_node.data, str):
            self._selected_checkpoint_name = selected_node.data
            tree.select_node(selected_node)
            self._update_action_buttons()
            self._update_details(self._selected_checkpoint_name)

    def _update_details(self, checkpoint_name: str | None) -> None:
        """Update the details owned by this checkpoint configuration dialog."""
        panel = self.query_one("#checkpoint_configuration_details", SelectionDetailsPanel)
        effective_checkpoint = None if checkpoint_name == self._FRESH_TRAINING_VALUE else checkpoint_name
        detail = (
            self._APPLICATION
            .get_application_services()
            .catalog.checkpoints
            .build_selection_detail(self._session_name, effective_checkpoint)
        )
        if detail.checkpoint_kind == "fresh":
            panel.show_rows(
                [
                    SelectionDetailsPanel.DetailRow("Mode", self._fresh_selection_mode_label),
                    SelectionDetailsPanel.DetailRow("Session", detail.session_display_name),
                    SelectionDetailsPanel.DetailRow("Framework", detail.framework_uuid),
                    SelectionDetailsPanel.DetailRow("Checkpoint", "None"),
                ]
            )
            return

        rows = [
            SelectionDetailsPanel.DetailRow("Checkpoint", detail.checkpoint_name or "-"),
            SelectionDetailsPanel.DetailRow("Session", detail.session_display_name),
            SelectionDetailsPanel.DetailRow("Framework", detail.framework_uuid),
            SelectionDetailsPanel.DetailRow("Kind", detail.checkpoint_kind),
            SelectionDetailsPanel.DetailRow("Steps", str(detail.step_count) if detail.step_count is not None and detail.step_count >= 0 else "-"),
            SelectionDetailsPanel.DetailRow("Scenario", str(detail.scenario_index) if detail.scenario_index is not None else "-"),
            SelectionDetailsPanel.DetailRow("Curriculum", detail.curriculum_label),
            SelectionDetailsPanel.DetailRow("Parent", detail.parent_checkpoint_name or "-"),
            SelectionDetailsPanel.DetailRow("Annotation", detail.annotation or "-"),
            SelectionDetailsPanel.DetailRow("Saved", format_local_timestamp(detail.saved_at_raw)),
            SelectionDetailsPanel.DetailRow("Exists", "yes" if detail.exists else "no"),
        ]
        panel.show_rows(rows)

    def on_mount(self) -> None:
        """Handle the mount event for this checkpoint configuration dialog."""
        self._start_checkpoint_load()
        self._focus_if_navigable("checkpoint_configuration_search")

    def on_unmount(self) -> None:
        """Handle the unmount event for this checkpoint configuration dialog."""
        if self._checkpoint_load_task is not None and not self._checkpoint_load_task.done():
            self._checkpoint_load_task.cancel()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle the input changed event for this checkpoint configuration dialog."""
        if event.input.id == "checkpoint_configuration_search":
            self._rebuild_tree(event.value)
            return

        if event.input.id == "checkpoint_configuration_rename_input":
            self._update_action_buttons()

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle the tree node selected event for this checkpoint configuration dialog."""
        self._selected_checkpoint_name = event.node.data if isinstance(event.node.data, str) else None
        self._update_action_buttons()
        self._update_details(self._selected_checkpoint_name)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this checkpoint configuration dialog."""
        if event.button.id == "checkpoint_configuration_confirm":
            checkpoint_name = self._selected_checkpoint_name
            if checkpoint_name == self._FRESH_TRAINING_VALUE:
                checkpoint_name = None
            self.dismiss(CheckpointConfigurationDialogResult("select", checkpoint_name))
            return
        if event.button.id == "checkpoint_configuration_delete":
            checkpoint_name = self._selected_checkpoint_name
            if checkpoint_name == self._FRESH_TRAINING_VALUE:
                checkpoint_name = None
            self.dismiss(CheckpointConfigurationDialogResult("delete", checkpoint_name, None))
            return
        if event.button.id == "checkpoint_configuration_edit":
            checkpoint_name = self._selected_checkpoint_name
            if checkpoint_name == self._FRESH_TRAINING_VALUE:
                checkpoint_name = None
            self.dismiss(CheckpointConfigurationDialogResult("edit", checkpoint_name, None))
            return
        if event.button.id == "checkpoint_configuration_rename":
            checkpoint_name = self._selected_checkpoint_name
            renamed_checkpoint_name = str(self.query_one("#checkpoint_configuration_rename_input", Input).value or "").strip()
            if checkpoint_name == self._FRESH_TRAINING_VALUE:
                checkpoint_name = None
            self.dismiss(CheckpointConfigurationDialogResult("rename", checkpoint_name, renamed_checkpoint_name))
            return
        self.dismiss(None)
