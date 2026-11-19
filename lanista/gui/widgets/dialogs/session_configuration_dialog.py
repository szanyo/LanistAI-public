# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for session configuration."""

from dataclasses import dataclass
from typing import Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Tree
from textual.widgets._tree import TreeNode

from lanista.application import LanistAIApplication
from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.widgets.dialogs.button_templates import primary_button
from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.fragments.selection_details_panel import SelectionDetailsPanel
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.gui.utils.timestamps import format_local_timestamp
from lanista.locations import get_session_configuration_dialog_css_path


@dataclass
class SessionConfigurationDialogResult:
    """Report the result of session configuration dialog."""

    action: str
    session_name: Optional[str]


class SessionConfigurationDialog(
    DirectionalFocusMixin,
    HelpEnabledModalMixin,
    ModalScreen[Optional[SessionConfigurationDialogResult]],
):
    """Present session configuration in the Textual interface."""

    _HELP_TOPIC = "session_configuration"
    CSS_PATH = get_session_configuration_dialog_css_path()
    _APPLICATION = LanistAIApplication()

    BINDINGS = [
        Binding("f3,e", "shortcut_edit", "Edit", key_display="f3"),
        Binding("delete", "shortcut_delete", "Delete", key_display="del"),
        Binding("enter", "shortcut_confirm", "Confirm"),
        Binding("escape", "dismiss(None)", "Cancel", key_display="esc"),
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
    ]

    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "session_configuration_search": {
            "down": "session_configuration_tree",
            "right": "session_configuration_tree",
        },
        "session_configuration_tree": {
            "up": "session_configuration_search",
            "down": "session_configuration_edit",
            "right": "session_configuration_edit",
        },
        "session_configuration_edit": {
            "up": "session_configuration_tree",
            "down": "session_configuration_delete",
            "right": "session_configuration_delete",
        },
        "session_configuration_delete": {
            "up": "session_configuration_edit",
            "down": "session_configuration_confirm",
            "right": "session_configuration_confirm",
        },
        "session_configuration_confirm": {
            "up": "session_configuration_delete",
            "left": "session_configuration_delete",
            "right": "session_configuration_cancel",
        },
        "session_configuration_cancel": {
            "up": "session_configuration_delete",
            "left": "session_configuration_confirm",
        },
    }

    def __init__(
        self,
        *,
        session_options: list[tuple[str, str]],
        selected_session_name: Optional[str],
        allow_delete: bool = True,
        allow_edit: bool = False,
    ) -> None:
        """Initialize the session configuration dialog and its owned state."""
        super().__init__()
        self._session_options = list(session_options)
        self._selected_session_name = selected_session_name
        self._selected_search = ""
        self._allow_delete = allow_delete
        self._allow_edit = allow_edit

    def on_key(self, event: Key) -> None:
        """Handle the key event for this session configuration dialog."""
        if event.key in {"up", "left", "down", "right"} and self._move_focus_in_direction(event.key):
            event.stop()
            event.prevent_default()

    def action_shortcut_confirm(self) -> None:
        """Handle the shortcut confirm user action for this session configuration dialog."""
        button = self.query_one("#session_configuration_confirm", Button)
        if button.display and not button.disabled:
            button.press()

    def action_shortcut_delete(self) -> None:
        """Handle the shortcut delete user action for this session configuration dialog."""
        button = self.query_one("#session_configuration_delete", Button)
        if button.display and not button.disabled:
            button.press()

    def action_shortcut_edit(self) -> None:
        """Handle the shortcut edit user action for this session configuration dialog."""
        button = self.query_one("#session_configuration_edit", Button)
        if button.display and not button.disabled:
            button.press()

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this session configuration dialog."""
        with Vertical(id="session_configuration_dialog_container") as container:
            container.border_title = "Session Configuration"
            with ScrollableContainer(id="session_configuration_dialog_scroll"):
                with Vertical(id="session_configuration_dialog_content"):
                    yield Input(placeholder="Search sessions...", id="session_configuration_search")
                    tree = Tree("Sessions", id="session_configuration_tree")
                    tree.show_root = False
                    tree.show_guides = False
                    yield tree
                    yield SelectionDetailsPanel(
                        title="Selected Session",
                        empty_message="No session selected.",
                        id="session_configuration_details",
                    )
                    yield primary_button(
                        "Edit",
                        id="session_configuration_edit",
                        disabled=(not self._allow_edit or self._selected_session_name is None),
                    )
                    yield Button(
                        "Delete",
                        id="session_configuration_delete",
                        variant="error",
                        disabled=(not self._allow_delete or self._selected_session_name is None),
                    )
                    with Horizontal(id="session_configuration_actions"):
                        yield Button("Confirm", id="session_configuration_confirm", variant="success", disabled=True)
                        yield Button("Cancel", id="session_configuration_cancel", variant="default")
            yield ModalFooter.from_binding_rows(
                self.BINDINGS,
                [
                    ["shortcut_edit", "shortcut_delete"],
                    ["shortcut_confirm", "dismiss(None)"],
                    ["open_help_dialog"],
                ],
                id="session_configuration_modal_footer",
                label_overrides={
                    "shortcut_edit": "Edit",
                    "shortcut_confirm": "Confirm",
                    "shortcut_delete": "Delete",
                    "dismiss(None)": "Cancel / Close",
                },
            )

    def _rebuild_tree(self, filter_text: str = "") -> None:
        """Rebuild the tree owned by this session configuration dialog."""
        tree = self.query_one("#session_configuration_tree", Tree)
        tree.clear()
        tree.root.expand()
        tree.root.allow_expand = False

        normalized_filter = str(filter_text or "").strip().lower()
        visible_entries = [
            (display, value)
            for display, value in self._session_options
            if not normalized_filter or normalized_filter in display.lower() or normalized_filter in value.lower()
        ]

        if not visible_entries:
            empty_node = tree.root.add_leaf(Text("No matching sessions.", style="dim"), data=None)
            empty_node.allow_expand = False
            self._selected_session_name = None
            self.query_one("#session_configuration_confirm", Button).disabled = True
            self.query_one("#session_configuration_delete", Button).disabled = True
            self.query_one("#session_configuration_edit", Button).disabled = True
            self.query_one("#session_configuration_details", SelectionDetailsPanel).show_message("No matching sessions.")
            return

        selected_node: Optional[TreeNode] = None
        for display, value in visible_entries:
            node = tree.root.add_leaf(Text(display), data=value)
            node.allow_expand = False
            if value == self._selected_session_name:
                selected_node = node

        if selected_node is None and tree.root.children:
            selected_node = tree.root.children[0]

        if selected_node is not None and isinstance(selected_node.data, str):
            self._selected_session_name = selected_node.data
            tree.select_node(selected_node)
            self.query_one("#session_configuration_confirm", Button).disabled = False
            self.query_one("#session_configuration_delete", Button).disabled = not self._allow_delete
            self.query_one("#session_configuration_edit", Button).disabled = not self._allow_edit
            self._update_details(self._selected_session_name)

    def _update_details(self, session_uuid: Optional[str]) -> None:
        """Update the details owned by this session configuration dialog."""
        panel = self.query_one("#session_configuration_details", SelectionDetailsPanel)
        detail = (
            self._APPLICATION
            .get_application_services()
            .sessions.selection
            .build_session_selection_detail(session_uuid)
        )
        if detail is None:
            panel.show_message("No session selected.")
            return

        child_suffix = "child" if detail.child_count == 1 else "children"
        rows = [
            SelectionDetailsPanel.DetailRow("Session", detail.display_name),
            SelectionDetailsPanel.DetailRow("UUID", detail.session_uuid),
            SelectionDetailsPanel.DetailRow("Framework", detail.framework_uuid),
            SelectionDetailsPanel.DetailRow("Parent", detail.parent_display_name),
            SelectionDetailsPanel.DetailRow("Forked From", detail.fork_source_session_display_name),
            SelectionDetailsPanel.DetailRow("Source Checkpoint", detail.fork_source_checkpoint_name or "-"),
            SelectionDetailsPanel.DetailRow("Step Offset", str(detail.base_step_offset)),
            SelectionDetailsPanel.DetailRow("Children", f"{detail.child_count} {child_suffix}"),
            SelectionDetailsPanel.DetailRow("Checkpoints", str(detail.checkpoint_count)),
            SelectionDetailsPanel.DetailRow("Latest", detail.latest_checkpoint_name or "-"),
            SelectionDetailsPanel.DetailRow("Created", format_local_timestamp(detail.created_at_raw)),
            SelectionDetailsPanel.DetailRow("Updated", format_local_timestamp(detail.modified_at_raw)),
        ]
        if detail.description:
            rows.append(SelectionDetailsPanel.DetailRow("Description", detail.description))
        panel.show_rows(rows)

    def on_mount(self) -> None:
        """Handle the mount event for this session configuration dialog."""
        self._rebuild_tree()
        self._focus_if_navigable("session_configuration_search")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle the input changed event for this session configuration dialog."""
        if event.input.id != "session_configuration_search":
            return
        self._selected_search = event.value
        self._rebuild_tree(event.value)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle the tree node selected event for this session configuration dialog."""
        self._selected_session_name = event.node.data if isinstance(event.node.data, str) else None
        self.query_one("#session_configuration_confirm", Button).disabled = self._selected_session_name is None
        self.query_one("#session_configuration_delete", Button).disabled = (
            (not self._allow_delete) or self._selected_session_name is None
        )
        self.query_one("#session_configuration_edit", Button).disabled = (
            (not self._allow_edit) or self._selected_session_name is None
        )
        self._update_details(self._selected_session_name)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this session configuration dialog."""
        if event.button.id == "session_configuration_confirm":
            self.dismiss(SessionConfigurationDialogResult("select", self._selected_session_name))
            return
        if event.button.id == "session_configuration_delete":
            self.dismiss(SessionConfigurationDialogResult("delete", self._selected_session_name))
            return
        if event.button.id == "session_configuration_edit":
            self.dismiss(SessionConfigurationDialogResult("edit", self._selected_session_name))
            return
        self.dismiss(None)
