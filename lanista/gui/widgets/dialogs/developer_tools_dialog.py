# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for developer tools."""

from dataclasses import dataclass
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Select, Static, Tree

from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.widgets.dialogs.button_templates import cancel_button, selector_button, selector_label
from lanista.gui.widgets.fragments.selection_details_panel import SelectionDetailsPanel
from lanista.gui.widgets.modal_footer import ModalFooter


@dataclass(frozen=True)
class DesignAtlasEntry:
    """Describe design atlas data."""

    code: str
    name: str
    summary: str
    behavior: str
    used_in: tuple[str, ...]


def build_atlas_entry(
    code: str,
    name: str,
    summary: str,
    behavior: str,
    used_in: tuple[str, ...],
) -> DesignAtlasEntry:
    """Build the atlas entry required by this developer tools dialog module."""
    return DesignAtlasEntry(
        code=code,
        name=name,
        summary=summary,
        behavior=behavior,
        used_in=used_in,
    )


class DeveloperToolsDialog(DirectionalFocusMixin, ModalScreen[None]):
    """Present developer tools in the Textual interface."""

    BINDINGS = [
        Binding("escape", "dismiss(None)", "Close", key_display="esc"),
    ]

    DEFAULT_CSS = """
    DeveloperToolsDialog {
        overflow-x: hidden;
        overflow-y: hidden;
    }

    #developer_tools_dialog_container {
        width: 96;
        max-width: 96;
        height: 54;
        min-height: 46;
        padding: 1 3;
        border: thick $background 80%;
        border-title-color: $accent;
        border-title-background: $surface;
        border-title-style: bold;
        background: $surface;
        overflow-x: hidden;
        overflow-y: hidden;
    }

    #developer_tools_dialog_scroll {
        width: 1fr;
        height: 1fr;
        min-height: 0;
        overflow-x: hidden;
        overflow-y: auto;
    }

    #developer_tools_dialog_content {
        width: 1fr;
        height: auto;
        min-height: 0;
    }

    #developer_tools_family_select {
        width: 1fr;
        height: 3;
        min-height: 3;
        margin-bottom: 1;
    }

    #developer_tools_family_details {
        height: 4;
        min-height: 4;
        margin-bottom: 1;
    }

    #developer_tools_family_details > #selection_details_panel_body {
        height: 4;
        min-height: 4;
    }

    #developer_tools_template_tree {
        width: 1fr;
        height: 11;
        min-height: 11;
        margin-top: 1;
        padding: 0 1;
        overflow-x: auto;
        overflow-y: auto;
        border: round $primary-lighten-1 50%;
        background: $surface;
    }

    #developer_tools_template_tree > .tree--cursor {
        color: $text;
        background: $primary 20%;
        text-style: bold;
    }

    #developer_tools_template_tree > .tree--highlight-line {
        background: $primary 10%;
    }

    #developer_tools_details {
        margin-top: 1;
    }

    #developer_tools_preview_row {
        width: 1fr;
        height: auto;
        min-height: 3;
        margin-top: 1;
    }

    #developer_tools_button_preview,
    #developer_tools_textbox_preview,
    #developer_tools_selector_preview,
    #developer_tools_static_preview,
    #developer_tools_tree_preview,
    #developer_tools_lineage_preview,
    #developer_tools_details_preview,
    #developer_tools_status_preview,
    #developer_tools_transfer_preview,
    #developer_tools_action_preview,
    #developer_tools_footer_preview {
        width: 1fr;
    }

    #developer_tools_button_preview,
    #developer_tools_textbox_preview,
    #developer_tools_selector_preview {
        height: 3;
        min-height: 3;
        content-align: center middle;
    }

    #developer_tools_tree_preview,
    #developer_tools_lineage_preview,
    #developer_tools_status_preview,
    #developer_tools_transfer_preview {
        height: 7;
        min-height: 7;
        padding: 0 1;
        overflow-x: auto;
        overflow-y: auto;
        border: round $primary-lighten-1 50%;
        background: $surface;
    }

    #developer_tools_details_preview {
        height: 7;
        min-height: 7;
    }

    #developer_tools_details_preview > #selection_details_panel_body {
        height: 7;
        min-height: 7;
    }

    #developer_tools_action_preview {
        height: 3;
        min-height: 3;
    }

    #developer_tools_action_preview > Button {
        width: 1fr;
        margin-right: 1;
    }

    #developer_tools_action_preview > Button:last-child {
        margin-right: 0;
    }

    #developer_tools_footer_preview {
        height: 4;
        min-height: 4;
        border-top: solid $panel;
        padding-top: 1;
    }

    #developer_tools_static_preview {
        height: 5;
        min-height: 5;
        border: round $primary-lighten-1 50%;
        background: $surface;
        padding: 0 1;
        content-align: left middle;
    }

    #developer_tools_actions {
        width: 1fr;
        height: 3;
        min-height: 3;
        margin-top: 1;
        margin-bottom: 1;
    }

    #developer_tools_cancel {
        width: 100%;
        min-width: 100%;
        height: 3;
        min-height: 3;
        content-align: center middle;
    }

    """

    _DEMOS = [
        ("Button Atlas", "buttons"),
        ("TextBox Atlas", "textboxes"),
        ("Selector Atlas", "selectors"),
        ("Tree Atlas", "trees"),
        ("Details Panel Atlas", "details"),
        ("Lineage Atlas", "lineage"),
        ("Action Row Atlas", "actions"),
        ("Modal Footer Atlas", "footers"),
        ("Dialog Atlas", "dialogs"),
        ("Status Atlas", "status"),
        ("Transfer Atlas", "transfer"),
    ]

    _TEMPLATES: dict[str, list[DesignAtlasEntry]] = {
        "buttons": [
            build_atlas_entry(
                "BT-101",
                "Selector Button",
                "Opens a focused selection dialog while preserving the current context.",
                "Uses the shared selector button template with a label, separator, and selected value.",
                ("Configuration dialog", "Edit Target dialog"),
            ),
            build_atlas_entry(
                "BT-102",
                "Primary Action Button",
                "Starts a primary navigation or creation action.",
                "Uses the standard primary color treatment for available non-final actions.",
                ("Model Context dialog", "Model Detail dialog", "Edit Target dialog"),
            ),
            build_atlas_entry(
                "BT-103",
                "Success Commit Button",
                "Commits a completed configuration or save operation.",
                "Uses the success treatment only when the action writes or confirms state.",
                ("Session Configuration dialog", "Checkpoint Configuration dialog", "Model Detail dialog"),
            ),
            build_atlas_entry(
                "BT-104",
                "Warning Utility Button",
                "Performs a reversible helper action that changes the current working set.",
                "Uses warning color for utility changes that need attention but are not destructive.",
                ("Model Context dialog", "Session Switch dialog", "Comparison Pack Builder dialog"),
            ),
            build_atlas_entry(
                "BT-105",
                "Destructive Button",
                "Deletes or removes persisted data after explicit user intent.",
                "Uses the destructive treatment for delete actions and should pair with confirmation when needed.",
                ("Session Configuration dialog", "Checkpoint Configuration dialog", "Model Detail dialog"),
            ),
        ],
        "textboxes": [
            build_atlas_entry(
                "TB-201",
                "Search TextBox",
                "Filters a tree or list without changing the selected record directly.",
                "Updates matching results as the user types and keeps selection details in sync.",
                ("Session Configuration dialog", "Checkpoint Configuration dialog", "Load Model dialog"),
            ),
            build_atlas_entry(
                "TB-202",
                "Edit TextBox",
                "Edits short persisted metadata such as names and versions.",
                "Uses a bordered single-line input and validates through the owning dialog.",
                ("Edit Target dialog", "Session detail edit dialog", "Checkpoint Configuration dialog"),
            ),
            build_atlas_entry(
                "TB-203",
                "Note TextBox",
                "Captures a short free-form note or description for an existing object.",
                "Keeps user-entered text visible and saves only through an explicit action.",
                ("Checkpoint Configuration dialog", "Developer Tools dialog", "About dialog"),
            ),
        ],
        "selectors": [
            build_atlas_entry(
                "SL-301",
                "Model Selector",
                "Chooses the active model context for downstream dialogs.",
                "Opens a dedicated selection view instead of hiding model state inside a dropdown.",
                ("Load Model dialog", "Edit Target dialog", "Model Context dialog"),
            ),
            build_atlas_entry(
                "SL-302",
                "Session Selector",
                "Chooses the session branch used by view, edit, and run flows.",
                "Shows session lineage details before the selected session is applied.",
                ("Session Configuration dialog", "Model Context dialog", "Session Switch dialog"),
            ),
            build_atlas_entry(
                "SL-303",
                "Checkpoint Selector",
                "Chooses a checkpoint inside the selected model and session.",
                "Keeps checkpoint selection in its own dialog so edit flows receive a full target.",
                ("Checkpoint Configuration dialog", "Edit Target dialog", "Model Detail dialog"),
            ),
            build_atlas_entry(
                "SL-304",
                "Family Selector",
                "Switches between design template families in the atlas.",
                "Uses a compact dropdown because it changes only the preview category.",
                ("Developer Tools dialog", "Model creation dialog", "Session Switch dialog"),
            ),
        ],
        "trees": [
            build_atlas_entry(
                "TR-401",
                "Searchable Selection Tree",
                "Lists selectable records with keyboard navigation and a synchronized details panel.",
                "Use for model, session, checkpoint, TensorBoard, and clone source selection.",
                ("Load Model dialog", "Session Configuration dialog", "Checkpoint Configuration dialog"),
            ),
            build_atlas_entry(
                "TR-402",
                "Grouped Source Tree",
                "Groups available records before selection so cross-model sources stay readable.",
                "Use for source-heavy workflows where a flat dropdown hides too much context.",
                ("Comparison Pack Builder dialog", "Clone Selection dialog"),
            ),
            build_atlas_entry(
                "TR-403",
                "Invalid/Diagnostic Tree",
                "Shows invalid model folders or diagnostic structure checks without making them primary choices.",
                "Use for read-only diagnostics next to the valid selector.",
                ("Load Model dialog", "Run Configuration dialog"),
            ),
        ],
        "details": [
            build_atlas_entry(
                "DP-501",
                "Selection Details Panel",
                "Shows key/value metadata for the currently selected item.",
                "Use beside trees and selectors to make the selected target explicit before acting.",
                ("Load Model dialog", "TensorBoard Pack dialog", "Session Configuration dialog"),
            ),
            build_atlas_entry(
                "DP-502",
                "Metadata Details Panel",
                "Shows model/session/checkpoint metadata in a compact bordered block.",
                "Use for read-only context inside edit and view flows.",
                ("Model Context dialog", "Session Detail dialog", "Checkpoint Configuration dialog"),
            ),
        ],
        "lineage": [
            build_atlas_entry(
                "LG-601",
                "Lineage Preview",
                "Shows compact model or session ancestry next to the current edit target.",
                "Use inside dialogs when lineage matters but should not dominate the flow.",
                ("Edit Target dialog", "Model Detail dialog", "Model Context dialog"),
            ),
            build_atlas_entry(
                "LG-602",
                "Lineage Detail Viewer",
                "Shows a larger searchable lineage tree with selected-node metadata.",
                "Use when the user opens detailed model or session lineage.",
                ("Lineage Detail dialog",),
            ),
        ],
        "actions": [
            build_atlas_entry(
                "AR-701",
                "Primary Action Row",
                "Places forward actions and Cancel in a predictable full-width row.",
                "Use for confirm/save/open workflows at the bottom of dialogs.",
                ("Configuration dialog", "Load Model dialog", "Session Detail dialog"),
            ),
            build_atlas_entry(
                "AR-702",
                "Danger Action Row",
                "Combines destructive actions with save/cancel actions without changing their semantic colors.",
                "Use in edit/admin flows where delete is available but not the default action.",
                ("Model Detail dialog", "Session Configuration dialog", "Checkpoint Configuration dialog"),
            ),
        ],
        "footers": [
            build_atlas_entry(
                "FT-801",
                "Modal Footer",
                "Shows keyboard bindings below the modal content using the shared divider and spacing.",
                "Use `ModalFooter.from_binding_rows` and the shared footer CSS instead of local footer overrides.",
                ("All modal dialogs",),
            ),
            build_atlas_entry(
                "FT-802",
                "Grouped Keybinding Rows",
                "Splits bindings into readable rows when a dialog has several actions.",
                "Use when shortcuts would otherwise crowd a single footer line.",
                ("Configuration dialog", "Comparison Pack Builder dialog", "Clone Selection dialog"),
            ),
        ],
        "dialogs": [
            build_atlas_entry(
                "DG-901",
                "Standard Modal Container",
                "Uses the shared modal width, border title, scroll content, action row, and footer structure.",
                "Use for dialog-centered workflows so focus, scrolling, and keybinding layout stay consistent.",
                ("Configuration dialog", "Load Model dialog", "Developer Tools dialog"),
            ),
            build_atlas_entry(
                "DG-902",
                "Scrollable Dialog Content",
                "Keeps long form content scrollable while action rows and modal footer remain reachable.",
                "Use for edit dialogs and source-heavy selectors.",
                ("Model Detail dialog", "Compare Pack Builder dialog", "Developer Tools dialog"),
            ),
        ],
        "status": [
            build_atlas_entry(
                "ST-1001",
                "Configuration Status Tree",
                "Shows model readiness checks with pass/fail icons and concise detail text.",
                "Use when the user needs to understand why an action is available or blocked.",
                ("Run Configuration dialog",),
            ),
            build_atlas_entry(
                "ST-1002",
                "Info Readout",
                "Shows current model/runtime context as compact label/value rows.",
                "Use in persistent side panels and activity summaries.",
                ("Info Panel", "Control Panel"),
            ),
            build_atlas_entry(
                "ST-1003",
                "Control Panel",
                "Hosts the primary run control strip with Start, Resume, Pause, Stop, and inspection actions.",
                "Use for the shared execution control surface that drives training and inspection state.",
                ("Home screen", "Execution controls"),
            ),
        ],
        "transfer": [
            build_atlas_entry(
                "TF-1101",
                "Source/Selected Transfer",
                "Moves chosen items from available sources into a curated working set.",
                "Use where multi-source selection needs explicit review before commit.",
                ("Clone Selection dialog", "Comparison Pack Builder dialog"),
            ),
            build_atlas_entry(
                "TF-1102",
                "Curated Pack Editor",
                "Edits a curated pack description without reopening the creation workflow.",
                "Use for comparison pack notes and future pack administration.",
                ("Comparison Pack Builder dialog",),
            ),
        ],
    }

    _PREVIEW_WIDGET_IDS = {
        "buttons": "developer_tools_button_preview",
        "textboxes": "developer_tools_textbox_preview",
        "selectors": "developer_tools_selector_preview",
        "trees": "developer_tools_tree_preview",
        "details": "developer_tools_details_preview",
        "lineage": "developer_tools_lineage_preview",
        "actions": "developer_tools_action_preview",
        "footers": "developer_tools_footer_preview",
        "dialogs": "developer_tools_static_preview",
        "status": "developer_tools_status_preview",
        "transfer": "developer_tools_transfer_preview",
    }
    _BUTTON_VARIANTS = {
        "BT-101": "default",
        "BT-102": "primary",
        "BT-103": "success",
        "BT-104": "warning",
        "BT-105": "error",
    }

    def __init__(self) -> None:
        """Initialize the developer tools dialog and its owned state."""
        super().__init__()
        self._selected_demo = "buttons"
        self._selected_entry_code: Optional[str] = None

    def _current_entries(self) -> list[DesignAtlasEntry]:
        """Return the current entries value owned by this developer tools dialog."""
        return self._TEMPLATES[self._selected_demo]

    def _demo_label(self) -> str:
        """Return the demo label value owned by this developer tools dialog."""
        return dict(self._DEMOS).get(self._selected_demo, "Template Atlas")

    def _preview_widget_id(self) -> str:
        """Return the preview widget id value owned by this developer tools dialog."""
        return self._PREVIEW_WIDGET_IDS[self._selected_demo]

    def _preview_widget_ids(self) -> tuple[str, str, str]:
        """Return the preview widget ids value owned by this developer tools dialog."""
        return tuple(dict.fromkeys(self._PREVIEW_WIDGET_IDS.values()))

    def _set_preview_visibility(self) -> None:
        """Set the preview visibility owned by this developer tools dialog."""
        visible_id = self._preview_widget_id()
        for widget_id in self._preview_widget_ids():
            try:
                widget = self.query_one(f"#{widget_id}")
            except Exception:
                continue
            widget.display = widget_id == visible_id

    @staticmethod
    def _reset_tree(tree: Tree, title: str) -> None:
        """Reset the tree owned by this developer tools dialog."""
        tree.border_title = title
        tree.clear()
        tree.root.expand()
        tree.root.allow_expand = False

    def _update_tree_preview(self, entry: DesignAtlasEntry) -> None:
        """Update the tree preview owned by this developer tools dialog."""
        tree = self.query_one("#developer_tools_tree_preview", Tree)
        self._reset_tree(tree, f"{entry.name} Preview")
        if entry.code == "TR-402":
            model_node = tree.root.add("eve_project", expand=True, data=None)
            model_node.add_leaf("training | ray | 2026-04-22_00-11-56", data="source")
            other_node = tree.root.add("jasmine_project", expand=True, data=None)
            other_node.add_leaf("evaluation | sb3 | eval_2026-04-27", data="source")
        elif entry.code == "TR-403":
            tree.root.add_leaf("✓ Model detected                    eve_project [v1.1.10]")
            tree.root.add_leaf("✗ Selected checkpoint              fresh training")
            tree.root.add_leaf("✗ unreadable_manifest_project      not a manifest folder")
        else:
            tree.root.add_leaf("eve_project  v1.1.10  [CURRENT]", data="model")
            tree.root.add_leaf("jasmine_project  v1.2.3", data="model")

    def _update_details_preview(self, entry: DesignAtlasEntry) -> None:
        """Update the details preview owned by this developer tools dialog."""
        panel = self.query_one("#developer_tools_details_preview", SelectionDetailsPanel)
        panel.show_rows(
            [
                SelectionDetailsPanel.DetailRow("Name", "eve_project"),
                SelectionDetailsPanel.DetailRow("UUID", "8f4f0ae-..."),
                SelectionDetailsPanel.DetailRow("Version", "v1.1.10"),
                SelectionDetailsPanel.DetailRow("Context", entry.name),
            ],
            title=entry.name,
        )

    def _update_lineage_preview(self, entry: DesignAtlasEntry) -> None:
        """Update the lineage preview owned by this developer tools dialog."""
        tree = self.query_one("#developer_tools_lineage_preview", Tree)
        self._reset_tree(tree, f"{entry.name} Preview")
        root = tree.root.add("eve_project  v1.1.10  [CURRENT]", expand=True)
        root.add_leaf("2026-04-22_00-11-56  [6e4d603c]  ray")
        root.add_leaf("2026-04-22_06-57-35  [d1d4b54b]  sb3")

    def _update_action_preview(self, entry: DesignAtlasEntry) -> None:
        """Update the action preview owned by this developer tools dialog."""
        edit = self.query_one("#developer_tools_action_primary", Button)
        save = self.query_one("#developer_tools_action_success", Button)
        danger = self.query_one("#developer_tools_action_danger", Button)
        cancel = self.query_one("#developer_tools_action_cancel", Button)
        if entry.code == "AR-702":
            edit.display = False
            danger.display = True
            save.label = "Save Changes"
            cancel.label = "Cancel"
        else:
            edit.display = True
            danger.display = False
            edit.label = "Open"
            save.label = "Confirm"
            cancel.label = "Cancel"

    def _update_footer_preview(self, entry: DesignAtlasEntry) -> None:
        """Update the footer preview owned by this developer tools dialog."""
        footer = self.query_one("#developer_tools_footer_preview", ModalFooter)
        if entry.code == "FT-802":
            footer._bindings = [("f2 | a", "Add"), ("f3 | r", "Remove Last"), None, ("enter", "Create"), ("esc", "Cancel")]
            footer._max_per_row = 2
        else:
            footer._bindings = [("enter", "Confirm"), ("esc", "Cancel")]
            footer._max_per_row = 2
        footer.update(footer._build_text())

    def _update_status_preview(self, entry: DesignAtlasEntry) -> None:
        """Update the status preview owned by this developer tools dialog."""
        tree = self.query_one("#developer_tools_status_preview", Tree)
        self._reset_tree(tree, f"{entry.name} Preview")
        if entry.code == "ST-1002":
            tree.root.add_leaf("Model        eve_project [v1.1.10]")
            tree.root.add_leaf("Framework    ray")
            tree.root.add_leaf("Session      2026-04-22_00-11-56")
        else:
            tree.root.add_leaf("✓ Model detected                  eve_project [v1.1.10]")
            tree.root.add_leaf("✓ Sessions detected               3")
            tree.root.add_leaf("✗ Selected checkpoint             fresh training")

    def _update_transfer_preview(self, entry: DesignAtlasEntry) -> None:
        """Update the transfer preview owned by this developer tools dialog."""
        tree = self.query_one("#developer_tools_transfer_preview", Tree)
        self._reset_tree(tree, f"{entry.name} Preview")
        if entry.code == "TF-1102":
            tree.root.add_leaf("comparison_pack_2026-04-27")
            tree.root.add_leaf("Description  editable after pack selection")
        else:
            available = tree.root.add("Available Sources", expand=True)
            available.add_leaf("eve_project | training | ray")
            selected = tree.root.add("Selected", expand=True)
            selected.add_leaf("jasmine_project | evaluation | sb3")

    def _update_static_preview(self, entry: DesignAtlasEntry) -> None:
        """Update the static preview owned by this developer tools dialog."""
        preview = self.query_one("#developer_tools_static_preview", Static)
        preview.border_title = f"{entry.name} Preview"
        preview.update(f"{entry.code}  {entry.name}\n{entry.summary}")

    def _update_preview(self, entry: DesignAtlasEntry) -> None:
        """Update the preview owned by this developer tools dialog."""
        self._set_preview_visibility()
        if self._selected_demo == "buttons":
            button = self.query_one("#developer_tools_button_preview", Button)
            button.add_class("dialog-button-base")
            button.variant = self._BUTTON_VARIANTS.get(entry.code, "default")
            if entry.code == "BT-101":
                button.add_class("dialog-selector-button")
                button.label = selector_label(
                    "Selector",
                    f"{entry.code} • Model • eve_project [v1.1.10]",
                )
            else:
                button.remove_class("dialog-selector-button")
                button.label = entry.name
            return
        if self._selected_demo == "textboxes":
            textbox = self.query_one("#developer_tools_textbox_preview", Input)
            textbox.value = f"{entry.code} sample text"
            textbox.placeholder = entry.name
            return
        if self._selected_demo == "selectors":
            selector = self.query_one("#developer_tools_selector_preview", Select)
            selector.set_options(
                [
                    (f"{entry.code}  {entry.name}", entry.code),
                    ("Alternative sample", "__alternative__"),
                ]
            )
            selector.value = entry.code
            return
        if self._selected_demo == "trees":
            self._update_tree_preview(entry)
            return
        if self._selected_demo == "details":
            self._update_details_preview(entry)
            return
        if self._selected_demo == "lineage":
            self._update_lineage_preview(entry)
            return
        if self._selected_demo == "actions":
            self._update_action_preview(entry)
            return
        if self._selected_demo == "footers":
            self._update_footer_preview(entry)
            return
        if self._selected_demo == "status":
            self._update_status_preview(entry)
            return
        if self._selected_demo == "transfer":
            self._update_transfer_preview(entry)
            return
        self._update_static_preview(entry)

    def _build_tree(self) -> None:
        """Build the tree required by this developer tools dialog."""
        tree = self.query_one("#developer_tools_template_tree", Tree)
        tree.clear()
        tree.root.expand()
        tree.root.allow_expand = False
        selected_node = None
        for entry in self._current_entries():
            node = tree.root.add_leaf(f"{entry.code}  {entry.name}", data=entry.code)
            node.allow_expand = False
            if entry.code == self._selected_entry_code:
                selected_node = node
        if selected_node is None and tree.root.children:
            selected_node = tree.root.children[0]
        if selected_node is not None:
            tree.select_node(selected_node)

    def _show_entry(self, entry_code: str) -> None:
        """Show the entry owned by this developer tools dialog."""
        entry = next((item for item in self._current_entries() if item.code == entry_code), None)
        if entry is None:
            return
        self._selected_entry_code = entry.code
        self._set_preview_visibility()
        details = self.query_one("#developer_tools_details", SelectionDetailsPanel)
        details.show_rows(
            [
                SelectionDetailsPanel.DetailRow("Code", entry.code),
                SelectionDetailsPanel.DetailRow("Name", entry.name),
                SelectionDetailsPanel.DetailRow("Summary", entry.summary),
                SelectionDetailsPanel.DetailRow("Behavior", entry.behavior),
                SelectionDetailsPanel.DetailRow("Used in", ", ".join(entry.used_in)),
            ],
            title=f"{self._demo_label()} / {entry.code}",
        )
        self._update_preview(entry)

    def _show_family(self) -> None:
        """Show the family owned by this developer tools dialog."""
        family_details = self.query_one("#developer_tools_family_details", SelectionDetailsPanel)
        family_details.show_rows(
            [
                SelectionDetailsPanel.DetailRow("Family", self._demo_label()),
                SelectionDetailsPanel.DetailRow("Templates", str(len(self._current_entries()))),
            ],
            title="Component Family",
        )

    def _next_focus_id(self, current_id: str, direction: str) -> str | None:
        """Return the next focus id value owned by this developer tools dialog."""
        if current_id == "developer_tools_family_select":
            if direction == "down":
                return "developer_tools_template_tree"
            return None

        if current_id == "developer_tools_template_tree":
            if direction == "up":
                return "developer_tools_family_select"
            if direction == "down":
                return self._preview_widget_id()
            return None

        if current_id in self._preview_widget_ids():
            if direction == "up":
                return "developer_tools_template_tree"
            if direction == "down":
                return "developer_tools_cancel"
            return None

        if current_id == "developer_tools_cancel":
            if direction == "up":
                return self._preview_widget_id()
            return None

        return None

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this developer tools dialog."""
        with Vertical(id="developer_tools_dialog_container") as container:
            container.border_title = "Developer Tools"
            with ScrollableContainer(id="developer_tools_dialog_scroll"):
                with Vertical(id="developer_tools_dialog_content"):
                    family_select = Select(self._DEMOS, value=self._selected_demo, allow_blank=False, id="developer_tools_family_select")
                    family_select.border_title = "Component Family"
                    yield family_select
                    yield SelectionDetailsPanel(
                        title="Design Family",
                        empty_message="Choose a family.",
                        id="developer_tools_family_details",
                    )
                    tree = Tree("Design Atlas", id="developer_tools_template_tree")
                    tree.border_title = "Template Selector"
                    tree.show_root = False
                    tree.show_guides = True
                    yield tree
                    yield SelectionDetailsPanel(
                        title="Design Element",
                        empty_message="Choose a design element.",
                        id="developer_tools_details",
                    )
                    with Vertical(id="developer_tools_preview_row"):
                        yield selector_button("Selector Button", id="developer_tools_button_preview")
                        textbox_preview = Input(value="Sample text", id="developer_tools_textbox_preview")
                        textbox_preview.border_title = "TextBox Preview"
                        yield textbox_preview
                        selector_preview = Select(
                            [("Selector preview", "__preview__")],
                            value="__preview__",
                            allow_blank=False,
                            id="developer_tools_selector_preview",
                        )
                        selector_preview.border_title = "Selector Preview"
                        yield selector_preview
                        tree_preview = Tree("Tree Preview", id="developer_tools_tree_preview")
                        tree_preview.show_root = False
                        tree_preview.show_guides = True
                        yield tree_preview
                        yield SelectionDetailsPanel(
                            title="Details Preview",
                            empty_message="No preview selected.",
                            id="developer_tools_details_preview",
                        )
                        lineage_preview = Tree("Lineage Preview", id="developer_tools_lineage_preview")
                        lineage_preview.show_root = False
                        lineage_preview.show_guides = True
                        yield lineage_preview
                        with Horizontal(id="developer_tools_action_preview"):
                            yield Button("Open", id="developer_tools_action_primary", variant="primary")
                            yield Button("Confirm", id="developer_tools_action_success", variant="success")
                            yield Button("Delete", id="developer_tools_action_danger", variant="error")
                            yield Button("Cancel", id="developer_tools_action_cancel", variant="default")
                        yield ModalFooter(
                            [("enter", "Confirm"), ("esc", "Cancel")],
                            id="developer_tools_footer_preview",
                            max_per_row=2,
                        )
                        status_preview = Tree("Status Preview", id="developer_tools_status_preview")
                        status_preview.show_root = False
                        status_preview.show_guides = True
                        yield status_preview
                        transfer_preview = Tree("Transfer Preview", id="developer_tools_transfer_preview")
                        transfer_preview.show_root = False
                        transfer_preview.show_guides = True
                        yield transfer_preview
                        static_preview = Static("", id="developer_tools_static_preview")
                        static_preview.border_title = "Component Preview"
                        yield static_preview
            with Vertical(id="developer_tools_actions"):
                yield cancel_button(id="developer_tools_cancel")
            yield ModalFooter.from_binding_rows(
                self.BINDINGS,
                [["dismiss(None)"]],
                id="developer_tools_modal_footer",
                label_overrides={"dismiss(None)": "Cancel"},
            )

    def on_mount(self) -> None:
        """Handle the mount event for this developer tools dialog."""
        self._build_tree()
        self._show_family()
        current_entries = self._current_entries()
        if current_entries:
            self._show_entry(current_entries[0].code)
        self._focus_if_navigable("developer_tools_family_select")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle the select changed event for this developer tools dialog."""
        if event.select.id != "developer_tools_family_select":
            return
        self._selected_demo = str(event.value)
        self._selected_entry_code = None
        self._build_tree()
        self._show_family()
        self._show_entry(self._current_entries()[0].code)
        self._set_preview_visibility()

    def on_key(self, event: Key) -> None:
        """Handle the key event for this developer tools dialog."""
        direction = {
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
        }.get(event.key)
        if direction is None:
            return
        if self._move_focus_in_direction(direction):
            event.stop()
            event.prevent_default()

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle the tree node selected event for this developer tools dialog."""
        if isinstance(event.node.data, str):
            self._show_entry(event.node.data)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this developer tools dialog."""
        if event.button.id == "developer_tools_cancel":
            self.dismiss(None)
            return
        if event.button.id == "developer_tools_button_preview":
            event.stop()
