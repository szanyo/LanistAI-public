# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for load model."""

import os
from dataclasses import dataclass
from typing import Optional

from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.utils.text_cells import fit_text_cell
from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.fragments.lineage_tree_viewer import LineageTreeViewer
from lanista.gui.widgets.fragments.model_selector_panel import ModelSelectorPanel
from lanista.gui.widgets.fragments.selection_details_panel import SelectionDetailsPanel
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.locations import get_load_model_dialog_css_path
from lanista.reinforcengine.rl_model.empty_model_generator import EmptyModelGenerator
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Static, Tree


@dataclass
class LoadModelDialogResult:
    """Report the result of load model dialog."""

    action: str  # "load" | "edit"
    model_manifest: Optional[ModelManifest]


class LoadModelDialog(
    DirectionalFocusMixin,
    HelpEnabledModalMixin,
    ModalScreen[Optional[LoadModelDialogResult]],
):
    """Modal dialog to select and load a model."""

    _HELP_TOPIC = "load_model"
    CSS_PATH = get_load_model_dialog_css_path()

    BINDINGS = [
        Binding("f4,enter", "shortcut_confirm", "Confirm"),
        Binding("f3,e", "shortcut_edit", "Edit", key_display="f3"),
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
        Binding("escape", "shortcut_cancel", "Cancel", key_display="esc"),
    ]

    def __init__(self, current_model_uuid: Optional[str] = None):
        """Initialize the load model dialog and its owned state."""
        super().__init__()
        self._selected_manifest: Optional[ModelManifest] = None
        self._current_model_uuid = str(current_model_uuid or "").strip()
        self._invalid_name_width = 40
        self._invalid_reason_width = 32

    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "model_selector_search": {
            "down": "rl_model_tree",
            "right": "rl_model_tree",
        },
        "rl_model_tree": {
            "left": "model_selector_search",
            "right": "load_model_confirm",
        },
        "load_model_edit": {
            "up": "rl_model_tree",
            "right": "load_model_confirm",
        },
        "load_model_confirm": {
            "up": "rl_model_tree",
            "left": "load_model_edit",
            "right": "load_model_cancel",
        },
        "load_model_cancel": {
            "up": "rl_model_tree",
            "left": "load_model_confirm",
        },
    }

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this load model dialog."""
        with Vertical(id="load_model_dialog_container") as container:
            container.border_title = "Load Model"
            with VerticalScroll(id="load_model_dialog_scroll"):
                with Vertical(id="load_model_dialog_content"):
                    with Vertical(id="load_model_valid_section"):
                        yield ModelSelectorPanel(
                            id="dialog_model_selector_panel",
                            current_model_uuid=self._current_model_uuid,
                        )
                    yield Static("• current", id="load_model_legend")
                    with Horizontal(id="load_model_preview_region"):
                        yield SelectionDetailsPanel(
                            title="Selected Model",
                            empty_message="No model selected.",
                            id="load_model_selected_details",
                        )
                        yield LineageTreeViewer(
                            kind="model",
                            compact=True,
                            show_search=False,
                            show_details=False,
                            id="load_model_lineage_preview",
                        )
                    with Vertical(id="load_model_invalid_section"):
                        invalid_tree = Tree("Invalid Models", id="load_model_invalid_tree")
                        invalid_tree.show_root = False
                        invalid_tree.show_guides = False
                        invalid_tree.border_title = "Invalid Model Folders"
                        yield invalid_tree
                    with Vertical(id="load_model_actions"):
                        with Horizontal(id="load_model_actions_edit_row"):
                            yield Button("Edit", id="load_model_edit", variant="primary", disabled=True)
                        with Horizontal(id="load_model_actions_confirm_row"):
                            yield Button("Confirm", id="load_model_confirm", variant="success", disabled=True)
                            yield Button("Cancel", id="load_model_cancel", variant="default")
            yield ModalFooter.from_binding_rows(
                self.BINDINGS,
                [["shortcut_edit", "open_help_dialog"],
                 ["shortcut_confirm", "shortcut_cancel"]],
                id="load_model_modal_footer",
                label_overrides={
                    "shortcut_edit": "Edit",
                    "shortcut_confirm": "Confirm",
                    "shortcut_cancel": "Cancel",
                },
            )

    def action_shortcut_confirm(self) -> None:
        """Handle the shortcut confirm user action for this load model dialog."""
        if self._selected_manifest is not None:
            self.dismiss(LoadModelDialogResult(action="load", model_manifest=self._selected_manifest))

    def action_shortcut_edit(self) -> None:
        """Handle the shortcut edit user action for this load model dialog."""
        if self._selected_manifest is not None:
            self.dismiss(LoadModelDialogResult(action="edit", model_manifest=self._selected_manifest))

    def action_shortcut_cancel(self) -> None:
        """Handle the shortcut cancel user action for this load model dialog."""
        self.dismiss(None)

    def on_model_selector_panel_model_selected(self, event: ModelSelectorPanel.ModelSelected) -> None:
        """Handle the model selector panel model selected event for this load model dialog."""
        self._selected_manifest = event.model_manifest
        self.query_one("#load_model_edit", Button).disabled = event.model_manifest is None
        selected_uuid = ""
        if event.model_manifest is not None:
            selected_uuid = str(getattr(event.model_manifest.get_configuration(), "model_uuid", "") or "").strip()
            self._update_selected_preview(event.model_manifest)
        else:
            self.query_one("#load_model_selected_details", SelectionDetailsPanel).show_message("No model selected.")
            self.query_one("#load_model_lineage_preview", LineageTreeViewer).set_model(None)
        self.query_one("#load_model_confirm", Button).disabled = (
            event.model_manifest is None or (selected_uuid != "" and selected_uuid == self._current_model_uuid)
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this load model dialog."""
        if event.button.id == "load_model_cancel":
            self.dismiss(None)
            return
        if event.button.id == "load_model_edit" and self._selected_manifest is not None:
            self.dismiss(LoadModelDialogResult(action="edit", model_manifest=self._selected_manifest))
            return
        if event.button.id == "load_model_confirm" and self._selected_manifest is not None:
            self.dismiss(LoadModelDialogResult(action="load", model_manifest=self._selected_manifest))

    def on_key(self, event: Key) -> None:
        """Handle the key event for this load model dialog."""
        if event.key not in {"up", "left", "down", "right"}:
            return

        focused_id = self.focused.id if self.focused is not None else None
        if focused_id == "model_selector_search" and event.key != "down":
            return
        if focused_id == "rl_model_tree" and event.key in {"up", "down"}:
            return

        if self._move_focus_in_direction(event.key):
            event.stop()
            event.prevent_default()

    def on_mount(self) -> None:
        """Handle the mount event for this load model dialog."""
        self.call_after_refresh(self._refresh_invalid_models_block)
        for candidate in (
            "#model_selector_search",
            "#rl_model_tree",
            "#load_model_cancel",
        ):
            try:
                widget = self.query_one(candidate)
                if widget.display and not getattr(widget, "disabled", False) and widget.can_focus:
                    widget.focus()
                    break
            except Exception:
                continue

    def _refresh_invalid_models_block(self) -> None:
        """Refresh the invalid models block owned by this load model dialog."""
        invalid_section = self.query_one("#load_model_invalid_section", Vertical)
        invalid_tree = self.query_one("#load_model_invalid_tree", Tree)
        selector_panel = self.query_one("#dialog_model_selector_panel", ModelSelectorPanel)
        invalid_entries = selector_panel.get_invalid_entries()
        invalid_tree.clear()
        invalid_tree.root.expand()
        invalid_tree.root.allow_expand = False
        if not invalid_entries:
            invalid_section.display = False
            invalid_tree.display = False
            return

        invalid_section.display = True
        invalid_tree.display = True
        for entry in invalid_entries:
            label = Text()
            label.append(fit_text_cell(str(entry.folder_name or ""), self._invalid_name_width), style="bold red")
            label.append("  ")
            label.append(fit_text_cell(str(entry.reason or ""), self._invalid_reason_width), style="dim")
            node = invalid_tree.root.add_leaf(label, data=None)
            node.allow_expand = False

    @staticmethod
    def _model_session_count(model_folder_path: str) -> int:
        """Return the model session count value owned by this load model dialog."""
        sessions_root = os.path.join(str(model_folder_path or ""), "training_sessions")
        if not os.path.isdir(sessions_root):
            return 0
        return sum(
            1
            for name in os.listdir(sessions_root)
            if os.path.isdir(os.path.join(sessions_root, name))
        )

    def _update_selected_preview(self, manifest: ModelManifest) -> None:
        """Update the selected preview owned by this load model dialog."""
        cfg = manifest.get_configuration()
        model_uuid = str(getattr(cfg, "model_uuid", "") or "").strip()
        model_folder_path = str(getattr(cfg, "model_folder_path", "") or "").strip()
        parent_uuid = str(getattr(cfg, "parent_model_uuid", "") or "").strip()
        framework_uuid = (
            str(getattr(cfg, "active_runtime_framework_uuid", "") or "").strip()
            or str(getattr(cfg, "preferred_framework_uuid", "") or "").strip()
            or "-"
        )
        rows = [
            SelectionDetailsPanel.DetailRow("Name", str(getattr(cfg, "model_name", "") or "-")),
            SelectionDetailsPanel.DetailRow("UUID", EmptyModelGenerator.shorten_uuid(model_uuid) or "-"),
            SelectionDetailsPanel.DetailRow("Version", str(getattr(cfg, "model_version", "") or "-")),
            SelectionDetailsPanel.DetailRow("Framework", framework_uuid),
            SelectionDetailsPanel.DetailRow("Parent", EmptyModelGenerator.shorten_uuid(parent_uuid) or "-"),
            SelectionDetailsPanel.DetailRow("Sessions", str(self._model_session_count(model_folder_path))),
        ]
        description = str(getattr(cfg, "description", "") or "").strip()
        if description:
            rows.append(SelectionDetailsPanel.DetailRow("Description", description))
        self.query_one("#load_model_selected_details", SelectionDetailsPanel).show_rows(rows)
        self.query_one("#load_model_lineage_preview", LineageTreeViewer).set_model(model_uuid)
