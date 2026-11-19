# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for edit target selection."""

from dataclasses import dataclass
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button

from lanista.gui.widgets.dialogs.button_templates import (
    cancel_button,
    primary_button,
    selector_button,
    selector_label,
    success_button,
)
from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.dialogs.lineage_detail_dialog import LineageDetailDialog
from lanista.gui.widgets.fragments.lineage_tree_viewer import LineageTreeViewer
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.locations import get_configuration_dialog_css_path


@dataclass
class EditTargetSelectionDialogResult:
    """Report the result of edit target selection dialog."""

    action: str  # edit | load_model | select_session | select_checkpoint | cancel
    session_name: Optional[str]
    checkpoint_name: Optional[str]


class EditTargetSelectionDialog(HelpEnabledModalMixin, ModalScreen[Optional[EditTargetSelectionDialogResult]]):
    """Present edit target selection in the Textual interface."""

    _HELP_TOPIC = "edit_target"
    CSS_PATH = get_configuration_dialog_css_path()

    BINDINGS = [
        Binding("m", "shortcut_model", "Model"),
        Binding("s", "shortcut_session", "Session"),
        Binding("k", "shortcut_checkpoint", "Checkpoint"),
        Binding("enter", "shortcut_edit", "Edit"),
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
        Binding("escape", "dismiss(None)", "Cancel", key_display="esc"),
    ]

    def __init__(
        self,
        *,
        loaded_model_name: Optional[str],
        model_uuid: Optional[str],
        session_options: list[tuple[str, str]],
        selected_session_name: Optional[str],
        checkpoint_options: list[tuple[str, str]],
        selected_checkpoint_name: Optional[str],
    ) -> None:
        """Initialize the edit target selection dialog and its owned state."""
        super().__init__()
        self._loaded_model_name = loaded_model_name
        self._model_uuid = model_uuid
        self._session_options = session_options
        self._selected_session_name = selected_session_name
        self._checkpoint_options = checkpoint_options
        self._selected_checkpoint_name = selected_checkpoint_name

    def _session_display_map(self) -> dict[str, str]:
        """Return the session display map value owned by this edit target selection dialog."""
        return {value: display for display, value in self._session_options}

    def _checkpoint_display_map(self) -> dict[str, str]:
        """Checkpoint the display map owned by this edit target selection dialog."""
        return {value: display for display, value in self._checkpoint_options}

    def _model_button_label(self) -> str:
        """Return the model button label value owned by this edit target selection dialog."""
        if self._loaded_model_name:
            return selector_label("Model", self._loaded_model_name)
        return selector_label("Model", "Choose model...")

    def _session_button_label(self) -> str:
        """Return the session button label value owned by this edit target selection dialog."""
        if self._selected_session_name:
            return selector_label("Session", self._session_display_map().get(self._selected_session_name, self._selected_session_name))
        if self._session_options:
            return selector_label("Session", "Choose session...")
        return selector_label("Session", "No session available yet.")

    def _checkpoint_button_label(self) -> str:
        """Checkpoint the button label owned by this edit target selection dialog."""
        if not self._selected_session_name:
            return selector_label("Checkpoint", "Select session first.")
        if self._selected_checkpoint_name:
            return selector_label("Checkpoint", self._checkpoint_display_map().get(self._selected_checkpoint_name, self._selected_checkpoint_name))
        if self._checkpoint_options:
            return selector_label("Checkpoint", "Choose checkpoint...")
        return selector_label("Checkpoint", "No checkpoint selected.")

    def _result(self, action: str) -> EditTargetSelectionDialogResult:
        """Return the result value owned by this edit target selection dialog."""
        return EditTargetSelectionDialogResult(
            action=action,
            session_name=self._selected_session_name,
            checkpoint_name=self._selected_checkpoint_name,
        )

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this edit target selection dialog."""
        with Vertical(id="edit_target_dialog_container") as container:
            container.border_title = "Edit Target"
            with ScrollableContainer(id="edit_target_dialog_scroll"):
                with Vertical(id="edit_target_dialog_content"):
                    yield selector_button(self._model_button_label(), id="edit_target_model_select")
                    yield selector_button(
                        self._session_button_label(),
                        id="edit_target_session_select",
                        disabled=not bool(self._session_options),
                    )
                    yield selector_button(
                        self._checkpoint_button_label(),
                        id="edit_target_checkpoint_select",
                        disabled=not bool(self._selected_session_name),
                    )
                    with Horizontal(id="edit_target_lineage_previews"):
                        yield LineageTreeViewer(
                            kind="model",
                            compact=True,
                            show_search=False,
                            show_details=False,
                            id="edit_target_model_lineage_preview",
                        )
                        yield LineageTreeViewer(
                            kind="session",
                            compact=True,
                            show_search=False,
                            show_details=False,
                            id="edit_target_session_lineage_preview",
                        )
                    with Horizontal(id="edit_target_lineage_actions"):
                        yield primary_button("Model Lineage", id="edit_target_model_lineage")
                        yield primary_button(
                            "Session Lineage",
                            id="edit_target_session_lineage",
                            disabled=not bool(self._selected_session_name),
                        )
                    with Horizontal(id="edit_target_actions"):
                        yield success_button(
                            "Edit",
                            id="edit_target_edit",
                            disabled=not bool(self._loaded_model_name),
                        )
                        yield cancel_button(id="edit_target_cancel")
            yield ModalFooter.from_binding_rows(
                self.BINDINGS,
                [["shortcut_model", "shortcut_session", "shortcut_checkpoint"], ["shortcut_edit", "dismiss(None)"], ["open_help_dialog"]],
                id="edit_target_modal_footer",
                label_overrides={"shortcut_edit": "Edit", "dismiss(None)": "Cancel"},
            )

    def on_mount(self) -> None:
        """Handle the mount event for this edit target selection dialog."""
        self.query_one("#edit_target_model_lineage_preview", LineageTreeViewer).set_model(self._model_uuid)
        self.query_one("#edit_target_session_lineage_preview", LineageTreeViewer).set_session(self._selected_session_name)
        self.query_one("#edit_target_edit", Button).focus()

    def action_shortcut_model(self) -> None:
        """Handle the shortcut model user action for this edit target selection dialog."""
        self.dismiss(self._result("load_model"))

    def action_shortcut_session(self) -> None:
        """Handle the shortcut session user action for this edit target selection dialog."""
        if self._session_options:
            self.dismiss(self._result("select_session"))

    def action_shortcut_checkpoint(self) -> None:
        """Handle the shortcut checkpoint user action for this edit target selection dialog."""
        if self._selected_session_name:
            self.dismiss(self._result("select_checkpoint"))

    def action_shortcut_edit(self) -> None:
        """Handle the shortcut edit user action for this edit target selection dialog."""
        button = self.query_one("#edit_target_edit", Button)
        if not button.disabled:
            button.press()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this edit target selection dialog."""
        if event.button.id == "edit_target_model_select":
            self.dismiss(self._result("load_model"))
            return
        if event.button.id == "edit_target_session_select":
            self.dismiss(self._result("select_session"))
            return
        if event.button.id == "edit_target_checkpoint_select":
            self.dismiss(self._result("select_checkpoint"))
            return
        if event.button.id == "edit_target_model_lineage":
            self.app.push_screen(LineageDetailDialog(kind="model", model_uuid=self._model_uuid))
            return
        if event.button.id == "edit_target_session_lineage":
            self.app.push_screen(LineageDetailDialog(kind="session", session_uuid=self._selected_session_name))
            return
        if event.button.id == "edit_target_edit":
            self.dismiss(self._result("edit"))
            return
        self.dismiss(None)
