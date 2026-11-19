# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for session detail edit."""

from dataclasses import dataclass
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, TextArea

from lanista.application import LanistAIApplication
from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.fragments.selection_details_panel import SelectionDetailsPanel
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.gui.utils.timestamps import format_local_timestamp
from lanista.locations import get_session_detail_edit_dialog_css_path


@dataclass
class SessionDetailEditDialogResult:
    """Report the result of session detail edit dialog."""

    action: str  # save | cancel
    session_uuid: Optional[str]
    session_name: Optional[str]
    session_description: Optional[str]


class SessionDetailEditDialog(
    DirectionalFocusMixin,
    HelpEnabledModalMixin,
    ModalScreen[Optional[SessionDetailEditDialogResult]],
):
    """Present session detail edit in the Textual interface."""

    _HELP_TOPIC = "session_detail_edit"
    CSS_PATH = get_session_detail_edit_dialog_css_path()
    _APPLICATION = LanistAIApplication()

    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "session_detail_name_input": {
            "down": "session_detail_description_input",
        },
        "session_detail_description_input": {
            "up": "session_detail_name_input",
            "down": "session_detail_save",
        },
        "session_detail_save": {
            "up": "session_detail_description_input",
            "right": "session_detail_cancel",
        },
        "session_detail_cancel": {
            "up": "session_detail_description_input",
            "left": "session_detail_save",
        },
    }

    BINDINGS = [
        Binding("f4,enter", "shortcut_confirm", "Save"),
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
        Binding("escape", "shortcut_cancel", "Cancel", key_display="esc"),
    ]

    def __init__(self, session_uuid: str):
        """Initialize the session detail edit dialog and its owned state."""
        super().__init__()
        self._session_uuid = str(session_uuid or "").strip() or None
        self._detail = None

    def _validation_error(self) -> Optional[str]:
        """Return the validation error value owned by this session detail edit dialog."""
        session_name = self.query_one("#session_detail_name_input", Input).value.strip()
        if not session_name:
            return "Session name is required."
        return None

    def _refresh_save_state(self) -> None:
        """Refresh the save state owned by this session detail edit dialog."""
        save_button = self.query_one("#session_detail_save", Button)
        error_label = self.query_one("#session_detail_error", Static)

        validation_error = self._validation_error()
        save_button.disabled = validation_error is not None
        error_label.update(validation_error or "")

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this session detail edit dialog."""
        with Vertical(id="session_detail_dialog_container") as container:
            container.border_title = "Edit Session Details"
            with ScrollableContainer(id="session_detail_dialog_scroll"):
                with Vertical(id="session_detail_edit_dialog"):
                    yield SelectionDetailsPanel(
                        title="Session",
                        empty_message="No session selected.",
                        id="session_detail_session_details",
                    )

                    session_name_input = Input(id="session_detail_name_input", placeholder="<session name>")
                    session_name_input.border_title = "Session Name"
                    yield session_name_input

                    session_description_input = TextArea(id="session_detail_description_input", text="")
                    session_description_input.border_title = "Session Description"
                    yield session_description_input

                    yield Static("", id="session_detail_error")
                    with Horizontal(id="session_detail_actions"):
                        yield Button("Save Changes", id="session_detail_save", variant="success")
                        yield Button("Cancel", id="session_detail_cancel", variant="default")

            yield ModalFooter.from_binding_rows(
                self.BINDINGS,
                [["shortcut_confirm", "shortcut_cancel"], ["open_help_dialog"]],
                id="session_detail_modal_footer",
                label_overrides={
                    "shortcut_confirm": "Save",
                    "shortcut_cancel": "Cancel",
                },
            )

    def on_mount(self) -> None:
        """Handle the mount event for this session detail edit dialog."""
        self._detail = (
            self._APPLICATION
            .get_application_services()
            .sessions.selection
            .build_session_selection_detail(self._session_uuid)
        )

        detail_panel = self.query_one("#session_detail_session_details", SelectionDetailsPanel)
        if self._detail is None:
            detail_panel.show_message("No session selected.")
            self.query_one("#session_detail_save", Button).disabled = True
            self.query_one("#session_detail_cancel", Button).focus()
            return

        rows = [
            SelectionDetailsPanel.DetailRow("Session", self._detail.display_name),
            SelectionDetailsPanel.DetailRow("UUID", self._detail.session_uuid),
            SelectionDetailsPanel.DetailRow("Framework", self._detail.framework_uuid),
            SelectionDetailsPanel.DetailRow("Parent", self._detail.parent_display_name),
            SelectionDetailsPanel.DetailRow("Forked From", self._detail.fork_source_session_display_name),
            SelectionDetailsPanel.DetailRow("Source Checkpoint", self._detail.fork_source_checkpoint_name or "-"),
            SelectionDetailsPanel.DetailRow("Step Offset", str(self._detail.base_step_offset)),
            SelectionDetailsPanel.DetailRow("Children", f"{self._detail.child_count} direct children"),
            SelectionDetailsPanel.DetailRow("Checkpoints", str(self._detail.checkpoint_count)),
            SelectionDetailsPanel.DetailRow("Latest", self._detail.latest_checkpoint_name or "-"),
            SelectionDetailsPanel.DetailRow("Created", format_local_timestamp(self._detail.created_at_raw)),
            SelectionDetailsPanel.DetailRow("Updated", format_local_timestamp(self._detail.modified_at_raw)),
        ]
        if self._detail.description:
            rows.append(SelectionDetailsPanel.DetailRow("Description", self._detail.description))
        detail_panel.show_rows(rows)

        self.query_one("#session_detail_name_input", Input).value = self._detail.display_name
        self.query_one("#session_detail_description_input", TextArea).text = self._detail.description or ""
        self._refresh_save_state()
        self.query_one("#session_detail_name_input", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle the input changed event for this session detail edit dialog."""
        if event.input.id == "session_detail_name_input":
            self._refresh_save_state()

    def on_key(self, event: Key) -> None:
        """Handle the key event for this session detail edit dialog."""
        focused = self.focused
        focused_id = focused.id if focused is not None else None

        if focused_id == "session_detail_description_input":
            return

        if event.key in {"up", "down", "left", "right"} and self._move_focus_in_direction(event.key):
            event.stop()
            event.prevent_default()

    def action_shortcut_confirm(self) -> None:
        """Handle the shortcut confirm user action for this session detail edit dialog."""
        button = self.query_one("#session_detail_save", Button)
        if button.display and not button.disabled:
            button.press()

    def action_shortcut_cancel(self) -> None:
        """Handle the shortcut cancel user action for this session detail edit dialog."""
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this session detail edit dialog."""
        if event.button.id == "session_detail_cancel":
            self.dismiss(None)
            return

        if event.button.id != "session_detail_save":
            return

        validation_error = self._validation_error()
        if validation_error:
            self.query_one("#session_detail_error", Static).update(validation_error)
            return

        self.dismiss(
            SessionDetailEditDialogResult(
                action="save",
                session_uuid=self._session_uuid,
                session_name=self.query_one("#session_detail_name_input", Input).value.strip(),
                session_description=self.query_one("#session_detail_description_input", TextArea).text.strip(),
            )
        )
