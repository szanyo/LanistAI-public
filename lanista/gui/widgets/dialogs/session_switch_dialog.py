# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for session switch."""

from dataclasses import dataclass
from time import monotonic
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Select, Static

from lanista.application import LanistAIApplication
from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.locations import get_session_switch_dialog_css_path


@dataclass
class SessionSwitchDialogResult:
    """Report the result of session switch dialog."""

    action: str  # "load" | "delete" | "cancel"
    session_uuid: Optional[str]


class SessionSwitchDialog(
    DirectionalFocusMixin,
    HelpEnabledModalMixin,
    ModalScreen[Optional[SessionSwitchDialogResult]],
):
    """Session selector modal for switching active training session."""

    _HELP_TOPIC = "session_switch"
    CSS_PATH = get_session_switch_dialog_css_path()

    _APPLICATION = LanistAIApplication()
    _SHORTCUT_GUARD_SECONDS = 0.2

    BINDINGS = [
        Binding("f2,v", "dismiss(None)", "Close switch session"),
        Binding("l,enter", "shortcut_load_selected", "Load"),
        Binding("d", "shortcut_delete_selected", "Delete"),
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
        Binding("escape", "dismiss(None)", "Cancel", key_display="esc"),
    ]

    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "session_switch_select": {
            "down": "session_switch_delete",
            "right": "session_switch_load",
        },
        "session_switch_delete": {
            "up": "session_switch_select",
            "right": "session_switch_load",
        },
        "session_switch_load": {
            "up": "session_switch_select",
            "left": "session_switch_delete",
            "right": "session_switch_cancel",
        },
        "session_switch_cancel": {
            "up": "session_switch_select",
            "left": "session_switch_load",
        },
    }

    def __init__(self, current_session_uuid: Optional[str] = None):
        """Initialize the session switch dialog and its owned state."""
        super().__init__()
        self._current_session_uuid = current_session_uuid
        self._selected_session_uuid: Optional[str] = None
        self._mounted_at = 0.0

    def on_key(self, event: Key) -> None:
        """Handle the key event for this session switch dialog."""
        if event.key in {"up", "left", "down", "right"} and self._move_focus_in_direction(event.key):
            event.stop()
            event.prevent_default()

    def _press_button_if_enabled(self, button_id: str) -> None:
        """Return whether press button if enabled applies to this session switch dialog."""
        if self._mounted_at <= 0 or (monotonic() - self._mounted_at) < self._SHORTCUT_GUARD_SECONDS:
            return
        button = self.query_one(f"#{button_id}", Button)
        if button.display and not button.disabled:
            button.press()

    def action_shortcut_load_selected(self) -> None:
        """Handle the shortcut load selected user action for this session switch dialog."""
        self._press_button_if_enabled("session_switch_load")

    def action_shortcut_delete_selected(self) -> None:
        """Handle the shortcut delete selected user action for this session switch dialog."""
        self._press_button_if_enabled("session_switch_delete")

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this session switch dialog."""
        services = self._APPLICATION.get_application_services()
        sessions = services.catalog.session_checkpoints.list_sessions_for_current_model()
        options = [
            (services.sessions.metadata.get_session_reference_display(session_uuid), session_uuid)
            for session_uuid in sessions
        ]

        selected = self._current_session_uuid if self._current_session_uuid in {value for _, value in options} else None
        if selected is None and options:
            selected = options[0][1]

        self._selected_session_uuid = selected

        with Vertical(id="session_switch_dialog_container") as container:
            container.border_title = "Switch Training Session"
            yield Static("Select session (newest first):", id="session_switch_title")

            if options:
                yield Select(
                    options,
                    value=self._selected_session_uuid,
                    allow_blank=False,
                    id="session_switch_select",
                )
            else:
                yield Select(
                    [("No session available.", "__no_session__")],
                    value="__no_session__",
                    allow_blank=False,
                    disabled=True,
                    id="session_switch_select",
                )

            with Horizontal(id="session_switch_actions"):
                yield Button("Delete Session", id="session_switch_delete", variant="error", disabled=self._selected_session_uuid is None)
                yield Button("Load Selected", id="session_switch_load", variant="success", disabled=self._selected_session_uuid is None)
                yield Button("Cancel", id="session_switch_cancel", variant="default")
            yield ModalFooter.from_binding_rows(
                self.BINDINGS,
                [
                    ["shortcut_load_selected", "shortcut_delete_selected"],
                    ["open_help_dialog", "dismiss(None)"],
                ],
                id="session_switch_modal_footer",
                label_overrides={"dismiss(None)": "Close"},
            )

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle the select changed event for this session switch dialog."""
        if event.select.id != "session_switch_select":
            return

        value = str(event.value) if event.value else None
        if value == "__no_session__":
            value = None
        self._selected_session_uuid = value

        self.query_one("#session_switch_delete", Button).disabled = self._selected_session_uuid is None
        self.query_one("#session_switch_load", Button).disabled = self._selected_session_uuid is None

    def on_mount(self) -> None:
        """Handle the mount event for this session switch dialog."""
        self._mounted_at = monotonic()
        self._focus_if_navigable("session_switch_cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this session switch dialog."""
        button_id = event.button.id
        if button_id == "session_switch_load":
            self.dismiss(SessionSwitchDialogResult("load", self._selected_session_uuid))
            return
        if button_id == "session_switch_delete":
            self.dismiss(SessionSwitchDialogResult("delete", self._selected_session_uuid))
            return
        self.dismiss(None)
