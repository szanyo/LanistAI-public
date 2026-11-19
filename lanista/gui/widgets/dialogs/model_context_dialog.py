# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for model context."""

from dataclasses import dataclass

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from lanista.application import LanistAIApplication
from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.utils.timestamps import format_local_timestamp
from lanista.gui.widgets.dialogs.button_templates import (
    cancel_button,
    dialog_action_row,
    primary_button,
    success_button,
)
from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.dialogs.lineage_detail_dialog import LineageDetailDialog
from lanista.gui.widgets.fragments.lineage_tree_viewer import LineageTreeViewer
from lanista.gui.widgets.fragments.selection_details_panel import SelectionDetailsPanel
from lanista.gui.widgets.modal_footer import ModalFooter


@dataclass
class ModelContextDialogResult:
    """Report the result of model context dialog."""

    action: str  # edit | sessions | checkpoints | cancel


class ModelContextDialog(
    DirectionalFocusMixin,
    HelpEnabledModalMixin,
    ModalScreen[ModelContextDialogResult | None],
):
    """Present model context in the Textual interface."""

    _HELP_TOPIC = "model_context"
    _APPLICATION = LanistAIApplication()

    BINDINGS = [
        Binding("f2,v", "dismiss(None)", "Close context", key_display="f2"),
        Binding("e,f3", "edit_context", "Edit", key_display="f3"),
        Binding("s", "open_session_selector", "Sessions"),
        Binding("k", "open_checkpoint_selector", "Checkpoints"),
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
        Binding("escape", "dismiss(None)", "Close", key_display="esc"),
    ]

    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "model_context_model_lineage_preview_tree": {
            "right": "model_context_session_lineage_preview_tree",
            "down": "model_context_model_lineage",
        },
        "model_context_session_lineage_preview_tree": {
            "left": "model_context_model_lineage_preview_tree",
            "down": "model_context_session_lineage",
        },
        "model_context_model_lineage": {
            "up": "model_context_model_lineage_preview_tree",
            "right": "model_context_session_lineage",
            "down": "model_context_sessions",
        },
        "model_context_session_lineage": {
            "up": "model_context_session_lineage_preview_tree",
            "left": "model_context_model_lineage",
            "down": "model_context_checkpoints",
        },
        "model_context_sessions": {
            "up": "model_context_model_lineage",
            "right": "model_context_checkpoints",
            "down": "model_context_edit",
        },
        "model_context_checkpoints": {
            "up": "model_context_session_lineage",
            "left": "model_context_sessions",
            "down": "model_context_close",
        },
        "model_context_edit": {
            "up": "model_context_sessions",
            "right": "model_context_close",
        },
        "model_context_close": {
            "up": "model_context_checkpoints",
            "left": "model_context_edit",
        },
    }

    def __init__(self, *, session_uuid: str | None):
        """Initialize the model context dialog and its owned state."""
        super().__init__()
        self._session_uuid = session_uuid
        self._payload = None

    def _format_detail_rows(self) -> tuple[list[SelectionDetailsPanel.DetailRow], list[SelectionDetailsPanel.DetailRow]]:
        """Format the detail rows for presentation by this model context dialog."""
        payload = self._payload
        if payload is None:
            return [], []

        model_rows = [
            SelectionDetailsPanel.DetailRow("Name", payload.model_name),
            SelectionDetailsPanel.DetailRow("UUID", payload.model_uuid),
            SelectionDetailsPanel.DetailRow("Version", payload.model_version),
            SelectionDetailsPanel.DetailRow("Framework", payload.framework_uuid or "-"),
            SelectionDetailsPanel.DetailRow("Parent", payload.parent_model_display_name),
            SelectionDetailsPanel.DetailRow("Children", f"{len(payload.child_model_uuids)} direct children"),
            SelectionDetailsPanel.DetailRow("Created", format_local_timestamp(payload.model_created_at_raw)),
            SelectionDetailsPanel.DetailRow("Updated", format_local_timestamp(payload.model_modified_at_raw)),
        ]
        if payload.model_description:
            model_rows.append(SelectionDetailsPanel.DetailRow("Description", payload.model_description))

        session_rows = [
            SelectionDetailsPanel.DetailRow("Session", payload.current_session_display_name or "-"),
            SelectionDetailsPanel.DetailRow("UUID", payload.current_session_identity_uuid or "-"),
            SelectionDetailsPanel.DetailRow("Name", payload.current_session_name or "-"),
            SelectionDetailsPanel.DetailRow("Parent", payload.current_session_parent_display_name),
            SelectionDetailsPanel.DetailRow("Children", f"{payload.current_session_child_count} direct children"),
        ]
        if payload.current_session_description:
            session_rows.append(SelectionDetailsPanel.DetailRow("Description", payload.current_session_description))

        return model_rows, session_rows

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this model context dialog."""
        with Vertical(id="model_context_dialog_container") as container:
            container.border_title = "Model Context"
            with (
                ScrollableContainer(id="model_context_dialog_scroll"),
                Vertical(id="model_context_dialog_content"),
            ):
                    yield Static(
                        "Inspect the selected model and session context. Use the lineage "
                        "previews for orientation and the actions below for administration.",
                        id="model_context_hint",
                    )
                    with Horizontal(id="model_context_details_region"):
                        yield SelectionDetailsPanel(
                            title="Model",
                            empty_message="No model selected.",
                            id="model_context_model_details",
                        )
                        yield SelectionDetailsPanel(
                            title="Session",
                            empty_message="No session selected.",
                            id="model_context_session_details",
                        )
                    with Horizontal(id="model_context_lineage_previews"):
                        yield LineageTreeViewer(
                            kind="model",
                            compact=True,
                            show_search=False,
                            show_details=False,
                            id="model_context_model_lineage_preview",
                        )
                        yield LineageTreeViewer(
                            kind="session",
                            compact=True,
                            show_search=False,
                            show_details=False,
                            id="model_context_session_lineage_preview",
                        )
                    yield dialog_action_row(
                        primary_button(
                            "Model Lineage",
                            id="model_context_model_lineage",
                        ),
                        primary_button(
                            "Session Lineage",
                            id="model_context_session_lineage",
                        ),
                        id="model_context_actions_primary",
                    )
                    yield dialog_action_row(
                        primary_button(
                            "Select Session",
                            id="model_context_sessions",
                        ),
                        primary_button(
                            "Select Checkpoint",
                            id="model_context_checkpoints",
                        ),
                        id="model_context_actions_secondary",
                    )
                    yield dialog_action_row(
                        success_button("Edit Details", id="model_context_edit"),
                        cancel_button("Close", id="model_context_close"),
                        id="model_context_actions_bottom",
                    )
            yield ModalFooter.from_binding_rows(
                self.BINDINGS,
                [["edit_context", "open_session_selector", "open_checkpoint_selector"],
                 ["open_help_dialog", "dismiss(None)"]],
                id="model_context_modal_footer",
                label_overrides={"dismiss(None)": "Close"},
            )

    def on_mount(self) -> None:
        """Handle the mount event for this model context dialog."""
        self._payload = (
            self._APPLICATION
            .get_application_services()
            .inspection.presentation
            .build_model_context_payload(self._session_uuid)
        )
        model_rows, session_rows = self._format_detail_rows()
        model_panel = self.query_one("#model_context_model_details", SelectionDetailsPanel)
        session_panel = self.query_one("#model_context_session_details", SelectionDetailsPanel)
        if self._payload is None:
            model_panel.show_message("No model selected.")
            session_panel.show_message("No session selected.")
            self.query_one("#model_context_model_lineage", Button).disabled = True
            self.query_one("#model_context_session_lineage", Button).disabled = True
            self.query_one("#model_context_sessions", Button).disabled = True
            self.query_one("#model_context_checkpoints", Button).disabled = True
            self._focus_if_navigable("model_context_close")
            return

        model_panel.show_rows(model_rows, title="Model")
        session_panel.show_rows(session_rows, title="Session")
        self.query_one("#model_context_model_lineage_preview", LineageTreeViewer).set_model(self._payload.model_uuid)
        self.query_one("#model_context_session_lineage_preview", LineageTreeViewer).set_session(
            self._payload.current_session_uuid,
        )
        self.query_one("#model_context_session_lineage", Button).disabled = not bool(self._payload.current_session_uuid)
        self.query_one("#model_context_checkpoints", Button).disabled = not bool(self._payload.current_session_uuid)
        self._focus_if_navigable("model_context_edit")

    def on_key(self, event: Key) -> None:
        """Apply the shared directional-focus behavior to the context view."""

        focused_id = self.focused.id if self.focused is not None else None
        if focused_id in {
            "model_context_model_lineage_preview_tree",
            "model_context_session_lineage_preview_tree",
        } and event.key in {"up", "left", "down", "right"}:
            if self._move_focus_in_direction(event.key):
                event.stop()
                event.prevent_default()
            return
        if event.key in {"up", "left", "down", "right"} and self._move_focus_in_direction(
            event.key
        ):
            event.stop()
            event.prevent_default()

    def action_edit_context(self) -> None:
        """Handle the edit context user action for this model context dialog."""
        self.dismiss(ModelContextDialogResult(action="edit"))

    def action_open_session_selector(self) -> None:
        """Handle the open session selector user action for this model context dialog."""
        self.dismiss(ModelContextDialogResult(action="sessions"))

    def action_open_checkpoint_selector(self) -> None:
        """Handle the open checkpoint selector user action for this model context dialog."""
        self.dismiss(ModelContextDialogResult(action="checkpoints"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this model context dialog."""
        if event.button.id == "model_context_edit":
            self.dismiss(ModelContextDialogResult(action="edit"))
            return
        if event.button.id == "model_context_sessions":
            self.dismiss(ModelContextDialogResult(action="sessions"))
            return
        if event.button.id == "model_context_checkpoints":
            self.dismiss(ModelContextDialogResult(action="checkpoints"))
            return
        if event.button.id == "model_context_model_lineage" and self._payload is not None:
            self.app.push_screen(LineageDetailDialog(kind="model", model_uuid=self._payload.model_uuid))
            return
        if event.button.id == "model_context_session_lineage" and self._payload is not None:
            self.app.push_screen(LineageDetailDialog(kind="session", session_uuid=self._payload.current_session_uuid))
            return
        self.dismiss(None)
