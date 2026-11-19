# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for model detail edit."""

from dataclasses import dataclass
from typing import Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, TextArea

from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.widgets.dialogs.button_templates import cancel_button, destructive_button, primary_button, success_button
from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.dialogs.lineage_detail_dialog import LineageDetailDialog
from lanista.gui.widgets.fragments.lineage_tree_viewer import LineageTreeViewer
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.locations import get_model_detail_edit_dialog_css_path
from lanista.reinforcengine.rl_model.model_creation_service import ModelCreationService


@dataclass
class ModelDetailEditDialogResult:
    """Report the result of model detail edit dialog."""

    action: str  # save | delete_model | cancel
    model_name: Optional[str]
    model_version: Optional[str]
    model_description: Optional[str]
    session_uuid: Optional[str] = None
    session_name: Optional[str] = None
    session_description: Optional[str] = None
    checkpoint_name: Optional[str] = None
    checkpoint_annotation: Optional[str] = None


class ModelDetailEditDialog(
    DirectionalFocusMixin,
    HelpEnabledModalMixin,
    ModalScreen[Optional[ModelDetailEditDialogResult]],
):
    """Present model detail edit in the Textual interface."""

    _HELP_TOPIC = "model_detail_edit"
    CSS_PATH = get_model_detail_edit_dialog_css_path()
    _DETAIL_LABEL_WIDTH = 12

    BINDINGS = [
        Binding("f3,e", "dismiss(None)", "Close edit details"),
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
        Binding("escape", "dismiss(None)", "Cancel", key_display="esc"),
    ]

    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "model_detail_name_input": {
            "down": "model_detail_version_input",
        },
        "model_detail_version_input": {
            "up": "model_detail_name_input",
            "down": "model_detail_description_input",
        },
        "model_detail_description_input": {
            "up": "model_detail_version_input",
            "down": "model_detail_session_name_input",
        },
        "model_detail_session_name_input": {
            "up": "model_detail_description_input",
            "down": "model_detail_session_description_input",
        },
        "model_detail_session_description_input": {
            "up": "model_detail_session_name_input",
            "down": "model_detail_checkpoint_annotation_input",
        },
        "model_detail_checkpoint_annotation_input": {
            "up": "model_detail_session_description_input",
            "down": "model_detail_model_lineage_preview_tree",
        },
        "model_detail_model_lineage_preview_tree": {
            "left": "model_detail_checkpoint_annotation_input",
            "right": "model_detail_session_lineage_preview_tree",
            "up": "model_detail_checkpoint_annotation_input",
            "down": "model_detail_model_lineage",
        },
        "model_detail_session_lineage_preview_tree": {
            "left": "model_detail_model_lineage_preview_tree",
            "up": "model_detail_checkpoint_annotation_input",
            "down": "model_detail_session_lineage",
        },
        "model_detail_model_lineage": {
            "up": "model_detail_model_lineage_preview_tree",
            "down": "model_detail_delete",
            "right": "model_detail_session_lineage",
        },
        "model_detail_session_lineage": {
            "up": "model_detail_session_lineage_preview_tree",
            "left": "model_detail_model_lineage",
            "right": "model_detail_delete",
            "down": "model_detail_save",
        },
        "model_detail_delete": {
            "up": "model_detail_session_lineage",
            "left": "model_detail_session_lineage",
            "right": "model_detail_save",
        },
        "model_detail_save": {
            "up": "model_detail_description_input",
            "left": "model_detail_delete",
            "right": "model_detail_cancel",
        },
        "model_detail_cancel": {
            "up": "model_detail_description_input",
            "left": "model_detail_save",
        },
    }

    def __init__(self,
                 model_name: str,
                 model_uuid: str,
                 model_version: str,
                 framework_uuid: str,
                 model_description: str,
                 session_uuid: Optional[str] = None,
                 session_name: str = "",
                 session_description: str = "",
                 checkpoint_name: Optional[str] = None,
                 checkpoint_annotation: str = ""):
        """Initialize the model detail edit dialog and its owned state."""
        super().__init__()
        self._model_name = model_name
        self._model_uuid = model_uuid
        self._model_version = model_version
        self._framework_uuid = framework_uuid
        self._model_description = model_description
        self._session_uuid = session_uuid
        self._session_name = session_name
        self._session_description = session_description
        self._checkpoint_name = checkpoint_name
        self._checkpoint_annotation = checkpoint_annotation

    @classmethod
    def _format_detail_row(cls, label: str, value: str) -> Text:
        """Format the detail row for presentation by this model detail edit dialog."""
        row = Text()
        row.append(f"{label:<{cls._DETAIL_LABEL_WIDTH}}", style="dim")
        row.append("  ")
        row.append(value, style="bold")
        return row

    @staticmethod
    def _resolved_model_version(value: str, existing_version: str) -> str:
        """Return the resolved model version value owned by this model detail edit dialog."""
        return ModelCreationService.resolve_model_version_text(
            value,
            existing_version=existing_version,
        )

    @classmethod
    def _normalized_model_name(cls, value: str, version: str, existing_version: str) -> str:
        """Return the normalized model name value owned by this model detail edit dialog."""
        return ModelCreationService.normalize_model_name_text(
            value,
            cls._resolved_model_version(version, existing_version),
        )

    def on_key(self, event: Key) -> None:
        """Handle the key event for this model detail edit dialog."""
        focused = self.focused
        focused_id = focused.id if focused is not None else None
        if focused_id in {"model_detail_model_lineage_preview_tree", "model_detail_session_lineage_preview_tree"} and event.key in {"left", "right", "up", "down"}:
            if self._move_focus_in_direction(event.key):
                event.stop()
                event.prevent_default()
            return

        if focused_id in {
            "model_detail_description_input",
            "model_detail_session_description_input",
            "model_detail_checkpoint_annotation_input",
        }:
            return

        if event.key in {"up", "down"} and self._move_focus_in_direction(event.key):
            event.stop()
            event.prevent_default()
            return

        button_focus_ids = {
            "model_detail_model_lineage",
            "model_detail_session_lineage",
            "model_detail_delete",
            "model_detail_save",
            "model_detail_cancel",
        }
        if focused_id in button_focus_ids and event.key in {"left", "right"}:
            if self._move_focus_in_direction(event.key):
                event.stop()
                event.prevent_default()

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this model detail edit dialog."""
        with Vertical(id="model_detail_dialog_container") as container:
            container.border_title = "Edit Model Details"
            with ScrollableContainer(id="model_detail_dialog_scroll"):
                with Vertical(id="model_detail_edit_dialog"):
                    model_name_input = Input(value=self._model_name, id="model_detail_name_input")
                    model_name_input.border_title = "Model Name"
                    yield model_name_input

                    model_version_input = Input(value=self._model_version, id="model_detail_version_input")
                    model_version_input.border_title = "Model Version"
                    yield model_version_input

                    model_description_input = TextArea(text=self._model_description or "", id="model_detail_description_input")
                    model_description_input.border_title = "Model Description"
                    yield model_description_input

                    with Vertical(id="model_detail_meta_panel") as meta_panel:
                        meta_panel.border_title = "Model Metadata"
                        yield Static(self._format_detail_row("UUID", self._model_uuid), id="model_detail_uuid")
                        yield Static(self._format_detail_row("Framework", self._framework_uuid or "-"), id="model_detail_framework")
                        yield Static(self._format_detail_row("Checkpoint", self._checkpoint_name or "fresh training"), id="model_detail_checkpoint")

                    session_name_input = Input(
                        value=self._session_name or "",
                        id="model_detail_session_name_input",
                        disabled=self._session_uuid is None,
                    )
                    session_name_input.border_title = "Session Name"
                    yield session_name_input

                    session_description_input = TextArea(
                        text=self._session_description or "",
                        id="model_detail_session_description_input",
                        disabled=self._session_uuid is None,
                    )
                    session_description_input.border_title = "Session Description"
                    yield session_description_input

                    checkpoint_annotation_input = TextArea(
                        text=self._checkpoint_annotation or "",
                        id="model_detail_checkpoint_annotation_input",
                        disabled=self._checkpoint_name is None,
                    )
                    checkpoint_annotation_input.border_title = "Checkpoint Note"
                    yield checkpoint_annotation_input

                    with Horizontal(id="model_detail_lineage_previews"):
                        yield LineageTreeViewer(
                            kind="model",
                            compact=True,
                            show_search=False,
                            show_details=False,
                            id="model_detail_model_lineage_preview",
                        )
                        yield LineageTreeViewer(
                            kind="session",
                            compact=True,
                            show_search=False,
                            show_details=False,
                            id="model_detail_session_lineage_preview",
                        )
                    with Horizontal(id="model_detail_lineage_actions"):
                        yield primary_button("Model Lineage", id="model_detail_model_lineage")
                        yield primary_button(
                            "Session Lineage",
                            id="model_detail_session_lineage",
                            disabled=self._session_uuid is None,
                        )
                    with Horizontal(id="model_detail_actions"):
                        yield destructive_button("Delete Model", id="model_detail_delete")
                        yield success_button("Save Changes", id="model_detail_save")
                        yield cancel_button(id="model_detail_cancel")
            yield ModalFooter.from_binding_rows(
                self.BINDINGS,
                [["open_help_dialog", "dismiss(None)"]],
                id="model_detail_modal_footer",
                label_overrides={"dismiss(None)": "Close"},
            )

    def on_mount(self) -> None:
        """Handle the mount event for this model detail edit dialog."""
        self.query_one("#model_detail_model_lineage_preview", LineageTreeViewer).set_model(self._model_uuid)
        self.query_one("#model_detail_session_lineage_preview", LineageTreeViewer).set_session(self._session_uuid)
        self.query_one("#model_detail_cancel", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this model detail edit dialog."""
        if event.button.id == "model_detail_model_lineage":
            self.app.push_screen(LineageDetailDialog(kind="model", model_uuid=self._model_uuid))
            return

        if event.button.id == "model_detail_session_lineage" and self._session_uuid is not None:
            self.app.push_screen(LineageDetailDialog(kind="session", session_uuid=self._session_uuid))
            return

        if event.button.id == "model_detail_save":
            normalized_model_version = self._resolved_model_version(
                self.query_one("#model_detail_version_input", Input).value,
                self._model_version,
            )
            normalized_model_name = self._normalized_model_name(
                self.query_one("#model_detail_name_input", Input).value,
                normalized_model_version,
                self._model_version,
            )
            self.query_one("#model_detail_name_input", Input).value = normalized_model_name
            self.query_one("#model_detail_version_input", Input).value = normalized_model_version
            self.dismiss(
                ModelDetailEditDialogResult(
                    action="save",
                    model_name=normalized_model_name,
                    model_version=normalized_model_version,
                    model_description=self.query_one("#model_detail_description_input", TextArea).text,
                    session_uuid=self._session_uuid,
                    session_name=self.query_one("#model_detail_session_name_input", Input).value if self._session_uuid else None,
                    session_description=self.query_one("#model_detail_session_description_input", TextArea).text if self._session_uuid else None,
                    checkpoint_name=self._checkpoint_name,
                    checkpoint_annotation=(
                        self.query_one("#model_detail_checkpoint_annotation_input", TextArea).text
                        if self._checkpoint_name
                        else None
                    ),
                )
            )
            return

        if event.button.id == "model_detail_delete":
            normalized_model_version = self._resolved_model_version(
                self.query_one("#model_detail_version_input", Input).value,
                self._model_version,
            )
            normalized_model_name = self._normalized_model_name(
                self.query_one("#model_detail_name_input", Input).value,
                normalized_model_version,
                self._model_version,
            )
            self.query_one("#model_detail_name_input", Input).value = normalized_model_name
            self.query_one("#model_detail_version_input", Input).value = normalized_model_version
            self.dismiss(
                ModelDetailEditDialogResult(
                    action="delete_model",
                    model_name=normalized_model_name,
                    model_version=normalized_model_version,
                    model_description=self.query_one("#model_detail_description_input", TextArea).text,
                    session_uuid=self._session_uuid,
                    session_name=self.query_one("#model_detail_session_name_input", Input).value if self._session_uuid else None,
                    session_description=self.query_one("#model_detail_session_description_input", TextArea).text if self._session_uuid else None,
                    checkpoint_name=self._checkpoint_name,
                    checkpoint_annotation=(
                        self.query_one("#model_detail_checkpoint_annotation_input", TextArea).text
                        if self._checkpoint_name
                        else None
                    ),
                )
            )
            return

        self.dismiss(None)
