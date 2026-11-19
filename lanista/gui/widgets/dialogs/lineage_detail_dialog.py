# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for lineage detail."""

from dataclasses import dataclass
from typing import Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button

from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.fragments.lineage_tree_viewer import LineageTreeViewer
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.locations import get_configuration_dialog_css_path


@dataclass
class LineageDetailDialogResult:
    """Report the result of lineage detail dialog."""

    action: str  # close


class LineageDetailDialog(HelpEnabledModalMixin, ModalScreen[Optional[LineageDetailDialogResult]]):
    """Present lineage detail in the Textual interface."""

    _HELP_TOPIC = "lineage_detail"
    CSS_PATH = get_configuration_dialog_css_path()
    BINDINGS = [
        Binding("escape", "dismiss(None)", "Close", key_display="esc"),
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
    ]

    def __init__(self, *, kind: str, model_uuid: Optional[str] = None, session_uuid: Optional[str] = None):
        """Initialize the lineage detail dialog and its owned state."""
        super().__init__()
        self._kind = kind
        self._model_uuid = model_uuid
        self._session_uuid = session_uuid

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this lineage detail dialog."""
        title = "Model Lineage Details" if self._kind == "model" else "Session Lineage Details"
        with Vertical(id="lineage_detail_dialog_container") as container:
            container.border_title = title
            with ScrollableContainer(id="lineage_detail_dialog_scroll"):
                yield LineageTreeViewer(
                    kind=self._kind,
                    item_uuid=self._model_uuid if self._kind == "model" else self._session_uuid,
                    compact=False,
                    show_search=True,
                    show_details=True,
                    id="lineage_detail_viewer",
                )
            with Horizontal(id="lineage_detail_actions"):
                yield Button("Cancel", id="lineage_detail_cancel", variant="default")
            yield ModalFooter.from_binding_rows(
                self.BINDINGS,
                [["open_help_dialog", "dismiss(None)"]],
                id="lineage_detail_modal_footer",
                label_overrides={"dismiss(None)": "Close"},
            )

    def on_mount(self) -> None:
        """Handle the mount event for this lineage detail dialog."""
        viewer = self.query_one("#lineage_detail_viewer", LineageTreeViewer)
        if self._kind == "model":
            viewer.set_model(self._model_uuid)
        else:
            viewer.set_session(self._session_uuid)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this lineage detail dialog."""
        if event.button.id == "lineage_detail_cancel":
            self.dismiss(None)
