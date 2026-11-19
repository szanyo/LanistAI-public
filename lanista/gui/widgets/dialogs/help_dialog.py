# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present central or dialog-specific Help content."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Markdown

from lanista.gui.widgets.dialogs.help_content import build_help_document
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.locations import get_help_dialog_css_path


class HelpDialog(ModalScreen[None]):
    """Present central or contextual Help in the Textual interface."""

    CSS_PATH = get_help_dialog_css_path()
    BINDINGS = [
        Binding("escape,q,ctrl+c", "dismiss(None)", "Close help", key_display="esc"),
        Binding("f9,h,?", "dismiss(None)", "Close help", show=False, key_display="f9"),
    ]

    _FALLBACK_MARKDOWN = "## Help\n\nHelp content could not be loaded."

    def __init__(self, topic_id: str | None = None) -> None:
        """Initialize Help with an optional dialog-specific topic."""

        super().__init__()
        self._topic_id = topic_id

    @property
    def topic_id(self) -> str | None:
        """Return the contextual topic selected for this Help view."""

        return self._topic_id

    @classmethod
    def _load_markdown(cls, topic_id: str | None = None) -> str:
        """Load central or contextual Markdown owned by this Help dialog."""

        return build_help_document(topic_id).markdown

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this help dialog."""
        document = build_help_document(self._topic_id)
        with Vertical(id="help_dialog_container") as vertical:
            vertical.border_title = document.title
            with VerticalScroll(id="help_dialog_scroll"):
                yield Markdown(document.markdown, id="help-markdown")
            with Horizontal(id="help_dialog_actions"):
                yield Button("Close", id="help_dialog_cancel", variant="default")
            yield ModalFooter.from_binding_rows(
                self.BINDINGS,
                [
                    [("pageup", "Scroll Up"), ("pagedown", "Scroll Down")],
                    [("home | end", "Jump"), "dismiss(None)"],
                ],
                id="help_modal_footer",
                label_overrides={"dismiss(None)": "Close Help"},
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this help dialog."""
        if event.button.id == "help_dialog_cancel":
            self.dismiss(None)
