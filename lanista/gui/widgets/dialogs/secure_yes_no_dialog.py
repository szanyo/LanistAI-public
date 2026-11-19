# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for secure yes no."""

import random
import textwrap
from typing import ClassVar, cast

from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.widgets.byte_editor_widget import ByteEditorWidget
from lanista.gui.widgets.dialogs.confirmation_dialog_content import ConfirmationDialogContent
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.locations import get_secure_yes_no_dialog_css_path
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Static
from textual.widgets._button import ButtonVariant


class SecureYesNoDialog(DirectionalFocusMixin, ModalScreen[bool]):

    """Present secure yes no in the Textual interface."""

    CSS_PATH = get_secure_yes_no_dialog_css_path()

    BINDINGS: ClassVar[
        list[Binding | tuple[str, str] | tuple[str, str, str]]
    ] = [
        Binding("f4,enter", "shortcut_confirm", "Confirm"),
        Binding("escape", "shortcut_cancel", "Cancel", key_display="esc"),
    ]

    _QUESTION_WRAP_WIDTH = 72

    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "pin_input_synd": {
            "down": "secure_confirm_synd",
            "right": "secure_cancel_synd",
        },
        "secure_confirm_synd": {
            "up": "pin_input_synd",
            "right": "secure_cancel_synd",
        },
        "secure_cancel_synd": {
            "up": "pin_input_synd",
            "left": "secure_confirm_synd",
        },
    }

    def __init__(
        self,
        question: str = "Are you sure?",
        left_button: str = "Yes",
        left_variant: ButtonVariant = "error",
        right_button: str = "No",
        right_variant: ButtonVariant = "primary",
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the secure yes no dialog and its owned state."""
        self._content = ConfirmationDialogContent(
            question=question,
            affirmative_label=left_button,
            affirmative_variant=left_variant,
            negative_label=right_button,
            negative_variant=right_variant,
        )
        super().__init__(name=name, id=id, classes=classes)

    @classmethod
    def _format_question(cls, question: str) -> str:
        """Format the question for presentation by this secure yes no dialog."""
        paragraphs: list[str] = []
        current_lines: list[str] = []

        for raw_line in str(question or "").strip().splitlines():
            line = raw_line.strip()
            if not line:
                if current_lines:
                    paragraphs.append(" ".join(current_lines))
                    current_lines = []
                continue
            current_lines.append(line)

        if current_lines:
            paragraphs.append(" ".join(current_lines))

        return "\n\n".join(
            textwrap.fill(paragraph, width=cls._QUESTION_WRAP_WIDTH)
            for paragraph in paragraphs
        )

    def _focused_navigation_id(self) -> str | None:
        """Return the focused navigation id value owned by this secure yes no dialog."""
        focused = self.focused
        focused_id = focused.id if focused is not None else None
        if focused_id:
            return focused_id

        try:
            pin_input = self.query_one("#pin_input_synd")
        except NoMatches:
            return None

        if focused is pin_input or (focused is not None and focused in list(pin_input.query("*"))):
            return "pin_input_synd"
        return None

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this secure yes no dialog."""
        with Vertical(id="dialog_synd_container") as container:
            container.border_title = "Secure Confirm"
            yield Static(
                self._format_question(self._content.question),
                id="question_synd")
            yield ByteEditorWidget(
                bits=[random.choice([False, True]) for _ in range(4)],
                unlocker=self.left_unlock_by_pin_input,
                id="pin_input_synd")
            with Horizontal(id="dialog_synd_actions"):
                yield Button(self._content.affirmative_label,
                       variant=self._content.affirmative_variant,
                       id="secure_confirm_synd")
                yield Button(self._content.negative_label,
                       variant=self._content.negative_variant,
                       id="secure_cancel_synd")
            yield ModalFooter.from_binding_rows(
                cast(list[Binding], self.BINDINGS),
                [["shortcut_confirm", "shortcut_cancel"]],
                id="secure_yes_no_modal_footer",
                label_overrides={
                    "shortcut_confirm": self._content.affirmative_label,
                    "shortcut_cancel": self._content.negative_label,
                },
            )

    def on_mount(self) -> None:
        """Focus the PIN input on mount."""
        self.query_one("#secure_confirm_synd").disabled = True
        self.query_one("#pin_input_synd").focus()

    def on_key(self, event: Key) -> None:
        """Handle the key event for this secure yes no dialog."""
        if event.key in {"up", "left", "down", "right"} and self._move_focus_in_direction(event.key):
            event.stop()
            event.prevent_default()

    def action_shortcut_confirm(self) -> None:
        """Handle the shortcut confirm user action for this secure yes no dialog."""
        button = self.query_one("#secure_confirm_synd", Button)
        if button.display and not button.disabled:
            button.press()

    def action_shortcut_cancel(self) -> None:
        """Handle the shortcut cancel user action for this secure yes no dialog."""
        self.dismiss(False)

    def left_unlock_by_pin_input(self) -> None:
        """Unlocks the left button by PIN input."""
        self.query_one("#secure_confirm_synd").focus()
        self.query_one("#secure_confirm_synd").disabled = False
        self.query_one("#pin_input_synd").disabled = True


    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this secure yes no dialog."""
        if event.button.id == "secure_confirm_synd":
            self.dismiss(True)
        else:
            self.dismiss(False)
