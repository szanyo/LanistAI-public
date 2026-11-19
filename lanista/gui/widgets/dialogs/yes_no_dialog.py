# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for yes no."""

from typing import ClassVar, cast

from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.widgets.dialogs.confirmation_dialog_content import ConfirmationDialogContent
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.locations import get_yes_no_dialog_css_path
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label
from textual.widgets._button import ButtonVariant


class YesNoDialog(DirectionalFocusMixin, ModalScreen[bool]):
    """Present yes no in the Textual interface."""

    CSS_PATH = get_yes_no_dialog_css_path()

    BINDINGS: ClassVar[
        list[Binding | tuple[str, str] | tuple[str, str, str]]
    ] = [
        Binding("left,y", "left", "Yes"),
        Binding("right,n", "right", "No"),
        Binding("escape", "dismiss", "Dismiss"),
    ]

    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "left_ynd": {"right": "right_ynd"},
        "right_ynd": {"left": "left_ynd"},
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
        """Initialize the yes no dialog and its owned state."""
        self._content = ConfirmationDialogContent(
            question=question,
            affirmative_label=left_button,
            affirmative_variant=left_variant,
            negative_label=right_button,
            negative_variant=right_variant,
        )
        super().__init__(name=name, id=id, classes=classes)

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this yes no dialog."""
        with Vertical(id="dialog_ynd_container") as container:
            container.border_title = "Confirm"
            yield Label(
                self._content.question,
                id="question_ynd")
            with Horizontal(id="dialog_ynd_actions"):
                yield Button(self._content.affirmative_label,
                       variant=self._content.affirmative_variant,
                       id="left_ynd")
                yield Button(self._content.negative_label,
                       variant=self._content.negative_variant,
                       id="right_ynd")
            yield ModalFooter.from_binding_rows(
                cast(list[Binding], self.BINDINGS),
                [["left", "right"], ["dismiss"]],
                id="yes_no_modal_footer",
                label_overrides={
                    "left": self._content.affirmative_label,
                    "right": self._content.negative_label,
                    "dismiss": "Close",
                },
            )

    def on_mount(self) -> None:
        """Handle the mount event for this yes no dialog."""
        self._focus_if_navigable("right_ynd")

    def action_left(self) -> None:
        """Move toward the affirmative action."""

        if not self._move_focus_in_direction("left"):
            self._focus_if_navigable("left_ynd")

    def action_right(self) -> None:
        """Move toward the negative action."""

        if not self._move_focus_in_direction("right"):
            self._focus_if_navigable("right_ynd")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this yes no dialog."""
        self.dismiss(event.button.id == "left_ynd")
