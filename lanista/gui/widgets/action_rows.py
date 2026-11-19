# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Build reusable equal-width action rows for Textual views and dialogs."""

from textual.containers import Horizontal
from textual.widgets import Button


def action_row(
    *buttons: Button,
    id: str,
    displayed: bool = True,
    classes: str = "",
) -> Horizontal:
    """Group one to three buttons in the shared equal-width action grid."""

    button_count = len(buttons)
    if button_count not in {1, 2, 3}:
        raise ValueError("Action rows require one to three buttons.")
    shared_classes = f"action-row action-row--{button_count}"
    row = Horizontal(
        *buttons,
        id=id,
        classes=f"{shared_classes} {classes}".strip(),
    )
    row.display = displayed
    return row


__all__ = ["action_row"]
