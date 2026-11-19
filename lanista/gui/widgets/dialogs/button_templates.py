# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the button templates interface component."""

from textual.containers import Horizontal
from textual.widgets import Button
from textual.widgets._button import ButtonVariant

from lanista.gui.widgets.action_rows import action_row

SELECTOR_LABEL_WIDTH = 12
SELECTOR_BULLET_COLUMN = 14


def selector_label(
    label: str,
    value: str,
    *,
    label_width: int = SELECTOR_LABEL_WIDTH,
    leading_space: bool = True,
) -> str:
    """Return the selector label value owned by this button templates module."""
    prefix = " " if leading_space else ""
    effective_width = max(label_width, SELECTOR_BULLET_COLUMN - len(prefix) - 1)
    return f"{prefix}{label:<{effective_width}} •  {value}"


def selector_button(label: str, *, id: str, disabled: bool = False) -> Button:
    """Return the selector button value owned by this button templates module."""
    return Button(
        label,
        id=id,
        variant="default",
        disabled=disabled,
        classes="dialog-button-base dialog-selector-button",
    )


def dialog_button(
    label: str,
    *,
    id: str,
    variant: ButtonVariant = "default",
    disabled: bool = False,
    classes: str = "dialog-button-base",
) -> Button:
    """Return the dialog button value owned by this button templates module."""
    return Button(
        label,
        id=id,
        variant=variant,
        disabled=disabled,
        classes=classes,
    )


def dialog_action_row(
    *buttons: Button,
    id: str,
    displayed: bool = True,
) -> Horizontal:
    """Group one to three dialog buttons in the shared equal-width grid."""

    button_count = len(buttons)
    return action_row(
        *buttons,
        id=id,
        displayed=displayed,
        classes=f"dialog-action-row dialog-action-row--{button_count}",
    )


def primary_button(label: str, *, id: str, disabled: bool = False) -> Button:
    """Return the primary button value owned by this button templates module."""
    return dialog_button(
        label,
        id=id,
        variant="primary",
        disabled=disabled,
        classes="dialog-button-base",
    )


def success_button(label: str, *, id: str, disabled: bool = False) -> Button:
    """Return the success button value owned by this button templates module."""
    return dialog_button(label, id=id, variant="success", disabled=disabled)


def warning_button(label: str, *, id: str, disabled: bool = False) -> Button:
    """Return the warning button value owned by this button templates module."""
    return dialog_button(label, id=id, variant="warning", disabled=disabled)


def destructive_button(label: str, *, id: str, disabled: bool = False) -> Button:
    """Return the destructive button value owned by this button templates module."""
    return dialog_button(label, id=id, variant="error", disabled=disabled)


def cancel_button(label: str = "Cancel", *, id: str, disabled: bool = False) -> Button:
    """Cancel the button owned by this button templates module."""
    return dialog_button(label, id=id, variant="default", disabled=disabled)
