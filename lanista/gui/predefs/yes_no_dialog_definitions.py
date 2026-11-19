# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the yes no dialog definitions interface component."""

from lanista.gui.widgets.dialogs.yes_no_dialog import YesNoDialog


def quit_dialog() -> YesNoDialog:
    """Return the quit dialog value owned by this yes no dialog definitions module."""
    return YesNoDialog(
        question="Are you sure you want to quit?",
        left_button="Yes",
        left_variant="error",
        right_button="No",
        right_variant="primary",
        id="quit_modal_screen"
    )
