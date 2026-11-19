# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the secure yes no dialog definitions interface component."""

from lanista.gui.widgets.dialogs.secure_yes_no_dialog import SecureYesNoDialog

def secure_quit_dialog() -> SecureYesNoDialog:
    """Return the secure quit dialog value owned by this secure yes no dialog definitions module."""
    return SecureYesNoDialog(
        question="Are you sure you want to quit?",
        left_button="Yes",
        left_variant="error",
        right_button="No",
        right_variant="primary",
        id="quit_modal_screen"
    )


def secure_stop_training_dialog():
    """Return the secure stop training dialog value owned by this secure yes no dialog definitions
    module.
    """
    return SecureYesNoDialog(
        question=(
            "Are you sure you want to stop training?\n\n"
            "This training session will be closed for continuing from the main panel. "
            "Create a new session to train again."
        ),
        left_button="Stop training",
        left_variant="error",
        right_button="Keep running",
        right_variant="primary",
        id="stop_training_modal_screen",
    )
