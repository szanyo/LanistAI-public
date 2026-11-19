# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the secure delete dialog definitions interface component."""

from lanista.gui.widgets.dialogs.secure_yes_no_dialog import SecureYesNoDialog


def secure_delete_session_dialog(session_name: str) -> SecureYesNoDialog:
    """Return the secure delete session dialog value owned by this secure delete dialog definitions
    module.
    """
    return SecureYesNoDialog(
        question=(
            f"Are you sure you want to DELETE session:\n"
            f"  '{session_name}'?\n\n"
            f"This cannot be undone.\n"
            f"All checkpoint files and tensorboard logs will be removed."
        ),
        left_button="Delete",
        left_variant="error",
        right_button="Cancel",
        right_variant="primary",
    )


def secure_delete_model_dialog(model_name: str) -> SecureYesNoDialog:
    """Return the secure delete model dialog value owned by this secure delete dialog definitions
    module.
    """
    return SecureYesNoDialog(
        question=(
            f"Are you sure you want to DELETE model:\n"
            f"  '{model_name}'\n"
            f"and ALL its training sessions?\n\n"
            f"This cannot be undone.\n"
            f"The entire model folder will be permanently removed."
        ),
        left_button="Delete All",
        left_variant="error",
        right_button="Cancel",
        right_variant="primary",
    )


def secure_delete_checkpoint_dialog(checkpoint_name: str, session_name: str) -> SecureYesNoDialog:
    """Return the secure delete checkpoint dialog value owned by this secure delete dialog
    definitions module.
    """
    return SecureYesNoDialog(
        question=(
            f"Are you sure you want to DELETE checkpoint:\n"
            f"  '{checkpoint_name}'\n"
            f"from session '{session_name}'?\n\n"
            f"This cannot be undone.\n"
            f"The checkpoint files and checkpoint metadata will be removed."
        ),
        left_button="Delete",
        left_variant="error",
        right_button="Cancel",
        right_variant="primary",
    )


def secure_delete_tensorboard_comparison_pack_dialog(pack_name: str) -> SecureYesNoDialog:
    """Return the secure delete tensorboard comparison pack dialog value owned by this secure
    delete dialog definitions module.
    """
    return SecureYesNoDialog(
        question=(
            f"Are you sure you want to DELETE comparison pack:\n"
            f"  '{pack_name}'?\n\n"
            f"This cannot be undone.\n"
            f"The curated TensorBoard comparison folder will be removed."
        ),
        left_button="Delete",
        left_variant="error",
        right_button="Cancel",
        right_variant="primary",
    )
