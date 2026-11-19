# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the confirmation dialog content interface component."""

from dataclasses import dataclass

from textual.widgets._button import ButtonVariant


@dataclass(frozen=True, slots=True)
class ConfirmationDialogContent:
    """Immutable presentation content shared by confirmation dialogs."""

    question: str = "Are you sure?"
    affirmative_label: str = "Yes"
    affirmative_variant: ButtonVariant = "error"
    negative_label: str = "No"
    negative_variant: ButtonVariant = "primary"
