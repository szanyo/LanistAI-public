# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.


"""Provide the text cells interface component."""

def fit_text_cell(value: str, width: int) -> str:
    """Pad or truncate text to the requested presentation width."""
    text = str(value or "")
    if len(text) > width:
        if width <= 2:
            return "." * width
        return text[: max(0, width - 2)] + ".."
    return text.ljust(width)
