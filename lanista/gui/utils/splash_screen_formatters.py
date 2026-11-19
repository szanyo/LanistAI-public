# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

# Function to center each line of a multi-line string
"""Format display values for splash screen."""

def center_multiline_text(multiline_text: str, width: int) -> str:
    """Return the center multiline text value owned by this splash screen formatters module."""
    return "\n".join(line.center(width) for line in multiline_text.split("\n")) + "\n"


def center_text_block(
    multiline_text: str,
    width: int,
    *,
    horizontal_offset: int = 0,
) -> str:
    """Center a text block while preserving the relative position of its lines."""

    lines = multiline_text.split("\n")
    block_width = max((len(line) for line in lines), default=0)
    available_padding = max(width - block_width, 0)
    left_padding = min(
        max((available_padding // 2) + horizontal_offset, 0),
        available_padding,
    )
    right_padding = available_padding - left_padding
    return "\n".join(
        (" " * left_padding) + line.ljust(block_width) + (" " * right_padding)
        for line in lines
    ) + "\n"


def center_text_vertically(multiline_text: str, width: int, height: int) -> str:
    """Pad a text canvas vertically without truncating its existing content."""

    lines = multiline_text.splitlines()
    canvas_height = max(height, len(lines))
    padding = canvas_height - len(lines)
    top_padding = padding // 2
    bottom_padding = padding - top_padding
    empty_line = " " * width
    canvas_lines = (
        [empty_line] * top_padding
        + lines
        + [empty_line] * bottom_padding
    )
    return "\n".join(canvas_lines) + "\n"


def create_empty_line(width: int) -> str:
    """Create the empty line owned by this splash screen formatters module."""
    return " " * width + "\n"


def remove_blank_lines_from_end(text: str) -> str:
    """Remove the blank lines from end owned by this splash screen formatters module."""
    lines = text.split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def remove_blank_lines_from_start(text: str) -> str:
    """Remove the blank lines from start owned by this splash screen formatters module."""
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)
