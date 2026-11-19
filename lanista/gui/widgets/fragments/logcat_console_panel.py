# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the logcat console panel interface component."""

from contextlib import suppress

from textual._context import NoActiveAppError
from textual.widgets import RichLog


class LogCatConsolePanel(RichLog):
    """Present log cat console in the Textual interface."""

    DEFAULT_MAX_LINES = 2000
    file = False

    def __init__(self, *, id: str | None = None) -> None:
        """Initialize the log cat console panel and its owned state."""
        super().__init__(max_lines=self.DEFAULT_MAX_LINES, id=id)
        self.border_title = "LogCat Console"
        self.auto_scroll = True

    def print(self, content: object) -> None:
        """Handle the print operation owned by this log cat console panel."""
        with suppress(NoActiveAppError):
            self.write(content)
