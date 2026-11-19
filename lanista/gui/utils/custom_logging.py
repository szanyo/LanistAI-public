# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the custom logging interface component."""

from typing import cast

from rich.console import Console
from rich.logging import RichHandler

from lanista.gui.widgets.fragments.logcat_console_panel import LogCatConsolePanel

_LOGCAT_TIME_FORMAT = "[%Y-%m-%d %H:%M:%S]"


class CustomRichLogHandler(RichHandler):
    """A RichHandler that initializes a LogCatConsole without a singleton."""

    def __init__(
        self,
        *,
        show_time: bool = True,
        rich_tracebacks: bool = False,
    ) -> None:
        """Initialize the custom rich log handler and its owned state."""
        self.log_widget = LogCatConsolePanel(id="logcat-console")
        super().__init__(
            console=cast(Console, self.log_widget),
            show_time=show_time,
            rich_tracebacks=rich_tracebacks,
            log_time_format=_LOGCAT_TIME_FORMAT,
        )
