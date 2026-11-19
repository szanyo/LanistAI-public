# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Launch a desktop browser without attaching it to the terminal UI."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable


def open_browser_tab(
    url: str,
    *,
    platform: str | None = None,
    process_factory: Callable[..., object] = subprocess.Popen,
    start_file: Callable[[str], object] | None = None,
) -> bool:
    """Open ``url`` while preventing launcher output from corrupting a TUI.

    Linux desktop launchers and the browser processes they create otherwise
    inherit the application's terminal. GTK diagnostics can then write
    directly over Textual's alternate-screen rendering.
    """

    selected_platform = sys.platform if platform is None else platform
    if selected_platform == "win32":
        windows_opener = start_file or getattr(os, "startfile", None)
        if windows_opener is None:
            return False
        try:
            windows_opener(url)
        except OSError:
            return False
        return True

    command = ("open", url) if selected_platform == "darwin" else ("xdg-open", url)
    try:
        process_factory(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    except OSError:
        return False
    return True


__all__ = ["open_browser_tab"]
