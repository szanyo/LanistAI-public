# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Application policy for process images that used an embedded SUMO GUI."""

from __future__ import annotations

from lanista.reinforcengine.application_errors import ApplicationError
from lanista.reinforcengine.utils.simulation.native_gui_usage import (
    PROCESS_NATIVE_GUI_USAGE,
    NativeGuiUsageTracker,
)


class NativeGuiRestartGate:
    """Forbid native-session reuse after a GUI started in this process."""

    def __init__(
        self,
        usage: NativeGuiUsageTracker = PROCESS_NATIVE_GUI_USAGE,
    ) -> None:
        """Initialize the gate over process-scoped startup evidence."""
        self._usage = usage

    @property
    def restart_required(self) -> bool:
        """Return whether a fresh process image is now mandatory."""
        return self._usage.started

    def require_fresh_process(self) -> None:
        """Reject reuse after an embedded GUI has started."""
        if self.restart_required:
            raise ApplicationError(
                "Restart LanistAI before changing configuration or starting another run."
            )


__all__ = ["NativeGuiRestartGate"]
