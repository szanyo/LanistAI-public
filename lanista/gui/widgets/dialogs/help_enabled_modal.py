# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the help enabled modal interface component."""

from time import monotonic

from lanista.gui.widgets.dialogs.help_dialog import HelpDialog


class HelpEnabledModalMixin:
    """Add reusable help enabled modal behavior to interface components."""

    _HELP_TOPIC: str

    def action_open_help_dialog(self) -> None:
        """Handle the open help dialog user action for this help enabled modal mixin."""
        mounted_at = getattr(self, "_mounted_at", 0.0)
        guard_seconds = getattr(self, "_HELP_SHORTCUT_GUARD_SECONDS", getattr(self, "_SHORTCUT_GUARD_SECONDS", 0.0))
        if mounted_at > 0 and guard_seconds > 0 and (monotonic() - mounted_at) < guard_seconds:
            return
        self.app.push_screen(HelpDialog(topic_id=self._HELP_TOPIC))
