# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the info panel interface component."""

from lanista.gui.utils.timestamps import format_local_timestamp
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Static


class InfoPanel(Vertical):
    """Present info in the Textual interface."""

    _LABEL_WIDTH = 12

    def __init__(
        self,
        *children: Widget,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        markup: bool = True,
    ) -> None:
        """Initialize the info panel and its owned state."""
        super().__init__(
            *children,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
            markup=markup,
        )
        self.border_title = "Info Panel"

    @classmethod
    def _format_row(cls, label: str, value: str) -> Text:
        """Format the row for presentation by this info panel."""
        row = Text()
        row.append(f"{label:<{cls._LABEL_WIDTH}}", style="dim")
        row.append("  ")
        row.append(value, style="bold")
        return row

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this info panel."""
        yield Static(self._format_row("Model", "No model selected"), id="info_model_name", classes="info_panel_item")
        yield Static(self._format_row("Framework", "-"), id="info_framework", classes="info_panel_item")
        yield Static(self._format_row("Phase", "-"), id="info_phase", classes="info_panel_item")
        yield Static(self._format_row("Session", "-"), id="info_session", classes="info_panel_item")
        yield Static(self._format_row("Checkpoint", "-"), id="info_checkpoint", classes="info_panel_item")
        yield Static(self._format_row("TensorBoard", "offline"), id="info_tensorboard", classes="info_panel_item")

    def set_model_info(
        self,
        model_name: str | None,
        framework_name: str | None,
    ) -> None:
        """Set the model info owned by this info panel."""
        self.query_one("#info_model_name", Static).update(
            self._format_row("Model", model_name or "No model selected")
        )
        self.query_one("#info_framework", Static).update(
            self._format_row("Framework", framework_name or "-")
        )

    def set_runtime_info(
        self,
        phase_name: str,
        session_name: str | None,
        parent_session_name: str | None,
        checkpoint_name: str | None,
        tensorboard_status: str,
        session_created_at_raw: str | None = None,
    ) -> None:
        """Set the runtime info owned by this info panel."""
        session_text = session_name or "-"
        if parent_session_name:
            session_text = f"{session_text} <- {parent_session_name}"

        if session_name and session_created_at_raw:
            local_text = format_local_timestamp(session_created_at_raw)
            session_text = f"{session_name} | local: {local_text} | raw: {session_created_at_raw}"

        self.query_one("#info_phase", Static).update(self._format_row("Phase", phase_name))
        self.query_one("#info_session", Static).update(self._format_row("Session", session_text))
        self.query_one("#info_checkpoint", Static).update(self._format_row("Checkpoint", checkpoint_name or "-"))
        self.query_one("#info_tensorboard", Static).update(self._format_row("TensorBoard", tensorboard_status))
