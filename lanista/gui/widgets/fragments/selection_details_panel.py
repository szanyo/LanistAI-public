# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the selection details panel interface component."""

from dataclasses import dataclass
from typing import Optional

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static


class SelectionDetailsPanel(Vertical):
    """Reusable details panel for selection dialogs and metadata views."""

    DEFAULT_CSS = """
    SelectionDetailsPanel {
        padding: 0;
        height: 11;
        min-height: 11;
    }

    #selection_details_panel_body {
        height: 1fr;
        min-height: 11;
        padding: 0 1;
        border-title-align: left;
        border: round $primary-lighten-1 50%;
        background: $surface;
        color: $text;
    }
    """

    @dataclass
    class DetailRow:
        """Carry the data required for detail row."""

        label: str
        value: str
        value_style: str = "bold"

    def __init__(self, *, title: str = "Details", empty_message: str = "No item selected.", **kwargs):
        """Initialize the selection details panel and its owned state."""
        super().__init__(**kwargs)
        self._title = title
        self._empty_message = empty_message

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this selection details panel."""
        body = Static(id="selection_details_panel_body")
        body.border_title = self._title
        yield body

    def set_title(self, title: str) -> None:
        """Set the title owned by this selection details panel."""
        self._title = str(title or "Details")
        self.query_one("#selection_details_panel_body", Static).border_title = self._title

    def set_empty_message(self, message: str) -> None:
        """Set the empty message owned by this selection details panel."""
        self._empty_message = str(message or "No item selected.")

    def show_message(self, message: Optional[str] = None) -> None:
        """Show the message owned by this selection details panel."""
        body = self.query_one("#selection_details_panel_body", Static)
        body.border_title = self._title
        body.update(str(message or self._empty_message))

    def show_rows(self, rows: list["SelectionDetailsPanel.DetailRow"], *, title: Optional[str] = None) -> None:
        """Show the rows owned by this selection details panel."""
        body = self.query_one("#selection_details_panel_body", Static)
        body.border_title = str(title or self._title)

        if not rows:
            body.update(self._empty_message)
            return

        label_width = max(len(str(row.label or "")) for row in rows)
        rendered = Text()
        for index, row in enumerate(rows):
            rendered.append(f"{str(row.label or ''):<{label_width}}", style="dim")
            rendered.append("  ")
            rendered.append(str(row.value or "-"), style=row.value_style)
            if index < len(rows) - 1:
                rendered.append("\n")
        body.update(rendered)
