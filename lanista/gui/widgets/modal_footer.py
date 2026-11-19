# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the modal footer interface component."""

from __future__ import annotations

from collections.abc import Sequence

from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.widgets import Static

BindingCell = tuple[str, str]


def _symmetric_gaps(
    *,
    available_width: int,
    content_width: int,
    gap_slots: int,
) -> tuple[int, ...]:
    """Distribute spare width from the outside towards the centre."""
    remaining_width = max(gap_slots, available_width - content_width)
    base_gap = max(1, remaining_width // gap_slots)
    gaps = [base_gap] * gap_slots
    extra = max(0, remaining_width - (base_gap * gap_slots))

    distribution_order: list[int] = []
    left_index = 0
    right_index = gap_slots - 1
    while left_index <= right_index:
        distribution_order.append(left_index)
        if right_index != left_index:
            distribution_order.append(right_index)
        left_index += 1
        right_index -= 1
    for index in distribution_order[:extra]:
        gaps[index] += 1
    return tuple(gaps)


def _column_starts(
    *,
    cell_width: int,
    column_count: int,
    gaps: Sequence[int],
) -> tuple[int, ...]:
    """Return the column starts value owned by this modal footer module."""
    starts: list[int] = []
    cursor = gaps[0]
    for column_index in range(column_count):
        starts.append(cursor)
        cursor += cell_width
        if column_index + 1 < len(gaps):
            cursor += gaps[column_index + 1]
    return tuple(starts)


def _append_binding_row(
    text: Text,
    *,
    row: Sequence[BindingCell],
    key_width: int,
    cell_width: int,
    column_count: int,
    column_starts: Sequence[int],
    gaps: Sequence[int],
) -> None:
    """Append the binding row owned by this modal footer module."""
    leading_empty_columns = max(0, (column_count - len(row)) // 2)
    cursor = 0
    for column_index, (key, label) in enumerate(row):
        actual_column = leading_empty_columns + column_index
        target_start = column_starts[actual_column]
        if target_start > cursor:
            text.append(" " * (target_start - cursor))
            cursor = target_start

        text.append(key.rjust(key_width), style="bold #ffb347")
        text.append(" ")
        text.append(label, style="#f5f5f5")
        cursor += key_width + 1 + len(label)

        if column_index != len(row) - 1:
            gap = max(
                1,
                cell_width
                - (key_width + 1 + len(label))
                + gaps[actual_column + 1],
            )
            text.append(" " * gap)
            cursor += gap


def _render_binding_rows(
    rows: Sequence[Sequence[BindingCell]],
    *,
    key_width: int,
    cell_width: int,
    column_count: int,
    available_width: int,
) -> Text:
    """Return the render binding rows value owned by this modal footer module."""
    text = Text(no_wrap=False, overflow="fold")
    if not rows:
        return text

    content_width = cell_width * column_count
    gaps = _symmetric_gaps(
        available_width=available_width,
        content_width=content_width,
        gap_slots=column_count + 1,
    )
    column_starts = _column_starts(
        cell_width=cell_width,
        column_count=column_count,
        gaps=gaps,
    )
    for row_index, row in enumerate(rows):
        if row_index:
            text.append("\n")
        _append_binding_row(
            text,
            row=row,
            key_width=key_width,
            cell_width=cell_width,
            column_count=column_count,
            column_starts=column_starts,
            gaps=gaps,
        )
    return text


class ModalFooter(Static):
    """A footer-like binding bar that stays inside a modal container."""

    DEFAULT_CLASSES = "modal-footer"

    def __init__(
        self,
        bindings: Sequence[BindingCell | None],
        *,
        id: str | None = None,
        classes: str | None = None,
        max_per_row: int = 4,
    ) -> None:
        """Initialize the modal footer and its owned state."""
        super().__init__("", id=id, classes=classes)
        self._footer_bindings = list(bindings)
        self._max_per_row = max(1, int(max_per_row))

    @staticmethod
    def _display_key(key: str, key_display: str | None = None) -> str:
        """Return the display key value owned by this modal footer."""
        if key_display and "," not in key:
            return key_display
        return {
            "escape": "esc",
        }.get(key, key)

    @classmethod
    def _display_keys_for_bindings(cls, bindings: Sequence[Binding]) -> str:
        """Return the display keys for bindings value owned by this modal footer."""
        keys: list[str] = []
        seen: set[str] = set()
        for binding in bindings:
            raw_keys = [segment.strip() for segment in str(binding.key).split(",") if segment.strip()]
            for index, raw_key in enumerate(raw_keys):
                display = cls._display_key(
                    raw_key,
                    binding.key_display if len(raw_keys) == 1 and index == 0 else None,
                )
                if display in seen:
                    continue
                seen.add(display)
                keys.append(display)
        return " | ".join(keys)

    @classmethod
    def from_binding_rows(
        cls,
        bindings: Sequence[Binding],
        rows: Sequence[Sequence[str | tuple[str, str]]],
        *,
        id: str | None = None,
        classes: str | None = None,
        max_per_row: int | None = None,
        label_overrides: dict[str, str] | None = None,
    ) -> ModalFooter:
        """Create this modal footer's value from binding rows."""
        grouped_bindings: dict[str, list[Binding]] = {}
        for binding in bindings:
            if not binding.show or binding.system:
                continue
            grouped_bindings.setdefault(binding.action, []).append(binding)

        resolved_bindings: list[BindingCell | None] = []
        for row_index, row in enumerate(rows):
            if row_index:
                resolved_bindings.append(None)
            for entry in row:
                if isinstance(entry, tuple):
                    resolved_bindings.append((str(entry[0]), str(entry[1])))
                    continue
                action = str(entry)
                action_bindings = grouped_bindings.get(action, [])
                if not action_bindings:
                    continue
                label = (label_overrides or {}).get(action) or next(
                    (binding.description for binding in action_bindings if binding.description),
                    action,
                )
                resolved_bindings.append((cls._display_keys_for_bindings(action_bindings), label))

        computed_max_per_row = max_per_row or max((len(row) for row in rows), default=1)
        return cls(
            resolved_bindings,
            id=id,
            classes=classes,
            max_per_row=computed_max_per_row,
        )

    def on_mount(self) -> None:
        """Handle the mount event for this modal footer."""
        self.update(self._build_text())

    def on_resize(self, _: events.Resize) -> None:
        """Handle the resize event for this modal footer."""
        self.update(self._build_text())

    @staticmethod
    def _metrics(bindings: list[BindingCell]) -> tuple[int, int, int]:
        """Return the metrics value owned by this modal footer."""
        if not bindings:
            return 0, 0, 0
        key_width = max(len(key) for key, _ in bindings)
        label_width = max(len(label) for _, label in bindings)
        content_width = max(key_width, label_width)
        cell_width = (content_width * 2) + 1
        return key_width, label_width, cell_width

    def _layout_rows(self) -> tuple[list[list[BindingCell]], int, int, int, int]:
        """Return the layout rows value owned by this modal footer."""
        groups: list[list[BindingCell]] = [[]]
        for binding in self._footer_bindings:
            if binding is None:
                if groups[-1]:
                    groups.append([])
                continue
            groups[-1].append((str(binding[0]), str(binding[1])))
        if not groups[-1]:
            groups.pop()
        if not groups:
            return [], 0, 0, 0, 0

        flat_bindings = [binding for group in groups for binding in group]
        key_width, label_width, cell_width = self._metrics(flat_bindings)

        rows: list[list[BindingCell]] = []
        for group in groups:
            for start in range(0, len(group), self._max_per_row):
                rows.append(group[start:start + self._max_per_row])
        column_count = max((len(row) for row in rows), default=0)
        return rows, key_width, label_width, cell_width, column_count

    def get_row_count(self) -> int:
        """Return the row count exposed by this modal footer."""
        rows, _, _, _, _ = self._layout_rows()
        return max(1, len(rows)) if self._footer_bindings else 1

    def _build_text(self) -> Text:
        """Build the text required by this modal footer."""
        rows, key_width, _, cell_width, column_count = self._layout_rows()
        if not rows:
            return Text(no_wrap=False, overflow="fold")
        content_width = cell_width * column_count
        gap_slots = column_count + 1
        available_width = (
            max(1, self.size.width)
            if self.size.width
            else (content_width + gap_slots * 2)
        )
        return _render_binding_rows(
            rows,
            key_width=key_width,
            cell_width=cell_width,
            column_count=column_count,
            available_width=available_width,
        )
