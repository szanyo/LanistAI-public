# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Shared directional keyboard focus traversal for Textual components."""

from textual._context import NoActiveAppError
from textual.css.query import NoMatches
from textual.widget import Widget


class DirectionalFocusMixin:
    """Traverse a component-defined focus graph while skipping unavailable widgets."""

    _NAVIGATION_GRAPH: dict[str, dict[str, str]]

    @staticmethod
    def _is_widget_navigable(widget: Widget) -> bool:
        """Return whether widget navigable applies to this directional focus mixin."""
        return widget.display and not getattr(widget, "disabled", False) and widget.can_focus

    def _focus_if_navigable(self, widget_id: str) -> bool:
        """Return the focus if navigable value owned by this directional focus mixin."""
        try:
            widget = self.query_one(f"#{widget_id}", Widget)
        except NoMatches:
            return False
        if not self._is_widget_navigable(widget):
            return False
        widget.focus()
        return True

    def _focused_navigation_id(self) -> str | None:
        """Return the focused navigation id value owned by this directional focus mixin."""
        try:
            focused = self.app.focused
        except NoActiveAppError:
            return None
        return focused.id if focused is not None else None

    def _next_focus_id(self, current_id: str, direction: str) -> str | None:
        """Return the next focus id value owned by this directional focus mixin."""
        return self._NAVIGATION_GRAPH.get(current_id, {}).get(direction)

    def _move_focus_in_direction(self, direction: str) -> bool:
        """Move the focus in direction owned by this directional focus mixin."""
        focused_id = self._focused_navigation_id()
        if not focused_id:
            return False

        current_id = focused_id
        visited: set[str] = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            next_id = self._next_focus_id(current_id, direction)
            if not next_id:
                return False
            if self._focus_if_navigable(next_id):
                return True
            current_id = next_id
        return False
