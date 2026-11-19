# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the coordinator interface component."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from lanista.gui.utils.navigation.graph import NavigationGraph


class _TimerScreen(Protocol):
    """Define the typed boundary for timer screen."""

    def set_timer(
        self,
        delay: float,
        callback: Callable[[], None],
    ) -> object:
        """Set the timer owned by this timer screen."""
        ...


class NavigationCoordinator:
    """Coordinate navigation operations."""

    def __init__(
        self,
        screen: _TimerScreen,
        graph: NavigationGraph,
        opener_map: dict[str, Callable[[], None]],
        reopen_delay_seconds: float = 0.15,
    ) -> None:
        """Initialize the navigation coordinator and its owned state."""
        self._screen = screen
        self._graph = graph
        self._opener_map = opener_map
        self._reopen_delay_seconds = reopen_delay_seconds

    def schedule_later(
        self,
        opener: Callable[[], None],
        *,
        guard: Callable[[], bool] | None = None,
    ) -> bool:
        """Schedule an opener, optionally dropping it when its owner is stale."""

        def guarded_opener() -> None:
            """Perform the guarded opener step within the enclosing schedule later workflow."""
            if guard is None or guard():
                opener()

        self._screen.set_timer(self._reopen_delay_seconds, guarded_opener)
        return True

    def reopen_route_later(
        self,
        route_id: str,
        opener: Callable[[], None] | None = None,
        *,
        guard: Callable[[], bool] | None = None,
    ) -> bool:
        """Reopen the route later owned by this navigation coordinator."""
        selected_opener = opener or self._opener_map.get(route_id)
        if selected_opener is None:
            return False
        return self.schedule_later(selected_opener, guard=guard)

    def reopen_parent_later(
        self,
        child_route_id: str,
        opener: Callable[[], None] | None = None,
        *,
        guard: Callable[[], bool] | None = None,
    ) -> bool:
        """Reopen the parent later owned by this navigation coordinator."""
        parent_route_id = self._graph.get_parent_route_id(child_route_id)
        if parent_route_id is None:
            return False
        return self.reopen_route_later(
            parent_route_id,
            opener=opener,
            guard=guard,
        )
