# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the graph interface component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from lanista.gui.utils.navigation import route_ids


@dataclass(frozen=True)
class NavigationNode:
    """Carry the data required for navigation node."""

    route_id: str
    parent_route_id: Optional[str] = None
    return_to_parent_on_cancel: bool = True
    return_to_parent_on_noop: bool = True


class NavigationGraph:
    """Store and navigate the navigation structure."""

    def __init__(self, nodes: list[NavigationNode]):
        """Initialize the navigation graph and its owned state."""
        self._nodes = {node.route_id: node for node in nodes}

    def get(self, route_id: str) -> Optional[NavigationNode]:
        """Return the get value owned by this navigation graph."""
        return self._nodes.get(route_id)

    def get_parent_route_id(self, route_id: str) -> Optional[str]:
        """Return the parent route id exposed by this navigation graph."""
        node = self.get(route_id)
        if node is None:
            return None
        return node.parent_route_id


def build_default_navigation_graph() -> NavigationGraph:
    """Build the default navigation graph required by this graph module."""
    return NavigationGraph(
        [
            NavigationNode(route_ids.HOME_SCREEN, parent_route_id=None, return_to_parent_on_cancel=False, return_to_parent_on_noop=False),
            NavigationNode(route_ids.CONFIGURATION_DIALOG, parent_route_id=route_ids.HOME_SCREEN),
            NavigationNode(route_ids.LOAD_MODEL_DIALOG, parent_route_id=route_ids.CONFIGURATION_DIALOG),
            NavigationNode(route_ids.MODEL_CREATION_SCREEN, parent_route_id=route_ids.CONFIGURATION_DIALOG),
            NavigationNode(route_ids.NEW_SESSION_DIALOG, parent_route_id=route_ids.CONFIGURATION_DIALOG),
            NavigationNode(route_ids.MODEL_DETAIL_DIALOG, parent_route_id=route_ids.HOME_SCREEN),
            NavigationNode(route_ids.SECURE_DELETE_DIALOG, parent_route_id=route_ids.MODEL_DETAIL_DIALOG),
            NavigationNode(route_ids.YES_NO_DIALOG, parent_route_id=route_ids.HOME_SCREEN),
            NavigationNode(route_ids.SECURE_YES_NO_DIALOG, parent_route_id=route_ids.HOME_SCREEN),
        ]
    )
