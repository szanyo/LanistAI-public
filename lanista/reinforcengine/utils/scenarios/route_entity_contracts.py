# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Domain-neutral route-entity injection contract."""

from typing import Protocol


class RouteEntitiesProvider(Protocol):
    """Supply model-domain entities appended to generated SUMO routes."""

    def route_entity_lines(self) -> list[str]:
        """Return complete XML lines for the route entities to append."""
        ...
