# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Static ZalaZONE SUMO route entities injected into generated scenarios."""

from __future__ import annotations

from typing import Any


def ego_vehicle_route_lines() -> list[str]:
    """Return the fixed EGO vehicle element used by generated ZalaZone routes."""
    return [
        '    <vehicle id="EGO" depart="0.00" departSpeed="0" color="0,0,1">',
        '        <route edges="3526.14.64 3526.6.00 3526.0.00 3352.0.00 -4459.0.00 -4459.9.00 '
        '-4459.11.86 -4459.18.00 -4458.0.00 -4593.0.00 -4116.0.00 4312.0.50 4312.0.00 -5021.0.00 '
        '-5019.0.00 -3593.0.00 -3593.22.00 -6322.0.00 -6322.2.00 -6353.0.00 6529.83.00 6529.0.00 '
        '7690.27.00 7690.8.00 7690.0.00 6321.63.00 6321.56.41"/>',
        "    </vehicle>",
    ]


class ZalaZoneRouteEntitiesProvider:
    """Provide the fixed, stateless ZalaZONE route entities."""

    __slots__ = ()

    def route_entity_lines(self) -> list[str]:
        """Return the ZalaZONE EGO vehicle XML lines."""
        return ego_vehicle_route_lines()

    def __copy__(self) -> ZalaZoneRouteEntitiesProvider:
        """Return the shared stateless provider for shallow copies."""
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> ZalaZoneRouteEntitiesProvider:
        """Return the shared stateless provider for deep copies."""
        return self

    def __reduce__(self) -> tuple[object, tuple[()]]:
        """Restore the module singleton when deserializing the provider."""
        return _zala_zone_route_entities_provider, ()


ZALA_ZONE_ROUTE_ENTITIES_PROVIDER = ZalaZoneRouteEntitiesProvider()
"""Pickle-safe singleton used by each built-in automotive model plugin."""


def _zala_zone_route_entities_provider() -> ZalaZoneRouteEntitiesProvider:
    """Return the canonical ZalaZONE route-entities provider singleton."""
    return ZALA_ZONE_ROUTE_ENTITIES_PROVIDER
