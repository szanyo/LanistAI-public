# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Framework-neutral classification of mandatory EGO vehicle observations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from lanista.reinforcengine.utils.simulation.ports import VehiclePort


class EgoVehiclePresence(StrEnum):
    """Domain meaning of one explicit EGO presence observation."""

    ACTIVE = "active"
    ROUTE_END = "route_end"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class EgoVehicleObservation:
    """Current EGO road data or an explicitly confirmed absence."""

    presence: EgoVehiclePresence
    road_id: str | None = None


def observe_ego_vehicle(
    simulator: VehiclePort,
    *,
    previous_road_id: str | None,
    route_end_road_id: str,
    vehicle_id: str = "EGO",
) -> EgoVehicleObservation:
    """Classify confirmed absence while propagating all adapter failures."""

    if not simulator.has_vehicle(vehicle_id):
        presence = (
            EgoVehiclePresence.ROUTE_END
            if previous_road_id == route_end_road_id
            else EgoVehiclePresence.UNAVAILABLE
        )
        return EgoVehicleObservation(presence=presence)
    return EgoVehicleObservation(
        presence=EgoVehiclePresence.ACTIVE,
        road_id=simulator.get_vehicle_road_id(vehicle_id),
    )
