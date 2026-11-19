# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed Zala route metadata and simulator-context calculations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from lanista.domains.automotive.zala_zone.reference_speed import (
    zala_zone_reference_speed_mps,
)
from lanista.reinforcengine.utils.simulation.intersection_metadata import (
    TRAFFIC_LIGHT_STATE_GREEN,
    TRAFFIC_LIGHT_STATE_RED,
    TRAFFIC_LIGHT_STATE_RED_YELLOW,
    TRAFFIC_LIGHT_STATE_UNKNOWN,
    TRAFFIC_LIGHT_STATE_YELLOW,
)
from lanista.reinforcengine.utils.simulation.ports import (
    EntitySimulationPort,
    SimulationRead,
    SimulationReadQuality,
)

type PositionXY = tuple[float, float]
type ControlledLinks = tuple[tuple[tuple[object, ...], ...], ...]

ROUTE_INTERSECTION_NUMBERS: Mapping[str, int] = MappingProxyType(
    {
        "3526.14.64": 1,
        ":3526.14.64_4": 1,
        "3526.6.00": 1,
        ":3526.6.00_3": 1,
        "3526.0.00": 1,
        ":18_9": 1,
        "3352.0.00": 2,
        ":25_0": 2,
        "-4459.0.00": 2,
        ":4459.9.00": 2,
        ":4459.9.00_0": 2,
        "-4459.9.00": 2,
        ":4459.11.86_0": 2,
        "-4459.11.86": 2,
        "-4459.18.00": 2,
        ":4459.18.00_0": 2,
        ":23_0": 2,
        "-4458.0.00": 3,
        ":16_2": 3,
        "-4593.0.00": 3,
        ":24_1": 3,
        ":22_1": 3,
        "-4116.0.00": 3,
        "4312.0.50": 3,
        ":4312.0.50_0": 3,
        "4312.0.00": 3,
        ":43_1": 3,
        ":43_4": 3,
        "-5021.0.00": 4,
        ":27_2": 4,
        "-5019.0.00": 5,
        ":19_7": 5,
        "-3593.0.00": 6,
        ":3593.22.00_0": 6,
        "-3593.22.00": 6,
        ":31_4": 6,
        ":28_15": 7,
        "-6322.0.00": 7,
        ":6322.2.00_1": 7,
        "-6322.2.00": 7,
        "-6322.2.00:": 7,
        ":30_6": 7,
        "-6353.0.00": 7,
        "6529.83.00": 7,
        ":6529.83.00_0": 7,
        "6529.0.00": 7,
        ":35_9": 7,
        "7690.27.00": 8,
        ":7690.27.00_0": 8,
        "7690.8.00": 8,
        ":7690.8.00_0": 8,
        "7690.0.00": 8,
        ":2_5": 8,
        "6321.63.00": 9,
        ":6321.63.00_4": 9,
        "6321.56.41": 9,
    }
)


def reference_speed_mps(simulator: EntitySimulationPort) -> float:
    """Return the route reference speed for the mandatory EGO position."""

    ego_x_m, ego_y_m = simulator.vehicle.getPosition("EGO")
    return zala_zone_reference_speed_mps(ego_x_m, ego_y_m)


def nearby_vehicle_ids(
    simulator: EntitySimulationPort,
    *lane_ids: str,
) -> list[str]:
    """Return the legacy de-duplicated vehicle identifiers from several lanes."""

    vehicle_ids: list[str] = []
    for lane_id in lane_ids:
        vehicle_ids += simulator.lane.getLastStepVehicleIDs(lane_id)
    return list(set(vehicle_ids))


def euclidean_distance(point_a: Sequence[float], point_b: Sequence[float]) -> float:
    """Return the Euclidean distance between coordinate sequences."""

    return math.sqrt(
        sum(
            (coordinate_a - coordinate_b) ** 2
            for coordinate_a, coordinate_b in zip(
                point_a,
                point_b,
                strict=False,
            )
        )
    )


def nearest_pedestrian_id(
    simulator: EntitySimulationPort,
    ego_position_xy_m: Sequence[float],
) -> str:
    """Return the identifier of the pedestrian nearest to EGO."""

    pedestrian_ids = simulator.person.getIDList()
    return min(
        pedestrian_ids,
        key=lambda pedestrian_id: euclidean_distance(
            ego_position_xy_m,
            simulator.person.getPosition(pedestrian_id),
        ),
    )


def calculate_station_environment(
    simulator: EntitySimulationPort,
    station_environment_m: list[float],
    vehicle_ids: Sequence[str],
    target_coordinates_m: Sequence[float],
    target_road_id: str,
    multiplier: float = -1,
) -> tuple[list[float], list[str]]:
    """Project relevant vehicle positions onto the legacy station axis."""

    car_environment: list[str] = []
    for index, vehicle_id in enumerate(vehicle_ids):
        vehicle_position_m = simulator.vehicle.getPosition(vehicle_id)
        vehicle_route = simulator.vehicle.getRoute(vehicle_id)
        if target_road_id in vehicle_route:
            distance_m = euclidean_distance(
                vehicle_position_m,
                target_coordinates_m,
            )
            station_environment_m[index] = multiplier * distance_m
            car_environment.append(vehicle_id)
        else:
            car_environment.append("-1")
            station_environment_m[index] = -100
    return station_environment_m, car_environment


def calculate_traffic_light_station(
    simulator: EntitySimulationPort,
    traffic_light_id: str,
    green_phase: int,
    ego_position_xy_m: Sequence[float],
    *,
    green_value: float = -100,
    red_value: float = -5,
) -> tuple[float, float]:
    """Return the first-junction EGO and signal constraint stations."""

    active_phase = simulator.trafficlight.getPhase(traffic_light_id)
    station_value = green_value if active_phase == green_phase else red_value
    ego_station_m = ego_position_xy_m[1] - 396
    if ego_station_m < -25 or ego_station_m > 0:
        ego_station_m = 100
    return ego_station_m, station_value


def _signal_state_code(signal_character: str) -> int:
    """Return the signal state code value owned by this zala route context module."""
    if signal_character in {"G", "g", "s"}:
        return TRAFFIC_LIGHT_STATE_GREEN
    if signal_character == "y":
        return TRAFFIC_LIGHT_STATE_YELLOW
    if signal_character == "u":
        return TRAFFIC_LIGHT_STATE_RED_YELLOW
    if signal_character == "r":
        return TRAFFIC_LIGHT_STATE_RED
    return TRAFFIC_LIGHT_STATE_UNKNOWN


def _route_signal_codes(
    state: str,
    ego_lane_id: str,
    controlled_links: Sequence[Sequence[Sequence[object]]],
) -> list[int]:
    """Return the route signal codes value owned by this zala route context module."""
    matching_indices = [
        index
        for index, link_group in enumerate(controlled_links)
        if any(link and link[0] == ego_lane_id for link in link_group)
    ]
    return [_signal_state_code(state[index]) for index in matching_indices if index < len(state)]


def _preferred_signal_code(state_codes: Sequence[int]) -> int | None:
    """Return the preferred signal code value owned by this zala route context module."""
    for preferred_code in (
        TRAFFIC_LIGHT_STATE_YELLOW,
        TRAFFIC_LIGHT_STATE_RED_YELLOW,
        TRAFFIC_LIGHT_STATE_GREEN,
        TRAFFIC_LIGHT_STATE_RED,
    ):
        if preferred_code in state_codes:
            return preferred_code
    return state_codes[0] if state_codes else None


def _first_known_signal_code(state: str) -> int:
    """Return the first known signal code value owned by this zala route context module."""
    known_codes = {
        TRAFFIC_LIGHT_STATE_YELLOW,
        TRAFFIC_LIGHT_STATE_RED_YELLOW,
        TRAFFIC_LIGHT_STATE_GREEN,
        TRAFFIC_LIGHT_STATE_RED,
    }
    return next(
        (
            mapped_code
            for signal_character in state
            if (mapped_code := _signal_state_code(signal_character)) in known_codes
        ),
        TRAFFIC_LIGHT_STATE_UNKNOWN,
    )


def read_route_traffic_light_state(
    simulator: EntitySimulationPort,
    traffic_light_id: str,
    *,
    green_phase: int | None = None,
    ego_vehicle_id: str = "EGO",
) -> SimulationRead[int]:
    """Read the route-relevant compact signal state with fallback provenance."""

    context_read = simulator.optional_read(
        "read route-relevant traffic-light context",
        lambda: (
            simulator.trafficlight.getRedYellowGreenState(traffic_light_id),
            simulator.vehicle.getLaneID(ego_vehicle_id),
            simulator.trafficlight.getControlledLinks(traffic_light_id),
        ),
        None,
    )
    if context_read.value is None:
        return _fallback_traffic_light_state(
            simulator,
            traffic_light_id,
            green_phase,
            context_read,
        )

    state, ego_lane_id, controlled_links = context_read.value
    matched_code = _preferred_signal_code(_route_signal_codes(state, ego_lane_id, controlled_links))
    if matched_code is not None:
        return SimulationRead.live(matched_code)
    if green_phase is not None:
        return _phase_traffic_light_state(
            simulator,
            traffic_light_id,
            green_phase,
        )
    return SimulationRead.live(_first_known_signal_code(state))


def _fallback_traffic_light_state(
    simulator: EntitySimulationPort,
    traffic_light_id: str,
    green_phase: int | None,
    context_read: SimulationRead[
        tuple[
            str,
            str,
            ControlledLinks,
        ]
        | None
    ],
) -> SimulationRead[int]:
    """Return the fallback traffic light state value owned by this zala route context module."""
    if green_phase is None:
        return SimulationRead(
            TRAFFIC_LIGHT_STATE_UNKNOWN,
            context_read.quality,
            context_read.error,
        )
    phase_read = _phase_traffic_light_state(
        simulator,
        traffic_light_id,
        green_phase,
    )
    return SimulationRead(
        phase_read.value,
        SimulationReadQuality.FALLBACK,
        context_read.error or phase_read.error,
    )


def _phase_traffic_light_state(
    simulator: EntitySimulationPort,
    traffic_light_id: str,
    green_phase: int,
) -> SimulationRead[int]:
    """Return the phase traffic light state value owned by this zala route context module."""
    phase_read = simulator.optional_read(
        "read traffic-light phase fallback",
        lambda: simulator.trafficlight.getPhase(traffic_light_id),
        None,
    )
    state_code = (
        TRAFFIC_LIGHT_STATE_GREEN if phase_read.value == green_phase else TRAFFIC_LIGHT_STATE_RED
    )
    return SimulationRead(state_code, phase_read.quality, phase_read.error)


@dataclass(frozen=True, slots=True)
class SignalizedIntersectionContext:
    """Signal-specific values contributed to one scenario-context update."""

    ego_station_m: float
    object_station_m: float | None
    traffic_light_state_code: int | None
    read_quality: SimulationReadQuality = SimulationReadQuality.LIVE


class SignalizedIntersectionStrategy(Protocol):
    """Supply model-owned signal observation semantics to the shared route."""

    def small_intersection(
        self,
        simulator: EntitySimulationPort,
        ego_position_xy_m: Sequence[float],
    ) -> SignalizedIntersectionContext:
        """Return the small intersection value provided by this signalized intersection
strategy."""
        ...

    def large_intersection(
        self,
        simulator: EntitySimulationPort,
        ego_position_xy_m: Sequence[float],
    ) -> SignalizedIntersectionContext:
        """Return the large intersection value provided by this signalized intersection
strategy."""
        ...


@dataclass(frozen=True, slots=True)
class PhaseConstraintSignalStrategy:
    """Represent signals as longitudinal object-station constraints."""

    def small_intersection(
        self,
        simulator: EntitySimulationPort,
        ego_position_xy_m: Sequence[float],
    ) -> SignalizedIntersectionContext:
        """Return the small intersection value owned by this phase constraint signal strategy."""
        ego_station_m, object_station_m = calculate_traffic_light_station(
            simulator,
            "18",
            2,
            ego_position_xy_m,
        )
        return SignalizedIntersectionContext(
            ego_station_m=ego_station_m,
            object_station_m=object_station_m,
            traffic_light_state_code=None,
        )

    def large_intersection(
        self,
        simulator: EntitySimulationPort,
        ego_position_xy_m: Sequence[float],
    ) -> SignalizedIntersectionContext:
        """Return the large intersection value owned by this phase constraint signal strategy."""
        active_phase = simulator.trafficlight.getPhase("31")
        ego_station_m = 310 - ego_position_xy_m[1]
        if ego_station_m < -25 or ego_station_m > 5:
            ego_station_m = 100
        return SignalizedIntersectionContext(
            ego_station_m=ego_station_m,
            object_station_m=-100 if active_phase == 11 else -1,
            traffic_light_state_code=None,
        )


@dataclass(frozen=True, slots=True)
class EncodedSignalStateStrategy:
    """Represent signals as compact route-relevant observation features."""

    def small_intersection(
        self,
        simulator: EntitySimulationPort,
        ego_position_xy_m: Sequence[float],
    ) -> SignalizedIntersectionContext:
        """Return the small intersection value owned by this encoded signal state strategy."""
        ego_station_m, _ = calculate_traffic_light_station(
            simulator,
            "18",
            2,
            ego_position_xy_m,
        )
        state_read = read_route_traffic_light_state(
            simulator,
            "18",
            green_phase=2,
        )
        return SignalizedIntersectionContext(
            ego_station_m=ego_station_m,
            object_station_m=None,
            traffic_light_state_code=state_read.value,
            read_quality=state_read.quality,
        )

    def large_intersection(
        self,
        simulator: EntitySimulationPort,
        ego_position_xy_m: Sequence[float],
    ) -> SignalizedIntersectionContext:
        """Return the large intersection value owned by this encoded signal state strategy."""
        state_read = read_route_traffic_light_state(
            simulator,
            "31",
            green_phase=11,
        )
        ego_station_m = 310 - ego_position_xy_m[1]
        if ego_station_m < -25 or ego_station_m > 5:
            ego_station_m = 100
        return SignalizedIntersectionContext(
            ego_station_m=ego_station_m,
            object_station_m=None,
            traffic_light_state_code=state_read.value,
            read_quality=state_read.quality,
        )
