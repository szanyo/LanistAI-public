# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Susan simulator observation and traffic-light helper functions."""
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

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


def ref_speed_calc(simulator: EntitySimulationPort) -> float:
    """Return the EGO reference speed in metres per second."""

    ego_x, ego_y = simulator.vehicle.getPosition("EGO")
    return zala_zone_reference_speed_mps(ego_x, ego_y)


def get_nearby_vehicles(
    simulator: EntitySimulationPort,
    *lane_ids: str,
) -> list[str]:
    """Return unique vehicle IDs observed on the requested lanes."""

    nearby_vehicle_ids: list[str] = []
    for lane_id in lane_ids:
        nearby_vehicle_ids.extend(simulator.lane.getLastStepVehicleIDs(lane_id))
    # Preserve the original set-based selection semantics used by the golden
    # controller snapshots; route strategies define the subsequent ranking.
    return list(set(nearby_vehicle_ids))


def get_nearest_pedestrian(
    simulator: EntitySimulationPort,
    ego_position_xy_m: tuple[float, float],
) -> str:
    """Return the pedestrian nearest to EGO in Euclidean map distance."""

    pedestrian_ids = simulator.person.getIDList()
    return min(
        pedestrian_ids,
        key=lambda pedestrian_id: calculate_euclidean_distance_numba(
            ego_position_xy_m,
            simulator.person.getPosition(pedestrian_id),
        ),
    )


def calculate_station_environment(
    simulator: EntitySimulationPort,
    station_environment: list[float],
    vehicle_ids: Sequence[str],
    target_coords_xy_m: tuple[float, float],
    target_road_id: str,
    multiplier: int = -1,
) -> tuple[list[float], list[str]]:
    """Project route-matching vehicles to signed target distance in metres."""

    selected_vehicle_ids: list[str] = []
    for index, vehicle_id in enumerate(vehicle_ids):
        vehicle_position_xy_m = simulator.vehicle.getPosition(vehicle_id)
        vehicle_route = simulator.vehicle.getRoute(vehicle_id)
        route_matches = target_road_id in vehicle_route

        if route_matches:
            distance_m = calculate_euclidean_distance_numba(
                vehicle_position_xy_m,
                target_coords_xy_m,
            )
            station_environment[index] = multiplier * distance_m
            selected_vehicle_ids.append(vehicle_id)
        else:
            selected_vehicle_ids.append("-1")
            station_environment[index] = -100.0

    return station_environment, selected_vehicle_ids


def handle_traffic_light(
    simulator: EntitySimulationPort,
    traffic_light_id: str | None,
    green_phase: int | None,
    ego_position_xy_m: tuple[float, float],
    green_value: float = -100.0,
    red_value: float = -5.0,
) -> tuple[float, float]:
    """Return EGO station and the legacy phase-dependent station value."""

    signal_phase = simulator.trafficlight.getPhase(cast(str, traffic_light_id))
    station_value = green_value if signal_phase == green_phase else red_value
    ego_station_m = ego_position_xy_m[1] - 396.0
    return ego_station_m, station_value


def _map_signal_char_to_state_code(signal_char: object) -> int:
    """Map the signal char to state code owned by this helper functions module."""
    if signal_char is None:
        return TRAFFIC_LIGHT_STATE_UNKNOWN

    signal_char = str(signal_char)[:1]
    if signal_char in {"G", "g"}:
        return TRAFFIC_LIGHT_STATE_GREEN
    if signal_char in {"y", "Y"}:
        return TRAFFIC_LIGHT_STATE_YELLOW
    if signal_char in {"u", "U"}:
        return TRAFFIC_LIGHT_STATE_RED_YELLOW
    if signal_char in {"r", "R", "s"}:
        return TRAFFIC_LIGHT_STATE_RED
    return TRAFFIC_LIGHT_STATE_UNKNOWN


def _select_safest_traffic_light_state_code(state_codes: Sequence[int]) -> int:
    """Select the safest traffic light state code owned by this helper functions module."""
    for preferred_code in (
        TRAFFIC_LIGHT_STATE_YELLOW,
        TRAFFIC_LIGHT_STATE_RED_YELLOW,
        TRAFFIC_LIGHT_STATE_RED,
        TRAFFIC_LIGHT_STATE_GREEN,
    ):
        if preferred_code in state_codes:
            return preferred_code

    if state_codes:
        return state_codes[0]
    return TRAFFIC_LIGHT_STATE_UNKNOWN


def _get_next_tls_state_code(
    simulator: EntitySimulationPort,
    traffic_light_id: str | None,
    ego_vehicle_id: str,
) -> int | None:
    """
    Return the compact signal state for the route-relevant nextTLS entry.

    SUMO vehicle.getNextTLS() is route-aware for the selected vehicle. If the
    same TLS appears multiple times, do not aggregate all directions/links with
    a "safest" rule; pick the closest matching route entry instead.
    """
    next_tls_entries: Sequence[Sequence[object]] = (
        simulator.vehicle.getNextTLS(ego_vehicle_id) or ()
    )
    matched_entries: list[tuple[float, object]] = []

    for next_tls_entry in next_tls_entries:
        if next_tls_entry is None or len(next_tls_entry) < 4:
            continue

        next_traffic_light_id, _, distance, signal_state = next_tls_entry[:4]
        if traffic_light_id is not None and str(next_traffic_light_id) != str(traffic_light_id):
            continue

        try:
            distance_value = float(cast(float | int | str, distance))
        except (TypeError, ValueError):
            distance_value = math.inf

        matched_entries.append((distance_value, signal_state))

    if matched_entries:
        _, signal_state = min(matched_entries, key=lambda item: item[0])
        return _map_signal_char_to_state_code(signal_state)

    return None


@dataclass(frozen=True, slots=True)
class _TrafficLightRouteContext:
    """Carry the runtime context for traffic light route."""

    state: str
    ego_lane_id: str
    controlled_links: Sequence[Sequence[Sequence[object]]]


@dataclass(frozen=True, slots=True)
class _RouteSignalMatch:
    """Carry the data required for route signal match."""

    indices: tuple[int, ...]
    state_codes: tuple[int, ...]


def _read_next_tls_state_code(
    simulator: EntitySimulationPort,
    traffic_light_id: str | None,
    ego_vehicle_id: str,
) -> SimulationRead[int | None]:
    """Read the next tls state code without changing this helper functions module's state."""
    return simulator.optional_read(
        "read Susan route-relevant next traffic light",
        lambda: _get_next_tls_state_code(
            simulator,
            traffic_light_id,
            ego_vehicle_id,
        ),
        None,
    )


def _read_traffic_light_route_context(
    simulator: EntitySimulationPort,
    traffic_light_id: str,
    ego_vehicle_id: str,
) -> SimulationRead[_TrafficLightRouteContext | None]:
    """Read the traffic light route context without changing this helper functions module's state.
    """
    return simulator.optional_read(
        "read Susan route-relevant traffic-light context",
        lambda: _TrafficLightRouteContext(
            state=simulator.trafficlight.getRedYellowGreenState(
                traffic_light_id
            ),
            ego_lane_id=simulator.vehicle.getLaneID(ego_vehicle_id),
            controlled_links=simulator.trafficlight.getControlledLinks(
                traffic_light_id
            ),
        ),
        None,
    )


def _read_phase_state_code(
    simulator: EntitySimulationPort,
    operation: str,
    traffic_light_id: str,
    green_phase: int,
) -> SimulationRead[int]:
    """Read the phase state code without changing this helper functions module's state."""
    phase_read = simulator.optional_read(
        operation,
        lambda: simulator.trafficlight.getPhase(traffic_light_id),
        None,
    )
    state_code = (
        TRAFFIC_LIGHT_STATE_GREEN
        if phase_read.value == green_phase
        else TRAFFIC_LIGHT_STATE_RED
    )
    return SimulationRead(
        state_code,
        phase_read.quality,
        phase_read.error,
    )


def _matching_route_signal_codes(
    context: _TrafficLightRouteContext,
) -> _RouteSignalMatch:
    """Return the matching route signal codes value owned by this helper functions module."""
    matching_indices = []
    for index, link_group in enumerate(context.controlled_links):
        for link in link_group:
            incoming_lane = link[0]
            if incoming_lane == context.ego_lane_id:
                matching_indices.append(index)
                break

    unique_indices = tuple(sorted(set(matching_indices)))
    return _RouteSignalMatch(
        indices=unique_indices,
        state_codes=tuple(
            _map_signal_char_to_state_code(context.state[index])
            for index in unique_indices
            if 0 <= index < len(context.state)
        ),
    )


def _context_unavailable_result(
    simulator: EntitySimulationPort,
    traffic_light_id: str,
    green_phase: int | None,
    next_tls_read: SimulationRead[int | None],
    context_read: SimulationRead[_TrafficLightRouteContext | None],
) -> SimulationRead[int]:
    """Return the context unavailable result value owned by this helper functions module."""
    if green_phase is None:
        return SimulationRead(
            TRAFFIC_LIGHT_STATE_UNKNOWN,
            SimulationReadQuality.FALLBACK,
            next_tls_read.error or context_read.error,
        )

    phase_read = _read_phase_state_code(
        simulator,
        "read Susan traffic-light phase fallback",
        traffic_light_id,
        green_phase,
    )
    return SimulationRead(
        phase_read.value,
        SimulationReadQuality.FALLBACK,
        next_tls_read.error or context_read.error or phase_read.error,
    )


def _unmatched_link_result(
    simulator: EntitySimulationPort,
    traffic_light_id: str,
    green_phase: int | None,
    next_tls_read: SimulationRead[int | None],
) -> SimulationRead[int]:
    """Return the unmatched link result value owned by this helper functions module."""
    if green_phase is None:
        return SimulationRead(
            TRAFFIC_LIGHT_STATE_UNKNOWN,
            next_tls_read.quality,
            next_tls_read.error,
        )

    phase_read = _read_phase_state_code(
        simulator,
        "read Susan traffic-light phase",
        traffic_light_id,
        green_phase,
    )
    quality = (
        SimulationReadQuality.FALLBACK
        if next_tls_read.degraded or phase_read.degraded
        else SimulationReadQuality.LIVE
    )
    return SimulationRead(
        phase_read.value,
        quality,
        next_tls_read.error or phase_read.error,
    )


def get_traffic_light_state_code(
    simulator: EntitySimulationPort,
    traffic_light_id: str,
    green_phase: int | None = None,
    ego_vehicle_id: str = "EGO",
) -> SimulationRead[int]:
    """Return the compact state together with optional-read provenance."""

    next_tls_read = _read_next_tls_state_code(
        simulator,
        traffic_light_id,
        ego_vehicle_id,
    )
    if next_tls_read.value is not None:
        return SimulationRead(
            next_tls_read.value,
            next_tls_read.quality,
            next_tls_read.error,
        )

    context_read = _read_traffic_light_route_context(
        simulator,
        traffic_light_id,
        ego_vehicle_id,
    )
    if context_read.value is None:
        return _context_unavailable_result(
            simulator,
            traffic_light_id,
            green_phase,
            next_tls_read,
            context_read,
        )

    route_match = _matching_route_signal_codes(context_read.value)
    if route_match.indices:
        return SimulationRead(
            _select_safest_traffic_light_state_code(
                route_match.state_codes
            ),
            next_tls_read.quality,
            next_tls_read.error,
        )

    return _unmatched_link_result(
        simulator,
        traffic_light_id,
        green_phase,
        next_tls_read,
    )


def calculate_euclidean_distance_numba(
    point1: Sequence[float],
    point2: Sequence[float],
) -> float:
    """Return Euclidean distance between two map-coordinate sequences."""

    return math.sqrt(
        sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2, strict=False))
    )
