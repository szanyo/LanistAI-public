# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Immutable geometry and sampling descriptors for Susan's intersections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from types import MappingProxyType
from typing import Final


class CoordinateAxis(StrEnum):
    """Coordinate axes used by SUMO positions measured in metres."""

    X = "x"
    Y = "y"


class ComparisonOperator(StrEnum):
    """Supported comparisons for declarative blinker conditions."""

    LESS_THAN = "less_than"
    GREATER_THAN = "greater_than"


class PedestrianCrossingState(IntEnum):
    """Legacy three-state pedestrian signal with explicit semantics."""

    ON_CROSSING = 0
    CLEAR = 1
    NOT_APPLICABLE = 2


@dataclass(frozen=True, slots=True)
class AxisCondition:
    """One coordinate comparison; ``threshold_m`` is expressed in metres."""

    axis: CoordinateAxis
    operator: ComparisonOperator
    threshold_m: float


@dataclass(frozen=True, slots=True)
class ConditionGroup:
    """Conditions that must all match for one alternative to be true."""

    all_of: tuple[AxisCondition, ...]


@dataclass(frozen=True, slots=True)
class BlinkerRule:
    """A fixed or disjunctive blinker rule over metre-based coordinates."""

    always_on: bool = False
    any_of: tuple[ConditionGroup, ...] = ()


@dataclass(frozen=True, slots=True)
class StationProjection:
    """Map one SUMO coordinate in metres to a route station in metres."""

    axis: CoordinateAxis
    station_origin_m: float
    coordinate_scale: float


@dataclass(frozen=True, slots=True)
class ConstantStation:
    """A route station that is independent of the current position."""

    station_m: float


@dataclass(frozen=True, slots=True)
class TrafficLightStopStation:
    """Use the legacy traffic-light stop-line geometry helper."""


type EgoStationStrategy = StationProjection | ConstantStation | TrafficLightStopStation


@dataclass(frozen=True, slots=True)
class NoObjectSampling:
    """Keep the preprocessor's default object slots unchanged."""


@dataclass(frozen=True, slots=True)
class ConstantObjectSampling:
    """Fill every object slot with one fixed station in metres."""

    station_m: float


@dataclass(frozen=True, slots=True)
class CalculatedObjectSampling:
    """Delegate legacy road-aware projection to the shared geometry helper."""

    lane_ids: tuple[str, ...]
    target_xy_m: tuple[float, float]
    target_road_id: str
    station_multiplier: int


@dataclass(frozen=True, slots=True)
class PositionObjectSampling:
    """Project nearby vehicle positions onto one route-station axis."""

    lane_ids: tuple[str, ...]
    station_projection: StationProjection
    required_route_edge_id: str | None = None
    excluded_vehicle_station_m: float = -100.0
    excluded_vehicle_id: str = "-1"


@dataclass(frozen=True, slots=True)
class LaneQueueObjectSampling:
    """Represent one ordered lane queue with fixed metre spacing."""

    lane_id: str
    disabled_on_road_id: str
    first_vehicle_station_m: float
    vehicle_spacing_m: float
    supervisor_ego_distance_scale: float | None = None
    use_ego_lane: bool = False
    ego_vehicle_id: str = "EGO"


type ObjectSamplingStrategy = (
    NoObjectSampling
    | ConstantObjectSampling
    | CalculatedObjectSampling
    | PositionObjectSampling
    | LaneQueueObjectSampling
)


@dataclass(frozen=True, slots=True)
class PedestrianCrossingDescriptor:
    """Pedestrian-zone thresholds and fallback geometry, all in SI units."""

    entry_y_m: float
    exit_y_m: float
    on_crossing_lower_station_m: float
    beyond_crossing_station_m: float
    unavailable_station_m: float
    unavailable_speed_mps: float
    ego_station_with_pedestrian: StationProjection
    ego_station_without_pedestrian: StationProjection


@dataclass(frozen=True, slots=True)
class IntersectionDescriptor:
    """Complete immutable strategy selection for one route intersection."""

    number: int
    name: str
    left_blinker: BlinkerRule
    right_blinker: BlinkerRule
    ego_station: EgoStationStrategy
    object_sampling: ObjectSamplingStrategy
    pedestrian_crossing: PedestrianCrossingDescriptor | None = None
    uses_configured_traffic_light: bool = False


OFF: Final = BlinkerRule()
ON: Final = BlinkerRule(always_on=True)


def _condition(
    axis: CoordinateAxis,
    operator: ComparisonOperator,
    threshold_m: float,
) -> ConditionGroup:
    """Return the condition value owned by this intersection strategy descriptors module."""
    return ConditionGroup((AxisCondition(axis, operator, threshold_m),))


def _all_conditions(*conditions: AxisCondition) -> ConditionGroup:
    """Return the all conditions value owned by this intersection strategy descriptors module."""
    return ConditionGroup(tuple(conditions))


INTERSECTION_DESCRIPTORS: Mapping[int, IntersectionDescriptor] = MappingProxyType(
    {
        1: IntersectionDescriptor(
            number=1,
            name="small_traffic_light",
            left_blinker=ON,
            right_blinker=OFF,
            ego_station=TrafficLightStopStation(),
            object_sampling=NoObjectSampling(),
            uses_configured_traffic_light=True,
        ),
        2: IntersectionDescriptor(
            number=2,
            name="roundabout",
            left_blinker=BlinkerRule(
                any_of=(
                    _condition(
                        CoordinateAxis.X,
                        ComparisonOperator.GREATER_THAN,
                        148.0,
                    ),
                )
            ),
            right_blinker=OFF,
            ego_station=StationProjection(CoordinateAxis.X, 78.0, -1.0),
            object_sampling=CalculatedObjectSampling(
                lane_ids=(":22_0_0", ":22_2_0", ":22_4_0", "-4115.0.00_0", ":23_2_0"),
                target_xy_m=(81.0, 396.0),
                target_road_id="-4458.0.00",
                station_multiplier=-1,
            ),
        ),
        3: IntersectionDescriptor(
            number=3,
            name="left_t_junction",
            left_blinker=BlinkerRule(
                any_of=(
                    _condition(CoordinateAxis.Y, ComparisonOperator.LESS_THAN, 372.0),
                )
            ),
            right_blinker=BlinkerRule(
                any_of=(
                    _all_conditions(
                        AxisCondition(
                            CoordinateAxis.Y,
                            ComparisonOperator.GREATER_THAN,
                            377.0,
                        ),
                        AxisCondition(
                            CoordinateAxis.Y,
                            ComparisonOperator.LESS_THAN,
                            395.0,
                        ),
                    ),
                )
            ),
            ego_station=StationProjection(CoordinateAxis.Y, 356.0, -1.0),
            object_sampling=PositionObjectSampling(
                lane_ids=(
                    "-6245.0.00_0",
                    ":6245.2.00_1_0",
                    "-6245.2.00_0",
                    ":6245.17.00_1_0",
                    "-6245.17.00_0",
                    ":43_3_0",
                ),
                station_projection=StationProjection(CoordinateAxis.Y, -340.0, 1.0),
            ),
        ),
        4: IntersectionDescriptor(
            number=4,
            name="pedestrian_crossing",
            left_blinker=OFF,
            right_blinker=OFF,
            ego_station=StationProjection(CoordinateAxis.X, -116.0, 1.0),
            object_sampling=PositionObjectSampling(
                lane_ids=("-5018.1.00_1",),
                station_projection=StationProjection(CoordinateAxis.Y, -350.0, 1.0),
            ),
            pedestrian_crossing=PedestrianCrossingDescriptor(
                entry_y_m=344.0,
                exit_y_m=353.0,
                on_crossing_lower_station_m=-3.0,
                beyond_crossing_station_m=-100.0,
                unavailable_station_m=-100.0,
                unavailable_speed_mps=0.0,
                ego_station_with_pedestrian=StationProjection(
                    CoordinateAxis.X,
                    -104.0,
                    1.0,
                ),
                ego_station_without_pedestrian=StationProjection(
                    CoordinateAxis.X,
                    -116.0,
                    1.0,
                ),
            ),
        ),
        5: IntersectionDescriptor(
            number=5,
            name="middle_priority_intersection",
            left_blinker=OFF,
            right_blinker=BlinkerRule(
                any_of=(
                    _condition(
                        CoordinateAxis.X,
                        ComparisonOperator.GREATER_THAN,
                        133.0,
                    ),
                )
            ),
            ego_station=StationProjection(CoordinateAxis.X, -168.0, 1.0),
            object_sampling=PositionObjectSampling(
                lane_ids=(
                    "-3526.23.00_0",
                    "-3526.23.00_1",
                    "-3526.23.00_2",
                    ":3526.23.00_0_1",
                    ":3526.23.00_0_2",
                    "-3526.23.00_1",
                    "-3526.23.00_2",
                    "-3526.14.64_0",
                    ":3526.14.64_0_1",
                    ":3526.14.64_0_2",
                    "-3526.14.64_1",
                    "-3526.14.64_2",
                    "-3526.6.00_0",
                    ":3526.6.00_0_0",
                    "-3526.6.00_1",
                    "-3526.6.00_2",
                    "-3526.0.00_0",
                    "-3526.0.00_1",
                    ":19_0_1",
                    ":19_0_2",
                ),
                station_projection=StationProjection(CoordinateAxis.Y, 345.0, -1.0),
            ),
        ),
        6: IntersectionDescriptor(
            number=6,
            name="large_traffic_light",
            left_blinker=ON,
            right_blinker=OFF,
            ego_station=StationProjection(CoordinateAxis.Y, 310.0, -1.0),
            object_sampling=LaneQueueObjectSampling(
                lane_id="-3593.22.00_2",
                disabled_on_road_id=":31_4",
                first_vehicle_station_m=-1.0,
                vehicle_spacing_m=-10.0,
                supervisor_ego_distance_scale=1.2,
                use_ego_lane=True,
            ),
            uses_configured_traffic_light=True,
        ),
        7: IntersectionDescriptor(
            number=7,
            name="lower_right_priority_intersection",
            left_blinker=OFF,
            right_blinker=BlinkerRule(
                any_of=(
                    _condition(CoordinateAxis.Y, ComparisonOperator.LESS_THAN, 224.0),
                    _all_conditions(
                        AxisCondition(
                            CoordinateAxis.Y,
                            ComparisonOperator.GREATER_THAN,
                            288.0,
                        ),
                        AxisCondition(
                            CoordinateAxis.X,
                            ComparisonOperator.GREATER_THAN,
                            218.0,
                        ),
                    ),
                )
            ),
            ego_station=StationProjection(CoordinateAxis.Y, 206.0, -1.0),
            object_sampling=PositionObjectSampling(
                lane_ids=(
                    "7818.0.00_0",
                    ":7818.7.60_0_0",
                    "7818.0.70_0",
                    "7818.0.70_1",
                    ":7818.7.60_0_0",
                    "7818.7.60_0",
                    ":34_0_0",
                    ":7818.0.70_0_0",
                    ":35_1_0",
                    ":34_3_0",
                    ":34_6_0",
                ),
                station_projection=StationProjection(CoordinateAxis.X, 235.0, -1.0),
                required_route_edge_id="7690.27.00",
            ),
        ),
        8: IntersectionDescriptor(
            number=8,
            name="lower_left_priority_intersection",
            left_blinker=OFF,
            right_blinker=BlinkerRule(
                any_of=(
                    _condition(CoordinateAxis.X, ComparisonOperator.LESS_THAN, 200.0),
                )
            ),
            ego_station=StationProjection(CoordinateAxis.X, 175.0, -1.0),
            object_sampling=PositionObjectSampling(
                lane_ids=(
                    ":2_7_0",
                    "7692.0.00_0",
                    "7692.0.00_1",
                    "7692.0.00_2",
                    ":7692.11.00_5_0",
                    "7692.11.00_0",
                    "7692.2.00_0",
                    "7692.2.00_1",
                    "7692.2.00_2",
                    "7692.11.00_0",
                    "7692.11.00_1",
                    "7692.11.00_2",
                ),
                station_projection=StationProjection(CoordinateAxis.Y, -197.0, 1.0),
                required_route_edge_id="6321.63.00",
            ),
        ),
        9: IntersectionDescriptor(
            number=9,
            name="route_terminal",
            left_blinker=OFF,
            right_blinker=OFF,
            ego_station=ConstantStation(-100.0),
            object_sampling=ConstantObjectSampling(-100.0),
        ),
    }
)
