# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Route-neutral context contracts consumed by automotive controllers."""

from dataclasses import dataclass
from typing import Protocol

from lanista.reinforcengine.utils.simulation.ports import SimulationReadQuality


@dataclass(frozen=True, slots=True)
class BlinkerState:
    """Left, right, and encoded blinker values for one simulator step."""

    left: int
    right: int
    encoded: int


@dataclass(frozen=True, slots=True)
class ScenarioContextSnapshot:
    """Immutable, route-neutral controller inputs from one simulator step."""

    reference_speed_mps: float
    object_stations_m: tuple[float, float, float]
    object_speeds_mps: tuple[float, float, float]
    ego_station_m: float
    pedestrian_parameters: tuple[float, float, int]
    intersection_number: int
    ego_position_xy_m: tuple[float, float]
    ego_road_id: str | int
    blinkers: BlinkerState
    traffic_light_state_code: int | None
    upcoming_intersection_type_code: int | None
    simulator_read_quality: SimulationReadQuality


class ScenarioContextPreprocessorPort(Protocol):
    """Produce route-neutral context for one automotive control step."""

    def update(self, ego_speed_mps: float) -> ScenarioContextSnapshot:
        """Refresh and return the immutable scenario context."""
        ...
