# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed contracts shared by Susan's simulation pipeline components."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast

from lanista.reinforcengine.utils.simulation.intersection_metadata import (
    TRAFFIC_LIGHT_STATE_NOT_APPLICABLE,
)
from lanista.reinforcengine.utils.simulation.ports import SimulationReadQuality
from lanista.rl_model_storage.models.susan_project.resources.utils.observation.vehicle_telemetry import (
    VehicleControllerTelemetryState,
    VehicleTelemetry,
)
from lanista.rl_model_storage.models.susan_project.resources.utils.scenario import (
    intersection_strategy_descriptors,
)

type PedestrianParameters = tuple[
    float,
    float,
    intersection_strategy_descriptors.PedestrianCrossingState,
]


def has_valid_ego_lead_telemetry(
    lead_vehicle_present: bool,
    lead_distance_m: object,
) -> bool:
    """Return whether SUMO supplied usable lead-vehicle telemetry."""

    try:
        distance_m = float(cast(float | int | str, lead_distance_m))
    except (TypeError, ValueError):
        return False
    return (
        bool(lead_vehicle_present)
        and math.isfinite(distance_m)
        and distance_m >= 0.0
    )


def is_ego_blocked_by_pedestrian(
    pedestrian_parameters: PedestrianParameters | None,
) -> bool:
    """Return whether a live pedestrian still occupies the crossing."""

    if pedestrian_parameters is None:
        return False
    try:
        pedestrian_station_m, _pedestrian_speed_mps, crossing_state = (
            pedestrian_parameters
        )
        station_m = float(pedestrian_station_m)
        state = int(crossing_state)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(station_m)
        and state
        == int(intersection_strategy_descriptors.PedestrianCrossingState.ON_CROSSING)
    )


@dataclass(frozen=True)
class TrafficLightContext:
    """Canonical traffic-light identity and state for the active route."""

    tls_id: str | None = None
    green_phase: int | None = None
    link_indices: tuple[int, ...] = ()
    compact_state: int = TRAFFIC_LIGHT_STATE_NOT_APPLICABLE
    intersection_number: int = 0


@dataclass(frozen=True)
class ScenarioContext:
    """Immutable simulator inputs consumed by one controller pipeline step.

    Distances are expressed in metres, speeds in metres per second, and the
    pedestrian tuple contains ``(station_m, speed_mps, crossing_state)``, where
    ``crossing_state`` is
    :class:`lanista.rl_model_storage.models.susan_project.resources.utils.scenario.intersection_strategy_descriptors.PedestrianCrossingState`.
    """

    reference_speed_mps: float
    object_stations_m: tuple[float, ...]
    object_speeds_mps: tuple[float, ...]
    ego_station_m: float
    raw_ego_station_m: float
    target_ego_station_m: float
    supervisor_ego_station_m: float
    ego_lead_vehicle_present: bool
    ego_lead_distance_m: float
    pedestrian_parameters: PedestrianParameters
    intersection_number: int
    traffic_light_state_code: int
    upcoming_intersection_type_code: int
    traffic_light_context: TrafficLightContext
    ego_position_xy_m: tuple[float, float]
    ego_road_id: str | int
    simulator_read_quality: SimulationReadQuality = SimulationReadQuality.LIVE
    ego_lead_speed_mps: float = 0.0
    ego_lead_vehicle_id: str | None = None
    pedestrian_id: str | None = None
    traffic_light_route_internal_edge: str | None = None
    policy_object_stations_m: tuple[float, ...] = ()
    policy_object_speeds_mps: tuple[float, ...] = ()
    policy_object_ids: tuple[str, ...] = ()
    policy_object_kinds: tuple[str, ...] = ()
    policy_object_geometries: tuple[str, ...] = ()
    policy_object_distances_m: tuple[float, ...] = ()
    policy_object_constraint_clearances_m: tuple[float, ...] = ()
    supervisor_vehicle_ids: tuple[str, ...] = ()
    pedestrian_count: int = 0
    active_pedestrian_count: int = 0
    supervisor_vehicle_conflict: bool = False

    @property
    def effective_policy_object_stations_m(self) -> tuple[float, ...]:
        """Return the policy projection, falling back for legacy callers."""

        return self.policy_object_stations_m or self.object_stations_m

    @property
    def effective_policy_object_speeds_mps(self) -> tuple[float, ...]:
        """Return policy speeds aligned with the effective object stations."""

        return self.policy_object_speeds_mps or self.object_speeds_mps

    @property
    def road_speed_limit_mps(self) -> float:
        """Return the road limit used by the existing Susan controller."""

        return self.reference_speed_mps


class ScenarioContextPreprocessorPort(Protocol):
    """Builds the scenario snapshot consumed by the control pipeline."""

    @property
    def current_context(self) -> ScenarioContext:
        """Return the current context exposed by this scenario context preprocessor port."""
        ...

    def update(self, ego_speed_mps: float) -> ScenarioContext:
        """Return the update value provided by this scenario context preprocessor port."""
        ...


class ScalarActionPort(Protocol):
    """Define the typed boundary for scalar action."""

    def item(self) -> float:
        """Return the item value provided by this scalar action port."""
        ...


class EgoVehiclePort(Protocol):
    """Define the typed boundary for ego vehicle."""

    prev_speed: float

    def calculate(
        self,
        delta: float,
        reference_speed_mps: float,
    ) -> tuple[float, float, float, float]:
        """Return the calculate value provided by this ego vehicle port."""
        ...


class SumoCommandPort(Protocol):
    """Define the typed boundary for sumo command."""

    def read_acceleration_since_last_controller_step(
        self,
    ) -> tuple[float | None, float | None]:
        """Read the acceleration since last controller step without changing this sumo command
        port's state.
        """
        ...

    def remember_step_speed(self, current_speed_mps: float | None) -> None:
        """Remember the step speed owned by this sumo command port."""
        ...

    def set_ego_speed(self, speed_mps: float) -> None:
        """Set the ego speed owned by this sumo command port."""
        ...


class TelemetrySinkPort(Protocol):
    """Define the typed boundary for telemetry sink."""

    def append(self, telemetry: VehicleTelemetry) -> None:
        """Append the operation owned by this telemetry sink port."""
        ...


class SimulationMathPort(Protocol):
    """Math operations consumed directly by the simulation pipeline."""

    def _get_distance_to_stopline(
        self,
        ego_station_m: float | None,
    ) -> float | None:
        """Return the distance to stopline exposed by this simulation math port."""
        ...

    def _calculate_actual_acceleration(
        self,
        previous_speed_mps: float,
        current_speed_mps: float,
        actual_acceleration_override: float | None = None,
    ) -> float:
        """Calculate the actual acceleration owned by this simulation math port."""
        ...

    def _observation_road_speed_limit(
        self,
        road_speed_limit_mps: float | None,
        distance_to_stopline_m: float | None,
        traffic_light_state_code: int,
        upcoming_intersection_type_code: int | None = None,
        red_light_overdrive: bool = False,
    ) -> float:
        """Return the observation road speed limit value provided by this simulation math
port."""
        ...

class SimulationPipelineControllerPort(Protocol):
    """Controller operations required by ``SimulationPipelineService``."""

    _speed_out: float
    _delta: float
    _prev_upcoming_intersection_type_code: int | None
    _prev_traffic_light_state_code: int | None
    cumulative_reward_counter_over_episodes: float
    policy_action: float
    red_light_overdrive_this_step: bool
    controller_state: VehicleControllerTelemetryState
    ego_vehicle: EgoVehiclePort

    @property
    def current_scenario_context(self) -> ScenarioContext:
        """Return the current scenario context exposed by this simulation pipeline controller port.
        """
        ...

    def update_scenario_context(self, ego_speed_mps: float) -> ScenarioContext:
        """Update the scenario context owned by this simulation pipeline controller port."""
        ...

    def record_interaction_step(self, scenario: ScenarioContext) -> None:
        """Record the actor selected by the existing speed reward profile."""

        ...

    def _get_sumo_command_service(self) -> SumoCommandPort:
        """Return the sumo command service exposed by this simulation pipeline controller port."""
        ...

    def is_running_into_restricted_area(
        self,
        previous_context: ScenarioContext | None = None,
        current_context: ScenarioContext | None = None,
    ) -> bool:
        """Return whether running into restricted area applies to this simulation pipeline
        controller port.
        """
        ...

    def crossed_configured_signal_boundary(
        self,
        previous_context: ScenarioContext | None,
        current_context: ScenarioContext | None,
    ) -> bool:
        """Return whether the calibrated boundary of one signal was crossed."""

        ...


class TrafficLightControllerPort(Protocol):
    """Scenario-context boundary required by traffic-light timing."""

    @property
    def current_scenario_context(self) -> ScenarioContext:
        """Return the current scenario context exposed by this traffic light controller port."""
        ...
