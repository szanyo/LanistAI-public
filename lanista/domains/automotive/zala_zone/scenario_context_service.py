# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Model-neutral composition for the Zala route scenario context."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from lanista.domains.automotive.control.scenario_context_contracts import (
    BlinkerState,
    ScenarioContextSnapshot,
)
from lanista.domains.automotive.zala_zone.route_context import (
    ROUTE_INTERSECTION_NUMBERS,
    PhaseConstraintSignalStrategy,
    SignalizedIntersectionContext,
    SignalizedIntersectionStrategy,
    calculate_station_environment,
    euclidean_distance,
    nearby_vehicle_ids,
    nearest_pedestrian_id,
    reference_speed_mps,
)
from lanista.reinforcengine.utils.simulation.ports import (
    EntitySimulationPort,
    SimulationRead,
    SimulationReadQuality,
)

PED_STATION_DEFAULT_M = -200.0
PED_VELOCITY_DEFAULT_MPS = 0.0
PEDESTRIAN_STATE_DEFAULT = 2
OBJECT_CONTEXT_CAPACITY = 100
OBSERVED_OBJECT_COUNT = 3


class IntersectionTypeClassifier(Protocol):
    """Classify route context while retaining optional-read provenance."""

    def classify(
        self,
        simulator: EntitySimulationPort,
        ego_vehicle_id: str = "EGO",
        current_road_id: str | None = None,
    ) -> SimulationRead[int]:
        """Classify the operation owned by this intersection type classifier."""
        ...


class ScenarioContextLogger(Protocol):
    """Logging operations used by the scenario-context service."""

    def warning(self, message: str) -> None:
        """Publish a warning record through this scenario context logger."""
        ...

    def error(self, message: str) -> None:
        """Publish a error record through this scenario context logger."""
        ...


@dataclass(frozen=True, slots=True)
class ScenarioContextConfiguration:
    """Immutable policies that distinguish model-owned context features."""

    signal_strategy: SignalizedIntersectionStrategy = field(
        default_factory=PhaseConstraintSignalStrategy
    )
    intersection_classifier: IntersectionTypeClassifier | None = None
    traffic_light_default_code: int | None = None
    intersection_type_default_code: int | None = None
    route_intersections: Mapping[str, int] = field(default=ROUTE_INTERSECTION_NUMBERS)


class ScenarioContextService:
    """Read simulator entities and execute the configured route strategies."""

    def __init__(
        self,
        *,
        simulator: EntitySimulationPort,
        logger: ScenarioContextLogger,
        configuration: ScenarioContextConfiguration,
    ) -> None:
        """Initialize the scenario context service and its owned state."""
        self._simulator = simulator
        self._logger = logger
        self._configuration = configuration
        self._intersection_handlers: dict[int, Callable[[], None]] = {
            1: self._handle_intersection_1,
            2: self._handle_intersection_2,
            3: self._handle_intersection_3,
            4: self._handle_intersection_4,
            5: self._handle_intersection_5,
            6: self._handle_intersection_6,
            7: self._handle_intersection_7,
            8: self._handle_intersection_8,
            9: self._handle_intersection_9,
        }
        self._reference_speed_mps = 0.0
        self._pedestrian_station_m = 0.0
        self._pedestrian_speed_mps = 0.0
        self._pedestrian_state = 0
        self._intersection_number = 0
        self._station_environment_m: list[float] = []
        self._car_environment: list[str] = []
        self._object_stations_m = [0.0] * OBSERVED_OBJECT_COUNT
        self._object_speeds_mps = [0.0] * OBSERVED_OBJECT_COUNT
        self._ego_position_xy_m = (0.0, 0.0)
        self._ego_road_id: str | int = 0
        self._ego_station_m = 0.0
        self._blinker_left = 0
        self._blinker_right = 0
        self._blinker_encoded = 0
        self._traffic_light_state_code = configuration.traffic_light_default_code
        self._upcoming_intersection_type_code = configuration.intersection_type_default_code
        self._simulator_read_quality = SimulationReadQuality.LIVE

    def update(self, ego_speed_mps: float) -> ScenarioContextSnapshot:
        """Refresh controller inputs while preserving command/read ordering."""

        self._reset_step_context()
        self._simulator.vehicle.setSpeed("EGO", ego_speed_mps)
        handler = self._intersection_handlers.get(
            self._intersection_number,
            self._handle_intersection_error,
        )
        handler()
        self._finalize_blinkers()
        self._rank_object_context()
        self._normalize_ego_station()
        return self.snapshot

    def _reset_step_context(self) -> None:
        """Reset per-step values and load the mandatory EGO context."""

        self._simulator_read_quality = SimulationReadQuality.LIVE
        self._reference_speed_mps = reference_speed_mps(self._simulator)
        self._pedestrian_station_m = PED_STATION_DEFAULT_M
        self._pedestrian_speed_mps = PED_VELOCITY_DEFAULT_MPS
        self._pedestrian_state = PEDESTRIAN_STATE_DEFAULT
        self._traffic_light_state_code = self._configuration.traffic_light_default_code
        self._upcoming_intersection_type_code = self._configuration.intersection_type_default_code
        self._ego_position_xy_m = self._simulator.vehicle.getPosition("EGO")
        self._ego_road_id = self._simulator.vehicle.getRoadID("EGO")
        self._set_intersection_number()
        self._classify_intersection_type()
        self._station_environment_m = [PED_STATION_DEFAULT_M] * OBJECT_CONTEXT_CAPACITY
        self._car_environment = []
        self._object_speeds_mps = [0.0] * OBSERVED_OBJECT_COUNT

    def _classify_intersection_type(self) -> None:
        """Classify the intersection type owned by this scenario context service."""
        classifier = self._configuration.intersection_classifier
        if classifier is None:
            return
        classification = classifier.classify(
            self._simulator,
            current_road_id=str(self._ego_road_id),
        )
        self._upcoming_intersection_type_code = classification.value
        self._record_read_quality(classification.quality)

    def _set_intersection_number(self) -> None:
        """Set the intersection number owned by this scenario context service."""
        self._intersection_number = self._configuration.route_intersections.get(
            str(self._ego_road_id),
            -1,
        )
        if self._intersection_number == -1:
            self._logger.warning("Not defined road!!")
            self._logger.warning(f"ego_road_id: {self._ego_road_id}")

    def _finalize_blinkers(self) -> None:
        """Finalize the blinkers owned by this scenario context service."""
        if self._ego_station_m > 99 and self._intersection_number not in {1, 6}:
            self._blinker_left = 0
            self._blinker_right = 0
        self._blinker_encoded = (
            2 if self._blinker_left == 1 else 0 if self._blinker_right == 1 else 1
        )

    def _rank_object_context(self) -> None:
        """Handle the rank object context operation owned by this scenario context service."""
        ranked_stations = sorted(
            ((station_m, index + 1) for index, station_m in enumerate(self._station_environment_m)),
            key=lambda item: item[0],
            reverse=True,
        )[:OBSERVED_OBJECT_COUNT]
        self._object_stations_m = [float(station_m) for station_m, _ in ranked_stations]
        for output_index in range(min(len(self._car_environment), OBSERVED_OBJECT_COUNT)):
            source_index = ranked_stations[output_index][1] - 1
            vehicle_id = self._car_environment[source_index]
            self._object_speeds_mps[output_index] = (
                self._simulator.vehicle.getSpeed(vehicle_id) if vehicle_id != "-1" else 0.0
            )

    def _normalize_ego_station(self) -> None:
        """Normalize the ego station accepted by this scenario context service."""
        if self._ego_station_m < -25 or (
            self._ego_station_m > -6 and self._pedestrian_state == PEDESTRIAN_STATE_DEFAULT
        ):
            self._ego_station_m = 100

    def _apply_signal_context(
        self,
        signal_context: SignalizedIntersectionContext,
    ) -> None:
        """Apply the signal context transition owned by this scenario context service."""
        self._ego_station_m = signal_context.ego_station_m
        if signal_context.object_station_m is not None:
            self._station_environment_m[0] = signal_context.object_station_m
        if signal_context.traffic_light_state_code is not None:
            self._traffic_light_state_code = signal_context.traffic_light_state_code
        self._record_read_quality(signal_context.read_quality)

    def _handle_intersection_1(self) -> None:
        """Handle the intersection 1 workflow for this scenario context service."""
        self._blinker_left = 1
        self._blinker_right = 0
        self._apply_signal_context(
            self._configuration.signal_strategy.small_intersection(
                self._simulator,
                self._ego_position_xy_m,
            )
        )

    def _handle_intersection_2(self) -> None:
        """Handle the intersection 2 workflow for this scenario context service."""
        self._blinker_left = 1 if self._ego_position_xy_m[0] > 148 else 0
        self._blinker_right = 0
        vehicle_ids = nearby_vehicle_ids(
            self._simulator,
            ":22_0_0",
            ":22_2_0",
            ":22_4_0",
            "-4115.0.00_0",
            ":23_2_0",
        )
        (
            self._station_environment_m,
            self._car_environment,
        ) = calculate_station_environment(
            self._simulator,
            self._station_environment_m,
            vehicle_ids,
            (81, 396),
            "-4458.0.00",
            multiplier=-1,
        )
        self._ego_station_m = 78 - self._ego_position_xy_m[0]

    def _handle_intersection_3(self) -> None:
        """Handle the intersection 3 workflow for this scenario context service."""
        self._blinker_left = 1 if self._ego_position_xy_m[1] < 372 else 0
        self._blinker_right = 1 if 377 < self._ego_position_xy_m[1] < 395 else 0
        vehicle_ids = nearby_vehicle_ids(
            self._simulator,
            "-6245.2.00_0",
            ":6245.2.00_1_0",
            "-6245.2.00_0",
            ":6245.17.00_1_0",
            "-6245.17.00_0",
            ":43_3_0",
        )
        for index, vehicle_id in enumerate(vehicle_ids):
            vehicle_position_m = self._simulator.vehicle.getPosition(vehicle_id)
            self._station_environment_m[index] = vehicle_position_m[1] - 340
            self._car_environment.append(vehicle_id)
        self._ego_station_m = 356 - self._ego_position_xy_m[1]
        if self._ego_station_m > 5:
            self._ego_station_m = 100

    def _handle_intersection_4(self) -> None:
        """Handle the intersection 4 workflow for this scenario context service."""
        self._blinker_left = 0
        self._blinker_right = 0
        vehicle_ids = nearby_vehicle_ids(self._simulator, "-5018.1.00_1")
        for index, vehicle_id in enumerate(vehicle_ids):
            vehicle_position_m = self._simulator.vehicle.getPosition(vehicle_id)
            self._station_environment_m[index] = vehicle_position_m[1] - 350
            self._car_environment.append(vehicle_id)
        pedestrian_read = self._read_pedestrian_context()
        self._record_read_quality(pedestrian_read.quality)
        if pedestrian_read.value is not None:
            self._apply_pedestrian_context(*pedestrian_read.value)
            return
        if pedestrian_read.degraded:
            self._logger.warning("Pedestrian context unavailable; using the declared fallback.")
        self._pedestrian_station_m = -100
        self._pedestrian_speed_mps = 0
        self._pedestrian_state = 1
        self._ego_station_m = self._ego_position_xy_m[0] - 116

    def _read_pedestrian_context(
        self,
    ) -> SimulationRead[tuple[tuple[float, float], float] | None]:
        """Read the pedestrian context without changing this scenario context service's state."""
        def read_context() -> tuple[tuple[float, float], float] | None:
            """Read the context without changing this scenario context service's state."""
            if not self._simulator.person.getIDList():
                return None
            pedestrian_id = nearest_pedestrian_id(
                self._simulator,
                self._ego_position_xy_m,
            )
            return (
                self._simulator.person.getPosition(pedestrian_id),
                self._simulator.person.getSpeed(pedestrian_id),
            )

        return self._simulator.optional_read(
            "read pedestrian crossing context",
            read_context,
            None,
        )

    def _apply_pedestrian_context(
        self,
        pedestrian_position_xy_m: tuple[float, float],
        pedestrian_speed_mps: float,
    ) -> None:
        """Apply the pedestrian context transition owned by this scenario context service."""
        pedestrian_y_m = pedestrian_position_xy_m[1]
        self._pedestrian_station_m = (
            pedestrian_y_m - 344 if pedestrian_y_m < 344 else 0 if pedestrian_y_m < 353 else -100
        )
        self._pedestrian_speed_mps = pedestrian_speed_mps
        self._pedestrian_state = (
            1 if self._pedestrian_station_m < -3 or self._pedestrian_station_m > 0 else 0
        )
        self._ego_station_m = self._ego_position_xy_m[0] - 104

    def _handle_intersection_5(self) -> None:
        """Handle the intersection 5 workflow for this scenario context service."""
        self._blinker_left = 0
        self._blinker_right = 1 if self._ego_position_xy_m[0] > 133 else 0
        vehicle_ids = nearby_vehicle_ids(
            self._simulator,
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
        )
        for index, vehicle_id in enumerate(vehicle_ids):
            vehicle_position_m = self._simulator.vehicle.getPosition(vehicle_id)
            self._station_environment_m[index] = 345 - vehicle_position_m[1]
            self._car_environment.append(vehicle_id)
        self._ego_station_m = self._ego_position_xy_m[0] - 168

    def _handle_intersection_6(self) -> None:
        """Handle the intersection 6 workflow for this scenario context service."""
        self._blinker_left = 1
        self._blinker_right = 0
        self._apply_signal_context(
            self._configuration.signal_strategy.large_intersection(
                self._simulator,
                self._ego_position_xy_m,
            )
        )
        if self._ego_road_id == ":31_4":
            return
        leading_vehicle_ids = list(self._simulator.lane.getLastStepVehicleIDs("-3593.22.00_2"))
        if leading_vehicle_ids and leading_vehicle_ids[0] == "EGO":
            del leading_vehicle_ids[0]
        for index, vehicle_id in enumerate(leading_vehicle_ids):
            vehicle_position_m = self._simulator.vehicle.getPosition(vehicle_id)
            self._station_environment_m[index] = -1 - index * 10
            distance_m = euclidean_distance(
                vehicle_position_m,
                self._ego_position_xy_m,
            )
            self._ego_station_m = -1.2 * distance_m
            self._car_environment.append(vehicle_id)

    def _handle_intersection_7(self) -> None:
        """Handle the intersection 7 workflow for this scenario context service."""
        self._blinker_left = 0
        self._blinker_right = (
            1
            if self._ego_position_xy_m[1] < 224
            or (self._ego_position_xy_m[1] > 288 and self._ego_position_xy_m[0] > 218)
            else 0
        )
        vehicle_ids = nearby_vehicle_ids(
            self._simulator,
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
        )
        self._record_route_vehicles(
            vehicle_ids,
            target_route_edge="7690.27.00",
            station_calculator=lambda position: 235 - position[0],
        )
        self._ego_station_m = 206 - self._ego_position_xy_m[1]

    def _handle_intersection_8(self) -> None:
        """Handle the intersection 8 workflow for this scenario context service."""
        self._blinker_left = 0
        self._blinker_right = 1 if self._ego_position_xy_m[0] < 200 else 0
        vehicle_ids = nearby_vehicle_ids(
            self._simulator,
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
        )
        self._record_route_vehicles(
            vehicle_ids,
            target_route_edge="6321.63.00",
            station_calculator=lambda position: position[1] - 197,
        )
        self._ego_station_m = 175 - self._ego_position_xy_m[0]

    def _record_route_vehicles(
        self,
        vehicle_ids: Sequence[str],
        *,
        target_route_edge: str,
        station_calculator: Callable[[tuple[float, float]], float],
    ) -> None:
        """Record the route vehicles owned by this scenario context service."""
        for index, vehicle_id in enumerate(vehicle_ids):
            vehicle_position_m = self._simulator.vehicle.getPosition(vehicle_id)
            vehicle_route = self._simulator.vehicle.getRoute(vehicle_id)
            if target_route_edge in vehicle_route:
                self._station_environment_m[index] = station_calculator(vehicle_position_m)
                self._car_environment.append(vehicle_id)
            else:
                self._car_environment.append("-1")
                self._station_environment_m[index] = -100

    def _handle_intersection_9(self) -> None:
        """Handle the intersection 9 workflow for this scenario context service."""
        self._blinker_left = 0
        self._blinker_right = 0
        self._station_environment_m = [-100] * OBJECT_CONTEXT_CAPACITY
        self._ego_station_m = -100

    def _handle_intersection_error(self) -> None:
        """Handle the intersection error workflow for this scenario context service."""
        self._blinker_left = 0
        self._blinker_right = 0
        self._station_environment_m = [-5] * OBJECT_CONTEXT_CAPACITY
        self._ego_station_m = -1
        self._logger.error("WRONG INTERSECTION")
        diagnostic = self._simulator.optional_read(
            "read unknown-intersection diagnostics",
            lambda: (
                self._simulator.vehicle.getRoadID("EGO"),
                self._simulator.vehicle.getLaneID("EGO"),
            ),
            None,
        )
        self._record_read_quality(diagnostic.quality)
        if diagnostic.value is None:
            self._logger.warning(
                "Unknown-intersection diagnostics unavailable; using the declared fallback."
            )
            return
        road_id, lane_id = diagnostic.value
        self._logger.error(f"EGO road: {road_id}")
        self._logger.error(f"EGO lane: {lane_id}")

    def _record_read_quality(self, quality: SimulationReadQuality) -> None:
        """Record the read quality owned by this scenario context service."""
        if quality is SimulationReadQuality.FALLBACK:
            self._simulator_read_quality = quality

    @property
    def snapshot(self) -> ScenarioContextSnapshot:
        """Return an immutable projection of the current context."""

        return ScenarioContextSnapshot(
            reference_speed_mps=float(self._reference_speed_mps),
            object_stations_m=_three_values(self._object_stations_m),
            object_speeds_mps=_three_values(self._object_speeds_mps),
            ego_station_m=float(self._ego_station_m),
            pedestrian_parameters=(
                float(self._pedestrian_station_m),
                float(self._pedestrian_speed_mps),
                int(self._pedestrian_state),
            ),
            intersection_number=int(self._intersection_number),
            ego_position_xy_m=(
                float(self._ego_position_xy_m[0]),
                float(self._ego_position_xy_m[1]),
            ),
            ego_road_id=self._ego_road_id,
            blinkers=BlinkerState(
                left=int(self._blinker_left),
                right=int(self._blinker_right),
                encoded=int(self._blinker_encoded),
            ),
            traffic_light_state_code=self._traffic_light_state_code,
            upcoming_intersection_type_code=(self._upcoming_intersection_type_code),
            simulator_read_quality=self._simulator_read_quality,
        )


def _three_values(values: Sequence[float]) -> tuple[float, float, float]:
    """Return the three values value owned by this scenario context service module."""
    if len(values) != OBSERVED_OBJECT_COUNT:
        raise ValueError("scenario object context must contain exactly three values")
    return float(values[0]), float(values[1]), float(values[2])
