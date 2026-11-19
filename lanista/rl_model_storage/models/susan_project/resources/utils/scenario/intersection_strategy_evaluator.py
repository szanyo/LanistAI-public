# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Stateless evaluator for Susan's declarative intersection descriptors."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, assert_never

from lanista.reinforcengine.utils.simulation.intersection_metadata import (
    TRAFFIC_LIGHT_STATE_NOT_APPLICABLE,
    TRAFFIC_LIGHT_STATE_UNKNOWN,
)
from lanista.reinforcengine.utils.simulation.ports import (
    EntitySimulationPort,
    SimulationOperationError,
    SimulationReadQuality,
)

from .intersection_strategy_descriptors import (
    AxisCondition,
    BlinkerRule,
    CalculatedObjectSampling,
    ComparisonOperator,
    ConstantObjectSampling,
    ConstantStation,
    CoordinateAxis,
    EgoStationStrategy,
    IntersectionDescriptor,
    LaneQueueObjectSampling,
    NoObjectSampling,
    ObjectSamplingStrategy,
    PedestrianCrossingDescriptor,
    PedestrianCrossingState,
    PositionObjectSampling,
    StationProjection,
    TrafficLightStopStation,
)


class TrafficLightRouteContextPort(Protocol):
    """Configured route-specific signal metadata."""

    tls_id: str
    green_phase: int
    link_index: int
    route_internal_edge: str


@dataclass(frozen=True, slots=True)
class IntersectionEvaluationInput:
    """Immutable simulator snapshot used by one intersection evaluation."""

    ego_position_xy_m: tuple[float, float]
    ego_road_id: str | int
    initial_object_stations_m: tuple[float, ...]
    ego_lane_id: str | None = None


@dataclass(frozen=True, slots=True)
class ConfiguredTrafficLightEvaluation:
    """Route-relevant signal identity selected by controller configuration."""

    tls_id: str
    green_phase: int
    link_indices: tuple[int, ...]
    route_internal_edge: str


@dataclass(frozen=True, slots=True)
class IntersectionEvaluation:
    """Complete immutable result applied by ``ScenarioContextPreprocessor``."""

    left_blinker: int
    right_blinker: int
    ego_station_m: float
    supervisor_ego_station_m: float
    object_stations_m: tuple[float, ...]
    object_vehicle_ids: tuple[str, ...]
    pedestrian_station_m: float
    pedestrian_speed_mps: float
    pedestrian_state: PedestrianCrossingState
    pedestrians: tuple[PedestrianObstacle, ...]
    traffic_light_state_code: int
    configured_traffic_light: ConfiguredTrafficLightEvaluation | None
    simulator_read_quality: SimulationReadQuality
    pedestrian_id: str | None = None

    @classmethod
    def unknown(cls, object_slot_count: int) -> IntersectionEvaluation:
        """Return the existing fail-safe values for an unmapped road."""

        return cls(
            left_blinker=0,
            right_blinker=0,
            ego_station_m=-1.0,
            supervisor_ego_station_m=-1.0,
            object_stations_m=(-5.0,) * object_slot_count,
            object_vehicle_ids=(),
            pedestrian_station_m=-200.0,
            pedestrian_speed_mps=0.0,
            pedestrian_state=PedestrianCrossingState.NOT_APPLICABLE,
            pedestrians=(),
            traffic_light_state_code=TRAFFIC_LIGHT_STATE_NOT_APPLICABLE,
            configured_traffic_light=None,
            simulator_read_quality=SimulationReadQuality.LIVE,
        )


@dataclass(frozen=True, slots=True)
class IntersectionEvaluatorDependencies:
    """Injected helper seams around legacy geometry and optional telemetry."""

    simulator: EntitySimulationPort
    nearby_vehicle_ids: Callable[[tuple[str, ...]], Sequence[str]]
    calculated_station_environment: Callable[
        [list[float], Sequence[str], tuple[float, float], str, int],
        tuple[Sequence[float], Sequence[str]],
    ]
    traffic_light_context: Callable[[int], TrafficLightRouteContextPort | None]
    traffic_light_stop_station_m: Callable[
        [str | None, int | None, tuple[float, float]],
        float,
    ]
    pedestrian_error_sink: Callable[[SimulationOperationError], None]


@dataclass(frozen=True, slots=True)
class PedestrianObstacle:
    """One observed pedestrian in the crossing's conflict coordinates."""

    pedestrian_id: str
    station_m: float
    speed_mps: float | None
    state: PedestrianCrossingState
    distance_to_ego_m: float


@dataclass(frozen=True, slots=True)
class _ObjectSamplingEvaluation:
    """Object slots plus an optional strategy-specific QP EGO coordinate."""

    stations_m: tuple[float, ...]
    vehicle_ids: tuple[str, ...]
    supervisor_ego_station_m: float | None = None


@dataclass(frozen=True, slots=True)
class _PedestrianEvaluation:
    """Carry the data required for pedestrian evaluation."""

    station_m: float
    speed_mps: float
    state: PedestrianCrossingState
    ego_station_m: float
    quality: SimulationReadQuality = SimulationReadQuality.LIVE
    pedestrian_id: str | None = None
    pedestrians: tuple[PedestrianObstacle, ...] = ()


@dataclass(frozen=True, slots=True)
class IntersectionStrategyEvaluator:
    """Evaluate one descriptor without retaining mutable step state."""

    _dependencies: IntersectionEvaluatorDependencies

    def evaluate(
        self,
        descriptor: IntersectionDescriptor,
        inputs: IntersectionEvaluationInput,
    ) -> IntersectionEvaluation:
        """Read required simulator data and return one immutable result."""

        objects = self._evaluate_object_sampling(
            descriptor.object_sampling,
            inputs,
        )
        configured_light = self._configured_traffic_light(descriptor)
        ego_station_m = self._evaluate_ego_station(
            descriptor.ego_station,
            inputs.ego_position_xy_m,
            configured_light,
        )
        pedestrian = self._evaluate_pedestrian(descriptor.pedestrian_crossing, inputs)
        if pedestrian is not None:
            ego_station_m = pedestrian.ego_station_m

        return IntersectionEvaluation(
            left_blinker=int(self._blinker_on(descriptor.left_blinker, inputs)),
            right_blinker=int(self._blinker_on(descriptor.right_blinker, inputs)),
            ego_station_m=ego_station_m,
            supervisor_ego_station_m=(
                ego_station_m
                if objects.supervisor_ego_station_m is None
                else objects.supervisor_ego_station_m
            ),
            object_stations_m=objects.stations_m,
            object_vehicle_ids=objects.vehicle_ids,
            pedestrian_station_m=(pedestrian.station_m if pedestrian else -200.0),
            pedestrian_speed_mps=(pedestrian.speed_mps if pedestrian else 0.0),
            pedestrian_state=(
                pedestrian.state
                if pedestrian
                else PedestrianCrossingState.NOT_APPLICABLE
            ),
            pedestrians=(pedestrian.pedestrians if pedestrian else ()),
            traffic_light_state_code=self._traffic_light_state_code(
                descriptor,
                pedestrian,
            ),
            configured_traffic_light=configured_light,
            simulator_read_quality=(
                pedestrian.quality
                if pedestrian is not None
                else SimulationReadQuality.LIVE
            ),
            pedestrian_id=pedestrian.pedestrian_id if pedestrian else None,
        )

    def _evaluate_object_sampling(
        self,
        strategy: ObjectSamplingStrategy,
        inputs: IntersectionEvaluationInput,
    ) -> _ObjectSamplingEvaluation:
        """Evaluate the object sampling owned by this intersection strategy evaluator."""
        if isinstance(strategy, NoObjectSampling):
            return _ObjectSamplingEvaluation(inputs.initial_object_stations_m, ())
        if isinstance(strategy, ConstantObjectSampling):
            return _ObjectSamplingEvaluation(
                (strategy.station_m,) * len(inputs.initial_object_stations_m),
                (),
            )
        if isinstance(strategy, CalculatedObjectSampling):
            stations_m, vehicle_ids = self._calculated_objects(
                strategy,
                inputs.initial_object_stations_m,
            )
            return _ObjectSamplingEvaluation(stations_m, vehicle_ids)
        if isinstance(strategy, PositionObjectSampling):
            stations_m, vehicle_ids = self._position_objects(
                strategy,
                inputs.initial_object_stations_m,
            )
            return _ObjectSamplingEvaluation(stations_m, vehicle_ids)
        if isinstance(strategy, LaneQueueObjectSampling):
            return self._lane_queue_objects(strategy, inputs)
        assert_never(strategy)

    def _calculated_objects(
        self,
        strategy: CalculatedObjectSampling,
        initial_stations_m: tuple[float, ...],
    ) -> tuple[tuple[float, ...], tuple[str, ...]]:
        """Return the calculated objects value owned by this intersection strategy evaluator."""
        vehicle_ids = self._dependencies.nearby_vehicle_ids(strategy.lane_ids)
        stations_m, sampled_ids = self._dependencies.calculated_station_environment(
            list(initial_stations_m),
            vehicle_ids,
            strategy.target_xy_m,
            strategy.target_road_id,
            strategy.station_multiplier,
        )
        return tuple(float(value) for value in stations_m), tuple(sampled_ids)

    def _position_objects(
        self,
        strategy: PositionObjectSampling,
        initial_stations_m: tuple[float, ...],
    ) -> tuple[tuple[float, ...], tuple[str, ...]]:
        """Return the position objects value owned by this intersection strategy evaluator."""
        stations_m = list(initial_stations_m)
        sampled_vehicle_ids: list[str] = []
        vehicle_ids = self._dependencies.nearby_vehicle_ids(strategy.lane_ids)
        for index, vehicle_id in enumerate(vehicle_ids):
            route_matches = self._vehicle_route_matches(
                vehicle_id,
                strategy.required_route_edge_id,
            )
            if route_matches:
                position_xy_m = self._vehicle_position_xy_m(vehicle_id)
                stations_m[index] = self._project(strategy.station_projection, position_xy_m)
                sampled_vehicle_ids.append(vehicle_id)
            else:
                stations_m[index] = strategy.excluded_vehicle_station_m
                sampled_vehicle_ids.append(strategy.excluded_vehicle_id)
        return tuple(stations_m), tuple(sampled_vehicle_ids)

    def _lane_queue_objects(
        self,
        strategy: LaneQueueObjectSampling,
        inputs: IntersectionEvaluationInput,
    ) -> _ObjectSamplingEvaluation:
        """Return the lane queue objects value owned by this intersection strategy evaluator."""
        if str(inputs.ego_road_id) == strategy.disabled_on_road_id:
            return _ObjectSamplingEvaluation(inputs.initial_object_stations_m, ())

        lane_id = (
            inputs.ego_lane_id
            if strategy.use_ego_lane and inputs.ego_lane_id
            else strategy.lane_id
        )
        vehicle_ids = list(
            self._dependencies.simulator.lane.getLastStepVehicleIDs(lane_id)
        )

        stations_m = list(inputs.initial_object_stations_m)
        if strategy.use_ego_lane:
            if strategy.ego_vehicle_id not in vehicle_ids:
                # A current-lane snapshot without EGO cannot establish which
                # vehicles are ahead. Fail open instead of treating followers
                # as longitudinal obstacles.
                vehicle_ids = []
            else:
                ego_lane_position_m = float(
                    self._dependencies.simulator.vehicle.getLanePosition(
                        strategy.ego_vehicle_id
                    )
                )
                ahead_vehicles: list[tuple[float, str]] = []
                for vehicle_id in vehicle_ids:
                    if vehicle_id == strategy.ego_vehicle_id:
                        continue
                    distance_ahead_m = float(
                        self._dependencies.simulator.vehicle.getLanePosition(vehicle_id)
                    ) - ego_lane_position_m
                    if distance_ahead_m > 0.0:
                        ahead_vehicles.append((distance_ahead_m, vehicle_id))
                ahead_vehicles.sort(key=lambda item: (item[0], item[1]))
                vehicle_ids = [
                    vehicle_id
                    for _distance_ahead_m, vehicle_id in ahead_vehicles[: len(stations_m)]
                ]
        else:
            vehicle_ids = [
                vehicle_id
                for vehicle_id in vehicle_ids
                if vehicle_id != strategy.ego_vehicle_id
            ][: len(stations_m)]

        for index, _vehicle_id in enumerate(vehicle_ids):
            stations_m[index] = (
                strategy.first_vehicle_station_m + index * strategy.vehicle_spacing_m
            )
        supervisor_ego_station_m = None
        if vehicle_ids and strategy.supervisor_ego_distance_scale is not None:
            nearest_position_xy_m = self._vehicle_position_xy_m(vehicle_ids[0])
            supervisor_ego_station_m = -abs(
                float(strategy.supervisor_ego_distance_scale)
            ) * math.dist(inputs.ego_position_xy_m, nearest_position_xy_m)
        return _ObjectSamplingEvaluation(
            tuple(stations_m),
            tuple(vehicle_ids),
            supervisor_ego_station_m,
        )

    def _evaluate_pedestrian(
        self,
        descriptor: PedestrianCrossingDescriptor | None,
        inputs: IntersectionEvaluationInput,
    ) -> _PedestrianEvaluation | None:
        """Evaluate the pedestrian owned by this intersection strategy evaluator."""
        if descriptor is None:
            return None

        def read_pedestrian() -> _PedestrianEvaluation | None:
            """Read the pedestrian without changing this intersection strategy evaluator's state."""
            pedestrian_ids = tuple(self._dependencies.simulator.person.getIDList())
            if not pedestrian_ids:
                return None
            pedestrians = tuple(
                self._pedestrian_obstacle(
                    descriptor,
                    pedestrian_id,
                    inputs.ego_position_xy_m,
                )
                for pedestrian_id in pedestrian_ids
            )
            active_pedestrians = tuple(
                pedestrian
                for pedestrian in pedestrians
                if pedestrian.state is PedestrianCrossingState.ON_CROSSING
            )
            candidates = active_pedestrians or pedestrians
            ego_station_m = self._project(
                descriptor.ego_station_with_pedestrian,
                inputs.ego_position_xy_m,
            )
            selected = min(
                candidates,
                key=lambda pedestrian: (
                    pedestrian.distance_to_ego_m,
                    pedestrian.pedestrian_id,
                ),
            )
            selected_speed_mps = float(
                self._dependencies.simulator.person.getSpeed(
                    selected.pedestrian_id
                )
            )
            selected = replace(
                selected,
                speed_mps=selected_speed_mps,
            )
            pedestrians = tuple(
                selected
                if pedestrian.pedestrian_id == selected.pedestrian_id
                else pedestrian
                for pedestrian in pedestrians
            )
            return _PedestrianEvaluation(
                station_m=selected.station_m,
                speed_mps=selected_speed_mps,
                state=selected.state,
                pedestrian_id=selected.pedestrian_id,
                ego_station_m=ego_station_m,
                pedestrians=pedestrians,
            )

        pedestrian_read = self._dependencies.simulator.optional_read(
            "read Susan pedestrian crossing context",
            read_pedestrian,
            None,
        )
        if pedestrian_read.value is not None:
            return pedestrian_read.value
        if pedestrian_read.error is not None:
            self._dependencies.pedestrian_error_sink(pedestrian_read.error)
        return self._unavailable_pedestrian(
            descriptor,
            inputs.ego_position_xy_m,
            quality=pedestrian_read.quality,
        )

    def _pedestrian_obstacle(
        self,
        descriptor: PedestrianCrossingDescriptor,
        pedestrian_id: str,
        ego_position_xy_m: tuple[float, float],
    ) -> PedestrianObstacle:
        """Read one pedestrian exactly once into the crossing coordinates."""

        pedestrian_position = self._dependencies.simulator.person.getPosition(
            pedestrian_id
        )
        pedestrian_position_xy_m = (
            float(pedestrian_position[0]),
            float(pedestrian_position[1]),
        )
        station_m = self._pedestrian_station_m(
            descriptor,
            pedestrian_position_xy_m[1],
        )
        state = (
            PedestrianCrossingState.CLEAR
            if station_m < descriptor.on_crossing_lower_station_m
            or station_m > 0.0
            else PedestrianCrossingState.ON_CROSSING
        )
        return PedestrianObstacle(
            pedestrian_id=str(pedestrian_id),
            station_m=station_m,
            speed_mps=None,
            state=state,
            distance_to_ego_m=math.dist(
                ego_position_xy_m,
                pedestrian_position_xy_m,
            ),
        )

    @staticmethod
    def _pedestrian_station_m(
        descriptor: PedestrianCrossingDescriptor,
        pedestrian_y_m: float,
    ) -> float:
        """Return the pedestrian station m value owned by this intersection strategy evaluator."""
        if pedestrian_y_m < descriptor.entry_y_m:
            return pedestrian_y_m - descriptor.entry_y_m
        if pedestrian_y_m < descriptor.exit_y_m:
            return 0.0
        return descriptor.beyond_crossing_station_m

    @classmethod
    def _unavailable_pedestrian(
        cls,
        descriptor: PedestrianCrossingDescriptor,
        ego_position_xy_m: tuple[float, float],
        *,
        quality: SimulationReadQuality = SimulationReadQuality.LIVE,
    ) -> _PedestrianEvaluation:
        """Return the unavailable pedestrian value owned by this intersection strategy evaluator."""
        return _PedestrianEvaluation(
            station_m=descriptor.unavailable_station_m,
            speed_mps=descriptor.unavailable_speed_mps,
            state=PedestrianCrossingState.CLEAR,
            ego_station_m=cls._project(
                descriptor.ego_station_without_pedestrian,
                ego_position_xy_m,
            ),
            quality=quality,
        )

    def _configured_traffic_light(
        self,
        descriptor: IntersectionDescriptor,
    ) -> ConfiguredTrafficLightEvaluation | None:
        """Return the configured traffic light value owned by this intersection strategy evaluator.
        """
        if not descriptor.uses_configured_traffic_light:
            return None
        context = self._dependencies.traffic_light_context(descriptor.number)
        if context is None:
            return None
        return ConfiguredTrafficLightEvaluation(
            tls_id=str(context.tls_id),
            green_phase=int(context.green_phase),
            link_indices=(int(context.link_index),),
            route_internal_edge=str(context.route_internal_edge),
        )

    def _evaluate_ego_station(
        self,
        strategy: EgoStationStrategy,
        ego_position_xy_m: tuple[float, float],
        traffic_light: ConfiguredTrafficLightEvaluation | None,
    ) -> float:
        """Evaluate the ego station owned by this intersection strategy evaluator."""
        if isinstance(strategy, StationProjection):
            return self._project(strategy, ego_position_xy_m)
        if isinstance(strategy, ConstantStation):
            return strategy.station_m
        if isinstance(strategy, TrafficLightStopStation):
            tls_id = traffic_light.tls_id if traffic_light else None
            green_phase = traffic_light.green_phase if traffic_light else None
            return self._dependencies.traffic_light_stop_station_m(
                tls_id,
                green_phase,
                ego_position_xy_m,
            )
        assert_never(strategy)

    @staticmethod
    def _traffic_light_state_code(
        descriptor: IntersectionDescriptor,
        pedestrian: _PedestrianEvaluation | None,
    ) -> int:
        """Return the traffic light state code value owned by this intersection strategy evaluator.
        """
        if pedestrian is not None:
            return int(pedestrian.state)
        if descriptor.uses_configured_traffic_light:
            return TRAFFIC_LIGHT_STATE_UNKNOWN
        return TRAFFIC_LIGHT_STATE_NOT_APPLICABLE

    def _vehicle_position_xy_m(self, vehicle_id: str) -> tuple[float, float]:
        """Return the vehicle position xy m value owned by this intersection strategy evaluator."""
        position = self._dependencies.simulator.vehicle.getPosition(vehicle_id)
        return float(position[0]), float(position[1])

    def _vehicle_route_matches(
        self,
        vehicle_id: str,
        required_route_edge_id: str | None,
    ) -> bool:
        """Return the vehicle route matches value owned by this intersection strategy evaluator."""
        if required_route_edge_id is None:
            return True
        route = self._dependencies.simulator.vehicle.getRoute(vehicle_id)
        return required_route_edge_id in route

    @classmethod
    def _blinker_on(
        cls,
        rule: BlinkerRule,
        inputs: IntersectionEvaluationInput,
    ) -> bool:
        """Return the blinker on value owned by this intersection strategy evaluator."""
        if rule.always_on:
            return True
        return any(
            all(
                cls._condition_matches(condition, inputs.ego_position_xy_m)
                for condition in group.all_of
            )
            for group in rule.any_of
        )

    @classmethod
    def _condition_matches(
        cls,
        condition: AxisCondition,
        position_xy_m: tuple[float, float],
    ) -> bool:
        """Return the condition matches value owned by this intersection strategy evaluator."""
        coordinate_m = cls._coordinate_m(condition.axis, position_xy_m)
        if condition.operator is ComparisonOperator.LESS_THAN:
            return coordinate_m < condition.threshold_m
        return coordinate_m > condition.threshold_m

    @classmethod
    def _project(
        cls,
        projection: StationProjection,
        position_xy_m: tuple[float, float],
    ) -> float:
        """Project the operation owned by this intersection strategy evaluator."""
        coordinate_m = cls._coordinate_m(projection.axis, position_xy_m)
        return projection.station_origin_m + projection.coordinate_scale * coordinate_m

    @staticmethod
    def _coordinate_m(
        axis: CoordinateAxis,
        position_xy_m: tuple[float, float],
    ) -> float:
        """Return the coordinate m value owned by this intersection strategy evaluator."""
        return position_xy_m[0] if axis is CoordinateAxis.X else position_xy_m[1]
