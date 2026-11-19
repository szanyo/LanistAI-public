# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Build immutable scenario snapshots from the current SUMO state."""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, cast

from lanista.domains.automotive.control.longitudinal_control_optimizer import (
    nominal_pedestrian_avoidance_approach_station_m,
    nominal_vehicle_avoidance_approach_station_m,
)
from lanista.reinforcengine.utils.simulation.intersection_metadata import (
    INTERSECTION_TYPE_UNKNOWN,
    TRAFFIC_LIGHT_STATE_GREEN,
    TRAFFIC_LIGHT_STATE_NOT_APPLICABLE,
    TRAFFIC_LIGHT_STATE_RED,
    TRAFFIC_LIGHT_STATE_RED_YELLOW,
    TRAFFIC_LIGHT_STATE_UNKNOWN,
    TRAFFIC_LIGHT_STATE_YELLOW,
    DynamicIntersectionTypeClassifier,
)
from lanista.reinforcengine.utils.simulation.ports import (
    EntitySimulationPort,
    SimulationOperationError,
    SimulationRead,
    SimulationReadQuality,
)

from ..control.controller_config import (
    CONTROLLER_CONFIG,
    INACTIVE_SUPERVISOR_EGO_STATION_M,
    LEGACY_VEHICLE_CONFLICT_RELEASE_M,
    LEGACY_VEHICLE_CONFLICT_WINDOW_START_M,
    LongitudinalLeaderQpProjection,
    TrafficLightRouteContext,
)
from ..observation.policy_obstacle_projection import (
    PolicyObstacleCandidate,
    PolicyObstacleGeometry,
    PolicyObstacleKind,
    build_policy_obstacle_projection,
    select_nearest_policy_obstacles,
)
from ..simulation.simulation_contracts import ScenarioContext, TrafficLightContext
from .helper_functions import (
    calculate_station_environment,
    get_nearby_vehicles,
    handle_traffic_light,
    ref_speed_calc,
)
from .intersection_strategy_descriptors import (
    INTERSECTION_DESCRIPTORS,
    LaneQueueObjectSampling,
    PedestrianCrossingState,
)
from .intersection_strategy_evaluator import (
    IntersectionEvaluation,
    IntersectionEvaluationInput,
    IntersectionEvaluatorDependencies,
    IntersectionStrategyEvaluator,
    PedestrianObstacle,
    TrafficLightRouteContextPort,
)
from .preprocessor_sumo_context_service import PreprocessorSumoContextService
from .switch_dict import switch_dict

PED_STATION_DEFAULT_M = -200.0
PED_VELOCITY_DEFAULT_MPS = 0.0
PEDESTRIAN_STATE_DEFAULT = PedestrianCrossingState.NOT_APPLICABLE

_LOGGER = logging.getLogger(__name__)


class _ControllerConfigPort(Protocol):
    """Define the typed boundary for controller config."""

    def get_ego_station_observation_offset(
        self,
        intersection_number: int,
        default: float = 0.0,
    ) -> float:
        """Return the ego station observation offset exposed by this controller config port."""
        ...

    def get_ego_lead_vehicle_lookahead_meters(
        self,
        default: float = 60.0,
    ) -> float:
        """Return the ego lead vehicle lookahead meters exposed by this controller config port."""
        ...

    def get_traffic_light_route_context(
        self,
        intersection_number: int,
    ) -> TrafficLightRouteContext | None:
        """Return the traffic light route context exposed by this controller config port."""
        ...

    def project_longitudinal_leader_constraint(
        self,
        bumper_gap_m: float,
    ) -> LongitudinalLeaderQpProjection:
        """Project an already-read leader gap into the original QP coordinates."""
        ...


class _IntersectionTypeClassifierPort(Protocol):
    """Define the typed boundary for intersection type classifier."""

    def classify(
        self,
        sumo_module: EntitySimulationPort,
        ego_vehicle_id: str = "EGO",
        current_road_id: str | None = None,
    ) -> SimulationRead[int]:
        """Classify the operation owned by this intersection type classifier port."""
        ...


class _LoggerPort(Protocol):
    """Define the typed boundary for logger."""

    def warning(self, message: object) -> None:
        """Publish a warning record through this logger port."""
        ...

    def error(self, message: object) -> None:
        """Publish a error record through this logger port."""
        ...


@dataclass(frozen=True, slots=True)
class _RankedObjectStation:
    """A station value paired with its original vehicle slot."""

    station_m: float
    source_index: int


class ScenarioContextPreprocessor:
    """Translate mutable SUMO state into one immutable controller snapshot."""

    def __init__(
        self,
        *,
        simulator_port: EntitySimulationPort,
        network_file_path: str | Path | None = None,
        controller_config: _ControllerConfigPort = CONTROLLER_CONFIG,
    ) -> None:
        """Initialize the intersection strategies and their mutable state."""
        self._controller_config = controller_config
        self._simulator = simulator_port
        self._sumo_context = PreprocessorSumoContextService(simulator_port)
        self._logger: _LoggerPort = _LOGGER
        self._intersection_evaluator: IntersectionStrategyEvaluator = (
            self._build_intersection_evaluator()
        )

        self._ref_speed: float = 0.0
        self._ped_station: float = 0.0
        self._ped_velocity: float = 0.0
        self._pedestrian_state = PedestrianCrossingState.ON_CROSSING
        self._pedestrian_id: str | None = None
        self._pedestrians: tuple[PedestrianObstacle, ...] = ()

        self._intersection_number: int = 0
        self._station_environment: list[float] = []
        self._car_environment: list[str] = []
        self._physical_vehicle_obstacles: tuple[PolicyObstacleCandidate, ...] = ()
        self._ranked_supervisor_vehicle_ids: tuple[str | None, ...] = ()
        self._policy_obstacle_candidates: tuple[PolicyObstacleCandidate, ...] = ()
        self._policy_object_stations_m: tuple[float, ...] = ()
        self._policy_object_speeds_mps: tuple[float, ...] = ()
        self._policy_object_constraint_clearances_m: tuple[float, ...] = ()
        self._vehicle_speed_cache: dict[str, float] = {}
        self._pedestrian_speed_cache: dict[str, float] = {}

        self._ego_position_xy_m = (0.0, 0.0)
        self._ego_road_id = ""
        self._speed_environment: list[float] = []

        self._ego_station_m: float = 0.0
        self._raw_ego_station_m: float = 0.0
        self._target_ego_station_m: float = 0.0
        self._supervisor_ego_station_m: float = 0.0
        self._ego_lead_vehicle_present: bool = False
        self._ego_lead_distance: float = 0.0
        self._ego_lead_speed_mps: float = 0.0
        self._ego_lead_vehicle_id: str | None = None
        self._blinker_out: int = 0
        self._blinker_left: int = 0
        self._blinker_right: int = 0
        self._traffic_light_state_code: int = TRAFFIC_LIGHT_STATE_UNKNOWN
        self._traffic_light_id: str | None = None
        self._traffic_light_green_phase: int | None = None
        self._traffic_light_context: TrafficLightContext = TrafficLightContext()
        self._pinned_traffic_light_context: TrafficLightContext = (
            TrafficLightContext()
        )
        self._traffic_light_link_indices: tuple[int, ...] = ()
        self._traffic_light_route_internal_edge: str | None = None
        self._traffic_light_virtual_qp_constraint_active: bool = False
        self._longitudinal_leader_qp_constraint_active: bool = False
        self._ego_lead_qp_projection: LongitudinalLeaderQpProjection | None = None
        self._intersection_type_classifier: _IntersectionTypeClassifierPort = (
            DynamicIntersectionTypeClassifier(network_file_path)
        )
        self._upcoming_intersection_type_code: int = INTERSECTION_TYPE_UNKNOWN
        self._simulator_read_quality: SimulationReadQuality = (
            SimulationReadQuality.LIVE
        )

    @property
    def controller_config(self) -> _ControllerConfigPort:
        """Return the controller config exposed by this scenario context preprocessor."""
        return self._controller_config

    def reset_episode(self) -> None:
        """Clear caches whose identity must not survive a SUMO episode reset."""

        self._pinned_traffic_light_context = TrafficLightContext()

    def _reset(self) -> None:
        """Reset per-step values and load the mandatory EGO context."""
        # Reference speed is mandatory control context. Continuing with zero
        # after losing SUMO would corrupt scenario, reward and observation data.
        self._simulator_read_quality = SimulationReadQuality.LIVE
        self._ref_speed = ref_speed_calc(self._simulator)

        self._ped_station = PED_STATION_DEFAULT_M
        self._ped_velocity = PED_VELOCITY_DEFAULT_MPS
        self._pedestrian_state = PEDESTRIAN_STATE_DEFAULT
        self._pedestrian_id = None
        self._pedestrians = ()
        self._traffic_light_state_code = TRAFFIC_LIGHT_STATE_NOT_APPLICABLE
        self._traffic_light_id = None
        self._traffic_light_green_phase = None
        self._traffic_light_link_indices = ()
        self._traffic_light_route_internal_edge = None
        self._traffic_light_virtual_qp_constraint_active = False
        self._longitudinal_leader_qp_constraint_active = False
        self._ego_lead_qp_projection = None
        self._traffic_light_context = TrafficLightContext()

        self._ego_position_xy_m = self._simulator.vehicle.getPosition("EGO")
        self._ego_road_id = self._simulator.vehicle.getRoadID("EGO")

        self._set_intersection_number()
        classification = self._intersection_type_classifier.classify(
            self._simulator,
            current_road_id=self._ego_road_id,
        )
        self._upcoming_intersection_type_code = classification.value
        self._record_read_quality(classification.quality)

        self._station_environment = [-200] * 100
        self._car_environment = []
        self._physical_vehicle_obstacles = ()
        self._ranked_supervisor_vehicle_ids = ()
        self._policy_obstacle_candidates = ()
        self._policy_object_stations_m = (-200.0, -200.0, -200.0)
        self._policy_object_speeds_mps = (0.0, 0.0, 0.0)
        self._policy_object_constraint_clearances_m = (
            math.nan,
            math.nan,
            math.nan,
        )
        self._vehicle_speed_cache = {}
        self._pedestrian_speed_cache = {}
        self._speed_environment = [0.0] * 3
        self._target_ego_station_m = 0
        self._supervisor_ego_station_m = 0.0
        self._ego_lead_vehicle_present = False
        self._ego_lead_distance = 0.0
        self._ego_lead_speed_mps = 0.0
        self._ego_lead_vehicle_id = None

    def update(self, ego_speed_mps: float) -> ScenarioContext:
        """Advance preprocessing and return the context for this control step."""

        self._reset()

        # This command is part of the control step, so a rejected command must
        # terminate the step instead of producing observations for stale input.
        self._simulator.vehicle.setSpeed("EGO", ego_speed_mps)

        self._evaluate_intersection()
        self._target_ego_station_m = self._ego_station_m
        self._refresh_ego_lead_vehicle_context()
        self._refresh_traffic_light_context()
        self._apply_traffic_light_qp_constraint()
        self._apply_longitudinal_leader_qp_constraint()
        self._apply_physical_vehicle_supervisor_window()
        self._apply_ego_station_observation_offset()

        if self._ego_station_m > 99 and self._intersection_number not in {1, 6}:
            self._blinker_left = self._blinker_right = 0

        self._blinker_out = 2 if self._blinker_left == 1 else 0 if self._blinker_right == 1 else 1

        self._rank_supervisor_vehicle_context()
        self._project_policy_obstacles()

        return self.current_context

    def _set_intersection_number(self) -> None:
        """Map the current road identifier to its route intersection."""
        self._intersection_number = switch_dict.get(str(self._ego_road_id), -1)
        if self._intersection_number == -1:
            self._logger.warning("Not defined road!!")
            self._logger.warning(f"ego_road_id: {self._ego_road_id}")

    def _build_intersection_evaluator(self) -> IntersectionStrategyEvaluator:
        """Bind legacy geometry helpers behind the typed evaluator ports."""

        return IntersectionStrategyEvaluator(
            IntersectionEvaluatorDependencies(
                simulator=self._simulator,
                nearby_vehicle_ids=lambda lane_ids: get_nearby_vehicles(
                    self._simulator,
                    *lane_ids,
                ),
                calculated_station_environment=lambda stations, vehicles, target, road, multiplier: (
                    calculate_station_environment(
                        self._simulator,
                        stations,
                        vehicles,
                        target,
                        road,
                        multiplier=multiplier,
                    )
                ),
                traffic_light_context=cast(
                    Callable[
                        [int],
                        TrafficLightRouteContextPort | None,
                    ],
                    self.controller_config.get_traffic_light_route_context,
                ),
                traffic_light_stop_station_m=lambda tls_id, green_phase, position: float(
                    handle_traffic_light(
                        self._simulator,
                        tls_id,
                        green_phase,
                        position,
                    )[0]
                ),
                pedestrian_error_sink=self._report_pedestrian_read_error,
            )
        )

    def _evaluate_intersection(self) -> None:
        """Evaluate and apply the descriptor selected by the current road."""

        descriptor = INTERSECTION_DESCRIPTORS.get(self._intersection_number)
        if descriptor is None:
            result = IntersectionEvaluation.unknown(len(self._station_environment))
            self._log_unknown_intersection()
        else:
            ego_lane_id = None
            if (
                isinstance(descriptor.object_sampling, LaneQueueObjectSampling)
                and descriptor.object_sampling.use_ego_lane
            ):
                ego_lane_id = str(self._simulator.vehicle.getLaneID("EGO"))
            result = self._intersection_evaluator.evaluate(
                descriptor,
                IntersectionEvaluationInput(
                    ego_position_xy_m=(
                        float(self._ego_position_xy_m[0]),
                        float(self._ego_position_xy_m[1]),
                    ),
                    ego_road_id=self._ego_road_id,
                    initial_object_stations_m=tuple(self._station_environment),
                    ego_lane_id=ego_lane_id,
                ),
            )
        self._apply_intersection_evaluation(result)

    def _apply_intersection_evaluation(
        self,
        result: IntersectionEvaluation,
    ) -> None:
        """Copy one immutable strategy result into the legacy facade state."""

        self._blinker_left = result.left_blinker
        self._blinker_right = result.right_blinker
        self._ego_station_m = result.ego_station_m
        self._supervisor_ego_station_m = result.supervisor_ego_station_m
        self._station_environment = list(result.object_stations_m)
        self._car_environment = list(result.object_vehicle_ids)
        self._physical_vehicle_obstacles = tuple(
            PolicyObstacleCandidate(
                actor_id=vehicle_id,
                kind=PolicyObstacleKind.VEHICLE,
                station_m=float(result.object_stations_m[index]),
                speed_mps=None,
            )
            for index, vehicle_id in enumerate(result.object_vehicle_ids)
            if vehicle_id != "-1" and index < len(result.object_stations_m)
        )
        self._ped_station = result.pedestrian_station_m
        self._ped_velocity = result.pedestrian_speed_mps
        self._pedestrian_state = result.pedestrian_state
        self._pedestrian_id = result.pedestrian_id
        self._pedestrians = result.pedestrians
        self._pedestrian_speed_cache = {
            pedestrian.pedestrian_id: float(pedestrian.speed_mps)
            for pedestrian in result.pedestrians
            if pedestrian.speed_mps is not None
        }
        self._traffic_light_state_code = result.traffic_light_state_code
        self._record_read_quality(result.simulator_read_quality)

        light = result.configured_traffic_light
        self._traffic_light_id = light.tls_id if light else None
        self._traffic_light_green_phase = light.green_phase if light else None
        self._traffic_light_link_indices = light.link_indices if light else ()
        self._traffic_light_route_internal_edge = light.route_internal_edge if light else None

    def _rank_supervisor_vehicle_context(self) -> None:
        """Keep the original three vehicle-only QP slots and aligned speeds."""

        ranked_stations = sorted(
            (
                _RankedObjectStation(float(station_m), source_index)
                for source_index, station_m in enumerate(self._station_environment)
            ),
            key=lambda station: station.station_m,
            reverse=True,
        )[:3]
        self._station_environment = [station.station_m for station in ranked_stations]
        ranked_ids: list[str | None] = []
        ranked_speeds: list[float] = []
        for station in ranked_stations:
            vehicle_id = (
                self._car_environment[station.source_index]
                if station.source_index < len(self._car_environment)
                else None
            )
            ranked_ids.append(vehicle_id)
            is_fixed_longitudinal_axis = (
                self._longitudinal_leader_qp_constraint_active
                and vehicle_id == self._ego_lead_vehicle_id
            )
            ranked_speeds.append(
                0.0
                if is_fixed_longitudinal_axis
                else self._vehicle_speed_mps(vehicle_id)
            )
        self._ranked_supervisor_vehicle_ids = tuple(ranked_ids)
        self._speed_environment = ranked_speeds

    def _project_policy_obstacles(self) -> None:
        """Fill policy slots nearest-first while keeping each actor record aligned."""

        direct_leader_candidates: tuple[PolicyObstacleCandidate, ...] = ()
        direct_leader_id: str | None = None
        if (
            self._ego_lead_vehicle_present
            and self._ego_lead_vehicle_id
        ):
            direct_leader_id = self._ego_lead_vehicle_id
            direct_leader_candidates = (
                PolicyObstacleCandidate(
                    actor_id=direct_leader_id,
                    kind=PolicyObstacleKind.VEHICLE,
                    station_m=-max(0.0, float(self._ego_lead_distance)),
                    speed_mps=max(0.0, float(self._ego_lead_speed_mps)),
                    distance_to_ego_m=max(
                        0.0,
                        float(self._ego_lead_distance),
                    ),
                    geometry=PolicyObstacleGeometry.LONGITUDINAL_GAP,
                    constraint_clearance_m=(
                        projection.constraint_clearance_m
                        if (
                            projection := getattr(
                                self,
                                "_ego_lead_qp_projection",
                                None,
                            )
                        )
                        is not None
                        else None
                    ),
                ),
            )
        physical_vehicle_candidates = tuple(
            candidate
            for candidate in self._physical_vehicle_obstacles
            if candidate.actor_id != direct_leader_id
        )
        pedestrian_candidates = tuple(
            PolicyObstacleCandidate(
                actor_id=pedestrian.pedestrian_id,
                kind=PolicyObstacleKind.PEDESTRIAN,
                station_m=pedestrian.station_m,
                speed_mps=pedestrian.speed_mps,
                distance_to_ego_m=pedestrian.distance_to_ego_m,
            )
            for pedestrian in self._pedestrians
            if pedestrian.state is PedestrianCrossingState.ON_CROSSING
        )
        selected = select_nearest_policy_obstacles(
            direct_leader_candidates
            + physical_vehicle_candidates
            + pedestrian_candidates,
            ego_station_m=self._raw_ego_station_m,
        )
        resolved = tuple(
            self._resolve_vehicle_policy_candidate(candidate)
            if candidate.kind is PolicyObstacleKind.VEHICLE
            else self._resolve_pedestrian_policy_candidate(candidate)
            for candidate in selected
        )
        projection = build_policy_obstacle_projection(resolved)
        self._policy_obstacle_candidates = projection.candidates
        self._policy_object_stations_m = projection.stations_m
        self._policy_object_speeds_mps = projection.speeds_mps
        self._policy_object_constraint_clearances_m = (
            projection.constraint_clearances_m
        )

    def _resolve_vehicle_policy_candidate(
        self,
        candidate: PolicyObstacleCandidate,
    ) -> PolicyObstacleCandidate:
        """Resolve speed and fixed QP clearance for one selected vehicle."""

        speed_mps = (
            candidate.speed_mps
            if candidate.speed_mps is not None
            else self._vehicle_speed_mps(candidate.actor_id)
        )
        clearance_m = candidate.constraint_clearance_m
        if (
            clearance_m is None
            and not getattr(
                self,
                "_traffic_light_virtual_qp_constraint_active",
                False,
            )
            and not getattr(
                self,
                "_longitudinal_leader_qp_constraint_active",
                False,
            )
            and candidate.geometry is PolicyObstacleGeometry.CONFLICT_POINT
        ):
            try:
                boundary_station_m = nominal_vehicle_avoidance_approach_station_m(
                    candidate.station_m,
                )
            except ValueError:
                pass
            else:
                clearance_m = max(
                    0.0,
                    boundary_station_m
                    - float(
                        getattr(
                            self,
                            "_supervisor_ego_station_m",
                            self._raw_ego_station_m,
                        )
                    ),
                )
        return replace(
            candidate,
            speed_mps=speed_mps,
            constraint_clearance_m=clearance_m,
        )

    def _resolve_pedestrian_policy_candidate(
        self,
        candidate: PolicyObstacleCandidate,
    ) -> PolicyObstacleCandidate:
        """Resolve speed and clearance to the existing pedestrian QP circle."""

        speed_mps = (
            candidate.speed_mps
            if candidate.speed_mps is not None
            else self._pedestrian_speed_mps(candidate.actor_id)
        )
        clearance_m = candidate.constraint_clearance_m
        if clearance_m is None:
            try:
                boundary_station_m = (
                    nominal_pedestrian_avoidance_approach_station_m(
                        candidate.station_m,
                    )
                )
            except ValueError:
                pass
            else:
                clearance_m = max(
                    0.0,
                    boundary_station_m
                    - float(
                        getattr(
                            self,
                            "_supervisor_ego_station_m",
                            self._raw_ego_station_m,
                        )
                    ),
                )
        return replace(
            candidate,
            speed_mps=speed_mps,
            constraint_clearance_m=clearance_m,
        )

    def _vehicle_speed_mps(self, vehicle_id: str | None) -> float:
        """Read each selected physical vehicle speed at most once per step."""

        if not vehicle_id or vehicle_id == "-1":
            return 0.0
        if vehicle_id not in self._vehicle_speed_cache:
            self._vehicle_speed_cache[vehicle_id] = float(
                self._simulator.vehicle.getSpeed(vehicle_id)
            )
        return self._vehicle_speed_cache[vehicle_id]

    def _pedestrian_speed_mps(self, pedestrian_id: str) -> float:
        """Read each policy-selected pedestrian speed at most once per step."""

        if pedestrian_id not in self._pedestrian_speed_cache:
            self._pedestrian_speed_cache[pedestrian_id] = float(
                self._simulator.person.getSpeed(pedestrian_id)
            )
        return self._pedestrian_speed_cache[pedestrian_id]

    def _report_pedestrian_read_error(
        self,
        error: SimulationOperationError,
    ) -> None:
        """Report the pedestrian read error owned by this scenario context
preprocessor."""
        self._logger.warning("Pedestrian context could not be read; using safe defaults.")
        self._logger.error(error)

    def _log_unknown_intersection(self) -> None:
        """Log the unknown intersection owned by this scenario context preprocessor."""
        self._logger.error("WRONG INTERSECTION")
        diagnostic = self._simulator.optional_read(
            "read Susan unknown-intersection diagnostics",
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

    def _apply_ego_station_observation_offset(self) -> None:
        """Project one physical signal zero into policy and supervisor spaces."""
        self._raw_ego_station_m = self._ego_station_m
        route_context = self.controller_config.get_traffic_light_route_context(
            self._intersection_number
        )
        if route_context is not None:
            self._target_ego_station_m = (
                self._raw_ego_station_m
                - route_context.signal_boundary_raw_station_m
            )
            self._ego_station_m = self._target_ego_station_m
            if self._traffic_light_virtual_qp_constraint_active:
                # The QP keeps its original circular object geometry, but its
                # stationary negative branch must touch the same physical zero
                # consumed by observation, reward, and overdrive detection.
                self._supervisor_ego_station_m = (
                    self._target_ego_station_m
                    + route_context.qp_stationary_stop_raw_station_m
                )
            return

        self._ego_station_m = self._raw_ego_station_m
        self._target_ego_station_m = self._raw_ego_station_m

    def _apply_physical_vehicle_supervisor_window(self) -> None:
        """Release legacy vehicle constraints outside their conflict window.

        The original Bence preprocessor used the ``[-25, -6]`` station window
        to keep cross-traffic constraints active only while approaching the
        conflict point.  Preserve that behavior exclusively on the supervisor
        coordinate: policy observations and reward geometry must retain the
        continuous physical station.
        """

        if self._longitudinal_leader_qp_constraint_active:
            return
        if not self._physical_vehicle_obstacles:
            return
        if self._pedestrian_state is not PedestrianCrossingState.NOT_APPLICABLE:
            return
        if (
            self.controller_config.get_traffic_light_route_context(
                self._intersection_number
            )
            is not None
        ):
            # Configured signal routes use the same QP slots for a real queue;
            # those longitudinal leaders must never be released as cross traffic.
            return
        if (
            self._supervisor_ego_station_m
            < LEGACY_VEHICLE_CONFLICT_WINDOW_START_M
            or self._supervisor_ego_station_m
            > LEGACY_VEHICLE_CONFLICT_RELEASE_M
        ):
            self._supervisor_ego_station_m = INACTIVE_SUPERVISOR_EGO_STATION_M

    def _apply_traffic_light_qp_constraint(self) -> None:
        """Restore the original non-green virtual object used by the QP."""

        self._traffic_light_virtual_qp_constraint_active = False

        route_context = self.controller_config.get_traffic_light_route_context(
            self._intersection_number
        )
        if route_context is None:
            return
        if str(self._ego_road_id) == route_context.route_internal_edge:
            return
        if self._traffic_light_context.compact_state == TRAFFIC_LIGHT_STATE_GREEN:
            return
        if any(vehicle_id != "-1" for vehicle_id in self._car_environment):
            # A real object ahead already owns the original Bence QP slot.
            return
        if not self._station_environment:
            return

        self._station_environment[0] = route_context.qp_restricted_object_station_m
        if self._car_environment:
            self._car_environment[0] = "-1"
        else:
            self._car_environment.append("-1")
        self._traffic_light_virtual_qp_constraint_active = True

    def _apply_longitudinal_leader_qp_constraint(self) -> None:
        """Restore the original global same-lane leader QP projection.

        The signal and direct leader remain independent physical constraints.
        Whichever fixed boundary is nearer to EGO owns the shared QP vehicle
        coordinate for this step.
        """

        self._longitudinal_leader_qp_constraint_active = False
        self._ego_lead_qp_projection = None
        if (
            not self._ego_lead_vehicle_present
            or not self._ego_lead_vehicle_id
        ):
            return
        try:
            projection = self.controller_config.project_longitudinal_leader_constraint(
                self._ego_lead_distance
            )
        except (TypeError, ValueError):
            return
        self._ego_lead_qp_projection = projection

        signal_clearance_m = self._active_signal_clearance_m()
        if (
            signal_clearance_m is not None
            and signal_clearance_m <= projection.constraint_clearance_m
        ):
            route_context = self.controller_config.get_traffic_light_route_context(
                self._intersection_number
            )
            if route_context is not None:
                self._station_environment = [
                    route_context.qp_restricted_object_station_m,
                    -200.0,
                    -200.0,
                ]
                self._car_environment = ["-1"]
                self._traffic_light_virtual_qp_constraint_active = True
            return

        self._station_environment = [
            projection.object_station_m,
            -200.0,
            -200.0,
        ]
        self._car_environment = [self._ego_lead_vehicle_id]
        self._supervisor_ego_station_m = projection.ego_station_m
        self._traffic_light_virtual_qp_constraint_active = False
        self._longitudinal_leader_qp_constraint_active = True

    def _active_signal_clearance_m(self) -> float | None:
        """Return the active configured signal's fixed physical clearance."""

        route_context = self.controller_config.get_traffic_light_route_context(
            self._intersection_number
        )
        if route_context is None:
            return None
        if str(self._ego_road_id) == route_context.route_internal_edge:
            return None
        if self._traffic_light_context.compact_state == TRAFFIC_LIGHT_STATE_GREEN:
            return None
        return max(
            0.0,
            route_context.signal_boundary_raw_station_m - self._ego_station_m,
        )

    def _refresh_ego_lead_vehicle_context(self) -> None:
        """Refresh the ego lead vehicle context owned by this scenario
context preprocessor."""
        context = self._sumo_context.read_ego_leader(
            self.controller_config.get_ego_lead_vehicle_lookahead_meters(60.0)
        )
        self._ego_lead_vehicle_present = context.present
        self._ego_lead_distance = context.distance
        self._ego_lead_speed_mps = context.speed_mps
        self._ego_lead_vehicle_id = context.vehicle_id
        if context.present and context.vehicle_id:
            self._vehicle_speed_cache[context.vehicle_id] = context.speed_mps
        self._record_read_quality(context.quality)

    def _record_read_quality(
        self,
        quality: SimulationReadQuality,
    ) -> None:
        """Keep fallback evidence sticky for the whole preprocessing step."""

        if quality is SimulationReadQuality.FALLBACK:
            self._simulator_read_quality = quality

    def _traffic_light_link_indices_from_configured_context(
        self,
        traffic_light_id: str | None,
    ) -> tuple[int, ...]:
        """Return the configured route-relevant traffic-light link."""
        if traffic_light_id is None:
            return ()

        configured_context = self.controller_config.get_traffic_light_route_context(
            self._intersection_number
        )
        if configured_context is None:
            return ()

        configured_tls_id = str(configured_context.tls_id)
        if str(traffic_light_id) != configured_tls_id:
            return ()

        return (int(configured_context.link_index),)


    def _traffic_light_link_indices_from_controlled_links(
        self,
        traffic_light_id: str,
    ) -> tuple[int, ...]:
        """Return the traffic light link indices from controlled links value owned by this scenario
        context preprocessor.
        """
        read = self._sumo_context.route_link_indices(traffic_light_id)
        self._record_read_quality(read.quality)
        return read.value

    def _resolve_traffic_light_link_indices(
        self,
        traffic_light_id: str | None,
    ) -> tuple[int, ...]:
        """Resolve the traffic light link indices required by this scenario context preprocessor."""
        if traffic_light_id is None:
            return ()

        # Canonical path: use the route context derived by this preprocessor.
        link_indices = self._traffic_light_link_indices_from_configured_context(
            traffic_light_id
        )
        if link_indices:
            return link_indices

        # Fallback only for non-configured future lights.
        return self._traffic_light_link_indices_from_controlled_links(traffic_light_id)

    @staticmethod
    def _map_signal_char_to_state_code(signal_char: object) -> int:
        """Map the signal char to state code owned by this scenario context preprocessor."""
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

    def _traffic_light_state_from_link_index(
        self,
        traffic_light_id: str | None,
        link_indices: tuple[int, ...],
    ) -> int | None:
        """
        Read the compact traffic-light state from the exact route-relevant TLS
        link index used by TrafficLightContext.

        This keeps traffic_light_state_code and tl_phase_remaining_norm on the same
        canonical source. The old green_phase fallback can collapse yellow,
        red-yellow and other non-green phases into RED, which makes the state
        stay unchanged while the schedule resets underneath it.
        """
        if traffic_light_id is None:
            return None
        if len(link_indices or ()) != 1:
            return None

        signal_read = self._sumo_context.signal_character(
            traffic_light_id, link_indices[0]
        )
        self._record_read_quality(signal_read.quality)
        signal = signal_read.value
        return self._map_signal_char_to_state_code(signal) if signal is not None else None

    def _refresh_traffic_light_context(self) -> None:
        """Refresh the traffic light context owned by this scenario context
preprocessor."""
        tls_id = str(self._traffic_light_id) if self._traffic_light_id is not None else None

        if tls_id is None:
            self._traffic_light_context = TrafficLightContext()
            self._pinned_traffic_light_context = TrafficLightContext()
            return

        pinned_context = getattr(self, "_pinned_traffic_light_context", TrafficLightContext())
        pinned_is_current_light = (
            pinned_context.tls_id == tls_id
            and pinned_context.intersection_number == self._intersection_number
            and bool(pinned_context.link_indices)
        )

        configured_link_indices = self._resolve_traffic_light_link_indices(self._traffic_light_id)
        if configured_link_indices:
            link_indices = configured_link_indices
        elif pinned_is_current_light:
            # Keep the exact TLS link selected for this traffic light.
            link_indices = pinned_context.link_indices
        else:
            link_indices = ()

        compact_state = self._traffic_light_state_code
        link_state = self._traffic_light_state_from_link_index(self._traffic_light_id, link_indices)
        if link_state is not None:
            compact_state = link_state
            self._traffic_light_state_code = compact_state

        context = TrafficLightContext(
            tls_id=tls_id,
            green_phase=self._traffic_light_green_phase,
            link_indices=link_indices,
            compact_state=compact_state,
            intersection_number=self._intersection_number,
        )
        self._traffic_light_context = context

        if link_indices:
            self._pinned_traffic_light_context = context
        else:
            self._pinned_traffic_light_context = TrafficLightContext()

    @property
    def current_context(self) -> ScenarioContext:
        """Return the latest preprocessing result without mutating SUMO."""

        ego_x_m, ego_y_m = self._ego_position_xy_m

        return ScenarioContext(
            reference_speed_mps=float(self._ref_speed),
            object_stations_m=tuple(self._station_environment),
            object_speeds_mps=tuple(float(speed) for speed in self._speed_environment),
            ego_station_m=float(self._ego_station_m),
            raw_ego_station_m=float(self._raw_ego_station_m),
            target_ego_station_m=float(self._target_ego_station_m),
            supervisor_ego_station_m=float(self._supervisor_ego_station_m),
            ego_lead_vehicle_present=bool(self._ego_lead_vehicle_present),
            ego_lead_distance_m=float(self._ego_lead_distance),
            ego_lead_speed_mps=float(self._ego_lead_speed_mps),
            ego_lead_vehicle_id=self._ego_lead_vehicle_id,
            pedestrian_parameters=(
                float(self._ped_station),
                float(self._ped_velocity),
                self._pedestrian_state,
            ),
            pedestrian_id=self._pedestrian_id,
            intersection_number=int(self._intersection_number),
            traffic_light_state_code=int(self._traffic_light_state_code),
            upcoming_intersection_type_code=int(self._upcoming_intersection_type_code),
            traffic_light_context=self._traffic_light_context,
            ego_position_xy_m=(float(ego_x_m), float(ego_y_m)),
            ego_road_id=self._ego_road_id,
            simulator_read_quality=self._simulator_read_quality,
            traffic_light_route_internal_edge=(
                self._traffic_light_route_internal_edge
            ),
            policy_object_stations_m=self._policy_object_stations_m,
            policy_object_speeds_mps=self._policy_object_speeds_mps,
            policy_object_ids=tuple(
                candidate.actor_id
                for candidate in self._policy_obstacle_candidates
            ),
            policy_object_kinds=tuple(
                candidate.kind.value
                for candidate in self._policy_obstacle_candidates
            ),
            policy_object_geometries=tuple(
                candidate.geometry.value
                for candidate in self._policy_obstacle_candidates
            ),
            policy_object_distances_m=tuple(
                float(candidate.distance_to_ego_m)
                for candidate in self._policy_obstacle_candidates
                if candidate.distance_to_ego_m is not None
            ),
            policy_object_constraint_clearances_m=(
                getattr(
                    self,
                    "_policy_object_constraint_clearances_m",
                    (),
                )
            ),
            supervisor_vehicle_ids=tuple(
                vehicle_id
                for vehicle_id in self._ranked_supervisor_vehicle_ids
                if vehicle_id not in (None, "-1")
            ),
            pedestrian_count=len(self._pedestrians),
            active_pedestrian_count=sum(
                pedestrian.state is PedestrianCrossingState.ON_CROSSING
                for pedestrian in self._pedestrians
            ),
            supervisor_vehicle_conflict=any(
                vehicle_id not in (None, "-1")
                for vehicle_id in self._ranked_supervisor_vehicle_ids
            ),
        )

    @property
    def traffic_light_context(self) -> TrafficLightContext:
        """Return the canonical signal context using a snake-case API."""

        return self._traffic_light_context

    @property
    def reference_speed_mps(self) -> float:
        """Return the reference speed mps exposed by this scenario context preprocessor."""
        return self._ref_speed

    @property
    def object_stations_m(self) -> list[float]:
        """Return the object stations m exposed by this scenario context preprocessor."""
        return list(self._station_environment)

    @property
    def object_speeds_mps(self) -> list[float]:
        """Return the object speeds mps exposed by this scenario context preprocessor."""
        return self._speed_environment

    @property
    def ego_station_m(self) -> float:
        """Return the ego station m exposed by this scenario context preprocessor."""
        return self._ego_station_m

    @property
    def raw_ego_station_m(self) -> float:
        """Return the raw ego station m exposed by this scenario context preprocessor."""
        return self._raw_ego_station_m

    @property
    def target_ego_station_m(self) -> float:
        """Return the configured signed signal-boundary station."""
        return self._target_ego_station_m

    @property
    def ego_lead_vehicle_present(self) -> bool:
        """Return the ego lead vehicle present exposed by this scenario context preprocessor."""
        return self._ego_lead_vehicle_present

    @property
    def ego_lead_distance_m(self) -> float:
        """Return the ego lead distance m exposed by this scenario context preprocessor."""
        return self._ego_lead_distance

    @property
    def ego_lead_speed_mps(self) -> float:
        """Return the current SUMO leader speed in metres per second."""

        return self._ego_lead_speed_mps

    @property
    def pedestrian_parameters(
        self,
    ) -> tuple[float, float, PedestrianCrossingState]:
        """Return the pedestrian parameters exposed by this scenario context preprocessor."""
        return self._ped_station, self._ped_velocity, self._pedestrian_state

    @property
    def intersection_number(self) -> int:
        """Return the intersection number exposed by this scenario context preprocessor."""
        return self._intersection_number

    @property
    def traffic_light_state_code(self) -> int:
        """Return the traffic light state code exposed by this scenario context preprocessor."""
        return self._traffic_light_state_code

    @property
    def upcoming_intersection_type_code(self) -> int:
        """Return the upcoming intersection type code exposed by this scenario context
        preprocessor.
        """
        return self._upcoming_intersection_type_code

    @property
    def traffic_light_id(self) -> str | None:
        """Return the traffic light id exposed by this scenario context preprocessor."""
        return self._traffic_light_id

    @property
    def traffic_light_green_phase(self) -> int | None:
        """Return the traffic light green phase exposed by this scenario context preprocessor."""
        return self._traffic_light_green_phase

    @property
    def traffic_light_link_indices(self) -> tuple[int, ...]:
        """Return the traffic light link indices exposed by this scenario context preprocessor."""
        return self._traffic_light_link_indices

    @property
    def traffic_light_route_internal_edge(self) -> str | None:
        """Return the traffic light route internal edge exposed by this scenario context
        preprocessor.
        """
        return self._traffic_light_route_internal_edge

    @property
    def road_speed_limit_mps(self) -> float:
        """Return the road speed limit mps exposed by this scenario context preprocessor."""
        return self._ref_speed

    @property
    def ego_position_xy_m(self) -> tuple[float, float]:
        """Return the ego position xy m exposed by this scenario context preprocessor."""
        return self._ego_position_xy_m

    @property
    def ego_road_id(self) -> str | int:
        """Return the ego road id exposed by this scenario context preprocessor."""
        return self._ego_road_id
