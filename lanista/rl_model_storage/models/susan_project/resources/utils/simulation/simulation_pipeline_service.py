# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Ordered Susan simulation-step pipeline and stage orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from lanista.domains.automotive.control.longitudinal_control_optimizer import (
    LongitudinalControlOptimizer,
    LongitudinalControlRequest,
)
from lanista.domains.automotive.control.saturation import (
    saturation_block as saturate_policy_action,
)
from lanista.reinforcengine.utils.simulation.intersection_metadata import (
    INTERSECTION_TYPE_TRAFFIC_LIGHT,
    INTERSECTION_TYPE_UNKNOWN,
    TRAFFIC_LIGHT_STATE_GREEN,
    TRAFFIC_LIGHT_STATE_NOT_APPLICABLE,
    TRAFFIC_LIGHT_STATE_YELLOW,
    TRAFFIC_LIGHT_STOP_PROFILE_STATE_CODES,
)

from ..observation.observation_service import (
    ObservationFormattingRequest,
    ObservationService,
)
from ..observation.vehicle_telemetry import VehicleTelemetry
from ..reward.reward_service import RewardCalculationRequest, RewardService
from .diagnostic_trace import SusanDiagnosticStepSnapshot
from .simulation_contracts import (
    PedestrianParameters,
    ScalarActionPort,
    ScenarioContext,
    SimulationMathPort,
    SimulationPipelineControllerPort,
    TelemetrySinkPort,
    is_ego_blocked_by_pedestrian,
)

_saturate_policy_action = cast(Callable[[float], float], saturate_policy_action)

@dataclass(frozen=True)
class _ControlStepState:
    """Represent runtime state for control step."""

    policy_action_mps2: float
    previous_speed_mps: float
    current_speed_mps: float
    speed_out_mps: float
    base_command_mps2: float
    applied_command_mps2: float
    distance_to_stopline_m: float | None
    actual_acceleration_mps2: float
    observation_applied_command_mps2: float
    observation_supervised_command_mps2: float
    reward_delta_mps2: float
    optimizer_delta_mps2: float
    optimizer_cost: float


@dataclass
class SimulationPipelineService:
    """Run Susan's named control pipeline for one simulator step."""

    controller: SimulationPipelineControllerPort
    observation_service: ObservationService
    reward_service: RewardService
    math: SimulationMathPort
    telemetry_sink: TelemetrySinkPort | None
    longitudinal_optimizer: LongitudinalControlOptimizer = field(
        default_factory=LongitudinalControlOptimizer,
    )
    latest_diagnostic_snapshot: SusanDiagnosticStepSnapshot | None = field(
        init=False,
        default=None,
    )
    _latest_control_step_state: _ControlStepState | None = field(
        init=False,
        default=None,
    )
    _previous_scenario_context: ScenarioContext | None = field(
        init=False,
        default=None,
    )
    _cleared_signal_internal_edge: str | None = field(
        init=False,
        default=None,
    )

    def reset_episode(self) -> None:
        """Clear all pipeline state that must not cross an episode boundary."""

        self.latest_diagnostic_snapshot = None
        self._latest_control_step_state = None
        self._previous_scenario_context = None
        self._cleared_signal_internal_edge = None

    @staticmethod
    def _resolve_current_ego_speed(
        current_sumo_speed_mps: float | None,
        fallback_speed_mps: float,
    ) -> float:
        """Return one finite forward speed without another simulator read."""

        for candidate in (current_sumo_speed_mps, fallback_speed_mps, 0.0):
            if candidate is None:
                continue
            try:
                speed_mps = float(candidate)
            except (TypeError, ValueError):
                continue
            if np.isfinite(speed_mps) and speed_mps >= 0.0:
                return speed_mps
        return 0.0

    def step(
        self,
        actions: Mapping[str, ScalarActionPort],
        step_counter: int,
    ) -> tuple[dict[str, np.ndarray], float]:
        """Advance the workflow owned by this simulation pipeline service by one step."""
        sumo_commands = self.controller._get_sumo_command_service()
        current_sumo_speed, sumo_acceleration = (
            sumo_commands.read_acceleration_since_last_controller_step()
        )
        current_ego_speed = self._resolve_current_ego_speed(
            current_sumo_speed,
            self.controller._speed_out,
        )
        self.controller._speed_out = current_ego_speed
        self.controller.ego_vehicle.prev_speed = current_ego_speed
        previous_context = self._remember_previous_scenario_codes()
        scenario = self.controller.update_scenario_context(current_ego_speed)
        red_light_overdrive = self.controller.is_running_into_restricted_area(
            previous_context,
            scenario,
        )
        self.controller.red_light_overdrive_this_step = red_light_overdrive
        signal_constraint_cleared = self._signal_constraint_cleared(
            previous_context,
            scenario,
            red_light_overdrive,
        )
        traffic_light_state_code, upcoming_intersection_type_code = (
            self._policy_visible_signal_state(
                scenario,
                signal_constraint_cleared=signal_constraint_cleared,
            )
        )
        policy_action = actions["EGO_agent_0"].item()
        next_observation, reward = self.calculate_simulink_environment_output(
            policy_action,
            scenario.reference_speed_mps,
            scenario.object_stations_m,
            scenario.object_speeds_mps,
            scenario.ego_station_m,
            scenario.pedestrian_parameters,
            traffic_light_state_code,
            upcoming_intersection_type_code,
            scenario.road_speed_limit_mps,
            step_counter,
            sumo_acceleration,
            red_light_overdrive,
            ego_lead_vehicle_present=scenario.ego_lead_vehicle_present,
            ego_lead_distance=scenario.ego_lead_distance_m,
            ego_lead_speed_mps=scenario.ego_lead_speed_mps,
            target_ego_station_m=scenario.target_ego_station_m,
            supervisor_ego_station_m=scenario.supervisor_ego_station_m,
            policy_object_stations_m=(
                scenario.effective_policy_object_stations_m
            ),
            policy_object_speeds_mps=(
                scenario.effective_policy_object_speeds_mps
            ),
            policy_object_kinds=scenario.policy_object_kinds,
            policy_object_geometries=scenario.policy_object_geometries,
            policy_object_distances_m=scenario.policy_object_distances_m,
            policy_object_constraint_clearances_m=(
                scenario.policy_object_constraint_clearances_m
            ),
            current_ego_speed_mps=current_ego_speed,
        )
        self._complete_step(
            scenario=scenario,
            remember_step_speed=sumo_commands.remember_step_speed,
            current_sumo_speed=current_sumo_speed,
            policy_action=policy_action,
            step_counter=step_counter,
            next_observation=next_observation,
            reward=reward,
            traffic_light_state_code=traffic_light_state_code,
            upcoming_intersection_type_code=upcoming_intersection_type_code,
            signal_constraint_cleared=signal_constraint_cleared,
            red_light_overdrive=red_light_overdrive,
        )
        return self.observation_service.extend_with_history(next_observation), reward

    def _remember_previous_scenario_codes(self) -> ScenarioContext | None:
        """Remember policy-visible codes without performing simulator reads."""

        previous_context = self._previous_scenario_context
        try:
            if previous_context is None:
                raise AttributeError
            prev_upcoming = int(previous_context.upcoming_intersection_type_code)
            prev_tl = int(previous_context.traffic_light_state_code)
        except (AttributeError, TypeError, ValueError):
            prev_upcoming = None
            prev_tl = None
        self.controller._prev_upcoming_intersection_type_code = prev_upcoming
        self.controller._prev_traffic_light_state_code = prev_tl
        return previous_context

    @staticmethod
    def _policy_visible_signal_state(
        scenario: ScenarioContext,
        *,
        signal_constraint_cleared: bool,
    ) -> tuple[int, int]:
        """Return signal codes after applying the existing crossing rule."""

        if signal_constraint_cleared:
            return TRAFFIC_LIGHT_STATE_NOT_APPLICABLE, INTERSECTION_TYPE_UNKNOWN
        return (
            scenario.traffic_light_state_code,
            scenario.upcoming_intersection_type_code,
        )

    def _complete_step(
        self,
        *,
        scenario: ScenarioContext,
        remember_step_speed: Callable[[float | None], None],
        current_sumo_speed: float | None,
        policy_action: float,
        step_counter: int,
        next_observation: dict[str, float],
        reward: float,
        traffic_light_state_code: int,
        upcoming_intersection_type_code: int,
        signal_constraint_cleared: bool,
        red_light_overdrive: bool,
    ) -> None:
        """Commit already-computed state and optional diagnostic telemetry."""

        self.controller.record_interaction_step(scenario)
        remember_step_speed(current_sumo_speed)
        controller_state = self.controller.controller_state
        if scenario.intersection_number > controller_state.prev_intersection_number:
            controller_state.prev_intersection_number = scenario.intersection_number
        self.controller.cumulative_reward_counter_over_episodes += reward
        self.observation_service.append(next_observation)
        self._previous_scenario_context = scenario
        self._capture_diagnostic_telemetry(
            scenario=scenario,
            current_sumo_speed=current_sumo_speed,
            policy_action=policy_action,
            step_counter=step_counter,
            next_observation=next_observation,
            traffic_light_state_code=traffic_light_state_code,
            upcoming_intersection_type_code=upcoming_intersection_type_code,
            signal_constraint_cleared=signal_constraint_cleared,
            red_light_overdrive=red_light_overdrive,
        )

    def _capture_diagnostic_telemetry(
        self,
        *,
        scenario: ScenarioContext,
        current_sumo_speed: float | None,
        policy_action: float,
        step_counter: int,
        next_observation: dict[str, float],
        traffic_light_state_code: int,
        upcoming_intersection_type_code: int,
        signal_constraint_cleared: bool,
        red_light_overdrive: bool,
    ) -> None:
        """Capture diagnostics from values already produced by this step."""

        if self.telemetry_sink is None:
            return
        observation_telemetry = self.observation_service.telemetry_observation(
            next_observation
        )
        reward_telemetry = self.reward_service.telemetry()
        self.telemetry_sink.append(
            VehicleTelemetry(
                step_counter=step_counter,
                observation=observation_telemetry,
                reward=reward_telemetry,
                simulator_read_quality=scenario.simulator_read_quality,
            )
        )
        control_state = self._latest_control_step_state
        if control_state is None:
            return
        self.latest_diagnostic_snapshot = self._diagnostic_snapshot(
            step_counter=step_counter,
            policy_action=policy_action,
            actual_speed_mps=current_sumo_speed,
            scenario=scenario,
            control_state=control_state,
            observation_values=observation_telemetry.to_telemetry_dict(),
            reward_values={
                **reward_telemetry.to_flat_series(),
                "control_supervised_cmd_accel_local_jitter": (
                    self.reward_service.state.supervised_cmd_accel_local_jitter
                ),
                "control_supervised_cmd_accel_trend_jitter": (
                    self.reward_service.state.supervised_cmd_accel_trend_jitter
                ),
                "control_supervised_cmd_accel_weighted_comfort_jitter": (
                    self.reward_service.state.supervised_cmd_accel_weighted_comfort_jitter
                ),
            },
            traffic_light_state_code=traffic_light_state_code,
            upcoming_intersection_type_code=upcoming_intersection_type_code,
            signal_constraint_cleared=signal_constraint_cleared,
            red_light_overdrive=red_light_overdrive,
        )

    def _signal_constraint_cleared(
        self,
        previous_scenario: ScenarioContext | None,
        scenario: ScenarioContext,
        red_light_overdrive: bool,
    ) -> bool:
        """Latch a legal crossing for the active configured traffic signal."""

        internal_edge = scenario.traffic_light_route_internal_edge
        signal_key = str(internal_edge) if internal_edge else None
        if self._cleared_signal_internal_edge != signal_key:
            self._cleared_signal_internal_edge = None

        legal_crossing = bool(
            not red_light_overdrive
            and signal_key
            and scenario.upcoming_intersection_type_code
            == INTERSECTION_TYPE_TRAFFIC_LIGHT
            and scenario.traffic_light_state_code
            in {TRAFFIC_LIGHT_STATE_GREEN, TRAFFIC_LIGHT_STATE_YELLOW}
            and self.controller.crossed_configured_signal_boundary(
                previous_scenario,
                scenario,
            )
        )
        if legal_crossing:
            self._cleared_signal_internal_edge = signal_key
        return bool(
            signal_key
            and self._cleared_signal_internal_edge == signal_key
        )

    def calculate_simulink_environment_output(
        self,
        action: float,
        reference_speed_mps: float,
        object_stations_m: Sequence[float],
        object_speeds_mps: Sequence[float],
        ego_station_m: float,
        pedestrian_parameters: PedestrianParameters,
        traffic_light_state_code: int,
        upcoming_intersection_type_code: int | None = None,
        road_speed_limit_mps: float | None = None,
        step_counter: int | None = None,
        sumo_acceleration: float | None = None,
        red_light_overdrive: bool = False,
        ego_lead_vehicle_present: bool = False,
        ego_lead_distance: float = 0.0,
        ego_lead_speed_mps: float = 0.0,
        target_ego_station_m: float | None = None,
        supervisor_ego_station_m: float | None = None,
        policy_object_stations_m: Sequence[float] | None = None,
        policy_object_speeds_mps: Sequence[float] | None = None,
        policy_object_kinds: Sequence[str] | None = None,
        policy_object_geometries: Sequence[str] | None = None,
        policy_object_distances_m: Sequence[float] | None = None,
        policy_object_constraint_clearances_m: Sequence[float] | None = None,
        current_ego_speed_mps: float | None = None,
    ) -> tuple[dict[str, float], float]:
        """Calculate the simulink environment output owned by this simulation pipeline service."""
        observation_object_stations_m = (
            object_stations_m if policy_object_stations_m is None else policy_object_stations_m
        )
        observation_object_speeds_mps = (
            object_speeds_mps
            if policy_object_speeds_mps is None
            else policy_object_speeds_mps
        )
        control_state = self._execute_control_command(
            action=action,
            reference_speed_mps=reference_speed_mps,
            object_stations_m=object_stations_m,
            object_speeds_mps=object_speeds_mps,
            ego_station_m=ego_station_m,
            pedestrian_parameters=pedestrian_parameters,
            sumo_acceleration=sumo_acceleration,
            target_ego_station_m=target_ego_station_m,
            supervisor_ego_station_m=supervisor_ego_station_m,
            current_ego_speed_mps=current_ego_speed_mps,
        )
        self._latest_control_step_state = control_state
        observation = self._build_observation(
            control_state=control_state,
            object_stations_m=observation_object_stations_m,
            object_speeds_mps=observation_object_speeds_mps,
            ego_station_m=ego_station_m,
            traffic_light_state_code=traffic_light_state_code,
            upcoming_intersection_type_code=upcoming_intersection_type_code,
            road_speed_limit_mps=road_speed_limit_mps,
            red_light_overdrive=red_light_overdrive,
            ego_lead_vehicle_present=ego_lead_vehicle_present,
            ego_lead_distance=ego_lead_distance,
            ego_lead_speed_mps=ego_lead_speed_mps,
        )
        reward = self._calculate_step_reward(
            control_state=control_state,
            reference_speed_mps=reference_speed_mps,
            object_stations_m=observation_object_stations_m,
            policy_object_speeds_mps=observation_object_speeds_mps,
            policy_object_kinds=policy_object_kinds,
            policy_object_geometries=policy_object_geometries,
            policy_object_distances_m=policy_object_distances_m,
            policy_object_constraint_clearances_m=(
                policy_object_constraint_clearances_m
            ),
            ego_station_m=ego_station_m,
            pedestrian_parameters=pedestrian_parameters,
            traffic_light_state_code=traffic_light_state_code,
            upcoming_intersection_type_code=upcoming_intersection_type_code,
            road_speed_limit_mps=road_speed_limit_mps,
            constraint_ego_station_m=(
                ego_station_m
                if supervisor_ego_station_m is None
                else supervisor_ego_station_m
            ),
            step_counter=step_counter,
            sumo_acceleration=sumo_acceleration,
            red_light_overdrive=red_light_overdrive,
            ego_lead_vehicle_present=ego_lead_vehicle_present,
            ego_lead_distance_m=ego_lead_distance,
            ego_lead_speed_mps=ego_lead_speed_mps,
        )
        return observation, reward

    def _execute_control_command(
        self,
        *,
        action: float,
        reference_speed_mps: float,
        object_stations_m: Sequence[float],
        object_speeds_mps: Sequence[float],
        ego_station_m: float,
        pedestrian_parameters: PedestrianParameters,
        sumo_acceleration: float | None,
        target_ego_station_m: float | None,
        supervisor_ego_station_m: float | None = None,
        current_ego_speed_mps: float | None = None,
    ) -> _ControlStepState:
        """Execute the control command owned by this simulation pipeline service."""
        action = _saturate_policy_action(action)
        self.controller.policy_action = float(action)
        previous_speed_mps = self.controller.ego_vehicle.prev_speed

        base_command, speed_out_mps, applied_command_mps2, _ = (
            self.controller.ego_vehicle.calculate(
                self.controller._delta,
                reference_speed_mps,
            )
        )
        self.controller._speed_out = speed_out_mps
        optimization_result = self.longitudinal_optimizer.optimize(
            LongitudinalControlRequest.from_runtime_values(
                ego_station_m=(
                    ego_station_m
                    if supervisor_ego_station_m is None
                    else supervisor_ego_station_m
                ),
                object_stations_m=object_stations_m,
                ego_speed_mps=speed_out_mps,
                object_speeds_mps=object_speeds_mps,
                previous_command_mps2=base_command,
                policy_command_mps2=action,
                pedestrian_parameters=pedestrian_parameters,
            )
        )
        self.controller._delta = optimization_result.delta_mps2

        self.controller.controller_state.applied_delta = (
            applied_command_mps2 - base_command
        )
        self.controller._get_sumo_command_service().set_ego_speed(speed_out_mps)

        stopline_station_m = (
            ego_station_m
            if target_ego_station_m is None
            else target_ego_station_m
        )
        distance_to_stopline_m = self.math._get_distance_to_stopline(
            stopline_station_m,
        )
        observation_command = (
            self.controller.controller_state.last_supervised_cmd_accel
            if sumo_acceleration is not None
            else applied_command_mps2
        )
        actual_acceleration_mps2 = self.math._calculate_actual_acceleration(
            previous_speed_mps,
            speed_out_mps,
            sumo_acceleration,
        )
        observed_speed_mps = self._resolve_current_ego_speed(
            current_ego_speed_mps,
            speed_out_mps,
        )
        return _ControlStepState(
            policy_action_mps2=action,
            previous_speed_mps=previous_speed_mps,
            current_speed_mps=observed_speed_mps,
            speed_out_mps=speed_out_mps,
            base_command_mps2=base_command,
            applied_command_mps2=applied_command_mps2,
            distance_to_stopline_m=distance_to_stopline_m,
            actual_acceleration_mps2=actual_acceleration_mps2,
            observation_applied_command_mps2=observation_command,
            observation_supervised_command_mps2=observation_command,
            reward_delta_mps2=self.controller.controller_state.applied_delta,
            optimizer_delta_mps2=optimization_result.delta_mps2,
            optimizer_cost=optimization_result.cost,
        )

    @staticmethod
    def _diagnostic_snapshot(
        *,
        step_counter: int,
        policy_action: float,
        actual_speed_mps: float | None,
        scenario: ScenarioContext,
        control_state: _ControlStepState,
        observation_values: Mapping[str, object],
        reward_values: Mapping[str, object],
        traffic_light_state_code: int,
        upcoming_intersection_type_code: int,
        signal_constraint_cleared: bool,
        red_light_overdrive: bool,
    ) -> SusanDiagnosticStepSnapshot:
        """Compose a diagnostic snapshot only from already-read step values."""

        pedestrian_station, pedestrian_speed, pedestrian_state = (
            scenario.pedestrian_parameters
        )
        signal_constraint_active = (
            upcoming_intersection_type_code == INTERSECTION_TYPE_TRAFFIC_LIGHT
            and traffic_light_state_code in TRAFFIC_LIGHT_STOP_PROFILE_STATE_CODES
        )
        link_indices = scenario.traffic_light_context.link_indices
        return SusanDiagnosticStepSnapshot(
            step_counter=step_counter,
            policy_action=policy_action,
            actual_speed_mps=actual_speed_mps,
            requested_speed_mps=control_state.speed_out_mps,
            reference_speed_mps=scenario.reference_speed_mps,
            road_speed_limit_mps=scenario.road_speed_limit_mps,
            previous_speed_mps=control_state.previous_speed_mps,
            base_command_mps2=control_state.base_command_mps2,
            applied_command_mps2=control_state.applied_command_mps2,
            supervised_command_mps2=(
                control_state.observation_supervised_command_mps2
            ),
            actual_acceleration_mps2=control_state.actual_acceleration_mps2,
            optimizer_delta_mps2=control_state.optimizer_delta_mps2,
            optimizer_correction_mps2=(
                control_state.base_command_mps2
                + control_state.optimizer_delta_mps2
                - policy_action
            ),
            optimizer_cost=control_state.optimizer_cost,
            distance_to_stopline_m=control_state.distance_to_stopline_m,
            ego_station_m=scenario.ego_station_m,
            raw_ego_station_m=scenario.raw_ego_station_m,
            target_ego_station_m=scenario.target_ego_station_m,
            supervisor_ego_station_m=scenario.supervisor_ego_station_m,
            ego_position_x_m=scenario.ego_position_xy_m[0],
            ego_position_y_m=scenario.ego_position_xy_m[1],
            ego_road_id=scenario.ego_road_id,
            intersection_number=scenario.intersection_number,
            traffic_light_state_code=traffic_light_state_code,
            traffic_light_tls_id=scenario.traffic_light_context.tls_id,
            traffic_light_link_index=(link_indices[0] if link_indices else None),
            traffic_signal_constraint_active=signal_constraint_active,
            traffic_signal_constraint_cleared=signal_constraint_cleared,
            red_light_overdrive=red_light_overdrive,
            ego_lead_vehicle_present=scenario.ego_lead_vehicle_present,
            ego_lead_vehicle_id=scenario.ego_lead_vehicle_id,
            ego_lead_distance_m=scenario.ego_lead_distance_m,
            ego_lead_speed_mps=scenario.ego_lead_speed_mps,
            supervisor_vehicle_conflict=scenario.supervisor_vehicle_conflict,
            policy_object_ids=scenario.policy_object_ids,
            policy_object_kinds=scenario.policy_object_kinds,
            policy_object_distances_m=scenario.policy_object_distances_m,
            policy_vehicle_slot_count=sum(
                kind == "vehicle" for kind in scenario.policy_object_kinds
            ),
            policy_pedestrian_slot_count=sum(
                kind == "pedestrian" for kind in scenario.policy_object_kinds
            ),
            active_pedestrian_count=scenario.active_pedestrian_count,
            pedestrian_constraint_active=is_ego_blocked_by_pedestrian(
                scenario.pedestrian_parameters
            ),
            pedestrian_id=scenario.pedestrian_id,
            pedestrian_station_m=pedestrian_station,
            pedestrian_speed_mps=pedestrian_speed,
            pedestrian_crossing_state=int(pedestrian_state),
            simulator_read_quality=scenario.simulator_read_quality.value,
            observation_values=dict(observation_values),
            reward_values=dict(reward_values),
        )

    def _calculate_step_reward(
        self,
        *,
        control_state: _ControlStepState,
        reference_speed_mps: float,
        object_stations_m: Sequence[float],
        policy_object_speeds_mps: Sequence[float] | None = None,
        policy_object_kinds: Sequence[str] | None = None,
        policy_object_geometries: Sequence[str] | None = None,
        policy_object_distances_m: Sequence[float] | None = None,
        policy_object_constraint_clearances_m: Sequence[float] | None = None,
        constraint_ego_station_m: float | None = None,
        ego_station_m: float,
        pedestrian_parameters: PedestrianParameters,
        traffic_light_state_code: int,
        upcoming_intersection_type_code: int | None,
        road_speed_limit_mps: float | None,
        step_counter: int | None,
        sumo_acceleration: float | None,
        red_light_overdrive: bool,
        ego_lead_vehicle_present: bool,
        ego_lead_distance_m: float,
        ego_lead_speed_mps: float = 0.0,
    ) -> float:
        """Calculate the step reward owned by this simulation pipeline service."""
        result = self.reward_service.calculate(
            RewardCalculationRequest(
                supervised_command_mps2=control_state.applied_command_mps2,
                previous_speed_mps=control_state.previous_speed_mps,
                current_speed_mps=control_state.current_speed_mps,
                applied_delta_mps2=control_state.reward_delta_mps2,
                step_counter=step_counter,
                policy_action_mps2=control_state.policy_action_mps2,
                reference_speed_mps=reference_speed_mps,
                actual_acceleration_override_mps2=sumo_acceleration,
                traffic_light_state_code=traffic_light_state_code,
                upcoming_intersection_type_code=(
                    upcoming_intersection_type_code
                ),
                constraint_ego_station_m=constraint_ego_station_m,
                ego_station_m=ego_station_m,
                distance_to_stopline_m=control_state.distance_to_stopline_m,
                road_speed_limit_mps=road_speed_limit_mps,
                red_light_overdrive=red_light_overdrive,
                object_stations_m=tuple(
                    float(value) for value in object_stations_m
                ),
                policy_object_distances_m=(
                    tuple(float(value) for value in policy_object_distances_m)
                    if policy_object_distances_m is not None
                    else None
                ),
                policy_object_speeds_mps=(
                    tuple(float(value) for value in policy_object_speeds_mps)
                    if policy_object_speeds_mps is not None
                    else None
                ),
                policy_object_kinds=(
                    tuple(str(value) for value in policy_object_kinds)
                    if policy_object_kinds is not None
                    else None
                ),
                policy_object_geometries=(
                    tuple(str(value) for value in policy_object_geometries)
                    if policy_object_geometries is not None
                    else None
                ),
                policy_object_constraint_clearances_m=(
                    tuple(
                        float(value)
                        for value in policy_object_constraint_clearances_m
                    )
                    if policy_object_constraint_clearances_m is not None
                    else None
                ),
                pedestrian_parameters=pedestrian_parameters,
                ego_lead_vehicle_present=ego_lead_vehicle_present,
                ego_lead_distance_m=ego_lead_distance_m,
                ego_lead_speed_mps=ego_lead_speed_mps,
            )
        )
        return float(result.reward)

    def _build_observation(
        self,
        *,
        control_state: _ControlStepState,
        object_stations_m: Sequence[float],
        object_speeds_mps: Sequence[float],
        ego_station_m: float,
        traffic_light_state_code: int,
        upcoming_intersection_type_code: int | None,
        road_speed_limit_mps: float | None,
        red_light_overdrive: bool,
        ego_lead_vehicle_present: bool,
        ego_lead_distance: float,
        ego_lead_speed_mps: float = 0.0,
    ) -> dict[str, float]:
        """Build the observation required by this simulation pipeline service."""
        observation_road_speed_limit = self.math._observation_road_speed_limit(
            road_speed_limit_mps,
            control_state.distance_to_stopline_m,
            traffic_light_state_code,
            upcoming_intersection_type_code,
            red_light_overdrive=red_light_overdrive,
        )

        result = self.observation_service.format(
            ObservationFormattingRequest(
                ego_station_m=ego_station_m,
                object_stations_m=tuple(
                    float(value) for value in object_stations_m
                ),
                ego_speed_mps=control_state.current_speed_mps,
                object_speeds_mps=tuple(
                    float(value) for value in object_speeds_mps
                ),
                applied_command_mps2=(
                    control_state.observation_applied_command_mps2
                ),
                supervised_command_mps2=(
                    control_state.observation_supervised_command_mps2
                ),
                actual_acceleration_mps2=(
                    control_state.actual_acceleration_mps2
                ),
                observation_road_speed_limit_mps=(
                    observation_road_speed_limit
                ),
                traffic_light_state_code=traffic_light_state_code,
                upcoming_intersection_type_code=(
                    upcoming_intersection_type_code
                ),
                distance_to_stopline_m=control_state.distance_to_stopline_m,
                reward_road_speed_limit_mps=road_speed_limit_mps,
                ego_lead_vehicle_present=ego_lead_vehicle_present,
                ego_lead_distance_m=ego_lead_distance,
                ego_lead_speed_mps=ego_lead_speed_mps,
            )
        )
        return dict(result.values)
