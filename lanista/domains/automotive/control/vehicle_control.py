# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Shared longitudinal vehicle-control workflow with model-owned observations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from lanista.domains.automotive.control.controller_reward import (
    ControllerRewardCalculator,
    ControllerRewardRequest,
    ControllerRewardResult,
)
from lanista.domains.automotive.control.ego import EGO
from lanista.domains.automotive.control.longitudinal_control_optimizer import (
    LongitudinalControlOptimizer,
    LongitudinalControlRequest,
)
from lanista.domains.automotive.control.saturation import saturation_block
from lanista.domains.automotive.control.scenario_context_contracts import (
    ScenarioContextPreprocessorPort,
    ScenarioContextSnapshot,
)

EGO_AGENT_ID = "EGO_agent_0"


class ScalarAction(Protocol):
    """Scalar action wrapper supplied by the framework adapters."""

    def item(self) -> float:
        """Return the item value provided by this scalar action."""
        ...


class EgoVehicleState(Protocol):
    """Mutable longitudinal EGO state retained across control steps."""

    prev_speed: float

    @property
    def AverageSpeed(self) -> float:
        """Return the average speed exposed by this ego vehicle state."""
        ...

    def calculate(
        self,
        delta_mps2: float,
        reference_speed_mps: float,
    ) -> tuple[float, float, float, float]:
        """Return the calculate value provided by this ego vehicle state."""
        ...


@dataclass(frozen=True, slots=True)
class ControllerObservation:
    """Typed values from which a model plugin builds its observation vector."""

    ego_station_m: float
    object_stations_m: tuple[float, float, float]
    ego_speed_mps: float
    object_speeds_mps: tuple[float, float, float]
    commanded_acceleration_mps2: float
    actual_acceleration_mps2: float
    traffic_light_state_code: int | None
    upcoming_intersection_type_code: int | None

    def base_features(self) -> list[float]:
        """Return the stable ten-feature Eve controller observation."""

        return [
            self.ego_station_m,
            *self.object_stations_m,
            self.ego_speed_mps,
            *self.object_speeds_mps,
            self.commanded_acceleration_mps2,
            self.actual_acceleration_mps2,
        ]


class LongitudinalVehicleController(ABC):
    """Own the shared controller, optimizer, reward, and metric lifecycle."""

    def __init__(
        self,
        *,
        scenario_context_preprocessor: ScenarioContextPreprocessorPort,
        longitudinal_optimizer: LongitudinalControlOptimizer | None = None,
        reward_calculator: ControllerRewardCalculator | None = None,
        ego_vehicle: EgoVehicleState | None = None,
    ) -> None:
        """Initialize the longitudinal vehicle controller and its owned state."""
        self.scenario_context_preprocessor = scenario_context_preprocessor
        self._longitudinal_optimizer = longitudinal_optimizer or LongitudinalControlOptimizer()
        self._reward_calculator = reward_calculator or ControllerRewardCalculator()
        self.ego_vehicle = ego_vehicle or EGO()
        self.cumulative_reward_counter_over_episodes = 0.0
        self._moving_window_size = 100
        self.speed_reward_window: deque[float] = deque(maxlen=self._moving_window_size)
        self._initialize_control_state()

    def _initialize_control_state(self) -> None:
        """Initialize the control state owned by this longitudinal vehicle controller."""
        self._speed_out = 0.0
        self._delta = 0.0
        self._applied_delta = 0.0
        self.pedestrian_id_counter = 0
        self.last_cmd_accel = 0.0
        self.sync_reward = 0.0
        self.normalized_sync_reward = 0.0
        self.accel_change_reward = 0.0
        self.normalized_accel_change_reward = 0.0
        self.avg_speed_reward = 0.0
        self.normalized_avg_speed_reward = 0.0
        self.delta_change_reward = 0.0
        self.normalized_delta_change_reward = 0.0
        self.step_reward = 0.0
        self.sync_penalty_sum = 0.0
        self.accel_penalty_sum = 0.0
        self.delta_penalty_sum = 0.0
        self.avg_sync_penalty = 0.0
        self.avg_accel_penalty = 0.0
        self.avg_delta_penalty = 0.0
        self._prev_intersection_number = 0
        self.moving_avg_speed_reward = 0.0

    def reset(self) -> None:
        """Reset episode-local controller state while retaining cumulative reward."""

        self.ego_vehicle = EGO()
        self.speed_reward_window.clear()
        self._initialize_control_state()

    def step(
        self,
        actions: Mapping[str, ScalarAction],
        step_counter: int,
    ) -> tuple[list[float], float]:
        """Run one ordered context, optimizer, reward, and observation step."""

        context = self.scenario_context_preprocessor.update(self._speed_out)
        action = float(actions[EGO_AGENT_ID].item())
        observation, reward = self._calculate_controller_output(
            action,
            context,
            step_counter,
        )
        if context.intersection_number > self._prev_intersection_number:
            self._prev_intersection_number = context.intersection_number
        self.cumulative_reward_counter_over_episodes += reward
        return observation, reward

    def _calculate_controller_output(
        self,
        action: float,
        context: ScenarioContextSnapshot,
        step_counter: int,
    ) -> tuple[list[float], float]:
        """Calculate the controller output owned by this longitudinal vehicle controller."""
        saturated_action = float(
            saturation_block(action)
        )
        previous_speed_mps = float(self.ego_vehicle.prev_speed)
        previous_command_mps2, self._speed_out, command_mps2, _ = self.ego_vehicle.calculate(
            self._delta,
            context.reference_speed_mps,
        )
        optimization = self._longitudinal_optimizer.optimize(
            LongitudinalControlRequest.from_runtime_values(
                ego_station_m=context.ego_station_m,
                object_stations_m=context.object_stations_m,
                ego_speed_mps=self._speed_out,
                object_speeds_mps=context.object_speeds_mps,
                previous_command_mps2=previous_command_mps2,
                policy_command_mps2=saturated_action,
                pedestrian_parameters=context.pedestrian_parameters,
            )
        )
        self._delta = optimization.delta_mps2
        self._applied_delta = command_mps2 - previous_command_mps2
        reward_result = self._calculate_reward(
            command_mps2=command_mps2,
            previous_speed_mps=previous_speed_mps,
            current_speed_mps=self._speed_out,
            applied_delta_mps2=self._applied_delta,
            step_counter=step_counter,
        )
        observation = self._format_observation(
            ControllerObservation(
                ego_station_m=context.ego_station_m,
                object_stations_m=context.object_stations_m,
                ego_speed_mps=self._speed_out,
                object_speeds_mps=context.object_speeds_mps,
                commanded_acceleration_mps2=command_mps2,
                actual_acceleration_mps2=(reward_result.actual_acceleration_mps2),
                traffic_light_state_code=context.traffic_light_state_code,
                upcoming_intersection_type_code=(context.upcoming_intersection_type_code),
            )
        )
        return observation, reward_result.reward

    def _calculate_reward(
        self,
        *,
        command_mps2: float,
        previous_speed_mps: float,
        current_speed_mps: float,
        applied_delta_mps2: float,
        step_counter: int,
    ) -> ControllerRewardResult:
        """Calculate the reward owned by this longitudinal vehicle controller."""
        result = self._reward_calculator.calculate(
            ControllerRewardRequest(
                commanded_acceleration_mps2=command_mps2,
                previous_commanded_acceleration_mps2=self.last_cmd_accel,
                previous_speed_mps=previous_speed_mps,
                current_speed_mps=current_speed_mps,
                average_speed_mps=float(self.ego_vehicle.AverageSpeed),
                applied_delta_mps2=applied_delta_mps2,
                step_counter=step_counter,
                speed_reward_window=tuple(self.speed_reward_window),
                synchronization_penalty_sum=self.sync_penalty_sum,
                acceleration_penalty_sum=self.accel_penalty_sum,
                delta_penalty_sum=self.delta_penalty_sum,
            )
        )
        self._apply_reward_result(result)
        self.last_cmd_accel = command_mps2
        return result

    def _apply_reward_result(self, result: ControllerRewardResult) -> None:
        """Apply the reward result transition owned by this longitudinal vehicle controller."""
        self.sync_reward = result.synchronization_reward
        self.normalized_sync_reward = result.normalized_synchronization_reward
        self.accel_change_reward = result.acceleration_change_reward
        self.normalized_accel_change_reward = result.normalized_acceleration_change_reward
        self.avg_speed_reward = result.average_speed_reward
        self.normalized_avg_speed_reward = result.normalized_average_speed_reward
        self.delta_change_reward = result.delta_change_reward
        self.normalized_delta_change_reward = result.normalized_delta_change_reward
        self.speed_reward_window.clear()
        self.speed_reward_window.extend(result.speed_reward_window)
        self.moving_avg_speed_reward = result.moving_average_speed_reward
        self.sync_penalty_sum = result.synchronization_penalty_sum
        self.accel_penalty_sum = result.acceleration_penalty_sum
        self.delta_penalty_sum = result.delta_penalty_sum
        self.avg_sync_penalty = result.average_synchronization_penalty
        self.avg_accel_penalty = result.average_acceleration_penalty
        self.avg_delta_penalty = result.average_delta_penalty
        self.step_reward = result.reward

    @abstractmethod
    def _format_observation(
        self,
        observation: ControllerObservation,
    ) -> list[float]:
        """Build model-owned features from the shared controller state."""
