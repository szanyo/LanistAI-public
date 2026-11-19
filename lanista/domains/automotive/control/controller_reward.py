# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Immutable reward requests and the shared longitudinal reward calculation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ControllerRewardConfiguration:
    """Weights, normalizers, and timing retained from the controller model."""

    controller_interval_s: float = 0.05
    synchronization_weight: float = 0.80
    acceleration_change_weight: float = 0.04
    average_speed_weight: float = 0.08
    delta_change_weight: float = 0.08
    synchronization_error_max_mps2: float = 3.0
    synchronization_exponent: float = 4.0
    acceleration_change_normalizer_mps2: float = 7.0
    average_speed_normalizer_mps: float = 8.3333
    delta_change_normalizer_mps2: float = 3.5
    speed_window_size: int = 100


@dataclass(frozen=True, slots=True)
class ControllerRewardRequest:
    """All instantaneous and accumulated inputs to one reward calculation."""

    commanded_acceleration_mps2: float
    previous_commanded_acceleration_mps2: float
    previous_speed_mps: float
    current_speed_mps: float
    average_speed_mps: float
    applied_delta_mps2: float
    step_counter: int
    speed_reward_window: tuple[float, ...]
    synchronization_penalty_sum: float
    acceleration_penalty_sum: float
    delta_penalty_sum: float


@dataclass(frozen=True, slots=True)
class ControllerRewardResult:
    """Reward value, telemetry components, and updated aggregates."""

    reward: float
    actual_acceleration_mps2: float
    synchronization_reward: float
    normalized_synchronization_reward: float
    acceleration_change_reward: float
    normalized_acceleration_change_reward: float
    average_speed_reward: float
    normalized_average_speed_reward: float
    delta_change_reward: float
    normalized_delta_change_reward: float
    speed_reward_window: tuple[float, ...]
    moving_average_speed_reward: float
    synchronization_penalty_sum: float
    acceleration_penalty_sum: float
    delta_penalty_sum: float
    average_synchronization_penalty: float
    average_acceleration_penalty: float
    average_delta_penalty: float


@dataclass(frozen=True, slots=True)
class _InstantaneousRewardComponents:
    """Carry the data required for instantaneous reward components."""

    actual_acceleration_mps2: float
    synchronization_reward: float
    normalized_synchronization_reward: float
    acceleration_change_reward: float
    normalized_acceleration_change_reward: float
    average_speed_reward: float
    normalized_average_speed_reward: float
    delta_change_reward: float
    normalized_delta_change_reward: float


@dataclass(frozen=True, slots=True)
class _RewardAggregates:
    """Carry the data required for reward aggregates."""

    speed_window: tuple[float, ...]
    moving_average_speed_reward: float
    synchronization_penalty_sum: float
    acceleration_penalty_sum: float
    delta_penalty_sum: float


class ControllerRewardCalculator:
    """Calculate the characterized four-component controller reward."""

    def __init__(
        self,
        configuration: ControllerRewardConfiguration | None = None,
    ) -> None:
        """Initialize the controller reward calculator and its owned state."""
        self._configuration = configuration or ControllerRewardConfiguration()

    def calculate(
        self,
        request: ControllerRewardRequest,
    ) -> ControllerRewardResult:
        """Return reward telemetry and updated history without mutating inputs."""

        components = self._instantaneous_components(request)
        aggregates = self._aggregates(request, components)
        effective_step_counter = max(1, request.step_counter)
        config = self._configuration
        reward = (
            config.synchronization_weight * components.normalized_synchronization_reward
            + config.acceleration_change_weight * components.normalized_acceleration_change_reward
            + config.average_speed_weight * aggregates.moving_average_speed_reward
            + config.delta_change_weight * components.normalized_delta_change_reward
        )
        return ControllerRewardResult(
            reward=reward,
            actual_acceleration_mps2=components.actual_acceleration_mps2,
            synchronization_reward=components.synchronization_reward,
            normalized_synchronization_reward=(components.normalized_synchronization_reward),
            acceleration_change_reward=(components.acceleration_change_reward),
            normalized_acceleration_change_reward=(
                components.normalized_acceleration_change_reward
            ),
            average_speed_reward=components.average_speed_reward,
            normalized_average_speed_reward=(components.normalized_average_speed_reward),
            delta_change_reward=components.delta_change_reward,
            normalized_delta_change_reward=(components.normalized_delta_change_reward),
            speed_reward_window=aggregates.speed_window,
            moving_average_speed_reward=(aggregates.moving_average_speed_reward),
            synchronization_penalty_sum=(aggregates.synchronization_penalty_sum),
            acceleration_penalty_sum=aggregates.acceleration_penalty_sum,
            delta_penalty_sum=aggregates.delta_penalty_sum,
            average_synchronization_penalty=(
                -(aggregates.synchronization_penalty_sum / effective_step_counter)
            ),
            average_acceleration_penalty=(
                -(aggregates.acceleration_penalty_sum / effective_step_counter)
            ),
            average_delta_penalty=(-(aggregates.delta_penalty_sum / effective_step_counter)),
        )

    def _instantaneous_components(
        self,
        request: ControllerRewardRequest,
    ) -> _InstantaneousRewardComponents:
        """Return the instantaneous components value owned by this controller reward calculator."""
        config = self._configuration
        actual_acceleration_mps2 = (
            request.current_speed_mps - request.previous_speed_mps
        ) / config.controller_interval_s
        synchronization_error = abs(actual_acceleration_mps2 - request.commanded_acceleration_mps2)
        synchronization_reward = -synchronization_error
        normalized_synchronization_reward = float(
            -np.expm1(config.synchronization_exponent * synchronization_error)
            / np.expm1(config.synchronization_exponent * config.synchronization_error_max_mps2)
        )
        acceleration_change_reward = -abs(
            request.commanded_acceleration_mps2 - request.previous_commanded_acceleration_mps2
        )
        normalized_acceleration_change_reward = (
            acceleration_change_reward / config.acceleration_change_normalizer_mps2
        )
        average_speed_reward = request.average_speed_mps
        normalized_average_speed_reward = average_speed_reward / config.average_speed_normalizer_mps
        delta_change_reward = -abs(request.applied_delta_mps2)
        normalized_delta_change_reward = delta_change_reward / config.delta_change_normalizer_mps2
        return _InstantaneousRewardComponents(
            actual_acceleration_mps2=actual_acceleration_mps2,
            synchronization_reward=synchronization_reward,
            normalized_synchronization_reward=(normalized_synchronization_reward),
            acceleration_change_reward=acceleration_change_reward,
            normalized_acceleration_change_reward=(normalized_acceleration_change_reward),
            average_speed_reward=average_speed_reward,
            normalized_average_speed_reward=(normalized_average_speed_reward),
            delta_change_reward=delta_change_reward,
            normalized_delta_change_reward=normalized_delta_change_reward,
        )

    def _aggregates(
        self,
        request: ControllerRewardRequest,
        components: _InstantaneousRewardComponents,
    ) -> _RewardAggregates:
        """Return the aggregates value owned by this controller reward calculator."""
        config = self._configuration
        speed_window = (
            *request.speed_reward_window,
            components.normalized_average_speed_reward,
        )[-config.speed_window_size :]
        moving_average_speed_reward = sum(speed_window) / len(speed_window)
        synchronization_penalty_sum = request.synchronization_penalty_sum + abs(
            components.normalized_synchronization_reward
        )
        acceleration_penalty_sum = request.acceleration_penalty_sum + abs(
            components.normalized_acceleration_change_reward
        )
        delta_penalty_sum = request.delta_penalty_sum + abs(
            components.normalized_delta_change_reward
        )
        return _RewardAggregates(
            speed_window=speed_window,
            moving_average_speed_reward=moving_average_speed_reward,
            synchronization_penalty_sum=synchronization_penalty_sum,
            acceleration_penalty_sum=acceleration_penalty_sum,
            delta_penalty_sum=delta_penalty_sum,
        )
