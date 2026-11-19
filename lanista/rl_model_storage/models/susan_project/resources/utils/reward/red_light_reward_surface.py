# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Pure, unit-explicit stages of Susan's red-light speed reward surface."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RedLightSpeedProfileInput:
    """Immutable geometry for one red-light speed-profile evaluation."""

    current_speed_mps: float
    distance_to_stopline_m: float
    road_speed_limit_mps: float
    epsilon: float
    preferred_stop_offset_m: float
    maximum_negative_reward: float
    brake_horizon_m: float
    reward_horizon_m: float
    brake_limit_speed_mps: float
    preferred_speed_mps: float
    lower_preferred_speed_mps: float


@dataclass(frozen=True, slots=True)
class RedLightSpeedProfileResult:
    """Numerical result before the controller publishes telemetry."""

    reward: float
    brake_limit_reward: float
    zero_speed_reward: float


@dataclass(frozen=True, slots=True)
class RedLightFarWaitPenaltyInput:
    """Inputs evaluated after the speed profile, preserving stage order."""

    far_distance_gate: float
    slow_speed_gate: float
    penalty_weight: float
    blocked_by_obstacle: bool


def calculate_red_light_speed_profile(
    inputs: RedLightSpeedProfileInput,
) -> RedLightSpeedProfileResult:
    """Evaluate the existing obstacle-compatible piecewise profile."""

    return _calculate_speed_profile_result(inputs)


def calculate_continuous_signal_speed_profile(
    inputs: RedLightSpeedProfileInput,
    stationary_plateau_width_m: float,
) -> RedLightSpeedProfileResult:
    """Evaluate the signal profile with continuous preferred/brake boundaries."""

    return _calculate_speed_profile_result(
        inputs,
        continuous_signal_boundary=True,
        stationary_plateau_width_m=max(0.0, stationary_plateau_width_m),
    )


def _calculate_speed_profile_result(
    inputs: RedLightSpeedProfileInput,
    *,
    continuous_signal_boundary: bool = False,
    stationary_plateau_width_m: float = 0.0,
) -> RedLightSpeedProfileResult:
    """Evaluate one speed profile while preserving the obstacle legacy mode."""

    distance_ratio = max(
        0.0,
        min(
            1.0,
            inputs.distance_to_stopline_m
            / max(inputs.reward_horizon_m, inputs.epsilon),
        ),
    )
    brake_limit_reward = inputs.maximum_negative_reward * (
        1.0 - distance_ratio
    )
    zero_speed_reward = _calculate_zero_speed_reward(inputs)
    if continuous_signal_boundary:
        profile_reward = _calculate_continuous_signal_segment(
            inputs,
            zero_speed_reward,
            stationary_plateau_width_m,
        )
    else:
        profile_reward = _calculate_profile_segment(
            inputs,
            brake_limit_reward,
            zero_speed_reward,
        )
    reward = max(
        inputs.maximum_negative_reward,
        min(1.0, profile_reward),
    )
    return RedLightSpeedProfileResult(
        reward=reward,
        brake_limit_reward=brake_limit_reward,
        zero_speed_reward=zero_speed_reward,
    )


def _calculate_continuous_signal_segment(
    inputs: RedLightSpeedProfileInput,
    zero_speed_reward: float,
    stationary_plateau_width_m: float,
) -> float:
    """Keep the exact QP target while removing sampled reward cliffs."""

    current = inputs.current_speed_mps
    lower_preferred = inputs.lower_preferred_speed_mps
    preferred = inputs.preferred_speed_mps
    brake_limit = inputs.brake_limit_speed_mps
    epsilon = inputs.epsilon

    stationary_plateau_limit_m = (
        inputs.preferred_stop_offset_m + stationary_plateau_width_m
    )
    if (
        current <= epsilon
        and inputs.distance_to_stopline_m <= stationary_plateau_limit_m
    ):
        return 1.0

    if preferred <= epsilon:
        if current <= epsilon:
            return zero_speed_reward
        creep_scale_mps = max(brake_limit, epsilon)
        creep_ratio = max(0.0, min(1.0, current / creep_scale_mps))
        return inputs.maximum_negative_reward * creep_ratio

    if current <= lower_preferred and lower_preferred > epsilon:
        ratio = max(0.0, min(1.0, current / lower_preferred))
        return zero_speed_reward + ((1.0 - zero_speed_reward) * ratio)
    if current <= preferred:
        return 1.0
    if brake_limit - preferred > epsilon and current < brake_limit:
        ratio = max(
            0.0,
            min(
                1.0,
                (current - preferred) / (brake_limit - preferred),
            ),
        )
        return 1.0 - ratio

    # The preferred-band width tends to zero near both curve endpoints.
    # Use the fixed road reference so different overspeeds remain distinct.
    overspeed_scale_mps = max(inputs.road_speed_limit_mps, epsilon)
    overspeed_ratio = max(
        0.0,
        min(
            1.0,
            (current - brake_limit) / overspeed_scale_mps,
        ),
    )
    return inputs.maximum_negative_reward * overspeed_ratio


def _calculate_zero_speed_reward(
    inputs: RedLightSpeedProfileInput,
) -> float:
    """Calculate the zero speed reward owned by this red light reward surface module."""
    if inputs.preferred_stop_offset_m <= inputs.epsilon:
        return float(inputs.distance_to_stopline_m <= inputs.epsilon)

    if inputs.distance_to_stopline_m <= inputs.preferred_stop_offset_m:
        stop_progress = (
            inputs.distance_to_stopline_m
            / inputs.preferred_stop_offset_m
        )
        return inputs.maximum_negative_reward + (
            (1.0 - inputs.maximum_negative_reward) * stop_progress
        )

    release_progress = min(
        1.0,
        (
            inputs.distance_to_stopline_m
            - inputs.preferred_stop_offset_m
        )
        / inputs.preferred_stop_offset_m,
    )
    return 1.0 - release_progress


def _calculate_profile_segment(
    inputs: RedLightSpeedProfileInput,
    brake_limit_reward: float,
    zero_speed_reward: float,
) -> float:
    """Calculate the profile segment owned by this red light reward surface module."""
    current = inputs.current_speed_mps
    lower_preferred = inputs.lower_preferred_speed_mps
    preferred = inputs.preferred_speed_mps
    brake_limit = inputs.brake_limit_speed_mps
    epsilon = inputs.epsilon

    if current <= lower_preferred and lower_preferred > epsilon:
        ratio = max(0.0, min(1.0, current / lower_preferred))
        return zero_speed_reward + (
            (1.0 - zero_speed_reward) * ratio
        )
    if current <= preferred and preferred > epsilon:
        return 1.0
    if brake_limit - preferred > epsilon and current < brake_limit:
        return _interpolate_to_brake_limit(
            current,
            preferred,
            brake_limit,
            brake_limit_reward,
            zero_speed_reward,
            epsilon,
        )
    if brake_limit <= epsilon:
        return (
            zero_speed_reward
            if current <= epsilon
            else inputs.maximum_negative_reward
        )
    return _calculate_overspeed_reward(
        inputs,
        brake_limit_reward,
    )


def _interpolate_to_brake_limit(
    current_speed_mps: float,
    preferred_speed_mps: float,
    brake_limit_speed_mps: float,
    brake_limit_reward: float,
    zero_speed_reward: float,
    epsilon: float,
) -> float:
    """Return the interpolate to brake limit value owned by this red light reward surface module."""
    ratio = max(
        0.0,
        min(
            1.0,
            (current_speed_mps - preferred_speed_mps)
            / (brake_limit_speed_mps - preferred_speed_mps),
        ),
    )
    shaped_ratio = math.sqrt(ratio)
    segment_start_reward = (
        1.0 if preferred_speed_mps > epsilon else zero_speed_reward
    )
    return segment_start_reward + (
        (brake_limit_reward - segment_start_reward) * shaped_ratio
    )


def _calculate_overspeed_reward(
    inputs: RedLightSpeedProfileInput,
    brake_limit_reward: float,
) -> float:
    """Calculate the overspeed reward owned by this red light reward surface module."""
    overspeed_ratio = max(
        0.0,
        min(
            1.0,
            (
                inputs.current_speed_mps
                - inputs.brake_limit_speed_mps
            )
            / max(inputs.road_speed_limit_mps, inputs.epsilon),
        ),
    )
    shaped_ratio = math.sqrt(overspeed_ratio)
    return brake_limit_reward + (
        (inputs.maximum_negative_reward - brake_limit_reward)
        * shaped_ratio
    )


def calculate_red_light_far_wait_penalty(
    inputs: RedLightFarWaitPenaltyInput,
) -> float:
    """Apply the existing obstacle gate after distance and speed gates."""

    if inputs.blocked_by_obstacle:
        return 0.0
    return (
        inputs.penalty_weight
        * inputs.far_distance_gate
        * inputs.slow_speed_gate
    )
