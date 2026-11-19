# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Numerical control and reward-surface calculations for one simulation step."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Protocol, cast, overload

from lanista.domains.automotive.control.longitudinal_control_optimizer import (
    PEDESTRIAN_AVOIDANCE_RADIUS_M,
    VEHICLE_CONTROL_HORIZON_S,
)
from lanista.reinforcengine.utils.simulation.intersection_metadata import (
    INTERSECTION_TYPE_TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_PROHIBITED_CROSSING_STATE_CODES,
    TRAFFIC_LIGHT_STATE_GREEN,
    TRAFFIC_LIGHT_STOP_PROFILE_STATE_CODES,
)

from ..observation.observation_config import (
    GREEN_LIGHT_IDLE_PENALTY_WEIGHT,
    GREEN_LIGHT_IDLE_SLOW_SPEED_SCALE_MPS,
    LEAD_HEADWAY_SECONDS,
    RED_LIGHT_FAR_WAIT_DISTANCE_WIDTH_METERS,
    RED_LIGHT_FAR_WAIT_PENALTY_WEIGHT,
    RED_LIGHT_FAR_WAIT_SLOW_SPEED_SCALE_MPS,
    RED_LIGHT_MAX_NEGATIVE_REWARD,
    RED_LIGHT_PREFERRED_SPEED_PLATEAU_WIDTH_METERS,
    RED_LIGHT_RELEASE_CROSSING_REWARD_WEIGHT,
    REWARD_WEIGHT_ACCEL,
    REWARD_WEIGHT_DELTA,
    REWARD_WEIGHT_SPEED,
    REWARD_WEIGHT_SYNC,
    SIGNAL_STOP_MOTION_ENTRY_SPEED_MPS,
    SIGNAL_STOP_MOTION_EXIT_SPEED_MPS,
    SIGNAL_STOP_POSITIVE_ACCEL_PENALTY_SCALE_MPS2,
    SIGNAL_STOP_SPEED_TOLERANCE_MPS,
    SPEED_REWARD_JITTER_COMFORT_WINDOW,
    SPEED_REWARD_JITTER_DISCOUNT_DEADBAND,
    SPEED_REWARD_JITTER_DISCOUNT_FLOOR,
    SPEED_REWARD_JITTER_DISCOUNT_SCALE,
    SPEED_REWARD_JITTER_DISCOUNT_STRENGTH,
    SPEED_REWARD_JITTER_TREND_WINDOW,
    SUPERVISED_CMD_ACCEL_DELTA_CHANGE_NORMALIZER,
    SUPERVISED_CMD_ACCEL_DELTA_NORMALIZER,
    SUPERVISED_CMD_ACCEL_SIGN_CHANGE_DEADBAND_MPS2,
    SUPERVISED_CMD_ACCEL_SIGN_CHANGE_PENALTY_MULTIPLIER,
)
from ..observation.vehicle_telemetry import VehicleControllerTelemetryState
from ..reward.red_light_reward_surface import (
    RedLightFarWaitPenaltyInput,
    RedLightSpeedProfileInput,
    calculate_continuous_signal_speed_profile,
    calculate_red_light_far_wait_penalty,
    calculate_red_light_speed_profile,
)
from ..reward.reward_telemetry import (
    SYNC_ERROR_DEADBAND_MPS2,
    RewardConfigTelemetry,
    RewardStateTelemetry,
)
from ..simulation.simulation_contracts import (
    ScenarioContextPreprocessorPort,
)


class ControllerMathStatePort(Protocol):
    """Explicit state used by Susan's numerical reward calculations."""

    reward_config: RewardConfigTelemetry
    reward_state: RewardStateTelemetry
    controller_state: VehicleControllerTelemetryState
    _current_supervised_cmd_accel_delta: float
    supervised_cmd_accel_delta: float
    abs_supervised_cmd_accel_delta: float
    supervised_cmd_accel_delta_change: float
    abs_supervised_cmd_accel_delta_change: float
    supervised_cmd_accel_local_jitter: float
    supervised_cmd_accel_local_jitter_sign_change: float
    supervised_cmd_accel_local_effective_jitter: float
    supervised_cmd_accel_trend_jitter: float
    supervised_cmd_accel_trend_jitter_sign_change: float
    supervised_cmd_accel_trend_effective_jitter: float
    supervised_cmd_accel_comfort_jitter: float
    supervised_cmd_accel_weighted_comfort_jitter: float
    supervised_cmd_accel_comfort_jitter_sign_change: float
    supervised_cmd_accel_comfort_effective_jitter: float
    supervised_cmd_accel_discount_jitter: float
    normalized_supervised_cmd_accel_discount_jitter: float
    pre_jitter_discount_speed_reward: float
    speed_reward_jitter_discount_applied: float
    speed_reward_jitter_discount: float
    supervised_cmd_accel_history: deque[float]
    supervised_cmd_accel_delta_history: deque[float]

    @property
    def scenario_context_preprocessor(
        self,
    ) -> ScenarioContextPreprocessorPort | None:
        """Return the scenario context preprocessor exposed by this controller math state port."""
        ...


@dataclass
class ControllerMathService:
    """Coordinate controller math operations."""

    controller: ControllerMathStatePort

    @staticmethod
    def _get_reward_weights() -> dict[str, float]:
        """Return the reward weights exposed by this controller math service."""
        return {
            "sync": REWARD_WEIGHT_SYNC,
            "speed": REWARD_WEIGHT_SPEED,
            "delta": REWARD_WEIGHT_DELTA,
            "accel": REWARD_WEIGHT_ACCEL,
        }

    @staticmethod
    def _calculate_actual_acceleration(
        prev_speed: float,
        current_speed: float,
        actual_acceleration_override: float | None = None,
    ) -> float:
        """Calculate the actual acceleration owned by this controller math service."""
        ts = 0.05
        internal_acceleration = (current_speed - prev_speed) / ts
        if actual_acceleration_override is not None:
            return float(actual_acceleration_override)
        return internal_acceleration

    def _reward_speed_limit(
        self,
        road_speed_limit: float | None = None,
    ) -> float:
        """Return the reward speed limit value owned by this controller math service."""
        limit = self._safe_float(road_speed_limit, None)
        if limit is None or limit <= 0:
            limit = self.controller.reward_config.speed_reward_v_max
        return max(1e-6, limit)

    def _red_light_brake_horizon(
        self,
        road_speed_limit: float | None = None,
    ) -> float:
        """Return the red light brake horizon value owned by this controller math service."""
        speed_limit = self._reward_speed_limit(road_speed_limit)
        fallback_horizon = (
            speed_limit * speed_limit
        ) / (2.0 * self.controller.reward_config.red_light_a_comfort)
        horizon = self._safe_float(
            self.controller.reward_config.red_light_brake_horizon_distance,
            fallback_horizon,
        )
        if horizon is None or horizon <= 0.0:
            horizon = fallback_horizon
        return max(1e-6, horizon)

    def _red_light_reward_horizon(
        self,
        road_speed_limit: float | None = None,
        preferred_stop_offset_m: float | None = None,
    ) -> float:
        """Return the red light reward horizon value owned by this controller math service."""
        stop_offset = self._safe_float(
            preferred_stop_offset_m,
            0.0,
        )
        return (
            self._red_light_brake_horizon(road_speed_limit)
            + max(0.0, stop_offset)
        )

    def _signal_reward_curve_offset_m(
        self,
        road_speed_limit: float | None = None,
    ) -> float:
        """Return the offset making the signal curve tangent to the nominal QP."""

        speed_limit = self._reward_speed_limit(road_speed_limit)
        brake_horizon = self._red_light_brake_horizon(speed_limit)
        qp_horizon = max(0.0, float(VEHICLE_CONTROL_HORIZON_S))
        return (
            (speed_limit * qp_horizon) ** 2
            / (4.0 * max(brake_horizon, 1e-6))
        )

    def _signal_reward_curve_distance(
        self,
        distance_to_stopline: float,
        road_speed_limit: float | None = None,
    ) -> float:
        """Map true signal distance onto the QP-tangent reward curve."""

        return max(
            0.0,
            float(distance_to_stopline)
            - self._signal_reward_curve_offset_m(road_speed_limit),
        )

    def _signal_reward_horizon(
        self,
        road_speed_limit: float | None = None,
    ) -> float:
        """Return the signal horizon including its QP-tangent curve offset."""

        return (
            self._red_light_brake_horizon(road_speed_limit)
            + self._signal_reward_curve_offset_m(road_speed_limit)
        )

    def _observation_road_speed_limit(
        self,
        road_speed_limit: float | None,
        distance_to_stopline: float | None,
        traffic_light_state_code: int,
        upcoming_intersection_type_code: int | None = None,
        red_light_overdrive: bool = False,
    ) -> float:
        """Return the observation road speed limit value owned by this controller math service."""
        speed_limit = self._reward_speed_limit(road_speed_limit)
        uses_stop_profile = (
            self._uses_stop_profile(
                traffic_light_state_code,
                upcoming_intersection_type_code,
            )
            or bool(red_light_overdrive)
        )
        distance = self._safe_float(distance_to_stopline, None)
        if red_light_overdrive:
            distance = 0.0
        profile_active = (
            distance is not None
            and uses_stop_profile
            and distance <= self._signal_reward_horizon(speed_limit)
        )
        if not profile_active:
            return speed_limit
        assert distance is not None
        curve_distance = self._signal_reward_curve_distance(distance, speed_limit)
        safe_speed = self._red_light_safe_speed(distance, speed_limit)
        red_factor = self._red_light_factor(
            curve_distance,
            speed_limit,
            red_light_overdrive,
        )
        return ((1.0 - red_factor) * speed_limit) + (red_factor * safe_speed)

    def _calculate_sync_component(
        self,
        actual_acceleration: float,
        sync_cmd_accel: float,
        current_speed: float,
        road_speed_limit: float | None = None,
    ) -> float:
        """Calculate the sync component owned by this controller math service."""
        sync_error_max = 3.0
        sync_deadband = SYNC_ERROR_DEADBAND_MPS2
        sync_error = abs(float(sync_cmd_accel) - float(actual_acceleration))
        effective_sync_error = max(0.0, sync_error - sync_deadband)
        speed_factor = max(0.0, min(1.0, self._safe_float(current_speed, 0.0) / self._reward_speed_limit(road_speed_limit)))
        sync_state = self.controller.reward_state.sync
        sync_state.command = float(sync_cmd_accel)
        sync_state.actual_acceleration = float(actual_acceleration)
        sync_state.absolute_error = sync_error
        sync_state.effective_error = effective_sync_error
        sync_state.speed_factor = speed_factor

        if sync_error > sync_deadband:
            sync_state.raw = -effective_sync_error
            sync_state.direction_penalty = 0.0
            sync_state.magnitude_penalty = min(
                effective_sync_error / sync_error_max,
                1.0,
            )
            base_normalized_sync_reward = -sync_state.magnitude_penalty * speed_factor
        else:
            sync_state.raw = 0.0
            sync_state.direction_penalty = 0.0
            sync_state.magnitude_penalty = 0.0
            base_normalized_sync_reward = 0.0

        sync_state.normalized = base_normalized_sync_reward
        return sync_state.normalized

    def _calculate_accel_component(
        self,
        cmd_accel: float,
        prev_cmd_accel: float,
        current_speed_mps: float,
    ) -> float:
        # cmd_accel is the generator-limited command applied by EGO.calculate()
        # during this controller interval. The QP proposal for the next interval
        # must never enter this reward history.
        """Calculate the accel component owned by this controller math service."""
        cmd_accel_delta = float(cmd_accel) - float(prev_cmd_accel)
        self.controller._current_supervised_cmd_accel_delta = cmd_accel_delta
        self.controller.supervised_cmd_accel_delta = cmd_accel_delta
        self.controller.abs_supervised_cmd_accel_delta = abs(cmd_accel_delta)
        accel_state = self.controller.reward_state.accel
        base_cost = abs(cmd_accel_delta)
        # Keep accel as the independent first-difference cost. Sign-change
        # scaling and telemetry belong exclusively to the three jitter windows.
        accel_state.raw = -base_cost
        accel_state.normalized = (
            accel_state.raw / SUPERVISED_CMD_ACCEL_DELTA_NORMALIZER
        )
        return accel_state.normalized

    @overload
    @staticmethod
    def _safe_float(value: object, default: float) -> float:
        """Return float converted to a safe value for this controller math service."""
        ...

    @overload
    @staticmethod
    def _safe_float(value: object, default: None = None) -> float | None:
        """Return float converted to a safe value for this controller math service."""
        ...

    @staticmethod
    def _safe_float(
        value: object,
        default: float | None = None,
    ) -> float | None:
        """Return the safe float value owned by this controller math service."""
        try:
            result = float(cast(float | int | str, value))
        except (TypeError, ValueError):
            return default
        if not math.isfinite(result):
            return default
        return result

    @staticmethod
    def _softplus(value: float, beta: float = 5.0) -> float:
        """Return the softplus value owned by this controller math service."""
        z = float(beta) * float(value)
        if z > 50.0:
            return float(value)
        if z < -50.0:
            return 0.0
        return math.log1p(math.exp(z)) / float(beta)

    @classmethod
    def _soft_relu(
        cls,
        value: float,
        beta: float = 5.0,
    ) -> float:
        """Return the soft relu value owned by this controller math service."""
        zero = math.log1p(math.exp(0.0)) / float(beta)
        return max(0.0, cls._softplus(value, beta) - zero)

    @staticmethod
    def _sigmoid(value: float) -> float:
        """Return the sigmoid value owned by this controller math service."""
        value = max(-60.0, min(60.0, float(value)))
        return 1.0 / (1.0 + math.exp(-value))

    def _calculate_normal_speed_reward(
        self,
        current_speed: float,
        road_speed_limit: float | None = None,
    ) -> float:
        """Calculate the normal speed reward owned by this controller math service."""
        current = max(0.0, self._safe_float(current_speed, 0.0))
        speed_limit = self._reward_speed_limit(road_speed_limit)
        ratio = current / max(speed_limit, 1e-6)

        if ratio <= 1.0:
            low_speed_floor = -0.5
            reward = low_speed_floor + ((1.0 - low_speed_floor) * ratio)
            return max(low_speed_floor, min(1.0, reward))

        return max(-1.0, 1.0 - 2.0 * (ratio - 1.0))

    def _calculate_green_light_idle_penalty(
        self,
        current_speed: float,
        traffic_light_state_code: int | None,
        upcoming_intersection_type_code: int | None,
        blocked_by_obstacle: bool = False,
    ) -> float:
        """Calculate the green light idle penalty owned by this controller math service."""
        is_green_traffic_light = (
            self._as_int_code(traffic_light_state_code) == TRAFFIC_LIGHT_STATE_GREEN
            and self._as_int_code(upcoming_intersection_type_code) == INTERSECTION_TYPE_TRAFFIC_LIGHT
        )
        if not is_green_traffic_light:
            return 0.0

        if blocked_by_obstacle:
            return 0.0

        current = max(0.0, self._safe_float(current_speed, 0.0))
        speed_scale = max(GREEN_LIGHT_IDLE_SLOW_SPEED_SCALE_MPS, 1e-6)
        slow_gate = math.exp(-(current / speed_scale))
        return GREEN_LIGHT_IDLE_PENALTY_WEIGHT * slow_gate

    def _apply_speed_reward_jitter_discount(self, speed_reward: float) -> float:
        """Apply the speed reward jitter discount transition owned by this controller math service.
        """
        reward = self._safe_float(speed_reward, 0.0)
        self.controller.pre_jitter_discount_speed_reward = reward
        self.controller.speed_reward_jitter_discount_applied = 0.0
        current_cmd_delta = self.controller._current_supervised_cmd_accel_delta
        trend_penalty_jitter = self._calculate_supervised_cmd_accel_trend_jitter(
            current_cmd_delta
        )
        comfort_penalty_jitter = self._calculate_supervised_cmd_accel_comfort_jitter(
            current_cmd_delta
        )
        prev_cmd_delta = self.controller.controller_state.last_supervised_cmd_accel_delta
        local_jitter = abs(float(current_cmd_delta) - float(prev_cmd_delta))
        self.controller.supervised_cmd_accel_local_jitter = local_jitter
        jitter = max(trend_penalty_jitter, comfort_penalty_jitter)
        self.controller.supervised_cmd_accel_discount_jitter = jitter
        self.controller.normalized_supervised_cmd_accel_discount_jitter = -(
            jitter / max(float(SUPERVISED_CMD_ACCEL_DELTA_CHANGE_NORMALIZER), 1e-6)
        )

        jitter = max(0.0, jitter - float(SPEED_REWARD_JITTER_DISCOUNT_DEADBAND))
        scale = max(float(SPEED_REWARD_JITTER_DISCOUNT_SCALE), 1e-6)
        jitter_cost = jitter / (jitter + scale)
        discount = 1.0 - (float(SPEED_REWARD_JITTER_DISCOUNT_STRENGTH) * jitter_cost)
        floor = max(0.0, min(1.0, float(SPEED_REWARD_JITTER_DISCOUNT_FLOOR)))
        self.controller.speed_reward_jitter_discount = max(floor, min(1.0, discount))
        if reward >= 0.0:
            self.controller.speed_reward_jitter_discount_applied = 1.0
            return reward * self.controller.speed_reward_jitter_discount

        return reward

    def _calculate_supervised_cmd_accel_trend_jitter(
        self,
        current_cmd_delta: float | None = None,
    ) -> float:
        """Calculate the supervised cmd accel trend jitter owned by this controller math service."""
        jitter, sign_changed = self._analyze_supervised_cmd_accel_window_jitter(
            current_cmd_delta,
            SPEED_REWARD_JITTER_TREND_WINDOW,
        )
        self.controller.supervised_cmd_accel_trend_jitter = jitter
        effective_jitter = self._scale_jitter_for_sign_change(jitter, sign_changed)
        self.controller.supervised_cmd_accel_trend_jitter_sign_change = float(
            sign_changed
        )
        self.controller.supervised_cmd_accel_trend_effective_jitter = (
            effective_jitter
        )
        return effective_jitter

    def _calculate_supervised_cmd_accel_comfort_jitter(
        self,
        current_cmd_delta: float | None = None,
    ) -> float:
        """Calculate the supervised cmd accel comfort jitter owned by this controller math service.
        """
        comfort_window = max(1, int(SPEED_REWARD_JITTER_COMFORT_WINDOW))
        trend_window = max(1, int(SPEED_REWARD_JITTER_TREND_WINDOW))
        raw_jitter, sign_changed = self._analyze_supervised_cmd_accel_window_jitter(
            current_cmd_delta,
            comfort_window,
        )
        self.controller.supervised_cmd_accel_comfort_jitter = raw_jitter
        scale = min(1.0, float(trend_window) / max(float(comfort_window), 1.0))
        weighted_jitter = raw_jitter * scale
        self.controller.supervised_cmd_accel_weighted_comfort_jitter = weighted_jitter
        effective_jitter = self._scale_jitter_for_sign_change(
            weighted_jitter,
            sign_changed,
        )
        self.controller.supervised_cmd_accel_comfort_jitter_sign_change = float(
            sign_changed
        )
        self.controller.supervised_cmd_accel_comfort_effective_jitter = (
            effective_jitter
        )
        return effective_jitter

    def _calculate_supervised_cmd_accel_window_jitter(
        self,
        current_cmd_delta: float | None = None,
        window: int = 1,
    ) -> float:
        """Calculate the supervised cmd accel window jitter owned by this controller math service.
        """
        jitter, _sign_changed = self._analyze_supervised_cmd_accel_window_jitter(
            current_cmd_delta,
            window,
        )
        return jitter

    def _analyze_supervised_cmd_accel_window_jitter(
        self,
        current_cmd_delta: float | None = None,
        window: int = 1,
    ) -> tuple[float, bool]:
        """Return raw jitter and whether its command window crosses zero."""
        if current_cmd_delta is None:
            current_cmd_delta = self.controller._current_supervised_cmd_accel_delta

        history_values = [
            float(value)
            for value in self.controller.supervised_cmd_accel_delta_history
        ]
        if not history_values:
            history_values = [
                float(
                    self.controller.controller_state.last_supervised_cmd_accel_delta
                )
            ]

        # History contains completed intervals only. Add the current first
        # difference exactly once for reward evaluation; _update_reward_memory()
        # commits it after every reward component has read the same snapshot.
        values = history_values + [float(current_cmd_delta)]
        values = values[-(max(1, int(window)) + 1):]
        if len(values) < 2:
            return 0.0, False

        changes = [
            abs(values[index] - values[index - 1])
            for index in range(1, len(values))
        ]
        window_jitter = sum(changes) / max(len(changes), 1)
        command_values = [
            float(value)
            for value in self.controller.supervised_cmd_accel_history
        ]
        if not command_values:
            command_values = [
                float(self.controller.controller_state.last_supervised_cmd_accel)
            ]
        current_cmd_accel = (
            float(self.controller.controller_state.last_supervised_cmd_accel)
            + float(current_cmd_delta)
        )
        command_values.append(current_cmd_accel)
        command_values = command_values[-(max(1, int(window)) + 1):]
        return window_jitter, self._values_have_sign_change(command_values)

    @staticmethod
    def _values_have_sign_change(values: list[float] | tuple[float, ...]) -> bool:
        """Return whether one window contains propulsion and braking commands."""
        deadband = max(
            0.0,
            float(SUPERVISED_CMD_ACCEL_SIGN_CHANGE_DEADBAND_MPS2),
        )
        has_positive = any(float(value) > deadband for value in values)
        has_negative = any(float(value) < -deadband for value in values)
        return has_positive and has_negative

    @staticmethod
    def _scale_jitter_for_sign_change(jitter: float, sign_changed: bool) -> float:
        """Scale only genuinely reversing jitter windows."""
        if not sign_changed:
            return float(jitter)
        multiplier = max(
            1.0,
            float(SUPERVISED_CMD_ACCEL_SIGN_CHANGE_PENALTY_MULTIPLIER),
        )
        return float(jitter) * multiplier

    def _red_light_target_speed(
        self,
        distance_to_stopline: float,
        road_speed_limit: float | None = None,
    ) -> float:
        """Return the red light target speed value owned by this controller math service."""
        return self._brake_curve_speed(
            self._signal_reward_curve_distance(
                distance_to_stopline,
                road_speed_limit,
            ),
            road_speed_limit,
        )

    def _red_light_safe_speed(
        self,
        distance_to_stopline: float,
        road_speed_limit: float | None = None,
    ) -> float:
        """Return the red light safe speed value owned by this controller math service."""
        return 0.80 * self._red_light_target_speed(distance_to_stopline, road_speed_limit)

    def _brake_curve_speed(
        self,
        effective_distance: float,
        road_speed_limit: float | None = None,
    ) -> float:
        """Physical braking curve without a separate Hermite toe section."""
        speed_limit = self._reward_speed_limit(road_speed_limit)
        horizon = self._red_light_brake_horizon(speed_limit)
        distance = max(0.0, self._safe_float(effective_distance, 0.0))
        progress = max(0.0, distance / max(horizon, 1e-6))
        return min(speed_limit, speed_limit * math.sqrt(progress))

    def _preferred_brake_curve_speed(
        self,
        effective_distance: float,
        road_speed_limit: float | None = None,
    ) -> float:
        """Return the preferred brake curve speed value owned by this controller math service."""
        speed_limit = self._reward_speed_limit(road_speed_limit)
        distance = max(0.0, self._safe_float(effective_distance, 0.0))
        return self._brake_curve_speed(distance, speed_limit)

    def _red_light_factor(
        self,
        distance_to_stopline: float,
        road_speed_limit: float | None = None,
        red_light_overdrive: bool = False,
        preferred_stop_offset_m: float | None = None,
    ) -> float:
        """Return the red light factor value owned by this controller math service."""
        speed_limit = self._reward_speed_limit(road_speed_limit)
        distance = max(0.0, self._safe_float(distance_to_stopline, 0.0))
        reward_horizon = self._red_light_reward_horizon(
            speed_limit,
            preferred_stop_offset_m=preferred_stop_offset_m,
        )
        blend_center = 0.55 * reward_horizon
        blend_width = max(2.0, 0.12 * reward_horizon)
        factor = self._sigmoid((blend_center - distance) / blend_width)
        if red_light_overdrive:
            return 1.0
        return factor

    def _calculate_red_light_speed_reward(
        self,
        current_speed: float,
        distance_to_stopline: float,
        road_speed_limit: float | None = None,
        blocked_by_obstacle: bool = False,
    ) -> float:
        """Evaluate and publish the existing red-light speed reward."""

        preferred_curve_offset_m = self._signal_reward_curve_offset_m(
            road_speed_limit,
        )
        inputs = self._red_light_speed_profile_input(
            current_speed,
            distance_to_stopline,
            road_speed_limit,
            preferred_stop_offset_m=preferred_curve_offset_m,
        )
        profile = calculate_continuous_signal_speed_profile(
            inputs,
            RED_LIGHT_PREFERRED_SPEED_PLATEAU_WIDTH_METERS,
        )

        far_wait_excess_m = max(
            0.0,
            inputs.distance_to_stopline_m
            - self._signal_stop_zone_limit_m(
                inputs.preferred_stop_offset_m
                + RED_LIGHT_PREFERRED_SPEED_PLATEAU_WIDTH_METERS
            ),
        )
        far_gate = 1.0 - math.exp(
            -far_wait_excess_m
            / max(RED_LIGHT_FAR_WAIT_DISTANCE_WIDTH_METERS, inputs.epsilon)
        )
        slow_gate = math.exp(
            -inputs.current_speed_mps
            / max(
                RED_LIGHT_FAR_WAIT_SLOW_SPEED_SCALE_MPS,
                inputs.epsilon,
            )
        )
        far_wait_penalty = calculate_red_light_far_wait_penalty(
            RedLightFarWaitPenaltyInput(
                far_distance_gate=far_gate,
                slow_speed_gate=slow_gate,
                penalty_weight=RED_LIGHT_FAR_WAIT_PENALTY_WEIGHT,
                blocked_by_obstacle=blocked_by_obstacle,
            )
        )

        diagnostics = self.controller.reward_state.speed.diagnostics
        diagnostics.red_light_speed_reward = profile.reward
        diagnostics.red_light_target_speed = inputs.brake_limit_speed_mps
        diagnostics.red_light_preferred_speed = inputs.preferred_speed_mps
        diagnostics.red_light_far_wait_penalty = far_wait_penalty
        diagnostics.red_light_profile_reward = (
            profile.reward * diagnostics.red_light_stopline_factor
        )
        return profile.reward

    def _red_light_speed_profile_input(
        self,
        current_speed: float,
        distance_to_stopline: float,
        road_speed_limit: float | None,
        preferred_stop_offset_m: float | None = None,
    ) -> RedLightSpeedProfileInput:
        """Normalize and derive curve geometry in the original operation order."""

        current = max(0.0, self._safe_float(current_speed, 0.0))
        distance = max(0.0, self._safe_float(distance_to_stopline, 0.0))
        speed_limit = self._reward_speed_limit(road_speed_limit)
        eps = 1e-6
        configured_stop_offset = self._safe_float(
            preferred_stop_offset_m,
            0.0,
        )
        stop_offset = max(configured_stop_offset, 0.0)
        max_negative_reward = RED_LIGHT_MAX_NEGATIVE_REWARD
        horizon = self._red_light_brake_horizon(speed_limit)
        reward_horizon = self._red_light_reward_horizon(
            speed_limit,
            preferred_stop_offset_m=stop_offset,
        )
        brake_limit_speed = self._brake_curve_speed(distance, speed_limit)
        preferred_speed = self._preferred_brake_curve_speed(
            distance - stop_offset,
            speed_limit,
        )
        lower_preferred_speed = self._preferred_brake_curve_speed(
            distance
            - stop_offset
            - max(
                0.0,
                RED_LIGHT_PREFERRED_SPEED_PLATEAU_WIDTH_METERS,
            ),
            speed_limit,
        )
        return RedLightSpeedProfileInput(
            current_speed_mps=current,
            distance_to_stopline_m=distance,
            road_speed_limit_mps=speed_limit,
            epsilon=eps,
            preferred_stop_offset_m=stop_offset,
            maximum_negative_reward=max_negative_reward,
            brake_horizon_m=horizon,
            reward_horizon_m=reward_horizon,
            brake_limit_speed_mps=brake_limit_speed,
            preferred_speed_mps=preferred_speed,
            lower_preferred_speed_mps=lower_preferred_speed,
        )

    def _get_distance_to_stopline(
        self,
        ego_station: float | None = None,
    ) -> float | None:
        """Return distance to the canonical signal boundary."""
        ego_station_value = self._safe_float(ego_station, None)
        if ego_station_value is None:
            return None
        return max(0.0, -ego_station_value)

    def _uses_stop_profile(
        self,
        traffic_light_state_code: int | float | None,
        upcoming_intersection_type_code: int | float | None,
    ) -> bool:
        """Return whether the distance-based stop profile applies."""
        traffic_state = self._as_int_code(traffic_light_state_code)
        intersection_type = self._as_int_code(upcoming_intersection_type_code)
        return (
            traffic_state in TRAFFIC_LIGHT_STOP_PROFILE_STATE_CODES
            and intersection_type
            == self.controller.reward_config.traffic_light_intersection_code
        )

    def _calculate_speed_component(
        self,
        current_speed: float,
        distance_to_stopline: float | None = None,
        traffic_light_state_code: int | None = None,
        upcoming_intersection_type_code: int | None = None,
        road_speed_limit: float | None = None,
        red_light_overdrive: bool = False,
        blocked_by_obstacle: bool = False,
        blocking_distance_m: float | None = None,
        blocking_obstacle_speed_mps: float | None = None,
        blocking_distance_is_constraint_clearance: bool = False,
        policy_action_mps2: float = 0.0,
        blocking_closing_speed_mps: float | None = None,
    ) -> float:
        """Calculate the speed component owned by this controller math service."""
        current = max(0.0, self._safe_float(current_speed, 0.0))
        speed_limit = self._reward_speed_limit(road_speed_limit)
        normal_reward = self._calculate_normal_speed_reward(current, speed_limit)

        speed_state = self.controller.reward_state.speed
        diagnostics = speed_state.diagnostics
        speed_state.raw = current
        speed_state.current = current
        speed_state.blocking_profile_active = False
        diagnostics.reset()
        crossing_reward = self._consume_signal_release_reward(
            current,
            traffic_light_state_code,
            upcoming_intersection_type_code,
            red_light_overdrive,
        )

        stopline_distance = self._active_stopline_distance(
            distance_to_stopline,
            traffic_light_state_code,
            upcoming_intersection_type_code,
            speed_limit,
            red_light_overdrive,
        )
        blocking_distance = self._active_blocking_distance(
            current,
            blocking_distance_m,
            speed_limit,
            red_light_overdrive,
            blocking_obstacle_speed_mps,
            blocking_distance_is_constraint_clearance,
        )
        blocking_owns_profile = blocking_distance is not None and (
            stopline_distance is None or blocking_distance < stopline_distance
        )
        if (
            stopline_distance is not None
            and not red_light_overdrive
            and not blocking_owns_profile
        ):
            self._record_signal_stop_observation(
                current,
                traffic_light_state_code,
            )
        if blocking_owns_profile:
            speed_state.blocking_profile_active = True
            speed_reward = self._calculate_blocking_speed_component(
                current,
                blocking_distance,
                speed_limit,
                normal_reward,
                blocking_obstacle_speed_mps,
                blocking_distance_is_constraint_clearance,
                policy_action_mps2,
                blocking_closing_speed_mps=blocking_closing_speed_mps,
            )
        elif stopline_distance is None:
            speed_reward = self._calculate_normal_context_speed_component(
                current,
                normal_reward,
                traffic_light_state_code,
                upcoming_intersection_type_code,
                blocked_by_obstacle,
            )
        else:
            speed_reward = self._calculate_stopline_speed_component(
                current,
                stopline_distance,
                speed_limit,
                normal_reward,
                red_light_overdrive,
                blocked_by_obstacle,
                traffic_light_state_code,
            )
        return self._add_speed_event_reward(speed_reward, crossing_reward)

    def _record_signal_stop_observation(
        self,
        current_speed_mps: float,
        traffic_light_state_code: int | None,
    ) -> None:
        """Observe physical stops independently of earning the stop-zone award."""

        signal_stop = self.controller.reward_state.signal_stop
        signal_stop.encounter_active = True
        prohibited_signal = (
            self._as_int_code(traffic_light_state_code)
            in TRAFFIC_LIGHT_PROHIBITED_CROSSING_STATE_CODES
        )
        if current_speed_mps >= SIGNAL_STOP_MOTION_EXIT_SPEED_MPS:
            if signal_stop.motion_qualified_stop_observed and prohibited_signal:
                signal_stop.positive_profile_locked = True
            signal_stop.approach_motion_seen = True
        elif (
            current_speed_mps <= SIGNAL_STOP_MOTION_ENTRY_SPEED_MPS
            and signal_stop.approach_motion_seen
            and prohibited_signal
        ):
            signal_stop.motion_qualified_stop_observed = True
        if self._is_stationary_signal_speed(current_speed_mps):
            signal_stop.stop_observed = True
        self.controller.reward_state.speed.diagnostics.signal_stop_observed = float(
            signal_stop.stop_observed
        )

    def _calculate_stopline_speed_component(
        self,
        current_speed_mps: float,
        stopline_distance_m: float,
        road_speed_limit_mps: float,
        normal_reward: float,
        red_light_overdrive: bool,
        blocked_by_obstacle: bool,
        traffic_light_state_code: int | None,
    ) -> float:
        """Apply the shared signal stop profile at the selected stopline."""

        speed_state = self.controller.reward_state.speed
        diagnostics = speed_state.diagnostics
        diagnostics.red_light_distance_to_stopline = stopline_distance_m
        curve_distance_m = self._signal_reward_curve_distance(
            stopline_distance_m,
            road_speed_limit_mps,
        )
        red_factor = self._red_light_factor(
            curve_distance_m,
            road_speed_limit_mps,
            red_light_overdrive,
        )
        diagnostics.red_light_stopline_factor = red_factor
        profile_speed_mps = (
            0.0
            if self._is_stationary_signal_speed(current_speed_mps)
            else current_speed_mps
        )
        red_reward = self._calculate_red_light_speed_reward(
            profile_speed_mps,
            stopline_distance_m,
            road_speed_limit_mps,
            blocked_by_obstacle,
        )
        if red_light_overdrive:
            red_reward = RED_LIGHT_MAX_NEGATIVE_REWARD
            diagnostics.red_light_speed_reward = red_reward
        brake_limit_speed = max(diagnostics.red_light_target_speed, 1e-6)
        preferred_speed = max(diagnostics.red_light_preferred_speed, 0.0)
        red_speed_control_ratio = self._stop_profile_speed_control_ratio(
            current_speed_mps,
            brake_limit_speed,
            preferred_speed,
        )
        red_blend_weight = max(red_factor, red_speed_control_ratio)
        blended_speed_reward = (
            ((1.0 - red_blend_weight) * normal_reward)
            + (red_blend_weight * red_reward)
        )
        effective_reward = blended_speed_reward - diagnostics.red_light_far_wait_penalty
        if (
            self.controller.reward_state.signal_stop.positive_profile_locked
            and self._as_int_code(traffic_light_state_code)
            in TRAFFIC_LIGHT_PROHIBITED_CROSSING_STATE_CODES
        ):
            effective_reward = min(0.0, effective_reward)
        stationary_reward = self._stationary_signal_stop_reward(
            current_speed_mps,
            stopline_distance_m,
            road_speed_limit_mps,
            red_reward,
            red_light_overdrive,
            traffic_light_state_code,
        )
        if stationary_reward is not None:
            effective_reward = stationary_reward
        speed_state.normalized = self._apply_speed_reward_jitter_discount(effective_reward)
        speed_state.effective = speed_state.normalized
        diagnostics.red_light_profile_reward = blended_speed_reward
        return speed_state.effective

    def _stationary_signal_stop_reward(
        self,
        current_speed_mps: float,
        stopline_distance_m: float,
        road_speed_limit_mps: float,
        profile_reward: float,
        red_light_overdrive: bool,
        traffic_light_state_code: int | None,
    ) -> float | None:
        """Reward the first true stop once and keep a prohibited hold neutral."""

        stop_zone_limit_m = self._signal_stop_zone_limit_m(
            self._signal_reward_curve_offset_m(road_speed_limit_mps)
            + RED_LIGHT_PREFERRED_SPEED_PLATEAU_WIDTH_METERS
        )
        if (
            red_light_overdrive
            or stopline_distance_m > stop_zone_limit_m
        ):
            return None

        signal_stop = self.controller.reward_state.signal_stop
        is_stationary = self._is_stationary_signal_speed(current_speed_mps)
        if not is_stationary:
            if (
                signal_stop.quality_recorded
                and self._as_int_code(traffic_light_state_code)
                in TRAFFIC_LIGHT_PROHIBITED_CROSSING_STATE_CODES
            ):
                return 0.0
            return None

        first_stop = not signal_stop.quality_recorded
        diagnostics = self.controller.reward_state.speed.diagnostics
        diagnostics.signal_stop_observed = float(signal_stop.stop_observed)
        diagnostics.signal_stop_quality = signal_stop.stop_quality
        if first_stop:
            signal_stop.stop_observed = True
            signal_stop.quality_recorded = True
            signal_stop.stop_quality = max(0.0, min(1.0, profile_reward))
        diagnostics.signal_stop_observed = 1.0
        diagnostics.signal_stop_quality = signal_stop.stop_quality
        if first_stop:
            return signal_stop.stop_quality
        return 0.0

    def _apply_stationary_signal_positive_accel_penalty(
        self,
        policy_action_mps2: float,
        current_speed_mps: float,
        distance_to_stopline_m: float | None,
        traffic_light_state_code: int | None,
        upcoming_intersection_type_code: int | None,
        road_speed_limit_mps: float | None,
        red_light_overdrive: bool,
    ) -> float:
        """Penalize policy throttle after a prohibited-signal stop."""

        diagnostics = self.controller.reward_state.speed.diagnostics
        diagnostics.signal_stop_positive_accel_penalty = 0.0
        traffic_state = self._as_int_code(traffic_light_state_code)
        intersection_type = self._as_int_code(upcoming_intersection_type_code)
        stopline_distance = self._safe_float(distance_to_stopline_m, None)
        if (
            red_light_overdrive
            or self.controller.reward_state.speed.blocking_profile_active
            or traffic_state not in TRAFFIC_LIGHT_PROHIBITED_CROSSING_STATE_CODES
            or intersection_type != INTERSECTION_TYPE_TRAFFIC_LIGHT
            or not (
                self.controller.reward_state.signal_stop.stop_observed
                or self.controller.reward_state.signal_stop.motion_qualified_stop_observed
            )
            or stopline_distance is None
            or stopline_distance < 0.0
        ):
            return 0.0

        speed_limit = self._reward_speed_limit(road_speed_limit_mps)
        stop_zone_limit_m = self._signal_stop_zone_limit_m(
            self._signal_reward_curve_offset_m(speed_limit)
            + RED_LIGHT_PREFERRED_SPEED_PLATEAU_WIDTH_METERS
        )
        positive_profile_locked = (
            self.controller.reward_state.signal_stop.positive_profile_locked
        )
        if (
            stopline_distance > self._signal_reward_horizon(speed_limit)
            or (
                stopline_distance > stop_zone_limit_m
                and not positive_profile_locked
            )
        ):
            return 0.0

        positive_command_mps2 = max(
            0.0,
            self._safe_float(policy_action_mps2, 0.0),
        )
        penalty = self._positive_stop_accel_penalty(positive_command_mps2)
        diagnostics.signal_stop_positive_accel_penalty = penalty
        speed_state = self.controller.reward_state.speed
        speed_state.normalized -= penalty
        speed_state.effective -= penalty
        return penalty

    @staticmethod
    def _positive_stop_accel_penalty(policy_action_mps2: float) -> float:
        """Use the same throttle cost for a signal hold or a stopped obstacle."""

        return min(
            abs(RED_LIGHT_MAX_NEGATIVE_REWARD),
            max(0.0, policy_action_mps2)
            / max(SIGNAL_STOP_POSITIVE_ACCEL_PENALTY_SCALE_MPS2, 1e-6),
        )

    @staticmethod
    def _is_stationary_signal_speed(current_speed_mps: float) -> bool:
        """Treat numerical creep inside the shared tolerance as a true stop."""

        return current_speed_mps <= SIGNAL_STOP_SPEED_TOLERANCE_MPS

    @staticmethod
    def _signal_stop_zone_limit_m(stop_zone_width_m: float) -> float:
        """Return the shared outer bound of the signal stop-quality zone."""

        return max(1e-6, float(stop_zone_width_m))

    def _consume_signal_release_reward(
        self,
        current_speed_mps: float,
        traffic_light_state_code: int | None,
        upcoming_intersection_type_code: int | None,
        red_light_overdrive: bool,
    ) -> float:
        """Return one completion reward after a valid signal release."""

        signal_stop = self.controller.reward_state.signal_stop
        diagnostics = self.controller.reward_state.speed.diagnostics
        diagnostics.signal_stop_observed = float(signal_stop.stop_observed)
        diagnostics.signal_stop_quality = signal_stop.stop_quality
        if red_light_overdrive:
            signal_stop.reset()
            return 0.0

        intersection_type = self._as_int_code(upcoming_intersection_type_code)
        traffic_state = self._as_int_code(traffic_light_state_code)
        is_traffic_light = (
            intersection_type
            == self.controller.reward_config.traffic_light_intersection_code
        )
        if is_traffic_light:
            if (
                signal_stop.encounter_active
                and signal_stop.stop_observed
                and traffic_state == TRAFFIC_LIGHT_STATE_GREEN
                and not self._is_stationary_signal_speed(current_speed_mps)
            ):
                signal_stop.release_motion_seen = True
            return 0.0

        if not signal_stop.encounter_active:
            return 0.0

        crossing_reward = 0.0
        if signal_stop.stop_observed and signal_stop.release_motion_seen:
            crossing_reward = (
                RED_LIGHT_RELEASE_CROSSING_REWARD_WEIGHT
                * signal_stop.stop_quality
            )
        diagnostics.signal_release_reward = crossing_reward
        signal_stop.reset()
        return crossing_reward

    def _add_speed_event_reward(
        self,
        speed_reward: float,
        event_reward: float,
    ) -> float:
        """Add a one-shot signal event without changing continuous curve shaping."""

        if event_reward == 0.0:
            return speed_reward
        speed_state = self.controller.reward_state.speed
        speed_state.normalized += event_reward
        speed_state.effective += event_reward
        return speed_state.effective

    def _active_stopline_distance(
        self,
        distance_to_stopline_m: float | None,
        traffic_light_state_code: int | None,
        upcoming_intersection_type_code: int | None,
        road_speed_limit_mps: float,
        red_light_overdrive: bool,
    ) -> float | None:
        """Return the active signal stopline distance within its reward horizon."""

        distance = self._safe_float(distance_to_stopline_m, None)
        if red_light_overdrive:
            distance = 0.0
        uses_stop_profile = self._uses_stop_profile(
            traffic_light_state_code,
            upcoming_intersection_type_code,
        ) or red_light_overdrive
        if (
            distance is None
            or not uses_stop_profile
            or distance > self._signal_reward_horizon(road_speed_limit_mps)
        ):
            return None
        return max(0.0, distance)

    def _calculate_normal_context_speed_component(
        self,
        current_speed_mps: float,
        normal_reward: float,
        traffic_light_state_code: int | None,
        upcoming_intersection_type_code: int | None,
        blocked_by_obstacle: bool,
    ) -> float:
        """Apply the unblocked road-speed reward and contextual idle gate."""

        speed_state = self.controller.reward_state.speed
        diagnostics = speed_state.diagnostics
        diagnostics.green_light_idle_penalty = self._calculate_green_light_idle_penalty(
            current_speed_mps,
            traffic_light_state_code,
            upcoming_intersection_type_code,
            blocked_by_obstacle,
        )
        speed_state.normalized = self._apply_speed_reward_jitter_discount(
            normal_reward - diagnostics.green_light_idle_penalty
        )
        speed_state.effective = speed_state.normalized
        return speed_state.effective

    def _active_blocking_distance(
        self,
        current_speed_mps: float,
        blocking_distance_m: float | None,
        road_speed_limit_mps: float,
        red_light_overdrive: bool,
        blocking_obstacle_speed_mps: float | None = None,
        blocking_distance_is_constraint_clearance: bool = False,
    ) -> float | None:
        """Return an active obstacle distance, or ``None`` outside its horizon."""

        distance = self._safe_float(blocking_distance_m, None)
        preferred_gap_m = (
            0.0
            if blocking_distance_is_constraint_clearance
            else self._blocking_preferred_gap_m(
                current_speed_mps,
                road_speed_limit_mps,
                blocking_obstacle_speed_mps,
            )
        )
        reward_horizon_m = self._red_light_reward_horizon(
            road_speed_limit_mps,
            preferred_stop_offset_m=preferred_gap_m,
        )
        if (
            red_light_overdrive
            or distance is None
            or distance > reward_horizon_m
        ):
            return None
        return max(0.0, distance)

    def _calculate_blocking_speed_component(
        self,
        current_speed_mps: float,
        blocking_distance_m: float,
        road_speed_limit_mps: float,
        normal_reward: float,
        blocking_obstacle_speed_mps: float | None = None,
        blocking_distance_is_constraint_clearance: bool = False,
        policy_action_mps2: float = 0.0,
        blocking_closing_speed_mps: float | None = None,
    ) -> float:
        """Reuse the stop curve for a policy-visible crossing obstacle."""

        obstacle_speed_mps = max(
            0.0,
            self._safe_float(blocking_obstacle_speed_mps, 0.0),
        )
        ego_is_stationary = self._is_stationary_signal_speed(current_speed_mps)
        obstacle_is_stationary = self._is_stationary_signal_speed(
            obstacle_speed_mps
        )
        conflict_closing_speed_mps = self._safe_float(
            blocking_closing_speed_mps,
            None,
        )
        closing_speed = max(
            0.0,
            (
                current_speed_mps - obstacle_speed_mps
                if conflict_closing_speed_mps is None
                else conflict_closing_speed_mps
            ),
        )
        if ego_is_stationary and (
            conflict_closing_speed_mps is not None or obstacle_is_stationary
        ):
            # SUMO can retain a minute positive speed after the QP has settled.
            # A stopped ego must also remain neutral while perpendicular
            # cross-traffic moves; that actor's motion is not ego-controllable.
            closing_speed = 0.0
        preferred_gap_m = (
            0.0
            if blocking_distance_is_constraint_clearance
            else self._blocking_preferred_gap_m(
                current_speed_mps,
                road_speed_limit_mps,
                obstacle_speed_mps,
            )
        )

        inputs = self._red_light_speed_profile_input(
            closing_speed,
            blocking_distance_m,
            road_speed_limit_mps,
            preferred_stop_offset_m=preferred_gap_m,
        )
        profile = calculate_red_light_speed_profile(inputs)
        blocking_reward = (
            min(0.0, profile.reward)
            if ego_is_stationary
            else profile.reward
        )
        blocking_reward = self._blocking_progress_reward(
            blocking_reward, current_speed_mps, road_speed_limit_mps,
        )
        distance_factor = self._red_light_factor(
            blocking_distance_m,
            road_speed_limit_mps,
            preferred_stop_offset_m=preferred_gap_m,
        )
        speed_control_ratio = self._stop_profile_speed_control_ratio(
            closing_speed,
            max(inputs.brake_limit_speed_mps, 1e-6),
            max(inputs.preferred_speed_mps, 0.0),
        )
        blend_weight = max(distance_factor, speed_control_ratio)
        blended_reward = (
            blocking_reward
            if ego_is_stationary
            else (
                (1.0 - blend_weight) * normal_reward
                + blend_weight * blocking_reward
            )
        )
        blended_reward -= self._blocking_hold_throttle_penalty(
            blocking_distance_m, preferred_gap_m, obstacle_speed_mps, policy_action_mps2,
        )
        speed_state = self.controller.reward_state.speed
        diagnostics = speed_state.diagnostics
        diagnostics.blocking_profile_distance_m = blocking_distance_m
        diagnostics.blocking_profile_obstacle_speed_mps = obstacle_speed_mps
        diagnostics.blocking_profile_closing_speed_mps = closing_speed
        diagnostics.blocking_profile_target_speed_mps = (
            inputs.brake_limit_speed_mps
        )
        diagnostics.blocking_profile_preferred_speed_mps = (
            inputs.preferred_speed_mps
        )
        diagnostics.blocking_profile_reward = blended_reward
        speed_state.normalized = self._apply_speed_reward_jitter_discount(
            blended_reward
        )
        speed_state.effective = speed_state.normalized
        return speed_state.effective

    def _blocking_progress_reward(
        self, profile_reward: float, current_speed_mps: float, road_speed_limit_mps: float,
    ) -> float:
        """Scale positive plateau credit by progress; leave negative credit intact."""

        if profile_reward <= 0.0:
            return profile_reward
        band_speed_mps = self._brake_curve_speed(
            RED_LIGHT_PREFERRED_SPEED_PLATEAU_WIDTH_METERS, road_speed_limit_mps,
        )
        return profile_reward * min(1.0, current_speed_mps / max(band_speed_mps, 1e-6))

    def _blocking_hold_throttle_penalty(
        self, distance_m: float, preferred_gap_m: float, obstacle_speed_mps: float,
        policy_action_mps2: float,
    ) -> float:
        """Ramp the existing throttle cost across an observed stationary stop band."""

        if not self._is_stationary_signal_speed(obstacle_speed_mps):
            return 0.0
        band_m = max(RED_LIGHT_PREFERRED_SPEED_PLATEAU_WIDTH_METERS, 1e-6)
        weight = max(0.0, min(1.0, (preferred_gap_m + band_m - distance_m) / band_m))
        return weight * self._positive_stop_accel_penalty(
            self._safe_float(policy_action_mps2, 0.0)
        )

    def _blocking_preferred_gap_m(
        self,
        current_speed_mps: float,
        road_speed_limit_mps: float,
        blocking_obstacle_speed_mps: float | None = None,
    ) -> float:
        """Adapt the shared stop offset to reaction and braking distance."""

        speed_limit = self._reward_speed_limit(road_speed_limit_mps)
        current_speed = max(
            0.0,
            min(
                speed_limit,
                self._safe_float(current_speed_mps, 0.0),
            ),
        )
        headway_seconds = max(
            0.0,
            self._safe_float(LEAD_HEADWAY_SECONDS, 0.0),
        )
        obstacle_speed = max(
            0.0,
            self._safe_float(blocking_obstacle_speed_mps, 0.0),
        )
        comfort_deceleration = self._safe_float(
            self.controller.reward_config.red_light_a_comfort,
            1.5,
        )
        relative_braking_reserve_m = max(
            0.0,
            (current_speed * current_speed) - (obstacle_speed * obstacle_speed),
        ) / (2.0 * max(comfort_deceleration, 1e-6))
        return (
            PEDESTRIAN_AVOIDANCE_RADIUS_M
            + (headway_seconds * current_speed)
            + relative_braking_reserve_m
        )

    @staticmethod
    def _stop_profile_speed_control_ratio(
        current_speed_mps: float,
        brake_limit_speed_mps: float,
        preferred_speed_mps: float,
    ) -> float:
        """Return how strongly overspeed requires the shared stop profile."""

        if current_speed_mps >= brake_limit_speed_mps:
            return 1.0
        profile_width = brake_limit_speed_mps - preferred_speed_mps
        if profile_width <= 1e-6 or current_speed_mps <= preferred_speed_mps:
            return 0.0
        ratio = (
            current_speed_mps - preferred_speed_mps
        ) / profile_width
        return max(0.0, min(1.0, ratio))

    def _calculate_delta_component(
        self,
        delta: float,
        prev_delta: float,
    ) -> float:
        """Calculate the delta component owned by this controller math service."""
        current_cmd_delta = self.controller._current_supervised_cmd_accel_delta
        prev_cmd_delta = self.controller.controller_state.last_supervised_cmd_accel_delta
        cmd_delta_change = float(current_cmd_delta) - float(prev_cmd_delta)
        local_jitter = abs(cmd_delta_change)
        previous_cmd_accel = float(
            self.controller.controller_state.last_supervised_cmd_accel
        )
        current_cmd_accel = previous_cmd_accel + float(current_cmd_delta)
        local_sign_changed = self._values_have_sign_change(
            (previous_cmd_accel, current_cmd_accel)
        )
        local_penalty_jitter = self._scale_jitter_for_sign_change(
            local_jitter,
            local_sign_changed,
        )
        trend_penalty_jitter = self._calculate_supervised_cmd_accel_trend_jitter(
            current_cmd_delta
        )
        comfort_penalty_jitter = self._calculate_supervised_cmd_accel_comfort_jitter(
            current_cmd_delta
        )
        dominant_jitter = max(
            local_penalty_jitter,
            trend_penalty_jitter,
            comfort_penalty_jitter,
        )

        self.controller.supervised_cmd_accel_delta_change = cmd_delta_change
        self.controller.abs_supervised_cmd_accel_delta_change = local_jitter
        self.controller.supervised_cmd_accel_local_jitter = local_jitter
        self.controller.supervised_cmd_accel_local_jitter_sign_change = float(
            local_sign_changed
        )
        self.controller.supervised_cmd_accel_local_effective_jitter = (
            local_penalty_jitter
        )
        delta_state = self.controller.reward_state.delta
        delta_state.raw = -dominant_jitter
        delta_state.normalized = (
            delta_state.raw / SUPERVISED_CMD_ACCEL_DELTA_CHANGE_NORMALIZER
        )
        return delta_state.normalized

    def _clear_sync_component(self) -> None:
        """Clear the sync component owned by this controller math service."""
        sync_state = self.controller.reward_state.sync
        sync_state.raw = 0.0
        sync_state.normalized = 0.0
        sync_state.command = 0.0
        sync_state.actual_acceleration = 0.0
        sync_state.absolute_error = 0.0
        sync_state.effective_error = 0.0
        sync_state.direction_penalty = 0.0
        sync_state.magnitude_penalty = 0.0
        sync_state.speed_factor = 0.0

    def _clear_delta_component(self) -> None:
        """Clear the delta component owned by this controller math service."""
        delta_state = self.controller.reward_state.delta
        delta_state.raw = 0.0
        delta_state.normalized = 0.0

    def _update_reward_penalty_averages(
        self,
        step_counter: int | None,
    ) -> None:
        """Update the reward penalty averages owned by this controller math service."""
        sync_state = self.controller.reward_state.sync
        accel_state = self.controller.reward_state.accel
        delta_state = self.controller.reward_state.delta
        sync_state.penalty_sum += abs(sync_state.normalized)
        accel_state.penalty_sum += abs(accel_state.normalized)
        delta_state.penalty_sum += abs(delta_state.normalized)

        effective_step_counter = max(1, step_counter or 1)
        sync_state.rolling_average = -(sync_state.penalty_sum / effective_step_counter)
        accel_state.rolling_average = -(accel_state.penalty_sum / effective_step_counter)
        delta_state.rolling_average = -(delta_state.penalty_sum / effective_step_counter)
        sync_state.cumulative_average = sync_state.rolling_average
        accel_state.cumulative_average = accel_state.rolling_average
        delta_state.cumulative_average = delta_state.rolling_average

    def _apply_reward_weights(self) -> None:
        """Apply the reward weights transition owned by this controller math service."""
        weights = self._get_reward_weights()
        state = self.controller.reward_state
        state.sync.weighted = weights["sync"] * state.sync.normalized
        state.accel.weighted = weights["accel"] * state.accel.normalized
        state.speed.weighted = weights["speed"] * state.speed.effective
        state.delta.weighted = weights["delta"] * state.delta.normalized
        state.sync.weighted_uncapped = state.sync.weighted
        state.accel.weighted_uncapped = state.accel.weighted
        state.speed.weighted_uncapped = state.speed.weighted
        state.delta.weighted_uncapped = state.delta.weighted

    def _base_reward_sum(self) -> float:
        """Return the base reward sum value owned by this controller math service."""
        return (
            self.controller.reward_state.sync.weighted
            + self.controller.reward_state.accel.weighted
            + self.controller.reward_state.speed.weighted
            + self.controller.reward_state.delta.weighted
        )

    @staticmethod
    def _as_int_code(value: object) -> int | None:
        """Return this controller math service's value as int code."""
        try:
            return int(round(float(cast(float | int | str, value))))
        except (TypeError, ValueError):
            return None

    def _update_reward_memory(
        self,
        reward_value: float,
        cmd_accel: float,
        delta: float,
        ego_station: float | None,
        traffic_light_state_code: int | None = None,
        upcoming_intersection_type_code: int | None = None,
    ) -> None:
        """Update the reward memory owned by this controller math service."""
        self.controller.reward_state.total = reward_value
        controller_state = self.controller.controller_state
        controller_state.last_supervised_cmd_accel = cmd_accel
        controller_state.last_supervised_cmd_accel_delta = (
            self.controller._current_supervised_cmd_accel_delta
        )
        controller_state.last_supervised_delta = delta
        self.controller.supervised_cmd_accel_history.append(cmd_accel)
        self.controller.supervised_cmd_accel_delta_history.append(
            controller_state.last_supervised_cmd_accel_delta
        )
