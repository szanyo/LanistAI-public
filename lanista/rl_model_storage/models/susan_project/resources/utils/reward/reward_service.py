# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed reward workflow and metric publication for Susan."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast

from lanista.domains.automotive.control.longitudinal_control_optimizer import (
    nominal_pedestrian_avoidance_approach_station_m,
    nominal_vehicle_avoidance_approach_station_m,
)
from lanista.reinforcengine.utils.simulation.intersection_metadata import (
    INTERSECTION_TYPE_TRAFFIC_LIGHT,
)

from ..control.controller_config import LEGACY_VEHICLE_CONFLICT_RELEASE_M
from ..observation.policy_obstacle_projection import (
    PolicyObstacleGeometry,
    PolicyObstacleKind,
)
from ..observation.vehicle_telemetry import VehicleControllerTelemetryState
from ..simulation.simulation_contracts import (
    PedestrianParameters,
    has_valid_ego_lead_telemetry,
    is_ego_blocked_by_pedestrian,
)
from .reward_telemetry import (
    SYNC_ERROR_DEADBAND_MPS2,
    RewardStateTelemetry,
    RewardTelemetry,
)


class RewardStatePort(Protocol):
    """Mutable controller state intentionally updated by the reward workflow."""

    reward_state: RewardStateTelemetry
    controller_state: VehicleControllerTelemetryState
    policy_action: float
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
    speed_reward_jitter_discount: float
    speed_reward_jitter_discount_applied: float

class RewardMathPort(Protocol):
    """Numerical operations used by the reward workflow."""

    def _calculate_actual_acceleration(
        self,
        previous_speed_mps: float,
        current_speed_mps: float,
        actual_acceleration_override: float | None = None,
    ) -> float:
        """Calculate the actual acceleration owned by this reward math port."""
        ...

    def _get_distance_to_stopline(
        self,
        ego_station_m: float | None,
    ) -> float | None:
        """Return the distance to stopline exposed by this reward math port."""
        ...

    def _uses_stop_profile(
        self,
        traffic_light_state_code: int | None,
        upcoming_intersection_type_code: int | None,
    ) -> bool:
        """Return whether the distance-based stop profile applies."""
        ...

    def _clear_sync_component(self) -> None:
        """Clear the sync component owned by this reward math port."""
        ...

    def _calculate_sync_component(
        self,
        actual_acceleration_mps2: float,
        command_acceleration_mps2: float,
        current_speed_mps: float,
        road_speed_limit_mps: float | None = None,
    ) -> float:
        """Calculate the sync component owned by this reward math port."""
        ...

    def _calculate_accel_component(
        self,
        command_acceleration_mps2: float,
        previous_command_acceleration_mps2: float,
        current_speed_mps: float,
    ) -> float:
        """Calculate the accel component owned by this reward math port."""
        ...

    def _calculate_speed_component(
        self,
        current_speed_mps: float,
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
        """Calculate the speed component owned by this reward math port."""
        ...

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
        """Apply the prohibited-signal stationary policy-throttle penalty."""
        ...

    def _calculate_delta_component(
        self,
        delta_mps2: float,
        previous_delta_mps2: float,
    ) -> float:
        """Calculate the delta component owned by this reward math port."""
        ...

    def _update_reward_penalty_averages(
        self,
        step_counter: int | None,
    ) -> None:
        """Update the reward penalty averages owned by this reward math port."""
        ...

    def _apply_reward_weights(self) -> None:
        """Apply the reward weights transition owned by this reward math port."""
        ...

    def _base_reward_sum(self) -> float:
        """Return the base reward sum value provided by this reward math port."""
        ...

    def _update_reward_memory(
        self,
        reward_value: float,
        command_acceleration_mps2: float,
        delta_mps2: float,
        ego_station_m: float | None,
        traffic_light_state_code: int | None = None,
        upcoming_intersection_type_code: int | None = None,
    ) -> None:
        """Update the reward memory owned by this reward math port."""
        ...


@dataclass(frozen=True, slots=True)
class BlockingObstacleContext:
    """Nearest policy obstacle inputs used by the shared stop profile."""

    distance_m: float
    speed_mps: float
    kind: PolicyObstacleKind
    closing_speed_mps: float | None = None
    distance_is_constraint_clearance: bool = False


@dataclass(frozen=True)
class RewardCalculationRequest:
    """Unit-explicit inputs for one Susan reward calculation."""

    supervised_command_mps2: float
    previous_speed_mps: float
    current_speed_mps: float
    applied_delta_mps2: float
    step_counter: int | None
    policy_action_mps2: float | None = None
    reference_speed_mps: float | None = None
    actual_acceleration_override_mps2: float | None = None
    traffic_light_state_code: int | None = None
    upcoming_intersection_type_code: int | None = None
    ego_station_m: float | None = None
    constraint_ego_station_m: float | None = None
    distance_to_stopline_m: float | None = None
    road_speed_limit_mps: float | None = None
    red_light_overdrive: bool = False
    object_stations_m: tuple[float, ...] | None = None
    policy_object_distances_m: tuple[float, ...] | None = None
    policy_object_speeds_mps: tuple[float, ...] | None = None
    policy_object_kinds: tuple[str, ...] | None = None
    policy_object_geometries: tuple[str, ...] | None = None
    policy_object_constraint_clearances_m: tuple[float, ...] | None = None
    pedestrian_parameters: PedestrianParameters | None = None
    ego_lead_vehicle_present: bool = False
    ego_lead_distance_m: float = math.nan
    ego_lead_speed_mps: float = 0.0

    @property
    def blocking_distance_m(self) -> float | None:
        """Return the nearest observation-visible obstacle distance."""

        obstacle = self.blocking_obstacle
        return obstacle.distance_m if obstacle is not None else None

    @property
    def blocking_obstacle(self) -> BlockingObstacleContext | None:
        """Return the nearest observed obstacle with its aligned speed."""

        release_vehicles = self._non_signal_vehicle_conflict_released()
        longitudinal_leader_present = self._has_projected_longitudinal_leader()
        projected_pedestrian_present = False
        obstacles: list[BlockingObstacleContext] = []
        for index, value in enumerate(self.policy_object_distances_m or ()):
            try:
                distance_m = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(distance_m) or distance_m < 0.0:
                continue

            kind = self._policy_object_kind(index)
            geometry = self._policy_object_geometry(index)
            if (
                (release_vehicles or longitudinal_leader_present)
                and kind is PolicyObstacleKind.VEHICLE
                and geometry is PolicyObstacleGeometry.CONFLICT_POINT
            ):
                continue
            constraint_clearance_m = self._policy_object_constraint_clearance_m(
                index
            )
            constraint_violated = False
            if kind is PolicyObstacleKind.VEHICLE:
                if (
                    constraint_clearance_m is None
                    and geometry is PolicyObstacleGeometry.CONFLICT_POINT
                ):
                    constraint_clearance_m = self._vehicle_constraint_clearance_m(
                        index
                    )
            elif kind is PolicyObstacleKind.PEDESTRIAN:
                projected_pedestrian_present = True
                signed_clearance_m = self._pedestrian_signed_constraint_clearance_m(
                    index=index
                )
                constraint_violated = bool(
                    signed_clearance_m is not None and signed_clearance_m < 0.0
                )
            distance_is_constraint_clearance = bool(
                constraint_clearance_m is not None and not constraint_violated
            )
            if constraint_clearance_m is not None:
                distance_m = constraint_clearance_m
            obstacle_speed_mps = (
                self._policy_object_speed_mps(index)
                if kind is PolicyObstacleKind.VEHICLE
                else 0.0
            )
            obstacles.append(
                BlockingObstacleContext(
                    distance_m=distance_m,
                    speed_mps=obstacle_speed_mps,
                    kind=kind,
                    closing_speed_mps=(
                        self._conflict_point_closing_speed_mps(index=index)
                        if geometry is PolicyObstacleGeometry.CONFLICT_POINT
                        else None
                    ),
                    distance_is_constraint_clearance=distance_is_constraint_clearance,
                )
            )

        if (
            not projected_pedestrian_present
            and is_ego_blocked_by_pedestrian(self.pedestrian_parameters)
        ):
            signed_clearance_m = self._pedestrian_signed_constraint_clearance_m()
            if signed_clearance_m is not None:
                obstacles.append(
                    BlockingObstacleContext(
                        distance_m=max(0.0, signed_clearance_m),
                        speed_mps=0.0,
                        kind=PolicyObstacleKind.PEDESTRIAN,
                        closing_speed_mps=(
                            self._conflict_point_closing_speed_mps()
                        ),
                        distance_is_constraint_clearance=(
                            signed_clearance_m >= 0.0
                        ),
                    )
                )

        return min(
            obstacles,
            key=lambda obstacle: obstacle.distance_m,
            default=None,
        )

    def _non_signal_vehicle_conflict_released(self) -> bool:
        """Return whether the shared legacy vehicle window has been cleared."""

        try:
            ego_station_m = float(cast(float | int | str, self.ego_station_m))
            intersection_type = int(
                cast(int | float | str, self.upcoming_intersection_type_code)
            )
        except (TypeError, ValueError):
            return False
        return bool(
            math.isfinite(ego_station_m)
            and ego_station_m > LEGACY_VEHICLE_CONFLICT_RELEASE_M
            and intersection_type != INTERSECTION_TYPE_TRAFFIC_LIGHT
        )

    def _policy_object_kind(self, index: int) -> PolicyObstacleKind:
        """Return one aligned policy actor kind with a legacy vehicle default."""

        try:
            return PolicyObstacleKind((self.policy_object_kinds or ())[index])
        except (IndexError, TypeError, ValueError):
            return PolicyObstacleKind.VEHICLE

    def _policy_object_geometry(self, index: int) -> PolicyObstacleGeometry:
        """Return the coordinate meaning aligned with one policy object slot."""

        try:
            return PolicyObstacleGeometry(
                (self.policy_object_geometries or ())[index]
            )
        except (IndexError, TypeError, ValueError):
            return PolicyObstacleGeometry.CONFLICT_POINT

    def _policy_object_speed_mps(self, index: int) -> float:
        """Return one finite non-negative aligned vehicle speed."""

        try:
            speed_mps = float((self.policy_object_speeds_mps or ())[index])
        except (IndexError, TypeError, ValueError):
            return 0.0
        return max(0.0, speed_mps) if math.isfinite(speed_mps) else 0.0

    def _conflict_point_closing_speed_mps(
        self,
        *,
        index: int | None = None,
    ) -> float | None:
        """Return inward radial speed in the supervisor conflict plane."""

        constraint_ego_station_m = (
            self.constraint_ego_station_m
            if self.constraint_ego_station_m is not None
            else self.ego_station_m
        )
        try:
            ego_station_m = float(
                cast(float | int | str, constraint_ego_station_m)
            )
            ego_speed_mps = max(0.0, float(self.current_speed_mps))
            if index is None:
                pedestrian_parameters = self.pedestrian_parameters
                if pedestrian_parameters is None:
                    return None
                object_station_m = float(pedestrian_parameters[0])
                object_speed_mps = max(0.0, float(pedestrian_parameters[1]))
            else:
                object_station_m = float((self.object_stations_m or ())[index])
                object_speed_mps = self._policy_object_speed_mps(index)
        except (IndexError, TypeError, ValueError):
            return None
        if not all(
            math.isfinite(value)
            for value in (
                ego_station_m,
                ego_speed_mps,
                object_station_m,
                object_speed_mps,
            )
        ):
            return None

        radius_m = math.hypot(ego_station_m, object_station_m)
        if radius_m <= 1e-12:
            return ego_speed_mps
        radial_rate_mps = (
            ego_station_m * ego_speed_mps
            + object_station_m * object_speed_mps
        ) / radius_m
        return max(0.0, -radial_rate_mps)

    def _policy_object_constraint_clearance_m(self, index: int) -> float | None:
        """Return one aligned fixed QP clearance when preprocessing supplied it."""

        try:
            clearance_m = float(
                (self.policy_object_constraint_clearances_m or ())[index]
            )
        except (IndexError, TypeError, ValueError):
            return None
        if not math.isfinite(clearance_m) or clearance_m < 0.0:
            return None
        return clearance_m

    def _has_projected_longitudinal_leader(self) -> bool:
        """Return whether the slots contain a leader with an aligned QP boundary."""

        return any(
            self._policy_object_kind(index) is PolicyObstacleKind.VEHICLE
            and self._policy_object_geometry(index)
            is PolicyObstacleGeometry.LONGITUDINAL_GAP
            and self._policy_object_constraint_clearance_m(index) is not None
            for index, _value in enumerate(self.policy_object_distances_m or ())
        )

    def _vehicle_constraint_clearance_m(self, index: int) -> float | None:
        """Return distance remaining before the nominal QP vehicle boundary."""

        constraint_ego_station_m = (
            self.constraint_ego_station_m
            if self.constraint_ego_station_m is not None
            else self.ego_station_m
        )
        try:
            object_station_m = float((self.object_stations_m or ())[index])
            ego_station_m = float(
                cast(float | int | str, constraint_ego_station_m)
            )
        except (IndexError, TypeError, ValueError):
            return None
        if not all(
            math.isfinite(value)
            for value in (object_station_m, ego_station_m)
        ):
            return None

        try:
            boundary_station_m = nominal_vehicle_avoidance_approach_station_m(
                object_station_m,
            )
        except ValueError:
            return None
        return max(0.0, boundary_station_m - ego_station_m)

    def _pedestrian_signed_constraint_clearance_m(
        self,
        *,
        index: int | None = None,
    ) -> float | None:
        """Return signed pedestrian clearance in the supervisor QP coordinate."""

        constraint_ego_station_m = (
            self.constraint_ego_station_m
            if self.constraint_ego_station_m is not None
            else self.ego_station_m
        )
        try:
            if index is None:
                pedestrian_parameters = self.pedestrian_parameters
                if pedestrian_parameters is None:
                    return None
                pedestrian_station_m = float(pedestrian_parameters[0])
            else:
                pedestrian_station_m = float((self.object_stations_m or ())[index])
            ego_station_m = float(
                cast(float | int | str, constraint_ego_station_m)
            )
        except (IndexError, TypeError, ValueError):
            return None
        if not all(
            math.isfinite(value)
            for value in (pedestrian_station_m, ego_station_m)
        ):
            return None
        try:
            boundary_station_m = nominal_pedestrian_avoidance_approach_station_m(
                pedestrian_station_m,
            )
        except ValueError:
            return None
        return boundary_station_m - ego_station_m


@dataclass(frozen=True)
class RewardCalculationResult:
    """Report the result of reward calculation."""

    reward: float
    actual_acceleration_mps2: float


@dataclass(frozen=True)
class RewardService:
    """Calculate Susan rewards and publish their typed metric series."""

    state: RewardStatePort
    math: RewardMathPort

    def calculate(
        self,
        request: RewardCalculationRequest,
    ) -> RewardCalculationResult:
        """Return the calculate value owned by this reward service."""
        actual_acceleration = self.math._calculate_actual_acceleration(
            request.previous_speed_mps,
            request.current_speed_mps,
            request.actual_acceleration_override_mps2,
        )
        previous_command = self.state.controller_state.last_supervised_cmd_accel
        previous_delta = self.state.controller_state.last_supervised_delta
        sync_command = (
            previous_command
            if request.actual_acceleration_override_mps2 is not None
            else request.supervised_command_mps2
        )
        self._update_sync_state(request, actual_acceleration, sync_command)
        blocking_obstacle = request.blocking_obstacle
        blocking_distance_m = (
            blocking_obstacle.distance_m
            if blocking_obstacle is not None
            else None
        )
        blocking_obstacle_speed_mps = (
            blocking_obstacle.speed_mps
            if blocking_obstacle is not None
            else None
        )
        blocking_closing_speed_mps = (
            blocking_obstacle.closing_speed_mps
            if blocking_obstacle is not None
            else None
        )
        blocking_distance_is_constraint_clearance = bool(
            blocking_obstacle is not None
            and blocking_obstacle.distance_is_constraint_clearance
        )
        distance_to_stopline = request.distance_to_stopline_m
        if distance_to_stopline is None:
            distance_to_stopline = self.math._get_distance_to_stopline(
                request.ego_station_m,
            )

        stop_profile_context = (
            self.math._uses_stop_profile(
                request.traffic_light_state_code,
                request.upcoming_intersection_type_code,
            )
            or request.red_light_overdrive
        )
        blocking_owns_profile = bool(
            blocking_obstacle is not None
            and not request.red_light_overdrive
            and (
                distance_to_stopline is None
                or blocking_obstacle.distance_m < distance_to_stopline
            )
        )
        if stop_profile_context and not blocking_owns_profile:
            self.math._clear_sync_component()
        else:
            self.math._calculate_sync_component(
                actual_acceleration,
                sync_command,
                request.current_speed_mps,
                request.road_speed_limit_mps,
            )
        self._calculate_components(
            request,
            previous_command,
            previous_delta,
            distance_to_stopline,
            blocking_distance_m,
            blocking_obstacle_speed_mps,
            blocking_distance_is_constraint_clearance,
            blocking_closing_speed_mps,
        )
        reward = self.math._base_reward_sum()
        self.math._update_reward_memory(
            reward,
            request.supervised_command_mps2,
            request.applied_delta_mps2,
            request.ego_station_m,
            request.traffic_light_state_code,
            request.upcoming_intersection_type_code,
        )
        return RewardCalculationResult(reward, actual_acceleration)

    def _update_sync_state(
        self,
        request: RewardCalculationRequest,
        actual_acceleration_mps2: float,
        sync_command_mps2: float,
    ) -> None:
        """Publish observed-lead and SUMO-intervention telemetry for this step."""

        lead_present = has_valid_ego_lead_telemetry(
            request.ego_lead_vehicle_present,
            request.ego_lead_distance_m,
        )
        sync_state = self.state.reward_state.sync
        sync_state.lead_vehicle_present = float(lead_present)
        sync_state.sumo_intervention = float(
            request.actual_acceleration_override_mps2 is not None
            and abs(actual_acceleration_mps2 - sync_command_mps2)
            > SYNC_ERROR_DEADBAND_MPS2
        )

    def _calculate_components(
        self,
        request: RewardCalculationRequest,
        previous_command_mps2: float,
        previous_delta_mps2: float,
        distance_to_stopline_m: float | None,
        blocking_distance_m: float | None,
        blocking_obstacle_speed_mps: float | None,
        blocking_distance_is_constraint_clearance: bool,
        blocking_closing_speed_mps: float | None,
    ) -> None:
        """Calculate the components owned by this reward service."""
        self.math._calculate_accel_component(
            request.supervised_command_mps2,
            previous_command_mps2,
            request.current_speed_mps,
        )
        policy_action_mps2 = (
            request.supervised_command_mps2
            if request.policy_action_mps2 is None
            else request.policy_action_mps2
        )
        self.math._calculate_speed_component(
            request.current_speed_mps,
            distance_to_stopline=distance_to_stopline_m,
            traffic_light_state_code=request.traffic_light_state_code,
            upcoming_intersection_type_code=(
                request.upcoming_intersection_type_code
            ),
            road_speed_limit=request.road_speed_limit_mps,
            red_light_overdrive=request.red_light_overdrive,
            blocking_distance_m=blocking_distance_m,
            blocking_obstacle_speed_mps=blocking_obstacle_speed_mps,
            blocking_distance_is_constraint_clearance=(
                blocking_distance_is_constraint_clearance
            ),
            policy_action_mps2=policy_action_mps2,
            blocking_closing_speed_mps=blocking_closing_speed_mps,
        )
        self.math._apply_stationary_signal_positive_accel_penalty(
            policy_action_mps2,
            request.current_speed_mps,
            distance_to_stopline_m,
            request.traffic_light_state_code,
            request.upcoming_intersection_type_code,
            request.road_speed_limit_mps,
            request.red_light_overdrive,
        )
        self.math._calculate_delta_component(
            request.applied_delta_mps2,
            previous_delta_mps2,
        )
        self.math._update_reward_penalty_averages(request.step_counter)
        self.math._apply_reward_weights()

    def telemetry(self) -> RewardTelemetry:
        """Return the telemetry value owned by this reward service."""
        return RewardTelemetry.from_state(self.state.reward_state)

    def component_series(self) -> dict[str, float]:
        """Return the component series value owned by this reward service."""
        series = self.telemetry().to_flat_series()
        diagnostics = self.state.reward_state.speed.diagnostics
        sync = self.state.reward_state.sync
        lead_present = bool(sync.lead_vehicle_present)
        series.update({
            "red_light_speed_reward": diagnostics.red_light_speed_reward,
            "red_light_profile_reward": diagnostics.red_light_profile_reward,
            "red_light_stopline_factor": diagnostics.red_light_stopline_factor,
            "red_light_target_speed": diagnostics.red_light_target_speed,
            "red_light_preferred_speed": diagnostics.red_light_preferred_speed,
            "red_light_distance_to_stopline": (
                diagnostics.red_light_distance_to_stopline
            ),
            "red_light_far_wait_penalty": diagnostics.red_light_far_wait_penalty,
            "green_light_idle_penalty": diagnostics.green_light_idle_penalty,
            "control_policy_action": self.state.policy_action,
            "control_supervised_cmd_accel_delta": (
                self.state.supervised_cmd_accel_delta
            ),
            "control_abs_supervised_cmd_accel_delta": (
                self.state.abs_supervised_cmd_accel_delta
            ),
            "control_supervised_cmd_accel_delta_change": (
                self.state.supervised_cmd_accel_delta_change
            ),
            "control_abs_supervised_cmd_accel_delta_change": (
                self.state.abs_supervised_cmd_accel_delta_change
            ),
            "control_supervised_cmd_accel_local_jitter": (
                self.state.supervised_cmd_accel_local_jitter
            ),
            "control_supervised_cmd_accel_local_jitter_sign_change": (
                self.state.supervised_cmd_accel_local_jitter_sign_change
            ),
            "control_supervised_cmd_accel_local_effective_jitter": (
                self.state.supervised_cmd_accel_local_effective_jitter
            ),
            "control_supervised_cmd_accel_trend_jitter": (
                self.state.supervised_cmd_accel_trend_jitter
            ),
            "control_supervised_cmd_accel_trend_jitter_sign_change": (
                self.state.supervised_cmd_accel_trend_jitter_sign_change
            ),
            "control_supervised_cmd_accel_trend_effective_jitter": (
                self.state.supervised_cmd_accel_trend_effective_jitter
            ),
            "control_supervised_cmd_accel_comfort_jitter": (
                self.state.supervised_cmd_accel_comfort_jitter
            ),
            "control_supervised_cmd_accel_weighted_comfort_jitter": (
                self.state.supervised_cmd_accel_weighted_comfort_jitter
            ),
            "control_supervised_cmd_accel_comfort_jitter_sign_change": (
                self.state.supervised_cmd_accel_comfort_jitter_sign_change
            ),
            "control_supervised_cmd_accel_comfort_effective_jitter": (
                self.state.supervised_cmd_accel_comfort_effective_jitter
            ),
            "control_supervised_cmd_accel_discount_jitter": (
                self.state.supervised_cmd_accel_discount_jitter
            ),
            "control_normalized_supervised_cmd_accel_discount_jitter": (
                self.state.normalized_supervised_cmd_accel_discount_jitter
            ),
            "speed_pre_jitter_discount_effective": (
                self.state.pre_jitter_discount_speed_reward
            ),
            "speed_reward_jitter_discount": (
                self.state.speed_reward_jitter_discount
            ),
            "speed_reward_jitter_discount_applied": (
                self.state.speed_reward_jitter_discount_applied
            ),
        })
        series.update(self._lead_component_series(lead_present))
        return series

    def _lead_component_series(self, lead_present: bool) -> dict[str, float]:
        """Project mutually exclusive lead-presence diagnostic channels."""

        sync = self.state.reward_state.sync
        jitter = self.state.supervised_cmd_accel_local_jitter
        inactive = math.nan
        return {
            "sync_lead_present_absolute_error": (
                sync.absolute_error if lead_present else inactive
            ),
            "sync_lead_absent_absolute_error": (
                inactive if lead_present else sync.absolute_error
            ),
            "control_lead_present_local_jitter": (
                jitter if lead_present else inactive
            ),
            "control_lead_absent_local_jitter": (
                inactive if lead_present else jitter
            ),
            "sync_lead_present_sumo_intervention": (
                sync.sumo_intervention if lead_present else inactive
            ),
            "sync_lead_absent_sumo_intervention": (
                inactive if lead_present else sync.sumo_intervention
            ),
        }
