# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Pure observation-feature assembly for Susan's controller."""

import math
from dataclasses import dataclass
from typing import Protocol

from lanista.reinforcengine.utils.simulation.intersection_metadata import (
    INTERSECTION_TYPE_TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_STATE_GREEN,
    TRAFFIC_LIGHT_STATE_NOT_APPLICABLE,
    TRAFFIC_LIGHT_STATE_RED,
    TRAFFIC_LIGHT_STATE_RED_YELLOW,
    TRAFFIC_LIGHT_STATE_YELLOW,
)

from .observation_config import (
    OBSERVATION_STATION_NORMALIZATION_METERS,
    OBSERVATION_STATION_PRESENT_THRESHOLD,
)


class ObservationFeatureMathPort(Protocol):
    """Numerical operations needed by observation feature shaping."""

    def _reward_speed_limit(self, road_speed_limit: float | None = None) -> float:
        """Return the reward speed limit value provided by this observation feature math
port."""
        ...

    def _uses_stop_profile(
        self,
        traffic_light_state_code: int | float | None,
        upcoming_intersection_type_code: int | float | None,
    ) -> bool:
        """Return whether the distance-based stop profile applies."""
        ...

    def _red_light_brake_horizon(
        self,
        road_speed_limit: float | None = None,
    ) -> float:
        """Return the red light brake horizon value provided by this observation feature math port.
        """
        ...

    def _signal_reward_curve_distance(
        self,
        distance_to_stopline: float,
        road_speed_limit: float | None = None,
    ) -> float:
        """Map the true signal distance onto the reward curve."""
        ...

    def _signal_reward_horizon(
        self,
        road_speed_limit: float | None = None,
    ) -> float:
        """Return the signal reward horizon including its safety margin."""
        ...

    def _preferred_brake_curve_speed(
        self,
        effective_distance: float,
        road_speed_limit: float | None = None,
    ) -> float:
        """Return the preferred brake curve speed value provided by this observation feature math
        port.
        """
        ...

    def _brake_curve_speed(
        self,
        effective_distance: float,
        road_speed_limit: float | None = None,
    ) -> float:
        """Return the brake curve speed value provided by this observation feature math
port."""
        ...


def _safe_float(value: float | int | None, default: float) -> float:
    """Return the safe float value owned by this observation feature service module."""
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clip(value: float, lower: float, upper: float) -> float:
    """Return the clip value owned by this observation feature service module."""
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class ObservationFeatureService:
    """Calculate normalized object and traffic-light observation features."""

    math: ObservationFeatureMathPort

    def object_slot_features(
        self,
        station_value: float | int | None,
        speed_value: float | int | None,
        road_speed_limit: float | None = None,
    ) -> tuple[float, float, float]:
        """Return the object slot features value owned by this observation feature service."""
        station = _safe_float(station_value, 0.0)
        speed = max(0.0, _safe_float(speed_value, 0.0))
        present = 1.0 if station > OBSERVATION_STATION_PRESENT_THRESHOLD else 0.0
        if present <= 0.0:
            return 0.0, 0.0, 0.0
        station_norm = _clip(
            station / max(float(OBSERVATION_STATION_NORMALIZATION_METERS), 1e-6), -1.0, 1.0
        )
        speed_norm = _clip(
            speed / self.math._reward_speed_limit(road_speed_limit), 0.0, 1.5
        )
        return present, station_norm, speed_norm

    def red_light_features(
        self,
        current_speed: float | int | None,
        distance_to_stopline: float | int | None,
        road_speed_limit: float | None = None,
        traffic_light_state_code: int | float | None = None,
        upcoming_intersection_type_code: int | float | None = None,
    ) -> dict[str, float]:
        """Return the red/yellow stop-profile observation features."""
        if not self.math._uses_stop_profile(
            traffic_light_state_code, upcoming_intersection_type_code
        ):
            return {
                "red_light_distance_norm": 0.0,
                "red_light_preferred_speed_norm": 0.0,
                "red_light_brake_limit_speed_norm": 0.0,
                "speed_error_to_preferred_norm": 0.0,
                "speed_error_to_brake_limit_norm": 0.0,
                "red_brake_horizon_active": 0.0,
            }
        speed_limit = self.math._reward_speed_limit(road_speed_limit)
        distance = max(0.0, _safe_float(distance_to_stopline, 0.0))
        preferred_curve_distance = self.math._signal_reward_curve_distance(
            distance,
            speed_limit,
        )
        current = max(0.0, _safe_float(current_speed, 0.0))
        active_horizon = self.math._signal_reward_horizon(speed_limit)
        preferred_speed = self.math._preferred_brake_curve_speed(
            preferred_curve_distance, speed_limit
        )
        brake_limit_speed = self.math._brake_curve_speed(
            distance,
            speed_limit,
        )
        brake_horizon_active = 1.0 if distance <= active_horizon else 0.0
        brake_error = (
            _clip((current - brake_limit_speed) / speed_limit, -1.5, 1.5)
            if brake_horizon_active > 0.0 else 0.0
        )
        return {
            "red_light_distance_norm": _clip(
                distance / max(active_horizon, 1e-6), 0.0, 1.5
            ),
            "red_light_preferred_speed_norm": _clip(
                preferred_speed / speed_limit, 0.0, 1.5
            ),
            "red_light_brake_limit_speed_norm": _clip(
                brake_limit_speed / speed_limit, 0.0, 1.5
            ),
            "speed_error_to_preferred_norm": _clip(
                (current - preferred_speed) / speed_limit, -1.5, 1.5
            ),
            "speed_error_to_brake_limit_norm": brake_error,
            "red_brake_horizon_active": brake_horizon_active,
        }

    @staticmethod
    def traffic_light_flag_features(
        traffic_light_state_code: int | float | None,
        upcoming_intersection_type_code: int | float | None,
    ) -> dict[str, float]:
        """Return the traffic light flag features value owned by this observation feature service.
        """
        try:
            intersection_type = (
                int(upcoming_intersection_type_code)
                if upcoming_intersection_type_code is not None
                else None
            )
        except (TypeError, ValueError):
            intersection_type = None
        applicable = 1.0 if intersection_type == INTERSECTION_TYPE_TRAFFIC_LIGHT else 0.0
        try:
            state = (
                int(traffic_light_state_code)
                if applicable and traffic_light_state_code is not None
                else TRAFFIC_LIGHT_STATE_NOT_APPLICABLE
            )
        except (TypeError, ValueError):
            state = TRAFFIC_LIGHT_STATE_NOT_APPLICABLE
        return {
            "tl_is_red": 1.0 if state == TRAFFIC_LIGHT_STATE_RED else 0.0,
            "tl_is_yellow": 1.0 if state == TRAFFIC_LIGHT_STATE_YELLOW else 0.0,
            "tl_is_red_yellow": 1.0 if state == TRAFFIC_LIGHT_STATE_RED_YELLOW else 0.0,
            "tl_is_green": 1.0 if state == TRAFFIC_LIGHT_STATE_GREEN else 0.0,
            "tl_is_applicable": applicable,
        }
