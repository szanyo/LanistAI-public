# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed observation formatting, history, and metric publication for Susan."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, cast

import numpy as np

from lanista.reinforcengine.rl_metrics.observation_contract import (
    TRAFFIC_LIGHT_TELEMETRY_OBSERVATION_KEYS as TRAFFIC_LIGHT_OBSERVATION_KEYS,
)

from ..control.traffic_light_timing_state import TrafficLightTimingDiagnostics
from .observation_config import (
    OBSERVATION_HISTORY_KEYS,
    OBSERVATION_SCALAR_KEYS,
    OBSERVATION_STATION_NORMALIZATION_METERS,
)
from .observation_feature_service import ObservationFeatureService
from .vehicle_telemetry import VehicleObservationTelemetry
from .vehicle_telemetry_container import (
    VehicleObservationTelemetryContainer,
    VehicleTelemetryContainer,
)


class ObservationMathPort(Protocol):
    """Math boundary required while formatting one policy observation."""

    def _reward_speed_limit(self, road_speed_limit: float | None = None) -> float:
        """Return the reward speed limit value provided by this observation math port."""
        ...


class TrafficLightObservationPort(Protocol):
    """Traffic-light features owned by the timing service."""

    def timing_diagnostics(self) -> TrafficLightTimingDiagnostics:
        """Return the timing diagnostics value provided by this traffic light observation
port."""
        ...

    def timing_observation_features(
        self,
        upcoming_intersection_type_code: int | None,
        traffic_light_state_code: int,
    ) -> dict[str, float]:
        """Return the timing observation features value provided by this traffic light observation
        port.
        """
        ...

    def context_diagnostic_features(self) -> dict[str, float]:
        """Return the context diagnostic features value provided by this traffic light observation
        port.
        """
        ...


@dataclass(frozen=True)
class ObservationFormattingRequest:
    """Unit-explicit inputs needed to build one Susan observation."""

    ego_station_m: float
    object_stations_m: tuple[float, ...]
    ego_speed_mps: float
    object_speeds_mps: tuple[float, ...]
    applied_command_mps2: float
    supervised_command_mps2: float
    actual_acceleration_mps2: float
    observation_road_speed_limit_mps: float
    traffic_light_state_code: int
    upcoming_intersection_type_code: int | None
    distance_to_stopline_m: float | None = None
    reward_road_speed_limit_mps: float | None = None
    ego_lead_vehicle_present: bool = False
    ego_lead_distance_m: float = 0.0
    ego_lead_speed_mps: float = 0.0


@dataclass(frozen=True)
class FormattedObservation:
    """Immutable view of one ordered observation mapping."""

    values: Mapping[str, float]


@dataclass(frozen=True)
class _ObjectSensorValues:
    """Carry the data required for object sensor values."""

    stations_m: tuple[float, float, float]
    speeds_mps: tuple[float, float, float]


def _safe_float(value: float | int | None, default: float) -> float:
    """Return the safe float value owned by this observation service module."""
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clip(value: float, lower: float, upper: float) -> float:
    """Return the clip value owned by this observation service module."""
    return max(lower, min(upper, float(value)))


def _safe_sensor_value(values: tuple[float, ...], index: int) -> float:
    """Return the safe sensor value value owned by this observation service module."""
    return float(values[index]) if index < len(values) else 0.0


@dataclass(frozen=True)
class ObservationService:
    """Own Susan observation formatting, history, and metric series."""

    history: VehicleObservationTelemetryContainer
    telemetry: VehicleTelemetryContainer
    features: ObservationFeatureService
    math: ObservationMathPort
    traffic_lights: TrafficLightObservationPort

    def format(self, request: ObservationFormattingRequest) -> FormattedObservation:
        """Format this observation service's value for presentation."""
        sensors = self._sensor_values(request)
        speed_limit_mps = self.math._reward_speed_limit(
            request.reward_road_speed_limit_mps
            or request.observation_road_speed_limit_mps
        )
        values = self._base_values(request, sensors, speed_limit_mps)
        values.update(self._object_values(sensors, speed_limit_mps))
        values.update(self._post_object_values(request, speed_limit_mps))
        values.update(
            self.features.red_light_features(
                request.ego_speed_mps,
                request.distance_to_stopline_m,
                request.reward_road_speed_limit_mps
                or request.observation_road_speed_limit_mps,
                request.traffic_light_state_code,
                request.upcoming_intersection_type_code,
            )
        )
        values.update(
            self.features.traffic_light_flag_features(
                request.traffic_light_state_code,
                request.upcoming_intersection_type_code,
            )
        )
        timing = self.traffic_lights.timing_observation_features(
            request.upcoming_intersection_type_code,
            request.traffic_light_state_code,
        )
        values["tl_phase_remaining_norm"] = float(
            timing.get("tl_phase_remaining_norm", 0.0)
        )
        values["upcoming_intersection_type"] = float(
            cast(int, request.upcoming_intersection_type_code)
        )
        values["u1_supervised_command"] = float(
            request.supervised_command_mps2
        )
        return FormattedObservation(MappingProxyType(values))

    @staticmethod
    def _sensor_values(
        request: ObservationFormattingRequest,
    ) -> _ObjectSensorValues:
        """Return the sensor values value owned by this observation service."""
        stations = tuple(
            np.asarray(request.object_stations_m, dtype=float).reshape(-1)
        )
        speeds = tuple(
            np.asarray(request.object_speeds_mps, dtype=float).reshape(-1)
        )
        return _ObjectSensorValues(
            stations_m=(
                _safe_sensor_value(stations, 0),
                _safe_sensor_value(stations, 1),
                _safe_sensor_value(stations, 2),
            ),
            speeds_mps=(
                _safe_sensor_value(speeds, 0),
                _safe_sensor_value(speeds, 1),
                _safe_sensor_value(speeds, 2),
            ),
        )

    def _base_values(
        self,
        request: ObservationFormattingRequest,
        sensors: _ObjectSensorValues,
        speed_limit_mps: float,
    ) -> dict[str, float]:
        """Return the base values value owned by this observation service."""
        lead_present = 1.0 if request.ego_lead_vehicle_present else 0.0
        lead_distance_m = max(0.0, _safe_float(request.ego_lead_distance_m, 0.0))
        lead_speed_mps = max(0.0, _safe_float(request.ego_lead_speed_mps, 0.0))
        station_scale = max(
            float(OBSERVATION_STATION_NORMALIZATION_METERS),
            1e-6,
        )
        return {
            "ego_station": float(request.ego_station_m),
            "station_1": sensors.stations_m[0],
            "station_2": sensors.stations_m[1],
            "station_3": sensors.stations_m[2],
            "ego_speed": float(request.ego_speed_mps),
            "speed_1": sensors.speeds_mps[0],
            "speed_2": sensors.speeds_mps[1],
            "speed_3": sensors.speeds_mps[2],
            "ego_station_norm": _clip(
                float(request.ego_station_m) / station_scale,
                -1.0,
                1.0,
            ),
            "ego_lead_vehicle_present": lead_present,
            "ego_lead_distance_norm": (
                _clip(lead_distance_m / station_scale, 0.0, 1.5)
                if lead_present > 0.0
                else 0.0
            ),
            "ego_lead_speed_norm": (
                _clip(lead_speed_mps / speed_limit_mps, 0.0, 1.5)
                if lead_present > 0.0
                else 0.0
            ),
        }

    def _post_object_values(
        self,
        request: ObservationFormattingRequest,
        speed_limit_mps: float,
    ) -> dict[str, float]:
        """Return the post object values value owned by this observation service."""
        return {
            "ego_speed_norm": _clip(
                float(request.ego_speed_mps) / speed_limit_mps,
                0.0,
                1.5,
            ),
            "u1_applied_command": float(request.applied_command_mps2),
            "u1_actual_acceleration": float(request.actual_acceleration_mps2),
            "road_speed_limit": float(
                request.observation_road_speed_limit_mps
            ),
            "road_speed_limit_norm": _clip(
                float(request.observation_road_speed_limit_mps)
                / self.math._reward_speed_limit(None),
                0.0,
                1.5,
            ),
        }

    def _object_values(
        self,
        sensors: _ObjectSensorValues,
        speed_limit_mps: float,
    ) -> dict[str, float]:
        """Return the object values value owned by this observation service."""
        values: dict[str, float] = {}
        for slot, (station_m, speed_mps) in enumerate(
            zip(sensors.stations_m, sensors.speeds_mps, strict=True),
            start=1,
        ):
            present, station_norm, speed_norm = self.features.object_slot_features(
                station_m,
                speed_mps,
                speed_limit_mps,
            )
            values[f"object_{slot}_present"] = present
            values[f"object_{slot}_station_norm"] = station_norm
            values[f"object_{slot}_speed_norm"] = speed_norm
        return values

    def append(self, observation: Mapping[str, float]) -> None:
        """Append the operation owned by this observation service."""
        self.history.append(dict(observation))

    def extend_with_history(
        self,
        current_observation: Mapping[str, float],
    ) -> dict[str, np.ndarray]:
        """Return the extend with history value owned by this observation service."""
        if not self.history.latest() and current_observation:
            self.history.append(dict(current_observation))
        full_observation = self.history.to_time_window_dict()
        policy_keys = tuple(OBSERVATION_HISTORY_KEYS) + tuple(
            OBSERVATION_SCALAR_KEYS
        )
        return {
            key: full_observation[key]
            for key in policy_keys
            if key in full_observation
        }

    def component_series(self) -> dict[str, float]:
        """Return the component series value owned by this observation service."""
        latest_observation = self.history.latest()
        if latest_observation is None:
            latest_telemetry = self.telemetry.latest()
            latest_observation = (
                latest_telemetry.observation
                if latest_telemetry is not None
                else None
            )

        values = {key: 0.0 for key in TRAFFIC_LIGHT_OBSERVATION_KEYS}
        if latest_observation is not None:
            observation_values = latest_observation.to_telemetry_dict()
            values.update({
                key: observation_values[key]
                for key in TRAFFIC_LIGHT_OBSERVATION_KEYS
            })

        diagnostics = self.traffic_lights.timing_diagnostics()
        values.update({
            "tl_phase_remaining_delta_norm": (
                diagnostics.phase_remaining_delta_norm
            ),
            "tl_phase_remaining_raw_delta_norm": (
                diagnostics.phase_remaining_raw_delta_norm
            ),
            "tl_timing_context_changed": diagnostics.context_changed,
            "tl_phase_remaining_monotonic_violation": (
                diagnostics.phase_remaining_monotonic_violation
            ),
        })
        values.update(self.traffic_lights.context_diagnostic_features())
        return values

    def telemetry_observation(
        self,
        observation: Mapping[str, float],
    ) -> VehicleObservationTelemetry:
        """Return the telemetry observation value owned by this observation service."""
        values = dict(observation)
        values.update(self.component_series())
        return VehicleObservationTelemetry.from_dict(values)
