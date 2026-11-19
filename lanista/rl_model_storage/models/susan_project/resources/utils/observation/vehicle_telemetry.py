# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed observation, controller-state, and per-step telemetry payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from lanista.reinforcengine.rl_metrics.observation_contract import (
    LEGACY_TELEMETRY_OBSERVATION_KEYS as LEGACY_OBSERVATION_KEYS,
)
from lanista.reinforcengine.rl_metrics.observation_contract import (
    TELEMETRY_OBSERVATION_KEYS,
)
from lanista.reinforcengine.rl_metrics.observation_contract import (
    TRAFFIC_LIGHT_TELEMETRY_OBSERVATION_KEYS as TRAFFIC_LIGHT_OBSERVATION_KEYS,
)
from lanista.reinforcengine.utils.simulation.ports import SimulationReadQuality

from .observation_config import (
    OBSERVATION_CHANNEL_KEYS as OBSERVATION_CHANNEL_DESCRIPTOR_KEYS,
)
from .observation_config import (
    OBSERVATION_KEYS as POLICY_OBSERVATION_KEYS,
)
from ..reward.reward_telemetry import (
    RewardComponentTelemetry,
    RewardConfigTelemetry,
    RewardStateTelemetry,
    RewardTelemetry,
    RewardTelemetryChannelPayload,
)

_LEGACY_OBSERVATION_KEYS_DOCUMENTATION = (
    "ego_station",
    "station_1",
    "station_2",
    "station_3",
    "ego_speed",
    "speed_1",
    "speed_2",
    "speed_3",
    "ego_station_norm",
    "ego_lead_vehicle_present",
    "ego_lead_distance_norm",
    "ego_lead_speed_norm",
    "object_1_present",
    "object_1_station_norm",
    "object_1_speed_norm",
    "object_2_present",
    "object_2_station_norm",
    "object_2_speed_norm",
    "object_3_present",
    "object_3_station_norm",
    "object_3_speed_norm",
    "ego_speed_norm",
    "u1_applied_command",
    "u1_actual_acceleration",
    "road_speed_limit",
    "road_speed_limit_norm",
    "red_light_distance_norm",
    "red_light_preferred_speed_norm",
    "red_light_brake_limit_speed_norm",
    "speed_error_to_preferred_norm",
    "speed_error_to_brake_limit_norm",
    "red_brake_horizon_active",
    "upcoming_intersection_type",
    "u1_supervised_command",
)

_TRAFFIC_LIGHT_OBSERVATION_KEYS_DOCUMENTATION = (
    "tl_is_red",
    "tl_is_yellow",
    "tl_is_red_yellow",
    "tl_is_green",
    "tl_is_applicable",
    "tl_phase_remaining_norm",
    "tl_phase_remaining_delta_norm",
    "tl_phase_remaining_raw_delta_norm",
    "tl_timing_context_changed",
    "tl_phase_remaining_monotonic_violation",
    "tl_tls_id_numeric",
    "tl_link_index",
    "tl_intersection_number",
)

TELEMETRY_OBSERVATION_EXTRA_KEYS = tuple(
    key for key in TELEMETRY_OBSERVATION_KEYS if key not in POLICY_OBSERVATION_KEYS
)

OBSERVATION_KEYS = TELEMETRY_OBSERVATION_KEYS
OBSERVATION_CHANNEL_KEYS = OBSERVATION_CHANNEL_DESCRIPTOR_KEYS
OBSERVATION_CHANNEL_NAMES = ("observation", "telemetry", "reward")
TELEMETRY_CHANNEL_KEYS = TELEMETRY_OBSERVATION_EXTRA_KEYS
SIMULATOR_READ_FALLBACK_TELEMETRY_KEY = "simulator_read_fallback"


class VehicleObservationChannels(TypedDict):
    """Describe the typed mapping for vehicle observation channels."""

    observation: dict[str, float]
    telemetry: dict[str, float]


class VehicleTelemetryChannels(TypedDict):
    """Describe the typed mapping for vehicle telemetry channels."""

    observation: dict[str, float]
    telemetry: dict[str, float]
    reward: RewardTelemetryChannelPayload


@dataclass
class VehicleObservationTelemetry:
    """Carry the data required for vehicle observation telemetry."""

    ego_station: float = 0.0
    station_1: float = 0.0
    station_2: float = 0.0
    station_3: float = 0.0
    ego_speed: float = 0.0
    speed_1: float = 0.0
    speed_2: float = 0.0
    speed_3: float = 0.0
    ego_station_norm: float = 0.0
    ego_lead_vehicle_present: float = 0.0
    ego_lead_distance_norm: float = 0.0
    ego_lead_speed_norm: float = 0.0
    object_1_present: float = 0.0
    object_1_station_norm: float = 0.0
    object_1_speed_norm: float = 0.0
    object_2_present: float = 0.0
    object_2_station_norm: float = 0.0
    object_2_speed_norm: float = 0.0
    object_3_present: float = 0.0
    object_3_station_norm: float = 0.0
    object_3_speed_norm: float = 0.0
    ego_speed_norm: float = 0.0
    u1_applied_command: float = 0.0
    u1_actual_acceleration: float = 0.0
    road_speed_limit: float = 0.0
    road_speed_limit_norm: float = 0.0
    red_light_distance_norm: float = 0.0
    red_light_preferred_speed_norm: float = 0.0
    red_light_brake_limit_speed_norm: float = 0.0
    speed_error_to_preferred_norm: float = 0.0
    speed_error_to_brake_limit_norm: float = 0.0
    red_brake_horizon_active: float = 0.0
    tl_is_red: float = 0.0
    tl_is_yellow: float = 0.0
    tl_is_red_yellow: float = 0.0
    tl_is_green: float = 0.0
    tl_is_applicable: float = 0.0
    tl_phase_remaining_norm: float = 0.0
    tl_phase_remaining_delta_norm: float = 0.0
    tl_phase_remaining_raw_delta_norm: float = 0.0
    tl_timing_context_changed: float = 0.0
    tl_phase_remaining_monotonic_violation: float = 0.0
    tl_tls_id_numeric: float = -1.0
    tl_link_index: float = -1.0
    tl_intersection_number: float = -1.0
    upcoming_intersection_type: float = 0.0
    u1_supervised_command: float = 0.0

    @classmethod
    def zero(cls) -> VehicleObservationTelemetry:
        """Return the zero value owned by this vehicle observation telemetry."""
        return cls()

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> VehicleObservationTelemetry:
        """Create this vehicle observation telemetry's value from dict."""
        return cls(**{key: float(values.get(key, 0.0)) for key in OBSERVATION_KEYS})

    def to_policy_dict(self) -> dict[str, float]:
        """Convert this vehicle observation telemetry's value to policy dict."""
        return {key: float(getattr(self, key)) for key in POLICY_OBSERVATION_KEYS}

    def to_telemetry_dict(self) -> dict[str, float]:
        """Convert this vehicle observation telemetry's value to telemetry dict."""
        return {key: float(getattr(self, key)) for key in TELEMETRY_OBSERVATION_KEYS}

    def to_channels(self) -> VehicleObservationChannels:
        """Convert this vehicle observation telemetry's value to channels."""
        return {
            "observation": self.to_policy_dict(),
            "telemetry": self.to_telemetry_dict(),
        }

    def to_telemetry_channel_dict(self) -> dict[str, float]:
        """Convert this vehicle observation telemetry's value to telemetry channel dict."""
        return {
            key: float(getattr(self, key))
            for key in TELEMETRY_CHANNEL_KEYS
        }

    def to_dict(self) -> dict[str, float]:
        """Convert this vehicle observation telemetry's value to dict."""
        return {key: float(getattr(self, key)) for key in OBSERVATION_KEYS}


@dataclass
class VehicleControllerTelemetryState:
    """Represent runtime state for vehicle controller telemetry."""

    last_supervised_cmd_accel: float = 0.0
    last_supervised_cmd_accel_delta: float = 0.0
    last_supervised_delta: float = 0.0
    last_sumo_speed: float | None = None
    applied_delta: float = 0.0
    prev_intersection_number: int = 0

    def reset_episode_values(self) -> None:
        """Reset the episode values owned by this vehicle controller telemetry state."""
        self.last_supervised_cmd_accel = 0.0
        self.last_supervised_cmd_accel_delta = 0.0
        self.last_supervised_delta = 0.0
        self.last_sumo_speed = None
        self.applied_delta = 0.0
        self.prev_intersection_number = 0


@dataclass
class VehicleTelemetry:
    """Carry the data required for vehicle telemetry."""

    step_counter: int
    observation: VehicleObservationTelemetry
    reward: RewardTelemetry
    simulator_read_quality: SimulationReadQuality = SimulationReadQuality.LIVE

    @property
    def simulator_read_fallback(self) -> float:
        """Project simulator quality as a numeric telemetry-only signal."""

        return float(
            self.simulator_read_quality is SimulationReadQuality.FALLBACK
        )

    def to_channels(self) -> VehicleTelemetryChannels:
        """Convert this vehicle telemetry's value to channels."""
        observation_channels = self.observation.to_channels()
        reward_channels = self.reward.to_channels()
        telemetry_channel = dict(observation_channels["telemetry"])
        telemetry_channel[SIMULATOR_READ_FALLBACK_TELEMETRY_KEY] = (
            self.simulator_read_fallback
        )
        return {
            "observation": observation_channels["observation"],
            "telemetry": telemetry_channel,
            "reward": reward_channels["reward"],
        }

    def to_channel_row(self) -> dict[str, float]:
        """Convert this vehicle telemetry's value to channel row."""
        row = {"step_counter": float(self.step_counter)}
        for key, value in self.observation.to_policy_dict().items():
            row[f"observation_{key}"] = float(value)
        for key, value in self.observation.to_telemetry_dict().items():
            row[f"telemetry_{key}"] = float(value)
        row[
            f"telemetry_{SIMULATOR_READ_FALLBACK_TELEMETRY_KEY}"
        ] = self.simulator_read_fallback
        for key, value in self.reward.to_flat_series().items():
            row[f"reward_{key}"] = float(value)
        return row

    def to_row(self) -> dict[str, float]:
        # Keep the legacy telemetry CSV column order stable:
        #   step_counter -> old observation columns -> reward_* columns
        # New traffic-light observation columns are appended after the legacy
        # metrics so older analysis scripts that use positional columns are less
        # likely to break.
        """Convert this vehicle telemetry's value to row."""
        row = {"step_counter": float(self.step_counter)}
        for key in LEGACY_OBSERVATION_KEYS:
            row[key] = float(getattr(self.observation, key))
        for key, value in self.reward.to_flat_series().items():
            row[f"reward_{key}"] = float(value)
        for key in TRAFFIC_LIGHT_OBSERVATION_KEYS:
            row[key] = float(getattr(self.observation, key))
        row[SIMULATOR_READ_FALLBACK_TELEMETRY_KEY] = (
            self.simulator_read_fallback
        )
        return row


__all__ = [
    "LEGACY_OBSERVATION_KEYS",
    "OBSERVATION_KEYS",
    "OBSERVATION_CHANNEL_KEYS",
    "OBSERVATION_CHANNEL_NAMES",
    "POLICY_OBSERVATION_KEYS",
    "SIMULATOR_READ_FALLBACK_TELEMETRY_KEY",
    "TELEMETRY_CHANNEL_KEYS",
    "TELEMETRY_OBSERVATION_EXTRA_KEYS",
    "TELEMETRY_OBSERVATION_KEYS",
    "TRAFFIC_LIGHT_OBSERVATION_KEYS",
    "VehicleObservationChannels",
    "RewardComponentTelemetry",
    "RewardConfigTelemetry",
    "RewardStateTelemetry",
    "RewardTelemetry",
    "VehicleTelemetryChannels",
    "VehicleControllerTelemetryState",
    "VehicleObservationTelemetry",
    "VehicleTelemetry",
]
