# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Susan policy-observation contract.

The policy receives two kinds of values:

- history fields: the last ``history_length`` samples are stacked in temporal
  order by ``VehicleObservationTelemetryContainer``;
- scalar fields: only the current sample is exposed.

Producer-side normalization happens in
``ObservationService.format`` and its injected feature/math services. The
bounds below are the Gym/RL space guardrails; some
distance-like guardrails are intentionally wider than the producer-side clamps
so existing checkpoints/configs keep accepting historical values. The comments
next to each key document the actual emitted value and normalization formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lanista.reinforcengine.rl_metrics.observation_contract import (
    POLICY_OBSERVATION_HISTORY_KEYS,
    POLICY_OBSERVATION_SCALAR_KEYS,
)

SusanObservationChannelName = Literal["history", "scalar"]


@dataclass(frozen=True)
class SusanObservationField:
    """Carry the data required for susan observation field."""

    key: str
    length: int
    bounds: tuple[float, float]
    channel: SusanObservationChannelName


@dataclass(frozen=True)
class SusanObservationDescriptor:
    """Describe susan observation data."""

    fields: tuple[SusanObservationField, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        """Return the keys exposed by this susan observation descriptor."""
        return tuple(field.key for field in self.fields)

    @property
    def key_lengths(self) -> dict[str, int]:
        """Return the key lengths exposed by this susan observation descriptor."""
        return {field.key: field.length for field in self.fields}

    @property
    def key_bounds(self) -> dict[str, tuple[float, float]]:
        """Return the key bounds exposed by this susan observation descriptor."""
        return {field.key: field.bounds for field in self.fields}

    @property
    def channel_keys(self) -> dict[SusanObservationChannelName, tuple[str, ...]]:
        """Return the channel keys exposed by this susan observation descriptor."""
        channels: dict[SusanObservationChannelName, list[str]] = {}
        for field in self.fields:
            channels.setdefault(field.channel, []).append(field.key)
        return {channel: tuple(keys) for channel, keys in channels.items()}

    @property
    def channel_names(self) -> tuple[SusanObservationChannelName, ...]:
        """Return the channel names exposed by this susan observation descriptor."""
        return tuple(self.channel_keys.keys())

    @property
    def size(self) -> int:
        """Return the size exposed by this susan observation descriptor."""
        return sum(field.length for field in self.fields)

    def length(self, key: str) -> int:
        """Return the length value owned by this susan observation descriptor."""
        return self.key_lengths[key]

    def bounds(self, key: str) -> tuple[float, float]:
        """Return the bounds value owned by this susan observation descriptor."""
        return self.key_bounds[key]

    def channel(self, key: str) -> str:
        """Return the channel value owned by this susan observation descriptor."""
        for field in self.fields:
            if field.key == key:
                return field.channel
        raise KeyError(key)


class SusanObservationConfig(BaseModel):
    """Validate and represent susan observation config data."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    history_length: int = 30
    station_sentinel_value: float = -200.0
    station_present_threshold: float = -199.0
    station_normalization_meters: float = 60.0
    lead_headway_seconds: float = 1.0
    red_light_preferred_speed_plateau_width_meters: float = 0.5
    red_light_far_wait_penalty_weight: float = 0.16
    red_light_far_wait_distance_width_meters: float = 2.0
    red_light_far_wait_slow_speed_scale_mps: float = 0.50
    red_light_max_negative_reward: float = -3.0
    red_light_release_crossing_reward_weight: float = 0.50
    signal_stop_speed_tolerance_mps: float = 0.001
    signal_stop_motion_entry_speed_mps: float = 0.05
    signal_stop_motion_exit_speed_mps: float = 0.20
    # Deliberate throttle while holding a prohibited signal should become
    # expensive before the policy can preload the next control interval.
    signal_stop_positive_accel_penalty_scale_mps2: float = 0.5
    green_light_idle_penalty_weight: float = 0.50
    green_light_idle_slow_speed_scale_mps: float = 1.00
    supervised_cmd_accel_delta_normalizer: float = 3.0
    supervised_cmd_accel_delta_change_normalizer: float = 3.0
    supervised_cmd_accel_sign_change_deadband_mps2: float = 0.0
    supervised_cmd_accel_sign_change_penalty_multiplier: float = 1.50
    speed_reward_jitter_discount_strength: float = 0.50
    speed_reward_jitter_discount_scale: float = 1.00
    speed_reward_jitter_discount_deadband: float = 0.15
    speed_reward_jitter_discount_floor: float = 0.70
    speed_reward_jitter_trend_window: int = 5
    speed_reward_jitter_comfort_window: int = 10
    reward_weight_sync: float = 0.08
    reward_weight_speed: float = 0.66
    reward_weight_delta: float = 0.18
    reward_weight_accel: float = 0.08

    observation_history_keys: tuple[str, ...] = POLICY_OBSERVATION_HISTORY_KEYS
    _observation_history_key_documentation: tuple[str, ...] = (
        # Ego speed: raw speed_out is SUMO/controller speed in m/s, effectively
        # expected around [0, speed_reward_v_max]. Policy value:
        # clip(speed_out / active_speed_limit, 0.0, 1.5).
        # On red approach, active_speed_limit is blended toward the safe speed.
        "ego_speed_norm",
        # Previous command interval acceleration [m/s^2], guardrail [-8, 8].
        # This is a physical value, not unit-normalized.
        "u1_applied_command",
        # SUMO-observed previous-interval acceleration [m/s^2], guardrail [-8, 8].
        "u1_actual_acceleration",
        # Active speed limit: raw road speed limit is m/s from the preprocessor,
        # with speed_reward_v_max fallback. Policy value:
        # clip(active_speed_limit / speed_reward_v_max, 0.0, 1.5).
        # speed_reward_v_max defaults to 8.3333 m/s.
        "road_speed_limit_norm",
        # Red/yellow stop-line distance: raw distance is
        # max(0, -signal_target_EGO_station), kept separate from both the
        # lead-following station and its object distance. Its zero is the QP
        # stationary front-bumper stop point. Policy value:
        # clip(distance_to_stopline / red_light_brake_horizon, 0.0, 1.5).
        # Emits 0.0 outside red traffic-light context.
        "red_light_distance_norm",
        # Speed error to brake-limit curve, only inside the active red/yellow braking
        # horizon: raw ego_speed - brake_limit_speed [m/s]. Policy value:
        # clip((ego_speed - brake_limit_speed) / speed_limit, -1.5, 1.5).
        # Emits 0.0 outside red traffic-light context or before the brake horizon.
        # History exposes whether the policy is running out of braking margin.
        "speed_error_to_brake_limit_norm",
        # Route-relevant traffic light is red: 1.0/0.0. History gives the
        # policy phase-continuity context instead of only a single-frame state.
        "tl_is_red",
        # Route-relevant traffic light is yellow: 1.0/0.0.
        "tl_is_yellow",
        # Route-relevant traffic light is red-yellow: 1.0/0.0.
        "tl_is_red_yellow",
        # Route-relevant traffic light is green: 1.0/0.0.
        "tl_is_green",
        # Upcoming target is a traffic-light intersection: 1.0/0.0.
        "tl_is_applicable",
        # Lead-vehicle mode flag from SUMO getLeader within the configured lookahead.
        # History stabilizes ego_station_norm interpretation across mode switches.
        "ego_lead_vehicle_present",
        # Route-relevant traffic-light state remaining time in [0, 1].
        # Pinned to the configured TLS/link context to avoid unrelated signal heads.
        "tl_phase_remaining_norm",
    )

    observation_scalar_keys: tuple[str, ...] = POLICY_OBSERVATION_SCALAR_KEYS
    _observation_scalar_key_documentation: tuple[str, ...] = (
        # Route station: traffic-light contexts use the shared reward/overdrive
        # boundary as zero; other intersections preserve the supervisor's raw
        # conflict-point origin so ego and object stations remain geometrically
        # consistent. A direct leader occupies an existing object slot and
        # does not replace this station coordinate.
        # Policy value: clip(EGO_station / station_normalization_meters, -1.0, 1.0).
        "ego_station_norm",
        # Preferred red braking curve speed in m/s. Raw value is in [0, speed_limit]
        # from brake_curve(max(distance - QP-tangent offset, 0)). Policy value:
        # clip(preferred_speed / speed_limit, 0.0, 1.5).
        # Kept scalar: it describes the current red-speed profile, but its
        # history would invite memorizing a computed speed curve.
        "red_light_preferred_speed_norm",
        # Brake-limit red curve speed in m/s. Raw value is in [0, speed_limit]
        # from the same signal-margin-adjusted curve. Policy value:
        # clip(brake_limit_speed / speed_limit, 0.0, 1.5).
        # Kept scalar for the current QP-tangent safety boundary only.
        "red_light_brake_limit_speed_norm",
        # Speed error to preferred curve: raw ego_speed - preferred_speed [m/s].
        # Policy value: clip((ego_speed - preferred_speed) / speed_limit, -1.5, 1.5).
        # Kept scalar because the model already has ego_speed history.
        "speed_error_to_preferred_norm",
        # Active stop-profile zone flag: 1.0 when the route-relevant traffic
        # light is red, yellow, or red-yellow and the ego vehicle is at or inside
        # the signal reward horizon, including its QP-tangent curve offset:
        # distance_to_stopline <= brake_horizon.
        # Else 0.0. This scalar tells the policy when the historized
        # speed_error_to_brake_limit_norm channel has physical braking meaning.
        "red_brake_horizon_active",
        # Object slot 1 presence: 1.0 if raw station_1 > station_present_threshold, else 0.0.
        "object_1_present",
        # Object slot 1 station: raw station uses -200 sentinel, -100 ignored
        # conflict/default, or per-intersection distance/following codes.
        # Policy value: clip(raw_station_1 / station_normalization_meters, -1.0, 1.0), else 0.0.
        "object_1_station_norm",
        # Object slot 1 speed: clip(raw_speed_1 / active_speed_limit, 0.0, 1.5), else 0.0.
        "object_1_speed_norm",
        # Object slot 2 presence: 1.0 if raw station_2 > station_present_threshold, else 0.0.
        "object_2_present",
        # Object slot 2 station: raw station uses -200 sentinel, -100 ignored
        # conflict/default, or per-intersection distance/following codes.
        # Policy value: clip(raw_station_2 / station_normalization_meters, -1.0, 1.0), else 0.0.
        "object_2_station_norm",
        # Object slot 2 speed: clip(raw_speed_2 / active_speed_limit, 0.0, 1.5), else 0.0.
        "object_2_speed_norm",
        # Object slot 3 presence: 1.0 if raw station_3 > station_present_threshold, else 0.0.
        "object_3_present",
        # Object slot 3 station: raw station uses -200 sentinel, -100 ignored
        # conflict/default, or per-intersection distance/following codes.
        # Policy value: clip(raw_station_3 / station_normalization_meters, -1.0, 1.0), else 0.0.
        "object_3_station_norm",
        # Object slot 3 speed: clip(raw_speed_3 / active_speed_limit, 0.0, 1.5), else 0.0.
        "object_3_speed_norm",
        # Compact intersection-type code. This is a route-context category, not a metric scale.
        "upcoming_intersection_type",
    )
    observation_default_bounds: tuple[float, float] = (-10.0, 10.0)
    observation_distance_bounds: tuple[float, float] = (-100.0, 10.0)
    observation_normalized_speed_bounds: tuple[float, float] = (0.0, 4.0)
    observation_command_bounds: tuple[float, float] = (-8.0, 8.0)
    observation_speed_error_bounds: tuple[float, float] = (-4.0, 4.0)
    observation_key_bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)

    @field_validator(
        "history_length",
        "speed_reward_jitter_trend_window",
        "speed_reward_jitter_comfort_window",
        mode="before",
    )
    @classmethod
    def _normalize_positive_int(cls, value: Any) -> int:
        """Normalize the positive int accepted by this susan observation config."""
        return max(1, int(value or 0))

    @field_validator(
        "station_sentinel_value",
        "station_present_threshold",
        "station_normalization_meters",
        "red_light_preferred_speed_plateau_width_meters",
        "red_light_far_wait_penalty_weight",
        "red_light_far_wait_distance_width_meters",
        "red_light_far_wait_slow_speed_scale_mps",
        "red_light_max_negative_reward",
        "signal_stop_speed_tolerance_mps",
        "signal_stop_motion_entry_speed_mps",
        "signal_stop_motion_exit_speed_mps",
        "signal_stop_positive_accel_penalty_scale_mps2",
        "green_light_idle_penalty_weight",
        "green_light_idle_slow_speed_scale_mps",
        "supervised_cmd_accel_delta_normalizer",
        "supervised_cmd_accel_delta_change_normalizer",
        "supervised_cmd_accel_sign_change_deadband_mps2",
        "supervised_cmd_accel_sign_change_penalty_multiplier",
        "speed_reward_jitter_discount_strength",
        "speed_reward_jitter_discount_scale",
        "speed_reward_jitter_discount_deadband",
        "speed_reward_jitter_discount_floor",
        "reward_weight_sync",
        "reward_weight_speed",
        "reward_weight_delta",
        "reward_weight_accel",
        mode="before",
    )
    @classmethod
    def _normalize_float(cls, value: Any) -> float:
        """Normalize the float accepted by this susan observation config."""
        return float(value)

    @field_validator("observation_history_keys", "observation_scalar_keys", mode="before")
    @classmethod
    def _normalize_key_tuple(cls, value: Any) -> tuple[str, ...]:
        """Normalize the key tuple accepted by this susan observation config."""
        if value is None:
            return ()
        return tuple(str(item) for item in value)

    @field_validator("observation_key_bounds", mode="before")
    @classmethod
    def _normalize_bounds_map(cls, value: Any) -> dict[str, tuple[float, float]]:
        """Normalize the bounds map accepted by this susan observation config."""
        if value is None:
            return {}
        return {
            str(key): (float(bounds[0]), float(bounds[1]))
            for key, bounds in dict(value).items()
        }

    @property
    def observation_fields(self) -> tuple[SusanObservationField, ...]:
        """Return the observation fields exposed by this susan observation config."""
        return tuple(
            SusanObservationField(
                key=key,
                length=self.history_length if key in self.observation_history_keys else 1,
                bounds=self.key_bounds(key),
                channel="history" if key in self.observation_history_keys else "scalar",
            )
            for key in self.observation_keys
        )

    @property
    def observation_descriptor(self) -> SusanObservationDescriptor:
        """Return the observation descriptor exposed by this susan observation config."""
        return SusanObservationDescriptor(fields=self.observation_fields)

    @property
    def observation_keys(self) -> tuple[str, ...]:
        """Return the observation keys exposed by this susan observation config."""
        return self.observation_history_keys + self.observation_scalar_keys

    @property
    def observation_key_lengths(self) -> dict[str, int]:
        """Return the observation key lengths exposed by this susan observation config."""
        return {
            key: self.history_length if key in self.observation_history_keys else 1
            for key in self.observation_keys
        }

    @property
    def observation_size(self) -> int:
        """Return the observation size exposed by this susan observation config."""
        return sum(self.observation_key_lengths.values())

    @classmethod
    def build_default_bounds(cls) -> dict[str, tuple[float, float]]:
        """Build the default bounds required by this susan observation config."""
        config = cls()
        # Declared observation-space guardrails. These are not all producer
        # clamp values: for example, station-like producer values are usually
        # clipped to [-1, 1], while the broader distance guardrail preserves
        # compatibility with older telemetry/checkpoints and diagnostic data.
        return {
            **{
                key: config.observation_normalized_speed_bounds
                for key in (
                    "ego_speed_norm",
                    "road_speed_limit_norm",
                    "red_light_preferred_speed_norm",
                    "red_light_brake_limit_speed_norm",
                    "object_1_speed_norm",
                    "object_2_speed_norm",
                    "object_3_speed_norm",
                )
            },
            "u1_applied_command": config.observation_command_bounds,
            "u1_actual_acceleration": config.observation_command_bounds,
            "red_light_distance_norm": config.observation_distance_bounds,
            "speed_error_to_preferred_norm": config.observation_speed_error_bounds,
            "speed_error_to_brake_limit_norm": config.observation_speed_error_bounds,
            "ego_station_norm": config.observation_distance_bounds,
            "ego_lead_vehicle_present": (0.0, 1.0),
            "object_1_present": (0.0, 1.0),
            "object_1_station_norm": config.observation_distance_bounds,
            "object_2_present": (0.0, 1.0),
            "object_2_station_norm": config.observation_distance_bounds,
            "object_3_present": (0.0, 1.0),
            "object_3_station_norm": config.observation_distance_bounds,
            "tl_is_red": (0.0, 1.0),
            "tl_is_yellow": (0.0, 1.0),
            "tl_is_red_yellow": (0.0, 1.0),
            "tl_is_green": (0.0, 1.0),
            "tl_is_applicable": (0.0, 1.0),
            "red_brake_horizon_active": (0.0, 1.0),
            "tl_phase_remaining_norm": (0.0, 1.0),
            "upcoming_intersection_type": (0.0, 6.0),
        }

    def key_bounds(self, key: str) -> tuple[float, float]:
        """Return the key bounds value owned by this susan observation config."""
        return self.observation_key_bounds.get(key, self.observation_default_bounds)
SUSAN_OBSERVATION_CONFIG = SusanObservationConfig(
    observation_key_bounds=SusanObservationConfig.build_default_bounds(),
)

SUSAN_OBSERVATION_DESCRIPTOR = SUSAN_OBSERVATION_CONFIG.observation_descriptor

OBSERVATION_HISTORY_LENGTH = SUSAN_OBSERVATION_CONFIG.history_length
OBSERVATION_STATION_SENTINEL_VALUE = SUSAN_OBSERVATION_CONFIG.station_sentinel_value
OBSERVATION_STATION_PRESENT_THRESHOLD = SUSAN_OBSERVATION_CONFIG.station_present_threshold
OBSERVATION_STATION_NORMALIZATION_METERS = SUSAN_OBSERVATION_CONFIG.station_normalization_meters
LEAD_HEADWAY_SECONDS = SUSAN_OBSERVATION_CONFIG.lead_headway_seconds
RED_LIGHT_PREFERRED_SPEED_PLATEAU_WIDTH_METERS = SUSAN_OBSERVATION_CONFIG.red_light_preferred_speed_plateau_width_meters
RED_LIGHT_FAR_WAIT_PENALTY_WEIGHT = SUSAN_OBSERVATION_CONFIG.red_light_far_wait_penalty_weight
RED_LIGHT_FAR_WAIT_DISTANCE_WIDTH_METERS = SUSAN_OBSERVATION_CONFIG.red_light_far_wait_distance_width_meters
RED_LIGHT_FAR_WAIT_SLOW_SPEED_SCALE_MPS = SUSAN_OBSERVATION_CONFIG.red_light_far_wait_slow_speed_scale_mps
RED_LIGHT_MAX_NEGATIVE_REWARD = SUSAN_OBSERVATION_CONFIG.red_light_max_negative_reward
RED_LIGHT_RELEASE_CROSSING_REWARD_WEIGHT = SUSAN_OBSERVATION_CONFIG.red_light_release_crossing_reward_weight
SIGNAL_STOP_SPEED_TOLERANCE_MPS = SUSAN_OBSERVATION_CONFIG.signal_stop_speed_tolerance_mps
SIGNAL_STOP_MOTION_ENTRY_SPEED_MPS = (
    SUSAN_OBSERVATION_CONFIG.signal_stop_motion_entry_speed_mps
)
SIGNAL_STOP_MOTION_EXIT_SPEED_MPS = (
    SUSAN_OBSERVATION_CONFIG.signal_stop_motion_exit_speed_mps
)
SIGNAL_STOP_POSITIVE_ACCEL_PENALTY_SCALE_MPS2 = (
    SUSAN_OBSERVATION_CONFIG.signal_stop_positive_accel_penalty_scale_mps2
)
GREEN_LIGHT_IDLE_PENALTY_WEIGHT = SUSAN_OBSERVATION_CONFIG.green_light_idle_penalty_weight
GREEN_LIGHT_IDLE_SLOW_SPEED_SCALE_MPS = SUSAN_OBSERVATION_CONFIG.green_light_idle_slow_speed_scale_mps
SUPERVISED_CMD_ACCEL_DELTA_NORMALIZER = SUSAN_OBSERVATION_CONFIG.supervised_cmd_accel_delta_normalizer
SUPERVISED_CMD_ACCEL_DELTA_CHANGE_NORMALIZER = SUSAN_OBSERVATION_CONFIG.supervised_cmd_accel_delta_change_normalizer
SUPERVISED_CMD_ACCEL_SIGN_CHANGE_DEADBAND_MPS2 = (
    SUSAN_OBSERVATION_CONFIG.supervised_cmd_accel_sign_change_deadband_mps2
)
SUPERVISED_CMD_ACCEL_SIGN_CHANGE_PENALTY_MULTIPLIER = (
    SUSAN_OBSERVATION_CONFIG.supervised_cmd_accel_sign_change_penalty_multiplier
)
SPEED_REWARD_JITTER_DISCOUNT_STRENGTH = SUSAN_OBSERVATION_CONFIG.speed_reward_jitter_discount_strength
SPEED_REWARD_JITTER_DISCOUNT_SCALE = SUSAN_OBSERVATION_CONFIG.speed_reward_jitter_discount_scale
SPEED_REWARD_JITTER_DISCOUNT_DEADBAND = SUSAN_OBSERVATION_CONFIG.speed_reward_jitter_discount_deadband
SPEED_REWARD_JITTER_DISCOUNT_FLOOR = SUSAN_OBSERVATION_CONFIG.speed_reward_jitter_discount_floor
SPEED_REWARD_JITTER_TREND_WINDOW = SUSAN_OBSERVATION_CONFIG.speed_reward_jitter_trend_window
SPEED_REWARD_JITTER_COMFORT_WINDOW = SUSAN_OBSERVATION_CONFIG.speed_reward_jitter_comfort_window

REWARD_WEIGHT_SYNC = SUSAN_OBSERVATION_CONFIG.reward_weight_sync
REWARD_WEIGHT_SPEED = SUSAN_OBSERVATION_CONFIG.reward_weight_speed
REWARD_WEIGHT_DELTA = SUSAN_OBSERVATION_CONFIG.reward_weight_delta
REWARD_WEIGHT_ACCEL = SUSAN_OBSERVATION_CONFIG.reward_weight_accel

OBSERVATION_HISTORY_KEYS = SUSAN_OBSERVATION_CONFIG.observation_history_keys
OBSERVATION_SCALAR_KEYS = SUSAN_OBSERVATION_CONFIG.observation_scalar_keys
OBSERVATION_KEYS = SUSAN_OBSERVATION_CONFIG.observation_keys
OBSERVATION_KEY_LENGTHS = SUSAN_OBSERVATION_CONFIG.observation_key_lengths
OBSERVATION_DESCRIPTOR = SUSAN_OBSERVATION_DESCRIPTOR
OBSERVATION_KEY_BOUNDS = SUSAN_OBSERVATION_DESCRIPTOR.key_bounds
OBSERVATION_CHANNEL_KEYS = SUSAN_OBSERVATION_DESCRIPTOR.channel_keys
OBSERVATION_CHANNEL_NAMES = tuple(OBSERVATION_CHANNEL_KEYS.keys())

OBSERVATION_DEFAULT_BOUNDS = SUSAN_OBSERVATION_CONFIG.observation_default_bounds
OBSERVATION_DISTANCE_BOUNDS = SUSAN_OBSERVATION_CONFIG.observation_distance_bounds
OBSERVATION_NORMALIZED_SPEED_BOUNDS = SUSAN_OBSERVATION_CONFIG.observation_normalized_speed_bounds
OBSERVATION_COMMAND_BOUNDS = SUSAN_OBSERVATION_CONFIG.observation_command_bounds
OBSERVATION_SPEED_ERROR_BOUNDS = SUSAN_OBSERVATION_CONFIG.observation_speed_error_bounds


def observation_key_length(key: str) -> int:
    """Return the observation key length value owned by this observation config module."""
    return OBSERVATION_HISTORY_LENGTH if key in OBSERVATION_HISTORY_KEYS else 1


def observation_key_lengths(keys) -> dict[str, int]:
    """Return the observation key lengths value owned by this observation config module."""
    return {key: observation_key_length(key) for key in keys}


def observation_key_bounds(key: str) -> tuple[float, float]:
    """Return the observation key bounds value owned by this observation config module."""
    return SUSAN_OBSERVATION_CONFIG.key_bounds(key)


def observation_descriptor() -> SusanObservationDescriptor:
    """Return the observation descriptor value owned by this observation config module."""
    return SUSAN_OBSERVATION_DESCRIPTOR
