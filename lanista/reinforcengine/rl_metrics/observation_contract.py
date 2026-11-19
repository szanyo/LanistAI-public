# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Neutral LanistAI observation/telemetry contract shared by metrics and models."""

POLICY_OBSERVATION_HISTORY_LENGTH = 30

POLICY_OBSERVATION_HISTORY_KEYS = (
    "ego_speed_norm",
    "u1_applied_command",
    "u1_actual_acceleration",
    "road_speed_limit_norm",
    "red_light_distance_norm",
    "speed_error_to_brake_limit_norm",
    "tl_is_red",
    "tl_is_yellow",
    "tl_is_red_yellow",
    "tl_is_green",
    "tl_is_applicable",
    "ego_lead_vehicle_present",
    "tl_phase_remaining_norm",
)

POLICY_OBSERVATION_SCALAR_KEYS = (
    "ego_station_norm",
    "red_light_preferred_speed_norm",
    "red_light_brake_limit_speed_norm",
    "speed_error_to_preferred_norm",
    "red_brake_horizon_active",
    "object_1_present",
    "object_1_station_norm",
    "object_1_speed_norm",
    "object_2_present",
    "object_2_station_norm",
    "object_2_speed_norm",
    "object_3_present",
    "object_3_station_norm",
    "object_3_speed_norm",
    "upcoming_intersection_type",
)

POLICY_OBSERVATION_KEYS = POLICY_OBSERVATION_HISTORY_KEYS + POLICY_OBSERVATION_SCALAR_KEYS

LEGACY_TELEMETRY_OBSERVATION_KEYS = (
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

TRAFFIC_LIGHT_TELEMETRY_OBSERVATION_KEYS = (
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

TELEMETRY_OBSERVATION_KEYS = (
    LEGACY_TELEMETRY_OBSERVATION_KEYS + TRAFFIC_LIGHT_TELEMETRY_OBSERVATION_KEYS
)

POLICY_OBSERVATION_KEY_LENGTHS = {
    key: POLICY_OBSERVATION_HISTORY_LENGTH if key in POLICY_OBSERVATION_HISTORY_KEYS else 1
    for key in POLICY_OBSERVATION_KEYS
}

_DEFAULT_BOUNDS = (-10.0, 10.0)
_DISTANCE_BOUNDS = (-100.0, 10.0)
_NORMALIZED_SPEED_BOUNDS = (0.0, 4.0)
_COMMAND_BOUNDS = (-8.0, 8.0)
_SPEED_ERROR_BOUNDS = (-4.0, 4.0)

POLICY_OBSERVATION_KEY_BOUNDS = {key: _DEFAULT_BOUNDS for key in POLICY_OBSERVATION_KEYS}
POLICY_OBSERVATION_KEY_BOUNDS.update(
    {
        **{
            key: _NORMALIZED_SPEED_BOUNDS
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
        "u1_applied_command": _COMMAND_BOUNDS,
        "u1_actual_acceleration": _COMMAND_BOUNDS,
        "red_light_distance_norm": _DISTANCE_BOUNDS,
        "speed_error_to_preferred_norm": _SPEED_ERROR_BOUNDS,
        "speed_error_to_brake_limit_norm": _SPEED_ERROR_BOUNDS,
        "ego_station_norm": _DISTANCE_BOUNDS,
        "ego_lead_vehicle_present": (0.0, 1.0),
        "object_1_present": (0.0, 1.0),
        "object_1_station_norm": _DISTANCE_BOUNDS,
        "object_2_present": (0.0, 1.0),
        "object_2_station_norm": _DISTANCE_BOUNDS,
        "object_3_present": (0.0, 1.0),
        "object_3_station_norm": _DISTANCE_BOUNDS,
        "tl_is_red": (0.0, 1.0),
        "tl_is_yellow": (0.0, 1.0),
        "tl_is_red_yellow": (0.0, 1.0),
        "tl_is_green": (0.0, 1.0),
        "tl_is_applicable": (0.0, 1.0),
        "red_brake_horizon_active": (0.0, 1.0),
        "tl_phase_remaining_norm": (0.0, 1.0),
        "upcoming_intersection_type": (0.0, 6.0),
    }
)
