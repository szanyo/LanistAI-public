# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Stable channel names, dimensions, bounds, and metric-key ordering."""

from __future__ import annotations

from typing import Literal, TypedDict

from lanista.reinforcengine.rl_metrics.observation_contract import (
    POLICY_OBSERVATION_HISTORY_KEYS as OBSERVATION_HISTORY_KEYS,
)
from lanista.reinforcengine.rl_metrics.observation_contract import (
    POLICY_OBSERVATION_KEY_BOUNDS as OBSERVATION_KEY_BOUNDS,
)
from lanista.reinforcengine.rl_metrics.observation_contract import (
    POLICY_OBSERVATION_KEY_LENGTHS as OBSERVATION_KEY_LENGTHS,
)
from lanista.reinforcengine.rl_metrics.observation_contract import (
    POLICY_OBSERVATION_KEYS as SHARED_POLICY_OBSERVATION_KEYS,
)
from lanista.reinforcengine.rl_metrics.observation_contract import (
    TELEMETRY_OBSERVATION_KEYS as VEHICLE_TELEMETRY_OBSERVATION_PARAM_NAMES,
)

LANISTAI_CHANNEL_NAMES = ("observation", "telemetry", "reward")
LANISTAI_CHANNEL_NAME = Literal["observation", "telemetry", "reward"]

TRAINING_DIAGNOSTIC_PARAM_NAMES = (
    "policy_action",
    "actual_speed_mps",
    "requested_speed_mps",
    "reference_speed_mps",
    "road_speed_limit_mps",
    "previous_speed_mps",
    "base_command_mps2",
    "applied_command_mps2",
    "supervised_command_mps2",
    "actual_acceleration_mps2",
    "optimizer_delta_mps2",
    "optimizer_correction_mps2",
    "optimizer_cost",
    "distance_to_stopline_m",
    "ego_station_m",
    "raw_ego_station_m",
    "target_ego_station_m",
    "ego_position_x_m",
    "ego_position_y_m",
    "intersection_number",
    "traffic_light_state_code",
    "traffic_light_link_index",
    "red_light_overdrive",
    "ego_lead_vehicle_present",
    "ego_lead_distance_m",
    "ego_lead_speed_mps",
    "supervisor_vehicle_conflict",
    "policy_vehicle_slot_count",
    "policy_pedestrian_slot_count",
    "active_pedestrian_count",
    "supervisor_pedestrian_conflict",
    "pedestrian_station_m",
    "pedestrian_speed_mps",
    "pedestrian_crossing_state",
    "motion_state_unknown",
    "motion_state_moving",
    "motion_state_decelerating",
    "motion_state_stopped",
    "motion_state_accelerating",
    "stop_event_none",
    "stop_event_stop_begin",
    "stop_event_stop_active",
    "stop_event_stop_end",
    "stop_cause_none",
    "stop_cause_traffic_signal",
    "stop_cause_lead_vehicle",
    "stop_cause_pedestrian",
    "stop_cause_multiple",
    "stop_cause_unknown",
    "use_case_free_drive",
    "use_case_speed_tracking",
    "use_case_traffic_signal_approach",
    "use_case_traffic_signal_stop",
    "use_case_traffic_signal_release",
    "use_case_lead_following",
    "use_case_pedestrian_avoidance",
    "use_case_optimizer_intervention",
    "use_case_sumo_intervention",
    "use_case_terminal_step",
)


class LanistAIChannelNames(TypedDict, total=False):
    """Ordered metric names published by each semantic channel."""

    observation: tuple[str, ...]
    telemetry: tuple[str, ...]
    reward: tuple[str, ...]


class LanistAIChannelLengths(TypedDict, total=False):
    """Encoded value length for every metric in each semantic channel."""

    observation: dict[str, int]
    telemetry: dict[str, int]
    reward: dict[str, int]


class LanistAIChannelBounds(TypedDict, total=False):
    """Inclusive numeric bounds for metrics in each semantic channel."""

    observation: dict[str, tuple[float, float]]
    telemetry: dict[str, tuple[float, float]]
    reward: dict[str, tuple[float, float]]


class RewardComponentSpec(TypedDict):
    """Fallback attribute and canonical path for one reward component."""

    path: tuple[str, ...]
    attr: str

OBSERVATION_PARAM_NAMES = [
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
    "upcoming_intersection_type",
    "u1_supervised_command",
]

TL_OBSERVATION_PARAM_NAMES = [
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
]

TELEMETRY_OBSERVATION_PARAM_NAMES = tuple(OBSERVATION_PARAM_NAMES + TL_OBSERVATION_PARAM_NAMES)
OBSERVATION_PARAM_NAMES = list(TELEMETRY_OBSERVATION_PARAM_NAMES)

OBSERVATION_CHANNEL_PARAM_NAMES: LanistAIChannelNames = {
    "observation": tuple(SHARED_POLICY_OBSERVATION_KEYS),
    "telemetry": VEHICLE_TELEMETRY_OBSERVATION_PARAM_NAMES,
}

# Some diagnostic observation metrics are intentionally not exposed to the
# policy. Keep the policy layout explicit so flat Ray observations can still be
# decoded without removing those metrics from TensorBoard or checkpoint CSVs.
POLICY_OBSERVATION_PARAM_NAMES = list(SHARED_POLICY_OBSERVATION_KEYS)
DIAGNOSTIC_OBSERVATION_PARAM_NAMES = tuple(
    name
    for name in TELEMETRY_OBSERVATION_PARAM_NAMES
    if name not in POLICY_OBSERVATION_PARAM_NAMES
)
POLICY_OBSERVATION_PARAM_LENGTHS = {
    name: OBSERVATION_KEY_LENGTHS[name]
    for name in POLICY_OBSERVATION_PARAM_NAMES
}
POLICY_OBSERVATION_PARAM_BOUNDS = {
    name: OBSERVATION_KEY_BOUNDS[name]
    for name in POLICY_OBSERVATION_PARAM_NAMES
}

OBSERVATION_HISTORY_LENGTH = 30
OBSERVATION_HISTORY_PARAM_NAMES = set(OBSERVATION_HISTORY_KEYS)
OBSERVATION_PARAM_LENGTHS = {
    name: OBSERVATION_HISTORY_LENGTH if name in OBSERVATION_HISTORY_PARAM_NAMES else 1
    for name in OBSERVATION_PARAM_NAMES
}
OBSERVATION_CHANNEL_PARAM_LENGTHS: LanistAIChannelLengths = {
    "observation": POLICY_OBSERVATION_PARAM_LENGTHS,
    "telemetry": OBSERVATION_PARAM_LENGTHS,
}
TELEMETRY_OBSERVATION_EXTRA_BOUNDS = {
    "tl_phase_remaining_delta_norm": (-1.0, 1.0),
    "tl_phase_remaining_raw_delta_norm": (-1.0, 1.0),
    "tl_timing_context_changed": (0.0, 1.0),
    "tl_phase_remaining_monotonic_violation": (0.0, 1.0),
    "tl_tls_id_numeric": (-1000.0, 1000.0),
    "tl_link_index": (-1000.0, 1000.0),
    "tl_intersection_number": (-1000.0, 1000.0),
}
TELEMETRY_OBSERVATION_PARAM_BOUNDS = {
    name: OBSERVATION_KEY_BOUNDS.get(name, TELEMETRY_OBSERVATION_EXTRA_BOUNDS.get(name, (-1.0, 1.0)))
    for name in TELEMETRY_OBSERVATION_PARAM_NAMES
}
OBSERVATION_CHANNEL_PARAM_BOUNDS: LanistAIChannelBounds = {
    "observation": POLICY_OBSERVATION_PARAM_BOUNDS,
    "telemetry": TELEMETRY_OBSERVATION_PARAM_BOUNDS,
}
TELEMETRY_OBSERVATION_PARAM_LENGTHS = OBSERVATION_PARAM_LENGTHS

OBSERVATION_STAT_ORDER = [
    "observations.ego_station.stats",
    "observations.station_1.stats",
    "observations.station_2.stats",
    "observations.station_3.stats",
    "observations.ego_speed.stats",
    "observations.speed_1.stats",
    "observations.speed_2.stats",
    "observations.speed_3.stats",
    "observations.ego_station_norm.stats",
    "observations.ego_lead_vehicle_present.stats",
    "observations.ego_lead_distance_norm.stats",
    "observations.object_1_present.stats",
    "observations.object_1_station_norm.stats",
    "observations.object_1_speed_norm.stats",
    "observations.object_2_present.stats",
    "observations.object_2_station_norm.stats",
    "observations.object_2_speed_norm.stats",
    "observations.object_3_present.stats",
    "observations.object_3_station_norm.stats",
    "observations.object_3_speed_norm.stats",
    "observations.ego_speed_norm.stats",
    "observations.u1_applied_command.stats",
    "observations.u1_actual_acceleration.stats",
    "observations.road_speed_limit.stats",
    "observations.road_speed_limit_norm.stats",
    "observations.red_light_distance_norm.stats",
    "observations.red_light_preferred_speed_norm.stats",
    "observations.red_light_brake_limit_speed_norm.stats",
    "observations.red_brake_horizon_active.stats",
    "observations.speed_error_to_preferred_norm.stats",
    "observations.speed_error_to_brake_limit_norm.stats",
    "observations.tl_is_red.stats",
    "observations.tl_is_yellow.stats",
    "observations.tl_is_red_yellow.stats",
    "observations.tl_is_green.stats",
    "observations.tl_is_applicable.stats",
    "observations.tl_phase_remaining_norm.stats",
    "observations.tl_phase_remaining_delta_norm.stats",
    "observations.tl_phase_remaining_raw_delta_norm.stats",
    "observations.tl_timing_context_changed.stats",
    "observations.tl_phase_remaining_monotonic_violation.stats",
    "observations.tl_tls_id_numeric.stats",
    "observations.tl_link_index.stats",
    "observations.tl_intersection_number.stats",
    "observations.upcoming_intersection_type.stats",
    "observations.u1_supervised_command.stats",
]

REWARD_STAT_ORDER = [
    "rewards.total.raw_stats",
    "rewards.speed.effective_stats",
    "rewards.speed.red_light_speed_reward_stats",
    "rewards.speed.red_light_profile_reward_stats",
    "rewards.speed.red_light_stopline_factor_stats",
    "rewards.speed.red_light_target_speed_stats",
    "rewards.speed.red_light_distance_to_stopline_stats",
    "rewards.speed.red_light_far_wait_penalty_stats",
    "rewards.speed.green_light_idle_penalty_stats",
    "rewards.speed.blocking_profile_active_stats",
    "rewards.speed.blocking_profile_distance_m_stats",
    "rewards.speed.blocking_profile_obstacle_speed_mps_stats",
    "rewards.speed.blocking_profile_closing_speed_mps_stats",
    "rewards.speed.blocking_profile_target_speed_mps_stats",
    "rewards.speed.blocking_profile_reward_stats",
    "rewards.speed.signal_stop_observed_stats",
    "rewards.speed.signal_stop_quality_stats",
    "rewards.speed.signal_stop_positive_accel_penalty_stats",
    "rewards.speed.signal_release_reward_stats",
    "rewards.speed.pre_jitter_discount_effective_stats",
    "rewards.speed.jitter_discount_stats",
    "rewards.speed.jitter_discount_applied_stats",
    "rewards.control.normalized_supervised_cmd_accel_discount_jitter_stats",
    "rewards.control.supervised_cmd_accel_discount_jitter_stats",
    "rewards.control.supervised_cmd_accel_local_jitter_stats",
    "rewards.control.supervised_cmd_accel_local_jitter_sign_change_stats",
    "rewards.control.supervised_cmd_accel_local_effective_jitter_stats",
    "rewards.control.lead_present_local_jitter_stats",
    "rewards.control.lead_absent_local_jitter_stats",
    "rewards.control.supervised_cmd_accel_trend_jitter_stats",
    "rewards.control.supervised_cmd_accel_trend_jitter_sign_change_stats",
    "rewards.control.supervised_cmd_accel_trend_effective_jitter_stats",
    "rewards.control.supervised_cmd_accel_comfort_jitter_stats",
    "rewards.control.supervised_cmd_accel_weighted_comfort_jitter_stats",
    "rewards.control.supervised_cmd_accel_comfort_jitter_sign_change_stats",
    "rewards.control.supervised_cmd_accel_comfort_effective_jitter_stats",
    "rewards.control.abs_supervised_cmd_accel_delta_change_stats",
    "rewards.control.abs_supervised_cmd_accel_delta_stats",
    "rewards.control.supervised_cmd_accel_delta_change_stats",
    "rewards.control.supervised_cmd_accel_delta_stats",
    "rewards.speed.current_stats",
    "rewards.speed.normalized_stats",
    "rewards.speed.weighted_stats",
    "rewards.speed.weighted_uncapped_stats",
    "rewards.delta.raw_stats",
    "rewards.delta.normalized_stats",
    "rewards.delta.rolling_average_stats",
    "rewards.delta.cumulative_average_stats",
    "rewards.delta.weighted_stats",
    "rewards.delta.weighted_uncapped_stats",
    "rewards.accel.raw_stats",
    "rewards.accel.normalized_stats",
    "rewards.accel.rolling_average_stats",
    "rewards.accel.cumulative_average_stats",
    "rewards.accel.weighted_stats",
    "rewards.accel.weighted_uncapped_stats",
    "rewards.sync.raw_stats",
    "rewards.sync.normalized_stats",
    "rewards.sync.rolling_average_stats",
    "rewards.sync.cumulative_average_stats",
    "rewards.sync.command_stats",
    "rewards.sync.actual_acceleration_stats",
    "rewards.sync.absolute_error_stats",
    "rewards.sync.lead_present_absolute_error_stats",
    "rewards.sync.lead_absent_absolute_error_stats",
    "rewards.sync.sumo_intervention_stats",
    "rewards.sync.lead_present_sumo_intervention_stats",
    "rewards.sync.lead_absent_sumo_intervention_stats",
    "rewards.sync.effective_error_stats",
    "rewards.sync.speed_factor_stats",
    "rewards.sync.direction_penalty_stats",
    "rewards.sync.magnitude_penalty_stats",
    "rewards.sync.weighted_stats",
    "rewards.sync.weighted_uncapped_stats",
]

EPISODE_SETUP_STAT_ORDER = [
    "episode_setup.warmup_seconds",
    "episode_setup.warmup_steps",
    "episode_setup.warmup_max_seconds",
    "episode_setup.warmup_max_steps",
    "episode_setup.startup_steps",
    "episode_setup.active_policy_step_limit",
    "episode_setup.active_policy_seconds_limit",
]

INTERACTION_EPISODE_METRIC_NAMES = (
    "unique_followed_vehicles",
    "unique_blocking_pedestrians",
    "lead_interaction_step_fraction",
    "pedestrian_interaction_step_fraction",
    "any_interaction_step_fraction",
)

EPISODE_INFO_STAT_ORDER = [
    "episode_info.episode_terminated.rate",
    "episode_info.episode_truncated.rate",
    "episode_info.termination_curriculum_target.rate",
    "episode_info.termination_route_end.rate",
    "episode_info.truncation_red_light_overdrive.rate",
    "episode_info.truncation_time_limit.rate",
    "episode_info.curriculum_target_detected.rate",
    "episode_info.curriculum_target_accepted.rate",
    "episode_info.curriculum_target_reached.rate",
    "episode_info.curriculum_target_rejected_by_red_light_overdrive.rate",
    "episode_info.episode_end_conflict.rate",
    "episode_info.episode_end_conflict_target_time_limit.rate",
    "episode_info.episode_end_conflict_target_red_light_overdrive.rate",
    "episode_info.terminated_reached_intersection.value",
    "episode_info.terminated_max_speed.value",
    "episode_info.terminated_average_speed.value",
    "episode_info.truncated_reached_intersection.value",
    "episode_info.truncated_max_speed.value",
    "episode_info.truncated_average_speed.value",
    *(f"episode_info.{name}.value" for name in INTERACTION_EPISODE_METRIC_NAMES),
]

PER_EPISODE_METRICS = [
    ("episode_terminated", ("episode_info", "episode_terminated"), ["mean", "sum"]),
    ("episode_truncated", ("episode_info", "episode_truncated"), ["mean", "sum"]),
    ("episode_end_conflict", ("episode_info", "episode_end_conflict"), ["mean", "sum"]),
    (
        "episode_end_conflict_target_time_limit",
        ("episode_info", "episode_end_conflict_target_time_limit"),
        ["mean", "sum"],
    ),
    (
        "episode_end_conflict_target_red_light_overdrive",
        ("episode_info", "episode_end_conflict_target_red_light_overdrive"),
        ["mean", "sum"],
    ),
    ("curriculum_target_detected", ("episode_info", "curriculum_target_detected"), ["mean", "sum"]),
    ("curriculum_target_accepted", ("episode_info", "curriculum_target_accepted"), ["mean", "sum"]),
    ("curriculum_target_reached", ("episode_info", "curriculum_target_reached"), ["mean", "sum"]),
    (
        "curriculum_target_rejected_by_red_light_overdrive",
        ("episode_info", "curriculum_target_rejected_by_red_light_overdrive"),
        ["mean", "sum"],
    ),
    ("termination_curriculum_target", ("episode_info", "termination_curriculum_target"), ["mean", "sum"]),
    ("termination_route_end", ("episode_info", "termination_route_end"), ["mean", "sum"]),
    ("truncation_time_limit", ("episode_info", "truncation_time_limit"), ["mean", "sum"]),
    ("truncation_red_light_overdrive", ("episode_info", "truncation_red_light_overdrive"), ["mean", "sum"]),
    *[
        (name, ("episode_info", name), ["mean"])
        for name in INTERACTION_EPISODE_METRIC_NAMES
    ],
    ("terminated_reached_intersection", ("episode_info", "terminated_reached_intersection"), ["mean"]),
    ("terminated_max_speed", ("episode_info", "terminated_max_speed"), ["mean"]),
    ("terminated_average_speed", ("episode_info", "terminated_average_speed"), ["mean"]),
    ("truncated_reached_intersection", ("episode_info", "truncated_reached_intersection"), ["mean"]),
    ("truncated_max_speed", ("episode_info", "truncated_max_speed"), ["mean"]),
    ("truncated_average_speed", ("episode_info", "truncated_average_speed"), ["mean"]),
    ("max_speed_per_episode", ("vehicle_info", "max_speed_per_episode"), ["mean"]),
    ("average_speed_per_episode", ("vehicle_info", "average_speed_per_episode"), ["mean"]),
    ("reached_intersection_number", ("vehicle_info", "reached_intersection_number"), ["mean"]),
    ("number_of_vehicles", ("vehicle_info", "number_of_vehicles"), ["max", "mean", "min"]),
    ("number_of_pedestrians", ("vehicle_info", "number_of_pedestrians"), ["max", "mean", "min"]),
    ("sync_reward", ("rewards", "sync", "raw"), ["mean", "min"]),
    ("normalized_sync_reward", ("rewards", "sync", "normalized"), ["mean", "min"]),
    ("avg_sync_penalty", ("rewards", "sync", "rolling_average"), ["mean", "min"]),
    ("avg_sync_penalty", ("rewards", "sync", "cumulative_average"), ["mean", "min"]),
    ("accel_change_reward", ("rewards", "accel", "raw"), ["mean", "min"]),
    ("normalized_accel_change_reward", ("rewards", "accel", "normalized"), ["mean", "min"]),
    ("avg_accel_penalty", ("rewards", "accel", "rolling_average"), ["mean", "min"]),
    ("avg_accel_penalty", ("rewards", "accel", "cumulative_average"), ["mean", "min"]),
    ("avg_speed_reward", ("rewards", "speed", "raw"), ["mean", "max"]),
    ("avg_speed_reward", ("rewards", "speed", "current"), ["mean", "max"]),
    ("normalized_avg_speed_reward", ("rewards", "speed", "normalized"), ["mean", "max"]),
    ("max_speed_per_episode", ("rewards", "speed", "max_per_episode"), ["mean"]),
    ("average_speed_per_episode", ("rewards", "speed", "avg_per_episode"), ["mean"]),
    ("delta_change_reward", ("rewards", "delta", "raw"), ["mean", "min"]),
    ("normalized_delta_change_reward", ("rewards", "delta", "normalized"), ["mean", "min"]),
    ("avg_delta_penalty", ("rewards", "delta", "rolling_average"), ["mean", "min"]),
    ("avg_delta_penalty", ("rewards", "delta", "cumulative_average"), ["mean", "min"]),
]

REWARD_SERIES_KEYS = {
    "total": ("rewards", "total", "raw_stats"),
    "sync_raw": ("rewards", "sync", "raw_stats"),
    "sync_normalized": ("rewards", "sync", "normalized_stats"),
    "sync_rolling_average": ("rewards", "sync", "rolling_average_stats"),
    "sync_cumulative_average": ("rewards", "sync", "cumulative_average_stats"),
    "sync_command": ("rewards", "sync", "command_stats"),
    "sync_actual_acceleration": ("rewards", "sync", "actual_acceleration_stats"),
    "sync_absolute_error": ("rewards", "sync", "absolute_error_stats"),
    "sync_lead_present_absolute_error": (
        "rewards",
        "sync",
        "lead_present_absolute_error_stats",
    ),
    "sync_lead_absent_absolute_error": (
        "rewards",
        "sync",
        "lead_absent_absolute_error_stats",
    ),
    "sync_sumo_intervention": (
        "rewards",
        "sync",
        "sumo_intervention_stats",
    ),
    "sync_lead_present_sumo_intervention": (
        "rewards",
        "sync",
        "lead_present_sumo_intervention_stats",
    ),
    "sync_lead_absent_sumo_intervention": (
        "rewards",
        "sync",
        "lead_absent_sumo_intervention_stats",
    ),
    "sync_effective_error": ("rewards", "sync", "effective_error_stats"),
    "sync_speed_factor": ("rewards", "sync", "speed_factor_stats"),
    "sync_direction_penalty": ("rewards", "sync", "direction_penalty_stats"),
    "sync_magnitude_penalty": ("rewards", "sync", "magnitude_penalty_stats"),
    "sync_weighted": ("rewards", "sync", "weighted_stats"),
    "sync_weighted_uncapped": ("rewards", "sync", "weighted_uncapped_stats"),
    "accel_raw": ("rewards", "accel", "raw_stats"),
    "accel_normalized": ("rewards", "accel", "normalized_stats"),
    "accel_rolling_average": ("rewards", "accel", "rolling_average_stats"),
    "accel_cumulative_average": ("rewards", "accel", "cumulative_average_stats"),
    "accel_weighted": ("rewards", "accel", "weighted_stats"),
    "accel_weighted_uncapped": ("rewards", "accel", "weighted_uncapped_stats"),
    "speed_raw": ("rewards", "speed", "raw_stats"),
    "speed_current": ("rewards", "speed", "current_stats"),
    "speed_normalized": ("rewards", "speed", "normalized_stats"),
    "speed_effective": ("rewards", "speed", "effective_stats"),
    "speed_weighted": ("rewards", "speed", "weighted_stats"),
    "speed_weighted_uncapped": ("rewards", "speed", "weighted_uncapped_stats"),
    "speed_red_light_speed_reward": ("rewards", "speed", "red_light_speed_reward_stats"),
    "speed_red_light_profile_reward": ("rewards", "speed", "red_light_profile_reward_stats"),
    "speed_red_light_stopline_factor": ("rewards", "speed", "red_light_stopline_factor_stats"),
    "speed_red_light_target_speed": ("rewards", "speed", "red_light_target_speed_stats"),
    "speed_red_light_distance_to_stopline": ("rewards", "speed", "red_light_distance_to_stopline_stats"),
    "speed_red_light_far_wait_penalty": ("rewards", "speed", "red_light_far_wait_penalty_stats"),
    "speed_green_light_idle_penalty": ("rewards", "speed", "green_light_idle_penalty_stats"),
    "speed_blocking_profile_active": ("rewards", "speed", "blocking_profile_active_stats"),
    "speed_blocking_profile_distance_m": ("rewards", "speed", "blocking_profile_distance_m_stats"),
    "speed_blocking_profile_obstacle_speed_mps": (
        "rewards",
        "speed",
        "blocking_profile_obstacle_speed_mps_stats",
    ),
    "speed_blocking_profile_closing_speed_mps": (
        "rewards",
        "speed",
        "blocking_profile_closing_speed_mps_stats",
    ),
    "speed_blocking_profile_target_speed_mps": ("rewards", "speed", "blocking_profile_target_speed_mps_stats"),
    "speed_blocking_profile_reward": ("rewards", "speed", "blocking_profile_reward_stats"),
    "speed_signal_stop_observed": ("rewards", "speed", "signal_stop_observed_stats"),
    "speed_signal_stop_quality": ("rewards", "speed", "signal_stop_quality_stats"),
    "speed_signal_stop_positive_accel_penalty": (
        "rewards",
        "speed",
        "signal_stop_positive_accel_penalty_stats",
    ),
    "speed_signal_release_reward": ("rewards", "speed", "signal_release_reward_stats"),
    "speed_pre_jitter_discount_effective": ("rewards", "speed", "pre_jitter_discount_effective_stats"),
    "red_light_speed_reward": ("rewards", "red_light", "speed_reward_stats"),
    "red_light_profile_reward": ("rewards", "red_light", "profile_reward_stats"),
    "red_light_stopline_factor": ("rewards", "red_light", "stopline_factor_stats"),
    "red_light_target_speed": ("rewards", "red_light", "target_speed_stats"),
    "red_light_distance_to_stopline": ("rewards", "red_light", "distance_to_stopline_stats"),
    "red_light_far_wait_penalty": ("rewards", "red_light", "far_wait_penalty_stats"),
    "green_light_idle_penalty": ("rewards", "green_light", "idle_penalty_stats"),
    "delta_raw": ("rewards", "delta", "raw_stats"),
    "delta_normalized": ("rewards", "delta", "normalized_stats"),
    "delta_rolling_average": ("rewards", "delta", "rolling_average_stats"),
    "delta_cumulative_average": ("rewards", "delta", "cumulative_average_stats"),
    "delta_weighted": ("rewards", "delta", "weighted_stats"),
    "delta_weighted_uncapped": ("rewards", "delta", "weighted_uncapped_stats"),
    "control_policy_action": ("rewards", "control", "policy_action_stats"),
    "control_supervised_cmd_accel_delta": ("rewards", "control", "supervised_cmd_accel_delta_stats"),
    "control_abs_supervised_cmd_accel_delta": ("rewards", "control", "abs_supervised_cmd_accel_delta_stats"),
    "control_supervised_cmd_accel_delta_change": (
        "rewards",
        "control",
        "supervised_cmd_accel_delta_change_stats",
    ),
    "control_abs_supervised_cmd_accel_delta_change": (
        "rewards",
        "control",
        "abs_supervised_cmd_accel_delta_change_stats",
    ),
    "control_supervised_cmd_accel_local_jitter": (
        "rewards",
        "control",
        "supervised_cmd_accel_local_jitter_stats",
    ),
    "control_supervised_cmd_accel_local_jitter_sign_change": (
        "rewards",
        "control",
        "supervised_cmd_accel_local_jitter_sign_change_stats",
    ),
    "control_supervised_cmd_accel_local_effective_jitter": (
        "rewards",
        "control",
        "supervised_cmd_accel_local_effective_jitter_stats",
    ),
    "control_lead_present_local_jitter": (
        "rewards",
        "control",
        "lead_present_local_jitter_stats",
    ),
    "control_lead_absent_local_jitter": (
        "rewards",
        "control",
        "lead_absent_local_jitter_stats",
    ),
    "control_supervised_cmd_accel_trend_jitter": (
        "rewards",
        "control",
        "supervised_cmd_accel_trend_jitter_stats",
    ),
    "control_supervised_cmd_accel_trend_jitter_sign_change": (
        "rewards",
        "control",
        "supervised_cmd_accel_trend_jitter_sign_change_stats",
    ),
    "control_supervised_cmd_accel_trend_effective_jitter": (
        "rewards",
        "control",
        "supervised_cmd_accel_trend_effective_jitter_stats",
    ),
    "control_supervised_cmd_accel_comfort_jitter": (
        "rewards",
        "control",
        "supervised_cmd_accel_comfort_jitter_stats",
    ),
    "control_supervised_cmd_accel_weighted_comfort_jitter": (
        "rewards",
        "control",
        "supervised_cmd_accel_weighted_comfort_jitter_stats",
    ),
    "control_supervised_cmd_accel_comfort_jitter_sign_change": (
        "rewards",
        "control",
        "supervised_cmd_accel_comfort_jitter_sign_change_stats",
    ),
    "control_supervised_cmd_accel_comfort_effective_jitter": (
        "rewards",
        "control",
        "supervised_cmd_accel_comfort_effective_jitter_stats",
    ),
    "control_supervised_cmd_accel_discount_jitter": (
        "rewards",
        "control",
        "supervised_cmd_accel_discount_jitter_stats",
    ),
    "control_normalized_supervised_cmd_accel_discount_jitter": (
        "rewards",
        "control",
        "normalized_supervised_cmd_accel_discount_jitter_stats",
    ),
    "speed_reward_jitter_discount": ("rewards", "speed", "jitter_discount_stats"),
    "speed_reward_jitter_discount_applied": ("rewards", "speed", "jitter_discount_applied_stats"),
}

REWARD_CHANNEL_PARAM_NAMES = tuple(REWARD_SERIES_KEYS.keys())
OBSERVATION_CHANNEL_PARAM_NAMES = {
    **OBSERVATION_CHANNEL_PARAM_NAMES,
    "reward": REWARD_CHANNEL_PARAM_NAMES,
}
OBSERVATION_CHANNEL_PARAM_LENGTHS = {
    **OBSERVATION_CHANNEL_PARAM_LENGTHS,
    "reward": {name: 1 for name in REWARD_CHANNEL_PARAM_NAMES},
}
OBSERVATION_CHANNEL_PARAM_BOUNDS = {
    **OBSERVATION_CHANNEL_PARAM_BOUNDS,
    "reward": {
        name: (float("-inf"), float("inf"))
        for name in REWARD_CHANNEL_PARAM_NAMES
    },
}
REWARD_CHANNEL_PARAM_LENGTHS = OBSERVATION_CHANNEL_PARAM_LENGTHS["reward"]
REWARD_CHANNEL_PARAM_BOUNDS = OBSERVATION_CHANNEL_PARAM_BOUNDS["reward"]


REWARD_COMPONENT_SERIES: dict[str, RewardComponentSpec] = {
    "sync_raw": {"path": REWARD_SERIES_KEYS["sync_raw"], "attr": "sync_reward"},
    "sync_normalized": {"path": REWARD_SERIES_KEYS["sync_normalized"], "attr": "normalized_sync_reward"},
    "sync_rolling_average": {"path": REWARD_SERIES_KEYS["sync_rolling_average"], "attr": "avg_sync_penalty"},
    "sync_cumulative_average": {
        "path": REWARD_SERIES_KEYS["sync_cumulative_average"],
        "attr": "avg_sync_cumulative_penalty",
    },
    "sync_command": {
        "path": REWARD_SERIES_KEYS["sync_command"],
        "attr": "sync_command",
    },
    "sync_actual_acceleration": {
        "path": REWARD_SERIES_KEYS["sync_actual_acceleration"],
        "attr": "sync_actual_acceleration",
    },
    "sync_absolute_error": {
        "path": REWARD_SERIES_KEYS["sync_absolute_error"],
        "attr": "sync_absolute_error",
    },
    "sync_effective_error": {
        "path": REWARD_SERIES_KEYS["sync_effective_error"],
        "attr": "sync_effective_error",
    },
    "sync_direction_penalty": {
        "path": REWARD_SERIES_KEYS["sync_direction_penalty"],
        "attr": "sync_direction_penalty",
    },
    "sync_magnitude_penalty": {
        "path": REWARD_SERIES_KEYS["sync_magnitude_penalty"],
        "attr": "sync_magnitude_penalty",
    },
    "sync_speed_factor": {"path": REWARD_SERIES_KEYS["sync_speed_factor"], "attr": "sync_speed_factor"},
    "sync_weighted": {"path": REWARD_SERIES_KEYS["sync_weighted"], "attr": "weighted_sync_reward"},
    "sync_weighted_uncapped": {
        "path": REWARD_SERIES_KEYS["sync_weighted_uncapped"],
        "attr": "uncapped_weighted_sync_reward",
    },
    "accel_raw": {"path": REWARD_SERIES_KEYS["accel_raw"], "attr": "accel_change_reward"},
    "accel_normalized": {
        "path": REWARD_SERIES_KEYS["accel_normalized"],
        "attr": "normalized_accel_change_reward",
    },
    "accel_rolling_average": {
        "path": REWARD_SERIES_KEYS["accel_rolling_average"],
        "attr": "avg_accel_penalty",
    },
    "accel_cumulative_average": {
        "path": REWARD_SERIES_KEYS["accel_cumulative_average"],
        "attr": "avg_accel_cumulative_penalty",
    },
    "accel_weighted": {"path": REWARD_SERIES_KEYS["accel_weighted"], "attr": "weighted_accel_reward"},
    "accel_weighted_uncapped": {
        "path": REWARD_SERIES_KEYS["accel_weighted_uncapped"],
        "attr": "uncapped_weighted_accel_reward",
    },
    "speed_raw": {"path": REWARD_SERIES_KEYS["speed_raw"], "attr": "avg_speed_reward"},
    "speed_current": {"path": REWARD_SERIES_KEYS["speed_current"], "attr": "avg_speed_reward"},
    "speed_normalized": {
        "path": REWARD_SERIES_KEYS["speed_normalized"],
        "attr": "normalized_avg_speed_reward",
    },
    "speed_effective": {"path": REWARD_SERIES_KEYS["speed_effective"], "attr": "effective_speed_reward"},
    "speed_weighted": {"path": REWARD_SERIES_KEYS["speed_weighted"], "attr": "weighted_speed_reward"},
    "speed_weighted_uncapped": {
        "path": REWARD_SERIES_KEYS["speed_weighted_uncapped"],
        "attr": "uncapped_weighted_speed_reward",
    },
    "red_light_speed_reward": {
        "path": REWARD_SERIES_KEYS["red_light_speed_reward"],
        "attr": "red_light_speed_reward",
    },
    "red_light_profile_reward": {
        "path": REWARD_SERIES_KEYS["red_light_profile_reward"],
        "attr": "red_light_profile_reward",
    },
    "red_light_stopline_factor": {
        "path": REWARD_SERIES_KEYS["red_light_stopline_factor"],
        "attr": "red_light_stopline_factor",
    },
    "red_light_target_speed": {
        "path": REWARD_SERIES_KEYS["red_light_target_speed"],
        "attr": "red_light_target_speed",
    },
    "red_light_distance_to_stopline": {
        "path": REWARD_SERIES_KEYS["red_light_distance_to_stopline"],
        "attr": "red_light_distance_to_stopline",
    },
    "red_light_far_wait_penalty": {
        "path": REWARD_SERIES_KEYS["red_light_far_wait_penalty"],
        "attr": "red_light_far_wait_penalty",
    },
    "green_light_idle_penalty": {
        "path": REWARD_SERIES_KEYS["green_light_idle_penalty"],
        "attr": "green_light_idle_penalty",
    },
    "delta_raw": {"path": REWARD_SERIES_KEYS["delta_raw"], "attr": "delta_change_reward"},
    "delta_normalized": {
        "path": REWARD_SERIES_KEYS["delta_normalized"],
        "attr": "normalized_delta_change_reward",
    },
    "delta_rolling_average": {
        "path": REWARD_SERIES_KEYS["delta_rolling_average"],
        "attr": "avg_delta_penalty",
    },
    "delta_cumulative_average": {
        "path": REWARD_SERIES_KEYS["delta_cumulative_average"],
        "attr": "avg_delta_cumulative_penalty",
    },
    "delta_weighted": {"path": REWARD_SERIES_KEYS["delta_weighted"], "attr": "weighted_delta_reward"},
    "delta_weighted_uncapped": {
        "path": REWARD_SERIES_KEYS["delta_weighted_uncapped"],
        "attr": "uncapped_weighted_delta_reward",
    },
}
