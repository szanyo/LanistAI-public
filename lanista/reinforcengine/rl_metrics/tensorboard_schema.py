# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed schema-v2 policy for compact TensorBoard publication.

The metric producers expose framework-neutral flat keys.  This module owns
their logging profile and semantic TensorBoard placement so SB3 and Ray cannot
drift into separate KPI namespaces.
"""

from __future__ import annotations

import os
from enum import StrEnum


class TensorboardProfile(StrEnum):
    """Supported storage/detail profiles for new TensorBoard events."""

    COMPACT = "compact"
    DIAGNOSTIC = "diagnostic"
    FULL = "full"


TENSORBOARD_SCHEMA_VERSION = 2
TENSORBOARD_PROFILE_ENV = "LANISTAI_TENSORBOARD_PROFILE"
TENSORBOARD_LEGACY_HISTOGRAM_ENV = "LANISTAI_TENSORBOARD_HISTOGRAMS"
TENSORBOARD_HISTOGRAM_CADENCE = 10
TENSORBOARD_HISTOGRAM_SAMPLE_LIMIT = 512

TRAINING_TENSORBOARD_ROOTS = (
    "lanistai_overview",
    "lanistai_episode",
    "lanistai_curriculum",
    "lanistai_vehicle",
    "lanistai_observation",
    "lanistai_signal",
    "lanistai_reward",
    "lanistai_control",
    "lanistai_quality",
    "lanistai_distribution",
)

EVALUATION_TENSORBOARD_ROOTS = (
    "lanistai_eval_overview",
    "lanistai_eval_motion",
    "lanistai_eval_safety",
    "lanistai_eval_reward",
    "lanistai_eval_quality",
)

# This is the complete pre-schema-v2 dashboard surface. The numbered aliases
# remain canonical under ``lanistai_overview`` so ordering and familiar names
# survive without writing a second semantic copy of the same source metric.
NUMBERED_DASHBOARD_KPI_TAGS: dict[str, str] = {
    "episode_info.episode_terminated.sum": "01_terminated_episodes",
    "episode_info.episode_truncated.sum": "02_truncated_episodes",
    "vehicle_info.average_speed_per_episode.mean": "03_avg_episode_speed",
    "vehicle_info.max_speed_per_episode.mean": "04_max_episode_speed",
    "vehicle_info.reached_intersection_number.mean": "05_reached_intersection",
    "episode_info.truncation_red_light_overdrive.sum": (
        "06_red_light_overdrive_truncations"
    ),
    "episode_info.truncation_time_limit.sum": "07_time_limit_truncations",
    "episode_info.curriculum_target_reached.sum": "08_curriculum_target_reached",
    "episode_info.curriculum_target_rejected_by_red_light_overdrive.sum": (
        "09_curriculum_red_light_rejections"
    ),
    "rewards.total.raw_stats.mean": "10_total_reward",
    "rewards.speed.effective_stats.mean": "11_speed_effective_reward",
    "rewards.speed.red_light_speed_reward_stats.mean": "12_red_light_speed_reward",
    "rewards.speed.red_light_far_wait_penalty_stats.mean": "13_far_wait_penalty",
    "rewards.speed.green_light_idle_penalty_stats.mean": "14_green_idle_penalty",
    "rewards.speed.jitter_discount_stats.mean": "16_speed_jitter_discount",
    "rewards.control.supervised_cmd_accel_local_jitter_stats.mean": "17_local_jitter",
    "rewards.control.supervised_cmd_accel_trend_jitter_stats.mean": "18_trend_jitter",
    "rewards.control.supervised_cmd_accel_weighted_comfort_jitter_stats.mean": (
        "19_weighted_comfort_jitter"
    ),
    "rewards.control.supervised_cmd_accel_discount_jitter_stats.mean": (
        "20_discount_jitter"
    ),
    "observations.tl_is_applicable.stats.mean": "21_tl_applicable",
    "observations.tl_is_red.stats.mean": "22_tl_red",
    "observations.tl_is_red_yellow.stats.mean": "23_tl_red_yellow",
    "observations.tl_is_green.stats.mean": "24_tl_green",
    "observations.u1_applied_command.stats.mean": "25_u1_applied_command",
    "observations.u1_actual_acceleration.stats.mean": "26_u1_actual_acceleration",
    "observations.u1_supervised_command.stats.mean": "27_u1_supervised_command",
    "rewards.control.policy_action_stats.mean": "28_policy_action",
    "rewards.sync.absolute_error_stats.mean": "29_sync_absolute_error",
    "observations.ego_lead_vehicle_present.stats.mean": "30_lead_vehicle_presence",
    "rewards.sync.sumo_intervention_stats.mean": "31_sumo_intervention_rate",
    "rewards.sync.lead_present_absolute_error_stats.mean": (
        "32_lead_present_sync_error"
    ),
    "rewards.sync.lead_absent_absolute_error_stats.mean": (
        "33_lead_absent_sync_error"
    ),
    "rewards.control.lead_present_local_jitter_stats.mean": (
        "34_lead_present_local_jitter"
    ),
    "rewards.control.lead_absent_local_jitter_stats.mean": (
        "35_lead_absent_local_jitter"
    ),
    "rewards.sync.lead_present_sumo_intervention_stats.mean": (
        "36_lead_present_sumo_intervention_rate"
    ),
    "rewards.sync.lead_absent_sumo_intervention_stats.mean": (
        "37_lead_absent_sumo_intervention_rate"
    ),
}

# Means/rates share a framework-neutral aggregation contract. The historical
# dashboard means keep their numbered names; the additional outcome/conflict
# rates are distinct evidence and therefore are not redundant aliases.
COMPARABLE_KPI_TAGS: dict[str, str] = {
    **{
        key: alias
        for key, alias in NUMBERED_DASHBOARD_KPI_TAGS.items()
        if key.endswith(".mean")
    },
    "episode_info.episode_terminated.mean": "terminated_rate",
    "episode_info.episode_truncated.mean": "truncated_rate",
    "episode_info.truncation_red_light_overdrive.mean": "red_light_overdrive_rate",
    "episode_info.truncation_time_limit.mean": "time_limit_rate",
    "episode_info.curriculum_target_reached.mean": "curriculum_target_reached_rate",
    "episode_info.curriculum_target_rejected_by_red_light_overdrive.mean": (
        "curriculum_target_rejected_rate"
    ),
    "diagnostics.supervisor_vehicle_conflict.stats.mean": "vehicle_conflict_rate",
    "diagnostics.supervisor_pedestrian_conflict.stats.mean": "pedestrian_conflict_rate",
}

# SB3 carries these sums across callback replacement within one training
# session; RLlib's unbounded ``sum`` reduction carries the same per-episode
# indicators across result windows.
COMPARABLE_CUMULATIVE_KPI_TAGS: dict[str, str] = {
    key: alias
    for key, alias in NUMBERED_DASHBOARD_KPI_TAGS.items()
    if key.endswith(".sum")
}

_COMPACT_EPISODE_KEYS = {
    *COMPARABLE_CUMULATIVE_KPI_TAGS,
    "episode_info.episode_terminated.mean",
    "episode_info.episode_truncated.mean",
    "episode_info.termination_curriculum_target.mean",
    "episode_info.termination_route_end.mean",
    "episode_info.truncation_red_light_overdrive.mean",
    "episode_info.truncation_time_limit.mean",
    "episode_info.curriculum_target_detected.mean",
    "episode_info.curriculum_target_accepted.mean",
    "episode_info.curriculum_target_reached.mean",
    "episode_info.curriculum_target_rejected_by_red_light_overdrive.mean",
    "episode_info.episode_end_conflict.mean",
    "episode_info.episode_end_conflict_target_time_limit.mean",
    "episode_info.episode_end_conflict_target_red_light_overdrive.mean",
    "episode_info.terminated_reached_intersection.mean",
    "episode_info.terminated_max_speed.mean",
    "episode_info.terminated_average_speed.mean",
    "episode_info.truncated_reached_intersection.mean",
    "episode_info.truncated_max_speed.mean",
    "episode_info.truncated_average_speed.mean",
    "episode_info.unique_followed_vehicles.mean",
    "episode_info.unique_blocking_pedestrians.mean",
    "episode_info.lead_interaction_step_fraction.mean",
    "episode_info.pedestrian_interaction_step_fraction.mean",
    "episode_info.any_interaction_step_fraction.mean",
    "episode_setup.warmup_seconds.value",
    "episode_setup.warmup_steps.value",
    "episode_setup.active_policy_step_limit.value",
    "episode_setup.active_policy_seconds_limit.value",
}

_COMPACT_VEHICLE_KEYS = {
    "vehicle_info.average_speed_per_episode.mean",
    "vehicle_info.max_speed_per_episode.mean",
    "vehicle_info.reached_intersection_number.mean",
    "vehicle_info.number_of_vehicles.mean",
    "vehicle_info.number_of_pedestrians.mean",
}

_COMPACT_OBSERVATION_KEYS = {
    "observations.ego_speed_norm.stats.mean",
    "observations.ego_station_norm.stats.mean",
    "observations.ego_lead_vehicle_present.stats.mean",
    "observations.ego_lead_distance_norm.stats.mean",
    "observations.road_speed_limit_norm.stats.mean",
    "observations.red_light_distance_norm.stats.mean",
    "observations.red_light_preferred_speed_norm.stats.mean",
    "observations.red_light_brake_limit_speed_norm.stats.mean",
    "observations.red_brake_horizon_active.stats.mean",
    "observations.speed_error_to_preferred_norm.stats.mean",
    "observations.speed_error_to_brake_limit_norm.stats.mean",
    "observations.tl_is_red.stats.mean",
    "observations.tl_is_yellow.stats.mean",
    "observations.tl_is_red_yellow.stats.mean",
    "observations.tl_is_green.stats.mean",
    "observations.tl_is_applicable.stats.mean",
    "observations.tl_phase_remaining_norm.stats.mean",
    "observations.tl_timing_context_changed.stats.mean",
    "observations.tl_phase_remaining_monotonic_violation.stats.mean",
    "observations.u1_applied_command.stats.mean",
    "observations.u1_actual_acceleration.stats.mean",
    "observations.u1_supervised_command.stats.mean",
    "observations.upcoming_intersection_type.stats.mean",
}

_COMPACT_REWARD_KEYS = {
    "rewards.total.raw_stats.mean",
    "rewards.speed.effective_stats.mean",
    "rewards.speed.red_light_speed_reward_stats.mean",
    "rewards.speed.red_light_profile_reward_stats.mean",
    "rewards.speed.red_light_stopline_factor_stats.mean",
    "rewards.speed.red_light_target_speed_stats.mean",
    "rewards.speed.red_light_distance_to_stopline_stats.mean",
    "rewards.speed.red_light_far_wait_penalty_stats.mean",
    "rewards.speed.green_light_idle_penalty_stats.mean",
    "rewards.speed.blocking_profile_active_stats.mean",
    "rewards.speed.blocking_profile_distance_m_stats.mean",
    "rewards.speed.blocking_profile_obstacle_speed_mps_stats.mean",
    "rewards.speed.blocking_profile_closing_speed_mps_stats.mean",
    "rewards.speed.blocking_profile_target_speed_mps_stats.mean",
    "rewards.speed.blocking_profile_reward_stats.mean",
    "rewards.speed.signal_stop_positive_accel_penalty_stats.mean",
    "rewards.speed.signal_stop_quality_stats.mean",
    "rewards.speed.signal_release_reward_stats.mean",
    "rewards.speed.jitter_discount_stats.mean",
    "rewards.speed.jitter_discount_applied_stats.mean",
    "rewards.sync.weighted_stats.mean",
    "rewards.sync.absolute_error_stats.mean",
    "rewards.sync.sumo_intervention_stats.mean",
    "rewards.sync.lead_present_absolute_error_stats.mean",
    "rewards.sync.lead_absent_absolute_error_stats.mean",
    "rewards.sync.lead_present_sumo_intervention_stats.mean",
    "rewards.sync.lead_absent_sumo_intervention_stats.mean",
    "rewards.sync.effective_error_stats.mean",
    "rewards.delta.weighted_stats.mean",
    "rewards.accel.weighted_stats.mean",
    "rewards.control.policy_action_stats.mean",
    "rewards.control.supervised_cmd_accel_local_jitter_stats.mean",
    "rewards.control.supervised_cmd_accel_local_jitter_sign_change_stats.mean",
    "rewards.control.lead_present_local_jitter_stats.mean",
    "rewards.control.lead_absent_local_jitter_stats.mean",
    "rewards.control.supervised_cmd_accel_trend_jitter_stats.mean",
    "rewards.control.supervised_cmd_accel_trend_jitter_sign_change_stats.mean",
    "rewards.control.supervised_cmd_accel_weighted_comfort_jitter_stats.mean",
    "rewards.control.supervised_cmd_accel_comfort_jitter_sign_change_stats.mean",
    "rewards.control.supervised_cmd_accel_discount_jitter_stats.mean",
}

_COMPACT_DIAGNOSTIC_NAMES = {
    "actual_speed_mps",
    "requested_speed_mps",
    "reference_speed_mps",
    "distance_to_stopline_m",
    "red_light_overdrive",
    "supervisor_vehicle_conflict",
    "supervisor_pedestrian_conflict",
    "ego_lead_distance_m",
    "optimizer_correction_mps2",
    "motion_state_stopped",
    "stop_event_stop_begin",
    "use_case_optimizer_intervention",
    "use_case_sumo_intervention",
}

COMPACT_SCALAR_METRIC_KEYS = frozenset(
    {
        *_COMPACT_EPISODE_KEYS,
        *_COMPACT_VEHICLE_KEYS,
        *_COMPACT_OBSERVATION_KEYS,
        *_COMPACT_REWARD_KEYS,
        *(
            f"diagnostics.{name}.stats.mean"
            for name in _COMPACT_DIAGNOSTIC_NAMES
        ),
    }
)

DIAGNOSTIC_HISTOGRAM_METRIC_KEYS = frozenset(
    {
        "observations.ego_speed_norm.stats.mean",
        "observations.red_light_distance_norm.stats.mean",
        "observations.u1_applied_command.stats.mean",
        "observations.u1_actual_acceleration.stats.mean",
        "rewards.total.raw_stats.mean",
        "rewards.speed.effective_stats.mean",
        "rewards.sync.absolute_error_stats.mean",
        "rewards.control.policy_action_stats.mean",
    }
)


def resolve_tensorboard_profile(
    profile: TensorboardProfile | str | None = None,
) -> TensorboardProfile:
    """Resolve an explicit or environment-selected profile safely."""

    if isinstance(profile, TensorboardProfile):
        return profile
    candidate = str(profile or "").strip().lower()
    if not candidate:
        candidate = str(os.environ.get(TENSORBOARD_PROFILE_ENV, "") or "").strip().lower()
    if candidate:
        try:
            return TensorboardProfile(candidate)
        except ValueError:
            return TensorboardProfile.COMPACT

    # Compatibility bridge: the old histogram switch only changes behavior
    # when it is explicitly present.  Its historical implicit default does not
    # override schema-v2's compact default.
    legacy_histograms = os.environ.get(TENSORBOARD_LEGACY_HISTOGRAM_ENV)
    if legacy_histograms is not None and str(legacy_histograms).strip().lower() not in {
        "0",
        "off",
        "none",
        "false",
    }:
        return TensorboardProfile.DIAGNOSTIC
    return TensorboardProfile.COMPACT


def comparable_kpi_tag(flat_metric_key: str) -> str | None:
    """Return one framework-neutral overview tag for a comparable KPI."""

    key = str(flat_metric_key)
    alias = NUMBERED_DASHBOARD_KPI_TAGS.get(key)
    if alias is None:
        alias = COMPARABLE_KPI_TAGS.get(key)
    return f"lanistai_overview/{alias}" if alias is not None else None


def _normalized_metric_parts(flat_metric_key: str) -> list[str]:
    """Remove implementation-only ``stats`` path noise from one key."""

    parts = str(flat_metric_key).split(".")
    normalized: list[str] = []
    for part in parts:
        if part == "stats":
            continue
        if part.endswith("_stats"):
            part = part[: -len("_stats")]
        normalized.append(part)
    return normalized


def _observation_tensorboard_root(metric_name: str) -> str:
    """Return the semantic group for one policy observation."""

    if metric_name.startswith("tl_") or metric_name.startswith("red_light") or metric_name in {
            "red_brake_horizon_active",
            "speed_error_to_preferred_norm",
            "speed_error_to_brake_limit_norm",
    }:
        return "lanistai_signal"
    if metric_name.startswith("u1_"):
        return "lanistai_control"
    if metric_name.startswith("ego_lead"):
        return "lanistai_vehicle"
    return "lanistai_observation"


def _diagnostic_tensorboard_root(metric_name: str) -> str:
    """Return the semantic group for one training diagnostic."""

    if any(
        token in metric_name
        for token in ("traffic_light", "stopline", "red_light", "stop_")
    ):
        return "lanistai_signal"
    if any(
        token in metric_name
        for token in (
            "command",
            "acceleration",
            "optimizer",
            "policy_action",
            "sumo_intervention",
        )
    ):
        return "lanistai_control"
    return "lanistai_vehicle"


def _training_tensorboard_group(
    key: str,
    parts: list[str],
) -> tuple[str, list[str]]:
    """Resolve one non-KPI metric into its root and implementation-free tail."""

    head = parts[0] if parts else "quality"
    tail = parts[1:] if len(parts) > 1 else parts
    if head in {"episode_info", "episode_setup"}:
        return "lanistai_episode", tail
    if head == "vehicle_info":
        return "lanistai_vehicle", tail
    if head == "observations":
        metric_name = parts[1] if len(parts) > 1 else "unknown"
        return _observation_tensorboard_root(metric_name), tail
    elif head == "rewards":
        component = parts[1] if len(parts) > 1 else "total"
        if component in {"control", "sync"}:
            return (
                "lanistai_control",
                parts[2:] if component == "control" else parts[1:],
            )
        if component == "speed" and any(
            token in key
            for token in ("red_light", "green_light", "blocking_profile", "signal_")
        ):
            return "lanistai_signal", parts[1:]
        return "lanistai_reward", parts[1:]
    if head == "diagnostics":
        metric_name = parts[1] if len(parts) > 1 else "unknown"
        return _diagnostic_tensorboard_root(metric_name), parts[1:]
    return "lanistai_quality", tail


def training_tensorboard_tag(flat_metric_key: str) -> str:
    """Map one flat metric key to its single semantic schema-v2 tag."""

    key = str(flat_metric_key)
    kpi_tag = comparable_kpi_tag(key)
    if kpi_tag is not None:
        return kpi_tag

    parts = _normalized_metric_parts(key)
    root, tail = _training_tensorboard_group(key, parts)
    return "/".join((root, *tail))


def training_tensorboard_path(flat_metric_key: str) -> tuple[str, ...]:
    """Return the tuple-path equivalent of :func:`training_tensorboard_tag`."""

    return tuple(training_tensorboard_tag(flat_metric_key).split("/"))


def training_histogram_tag(flat_metric_key: str) -> str:
    """Place one selected distribution under the dedicated top-level root."""

    scalar_parts = training_tensorboard_tag(flat_metric_key).split("/")
    semantic_root = scalar_parts[0].removeprefix("lanistai_")
    parts = scalar_parts[1:]
    if parts and parts[-1] == "mean":
        parts = parts[:-1]
    return "/".join(("lanistai_distribution", semantic_root, *parts))


EVALUATION_FIELD_TAGS: dict[str, str] = {
    "scenario_index": "lanistai_eval_overview/scenario_index",
    "policy_action": "lanistai_eval_motion/policy_action",
    "actual_speed_mps": "lanistai_eval_motion/actual_speed_mps",
    "requested_speed_mps": "lanistai_eval_motion/requested_speed_mps",
    "reference_speed_mps": "lanistai_eval_motion/reference_speed_mps",
    "road_speed_limit_mps": "lanistai_eval_motion/road_speed_limit_mps",
    "base_command_mps2": "lanistai_eval_motion/base_command_mps2",
    "applied_command_mps2": "lanistai_eval_motion/applied_command_mps2",
    "supervised_command_mps2": "lanistai_eval_motion/supervised_command_mps2",
    "actual_acceleration_mps2": "lanistai_eval_motion/actual_acceleration_mps2",
    "optimizer_delta_mps2": "lanistai_eval_motion/optimizer_delta_mps2",
    "optimizer_correction_mps2": "lanistai_eval_motion/optimizer_correction_mps2",
    "optimizer_cost": "lanistai_eval_motion/optimizer_cost",
    "distance_to_stopline_m": "lanistai_eval_safety/distance_to_stopline_m",
    "intersection_number": "lanistai_eval_safety/intersection_number",
    "traffic_light_state_code": "lanistai_eval_safety/traffic_light_state_code",
    "red_light_overdrive": "lanistai_eval_safety/red_light_overdrive",
    "ego_lead_vehicle_present": "lanistai_eval_safety/lead_vehicle_present",
    "ego_lead_distance_m": "lanistai_eval_safety/lead_vehicle_distance_m",
    "ego_lead_speed_mps": "lanistai_eval_safety/lead_vehicle_speed_mps",
    "supervisor_vehicle_conflict": "lanistai_eval_safety/vehicle_conflict",
    "supervisor_pedestrian_conflict": "lanistai_eval_safety/pedestrian_conflict",
    "active_pedestrian_count": "lanistai_eval_safety/active_pedestrian_count",
    "motion_state_stopped": "lanistai_eval_safety/stopped",
    "use_case_optimizer_intervention": "lanistai_eval_safety/optimizer_intervention",
    "use_case_sumo_intervention": "lanistai_eval_safety/sumo_intervention",
    "reward_total": "lanistai_eval_reward/total",
    "reward_speed_effective": "lanistai_eval_reward/speed_effective",
    "reward_speed_red_light_speed_reward": "lanistai_eval_reward/red_light_speed",
    "reward_sync_weighted": "lanistai_eval_reward/sync_weighted",
    "reward_accel_weighted": "lanistai_eval_reward/accel_weighted",
    "reward_delta_weighted": "lanistai_eval_reward/delta_weighted",
}


def evaluation_tensorboard_tag(field_name: str) -> str | None:
    """Return the curated evaluation tag for one lossless trace field."""

    return EVALUATION_FIELD_TAGS.get(str(field_name))


__all__ = [
    "COMPACT_SCALAR_METRIC_KEYS",
    "COMPARABLE_CUMULATIVE_KPI_TAGS",
    "COMPARABLE_KPI_TAGS",
    "DIAGNOSTIC_HISTOGRAM_METRIC_KEYS",
    "EVALUATION_FIELD_TAGS",
    "EVALUATION_TENSORBOARD_ROOTS",
    "NUMBERED_DASHBOARD_KPI_TAGS",
    "TENSORBOARD_HISTOGRAM_CADENCE",
    "TENSORBOARD_HISTOGRAM_SAMPLE_LIMIT",
    "TENSORBOARD_SCHEMA_VERSION",
    "TRAINING_TENSORBOARD_ROOTS",
    "TensorboardProfile",
    "comparable_kpi_tag",
    "evaluation_tensorboard_tag",
    "resolve_tensorboard_profile",
    "training_histogram_tag",
    "training_tensorboard_path",
    "training_tensorboard_tag",
]
